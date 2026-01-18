"""
Sparse Matrix Formats for Legacy GPU AI Platform
================================================

This module implements efficient sparse matrix storage formats optimized
for AMD Radeon RX 580 (GCN 4.0, Polaris architecture).

Formats Implemented:
-------------------
1. CSR (Compressed Sparse Row) - Efficient row-major operations
2. CSC (Compressed Sparse Column) - Efficient column-major operations  
3. Block-Sparse - Wavefront-aligned blocks (64 elements for RX 580)
4. Dynamic Selection - Automatic format selection based on sparsity

Target Hardware:
---------------
- AMD Radeon RX 580
- Wavefront size: 64
- Memory bandwidth: 256 GB/s
- L2 Cache: 2 MB

Performance Goals:
-----------------
- Memory reduction: 10-100x for sparsity > 90%
- SpMM speedup: 2-5x vs dense (CSR/CSC)
- Block-sparse: 3-8x speedup (wavefront-aligned)

References:
----------
- Gray et al. (2017) "GPU Kernels for Block-Sparse Weights"
- NVIDIA (2020) "Accelerating Sparse Deep Neural Networks"
- Buluc et al. (2009) "Parallel Sparse Matrix-Matrix Multiplication"

Example Usage:
-------------
    import numpy as np
    from src.compute.sparse_formats import CSRMatrix
    
    # Convert dense to CSR
    dense = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    csr = CSRMatrix.from_dense(dense)
    
    # Sparse matrix multiplication
    result = csr.sparse_matmul(input_vector)
    
    # Check compression
    print(f"Memory reduction: {csr.compression_ratio()}x")

Version: 0.6.0-dev
Author: Legacy GPU AI Platform Contributors
License: MIT
"""

from typing import Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
import numpy as np
import warnings


@dataclass
class SparseMatrixStats:
    """Statistics about a sparse matrix."""
    nnz: int  # Number of non-zero elements
    sparsity: float  # Fraction of zero elements (0-1)
    density: float  # Fraction of non-zero elements (0-1)
    shape: Tuple[int, int]  # Matrix dimensions
    memory_dense: int  # Memory in bytes if dense
    memory_sparse: int  # Memory in bytes in sparse format
    compression_ratio: float  # memory_dense / memory_sparse


class CSRMatrix:
    """
    Compressed Sparse Row (CSR) matrix format.
    
    CSR is optimal for:
    - Row-major operations (matrix-vector products)
    - Sequential row access
    - Sparsity > 70-80%
    
    Storage Format:
    --------------
    - values: array of non-zero values (length nnz)
    - col_indices: column indices for each value (length nnz)
    - row_ptr: pointers to start of each row (length nrows+1)
    
    Example:
    -------
    Dense matrix:
        [[1, 0, 0],
         [0, 2, 0],
         [0, 0, 3]]
    
    CSR representation:
        values = [1, 2, 3]
        col_indices = [0, 1, 2]
        row_ptr = [0, 1, 2, 3]
    
    Memory Complexity:
    -----------------
    - Dense: O(nrows * ncols)
    - CSR: O(nnz + nrows)
    
    Time Complexity:
    ---------------
    - Construction: O(nrows * ncols)
    - SpMV (Sparse Matrix-Vector): O(nnz)
    - SpMM (Sparse Matrix-Matrix): O(nnz * ncols_B)
    
    References:
    ----------
    - Buluc et al. (2009) "Parallel Sparse Matrix-Matrix Multiplication"
    - Intel MKL Sparse BLAS documentation
    """
    
    def __init__(
        self,
        values: np.ndarray,
        col_indices: np.ndarray,
        row_ptr: np.ndarray,
        shape: Tuple[int, int]
    ):
        """
        Initialize CSR matrix from components.
        
        Args:
            values: Non-zero values (length nnz)
            col_indices: Column indices (length nnz)
            row_ptr: Row pointers (length nrows+1)
            shape: Matrix dimensions (nrows, ncols)
        
        Raises:
            ValueError: If dimensions are inconsistent
        """
        self.values = np.asarray(values, dtype=np.float32)
        self.col_indices = np.asarray(col_indices, dtype=np.int32)
        self.row_ptr = np.asarray(row_ptr, dtype=np.int32)
        self.shape = shape
        
        # Validate dimensions
        nrows, ncols = shape
        if len(row_ptr) != nrows + 1:
            raise ValueError(
                f"row_ptr length must be nrows+1 ({nrows+1}), got {len(row_ptr)}"
            )
        
        nnz = len(values)
        if len(col_indices) != nnz:
            raise ValueError(
                f"col_indices length must match values ({nnz}), got {len(col_indices)}"
            )
        
        if row_ptr[-1] != nnz:
            raise ValueError(
                f"row_ptr[-1] must equal nnz ({nnz}), got {row_ptr[-1]}"
            )
    
    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.values)
    
    @property
    def nrows(self) -> int:
        """Number of rows."""
        return self.shape[0]
    
    @property
    def ncols(self) -> int:
        """Number of columns."""
        return self.shape[1]
    
    @classmethod
    def from_dense(
        cls,
        dense: np.ndarray,
        threshold: float = 1e-10
    ) -> 'CSRMatrix':
        """
        Convert dense matrix to CSR format.
        
        Args:
            dense: Dense matrix (2D numpy array)
            threshold: Values below this are considered zero
        
        Returns:
            CSRMatrix instance
        
        Example:
            >>> dense = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
            >>> csr = CSRMatrix.from_dense(dense)
            >>> csr.nnz
            3
        """
        if dense.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {dense.shape}")
        
        nrows, ncols = dense.shape
        
        # Find non-zero elements
        mask = np.abs(dense) > threshold
        row_indices, col_indices_full = np.where(mask)
        values = dense[mask]
        
        # Build row_ptr
        row_ptr = np.zeros(nrows + 1, dtype=np.int32)
        for row in row_indices:
            row_ptr[row + 1] += 1
        row_ptr = np.cumsum(row_ptr)
        
        return cls(
            values=values.astype(np.float32),
            col_indices=col_indices_full.astype(np.int32),
            row_ptr=row_ptr,
            shape=(nrows, ncols)
        )
    
    def to_dense(self) -> np.ndarray:
        """
        Convert CSR matrix back to dense format.
        
        Returns:
            Dense numpy array
        
        Example:
            >>> csr = CSRMatrix.from_dense(np.eye(3))
            >>> dense = csr.to_dense()
            >>> np.allclose(dense, np.eye(3))
            True
        """
        dense = np.zeros(self.shape, dtype=np.float32)
        
        for i in range(self.nrows):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            
            for j in range(start, end):
                col = self.col_indices[j]
                dense[i, col] = self.values[j]
        
        return dense
    
    def sparse_matmul(self, other: np.ndarray) -> np.ndarray:
        """
        Sparse matrix multiplication (CSR @ dense).
        
        Supports:
        - Matrix-Vector: (m, n) @ (n,) -> (m,)
        - Matrix-Matrix: (m, n) @ (n, k) -> (m, k)
        
        Args:
            other: Dense matrix or vector
        
        Returns:
            Result of multiplication
        
        Raises:
            ValueError: If dimensions incompatible
        
        Algorithm:
        ---------
        For each row i:
            result[i] = sum(values[j] * other[col_indices[j]])
            for j in [row_ptr[i], row_ptr[i+1])
        
        Time Complexity: O(nnz * k) where k is ncols of other
        """
        other = np.asarray(other)
        
        # Handle vector input
        is_vector = (other.ndim == 1)
        if is_vector:
            other = other.reshape(-1, 1)
        
        # Validate dimensions
        if other.shape[0] != self.ncols:
            raise ValueError(
                f"Incompatible dimensions: ({self.nrows}, {self.ncols}) @ {other.shape}"
            )
        
        result = np.zeros((self.nrows, other.shape[1]), dtype=np.float32)
        
        # CSR matrix-matrix multiplication
        for i in range(self.nrows):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            
            for j in range(start, end):
                col = self.col_indices[j]
                value = self.values[j]
                result[i] += value * other[col]
        
        # Return vector if input was vector
        if is_vector:
            result = result.flatten()
        
        return result
    
    def memory_footprint(self) -> Dict[str, int]:
        """
        Calculate memory footprint in bytes.
        
        Returns:
            Dictionary with memory breakdown
        
        Example:
            >>> csr = CSRMatrix.from_dense(np.eye(1000))
            >>> mem = csr.memory_footprint()
            >>> print(f"CSR: {mem['total_sparse']/1024:.1f} KB")
            >>> print(f"Dense would be: {mem['total_dense']/1024:.1f} KB")
        """
        # CSR memory
        mem_values = self.values.nbytes
        mem_col_indices = self.col_indices.nbytes
        mem_row_ptr = self.row_ptr.nbytes
        mem_sparse = mem_values + mem_col_indices + mem_row_ptr
        
        # Dense memory (for comparison)
        mem_dense = self.nrows * self.ncols * 4  # float32
        
        return {
            'values': mem_values,
            'col_indices': mem_col_indices,
            'row_ptr': mem_row_ptr,
            'total_sparse': mem_sparse,
            'total_dense': mem_dense,
            'compression_ratio': mem_dense / mem_sparse if mem_sparse > 0 else 1.0
        }
    
    def get_statistics(self) -> SparseMatrixStats:
        """
        Get comprehensive statistics about this sparse matrix.
        
        Returns:
            SparseMatrixStats with detailed information
        """
        mem = self.memory_footprint()
        total_elements = self.nrows * self.ncols
        
        return SparseMatrixStats(
            nnz=self.nnz,
            sparsity=1.0 - (self.nnz / total_elements) if total_elements > 0 else 0.0,
            density=self.nnz / total_elements if total_elements > 0 else 0.0,
            shape=self.shape,
            memory_dense=mem['total_dense'],
            memory_sparse=mem['total_sparse'],
            compression_ratio=mem['compression_ratio']
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CSRMatrix(shape={self.shape}, nnz={self.nnz}, "
            f"sparsity={1.0 - self.nnz/(self.nrows*self.ncols):.2%})"
        )


# Placeholder classes for future implementation
class CSCMatrix:
    """
    Compressed Sparse Column (CSC) matrix format.
    
    CSC is optimal for:
    - Column-major operations (transpose-vector products)
    - Sequential column access
    - Algorithms that process columns (gradient descent, feature selection)
    - Complementary to CSR for different access patterns
    
    Storage Format:
    --------------
    - values: array of non-zero values (length nnz)
    - row_indices: row indices for each value (length nnz)
    - col_ptr: pointers to start of each column (length ncols+1)
    
    Example:
    -------
    Dense matrix:
        [[1, 0, 0],
         [0, 2, 0],
         [0, 0, 3]]
    
    CSC representation:
        values = [1, 2, 3]
        row_indices = [0, 1, 2]
        col_ptr = [0, 1, 2, 3]
    
    Memory Complexity:
    -----------------
    - Dense: O(nrows * ncols)
    - CSC: O(nnz + ncols)
    
    Time Complexity:
    ---------------
    - Construction: O(nrows * ncols)
    - Column access: O(nnz_col)
    - Matrix-Vector (A.T @ v): O(nnz)
    
    Comparison with CSR:
    -------------------
    - CSR: Better for row-wise operations (A @ x)
    - CSC: Better for column-wise operations (A.T @ x, feature extraction)
    - Memory: Same asymptotic complexity
    - Choose based on primary access pattern
    
    References:
    ----------
    - Davis (2006) "Direct Methods for Sparse Linear Systems"
    - Gilbert et al. (1992) "Sparse Matrices in MATLAB"
    """
    
    def __init__(
        self,
        values: np.ndarray,
        row_indices: np.ndarray,
        col_ptr: np.ndarray,
        shape: Tuple[int, int]
    ):
        """
        Initialize CSC matrix from components.
        
        Args:
            values: Non-zero values (length nnz)
            row_indices: Row indices (length nnz)
            col_ptr: Column pointers (length ncols+1)
            shape: Matrix dimensions (nrows, ncols)
        
        Raises:
            ValueError: If dimensions are inconsistent
        """
        self.values = np.asarray(values, dtype=np.float32)
        self.row_indices = np.asarray(row_indices, dtype=np.int32)
        self.col_ptr = np.asarray(col_ptr, dtype=np.int32)
        self.shape = shape
        
        # Validate dimensions
        nrows, ncols = shape
        if len(col_ptr) != ncols + 1:
            raise ValueError(
                f"col_ptr length must be ncols+1 ({ncols+1}), got {len(col_ptr)}"
            )
        
        nnz = len(values)
        if len(row_indices) != nnz:
            raise ValueError(
                f"row_indices length must match values ({nnz}), got {len(row_indices)}"
            )
        
        if col_ptr[-1] != nnz:
            raise ValueError(
                f"col_ptr[-1] must equal nnz ({nnz}), got {col_ptr[-1]}"
            )
    
    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.values)
    
    @property
    def nrows(self) -> int:
        """Number of rows."""
        return self.shape[0]
    
    @property
    def ncols(self) -> int:
        """Number of columns."""
        return self.shape[1]
    
    @classmethod
    def from_dense(
        cls,
        dense: np.ndarray,
        threshold: float = 1e-10
    ) -> 'CSCMatrix':
        """
        Convert dense matrix to CSC format.
        
        Args:
            dense: Dense matrix (2D numpy array)
            threshold: Values below this are considered zero
        
        Returns:
            CSCMatrix instance
        
        Algorithm:
        ---------
        1. Process column by column
        2. Extract non-zero values and row indices per column
        3. Build col_ptr incrementally
        
        Example:
            >>> dense = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
            >>> csc = CSCMatrix.from_dense(dense)
            >>> csc.nnz
            3
        """
        if dense.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {dense.shape}")
        
        nrows, ncols = dense.shape
        
        # Find non-zero elements (column-major order)
        mask = np.abs(dense) > threshold
        row_indices_full, col_indices = np.where(mask)
        values = dense[mask]
        
        # Sort by column (then by row for canonical form)
        sort_indices = np.lexsort((row_indices_full, col_indices))
        col_indices = col_indices[sort_indices]
        row_indices_full = row_indices_full[sort_indices]
        values = values[sort_indices]
        
        # Build col_ptr
        col_ptr = np.zeros(ncols + 1, dtype=np.int32)
        for col in col_indices:
            col_ptr[col + 1] += 1
        col_ptr = np.cumsum(col_ptr)
        
        return cls(
            values=values.astype(np.float32),
            row_indices=row_indices_full.astype(np.int32),
            col_ptr=col_ptr,
            shape=(nrows, ncols)
        )
    
    @classmethod
    def from_csr(cls, csr: CSRMatrix) -> 'CSCMatrix':
        """
        Convert CSR matrix to CSC format.
        
        This is useful when you need both row and column access patterns.
        
        Args:
            csr: CSRMatrix instance
        
        Returns:
            CSCMatrix instance
        
        Note:
            This conversion is O(nnz) and involves re-sorting the data.
        """
        # Convert through dense (simple but not most efficient)
        # For production, would implement direct CSR→CSC conversion
        dense = csr.to_dense()
        return cls.from_dense(dense)
    
    def to_dense(self) -> np.ndarray:
        """
        Convert CSC matrix back to dense format.
        
        Returns:
            Dense numpy array
        
        Example:
            >>> csc = CSCMatrix.from_dense(np.eye(3))
            >>> dense = csc.to_dense()
            >>> np.allclose(dense, np.eye(3))
            True
        """
        dense = np.zeros(self.shape, dtype=np.float32)
        
        for j in range(self.ncols):
            start = self.col_ptr[j]
            end = self.col_ptr[j + 1]
            
            for k in range(start, end):
                row = self.row_indices[k]
                dense[row, j] = self.values[k]
        
        return dense
    
    def sparse_matmul(self, other: np.ndarray) -> np.ndarray:
        """
        Sparse matrix multiplication (CSC @ dense).
        
        For CSC format, this is less efficient than CSR for A @ x,
        but more efficient for A.T @ x (transpose operations).
        
        Supports:
        - Matrix-Vector: (m, n) @ (n,) -> (m,)
        - Matrix-Matrix: (m, n) @ (n, k) -> (m, k)
        
        Args:
            other: Dense matrix or vector
        
        Returns:
            Result of multiplication
        
        Algorithm:
        ---------
        For each column j of A:
            For each non-zero A[i,j]:
                result[i] += A[i,j] * other[j]
        
        Time Complexity: O(nnz * k) where k is ncols of other
        """
        other = np.asarray(other)
        
        # Handle vector input
        is_vector = (other.ndim == 1)
        if is_vector:
            other = other.reshape(-1, 1)
        
        # Validate dimensions
        if other.shape[0] != self.ncols:
            raise ValueError(
                f"Incompatible dimensions: ({self.nrows}, {self.ncols}) @ {other.shape}"
            )
        
        result = np.zeros((self.nrows, other.shape[1]), dtype=np.float32)
        
        # CSC matrix-matrix multiplication (column-wise)
        for j in range(self.ncols):
            start = self.col_ptr[j]
            end = self.col_ptr[j + 1]
            
            for k in range(start, end):
                row = self.row_indices[k]
                value = self.values[k]
                result[row] += value * other[j]
        
        # Return vector if input was vector
        if is_vector:
            result = result.flatten()
        
        return result
    
    def transpose_matmul(self, other: np.ndarray) -> np.ndarray:
        """
        Efficient transpose matrix multiplication (A.T @ other).
        
        This is where CSC shines - column access is efficient.
        
        Args:
            other: Dense matrix or vector
        
        Returns:
            Result of A.T @ other
        
        Time Complexity: O(nnz * k)
        """
        other = np.asarray(other)
        
        # Handle vector input
        is_vector = (other.ndim == 1)
        if is_vector:
            other = other.reshape(-1, 1)
        
        # Validate dimensions (A.T is ncols × nrows)
        if other.shape[0] != self.nrows:
            raise ValueError(
                f"Incompatible dimensions for A.T @ x: "
                f"({self.ncols}, {self.nrows}) @ {other.shape}"
            )
        
        result = np.zeros((self.ncols, other.shape[1]), dtype=np.float32)
        
        # CSC transpose multiplication (very efficient)
        for j in range(self.ncols):
            start = self.col_ptr[j]
            end = self.col_ptr[j + 1]
            
            for k in range(start, end):
                row = self.row_indices[k]
                value = self.values[k]
                result[j] += value * other[row]
        
        # Return vector if input was vector
        if is_vector:
            result = result.flatten()
        
        return result
    
    def memory_footprint(self) -> Dict[str, int]:
        """Calculate memory footprint in bytes."""
        mem_values = self.values.nbytes
        mem_row_indices = self.row_indices.nbytes
        mem_col_ptr = self.col_ptr.nbytes
        mem_sparse = mem_values + mem_row_indices + mem_col_ptr
        
        mem_dense = self.nrows * self.ncols * 4  # float32
        
        return {
            'values': mem_values,
            'row_indices': mem_row_indices,
            'col_ptr': mem_col_ptr,
            'total_sparse': mem_sparse,
            'total_dense': mem_dense,
            'compression_ratio': mem_dense / mem_sparse if mem_sparse > 0 else 1.0
        }
    
    def get_statistics(self) -> SparseMatrixStats:
        """Get comprehensive statistics about this sparse matrix."""
        mem = self.memory_footprint()
        total_elements = self.nrows * self.ncols
        
        return SparseMatrixStats(
            nnz=self.nnz,
            sparsity=1.0 - (self.nnz / total_elements) if total_elements > 0 else 0.0,
            density=self.nnz / total_elements if total_elements > 0 else 0.0,
            shape=self.shape,
            memory_dense=mem['total_dense'],
            memory_sparse=mem['total_sparse'],
            compression_ratio=mem['compression_ratio']
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CSCMatrix(shape={self.shape}, nnz={self.nnz}, "
            f"sparsity={1.0 - self.nnz/(self.nrows*self.ncols):.2%})"
        )


class BlockSparseMatrix:
    """
    Block-sparse matrix with wavefront alignment for AMD GCN GPUs.
    
    Block-sparse format is optimized for:
    - GPU execution (wavefront-aligned memory access)
    - Dense operations on sparse structure
    - Better cache utilization than unstructured sparsity
    - RX 580: Wavefront size = 64, optimal block size = 8×8 or 16×4
    
    Concept:
    -------
    Instead of storing individual non-zero elements, store dense blocks
    that contain mostly non-zero values. This allows:
    - Using fast dense GEMM kernels on blocks
    - Coalesced memory access (GPU friendly)
    - Better vectorization (SIMD friendly)
    - Trade-off: Some zeros stored, but faster overall
    
    Storage Format:
    --------------
    - blocks: List of dense sub-matrices (shape: block_size × block_size)
    - block_indices: (row_block, col_block) indices for each block
    - block_size: Size of each square block (typically 4, 8, 16, 32)
    
    Example (block_size=2):
    ----------------------
    Dense matrix (4×4):
        [[1, 2, 0, 0],
         [3, 4, 0, 0],
         [0, 0, 5, 6],
         [0, 0, 7, 8]]
    
    Block representation (2 blocks of 2×2):
        blocks = [[[1,2], [3,4]], [[5,6], [7,8]]]
        block_indices = [(0,0), (1,1)]
    
    Advantages:
    ----------
    - GPU-friendly: Aligned to wavefront size (64 threads)
    - Cache-friendly: Block operations have spatial locality
    - Vectorizable: SIMD operations on dense blocks
    - Flexible: Adjust block_size for workload (4-32 typical)
    
    Disadvantages:
    -------------
    - May store some zeros (padding within blocks)
    - Requires block structure in sparsity pattern
    - Memory overhead if sparsity is very random
    
    Optimal Use Cases:
    -----------------
    - Structured pruning (filter/channel pruning)
    - Attention masks (block diagonal patterns)
    - Convolutional layers (local receptive fields)
    - Block diagonal matrices
    
    RX 580 Optimization:
    -------------------
    - Wavefront size: 64 threads
    - Recommended block sizes: 8×8 (64 elements) or 16×4 (64 elements)
    - Align blocks to 256-byte boundaries (cache line)
    - Use multiple blocks per wavefront for efficiency
    
    References:
    ----------
    - Gray et al. (2017) "GPU Kernels for Block-Sparse Weights"
    - NVIDIA (2020) "Accelerating Sparse Deep Neural Networks"
    - Gale et al. (2020) "Sparse GPU Kernels for DL"
    """
    
    def __init__(
        self,
        blocks: list,
        block_indices: np.ndarray,
        block_size: int,
        shape: Tuple[int, int]
    ):
        """
        Initialize block-sparse matrix.
        
        Args:
            blocks: List of dense blocks (each block_size × block_size)
            block_indices: Array of (row_block, col_block) indices
            block_size: Size of each square block
            shape: Original matrix dimensions (nrows, ncols)
        
        Raises:
            ValueError: If dimensions inconsistent
        """
        self.blocks = [np.asarray(b, dtype=np.float32) for b in blocks]
        self.block_indices = np.asarray(block_indices, dtype=np.int32)
        self.block_size = block_size
        self.shape = shape
        
        # Validate
        if len(self.blocks) != len(self.block_indices):
            raise ValueError(
                f"Number of blocks ({len(self.blocks)}) must match "
                f"number of indices ({len(self.block_indices)})"
            )
        
        for i, block in enumerate(self.blocks):
            if block.shape != (block_size, block_size):
                raise ValueError(
                    f"Block {i} has shape {block.shape}, expected ({block_size}, {block_size})"
                )
    
    @property
    def num_blocks(self) -> int:
        """Number of non-zero blocks."""
        return len(self.blocks)
    
    @property
    def nrows(self) -> int:
        """Number of rows."""
        return self.shape[0]
    
    @property
    def ncols(self) -> int:
        """Number of columns."""
        return self.shape[1]
    
    @classmethod
    def from_dense(
        cls,
        dense: np.ndarray,
        block_size: int = 8,
        threshold: float = 0.3
    ) -> 'BlockSparseMatrix':
        """
        Convert dense matrix to block-sparse format.
        
        Args:
            dense: Dense matrix (2D numpy array)
            block_size: Size of square blocks (default: 8 for RX 580)
            threshold: Minimum density within block to keep (default: 0.3)
                      If block has < 30% non-zeros, skip it
        
        Returns:
            BlockSparseMatrix instance
        
        Algorithm:
        ---------
        1. Pad matrix to multiple of block_size
        2. Divide into blocks
        3. Calculate density of each block
        4. Keep blocks with density > threshold
        
        Note:
            block_size=8 is optimal for RX 580 (8×8=64 = wavefront size)
        """
        if dense.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {dense.shape}")
        
        nrows, ncols = dense.shape
        
        # Pad to multiple of block_size
        pad_rows = (block_size - nrows % block_size) % block_size
        pad_cols = (block_size - ncols % block_size) % block_size
        
        if pad_rows > 0 or pad_cols > 0:
            dense_padded = np.pad(
                dense,
                ((0, pad_rows), (0, pad_cols)),
                mode='constant',
                constant_values=0
            )
        else:
            dense_padded = dense
        
        nrows_padded, ncols_padded = dense_padded.shape
        
        # Extract blocks
        blocks = []
        block_indices = []
        
        for i in range(0, nrows_padded, block_size):
            for j in range(0, ncols_padded, block_size):
                block = dense_padded[i:i+block_size, j:j+block_size]
                
                # Calculate block density
                nnz_block = np.count_nonzero(block)
                density = nnz_block / (block_size * block_size)
                
                # Keep block if density exceeds threshold
                if density >= threshold:
                    blocks.append(block.copy())
                    block_indices.append((i // block_size, j // block_size))
        
        return cls(
            blocks=blocks,
            block_indices=np.array(block_indices, dtype=np.int32),
            block_size=block_size,
            shape=(nrows, ncols)
        )
    
    def to_dense(self) -> np.ndarray:
        """
        Convert block-sparse matrix back to dense format.
        
        Returns:
            Dense numpy array
        """
        # Create padded dense matrix
        nrows_padded = ((self.nrows + self.block_size - 1) // self.block_size) * self.block_size
        ncols_padded = ((self.ncols + self.block_size - 1) // self.block_size) * self.block_size
        
        dense_padded = np.zeros((nrows_padded, ncols_padded), dtype=np.float32)
        
        # Place blocks
        for block, (row_idx, col_idx) in zip(self.blocks, self.block_indices):
            row_start = row_idx * self.block_size
            col_start = col_idx * self.block_size
            dense_padded[row_start:row_start+self.block_size,
                        col_start:col_start+self.block_size] = block
        
        # Unpad to original shape
        return dense_padded[:self.nrows, :self.ncols]
    
    def sparse_matmul(self, other: np.ndarray) -> np.ndarray:
        """
        Block-sparse matrix multiplication.
        
        This uses dense GEMM on each block, which is GPU-friendly.
        
        Args:
            other: Dense matrix or vector
        
        Returns:
            Result of multiplication
        
        Algorithm:
        ---------
        For each non-zero block B at (i, j):
            result[i*bs:(i+1)*bs] += B @ other[j*bs:(j+1)*bs]
        
        where bs = block_size
        """
        other = np.asarray(other)
        
        # Handle vector input
        is_vector = (other.ndim == 1)
        if is_vector:
            other = other.reshape(-1, 1)
        
        # Validate dimensions
        if other.shape[0] != self.ncols:
            raise ValueError(
                f"Incompatible dimensions: ({self.nrows}, {self.ncols}) @ {other.shape}"
            )
        
        # Pad other if needed
        ncols_padded = ((self.ncols + self.block_size - 1) // self.block_size) * self.block_size
        if other.shape[0] < ncols_padded:
            other_padded = np.pad(
                other,
                ((0, ncols_padded - other.shape[0]), (0, 0)),
                mode='constant'
            )
        else:
            other_padded = other
        
        # Initialize result
        nrows_padded = ((self.nrows + self.block_size - 1) // self.block_size) * self.block_size
        result_padded = np.zeros((nrows_padded, other.shape[1]), dtype=np.float32)
        
        # Block-wise multiplication
        for block, (row_idx, col_idx) in zip(self.blocks, self.block_indices):
            row_start = row_idx * self.block_size
            col_start = col_idx * self.block_size
            
            # Extract corresponding slice from other
            other_slice = other_padded[col_start:col_start+self.block_size]
            
            # Dense matmul on this block (GPU-friendly operation)
            result_padded[row_start:row_start+self.block_size] += block @ other_slice
        
        # Unpad result
        result = result_padded[:self.nrows]
        
        # Return vector if input was vector
        if is_vector:
            result = result.flatten()
        
        return result
    
    def memory_footprint(self) -> Dict[str, int]:
        """Calculate memory footprint in bytes."""
        # Block-sparse memory
        mem_blocks = sum(b.nbytes for b in self.blocks)
        mem_indices = self.block_indices.nbytes
        mem_sparse = mem_blocks + mem_indices
        
        # Dense memory (for comparison)
        mem_dense = self.nrows * self.ncols * 4  # float32
        
        # Actual non-zeros stored (including padding within blocks)
        nnz_stored = self.num_blocks * self.block_size * self.block_size
        
        return {
            'blocks': mem_blocks,
            'indices': mem_indices,
            'total_sparse': mem_sparse,
            'total_dense': mem_dense,
            'compression_ratio': mem_dense / mem_sparse if mem_sparse > 0 else 1.0,
            'elements_stored': nnz_stored,
            'overhead_factor': nnz_stored / self.nrows / self.ncols if self.nrows * self.ncols > 0 else 0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about this block-sparse matrix."""
        mem = self.memory_footprint()
        
        # Calculate actual non-zeros in blocks
        actual_nnz = sum(np.count_nonzero(block) for block in self.blocks)
        total_elements = self.nrows * self.ncols
        
        return {
            'num_blocks': self.num_blocks,
            'block_size': self.block_size,
            'shape': self.shape,
            'actual_nnz': actual_nnz,
            'stored_elements': mem['elements_stored'],
            'sparsity': 1.0 - (actual_nnz / total_elements) if total_elements > 0 else 0.0,
            'memory_sparse': mem['total_sparse'],
            'memory_dense': mem['total_dense'],
            'compression_ratio': mem['compression_ratio'],
            'storage_efficiency': actual_nnz / mem['elements_stored'] if mem['elements_stored'] > 0 else 0,
            'wavefront_aligned': self.block_size * self.block_size == 64  # RX 580
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BlockSparseMatrix(shape={self.shape}, num_blocks={self.num_blocks}, "
            f"block_size={self.block_size}×{self.block_size})"
        )


class DynamicFormatSelector:
    """
    Automatic sparse format selection based on matrix characteristics.
    
    This class analyzes matrix properties (sparsity, size, access patterns) and
    automatically selects the most efficient sparse format for the RX 580.
    
    Selection Strategy:
    ------------------
    1. Analyze sparsity level
    2. Check for structured patterns (blocks)
    3. Consider matrix size
    4. Select optimal format
    
    Format Rules:
    ------------
    - Dense:        sparsity < 50%
    - Block-sparse: 50% ≤ sparsity < 75% AND structured
    - CSR:          sparsity ≥ 75% (row-major operations)
    - CSC:          sparsity ≥ 75% (column-major operations)
    
    Example:
    -------
    >>> selector = DynamicFormatSelector()
    >>> sparse_matrix = selector.select_format(dense_matrix, preferred_op='row')
    >>> result = sparse_matrix.sparse_matmul(x)
    
    RX 580 Optimization:
    -------------------
    - Considers wavefront size (64) for block selection
    - Balances compression vs compute efficiency
    - Adaptive thresholds based on matrix size
    
    References:
    ----------
    - Gale et al. (2020) "Sparse GPU Kernels for Deep Learning"
    - NVIDIA (2020) "Accelerating Sparse DNN"
    """
    
    def __init__(
        self,
        sparsity_threshold_dense: float = 0.50,
        sparsity_threshold_block: float = 0.75,
        block_size: int = 8,
        block_density_threshold: float = 0.20
    ):
        """
        Initialize format selector with thresholds.
        
        Args:
            sparsity_threshold_dense: Below this, use dense (default: 0.50)
            sparsity_threshold_block: Above this, use CSR/CSC (default: 0.75)
            block_size: Block size for block-sparse (default: 8 for RX 580)
            block_density_threshold: Min density for blocks (default: 0.20)
        """
        self.sparsity_threshold_dense = sparsity_threshold_dense
        self.sparsity_threshold_block = sparsity_threshold_block
        self.block_size = block_size
        self.block_density_threshold = block_density_threshold
    
    def analyze_sparsity(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Analyze matrix sparsity and structure.
        
        Args:
            matrix: Dense numpy array
        
        Returns:
            Dictionary with analysis results
        """
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {matrix.shape}")
        
        nrows, ncols = matrix.shape
        total_elements = nrows * ncols
        nnz = np.count_nonzero(matrix)
        sparsity = 1.0 - (nnz / total_elements) if total_elements > 0 else 0.0
        
        # Check for block structure
        has_blocks = self._detect_block_structure(matrix)
        
        # Estimate compute cost
        dense_cost = nrows * ncols
        sparse_cost = nnz * 2  # Approximate sparse ops cost
        
        return {
            'shape': (nrows, ncols),
            'nnz': nnz,
            'sparsity': sparsity,
            'density': 1.0 - sparsity,
            'has_block_structure': has_blocks,
            'dense_cost': dense_cost,
            'sparse_cost': sparse_cost,
            'size_category': self._categorize_size(nrows, ncols)
        }
    
    def _detect_block_structure(self, matrix: np.ndarray) -> bool:
        """
        Detect if matrix has block-diagonal or block structure.
        
        Simple heuristic: check if non-zeros cluster in blocks.
        """
        nrows, ncols = matrix.shape
        block_size = self.block_size
        
        if nrows < block_size or ncols < block_size:
            return False
        
        # Sample a few blocks and check density variance
        num_row_blocks = nrows // block_size
        num_col_blocks = ncols // block_size
        
        if num_row_blocks == 0 or num_col_blocks == 0:
            return False
        
        # Sample up to 16 blocks
        sample_blocks = min(16, num_row_blocks * num_col_blocks)
        densities = []
        
        for _ in range(sample_blocks):
            i = np.random.randint(0, num_row_blocks)
            j = np.random.randint(0, num_col_blocks)
            
            block = matrix[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ]
            
            block_nnz = np.count_nonzero(block)
            block_density = block_nnz / (block_size * block_size)
            densities.append(block_density)
        
        # High variance in block densities suggests block structure
        # (some blocks dense, some sparse)
        if len(densities) > 1:
            variance = np.var(densities)
            return variance > 0.05  # Threshold for "structured"
        
        return False
    
    def _categorize_size(self, nrows: int, ncols: int) -> str:
        """Categorize matrix size for optimization decisions."""
        total = nrows * ncols
        
        if total < 10000:
            return 'small'
        elif total < 1000000:
            return 'medium'
        else:
            return 'large'
    
    def select_format(
        self,
        matrix: np.ndarray,
        preferred_op: str = 'row',
        force_sparse: bool = False
    ) -> Union[np.ndarray, CSRMatrix, CSCMatrix, BlockSparseMatrix]:
        """
        Automatically select best sparse format for given matrix.
        
        Args:
            matrix: Dense numpy array
            preferred_op: 'row' for A@x, 'col' for A.T@x (default: 'row')
            force_sparse: Force sparse format even if dense is better
        
        Returns:
            Matrix in optimal format (dense array or sparse matrix object)
        
        Selection Logic:
        ---------------
        1. If sparsity < 50% and not forced: return dense
        2. If 50% ≤ sparsity < 75% and has block structure: Block-sparse
        3. If sparsity ≥ 75%:
           - Use CSR for row operations
           - Use CSC for column operations
        
        Example:
        -------
        >>> selector = DynamicFormatSelector()
        >>> # 90% sparse matrix
        >>> dense = np.random.randn(256, 512)
        >>> mask = np.random.rand(256, 512) > 0.9
        >>> dense = dense * mask
        >>> 
        >>> # Auto-select format
        >>> sparse = selector.select_format(dense, preferred_op='row')
        >>> type(sparse).__name__
        'CSRMatrix'
        """
        analysis = self.analyze_sparsity(matrix)
        sparsity = analysis['sparsity']
        
        # Decision tree
        if sparsity < self.sparsity_threshold_dense and not force_sparse:
            # Use dense
            return matrix.astype(np.float32)
        
        elif sparsity < self.sparsity_threshold_block:
            # Medium sparsity: check for block structure
            if analysis['has_block_structure']:
                return BlockSparseMatrix.from_dense(
                    matrix,
                    block_size=self.block_size,
                    threshold=self.block_density_threshold
                )
            else:
                # No block structure, use CSR/CSC
                if preferred_op == 'col':
                    return CSCMatrix.from_dense(matrix)
                else:
                    return CSRMatrix.from_dense(matrix)
        
        else:
            # High sparsity: use CSR or CSC
            if preferred_op == 'col':
                return CSCMatrix.from_dense(matrix)
            else:
                return CSRMatrix.from_dense(matrix)
    
    def recommend_format(
        self,
        matrix: np.ndarray,
        context: str = 'inference'
    ) -> Dict[str, Any]:
        """
        Provide detailed format recommendation with reasoning.
        
        Args:
            matrix: Dense numpy array
            context: 'inference', 'training', or 'generic'
        
        Returns:
            Dictionary with recommendation and explanation
        
        Example:
        -------
        >>> selector = DynamicFormatSelector()
        >>> recommendation = selector.recommend_format(matrix, context='training')
        >>> print(recommendation['format'])
        'CSC'
        >>> print(recommendation['reason'])
        'High sparsity (92%), training needs transpose ops'
        """
        analysis = self.analyze_sparsity(matrix)
        sparsity = analysis['sparsity']
        
        # Base recommendation
        if sparsity < self.sparsity_threshold_dense:
            format_name = 'Dense'
            reason = f"Low sparsity ({sparsity*100:.1f}%), overhead not justified"
        elif sparsity < self.sparsity_threshold_block:
            if analysis['has_block_structure']:
                format_name = 'Block-sparse'
                reason = f"Medium sparsity ({sparsity*100:.1f}%) with block structure"
            else:
                format_name = 'CSR'
                reason = f"Medium sparsity ({sparsity*100:.1f}%), unstructured"
        else:
            format_name = 'CSR/CSC'
            reason = f"High sparsity ({sparsity*100:.1f}%), maximum compression"
        
        # Context-specific adjustments
        if context == 'training' and sparsity >= self.sparsity_threshold_dense:
            format_name = 'CSC'
            reason += ", training needs transpose (backward pass)"
        elif context == 'inference' and sparsity >= self.sparsity_threshold_dense:
            format_name = 'CSR'
            reason += ", inference uses forward pass only"
        
        # Estimate savings
        if format_name != 'Dense':
            mem_dense = analysis['shape'][0] * analysis['shape'][1] * 4
            if 'Block' in format_name:
                # Estimate block-sparse memory
                mem_sparse = analysis['nnz'] * 6  # Approximate
            else:
                # CSR/CSC memory
                mem_sparse = analysis['nnz'] * 8 + analysis['shape'][0] * 4
            
            compression = mem_dense / mem_sparse if mem_sparse > 0 else 1.0
            savings_mb = (mem_dense - mem_sparse) / 1024 / 1024
        else:
            compression = 1.0
            savings_mb = 0.0
        
        return {
            'format': format_name,
            'reason': reason,
            'sparsity': sparsity,
            'compression_ratio': compression,
            'memory_savings_mb': savings_mb,
            'size_category': analysis['size_category'],
            'has_structure': analysis['has_block_structure']
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DynamicFormatSelector("
            f"dense<{self.sparsity_threshold_dense*100:.0f}%, "
            f"block<{self.sparsity_threshold_block*100:.0f}%, "
            f"block_size={self.block_size})"
        )


class DynamicSparseActivations:
    """
    DEPRECATED: Use DynamicFormatSelector instead.
    
    This class name is kept for backward compatibility but redirects
    to DynamicFormatSelector.
    """
    
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "DynamicSparseActivations is deprecated, use DynamicFormatSelector instead",
            DeprecationWarning,
            stacklevel=2
        )
        self._selector = DynamicFormatSelector(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self._selector, name)
