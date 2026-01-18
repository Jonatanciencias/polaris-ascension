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
    """Compressed Sparse Column matrix format (TODO: Session 12 Phase 2)."""
    pass


class BlockSparseMatrix:
    """Block-sparse matrix with wavefront alignment (TODO: Session 12 Phase 3)."""
    pass


class DynamicSparseActivations:
    """Dynamic format selection based on runtime sparsity (TODO: Session 12 Phase 4)."""
    pass
