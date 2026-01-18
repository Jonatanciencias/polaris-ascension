"""
Tests for Sparse Matrix Formats
===============================

Test suite for CSR, CSC, Block-sparse, and Dynamic Selection formats.

Test Strategy:
-------------
1. Unit tests for each format (conversion, correctness, memory)
2. Integration tests (with pruners, real workloads)
3. Performance benchmarks (vs dense, vs scipy.sparse)
4. Edge cases (empty matrices, high sparsity, diagonal)

Coverage Goals:
--------------
- CSRMatrix: 15+ tests
- CSCMatrix: 10+ tests
- BlockSparseMatrix: 10+ tests
- Dynamic Selection: 5+ tests
- Integration: 5+ tests

Total: 45+ tests
"""

import pytest
import numpy as np
from src.compute.sparse_formats import (
    CSRMatrix,
    CSCMatrix,
    BlockSparseMatrix,
    SparseMatrixStats
)


class TestCSRMatrix:
    """Test suite for CSR (Compressed Sparse Row) format."""
    
    def test_initialization_valid(self):
        """Test CSR initialization with valid data."""
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        col_indices = np.array([0, 1, 2], dtype=np.int32)
        row_ptr = np.array([0, 1, 2, 3], dtype=np.int32)
        shape = (3, 3)
        
        csr = CSRMatrix(values, col_indices, row_ptr, shape)
        
        assert csr.shape == (3, 3)
        assert csr.nnz == 3
        assert csr.nrows == 3
        assert csr.ncols == 3
        np.testing.assert_array_equal(csr.values, values)
        np.testing.assert_array_equal(csr.col_indices, col_indices)
        np.testing.assert_array_equal(csr.row_ptr, row_ptr)
    
    def test_initialization_invalid_row_ptr_length(self):
        """Test CSR initialization fails with wrong row_ptr length."""
        values = np.array([1.0, 2.0])
        col_indices = np.array([0, 1])
        row_ptr = np.array([0, 2])  # Should be length nrows+1 = 4
        shape = (3, 3)
        
        with pytest.raises(ValueError, match="row_ptr length must be"):
            CSRMatrix(values, col_indices, row_ptr, shape)
    
    def test_initialization_invalid_col_indices_length(self):
        """Test CSR initialization fails with mismatched col_indices."""
        values = np.array([1.0, 2.0])
        col_indices = np.array([0])  # Should match values length
        row_ptr = np.array([0, 1, 2, 2])
        shape = (3, 3)
        
        with pytest.raises(ValueError, match="col_indices length must match"):
            CSRMatrix(values, col_indices, row_ptr, shape)
    
    def test_from_dense_identity_matrix(self):
        """Test conversion from dense identity matrix."""
        dense = np.eye(3, dtype=np.float32)
        csr = CSRMatrix.from_dense(dense)
        
        assert csr.shape == (3, 3)
        assert csr.nnz == 3
        np.testing.assert_array_equal(csr.values, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(csr.col_indices, [0, 1, 2])
        np.testing.assert_array_equal(csr.row_ptr, [0, 1, 2, 3])
    
    def test_from_dense_sparse_matrix(self):
        """Test conversion from dense sparse matrix."""
        dense = np.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ], dtype=np.float32)
        
        csr = CSRMatrix.from_dense(dense)
        
        assert csr.nnz == 3
        np.testing.assert_array_almost_equal(csr.values, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(csr.col_indices, [0, 1, 2])
        np.testing.assert_array_equal(csr.row_ptr, [0, 1, 2, 3])
    
    def test_from_dense_empty_matrix(self):
        """Test conversion from empty (all zeros) matrix."""
        dense = np.zeros((3, 3), dtype=np.float32)
        csr = CSRMatrix.from_dense(dense)
        
        assert csr.nnz == 0
        assert len(csr.values) == 0
        assert len(csr.col_indices) == 0
        np.testing.assert_array_equal(csr.row_ptr, [0, 0, 0, 0])
    
    def test_from_dense_with_threshold(self):
        """Test conversion with custom zero threshold."""
        dense = np.array([
            [1.0, 1e-12, 0],
            [0, 2.0, 1e-11],
            [1e-9, 0, 3.0]
        ], dtype=np.float32)
        
        # Default threshold should keep values > 1e-10
        csr = CSRMatrix.from_dense(dense, threshold=1e-10)
        
        assert csr.nnz == 4  # 1.0, 2.0, 1e-9, 3.0
        
        # Higher threshold should filter out 1e-9
        csr_strict = CSRMatrix.from_dense(dense, threshold=1e-8)
        assert csr_strict.nnz == 3  # Only 1.0, 2.0, 3.0
    
    def test_to_dense_identity(self):
        """Test conversion back to dense for identity matrix."""
        original = np.eye(3, dtype=np.float32)
        csr = CSRMatrix.from_dense(original)
        reconstructed = csr.to_dense()
        
        np.testing.assert_array_almost_equal(reconstructed, original)
    
    def test_to_dense_sparse_matrix(self):
        """Test conversion back to dense for general sparse matrix."""
        original = np.array([
            [1, 0, 2],
            [0, 0, 0],
            [3, 4, 0]
        ], dtype=np.float32)
        
        csr = CSRMatrix.from_dense(original)
        reconstructed = csr.to_dense()
        
        np.testing.assert_array_almost_equal(reconstructed, original)
    
    def test_sparse_matmul_vector(self):
        """Test sparse matrix-vector multiplication."""
        dense = np.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ], dtype=np.float32)
        
        csr = CSRMatrix.from_dense(dense)
        vector = np.array([1, 2, 3], dtype=np.float32)
        
        result = csr.sparse_matmul(vector)
        expected = dense @ vector
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_sparse_matmul_matrix(self):
        """Test sparse matrix-matrix multiplication."""
        A_dense = np.array([
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 5]
        ], dtype=np.float32)
        
        B = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ], dtype=np.float32)
        
        csr = CSRMatrix.from_dense(A_dense)
        result = csr.sparse_matmul(B)
        expected = A_dense @ B
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_sparse_matmul_dimension_mismatch(self):
        """Test sparse matmul fails with incompatible dimensions."""
        csr = CSRMatrix.from_dense(np.eye(3))
        vector = np.array([1, 2], dtype=np.float32)  # Wrong size
        
        with pytest.raises(ValueError, match="Incompatible dimensions"):
            csr.sparse_matmul(vector)
    
    def test_memory_footprint(self):
        """Test memory footprint calculation."""
        dense = np.eye(100, dtype=np.float32)
        csr = CSRMatrix.from_dense(dense)
        
        mem = csr.memory_footprint()
        
        assert mem['values'] == 100 * 4  # 100 floats
        assert mem['col_indices'] == 100 * 4  # 100 int32
        assert mem['row_ptr'] == 101 * 4  # 101 int32
        assert mem['total_dense'] == 100 * 100 * 4  # Dense would be 40KB
        assert mem['total_sparse'] < mem['total_dense']
        assert mem['compression_ratio'] > 1.0
    
    def test_get_statistics(self):
        """Test comprehensive statistics."""
        # 90% sparse matrix
        dense = np.eye(100, dtype=np.float32)
        csr = CSRMatrix.from_dense(dense)
        
        stats = csr.get_statistics()
        
        assert stats.nnz == 100
        assert stats.sparsity == pytest.approx(0.99, abs=0.01)  # 99% sparse
        assert stats.density == pytest.approx(0.01, abs=0.01)
        assert stats.shape == (100, 100)
        assert stats.compression_ratio > 1.0
    
    def test_repr(self):
        """Test string representation."""
        csr = CSRMatrix.from_dense(np.eye(3))
        repr_str = repr(csr)
        
        assert "CSRMatrix" in repr_str
        assert "shape=(3, 3)" in repr_str
        assert "nnz=3" in repr_str
        assert "sparsity" in repr_str
    
    def test_high_sparsity_matrix(self):
        """Test with 95% sparse matrix."""
        # Create 1000x1000 matrix with 5% density
        np.random.seed(42)
        dense = np.zeros((1000, 1000), dtype=np.float32)
        
        # Add 50,000 non-zero elements (5%)
        for _ in range(50000):
            i = np.random.randint(0, 1000)
            j = np.random.randint(0, 1000)
            dense[i, j] = np.random.randn()
        
        csr = CSRMatrix.from_dense(dense)
        
        # Verify conversion
        assert csr.nnz > 0
        assert csr.shape == (1000, 1000)
        
        # Verify correctness
        reconstructed = csr.to_dense()
        np.testing.assert_array_almost_equal(reconstructed, dense, decimal=5)
        
        # Verify memory savings
        stats = csr.get_statistics()
        assert stats.compression_ratio > 5.0  # Should have significant savings
    
    def test_rectangular_matrix(self):
        """Test with non-square matrix."""
        dense = np.array([
            [1, 0, 0, 2],
            [0, 3, 0, 0],
            [4, 0, 5, 0]
        ], dtype=np.float32)
        
        csr = CSRMatrix.from_dense(dense)
        
        assert csr.shape == (3, 4)
        assert csr.nnz == 5
        
        # Test matmul with vector
        vector = np.array([1, 1, 1, 1], dtype=np.float32)
        result = csr.sparse_matmul(vector)
        expected = dense @ vector
        
        np.testing.assert_array_almost_equal(result, expected)


# ==================== Phase 2: CSC and Block-Sparse Tests ====================

class TestCSCMatrix:
    """Tests for Compressed Sparse Column format (Phase 2)."""
    
    def test_basic_initialization(self):
        """Test basic CSC matrix creation."""
        # Create simple 3x3 identity matrix in CSC format
        values = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        row_indices = np.array([0, 1, 2], dtype=np.int32)
        col_ptr = np.array([0, 1, 2, 3], dtype=np.int32)
        
        csc = CSCMatrix(values, row_indices, col_ptr, shape=(3, 3))
        
        assert csc.shape == (3, 3)
        assert csc.nnz == 3
        assert csc.nrows == 3
        assert csc.ncols == 3
        assert np.array_equal(csc.values, values)
    
    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        values = np.array([1.0], dtype=np.float32)
        row_indices = np.array([0], dtype=np.int32)
        
        # Wrong col_ptr length
        with pytest.raises(ValueError, match="col_ptr length"):
            col_ptr = np.array([0, 1], dtype=np.int32)
            CSCMatrix(values, row_indices, col_ptr, shape=(3, 3))
        
        # Mismatched row_indices length
        with pytest.raises(ValueError, match="row_indices length"):
            col_ptr = np.array([0, 1, 1, 1], dtype=np.int32)
            row_indices = np.array([0, 1], dtype=np.int32)
            CSCMatrix(values, row_indices, col_ptr, shape=(3, 3))
    
    def test_from_dense_conversion(self):
        """Test conversion from dense to CSC."""
        # Create a simple sparse matrix
        dense = np.array([
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 5]
        ], dtype=np.float32)
        
        csc = CSCMatrix.from_dense(dense)
        
        # Check nnz
        assert csc.nnz == 5
        
        # Verify reconstruction
        reconstructed = csc.to_dense()
        assert np.allclose(reconstructed, dense)
    
    def test_to_dense_conversion(self):
        """Test conversion from CSC to dense."""
        # Create CSC matrix (3x3 diagonal)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        row_indices = np.array([0, 1, 2], dtype=np.int32)
        col_ptr = np.array([0, 1, 2, 3], dtype=np.int32)
        
        csc = CSCMatrix(values, row_indices, col_ptr, shape=(3, 3))
        dense = csc.to_dense()
        
        expected = np.diag([1.0, 2.0, 3.0])
        assert np.allclose(dense, expected)
    
    def test_sparse_matmul_vector(self):
        """Test CSC matrix-vector multiplication."""
        # Create simple CSC matrix
        dense = np.array([
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 5]
        ], dtype=np.float32)
        
        csc = CSCMatrix.from_dense(dense)
        
        # Test vector multiplication
        x = np.array([1, 2, 3], dtype=np.float32)
        result = csc.sparse_matmul(x)
        expected = dense @ x
        
        assert np.allclose(result, expected)
    
    def test_sparse_matmul_matrix(self):
        """Test CSC matrix-matrix multiplication."""
        dense_a = np.array([
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 5]
        ], dtype=np.float32)
        
        csc = CSCMatrix.from_dense(dense_a)
        
        # Test matrix multiplication
        B = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ], dtype=np.float32)
        
        result = csc.sparse_matmul(B)
        expected = dense_a @ B
        
        assert np.allclose(result, expected)
    
    def test_transpose_matmul(self):
        """Test efficient transpose multiplication (CSC strength)."""
        dense = np.array([
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 5]
        ], dtype=np.float32)
        
        csc = CSCMatrix.from_dense(dense)
        
        # Test A.T @ x (this is where CSC excels)
        x = np.array([1, 2, 3], dtype=np.float32)
        result = csc.transpose_matmul(x)
        expected = dense.T @ x
        
        assert np.allclose(result, expected)
    
    def test_memory_footprint(self):
        """Test memory calculation for CSC."""
        dense = np.eye(100, dtype=np.float32)
        csc = CSCMatrix.from_dense(dense)
        
        mem = csc.memory_footprint()
        
        assert 'values' in mem
        assert 'row_indices' in mem
        assert 'col_ptr' in mem
        assert 'total_sparse' in mem
        assert 'total_dense' in mem
        assert 'compression_ratio' in mem
        
        # Sparse should be smaller for diagonal matrix
        assert mem['total_sparse'] < mem['total_dense']
    
    def test_get_statistics(self):
        """Test statistics gathering for CSC."""
        dense = np.eye(50, dtype=np.float32)
        csc = CSCMatrix.from_dense(dense)
        
        stats = csc.get_statistics()
        
        assert stats.nnz == 50
        assert stats.shape == (50, 50)
        assert 0.95 < stats.sparsity < 1.0  # High sparsity for diagonal
        assert stats.compression_ratio > 1.0
    
    def test_from_csr_conversion(self):
        """Test CSR to CSC conversion."""
        dense = np.array([
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 5]
        ], dtype=np.float32)
        
        # Create CSR
        csr = CSRMatrix.from_dense(dense)
        
        # Convert to CSC
        csc = CSCMatrix.from_csr(csr)
        
        # Verify
        assert csc.shape == csr.shape
        assert np.allclose(csc.to_dense(), dense)
    
    def test_high_sparsity(self):
        """Test CSC with very sparse matrix (99%)."""
        size = 1000
        dense = np.zeros((size, size), dtype=np.float32)
        
        # Add 1% non-zeros
        num_nonzeros = int(size * size * 0.01)
        for _ in range(num_nonzeros):
            i, j = np.random.randint(0, size, 2)
            dense[i, j] = np.random.randn()
        
        csc = CSCMatrix.from_dense(dense)
        
        # Verify sparsity
        stats = csc.get_statistics()
        assert stats.sparsity > 0.98
        assert stats.compression_ratio > 10.0


class TestBlockSparseMatrix:
    """Tests for Block-sparse format (Phase 2)."""
    
    def test_basic_initialization(self):
        """Test basic block-sparse matrix creation."""
        # Create 2 blocks of 2x2
        blocks = [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32)
        ]
        block_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
        
        bsm = BlockSparseMatrix(
            blocks=blocks,
            block_indices=block_indices,
            block_size=2,
            shape=(4, 4)
        )
        
        assert bsm.shape == (4, 4)
        assert bsm.num_blocks == 2
        assert bsm.block_size == 2
    
    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        blocks = [np.array([[1, 2], [3, 4]], dtype=np.float32)]
        
        # Mismatched indices length
        with pytest.raises(ValueError, match="Number of blocks"):
            block_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
            BlockSparseMatrix(blocks, block_indices, 2, (4, 4))
        
        # Wrong block shape
        with pytest.raises(ValueError, match="Block .* has shape"):
            blocks_wrong = [np.array([[1, 2, 3]], dtype=np.float32)]
            block_indices = np.array([[0, 0]], dtype=np.int32)
            BlockSparseMatrix(blocks_wrong, block_indices, 2, (4, 4))
    
    def test_from_dense_conversion(self):
        """Test conversion from dense to block-sparse."""
        # Create block-diagonal matrix
        dense = np.block([
            [np.eye(4), np.zeros((4, 4))],
            [np.zeros((4, 4)), np.eye(4) * 2]
        ]).astype(np.float32)
        
        # Convert with block_size=4, threshold=0.2 (20% density minimum)
        # Diagonal blocks have 4/16 = 0.25 density, so they'll be kept
        bsm = BlockSparseMatrix.from_dense(dense, block_size=4, threshold=0.2)
        
        # Should have 2 blocks (diagonal blocks)
        assert bsm.num_blocks >= 2
        assert bsm.block_size == 4
    
    def test_to_dense_conversion(self):
        """Test conversion from block-sparse to dense."""
        # Create simple block-sparse matrix
        blocks = [
            np.eye(2, dtype=np.float32),
            np.eye(2, dtype=np.float32) * 2
        ]
        block_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
        
        bsm = BlockSparseMatrix(blocks, block_indices, 2, (4, 4))
        dense = bsm.to_dense()
        
        expected = np.block([
            [np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 2)), np.eye(2) * 2]
        ]).astype(np.float32)
        
        assert np.allclose(dense, expected)
    
    def test_sparse_matmul_vector(self):
        """Test block-sparse matrix-vector multiplication."""
        # Create block-diagonal matrix
        blocks = [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32)
        ]
        block_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
        
        bsm = BlockSparseMatrix(blocks, block_indices, 2, (4, 4))
        
        # Test vector multiplication
        x = np.array([1, 2, 3, 4], dtype=np.float32)
        result = bsm.sparse_matmul(x)
        
        # Compare with dense
        dense = bsm.to_dense()
        expected = dense @ x
        
        assert np.allclose(result, expected)
    
    def test_sparse_matmul_matrix(self):
        """Test block-sparse matrix-matrix multiplication."""
        blocks = [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32)
        ]
        block_indices = np.array([[0, 0], [1, 1]], dtype=np.int32)
        
        bsm = BlockSparseMatrix(blocks, block_indices, 2, (4, 4))
        
        # Test matrix multiplication
        B = np.array([
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1]
        ], dtype=np.float32)
        
        result = bsm.sparse_matmul(B)
        
        # Compare with dense
        dense = bsm.to_dense()
        expected = dense @ B
        
        assert np.allclose(result, expected)
    
    def test_memory_footprint(self):
        """Test memory calculation for block-sparse."""
        dense = np.block([
            [np.eye(8), np.zeros((8, 8))],
            [np.zeros((8, 8)), np.eye(8)]
        ]).astype(np.float32)
        
        bsm = BlockSparseMatrix.from_dense(dense, block_size=8, threshold=0.1)
        mem = bsm.memory_footprint()
        
        assert 'blocks' in mem
        assert 'indices' in mem
        assert 'total_sparse' in mem
        assert 'compression_ratio' in mem
        
        # Should have compression
        assert mem['compression_ratio'] > 1.0
    
    def test_get_statistics(self):
        """Test statistics gathering for block-sparse."""
        dense = np.eye(16, dtype=np.float32)
        bsm = BlockSparseMatrix.from_dense(dense, block_size=8, threshold=0.1)
        
        stats = bsm.get_statistics()
        
        assert 'num_blocks' in stats
        assert 'block_size' in stats
        assert 'sparsity' in stats
        assert 'compression_ratio' in stats
        assert stats['compression_ratio'] > 1.0
    
    def test_wavefront_alignment_rx580(self):
        """Test that 8x8 blocks are wavefront-aligned for RX 580."""
        dense = np.eye(64, dtype=np.float32)
        
        # Use block_size=8 (8x8=64 elements = RX 580 wavefront size)
        bsm = BlockSparseMatrix.from_dense(dense, block_size=8, threshold=0.1)
        
        stats = bsm.get_statistics()
        
        # Check wavefront alignment
        assert stats['wavefront_aligned'] == True
        assert bsm.block_size == 8
    
    def test_optimal_block_sizes(self):
        """Test various block sizes for different use cases."""
        sizes = [4, 8, 16]
        dense = np.eye(64, dtype=np.float32)
        
        for block_size in sizes:
            # Diagonal matrix has 1/block_size^2 density per block
            # For 4x4: 1/16 = 0.0625, for 8x8: 1/64 = 0.0156, for 16x16: 1/256 = 0.0039
            # Use threshold=0.01 to keep diagonal blocks
            bsm = BlockSparseMatrix.from_dense(
                dense,
                block_size=block_size,
                threshold=0.01
            )
            
            assert bsm.block_size == block_size
            assert bsm.num_blocks > 0
    
    def test_threshold_filtering(self):
        """Test that low-density blocks are filtered out."""
        # Create matrix with sparse and dense regions
        dense = np.zeros((16, 16), dtype=np.float32)
        
        # Dense block (will be kept)
        dense[0:4, 0:4] = np.random.randn(4, 4)
        
        # Very sparse block (should be filtered with high threshold)
        dense[8:12, 8:12] = np.eye(4) * 0.1  # Only diagonal
        
        # High threshold should filter sparse blocks
        bsm_high = BlockSparseMatrix.from_dense(dense, block_size=4, threshold=0.8)
        
        # Low threshold should keep both
        bsm_low = BlockSparseMatrix.from_dense(dense, block_size=4, threshold=0.1)
        
        # High threshold should have fewer blocks
        assert bsm_high.num_blocks < bsm_low.num_blocks


class TestDynamicSelection:
    """Tests for dynamic format selection (TODO: Phase 4)."""
    pass


class TestIntegration:
    """Integration tests with pruners and real workloads (TODO: Phase 5)."""
    pass
