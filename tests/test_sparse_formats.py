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


# Placeholder test classes for future phases
class TestCSCMatrix:
    """Tests for CSC format (TODO: Phase 2)."""
    pass


class TestBlockSparseMatrix:
    """Tests for Block-sparse format (TODO: Phase 3)."""
    pass


class TestDynamicSelection:
    """Tests for dynamic format selection (TODO: Phase 4)."""
    pass


class TestIntegration:
    """Integration tests with pruners and real workloads (TODO: Phase 5)."""
    pass
