"""
Unit Tests for OpenCL GEMM Operations
======================================

Comprehensive tests for GEMM kernel correctness and performance.

Test Categories:
----------------
1. Correctness: Verify against NumPy reference
2. Edge Cases: Empty, single element, non-square matrices
3. Alpha/Beta: Test scaling parameters
4. Performance: Ensure reasonable GFLOPS achieved
5. Memory: Verify no leaks or corruption

Run with: pytest tests/test_opencl_gemm.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.opencl import CLContext, gemm
    from src.opencl.ops import benchmark_gemm
    OPENCL_AVAILABLE = True
except (ImportError, RuntimeError):
    OPENCL_AVAILABLE = False


# Skip all tests if OpenCL not available
pytestmark = pytest.mark.skipif(
    not OPENCL_AVAILABLE,
    reason="OpenCL not available"
)


@pytest.fixture(scope="module")
def cl_context():
    """Create OpenCL context once for all tests."""
    if not OPENCL_AVAILABLE:
        return None
    return CLContext()


class TestGEMMCorrectness:
    """Test GEMM correctness against NumPy reference."""
    
    def test_basic_multiplication(self, cl_context):
        """Test basic matrix multiplication."""
        A = np.random.randn(64, 32).astype(np.float32)
        B = np.random.randn(32, 48).astype(np.float32)
        
        C_opencl = gemm(cl_context, A, B)
        C_numpy = A @ B
        
        assert np.allclose(C_opencl, C_numpy, rtol=1e-4, atol=1e-5), \
            f"Max error: {np.max(np.abs(C_opencl - C_numpy))}"
    
    def test_square_matrices(self, cl_context):
        """Test square matrix multiplication."""
        for size in [16, 32, 64, 128]:
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            
            C_opencl = gemm(cl_context, A, B)
            C_numpy = A @ B
            
            assert np.allclose(C_opencl, C_numpy, rtol=1e-4, atol=1e-5), \
                f"Failed for size {size}"
    
    def test_rectangular_matrices(self, cl_context):
        """Test non-square matrices."""
        test_cases = [
            (100, 50, 75),   # Tall x Wide
            (50, 100, 25),   # Wide x Tall
            (1, 100, 1),     # Row vector x column vector
            (100, 1, 100),   # Column vector x row vector
        ]
        
        for M, K, N in test_cases:
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            
            C_opencl = gemm(cl_context, A, B)
            C_numpy = A @ B
            
            assert np.allclose(C_opencl, C_numpy, rtol=1e-4, atol=1e-5), \
                f"Failed for shape ({M}, {K}) @ ({K}, {N})"
    
    def test_alpha_beta_parameters(self, cl_context):
        """Test alpha and beta scaling."""
        A = np.random.randn(32, 16).astype(np.float32)
        B = np.random.randn(16, 24).astype(np.float32)
        C_init = np.random.randn(32, 24).astype(np.float32)
        
        # Test different alpha/beta combinations
        test_cases = [
            (1.0, 0.0),   # Standard multiplication
            (2.0, 0.0),   # Scaled multiplication
            (1.0, 1.0),   # With accumulation
            (0.5, 2.0),   # Both scaled
            (0.0, 1.0),   # Only beta (should return beta * C)
        ]
        
        for alpha, beta in test_cases:
            C = C_init.copy()
            C_opencl = gemm(cl_context, A, B, alpha=alpha, beta=beta, C=C)
            C_numpy = alpha * (A @ B) + beta * C_init
            
            assert np.allclose(C_opencl, C_numpy, rtol=1e-4, atol=1e-5), \
                f"Failed for alpha={alpha}, beta={beta}"
    
    def test_identity_matrix(self, cl_context):
        """Test multiplication with identity matrix."""
        size = 64
        A = np.random.randn(size, size).astype(np.float32)
        I = np.eye(size, dtype=np.float32)
        
        # A @ I should equal A
        C1 = gemm(cl_context, A, I)
        assert np.allclose(C1, A, rtol=1e-4), "A @ I != A"
        
        # I @ A should equal A
        C2 = gemm(cl_context, I, A)
        assert np.allclose(C2, A, rtol=1e-4), "I @ A != A"
    
    def test_zero_matrix(self, cl_context):
        """Test multiplication with zero matrix."""
        A = np.random.randn(32, 16).astype(np.float32)
        Z = np.zeros((16, 24), dtype=np.float32)
        
        C = gemm(cl_context, A, Z)
        assert np.allclose(C, 0.0, atol=1e-6), "A @ 0 should be 0"


class TestGEMMEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_small_matrices(self, cl_context):
        """Test very small matrices."""
        # Single element
        A = np.array([[2.0]], dtype=np.float32)
        B = np.array([[3.0]], dtype=np.float32)
        C = gemm(cl_context, A, B)
        assert np.allclose(C, [[6.0]], rtol=1e-5)
        
        # 2x2
        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        C = gemm(cl_context, A, B)
        C_numpy = A @ B
        assert np.allclose(C, C_numpy, rtol=1e-5)
    
    def test_non_tile_aligned(self, cl_context):
        """Test matrices not aligned to tile size (16)."""
        # Prime numbers to ensure non-alignment
        test_sizes = [
            (17, 13, 19),
            (31, 29, 37),
            (100, 99, 101),
        ]
        
        for M, K, N in test_sizes:
            A = np.random.randn(M, K).astype(np.float32)
            B = np.random.randn(K, N).astype(np.float32)
            
            C_opencl = gemm(cl_context, A, B)
            C_numpy = A @ B
            
            assert np.allclose(C_opencl, C_numpy, rtol=1e-4, atol=1e-5), \
                f"Failed for non-aligned size ({M}, {K}, {N})"
    
    def test_large_matrices(self, cl_context):
        """Test large matrices (stress test)."""
        # Should not crash or produce NaN
        A = np.random.randn(1024, 512).astype(np.float32)
        B = np.random.randn(512, 2048).astype(np.float32)
        
        C = gemm(cl_context, A, B)
        
        assert not np.isnan(C).any(), "Result contains NaN"
        assert not np.isinf(C).any(), "Result contains Inf"
        assert C.shape == (1024, 2048), f"Wrong shape: {C.shape}"


class TestGEMMKernelVariants:
    """Test different kernel implementations."""
    
    def test_naive_kernel(self, cl_context):
        """Test naive (non-tiled) kernel."""
        A = np.random.randn(64, 32).astype(np.float32)
        B = np.random.randn(32, 48).astype(np.float32)
        
        C = gemm(cl_context, A, B, use_tiled=False)
        C_numpy = A @ B
        
        assert np.allclose(C, C_numpy, rtol=1e-4, atol=1e-5)
    
    def test_tiled_kernel(self, cl_context):
        """Test tiled kernel (default)."""
        A = np.random.randn(128, 64).astype(np.float32)
        B = np.random.randn(64, 256).astype(np.float32)
        
        C = gemm(cl_context, A, B, use_tiled=True)
        C_numpy = A @ B
        
        assert np.allclose(C, C_numpy, rtol=1e-4, atol=1e-5)
    
    def test_2x2_kernel(self, cl_context):
        """Test 2x2 blocking kernel."""
        A = np.random.randn(256, 128).astype(np.float32)
        B = np.random.randn(128, 512).astype(np.float32)
        
        C = gemm(cl_context, A, B, use_2x2=True)
        C_numpy = A @ B
        
        assert np.allclose(C, C_numpy, rtol=1e-4, atol=1e-5)
    
    def test_kernel_consistency(self, cl_context):
        """Verify all kernel variants produce same result."""
        A = np.random.randn(128, 64).astype(np.float32)
        B = np.random.randn(64, 128).astype(np.float32)
        
        C_naive = gemm(cl_context, A, B, use_tiled=False)
        C_tiled = gemm(cl_context, A, B, use_tiled=True)
        C_2x2 = gemm(cl_context, A, B, use_2x2=True)
        
        # All should match
        assert np.allclose(C_naive, C_tiled, rtol=1e-4, atol=1e-5), \
            "Naive and tiled kernels differ"
        assert np.allclose(C_tiled, C_2x2, rtol=1e-4, atol=1e-5), \
            "Tiled and 2x2 kernels differ"


class TestGEMMPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.slow
    def test_benchmark_small(self, cl_context):
        """Benchmark small matrix multiplication."""
        results = benchmark_gemm(cl_context, M=256, N=256, K=256, num_trials=10)
        
        print(f"\nSmall GEMM (256x256x256):")
        print(f"  GFLOPS: {results['gflops']:.2f}")
        print(f"  Time: {results['time_ms']:.3f} ms")
        print(f"  Bandwidth: {results['bandwidth_gb_s']:.2f} GB/s")
        
        # Should achieve at least 50 GFLOPS (very conservative)
        assert results['gflops'] > 50, \
            f"Performance too low: {results['gflops']:.2f} GFLOPS"
    
    @pytest.mark.slow
    def test_benchmark_medium(self, cl_context):
        """Benchmark medium matrix multiplication."""
        results = benchmark_gemm(cl_context, M=512, N=512, K=512, num_trials=10)
        
        print(f"\nMedium GEMM (512x512x512):")
        print(f"  GFLOPS: {results['gflops']:.2f}")
        print(f"  Time: {results['time_ms']:.3f} ms")
        print(f"  Bandwidth: {results['bandwidth_gb_s']:.2f} GB/s")
        
        # Should improve with larger matrices
        assert results['gflops'] > 100, \
            f"Performance too low: {results['gflops']:.2f} GFLOPS"
    
    @pytest.mark.slow
    def test_benchmark_large(self, cl_context):
        """Benchmark large matrix multiplication."""
        results = benchmark_gemm(cl_context, M=1024, N=1024, K=1024, num_trials=5)
        
        print(f"\nLarge GEMM (1024x1024x1024):")
        print(f"  GFLOPS: {results['gflops']:.2f}")
        print(f"  Time: {results['time_ms']:.3f} ms")
        print(f"  Bandwidth: {results['bandwidth_gb_s']:.2f} GB/s")
        
        # Should achieve 500+ GFLOPS on RX 580
        assert results['gflops'] > 200, \
            f"Performance too low: {results['gflops']:.2f} GFLOPS"


class TestGEMMErrors:
    """Test error handling."""
    
    def test_dimension_mismatch(self, cl_context):
        """Test incompatible matrix dimensions."""
        A = np.random.randn(10, 20).astype(np.float32)
        B = np.random.randn(15, 30).astype(np.float32)  # Wrong K dimension
        
        with pytest.raises(ValueError, match="incompatible"):
            gemm(cl_context, A, B)
    
    def test_wrong_dtype(self, cl_context):
        """Test non-float32 dtype."""
        A = np.random.randn(10, 20).astype(np.float64)
        B = np.random.randn(20, 30).astype(np.float32)
        
        with pytest.raises(ValueError, match="float32"):
            gemm(cl_context, A, B)
    
    def test_wrong_ndim(self, cl_context):
        """Test non-2D arrays."""
        A = np.random.randn(10, 20, 5).astype(np.float32)  # 3D
        B = np.random.randn(20, 30).astype(np.float32)
        
        with pytest.raises(ValueError, match="2D"):
            gemm(cl_context, A, B)
    
    def test_wrong_c_shape(self, cl_context):
        """Test incompatible C matrix shape."""
        A = np.random.randn(10, 20).astype(np.float32)
        B = np.random.randn(20, 30).astype(np.float32)
        C = np.zeros((10, 40), dtype=np.float32)  # Wrong N
        
        with pytest.raises(ValueError, match="must match"):
            gemm(cl_context, A, B, C=C)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
