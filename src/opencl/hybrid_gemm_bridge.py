"""
Integration module for Hybrid GEMM with existing GEMM infrastructure.

Provides drop-in replacement and comparison with existing kernels.
"""

import numpy as np
import logging
from typing import Optional, Callable
from src.opencl.hybrid_gemm import HybridGEMMExecutor, HybridGEMMConfig

logger = logging.getLogger(__name__)


class HybridGEMMBridge:
    """
    Bridge between new Hybrid GEMM and existing GEMM infrastructure.
    
    Provides:
    - Unified interface for comparing kernels
    - Automatic kernel selection based on problem size
    - Fallback mechanisms
    - Performance statistics
    """
    
    def __init__(self, fallback_gemm_func: Optional[Callable] = None):
        """
        Initialize bridge.
        
        Args:
            fallback_gemm_func: Function to call if hybrid kernel fails
                              Signature: gemm(A, B, C=None, alpha=1.0, beta=0.0) -> ndarray
        """
        self.executor = HybridGEMMExecutor()
        self.fallback_gemm = fallback_gemm_func
        self.stats = {
            'hybrid_count': 0,
            'fallback_count': 0,
            'hybrid_gflops': [],
            'fallback_gflops': [],
        }
    
    def gemm(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        force_hybrid: bool = False,
        force_fallback: bool = False,
    ) -> np.ndarray:
        """
        Execute GEMM with automatic kernel selection.
        
        Args:
            A, B, C: Matrices
            alpha, beta: Scalars
            force_hybrid: Force use of hybrid kernel
            force_fallback: Force use of fallback kernel
        
        Returns:
            Result matrix C
        """
        # Validate inputs
        M, K = A.shape
        K2, N = B.shape
        
        if K != K2:
            raise ValueError(f"Dimension mismatch: A[1]={K} != B[0]={K2}")
        
        # Decide which kernel to use
        use_hybrid = self._should_use_hybrid(M, N, K, force_hybrid, force_fallback)
        
        try:
            if use_hybrid:
                result = self.executor(A, B, C, alpha=alpha, beta=beta)
                self.stats['hybrid_count'] += 1
                return result
            else:
                if self.fallback_gemm is None:
                    raise RuntimeError("No fallback GEMM available")
                result = self.fallback_gemm(A, B, C, alpha=alpha, beta=beta)
                self.stats['fallback_count'] += 1
                return result
        
        except Exception as e:
            logger.warning(f"Hybrid GEMM failed: {e}. Trying fallback.")
            
            if self.fallback_gemm is None:
                raise RuntimeError(f"Both hybrid and fallback failed: {e}")
            
            result = self.fallback_gemm(A, B, C, alpha=alpha, beta=beta)
            self.stats['fallback_count'] += 1
            return result
    
    def _should_use_hybrid(
        self,
        M: int,
        N: int,
        K: int,
        force_hybrid: bool = False,
        force_fallback: bool = False
    ) -> bool:
        """Decide whether to use hybrid kernel."""
        if force_hybrid:
            return True
        if force_fallback:
            return False
        
        # Heuristics for when hybrid kernel is best
        # - Works well for square(ish) matrices
        # - Works well for larger sizes (n > 256)
        # - Tile size 16 assumption
        
        size = max(M, N, K)
        
        # Hybrid kernel works best for medium-to-large matrices
        if size < 128:
            return False  # Too small, overhead dominates
        
        if size > 16384:
            return False  # Too large, need specialized kernels (FFT, recursive)
        
        return True
    
    def compare_kernels(
        self,
        A: np.ndarray,
        B: np.ndarray,
        alpha: float = 1.0,
        beta: float = 0.0,
        verbose: bool = True
    ) -> dict:
        """
        Compare hybrid and fallback kernels on same problem.
        
        Returns:
            Dictionary with timing and accuracy results
        """
        if self.fallback_gemm is None:
            logger.warning("No fallback kernel for comparison")
            return {}
        
        import time
        
        # Execute hybrid
        start = time.perf_counter()
        C_hybrid = self.executor(A, B, alpha=alpha, beta=beta)
        time_hybrid = (time.perf_counter() - start) * 1000
        
        # Execute fallback
        C = None
        if beta != 0.0:
            C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
        
        start = time.perf_counter()
        C_fallback = self.fallback_gemm(A, B, C, alpha=alpha, beta=beta)
        time_fallback = (time.perf_counter() - start) * 1000
        
        # Compute metrics
        M, N = A.shape[0], B.shape[1]
        gflops = 2 * M * N * A.shape[1] / 1e9
        gflops_hybrid = gflops / (time_hybrid / 1000)
        gflops_fallback = gflops / (time_fallback / 1000)
        
        error = np.linalg.norm(C_hybrid - C_fallback) / np.linalg.norm(C_fallback)
        speedup = time_fallback / time_hybrid
        
        result = {
            'matrix_size': (M, N, A.shape[1]),
            'time_hybrid_ms': time_hybrid,
            'time_fallback_ms': time_fallback,
            'gflops_hybrid': gflops_hybrid,
            'gflops_fallback': gflops_fallback,
            'speedup': speedup,
            'relative_error': error,
        }
        
        if verbose:
            logger.info("Kernel Comparison:")
            logger.info(f"  Size: {M}×{N} (K={A.shape[1]})")
            logger.info(f"  Hybrid:   {gflops_hybrid:.1f} GFLOPS ({time_hybrid:.3f} ms)")
            logger.info(f"  Fallback: {gflops_fallback:.1f} GFLOPS ({time_fallback:.3f} ms)")
            logger.info(f"  Speedup:  {speedup:.2f}x")
            logger.info(f"  Error:    {error:.2e}")
        
        return result
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'hybrid_count': 0,
            'fallback_count': 0,
            'hybrid_gflops': [],
            'fallback_gflops': [],
        }


def create_unified_gemm(
    existing_gemm_func: Optional[Callable] = None
) -> HybridGEMMBridge:
    """
    Create unified GEMM interface combining hybrid and existing kernels.
    
    Args:
        existing_gemm_func: Existing GEMM implementation to use as fallback
    
    Returns:
        HybridGEMMBridge instance ready to use
    """
    return HybridGEMMBridge(fallback_gemm_func=existing_gemm_func)


# Example integration with existing code
def integrate_with_existing():
    """
    Example of how to integrate hybrid GEMM with existing infrastructure.
    
    This shows how to replace existing GEMM calls with hybrid kernel.
    """
    
    # Create bridge with existing GEMM as fallback
    from src.compute.matrix_operations import gemm as existing_gemm  # Hypothetical
    
    unified_gemm = create_unified_gemm(existing_gemm)
    
    # Now use unified_gemm instead of gemm:
    # OLD: C = existing_gemm(A, B)
    # NEW: C = unified_gemm.gemm(A, B)
    
    return unified_gemm


if __name__ == "__main__":
    # Test integration
    logging.basicConfig(level=logging.INFO)
    
    # Create bridge without fallback (hybrid only)
    bridge = HybridGEMMBridge()
    
    # Test on various sizes
    sizes = [256, 512, 1024, 2048]
    
    logger.info("Integration Test: Hybrid GEMM\n")
    
    for size in sizes:
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        
        C = bridge.gemm(A, B, alpha=1.0, beta=0.0)
        
        # Verify correctness
        C_ref = A @ B
        error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)
        
        logger.info(f"n={size}: ✅ error={error:.2e}")
    
    logger.info(f"\nTotal calls: {bridge.stats['hybrid_count']} hybrid")
