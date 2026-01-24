#!/usr/bin/env python3
"""
Task 1.1.3 - Optimized Hybrid GEMM Kernel Wrapper

Manages compilation and execution of optimized kernels with:
- Enhanced LDS padding for bank conflict avoidance
- Optimized memory coalescing patterns
- Refined register allocation
- Kernel variant selection

Usage:
    from src.opencl.hybrid_gemm_opt import OptimizedHybridGEMMExecutor
    
    executor = OptimizedHybridGEMMExecutor()
    C = executor.gemm(A, B, alpha=1.0, beta=0.0)
"""

import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    cl = None
    logging.warning("PyOpenCL not available. Install with: pip install pyopencl")


logger = logging.getLogger(__name__)


@dataclass
class OptimizedConfig:
    """Configuration for optimized GEMM kernel."""
    
    tile_size: int = 16
    block_size: int = 2
    lds_padding: int = 2  # Enhanced from 4 to 8 bytes (2 floats)
    workgroup_size: int = 64
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.tile_size >= 8, "tile_size must be >= 8"
        assert self.block_size >= 1, "block_size must be >= 1"
        assert self.lds_padding >= 1, "lds_padding must be >= 1"
        assert self.workgroup_size == 64, "workgroup_size fixed at 64"
        logger.info(f"OptimizedConfig: tile={self.tile_size}, "
                   f"block={self.block_size}, lds_pad={self.lds_padding*4}B")
    
    def get_global_size(self, M: int, N: int) -> Tuple[int, int]:
        """Calculate global work size."""
        gx = (M + self.tile_size - 1) // self.tile_size
        gy = (N + self.tile_size - 1) // self.tile_size
        return (gx * 8, gy * 8)  # 8×8 threads per tile
    
    def get_local_size(self) -> Tuple[int, int]:
        """Return local work size."""
        return (8, 8)  # 8×8 = 64 threads
    
    def get_lds_bytes(self) -> int:
        """Calculate LDS usage."""
        # Double buffer: 2 × [16×(16+2)] floats
        lds_per_buffer = self.tile_size * (self.tile_size + self.lds_padding) * 4
        return 2 * lds_per_buffer
    
    def get_compile_options(self) -> list:
        """Get compiler options."""
        options = [
            f"-DTILE_SIZE={self.tile_size}",
            f"-DBLOCK_SIZE={self.block_size}",
            f"-DLDS_PADDING={self.lds_padding}",
            f"-DPREFETCH_DISTANCE=2",
            f"-DWORKGROUP_SIZE={self.workgroup_size}",
            "-cl-mad-enable",           # Enable MAD (fused multiply-add)
            "-cl-no-signed-zeros",      # Allow sign changes
            "-cl-unsafe-math-optimizations",  # Enable unsafe optimizations
            "-cl-finite-math-only",     # Assume finite math
            "-cl-fast-relaxed-math",    # Fast relaxed math
        ]
        return options


class OptimizedKernelManager:
    """Manages compilation and caching of optimized kernels."""
    
    def __init__(self, context: Optional[cl.Context] = None, 
                 config: Optional[OptimizedConfig] = None):
        """Initialize kernel manager.
        
        Args:
            context: PyOpenCL context (created if None)
            config: OptimizedConfig (default if None)
        """
        if cl is None:
            raise RuntimeError("PyOpenCL not available")
        
        self.config = config or OptimizedConfig()
        self.context = context or self._create_context()
        self.kernels = {}
        
        logger.info(f"Initializing OptimizedKernelManager")
        self._compile_kernels()
    
    def _create_context(self) -> cl.Context:
        """Create OpenCL context."""
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
        
        platform = platforms[0]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        
        if not devices:
            raise RuntimeError("No GPU devices found")
        
        return cl.Context(devices)
    
    def _compile_kernels(self):
        """Compile all optimized kernel variants."""
        kernel_path = Path(__file__).parent / "kernels" / "gemm_hybrid_opt.cl"
        
        if not kernel_path.exists():
            raise FileNotFoundError(f"Kernel source not found: {kernel_path}")
        
        with open(kernel_path, 'r') as f:
            source = f.read()
        
        logger.info(f"Compiling optimized kernels from {kernel_path}")
        
        try:
            program = cl.Program(self.context, source).build(
                options=self.config.get_compile_options()
            )
            logger.info("✅ Kernel compilation successful")
        except cl.cffi_cl.LogicError as e:
            logger.error(f"❌ Compilation failed:\n{e}")
            raise
        
        # Cache kernel variants
        kernel_names = [
            "gemm_hybrid_float4_lds_opt",
            "gemm_hybrid_float4_full_opt",
            "gemm_hybrid_float4_beta_zero_opt"
        ]
        
        for name in kernel_names:
            try:
                self.kernels[name] = getattr(program, name)
                logger.info(f"  ✓ {name}")
            except AttributeError:
                logger.warning(f"  ⚠ {name} not found in program")
    
    def select_kernel(self, beta: float) -> str:
        """Select optimal kernel based on parameters.
        
        Args:
            beta: Beta scaling factor
            
        Returns:
            Kernel name to use
        """
        if beta < 1e-10:  # Effectively zero
            return "gemm_hybrid_float4_beta_zero_opt"
        else:
            return "gemm_hybrid_float4_full_opt"  # General case


class OptimizedHybridGEMMExecutor:
    """High-level executor for optimized GEMM operations."""
    
    def __init__(self, context: Optional[cl.Context] = None,
                 config: Optional[OptimizedConfig] = None,
                 kernel_manager: Optional[OptimizedKernelManager] = None):
        """Initialize executor.
        
        Args:
            context: PyOpenCL context
            config: OptimizedConfig
            kernel_manager: Custom kernel manager
        """
        if cl is None:
            raise RuntimeError("PyOpenCL not available")
        
        self.config = config or OptimizedConfig()
        self.manager = kernel_manager or OptimizedKernelManager(context, config)
        self.context = self.manager.context
        self.queue = cl.CommandQueue(self.context)
        
        logger.info("OptimizedHybridGEMMExecutor initialized")
    
    def gemm(self, A: np.ndarray, B: np.ndarray, 
             C: Optional[np.ndarray] = None,
             alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
        """Execute optimized GEMM: C = alpha*A@B + beta*C
        
        Args:
            A: Matrix A (M×K, float32)
            B: Matrix B (K×N, float32)
            C: Matrix C (M×N, float32) - initialized if None
            alpha: Scaling factor for A@B
            beta: Scaling factor for C
            
        Returns:
            Result matrix C (M×N, float32)
        """
        # Validate inputs
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        
        if not A.flags['C_CONTIGUOUS']:
            A = np.ascontiguousarray(A)
        if not B.flags['C_CONTIGUOUS']:
            B = np.ascontiguousarray(B)
        
        M, K = A.shape
        K_B, N = B.shape
        
        if K != K_B:
            raise ValueError(f"Dimension mismatch: A ({M}×{K}) @ B ({K_B}×{N})")
        
        if C is None:
            C = np.zeros((M, N), dtype=np.float32)
        else:
            C = np.asarray(C, dtype=np.float32)
            if not C.flags['C_CONTIGUOUS']:
                C = np.ascontiguousarray(C)
        
        if C.shape != (M, N):
            raise ValueError(f"C shape mismatch: expected ({M}×{N}), got {C.shape}")
        
        logger.debug(f"GEMM: {M}×{K} @ {K}×{N} = {M}×{N}")
        
        # Select kernel variant
        kernel_name = self.manager.select_kernel(beta)
        kernel = self.manager.kernels[kernel_name]
        logger.debug(f"Selected kernel: {kernel_name}")
        
        # Allocate GPU buffers
        mf = cl.mem_flags
        A_gpu = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_gpu = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_gpu = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)
        
        try:
            # Set kernel arguments
            kernel.set_arg(0, A_gpu)
            kernel.set_arg(1, B_gpu)
            kernel.set_arg(2, C_gpu)
            kernel.set_arg(3, np.int32(M))
            kernel.set_arg(4, np.int32(N))
            kernel.set_arg(5, np.int32(K))
            kernel.set_arg(6, np.float32(alpha))
            kernel.set_arg(7, np.float32(beta))
            
            # Execute kernel
            global_size = self.config.get_global_size(M, N)
            local_size = self.config.get_local_size()
            
            logger.debug(f"Launching kernel: global={global_size}, local={local_size}")
            
            event = cl.enqueue_nd_range_kernel(
                self.queue, kernel, global_size, local_size
            )
            event.wait()
            
            # Read result
            result = np.empty_like(C)
            cl.enqueue_copy(self.queue, result, C_gpu)
            self.queue.finish()
            
            logger.debug("Kernel execution complete")
            return result
        
        finally:
            A_gpu.release()
            B_gpu.release()
            C_gpu.release()
    
    def benchmark(self, M: int, N: int, K: int, 
                  iterations: int = 10) -> dict:
        """Benchmark optimized kernel.
        
        Args:
            M, N, K: Matrix dimensions
            iterations: Number of iterations
            
        Returns:
            Dictionary with timing and GFLOPS
        """
        import time
        
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # Warmup
        _ = self.gemm(A, B)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = self.gemm(A, B)
            times.append((time.perf_counter() - start) * 1000)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        gflops = (2 * M * N * K) / (mean_time / 1000) / 1e9
        
        return {
            'mean_time_ms': mean_time,
            'std_time_ms': std_time,
            'gflops': gflops,
            'iterations': iterations
        }


def main():
    """Test optimized executor."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Testing OptimizedHybridGEMMExecutor")
    
    try:
        executor = OptimizedHybridGEMMExecutor()
        
        # Test with small matrix
        n = 256
        A = np.random.randn(n, n).astype(np.float32)
        B = np.random.randn(n, n).astype(np.float32)
        
        C = executor.gemm(A, B)
        
        # Verify correctness
        C_ref = A @ B
        error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)
        
        logger.info(f"Test {n}×{n}: error={error:.2e}")
        
        if error < 1e-4:
            logger.info("✅ Correctness test PASSED")
        else:
            logger.error("❌ Correctness test FAILED")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == '__main__':
    main()
