#!/usr/bin/env python3
"""
Phase 2, Technique 1: Block Recursive GEMM Wrapper

Based on successful hybrid_gemm_opt.py architecture from Phase 1.
Uses proven configuration and compilation strategy.

Target: 850-870 GFLOPS (+10-12% vs Phase 1's 775 GFLOPS)
"""

import sys
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import time

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    cl = None
    logging.warning("PyOpenCL not available")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class RecursiveGEMMConfig:
    """Configuration for recursive GEMM kernels."""
    
    tile_size: int = 16
    block_size: int = 2  
    lds_padding: int = 2  # 8 bytes (2 floats)
    workgroup_size: int = 64  # 8×8
    
    def get_global_size(self, M: int, N: int) -> Tuple[int, int]:
        """Calculate global work size."""
        gx = (M + self.tile_size - 1) // self.tile_size
        gy = (N + self.tile_size - 1) // self.tile_size
        return (gx * 8, gy * 8)
    
    def get_local_size(self) -> Tuple[int, int]:
        """Return local work size."""
        return (8, 8)
    
    def get_compile_options(self) -> list:
        """Get compiler options with defines."""
        return [
            f"-DTS={self.tile_size}",
            f"-DBS={self.block_size}",
            f"-DLDS_PAD={self.lds_padding}",
            f"-DWG_SIZE={self.workgroup_size}",
            "-cl-mad-enable",
            "-cl-unsafe-math-optimizations",
            "-cl-fast-relaxed-math",
        ]


class RecursiveKernelManager:
    """Manages kernel compilation and caching."""
    
    def __init__(self, context: Optional[cl.Context] = None,
                 config: Optional[RecursiveGEMMConfig] = None):
        if cl is None:
            raise RuntimeError("PyOpenCL not available")
        
        self.config = config or RecursiveGEMMConfig()
        self.context = context or self._create_context()
        self.device = self.context.devices[0]
        self.kernels = {}
        
        logger.info(f"Device: {self.device.name}")
        logger.info(f"Config: TS={self.config.tile_size}, BS={self.config.block_size}")
        
        self._compile_kernels()
    
    def _create_context(self) -> cl.Context:
        """Create OpenCL context."""
        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                logger.info(f"Using platform: {platform.name}")
                return cl.Context(devices[:1])
        
        raise RuntimeError("No GPU devices found")
    
    def _compile_kernels(self):
        """Compile all kernel variants."""
        # Find kernel source file
        kernel_paths = [
            Path(__file__).parent / "kernels" / "gemm_recursive_v5.cl",
            Path("src/opencl/kernels/gemm_recursive_v5.cl"),
            Path(__file__).parent / "kernels" / "gemm_recursive_v4.cl",
            Path("src/opencl/kernels/gemm_recursive_v4.cl"),
        ]
        
        kernel_path = None
        for path in kernel_paths:
            if path.exists():
                kernel_path = path
                break
        
        if not kernel_path:
            raise FileNotFoundError("Could not find gemm_recursive_v4.cl")
        
        logger.info(f"Loading kernels from: {kernel_path}")
        
        with open(kernel_path, 'r') as f:
            source = f.read()
        
        # Compile with options
        options = " ".join(self.config.get_compile_options())
        logger.info(f"Compile options: {options}")
        
        try:
            program = cl.Program(self.context, source).build(options=options)
            
            # Cache kernel objects
            kernel_names = [
                "gemm_recursive_block",
                "gemm_recursive_two_level", 
                "gemm_recursive_optimized"
            ]
            
            for name in kernel_names:
                try:
                    self.kernels[name] = getattr(program, name)
                    logger.info(f"  ✓ {name}")
                except AttributeError:
                    logger.warning(f"  ⚠ {name} not found")
                    
        except cl.RuntimeError as e:
            logger.error(f"Kernel compilation failed: {e}")
            raise


class RecursiveGEMMExecutor:
    """High-level executor for recursive GEMM."""
    
    def __init__(self, variant: str = "optimized",
                 context: Optional[cl.Context] = None,
                 config: Optional[RecursiveGEMMConfig] = None):
        """Initialize executor.
        
        Args:
            variant: Kernel variant ('basic', 'two_level', 'optimized')
            context: PyOpenCL context
            config: Configuration
        """
        if cl is None:
            raise RuntimeError("PyOpenCL not available")
        
        self.variant = variant
        self.config = config or RecursiveGEMMConfig()
        self.manager = RecursiveKernelManager(context, self.config)
        self.context = self.manager.context
        self.queue = cl.CommandQueue(
            self.context,
            properties=cl.command_queue_properties.PROFILING_ENABLE
        )
        
        # Select kernel
        kernel_map = {
            "basic": "gemm_recursive_block",
            "two_level": "gemm_recursive_two_level",
            "optimized": "gemm_recursive_optimized"
        }
        
        kernel_name = kernel_map.get(variant, "gemm_recursive_optimized")
        self.kernel = self.manager.kernels[kernel_name]
        
        logger.info(f"Executor initialized with variant: {variant}")
    
    def gemm(self, A: np.ndarray, B: np.ndarray,
             C: Optional[np.ndarray] = None,
             alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
        """Execute GEMM: C = alpha*A@B + beta*C
        
        Args:
            A: Matrix A (M×K, float32)
            B: Matrix B (K×N, float32)  
            C: Matrix C (M×N, float32)
            alpha: Scaling for A@B
            beta: Scaling for C
            
        Returns:
            Result matrix C
        """
        # Ensure contiguous float32
        A = np.ascontiguousarray(A, dtype=np.float32)
        B = np.ascontiguousarray(B, dtype=np.float32)
        
        M, K = A.shape
        K_B, N = B.shape
        
        if K != K_B:
            raise ValueError(f"Dimension mismatch: A ({M}×{K}) @ B ({K_B}×{N})")
        
        if C is None:
            C = np.zeros((M, N), dtype=np.float32)
        else:
            C = np.ascontiguousarray(C, dtype=np.float32)
        
        # Allocate GPU buffers
        mf = cl.mem_flags
        A_gpu = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_gpu = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_gpu = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)
        
        try:
            # Set kernel arguments (individually for compatibility)
            self.kernel.set_arg(0, A_gpu)
            self.kernel.set_arg(1, B_gpu)
            self.kernel.set_arg(2, C_gpu)
            self.kernel.set_arg(3, np.int32(M))
            self.kernel.set_arg(4, np.int32(N))
            self.kernel.set_arg(5, np.int32(K))
            self.kernel.set_arg(6, np.float32(alpha))
            self.kernel.set_arg(7, np.float32(beta))
            
            # Execute
            global_size = self.config.get_global_size(M, N)
            local_size = self.config.get_local_size()
            
            event = cl.enqueue_nd_range_kernel(
                self.queue, self.kernel, global_size, local_size
            )
            event.wait()
            
            # Read result
            result = np.empty_like(C)
            cl.enqueue_copy(self.queue, result, C_gpu)
            self.queue.finish()
            
            return result
            
        finally:
            A_gpu.release()
            B_gpu.release()
            C_gpu.release()
    
    def benchmark(self, sizes: list = [256, 512, 1024, 2048],
                  iterations: int = 10) -> dict:
        """Benchmark kernel across different sizes.
        
        Args:
            sizes: Matrix sizes to test
            iterations: Number of iterations per size
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        for n in sizes:
            logger.info(f"Benchmarking {n}×{n}...")
            
            A = np.random.randn(n, n).astype(np.float32)
            B = np.random.randn(n, n).astype(np.float32)
            
            # Warmup
            _ = self.gemm(A, B)
            
            # Benchmark
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                C = self.gemm(A, B)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            flops = 2 * n**3
            gflops = flops / avg_time / 1e9
            
            # Verify
            C_ref = A @ B
            error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)
            
            results[n] = {
                'gflops': gflops,
                'time_mean': avg_time,
                'time_std': std_time,
                'error': error,
                'cv_percent': (std_time / avg_time) * 100
            }
            
            logger.info(f"  {gflops:.1f} GFLOPS (error: {error:.2e}, CV: {results[n]['cv_percent']:.2f}%)")
        
        return results


def main():
    """Quick test."""
    print("Phase 2, Technique 1: Block Recursive GEMM")
    print("=" * 60)
    
    # Test all 3 variants
    for variant in ["basic", "two_level", "optimized"]:
        print(f"\nTesting {variant} variant...")
        
        executor = RecursiveGEMMExecutor(variant=variant)
        results = executor.benchmark(sizes=[1024], iterations=10)
        
        for size, metrics in results.items():
            print(f"  {size}×{size}: {metrics['gflops']:.1f} GFLOPS")


if __name__ == '__main__':
    main()
