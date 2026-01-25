#!/usr/bin/env python3
"""
Phase 3 - SIMD Vectorized GEMM Kernel Wrapper

Manages compilation and execution of vectorized kernels with:
- Float4 vectorization for 4x bandwidth utilization
- SIMD lane optimization for GCN 4.0
- Double buffering with vectorized loads
- Memory coalescing for optimal global memory access

Target: 200-300 GFLOPS (3-5x improvement over scalar version)

Usage:
    from src.opencl.gemm_vectorized import VectorizedGEMMExecutor

    executor = VectorizedGEMMExecutor()
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
class VectorizedConfig:
    """Configuration for vectorized GEMM kernel."""

    tile_size: int = 16
    wg_size_x: int = 16
    wg_size_y: int = 16
    lds_padding: int = 2  # 8 bytes (2 floats) for bank conflict avoidance
    vector_width: int = 4  # Float4 vectorization

    def __post_init__(self):
        """Validate configuration."""
        assert self.tile_size >= 8, "tile_size must be >= 8"
        assert self.wg_size_x >= 4, "wg_size_x must be >= 4"
        assert self.wg_size_y >= 4, "wg_size_y must be >= 4"
        assert self.lds_padding >= 1, "lds_padding must be >= 1"
        assert self.vector_width in [4, 8], "vector_width must be 4 or 8"


class VectorizedGEMMExecutor:
    """
    SIMD Vectorized GEMM Executor for GCN 4.0

    Features:
    - Float4 vectorization for maximum bandwidth utilization
    - SIMD lane optimization for Polaris 10
    - Double buffering with vectorized memory operations
    - Memory coalescing for optimal global memory access
    """

    def __init__(self, config: Optional[VectorizedConfig] = None):
        """Initialize the vectorized GEMM executor."""
        self.config = config or VectorizedConfig()
        self.context = None
        self.queue = None
        self.program = None
        self.kernels = {}

        # Initialize OpenCL
        self._init_opencl()

        # Build kernels
        self._build_kernels()

    def _init_opencl(self):
        """Initialize OpenCL context and queue."""
        if cl is None:
            raise RuntimeError("PyOpenCL not available")

        try:
            # Get platform and device
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found")

            # Prefer AMD platform
            amd_platform = None
            for platform in platforms:
                if 'AMD' in platform.name.upper() or 'RADEON' in platform.name.upper():
                    amd_platform = platform
                    break

            if amd_platform:
                platform = amd_platform
                logger.info(f"Using AMD platform: {platform.name}")
            else:
                platform = platforms[0]
                logger.warning(f"No AMD platform found, using: {platform.name}")

            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if not devices:
                devices = platform.get_devices()  # Fallback to any device

            if not devices:
                raise RuntimeError("No OpenCL devices found")

            device = devices[0]
            logger.info(f"Using device: {device.name}")

            # Create context and queue
            self.context = cl.Context([device])
            self.queue = cl.CommandQueue(self.context)

        except Exception as e:
            logger.error(f"Failed to initialize OpenCL: {e}")
            raise

    def _build_kernels(self):
        """Build the vectorized GEMM kernels."""
        try:
            # Get kernel source
            kernel_path = Path(__file__).parent / "kernels" / "gemm_wave_vectorized.cl"
            with open(kernel_path, 'r') as f:
                kernel_source = f.read()

            # Define build options for vectorization
            build_options = [
                f"-DTILE_SIZE={self.config.tile_size}",
                f"-DWG_SIZE_X={self.config.wg_size_x}",
                f"-DWG_SIZE_Y={self.config.wg_size_y}",
                f"-DLDS_PADDING={self.config.lds_padding}",
                "-cl-mad-enable",  # Enable fused multiply-add
                "-cl-no-signed-zeros",  # Optimize for unsigned zeros
                "-cl-fast-relaxed-math",  # Use fast relaxed math
            ]

            # Build program
            self.program = cl.Program(self.context, kernel_source).build(options=build_options)

            # Get kernels
            self.kernels['vectorized'] = self.program.gemm_wave_vectorized
            self.kernels['vectorized_f8'] = self.program.gemm_wave_vectorized_f8

            logger.info("Vectorized kernels built successfully")

        except Exception as e:
            logger.error(f"Failed to build kernels: {e}")
            raise

    def gemm(self, A: np.ndarray, B: np.ndarray,
             alpha: float = 1.0, beta: float = 0.0,
             kernel_variant: str = 'vectorized') -> np.ndarray:
        """
        Execute vectorized GEMM: C = alpha * A @ B + beta * C

        Args:
            A: Input matrix A (M x K)
            B: Input matrix B (K x N)
            alpha: Scalar multiplier for A @ B
            beta: Scalar multiplier for C
            kernel_variant: Kernel to use ('vectorized' or 'vectorized_f8')

        Returns:
            Result matrix C (M x N)
        """
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Incompatible dimensions: A {A.shape} @ B {B.shape}")

        M, K = A.shape
        N = B.shape[1]

        # Initialize C if beta != 0, otherwise create zero matrix
        C = np.zeros((M, N), dtype=np.float32)
        if beta != 0.0:
            C = np.random.randn(M, N).astype(np.float32) * 0.1

        # Get appropriate kernel
        if kernel_variant not in self.kernels:
            raise ValueError(f"Unknown kernel variant: {kernel_variant}")

        kernel = self.kernels[kernel_variant]

        try:
            # Create OpenCL buffers
            A_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
            B_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
            C_buf = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=C)

            # Calculate workgroup dimensions
            wg_x = (M + self.config.tile_size - 1) // self.config.tile_size
            wg_y = (N + self.config.tile_size - 1) // self.config.tile_size

            # Set kernel arguments
            kernel.set_args(
                np.int32(M), np.int32(N), np.int32(K),
                np.float32(alpha), np.float32(beta),
                A_buf, B_buf, C_buf
            )

            # Execute kernel
            global_size = (wg_x * self.config.wg_size_x, wg_y * self.config.wg_size_y)
            local_size = (self.config.wg_size_x, self.config.wg_size_y)

            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size).wait()

            # Read result
            result = np.empty_like(C)
            cl.enqueue_copy(self.queue, result, C_buf).wait()

            return result

        except Exception as e:
            logger.error(f"Kernel execution failed: {e}")
            raise

    def benchmark(self, sizes: list, iterations: int = 10) -> dict:
        """
        Benchmark the vectorized kernel across different matrix sizes.

        Args:
            sizes: List of matrix sizes to test
            iterations: Number of iterations per size

        Returns:
            Dictionary with benchmark results
        """
        results = {}

        for size in sizes:
            logger.info(f"Benchmarking size {size}x{size}x{size}")

            # Generate test matrices
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            # Warmup
            _ = self.gemm(A, B)

            # Benchmark
            import time
            start_time = time.time()

            for _ in range(iterations):
                C = self.gemm(A, B)

            end_time = time.time()

            # Calculate performance
            total_time = end_time - start_time
            avg_time = total_time / iterations
            gflops = (2 * size**3) / (avg_time * 1e9)  # 2 operations per FMA

            results[size] = {
                'avg_time_ms': avg_time * 1000,
                'gflops': gflops,
                'iterations': iterations
            }

            logger.info(".2f")

        return results