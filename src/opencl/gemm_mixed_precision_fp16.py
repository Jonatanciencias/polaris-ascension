"""
Mixed Precision FP16 GEMM Implementation
Python wrapper for FP16 compute + FP32 accumulate kernel

Target: +15-20% performance improvement over SIMD vectorization
Strategy: Leverage Polaris 10 FP16 units for 2x compute throughput
"""

import numpy as np
import pyopencl as cl
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Check PyOpenCL availability
try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    logger.warning("PyOpenCL not available - FP16 kernel will not function")

class MixedPrecisionConfig:
    """Configuration for Mixed Precision FP16 kernels."""

    def __init__(self,
                 tile_size: int = 16,
                 workgroup_size: Tuple[int, int] = (16, 16),
                 lds_padding: int = 2):
        self.tile_size = tile_size
        self.workgroup_size = workgroup_size
        self.lds_padding = lds_padding

    def get_compiler_options(self) -> str:
        """Get OpenCL compiler options for FP16 kernel."""
        return (f"-cl-mad-enable "
                f"-cl-unsafe-math-optimizations "
                f"-cl-fast-relaxed-math "
                f"-DTILE_SIZE={self.tile_size} "
                f"-DWG_SIZE_X={self.workgroup_size[0]} "
                f"-DWG_SIZE_Y={self.workgroup_size[1]} "
                f"-DLDS_PADDING={self.lds_padding}")

class MixedPrecisionKernelManager:
    """Manages OpenCL context and Mixed Precision kernel compilation."""

    def __init__(self, config: MixedPrecisionConfig):
        self.config = config
        self.context: Optional[cl.Context] = None
        self.queue: Optional[cl.CommandQueue] = None
        self.program: Optional[cl.Program] = None
        self.kernels: Dict[str, cl.Kernel] = {}

        if PYOPENCL_AVAILABLE:
            self._initialize_opencl()

    def _initialize_opencl(self):
        """Initialize OpenCL context and device."""
        try:
            # Get platform and device
            platforms = cl.get_platforms()
            amd_platform = next((p for p in platforms if 'AMD' in p.name), platforms[0])

            devices = amd_platform.get_devices(device_type=cl.device_type.GPU)
            if not devices:
                raise RuntimeError("No GPU devices found")

            self.device = devices[0]
            logger.info(f"Using device: {self.device.name}")

            # Create context and queue
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context,
                                        properties=cl.command_queue_properties.PROFILING_ENABLE)

        except Exception as e:
            logger.error(f"OpenCL initialization failed: {e}")
            raise

    def compile_kernels(self):
        """Compile Mixed Precision kernels."""
        if not PYOPENCL_AVAILABLE or self.context is None:
            logger.warning("OpenCL not available - skipping kernel compilation")
            return

        try:
            kernel_path = Path(__file__).parent / 'kernels' / 'gemm_mixed_precision_fp16.cl'
            with open(kernel_path, 'r') as f:
                kernel_source = f.read()

            compiler_options = self.config.get_compiler_options()
            logger.info(f"Compiling Mixed Precision kernels from: {kernel_path}")
            logger.info(f"Compiler options: {compiler_options}")

            # Compile program
            self.program = cl.Program(self.context, kernel_source).build(options=compiler_options)

            # Get kernels
            self.kernels = {
                'mixed_precision_fp16': self.program.gemm_mixed_precision_fp16,
            }

            logger.info(f"Successfully compiled {len(self.kernels)} Mixed Precision kernel variants")

        except Exception as e:
            logger.error(f"Kernel compilation failed: {e}")
            raise

class MixedPrecisionGEMMExecutor:
    """Executes Mixed Precision GEMM with GPU."""

    def __init__(self, config: Optional[MixedPrecisionConfig] = None):
        self.config = config or MixedPrecisionConfig()
        self.manager = MixedPrecisionKernelManager(self.config) if PYOPENCL_AVAILABLE else None

        # Compile kernels on initialization
        if self.manager is not None:
            self.manager.compile_kernels()

    def gemm(self, A: np.ndarray, B: np.ndarray,
             C: Optional[np.ndarray] = None,
             alpha: float = 1.0, beta: float = 0.0) -> Tuple[np.ndarray, float]:
        """
        Execute Mixed Precision GEMM: C = alpha * A @ B + beta * C

        Args:
            A: Input matrix A (M x K)
            B: Input matrix B (K x N)
            C: Output matrix C (M x N), created if None
            alpha: Scaling factor for A @ B
            beta: Scaling factor for C

        Returns:
            Tuple of (result matrix, execution time in ms)
        """
        if not PYOPENCL_AVAILABLE or self.manager is None:
            # Fallback to NumPy
            start_time = time.time()
            if C is None:
                C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
            C = alpha * (A @ B) + beta * C
            end_time = time.time()
            return C, (end_time - start_time) * 1000

        return self._execute_gpu(A, B, C, alpha, beta)

    def _execute_gpu(self, A: np.ndarray, B: np.ndarray,
                    C: Optional[np.ndarray], alpha: float, beta: float) -> Tuple[np.ndarray, float]:
        """Execute GEMM on GPU."""
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Matrix dimension mismatch: A {A.shape}, B {B.shape}"

        if C is None:
            C = np.zeros((M, N), dtype=np.float32)
        else:
            assert C.shape == (M, N), f"C shape mismatch: expected {(M, N)}, got {C.shape}"

        # Create OpenCL buffers
        mf = cl.mem_flags
        A_buf = cl.Buffer(self.manager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A.astype(np.float32))
        B_buf = cl.Buffer(self.manager.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B.astype(np.float32))
        C_buf = cl.Buffer(self.manager.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)

        # Kernel arguments
        kernel = self.manager.kernels['mixed_precision_fp16']
        kernel.set_args(
            np.int32(M), np.int32(N), np.int32(K),
            np.float32(alpha), np.float32(beta),
            A_buf, B_buf, C_buf
        )

        # Work group sizing
        wg_x, wg_y = self.config.workgroup_size
        global_x = ((N + wg_x - 1) // wg_x) * wg_x
        global_y = ((M + wg_y - 1) // wg_y) * wg_y
        global_size = (global_x, global_y)
        local_size = (wg_x, wg_y)

        # Execute kernel
        event = cl.enqueue_nd_range_kernel(self.manager.queue, kernel, global_size, local_size)
        event.wait()

        # Get execution time
        exec_time_ms = (event.profile.end - event.profile.start) * 1e-6  # Convert to milliseconds

        # Read result
        result = np.empty_like(C)
        cl.enqueue_copy(self.manager.queue, result, C_buf).wait()

        return result, exec_time_ms

    def benchmark(self, sizes: list, num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark Mixed Precision kernel across different matrix sizes."""
        results = {}

        for size in sizes:
            logger.info(f"Benchmarking Mixed Precision FP16, size {size}×{size}")

            # Create test matrices
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            times = []
            for _ in range(num_runs):
                _, exec_time = self.gemm(A, B)
                times.append(exec_time)

            avg_time = np.mean(times)
            gflops = (2 * size**3) / (avg_time * 1e-3) / 1e9  # GFLOPS

            results[f"{size}x{size}x{size}"] = {
                'avg_time_ms': avg_time,
                'min_time_ms': np.min(times),
                'max_time_ms': np.max(times),
                'std_time_ms': np.std(times),
                'gflops': gflops,
                'matrix_size': f"{size}x{size}x{size}",
                'kernel': 'gemm_mixed_precision_fp16'
            }

        return results