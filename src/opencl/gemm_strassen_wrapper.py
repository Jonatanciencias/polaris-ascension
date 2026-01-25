#!/usr/bin/env python3
"""
Strassen Matrix Multiplication - Python Wrapper

Phase 2, Technique 4: Advanced Algorithm Research
Target: Evaluate O(n^2.807) vs practical GPU performance

Implements complete Strassen algorithm with hybrid approach:
- Recursive Strassen for large blocks
- Tiled GEMM for base case
- Memory-efficient implementation

Expected performance: 300-400 GFLOPS for large matrices
Crossover point: n > 512-1024 theoretically
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    logger.warning("PyOpenCL not available. Running in simulation mode.")


@dataclass
class StrassenConfig:
    """Configuration for Strassen kernels."""
    base_case_size: int = 128  # Size below which to use tiled GEMM
    tile_size: int = 64        # Tile size for base case GEMM
    kernel_variant: str = "complete"  # "complete", "simple"

    def __post_init__(self):
        assert self.base_case_size > 0, "base_case_size must be positive"
        assert self.tile_size > 0, "tile_size must be positive"
        assert self.kernel_variant in ["complete", "simple"], \
            f"Unknown kernel variant: {self.kernel_variant}"


class StrassenKernelManager:
    """Manages OpenCL context and Strassen kernel compilation."""

    def __init__(self, config: StrassenConfig):
        self.config = config
        self.context = None
        self.queue = None
        self.program = None
        self.kernels = {}

        if PYOPENCL_AVAILABLE:
            self._create_context()
            self._compile_kernels()

    def _create_context(self):
        """Create OpenCL context and command queue."""
        try:
            platforms = cl.get_platforms()
            gpu_platform = None
            for platform in platforms:
                if 'AMD' in platform.name.upper():
                    gpu_platform = platform
                    break

            if gpu_platform is None:
                gpu_platform = platforms[0]  # Fallback

            devices = gpu_platform.get_devices(device_type=cl.device_type.GPU)
            if not devices:
                devices = gpu_platform.get_devices()  # Fallback to any device

            self.context = cl.Context(devices=[devices[0]])
            self.queue = cl.CommandQueue(self.context)

            logger.info(f"Using device: {devices[0].name}")

        except Exception as e:
            logger.error(f"Failed to create OpenCL context: {e}")
            raise

    def _compile_kernels(self):
        """Compile Strassen kernels."""
        try:
            kernel_path = Path(__file__).parent / 'kernels' / 'gemm_strassen.cl'
            with open(kernel_path, 'r') as f:
                kernel_source = f.read()

            # Compile with optimization flags
            compile_options = [
                "-cl-mad-enable",
                "-cl-unsafe-math-optimizations",
                "-cl-fast-relaxed-math",
                f"-DBASE_CASE_SIZE={self.config.base_case_size}",
                f"-DSTRASSEN_TILE_SIZE={self.config.tile_size}"
            ]

            logger.info(f"Compiling Strassen kernels from: {kernel_path}")
            logger.info(f"Compiler options: {' '.join(compile_options)}")

            self.program = cl.Program(self.context, kernel_source).build(options=compile_options)

            # Cache kernel objects
            self.kernels = {
                'complete': self.program.gemm_strassen_complete,
                'simple': self.program.gemm_strassen_simple
            }

            logger.info(f"Successfully compiled {len(self.kernels)} Strassen kernel variants")

        except Exception as e:
            logger.error(f"Kernel compilation failed: {e}")
            raise

    def select_kernel(self) -> cl.Kernel:
        """Select Strassen kernel variant."""
        return self.kernels[self.config.kernel_variant]


class StrassenGEMMExecutor:
    """Executes Strassen GEMM with GPU."""

    def __init__(self, config: Optional[StrassenConfig] = None):
        self.config = config or StrassenConfig()
        self.manager = StrassenKernelManager(self.config) if PYOPENCL_AVAILABLE else None

    def gemm(self,
             A: np.ndarray,
             B: np.ndarray,
             alpha: float = 1.0,
             beta: float = 0.0,
             C: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Execute Strassen GEMM: C = alpha * A @ B + beta * C

        Args:
            A: Input matrix (M × K), float32
            B: Input matrix (K × N), float32
            alpha: Scalar multiplier for A @ B
            beta: Scalar multiplier for C
            C: Optional input/output matrix (M × N)

        Returns:
            Result matrix C (M × N)
        """
        # Input validation
        assert A.dtype == np.float32, "A must be float32"
        assert B.dtype == np.float32, "B must be float32"
        assert A.ndim == 2 and B.ndim == 2, "Inputs must be 2D"
        assert A.shape[1] == B.shape[0], f"Dimension mismatch: {A.shape} @ {B.shape}"

        M, K = A.shape
        K2, N = B.shape

        # Initialize C if not provided
        if C is None:
            C = np.zeros((M, N), dtype=np.float32)
        else:
            assert C.shape == (M, N), f"C shape mismatch: expected {(M, N)}, got {C.shape}"
            assert C.dtype == np.float32, "C must be float32"

        if not PYOPENCL_AVAILABLE or self.manager is None:
            # Fallback to NumPy
            logger.warning("Running Strassen in simulation mode (NumPy)")
            result = alpha * (A @ B)
            if beta != 0.0:
                result += beta * C
            return result.astype(np.float32)

        try:
            return self._execute_gpu(A, B, C, alpha, beta)
        except Exception as e:
            logger.error(f"GPU execution failed: {e}")
            raise

    def _execute_gpu(self, A: np.ndarray, B: np.ndarray, C: np.ndarray,
                     alpha: float, beta: float) -> np.ndarray:
        """Execute on GPU."""
        M, K = A.shape
        N = B.shape[1]

        # Create buffers
        A_buf = cl.Buffer(self.manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(self.manager.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(self.manager.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=C)

        # Select kernel
        kernel = self.manager.select_kernel()

        # Set kernel arguments
        kernel.set_args(
            np.int32(M), np.int32(N), np.int32(K),
            np.float32(alpha), np.float32(beta),
            A_buf, B_buf, C_buf
        )

        # For the simple version, use 2x2 blocking
        if self.config.kernel_variant == "simple":
            global_size = ((M + 1) // 2, (N + 1) // 2)  # Each work item handles 2x2 block
            local_size = (16, 16)
        else:
            # For complete version, use different work size
            # This would need adjustment based on the actual recursive implementation
            global_size = (M, N)
            local_size = None  # Let OpenCL decide

        # Execute kernel
        if local_size:
            cl.enqueue_nd_range_kernel(self.manager.queue, kernel, global_size, local_size)
        else:
            cl.enqueue_nd_range_kernel(self.manager.queue, kernel, global_size)

        self.manager.queue.finish()

        # Read result
        result = np.empty_like(C)
        cl.enqueue_copy(self.manager.queue, result, C_buf)

        return result