"""
Hybrid GEMM Kernel Manager and Wrapper

Provides high-level interface for the hybrid float4+2x2 GEMM kernel.

Classes:
    HybridGEMMConfig: Configuration container for kernel parameters
    HybridGEMMKernel: Kernel wrapper with memory management
    HybridGEMMExecutor: High-level execution interface

Example:
    >>> config = HybridGEMMConfig(tile_size=16, block_size=2)
    >>> kernel = HybridGEMMKernel(context, config)
    >>> result = kernel.gemm(A, B, alpha=1.0, beta=0.0)
"""

import logging
import numpy as np
import pyopencl as cl
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class GEMMKernelVariant(Enum):
    """Available kernel variants."""
    GENERAL = "gemm_hybrid_float4_2x2_v1"
    BETA_ZERO = "gemm_hybrid_float4_2x2_beta_zero"


@dataclass
class HybridGEMMConfig:
    """
    Configuration for Hybrid GEMM Kernel.
    
    Attributes:
        tile_size: Size of local memory tiles (default 16)
        block_size: Register blocking size (default 2, meaning 2×2)
        lds_padding: Padding to avoid bank conflicts in LDS (default 4)
        enable_async_prefetch: Enable double buffering (default True)
        enable_beta_zero_variant: Use optimized kernel when beta=0 (default True)
    """
    tile_size: int = 16
    block_size: int = 2
    lds_padding: int = 4
    enable_async_prefetch: bool = True
    enable_beta_zero_variant: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.tile_size not in [8, 12, 16, 20, 24]:
            logger.warning(f"Unusual tile_size={self.tile_size}, expected power of 2")
        
        if self.block_size > self.tile_size:
            raise ValueError(f"block_size ({self.block_size}) > tile_size ({self.tile_size})")
        
        if self.tile_size % (self.block_size * 2) != 0:
            logger.warning(f"tile_size should be divisible by block_size*2 for optimal warps")
    
    def get_global_size(self, M: int, N: int) -> Tuple[int, int]:
        """Calculate global work size for given problem dimensions."""
        return (
            (M + self.tile_size - 1) // self.tile_size,
            (N + self.tile_size - 1) // self.tile_size
        )
    
    def get_local_size(self) -> Tuple[int, int]:
        """Calculate local work size (workgroup size)."""
        threads_per_dim = self.tile_size // self.block_size
        return (threads_per_dim, threads_per_dim)
    
    def get_lds_bytes(self) -> int:
        """Calculate LDS memory required in bytes."""
        # 2 buffers × 2 matrices × tile_size × (tile_size + padding) × 4 bytes
        return 2 * 2 * self.tile_size * (self.tile_size + self.lds_padding) * 4
    
    def get_compile_options(self) -> list:
        """Generate compiler options for kernel."""
        return [
            f"-DTILE_SIZE={self.tile_size}",
            f"-DBLOCK_SIZE={self.block_size}",
            f"-DLDS_PADDING={self.lds_padding}",
        ]


class HybridGEMMKernel:
    """
    Manages compilation and execution of hybrid GEMM kernels.
    
    Handles:
    - Kernel compilation with proper options
    - Memory buffer management
    - Kernel launching with correct work configurations
    - Result validation
    """
    
    def __init__(self, context: cl.Context, config: Optional[HybridGEMMConfig] = None):
        """
        Initialize kernel wrapper.
        
        Args:
            context: PyOpenCL context
            config: HybridGEMMConfig instance (uses defaults if None)
        """
        self.context = context
        self.config = config or HybridGEMMConfig()
        self.program = None
        self.kernels = {}
        
        self._compile_kernels()
        logger.info(f"HybridGEMMKernel initialized with config: {self.config}")
    
    def _compile_kernels(self):
        """Compile OpenCL kernels from source file."""
        kernel_dir = Path(__file__).parent
        kernel_file = kernel_dir / "gemm_hybrid.cl"
        
        if not kernel_file.exists():
            raise FileNotFoundError(f"Kernel source not found: {kernel_file}")
        
        with open(kernel_file, 'r') as f:
            kernel_source = f.read()
        
        try:
            compile_options = self.config.get_compile_options()
            self.program = cl.Program(
                self.context, kernel_source
            ).build(options=compile_options)
            
            # Cache kernel objects
            self.kernels[GEMMKernelVariant.GENERAL] = self.program.gemm_hybrid_float4_2x2_v1
            self.kernels[GEMMKernelVariant.BETA_ZERO] = self.program.gemm_hybrid_float4_2x2_beta_zero
            
            logger.info("Kernel compilation successful")
            
        except cl.ProgramBuildError as e:
            logger.error(f"Kernel compilation failed:\n{e.build_log}")
            raise
    
    def get_kernel(self, variant: GEMMKernelVariant) -> cl.Kernel:
        """Get compiled kernel object."""
        if variant not in self.kernels:
            raise ValueError(f"Unknown kernel variant: {variant}")
        return self.kernels[variant]
    
    def select_variant(self, beta: float) -> GEMMKernelVariant:
        """Select appropriate kernel variant based on parameters."""
        if self.config.enable_beta_zero_variant and abs(beta) < 1e-10:
            return GEMMKernelVariant.BETA_ZERO
        return GEMMKernelVariant.GENERAL
    
    def gemm(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        queue: Optional[cl.CommandQueue] = None
    ) -> np.ndarray:
        """
        Execute GEMM operation: C = alpha*A*B + beta*C
        
        Args:
            A: Input matrix (M×K), float32, C-contiguous
            B: Input matrix (K×N), float32, C-contiguous
            C: Output matrix (M×N), float32, C-contiguous. If None, initialized to zeros
            alpha: Scalar multiplier for A*B
            beta: Scalar multiplier for C (ignored if beta=0 variant selected)
            queue: PyOpenCL command queue (creates one if None)
        
        Returns:
            Output matrix C with result of multiplication
        
        Raises:
            ValueError: If input dimensions don't match or matrices not C-contiguous
            RuntimeError: If kernel execution fails
        """
        # Validate inputs
        self._validate_inputs(A, B, C)
        
        # Prepare output matrix
        if C is None:
            C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
        else:
            C = C.copy()  # Don't modify input
        
        M, K = A.shape
        K2, N = B.shape
        
        if queue is None:
            queue = cl.CommandQueue(self.context)
        
        # Create GPU buffers
        A_buf = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=A
        )
        B_buf = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=B
        )
        C_buf = cl.Buffer(
            self.context,
            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=C
        )
        
        try:
            # Select appropriate kernel variant
            variant = self.select_variant(beta)
            kernel = self.get_kernel(variant)
            
            # Calculate work sizes
            global_size = self.config.get_global_size(M, N)
            local_size = self.config.get_local_size()
            
            logger.debug(f"Launching {variant.value} with global={global_size}, local={local_size}")
            
            # Launch kernel
            if variant == GEMMKernelVariant.BETA_ZERO:
                event = kernel(
                    queue, global_size, local_size,
                    A_buf, B_buf, C_buf,
                    np.int32(M), np.int32(N), np.int32(K),
                    np.float32(alpha)
                )
            else:
                event = kernel(
                    queue, global_size, local_size,
                    A_buf, B_buf, C_buf,
                    np.int32(M), np.int32(N), np.int32(K),
                    np.float32(alpha), np.float32(beta)
                )
            
            # Wait for completion
            event.wait()
            
            # Copy result back to host
            cl.enqueue_copy(queue, C, C_buf).wait()
            
            logger.debug("GEMM execution completed successfully")
            
            return C
            
        except cl.RuntimeError as e:
            logger.error(f"Kernel execution failed: {e}")
            raise RuntimeError(f"GEMM kernel failed: {e}")
        
        finally:
            # Clean up GPU buffers
            A_buf.release()
            B_buf.release()
            C_buf.release()
    
    @staticmethod
    def _validate_inputs(A: np.ndarray, B: np.ndarray, C: Optional[np.ndarray]):
        """Validate input matrices."""
        # Check data type
        if A.dtype != np.float32:
            raise ValueError(f"A must be float32, got {A.dtype}")
        if B.dtype != np.float32:
            raise ValueError(f"B must be float32, got {B.dtype}")
        if C is not None and C.dtype != np.float32:
            raise ValueError(f"C must be float32, got {C.dtype}")
        
        # Check dimensions
        M, K = A.shape
        K2, N = B.shape
        
        if K != K2:
            raise ValueError(f"Dimension mismatch: A.shape[1]={K} != B.shape[0]={K2}")
        
        if C is not None:
            if C.shape != (M, N):
                raise ValueError(f"C shape {C.shape} doesn't match expected ({M}, {N})")
        
        # Check memory layout
        if not A.flags['C_CONTIGUOUS']:
            raise ValueError("A must be C-contiguous")
        if not B.flags['C_CONTIGUOUS']:
            raise ValueError("B must be C-contiguous")
        if C is not None and not C.flags['C_CONTIGUOUS']:
            raise ValueError("C must be C-contiguous")


class HybridGEMMExecutor:
    """
    High-level interface for hybrid GEMM operations.
    
    Provides convenience methods and automatic optimization selection.
    """
    
    def __init__(
        self,
        context: Optional[cl.Context] = None,
        config: Optional[HybridGEMMConfig] = None,
        queue: Optional[cl.CommandQueue] = None
    ):
        """
        Initialize executor.
        
        Args:
            context: PyOpenCL context (creates one if None)
            config: HybridGEMMConfig instance
            queue: PyOpenCL command queue (creates one if None)
        """
        if context is None:
            context = cl.create_some_context()
        
        self.context = context
        self.queue = queue or cl.CommandQueue(self.context)
        self.kernel = HybridGEMMKernel(context, config)
        self.config = config or HybridGEMMConfig()
    
    def __call__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        beta: float = 0.0
    ) -> np.ndarray:
        """Execute GEMM: C = alpha*A*B + beta*C"""
        return self.kernel.gemm(A, B, C, alpha, beta, self.queue)
    
    def gemm_batched(
        self,
        A_list: list,
        B_list: list,
        alpha: float = 1.0,
        beta: float = 0.0
    ) -> list:
        """Execute multiple GEMM operations."""
        return [
            self.kernel.gemm(A, B, alpha=alpha, beta=beta, queue=self.queue)
            for A, B in zip(A_list, B_list)
        ]


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.DEBUG)
    
    executor = HybridGEMMExecutor()
    
    A = np.random.randn(256, 256).astype(np.float32)
    B = np.random.randn(256, 256).astype(np.float32)
    
    C = executor(A, B)
    C_ref = A @ B
    
    error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)
    print(f"Relative error: {error:.2e}")
