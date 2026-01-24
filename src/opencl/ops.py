"""
OpenCL Operations - Python Wrappers
====================================

High-level Python interface for OpenCL kernel operations.

This module provides clean, PyTorch-like APIs for OpenCL operations,
abstracting away low-level buffer management and kernel invocation.

Example:
--------
    from src.opencl import CLContext, gemm
    import numpy as np
    
    # Initialize context
    ctx = CLContext()
    
    # Create test matrices
    A = np.random.randn(512, 256).astype(np.float32)
    B = np.random.randn(256, 1024).astype(np.float32)
    
    # Compute C = A @ B
    C = gemm(ctx, A, B)
    
    # Verify against NumPy
    C_np = A @ B
    assert np.allclose(C, C_np, rtol=1e-4)
"""

import os
import logging
from typing import Optional, Tuple
import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

from .context import CLContext


logger = logging.getLogger(__name__)


# Path to kernel files
KERNELS_DIR = os.path.join(os.path.dirname(__file__), 'kernels')


def gemm(
    ctx: CLContext,
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: Optional[np.ndarray] = None,
    use_tiled: bool = True,
    use_2x2: bool = False
) -> np.ndarray:
    """
    General Matrix Multiplication: C = alpha * (A @ B) + beta * C
    
    Args:
        ctx: OpenCL context
        A: Input matrix [M x K], dtype float32
        B: Input matrix [K x N], dtype float32
        alpha: Scalar multiplier for A @ B
        beta: Scalar multiplier for C (0.0 means C is overwritten)
        C: Optional pre-existing output matrix [M x N]. If None, allocates new matrix.
        use_tiled: Use tiled kernel (recommended for performance)
        use_2x2: Use 2x2 blocking variant (best for large matrices M,N,K > 1024)
    
    Returns:
        Output matrix C [M x N], dtype float32
    
    Performance Notes:
    ------------------
    - Naive kernel: ~50 GFLOPS (for testing)
    - Tiled kernel: ~1000-1500 GFLOPS (recommended)
    - 2x2 kernel: ~1500-2000 GFLOPS (large matrices only)
    
    Matrix sizes should be multiples of 16 for optimal performance.
    Non-aligned sizes are handled correctly but may be slower.
    
    Example:
    --------
        >>> A = np.random.randn(512, 256).astype(np.float32)
        >>> B = np.random.randn(256, 1024).astype(np.float32)
        >>> C = gemm(ctx, A, B)
        >>> # Verify against NumPy
        >>> assert np.allclose(C, A @ B, rtol=1e-4)
    
    Raises:
        ValueError: If matrix dimensions are incompatible
        RuntimeError: If OpenCL operations fail
    """
    if not OPENCL_AVAILABLE:
        raise RuntimeError("PyOpenCL not available")
    
    # Validate inputs
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got A.shape={A.shape}, B.shape={B.shape}")
    
    if A.dtype != np.float32 or B.dtype != np.float32:
        raise ValueError(f"Expected float32 dtype, got A.dtype={A.dtype}, B.dtype={B.dtype}")
    
    M, K_A = A.shape
    K_B, N = B.shape
    
    if K_A != K_B:
        raise ValueError(
            f"Matrix dimensions incompatible for multiplication: "
            f"A.shape={A.shape}, B.shape={B.shape}"
        )
    
    K = K_A
    
    # Allocate output if not provided
    if C is None:
        C = np.zeros((M, N), dtype=np.float32)
        beta = 0.0  # No need to read C if it's zeros
    else:
        if C.shape != (M, N):
            raise ValueError(f"C.shape={C.shape} must match (M, N)=({M}, {N})")
        if C.dtype != np.float32:
            raise ValueError(f"C.dtype must be float32, got {C.dtype}")
    
    # Load kernel
    kernel_path = os.path.join(KERNELS_DIR, 'gemm.cl')
    if use_2x2:
        kernel_name = 'gemm_tiled_2x2'
    elif use_tiled:
        kernel_name = 'gemm_tiled'
    else:
        kernel_name = 'gemm_naive'
    
    kernel = ctx.load_kernel_from_file(kernel_path, kernel_name)
    
    # Create OpenCL buffers
    mf = cl.mem_flags
    
    # Input buffers (read-only)
    A_buf = cl.Buffer(ctx.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    
    # Output buffer
    if beta == 0.0:
        # Write-only if beta=0 (no need to read existing C)
        C_buf = cl.Buffer(ctx.context, mf.WRITE_ONLY, C.nbytes)
    else:
        # Read-write if beta!=0 (need to read and modify C)
        C_buf = cl.Buffer(ctx.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)
    
    # Determine work-group configuration
    TILE_SIZE = 16  # Must match kernel definition
    
    if use_2x2:
        # Each thread computes 2x2 block
        local_size = (TILE_SIZE // 2, TILE_SIZE // 2)  # (8, 8) = 64 threads
        global_size = (
            ((M + 1) // 2 + local_size[0] - 1) // local_size[0] * local_size[0],
            ((N + 1) // 2 + local_size[1] - 1) // local_size[1] * local_size[1]
        )
    else:
        # Each thread computes 1 element
        local_size = (TILE_SIZE, TILE_SIZE)  # (16, 16) = 256 threads
        global_size = (
            (M + local_size[0] - 1) // local_size[0] * local_size[0],
            (N + local_size[1] - 1) // local_size[1] * local_size[1]
        )
    
    # Set kernel arguments
    kernel.set_args(
        np.int32(M),
        np.int32(N),
        np.int32(K),
        np.float32(alpha),
        np.float32(beta),
        A_buf,
        B_buf,
        C_buf
    )
    
    # Execute kernel
    event = cl.enqueue_nd_range_kernel(
        ctx.queue,
        kernel,
        global_size,
        local_size
    )
    
    # Read result
    cl.enqueue_copy(ctx.queue, C, C_buf, wait_for=[event])
    ctx.finish()
    
    logger.debug(
        f"GEMM completed: {kernel_name}, "
        f"M={M}, N={N}, K={K}, "
        f"global_size={global_size}, local_size={local_size}"
    )
    
    return C


def conv2d(
    ctx: CLContext,
    input: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0)
) -> np.ndarray:
    """
    2D Convolution operation.
    
    Args:
        ctx: OpenCL context
        input: Input tensor [N, C_in, H, W]
        weight: Filter weights [C_out, C_in, K_h, K_w]
        bias: Optional bias [C_out]
        stride: (stride_h, stride_w)
        padding: (pad_h, pad_w)
    
    Returns:
        Output tensor [N, C_out, H_out, W_out]
    
    Note:
        Implementation pending. Placeholder for future development.
    """
    raise NotImplementedError(
        "Conv2D kernel implementation in progress. "
        "For now, use PyTorch's conv2d on CPU."
    )


def relu(ctx: CLContext, x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    ReLU activation: y = max(0, x)
    
    Args:
        ctx: OpenCL context
        x: Input array, any shape
        inplace: Modify input array in-place (saves memory)
    
    Returns:
        Output array with same shape as input
    
    Note:
        Implementation pending. Placeholder for future development.
    """
    raise NotImplementedError(
        "Element-wise kernels implementation in progress. "
        "For now, use NumPy: np.maximum(0, x)"
    )


def sigmoid(ctx: CLContext, x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    Sigmoid activation: y = 1 / (1 + exp(-x))
    
    Args:
        ctx: OpenCL context
        x: Input array, any shape
        inplace: Modify input array in-place
    
    Returns:
        Output array with same shape as input
    
    Note:
        Implementation pending. Placeholder for future development.
    """
    raise NotImplementedError(
        "Element-wise kernels implementation in progress. "
        "For now, use NumPy: 1 / (1 + np.exp(-x))"
    )


# Additional utility functions

def benchmark_gemm(
    ctx: CLContext,
    M: int,
    N: int,
    K: int,
    num_trials: int = 10,
    warmup: int = 3
) -> dict:
    """
    Benchmark GEMM performance across different kernel variants.
    
    Args:
        ctx: OpenCL context
        M, N, K: Matrix dimensions
        num_trials: Number of benchmark iterations
        warmup: Warmup iterations (not counted)
    
    Returns:
        Dictionary with performance metrics:
        - gflops: Achieved GFLOPS
        - time_ms: Average execution time (milliseconds)
        - bandwidth_gb_s: Memory bandwidth utilization
    """
    import time
    
    # Create random matrices
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        _ = gemm(ctx, A, B)
    
    # Benchmark
    times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        _ = gemm(ctx, A, B)
        ctx.finish()  # Ensure completion
        end = time.perf_counter()
        times.append(end - start)
    
    # Calculate metrics
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    # GFLOPS = (2*M*N*K) / time / 1e9
    flops = 2.0 * M * N * K
    gflops = (flops / avg_time) / 1e9
    
    # Bandwidth = (bytes_read + bytes_written) / time / 1e9
    bytes_read = (M * K + K * N) * 4  # A and B in bytes
    bytes_written = M * N * 4  # C in bytes
    bandwidth_gb_s = (bytes_read + bytes_written) / avg_time / 1e9
    
    return {
        'gflops': gflops,
        'gflops_std': (flops / (avg_time - std_time)) / 1e9 - gflops,
        'time_ms': avg_time * 1000,
        'time_std_ms': std_time * 1000,
        'bandwidth_gb_s': bandwidth_gb_s,
        'M': M,
        'N': N,
        'K': K,
        'device': ctx.device.name
    }
