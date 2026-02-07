"""
OpenCL compatibility facade for GEMM operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..optimization_engines.optimized_kernel_engine import KernelType, OptimizedKernelEngine


@dataclass
class CLContext:
    """Thin context wrapper around the optimized OpenCL engine."""

    engine: OptimizedKernelEngine

    def __init__(self) -> None:
        self.engine = OptimizedKernelEngine()


def _validate_inputs(
    A: np.ndarray,
    B: np.ndarray,
    C: Optional[np.ndarray],
) -> None:
    if A.dtype != np.float32 or B.dtype != np.float32:
        raise ValueError("A and B must be float32 arrays")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2D matrices")
    if A.shape[1] != B.shape[0]:
        raise ValueError("incompatible matrix dimensions for GEMM")
    if C is not None:
        if C.dtype != np.float32:
            raise ValueError("C must be float32")
        if C.shape != (A.shape[0], B.shape[1]):
            raise ValueError("C shape must match output matrix dimensions")


def gemm(
    cl_context: CLContext,
    A: np.ndarray,
    B: np.ndarray,
    *,
    C: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    use_tiled: bool = True,
    use_2x2: bool = False,
) -> np.ndarray:
    """
    GEMM compatibility function.

    Args:
        cl_context: OpenCL context wrapper.
        A: Left matrix (M, K), float32.
        B: Right matrix (K, N), float32.
        C: Optional accumulation matrix (M, N), float32.
        alpha: Scale for A @ B.
        beta: Scale for C.
        use_tiled: Select tiled kernel mode (legacy option).
        use_2x2: Select 2x2 variant (legacy option).
    """
    _validate_inputs(A, B, C)

    kernel_type = None
    if use_2x2:
        kernel_type = KernelType.GEMM_BASIC
    elif not use_tiled:
        kernel_type = KernelType.GEMM_BASIC

    # Compute raw GEMM first and apply BLAS alpha/beta semantics explicitly.
    # This avoids inconsistencies between kernel variants.
    result = cl_context.engine.gemm(
        A,
        B,
        kernel_type=kernel_type,
    )
    output = result.result

    # Enforce BLAS-like alpha/beta semantics consistently, even if selected
    # kernel path internally ignores alpha/beta in some variants.
    if alpha == 1.0 and beta == 0.0:
        return output

    adjusted = float(alpha) * output
    if C is not None and beta != 0.0:
        adjusted = adjusted + float(beta) * C
    return adjusted.astype(np.float32, copy=False)


__all__ = ["CLContext", "gemm"]
