"""
OpenCL Acceleration Module
===========================

Custom OpenCL kernels for AMD Polaris GPUs.

This module provides optimized OpenCL implementations for core ML operations,
enabling hardware independence and full control over GPU execution.

Philosophy:
-----------
- Hardware Independence: Not tied to vendor-specific frameworks
- Educational: Clear, documented kernel code
- Portable: Works across OpenCL-capable devices
- Sustainable: Long-term maintainability (10+ years)

Modules:
--------
- context: OpenCL context and queue management
- ops: High-level Python operations (GEMM, Conv2D, etc.)
- kernels: Low-level OpenCL kernel implementations

Example:
--------
    from src.opencl import CLContext, gemm
    
    ctx = CLContext()
    C = gemm(ctx, A, B, alpha=1.0, beta=0.0)
"""

from .context import CLContext, CLDevice
from .ops import gemm, conv2d, relu, sigmoid

__version__ = "0.1.0"
__all__ = [
    "CLContext",
    "CLDevice",
    "gemm",
    "conv2d", 
    "relu",
    "sigmoid",
]
