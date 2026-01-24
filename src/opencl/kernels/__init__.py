"""
OpenCL Kernels Package
======================

Contains optimized OpenCL kernel implementations for core ML operations.

Kernels are written in OpenCL C and optimized for AMD Polaris architecture (GCN 4.0).

Available Kernels:
------------------
- gemm.cl: General Matrix Multiplication (GEMM) with tiling
- conv2d.cl: 2D Convolution operations
- elementwise.cl: Element-wise operations (ReLU, Sigmoid, Tanh, etc.)
- pooling.cl: Pooling operations (Max, Average)

Performance Notes:
------------------
For AMD Polaris (RX 580):
- Compute Units: 36
- Max Work Group Size: 256
- Local Memory: 32 KB per CU
- Wavefront Size: 64 (optimal for coalescing)

Optimization Guidelines:
------------------------
1. Use local memory for tile-based algorithms
2. Ensure coalesced global memory access (stride of 64 elements)
3. Vectorize operations when possible (float4, float8)
4. Balance work-group size with occupancy
5. Minimize barriers and divergence
"""

__all__ = [
    "GEMM_KERNEL",
    "CONV2D_KERNEL", 
    "ELEMENTWISE_KERNEL",
    "POOLING_KERNEL",
]


# Kernel source code is loaded from .cl files at runtime
# This approach allows easy editing and debugging of kernel code
