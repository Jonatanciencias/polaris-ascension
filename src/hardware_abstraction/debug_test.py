#!/usr/bin/env python3
"""
Debug script for testing kernel loading
"""

import os
import sys

import numpy as np
import pyopencl as cl

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    # Initialize OpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Load debug kernel
    with open("debug_kernel.cl", "r") as f:
        source = f.read()

    program = cl.Program(ctx, source).build()
    kernel = getattr(program, "gemm_debug_copy")

    # Test with small matrices
    M = N = K = 8
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    # OpenCL buffers
    A_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, C.nbytes)

    # Execute
    TILE_SIZE = 8
    global_size = (M, N)
    local_size = (TILE_SIZE, TILE_SIZE)

    kernel(
        queue,
        global_size,
        local_size,
        np.int32(M),
        np.int32(N),
        np.int32(K),
        np.float32(1.0),
        np.float32(0.0),
        A_buf,
        B_buf,
        C_buf,
    )

    # Copy result back
    cl.enqueue_copy(queue, C, C_buf).wait()

    print("A (first 8x8):")
    print(A[:8, :8])
    print("\nC (result):")
    print(C)
    print("\nAre they equal?", np.allclose(A[:8, :8], C))


if __name__ == "__main__":
    main()
