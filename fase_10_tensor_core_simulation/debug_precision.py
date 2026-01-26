#!/usr/bin/env python3
"""
Debug Tensor Core Precision Issues
===================================

Script to debug precision issues in tensor core implementation
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.tensor_core_emulator import TensorCoreEmulator
import numpy as np

def debug_precision():
    """Debug precision issues with small matrices"""
    print("🔍 DEBUGGING TENSOR CORE PRECISION")
    print("=" * 50)

    emulator = TensorCoreEmulator()

    # Use very small matrices for debugging
    size = 4
    print(f"Testing {size}x{size} matrix multiplication")

    # Create simple test matrices
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    A = np.tile(A, (2, 2))  # Make it 4x4
    B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    B = np.tile(B, (2, 2))  # Make it 4x4
    C = np.ones((4, 4), dtype=np.float32) * 0.5

    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print("\nMatrix C:")
    print(C)

    # Expected result: A @ B + C
    D_expected = A @ B + C
    print("\nExpected result (A @ B + C):")
    print(D_expected)

    # Tensor core result
    D_tensor, metrics = emulator.matmul(A, B, C, alpha=1.0, beta=1.0)
    print("\nTensor core result:")
    print(D_tensor)

    # Calculate error
    error = D_tensor - D_expected
    max_error = np.max(np.abs(error))
    print("\nError (Tensor - Expected):")
    print(error)
    print(f"\nMax absolute error: {max_error:.2e}")

    # Check if matrices are close
    close = np.allclose(D_tensor, D_expected, rtol=1e-5, atol=1e-5)
    print(f"Are results close? {close}")

    return max_error < 1e-5

if __name__ == "__main__":
    success = debug_precision()
    if success:
        print("\n✅ PRECISION OK")
    else:
        print("\n❌ PRECISION ISSUES")