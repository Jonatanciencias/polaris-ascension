#!/usr/bin/env python3
"""
Performance Test - Check GFLOPS Calculation
==========================================

Test para verificar el cálculo correcto de GFLOPS.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.tensor_core_emulator import TensorCoreEmulator
import numpy as np
import time

def test_gflops_calculation():
    """Test GFLOPS calculation accuracy"""
    print("🔬 TESTING GFLOPS CALCULATION")
    print("=" * 50)

    emulator = TensorCoreEmulator()

    # Test with known size
    size = 512
    print(f"Testing {size}x{size} matrix multiplication")

    # Create test matrices
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    C = np.random.randn(size, size).astype(np.float32)

    # Expected operations: 2 * M * N * K multiply-add operations
    M, K = A.shape
    K2, N = B.shape
    expected_operations = 2 * M * N * K
    print(f"Expected operations: {expected_operations:,}")

    # Run tensor core operation
    start_total = time.time()
    D_tensor, metrics = emulator.matmul(A, B, C, alpha=1.0, beta=1.0)
    end_total = time.time()

    total_time = end_total - start_total
    kernel_time = metrics.kernel_time_ms / 1000  # Convert to seconds

    # Calculate GFLOPS using kernel time (computation only)
    gflops_kernel = expected_operations / (kernel_time * 1e9)
    # Calculate GFLOPS using total time (including memory transfers)
    gflops_total = expected_operations / (total_time * 1e9)

    print("\n📊 PERFORMANCE ANALYSIS:")
    print(f"Kernel time: {kernel_time:.6f} seconds")
    print(f"Total time: {total_time:.6f} seconds")
    print(f"GFLOPS (kernel only): {gflops_kernel:.2f}")
    print(f"GFLOPS (total time): {gflops_total:.2f}")

    # Theoretical peak for RX 580 (GCN 4.0)
    theoretical_peak = 4608  # GFLOPS
    efficiency_kernel = (gflops_kernel / theoretical_peak) * 100

    print("\n🎯 EFFICIENCY ANALYSIS:")
    print(f"Theoretical peak: {theoretical_peak} GFLOPS")
    print(f"Efficiency (kernel): {efficiency_kernel:.1f}%")

    # Check precision
    D_expected = A @ B + C
    max_error = np.max(np.abs(D_tensor - D_expected))
    print("\n🎯 PRECISION:")
    print(f"Max error: {max_error:.2e}")

    if max_error < 1e-3:
        print("✅ EXCELLENT PRECISION")
    else:
        print("❌ PRECISION ISSUES")

    return {
        'gflops_kernel': gflops_kernel,
        'efficiency_kernel': efficiency_kernel,
        'max_error': max_error
    }

if __name__ == "__main__":
    results = test_gflops_calculation()

    print("\n🏆 SUMMARY:")
    print(f"GFLOPS: {results['gflops_kernel']:.2f}")
    print(f"Efficiency: {results['efficiency_kernel']:.1f}%")
    print(f"Max error: {results['max_error']:.2e}")

    if results['max_error'] < 1e-3:
        print("✅ TENSOR CORE WORKING CORRECTLY")
    else:
        print("❌ PRECISION ISSUES DETECTED")
        print("⚠️ REQUIRES OPTIMIZATION")