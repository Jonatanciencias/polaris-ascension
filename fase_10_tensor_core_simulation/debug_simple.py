#!/usr/bin/env python3
"""
Debug Tensor Core - Simple Test
===============================

Test muy simple para aislar el problema de precisión en Tensor Core.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.tensor_core_emulator import TensorCoreEmulator
import numpy as np

def debug_tensor_core():
    """Debug muy simple del tensor core"""
    print("🔧 DEBUG TENSOR CORE")
    print("=" * 40)

    emulator = TensorCoreEmulator()

    # Test con matrices muy pequeñas para fácil debugging
    size = 4
    print(f"Testing {size}x{size} matrices")

    # Matrices simples y predecibles - CAMBIADO A ALEATORIAS
    np.random.seed(42)  # Para reproducibilidad
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    C = np.random.randn(size, size).astype(np.float32)

    print("A =")
    print(A)
    print("B =")
    print(B)
    print("C =")
    print(C)

    # Resultado esperado: A @ B + C
    D_expected = A @ B + C
    print("Expected D = A @ B + C =")
    print(D_expected)

    # Resultado del tensor core
    D_tensor, metrics = emulator.matmul(A, B, C, alpha=1.0, beta=1.0)
    print("Tensor Core D =")
    print(D_tensor)

    # Comparación
    diff = D_tensor - D_expected
    print("Difference =")
    print(diff)

    max_error = np.max(np.abs(diff))
    print(f"Max error: {max_error}")

    # Verificar si el error es consistente
    if max_error < 1e-5:
        print("✅ PRECISIÓN EXCELENTE")
    elif max_error < 1e-3:
        print("✅ PRECISIÓN BUENA")
    elif max_error < 1e-1:
        print("⚠️ PRECISIÓN ACEPTABLE")
    else:
        print("❌ ERROR SIGNIFICATIVO")

    return max_error

if __name__ == "__main__":
    error = debug_tensor_core()
    print(f"\nFinal max error: {error}")
    if error > 1e-3:
        print("❌ DEBUGGING REQUIRED")
        sys.exit(1)
    else:
        print("✅ DEBUGGING SUCCESSFUL")
        sys.exit(0)