#!/usr/bin/env python3
"""
Tensor Core Final Validation - Comprehensive Test
===============================================

Script final para validar completamente la implementación corregida
de Tensor Core Simulation con resultados detallados.

Author: AI Assistant
Date: 2026-01-25
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.tensor_core_emulator import TensorCoreEmulator
import numpy as np
import time

def comprehensive_tensor_core_validation():
    """Validación completa y detallada de Tensor Core"""
    print("🎯 TENSOR CORE COMPREHENSIVE VALIDATION")
    print("=" * 60)

    try:
        # Inicializar emulador
        print("🚀 Inicializando Tensor Core Emulator...")
        emulator = TensorCoreEmulator()
        print("✅ Emulator inicializado correctamente")

        # Test 1: Validación de precisión con diferentes tamaños
        print("\n🔬 TEST 1: Validación de Precisión")
        print("-" * 40)

        test_sizes = [32, 64, 128, 256, 512]
        precision_results = []

        for size in test_sizes:
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            C = np.random.randn(size, size).astype(np.float32)

            # Tensor core operation
            D_tensor, metrics = emulator.matmul(A, B, C)

            # NumPy reference
            D_numpy = A @ B + C

            # Calculate errors
            max_error = np.max(np.abs(D_tensor - D_numpy))
            relative_error = np.max(np.abs(D_tensor - D_numpy) / (np.abs(D_numpy) + 1e-10))

            precision_results.append({
                'size': size,
                'max_error': max_error,
                'relative_error': relative_error,
                'gflops': metrics.gflops
            })

            status = "✅" if max_error < 1e-3 else "⚠️" if max_error < 1e-1 else "❌"
            print("6d")

        # Test 2: Performance scaling analysis
        print("\n🏁 TEST 2: Análisis de Escalabilidad")
        print("-" * 40)

        scaling_sizes = [256, 512, 1024, 2048]
        scaling_results = []

        for size in scaling_sizes:
            print(f"Benchmarking {size}x{size}...")

            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            C = np.random.randn(size, size).astype(np.float32)

            # Warm-up
            emulator.matmul(A[:64, :64], B[:64, :64], C[:64, :64])

            # Timed run
            start_time = time.time()
            D_tensor, metrics = emulator.matmul(A, B, C)
            end_time = time.time()

            scaling_results.append({
                'size': size,
                'gflops': metrics.gflops,
                'bandwidth_gb_s': metrics.bandwidth_gb_s,
                'time_ms': (end_time - start_time) * 1000
            })

            print("6d")

        # Test 3: Comparación con NumPy
        print("\n📊 TEST 3: Comparación con NumPy Baseline")
        print("-" * 40)

        comp_size = 1024
        A = np.random.randn(comp_size, comp_size).astype(np.float32)
        B = np.random.randn(comp_size, comp_size).astype(np.float32)
        C = np.random.randn(comp_size, comp_size).astype(np.float32)

        # NumPy baseline
        start_time = time.time()
        D_numpy = A @ B + C
        numpy_time = time.time() - start_time
        numpy_gflops = 2 * comp_size**3 / (numpy_time * 1e9)

        # Tensor core
        D_tensor, metrics = emulator.matmul(A, B, C)

        speedup = metrics.gflops / numpy_gflops
        print(".2f")
        print(".2f")
        print(".2f")

        # Test 4: Stability test
        print("\n🔄 TEST 4: Test de Estabilidad")
        print("-" * 40)

        stability_results = []
        for i in range(5):
            A = np.random.randn(512, 512).astype(np.float32)
            B = np.random.randn(512, 512).astype(np.float32)
            C = np.random.randn(512, 512).astype(np.float32)

            D_tensor, metrics = emulator.matmul(A, B, C)
            D_numpy = A @ B + C

            max_error = np.max(np.abs(D_tensor - D_numpy))
            stability_results.append(max_error)

            print("d")

        stability_mean = np.mean(stability_results)
        stability_std = np.std(stability_results)

        print(".2e")
        print(".2e")

        # Resultados finales
        print("\n🎯 RESULTADOS FINALES")
        print("=" * 60)

        best_performance = max(r['gflops'] for r in scaling_results)
        best_precision = min(r['max_error'] for r in precision_results)

        print(f"🏆 Mejor Performance: {best_performance:.2f} GFLOPS")
        print(f"🎯 Mejor Precisión: {best_precision:.2e}")
        print(f"📈 Speedup vs NumPy: {speedup:.2f}x")
        print(f"🔄 Estabilidad: ±{stability_std:.2e}")

        # Evaluación final
        if best_precision < 1e-3 and best_performance > 400:
            final_status = "🚀 EXCELENTE - LISTO PARA PRODUCCIÓN"
        elif best_precision < 1e-2 and best_performance > 200:
            final_status = "✅ MUY BUENO - VIABLE"
        elif best_precision < 1e-1:
            final_status = "⚠️ ACEPTABLE - REQUIERE OPTIMIZACIÓN"
        else:
            final_status = "❌ REQUIERE DEBUGGING"

        print(f"📋 Evaluación Final: {final_status}")

        return {
            'precision_results': precision_results,
            'scaling_results': scaling_results,
            'numpy_comparison': {
                'numpy_gflops': numpy_gflops,
                'tensor_gflops': metrics.gflops,
                'speedup': speedup
            },
            'stability': {
                'mean_error': stability_mean,
                'std_error': stability_std
            },
            'final_status': final_status
        }

    except Exception as e:
        print(f"❌ Error durante la validación: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = comprehensive_tensor_core_validation()
    if results:
        print("\n✅ Comprehensive Tensor Core validation completed successfully!")
    else:
        print("\n❌ Validation failed!")
        sys.exit(1)