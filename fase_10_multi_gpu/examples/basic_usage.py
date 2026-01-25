#!/usr/bin/env python3
"""
🚀 EJEMPLO BÁSICO: Multi-GPU Matrix Multiplication
==================================================

Ejemplo simple de uso del framework multi-GPU para multiplicación de matrices.
Demuestra la funcionalidad básica y sirve como punto de partida para desarrolladores.

Requisitos:
- Múltiples GPUs AMD Radeon (o al menos una para testing)
- PyOpenCL instalado
- Python 3.8+

Autor: AI Assistant
Fecha: 2026-01-25
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Añadir el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from multi_gpu_manager import MultiGPUManager, distributed_gemm, create_multi_gpu_gemm
    IMPORTS_OK = True
except ImportError as e:
    print(f"❌ Error importando multi_gpu_manager: {e}")
    print("Asegúrate de que PyOpenCL esté instalado: pip install pyopencl")
    IMPORTS_OK = False

def demo_basic_multi_gpu():
    """Demo básico del framework multi-GPU."""
    print("🚀 Demo Básico: Multi-GPU Matrix Multiplication")
    print("=" * 60)

    if not IMPORTS_OK:
        return

    try:
        # 1. Crear manager multi-GPU
        print("🔧 Inicializando Multi-GPU Manager...")
        manager = MultiGPUManager(log_level="INFO")

        # 2. Mostrar información de GPUs disponibles
        print(f"📊 GPUs disponibles: {len(manager.devices)}")
        for i, device in enumerate(manager.devices):
            print(f"  GPU {i}: {device.name} ({device.memory_gb:.1f} GB)")

        # 3. Crear matrices de prueba
        print("\n🧮 Creando matrices de prueba...")
        sizes = [
            (256, 256, 256),   # Pequeño para testing rápido
            (512, 512, 512),   # Mediano
            # (1024, 1024, 1024)  # Grande (comentar si es lento)
        ]

        for M, N, K in sizes:
            print(f"\n--- Test: {M}x{N}x{K} ---")

            # Generar matrices aleatorias
            A = np.random.rand(M, K).astype(np.float32)
            B = np.random.rand(K, N).astype(np.float32)

            # 4. Calcular distribución óptima
            print("📋 Calculando distribución de carga...")
            distributions = manager.get_optimal_workload_distribution(M, N, K)

            # 5. Ejecutar multiplicación distribuida
            print("🚀 Ejecutando multiplicación distribuida...")
            start_time = time.time()

            C_distributed = distributed_gemm(A, B, manager)

            end_time = time.time()
            distributed_time = end_time - start_time

            # 6. Verificar resultado (comparar con NumPy)
            print("✅ Verificando resultado...")
            start_time = time.time()
            C_numpy = A @ B  # Multiplicación de referencia
            end_time = time.time()
            numpy_time = end_time - start_time

            # Calcular error
            max_error = np.max(np.abs(C_distributed - C_numpy))
            relative_error = max_error / np.max(np.abs(C_numpy)) if np.max(np.abs(C_numpy)) > 0 else 0

            # 7. Mostrar resultados
            print("📊 Resultados:")
            print(".2f")
            print(".2f")
            print(".2e")
            print(".2e")

            if len(manager.devices) > 1:
                speedup = numpy_time / distributed_time
                efficiency = manager.get_scaling_efficiency(numpy_time, distributed_time)
                print(".2f")
                print(".1f")

        # 8. Benchmark del setup
        print("\n📈 Benchmark del setup multi-GPU:")
        metrics = manager.benchmark_multi_gpu_setup()
        print(f"   GPUs totales: {metrics['num_gpus']}")
        print(".1f")
        print(f"   Compute Units totales: {metrics['total_compute_units']}")

        if 'communication_test' in metrics:
            comm = metrics['communication_test']
            print(".2f")

        # 9. Cleanup
        print("\n🧹 Limpiando recursos...")
        manager.cleanup()

        print("✅ Demo completado exitosamente!")

    except Exception as e:
        print(f"❌ Error en demo: {e}")
        import traceback
        traceback.print_exc()

def demo_single_gpu_fallback():
    """Demo de fallback a single GPU cuando solo hay una disponible."""
    print("\n🔄 Demo: Fallback a Single GPU")
    print("=" * 40)

    if not IMPORTS_OK:
        return

    try:
        # Forzar uso de solo 1 GPU aunque haya más
        manager = create_multi_gpu_gemm(512, 512, 512, num_gpus_desired=1)

        print(f"🔧 Usando {len(manager.devices)} GPU(s) (forzado a 1)")

        # Test simple
        A = np.random.rand(256, 256).astype(np.float32)
        B = np.random.rand(256, 256).astype(np.float32)

        start_time = time.time()
        C = distributed_gemm(A, B, manager)
        end_time = time.time()

        print(".2f")
        print(f"✅ Resultado: {C.shape}")

        manager.cleanup()

    except Exception as e:
        print(f"❌ Error en fallback demo: {e}")

def demo_scaling_analysis():
    """Análisis de escalabilidad con diferentes números de GPUs."""
    print("\n📊 Demo: Análisis de Escalabilidad")
    print("=" * 40)

    if not IMPORTS_OK:
        return

    try:
        base_manager = MultiGPUManager()

        if len(base_manager.devices) < 2:
            print("⚠️  Se necesita al menos 2 GPUs para análisis de escalabilidad")
            base_manager.cleanup()
            return

        # Matriz de tamaño fijo
        M, N, K = 1024, 1024, 1024
        A = np.random.rand(M, K).astype(np.float32)
        B = np.random.rand(K, N).astype(np.float32)

        print(f"🧮 Matriz: {M}x{N}x{K}")
        print("📈 Probando escalabilidad:")

        results = []

        for num_gpus in range(1, len(base_manager.devices) + 1):
            print(f"\n  --- {num_gpus} GPU(s) ---")

            # Crear manager con N GPUs
            manager = create_multi_gpu_gemm(M, N, K, num_gpus_desired=num_gpus)

            start_time = time.time()
            C = distributed_gemm(A, B, manager)
            end_time = time.time()

            elapsed = end_time - start_time
            speedup = results[0][1] / elapsed if results else 1.0
            efficiency = speedup / num_gpus

            results.append((num_gpus, elapsed, speedup, efficiency))

            print(".2f")
            print(".2f")
            print(".1f")

            manager.cleanup()

        # Resumen
        print("\n📋 Resumen de Escalabilidad:")
        print("GPUs | Tiempo (s) | Speedup | Efficiency")
        print("-----|-----------|---------|-----------")
        for num_gpus, time_taken, speedup, efficiency in results:
            print("4d")

        base_manager.cleanup()

    except Exception as e:
        print(f"❌ Error en análisis de escalabilidad: {e}")

if __name__ == "__main__":
    print("🎯 Multi-GPU Framework Examples")
    print("==============================\n")

    # Ejecutar demos
    demo_basic_multi_gpu()
    demo_single_gpu_fallback()
    demo_scaling_analysis()

    print("\n🎉 Todos los ejemplos completados!")
    print("\n💡 Próximos pasos:")
    print("   - Revisar multi_gpu_manager.py para entender la arquitectura")
    print("   - Contribuir con optimizaciones en kernels/")
    print("   - Añadir tests en tests/")
    print("   - Ver docs/ para documentación completa")