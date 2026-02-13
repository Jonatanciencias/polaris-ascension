#!/usr/bin/env python3
"""
Demo: Sistema de Caché de Kernels OpenCL

Demuestra la mejora de performance del sistema de caché persistente:
- Primera ejecución: Compila kernels (~2.8s)
- Ejecuciones subsiguientes: Carga desde caché (~0ms)

Uso:
    # Primera vez (compila)
    python examples/demo_kernel_cache.py --clear-cache

    # Segunda vez (usa caché)
    python examples/demo_kernel_cache.py
"""

import sys
from pathlib import Path
import time
import shutil
import argparse

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization_engines.optimized_kernel_engine import OptimizedKernelEngine
import numpy as np


def clear_cache():
    """Limpiar caché de kernels"""
    cache_dir = Path.home() / ".cache" / "radeon_rx580_kernels"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"🗑️  Caché limpiado: {cache_dir}")
    else:
        print("ℹ️  No hay caché para limpiar")


def demo_kernel_cache():
    """Demostración del sistema de caché"""

    print("\n" + "=" * 70)
    print("DEMO: Sistema de Caché de Kernels OpenCL")
    print("=" * 70 + "\n")

    # === Fase 1: Inicialización ===
    print("📦 Inicializando OptimizedKernelEngine...")
    print("   (Observa el tiempo de carga de kernels)\n")

    start_init = time.time()
    engine = OptimizedKernelEngine(
        device_index=0, enable_profiling=True, enable_advanced_memory=True
    )
    init_time = (time.time() - start_init) * 1000

    print(f"\n⏱️  Tiempo de inicialización: {init_time:.1f}ms")

    # Determinar si usó caché
    if init_time < 1500:
        print("   ✅ Kernels cargados desde CACHÉ (~0ms compilación)")
    else:
        print("   ⚡ Kernels COMPILADOS desde cero (~2.8s)")

    print("\n" + "-" * 70 + "\n")

    # === Fase 2: Operación GEMM ===
    print("🧮 Ejecutando operación GEMM como prueba...\n")

    size = 1024
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)

    start_gemm = time.time()
    result = engine.gemm(A, B)
    gemm_time = (time.time() - start_gemm) * 1000

    gflops = result.kernel_metrics.gflops
    kernel_name = result.kernel_metrics.kernel_name

    print(f"   Matriz: {size}x{size}")
    print(f"   Kernel: {kernel_name}")
    print(f"   Tiempo: {gemm_time:.2f}ms")
    print(f"   Performance: {gflops:.1f} GFLOPS")

    # Verificar corrección
    C_cpu = A @ B
    error = np.abs(result.result - C_cpu).mean()
    print(f"   Error vs CPU: {error:.2e}")

    if error < 1e-4:
        print("   ✅ Resultado CORRECTO")
    else:
        print("   ⚠️  Error mayor al esperado")

    print("\n" + "-" * 70 + "\n")

    # === Fase 3: Estadísticas ===
    print("📊 Estadísticas del Caché:\n")

    cache_dir = Path.home() / ".cache" / "radeon_rx580_kernels"
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.bin"))
        if cache_files:
            total_size = sum(f.stat().st_size for f in cache_files)
            print(f"   Archivos en caché: {len(cache_files)}")
            print(f"   Tamaño total: {total_size / 1024:.1f} KB")
            print(f"   Ubicación: {cache_dir}")
        else:
            print("   ⚠️  Caché vacío")
    else:
        print("   ⚠️  Directorio de caché no existe")

    print("\n" + "-" * 70 + "\n")

    # === Fase 4: Recomendaciones ===
    print("💡 Recomendaciones:\n")
    if init_time > 1500:
        print("   🔄 Ejecuta este script de nuevo para ver la mejora con caché")
        print("   📈 Tiempo esperado: ~600ms (5.7x más rápido)")
    else:
        print("   ✅ Caché funcionando correctamente")
        print("   🗑️  Para limpiar: python examples/demo_kernel_cache.py --clear-cache")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Demo del sistema de caché de kernels OpenCL")
    parser.add_argument(
        "--clear-cache", action="store_true", help="Limpiar caché antes de ejecutar"
    )

    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
        print()

    try:
        demo_kernel_cache()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
