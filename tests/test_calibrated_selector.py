#!/usr/bin/env python3
"""
🧪 TEST: CALIBRATED INTELLIGENT SELECTOR
=========================================

Test exhaustivo del selector inteligente calibrado.
Valida los objetivos:
- Selección de AI Predictor/alto rendimiento en 90%+ de casos
- Confianza promedio >80%

Author: AI Assistant
Date: 2026-02-02
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Agregar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from ml_models.calibrated_intelligent_selector import (
    CalibratedIntelligentSelector,
    MatrixCharacteristics,
    OptimizationTechnique,
    SelectionResult,
)


def create_test_matrices() -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Crea matrices de prueba con diferentes características."""

    np.random.seed(42)
    matrices = []

    # 1. Matrices densas de diferentes tamaños
    for size in [64, 128, 256, 512, 768, 1024, 1536, 2048]:
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        matrices.append((f"dense_{size}x{size}", A, B))

    # 2. Matrices sparse (muchos ceros)
    for size in [256, 512, 1024]:
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        # Crear sparsidad al 70%
        mask_a = np.random.random((size, size)) > 0.7
        mask_b = np.random.random((size, size)) > 0.7
        A[mask_a] = 0
        B[mask_b] = 0
        matrices.append((f"sparse_70%_{size}x{size}", A, B))

    # 3. Matrices de bajo rango
    for size in [256, 512, 1024]:
        rank = size // 8  # Rango efectivo = 12.5% del tamaño
        U = np.random.randn(size, rank).astype(np.float32)
        V = np.random.randn(rank, size).astype(np.float32)
        A = U @ V  # Matriz de bajo rango
        B = np.random.randn(size, size).astype(np.float32)
        matrices.append((f"low_rank_{size}x{size}", A, B))

    # 4. Matrices simétricas
    for size in [256, 512]:
        temp = np.random.randn(size, size).astype(np.float32)
        A = (temp + temp.T) / 2  # Simétrica
        B = np.random.randn(size, size).astype(np.float32)
        matrices.append((f"symmetric_{size}x{size}", A, B))

    # 5. Matrices con valores específicos (bien condicionadas)
    for size in [256, 512]:
        A = np.eye(size, dtype=np.float32) + 0.1 * np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        matrices.append((f"well_conditioned_{size}x{size}", A, B))

    return matrices


def run_comprehensive_test():
    """Ejecuta test exhaustivo del selector."""

    print("🧪 TEST EXHAUSTIVO: CALIBRATED INTELLIGENT SELECTOR")
    print("=" * 70)
    print("Objetivo 1: Selección de alto rendimiento >= 90%")
    print("Objetivo 2: Confianza promedio >= 80%")
    print("=" * 70)

    # Crear selector
    selector = CalibratedIntelligentSelector(prefer_ai_predictor=True)

    # Crear matrices de prueba
    test_matrices = create_test_matrices()

    print(f"\n📊 Total de casos de prueba: {len(test_matrices)}")

    # Técnicas de alto rendimiento
    HIGH_PERF_TECHNIQUES = {
        OptimizationTechnique.AI_PREDICTOR,
        OptimizationTechnique.OPENCL_GEMM,
        OptimizationTechnique.QUANTUM,
    }

    # Ejecutar tests
    results = []
    high_perf_count = 0
    high_confidence_count = 0
    total_confidence = 0.0

    print("\n🔬 EJECUTANDO TESTS:")
    print("-" * 70)

    for name, A, B in test_matrices:
        start_time = time.time()
        result = selector.select_technique(A, B)
        elapsed = time.time() - start_time

        is_high_perf = result.technique in HIGH_PERF_TECHNIQUES
        is_high_conf = result.confidence >= 0.80

        if is_high_perf:
            high_perf_count += 1
        if is_high_conf:
            high_confidence_count += 1
        total_confidence += result.confidence

        results.append(
            {
                "name": name,
                "technique": result.technique,
                "confidence": result.confidence,
                "gflops": result.predicted_gflops,
                "time_ms": result.selection_time_ms,
                "is_high_perf": is_high_perf,
                "is_high_conf": is_high_conf,
            }
        )

        # Mostrar resultado
        status = "✅" if is_high_perf and is_high_conf else "⚠️" if is_high_perf else "❌"
        print(
            f"  {status} {name:35} → {result.technique.value:15} "
            f"conf={result.confidence:.2f} gflops={result.predicted_gflops:.1f}"
        )

    # Calcular estadísticas
    total = len(results)
    high_perf_rate = high_perf_count / total
    high_conf_rate = high_confidence_count / total
    avg_confidence = total_confidence / total

    # Mostrar resumen
    print("\n" + "=" * 70)
    print("📈 RESUMEN DE RESULTADOS")
    print("=" * 70)

    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"   Total de pruebas: {total}")
    print(
        f"   Selecciones de alto rendimiento: {high_perf_count}/{total} ({high_perf_rate*100:.1f}%)"
    )
    print(
        f"   Selecciones con alta confianza (>=80%): {high_confidence_count}/{total} ({high_conf_rate*100:.1f}%)"
    )
    print(f"   Confianza promedio: {avg_confidence*100:.1f}%")

    # Distribución de técnicas
    from collections import Counter

    technique_dist = Counter([r["technique"].value for r in results])

    print(f"\n📊 DISTRIBUCIÓN DE TÉCNICAS:")
    for tech, count in technique_dist.most_common():
        pct = count / total * 100
        print(f"   {tech}: {count} ({pct:.1f}%)")

    # Distribución por tipo de matriz
    print(f"\n📊 RESULTADOS POR TIPO DE MATRIZ:")

    matrix_types = {
        "dense": [r for r in results if "dense" in r["name"]],
        "sparse": [r for r in results if "sparse" in r["name"]],
        "low_rank": [r for r in results if "low_rank" in r["name"]],
        "symmetric": [r for r in results if "symmetric" in r["name"]],
        "well_conditioned": [r for r in results if "well_conditioned" in r["name"]],
    }

    for mtype, mresults in matrix_types.items():
        if mresults:
            hp_count = sum(1 for r in mresults if r["is_high_perf"])
            hp_rate = hp_count / len(mresults) * 100
            avg_conf = sum(r["confidence"] for r in mresults) / len(mresults) * 100
            print(
                f"   {mtype}: {len(mresults)} tests, {hp_rate:.0f}% high-perf, {avg_conf:.0f}% avg conf"
            )

    # Validación de objetivos
    print("\n" + "=" * 70)
    print("🎯 VALIDACIÓN DE OBJETIVOS")
    print("=" * 70)

    obj1_pass = high_perf_rate >= 0.90
    obj2_pass = avg_confidence >= 0.80

    print(f"\n   Objetivo 1: Selección alto rendimiento >= 90%")
    print(
        f"              Resultado: {high_perf_rate*100:.1f}% {'✅ PASS' if obj1_pass else '❌ FAIL'}"
    )

    print(f"\n   Objetivo 2: Confianza promedio >= 80%")
    print(
        f"              Resultado: {avg_confidence*100:.1f}% {'✅ PASS' if obj2_pass else '❌ FAIL'}"
    )

    # Resultado final
    print("\n" + "=" * 70)
    if obj1_pass and obj2_pass:
        print("🎉 ¡TODOS LOS OBJETIVOS CUMPLIDOS!")
        print("   El selector inteligente está correctamente calibrado.")
    else:
        print("⚠️  Algunos objetivos no se cumplieron.")
        print("   Se recomienda más calibración o ajuste de pesos.")
    print("=" * 70)

    # Estadísticas del selector
    stats = selector.get_stats()
    print(f"\n📊 ESTADÍSTICAS INTERNAS DEL SELECTOR:")
    print(f"   Total selecciones: {stats['total_selections']}")
    print(f"   Tasa alta confianza: {stats['high_confidence_rate']*100:.1f}%")
    print(f"   Tasa alto rendimiento: {stats['high_performance_selection_rate']*100:.1f}%")

    return obj1_pass and obj2_pass, results


def test_edge_cases():
    """Prueba casos especiales."""

    print("\n" + "=" * 70)
    print("🔍 TEST DE CASOS ESPECIALES")
    print("=" * 70)

    selector = CalibratedIntelligentSelector()

    # 1. Matriz muy pequeña
    print("\n1. Matriz muy pequeña (16x16):")
    A = np.random.randn(16, 16).astype(np.float32)
    result = selector.select_technique(A)
    print(f"   Técnica: {result.technique.value}, Confianza: {result.confidence:.2f}")

    # 2. Matriz muy grande
    print("\n2. Matriz grande (4096x4096):")
    A = np.random.randn(4096, 4096).astype(np.float32)
    result = selector.select_technique(A)
    print(f"   Técnica: {result.technique.value}, Confianza: {result.confidence:.2f}")

    # 3. Matriz identidad
    print("\n3. Matriz identidad (512x512):")
    A = np.eye(512, dtype=np.float32)
    result = selector.select_technique(A)
    print(f"   Técnica: {result.technique.value}, Confianza: {result.confidence:.2f}")

    # 4. Matriz de ceros
    print("\n4. Matriz casi-ceros (512x512, 99% sparse):")
    A = np.zeros((512, 512), dtype=np.float32)
    A[
        np.random.choice(512 * 512, size=512 * 512 // 100, replace=False) // 512,
        np.random.choice(512 * 512, size=512 * 512 // 100, replace=False) % 512,
    ] = np.random.randn(512 * 512 // 100)
    result = selector.select_technique(A)
    print(f"   Técnica: {result.technique.value}, Confianza: {result.confidence:.2f}")

    # 5. Matriz con valores extremos
    print("\n5. Matriz con valores extremos (512x512):")
    A = np.random.randn(512, 512).astype(np.float32) * 1e6
    result = selector.select_technique(A)
    print(f"   Técnica: {result.technique.value}, Confianza: {result.confidence:.2f}")

    print("\n✅ Tests de casos especiales completados")


def benchmark_selection_speed():
    """Benchmark de velocidad de selección."""

    print("\n" + "=" * 70)
    print("⚡ BENCHMARK DE VELOCIDAD DE SELECCIÓN")
    print("=" * 70)

    selector = CalibratedIntelligentSelector()

    sizes = [128, 256, 512, 1024, 2048]
    iterations = 10

    print(f"\n   Iteraciones por tamaño: {iterations}")

    for size in sizes:
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        times = []
        for _ in range(iterations):
            start = time.time()
            selector.select_technique(A, B)
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000

        print(f"   {size}x{size}: {avg_time:.2f} ± {std_time:.2f} ms")

    print("\n✅ Benchmark de velocidad completado")


if __name__ == "__main__":
    print("\n" * 2)

    # Test principal
    passed, results = run_comprehensive_test()

    # Tests adicionales
    test_edge_cases()
    benchmark_selection_speed()

    print("\n")

    # Código de salida
    sys.exit(0 if passed else 1)
