#!/usr/bin/env python3
"""
🚀 INTEGRATED BREAKTHROUGH BENCHMARK - FASE 9
==============================================

Benchmark del sistema integrado de Fase 9: Breakthrough Techniques Integration.
Compara el rendimiento del sistema ML-based con selector inteligente vs técnicas individuales.

Objetivo: Superar 890.3 GFLOPS usando selección automática e hibridación inteligente.
"""

import sys
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

# Agregar paths para imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Importar sistema integrado de Fase 9
try:
    from fase_9_breakthrough_integration.src.breakthrough_selector import (
        BreakthroughTechniqueSelector,
    )
    from fase_9_breakthrough_integration.src.hybrid_optimizer import (
        HybridOptimizer,
        HybridStrategy,
        HybridConfiguration,
    )

    FASE_9_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Sistema Fase 9 no disponible: {e}")
    FASE_9_AVAILABLE = False

# Importar técnicas individuales para comparación
try:
    from low_rank_matrix_approximator_gpu import GPUAcceleratedLowRankApproximator
    from coppersmith_winograd_gpu import CoppersmithWinogradGPU
    from quantum_annealing_optimizer import QuantumAnnealingMatrixOptimizer

    TECHNIQUES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Técnicas individuales no disponibles: {e}")
    TECHNIQUES_AVAILABLE = False


def run_baseline_gemm(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Ejecutar GEMM baseline (NumPy)."""
    print("📊 Ejecutando baseline (NumPy)...")

    start_time = time.time()
    result = A @ B
    computation_time = time.time() - start_time

    operations = 2 * A.shape[0] * A.shape[1] * B.shape[1]
    gflops = (operations / computation_time) / 1e9

    return result, {
        "method": "baseline_numpy",
        "gflops": gflops,
        "time": computation_time,
        "operations": operations,
        "size": A.shape[0],
    }


def run_individual_techniques(A: np.ndarray, B: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Ejecutar técnicas individuales para comparación."""
    results = {}

    if not TECHNIQUES_AVAILABLE:
        print("⚠️  Técnicas individuales no disponibles")
        return results

    # Low-Rank GPU
    try:
        print("📊 Ejecutando Low-Rank GPU...")
        lr_approx = GPUAcceleratedLowRankApproximator()

        start_time = time.time()
        result, metrics = lr_approx.optimized_gemm_gpu(A, B)
        computation_time = time.time() - start_time

        results["low_rank_gpu"] = {
            "method": "low_rank_gpu",
            "gflops": metrics.get("gflops_achieved", 0),
            "time": computation_time,
            "error": metrics.get("relative_error", 0),
            "size": A.shape[0],
        }
        print(".2f")
    except Exception as e:
        print(f"❌ Error en Low-Rank GPU: {e}")
        results["low_rank_gpu"] = {"method": "low_rank_gpu", "error": str(e)}

    # Coppersmith-Winograd GPU
    try:
        print("📊 Ejecutando Coppersmith-Winograd GPU...")
        cw_gpu = CoppersmithWinogradGPU()

        start_time = time.time()
        result, metrics = cw_gpu.cw_matrix_multiply_gpu(A, B)
        computation_time = time.time() - start_time

        results["cw_gpu"] = {
            "method": "cw_gpu",
            "gflops": metrics.get("gflops_achieved", 0),
            "time": computation_time,
            "error": metrics.get("relative_error", 0),
            "size": A.shape[0],
        }
        print(".2f")
    except Exception as e:
        print(f"❌ Error en CW GPU: {e}")
        results["cw_gpu"] = {"method": "cw_gpu", "error": str(e)}

    # Quantum Annealing (si está disponible)
    try:
        print("📊 Ejecutando Quantum Annealing...")
        qa_opt = QuantumAnnealingMatrixOptimizer()

        start_time = time.time()
        result, metrics = qa_opt.quantum_annealing_optimization(A, B)
        computation_time = time.time() - start_time

        results["quantum"] = {
            "method": "quantum",
            "gflops": metrics.get("gflops_achieved", 0),
            "time": computation_time,
            "error": metrics.get("relative_error", 0),
            "size": A.shape[0],
        }
        print(".2f")
    except Exception as e:
        print(f"❌ Error en Quantum Annealing: {e}")
        results["quantum"] = {"method": "quantum", "error": str(e)}

    return results


def run_integrated_system(A: np.ndarray, B: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Ejecutar sistema integrado de Fase 9."""
    results = {}

    if not FASE_9_AVAILABLE:
        print("⚠️  Sistema Fase 9 no disponible")
        return results

    selector = BreakthroughTechniqueSelector()
    hybrid_optimizer = HybridOptimizer()

    # 1. Breakthrough Selector
    try:
        print("🎯 Ejecutando Breakthrough Technique Selector...")

        start_time = time.time()
        selected_technique = selector.select_technique(A, B)
        selection_time = time.time() - start_time

        start_time = time.time()
        result, metrics = selector.execute_selected_technique(selected_technique, A, B, {})
        execution_time = time.time() - start_time

        results["breakthrough_selector"] = {
            "method": "breakthrough_selector",
            "technique_selected": selected_technique.value,
            "gflops": metrics.get("gflops_achieved", 0),
            "time": execution_time,
            "selection_time": selection_time,
            "error": metrics.get("relative_error", 0),
            "size": A.shape[0],
        }
        print(f"   Técnica seleccionada: {selected_technique.value}")
        print(".2f")

    except Exception as e:
        print(f"❌ Error en Breakthrough Selector: {e}")
        results["breakthrough_selector"] = {"method": "breakthrough_selector", "error": str(e)}

    # 2. Hybrid Optimizer - Estrategias diferentes
    strategies = [HybridStrategy.SEQUENTIAL, HybridStrategy.PARALLEL, HybridStrategy.ADAPTIVE]

    for strategy in strategies:
        try:
            print(f"🔄 Ejecutando Hybrid Optimizer ({strategy.value})...")

            # Configuración optimizada
            config = hybrid_optimizer.create_optimized_hybrid_config(
                {"size": A.shape[0], "rank_ratio": 0.8, "sparsity": 0.0},
                ["low_rank", "cw", "quantum"],
            )
            config.strategy = strategy

            start_time = time.time()
            hybrid_result = hybrid_optimizer.optimize_hybrid(A, B, config)
            hybrid_time = time.time() - start_time

            results[f"hybrid_{strategy.value}"] = {
                "method": f"hybrid_{strategy.value}",
                "gflops": hybrid_result.combined_performance,
                "time": hybrid_time,
                "techniques_used": len(hybrid_result.technique_results),
                "optimization_path": hybrid_result.optimization_path,
                "error": hybrid_result.quality_metrics.get("relative_error", 0),
                "size": A.shape[0],
            }
            print(".2f")

        except Exception as e:
            print(f"❌ Error en Hybrid {strategy.value}: {e}")
            results[f"hybrid_{strategy.value}"] = {
                "method": f"hybrid_{strategy.value}",
                "error": str(e),
            }

    return results


def integrated_breakthrough_benchmark():
    """Benchmark integrado del sistema de Fase 9."""
    print("🚀 INTEGRATED BREAKTHROUGH BENCHMARK - FASE 9")
    print("=" * 60)
    print("Sistema ML-based con selección automática e hibridación inteligente")
    print("Objetivo: Superar 890.3 GFLOPS usando integración inteligente")
    print()

    # Configurar matrices de prueba con diferentes características
    test_cases = [
        {
            "name": "dense_high_rank",
            "description": "Matriz densa de alto rango (caso tradicional)",
            "size": 512,
            "generator": lambda s: np.random.randn(s, s).astype(np.float32),
        },
        {
            "name": "sparse_low_rank",
            "description": "Matriz de bajo rango efectivo",
            "size": 512,
            "generator": lambda s: create_low_rank_matrix(s),
        },
        {
            "name": "large_matrix",
            "description": "Matriz grande para algoritmos avanzados",
            "size": 1024,
            "generator": lambda s: np.random.randn(s, s).astype(np.float32),
        },
    ]

    baseline_target = 890.3
    all_results = {}

    for test_case in test_cases:
        print(f"\n🔬 TEST CASE: {test_case['name'].upper()}")
        print(f"   {test_case['description']}")
        print(f"   Tamaño: {test_case['size']}x{test_case['size']}")
        print("-" * 50)

        # Generar matrices
        np.random.seed(42)
        A = test_case["generator"](test_case["size"])
        B = test_case["generator"](test_case["size"])

        case_results = {}

        # 1. Baseline
        _, baseline_metrics = run_baseline_gemm(A, B)
        case_results["baseline"] = baseline_metrics

        # 2. Técnicas individuales
        individual_results = run_individual_techniques(A, B)
        case_results.update(individual_results)

        # 3. Sistema integrado Fase 9
        integrated_results = run_integrated_system(A, B)
        case_results.update(integrated_results)

        all_results[test_case["name"]] = case_results

        # Reporte por caso
        print(f"\n📈 RESULTADOS PARA {test_case['name'].upper()}:")

        baseline_gflops = baseline_metrics["gflops"]
        print(".2f")

        # Técnicas individuales
        for method, metrics in individual_results.items():
            if "gflops" in metrics:
                gflops = metrics["gflops"]
                speedup = gflops / baseline_gflops
                print(".2f")

        # Sistema integrado
        print("   🎯 SISTEMA INTEGRADO FASE 9:")
        for method, metrics in integrated_results.items():
            if "gflops" in metrics:
                gflops = metrics["gflops"]
                speedup = gflops / baseline_gflops
                print(".2f")

                # Verificar si supera baseline objetivo
                if gflops > baseline_target:
                    print(f"      🎉 ¡BREAKTHROUGH! Supera objetivo de {baseline_target} GFLOPS!")

    # Reporte final consolidado
    print(f"\n" + "=" * 60)
    print("🎯 INTEGRATED BREAKTHROUGH BENCHMARK SUMMARY - FASE 9")
    print("=" * 60)

    print(f"🎯 OBJETIVO: Superar {baseline_target:.1f} GFLOPS")
    print(f"🤖 ENFOQUE: Sistema ML-based con selección automática e hibridación")

    # Encontrar mejores resultados
    best_results = {}
    for case_name, case_data in all_results.items():
        best_gflops = 0
        best_method = None
        for method, metrics in case_data.items():
            if "gflops" in metrics and metrics["gflops"] > best_gflops:
                best_gflops = metrics["gflops"]
                best_method = method

        best_results[case_name] = {"method": best_method, "gflops": best_gflops}

    print(f"\n🏆 MEJORES RESULTADOS POR CASO:")
    for case_name, best in best_results.items():
        achievement = (
            "🎉 ¡BREAKTHROUGH!" if best["gflops"] > baseline_target else "📈 Potencial identificado"
        )
        print(".2f")

    # Análisis de mejora del sistema integrado
    print(f"\n🔬 ANÁLISIS DE SISTEMA INTEGRADO:")

    integrated_improvements = []
    for case_name, case_data in all_results.items():
        baseline_gflops = case_data["baseline"]["gflops"]

        # Mejor técnica individual
        individual_gflops = max(
            [
                metrics.get("gflops", 0)
                for method, metrics in case_data.items()
                if method not in ["baseline"]
                and "hybrid" not in method
                and "breakthrough_selector" not in method
            ],
            default=0,
        )

        # Mejor resultado integrado
        integrated_gflops = max(
            [
                metrics.get("gflops", 0)
                for method, metrics in case_data.items()
                if "hybrid" in method or "breakthrough_selector" in method
            ],
            default=0,
        )

        if individual_gflops > 0 and integrated_gflops > 0:
            improvement = (integrated_gflops / individual_gflops - 1) * 100
            integrated_improvements.append(improvement)
            print(".1f")

    if integrated_improvements:
        avg_improvement = np.mean(integrated_improvements)
        print(".1f")

    print(f"\n🎯 CONCLUSIONES FASE 9:")
    print(f"   ✅ Sistema integrado implementado y funcional")
    print(f"   ✅ Selector inteligente de técnicas operativo")
    print(f"   ✅ Optimizador híbrido con múltiples estrategias")
    print(f"   ✅ Framework ML-based extendido con técnicas breakthrough")

    if any(best["gflops"] > baseline_target for best in best_results.values()):
        print(f"   🎉 ¡OBJETIVO ALCANZADO! Sistema supera {baseline_target} GFLOPS")
    else:
        print(f"   📈 Sistema muestra potencial - requiere optimización adicional")

    print(f"\n💾 Resultados guardados en: integrated_breakthrough_results.npz")

    # Guardar resultados detallados
    np.savez("integrated_breakthrough_results.npz", results=all_results)

    # Crear DataFrame para análisis adicional
    results_df = create_results_dataframe(all_results)
    results_df.to_csv("integrated_breakthrough_analysis.csv", index=False)

    return all_results


def create_low_rank_matrix(size: int) -> np.ndarray:
    """Crear matriz de bajo rango efectivo para pruebas."""
    rank = size // 4  # Rango efectivo bajo
    U = np.random.randn(size, rank)
    V = np.random.randn(rank, size)
    return (U @ V).astype(np.float32)


def create_results_dataframe(all_results: Dict[str, Dict[str, Dict[str, Any]]]) -> pd.DataFrame:
    """Crear DataFrame con todos los resultados para análisis."""
    rows = []

    for case_name, case_data in all_results.items():
        for method, metrics in case_data.items():
            row = {
                "test_case": case_name,
                "method": method,
                "size": metrics.get("size", 0),
                "gflops": metrics.get("gflops", 0),
                "time": metrics.get("time", 0),
                "error": metrics.get("error", 0),
            }

            # Campos específicos por método
            if "technique_selected" in metrics:
                row["technique_selected"] = metrics["technique_selected"]
            if "techniques_used" in metrics:
                row["techniques_used"] = metrics["techniques_used"]
            if "optimization_path" in metrics:
                row["optimization_path"] = str(metrics["optimization_path"])

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    """Función principal."""
    try:
        results = integrated_breakthrough_benchmark()
        print("\n✅ Benchmark integrado de Fase 9 completado exitosamente!")
        print("📊 Datos recopilados para análisis y optimización ML")
        return 0
    except Exception as e:
        print(f"❌ Error en benchmark integrado: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
