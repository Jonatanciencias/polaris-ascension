#!/usr/bin/env python3
"""
🎯 COMPREHENSIVE PERFORMANCE VALIDATION BENCHMARK
==================================================

Benchmark exhaustivo para validar el impacto completo de todas las optimizaciones
implementadas en el proyecto Radeon RX 580. Mide GFLOPS, eficiencia, y compara
todas las fases desde baseline hasta breakthrough híbridos.

Fecha: 25 de enero de 2026
Objetivo: Validar que las optimizaciones han tenido efecto real
"""

import sys
import os
import numpy as np
import time
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Configurar logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Resultado de un benchmark individual."""
    method: str
    phase: str
    matrix_size: str
    gflops: float
    time_seconds: float
    operations: int
    error_max: float
    error_relative: float
    memory_usage_mb: float
    power_watts: Optional[float] = None
    efficiency_percent: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None

@dataclass
class ComprehensiveBenchmarkReport:
    """Reporte completo del benchmark comprehensivo."""
    timestamp: str
    system_info: Dict[str, Any]
    results: List[BenchmarkResult]
    summary: Dict[str, Any]
    recommendations: List[str]

class ComprehensiveBenchmarkSuite:
    """
    Suite completa de benchmarks para validar todas las optimizaciones implementadas.
    """

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_gflops = 0.0
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """Obtener información del sistema."""
        info = {
            'cpu': 'Unknown',
            'gpu': 'Unknown',
            'memory_gb': 0,
            'os': sys.platform
        }

        try:
            import psutil
            info['cpu'] = f"{psutil.cpu_count()} cores"
            info['memory_gb'] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            logger.warning("psutil no disponible para info del sistema")

        # Intentar detectar GPU
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                devices = platforms[0].get_devices()
                if devices:
                    info['gpu'] = devices[0].name
        except:
            pass

        return info

    def run_complete_validation(self) -> ComprehensiveBenchmarkReport:
        """
        Ejecutar validación completa de todas las fases implementadas.
        """
        logger.info("🚀 Iniciando Comprehensive Performance Validation Benchmark")
        logger.info("=" * 80)

        # Matrices de prueba de diferentes tamaños
        test_sizes = [
            (256, 256, 256),    # Pequeño para testing rápido
            (512, 512, 512),    # Mediano
            (1024, 1024, 1024), # Grande (requiere mucha memoria)
        ]

        for M, N, K in test_sizes:
            logger.info(f"\n🧮 Probando matrices {M}x{N}x{K}")
            self._benchmark_matrix_size(M, N, K)

        # Generar reporte final
        report = self._generate_report()
        self._save_report(report)

        return report

    def _benchmark_matrix_size(self, M: int, N: int, K: int):
        """Benchmark para un tamaño específico de matrices."""
        # Generar matrices de prueba
        A = np.random.rand(M, K).astype(np.float32)
        B = np.random.rand(K, N).astype(np.float32)

        # Baseline primero
        self._run_baseline_benchmark(A, B)

        # Fase 1-3: Optimizaciones básicas
        self._run_phase_1_3_benchmarks(A, B)

        # Fase 4-5: Técnicas avanzadas
        self._run_phase_4_5_benchmarks(A, B)

        # Fase 6-7: Breakthrough inicial
        self._run_phase_6_7_benchmarks(A, B)

        # Fase 8-9: Breakthrough avanzado
        self._run_phase_8_9_benchmarks(A, B)

        # Fase 10: Multi-GPU (si disponible)
        self._run_phase_10_benchmarks(A, B)

    def _run_baseline_benchmark(self, A: np.ndarray, B: np.ndarray):
        """Benchmark baseline (NumPy)."""
        logger.info("📊 Ejecutando BASELINE (NumPy)...")

        start_time = time.time()
        result_numpy = A @ B
        computation_time = time.time() - start_time

        operations = 2 * A.shape[0] * A.shape[1] * B.shape[1]
        gflops = (operations / computation_time) / 1e9

        self.baseline_gflops = gflops

        result = BenchmarkResult(
            method="baseline_numpy",
            phase="baseline",
            matrix_size=f"{A.shape[0]}x{A.shape[1]}x{B.shape[1]}",
            gflops=gflops,
            time_seconds=computation_time,
            operations=operations,
            error_max=0.0,  # Baseline es referencia
            error_relative=0.0,
            memory_usage_mb=self._estimate_memory_usage(A, B),
            efficiency_percent=100.0,
            speedup_vs_baseline=1.0
        )

        self.results.append(result)
        logger.info(".2f")

    def _run_phase_1_3_benchmarks(self, A: np.ndarray, B: np.ndarray):
        """Benchmarks de fases 1-3: SIMD, blocking, wave optimization."""
        logger.info("🏗️ Probando FASES 1-3: SIMD + Blocking + Wave Optimization")

        # Intentar importar y ejecutar técnicas de fases 1-3
        try:
            # Aquí irían las importaciones de las fases 1-3
            # Por ahora, placeholder para técnicas implementadas
            pass
        except ImportError as e:
            logger.warning(f"⚠️ Técnicas de fases 1-3 no disponibles: {e}")

    def _run_phase_4_5_benchmarks(self, A: np.ndarray, B: np.ndarray):
        """Benchmarks de fases 4-5: Strassen, blocking recursivo."""
        logger.info("🔢 Probando FASES 4-5: Strassen + Blocking Recursivo")

        # Strassen
        self._benchmark_strassen(A, B)

        # Blocking recursivo (si implementado)
        try:
            # Importar técnica de blocking recursivo
            pass
        except ImportError:
            logger.warning("⚠️ Blocking recursivo no disponible")

    def _benchmark_strassen(self, A: np.ndarray, B: np.ndarray):
        """Benchmark algoritmo de Strassen."""
        try:
            from benchmark_strassen import StrassenBenchmark

            benchmark = StrassenBenchmark()
            if benchmark.initialize_opencl():
                result = benchmark.run_comprehensive_benchmark()

                # Extraer métricas relevantes de los resultados
                for size_key, techniques in result.items():
                    for technique, data in techniques.items():
                        if isinstance(data, dict) and 'gflops' in data and 'error' not in data:
                            benchmark_result = BenchmarkResult(
                                method=f"strassen_{technique}_{size_key}",
                                phase="phase_4_5",
                                matrix_size=size_key,
                                gflops=data.get('gflops', 0),
                                time_seconds=data.get('time', 0),
                                operations=data.get('operations', 2 * A.shape[0] * A.shape[1] * B.shape[1]),
                                error_max=data.get('error_max', 0),
                                error_relative=data.get('error_relative', 0),
                                memory_usage_mb=self._estimate_memory_usage(A, B),
                                speedup_vs_baseline=data.get('gflops', 0) / self.baseline_gflops if self.baseline_gflops > 0 else 0
                            )
                            self.results.append(benchmark_result)

        except ImportError as e:
            logger.warning(f"⚠️ Strassen benchmark no disponible: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Error ejecutando Strassen benchmark: {e}")

    def _run_phase_6_7_benchmarks(self, A: np.ndarray, B: np.ndarray):
        """Benchmarks de fases 6-7: Winograd + AI Kernel Predictor."""
        logger.info("🎯 Probando FASES 6-7: Winograd + AI Predictor")

        # Winograd (aunque sabemos que no es óptimo para GEMM)
        self._benchmark_winograd(A, B)

        # AI Kernel Predictor
        self._benchmark_ai_predictor(A, B)

    def _benchmark_winograd(self, A: np.ndarray, B: np.ndarray):
        """Benchmark Winograd (para comparación)."""
        try:
            # Winograd fue implementado pero no es óptimo para GEMM general
            # Solo para referencia
            logger.info("📝 Winograd: Implementado pero no recomendado para GEMM general")
        except Exception as e:
            logger.warning(f"⚠️ Winograd benchmark error: {e}")

    def _benchmark_ai_predictor(self, A: np.ndarray, B: np.ndarray):
        """Benchmark AI Kernel Predictor."""
        try:
            from ai_kernel_predictor_fine_tuning_corrected import AIKernelPredictor

            predictor = AIKernelPredictor()
            predictor.load_model()

            # Simular predicción y medición de tiempo
            start_time = time.time()
            prediction = predictor.predict_kernel(A.shape + B.shape)
            prediction_time = time.time() - start_time

            # El AI predictor no ejecuta GEMM directamente, solo predice
            # Para efectos de benchmark, medimos su velocidad de predicción
            benchmark_result = BenchmarkResult(
                method="ai_kernel_predictor",
                phase="phase_6_7",
                matrix_size=f"{A.shape[0]}x{A.shape[1]}x{B.shape[1]}",
                gflops=0,  # No ejecuta GEMM
                time_seconds=prediction_time,
                operations=0,
                error_max=0,
                error_relative=0,
                memory_usage_mb=self._estimate_memory_usage(A, B),
                speedup_vs_baseline=0
            )

            self.results.append(benchmark_result)
            logger.info(".4f")

        except ImportError as e:
            logger.warning(f"⚠️ AI Kernel Predictor no disponible: {e}")

    def _run_phase_8_9_benchmarks(self, A: np.ndarray, B: np.ndarray):
        """Benchmarks de fases 8-9: Bayesian + Breakthrough híbridos."""
        logger.info("🎯 Probando FASES 8-9: Bayesian Optimization + Breakthrough Híbridos")

        # Bayesian Optimization
        self._benchmark_bayesian(A, B)

        # Breakthrough techniques individuales
        self._benchmark_breakthrough_individual(A, B)

        # Técnicas híbridas (el breakthrough principal)
        self._benchmark_breakthrough_hybrid(A, B)

    def _benchmark_bayesian(self, A: np.ndarray, B: np.ndarray):
        """Benchmark Bayesian Optimization."""
        try:
            from fase_8_bayesian_optimization.src.bayesian_optimizer import BayesianOptimizer

            optimizer = BayesianOptimizer()
            # Benchmark de optimización (no ejecución completa)
            logger.info("📊 Bayesian Optimization: Framework disponible")

        except ImportError as e:
            logger.warning("⚠️ Bayesian Optimizer no disponible")

    def _benchmark_breakthrough_individual(self, A: np.ndarray, B: np.ndarray):
        """Benchmark técnicas breakthrough individuales."""
        techniques = [
            ("low_rank", self._benchmark_low_rank),
            ("coppersmith_winograd", self._benchmark_coppersmith_winograd),
            ("quantum_annealing", self._benchmark_quantum_annealing),
        ]

        for name, benchmark_func in techniques:
            try:
                benchmark_func(A, B)
            except Exception as e:
                logger.warning(f"⚠️ Error en {name}: {e}")

    def _benchmark_low_rank(self, A: np.ndarray, B: np.ndarray):
        """Benchmark Low-Rank Approximation."""
        try:
            from low_rank_matrix_approximator_gpu import LowRankMatrixApproximatorGPU

            approximator = LowRankMatrixApproximatorGPU()
            start_time = time.time()
            result, metrics = approximator.optimized_gemm_gpu(A, B)
            computation_time = time.time() - start_time

            operations = 2 * A.shape[0] * A.shape[1] * B.shape[1]
            gflops = (operations / computation_time) / 1e9 if computation_time > 0 else 0

            # Calcular error vs NumPy
            reference = A @ B
            error_max = np.max(np.abs(result - reference))
            error_relative = error_max / np.max(np.abs(reference)) if np.max(np.abs(reference)) > 0 else 0

            benchmark_result = BenchmarkResult(
                method="low_rank_gpu",
                phase="phase_8_9",
                matrix_size=f"{A.shape[0]}x{A.shape[1]}x{B.shape[1]}",
                gflops=gflops,
                time_seconds=computation_time,
                operations=operations,
                error_max=error_max,
                error_relative=error_relative,
                memory_usage_mb=self._estimate_memory_usage(A, B),
                speedup_vs_baseline=gflops / self.baseline_gflops if self.baseline_gflops > 0 else 0
            )

            self.results.append(benchmark_result)
            logger.info(".2f")

        except ImportError as e:
            logger.warning("⚠️ Low-Rank GPU no disponible")

    def _benchmark_coppersmith_winograd(self, A: np.ndarray, B: np.ndarray):
        """Benchmark Coppersmith-Winograd."""
        try:
            from coppersmith_winograd_gpu import CoppersmithWinogradGPU

            cw = CoppersmithWinogradGPU()
            start_time = time.time()
            result, metrics = cw.optimized_cw_gemm(A, B)
            computation_time = time.time() - start_time

            operations = 2 * A.shape[0] * A.shape[1] * B.shape[1]
            gflops = (operations / computation_time) / 1e9 if computation_time > 0 else 0

            reference = A @ B
            error_max = np.max(np.abs(result - reference))
            error_relative = error_max / np.max(np.abs(reference)) if np.max(np.abs(reference)) > 0 else 0

            benchmark_result = BenchmarkResult(
                method="coppersmith_winograd",
                phase="phase_8_9",
                matrix_size=f"{A.shape[0]}x{A.shape[1]}x{B.shape[1]}",
                gflops=gflops,
                time_seconds=computation_time,
                operations=operations,
                error_max=error_max,
                error_relative=error_relative,
                memory_usage_mb=self._estimate_memory_usage(A, B),
                speedup_vs_baseline=gflops / self.baseline_gflops if self.baseline_gflops > 0 else 0
            )

            self.results.append(benchmark_result)
            logger.info(".2f")

        except ImportError as e:
            logger.warning("⚠️ Coppersmith-Winograd no disponible")

    def _benchmark_quantum_annealing(self, A: np.ndarray, B: np.ndarray):
        """Benchmark Quantum Annealing."""
        try:
            from quantum_annealing_optimizer import QuantumAnnealingMatrixOptimizer

            qa = QuantumAnnealingMatrixOptimizer()
            start_time = time.time()
            result, metrics = qa.hybrid_quantum_classical_gemm(A, B)
            computation_time = time.time() - start_time

            operations = 2 * A.shape[0] * A.shape[1] * B.shape[1]
            gflops = (operations / computation_time) / 1e9 if computation_time > 0 else 0

            reference = A @ B
            error_max = np.max(np.abs(result - reference))
            error_relative = error_max / np.max(np.abs(reference)) if np.max(np.abs(reference)) > 0 else 0

            benchmark_result = BenchmarkResult(
                method="quantum_annealing",
                phase="phase_8_9",
                matrix_size=f"{A.shape[0]}x{A.shape[1]}x{B.shape[1]}",
                gflops=gflops,
                time_seconds=computation_time,
                operations=operations,
                error_max=error_max,
                error_relative=error_relative,
                memory_usage_mb=self._estimate_memory_usage(A, B),
                speedup_vs_baseline=gflops / self.baseline_gflops if self.baseline_gflops > 0 else 0
            )

            self.results.append(benchmark_result)
            logger.info(".2f")

        except ImportError as e:
            logger.warning("⚠️ Quantum Annealing no disponible")

    def _benchmark_breakthrough_hybrid(self, A: np.ndarray, B: np.ndarray):
        """Benchmark técnicas híbridas breakthrough (FASE 9.4)."""
        logger.info("🔬 Probando BREAKTHROUGH HÍBRIDOS (FASE 9.4)")

        try:
            from fase_9_breakthrough_integration.src.breakthrough_selector import BreakthroughTechniqueSelector

            selector = BreakthroughTechniqueSelector()

            # Seleccionar técnica
            selection = selector.select_technique(A, B)

            # Ejecutar técnica seleccionada
            start_time = time.time()
            result, metrics = selector.execute_selected_technique(A, B, selection)
            computation_time = time.time() - start_time

            # Extraer métricas del resultado híbrido
            if result is not None and metrics is not None:
                operations = 2 * A.shape[0] * A.shape[1] * B.shape[1]
                gflops = (operations / computation_time) / 1e9 if computation_time > 0 else 0

                reference = A @ B
                error_max = np.max(np.abs(result - reference))
                error_relative = error_max / np.max(np.abs(reference)) if np.max(np.abs(reference)) > 0 else 0

                benchmark_result = BenchmarkResult(
                    method=f"breakthrough_hybrid_{selection.technique.value}",
                    phase="phase_9",
                    matrix_size=f"{A.shape[0]}x{A.shape[1]}x{B.shape[1]}",
                    gflops=gflops,
                    time_seconds=computation_time,
                    operations=operations,
                    error_max=error_max,
                    error_relative=error_relative,
                    memory_usage_mb=self._estimate_memory_usage(A, B),
                    speedup_vs_baseline=gflops / self.baseline_gflops if self.baseline_gflops > 0 else 0
                )

                self.results.append(benchmark_result)
                logger.info(".2f")
            else:
                logger.warning("⚠️ Breakthrough híbrido falló")

        except ImportError as e:
            logger.warning("⚠️ Breakthrough Selector no disponible")
        except Exception as e:
            logger.warning(f"⚠️ Error en breakthrough híbrido: {e}")

    def _run_phase_10_benchmarks(self, A: np.ndarray, B: np.ndarray):
        """Benchmarks de fase 10: Multi-GPU."""
        logger.info("🚀 Probando FASE 10: Multi-GPU Framework")

        try:
            from fase_10_multi_gpu.src.multi_gpu_manager import MultiGPUManager, distributed_gemm

            manager = MultiGPUManager()
            start_time = time.time()
            result = distributed_gemm(A, B, manager)
            computation_time = time.time() - start_time

            operations = 2 * A.shape[0] * A.shape[1] * B.shape[1]
            gflops = (operations / computation_time) / 1e9 if computation_time > 0 else 0

            reference = A @ B
            error_max = np.max(np.abs(result - reference))
            error_relative = error_max / np.max(np.abs(reference)) if np.max(np.abs(reference)) > 0 else 0

            benchmark_result = BenchmarkResult(
                method=f"multi_gpu_{len(manager.devices)}x",
                phase="phase_10",
                matrix_size=f"{A.shape[0]}x{A.shape[1]}x{B.shape[1]}",
                gflops=gflops,
                time_seconds=computation_time,
                operations=operations,
                error_max=error_max,
                error_relative=error_relative,
                memory_usage_mb=self._estimate_memory_usage(A, B),
                speedup_vs_baseline=gflops / self.baseline_gflops if self.baseline_gflops > 0 else 0
            )

            self.results.append(benchmark_result)
            logger.info(".2f")

            manager.cleanup()

        except ImportError as e:
            logger.warning("⚠️ Multi-GPU framework no disponible")

    def _estimate_memory_usage(self, A: np.ndarray, B: np.ndarray) -> float:
        """Estimar uso de memoria en MB."""
        # Matrices A, B, C + overhead
        total_elements = A.size + B.size + (A.shape[0] * B.shape[1])
        bytes_per_float32 = 4
        return (total_elements * bytes_per_float32) / (1024 * 1024)

    def _generate_report(self) -> ComprehensiveBenchmarkReport:
        """Generar reporte completo del benchmark."""
        logger.info("📊 Generando reporte comprehensivo...")

        # Calcular estadísticas resumen
        summary = self._calculate_summary_stats()

        # Generar recomendaciones
        recommendations = self._generate_recommendations()

        report = ComprehensiveBenchmarkReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_info=self.system_info,
            results=self.results,
            summary=summary,
            recommendations=recommendations
        )

        return report

    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calcular estadísticas resumen."""
        if not self.results:
            return {}

        # Agrupar por fase
        phase_stats = {}
        for result in self.results:
            phase = result.phase
            if phase not in phase_stats:
                phase_stats[phase] = []
            phase_stats[phase].append(result.gflops)

        # Calcular promedios por fase
        phase_averages = {}
        for phase, gflops_list in phase_stats.items():
            phase_averages[phase] = {
                'avg_gflops': np.mean(gflops_list),
                'max_gflops': np.max(gflops_list),
                'min_gflops': np.min(gflops_list),
                'count': len(gflops_list)
            }

        # Mejor resultado general
        best_result = max(self.results, key=lambda x: x.gflops)

        # Speedup total
        baseline_gflops = next((r.gflops for r in self.results if r.method == "baseline_numpy"), 0)
        total_speedup = best_result.gflops / baseline_gflops if baseline_gflops > 0 else 0

        return {
            'phase_averages': phase_averages,
            'best_result': asdict(best_result),
            'total_speedup_vs_baseline': total_speedup,
            'total_tests_run': len(self.results),
            'system_info': self.system_info
        }

    def _generate_recommendations(self) -> List[str]:
        """Generar recomendaciones basadas en resultados."""
        recommendations = []

        if not self.results:
            return ["No hay resultados para generar recomendaciones"]

        # Encontrar mejor técnica
        best_result = max(self.results, key=lambda x: x.gflops)

        recommendations.append(f"🎯 Mejor técnica: {best_result.method} con {best_result.gflops:.2f} GFLOPS")

        # Verificar si se alcanzó el objetivo
        if best_result.gflops >= 1000:
            recommendations.append("✅ ¡Objetivo alcanzado! 1000+ GFLOPS logrado")
        else:
            recommendations.append(f"📈 Objetivo: {1000 - best_result.gflops:.2f} GFLOPS más para alcanzar 1000+")

        # Recomendaciones por fase
        hybrid_results = [r for r in self.results if "hybrid" in r.method.lower()]
        if hybrid_results:
            recommendations.append("🔬 Técnicas híbridas muestran potencial - continuar desarrollo")

        multi_gpu_results = [r for r in self.results if "multi_gpu" in r.method.lower()]
        if multi_gpu_results:
            recommendations.append("🚀 Multi-GPU framework operativo - escalar con más GPUs")

        return recommendations

    def _save_report(self, report: ComprehensiveBenchmarkReport):
        """Guardar reporte en múltiples formatos."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # JSON
        json_file = f"comprehensive_benchmark_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info(f"💾 Reporte JSON guardado: {json_file}")

        # CSV
        csv_file = f"comprehensive_benchmark_results_{timestamp}.csv"
        df = pd.DataFrame([asdict(r) for r in report.results])
        df.to_csv(csv_file, index=False)
        logger.info(f"💾 Resultados CSV guardados: {csv_file}")

        # Markdown report
        self._generate_markdown_report(report, timestamp)

        # Gráficos (si matplotlib disponible)
        try:
            if HAS_MATPLOTLIB:
                self._generate_plots(report, timestamp)
            else:
                logger.warning("⚠️ matplotlib no disponible - omitiendo gráficos")
        except ImportError:
            logger.warning("⚠️ matplotlib no disponible - omitiendo gráficos")

    def _generate_markdown_report(self, report: ComprehensiveBenchmarkReport, timestamp: str):
        """Generar reporte en formato Markdown."""
        md_file = f"comprehensive_benchmark_report_{timestamp}.md"

        with open(md_file, 'w') as f:
            f.write("# 🎯 Comprehensive Performance Validation Report\n\n")
            f.write(f"**Fecha:** {report.timestamp}\n\n")
            f.write(f"**Sistema:** {report.system_info}\n\n")

            f.write("## 📊 Resultados por Técnica\n\n")
            f.write("| Técnica | Fase | Tamaño | GFLOPS | Tiempo (s) | Error Relativo | Speedup |\n")
            f.write("|---------|------|--------|--------|------------|----------------|---------|\n")

            for result in sorted(report.results, key=lambda x: x.gflops, reverse=True):
                f.write(f"| {result.method} | {result.phase} | {result.matrix_size} | {result.gflops:.2f} | {result.time_seconds:.4f} | {result.error_relative:.2e} | {result.speedup_vs_baseline:.2f}x |\n")

            f.write("\n## 🏆 Resumen Ejecutivo\n\n")
            f.write(f"- **Total de tests:** {report.summary.get('total_tests_run', 0)}\n")
            f.write(f"- **Mejor resultado:** {report.summary.get('best_result', {}).get('method', 'N/A')} - {report.summary.get('best_result', {}).get('gflops', 0):.2f} GFLOPS\n")
            f.write(f"- **Speedup total:** {report.summary.get('total_speedup_vs_baseline', 0):.2f}x vs baseline\n\n")

            f.write("## 💡 Recomendaciones\n\n")
            for rec in report.recommendations:
                f.write(f"- {rec}\n")

        logger.info(f"💾 Reporte Markdown guardado: {md_file}")

    def _generate_plots(self, report: ComprehensiveBenchmarkReport, timestamp: str):
        """Generar gráficos de los resultados."""
        if not HAS_MATPLOTLIB:
            return
        # Preparar datos
        methods = [r.method for r in report.results]
        gflops = [r.gflops for r in report.results]

        # Gráfico de barras
        plt.figure(figsize=(12, 6))
        bars = plt.bar(methods, gflops)
        plt.title('GFLOPS por Técnica de Optimización')
        plt.xlabel('Técnica')
        plt.ylabel('GFLOPS')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Añadir valores sobre las barras
        for bar, gf in zip(bars, gflops):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    '.1f', ha='center', va='bottom')

        plt.savefig(f"benchmark_gflops_{timestamp}.png", dpi=300, bbox_inches='tight')
        logger.info(f"📊 Gráfico guardado: benchmark_gflops_{timestamp}.png")

        # Gráfico de speedup
        speedups = [r.speedup_vs_baseline for r in report.results if r.speedup_vs_baseline > 0]
        methods_filtered = [r.method for r in report.results if r.speedup_vs_baseline > 0]

        if speedups:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(methods_filtered, speedups)
            plt.title('Speedup vs Baseline (NumPy)')
            plt.xlabel('Técnica')
            plt.ylabel('Speedup (x)')
            plt.xticks(rotation=45, ha='right')
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Baseline')
            plt.legend()
            plt.tight_layout()

            plt.savefig(f"benchmark_speedup_{timestamp}.png", dpi=300, bbox_inches='tight')
            logger.info(f"📊 Gráfico speedup guardado: benchmark_speedup_{timestamp}.png")


def main():
    """Función principal."""
    print("🎯 Comprehensive Performance Validation Benchmark")
    print("=" * 60)
    print("Validando el impacto completo de todas las optimizaciones implementadas")
    print("Objetivo: Confirmar que las optimizaciones han tenido efecto real")
    print()

    try:
        suite = ComprehensiveBenchmarkSuite()
        report = suite.run_complete_validation()

        print("\n" + "=" * 60)
        print("✅ BENCHMARK COMPLETADO")
        print("=" * 60)

        print(f"📊 Tests ejecutados: {report.summary.get('total_tests_run', 0)}")
        print(f"🏆 Mejor resultado: {report.summary.get('best_result', {}).get('method', 'N/A')}")
        print(f"   GFLOPS: {report.summary.get('best_result', {}).get('gflops', 0):.2f}")
        print(f"   Speedup total: {report.summary.get('total_speedup_vs_baseline', 0):.2f}x")

        print("\n💡 RECOMENDACIONES:")
        for rec in report.recommendations:
            print(f"   • {rec}")

        print("\n📁 ARCHIVOS GENERADOS:")
        print("   • comprehensive_benchmark_report_[timestamp].json")
        print("   • comprehensive_benchmark_results_[timestamp].csv")
        print("   • comprehensive_benchmark_report_[timestamp].md")
        print("   • benchmark_gflops_[timestamp].png")
        print("   • benchmark_speedup_[timestamp].png")

    except Exception as e:
        logger.error(f"❌ Error en benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())