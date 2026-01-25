#!/usr/bin/env python3
"""
🔄 HYBRID OPTIMIZER - COMBINACIÓN DE TÉCNICAS BREAKTHROUGH
==========================================================

Optimizador híbrido que combina múltiples técnicas de breakthrough
para lograr el máximo rendimiento posible.

Técnicas soportadas:
- Low-Rank + Coppersmith-Winograd
- Quantum Annealing + Low-Rank
- Multi-stage optimization

Autor: AI Assistant
Fecha: 2026-01-25
"""

import sys
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Importar técnicas de breakthrough
try:
    breakthrough_path = Path(__file__).parent.parent.parent
    sys.path.append(str(breakthrough_path))

    from low_rank_matrix_approximator_gpu import GPUAcceleratedLowRankApproximator
    from coppersmith_winograd_gpu import CoppersmithWinogradGPU
    from quantum_annealing_optimizer import QuantumAnnealingMatrixOptimizer

    TECHNIQUES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Técnicas de breakthrough no disponibles: {e}")
    TECHNIQUES_AVAILABLE = False


class HybridStrategy(Enum):
    """Estrategias de optimización híbrida."""
    SEQUENTIAL = "sequential"      # Aplicar técnicas en secuencia
    PARALLEL = "parallel"          # Ejecutar en paralelo y seleccionar mejor
    ADAPTIVE = "adaptive"          # Adaptar basado en resultados intermedios
    CASCADE = "cascade"            # Aplicar una técnica sobre el resultado de otra


@dataclass
class HybridConfiguration:
    """Configuración para optimización híbrida."""
    strategy: HybridStrategy
    techniques: List[str]  # Lista de técnicas a combinar
    parameters: Dict[str, Any]  # Parámetros para cada técnica
    weights: Dict[str, float]  # Pesos para combinación de resultados
    stopping_criteria: Dict[str, Any]  # Criterios para detener optimización


@dataclass
class HybridResult:
    """Resultado de optimización híbrida."""
    final_result: np.ndarray
    technique_results: Dict[str, Tuple[np.ndarray, Dict[str, Any]]]
    total_time: float
    combined_performance: float
    quality_metrics: Dict[str, Any]
    optimization_path: List[str]  # Secuencia de técnicas aplicadas


class HybridOptimizer:
    """
    Optimizador híbrido que combina múltiples técnicas de breakthrough.

    Estrategias soportadas:
    - Sequential: Aplicar técnicas en orden específico
    - Parallel: Ejecutar todas en paralelo y seleccionar mejor resultado
    - Adaptive: Modificar estrategia basado en resultados parciales
    - Cascade: Usar resultado de una técnica como entrada para otra
    """

    def __init__(self):
        """Inicializa el optimizador híbrido."""
        # Configurar logging primero
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.technique_implementations = {}
        self._load_techniques()

    def _load_techniques(self):
        """Carga las implementaciones de técnicas disponibles."""
        if not TECHNIQUES_AVAILABLE:
            self.logger.warning("Técnicas de breakthrough no disponibles")
            return

        try:
            self.technique_implementations['low_rank'] = GPUAcceleratedLowRankApproximator()
            self.logger.info("✅ Low-Rank Approximator cargado")
        except Exception as e:
            self.logger.warning(f"Error cargando Low-Rank: {e}")

        try:
            self.technique_implementations['cw'] = CoppersmithWinogradGPU()
            self.logger.info("✅ Coppersmith-Winograd cargado")
        except Exception as e:
            self.logger.warning(f"Error cargando CW: {e}")

        try:
            self.technique_implementations['quantum'] = QuantumAnnealingMatrixOptimizer()
            self.logger.info("✅ Quantum Annealing cargado")
        except Exception as e:
            self.logger.warning(f"Error cargando Quantum: {e}")

    def optimize_hybrid(self,
                       matrix_a: np.ndarray,
                       matrix_b: np.ndarray,
                       config: HybridConfiguration) -> HybridResult:
        """
        Ejecuta optimización híbrida según la configuración especificada.

        Args:
            matrix_a, matrix_b: Matrices de entrada
            config: Configuración de optimización híbrida

        Returns:
            Resultado de la optimización híbrida
        """
        start_time = time.time()

        self.logger.info(f"Iniciando optimización híbrida: {config.strategy.value}")
        self.logger.info(f"Técnicas: {config.techniques}")

        if config.strategy == HybridStrategy.SEQUENTIAL:
            result = self._optimize_sequential(matrix_a, matrix_b, config)

        elif config.strategy == HybridStrategy.PARALLEL:
            result = self._optimize_parallel(matrix_a, matrix_b, config)

        elif config.strategy == HybridStrategy.ADAPTIVE:
            result = self._optimize_adaptive(matrix_a, matrix_b, config)

        elif config.strategy == HybridStrategy.CASCADE:
            result = self._optimize_cascade(matrix_a, matrix_b, config)

        else:
            raise ValueError(f"Estrategia no soportada: {config.strategy}")

        result.total_time = time.time() - start_time

        # Calcular métricas finales
        result.combined_performance = self._calculate_combined_performance(result)
        result.quality_metrics = self._calculate_quality_metrics(result, matrix_a, matrix_b)

        self.logger.info(f"Optimización híbrida completada en {result.total_time:.3f}s")
        self.logger.info(f"Performance combinada: {result.combined_performance:.2f} GFLOPS")

        return result

    def _optimize_sequential(self,
                           matrix_a: np.ndarray,
                           matrix_b: np.ndarray,
                           config: HybridConfiguration) -> HybridResult:
        """
        Optimización secuencial: aplicar técnicas en orden.
        """
        technique_results = {}
        optimization_path = []

        current_a, current_b = matrix_a, matrix_b

        for technique_name in config.techniques:
            if technique_name not in self.technique_implementations:
                self.logger.warning(f"Técnica {technique_name} no disponible, saltando")
                continue

            self.logger.info(f"Aplicando técnica secuencial: {technique_name}")

            technique = self.technique_implementations[technique_name]
            params = config.parameters.get(technique_name, {})

            try:
                if technique_name == 'low_rank':
                    # El método acepta 'target_rank', no 'rank_target'
                    params_copy = params.copy()
                    if 'rank_target' in params_copy:
                        params_copy['target_rank'] = params_copy.pop('rank_target')
                    result, metrics = technique.optimized_gemm_gpu(current_a, current_b, **params_copy)
                elif technique_name == 'cw':
                    result, metrics = technique.cw_matrix_multiply_gpu(current_a, current_b)
                elif technique_name == 'quantum':
                    result, metrics = technique.quantum_annealing_optimization(
                        current_a, current_b, **params)

                technique_results[technique_name] = (result, metrics)
                optimization_path.append(technique_name)

                # Para siguiente iteración, usar resultado como entrada
                # (Esto es conceptual - en la práctica depende de la técnica)
                current_a, current_b = result, current_b

            except Exception as e:
                self.logger.error(f"Error en técnica {technique_name}: {e}")
                continue

        # Combinar resultados usando pesos
        final_result = self._combine_results_weighted(technique_results, config.weights)

        return HybridResult(
            final_result=final_result,
            technique_results=technique_results,
            total_time=0,  # Se calcula después
            combined_performance=0,  # Se calcula después
            quality_metrics={},  # Se calcula después
            optimization_path=optimization_path
        )

    def _optimize_parallel(self,
                         matrix_a: np.ndarray,
                         matrix_b: np.ndarray,
                         config: HybridConfiguration) -> HybridResult:
        """
        Optimización paralela: ejecutar todas las técnicas en paralelo.
        """
        technique_results = {}
        optimization_path = config.techniques.copy()  # Todas se ejecutan

        def execute_technique(technique_name):
            if technique_name not in self.technique_implementations:
                return technique_name, (None, {'error': 'Technique not available'})
            
            technique = self.technique_implementations[technique_name]
            params = config.parameters.get(technique_name, {})
            
            try:
                if technique_name == 'low_rank':
                    # El método acepta 'target_rank', no 'rank_target'
                    params_copy = params.copy()
                    if 'rank_target' in params_copy:
                        params_copy['target_rank'] = params_copy.pop('rank_target')
                    result, metrics = technique.optimized_gemm_gpu(matrix_a, matrix_b, **params_copy)
                elif technique_name == 'cw':
                    result, metrics = technique.cw_matrix_multiply_gpu(matrix_a, matrix_b)
                elif technique_name == 'quantum':
                    result, metrics = technique.quantum_annealing_optimization(matrix_a, matrix_b, **params)
                else:
                    return technique_name, (None, {'error': 'Unknown technique'})
                
                return technique_name, (result, metrics)
            except Exception as e:
                return technique_name, (None, {'error': str(e)})

        # Ejecutar en paralelo
        with ThreadPoolExecutor(max_workers=len(config.techniques)) as executor:
            futures = [executor.submit(execute_technique, tech) for tech in config.techniques]

            for future in as_completed(futures):
                technique_name, result_metrics = future.result()
                technique_results[technique_name] = result_metrics

        # Seleccionar mejor resultado basado en performance
        best_technique = max(
            [(k, v) for k, v in technique_results.items() if v[1].get('gflops_achieved', 0) > 0],
            key=lambda x: x[1][1].get('gflops_achieved', 0),
            default=None
        )

        if best_technique:
            final_result = best_technique[1][0]
            self.logger.info(f"Mejor técnica paralela: {best_technique[0]}")
        else:
            # Fallback a multiplicación NumPy
            final_result = matrix_a @ matrix_b
            self.logger.warning("Ninguna técnica paralela funcionó, usando fallback")

        return HybridResult(
            final_result=final_result,
            technique_results=technique_results,
            total_time=0,
            combined_performance=0,
            quality_metrics={},
            optimization_path=optimization_path
        )

    def _optimize_adaptive(self,
                         matrix_a: np.ndarray,
                         matrix_b: np.ndarray,
                         config: HybridConfiguration) -> HybridResult:
        """
        Optimización adaptativa: modificar estrategia basado en resultados.
        """
        technique_results = {}
        optimization_path = []

        # Análisis inicial de matrices
        matrix_analysis = self._analyze_matrices(matrix_a, matrix_b)

        # Seleccionar técnicas basado en análisis
        selected_techniques = self._select_adaptive_techniques(matrix_analysis, config.techniques)

        self.logger.info(f"Técnicas seleccionadas adaptativamente: {selected_techniques}")

        # Ejecutar técnicas seleccionadas
        for technique_name in selected_techniques:
            if technique_name not in self.technique_implementations:
                continue

            technique = self.technique_implementations[technique_name]
            params = config.parameters.get(technique_name, {})

            try:
                if technique_name == 'low_rank':
                    # El método acepta 'target_rank', no 'rank_target'
                    params_copy = params.copy()
                    if 'rank_target' in params_copy:
                        params_copy['target_rank'] = params_copy.pop('rank_target')
                    result, metrics = technique.optimized_gemm_gpu(matrix_a, matrix_b, **params_copy)
                elif technique_name == 'cw':
                    result, metrics = technique.cw_matrix_multiply_gpu(matrix_a, matrix_b)
                elif technique_name == 'quantum':
                    result, metrics = technique.quantum_annealing_optimization(
                        matrix_a, matrix_b, **params)

                technique_results[technique_name] = (result, metrics)
                optimization_path.append(technique_name)

                # Verificar criterios de parada
                if self._check_stopping_criteria(metrics, config.stopping_criteria):
                    self.logger.info(f"Criterio de parada alcanzado con {technique_name}")
                    break

            except Exception as e:
                self.logger.error(f"Error en técnica adaptativa {technique_name}: {e}")
                continue

        # Seleccionar mejor resultado
        if technique_results:
            best_result = max(
                technique_results.values(),
                key=lambda x: x[1].get('gflops_achieved', 0)
            )[0]
        else:
            best_result = matrix_a @ matrix_b

        return HybridResult(
            final_result=best_result,
            technique_results=technique_results,
            total_time=0,
            combined_performance=0,
            quality_metrics={},
            optimization_path=optimization_path
        )

    def _optimize_cascade(self,
                        matrix_a: np.ndarray,
                        matrix_b: np.ndarray,
                        config: HybridConfiguration) -> HybridResult:
        """
        Optimización en cascada: resultado de una técnica alimenta a la siguiente.
        """
        technique_results = {}
        optimization_path = []

        current_result = None
        current_metrics = None

        for i, technique_name in enumerate(config.techniques):
            if technique_name not in self.technique_implementations:
                continue

            technique = self.technique_implementations[technique_name]

            # Para la primera técnica, usar matrices originales
            if i == 0:
                input_a, input_b = matrix_a, matrix_b
            else:
                # Para técnicas siguientes, usar resultado anterior como entrada
                # (Esto requiere adaptación - conceptual)
                input_a, input_b = current_result, matrix_b

            params = config.parameters.get(technique_name, {})

            try:
                if technique_name == 'low_rank':
                    # El método acepta 'target_rank', no 'rank_target'
                    params_copy = params.copy()
                    if 'rank_target' in params_copy:
                        params_copy['target_rank'] = params_copy.pop('rank_target')
                    result, metrics = technique.optimized_gemm_gpu(input_a, input_b, **params_copy)
                elif technique_name == 'cw':
                    result, metrics = technique.cw_matrix_multiply_gpu(input_a, input_b)
                elif technique_name == 'quantum':
                    result, metrics = technique.quantum_annealing_optimization(
                        input_a, input_b, **params)

                technique_results[technique_name] = (result, metrics)
                optimization_path.append(technique_name)

                current_result = result
                current_metrics = metrics

            except Exception as e:
                self.logger.error(f"Error en cascada {technique_name}: {e}")
                continue

        final_result = current_result if current_result is not None else matrix_a @ matrix_b

        return HybridResult(
            final_result=final_result,
            technique_results=technique_results,
            total_time=0,
            combined_performance=0,
            quality_metrics={},
            optimization_path=optimization_path
        )

    def _analyze_matrices(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> Dict[str, Any]:
        """Analiza características de las matrices para selección adaptativa."""
        analysis = {
            'size_a': matrix_a.shape,
            'size_b': matrix_b.shape,
            'rank_a': np.linalg.matrix_rank(matrix_a),
            'rank_b': np.linalg.matrix_rank(matrix_b),
            'sparsity_a': 1.0 - np.count_nonzero(matrix_a) / matrix_a.size,
            'sparsity_b': 1.0 - np.count_nonzero(matrix_b) / matrix_b.size,
        }

        analysis['rank_ratio_a'] = analysis['rank_a'] / analysis['size_a'][0]
        analysis['rank_ratio_b'] = analysis['rank_b'] / analysis['size_b'][0]

        return analysis

    def _select_adaptive_techniques(self, analysis: Dict[str, Any],
                                  available_techniques: List[str]) -> List[str]:
        """Selecciona técnicas basado en análisis adaptativo."""
        selected = []

        # Priorizar low-rank si las matrices tienen bajo rango efectivo
        avg_rank_ratio = (analysis['rank_ratio_a'] + analysis['rank_ratio_b']) / 2
        if avg_rank_ratio < 0.7 and 'low_rank' in available_techniques:
            selected.append('low_rank')

        # Priorizar CW para matrices grandes
        min_size = min(analysis['size_a'] + analysis['size_b'])
        if min_size >= 512 and 'cw' in available_techniques:
            selected.append('cw')

        # Quantum annealing como último recurso
        if 'quantum' in available_techniques and not selected:
            selected.append('quantum')

        # Si no se seleccionó nada, usar todas disponibles
        if not selected:
            selected = available_techniques[:2]  # Máximo 2 para evitar sobrecarga

        return selected

    def _check_stopping_criteria(self, metrics: Dict[str, Any],
                               stopping_criteria: Dict[str, Any]) -> bool:
        """Verifica si se cumplen criterios de parada."""
        if not stopping_criteria:
            return False

        # Criterio de performance mínima
        min_performance = stopping_criteria.get('min_gflops', 0)
        achieved = metrics.get('gflops_achieved', 0)
        if achieved >= min_performance:
            return True

        # Criterio de error máximo
        max_error = stopping_criteria.get('max_error', 1.0)
        error = metrics.get('relative_error', 0)
        if error <= max_error:
            return True

        return False

    def _combine_results_weighted(self, technique_results: Dict[str, Tuple[np.ndarray, Dict[str, Any]]],
                                weights: Dict[str, float]) -> np.ndarray:
        """Combina resultados de múltiples técnicas usando pesos."""
        if not technique_results:
            raise ValueError("No hay resultados para combinar")

        # Normalizar pesos
        total_weight = sum(weights.get(tech, 1.0) for tech in technique_results.keys())
        if total_weight == 0:
            total_weight = len(technique_results)

        normalized_weights = {tech: weights.get(tech, 1.0) / total_weight
                            for tech in technique_results.keys()}

        # Combinar resultados ponderados
        combined = None
        for tech, (result, _) in technique_results.items():
            if result is not None:
                weight = normalized_weights[tech]
                if combined is None:
                    combined = weight * result
                else:
                    combined += weight * result

        return combined if combined is not None else np.zeros_like(list(technique_results.values())[0][0])

    def _calculate_combined_performance(self, result: HybridResult) -> float:
        """Calcula performance combinada de todas las técnicas."""
        if not result.technique_results:
            return 0.0

        # Performance promedio ponderada
        total_performance = 0
        total_weight = 0

        for tech, (_, metrics) in result.technique_results.items():
            perf = metrics.get('gflops_achieved', 0)
            # Peso basado en tiempo de ejecución (técnicas más rápidas tienen más peso)
            time_weight = 1.0 / (metrics.get('computation_time', 1.0) + 0.1)

            total_performance += perf * time_weight
            total_weight += time_weight

        return total_performance / total_weight if total_weight > 0 else 0.0

    def _calculate_quality_metrics(self, result: HybridResult,
                                 original_a: np.ndarray,
                                 original_b: np.ndarray) -> Dict[str, Any]:
        """Calcula métricas de calidad para el resultado híbrido."""
        reference = original_a @ original_b

        error = np.linalg.norm(result.final_result - reference, 'fro')
        relative_error = error / np.linalg.norm(reference, 'fro')

        # Calcular speedup vs NumPy
        numpy_time = self._benchmark_numpy(original_a, original_b)
        speedup = numpy_time / result.total_time if result.total_time > 0 else 1.0

        return {
            'relative_error': relative_error,
            'speedup_vs_numpy': speedup,
            'techniques_used': len(result.technique_results),
            'optimization_depth': len(result.optimization_path)
        }

    def _benchmark_numpy(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
        """Benchmark de NumPy para comparación."""
        import time
        start = time.time()
        _ = matrix_a @ matrix_b
        return time.time() - start

    def create_optimized_hybrid_config(self,
                                     matrix_characteristics: Dict[str, Any],
                                     available_techniques: List[str]) -> HybridConfiguration:
        """
        Crea configuración híbrida optimizada basada en características de matrices.

        Args:
            matrix_characteristics: Características de las matrices
            available_techniques: Técnicas disponibles

        Returns:
            Configuración optimizada
        """
        # Análisis de características
        size = matrix_characteristics.get('size', 512)
        rank_ratio = matrix_characteristics.get('rank_ratio', 0.8)
        sparsity = matrix_characteristics.get('sparsity', 0.0)

        # Seleccionar estrategia basada en características
        if rank_ratio < 0.6 and 'low_rank' in available_techniques:
            # Matrices de bajo rango - usar cascada con low-rank primero
            strategy = HybridStrategy.CASCADE
            techniques = ['low_rank', 'cw'] if 'cw' in available_techniques else ['low_rank']
        elif size >= 1024 and 'cw' in available_techniques:
            # Matrices grandes - usar CW con fallback
            strategy = HybridStrategy.ADAPTIVE
            techniques = ['cw', 'low_rank']
        else:
            # Caso general - usar paralelo
            strategy = HybridStrategy.PARALLEL
            techniques = available_techniques[:3]  # Máximo 3 técnicas

        # Configurar parámetros
        parameters = {}
        weights = {}

        for tech in techniques:
            if tech == 'low_rank':
                parameters[tech] = {'rank_target': int(min(size * rank_ratio, size // 2))}
                weights[tech] = 1.5 if rank_ratio < 0.7 else 1.0
            elif tech == 'cw':
                parameters[tech] = {}
                weights[tech] = 1.2
            elif tech == 'quantum':
                parameters[tech] = {'num_sweeps': 30}  # Menos sweeps para híbrido
                weights[tech] = 0.8

        # Criterios de parada
        stopping_criteria = {
            'min_gflops': 1.0,  # Al menos 1 GFLOPS
            'max_error': 0.1    # Máximo 10% de error
        }

        return HybridConfiguration(
            strategy=strategy,
            techniques=techniques,
            parameters=parameters,
            weights=weights,
            stopping_criteria=stopping_criteria
        )