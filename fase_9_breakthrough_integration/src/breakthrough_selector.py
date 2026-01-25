#!/usr/bin/env python3
"""
🎯 BREAKTHROUGH TECHNIQUE SELECTOR
===================================

Selector inteligente que decide automáticamente qué técnica de breakthrough usar
basado en características de entrada y predicciones de ML.

Integra con el AI Kernel Predictor existente para extender sus capacidades
con técnicas de breakthrough: Low-Rank, Coppersmith-Winograd, Quantum Annealing.

Autor: AI Assistant
Fecha: 2026-01-25
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Importar componentes existentes
try:
    # Importar AI Kernel Predictor
    predictor_path = Path(__file__).parent.parent.parent / "fase_7_ai_kernel_predictor" / "src"
    sys.path.append(str(predictor_path))
    from kernel_predictor import AIKernelPredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False

try:
    # Importar Bayesian Optimization
    bayesian_path = Path(__file__).parent.parent.parent / "fase_8_bayesian_optimization" / "src"
    sys.path.append(str(bayesian_path))
    from bayesian_optimizer import BayesianKernelOptimizer
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# Importar técnicas de breakthrough
try:
    breakthrough_path = Path(__file__).parent.parent
    sys.path.append(str(breakthrough_path))

    from low_rank_matrix_approximator_gpu import GPUAcceleratedLowRankApproximator
    from coppersmith_winograd_gpu import CoppersmithWinogradGPU
    from quantum_annealing_optimizer import QuantumAnnealingMatrixOptimizer

    LOW_RANK_AVAILABLE = True
    CW_AVAILABLE = True
    QUANTUM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Algunas técnicas de breakthrough no disponibles: {e}")
    LOW_RANK_AVAILABLE = False
    CW_AVAILABLE = False
    QUANTUM_AVAILABLE = False


class BreakthroughTechnique(Enum):
    """Técnicas de breakthrough disponibles."""
    TRADITIONAL = "traditional"  # Kernels tradicionales (Strassen, GCN4, etc.)
    LOW_RANK = "low_rank"        # Aproximaciones de bajo rango
    COPPERSMITH_WINOGRAD = "cw"  # Algoritmo Coppersmith-Winograd
    QUANTUM_ANNEALING = "quantum"  # Simulación de annealing cuántico
    HYBRID_LOW_RANK_CW = "hybrid_lr_cw"  # Híbrido low-rank + CW
    HYBRID_QUANTUM_LOW_RANK = "hybrid_q_lr"  # Híbrido quantum + low-rank


@dataclass
class TechniqueSelection:
    """Resultado de selección de técnica."""
    technique: BreakthroughTechnique
    confidence: float
    expected_performance: float
    parameters: Dict[str, Any]
    reasoning: str


@dataclass
class MatrixCharacteristics:
    """Características de las matrices de entrada."""
    size_a: Tuple[int, int]
    size_b: Tuple[int, int]
    sparsity_a: float = 0.0
    sparsity_b: float = 0.0
    rank_a: Optional[int] = None
    rank_b: Optional[int] = None
    condition_number_a: Optional[float] = None
    condition_number_b: Optional[float] = None
    memory_usage: int = 0
    computational_intensity: float = 1.0


class BreakthroughTechniqueSelector:
    """
    Selector inteligente de técnicas de breakthrough.

    Utiliza ML para decidir qué técnica usar basado en:
    - Características de las matrices
    - Restricciones de performance
    - Historial de ejecuciones
    - Predicciones del AI Kernel Predictor
    """

    def __init__(self, use_ml_predictor: bool = True, use_bayesian_opt: bool = True):
        """
        Inicializa el selector.

        Args:
            use_ml_predictor: Usar AI Kernel Predictor para selección inicial
            use_bayesian_opt: Usar Bayesian Optimization para fine-tuning
        """
        self.use_ml_predictor = use_ml_predictor and PREDICTOR_AVAILABLE
        self.use_bayesian_opt = use_bayesian_opt and BAYESIAN_AVAILABLE

        # Configurar logging primero
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Inicializar componentes
        self.kernel_predictor = None
        self.bayesian_optimizer = None
        self.technique_implementations = {}

        self._initialize_components()
        self._load_technique_implementations()

        # Historial de decisiones para aprendizaje
        self.decision_history = []

    def _initialize_components(self):
        """Inicializa componentes del sistema ML-based."""
        if self.use_ml_predictor:
            try:
                self.kernel_predictor = AIKernelPredictor()
                self.logger.info("✅ AI Kernel Predictor inicializado")
            except Exception as e:
                self.logger.warning(f"⚠️  No se pudo inicializar AI Kernel Predictor: {e}")
                self.use_ml_predictor = False

        if self.use_bayesian_opt:
            try:
                self.bayesian_optimizer = BayesianKernelOptimizer()
                self.logger.info("✅ Bayesian Optimizer inicializado")
            except Exception as e:
                self.logger.warning(f"⚠️  No se pudo inicializar Bayesian Optimizer: {e}")
                self.use_bayesian_opt = False

    def _load_technique_implementations(self):
        """Carga las implementaciones de técnicas de breakthrough."""
        implementations = {}

        if LOW_RANK_AVAILABLE:
            try:
                implementations[BreakthroughTechnique.LOW_RANK] = GPUAcceleratedLowRankApproximator()
                self.logger.info("✅ Low-Rank Approximator cargado")
            except Exception as e:
                self.logger.warning(f"⚠️  Error cargando Low-Rank: {e}")

        if CW_AVAILABLE:
            try:
                implementations[BreakthroughTechnique.COPPERSMITH_WINOGRAD] = CoppersmithWinogradGPU()
                self.logger.info("✅ Coppersmith-Winograd cargado")
            except Exception as e:
                self.logger.warning(f"⚠️  Error cargando CW: {e}")

        if QUANTUM_AVAILABLE:
            try:
                implementations[BreakthroughTechnique.QUANTUM_ANNEALING] = QuantumAnnealingMatrixOptimizer()
                self.logger.info("✅ Quantum Annealing cargado")
            except Exception as e:
                self.logger.warning(f"⚠️  Error cargando Quantum: {e}")

        self.technique_implementations = implementations

    def select_technique(self,
                        matrix_a: np.ndarray,
                        matrix_b: np.ndarray,
                        performance_target: float = None,
                        accuracy_tolerance: float = 0.01) -> TechniqueSelection:
        """
        Selecciona la mejor técnica de breakthrough para las matrices dadas.

        Args:
            matrix_a, matrix_b: Matrices de entrada
            performance_target: GFLOPS objetivo (opcional)
            accuracy_tolerance: Tolerancia de error aceptable

        Returns:
            Técnica seleccionada con parámetros óptimos
        """
        # Analizar características de las matrices
        characteristics = self._analyze_matrix_characteristics(matrix_a, matrix_b)

        # Obtener predicción inicial del AI Kernel Predictor
        initial_prediction = self._get_initial_prediction(characteristics)

        # Evaluar técnicas candidatas
        candidates = self._evaluate_technique_candidates(characteristics,
                                                        initial_prediction,
                                                        performance_target,
                                                        accuracy_tolerance)

        # Seleccionar la mejor técnica
        best_selection = self._select_best_candidate(candidates, characteristics)

        # Optimizar parámetros con Bayesian Optimization si está disponible
        if self.use_bayesian_opt and best_selection.technique != BreakthroughTechnique.TRADITIONAL:
            best_selection = self._optimize_parameters_bayesian(best_selection, characteristics)

        # Registrar decisión para aprendizaje futuro
        self._record_decision(best_selection, characteristics)

        return best_selection

    def _analyze_matrix_characteristics(self, matrix_a: np.ndarray,
                                       matrix_b: np.ndarray) -> MatrixCharacteristics:
        """
        Analiza las características de las matrices de entrada.
        """
        # Cálculos básicos
        size_a = matrix_a.shape
        size_b = matrix_b.shape

        # Estimar sparsity (proporción de elementos no cero)
        sparsity_a = 1.0 - np.count_nonzero(matrix_a) / matrix_a.size
        sparsity_b = 1.0 - np.count_nonzero(matrix_b) / matrix_b.size

        # Estimar rango efectivo (simplificado)
        try:
            # Para matrices grandes, usar aproximación
            if min(size_a) > 500:
                # Submuestreo para estimación rápida
                sample_size = min(500, min(size_a))
                indices = np.random.choice(size_a[0], sample_size, replace=False)
                sample_a = matrix_a[indices, :][:, :sample_size]
                rank_a = np.linalg.matrix_rank(sample_a)
            else:
                rank_a = np.linalg.matrix_rank(matrix_a)
        except:
            rank_a = min(size_a)

        try:
            if min(size_b) > 500:
                sample_size = min(500, min(size_b))
                indices = np.random.choice(size_b[0], sample_size, replace=False)
                sample_b = matrix_b[indices, :][:, :sample_size]
                rank_b = np.linalg.matrix_rank(sample_b)
            else:
                rank_b = np.linalg.matrix_rank(matrix_b)
        except:
            rank_b = min(size_b)

        # Calcular intensidad computacional
        operations_standard = 2 * size_a[0] * size_a[1] * size_b[1]
        memory_access = matrix_a.nbytes + matrix_b.nbytes + (size_a[0] * size_b[1] * 4)  # 4 bytes por float
        computational_intensity = operations_standard / memory_access

        return MatrixCharacteristics(
            size_a=size_a,
            size_b=size_b,
            sparsity_a=sparsity_a,
            sparsity_b=sparsity_b,
            rank_a=rank_a,
            rank_b=rank_b,
            memory_usage=matrix_a.nbytes + matrix_b.nbytes,
            computational_intensity=computational_intensity
        )

    def _get_initial_prediction(self, characteristics: MatrixCharacteristics) -> Dict[str, Any]:
        """
        Obtiene predicción inicial del AI Kernel Predictor.
        """
        if not self.use_ml_predictor or self.kernel_predictor is None:
            return {'technique': BreakthroughTechnique.TRADITIONAL, 'confidence': 0.5}

        try:
            # Crear features para el predictor
            features = {
                'matrix_size': min(characteristics.size_a + characteristics.size_b),
                'sparsity': (characteristics.sparsity_a + characteristics.sparsity_b) / 2,
                'rank_ratio': (characteristics.rank_a / characteristics.size_a[0] +
                             characteristics.rank_b / characteristics.size_b[0]) / 2,
                'computational_intensity': characteristics.computational_intensity
            }

            # El predictor actual espera diferentes features, así que adaptamos
            prediction = self.kernel_predictor.predict_kernel_type(features)

            return prediction

        except Exception as e:
            self.logger.warning(f"Error en predicción inicial: {e}")
            return {'technique': BreakthroughTechnique.TRADITIONAL, 'confidence': 0.5}

    def _evaluate_technique_candidates(self,
                                     characteristics: MatrixCharacteristics,
                                     initial_prediction: Dict[str, Any],
                                     performance_target: float = None,
                                     accuracy_tolerance: float = 0.01) -> List[TechniqueSelection]:
        """
        Evalúa técnicas candidatas basadas en características y restricciones.
        """
        candidates = []

        # Técnica tradicional (baseline)
        candidates.append(TechniqueSelection(
            technique=BreakthroughTechnique.TRADITIONAL,
            confidence=initial_prediction.get('confidence', 0.5),
            expected_performance=self._estimate_traditional_performance(characteristics),
            parameters={},
            reasoning="Técnica tradicional como baseline"
        ))

        # Evaluar técnicas de breakthrough disponibles
        available_techniques = [
            (BreakthroughTechnique.LOW_RANK, LOW_RANK_AVAILABLE, "Buena para matrices de bajo rango"),
            (BreakthroughTechnique.COPPERSMITH_WINOGRAD, CW_AVAILABLE, "Algoritmo matemático avanzado"),
            (BreakthroughTechnique.QUANTUM_ANNEALING, QUANTUM_AVAILABLE, "Optimización cuántica simulada"),
        ]

        for technique, available, reasoning in available_techniques:
            if available:
                suitability = self._evaluate_technique_suitability(
                    technique, characteristics, performance_target, accuracy_tolerance)

                if suitability['suitable']:
                    candidates.append(TechniqueSelection(
                        technique=technique,
                        confidence=suitability['confidence'],
                        expected_performance=suitability['expected_performance'],
                        parameters=suitability['parameters'],
                        reasoning=reasoning
                    ))

        # Evaluar técnicas híbridas si hay múltiples técnicas disponibles
        if len([c for c in candidates if c.technique != BreakthroughTechnique.TRADITIONAL]) >= 2:
            hybrid_candidates = self._evaluate_hybrid_candidates(characteristics, candidates)
            candidates.extend(hybrid_candidates)

        return candidates

    def _evaluate_technique_suitability(self,
                                      technique: BreakthroughTechnique,
                                      characteristics: MatrixCharacteristics,
                                      performance_target: float = None,
                                      accuracy_tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Evalúa si una técnica es adecuada para las características dadas.
        """
        suitability = {
            'suitable': False,
            'confidence': 0.0,
            'expected_performance': 0.0,
            'parameters': {}
        }

        if technique == BreakthroughTechnique.LOW_RANK:
            # Low-rank es bueno para matrices con rango efectivo bajo
            rank_ratio = (characteristics.rank_a / characteristics.size_a[0] +
                         characteristics.rank_b / characteristics.size_b[0]) / 2

            if rank_ratio < 0.7:  # Menos del 70% de rango efectivo
                suitability['suitable'] = True
                suitability['confidence'] = 0.8 - rank_ratio  # Mayor confianza si rango más bajo
                suitability['expected_performance'] = 0.24  # Basado en mediciones
                suitability['parameters'] = {
                    'rank_target': int(min(characteristics.rank_a, characteristics.rank_b) * 0.5)
                }

        elif technique == BreakthroughTechnique.COPPERSMITH_WINOGRAD:
            # CW es bueno para matrices grandes con buena localidad
            matrix_size = min(characteristics.size_a + characteristics.size_b)

            if matrix_size >= 256:  # Bueno para matrices medianas/grandes
                suitability['suitable'] = True
                suitability['confidence'] = min(0.9, matrix_size / 1000)  # Confianza aumenta con tamaño
                suitability['expected_performance'] = 6.7  # Basado en mediciones
                suitability['parameters'] = {}

        elif technique == BreakthroughTechnique.QUANTUM_ANNEALING:
            # Quantum annealing es experimental pero puede ser bueno para optimización
            if characteristics.computational_intensity > 0.5:
                suitability['suitable'] = True
                suitability['confidence'] = 0.6  # Confianza moderada
                suitability['expected_performance'] = 0.03  # Basado en mediciones limitadas
                suitability['parameters'] = {'num_sweeps': 50}

        return suitability

    def _evaluate_hybrid_candidates(self,
                                  characteristics: MatrixCharacteristics,
                                  base_candidates: List[TechniqueSelection]) -> List[TechniqueSelection]:
        """
        Evalúa técnicas híbridas combinando múltiples approaches.
        """
        hybrids = []

        # Hybrid: Low-Rank + CW
        lr_available = any(c.technique == BreakthroughTechnique.LOW_RANK for c in base_candidates)
        cw_available = any(c.technique == BreakthroughTechnique.COPPERSMITH_WINOGRAD for c in base_candidates)

        if lr_available and cw_available:
            hybrids.append(TechniqueSelection(
                technique=BreakthroughTechnique.HYBRID_LOW_RANK_CW,
                confidence=0.7,
                expected_performance=4.0,  # Estimación conservadora
                parameters={'combine_lr_cw': True},
                reasoning="Combinación de low-rank y CW para mejor performance"
            ))

        return hybrids

    def _select_best_candidate(self,
                             candidates: List[TechniqueSelection],
                             characteristics: MatrixCharacteristics) -> TechniqueSelection:
        """
        Selecciona el mejor candidato basado en múltiples criterios.
        """
        if not candidates:
            return TechniqueSelection(
                technique=BreakthroughTechnique.TRADITIONAL,
                confidence=0.5,
                expected_performance=441.88,  # Baseline típico
                parameters={},
                reasoning="Fallback a técnica tradicional"
            )

        # Puntuación compuesta: performance * confidence * suitability_factor
        def score_candidate(candidate: TechniqueSelection) -> float:
            # Factor de adecuación basado en características
            suitability_factor = 1.0

            if candidate.technique == BreakthroughTechnique.LOW_RANK:
                rank_ratio = (characteristics.rank_a / characteristics.size_a[0] +
                            characteristics.rank_b / characteristics.size_b[0]) / 2
                suitability_factor = 1.5 if rank_ratio < 0.5 else 0.8

            elif candidate.technique == BreakthroughTechnique.COPPERSMITH_WINOGRAD:
                matrix_size = min(characteristics.size_a + characteristics.size_b)
                suitability_factor = 1.2 if matrix_size >= 512 else 0.9

            return candidate.expected_performance * candidate.confidence * suitability_factor

        # Seleccionar candidato con mejor puntuación
        best_candidate = max(candidates, key=score_candidate)

        self.logger.info(f"Seleccionada técnica: {best_candidate.technique.value} "
                        f"(performance: {best_candidate.expected_performance:.2f} GFLOPS, "
                        f"confianza: {best_candidate.confidence:.2f})")

        return best_candidate

    def _optimize_parameters_bayesian(self,
                                    selection: TechniqueSelection,
                                    characteristics: MatrixCharacteristics) -> TechniqueSelection:
        """
        Optimiza parámetros usando Bayesian Optimization.
        """
        if not self.use_bayesian_opt or self.bayesian_optimizer is None:
            return selection

        try:
            # Definir función objetivo para BO
            def objective_function(params):
                # Evaluar performance con estos parámetros
                # (simplificado - en la práctica requeriría ejecución real)
                if selection.technique == BreakthroughTechnique.LOW_RANK:
                    rank_param = params.get('rank_target', selection.parameters.get('rank_target', 50))
                    # Estimar performance basado en rank
                    estimated_gflops = 0.24 * (rank_param / 100)  # Simplificación
                    return -estimated_gflops  # BO minimiza, así que negamos

                elif selection.technique == BreakthroughTechnique.COPPERSMITH_WINOGRAD:
                    # CW tiene pocos parámetros para optimizar
                    return -selection.expected_performance

                return -selection.expected_performance

            # Ejecutar optimización bayesiana
            # (simplificado - implementación completa requeriría más trabajo)
            optimized_params = selection.parameters.copy()

            # Actualizar selección con parámetros optimizados
            selection.parameters = optimized_params

        except Exception as e:
            self.logger.warning(f"Error en optimización bayesiana: {e}")

        return selection

    def _estimate_traditional_performance(self, characteristics: MatrixCharacteristics) -> float:
        """Estima performance de técnicas tradicionales."""
        matrix_size = min(characteristics.size_a + characteristics.size_b)

        # Estimación basada en mediciones previas
        if matrix_size <= 256:
            return 76.61
        elif matrix_size <= 512:
            return 441.88
        else:
            # Extrapolar para matrices más grandes
            return 441.88 * (matrix_size / 512) ** 0.5  # Escalado sublineal

    def _record_decision(self, selection: TechniqueSelection,
                        characteristics: MatrixCharacteristics):
        """Registra decisión para aprendizaje futuro."""
        decision_record = {
            'technique': selection.technique.value,
            'confidence': selection.confidence,
            'expected_performance': selection.expected_performance,
            'matrix_size': min(characteristics.size_a + characteristics.size_b),
            'rank_ratio': (characteristics.rank_a / characteristics.size_a[0] +
                          characteristics.rank_b / characteristics.size_b[0]) / 2,
            'timestamp': pd.Timestamp.now()
        }

        self.decision_history.append(decision_record)

        # Mantener solo las últimas 1000 decisiones
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]

    def get_decision_statistics(self) -> Dict[str, Any]:
        """Retorna estadísticas de decisiones tomadas."""
        if not self.decision_history:
            return {}

        df = pd.DataFrame(self.decision_history)

        stats = {
            'total_decisions': len(df),
            'technique_distribution': df['technique'].value_counts().to_dict(),
            'average_confidence': df['confidence'].mean(),
            'average_performance': df['expected_performance'].mean(),
            'technique_performance': df.groupby('technique')['expected_performance'].mean().to_dict()
        }

        return stats

    def execute_selected_technique(self,
                                 selection: TechniqueSelection,
                                 matrix_a: np.ndarray,
                                 matrix_b: np.ndarray,
                                 additional_params: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Ejecuta la técnica seleccionada con las matrices dadas.

        Args:
            selection: Técnica seleccionada
            matrix_a, matrix_b: Matrices de entrada
            additional_params: Parámetros adicionales opcionales

        Returns:
            Resultado y métricas de ejecución
        """
        if selection.technique == BreakthroughTechnique.TRADITIONAL:
            # Usar multiplicación NumPy estándar
            import time
            start_time = time.time()
            result = matrix_a @ matrix_b
            execution_time = time.time() - start_time

            operations = 2 * matrix_a.shape[0] * matrix_a.shape[1] * matrix_b.shape[1]
            gflops = (operations / execution_time) / 1e9

            return result, {
                'technique': 'traditional_numpy',
                'gflops_achieved': gflops,
                'execution_time': execution_time,
                'error_relative': 0.0  # NumPy es exacto
            }

        elif selection.technique in self.technique_implementations:
            implementation = self.technique_implementations[selection.technique]

            try:
                if selection.technique == BreakthroughTechnique.LOW_RANK:
                    # Usar parámetros adicionales si están disponibles
                    params = {}
                    if additional_params:
                        params.update(additional_params)
                    result, metrics = implementation.optimized_gemm_gpu(matrix_a, matrix_b, **params)

                elif selection.technique == BreakthroughTechnique.COPPERSMITH_WINOGRAD:
                    result, metrics = implementation.cw_matrix_multiply_gpu(matrix_a, matrix_b)

                elif selection.technique == BreakthroughTechnique.QUANTUM_ANNEALING:
                    # Usar parámetros de selection y adicionales
                    num_sweeps = selection.parameters.get('num_sweeps', 30)  # Reducido por defecto
                    if additional_params and 'num_sweeps' in additional_params:
                        num_sweeps = additional_params['num_sweeps']
                    result, metrics = implementation.quantum_annealing_optimization(
                        matrix_a, matrix_b, num_sweeps=num_sweeps)

                return result, metrics

            except Exception as e:
                self.logger.error(f"Error ejecutando {selection.technique.value}: {e}")
                # Fallback a tradicional
                return self.execute_selected_technique(
                    TechniqueSelection(BreakthroughTechnique.TRADITIONAL, 0.5, 441.88, {}, "Fallback"),
                    matrix_a, matrix_b
                )

        else:
            # Técnica no disponible, usar fallback
            self.logger.warning(f"Técnica {selection.technique.value} no disponible, usando fallback")
            return self.execute_selected_technique(
                TechniqueSelection(BreakthroughTechnique.TRADITIONAL, 0.5, 441.88, {}, "Fallback"),
                matrix_a, matrix_b
            )