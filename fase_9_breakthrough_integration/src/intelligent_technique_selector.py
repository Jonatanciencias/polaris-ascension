#!/usr/bin/env python3
"""
🎯 INTELLIGENT TECHNIQUE SELECTOR
==================================

Sistema de selección automática inteligente que elige la mejor técnica de optimización
basado en características de las matrices, historial de performance y aprendizaje continuo.

Características:
- Feature extraction avanzada de matrices
- AI Kernel Predictor integration
- Sistema de feedback learning
- Selección multi-criterio
- Auto-tuning continuo

Autor: AI Assistant
Fecha: 2026
"""

import sys
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Importar AI Kernel Predictor
try:
    ai_predictor_path = Path(__file__).parent.parent / "fase_7_ai_kernel_predictor" / "src"
    sys.path.insert(0, str(ai_predictor_path))
    from kernel_predictor import AIKernelPredictor
    AI_PREDICTOR_AVAILABLE = True
    print("✅ AI Kernel Predictor disponible")
except ImportError as e:
    print(f"⚠️  AI Kernel Predictor no disponible: {e}")
    AI_PREDICTOR_AVAILABLE = False

class TechniqueType(Enum):
    """Tipos de técnicas disponibles en el sistema híbrido"""
    LOW_RANK = "low_rank"
    COPPERSMITH_WINOGRAD = "cw"
    QUANTUM_ANNEALING = "quantum"
    AI_PREDICTOR = "ai_predictor"
    BAYESIAN_OPTIMIZATION = "bayesian_opt"
    NEUROMORPHIC_COMPUTING = "neuromorphic"
    TENSOR_CORE_SIMULATION = "tensor_core"
    HYBRID_QUANTUM_CLASSICAL = "hybrid_quantum_classical"

@dataclass
class MatrixFeatures:
    """Características extraídas de las matrices de entrada"""
    size_a: Tuple[int, int]
    size_b: Tuple[int, int]
    dtype: str
    sparsity_a: float
    sparsity_b: float
    condition_number_a: float
    condition_number_b: float
    memory_footprint_mb: float
    compute_intensity: float
    structure_type: str  # 'dense', 'sparse', 'diagonal', 'block'
    symmetry_a: bool
    symmetry_b: bool
    matrix_type: str  # 'square', 'rectangular', 'vector'

@dataclass
class TechniqueScore:
    """Score de una técnica para una matriz específica"""
    technique: TechniqueType
    predicted_performance: float
    confidence: float
    suitability_score: float
    computational_cost: float
    memory_efficiency: float
    numerical_stability: float
    training_required: bool

@dataclass
class SelectionResult:
    """Resultado de la selección automática"""
    recommended_technique: TechniqueType
    technique_scores: Dict[TechniqueType, TechniqueScore]
    matrix_features: MatrixFeatures
    selection_confidence: float
    expected_performance: float
    reasoning: List[str]
    alternative_options: List[TechniqueType]
    timestamp: float

class MatrixFeatureExtractor:
    """Extractor avanzado de características de matrices"""

    def __init__(self):
        self.cache = {}  # Cache para evitar recálculos

    def extract_features(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> MatrixFeatures:
        """
        Extrae características completas de las matrices de entrada.

        Args:
            matrix_a, matrix_b: Matrices de entrada

        Returns:
            MatrixFeatures con todas las características extraídas
        """
        # Crear clave de cache
        cache_key = (matrix_a.shape, matrix_b.shape, matrix_a.dtype, hash(matrix_a.tobytes()[:100]))

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Extraer características básicas
        size_a = matrix_a.shape
        size_b = matrix_b.shape
        dtype = str(matrix_a.dtype)

        # Sparsity (proporción de elementos no cero)
        sparsity_a = 1.0 - (np.count_nonzero(matrix_a) / matrix_a.size)
        sparsity_b = 1.0 - (np.count_nonzero(matrix_b) / matrix_b.size)

        # Condition number (estabilidad numérica)
        try:
            if matrix_a.shape[0] == matrix_a.shape[1] and matrix_a.shape[0] <= 1000:
                condition_number_a = np.linalg.cond(matrix_a.astype(np.float64))
            else:
                # Para matrices grandes o no cuadradas, estimación simplificada
                condition_number_a = np.linalg.norm(matrix_a) / np.linalg.norm(np.linalg.pinv(matrix_a))
        except:
            condition_number_a = 1.0

        try:
            if matrix_b.shape[0] == matrix_b.shape[1] and matrix_b.shape[0] <= 1000:
                condition_number_b = np.linalg.cond(matrix_b.astype(np.float64))
            else:
                condition_number_b = np.linalg.norm(matrix_b) / np.linalg.norm(np.linalg.pinv(matrix_b))
        except:
            condition_number_b = 1.0

        # Memory footprint
        memory_footprint_mb = (matrix_a.nbytes + matrix_b.nbytes) / (1024 ** 2)

        # Compute intensity (FLOPS/byte)
        operations = 2 * size_a[0] * size_a[1] * size_b[1]
        memory_access = matrix_a.nbytes + matrix_b.nbytes + (size_a[0] * size_b[1] * matrix_a.dtype.itemsize)
        compute_intensity = operations / memory_access if memory_access > 0 else 0

        # Structure type
        structure_type = self._classify_structure(matrix_a, matrix_b)

        # Symmetry
        symmetry_a = self._check_symmetry(matrix_a)
        symmetry_b = self._check_symmetry(matrix_b)

        # Matrix type
        if size_a[0] == size_a[1] == size_b[0] == size_b[1]:
            matrix_type = 'square'
        elif size_a[0] == 1 or size_a[1] == 1 or size_b[0] == 1 or size_b[1] == 1:
            matrix_type = 'vector'
        else:
            matrix_type = 'rectangular'

        features = MatrixFeatures(
            size_a=size_a,
            size_b=size_b,
            dtype=dtype,
            sparsity_a=sparsity_a,
            sparsity_b=sparsity_b,
            condition_number_a=condition_number_a,
            condition_number_b=condition_number_b,
            memory_footprint_mb=memory_footprint_mb,
            compute_intensity=compute_intensity,
            structure_type=structure_type,
            symmetry_a=symmetry_a,
            symmetry_b=symmetry_b,
            matrix_type=matrix_type
        )

        # Cachear resultado
        self.cache[cache_key] = features

        return features

    def _classify_structure(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> str:
        """Clasifica el tipo de estructura de las matrices"""
        avg_sparsity = (np.count_nonzero(matrix_a) / matrix_a.size +
                       np.count_nonzero(matrix_b) / matrix_b.size) / 2

        if avg_sparsity > 0.9:
            return 'sparse'
        elif self._is_diagonal(matrix_a) or self._is_diagonal(matrix_b):
            return 'diagonal'
        elif self._is_block_structure(matrix_a) or self._is_block_structure(matrix_b):
            return 'block'
        else:
            return 'dense'

    def _is_diagonal(self, matrix: np.ndarray) -> bool:
        """Verifica si una matriz es aproximadamente diagonal"""
        if matrix.shape[0] != matrix.shape[1]:
            return False
        off_diagonal = matrix - np.diag(np.diag(matrix))
        return np.linalg.norm(off_diagonal) < 0.01 * np.linalg.norm(matrix)

    def _is_block_structure(self, matrix: np.ndarray) -> bool:
        """Verifica si una matriz tiene estructura de bloques"""
        # Implementación simplificada - verificar si hay bloques densos
        size = min(matrix.shape)
        if size < 16:
            return False

        block_size = size // 4
        for i in range(0, size, block_size):
            for j in range(0, size, block_size):
                block = matrix[i:i+block_size, j:j+block_size]
                if np.count_nonzero(block) / block.size > 0.8:
                    return True
        return False

    def _check_symmetry(self, matrix: np.ndarray) -> bool:
        """Verifica si una matriz es simétrica"""
        if matrix.shape[0] != matrix.shape[1]:
            return False
        return np.allclose(matrix, matrix.T, rtol=1e-5)

class IntelligentTechniqueSelector:
    """
    Selector inteligente de técnicas de optimización basado en ML y reglas expertas.
    """

    def __init__(self, learning_enabled: bool = True):
        """
        Inicializa el selector inteligente.

        Args:
            learning_enabled: Si habilitar aprendizaje continuo
        """
        self.feature_extractor = MatrixFeatureExtractor()
        self.ai_predictor = None
        self.learning_enabled = learning_enabled

        # Historial de selecciones para aprendizaje
        self.selection_history = []
        self.performance_history = []

        # Reglas expertas por defecto
        self.expert_rules = self._load_expert_rules()

        # Inicializar AI predictor si disponible
        if AI_PREDICTOR_AVAILABLE:
            try:
                self.ai_predictor = AIKernelPredictor()
                print("✅ AI Kernel Predictor integrado")
            except Exception as e:
                print(f"⚠️  Error cargando AI Predictor: {e}")
                self.ai_predictor = None

        # Cargar historial si existe
        self._load_history()

    def _load_expert_rules(self) -> Dict[str, Dict]:
        """Carga las reglas expertas para selección de técnicas"""
        return {
            'size_thresholds': {
                'small': 256,      # Matrices pequeñas
                'medium': 1024,    # Matrices medianas
                'large': 4096      # Matrices grandes
            },
            'sparsity_thresholds': {
                'dense': 0.1,      # Alta densidad
                'moderate': 0.5,   # Densidad moderada
                'sparse': 0.8      # Alta sparsidad
            },
            'technique_preferences': {
                TechniqueType.LOW_RANK: {
                    'min_size': 512,
                    'max_sparsity': 0.3,
                    'preferred_structure': ['dense', 'block'],
                    'performance_factor': 1.2
                },
                TechniqueType.COPPERSMITH_WINOGRAD: {
                    'min_size': 256,
                    'max_condition': 1e10,
                    'preferred_structure': ['dense'],
                    'performance_factor': 1.5
                },
                TechniqueType.QUANTUM_ANNEALING: {
                    'max_size': 512,  # Limitado por optimización actual
                    'max_sparsity': 0.1,
                    'preferred_structure': ['dense'],
                    'performance_factor': 1.1
                },
                TechniqueType.AI_PREDICTOR: {
                    'min_size': 128,
                    'max_size': 2048,
                    'performance_factor': 1.3
                },
                TechniqueType.BAYESIAN_OPTIMIZATION: {
                    'min_size': 256,
                    'max_size': 1024,
                    'performance_factor': 1.4
                },
                TechniqueType.NEUROMORPHIC_COMPUTING: {
                    'max_sparsity': 0.7,
                    'preferred_structure': ['sparse', 'dense'],
                    'performance_factor': 1.1
                },
                TechniqueType.TENSOR_CORE_SIMULATION: {
                    'min_size': 512,
                    'preferred_structure': ['dense'],
                    'performance_factor': 1.6
                },
                TechniqueType.HYBRID_QUANTUM_CLASSICAL: {
                    'min_size': 256,
                    'max_size': 1024,
                    'performance_factor': 1.2
                }
            }
        }

    def select_technique(self, matrix_a: np.ndarray, matrix_b: np.ndarray,
                        context: Optional[Dict[str, Any]] = None) -> SelectionResult:
        """
        Selecciona automáticamente la mejor técnica de optimización.

        Args:
            matrix_a, matrix_b: Matrices de entrada
            context: Contexto adicional (tiempo disponible, precisión requerida, etc.)

        Returns:
            SelectionResult con la técnica recomendada y análisis completo
        """
        context = context or {}
        start_time = time.time()

        # Extraer características de las matrices
        features = self.feature_extractor.extract_features(matrix_a, matrix_b)

        # Calcular scores para todas las técnicas
        technique_scores = {}
        reasoning = []

        for technique in TechniqueType:
            score = self._calculate_technique_score(technique, features, context)
            technique_scores[technique] = score

        # Seleccionar la mejor técnica
        best_technique = max(technique_scores.items(), key=lambda x: x[1].suitability_score)

        # Calcular confianza general
        scores = [ts.suitability_score for ts in technique_scores.values()]
        selection_confidence = self._calculate_selection_confidence(scores)

        # Generar reasoning
        reasoning = self._generate_reasoning(best_technique[0], features, technique_scores)

        # Encontrar alternativas
        sorted_techniques = sorted(technique_scores.items(),
                                 key=lambda x: x[1].suitability_score, reverse=True)
        alternative_options = [tech for tech, _ in sorted_techniques[1:4]]  # Top 3 alternativas

        result = SelectionResult(
            recommended_technique=best_technique[0],
            technique_scores=technique_scores,
            matrix_features=features,
            selection_confidence=selection_confidence,
            expected_performance=best_technique[1].predicted_performance,
            reasoning=reasoning,
            alternative_options=alternative_options,
            timestamp=time.time()
        )

        # Registrar selección para aprendizaje
        if self.learning_enabled:
            self._record_selection(result, context)

        return result

    def _calculate_technique_score(self, technique: TechniqueType,
                                 features: MatrixFeatures,
                                 context: Dict[str, Any]) -> TechniqueScore:
        """
        Calcula el score completo para una técnica específica.
        """
        # Score base de reglas expertas
        rule_score = self._calculate_rule_score(technique, features)

        # Score de AI predictor si disponible
        ai_score = self._calculate_ai_score(technique, features)

        # Score de historial de performance
        history_score = self._calculate_history_score(technique, features)

        # Factores contextuales
        context_multiplier = self._calculate_context_multiplier(technique, context)

        # Combinar scores
        combined_score = (rule_score * 0.4 + ai_score * 0.4 + history_score * 0.2) * context_multiplier

        # Estimar performance
        predicted_performance = self._estimate_performance(technique, features)

        # Calcular confianza
        confidence = min(1.0, combined_score / 10.0)  # Normalizar

        # Estimar otros factores
        computational_cost = self._estimate_computational_cost(technique, features)
        memory_efficiency = self._estimate_memory_efficiency(technique, features)
        numerical_stability = self._estimate_numerical_stability(technique, features)
        training_required = technique in [TechniqueType.AI_PREDICTOR, TechniqueType.BAYESIAN_OPTIMIZATION]

        return TechniqueScore(
            technique=technique,
            predicted_performance=predicted_performance,
            confidence=confidence,
            suitability_score=combined_score,
            computational_cost=computational_cost,
            memory_efficiency=memory_efficiency,
            numerical_stability=numerical_stability,
            training_required=training_required
        )

    def _calculate_rule_score(self, technique: TechniqueType, features: MatrixFeatures) -> float:
        """Calcula score basado en reglas expertas"""
        rules = self.expert_rules['technique_preferences'].get(technique.value, {})

        score = 0.0
        reasons = []

        # Factor de tamaño
        matrix_size = max(features.size_a[0], features.size_a[1], features.size_b[0], features.size_b[1])

        if 'min_size' in rules and matrix_size >= rules['min_size']:
            score += 2.0
            reasons.append(f"Tamaño adecuado (>= {rules['min_size']})")
        elif 'max_size' in rules and matrix_size <= rules['max_size']:
            score += 2.0
            reasons.append(f"Tamaño dentro límite (<= {rules['max_size']})")

        # Factor de sparsidad
        avg_sparsity = (features.sparsity_a + features.sparsity_b) / 2

        if 'max_sparsity' in rules and avg_sparsity <= rules['max_sparsity']:
            score += 1.5
            reasons.append(f"Sparsity adecuada (<= {rules['max_sparsity']:.1f})")

        # Factor de estructura
        if 'preferred_structure' in rules and features.structure_type in rules['preferred_structure']:
            score += 1.0
            reasons.append(f"Estructura preferida ({features.structure_type})")

        # Factor de condición numérica
        if 'max_condition' in rules and features.condition_number_a <= rules['max_condition']:
            score += 1.0
            reasons.append("Buena estabilidad numérica")

        # Performance factor base
        score *= rules.get('performance_factor', 1.0)

        return score

    def _calculate_ai_score(self, technique: TechniqueType, features: MatrixFeatures) -> float:
        """Calcula score usando AI Kernel Predictor"""
        if not self.ai_predictor:
            return 1.0  # Score neutral

        try:
            matrix_size = max(features.size_a[0], features.size_a[1])

            # Mapear technique a kernel type del predictor
            kernel_mapping = {
                TechniqueType.AI_PREDICTOR: 'gcn4_optimized',
                TechniqueType.LOW_RANK: 'gcn4_optimized',
                TechniqueType.COPPERSMITH_WINOGRAD: 'strassen',
                TechniqueType.QUANTUM_ANNEALING: 'gcn4_optimized',
                TechniqueType.BAYESIAN_OPTIMIZATION: 'gcn4_optimized',
                TechniqueType.NEUROMORPHIC_COMPUTING: 'gcn4_optimized',
                TechniqueType.TENSOR_CORE_SIMULATION: 'gcn4_optimized',
                TechniqueType.HYBRID_QUANTUM_CLASSICAL: 'gcn4_optimized'
            }

            kernel_type = kernel_mapping.get(technique, 'gcn4_optimized')

            # Predecir performance
            prediction = self.ai_predictor.predict_performance(matrix_size, kernel_type)

            # Normalizar score (mayor performance = mayor score)
            return min(10.0, prediction / 10.0)

        except Exception as e:
            print(f"⚠️  Error en AI prediction para {technique.value}: {e}")
            return 1.0

    def _calculate_history_score(self, technique: TechniqueType, features: MatrixFeatures) -> float:
        """Calcula score basado en historial de performance"""
        if not self.performance_history:
            return 1.0

        # Buscar casos similares en el historial
        similar_cases = []
        for entry in self.performance_history:
            if (abs(entry['matrix_size'] - max(features.size_a)) < 100 and
                abs(entry['sparsity'] - (features.sparsity_a + features.sparsity_b)/2) < 0.1):
                similar_cases.append(entry)

        if not similar_cases:
            return 1.0

        # Calcular score promedio para esta técnica en casos similares
        technique_scores = [case['score'] for case in similar_cases if case['technique'] == technique.value]

        if technique_scores:
            return np.mean(technique_scores)
        else:
            return 0.5  # Penalización por falta de historial

    def _calculate_context_multiplier(self, technique: TechniqueType, context: Dict[str, Any]) -> float:
        """Calcula multiplicador basado en contexto"""
        multiplier = 1.0

        # Factor de tiempo disponible
        time_available = context.get('time_available', 'normal')
        if time_available == 'limited':
            # Preferir técnicas rápidas
            fast_techniques = [TechniqueType.AI_PREDICTOR, TechniqueType.LOW_RANK]
            if technique in fast_techniques:
                multiplier *= 1.2
            else:
                multiplier *= 0.8

        # Factor de precisión requerida
        precision_required = context.get('precision_required', 'normal')
        if precision_required == 'high':
            # Preferir técnicas numéricamente estables
            stable_techniques = [TechniqueType.COPPERSMITH_WINOGRAD, TechniqueType.TENSOR_CORE_SIMULATION]
            if technique in stable_techniques:
                multiplier *= 1.1

        return multiplier

    def _estimate_performance(self, technique: TechniqueType, features: MatrixFeatures) -> float:
        """Estima performance esperada para una técnica"""
        matrix_size = max(features.size_a[0], features.size_a[1])

        # Estimaciones base por técnica (basadas en benchmarks previos)
        base_performance = {
            TechniqueType.LOW_RANK: 150.0,
            TechniqueType.COPPERSMITH_WINOGRAD: 200.0,
            TechniqueType.QUANTUM_ANNEALING: 50.0,  # Limitado por optimización actual
            TechniqueType.AI_PREDICTOR: 100.0,
            TechniqueType.BAYESIAN_OPTIMIZATION: 180.0,
            TechniqueType.NEUROMORPHIC_COMPUTING: 80.0,
            TechniqueType.TENSOR_CORE_SIMULATION: 250.0,
            TechniqueType.HYBRID_QUANTUM_CLASSICAL: 120.0
        }

        perf = base_performance.get(technique, 50.0)

        # Ajustar por tamaño
        if matrix_size > 1024:
            perf *= 1.5  # Mejor escalabilidad en matrices grandes
        elif matrix_size < 256:
            perf *= 0.7  # Peor en matrices pequeñas

        # Ajustar por sparsidad
        avg_sparsity = (features.sparsity_a + features.sparsity_b) / 2
        if avg_sparsity > 0.5:
            if technique == TechniqueType.NEUROMORPHIC_COMPUTING:
                perf *= 1.3  # Bonus para técnicas neuromórficas en sparse
            else:
                perf *= 0.8  # Penalización general

        return perf

    def _estimate_computational_cost(self, technique: TechniqueType, features: MatrixFeatures) -> float:
        """Estima costo computacional (0-1, menor es mejor)"""
        matrix_size = max(features.size_a)

        base_costs = {
            TechniqueType.AI_PREDICTOR: 0.2,  # Muy rápido
            TechniqueType.LOW_RANK: 0.3,
            TechniqueType.BAYESIAN_OPTIMIZATION: 0.7,  # Más costoso
            TechniqueType.QUANTUM_ANNEALING: 0.8,  # Muy costoso actualmente
            TechniqueType.COPPERSMITH_WINOGRAD: 0.4,
            TechniqueType.NEUROMORPHIC_COMPUTING: 0.5,
            TechniqueType.TENSOR_CORE_SIMULATION: 0.3,
            TechniqueType.HYBRID_QUANTUM_CLASSICAL: 0.6
        }

        cost = base_costs.get(technique, 0.5)

        # Ajustar por tamaño
        if matrix_size > 1024:
            cost *= 1.5

        return min(1.0, cost)

    def _estimate_memory_efficiency(self, technique: TechniqueType, features: MatrixFeatures) -> float:
        """Estima eficiencia de memoria (0-1, mayor es mejor)"""
        base_efficiency = {
            TechniqueType.LOW_RANK: 0.8,  # Buena compresión
            TechniqueType.NEUROMORPHIC_COMPUTING: 0.9,  # Muy eficiente en sparse
            TechniqueType.TENSOR_CORE_SIMULATION: 0.7,
            TechniqueType.COPPERSMITH_WINOGRAD: 0.6,
            TechniqueType.AI_PREDICTOR: 0.5,
            TechniqueType.BAYESIAN_OPTIMIZATION: 0.4,
            TechniqueType.QUANTUM_ANNEALING: 0.3,  # Menos eficiente
            TechniqueType.HYBRID_QUANTUM_CLASSICAL: 0.6
        }

        efficiency = base_efficiency.get(technique, 0.5)

        # Bonus por sparsidad
        avg_sparsity = (features.sparsity_a + features.sparsity_b) / 2
        if avg_sparsity > 0.5:
            efficiency *= 1.2

        return min(1.0, efficiency)

    def _estimate_numerical_stability(self, technique: TechniqueType, features: MatrixFeatures) -> float:
        """Estima estabilidad numérica (0-1, mayor es mejor)"""
        base_stability = {
            TechniqueType.COPPERSMITH_WINOGRAD: 0.9,  # Muy estable
            TechniqueType.TENSOR_CORE_SIMULATION: 0.8,
            TechniqueType.LOW_RANK: 0.7,
            TechniqueType.AI_PREDICTOR: 0.6,
            TechniqueType.BAYESIAN_OPTIMIZATION: 0.6,
            TechniqueType.HYBRID_QUANTUM_CLASSICAL: 0.5,
            TechniqueType.NEUROMORPHIC_COMPUTING: 0.4,
            TechniqueType.QUANTUM_ANNEALING: 0.3  # Menos estable
        }

        stability = base_stability.get(technique, 0.5)

        # Penalizar por alto condition number
        avg_condition = (features.condition_number_a + features.condition_number_b) / 2
        if avg_condition > 1e8:
            stability *= 0.7

        return stability

    def _calculate_selection_confidence(self, scores: List[float]) -> float:
        """Calcula confianza en la selección basada en distribución de scores"""
        if len(scores) < 2:
            return 1.0

        scores = np.array(scores)
        best_score = np.max(scores)
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        if mean_score == 0:
            return 0.0

        # Confianza basada en separación del mejor score
        separation = (best_score - mean_score) / (std_score + 1e-6)
        confidence = 1.0 / (1.0 + np.exp(-separation))  # Sigmoid

        return confidence

    def _generate_reasoning(self, selected_technique: TechniqueType,
                          features: MatrixFeatures,
                          technique_scores: Dict[TechniqueType, TechniqueScore]) -> List[str]:
        """Genera explicación del reasoning de la selección"""
        reasoning = []

        selected_score = technique_scores[selected_technique]

        reasoning.append(f"🎯 Técnica seleccionada: {selected_technique.value}")
        reasoning.append(f"   Score de suitability: {selected_score.suitability_score:.2f}")
        reasoning.append(f"   Performance esperado: {selected_score.predicted_performance:.1f} GFLOPS")

        # Explicar factores principales
        matrix_size = max(features.size_a)
        reasoning.append(f"   Tamaño de matriz: {matrix_size}x{matrix_size}")

        avg_sparsity = (features.sparsity_a + features.sparsity_b) / 2
        reasoning.append(f"   Sparsity promedio: {avg_sparsity:.1%}")

        reasoning.append(f"   Estructura: {features.structure_type}")

        # Comparar con alternativas
        sorted_scores = sorted(technique_scores.items(),
                             key=lambda x: x[1].suitability_score, reverse=True)

        if len(sorted_scores) > 1:
            second_best = sorted_scores[1]
            margin = selected_score.suitability_score - second_best[1].suitability_score
            reasoning.append(f"   Margen vs alternativa: {margin:.2f} ({second_best[0].value})")

        return reasoning

    def _record_selection(self, result: SelectionResult, context: Dict[str, Any]):
        """Registra selección para aprendizaje futuro"""
        entry = {
            'timestamp': result.timestamp,
            'selected_technique': result.recommended_technique.value,
            'matrix_size': max(result.matrix_features.size_a),
            'sparsity': (result.matrix_features.sparsity_a + result.matrix_features.sparsity_b) / 2,
            'structure_type': result.matrix_features.structure_type,
            'selection_confidence': result.selection_confidence,
            'expected_performance': result.expected_performance,
            'context': context
        }

        self.selection_history.append(entry)

        # Limitar tamaño del historial
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-500:]

    def record_performance_feedback(self, technique: TechniqueType,
                                  actual_performance: float,
                                  matrix_features: MatrixFeatures,
                                  success: bool = True):
        """
        Registra feedback de performance real para mejorar futuras selecciones.

        Args:
            technique: Técnica que se ejecutó
            actual_performance: Performance real obtenida
            matrix_features: Características de las matrices
            success: Si la ejecución fue exitosa
        """
        entry = {
            'technique': technique.value,
            'matrix_size': max(matrix_features.size_a),
            'sparsity': (matrix_features.sparsity_a + matrix_features.sparsity_b) / 2,
            'actual_performance': actual_performance,
            'success': success,
            'timestamp': time.time()
        }

        # Calcular score basado en performance
        expected_base = self._estimate_performance(technique, matrix_features)
        if expected_base > 0:
            performance_ratio = actual_performance / expected_base
            entry['score'] = min(10.0, performance_ratio * 5.0)  # Normalizar a 0-10
        else:
            entry['score'] = 5.0

        self.performance_history.append(entry)

        # Limitar historial
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]

        # Guardar en disco
        self._save_history()

    def _load_history(self):
        """Carga historial de selecciones y performance"""
        try:
            history_file = Path(__file__).parent / "selector_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)

                self.selection_history = data.get('selection_history', [])
                self.performance_history = data.get('performance_history', [])

                print(f"✅ Historial cargado: {len(self.selection_history)} selecciones, {len(self.performance_history)} feedbacks")
        except Exception as e:
            print(f"⚠️  Error cargando historial: {e}")

    def _save_history(self):
        """Guarda historial en disco"""
        try:
            history_file = Path(__file__).parent / "selector_history.json"
            data = {
                'selection_history': self.selection_history,
                'performance_history': self.performance_history,
                'last_updated': time.time()
            }

            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error guardando historial: {e}")

    def get_selection_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del selector inteligente"""
        if not self.selection_history:
            return {'message': 'No hay historial disponible'}

        df = pd.DataFrame(self.selection_history)

        stats = {
            'total_selections': len(df),
            'technique_distribution': df['selected_technique'].value_counts().to_dict(),
            'avg_confidence': df['selection_confidence'].mean(),
            'avg_matrix_size': df['matrix_size'].mean(),
            'most_common_structure': df['structure_type'].mode().iloc[0] if not df.empty else 'unknown',
            'learning_enabled': self.learning_enabled,
            'ai_predictor_available': self.ai_predictor is not None
        }

        return stats

# Funciones de utilidad para integración con sistema híbrido
def select_optimal_technique(matrix_a: np.ndarray, matrix_b: np.ndarray,
                           context: Optional[Dict[str, Any]] = None) -> SelectionResult:
    """
    Función de conveniencia para seleccionar técnica óptima.

    Args:
        matrix_a, matrix_b: Matrices de entrada
        context: Contexto adicional

    Returns:
        Resultado de selección
    """
    selector = IntelligentTechniqueSelector()
    return selector.select_technique(matrix_a, matrix_b, context)

def benchmark_intelligent_selection(matrix_sizes: List[int] = None) -> Dict[str, Any]:
    """
    Benchmark del sistema de selección inteligente.

    Args:
        matrix_sizes: Lista de tamaños a probar

    Returns:
        Resultados del benchmark
    """
    if matrix_sizes is None:
        matrix_sizes = [128, 256, 512, 1024]

    selector = IntelligentTechniqueSelector()
    results = {}

    print("🎯 BENCHMARK INTELLIGENT TECHNIQUE SELECTION")
    print("=" * 50)

    for size in matrix_sizes:
        print(f"\n🧪 Probando tamaño {size}x{size}")

        # Crear matrices de prueba
        np.random.seed(42)
        A = np.random.randn(size, size).astype(np.float32) * 0.1
        B = np.random.randn(size, size).astype(np.float32) * 0.1

        # Medir tiempo de selección
        start_time = time.time()
        result = selector.select_technique(A, B)
        selection_time = time.time() - start_time

        results[size] = {
            'selection_time': selection_time,
            'recommended_technique': result.recommended_technique.value,
            'confidence': result.selection_confidence,
            'expected_performance': result.expected_performance,
            'num_alternatives': len(result.alternative_options),
            'reasoning': result.reasoning[:3]  # Primeras 3 líneas de reasoning
        }

        print(f"   ✅ Técnica recomendada: {result.recommended_technique.value}")
        print(f"   📊 Confianza: {result.selection_confidence:.2%}")
        print(f"   🚀 Performance esperado: {result.expected_performance:.1f} GFLOPS")
        print(f"   ⏱️  Tiempo de selección: {selection_time:.3f}s")

    # Estadísticas generales
    selector_stats = selector.get_selection_statistics()

    benchmark_results = {
        'individual_results': results,
        'selector_statistics': selector_stats,
        'benchmark_timestamp': time.time(),
        'matrix_sizes_tested': matrix_sizes
    }

    # Guardar resultados
    np.savez('intelligent_selection_benchmark.npz',
             results=results,
             stats=selector_stats,
             timestamp=time.time())

    print("\n💾 Benchmark guardado en: intelligent_selection_benchmark.npz")
    return benchmark_results

if __name__ == "__main__":
    # Demo del selector inteligente
    print("🎯 INTELLIGENT TECHNIQUE SELECTOR DEMO")
    print("=" * 45)

    # Crear selector
    selector = IntelligentTechniqueSelector()

    # Matrices de prueba
    sizes = [256, 512, 1024]

    for size in sizes:
        print(f"\n🧪 Probando selección para matrices {size}x{size}")

        np.random.seed(42)
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)

        # Seleccionar técnica
        result = selector.select_technique(A, B)

        print(f"   🎯 Recomendación: {result.recommended_technique.value}")
        print(f"   📊 Confianza: {result.selection_confidence:.2%}")
        print(f"   🚀 Performance: {result.expected_performance:.1f} GFLOPS")

        # Mostrar reasoning
        for reason in result.reasoning[:2]:
            print(f"   💡 {reason}")

        # Simular feedback (performance real ligeramente diferente)
        actual_performance = result.expected_performance * (0.9 + np.random.random() * 0.2)
        selector.record_performance_feedback(
            result.recommended_technique,
            actual_performance,
            result.matrix_features,
            success=True
        )

    # Ejecutar benchmark
    print("\n📊 EJECUTANDO BENCHMARK COMPLETO...")
    benchmark_results = benchmark_intelligent_selection()

    print("\n✅ Demo completada exitosamente!")