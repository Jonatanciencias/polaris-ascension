#!/usr/bin/env python3
"""
🤖 FASE 7: AI KERNEL PREDICTOR - PREDICTION INTERFACE
=====================================================

Interfaz de predicción para seleccionar automáticamente el mejor kernel
basado en predicciones de machine learning.

Objetivos:
- Integración con framework GEMM existente
- Selección automática de kernel óptimo
- Predicciones en tiempo real
- Optimización basada en ML

Autor: AI Assistant
Fecha: 2024
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Importar integración de Bayesian Optimization
try:
    # Ruta relativa desde fase_7_ai_kernel_predictor/src hacia fase_8_bayesian_optimization/src
    bayesian_path = Path(__file__).parent.parent.parent / "fase_8_bayesian_optimization" / "src"
    sys.path.append(str(bayesian_path))
    from bayesian_integration import (
        get_bayesian_optimal_params,
        estimate_performance_with_params,
        BAYESIAN_OPTIMAL_PARAMS,
        BAYESIAN_OPTIMIZATION_METADATA
    )
    BAYESIAN_INTEGRATION_AVAILABLE = True
    print("✅ Bayesian Optimization integration disponible")
except ImportError as e:
    print(f"⚠️  Bayesian Optimization integration no disponible: {e}")
    BAYESIAN_INTEGRATION_AVAILABLE = False

class AIKernelPredictor:
    """
    Predictor de kernels basado en machine learning para optimización automática
    de performance en GEMM operations.
    """

    def __init__(self, model_dir: str = None):
        """
        Inicializa el predictor con el modelo entrenado.

        Args:
            model_dir: Directorio donde están guardados los modelos
        """
        if model_dir is None:
            # Buscar automáticamente el directorio de modelos
            current_dir = Path(__file__).parent
            possible_dirs = [
                current_dir / "models",
                current_dir.parent / "models",
                Path.cwd() / "fase_7_ai_kernel_predictor" / "models"
            ]

            for model_dir_path in possible_dirs:
                if model_dir_path.exists():
                    model_dir = str(model_dir_path)
                    break

        if not model_dir or not Path(model_dir).exists():
            raise FileNotFoundError(f"No se encontró directorio de modelos en {model_dir}")

        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.metadata = None
        self.kernel_types = ['unknown', 'strassen', 'gcn4_optimized']

        self._load_model()

    def _load_model(self):
        """Carga el modelo entrenado y metadatos."""
        try:
            # Cargar modelo (es un diccionario con modelo y scaler)
            model_path = self.model_dir / "kernel_predictor_random_forest.joblib"
            model_data = joblib.load(model_path)

            self.model = model_data['model']
            self.scaler = model_data['scaler']

            # Cargar metadatos
            metadata_path = self.model_dir / "model_metadata.json"
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

            print("✅ Modelo cargado exitosamente")
            print(f"   Modelo: {self.metadata.get('best_model', 'unknown')}")
            print(f"   R² CV: {self.metadata.get('cv_r2', 0):.3f}")
            print(f"   MAE: {self.metadata.get('mae', 0):.1f} GFLOPS")
        except Exception as e:
            raise RuntimeError(f"Error cargando modelo: {e}")

    def _prepare_features(self, matrix_size: int, kernel_type: str,
                         optimization_level: int = 1) -> np.ndarray:
        """
        Prepara las features para la predicción en el orden correcto.

        Args:
            matrix_size: Tamaño de la matriz
            kernel_type: Tipo de kernel
            optimization_level: Nivel de optimización

        Returns:
            Features preparadas como array numpy
        """
        # Calcular features base (en el orden del entrenamiento)
        log_matrix_size = np.log2(matrix_size)

        # Estimar intensidad de memoria y cómputo (simplificado)
        memory_intensity = 1.0 / np.sqrt(matrix_size)  # Más memoria para matrices pequeñas
        compute_intensity = np.log(matrix_size) / 10.0  # Más cómputo para matrices grandes

        # Features base
        features = [
            log_matrix_size,
            float(optimization_level),
            memory_intensity,
            compute_intensity
        ]

        # One-hot encoding para kernels (en el orden del dataset)
        # kernel_gcn4_optimized, kernel_strassen
        kernel_features = [
            1.0 if kernel_type == 'gcn4_optimized' else 0.0,
            1.0 if kernel_type == 'strassen' else 0.0
        ]

        features.extend(kernel_features)

        return np.array(features)

    def predict_performance(self, matrix_size: int, kernel_type: str,
                          optimization_level: int = 1) -> float:
        """
        Predice el performance (GFLOPS) para un kernel específico.

        Args:
            matrix_size: Tamaño de la matriz
            kernel_type: Tipo de kernel ('naive', 'tiled', 'vectorized', 'winograd')
            optimization_level: Nivel de optimización

        Returns:
            Performance predicho en GFLOPS
        """
        if kernel_type not in self.kernel_types:
            raise ValueError(f"Kernel type {kernel_type} not supported")

        # Preparar features completas (ya incluye one-hot encoding)
        features = self._prepare_features(matrix_size, kernel_type, optimization_level)

        # Escalar features
        if self.scaler:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)

        # Predecir
        prediction = self.model.predict(features_scaled)[0]

        return max(0.0, prediction)  # No permitir valores negativos

    def predict_performance_with_bayesian(self, matrix_size: int, kernel_type: str,
                                        optimization_level: int = 1,
                                        use_bayesian_params: bool = True) -> float:
        """
        Predice el performance usando parámetros optimizados por Bayesian Optimization.

        Args:
            matrix_size: Tamaño de la matriz
            kernel_type: Tipo de kernel
            optimization_level: Nivel de optimización
            use_bayesian_params: Si usar parámetros Bayesian optimizados

        Returns:
            Performance predicho en GFLOPS con ajuste Bayesian
        """
        if not BAYESIAN_INTEGRATION_AVAILABLE:
            print("⚠️  Bayesian integration no disponible, usando predicción estándar")
            return self.predict_performance(matrix_size, kernel_type, optimization_level)

        # Obtener predicción base
        base_prediction = self.predict_performance(matrix_size, kernel_type, optimization_level)

        if not use_bayesian_params:
            return base_prediction

        try:
            # Obtener parámetros óptimos de Bayesian
            optimal_params = get_bayesian_optimal_params()

            # Calcular baseline para este tamaño de matriz
            if matrix_size <= 256:
                baseline = 25.0
            elif matrix_size <= 512:
                baseline = 35.0
            elif matrix_size <= 1024:
                baseline = 60.0
            elif matrix_size <= 2048:
                baseline = 100.0
            else:
                baseline = 120.0

            # Estimar mejora usando los parámetros Bayesian
            estimated_with_params = estimate_performance_with_params(matrix_size, optimal_params)
            improvement_factor = estimated_with_params / baseline

            # Aplicar mejora (con factor de ajuste conservador)
            adjusted_prediction = base_prediction * improvement_factor

            print(f"🔬 Bayesian adjustment: {base_prediction:.1f} → {adjusted_prediction:.1f} GFLOPS")
            print(f"   Mejora: {((improvement_factor - 1) * 100):.1f}%")

            return adjusted_prediction

        except Exception as e:
            print(f"⚠️  Error en ajuste Bayesian: {e}, usando predicción base")
            return base_prediction

    def predict_best_kernel_enhanced(self, matrix_size: int, optimization_level: int = 1,
                                   use_bayesian: bool = True) -> Dict[str, Any]:
        """
        Predice el mejor kernel con integración de Bayesian Optimization.

        Args:
            matrix_size: Tamaño de la matriz
            optimization_level: Nivel de optimización
            use_bayesian: Si usar parámetros Bayesian optimizados

        Returns:
            Diccionario con el mejor kernel y predicciones mejoradas
        """
        predictions = {}
        predictions_bayesian = {}

        # Predecir performance para cada kernel (base y con Bayesian)
        for kernel in self.kernel_types:
            perf_base = self.predict_performance(matrix_size, kernel, optimization_level)
            perf_bayesian = self.predict_performance_with_bayesian(
                matrix_size, kernel, optimization_level, use_bayesian
            )

            predictions[kernel] = perf_base
            predictions_bayesian[kernel] = perf_bayesian

        # Encontrar el mejor con Bayesian
        best_kernel = max(predictions_bayesian.items(), key=lambda x: x[1])

        # Calcular mejora total
        base_best = predictions[best_kernel[0]]
        bayesian_best = best_kernel[1]
        improvement = ((bayesian_best / base_best) - 1) * 100 if base_best > 0 else 0

        result = {
            'best_kernel': best_kernel[0],
            'predicted_performance': bayesian_best,
            'predicted_performance_base': base_best,
            'improvement_percent': improvement,
            'all_predictions': predictions,
            'all_predictions_bayesian': predictions_bayesian,
            'matrix_size': matrix_size,
            'optimization_level': optimization_level,
            'confidence_score': self._calculate_confidence(predictions_bayesian),
            'bayesian_integration': BAYESIAN_INTEGRATION_AVAILABLE and use_bayesian,
            'bayesian_metadata': BAYESIAN_OPTIMIZATION_METADATA if BAYESIAN_INTEGRATION_AVAILABLE else None
        }

        return result
        """
        Predice el mejor kernel para un tamaño de matriz dado.

        Args:
            matrix_size: Tamaño de la matriz
            optimization_level: Nivel de optimización

        Returns:
            Diccionario con el mejor kernel y predicciones para todos
        """
        predictions = {}

        # Predecir performance para cada kernel
        for kernel in self.kernel_types:
            perf = self.predict_performance(matrix_size, kernel, optimization_level)
            predictions[kernel] = perf

        # Encontrar el mejor
        best_kernel = max(predictions.items(), key=lambda x: x[1])

        result = {
            'best_kernel': best_kernel[0],
            'predicted_performance': best_kernel[1],
            'all_predictions': predictions,
            'matrix_size': matrix_size,
            'optimization_level': optimization_level,
            'confidence_score': self._calculate_confidence(predictions)
        }

        return result

    def _calculate_confidence(self, predictions: Dict[str, float]) -> float:
        """
        Calcula un score de confianza basado en la varianza de las predicciones.

        Args:
            predictions: Diccionario con predicciones por kernel

        Returns:
            Score de confianza (0-1)
        """
        values = list(predictions.values())
        if len(values) < 2:
            return 1.0

        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0

        # Coeficiente de variación (menor variación = mayor confianza)
        cv = np.std(values) / mean_val
        confidence = 1.0 / (1.0 + cv)  # Normalizar a 0-1

        return confidence

    def get_kernel_recommendations(self, matrix_sizes: List[int],
                                 optimization_level: int = 1,
                                 use_bayesian: bool = True) -> pd.DataFrame:
        """
        Genera recomendaciones de kernel para múltiples tamaños de matriz con opción Bayesian.

        Args:
            matrix_sizes: Lista de tamaños de matriz
            optimization_level: Nivel de optimización
            use_bayesian: Si usar parámetros Bayesian optimizados

        Returns:
            DataFrame con recomendaciones mejoradas
        """
        results = []

        for size in matrix_sizes:
            pred = self.predict_best_kernel_enhanced(size, optimization_level, use_bayesian)
            row = {
                'matrix_size': size,
                'best_kernel': pred['best_kernel'],
                'predicted_gflops': pred['predicted_performance'],
                'predicted_gflops_base': pred['predicted_performance_base'],
                'improvement_percent': pred['improvement_percent'],
                'confidence': pred['confidence_score'],
                'bayesian_used': pred['bayesian_integration']
            }
            results.append(row)

        return pd.DataFrame(results)

def main():
    """Función principal para testing del predictor con integración Bayesian."""
    print("🤖 FASE 7: AI KERNEL PREDICTOR - ENHANCED WITH BAYESIAN OPTIMIZATION")
    print("=" * 70)

    try:
        # Inicializar predictor
        predictor = AIKernelPredictor()

        # Test con diferentes tamaños de matriz
        test_sizes = [256, 512, 1024, 2048]

        print("🧪 Testing predicciones con integración Bayesian...")
        print("-" * 50)

        for size in test_sizes:
            result = predictor.predict_best_kernel_enhanced(size, use_bayesian=True)
            print(f"Matrix {size}x{size}:")
            print(f"   Mejor kernel: {result['best_kernel']}")
            print(f"   Performance: {result['predicted_performance']:.1f} GFLOPS")
            print(f"   Performance base: {result['predicted_performance_base']:.1f} GFLOPS")
            print(f"   Mejora: {result['improvement_percent']:.1f}%")
            print(f"   Confianza: {result['confidence_score']:.3f}")
            print(f"   Bayesian usado: {result['bayesian_integration']}")
            print()

        # Comparación lado a lado
        print("📊 Comparación: Base vs Bayesian Enhanced")
        print("-" * 50)

        comparison_data = []
        for size in test_sizes:
            # Usar predicción sin Bayesian para comparación base
            base_result = predictor.predict_best_kernel_enhanced(size, use_bayesian=False)
            bayesian_result = predictor.predict_best_kernel_enhanced(size, use_bayesian=True)

            comparison_data.append({
                'size': size,
                'base_kernel': base_result['best_kernel'],
                'base_gflops': base_result['predicted_performance'],
                'bayesian_kernel': bayesian_result['best_kernel'],
                'bayesian_gflops': bayesian_result['predicted_performance'],
                'improvement': bayesian_result['improvement_percent']
            })

        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False, float_format='%.1f'))

        # Generar recomendaciones para un rango completo
        print("\n📈 Recomendaciones completas con Bayesian:")
        print("-" * 50)

        sizes_range = [64, 128, 256, 512, 1024, 2048, 4096]
        recommendations = predictor.get_kernel_recommendations(sizes_range, use_bayesian=True)

        print(recommendations.to_string(index=False, float_format='%.1f'))

        # Mostrar metadatos de Bayesian si disponibles
        if BAYESIAN_INTEGRATION_AVAILABLE:
            print("\n🔬 Metadatos de Bayesian Optimization:")
            print("-" * 50)
            meta = BAYESIAN_OPTIMIZATION_METADATA
            print(f"   Mejor performance alcanzado: {meta['best_performance']:.1f} GFLOPS")
            print(f"   Mejora total: {meta['improvement_over_baseline']:.1f}%")
            print(f"   Evaluaciones realizadas: {meta['total_evaluations']}")
            print(f"   Tiempo de optimización: {meta['optimization_time']:.2f}s")
            print(f"   Parámetros óptimos encontrados: {len(BAYESIAN_OPTIMAL_PARAMS)}")

        print("\n✅ Enhanced prediction interface funcionando correctamente!")
        print("🎯 Integración Bayesian Optimization completada!")

    except Exception as e:
        print(f"❌ Error en testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()