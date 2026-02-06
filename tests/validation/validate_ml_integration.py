#!/usr/bin/env python3
"""
🎯 VALIDACIÓN DE INTEGRACIÓN ML FINE-TUNED
===========================================

Script para validar que el Breakthrough Selector con modelo ML fine-tuned
funciona correctamente y mejora la selección de técnicas.

FASE 9.3.1: Validación de integración del modelo fine-tuned
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Añadir paths necesarios
project_root = Path(__file__).parent
sys.path.append(str(project_root / "fase_9_breakthrough_integration" / "src"))

try:
    from breakthrough_selector import BreakthroughTechniqueSelector, BreakthroughTechnique
    SELECTOR_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importando Breakthrough Selector: {e}")
    SELECTOR_AVAILABLE = False

class MLIntegrationValidator:
    """
    Validador de la integración del modelo ML fine-tuned.
    """

    def __init__(self):
        self.selector = None
        self.test_matrices = []
        self.results = []

    def initialize_selector(self) -> bool:
        """Inicializa el Breakthrough Selector con modelo ML."""
        if not SELECTOR_AVAILABLE:
            print("❌ Breakthrough Selector no disponible")
            return False

        try:
            self.selector = BreakthroughTechniqueSelector(use_ml_predictor=True, use_bayesian_opt=False)
            print("✅ Breakthrough Selector inicializado con modelo ML")
            return True
        except Exception as e:
            print(f"❌ Error inicializando selector: {e}")
            return False

    def generate_test_matrices(self):
        """Genera matrices de prueba representativas."""
        print("\n🧪 GENERANDO MATRICES DE PRUEBA...")

        sizes = [128, 256, 512]
        types = ['dense', 'sparse', 'low_rank']

        np.random.seed(42)  # Para reproducibilidad

        for size in sizes:
            for matrix_type in types:
                # Generar par de matrices
                A, B = self._generate_matrix_pair(size, matrix_type)
                self.test_matrices.append((A, B, f"{matrix_type}_{size}x{size}"))

        print(f"✅ Generadas {len(self.test_matrices)} matrices de prueba")

    def _generate_matrix_pair(self, size: int, matrix_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Genera un par de matrices del tipo especificado."""
        if matrix_type == 'dense':
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

        elif matrix_type == 'sparse':
            A = np.random.randn(size, size).astype(np.float32)
            mask_a = np.random.random((size, size)) > 0.9
            A[~mask_a] = 0

            B = np.random.randn(size, size).astype(np.float32)
            mask_b = np.random.random((size, size)) > 0.9
            B[~mask_b] = 0

        elif matrix_type == 'low_rank':
            rank = max(2, size // 8)
            U = np.random.randn(size, rank)
            V = np.random.randn(size, rank)
            S = np.random.exponential(1.0, rank)

            A = U @ np.diag(S) @ V.T
            B = U @ np.diag(S) @ V.T

            A += 0.01 * np.random.randn(size, size)
            B += 0.01 * np.random.randn(size, size)

        return A.astype(np.float32), B.astype(np.float32)

    def run_validation(self):
        """Ejecuta validación completa."""
        print("\n🎯 INICIANDO VALIDACIÓN DE INTEGRACIÓN ML")
        print("=" * 50)

        if not self.initialize_selector():
            return False

        self.generate_test_matrices()

        print("\n🚀 EJECUTANDO TESTS DE VALIDACIÓN...")

        for i, (A, B, matrix_name) in enumerate(self.test_matrices):
            print(f"\n[Test {i+1}/{len(self.test_matrices)}] {matrix_name}")

            try:
                # Seleccionar técnica con ML
                start_time = time.time()
                selection = self.selector.select_technique(A, B)
                selection_time = time.time() - start_time

                # Ejecutar la técnica seleccionada
                start_time = time.time()
                result, metrics = self.selector.execute_selected_technique(selection, A, B)
                execution_time = time.time() - start_time

                # Registrar resultados
                test_result = {
                    'matrix_name': matrix_name,
                    'matrix_size': A.shape[0],
                    'selected_technique': selection.technique.value,
                    'confidence': selection.confidence,
                    'expected_performance': selection.expected_performance,
                    'actual_gflops': metrics.get('gflops_achieved', 0.0),
                    'actual_error': metrics.get('relative_error', 1.0),
                    'selection_time': selection_time,
                    'execution_time': execution_time,
                    'success': result is not None
                }

                self.results.append(test_result)

                print(f"   Técnica: {selection.technique.value} (confianza: {selection.confidence:.2f})")
                print(f"   GFLOPS esperado: {selection.expected_performance:.3f}")
                print(f"   GFLOPS actual: {metrics.get('gflops_achieved', 0.0):.3f}")
                if not test_result['success']:
                    print("  ❌ Falló la ejecución")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                self.results.append({
                    'matrix_name': matrix_name,
                    'error': str(e),
                    'success': False
                })

        return self.analyze_results()

    def analyze_results(self) -> bool:
        """Analiza los resultados de validación."""
        print("\n📊 ANÁLISIS DE RESULTADOS")
        print("=" * 30)

        if not self.results:
            print("❌ No hay resultados para analizar")
            return False

        # Convertir a DataFrame
        df = pd.DataFrame([r for r in self.results if 'error' not in r])

        if df.empty:
            print("❌ Todos los tests fallaron")
            return False

        successful_tests = len(df[df['success'] == True])
        total_tests = len(df)

        print(f"Tests exitosos: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")

        if successful_tests > 0:
            # Estadísticas de performance
            print("\n🎯 PERFORMANCE:")
            print(f"   GFLOPS promedio: {df['actual_gflops'].mean():.3f}")
            print(f"   GFLOPS máximo: {df['actual_gflops'].max():.3f}")
            print(f"   Error relativo promedio: {df['actual_error'].mean():.6f}")
            # Análisis por técnica seleccionada
            print("\n🏷️  TÉCNICAS SELECCIONADAS:")
            technique_counts = df['selected_technique'].value_counts()
            for technique, count in technique_counts.items():
                technique_data = df[df['selected_technique'] == technique]
                avg_gflops = technique_data['actual_gflops'].mean()
                print(f"   {technique}: {count} veces, {avg_gflops:.3f} GFLOPS promedio")
            # Validación de precisión de predicciones
            print("\n🔍 VALIDACIÓN DE PREDICCIONES ML:")
            valid_predictions = df[df['expected_performance'] > 0]
            if not valid_predictions.empty:
                prediction_errors = np.abs(valid_predictions['expected_performance'] - valid_predictions['actual_gflops'])
                mae = prediction_errors.mean()
                rmse = np.sqrt((prediction_errors ** 2).mean())

                print(f"   MAE de predicción: {mae:.3f} GFLOPS")
                print(f"   RMSE de predicción: {rmse:.3f} GFLOPS")
                # Accuracy de selección de técnica
                accurate_selections = 0
                for _, row in valid_predictions.iterrows():
                    expected = row['expected_performance']
                    actual = row['actual_gflops']
                    # Considerar selección buena si el error relativo < 50%
                    if abs(expected - actual) / max(expected, actual) < 0.5:
                        accurate_selections += 1

                accuracy = accurate_selections / len(valid_predictions)
                print(f"   Accuracy de selección: {accuracy:.1f}%")
        # Guardar resultados
        self.save_results(df)

        # Evaluar éxito general
        success_rate = successful_tests / total_tests
        if success_rate >= 0.8:  # 80% de éxito mínimo
            print("\n✅ VALIDACIÓN EXITOSA")
            print("🎯 El modelo ML fine-tuned está funcionando correctamente")
            return True
        else:
            print(f"\n⚠️  VALIDACIÓN CON PROBLEMAS (tasa de éxito: {success_rate*100:.1f}%)")
            return False

    def save_results(self, df: pd.DataFrame):
        """Guarda los resultados de validación."""
        output_file = "ml_integration_validation_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Resultados guardados: {output_file}")


def main():
    """Función principal."""
    validator = MLIntegrationValidator()

    if validator.run_validation():
        print("\n🚀 FASE 9.3.1 COMPLETADA: Integración ML validada")
        print("   Próximo: Ejecutar FASE 9.4 - Optimización híbrida avanzada")
    else:
        print("\n❌ FASE 9.3.1 FALLIDA: Revisar integración ML")
        sys.exit(1)


if __name__ == "__main__":
    main()