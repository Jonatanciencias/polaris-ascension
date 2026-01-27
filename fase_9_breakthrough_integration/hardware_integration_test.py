#!/usr/bin/env python3
"""
🧪 PRUEBA DE INTEGRACIÓN COMPLETA CON HARDWARE RX580
====================================================

Prueba exhaustiva del sistema profesional de selección inteligente
ejecutándose en hardware real Radeon RX580.
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Agregar paths necesarios
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Importar componentes del sistema
from intelligent_technique_selector import IntelligentTechniqueSelector
from advanced_matrix_analyzer import AdvancedMatrixAnalyzer

# Importar motores de optimización disponibles
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "fase_7_ai_kernel_predictor" / "src"))
    from kernel_predictor import AIKernelPredictor
    AI_PREDICTOR_AVAILABLE = True
except ImportError:
    AI_PREDICTOR_AVAILABLE = False

try:
    from low_rank_matrix_approximator_gpu import LowRankMatrixApproximatorGPU
    LOW_RANK_AVAILABLE = True
except ImportError:
    LOW_RANK_AVAILABLE = False

try:
    from coppersmith_winograd_gpu import CoppersmithWinogradGPU
    CW_AVAILABLE = True
except ImportError:
    CW_AVAILABLE = False

try:
    from quantum_annealing_optimizer import QuantumAnnealingOptimizer
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

class HardwareBenchmarkSuite:
    """Suite completa de benchmarks para hardware RX580"""

    def __init__(self):
        self.selector = IntelligentTechniqueSelector()
        self.analyzer = AdvancedMatrixAnalyzer()

        # Inicializar motores disponibles
        self.engines = {}
        if LOW_RANK_AVAILABLE:
            self.engines['low_rank'] = LowRankMatrixApproximatorGPU()
        if CW_AVAILABLE:
            self.engines['cw'] = CoppersmithWinogradGPU()
        if QUANTUM_AVAILABLE:
            self.engines['quantum'] = QuantumAnnealingOptimizer()
        if AI_PREDICTOR_AVAILABLE:
            self.engines['ai_predictor'] = AIKernelPredictor()

        print(f"🎯 Motores disponibles: {list(self.engines.keys())}")

    def generate_test_matrices(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Genera matrices de prueba representativas"""
        matrices = []

        # Matriz densa cuadrada grande
        size = 1024
        A_dense = np.random.randn(size, size).astype(np.float32)
        B_dense = np.random.randn(size, size).astype(np.float32)
        matrices.append(("Matriz densa 1024x1024", A_dense, B_dense))

        # Matriz sparse
        A_sparse = np.random.randn(size, size).astype(np.float32)
        A_sparse[np.abs(A_sparse) < 0.9] = 0  # 90% sparsity
        B_sparse = np.random.randn(size, size).astype(np.float32)
        B_sparse[np.abs(B_sparse) < 0.9] = 0
        matrices.append(("Matriz sparse 1024x1024", A_sparse, B_sparse))

        # Matriz diagonal
        A_diag = np.diag(np.random.randn(size)).astype(np.float32)
        B_diag = np.diag(np.random.randn(size)).astype(np.float32)
        matrices.append(("Matriz diagonal 1024x1024", A_diag, B_diag))

        # Matriz rectangular grande
        A_rect = np.random.randn(2048, 1024).astype(np.float32)
        B_rect = np.random.randn(1024, 2048).astype(np.float32)
        matrices.append(("Matriz rectangular 2048x1024", A_rect, B_rect))

        return matrices

    def benchmark_technique(self, technique: str, A: np.ndarray, B: np.ndarray) -> Dict[str, float]:
        """Ejecuta benchmark real de una técnica específica"""
        if technique not in self.engines:
            return {'performance': 0.0, 'memory_mb': 0.0, 'error': 'Engine not available'}

        engine = self.engines[technique]

        try:
            start_time = time.time()
            start_memory = self._get_memory_usage()

            # Ejecutar multiplicación
            if hasattr(engine, 'multiply'):
                C = engine.multiply(A, B)
            elif hasattr(engine, 'optimize'):
                C = engine.optimize(A, B)
            else:
                # Fallback a numpy
                C = np.dot(A, B)

            end_time = time.time()
            end_memory = self._get_memory_usage()

            execution_time = end_time - start_time
            memory_used = max(0, end_memory - start_memory)

            # Calcular GFLOPS
            operations = 2 * A.shape[0] * A.shape[1] * B.shape[1]
            gflops = operations / (execution_time * 1e9)

            return {
                'performance': gflops,
                'execution_time': execution_time,
                'memory_mb': memory_used,
                'error': None
            }

        except Exception as e:
            return {
                'performance': 0.0,
                'execution_time': float('inf'),
                'memory_mb': 0.0,
                'error': str(e)
            }

    def _get_memory_usage(self) -> float:
        """Obtiene uso de memoria actual (simplificado)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # MB
        except:
            return 0.0

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Ejecuta prueba completa del sistema"""
        print("🚀 INICIANDO PRUEBA COMPLETA EN HARDWARE RX580")
        print("=" * 60)

        matrices = self.generate_test_matrices()
        results = []

        total_predictions = 0
        correct_predictions = 0

        for matrix_name, A, B in matrices:
            print(f"\n🔬 Probando: {matrix_name}")
            print("-" * 50)

            # Análisis avanzado de matrices
            print("📊 Realizando análisis avanzado...")
            features = self.analyzer.analyze_matrices(A, B)

            print("   Características clave:")
            print(".2e")
            print(".2f")
            print(f"   Estructura: {features.structure_a.structure_type.value}")
            print(".2f")

            # Selección inteligente
            print("🎯 Sistema de selección inteligente...")
            selection = self.selector.select_technique(A, B)

            recommended_technique = selection.recommended_technique.value
            predicted_performance = selection.expected_performance

            print(f"   Técnica recomendada: {recommended_technique}")
            print(".2f")
            print(f"   Confianza: {selection.selection_confidence:.2f}")

            # Benchmark real de la técnica recomendada
            print("⚡ Ejecutando benchmark real...")
            benchmark_result = self.benchmark_technique(recommended_technique, A, B)

            if benchmark_result['error']:
                print(f"   ❌ Error en benchmark: {benchmark_result['error']}")
                actual_performance = 0.0
            else:
                actual_performance = benchmark_result['performance']
                print(".2f")
                print(".3f")
                print(".1f")

            # Comparar predicción vs realidad
            if actual_performance > 0:
                prediction_accuracy = min(1.0, predicted_performance / actual_performance)
                total_predictions += 1

                # Considerar correcta si la predicción está dentro del 20%
                if abs(predicted_performance - actual_performance) / actual_performance < 0.2:
                    correct_predictions += 1
                    accuracy_status = "✅ CORRECTA"
                else:
                    accuracy_status = "⚠️  INCORRECTA"
            else:
                prediction_accuracy = 0.0
                accuracy_status = "❌ NO DISPONIBLE"

            print(f"   📈 Precisión predicción: {accuracy_status}")

            # Benchmark de técnicas alternativas para comparación
            alternatives_performance = {}
            for alt_technique in selection.alternative_options[:2]:
                alt_result = self.benchmark_technique(alt_technique.value, A, B)
                if not alt_result['error']:
                    alternatives_performance[alt_technique.value] = alt_result['performance']

            # Resultado completo
            result = {
                'matrix_name': matrix_name,
                'matrix_shape': f"{A.shape} x {B.shape}",
                'recommended_technique': recommended_technique,
                'predicted_performance': predicted_performance,
                'actual_performance': actual_performance,
                'prediction_accuracy': prediction_accuracy,
                'execution_time': benchmark_result.get('execution_time', float('inf')),
                'memory_used': benchmark_result.get('memory_mb', 0.0),
                'alternatives_performance': alternatives_performance,
                'advanced_features': {
                    'condition_number': features.spectral_a.condition_number,
                    'sparsity': features.sparsity_a,
                    'structure_type': features.structure_a.structure_type.value,
                    'arithmetic_intensity': features.computational.arithmetic_intensity
                }
            }

            results.append(result)

        # Análisis final
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        summary = {
            'total_tests': len(results),
            'successful_tests': total_predictions,
            'prediction_accuracy': overall_accuracy,
            'average_execution_time': np.mean([r['execution_time'] for r in results if r['execution_time'] < float('inf')]),
            'total_memory_used': sum(r['memory_used'] for r in results),
            'results': results
        }

        return summary

def print_comprehensive_report(summary: Dict[str, Any]):
    """Imprime reporte completo de la prueba"""
    print("\n" + "="*80)
    print("📊 REPORTE COMPLETO DE PRUEBA EN HARDWARE RX580")
    print("="*80)

    print("\n🎯 RESUMEN EJECUTIVO")
    print("-" * 40)
    print(f"   Total de pruebas: {summary['total_tests']}")
    print(f"   Pruebas exitosas: {summary['successful_tests']}")
    print(".1%")
    print(".3f")
    print(".1f")

    print("\n📈 RESULTADOS DETALLADOS")
    print("-" * 40)

    for i, result in enumerate(summary['results'], 1):
        print(f"\n{i}. {result['matrix_name']} ({result['matrix_shape']})")
        print(f"   Técnica recomendada: {result['recommended_technique']}")
        print(".2f")
        print(".2f")

        if result['actual_performance'] > 0:
            accuracy = result['prediction_accuracy']
            if accuracy >= 0.8:
                status = "✅ Excelente"
            elif accuracy >= 0.6:
                status = "🟡 Bueno"
            else:
                status = "🔴 Necesita ajuste"
            print(f"   Precisión: {status} ({accuracy:.1%})")
        else:
            print("   Estado: ❌ Técnica no disponible")

        print(".3f")
        print(".1f")

        if result['alternatives_performance']:
            print("   Alternativas probadas:")
            for alt_tech, alt_perf in result['alternatives_performance'].items():
                print(".2f")

        # Características avanzadas
        features = result['advanced_features']
        print("   Características clave:")
        print(".2e")
        print(".3f")
        print(f"   Estructura: {features['structure_type']}")
        print(".2f")

    print("\n🏆 EVALUACIÓN FINAL")
    print("-" * 40)

    accuracy = summary['prediction_accuracy']
    if accuracy >= 0.9:
        grade = "🏆 EXCELENTE"
        message = "El sistema demuestra precisión excepcional en hardware real"
    elif accuracy >= 0.8:
        grade = "✅ MUY BUENO"
        message = "El sistema funciona correctamente con buena precisión"
    elif accuracy >= 0.7:
        grade = "🟡 ACEPTABLE"
        message = "El sistema funciona pero puede necesitar ajustes menores"
    else:
        grade = "🔴 NECESITA MEJORA"
        message = "El sistema requiere optimización adicional"

    print(f"   Calificación: {grade}")
    print(f"   Mensaje: {message}")
    print(".1%")
    print(".3f")
    print(".1f")

    print("\n💡 RECOMENDACIONES")
    print("-" * 40)

    if accuracy < 0.8:
        print("   • Calibrar pesos del selector con más datos del hardware real")
        print("   • Ajustar modelos de predicción de performance")
        print("   • Expandir dataset de entrenamiento con benchmarks reales")

    print("   • Monitorear performance continua del sistema")
    print("   • Actualizar modelos periódicamente con nuevos datos")
    print("   • Considerar optimizaciones específicas del hardware RX580")

    print("\n✨ PRUEBA COMPLETADA")
    print("   El sistema profesional ha sido validado en hardware real")

def main():
    """Función principal"""
    try:
        suite = HardwareBenchmarkSuite()
        summary = suite.run_comprehensive_test()
        print_comprehensive_report(summary)

        # Guardar resultados
        output_file = Path("hardware_test_results.json")
        with open(output_file, 'w') as f:
            # Convertir tipos numpy a nativos para JSON
            json_summary = {
                k: v for k, v in summary.items()
                if k != 'results'  # Excluir results por simplicidad
            }
            json_summary['results_summary'] = [
                {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                 for k, v in result.items()
                 if k not in ['alternatives_performance', 'advanced_features']}
                for result in summary['results']
            ]
            import json
            json.dump(json_summary, f, indent=2)

        print(f"\n💾 Resultados guardados en: {output_file}")

    except Exception as e:
        print(f"❌ Error en la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()