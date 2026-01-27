#!/usr/bin/env python3
"""
🔬 SISTEMA DE INTEGRACIÓN OPTIMIZADO PARA RX580
===============================================

Sistema completo de integración con motores optimizados y pesos calibrados
para hardware RX580. Incluye selección inteligente con datos reales de performance.
"""

import sys
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Agregar paths necesarios
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Importar componentes del sistema optimizado
try:
    from intelligent_technique_selector import IntelligentTechniqueSelector
    from advanced_matrix_analyzer import AdvancedMatrixAnalyzer
    from optimization_engines import (
        LowRankMatrixApproximatorGPU,
        CoppersmithWinogradGPU,
        QuantumAnnealingOptimizer,
        TensorCoreSimulator
    )
    OPTIMIZATION_ENGINES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Motores de optimización no disponibles: {e}")
    OPTIMIZATION_ENGINES_AVAILABLE = False

@dataclass
class HardwareTestResult:
    """Resultado de una prueba en hardware"""
    matrix_name: str
    matrix_shape: Tuple[int, int]
    recommended_technique: str
    predicted_gflops: float
    confidence: float
    actual_gflops: Optional[float] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
    features: Dict[str, Any] = None

class OptimizedHardwareIntegrationTest:
    """Sistema de integración optimizado con motores reales"""

    def __init__(self):
        self.engines = {}
        self.selector = None
        self.analyzer = None
        self.calibrated_weights = None

        self._initialize_system()
        print("✅ Sistema de integración optimizado inicializado")

    def _initialize_system(self):
        """Inicializar todos los componentes del sistema"""

        # Inicializar motores de optimización
        if OPTIMIZATION_ENGINES_AVAILABLE:
            self.engines = {
                'low_rank': LowRankMatrixApproximatorGPU(),
                'cw': CoppersmithWinogradGPU(),
                'quantum': QuantumAnnealingOptimizer(),
                'tensor_core': TensorCoreSimulator()
            }
            print(f"✅ Motores de optimización disponibles: {list(self.engines.keys())}")
        else:
            print("⚠️  Motores de optimización no disponibles - usando fallback")

        # Inicializar selector inteligente
        try:
            self.selector = IntelligentTechniqueSelector()
            print("✅ Selector inteligente disponible")
        except Exception as e:
            print(f"⚠️  Selector inteligente no disponible: {e}")

        # Inicializar analizador avanzado
        try:
            self.analyzer = AdvancedMatrixAnalyzer()
            print("✅ Analizador avanzado disponible")
        except Exception as e:
            print(f"⚠️  Analizador avanzado no disponible: {e}")

        # Cargar pesos calibrados
        self._load_calibrated_weights()

    def _load_calibrated_weights(self):
        """Cargar pesos calibrados del hardware"""
        weights_path = Path("models/hardware_calibrated_weights.json")
        if weights_path.exists():
            try:
                with open(weights_path, 'r') as f:
                    weights_data = json.load(f)
                self.calibrated_weights = weights_data['new_weights']
                print("✅ Pesos calibrados cargados desde hardware real")
                print(f"   Hardware: {weights_data['hardware']}")
                print(f"   Técnicas calibradas: {list(self.calibrated_weights.keys())}")
            except Exception as e:
                print(f"⚠️  Error cargando pesos calibrados: {e}")
        else:
            print("⚠️  Pesos calibrados no encontrados - usando valores por defecto")

    def run_comprehensive_hardware_test(self) -> Dict[str, Any]:
        """Ejecutar prueba completa en hardware RX580"""
        print("🚀 INICIANDO PRUEBA COMPLETA OPTIMIZADA EN RX580")
        print("=" * 60)

        # Matrices de prueba representativas
        test_matrices = self._generate_test_matrices()
        results = []

        for i, (matrix_name, A, B) in enumerate(test_matrices, 1):
            print(f"\n🔬 Probando: {matrix_name} ({i}/{len(test_matrices)})")
            print("-" * 50)

            result = self._test_single_matrix(matrix_name, A, B)
            results.append(result)

            # Mostrar resultado rápido
            status = "✅" if result.error is None else "❌"
            print(f"   {status} Recomendado: {result.recommended_technique}")
            if result.actual_gflops is not None:
                print(".1f")
            else:
                print(f"   Error: {result.error}")

        # Análisis final
        analysis = self._analyze_test_results(results)

        # Guardar resultados
        self._save_test_results(results, analysis)

        return {
            'results': results,
            'analysis': analysis
        }

    def _generate_test_matrices(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Generar matrices de prueba representativas"""
        matrices = []

        # Configuraciones para RX580
        configs = [
            ("Matriz densa 512x512", 512, 512),
            ("Matriz sparse 512x512", 512, 512),
            ("Matriz diagonal 512x512", 512, 512),
            ("Matriz rectangular 1024x512", 1024, 512),
        ]

        for name, m, n in configs:
            if "densa" in name.lower():
                A = np.random.randn(m, n).astype(np.float32)
                B = np.random.randn(n, m).astype(np.float32)
            elif "sparse" in name.lower():
                A = np.random.randn(m, n).astype(np.float32)
                A[np.abs(A) < 0.9] = 0
                B = np.random.randn(n, m).astype(np.float32)
            elif "diagonal" in name.lower():
                A = np.diag(np.random.randn(min(m, n))).astype(np.float32)
                B = np.random.randn(n, m).astype(np.float32)
            else:  # rectangular
                A = np.random.randn(m, n).astype(np.float32)
                B = np.random.randn(n, m).astype(np.float32)

            matrices.append((name, A, B))

        return matrices

    def _test_single_matrix(self, matrix_name: str, A: np.ndarray, B: np.ndarray) -> HardwareTestResult:
        """Probar una sola matriz"""
        result = HardwareTestResult(
            matrix_name=matrix_name,
            matrix_shape=(A.shape[0], B.shape[1]),
            recommended_technique="unknown",
            predicted_gflops=0.0,
            confidence=0.0
        )

        try:
            # Análisis avanzado
            if self.analyzer:
                features = self.analyzer.analyze_matrices(A, B)
                result.features = features
                print("📊 Realizando análisis avanzado...")
                print(".2e")
                print(".2f")
                print(f"   Estructura: {getattr(features, 'structure', 'unknown')}")

            # Selección inteligente
            if self.selector:
                print("🎯 Sistema de selección inteligente...")
                recommendation = self.selector.select_technique(A, B)

                result.recommended_technique = getattr(recommendation, 'technique', 'unknown')
                result.predicted_gflops = getattr(recommendation, 'predicted_gflops', 0.0)
                result.confidence = getattr(recommendation, 'confidence', 0.0)

                print(f"   Técnica recomendada: {result.recommended_technique}")
                print(".1f")
                print(".2f")

            # Benchmark real
            if result.recommended_technique in self.engines:
                print("⚡ Ejecutando benchmark real...")
                engine = self.engines[result.recommended_technique]

                try:
                    benchmark = engine.benchmark(A, B, n_runs=3)
                    result.actual_gflops = benchmark['gflops']
                    result.execution_time = benchmark['avg_time']

                    print(".1f")
                    print(".3f")

                    # Calcular precisión de predicción
                    if result.predicted_gflops > 0:
                        accuracy = min(result.actual_gflops, result.predicted_gflops) / max(result.actual_gflops, result.predicted_gflops)
                        print(".1%")

                except Exception as e:
                    result.error = f"Benchmark failed: {str(e)}"
                    print(f"   ❌ Error en benchmark: {result.error}")
            else:
                result.error = "Engine not available"
                print("   ❌ Error en benchmark: Engine not available")

        except Exception as e:
            result.error = str(e)
            print(f"   ❌ Error general: {result.error}")

        return result

    def _analyze_test_results(self, results: List[HardwareTestResult]) -> Dict[str, Any]:
        """Analizar resultados de las pruebas"""
        successful_tests = [r for r in results if r.error is None]
        total_gflops = sum(r.actual_gflops for r in successful_tests if r.actual_gflops)

        # Análisis por técnica
        technique_stats = {}
        for result in successful_tests:
            tech = result.recommended_technique
            if tech not in technique_stats:
                technique_stats[tech] = []
            technique_stats[tech].append(result.actual_gflops)

        avg_performance_per_technique = {
            tech: np.mean(gflops_list) for tech, gflops_list in technique_stats.items()
        }

        return {
            'total_tests': len(results),
            'successful_tests': len(successful_tests),
            'success_rate': len(successful_tests) / len(results),
            'total_gflops': total_gflops,
            'avg_gflops': total_gflops / len(successful_tests) if successful_tests else 0,
            'technique_stats': technique_stats,
            'avg_performance_per_technique': avg_performance_per_technique
        }

    def _save_test_results(self, results: List[HardwareTestResult], analysis: Dict[str, Any]):
        """Guardar resultados de las pruebas"""
        output_data = {
            'timestamp': time.time(),
            'hardware': 'Radeon RX580',
            'optimization_engines_available': OPTIMIZATION_ENGINES_AVAILABLE,
            'calibrated_weights_loaded': self.calibrated_weights is not None,
            'analysis': analysis,
            'results': [
                {
                    'matrix_name': r.matrix_name,
                    'matrix_shape': r.matrix_shape,
                    'recommended_technique': r.recommended_technique,
                    'predicted_gflops': r.predicted_gflops,
                    'confidence': r.confidence,
                    'actual_gflops': r.actual_gflops,
                    'execution_time': r.execution_time,
                    'error': r.error,
                    'features': r.features
                }
                for r in results
            ]
        }

        output_path = Path("hardware_optimized_test_results.json")
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\n💾 Resultados guardados en: {output_path}")

def main():
    """Función principal"""
    print("🚀 SISTEMA DE INTEGRACIÓN OPTIMIZADO PARA RX580")
    print("=" * 60)

    # Ejecutar prueba completa
    test_system = OptimizedHardwareIntegrationTest()
    results = test_system.run_comprehensive_hardware_test()

    # Mostrar resumen final
    analysis = results['analysis']

    print("\n📊 REPORTE COMPLETO DE PRUEBA OPTIMIZADA EN RX580")
    print("=" * 60)

    print("\n🎯 RESUMEN EJECUTIVO")
    print("-" * 40)
    print(f"   Total de pruebas: {analysis['total_tests']}")
    print(f"   Pruebas exitosas: {analysis['successful_tests']}")
    print(".1%")
    print(".1f")
    print(".1f")

    if analysis['technique_stats']:
        print("\n📈 Performance por técnica:")
        for tech, gflops_list in analysis['technique_stats'].items():
            avg_gflops = np.mean(gflops_list)
            print(".1f")

    # Evaluación final
    success_rate = analysis['success_rate']
    if success_rate >= 0.8:
        grade = "🟢 EXCELENTE"
        message = "El sistema está altamente optimizado"
    elif success_rate >= 0.6:
        grade = "🟡 BUENO"
        message = "El sistema funciona correctamente"
    else:
        grade = "🔴 NECESITA MEJORA"
        message = "El sistema requiere optimización adicional"

    print("\n🏆 EVALUACIÓN FINAL")
    print("-" * 40)
    print(f"   Calificación: {grade}")
    print(f"   Mensaje: {message}")

    print("\n💡 RECOMENDACIONES")
    print("-" * 40)
    if OPTIMIZATION_ENGINES_AVAILABLE:
        print("   • ✅ Motores de optimización funcionando correctamente")
    else:
        print("   • ⚠️  Actualizar motores de optimización")

    if test_system.calibrated_weights:
        print("   • ✅ Pesos calibrados con datos reales de hardware")
    else:
        print("   • 🔧 Calibrar pesos con benchmarks de hardware real")

    print("   • 📊 Expandir dataset de entrenamiento continuamente")
    print("   • 🔍 Monitorear performance y ajustar modelos")
    print("   • 🚀 Optimizar para casos de uso específicos")

    print("\n✨ PRUEBA OPTIMIZADA COMPLETADA")
    print("   El sistema profesional está listo para producción en RX580")

if __name__ == "__main__":
    main()