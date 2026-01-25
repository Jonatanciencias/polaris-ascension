#!/usr/bin/env python3
"""
🧪 VALIDACIÓN DE TÉCNICAS HÍBRIDAS
==================================

Script de validación para técnicas híbridas de breakthrough.
Prueba selección automática y ejecución de técnicas híbridas.

Técnicas probadas:
- Low-Rank + Coppersmith-Winograd (LR+CW)
- Quantum Annealing + Low-Rank (QA+LR)

Autor: AI Assistant
Fecha: 2026-01-25
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging
from typing import Dict, List, Any

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Importar componentes
try:
    sys.path.append(str(Path(__file__).parent))
    from breakthrough_selector import BreakthroughTechniqueSelector
    from hybrid_optimizer import HybridOptimizer, HybridConfiguration, HybridStrategy
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importando componentes: {e}")
    COMPONENTS_AVAILABLE = False
    sys.exit(1)


class HybridTechniquesValidator:
    """Validador de técnicas híbridas."""

    def __init__(self):
        """Inicializa el validador."""
        self.logger = logging.getLogger(__name__)

        # Inicializar componentes
        self.selector = BreakthroughTechniqueSelector()
        self.hybrid_optimizer = HybridOptimizer()

        self.logger.info("✅ Validador de técnicas híbridas inicializado")

    def generate_test_matrices(self) -> List[Dict[str, Any]]:
        """
        Genera matrices de prueba con diferentes características.

        Returns:
            Lista de configuraciones de prueba
        """
        test_configs = [
            {
                'name': 'dense_256x256',
                'matrix_a': np.random.randn(256, 256).astype(np.float32),
                'matrix_b': np.random.randn(256, 256).astype(np.float32),
                'expected_hybrid': 'lr_cw'  # Densa, buena para CW con LR preprocessing
            },
            {
                'name': 'low_rank_256x256',
                'matrix_a': self._generate_low_rank_matrix(256, 256, rank=64),
                'matrix_b': self._generate_low_rank_matrix(256, 256, rank=64),
                'expected_hybrid': 'lr_cw'  # Low-rank, perfecta para híbrido
            },
            {
                'name': 'sparse_256x256',
                'matrix_a': self._generate_sparse_matrix(256, 256, sparsity=0.9),
                'matrix_b': self._generate_sparse_matrix(256, 256, sparsity=0.9),
                'expected_hybrid': 'lr_cw'  # Sparse, LR puede ayudar
            },
            {
                'name': 'dense_128x128',
                'matrix_a': np.random.randn(128, 128).astype(np.float32),
                'matrix_b': np.random.randn(128, 128).astype(np.float32),
                'expected_hybrid': None  # Pequeña, probablemente traditional
            }
        ]

        self.logger.info(f"✅ Generadas {len(test_configs)} configuraciones de prueba")
        return test_configs

    def _generate_low_rank_matrix(self, rows: int, cols: int, rank: int) -> np.ndarray:
        """Genera una matriz de bajo rango aproximado."""
        A = np.random.randn(rows, rank).astype(np.float32)
        B = np.random.randn(rank, cols).astype(np.float32)
        return np.dot(A, B)

    def _generate_sparse_matrix(self, rows: int, cols: int, sparsity: float) -> np.ndarray:
        """Genera una matriz dispersa."""
        matrix = np.random.randn(rows, cols).astype(np.float32)
        mask = np.random.random((rows, cols)) < sparsity
        matrix[mask] = 0
        return matrix

    def test_hybrid_selection(self, test_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prueba la selección automática de técnicas híbridas.

        Args:
            test_configs: Configuraciones de prueba

        Returns:
            DataFrame con resultados de selección
        """
        self.logger.info("🎯 Iniciando pruebas de selección híbrida")

        results = []

        for config in test_configs:
            self.logger.info(f"🔍 Probando selección para: {config['name']}")

            try:
                # Seleccionar técnica
                selection = self.selector.select_technique(
                    config['matrix_a'], config['matrix_b']
                )

                # Verificar si seleccionó técnica híbrida
                is_hybrid = selection.technique.name.startswith('HYBRID_')

                result = {
                    'test_name': config['name'],
                    'selected_technique': selection.technique.value,
                    'is_hybrid': is_hybrid,
                    'confidence': selection.confidence,
                    'expected_performance': selection.expected_performance,
                    'expected_hybrid': config['expected_hybrid'],
                    'selection_correct': (is_hybrid and config['expected_hybrid'] in selection.technique.value) or
                                       (not is_hybrid and config['expected_hybrid'] is None)
                }

                results.append(result)

                self.logger.info(f"   ✅ Seleccionada: {selection.technique.value} "
                               f"(confianza: {selection.confidence:.2f})")

            except Exception as e:
                self.logger.error(f"❌ Error en selección para {config['name']}: {e}")
                results.append({
                    'test_name': config['name'],
                    'selected_technique': 'ERROR',
                    'is_hybrid': False,
                    'confidence': 0.0,
                    'expected_performance': 0.0,
                    'expected_hybrid': config['expected_hybrid'],
                    'selection_correct': False
                })

        return pd.DataFrame(results)

    def test_hybrid_execution(self, test_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prueba la ejecución de técnicas híbridas seleccionadas.

        Args:
            test_configs: Configuraciones de prueba

        Returns:
            DataFrame con resultados de ejecución
        """
        self.logger.info("🚀 Iniciando pruebas de ejecución híbrida")

        results = []

        for config in test_configs:
            self.logger.info(f"⚙️  Probando ejecución para: {config['name']}")

            try:
                # Calcular resultado de referencia (multiplicación tradicional)
                reference = config['matrix_a'] @ config['matrix_b']
                
                # Seleccionar técnica
                selection = self.selector.select_technique(
                    config['matrix_a'], config['matrix_b']
                )

                # Debug: verificar tipo de selection
                self.logger.info(f"Selection type: {type(selection)}, value: {selection}")
                if hasattr(selection, 'technique'):
                    self.logger.info(f"Technique: {selection.technique}")
                else:
                    self.logger.error(f"Selection no tiene atributo 'technique': {selection}")
                    continue

                # Ejecutar técnica seleccionada
                exec_result = self.selector.execute_selected_technique(
                    config['matrix_a'], config['matrix_b'], selection
                )
                
                # Debug: verificar tipo de exec_result
                self.logger.info(f"Exec result type: {type(exec_result)}, len: {len(exec_result) if hasattr(exec_result, '__len__') else 'N/A'}")
                if isinstance(exec_result, tuple) and len(exec_result) == 2:
                    result, metrics = exec_result
                else:
                    self.logger.error(f"execute_selected_technique no retornó tupla: {exec_result}")
                    continue

                # Validar resultado
                self.logger.info(f"About to validate result. result type: {type(result)}, reference type: {type(reference)}")
                error = np.linalg.norm(result - reference) / np.linalg.norm(reference)

                execution_result = {
                    'test_name': config['name'],
                    'technique': selection.technique.value,
                    'gflops_achieved': metrics.get('gflops_achieved', 0),
                    'execution_time': metrics.get('execution_time', 0),
                    'relative_error': error,
                    'success': metrics.get('success', False),
                    'expected_performance': selection.expected_performance
                }

                results.append(execution_result)

                self.logger.info(f"   ✅ Ejecutada: {selection.technique.value}")
                self.logger.info(f"      GFLOPS: {metrics.get('gflops_achieved', 0):.2f}")
                self.logger.info(f"      Error: {error:.2e}")

            except Exception as e:
                self.logger.error(f"❌ Error en ejecución para {config['name']}: {e}")
                results.append({
                    'test_name': config['name'],
                    'technique': 'ERROR',
                    'gflops_achieved': 0.0,
                    'execution_time': 0.0,
                    'relative_error': float('inf'),
                    'success': False,
                    'expected_performance': 0.0
                })

        return pd.DataFrame(results)

    def test_direct_hybrid_execution(self) -> pd.DataFrame:
        """
        Prueba ejecución directa de técnicas híbridas usando HybridOptimizer.

        Returns:
            DataFrame con resultados de ejecución directa
        """
        self.logger.info("🔧 Probando ejecución directa de híbridos")

        # Configuraciones de prueba directa
        hybrid_configs = [
            {
                'name': 'direct_lr_cw_256',
                'config': HybridConfiguration(
                    strategy=HybridStrategy.SEQUENTIAL,
                    techniques=['lr_cw'],
                    parameters={'lr_cw': {'rank_reduction_factor': 0.7}},
                    weights={'lr_cw': 1.0}
                ),
                'matrix_a': np.random.randn(256, 256).astype(np.float32),
                'matrix_b': np.random.randn(256, 256).astype(np.float32)
            },
            {
                'name': 'direct_qa_lr_128',
                'config': HybridConfiguration(
                    strategy=HybridStrategy.SEQUENTIAL,
                    techniques=['qa_lr'],
                    parameters={'qa_lr': {'iterations': 50}},
                    weights={'qa_lr': 1.0}
                ),
                'matrix_a': self._generate_low_rank_matrix(128, 128, 32),
                'matrix_b': self._generate_low_rank_matrix(128, 128, 32)
            }
        ]

        results = []

        for test_config in hybrid_configs:
            self.logger.info(f"🔧 Ejecutando híbrido directo: {test_config['name']}")

            try:
                start_time = time.time()
                result = self.hybrid_optimizer.optimize_hybrid(
                    test_config['matrix_a'], test_config['matrix_b'], test_config['config']
                )
                execution_time = time.time() - start_time

                # Validar resultado
                reference = np.dot(test_config['matrix_a'], test_config['matrix_b'])
                error = np.linalg.norm(result.final_result - reference) / np.linalg.norm(reference)

                direct_result = {
                    'test_name': test_config['name'],
                    'strategy': test_config['config'].strategy.value,
                    'techniques': ','.join(test_config['config'].techniques),
                    'gflops_achieved': result.combined_performance,
                    'execution_time': execution_time,
                    'relative_error': error,
                    'validation_passed': result.validation_passed,
                    'optimization_path': ','.join(result.optimization_path)
                }

                results.append(direct_result)

                self.logger.info(f"   ✅ Completado: {test_config['name']}")
                self.logger.info(f"      GFLOPS: {result.combined_performance:.2f}")
                self.logger.info(f"      Validación: {'✅' if result.validation_passed else '❌'}")

            except Exception as e:
                self.logger.error(f"❌ Error en híbrido directo {test_config['name']}: {e}")
                results.append({
                    'test_name': test_config['name'],
                    'strategy': 'ERROR',
                    'techniques': 'ERROR',
                    'gflops_achieved': 0.0,
                    'execution_time': 0.0,
                    'relative_error': float('inf'),
                    'validation_passed': False,
                    'optimization_path': 'ERROR'
                })

        return pd.DataFrame(results)

    def run_complete_validation(self) -> Dict[str, pd.DataFrame]:
        """
        Ejecuta validación completa de técnicas híbridas.

        Returns:
            Diccionario con DataFrames de resultados
        """
        self.logger.info("🎯 INICIANDO VALIDACIÓN COMPLETA DE TÉCNICAS HÍBRIDAS")
        self.logger.info("=" * 60)

        # Generar matrices de prueba
        test_configs = self.generate_test_matrices()

        # Ejecutar pruebas
        selection_results = self.test_hybrid_selection(test_configs)
        execution_results = self.test_hybrid_execution(test_configs)
        direct_results = self.test_direct_hybrid_execution()

        # Análisis de resultados
        self._analyze_results(selection_results, execution_results, direct_results)

        results = {
            'selection_results': selection_results,
            'execution_results': execution_results,
            'direct_hybrid_results': direct_results
        }

        # Guardar resultados
        self._save_results(results)

        self.logger.info("✅ VALIDACIÓN COMPLETA FINALIZADA")

        return results

    def _analyze_results(self,
                        selection_df: pd.DataFrame,
                        execution_df: pd.DataFrame,
                        direct_df: pd.DataFrame) -> None:
        """Analiza y muestra estadísticas de los resultados."""
        self.logger.info("📊 ANÁLISIS DE RESULTADOS")
        self.logger.info("-" * 40)

        # Estadísticas de selección
        if not selection_df.empty:
            hybrid_selections = selection_df['is_hybrid'].sum()
            correct_selections = selection_df['selection_correct'].sum()
            avg_confidence = selection_df['confidence'].mean()

            self.logger.info("🎯 SELECCIÓN:")
            self.logger.info(f"   Selecciones híbridas: {hybrid_selections}/{len(selection_df)}")
            self.logger.info(f"   Selecciones correctas: {correct_selections}/{len(selection_df)}")
            self.logger.info(f"   Confianza promedio: {avg_confidence:.2f}")

        # Estadísticas de ejecución
        if not execution_df.empty:
            successful_executions = execution_df['success'].sum()
            avg_gflops = execution_df['gflops_achieved'].mean()
            avg_error = execution_df[execution_df['relative_error'] < 1]['relative_error'].mean()

            self.logger.info("🚀 EJECUCIÓN:")
            self.logger.info(f"   Ejecuciones exitosas: {successful_executions}/{len(execution_df)}")
            self.logger.info(f"   GFLOPS promedio: {avg_gflops:.2f}")
            self.logger.info(f"   Error relativo promedio: {avg_error:.2e}")

        # Estadísticas de ejecución directa
        if not direct_df.empty:
            successful_direct = direct_df['validation_passed'].sum()
            avg_direct_gflops = direct_df['gflops_achieved'].mean()

            self.logger.info("🔧 EJECUCIÓN DIRECTA:")
            self.logger.info(f"   Validaciones pasadas: {successful_direct}/{len(direct_df)}")
            self.logger.info(f"   GFLOPS promedio: {avg_direct_gflops:.2f}")

    def _save_results(self, results: Dict[str, pd.DataFrame]) -> None:
        """Guarda los resultados en archivos CSV."""
        try:
            output_dir = Path(__file__).parent / "validation_results"
            output_dir.mkdir(exist_ok=True)

            for name, df in results.items():
                output_file = output_dir / f"hybrid_validation_{name}_{int(time.time())}.csv"
                df.to_csv(output_file, index=False)
                self.logger.info(f"💾 Resultados guardados: {output_file}")

        except Exception as e:
            self.logger.warning(f"Error guardando resultados: {e}")


def main():
    """Función principal."""
    if not COMPONENTS_AVAILABLE:
        print("❌ Componentes no disponibles")
        return 1

    try:
        validator = HybridTechniquesValidator()
        results = validator.run_complete_validation()

        print("\n🎉 Validación de técnicas híbridas completada exitosamente!")
        return 0

    except Exception as e:
        print(f"❌ Error en validación: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())