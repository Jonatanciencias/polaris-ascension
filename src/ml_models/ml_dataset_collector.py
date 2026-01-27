#!/usr/bin/env python3
"""
🎯 DATASET COLLECTION FOR AI KERNEL PREDICTOR FINE-TUNING
==========================================================

Recopila datos exhaustivos de performance para todas las técnicas
breakthrough y tradicionales para re-entrenar el AI Kernel Predictor.

Objetivo: Generar dataset de alta calidad para mejorar accuracy del ML
en selección automática de técnicas óptimas.

Autor: AI Assistant
Fecha: 2026-01-25
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import warnings
warnings.filterwarnings('ignore')

# Importar técnicas disponibles
try:
    # Técnicas breakthrough
    sys.path.append(str(Path(__file__).parent / "fase_9_breakthrough_integration" / "src"))
    from low_rank_matrix_approximator_gpu import GPUAcceleratedLowRankApproximator
    from coppersmith_winograd_gpu import CoppersmithWinogradGPU
    from quantum_annealing_optimizer import QuantumAnnealingMatrixOptimizer

    # AI Kernel Predictor
    sys.path.append(str(Path(__file__).parent / "fase_7_ai_kernel_predictor" / "src"))
    from kernel_predictor import AIKernelPredictor

    TECHNIQUES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Algunas técnicas no disponibles: {e}")
    TECHNIQUES_AVAILABLE = False

class MLDatasetCollector:
    """
    Recopilador de datos para fine-tuning del AI Kernel Predictor.
    """

    def __init__(self):
        """Inicializa el recopilador de datos."""
        self.techniques = {}
        self.predictor = None
        self.dataset = []

        self._initialize_techniques()
        self._initialize_predictor()

    def _initialize_techniques(self):
        """Inicializa todas las técnicas disponibles."""
        if not TECHNIQUES_AVAILABLE:
            return

        try:
            self.techniques['low_rank'] = GPUAcceleratedLowRankApproximator()
            print("✅ Low-Rank GPU inicializado")
        except Exception as e:
            print(f"⚠️  Low-Rank no disponible: {e}")

        try:
            self.techniques['cw'] = CoppersmithWinogradGPU()
            print("✅ Coppersmith-Winograd GPU inicializado")
        except Exception as e:
            print(f"⚠️  CW no disponible: {e}")

        try:
            self.techniques['quantum'] = QuantumAnnealingMatrixOptimizer()
            print("✅ Quantum Annealing inicializado")
        except Exception as e:
            print(f"⚠️  Quantum no disponible: {e}")

    def _initialize_predictor(self):
        """Inicializa el AI Kernel Predictor."""
        try:
            self.predictor = AIKernelPredictor()
            print("✅ AI Kernel Predictor inicializado")
        except Exception as e:
            print(f"⚠️  Predictor no disponible: {e}")

    def generate_matrix_suite(self, sizes: List[int], types: List[str]) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Genera suite de matrices de prueba con diferentes características.

        Args:
            sizes: Lista de tamaños de matriz
            types: Tipos de matrices ('dense', 'sparse', 'low_rank', 'random')

        Returns:
            Lista de tuplas (A, B, type_name)
        """
        matrices = []

        for size in sizes:
            for matrix_type in types:
                # Generar matrices A y B del mismo tipo
                A, B = self._generate_matrix_pair(size, matrix_type)
                type_name = f"{matrix_type}_{size}x{size}"
                matrices.append((A, B, type_name))

        return matrices

    def _generate_matrix_pair(self, size: int, matrix_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Genera un par de matrices del tipo especificado."""
        np.random.seed(42)  # Para reproducibilidad

        if matrix_type == 'dense':
            # Matrices densas aleatorias
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

        elif matrix_type == 'sparse':
            # Matrices sparse (90% zeros)
            A = np.random.randn(size, size).astype(np.float32)
            mask_a = np.random.random((size, size)) > 0.9
            A[~mask_a] = 0

            B = np.random.randn(size, size).astype(np.float32)
            mask_b = np.random.random((size, size)) > 0.9
            B[~mask_b] = 0

        elif matrix_type == 'low_rank':
            # Matrices de bajo rango efectivo
            rank = max(2, size // 8)  # Rango efectivo bajo
            U = np.random.randn(size, rank)
            V = np.random.randn(size, rank)
            S = np.random.exponential(1.0, rank)

            A = U @ np.diag(S) @ V.T
            B = U @ np.diag(S) @ V.T

            # Añadir ruido pequeño
            A += 0.01 * np.random.randn(size, size)
            B += 0.01 * np.random.randn(size, size)

        elif matrix_type == 'random':
            # Matrices completamente aleatorias
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

        else:
            raise ValueError(f"Tipo de matriz desconocido: {matrix_type}")

        return A.astype(np.float32), B.astype(np.float32)

    def benchmark_technique(self, technique_name: str, A: np.ndarray, B: np.ndarray) -> Dict[str, Any]:
        """
        Ejecuta benchmark de una técnica específica.

        Returns:
            Diccionario con métricas de performance
        """
        if technique_name not in self.techniques:
            return {'error': f'Técnica {technique_name} no disponible'}

        technique = self.techniques[technique_name]

        try:
            start_time = time.time()

            if technique_name == 'low_rank':
                result, metrics = technique.optimized_gemm_gpu(A, B)
                gflops = metrics['quality_metrics']['gflops_achieved']
                error = metrics['quality_metrics']['relative_error']

            elif technique_name == 'cw':
                result, metrics = technique.cw_matrix_multiply_gpu(A, B)
                gflops = metrics['gflops_achieved']
                error = metrics['relative_error']

            elif technique_name == 'quantum':
                # Quantum annealing es muy lento, usar versión limitada
                result, metrics = technique.quantum_annealing_optimization(
                    A, B, max_sweeps=10)  # Limitado para dataset
                gflops = metrics.get('gflops_achieved', 0.1)  # Estimación conservadora
                error = metrics.get('error', 1.0)

            execution_time = time.time() - start_time

            return {
                'technique': technique_name,
                'gflops': gflops,
                'error': error,
                'time': execution_time,
                'success': True
            }

        except Exception as e:
            return {
                'technique': technique_name,
                'error': str(e),
                'gflops': 0.0,
                'time': 0.0,
                'success': False
            }

    def collect_dataset(self, matrices: List[Tuple[np.ndarray, np.ndarray, str]],
                       techniques: List[str] = None) -> pd.DataFrame:
        """
        Recopila dataset completo ejecutando todas las técnicas en todas las matrices.

        Args:
            matrices: Lista de matrices de prueba
            techniques: Lista de técnicas a benchmarkear (None = todas disponibles)

        Returns:
            DataFrame con todos los datos recopilados
        """
        if techniques is None:
            techniques = list(self.techniques.keys())

        dataset = []

        total_tests = len(matrices) * len(techniques)
        test_count = 0

        print(f"🚀 Iniciando recopilación de dataset: {total_tests} tests")
        print(f"   Matrices: {len(matrices)}")
        print(f"   Técnicas: {len(techniques)}")
        print()

        for A, B, matrix_name in matrices:
            matrix_size = A.shape[0]

            print(f"🔬 Benchmarking {matrix_name}...")

            for technique in techniques:
                test_count += 1
                print(f"   [{test_count}/{total_tests}] {technique}...")

                # Ejecutar benchmark
                result = self.benchmark_technique(technique, A, B)

                # Extraer características de la matriz
                sparsity_a = 1.0 - np.count_nonzero(A) / A.size
                sparsity_b = 1.0 - np.count_nonzero(B) / B.size

                try:
                    rank_a = np.linalg.matrix_rank(A)
                    rank_b = np.linalg.matrix_rank(B)
                except:
                    rank_a = min(A.shape)
                    rank_b = min(B.shape)

                # Calcular intensidad computacional
                operations = 2 * A.shape[0] * A.shape[1] * B.shape[1]
                memory_access = A.nbytes + B.nbytes + (A.shape[0] * B.shape[1] * 4)
                computational_intensity = operations / memory_access if memory_access > 0 else 1.0

                # Crear registro de datos
                data_point = {
                    'matrix_size': matrix_size,
                    'matrix_type': matrix_name.split('_')[0],
                    'technique': result.get('technique', technique),
                    'gflops_achieved': result.get('gflops', 0.0),
                    'relative_error': result.get('error', 1.0),
                    'execution_time': result.get('time', 0.0),
                    'success': result.get('success', False),
                    'sparsity_a': sparsity_a,
                    'sparsity_b': sparsity_b,
                    'rank_a': rank_a,
                    'rank_b': rank_b,
                    'rank_ratio_a': rank_a / A.shape[0],
                    'rank_ratio_b': rank_b / B.shape[0],
                    'computational_intensity': computational_intensity,
                    'memory_usage_mb': (A.nbytes + B.nbytes) / (1024**2),
                    'timestamp': pd.Timestamp.now()
                }

                dataset.append(data_point)

        # Convertir a DataFrame
        df = pd.DataFrame(dataset)

        # Añadir columna de mejor técnica por matriz
        df['is_best'] = False
        for matrix_name in df['matrix_type'].unique():
            matrix_data = df[df['matrix_type'] == matrix_name]
            if not matrix_data.empty:
                best_idx = matrix_data['gflops_achieved'].idxmax()
                df.loc[best_idx, 'is_best'] = True

        return df

    def save_dataset(self, df: pd.DataFrame, filename: str = "ml_training_dataset.csv"):
        """Guarda el dataset recopilado."""
        df.to_csv(filename, index=False)
        print(f"💾 Dataset guardado: {filename}")
        print(f"   Registros: {len(df)}")
        print(f"   Columnas: {len(df.columns)}")

    def analyze_dataset(self, df: pd.DataFrame):
        """Analiza el dataset recopilado."""
        print("\n📊 ANÁLISIS DEL DATASET RECOPILADO")
        print("=" * 50)

        print(f"Total de registros: {len(df)}")
        print(f"Técnicas evaluadas: {df['technique'].nunique()}")
        print(f"Tipos de matriz: {df['matrix_type'].nunique()}")
        print(f"Tamaños de matriz: {sorted(df['matrix_size'].unique())}")

        print("\n🎯 PERFORMANCE POR TÉCNICA:")
        for technique in df['technique'].unique():
            tech_data = df[df['technique'] == technique]
            successful = tech_data[tech_data['success']]
            if not successful.empty:
                avg_gflops = successful['gflops_achieved'].mean()
                max_gflops = successful['gflops_achieved'].max()
                success_rate = len(successful) / len(tech_data) * 100

                print(f"   {technique}:")
                print(f"     GFLOPS promedio: {avg_gflops:.2f}")
                print(f"     GFLOPS máximo: {max_gflops:.2f}")
                print(f"     Tasa de éxito: {success_rate:.1f}%")

        print("\n🏆 TÉCNICAS MÁS EXITOSAS:")
        best_by_matrix = df[df['is_best']].groupby('technique').size()
        if not best_by_matrix.empty:
            for technique, count in best_by_matrix.items():
                percentage = count / len(df['matrix_type'].unique()) * 100
                print(f"   {technique}: {count} matrices ({percentage:.1f}%)")
def main():
    """Función principal para recopilación de dataset."""
    print("🎯 ML DATASET COLLECTION FOR AI KERNEL PREDICTOR")
    print("=" * 60)
    print("Recopilando datos exhaustivos para fine-tuning del predictor ML")
    print()

    # Configurar tamaños y tipos de matriz
    matrix_sizes = [128, 256, 512]  # Tamaños manejables para recopilación rápida
    matrix_types = ['dense', 'sparse', 'low_rank', 'random']

    # Técnicas a evaluar
    techniques_to_test = ['low_rank', 'cw']  # Excluir quantum por ser muy lento

    print("🔧 CONFIGURACIÓN:")
    print(f"   Tamaños de matriz: {matrix_sizes}")
    print(f"   Tipos de matriz: {matrix_types}")
    print(f"   Técnicas: {techniques_to_test}")
    print(f"   Total de tests: {len(matrix_sizes) * len(matrix_types) * len(techniques_to_test)}")
    print()

    # Inicializar recopilador
    collector = MLDatasetCollector()

    # Generar suite de matrices
    print("🧪 Generando matrices de prueba...")
    matrices = collector.generate_matrix_suite(matrix_sizes, matrix_types)
    print(f"   Generadas {len(matrices)} pares de matrices")
    print()

    # Recopilar dataset
    print("🚀 Iniciando recopilación de datos...")
    start_time = time.time()

    dataset = collector.collect_dataset(matrices, techniques_to_test)

    collection_time = time.time() - start_time
    print(f"✅ Recopilación completada en {collection_time:.1f} segundos")
    # Analizar dataset
    collector.analyze_dataset(dataset)

    # Guardar dataset
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ml_training_dataset_{timestamp}.csv"
    collector.save_dataset(dataset, filename)

    # Guardar también en formato numpy para uso posterior
    np.savez(f"ml_training_dataset_{timestamp}.npz",
             data=dataset.to_numpy(),
             columns=dataset.columns.tolist(),
             metadata={
                 'collection_time': collection_time,
                 'matrix_sizes': matrix_sizes,
                 'matrix_types': matrix_types,
                 'techniques': techniques_to_test,
                 'total_tests': len(dataset)
             })

    print("\n✅ RECOPILACIÓN DE DATASET COMPLETADA")
    print("🎯 PRÓXIMOS PASOS:")
    print("   1. Revisar datos recopilados")
    print("   2. Limpiar datos outliers")
    print("   3. Re-entrenar AI Kernel Predictor")
    print("   4. Validar mejoras en accuracy")

if __name__ == "__main__":
    main()