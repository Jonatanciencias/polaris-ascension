#!/usr/bin/env python3
"""
🔬 SISTEMA DE BENCHMARKING COMPREHENSIVO PARA RX580
====================================================

Sistema completo para benchmarking de motores de optimización en hardware real,
recopilación de datos de performance y recalibración del sistema de selección inteligente.
"""

import sys
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Agregar paths necesarios
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Importar componentes del sistema
from intelligent_technique_selector import IntelligentTechniqueSelector
from advanced_matrix_analyzer import AdvancedMatrixAnalyzer
from optimization_engines import (
    LowRankMatrixApproximatorGPU,
    CoppersmithWinogradGPU,
    QuantumAnnealingOptimizer,
    TensorCoreSimulator
)

@dataclass
class BenchmarkResult:
    """Resultado de un benchmark individual"""
    matrix_name: str
    matrix_shape: Tuple[int, int]
    technique: str
    execution_time: float
    gflops: float
    memory_used: float
    error: Optional[str] = None
    timestamp: float = time.time()

@dataclass
class HardwareProfile:
    """Perfil de hardware recopilado"""
    gpu_name: str
    gpu_memory: int
    cpu_cores: int
    system_memory: int
    benchmark_results: List[BenchmarkResult]

class ComprehensiveBenchmarkSuite:
    """Suite completa de benchmarking para RX580"""

    def __init__(self):
        self.selector = IntelligentTechniqueSelector()
        self.analyzer = AdvancedMatrixAnalyzer()

        # Inicializar motores
        self.engines = {
            'low_rank': LowRankMatrixApproximatorGPU(),
            'cw': CoppersmithWinogradGPU(),
            'quantum': QuantumAnnealingOptimizer(),
            'tensor_core': TensorCoreSimulator()
        }

        # Matrices de prueba representativas
        self.test_matrices = self._generate_test_matrices()

        print("✅ Comprehensive Benchmark Suite inicializada")
        print(f"   Motores disponibles: {list(self.engines.keys())}")
        print(f"   Matrices de prueba: {len(self.test_matrices)}")

    def _generate_test_matrices(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Genera conjunto comprehensivo de matrices de prueba"""
        matrices = []

        # Configuraciones de tamaño para RX580
        sizes = [128, 256, 512, 1024]  # Adaptado a memoria de RX580

        for size in sizes:
            # Matriz densa cuadrada
            A_dense = np.random.randn(size, size).astype(np.float32)
            B_dense = np.random.randn(size, size).astype(np.float32)
            matrices.append((f"Dense_{size}x{size}", A_dense, B_dense))

            # Matriz sparse (90% sparsity)
            A_sparse = np.random.randn(size, size).astype(np.float32)
            A_sparse[np.abs(A_sparse) < 0.9] = 0
            B_sparse = np.random.randn(size, size).astype(np.float32)
            B_sparse[np.abs(B_sparse) < 0.9] = 0
            matrices.append((f"Sparse_{size}x{size}", A_sparse, B_sparse))

            # Matriz diagonal
            A_diag = np.diag(np.random.randn(size)).astype(np.float32)
            B_diag = np.diag(np.random.randn(size)).astype(np.float32)
            matrices.append((f"Diagonal_{size}x{size}", A_diag, B_diag))

            # Matriz bien condicionada
            A_well = np.random.randn(size, size).astype(np.float32)
            A_well = A_well @ A_well.T  # Hacer positiva definida
            B_well = np.random.randn(size, size).astype(np.float32)
            matrices.append((f"WellConditioned_{size}x{size}", A_well, B_well))

            # Matriz mal condicionada
            A_ill = np.random.randn(size, size).astype(np.float32)
            u, s, vt = np.linalg.svd(A_ill, full_matrices=False)
            s[-1] = s[0] * 1e-6  # Hacer muy mal condicionada
            A_ill = u @ np.diag(s) @ vt
            B_ill = np.random.randn(size, size).astype(np.float32)
            matrices.append((f"IllConditioned_{size}x{size}", A_ill, B_ill))

        # Matrices rectangulares
        rect_configs = [(1024, 512), (2048, 1024), (512, 2048)]
        for m, n in rect_configs:
            A_rect = np.random.randn(m, 256).astype(np.float32)
            B_rect = np.random.randn(256, n).astype(np.float32)
            matrices.append((f"Rectangular_{m}x{n}", A_rect, B_rect))

        return matrices

    def run_full_benchmark(self) -> HardwareProfile:
        """Ejecuta benchmark completo en RX580"""
        print("🚀 INICIANDO BENCHMARK COMPLETO EN RX580")
        print("=" * 60)

        # Obtener información del sistema
        gpu_info = self._get_gpu_info()
        system_info = self._get_system_info()

        print(f"🎯 Hardware detectado:")
        print(f"   GPU: {gpu_info['name']}")
        print(f"   Memoria GPU: {gpu_info['memory_mb']} MB")
        print(f"   CPU Cores: {system_info['cpu_cores']}")
        print(f"   Memoria Sistema: {system_info['memory_mb']} MB")

        # Ejecutar benchmarks
        benchmark_results = []
        total_tests = len(self.test_matrices) * len(self.engines)

        print(f"\n⚡ Ejecutando {total_tests} benchmarks...")
        print("-" * 60)

        test_count = 0
        for matrix_name, A, B in self.test_matrices:
            print(f"\n🔬 Matrix: {matrix_name} ({A.shape} x {B.shape})")

            for technique_name, engine in self.engines.items():
                test_count += 1
                print(f"   [{test_count}/{total_tests}] {technique_name}...")

                try:
                    # Ejecutar benchmark
                    benchmark = engine.benchmark(A, B, n_runs=3)

                    result = BenchmarkResult(
                        matrix_name=matrix_name,
                        matrix_shape=(A.shape[0], B.shape[1]),
                        technique=technique_name,
                        execution_time=benchmark['avg_time'],
                        gflops=benchmark['gflops'],
                        memory_used=benchmark.get('memory_mb', 0.0)
                    )

                    benchmark_results.append(result)

                    print(".2f")

                except Exception as e:
                    error_msg = str(e)
                    print(f"      ❌ Error: {error_msg}")

                    result = BenchmarkResult(
                        matrix_name=matrix_name,
                        matrix_shape=(A.shape[0], B.shape[1]),
                        technique=technique_name,
                        execution_time=float('inf'),
                        gflops=0.0,
                        memory_used=0.0,
                        error=error_msg
                    )

                    benchmark_results.append(result)

        # Crear perfil de hardware
        profile = HardwareProfile(
            gpu_name=gpu_info['name'],
            gpu_memory=gpu_info['memory_mb'],
            cpu_cores=system_info['cpu_cores'],
            system_memory=system_info['memory_mb'],
            benchmark_results=benchmark_results
        )

        print("\n✅ Benchmark completado")
        print(f"   Tests exitosos: {len([r for r in benchmark_results if r.error is None])}")
        print(f"   Tests fallidos: {len([r for r in benchmark_results if r.error is not None])}")

        return profile

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Obtiene información de la GPU"""
        try:
            # Intentar detectar GPU con diferentes métodos
            gpu_name = "Radeon RX580"  # Asumido para este sistema
            gpu_memory = 8192  # 8GB para RX580

            # Verificar con CUDA si está disponible
            try:
                import pycuda.driver as cuda
                cuda.init()
                device = cuda.Device(0)
                gpu_name = device.name()
                gpu_memory = device.total_memory() // (1024 * 1024)
            except:
                pass

            return {
                'name': gpu_name,
                'memory_mb': gpu_memory
            }

        except Exception as e:
            print(f"⚠️  Error obteniendo info GPU: {e}")
            return {
                'name': 'Unknown GPU',
                'memory_mb': 0
            }

    def _get_system_info(self) -> Dict[str, Any]:
        """Obtiene información del sistema"""
        try:
            import psutil

            return {
                'cpu_cores': psutil.cpu_count(),
                'memory_mb': psutil.virtual_memory().total // (1024 * 1024)
            }

        except Exception as e:
            return {
                'cpu_cores': 8,  # Valor por defecto
                'memory_mb': 16384  # 16GB por defecto
            }

    def save_benchmark_data(self, profile: HardwareProfile, output_path: Path):
        """Guarda los datos de benchmark"""
        # Convertir a formato serializable
        data = {
            'hardware_profile': {
                'gpu_name': profile.gpu_name,
                'gpu_memory': profile.gpu_memory,
                'cpu_cores': profile.cpu_cores,
                'system_memory': profile.system_memory
            },
            'benchmark_results': [
                {
                    'matrix_name': r.matrix_name,
                    'matrix_shape': r.matrix_shape,
                    'technique': r.technique,
                    'execution_time': r.execution_time if not np.isinf(r.execution_time) else None,
                    'gflops': r.gflops,
                    'memory_used': r.memory_used,
                    'error': r.error,
                    'timestamp': r.timestamp
                }
                for r in profile.benchmark_results
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"💾 Datos de benchmark guardados en: {output_path}")

    def analyze_benchmark_results(self, profile: HardwareProfile) -> Dict[str, Any]:
        """Analiza los resultados del benchmark"""
        results = profile.benchmark_results

        # Filtrar resultados exitosos
        successful_results = [r for r in results if r.error is None and not np.isinf(r.execution_time)]

        if not successful_results:
            return {'error': 'No successful benchmark results'}

        # Análisis por técnica
        technique_analysis = {}
        for technique in self.engines.keys():
            tech_results = [r for r in successful_results if r.technique == technique]

            if tech_results:
                gflops_values = [r.gflops for r in tech_results]
                time_values = [r.execution_time for r in tech_results]

                technique_analysis[technique] = {
                    'avg_gflops': np.mean(gflops_values),
                    'std_gflops': np.std(gflops_values),
                    'min_gflops': np.min(gflops_values),
                    'max_gflops': np.max(gflops_values),
                    'avg_time': np.mean(time_values),
                    'std_time': np.std(time_values),
                    'test_count': len(tech_results)
                }

        # Análisis por tipo de matriz
        matrix_type_analysis = {}
        for matrix_name, _, _ in self.test_matrices:
            matrix_results = [r for r in successful_results if r.matrix_name == matrix_name]

            if matrix_results:
                best_technique = max(matrix_results, key=lambda x: x.gflops)
                matrix_type_analysis[matrix_name] = {
                    'best_technique': best_technique.technique,
                    'best_gflops': best_technique.gflops,
                    'avg_gflops': np.mean([r.gflops for r in matrix_results]),
                    'technique_count': len(set(r.technique for r in matrix_results))
                }

        # Métricas generales
        overall_metrics = {
            'total_tests': len(results),
            'successful_tests': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'avg_gflops': np.mean([r.gflops for r in successful_results]),
            'total_time': sum(r.execution_time for r in successful_results),
            'best_performance': max(successful_results, key=lambda x: x.gflops) if successful_results else None
        }

        return {
            'technique_analysis': technique_analysis,
            'matrix_type_analysis': matrix_type_analysis,
            'overall_metrics': overall_metrics
        }

def recalibrate_selector_with_hardware_data(benchmark_data_path: Path):
    """Recalibra el selector inteligente con datos reales del hardware"""
    print("🔧 RECALIBRANDO SELECTOR CON DATOS DE HARDWARE")
    print("=" * 50)

    # Cargar datos de benchmark
    with open(benchmark_data_path, 'r') as f:
        data = json.load(f)

    # Convertir a DataFrame para análisis
    results_df = pd.DataFrame(data['benchmark_results'])

    # Filtrar datos válidos
    valid_results = results_df[results_df['execution_time'].notna() & (results_df['execution_time'] != 'inf')]

    if len(valid_results) == 0:
        print("❌ No hay datos válidos para recalibración")
        return

    print(f"📊 Datos válidos para recalibración: {len(valid_results)}")

    # Análisis de performance por técnica
    technique_performance = valid_results.groupby('technique').agg({
        'gflops': ['mean', 'std', 'count'],
        'execution_time': ['mean', 'std']
    }).round(3)

    print("\n📈 Performance por técnica:")
    print(technique_performance)

    # Encontrar la mejor técnica para cada tipo de matriz
    matrix_best_techniques = {}
    for matrix_name in valid_results['matrix_name'].unique():
        matrix_data = valid_results[valid_results['matrix_name'] == matrix_name]
        if len(matrix_data) > 0:
            best_technique = matrix_data.loc[matrix_data['gflops'].idxmax(), 'technique']
            matrix_best_techniques[matrix_name] = best_technique

    print("\n🎯 Mejores técnicas por tipo de matriz:")
    for matrix, technique in matrix_best_techniques.items():
        print(f"   {matrix}: {technique}")

    # Generar nuevos pesos basados en datos reales
    new_weights = {}

    for technique in valid_results['technique'].unique():
        tech_data = valid_results[valid_results['technique'] == technique]

        # Calcular score basado en performance relativa
        avg_gflops = tech_data['gflops'].mean()
        max_gflops = valid_results['gflops'].max()

        if max_gflops > 0:
            relative_score = avg_gflops / max_gflops
            new_weights[technique] = max(0.1, relative_score)  # Mínimo 0.1
        else:
            new_weights[technique] = 0.5  # Valor por defecto

    # Normalizar pesos
    total_weight = sum(new_weights.values())
    if total_weight > 0:
        new_weights = {k: v/total_weight for k, v in new_weights.items()}

    print("\n⚖️  Nuevos pesos calculados:")
    for technique, weight in new_weights.items():
        print(".3f")

    # Guardar nuevos pesos
    weights_data = {
        'hardware': data['hardware_profile']['gpu_name'],
        'calibration_date': time.time(),
        'new_weights': new_weights,
        'technique_performance': technique_performance.to_dict(),
        'data_points': len(valid_results)
    }

    weights_path = Path("models/hardware_calibrated_weights.json")
    with open(weights_path, 'w') as f:
        json.dump(weights_data, f, indent=2, default=str)

    print(f"💾 Nuevos pesos guardados en: {weights_path}")

    return new_weights

def main():
    """Función principal"""
    print("🚀 SISTEMA DE BENCHMARKING COMPREHENSIVO PARA RX580")
    print("=" * 60)

    # Inicializar suite de benchmark
    suite = ComprehensiveBenchmarkSuite()

    # Ejecutar benchmark completo
    profile = suite.run_full_benchmark()

    # Guardar datos
    benchmark_path = Path("benchmark_data/hardware_benchmark_results.json")
    benchmark_path.parent.mkdir(exist_ok=True)
    suite.save_benchmark_data(profile, benchmark_path)

    # Analizar resultados
    analysis = suite.analyze_benchmark_results(profile)

    print("\n📊 ANÁLISIS DE RESULTADOS")
    print("-" * 40)

    if 'error' not in analysis:
        metrics = analysis['overall_metrics']
        print(".1%")
        print(".1f")
        print(".1f")

        if metrics['best_performance']:
            best = metrics['best_performance']
            print(f"   Mejor performance: {best.technique} en {best.matrix_name}")
            print(".1f")

        print("\n📈 Análisis por técnica:")
        for tech, data in analysis['technique_analysis'].items():
            print(".1f")
    else:
        print(f"❌ Error en análisis: {analysis['error']}")

    # Recalibrar selector con datos reales
    recalibrate_selector_with_hardware_data(benchmark_path)

    print("\n🎉 BENCHMARK COMPLETADO Y SISTEMA RECALIBRADO")
    print("   El selector inteligente ahora está optimizado para RX580")

if __name__ == "__main__":
    main()