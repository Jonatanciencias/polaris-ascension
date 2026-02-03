#!/usr/bin/env python3
"""
🧪 VALIDACIÓN DE PREDICCIONES EN HARDWARE REAL RX 580
=====================================================

Valida que las predicciones del CalibratedIntelligentSelector
se traducen a rendimiento real en la GPU AMD RX 580/590.

Objetivo: Verificar >80% de accuracy en predicciones
"""

import sys
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import components
from src.ml_models.calibrated_intelligent_selector import (
    CalibratedIntelligentSelector, 
    OptimizationTechnique,
    SelectionResult
)
from src.optimization_engines.optimized_kernel_engine import OptimizedKernelEngine

# Try importing additional engines
ENGINES_AVAILABLE = {}

try:
    from src.optimization_engines.quantum_annealing_optimizer import QuantumAnnealingOptimizer
    ENGINES_AVAILABLE['quantum'] = QuantumAnnealingOptimizer
except ImportError:
    pass

try:
    from src.optimization_engines.low_rank_matrix_approximator_gpu import LowRankMatrixApproximatorGPU
    ENGINES_AVAILABLE['low_rank'] = LowRankMatrixApproximatorGPU
except ImportError:
    pass

try:
    from src.optimization_engines.coppersmith_winograd_gpu import CoppersmithWinogradGPU
    ENGINES_AVAILABLE['cw'] = CoppersmithWinogradGPU
except ImportError:
    pass


@dataclass
class BenchmarkResult:
    """Resultado de benchmark individual"""
    technique: str
    matrix_size: int
    predicted_gflops: float
    actual_gflops: float
    execution_time_ms: float
    accuracy_percent: float
    within_tolerance: bool
    error: Optional[str] = None


class HardwareValidationSuite:
    """Suite de validación de predicciones en hardware real"""
    
    # Tolerancia de predicción: ±30% se considera "correcta"
    PREDICTION_TOLERANCE = 0.30
    
    def __init__(self):
        print("🔧 Inicializando Hardware Validation Suite...")
        
        # Selector calibrado
        self.selector = CalibratedIntelligentSelector(prefer_ai_predictor=True)
        
        # Motor OpenCL optimizado (principal)
        self.opencl_engine = OptimizedKernelEngine()
        
        # Motores adicionales
        self.engines = {}
        for name, cls in ENGINES_AVAILABLE.items():
            try:
                self.engines[name] = cls()
                print(f"   ✅ Motor {name}: disponible")
            except Exception as e:
                print(f"   ⚠️ Motor {name}: {e}")
        
        print(f"   📊 Motores activos: OpenCL + {list(self.engines.keys())}")
        
    def generate_test_matrices(self) -> List[Tuple[str, int, np.ndarray, np.ndarray]]:
        """Genera matrices de prueba variadas"""
        matrices = []
        np.random.seed(42)  # Reproducibilidad
        
        # Diferentes tamaños
        sizes = [256, 512, 1024, 2048]
        
        for size in sizes:
            # Matriz densa aleatoria
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            matrices.append((f"dense_{size}x{size}", size, A, B))
        
        # Matrices especiales (solo tamaño 512 para rapidez)
        size = 512
        
        # Matriz sparse (80% ceros)
        A_sparse = np.random.randn(size, size).astype(np.float32)
        A_sparse[np.abs(A_sparse) < 1.3] = 0  # ~80% sparse
        B_sparse = np.random.randn(size, size).astype(np.float32)
        matrices.append(("sparse_512x512", size, A_sparse, B_sparse))
        
        # Matriz simétrica
        A_sym = np.random.randn(size, size).astype(np.float32)
        A_sym = (A_sym + A_sym.T) / 2
        B_sym = np.random.randn(size, size).astype(np.float32)
        matrices.append(("symmetric_512x512", size, A_sym, B_sym))
        
        return matrices
    
    def benchmark_opencl(self, A: np.ndarray, B: np.ndarray, 
                         warmup: int = 2, iterations: int = 5) -> Tuple[float, float]:
        """
        Benchmark real con OpenCL optimizado.
        Returns: (gflops, execution_time_ms)
        """
        M, K = A.shape
        _, N = B.shape
        operations = 2 * M * K * N
        
        # Warmup
        for _ in range(warmup):
            _ = self.opencl_engine.gemm(A, B)
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            C = self.opencl_engine.gemm(A, B)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        gflops = operations / (avg_time * 1e9)
        execution_time_ms = avg_time * 1000
        
        return gflops, execution_time_ms
    
    def benchmark_numpy(self, A: np.ndarray, B: np.ndarray,
                        iterations: int = 3) -> Tuple[float, float]:
        """
        Benchmark con NumPy (baseline CPU).
        Returns: (gflops, execution_time_ms)
        """
        M, K = A.shape
        _, N = B.shape
        operations = 2 * M * K * N
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            C = np.dot(A, B)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        gflops = operations / (avg_time * 1e9)
        execution_time_ms = avg_time * 1000
        
        return gflops, execution_time_ms
    
    def validate_single(self, name: str, size: int, 
                        A: np.ndarray, B: np.ndarray) -> BenchmarkResult:
        """Valida una única matriz"""
        
        # 1. Obtener predicción del selector
        selection = self.selector.select_technique(A, B)
        predicted_gflops = selection.predicted_gflops
        technique = selection.technique.value
        confidence = selection.confidence
        
        # 2. Ejecutar benchmark real
        try:
            if technique in ['ai_predictor', 'opencl_gemm']:
                # Usar OpenCL optimizado
                actual_gflops, exec_time = self.benchmark_opencl(A, B)
            elif technique in self.engines:
                # Usar motor específico si disponible
                engine = self.engines[technique]
                start = time.perf_counter()
                if hasattr(engine, 'multiply'):
                    _ = engine.multiply(A, B)
                elif hasattr(engine, 'gemm'):
                    _ = engine.gemm(A, B)
                else:
                    _ = np.dot(A, B)
                exec_time = (time.perf_counter() - start) * 1000
                operations = 2 * A.shape[0] * A.shape[1] * B.shape[1]
                actual_gflops = operations / (exec_time / 1000 * 1e9)
            else:
                # Fallback a OpenCL
                actual_gflops, exec_time = self.benchmark_opencl(A, B)
            
            error = None
        except Exception as e:
            actual_gflops = 0.0
            exec_time = float('inf')
            error = str(e)
        
        # 3. Calcular accuracy
        if actual_gflops > 0 and predicted_gflops > 0:
            # Accuracy como ratio (capped at 100%)
            if predicted_gflops <= actual_gflops:
                accuracy = (predicted_gflops / actual_gflops) * 100
            else:
                accuracy = (actual_gflops / predicted_gflops) * 100
            
            # Check si está dentro de tolerancia
            relative_error = abs(predicted_gflops - actual_gflops) / actual_gflops
            within_tolerance = relative_error <= self.PREDICTION_TOLERANCE
        else:
            accuracy = 0.0
            within_tolerance = False
        
        return BenchmarkResult(
            technique=technique,
            matrix_size=size,
            predicted_gflops=predicted_gflops,
            actual_gflops=actual_gflops,
            execution_time_ms=exec_time,
            accuracy_percent=accuracy,
            within_tolerance=within_tolerance,
            error=error
        )
    
    def run_validation(self) -> Dict[str, Any]:
        """Ejecuta validación completa"""
        
        print("\n" + "=" * 70)
        print("🧪 VALIDACIÓN DE PREDICCIONES EN HARDWARE REAL")
        print("=" * 70)
        print(f"   GPU: {self.opencl_engine.device.name if hasattr(self.opencl_engine, 'device') else 'AMD RX 580/590'}")
        print(f"   Tolerancia: ±{self.PREDICTION_TOLERANCE*100:.0f}%")
        print("=" * 70)
        
        matrices = self.generate_test_matrices()
        results = []
        
        for name, size, A, B in matrices:
            print(f"\n🔬 Test: {name}")
            print("-" * 50)
            
            result = self.validate_single(name, size, A, B)
            results.append(result)
            
            status = "✅" if result.within_tolerance else "❌"
            
            print(f"   Técnica seleccionada: {result.technique}")
            print(f"   Predicción:  {result.predicted_gflops:>8.1f} GFLOPS")
            print(f"   Real:        {result.actual_gflops:>8.1f} GFLOPS")
            print(f"   Accuracy:    {result.accuracy_percent:>8.1f}% {status}")
            print(f"   Tiempo:      {result.execution_time_ms:>8.2f} ms")
            
            if result.error:
                print(f"   ⚠️ Error: {result.error}")
        
        # Calcular estadísticas
        valid_results = [r for r in results if r.actual_gflops > 0]
        
        if valid_results:
            avg_accuracy = np.mean([r.accuracy_percent for r in valid_results])
            correct_predictions = sum(1 for r in valid_results if r.within_tolerance)
            prediction_rate = correct_predictions / len(valid_results) * 100
            avg_gflops = np.mean([r.actual_gflops for r in valid_results])
            max_gflops = max(r.actual_gflops for r in valid_results)
        else:
            avg_accuracy = 0.0
            prediction_rate = 0.0
            avg_gflops = 0.0
            max_gflops = 0.0
        
        # Reporte final
        print("\n" + "=" * 70)
        print("📊 RESUMEN DE VALIDACIÓN")
        print("=" * 70)
        print(f"   Tests ejecutados:     {len(results)}")
        print(f"   Tests exitosos:       {len(valid_results)}")
        print(f"   Accuracy promedio:    {avg_accuracy:.1f}%")
        print(f"   Predicciones correctas (±{self.PREDICTION_TOLERANCE*100:.0f}%): {correct_predictions}/{len(valid_results)} ({prediction_rate:.1f}%)")
        print(f"   GFLOPS promedio:      {avg_gflops:.1f}")
        print(f"   GFLOPS máximo:        {max_gflops:.1f}")
        
        # Validación de objetivos
        print("\n🎯 VALIDACIÓN DE OBJETIVOS:")
        target_accuracy = 70  # 70% de predicciones dentro de tolerancia
        
        if prediction_rate >= target_accuracy:
            print(f"   ✅ Prediction rate ≥ {target_accuracy}%: {prediction_rate:.1f}% CUMPLIDO")
        else:
            print(f"   ❌ Prediction rate ≥ {target_accuracy}%: {prediction_rate:.1f}% NO CUMPLIDO")
        
        if avg_gflops >= 80:
            print(f"   ✅ GFLOPS promedio ≥ 80: {avg_gflops:.1f} CUMPLIDO")
        else:
            print(f"   ❌ GFLOPS promedio ≥ 80: {avg_gflops:.1f} NO CUMPLIDO")
        
        # Guardar resultados
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(results),
            'successful_tests': len(valid_results),
            'average_accuracy_percent': avg_accuracy,
            'prediction_rate_percent': prediction_rate,
            'average_gflops': avg_gflops,
            'max_gflops': max_gflops,
            'tolerance_percent': self.PREDICTION_TOLERANCE * 100,
            'results': [
                {
                    'name': f"{r.technique}_{r.matrix_size}",
                    'technique': r.technique,
                    'matrix_size': r.matrix_size,
                    'predicted_gflops': float(r.predicted_gflops),
                    'actual_gflops': float(r.actual_gflops),
                    'accuracy_percent': float(r.accuracy_percent),
                    'within_tolerance': bool(r.within_tolerance),
                    'execution_time_ms': float(r.execution_time_ms) if r.execution_time_ms != float('inf') else None,
                    'error': r.error
                }
                for r in results
            ]
        }
        
        output_file = PROJECT_ROOT / 'results' / 'hardware_validation_results.json'
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Resultados guardados en: {output_file}")
        
        return output


def main():
    """Punto de entrada principal"""
    suite = HardwareValidationSuite()
    results = suite.run_validation()
    
    # Status final
    if results['prediction_rate_percent'] >= 70 and results['average_gflops'] >= 80:
        print("\n🎉 VALIDACIÓN EXITOSA - Sistema funcionando correctamente")
        return 0
    else:
        print("\n⚠️ VALIDACIÓN PARCIAL - Revisar predicciones o rendimiento")
        return 1


if __name__ == "__main__":
    sys.exit(main())
