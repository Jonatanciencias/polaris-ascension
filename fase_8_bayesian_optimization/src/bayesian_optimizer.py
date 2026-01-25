#!/usr/bin/env python3
"""
🤖 FASE 8: BAYESIAN OPTIMIZATION FOR KERNEL TUNING
===================================================

Optimización bayesiana para auto-tuning automático de parámetros de kernels GEMM.
Utiliza Gaussian Processes para explorar eficientemente el espacio de parámetros.

Objetivos:
- Exploración inteligente del espacio de hiperparámetros
- Optimización automática de kernels más allá del ML predictor
- Mejora de +15-25% en performance GFLOPS
- Integración con framework GEMM existente

Autor: AI Assistant
Fecha: 2026-01-25
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Dependencias para Bayesian Optimization
try:
    from skopt import gp_minimize, Optimizer
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.callbacks import CheckpointSaver
    SKOPT_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-optimize no disponible. Instalar con: pip install scikit-optimize")
    SKOPT_AVAILABLE = False

try:
    from bayes_opt import BayesianOptimization
    BAYES_OPT_AVAILABLE = True
except ImportError:
    print("⚠️  bayesian-optimization no disponible. Instalar con: pip install bayesian-optimization")
    BAYES_OPT_AVAILABLE = False

# Importar componentes del proyecto
try:
    sys.path.append(str(Path(__file__).parent.parent / "fase_7_ai_kernel_predictor" / "src"))
    from kernel_predictor import AIKernelPredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    print("⚠️  AI Kernel Predictor no disponible")
    PREDICTOR_AVAILABLE = False

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bayesian_optimization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Resultado de una optimización bayesiana."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float
    convergence_info: Dict[str, Any]


class KernelParameterSpace:
    """
    Define el espacio de parámetros para optimización de kernels GEMM.

    Parámetros típicos para kernels tiled/vectorized:
    - TILE_SIZE: Tamaño del bloque de tiling
    - VECTOR_WIDTH: Ancho del vector SIMD
    - WORKGROUP_SIZE: Tamaño del workgroup OpenCL
    - UNROLL_FACTOR: Factor de desenrollado de bucles
    - PREFETCH_DISTANCE: Distancia de prefetch
    """

    def __init__(self):
        """Inicializa el espacio de parámetros."""
        # Definir rangos para cada parámetro
        self.parameter_ranges = {
            'tile_size': (8, 256),  # Tamaño del tile
            'vector_width': (1, 16),  # Ancho del vector
            'workgroup_size': (32, 512),  # Workgroup OpenCL
            'unroll_factor': (1, 8),  # Factor de desenrollado
            'prefetch_distance': (0, 8),  # Prefetch
            'local_memory_factor': (0.1, 2.0),  # Factor de memoria local
        }

    def get_skopt_space(self) -> List:
        """
        Retorna el espacio de parámetros para scikit-optimize.

        Returns:
            Lista de dimensiones para skopt
        """
        return [
            Integer(*self.parameter_ranges['tile_size'], name='tile_size'),
            Integer(*self.parameter_ranges['vector_width'], name='vector_width'),
            Integer(*self.parameter_ranges['workgroup_size'], name='workgroup_size'),
            Integer(*self.parameter_ranges['unroll_factor'], name='unroll_factor'),
            Integer(*self.parameter_ranges['prefetch_distance'], name='prefetch_distance'),
            Real(*self.parameter_ranges['local_memory_factor'], name='local_memory_factor'),
        ]

    def get_bayes_opt_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Retorna los límites para bayesian-optimization.

        Returns:
            Diccionario con límites para cada parámetro
        """
        return {
            'tile_size': self.parameter_ranges['tile_size'],
            'vector_width': self.parameter_ranges['vector_width'],
            'workgroup_size': self.parameter_ranges['workgroup_size'],
            'unroll_factor': self.parameter_ranges['unroll_factor'],
            'prefetch_distance': self.parameter_ranges['prefetch_distance'],
            'local_memory_factor': self.parameter_ranges['local_memory_factor'],
        }


class BayesianKernelOptimizer:
    """
    Optimizador bayesiano para parámetros de kernels GEMM.

    Utiliza Gaussian Processes para modelar la función objetivo
    y decidir dónde muestrear eficientemente.
    """

    def __init__(self,
                 matrix_size: int = 1024,
                 optimization_target: str = 'gflops',
                 max_evaluations: int = 50,
                 random_starts: int = 10,
                 n_jobs: int = 1,
                 use_checkpoint: bool = True):
        """
        Inicializa el optimizador bayesiano.

        Args:
            matrix_size: Tamaño de la matriz para optimización
            optimization_target: Métrica a optimizar ('gflops', 'latency', etc.)
            max_evaluations: Número máximo de evaluaciones
            random_starts: Número de evaluaciones aleatorias iniciales
            n_jobs: Número de jobs paralelos
            use_checkpoint: Si guardar checkpoints
        """
        self.matrix_size = matrix_size
        self.optimization_target = optimization_target
        self.max_evaluations = max_evaluations
        self.random_starts = random_starts
        self.n_jobs = n_jobs
        self.use_checkpoint = use_checkpoint

        # Componentes
        self.param_space = KernelParameterSpace()
        self.predictor = AIKernelPredictor() if PREDICTOR_AVAILABLE else None

        # Estado de optimización
        self.optimization_history = []
        self.best_params = None
        self.best_score = float('-inf')

        # Configurar checkpoint
        if self.use_checkpoint:
            self.checkpoint_path = Path("bayesian_checkpoint.pkl")
        else:
            self.checkpoint_path = None

        logger.info("🤖 Bayesian Kernel Optimizer inicializado")
        logger.info(f"   Matriz: {matrix_size}x{matrix_size}")
        logger.info(f"   Target: {optimization_target}")
        logger.info(f"   Max evaluations: {max_evaluations}")

    def objective_function(self, **params) -> float:
        """
        Función objetivo para optimización.

        Evalúa la performance de un kernel con parámetros dados.
        En producción, esto ejecutaría el kernel real.

        Args:
            params: Parámetros del kernel

        Returns:
            Score de performance (negativo para minimización)
        """
        start_time = time.time()

        try:
            # Simular evaluación del kernel (placeholder)
            # En producción: ejecutar kernel OpenCL con estos parámetros
            score = self._simulate_kernel_evaluation(params)

            evaluation_time = time.time() - start_time

            # Registrar evaluación
            evaluation_record = {
                'params': params.copy(),
                'score': score,
                'evaluation_time': evaluation_time,
                'timestamp': time.time(),
                'matrix_size': self.matrix_size
            }

            self.optimization_history.append(evaluation_record)

            # Actualizar mejor score
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                logger.info(f"🆕 Nuevo mejor score: {score:.2f} GFLOPS")
            logger.debug(f"Evaluación completada: {score:.2f} GFLOPS en {evaluation_time:.3f}s")

            return score

        except Exception as e:
            logger.error(f"Error en evaluación: {e}")
            return float('-inf')

    def _simulate_kernel_evaluation(self, params: Dict[str, Any]) -> float:
        """
        Simula la evaluación de un kernel con parámetros dados.

        En producción, esto debería:
        1. Generar/compilar kernel OpenCL con parámetros
        2. Ejecutar benchmark
        3. Medir performance real

        Args:
            params: Parámetros del kernel

        Returns:
            Performance simulada en GFLOPS
        """
        # Baseline performance (estimación conservadora)
        baseline_gflops = 50.0

        # Factores de mejora basados en parámetros
        tile_factor = min(2.0, params['tile_size'] / 64.0)  # Mejor tiling = mejor performance
        vector_factor = min(4.0, params['vector_width'] / 4.0)  # SIMD mejora
        workgroup_factor = min(1.5, params['workgroup_size'] / 128.0)  # Workgroup óptimo
        unroll_factor = min(1.3, params['unroll_factor'] / 2.0)  # Desenrollado
        prefetch_factor = min(1.2, (params['prefetch_distance'] + 1) / 4.0)  # Prefetch
        memory_factor = min(1.4, params['local_memory_factor'])  # Memoria local

        # Combinar factores con algo de ruido realista
        total_factor = (tile_factor * vector_factor * workgroup_factor *
                       unroll_factor * prefetch_factor * memory_factor)

        # Añadir ruido gaussiano para simular variabilidad real
        noise = np.random.normal(0, 0.05)  # ±5% variabilidad
        final_gflops = baseline_gflops * total_factor * (1 + noise)

        # Asegurar valores razonables
        final_gflops = max(10.0, min(500.0, final_gflops))

        return final_gflops

    def optimize_with_skopt(self) -> OptimizationResult:
        """
        Ejecuta optimización usando scikit-optimize.

        Returns:
            Resultado de la optimización
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize no está disponible")

        logger.info("🚀 Iniciando optimización con scikit-optimize")

        start_time = time.time()

        # Definir función objetivo para skopt
        @use_named_args(self.param_space.get_skopt_space())
        def skopt_objective(**params):
            return -self.objective_function(**params)  # Negativo para minimización

        # Configurar callbacks
        callbacks = []
        if self.use_checkpoint:
            callbacks.append(CheckpointSaver(self.checkpoint_path))

        # Ejecutar optimización
        result = gp_minimize(
            func=skopt_objective,
            dimensions=self.param_space.get_skopt_space(),
            n_calls=self.max_evaluations,
            n_random_starts=self.random_starts,
            callback=callbacks,
            random_state=42,
            verbose=True
        )

        optimization_time = time.time() - start_time

        # Preparar resultado
        opt_result = OptimizationResult(
            best_params=dict(zip([dim.name for dim in self.param_space.get_skopt_space()],
                               result.x)),
            best_score=-result.fun,  # Negativo de vuelta
            optimization_history=self.optimization_history,
            total_evaluations=len(self.optimization_history),
            optimization_time=optimization_time,
            convergence_info={
                'convergence_value': result.fun,
                'n_evaluations': len(result.func_vals),
                'method': 'scikit-optimize'
            }
        )

        logger.info("✅ Optimización completada con scikit-optimize")
        logger.info(f"   Mejor score: {opt_result.best_score:.2f} GFLOPS")
        logger.info(f"   Tiempo total: {optimization_time:.2f}s")

        return opt_result

    def optimize_with_bayes_opt(self) -> OptimizationResult:
        """
        Ejecuta optimización usando bayesian-optimization.

        Returns:
            Resultado de la optimización
        """
        if not BAYES_OPT_AVAILABLE:
            raise ImportError("bayesian-optimization no está disponible")

        logger.info("🚀 Iniciando optimización con bayesian-optimization")

        start_time = time.time()

        # Configurar optimizer
        optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=self.param_space.get_bayes_opt_bounds(),
            random_state=42,
            verbose=2
        )

        # Ejecutar optimización
        optimizer.maximize(
            init_points=self.random_starts,
            n_iter=self.max_evaluations - self.random_starts
        )

        optimization_time = time.time() - start_time

        # Preparar resultado
        opt_result = OptimizationResult(
            best_params=optimizer.max['params'],
            best_score=optimizer.max['target'],
            optimization_history=self.optimization_history,
            total_evaluations=len(self.optimization_history),
            optimization_time=optimization_time,
            convergence_info={
                'total_evaluations': len(optimizer.res),
                'method': 'bayesian-optimization'
            }
        )

        logger.info("✅ Optimización completada con bayesian-optimization")
        logger.info(f"   Mejor score: {opt_result.best_score:.2f} GFLOPS")
        logger.info(f"   Tiempo total: {optimization_time:.2f}s")

        return opt_result

    def run_optimization(self, method: str = 'auto') -> OptimizationResult:
        """
        Ejecuta la optimización completa.

        Args:
            method: Método a usar ('skopt', 'bayes_opt', 'auto')

        Returns:
            Resultado de la optimización
        """
        if method == 'auto':
            if SKOPT_AVAILABLE:
                method = 'skopt'
            elif BAYES_OPT_AVAILABLE:
                method = 'bayes_opt'
            else:
                raise ImportError("Ninguna librería de Bayesian Optimization disponible")

        logger.info(f"🎯 Ejecutando optimización con método: {method}")

        if method == 'skopt':
            return self.optimize_with_skopt()
        elif method == 'bayes_opt':
            return self.optimize_with_bayes_opt()
        else:
            raise ValueError(f"Método {method} no soportado")

    def save_results(self, result: OptimizationResult, filename: str = None):
        """
        Guarda los resultados de la optimización.

        Args:
            result: Resultado a guardar
            filename: Nombre del archivo (opcional)
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"bayesian_optimization_results_{timestamp}.json"

        # Convertir a diccionario serializable
        result_dict = {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'total_evaluations': result.total_evaluations,
            'optimization_time': result.optimization_time,
            'convergence_info': result.convergence_info,
            'optimization_history': result.optimization_history,
            'metadata': {
                'matrix_size': self.matrix_size,
                'optimization_target': self.optimization_target,
                'timestamp': time.time()
            }
        }

        # Guardar como JSON
        with open(filename, 'w') as f:
            import json
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"💾 Resultados guardados en: {filename}")

    def plot_optimization_history(self, result: OptimizationResult):
        """
        Genera gráficos del historial de optimización.

        Args:
            result: Resultado de la optimización
        """
        try:
            import matplotlib.pyplot as plt

            # Extraer datos
            scores = [record['score'] for record in result.optimization_history]
            evaluations = list(range(1, len(scores) + 1))

            # Gráfico de convergencia
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.plot(evaluations, scores, 'b-', alpha=0.7)
            plt.plot(evaluations, scores, 'ro', markersize=3)
            plt.xlabel('Evaluación')
            plt.ylabel('GFLOPS')
            plt.title('Historial de Optimización')
            plt.grid(True, alpha=0.3)

            # Gráfico de mejores scores acumulados
            best_scores = []
            current_best = float('-inf')
            for score in scores:
                current_best = max(current_best, score)
                best_scores.append(current_best)

            plt.subplot(2, 2, 2)
            plt.plot(evaluations, best_scores, 'g-', linewidth=2)
            plt.xlabel('Evaluación')
            plt.ylabel('Mejor GFLOPS')
            plt.title('Mejor Score Acumulado')
            plt.grid(True, alpha=0.3)

            # Distribución de parámetros
            params_df = pd.DataFrame([r['params'] for r in result.optimization_history])

            plt.subplot(2, 2, 3)
            params_df.boxplot()
            plt.xticks(rotation=45)
            plt.title('Distribución de Parámetros')
            plt.tight_layout()

            # Scatter plot de dos parámetros principales
            plt.subplot(2, 2, 4)
            plt.scatter(params_df['tile_size'], params_df['vector_width'],
                       c=scores, cmap='viridis', alpha=0.7)
            plt.xlabel('Tile Size')
            plt.ylabel('Vector Width')
            plt.title('Tile Size vs Vector Width')
            plt.colorbar(label='GFLOPS')

            plt.tight_layout()
            plt.savefig('bayesian_optimization_plots.png', dpi=300, bbox_inches='tight')
            logger.info("📊 Gráficos guardados en: bayesian_optimization_plots.png")

        except ImportError:
            logger.warning("matplotlib no disponible para generar gráficos")


def main():
    """Función principal para ejecutar optimización bayesiana."""
    print("🤖 FASE 8: BAYESIAN OPTIMIZATION FOR KERNEL TUNING")
    print("=" * 60)

    # Configurar optimizador
    optimizer = BayesianKernelOptimizer(
        matrix_size=1024,
        max_evaluations=30,  # Reducido para demo
        random_starts=5
    )

    try:
        # Ejecutar optimización
        print("🚀 Iniciando optimización bayesiana...")
        result = optimizer.run_optimization(method='auto')

        # Mostrar resultados
        print("\n📊 RESULTADOS DE OPTIMIZACIÓN:")
        print("-" * 40)
        print(f"Mejores parámetros: {result.best_params}")
        print(f"Mejor performance: {result.best_score:.2f} GFLOPS")
        print(f"Evaluaciones totales: {result.total_evaluations}")
        print(f"Tiempo total: {result.optimization_time:.2f}s")
        # Guardar resultados
        optimizer.save_results(result)

        # Generar gráficos
        optimizer.plot_optimization_history(result)

        print("\n✅ Optimización completada exitosamente!")
        print("📁 Resultados guardados en archivos locales")

    except Exception as e:
        print(f"❌ Error en optimización: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()