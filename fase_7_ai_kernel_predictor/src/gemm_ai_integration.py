#!/usr/bin/env python3
"""
🤖 FASE 7: AI KERNEL PREDICTOR - GEMM INTEGRATION
=================================================

Integración del predictor de kernels AI con el framework GEMM existente.
Permite selección automática de kernels basada en predicciones de ML.

Objetivos:
- Integración seamless con GEMM framework
- Selección automática de kernel óptimo
- Fallback a kernels manuales si es necesario
- Logging y monitoring de decisiones

Autor: AI Assistant
Fecha: 2024
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Importar predictor AI
try:
    from kernel_predictor import AIKernelPredictor
except ImportError:
    print("❌ Error: No se puede importar AIKernelPredictor")
    print("   Asegúrate de que kernel_predictor.py esté en el mismo directorio")
    sys.exit(1)

class GEMMAIKernelSelector:
    """
    Selector de kernels que integra AI predictions con el framework GEMM existente.
    """

    def __init__(self, predictor_path: str = None, enable_ai: bool = True,
                 fallback_mode: str = 'best_available'):
        """
        Inicializa el selector de kernels.

        Args:
            predictor_path: Path al directorio del predictor AI
            enable_ai: Si usar predicciones AI o modo manual
            fallback_mode: Modo de fallback ('best_available', 'gcn4_optimized', 'strassen')
        """
        self.enable_ai = enable_ai
        self.fallback_mode = fallback_mode
        self.predictor = None

        # Configurar logging
        self._setup_logging()

        if self.enable_ai:
            try:
                self.predictor = AIKernelPredictor(predictor_path)
                self.logger.info("✅ AI Kernel Predictor inicializado")
            except Exception as e:
                self.logger.warning(f"⚠️  No se pudo inicializar AI predictor: {e}")
                self.logger.info("🔄 Cambiando a modo fallback")
                self.enable_ai = False

        # Mapear kernels disponibles en el sistema GEMM
        self.available_kernels = {
            'gcn4_optimized': self._run_gcn4_kernel,
            'strassen': self._run_strassen_kernel,
            'unknown': self._run_basic_kernel  # fallback
        }

        # Estadísticas de uso
        self.stats = {
            'total_selections': 0,
            'ai_selections': 0,
            'fallback_selections': 0,
            'performance_predictions': [],
            'actual_performance': []
        }

    def _setup_logging(self):
        """Configura el sistema de logging."""
        self.logger = logging.getLogger('GEMM_AI_Selector')
        self.logger.setLevel(logging.INFO)

        # Crear directorio de logs si no existe
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Handler para archivo
        fh = logging.FileHandler(log_dir / "gemm_ai_selector.log")
        fh.setLevel(logging.INFO)

        # Handler para consola
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        # Formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def select_and_run_kernel(self, matrix_a: np.ndarray, matrix_b: np.ndarray,
                             optimization_level: int = 1) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Selecciona automáticamente el mejor kernel y ejecuta la operación GEMM.

        Args:
            matrix_a: Matriz A (MxK)
            matrix_b: Matriz B (KxN)
            optimization_level: Nivel de optimización (1-3)

        Returns:
            Tupla de (resultado, metadata)
        """
        if matrix_a.shape[1] != matrix_b.shape[0]:
            raise ValueError("Dimensiones incompatibles para multiplicación de matrices")

        matrix_size = max(matrix_a.shape[0], matrix_a.shape[1], matrix_b.shape[1])
        self.stats['total_selections'] += 1

        # Seleccionar kernel
        if self.enable_ai and self.predictor:
            kernel_choice = self._select_kernel_ai(matrix_size, optimization_level)
            self.stats['ai_selections'] += 1
        else:
            kernel_choice = self._select_kernel_fallback(matrix_size)
            self.stats['fallback_selections'] += 1

        self.logger.info(f"🎯 Kernel seleccionado: {kernel_choice} para matriz {matrix_size}x{matrix_size}")

        # Ejecutar kernel
        start_time = time.time()
        try:
            result = self.available_kernels[kernel_choice](matrix_a, matrix_b)
            execution_time = time.time() - start_time

            # Calcular performance
            operations = 2 * matrix_a.shape[0] * matrix_a.shape[1] * matrix_b.shape[1]
            gflops = operations / (execution_time * 1e9)

            self.stats['actual_performance'].append(gflops)

            metadata = {
                'kernel_used': kernel_choice,
                'matrix_size': matrix_size,
                'execution_time': execution_time,
                'gflops_achieved': gflops,
                'ai_enabled': self.enable_ai,
                'optimization_level': optimization_level
            }

            if self.enable_ai and self.predictor:
                pred = self.predictor.predict_best_kernel(matrix_size, optimization_level)
                metadata['predicted_gflops'] = pred['predicted_performance']
                metadata['prediction_confidence'] = pred['confidence_score']
                self.stats['performance_predictions'].append(pred['predicted_performance'])

            self.logger.info(f"✅ GEMM completado: {gflops:.2f} GFLOPS")
            return result, metadata

        except Exception as e:
            self.logger.error(f"❌ Error ejecutando kernel {kernel_choice}: {e}")
            # Fallback a kernel básico
            return self._run_basic_kernel(matrix_a, matrix_b), {
                'kernel_used': 'fallback_basic',
                'error': str(e)
            }

    def _select_kernel_ai(self, matrix_size: int, optimization_level: int) -> str:
        """Selecciona kernel usando predicciones AI."""
        try:
            prediction = self.predictor.predict_best_kernel(matrix_size, optimization_level)
            kernel = prediction['best_kernel']

            # Mapear 'unknown' a un kernel disponible
            if kernel == 'unknown':
                # Para unknown, elegir basado en tamaño
                if matrix_size >= 1024:
                    kernel = 'gcn4_optimized'
                else:
                    kernel = 'strassen'

            # Verificar que el kernel esté disponible
            if kernel not in self.available_kernels:
                kernel = 'gcn4_optimized'  # fallback seguro

            return kernel

        except Exception as e:
            self.logger.warning(f"⚠️  Error en predicción AI: {e}")
            return self._select_kernel_fallback(matrix_size)

    def _select_kernel_fallback(self, matrix_size: int) -> str:
        """Selecciona kernel en modo fallback."""
        if self.fallback_mode == 'best_available':
            # Lógica simple basada en tamaño
            if matrix_size >= 1024:
                return 'gcn4_optimized'
            else:
                return 'strassen'
        elif self.fallback_mode == 'gcn4_optimized':
            return 'gcn4_optimized'
        elif self.fallback_mode == 'strassen':
            return 'strassen'
        else:
            return 'gcn4_optimized'  # default seguro

    def _run_gcn4_kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Ejecuta kernel GCN4 optimizado."""
        # Placeholder - integrar con kernel real GCN4
        self.logger.info("🚀 Ejecutando kernel GCN4 optimizado")
        return np.dot(a, b)

    def _run_strassen_kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Ejecuta kernel Strassen."""
        # Placeholder - integrar con kernel real Strassen
        self.logger.info("🚀 Ejecutando kernel Strassen")
        return np.dot(a, b)

    def _run_basic_kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Ejecuta kernel básico."""
        # Placeholder - integrar con kernel real básico
        self.logger.info("🚀 Ejecutando kernel básico")
        return np.dot(a, b)

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de uso."""
        stats = self.stats.copy()

        if self.stats['performance_predictions']:
            stats['avg_prediction_error'] = np.mean([
                abs(p - a) for p, a in zip(
                    self.stats['performance_predictions'],
                    self.stats['actual_performance']
                )
            ])

        return stats

    def enable_ai_mode(self, enabled: bool = True):
        """Habilita/deshabilita modo AI."""
        self.enable_ai = enabled and (self.predictor is not None)
        self.logger.info(f"🎛️  Modo AI: {'habilitado' if self.enable_ai else 'deshabilitado'}")

def benchmark_ai_selector():
    """Benchmark del selector AI con diferentes tamaños de matriz."""
    print("🤖 GEMM AI KERNEL SELECTOR - BENCHMARK")
    print("=" * 50)

    selector = GEMMAIKernelSelector()

    test_sizes = [256, 512, 1024]

    for size in test_sizes:
        print(f"\n🧪 Testing {size}x{size} matrices...")

        # Crear matrices de prueba
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)

        # Ejecutar
        result, metadata = selector.select_and_run_kernel(a, b)

        print(f"   Kernel: {metadata['kernel_used']}")
        print(f"   Performance: {metadata['gflops_achieved']:.2f} GFLOPS")
        if 'predicted_gflops' in metadata:
            print(f"   Predicho: {metadata['predicted_gflops']:.2f} GFLOPS")
            print(f"   Confianza: {metadata['prediction_confidence']:.3f}")

    # Mostrar estadísticas finales
    stats = selector.get_stats()
    print("\n📊 Estadísticas finales:")
    print(f"   Total selecciones: {stats['total_selections']}")
    print(f"   Selecciones AI: {stats['ai_selections']}")
    print(f"   Selecciones fallback: {stats['fallback_selections']}")
    if 'avg_prediction_error' in stats:
        print(f"   Error promedio predicción: {stats['avg_prediction_error']:.2f} GFLOPS")
if __name__ == "__main__":
    benchmark_ai_selector()