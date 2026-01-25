"""
🤖 FASE 8: BAYESIAN OPTIMIZATION FOR KERNEL TUNING

Paquete para optimización bayesiana de parámetros de kernels GEMM.
Utiliza Gaussian Processes para exploración eficiente del espacio de parámetros.

Módulos principales:
- bayesian_optimizer: Implementación principal del optimizador
- KernelParameterSpace: Definición del espacio de parámetros
- OptimizationResult: Estructura de resultados

Autor: AI Assistant
Fecha: 2026-01-25
"""

from .src.bayesian_optimizer import (
    BayesianKernelOptimizer,
    KernelParameterSpace,
    OptimizationResult
)

__version__ = "0.1.0"
__author__ = "AI Assistant"
__description__ = "Bayesian Optimization for GEMM Kernel Tuning"

__all__ = [
    "BayesianKernelOptimizer",
    "KernelParameterSpace",
    "OptimizationResult"
]