"""
Mathematical Experiments Module

Implements innovative mathematical approaches for AI inference on RX 580:
- Precision experiments (FP32, FP16, INT8)
- Sparse vs dense network comparison
- Quantization sensitivity analysis
- Dynamic precision adaptation

These experiments demonstrate how mathematical optimization enables:
- Medical imaging in resource-limited settings
- Genomic analysis on affordable hardware
- Drug discovery with limited compute budgets
- Protein structure prediction accessibility
"""

from .precision_experiments import PrecisionExperiment, compare_precisions
from .sparse_networks import SparseNetwork, sparse_vs_dense_benchmark
from .quantization_analysis import QuantizationAnalyzer, sensitivity_analysis

__all__ = [
    'PrecisionExperiment',
    'compare_precisions',
    'SparseNetwork',
    'sparse_vs_dense_benchmark',
    'QuantizationAnalyzer',
    'sensitivity_analysis',
]
