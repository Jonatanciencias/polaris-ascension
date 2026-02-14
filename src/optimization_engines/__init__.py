# Optimization Engines Package
# Contains all matrix multiplication optimization algorithms
from importlib import import_module
from typing import Any


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    try:
        module = import_module(f".{module_name}", __name__)
        return getattr(module, symbol_name)
    except (ImportError, AttributeError):
        return None


AdvancedPolarisOpenCLEngine = _load_symbol(
    "advanced_polaris_opencl_engine", "AdvancedPolarisOpenCLEngine"
)
OptimizedOpenCLEngine = _load_symbol("optimized_opencl_engine", "OptimizedOpenCLEngine")
LowRankMatrixApproximator = _load_symbol(
    "low_rank_matrix_approximator", "LowRankMatrixApproximator"
)
GCNOptimizedLowRankApproximator = _load_symbol(
    "low_rank_matrix_approximator_gcn", "GCNOptimizedLowRankApproximator"
)
GPUAcceleratedLowRankApproximator = _load_symbol(
    "low_rank_matrix_approximator_gpu", "GPUAcceleratedLowRankApproximator"
)
CoppersmithWinogradGPU = _load_symbol("coppersmith_winograd_gpu", "CoppersmithWinogradGPU")
QuantumAnnealingMatrixOptimizer = _load_symbol(
    "quantum_annealing_optimizer", "QuantumAnnealingMatrixOptimizer"
)

__all__ = [
    "AdvancedPolarisOpenCLEngine",
    "OptimizedOpenCLEngine",
    "LowRankMatrixApproximator",
    "GCNOptimizedLowRankApproximator",
    "GPUAcceleratedLowRankApproximator",
    "CoppersmithWinogradGPU",
    "QuantumAnnealingMatrixOptimizer",
]
