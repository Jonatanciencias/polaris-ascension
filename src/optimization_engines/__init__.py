# Optimization Engines Package
# Contains all matrix multiplication optimization algorithms

# Import individual modules with error handling
try:
    from .advanced_polaris_opencl_engine import AdvancedPolarisOpenCLEngine
except ImportError:
    AdvancedPolarisOpenCLEngine = None

try:
    from .optimized_opencl_engine import OptimizedOpenCLEngine
except ImportError:
    OptimizedOpenCLEngine = None

try:
    from .low_rank_matrix_approximator import LowRankMatrixApproximator
except ImportError:
    LowRankMatrixApproximator = None

try:
    from .low_rank_matrix_approximator_gcn import GCNOptimizedLowRankApproximator
except ImportError:
    GCNOptimizedLowRankApproximator = None

try:
    from .low_rank_matrix_approximator_gpu import LowRankMatrixApproximatorGPU
except ImportError:
    LowRankMatrixApproximatorGPU = None

try:
    from .coppersmith_winograd_gpu import CoppersmithWinogradGPU
except ImportError:
    CoppersmithWinogradGPU = None

try:
    from .quantum_annealing_optimizer import QuantumAnnealingOptimizer
except ImportError:
    QuantumAnnealingOptimizer = None

__all__ = [
    'AdvancedPolarisOpenCLEngine',
    'OptimizedOpenCLEngine',
    'LowRankMatrixApproximator',
    'GCNOptimizedLowRankApproximator',
    'LowRankMatrixApproximatorGPU',
    'CoppersmithWinogradGPU',
    'QuantumAnnealingOptimizer'
]