"""
Radeon RX 580 Energy-Efficient Computing Framework

A comprehensive framework for energy-efficient deep learning inference
on AMD Polaris architecture GPUs, featuring multi-algorithm optimization
and hardware-based power profiling.
"""

__version__ = "1.0.0"
__author__ = "Jonathan Ciencias"
__description__ = "Energy-efficient deep learning inference framework for AMD Polaris GPUs"

# Import main modules
try:
    from .optimization_engines import *
    from .benchmarking import *
    from .ml_models import *
    from .hardware_abstraction import *
    from .utilities import *
except ImportError:
    # Fallback for development
    pass

__all__ = [
    'optimization_engines',
    'benchmarking',
    'ml_models',
    'hardware_abstraction',
    'utilities'
]
