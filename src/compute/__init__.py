"""
Compute Module - Advanced Computing Techniques

Módulo que contiene implementaciones de técnicas avanzadas de computación:
- Neural Architecture Search (NAS/DARTS)
- Tensor Decomposition
- Sparse Computing
- Quantization
- Mixed Precision
- Y más...

Author: AMD GPU Computing Team
Date: February 2026
"""

# NAS/DARTS exports
try:
    from .nas_darts import (
        DARTSConfig,
        SearchSpace,
        SearchResult,
        DARTSNetwork,
        DARTSTrainer,
        search_architecture,
        PRIMITIVES,
    )
    NAS_AVAILABLE = True
except ImportError as e:
    NAS_AVAILABLE = False
    import warnings
    warnings.warn(f"NAS/DARTS module not fully available: {e}")

__all__ = [
    'DARTSConfig',
    'SearchSpace',
    'SearchResult',
    'DARTSNetwork',
    'DARTSTrainer',
    'search_architecture',
    'PRIMITIVES',
    'NAS_AVAILABLE',
]

__version__ = '1.0.0'
