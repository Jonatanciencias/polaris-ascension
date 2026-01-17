"""
Legacy GPU AI Platform - Compute Layer
======================================

This module provides optimized computational primitives and algorithms
specifically designed for AMD GCN (Graphics Core Next) architecture.

Architecture Design:
-------------------
The compute layer sits between the core hardware abstraction and
the inference engines, providing:

1. Sparse Operations - Memory-efficient sparse tensor operations
2. Quantization - INT8/INT4 quantization optimized for GCN
3. Neural Architecture Search - Hardware-aware NAS for legacy GPUs
4. Hybrid Scheduling - CPU-GPU task distribution

Target Hardware:
---------------
- AMD Polaris (RX 400/500 series) - Primary
- AMD Vega (Vega 56/64) - Secondary
- AMD Navi (RX 5000 series) - Experimental

Example Usage:
-------------
    from src.compute import SparseOperations, AdaptiveQuantizer
    
    # Sparse matrix multiplication optimized for 8GB VRAM
    sparse_ops = SparseOperations(target_density=0.3)
    result = sparse_ops.sparse_matmul(weight_matrix, input_tensor)
    
    # Adaptive quantization for Polaris architecture
    quantizer = AdaptiveQuantizer(gpu_family="polaris")
    quantized_model = quantizer.quantize(model, precision="int8")

Version: 0.5.0-dev
License: MIT
"""

__version__ = "0.5.0-dev"
__all__ = [
    "SparseOperations",
    "AdaptiveQuantizer",
    "HybridScheduler",
    "NeuralArchitectureSearch",
]

# Placeholder imports - will be implemented progressively
# from .sparse import SparseOperations
# from .quantization import AdaptiveQuantizer
# from .scheduler import HybridScheduler
# from .nas import NeuralArchitectureSearch

class ComputeLayerNotReady(Exception):
    """Raised when attempting to use unimplemented compute features."""
    pass


def get_available_algorithms():
    """
    Return list of implemented algorithms in the compute layer.
    
    Returns:
        dict: Algorithm names mapped to their implementation status
    """
    return {
        "sparse_operations": {
            "status": "planned",
            "version": "0.6.0",
            "description": "Sparse tensor operations for GCN wavefront optimization",
        },
        "adaptive_quantization": {
            "status": "planned", 
            "version": "0.6.0",
            "description": "Per-layer INT8/INT4 quantization with accuracy preservation",
        },
        "hybrid_scheduler": {
            "status": "planned",
            "version": "0.7.0",
            "description": "Dynamic CPU-GPU task distribution based on operation type",
        },
        "neural_architecture_search": {
            "status": "planned",
            "version": "0.8.0",
            "description": "Hardware-aware NAS optimized for 8GB VRAM constraints",
        },
    }


def compute_status():
    """Print current status of the compute layer."""
    algorithms = get_available_algorithms()
    print("\n" + "=" * 60)
    print("Legacy GPU AI Platform - Compute Layer Status")
    print("=" * 60)
    
    for name, info in algorithms.items():
        status_icon = "✅" if info["status"] == "implemented" else "⏳"
        print(f"\n{status_icon} {name}")
        print(f"   Status: {info['status']}")
        print(f"   Target Version: {info['version']}")
        print(f"   Description: {info['description']}")
    
    print("\n" + "=" * 60)
