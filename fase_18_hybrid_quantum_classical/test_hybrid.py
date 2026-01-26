#!/usr/bin/env python3
"""
Test script for Hybrid Quantum-Classical Optimizer
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_quantum_classical_optimizer import HybridQuantumClassicalOptimizer, HybridConfig

def test_hybrid_system():
    """Test the hybrid quantum-classical system"""
    print("🚀 Testing Hybrid Quantum-Classical System")

    # Create configuration
    config = HybridConfig(
        quantum_threshold=0.8,
        classical_fallback=True,
        adaptive_switching=True,
        neural_quantum_integration=True,
        energy_optimization=True
    )

    # Initialize optimizer
    optimizer = HybridQuantumClassicalOptimizer(config)

    # Create test matrices
    size = 256
    matrix_a = np.random.rand(size, size).astype(np.float32)
    matrix_b = np.random.rand(size, size).astype(np.float32)

    print(f"📊 Testing with {size}x{size} matrices")

    # Run optimization
    try:
        result = optimizer.optimize(matrix_a, matrix_b)

        # Get metrics
        metrics = optimizer.get_metrics()

        print("✅ Hybrid optimization completed")
        print(f"📈 Quantum contribution: {metrics['fusion_metrics']['quantum_contribution']:.3f}")
        print(f"📈 Classical contribution: {metrics['fusion_metrics']['classical_contribution']:.3f}")
        print(f"📈 Hybrid speedup: {metrics['fusion_metrics']['hybrid_speedup']:.3f}")

        return True

    except Exception as e:
        print(f"❌ Error in hybrid optimization: {e}")
        return False

if __name__ == "__main__":
    success = test_hybrid_system()
    sys.exit(0 if success else 1)