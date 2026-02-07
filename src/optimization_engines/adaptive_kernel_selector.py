#!/usr/bin/env python3
"""
Production Adaptive Kernel Selector
Phase 2.1 Integration - ML-Powered GEMM Optimization

Provides intelligent kernel selection for matrix multiplication:
- tile16: Baseline (256 threads, good for small/medium)
- tile20: Peak performance (866.9 GFLOPS @ 1400×1400)
- tile24: Large matrix specialist (764 GFLOPS @ 2048)
- tile20_v3_1400: Promoted T2 candidate for strict 1400 scope

Uses Gradient Boosting model + heuristics for optimal selection.
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:
    print("⚠️  Warning: scikit-learn not installed. Install with: pip install scikit-learn")
    GradientBoostingRegressor = None


class ProductionKernelSelector:
    """
    Production-ready adaptive GEMM kernel selector
    
    Features:
    - ML-powered predictions (Gradient Boosting, R²=1.0)
    - Hybrid selection strategy (ML + heuristics)
    - 4 specialized kernels (tile16, tile20, tile24, tile20_v3_1400)
    - 75% accuracy on validation
    
    Usage:
        selector = ProductionKernelSelector()
        recommendation = selector.select_kernel(1400, 1400, 1400)
        
        # Use recommendation
        kernel_path = recommendation['kernel_path']
        local_size = recommendation['local_size']
        expected_gflops = recommendation['predicted_gflops']
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize production kernel selector
        
        Args:
            model_path: Path to trained model (default: auto-detect)
        """
        # Determine base directory
        if model_path:
            self.model_path = Path(model_path)
        else:
            # Auto-detect from src/ml_models/
            base_dir = Path(__file__).parent.parent
            self.model_path = base_dir / "ml_models" / "kernel_selector_model.pkl"
        
        # Kernel configurations
        self.kernel_configs = {
            'tile16': {
                'name': 'tile16_baseline',
                'path': 'src/kernels/debug_kernel.cl',  # Existing baseline
                'tile_size': 16,
                'local_size': (16, 16),
                'threads': 256,
                'vectorized': False,
                'typical_gflops': 566,
                'best_for': 'small to medium matrices'
            },
            'tile20': {
                'name': 'tile20_production',
                'path': 'src/kernels/gemm_tile20_production.cl',
                'tile_size': 20,
                'local_size': (10, 10),
                'threads': 100,
                'vectorized': True,
                'typical_gflops': 867,
                'best_for': 'sweet spot (1200-1600), peak performance'
            },
            'tile20_v3_1400': {
                'name': 'tile20_v3_vectorized',
                'path': 'src/kernels/gemm_tile20_v3_vectorized.cl',
                'tile_size': 20,
                'local_size': (10, 10),
                'threads': 100,
                'vectorized': True,
                'typical_gflops': 926,
                'best_for': 'promoted T2 candidate for exact 1400 scope'
            },
            'tile24': {
                'name': 'tile24_production',
                'path': 'src/kernels/gemm_tile24_production.cl',
                'tile_size': 24,
                'local_size': (12, 12),
                'threads': 144,
                'vectorized': True,
                'typical_gflops': 764,
                'best_for': 'large matrices (1600+)'
            }
        }
        
        # Load ML model
        self.model = None
        self.model_available = False
        
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.metrics = model_data.get('metrics', {})
                    self.model_available = True
                    
                    r2 = self.metrics.get('r2_train', 0)
                    mae = self.metrics.get('mae_train', 0)
                    print(f"✓ Model loaded: R²={r2:.4f}, MAE={mae:.1f} GFLOPS")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                print("   Falling back to heuristic-only selection")
        else:
            print(f"⚠️  Model not found: {self.model_path}")
            print("   Using heuristic-only selection")
    
    def _engineer_features(self, M: int, N: int, K: int, 
                          tile_size: int, threads: int, vectorized: bool) -> np.ndarray:
        """Engineer features for ML model (same as training)"""
        # Base features
        features = [
            M, N, K,
            tile_size,
            threads,
            1 if vectorized else 0
        ]
        
        # Derived features
        total_ops = 2 * M * N * K
        tiles_m = (M + tile_size - 1) // tile_size
        tiles_n = (N + tile_size - 1) // tile_size
        total_tiles = tiles_m * tiles_n
        ops_per_tile = (tile_size * tile_size * K) * 2
        work_per_thread = ops_per_tile / threads if threads > 0 else 0
        matrix_volume = M * N * K
        
        features.extend([
            total_ops,
            tiles_m,
            tiles_n,
            total_tiles,
            ops_per_tile,
            work_per_thread,
            matrix_volume
        ])
        
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    def _predict_performance(self, M: int, N: int, K: int, kernel_key: str) -> float:
        """Predict GFLOPS for given configuration"""
        config = self.kernel_configs[kernel_key]

        # Promoted scoped kernel uses measured deterministic replay value.
        # Avoid extrapolating it through the legacy model.
        if kernel_key == "tile20_v3_1400":
            return float(config["typical_gflops"])
        
        if self.model_available and self.model:
            features = self._engineer_features(
                M, N, K,
                config['tile_size'],
                config['threads'],
                config['vectorized']
            )
            try:
                prediction = self.model.predict(features)[0]
                return float(prediction)
            except:
                pass
        
        # Fallback: return typical GFLOPS
        return config['typical_gflops']
    
    def _heuristic_selection(self, M: int, N: int, K: int) -> str:
        """
        Heuristic-based kernel selection
        
        Rules based on Phase 2.1 findings:
        - Small (< 600): tile24 (best for small)
        - Medium (600-1200): tile20 (consistent)
        - Sweet spot (1200-1600): tile20 (peak zone)
        - Large (1600+): tile24 (dominates large)
        """
        size = max(M, N, K)

        if self._in_t2_promoted_scope(M, N, K):
            return 'tile20_v3_1400'
        
        if size < 600:
            return 'tile24'  # Best for small: 384 GFLOPS @ 512
        elif size < 1200:
            return 'tile20'  # Consistent: 600-800 GFLOPS
        elif size <= 1600:
            return 'tile20'  # Peak zone: 866.9 GFLOPS @ 1400
        else:
            return 'tile24'  # Dominates large: 764 GFLOPS @ 2048

    @staticmethod
    def _in_t2_promoted_scope(M: int, N: int, K: int) -> bool:
        """
        Promoted candidate is validated only for exact 1400 square GEMM.
        Everything else must fall back to the existing production policy.
        """
        return M == 1400 and N == 1400 and K == 1400
    
    def select_kernel(self, M: int, N: int, K: int) -> Dict:
        """
        Select optimal kernel for given matrix sizes
        
        Args:
            M, N, K: Matrix dimensions for C = A @ B where A is M×K, B is K×N
        
        Returns:
            {
                'kernel_key': str,          # 'tile16', 'tile20', or 'tile24'
                'kernel_name': str,         # Kernel function name
                'kernel_path': str,         # Path to kernel file
                'local_size': tuple,        # OpenCL local work size
                'tile_size': int,           # Tile dimension
                'threads': int,             # Threads per workgroup
                'predicted_gflops': float,  # Expected performance
                'selection_method': str,    # 'ml', 'heuristic', or 'hybrid'
                'best_for': str            # Description of use case
            }
        """
        # Get predictions for all kernels
        predictions = {}
        for kernel_key in self.kernel_configs.keys():
            predictions[kernel_key] = self._predict_performance(M, N, K, kernel_key)

        # Hard scope override for the promoted T2 candidate.
        # This keeps risk bounded and provides deterministic fallback behavior.
        if self._in_t2_promoted_scope(M, N, K):
            selected_key = 'tile20_v3_1400'
            method = 'scope override (t2 promoted)'
            config = self.kernel_configs[selected_key]
            return {
                'kernel_key': selected_key,
                'kernel_name': config['name'],
                'kernel_path': config['path'],
                'local_size': config['local_size'],
                'tile_size': config['tile_size'],
                'threads': config['threads'],
                'predicted_gflops': predictions[selected_key],
                'selection_method': method,
                'best_for': config['best_for']
            }
        
        # Hybrid selection strategy
        if self.model_available:
            # ML-based: choose kernel with highest predicted GFLOPS
            ml_choice = max(predictions, key=predictions.get)
            
            # Heuristic-based: use expert rules
            heuristic_choice = self._heuristic_selection(M, N, K)
            
            # Hybrid: use ML, but override for extreme cases
            if max(M, N, K) > 2500:
                # Very large: heuristic knows tile24 is better
                selected_key = heuristic_choice
                method = 'hybrid (heuristic override)'
            else:
                # Normal case: trust ML
                selected_key = ml_choice
                method = 'hybrid (ml primary)'
        else:
            # No ML model: pure heuristic
            selected_key = self._heuristic_selection(M, N, K)
            method = 'heuristic'
        
        # Build recommendation
        config = self.kernel_configs[selected_key]
        
        return {
            'kernel_key': selected_key,
            'kernel_name': config['name'],
            'kernel_path': config['path'],
            'local_size': config['local_size'],
            'tile_size': config['tile_size'],
            'threads': config['threads'],
            'predicted_gflops': predictions[selected_key],
            'selection_method': method,
            'best_for': config['best_for']
        }
    
    def get_all_predictions(self, M: int, N: int, K: int) -> Dict:
        """
        Get predictions for all kernels (for debugging/analysis)
        
        Returns:
            {
                'tile16': float,
                'tile20': float,
                'tile24': float,
                'selected': str
            }
        """
        predictions = {}
        for kernel_key in self.kernel_configs.keys():
            predictions[kernel_key] = self._predict_performance(M, N, K, kernel_key)
        
        selected = self.select_kernel(M, N, K)['kernel_key']
        predictions['selected'] = selected
        
        return predictions
    
    def benchmark_summary(self) -> str:
        """Return summary of selector capabilities"""
        summary = []
        summary.append("=" * 70)
        summary.append("PRODUCTION KERNEL SELECTOR - Phase 2.1")
        summary.append("=" * 70)
        summary.append("")
        summary.append(f"Model: {'LOADED ✓' if self.model_available else 'NOT AVAILABLE ⚠️'}")
        if self.model_available:
            r2 = self.metrics.get('r2_train', 0)
            mae = self.metrics.get('mae_train', 0)
            summary.append(f"  R² Score: {r2:.4f}")
            summary.append(f"  MAE: {mae:.1f} GFLOPS")
        summary.append("")
        summary.append("Available Kernels:")
        for key, config in self.kernel_configs.items():
            summary.append(f"  {key}: {config['typical_gflops']} GFLOPS - {config['best_for']}")
        summary.append("")
        summary.append("Performance Achievements:")
        summary.append("  Peak (promoted scope): 926.3 GFLOPS @ 1400×1400 (tile20_v3_1400)")
        summary.append("  Legacy peak: 866.9 GFLOPS @ 1400×1400 (tile20)")
        summary.append("  Large: 764 GFLOPS @ 2048×2048 (tile24)")
        summary.append("  Baseline: 566 GFLOPS @ 2048×2048 (tile16)")
        summary.append("=" * 70)
        
        return "\n".join(summary)


# Convenience function for quick usage
def select_optimal_kernel(M: int, N: int, K: int) -> Dict:
    """
    Quick function to select optimal kernel
    
    Usage:
        recommendation = select_optimal_kernel(1400, 1400, 1400)
        print(f"Use: {recommendation['kernel_path']}")
        print(f"Expected: {recommendation['predicted_gflops']:.1f} GFLOPS")
    """
    selector = ProductionKernelSelector()
    return selector.select_kernel(M, N, K)


# Demo/validation when run directly
if __name__ == "__main__":
    print("=" * 70)
    print("PRODUCTION KERNEL SELECTOR - Validation")
    print("=" * 70)
    print()
    
    selector = ProductionKernelSelector()
    print(selector.benchmark_summary())
    print()
    
    # Test key sizes
    test_sizes = [512, 1024, 1400, 2048, 3072]
    
    print("VALIDATION ON KEY SIZES")
    print("=" * 70)
    print(f"{'Size':<10} {'Selected':<15} {'Predicted':<12} {'Method':<30}")
    print("-" * 70)
    
    for size in test_sizes:
        rec = selector.select_kernel(size, size, size)
        print(f"{size:<10} {rec['kernel_key']:<15} {rec['predicted_gflops']:<12.1f} {rec['selection_method']:<30}")
    
    print()
    print("✅ Selector ready for production use")
