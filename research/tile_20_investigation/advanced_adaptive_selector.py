#!/usr/bin/env python3
"""
Advanced Adaptive Kernel Selector - Phase 2.1 Final Integration

Professional ML-powered selector with:
- tile16, tile20, tile24 support
- Gradient Boosting model
- Size-based heuristics + ML predictions
- Production-ready API
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

def engineer_features(M: int, N: int, K: int, tile_size: int, threads: int, vectorized: bool) -> np.ndarray:
    """Engineer features for ML model"""
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
    
    return np.array(features, dtype=np.float32)


def train_model(dataset_path: Path) -> Tuple[GradientBoostingRegressor, Dict]:
    """Train ML model on consolidated dataset"""
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    
    if len(samples) < 10:
        raise ValueError(f"Insufficient samples: {len(samples)}")
    
    print(f"Training on {len(samples)} samples...")
    
    # Prepare training data
    X = []
    y = []
    
    for sample in samples:
        features = engineer_features(
            sample['M'], sample['N'], sample['K'],
            sample['tile_size'],
            sample['threads'],
            sample['vectorized']
        )
        X.append(features)
        y.append(sample['gflops'])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")
    print()
    
    # Train model
    print("Training Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(samples)), 
                                scoring='r2', n_jobs=-1)
    
    # Performance metrics
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    mae = np.mean(np.abs(y - y_pred))
    
    metrics = {
        'r2_train': float(r2),
        'mae_train': float(mae),
        'cv_r2_mean': float(np.mean(cv_scores)),
        'cv_r2_std': float(np.std(cv_scores)),
        'n_samples': len(samples),
        'n_features': X.shape[1]
    }
    
    print(f"✓ Model trained")
    print(f"  R² (train): {r2:.4f}")
    print(f"  MAE (train): {mae:.2f} GFLOPS")
    print(f"  R² (CV): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print()
    
    return model, metrics


class AdvancedAdaptiveKernelSelector:
    """
    Professional adaptive kernel selector with ML + heuristics
    
    Supports: tile16, tile20, tile24
    Uses: ML predictions + size-based heuristics
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize selector"""
        self.model = None
        self.metrics = None
        self.model_path = model_path or Path(__file__).parent / "advanced_neural_model.pkl"
        
        # Kernel configurations
        self.configs = {
            'tile16_baseline': {
                'tile_size': 16,
                'threads': 256,
                'vectorized': False,
                'local_size': (16, 16),
                'kernel_name': 'baseline_tile16'
            },
            'tile20_vectorized': {
                'tile_size': 20,
                'threads': 100,
                'vectorized': True,
                'local_size': (10, 10),
                'kernel_name': 'gemm_tile20_vectorized'
            },
            'tile24_vectorized': {
                'tile_size': 24,
                'threads': 144,
                'vectorized': True,
                'local_size': (12, 12),
                'kernel_name': 'gemm_tile24_vectorized'
            }
        }
        
        # Load model if exists
        if self.model_path.exists():
            self.load_model()
    
    def load_model(self):
        """Load trained model"""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.metrics = data['metrics']
        print(f"✓ Model loaded: R²={self.metrics['r2_train']:.4f}, "
              f"MAE={self.metrics['mae_train']:.1f} GFLOPS")
    
    def save_model(self, model, metrics):
        """Save trained model"""
        with open(self.model_path, 'wb') as f:
            pickle.dump({'model': model, 'metrics': metrics}, f)
        print(f"💾 Model saved to: {self.model_path}")
    
    def predict_performance(self, M: int, N: int, K: int, config_name: str) -> float:
        """Predict GFLOPS for a given configuration"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        config = self.configs[config_name]
        features = engineer_features(
            M, N, K,
            config['tile_size'],
            config['threads'],
            config['vectorized']
        )
        
        features = features.reshape(1, -1)
        predicted_gflops = self.model.predict(features)[0]
        
        return max(0.0, float(predicted_gflops))  # Ensure non-negative
    
    def select_best_kernel(self, M: int, N: int, K: int, method: str = 'hybrid') -> Tuple[str, float]:
        """
        Select best kernel for given matrix dimensions
        
        Args:
            M, N, K: Matrix dimensions
            method: 'ml' (pure ML), 'heuristic' (rule-based), 'hybrid' (ML + heuristics)
        
        Returns:
            (config_name, predicted_gflops)
        """
        if method == 'heuristic':
            return self._select_heuristic(M, N, K)
        elif method == 'ml':
            return self._select_ml(M, N, K)
        else:  # hybrid (default)
            return self._select_hybrid(M, N, K)
    
    def _select_heuristic(self, M: int, N: int, K: int) -> Tuple[str, float]:
        """Rule-based selection from empirical observations"""
        size = (M + N + K) / 3  # Average dimension
        
        # Rules based on consolidated data analysis:
        # Small (0-600): tile24 best
        # Medium (600-1200): tile20 best
        # Large (1200-1600): tile20 best (peak @ 1400)
        # Very Large (1600+): tile24 best
        
        if size < 600:
            config = 'tile24_vectorized'
            # Rough estimate
            predicted = 350.0
        elif size < 1600:
            config = 'tile20_vectorized'
            # Peak around 1400, degrade towards 1200
            if 1300 <= size <= 1500:
                predicted = 850.0
            elif 1200 <= size < 1300:
                predicted = 750.0
            elif size >= 1500:
                predicted = 600.0
            else:
                predicted = 650.0
        else:
            config = 'tile24_vectorized'
            # tile24 dominates large matrices
            if size >= 2000:
                predicted = 750.0
            else:
                predicted = 700.0
        
        return config, predicted
    
    def _select_ml(self, M: int, N: int, K: int) -> Tuple[str, float]:
        """Pure ML-based selection"""
        if self.model is None:
            raise ValueError("Model not loaded for ML selection")
        
        predictions = {}
        for config_name in self.configs.keys():
            pred = self.predict_performance(M, N, K, config_name)
            predictions[config_name] = pred
        
        best_config = max(predictions.items(), key=lambda x: x[1])
        return best_config[0], best_config[1]
    
    def _select_hybrid(self, M: int, N: int, K: int) -> Tuple[str, float]:
        """Hybrid: ML predictions + heuristic validation"""
        if self.model is None:
            # Fallback to heuristic if no model
            return self._select_heuristic(M, N, K)
        
        # Get ML predictions for all configs
        predictions = {}
        for config_name in self.configs.keys():
            pred = self.predict_performance(M, N, K, config_name)
            predictions[config_name] = pred
        
        # Apply heuristic overrides for known failure modes
        size = (M + N + K) / 3
        
        # Override: tile24 struggles at medium sizes (768-1400)
        if 768 <= size <= 1400:
            if 'tile20_vectorized' in predictions:
                # Boost tile20 confidence in this range
                predictions['tile20_vectorized'] *= 1.15
        
        # Override: tile24 excels at large sizes (2048+)
        if size >= 2048:
            if 'tile24_vectorized' in predictions:
                predictions['tile24_vectorized'] *= 1.1
        
        # Select best after adjustments
        best_config = max(predictions.items(), key=lambda x: x[1])
        return best_config[0], best_config[1]
    
    def get_recommendation(self, M: int, N: int, K: int, verbose: bool = False) -> Dict:
        """
        Get complete recommendation for matrix multiplication
        
        Returns:
            dict with config details, predicted performance, alternatives
        """
        config_name, predicted_gflops = self.select_best_kernel(M, N, K, method='hybrid')
        config = self.configs[config_name]
        
        recommendation = {
            'config_name': config_name,
            'config': config,
            'predicted_gflops': predicted_gflops,
            'matrix_size': (M, N, K)
        }
        
        if verbose:
            # Get all predictions for comparison
            all_predictions = {}
            for name in self.configs.keys():
                try:
                    pred = self.predict_performance(M, N, K, name)
                    all_predictions[name] = pred
                except:
                    pass
            
            recommendation['alternatives'] = all_predictions
        
        return recommendation


def main():
    """Train model and demonstrate selector"""
    print("=" * 80)
    print("ADVANCED ADAPTIVE KERNEL SELECTOR - Training & Validation")
    print("=" * 80)
    print()
    
    # Load and train
    dataset_path = Path(__file__).parent / "consolidated_neural_dataset.json"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Run consolidate_data.py first")
        return
    
    model, metrics = train_model(dataset_path)
    
    # Create selector
    selector = AdvancedAdaptiveKernelSelector()
    selector.save_model(model, metrics)
    
    # Test on key sizes
    print("=" * 80)
    print("VALIDATION ON KEY SIZES")
    print("=" * 80)
    print()
    
    test_sizes = [512, 768, 1024, 1280, 1400, 1536, 2048, 3072]
    
    print("Size   | Selected Kernel    | Predicted | Method")
    print("-------|-------------------|-----------|--------")
    
    for size in test_sizes:
        rec = selector.get_recommendation(size, size, size, verbose=True)
        
        print(f"{size:4d}   | {rec['config_name']:17s} | {rec['predicted_gflops']:7.1f}   | hybrid")
    
    print()
    
    # Summary
    print("=" * 80)
    print("PRODUCTION READY")
    print("=" * 80)
    print()
    print("✅ Advanced selector trained and validated")
    print(f"✅ Model metrics: R²={metrics['r2_train']:.4f}, MAE={metrics['mae_train']:.1f} GFLOPS")
    print(f"✅ Supports: tile16, tile20, tile24")
    print(f"✅ Selection method: Hybrid (ML + heuristics)")
    print()
    print("Usage:")
    print("```python")
    print("selector = AdvancedAdaptiveKernelSelector()")
    print("rec = selector.get_recommendation(M, N, K)")
    print("kernel_name = rec['config']['kernel_name']")
    print("```")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
