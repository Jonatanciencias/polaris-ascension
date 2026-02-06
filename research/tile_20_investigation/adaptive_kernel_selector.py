"""
Adaptive Kernel Selector - ML-Powered

Usa el modelo entrenado para seleccionar automáticamente el mejor kernel
"""

import pickle
import numpy as np
import pyopencl as cl


class AdaptiveKernelSelector:
    """
    Selector inteligente de kernels GEMM basado en ML
    
    Usa modelo entrenado para predecir performance y seleccionar
    la mejor configuración de kernel para cada tamaño de matriz.
    """
    
    def __init__(self, model_path='neural_predictor_model.pkl'):
        """Load trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        self.feature_cols = model_data['feature_cols']
        self.mae = model_data['mae']
        self.r2 = model_data['r2']
        
        print(f"✅ Loaded {self.model_name} (R²={self.r2:.4f}, MAE={self.mae:.2f})")
    
    def _engineer_features(self, M, N, K, tile_size, threads, vectorized):
        """Engineer features for prediction"""
        features = np.array([[
            M, N, K, tile_size, threads, vectorized,  # Original
            M * N * K,  # total_ops
            M / tile_size,  # tiles_m
            N / tile_size,  # tiles_n
            K / tile_size,  # tiles_k
            M == N,  # is_square
            threads / 256.0  # thread_util
        ]])
        return features
    
    def predict_performance(self, M, N, K, tile_size, threads, vectorized):
        """Predict GFLOPS for a specific configuration"""
        X = self._engineer_features(M, N, K, tile_size, threads, vectorized)
        return self.model.predict(X)[0]
    
    def select_best_kernel(self, M, N, K, available_configs):
        """
        Select best kernel configuration
        
        Args:
            M, N, K: Matrix dimensions
            available_configs: List of dicts with keys:
                - name: str
                - tile_size: int
                - threads: int (local_x * local_y)
                - vectorized: bool
                
        Returns:
            Dict with best config + predicted performance
        """
        predictions = []
        
        for config in available_configs:
            pred = self.predict_performance(
                M, N, K,
                config['tile_size'],
                config['threads'],
                1 if config['vectorized'] else 0
            )
            
            predictions.append({
                'config': config,
                'predicted_gflops': pred
            })
        
        # Select best
        best = max(predictions, key=lambda x: x['predicted_gflops'])
        return best
    
    def get_recommendation(self, M, N, K, verbose=True):
        """
        Get kernel recommendation for given matrix size
        
        Returns kernel name and expected performance
        """
        # Standard configurations available
        configs = [
            {
                'name': 'tile16_baseline',
                'tile_size': 16,
                'threads': 256,  # 16×16
                'vectorized': True,  # float4
                'local_x': 16,
                'local_y': 16
            },
            {
                'name': 'tile20_vectorized',
                'tile_size': 20,
                'threads': 100,  # 10×10
                'vectorized': True,  # float4
                'local_x': 10,
                'local_y': 10
            }
        ]
        
        best = self.select_best_kernel(M, N, K, configs)
        
        if verbose:
            print(f"\n📊 Recommendation for {M}×{N}×{K}:")
            print(f"  Best: {best['config']['name']}")
            print(f"  Expected: {best['predicted_gflops']:.1f} GFLOPS")
            
            # Show alternatives
            print(f"\n  Alternatives:")
            for config in configs:
                pred = self.predict_performance(
                    M, N, K,
                    config['tile_size'],
                    config['threads'],
                    1 if config['vectorized'] else 0
                )
                symbol = "✅" if config['name'] == best['config']['name'] else "  "
                print(f"    {symbol} {config['name']:20s}: {pred:6.1f} GFLOPS")
        
        return best


def demo():
    """Demonstration of adaptive selector"""
    print("=" * 80)
    print("ADAPTIVE KERNEL SELECTOR - DEMO")
    print("=" * 80)
    
    selector = AdaptiveKernelSelector('neural_predictor_model.pkl')
    
    print("\n" + "=" * 80)
    print("KERNEL SELECTION FOR VARIOUS MATRIX SIZES")
    print("=" * 80)
    
    test_sizes = [256, 512, 768, 1024, 1280, 1536, 2048, 3072, 4096]
    
    print("\nSize  | Best Kernel      | Expected GFLOPS | vs tile16 Baseline")
    print("-" * 80)
    
    results = []
    
    for size in test_sizes:
        best = selector.get_recommendation(size, size, size, verbose=False)
        
        # Get baseline prediction
        baseline_pred = selector.predict_performance(size, size, size, 16, 256, 1)
        
        improvement = ((best['predicted_gflops'] - baseline_pred) / baseline_pred) * 100
        
        symbol = "🚀" if improvement > 50 else ("✅" if improvement > 10 else "  ")
        
        print(f"{size:4d}  | {best['config']['name']:16s} | {best['predicted_gflops']:15.1f} | "
              f"{improvement:+6.1f}% {symbol}")
        
        results.append({
            'size': size,
            'kernel': best['config']['name'],
            'gflops': best['predicted_gflops'],
            'improvement': improvement
        })
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    avg_gflops = np.mean([r['gflops'] for r in results])
    avg_improvement = np.mean([r['improvement'] for r in results])
    
    tile20_count = sum(1 for r in results if 'tile20' in r['kernel'])
    
    print(f"\nAverage Performance: {avg_gflops:.1f} GFLOPS")
    print(f"Average Improvement: {avg_improvement:+.1f}% vs baseline")
    print(f"tile20 selected: {tile20_count}/{len(results)} times")
    
    # Best size
    best_size = max(results, key=lambda r: r['gflops'])
    print(f"\nBest Size: {best_size['size']}×{best_size['size']}")
    print(f"  Kernel: {best_size['kernel']}")
    print(f"  Performance: {best_size['gflops']:.1f} GFLOPS")
    
    print("\n" + "=" * 80)
    print("✅ ADAPTIVE SELECTION READY FOR PRODUCTION")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = demo()
