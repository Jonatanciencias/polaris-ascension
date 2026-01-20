"""
Tests for advanced quantization features (Session 19 - Phase 2)

Tests:
- INT4 quantization with packing
- Mixed precision quantization
- Dynamic quantization
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compute.quantization import (
    QuantizationPrecision,
    AdaptiveQuantizer,
    MixedPrecisionQuantizer,
    MixedPrecisionConfig,
    DynamicQuantizer,
    create_mixed_precision_quantizer,
    create_dynamic_quantizer,
)


class TestINT4Quantization:
    """Tests for INT4 quantization"""
    
    def test_int4_basic_quantization(self):
        """Test basic INT4 quantization"""
        quantizer = AdaptiveQuantizer()
        weights = np.random.randn(128, 64).astype(np.float32)
        
        q_weights, scale, zp = quantizer.quantize_tensor(
            weights,
            precision=QuantizationPrecision.INT4
        )
        
        assert q_weights.dtype == np.int8
        assert q_weights.min() >= -8
        assert q_weights.max() <= 7
        assert isinstance(scale, (float, np.floating, np.ndarray))
    
    def test_int4_pack_unpack(self):
        """Test INT4 packing and unpacking"""
        quantizer = AdaptiveQuantizer()
        
        # Create INT4 values
        values = np.array([3, -5, 7, 2], dtype=np.int8)
        
        # Pack
        packed = quantizer.pack_int4(values)
        assert packed.shape[0] == 2  # 4 values -> 2 bytes
        
        # Unpack
        unpacked = quantizer.unpack_int4(packed)
        np.testing.assert_array_equal(unpacked[:4], values)
    
    def test_int4_memory_reduction(self):
        """Test INT4 achieves expected memory reduction"""
        quantizer = AdaptiveQuantizer()
        weights = np.random.randn(1024, 512).astype(np.float32)
        
        q_weights, _, _ = quantizer.quantize_tensor(
            weights,
            precision=QuantizationPrecision.INT4
        )
        
        # Pack INT4 values
        packed = quantizer.pack_int4(q_weights.flatten())
        
        original_bytes = weights.nbytes
        packed_bytes = packed.nbytes
        
        # INT4 should be ~8x smaller (32 bits -> 4 bits)
        reduction_ratio = original_bytes / packed_bytes
        assert reduction_ratio > 7.0  # Allow some overhead
    
    def test_int4_quantization_accuracy(self):
        """Test INT4 quantization maintains reasonable accuracy"""
        quantizer = AdaptiveQuantizer()
        weights = np.random.randn(256, 128).astype(np.float32) * 0.1
        
        q_weights, scale, zp = quantizer.quantize_tensor(
            weights,
            precision=QuantizationPrecision.INT4
        )
        
        # Dequantize
        dequant = quantizer.dequantize_tensor(q_weights, scale, zp)
        
        # Check error is reasonable
        error = np.mean(np.abs(weights - dequant))
        assert error < 0.05  # Less than 5% of typical weight magnitude


class TestMixedPrecisionQuantization:
    """Tests for mixed precision quantization"""
    
    def test_mixed_precision_initialization(self):
        """Test mixed precision quantizer initialization"""
        quantizer = MixedPrecisionQuantizer()
        assert quantizer is not None
        assert quantizer.config.strategy == 'sensitivity'
    
    def test_sensitivity_computation(self):
        """Test layer sensitivity computation"""
        quantizer = MixedPrecisionQuantizer()
        
        # Small weights -> low sensitivity
        small_weights = np.random.randn(64, 32).astype(np.float32) * 0.01
        small_sens = quantizer.compute_layer_sensitivity(small_weights)
        
        # Large weights -> high sensitivity
        large_weights = np.random.randn(64, 32).astype(np.float32) * 10.0
        large_sens = quantizer.compute_layer_sensitivity(large_weights)
        
        assert large_sens > small_sens
    
    def test_precision_assignment_by_sensitivity(self):
        """Test automatic precision assignment based on sensitivity"""
        quantizer = MixedPrecisionQuantizer()
        
        layer_weights = {
            'conv1': np.random.randn(64, 3, 3, 3).astype(np.float32) * 0.01,  # Low sensitivity
            'conv2': np.random.randn(128, 64, 3, 3).astype(np.float32) * 0.1,  # Medium
            'fc': np.random.randn(1000, 512).astype(np.float32) * 1.0,  # High sensitivity
        }
        
        precisions = quantizer.assign_precisions_by_sensitivity(layer_weights)
        
        assert 'conv1' in precisions
        assert 'conv2' in precisions
        assert 'fc' in precisions
        
        # FC layer (high sensitivity) should get higher precision
        assert precisions['fc'].bits >= precisions['conv1'].bits
    
    def test_mixed_precision_quantize_model(self):
        """Test quantizing entire model with mixed precision"""
        quantizer = MixedPrecisionQuantizer()
        
        layer_weights = {
            'layer1': np.random.randn(128, 64).astype(np.float32),
            'layer2': np.random.randn(64, 32).astype(np.float32),
            'layer3': np.random.randn(32, 10).astype(np.float32),
        }
        
        quantized = quantizer.quantize_model(layer_weights)
        
        assert len(quantized) == 3
        assert 'layer1' in quantized
        assert 'layer2' in quantized
        assert 'layer3' in quantized
        
        # Each layer should have (q_weights, scale, zp)
        for layer_name, (q_weights, scale, zp) in quantized.items():
            assert isinstance(q_weights, np.ndarray)
            assert q_weights.dtype in [np.int8, np.uint8, np.float16]
    
    def test_memory_footprint_calculation(self):
        """Test memory footprint calculation"""
        quantizer = MixedPrecisionQuantizer()
        
        layer_weights = {
            'layer1': np.random.randn(256, 128).astype(np.float32),
            'layer2': np.random.randn(128, 64).astype(np.float32),
        }
        
        # Assign precisions first
        quantizer.assign_precisions_by_sensitivity(layer_weights)
        
        footprint = quantizer.get_memory_footprint(layer_weights)
        
        assert 'original_mb' in footprint
        assert 'quantized_mb' in footprint
        assert 'reduction_percent' in footprint
        
        assert footprint['quantized_mb'] < footprint['original_mb']
        assert footprint['reduction_percent'] > 0
    
    def test_manual_precision_assignment(self):
        """Test manual precision assignment"""
        config = MixedPrecisionConfig(
            strategy='manual',
            layer_precision_map={
                'layer1': QuantizationPrecision.INT4,
                'layer2': QuantizationPrecision.INT8,
                'layer3': QuantizationPrecision.FP16,
            }
        )
        quantizer = MixedPrecisionQuantizer(config=config)
        
        layer_weights = {
            'layer1': np.random.randn(64, 32).astype(np.float32),
            'layer2': np.random.randn(32, 16).astype(np.float32),
            'layer3': np.random.randn(16, 10).astype(np.float32),
        }
        
        quantized = quantizer.quantize_model(layer_weights)
        
        # Check correct precisions were used
        assert quantized['layer1'][0].dtype == np.int8  # INT4 stored in int8
        assert quantized['layer2'][0].dtype in [np.int8, np.uint8]  # INT8
        assert quantized['layer3'][0].dtype == np.float16  # FP16
    
    def test_create_mixed_precision_quantizer_factory(self):
        """Test factory function for mixed precision quantizer"""
        quantizer = create_mixed_precision_quantizer(
            memory_budget_mb=2048.0,
            sensitivity_threshold=0.6
        )
        
        assert quantizer is not None
        assert quantizer.config.memory_budget_mb == 2048.0
        assert quantizer.config.sensitivity_threshold == 0.6


class TestDynamicQuantization:
    """Tests for dynamic quantization"""
    
    def test_dynamic_quantizer_initialization(self):
        """Test dynamic quantizer initialization"""
        quantizer = DynamicQuantizer()
        assert quantizer is not None
        assert quantizer.precision == QuantizationPrecision.INT8
        assert quantizer.cache_scales == True
    
    def test_dynamic_weight_quantization(self):
        """Test dynamic weight quantization"""
        quantizer = DynamicQuantizer()
        weights = np.random.randn(128, 64).astype(np.float32)
        
        q_weights, scale, zp = quantizer.quantize_weights_dynamic(weights)
        
        assert q_weights.dtype == np.int8
        assert isinstance(scale, (float, np.floating))
        assert isinstance(zp, (int, np.integer))
    
    def test_dynamic_weight_caching(self):
        """Test weight scale caching"""
        quantizer = DynamicQuantizer(cache_scales=True)
        weights = np.random.randn(128, 64).astype(np.float32)
        
        # First call - should compute scales
        q1, s1, zp1 = quantizer.quantize_weights_dynamic(weights, weight_id='w1')
        
        # Second call - should use cached scales
        q2, s2, zp2 = quantizer.quantize_weights_dynamic(weights, weight_id='w1')
        
        # Scales should be identical (cached)
        assert s1 == s2
        assert zp1 == zp2
    
    def test_dynamic_activation_quantization(self):
        """Test dynamic activation quantization"""
        quantizer = DynamicQuantizer()
        activations = np.random.randn(32, 128).astype(np.float32)
        
        q_acts, scale, zp = quantizer.quantize_activations_dynamic(activations)
        
        assert q_acts.dtype == np.int8
        assert q_acts.shape == activations.shape
    
    def test_quantized_matmul_simulation(self):
        """Test INT8 matrix multiplication simulation"""
        quantizer = DynamicQuantizer()
        
        # weights: (out_features, in_features), activations: (batch, in_features)
        weights = np.random.randn(32, 64).astype(np.float32)  # 32 output, 64 input
        activations = np.random.randn(16, 64).astype(np.float32)  # 16 batch, 64 input
        
        # Reference FP32 result: (16, 64) @ (64, 32) = (16, 32)
        result_fp32 = np.matmul(activations, weights.T)
        
        # Quantized result
        result_q = quantizer.simulate_quantized_matmul(
            weights, activations, weight_id='test'
        )
        
        # Check shapes match
        assert result_q.shape == result_fp32.shape
        
        # Check results are close (some error expected from quantization)
        error = np.mean(np.abs(result_fp32 - result_q))
        relative_error = error / (np.mean(np.abs(result_fp32)) + 1e-8)
        assert relative_error < 0.1  # Less than 10% relative error
    
    def test_cache_clearing(self):
        """Test cache clearing"""
        quantizer = DynamicQuantizer(cache_scales=True)
        weights = np.random.randn(64, 32).astype(np.float32)
        
        # Add to cache
        quantizer.quantize_weights_dynamic(weights, weight_id='w1')
        assert len(quantizer._scale_cache) == 1
        
        # Clear cache
        quantizer.clear_cache()
        assert len(quantizer._scale_cache) == 0
    
    def test_dynamic_int4_quantization(self):
        """Test dynamic quantization with INT4"""
        quantizer = DynamicQuantizer(precision=QuantizationPrecision.INT4)
        weights = np.random.randn(128, 64).astype(np.float32) * 0.1
        
        q_weights, scale, zp = quantizer.quantize_weights_dynamic(weights)
        
        assert q_weights.min() >= -8
        assert q_weights.max() <= 7
    
    def test_create_dynamic_quantizer_factory(self):
        """Test factory function for dynamic quantizer"""
        quantizer = create_dynamic_quantizer(
            precision=QuantizationPrecision.INT4,
            cache_weights=False
        )
        
        assert quantizer is not None
        assert quantizer.precision == QuantizationPrecision.INT4
        assert quantizer.cache_scales == False


class TestIntegration:
    """Integration tests for advanced quantization"""
    
    def test_int4_vs_int8_memory_comparison(self):
        """Compare memory usage of INT4 vs INT8"""
        weights = np.random.randn(1024, 512).astype(np.float32)
        
        quantizer = AdaptiveQuantizer()
        
        # INT8
        q8, _, _ = quantizer.quantize_tensor(weights, precision=QuantizationPrecision.INT8)
        
        # INT4
        q4, _, _ = quantizer.quantize_tensor(weights, precision=QuantizationPrecision.INT4)
        packed4 = quantizer.pack_int4(q4.flatten())
        
        # INT4 should use ~half the memory of INT8
        ratio = q8.nbytes / packed4.nbytes
        assert ratio > 1.9  # Should be close to 2x
    
    def test_mixed_precision_with_dynamic_quantization(self):
        """Test using mixed precision strategy with dynamic quantization"""
        # Create a simple model
        model_weights = {
            'input': np.random.randn(128, 32).astype(np.float32),
            'hidden': np.random.randn(64, 128).astype(np.float32) * 10,  # High sensitivity
            'output': np.random.randn(10, 64).astype(np.float32),
        }
        
        # Use mixed precision to assign precisions
        mp_quantizer = MixedPrecisionQuantizer()
        precisions = mp_quantizer.assign_precisions_by_sensitivity(model_weights)
        
        # Dynamic quantization with assigned precisions
        dynamic_quantizer = DynamicQuantizer()
        
        for layer_name, weights in model_weights.items():
            precision = precisions[layer_name]
            dynamic_quantizer.precision = precision
            
            q_weights, scale, zp = dynamic_quantizer.quantize_weights_dynamic(
                weights, weight_id=layer_name
            )
            
            # Verify quantization worked
            assert q_weights is not None
            assert q_weights.shape == weights.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
