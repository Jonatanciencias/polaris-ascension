"""
Comprehensive test suite for adaptive quantization module.

Tests cover:
- All calibration methods (minmax, percentile, KL divergence, MSE)
- Sensitivity analysis with SQNR and Hessian
- Quantization-Aware Training (QAT) with fake quantization
- INT4 packing/unpacking
- Mixed-precision optimization
- Export/import of quantization configurations
- GPU-specific recommendations

Version: 0.5.0-dev
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

# Import quantization module
from src.compute.quantization import (
    AdaptiveQuantizer,
    QuantizationPrecision,
    CalibrationMethod,
    QuantizationConfig,
    LayerQuantizationStats,
    create_quantizer_for_gpu,
    benchmark_calibration_methods,
)


class TestQuantizationPrecision:
    """Test QuantizationPrecision enum and properties."""
    
    def test_precision_bits(self):
        """Verify bit widths for each precision."""
        assert QuantizationPrecision.FP32.bits == 32
        assert QuantizationPrecision.FP16.bits == 16
        assert QuantizationPrecision.INT8.bits == 8
        assert QuantizationPrecision.INT4.bits == 4
    
    def test_compression_ratios(self):
        """Verify compression ratios relative to FP32."""
        assert QuantizationPrecision.FP32.compression_ratio == 1.0
        assert QuantizationPrecision.FP16.compression_ratio == 2.0
        assert QuantizationPrecision.INT8.compression_ratio == 4.0
        assert QuantizationPrecision.INT4.compression_ratio == 8.0
    
    def test_qmin_qmax_ranges(self):
        """Verify quantization ranges."""
        assert QuantizationPrecision.INT8.qmin == -128
        assert QuantizationPrecision.INT8.qmax == 127
        assert QuantizationPrecision.INT4.qmin == -8
        assert QuantizationPrecision.INT4.qmax == 7


class TestAdaptiveQuantizer:
    """Test core AdaptiveQuantizer functionality."""
    
    def test_initialization_polaris(self):
        """Test quantizer initialization for Polaris."""
        quantizer = AdaptiveQuantizer(gpu_family="polaris")
        assert quantizer.gpu_family == "polaris"
        assert quantizer.gpu_config["wavefront_size"] == 64
        assert quantizer.gpu_config["tflops_fp32"] == 6.17
    
    def test_initialization_vega(self):
        """Test quantizer initialization for Vega."""
        quantizer = AdaptiveQuantizer(gpu_family="vega")
        assert quantizer.gpu_config["fp16_acceleration"] is True
        assert quantizer.gpu_config["wavefront_size"] == 64
    
    def test_initialization_navi(self):
        """Test quantizer initialization for Navi."""
        quantizer = AdaptiveQuantizer(gpu_family="navi")
        assert quantizer.gpu_config["wavefront_size"] == 32  # RDNA uses wave32
    
    def test_unknown_gpu_family_fallback(self):
        """Test fallback to Polaris for unknown GPU."""
        quantizer = AdaptiveQuantizer(gpu_family="unknown_gpu")
        assert quantizer.gpu_family == "polaris"


class TestCalibrationMethods:
    """Test different calibration methods."""
    
    @pytest.fixture
    def test_tensor(self):
        """Create test tensor with known distribution."""
        np.random.seed(42)
        # Normal distribution with some outliers
        tensor = np.random.randn(1000).astype(np.float32)
        # Add outliers
        tensor[0] = 100.0
        tensor[1] = -100.0
        return tensor
    
    def test_minmax_calibration(self, test_tensor):
        """Test min-max calibration method."""
        quantizer = AdaptiveQuantizer(
            config=QuantizationConfig(calibration_method=CalibrationMethod.MINMAX)
        )
        
        q_tensor, scale, zp = quantizer.quantize_tensor(test_tensor)
        
        assert q_tensor.dtype == np.int8
        assert isinstance(scale, (float, np.floating))
        assert isinstance(zp, (int, np.integer))
        assert scale > 0
        
        # Verify quantization range
        assert q_tensor.min() >= -128
        assert q_tensor.max() <= 127
    
    def test_percentile_calibration(self, test_tensor):
        """Test percentile-based calibration (outlier robust)."""
        quantizer = AdaptiveQuantizer(
            config=QuantizationConfig(
                calibration_method=CalibrationMethod.PERCENTILE,
                percentile=99.9
            )
        )
        
        q_tensor, scale, zp = quantizer.quantize_tensor(test_tensor)
        
        # Should be more robust than minmax (scale should be smaller)
        quantizer_minmax = AdaptiveQuantizer(
            config=QuantizationConfig(calibration_method=CalibrationMethod.MINMAX)
        )
        _, scale_minmax, _ = quantizer_minmax.quantize_tensor(test_tensor)
        
        # Percentile scale should be < minmax due to outliers
        assert scale < scale_minmax
    
    def test_kl_divergence_calibration(self, test_tensor):
        """Test KL divergence calibration (TensorRT method)."""
        quantizer = AdaptiveQuantizer(
            config=QuantizationConfig(
                calibration_method=CalibrationMethod.KL_DIVERGENCE,
                num_bins=2048
            )
        )
        
        q_tensor, scale, zp = quantizer.quantize_tensor(test_tensor)
        
        assert q_tensor.dtype == np.int8
        assert scale > 0
        
        # Dequantize and check error
        dequantized = quantizer.dequantize_tensor(q_tensor, scale, zp)
        error = np.mean(np.abs(test_tensor - dequantized))
        
        # Error should be reasonable (< 5% of std)
        assert error < np.std(test_tensor) * 0.05
    
    def test_mse_calibration(self, test_tensor):
        """Test MSE minimization calibration."""
        quantizer = AdaptiveQuantizer(
            config=QuantizationConfig(calibration_method=CalibrationMethod.MSE)
        )
        
        q_tensor, scale, zp = quantizer.quantize_tensor(test_tensor)
        
        # Verify reconstruction error is minimized
        dequantized = quantizer.dequantize_tensor(q_tensor, scale, zp)
        mse = np.mean((test_tensor - dequantized) ** 2)
        
        assert mse < 1.0  # Reasonable MSE


class TestSensitivityAnalysis:
    """Test sensitivity analysis and metrics."""
    
    def test_basic_sensitivity_analysis(self):
        """Test basic layer sensitivity analysis."""
        np.random.seed(42)
        weights = np.random.randn(128, 128).astype(np.float32)
        
        quantizer = AdaptiveQuantizer()
        stats = quantizer.analyze_layer_sensitivity(weights, "test_layer")
        
        assert isinstance(stats, LayerQuantizationStats)
        assert stats.layer_name == "test_layer"
        assert stats.scale > 0
        assert 0 <= stats.memory_reduction <= 1
        assert stats.sensitivity_score >= 0
    
    def test_sqnr_calculation(self):
        """Test SQNR (Signal-to-Quantization-Noise Ratio) metric."""
        np.random.seed(42)
        weights = np.random.randn(256, 256).astype(np.float32)
        
        quantizer = AdaptiveQuantizer()
        stats = quantizer.analyze_layer_sensitivity(weights, "test_layer")
        
        # SQNR should be positive (signal > noise for good quantization)
        assert stats.sqnr_db > 0
        # Typical INT8 SQNR is 30-50 dB
        assert 20 < stats.sqnr_db < 60
    
    def test_cosine_similarity(self):
        """Test cosine similarity metric."""
        np.random.seed(42)
        weights = np.random.randn(100, 100).astype(np.float32)
        
        quantizer = AdaptiveQuantizer()
        stats = quantizer.analyze_layer_sensitivity(weights, "test_layer")
        
        # Cosine similarity should be close to 1 (direction preserved)
        assert 0.9 < stats.cosine_similarity <= 1.0
    
    def test_hessian_trace_approximation(self):
        """Test Hessian trace computation."""
        np.random.seed(42)
        weights = np.random.randn(64, 64).astype(np.float32)
        
        quantizer = AdaptiveQuantizer()
        stats = quantizer.analyze_layer_sensitivity(
            weights, "test_layer", compute_hessian=True
        )
        
        assert stats.hessian_trace is not None
        assert stats.hessian_trace > 0
    
    def test_different_calibration_methods_stats(self):
        """Test that different calibration methods produce different stats."""
        np.random.seed(42)
        weights = np.random.randn(128, 128).astype(np.float32)
        
        methods = [
            CalibrationMethod.MINMAX,
            CalibrationMethod.PERCENTILE,
            CalibrationMethod.KL_DIVERGENCE,
        ]
        
        scales = []
        for method in methods:
            quantizer = AdaptiveQuantizer(
                config=QuantizationConfig(calibration_method=method)
            )
            stats = quantizer.analyze_layer_sensitivity(weights, "test")
            scales.append(stats.scale)
        
        # Different methods should produce different scales
        assert len(set(scales)) == len(scales)  # All unique


class TestQuantizationAwareTraining:
    """Test QAT (Quantization-Aware Training) features."""
    
    def test_fake_quantization(self):
        """Test fake quantization operator."""
        np.random.seed(42)
        tensor = np.random.randn(64, 64).astype(np.float32)
        
        config = QuantizationConfig(enable_qat=True)
        quantizer = AdaptiveQuantizer(config=config)
        
        fake_quant = quantizer.fake_quantize(tensor)
        
        # Output should be FP32 but with quantized values
        assert fake_quant.dtype == np.float32
        assert not np.array_equal(tensor, fake_quant)
        
        # Values should be on quantization grid
        _, scale, zp = quantizer.quantize_tensor(tensor)
        dequantized_expected = quantizer.dequantize_tensor(
            *quantizer.quantize_tensor(tensor)
        )
        np.testing.assert_allclose(fake_quant, dequantized_expected, rtol=1e-5)
    
    def test_fake_quantization_preserves_shape(self):
        """Test that fake quantization preserves tensor shape."""
        tensor = np.random.randn(10, 20, 30).astype(np.float32)
        
        config = QuantizationConfig(enable_qat=True)
        quantizer = AdaptiveQuantizer(config=config)
        fake_quant = quantizer.fake_quantize(tensor)
        
        assert fake_quant.shape == tensor.shape


class TestINT4Operations:
    """Test INT4 packing and unpacking."""
    
    def test_int4_packing_unpacking(self):
        """Test INT4 pack/unpack round-trip."""
        # Create INT4 values (-8 to 7)
        values = np.array([7, -8, 3, -5, 0, 4, -2, 6], dtype=np.int8)
        
        quantizer = AdaptiveQuantizer()
        
        # Pack
        packed = quantizer.pack_int4(values)
        assert len(packed) == len(values) // 2
        
        # Unpack
        unpacked = quantizer.unpack_int4(packed, original_shape=values.shape)
        
        # Should recover original values
        np.testing.assert_array_equal(values, unpacked)
    
    def test_int4_packing_with_padding(self):
        """Test INT4 packing with odd number of elements."""
        values = np.array([7, -8, 3], dtype=np.int8)  # Odd length
        
        quantizer = AdaptiveQuantizer()
        packed = quantizer.pack_int4(values)
        
        # Should pad to even length
        assert len(packed) == 2  # (3+1) / 2
        
        unpacked = quantizer.unpack_int4(packed, original_shape=values.shape)
        assert len(unpacked) == len(values)
    
    def test_int4_range_clipping(self):
        """Test that values outside INT4 range are clipped."""
        values = np.array([100, -100, 5], dtype=np.int8)
        
        quantizer = AdaptiveQuantizer()
        packed = quantizer.pack_int4(values)
        unpacked = quantizer.unpack_int4(packed)
        
        # Values should be clipped to [-8, 7]
        assert unpacked[0] == 7   # 100 → 7
        assert unpacked[1] == -8  # -100 → -8
        assert unpacked[2] == 5   # 5 → 5


class TestMixedPrecisionOptimization:
    """Test mixed-precision optimization."""
    
    def test_mixed_precision_assignment(self):
        """Test automatic precision assignment."""
        np.random.seed(42)
        
        # Create layers with different sensitivities
        layers = {
            "conv1": np.random.randn(64, 3, 3, 3).astype(np.float32),  # Less params
            "fc1": np.random.randn(1000, 512).astype(np.float32),      # Many params
            "fc2": np.random.randn(10, 1000).astype(np.float32),       # Output layer
        }
        
        quantizer = AdaptiveQuantizer()
        precision_map = quantizer.optimize_mixed_precision(
            layers, accuracy_threshold=0.01
        )
        
        assert len(precision_map) == 3
        for layer_name, precision in precision_map.items():
            assert isinstance(precision, QuantizationPrecision)
    
    def test_mixed_precision_memory_budget(self):
        """Test mixed-precision with memory constraint."""
        np.random.seed(42)
        
        layers = {
            "layer1": np.random.randn(1000, 1000).astype(np.float32),
            "layer2": np.random.randn(1000, 1000).astype(np.float32),
        }
        
        quantizer = AdaptiveQuantizer(verbose=False)
        
        # Tight memory budget should force lower precision
        precision_map = quantizer.optimize_mixed_precision(
            layers, memory_budget_gb=0.001  # 1 MB
        )
        
        # At least some layers should be INT8 to fit budget
        int8_count = sum(1 for p in precision_map.values() 
                        if p == QuantizationPrecision.INT8)
        assert int8_count > 0


class TestQuantizationReport:
    """Test reporting and export functionality."""
    
    def test_generate_report(self):
        """Test quantization report generation."""
        np.random.seed(42)
        weights = np.random.randn(128, 128).astype(np.float32)
        
        quantizer = AdaptiveQuantizer()
        quantizer.analyze_layer_sensitivity(weights, "test_layer")
        
        report = quantizer.generate_quantization_report()
        
        assert isinstance(report, str)
        assert "test_layer" in report
        assert "Sensitivity" in report or "Sens" in report
        assert "SQNR" in report
    
    def test_export_import_config(self):
        """Test export and import of quantization config."""
        np.random.seed(42)
        weights = np.random.randn(64, 64).astype(np.float32)
        
        quantizer = AdaptiveQuantizer()
        quantizer.analyze_layer_sensitivity(weights, "test_layer")
        
        # Export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            quantizer.export_quantization_config(filepath)
            
            # Verify file exists and is valid JSON
            assert Path(filepath).exists()
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "gpu_family" in data
            assert "layers" in data
            assert "test_layer" in data["layers"]
            
            # Import into new quantizer
            quantizer2 = AdaptiveQuantizer()
            quantizer2.load_quantization_config(filepath)
            
            assert "test_layer" in quantizer2.layer_stats
            assert quantizer2.layer_stats["test_layer"].scale == \
                   quantizer.layer_stats["test_layer"].scale
        finally:
            Path(filepath).unlink()


class TestFactoryFunctions:
    """Test factory functions and utilities."""
    
    def test_create_quantizer_for_polaris(self):
        """Test factory function for Polaris."""
        quantizer = create_quantizer_for_gpu("polaris", aggressive=False)
        assert quantizer.gpu_family == "polaris"
        assert quantizer.config.precision == QuantizationPrecision.INT8
    
    def test_create_quantizer_for_polaris_aggressive(self):
        """Test aggressive quantization for Polaris."""
        quantizer = create_quantizer_for_gpu("polaris", aggressive=True)
        assert quantizer.config.precision == QuantizationPrecision.INT4
    
    def test_create_quantizer_for_vega(self):
        """Test factory function for Vega."""
        quantizer = create_quantizer_for_gpu("vega")
        assert quantizer.config.precision == QuantizationPrecision.FP16
    
    def test_benchmark_calibration_methods(self):
        """Test calibration method benchmarking."""
        np.random.seed(42)
        tensor = np.random.randn(1000).astype(np.float32)
        
        results = benchmark_calibration_methods(tensor)
        
        assert "minmax" in results
        assert "percentile" in results
        assert "kl" in results
        
        for method, metrics in results.items():
            assert "time_ms" in metrics
            assert "error" in metrics
            assert "sqnr_db" in metrics
            assert metrics["time_ms"] > 0
            assert metrics["error"] >= 0


class TestPrecisionSpecificFeatures:
    """Test precision-specific features."""
    
    def test_fp16_quantization(self):
        """Test FP16 quantization."""
        tensor = np.random.randn(100, 100).astype(np.float32)
        
        config = QuantizationConfig(precision=QuantizationPrecision.FP16)
        quantizer = AdaptiveQuantizer(config=config)
        
        q_tensor, scale, zp = quantizer.quantize_tensor(tensor)
        
        assert q_tensor.dtype == np.float16
        assert scale == 1.0  # No scaling for FP16
        assert zp == 0
    
    def test_int8_symmetric_quantization(self):
        """Test symmetric INT8 quantization."""
        tensor = np.random.randn(100, 100).astype(np.float32)
        
        config = QuantizationConfig(
            precision=QuantizationPrecision.INT8,
            symmetric=True
        )
        quantizer = AdaptiveQuantizer(config=config)
        
        _, scale, zp = quantizer.quantize_tensor(tensor)
        
        assert zp == 0  # Symmetric → zero_point = 0
        assert scale > 0
    
    def test_int8_asymmetric_quantization(self):
        """Test asymmetric INT8 quantization."""
        tensor = np.random.randn(100, 100).astype(np.float32) + 10.0  # Shifted
        
        config = QuantizationConfig(
            precision=QuantizationPrecision.INT8,
            symmetric=False
        )
        quantizer = AdaptiveQuantizer(config=config)
        
        _, scale, zp = quantizer.quantize_tensor(tensor)
        
        # Asymmetric can have non-zero zero_point
        assert scale > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_tensor(self):
        """Test quantization of all-zero tensor."""
        tensor = np.zeros((100, 100), dtype=np.float32)
        
        quantizer = AdaptiveQuantizer()
        q_tensor, scale, zp = quantizer.quantize_tensor(tensor)
        
        assert np.all(q_tensor == 0) or np.all(q_tensor == zp)
        # Scale should be small but non-zero (clamped)
        assert scale > 0
    
    def test_constant_tensor(self):
        """Test quantization of constant tensor."""
        tensor = np.ones((50, 50), dtype=np.float32) * 5.0
        
        quantizer = AdaptiveQuantizer()
        stats = quantizer.analyze_layer_sensitivity(tensor, "const_layer")
        
        # Should handle gracefully (no division by zero)
        assert stats.quantization_error >= 0
        assert stats.scale > 0
    
    def test_very_large_values(self):
        """Test quantization with very large values."""
        tensor = np.random.randn(100, 100).astype(np.float32) * 1000
        
        quantizer = AdaptiveQuantizer()
        q_tensor, scale, zp = quantizer.quantize_tensor(tensor)
        
        # Should clip to valid range
        assert q_tensor.min() >= -128
        assert q_tensor.max() <= 127
        assert scale > 0
    
    def test_empty_layer_dict(self):
        """Test mixed-precision with empty layer dict."""
        quantizer = AdaptiveQuantizer()
        precision_map = quantizer.optimize_mixed_precision({})
        
        assert precision_map == {}


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_quantization_workflow(self):
        """Test complete workflow: analyze → quantize → export → import."""
        np.random.seed(42)
        
        # Create model layers
        layers = {
            "conv1": np.random.randn(32, 3, 3, 3).astype(np.float32),
            "conv2": np.random.randn(64, 32, 3, 3).astype(np.float32),
            "fc1": np.random.randn(128, 1024).astype(np.float32),
        }
        
        # Step 1: Analyze sensitivity
        quantizer = AdaptiveQuantizer(gpu_family="polaris")
        for name, weights in layers.items():
            quantizer.analyze_layer_sensitivity(weights, name)
        
        # Step 2: Optimize mixed-precision
        precision_map = quantizer.optimize_mixed_precision(layers)
        assert len(precision_map) == 3
        
        # Step 3: Generate report
        report = quantizer.generate_quantization_report()
        assert len(report) > 100
        
        # Step 4: Export configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            quantizer.export_quantization_config(filepath)
            
            # Step 5: Import into new quantizer
            quantizer2 = AdaptiveQuantizer()
            quantizer2.load_quantization_config(filepath)
            
            assert len(quantizer2.layer_stats) == 3
        finally:
            Path(filepath).unlink()
    
    def test_rx580_specific_workflow(self):
        """Test RX 580 specific quantization workflow."""
        np.random.seed(42)
        
        # Simulate typical CNN layer for RX 580
        weights = np.random.randn(256, 128).astype(np.float32)
        
        # Use factory function
        quantizer = create_quantizer_for_gpu("polaris", aggressive=False)
        
        # Analyze
        stats = quantizer.analyze_layer_sensitivity(weights, "rx580_layer")
        
        # Verify RX 580 characteristics
        assert quantizer.gpu_config["vram_gb"] == 8
        assert quantizer.gpu_config["wavefront_size"] == 64
        assert stats.memory_reduction == 0.75  # INT8 → 75% reduction
    
    def test_qat_workflow(self):
        """Test Quantization-Aware Training workflow."""
        np.random.seed(42)
        weights = np.random.randn(64, 64).astype(np.float32)
        
        # Enable QAT
        config = QuantizationConfig(enable_qat=True)
        quantizer = AdaptiveQuantizer(config=config)
        
        # Simulate forward pass with fake quantization
        fake_quant_weights = quantizer.fake_quantize(weights)
        
        # Verify it's FP32 (for gradient flow)
        assert fake_quant_weights.dtype == np.float32
        
        # Verify it's quantized values
        assert not np.array_equal(weights, fake_quant_weights)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
