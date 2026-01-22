"""
Tests for ONNX Export Utilities

Comprehensive test suite for PyTorch to ONNX export functionality.

Author: AMD GPU Computing Team
Date: January 21, 2026
"""

import pytest
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import tempfile

from src.inference.onnx_export import (
    export_to_onnx,
    optimize_onnx_model,
    validate_onnx_model,
    compare_outputs,
    quantize_onnx_model,
    get_model_info,
    batch_export_models
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
    """Create simple test model"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    return SimpleModel()


@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Test Export
# ============================================================================

class TestExport:
    """Test ONNX export functionality"""
    
    def test_basic_export(self, simple_model, temp_dir):
        """Test basic model export"""
        output_path = temp_dir / "model.onnx"
        
        success = export_to_onnx(
            simple_model,
            output_path,
            input_shape=(1, 3, 32, 32),
            verify=False
        )
        
        assert success
        assert output_path.exists()
    
    def test_export_with_optimization(self, simple_model, temp_dir):
        """Test export with optimization"""
        output_path = temp_dir / "model_opt.onnx"
        
        success = export_to_onnx(
            simple_model,
            output_path,
            input_shape=(1, 3, 32, 32),
            optimize=True,
            verify=False
        )
        
        assert success
        assert output_path.exists()
    
    def test_export_with_verification(self, simple_model, temp_dir):
        """Test export with verification"""
        output_path = temp_dir / "model_verify.onnx"
        
        success = export_to_onnx(
            simple_model,
            output_path,
            input_shape=(1, 3, 32, 32),
            optimize=True,
            verify=True
        )
        
        assert success
    
    def test_export_dynamic_axes(self, simple_model, temp_dir):
        """Test export with dynamic axes"""
        output_path = temp_dir / "model_dynamic.onnx"
        
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        success = export_to_onnx(
            simple_model,
            output_path,
            input_shape=(1, 3, 32, 32),
            dynamic_axes=dynamic_axes,
            verify=False
        )
        
        assert success
    
    def test_export_different_opset(self, simple_model, temp_dir):
        """Test export with different opset versions"""
        for opset in [11, 12, 13]:
            output_path = temp_dir / f"model_opset{opset}.onnx"
            
            success = export_to_onnx(
                simple_model,
                output_path,
                input_shape=(1, 3, 32, 32),
                opset_version=opset,
                verify=False
            )
            
            assert success


# ============================================================================
# Test Optimization
# ============================================================================

class TestOptimization:
    """Test ONNX optimization"""
    
    def test_optimize_model(self, simple_model, temp_dir):
        """Test model optimization"""
        # Export first
        output_path = temp_dir / "model.onnx"
        export_to_onnx(
            simple_model,
            output_path,
            input_shape=(1, 3, 32, 32),
            optimize=False,
            verify=False
        )
        
        # Optimize
        success = optimize_onnx_model(output_path)
        assert success
    
    def test_optimization_reduces_nodes(self, simple_model, temp_dir):
        """Test that optimization reduces node count"""
        # Export without optimization
        path_unopt = temp_dir / "model_unopt.onnx"
        export_to_onnx(
            simple_model,
            path_unopt,
            input_shape=(1, 3, 32, 32),
            optimize=False,
            verify=False
        )
        
        # Get node count
        model_unopt = onnx.load(str(path_unopt))
        nodes_before = len(model_unopt.graph.node)
        
        # Optimize
        optimize_onnx_model(path_unopt)
        
        # Check node count reduced
        model_opt = onnx.load(str(path_unopt))
        nodes_after = len(model_opt.graph.node)
        
        assert nodes_after <= nodes_before


# ============================================================================
# Test Validation
# ============================================================================

class TestValidation:
    """Test ONNX validation"""
    
    def test_validate_valid_model(self, simple_model, temp_dir):
        """Test validation of valid model"""
        output_path = temp_dir / "model.onnx"
        export_to_onnx(
            simple_model,
            output_path,
            input_shape=(1, 3, 32, 32),
            verify=False
        )
        
        is_valid = validate_onnx_model(output_path)
        assert is_valid
    
    def test_validate_invalid_model(self, temp_dir):
        """Test validation of invalid model"""
        # Create invalid ONNX file
        invalid_path = temp_dir / "invalid.onnx"
        invalid_path.write_text("not an onnx model")
        
        is_valid = validate_onnx_model(invalid_path)
        assert not is_valid


# ============================================================================
# Test Output Comparison
# ============================================================================

class TestComparison:
    """Test output comparison"""
    
    def test_compare_outputs_identical(self, simple_model, temp_dir):
        """Test comparison with identical model"""
        output_path = temp_dir / "model.onnx"
        test_input = torch.randn(1, 3, 32, 32)
        
        export_to_onnx(
            simple_model,
            output_path,
            input_shape=(1, 3, 32, 32),
            verify=False
        )
        
        max_diff = compare_outputs(simple_model, output_path, test_input)
        
        # Should be very small (numerical precision)
        assert max_diff < 1e-4
    
    def test_compare_outputs_different_inputs(self, simple_model, temp_dir):
        """Test comparison with different input sizes"""
        output_path = temp_dir / "model.onnx"
        
        export_to_onnx(
            simple_model,
            output_path,
            input_shape=(1, 3, 32, 32),
            verify=False
        )
        
        # Test with different batch size
        test_input = torch.randn(4, 3, 32, 32)
        max_diff = compare_outputs(simple_model, output_path, test_input)
        
        assert max_diff < 1e-4


# ============================================================================
# Test Quantization
# ============================================================================

class TestQuantization:
    """Test ONNX quantization"""
    
    def test_quantize_model(self, simple_model, temp_dir):
        """Test model quantization"""
        # Export first
        fp32_path = temp_dir / "model_fp32.onnx"
        export_to_onnx(
            simple_model,
            fp32_path,
            input_shape=(1, 3, 32, 32),
            verify=False
        )
        
        # Quantize
        int8_path = temp_dir / "model_int8.onnx"
        success = quantize_onnx_model(fp32_path, int8_path)
        
        assert success
        assert int8_path.exists()
    
    def test_quantization_reduces_size(self, simple_model, temp_dir):
        """Test that quantization reduces model size"""
        # Export FP32
        fp32_path = temp_dir / "model_fp32.onnx"
        export_to_onnx(
            simple_model,
            fp32_path,
            input_shape=(1, 3, 32, 32),
            verify=False
        )
        
        fp32_size = fp32_path.stat().st_size
        
        # Quantize to INT8
        int8_path = temp_dir / "model_int8.onnx"
        quantize_onnx_model(fp32_path, int8_path)
        
        int8_size = int8_path.stat().st_size
        
        # INT8 should be smaller
        assert int8_size < fp32_size


# ============================================================================
# Test Model Info
# ============================================================================

class TestModelInfo:
    """Test model info extraction"""
    
    def test_get_model_info(self, simple_model, temp_dir):
        """Test getting model information"""
        output_path = temp_dir / "model.onnx"
        export_to_onnx(
            simple_model,
            output_path,
            input_shape=(1, 3, 32, 32),
            verify=False
        )
        
        info = get_model_info(output_path)
        
        assert 'opset_version' in info
        assert 'total_parameters' in info
        assert 'num_nodes' in info
        assert 'operators' in info
        assert 'file_size_mb' in info
        
        assert info['total_parameters'] > 0
        assert info['num_nodes'] > 0
    
    def test_model_info_operators(self, simple_model, temp_dir):
        """Test operator counting in model info"""
        output_path = temp_dir / "model.onnx"
        export_to_onnx(
            simple_model,
            output_path,
            input_shape=(1, 3, 32, 32),
            verify=False
        )
        
        info = get_model_info(output_path)
        operators = info['operators']
        
        # Should have Conv, BatchNormalization, Relu, etc.
        assert isinstance(operators, dict)
        assert len(operators) > 0


# ============================================================================
# Test Batch Export
# ============================================================================

class TestBatchExport:
    """Test batch export functionality"""
    
    def test_batch_export(self, simple_model, temp_dir):
        """Test exporting multiple models"""
        models = {
            'model1': simple_model,
            'model2': simple_model
        }
        
        results = batch_export_models(
            models,
            temp_dir,
            input_shape=(1, 3, 32, 32),
            verify=False
        )
        
        assert len(results) == 2
        assert all(results.values())
        
        # Check files exist
        assert (temp_dir / "model1.onnx").exists()
        assert (temp_dir / "model2.onnx").exists()


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_export_to_nonexistent_dir(self, simple_model, temp_dir):
        """Test export to non-existent directory"""
        output_path = temp_dir / "subdir" / "model.onnx"
        
        # Should create directory automatically
        success = export_to_onnx(
            simple_model,
            output_path,
            input_shape=(1, 3, 32, 32),
            verify=False
        )
        
        assert success
        assert output_path.exists()
    
    def test_export_invalid_input_shape(self, simple_model, temp_dir):
        """Test export with invalid input shape"""
        output_path = temp_dir / "model.onnx"
        
        # Should handle gracefully
        success = export_to_onnx(
            simple_model,
            output_path,
            input_shape=(0, 3, 32, 32),  # Invalid batch size
            verify=False
        )
        
        # May fail, but shouldn't crash
        assert isinstance(success, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
