"""
Tests for advanced model loaders (TFLite, JAX, GGUF)

Session 19 - Phase 1
Tests for TensorFlow Lite, JAX/Flax, and GGUF model loaders
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import struct

# Import loaders
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.model_loaders import (
    TFLiteModelLoader,
    JAXModelLoader,
    GGUFModelLoader,
    create_loader,
    ModelMetadata
)


class TestTFLiteModelLoader:
    """Tests for TensorFlow Lite model loader"""
    
    def test_tflite_loader_initialization(self):
        """Test TFLite loader can be created"""
        loader = TFLiteModelLoader(optimization_level=2)
        assert loader is not None
        assert loader.optimization_level == 2
        assert loader.use_xnnpack == True
        assert loader.num_threads == 4
    
    def test_tflite_loader_providers(self):
        """Test TFLite loader returns available providers"""
        loader = TFLiteModelLoader(use_xnnpack=True)
        providers = loader.get_available_providers()
        assert isinstance(providers, list)
        assert 'CPU' in providers
    
    def test_tflite_loader_missing_file(self):
        """Test TFLite loader handles missing file"""
        loader = TFLiteModelLoader()
        if loader.tf is None:
            pytest.skip("TensorFlow not installed")
        with pytest.raises(FileNotFoundError):
            loader.load('nonexistent_model.tflite')
    
    def test_tflite_loader_without_tensorflow(self, monkeypatch):
        """Test TFLite loader gracefully handles missing TensorFlow"""
        # Simulate TensorFlow not installed
        loader = TFLiteModelLoader()
        if loader.tf is None:
            # TensorFlow not installed, should raise RuntimeError
            with pytest.raises(RuntimeError, match="TensorFlow not available"):
                loader.load('dummy.tflite')
        else:
            pytest.skip("TensorFlow is installed")
    
    def test_tflite_predict_without_load(self):
        """Test TFLite predict fails without loading model"""
        loader = TFLiteModelLoader()
        dummy_input = np.zeros((1, 10), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            loader.predict(dummy_input)
    
    def test_create_loader_tflite(self):
        """Test factory creates TFLite loader"""
        loader = create_loader('model.tflite', optimization_level=1)
        assert isinstance(loader, TFLiteModelLoader)
        assert loader.optimization_level == 1


class TestJAXModelLoader:
    """Tests for JAX/Flax model loader"""
    
    def test_jax_loader_initialization(self):
        """Test JAX loader can be created"""
        loader = JAXModelLoader(optimization_level=2)
        assert loader is not None
        assert loader.optimization_level == 2
        assert loader.preferred_device == 'auto'
    
    def test_jax_loader_providers(self):
        """Test JAX loader returns available providers"""
        loader = JAXModelLoader()
        providers = loader.get_available_providers()
        assert isinstance(providers, list)
        # Should be empty if JAX not installed, or contain CPU/GPU if installed
    
    def test_jax_loader_missing_file(self):
        """Test JAX loader handles missing file"""
        loader = JAXModelLoader()
        if loader.jax is None:
            pytest.skip("JAX not installed")
        with pytest.raises(FileNotFoundError):
            loader.load('nonexistent_model.pkl')
    
    def test_jax_loader_without_jax(self):
        """Test JAX loader gracefully handles missing JAX"""
        loader = JAXModelLoader()
        if loader.jax is None:
            # JAX not installed, should raise RuntimeError
            with pytest.raises(RuntimeError, match="JAX not available"):
                loader.load('dummy.pkl')
        else:
            pytest.skip("JAX is installed")
    
    def test_jax_predict_without_load(self):
        """Test JAX predict fails without loading model"""
        loader = JAXModelLoader()
        dummy_input = np.zeros((1, 10), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            loader.predict(dummy_input)
    
    def test_jax_predict_without_apply_fn(self):
        """Test JAX predict fails without apply function"""
        loader = JAXModelLoader()
        # Manually set params without apply_fn
        loader._params = {'param': np.zeros((10, 5))}
        dummy_input = np.zeros((1, 10), dtype=np.float32)
        with pytest.raises(RuntimeError, match="No apply function"):
            loader.predict(dummy_input)
    
    def test_create_loader_jax(self):
        """Test factory creates JAX loader"""
        loader = create_loader('model.pkl', framework='jax', optimization_level=1)
        assert isinstance(loader, JAXModelLoader)
        assert loader.optimization_level == 1


class TestGGUFModelLoader:
    """Tests for GGUF model loader"""
    
    def test_gguf_loader_initialization(self):
        """Test GGUF loader can be created"""
        loader = GGUFModelLoader(optimization_level=2)
        assert loader is not None
        assert loader.optimization_level == 2
        assert loader.n_threads == 4
        assert loader.use_mmap == True
    
    def test_gguf_loader_providers(self):
        """Test GGUF loader returns available providers"""
        loader = GGUFModelLoader()
        providers = loader.get_available_providers()
        assert isinstance(providers, list)
        assert 'CPU' in providers
        assert 'AVX2' in providers
    
    def test_gguf_loader_missing_file(self):
        """Test GGUF loader handles missing file"""
        loader = GGUFModelLoader()
        with pytest.raises(FileNotFoundError):
            loader.load('nonexistent_model.gguf')
    
    def test_gguf_loader_invalid_file(self):
        """Test GGUF loader handles invalid file"""
        loader = GGUFModelLoader()
        
        # Create a dummy file with wrong magic number
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(b'FAKE')  # Wrong magic
            f.write(struct.pack('<I', 2))  # Version
            dummy_path = Path(f.name)
        
        try:
            with pytest.raises(RuntimeError, match="Failed to load GGUF"):
                loader.load(dummy_path)
        finally:
            dummy_path.unlink()
    
    def test_gguf_loader_valid_header(self):
        """Test GGUF loader can parse valid header"""
        loader = GGUFModelLoader()
        
        # Create a valid GGUF file header
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(b'GGUF')  # Magic
            f.write(struct.pack('<I', 2))  # Version
            f.write(struct.pack('<Q', 100))  # Tensor count
            f.write(struct.pack('<Q', 50))  # Metadata count
            dummy_path = Path(f.name)
        
        try:
            metadata = loader.load(dummy_path)
            assert metadata is not None
            assert metadata.framework == 'gguf'
            assert metadata.extra_info['gguf_version'] == 2
            assert metadata.extra_info['tensor_count'] == 100
            assert metadata.extra_info['metadata_count'] == 50
        finally:
            dummy_path.unlink()
    
    def test_gguf_predict_placeholder(self):
        """Test GGUF predict returns placeholder output"""
        loader = GGUFModelLoader()
        
        # Create minimal GGUF file
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(b'GGUF')
            f.write(struct.pack('<I', 2))
            f.write(struct.pack('<Q', 10))
            f.write(struct.pack('<Q', 5))
            dummy_path = Path(f.name)
        
        try:
            loader.load(dummy_path)
            dummy_input = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
            output = loader.predict(dummy_input)
            assert output is not None
            assert output.shape == (1, 5, 32000)  # Placeholder shape
        finally:
            dummy_path.unlink()
    
    def test_create_loader_gguf(self):
        """Test factory creates GGUF loader"""
        loader = create_loader('model.gguf', optimization_level=1)
        assert isinstance(loader, GGUFModelLoader)
        assert loader.optimization_level == 1


class TestCreateLoaderFactory:
    """Tests for create_loader factory function"""
    
    def test_create_loader_auto_detect_tflite(self):
        """Test auto-detection of TFLite"""
        loader = create_loader('model.tflite')
        assert isinstance(loader, TFLiteModelLoader)
    
    def test_create_loader_auto_detect_gguf(self):
        """Test auto-detection of GGUF"""
        loader = create_loader('model.gguf')
        assert isinstance(loader, GGUFModelLoader)
    
    def test_create_loader_auto_detect_pickle(self):
        """Test auto-detection of JAX from pickle"""
        loader = create_loader('model.pkl')
        assert isinstance(loader, JAXModelLoader)
    
    def test_create_loader_explicit_framework(self):
        """Test explicit framework specification"""
        loader = create_loader('my_model.bin', framework='jax')
        assert isinstance(loader, JAXModelLoader)
    
    def test_create_loader_unknown_extension(self):
        """Test error on unknown extension"""
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            create_loader('model.xyz')
    
    def test_create_loader_unknown_framework(self):
        """Test error on unknown framework"""
        with pytest.raises(ValueError, match="Unsupported framework"):
            create_loader('model.bin', framework='unknown')


class TestIntegration:
    """Integration tests for all loaders"""
    
    def test_all_loaders_implement_base_interface(self):
        """Test all loaders implement BaseModelLoader interface"""
        loaders = [
            TFLiteModelLoader(),
            JAXModelLoader(),
            GGUFModelLoader()
        ]
        
        for loader in loaders:
            # Check required methods exist
            assert hasattr(loader, 'load')
            assert hasattr(loader, 'predict')
            assert hasattr(loader, 'get_available_providers')
            assert hasattr(loader, 'is_loaded')
            
            # Check attributes
            assert hasattr(loader, 'optimization_level')
            assert loader.is_loaded() == False  # No model loaded yet
    
    def test_all_loaders_have_metadata_format(self):
        """Test all loaders return correct metadata format"""
        # Create a valid GGUF file for testing
        with tempfile.NamedTemporaryFile(suffix='.gguf', delete=False) as f:
            f.write(b'GGUF')
            f.write(struct.pack('<I', 2))
            f.write(struct.pack('<Q', 10))
            f.write(struct.pack('<Q', 5))
            dummy_path = Path(f.name)
        
        try:
            loader = GGUFModelLoader()
            metadata = loader.load(dummy_path)
            
            # Check all required fields
            assert hasattr(metadata, 'name')
            assert hasattr(metadata, 'framework')
            assert hasattr(metadata, 'input_names')
            assert hasattr(metadata, 'output_names')
            assert hasattr(metadata, 'input_shapes')
            assert hasattr(metadata, 'output_shapes')
            assert hasattr(metadata, 'input_dtypes')
            assert hasattr(metadata, 'output_dtypes')
            assert hasattr(metadata, 'file_size_mb')
            assert hasattr(metadata, 'estimated_memory_mb')
            assert hasattr(metadata, 'provider')
            assert hasattr(metadata, 'optimization_level')
            assert hasattr(metadata, 'extra_info')
            
            assert isinstance(metadata.extra_info, dict)
        finally:
            dummy_path.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
