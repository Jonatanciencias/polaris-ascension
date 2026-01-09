"""Tests for Configuration module"""
import pytest
import tempfile
import os
from utils.config import Config, get_config, set_config, load_config


def test_config_default_values():
    """Test config default values"""
    config = Config()
    
    assert config.gpu_backend == "auto"
    assert config.gpu_device_id == 0
    assert config.max_vram_usage_mb == 7168
    assert config.enable_cpu_offload is True
    assert config.default_batch_size == 1


def test_config_to_dict():
    """Test config to dictionary conversion"""
    config = Config()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert 'gpu_backend' in config_dict
    assert 'max_vram_usage_mb' in config_dict


def test_config_validation():
    """Test config validation"""
    config = Config()
    
    # Valid config should pass
    assert config.validate()
    
    # Invalid backend
    config_invalid = Config(gpu_backend="invalid")
    with pytest.raises(ValueError):
        config_invalid.validate()
    
    # Invalid quantization bits
    config_invalid = Config(quantization_bits=16)
    with pytest.raises(ValueError):
        config_invalid.validate()


def test_config_yaml_save_load():
    """Test saving and loading config from YAML"""
    config = Config(
        gpu_backend="opencl",
        max_vram_usage_mb=6000,
        use_quantization=True
    )
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        config.save_yaml(temp_path)
        
        # Load back
        loaded_config = Config.from_yaml(temp_path)
        
        assert loaded_config.gpu_backend == "opencl"
        assert loaded_config.max_vram_usage_mb == 6000
        assert loaded_config.use_quantization is True
    finally:
        os.unlink(temp_path)


def test_global_config():
    """Test global config management"""
    # Get default config
    config1 = get_config()
    assert config1 is not None
    
    # Set new config
    new_config = Config(gpu_backend="rocm")
    set_config(new_config)
    
    # Get should return the new config
    config2 = get_config()
    assert config2.gpu_backend == "rocm"


def test_config_from_dict():
    """Test creating config from dictionary"""
    data = {
        'gpu_backend': 'opencl',
        'max_vram_usage_mb': 5000,
        'use_quantization': True
    }
    
    config = Config.from_dict(data)
    assert config.gpu_backend == 'opencl'
    assert config.max_vram_usage_mb == 5000
    assert config.use_quantization is True
