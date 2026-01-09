"""
Configuration Management Module

Handles loading and managing configuration settings.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration container"""
    
    # GPU settings
    gpu_backend: str = "auto"  # auto, opencl, rocm, cpu
    gpu_device_id: int = 0
    
    # Memory settings
    max_vram_usage_mb: int = 7168  # Leave 1GB headroom for 8GB cards
    enable_cpu_offload: bool = True
    offload_threshold_mb: int = 512
    
    # Optimization settings
    use_quantization: bool = False
    quantization_bits: int = 8  # 8 or 4
    use_flash_attention: bool = False
    use_xformers: bool = False
    
    # Inference settings
    default_batch_size: int = 1
    max_batch_size: int = 4
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    
    # Model settings
    model_cache_dir: str = "./models"
    default_model: str = "stabilityai/stable-diffusion-2-1-base"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Performance
    enable_profiling: bool = False
    benchmark_mode: bool = False
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Config object
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """
        Create config from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Config object
        """
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'gpu_backend': self.gpu_backend,
            'gpu_device_id': self.gpu_device_id,
            'max_vram_usage_mb': self.max_vram_usage_mb,
            'enable_cpu_offload': self.enable_cpu_offload,
            'offload_threshold_mb': self.offload_threshold_mb,
            'use_quantization': self.use_quantization,
            'quantization_bits': self.quantization_bits,
            'use_flash_attention': self.use_flash_attention,
            'use_xformers': self.use_xformers,
            'default_batch_size': self.default_batch_size,
            'max_batch_size': self.max_batch_size,
            'num_inference_steps': self.num_inference_steps,
            'guidance_scale': self.guidance_scale,
            'model_cache_dir': self.model_cache_dir,
            'default_model': self.default_model,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'enable_profiling': self.enable_profiling,
            'benchmark_mode': self.benchmark_mode,
        }
    
    def save_yaml(self, path: str):
        """
        Save configuration to YAML file.
        
        Args:
            path: Output path for YAML file
        """
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        if self.gpu_backend not in ['auto', 'opencl', 'rocm', 'cpu']:
            raise ValueError(f"Invalid GPU backend: {self.gpu_backend}")
        
        if self.quantization_bits not in [4, 8]:
            raise ValueError(f"Invalid quantization bits: {self.quantization_bits}")
        
        if self.max_vram_usage_mb < 1024:
            raise ValueError("max_vram_usage_mb must be at least 1024 MB")
        
        if self.default_batch_size > self.max_batch_size:
            raise ValueError("default_batch_size cannot exceed max_batch_size")
        
        return True


# Global config instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config):
    """Set global configuration instance"""
    global _global_config
    config.validate()
    _global_config = config


def load_config(path: str) -> Config:
    """Load and set global configuration from file"""
    config = Config.from_yaml(path)
    set_config(config)
    return config
