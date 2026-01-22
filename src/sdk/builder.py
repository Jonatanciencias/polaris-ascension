"""
Builder Pattern API for Fluent SDK Usage
========================================

This module provides a fluent, chainable API for building
inference pipelines and configurations.

Design Pattern:
--------------
Builder pattern allows constructing complex objects step by step
with a clean, readable syntax.

Example:
-------
    from src.sdk.builder import InferencePipeline
    
    # Fluent API - chain method calls
    pipeline = (InferencePipeline()
        .use_model("mobilenetv2.onnx")
        .on_device("rx580")
        .with_batch_size(32)
        .enable_int8_quantization()
        .add_preprocessing(resize=(224, 224))
        .add_postprocessing(top_k=5)
        .build()
    )
    
    # Execute pipeline
    results = pipeline.run("image.jpg")
    
    # Or use ConfigBuilder for more control
    config = (ConfigBuilder()
        .for_task("classification")
        .optimize_for("speed")  # or "accuracy", "balanced"
        .target_fps(60)
        .max_memory_mb(2000)
        .build()
    )

Benefits:
--------
- Readable, self-documenting code
- Type-safe configuration
- Sensible defaults
- Easy to extend
- IDE auto-completion friendly

Version: 0.6.0-dev
Author: Legacy GPU AI Platform Team
License: MIT
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
import numpy as np


class OptimizationGoal(Enum):
    """Optimization goal for pipeline."""
    SPEED = "speed"
    ACCURACY = "accuracy"
    BALANCED = "balanced"
    MEMORY = "memory"


class DeviceType(Enum):
    """Supported device types."""
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"
    RX580 = "rx580"
    RX570 = "rx570"
    VEGA = "vega"


@dataclass
class PipelineConfig:
    """
    Complete configuration for an inference pipeline.
    
    This dataclass holds all configuration parameters needed
    to build and execute an inference pipeline.
    """
    # Model configuration
    model_path: Optional[Path] = None
    model_format: str = "onnx"
    
    # Device configuration
    device: DeviceType = DeviceType.AUTO
    batch_size: int = 1
    num_threads: int = 4
    
    # Optimization configuration
    optimization_goal: OptimizationGoal = OptimizationGoal.BALANCED
    enable_quantization: bool = False
    quantization_level: str = "int8"
    enable_pruning: bool = False
    pruning_ratio: float = 0.5
    
    # Preprocessing
    preprocessing_steps: List[Dict[str, Any]] = field(default_factory=list)
    input_shape: Optional[Tuple[int, ...]] = None
    normalization_mean: Optional[List[float]] = None
    normalization_std: Optional[List[float]] = None
    
    # Postprocessing
    postprocessing_steps: List[Dict[str, Any]] = field(default_factory=list)
    top_k: int = 5
    confidence_threshold: float = 0.5
    
    # Performance constraints
    target_fps: Optional[float] = None
    max_memory_mb: Optional[int] = None
    max_latency_ms: Optional[float] = None
    
    # Monitoring
    enable_profiling: bool = False
    enable_logging: bool = True
    log_level: str = "INFO"


class InferencePipeline:
    """
    Fluent API for building inference pipelines.
    
    This class provides a chainable interface for configuring
    and executing inference pipelines.
    
    Example:
        pipeline = (InferencePipeline()
            .use_model("model.onnx")
            .on_device("rx580")
            .with_batch_size(16)
            .enable_int8_quantization()
            .build()
        )
        
        result = pipeline.run("input.jpg")
    """
    
    def __init__(self):
        """Initialize empty pipeline."""
        self._config = PipelineConfig()
        self._model = None
        self._built = False
    
    def use_model(self, model_path: Union[str, Path]) -> "InferencePipeline":
        """
        Set the model to use.
        
        Args:
            model_path: Path to model file
        
        Returns:
            Self for chaining
        """
        self._config.model_path = Path(model_path)
        return self
    
    def on_device(
        self,
        device: Union[str, DeviceType]
    ) -> "InferencePipeline":
        """
        Set target device.
        
        Args:
            device: Device type or name
        
        Returns:
            Self for chaining
        """
        if isinstance(device, str):
            # Map common names
            device_map = {
                'auto': DeviceType.AUTO,
                'cpu': DeviceType.CPU,
                'gpu': DeviceType.GPU,
                'rx580': DeviceType.RX580,
                'rx570': DeviceType.RX570,
                'vega': DeviceType.VEGA,
            }
            device = device_map.get(device.lower(), DeviceType.AUTO)
        
        self._config.device = device
        return self
    
    def with_batch_size(self, batch_size: int) -> "InferencePipeline":
        """
        Set batch size.
        
        Args:
            batch_size: Batch size (must be > 0)
        
        Returns:
            Self for chaining
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        self._config.batch_size = batch_size
        return self
    
    def optimize_for(
        self,
        goal: Union[str, OptimizationGoal]
    ) -> "InferencePipeline":
        """
        Set optimization goal.
        
        Args:
            goal: Optimization goal
        
        Returns:
            Self for chaining
        """
        if isinstance(goal, str):
            goal = OptimizationGoal(goal.lower())
        
        self._config.optimization_goal = goal
        return self
    
    def enable_int8_quantization(self) -> "InferencePipeline":
        """
        Enable INT8 quantization.
        
        Returns:
            Self for chaining
        """
        self._config.enable_quantization = True
        self._config.quantization_level = "int8"
        return self
    
    def enable_fp16_quantization(self) -> "InferencePipeline":
        """
        Enable FP16 quantization.
        
        Returns:
            Self for chaining
        """
        self._config.enable_quantization = True
        self._config.quantization_level = "fp16"
        return self
    
    def enable_pruning(
        self,
        ratio: float = 0.5
    ) -> "InferencePipeline":
        """
        Enable model pruning.
        
        Args:
            ratio: Pruning ratio (0-1)
        
        Returns:
            Self for chaining
        """
        if not 0 <= ratio <= 1:
            raise ValueError("Pruning ratio must be in [0, 1]")
        
        self._config.enable_pruning = True
        self._config.pruning_ratio = ratio
        return self
    
    def add_preprocessing(
        self,
        resize: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        **kwargs
    ) -> "InferencePipeline":
        """
        Add preprocessing step.
        
        Args:
            resize: Target size (height, width)
            normalize: Whether to normalize
            mean: Normalization mean
            std: Normalization std
            **kwargs: Additional preprocessing parameters
        
        Returns:
            Self for chaining
        """
        step = {'type': 'preprocess'}
        
        if resize:
            step['resize'] = resize
        
        if normalize:
            step['normalize'] = True
            step['mean'] = mean or [0.485, 0.456, 0.406]
            step['std'] = std or [0.229, 0.224, 0.225]
        
        step.update(kwargs)
        
        self._config.preprocessing_steps.append(step)
        return self
    
    def add_postprocessing(
        self,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        **kwargs
    ) -> "InferencePipeline":
        """
        Add postprocessing step.
        
        Args:
            top_k: Number of top predictions
            threshold: Confidence threshold
            **kwargs: Additional postprocessing parameters
        
        Returns:
            Self for chaining
        """
        step = {'type': 'postprocess'}
        
        if top_k is not None:
            step['top_k'] = top_k
            self._config.top_k = top_k
        
        if threshold is not None:
            step['threshold'] = threshold
            self._config.confidence_threshold = threshold
        
        step.update(kwargs)
        
        self._config.postprocessing_steps.append(step)
        return self
    
    def target_fps(self, fps: float) -> "InferencePipeline":
        """
        Set target FPS constraint.
        
        Args:
            fps: Target frames per second
        
        Returns:
            Self for chaining
        """
        self._config.target_fps = fps
        return self
    
    def max_memory(self, memory_mb: int) -> "InferencePipeline":
        """
        Set maximum memory constraint.
        
        Args:
            memory_mb: Maximum memory in MB
        
        Returns:
            Self for chaining
        """
        self._config.max_memory_mb = memory_mb
        return self
    
    def enable_profiling(self) -> "InferencePipeline":
        """
        Enable performance profiling.
        
        Returns:
            Self for chaining
        """
        self._config.enable_profiling = True
        return self
    
    def with_logging(self, level: str = "INFO") -> "InferencePipeline":
        """
        Configure logging.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        
        Returns:
            Self for chaining
        """
        self._config.enable_logging = True
        self._config.log_level = level.upper()
        return self
    
    def build(self) -> "InferencePipeline":
        """
        Build the pipeline.
        
        This finalizes the configuration and initializes
        the inference engine.
        
        Returns:
            Self (now ready to run)
        """
        # Validate configuration
        if self._config.model_path is None:
            raise ValueError("Model path not set. Use .use_model()")
        
        if not self._config.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._config.model_path}")
        
        # In full implementation, would:
        # 1. Load model
        # 2. Apply optimizations
        # 3. Initialize inference engine
        # 4. Set up preprocessing/postprocessing
        
        self._built = True
        return self
    
    def run(
        self,
        input_data: Union[str, Path, np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the pipeline on input data.
        
        Args:
            input_data: Input image path or array
            **kwargs: Additional runtime parameters
        
        Returns:
            Inference results
        """
        if not self._built:
            raise RuntimeError("Pipeline not built. Call .build() first")
        
        # Placeholder implementation
        result = {
            'predictions': [],
            'confidence': 0.0,
            'inference_time_ms': 0.0,
        }
        
        return result
    
    def get_config(self) -> PipelineConfig:
        """
        Get pipeline configuration.
        
        Returns:
            PipelineConfig object
        """
        return self._config


class ConfigBuilder:
    """
    Builder for configuration objects.
    
    Provides a fluent API for creating configurations
    for various components.
    
    Example:
        config = (ConfigBuilder()
            .for_task("classification")
            .optimize_for("speed")
            .target_fps(60)
            .build()
        )
    """
    
    def __init__(self):
        """Initialize builder."""
        self._task: Optional[str] = None
        self._optimization_goal = OptimizationGoal.BALANCED
        self._constraints: Dict[str, Any] = {}
        self._features: Dict[str, bool] = {}
    
    def for_task(self, task: str) -> "ConfigBuilder":
        """
        Set target task.
        
        Args:
            task: Task name (classification, detection, etc.)
        
        Returns:
            Self for chaining
        """
        self._task = task
        return self
    
    def optimize_for(
        self,
        goal: Union[str, OptimizationGoal]
    ) -> "ConfigBuilder":
        """
        Set optimization goal.
        
        Args:
            goal: Optimization goal
        
        Returns:
            Self for chaining
        """
        if isinstance(goal, str):
            goal = OptimizationGoal(goal.lower())
        
        self._optimization_goal = goal
        return self
    
    def target_fps(self, fps: float) -> "ConfigBuilder":
        """
        Set target FPS.
        
        Args:
            fps: Target frames per second
        
        Returns:
            Self for chaining
        """
        self._constraints['target_fps'] = fps
        return self
    
    def max_memory_mb(self, memory_mb: int) -> "ConfigBuilder":
        """
        Set maximum memory.
        
        Args:
            memory_mb: Maximum memory in MB
        
        Returns:
            Self for chaining
        """
        self._constraints['max_memory_mb'] = memory_mb
        return self
    
    def enable_feature(self, feature: str) -> "ConfigBuilder":
        """
        Enable a feature.
        
        Args:
            feature: Feature name
        
        Returns:
            Self for chaining
        """
        self._features[feature] = True
        return self
    
    def disable_feature(self, feature: str) -> "ConfigBuilder":
        """
        Disable a feature.
        
        Args:
            feature: Feature name
        
        Returns:
            Self for chaining
        """
        self._features[feature] = False
        return self
    
    def build(self) -> Dict[str, Any]:
        """
        Build configuration dictionary.
        
        Returns:
            Configuration dictionary
        """
        config = {
            'task': self._task,
            'optimization_goal': self._optimization_goal.value,
            'constraints': self._constraints,
            'features': self._features,
        }
        
        return config


class ModelBuilder:
    """
    Builder for model configurations.
    
    Simplifies creating model configurations with
    hardware-specific optimizations.
    
    Example:
        model = (ModelBuilder()
            .load("resnet18.onnx")
            .for_hardware("rx580")
            .quantize_to("int8")
            .fuse_operations()
            .optimize_memory_layout()
            .build()
        )
    """
    
    def __init__(self):
        """Initialize builder."""
        self._model_path: Optional[Path] = None
        self._hardware: str = "auto"
        self._optimizations: List[str] = []
        self._quantization: Optional[str] = None
    
    def load(self, model_path: Union[str, Path]) -> "ModelBuilder":
        """
        Set model path.
        
        Args:
            model_path: Path to model file
        
        Returns:
            Self for chaining
        """
        self._model_path = Path(model_path)
        return self
    
    def for_hardware(self, hardware: str) -> "ModelBuilder":
        """
        Target specific hardware.
        
        Args:
            hardware: Hardware identifier (rx580, vega, etc.)
        
        Returns:
            Self for chaining
        """
        self._hardware = hardware
        return self
    
    def quantize_to(self, level: str) -> "ModelBuilder":
        """
        Set quantization level.
        
        Args:
            level: Quantization level (int8, fp16, etc.)
        
        Returns:
            Self for chaining
        """
        self._quantization = level
        self._optimizations.append(f"quantize_{level}")
        return self
    
    def fuse_operations(self) -> "ModelBuilder":
        """
        Enable operation fusion.
        
        Returns:
            Self for chaining
        """
        self._optimizations.append("fuse_ops")
        return self
    
    def optimize_memory_layout(self) -> "ModelBuilder":
        """
        Optimize memory layout.
        
        Returns:
            Self for chaining
        """
        self._optimizations.append("optimize_memory")
        return self
    
    def build(self) -> Dict[str, Any]:
        """
        Build model configuration.
        
        Returns:
            Model configuration dictionary
        """
        config = {
            'model_path': str(self._model_path) if self._model_path else None,
            'hardware': self._hardware,
            'optimizations': self._optimizations,
            'quantization': self._quantization,
        }
        
        return config


# Demo code
if __name__ == "__main__":
    print("=" * 70)
    print("Legacy GPU AI Platform - Builder Pattern API Demo")
    print("=" * 70)
    
    print("\n1. InferencePipeline - Fluent API")
    print("-" * 70)
    print("Code:")
    print("""
    pipeline = (InferencePipeline()
        .use_model("mobilenetv2.onnx")
        .on_device("rx580")
        .with_batch_size(32)
        .optimize_for("speed")
        .enable_int8_quantization()
        .add_preprocessing(resize=(224, 224))
        .add_postprocessing(top_k=5)
        .target_fps(60)
        .enable_profiling()
        .build()
    )
    """)
    
    print("\n2. ConfigBuilder")
    print("-" * 70)
    builder = ConfigBuilder()
    config = (builder
        .for_task("classification")
        .optimize_for("speed")
        .target_fps(60)
        .max_memory_mb(2000)
        .enable_feature("quantization")
        .build()
    )
    print("Configuration:", config)
    
    print("\n3. ModelBuilder")
    print("-" * 70)
    print("Code:")
    print("""
    model = (ModelBuilder()
        .load("resnet18.onnx")
        .for_hardware("rx580")
        .quantize_to("int8")
        .fuse_operations()
        .optimize_memory_layout()
        .build()
    )
    """)
    
    print("\n" + "=" * 70)
    print("✅ Builder pattern provides clean, readable configuration!")
    print("=" * 70)
