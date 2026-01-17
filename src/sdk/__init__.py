"""
Legacy GPU AI Platform - SDK Layer
==================================

This module provides the public-facing API for developers building
AI applications on AMD legacy GPUs.

Design Philosophy:
-----------------
The SDK is designed to be:
1. Simple - Common operations should be one-liners
2. Explicit - No magic, users understand what's happening
3. Portable - Code written for Polaris works on Vega with minimal changes
4. Educational - Extensive documentation and examples

API Levels:
----------
1. High-Level API (sdk.easy)
   - One-line model loading and inference
   - Automatic hardware detection
   - Sensible defaults for everything

2. Standard API (sdk.core)
   - Full control over model configuration
   - Custom optimization pipelines
   - Memory management

3. Expert API (sdk.advanced)
   - Direct access to compute primitives
   - Custom kernel development
   - Cluster coordination

Target Audience:
---------------
- Students learning AI with limited hardware
- Researchers in emerging countries
- Developers repurposing consumer GPUs
- Educators building accessible AI curricula

Version: 0.5.0-dev
License: MIT
"""

__version__ = "0.5.0-dev"
__all__ = [
    "Platform",
    "Model",
    "Inference",
    "get_platform",
    "quick_inference",
]


class PlatformNotInitializedError(Exception):
    """Raised when SDK methods are called before platform initialization."""
    pass


class Platform:
    """
    Main entry point for the Legacy GPU AI Platform SDK.
    
    This class provides hardware detection, configuration, and
    serves as the factory for creating inference engines.
    
    Example:
        from src.sdk import Platform
        
        # Initialize platform (auto-detects hardware)
        platform = Platform.initialize()
        
        # Get platform information
        print(platform.gpu_info())
        
        # Load a model
        model = platform.load_model("mobilenetv2.onnx")
        
        # Run inference
        result = model.infer(image_tensor)
    """
    
    _instance = None
    _initialized = False
    
    def __init__(self):
        """Private constructor - use Platform.initialize() instead."""
        self.gpu_family = None
        self.gpu_name = None
        self.vram_gb = 0
        self.driver_version = None
        self._models = {}
        
    @classmethod
    def initialize(cls, auto_detect: bool = True) -> "Platform":
        """
        Initialize the platform and detect hardware.
        
        Args:
            auto_detect: Automatically detect GPU (default True)
            
        Returns:
            Initialized Platform instance
        """
        if cls._instance is None:
            cls._instance = cls()
            
        if auto_detect:
            cls._instance._detect_hardware()
            
        cls._initialized = True
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> "Platform":
        """Get the current platform instance."""
        if not cls._initialized:
            raise PlatformNotInitializedError(
                "Call Platform.initialize() first"
            )
        return cls._instance
    
    def _detect_hardware(self):
        """Detect AMD GPU hardware."""
        # Import core GPU module
        try:
            from src.core.gpu import GPUManager
            gpu_manager = GPUManager()
            info = gpu_manager.get_info()
            
            self.gpu_name = info.get("device_name", "Unknown")
            self.vram_gb = info.get("memory_total_gb", 0)
            self.driver_version = info.get("driver_version", "Unknown")
            
            # Determine GPU family
            self.gpu_family = self._classify_gpu(self.gpu_name)
            
        except Exception as e:
            # Fallback to basic detection
            self.gpu_name = "Unknown AMD GPU"
            self.gpu_family = "unknown"
            self.vram_gb = 0
            
    def _classify_gpu(self, gpu_name: str) -> str:
        """Classify GPU into family based on name."""
        name_lower = gpu_name.lower()
        
        if any(x in name_lower for x in ["rx 580", "rx 570", "rx 480", "rx 470"]):
            return "polaris"
        elif any(x in name_lower for x in ["vega", "radeon vii"]):
            return "vega"
        elif any(x in name_lower for x in ["rx 5", "navi"]):
            return "navi"
        elif any(x in name_lower for x in ["rx 6", "rdna2"]):
            return "rdna2"
        else:
            return "unknown"
    
    def gpu_info(self) -> dict:
        """
        Get current GPU information.
        
        Returns:
            dict with GPU details
        """
        return {
            "name": self.gpu_name,
            "family": self.gpu_family,
            "vram_gb": self.vram_gb,
            "driver_version": self.driver_version,
            "supported": self.gpu_family in ["polaris", "vega", "navi"],
        }
    
    def capabilities(self) -> dict:
        """
        Get platform capabilities for current hardware.
        
        Returns:
            dict with capability flags
        """
        base_caps = {
            "onnx_inference": True,
            "fp32_compute": True,
            "memory_optimization": True,
        }
        
        family_caps = {
            "polaris": {
                "fp16_acceleration": False,
                "int8_emulation": True,
                "max_batch_size": 8,  # 8GB VRAM constraint
                "recommended_models": ["mobilenet", "efficientnet-lite", "yolov5s"],
            },
            "vega": {
                "fp16_acceleration": True,  # Rapid Packed Math
                "int8_emulation": True,
                "max_batch_size": 16,
                "recommended_models": ["resnet50", "yolov5m", "bert-base"],
            },
            "navi": {
                "fp16_acceleration": True,
                "int8_emulation": True,
                "max_batch_size": 16,
                "recommended_models": ["resnet101", "yolov5l", "bert-large"],
            },
        }
        
        return {**base_caps, **family_caps.get(self.gpu_family, {})}
    
    def load_model(self, model_path: str, **kwargs) -> "Model":
        """
        Load a model for inference.
        
        Args:
            model_path: Path to ONNX model file
            **kwargs: Additional model configuration
            
        Returns:
            Model instance ready for inference
        """
        model = Model(model_path, platform=self, **kwargs)
        self._models[model_path] = model
        return model
    
    def status(self) -> str:
        """Get a formatted status string."""
        info = self.gpu_info()
        caps = self.capabilities()
        
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║           Legacy GPU AI Platform - Status                        ║
╠══════════════════════════════════════════════════════════════════╣
║ GPU: {info['name']:<57} ║
║ Family: {info['family']:<54} ║
║ VRAM: {info['vram_gb']:.1f} GB{' '*51} ║
║ Status: {'✅ Supported' if info['supported'] else '⚠️ Unknown':<55} ║
╠══════════════════════════════════════════════════════════════════╣
║ Capabilities:                                                    ║
║   • ONNX Inference: {'Yes' if caps.get('onnx_inference') else 'No':<42} ║
║   • FP16 Acceleration: {'Yes' if caps.get('fp16_acceleration') else 'No':<39} ║
║   • Memory Optimization: {'Yes' if caps.get('memory_optimization') else 'No':<37} ║
╚══════════════════════════════════════════════════════════════════╝
"""


class Model:
    """
    Represents a loaded AI model ready for inference.
    
    This class wraps the underlying inference engine and provides
    a simple, consistent interface for running predictions.
    """
    
    def __init__(self, model_path: str, platform: Platform, **kwargs):
        """
        Initialize model (use Platform.load_model instead).
        
        Args:
            model_path: Path to model file
            platform: Parent Platform instance
            **kwargs: Model configuration
        """
        self.model_path = model_path
        self.platform = platform
        self.config = kwargs
        self._engine = None
        self._loaded = False
        
        self._load()
        
    def _load(self):
        """Load the model into the inference engine."""
        try:
            from src.inference.onnx_engine import ONNXInferenceEngine
            self._engine = ONNXInferenceEngine(self.model_path)
            self._loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def infer(self, input_data, **kwargs):
        """
        Run inference on input data.
        
        Args:
            input_data: Input tensor or batch
            **kwargs: Inference options
            
        Returns:
            Model predictions
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        return self._engine.infer(input_data)
    
    def benchmark(self, iterations: int = 100) -> dict:
        """
        Run inference benchmark.
        
        Args:
            iterations: Number of iterations to run
            
        Returns:
            dict with timing statistics
        """
        # Placeholder - will integrate with profiler
        return {
            "iterations": iterations,
            "status": "not_implemented",
            "message": "Benchmark integration coming in v0.6.0"
        }


def get_platform() -> Platform:
    """
    Get the initialized platform instance.
    
    Shorthand for Platform.get_instance().
    
    Returns:
        Platform instance
    """
    try:
        return Platform.get_instance()
    except PlatformNotInitializedError:
        return Platform.initialize()


def quick_inference(model_path: str, input_data) -> any:
    """
    One-liner inference for simple use cases.
    
    Args:
        model_path: Path to ONNX model
        input_data: Input tensor
        
    Returns:
        Model predictions
    """
    platform = get_platform()
    model = platform.load_model(model_path)
    return model.infer(input_data)
