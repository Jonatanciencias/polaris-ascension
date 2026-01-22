"""
High-Level SDK API - Easy Mode
===============================

This module provides the simplest possible API for developers who want
to get started quickly without diving into configuration details.

Philosophy:
----------
- One-liners for common tasks
- Sensible defaults for everything
- Automatic optimization
- "It just works" experience

Target Audience:
---------------
- Students learning AI
- Rapid prototyping
- Demos and presentations
- Non-experts who need quick results

Example:
-------
    # Literally one line to run inference
    from src.sdk.easy import quick_inference
    
    result = quick_inference("image.jpg", "mobilenetv2.onnx")
    print(f"Prediction: {result['class']}")
    
    # Or use the QuickModel class
    from src.sdk.easy import QuickModel
    
    model = QuickModel.from_zoo("mobilenetv2")
    result = model.predict("image.jpg")

Version: 0.6.0-dev
Author: Legacy GPU AI Platform Team
License: MIT
"""

import os
import sys
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
import numpy as np
from dataclasses import dataclass
import warnings

# Import core components
try:
    from ..core.gpu import GPUManager
    from ..inference.onnx_engine import ONNXInferenceEngine, InferenceConfig
    from ..utils.config import Config
except ImportError:
    # Fallback for direct execution
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.gpu import GPUManager
    from src.inference.onnx_engine import ONNXInferenceEngine, InferenceConfig
    from src.utils.config import Config


@dataclass
class QuickResult:
    """
    Simple result container for quick inference.
    
    Attributes:
        class_name: Top predicted class name
        confidence: Confidence score (0-1)
        top_k: List of (class, confidence) tuples
        raw_output: Raw model output
        inference_time_ms: Time taken in milliseconds
        device: Device used (CPU/GPU)
    """
    class_name: str
    confidence: float
    top_k: List[tuple]
    raw_output: np.ndarray
    inference_time_ms: float
    device: str


class QuickModel:
    """
    Ultra-simple model wrapper for quick inference.
    
    This class provides the easiest possible API for running inference.
    Everything is automatic: hardware detection, optimization, preprocessing.
    
    Example:
        # From local file
        model = QuickModel("mobilenet.onnx")
        result = model.predict("cat.jpg")
        
        # From model zoo (planned)
        model = QuickModel.from_zoo("mobilenetv2")
        
        # Batch prediction
        results = model.predict_batch(["img1.jpg", "img2.jpg"])
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        auto_optimize: bool = True,
        device: str = "auto"
    ):
        """
        Initialize QuickModel.
        
        Args:
            model_path: Path to ONNX model file
            auto_optimize: Automatically apply optimizations (default: True)
            device: Device to use ('auto', 'cpu', 'gpu')
        """
        self.model_path = Path(model_path)
        self.auto_optimize = auto_optimize
        self.device = device
        
        # Validate model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Initialize hardware
        self.gpu_manager = GPUManager()
        self.gpu_available = self.gpu_manager.initialize()
        
        # Determine actual device
        if device == "auto":
            self.actual_device = "cuda" if self.gpu_available else "cpu"
        else:
            self.actual_device = device
        
        # Create inference config
        self.config = InferenceConfig(
            device=self.actual_device,
            batch_size=1,
            enable_profiling=False
        )
        
        # Initialize engine
        self.engine = ONNXInferenceEngine(self.config)
        
        # Load model
        self._load_model()
        
        # ImageNet class names (top-1000)
        self.class_names = self._get_imagenet_classes()
    
    def _load_model(self):
        """Load and optionally optimize the model."""
        try:
            self.engine.load_model(self.model_path)
            
            if self.auto_optimize:
                # Apply automatic optimizations
                # (In full implementation, would apply quantization, pruning, etc.)
                pass
                
        except Exception as e:
            warnings.warn(f"Model loading issue: {e}")
    
    def _get_imagenet_classes(self) -> List[str]:
        """Get ImageNet class names."""
        # Simplified - in full version would load from file
        return [f"class_{i}" for i in range(1000)]
    
    def predict(
        self,
        input_data: Union[str, Path, np.ndarray],
        top_k: int = 5
    ) -> QuickResult:
        """
        Run inference on a single input.
        
        Args:
            input_data: Image path or numpy array
            top_k: Number of top predictions to return
        
        Returns:
            QuickResult with prediction details
        """
        import time
        
        # Preprocess input
        if isinstance(input_data, (str, Path)):
            input_tensor = self._load_and_preprocess_image(input_data)
        else:
            input_tensor = input_data
        
        # Run inference
        start_time = time.time()
        output = self.engine.infer(input_tensor)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Process output
        if isinstance(output, dict):
            output = list(output.values())[0]
        
        # Get top-k predictions
        output_flat = output.flatten()
        top_indices = np.argsort(output_flat)[-top_k:][::-1]
        
        top_predictions = [
            (self.class_names[idx], float(output_flat[idx]))
            for idx in top_indices
        ]
        
        return QuickResult(
            class_name=top_predictions[0][0],
            confidence=top_predictions[0][1],
            top_k=top_predictions,
            raw_output=output,
            inference_time_ms=inference_time,
            device=self.actual_device
        )
    
    def predict_batch(
        self,
        inputs: List[Union[str, Path, np.ndarray]],
        top_k: int = 5
    ) -> List[QuickResult]:
        """
        Run inference on multiple inputs.
        
        Args:
            inputs: List of image paths or numpy arrays
            top_k: Number of top predictions per input
        
        Returns:
            List of QuickResult objects
        """
        results = []
        for input_data in inputs:
            result = self.predict(input_data, top_k)
            results.append(result)
        return results
    
    def _load_and_preprocess_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load and preprocess an image for inference.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed numpy array (1, 3, 224, 224)
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL required for image loading. Install: pip install pillow")
        
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to 224x224 (standard for ImageNet models)
        img = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize (ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array / 255.0 - mean) / std
        
        # Transpose to (C, H, W) and add batch dimension
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array.astype(np.float32)
    
    @classmethod
    def from_zoo(cls, model_name: str, **kwargs) -> "QuickModel":
        """
        Load a pre-configured model from the model zoo.
        
        Args:
            model_name: Name of model ('mobilenetv2', 'resnet18', etc.)
            **kwargs: Additional arguments for QuickModel
        
        Returns:
            Initialized QuickModel
        
        Note:
            This is a placeholder. Full implementation would download
            models from a hosted zoo.
        """
        # Model zoo paths (would be remote URLs in production)
        zoo_paths = {
            'mobilenetv2': 'examples/models/mobilenetv2.onnx',
            'resnet18': 'examples/models/resnet18.onnx',
            'resnet50': 'examples/models/resnet50.onnx',
        }
        
        if model_name not in zoo_paths:
            raise ValueError(
                f"Model '{model_name}' not in zoo. "
                f"Available: {list(zoo_paths.keys())}"
            )
        
        model_path = zoo_paths[model_name]
        
        # Check if model exists locally
        if not Path(model_path).exists():
            warnings.warn(
                f"Model {model_name} not found locally. "
                f"Please download from the model zoo or provide your own."
            )
        
        return cls(model_path, **kwargs)
    
    def benchmark(self, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark the model performance.
        
        Args:
            num_runs: Number of inference runs
        
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            self.engine.infer(dummy_input)
            times.append((time.time() - start) * 1000)  # ms
        
        times = np.array(times)
        
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'median_ms': float(np.median(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
        }


def quick_inference(
    input_data: Union[str, Path, np.ndarray],
    model_path: Union[str, Path],
    top_k: int = 5,
    device: str = "auto"
) -> QuickResult:
    """
    Run inference with a single function call.
    
    This is the absolute simplest way to use the platform:
    just provide an input and model, get a prediction.
    
    Args:
        input_data: Image path or numpy array
        model_path: Path to ONNX model
        top_k: Number of top predictions
        device: Device to use ('auto', 'cpu', 'gpu')
    
    Returns:
        QuickResult with prediction
    
    Example:
        >>> result = quick_inference("cat.jpg", "mobilenet.onnx")
        >>> print(f"{result.class_name}: {result.confidence:.2%}")
        tabby_cat: 87.3%
    """
    model = QuickModel(model_path, device=device)
    return model.predict(input_data, top_k)


def quick_benchmark(
    model_path: Union[str, Path],
    num_runs: int = 100,
    device: str = "auto"
) -> Dict[str, float]:
    """
    Benchmark a model with a single function call.
    
    Args:
        model_path: Path to ONNX model
        num_runs: Number of inference runs
        device: Device to use
    
    Returns:
        Dictionary with timing statistics
    
    Example:
        >>> stats = quick_benchmark("mobilenet.onnx")
        >>> print(f"Average: {stats['mean_ms']:.2f} ms")
        Average: 12.34 ms
    """
    model = QuickModel(model_path, device=device)
    return model.benchmark(num_runs)


class AutoOptimizer:
    """
    Automatically apply optimizations to models.
    
    This class analyzes a model and applies appropriate optimizations
    based on the target hardware and accuracy requirements.
    
    Example:
        optimizer = AutoOptimizer(target_device="rx580")
        optimized_model = optimizer.optimize("model.onnx")
    """
    
    def __init__(
        self,
        target_device: str = "auto",
        target_accuracy_loss: float = 0.01,  # Max 1% accuracy loss
        target_speedup: float = 2.0,  # Aim for 2x speedup
    ):
        """
        Initialize AutoOptimizer.
        
        Args:
            target_device: Target hardware ('auto', 'rx580', 'vega', etc.)
            target_accuracy_loss: Maximum acceptable accuracy loss
            target_speedup: Target speedup factor
        """
        self.target_device = target_device
        self.target_accuracy_loss = target_accuracy_loss
        self.target_speedup = target_speedup
        
        # Initialize hardware detector
        self.gpu_manager = GPUManager()
        self.gpu_manager.initialize()
    
    def optimize(
        self,
        model_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Automatically optimize a model.
        
        Args:
            model_path: Input model path
            output_path: Output model path (default: add '_optimized' suffix)
        
        Returns:
            Path to optimized model
        """
        model_path = Path(model_path)
        
        if output_path is None:
            output_path = model_path.parent / f"{model_path.stem}_optimized{model_path.suffix}"
        else:
            output_path = Path(output_path)
        
        # In full implementation, would:
        # 1. Analyze model architecture
        # 2. Detect hardware capabilities
        # 3. Apply quantization if beneficial
        # 4. Apply pruning if beneficial
        # 5. Fuse operations
        # 6. Optimize memory layout
        
        # For now, just copy (placeholder)
        import shutil
        shutil.copy(model_path, output_path)
        
        return output_path
    
    def suggest_optimizations(
        self,
        model_path: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """
        Suggest possible optimizations without applying them.
        
        Args:
            model_path: Path to model
        
        Returns:
            List of suggested optimizations with expected benefits
        """
        # Placeholder - would analyze model and return suggestions
        suggestions = [
            {
                'name': 'INT8 Quantization',
                'expected_speedup': 2.5,
                'expected_memory_reduction': 0.75,
                'expected_accuracy_loss': 0.005,
                'recommended': True
            },
            {
                'name': 'Operator Fusion',
                'expected_speedup': 1.3,
                'expected_memory_reduction': 0.1,
                'expected_accuracy_loss': 0.0,
                'recommended': True
            },
        ]
        
        return suggestions


# Demo code
if __name__ == "__main__":
    print("=" * 70)
    print("Legacy GPU AI Platform - High-Level SDK Demo")
    print("=" * 70)
    
    print("\n1. Quick Inference (one-liner)")
    print("-" * 70)
    print("# Just one line to run inference!")
    print("result = quick_inference('cat.jpg', 'mobilenet.onnx')")
    print("print(f'{result.class_name}: {result.confidence:.2%}')")
    
    print("\n2. QuickModel Class")
    print("-" * 70)
    print("# For repeated inference")
    print("model = QuickModel('mobilenet.onnx')")
    print("result = model.predict('cat.jpg')")
    print("print(result.class_name)")
    
    print("\n3. Batch Prediction")
    print("-" * 70)
    print("# Process multiple images")
    print("images = ['cat1.jpg', 'cat2.jpg', 'dog.jpg']")
    print("results = model.predict_batch(images)")
    print("for r in results: print(r.class_name)")
    
    print("\n4. Benchmarking")
    print("-" * 70)
    print("# Quick performance check")
    print("stats = quick_benchmark('mobilenet.onnx')")
    print("print(f'Average: {stats[\"mean_ms\"]:.2f} ms')")
    
    print("\n5. Auto-Optimization")
    print("-" * 70)
    print("# Automatically optimize for your hardware")
    print("optimizer = AutoOptimizer(target_device='rx580')")
    print("optimized = optimizer.optimize('model.onnx')")
    print("suggestions = optimizer.suggest_optimizations('model.onnx')")
    
    print("\n" + "=" * 70)
    print("✅ High-Level SDK provides the easiest API possible!")
    print("=" * 70)
