"""
Base Inference Engine

Abstract base class for all inference engines in the framework.
Provides common interface and integration with GPU/Memory managers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from ..core.gpu import GPUManager
from ..core.memory import MemoryManager
from ..core.profiler import Profiler


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    path: str
    input_shape: tuple
    output_shape: tuple
    precision: str  # 'fp32', 'fp16', 'int8'
    memory_usage_mb: float
    backend: str  # 'opencl', 'cpu', 'rocm'


@dataclass
class InferenceConfig:
    """Configuration for inference engine"""
    device: str = 'auto'  # 'auto', 'opencl', 'cpu'
    precision: str = 'fp32'  # 'fp32', 'fp16', 'int8'
    batch_size: int = 1
    enable_profiling: bool = True
    memory_limit_mb: Optional[float] = None
    optimization_level: int = 2  # 0=none, 1=basic, 2=aggressive


class BaseInferenceEngine(ABC):
    """
    Abstract base class for inference engines.
    
    Provides common functionality:
    - Model loading and validation
    - Input/output preprocessing
    - Memory management integration
    - Performance profiling
    - Hardware optimization
    
    Real-world applications this enables:
    - Medical imaging analysis on affordable hardware
    - Wildlife monitoring in remote locations
    - Educational AI tools for underserved communities
    - Small business automation (inventory, QA)
    - Personal creative tools (art, photography)
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        gpu_manager: Optional[GPUManager] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            config: Inference configuration
            gpu_manager: Optional GPU manager (created if not provided)
            memory_manager: Optional memory manager (created if not provided)
        """
        self.config = config
        self.gpu_manager = gpu_manager or GPUManager()
        self.memory_manager = memory_manager or MemoryManager()
        self.profiler = Profiler() if config.enable_profiling else None
        self.model_info: Optional[ModelInfo] = None
        self._session = None
        
        # Initialize GPU if available
        if self.config.device == 'auto':
            self.gpu_manager.initialize()
            self.config.device = self.gpu_manager.get_compute_backend()
    
    @abstractmethod
    def load_model(self, model_path: Union[str, Path]) -> ModelInfo:
        """
        Load a model from file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            ModelInfo with model metadata
        """
        pass
    
    @abstractmethod
    def preprocess(self, inputs: Any) -> np.ndarray:
        """
        Preprocess inputs for model.
        
        Args:
            inputs: Raw inputs (image, text, etc.)
            
        Returns:
            Preprocessed numpy array ready for inference
        """
        pass
    
    @abstractmethod
    def postprocess(self, outputs: np.ndarray) -> Any:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Processed results (labels, boxes, etc.)
        """
        pass
    
    @abstractmethod
    def _run_inference(self, inputs: np.ndarray) -> np.ndarray:
        """
        Internal method to run model inference.
        
        Args:
            inputs: Preprocessed inputs
            
        Returns:
            Raw model outputs
        """
        pass
    
    def infer(self, inputs: Any, profile: bool = False) -> Any:
        """
        Run complete inference pipeline.
        
        Args:
            inputs: Raw inputs
            profile: Whether to profile this inference
            
        Returns:
            Processed model outputs
        """
        if self.model_info is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Start profiling if enabled
        if profile and self.profiler:
            self.profiler.start('total_inference')
        
        try:
            # Preprocess
            if profile and self.profiler:
                self.profiler.start('preprocessing')
            processed_inputs = self.preprocess(inputs)
            if profile and self.profiler:
                self.profiler.end('preprocessing')
            
            # Check memory availability
            required_memory = processed_inputs.nbytes / (1024 * 1024)  # MB
            if not self.memory_manager.can_allocate(required_memory):
                raise MemoryError(
                    f"Insufficient memory for inference. "
                    f"Required: {required_memory:.1f}MB"
                )
            
            # Run inference
            if profile and self.profiler:
                self.profiler.start('inference')
            outputs = self._run_inference(processed_inputs)
            if profile and self.profiler:
                self.profiler.end('inference')
            
            # Postprocess
            if profile and self.profiler:
                self.profiler.start('postprocessing')
            results = self.postprocess(outputs)
            if profile and self.profiler:
                self.profiler.end('postprocessing')
            
            return results
            
        finally:
            if profile and self.profiler:
                self.profiler.end('total_inference')
    
    def batch_infer(self, inputs_list: List[Any]) -> List[Any]:
        """
        Run inference on batch of inputs.
        
        Args:
            inputs_list: List of inputs
            
        Returns:
            List of results
        """
        results = []
        for inputs in inputs_list:
            results.append(self.infer(inputs))
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from profiler"""
        if self.profiler is None:
            return {}
        return self.profiler.get_summary()
    
    def print_performance_stats(self):
        """Print performance statistics"""
        if self.profiler:
            self.profiler.print_summary()
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit"""
        if self._session:
            del self._session
        return False
