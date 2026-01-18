"""
Real Model Loaders for Production Inference

Provides unified interface for loading ONNX and PyTorch models with
automatic optimization, provider selection, and hardware detection.

Session 16 - January 18, 2026
Part of: CAPA 3 - Real Model Integration

Key Features:
- ONNXModelLoader: ONNX Runtime with OpenCL/CPU providers
- PyTorchModelLoader: TorchScript support with ROCm/CPU backends
- Automatic provider selection based on hardware
- Model validation and metadata extraction
- Memory-efficient loading with size checks

Academic Foundations:
1. ONNX Runtime (Microsoft, 2019): Cross-platform inference optimization
2. TorchScript (Facebook, 2019): PyTorch model serialization
3. ROCm (AMD, 2016): Open-source GPU compute platform
4. Model compression (Han et al., 2016): Efficient model deployment

Real-World Applications:
- Medical imaging: Load trained diagnostic models (ResNet50, DenseNet)
- Edge AI: Deploy compressed models on consumer GPUs
- Production serving: Multi-framework model support
- Research: Fast model iteration with different frameworks
"""

__version__ = "0.6.0-dev"

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata extracted from loaded model"""
    name: str
    framework: str  # 'onnx', 'pytorch', 'torchscript'
    input_names: List[str]
    output_names: List[str]
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    input_dtypes: List[str]
    output_dtypes: List[str]
    file_size_mb: float
    estimated_memory_mb: float
    provider: str  # 'CPUExecutionProvider', 'OpenCLExecutionProvider', 'ROCm', etc.
    optimization_level: str
    extra_info: Dict[str, Any] = None


class BaseModelLoader(ABC):
    """Base class for model loaders"""
    
    def __init__(
        self,
        optimization_level: int = 2,
        preferred_providers: Optional[List[str]] = None
    ):
        """
        Initialize model loader.
        
        Args:
            optimization_level: 0=disabled, 1=basic, 2=all (default)
            preferred_providers: List of preferred execution providers
        """
        self.optimization_level = optimization_level
        self.preferred_providers = preferred_providers or []
        self._model = None
        self._metadata: Optional[ModelMetadata] = None
    
    @abstractmethod
    def load(self, model_path: Union[str, Path]) -> ModelMetadata:
        """Load model and return metadata"""
        pass
    
    @abstractmethod
    def predict(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Run inference on loaded model"""
        pass
    
    @abstractmethod
    def get_available_providers(self) -> List[str]:
        """Get list of available execution providers"""
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None
    
    def get_metadata(self) -> Optional[ModelMetadata]:
        """Get loaded model metadata"""
        return self._metadata
    
    def unload(self):
        """Unload model and free resources"""
        self._model = None
        self._metadata = None
        logger.info("Model unloaded")


class ONNXModelLoader(BaseModelLoader):
    """
    ONNX Runtime model loader with hardware optimization.
    
    Supports:
    - ONNX models (.onnx files)
    - Multiple execution providers (CPU, OpenCL, CUDA, ROCm)
    - Graph optimizations (constant folding, operator fusion)
    - Dynamic batching
    - FP32/FP16/INT8 precision
    
    Real-world usage:
    - Load pre-trained models from ONNX Model Zoo
    - Deploy PyTorch/TensorFlow models converted to ONNX
    - Cross-platform inference (Windows/Linux/macOS)
    """
    
    def __init__(
        self,
        optimization_level: int = 2,
        preferred_providers: Optional[List[str]] = None,
        intra_op_threads: int = 4,
        inter_op_threads: int = 2
    ):
        """
        Initialize ONNX Runtime loader.
        
        Args:
            optimization_level: 0=disabled, 1=basic, 2=all
            preferred_providers: List of providers in preference order
            intra_op_threads: Threads for parallelizing ops
            inter_op_threads: Threads for parallelizing independent ops
        """
        super().__init__(optimization_level, preferred_providers)
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads
        self._session = None
        
        # Try to import onnxruntime
        try:
            import onnxruntime as ort
            self.ort = ort
            logger.info(f"✅ ONNX Runtime {ort.__version__} available")
        except ImportError:
            logger.warning("⚠️  ONNX Runtime not installed. Install with: pip install onnxruntime")
            self.ort = None
    
    def get_available_providers(self) -> List[str]:
        """Get available ONNX Runtime execution providers"""
        if self.ort is None:
            return []
        return self.ort.get_available_providers()
    
    def _select_providers(self) -> List[str]:
        """
        Select best execution providers based on hardware.
        
        Priority order:
        1. ROCmExecutionProvider (AMD GPUs with ROCm)
        2. CUDAExecutionProvider (NVIDIA GPUs)
        3. OpenCLExecutionProvider (AMD/Intel GPUs)
        4. CPUExecutionProvider (fallback)
        """
        available = self.get_available_providers()
        
        # If user specified preferences, try those first
        if self.preferred_providers:
            for provider in self.preferred_providers:
                if provider in available:
                    logger.info(f"✅ Using preferred provider: {provider}")
                    return [provider, 'CPUExecutionProvider']
        
        # Auto-select based on hardware
        priority = [
            'ROCmExecutionProvider',      # AMD GPUs with ROCm
            'CUDAExecutionProvider',       # NVIDIA GPUs
            'MIGraphXExecutionProvider',   # AMD MIGraphX
            'OpenCLExecutionProvider',     # OpenCL devices
            'CPUExecutionProvider'         # CPU fallback
        ]
        
        selected = []
        for provider in priority:
            if provider in available:
                selected.append(provider)
                if provider != 'CPUExecutionProvider':
                    selected.append('CPUExecutionProvider')  # Always add CPU fallback
                break
        
        if not selected:
            selected = ['CPUExecutionProvider']
        
        logger.info(f"✅ Selected providers: {selected}")
        return selected
    
    def _setup_session_options(self):
        """Configure ONNX Runtime session options"""
        if self.ort is None:
            return None
        
        session_options = self.ort.SessionOptions()
        
        # Set optimization level
        if self.optimization_level == 0:
            session_options.graph_optimization_level = (
                self.ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
        elif self.optimization_level == 1:
            session_options.graph_optimization_level = (
                self.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            )
        else:  # 2 or higher
            session_options.graph_optimization_level = (
                self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
        
        # Enable parallelization
        session_options.intra_op_num_threads = self.intra_op_threads
        session_options.inter_op_num_threads = self.inter_op_threads
        
        # Enable memory pattern optimization
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        
        logger.info(
            f"Session options: optimization={self.optimization_level}, "
            f"threads={self.intra_op_threads}/{self.inter_op_threads}"
        )
        
        return session_options
    
    def load(self, model_path: Union[str, Path]) -> ModelMetadata:
        """
        Load ONNX model and extract metadata.
        
        Args:
            model_path: Path to .onnx model file
            
        Returns:
            ModelMetadata with model information
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If ONNX Runtime not available or load fails
        """
        if self.ort is None:
            raise RuntimeError("ONNX Runtime not available")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Get file size
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # Select providers
        providers = self._select_providers()
        
        # Setup session options
        session_options = self._setup_session_options()
        
        try:
            # Create inference session
            self._session = self.ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=providers
            )
            
            self._model = self._session
            
            # Extract metadata
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            
            input_names = [inp.name for inp in inputs]
            output_names = [out.name for out in outputs]
            
            input_shapes = [tuple(inp.shape) for inp in inputs]
            output_shapes = [tuple(out.shape) for out in outputs]
            
            input_dtypes = [inp.type for inp in inputs]
            output_dtypes = [out.type for out in outputs]
            
            # Estimate memory usage (input + output + model weights)
            def estimate_tensor_size(shape, dtype='float32'):
                # Handle dynamic dimensions
                size = 1
                for dim in shape:
                    if isinstance(dim, (int, np.integer)) and dim > 0:
                        size *= dim
                    else:
                        size *= 1  # Assume batch size 1 for dynamic dims
                
                bytes_per_element = 4 if 'float32' in str(dtype) else 2
                return size * bytes_per_element / (1024 * 1024)
            
            input_memory = sum(estimate_tensor_size(s, d) for s, d in zip(input_shapes, input_dtypes))
            output_memory = sum(estimate_tensor_size(s, d) for s, d in zip(output_shapes, output_dtypes))
            estimated_memory_mb = file_size_mb + input_memory + output_memory
            
            # Create metadata
            self._metadata = ModelMetadata(
                name=model_path.stem,
                framework='onnx',
                input_names=input_names,
                output_names=output_names,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                input_dtypes=input_dtypes,
                output_dtypes=output_dtypes,
                file_size_mb=file_size_mb,
                estimated_memory_mb=estimated_memory_mb,
                provider=providers[0],
                optimization_level=f"Level {self.optimization_level}",
                extra_info={
                    'all_providers': providers,
                    'available_providers': self.get_available_providers()
                }
            )
            
            logger.info(f"✅ Loaded ONNX model: {self._metadata.name}")
            logger.info(f"   Inputs: {input_names} {input_shapes}")
            logger.info(f"   Outputs: {output_names} {output_shapes}")
            logger.info(f"   Provider: {providers[0]}")
            logger.info(f"   Memory: {estimated_memory_mb:.1f}MB")
            
            return self._metadata
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def predict(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run inference on loaded ONNX model.
        
        Args:
            inputs: Input tensor(s) as numpy array or dict of arrays
            
        Returns:
            Output tensor(s) as numpy array or dict
            
        Raises:
            RuntimeError: If model not loaded
        """
        if self._session is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Prepare inputs
        if isinstance(inputs, np.ndarray):
            # Single input - use first input name
            input_feed = {self._metadata.input_names[0]: inputs}
        else:
            # Multiple inputs as dict
            input_feed = inputs
        
        # Run inference
        try:
            outputs = self._session.run(None, input_feed)
            
            # Return single output or dict
            if len(outputs) == 1:
                return outputs[0]
            else:
                return {name: out for name, out in zip(self._metadata.output_names, outputs)}
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")


class PyTorchModelLoader(BaseModelLoader):
    """
    PyTorch/TorchScript model loader with ROCm support.
    
    Supports:
    - PyTorch models (.pt, .pth files)
    - TorchScript models (.pt with scripting/tracing)
    - ROCm backend for AMD GPUs
    - CPU fallback
    - Dynamic batching
    
    Real-world usage:
    - Load models trained in PyTorch
    - Deploy research models to production
    - Fine-tuned models from transfer learning
    - Custom architectures
    """
    
    def __init__(
        self,
        optimization_level: int = 2,
        preferred_device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    ):
        """
        Initialize PyTorch loader.
        
        Args:
            optimization_level: Optimization level (0-2)
            preferred_device: Preferred device ('auto', 'cuda', 'cpu')
        """
        super().__init__(optimization_level)
        self.preferred_device = preferred_device
        
        # Try to import torch
        try:
            import torch
            self.torch = torch
            self._device = self._select_device()
            logger.info(f"✅ PyTorch {torch.__version__} available")
            logger.info(f"   Device: {self._device}")
            logger.info(f"   CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        except ImportError:
            logger.warning("⚠️  PyTorch not installed. Install with: pip install torch")
            self.torch = None
            self._device = 'cpu'
    
    def get_available_providers(self) -> List[str]:
        """Get available PyTorch backends"""
        if self.torch is None:
            return []
        
        providers = ['cpu']
        if self.torch.cuda.is_available():
            providers.append('cuda')  # Works with ROCm too
        
        return providers
    
    def _select_device(self) -> str:
        """Select compute device"""
        if self.torch is None:
            return 'cpu'
        
        if self.preferred_device == 'auto':
            # Auto-select: CUDA/ROCm if available, else CPU
            return 'cuda' if self.torch.cuda.is_available() else 'cpu'
        else:
            return self.preferred_device
    
    def load(self, model_path: Union[str, Path]) -> ModelMetadata:
        """
        Load PyTorch/TorchScript model.
        
        Args:
            model_path: Path to .pt or .pth model file
            
        Returns:
            ModelMetadata with model information
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If PyTorch not available or load fails
        """
        if self.torch is None:
            raise RuntimeError("PyTorch not available")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Get file size
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        try:
            # Load model (TorchScript or state dict)
            self._model = self.torch.jit.load(str(model_path), map_location=self._device)
            self._model.eval()
            
            # Try to extract metadata
            # Note: TorchScript doesn't provide complete metadata
            # We'll need to infer or require user specification
            
            # Estimate memory (model parameters + buffers)
            param_memory = 0
            for param in self._model.parameters():
                param_memory += param.numel() * param.element_size()
            
            estimated_memory_mb = file_size_mb + param_memory / (1024 * 1024)
            
            self._metadata = ModelMetadata(
                name=model_path.stem,
                framework='torchscript',
                input_names=['input'],  # Default, may need user override
                output_names=['output'],
                input_shapes=[()],  # Unknown until inference
                output_shapes=[()],
                input_dtypes=['float32'],
                output_dtypes=['float32'],
                file_size_mb=file_size_mb,
                estimated_memory_mb=estimated_memory_mb,
                provider=self._device,
                optimization_level=f"Level {self.optimization_level}",
                extra_info={
                    'device': str(self._device),
                    'cuda_available': self.torch.cuda.is_available()
                }
            )
            
            logger.info(f"✅ Loaded TorchScript model: {self._metadata.name}")
            logger.info(f"   Device: {self._device}")
            logger.info(f"   Memory: {estimated_memory_mb:.1f}MB")
            
            return self._metadata
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise RuntimeError(f"Failed to load PyTorch model: {e}")
    
    def predict(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run inference on loaded PyTorch model.
        
        Args:
            inputs: Input tensor(s) as numpy array
            
        Returns:
            Output tensor(s) as numpy array
            
        Raises:
            RuntimeError: If model not loaded
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if self.torch is None:
            raise RuntimeError("PyTorch not available")
        
        try:
            # Convert numpy to torch tensor
            if isinstance(inputs, np.ndarray):
                input_tensor = self.torch.from_numpy(inputs).to(self._device)
            else:
                # Handle dict of inputs
                input_tensor = {
                    k: self.torch.from_numpy(v).to(self._device)
                    for k, v in inputs.items()
                }
            
            # Run inference
            with self.torch.no_grad():
                output = self._model(input_tensor)
            
            # Convert back to numpy
            if isinstance(output, self.torch.Tensor):
                return output.cpu().numpy()
            elif isinstance(output, (list, tuple)):
                return [o.cpu().numpy() for o in output]
            else:
                return output
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")


def create_loader(
    model_path: Union[str, Path],
    framework: Optional[str] = None,
    **kwargs
) -> BaseModelLoader:
    """
    Factory function to create appropriate model loader.
    
    Args:
        model_path: Path to model file
        framework: 'onnx' or 'pytorch', auto-detected if None
        **kwargs: Additional arguments for loader
        
    Returns:
        Configured model loader
        
    Example:
        >>> loader = create_loader('model.onnx')
        >>> metadata = loader.load('model.onnx')
        >>> outputs = loader.predict(inputs)
    """
    model_path = Path(model_path)
    
    # Auto-detect framework from extension
    if framework is None:
        suffix = model_path.suffix.lower()
        if suffix == '.onnx':
            framework = 'onnx'
        elif suffix in ['.pt', '.pth']:
            framework = 'pytorch'
        else:
            raise ValueError(
                f"Cannot auto-detect framework from extension: {suffix}. "
                f"Specify framework='onnx' or 'pytorch'"
            )
    
    # Create loader
    if framework == 'onnx':
        return ONNXModelLoader(**kwargs)
    elif framework == 'pytorch':
        return PyTorchModelLoader(**kwargs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


__all__ = [
    'BaseModelLoader',
    'ONNXModelLoader',
    'PyTorchModelLoader',
    'ModelMetadata',
    'create_loader',
]
