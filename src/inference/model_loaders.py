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

__version__ = "0.7.0-dev"  # Session 19: Added TFLite, JAX, GGUF loaders

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import logging
import struct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata extracted from loaded model"""
    name: str
    framework: str  # 'onnx', 'pytorch', 'torchscript', 'tflite', 'jax', 'gguf'
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


class TFLiteModelLoader(BaseModelLoader):
    """
    TensorFlow Lite model loader for mobile/edge deployment.
    
    Supports:
    - TensorFlow Lite models (.tflite files)
    - Quantized models (INT8, FP16)
    - Optimized for mobile and edge devices
    - CPU execution with XNNPACK delegate
    
    Real-world usage:
    - Mobile applications (Android/iOS)
    - Edge devices (Raspberry Pi, Coral)
    - IoT deployments
    - Real-time inference on resource-constrained devices
    """
    
    def __init__(
        self,
        optimization_level: int = 2,
        use_xnnpack: bool = True,
        num_threads: int = 4
    ):
        """
        Initialize TFLite loader.
        
        Args:
            optimization_level: Optimization level (0-2)
            use_xnnpack: Use XNNPACK delegate for CPU acceleration
            num_threads: Number of threads for inference
        """
        super().__init__(optimization_level)
        self.use_xnnpack = use_xnnpack
        self.num_threads = num_threads
        self._interpreter = None
        
        # Try to import tensorflow lite
        try:
            import tensorflow as tf
            self.tf = tf
            logger.info(f"✅ TensorFlow {tf.__version__} available")
            logger.info(f"   TFLite interpreter ready")
            logger.info(f"   XNNPACK delegate: {use_xnnpack}")
        except ImportError:
            logger.warning("⚠️  TensorFlow not installed. Install with: pip install tensorflow")
            self.tf = None
    
    def get_available_providers(self) -> List[str]:
        """Get available TFLite execution providers"""
        providers = ['CPU']
        if self.use_xnnpack:
            providers.append('XNNPACK')
        return providers
    
    def load(self, model_path: Union[str, Path]) -> ModelMetadata:
        """
        Load TFLite model and extract metadata.
        
        Args:
            model_path: Path to .tflite model file
            
        Returns:
            ModelMetadata with model information
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If TensorFlow not available or load fails
        """
        if self.tf is None:
            raise RuntimeError("TensorFlow not available")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Get file size
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        try:
            # Load TFLite model
            self._interpreter = self.tf.lite.Interpreter(
                model_path=str(model_path),
                num_threads=self.num_threads
            )
            
            # Allocate tensors
            self._interpreter.allocate_tensors()
            
            # Get input/output details
            input_details = self._interpreter.get_input_details()
            output_details = self._interpreter.get_output_details()
            
            input_names = [inp['name'] for inp in input_details]
            output_names = [out['name'] for out in output_details]
            
            input_shapes = [tuple(inp['shape']) for inp in input_details]
            output_shapes = [tuple(out['shape']) for out in output_details]
            
            # Map TFLite dtype to string
            def dtype_to_str(dtype_code):
                dtype_map = {
                    0: 'float32',  # FLOAT32
                    1: 'float16',  # FLOAT16
                    2: 'int32',    # INT32
                    3: 'uint8',    # UINT8
                    9: 'int8',     # INT8
                }
                return dtype_map.get(dtype_code, f'unknown({dtype_code})')
            
            input_dtypes = [dtype_to_str(inp['dtype']) for inp in input_details]
            output_dtypes = [dtype_to_str(out['dtype']) for out in output_details]
            
            # Estimate memory usage
            def estimate_tensor_size(shape, dtype):
                size = 1
                for dim in shape:
                    if isinstance(dim, (int, np.integer)) and dim > 0:
                        size *= dim
                    else:
                        size *= 1  # Dynamic dimension
                
                bytes_per_elem = {
                    'float32': 4, 'float16': 2, 'int32': 4,
                    'int16': 2, 'int8': 1, 'uint8': 1
                }.get(dtype, 4)
                
                return size * bytes_per_elem / (1024 * 1024)
            
            input_memory = sum(estimate_tensor_size(s, d) 
                             for s, d in zip(input_shapes, input_dtypes))
            output_memory = sum(estimate_tensor_size(s, d) 
                              for s, d in zip(output_shapes, output_dtypes))
            
            estimated_memory_mb = file_size_mb + input_memory + output_memory
            
            # Detect quantization
            is_quantized = any('int8' in dt or 'uint8' in dt 
                             for dt in input_dtypes + output_dtypes)
            
            self._metadata = ModelMetadata(
                name=model_path.stem,
                framework='tflite',
                input_names=input_names,
                output_names=output_names,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                input_dtypes=input_dtypes,
                output_dtypes=output_dtypes,
                file_size_mb=file_size_mb,
                estimated_memory_mb=estimated_memory_mb,
                provider='XNNPACK' if self.use_xnnpack else 'CPU',
                optimization_level=f"Level {self.optimization_level}",
                extra_info={
                    'quantized': is_quantized,
                    'num_threads': self.num_threads,
                    'tflite_version': self.tf.__version__
                }
            )
            
            logger.info(f"✅ Loaded TFLite model: {self._metadata.name}")
            logger.info(f"   Inputs: {input_names} {input_shapes}")
            logger.info(f"   Outputs: {output_names} {output_shapes}")
            logger.info(f"   Quantized: {is_quantized}")
            logger.info(f"   Memory: {estimated_memory_mb:.1f}MB")
            
            return self._metadata
            
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            raise RuntimeError(f"Failed to load TFLite model: {e}")
    
    def predict(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run inference on loaded TFLite model.
        
        Args:
            inputs: Input tensor(s) as numpy array
            
        Returns:
            Output tensor(s) as numpy array
            
        Raises:
            RuntimeError: If model not loaded
        """
        if self._interpreter is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if self.tf is None:
            raise RuntimeError("TensorFlow not available")
        
        try:
            # Set input tensor(s)
            input_details = self._interpreter.get_input_details()
            
            if isinstance(inputs, np.ndarray):
                # Single input
                self._interpreter.set_tensor(input_details[0]['index'], inputs)
            else:
                # Multiple inputs (dict)
                for i, (name, data) in enumerate(inputs.items()):
                    self._interpreter.set_tensor(input_details[i]['index'], data)
            
            # Run inference
            self._interpreter.invoke()
            
            # Get output tensor(s)
            output_details = self._interpreter.get_output_details()
            
            if len(output_details) == 1:
                # Single output
                return self._interpreter.get_tensor(output_details[0]['index'])
            else:
                # Multiple outputs
                return {
                    out['name']: self._interpreter.get_tensor(out['index'])
                    for out in output_details
                }
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")


class JAXModelLoader(BaseModelLoader):
    """
    JAX/Flax model loader for high-performance research.
    
    Supports:
    - JAX models (pytree format)
    - Flax models (checkpoints)
    - XLA compilation for performance
    - CPU/GPU execution
    
    Real-world usage:
    - Research models (Flax, Haiku, Equinox)
    - High-performance training
    - Functional ML research
    - Custom architectures
    """
    
    def __init__(
        self,
        optimization_level: int = 2,
        preferred_device: str = 'auto'  # 'auto', 'gpu', 'cpu'
    ):
        """
        Initialize JAX loader.
        
        Args:
            optimization_level: Optimization level (0-2)
            preferred_device: Preferred device ('auto', 'gpu', 'cpu')
        """
        super().__init__(optimization_level)
        self.preferred_device = preferred_device
        self._params = None
        self._apply_fn = None
        
        # Try to import JAX
        try:
            import jax
            import jax.numpy as jnp
            self.jax = jax
            self.jnp = jnp
            
            # Try to import Flax
            try:
                import flax
                self.flax = flax
                logger.info(f"✅ JAX {jax.__version__} available")
                logger.info(f"✅ Flax {flax.__version__} available")
            except ImportError:
                self.flax = None
                logger.info(f"✅ JAX {jax.__version__} available (Flax not installed)")
            
            # Get available devices
            devices = jax.devices()
            logger.info(f"   Available devices: {[str(d) for d in devices]}")
            
        except ImportError:
            logger.warning("⚠️  JAX not installed. Install with: pip install jax jaxlib")
            self.jax = None
            self.jnp = None
            self.flax = None
    
    def get_available_providers(self) -> List[str]:
        """Get available JAX backends"""
        if self.jax is None:
            return []
        
        providers = []
        for device in self.jax.devices():
            providers.append(str(device.platform).upper())
        
        return list(set(providers))  # Remove duplicates
    
    def load(self, model_path: Union[str, Path]) -> ModelMetadata:
        """
        Load JAX/Flax model (pytree or checkpoint).
        
        Args:
            model_path: Path to model file (pickle, msgpack, or checkpoint)
            
        Returns:
            ModelMetadata with model information
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If JAX not available or load fails
        """
        if self.jax is None:
            raise RuntimeError("JAX not available")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Get file size
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        try:
            # Try to load as pytree (pickle format)
            import pickle
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract parameters and apply function if available
            if isinstance(data, dict):
                if 'params' in data:
                    self._params = data['params']
                    self._apply_fn = data.get('apply_fn', None)
                else:
                    self._params = data
            else:
                self._params = data
            
            # Count parameters
            def count_params(pytree):
                if isinstance(pytree, dict):
                    return sum(count_params(v) for v in pytree.values())
                elif isinstance(pytree, (list, tuple)):
                    return sum(count_params(v) for v in pytree)
                elif hasattr(pytree, 'shape'):
                    return int(np.prod(pytree.shape))
                else:
                    return 0
            
            num_params = count_params(self._params)
            param_memory_mb = num_params * 4 / (1024 * 1024)  # Assume float32
            
            estimated_memory_mb = file_size_mb + param_memory_mb
            
            # Get available device
            device = self.jax.devices()[0]
            
            self._metadata = ModelMetadata(
                name=model_path.stem,
                framework='jax',
                input_names=['input'],  # Generic, needs user specification
                output_names=['output'],
                input_shapes=[()],  # Unknown until inference
                output_shapes=[()],
                input_dtypes=['float32'],
                output_dtypes=['float32'],
                file_size_mb=file_size_mb,
                estimated_memory_mb=estimated_memory_mb,
                provider=str(device.platform).upper(),
                optimization_level=f"Level {self.optimization_level}",
                extra_info={
                    'num_params': num_params,
                    'has_apply_fn': self._apply_fn is not None,
                    'jax_version': self.jax.__version__,
                    'device': str(device)
                }
            )
            
            logger.info(f"✅ Loaded JAX model: {self._metadata.name}")
            logger.info(f"   Parameters: {num_params:,}")
            logger.info(f"   Memory: {estimated_memory_mb:.1f}MB")
            logger.info(f"   Device: {device}")
            
            return self._metadata
            
        except Exception as e:
            logger.error(f"Failed to load JAX model: {e}")
            raise RuntimeError(f"Failed to load JAX model: {e}")
    
    def predict(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run inference on loaded JAX model.
        
        Args:
            inputs: Input tensor(s) as numpy array
            
        Returns:
            Output tensor(s) as numpy array
            
        Note:
            Requires apply_fn to be provided during loading or set manually.
        """
        if self._params is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if self._apply_fn is None:
            raise RuntimeError(
                "No apply function available. "
                "Set loader._apply_fn manually for custom models."
            )
        
        if self.jax is None:
            raise RuntimeError("JAX not available")
        
        try:
            # Convert to JAX array
            if isinstance(inputs, np.ndarray):
                jax_inputs = self.jnp.array(inputs)
            else:
                jax_inputs = {k: self.jnp.array(v) for k, v in inputs.items()}
            
            # Run inference
            output = self._apply_fn(self._params, jax_inputs)
            
            # Convert back to numpy
            if isinstance(output, dict):
                return {k: np.array(v) for k, v in output.items()}
            else:
                return np.array(output)
                
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}")


class GGUFModelLoader(BaseModelLoader):
    """
    GGUF model loader for quantized LLMs (llama.cpp format).
    
    Supports:
    - GGUF format (.gguf files)
    - Quantized models (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
    - Large language models (Llama 2/3, Mistral, etc.)
    - CPU inference with AVX2/AVX512
    
    Real-world usage:
    - Running LLMs locally (Llama 2 7B, 13B)
    - Low-memory inference (4-8GB VRAM)
    - Text generation on consumer hardware
    - Fine-tuned models (LoRA, QLoRA)
    """
    
    def __init__(
        self,
        optimization_level: int = 2,
        n_threads: int = 4,
        use_mmap: bool = True
    ):
        """
        Initialize GGUF loader.
        
        Args:
            optimization_level: Optimization level (0-2)
            n_threads: Number of threads for inference
            use_mmap: Use memory mapping for large models
        """
        super().__init__(optimization_level)
        self.n_threads = n_threads
        self.use_mmap = use_mmap
        self._model_data = None
        self._metadata_dict = {}
        
        logger.info(f"✅ GGUF loader initialized")
        logger.info(f"   Threads: {n_threads}")
        logger.info(f"   Memory mapping: {use_mmap}")
    
    def get_available_providers(self) -> List[str]:
        """Get available GGUF execution providers"""
        return ['CPU', 'AVX2', 'AVX512']  # GGUF is primarily CPU-based
    
    def _read_gguf_header(self, file_path: Path) -> Dict[str, Any]:
        """Parse GGUF file header for metadata"""
        with open(file_path, 'rb') as f:
            # Read magic number (4 bytes)
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError(f"Invalid GGUF file: magic={magic}")
            
            # Read version (4 bytes)
            version = struct.unpack('<I', f.read(4))[0]
            
            # Read tensor count (8 bytes)
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            
            # Read metadata count (8 bytes)
            metadata_count = struct.unpack('<Q', f.read(8))[0]
            
            return {
                'version': version,
                'tensor_count': tensor_count,
                'metadata_count': metadata_count
            }
    
    def load(self, model_path: Union[str, Path]) -> ModelMetadata:
        """
        Load GGUF model and extract metadata.
        
        Args:
            model_path: Path to .gguf model file
            
        Returns:
            ModelMetadata with model information
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If load fails
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Get file size
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        try:
            # Parse GGUF header
            header = self._read_gguf_header(model_path)
            
            # Store model path for later use (actual loading would use llama-cpp-python)
            self._model_data = {
                'path': model_path,
                'header': header
            }
            
            # Estimate memory (GGUF models are memory-mapped, so RAM usage is lower)
            estimated_memory_mb = file_size_mb * 0.3  # Only ~30% loaded in RAM
            
            # Detect quantization type from filename
            quant_type = 'unknown'
            for qtype in ['Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'F16', 'F32']:
                if qtype.lower() in model_path.name.lower():
                    quant_type = qtype
                    break
            
            self._metadata = ModelMetadata(
                name=model_path.stem,
                framework='gguf',
                input_names=['input_ids'],
                output_names=['logits'],
                input_shapes=[(-1, -1)],  # Dynamic sequence length
                output_shapes=[(-1, -1, -1)],  # (batch, seq_len, vocab_size)
                input_dtypes=['int32'],
                output_dtypes=['float32'],
                file_size_mb=file_size_mb,
                estimated_memory_mb=estimated_memory_mb,
                provider='CPU',
                optimization_level=f"Level {self.optimization_level}",
                extra_info={
                    'gguf_version': header['version'],
                    'tensor_count': header['tensor_count'],
                    'metadata_count': header['metadata_count'],
                    'quantization': quant_type,
                    'n_threads': self.n_threads,
                    'use_mmap': self.use_mmap
                }
            )
            
            logger.info(f"✅ Loaded GGUF model: {self._metadata.name}")
            logger.info(f"   Version: {header['version']}")
            logger.info(f"   Tensors: {header['tensor_count']}")
            logger.info(f"   Quantization: {quant_type}")
            logger.info(f"   File size: {file_size_mb:.1f}MB")
            logger.info(f"   Est. memory: {estimated_memory_mb:.1f}MB (mmap)")
            
            return self._metadata
            
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            raise RuntimeError(f"Failed to load GGUF model: {e}")
    
    def predict(
        self,
        inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run inference on loaded GGUF model.
        
        Args:
            inputs: Input token IDs as numpy array
            
        Returns:
            Output logits as numpy array
            
        Note:
            Full inference requires llama-cpp-python library.
            This is a placeholder implementation.
        """
        if self._model_data is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        logger.warning(
            "⚠️  GGUF inference requires llama-cpp-python library. "
            "Install with: pip install llama-cpp-python"
        )
        
        # Placeholder: return dummy logits
        if isinstance(inputs, np.ndarray):
            batch_size, seq_len = inputs.shape[:2]
            vocab_size = 32000  # Typical for Llama models
            return np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)
        else:
            raise NotImplementedError("Dict inputs not supported for GGUF")


def create_loader(
    model_path: Union[str, Path],
    framework: Optional[str] = None,
    **kwargs
) -> BaseModelLoader:
    """
    Factory function to create appropriate model loader.
    
    Args:
        model_path: Path to model file
        framework: 'onnx', 'pytorch', 'tflite', 'jax', or 'gguf', auto-detected if None
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
        elif suffix == '.tflite':
            framework = 'tflite'
        elif suffix == '.gguf':
            framework = 'gguf'
        elif suffix in ['.pkl', '.pickle', '.msgpack']:
            framework = 'jax'  # Assume JAX for pickle/msgpack
        else:
            raise ValueError(
                f"Cannot auto-detect framework from extension: {suffix}. "
                f"Specify framework='onnx', 'pytorch', 'tflite', 'jax', or 'gguf'"
            )
    
    # Create loader
    if framework == 'onnx':
        return ONNXModelLoader(**kwargs)
    elif framework == 'pytorch':
        return PyTorchModelLoader(**kwargs)
    elif framework == 'tflite':
        return TFLiteModelLoader(**kwargs)
    elif framework == 'jax':
        return JAXModelLoader(**kwargs)
    elif framework == 'gguf':
        return GGUFModelLoader(**kwargs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


__all__ = [
    'BaseModelLoader',
    'ONNXModelLoader',
    'PyTorchModelLoader',
    'TFLiteModelLoader',
    'JAXModelLoader',
    'GGUFModelLoader',
    'ModelMetadata',
    'create_loader',
]
