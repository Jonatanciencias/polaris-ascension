"""
Radeon RX 580 AI - Inference Module

This module provides inference capabilities for AI models optimized for AMD GPUs.
Supports ONNX and PyTorch models with automatic hardware optimization.

Session 15 Enhancement: Integrated compute primitives for production deployment.
Session 16 Enhancement: Real ONNX/PyTorch model integration with hardware-aware loading.
"""

from .base import BaseInferenceEngine, InferenceConfig, ModelInfo
from .onnx_engine import ONNXInferenceEngine

# Session 16: Real model loaders
from .model_loaders import (
    BaseModelLoader,
    ONNXModelLoader,
    PyTorchModelLoader,
    ModelMetadata,
    create_loader
)

# Session 15: Enhanced inference
from .enhanced import (
    EnhancedInferenceEngine,
    ModelCompressor,
    AdaptiveBatchScheduler,
    MultiModelServer,
    CompressionStrategy,
    CompressionConfig,
    CompressionResult,
    BatchRequest,
    BatchResponse,
    ModelStats,
)

__all__ = [
    # Base components
    'BaseInferenceEngine',
    'InferenceConfig',
    'ModelInfo',
    'ONNXInferenceEngine',
    # Session 16: Model loaders
    'BaseModelLoader',
    'ONNXModelLoader',
    'PyTorchModelLoader',
    'ModelMetadata',
    'create_loader',
    # Session 15: Enhanced components
    'EnhancedInferenceEngine',
    'ModelCompressor',
    'AdaptiveBatchScheduler',
    'MultiModelServer',
    'CompressionStrategy',
    'CompressionConfig',
    'CompressionResult',
    'BatchRequest',
    'BatchResponse',
    'ModelStats',
]
