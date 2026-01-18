"""
Radeon RX 580 AI - Inference Module

This module provides inference capabilities for AI models optimized for AMD GPUs.
Supports ONNX and PyTorch models with automatic hardware optimization.

Session 15 Enhancement: Integrated compute primitives for production deployment.
"""

from .base import BaseInferenceEngine, InferenceConfig, ModelInfo
from .onnx_engine import ONNXInferenceEngine
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
    # Enhanced components (Session 15)
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
