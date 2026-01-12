"""
Radeon RX 580 AI - Inference Module

This module provides inference capabilities for AI models optimized for AMD GPUs.
Supports ONNX and PyTorch models with automatic hardware optimization.
"""

from .base import BaseInferenceEngine, InferenceConfig, ModelInfo
from .onnx_engine import ONNXInferenceEngine

__all__ = [
    'BaseInferenceEngine',
    'InferenceConfig',
    'ModelInfo',
    'ONNXInferenceEngine',
]
