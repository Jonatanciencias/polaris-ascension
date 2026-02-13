"""
Enhanced inference compatibility shims.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional

from .base import InferenceConfig
from .onnx_engine import ONNXInferenceEngine


class EnhancedInferenceEngine(ONNXInferenceEngine):
    """Compatibility alias for enhanced engine use-cases."""

    def get_stats(self) -> Dict[str, float]:
        return self.profiler.get_statistics()


class MultiModelServer:
    """
    Minimal multi-model server.

    Keeps loaded engines in memory and evicts old ones when max_models is reached.
    """

    def __init__(self, max_models: int = 5, memory_limit_mb: float = 4096.0) -> None:
        self.max_models = int(max_models)
        self.memory_limit_mb = float(memory_limit_mb)
        self._engines: "OrderedDict[str, ONNXInferenceEngine]" = OrderedDict()

    def load_model(
        self, model_name: str, model_path: str, config: Optional[InferenceConfig] = None
    ) -> None:
        if model_name in self._engines:
            self._engines.move_to_end(model_name)
            return

        while len(self._engines) >= self.max_models:
            self._engines.popitem(last=False)

        engine = ONNXInferenceEngine(config=config or InferenceConfig())
        engine.load_model(Path(model_path))
        self._engines[model_name] = engine

    def infer(self, model_name: str, data):
        if model_name not in self._engines:
            raise KeyError(f"Model not loaded: {model_name}")
        self._engines.move_to_end(model_name)
        return self._engines[model_name].infer(data)

    def unload_model(self, model_name: str) -> bool:
        return self._engines.pop(model_name, None) is not None

    def get_stats(self) -> Dict[str, float]:
        return {
            "loaded_models": float(len(self._engines)),
            "max_models": float(self.max_models),
            "memory_limit_mb": float(self.memory_limit_mb),
        }
