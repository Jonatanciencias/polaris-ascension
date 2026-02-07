"""
Base inference configuration types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceConfig:
    """Runtime configuration for inference engines."""

    device: str = "auto"
    precision: str = "fp32"
    batch_size: int = 1
    enable_profiling: bool = False
    optimization_level: int = 2
    model_path: Optional[str] = None

    def normalized_precision(self) -> str:
        value = (self.precision or "fp32").lower()
        if value in {"fp32", "float32"}:
            return "fp32"
        if value in {"fp16", "float16", "half"}:
            return "fp16"
        if value in {"int8", "i8"}:
            return "int8"
        return "fp32"
