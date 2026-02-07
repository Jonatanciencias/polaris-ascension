"""
ONNX inference engine compatibility implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import time

import numpy as np
from PIL import Image

from ..core.gpu import GPUManager
from ..core.memory import MemoryManager
from ..core.profiler import Profiler
from .base import InferenceConfig

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    ort = None  # type: ignore[assignment]


ArrayLikeInput = Union[str, Path, np.ndarray, Image.Image]


@dataclass
class ModelInfo:
    """Loaded model metadata."""

    model_path: str
    backend: str
    provider: str
    input_name: str
    input_shape: Sequence[Any]
    output_name: str
    output_shape: Sequence[Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_softmax(values: np.ndarray) -> np.ndarray:
    max_v = np.max(values)
    exp_v = np.exp(values - max_v)
    denom = np.sum(exp_v)
    if denom <= 0:
        return np.full_like(values, 1.0 / values.size)
    return exp_v / denom


class ONNXInferenceEngine:
    """ONNX runtime-backed inference engine with legacy-compatible API."""

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        gpu_manager: Optional[GPUManager] = None,
        memory_manager: Optional[MemoryManager] = None,
    ) -> None:
        self.config = config or InferenceConfig()
        self.gpu_manager = gpu_manager or GPUManager()
        self.memory_manager = memory_manager or MemoryManager()
        self.profiler = Profiler(enabled=self.config.enable_profiling)

        self._session: Optional["ort.InferenceSession"] = None
        self._model_info: Optional[ModelInfo] = None

    def _select_providers(self) -> List[str]:
        if not HAS_ORT:
            return []
        available = ort.get_available_providers()
        device = (self.config.device or "auto").lower()

        if device == "cpu":
            return ["CPUExecutionProvider"]
        if device in {"gpu", "auto"}:
            preferred = [
                "ROCMExecutionProvider",
                "CUDAExecutionProvider",
                "OpenVINOExecutionProvider",
                "CPUExecutionProvider",
            ]
            selected = [p for p in preferred if p in available]
            return selected or ["CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def load_model(self, model_path: Union[str, Path]) -> ModelInfo:
        """Load an ONNX model and return metadata."""
        if not HAS_ORT:
            raise RuntimeError("onnxruntime is required for ONNX inference.")

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        providers = self._select_providers()
        sess_opts = ort.SessionOptions()
        if self.config.optimization_level >= 2:
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        self._session = ort.InferenceSession(str(path), providers=providers, sess_options=sess_opts)
        inp = self._session.get_inputs()[0]
        out = self._session.get_outputs()[0]
        provider = self._session.get_providers()[0] if self._session.get_providers() else "CPUExecutionProvider"

        self._model_info = ModelInfo(
            model_path=str(path),
            backend="onnxruntime",
            provider=provider,
            input_name=inp.name,
            input_shape=inp.shape,
            output_name=out.name,
            output_shape=out.shape,
        )
        return self._model_info

    def get_model_info(self) -> Dict[str, Any]:
        return self._model_info.to_dict() if self._model_info else {}

    def _require_model(self) -> None:
        if self._session is None:
            if self.config.model_path:
                self.load_model(self.config.model_path)
            else:
                raise RuntimeError("No model loaded. Call load_model() first.")

    def _input_hw_channels(self) -> tuple[int, int, str]:
        """
        Infer input H/W layout.
        Returns:
            (height, width, layout) where layout is 'nchw' or 'nhwc'.
        """
        if self._model_info is None:
            return 224, 224, "nchw"

        shape = list(self._model_info.input_shape)
        if len(shape) != 4:
            return 224, 224, "nchw"

        # Common NCHW: [N, 3, H, W]
        c = shape[1]
        if isinstance(c, int) and c in (1, 3, 4):
            h = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 224
            w = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else 224
            return int(h), int(w), "nchw"

        # NHWC: [N, H, W, 3]
        c_last = shape[3]
        if isinstance(c_last, int) and c_last in (1, 3, 4):
            h = shape[1] if isinstance(shape[1], int) and shape[1] > 0 else 224
            w = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 224
            return int(h), int(w), "nhwc"

        return 224, 224, "nchw"

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        h, w, layout = self._input_hw_channels()
        image = image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0

        if layout == "nhwc":
            return np.expand_dims(arr, axis=0).astype(np.float32)

        # Default NCHW
        chw = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def _prepare_input(self, data: ArrayLikeInput) -> np.ndarray:
        if isinstance(data, np.ndarray):
            arr = data.astype(np.float32, copy=False)
            if arr.ndim == 3:
                return np.expand_dims(arr, axis=0)
            return arr

        if isinstance(data, Path):
            image = Image.open(data)
            return self._preprocess_image(image)

        if isinstance(data, str):
            image = Image.open(Path(data))
            return self._preprocess_image(image)

        if isinstance(data, Image.Image):
            return self._preprocess_image(data)

        raise TypeError(f"Unsupported inference input type: {type(data)}")

    def _format_predictions(self, output: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        vec = output
        if vec.ndim > 1:
            vec = vec[0]
        if vec.ndim != 1:
            vec = vec.reshape(-1)

        probs = _safe_softmax(vec.astype(np.float64))
        k = max(1, min(int(top_k), int(probs.size)))
        top_idx = np.argsort(probs)[::-1][:k]

        predictions = [
            {"class_id": int(i), "confidence": float(probs[i])}
            for i in top_idx
        ]
        top1 = predictions[0]
        return {
            "top1_class": top1["class_id"],
            "top1_confidence": top1["confidence"],
            "predictions": predictions,
        }

    def infer(self, data: ArrayLikeInput, profile: bool = False, top_k: int = 5) -> Dict[str, Any]:
        """Run one inference and return predictions + metadata."""
        self._require_model()
        assert self._session is not None
        assert self._model_info is not None

        input_tensor = self._prepare_input(data)
        start = time.perf_counter()
        outputs = self._session.run(
            [self._model_info.output_name],
            {self._model_info.input_name: input_tensor},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if self.config.enable_profiling:
            self.profiler._samples.setdefault("inference", []).append(elapsed_ms)  # noqa: SLF001

        result = self._format_predictions(outputs[0], top_k=top_k)
        result["time_ms"] = float(elapsed_ms)
        if profile:
            result["profile"] = {"inference_ms": float(elapsed_ms)}
        return result

    def infer_batch(
        self,
        data_items: Sequence[ArrayLikeInput],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batch inference helper.

        Note: current implementation executes per-sample for broad model compatibility.
        """
        _ = batch_size or self.config.batch_size
        return [self.infer(item) for item in data_items]

    def batch_infer(
        self,
        data_items: Sequence[ArrayLikeInput],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Alias for legacy APIs."""
        return self.infer_batch(data_items, batch_size=batch_size)

    def get_optimization_info(self) -> Dict[str, Any]:
        precision = self.config.normalized_precision()
        if precision == "int8":
            return {
                "precision": "int8",
                "batch_size": self.config.batch_size,
                "expected_speedup": "~2.5x (workload dependent)",
                "memory_savings": "~75%",
                "accuracy": "High, model dependent",
            }
        if precision == "fp16":
            return {
                "precision": "fp16",
                "batch_size": self.config.batch_size,
                "expected_speedup": "~1.5x (workload dependent)",
                "memory_savings": "~50%",
                "accuracy": "Very high",
            }
        return {
            "precision": "fp32",
            "batch_size": self.config.batch_size,
            "expected_speedup": "baseline",
            "memory_savings": "none",
            "accuracy": "maximum",
        }

    def print_performance_stats(self) -> None:
        stats = self.profiler.get_statistics()
        print("Performance Stats")
        print(f"  Calls: {int(stats['count'])}")
        print(f"  Mean: {stats['mean']:.2f} ms")
        print(f"  Min: {stats['min']:.2f} ms")
        print(f"  Max: {stats['max']:.2f} ms")
        print(f"  Total: {stats['total']:.2f} ms")


class ONNXEngine(ONNXInferenceEngine):
    """Compatibility alias used by older examples."""

