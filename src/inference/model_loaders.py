"""
Model loader compatibility module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union, cast

import numpy as np

try:
    import onnxruntime as ort  # type: ignore[import-untyped]

    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    ort = cast(Any, None)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = cast(Any, None)


@dataclass
class ModelMetadata:
    """Metadata for loaded model artifacts."""

    name: str
    framework: str
    provider: str
    input_names: List[str]
    input_shapes: List[Sequence[Any]]
    output_names: List[str]
    output_shapes: List[Sequence[Any]]
    file_size_mb: float
    estimated_memory_mb: float


class ONNXModelLoader:
    """ONNX model loader with a simple infer interface."""

    def __init__(self, optimization_level: int = 2, **_: Any) -> None:
        self.optimization_level = optimization_level
        self._session: Optional["ort.InferenceSession"] = None
        self._metadata: Optional[ModelMetadata] = None

    @staticmethod
    def get_available_providers() -> List[str]:
        if not HAS_ORT:
            return []
        return cast(List[str], ort.get_available_providers())

    def load(self, model_path: Union[str, Path]) -> ModelMetadata:
        if not HAS_ORT:
            raise RuntimeError("onnxruntime is required.")

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)

        providers = self.get_available_providers()
        provider = providers[0] if providers else "CPUExecutionProvider"

        sess_opts = ort.SessionOptions()
        if self.optimization_level >= 2:
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        self._session = ort.InferenceSession(
            str(path), providers=[provider], sess_options=sess_opts
        )

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        size_mb = path.stat().st_size / (1024**2)
        self._metadata = ModelMetadata(
            name=path.stem,
            framework="onnx",
            provider=provider,
            input_names=[inp.name for inp in inputs],
            input_shapes=[inp.shape for inp in inputs],
            output_names=[out.name for out in outputs],
            output_shapes=[out.shape for out in outputs],
            file_size_mb=float(size_mb),
            estimated_memory_mb=float(size_mb * 2.0),
        )
        return self._metadata

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("Model not loaded.")
        inp = self._session.get_inputs()[0]
        output = self._session.run(None, {inp.name: input_data.astype(np.float32, copy=False)})
        return cast(np.ndarray, output[0])

    def unload(self) -> None:
        self._session = None
        self._metadata = None


class PyTorchModelLoader:
    """PyTorch loader compatibility shim."""

    def __init__(
        self, optimization_level: int = 2, preferred_device: str = "auto", **_: Any
    ) -> None:
        self.optimization_level = optimization_level
        self.preferred_device = preferred_device
        self._model: Optional["torch.nn.Module"] = None
        self._device = "cpu"
        if HAS_TORCH and preferred_device != "cpu" and torch.cuda.is_available():
            self._device = "cuda"

    @staticmethod
    def get_available_providers() -> List[str]:
        if not HAS_TORCH:
            return ["cpu"]
        providers = ["cpu"]
        if torch.cuda.is_available():
            providers.append("cuda")
        return providers

    def load(self, model_path: Union[str, Path]) -> ModelMetadata:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed.")
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)

        model = torch.jit.load(str(path), map_location=self._device)
        model.eval()
        self._model = model
        size_mb = path.stat().st_size / (1024**2)
        return ModelMetadata(
            name=path.stem,
            framework="pytorch",
            provider=self._device,
            input_names=["input"],
            input_shapes=[],
            output_names=["output"],
            output_shapes=[],
            file_size_mb=float(size_mb),
            estimated_memory_mb=float(size_mb * 2.0),
        )

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not loaded.")
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed.")

        tensor = torch.from_numpy(input_data).to(self._device)
        with torch.no_grad():
            out = self._model(tensor)
        return cast(np.ndarray, out.detach().cpu().numpy())

    def unload(self) -> None:
        self._model = None


def create_loader(
    model_path: Union[str, Path], optimization_level: int = 2, **kwargs: Any
) -> Union[ONNXModelLoader, PyTorchModelLoader]:
    """Factory for loading models by file extension."""
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        return ONNXModelLoader(optimization_level=optimization_level, **kwargs)
    if suffix in {".pt", ".pth", ".jit"}:
        return PyTorchModelLoader(optimization_level=optimization_level, **kwargs)
    raise ValueError(f"Unsupported model extension: {suffix}")
