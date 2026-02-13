"""
ONNX export compatibility helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    ort = None  # type: ignore[assignment]


def export_to_onnx(
    model: "torch.nn.Module",
    sample_input: "torch.Tensor",
    output_path: str,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    opset_version: int = 13,
) -> str:
    """Export a torch model to ONNX."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for export_to_onnx.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        str(path),
        input_names=list(input_names or ["input"]),
        output_names=list(output_names or ["output"]),
        opset_version=opset_version,
    )
    return str(path)


def get_model_info(model_path: str) -> Dict[str, Any]:
    """Read lightweight model info via ONNX Runtime session inspection."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(path)

    info: Dict[str, Any] = {
        "path": str(path),
        "size_mb": path.stat().st_size / (1024**2),
    }

    if HAS_ORT:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        info["providers"] = session.get_providers()
        info["inputs"] = [
            {"name": i.name, "shape": i.shape, "type": i.type} for i in session.get_inputs()
        ]
        info["outputs"] = [
            {"name": o.name, "shape": o.shape, "type": o.type} for o in session.get_outputs()
        ]

    return info
