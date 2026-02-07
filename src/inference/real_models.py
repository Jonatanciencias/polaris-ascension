"""
Real-model integration compatibility stubs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RealModelIntegration:
    name: str
    status: str
    notes: str

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "status": self.status, "notes": self.notes}


def _build(name: str) -> RealModelIntegration:
    return RealModelIntegration(
        name=name,
        status="placeholder",
        notes="Integration scaffold available; connect your model pipeline here.",
    )


def create_whisper_integration() -> RealModelIntegration:
    return _build("whisper")


def create_stable_diffusion_integration() -> RealModelIntegration:
    return _build("stable_diffusion")


def create_llama2_integration() -> RealModelIntegration:
    return _build("llama2")


def create_bert_integration() -> RealModelIntegration:
    return _build("bert")
