#!/usr/bin/env python3
"""Python client for T2 Rust sidecar scaffold.

Uses native module (`t2_rust_sidecar`) when available, and a deterministic
pure-Python fallback otherwise so local workflows stay usable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

try:
    import t2_rust_sidecar as _native  # type: ignore

    _NATIVE_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    _native = None
    _NATIVE_AVAILABLE = False


@dataclass(frozen=True)
class SearchRequest:
    kernels: list[str]
    vector_widths: list[int]
    unroll_k_values: list[int]
    local_sizes: list[list[int]]
    top_k: int | None = None


def _kernel_bias(kernel: str) -> float:
    if "tile20_v3" in kernel or "v3" in kernel:
        return 10.0
    if "tile24" in kernel:
        return 8.0
    if "tile20" in kernel:
        return 6.5
    return 5.0


def _vector_bonus(vector_width: int) -> float:
    lookup = {1: 0.0, 2: 0.8, 4: 1.6, 8: 2.1}
    return lookup.get(vector_width, 1.0 + (int(vector_width).bit_length() - 1) * 0.25)


def _estimate_score(kernel: str, vector_width: int, unroll_k: int, local_size: list[int]) -> float:
    area = int(local_size[0]) * int(local_size[1])
    area_penalty = abs(area - 100) * 0.015
    warp_penalty = (area - 256) * 0.01 if area > 256 else 0.0
    return _kernel_bias(kernel) + _vector_bonus(vector_width) + (unroll_k * 0.12) - area_penalty - warp_penalty


def _enumerate_python(req: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for kernel in req["kernels"]:
        for vw in req["vector_widths"]:
            for uk in req["unroll_k_values"]:
                for ls in req["local_sizes"]:
                    out.append(
                        {
                            "candidate_id": f"{kernel}_vw{vw}_u{uk}_l{ls[0]}x{ls[1]}",
                            "kernel": kernel,
                            "vector_width": int(vw),
                            "unroll_k": int(uk),
                            "local_size": [int(ls[0]), int(ls[1])],
                            "estimated_score": float(_estimate_score(kernel, int(vw), int(uk), ls)),
                        }
                    )
    out.sort(key=lambda x: (-x["estimated_score"], x["candidate_id"]))
    top_k = req.get("top_k")
    return out[: int(top_k)] if top_k else out


def _build_replay_plan_python(
    candidates: list[dict[str, Any]],
    sessions: int,
    runs: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        idx_base = idx * 100_000
        for session in range(sessions):
            for run in range(runs):
                seed = int(base_seed + idx_base + session * 1_000 + run)
                out.append(
                    {
                        "candidate_id": cand["candidate_id"],
                        "session": session,
                        "run": run,
                        "seed": seed,
                        "execution_tag": f"{cand['candidate_id']}_s{session}_r{run}",
                    }
                )
    return out


def native_available() -> bool:
    return _NATIVE_AVAILABLE


def sidecar_info() -> dict[str, Any]:
    if _NATIVE_AVAILABLE:
        return json.loads(_native.sidecar_info_json())
    return {
        "sidecar_name": "t2_rust_sidecar_python_fallback",
        "version": "0.1.0-fallback",
        "capabilities": ["candidate_enumeration", "deterministic_replay_plan"],
    }


def enumerate_candidates(request: SearchRequest | dict[str, Any]) -> list[dict[str, Any]]:
    req_dict = asdict(request) if isinstance(request, SearchRequest) else request
    if _NATIVE_AVAILABLE:
        return json.loads(_native.enumerate_candidates_json(json.dumps(req_dict)))
    return _enumerate_python(req_dict)


def build_replay_plan(
    candidates: list[dict[str, Any]],
    *,
    sessions: int = 5,
    runs: int = 10,
    base_seed: int = 42,
) -> list[dict[str, Any]]:
    if _NATIVE_AVAILABLE:
        return json.loads(_native.build_replay_plan_json(json.dumps(candidates), sessions, runs, base_seed))
    return _build_replay_plan_python(candidates, sessions=sessions, runs=runs, base_seed=base_seed)
