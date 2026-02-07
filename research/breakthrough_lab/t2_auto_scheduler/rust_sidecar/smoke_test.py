#!/usr/bin/env python3
from __future__ import annotations

from python_client import (
    SearchRequest,
    build_replay_plan,
    enumerate_candidates,
    native_available,
    sidecar_info,
)


def main() -> int:
    request = SearchRequest(
        kernels=["tile20_v3", "tile24"],
        vector_widths=[4, 8],
        unroll_k_values=[0, 4],
        local_sizes=[[10, 10], [12, 12]],
        top_k=4,
    )

    c1 = enumerate_candidates(request)
    c2 = enumerate_candidates(request)
    assert c1 == c2, "candidate enumeration must be deterministic"
    assert len(c1) == 4, "top_k should limit candidates"

    replay = build_replay_plan(c1, sessions=3, runs=2, base_seed=42)
    assert len(replay) == 4 * 3 * 2, "replay cardinality mismatch"

    info = sidecar_info()
    mode = "native-rust" if native_available() else "python-fallback"
    print(f"Sidecar mode: {mode}")
    print(f"Info: {info}")
    print(f"Top candidate: {c1[0]['candidate_id']} score={c1[0]['estimated_score']:.3f}")
    print(f"Replay entries: {len(replay)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
