#!/usr/bin/env python3
"""
Week 2 T2 bounded-search runner.

Supports a basic search space (production kernels) and an expanded space
including vector/unroll/local-size variants from research kernels.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.auto_tuner.gemm_auto_tuner import GEMMAutoTuner


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _build_search_configs(*, kernels: list[str], search_space: str) -> list[dict[str, Any]]:
    all_configs: list[dict[str, Any]] = [
        {
            "config_id": "t20_prod_v4_u10_l10",
            "family": "tile20",
            "kernel_file": "src/kernels/gemm_tile20_production.cl",
            "kernel_name": "gemm_tile20_optimized",
            "tile_size": 20,
            "local_size": [10, 10],
            "vector_width": 4,
            "unroll_k": 10,
            "source": "production",
        },
        {
            "config_id": "t24_prod_v4_u0_l12",
            "family": "tile24",
            "kernel_file": "src/kernels/gemm_tile24_production.cl",
            "kernel_name": "gemm_tile24_vectorized",
            "tile_size": 24,
            "local_size": [12, 12],
            "vector_width": 4,
            "unroll_k": 0,
            "source": "production",
        },
        {
            "config_id": "t20_prefetch_v4_u4_l10",
            "family": "tile20",
            "kernel_file": "research/tile_20_investigation/kernels/tile20_prefetch.cl",
            "kernel_name": "gemm_tile20_prefetch",
            "tile_size": 20,
            "local_size": [10, 10],
            "vector_width": 4,
            "unroll_k": 4,
            "source": "research",
        },
        {
            "config_id": "t20_v3vec_v4_u0_l10",
            "family": "tile20",
            "kernel_file": "research/tile_20_investigation/kernels/approach_2_v3_vectorized.cl",
            "kernel_name": "gemm_tile20_vectorized",
            "tile_size": 20,
            "local_size": [10, 10],
            "vector_width": 4,
            "unroll_k": 0,
            "source": "research",
        },
        {
            "config_id": "t20_regblock_v4_u0_l5",
            "family": "tile20",
            "kernel_file": "research/tile_20_investigation/kernels/approach_5_optimized.cl",
            "kernel_name": "gemm_tile20_register_blocking",
            "tile_size": 20,
            "local_size": [5, 5],
            "vector_width": 4,
            "unroll_k": 0,
            "source": "research",
        },
        {
            "config_id": "t20_float8_v8_u8_l10",
            "family": "tile20",
            "kernel_file": "research/tile_20_investigation/kernels/tile20_float8.cl",
            "kernel_name": "gemm_tile20_float8",
            "tile_size": 20,
            "local_size": [10, 10],
            "vector_width": 8,
            "unroll_k": 8,
            "source": "research",
        },
    ]

    filtered = [cfg for cfg in all_configs if cfg["family"] in kernels]
    if search_space == "basic":
        return [cfg for cfg in filtered if cfg["source"] == "production"]
    return filtered


def _baseline_config_for_size(size: int) -> str:
    return "t20_prod_v4_u10_l10" if size < 1800 else "t24_prod_v4_u0_l12"


def _markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T2 Week 2 Search (Deterministic + Strict)")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Search space: {report['metadata']['search_space']}")
    lines.append(
        "- Budget: "
        f"{report['metadata']['config_count']} configs x {report['metadata']['sizes']} sizes x "
        f"{report['metadata']['runs_per_config']} runs"
    )
    lines.append(f"- Input distribution: {report['metadata']['input_distribution']}")
    lines.append(f"- Correctness threshold: {report['metadata']['correctness_threshold']}")
    lines.append("")
    lines.append("## Search Results")
    lines.append("")
    lines.append(
        "| Rank | Config | Family | Vec | Unroll | Local | Size | GFLOPS | Max Error | Delta vs Baseline | Status |"
    )
    lines.append("| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |")

    for idx, item in enumerate(report["search"]["ranked_candidates"], start=1):
        delta = item["delta_vs_baseline_percent"]
        delta_text = "n/a" if delta is None else f"{delta:+.3f}%"
        status = "valid" if item["correctness_passed"] else "filtered"
        lines.append(
            f"| {idx} | {item['config_id']} | {item['family']} | {item['vector_width']} | "
            f"{item['unroll_k']} | {item['local_size'][0]}x{item['local_size'][1]} | "
            f"{item['size']} | {item['gflops']:.3f} | {item['max_error']:.6f} | {delta_text} | {status} |"
        )

    lines.append("")
    lines.append("## Replay Summary")
    lines.append("")
    lines.append(
        "| Candidate | Vec | Unroll | Local | Mean GFLOPS | CV | Max Error (max) | Delta vs Baseline | Correctness | Stability | Promotion |"
    )
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |")

    for item in report["replay"]["candidates"]:
        if item.get("status") != "completed":
            lines.append(
                f"| {item['config_id']}@{item['size']} | - | - | - | - | - | - | - | - | - | failed |"
            )
            continue
        delta = item["mean_delta_vs_baseline_percent"]
        delta_text = "n/a" if delta is None else f"{delta:+.3f}%"
        lines.append(
            f"| {item['config_id']}@{item['size']} | {item['vector_width']} | {item['unroll_k']} | "
            f"{item['local_size'][0]}x{item['local_size'][1]} | {item['gflops']['mean']:.3f} | "
            f"{item['cv_peak']:.5f} | {item['max_error']['max']:.6f} | {delta_text} | "
            f"{item['correctness_passed']} | {item['stability_passed']} | {item['promotion_gate_passed']} |"
        )

    lines.append("")
    lines.append("## Decision Hint")
    lines.append("")
    lines.append(f"- Suggested decision: `{report['summary']['decision_hint']}`")
    lines.append(f"- Decision rationale: {report['summary']['decision_rationale']}")
    return "\n".join(lines) + "\n"


def run_search(
    *,
    search_space: str,
    kernels: list[str],
    sizes: list[int],
    runs_per_config: int,
    warmup: int,
    replay_sessions: int,
    replay_runs: int,
    top_k: int,
    correctness_threshold: float,
    seed: int,
    input_distribution: str,
) -> dict[str, Any]:
    tuner = GEMMAutoTuner(output_dir="results/auto_tuner", verbose=False)
    configs = _build_search_configs(kernels=kernels, search_space=search_space)
    raw_candidates: list[dict[str, Any]] = []
    if not configs:
        return {
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "search_space": search_space,
                "kernels": kernels,
                "sizes": sizes,
                "runs_per_config": runs_per_config,
                "warmup": warmup,
                "replay_sessions": replay_sessions,
                "replay_runs": replay_runs,
                "top_k": top_k,
                "correctness_threshold": correctness_threshold,
                "seed": seed,
                "input_distribution": input_distribution,
                "config_count": 0,
            },
            "search": {
                "raw_candidates": [],
                "baseline_by_size": {},
                "ranked_candidates": [],
                "valid_count": 0,
                "completed_count": 0,
            },
            "replay": {"candidates": []},
            "summary": {
                "decision_hint": "refine",
                "decision_rationale": "No valid configurations selected for this search space.",
            },
        }

    for cfg_idx, config in enumerate(configs):
        for size in sizes:
            config_seed = seed + cfg_idx * 1000 + size
            result = tuner.benchmark_custom_kernel(
                kernel_file=config["kernel_file"],
                kernel_name=config["kernel_name"],
                tile_size=int(config["tile_size"]),
                local_size=(int(config["local_size"][0]), int(config["local_size"][1])),
                M=size,
                N=size,
                K=size,
                runs=runs_per_config,
                warmup=warmup,
                seed=config_seed,
                input_distribution=input_distribution,
            )
            if result is None:
                raw_candidates.append(
                    {
                        "config_id": config["config_id"],
                        "family": config["family"],
                        "tile_size": int(config["tile_size"]),
                        "vector_width": int(config["vector_width"]),
                        "unroll_k": int(config["unroll_k"]),
                        "local_size": list(config["local_size"]),
                        "kernel_file": config["kernel_file"],
                        "kernel_name": config["kernel_name"],
                        "size": size,
                        "status": "failed",
                        "error": "benchmark_custom_kernel returned None",
                    }
                )
                continue

            raw_candidates.append(
                {
                    "config_id": config["config_id"],
                    "family": config["family"],
                    "tile_size": int(config["tile_size"]),
                    "vector_width": int(config["vector_width"]),
                    "unroll_k": int(config["unroll_k"]),
                    "local_size": list(config["local_size"]),
                    "kernel_file": config["kernel_file"],
                    "kernel_name": config["kernel_name"],
                    "size": size,
                    "status": "completed",
                    "gflops": float(result.gflops),
                    "avg_time_ms": float(result.avg_time_ms),
                    "max_error": float(result.max_error),
                    "runs": int(result.runs),
                    "seed": config_seed,
                }
            )

    completed = [item for item in raw_candidates if item.get("status") == "completed"]
    baseline_by_size: dict[int, float] = {}
    baseline_config_by_size: dict[int, str] = {}
    for size in sizes:
        baseline_config = _baseline_config_for_size(size)
        baseline_items = [
            item
            for item in completed
            if item["size"] == size and item["config_id"] == baseline_config
        ]
        if baseline_items:
            baseline_by_size[size] = baseline_items[0]["gflops"]
            baseline_config_by_size[size] = baseline_config

    ranked_input: list[dict[str, Any]] = []
    for item in completed:
        size = int(item["size"])
        baseline = baseline_by_size.get(size)
        delta = None
        if baseline and baseline > 0:
            delta = (item["gflops"] - baseline) / baseline * 100.0

        decorated = {
            **item,
            "correctness_passed": item["max_error"] <= correctness_threshold,
            "delta_vs_baseline_percent": None if delta is None else float(delta),
        }
        ranked_input.append(decorated)

    valid = [item for item in ranked_input if item["correctness_passed"]]
    pool = valid if valid else ranked_input
    ranked = sorted(
        pool,
        key=lambda item: (
            item["delta_vs_baseline_percent"] if item["delta_vs_baseline_percent"] is not None else -1e9,
            item["gflops"],
        ),
        reverse=True,
    )
    top_candidates = ranked[:top_k]

    replay_candidates: list[dict[str, Any]] = []
    for idx, candidate in enumerate(top_candidates):
        gflops_values: list[float] = []
        time_values: list[float] = []
        error_values: list[float] = []

        for session_idx in range(replay_sessions):
            replay_seed = seed + 10_000 + idx * 100 + session_idx
            replay_result = tuner.benchmark_custom_kernel(
                kernel_file=candidate["kernel_file"],
                kernel_name=candidate["kernel_name"],
                tile_size=int(candidate["tile_size"]),
                local_size=(int(candidate["local_size"][0]), int(candidate["local_size"][1])),
                M=candidate["size"],
                N=candidate["size"],
                K=candidate["size"],
                runs=replay_runs,
                warmup=warmup,
                seed=replay_seed,
                input_distribution=input_distribution,
            )
            if replay_result is None:
                continue
            gflops_values.append(float(replay_result.gflops))
            time_values.append(float(replay_result.avg_time_ms))
            error_values.append(float(replay_result.max_error))

        if not gflops_values:
            replay_candidates.append(
                {
                    "config_id": candidate["config_id"],
                    "family": candidate["family"],
                    "tile_size": candidate["tile_size"],
                    "vector_width": candidate["vector_width"],
                    "unroll_k": candidate["unroll_k"],
                    "local_size": candidate["local_size"],
                    "size": candidate["size"],
                    "status": "failed",
                    "error": "no replay samples",
                }
            )
            continue

        size = int(candidate["size"])
        baseline = baseline_by_size.get(size)
        gflops_mean = float(np.mean(gflops_values))
        delta_mean = None
        if baseline and baseline > 0:
            delta_mean = (gflops_mean - baseline) / baseline * 100.0

        cv_peak = float(np.std(gflops_values) / np.mean(gflops_values))
        max_error_max = float(np.max(error_values))
        correctness_passed = max_error_max <= correctness_threshold
        stability_passed = cv_peak <= 0.03
        promotion_passed = (
            delta_mean is not None
            and delta_mean >= 10.0
            and correctness_passed
            and stability_passed
        )

        replay_candidates.append(
            {
                "config_id": candidate["config_id"],
                "family": candidate["family"],
                "tile_size": candidate["tile_size"],
                "vector_width": candidate["vector_width"],
                "unroll_k": candidate["unroll_k"],
                "local_size": candidate["local_size"],
                "size": candidate["size"],
                "status": "completed",
                "gflops": _stats(gflops_values),
                "avg_time_ms": _stats(time_values),
                "max_error": _stats(error_values),
                "cv_peak": cv_peak,
                "mean_delta_vs_baseline_percent": None if delta_mean is None else float(delta_mean),
                "correctness_passed": correctness_passed,
                "stability_passed": stability_passed,
                "promotion_gate_passed": promotion_passed,
            }
        )

    replay_completed = [item for item in replay_candidates if item.get("status") == "completed"]

    if any(item["promotion_gate_passed"] for item in replay_completed):
        decision_hint = "promote"
        rationale = "At least one replayed candidate passed performance, correctness and stability gates."
    elif not valid:
        decision_hint = "refine"
        rationale = "No candidate passed strict correctness filter in search phase."
    elif any(item["correctness_passed"] and item["stability_passed"] for item in replay_completed):
        decision_hint = "iterate"
        rationale = "Correct/stable candidates exist but promotion uplift threshold is not met."
    else:
        decision_hint = "refine"
        rationale = "Replay did not produce stable and correct candidates under strict filter."

    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "search_space": search_space,
            "kernels": kernels,
            "sizes": sizes,
            "runs_per_config": runs_per_config,
            "warmup": warmup,
            "replay_sessions": replay_sessions,
            "replay_runs": replay_runs,
            "top_k": top_k,
            "correctness_threshold": correctness_threshold,
            "seed": seed,
            "input_distribution": input_distribution,
            "config_count": len(configs),
            "dimensions": {
                "vector_widths": sorted({int(cfg["vector_width"]) for cfg in configs}),
                "unroll_k": sorted({int(cfg["unroll_k"]) for cfg in configs}),
                "local_sizes": sorted({f"{cfg['local_size'][0]}x{cfg['local_size'][1]}" for cfg in configs}),
            },
        },
        "search": {
            "configs": configs,
            "raw_candidates": raw_candidates,
            "baseline_by_size": baseline_by_size,
            "baseline_config_by_size": baseline_config_by_size,
            "ranked_candidates": ranked,
            "valid_count": len(valid),
            "completed_count": len(completed),
        },
        "replay": {
            "candidates": replay_candidates,
        },
        "summary": {
            "decision_hint": decision_hint,
            "decision_rationale": rationale,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week 2 T2 bounded search")
    parser.add_argument(
        "--search-space",
        choices=["basic", "expanded"],
        default="expanded",
    )
    parser.add_argument("--kernels", nargs="+", default=["tile20", "tile24"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048, 3072])
    parser.add_argument("--runs-per-config", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--replay-sessions", type=int, default=5)
    parser.add_argument("--replay-runs", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--correctness-threshold", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input-distribution",
        choices=["standard_normal", "uniform"],
        default="standard_normal",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research/breakthrough_lab/t2_auto_scheduler",
    )
    args = parser.parse_args()

    report = run_search(
        search_space=args.search_space,
        kernels=args.kernels,
        sizes=args.sizes,
        runs_per_config=args.runs_per_config,
        warmup=args.warmup,
        replay_sessions=args.replay_sessions,
        replay_runs=args.replay_runs,
        top_k=args.top_k,
        correctness_threshold=args.correctness_threshold,
        seed=args.seed,
        input_distribution=args.input_distribution,
    )

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    prefix = "week2_t2_expanded_search" if args.search_space == "expanded" else "week2_t2_bounded_search"
    json_path = output_dir / f"{prefix}_{timestamp}.json"
    md_path = output_dir / f"{prefix}_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown_report(report))

    print(f"T2 search JSON: {json_path}")
    print(f"T2 search MD:   {md_path}")
    print(f"Decision hint:  {report['summary']['decision_hint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
