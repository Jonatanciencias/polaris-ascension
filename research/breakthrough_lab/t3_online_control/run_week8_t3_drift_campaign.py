#!/usr/bin/env python3
"""Week 8 Block 3: controlled drift campaign for T3 policy hardening.

Runs reproducible scenarios for:
- cold
- warm
- warm_queue_pressure

Comparisons are generated for:
- auto
- auto_t3_controlled
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.benchmarking.production_kernel_benchmark import run_production_benchmark

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_policy(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    required = [
        "policy_id",
        "scope",
        "guardrails",
        "drift_guardrails",
        "promotion_gate",
        "rollback_defaults",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Policy missing required fields: {missing}")
    return data


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


def _run_silent_benchmark(**kwargs: Any) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_production_benchmark(**kwargs)


def _queue_pressure_pulse(
    *,
    pulses: int,
    size: int,
    iterations: int,
    seed: int,
    pause_ms: int,
    policy_path: Path,
) -> dict[str, Any]:
    completed = 0
    failures = 0
    errors: list[str] = []
    for pulse_idx in range(max(0, int(pulses))):
        try:
            _run_silent_benchmark(
                size=int(size),
                iterations=int(iterations),
                sessions=1,
                kernel="auto_t3_controlled",
                seed=int(seed + pulse_idx * 17),
                t3_policy_path=str(policy_path),
            )
            completed += 1
        except Exception as exc:  # pragma: no cover - runtime safety path
            failures += 1
            errors.append(str(exc))
        if pause_ms > 0:
            time.sleep(float(pause_ms) / 1000.0)
    return {
        "requested": int(pulses),
        "completed": int(completed),
        "failures": int(failures),
        "errors": errors[:3],
    }


def _aggregate_kernel_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    avg_vals = [float(r["avg_gflops_mean"]) for r in rows]
    peak_vals = [float(r["peak_gflops_mean"]) for r in rows]
    p95_vals = [float(r["p95_time_ms"]) for r in rows]
    error_vals = [float(r["max_error_max"]) for r in rows]
    fallback_count = int(sum(int(r.get("fallback_count", 0)) for r in rows))
    total_sessions = int(sum(int(r["sessions"]) for r in rows))
    correctness_failures = int(sum(int(r.get("correctness_failures", 0)) for r in rows))
    disable_events = int(sum(int(r.get("disable_events", 0)) for r in rows))
    policy_disabled = bool(any(bool(r.get("policy_disabled", False)) for r in rows))

    avg_stats = _stats(avg_vals)
    peak_stats = _stats(peak_vals)
    p95_stats = _stats(p95_vals)
    cv_peak = float(peak_stats["std"] / peak_stats["mean"]) if peak_stats["mean"] > 0 else 0.0

    return {
        "avg_gflops": avg_stats,
        "peak_gflops": peak_stats,
        "p95_time_ms": p95_stats,
        "max_error_max": float(max(error_vals)),
        "cv_peak": cv_peak,
        "fallback_count": int(fallback_count),
        "fallback_rate": float(fallback_count / max(1, total_sessions)),
        "correctness_failures": correctness_failures,
        "disable_events": disable_events,
        "policy_disabled": policy_disabled,
    }


def _scenario_comparison(*, auto_summary: dict[str, Any], t3_summary: dict[str, Any]) -> dict[str, float]:
    auto_avg = float(auto_summary["avg_gflops"]["mean"])
    t3_avg = float(t3_summary["avg_gflops"]["mean"])
    auto_p95 = float(auto_summary["p95_time_ms"]["mean"])
    t3_p95 = float(t3_summary["p95_time_ms"]["mean"])

    delta_vs_auto = ((t3_avg - auto_avg) / auto_avg * 100.0) if auto_avg > 0 else 0.0
    p95_delta_vs_auto = ((t3_p95 - auto_p95) / auto_p95 * 100.0) if auto_p95 > 0 else 0.0

    return {
        "t3_delta_vs_auto_percent": float(delta_vs_auto),
        "t3_p95_delta_vs_auto_percent": float(p95_delta_vs_auto),
    }


def _decision(report: dict[str, Any], policy: dict[str, Any]) -> tuple[str, str]:
    checks = report["evaluation"]["checks"]
    failed = [name for name, payload in checks.items() if not payload["pass"]]
    if not failed:
        return (
            "promote",
            "Drift campaign passed correctness and rollback-safe guardrails under warm/cold/pressure scenarios.",
        )

    correctness_failed = not bool(checks["correctness_guard"]["pass"])
    if correctness_failed:
        return (
            "drop",
            "Correctness guard failed under drift campaign; enforce rollback to static mode.",
        )

    rollback_mode = policy["rollback_defaults"]["default_kernel_mode"]
    return (
        "iterate",
        f"One or more drift guardrails failed; keep controlled rollout blocked and use `{rollback_mode}` default.",
    )


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# T3 Week 8 Block 3 - Drift Campaign Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Policy: `{report['metadata']['policy_id']}`")
    lines.append(f"- Sizes: {report['metadata']['sizes']}")
    lines.append(
        f"- Sessions={report['metadata']['sessions']} | Iterations={report['metadata']['iterations']} | Seed={report['metadata']['seed']}"
    )
    lines.append("")
    lines.append("## Scenario Summary")
    lines.append("")
    lines.append(
        "| Scenario | Auto avg GFLOPS | T3 avg GFLOPS | T3 delta vs auto | T3 p95 delta vs auto | T3 fallback rate |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, scenario in report["scenario_summary"].items():
        comp = scenario["comparison"]
        lines.append(
            f"| {name} | {scenario['auto']['avg_gflops']['mean']:.3f} | {scenario['auto_t3_controlled']['avg_gflops']['mean']:.3f} | {comp['t3_delta_vs_auto_percent']:+.3f}% | {comp['t3_p95_delta_vs_auto_percent']:+.3f}% | {scenario['auto_t3_controlled']['fallback_rate']:.3f} |"
        )
    lines.append("")
    lines.append("## Drift Drop vs Cold")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["drift_drop_vs_cold_percent"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Queue Pressure Pulses")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["queue_pressure_pulses"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Gate Evaluation")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["evaluation"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['decision']['decision']}`")
    lines.append(f"- Rationale: {report['decision']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_campaign(
    *,
    policy_path: Path,
    sizes: list[int],
    sessions: int,
    iterations: int,
    seed: int,
    pressure_size: int,
    pressure_iterations: int,
    pressure_pulses: int,
    pressure_pause_ms: int,
) -> dict[str, Any]:
    policy = _load_policy(policy_path)
    scenarios = [
        {"name": "cold", "warmup_runs": 0, "queue_pressure": False},
        {"name": "warm", "warmup_runs": 1, "queue_pressure": False},
        {"name": "warm_queue_pressure", "warmup_runs": 1, "queue_pressure": True},
    ]
    kernels = ["auto", "auto_t3_controlled"]

    all_rows: list[dict[str, Any]] = []
    scenario_summary: dict[str, Any] = {}
    queue_pulse_meta: dict[str, Any] = {}

    for scenario_idx, scenario in enumerate(scenarios):
        scenario_name = str(scenario["name"])
        scenario_queue_meta = {
            "requested": 0,
            "completed": 0,
            "failures": 0,
            "errors": [],
        }

        for kernel_idx, kernel in enumerate(kernels):
            for size_idx, size in enumerate(sizes):
                run_seed = int(
                    seed
                    + scenario_idx * 1_000_000
                    + kernel_idx * 100_000
                    + size_idx * 10_000
                )

                warmup_runs = int(scenario["warmup_runs"])
                for warmup_idx in range(warmup_runs):
                    _run_silent_benchmark(
                        size=int(size),
                        iterations=max(3, int(iterations // 2)),
                        sessions=1,
                        kernel=str(kernel),
                        seed=run_seed + warmup_idx * 101 + 7,
                        t3_policy_path=str(policy_path),
                    )

                if bool(scenario["queue_pressure"]):
                    pulse = _queue_pressure_pulse(
                        pulses=int(pressure_pulses),
                        size=int(pressure_size),
                        iterations=int(pressure_iterations),
                        seed=run_seed + 19,
                        pause_ms=int(pressure_pause_ms),
                        policy_path=policy_path,
                    )
                    scenario_queue_meta["requested"] += int(pulse["requested"])
                    scenario_queue_meta["completed"] += int(pulse["completed"])
                    scenario_queue_meta["failures"] += int(pulse["failures"])
                    if pulse["errors"]:
                        scenario_queue_meta["errors"].extend([str(x) for x in pulse["errors"]])

                run_report = _run_silent_benchmark(
                    size=int(size),
                    iterations=int(iterations),
                    sessions=int(sessions),
                    kernel=str(kernel),
                    seed=run_seed,
                    t3_policy_path=str(policy_path),
                )

                summary = run_report["summary"]
                decisions = run_report.get("decisions", [])
                correctness_failures = sum(
                    1 for d in decisions if str(d.get("fallback_reason")) == "correctness"
                )
                disable_events = sum(1 for d in decisions if bool(d.get("disable_signal", False)))

                row = {
                    "scenario": scenario_name,
                    "kernel": kernel,
                    "size": int(size),
                    "seed": int(run_seed),
                    "sessions": int(sessions),
                    "iterations": int(iterations),
                    "avg_gflops_mean": float(summary["avg_gflops"]["mean"]),
                    "peak_gflops_mean": float(summary["peak_gflops"]["mean"]),
                    "p95_time_ms": float(summary["time_ms"]["p95"]),
                    "max_error_max": float(summary["max_error"]["max"]),
                    "fallback_count": int(summary.get("fallback_count", 0)),
                    "fallback_rate": float(summary.get("fallback_rate", 0.0)),
                    "policy_disabled": bool(summary.get("policy_disabled", False)),
                    "correctness_failures": int(correctness_failures),
                    "disable_events": int(disable_events),
                    "platform": str(run_report["metadata"]["platform"]),
                    "device": str(run_report["metadata"]["device"]),
                }
                if kernel == "auto_t3_controlled":
                    row["policy_snapshot"] = run_report.get("policy_snapshot")
                all_rows.append(row)

        queue_pulse_meta[scenario_name] = {
            "requested": int(scenario_queue_meta["requested"]),
            "completed": int(scenario_queue_meta["completed"]),
            "failures": int(scenario_queue_meta["failures"]),
            "errors": [str(x) for x in scenario_queue_meta["errors"][:3]],
        }

        scenario_rows = [r for r in all_rows if r["scenario"] == scenario_name]
        auto_rows = [r for r in scenario_rows if r["kernel"] == "auto"]
        t3_rows = [r for r in scenario_rows if r["kernel"] == "auto_t3_controlled"]

        auto_summary = _aggregate_kernel_rows(auto_rows)
        t3_summary = _aggregate_kernel_rows(t3_rows)
        comparison = _scenario_comparison(auto_summary=auto_summary, t3_summary=t3_summary)
        scenario_summary[scenario_name] = {
            "auto": auto_summary,
            "auto_t3_controlled": t3_summary,
            "comparison": comparison,
        }

    cold_auto = float(scenario_summary["cold"]["auto"]["avg_gflops"]["mean"])
    cold_t3 = float(scenario_summary["cold"]["auto_t3_controlled"]["avg_gflops"]["mean"])
    warmq_auto = float(scenario_summary["warm_queue_pressure"]["auto"]["avg_gflops"]["mean"])
    warmq_t3 = float(scenario_summary["warm_queue_pressure"]["auto_t3_controlled"]["avg_gflops"]["mean"])

    drift_drop = {
        "auto": ((cold_auto - warmq_auto) / cold_auto * 100.0) if cold_auto > 0 else 0.0,
        "auto_t3_controlled": ((cold_t3 - warmq_t3) / cold_t3 * 100.0) if cold_t3 > 0 else 0.0,
    }

    correctness_limit = float(policy["guardrails"]["disable_if_correctness_error_gt"])
    fallback_limit_pressure = float(policy["drift_guardrails"]["max_t3_fallback_rate_under_pressure"])
    p95_delta_limit = float(
        policy["drift_guardrails"]["max_t3_p95_latency_delta_vs_auto_percent_under_pressure"]
    )
    t3_drop_limit = float(policy["drift_guardrails"]["max_t3_throughput_drop_vs_cold_percent"])
    auto_drop_limit = float(policy["drift_guardrails"]["max_auto_throughput_drop_vs_cold_percent"])
    min_delta_under_pressure = float(
        policy["promotion_gate"]["min_t3_delta_vs_auto_percent_under_pressure"]
    )
    max_correctness_failures = int(policy["promotion_gate"]["max_correctness_failures"])

    max_error_all = max(float(r["max_error_max"]) for r in all_rows)
    total_correctness_failures = int(sum(int(r["correctness_failures"]) for r in all_rows))
    warmq = scenario_summary["warm_queue_pressure"]
    warmq_comp = warmq["comparison"]
    warmq_t3_summary = warmq["auto_t3_controlled"]

    checks = {
        "correctness_guard": {
            "observed_max_error": max_error_all,
            "threshold_max_error": correctness_limit,
            "observed_correctness_failures": total_correctness_failures,
            "threshold_correctness_failures": max_correctness_failures,
            "pass": bool(
                max_error_all <= correctness_limit
                and total_correctness_failures <= max_correctness_failures
            ),
        },
        "pressure_t3_fallback_rate": {
            "observed": float(warmq_t3_summary["fallback_rate"]),
            "threshold_max": fallback_limit_pressure,
            "pass": bool(float(warmq_t3_summary["fallback_rate"]) <= fallback_limit_pressure),
        },
        "pressure_t3_p95_delta_vs_auto": {
            "observed_percent": float(warmq_comp["t3_p95_delta_vs_auto_percent"]),
            "threshold_max_percent": p95_delta_limit,
            "pass": bool(float(warmq_comp["t3_p95_delta_vs_auto_percent"]) <= p95_delta_limit),
        },
        "pressure_t3_drop_vs_cold": {
            "observed_percent": float(drift_drop["auto_t3_controlled"]),
            "threshold_max_percent": t3_drop_limit,
            "pass": bool(float(drift_drop["auto_t3_controlled"]) <= t3_drop_limit),
        },
        "pressure_auto_drop_vs_cold": {
            "observed_percent": float(drift_drop["auto"]),
            "threshold_max_percent": auto_drop_limit,
            "pass": bool(float(drift_drop["auto"]) <= auto_drop_limit),
        },
        "pressure_t3_delta_vs_auto": {
            "observed_percent": float(warmq_comp["t3_delta_vs_auto_percent"]),
            "threshold_min_percent": min_delta_under_pressure,
            "pass": bool(float(warmq_comp["t3_delta_vs_auto_percent"]) >= min_delta_under_pressure),
        },
        "policy_not_disabled": {
            "observed": bool(warmq_t3_summary["policy_disabled"]),
            "required": False,
            "pass": not bool(warmq_t3_summary["policy_disabled"]),
        },
    }

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "branch": subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(REPO_ROOT),
                text=True,
            ).strip(),
            "policy_path": str(policy_path),
            "policy_id": str(policy["policy_id"]),
            "sizes": [int(x) for x in sizes],
            "sessions": int(sessions),
            "iterations": int(iterations),
            "seed": int(seed),
            "pressure_config": {
                "size": int(pressure_size),
                "iterations": int(pressure_iterations),
                "pulses": int(pressure_pulses),
                "pause_ms": int(pressure_pause_ms),
            },
        },
        "policy": policy,
        "scenario_summary": scenario_summary,
        "drift_drop_vs_cold_percent": drift_drop,
        "queue_pressure_pulses": queue_pulse_meta,
        "runs": all_rows,
        "evaluation": {
            "checks": checks,
            "failed_checks": [name for name, payload in checks.items() if not payload["pass"]],
        },
    }
    decision, rationale = _decision(report, policy)
    report["decision"] = {
        "decision": decision,
        "rationale": rationale,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week 8 Block 3 T3 drift campaign.")
    parser.add_argument(
        "--policy-path",
        default="research/breakthrough_lab/t3_online_control/policy_hardening_block3.json",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 1536, 2048])
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pressure-size", type=int, default=896)
    parser.add_argument("--pressure-iterations", type=int, default=3)
    parser.add_argument("--pressure-pulses", type=int, default=2)
    parser.add_argument("--pressure-pause-ms", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/t3_online_control",
    )
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = (REPO_ROOT / args.policy_path).resolve()

    report = run_campaign(
        policy_path=policy_path,
        sizes=list(args.sizes),
        sessions=int(args.sessions),
        iterations=int(args.iterations),
        seed=int(args.seed),
        pressure_size=int(args.pressure_size),
        pressure_iterations=int(args.pressure_iterations),
        pressure_pulses=int(args.pressure_pulses),
        pressure_pause_ms=int(args.pressure_pause_ms),
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week8_t3_drift_campaign_{ts}.json"
    md_path = output_dir / f"week8_t3_drift_campaign_{ts}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown(report))

    print(f"T3 drift JSON: {json_path}")
    print(f"T3 drift MD:   {md_path}")
    print(f"Decision: {report['decision']['decision']}")
    print(
        "Warm+pressure delta vs auto: "
        f"{report['scenario_summary']['warm_queue_pressure']['comparison']['t3_delta_vs_auto_percent']:+.3f}%"
    )
    print(
        "Warm+pressure fallback rate: "
        f"{report['scenario_summary']['warm_queue_pressure']['auto_t3_controlled']['fallback_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
