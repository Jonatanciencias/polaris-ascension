#!/usr/bin/env python3
"""Week 9 Block 1: long mixed canary under queue pressure (T3 + T5)."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(REPO_ROOT))

from src.benchmarking.production_kernel_benchmark import run_production_benchmark


def _run_silent_benchmark(**kwargs: Any) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_production_benchmark(**kwargs)


def _queue_pressure(
    *,
    pulses: int,
    size: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    completed = 0
    failures = 0
    errors: list[str] = []
    for idx in range(int(pulses)):
        try:
            _run_silent_benchmark(
                size=int(size),
                iterations=int(iterations),
                sessions=1,
                kernel="auto",
                seed=int(seed + idx * 17),
            )
            completed += 1
        except Exception as exc:  # pragma: no cover - runtime safety path
            failures += 1
            errors.append(str(exc))
    return {
        "requested": int(pulses),
        "completed": int(completed),
        "failures": int(failures),
        "errors": errors[:3],
    }


def _drift_percent(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    q = max(1, len(values) // 4)
    head = statistics.mean(values[:q])
    tail = statistics.mean(values[-q:])
    if head == 0.0:
        return 0.0
    return float(((tail - head) / head) * 100.0)


def _aggregate_group(rows: list[dict[str, Any]], kernel: str, size: int) -> dict[str, Any]:
    peak = [float(r["peak_mean_gflops"]) for r in rows]
    avg = [float(r["avg_mean_gflops"]) for r in rows]
    p95 = [float(r["p95_time_ms"]) for r in rows]
    max_error = [float(r["max_error_max"]) for r in rows]

    out: dict[str, Any] = {
        "kernel": kernel,
        "size": int(size),
        "batches": len(rows),
        "peak_mean_gflops": float(statistics.mean(peak)),
        "avg_mean_gflops": float(statistics.mean(avg)),
        "p95_time_ms": float(statistics.mean(p95)),
        "max_error_max": float(max(max_error)),
        "avg_gflops_cv": float(statistics.pstdev(avg) / statistics.mean(avg))
        if statistics.mean(avg) > 0.0
        else 0.0,
        "drift_percent": _drift_percent(avg),
    }
    if kernel == "auto_t3_controlled":
        fallback = [float(r["t3_fallback_rate"]) for r in rows]
        out["t3_fallback_rate_mean"] = float(statistics.mean(fallback))
        out["t3_fallback_rate_max"] = float(max(fallback))
        out["t3_policy_disabled_count"] = int(sum(int(r["t3_policy_disabled"]) for r in rows))
    if kernel == "auto_t5_guarded":
        overhead = [float(r["t5_overhead_percent"]) for r in rows]
        fp = [float(r["t5_false_positive_rate"]) for r in rows]
        disable = [int(r["t5_disable_events"]) for r in rows]
        out["t5_overhead_mean_percent"] = float(statistics.mean(overhead))
        out["t5_false_positive_rate_mean"] = float(statistics.mean(fp))
        out["t5_disable_events_total"] = int(sum(disable))
    return out


def _evaluate(report: dict[str, Any]) -> dict[str, Any]:
    groups = report["summary"]["groups"]
    pressure = report["summary"]["pressure"]
    t3_groups = [g for g in groups if g["kernel"] == "auto_t3_controlled"]
    t5_groups = [g for g in groups if g["kernel"] == "auto_t5_guarded"]

    max_t3_error = max(float(g["max_error_max"]) for g in t3_groups) if t3_groups else 1.0
    max_t5_error = max(float(g["max_error_max"]) for g in t5_groups) if t5_groups else 1.0
    max_t3_drift = max(abs(float(g["drift_percent"])) for g in t3_groups) if t3_groups else 999.0
    max_t5_drift = max(abs(float(g["drift_percent"])) for g in t5_groups) if t5_groups else 999.0
    t3_fallback_mean = (
        statistics.mean(float(g["t3_fallback_rate_mean"]) for g in t3_groups) if t3_groups else 1.0
    )
    t3_disabled_total = sum(int(g["t3_policy_disabled_count"]) for g in t3_groups)
    t5_overhead_mean = (
        statistics.mean(float(g["t5_overhead_mean_percent"]) for g in t5_groups) if t5_groups else 999.0
    )
    t5_fp_mean = (
        statistics.mean(float(g["t5_false_positive_rate_mean"]) for g in t5_groups)
        if t5_groups
        else 1.0
    )
    t5_disable_total = sum(int(g["t5_disable_events_total"]) for g in t5_groups)

    checks = {
        "pressure_failures_zero": {
            "observed": int(pressure["failures"]),
            "required": 0,
            "pass": int(pressure["failures"]) == 0,
        },
        "t3_correctness_bound": {
            "observed_max": max_t3_error,
            "required_max": 1e-3,
            "pass": max_t3_error <= 1e-3,
        },
        "t3_fallback_rate_mean": {
            "observed": float(t3_fallback_mean),
            "required_max": 0.08,
            "pass": float(t3_fallback_mean) <= 0.08,
        },
        "t3_policy_not_disabled": {
            "observed": int(t3_disabled_total),
            "required": 0,
            "pass": int(t3_disabled_total) == 0,
        },
        "t3_drift_abs_percent": {
            "observed": float(max_t3_drift),
            "required_max": 8.0,
            "pass": float(max_t3_drift) <= 8.0,
        },
        "t5_correctness_bound": {
            "observed_max": max_t5_error,
            "required_max": 1e-3,
            "pass": max_t5_error <= 1e-3,
        },
        "t5_overhead_mean_percent": {
            "observed": float(t5_overhead_mean),
            "required_max": 3.0,
            "pass": float(t5_overhead_mean) <= 3.0,
        },
        "t5_false_positive_rate_mean": {
            "observed": float(t5_fp_mean),
            "required_max": 0.05,
            "pass": float(t5_fp_mean) <= 0.05,
        },
        "t5_disable_events_zero": {
            "observed": int(t5_disable_total),
            "required": 0,
            "pass": int(t5_disable_total) == 0,
        },
        "t5_drift_abs_percent": {
            "observed": float(max_t5_drift),
            "required_max": 8.0,
            "pass": float(max_t5_drift) <= 8.0,
        },
    }

    failed = [name for name, payload in checks.items() if not payload["pass"]]
    if not checks["t3_correctness_bound"]["pass"] or not checks["t5_correctness_bound"]["pass"]:
        decision = "drop"
        rationale = "Correctness guard failed in long canary."
    elif failed:
        decision = "iterate"
        rationale = (
            "Long canary is operational but one or more drift/guardrail checks failed."
        )
    else:
        decision = "promote"
        rationale = (
            "Long canary passed queue-pressure, drift and guardrail checks for T3/T5."
        )
    return {
        "checks": checks,
        "failed_checks": failed,
        "decision": decision,
        "rationale": rationale,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 9 Block 1 - Long Mixed Canary Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Sizes: {report['metadata']['sizes']}")
    lines.append(
        f"- Batches={report['metadata']['batches']} | Sessions/batch={report['metadata']['sessions_per_batch']} | Iterations/session={report['metadata']['iterations_per_session']}"
    )
    lines.append("")
    lines.append("## Queue Pressure")
    lines.append("")
    p = report["summary"]["pressure"]
    lines.append(
        f"- Pulses requested/completed/failures: {p['requested']}/{p['completed']}/{p['failures']}"
    )
    lines.append("")
    lines.append("## Group Summary")
    lines.append("")
    lines.append(
        "| Kernel | Size | Avg GFLOPS | P95 ms | Max error | Drift % | Extra |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for g in report["summary"]["groups"]:
        extra = "-"
        if g["kernel"] == "auto_t3_controlled":
            extra = (
                f"fallback_mean={g['t3_fallback_rate_mean']:.3f}, "
                f"disabled={g['t3_policy_disabled_count']}"
            )
        elif g["kernel"] == "auto_t5_guarded":
            extra = (
                f"overhead={g['t5_overhead_mean_percent']:.3f}%, "
                f"fp={g['t5_false_positive_rate_mean']:.3f}, "
                f"disable={g['t5_disable_events_total']}"
            )
        lines.append(
            f"| {g['kernel']} | {g['size']} | {g['avg_mean_gflops']:.3f} | {g['p95_time_ms']:.3f} | {g['max_error_max']:.7f} | {g['drift_percent']:+.3f} | {extra} |"
        )
    lines.append("")
    lines.append("## Guardrail Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, payload in report["evaluation"]["checks"].items():
        lines.append(f"| {name} | {payload['pass']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['evaluation']['decision']}`")
    lines.append(f"- Rationale: {report['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_campaign(
    *,
    sizes: list[int],
    batches: int,
    sessions_per_batch: int,
    iterations_per_session: int,
    seed: int,
    pressure_size: int,
    pressure_iterations: int,
    pressure_pulses_per_batch: int,
    t5_policy_path: str | None,
    t5_state_path: str | None,
) -> dict[str, Any]:
    kernels = ["auto_t3_controlled", "auto_t5_guarded"]
    rows: list[dict[str, Any]] = []
    pressure_requested = 0
    pressure_completed = 0
    pressure_failures = 0
    pressure_errors: list[str] = []

    for batch in range(int(batches)):
        p = _queue_pressure(
            pulses=int(pressure_pulses_per_batch),
            size=int(pressure_size),
            iterations=int(pressure_iterations),
            seed=int(seed + batch * 1_000_000),
        )
        pressure_requested += int(p["requested"])
        pressure_completed += int(p["completed"])
        pressure_failures += int(p["failures"])
        pressure_errors.extend([str(x) for x in p["errors"]])

        for k_idx, kernel in enumerate(kernels):
            for s_idx, size in enumerate(sizes):
                run_seed = int(seed + batch * 100_000 + k_idx * 10_000 + s_idx * 100)
                extra: dict[str, Any] = {}
                if t5_policy_path not in (None, ""):
                    extra["t5_policy_path"] = str(t5_policy_path)
                if t5_state_path not in (None, ""):
                    extra["t5_state_path"] = str(t5_state_path)

                run = _run_silent_benchmark(
                    size=int(size),
                    iterations=int(iterations_per_session),
                    sessions=int(sessions_per_batch),
                    kernel=str(kernel),
                    seed=run_seed,
                    **extra,
                )
                summary = run["summary"]
                row: dict[str, Any] = {
                    "batch": int(batch),
                    "kernel": kernel,
                    "size": int(size),
                    "seed": run_seed,
                    "peak_mean_gflops": float(summary["peak_gflops"]["mean"]),
                    "avg_mean_gflops": float(summary["avg_gflops"]["mean"]),
                    "p95_time_ms": float(summary["time_ms"]["p95"]),
                    "max_error_max": float(summary["max_error"]["max"]),
                }
                if kernel == "auto_t3_controlled":
                    row["t3_fallback_rate"] = float(summary.get("fallback_rate", 0.0))
                    row["t3_policy_disabled"] = int(bool(summary.get("policy_disabled", False)))
                if kernel == "auto_t5_guarded":
                    t5 = summary.get("t5_abft", {})
                    row["t5_overhead_percent"] = float(t5.get("effective_overhead_percent", 0.0))
                    row["t5_false_positive_rate"] = float(t5.get("false_positive_rate", 0.0))
                    row["t5_disable_events"] = int(t5.get("disable_events", 0))
                    row["t5_disable_reason"] = t5.get("disable_reason")
                rows.append(row)

    groups: list[dict[str, Any]] = []
    for kernel in kernels:
        for size in sizes:
            g_rows = [r for r in rows if r["kernel"] == kernel and r["size"] == int(size)]
            groups.append(_aggregate_group(g_rows, kernel=kernel, size=int(size)))

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "branch": subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(REPO_ROOT),
                text=True,
            ).strip(),
            "sizes": [int(x) for x in sizes],
            "batches": int(batches),
            "sessions_per_batch": int(sessions_per_batch),
            "iterations_per_session": int(iterations_per_session),
            "seed": int(seed),
            "pressure": {
                "size": int(pressure_size),
                "iterations": int(pressure_iterations),
                "pulses_per_batch": int(pressure_pulses_per_batch),
            },
            "t5_policy_path": None if t5_policy_path in (None, "") else str(t5_policy_path),
            "t5_state_path": None if t5_state_path in (None, "") else str(t5_state_path),
            "horizon_label": "24h_equivalent_batches",
        },
        "rows": rows,
        "summary": {
            "groups": groups,
            "pressure": {
                "requested": int(pressure_requested),
                "completed": int(pressure_completed),
                "failures": int(pressure_failures),
                "errors": pressure_errors[:5],
            },
        },
    }
    report["evaluation"] = _evaluate(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week9 Block1 long mixed canary.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048])
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--sessions-per-batch", type=int, default=1)
    parser.add_argument("--iterations-per-session", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pressure-size", type=int, default=896)
    parser.add_argument("--pressure-iterations", type=int, default=3)
    parser.add_argument("--pressure-pulses-per-batch", type=int, default=2)
    parser.add_argument("--output-dir", default="research/breakthrough_lab")
    parser.add_argument(
        "--output-prefix",
        default="week9_block1_long_canary",
        help="Artifact prefix for output JSON/MD files.",
    )
    parser.add_argument(
        "--t5-policy-path",
        default=None,
        help="Optional T5 policy path passed to auto_t5_guarded runs.",
    )
    parser.add_argument(
        "--t5-state-path",
        default=None,
        help="Optional T5 runtime state path passed to auto_t5_guarded runs.",
    )
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_campaign(
        sizes=list(args.sizes),
        batches=int(args.batches),
        sessions_per_batch=int(args.sessions_per_batch),
        iterations_per_session=int(args.iterations_per_session),
        seed=int(args.seed),
        pressure_size=int(args.pressure_size),
        pressure_iterations=int(args.pressure_iterations),
        pressure_pulses_per_batch=int(args.pressure_pulses_per_batch),
        t5_policy_path=args.t5_policy_path,
        t5_state_path=args.t5_state_path,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{args.output_prefix}_{timestamp}.json"
    md_path = output_dir / f"{args.output_prefix}_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report))

    print(f"Week9 block1 JSON: {json_path}")
    print(f"Week9 block1 MD:   {md_path}")
    print(f"Decision: {report['evaluation']['decision']}")
    print(
        "Pressure failures: "
        f"{report['summary']['pressure']['failures']} / {report['summary']['pressure']['requested']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
