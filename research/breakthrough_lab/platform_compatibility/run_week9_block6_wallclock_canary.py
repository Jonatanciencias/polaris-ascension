#!/usr/bin/env python3
"""Week 9 Block 6: wall-clock canary before final production recommendation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_T5_POLICY = (
    "research/breakthrough_lab/t5_reliability_abft/policy_hardening_week9_block2.json"
)
DEFAULT_BLOCK5_BASELINE = (
    "research/breakthrough_lab/platform_compatibility/week9_block5_preprod_pilot_20260208_035240.json"
)


def _extract_json_payload(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _run_benchmark_subprocess(
    *,
    platform_selector: str,
    kernel: str,
    size: int,
    sessions: int,
    iterations: int,
    seed: int,
    t5_policy_path: str,
    t5_state_path: str,
    env_patch: dict[str, str] | None = None,
) -> dict[str, Any]:
    snippet = (
        "import json\n"
        "from src.benchmarking.production_kernel_benchmark import run_production_benchmark\n"
        f"report = run_production_benchmark(size={size}, sessions={sessions}, iterations={iterations}, "
        f"kernel={kernel!r}, seed={seed}, opencl_platform={platform_selector!r}, "
        f"t5_policy_path={t5_policy_path!r}, t5_state_path={t5_state_path!r})\n"
        "summary = report['summary']\n"
        "payload = {\n"
        "  'platform': report['metadata']['platform'],\n"
        "  'device': report['metadata']['device'],\n"
        "  'peak_mean_gflops': summary['peak_gflops']['mean'],\n"
        "  'avg_mean_gflops': summary['avg_gflops']['mean'],\n"
        "  'p95_time_ms': summary['time_ms']['p95'],\n"
        "  'max_error_max': summary['max_error']['max'],\n"
        "}\n"
        f"if {kernel!r} == 'auto_t3_controlled':\n"
        "  payload['t3_fallback_rate'] = summary.get('fallback_rate', 0.0)\n"
        "  payload['t3_policy_disabled'] = summary.get('policy_disabled', False)\n"
        f"if {kernel!r} == 'auto_t5_guarded':\n"
        "  t5 = summary.get('t5_abft', {})\n"
        "  payload['t5_overhead_percent'] = t5.get('effective_overhead_percent', 0.0)\n"
        "  payload['t5_false_positive_rate'] = t5.get('false_positive_rate', 0.0)\n"
        "  payload['t5_disable_events'] = t5.get('disable_events', 0)\n"
        "  payload['t5_disable_reason'] = t5.get('disable_reason')\n"
        "print(json.dumps(payload))\n"
    )
    env = os.environ.copy()
    if env_patch:
        env.update(env_patch)
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    payload = _extract_json_payload(proc.stdout)
    if proc.returncode != 0:
        return {
            "status": "error",
            "returncode": int(proc.returncode),
            "platform_selector": platform_selector,
            "kernel": kernel,
            "size": int(size),
            "seed": int(seed),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    if payload is None:
        return {
            "status": "error",
            "returncode": int(proc.returncode),
            "platform_selector": platform_selector,
            "kernel": kernel,
            "size": int(size),
            "seed": int(seed),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "parse_error": "json_payload_not_found",
        }
    return {
        "status": "ok",
        "returncode": int(proc.returncode),
        "platform_selector": platform_selector,
        "kernel": kernel,
        "size": int(size),
        "seed": int(seed),
        "metrics": payload,
    }


def _run_pressure_pulses(
    *,
    platform_selector: str,
    env_patch: dict[str, str] | None,
    size: int,
    iterations: int,
    pulses: int,
    seed: int,
) -> dict[str, Any]:
    completed = 0
    failures = 0
    errors: list[str] = []
    for idx in range(int(pulses)):
        snippet = (
            "from src.benchmarking.production_kernel_benchmark import run_production_benchmark\n"
            f"run_production_benchmark(size={size}, sessions=1, iterations={iterations}, "
            f"kernel='auto', seed={seed + idx * 17}, opencl_platform={platform_selector!r})\n"
            "print('ok')\n"
        )
        env = os.environ.copy()
        if env_patch:
            env.update(env_patch)
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            completed += 1
        else:
            failures += 1
            err = proc.stderr.strip() or proc.stdout.strip() or "unknown_error"
            errors.append(err)
    return {
        "requested": int(pulses),
        "completed": int(completed),
        "failures": int(failures),
        "errors": errors[:3],
    }


def _load_block5_baseline(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    payload = json.loads(path.read_text())
    runs = [
        r
        for r in payload.get("runs", [])
        if r.get("status") == "ok" and str(r.get("platform_selector", "")).lower() == "clover"
    ]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in runs:
        key = (str(run["kernel"]), int(run["size"]))
        grouped.setdefault(key, []).append(run)
    out: dict[tuple[str, int], dict[str, float]] = {}
    for key, entries in grouped.items():
        out[key] = {
            "avg_mean_gflops": float(
                statistics.mean(float(e["metrics"]["avg_mean_gflops"]) for e in entries)
            ),
            "p95_time_ms": float(
                statistics.mean(float(e["metrics"]["p95_time_ms"]) for e in entries)
            ),
        }
    return out


def _evaluate(report: dict[str, Any], baseline: dict[tuple[str, int], dict[str, float]]) -> dict[str, Any]:
    runs = report["runs"]
    pressure = report["pressure"]
    ok_runs = [r for r in runs if r["status"] == "ok"]

    all_ok = len(ok_runs) == len(runs) and len(runs) > 0
    pressure_failures_total = sum(int(p["failures"]) for p in pressure)
    pressure_ok = pressure_failures_total == 0

    max_error = max((float(r["metrics"]["max_error_max"]) for r in ok_runs), default=999.0)
    correctness_ok = max_error <= 1e-3

    t3_runs = [r for r in ok_runs if r["kernel"] == "auto_t3_controlled"]
    t5_runs = [r for r in ok_runs if r["kernel"] == "auto_t5_guarded"]

    t3_fallback_max = max((float(r["metrics"].get("t3_fallback_rate", 0.0)) for r in t3_runs), default=0.0)
    t3_policy_disabled_total = sum(
        int(bool(r["metrics"].get("t3_policy_disabled", False))) for r in t3_runs
    )
    t3_guard_ok = t3_fallback_max <= 0.08 and t3_policy_disabled_total == 0

    t5_disable_total = sum(int(r["metrics"].get("t5_disable_events", 0)) for r in t5_runs)
    t5_fp_max = max((float(r["metrics"].get("t5_false_positive_rate", 0.0)) for r in t5_runs), default=0.0)
    t5_overhead_max = max((float(r["metrics"].get("t5_overhead_percent", 0.0)) for r in t5_runs), default=0.0)
    t5_guard_ok = t5_disable_total == 0 and t5_fp_max <= 0.05 and t5_overhead_max <= 3.0

    clover_runs = [r for r in ok_runs if str(r["platform_selector"]).lower() == "clover"]
    rusticl_runs = [r for r in ok_runs if str(r["platform_selector"]).lower() == "rusticl"]
    split_ok = len(clover_runs) > 0 and len(rusticl_runs) > 0

    ratio_rows: list[dict[str, Any]] = []
    snapshots = report["metadata"]["snapshots"]
    for snapshot in snapshots:
        for size in report["metadata"]["sizes"]:
            for kernel in report["metadata"]["kernels"]:
                c = next(
                    (
                        r
                        for r in clover_runs
                        if int(r["snapshot"]) == int(snapshot)
                        and int(r["size"]) == int(size)
                        and str(r["kernel"]) == kernel
                    ),
                    None,
                )
                z = next(
                    (
                        r
                        for r in rusticl_runs
                        if int(r["snapshot"]) == int(snapshot)
                        and int(r["size"]) == int(size)
                        and str(r["kernel"]) == kernel
                    ),
                    None,
                )
                ratio = 0.0
                if c and z:
                    c_peak = float(c["metrics"]["peak_mean_gflops"])
                    z_peak = float(z["metrics"]["peak_mean_gflops"])
                    ratio = z_peak / c_peak if c_peak > 0.0 else 0.0
                ratio_rows.append(
                    {
                        "snapshot": int(snapshot),
                        "size": int(size),
                        "kernel": kernel,
                        "rusticl_peak_ratio_vs_clover": float(ratio),
                    }
                )
    min_ratio = min((r["rusticl_peak_ratio_vs_clover"] for r in ratio_rows), default=0.0)
    ratio_ok = min_ratio >= 0.80

    drift_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for run in ok_runs:
        key = (str(run["platform_selector"]), str(run["kernel"]), int(run["size"]))
        grouped.setdefault(key, []).append(run)
    for (platform, kernel, size), entries in grouped.items():
        entries_sorted = sorted(entries, key=lambda x: int(x["snapshot"]))
        first = float(entries_sorted[0]["metrics"]["avg_mean_gflops"])
        last = float(entries_sorted[-1]["metrics"]["avg_mean_gflops"])
        drift = 0.0 if first == 0.0 else (last - first) / first * 100.0
        drift_rows.append(
            {
                "platform": platform,
                "kernel": kernel,
                "size": int(size),
                "drift_percent": float(drift),
            }
        )
    drift_ok = all(abs(float(r["drift_percent"])) <= 15.0 for r in drift_rows)

    regression_rows: list[dict[str, Any]] = []
    grouped_clover: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in clover_runs:
        key = (str(run["kernel"]), int(run["size"]))
        grouped_clover.setdefault(key, []).append(run)
    for key, ref in baseline.items():
        entries = grouped_clover.get(key, [])
        if not entries:
            continue
        avg = statistics.mean(float(e["metrics"]["avg_mean_gflops"]) for e in entries)
        p95 = statistics.mean(float(e["metrics"]["p95_time_ms"]) for e in entries)
        delta_thr = 0.0 if ref["avg_mean_gflops"] == 0.0 else (avg - ref["avg_mean_gflops"]) / ref["avg_mean_gflops"] * 100.0
        delta_p95 = 0.0 if ref["p95_time_ms"] == 0.0 else (p95 - ref["p95_time_ms"]) / ref["p95_time_ms"] * 100.0
        regression_rows.append(
            {
                "kernel": key[0],
                "size": int(key[1]),
                "throughput_delta_percent": float(delta_thr),
                "p95_delta_percent": float(delta_p95),
            }
        )
    no_regression = all(
        float(r["throughput_delta_percent"]) >= -12.0 and float(r["p95_delta_percent"]) <= 25.0
        for r in regression_rows
    )

    wallclock_minutes = float(report["metadata"]["wallclock"]["actual_minutes"])
    target_minutes = float(report["metadata"]["wallclock"]["target_minutes"])
    wallclock_ok = wallclock_minutes >= target_minutes * 0.95

    checks = {
        "wallclock_duration_target": {
            "observed_minutes": float(wallclock_minutes),
            "required_min_minutes": float(target_minutes * 0.95),
            "pass": bool(wallclock_ok),
        },
        "all_runs_success": {"pass": bool(all_ok)},
        "pressure_failures_zero": {
            "observed": int(pressure_failures_total),
            "required": 0,
            "pass": bool(pressure_ok),
        },
        "platform_split_clover_and_rusticl": {"pass": bool(split_ok)},
        "correctness_bound_all_runs": {
            "observed_max": float(max_error),
            "required_max": 1e-3,
            "pass": bool(correctness_ok),
        },
        "t3_guardrails_all_runs": {
            "observed_fallback_max": float(t3_fallback_max),
            "observed_policy_disabled_total": int(t3_policy_disabled_total),
            "pass": bool(t3_guard_ok),
        },
        "t5_guardrails_all_runs": {
            "observed_disable_total": int(t5_disable_total),
            "observed_fp_max": float(t5_fp_max),
            "observed_overhead_max": float(t5_overhead_max),
            "pass": bool(t5_guard_ok),
        },
        "rusticl_peak_ratio_min": {
            "observed_min": float(min_ratio),
            "required_min": 0.80,
            "pass": bool(ratio_ok),
        },
        "drift_abs_percent_bounded": {
            "required_abs_max": 15.0,
            "pass": bool(drift_ok),
        },
        "no_regression_vs_block5_clover": {"pass": bool(no_regression)},
    }
    failed = [name for name, payload in checks.items() if not payload["pass"]]
    if not checks["correctness_bound_all_runs"]["pass"]:
        decision = "drop"
        rationale = "Correctness guard failed during wall-clock canary."
    elif failed:
        decision = "iterate"
        rationale = "Wall-clock canary found one or more guardrail/regression failures."
    else:
        decision = "promote"
        rationale = "Wall-clock canary passed with stable guardrails and platform split behavior."
    return {
        "checks": checks,
        "failed_checks": failed,
        "decision": decision,
        "rationale": rationale,
        "ratio_rows": ratio_rows,
        "drift_rows": drift_rows,
        "regression_rows": regression_rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    eval_data = report["evaluation"]
    lines: list[str] = []
    lines.append("# Week 9 Block 6 - Wall-Clock Long Canary")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(
        f"- Wall-clock target/actual (min): {report['metadata']['wallclock']['target_minutes']:.1f}/{report['metadata']['wallclock']['actual_minutes']:.1f}"
    )
    lines.append(f"- Snapshot interval (min): {report['metadata']['wallclock']['snapshot_interval_minutes']}")
    lines.append(f"- Snapshots: {report['metadata']['snapshots']}")
    lines.append("")
    lines.append("## Pressure Summary")
    lines.append("")
    lines.append("| Platform | Snapshot | Requested | Completed | Failures |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in report["pressure"]:
        lines.append(
            f"| {row['platform_selector']} | {row['snapshot']} | {row['requested']} | {row['completed']} | {row['failures']} |"
        )
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, payload in eval_data["checks"].items():
        lines.append(f"| {name} | {payload['pass']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{eval_data['decision']}`")
    lines.append(f"- Rationale: {eval_data['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_campaign(
    *,
    duration_minutes: float,
    snapshot_interval_minutes: float,
    sizes: list[int],
    kernels: list[str],
    sessions: int,
    iterations: int,
    pressure_size: int,
    pressure_iterations: int,
    pressure_pulses_per_snapshot: int,
    seed: int,
    t5_policy_path: str,
    baseline_block5_path: str,
    state_tag: str,
) -> dict[str, Any]:
    start_ts = time.time()
    end_ts = start_ts + float(duration_minutes) * 60.0
    interval_seconds = max(1.0, float(snapshot_interval_minutes) * 60.0)

    platforms: list[tuple[str, dict[str, str] | None, str]] = [
        ("Clover", None, f"results/runtime_states/t5_abft_guard_state_week9_block6_clover_{state_tag}.json"),
        (
            "rusticl",
            {"RUSTICL_ENABLE": "radeonsi"},
            f"results/runtime_states/t5_abft_guard_state_week9_block6_rusticl_{state_tag}.json",
        ),
    ]

    snapshots: list[int] = []
    pressure_rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []

    snapshot = 0
    while time.time() < end_ts:
        snapshot += 1
        snapshots.append(snapshot)
        snapshot_seed = int(seed + snapshot * 1000)

        for platform_selector, env_patch, state_path in platforms:
            pressure = _run_pressure_pulses(
                platform_selector=platform_selector,
                env_patch=env_patch,
                size=int(pressure_size),
                iterations=int(pressure_iterations),
                pulses=int(pressure_pulses_per_snapshot),
                seed=int(snapshot_seed + 50_000),
            )
            pressure_rows.append(
                {
                    "platform_selector": platform_selector,
                    "snapshot": int(snapshot),
                    **pressure,
                }
            )
            for kernel in kernels:
                for size in sizes:
                    run = _run_benchmark_subprocess(
                        platform_selector=platform_selector,
                        kernel=str(kernel),
                        size=int(size),
                        sessions=int(sessions),
                        iterations=int(iterations),
                        seed=int(snapshot_seed + size),
                        t5_policy_path=str(t5_policy_path),
                        t5_state_path=state_path,
                        env_patch=env_patch,
                    )
                    run["snapshot"] = int(snapshot)
                    runs.append(run)

        target_next = start_ts + snapshot * interval_seconds
        now = time.time()
        if now < target_next and now < end_ts:
            sleep_s = min(target_next - now, end_ts - now)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    actual_minutes = max(0.0, (time.time() - start_ts) / 60.0)
    baseline = _load_block5_baseline((REPO_ROOT / baseline_block5_path).resolve())
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sizes": [int(x) for x in sizes],
            "kernels": [str(x) for x in kernels],
            "sessions": int(sessions),
            "iterations": int(iterations),
            "seed": int(seed),
            "t5_policy_path": str(t5_policy_path),
            "baseline_block5_path": str(baseline_block5_path),
            "snapshots": snapshots,
            "wallclock": {
                "target_minutes": float(duration_minutes),
                "actual_minutes": float(actual_minutes),
                "snapshot_interval_minutes": float(snapshot_interval_minutes),
            },
            "pressure": {
                "size": int(pressure_size),
                "iterations": int(pressure_iterations),
                "pulses_per_snapshot": int(pressure_pulses_per_snapshot),
            },
        },
        "pressure": pressure_rows,
        "runs": runs,
    }
    report["evaluation"] = _evaluate(report, baseline)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week9 Block6 wall-clock canary.")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--snapshot-interval-minutes", type=float, default=5.0)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048])
    parser.add_argument("--kernels", nargs="+", default=["auto_t3_controlled", "auto_t5_guarded"])
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--pressure-size", type=int, default=896)
    parser.add_argument("--pressure-iterations", type=int, default=2)
    parser.add_argument("--pressure-pulses-per-snapshot", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2602)
    parser.add_argument("--t5-policy-path", default=DEFAULT_T5_POLICY)
    parser.add_argument("--baseline-block5-path", default=DEFAULT_BLOCK5_BASELINE)
    parser.add_argument("--output-dir", default="research/breakthrough_lab/platform_compatibility")
    parser.add_argument("--output-prefix", default="week9_block6_wallclock_canary")
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    report = run_campaign(
        duration_minutes=float(args.duration_minutes),
        snapshot_interval_minutes=float(args.snapshot_interval_minutes),
        sizes=list(args.sizes),
        kernels=list(args.kernels),
        sessions=int(args.sessions),
        iterations=int(args.iterations),
        pressure_size=int(args.pressure_size),
        pressure_iterations=int(args.pressure_iterations),
        pressure_pulses_per_snapshot=int(args.pressure_pulses_per_snapshot),
        seed=int(args.seed),
        t5_policy_path=str(args.t5_policy_path),
        baseline_block5_path=str(args.baseline_block5_path),
        state_tag=state_tag,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{args.output_prefix}_{timestamp}.json"
    md_path = output_dir / f"{args.output_prefix}_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report))

    print(f"Week9 block6 JSON: {json_path}")
    print(f"Week9 block6 MD:   {md_path}")
    print(f"Decision: {report['evaluation']['decision']}")
    print(f"Failed checks: {report['evaluation']['failed_checks']}")
    print(f"Actual wall-clock minutes: {report['metadata']['wallclock']['actual_minutes']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
