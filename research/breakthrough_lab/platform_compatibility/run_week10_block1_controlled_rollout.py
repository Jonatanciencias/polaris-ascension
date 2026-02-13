#!/usr/bin/env python3
"""Week 10 Block 1: controlled low-scope rollout with hourly snapshots and auto rollback."""

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
DEFAULT_BLOCK6_BASELINE = (
    "research/breakthrough_lab/platform_compatibility/week9_block6_wallclock_canary_20260208_043949.json"
)
DEFAULT_ROLLBACK_SCRIPT = (
    "research/breakthrough_lab/platform_compatibility/week9_block5_rusticl_rollback.sh"
)
DEFAULT_ROLLBACK_SLA = (
    "research/breakthrough_lab/preprod_signoff/WEEK9_BLOCK6_ROLLBACK_SLA.md"
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
    kernel: str,
    size: int,
    sessions: int,
    iterations: int,
    seed: int,
    t5_policy_path: str,
    t5_state_path: str,
) -> dict[str, Any]:
    snippet = (
        "import json\n"
        "from src.benchmarking.production_kernel_benchmark import run_production_benchmark\n"
        f"report = run_production_benchmark(size={size}, sessions={sessions}, iterations={iterations}, "
        f"kernel={kernel!r}, seed={seed}, opencl_platform='Clover', "
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
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    payload = _extract_json_payload(proc.stdout)
    if proc.returncode != 0:
        return {
            "status": "error",
            "returncode": int(proc.returncode),
            "platform_selector": "Clover",
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
            "platform_selector": "Clover",
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
        "platform_selector": "Clover",
        "kernel": kernel,
        "size": int(size),
        "seed": int(seed),
        "metrics": payload,
    }


def _run_pressure_pulses(*, size: int, iterations: int, pulses: int, seed: int) -> dict[str, Any]:
    completed = 0
    failures = 0
    errors: list[str] = []
    for idx in range(int(pulses)):
        snippet = (
            "from src.benchmarking.production_kernel_benchmark import run_production_benchmark\n"
            f"run_production_benchmark(size={size}, sessions=1, iterations={iterations}, "
            f"kernel='auto', seed={seed + idx * 17}, opencl_platform='Clover')\n"
            "print('ok')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=str(REPO_ROOT),
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


def _load_block6_baseline(path: Path) -> dict[tuple[str, int], dict[str, float]]:
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


def _evaluate_snapshot(
    *,
    snapshot_runs: list[dict[str, Any]],
    pressure: dict[str, Any],
    baseline: dict[tuple[str, int], dict[str, float]],
) -> dict[str, Any]:
    ok_runs = [r for r in snapshot_runs if r["status"] == "ok"]
    all_ok = len(ok_runs) == len(snapshot_runs) and len(snapshot_runs) > 0
    pressure_ok = int(pressure.get("failures", 0)) == 0

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
    t5_overhead_soft_limit = 3.0
    t5_overhead_hard_limit = 5.0
    t5_overhead_soft_violation = t5_overhead_max > t5_overhead_soft_limit
    t5_overhead_hard_violation = t5_overhead_max > t5_overhead_hard_limit
    t5_guard_hard_ok = (
        t5_disable_total == 0 and t5_fp_max <= 0.05 and not t5_overhead_hard_violation
    )

    regression_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in ok_runs:
        key = (str(run["kernel"]), int(run["size"]))
        grouped.setdefault(key, []).append(run)
    for key, ref in baseline.items():
        rows = grouped.get(key, [])
        if not rows:
            continue
        avg = statistics.mean(float(r["metrics"]["avg_mean_gflops"]) for r in rows)
        p95 = statistics.mean(float(r["metrics"]["p95_time_ms"]) for r in rows)
        thr_delta = 0.0 if ref["avg_mean_gflops"] == 0 else (avg - ref["avg_mean_gflops"]) / ref["avg_mean_gflops"] * 100.0
        p95_delta = 0.0 if ref["p95_time_ms"] == 0 else (p95 - ref["p95_time_ms"]) / ref["p95_time_ms"] * 100.0
        regression_rows.append(
            {
                "kernel": key[0],
                "size": int(key[1]),
                "throughput_delta_percent": float(thr_delta),
                "p95_delta_percent": float(p95_delta),
            }
        )
    no_regression = all(
        float(r["throughput_delta_percent"]) >= -8.0 and float(r["p95_delta_percent"]) <= 15.0
        for r in regression_rows
    )

    checks = {
        "all_runs_success": {"pass": bool(all_ok)},
        "pressure_failures_zero": {
            "observed": int(pressure.get("failures", 0)),
            "required": 0,
            "pass": bool(pressure_ok),
        },
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
        "t5_guardrails_hard": {
            "observed_disable_total": int(t5_disable_total),
            "observed_fp_max": float(t5_fp_max),
            "observed_overhead_max": float(t5_overhead_max),
            "soft_limit_percent": float(t5_overhead_soft_limit),
            "hard_limit_percent": float(t5_overhead_hard_limit),
            "pass": bool(t5_guard_hard_ok),
        },
        "t5_overhead_soft_limit": {
            "observed_overhead_max": float(t5_overhead_max),
            "required_max": float(t5_overhead_soft_limit),
            "pass": not bool(t5_overhead_soft_violation),
        },
        "no_regression_vs_block6_clover": {"pass": bool(no_regression)},
    }
    hard_check_names = [
        "all_runs_success",
        "pressure_failures_zero",
        "correctness_bound_all_runs",
        "t3_guardrails_all_runs",
        "t5_guardrails_hard",
        "no_regression_vs_block6_clover",
    ]
    hard_failed = [name for name in hard_check_names if not checks[name]["pass"]]
    failed = [name for name, payload in checks.items() if not payload["pass"]]
    if hard_failed:
        decision = "rollback"
    elif t5_overhead_soft_violation:
        decision = "warn"
    else:
        decision = "continue"
    return {
        "checks": checks,
        "failed_checks": failed,
        "hard_failed_checks": hard_failed,
        "soft_overhead_violation": bool(t5_overhead_soft_violation),
        "decision": decision,
        "regression_rows": regression_rows,
    }


def _run_rollback(rollback_script_path: str) -> dict[str, Any]:
    path = (REPO_ROOT / rollback_script_path).resolve()
    if not path.exists():
        return {
            "invoked": False,
            "success": False,
            "reason": "rollback_script_not_found",
            "script_path": str(path),
        }
    proc = subprocess.run(
        [str(path), "apply"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    stdout_tail = "\n".join(proc.stdout.splitlines()[-20:])
    stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
    return {
        "invoked": True,
        "success": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "script_path": str(path),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def _build_drift_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_runs = [r for r in runs if r.get("status") == "ok"]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in ok_runs:
        key = (str(run["kernel"]), int(run["size"]))
        grouped.setdefault(key, []).append(run)
    rows: list[dict[str, Any]] = []
    for (kernel, size), entries in grouped.items():
        ordered = sorted(entries, key=lambda x: int(x["snapshot"]))
        first = float(ordered[0]["metrics"]["avg_mean_gflops"])
        last = float(ordered[-1]["metrics"]["avg_mean_gflops"])
        drift = 0.0 if first == 0.0 else (last - first) / first * 100.0
        rows.append(
            {
                "kernel": kernel,
                "size": int(size),
                "first_avg_gflops": float(first),
                "last_avg_gflops": float(last),
                "drift_percent": float(drift),
            }
        )
    return rows


def _evaluate_campaign(report: dict[str, Any]) -> dict[str, Any]:
    snapshots = report["snapshots"]
    hard_failures = [s for s in snapshots if s["evaluation"]["decision"] == "rollback"]
    warnings = [s for s in snapshots if s["evaluation"]["decision"] == "warn"]
    rollback_event = report.get("rollback_event")
    rollback_triggered = rollback_event is not None
    soft_threshold = int(report["metadata"]["rollback_after_consecutive_soft_overhead_violations"])
    max_soft_streak = max(
        (int(s["evaluation"].get("soft_overhead_streak", 0)) for s in snapshots),
        default=0,
    )

    drift_rows = _build_drift_rows(report["runs"])
    drift_ok = all(abs(float(r["drift_percent"])) <= 10.0 for r in drift_rows) if drift_rows else False
    all_snapshot_hard_checks_pass = len(hard_failures) == 0 and len(snapshots) > 0
    soft_overhead_limit_respected = max_soft_streak < soft_threshold
    enough_snapshots = int(report["metadata"]["executed_snapshots"]) >= 2
    rollback_expected = len(hard_failures) > 0 or max_soft_streak >= soft_threshold
    rollback_policy_enforced = (not rollback_expected and not rollback_triggered) or (
        rollback_expected and rollback_triggered
    )

    checks = {
        "minimum_snapshots_completed": {
            "observed": int(report["metadata"]["executed_snapshots"]),
            "required_min": 2,
            "pass": bool(enough_snapshots),
        },
        "all_snapshot_hard_checks_passed": {"pass": bool(all_snapshot_hard_checks_pass)},
        "soft_overhead_consecutive_below_limit": {
            "observed_max_streak": int(max_soft_streak),
            "required_lt": int(soft_threshold),
            "pass": bool(soft_overhead_limit_respected),
        },
        "rollback_policy_enforced": {"pass": bool(rollback_policy_enforced)},
        "drift_abs_percent_bounded": {
            "required_abs_max": 10.0,
            "pass": bool(drift_ok),
        },
    }
    failed = [name for name, payload in checks.items() if not payload["pass"]]

    if rollback_triggered:
        rollback_failed_checks = rollback_event.get("failed_checks", [])
        if "correctness_bound_all_runs" in rollback_failed_checks:
            decision = "stop"
            rationale = "Auto rollback was triggered by correctness breach during controlled rollout."
        else:
            decision = "iterate"
            rationale = "Auto rollback triggered by guardrail breach; rollout remains in iterate mode."
    elif failed:
        decision = "iterate"
        rationale = "Controlled rollout completed but campaign-level gates require refinement."
    else:
        decision = "promote"
        rationale = "Low-scope rollout passed hourly snapshots with stable guardrails and no rollback trigger."

    return {
        "checks": checks,
        "failed_checks": failed,
        "decision": decision,
        "rationale": rationale,
        "drift_rows": drift_rows,
        "warning_snapshots": [int(s["snapshot"]) for s in warnings],
        "hard_failure_snapshots": [int(s["snapshot"]) for s in hard_failures],
    }


def _markdown(report: dict[str, Any]) -> str:
    eval_data = report["evaluation"]
    lines: list[str] = []
    lines.append("# Week 10 Block 1 - Controlled Low-Scope Rollout")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Planned snapshots: {report['metadata']['planned_snapshots']}")
    lines.append(f"- Executed snapshots: {report['metadata']['executed_snapshots']}")
    lines.append(f"- Hourly interval (logical): {report['metadata']['snapshot_interval_minutes']} minutes")
    lines.append(f"- Auto rollback enabled: {report['metadata']['auto_rollback_enabled']}")
    lines.append("")
    lines.append("## Snapshot Decisions")
    lines.append("")
    lines.append("| Snapshot | Decision | Failed checks |")
    lines.append("| ---: | --- | --- |")
    for snapshot in report["snapshots"]:
        failed = snapshot["evaluation"]["failed_checks"]
        lines.append(f"| {snapshot['snapshot']} | {snapshot['evaluation']['decision']} | {', '.join(failed) if failed else '-'} |")
    lines.append("")
    lines.append("## Campaign Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, payload in eval_data["checks"].items():
        lines.append(f"| {name} | {payload['pass']} |")
    lines.append("")
    if report.get("rollback_event"):
        rollback = report["rollback_event"]["rollback_result"]
        lines.append("## Auto Rollback Event")
        lines.append("")
        lines.append(f"- Trigger snapshot: {report['rollback_event']['snapshot']}")
        lines.append(f"- Trigger checks: {report['rollback_event']['failed_checks']}")
        lines.append(f"- Rollback invoked: {rollback.get('invoked')}")
        lines.append(f"- Rollback success: {rollback.get('success')}")
        lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{eval_data['decision']}`")
    lines.append(f"- Rationale: {eval_data['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_campaign(
    *,
    snapshots: int,
    snapshot_interval_minutes: float,
    sleep_between_snapshots_seconds: float,
    sizes: list[int],
    kernels: list[str],
    sessions: int,
    iterations: int,
    pressure_size: int,
    pressure_iterations: int,
    pressure_pulses_per_snapshot: int,
    seed: int,
    t5_policy_path: str,
    baseline_block6_path: str,
    rollback_script_path: str,
    rollback_sla_path: str,
    auto_rollback: bool,
    rollback_after_consecutive_soft_overhead_violations: int,
    state_tag: str,
) -> dict[str, Any]:
    baseline = _load_block6_baseline((REPO_ROOT / baseline_block6_path).resolve())
    t5_state_path = f"results/runtime_states/t5_abft_guard_state_week10_block1_clover_{state_tag}.json"

    snapshot_rows: list[dict[str, Any]] = []
    pressure_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    rollback_event: dict[str, Any] | None = None
    soft_overhead_streak = 0

    for snapshot in range(1, int(snapshots) + 1):
        snapshot_seed = int(seed + snapshot * 1000)
        pressure = _run_pressure_pulses(
            size=int(pressure_size),
            iterations=int(pressure_iterations),
            pulses=int(pressure_pulses_per_snapshot),
            seed=snapshot_seed,
        )
        pressure_row = {"snapshot": int(snapshot), **pressure}
        pressure_rows.append(pressure_row)

        snapshot_runs: list[dict[str, Any]] = []
        for kernel in kernels:
            for size in sizes:
                run = _run_benchmark_subprocess(
                    kernel=str(kernel),
                    size=int(size),
                    sessions=int(sessions),
                    iterations=int(iterations),
                    seed=int(snapshot_seed + size),
                    t5_policy_path=str(t5_policy_path),
                    t5_state_path=t5_state_path,
                )
                run["snapshot"] = int(snapshot)
                snapshot_runs.append(run)
                run_rows.append(run)

        snapshot_eval = _evaluate_snapshot(
            snapshot_runs=snapshot_runs,
            pressure=pressure_row,
            baseline=baseline,
        )
        snapshot_rows.append(
            {
                "snapshot": int(snapshot),
                "pressure": pressure_row,
                "runs": snapshot_runs,
                "evaluation": snapshot_eval,
            }
        )

        if snapshot_eval.get("soft_overhead_violation", False):
            soft_overhead_streak += 1
        else:
            soft_overhead_streak = 0
        snapshot_rows[-1]["evaluation"]["soft_overhead_streak"] = int(soft_overhead_streak)

        if snapshot_eval["decision"] == "rollback" and auto_rollback:
            rollback_result = _run_rollback(rollback_script_path)
            rollback_event = {
                "snapshot": int(snapshot),
                "failed_checks": snapshot_eval.get("hard_failed_checks", snapshot_eval["failed_checks"]),
                "trigger_type": "hard_guardrail",
                "rollback_result": rollback_result,
            }
            break
        if (
            auto_rollback
            and soft_overhead_streak >= int(rollback_after_consecutive_soft_overhead_violations)
        ):
            rollback_result = _run_rollback(rollback_script_path)
            rollback_event = {
                "snapshot": int(snapshot),
                "failed_checks": ["t5_overhead_soft_limit_consecutive"],
                "trigger_type": "soft_overhead_consecutive",
                "rollback_result": rollback_result,
            }
            break

        if sleep_between_snapshots_seconds > 0 and snapshot < int(snapshots):
            time.sleep(float(sleep_between_snapshots_seconds))

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "planned_snapshots": int(snapshots),
            "executed_snapshots": int(len(snapshot_rows)),
            "snapshot_interval_minutes": float(snapshot_interval_minutes),
            "sleep_between_snapshots_seconds": float(sleep_between_snapshots_seconds),
            "sizes": [int(x) for x in sizes],
            "kernels": [str(x) for x in kernels],
            "sessions": int(sessions),
            "iterations": int(iterations),
            "seed": int(seed),
            "pressure": {
                "size": int(pressure_size),
                "iterations": int(pressure_iterations),
                "pulses_per_snapshot": int(pressure_pulses_per_snapshot),
            },
            "t5_policy_path": str(t5_policy_path),
            "baseline_block6_path": str(baseline_block6_path),
            "rollback_script_path": str(rollback_script_path),
            "rollback_sla_path": str(rollback_sla_path),
            "auto_rollback_enabled": bool(auto_rollback),
            "rollback_after_consecutive_soft_overhead_violations": int(
                rollback_after_consecutive_soft_overhead_violations
            ),
        },
        "snapshots": snapshot_rows,
        "pressure": pressure_rows,
        "runs": run_rows,
        "rollback_event": rollback_event,
    }
    report["evaluation"] = _evaluate_campaign(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week10 Block1 controlled low-scope rollout.")
    parser.add_argument("--snapshots", type=int, default=3, help="Logical hourly snapshots.")
    parser.add_argument("--snapshot-interval-minutes", type=float, default=60.0)
    parser.add_argument("--sleep-between-snapshots-seconds", type=float, default=0.0)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400])
    parser.add_argument("--kernels", nargs="+", default=["auto_t3_controlled", "auto_t5_guarded"])
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--pressure-size", type=int, default=896)
    parser.add_argument("--pressure-iterations", type=int, default=2)
    parser.add_argument("--pressure-pulses-per-snapshot", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2610)
    parser.add_argument("--t5-policy-path", default=DEFAULT_T5_POLICY)
    parser.add_argument("--baseline-block6-path", default=DEFAULT_BLOCK6_BASELINE)
    parser.add_argument("--rollback-script-path", default=DEFAULT_ROLLBACK_SCRIPT)
    parser.add_argument("--rollback-sla-path", default=DEFAULT_ROLLBACK_SLA)
    parser.add_argument(
        "--rollback-after-consecutive-soft-overhead-violations",
        type=int,
        default=2,
    )
    parser.add_argument("--disable-auto-rollback", action="store_true")
    parser.add_argument("--output-dir", default="research/breakthrough_lab/platform_compatibility")
    parser.add_argument("--output-prefix", default="week10_block1_controlled_rollout")
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    report = run_campaign(
        snapshots=int(args.snapshots),
        snapshot_interval_minutes=float(args.snapshot_interval_minutes),
        sleep_between_snapshots_seconds=float(args.sleep_between_snapshots_seconds),
        sizes=list(args.sizes),
        kernels=list(args.kernels),
        sessions=int(args.sessions),
        iterations=int(args.iterations),
        pressure_size=int(args.pressure_size),
        pressure_iterations=int(args.pressure_iterations),
        pressure_pulses_per_snapshot=int(args.pressure_pulses_per_snapshot),
        seed=int(args.seed),
        t5_policy_path=str(args.t5_policy_path),
        baseline_block6_path=str(args.baseline_block6_path),
        rollback_script_path=str(args.rollback_script_path),
        rollback_sla_path=str(args.rollback_sla_path),
        auto_rollback=not bool(args.disable_auto_rollback),
        rollback_after_consecutive_soft_overhead_violations=int(
            args.rollback_after_consecutive_soft_overhead_violations
        ),
        state_tag=state_tag,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{args.output_prefix}_{timestamp}.json"
    md_path = output_dir / f"{args.output_prefix}_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report))

    print(f"Week10 block1 JSON: {json_path}")
    print(f"Week10 block1 MD:   {md_path}")
    print(f"Decision: {report['evaluation']['decision']}")
    print(f"Failed checks: {report['evaluation']['failed_checks']}")
    print(f"Rollback triggered: {report.get('rollback_event') is not None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
