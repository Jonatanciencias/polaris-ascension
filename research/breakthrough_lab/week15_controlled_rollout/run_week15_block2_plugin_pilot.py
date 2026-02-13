#!/usr/bin/env python3
"""Week 15 Block 2: first real plugin pilot using Week14 template contracts."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run(command: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True)
    return {
        "command": command,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _extract_line_value(stdout: str, prefix: str) -> str | None:
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(prefix):
            value = line[len(prefix) :].strip()
            return value if value else None
    return None


def _extract_json_payload(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _run_benchmark(
    *,
    kernel: str,
    size: int,
    sessions: int,
    iterations: int,
    seed: int,
    t5_policy_path: str,
) -> dict[str, Any]:
    snippet = (
        "import json\n"
        "from src.benchmarking.production_kernel_benchmark import run_production_benchmark\n"
        f"report = run_production_benchmark(size={size}, sessions={sessions}, iterations={iterations}, "
        f"kernel={kernel!r}, seed={seed}, opencl_platform='Clover', "
        f"t5_policy_path={t5_policy_path!r}, "
        "t5_state_path='results/runtime_states/t5_abft_guard_state_week15_block2_plugin.json')\n"
        "summary = report['summary']\n"
        "payload = {\n"
        "  'platform': report['metadata']['platform'],\n"
        "  'device': report['metadata']['device'],\n"
        "  'driver': report['metadata'].get('driver'),\n"
        "  'peak_mean_gflops': summary['peak_gflops']['mean'],\n"
        "  'avg_mean_gflops': summary['avg_gflops']['mean'],\n"
        "  'p95_time_ms': summary['time_ms']['p95'],\n"
        "  'max_error_max': summary['max_error']['max']\n"
        "}\n"
        f"if {kernel!r} == 'auto_t3_controlled':\n"
        "  payload['t3_fallback_rate'] = summary.get('fallback_rate', 0.0)\n"
        "  payload['t3_policy_disabled'] = summary.get('policy_disabled', False)\n"
        f"if {kernel!r} == 'auto_t5_guarded':\n"
        "  t5 = summary.get('t5_abft', {})\n"
        "  payload['t5_overhead_percent'] = t5.get('effective_overhead_percent', 0.0)\n"
        "  payload['t5_disable_events'] = t5.get('disable_events', 0)\n"
        "print(json.dumps(payload))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    payload = _extract_json_payload(proc.stdout)
    if proc.returncode != 0 or payload is None:
        return {
            "status": "error",
            "kernel": kernel,
            "size": int(size),
            "seed": int(seed),
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    return {
        "status": "ok",
        "kernel": kernel,
        "size": int(size),
        "seed": int(seed),
        "metrics": payload,
    }


def _load_baseline(path: Path) -> dict[tuple[str, int], float]:
    payload = json.loads(path.read_text())
    runs = payload.get("runs", [])
    grouped: dict[tuple[str, int], list[float]] = {}
    for run in runs:
        if run.get("status") != "ok":
            continue
        if str(run.get("platform_selector", "")).lower() != "clover":
            continue
        key = (str(run.get("kernel")), int(run.get("size")))
        grouped.setdefault(key, []).append(float(run["metrics"]["avg_mean_gflops"]))
    out: dict[tuple[str, int], float] = {}
    for key, values in grouped.items():
        out[key] = float(statistics.mean(values))
    return out


def _md_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 15 Block 2 - Plugin Pilot")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Plugin ID: `{payload['metadata']['plugin_id']}`")
    lines.append(f"- Owner: `{payload['metadata']['owner']}`")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("| Kernel | Size | Avg GFLOPS | P95 ms | Max error |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for run in payload["runs"]:
        if run["status"] != "ok":
            continue
        metrics = run["metrics"]
        lines.append(
            f"| {run['kernel']} | {run['size']} | {metrics['avg_mean_gflops']:.3f} | {metrics['p95_time_ms']:.3f} | {metrics['max_error_max']:.7f} |"
        )
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for key, check in payload["evaluation"]["checks"].items():
        lines.append(f"| {key} | {check['pass']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{payload['evaluation']['decision']}`")
    lines.append(f"- Failed checks: {payload['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {payload['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week15 Block2 plugin pilot.")
    parser.add_argument("--plugin-id", default="rx590_plugin_pilot_v1")
    parser.add_argument("--owner", default="gpu-lab")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048])
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["auto_t3_controlled", "auto_t5_guarded"],
    )
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=19011)
    parser.add_argument(
        "--t5-policy-path",
        default="research/breakthrough_lab/t5_reliability_abft/policy_hardening_week15_block1_expanded_sizes.json",
    )
    parser.add_argument(
        "--baseline-path",
        default="research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_rerun_canary_20260210_011736.json",
    )
    parser.add_argument(
        "--template-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK14_BLOCK6_PLUGIN_TEMPLATE.md",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week15_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week15_block2_plugin_pilot")
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_gate = _run(
        [
            "./venv/bin/python",
            "scripts/run_validation_suite.py",
            "--tier",
            "canonical",
            "--driver-smoke",
            "--report-dir",
            "research/breakthrough_lab/week8_validation_discipline",
        ],
        cwd=REPO_ROOT,
    )
    pre_gate_json = _extract_line_value(pre_gate["stdout"], "Wrote JSON report:")

    driver_smoke = _run(
        ["./venv/bin/python", "scripts/verify_drivers.py", "--json"],
        cwd=REPO_ROOT,
    )
    driver_payload = _extract_json_payload(driver_smoke["stdout"]) if driver_smoke["returncode"] == 0 else None

    runs: list[dict[str, Any]] = []
    for idx, size in enumerate(args.sizes):
        for kernel in args.kernels:
            run = _run_benchmark(
                kernel=str(kernel),
                size=int(size),
                sessions=int(args.sessions),
                iterations=int(args.iterations),
                seed=int(args.seed + idx * 1000 + size),
                t5_policy_path=str(args.t5_policy_path),
            )
            runs.append(run)

    post_gate = _run(
        [
            "./venv/bin/python",
            "scripts/run_validation_suite.py",
            "--tier",
            "canonical",
            "--driver-smoke",
            "--report-dir",
            "research/breakthrough_lab/week8_validation_discipline",
        ],
        cwd=REPO_ROOT,
    )
    post_gate_json = _extract_line_value(post_gate["stdout"], "Wrote JSON report:")

    ok_runs = [run for run in runs if run["status"] == "ok"]
    peak_values = [float(run["metrics"]["peak_mean_gflops"]) for run in ok_runs]
    avg_values = [float(run["metrics"]["avg_mean_gflops"]) for run in ok_runs]
    p95_values = [float(run["metrics"]["p95_time_ms"]) for run in ok_runs]
    max_errors = [float(run["metrics"]["max_error_max"]) for run in ok_runs]
    cv_peak = (
        float(statistics.pstdev(peak_values) / statistics.mean(peak_values))
        if len(peak_values) >= 2 and statistics.mean(peak_values) > 0
        else 0.0
    )
    reproducibility_score = max(0.0, min(1.0, 1.0 - (cv_peak / 0.05)))

    baseline = _load_baseline((REPO_ROOT / args.baseline_path).resolve())
    deltas: list[float] = []
    for run in ok_runs:
        key = (str(run["kernel"]), int(run["size"]))
        base = baseline.get(key)
        if base and base > 0:
            deltas.append((float(run["metrics"]["avg_mean_gflops"]) - base) / base * 100.0)
    delta_vs_baseline = float(statistics.mean(deltas)) if deltas else 0.0

    t3_fallback_max = max(
        (
            float(run["metrics"].get("t3_fallback_rate", 0.0))
            for run in ok_runs
            if run["kernel"] == "auto_t3_controlled"
        ),
        default=0.0,
    )
    t5_disable_total = sum(
        int(run["metrics"].get("t5_disable_events", 0))
        for run in ok_runs
        if run["kernel"] == "auto_t5_guarded"
    )
    max_error = max(max_errors) if max_errors else 999.0
    all_runs_ok = len(ok_runs) == len(runs) and len(runs) > 0

    pre_decision = "unknown"
    post_decision = "unknown"
    if pre_gate_json:
        pre_decision = str(
            _read_json((REPO_ROOT / pre_gate_json).resolve()).get("evaluation", {}).get("decision", "unknown")
        )
    if post_gate_json:
        post_decision = str(
            _read_json((REPO_ROOT / post_gate_json).resolve()).get("evaluation", {}).get("decision", "unknown")
        )

    checks: dict[str, dict[str, Any]] = {}
    checks["template_exists"] = {
        "observed": (REPO_ROOT / args.template_path).exists(),
        "required": True,
        "pass": (REPO_ROOT / args.template_path).exists(),
    }
    checks["driver_smoke_good"] = {
        "observed": str((driver_payload or {}).get("overall_status", "unknown")),
        "required": "good",
        "pass": str((driver_payload or {}).get("overall_status", "")) == "good",
    }
    checks["all_runs_success"] = {
        "observed": all_runs_ok,
        "required": True,
        "pass": all_runs_ok,
    }
    checks["correctness_bound"] = {
        "observed_max": float(max_error),
        "required_max": 1e-3,
        "pass": max_error <= 1e-3,
    }
    checks["t3_fallback_bound"] = {
        "observed_max": float(t3_fallback_max),
        "required_max": 0.08,
        "pass": t3_fallback_max <= 0.08,
    }
    checks["t5_disable_zero"] = {
        "observed": int(t5_disable_total),
        "required": 0,
        "pass": t5_disable_total == 0,
    }
    checks["pre_gate_promote"] = {
        "observed": pre_decision,
        "required": "promote",
        "pass": pre_decision == "promote",
    }
    checks["post_gate_promote"] = {
        "observed": post_decision,
        "required": "promote",
        "pass": post_decision == "promote",
    }

    failed_checks = [name for name, check in checks.items() if not bool(check.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Plugin pilot satisfies template contracts and performance/correctness guardrails."
        if decision == "promote"
        else "Plugin pilot did not satisfy one or more mandatory contracts/guardrails."
    )

    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%d_%H%M%S")
    timestamp_utc = timestamp.isoformat()

    report_payload = {
        "metadata": {
            "timestamp_utc": timestamp_utc,
            "plugin_id": args.plugin_id,
            "owner": args.owner,
            "template_path": str((REPO_ROOT / args.template_path).resolve()),
            "sizes": [int(s) for s in args.sizes],
            "kernels": [str(k) for k in args.kernels],
            "sessions": int(args.sessions),
            "iterations": int(args.iterations),
            "seed": int(args.seed),
            "t5_policy_path": str((REPO_ROOT / args.t5_policy_path).resolve()),
            "baseline_path": str((REPO_ROOT / args.baseline_path).resolve()),
        },
        "commands": {
            "pre_gate": pre_gate,
            "driver_smoke": driver_smoke,
            "post_gate": post_gate,
        },
        "runs": runs,
        "metrics": {
            "peak_gflops_mean": float(statistics.mean(peak_values)) if peak_values else 0.0,
            "avg_gflops_mean": float(statistics.mean(avg_values)) if avg_values else 0.0,
            "p95_time_ms": float(statistics.mean(p95_values)) if p95_values else 0.0,
            "max_error": float(max_error),
            "cv_peak": float(cv_peak),
            "delta_vs_baseline_percent": float(delta_vs_baseline),
            "reproducibility_score": float(reproducibility_score),
        },
        "artifacts": {
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
        },
        "evaluation": {
            "checks": checks,
            "failed_checks": failed_checks,
            "decision": decision,
            "rationale": rationale,
        },
    }

    report_json = out_dir / f"{args.output_prefix}_{stamp}.json"
    report_md = out_dir / f"{args.output_prefix}_{stamp}.md"
    _write_json(report_json, report_payload)
    report_md.write_text(_md_report(report_payload))

    plugin_results = {
        "$schema": "https://radeon-rx580.local/breakthrough/results.schema.json",
        "schema_version": "1.0.0",
        "experiment_id": f"week15-block2-{args.plugin_id}-{stamp}",
        "track": "t3_online_control",
        "title": "Week15 Block2 plugin pilot baseline",
        "status": "completed",
        "owner": args.owner,
        "branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT), text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip(),
        "timestamps": {
            "created_at": timestamp_utc,
            "started_at": timestamp_utc,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        },
        "environment": {
            "os": sys.platform,
            "python": sys.version.split()[0],
            "opencl_platform": str((driver_payload or {}).get("opencl", {}).get("platform", "unknown")),
            "opencl_device": str((driver_payload or {}).get("opencl", {}).get("device", "unknown")),
            "driver": str((driver_payload or {}).get("opencl", {}).get("driver", "unknown")),
        },
        "baseline_reference": {
            "protocol_doc": str((REPO_ROOT / args.template_path).resolve()),
            "baseline_commit": "2335707",
            "sizes": [int(s) for s in args.sizes],
        },
        "parameters": {
            "plugin_id": args.plugin_id,
            "kernels": [str(k) for k in args.kernels],
            "sessions": int(args.sessions),
            "iterations": int(args.iterations),
            "seed": int(args.seed),
            "t5_policy_path": str((REPO_ROOT / args.t5_policy_path).resolve()),
        },
        "metrics": report_payload["metrics"],
        "validation": {
            "correctness_passed": checks["correctness_bound"]["pass"],
            "stability_passed": checks["all_runs_success"]["pass"] and checks["t5_disable_zero"]["pass"],
            "promotion_gate_passed": decision == "promote",
        },
        "artifacts": [
            str(report_json),
            str(report_md),
            str((REPO_ROOT / args.template_path).resolve()),
        ],
        "decision": decision,
        "notes": rationale,
    }
    results_json = out_dir / "week15_block2_plugin_pilot_results.json"
    _write_json(results_json, plugin_results)

    print(f"Week15 block2 JSON: {report_json}")
    print(f"Week15 block2 MD:   {report_md}")
    print(f"Week15 block2 results.json: {results_json}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
