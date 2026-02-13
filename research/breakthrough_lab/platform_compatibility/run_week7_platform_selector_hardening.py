#!/usr/bin/env python3
"""Week 7 Block 1: explicit OpenCL platform selector hardening."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    body = text.strip()
    if not body:
        return None
    try:
        payload = json.loads(body)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start = body.find("{")
        if start < 0:
            return None
        try:
            payload = json.loads(body[start:])
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None


def _run_one(
    *,
    platform_selector: str,
    size: int,
    sessions: int,
    iterations: int,
    seed: int,
    env_patch: dict[str, str] | None = None,
) -> dict[str, Any]:
    snippet = (
        "import json\n"
        "from src.benchmarking.production_kernel_benchmark import run_production_benchmark\n"
        f"report = run_production_benchmark(size={size}, sessions={sessions}, iterations={iterations}, "
        f"kernel='auto', seed={seed}, opencl_platform={platform_selector!r})\n"
        "payload = {\n"
        "  'platform': report['metadata']['platform'],\n"
        "  'device': report['metadata']['device'],\n"
        "  'platform_selection': report['metadata'].get('platform_selection', {}),\n"
        "  'peak_mean_gflops': report['summary']['peak_gflops']['mean'],\n"
        "  'avg_mean_gflops': report['summary']['avg_gflops']['mean'],\n"
        "  'max_error_max': report['summary']['max_error']['max'],\n"
        "}\n"
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
    if proc.returncode != 0:
        return {
            "selector": platform_selector,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "status": "error",
        }
    metrics = _extract_json_payload(proc.stdout)
    if metrics is None:
        return {
            "selector": platform_selector,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "status": "error",
            "error": "benchmark stdout did not contain valid JSON payload",
        }

    return {
        "selector": platform_selector,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "status": "ok",
        "metrics": metrics,
    }


def _evaluate(clover: dict[str, Any], rusticl: dict[str, Any]) -> dict[str, Any]:
    clover_ok = clover["status"] == "ok" and str(clover["metrics"]["platform"]).lower() == "clover"
    rusticl_ok = (
        rusticl["status"] == "ok" and str(rusticl["metrics"]["platform"]).lower() == "rusticl"
    )

    clover_peak = float(clover["metrics"]["peak_mean_gflops"]) if clover_ok else 0.0
    rusticl_peak = float(rusticl["metrics"]["peak_mean_gflops"]) if rusticl_ok else 0.0
    peak_ratio = float(rusticl_peak / clover_peak) if clover_peak > 0 else 0.0

    clover_error_ok = clover_ok and float(clover["metrics"]["max_error_max"]) <= 1e-3
    rusticl_error_ok = rusticl_ok and float(rusticl["metrics"]["max_error_max"]) <= 1e-3
    perf_ratio_ok = peak_ratio >= 0.85

    checks = {
        "clover_explicit_selection": {
            "pass": bool(clover_ok),
            "observed": clover.get("metrics", {}),
        },
        "rusticl_canary_selection": {
            "pass": bool(rusticl_ok),
            "observed": rusticl.get("metrics", {}),
        },
        "correctness_bound_clover": {
            "pass": bool(clover_error_ok),
            "observed": clover.get("metrics", {}).get("max_error_max"),
            "required_max": 1e-3,
        },
        "correctness_bound_rusticl": {
            "pass": bool(rusticl_error_ok),
            "observed": rusticl.get("metrics", {}).get("max_error_max"),
            "required_max": 1e-3,
        },
        "rusticl_peak_ratio_vs_clover": {
            "pass": bool(perf_ratio_ok),
            "observed": peak_ratio,
            "required_min": 0.85,
        },
    }
    failed = [name for name, payload in checks.items() if not payload["pass"]]
    decision = "promote" if not failed else "iterate"
    rationale = (
        "Explicit platform selection works for Clover and Rusticl canary with bounded correctness."
        if decision == "promote"
        else "One or more selector hardening checks failed; keep canary in refine mode."
    )
    return {"checks": checks, "failed_checks": failed, "decision": decision, "rationale": rationale}


def _markdown(report: dict[str, Any]) -> str:
    eval_data = report["evaluation"]
    clover = report["runs"]["clover"]
    rusticl = report["runs"]["rusticl"]
    lines: list[str] = []
    lines.append("# Week 7 Block 1 - Platform Selector Hardening Report")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Size: {report['metadata']['size']}")
    lines.append(
        f"- Sessions/Iterations: {report['metadata']['sessions']}/{report['metadata']['iterations']}"
    )
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append("| Route | Status | Platform | Peak mean GFLOPS | Max error |")
    lines.append("| --- | --- | --- | ---: | ---: |")
    for label, run in [("Clover explicit", clover), ("Rusticl canary", rusticl)]:
        metrics = run.get("metrics", {})
        lines.append(
            f"| {label} | {run['status']} | {metrics.get('platform', 'n/a')} | "
            f"{float(metrics.get('peak_mean_gflops', 0.0)):.3f} | {float(metrics.get('max_error_max', 0.0)):.7f} |"
        )
    lines.append("")
    lines.append("## Guardrail Checks")
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


def run_campaign(*, size: int, sessions: int, iterations: int, seed: int) -> dict[str, Any]:
    clover = _run_one(
        platform_selector="Clover",
        size=size,
        sessions=sessions,
        iterations=iterations,
        seed=seed,
    )
    rusticl = _run_one(
        platform_selector="rusticl",
        size=size,
        sessions=sessions,
        iterations=iterations,
        seed=seed,
        env_patch={"RUSTICL_ENABLE": "radeonsi"},
    )
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "size": int(size),
            "sessions": int(sessions),
            "iterations": int(iterations),
            "seed": int(seed),
        },
        "runs": {"clover": clover, "rusticl": rusticl},
    }
    report["evaluation"] = _evaluate(clover, rusticl)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week7 Block1 platform selector hardening.")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/platform_compatibility",
    )
    args = parser.parse_args()

    report = run_campaign(
        size=int(args.size),
        sessions=int(args.sessions),
        iterations=int(args.iterations),
        seed=int(args.seed),
    )
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week7_platform_selector_hardening_{timestamp}.json"
    md_path = output_dir / f"week7_platform_selector_hardening_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(_markdown(report))

    print(f"Week7 selector JSON: {json_path}")
    print(f"Week7 selector MD:   {md_path}")
    print(f"Decision: {report['evaluation']['decision']}")
    print(f"Failed checks: {report['evaluation']['failed_checks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
