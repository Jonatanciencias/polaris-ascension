#!/usr/bin/env python3
"""Week 8 Block 6: short platform canary on critical sizes (Clover vs rusticl)."""

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


def _run_one(
    *,
    platform_selector: str,
    kernel: str,
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
        f"kernel={kernel!r}, seed={seed}, opencl_platform={platform_selector!r})\n"
        "summary = report['summary']\n"
        "payload = {\n"
        "  'platform': report['metadata']['platform'],\n"
        "  'device': report['metadata']['device'],\n"
        "  'platform_selection': report['metadata'].get('platform_selection', {}),\n"
        "  'peak_mean_gflops': summary['peak_gflops']['mean'],\n"
        "  'avg_mean_gflops': summary['avg_gflops']['mean'],\n"
        "  'p95_time_ms': summary['time_ms']['p95'],\n"
        "  'max_error_max': summary['max_error']['max'],\n"
        "}\n"
        f"if {kernel!r} == 'auto_t3_controlled':\n"
        "  payload['t3_fallback_rate'] = summary.get('fallback_rate', 0.0)\n"
        f"if {kernel!r} == 'auto_t5_guarded':\n"
        "  t5 = summary.get('t5_abft', {})\n"
        "  payload['t5_overhead_percent'] = t5.get('effective_overhead_percent', 0.0)\n"
        "  payload['t5_false_positive_rate'] = t5.get('false_positive_rate', 0.0)\n"
        "  payload['t5_disable_events'] = t5.get('disable_events', 0)\n"
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
    payload: dict[str, Any] | None = None
    body = proc.stdout.strip()
    if body:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            start = body.find("{")
            if start >= 0:
                try:
                    payload = json.loads(body[start:])
                except json.JSONDecodeError:
                    payload = None
    if proc.returncode != 0:
        return {
            "status": "error",
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "platform_selector": platform_selector,
            "kernel": kernel,
            "size": int(size),
        }
    if payload is None:
        return {
            "status": "error",
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "platform_selector": platform_selector,
            "kernel": kernel,
            "size": int(size),
            "parse_error": "json_payload_not_found",
        }
    return {
        "status": "ok",
        "returncode": int(proc.returncode),
        "platform_selector": platform_selector,
        "kernel": kernel,
        "size": int(size),
        "metrics": payload,
    }


def _evaluate(report: dict[str, Any]) -> dict[str, Any]:
    runs = report["runs"]
    clover_runs = [r for r in runs if r["platform_selector"] == "Clover"]
    rusticl_runs = [r for r in runs if r["platform_selector"] == "rusticl"]

    clover_ok = all(
        r["status"] == "ok"
        and str(r["metrics"]["platform"]).lower() == "clover"
        for r in clover_runs
    )
    rusticl_ok = all(
        r["status"] == "ok"
        and str(r["metrics"]["platform"]).lower() == "rusticl"
        for r in rusticl_runs
    )

    all_max_error = [
        float(r["metrics"]["max_error_max"]) for r in runs if r["status"] == "ok"
    ]
    correctness_ok = all(x <= 1e-3 for x in all_max_error)

    ratio_rows: list[dict[str, Any]] = []
    for size in report["metadata"]["sizes"]:
        for kernel in report["metadata"]["kernels"]:
            c = next(
                (r for r in clover_runs if r["size"] == size and r["kernel"] == kernel and r["status"] == "ok"),
                None,
            )
            r = next(
                (r for r in rusticl_runs if r["size"] == size and r["kernel"] == kernel and r["status"] == "ok"),
                None,
            )
            ratio = 0.0
            if c and r and float(c["metrics"]["peak_mean_gflops"]) > 0:
                ratio = float(r["metrics"]["peak_mean_gflops"]) / float(
                    c["metrics"]["peak_mean_gflops"]
                )
            ratio_rows.append(
                {"size": int(size), "kernel": kernel, "rusticl_peak_ratio_vs_clover": ratio}
            )
    min_ratio = min((row["rusticl_peak_ratio_vs_clover"] for row in ratio_rows), default=0.0)
    ratio_ok = min_ratio >= 0.80

    t3_guard_ok = all(
        float(r["metrics"].get("t3_fallback_rate", 0.0)) <= 0.10
        for r in runs
        if r["status"] == "ok" and r["kernel"] == "auto_t3_controlled"
    )
    t5_guard_ok = all(
        int(r["metrics"].get("t5_disable_events", 0)) == 0
        and float(r["metrics"].get("t5_false_positive_rate", 0.0)) <= 0.05
        and float(r["metrics"].get("t5_overhead_percent", 0.0)) <= 3.0
        for r in runs
        if r["status"] == "ok" and r["kernel"] == "auto_t5_guarded"
    )

    checks = {
        "clover_selection_all_runs": {"pass": bool(clover_ok)},
        "rusticl_selection_all_runs": {"pass": bool(rusticl_ok)},
        "correctness_bound_all_runs": {
            "observed_max": max(all_max_error) if all_max_error else None,
            "required_max": 1e-3,
            "pass": bool(correctness_ok),
        },
        "rusticl_peak_ratio_min": {
            "observed_min": float(min_ratio),
            "required_min": 0.80,
            "pass": bool(ratio_ok),
        },
        "t3_guardrails_all_platforms": {"pass": bool(t3_guard_ok)},
        "t5_guardrails_all_platforms": {"pass": bool(t5_guard_ok)},
    }
    failed = [name for name, payload in checks.items() if not payload["pass"]]
    decision = "promote" if not failed else "iterate"
    rationale = (
        "Critical-size short canary validates both platforms with bounded correctness and guardrails."
        if decision == "promote"
        else "At least one platform canary guardrail failed; keep canary limited."
    )
    return {
        "checks": checks,
        "failed_checks": failed,
        "decision": decision,
        "rationale": rationale,
        "ratio_rows": ratio_rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    eval_data = report["evaluation"]
    lines: list[str] = []
    lines.append("# Week 8 Block 6 - Platform Canary (Critical Sizes)")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Sizes: {report['metadata']['sizes']}")
    lines.append(f"- Kernels: {report['metadata']['kernels']}")
    lines.append("")
    lines.append("## Run Matrix")
    lines.append("")
    lines.append("| Platform selector | Kernel | Size | Status | Peak mean GFLOPS | P95 ms | Max error |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: |")
    for run in report["runs"]:
        metrics = run.get("metrics", {})
        lines.append(
            f"| {run['platform_selector']} | {run['kernel']} | {run['size']} | {run['status']} | "
            f"{float(metrics.get('peak_mean_gflops', 0.0)):.3f} | {float(metrics.get('p95_time_ms', 0.0)):.3f} | "
            f"{float(metrics.get('max_error_max', 0.0)):.7f} |"
        )
    lines.append("")
    lines.append("## Rusticl/Clover Peak Ratio")
    lines.append("")
    lines.append("| Size | Kernel | Ratio |")
    lines.append("| ---: | --- | ---: |")
    for row in eval_data["ratio_rows"]:
        lines.append(
            f"| {row['size']} | {row['kernel']} | {row['rusticl_peak_ratio_vs_clover']:.3f} |"
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
    sizes: list[int],
    kernels: list[str],
    sessions: int,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    platforms: list[tuple[str, dict[str, str] | None]] = [
        ("Clover", None),
        ("rusticl", {"RUSTICL_ENABLE": "radeonsi"}),
    ]
    for p_idx, (platform_selector, env_patch) in enumerate(platforms):
        for k_idx, kernel in enumerate(kernels):
            for s_idx, size in enumerate(sizes):
                run_seed = int(seed + p_idx * 1_000_000 + k_idx * 10_000 + s_idx * 100)
                runs.append(
                    _run_one(
                        platform_selector=platform_selector,
                        kernel=kernel,
                        size=int(size),
                        sessions=int(sessions),
                        iterations=int(iterations),
                        seed=run_seed,
                        env_patch=env_patch,
                    )
                )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sizes": [int(x) for x in sizes],
            "kernels": kernels,
            "sessions": int(sessions),
            "iterations": int(iterations),
            "seed": int(seed),
        },
        "runs": runs,
    }
    report["evaluation"] = _evaluate(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week8 critical-size platform canary.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048])
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=["auto", "auto_t3_controlled", "auto_t5_guarded"],
    )
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/platform_compatibility",
    )
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_campaign(
        sizes=list(args.sizes),
        kernels=list(args.kernels),
        sessions=int(args.sessions),
        iterations=int(args.iterations),
        seed=int(args.seed),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"week8_platform_canary_critical_{timestamp}.json"
    md_path = output_dir / f"week8_platform_canary_critical_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report))

    print(f"Week8 platform canary JSON: {json_path}")
    print(f"Week8 platform canary MD:   {md_path}")
    print(f"Decision: {report['evaluation']['decision']}")
    print(f"Failed checks: {report['evaluation']['failed_checks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
