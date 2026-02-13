#!/usr/bin/env python3
"""Week 9 Block 3: robustness replay with alternate seeds and short platform split."""

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
DEFAULT_T5_POLICY = (
    "research/breakthrough_lab/t5_reliability_abft/policy_hardening_week9_block2.json"
)
DEFAULT_BLOCK2_BASELINE = (
    "research/breakthrough_lab/week9_block2_long_canary_rerun_20260208_032017.json"
)


def _extract_json_payload(stdout: str) -> dict[str, Any] | None:
    body = stdout.strip()
    if not body:
        return None
    lines = [line.strip() for line in body.splitlines() if line.strip()]
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
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _run_one(
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
        "  'platform_selection': report['metadata'].get('platform_selection', {}),\n"
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
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "platform_selector": platform_selector,
            "kernel": kernel,
            "size": int(size),
            "seed": int(seed),
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
            "seed": int(seed),
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


def _load_block2_baseline(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    payload = json.loads(path.read_text())
    groups = payload.get("summary", {}).get("groups", [])
    out: dict[tuple[str, int], dict[str, float]] = {}
    for group in groups:
        key = (str(group["kernel"]), int(group["size"]))
        out[key] = {
            "avg_mean_gflops": float(group["avg_mean_gflops"]),
            "p95_time_ms": float(group["p95_time_ms"]),
        }
    return out


def _evaluate(report: dict[str, Any], baseline: dict[tuple[str, int], dict[str, float]]) -> dict[str, Any]:
    runs = report["runs"]
    ok_runs = [r for r in runs if r["status"] == "ok"]

    all_ok = len(ok_runs) == len(runs) and len(runs) > 0
    max_error = max((float(r["metrics"]["max_error_max"]) for r in ok_runs), default=999.0)
    correctness_ok = max_error <= 1e-3

    t3_runs = [r for r in ok_runs if r["kernel"] == "auto_t3_controlled"]
    t5_runs = [r for r in ok_runs if r["kernel"] == "auto_t5_guarded"]

    t3_fallback_max = max((float(r["metrics"].get("t3_fallback_rate", 0.0)) for r in t3_runs), default=0.0)
    t3_policy_disabled_count = sum(
        int(bool(r["metrics"].get("t3_policy_disabled", False))) for r in t3_runs
    )
    t3_guard_ok = t3_fallback_max <= 0.08 and t3_policy_disabled_count == 0

    t5_disable_total = sum(int(r["metrics"].get("t5_disable_events", 0)) for r in t5_runs)
    t5_fp_max = max((float(r["metrics"].get("t5_false_positive_rate", 0.0)) for r in t5_runs), default=0.0)
    t5_overhead_max = max((float(r["metrics"].get("t5_overhead_percent", 0.0)) for r in t5_runs), default=0.0)
    t5_guard_ok = t5_disable_total == 0 and t5_fp_max <= 0.05 and t5_overhead_max <= 3.0

    by_platform = {
        "Clover": [r for r in ok_runs if str(r["platform_selector"]).lower() == "clover"],
        "rusticl": [r for r in ok_runs if str(r["platform_selector"]).lower() == "rusticl"],
    }
    split_ok = len(by_platform["Clover"]) > 0 and len(by_platform["rusticl"]) > 0

    ratio_rows: list[dict[str, Any]] = []
    for seed in report["metadata"]["seeds"]:
        for size in report["metadata"]["sizes"]:
            for kernel in report["metadata"]["kernels"]:
                c = next(
                    (
                        r
                        for r in by_platform["Clover"]
                        if int(r["seed"]) == int(seed)
                        and int(r["size"]) == int(size)
                        and str(r["kernel"]) == kernel
                    ),
                    None,
                )
                z = next(
                    (
                        r
                        for r in by_platform["rusticl"]
                        if int(r["seed"]) == int(seed)
                        and int(r["size"]) == int(size)
                        and str(r["kernel"]) == kernel
                    ),
                    None,
                )
                ratio = 0.0
                if c and z:
                    c_peak = float(c["metrics"].get("peak_mean_gflops", 0.0))
                    z_peak = float(z["metrics"].get("peak_mean_gflops", 0.0))
                    ratio = z_peak / c_peak if c_peak > 0.0 else 0.0
                ratio_rows.append(
                    {
                        "seed": int(seed),
                        "size": int(size),
                        "kernel": kernel,
                        "rusticl_peak_ratio_vs_clover": float(ratio),
                    }
                )
    min_ratio = min((row["rusticl_peak_ratio_vs_clover"] for row in ratio_rows), default=0.0)
    ratio_ok = min_ratio >= 0.80

    regression_rows: list[dict[str, Any]] = []
    clover_grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in by_platform["Clover"]:
        key = (str(run["kernel"]), int(run["size"]))
        clover_grouped.setdefault(key, []).append(run)

    for key, block2_ref in baseline.items():
        kernel, size = key
        entries = clover_grouped.get(key, [])
        if not entries:
            continue
        avg_mean = statistics.mean(float(e["metrics"]["avg_mean_gflops"]) for e in entries)
        p95_mean = statistics.mean(float(e["metrics"]["p95_time_ms"]) for e in entries)
        throughput_delta_percent = (
            (avg_mean - block2_ref["avg_mean_gflops"]) / block2_ref["avg_mean_gflops"] * 100.0
            if block2_ref["avg_mean_gflops"] > 0.0
            else 0.0
        )
        p95_delta_percent = (
            (p95_mean - block2_ref["p95_time_ms"]) / block2_ref["p95_time_ms"] * 100.0
            if block2_ref["p95_time_ms"] > 0.0
            else 0.0
        )
        regression_rows.append(
            {
                "kernel": kernel,
                "size": int(size),
                "avg_mean_gflops_replay_clover": float(avg_mean),
                "avg_mean_gflops_block2_ref": float(block2_ref["avg_mean_gflops"]),
                "throughput_delta_percent": float(throughput_delta_percent),
                "p95_time_ms_replay_clover": float(p95_mean),
                "p95_time_ms_block2_ref": float(block2_ref["p95_time_ms"]),
                "p95_delta_percent": float(p95_delta_percent),
            }
        )

    t5_regr_rows = [r for r in regression_rows if r["kernel"] == "auto_t5_guarded"]
    no_t5_regression = all(
        float(r["throughput_delta_percent"]) >= -5.0 and float(r["p95_delta_percent"]) <= 8.0
        for r in t5_regr_rows
    )

    checks = {
        "all_runs_success": {"pass": bool(all_ok)},
        "platform_split_clover_and_rusticl": {"pass": bool(split_ok)},
        "correctness_bound_all_runs": {
            "observed_max": float(max_error),
            "required_max": 1e-3,
            "pass": bool(correctness_ok),
        },
        "t3_guardrails_all_runs": {
            "observed_fallback_max": float(t3_fallback_max),
            "observed_policy_disabled_count": int(t3_policy_disabled_count),
            "required_fallback_max": 0.08,
            "required_policy_disabled_count": 0,
            "pass": bool(t3_guard_ok),
        },
        "t5_guardrails_all_runs": {
            "observed_disable_total": int(t5_disable_total),
            "observed_fp_max": float(t5_fp_max),
            "observed_overhead_max": float(t5_overhead_max),
            "required_disable_total": 0,
            "required_fp_max": 0.05,
            "required_overhead_max": 3.0,
            "pass": bool(t5_guard_ok),
        },
        "rusticl_peak_ratio_min": {
            "observed_min": float(min_ratio),
            "required_min": 0.80,
            "pass": bool(ratio_ok),
        },
        "t5_no_regression_vs_block2_clover": {
            "required_throughput_delta_min_percent": -5.0,
            "required_p95_delta_max_percent": 8.0,
            "pass": bool(no_t5_regression),
        },
    }
    failed = [name for name, payload in checks.items() if not payload["pass"]]
    if not checks["correctness_bound_all_runs"]["pass"]:
        decision = "drop"
        rationale = "Correctness guard failed during robustness replay."
    elif failed:
        decision = "iterate"
        rationale = "Robustness replay found at least one guardrail/platform regression issue."
    else:
        decision = "promote"
        rationale = "Alternate-seed replay and short platform split passed with no post-hardening regressions."
    return {
        "checks": checks,
        "failed_checks": failed,
        "decision": decision,
        "rationale": rationale,
        "ratio_rows": ratio_rows,
        "regression_rows": regression_rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    evaluation = report["evaluation"]
    lines: list[str] = []
    lines.append("# Week 9 Block 3 - Robustness Replay (Seeds + Platform Split)")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Seeds: {report['metadata']['seeds']}")
    lines.append(f"- Sizes: {report['metadata']['sizes']}")
    lines.append(f"- Kernels: {report['metadata']['kernels']}")
    lines.append(f"- Sessions: {report['metadata']['sessions']} | Iterations: {report['metadata']['iterations']}")
    lines.append("")
    lines.append("## Run Matrix")
    lines.append("")
    lines.append("| Platform | Seed | Kernel | Size | Status | Avg GFLOPS | P95 ms | Max error | T5 overhead % | T5 disable |")
    lines.append("| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for run in report["runs"]:
        metrics = run.get("metrics", {})
        lines.append(
            f"| {run['platform_selector']} | {run['seed']} | {run['kernel']} | {run['size']} | {run['status']} | "
            f"{float(metrics.get('avg_mean_gflops', 0.0)):.3f} | {float(metrics.get('p95_time_ms', 0.0)):.3f} | "
            f"{float(metrics.get('max_error_max', 0.0)):.7f} | {float(metrics.get('t5_overhead_percent', 0.0)):.3f} | "
            f"{int(metrics.get('t5_disable_events', 0))} |"
        )
    lines.append("")
    lines.append("## Rusticl vs Clover Ratio")
    lines.append("")
    lines.append("| Seed | Kernel | Size | Ratio |")
    lines.append("| ---: | --- | ---: | ---: |")
    for row in evaluation["ratio_rows"]:
        lines.append(
            f"| {row['seed']} | {row['kernel']} | {row['size']} | {row['rusticl_peak_ratio_vs_clover']:.3f} |"
        )
    lines.append("")
    lines.append("## Regression vs Week9 Block2 (Clover Aggregate)")
    lines.append("")
    lines.append("| Kernel | Size | Throughput delta % | P95 delta % |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in evaluation["regression_rows"]:
        lines.append(
            f"| {row['kernel']} | {row['size']} | {row['throughput_delta_percent']:+.3f} | {row['p95_delta_percent']:+.3f} |"
        )
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, payload in evaluation["checks"].items():
        lines.append(f"| {name} | {payload['pass']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{evaluation['decision']}`")
    lines.append(f"- Rationale: {evaluation['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_campaign(
    *,
    seeds: list[int],
    sizes: list[int],
    kernels: list[str],
    sessions: int,
    iterations: int,
    t5_policy_path: str,
    t5_state_tag: str,
    baseline_path: str,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    platforms: list[tuple[str, dict[str, str] | None, str]] = [
        ("Clover", None, f"results/runtime_states/t5_abft_guard_state_week9_block3_clover_{t5_state_tag}.json"),
        (
            "rusticl",
            {"RUSTICL_ENABLE": "radeonsi"},
            f"results/runtime_states/t5_abft_guard_state_week9_block3_rusticl_{t5_state_tag}.json",
        ),
    ]
    for _p_idx, (platform_selector, env_patch, state_path) in enumerate(platforms):
        for seed in seeds:
            for _k_idx, kernel in enumerate(kernels):
                for _s_idx, size in enumerate(sizes):
                    run_seed = int(seed)
                    runs.append(
                        _run_one(
                            platform_selector=platform_selector,
                            kernel=str(kernel),
                            size=int(size),
                            sessions=int(sessions),
                            iterations=int(iterations),
                            seed=run_seed,
                            t5_policy_path=str(t5_policy_path),
                            t5_state_path=state_path,
                            env_patch=env_patch,
                        )
                    )

    baseline = _load_block2_baseline((REPO_ROOT / baseline_path).resolve())
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": [int(x) for x in seeds],
            "sizes": [int(x) for x in sizes],
            "kernels": [str(x) for x in kernels],
            "sessions": int(sessions),
            "iterations": int(iterations),
            "t5_policy_path": str(t5_policy_path),
            "baseline_block2_path": str(baseline_path),
            "state_tag": str(t5_state_tag),
        },
        "runs": runs,
    }
    report["evaluation"] = _evaluate(report, baseline)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week9 Block3 robustness replay.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 1337])
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048])
    parser.add_argument("--kernels", nargs="+", default=["auto_t3_controlled", "auto_t5_guarded"])
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--t5-policy-path", default=DEFAULT_T5_POLICY)
    parser.add_argument("--baseline-block2-path", default=DEFAULT_BLOCK2_BASELINE)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/platform_compatibility",
    )
    parser.add_argument("--output-prefix", default="week9_block3_robustness_replay")
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = run_campaign(
        seeds=list(args.seeds),
        sizes=list(args.sizes),
        kernels=list(args.kernels),
        sessions=int(args.sessions),
        iterations=int(args.iterations),
        t5_policy_path=str(args.t5_policy_path),
        t5_state_tag=state_tag,
        baseline_path=str(args.baseline_block2_path),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{args.output_prefix}_{timestamp}.json"
    md_path = output_dir / f"{args.output_prefix}_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report))

    print(f"Week9 block3 JSON: {json_path}")
    print(f"Week9 block3 MD:   {md_path}")
    print(f"Decision: {report['evaluation']['decision']}")
    print(f"Failed checks: {report['evaluation']['failed_checks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
