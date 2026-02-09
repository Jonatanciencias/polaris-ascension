#!/usr/bin/env python3
"""Evaluate Week 12 platform split replay against weekly formal policy."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _key(kernel: str, size: int) -> str:
    return f"{kernel}:{size}"


def _evaluate(
    split_report: dict[str, Any],
    policy: dict[str, Any],
    *,
    min_rusticl_ratio: float,
) -> dict[str, Any]:
    runs = [r for r in split_report.get("runs", []) if r.get("status") == "ok"]
    clover = [r for r in runs if str(r.get("platform_selector", "")).lower() == "clover"]
    rusticl = [r for r in runs if str(r.get("platform_selector", "")).lower() == "rusticl"]

    checks: dict[str, dict[str, Any]] = {}
    checks["all_runs_success"] = {
        "pass": len(runs) == len(split_report.get("runs", [])) and len(runs) > 0
    }
    checks["platform_split_present"] = {"pass": len(clover) > 0 and len(rusticl) > 0}

    guardrails = policy["weekly_slo"]["global_guardrails"]
    max_error = max((float(r["metrics"]["max_error_max"]) for r in runs), default=999.0)
    checks["correctness_error_bound"] = {
        "observed_max": float(max_error),
        "required_max": float(guardrails["max_correctness_error"]),
        "pass": float(max_error) <= float(guardrails["max_correctness_error"]),
    }

    t3_fallback_max = max(
        (
            float(r["metrics"].get("t3_fallback_rate", 0.0))
            for r in runs
            if str(r.get("kernel")) == "auto_t3_controlled"
        ),
        default=0.0,
    )
    checks["t3_fallback_bound"] = {
        "observed_max": float(t3_fallback_max),
        "required_max": float(guardrails["max_t3_fallback_rate"]),
        "pass": float(t3_fallback_max) <= float(guardrails["max_t3_fallback_rate"]),
    }

    t5_overhead_max = max(
        (
            float(r["metrics"].get("t5_overhead_percent", 0.0))
            for r in runs
            if str(r.get("kernel")) == "auto_t5_guarded"
        ),
        default=0.0,
    )
    t5_disable_total = sum(
        int(r["metrics"].get("t5_disable_events", 0))
        for r in runs
        if str(r.get("kernel")) == "auto_t5_guarded"
    )
    checks["t5_overhead_bound"] = {
        "observed_max": float(t5_overhead_max),
        "required_max": float(guardrails["max_t5_overhead_percent"]),
        "pass": float(t5_overhead_max) <= float(guardrails["max_t5_overhead_percent"]),
    }
    checks["t5_disable_events_total_bound"] = {
        "observed": int(t5_disable_total),
        "required_max": int(guardrails["max_t5_disable_events_total"]),
        "pass": int(t5_disable_total) <= int(guardrails["max_t5_disable_events_total"]),
    }

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((str(run["platform_selector"]).lower(), _key(str(run["kernel"]), int(run["size"]))), []).append(run)

    clover_row_checks: list[dict[str, Any]] = []
    per_policy = policy["weekly_slo"]["per_kernel_size"]
    for row_key, slo in per_policy.items():
        entries = grouped.get(("clover", row_key), [])
        if not entries:
            clover_row_checks.append(
                {
                    "platform": "clover",
                    "key": row_key,
                    "pass": False,
                    "reason": "missing_rows",
                }
            )
            continue
        avg = float(statistics.mean(float(e["metrics"]["avg_mean_gflops"]) for e in entries))
        p95 = float(statistics.mean(float(e["metrics"]["p95_time_ms"]) for e in entries))
        avg_pass = avg >= float(slo["min_avg_gflops"])
        p95_pass = p95 <= float(slo["max_p95_time_ms"])
        clover_row_checks.append(
            {
                "platform": "clover",
                "key": row_key,
                "observed_avg_gflops": avg,
                "required_min_avg_gflops": float(slo["min_avg_gflops"]),
                "observed_p95_time_ms": p95,
                "required_max_p95_time_ms": float(slo["max_p95_time_ms"]),
                "pass": bool(avg_pass and p95_pass),
            }
        )
    checks["clover_policy_rows"] = {
        "rows": clover_row_checks,
        "pass": all(bool(r["pass"]) for r in clover_row_checks) and len(clover_row_checks) > 0,
    }

    ratio_rows: list[dict[str, Any]] = []
    for row_key in sorted(per_policy):
        c_entries = grouped.get(("clover", row_key), [])
        r_entries = grouped.get(("rusticl", row_key), [])
        if not c_entries or not r_entries:
            ratio_rows.append(
                {
                    "key": row_key,
                    "ratio_rusticl_vs_clover": 0.0,
                    "pass": False,
                    "reason": "missing_rows",
                }
            )
            continue
        c_avg = float(statistics.mean(float(e["metrics"]["avg_mean_gflops"]) for e in c_entries))
        r_avg = float(statistics.mean(float(e["metrics"]["avg_mean_gflops"]) for e in r_entries))
        ratio = 0.0 if c_avg == 0.0 else r_avg / c_avg
        ratio_rows.append(
            {
                "key": row_key,
                "clover_avg_gflops": c_avg,
                "rusticl_avg_gflops": r_avg,
                "ratio_rusticl_vs_clover": ratio,
                "required_min_ratio": float(min_rusticl_ratio),
                "pass": bool(ratio >= float(min_rusticl_ratio)),
            }
        )

    checks["rusticl_ratio_vs_clover"] = {
        "rows": ratio_rows,
        "pass": all(bool(r["pass"]) for r in ratio_rows) and len(ratio_rows) > 0,
    }

    failed_checks = [name for name, info in checks.items() if not bool(info.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Platform split satisfies formal policy guardrails and rusticl/clover ratio requirements."
        if decision == "promote"
        else "Platform split violated one or more policy guardrails/ratio checks."
    )

    return {
        "checks": checks,
        "failed_checks": failed_checks,
        "decision": decision,
        "rationale": rationale,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 12 Block 2 - Platform Split Policy Evaluation")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Split artifact: `{report['metadata']['split_artifact']}`")
    lines.append(f"- Policy: `{report['metadata']['policy_path']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, info in report["evaluation"]["checks"].items():
        lines.append(f"| {name} | {info['pass']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['evaluation']['decision']}`")
    lines.append(f"- Failed checks: {report['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {report['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate platform split against formal weekly policy.")
    parser.add_argument("--split-artifact", required=True)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--min-rusticl-ratio", type=float, default=0.85)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week12_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week12_block2_platform_split_eval")
    args = parser.parse_args()

    split_path = Path(args.split_artifact).resolve()
    policy_path = Path(args.policy_path).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_report = _read_json(split_path)
    policy = _read_json(policy_path)
    evaluation = _evaluate(split_report, policy, min_rusticl_ratio=float(args.min_rusticl_ratio))

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "split_artifact": str(split_path),
            "policy_path": str(policy_path),
            "min_rusticl_ratio": float(args.min_rusticl_ratio),
        },
        "evaluation": evaluation,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{args.output_prefix}_{stamp}.json"
    md_path = out_dir / f"{args.output_prefix}_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    md_path.write_text(_markdown(payload) + "\n")

    print(f"Week12 block2 eval JSON: {json_path}")
    print(f"Week12 block2 eval MD:   {md_path}")
    print(f"Decision: {evaluation['decision']}")
    print(f"Failed checks: {evaluation['failed_checks']}")
    return 0 if evaluation["decision"] == "promote" else 7


if __name__ == "__main__":
    raise SystemExit(main())
