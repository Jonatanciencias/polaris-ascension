#!/usr/bin/env python3
"""Build Week 11 operational dashboard and drift status artifacts."""

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


def _canary_summary(canary: dict[str, Any]) -> dict[str, Any]:
    runs = [r for r in canary.get("runs", []) if r.get("status") == "ok"]
    t3_runs = [r for r in runs if str(r.get("kernel")) == "auto_t3_controlled"]
    t5_runs = [r for r in runs if str(r.get("kernel")) == "auto_t5_guarded"]
    return {
        "decision": canary.get("evaluation", {}).get("decision"),
        "executed_snapshots": int(canary.get("metadata", {}).get("executed_snapshots", 0)),
        "rollback_triggered": canary.get("rollback_event") is not None,
        "failed_checks": list(canary.get("evaluation", {}).get("failed_checks", [])),
        "correctness_max": float(
            max((float(r["metrics"]["max_error_max"]) for r in runs), default=999.0)
        ),
        "t3_fallback_max": float(
            max((float(r["metrics"].get("t3_fallback_rate", 0.0)) for r in t3_runs), default=0.0)
        ),
        "t5_overhead_max": float(
            max((float(r["metrics"].get("t5_overhead_percent", 0.0)) for r in t5_runs), default=0.0)
        ),
        "t5_disable_events_total": int(
            sum(int(r["metrics"].get("t5_disable_events", 0)) for r in t5_runs)
        ),
    }


def _per_key_means(canary: dict[str, Any]) -> dict[str, dict[str, float]]:
    runs = [r for r in canary.get("runs", []) if r.get("status") == "ok"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(_key(str(run["kernel"]), int(run["size"])), []).append(run)
    out: dict[str, dict[str, float]] = {}
    for key, entries in grouped.items():
        out[key] = {
            "avg_gflops_mean": float(
                statistics.mean(float(e["metrics"]["avg_mean_gflops"]) for e in entries)
            ),
            "p95_time_ms_mean": float(
                statistics.mean(float(e["metrics"]["p95_time_ms"]) for e in entries)
            ),
        }
    return out


def _dashboard_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 11 Operational Dashboard")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Block 2 canary: `{payload['metadata']['block2_canary_path']}`")
    lines.append(f"- Block 4 canary: `{payload['metadata']['block4_canary_path']}`")
    lines.append(f"- Block 4 eval: `{payload['metadata']['block4_eval_path']}`")
    lines.append("")
    lines.append("## Block Summaries")
    lines.append("")
    lines.append("| Block | Decision | Snapshots | Rollback | Correctness max | T3 fallback max | T5 overhead max % | T5 disable total |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for block_name in ("block2", "block4"):
        row = payload["blocks"][block_name]
        lines.append(
            f"| {block_name} | {row['decision']} | {row['executed_snapshots']} | {row['rollback_triggered']} | "
            f"{row['correctness_max']:.9f} | {row['t3_fallback_max']:.4f} | {row['t5_overhead_max']:.4f} | "
            f"{row['t5_disable_events_total']} |"
        )
    lines.append("")
    lines.append("## Comparative Deltas (Block4 vs Block2)")
    lines.append("")
    lines.append("| Key | Block2 avg GFLOPS | Block4 avg GFLOPS | Delta % | Block2 p95 ms | Block4 p95 ms | Delta % |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["comparison_rows"]:
        lines.append(
            f"| {row['key']} | {row['block2_avg_gflops']:.3f} | {row['block4_avg_gflops']:.3f} | "
            f"{row['avg_gflops_delta_percent']:.3f} | {row['block2_p95_ms']:.3f} | "
            f"{row['block4_p95_ms']:.3f} | {row['p95_delta_percent']:.3f} |"
        )
    lines.append("")
    lines.append("## Drift Status")
    lines.append("")
    drift = payload["drift_status"]
    lines.append(f"- Decision: `{drift['decision']}`")
    lines.append(f"- Failed checks: {drift['failed_checks']}")
    lines.append(f"- Rationale: {drift['rationale']}")
    lines.append("")
    lines.append("## Package Decision")
    lines.append("")
    lines.append(f"- Decision: `{payload['package_decision']}`")
    lines.append(f"- Rationale: {payload['package_rationale']}")
    lines.append("")
    return "\n".join(lines)


def _drift_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 11 Drift Status")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Source replay eval: `{payload['metadata']['block4_eval_path']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, info in payload["checks"].items():
        lines.append(f"| {name} | {info['pass']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{payload['decision']}`")
    lines.append(f"- Failed checks: {payload['failed_checks']}")
    lines.append(f"- Rationale: {payload['rationale']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Week 11 operational dashboard package.")
    parser.add_argument("--block2-canary", required=True)
    parser.add_argument("--block4-canary", required=True)
    parser.add_argument("--block4-eval", required=True)
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week11_controlled_rollout",
    )
    parser.add_argument("--dashboard-prefix", default="week11_operational_dashboard")
    parser.add_argument("--drift-prefix", default="week11_drift_status")
    args = parser.parse_args()

    block2_path = Path(args.block2_canary).resolve()
    block4_path = Path(args.block4_canary).resolve()
    block4_eval_path = Path(args.block4_eval).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    block2 = _read_json(block2_path)
    block4 = _read_json(block4_path)
    replay_eval = _read_json(block4_eval_path)

    block2_summary = _canary_summary(block2)
    block4_summary = _canary_summary(block4)

    block2_means = _per_key_means(block2)
    block4_means = _per_key_means(block4)
    common_keys = sorted(set(block2_means).intersection(block4_means))

    comparison_rows: list[dict[str, Any]] = []
    for key in common_keys:
        b2 = block2_means[key]
        b4 = block4_means[key]
        avg_delta = 0.0 if b2["avg_gflops_mean"] == 0.0 else (
            (b4["avg_gflops_mean"] - b2["avg_gflops_mean"]) / b2["avg_gflops_mean"] * 100.0
        )
        p95_delta = 0.0 if b2["p95_time_ms_mean"] == 0.0 else (
            (b4["p95_time_ms_mean"] - b2["p95_time_ms_mean"]) / b2["p95_time_ms_mean"] * 100.0
        )
        comparison_rows.append(
            {
                "key": key,
                "block2_avg_gflops": float(b2["avg_gflops_mean"]),
                "block4_avg_gflops": float(b4["avg_gflops_mean"]),
                "avg_gflops_delta_percent": float(avg_delta),
                "block2_p95_ms": float(b2["p95_time_ms_mean"]),
                "block4_p95_ms": float(b4["p95_time_ms_mean"]),
                "p95_delta_percent": float(p95_delta),
            }
        )

    drift_status = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "block4_eval_path": str(block4_eval_path),
        },
        "checks": replay_eval["evaluation"]["checks"],
        "failed_checks": replay_eval["evaluation"]["failed_checks"],
        "decision": replay_eval["evaluation"]["decision"],
        "rationale": replay_eval["evaluation"]["rationale"],
    }

    package_decision = (
        "promote"
        if replay_eval["evaluation"]["decision"] == "promote"
        and block4_summary["decision"] == "promote"
        and not block4_summary["rollback_triggered"]
        else "iterate"
    )
    package_rationale = (
        "Week 11 package keeps operational canary and policy replay in promote with healthy drift status."
        if package_decision == "promote"
        else "One or more operational checks require refinement before package promotion."
    )

    dashboard = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "block2_canary_path": str(block2_path),
            "block4_canary_path": str(block4_path),
            "block4_eval_path": str(block4_eval_path),
        },
        "blocks": {"block2": block2_summary, "block4": block4_summary},
        "comparison_rows": comparison_rows,
        "drift_status": {
            "decision": drift_status["decision"],
            "failed_checks": drift_status["failed_checks"],
            "rationale": drift_status["rationale"],
        },
        "package_decision": package_decision,
        "package_rationale": package_rationale,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dashboard_json = out_dir / f"{args.dashboard_prefix}_{stamp}.json"
    dashboard_md = out_dir / f"{args.dashboard_prefix}_{stamp}.md"
    drift_json = out_dir / f"{args.drift_prefix}_{stamp}.json"
    drift_md = out_dir / f"{args.drift_prefix}_{stamp}.md"

    dashboard_json.write_text(json.dumps(dashboard, indent=2) + "\n")
    dashboard_md.write_text(_dashboard_md(dashboard) + "\n")
    drift_json.write_text(json.dumps(drift_status, indent=2) + "\n")
    drift_md.write_text(_drift_md(drift_status) + "\n")

    print(f"Dashboard JSON: {dashboard_json}")
    print(f"Dashboard MD:   {dashboard_md}")
    print(f"Drift JSON:     {drift_json}")
    print(f"Drift MD:       {drift_md}")
    print(f"Package decision: {package_decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
