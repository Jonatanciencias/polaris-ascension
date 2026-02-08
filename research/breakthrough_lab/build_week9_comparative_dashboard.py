#!/usr/bin/env python3
"""Build comparative dashboard for T3/T4/T5 including Week 9 block deltas."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BLOCK1 = "research/breakthrough_lab/week9_block1_long_canary_20260208_030816.json"
DEFAULT_BLOCK2 = "research/breakthrough_lab/week9_block2_long_canary_rerun_20260208_032017.json"
DEFAULT_BLOCK3 = (
    "research/breakthrough_lab/platform_compatibility/week9_block3_robustness_replay_20260208_033111.json"
)
DEFAULT_T4_REFERENCE = "research/breakthrough_lab/week8_block6_t4_t5_interaction_20260208_024510.json"


def _safe_pct_delta(new: float, old: float) -> float:
    if old == 0.0:
        return 0.0
    return float((new - old) / old * 100.0)


def _summarize_block12(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    groups = payload["summary"]["groups"]

    def t3() -> dict[str, Any]:
        rows = [g for g in groups if g["kernel"] == "auto_t3_controlled"]
        return {
            "avg_gflops_mean": float(statistics.mean(float(r["avg_mean_gflops"]) for r in rows)),
            "p95_time_ms_mean": float(statistics.mean(float(r["p95_time_ms"]) for r in rows)),
            "max_error_max": float(max(float(r["max_error_max"]) for r in rows)),
            "drift_abs_max_percent": float(max(abs(float(r["drift_percent"])) for r in rows)),
            "fallback_rate_mean": float(
                statistics.mean(float(r.get("t3_fallback_rate_mean", 0.0)) for r in rows)
            ),
            "policy_disabled_total": int(sum(int(r.get("t3_policy_disabled_count", 0)) for r in rows)),
        }

    def t5() -> dict[str, Any]:
        rows = [g for g in groups if g["kernel"] == "auto_t5_guarded"]
        return {
            "avg_gflops_mean": float(statistics.mean(float(r["avg_mean_gflops"]) for r in rows)),
            "p95_time_ms_mean": float(statistics.mean(float(r["p95_time_ms"]) for r in rows)),
            "max_error_max": float(max(float(r["max_error_max"]) for r in rows)),
            "drift_abs_max_percent": float(max(abs(float(r["drift_percent"])) for r in rows)),
            "overhead_mean_percent": float(
                statistics.mean(float(r.get("t5_overhead_mean_percent", 0.0)) for r in rows)
            ),
            "false_positive_rate_mean": float(
                statistics.mean(float(r.get("t5_false_positive_rate_mean", 0.0)) for r in rows)
            ),
            "disable_events_total": int(sum(int(r.get("t5_disable_events_total", 0)) for r in rows)),
        }

    return {
        "metadata": payload["metadata"],
        "decision": payload.get("evaluation", {}).get("decision"),
        "t3": t3(),
        "t5": t5(),
    }


def _summarize_block3_or_4(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    runs = [r for r in payload.get("runs", []) if r.get("status") == "ok"]
    clover = [r for r in runs if str(r.get("platform_selector", "")).lower() == "clover"]

    def _agg(rows: list[dict[str, Any]], kernel: str) -> dict[str, Any]:
        subset = [r for r in rows if r["kernel"] == kernel]
        base = {
            "avg_gflops_mean": float(statistics.mean(float(r["metrics"]["avg_mean_gflops"]) for r in subset)),
            "p95_time_ms_mean": float(statistics.mean(float(r["metrics"]["p95_time_ms"]) for r in subset)),
            "max_error_max": float(max(float(r["metrics"]["max_error_max"]) for r in subset)),
        }
        if kernel == "auto_t3_controlled":
            base["fallback_rate_mean"] = float(
                statistics.mean(float(r["metrics"].get("t3_fallback_rate", 0.0)) for r in subset)
            )
            base["policy_disabled_total"] = int(
                sum(int(bool(r["metrics"].get("t3_policy_disabled", False))) for r in subset)
            )
        if kernel == "auto_t5_guarded":
            base["overhead_mean_percent"] = float(
                statistics.mean(float(r["metrics"].get("t5_overhead_percent", 0.0)) for r in subset)
            )
            base["false_positive_rate_mean"] = float(
                statistics.mean(float(r["metrics"].get("t5_false_positive_rate", 0.0)) for r in subset)
            )
            base["disable_events_total"] = int(
                sum(int(r["metrics"].get("t5_disable_events", 0)) for r in subset)
            )
        return base

    return {
        "metadata": payload.get("metadata", {}),
        "decision": payload.get("evaluation", {}).get("decision"),
        "t3": _agg(clover, "auto_t3_controlled"),
        "t5": _agg(clover, "auto_t5_guarded"),
    }


def _summarize_t4_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    summary = payload.get("summary", {})
    t4 = summary.get("t4_combined", {})
    cross = summary.get("cross_effect", {})
    return {
        "source": str(path),
        "contract_compliance_rate": float(t4.get("contract_compliance_rate", 0.0)),
        "fallback_rate": float(t4.get("fallback_rate", 0.0)),
        "post_fallback_violation_rate": float(t4.get("post_fallback_violation_rate", 0.0)),
        "compressible_speedup_vs_exact_mean": float(
            t4.get("compressible_speedup_vs_exact_mean", 0.0)
        ),
        "delta_vs_exact_percent": float(t4.get("delta_vs_exact_percent", 0.0)),
        "t5_cross_overhead_delta_percent": float(cross.get("t5_overhead_delta_percent", 0.0)),
        "t5_cross_p95_delta_percent": float(cross.get("t5_p95_delta_percent", 0.0)),
        "t5_cross_avg_gflops_delta_percent": float(cross.get("t5_avg_gflops_delta_percent", 0.0)),
    }


def _build_deltas(stages: dict[str, dict[str, Any]], track: str) -> dict[str, Any]:
    b1 = stages["block1"][track]
    b2 = stages["block2"][track]
    b3 = stages["block3"][track]
    out: dict[str, Any] = {
        "block2_vs_block1": {
            "avg_gflops_delta_percent": _safe_pct_delta(
                float(b2["avg_gflops_mean"]), float(b1["avg_gflops_mean"])
            ),
            "p95_time_ms_delta_percent": _safe_pct_delta(
                float(b2["p95_time_ms_mean"]), float(b1["p95_time_ms_mean"])
            ),
        },
        "block3_vs_block2": {
            "avg_gflops_delta_percent": _safe_pct_delta(
                float(b3["avg_gflops_mean"]), float(b2["avg_gflops_mean"])
            ),
            "p95_time_ms_delta_percent": _safe_pct_delta(
                float(b3["p95_time_ms_mean"]), float(b2["p95_time_ms_mean"])
            ),
        },
        "block3_vs_block1": {
            "avg_gflops_delta_percent": _safe_pct_delta(
                float(b3["avg_gflops_mean"]), float(b1["avg_gflops_mean"])
            ),
            "p95_time_ms_delta_percent": _safe_pct_delta(
                float(b3["p95_time_ms_mean"]), float(b1["p95_time_ms_mean"])
            ),
        },
    }
    if track == "t5":
        out["disable_events"] = {
            "block1": int(b1["disable_events_total"]),
            "block2": int(b2["disable_events_total"]),
            "block3": int(b3["disable_events_total"]),
        }
    return out


def _markdown(report: dict[str, Any]) -> str:
    stage_order = [s for s in ["block1", "block2", "block3", "block4", "block5"] if s in report["stages"]]
    lines: list[str] = []
    lines.append("# Week 9 Comparative Dashboard - T3/T4/T5")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Branch: {report['metadata']['branch']}")
    lines.append("")
    lines.append("## Stage Decisions")
    lines.append("")
    lines.append("| Stage | Decision |")
    lines.append("| --- | --- |")
    for stage in stage_order:
        lines.append(f"| {stage} | {report['stages'][stage]['decision']} |")
    lines.append("")
    lines.append("## T3/T5 Aggregates (Clover-normalized for block3/4)")
    lines.append("")
    lines.append("| Track | Stage | Avg GFLOPS | P95 ms | Max error | Extra |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- |")
    for track in ["t3", "t5"]:
        for stage in stage_order:
            s = report["stages"][stage][track]
            extra = "-"
            if track == "t3":
                extra = f"fallback={s['fallback_rate_mean']:.4f}, disabled={s['policy_disabled_total']}"
            if track == "t5":
                extra = f"overhead={s['overhead_mean_percent']:.3f}%, disable={s['disable_events_total']}"
            lines.append(
                f"| {track} | {stage} | {s['avg_gflops_mean']:.3f} | {s['p95_time_ms_mean']:.3f} | {s['max_error_max']:.7f} | {extra} |"
            )
    lines.append("")
    lines.append("## Week9 Deltas (Block1/2/3)")
    lines.append("")
    lines.append("| Track | Delta | Avg GFLOPS % | P95 % |")
    lines.append("| --- | --- | ---: | ---: |")
    for track in ["t3", "t5"]:
        deltas = report["deltas_week9_block123"][track]
        for key in ["block2_vs_block1", "block3_vs_block2", "block3_vs_block1"]:
            lines.append(
                f"| {track} | {key} | {deltas[key]['avg_gflops_delta_percent']:+.3f} | {deltas[key]['p95_time_ms_delta_percent']:+.3f} |"
            )
    lines.append("")
    lines.append("## T4 Reference")
    lines.append("")
    t4 = report["t4_reference"]
    lines.append(
        f"- Source: `{t4['source']}` (latest validated T4 evidence; Week9 blocks 1/2/3 did not modify T4 policy)"
    )
    lines.append(
        f"- Contract compliance: {t4['contract_compliance_rate']:.3f} | Fallback: {t4['fallback_rate']:.3f} | Post-fallback violation: {t4['post_fallback_violation_rate']:.3f}"
    )
    lines.append(
        f"- Compressible speedup vs exact: {t4['compressible_speedup_vs_exact_mean']:.3f} | Delta vs exact: {t4['delta_vs_exact_percent']:+.3f}%"
    )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Dashboard status: `{report['dashboard_decision']}`")
    lines.append(f"- Rationale: {report['dashboard_rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def build_dashboard(
    *,
    block1_path: str,
    block2_path: str,
    block3_path: str,
    t4_reference_path: str,
    block4_path: str | None = None,
    block5_path: str | None = None,
) -> dict[str, Any]:
    stages: dict[str, dict[str, Any]] = {
        "block1": _summarize_block12((REPO_ROOT / block1_path).resolve()),
        "block2": _summarize_block12((REPO_ROOT / block2_path).resolve()),
        "block3": _summarize_block3_or_4((REPO_ROOT / block3_path).resolve()),
    }
    if block4_path:
        stages["block4"] = _summarize_block3_or_4((REPO_ROOT / block4_path).resolve())
    if block5_path:
        stages["block5"] = _summarize_block3_or_4((REPO_ROOT / block5_path).resolve())

    deltas = {
        "t3": _build_deltas(stages, "t3"),
        "t5": _build_deltas(stages, "t5"),
    }
    t4_reference = _summarize_t4_reference((REPO_ROOT / t4_reference_path).resolve())

    block1_decision = str(stages["block1"].get("decision"))
    active_stage_decisions = [str(stages["block2"].get("decision")), str(stages["block3"].get("decision"))]
    if "block4" in stages:
        active_stage_decisions.append(str(stages["block4"].get("decision")))
    if "block5" in stages:
        active_stage_decisions.append(str(stages["block5"].get("decision")))
    active_chain_promote = all(d == "promote" for d in active_stage_decisions)

    if active_chain_promote:
        dashboard_decision = "promote"
        if block1_decision == "iterate":
            dashboard_rationale = (
                "Block1 iterate was superseded by Block2 hardening; active Week9 chain (Block2/3/4) is fully promote."
            )
        else:
            dashboard_rationale = (
                "Week9 chain keeps promote trend on T3/T5 after hardening; T4 remains stable at last validated baseline."
            )
    else:
        dashboard_decision = "iterate"
        dashboard_rationale = "One or more active Week9 stages are not in promote state."

    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "branch": subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(REPO_ROOT),
                text=True,
            ).strip(),
            "inputs": {
                "block1_path": block1_path,
                "block2_path": block2_path,
                "block3_path": block3_path,
                "block4_path": block4_path,
                "block5_path": block5_path,
                "t4_reference_path": t4_reference_path,
            },
        },
        "stages": stages,
        "deltas_week9_block123": deltas,
        "t4_reference": t4_reference,
        "dashboard_decision": dashboard_decision,
        "dashboard_rationale": dashboard_rationale,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build comparative dashboard for Week9.")
    parser.add_argument("--block1-path", default=DEFAULT_BLOCK1)
    parser.add_argument("--block2-path", default=DEFAULT_BLOCK2)
    parser.add_argument("--block3-path", default=DEFAULT_BLOCK3)
    parser.add_argument("--block4-path", default=None)
    parser.add_argument("--block5-path", default=None)
    parser.add_argument("--t4-reference-path", default=DEFAULT_T4_REFERENCE)
    parser.add_argument("--output-dir", default="research/breakthrough_lab")
    parser.add_argument("--output-prefix", default="week9_comparative_dashboard")
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_dashboard(
        block1_path=str(args.block1_path),
        block2_path=str(args.block2_path),
        block3_path=str(args.block3_path),
        block4_path=args.block4_path,
        block5_path=args.block5_path,
        t4_reference_path=str(args.t4_reference_path),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"{args.output_prefix}_{timestamp}.json"
    md_path = output_dir / f"{args.output_prefix}_{timestamp}.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(_markdown(report))

    print(f"Dashboard JSON: {json_path}")
    print(f"Dashboard MD:   {md_path}")
    print(f"Decision: {report['dashboard_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
