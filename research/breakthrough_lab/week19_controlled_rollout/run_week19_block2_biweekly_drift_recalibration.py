#!/usr/bin/env python3
"""Week 19 Block 2: biweekly drift review and conservative recalibration."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _extract_prefixed_line(text: str, prefix: str) -> str | None:
    for line in text.splitlines():
        row = line.strip()
        if row.startswith(prefix):
            return row[len(prefix) :].strip()
    return None


def _normalize_policy_metadata(policy_path: Path, *, base_policy_path: Path) -> dict[str, Any]:
    payload = _read_json(policy_path)
    payload["policy_id"] = "week19-block2-weekly-slo-v3-2026-02-11"
    payload["status"] = "candidate_weekly_monitoring_policy_week19_block2"
    payload["inherits_from"] = str(base_policy_path)
    rec = payload.setdefault("recalibration", {})
    rec["evidence_source"] = "week17_block4 + week19_block1"
    rec["method"] = "conservative_headroom_tightening_week19_block2"
    _write_json(policy_path, payload)
    return payload


def _md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 19 Block 2 - Biweekly Drift Recalibration")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{report['metadata']['stable_tag']}`")
    lines.append(f"- Base policy: `{report['metadata']['base_policy_path']}`")
    lines.append(f"- Recalibrated policy: `{report['metadata']['recalibrated_policy_path']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for key, chk in report["evaluation"]["checks"].items():
        lines.append(f"| {key} | {chk['pass']} |")
    lines.append("")
    lines.append("## Highlights")
    lines.append("")
    lines.append(f"- Recalibration decision: `{report['highlights']['recalibration_decision']}`")
    lines.append(f"- Recalibration action: `{report['highlights']['recalibration_action']}`")
    lines.append(f"- Recalibrated policy eval decision: `{report['highlights']['recalibrated_eval_decision']}`")
    lines.append(
        f"- Observed max abs throughput drift: `{float(report['highlights']['global_max_abs_throughput_drift_percent']):.6f}`"
    )
    lines.append(
        f"- Observed max p95 drift: `{float(report['highlights']['global_max_p95_drift_percent']):.6f}`"
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for key, value in report["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{report['evaluation']['decision']}`")
    lines.append(f"- Failed checks: {report['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {report['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Week19 Block2 biweekly drift recalibration.")
    parser.add_argument(
        "--stable-manifest-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json",
    )
    parser.add_argument(
        "--base-policy-path",
        default="research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json",
    )
    parser.add_argument(
        "--eval-artifacts",
        nargs="+",
        default=[
            "research/breakthrough_lab/week17_controlled_rollout/week17_block4_posthardening_replay_automation_eval_20260211_011930.json",
            "research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_weekly_replay_eval_20260211_015423.json",
        ],
    )
    parser.add_argument(
        "--canary-path",
        default="research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_weekly_replay_canary_20260211_015423.json",
    )
    parser.add_argument("--min-windows", type=int, default=2)
    parser.add_argument("--max-global-abs-drift-for-recalibration", type=float, default=1.0)
    parser.add_argument("--max-global-p95-drift-for-recalibration", type=float, default=0.5)
    parser.add_argument("--max-avg-cv-for-recalibration", type=float, default=0.01)
    parser.add_argument("--max-p95-cv-for-recalibration", type=float, default=0.01)
    parser.add_argument("--tighten-headroom-fraction", type=float, default=0.15)
    parser.add_argument(
        "--recalibrated-policy-path",
        default="research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json",
    )
    parser.add_argument(
        "--report-dir",
        default="research/breakthrough_lab/week8_validation_discipline",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week19_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week19_block2_biweekly_drift_recalibration")
    args = parser.parse_args()

    stable_manifest_path = _resolve(args.stable_manifest_path)
    base_policy_path = _resolve(args.base_policy_path)
    eval_artifacts = [_resolve(path) for path in args.eval_artifacts]
    canary_path = _resolve(args.canary_path)
    recalibrated_policy_path = _resolve(args.recalibrated_policy_path)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stable_manifest: dict[str, Any] = {}
    stable_tag = "unknown"
    if stable_manifest_path.exists():
        stable_manifest = _read_json(stable_manifest_path)
        stable_tag = str(stable_manifest.get("stable_tag", "unknown"))

    pre_gate_cmd = [
        "./venv/bin/python",
        "scripts/run_validation_suite.py",
        "--tier",
        "canonical",
        "--driver-smoke",
        "--report-dir",
        str(args.report_dir),
    ]
    pre_gate = _run(pre_gate_cmd)
    pre_gate_json = _extract_prefixed_line(pre_gate["stdout"], "Wrote JSON report:")
    pre_gate_decision = "unknown"
    pre_gate_pytest_green = False
    if pre_gate_json:
        pre_payload = _read_json(_resolve(pre_gate_json))
        pre_gate_decision = str(pre_payload.get("evaluation", {}).get("decision", "unknown"))
        pre_gate_pytest_green = bool(
            pre_payload.get("evaluation", {})
            .get("checks", {})
            .get("pytest_tier_green", {})
            .get("pass", False)
        )

    recalibration_cmd = [
        "./venv/bin/python",
        "research/breakthrough_lab/week13_controlled_rollout/evaluate_week13_block3_drift_recalibration.py",
        "--base-policy-path",
        str(base_policy_path),
        "--eval-artifacts",
        *[str(path) for path in eval_artifacts],
        "--min-windows",
        str(args.min_windows),
        "--max-global-abs-drift-for-recalibration",
        str(args.max_global_abs_drift_for_recalibration),
        "--max-global-p95-drift-for-recalibration",
        str(args.max_global_p95_drift_for_recalibration),
        "--max-avg-cv-for-recalibration",
        str(args.max_avg_cv_for_recalibration),
        "--max-p95-cv-for-recalibration",
        str(args.max_p95_cv_for_recalibration),
        "--tighten-headroom-fraction",
        str(args.tighten_headroom_fraction),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        str(args.output_prefix),
        "--policy-output-path",
        str(recalibrated_policy_path),
    ]
    recalibration = _run(recalibration_cmd)
    recalibration_json = _extract_prefixed_line(recalibration["stdout"], "Week13 block3 JSON:")
    recalibration_md = _extract_prefixed_line(recalibration["stdout"], "Week13 block3 MD:")
    recalibration_policy = _extract_prefixed_line(recalibration["stdout"], "Recalibrated policy:")
    recalibration_decision = str(_extract_prefixed_line(recalibration["stdout"], "Decision:") or "unknown")
    recalibration_action = str(
        _extract_prefixed_line(recalibration["stdout"], "Recalibration action:") or "unknown"
    )

    normalized_policy = {}
    if recalibrated_policy_path.exists():
        normalized_policy = _normalize_policy_metadata(
            recalibrated_policy_path, base_policy_path=base_policy_path
        )

    recal_report_payload = _read_json(_resolve(recalibration_json)) if recalibration_json else {}
    recal_summary = recal_report_payload.get("evaluation", {}).get("summary", {})
    global_thr = float(recal_summary.get("global_max_abs_throughput_drift_percent", 999.0))
    global_p95 = float(recal_summary.get("global_max_p95_drift_percent", 999.0))

    reevaluate_cmd = [
        "./venv/bin/python",
        "research/breakthrough_lab/week11_controlled_rollout/evaluate_week11_weekly_replay.py",
        "--canary-path",
        str(canary_path),
        "--policy-path",
        str(recalibrated_policy_path),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        "week19_block2_recalibrated_policy_eval",
    ]
    reevaluate = _run(reevaluate_cmd)
    reevaluate_json = _extract_prefixed_line(reevaluate["stdout"], "Weekly replay eval JSON:")
    reevaluate_md = _extract_prefixed_line(reevaluate["stdout"], "Weekly replay eval MD:")
    reevaluate_decision = str(_extract_prefixed_line(reevaluate["stdout"], "Decision:") or "unknown")

    post_gate = _run(pre_gate_cmd)
    post_gate_json = _extract_prefixed_line(post_gate["stdout"], "Wrote JSON report:")
    post_gate_decision = "unknown"
    post_gate_pytest_green = False
    if post_gate_json:
        post_payload = _read_json(_resolve(post_gate_json))
        post_gate_decision = str(post_payload.get("evaluation", {}).get("decision", "unknown"))
        post_gate_pytest_green = bool(
            post_payload.get("evaluation", {})
            .get("checks", {})
            .get("pytest_tier_green", {})
            .get("pass", False)
        )

    checks: dict[str, dict[str, Any]] = {}
    checks["stable_manifest_exists"] = {
        "observed": stable_manifest_path.exists(),
        "required": True,
        "pass": stable_manifest_path.exists(),
    }
    checks["stable_tag_v0_15_0"] = {
        "observed": stable_tag,
        "required": "v0.15.0",
        "pass": stable_tag == "v0.15.0",
    }
    checks["recalibration_decision_promote"] = {
        "observed": recalibration_decision.lower(),
        "required": "promote",
        "pass": recalibration_decision.lower() == "promote",
    }
    checks["recalibrated_policy_exists"] = {
        "observed": recalibrated_policy_path.exists(),
        "required": True,
        "pass": recalibrated_policy_path.exists(),
    }
    checks["recalibrated_policy_eval_promote"] = {
        "observed": reevaluate_decision.lower(),
        "required": "promote",
        "pass": reevaluate_decision.lower() == "promote",
    }
    checks["global_abs_drift_conservative"] = {
        "observed_max": global_thr,
        "required_max": float(args.max_global_abs_drift_for_recalibration),
        "pass": global_thr <= float(args.max_global_abs_drift_for_recalibration),
    }
    checks["global_p95_drift_conservative"] = {
        "observed_max": global_p95,
        "required_max": float(args.max_global_p95_drift_for_recalibration),
        "pass": global_p95 <= float(args.max_global_p95_drift_for_recalibration),
    }
    checks["pre_gate_promote"] = {
        "observed": pre_gate_decision,
        "required": "promote",
        "pass": pre_gate_decision == "promote",
    }
    checks["pre_gate_pytest_tier_green"] = {
        "observed": bool(pre_gate_pytest_green),
        "required": True,
        "pass": bool(pre_gate_pytest_green),
    }
    checks["post_gate_promote"] = {
        "observed": post_gate_decision,
        "required": "promote",
        "pass": post_gate_decision == "promote",
    }
    checks["post_gate_pytest_tier_green"] = {
        "observed": bool(post_gate_pytest_green),
        "required": True,
        "pass": bool(post_gate_pytest_green),
    }

    failed_checks = [key for key, chk in checks.items() if not bool(chk.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Biweekly drift review confirms stable behavior and conservative recalibration remains policy-safe."
        if decision == "promote"
        else "Biweekly recalibration found unresolved drift/policy/gate checks."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": stable_tag,
            "stable_manifest_path": str(stable_manifest_path),
            "base_policy_path": str(base_policy_path),
            "recalibrated_policy_path": str(recalibrated_policy_path),
            "canary_path": str(canary_path),
            "eval_artifacts": [str(path) for path in eval_artifacts],
        },
        "commands": {
            "pre_gate": pre_gate,
            "recalibration": recalibration,
            "recalibrated_policy_eval": reevaluate,
            "post_gate": post_gate,
            "recalibration_cmd_pretty": " ".join(shlex.quote(x) for x in recalibration_cmd),
            "reevaluate_cmd_pretty": " ".join(shlex.quote(x) for x in reevaluate_cmd),
        },
        "artifacts": {
            "recalibration_json": recalibration_json,
            "recalibration_md": recalibration_md,
            "recalibration_policy_raw": recalibration_policy,
            "recalibrated_policy_path": str(recalibrated_policy_path),
            "recalibrated_policy_eval_json": reevaluate_json,
            "recalibrated_policy_eval_md": reevaluate_md,
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
        },
        "highlights": {
            "recalibration_decision": recalibration_decision,
            "recalibration_action": recalibration_action,
            "recalibrated_eval_decision": reevaluate_decision,
            "global_max_abs_throughput_drift_percent": global_thr,
            "global_max_p95_drift_percent": global_p95,
            "normalized_policy_id": normalized_policy.get("policy_id"),
        },
        "evaluation": {
            "checks": checks,
            "failed_checks": failed_checks,
            "decision": decision,
            "rationale": rationale,
        },
    }

    json_path = output_dir / f"{args.output_prefix}_package_{stamp}.json"
    md_path = output_dir / f"{args.output_prefix}_package_{stamp}.md"
    _write_json(json_path, report)
    md_path.write_text(_md(report))

    print(f"Week19 block2 JSON: {json_path}")
    print(f"Week19 block2 MD:   {md_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
