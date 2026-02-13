#!/usr/bin/env python3
"""Week 18 Block 1: stable operations package for dependent adoption."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
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


def _extract_decision(payload: dict[str, Any]) -> str:
    for key in ("decision", "block_decision", "operational_decision", "result"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    eval_decision = payload.get("evaluation", {}).get("decision")
    if isinstance(eval_decision, str) and eval_decision:
        return eval_decision
    return "unknown"


def _build_comparative_payload(
    *,
    week16_block2_report: dict[str, Any],
    week17_block1_report: dict[str, Any],
    week17_block2_report: dict[str, Any],
    week17_block3_report: dict[str, Any],
    week17_block4_report: dict[str, Any],
) -> dict[str, Any]:
    week16_thr = float(week16_block2_report.get("drift", {}).get("max_abs_throughput_drift_percent", 999.0))
    week16_p95 = float(week16_block2_report.get("drift", {}).get("max_positive_p95_drift_percent", 999.0))
    week17_thr = float(week17_block4_report.get("drift", {}).get("max_abs_throughput_drift_percent", 999.0))
    week17_p95 = float(week17_block4_report.get("drift", {}).get("max_positive_p95_drift_percent", 999.0))

    thr_delta = week17_thr - week16_thr
    p95_delta = week17_p95 - week16_p95

    block1_summary = week17_block1_report.get("summary", {})
    block2_checks = week17_block2_report.get("evaluation", {}).get("checks", {})
    block3_repeat = week17_block3_report.get("repeat_summary", {})

    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_report": "week16_block2_weekly_rc_replay_rerun",
            "current_report": "week17_block4_posthardening_replay",
        },
        "decisions": {
            "week17_block1": _extract_decision(week17_block1_report),
            "week17_block2": _extract_decision(week17_block2_report),
            "week17_block3": _extract_decision(week17_block3_report),
            "week17_block4": _extract_decision(week17_block4_report),
        },
        "metrics": {
            "week16_max_abs_throughput_drift_percent": week16_thr,
            "week17_max_abs_throughput_drift_percent": week17_thr,
            "delta_throughput_drift_percent": thr_delta,
            "week16_max_positive_p95_drift_percent": week16_p95,
            "week17_max_positive_p95_drift_percent": week17_p95,
            "delta_p95_drift_percent": p95_delta,
            "week17_block1_t5_disable_total": int(block1_summary.get("canary_t5_disable_total", -1)),
            "week17_block1_correctness_max": block1_summary.get("canary_correctness_max"),
            "week17_block1_executed_snapshots": int(block1_summary.get("executed_snapshots", -1)),
            "week17_block2_plugin_promote": bool(
                block2_checks.get("plugin_pilot_promote", {}).get("pass", False)
            ),
            "week17_block3_repeat_passed_runs": int(block3_repeat.get("passed_runs", 0)),
            "week17_block3_repeat_failed_runs": int(block3_repeat.get("failed_runs", 0)),
        },
        "conclusions": {
            "drift_improved": bool(thr_delta < 0.0 and p95_delta <= 0.0),
            "pytest_flake_stabilized": bool(block3_repeat.get("failed_runs", 1) == 0),
            "stable_rollout_go": str(_extract_decision(week17_block1_report)).lower() in {"go", "promote"},
        },
    }


def _comparative_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 18 Block 1 - Comparative Update")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Baseline: `{payload['metadata']['baseline_report']}`")
    lines.append(f"- Current: `{payload['metadata']['current_report']}`")
    lines.append("")
    lines.append("## Decision Chain")
    lines.append("")
    for name, decision in payload["decisions"].items():
        lines.append(f"- `{name}`: `{decision}`")
    lines.append("")
    lines.append("## Drift Delta")
    lines.append("")
    m = payload["metrics"]
    lines.append(
        f"- Throughput drift max abs: {m['week16_max_abs_throughput_drift_percent']:.4f}% -> {m['week17_max_abs_throughput_drift_percent']:.4f}% (delta {m['delta_throughput_drift_percent']:.4f}%)"
    )
    lines.append(
        f"- P95 drift max: {m['week16_max_positive_p95_drift_percent']:.4f}% -> {m['week17_max_positive_p95_drift_percent']:.4f}% (delta {m['delta_p95_drift_percent']:.4f}%)"
    )
    lines.append("")
    lines.append("## Stability Highlights")
    lines.append("")
    lines.append(f"- Week17 Block1 snapshots: {m['week17_block1_executed_snapshots']}")
    lines.append(f"- Week17 Block1 T5 disable total: {m['week17_block1_t5_disable_total']}")
    lines.append(f"- Week17 Block3 repeat campaign: {m['week17_block3_repeat_passed_runs']} passed / {m['week17_block3_repeat_failed_runs']} failed")
    lines.append("")
    lines.append("## Conclusions")
    lines.append("")
    for key, value in payload["conclusions"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def _release_package_md(*, stable_tag: str, comparative_report_md: Path) -> str:
    lines: list[str] = []
    lines.append(f"# Week 18 Block 1 - Stable Operations Package ({stable_tag})")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Operational package for dependent project adoption using stable manifest.")
    lines.append("- Includes updated comparative report and adoption controls.")
    lines.append("")
    lines.append("## Included")
    lines.append("")
    lines.append("- Stable manifest verification and dependency traceability.")
    lines.append("- Runbook and checklist for dependent onboarding.")
    lines.append(f"- Comparative report: `{comparative_report_md}`")
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- Canonical gate (`--tier canonical --driver-smoke`) pre and post packaging.")
    lines.append("- Weekly replay post-hardening must remain `promote`.")
    lines.append("- Any new scope expansion requires fresh acta + decision.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _adoption_runbook_md(stable_tag: str) -> str:
    lines: list[str] = []
    lines.append(f"# Week 18 Block 1 - Dependent Adoption Runbook ({stable_tag})")
    lines.append("")
    lines.append("1. Validate stable manifest and referenced evidence paths.")
    lines.append("2. Run canonical gate (`run_validation_suite.py --tier canonical --driver-smoke`).")
    lines.append("3. Execute dependent plugin pilot using stable profile (1400/2048/3072).")
    lines.append("4. Store JSON + MD evidence and close local decision.")
    lines.append("5. If guardrails fail, rollback to last known-good policy and stop expansion.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _adoption_checklist_md() -> str:
    lines: list[str] = []
    lines.append("# Week 18 Block 1 - Stable Adoption Checklist")
    lines.append("")
    lines.append("- [ ] Stable manifest `v0.15.0` verified.")
    lines.append("- [ ] Week17 Block1/2/3/4 decisions are `promote` (Block1 operational `go`).")
    lines.append("- [ ] Canonical gate pre package is `promote`.")
    lines.append("- [ ] Canonical gate post package is `promote`.")
    lines.append("- [ ] Comparative report updated and attached.")
    lines.append("- [ ] Dependent adoption runbook reviewed by operators.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _report_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 18 Block 1 - Stable Operations Package")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{payload['metadata']['stable_tag']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, check in payload["evaluation"]["checks"].items():
        lines.append(f"| {name} | {check['pass']} |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for key, value in payload["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Decision: `{payload['evaluation']['decision']}`")
    lines.append(f"- Failed checks: {payload['evaluation']['failed_checks']}")
    lines.append(f"- Rationale: {payload['evaluation']['rationale']}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Week18 Block1 stable operations package.")
    parser.add_argument(
        "--stable-manifest-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json",
    )
    parser.add_argument(
        "--week16-block2-report-path",
        default="research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_rerun_20260210_015504.json",
    )
    parser.add_argument(
        "--week17-block1-report-path",
        default="research/breakthrough_lab/week17_controlled_rollout/week17_block1_stable_rollout_rerun_20260211_004918.json",
    )
    parser.add_argument(
        "--week17-block2-report-path",
        default="research/breakthrough_lab/week17_controlled_rollout/week17_block2_second_dependent_pilot_20260211_011149.json",
    )
    parser.add_argument(
        "--week17-block3-report-path",
        default="research/breakthrough_lab/week17_controlled_rollout/week17_block3_pytest_flake_hardening_20260211_005256.json",
    )
    parser.add_argument(
        "--week17-block4-report-path",
        default="research/breakthrough_lab/week17_controlled_rollout/week17_block4_posthardening_replay_20260211_012010.json",
    )
    parser.add_argument(
        "--week17-block1-decision-path",
        default="research/breakthrough_lab/week17_block1_stable_rollout_decision.json",
    )
    parser.add_argument(
        "--week17-block2-decision-path",
        default="research/breakthrough_lab/week17_block2_second_dependent_pilot_decision.json",
    )
    parser.add_argument(
        "--week17-block3-decision-path",
        default="research/breakthrough_lab/week17_block3_pytest_flake_hardening_decision.json",
    )
    parser.add_argument(
        "--week17-block4-decision-path",
        default="research/breakthrough_lab/week17_block4_posthardening_replay_decision.json",
    )
    parser.add_argument(
        "--report-dir",
        default="research/breakthrough_lab/week8_validation_discipline",
    )
    parser.add_argument(
        "--preprod-signoff-dir",
        default="research/breakthrough_lab/preprod_signoff",
    )
    parser.add_argument(
        "--output-dir",
        default="research/breakthrough_lab/week18_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week18_block1_stable_operations_package")
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    signoff_dir = _resolve(args.preprod_signoff_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signoff_dir.mkdir(parents=True, exist_ok=True)

    stable_manifest_path = _resolve(args.stable_manifest_path)
    stable_manifest = _read_json(stable_manifest_path) if stable_manifest_path.exists() else {}
    stable_tag = str(stable_manifest.get("stable_tag", "unknown"))

    week16_block2_report = _read_json(_resolve(args.week16_block2_report_path))
    week17_block1_report = _read_json(_resolve(args.week17_block1_report_path))
    week17_block2_report = _read_json(_resolve(args.week17_block2_report_path))
    week17_block3_report = _read_json(_resolve(args.week17_block3_report_path))
    week17_block4_report = _read_json(_resolve(args.week17_block4_report_path))

    week17_block1_decision = _read_json(_resolve(args.week17_block1_decision_path))
    week17_block2_decision = _read_json(_resolve(args.week17_block2_decision_path))
    week17_block3_decision = _read_json(_resolve(args.week17_block3_decision_path))
    week17_block4_decision = _read_json(_resolve(args.week17_block4_decision_path))

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
    if pre_gate_json:
        pre_gate_decision = str(
            _read_json(_resolve(pre_gate_json)).get("evaluation", {}).get("decision", "unknown")
        )

    comparative_payload = _build_comparative_payload(
        week16_block2_report=week16_block2_report,
        week17_block1_report=week17_block1_report,
        week17_block2_report=week17_block2_report,
        week17_block3_report=week17_block3_report,
        week17_block4_report=week17_block4_report,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparative_json = output_dir / f"week18_block1_comparative_update_{stamp}.json"
    comparative_md = output_dir / f"week18_block1_comparative_update_{stamp}.md"
    _write_json(comparative_json, comparative_payload)
    comparative_md.write_text(_comparative_md(comparative_payload))

    release_package_md = signoff_dir / "WEEK18_BLOCK1_V0_15_0_STABLE_RELEASE_PACKAGE.md"
    adoption_runbook_md = signoff_dir / "WEEK18_BLOCK1_DEPENDENT_ADOPTION_RUNBOOK.md"
    adoption_checklist_md = signoff_dir / "WEEK18_BLOCK1_STABLE_ADOPTION_CHECKLIST.md"
    package_manifest_json = signoff_dir / "WEEK18_BLOCK1_STABLE_PACKAGE_MANIFEST.json"

    release_package_md.write_text(
        _release_package_md(stable_tag=stable_tag, comparative_report_md=comparative_md)
    )
    adoption_runbook_md.write_text(_adoption_runbook_md(stable_tag=stable_tag))
    adoption_checklist_md.write_text(_adoption_checklist_md())

    package_manifest = {
        "stable_tag": stable_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stable_manifest_path": str(stable_manifest_path),
        "comparative_report_json": str(comparative_json),
        "comparative_report_md": str(comparative_md),
        "documents": {
            "release_package_md": str(release_package_md),
            "adoption_runbook_md": str(adoption_runbook_md),
            "adoption_checklist_md": str(adoption_checklist_md),
        },
        "upstream_decisions": {
            "week17_block1": str(week17_block1_decision.get("block_decision", "unknown")),
            "week17_block1_operational": str(week17_block1_decision.get("operational_decision", "unknown")),
            "week17_block2": str(week17_block2_decision.get("block_decision", "unknown")),
            "week17_block3": str(week17_block3_decision.get("block_decision", "unknown")),
            "week17_block4": str(week17_block4_decision.get("block_decision", "unknown")),
        },
        "status": "candidate_operational_stable_package",
    }
    _write_json(package_manifest_json, package_manifest)

    post_gate = _run(pre_gate_cmd)
    post_gate_json = _extract_prefixed_line(post_gate["stdout"], "Wrote JSON report:")
    post_gate_decision = "unknown"
    if post_gate_json:
        post_gate_decision = str(
            _read_json(_resolve(post_gate_json)).get("evaluation", {}).get("decision", "unknown")
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
    checks["week17_block1_promote_go"] = {
        "observed_block_decision": str(week17_block1_decision.get("block_decision", "unknown")),
        "observed_operational_decision": str(week17_block1_decision.get("operational_decision", "unknown")),
        "required": "promote+go",
        "pass": (
            str(week17_block1_decision.get("block_decision", "")) == "promote"
            and str(week17_block1_decision.get("operational_decision", "")).lower() == "go"
        ),
    }
    checks["week17_block2_promote"] = {
        "observed": str(week17_block2_decision.get("block_decision", "unknown")),
        "required": "promote",
        "pass": str(week17_block2_decision.get("block_decision", "")) == "promote",
    }
    checks["week17_block3_promote"] = {
        "observed": str(week17_block3_decision.get("block_decision", "unknown")),
        "required": "promote",
        "pass": str(week17_block3_decision.get("block_decision", "")) == "promote",
    }
    checks["week17_block4_promote"] = {
        "observed": str(week17_block4_decision.get("block_decision", "unknown")),
        "required": "promote",
        "pass": str(week17_block4_decision.get("block_decision", "")) == "promote",
    }
    checks["comparative_report_written"] = {
        "observed": comparative_json.exists() and comparative_md.exists(),
        "required": True,
        "pass": comparative_json.exists() and comparative_md.exists(),
    }
    checks["operations_docs_written"] = {
        "observed": (
            release_package_md.exists()
            and adoption_runbook_md.exists()
            and adoption_checklist_md.exists()
            and package_manifest_json.exists()
        ),
        "required": True,
        "pass": (
            release_package_md.exists()
            and adoption_runbook_md.exists()
            and adoption_checklist_md.exists()
            and package_manifest_json.exists()
        ),
    }
    checks["pre_gate_promote"] = {
        "observed": pre_gate_decision,
        "required": "promote",
        "pass": pre_gate_decision == "promote",
    }
    checks["post_gate_promote"] = {
        "observed": post_gate_decision,
        "required": "promote",
        "pass": post_gate_decision == "promote",
    }

    failed_checks = [name for name, check in checks.items() if not bool(check.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Stable operations package is ready for dependent adoption with updated comparative evidence."
        if decision == "promote"
        else "Stable operations package has unresolved checks and cannot be promoted yet."
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": stable_tag,
            "stable_manifest_path": str(stable_manifest_path),
        },
        "commands": {
            "pre_gate": pre_gate,
            "post_gate": post_gate,
            "pre_gate_cmd_pretty": " ".join(shlex.quote(x) for x in pre_gate_cmd),
        },
        "artifacts": {
            "comparative_json": str(comparative_json),
            "comparative_md": str(comparative_md),
            "release_package_md": str(release_package_md),
            "adoption_runbook_md": str(adoption_runbook_md),
            "adoption_checklist_md": str(adoption_checklist_md),
            "package_manifest_json": str(package_manifest_json),
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

    report_json = output_dir / f"{args.output_prefix}_{stamp}.json"
    report_md = output_dir / f"{args.output_prefix}_{stamp}.md"
    _write_json(report_json, report)
    report_md.write_text(_report_md(report))

    print(f"Week18 block1 JSON: {report_json}")
    print(f"Week18 block1 MD:   {report_md}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 9


if __name__ == "__main__":
    raise SystemExit(main())
