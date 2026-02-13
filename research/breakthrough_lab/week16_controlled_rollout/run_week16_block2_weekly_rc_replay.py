#!/usr/bin/env python3
"""Week 16 Block 2: weekly automated replay on RC with drift acta payload."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RC_MANIFEST = (
    "research/breakthrough_lab/preprod_signoff/WEEK15_BLOCK3_FRAMEWORK_RC_MANIFEST.json"
)
DEFAULT_POLICY = (
    "research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json"
)
DEFAULT_T5_POLICY = (
    "research/breakthrough_lab/t5_reliability_abft/policy_hardening_week15_block1_expanded_sizes.json"
)
DEFAULT_BASELINE = (
    "research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_rerun_canary_20260210_011736.json"
)
DEFAULT_VALIDATION_DIR = "research/breakthrough_lab/week8_validation_discipline"
DEFAULT_OUTPUT_DIR = "research/breakthrough_lab/week16_controlled_rollout"


def _resolve(path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
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


def _aggregate(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for run in report.get("runs", []):
        if run.get("status") != "ok":
            continue
        key = (str(run.get("kernel")), int(run.get("size")))
        grouped.setdefault(key, []).append(run)

    out: dict[tuple[str, int], dict[str, float]] = {}
    for key, entries in grouped.items():
        avg_gflops = [float(e["metrics"]["avg_mean_gflops"]) for e in entries]
        p95_ms = [float(e["metrics"]["p95_time_ms"]) for e in entries]
        out[key] = {
            "avg_gflops_mean": float(statistics.mean(avg_gflops)),
            "p95_time_ms_mean": float(statistics.mean(p95_ms)),
        }
    return out


def _build_drift_rows(
    *,
    baseline_report: dict[str, Any],
    replay_report: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline = _aggregate(baseline_report)
    replay = _aggregate(replay_report)
    keys = sorted(set(baseline.keys()) | set(replay.keys()), key=lambda k: (k[0], k[1]))
    rows: list[dict[str, Any]] = []
    for kernel, size in keys:
        base = baseline.get((kernel, size))
        current = replay.get((kernel, size))
        row: dict[str, Any] = {
            "kernel": kernel,
            "size": int(size),
            "baseline_present": base is not None,
            "replay_present": current is not None,
        }
        if base and current:
            base_avg = float(base["avg_gflops_mean"])
            cur_avg = float(current["avg_gflops_mean"])
            base_p95 = float(base["p95_time_ms_mean"])
            cur_p95 = float(current["p95_time_ms_mean"])
            thr_drift = 0.0 if base_avg == 0.0 else (cur_avg - base_avg) / base_avg * 100.0
            p95_drift = 0.0 if base_p95 == 0.0 else (cur_p95 - base_p95) / base_p95 * 100.0
            row.update(
                {
                    "baseline_avg_gflops": base_avg,
                    "replay_avg_gflops": cur_avg,
                    "throughput_drift_percent": float(thr_drift),
                    "baseline_p95_ms": base_p95,
                    "replay_p95_ms": cur_p95,
                    "p95_drift_percent": float(p95_drift),
                }
            )
        rows.append(row)
    return rows


def _md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 16 Block 2 - Weekly RC Replay + Drift")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- RC tag: `{report['metadata']['rc_tag']}`")
    lines.append(f"- Policy: `{report['metadata']['policy_path']}`")
    lines.append(f"- Sizes: {report['metadata']['sizes']}")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for name, check in report["evaluation"]["checks"].items():
        lines.append(f"| {name} | {check['pass']} |")
    lines.append("")
    lines.append("## Drift Rows")
    lines.append("")
    lines.append("| Kernel | Size | Thr drift % | P95 drift % |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in report["drift"]["rows"]:
        if not row.get("baseline_present") or not row.get("replay_present"):
            continue
        lines.append(
            f"| {row['kernel']} | {row['size']} | {row['throughput_drift_percent']:.3f} | {row['p95_drift_percent']:.3f} |"
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
    parser = argparse.ArgumentParser(description="Run Week16 Block2 weekly replay over RC.")
    parser.add_argument("--rc-manifest-path", default=DEFAULT_RC_MANIFEST)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY)
    parser.add_argument("--t5-policy-path", default=DEFAULT_T5_POLICY)
    parser.add_argument("--baseline-path", default=DEFAULT_BASELINE)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048, 3072])
    parser.add_argument("--snapshots", type=int, default=6)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=26111)
    parser.add_argument("--validation-report-dir", default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="week16_block2_weekly_rc_replay")
    args = parser.parse_args()

    manifest_path = _resolve(args.rc_manifest_path)
    policy_path = _resolve(args.policy_path)
    baseline_path = _resolve(args.baseline_path)
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = _read_json(manifest_path)

    gate_cmd = [
        sys.executable,
        "scripts/run_validation_suite.py",
        "--tier",
        "canonical",
        "--driver-smoke",
        "--report-dir",
        str(args.validation_report_dir),
    ]
    pre_gate = _run(gate_cmd)
    pre_gate_json = _extract_prefixed_line(pre_gate["stdout"], "Wrote JSON report:")
    pre_gate_decision = "unknown"
    if pre_gate_json:
        pre_gate_decision = str(
            _read_json(_resolve(pre_gate_json)).get("evaluation", {}).get("decision", "unknown")
        )

    automation_prefix = f"{args.output_prefix}_automation"
    replay_cmd = [
        sys.executable,
        "research/breakthrough_lab/week12_controlled_rollout/run_week12_weekly_replay_automation.py",
        "--mode",
        "local",
        "--policy-path",
        str(args.policy_path),
        "--t5-policy-path",
        str(args.t5_policy_path),
        "--baseline-path",
        str(args.baseline_path),
        "--sizes",
        *[str(s) for s in args.sizes],
        "--snapshots",
        str(args.snapshots),
        "--snapshot-interval-minutes",
        "60",
        "--sessions",
        str(args.sessions),
        "--iterations",
        str(args.iterations),
        "--seed",
        str(args.seed),
        "--validation-report-dir",
        str(args.validation_report_dir),
        "--output-dir",
        str(args.output_dir),
        "--output-prefix",
        automation_prefix,
    ]
    replay_run = _run(replay_cmd)
    replay_json = _extract_prefixed_line(replay_run["stdout"], "Week12 block1 JSON:")
    replay_md = _extract_prefixed_line(replay_run["stdout"], "Week12 block1 MD:")
    replay_decision = _extract_prefixed_line(replay_run["stdout"], "Decision:")

    replay_payload: dict[str, Any] = {}
    replay_eval_payload: dict[str, Any] = {}
    replay_canary_payload: dict[str, Any] = {}
    eval_decision = "unknown"
    if replay_json:
        replay_payload = _read_json(_resolve(replay_json))
        eval_json = str(replay_payload.get("artifacts", {}).get("eval_json", ""))
        canary_json = str(replay_payload.get("artifacts", {}).get("canary_json", ""))
        if eval_json:
            replay_eval_payload = _read_json(_resolve(eval_json))
            eval_decision = str(
                replay_eval_payload.get("evaluation", {}).get("decision", "unknown")
            )
        if canary_json:
            replay_canary_payload = _read_json(_resolve(canary_json))

    drift_rows = _build_drift_rows(
        baseline_report=_read_json(baseline_path),
        replay_report=replay_canary_payload if replay_canary_payload else {"runs": []},
    )
    max_abs_throughput_drift = max(
        (
            abs(float(row.get("throughput_drift_percent", 0.0)))
            for row in drift_rows
            if row.get("baseline_present") and row.get("replay_present")
        ),
        default=999.0,
    )
    max_positive_p95_drift = max(
        (
            float(row.get("p95_drift_percent", 0.0))
            for row in drift_rows
            if row.get("baseline_present") and row.get("replay_present")
        ),
        default=999.0,
    )

    post_gate = _run(gate_cmd)
    post_gate_json = _extract_prefixed_line(post_gate["stdout"], "Wrote JSON report:")
    post_gate_decision = "unknown"
    if post_gate_json:
        post_gate_decision = str(
            _read_json(_resolve(post_gate_json)).get("evaluation", {}).get("decision", "unknown")
        )

    checks: dict[str, dict[str, Any]] = {}
    checks["rc_manifest_exists"] = {
        "observed": manifest_path.exists(),
        "required": True,
        "pass": manifest_path.exists(),
    }
    checks["rc_tag_is_rc"] = {
        "observed": str(manifest.get("rc_tag", "unknown")),
        "required_prefix": "v0.15.0-rc",
        "pass": str(manifest.get("rc_tag", "")).startswith("v0.15.0-rc"),
    }
    checks["automation_replay_promote"] = {
        "observed": str(replay_decision or "unknown").lower(),
        "required": "promote",
        "pass": str(replay_decision or "").lower() == "promote",
    }
    checks["weekly_eval_promote"] = {
        "observed": eval_decision,
        "required": "promote",
        "pass": eval_decision == "promote",
    }
    checks["drift_rows_present"] = {
        "observed": len(drift_rows),
        "required_min": 1,
        "pass": len(drift_rows) > 0,
    }
    checks["throughput_drift_bound"] = {
        "observed_max_abs_percent": float(max_abs_throughput_drift),
        "required_max_abs_percent": 15.0,
        "pass": float(max_abs_throughput_drift) <= 15.0,
    }
    checks["p95_drift_bound"] = {
        "observed_max_percent": float(max_positive_p95_drift),
        "required_max_percent": 20.0,
        "pass": float(max_positive_p95_drift) <= 20.0,
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
        "Weekly replay over RC is stable, policy-compliant, and within drift thresholds."
        if decision == "promote"
        else "Weekly replay over RC has unresolved drift or gate failures."
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "rc_tag": str(manifest.get("rc_tag", "unknown")),
            "rc_manifest_path": str(manifest_path),
            "policy_path": str(policy_path),
            "baseline_path": str(baseline_path),
            "sizes": [int(s) for s in args.sizes],
            "snapshots": int(args.snapshots),
            "seed": int(args.seed),
        },
        "commands": {
            "pre_gate": pre_gate,
            "replay_automation": replay_run,
            "post_gate": post_gate,
            "replay_cmd_pretty": " ".join(shlex.quote(x) for x in replay_cmd),
        },
        "artifacts": {
            "replay_json": replay_json,
            "replay_md": replay_md,
            "replay_eval_json": str(replay_payload.get("artifacts", {}).get("eval_json", "")),
            "replay_canary_json": str(replay_payload.get("artifacts", {}).get("canary_json", "")),
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
        },
        "drift": {
            "rows": drift_rows,
            "max_abs_throughput_drift_percent": float(max_abs_throughput_drift),
            "max_positive_p95_drift_percent": float(max_positive_p95_drift),
        },
        "evaluation": {
            "checks": checks,
            "failed_checks": failed_checks,
            "decision": decision,
            "rationale": rationale,
        },
    }

    json_path = out_dir / f"{args.output_prefix}_{stamp}.json"
    md_path = out_dir / f"{args.output_prefix}_{stamp}.md"
    _write_json(json_path, report)
    md_path.write_text(_md(report))

    print(f"Week16 block2 JSON: {json_path}")
    print(f"Week16 block2 MD:   {md_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 7


if __name__ == "__main__":
    raise SystemExit(main())
