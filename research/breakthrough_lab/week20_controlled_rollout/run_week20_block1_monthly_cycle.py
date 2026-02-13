#!/usr/bin/env python3
"""Week 20 Block 1: full monthly cycle (replay + split + consolidation)."""

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


def _dashboard_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 20 Block 1 - Monthly Full Cycle Dashboard")
    lines.append("")
    lines.append(f"- Date: {payload['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{payload['metadata']['stable_tag']}`")
    lines.append(f"- Policy: `{payload['metadata']['policy_path']}`")
    lines.append(f"- Baseline: `{payload['metadata']['baseline_path']}`")
    lines.append("")
    lines.append("## Cycle Decisions")
    lines.append("")
    lines.append("| Stage | Decision |")
    lines.append("| --- | --- |")
    lines.append(f"| weekly_replay | {payload['decisions']['weekly_replay']} |")
    lines.append(f"| weekly_replay_canary | {payload['decisions']['weekly_replay_canary']} |")
    lines.append(f"| weekly_replay_eval | {payload['decisions']['weekly_replay_eval']} |")
    lines.append(f"| split_canary | {payload['decisions']['split_canary']} |")
    lines.append(f"| split_eval | {payload['decisions']['split_eval']} |")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append(
        f"- rusticl/clover ratio min: `{float(payload['metrics']['split_ratio_min']):.6f}`"
    )
    lines.append(
        f"- split T5 overhead max: `{float(payload['metrics']['split_t5_overhead_max']):.6f}`"
    )
    lines.append(
        f"- split T5 disable total: `{int(payload['metrics']['split_t5_disable_total'])}`"
    )
    lines.append(f"- pre gate: `{payload['metrics']['pre_gate_decision']}`")
    lines.append(f"- post gate: `{payload['metrics']['post_gate_decision']}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def _runbook_md(policy_path: str, dashboard_json: str) -> str:
    lines: list[str] = []
    lines.append("# Week 20 Block 1 - Monthly Full Cycle Runbook")
    lines.append("")
    lines.append("- Cadence: mensual, ventana operativa de continuidad.")
    lines.append("- Scope: replay semanal automatizado + split Clover/rusticl + consolidacion formal.")
    lines.append(f"- Active policy: `{policy_path}`")
    lines.append(f"- Dashboard reference: `{dashboard_json}`")
    lines.append("")
    lines.append("## Mandatory Flow")
    lines.append("")
    lines.append("1. Canonical gate pre: `run_validation_suite.py --tier canonical --driver-smoke`.")
    lines.append("2. Ejecutar replay semanal automatizado sobre baseline estable.")
    lines.append("3. Ejecutar split Clover/rusticl y evaluacion contra policy activa.")
    lines.append("4. Consolidar dashboard + manifest + matriz de deuda operativa.")
    lines.append("5. Canonical gate post y cierre formal (acta + decision).")
    lines.append("")
    lines.append("## Rollback SLA")
    lines.append("")
    lines.append("- Si `split_t5_disable_total > 0`: rollback inmediato a policy previa.")
    lines.append("- Si ratio rusticl/clover cae por debajo del piso: congelar expansion cross-platform.")
    lines.append("- Si gate canonico pre/post no es promote: bloquear cierre del bloque.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _checklist_md() -> str:
    lines: list[str] = []
    lines.append("# Week 20 Block 1 - Monthly Full Cycle Checklist")
    lines.append("")
    lines.append("- [ ] Stable manifest `v0.15.0` vigente.")
    lines.append("- [ ] Replay semanal en `promote` (canary + eval).")
    lines.append("- [ ] Split Clover/rusticl en `promote` (canary + eval).")
    lines.append("- [ ] Ratio rusticl/clover >= policy floor.")
    lines.append("- [ ] T5 guardrails sin disable events.")
    lines.append("- [ ] Dashboard mensual actualizado.")
    lines.append("- [ ] Matriz de deuda sin high/critical abiertos.")
    lines.append("- [ ] Gate canonico pre/post en `promote`.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _debt_matrix() -> dict[str, Any]:
    debts = [
        {
            "debt_id": "week20_block2_scheduler_automation",
            "area": "automation",
            "severity": "medium",
            "status": "open",
            "owner": "ops-automation",
            "description": "Automatizar ejecucion programada mensual con retencion explicita de artefactos.",
            "mitigation": "Crear job mensual en CI/local con housekeeping de reportes.",
            "target_window": "week20_block2",
        },
        {
            "debt_id": "week20_block3_comparative_reporting",
            "area": "analytics",
            "severity": "medium",
            "status": "open",
            "owner": "ops-analytics",
            "description": "Consolidar informe mensual comparativo con recomendacion de plataforma por entorno.",
            "mitigation": "Cerrar reporte operativo con tabla comparativa y riesgo residual.",
            "target_window": "week20_block3",
        },
    ]
    return {
        "matrix_id": "week20-block1-monthly-full-cycle-debt-v1-2026-02-11",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "debts": debts,
        "summary": {
            "total": len(debts),
            "open": sum(1 for debt in debts if debt["status"] == "open"),
            "high_or_critical_open": sum(
                1
                for debt in debts
                if debt["status"] == "open" and debt["severity"] in {"high", "critical"}
            ),
        },
    }


def _report_md(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Week 20 Block 1 - Monthly Full Cycle")
    lines.append("")
    lines.append(f"- Date: {report['metadata']['timestamp_utc']}")
    lines.append(f"- Stable tag: `{report['metadata']['stable_tag']}`")
    lines.append(f"- Policy path: `{report['metadata']['policy_path']}`")
    lines.append(f"- T5 policy path: `{report['metadata']['t5_policy_path']}`")
    lines.append(f"- Baseline path: `{report['metadata']['baseline_path']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass |")
    lines.append("| --- | --- |")
    for key, check in report["evaluation"]["checks"].items():
        lines.append(f"| {key} | {check['pass']} |")
    lines.append("")
    lines.append("## Highlights")
    lines.append("")
    lines.append(f"- Weekly replay decision: `{report['highlights']['weekly_replay_decision']}`")
    lines.append(f"- Split canary decision: `{report['highlights']['split_canary_decision']}`")
    lines.append(f"- Split eval decision: `{report['highlights']['split_eval_decision']}`")
    lines.append(f"- split_ratio_min: `{float(report['highlights']['split_ratio_min']):.6f}`")
    lines.append(
        f"- split_t5_overhead_max: `{float(report['highlights']['split_t5_overhead_max']):.6f}`"
    )
    lines.append(
        f"- split_t5_disable_total: `{int(report['highlights']['split_t5_disable_total'])}`"
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
    parser = argparse.ArgumentParser(description="Run Week20 Block1 full monthly cycle.")
    parser.add_argument(
        "--stable-manifest-path",
        default="research/breakthrough_lab/preprod_signoff/WEEK16_BLOCK3_V0_15_0_STABLE_MANIFEST.json",
    )
    parser.add_argument(
        "--policy-path",
        default="research/breakthrough_lab/week19_controlled_rollout/policy_week19_block2_weekly_slo_v3.json",
    )
    parser.add_argument(
        "--t5-policy-path",
        default="research/breakthrough_lab/t5_reliability_abft/policy_hardening_week17_block1_stable_low_overhead.json",
    )
    parser.add_argument(
        "--baseline-path",
        default="research/breakthrough_lab/week19_controlled_rollout/week19_block1_weekly_split_maintenance_split_canary_20260211_015619.json",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[1400, 2048, 3072])
    parser.add_argument("--kernels", nargs="+", default=["auto_t3_controlled", "auto_t5_guarded"])
    parser.add_argument("--snapshots", type=int, default=6)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--pressure-size", type=int, default=896)
    parser.add_argument("--pressure-iterations", type=int, default=2)
    parser.add_argument("--pressure-pulses", type=int, default=2)
    parser.add_argument("--weekly-seed", type=int, default=20011)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[211, 509])
    parser.add_argument("--min-rusticl-ratio", type=float, default=0.85)
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
        default="research/breakthrough_lab/week20_controlled_rollout",
    )
    parser.add_argument("--output-prefix", default="week20_block1_monthly_cycle")
    args = parser.parse_args()

    stable_manifest_path = _resolve(args.stable_manifest_path)
    policy_path = _resolve(args.policy_path)
    t5_policy_path = _resolve(args.t5_policy_path)
    baseline_path = _resolve(args.baseline_path)
    preprod_dir = _resolve(args.preprod_signoff_dir)
    output_dir = _resolve(args.output_dir)
    preprod_dir.mkdir(parents=True, exist_ok=True)
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

    weekly_cmd = [
        "./venv/bin/python",
        "research/breakthrough_lab/week12_controlled_rollout/run_week12_weekly_replay_automation.py",
        "--mode",
        "local",
        "--policy-path",
        str(policy_path),
        "--t5-policy-path",
        str(t5_policy_path),
        "--baseline-path",
        str(baseline_path),
        "--sizes",
        *[str(size) for size in args.sizes],
        "--snapshots",
        str(args.snapshots),
        "--snapshot-interval-minutes",
        "60",
        "--sessions",
        str(args.sessions),
        "--iterations",
        str(args.iterations),
        "--pressure-size",
        str(args.pressure_size),
        "--pressure-iterations",
        str(args.pressure_iterations),
        "--pressure-pulses-per-snapshot",
        str(args.pressure_pulses),
        "--seed",
        str(args.weekly_seed),
        "--validation-report-dir",
        str(args.report_dir),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        f"{args.output_prefix}_weekly_replay",
    ]
    weekly = _run(weekly_cmd)
    weekly_json = _extract_prefixed_line(weekly["stdout"], "Week12 block1 JSON:")
    weekly_md = _extract_prefixed_line(weekly["stdout"], "Week12 block1 MD:")
    weekly_decision = str(_extract_prefixed_line(weekly["stdout"], "Decision:") or "unknown")

    weekly_eval_decision = "unknown"
    weekly_canary_decision = "unknown"
    weekly_eval_json = ""
    weekly_canary_json = ""
    if weekly_json:
        weekly_payload = _read_json(_resolve(weekly_json))
        weekly_eval_decision = str(
            weekly_payload.get("steps", {}).get("policy_eval", {}).get("decision", "unknown")
        )
        weekly_canary_decision = str(
            weekly_payload.get("steps", {}).get("canary_run", {}).get("decision", "unknown")
        )
        weekly_eval_json = str(weekly_payload.get("artifacts", {}).get("eval_json", ""))
        weekly_canary_json = str(weekly_payload.get("artifacts", {}).get("canary_json", ""))

    split_cmd = [
        "./venv/bin/python",
        "research/breakthrough_lab/platform_compatibility/run_week9_block4_stress_split.py",
        "--seeds",
        *[str(seed) for seed in args.split_seeds],
        "--sizes",
        *[str(size) for size in args.sizes],
        "--kernels",
        *[str(kernel) for kernel in args.kernels],
        "--sessions",
        str(args.sessions),
        "--iterations",
        str(args.iterations),
        "--pressure-size",
        str(args.pressure_size),
        "--pressure-iterations",
        str(args.pressure_iterations),
        "--pressure-pulses-per-seed",
        str(args.pressure_pulses),
        "--t5-policy-path",
        str(t5_policy_path),
        "--baseline-block3-path",
        str(baseline_path),
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        f"{args.output_prefix}_split_canary",
    ]
    split = _run(split_cmd)
    split_json = _extract_prefixed_line(split["stdout"], "Week9 block4 JSON:")
    split_md = _extract_prefixed_line(split["stdout"], "Week9 block4 MD:")
    split_decision = str(_extract_prefixed_line(split["stdout"], "Decision:") or "unknown")

    split_t5_guardrails_ok = False
    split_no_regression_ok = False
    split_t5_overhead_max = 999.0
    split_t5_disable_total = 999
    if split_json:
        split_payload = _read_json(_resolve(split_json))
        split_checks = split_payload.get("evaluation", {}).get("checks", {})
        split_t5_guardrails_ok = bool(split_checks.get("t5_guardrails_all_runs", {}).get("pass", False))
        split_no_regression_ok = bool(
            split_checks.get("no_regression_vs_block3_clover", {}).get("pass", False)
        )
        split_t5_overhead_max = float(
            split_checks.get("t5_guardrails_all_runs", {}).get("observed_overhead_max", 999.0)
        )
        split_t5_disable_total = int(
            split_checks.get("t5_guardrails_all_runs", {}).get("observed_disable_total", 999)
        )

    split_eval_cmd = [
        "./venv/bin/python",
        "research/breakthrough_lab/week12_controlled_rollout/evaluate_week12_platform_split_policy.py",
        "--split-artifact",
        str(split_json or ""),
        "--policy-path",
        str(policy_path),
        "--min-rusticl-ratio",
        str(args.min_rusticl_ratio),
        "--required-sizes",
        *[str(size) for size in args.sizes],
        "--output-dir",
        str(output_dir),
        "--output-prefix",
        f"{args.output_prefix}_split_eval",
    ]
    split_eval = _run(split_eval_cmd) if split_json else {
        "command": split_eval_cmd,
        "returncode": 2,
        "stdout": "",
        "stderr": "split_json_missing",
    }
    split_eval_json = _extract_prefixed_line(split_eval["stdout"], "Week12 block2 eval JSON:")
    split_eval_md = _extract_prefixed_line(split_eval["stdout"], "Week12 block2 eval MD:")
    split_eval_decision = str(_extract_prefixed_line(split_eval["stdout"], "Decision:") or "unknown")

    split_ratio_min = 0.0
    split_ratio_ok = False
    split_required_sizes_ok = False
    if split_eval_json:
        split_eval_payload = _read_json(_resolve(split_eval_json))
        split_eval_checks = split_eval_payload.get("evaluation", {}).get("checks", {})
        split_required_sizes_ok = bool(
            split_eval_checks.get("required_sizes_present_on_split", {}).get("pass", False)
        )
        split_ratio_rows = split_eval_checks.get("rusticl_ratio_vs_clover", {}).get("rows", [])
        ratios = [float(row.get("ratio_rusticl_vs_clover", 0.0)) for row in split_ratio_rows]
        split_ratio_min = float(min(ratios)) if ratios else 0.0
        split_ratio_ok = bool(split_eval_checks.get("rusticl_ratio_vs_clover", {}).get("pass", False))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dashboard_payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": stable_tag,
            "policy_path": str(policy_path),
            "baseline_path": str(baseline_path),
        },
        "decisions": {
            "weekly_replay": weekly_decision,
            "weekly_replay_canary": weekly_canary_decision,
            "weekly_replay_eval": weekly_eval_decision,
            "split_canary": split_decision,
            "split_eval": split_eval_decision,
        },
        "metrics": {
            "split_ratio_min": float(split_ratio_min),
            "split_t5_overhead_max": float(split_t5_overhead_max),
            "split_t5_disable_total": int(split_t5_disable_total),
            "pre_gate_decision": pre_gate_decision,
            "post_gate_decision": "pending",
        },
        "artifacts": {
            "weekly_json": weekly_json,
            "weekly_eval_json": weekly_eval_json,
            "weekly_canary_json": weekly_canary_json,
            "split_json": split_json,
            "split_eval_json": split_eval_json,
        },
    }
    dashboard_json = output_dir / f"week20_block1_monthly_cycle_dashboard_{stamp}.json"
    dashboard_md = output_dir / f"week20_block1_monthly_cycle_dashboard_{stamp}.md"
    _write_json(dashboard_json, dashboard_payload)
    dashboard_md.write_text(_dashboard_md(dashboard_payload))

    runbook_path = preprod_dir / "WEEK20_BLOCK1_MONTHLY_CYCLE_RUNBOOK.md"
    checklist_path = preprod_dir / "WEEK20_BLOCK1_MONTHLY_CYCLE_CHECKLIST.md"
    debt_path = preprod_dir / "WEEK20_BLOCK1_MONTHLY_CYCLE_LIVE_DEBT_MATRIX.json"
    manifest_path = preprod_dir / "WEEK20_BLOCK1_MONTHLY_CYCLE_MANIFEST.json"

    runbook_path.write_text(_runbook_md(str(policy_path), str(dashboard_json)))
    checklist_path.write_text(_checklist_md())
    debt_payload = _debt_matrix()
    _write_json(debt_path, debt_payload)

    manifest_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stable_tag": stable_tag,
        "policy_path": str(policy_path),
        "baseline_path": str(baseline_path),
        "dashboard_json": str(dashboard_json),
        "dashboard_md": str(dashboard_md),
        "runbook_path": str(runbook_path),
        "checklist_path": str(checklist_path),
        "debt_matrix_path": str(debt_path),
        "weekly_json": weekly_json,
        "weekly_eval_json": weekly_eval_json,
        "split_json": split_json,
        "split_eval_json": split_eval_json,
        "status": "week20_block1_monthly_cycle_candidate",
    }
    _write_json(manifest_path, manifest_payload)

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

    dashboard_payload["metrics"]["post_gate_decision"] = post_gate_decision
    _write_json(dashboard_json, dashboard_payload)
    dashboard_md.write_text(_dashboard_md(dashboard_payload))

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
    checks["weekly_replay_promote"] = {
        "observed": weekly_decision.lower(),
        "required": "promote",
        "pass": weekly_decision.lower() == "promote",
    }
    checks["weekly_replay_canary_promote"] = {
        "observed": weekly_canary_decision.lower(),
        "required": "promote",
        "pass": weekly_canary_decision.lower() == "promote",
    }
    checks["weekly_replay_eval_promote"] = {
        "observed": weekly_eval_decision.lower(),
        "required": "promote",
        "pass": weekly_eval_decision.lower() == "promote",
    }
    checks["split_canary_promote"] = {
        "observed": split_decision.lower(),
        "required": "promote",
        "pass": split_decision.lower() == "promote",
    }
    checks["split_eval_promote"] = {
        "observed": split_eval_decision.lower(),
        "required": "promote",
        "pass": split_eval_decision.lower() == "promote",
    }
    checks["split_required_sizes_present"] = {
        "observed": split_required_sizes_ok,
        "required": True,
        "pass": split_required_sizes_ok,
    }
    checks["split_ratio_floor"] = {
        "observed_min": float(split_ratio_min),
        "required_min": float(args.min_rusticl_ratio),
        "pass": split_ratio_ok and float(split_ratio_min) >= float(args.min_rusticl_ratio),
    }
    checks["split_t5_guardrails"] = {
        "observed": split_t5_guardrails_ok,
        "required": True,
        "pass": split_t5_guardrails_ok,
    }
    checks["split_no_regression_vs_baseline"] = {
        "observed": split_no_regression_ok,
        "required": True,
        "pass": split_no_regression_ok,
    }
    checks["consolidation_artifacts_written"] = {
        "observed": (
            dashboard_json.exists()
            and dashboard_md.exists()
            and runbook_path.exists()
            and checklist_path.exists()
            and debt_path.exists()
            and manifest_path.exists()
        ),
        "required": True,
        "pass": (
            dashboard_json.exists()
            and dashboard_md.exists()
            and runbook_path.exists()
            and checklist_path.exists()
            and debt_path.exists()
            and manifest_path.exists()
        ),
    }
    checks["no_high_critical_open_debt"] = {
        "observed": int(debt_payload.get("summary", {}).get("high_or_critical_open", 99)),
        "required_max": 0,
        "pass": int(debt_payload.get("summary", {}).get("high_or_critical_open", 99)) <= 0,
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

    failed_checks = [key for key, check in checks.items() if not bool(check.get("pass"))]
    decision = "promote" if not failed_checks else "iterate"
    rationale = (
        "Week20 Block1 monthly full cycle completed with replay/split/consolidation stable and canonical gates green."
        if decision == "promote"
        else "Week20 Block1 found unresolved failures in replay, split, consolidation, or canonical gates."
    )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stable_tag": stable_tag,
            "stable_manifest_path": str(stable_manifest_path),
            "policy_path": str(policy_path),
            "t5_policy_path": str(t5_policy_path),
            "baseline_path": str(baseline_path),
            "sizes": [int(size) for size in args.sizes],
            "split_seeds": [int(seed) for seed in args.split_seeds],
            "weekly_seed": int(args.weekly_seed),
        },
        "commands": {
            "pre_gate": pre_gate,
            "weekly_replay": weekly,
            "split_canary": split,
            "split_eval": split_eval,
            "post_gate": post_gate,
            "weekly_cmd_pretty": " ".join(shlex.quote(x) for x in weekly_cmd),
            "split_cmd_pretty": " ".join(shlex.quote(x) for x in split_cmd),
            "split_eval_cmd_pretty": " ".join(shlex.quote(x) for x in split_eval_cmd),
        },
        "artifacts": {
            "weekly_json": weekly_json,
            "weekly_md": weekly_md,
            "weekly_canary_json": weekly_canary_json,
            "weekly_eval_json": weekly_eval_json,
            "split_json": split_json,
            "split_md": split_md,
            "split_eval_json": split_eval_json,
            "split_eval_md": split_eval_md,
            "dashboard_json": str(dashboard_json),
            "dashboard_md": str(dashboard_md),
            "runbook_path": str(runbook_path),
            "checklist_path": str(checklist_path),
            "debt_matrix_path": str(debt_path),
            "manifest_path": str(manifest_path),
            "pre_gate_json": pre_gate_json,
            "post_gate_json": post_gate_json,
        },
        "highlights": {
            "weekly_replay_decision": weekly_decision,
            "split_canary_decision": split_decision,
            "split_eval_decision": split_eval_decision,
            "split_ratio_min": float(split_ratio_min),
            "split_t5_overhead_max": float(split_t5_overhead_max),
            "split_t5_disable_total": int(split_t5_disable_total),
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

    print(f"Week20 block1 JSON: {report_json}")
    print(f"Week20 block1 MD:   {report_md}")
    print(f"Dashboard JSON:     {dashboard_json}")
    print(f"Dashboard MD:       {dashboard_md}")
    print(f"Manifest:           {manifest_path}")
    print(f"Decision: {decision}")
    print(f"Failed checks: {failed_checks}")
    return 0 if decision == "promote" else 8


if __name__ == "__main__":
    raise SystemExit(main())
