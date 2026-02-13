# Week 12 Weekly Escalation Matrix

## Purpose

Define escalation ownership and response windows for weekly controlled rollout operations.

## Roles

- `Run Owner`: ejecuta canary/replay y valida artifacts.
- `Policy Owner`: decide refine/promote sobre thresholds SLO.
- `Platform Owner`: atiende desviaciones Clover/rusticl y disponibilidad OpenCL.

## Escalation Table

| Severity | Trigger examples | Owner | Response SLA | Mandatory action |
| --- | --- | --- | --- | --- |
| `SEV1` | correctness > 0.001, disable_events > 0, split size missing | Run Owner + Platform Owner | 15 min | Apply rollback, run canonical gate, freeze promotion |
| `SEV2` | t5_overhead > 3.0%, t3_fallback > 0.08, rusticl ratio < 0.85 | Policy Owner + Platform Owner | 60 min | Keep iterate, open hardening block, rerun controlled replay |
| `SEV3` | throughput drift abs > 3%, p95 drift > 8% | Policy Owner | 4 h | Log drift incident, tune thresholds/policy, schedule rerun |
| `SEV4` | all checks pass | Run Owner | 24 h | Record promote status and continue weekly cadence |

## Escalation Protocol

1. Classify event using `WEEK12_WEEKLY_ALERT_SLA.json`.
2. Execute mandatory action for severity level.
3. Attach evidence paths and decision JSON to weekly acta.
4. If `SEV1` or `SEV2`, block promotion until rerun is `promote`.
