# Acta Week 17 - Block 1 (Despliegue controlado inicial de v0.15.0)

- Date: 2026-02-11
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar rollout inicial controlado de `v0.15.0` con snapshots extendidos,
  - mantener gate canónico pre/post y rollback SLA activo,
  - cerrar decisión formal `go|no-go`.

## Objetivo

1. Validar que el manifest estable `v0.15.0` entra en operación controlada sin romper guardrails.
2. Cubrir horizonte extendido (10 snapshots) en `1400/2048/3072`.
3. Confirmar que rollback y gates canónicos siguen operativos.

## Ejecución Formal

Intento inicial:

- `./venv/bin/python research/breakthrough_lab/week17_controlled_rollout/run_week17_block1_stable_rollout.py`
  - Artifact JSON: `research/breakthrough_lab/week17_controlled_rollout/week17_block1_stable_rollout_20260211_003512.json`
  - Canary JSON: `research/breakthrough_lab/week17_controlled_rollout/week17_block1_stable_rollout_canary_20260211_003452.json`
  - Decision: `no-go`
  - Hallazgos:
    - `canary_decision=iterate` por `t5_guardrails_all_runs` (`t5_overhead_max=3.2772%` sobre umbral de 3.0% del canary),
    - bug de parser de snapshots en runner (`executed_snapshots=-1`) corregido antes del rerun.

Hardening y rerun:

- Policy añadida: `research/breakthrough_lab/t5_reliability_abft/policy_hardening_week17_block1_stable_low_overhead.json`
  - Ajustes clave: `sampling_period=20`, `row_samples=3`, `col_samples=3`.
- Smoke de policy:
  - `research/breakthrough_lab/week17_controlled_rollout/week17_block1_policy_smoke_20260211_003831.json` (`promote`).
- Rerun de cierre:
  - `./venv/bin/python research/breakthrough_lab/week17_controlled_rollout/run_week17_block1_stable_rollout.py --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week17_block1_stable_low_overhead.json --output-prefix week17_block1_stable_rollout_rerun --seed 27031`
  - Artifact JSON: `research/breakthrough_lab/week17_controlled_rollout/week17_block1_stable_rollout_rerun_20260211_004918.json`
  - Canary JSON: `research/breakthrough_lab/week17_controlled_rollout/week17_block1_stable_rollout_rerun_canary_20260211_004858.json`
  - Decision: `go`

## Resultados Finales (rerun)

- `stable_tag = v0.15.0`
- `pre_gate_decision = promote`
- `canary_decision = promote`
- `executed_snapshots = 10` (horizonte extendido alcanzado)
- `canary_t5_disable_total = 0`
- `canary_t5_overhead_max = 1.3281%`
- `canary_correctness_max = 0.0008392333984375`
- `post_gate_decision = promote`
- `rollback_dry_run_ok = true`

## Decisión Formal

Tracks:

- `week17_block1_initial_rollout_attempt`: **iterate / no-go**
- `week17_block1_t5_low_overhead_hardening`: **promote**
- `week17_block1_extended_rollout_rerun`: **promote / go**
- `week17_block1_mandatory_canonical_gate`: **promote**

Block decision:

- **promote**
- Operational decision: **go**

Razonamiento:

- El intento inicial reveló sensibilidad de overhead en el umbral estricto del canary.
- Con hardening focalizado y rerun extendido, el bloque cierra en `go` con guardrails y gates en verde.

## Estado del Bloque

`Week 17 - Block 1` cerrado en `promote` con `GO` operativo.
