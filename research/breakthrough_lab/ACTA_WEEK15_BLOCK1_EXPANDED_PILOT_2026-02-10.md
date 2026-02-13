# Acta Week 15 - Block 1 (Piloto ampliado 1400/2048/3072)

- Date: 2026-02-10
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ampliar piloto RX590 a `sizes 1400/2048/3072`,
  - mantener el mismo marco de gate canónico y rollback SLA,
  - cerrar decisión operativa formal (`go|no-go`).

## Objetivo

1. Validar estabilidad del perfil ampliado sin romper correctness ni rollback discipline.
2. Confirmar que T5 se mantiene dentro de guardrails en el nuevo alcance.
3. Cerrar el bloque con evidencia machine-readable.

## Ejecución Formal

Intento inicial (alcance ampliado):

- `./venv/bin/python research/breakthrough_lab/week14_controlled_rollout/run_week14_block5_rx590_dry_run.py --duration-minutes 4 --snapshot-interval-minutes 1 --sizes 1400 2048 3072 --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 0 --seed 18011 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week14_block1_low_overhead.json --baseline-block5-path research/breakthrough_lab/week14_controlled_rollout/week14_block2_extended_horizon_canary_20260209_132519.json --preprod-signoff-dir research/breakthrough_lab/week15_controlled_rollout --output-dir research/breakthrough_lab/week15_controlled_rollout --output-prefix week15_block1_expanded_pilot`
  - Artifact JSON: `research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_20260210_011240.json`
  - Canary JSON: `research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_canary_20260210_011220.json`
  - Decision: `no-go`
  - Hallazgo: `t5_disable_events_total=1`, `t5_overhead_max=7.4731%`.

Hardening aplicado:

- Nueva policy T5 para tamaños ampliados: `research/breakthrough_lab/t5_reliability_abft/policy_hardening_week15_block1_expanded_sizes.json`
  - Ajustes principales: `sampling_period=16`, `row_samples=4`, `col_samples=4`.

Rerun de cierre (post-hardening):

- `./venv/bin/python research/breakthrough_lab/week14_controlled_rollout/run_week14_block5_rx590_dry_run.py --duration-minutes 4 --snapshot-interval-minutes 1 --sizes 1400 2048 3072 --sessions 1 --iterations 8 --pressure-size 896 --pressure-iterations 2 --pressure-pulses-per-snapshot 0 --seed 18031 --t5-policy-path research/breakthrough_lab/t5_reliability_abft/policy_hardening_week15_block1_expanded_sizes.json --baseline-block5-path research/breakthrough_lab/week14_controlled_rollout/week14_block2_extended_horizon_canary_20260209_132519.json --preprod-signoff-dir research/breakthrough_lab/week15_controlled_rollout --output-dir research/breakthrough_lab/week15_controlled_rollout --output-prefix week15_block1_expanded_pilot_rerun`
  - Artifact JSON: `research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_rerun_20260210_011756.json`
  - Canary JSON: `research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_rerun_canary_20260210_011736.json`
  - Decision: `go`

## Resultados Finales (rerun)

- `pre_gate_decision = promote`
- `canary_decision = promote`
- `canary_t5_disable_total = 0`
- `canary_correctness_max = 0.0008697509765625`
- `post_gate_decision = promote`
- `rollback_dry_run_ok = true`

## Decisión Formal

Tracks:

- `week15_block1_initial_expanded_attempt`: **iterate / no-go**
- `week15_block1_t5_expanded_policy_hardening`: **promote**
- `week15_block1_expanded_rerun`: **promote / go**
- `week15_block1_mandatory_canonical_gate`: **promote**

Block decision:

- **promote**
- Operational decision: **go**

Razonamiento:

- El primer intento expuso inestabilidad T5 específica del alcance ampliado.
- Con hardening focalizado y rerun estricto, el bloque cierra en `go` manteniendo el contrato de guardrails.

## Estado del Bloque

`Week 15 - Block 1` cerrado en `promote` con `GO` operativo.
