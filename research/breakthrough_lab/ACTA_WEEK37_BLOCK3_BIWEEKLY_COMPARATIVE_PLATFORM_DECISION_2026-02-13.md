# Acta Week 37 - Block 3 (Comparativo dual plataforma + decision formal)

- Date: 2026-02-13
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar comparativo mensual dual plataforma sobre baseline Week 36,
  - consolidar politica de plataforma por entorno con evidencia machine-readable,
  - mantener gate canonico obligatorio pre/post (explicito e interno).

## Objetivo

1. Confirmar no regresion critica de Week 37 contra baseline comparativo Week 36.
2. Emitir decision formal de politica Clover/rusticl para operacion continua.
3. Cerrar Week 37 Block 3 y continuidad mensual.

## Ejecucion

Comandos ejecutados:

- Attempt inicial (iterate por path de split eval incorrecto):
  - Gate pre explicito:
    - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Block 3 attempt:
    - `./venv/bin/python research/breakthrough_lab/week30_controlled_rollout/run_week30_block3_biweekly_comparative_platform_decision.py --baseline-comparative-report-path research/breakthrough_lab/week36_controlled_rollout/week36_block3_biweekly_comparative_20260213_175545.json --current-block1-report-path research/breakthrough_lab/week37_controlled_rollout/week37_block1_monthly_continuity_20260213_181108.json --current-block2-report-path research/breakthrough_lab/week37_controlled_rollout/week37_block2_alert_bridge_observability_20260213_181244.json --current-split-eval-path research/breakthrough_lab/week37_controlled_rollout/week37_block1_monthly_continuity_split_eval_20260213_181108.json --week20-debt-review-path research/breakthrough_lab/week36_controlled_rollout/WEEK30_BLOCK3_OPERATIONAL_DEBT_REVIEW.json --week21-block1-debt-path research/breakthrough_lab/week37_controlled_rollout/WEEK20_BLOCK1_MONTHLY_CYCLE_LIVE_DEBT_MATRIX.json --week21-block2-debt-path research/breakthrough_lab/week37_controlled_rollout/week37_block2_alert_bridge_observability_operational_debt_20260213_181244.json --preprod-signoff-dir research/breakthrough_lab/week37_controlled_rollout --output-dir research/breakthrough_lab/week37_controlled_rollout --output-prefix week37_block3_biweekly_comparative --report-dir research/breakthrough_lab/week8_validation_discipline`
- Recovery rerun (promote):
  - Gate pre explicito:
    - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Block 3 rerun:
    - `./venv/bin/python research/breakthrough_lab/week30_controlled_rollout/run_week30_block3_biweekly_comparative_platform_decision.py --baseline-comparative-report-path research/breakthrough_lab/week36_controlled_rollout/week36_block3_biweekly_comparative_20260213_175545.json --current-block1-report-path research/breakthrough_lab/week37_controlled_rollout/week37_block1_monthly_continuity_20260213_181108.json --current-block2-report-path research/breakthrough_lab/week37_controlled_rollout/week37_block2_alert_bridge_observability_20260213_181244.json --current-split-eval-path research/breakthrough_lab/week37_controlled_rollout/week37_block1_monthly_continuity_split_eval_20260213_181107.json --week20-debt-review-path research/breakthrough_lab/week36_controlled_rollout/WEEK30_BLOCK3_OPERATIONAL_DEBT_REVIEW.json --week21-block1-debt-path research/breakthrough_lab/week37_controlled_rollout/WEEK20_BLOCK1_MONTHLY_CYCLE_LIVE_DEBT_MATRIX.json --week21-block2-debt-path research/breakthrough_lab/week37_controlled_rollout/week37_block2_alert_bridge_observability_operational_debt_20260213_181244.json --preprod-signoff-dir research/breakthrough_lab/week37_controlled_rollout --output-dir research/breakthrough_lab/week37_controlled_rollout --output-prefix week37_block3_biweekly_comparative_rerun --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Gate post explicito:
    - `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`

## Artefactos

- Attempt JSON: `research/breakthrough_lab/week37_controlled_rollout/week37_block3_biweekly_comparative_20260213_181427.json`
- Attempt MD: `research/breakthrough_lab/week37_controlled_rollout/week37_block3_biweekly_comparative_20260213_181427.md`
- Rerun JSON: `research/breakthrough_lab/week37_controlled_rollout/week37_block3_biweekly_comparative_rerun_20260213_181550.json`
- Rerun MD: `research/breakthrough_lab/week37_controlled_rollout/week37_block3_biweekly_comparative_rerun_20260213_181550.md`
- Dashboard JSON (rerun): `research/breakthrough_lab/week37_controlled_rollout/week30_block3_biweekly_comparative_dashboard_20260213_181550.json`
- Platform policy JSON: `research/breakthrough_lab/week37_controlled_rollout/WEEK30_BLOCK3_PLATFORM_POLICY_DECISION.json`
- Operational debt review JSON: `research/breakthrough_lab/week37_controlled_rollout/WEEK30_BLOCK3_OPERATIONAL_DEBT_REVIEW.json`
- Canonical gates explicitos (recovery pre/post):
  - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_181522.json`
  - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_181633.json`
- Canonical gates internos:
  - Attempt: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_181427.json` / `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_181446.json`
  - Rerun: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_181550.json` / `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260213_181609.json`

## Resultados

- Attempt inicial: `iterate`
  - `failed_checks = [split_eval_promote, platform_policy_not_shadow_only]`
  - causa: path de `split_eval` con timestamp incorrecto (`...181108.json`, inexistente)
- Rerun de recovery: `promote`
  - `failed_checks = []`
  - `platform_policy = dual_go_clover_rusticl`
  - `split_ratio_delta_percent = 0.010484`
  - `t5_overhead_delta_percent = -0.822793`
  - `debt_high_critical_open_total = 0`

## Decision Formal

Tracks:

- `week37_block3_comparative_attempt_initial`: **iterate**
- `week37_block3_comparative_recovery_rerun`: **promote**
- `week37_block3_platform_policy_decision`: **promote**
- `week37_block3_operational_debt_review`: **promote**
- `week37_block3_canonical_gate_internal`: **promote**
- `week37_block3_canonical_gate_explicit`: **promote**

Block decision:

- **promote**

Razonamiento:

- El intento inicial falló por un error operativo de path, no por degradación técnica; el rerun con path correcto cerró en `promote`, mantuvo deuda high/critical en `0` y fijó política `dual_go_clover_rusticl`.

## Estado del Bloque

`Week 37 - Block 3` cerrado en `promote` (via rerun de recovery).
