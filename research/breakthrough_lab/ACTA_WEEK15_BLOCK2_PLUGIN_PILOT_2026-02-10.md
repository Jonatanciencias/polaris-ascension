# Acta Week 15 - Block 2 (Primer plugin piloto real)

- Date: 2026-02-10
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - ejecutar primer plugin piloto real usando `WEEK14_BLOCK6_PLUGIN_TEMPLATE.md`,
  - producir `results.json` compatible con schema,
  - cerrar decisión formal del bloque.

## Objetivo

1. Validar el template de plugin en un flujo real de ejecución.
2. Verificar contratos mínimos: gate canónico, driver smoke, correctness y guardrails T3/T5.
3. Dejar evidencia reusable para siguientes plugins.

## Ejecución Formal

Implementación runner plugin:

- Script: `research/breakthrough_lab/week15_controlled_rollout/run_week15_block2_plugin_pilot.py`

Intento inicial:

- `./venv/bin/python research/breakthrough_lab/week15_controlled_rollout/run_week15_block2_plugin_pilot.py --plugin-id rx590_plugin_pilot_v1 --owner gpu-lab --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 8 --seed 19011 --output-dir research/breakthrough_lab/week15_controlled_rollout --output-prefix week15_block2_plugin_pilot`
  - Error de implementación (`NameError`) corregido en el runner.

Segundo intento:

- mismo comando base (post-fix parcial)
  - Decision: `iterate`
  - Causa: parser de `verify_drivers --json` no aceptaba JSON multilínea.

Rerun de cierre:

- `./venv/bin/python research/breakthrough_lab/week15_controlled_rollout/run_week15_block2_plugin_pilot.py --plugin-id rx590_plugin_pilot_v1 --owner gpu-lab --sizes 1400 2048 --kernels auto_t3_controlled auto_t5_guarded --sessions 1 --iterations 8 --seed 19021 --output-dir research/breakthrough_lab/week15_controlled_rollout --output-prefix week15_block2_plugin_pilot_rerun`
  - Artifact JSON: `research/breakthrough_lab/week15_controlled_rollout/week15_block2_plugin_pilot_rerun_20260210_012358.json`
  - Artifact MD: `research/breakthrough_lab/week15_controlled_rollout/week15_block2_plugin_pilot_rerun_20260210_012358.md`
  - Results JSON: `research/breakthrough_lab/week15_controlled_rollout/week15_block2_plugin_pilot_results.json`
  - Decision: `promote`

## Resultados Finales

Plugin profile (`1400/2048`, kernels `auto_t3_controlled` + `auto_t5_guarded`):

- `avg_gflops_mean = 832.8526`
- `p95_time_ms = 14.0732`
- `max_error = 0.000579833984375`
- `delta_vs_baseline_percent = 0.0837%`
- `t3_fallback_max = 0.0`
- `t5_disable_total = 0`

Checks formales:

- `driver_smoke_good = true`
- `pre_gate_promote = true`
- `post_gate_promote = true`
- `correctness_bound = true`
- `t3_fallback_bound = true`
- `t5_disable_zero = true`

## Decisión Formal

Tracks:

- `week15_block2_runner_bootstrap_fix`: **promote**
- `week15_block2_driver_json_parser_fix`: **promote**
- `week15_block2_plugin_pilot_rerun`: **promote**
- `week15_block2_schema_compatible_results`: **promote**

Block decision:

- **promote**

Razonamiento:

- El piloto plugin cerró todos los contratos del template y dejó artefactos consumibles por futuros plugins.

## Estado del Bloque

`Week 15 - Block 2` cerrado en `promote`.
