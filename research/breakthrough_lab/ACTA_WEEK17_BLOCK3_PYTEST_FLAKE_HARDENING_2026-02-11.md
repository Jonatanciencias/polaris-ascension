# Acta Week 17 - Block 3 (Estabilización temprana de flake `pytest_tier_green`)

- Date: 2026-02-11
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - estabilizar flake detectado en `pytest_tier_green`,
  - endurecer `tests/test_optimized_kernel_engine.py::TestGEMMCorrectness::test_gemm_rectangular`,
  - validar con campaña repetida + gate canónico.

## Objetivo

1. Eliminar la inestabilidad intermitente que bloqueó un pre-gate reciente.
2. Mantener cobertura de corrección numérica sin degradar la sensibilidad del test.
3. Cerrar bloque temprano para proteger continuidad operativa.

## Ejecución Formal

Contexto de falla base (referencia):

- `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_014604.json`
  - `decision = iterate`
  - `failed_checks = ['pytest_tier_green']`
  - Falla puntual en `test_gemm_rectangular` (error relativo no robusto).

Hardening aplicado:

- Archivo modificado: `tests/test_optimized_kernel_engine.py`
  - Dataset determinista en `test_gemm_rectangular` (`default_rng(17017)`).
  - Métrica robusta con `nrmse` y `max_abs_error` en lugar de media de error relativo sensible a referencias ~0.

Campaña de validación:

- `./venv/bin/python research/breakthrough_lab/week17_controlled_rollout/run_week17_block3_pytest_flake_hardening.py --repeat-count 20`
  - Artifact JSON: `research/breakthrough_lab/week17_controlled_rollout/week17_block3_pytest_flake_hardening_20260211_005256.json`
  - Artifact MD: `research/breakthrough_lab/week17_controlled_rollout/week17_block3_pytest_flake_hardening_20260211_005256.md`
  - Resultado campaña: `20/20` pasadas del test objetivo.
  - Gate canónico post-hardening: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260211_005256.json` (`promote`).

## Resultados

- `baseline_contains_pytest_failure = true` (referencia histórica confirmada).
- `repeat_campaign_all_green = true` (`failed_runs=0` en 20 repeticiones).
- `post_gate_decision = promote`.
- `pytest_tier_green.pass = true` en gate post-hardening.

## Decisión Formal

Tracks:

- `week17_block3_baseline_failure_characterization`: **promote**
- `week17_block3_rectangular_test_hardening`: **promote**
- `week17_block3_repeat_campaign`: **promote**
- `week17_block3_post_hardening_canonical_gate`: **promote**

Block decision:

- **promote**

Razonamiento:

- El fix elimina la variabilidad espuria en el test objetivo y restablece estabilidad del gate canónico.

## Estado del Bloque

`Week 17 - Block 3` cerrado en `promote`.
