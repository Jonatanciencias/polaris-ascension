# Acta Week 11 - Block 1 (Continuous Monitoring Baseline)

- Date: 2026-02-09
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - establecer baseline de monitoreo continuo post `GO` de Week 10 Block 2.4.1,
  - medir impacto de policy T5 nueva vs previa en GFLOPS/latencia/overhead,
  - cerrar decision formal con gate canonico obligatorio.

## Objetivo

1. Confirmar estabilidad de guardrails (`disable_events=0`) en scope bajo para `1400/2048`.
2. Cuantificar deltas de performance entre `t5_new_policy` y `t5_old_policy` sin perder correccion numerica.
3. Mantener disciplina de validacion: gate canonico `promote` antes de cierre del bloque.

## Ejecucion Formal

Gate canonico (precondicion):

- `./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline`
  - Artifact: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260209_003401.json`
  - Decision: `promote`

Probe baseline T5 effect (Clover + intento rusticl in-process):

- Artifact JSON: `research/breakthrough_lab/week11_controlled_rollout/week11_t5_effect_probe_20260209_003453.json`
- Artifact MD: `research/breakthrough_lab/week11_controlled_rollout/week11_t5_effect_probe_20260209_003453.md`

Probe rusticl en subprocess (metodo determinista para env de plataforma):

- Artifact JSON: `research/breakthrough_lab/week11_controlled_rollout/week11_t5_effect_probe_rusticl_20260209_003557.json`
- Artifact MD: `research/breakthrough_lab/week11_controlled_rollout/week11_t5_effect_probe_rusticl_20260209_003557.md`

## Resultados

### Gate canonico

- `validate_breakthrough_results.py`: `rc=0`
- `pytest -q tests/`: `85 passed`
- `verify_drivers.py --json`: `rc=0`, `overall_status=good`
- Decision final del gate: `promote`

### Clover (policy nueva vs previa)

- Size `1400`: delta GFLOPS `-0.284%` (`908.503` vs `911.087`)
- Size `2048`: delta GFLOPS `+0.303%` (`777.677` vs `775.330`)
- `t5_disable_events_total`: `0` en new y old (ambos sizes)
- `t5_overhead_percent_max` (new): `1.4485%`
- Error max observado: `5.79833984375e-4`

### Rusticl (subprocess, policy nueva vs previa)

- Size `1400`: delta GFLOPS `+0.549%` (`921.047` vs `916.018`)
- Size `2048`: delta GFLOPS `-0.123%` (`716.664` vs `717.547`)
- `t5_disable_events_total`: `0` en new y old (ambos sizes)
- `t5_overhead_percent_max` (new): `1.2709%` (1400), `0.4636%` (2048)
- Error max observado: `5.79833984375e-4`

### Nota operacional

- El primer probe registró `12` errores en rusticl in-process (inventario sin GPU por inicializacion tardia del entorno).
- El probe rusticl en subprocess cierra con `0` errores y evidencia valida para baseline.

## Decision Formal

Tracks:

- `week11_block1_validation_gate`: **promote**
- `week11_block1_t5_effect_probe_clover`: **promote**
- `week11_block1_t5_effect_probe_rusticl_subprocess`: **promote**
- `week11_block1_continuous_monitoring_baseline`: **promote**

Block decision:

- **promote**

Razonamiento:

- Guardrails T5 se mantienen estables (`disable_events=0`) con overhead bajo y sin deriva de error.
- Los deltas de GFLOPS son pequenos y mixtos por size/plataforma, sin regresion global material.
- La disciplina de validacion canonica permanece verde.

## Estado del Bloque

`Week 11 - Block 1` cerrado en `promote` como baseline de monitoreo continuo para abrir Week 11 Block 2.
