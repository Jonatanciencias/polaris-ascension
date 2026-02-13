# Acta Week 28 - Block 2 (Alert bridge observability hardening)

- Date: 2026-02-12
- Branch: `feat/breakthrough-roadmap-2026q1`
- Scope:
  - hardening incremental del bridge live (delivery ratio + latency + retry/backoff),
  - validar health-check operativo y degradación,
  - mantener gate canonico obligatorio pre/post.

## Objetivo

1. Verificar estabilidad del canal de alertas live con retry/backoff real.
2. Confirmar SLO de delivery y latencia bajo umbrales del bloque.
3. Dejar base formal para Week 28 - Block 3 (comparativo dual plataforma).

## Ejecucion

Comandos ejecutados:

- Receiver local:
  - `./venv/bin/python -u research/breakthrough_lab/week28_controlled_rollout/run_week28_block2_local_webhook_receiver.py --output-json research/breakthrough_lab/week28_controlled_rollout/week28_block2_live_webhook_capture_20260212_parallel.json --fail-first-posts 1 --response-delay-ms 12 --max-requests 13`
- Block 2:
  - `./venv/bin/python research/breakthrough_lab/week28_controlled_rollout/run_week28_block2_alert_bridge_observability.py --mode local --bridge-endpoint http://127.0.0.1:8795/week28/block2 --cycles 4 --retry-attempts 3 --backoff-seconds 0.25 --backoff-multiplier 2.0 --source-cycle-report-path research/breakthrough_lab/week28_controlled_rollout/week28_block1_monthly_continuity_20260212_152412.json --report-dir research/breakthrough_lab/week8_validation_discipline --output-dir research/breakthrough_lab/week28_controlled_rollout --output-prefix week28_block2_alert_bridge_observability`

## Artefactos

- Runner Block 2: `research/breakthrough_lab/week28_controlled_rollout/run_week28_block2_alert_bridge_observability.py`
- Receiver local: `research/breakthrough_lab/week28_controlled_rollout/run_week28_block2_local_webhook_receiver.py`
- Report JSON: `research/breakthrough_lab/week28_controlled_rollout/week28_block2_alert_bridge_observability_20260212_154302.json`
- Report MD: `research/breakthrough_lab/week28_controlled_rollout/week28_block2_alert_bridge_observability_20260212_154302.md`
- Dispatch JSON: `research/breakthrough_lab/week28_controlled_rollout/week28_block2_alert_bridge_observability_dispatch_20260212_154302.json`
- Alerts JSON: `research/breakthrough_lab/week28_controlled_rollout/week28_block2_alert_bridge_observability_alerts_20260212_154302.json`
- Scheduler health JSON: `research/breakthrough_lab/week28_controlled_rollout/week28_block2_alert_bridge_observability_scheduler_health_20260212_154302.json`
- Operational debt JSON: `research/breakthrough_lab/week28_controlled_rollout/week28_block2_alert_bridge_observability_operational_debt_20260212_154302.json`
- Webhook capture JSON: `research/breakthrough_lab/week28_controlled_rollout/week28_block2_live_webhook_capture_20260212_parallel.json`
- Canonical gates internos (pre/post):
  - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260212_154301.json`
  - `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260212_154321.json`

## Resultados

- `decision = promote`
- `failed_checks = []`
- `source_cycle_decision = promote`
- `source_alerts_decision = promote`
- `cycle_success_ratio = 1.000000`
- `dispatch_success_latency_p95_ms = 14.213787`
- `dispatch_success_latency_max_ms = 14.231512`
- `retries_rate = 0.250000`

## Decision Formal

Tracks:

- `week28_block2_bridge_delivery_reliability`: **promote**
- `week28_block2_scheduler_health_and_alerting`: **promote**
- `week28_block2_operational_debt_guardrails`: **promote**
- `week28_block2_canonical_gate_internal`: **promote**

Block decision:

- **promote**

Razonamiento:

- Week 28 Block 2 mantiene delivery estable con retry path ejercitado (`503 -> 200`), latencia dentro de policy y gate canónico pre/post en verde.

## Estado del Bloque

`Week 28 - Block 2` cerrado en `promote`.
