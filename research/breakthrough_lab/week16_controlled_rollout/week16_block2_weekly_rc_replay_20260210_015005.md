# Week 16 Block 2 - Weekly RC Replay + Drift

- Date: 2026-02-10T01:50:05.856814+00:00
- RC tag: `v0.15.0-rc1`
- Policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week11_controlled_rollout/policy_week11_block3_weekly_slo_v1.json`
- Sizes: [1400, 2048, 3072]

## Checks

| Check | Pass |
| --- | --- |
| rc_manifest_exists | True |
| rc_tag_is_rc | True |
| automation_replay_promote | True |
| weekly_eval_promote | True |
| drift_rows_present | True |
| throughput_drift_bound | True |
| p95_drift_bound | True |
| pre_gate_promote | False |
| post_gate_promote | True |

## Drift Rows

| Kernel | Size | Thr drift % | P95 drift % |
| --- | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | 0.173 | -0.073 |
| auto_t3_controlled | 2048 | 4.082 | -3.988 |
| auto_t3_controlled | 3072 | 3.532 | -3.836 |
| auto_t5_guarded | 1400 | -0.570 | 0.562 |
| auto_t5_guarded | 2048 | 4.029 | -3.885 |
| auto_t5_guarded | 3072 | 3.711 | -3.689 |

## Artifacts

- `replay_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_automation_20260210_014604.json`
- `replay_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_automation_20260210_014604.md`
- `replay_eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_automation_eval_20260210_014926.json`
- `replay_canary_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_automation_canary_20260210_014926.json`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_014604.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_015005.json`

## Decision

- Decision: `iterate`
- Failed checks: ['pre_gate_promote']
- Rationale: Weekly replay over RC has unresolved drift or gate failures.

