# Week 16 Block 2 - Weekly RC Replay + Drift

- Date: 2026-02-10T01:55:04.129492+00:00
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
| pre_gate_promote | True |
| post_gate_promote | True |

## Drift Rows

| Kernel | Size | Thr drift % | P95 drift % |
| --- | ---: | ---: | ---: |
| auto_t3_controlled | 1400 | 0.164 | -0.037 |
| auto_t3_controlled | 2048 | 4.092 | -4.015 |
| auto_t3_controlled | 3072 | 3.443 | -3.806 |
| auto_t5_guarded | 1400 | -1.404 | 0.622 |
| auto_t5_guarded | 2048 | 4.451 | -3.962 |
| auto_t5_guarded | 3072 | 3.656 | -3.648 |

## Artifacts

- `replay_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_rerun_automation_20260210_015103.json`
- `replay_md`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_rerun_automation_20260210_015103.md`
- `replay_eval_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_rerun_automation_eval_20260210_015424.json`
- `replay_canary_json`: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_rerun_automation_canary_20260210_015424.json`
- `pre_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_015103.json`
- `post_gate_json`: `research/breakthrough_lab/week8_validation_discipline/validation_suite_canonical_20260210_015504.json`

## Decision

- Decision: `promote`
- Failed checks: []
- Rationale: Weekly replay over RC is stable, policy-compliant, and within drift thresholds.

