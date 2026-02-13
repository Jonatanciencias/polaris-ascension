# Stable Release Proposal v0.15.0

## Summary

- Source RC: `v0.15.0-rc1`
- Proposed stable tag: `v0.15.0`

## Evidence Chain

- Week16 Block1 integration pilot: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block1_dependent_integration_20260210_014453.json`
- Week16 Block2 weekly replay + drift: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week16_controlled_rollout/week16_block2_weekly_rc_replay_rerun_20260210_015504.json`

## Stable Scope

- Deterministic controlled profile for `1400/2048/3072`.
- Mandatory canonical gate before promotion and scope expansion.
- Guardrails: correctness, T3 fallback, T5 overhead/disable events.

## Deferred

- Any new platform scope or larger sizes beyond `3072` need fresh evidence.
- Plugin API major changes remain out-of-scope for `v0.15.0`.

