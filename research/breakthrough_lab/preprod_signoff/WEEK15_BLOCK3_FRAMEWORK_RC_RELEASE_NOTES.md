# Framework Release Candidate v0.15.0-rc1

## Scope

Controlled-production baseline for RX590 with extension/plugin handoff contracts.

## Evidence Baseline

- Week15 Block1 report: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week15_controlled_rollout/week15_block1_expanded_pilot_rerun_20260210_011756.json`
- Week15 Block2 report: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week15_controlled_rollout/week15_block2_plugin_pilot_rerun_20260210_012358.json`

## Included Capabilities

- Stable canonical validation gate (`tier=canonical`, `driver-smoke`).
- Controlled pilot profile for sizes `1400/2048/3072` with rollback SLA.
- Plugin onboarding template and extension contracts (Week14 Block6).
- Formal decision workflow (`promote|iterate|refine|stop` or `go|no-go`).

## Known Limits

- Expansion beyond controlled scope requires fresh canary evidence.
- Any T5 disable-event spike requires immediate rollback protocol.

