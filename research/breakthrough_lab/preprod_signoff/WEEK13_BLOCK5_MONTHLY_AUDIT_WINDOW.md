# Week 13 Block 5 - Monthly Audit Window

- Audit mode: monthly continuity audit
- Frequency: first business day of each month
- Duration target: 180 minutes
- Active policy: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/week13_controlled_rollout/policy_week13_block3_weekly_slo_v2.json`
- Weekly cadence baseline: `/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/research/breakthrough_lab/preprod_signoff/WEEK13_BLOCK5_WEEKLY_CADENCE.json`

## Mandatory Audit Checklist

- Validate canonical gate before audit execution.
- Replay weekly profile on latest policy version.
- Execute Clover/rusticl split and verify ratio floor.
- Compare monthly drift against previous monthly window.
- Confirm rollback drill command remains executable.
- Capture go/no-go result and open debts in live matrix.

## Escalation Rules

- Any correctness violation: immediate rollback and SEV1 escalation.
- Any `t5_disable_events_total > 0`: hold promotion and SEV2 escalation.
- Any split ratio below floor: hold promotion and SEV2 escalation.

