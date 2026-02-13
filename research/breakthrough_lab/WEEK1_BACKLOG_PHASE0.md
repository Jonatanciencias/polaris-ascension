# Phase 0 - Week 1 Backlog (Atomic Tasks)

## Goal

Establish the minimum governance and execution baseline for the breakthrough lab.

## Definition of Done (Week 1)

- benchmark source-of-truth command list frozen
- promotion gate checklist adopted
- all tracks have experiment cards and results schema
- first two executable experiments prepared (T1 + T2)

## Progress Snapshot

- [x] `P0-01` baseline command set frozen (`BASELINE_RUNBOOK.md`)
- [x] `P0-02` promotion gate checklist created (`PROMOTION_GATE_CHECKLIST.md`)
- [x] `P0-03` results schema/template created (`results.schema.json`, `results.template.json`)
- [x] `P0-04` to `P0-09` experiment cards created for T1..T6
- [x] `P0-10` first executable T1 run completed (`t1_io_aware/results.json`, `t1_io_aware/report.md`)
- [x] `P0-11` first executable T2 dry-run completed (`t2_auto_scheduler/results.json`, `t2_auto_scheduler/report.md`)
- [x] `P0-12` week-1 review and kill/continue decisions (`ACTA_P0_12_WEEK1_REVIEW_2026-02-07.md`, `week1_review_decisions.json`)

## Atomic Tasks

1. `P0-01` Freeze baseline command set
- Owner: core-maintainer
- Estimate: 45 min
- Action: define exact command set for baseline run and output locations.
- DoD: command block committed in lab docs.

2. `P0-02` Create promotion gate checklist
- Owner: core-maintainer
- Estimate: 45 min
- Action: convert promotion policy into checklist with pass/fail fields.
- DoD: checklist document committed and referenced from each track card.

3. `P0-03` Standardize `results.json` schema
- Owner: data-maintainer
- Estimate: 30 min
- Action: validate schema structure and required fields for all tracks.
- DoD: schema + template parse successfully with local JSON loader.

4. `P0-04` Create T1 experiment card v1
- Owner: track-t1
- Estimate: 30 min
- Action: define hypothesis, variables, stop rule, and KPIs for IO-aware kernels.
- DoD: `t1_io_aware/experiment_card.md` is actionable.

5. `P0-05` Create T2 experiment card v1
- Owner: track-t2
- Estimate: 30 min
- Action: define hypothesis and bounded search space for auto-scheduler sidecar.
- DoD: `t2_auto_scheduler/experiment_card.md` is actionable.

6. `P0-06` Create T3 experiment card v1
- Owner: track-t3
- Estimate: 30 min
- Action: define online policy candidate and guardrails for selector adaptation.
- DoD: `t3_online_control/experiment_card.md` is actionable.

7. `P0-07` Create T4 experiment card v1
- Owner: track-t4
- Estimate: 30 min
- Action: define approximate mode error contract and fallback threshold.
- DoD: `t4_approximate_gemm/experiment_card.md` is actionable.

8. `P0-08` Create T5 experiment card v1
- Owner: track-t5
- Estimate: 30 min
- Action: define ABFT-lite validation hooks and overhead budget.
- DoD: `t5_reliability_abft/experiment_card.md` is actionable.

9. `P0-09` Create T6 experiment card v1
- Owner: track-t6
- Estimate: 30 min
- Action: define offline quantum-inspired search objective and baseline comparator.
- DoD: `t6_quantum_offline/experiment_card.md` is actionable.

10. `P0-10` Prepare first executable experiment for T1
- Owner: track-t1
- Estimate: 60 min
- Action: create `results.json` file and run stub benchmark using baseline harness.
- DoD: result file produced and schema-compliant.

11. `P0-11` Prepare first executable experiment for T2
- Owner: track-t2
- Estimate: 60 min
- Action: create `results.json` file and run bounded search dry-run.
- DoD: result file produced and schema-compliant.

12. `P0-12` Week-1 review and kill/continue decisions
- Owner: lead
- Estimate: 45 min
- Action: evaluate all cards and initial artifacts against promotion gate.
- DoD: each track tagged as `continue`, `refine`, or `stop`.

## Execution Order

1. `P0-01`, `P0-02`, `P0-03`
2. `P0-04` to `P0-09`
3. `P0-10`, `P0-11`
4. `P0-12`
