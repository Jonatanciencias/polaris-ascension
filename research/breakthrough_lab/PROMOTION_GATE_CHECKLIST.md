# Promotion Gate Checklist

Use this checklist before promoting any lab result into production.

## Experiment Metadata

- [ ] experiment card exists and is complete
- [ ] `results.json` exists and validates against schema
- [ ] baseline comparison uses canonical runbook

## Performance Gate

- [ ] sustained gain >= +10% vs baseline on at least one target size
- [ ] no critical regression on other target sizes
- [ ] improvement is reproducible across repeated sessions

## Correctness Gate

- [ ] correctness checks passed
- [ ] max error within agreed threshold
- [ ] no unstable numerical behavior observed

## Stability Gate

- [ ] coefficient of variation within acceptable range
- [ ] no crash/hang during repeated runs
- [ ] behavior stable under warm and hot GPU conditions

## Operational Gate

- [ ] compile/startup overhead acceptable
- [ ] complexity justified by measured gain
- [ ] rollback path documented

## Integration Gate

- [ ] tests green after integration
- [ ] docs updated (`README`, `docs/`, benchmark protocol)
- [ ] production owner approved merge

## Decision

- [ ] promote
- [ ] iterate
- [ ] drop
