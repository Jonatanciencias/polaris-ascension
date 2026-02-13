#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dry-run}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
ROLLBACK_NOTE="research/breakthrough_lab/platform_compatibility/week9_block5_rollback_${STAMP}.md"
RUNTIME_ENV_FILE="results/runtime_states/week9_block5_runtime_env.sh"

if [[ "${MODE}" != "dry-run" && "${MODE}" != "apply" ]]; then
  echo "Usage: $0 [dry-run|apply]" >&2
  exit 2
fi

cat > "${RUNTIME_ENV_FILE}" <<'EOF'
#!/usr/bin/env bash
# Week9 Block5 rollback runtime profile (generated)
export RX580_OPENCL_PLATFORM="Clover"
unset RUSTICL_ENABLE || true
export RX580_OPENCL_DEVICE="auto"
export RX580_T5_POLICY_PATH="research/breakthrough_lab/t5_reliability_abft/policy_hardening_week9_block2.json"
EOF
chmod +x "${RUNTIME_ENV_FILE}"

{
  echo "# Week9 Block5 Rusticl Rollback"
  echo
  echo "- Timestamp UTC: ${STAMP}"
  echo "- Mode: ${MODE}"
  echo "- Action: enforce Clover as runtime platform and disable rusticl env bootstrap."
  echo "- Runtime env file: \`${RUNTIME_ENV_FILE}\`"
  echo "- Safe T5 policy: \`research/breakthrough_lab/t5_reliability_abft/policy_hardening_week9_block2.json\`"
  echo
  echo "## Apply Steps"
  echo
  echo "1. \`source ${RUNTIME_ENV_FILE}\`"
  echo "2. run canonical gate:"
  echo "   - \`./venv/bin/python scripts/run_validation_suite.py --tier canonical --driver-smoke --report-dir research/breakthrough_lab/week8_validation_discipline\`"
} > "${ROLLBACK_NOTE}"

echo "Generated rollback runtime profile: ${RUNTIME_ENV_FILE}"
echo "Generated rollback note: ${ROLLBACK_NOTE}"

if [[ "${MODE}" == "apply" ]]; then
  # shellcheck source=/dev/null
  source "${RUNTIME_ENV_FILE}"
  ./venv/bin/python scripts/run_validation_suite.py \
    --tier canonical \
    --driver-smoke \
    --report-dir research/breakthrough_lab/week8_validation_discipline
  echo "Rollback applied and canonical gate executed."
else
  echo "Dry-run only. No process-wide env was modified."
fi
