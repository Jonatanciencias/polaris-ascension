#!/usr/bin/env bash
set -euo pipefail

ROUNDS="${1:-3}"
PYTEST_BIN="${PYTEST_BIN:-./venv/bin/pytest}"

TARGETS=(
  "tests/test_opencl_gemm.py::TestGEMMCorrectness::test_square_matrices"
  "tests/test_opencl_gemm.py::TestGEMMEdgeCases::test_non_tile_aligned"
  "tests/test_opencl_gemm.py::TestGEMMKernelVariants::test_kernel_consistency"
)

echo "[stability] Running critical numeric tests for ${ROUNDS} rounds"
for round in $(seq 1 "${ROUNDS}"); do
  echo "[stability] Round ${round}/${ROUNDS}"
  "${PYTEST_BIN}" -q "${TARGETS[@]}"
done

echo "[stability] PASSED: all rounds completed without failures"
