#!/usr/bin/env python3
"""
Quick functional validation for current OpenCL GEMM API.

Validates:
  1) correctness on 128x128
  2) correctness on 512x512
  3) alpha/beta semantics

Persists JSON+Markdown reports into results/benchmark_reports.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

# Workaround for pyopencl cache warning in specific runtime versions.
os.environ.setdefault("PYOPENCL_NO_CACHE", "1")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarking.reporting import markdown_table, report_paths, save_json_report, save_markdown_report
from src.opencl import CLContext, gemm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class QuickValidator:
    """Quick functional validator for OpenCL GEMM compatibility API."""

    def __init__(self, seed: int = 42) -> None:
        logger.info("Initializing Quick Validator...")
        self.seed = seed
        self.context = CLContext()
        self.results: dict[str, object] = {
            "metadata": {
                "timestamp": time.time(),
                "seed": seed,
                "validator": "scripts/quick_validation.py",
            },
            "tests": {},
            "summary": {},
        }

    def _run_case(self, name: str, size: int, seed: int) -> bool:
        logger.info("%s", "=" * 80)
        logger.info("TEST: %s (%dx%d)", name, size, size)
        logger.info("%s", "=" * 80)

        try:
            np.random.seed(seed)
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            # Warmup to avoid counting first-launch overhead in measured GFLOPS.
            _ = gemm(self.context, A, B)

            start = time.perf_counter()
            C_gpu = gemm(self.context, A, B)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            C_ref = A @ B
            error_rel = float(np.linalg.norm(C_gpu - C_ref) / np.linalg.norm(C_ref))
            gflops = float((2 * size**3) / (elapsed_ms / 1000.0) / 1e9)
            passed = error_rel < 1e-4

            logger.info("GPU time: %.3f ms", elapsed_ms)
            logger.info("GFLOPS: %.1f", gflops)
            logger.info("Error (relative): %.2e", error_rel)
            logger.info("Status: %s", "PASS" if passed else "FAIL")

            self.results["tests"][name] = {
                "size": size,
                "gpu_time_ms": elapsed_ms,
                "gflops": gflops,
                "error_rel": error_rel,
                "passed": passed,
            }
            return passed
        except Exception as exc:
            logger.exception("Test %s crashed: %s", name, exc)
            self.results["tests"][name] = {"passed": False, "error": str(exc)}
            return False

    def test_alpha_beta(self, size: int = 256, seed: int = 44) -> bool:
        logger.info("%s", "=" * 80)
        logger.info("TEST: alpha/beta semantics (%dx%d)", size, size)
        logger.info("%s", "=" * 80)

        np.random.seed(seed)
        A = np.random.randn(size, size).astype(np.float32)
        B = np.random.randn(size, size).astype(np.float32)
        C = np.random.randn(size, size).astype(np.float32)

        cases = [
            (1.0, 0.0, "alpha=1.0,beta=0.0"),
            (2.5, 0.0, "alpha=2.5,beta=0.0"),
            (1.0, 1.0, "alpha=1.0,beta=1.0"),
            (2.5, 0.5, "alpha=2.5,beta=0.5"),
        ]

        sub = {}
        all_passed = True
        for alpha, beta, label in cases:
            C_gpu = gemm(self.context, A, B, C=C.copy(), alpha=alpha, beta=beta)
            C_ref = alpha * (A @ B) + beta * C
            error_rel = float(np.linalg.norm(C_gpu - C_ref) / np.linalg.norm(C_ref))
            passed = error_rel < 1e-4
            sub[label] = {"error_rel": error_rel, "passed": passed}
            all_passed = all_passed and passed
            logger.info("%s -> %s (error %.2e)", label, "PASS" if passed else "FAIL", error_rel)

        self.results["tests"]["alpha_beta_params"] = {
            "sub_tests": sub,
            "passed": all_passed,
        }
        return all_passed

    def run_all(self) -> bool:
        outcomes = [
            self._run_case("small_matrix_128", 128, self.seed),
            self._run_case("medium_matrix_512", 512, self.seed + 1),
            self.test_alpha_beta(),
        ]

        total = len(outcomes)
        passed = int(sum(outcomes))
        all_passed = all(outcomes)

        self.results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "all_passed": all_passed,
        }

        logger.info("%s", "=" * 80)
        logger.info("SUMMARY: %d/%d passed", passed, total)
        logger.info("STATUS: %s", "ALL PASS" if all_passed else "FAILED")
        logger.info("%s", "=" * 80)
        return all_passed


def _build_markdown(results: dict[str, object]) -> str:
    tests = results.get("tests", {})
    rows = []
    for key in ["small_matrix_128", "medium_matrix_512"]:
        test = tests.get(key, {}) if isinstance(tests, dict) else {}
        rows.append(
            (
                key,
                test.get("size", "-"),
                f"{test.get('gflops', 0.0):.1f}" if isinstance(test.get("gflops"), (int, float)) else "-",
                f"{test.get('error_rel', 0.0):.2e}" if isinstance(test.get("error_rel"), (int, float)) else "-",
                "PASS" if test.get("passed") else "FAIL",
            )
        )

    alpha = tests.get("alpha_beta_params", {}) if isinstance(tests, dict) else {}
    rows.append(
        (
            "alpha_beta_params",
            "256",
            "-",
            "-",
            "PASS" if alpha.get("passed") else "FAIL",
        )
    )

    summary = results.get("summary", {}) if isinstance(results.get("summary"), dict) else {}
    return (
        "# Quick Validation Report\n\n"
        + markdown_table(
            headers=["Test", "Size", "GFLOPS", "Rel Error", "Status"],
            rows=rows,
        )
        + "\n\n"
        + markdown_table(
            headers=["Metric", "Value"],
            rows=[
                ("Total tests", summary.get("total_tests", "-")),
                ("Passed", summary.get("passed", "-")),
                ("Failed", summary.get("failed", "-")),
                ("All passed", summary.get("all_passed", "-")),
            ],
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick OpenCL GEMM functional validation")
    parser.add_argument(
        "--output-dir",
        default="results/benchmark_reports",
        help="Directory where report artifacts are written",
    )
    args = parser.parse_args()

    validator = QuickValidator(seed=42)
    success = validator.run_all()

    json_path, md_path = report_paths(prefix="quick_validation", output_dir=args.output_dir)
    save_json_report(json_path, validator.results)
    save_markdown_report(md_path, _build_markdown(validator.results))

    logger.info("JSON report: %s", json_path)
    logger.info("MD report:   %s", md_path)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
