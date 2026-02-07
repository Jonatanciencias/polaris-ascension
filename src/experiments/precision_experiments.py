"""
Precision analysis utilities.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _snr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    signal = np.mean(np.square(reference))
    noise = np.mean(np.square(reference - candidate)) + 1e-12
    return float(10.0 * np.log10((signal + 1e-12) / noise))


class PrecisionExperiment:
    """Evaluate precision tradeoffs for safety-critical workloads."""

    def test_medical_imaging_precision(self, data: np.ndarray) -> Dict[str, Dict[str, object]]:
        """
        Measure FP16 and INT8 distortion against FP32 reference.
        """
        ref = data.astype(np.float32, copy=False)
        fp16 = ref.astype(np.float16).astype(np.float32)

        # Symmetric int8 quantization/dequantization.
        max_abs = float(np.max(np.abs(ref)) + 1e-8)
        scale = max_abs / 127.0
        int8_q = np.clip(np.round(ref / scale), -127, 127).astype(np.int8)
        int8_dq = (int8_q.astype(np.float32) * scale).astype(np.float32)

        snr_fp16 = _snr_db(ref, fp16)
        snr_int8 = _snr_db(ref, int8_dq)

        return {
            "fp16": {
                "snr_db": snr_fp16,
                "diagnostic_quality": snr_fp16 >= 55.0,
                "screening_quality": snr_fp16 >= 40.0,
                "recommendation": "Use FP16 for near-lossless speedup.",
            },
            "int8": {
                "snr_db": snr_int8,
                "diagnostic_quality": snr_int8 >= 55.0,
                "screening_quality": snr_int8 >= 40.0,
                "recommendation": "Use INT8 for high-throughput screening pipelines.",
            },
        }


def compare_precisions(data: np.ndarray) -> Dict[str, Dict[str, object]]:
    """Helper wrapper for quick experiment calls."""
    return PrecisionExperiment().test_medical_imaging_precision(data)
