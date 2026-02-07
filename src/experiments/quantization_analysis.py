"""
Quantization safety analysis helpers.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _quantize_dequantize(values: np.ndarray, bits: int) -> np.ndarray:
    if bits >= 32:
        return values.astype(np.float32, copy=True)
    levels = (2 ** (bits - 1)) - 1
    max_abs = float(np.max(np.abs(values)) + 1e-8)
    scale = max_abs / levels
    q = np.clip(np.round(values / scale), -levels, levels)
    return (q * scale).astype(np.float32)


def _snr(reference: np.ndarray, candidate: np.ndarray) -> float:
    signal = np.mean(reference.astype(np.float64) ** 2)
    noise = np.mean((reference.astype(np.float64) - candidate.astype(np.float64)) ** 2) + 1e-12
    return float(10.0 * np.log10((signal + 1e-12) / noise))


class QuantizationAnalyzer:
    """Analyze ranking and safety impact of reduced precision."""

    def test_medical_safety(self, predictions: np.ndarray, bits: int = 8, task: str = "classification") -> Dict[str, float]:
        pred = predictions.astype(np.float32, copy=False)
        quant = _quantize_dequantize(pred, bits)
        base_label = np.argmax(pred, axis=-1)
        quant_label = np.argmax(quant, axis=-1)
        stability = float(np.mean(base_label == quant_label))
        snr_db = _snr(pred, quant)

        return {
            "task": task,
            "bits": float(bits),
            "decision_stability": stability,
            "snr_db": snr_db,
            "is_medically_safe": bool(stability >= 0.995 and snr_db >= 35.0),
        }

    def test_genomic_ranking_preservation(
        self,
        scores: np.ndarray,
        bits: int = 8,
        top_k: int = 1000,
    ) -> Dict[str, float]:
        base = scores.astype(np.float32, copy=False)
        quant = _quantize_dequantize(base, bits)

        base_rank = np.argsort(base)[::-1]
        quant_rank = np.argsort(quant)[::-1]

        k = min(int(top_k), base_rank.size)
        base_top = set(base_rank[:k].tolist())
        quant_top = set(quant_rank[:k].tolist())
        overlap = len(base_top.intersection(quant_top)) / max(k, 1)

        # Spearman approximation through ranked indices.
        inv_base = np.empty_like(base_rank)
        inv_quant = np.empty_like(quant_rank)
        inv_base[base_rank] = np.arange(base_rank.size)
        inv_quant[quant_rank] = np.arange(quant_rank.size)
        rho = np.corrcoef(inv_base.astype(np.float64), inv_quant.astype(np.float64))[0, 1]

        rank_shift = np.abs(inv_base - inv_quant)
        return {
            "bits": float(bits),
            "spearman_correlation": float(rho),
            "top_k_overlap": float(overlap),
            "mean_rank_shift": float(np.mean(rank_shift)),
            "max_rank_shift": float(np.max(rank_shift)),
            "is_safe_for_genomics": bool(rho >= 0.995 and overlap >= 0.98),
        }

    def test_drug_discovery_sensitivity(self, scores: np.ndarray, bits: int = 8) -> Dict[str, float]:
        base = scores.astype(np.float32, copy=False)
        quant = _quantize_dequantize(base, bits)
        err = np.abs(base - quant)

        top_n = min(1000, base.size)
        base_top = set(np.argsort(base)[:top_n].tolist())
        quant_top = set(np.argsort(quant)[:top_n].tolist())
        overlap = len(base_top.intersection(quant_top)) / max(top_n, 1)

        speedup = 1.0
        if bits == 16:
            speedup = 1.5
        elif bits <= 8:
            speedup = 2.5

        return {
            "bits": float(bits),
            "mean_error_kcal": float(np.mean(err)),
            "top_1000_overlap": float(overlap),
            "speedup_factor": float(speedup),
            "compounds_per_day_gain": float(10000 * (speedup - 1.0)),
            "is_safe_for_screening": bool(np.mean(err) < 1.0 and overlap > 0.95),
        }


def sensitivity_analysis(values: np.ndarray, bits: int = 8) -> Dict[str, float]:
    """Quick wrapper for drug-discovery-style sensitivity checks."""
    return QuantizationAnalyzer().test_drug_discovery_sensitivity(values, bits=bits)
