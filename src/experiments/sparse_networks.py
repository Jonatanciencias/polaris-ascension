"""
Sparse network simulation helpers.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


class SparseNetwork:
    """Simple sparse-network estimator for demo workflows."""

    def __init__(self, target_sparsity: float = 0.9, pruning_method: str = "magnitude") -> None:
        self.target_sparsity = float(target_sparsity)
        self.pruning_method = pruning_method

    def test_protein_structure_prediction(
        self,
        sequence_length: int = 1000,
        num_samples: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """
        Return rough memory/time estimates for several sparsity levels.
        """
        base_memory_mb = (sequence_length * sequence_length * 4) / (1024**2)
        base_time_ms = max(5.0, sequence_length / 12.0)

        levels = [0.0, 0.5, self.target_sparsity]
        results: Dict[str, Dict[str, float]] = {}
        for s in levels:
            keep = max(1e-3, 1.0 - s)
            cfg = f"sparsity_{int(round(s * 100))}"
            memory = base_memory_mb * keep
            # Assume sub-linear speedup due to sparse overheads.
            time_ms = base_time_ms * max(0.2, keep * 1.8)
            results[cfg] = {
                "memory_mb": float(memory),
                "memory_savings": float(s * 100.0),
                "inference_time_ms": float(time_ms),
                "samples": float(num_samples),
            }
        return results


def sparse_vs_dense_benchmark(
    model_size: Tuple[int, int] = (2048, 2048),
    sparsity_levels: Iterable[float] = (0.0, 0.9),
    num_iterations: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Estimate sparse-vs-dense throughput and memory footprint.
    """
    m, n = model_size
    dense_ops = float(m * n)
    dense_time_ms = max(2.0, dense_ops / 2.0e6)
    dense_memory_mb = dense_ops * 4.0 / (1024**2)
    dense_throughput = (num_iterations * 1000.0) / dense_time_ms

    results: Dict[str, Dict[str, float]] = {
        "dense": {
            "time_ms": float(dense_time_ms),
            "memory_mb": float(dense_memory_mb),
            "throughput": float(dense_throughput),
        }
    }

    for s in sparsity_levels:
        if s <= 0:
            continue
        keep = max(1e-3, 1.0 - float(s))
        sparse_time_ms = dense_time_ms * max(0.18, keep * 1.9)
        sparse_memory_mb = dense_memory_mb * keep
        key = f"sparse_{int(round(s * 100))}"
        results[key] = {
            "time_ms": float(sparse_time_ms),
            "memory_mb": float(sparse_memory_mb),
            "throughput": float((num_iterations * 1000.0) / sparse_time_ms),
            "speedup_vs_dense": float(dense_time_ms / sparse_time_ms),
            "compression_ratio": float(dense_memory_mb / max(sparse_memory_mb, 1e-9)),
        }

    return results
