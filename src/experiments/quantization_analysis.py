"""
Quantization Sensitivity Analysis

Mathematical framework for understanding when and how quantization 
preserves model quality for critical applications.

THEORETICAL FOUNDATION:

Quantization Error Bound (Uniform Quantization):
||Q(x) - x||_∞ ≤ Δ/2
where Δ = (x_max - x_min) / (2^bits - 1)

Signal-to-Quantization-Noise Ratio:
SQNR ≈ 6.02 · bits + 1.76 dB

For 8-bit: SQNR ≈ 50 dB (excellent for most applications)
For 4-bit: SQNR ≈ 26 dB (marginal for critical tasks)

CRITICAL APPLICATIONS ANALYSIS:

1. MEDICAL DIAGNOSIS:
   Question: Can we diagnose cancer with quantized networks?
   
   Risk analysis:
   - False negative (miss cancer): Catastrophic
   - False positive (healthy → suspicious): Costly but acceptable
   
   Quantization strategy:
   - Detection layers: FP16 (high sensitivity)
   - Classification layers: INT8 (sufficient precision)
   - Never quantize final output layer
   
2. DRUG DISCOVERY:
   Question: Does quantization preserve binding affinity ranking?
   
   Key insight: Ranking matters, not absolute values
   - Spearman correlation > 0.95: Safe
   - Top-1000 overlap > 90%: Acceptable
   - INT8 typically satisfies both
   
3. GENOMIC ANALYSIS:
   Question: Will quantization miss rare variants?
   
   Analysis:
   - Common variants (MAF > 1%): INT8 robust
   - Rare variants (MAF < 0.1%): Need FP16
   - Mixed precision: Use INT8 for bulk, FP16 for rare
   
This module provides tools to make these decisions rigorously.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from scipy.stats import spearmanr, pearsonr


@dataclass
class QuantizationConfig:
    """Configuration for quantization experiment"""
    bits: int  # 4, 8, 16, 32
    symmetric: bool  # Symmetric vs asymmetric
    per_channel: bool  # Per-channel vs per-tensor
    calibration_method: str  # 'minmax', 'entropy', 'percentile'


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis"""
    config: QuantizationConfig
    accuracy_drop: float
    inference_speedup: float
    memory_reduction: float
    snr_db: float
    worst_case_error: float
    safe_for_medical: bool
    safe_for_genomic: bool
    safe_for_drug_discovery: bool


class QuantizationAnalyzer:
    """
    Rigorous quantization sensitivity analysis for critical AI applications.
    
    This class answers fundamental questions:
    1. How many bits do we REALLY need?
    2. Which layers are sensitive to quantization?
    3. What's the mathematical error bound?
    4. Is it safe for life-critical applications?
    
    Methodology:
    - Layer-wise sensitivity profiling
    - Statistical error analysis
    - Task-specific validation metrics
    - Mathematical guarantees when possible
    """
    
    def __init__(self):
        self.sensitivity_map: Dict[str, float] = {}
        self.results: List[SensitivityResult] = []
        
    def analyze_quantization_error(
        self,
        values: np.ndarray,
        bits: int,
        symmetric: bool = True
    ) -> Dict[str, float]:
        """
        Mathematical analysis of quantization error.
        
        Computes:
        - Theoretical error bound
        - Actual error distribution
        - SNR (signal-to-noise ratio)
        - Worst-case error
        
        Args:
            values: Original values
            bits: Number of quantization bits
            symmetric: Symmetric vs asymmetric quantization
            
        Returns:
            Error analysis
        """
        # Quantize
        quantized, scale, zero_point = self._quantize(values, bits, symmetric)
        dequantized = self._dequantize(quantized, scale, zero_point, bits)
        
        # Error analysis
        error = dequantized - values
        
        # Theoretical bound
        if symmetric:
            abs_max = np.abs(values).max()
            delta = abs_max / (2**(bits-1) - 1)
        else:
            val_range = values.max() - values.min()
            delta = val_range / (2**bits - 1)
        
        theoretical_bound = delta / 2
        
        # Actual statistics
        analysis = {
            'theoretical_bound': float(theoretical_bound),
            'actual_max_error': float(np.abs(error).max()),
            'actual_mean_error': float(np.abs(error).mean()),
            'actual_std_error': float(np.std(error)),
            'snr_db': self._calculate_snr(values, error),
            'theoretical_snr_db': 6.02 * bits + 1.76,  # Formula from information theory
            'percentile_95_error': float(np.percentile(np.abs(error), 95)),
            'percentile_99_error': float(np.percentile(np.abs(error), 99))
        }
        
        return analysis
    
    def _quantize(
        self,
        values: np.ndarray,
        bits: int,
        symmetric: bool
    ) -> Tuple[np.ndarray, float, int]:
        """
        Quantize floating point values to integers.
        
        Symmetric quantization (recommended for weights):
        Q(x) = clip(round(x / scale), -2^(b-1), 2^(b-1)-1)
        scale = max(|x|) / (2^(b-1) - 1)
        
        Asymmetric quantization (recommended for activations):
        Q(x) = clip(round((x - zero_point) / scale), 0, 2^b-1)
        scale = (max(x) - min(x)) / (2^b - 1)
        zero_point = min(x)
        
        Args:
            values: Input values
            bits: Number of bits
            symmetric: Symmetric vs asymmetric
            
        Returns:
            Quantized values, scale, zero_point
        """
        if symmetric:
            abs_max = np.abs(values).max()
            if abs_max == 0:
                return np.zeros_like(values, dtype=np.int32), 1.0, 0
            
            scale = abs_max / (2**(bits-1) - 1)
            quantized = np.round(values / scale)
            quantized = np.clip(quantized, -(2**(bits-1)), 2**(bits-1) - 1)
            zero_point = 0
        else:
            val_min, val_max = values.min(), values.max()
            if val_min == val_max:
                return np.zeros_like(values, dtype=np.int32), 1.0, 0
            
            scale = (val_max - val_min) / (2**bits - 1)
            zero_point = val_min
            quantized = np.round((values - zero_point) / scale)
            quantized = np.clip(quantized, 0, 2**bits - 1)
        
        return quantized.astype(np.int32), scale, zero_point
    
    def _dequantize(
        self,
        quantized: np.ndarray,
        scale: float,
        zero_point: float,
        bits: int
    ) -> np.ndarray:
        """Dequantize integer values back to floating point"""
        if zero_point == 0:  # Symmetric
            return quantized.astype(np.float32) * scale
        else:  # Asymmetric
            return quantized.astype(np.float32) * scale + zero_point
    
    def _calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio in dB"""
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return np.inf
        
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)
    
    def test_medical_safety(
        self,
        predictions: np.ndarray,
        bits: int,
        task: str = 'classification'
    ) -> Dict[str, any]:
        """
        Test if quantization is safe for medical applications.
        
        Safety criteria:
        1. SNR > 40 dB: Diagnostic quality
        2. Top-1 accuracy drop < 1%
        3. No systematic bias (false negatives)
        4. Stable across edge cases
        
        Args:
            predictions: Model predictions (probabilities or logits)
            bits: Quantization bits
            task: 'classification' or 'regression'
            
        Returns:
            Safety analysis
        """
        # Quantize predictions
        error_analysis = self.analyze_quantization_error(predictions, bits)
        
        # Task-specific tests
        if task == 'classification':
            # For classification: Does quantization change predicted class?
            quantized, scale, zp = self._quantize(predictions, bits, symmetric=False)
            dequantized = self._dequantize(quantized, scale, zp, bits)
            
            # Assume predictions are class probabilities [N, C]
            if predictions.ndim == 2:
                original_classes = np.argmax(predictions, axis=1)
                quantized_classes = np.argmax(dequantized, axis=1)
                
                # Critical: How often does quantization change the decision?
                decision_stability = np.mean(original_classes == quantized_classes)
                
                # For medical: We need > 99.5% stability
                is_safe = decision_stability > 0.995 and error_analysis['snr_db'] > 40
            else:
                decision_stability = 1.0
                is_safe = error_analysis['snr_db'] > 40
        
        elif task == 'regression':
            # For regression: Relative error matters
            quantized, scale, zp = self._quantize(predictions, bits, symmetric=True)
            dequantized = self._dequantize(quantized, scale, zp, bits)
            
            relative_error = np.abs((dequantized - predictions) / (predictions + 1e-8))
            mean_relative_error = np.mean(relative_error)
            
            # For medical regression (e.g., tumor size): < 5% error
            is_safe = mean_relative_error < 0.05 and error_analysis['snr_db'] > 40
            decision_stability = 1.0 - mean_relative_error
        
        else:
            raise ValueError(f"Unknown task: {task}")
        
        return {
            **error_analysis,
            'decision_stability': float(decision_stability),
            'is_medically_safe': is_safe,
            'recommendation': (
                "Safe for medical use" if is_safe
                else "Not recommended for medical diagnosis - use higher precision"
            ),
            'required_bits': self._estimate_required_bits(error_analysis['snr_db'])
        }
    
    def test_genomic_ranking_preservation(
        self,
        scores: np.ndarray,
        bits: int,
        top_k: int = 1000
    ) -> Dict[str, float]:
        """
        Test if quantization preserves genomic variant rankings.
        
        Critical for:
        - Variant calling (finding disease mutations)
        - GWAS (genome-wide association studies)
        - Population genetics
        
        Key insight: We care about RANKING, not absolute scores
        
        Args:
            scores: Quality scores or alignment scores
            bits: Quantization bits
            top_k: Number of top variants to consider
            
        Returns:
            Ranking preservation metrics
        """
        # Quantize scores
        quantized, scale, zp = self._quantize(scores, bits, symmetric=False)
        dequantized = self._dequantize(quantized, scale, zp, bits)
        
        # Ranking metrics
        original_ranking = np.argsort(scores)[::-1]  # Descending
        quantized_ranking = np.argsort(dequantized)[::-1]
        
        # Spearman correlation (rank correlation)
        spearman_corr, _ = spearmanr(scores, dequantized)
        
        # Top-k overlap
        top_k = min(top_k, len(scores))
        original_top_k = set(original_ranking[:top_k])
        quantized_top_k = set(quantized_ranking[:top_k])
        top_k_overlap = len(original_top_k & quantized_top_k) / top_k
        
        # Rank distance for top variants
        rank_distances = []
        for i in range(top_k):
            orig_idx = original_ranking[i]
            quant_position = np.where(quantized_ranking == orig_idx)[0][0]
            rank_distances.append(abs(i - quant_position))
        
        mean_rank_shift = np.mean(rank_distances)
        max_rank_shift = np.max(rank_distances)
        
        # Safety criteria for genomics
        is_safe = (
            spearman_corr > 0.99 and
            top_k_overlap > 0.95 and
            mean_rank_shift < 10
        )
        
        return {
            'spearman_correlation': float(spearman_corr),
            'top_k_overlap': float(top_k_overlap),
            'mean_rank_shift': float(mean_rank_shift),
            'max_rank_shift': float(max_rank_shift),
            'is_safe_for_genomics': is_safe,
            'recommendation': (
                f"Safe for variant calling (correlation={spearman_corr:.4f})" if is_safe
                else f"Warning: Ranking correlation={spearman_corr:.4f}, use {self._estimate_required_bits(spearman_corr * 50)} bits"
            )
        }
    
    def test_drug_discovery_sensitivity(
        self,
        binding_affinities: np.ndarray,
        bits: int,
        tolerance_kcal: float = 1.0
    ) -> Dict[str, float]:
        """
        Test quantization for molecular docking / drug discovery.
        
        Context:
        - Binding affinity: -10 to 0 kcal/mol (lower = better)
        - Clinical significance: ±1 kcal/mol
        - Goal: Screen 10^6 compounds, select top 1000
        
        Requirements:
        - Absolute error < 1 kcal/mol
        - Top-1000 overlap > 90%
        - No systematic bias (missing good candidates)
        
        Args:
            binding_affinities: Predicted binding energies (kcal/mol)
            bits: Quantization bits
            tolerance_kcal: Acceptable error (default: 1.0 kcal/mol)
            
        Returns:
            Drug discovery suitability analysis
        """
        # Quantize
        error_analysis = self.analyze_quantization_error(binding_affinities, bits)
        
        quantized, scale, zp = self._quantize(binding_affinities, bits, symmetric=False)
        dequantized = self._dequantize(quantized, scale, zp, bits)
        
        # Absolute error (kcal/mol)
        absolute_errors = np.abs(dequantized - binding_affinities)
        mean_error = np.mean(absolute_errors)
        max_error = np.max(absolute_errors)
        
        # Ranking for top candidates
        top_k = min(1000, len(binding_affinities))
        original_ranking = np.argsort(binding_affinities)  # Ascending (lower is better)
        quantized_ranking = np.argsort(dequantized)
        
        original_top_k = set(original_ranking[:top_k])
        quantized_top_k = set(quantized_ranking[:top_k])
        top_k_overlap = len(original_top_k & quantized_top_k) / top_k
        
        # Safety criteria
        is_safe = (
            mean_error < tolerance_kcal and
            max_error < 2 * tolerance_kcal and
            top_k_overlap > 0.90
        )
        
        # Economic impact
        # If we can screen 2x faster with quantization, what's the value?
        baseline_compounds_per_day = 10000  # Example
        speedup_factor = 32 / bits  # Rough estimate
        quantized_compounds_per_day = baseline_compounds_per_day * speedup_factor
        
        return {
            **error_analysis,
            'mean_error_kcal': float(mean_error),
            'max_error_kcal': float(max_error),
            'top_1000_overlap': float(top_k_overlap),
            'is_safe_for_screening': is_safe,
            'compounds_per_day_gain': float(quantized_compounds_per_day - baseline_compounds_per_day),
            'speedup_factor': float(speedup_factor),
            'recommendation': (
                f"Safe for high-throughput screening (error={mean_error:.2f} kcal/mol)" if is_safe
                else f"Error too large ({mean_error:.2f} kcal/mol), use {self._estimate_required_bits(error_analysis['snr_db'])} bits"
            )
        }
    
    def _estimate_required_bits(self, snr_db: float) -> int:
        """
        Estimate required bits from desired SNR.
        
        From information theory:
        SNR_dB ≈ 6.02 * bits + 1.76
        
        Solving for bits:
        bits ≈ (SNR_dB - 1.76) / 6.02
        """
        bits = (snr_db - 1.76) / 6.02
        return int(np.ceil(bits))
    
    def layer_wise_sensitivity_analysis(
        self,
        layer_outputs: Dict[str, np.ndarray],
        bits: int
    ) -> Dict[str, Dict]:
        """
        Analyze quantization sensitivity for each layer.
        
        This identifies which layers are "fragile" and need higher precision.
        
        Strategy:
        - Quantize each layer independently
        - Measure impact on final output
        - Rank layers by sensitivity
        - Use mixed precision accordingly
        
        Args:
            layer_outputs: Dictionary of layer name -> output tensor
            bits: Target quantization bits
            
        Returns:
            Per-layer sensitivity analysis
        """
        results = {}
        
        for layer_name, outputs in layer_outputs.items():
            analysis = self.analyze_quantization_error(outputs, bits)
            
            # Classify sensitivity
            snr = analysis['snr_db']
            if snr > 50:
                sensitivity = 'low'
                recommended_bits = 8
            elif snr > 40:
                sensitivity = 'medium'
                recommended_bits = 16
            else:
                sensitivity = 'high'
                recommended_bits = 32
            
            results[layer_name] = {
                **analysis,
                'sensitivity': sensitivity,
                'recommended_bits': recommended_bits,
                'can_quantize_to_int8': snr > 45,
                'must_keep_fp16': snr < 40
            }
        
        return results


def sensitivity_analysis(
    data: np.ndarray,
    application: str,
    bits_range: List[int] = [4, 8, 16, 32]
) -> Dict:
    """
    Run comprehensive sensitivity analysis for an application.
    
    Args:
        data: Sample data (predictions, scores, etc.)
        application: 'medical', 'genomic', or 'drug_discovery'
        bits_range: List of bit widths to test
        
    Returns:
        Complete sensitivity analysis
    """
    analyzer = QuantizationAnalyzer()
    results = {}
    
    for bits in bits_range:
        if application == 'medical':
            result = analyzer.test_medical_safety(data, bits)
        elif application == 'genomic':
            result = analyzer.test_genomic_ranking_preservation(data, bits)
        elif application == 'drug_discovery':
            result = analyzer.test_drug_discovery_sensitivity(data, bits)
        else:
            raise ValueError(f"Unknown application: {application}")
        
        results[f'{bits}_bit'] = result
    
    # Add summary
    results['summary'] = {
        'application': application,
        'data_shape': data.shape,
        'data_range': (float(data.min()), float(data.max())),
        'recommended_bits': max([
            bits for bits in bits_range
            if results[f'{bits}_bit'].get('is_medically_safe') or
               results[f'{bits}_bit'].get('is_safe_for_genomics') or
               results[f'{bits}_bit'].get('is_safe_for_screening')
        ], default=32)
    }
    
    return results


if __name__ == "__main__":
    print("🔬 QUANTIZATION SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Test 1: Medical imaging
    print("\n1️⃣  MEDICAL IMAGING: Tumor detection probabilities")
    medical_predictions = np.random.rand(1000, 5).astype(np.float32)  # 1000 images, 5 classes
    medical_predictions /= medical_predictions.sum(axis=1, keepdims=True)  # Normalize
    
    analyzer = QuantizationAnalyzer()
    for bits in [32, 16, 8]:
        result = analyzer.test_medical_safety(medical_predictions, bits)
        print(f"\n{bits}-bit quantization:")
        print(f"  SNR: {result['snr_db']:.2f} dB")
        print(f"  Decision stability: {result['decision_stability']*100:.2f}%")
        print(f"  Safe: {result['is_medically_safe']}")
        print(f"  Recommendation: {result['recommendation']}")
    
    # Test 2: Genomic variant calling
    print("\n2️⃣  GENOMIC ANALYSIS: Variant quality scores")
    genomic_scores = np.random.exponential(20, 10000).astype(np.float32)  # Quality scores
    
    for bits in [32, 16, 8]:
        result = analyzer.test_genomic_ranking_preservation(genomic_scores, bits)
        print(f"\n{bits}-bit quantization:")
        print(f"  Spearman correlation: {result['spearman_correlation']:.4f}")
        print(f"  Top-1000 overlap: {result['top_k_overlap']*100:.1f}%")
        print(f"  Mean rank shift: {result['mean_rank_shift']:.1f}")
        print(f"  Safe: {result['is_safe_for_genomics']}")
    
    # Test 3: Drug discovery
    print("\n3️⃣  DRUG DISCOVERY: Binding affinities")
    binding_affinities = np.random.randn(100000).astype(np.float32) * 3 - 7  # kcal/mol
    
    for bits in [32, 16, 8]:
        result = analyzer.test_drug_discovery_sensitivity(binding_affinities, bits)
        print(f"\n{bits}-bit quantization:")
        print(f"  Mean error: {result['mean_error_kcal']:.3f} kcal/mol")
        print(f"  Top-1000 overlap: {result['top_1000_overlap']*100:.1f}%")
        print(f"  Speedup: {result['speedup_factor']:.1f}x")
        print(f"  Safe: {result['is_safe_for_screening']}")
