#!/usr/bin/env python3
"""
🔬 UNCERTAINTY QUANTIFICATION SYSTEM
====================================

Sistema avanzado para cuantificar y gestionar la incertidumbre en
optimización bayesiana, evaluando la confiabilidad de predicciones
y resultados de optimización.

Características principales:
- Análisis de incertidumbre en predicciones del AI Kernel Predictor
- Evaluación de confianza en resultados de optimización
- Métodos de muestreo para cuantificación de incertidumbre
- Análisis de sensibilidad y robustez
- Estimación de riesgo y factores de seguridad
- Validación cruzada de resultados

Author: AI Assistant
Date: 2026-01-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from scipy import stats
from scipy.stats import norm, t
import logging
import warnings
warnings.filterwarnings('ignore')

# Statistical modeling libraries
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available - limited statistical analysis")

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available - limited ML capabilities")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UncertaintyQuantification:
    """
    Uncertainty Quantification System

    Advanced system for quantifying uncertainty in Bayesian optimization
    results and AI predictions, providing confidence measures and risk assessment.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent.parent
        self.config_path = config_path or self.base_dir / "config" / "uncertainty_config.json"

        # Uncertainty state
        self.prediction_uncertainty = {}
        self.optimization_uncertainty = {}
        self.sensitivity_analysis = {}
        self.risk_assessment = {}

        # Statistical models
        self.confidence_models = {}
        self.uncertainty_models = {}

        # Configuration
        self.config = self._load_config()

        # Set random seed for reproducibility
        np.random.seed(42)

        logger.info("🔬 Uncertainty Quantification System initialized")
        logger.info(f"📊 Confidence level: {self.config['quantification']['confidence_level']:.1%}")
        logger.info(f"🎯 Risk tolerance: {self.config['quantification']['risk_tolerance']:.1%}")

    def _load_config(self) -> Dict[str, Any]:
        """Load uncertainty quantification configuration"""
        default_config = {
            'quantification': {
                'confidence_level': 0.95,
                'risk_tolerance': 0.05,
                'n_uncertainty_samples': 1000,
                'bootstrap_iterations': 1000,
                'sensitivity_samples': 500
            },
            'methods': {
                'prediction_uncertainty': 'gaussian_process',
                'optimization_uncertainty': 'bootstrap',
                'sensitivity_analysis': 'sobol',
                'risk_assessment': 'monte_carlo'
            },
            'validation': {
                'cross_validation_folds': 5,
                'holdout_fraction': 0.2,
                'significance_level': 0.05
            },
            'reporting': {
                'generate_confidence_intervals': True,
                'calculate_prediction_intervals': True,
                'compute_risk_metrics': True,
                'sensitivity_plots': True
            }
        }

        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                for key, value in loaded_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value

        return default_config

    def quantify_prediction_uncertainty(self, predictions: List[Dict[str, Any]],
                                      actual_values: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Quantify uncertainty in AI Kernel Predictor predictions

        Args:
            predictions: List of prediction results from AI predictor
            actual_values: Actual measured values for validation (optional)

        Returns:
            Prediction uncertainty analysis
        """
        logger.info("🔍 Quantifying prediction uncertainty...")

        if not predictions:
            return {'error': 'No predictions provided'}

        # Extract prediction data
        predicted_gflops = []
        confidence_scores = []
        prediction_errors = []

        for pred in predictions:
            if 'predicted_gflops' in pred:
                predicted_gflops.append(pred['predicted_gflops'])
                confidence_scores.append(pred.get('confidence', 0.5))

                if actual_values and len(actual_values) == len(predictions):
                    # Calculate prediction errors if actual values available
                    actual = actual_values[len(prediction_errors)]
                    error = abs(pred['predicted_gflops'] - actual) / actual
                    prediction_errors.append(error)

        predicted_gflops = np.array(predicted_gflops)
        confidence_scores = np.array(confidence_scores)

        # Calculate basic statistics
        mean_prediction = np.mean(predicted_gflops)
        std_prediction = np.std(predicted_gflops)
        cv_prediction = std_prediction / mean_prediction if mean_prediction > 0 else 0

        # Confidence intervals
        confidence_level = self.config['quantification']['confidence_level']
        z_score = norm.ppf(1 - (1 - confidence_level) / 2)

        ci_lower = mean_prediction - z_score * std_prediction / np.sqrt(len(predicted_gflops))
        ci_upper = mean_prediction + z_score * std_prediction / np.sqrt(len(predicted_gflops))

        # Prediction intervals (wider than confidence intervals)
        prediction_interval_factor = 2.0  # Conservative factor
        pi_lower = mean_prediction - prediction_interval_factor * std_prediction
        pi_upper = mean_prediction + prediction_interval_factor * std_prediction

        # Uncertainty metrics
        uncertainty_metrics = {
            'mean_prediction': mean_prediction,
            'std_prediction': std_prediction,
            'coefficient_of_variation': cv_prediction,
            'confidence_interval': {
                'level': confidence_level,
                'lower': ci_lower,
                'upper': ci_upper,
                'width': ci_upper - ci_lower
            },
            'prediction_interval': {
                'lower': pi_lower,
                'upper': pi_upper,
                'width': pi_upper - pi_lower
            },
            'confidence_distribution': {
                'mean_confidence': np.mean(confidence_scores),
                'std_confidence': np.std(confidence_scores),
                'min_confidence': np.min(confidence_scores),
                'max_confidence': np.max(confidence_scores)
            }
        }

        # Error analysis if actual values available
        if prediction_errors:
            uncertainty_metrics['error_analysis'] = {
                'mean_absolute_error': np.mean(prediction_errors),
                'root_mean_squared_error': np.sqrt(np.mean(np.array(prediction_errors)**2)),
                'max_error': np.max(prediction_errors),
                'error_std': np.std(prediction_errors),
                'predictions_within_10pct': np.mean(np.array(prediction_errors) <= 0.1),
                'predictions_within_20pct': np.mean(np.array(prediction_errors) <= 0.2)
            }

            # Calibration analysis
            uncertainty_metrics['calibration'] = self._analyze_prediction_calibration(
                confidence_scores, prediction_errors
            )

        # Bootstrap uncertainty estimation
        if self.config['quantification']['bootstrap_iterations'] > 0:
            uncertainty_metrics['bootstrap_analysis'] = self._bootstrap_uncertainty_analysis(
                predicted_gflops, self.config['quantification']['bootstrap_iterations']
            )

        self.prediction_uncertainty = uncertainty_metrics
        logger.info("✅ Prediction uncertainty quantified")
        logger.info(".2f"        logger.info(".2f"
        return uncertainty_metrics

    def _analyze_prediction_calibration(self, confidence_scores: np.ndarray,
                                      prediction_errors: List[float]) -> Dict[str, Any]:
        """Analyze calibration of confidence scores vs actual errors"""
        # Bin confidence scores and analyze error rates
        bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        bin_centers = (bins[:-1] + bins[1:]) / 2

        calibration_errors = []
        observed_confidence = []

        for i in range(len(bins) - 1):
            mask = (confidence_scores >= bins[i]) & (confidence_scores < bins[i+1])
            if np.sum(mask) > 0:
                bin_errors = np.array(prediction_errors)[mask]
                # Observed confidence = 1 - mean error rate in bin
                observed_conf = 1 - np.mean(bin_errors)
                observed_confidence.append(observed_conf)
                calibration_errors.append(abs(bin_centers[i] - observed_conf))

        return {
            'mean_calibration_error': np.mean(calibration_errors) if calibration_errors else 0,
            'max_calibration_error': np.max(calibration_errors) if calibration_errors else 0,
            'calibration_curve': {
                'expected_confidence': bin_centers.tolist(),
                'observed_confidence': observed_confidence
            }
        }

    def _bootstrap_uncertainty_analysis(self, data: np.ndarray, n_bootstrap: int) -> Dict[str, Any]:
        """Perform bootstrap analysis for uncertainty estimation"""
        bootstrap_means = []
        bootstrap_stds = []

        n_samples = len(data)
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_sample = data[indices]

            bootstrap_means.append(np.mean(bootstrap_sample))
            bootstrap_stds.append(np.std(bootstrap_sample))

        # Bootstrap confidence intervals
        bootstrap_means = np.array(bootstrap_means)
        bootstrap_stds = np.array(bootstrap_stds)

        ci_level = self.config['quantification']['confidence_level']
        ci_lower = np.percentile(bootstrap_means, (1 - ci_level) / 2 * 100)
        ci_upper = np.percentile(bootstrap_means, (1 + ci_level) / 2 * 100)

        return {
            'bootstrap_mean': np.mean(bootstrap_means),
            'bootstrap_std': np.std(bootstrap_means),
            'bootstrap_confidence_interval': {
                'lower': ci_lower,
                'upper': ci_upper,
                'width': ci_upper - ci_lower
            },
            'bootstrap_bias': np.mean(bootstrap_means) - np.mean(data)
        }

    def quantify_optimization_uncertainty(self, optimization_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Quantify uncertainty in optimization results

        Args:
            optimization_results: List of optimization run results

        Returns:
            Optimization uncertainty analysis
        """
        logger.info("🔍 Quantifying optimization uncertainty...")

        if not optimization_results:
            return {'error': 'No optimization results provided'}

        # Extract performance metrics from all runs
        best_performances = []
        convergence_iterations = []
        optimization_times = []

        for result in optimization_results:
            if 'best_performance' in result:
                best_performances.append(result['best_performance'])
            if 'total_evaluations' in result:
                convergence_iterations.append(result['total_evaluations'])
            if 'execution_time' in result:
                optimization_times.append(result['execution_time'])

        best_performances = np.array(best_performances)

        # Statistical analysis of optimization results
        mean_performance = np.mean(best_performances)
        std_performance = np.std(best_performances)
        cv_performance = std_performance / mean_performance if mean_performance > 0 else 0

        # Confidence intervals
        confidence_level = self.config['quantification']['confidence_level']
        if len(best_performances) > 1:
            if len(best_performances) < 30:
                # t-distribution for small samples
                t_score = t.ppf(1 - (1 - confidence_level) / 2, len(best_performances) - 1)
                ci_margin = t_score * std_performance / np.sqrt(len(best_performances))
            else:
                # Normal distribution for large samples
                z_score = norm.ppf(1 - (1 - confidence_level) / 2)
                ci_margin = z_score * std_performance / np.sqrt(len(best_performances))

            ci_lower = mean_performance - ci_margin
            ci_upper = mean_performance + ci_margin
        else:
            ci_lower = ci_upper = mean_performance

        # Robustness analysis
        robustness_metrics = self._analyze_optimization_robustness(best_performances)

        # Convergence analysis
        convergence_metrics = {}
        if convergence_iterations:
            convergence_metrics = {
                'mean_convergence_iterations': np.mean(convergence_iterations),
                'std_convergence_iterations': np.std(convergence_iterations),
                'min_convergence_iterations': np.min(convergence_iterations),
                'max_convergence_iterations': np.max(convergence_iterations)
            }

        # Performance distribution analysis
        performance_distribution = self._analyze_performance_distribution(best_performances)

        uncertainty_metrics = {
            'performance_statistics': {
                'mean_performance': mean_performance,
                'std_performance': std_performance,
                'coefficient_of_variation': cv_performance,
                'min_performance': np.min(best_performances),
                'max_performance': np.max(best_performances),
                'performance_range': np.max(best_performances) - np.min(best_performances)
            },
            'confidence_intervals': {
                'level': confidence_level,
                'lower': ci_lower,
                'upper': ci_upper,
                'width': ci_upper - ci_lower,
                'relative_width': (ci_upper - ci_lower) / mean_performance if mean_performance > 0 else 0
            },
            'robustness_analysis': robustness_metrics,
            'convergence_analysis': convergence_metrics,
            'performance_distribution': performance_distribution,
            'n_optimization_runs': len(optimization_results)
        }

        self.optimization_uncertainty = uncertainty_metrics
        logger.info("✅ Optimization uncertainty quantified")
        logger.info(".2f"        logger.info(".2f"
        return uncertainty_metrics

    def _analyze_optimization_robustness(self, performances: np.ndarray) -> Dict[str, Any]:
        """Analyze robustness of optimization results"""
        # Quartile analysis
        q25, q50, q75 = np.percentile(performances, [25, 50, 75])

        # Robustness metrics
        iqr = q75 - q25  # Interquartile range
        robustness_ratio = iqr / q50 if q50 > 0 else 0

        # Outlier analysis
        q1, q3 = np.percentile(performances, [25, 75])
        iqr_outlier = q3 - q1
        lower_bound = q1 - 1.5 * iqr_outlier
        upper_bound = q3 + 1.5 * iqr_outlier

        outliers = performances[(performances < lower_bound) | (performances > upper_bound)]
        outlier_percentage = len(outliers) / len(performances) * 100

        return {
            'quartiles': {
                'q25': q25,
                'q50_median': q50,
                'q75': q75
            },
            'iqr': iqr,
            'robustness_ratio': robustness_ratio,
            'outlier_analysis': {
                'n_outliers': len(outliers),
                'outlier_percentage': outlier_percentage,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            },
            'robustness_score': 1 - robustness_ratio  # Higher score = more robust
        }

    def _analyze_performance_distribution(self, performances: np.ndarray) -> Dict[str, Any]:
        """Analyze the distribution of optimization performances"""
        # Normality test
        if len(performances) >= 3:
            _, normality_p_value = stats.shapiro(performances)
            is_normal = normality_p_value > self.config['validation']['significance_level']
        else:
            is_normal = False
            normality_p_value = 1.0

        # Skewness and kurtosis
        skewness = stats.skew(performances)
        kurtosis = stats.kurtosis(performances)

        # Distribution type classification
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            distribution_type = 'normal'
        elif skewness > 1:
            distribution_type = 'right_skewed'
        elif skewness < -1:
            distribution_type = 'left_skewed'
        else:
            distribution_type = 'moderate_skew'

        return {
            'normality_test': {
                'p_value': normality_p_value,
                'is_normal': is_normal,
                'significance_level': self.config['validation']['significance_level']
            },
            'shape_parameters': {
                'skewness': skewness,
                'kurtosis': kurtosis
            },
            'distribution_type': distribution_type,
            'percentiles': {
                'p5': np.percentile(performances, 5),
                'p10': np.percentile(performances, 10),
                'p90': np.percentile(performances, 90),
                'p95': np.percentile(performances, 95)
            }
        }

    def sensitivity_analysis(self, config_space: Dict[str, Tuple[float, float]],
                           objective_function: Callable, n_samples: int = 500) -> Dict[str, Any]:
        """
        Perform sensitivity analysis to identify influential parameters

        Args:
            config_space: Parameter space bounds
            objective_function: Function to evaluate configurations
            n_samples: Number of samples for analysis

        Returns:
            Sensitivity analysis results
        """
        logger.info("🔍 Performing sensitivity analysis...")

        # Generate parameter samples
        param_names = list(config_space.keys())
        param_bounds = list(config_space.values())

        # Latin Hypercube Sampling for better space coverage
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=len(param_names))
        samples = sampler.random(n_samples)

        # Scale samples to parameter bounds
        scaled_samples = np.zeros_like(samples)
        for i, (low, high) in enumerate(param_bounds):
            scaled_samples[:, i] = low + samples[:, i] * (high - low)

        # Evaluate objective function for all samples
        logger.info("📊 Evaluating sensitivity samples...")
        objective_values = []
        for sample in scaled_samples:
            config = dict(zip(param_names, sample))
            try:
                obj_value = objective_function(**config)
                objective_values.append(obj_value)
            except Exception as e:
                logger.debug(f"Sensitivity evaluation failed for config {config}: {e}")
                objective_values.append(0)  # Default value

        objective_values = np.array(objective_values)

        # Calculate sensitivity indices
        sensitivity_indices = {}
        for i, param_name in enumerate(param_names):
            # Correlation-based sensitivity
            correlation = np.corrcoef(scaled_samples[:, i], objective_values)[0, 1]
            correlation = abs(correlation)  # Absolute value for importance

            # Standard deviation of objective vs parameter
            param_std = np.std(scaled_samples[:, i])
            obj_std = np.std(objective_values)

            if param_std > 0:
                sensitivity_ratio = correlation * (obj_std / param_std)
            else:
                sensitivity_ratio = 0

            sensitivity_indices[param_name] = {
                'correlation': correlation,
                'sensitivity_ratio': sensitivity_ratio,
                'parameter_std': param_std,
                'importance_rank': 0  # Will be set below
            }

        # Rank parameters by importance
        sorted_params = sorted(sensitivity_indices.items(),
                             key=lambda x: x[1]['sensitivity_ratio'], reverse=True)

        for rank, (param_name, _) in enumerate(sorted_params):
            sensitivity_indices[param_name]['importance_rank'] = rank + 1

        # Identify most influential parameters
        top_parameters = [param for param, _ in sorted_params[:3]]

        analysis_results = {
            'sensitivity_indices': sensitivity_indices,
            'parameter_ranking': [param for param, _ in sorted_params],
            'top_influential_parameters': top_parameters,
            'n_samples': n_samples,
            'objective_stats': {
                'mean': np.mean(objective_values),
                'std': np.std(objective_values),
                'min': np.min(objective_values),
                'max': np.max(objective_values)
            }
        }

        self.sensitivity_analysis = analysis_results
        logger.info("✅ Sensitivity analysis completed")
        logger.info(f"🎯 Top parameters: {', '.join(top_parameters)}")

        return analysis_results

    def risk_assessment(self, optimization_results: Dict[str, Any],
                       risk_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment

        Args:
            optimization_results: Results from optimization
            risk_thresholds: Custom risk thresholds (optional)

        Returns:
            Risk assessment results
        """
        logger.info("⚠️ Performing risk assessment...")

        if risk_thresholds is None:
            risk_thresholds = {
                'performance_drop_threshold': 0.1,  # 10% drop from best
                'uncertainty_threshold': 0.15,      # 15% coefficient of variation
                'failure_probability_threshold': 0.2  # 20% failure probability
            }

        # Extract key metrics
        best_performance = optimization_results.get('best_performance', 0)
        mean_performance = optimization_results.get('mean_performance', best_performance)
        std_performance = optimization_results.get('std_performance', 0)

        # Performance risk
        performance_risk = {
            'worst_case_performance': mean_performance - 2 * std_performance,
            'performance_drop_risk': (best_performance - (mean_performance - std_performance)) / best_performance,
            'below_threshold_probability': 0  # Would need more data for proper calculation
        }

        # Uncertainty risk
        uncertainty_risk = {
            'coefficient_of_variation': std_performance / mean_performance if mean_performance > 0 else 0,
            'uncertainty_level': 'high' if std_performance / mean_performance > risk_thresholds['uncertainty_threshold'] else 'low'
        }

        # Overall risk score (0-1, higher = more risky)
        risk_factors = [
            min(1.0, performance_risk['performance_drop_risk'] / risk_thresholds['performance_drop_threshold']),
            min(1.0, uncertainty_risk['coefficient_of_variation'] / risk_thresholds['uncertainty_threshold'])
        ]

        overall_risk_score = np.mean(risk_factors)

        # Risk mitigation recommendations
        recommendations = []
        if overall_risk_score > 0.7:
            recommendations.append("High risk detected - consider additional optimization runs")
        if uncertainty_risk['coefficient_of_variation'] > risk_thresholds['uncertainty_threshold']:
            recommendations.append("High uncertainty - increase sample size or use more robust methods")
        if performance_risk['performance_drop_risk'] > risk_thresholds['performance_drop_threshold']:
            recommendations.append("Performance variability high - focus on robust configurations")

        risk_assessment = {
            'overall_risk_score': overall_risk_score,
            'risk_level': 'high' if overall_risk_score > 0.7 else 'medium' if overall_risk_score > 0.4 else 'low',
            'performance_risk': performance_risk,
            'uncertainty_risk': uncertainty_risk,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'risk_thresholds_used': risk_thresholds
        }

        self.risk_assessment = risk_assessment
        logger.info("✅ Risk assessment completed")
        logger.info(".2f"        logger.info(f"⚠️ Risk level: {risk_assessment['risk_level']}")

        return risk_assessment

    def comprehensive_uncertainty_analysis(self, predictions: List[Dict[str, Any]] = None,
                                         optimization_results: List[Dict[str, Any]] = None,
                                         config_space: Dict[str, Tuple[float, float]] = None,
                                         objective_function: Callable = None) -> Dict[str, Any]:
        """
        Run comprehensive uncertainty quantification suite

        Args:
            predictions: AI predictor results
            optimization_results: Optimization run results
            config_space: Parameter space for sensitivity analysis
            objective_function: Objective function for sensitivity analysis

        Returns:
            Complete uncertainty analysis
        """
        logger.info("🚀 Starting comprehensive uncertainty analysis...")

        results = {
            'timestamp': time.time(),
            'prediction_uncertainty': {},
            'optimization_uncertainty': {},
            'sensitivity_analysis': {},
            'risk_assessment': {},
            'summary': {}
        }

        # 1. Prediction uncertainty
        if predictions:
            logger.info("📍 Phase 1: Prediction uncertainty analysis")
            results['prediction_uncertainty'] = self.quantify_prediction_uncertainty(predictions)

        # 2. Optimization uncertainty
        if optimization_results:
            logger.info("📍 Phase 2: Optimization uncertainty analysis")
            results['optimization_uncertainty'] = self.quantify_optimization_uncertainty(optimization_results)

        # 3. Sensitivity analysis
        if config_space and objective_function:
            logger.info("📍 Phase 3: Sensitivity analysis")
            results['sensitivity_analysis'] = self.sensitivity_analysis(
                config_space, objective_function, self.config['quantification']['sensitivity_samples']
            )

        # 4. Risk assessment
        if results['optimization_uncertainty']:
            logger.info("📍 Phase 4: Risk assessment")
            results['risk_assessment'] = self.risk_assessment(results['optimization_uncertainty'])

        # 5. Generate summary
        results['summary'] = self._generate_uncertainty_summary(results)

        logger.info("🎉 Comprehensive uncertainty analysis completed!")
        return results

    def _generate_uncertainty_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of uncertainty analysis"""
        summary = {
            'overall_confidence_level': 'high',
            'key_findings': [],
            'recommendations': [],
            'confidence_metrics': {}
        }

        # Assess overall confidence
        risk_score = results.get('risk_assessment', {}).get('overall_risk_score', 0.5)

        if risk_score < 0.3:
            summary['overall_confidence_level'] = 'high'
        elif risk_score < 0.6:
            summary['overall_confidence_level'] = 'medium'
        else:
            summary['overall_confidence_level'] = 'low'

        # Key findings
        if results.get('prediction_uncertainty'):
            pred_uncertainty = results['prediction_uncertainty']
            cv = pred_uncertainty.get('coefficient_of_variation', 0)
            summary['key_findings'].append(f"Prediction uncertainty: {cv:.1%} CV")

        if results.get('optimization_uncertainty'):
            opt_uncertainty = results['optimization_uncertainty']
            perf_std = opt_uncertainty['performance_statistics']['std_performance']
            summary['key_findings'].append(f"Optimization variability: ±{perf_std:.1f} GFLOPS")

        if results.get('sensitivity_analysis'):
            sens_analysis = results['sensitivity_analysis']
            top_params = sens_analysis.get('top_influential_parameters', [])
            if top_params:
                summary['key_findings'].append(f"Key parameters: {', '.join(top_params)}")

        # Recommendations
        if results.get('risk_assessment'):
            risk_assess = results['risk_assessment']
            summary['recommendations'].extend(risk_assess.get('recommendations', []))

        # Confidence metrics
        summary['confidence_metrics'] = {
            'prediction_confidence': results.get('prediction_uncertainty', {}).get('confidence_interval', {}),
            'optimization_confidence': results.get('optimization_uncertainty', {}).get('confidence_intervals', {}),
            'risk_score': risk_score
        }

        return summary

    def save_uncertainty_results(self, results: Dict[str, Any],
                               filename: str = "uncertainty_analysis_results.json") -> None:
        """Save uncertainty analysis results"""
        output_path = self.base_dir / "results" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"💾 Uncertainty analysis results saved to {output_path}")

def main():
    """Main function for uncertainty quantification"""
    try:
        # Initialize uncertainty quantifier
        quantifier = UncertaintyQuantification()

        # Example analysis with synthetic data
        logger.info("🔬 Running example uncertainty analysis with synthetic data...")

        # Generate synthetic prediction data
        np.random.seed(42)
        n_predictions = 50
        true_values = np.random.normal(420, 20, n_predictions)
        predicted_values = true_values + np.random.normal(0, 15, n_predictions)  # 17.7% MAPE simulation

        predictions = []
        for i in range(n_predictions):
            predictions.append({
                'predicted_gflops': predicted_values[i],
                'confidence': np.random.beta(2, 1)  # Skewed toward higher confidence
            })

        # Generate synthetic optimization results
        optimization_results = []
        for i in range(10):
            optimization_results.append({
                'best_performance': np.random.normal(450, 15),
                'total_evaluations': np.random.randint(20, 40),
                'execution_time': np.random.uniform(30, 120)
            })

        # Run comprehensive analysis
        results = quantifier.comprehensive_uncertainty_analysis(
            predictions=predictions,
            optimization_results=optimization_results
        )

        # Save results
        quantifier.save_uncertainty_results(results)

        # Print summary
        print("\n" + "="*60)
        print("🎉 UNCERTAINTY QUANTIFICATION COMPLETED")
        print("="*60)

        summary = results['summary']
        print(f"📊 Overall confidence: {summary['overall_confidence_level'].upper()}")
        print("🔍 Key findings:"
        for finding in summary['key_findings']:
            print(f"   • {finding}")
        print("💡 Recommendations:"
        for rec in summary['recommendations']:
            print(f"   • {rec}")
        print("="*60)

    except Exception as e:
        logger.error(f"Uncertainty quantification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()