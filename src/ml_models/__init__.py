# Machine Learning Models Package
# AI-driven algorithm selection and optimization models
from importlib import import_module
from typing import Any


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    try:
        module = import_module(f".{module_name}", __name__)
        return getattr(module, symbol_name)
    except (ImportError, AttributeError):
        return None


AIKernelPredictorFineTuner = _load_symbol(
    "ai_kernel_predictor_fine_tuning", "AIKernelPredictorFineTuner"
)
AIKernelPredictorFineTunerCorrected = _load_symbol(
    "ai_kernel_predictor_fine_tuning_corrected", "AIKernelPredictorFineTunerCorrected"
)
MLDatasetCollector = _load_symbol("ml_dataset_collector", "MLDatasetCollector")
recalibrate_selector_with_hardware_data = _load_symbol(
    "recalibrate_selector", "recalibrate_selector_with_hardware_data"
)
CalibratedIntelligentSelector = _load_symbol(
    "calibrated_intelligent_selector", "CalibratedIntelligentSelector"
)
OptimizationTechnique = _load_symbol("calibrated_intelligent_selector", "OptimizationTechnique")
SelectionResult = _load_symbol("calibrated_intelligent_selector", "SelectionResult")
MatrixCharacteristics = _load_symbol("calibrated_intelligent_selector", "MatrixCharacteristics")

__all__ = [
    "AIKernelPredictorFineTuner",
    "AIKernelPredictorFineTunerCorrected",
    "MLDatasetCollector",
    "recalibrate_selector_with_hardware_data",
    "CalibratedIntelligentSelector",
    "OptimizationTechnique",
    "SelectionResult",
    "MatrixCharacteristics",
]
