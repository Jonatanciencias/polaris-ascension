# Utilities Package
# Helper functions and analysis tools
from importlib import import_module
from typing import Any


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    try:
        module = import_module(f".{module_name}", __name__)
        return getattr(module, symbol_name)
    except (ImportError, AttributeError):
        return None


quick_analysis = _load_symbol("quick_analysis", "quick_analysis")
AdvancedTechniquesInvestigator = _load_symbol(
    "advanced_techniques_investigation", "AdvancedTechniquesInvestigator"
)

__all__ = ["quick_analysis", "AdvancedTechniquesInvestigator"]
