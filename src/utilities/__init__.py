# Utilities Package
# Helper functions and analysis tools

try:
    from .quick_analysis import QuickAnalysis
except (ImportError, AttributeError):
    QuickAnalysis = None

try:
    from .advanced_techniques_investigation import AdvancedTechniquesInvestigation
except (ImportError, AttributeError):
    AdvancedTechniquesInvestigation = None

__all__ = ["QuickAnalysis", "AdvancedTechniquesInvestigation"]
