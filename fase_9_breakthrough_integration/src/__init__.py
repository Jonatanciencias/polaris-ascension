# Fase 9: Breakthrough Techniques Integration
"""
Sistema integrado de técnicas breakthrough para optimización ML-based.

Este módulo proporciona acceso unificado a:
- BreakthroughTechniqueSelector: Selector inteligente de técnicas
- HybridOptimizer: Optimizador híbrido con múltiples estrategias
"""

from .breakthrough_selector import BreakthroughTechniqueSelector, BreakthroughTechnique
from .hybrid_optimizer import HybridOptimizer, HybridStrategy, HybridConfiguration

__version__ = "9.0.0"
__author__ = "AI Assistant"
__description__ = "Breakthrough Techniques Integration Framework"

__all__ = [
    'BreakthroughTechniqueSelector',
    'BreakthroughTechnique',
    'HybridOptimizer',
    'HybridStrategy',
    'HybridConfiguration'
]