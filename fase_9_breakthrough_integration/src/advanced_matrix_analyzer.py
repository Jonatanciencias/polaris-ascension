#!/usr/bin/env python3
"""
🔍 MÉTRICAS AVANZADAS DE ANÁLISIS DE MATRICES
============================================

Sistema comprehensivo para extraer características avanzadas de matrices
que mejoran significativamente la selección de técnicas de optimización.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

class MatrixStructure(Enum):
    """Tipos de estructura de matriz"""
    DENSE = "dense"
    SPARSE = "sparse"
    DIAGONAL = "diagonal"
    TRIANGULAR_UPPER = "triangular_upper"
    TRIANGULAR_LOWER = "triangular_lower"
    BANDED = "banded"
    BLOCK = "block"
    TOEPLITZ = "toeplitz"
    CIRCULANT = "circulant"

class MatrixSymmetry(Enum):
    """Tipos de simetría"""
    GENERAL = "general"
    SYMMETRIC = "symmetric"
    SKEW_SYMMETRIC = "skew_symmetric"
    HERMITIAN = "hermitian"
    ORTHOGONAL = "orthogonal"

@dataclass
class SpectralProperties:
    """Propiedades espectrales de la matriz"""
    eigenvalues: Optional[np.ndarray] = None
    condition_number: float = 1.0
    spectral_radius: float = 0.0
    numerical_rank: int = 0
    singular_values: Optional[np.ndarray] = None
    trace: float = 0.0
    determinant: Optional[float] = None

@dataclass
class StructuralProperties:
    """Propiedades estructurales"""
    structure_type: MatrixStructure = MatrixStructure.DENSE
    symmetry_type: MatrixSymmetry = MatrixSymmetry.GENERAL
    bandwidth: Optional[int] = None
    block_size: Optional[Tuple[int, int]] = None
    density_pattern: str = "uniform"
    clustering_coefficient: float = 0.0

@dataclass
class ComputationalProperties:
    """Propiedades computacionales"""
    memory_access_pattern: str = "regular"
    cache_locality: float = 0.0
    arithmetic_intensity: float = 0.0
    load_balance_factor: float = 1.0
    communication_to_computation_ratio: float = 0.0

@dataclass
class AdvancedMatrixFeatures:
    """Características avanzadas completas de matrices"""
    # Propiedades básicas (del sistema original)
    size_a: Tuple[int, int]
    size_b: Tuple[int, int]
    dtype: str
    sparsity_a: float
    sparsity_b: float
    memory_footprint_mb: float
    compute_intensity: float

    # Propiedades espectrales avanzadas
    spectral_a: SpectralProperties = field(default_factory=SpectralProperties)
    spectral_b: SpectralProperties = field(default_factory=SpectralProperties)

    # Propiedades estructurales
    structure_a: StructuralProperties = field(default_factory=StructuralProperties)
    structure_b: StructuralProperties = field(default_factory=StructuralProperties)

    # Propiedades computacionales
    computational: ComputationalProperties = field(default_factory=ComputationalProperties)

    # Métricas derivadas para ML
    ml_features: Dict[str, float] = field(default_factory=dict)

class AdvancedMatrixAnalyzer:
    """
    Analizador avanzado de matrices que extrae características
    comprehensivas para mejorar la selección de técnicas.
    """

    def __init__(self, max_eigenvalue_samples: int = 100, enable_full_spectral: bool = False):
        self.max_eigenvalue_samples = max_eigenvalue_samples
        self.enable_full_spectral = enable_full_spectral

    def analyze_matrices(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> AdvancedMatrixFeatures:
        """
        Análisis completo y avanzado de las matrices de entrada.

        Args:
            matrix_a, matrix_b: Matrices de entrada

        Returns:
            AdvancedMatrixFeatures con todas las características extraídas
        """
        print("🔍 Realizando análisis avanzado de matrices...")

        # Extraer características básicas
        basic_features = self._extract_basic_features(matrix_a, matrix_b)

        # Análisis espectral avanzado
        spectral_a = self._analyze_spectral_properties(matrix_a)
        spectral_b = self._analyze_spectral_properties(matrix_b)

        # Análisis estructural
        structure_a = self._analyze_structural_properties(matrix_a)
        structure_b = self._analyze_structural_properties(matrix_b)

        # Propiedades computacionales
        computational = self._analyze_computational_properties(matrix_a, matrix_b)

        # Features para machine learning
        ml_features = self._extract_ml_features(matrix_a, matrix_b, spectral_a, spectral_b)

        features = AdvancedMatrixFeatures(
            size_a=matrix_a.shape,
            size_b=matrix_b.shape,
            dtype=str(matrix_a.dtype),
            sparsity_a=basic_features['sparsity_a'],
            sparsity_b=basic_features['sparsity_b'],
            memory_footprint_mb=basic_features['memory_footprint'],
            compute_intensity=basic_features['compute_intensity'],
            spectral_a=spectral_a,
            spectral_b=spectral_b,
            structure_a=structure_a,
            structure_b=structure_b,
            computational=computational,
            ml_features=ml_features
        )

        return features

    def _extract_basic_features(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> Dict[str, float]:
        """Extrae características básicas (versión mejorada)"""
        # Sparsity con mejor estimación
        sparsity_a = self._calculate_advanced_sparsity(matrix_a)
        sparsity_b = self._calculate_advanced_sparsity(matrix_b)

        # Memory footprint
        memory_footprint = (matrix_a.nbytes + matrix_b.nbytes) / (1024 ** 2)

        # Compute intensity mejorada
        operations = 2 * matrix_a.shape[0] * matrix_a.shape[1] * matrix_b.shape[1]
        memory_access = (matrix_a.nbytes + matrix_b.nbytes +
                        matrix_a.shape[0] * matrix_b.shape[1] * matrix_a.dtype.itemsize)
        compute_intensity = operations / memory_access if memory_access > 0 else 0

        return {
            'sparsity_a': sparsity_a,
            'sparsity_b': sparsity_b,
            'memory_footprint': memory_footprint,
            'compute_intensity': compute_intensity
        }

    def _calculate_advanced_sparsity(self, matrix: np.ndarray) -> float:
        """Calcula sparsidad considerando patrones y distribución"""
        if matrix.size == 0:
            return 1.0

        # Sparsity básica
        basic_sparsity = 1.0 - (np.count_nonzero(matrix) / matrix.size)

        # Considerar valores pequeños como cero (numerical sparsity)
        threshold = 1e-10 * np.max(np.abs(matrix))
        numerical_sparsity = 1.0 - (np.sum(np.abs(matrix) > threshold) / matrix.size)

        # Promedio ponderado
        return 0.7 * basic_sparsity + 0.3 * numerical_sparsity

    def _analyze_spectral_properties(self, matrix: np.ndarray) -> SpectralProperties:
        """Análisis espectral avanzado de la matriz"""
        props = SpectralProperties()

        try:
            # Para matrices cuadradas, calcular propiedades espectrales
            if matrix.shape[0] == matrix.shape[1] and matrix.shape[0] <= self.max_eigenvalue_samples:
                # Eigenvalores y valores singulares
                if self.enable_full_spectral:
                    eigenvals = np.linalg.eigvals(matrix.astype(np.complex128))
                    props.eigenvalues = np.real(eigenvals)  # Solo parte real para estabilidad
                    props.spectral_radius = np.max(np.abs(eigenvals))
                else:
                    # Estimación más rápida
                    props.spectral_radius = np.linalg.norm(matrix, 2)

                # Condition number
                if np.linalg.cond(matrix) < 1e15:  # Evitar overflow
                    props.condition_number = np.linalg.cond(matrix)
                else:
                    props.condition_number = 1e15

                # Numerical rank
                singular_vals = np.linalg.svd(matrix, compute_uv=False)
                threshold = 1e-10 * singular_vals[0]
                props.numerical_rank = np.sum(singular_vals > threshold)
                props.singular_values = singular_vals[:10]  # Solo los primeros 10

                # Trace
                props.trace = np.trace(matrix)

                # Determinant (solo para matrices pequeñas)
                if matrix.shape[0] <= 10:
                    props.determinant = np.linalg.det(matrix)

            else:
                # Para matrices rectangulares o grandes, estimaciones
                props.spectral_radius = np.linalg.norm(matrix, 2)
                props.condition_number = np.linalg.cond(matrix)
                props.numerical_rank = min(matrix.shape)
                props.trace = np.sum(np.diag(matrix)) if matrix.shape[0] == matrix.shape[1] else 0.0

        except Exception as e:
            print(f"⚠️  Error en análisis espectral: {e}")
            # Valores por defecto
            props.condition_number = 1.0
            props.spectral_radius = np.linalg.norm(matrix, 2)

        return props

    def _analyze_structural_properties(self, matrix: np.ndarray) -> StructuralProperties:
        """Análisis estructural avanzado de la matriz"""
        props = StructuralProperties()

        # Determinar tipo de estructura
        props.structure_type = self._classify_matrix_structure(matrix)

        # Analizar simetría
        props.symmetry_type = self._analyze_symmetry(matrix)

        # Calcular bandwidth si aplica
        if props.structure_type in [MatrixStructure.BANDED, MatrixStructure.TRIANGULAR_UPPER,
                                   MatrixStructure.TRIANGULAR_LOWER]:
            props.bandwidth = self._calculate_bandwidth(matrix)

        # Detectar bloques
        props.block_size = self._detect_block_structure(matrix)

        # Patrón de densidad
        props.density_pattern = self._analyze_density_pattern(matrix)

        # Clustering coefficient (para matrices sparse)
        if props.structure_type == MatrixStructure.SPARSE:
            props.clustering_coefficient = self._calculate_clustering_coefficient(matrix)

        return props

    def _classify_matrix_structure(self, matrix: np.ndarray) -> MatrixStructure:
        """Clasifica la estructura de la matriz"""
        sparsity = 1.0 - (np.count_nonzero(matrix) / matrix.size)

        if sparsity > 0.9:
            return MatrixStructure.SPARSE
        elif self._is_diagonal(matrix):
            return MatrixStructure.DIAGONAL
        elif self._is_triangular(matrix, upper=True):
            return MatrixStructure.TRIANGULAR_UPPER
        elif self._is_triangular(matrix, upper=False):
            return MatrixStructure.TRIANGULAR_LOWER
        elif self._is_banded(matrix):
            return MatrixStructure.BANDED
        elif self._is_block_matrix(matrix):
            return MatrixStructure.BLOCK
        elif self._is_toeplitz(matrix):
            return MatrixStructure.TOEPLITZ
        else:
            return MatrixStructure.DENSE

    def _is_diagonal(self, matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Verifica si la matriz es diagonal"""
        if matrix.shape[0] != matrix.shape[1]:
            return False
        off_diagonal = matrix - np.diag(np.diag(matrix))
        return np.allclose(off_diagonal, 0, atol=tolerance)

    def _is_triangular(self, matrix: np.ndarray, upper: bool = True, tolerance: float = 1e-10) -> bool:
        """Verifica si la matriz es triangular"""
        if matrix.shape[0] != matrix.shape[1]:
            return False

        if upper:
            lower_part = np.tril(matrix, -1)
            return np.allclose(lower_part, 0, atol=tolerance)
        else:
            upper_part = np.triu(matrix, 1)
            return np.allclose(upper_part, 0, atol=tolerance)

    def _is_banded(self, matrix: np.ndarray, max_bandwidth: int = 10) -> bool:
        """Verifica si la matriz es banded"""
        if matrix.shape[0] != matrix.shape[1]:
            return False

        n = matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if abs(i - j) > max_bandwidth and abs(matrix[i, j]) > 1e-10:
                    return False
        return True

    def _is_block_matrix(self, matrix: np.ndarray) -> bool:
        """Detecta si es una matriz de bloques"""
        # Implementación simplificada
        n = min(matrix.shape)
        block_sizes = [n//2, n//3, n//4]

        for block_size in block_sizes:
            if n % block_size == 0:
                # Verificar si los bloques fuera de la diagonal son cero
                try:
                    blocks = self._extract_blocks(matrix, block_size)
                    # Verificar si bloques off-diagonal son aproximadamente cero
                    for i in range(len(blocks)):
                        for j in range(len(blocks[i])):
                            if i != j:
                                block = blocks[i][j]
                                if not np.allclose(block, 0, atol=1e-6):
                                    continue
                    return True
                except:
                    continue
        return False

    def _is_toeplitz(self, matrix: np.ndarray) -> bool:
        """Verifica si es una matriz Toeplitz"""
        if matrix.shape[0] != matrix.shape[1]:
            return False

        n = matrix.shape[0]
        for i in range(1, n-1):
            for j in range(1, n-i):
                if not np.isclose(matrix[i, j], matrix[i-1, j-1], atol=1e-10):
                    return False
        return True

    def _analyze_symmetry(self, matrix: np.ndarray) -> MatrixSymmetry:
        """Analiza el tipo de simetría de la matriz"""
        if matrix.shape[0] != matrix.shape[1]:
            return MatrixSymmetry.GENERAL

        n = matrix.shape[0]

        # Verificar simetría
        if np.allclose(matrix, matrix.T, atol=1e-10):
            # Verificar si es skew-symmetric
            if np.allclose(matrix, -matrix.T, atol=1e-10):
                return MatrixSymmetry.SKEW_SYMMETRIC
            else:
                return MatrixSymmetry.SYMMETRIC

        # Verificar hermitiana (para matrices complejas)
        if np.iscomplexobj(matrix):
            if np.allclose(matrix, matrix.conj().T, atol=1e-10):
                return MatrixSymmetry.HERMITIAN

        # Verificar ortogonal
        try:
            identity = np.eye(n)
            if np.allclose(matrix @ matrix.T, identity, atol=1e-6):
                return MatrixSymmetry.ORTHOGONAL
        except:
            pass

        return MatrixSymmetry.GENERAL

    def _calculate_bandwidth(self, matrix: np.ndarray) -> int:
        """Calcula el bandwidth de una matriz banded"""
        n = matrix.shape[0]
        max_band = 0

        for i in range(n):
            for j in range(n):
                if abs(matrix[i, j]) > 1e-10:
                    max_band = max(max_band, abs(i - j))

        return max_band

    def _detect_block_structure(self, matrix: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detecta estructura de bloques"""
        # Implementación simplificada
        n = min(matrix.shape)
        for block_size in [2, 4, 8, 16, 32]:
            if n % block_size == 0:
                return (block_size, block_size)
        return None

    def _analyze_density_pattern(self, matrix: np.ndarray) -> str:
        """Analiza el patrón de densidad de la matriz"""
        if matrix.size < 100:
            return "uniform"

        # Para matrices rectangulares, análisis simplificado
        if matrix.shape[0] != matrix.shape[1]:
            # Análisis basado en distribución de valores no cero
            flat_matrix = matrix.flatten()
            nonzero_indices = np.nonzero(flat_matrix)[0]

            if len(nonzero_indices) == 0:
                return "empty"

            # Calcular clustering simple basado en posiciones
            positions = nonzero_indices / len(flat_matrix)
            if np.std(positions) < 0.1:
                return "clustered"
            elif np.std(positions) > 0.3:
                return "scattered"
            else:
                return "semi_uniform"

        # Para matrices cuadradas, análisis completo
        try:
            # Convertir a sparse para análisis
            sparse_matrix = sp.csr_matrix(matrix)

            # Análisis de conectividad
            n_components, labels = sp.csgraph.connected_components(sparse_matrix)

            if n_components == 1:
                return "connected"
            elif n_components < matrix.shape[0] // 10:
                return "block_diagonal"
            else:
                return "scattered"
        except Exception:
            # Fallback para casos donde falla el análisis
            return "uniform"

    def _calculate_clustering_coefficient(self, matrix: np.ndarray) -> float:
        """Calcula clustering coefficient para matrices sparse"""
        try:
            # Para matrices rectangulares, usar análisis simplificado
            if matrix.shape[0] != matrix.shape[1]:
                # Análisis basado en distribución local de valores no cero
                sparsity = 1.0 - (np.count_nonzero(matrix) / matrix.size)
                # Clustering coefficient aproximado basado en sparsidad local
                return max(0.0, 1.0 - sparsity * 2)

            sparse_matrix = sp.csr_matrix(matrix)
            # Implementación simplificada del clustering coefficient
            n = sparse_matrix.shape[0]
            clustering = 0.0
            count = 0

            for i in range(min(n, 100)):  # Sample para eficiencia
                neighbors = sparse_matrix[i].nonzero()[1]
                if len(neighbors) < 2:
                    continue

                # Contar conexiones entre neighbors
                subgraph = sparse_matrix[neighbors][:, neighbors]
                possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                actual_edges = subgraph.nnz / 2  # Dividir por 2 para undirected

                if possible_edges > 0:
                    clustering += actual_edges / possible_edges
                    count += 1

            return clustering / count if count > 0 else 0.0

        except:
            return 0.0

    def _analyze_computational_properties(self, matrix_a: np.ndarray,
                                        matrix_b: np.ndarray) -> ComputationalProperties:
        """Analiza propiedades computacionales"""
        props = ComputationalProperties()

        # Patrón de acceso a memoria
        props.memory_access_pattern = self._analyze_memory_access_pattern(matrix_a, matrix_b)

        # Cache locality
        props.cache_locality = self._calculate_cache_locality(matrix_a, matrix_b)

        # Arithmetic intensity
        operations = 2 * matrix_a.shape[0] * matrix_a.shape[1] * matrix_b.shape[1]
        data_movement = matrix_a.nbytes + matrix_b.nbytes + (matrix_a.shape[0] * matrix_b.shape[1] * 4)
        props.arithmetic_intensity = operations / data_movement if data_movement > 0 else 0

        # Load balance factor
        props.load_balance_factor = self._calculate_load_balance(matrix_a, matrix_b)

        # Communication to computation ratio
        props.communication_to_computation_ratio = data_movement / operations if operations > 0 else 0

        return props

    def _analyze_memory_access_pattern(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> str:
        """Analiza el patrón de acceso a memoria"""
        # Análisis simplificado basado en estructura
        if matrix_a.shape[1] == matrix_b.shape[0]:
            return "regular"  # GEMM estándar
        elif matrix_a.shape[0] == matrix_b.shape[0]:
            return "broadcast"  # Broadcasting en una dimensión
        else:
            return "irregular"

    def _calculate_cache_locality(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
        """Calcula la localidad de cache estimada"""
        # Estimación simplificada basada en tamaños
        total_elements = matrix_a.size + matrix_b.size
        cache_size = 256 * 1024  # 256KB L2 cache asumido

        if total_elements * 4 < cache_size:
            return 1.0  # Todo cabe en cache
        else:
            return min(1.0, cache_size / (total_elements * 4))

    def _calculate_load_balance(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
        """Calcula el factor de balanceo de carga"""
        # Estimación simplificada
        size_a = matrix_a.shape[0] * matrix_a.shape[1]
        size_b = matrix_b.shape[0] * matrix_b.shape[1]

        if size_a == 0 or size_b == 0:
            return 1.0

        imbalance = abs(size_a - size_b) / max(size_a, size_b)
        return 1.0 - imbalance

    def _extract_ml_features(self, matrix_a: np.ndarray, matrix_b: np.ndarray,
                           spectral_a: SpectralProperties, spectral_b: SpectralProperties) -> Dict[str, float]:
        """Extrae features específicas para machine learning"""
        features = {}

        # Features básicas normalizadas
        features['size_ratio'] = matrix_b.shape[1] / matrix_a.shape[1] if matrix_a.shape[1] > 0 else 1.0
        features['sparsity_ratio'] = spectral_b.condition_number / spectral_a.condition_number if spectral_a.condition_number > 0 else 1.0
        features['memory_ratio'] = matrix_b.nbytes / matrix_a.nbytes if matrix_a.nbytes > 0 else 1.0

        # Features espectrales
        features['condition_ratio'] = spectral_b.condition_number / spectral_a.condition_number if spectral_a.condition_number > 0 else 1.0
        features['spectral_balance'] = min(spectral_a.spectral_radius, spectral_b.spectral_radius) / max(spectral_a.spectral_radius, spectral_b.spectral_radius) if max(spectral_a.spectral_radius, spectral_b.spectral_radius) > 0 else 0.5

        # Features de estructura
        features['structure_compatibility'] = 1.0 if matrix_a.shape[1] == matrix_b.shape[0] else 0.0

        # Features computacionales
        features['compute_to_memory_ratio'] = (matrix_a.shape[0] * matrix_a.shape[1] * matrix_b.shape[1]) / (matrix_a.nbytes + matrix_b.nbytes) if (matrix_a.nbytes + matrix_b.nbytes) > 0 else 0.0

        return features

    def _extract_blocks(self, matrix: np.ndarray, block_size: int) -> List[List[np.ndarray]]:
        """Extrae bloques de una matriz"""
        n = matrix.shape[0]
        blocks = []

        for i in range(0, n, block_size):
            row_blocks = []
            for j in range(0, n, block_size):
                block = matrix[i:i+block_size, j:j+block_size]
                row_blocks.append(block)
            blocks.append(row_blocks)

        return blocks

def demonstrate_advanced_analysis():
    """Demostración del análisis avanzado de matrices"""
    print("🚀 DEMOSTRACIÓN DE ANÁLISIS AVANZADO DE MATRICES")
    print("=" * 60)

    analyzer = AdvancedMatrixAnalyzer(enable_full_spectral=False)

    # Matrices de prueba
    test_matrices = [
        ("Matriz densa cuadrada", np.random.randn(64, 64)),
        ("Matriz sparse", np.random.randn(64, 64) * (np.random.rand(64, 64) > 0.9)),
        ("Matriz diagonal", np.diag(np.random.randn(64))),
        ("Matriz triangular superior", np.triu(np.random.randn(64, 64))),
    ]

    for name, matrix in test_matrices:
        print(f"\n📊 {name} ({matrix.shape})")
        print("-" * 40)

        # Análisis básico vs avanzado
        features = analyzer.analyze_matrices(matrix, matrix)  # Misma matriz para A y B

        print("🔍 Propiedades Espectrales:")
        print(f"   Condition number: {features.spectral_a.condition_number:.2e}")
        print(f"   Spectral radius: {features.spectral_a.spectral_radius:.2f}")
        print(f"   Numerical rank: {features.spectral_a.numerical_rank}")

        print("🏗️  Propiedades Estructurales:")
        print(f"   Estructura: {features.structure_a.structure_type.value}")
        print(f"   Simetría: {features.structure_a.symmetry_type.value}")
        if features.structure_a.bandwidth:
            print(f"   Bandwidth: {features.structure_a.bandwidth}")

        print("⚡ Propiedades Computacionales:")
        print(f"   Arithmetic intensity: {features.computational.arithmetic_intensity:.2f}")
        print(f"   Cache locality: {features.computational.cache_locality:.2f}")
        print(f"   Memory access pattern: {features.computational.memory_access_pattern}")

        print("🤖 Features para ML:")
        for key, value in list(features.ml_features.items())[:3]:  # Solo primeros 3
            print(f"   {key}: {value:.3f}")

    print("\n✅ Análisis avanzado completado")
    return analyzer

if __name__ == "__main__":
    analyzer = demonstrate_advanced_analysis()