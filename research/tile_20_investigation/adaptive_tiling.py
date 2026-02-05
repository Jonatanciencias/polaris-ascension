"""
Adaptive Tiling: Dynamic tile size calculation

Calcula el tile size óptimo basado en:
- Tamaño de la matriz (M, N, K)
- Capacidad de caché (L1/L2)
- Características del hardware

Teoría:
    Para GEMM tiled, los tiles de A, B y C deben caber en caché:
    - A tile: tile_m × tile_k elements
    - B tile: tile_k × tile_n elements  
    - C tile: tile_m × tile_n elements
    
    Total memory = (tile_m*tile_k + tile_k*tile_n + tile_m*tile_n) * sizeof(float)
    
    Queremos maximizar tile size sin exceder caché capacity
"""

import numpy as np
from typing import Tuple, Dict


class AdaptiveTiling:
    """Calcula tile sizes óptimos dinámicamente"""
    
    def __init__(self, 
                 l1_cache_size: int = 16384,    # 16 KB L1 per CU (RX 590)
                 l2_cache_size: int = 2097152,  # 2 MB L2 (RX 590)
                 max_work_group: int = 256,      # Max threads per work group
                 element_size: int = 4):         # sizeof(float)
        """
        Args:
            l1_cache_size: L1 cache size in bytes
            l2_cache_size: L2 cache size in bytes
            max_work_group: Maximum work group size
            element_size: Size of each element in bytes
        """
        self.l1_cache = l1_cache_size
        self.l2_cache = l2_cache_size
        self.max_work_group = max_work_group
        self.element_size = element_size
        
        # Available tile sizes (must be power-of-2 friendly for GPU)
        self.available_tiles = [8, 12, 16, 20, 24, 28, 32]
        
    def calculate_tile_memory(self, tile_m: int, tile_n: int, tile_k: int) -> int:
        """
        Calcula memoria requerida para tiles
        
        Memory layout in LDS:
        - A tile: tile_m × tile_k
        - B tile: tile_k × tile_n
        - C tile: tile_m × tile_n (en registros, no LDS)
        
        Returns:
            Bytes requeridos
        """
        # Solo A y B van a LDS, C está en registros
        a_tile_size = tile_m * tile_k * self.element_size
        b_tile_size = tile_k * tile_n * self.element_size
        
        return a_tile_size + b_tile_size
    
    def calculate_optimal_tile_l1(self, K: int) -> int:
        """
        Calcula tile óptimo para L1 cache
        
        Strategy: Maximizar tile que quepa en L1
        
        Args:
            K: K dimension (shared between A and B)
            
        Returns:
            Optimal tile size
        """
        # Para tile cuadrado: tile × K + K × tile ≤ L1_cache
        # 2 * K * tile ≤ L1_cache
        # tile ≤ L1_cache / (2 * K * element_size)
        
        max_tile_l1 = self.l1_cache // (2 * K * self.element_size)
        
        # Encontrar tile más grande que quepa
        for tile in reversed(self.available_tiles):
            if tile <= max_tile_l1:
                return tile
        
        # Si ninguno cabe, usar el más pequeño
        return self.available_tiles[0]
    
    def calculate_optimal_tile_l2(self, M: int, N: int, K: int) -> int:
        """
        Calcula tile óptimo para L2 cache
        
        L2 es más grande, podemos ser más agresivos
        
        Args:
            M, N, K: Matrix dimensions
            
        Returns:
            Optimal tile size
        """
        # Múltiples tiles pueden estar en L2
        # Heurística: 4 tiles de A, 4 tiles de B
        num_tiles = 8
        max_tile_l2 = int(np.sqrt(self.l2_cache / (num_tiles * 2 * K * self.element_size)))
        
        for tile in reversed(self.available_tiles):
            if tile <= max_tile_l2:
                return tile
                
        return self.available_tiles[-1]  # Si hay mucho espacio, usar el más grande
    
    def calculate_optimal_tile_work_group(self, tile_candidate: int) -> Tuple[int, int]:
        """
        Calcula configuración de work group óptima para un tile dado
        
        Args:
            tile_candidate: Candidate tile size
            
        Returns:
            (local_x, local_y) optimal work group configuration
        """
        # Queremos que local_x * local_y <= max_work_group
        # Y que cada thread procese tile/local_x × tile/local_y elementos
        
        # Opciones comunes para work groups
        configs = [
            (16, 16),  # 256 threads
            (16, 8),   # 128 threads
            (10, 10),  # 100 threads (nuestro v3)
            (8, 8),    # 64 threads
            (8, 4),    # 32 threads
        ]
        
        for local_x, local_y in configs:
            if local_x * local_y <= self.max_work_group:
                # Verificar si divide el tile limpiamente
                if tile_candidate % local_x == 0 or tile_candidate % local_y == 0:
                    return (local_x, local_y)
        
        # Fallback
        return (8, 8)
    
    def select_optimal_tile(self, M: int, N: int, K: int) -> Dict[str, any]:
        """
        Selecciona configuración óptima completa
        
        Strategy:
        1. Calcular tile basado en L1 (mejor localidad)
        2. Calcular tile basado en L2 (más capacidad)
        3. Considerar tamaño de matriz
        4. Seleccionar el mejor balance
        
        Args:
            M, N, K: Matrix dimensions
            
        Returns:
            Dict con configuración óptima:
            {
                'tile_size': int,
                'local_x': int,
                'local_y': int,
                'strategy': str,
                'expected_cache': str
            }
        """
        # Calcular candidatos
        tile_l1 = self.calculate_optimal_tile_l1(K)
        tile_l2 = self.calculate_optimal_tile_l2(M, N, K)
        
        # Heurística basada en tamaño de matriz
        matrix_size = max(M, N)
        
        if matrix_size <= 512:
            # Matrices pequeñas: priorizar L1 (mejor latencia)
            selected_tile = tile_l1
            strategy = "L1-optimized"
            cache_level = "L1"
            
        elif matrix_size <= 1536:
            # Matrices medianas: balance entre L1 y L2
            # Usar algo intermedio
            selected_tile = min(tile_l2, max(tile_l1, 16))
            strategy = "Balanced L1/L2"
            cache_level = "L1+L2"
            
        else:
            # Matrices grandes: priorizar tiles más grandes (menos tiles = menos overhead)
            selected_tile = tile_l2
            strategy = "L2-optimized"
            cache_level = "L2"
        
        # Ajustar a tiles disponibles
        if selected_tile not in self.available_tiles:
            # Encontrar el más cercano
            selected_tile = min(self.available_tiles, 
                              key=lambda x: abs(x - selected_tile))
        
        # Calcular work group óptimo
        local_x, local_y = self.calculate_optimal_tile_work_group(selected_tile)
        
        # Calcular memoria requerida
        memory_required = self.calculate_tile_memory(selected_tile, selected_tile, K)
        
        return {
            'tile_size': selected_tile,
            'local_x': local_x,
            'local_y': local_y,
            'strategy': strategy,
            'expected_cache': cache_level,
            'memory_required': memory_required,
            'memory_l1': self.l1_cache,
            'memory_l2': self.l2_cache,
            'fits_in_l1': memory_required <= self.l1_cache,
            'fits_in_l2': memory_required <= self.l2_cache,
        }
    
    def recommend_kernel(self, M: int, N: int, K: int) -> str:
        """
        Recomienda qué kernel usar basado en tamaño óptimo
        
        Args:
            M, N, K: Matrix dimensions
            
        Returns:
            Nombre del kernel recomendado
        """
        config = self.select_optimal_tile(M, N, K)
        tile = config['tile_size']
        
        # Mapear tile size a kernels disponibles
        if tile == 16:
            return "FLOAT4_VEC"  # Baseline production
        elif tile == 20:
            return "tile20_vectorized"  # Approach 2 v3
        elif tile <= 12:
            return "FLOAT4_VEC"  # Usar baseline para tiles pequeños
        else:
            # Para tiles que no tenemos, usar el más cercano
            if abs(tile - 16) < abs(tile - 20):
                return "FLOAT4_VEC"
            else:
                return "tile20_vectorized"


def demo_adaptive_tiling():
    """Demostración de adaptive tiling"""
    print("=" * 70)
    print("🔧 ADAPTIVE TILING DEMO")
    print("=" * 70)
    print()
    
    tiler = AdaptiveTiling()
    
    test_cases = [
        (512, 512, 512, "Small matrix"),
        (1024, 1024, 1024, "Medium matrix"),
        (2048, 2048, 2048, "Large matrix"),
        (4096, 4096, 4096, "Very large matrix"),
        (1536, 1536, 1536, "Odd size"),
    ]
    
    print("Matrix Size | Optimal Tile | Work Group | Strategy      | Cache | Kernel")
    print("-" * 70)
    
    for M, N, K, description in test_cases:
        config = tiler.select_optimal_tile(M, N, K)
        kernel = tiler.recommend_kernel(M, N, K)
        
        print(f"{M:4d}×{N:4d}  | {config['tile_size']:12d} | "
              f"{config['local_x']:2d}×{config['local_y']:2d}      | "
              f"{config['strategy']:13s} | {config['expected_cache']:5s} | "
              f"{kernel}")
    
    print()
    print("=" * 70)
    print()
    
    # Análisis detallado de un caso
    print("📊 DETAILED ANALYSIS for 1024×1024:")
    print()
    config = tiler.select_optimal_tile(1024, 1024, 1024)
    
    for key, value in config.items():
        if isinstance(value, bool):
            value = "✅ YES" if value else "❌ NO"
        print(f"  {key:20s}: {value}")
    
    print()
    print("💡 INSIGHTS:")
    print("  - Matrices pequeñas (512): Tile pequeño para L1")
    print("  - Matrices medianas (1024): Balance L1/L2")
    print("  - Matrices grandes (2048+): Tiles grandes para reducir overhead")
    print()


if __name__ == "__main__":
    demo_adaptive_tiling()
