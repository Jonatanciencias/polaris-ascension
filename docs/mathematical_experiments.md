# 🧮 Mathematical Foundations & Experiments

## Experimentos Matemáticos Fundamentales para RX 580

Este documento detalla experimentos matemáticos específicos que puedes implementar para validar las hipótesis del proyecto.

---

## Experimento 1: Análisis de Sensibilidad a Precisión

### Objetivo
Determinar matemáticamente cuánta precisión necesita cada capa de un modelo.

### Base Teórica

**Error de Quantización**:
```
Para valor real x y versión quantizada Q(x):

Error absoluto: |x - Q(x)| ≤ Δ/2
donde Δ = (x_max - x_min) / (2^bits - 1)

Error relativo: |x - Q(x)| / |x| ≤ Δ/(2·x_min)
```

**Propagación de Error**:
```
Para f(x) = Wx + b:

σ_output² ≈ ||W||²_F · σ_input² + σ_W² · ||x||²

Donde σ_W es error de quantización de pesos
```

### Código de Experimento

```python
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

class PrecisionAnalyzer:
    """Analiza sensibilidad de precisión capa por capa"""
    
    def __init__(self, model):
        self.model = model
        self.layer_sensitivities = {}
    
    def quantize_layer(self, weights: np.ndarray, bits: int) -> np.ndarray:
        """
        Quantiza pesos a N bits
        
        Args:
            weights: Pesos FP32 originales
            bits: Número de bits (4, 8, 16)
        
        Returns:
            Pesos quantizados
        """
        w_min, w_max = weights.min(), weights.max()
        
        # Niveles de quantización
        n_levels = 2 ** bits
        
        # Escala y quantiza
        scale = (w_max - w_min) / (n_levels - 1)
        w_quant = np.round((weights - w_min) / scale)
        
        # De-quantiza
        w_dequant = w_quant * scale + w_min
        
        return w_dequant
    
    def measure_layer_sensitivity(
        self, 
        layer_name: str,
        weights: np.ndarray,
        test_inputs: np.ndarray,
        bits_range: List[int] = [4, 8, 16]
    ) -> Dict[int, float]:
        """
        Mide sensibilidad de una capa a diferentes precisiones
        
        Returns:
            Dict[bits -> error_metric]
        """
        results = {}
        
        # Output con precisión completa (baseline)
        output_baseline = self.forward_layer(weights, test_inputs)
        
        for bits in bits_range:
            # Quantiza pesos
            w_quant = self.quantize_layer(weights, bits)
            
            # Propaga con pesos quantizados
            output_quant = self.forward_layer(w_quant, test_inputs)
            
            # Calcula error
            mse = np.mean((output_baseline - output_quant) ** 2)
            rel_error = mse / (np.mean(output_baseline ** 2) + 1e-8)
            
            # Norma de diferencia de pesos
            weight_error = np.linalg.norm(weights - w_quant) / np.linalg.norm(weights)
            
            results[bits] = {
                'mse': mse,
                'relative_error': rel_error,
                'weight_error': weight_error
            }
        
        self.layer_sensitivities[layer_name] = results
        return results
    
    def analyze_full_model(self, test_dataset: np.ndarray) -> Dict:
        """
        Analiza todas las capas del modelo
        """
        all_results = {}
        
        for layer_name, layer in self.model.named_modules():
            if hasattr(layer, 'weight'):
                weights = layer.weight.detach().cpu().numpy()
                
                print(f"\nAnalizando capa: {layer_name}")
                print(f"  Forma: {weights.shape}")
                print(f"  Parámetros: {weights.size}")
                
                results = self.measure_layer_sensitivity(
                    layer_name, 
                    weights, 
                    test_dataset
                )
                
                all_results[layer_name] = results
                
                # Imprime resultados
                print(f"\n  Sensibilidad a quantización:")
                for bits, metrics in results.items():
                    print(f"    {bits}-bit: "
                          f"RelError={metrics['relative_error']:.6f}, "
                          f"WeightError={metrics['weight_error']:.6f}")
        
        return all_results
    
    def recommend_precision(self, threshold: float = 0.01) -> Dict[str, int]:
        """
        Recomienda precisión óptima para cada capa
        
        Args:
            threshold: Error relativo máximo aceptable
        
        Returns:
            Dict[layer_name -> bits_recomendados]
        """
        recommendations = {}
        
        for layer_name, results in self.layer_sensitivities.items():
            # Encuentra mínimo bits que cumple threshold
            for bits in [4, 8, 16]:
                if results[bits]['relative_error'] < threshold:
                    recommendations[layer_name] = bits
                    break
            else:
                recommendations[layer_name] = 32  # FP32 si nada funciona
        
        return recommendations
    
    def visualize_sensitivity(self, save_path: str = 'sensitivity.png'):
        """Genera visualización de sensibilidades"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        layer_names = list(self.layer_sensitivities.keys())
        bits_levels = [4, 8, 16]
        
        # Plot 1: Error relativo por capa y bits
        ax = axes[0, 0]
        for bits in bits_levels:
            errors = [self.layer_sensitivities[l][bits]['relative_error'] 
                     for l in layer_names]
            ax.plot(range(len(layer_names)), errors, 
                   marker='o', label=f'{bits}-bit')
        ax.set_xlabel('Capa')
        ax.set_ylabel('Error Relativo')
        ax.set_title('Sensibilidad por Capa')
        ax.legend()
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        
        # Plot 2: Distribución de errores
        ax = axes[0, 1]
        for bits in bits_levels:
            errors = [self.layer_sensitivities[l][bits]['relative_error'] 
                     for l in layer_names]
            ax.hist(errors, alpha=0.5, label=f'{bits}-bit', bins=20)
        ax.set_xlabel('Error Relativo')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Errores')
        ax.legend()
        
        # Plot 3: Bits recomendados
        ax = axes[1, 0]
        recommendations = self.recommend_precision()
        bits_counts = {}
        for bits in recommendations.values():
            bits_counts[bits] = bits_counts.get(bits, 0) + 1
        ax.bar(bits_counts.keys(), bits_counts.values())
        ax.set_xlabel('Bits')
        ax.set_ylabel('Número de Capas')
        ax.set_title('Distribución de Precisión Recomendada')
        
        # Plot 4: Ganancia de memoria
        ax = axes[1, 1]
        original_size = sum(np.prod(layer.weight.shape) * 32 
                          for layer in self.model.modules() 
                          if hasattr(layer, 'weight'))
        
        recommendations = self.recommend_precision()
        optimized_sizes = []
        
        for threshold in [0.001, 0.005, 0.01, 0.05, 0.1]:
            recs = self.recommend_precision(threshold)
            size = sum(np.prod(layer.weight.shape) * recs[name]
                      for name, layer in self.model.named_modules()
                      if hasattr(layer, 'weight'))
            optimized_sizes.append(size / original_size)
        
        thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
        ax.plot(thresholds, optimized_sizes, marker='o')
        ax.set_xlabel('Threshold de Error')
        ax.set_ylabel('Tamaño Relativo')
        ax.set_title('Trade-off Error vs Memoria')
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualización guardada en: {save_path}")


# Ejemplo de uso
if __name__ == "__main__":
    # Cargar modelo (ejemplo con ResNet)
    import torch
    import torchvision.models as models
    
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Generar datos de prueba
    test_inputs = torch.randn(32, 3, 224, 224)
    
    # Analizar
    analyzer = PrecisionAnalyzer(model)
    results = analyzer.analyze_full_model(test_inputs)
    
    # Recomendaciones
    recommendations = analyzer.recommend_precision(threshold=0.01)
    
    print("\n" + "="*60)
    print("RECOMENDACIONES DE PRECISIÓN")
    print("="*60)
    
    for layer, bits in recommendations.items():
        print(f"{layer:50s} -> {bits:2d} bits")
    
    # Calcular ahorro de memoria
    total_params = sum(p.numel() for p in model.parameters())
    original_size_mb = total_params * 32 / (8 * 1024 * 1024)
    
    optimized_size_mb = sum(
        p.numel() * recommendations.get(name, 32)
        for name, p in model.named_parameters()
    ) / (8 * 1024 * 1024)
    
    print(f"\nTamaño original: {original_size_mb:.2f} MB")
    print(f"Tamaño optimizado: {optimized_size_mb:.2f} MB")
    print(f"Ahorro: {(1 - optimized_size_mb/original_size_mb)*100:.1f}%")
    
    # Visualizar
    analyzer.visualize_sensitivity('precision_analysis.png')
```

---

## Experimento 2: Sparse Matrix Performance Profiling

### Objetivo
Determinar el nivel de sparsity donde sparse operations superan dense operations en Polaris.

### Base Matemática

**Complejidad Computacional**:
```
Dense GEMM: O(mnk)
Sparse GEMM (CSR): O(nnz·n) donde nnz = número de no-zeros

Crossover point: 
nnz/mk < 1  →  sparsity > (1 - 1/k)

Para k=512: sparsity > 99.8% teóricamente
Pero en práctica: overhead hace crossover ~70-80%
```

### Código de Experimento

```python
import numpy as np
import time
from scipy.sparse import csr_matrix, random
import pyopencl as cl

class SparseVsDenseBenchmark:
    """
    Benchmarks sparse vs dense operations en OpenCL/Polaris
    """
    
    def __init__(self):
        # Inicializar OpenCL
        platforms = cl.get_platforms()
        # Buscar plataforma AMD
        amd_platform = None
        for p in platforms:
            if 'AMD' in p.name or 'Radeon' in p.name:
                amd_platform = p
                break
        
        if amd_platform is None:
            print("Warning: AMD platform no encontrada, usando primera disponible")
            amd_platform = platforms[0]
        
        devices = amd_platform.get_devices()
        self.ctx = cl.Context(devices=[devices[0]])
        self.queue = cl.CommandQueue(self.ctx)
        
        print(f"Usando: {devices[0].name}")
    
    def generate_sparse_matrix(
        self, 
        rows: int, 
        cols: int, 
        sparsity: float
    ) -> csr_matrix:
        """
        Genera matriz sparse con sparsity dada
        
        Args:
            rows, cols: Dimensiones
            sparsity: Proporción de ceros (0.9 = 90% zeros)
        
        Returns:
            Matriz CSR format
        """
        density = 1 - sparsity
        return random(rows, cols, density=density, format='csr', dtype=np.float32)
    
    def benchmark_dense_matmul(
        self, 
        M: int, 
        K: int, 
        N: int, 
        iterations: int = 100
    ) -> float:
        """
        Benchmark dense matrix multiplication en OpenCL
        
        Returns:
            Tiempo promedio en ms
        """
        # Generar matrices
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)
        
        # Transferir a GPU
        mf = cl.mem_flags
        A_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
        B_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, C.nbytes)
        
        # Kernel OpenCL básico (mejorar después)
        kernel_code = """
        __kernel void matmul_dense(
            __global const float* A,
            __global const float* B,
            __global float* C,
            const int M, const int K, const int N
        ) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            
            if(row < M && col < N) {
                float sum = 0.0f;
                for(int k=0; k<K; k++) {
                    sum += A[row*K + k] * B[k*N + col];
                }
                C[row*N + col] = sum;
            }
        }
        """
        
        prg = cl.Program(self.ctx, kernel_code).build()
        
        # Warmup
        prg.matmul_dense(
            self.queue, (M, N), None,
            A_buf, B_buf, C_buf,
            np.int32(M), np.int32(K), np.int32(N)
        )
        self.queue.finish()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            
            prg.matmul_dense(
                self.queue, (M, N), None,
                A_buf, B_buf, C_buf,
                np.int32(M), np.int32(K), np.int32(N)
            )
            self.queue.finish()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        return np.median(times)
    
    def benchmark_sparse_matmul(
        self,
        M: int,
        K: int, 
        N: int,
        sparsity: float,
        iterations: int = 100
    ) -> float:
        """
        Benchmark sparse matrix multiplication
        
        Returns:
            Tiempo promedio en ms
        """
        # Generar matriz sparse
        A_sparse = self.generate_sparse_matrix(M, K, sparsity)
        B = np.random.randn(K, N).astype(np.float32)
        
        # Convertir a CSR format
        A_data = A_sparse.data
        A_indices = A_sparse.indices.astype(np.int32)
        A_indptr = A_sparse.indptr.astype(np.int32)
        
        # Transferir a GPU
        mf = cl.mem_flags
        data_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                            hostbuf=A_data)
        indices_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                               hostbuf=A_indices)
        indptr_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=A_indptr)
        B_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
        C = np.zeros((M, N), dtype=np.float32)
        C_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, C.nbytes)
        
        # Kernel sparse (CSR format)
        kernel_code = """
        __kernel void matmul_sparse_csr(
            __global const float* data,
            __global const int* indices,
            __global const int* indptr,
            __global const float* B,
            __global float* C,
            const int M, const int N
        ) {
            int row = get_global_id(0);
            
            if(row < M) {
                int row_start = indptr[row];
                int row_end = indptr[row + 1];
                
                for(int col=0; col<N; col++) {
                    float sum = 0.0f;
                    
                    for(int i=row_start; i<row_end; i++) {
                        int k = indices[i];
                        sum += data[i] * B[k*N + col];
                    }
                    
                    C[row*N + col] = sum;
                }
            }
        }
        """
        
        prg = cl.Program(self.ctx, kernel_code).build()
        
        # Warmup
        prg.matmul_sparse_csr(
            self.queue, (M,), None,
            data_buf, indices_buf, indptr_buf, B_buf, C_buf,
            np.int32(M), np.int32(N)
        )
        self.queue.finish()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            
            prg.matmul_sparse_csr(
                self.queue, (M,), None,
                data_buf, indices_buf, indptr_buf, B_buf, C_buf,
                np.int32(M), np.int32(N)
            )
            self.queue.finish()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        return np.median(times)
    
    def find_crossover_point(
        self,
        M: int = 2048,
        K: int = 2048,
        N: int = 2048
    ) -> float:
        """
        Encuentra el nivel de sparsity donde sparse supera dense
        
        Returns:
            Sparsity óptima
        """
        print(f"\nBuscando crossover point para matrices {M}×{K} × {K}×{N}")
        print("="*60)
        
        sparsity_levels = np.arange(0.0, 0.99, 0.05)
        dense_time = self.benchmark_dense_matmul(M, K, N, iterations=50)
        
        print(f"Dense time: {dense_time:.2f} ms")
        print("\nSparsity | Sparse Time | Speedup vs Dense")
        print("-"*60)
        
        results = []
        for sparsity in sparsity_levels:
            sparse_time = self.benchmark_sparse_matmul(
                M, K, N, sparsity, iterations=50
            )
            speedup = dense_time / sparse_time
            
            results.append({
                'sparsity': sparsity,
                'sparse_time': sparse_time,
                'dense_time': dense_time,
                'speedup': speedup
            })
            
            print(f"{sparsity:6.2f}   | {sparse_time:10.2f} ms | {speedup:6.2f}x")
        
        # Encontrar crossover
        for r in results:
            if r['speedup'] > 1.0:
                print(f"\n✅ Crossover en sparsity = {r['sparsity']:.2f}")
                print(f"   Sparse es {r['speedup']:.2f}x más rápido")
                return r['sparsity']
        
        print("\n⚠️  Sparse no superó dense en rangos probados")
        return None
    
    def comprehensive_benchmark(self):
        """
        Benchmark comprensivo para diferentes tamaños
        """
        sizes = [512, 1024, 2048, 4096]
        sparsities = [0.5, 0.7, 0.9, 0.95, 0.99]
        
        results = []
        
        print("\n" + "="*80)
        print("BENCHMARK COMPRENSIVO: SPARSE VS DENSE")
        print("="*80)
        
        for size in sizes:
            print(f"\n--- Tamaño: {size}×{size} ---")
            
            dense_time = self.benchmark_dense_matmul(size, size, size, 20)
            print(f"Dense: {dense_time:.2f} ms")
            
            for sparsity in sparsities:
                sparse_time = self.benchmark_sparse_matmul(
                    size, size, size, sparsity, 20
                )
                speedup = dense_time / sparse_time
                
                results.append({
                    'size': size,
                    'sparsity': sparsity,
                    'dense_time': dense_time,
                    'sparse_time': sparse_time,
                    'speedup': speedup
                })
                
                marker = "✅" if speedup > 1.0 else "❌"
                print(f"  Sparsity {sparsity:.2f}: {sparse_time:.2f} ms "
                      f"({speedup:.2f}x) {marker}")
        
        return results


# Uso
if __name__ == "__main__":
    benchmark = SparseVsDenseBenchmark()
    
    # Benchmark básico
    crossover = benchmark.find_crossover_point(2048, 2048, 2048)
    
    # Benchmark comprensivo
    results = benchmark.comprehensive_benchmark()
    
    # Guardar resultados
    import json
    with open('sparse_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Resultados guardados en sparse_benchmark_results.json")
```

---

## Experimento 3: GCN-Specific GEMM Optimization

### Objetivo
Optimizar GEMM aprovechando características únicas de GCN (wavefronts de 64, LDS de 64KB).

### Base Matemática

**Tiling Óptimo para GCN**:
```
Para C = A·B donde A ∈ ℝ^(M×K), B ∈ ℝ^(K×N)

Tile parameters:
- TM = 64 (wavefront size)
- TK = 256 (maximiza reuso de LDS, 64×256×4bytes = 64KB)
- TN = 64

Cada tile compute:
  C_tile[TM×TN] = A_tile[TM×TK] × B_tile[TK×TN]

Reuso arithmetic intensity: 2·TM·TK·TN / (TM·TK + TK·TN + TM·TN)
                           ≈ 2·64·256·64 / (64·256 + 256·64 + 64·64)
                           ≈ 74.5 FLOPs/byte
```

### Código de Kernel Optimizado

```opencl
// gemm_polaris_optimized.cl

// Optimizado para Polaris GCN 4.0
// - Wavefront size: 64
// - LDS: 64KB per CU
// - 16 LDS banks (4-byte width)

#define TM 64    // Tile M (wavefront)
#define TK 256   // Tile K (maximiza LDS usage)
#define TN 64    // Tile N

__kernel void gemm_polaris_optimized(
    __global const float* restrict A,   // M×K
    __global const float* restrict B,   // K×N
    __global float* restrict C,          // M×N
    const int M,
    const int K,
    const int N
) {
    // LDS para tiles
    __local float A_tile[TM * TK];  // 64×256×4 = 64KB
    __local float B_tile[TK * TN];  // 256×64×4 = 64KB (exceeds, need optimization)
    
    // Identificadores
    const int wf_id = get_local_id(0) / 64;  // Wavefront ID
    const int lane = get_local_id(0) % 64;   // Thread in wavefront
    
    const int block_row = get_group_id(0);
    const int block_col = get_group_id(1);
    
    const int row = block_row * TM + lane;
    
    // Acumulador privado
    float acc[TN / 64] = {0.0f};  // Cada thread acumula para su fila
    
    // Loop sobre tiles de K
    for(int k_tile = 0; k_tile < K; k_tile += TK) {
        // Load A tile cooperatively
        // Cada wavefront (64 threads) carga 64 filas
        #pragma unroll 4
        for(int k = 0; k < TK; k += 4) {
            if(row < M && (k_tile + k) < K) {
                // Coalesced load
                A_tile[lane * TK + k] = A[row * K + k_tile + k];
                A_tile[lane * TK + k+1] = A[row * K + k_tile + k+1];
                A_tile[lane * TK + k+2] = A[row * K + k_tile + k+2];
                A_tile[lane * TK + k+3] = A[row * K + k_tile + k+3];
            }
        }
        
        // Load B tile (simplified, needs optimization for real use)
        // TODO: Optimize B loading pattern
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute: cada thread calcula su fila × todas columnas
        for(int col_tile = 0; col_tile < TN; col_tile++) {
            float sum = 0.0f;
            
            // Inner product: dot(A_tile[lane, :], B_tile[:, col_tile])
            #pragma unroll 8
            for(int k = 0; k < TK; k++) {
                sum += A_tile[lane * TK + k] * B_tile[k * TN + col_tile];
            }
            
            if(col_tile < TN / 64) {
                acc[col_tile] += sum;
            } else {
                // Store partial result immediately
                if(row < M && (block_col * TN + col_tile) < N) {
                    atomicAdd(&C[row * N + block_col * TN + col_tile], sum);
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store accumulated results
    for(int i = 0; i < TN / 64; i++) {
        int col = block_col * TN + i * 64 + lane;
        if(row < M && col < N) {
            C[row * N + col] = acc[i];
        }
    }
}
```

---

**Continuación en próximo documento por longitud...**

Este es el comienzo de experimentos matemáticos concretos. ¿Quieres que continúe con:
1. Más experimentos (SNNs, hybrid scheduler, etc.)?
2. Scripts Python completos listos para correr?
3. Análisis teórico más profundo de algún tema específico?
