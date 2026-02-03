/**
 * GEMM Kernels Optimizados para AMD Radeon RX 580/590 (Polaris 10)
 * 
 * Características del Hardware:
 * - 36 Compute Units
 * - 64KB Local Memory (LDS) por CU
 * - 256-bit GDDR5 memory bus
 * - GCN 4.0 architecture
 * - 4 SIMD units per CU (64 lanes each)
 * 
 * Optimizaciones implementadas:
 * 1. Vectorización SIMD con float4/float8
 * 2. Tiling óptimo para LDS (32x32 tiles)
 * 3. Bank conflict avoidance
 * 4. Memory coalescing
 * 5. Loop unrolling
 * 6. Prefetching
 * 7. Kernel fusion
 */

// ============================================================================
// CONSTANTES OPTIMIZADAS PARA RX 580
// ============================================================================

#define TILE_SIZE 32          // Óptimo para 64KB LDS
#define TILE_M 32
#define TILE_N 32  
#define TILE_K 16             // Para double buffering
#define VECTOR_SIZE 4         // float4 vectorization
#define WORK_PER_THREAD 8     // Work items process 8 elements
#define UNROLL_FACTOR 8       // Loop unroll factor
#define LDS_PADDING 1         // Bank conflict avoidance
#define NUM_BANKS 32          // LDS banks on GCN

// Build options recomendadas:
// -cl-mad-enable -cl-fast-relaxed-math -cl-unsafe-math-optimizations
// -cl-no-signed-zeros -cl-finite-math-only

// ============================================================================
// GEMM VECTORIZADO CON FLOAT4 - ALTA EFICIENCIA
// ============================================================================

/**
 * gemm_float4_optimized - GEMM con vectorización float4
 * 
 * C = alpha * A * B + beta * C
 * 
 * Optimizaciones:
 * - Carga vectorizada de 4 elementos simultáneos
 * - Tiles en LDS con padding anti bank-conflict
 * - Acceso coalescente a memoria global
 * - Loop unrolling de factor 4
 */
__kernel void gemm_float4_optimized(
    const int M, const int N, const int K,
    const float alpha, const float beta,
    __global const float4* restrict A,
    __global const float4* restrict B,
    __global float4* restrict C,
    __local float* restrict tile_A,    // TILE_SIZE x (TILE_K + LDS_PADDING)
    __local float* restrict tile_B     // TILE_K x (TILE_SIZE + LDS_PADDING)
) {
    // Identificadores de trabajo
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int tile_row = get_group_id(0) * TILE_SIZE;
    const int tile_col = get_group_id(1) * TILE_SIZE;
    
    // Stride con padding para evitar bank conflicts
    const int A_stride = TILE_K + LDS_PADDING;
    const int B_stride = TILE_SIZE + LDS_PADDING;
    
    // Acumuladores vectorizados (4 resultados por thread)
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);
    
    // Número de tiles en K
    const int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    // Procesar tiles
    for (int t = 0; t < num_tiles; t++) {
        const int k_offset = t * TILE_K;
        
        // Cargar tile de A en LDS (vectorizado)
        if (tile_row + local_row < M && k_offset + local_col < K) {
            #pragma unroll 4
            for (int i = 0; i < 4; i++) {
                int load_row = local_row * 4 + i;
                int load_col = local_col;
                if (tile_row + load_row < M && k_offset + load_col < K) {
                    int global_idx = (tile_row + load_row) * K + k_offset + load_col;
                    tile_A[load_row * A_stride + load_col] = ((const __global float*)A)[global_idx];
                }
            }
        }
        
        // Cargar tile de B en LDS (vectorizado)
        if (k_offset + local_row < K && tile_col + local_col < N) {
            #pragma unroll 4
            for (int i = 0; i < 4; i++) {
                int load_row = local_row;
                int load_col = local_col * 4 + i;
                if (k_offset + load_row < K && tile_col + load_col < N) {
                    int global_idx = (k_offset + load_row) * N + tile_col + load_col;
                    tile_B[load_row * B_stride + load_col] = ((const __global float*)B)[global_idx];
                }
            }
        }
        
        // Sincronizar antes de computar
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Multiplicación con loop unrolling
        #pragma unroll 4
        for (int k = 0; k < TILE_K; k++) {
            float4 a_vec;
            float4 b_vec;
            
            // Cargar valores de A
            a_vec.x = tile_A[(local_row * 4 + 0) * A_stride + k];
            a_vec.y = tile_A[(local_row * 4 + 1) * A_stride + k];
            a_vec.z = tile_A[(local_row * 4 + 2) * A_stride + k];
            a_vec.w = tile_A[(local_row * 4 + 3) * A_stride + k];
            
            // Cargar valores de B
            b_vec.x = tile_B[k * B_stride + local_col * 4 + 0];
            b_vec.y = tile_B[k * B_stride + local_col * 4 + 1];
            b_vec.z = tile_B[k * B_stride + local_col * 4 + 2];
            b_vec.w = tile_B[k * B_stride + local_col * 4 + 3];
            
            // MAD operations
            acc0 += a_vec.x * b_vec;
            acc1 += a_vec.y * b_vec;
            acc2 += a_vec.z * b_vec;
            acc3 += a_vec.w * b_vec;
        }
        
        // Sincronizar antes del siguiente tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Escribir resultados (con alpha/beta scaling)
    const int base_row = tile_row + local_row * 4;
    const int base_col = tile_col + local_col * 4;
    
    if (base_row + 0 < M && base_col < N) {
        int idx = (base_row + 0) * (N/4) + base_col/4;
        C[idx] = alpha * acc0 + beta * C[idx];
    }
    if (base_row + 1 < M && base_col < N) {
        int idx = (base_row + 1) * (N/4) + base_col/4;
        C[idx] = alpha * acc1 + beta * C[idx];
    }
    if (base_row + 2 < M && base_col < N) {
        int idx = (base_row + 2) * (N/4) + base_col/4;
        C[idx] = alpha * acc2 + beta * C[idx];
    }
    if (base_row + 3 < M && base_col < N) {
        int idx = (base_row + 3) * (N/4) + base_col/4;
        C[idx] = alpha * acc3 + beta * C[idx];
    }
}

// ============================================================================
// GEMM CON REGISTER TILING - MÁXIMO RENDIMIENTO
// ============================================================================

/**
 * gemm_register_tiled - GEMM con tiling a nivel de registros
 * 
 * Cada work-item computa un tile de WPT_REG x WPT_REG elementos
 * usando acumuladores en registros para máximo reuso.
 */
#ifndef WPT_REG
#define WPT_REG 8   // Work per thread (8x8 tile por thread)
#endif

__kernel void gemm_register_tiled(
    const int M, const int N, const int K,
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C
) {
    // Índices de trabajo
    const int tidm = get_local_id(0);    // Thread ID en M
    const int tidn = get_local_id(1);    // Thread ID en N
    const int gidm = get_group_id(0);    // Group ID en M
    const int gidn = get_group_id(1);    // Group ID en N
    
    // Tiles en LDS con padding
    __local float As[TILE_SIZE * (TILE_K + LDS_PADDING)];
    __local float Bs[TILE_K * (TILE_SIZE + LDS_PADDING)];
    
    // Registros acumuladores (8x8 = 64 valores)
    float acc[WPT_REG][WPT_REG];
    #pragma unroll
    for (int i = 0; i < WPT_REG; i++) {
        #pragma unroll
        for (int j = 0; j < WPT_REG; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Registros para valores cargados
    float regA[WPT_REG];
    float regB[WPT_REG];
    
    // Loop sobre tiles de K
    const int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        // Cargar tiles cooperativamente
        const int k_base = tile * TILE_K;
        
        // Cada thread carga múltiples elementos
        #pragma unroll
        for (int load = 0; load < (TILE_SIZE * TILE_K) / (TILE_SIZE * TILE_SIZE / WPT_REG / WPT_REG); load++) {
            int linear_id = tidm * (TILE_SIZE / WPT_REG) + tidn;
            int load_id = linear_id + load * (TILE_SIZE * TILE_SIZE / WPT_REG / WPT_REG);
            
            int a_row = load_id / TILE_K;
            int a_col = load_id % TILE_K;
            int global_a_row = gidm * TILE_SIZE + a_row;
            int global_a_col = k_base + a_col;
            
            if (global_a_row < M && global_a_col < K) {
                As[a_row * (TILE_K + LDS_PADDING) + a_col] = A[global_a_row * K + global_a_col];
            } else {
                As[a_row * (TILE_K + LDS_PADDING) + a_col] = 0.0f;
            }
            
            int b_row = load_id / TILE_SIZE;
            int b_col = load_id % TILE_SIZE;
            int global_b_row = k_base + b_row;
            int global_b_col = gidn * TILE_SIZE + b_col;
            
            if (global_b_row < K && global_b_col < N) {
                Bs[b_row * (TILE_SIZE + LDS_PADDING) + b_col] = B[global_b_row * N + global_b_col];
            } else {
                Bs[b_row * (TILE_SIZE + LDS_PADDING) + b_col] = 0.0f;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute sobre el tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // Cargar a registros
            #pragma unroll
            for (int w = 0; w < WPT_REG; w++) {
                regA[w] = As[(tidm * WPT_REG + w) * (TILE_K + LDS_PADDING) + k];
                regB[w] = Bs[k * (TILE_SIZE + LDS_PADDING) + tidn * WPT_REG + w];
            }
            
            // MAD en registros (rank-1 update)
            #pragma unroll
            for (int wm = 0; wm < WPT_REG; wm++) {
                #pragma unroll
                for (int wn = 0; wn < WPT_REG; wn++) {
                    acc[wm][wn] = mad(regA[wm], regB[wn], acc[wm][wn]);
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Escribir resultados
    const int base_m = gidm * TILE_SIZE + tidm * WPT_REG;
    const int base_n = gidn * TILE_SIZE + tidn * WPT_REG;
    
    #pragma unroll
    for (int wm = 0; wm < WPT_REG; wm++) {
        #pragma unroll
        for (int wn = 0; wn < WPT_REG; wn++) {
            if (base_m + wm < M && base_n + wn < N) {
                C[(base_m + wm) * N + base_n + wn] = acc[wm][wn];
            }
        }
    }
}

// ============================================================================
// KERNEL FUSION: TRANSPOSE + GEMM
// ============================================================================

/**
 * gemm_fused_transpose_b - GEMM con B transpuesta en el mismo kernel
 * 
 * C = A * B^T
 * 
 * Evita transferencia extra de la matriz transpuesta.
 */
__kernel void gemm_fused_transpose_b(
    const int M, const int N, const int K,
    __global const float* restrict A,
    __global const float* restrict B,  // Se lee como si estuviera transpuesta
    __global float* restrict C
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    
    __local float tile_A[TILE_SIZE][TILE_K + LDS_PADDING];
    __local float tile_B[TILE_SIZE][TILE_K + LDS_PADDING];  // B transpuesta
    
    float acc = 0.0f;
    
    const int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Cargar A normalmente
        int a_row = get_group_id(0) * TILE_SIZE + local_row;
        int a_col = t * TILE_K + local_col;
        if (a_row < M && a_col < K) {
            tile_A[local_row][local_col] = A[a_row * K + a_col];
        } else {
            tile_A[local_row][local_col] = 0.0f;
        }
        
        // Cargar B con acceso transpuesto (col-major → row-major)
        int b_row = get_group_id(1) * TILE_SIZE + local_row;  // Fila en B^T = Col en B
        int b_col = t * TILE_K + local_col;                    // Col en B^T = Fila en B
        if (b_row < N && b_col < K) {
            // B[k][n] leído como B^T[n][k]
            tile_B[local_row][local_col] = B[b_col * N + b_row];
        } else {
            tile_B[local_row][local_col] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute
        #pragma unroll 8
        for (int k = 0; k < TILE_K; k++) {
            acc = mad(tile_A[local_row][k], tile_B[local_col][k], acc);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ============================================================================
// KERNEL FUSION: GEMM + RELU + BIAS
// ============================================================================

/**
 * gemm_fused_relu_bias - GEMM fusionado con ReLU y bias
 * 
 * C = ReLU(A * B + bias)
 * 
 * Evita múltiples kernel launches y lecturas de memoria.
 */
__kernel void gemm_fused_relu_bias(
    const int M, const int N, const int K,
    __global const float* restrict A,
    __global const float* restrict B,
    __global const float* restrict bias,  // Vector de N elementos
    __global float* restrict C
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    
    __local float tile_A[TILE_SIZE][TILE_K + LDS_PADDING];
    __local float tile_B[TILE_K][TILE_SIZE + LDS_PADDING];
    
    // Precargar bias a registro (reutilizado por todas las filas)
    float bias_val = (col < N) ? bias[col] : 0.0f;
    
    float acc = 0.0f;
    
    const int num_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_tiles; t++) {
        // Cargar tiles
        int a_row = get_group_id(0) * TILE_SIZE + local_row;
        int a_col = t * TILE_K + local_col;
        tile_A[local_row][local_col] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        
        int b_row = t * TILE_K + local_row;
        int b_col = get_group_id(1) * TILE_SIZE + local_col;
        tile_B[local_row][local_col] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        #pragma unroll 8
        for (int k = 0; k < TILE_K; k++) {
            acc = mad(tile_A[local_row][k], tile_B[k][local_col], acc);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (row < M && col < N) {
        // Fusionar bias + ReLU
        float result = acc + bias_val;
        result = fmax(result, 0.0f);  // ReLU
        C[row * N + col] = result;
    }
}

// ============================================================================
// KERNEL FUSION: GEMM + SOFTMAX (Por filas)
// ============================================================================

/**
 * gemm_fused_softmax - GEMM fusionado con softmax por filas
 * 
 * C = softmax(A * B, dim=1)
 */
__kernel void gemm_fused_softmax(
    const int M, const int N, const int K,
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    __local float* scratch  // Para reducción de softmax
) {
    const int row = get_global_id(0);
    const int local_id = get_local_id(1);
    const int local_size = get_local_size(1);
    
    __local float row_max;
    __local float row_sum;
    
    // Primera pasada: encontrar máximo
    float my_max = -INFINITY;
    for (int col = local_id; col < N; col += local_size) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum = mad(A[row * K + k], B[k * N + col], sum);
        }
        scratch[local_id + (col / local_size) * local_size] = sum;
        my_max = fmax(my_max, sum);
    }
    
    // Reducción para máximo
    scratch[local_id] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            scratch[local_id] = fmax(scratch[local_id], scratch[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) row_max = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Segunda pasada: exp(x - max) y suma
    float my_sum = 0.0f;
    for (int col = local_id; col < N; col += local_size) {
        float val = scratch[local_id + (col / local_size) * local_size];
        val = exp(val - row_max);
        scratch[local_id + (col / local_size) * local_size] = val;
        my_sum += val;
    }
    
    // Reducción para suma
    scratch[local_id] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) row_sum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Normalizar y escribir
    for (int col = local_id; col < N; col += local_size) {
        if (row < M && col < N) {
            C[row * N + col] = scratch[local_id + (col / local_size) * local_size] / row_sum;
        }
    }
}

// ============================================================================
// GEMM ULTRA-OPTIMIZADO PARA RX 580
// ============================================================================

/**
 * gemm_rx580_ultra - Máximo rendimiento para RX 580
 * 
 * Combina todas las optimizaciones:
 * - Register tiling 8x8
 * - LDS double buffering
 * - Vectorización float4
 * - Prefetching
 * - Loop unrolling completo
 */
__kernel void gemm_rx580_ultra(
    const int M, const int N, const int K,
    __global const float4* restrict A,
    __global const float4* restrict B,
    __global float4* restrict C
) {
    // Configuración del thread
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);
    const int gidm = get_group_id(0);
    const int gidn = get_group_id(1);
    
    // Double buffer en LDS
    __local float4 As[2][TILE_SIZE/4][TILE_K/4 + 1];
    __local float4 Bs[2][TILE_K/4][TILE_SIZE/4 + 1];
    
    // 64 acumuladores (8x8) en registros
    float4 acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i] = (float4)(0.0f);
    }
    
    // Registros para prefetch
    float4 prefetch_a, prefetch_b;
    
    const int K_vec = K / 4;
    const int num_tiles = (K_vec + TILE_K/4 - 1) / (TILE_K/4);
    
    // Precargar primer tile
    int cur_buf = 0;
    int k_base = 0;
    
    // Cargar A
    if (gidm * TILE_SIZE/4 + tidm < M/4 && k_base + tidn < K_vec) {
        As[cur_buf][tidm][tidn] = A[(gidm * TILE_SIZE/4 + tidm) * K_vec + k_base + tidn];
    }
    
    // Cargar B
    if (k_base + tidm < K_vec && gidn * TILE_SIZE/4 + tidn < N/4) {
        Bs[cur_buf][tidm][tidn] = B[(k_base + tidm) * (N/4) + gidn * TILE_SIZE/4 + tidn];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Loop principal con double buffering
    for (int tile = 0; tile < num_tiles; tile++) {
        int next_buf = 1 - cur_buf;
        int next_k = (tile + 1) * (TILE_K/4);
        
        // Prefetch siguiente tile mientras se computa
        if (tile < num_tiles - 1) {
            if (gidm * TILE_SIZE/4 + tidm < M/4 && next_k + tidn < K_vec) {
                prefetch_a = A[(gidm * TILE_SIZE/4 + tidm) * K_vec + next_k + tidn];
            }
            if (next_k + tidm < K_vec && gidn * TILE_SIZE/4 + tidn < N/4) {
                prefetch_b = B[(next_k + tidm) * (N/4) + gidn * TILE_SIZE/4 + tidn];
            }
        }
        
        // Compute sobre tile actual
        #pragma unroll
        for (int k = 0; k < TILE_K/4; k++) {
            float4 a = As[cur_buf][tidm][k];
            float4 b = Bs[cur_buf][k][tidn];
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                float a_scalar = ((float*)&a)[i];
                acc[i*2] = mad(a_scalar, b, acc[i*2]);
                acc[i*2+1] = mad(a_scalar, b, acc[i*2+1]);
            }
        }
        
        // Escribir prefetch a siguiente buffer
        if (tile < num_tiles - 1) {
            barrier(CLK_LOCAL_MEM_FENCE);
            As[next_buf][tidm][tidn] = prefetch_a;
            Bs[next_buf][tidm][tidn] = prefetch_b;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        cur_buf = next_buf;
    }
    
    // Escribir resultados
    const int base_m = gidm * TILE_SIZE/4 + tidm;
    const int base_n = gidn * TILE_SIZE/4 + tidn;
    
    if (base_m < M/4 && base_n < N/4) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            C[(base_m * 8 + i) * (N/4) + base_n] = acc[i];
        }
    }
}

// ============================================================================
// KERNEL AUXILIAR: TRANSPOSE COALESCENTE
// ============================================================================

/**
 * matrix_transpose_optimized - Transposición con acceso coalescente
 * 
 * Utiliza LDS para convertir acceso strided a coalescente.
 */
__kernel void matrix_transpose_optimized(
    const int rows, const int cols,
    __global const float* restrict input,
    __global float* restrict output
) {
    // Tile con padding para evitar bank conflicts
    __local float tile[TILE_SIZE][TILE_SIZE + LDS_PADDING];
    
    const int x_in = get_group_id(0) * TILE_SIZE + get_local_id(0);
    const int y_in = get_group_id(1) * TILE_SIZE + get_local_id(1);
    
    // Cargar con acceso coalescente
    if (y_in < rows && x_in < cols) {
        tile[get_local_id(1)][get_local_id(0)] = input[y_in * cols + x_in];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Escribir transpuesto (intercambiar índices de bloque)
    const int x_out = get_group_id(1) * TILE_SIZE + get_local_id(0);
    const int y_out = get_group_id(0) * TILE_SIZE + get_local_id(1);
    
    if (x_out < rows && y_out < cols) {
        output[y_out * rows + x_out] = tile[get_local_id(0)][get_local_id(1)];
    }
}

// ============================================================================
// KERNEL DE PRUEBA: BENCHMARK MEMORY BANDWIDTH
// ============================================================================

/**
 * memory_bandwidth_test - Para medir ancho de banda de memoria
 */
__kernel void memory_bandwidth_test(
    __global const float4* restrict input,
    __global float4* restrict output,
    const int N
) {
    const int gid = get_global_id(0);
    if (gid < N) {
        output[gid] = input[gid];
    }
}

/**
 * compute_intensity_test - Para medir FLOPS máximos
 */
__kernel void compute_intensity_test(
    __global float4* data,
    const int N,
    const int iterations
) {
    const int gid = get_global_id(0);
    if (gid >= N) return;
    
    float4 val = data[gid];
    
    #pragma unroll 16
    for (int i = 0; i < iterations; i++) {
        val = mad(val, val, val);
        val = mad(val, val, val);
        val = mad(val, val, val);
        val = mad(val, val, val);
    }
    
    data[gid] = val;
}
