/*
 * Approach 1: Cooperative Loading with Fixed Indexing
 * 
 * Strategy:
 *   - Use 256 threads (16×16 local_size) to cooperatively load 400 elements (20×20 tile)
 *   - Separate loading phase from compute phase with barrier
 *   - Use modulo arithmetic for proper indexing in compute loop
 * 
 * Key Idea:
 *   Each thread loads ~1.56 elements (400/256 ≈ 1.56)
 *   Use linear indexing: thread_id loads elements at positions: thread_id, thread_id+256
 * 
 * Author: Tile=20 Research Branch
 * Date: February 2026
 * Status: EXPERIMENTAL
 */

#define TILE_SIZE 20
#define LOCAL_X 16
#define LOCAL_Y 16
#define TOTAL_THREADS (LOCAL_X * LOCAL_Y)  // 256
#define TILE_ELEMENTS (TILE_SIZE * TILE_SIZE)  // 400

__kernel void gemm_tile20_cooperative(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    // Local memory tiles
    __local float As[TILE_SIZE * TILE_SIZE];  // 20×20 = 400 floats
    __local float Bs[TILE_SIZE * TILE_SIZE * 4];  // 20×20×4 = 1600 floats (for float4)
    
    // Work-item indices
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int local_linear = local_y * LOCAL_X + local_x;  // 0-255
    
    // Global indices - each work-item processes 4 columns
    const int global_row = get_global_id(0);
    const int global_col_base = get_global_id(1) * 4;
    
    // Accumulator for 4 columns
    float4 acc = (float4)(0.0f);
    
    // Number of tiles in K dimension
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over tiles
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        const int tile_k_start = tile_idx * TILE_SIZE;
        
        // ========================================
        // PHASE 1: COOPERATIVE LOADING
        // ========================================
        
        // Load A tile cooperatively (20×20 = 400 elements, 256 threads)
        // Each thread loads: floor(400/256) = 1 element, first 144 threads load 2 elements
        const int a_loads = (TILE_ELEMENTS + TOTAL_THREADS - 1) / TOTAL_THREADS;
        for (int load = 0; load < a_loads; load++) {
            const int elem_idx = local_linear + load * TOTAL_THREADS;
            if (elem_idx < TILE_ELEMENTS) {
                const int tile_row = elem_idx / TILE_SIZE;
                const int tile_col = elem_idx % TILE_SIZE;
                const int global_row_a = global_row / LOCAL_X * TILE_SIZE + tile_row;
                const int global_col_a = tile_k_start + tile_col;
                
                if (global_row_a < M && global_col_a < K) {
                    As[elem_idx] = A[global_row_a * K + global_col_a];
                } else {
                    As[elem_idx] = 0.0f;
                }
            }
        }
        
        // Load B tile cooperatively (20×20×4 = 1600 elements, 256 threads)
        // Each thread loads: ceil(1600/256) = 7 elements
        const int b_elements = TILE_ELEMENTS * 4;
        const int b_loads = (b_elements + TOTAL_THREADS - 1) / TOTAL_THREADS;
        for (int load = 0; load < b_loads; load++) {
            const int elem_idx = local_linear + load * TOTAL_THREADS;
            if (elem_idx < b_elements) {
                const int tile_row = elem_idx / (TILE_SIZE * 4);
                const int tile_col_vec = (elem_idx % (TILE_SIZE * 4)) / 4;
                const int vec_component = elem_idx % 4;
                
                const int global_row_b = tile_k_start + tile_row;
                const int global_col_b = global_col_base / 4 * TILE_SIZE + tile_col_vec;
                const int global_col_b_actual = global_col_b * 4 + vec_component;
                
                if (global_row_b < K && global_col_b_actual < N) {
                    Bs[elem_idx] = B[global_row_b * N + global_col_b_actual];
                } else {
                    Bs[elem_idx] = 0.0f;
                }
            }
        }
        
        // Synchronize: wait for all threads to finish loading
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // ========================================
        // PHASE 2: COMPUTE WITH PROPER INDEXING
        // ========================================
        
        // Only threads within the compute region participate
        // For 20×20 tiles with 16×16 threads, we need to map properly
        
        // Each thread computes its contribution based on local_x and local_y
        // We'll compute multiple elements per thread if needed
        
        // Strategy: Use modulo to wrap indices
        #pragma unroll 4
        for (int k = 0; k < TILE_SIZE; k++) {
            // A element: from row local_x, column k
            float a_val;
            if (local_x < TILE_SIZE && local_y < TILE_SIZE) {
                a_val = As[local_x * TILE_SIZE + k];
            } else {
                a_val = 0.0f;
            }
            
            // B elements: from row k, columns [local_y*4 : local_y*4+3]
            float4 b_val;
            if (local_y < TILE_SIZE) {
                const int b_offset = k * TILE_SIZE * 4 + local_y * 4;
                b_val = vload4(0, &Bs[b_offset]);
            } else {
                b_val = (float4)(0.0f);
            }
            
            // Accumulate
            acc += a_val * b_val;
        }
        
        // Synchronize before next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // ========================================
    // PHASE 3: WRITE RESULTS
    // ========================================
    
    // Only write if within bounds and within active region
    if (global_row < M && global_col_base < N && local_x < TILE_SIZE && local_y < TILE_SIZE) {
        // Apply alpha and beta
        if (beta == 0.0f) {
            vstore4(alpha * acc, 0, &C[global_row * N + global_col_base]);
        } else {
            float4 c_val = vload4(0, &C[global_row * N + global_col_base]);
            vstore4(alpha * acc + beta * c_val, 0, &C[global_row * N + global_col_base]);
        }
    }
}
