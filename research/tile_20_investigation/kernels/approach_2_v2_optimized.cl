/*
 * Approach 2 v2: Optimized Occupancy (20×10 threads)
 * 
 * Improvement over v1 (10×10):
 *   - v1: 100 threads × 4 outputs = 400 (CORRECT but slow)
 *   - v2: 200 threads × 2 outputs = 400 (CORRECT and faster!)
 * 
 * Strategy:
 *   - 20×10 = 200 threads (2× more than v1)
 *   - Each thread computes 1 row × 2 columns
 *   - Clean mapping: 200 threads × 2 outputs = 400 total ✅
 *   - Better GPU occupancy = higher performance
 * 
 * Expected Performance: 600-750 GFLOPS
 * 
 * Author: Tile=20 Research Branch
 * Date: February 2026
 * Status: EXPERIMENTAL - Approach 2 v2 (Optimized)
 */

#define TILE_SIZE 20
#define LOCAL_X 20  // 20 threads in X (rows)
#define LOCAL_Y 10  // 10 threads in Y (cols)
#define TOTAL_THREADS (LOCAL_X * LOCAL_Y)  // 200

__kernel void gemm_tile20_optimized(
    const int M,
    const int N,
    const int K,
    const float alpha,
    __global const float* A,
    __global const float* B,
    const float beta,
    __global float* C
) {
    // Local memory for 20×20 tiles
    __local float As[TILE_SIZE * TILE_SIZE];  // 20×20 = 400 floats
    __local float Bs[TILE_SIZE * TILE_SIZE];  // 20×20 = 400 floats
    
    // Thread IDs
    const int local_x = get_local_id(0);  // 0-19
    const int local_y = get_local_id(1);  // 0-9
    const int local_linear = local_y * LOCAL_X + local_x;  // 0-199
    
    // Group IDs
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    
    // Each thread computes 1 row × 2 columns
    // Thread (local_x, local_y) computes outputs at:
    //   Row: local_x (0-19)
    //   Cols: local_y*2 and local_y*2+1 (0-19)
    float acc0 = 0.0f;  // Column 0 (local_y * 2)
    float acc1 = 0.0f;  // Column 1 (local_y * 2 + 1)
    
    // Number of tiles in K
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over K tiles
    for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
        
        // ========================================
        // COOPERATIVE LOADING
        // ========================================
        
        // Load A: 400 elements with 200 threads → each loads 2
        for (int i = local_linear; i < TILE_SIZE * TILE_SIZE; i += TOTAL_THREADS) {
            const int tile_row = i / TILE_SIZE;
            const int tile_col = i % TILE_SIZE;
            const int global_row = group_x * TILE_SIZE + tile_row;
            const int global_col = tile_k * TILE_SIZE + tile_col;
            
            if (global_row < M && global_col < K) {
                As[i] = A[global_row * K + global_col];
            } else {
                As[i] = 0.0f;
            }
        }
        
        // Load B: 400 elements with 200 threads → each loads 2
        for (int i = local_linear; i < TILE_SIZE * TILE_SIZE; i += TOTAL_THREADS) {
            const int tile_row = i / TILE_SIZE;
            const int tile_col = i % TILE_SIZE;
            const int global_row = tile_k * TILE_SIZE + tile_row;
            const int global_col = group_y * TILE_SIZE + tile_col;
            
            if (global_row < K && global_col < N) {
                Bs[i] = B[global_row * N + global_col];
            } else {
                Bs[i] = 0.0f;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // ========================================
        // COMPUTE: Each thread does 1 row × 2 cols
        // ========================================
        
        const int out_row = local_x;        // 0-19 (this thread's row)
        const int out_col0 = local_y * 2;   // 0,2,4,...,18 (first column)
        const int out_col1 = local_y * 2 + 1; // 1,3,5,...,19 (second column)
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            const float a_val = As[out_row * TILE_SIZE + k];
            const float b_val0 = Bs[k * TILE_SIZE + out_col0];
            const float b_val1 = Bs[k * TILE_SIZE + out_col1];
            
            acc0 += a_val * b_val0;
            acc1 += a_val * b_val1;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // ========================================
    // WRITE RESULTS: 1 row × 2 cols per thread
    // ========================================
    
    const int global_row = group_x * TILE_SIZE + local_x;
    const int global_col0 = group_y * TILE_SIZE + local_y * 2;
    const int global_col1 = group_y * TILE_SIZE + local_y * 2 + 1;
    
    // Write first column
    if (global_row < M && global_col0 < N) {
        const int idx = global_row * N + global_col0;
        if (beta == 0.0f) {
            C[idx] = alpha * acc0;
        } else {
            C[idx] = alpha * acc0 + beta * C[idx];
        }
    }
    
    // Write second column
    if (global_row < M && global_col1 < N) {
        const int idx = global_row * N + global_col1;
        if (beta == 0.0f) {
            C[idx] = alpha * acc1;
        } else {
            C[idx] = alpha * acc1 + beta * C[idx];
        }
    }
}
