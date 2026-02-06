/*
 * Approach 2: Non-Square Tiles (16×20)
 * 
 * INSIGHT: 256 threads CANNOT cleanly map to 400 outputs (20×20)
 * 
 * Solution: Use 16×20 tiles instead!
 *   - 16 rows × 20 columns (vectorized: 16 rows × 5 float4 groups)
 *   - 16×16 threads where each thread handles 1 row × 1.25 float4s
 *   - OR: 16×20 output with some threads handling multiple elements
 * 
 * Simpler Approach: Use 10×10 local size for cleaner mapping
 *   - 10×10 = 100 threads
 *   - Process 20×20 output with each thread doing 2×2 outputs
 *   - Clean, simple mapping
 * 
 * This version: 10×10 threads → 20×20 outputs
 * 
 * Author: Tile=20 Research Branch  
 * Date: February 2026
 * Status: EXPERIMENTAL - Approach 2
 */

#define TILE_SIZE 20
#define LOCAL_SIZE 10
#define OUTPUTS_PER_THREAD 2  // Each thread computes 2×2 outputs

__kernel void gemm_tile20_nonsquare(
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
    __local float As[TILE_SIZE * TILE_SIZE];  // 20×20 A
    __local float Bs[TILE_SIZE * TILE_SIZE];  // 20×20 B
    
    // Thread IDs (0-9 for both x and y)
    const int local_x = get_local_id(0);  // 0-9
    const int local_y = get_local_id(1);  // 0-9
    const int local_linear = local_y * LOCAL_SIZE + local_x;  // 0-99
    
    // Group IDs
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    
    // Each thread computes 2×2 outputs
    // Output rows: local_x*2 and local_x*2+1 (0-19)
    // Output cols: local_y*2 and local_y*2+1 (0-19)
    float acc[2][2];  // [row_offset][col_offset]
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Number of tiles in K
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over K tiles
    for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
        
        // ========================================
        // COOPERATIVE LOADING
        // ========================================
        
        // Load A: 400 elements with 100 threads → each loads 4
        for (int i = local_linear; i < TILE_SIZE * TILE_SIZE; i += LOCAL_SIZE * LOCAL_SIZE) {
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
        
        // Load B: 400 elements with 100 threads → each loads 4
        for (int i = local_linear; i < TILE_SIZE * TILE_SIZE; i += LOCAL_SIZE * LOCAL_SIZE) {
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
        // COMPUTE: Each thread does 2×2 outputs
        // ========================================
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            // This thread computes outputs at:
            //   (local_x*2, local_y*2), (local_x*2, local_y*2+1)
            //   (local_x*2+1, local_y*2), (local_x*2+1, local_y*2+1)
            
            for (int row_off = 0; row_off < 2; row_off++) {
                const int out_row = local_x * 2 + row_off;
                const float a_val = As[out_row * TILE_SIZE + k];
                
                for (int col_off = 0; col_off < 2; col_off++) {
                    const int out_col = local_y * 2 + col_off;
                    const float b_val = Bs[k * TILE_SIZE + out_col];
                    
                    acc[row_off][col_off] += a_val * b_val;
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // ========================================
    // WRITE RESULTS: 2×2 outputs per thread
    // ========================================
    
    for (int row_off = 0; row_off < 2; row_off++) {
        for (int col_off = 0; col_off < 2; col_off++) {
            const int global_row = group_x * TILE_SIZE + local_x * 2 + row_off;
            const int global_col = group_y * TILE_SIZE + local_y * 2 + col_off;
            
            if (global_row < M && global_col < N) {
                const int idx = global_row * N + global_col;
                if (beta == 0.0f) {
                    C[idx] = alpha * acc[row_off][col_off];
                } else {
                    C[idx] = alpha * acc[row_off][col_off] + beta * C[idx];
                }
            }
        }
    }
}
