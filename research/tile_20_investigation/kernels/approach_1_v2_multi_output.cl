/*
 * Approach 1 v2: Cooperative Loading with Multiple Outputs Per Thread
 * 
 * Strategy:
 *   - Use 256 threads (16×16) to cooperatively load 400 elements (20×20 tile)
 *   - Each thread computes MULTIPLE output elements to fully utilize loaded data
 *   - Thread (tx, ty) computes outputs at rows: tx, tx+16, tx+32, ... (up to 20)
 *                                    and columns: ty*4, ty*4+1, ty*4+2, ty*4+3
 * 
 * Key Improvement:
 *   - v1 only used first 16×16 of loaded 20×20 tile (wasteful!)
 *   - v2 uses full 20×20 tile by having threads compute multiple rows
 * 
 * Author: Tile=20 Research Branch
 * Date: February 2026
 * Status: EXPERIMENTAL v2
 */

#define TILE_SIZE 20
#define LOCAL_X 16
#define LOCAL_Y 16
#define TOTAL_THREADS (LOCAL_X * LOCAL_Y)  // 256
#define TILE_ELEMENTS (TILE_SIZE * TILE_SIZE)  // 400
#define ROWS_PER_THREAD 2  // Each thread handles up to 2 rows (16*2=32 > 20, covers all)

__kernel void gemm_tile20_multi_output(
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
    __local float Bs[TILE_SIZE * TILE_SIZE * 4];  // 20×20×4 = 1600 floats
    
    // Work-item indices
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int local_linear = local_y * LOCAL_X + local_x;  // 0-255
    
    // Global tile indices
    const int tile_row_base = get_group_id(0) * TILE_SIZE;
    const int tile_col_base = get_group_id(1) * TILE_SIZE;
    
    // Accumulators for multiple rows × 4 columns
    float4 acc[ROWS_PER_THREAD];
    for (int i = 0; i < ROWS_PER_THREAD; i++) {
        acc[i] = (float4)(0.0f);
    }
    
    // Number of tiles in K dimension
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over tiles in K dimension
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        const int tile_k_start = tile_idx * TILE_SIZE;
        
        // ========================================
        // PHASE 1: COOPERATIVE LOADING
        // ========================================
        
        // Load A tile: 400 elements with 256 threads
        // Each thread loads ~1.56 elements
        for (int elem = local_linear; elem < TILE_ELEMENTS; elem += TOTAL_THREADS) {
            const int tile_row = elem / TILE_SIZE;
            const int tile_col = elem % TILE_SIZE;
            const int global_row = tile_row_base + tile_row;
            const int global_col = tile_k_start + tile_col;
            
            if (global_row < M && global_col < K) {
                As[elem] = A[global_row * K + global_col];
            } else {
                As[elem] = 0.0f;
            }
        }
        
        // Load B tile: 1600 elements (20×20×4) with 256 threads
        // Each thread loads ~6.25 elements
        const int b_elements = TILE_ELEMENTS * 4;
        for (int elem = local_linear; elem < b_elements; elem += TOTAL_THREADS) {
            const int tile_row = elem / (TILE_SIZE * 4);
            const int tile_col_vec = (elem / 4) % TILE_SIZE;
            const int vec_elem = elem % 4;
            
            const int global_row = tile_k_start + tile_row;
            const int global_col = tile_col_base * 4 + tile_col_vec * 4 + vec_elem;
            
            if (global_row < K && global_col < N) {
                Bs[elem] = B[global_row * N + global_col];
            } else {
                Bs[elem] = 0.0f;
            }
        }
        
        // Synchronize: wait for all loads to complete
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // ========================================
        // PHASE 2: COMPUTE - Multiple Outputs Per Thread
        // ========================================
        
        // Each thread computes for rows: local_x, local_x+16, ...
        // and columns: local_y*4 to local_y*4+3
        
        #pragma unroll 4
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load B vector once (shared across all output rows)
            const int b_offset = k * TILE_SIZE * 4 + local_y * 4;
            float4 b_val = vload4(0, &Bs[b_offset]);
            
            // Compute for each output row this thread handles
            for (int row_offset = 0; row_offset < ROWS_PER_THREAD; row_offset++) {
                const int tile_row = local_x + row_offset * LOCAL_X;
                if (tile_row < TILE_SIZE) {
                    // Load A element
                    float a_val = As[tile_row * TILE_SIZE + k];
                    
                    // Accumulate
                    acc[row_offset] += a_val * b_val;
                }
            }
        }
        
        // Synchronize before next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // ========================================
    // PHASE 3: WRITE RESULTS
    // ========================================
    
    // Write all output rows this thread computed
    for (int row_offset = 0; row_offset < ROWS_PER_THREAD; row_offset++) {
        const int tile_row = local_x + row_offset * LOCAL_X;
        const int global_row = tile_row_base + tile_row;
        const int global_col_base = tile_col_base * 4 + local_y * 4;
        
        if (global_row < M && global_col_base < N && tile_row < TILE_SIZE) {
            // Apply alpha and beta
            if (beta == 0.0f) {
                vstore4(alpha * acc[row_offset], 0, &C[global_row * N + global_col_base]);
            } else {
                float4 c_val = vload4(0, &C[global_row * N + global_col_base]);
                vstore4(alpha * acc[row_offset] + beta * c_val, 0, &C[global_row * N + global_col_base]);
            }
        }
    }
}
