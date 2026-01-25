/*
 * BLOCK RECURSIVE GEMM KERNEL - Phase 2, Technique 1 (SIMPLIFIED)
 * 
 * Implements cache-optimized matrix multiplication with hierarchical blocking.
 * Target: 850-870 GFLOPS (+10-12% from Phase 1 baseline of 775 GFLOPS)
 * 
 * Algorithm:
 * - Uses hierarchical blocking optimized for AMD GCN cache hierarchy
 * - Level 1: L2 cache blocks (64×64 = 16 KB)
 * - Level 2: Local memory tiles (16×16)
 * - Register blocking for computation
 * 
 * Hardware: AMD Radeon RX 590 (Polaris 10, GCN 4.0)
 * L2 Cache: 2 MB total, 256 KB per CU
 * Local Memory: 64 KB per CU
 * 
 * Author: Phase 2 Development Team
 * Date: 2026-01-24
 */

// Tile dimensions
#define TS 16          // Tile size for local memory
#define WPT 4          // Work per thread

/**
 * KERNEL 1: Basic Tiled GEMM
 * 
 * Simple tiled implementation with local memory.
 * Expected: 780-800 GFLOPS
 */
__kernel void gemm_recursive_block(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    // Thread identifiers
    const int row = get_local_id(0);  // Local row ID (0..TS-1)
    const int col = get_local_id(1);  // Local column ID (0..TS-1)
    const int globalRow = TS*get_group_id(0) + row;  // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col;  // Column ID of C (0..N)
    
    // Local memory for tiles
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
    
    // Initialize accumulator
    float acc = 0.0f;
    
    // Loop over K dimension in tiles
    const int numTiles = (K + TS - 1) / TS;
    for (int t = 0; t < numTiles; t++) {
        // Load tile from A
        const int tiledRow = TS*t + col;
        Asub[row][col] = (globalRow < M && tiledRow < K) ? 
                         A[globalRow*K + tiledRow] : 0.0f;
        
        // Load tile from B
        const int tiledCol = TS*t + row;
        Bsub[row][col] = (tiledCol < K && globalCol < N) ? 
                         B[tiledCol*N + globalCol] : 0.0f;
        
        // Synchronize
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute dot product for this tile
        #pragma unroll
        for (int k = 0; k < TS; k++) {
            acc = fma(Asub[row][k], Bsub[k][col], acc);
        }
        
        // Synchronize
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store result
    if (globalRow < M && globalCol < N) {
        const int c_idx = globalRow*N + globalCol;
        C[c_idx] = (beta == 0.0f) ? alpha*acc : fma(alpha, acc, beta*C[c_idx]);
    }
}

/**
 * KERNEL 2: Register-Blocked GEMM
 * 
 * Each thread computes 2×2 output elements using register blocking.
 * Expected: 820-840 GFLOPS
 */
__kernel void gemm_recursive_two_level(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    // Thread and block identifiers
    const int row = get_local_id(0);  
    const int col = get_local_id(1);  
    const int globalRow = TS*get_group_id(0) + row*2;  // Each thread handles 2 rows
    const int globalCol = TS*get_group_id(1) + col*2;  // Each thread handles 2 cols
    
    // Local memory for tiles
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
    
    // Register array for 2×2 output block
    float acc[2][2];
    acc[0][0] = 0.0f;
    acc[0][1] = 0.0f;
    acc[1][0] = 0.0f;
    acc[1][1] = 0.0f;
    
    // Loop over K dimension in tiles
    const int numTiles = (K + TS - 1) / TS;
    for (int t = 0; t < numTiles; t++) {
        // Load tile from A (each thread loads 2 elements vertically)
        const int tiledK = TS*t + col;
        if (row*2 < TS && tiledK < K) {
            if (globalRow < M) Asub[row*2][col] = A[globalRow*K + tiledK];
            if (globalRow+1 < M) Asub[row*2+1][col] = A[(globalRow+1)*K + tiledK];
        }
        
        // Load tile from B (each thread loads 2 elements horizontally)
        const int tiledRow = TS*t + row;
        if (tiledRow < K && col*2 < TS) {
            if (globalCol < N) Bsub[row][col*2] = B[tiledRow*N + globalCol];
            if (globalCol+1 < N) Bsub[row][col*2+1] = B[tiledRow*N + globalCol+1];
        }
        
        // Synchronize
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute 2×2 block
        #pragma unroll
        for (int k = 0; k < TS; k++) {
            float a0 = Asub[row*2][k];
            float a1 = Asub[row*2+1][k];
            float b0 = Bsub[k][col*2];
            float b1 = Bsub[k][col*2+1];
            
            acc[0][0] = fma(a0, b0, acc[0][0]);
            acc[0][1] = fma(a0, b1, acc[0][1]);
            acc[1][0] = fma(a1, b0, acc[1][0]);
            acc[1][1] = fma(a1, b1, acc[1][1]);
        }
        
        // Synchronize
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store 2×2 results
    if (globalRow < M && globalCol < N) {
        const int idx00 = globalRow*N + globalCol;
        C[idx00] = (beta == 0.0f) ? alpha*acc[0][0] : fma(alpha, acc[0][0], beta*C[idx00]);
    }
    if (globalRow < M && globalCol+1 < N) {
        const int idx01 = globalRow*N + globalCol+1;
        C[idx01] = (beta == 0.0f) ? alpha*acc[0][1] : fma(alpha, acc[0][1], beta*C[idx01]);
    }
    if (globalRow+1 < M && globalCol < N) {
        const int idx10 = (globalRow+1)*N + globalCol;
        C[idx10] = (beta == 0.0f) ? alpha*acc[1][0] : fma(alpha, acc[1][0], beta*C[idx10]);
    }
    if (globalRow+1 < M && globalCol+1 < N) {
        const int idx11 = (globalRow+1)*N + globalCol+1;
        C[idx11] = (beta == 0.0f) ? alpha*acc[1][1] : fma(alpha, acc[1][1], beta*C[idx11]);
    }
}

/**
 * KERNEL 3: Optimized GEMM with 4×4 Register Blocking
 * 
 * Each thread computes 4×4 output elements.
 * Target: 850-870 GFLOPS (+10-12% improvement)
 */
__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void gemm_recursive_optimized(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M, const int N, const int K,
    const float alpha, const float beta
) {
    // Thread and block identifiers
    const int row = get_local_id(0);  // 0..15
    const int col = get_local_id(1);  // 0..15
    const int globalRow = 64*get_group_id(0) + row*4;  // Each thread handles 4 rows
    const int globalCol = 64*get_group_id(1) + col*4;  // Each thread handles 4 cols
    
    // Local memory for tiles (64×16 and 16×64 arranged as smaller tiles)
    __local float Asub[64][16];
    __local float Bsub[16][64];
    
    // Register array for 4×4 output block
    float acc[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Loop over K dimension in tiles of 16
    const int numTiles = (K + 15) / 16;
    for (int t = 0; t < numTiles; t++) {
        const int tiledK = 16*t;
        
        // Load 64×16 tile from A
        // Each thread loads 4 elements vertically
        for (int i = 0; i < 4; i++) {
            const int a_row = globalRow + i;
            const int a_col = tiledK + col;
            if (a_row < M && a_col < K) {
                Asub[row*4+i][col] = A[a_row*K + a_col];
            } else {
                Asub[row*4+i][col] = 0.0f;
            }
        }
        
        // Load 16×64 tile from B
        // Each thread loads 4 elements horizontally
        for (int j = 0; j < 4; j++) {
            const int b_row = tiledK + row;
            const int b_col = globalCol + j;
            if (b_row < K && b_col < N) {
                Bsub[row][col*4+j] = B[b_row*N + b_col];
            } else {
                Bsub[row][col*4+j] = 0.0f;
            }
        }
        
        // Synchronize
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute 4×4 block using register blocking
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            // Load 4 values from A and B
            float a[4], b[4];
            for (int i = 0; i < 4; i++) {
                a[i] = Asub[row*4+i][k];
                b[i] = Bsub[k][col*4+i];
            }
            
            // Compute outer product
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
                }
            }
        }
        
        // Synchronize
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Store 4×4 results
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            const int c_row = globalRow + i;
            const int c_col = globalCol + j;
            if (c_row < M && c_col < N) {
                const int idx = c_row*N + c_col;
                C[idx] = (beta == 0.0f) ? alpha*acc[i][j] : fma(alpha, acc[i][j], beta*C[idx]);
            }
        }
    }
}
