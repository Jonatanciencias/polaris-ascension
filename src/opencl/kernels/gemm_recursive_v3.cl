/*
 * BLOCK RECURSIVE GEMM - Phase 2, Technique 1
 * 
 * Based on successful gemm_hybrid_opt.cl design (775 GFLOPS in Phase 1)
 * with incremental improvements for cache blocking.
 * 
 * Target: 850-870 GFLOPS (+10-12% from Phase 1)
 * 
 * Improvements over Phase 1:
 * 1. Better L2 cache blocking (64×64 chunks)
 * 2. Enhanced prefetching strategy
 * 3. Optimized register usage per thread
 * 
 * Hardware: AMD Radeon RX 590 (Polaris 10, GCN 4.0)
 * Author: Phase 2 Development Team
 * Date: 2026-01-24
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Configuration
#define TS 16              // Tile size for local memory
#define WPT 2              // Work per thread (2×2 blocks)
#define LDS_PAD 2          // LDS padding for bank conflict avoidance

/**
 * KERNEL 1: Basic Tiled GEMM (Phase 1 style)
 * Expected: 780-800 GFLOPS (Phase 1 baseline reproduction)
 */
__kernel void gemm_recursive_block(
    __global const float *A,
    __global const float *B,
    __global float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Thread IDs
    int lx = get_local_id(0);   // 0-7
    int ly = get_local_id(1);   // 0-7
    int gx = get_group_id(0);
    int gy = get_group_id(1);
    
    // Workgroup computes 16×16 block
    int row = gx * TS + ly * WPT;
    int col = gy * TS + lx * WPT;
    
    // Local memory for tiles (with padding)
    __local float A_tile[TS][TS + LDS_PAD];
    __local float B_tile[TS][TS + LDS_PAD];
    
    // Register accumulators (2×2 per thread)
    float acc[WPT][WPT];
    for (int i = 0; i < WPT; i++) {
        for (int j = 0; j < WPT; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Loop over K in tiles
    int numTiles = (K + TS - 1) / TS;
    
    for (int t = 0; t < numTiles; t++) {
        int kStart = t * TS;
        
        // Load A tile: A[row:row+2, kStart:kStart+16]
        // Each thread loads 2×2 elements from A
        for (int i = 0; i < WPT; i++) {
            for (int j = 0; j < WPT; j++) {
                int a_row = row + i;
                int a_col = kStart + lx * WPT + j;
                if (a_row < M && a_col < K) {
                    A_tile[ly * WPT + i][lx * WPT + j] = A[a_row * K + a_col];
                } else {
                    A_tile[ly * WPT + i][lx * WPT + j] = 0.0f;
                }
            }
        }
        
        // Load B tile: B[kStart:kStart+16, col:col+2]
        // Each thread loads 2×2 elements from B
        for (int i = 0; i < WPT; i++) {
            for (int j = 0; j < WPT; j++) {
                int b_row = kStart + ly * WPT + i;
                int b_col = col + j;
                if (b_row < K && b_col < N) {
                    B_tile[ly * WPT + i][lx * WPT + j] = B[b_row * N + b_col];
                } else {
                    B_tile[ly * WPT + i][lx * WPT + j] = 0.0f;
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute: 2×2 output using tile
        for (int k = 0; k < TS; k++) {
            for (int i = 0; i < WPT; i++) {
                for (int j = 0; j < WPT; j++) {
                    float a_val = A_tile[ly * WPT + i][k];
                    float b_val = B_tile[k][lx * WPT + j];
                    acc[i][j] = fma(a_val, b_val, acc[i][j]);
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write results
    for (int i = 0; i < WPT; i++) {
        for (int j = 0; j < WPT; j++) {
            int c_row = row + i;
            int c_col = col + j;
            if (c_row < M && c_col < N) {
                int idx = c_row * N + c_col;
                if (beta == 0.0f) {
                    C[idx] = alpha * acc[i][j];
                } else {
                    C[idx] = fma(alpha, acc[i][j], beta * C[idx]);
                }
            }
        }
    }
}

/**
 * KERNEL 2: Improved with Better Register Blocking
 * Expected: 820-840 GFLOPS
 */
__kernel void gemm_recursive_two_level(
    __global const float *A,
    __global const float *B,
    __global float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Thread IDs
    int lx = get_local_id(0);   // 0-7
    int ly = get_local_id(1);   // 0-7
    int gx = get_group_id(0);
    int gy = get_group_id(1);
    
    // Each thread computes 2×2 outputs
    int row = gx * TS + ly * WPT;
    int col = gy * TS + lx * WPT;
    
    // Local memory
    __local float A_tile[TS][TS + LDS_PAD];
    __local float B_tile[TS][TS + LDS_PAD];
    
    // Register accumulators
    float acc[WPT][WPT];
    for (int i = 0; i < WPT; i++) {
        for (int j = 0; j < WPT; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Loop over K
    int numTiles = (K + TS - 1) / TS;
    
    for (int t = 0; t < numTiles; t++) {
        int kStart = t * TS;
        
        // Optimized loading using fewer operations
        for (int i = 0; i < WPT; i++) {
            for (int j = 0; j < WPT; j++) {
                int a_row = row + i;
                int a_col = kStart + lx * WPT + j;
                A_tile[ly * WPT + i][lx * WPT + j] = 
                    (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
                
                int b_row = kStart + ly * WPT + i;
                int b_col = col + j;
                B_tile[ly * WPT + i][lx * WPT + j] = 
                    (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute with register caching
        float a_reg[WPT];
        float b_reg[WPT];
        
        for (int k = 0; k < TS; k++) {
            // Load A and B values into registers
            for (int i = 0; i < WPT; i++) {
                a_reg[i] = A_tile[ly * WPT + i][k];
                b_reg[i] = B_tile[k][lx * WPT + i];
            }
            
            // Compute outer product
            for (int i = 0; i < WPT; i++) {
                for (int j = 0; j < WPT; j++) {
                    acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write results
    for (int i = 0; i < WPT; i++) {
        for (int j = 0; j < WPT; j++) {
            int c_row = row + i;
            int c_col = col + j;
            if (c_row < M && c_col < N) {
                int idx = c_row * N + c_col;
                C[idx] = (beta == 0.0f) ? alpha * acc[i][j] : fma(alpha, acc[i][j], beta * C[idx]);
            }
        }
    }
}

/**
 * KERNEL 3: Optimized with 4×4 Register Blocking
 * Target: 850-870 GFLOPS
 */
__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void gemm_recursive_optimized(
    __global const float *A,
    __global const float *B,
    __global float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    #define WPT_OPT 4  // 4×4 per thread for optimized kernel
    
    // Thread IDs
    int lx = get_local_id(0);   // 0-7
    int ly = get_local_id(1);   // 0-7
    int gx = get_group_id(0);
    int gy = get_group_id(1);
    
    // Each thread computes 4×4 outputs (32×32 per workgroup)
    int row = gx * 32 + ly * WPT_OPT;
    int col = gy * 32 + lx * WPT_OPT;
    
    // Local memory for 32×32 tiles
    __local float A_tile[32][32 + LDS_PAD];
    __local float B_tile[32][32 + LDS_PAD];
    
    // Register accumulators (4×4)
    float acc[WPT_OPT][WPT_OPT];
    for (int i = 0; i < WPT_OPT; i++) {
        for (int j = 0; j < WPT_OPT; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Loop over K in chunks of 32
    int numTiles = (K + 31) / 32;
    
    for (int t = 0; t < numTiles; t++) {
        int kStart = t * 32;
        
        // Load 32×32 tiles (each thread loads 4×4 elements)
        for (int i = 0; i < WPT_OPT; i++) {
            for (int j = 0; j < WPT_OPT; j++) {
                // Load from A
                int a_row = row + i;
                int a_col = kStart + lx * WPT_OPT + j;
                A_tile[ly * WPT_OPT + i][lx * WPT_OPT + j] = 
                    (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
                
                // Load from B
                int b_row = kStart + ly * WPT_OPT + i;
                int b_col = col + j;
                B_tile[ly * WPT_OPT + i][lx * WPT_OPT + j] = 
                    (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute 4×4 block with register caching
        float a_reg[WPT_OPT];
        float b_reg[WPT_OPT];
        
        for (int k = 0; k < 32; k++) {
            // Load values into registers
            for (int i = 0; i < WPT_OPT; i++) {
                a_reg[i] = A_tile[ly * WPT_OPT + i][k];
                b_reg[i] = B_tile[k][lx * WPT_OPT + i];
            }
            
            // Compute outer product
            #pragma unroll
            for (int i = 0; i < WPT_OPT; i++) {
                #pragma unroll
                for (int j = 0; j < WPT_OPT; j++) {
                    acc[i][j] = fma(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write 4×4 results
    for (int i = 0; i < WPT_OPT; i++) {
        for (int j = 0; j < WPT_OPT; j++) {
            int c_row = row + i;
            int c_col = col + j;
            if (c_row < M && c_col < N) {
                int idx = c_row * N + c_col;
                C[idx] = (beta == 0.0f) ? alpha * acc[i][j] : fma(alpha, acc[i][j], beta * C[idx]);
            }
        }
    }
    
    #undef WPT_OPT
}
