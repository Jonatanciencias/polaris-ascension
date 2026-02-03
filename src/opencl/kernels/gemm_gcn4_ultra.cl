/**
 * ============================================================================
 * GEMM GCN4 ULTRA - MAXIMUM PERFORMANCE KERNEL FOR POLARIS ARCHITECTURE
 * ============================================================================
 * 
 * Designed specifically for AMD Radeon RX 580/590 (Polaris 10/20)
 * Targeting maximum utilization of GCN 4.0 architecture
 * 
 * GCN 4.0 Architecture Characteristics:
 * ─────────────────────────────────────
 * • 36 Compute Units (CUs)
 * • 4 SIMD units per CU × 16 lanes = 64 lanes per wavefront
 * • 64KB LDS per CU (max 32KB per workgroup on Mesa)
 * • 256 VGPRs per SIMD (256KB total per CU)
 * • Dual ALU pipes: VALU (vector) + SALU (scalar)
 * • 256-bit GDDR5 memory bus @ 8Gbps = 256 GB/s
 * • L1 cache: 16KB per CU (read-only texture cache)
 * • L2 cache: 2MB shared
 * 
 * Optimization Techniques Implemented:
 * ────────────────────────────────────
 * 1. WAVEFRONT-LEVEL COMPUTATION
 *    - Full wavefront utilization (64 work-items)
 *    - Wavefront-uniform control flow
 *    - No divergent branches
 * 
 * 2. SOFTWARE PIPELINING (Double Buffering)
 *    - Overlap memory loads with computation
 *    - Prefetch next tile while computing current
 *    - Hide memory latency (~400 cycles GDDR5)
 * 
 * 3. INSTRUCTION-LEVEL PARALLELISM (ILP)
 *    - 8x8 register blocking per thread
 *    - Independent MAD chains for dual-issue
 *    - Minimized dependencies between instructions
 * 
 * 4. MEMORY ACCESS OPTIMIZATION
 *    - 128-byte aligned accesses (32 floats = full memory transaction)
 *    - Bank conflict-free LDS access patterns
 *    - Coalesced global memory access
 * 
 * 5. REGISTER FILE OPTIMIZATION
 *    - ~120 VGPRs per work-item for good occupancy
 *    - Accumulator registers in register file
 *    - No register spilling
 * 
 * 6. OCCUPANCY TARGETING
 *    - 4 wavefronts per SIMD target (256 threads per CU)
 *    - Balanced LDS and VGPR usage
 *    - 16KB LDS per workgroup (allows 4 WG per CU)
 * 
 * Expected Performance:
 * ────────────────────
 * • Target: 400-600 GFLOPS sustained (current baseline ~230 GFLOPS)
 * • Theoretical max: 6170 GFLOPS (memory-bound limits to ~15-20%)
 * • Target efficiency: 3-5x improvement over basic tiled GEMM
 * 
 * Author: Radeon RX 580 Optimization Framework
 * License: MIT
 * ============================================================================
 */

// ============================================================================
// GCN4 OPTIMIZED CONSTANTS
// ============================================================================

// Tile dimensions optimized for GCN4 LDS and register file
#ifndef GCN_TILE_M
#define GCN_TILE_M 64           // Tile height (must be multiple of wavefront)
#endif

#ifndef GCN_TILE_N
#define GCN_TILE_N 64           // Tile width
#endif

#ifndef GCN_TILE_K
#define GCN_TILE_K 16           // K-dimension tile (for double buffering)
#endif

// Work per thread - determines register blocking
#define WPT_M 8                 // Work-items per thread in M dimension
#define WPT_N 8                 // Work-items per thread in N dimension

// Derived constants
#define TSM (GCN_TILE_M / WPT_M)    // Thread block M = 8
#define TSN (GCN_TILE_N / WPT_N)    // Thread block N = 8
#define LOCAL_SIZE (TSM * TSN)      // 64 = 1 wavefront (optimal!)

// LDS dimensions with padding for bank conflict avoidance
// GCN has 32 banks, 4 bytes per bank = 128 byte stride
#define LDS_PADDING 1
#define LDS_A_STRIDE (GCN_TILE_K + LDS_PADDING)
#define LDS_B_STRIDE (GCN_TILE_N + LDS_PADDING)

// Memory transaction sizes
#define VECTOR_WIDTH 4          // float4 for 128-bit loads

// ============================================================================
// UTILITY MACROS FOR GCN OPTIMIZATION
// ============================================================================

// Fused multiply-add (uses v_fma_f32 on GCN)
#define FMA(a, b, c) mad(a, b, c)

// Vectorized FMA for float4
#define FMA4(a, b, c) (float4)(FMA(a.x, b, c.x), FMA(a.y, b, c.y), \
                               FMA(a.z, b, c.z), FMA(a.w, b, c.w))

// ============================================================================
// KERNEL 1: GCN4 ULTRA GEMM - MAXIMUM THROUGHPUT
// ============================================================================
/**
 * gemm_gcn4_ultra - Ultra-optimized GEMM for Polaris GPUs
 * 
 * C = alpha * A * B + beta * C
 * 
 * Workgroup: 8x8 = 64 threads (1 wavefront)
 * Work per thread: 8x8 = 64 outputs
 * Tile: 64x64 output tile per workgroup
 * 
 * This configuration achieves:
 * - Perfect wavefront utilization (no partial wavefronts)
 * - High arithmetic intensity (64 MADs per memory access)
 * - Good occupancy (low VGPR count allows multiple wavefronts)
 */
__kernel __attribute__((reqd_work_group_size(TSM, TSN, 1)))
void gemm_gcn4_ultra(
    const int M, const int N, const int K,
    const float alpha,
    __global const float* restrict A,
    __global const float* restrict B,
    const float beta,
    __global float* restrict C)
{
    // ========================================================================
    // IDENTIFICATION
    // ========================================================================
    const int tidm = get_local_id(0);      // Thread ID in M (0-7)
    const int tidn = get_local_id(1);      // Thread ID in N (0-7)
    const int tid = tidm * TSN + tidn;     // Linear thread ID (0-63)
    
    const int gidm = get_group_id(0);      // Workgroup ID in M
    const int gidn = get_group_id(1);      // Workgroup ID in N
    
    // Global base indices for this workgroup
    const int base_m = gidm * GCN_TILE_M;
    const int base_n = gidn * GCN_TILE_N;
    
    // ========================================================================
    // LOCAL MEMORY DECLARATIONS (Double Buffering)
    // ========================================================================
    // Two buffers for software pipelining
    __local float As[2][GCN_TILE_M][LDS_A_STRIDE];
    __local float Bs[2][GCN_TILE_K][LDS_B_STRIDE];
    
    // ========================================================================
    // REGISTER ACCUMULATORS (8x8 = 64 per thread)
    // ========================================================================
    // Using separate registers for ILP (compiler can schedule independently)
    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc04 = 0.0f, acc05 = 0.0f, acc06 = 0.0f, acc07 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc14 = 0.0f, acc15 = 0.0f, acc16 = 0.0f, acc17 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc24 = 0.0f, acc25 = 0.0f, acc26 = 0.0f, acc27 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;
    float acc34 = 0.0f, acc35 = 0.0f, acc36 = 0.0f, acc37 = 0.0f;
    float acc40 = 0.0f, acc41 = 0.0f, acc42 = 0.0f, acc43 = 0.0f;
    float acc44 = 0.0f, acc45 = 0.0f, acc46 = 0.0f, acc47 = 0.0f;
    float acc50 = 0.0f, acc51 = 0.0f, acc52 = 0.0f, acc53 = 0.0f;
    float acc54 = 0.0f, acc55 = 0.0f, acc56 = 0.0f, acc57 = 0.0f;
    float acc60 = 0.0f, acc61 = 0.0f, acc62 = 0.0f, acc63 = 0.0f;
    float acc64 = 0.0f, acc65 = 0.0f, acc66 = 0.0f, acc67 = 0.0f;
    float acc70 = 0.0f, acc71 = 0.0f, acc72 = 0.0f, acc73 = 0.0f;
    float acc74 = 0.0f, acc75 = 0.0f, acc76 = 0.0f, acc77 = 0.0f;
    
    // Register arrays for A and B values
    float regA[WPT_M];
    float regB[WPT_N];
    
    // ========================================================================
    // TILING LOOP WITH SOFTWARE PIPELINING
    // ========================================================================
    const int num_tiles = (K + GCN_TILE_K - 1) / GCN_TILE_K;
    int curr_buf = 0;
    
    // PROLOGUE: Load first tile
    {
        const int k_base = 0;
        
        // Load A tile: 64 threads load 64x16 = 1024 elements
        // Each thread loads 16 elements (1024/64)
        #pragma unroll
        for (int i = 0; i < GCN_TILE_M * GCN_TILE_K / LOCAL_SIZE; i++) {
            const int idx = tid + i * LOCAL_SIZE;
            const int row = idx / GCN_TILE_K;
            const int col = idx % GCN_TILE_K;
            const int global_row = base_m + row;
            const int global_col = k_base + col;
            
            if (global_row < M && global_col < K) {
                As[0][row][col] = A[global_row * K + global_col];
            } else {
                As[0][row][col] = 0.0f;
            }
        }
        
        // Load B tile: 64 threads load 16x64 = 1024 elements
        #pragma unroll
        for (int i = 0; i < GCN_TILE_K * GCN_TILE_N / LOCAL_SIZE; i++) {
            const int idx = tid + i * LOCAL_SIZE;
            const int row = idx / GCN_TILE_N;
            const int col = idx % GCN_TILE_N;
            const int global_row = k_base + row;
            const int global_col = base_n + col;
            
            if (global_row < K && global_col < N) {
                Bs[0][row][col] = B[global_row * N + global_col];
            } else {
                Bs[0][row][col] = 0.0f;
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // MAIN LOOP: Process tiles with double buffering
    for (int t = 0; t < num_tiles; t++) {
        const int next_buf = 1 - curr_buf;
        const int k_base_next = (t + 1) * GCN_TILE_K;
        
        // ====================================================================
        // PHASE 1: Start loading next tile (async conceptually)
        // ====================================================================
        if (t + 1 < num_tiles) {
            // Load A tile for next iteration
            #pragma unroll
            for (int i = 0; i < GCN_TILE_M * GCN_TILE_K / LOCAL_SIZE; i++) {
                const int idx = tid + i * LOCAL_SIZE;
                const int row = idx / GCN_TILE_K;
                const int col = idx % GCN_TILE_K;
                const int global_row = base_m + row;
                const int global_col = k_base_next + col;
                
                if (global_row < M && global_col < K) {
                    As[next_buf][row][col] = A[global_row * K + global_col];
                } else {
                    As[next_buf][row][col] = 0.0f;
                }
            }
            
            // Load B tile for next iteration
            #pragma unroll
            for (int i = 0; i < GCN_TILE_K * GCN_TILE_N / LOCAL_SIZE; i++) {
                const int idx = tid + i * LOCAL_SIZE;
                const int row = idx / GCN_TILE_N;
                const int col = idx % GCN_TILE_N;
                const int global_row = k_base_next + row;
                const int global_col = base_n + col;
                
                if (global_row < K && global_col < N) {
                    Bs[next_buf][row][col] = B[global_row * N + global_col];
                } else {
                    Bs[next_buf][row][col] = 0.0f;
                }
            }
        }
        
        // ====================================================================
        // PHASE 2: Compute on current tile
        // ====================================================================
        // Process K dimension with full unrolling for ILP
        #pragma unroll
        for (int k = 0; k < GCN_TILE_K; k++) {
            // Load A values into registers (8 values per thread)
            #pragma unroll
            for (int m = 0; m < WPT_M; m++) {
                regA[m] = As[curr_buf][tidm * WPT_M + m][k];
            }
            
            // Load B values into registers (8 values per thread)
            #pragma unroll
            for (int n = 0; n < WPT_N; n++) {
                regB[n] = Bs[curr_buf][k][tidn * WPT_N + n];
            }
            
            // Rank-1 update: 8x8 MADs
            // Fully unrolled for maximum ILP and dual-issue on GCN
            acc00 = FMA(regA[0], regB[0], acc00);
            acc01 = FMA(regA[0], regB[1], acc01);
            acc02 = FMA(regA[0], regB[2], acc02);
            acc03 = FMA(regA[0], regB[3], acc03);
            acc04 = FMA(regA[0], regB[4], acc04);
            acc05 = FMA(regA[0], regB[5], acc05);
            acc06 = FMA(regA[0], regB[6], acc06);
            acc07 = FMA(regA[0], regB[7], acc07);
            
            acc10 = FMA(regA[1], regB[0], acc10);
            acc11 = FMA(regA[1], regB[1], acc11);
            acc12 = FMA(regA[1], regB[2], acc12);
            acc13 = FMA(regA[1], regB[3], acc13);
            acc14 = FMA(regA[1], regB[4], acc14);
            acc15 = FMA(regA[1], regB[5], acc15);
            acc16 = FMA(regA[1], regB[6], acc16);
            acc17 = FMA(regA[1], regB[7], acc17);
            
            acc20 = FMA(regA[2], regB[0], acc20);
            acc21 = FMA(regA[2], regB[1], acc21);
            acc22 = FMA(regA[2], regB[2], acc22);
            acc23 = FMA(regA[2], regB[3], acc23);
            acc24 = FMA(regA[2], regB[4], acc24);
            acc25 = FMA(regA[2], regB[5], acc25);
            acc26 = FMA(regA[2], regB[6], acc26);
            acc27 = FMA(regA[2], regB[7], acc27);
            
            acc30 = FMA(regA[3], regB[0], acc30);
            acc31 = FMA(regA[3], regB[1], acc31);
            acc32 = FMA(regA[3], regB[2], acc32);
            acc33 = FMA(regA[3], regB[3], acc33);
            acc34 = FMA(regA[3], regB[4], acc34);
            acc35 = FMA(regA[3], regB[5], acc35);
            acc36 = FMA(regA[3], regB[6], acc36);
            acc37 = FMA(regA[3], regB[7], acc37);
            
            acc40 = FMA(regA[4], regB[0], acc40);
            acc41 = FMA(regA[4], regB[1], acc41);
            acc42 = FMA(regA[4], regB[2], acc42);
            acc43 = FMA(regA[4], regB[3], acc43);
            acc44 = FMA(regA[4], regB[4], acc44);
            acc45 = FMA(regA[4], regB[5], acc45);
            acc46 = FMA(regA[4], regB[6], acc46);
            acc47 = FMA(regA[4], regB[7], acc47);
            
            acc50 = FMA(regA[5], regB[0], acc50);
            acc51 = FMA(regA[5], regB[1], acc51);
            acc52 = FMA(regA[5], regB[2], acc52);
            acc53 = FMA(regA[5], regB[3], acc53);
            acc54 = FMA(regA[5], regB[4], acc54);
            acc55 = FMA(regA[5], regB[5], acc55);
            acc56 = FMA(regA[5], regB[6], acc56);
            acc57 = FMA(regA[5], regB[7], acc57);
            
            acc60 = FMA(regA[6], regB[0], acc60);
            acc61 = FMA(regA[6], regB[1], acc61);
            acc62 = FMA(regA[6], regB[2], acc62);
            acc63 = FMA(regA[6], regB[3], acc63);
            acc64 = FMA(regA[6], regB[4], acc64);
            acc65 = FMA(regA[6], regB[5], acc65);
            acc66 = FMA(regA[6], regB[6], acc66);
            acc67 = FMA(regA[6], regB[7], acc67);
            
            acc70 = FMA(regA[7], regB[0], acc70);
            acc71 = FMA(regA[7], regB[1], acc71);
            acc72 = FMA(regA[7], regB[2], acc72);
            acc73 = FMA(regA[7], regB[3], acc73);
            acc74 = FMA(regA[7], regB[4], acc74);
            acc75 = FMA(regA[7], regB[5], acc75);
            acc76 = FMA(regA[7], regB[6], acc76);
            acc77 = FMA(regA[7], regB[7], acc77);
        }
        
        // Switch buffers
        curr_buf = next_buf;
        
        // Synchronize before next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // ========================================================================
    // WRITE RESULTS TO GLOBAL MEMORY
    // ========================================================================
    // Each thread writes 8x8 = 64 elements
    const int out_base_m = base_m + tidm * WPT_M;
    const int out_base_n = base_n + tidn * WPT_N;
    
    // Apply alpha/beta scaling and write
    #define WRITE_RESULT(row, col, acc_val) \
        if (out_base_m + row < M && out_base_n + col < N) { \
            const int idx = (out_base_m + row) * N + out_base_n + col; \
            C[idx] = alpha * acc_val + beta * C[idx]; \
        }
    
    // Row 0
    WRITE_RESULT(0, 0, acc00); WRITE_RESULT(0, 1, acc01);
    WRITE_RESULT(0, 2, acc02); WRITE_RESULT(0, 3, acc03);
    WRITE_RESULT(0, 4, acc04); WRITE_RESULT(0, 5, acc05);
    WRITE_RESULT(0, 6, acc06); WRITE_RESULT(0, 7, acc07);
    
    // Row 1
    WRITE_RESULT(1, 0, acc10); WRITE_RESULT(1, 1, acc11);
    WRITE_RESULT(1, 2, acc12); WRITE_RESULT(1, 3, acc13);
    WRITE_RESULT(1, 4, acc14); WRITE_RESULT(1, 5, acc15);
    WRITE_RESULT(1, 6, acc16); WRITE_RESULT(1, 7, acc17);
    
    // Row 2
    WRITE_RESULT(2, 0, acc20); WRITE_RESULT(2, 1, acc21);
    WRITE_RESULT(2, 2, acc22); WRITE_RESULT(2, 3, acc23);
    WRITE_RESULT(2, 4, acc24); WRITE_RESULT(2, 5, acc25);
    WRITE_RESULT(2, 6, acc26); WRITE_RESULT(2, 7, acc27);
    
    // Row 3
    WRITE_RESULT(3, 0, acc30); WRITE_RESULT(3, 1, acc31);
    WRITE_RESULT(3, 2, acc32); WRITE_RESULT(3, 3, acc33);
    WRITE_RESULT(3, 4, acc34); WRITE_RESULT(3, 5, acc35);
    WRITE_RESULT(3, 6, acc36); WRITE_RESULT(3, 7, acc37);
    
    // Row 4
    WRITE_RESULT(4, 0, acc40); WRITE_RESULT(4, 1, acc41);
    WRITE_RESULT(4, 2, acc42); WRITE_RESULT(4, 3, acc43);
    WRITE_RESULT(4, 4, acc44); WRITE_RESULT(4, 5, acc45);
    WRITE_RESULT(4, 6, acc46); WRITE_RESULT(4, 7, acc47);
    
    // Row 5
    WRITE_RESULT(5, 0, acc50); WRITE_RESULT(5, 1, acc51);
    WRITE_RESULT(5, 2, acc52); WRITE_RESULT(5, 3, acc53);
    WRITE_RESULT(5, 4, acc54); WRITE_RESULT(5, 5, acc55);
    WRITE_RESULT(5, 6, acc56); WRITE_RESULT(5, 7, acc57);
    
    // Row 6
    WRITE_RESULT(6, 0, acc60); WRITE_RESULT(6, 1, acc61);
    WRITE_RESULT(6, 2, acc62); WRITE_RESULT(6, 3, acc63);
    WRITE_RESULT(6, 4, acc64); WRITE_RESULT(6, 5, acc65);
    WRITE_RESULT(6, 6, acc66); WRITE_RESULT(6, 7, acc67);
    
    // Row 7
    WRITE_RESULT(7, 0, acc70); WRITE_RESULT(7, 1, acc71);
    WRITE_RESULT(7, 2, acc72); WRITE_RESULT(7, 3, acc73);
    WRITE_RESULT(7, 4, acc74); WRITE_RESULT(7, 5, acc75);
    WRITE_RESULT(7, 6, acc76); WRITE_RESULT(7, 7, acc77);
    
    #undef WRITE_RESULT
}


// ============================================================================
// KERNEL 2: GCN4 VECTORIZED GEMM - FLOAT4 OPTIMIZED
// ============================================================================
/**
 * gemm_gcn4_vec4 - Vectorized GEMM using float4 for memory bandwidth
 * 
 * Uses float4 loads/stores to maximize memory throughput.
 * Better for bandwidth-bound scenarios (large K dimension).
 */
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_gcn4_vec4(
    const int M, const int N, const int K,
    const float alpha,
    __global const float4* restrict A,
    __global const float4* restrict B,
    const float beta,
    __global float4* restrict C)
{
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);
    const int gidm = get_group_id(0);
    const int gidn = get_group_id(1);
    
    // Each thread computes 4x4 output elements
    const int row = gidm * 32 + tidm * 4;
    const int col = gidn * 32 + tidn * 4;
    
    // LDS for tiles
    __local float4 As[32][4 + 1];  // 32x16 with padding
    __local float4 Bs[16][8 + 1];  // 16x32 with padding
    
    // Accumulators (4x4 = 16 per thread, stored as 4 float4)
    float4 acc0 = (float4)(0.0f);
    float4 acc1 = (float4)(0.0f);
    float4 acc2 = (float4)(0.0f);
    float4 acc3 = (float4)(0.0f);
    
    const int K4 = K / 4;
    const int num_tiles = (K + 15) / 16;
    
    for (int t = 0; t < num_tiles; t++) {
        const int k_base = t * 16;
        
        // Cooperative loading of A tile
        if (row + tidm < M && k_base / 4 + tidn < K4) {
            As[tidm * 4 + 0][tidn] = A[(row + 0) * K4 + k_base / 4 + tidn];
            As[tidm * 4 + 1][tidn] = A[(row + 1) * K4 + k_base / 4 + tidn];
            As[tidm * 4 + 2][tidn] = A[(row + 2) * K4 + k_base / 4 + tidn];
            As[tidm * 4 + 3][tidn] = A[(row + 3) * K4 + k_base / 4 + tidn];
        }
        
        // Cooperative loading of B tile
        if (k_base + tidm < K && col / 4 + tidn < N / 4) {
            Bs[tidm * 2 + 0][tidn] = B[(k_base + tidm * 2 + 0) * (N / 4) + col / 4 + tidn];
            Bs[tidm * 2 + 1][tidn] = B[(k_base + tidm * 2 + 1) * (N / 4) + col / 4 + tidn];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute
        #pragma unroll
        for (int k = 0; k < 16 && k_base + k < K; k++) {
            float4 a_val;
            a_val.x = As[tidm * 4 + 0][k / 4].s0;
            a_val.y = As[tidm * 4 + 1][k / 4].s0;
            a_val.z = As[tidm * 4 + 2][k / 4].s0;
            a_val.w = As[tidm * 4 + 3][k / 4].s0;
            
            float4 b_val = Bs[k][tidn];
            
            acc0 = FMA((float4)(a_val.x), b_val, acc0);
            acc1 = FMA((float4)(a_val.y), b_val, acc1);
            acc2 = FMA((float4)(a_val.z), b_val, acc2);
            acc3 = FMA((float4)(a_val.w), b_val, acc3);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write results
    if (row < M && col < N) {
        const int N4 = N / 4;
        if (row + 0 < M) C[(row + 0) * N4 + col / 4] = alpha * acc0 + beta * C[(row + 0) * N4 + col / 4];
        if (row + 1 < M) C[(row + 1) * N4 + col / 4] = alpha * acc1 + beta * C[(row + 1) * N4 + col / 4];
        if (row + 2 < M) C[(row + 2) * N4 + col / 4] = alpha * acc2 + beta * C[(row + 2) * N4 + col / 4];
        if (row + 3 < M) C[(row + 3) * N4 + col / 4] = alpha * acc3 + beta * C[(row + 3) * N4 + col / 4];
    }
}


// ============================================================================
// KERNEL 3: GCN4 OCCUPANCY OPTIMIZED - MAXIMUM WAVEFRONTS
// ============================================================================
/**
 * gemm_gcn4_highoccupancy - Optimized for maximum occupancy
 * 
 * Uses smaller register footprint to allow more wavefronts per CU.
 * Better for smaller matrices or latency-bound scenarios.
 * 
 * Target: 8 wavefronts per SIMD (vs 4 for ultra kernel)
 */
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void gemm_gcn4_highoccupancy(
    const int M, const int N, const int K,
    const float alpha,
    __global const float* restrict A,
    __global const float* restrict B,
    const float beta,
    __global float* restrict C)
{
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    
    // Smaller tiles for lower register pressure
    __local float As[16][17];  // 16x16 + padding
    __local float Bs[16][17];
    
    // Only 4 accumulators per thread (vs 64 in ultra kernel)
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    const int num_tiles = (K + 15) / 16;
    
    for (int t = 0; t < num_tiles; t++) {
        const int k_offset = t * 16;
        
        // Load A
        int a_row = get_group_id(0) * 16 + local_row;
        int a_col = k_offset + local_col;
        As[local_row][local_col] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        
        // Load B
        int b_row = k_offset + local_row;
        int b_col = get_group_id(1) * 16 + local_col;
        Bs[local_row][local_col] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute with 4-way unrolling
        #pragma unroll 4
        for (int k = 0; k < 16; k += 4) {
            acc0 = FMA(As[local_row][k + 0], Bs[k + 0][local_col], acc0);
            acc1 = FMA(As[local_row][k + 1], Bs[k + 1][local_col], acc1);
            acc2 = FMA(As[local_row][k + 2], Bs[k + 2][local_col], acc2);
            acc3 = FMA(As[local_row][k + 3], Bs[k + 3][local_col], acc3);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Sum accumulators and write result
    if (row < M && col < N) {
        float result = acc0 + acc1 + acc2 + acc3;
        C[row * N + col] = alpha * result + beta * C[row * N + col];
    }
}


// ============================================================================
// KERNEL 4: GCN4 STREAMING GEMM - LARGE MATRIX SUPPORT
// ============================================================================
/**
 * gemm_gcn4_streaming - Optimized for very large matrices
 * 
 * Uses streaming access pattern to minimize cache thrashing.
 * Better for matrices that don't fit in L2 cache (>2MB).
 */
__kernel __attribute__((reqd_work_group_size(8, 8, 1)))
void gemm_gcn4_streaming(
    const int M, const int N, const int K,
    const float alpha,
    __global const float* restrict A,
    __global const float* restrict B,
    const float beta,
    __global float* restrict C)
{
    const int tidm = get_local_id(0);
    const int tidn = get_local_id(1);
    const int tid = tidm * 8 + tidn;
    const int gidm = get_group_id(0);
    const int gidn = get_group_id(1);
    
    // Base positions
    const int base_m = gidm * 64;
    const int base_n = gidn * 64;
    
    // Double-buffered LDS
    __local float As[2][64][17];  // Extra for bank conflicts
    __local float Bs[2][16][65];
    
    // 8x8 accumulators per thread
    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    float regA[8], regB[8];
    
    const int num_tiles = (K + 15) / 16;
    int buf = 0;
    
    // Prologue: load first tile
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int row = tid + i * 64 / 16;
        int col = tid % 16;
        if (row < 64 && base_m + row < M && col < K) {
            As[0][row][col] = A[(base_m + row) * K + col];
        } else if (row < 64) {
            As[0][row][col] = 0.0f;
        }
    }
    
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int row = i;
        int col = tid;
        if (row < K && base_n + col < N && col < 64) {
            Bs[0][row][col] = B[row * N + base_n + col];
        } else if (col < 64) {
            Bs[0][row][col] = 0.0f;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Main loop
    for (int t = 0; t < num_tiles; t++) {
        int next_buf = 1 - buf;
        int k_next = (t + 1) * 16;
        
        // Load next tile while computing
        if (t + 1 < num_tiles) {
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int row = tid + i * 64 / 16;
                int col = tid % 16;
                if (row < 64 && base_m + row < M && k_next + col < K) {
                    As[next_buf][row][col] = A[(base_m + row) * K + k_next + col];
                } else if (row < 64) {
                    As[next_buf][row][col] = 0.0f;
                }
            }
            
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                int row = i;
                int col = tid;
                if (k_next + row < K && base_n + col < N && col < 64) {
                    Bs[next_buf][row][col] = B[(k_next + row) * N + base_n + col];
                } else if (col < 64) {
                    Bs[next_buf][row][col] = 0.0f;
                }
            }
        }
        
        // Compute on current tile
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            // Load A and B values
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                regA[i] = As[buf][tidm * 8 + i][k];
                regB[i] = Bs[buf][k][tidn * 8 + i];
            }
            
            // 8x8 rank-1 update
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc[i][j] = FMA(regA[i], regB[j], acc[i][j]);
                }
            }
        }
        
        buf = next_buf;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write results
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int out_row = base_m + tidm * 8 + i;
        if (out_row < M) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int out_col = base_n + tidn * 8 + j;
                if (out_col < N) {
                    int idx = out_row * N + out_col;
                    C[idx] = alpha * acc[i][j] + beta * C[idx];
                }
            }
        }
    }
}
