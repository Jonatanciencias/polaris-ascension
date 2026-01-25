# 🔢 FASE 6 - MATEMÁTICAS: Winograd Transform Matrices para GEMM

**Fecha**: Enero 2026
**Enfoque**: Derivación matemática de transform matrices Winograd para GEMM

---

## 🎯 Transform Matrices Winograd W(2×2, 3×3)

### Para Convolución 2D → Adaptación GEMM

**Convolución Original**:
- Input: 4×4 tile (para output 2×2 con kernel 3×3)
- Kernel: 3×3
- Output: 2×2

**Adaptación GEMM**:
- Interpretamos matrices como "feature maps"
- A: Input matrix (4×4 tile)
- B: Kernel matrix (3×3)
- C: Output matrix (2×2)

### Input Transform Matrix (G)

Para transformar input de dominio espacial a Winograd domain:

```
G = [1,  0, -1,  0,
     0,  1,  1,  0,
     0, -1,  1,  0,
     0,  1,  0, -1]
```

**Aplicación**: `U = G × A × G^T`

Donde:
- A: Input tile 4×4 (flattened to 16×1)
- U: Transformed input 4×4
- G: Transform matrix 4×4

### Kernel Transform Matrix (B_transform)

Para transformar kernel:

```
B^T = [1,  0,  0,
       0.5, 0.5, 0.5,
       0.5, -0.5, 0.5,
       0,  0,  1]
```

**Aplicación**: `V = B^T × B × B_transform`

Donde:
- B: Kernel 3×3
- V: Transformed kernel 4×4

### Output Transform Matrix (A^T)

Para transformar output de vuelta a dominio espacial:

```
A^T = [1, 1, 1, 0,
       0, 1, -1, -1]
```

**Aplicación**: `C = A^T × M × A^T^T`

Donde:
- M: Resultado de U ⊙ V (element-wise multiplication)
- C: Final output 2×2

---

## 🔄 Algoritmo Completo W(2×2, 3×3)

### Paso 1: Input Transform
```
U = G × A × G^T
```
Donde G es la input transform matrix 4×4.

### Paso 2: Kernel Transform
```
V = B^T × B × B_transform
```
Donde B^T y B_transform son las kernel transform matrices.

### Paso 3: Element-wise Multiplication
```
M[i,j] = U[i,j] × V[i,j]
```
Para todos i,j en 0..3

### Paso 4: Output Transform
```
C = A^T × M × A^T^T
```
Donde A^T es la output transform matrix 2×2.

---

## 💻 Implementación OpenCL

### Estructura del Kernel

```c
__kernel void gemm_winograd_w2x2(
    __global float* A,    // Input matrix
    __global float* B,    // Kernel matrix
    __global float* C,    // Output matrix
    int M, int N, int K   // Dimensions
) {
    // Get workgroup and local IDs
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    // Load input tile (4x4) to LDS
    __local float input_tile[4][4];
    __local float kernel_tile[4][4];

    // Input transform
    float U[4][4];
    winograd_input_transform(input_tile, U);

    // Kernel transform
    float V[4][4];
    winograd_kernel_transform(kernel_tile, V);

    // Element-wise multiplication
    float M[4][4];
    for(int i = 0; i < 4; i++) {
        for(int j = 0; j < 4; j++) {
            M[i][j] = U[i][j] * V[i][j];
        }
    }

    // Output transform
    float output_tile[2][2];
    winograd_output_transform(M, output_tile);

    // Store result
    C[gy * N + gx] = output_tile[0][0];
    // ... store other elements
}
```

### Transform Functions

```c
void winograd_input_transform(__local float input[4][4], float U[4][4]) {
    // G * input * G^T
    const float G[4][4] = {
        {1, 0, -1, 0},
        {0, 1, 1, 0},
        {0, -1, 1, 0},
        {0, 1, 0, -1}
    };

    // Matrix multiplication: U = G * input * G^T
    // Implementation here...
}

void winograd_kernel_transform(__local float kernel[3][3], float V[4][4]) {
    // B^T * kernel * B_transform
    const float BT[4][3] = {
        {1, 0, 0},
        {0.5, 0.5, 0.5},
        {0.5, -0.5, 0.5},
        {0, 0, 1}
    };

    // Implementation here...
}

void winograd_output_transform(float M[4][4], float output[2][2]) {
    // A^T * M * A^T^T
    const float AT[2][4] = {
        {1, 1, 1, 0},
        {0, 1, -1, -1}
    };

    // Implementation here...
}
```

---

## 🎯 Optimizaciones GCN 4.0

### LDS Utilization

```c
#define TILE_SIZE 4
#define LDS_SIZE (TILE_SIZE * TILE_SIZE)

__local float lds_input[LDS_SIZE];
__local float lds_kernel[LDS_SIZE];
__local float lds_transform_G[16];    // G matrix 4x4
__local float lds_transform_BT[12];   // B^T matrix 4x3
__local float lds_transform_AT[8];    // A^T matrix 2x4
```

### Memory Layout

- **Input tiles**: Stored coalesced in global memory
- **Transform matrices**: Pre-loaded in LDS at kernel start
- **Intermediate results**: Computed in registers where possible
- **Output tiles**: Written coalesced to global memory

### Workgroup Configuration

```c
// Workgroup size optimized for GCN 4.0
#define WG_SIZE_X 16
#define WG_SIZE_Y 16
#define WG_TOTAL (WG_SIZE_X * WG_SIZE_Y)  // 256 threads

// Each workgroup processes one 2x2 output tile
// Requires 4x4 input tile and 3x3 kernel tile
```

---

## 📊 Complejidad Computacional

### Operaciones por Tile

**Traditional GEMM (4×4 × 3×3)**:
- Multiplicaciones: 4 × 4 × 3 × 3 = 144
- Adiciones: ~144

**Winograd W(2×2, 3×3)**:
- Input transform: 4×4 × 4×4 × 2 = 128 ops
- Kernel transform: 4×3 × 3×3 × 4×3 = 432 ops
- Element-wise mult: 4×4 = 16 ops
- Output transform: 2×4 × 4×4 × 2×4 = 128 ops
- **Total**: ~704 ops

**Speedup**: 144 / 16 = **9x** reducción en multiplicaciones!

### Memory Access

**Traditional**: 144 reads + 16 writes
**Winograd**: ~200 reads + 16 writes (overhead de transforms)

**Net Benefit**: Cuando arithmetic >>> memory latency, Winograd wins.

---

## 🔍 Validación Numérica

### Test Case Simple

**Input A (4×4)**:
```
1 2 3 4
5 6 7 8
9 1 2 3
4 5 6 7
```

**Kernel B (3×3)**:
```
1 0 1
0 1 0
1 0 1
```

**Expected Output C (2×2)**: Calcular con convolución tradicional

### Accuracy Checks

- [ ] Comparar resultados Winograd vs Traditional GEMM
- [ ] Verificar error < 1e-6 para todos los elementos
- [ ] Test con matrices identidad
- [ ] Test con valores extremos (0, inf, nan)

---

## 🚀 Próximos Pasos de Implementación

### Semana 2: Proof of Concept

1. **Implementar transforms básicas**:
   - winograd_input_transform()
   - winograd_kernel_transform()
   - winograd_output_transform()

2. **Crear kernel skeleton**:
   - Memory loading/unloading
   - LDS utilization
   - Workgroup configuration

3. **Validación inicial**:
   - Correctness vs NumPy reference
   - Performance baseline
   - Memory usage analysis

### Optimizaciones Clave

1. **SIMD Vectorization**: Usar float4 para transforms
2. **LDS Banking**: Conflict-free access patterns
3. **Prefetching**: Hide memory latency
4. **Loop Unrolling**: Reduce control flow overhead

---

**Estado**: Transform matrices derivadas y analizadas
**Próximo**: Implementación de funciones transform en OpenCL</content>
<parameter name="filePath">/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/fase_6_winograd/research/winograd_transform_matrices.md