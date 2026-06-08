# Batched Dot Products in NumKong

NumKong implements batched GEMM computing C = A × Bᵀ (packed) and C = A × Aᵀ (symmetric). B is pre-packed once and reused across queries. This is the foundation for the spatials, sets, and maxsim modules.

Packed dot product computes the full cross-product matrix:

$$
C_{ij} = \sum_{k} A_{ik} \cdot B_{jk}^T
$$

Symmetric dot product uses the same matrix for both operands:

$$
C_{ij} = \sum_{k} A_{ik} \cdot A_{jk}
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def dots_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T

def dots_symmetric(a: np.ndarray) -> np.ndarray:
    return a @ a.T
```

## Input & Output Types

| Input Type | Output Type | Description                                    |
| ---------- | ----------- | ---------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision               |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision               |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output |
| `bf16`     | `f32`       | 16-bit brain float, widened output             |
| `e4m3`     | `f32`       | 8-bit Float8: 4 exponent, 3 mantissa bits      |
| `e5m2`     | `f32`       | 8-bit Float8: 5 exponent, 2 mantissa bits      |
| `e2m3`     | `f32`       | 8-bit MX format: 2 exponent, 3 mantissa bits   |
| `e3m2`     | `f32`       | 8-bit MX format: 3 exponent, 2 mantissa bits   |
| `i8`       | `i32`       | 8-bit signed integers                          |
| `u8`       | `u32`       | 8-bit unsigned integers                        |
| `i4`       | `i32`       | 4-bit signed integers, packed nibble pairs     |
| `u4`       | `u32`       | 4-bit unsigned integers, packed nibble pairs   |
| `u1`       | `u32`       | 1-bit binary packed octets, popcount of AND    |

## Optimizations

### B Matrix Pre-Packing with Stride Breaking

`nk_dots_pack_f32_serial`, `nk_dots_pack_f32_haswell`, `nk_dots_pack_bf16_haswell`, `nk_dots_pack_i8_haswell` pre-pack the B matrix into a contiguous buffer optimized for streaming access during GEMM.
Power-of-2 stride detection — when `stride_bytes & (stride_bytes - 1) == 0` — adds `depth_simd_dimensions` padding to avoid cache associativity conflicts on set-associative caches.
Type conversion is amortized into the pack step: BFloat16 → Float32, Float16 → Float32, and Float8 → Float32 conversions happen once during packing instead of per-row during GEMM.
A 64-byte header stores metadata: column count, depth dimensions, and padded depth.
Row grouping (`group_size=16`) zero-pads partial groups at matrix edges for uniform SIMD processing.

### Tiled Register Accumulation

`nk_dots_packed_f32_haswell`, `nk_dots_packed_f32_skylake`, `nk_dots_packed_f32_neon` use a 4×4 tile kernel with 16 accumulators to handle ~80% of the work.
A 1×8 tile kernel with 8 accumulators handles edge rows that don't fill a full 4-row tile.
No depth blocking is used — the kernel relies on hardware prefetch for streaming A/B access patterns.
Row loads are amortized across multiple dot products: each A row is loaded once and multiplied against 4 B columns per tile pass.

### AMX 2D Tile Engine

The Sapphire Rapids AMX backends for `bf16`, mini-floats, `i8`, and `u8` use Intel AMX's 8 tile registers (TMM0–TMM7), each 1 KB (16 rows × 64 bytes).
Convention: TMM0–1 hold A tiles, TMM2–3 hold B tiles, TMM4–7 are C accumulators — giving a 2×2 output tile (32×32 Float32 results) per tile pass.
`TDPBF16PS tmm_c, tmm_a, tmm_b` performs a 16×16 outer product with 32 BFloat16 multiply-adds per cell (16×16×32 = 8,192 MACs per instruction).
Each A row contains 16 BFloat16 pairs interleaved as [a₀, a₁, a₀, a₁, ...] and B columns as [b₀, b₁, b₀, b₁, ...] — the hardware consumes two BFloat16 elements per slot, accumulating into Float32.
`TDPBSSD tmm_c, tmm_a, tmm_b` does the same for Int8: 64 bytes per row gives 16×16×64 = 16,384 Int8 MACs per instruction.
Int8 data is quad-interleaved: [a₀, a₁, a₂, a₃, a₀, a₁, a₂, a₃, ...] so the hardware can consume four Int8 elements per 32-bit slot.
Tile configuration via `LDTILECFG` sets row counts and column byte-widths per tile — allows undersized tiles at matrix edges without masking.
Morton Z-curve ordering for tile traversal improves cache reuse when both A and B exceed L2.
This eliminates the explicit M×N×K loop nesting and register file pressure of vector ISAs — the entire dot-product reduction happens inside the tile instruction.
FP8 inputs on Sapphire AMX go through an on-the-fly E4M3/E5M2 → BF16 pack via the Ice Lake `VPERMI2W` LUT helpers — port-5-bound but the simplest correct route to feed `TDPBF16PS` tiles.
Granite Rapids adds `TDPFP16PS` (same tile shape, FP16 operands); the E5M2 variant widens inputs with a single `VPUNPCK*BW` against zero into FP16 tiles at pack time and then reuses the native FP16 compute loop — keeps the intermediate at FP16 precision instead of truncating to BF16 like the Sapphire path.

### SME Outer-Product Streaming

`nk_dots_packed_f32_smef64`, `nk_dots_packed_bf16_sme`, `nk_dots_packed_f64_smef64` use Arm's SME ZA tile array (up to 4 named tiles ZA0–ZA3 in 32-bit mode, each SVL×SVL elements).
`FMOPA za, pn/m, pm/m, zn.s, zm.s` computes a full SVL×SVL rank-1 update in one instruction — one row of A times one row of B, accumulated into ZA.
ZA0 time-shares between data staging and accumulation: A rows are loaded horizontally into ZA0 (`st1w {za0h.s[ws]}, ...`), then read vertically (`svread_ver_za32_f32_m`) to produce transposed column vectors for B.
This avoids explicit transpose operations — the tile's 2D addressing provides free transposition.
ZA1–ZA3 serve as accumulators while ZA0 stages the next data.
A 3-column-tile fast path handles B column count ≤ 3×SVL using ZA1–ZA3 as three separate accumulator tiles, avoiding spill/reload cycles.
For wider B, the kernel falls back to multi-pass accumulation with ZA store/load between passes.
`BFMOPA` for BFloat16 uses the same outer-product pattern but with BFloat16 → Float32 widening — 2× the depth per instruction vs Float32 `FMOPA`.
`SMSTART`/`SMSTOP` streaming mode transitions cost ~50–100 cycles, amortized across the full M×N output.
Ozaki splitting for Float64 (`nk_dots_packed_f64_smef64`) splits each Float64 into 3 mantissa-masked Float32 slices, computes 6 FMOPAs (all cross-products of 3×2 slices) into 3 ZA accumulators, then reconstructs the Float64 result — achieving Float64 precision using Float32 tile hardware.

### Compensated Integer GEMM

`nk_dots_packed_i8_icelake`, `nk_dots_packed_u8_icelake`, `nk_dots_packed_i8_haswell` work around the unsigned×signed operand requirement of integer dot-product instructions.
`VPDPBUSD` (Ice Lake+) computes UInt8×Int8 dot products accumulating directly to Int32 — but requires one unsigned and one signed operand.
For signed×signed (Int8×Int8), one operand is XOR'd with `0x80` to shift to unsigned range, introducing a bias of $128 \cdot \sum_k b_k$ per output element.
Rather than computing the bias correction per-element inside the inner loop (requiring extra registers for running sums), the B column sums $\sum_k b_k$ are pre-computed once during packing and stored in the packed buffer metadata.
The inner loop only needs the `VPDPBUSD` accumulator — the bias subtraction is a single post-loop correction: `result[i][j] -= 128 * b_column_sum[j]`.
This reduces per-accumulator state from 2 registers (dot + running sum) to 1 register (dot only), freeing registers for more accumulators in the 4×4 tile.
Haswell fallback uses `VPMADDUBSW` (UInt8×Int8→Int16) + `VPMADDWD` (Int16→Int32), a two-instruction chain with Int16 intermediate overflow risk — quantization ranges must be tighter ([-79, 79] vs [-127, 127]).

### 4-Way Finalizer Amortization

All packed and symmetric kernels across the dots, spatials, and sets modules share a finalizer-based design.
The 4×4 tile accumulates 16 dot products in registers, then stores results 4-wide via `nk_b128_vec_t` — a union of `f32[4]`, `i32[4]`, `u32[4]` fitting a 128-bit register.
A finalizer function pointer processes 4 results simultaneously, amortizing horizontal reductions and type conversions:

```
// 4-wide finalizer signature — per-lane arrays always pass as `nk_bXXX_vec_t const *`
void finalizer(nk_b128_vec_t const *dots_vec,          // 4 dot products
               nk_f32_t query_norm,                    // precomputed query squared-norm (scalar)
               nk_b128_vec_t const *target_norms_vec,  // 4 target squared-norms
               nk_b128_vec_t *result_vec)              // 4 output distances

// Angular: 4 divisions + 4 subtractions in one call
result_vec->f32s[i] = 1 - dots_vec->f32s[i] / sqrt(query_norm * target_norms_vec->f32s[i])

// Euclidean: 4 sqrt(a² + b² - 2ab) in one call
result_vec->f32s[i] = sqrt(query_norm + target_norms_vec->f32s[i] - 2 * dots_vec->f32s[i])
```

The 4×4 tile emits 4 rows of 4 results each — the finalizer is called 4 times per tile, once per query row.
For the 1×8 edge tile, two finalizer calls handle 8 results.
This design decouples the GEMM loop from the distance metric: the same tiled accumulation code serves dots, spatials, and sets by swapping only the finalizer function pointer.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by `NK_MATRIX_HEIGHT`, `NK_MATRIX_WIDTH`, and `NK_MATRIX_DEPTH` environment variables, all set to the same value for products of two square matrices.
Columns show throughput for 256³, 1024³, and 4096³ matrix products.
The throughput is measured in GSO/s as Giga Scalar Operations per Second, with `ops = 2 · M · N · K` arithmetic complexity for an M × K by K × N product.
Accuracy is reported as mean ULP (units in last place) unless noted otherwise — the average number of representable floating-point values between the result and the exact answer.
Rows marked `🧩` use external BLAS or MKL baselines rather than NumKong kernels.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                               |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_f64_with_blas` 🧩        |       58.7 gso/s, 16 ulp |       73.1 gso/s, 58 ulp |     73.8 gso/s, 56.2 ulp |
| `dots_packed_f64_with_mkl` 🧩         |       59.9 gso/s, 16 ulp |       73.7 gso/s, 58 ulp |     73.3 gso/s, 56.2 ulp |
| `dots_symmetric_f64_with_blas` 🧩     |       50.8 gso/s, 13 ulp |       70.4 gso/s, 30 ulp |       74 gso/s, 50.8 ulp |
| `nk_dots_packed_f64_serial`          |       0.850 gso/s, 2 ulp |     0.846 gso/s, 4.6 ulp |     0.862 gso/s, 5.9 ulp |
| `nk_dots_symmetric_f64_serial`       |       0.484 gso/s, 2 ulp |     0.472 gso/s, 2.9 ulp |     0.471 gso/s, 3.9 ulp |
| `nk_dots_packed_f64_haswell`         |        5.93 gso/s, 0 ulp |        6.11 gso/s, 0 ulp |        6.16 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_haswell`      |        5.68 gso/s, 0 ulp |        5.99 gso/s, 0 ulp |        5.86 gso/s, 0 ulp |
| `nk_dots_packed_f64_skylake`         |        8.26 gso/s, 0 ulp |        9.27 gso/s, 0 ulp |        9.06 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_skylake`      |        7.53 gso/s, 0 ulp |        8.63 gso/s, 0 ulp |        8.58 gso/s, 0 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_f32_with_blas` 🧩        |        113 gso/s, 18 ulp |        139 gso/s, 30 ulp |       147 gso/s, 267 ulp |
| `dots_symmetric_f32_with_blas` 🧩     |       94.5 gso/s, 23 ulp |        126 gso/s, 39 ulp |       146 gso/s, 260 ulp |
| `nk_dots_packed_f32_serial`          |      9.98 gso/s, 5.3 ulp |     10.1 gso/s, 11.8 ulp |     10.1 gso/s, 14.5 ulp |
| `nk_dots_symmetric_f32_serial`       |     4.96 gso/s, 11.1 ulp |     5.01 gso/s, 13.4 ulp |     5.01 gso/s, 14.1 ulp |
| `nk_dots_packed_f32_haswell`         |        30.4 gso/s, 0 ulp |        32.5 gso/s, 0 ulp |        31.9 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_haswell`      |        15.5 gso/s, 0 ulp |        17.9 gso/s, 0 ulp |        18.4 gso/s, 0 ulp |
| `nk_dots_packed_f32_skylake`         |        35.4 gso/s, 0 ulp |        41.4 gso/s, 0 ulp |        40.0 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_skylake`      |        22.4 gso/s, 0 ulp |        28.2 gso/s, 0 ulp |        28.1 gso/s, 0 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_bf16_with_mkl` 🧩        |         182 gso/s, 0 ulp |       523 gso/s, 0.7 ulp |       847 gso/s, 5.8 ulp |
| `nk_dots_packed_bf16_serial`         |        1.20 gso/s, 0 ulp |      1.21 gso/s, 0.5 ulp |      1.22 gso/s, 5.4 ulp |
| `nk_dots_symmetric_bf16_serial`      |        1.16 gso/s, 0 ulp |      1.19 gso/s, 0.9 ulp |      1.18 gso/s, 5.4 ulp |
| `nk_dots_packed_bf16_haswell`        |        65.6 gso/s, 0 ulp |      73.3 gso/s, 0.3 ulp |      76.8 gso/s, 4.4 ulp |
| `nk_dots_symmetric_bf16_haswell`     |        40.2 gso/s, 0 ulp |      55.6 gso/s, 0.5 ulp |      60.8 gso/s, 4.6 ulp |
| `nk_dots_packed_bf16_skylake`        |        79.8 gso/s, 0 ulp |      92.1 gso/s, 0.3 ulp |       102 gso/s, 3.5 ulp |
| `nk_dots_symmetric_bf16_skylake`     |        57.4 gso/s, 0 ulp |      78.9 gso/s, 0.5 ulp |      82.5 gso/s, 3.5 ulp |
| `nk_dots_packed_bf16_genoa`          |        65.8 gso/s, 0 ulp |      83.2 gso/s, 0.3 ulp |      88.9 gso/s, 3.5 ulp |
| `nk_dots_symmetric_bf16_genoa`       |        52.5 gso/s, 0 ulp |      70.5 gso/s, 0.5 ulp |      76.0 gso/s, 3.5 ulp |
| `nk_dots_packed_bf16_sapphireamx`    |         348 gso/s, 0 ulp |       706 gso/s, 0.7 ulp |       667 gso/s, 5.8 ulp |
| `nk_dots_symmetric_bf16_sapphireamx` |        84.2 gso/s, 0 ulp |       120 gso/s, 0.5 ulp |       120 gso/s, 5.8 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_f16_with_mkl` 🧩         |        123 gso/s, 17 ulp |        138 gso/s, 31 ulp |      138 gso/s, 39.5 ulp |
| `nk_dots_packed_f16_serial`          |       8.19 gso/s, 14 ulp |       8.21 gso/s, 40 ulp |      8.11 gso/s, 326 ulp |
| `nk_dots_symmetric_f16_serial`       |      4.02 gso/s, 8.9 ulp |       4.04 gso/s, 25 ulp |     4.03 gso/s, 55.6 ulp |
| `nk_dots_packed_f16_haswell`         |       65.1 gso/s, 12 ulp |       74.4 gso/s, 22 ulp |      71.5 gso/s, 374 ulp |
| `nk_dots_symmetric_f16_haswell`      |      34.4 gso/s, 7.7 ulp |       44.0 gso/s, 32 ulp |      46.5 gso/s, 486 ulp |
| `nk_dots_packed_f16_skylake`         |      74.7 gso/s, 7.3 ulp |       99.0 gso/s, 21 ulp |      94.0 gso/s, 138 ulp |
| `nk_dots_symmetric_f16_skylake`      |      40.9 gso/s, 5.9 ulp |       56.8 gso/s, 25 ulp |       58.8 gso/s, 32 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`         |        4.86 gso/s, 0 ulp |        4.75 gso/s, 0 ulp |        4.88 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`      |        3.97 gso/s, 0 ulp |        4.28 gso/s, 0 ulp |        4.50 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_haswell`        |        29.1 gso/s, 0 ulp |        31.5 gso/s, 0 ulp |        30.6 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_haswell`     |        15.6 gso/s, 0 ulp |        16.4 gso/s, 0 ulp |        17.0 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_skylake`        |        34.6 gso/s, 0 ulp |        37.9 gso/s, 0 ulp |        38.9 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_skylake`     |        21.2 gso/s, 0 ulp |        22.7 gso/s, 0 ulp |        22.5 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_genoa`          |        41.7 gso/s, 0 ulp |        48.7 gso/s, 0 ulp |        49.1 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_genoa`       |        30.0 gso/s, 0 ulp |        33.3 gso/s, 0 ulp |        33.7 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_sapphireamx`    |         254 gso/s, 0 ulp |         407 gso/s, 0 ulp |         419 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_sapphireamx` |        50.9 gso/s, 0 ulp |        69.9 gso/s, 0 ulp |        67.4 gso/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.489 gso/s, 0 ulp |       0.499 gso/s, 0 ulp |       0.489 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.394 gso/s, 0 ulp |       0.390 gso/s, 0 ulp |       0.391 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_haswell`        |        24.3 gso/s, 0 ulp |        26.1 gso/s, 0 ulp |        25.2 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_haswell`     |        13.5 gso/s, 0 ulp |        14.0 gso/s, 0 ulp |        14.3 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_skylake`        |        31.6 gso/s, 0 ulp |        32.6 gso/s, 0 ulp |        34.0 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_skylake`     |        17.3 gso/s, 0 ulp |        18.2 gso/s, 0 ulp |        18.6 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_genoa`          |        38.6 gso/s, 0 ulp |        43.8 gso/s, 0 ulp |        43.7 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_genoa`       |        27.3 gso/s, 0 ulp |        29.4 gso/s, 0 ulp |        29.2 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_sapphireamx`    |         222 gso/s, 0 ulp |         333 gso/s, 0 ulp |         332 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_sapphireamx` |        33.1 gso/s, 0 ulp |        36.3 gso/s, 0 ulp |        35.4 gso/s, 0 ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`         |        4.97 gso/s, 0 ulp |        4.90 gso/s, 0 ulp |        5.03 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_serial`      |        3.40 gso/s, 0 ulp |        3.81 gso/s, 0 ulp |        3.88 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_haswell`        |        31.0 gso/s, 0 ulp |        32.2 gso/s, 0 ulp |        33.9 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_haswell`     |        29.0 gso/s, 0 ulp |        31.7 gso/s, 0 ulp |        31.1 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_skylake`        |        39.3 gso/s, 0 ulp |        43.4 gso/s, 0 ulp |        44.1 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_skylake`     |        40.0 gso/s, 0 ulp |        46.6 gso/s, 0 ulp |        47.1 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_sapphireamx`    |         263 gso/s, 0 ulp |         471 gso/s, 0 ulp |         471 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_sapphireamx` |        62.9 gso/s, 0 ulp |         101 gso/s, 0 ulp |        89.1 gso/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`         |        4.98 gso/s, 0 ulp |        4.95 gso/s, 0 ulp |        5.00 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`      |        3.48 gso/s, 0 ulp |        3.83 gso/s, 0 ulp |        3.85 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_haswell`        |        58.6 gso/s, 0 ulp |        62.5 gso/s, 0 ulp |        65.3 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_haswell`     |        50.5 gso/s, 0 ulp |        61.2 gso/s, 0 ulp |        64.2 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_skylake`        |        69.8 gso/s, 0 ulp |        81.8 gso/s, 0 ulp |        88.4 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_skylake`     |        65.5 gso/s, 0 ulp |        83.4 gso/s, 0 ulp |        84.6 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_sapphireamx`    |         419 gso/s, 0 ulp |       1,195 gso/s, 0 ulp |       1,067 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_sapphireamx` |        94.5 gso/s, 0 ulp |         213 gso/s, 0 ulp |         184 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_alder`          |        72.9 gso/s, 0 ulp |        78.6 gso/s, 0 ulp |        85.7 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_alder`       |        61.6 gso/s, 0 ulp |        75.2 gso/s, 0 ulp |        54.9 gso/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_i8u8_with_mkl` 🧩        |                250 gso/s |                627 gso/s |              1,670 gso/s |
| `nk_dots_packed_i8_serial`           |               6.44 gso/s |               6.62 gso/s |               7.44 gso/s |
| `nk_dots_symmetric_i8_serial`        |               2.93 gso/s |               2.99 gso/s |               5.83 gso/s |
| `nk_dots_packed_i8_haswell`          |               87.7 gso/s |                104 gso/s |                108 gso/s |
| `nk_dots_symmetric_i8_haswell`       |                 64 gso/s |               80.9 gso/s |                173 gso/s |
| `nk_dots_packed_i8_icelake`          |                191 gso/s |                326 gso/s |                410 gso/s |
| `nk_dots_symmetric_i8_icelake`       |               79.2 gso/s |                303 gso/s |                760 gso/s |
| `nk_dots_packed_i8_sapphireamx`      |                547 gso/s |              1,610 gso/s |              1,300 gso/s |
| `nk_dots_symmetric_i8_sapphireamx`   |                112 gso/s |                266 gso/s |                221 gso/s |
| `nk_dots_packed_i8_alder`            |                180 gso/s |                229 gso/s |                270 gso/s |
| `nk_dots_symmetric_i8_alder`         |                108 gso/s |                218 gso/s |                263 gso/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               7.45 gso/s |               7.79 gso/s |               7.88 gso/s |
| `nk_dots_symmetric_u8_serial`        |               2.81 gso/s |               2.91 gso/s |               5.35 gso/s |
| `nk_dots_packed_u8_haswell`          |                 88 gso/s |                102 gso/s |                107 gso/s |
| `nk_dots_symmetric_u8_haswell`       |               64.3 gso/s |               79.8 gso/s |                181 gso/s |
| `nk_dots_packed_u8_icelake`          |                194 gso/s |                329 gso/s |                402 gso/s |
| `nk_dots_symmetric_u8_icelake`       |               83.9 gso/s |                300 gso/s |                755 gso/s |
| `nk_dots_packed_u8_sapphireamx`      |                550 gso/s |              1,680 gso/s |              1,330 gso/s |
| `nk_dots_symmetric_u8_sapphireamx`   |                113 gso/s |                270 gso/s |                223 gso/s |
| `nk_dots_packed_u8_alder`            |                181 gso/s |                230 gso/s |                266 gso/s |
| `nk_dots_symmetric_u8_alder`         |                108 gso/s |                216 gso/s |                257 gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               2.43 gso/s |               2.43 gso/s |               2.24 gso/s |
| `nk_dots_symmetric_i4_serial`        |               2.26 gso/s |               2.13 gso/s |               4.44 gso/s |
| `nk_dots_packed_i4_icelake`          |                135 gso/s |                211 gso/s |                254 gso/s |
| `nk_dots_symmetric_i4_icelake`       |               78.7 gso/s |                252 gso/s |                581 gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               3.27 gso/s |               3.37 gso/s |               3.33 gso/s |
| `nk_dots_symmetric_u4_serial`        |               3.02 gso/s |               3.06 gso/s |               6.13 gso/s |
| `nk_dots_packed_u4_icelake`          |                152 gso/s |                302 gso/s |                387 gso/s |
| `nk_dots_symmetric_u4_icelake`       |               97.3 gso/s |                311 gso/s |                697 gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_haswell`          |                225 gso/s |                261 gso/s |                344 gso/s |
| `nk_dots_symmetric_u1_haswell`       |                122 gso/s |                277 gso/s |                756 gso/s |
| `nk_dots_packed_u1_icelake`          |                196 gso/s |                750 gso/s |              1,390 gso/s |
| `nk_dots_symmetric_u1_icelake`       |                171 gso/s |                661 gso/s |              2,500 gso/s |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                               |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`          |     0.947 gso/s, 3.4 ulp |     0.969 gso/s, 2.4 ulp |       0.969 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_serial`       |     0.957 gso/s, 3.7 ulp |      1.11 gso/s, 2.5 ulp |        1.16 gso/s, 0 ulp |
| `nk_dots_packed_f64_v128relaxed`     |     2.73 gso/s, 23.6 ulp |     2.79 gso/s, 32.5 ulp |      2.81 gso/s, 3.9 ulp |
| `nk_dots_symmetric_f64_v128relaxed`  |     2.01 gso/s, 21.6 ulp |     2.55 gso/s, 41.2 ulp |      2.77 gso/s, 2.9 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`          |     4.27 gso/s, 14.6 ulp |     4.35 gso/s, 28.6 ulp |     4.47 gso/s, 25.3 ulp |
| `nk_dots_symmetric_f32_serial`       |     3.13 gso/s, 11.5 ulp |     5.09 gso/s, 34.8 ulp |     5.78 gso/s, 44.7 ulp |
| `nk_dots_packed_f32_v128relaxed`     |     10.4 gso/s, 12.9 ulp |     10.6 gso/s, 26.5 ulp |     10.9 gso/s, 39.7 ulp |
| `nk_dots_symmetric_f32_v128relaxed`  |     3.73 gso/s, 10.3 ulp |     6.27 gso/s, 28.6 ulp |     7.43 gso/s, 76.2 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`         |        4.33 gso/s, 0 ulp |      4.46 gso/s, 0.4 ulp |      4.45 gso/s, 9.5 ulp |
| `nk_dots_symmetric_bf16_serial`      |        3.76 gso/s, 0 ulp |      6.36 gso/s, 0.5 ulp |      7.43 gso/s, 4.9 ulp |
| `nk_dots_packed_bf16_v128relaxed`    |        23.2 gso/s, 0 ulp |      24.5 gso/s, 0.4 ulp |      24.9 gso/s, 6.8 ulp |
| `nk_dots_symmetric_bf16_v128relaxed` |        4.92 gso/s, 0 ulp |      10.5 gso/s, 0.5 ulp |      13.7 gso/s, 4.9 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_serial`          |       4.33 gso/s, 26 ulp |       4.46 gso/s, 26 ulp |       4.45 gso/s, 26 ulp |
| `nk_dots_symmetric_f16_serial`       |       3.76 gso/s, 28 ulp |       6.36 gso/s, 28 ulp |       7.43 gso/s, 28 ulp |
| `nk_dots_packed_f16_v128relaxed`     |       7.39 gso/s, 27 ulp |       7.36 gso/s, 27 ulp |       7.45 gso/s, 27 ulp |
| `nk_dots_symmetric_f16_v128relaxed`  |       3.70 gso/s, 28 ulp |       3.83 gso/s, 28 ulp |       3.87 gso/s, 28 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`         |        2.63 gso/s, 0 ulp |        2.69 gso/s, 0 ulp |        2.70 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`      |        1.62 gso/s, 0 ulp |        2.04 gso/s, 0 ulp |        2.16 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_v128relaxed`    |        6.25 gso/s, 0 ulp |        6.50 gso/s, 0 ulp |        6.55 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_v128relaxed` |        3.37 gso/s, 0 ulp |        5.23 gso/s, 0 ulp |        6.06 gso/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.348 gso/s, 0 ulp |       0.345 gso/s, 0 ulp |       0.345 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.321 gso/s, 0 ulp |       0.340 gso/s, 0 ulp |       0.345 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_v128relaxed`    |        4.80 gso/s, 0 ulp |        4.92 gso/s, 0 ulp |        4.96 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_v128relaxed` |        2.85 gso/s, 0 ulp |        4.17 gso/s, 0 ulp |        4.62 gso/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`         |        2.63 gso/s, 0 ulp |        2.69 gso/s, 0 ulp |        2.71 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`      |        1.62 gso/s, 0 ulp |        2.06 gso/s, 0 ulp |        2.14 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_v128relaxed`    |        17.2 gso/s, 0 ulp |        18.2 gso/s, 0 ulp |        18.7 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_v128relaxed` |        5.35 gso/s, 0 ulp |        11.6 gso/s, 0 ulp |        16.3 gso/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`           |               4.40 gso/s |               4.54 gso/s |               4.73 gso/s |
| `nk_dots_symmetric_i8_serial`        |               2.74 gso/s |               3.89 gso/s |               4.29 gso/s |
| `nk_dots_packed_i8_v128relaxed`      |               36.5 gso/s |               38.5 gso/s |               41.1 gso/s |
| `nk_dots_symmetric_i8_v128relaxed`   |               29.2 gso/s |               36.3 gso/s |               39.2 gso/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               4.94 gso/s |               5.14 gso/s |               4.88 gso/s |
| `nk_dots_symmetric_u8_serial`        |               2.74 gso/s |               3.94 gso/s |               4.40 gso/s |
| `nk_dots_packed_u8_v128relaxed`      |               35.2 gso/s |               37.7 gso/s |               40.5 gso/s |
| `nk_dots_symmetric_u8_v128relaxed`   |               21.0 gso/s |               26.6 gso/s |               28.6 gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               6.34 gso/s |               6.40 gso/s |               6.59 gso/s |
| `nk_dots_symmetric_i4_serial`        |               2.70 gso/s |               3.76 gso/s |               4.13 gso/s |
| `nk_dots_packed_i4_v128relaxed`      |               9.81 gso/s |               10.3 gso/s |               10.4 gso/s |
| `nk_dots_symmetric_i4_v128relaxed`   |               4.95 gso/s |               15.6 gso/s |               32.8 gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               5.61 gso/s |               5.76 gso/s |               5.79 gso/s |
| `nk_dots_symmetric_u4_serial`        |               3.01 gso/s |               4.34 gso/s |               4.94 gso/s |
| `nk_dots_packed_u4_v128relaxed`      |               58.6 gso/s |               71.0 gso/s |               76.5 gso/s |
| `nk_dots_symmetric_u4_v128relaxed`   |               6.97 gso/s |               21.9 gso/s |               46.7 gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`           |               96.2 gso/s |                143 gso/s |                151 gso/s |
| `nk_dots_packed_u1_v128relaxed`      |                166 gso/s |                280 gso/s |                294 gso/s |
| `nk_dots_symmetric_u1_serial`        |               7.42 gso/s |               27.9 gso/s |               87.3 gso/s |
| `nk_dots_symmetric_u1_v128relaxed`   |               7.35 gso/s |               27.5 gso/s |               81.9 gso/s |

### Apple M5

#### Native

| Kernel                             |                     256³ |                    1024³ |                    4096³ |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`        |        2.49 gso/s, 3 ulp |        2.36 gso/s, 5 ulp |        2.48 gso/s, 6 ulp |
| `nk_dots_symmetric_f64_serial`     |        1.38 gso/s, 0 ulp |        1.36 gso/s, 0 ulp |        1.49 gso/s, 0 ulp |
| `nk_dots_packed_f64_neon`          |        6.31 gso/s, 0 ulp |        6.00 gso/s, 0 ulp |        6.34 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_neon`       |        5.57 gso/s, 0 ulp |        5.41 gso/s, 0 ulp |        5.40 gso/s, 0 ulp |
| `nk_dots_packed_f64_smef64`        |      45.9 gso/s, 1.5 ulp |      46.3 gso/s, 1.1 ulp |      46.2 gso/s, 0.9 ulp |
| `nk_dots_symmetric_f64_smef64`     |      22.5 gso/s, 1.5 ulp |      24.3 gso/s, 1.2 ulp |      21.3 gso/s, 1.1 ulp |
| __f32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`        |       12.0 gso/s, 19 ulp |       11.4 gso/s, 30 ulp |      12.2 gso/s, 725 ulp |
| `nk_dots_symmetric_f32_serial`     |      8.75 gso/s, 3.1 ulp |     9.15 gso/s, 12.8 ulp |     9.62 gso/s, 39.9 ulp |
| `nk_dots_packed_f32_neon`          |        42.5 gso/s, 0 ulp |        40.6 gso/s, 0 ulp |        42.0 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_neon`       |      10.9 gso/s, 4.6 ulp |     10.5 gso/s, 17.7 ulp |       10.8 gso/s, 59 ulp |
| `nk_dots_packed_f32_smef64`        |         236 gso/s, 0 ulp |        268 gso/s, 15 ulp |         221 gso/s, 0 ulp |
| `nk_dots_symmetric_f32_smef64`     |      78.1 gso/s, 4.3 ulp |     94.1 gso/s, 19.0 ulp |        55.3 gso/s, 0 ulp |
| __bf16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`       |      20.4 gso/s, 0.1 ulp |      19.6 gso/s, 0.5 ulp |        20.3 gso/s, 5 ulp |
| `nk_dots_symmetric_bf16_serial`    |     16.3 gso/s, 0.01 ulp |      16.9 gso/s, 0.7 ulp |      17.8 gso/s, 115 ulp |
| `nk_dots_packed_bf16_neon`         |        83.0 gso/s, 0 ulp |        80.2 gso/s, 0 ulp |        84.0 gso/s, 0 ulp |
| `nk_dots_symmetric_bf16_neon`      |        39.5 gso/s, 0 ulp |        41.2 gso/s, 0 ulp |        41.9 gso/s, 0 ulp |
| `nk_dots_packed_bf16_neonbfdot`    |        57.9 gso/s, 0 ulp |      58.5 gso/s, 0.5 ulp |      63.4 gso/s, 7.2 ulp |
| `nk_dots_symmetric_bf16_neonbfdot` |        38.6 gso/s, 0 ulp |      41.1 gso/s, 0.5 ulp |        43.5 gso/s, 0 ulp |
| `nk_dots_packed_bf16_sme`          |       1,106 gso/s, 0 ulp |     1,208 gso/s, 4.2 ulp |     1,190 gso/s, 3.8 ulp |
| `nk_dots_symmetric_bf16_sme`       |      606 gso/s, 0.07 ulp |       650 gso/s, 1.2 ulp |       458 gso/s, 1.8 ulp |
| __f16__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_serial`        |      14.8 gso/s, 204 ulp |       14.2 gso/s, 36 ulp |      14.8 gso/s, 326 ulp |
| `nk_dots_symmetric_f16_serial`     |       24.3 gso/s, 13 ulp |     24.9 gso/s, 24.6 ulp |      26.7 gso/s, 506 ulp |
| `nk_dots_packed_f16_neonhalf`      |     77.0 gso/s, 16.8 ulp |     79.1 gso/s, 25.5 ulp |      84.2 gso/s, 618 ulp |
| `nk_dots_symmetric_f16_neonhalf`   |     20.5 gso/s, 12.1 ulp |     20.4 gso/s, 25.0 ulp |      22.5 gso/s, 506 ulp |
| `nk_dots_packed_f16_neonfhm`       |      104 gso/s, 16.7 ulp |      110 gso/s, 25.5 ulp |       118 gso/s, 618 ulp |
| `nk_dots_symmetric_f16_neonfhm`    |     34.5 gso/s, 12.1 ulp |     40.4 gso/s, 25.0 ulp |      41.5 gso/s, 506 ulp |
| `nk_dots_packed_f16_sme`           |    1,106 gso/s, 14.8 ulp |    1,213 gso/s, 28.2 ulp |    1,190 gso/s, 28.2 ulp |
| `nk_dots_symmetric_f16_sme`        |      607 gso/s, 12.1 ulp |      636 gso/s, 23.8 ulp |      458 gso/s, 24.4 ulp |
| __e5m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`       |        15.9 gso/s, 0 ulp |        16.7 gso/s, 0 ulp |        17.2 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`    |        7.56 gso/s, 0 ulp |        8.37 gso/s, 0 ulp |        8.99 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_neonfhm`      |        88.1 gso/s, 0 ulp |        97.3 gso/s, 0 ulp |         103 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_neonfhm`   |        61.0 gso/s, 0 ulp |        73.2 gso/s, 0 ulp |        79.3 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_sme`          |         729 gso/s, 0 ulp |         800 gso/s, 0 ulp |         792 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_sme`       |         208 gso/s, 0 ulp |         227 gso/s, 0 ulp |         229 gso/s, 0 ulp |
| __e4m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`       |        1.24 gso/s, 0 ulp |        1.20 gso/s, 0 ulp |        1.24 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`    |        1.20 gso/s, 0 ulp |        1.24 gso/s, 0 ulp |        1.32 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_neonfhm`      |        29.6 gso/s, 0 ulp |        32.2 gso/s, 0 ulp |        34.1 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_neonfhm`   |        32.0 gso/s, 0 ulp |        36.6 gso/s, 0 ulp |        38.9 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_sme`          |         284 gso/s, 0 ulp |         314 gso/s, 0 ulp |         316 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_sme`       |        74.3 gso/s, 0 ulp |        80.9 gso/s, 0 ulp |        77.8 gso/s, 0 ulp |
| __e3m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`       |        14.0 gso/s, 0 ulp |        14.6 gso/s, 0 ulp |        15.5 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_serial`    |        7.51 gso/s, 0 ulp |        8.10 gso/s, 0 ulp |        9.05 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_sme`          |         671 gso/s, 0 ulp |         738 gso/s, 0 ulp |         730 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_sme`       |         191 gso/s, 0 ulp |         206 gso/s, 0 ulp |         207 gso/s, 0 ulp |
| __e2m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`       |        14.4 gso/s, 0 ulp |        14.8 gso/s, 0 ulp |        15.5 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`    |        7.58 gso/s, 0 ulp |        8.21 gso/s, 0 ulp |        9.09 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_sme`          |       1,211 gso/s, 0 ulp |       1,404 gso/s, 0 ulp |       1,313 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_sme`       |         372 gso/s, 0 ulp |         410 gso/s, 0 ulp |         416 gso/s, 0 ulp |
| __i8__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`         |               18.9 gso/s |               20.0 gso/s |               20.2 gso/s |
| `nk_dots_symmetric_i8_serial`      |               12.6 gso/s |               13.9 gso/s |               14.8 gso/s |
| `nk_dots_packed_i8_neonsdot`       |                345 gso/s |                419 gso/s |                477 gso/s |
| `nk_dots_symmetric_i8_neonsdot`    |               76.6 gso/s |               86.9 gso/s |               87.2 gso/s |
| `nk_dots_packed_i8_sme`            |              2,348 gso/s |              2,687 gso/s |              2,570 gso/s |
| `nk_dots_symmetric_i8_sme`         |              1,390 gso/s |              1,531 gso/s |              1,369 gso/s |
| __u8__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`         |               16.3 gso/s |               16.3 gso/s |               17.4 gso/s |
| `nk_dots_symmetric_u8_serial`      |               14.8 gso/s |               16.2 gso/s |               17.5 gso/s |
| `nk_dots_packed_u8_neonsdot`       |                343 gso/s |                413 gso/s |                470 gso/s |
| `nk_dots_symmetric_u8_neonsdot`    |               76.1 gso/s |               87.4 gso/s |               87.7 gso/s |
| `nk_dots_packed_u8_sme`            |              2,351 gso/s |              2,684 gso/s |              2,570 gso/s |
| `nk_dots_symmetric_u8_sme`         |              1,390 gso/s |              1,543 gso/s |              1,371 gso/s |
| __i4__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`         |               18.3 gso/s |               18.2 gso/s |               19.6 gso/s |
| `nk_dots_symmetric_i4_serial`      |               13.7 gso/s |               14.9 gso/s |               15.6 gso/s |
| `nk_dots_packed_i4_neonsdot`       |                259 gso/s |                284 gso/s |                291 gso/s |
| `nk_dots_symmetric_i4_neonsdot`    |                129 gso/s |                162 gso/s |                171 gso/s |
| `nk_dots_packed_i4_sme`            |              2,269 gso/s |              2,455 gso/s |              2,396 gso/s |
| `nk_dots_symmetric_i4_sme`         |              1,585 gso/s |              1,692 gso/s |              1,737 gso/s |
| __u4__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`         |               19.4 gso/s |               19.4 gso/s |               20.6 gso/s |
| `nk_dots_symmetric_u4_serial`      |               14.9 gso/s |               16.4 gso/s |               17.4 gso/s |
| `nk_dots_packed_u4_neonsdot`       |                300 gso/s |                319 gso/s |                340 gso/s |
| `nk_dots_symmetric_u4_neonsdot`    |                128 gso/s |                166 gso/s |                173 gso/s |
| `nk_dots_packed_u4_sme`            |              2,342 gso/s |              2,503 gso/s |              2,471 gso/s |
| `nk_dots_symmetric_u4_sme`         |              1,695 gso/s |              1,925 gso/s |              2,055 gso/s |
| __u1__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`         |                405 gso/s |                467 gso/s |                534 gso/s |
| `nk_dots_symmetric_u1_serial`      |                254 gso/s |                430 gso/s |                519 gso/s |
| `nk_dots_packed_u1_neon`           |                849 gso/s |                932 gso/s |              1,014 gso/s |
| `nk_dots_symmetric_u1_neon`        |                318 gso/s |                580 gso/s |                664 gso/s |
| `nk_dots_packed_u1_smebi32`        |              1,903 gso/s |             12,029 gso/s |             26,354 gso/s |
| `nk_dots_symmetric_u1_smebi32`     |                176 gso/s |                768 gso/s |              2,153 gso/s |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                               |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`          |        2.15 gso/s, 3 ulp |        2.07 gso/s, 5 ulp |      2.23 gso/s, 2.2 ulp |
| `nk_dots_symmetric_f64_serial`       |        2.35 gso/s, 4 ulp |        2.24 gso/s, 3 ulp |      2.46 gso/s, 2.4 ulp |
| `nk_dots_packed_f64_v128relaxed`     |     5.59 gso/s, 32.4 ulp |     6.10 gso/s, 32.4 ulp |     6.24 gso/s, 32.4 ulp |
| `nk_dots_symmetric_f64_v128relaxed`  |     5.26 gso/s, 37.6 ulp |     5.89 gso/s, 37.6 ulp |     6.04 gso/s, 37.6 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`          |       8.95 gso/s, 19 ulp |       8.71 gso/s, 30 ulp |     9.17 gso/s, 41.7 ulp |
| `nk_dots_symmetric_f32_serial`       |       10.9 gso/s, 20 ulp |       10.5 gso/s, 29 ulp |     11.6 gso/s, 58.8 ulp |
| `nk_dots_packed_f32_v128relaxed`     |     27.4 gso/s, 44.1 ulp |     31.6 gso/s, 44.1 ulp |     32.7 gso/s, 44.1 ulp |
| `nk_dots_symmetric_f32_v128relaxed`  |     10.0 gso/s, 48.2 ulp |     10.9 gso/s, 48.2 ulp |     11.2 gso/s, 48.2 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`         |      23.1 gso/s, 0.1 ulp |      21.6 gso/s, 0.5 ulp |      24.3 gso/s, 1.3 ulp |
| `nk_dots_symmetric_bf16_serial`      |        24.3 gso/s, 0 ulp |      24.9 gso/s, 0.6 ulp |      28.0 gso/s, 1.1 ulp |
| `nk_dots_packed_bf16_v128relaxed`    |      70.4 gso/s, 1.4 ulp |      86.2 gso/s, 1.4 ulp |      90.3 gso/s, 1.4 ulp |
| `nk_dots_symmetric_bf16_v128relaxed` |      37.2 gso/s, 1.3 ulp |      45.5 gso/s, 1.3 ulp |      47.7 gso/s, 1.3 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_serial`          |      12.2 gso/s, 204 ulp |       11.6 gso/s, 36 ulp |     12.4 gso/s, 25.9 ulp |
| `nk_dots_symmetric_f16_serial`       |       1.65 gso/s, 13 ulp |       1.54 gso/s, 29 ulp |     1.70 gso/s, 27.9 ulp |
| `nk_dots_packed_f16_v128relaxed`     |        35.4 gso/s, ? ulp |        40.7 gso/s, ? ulp |        39.3 gso/s, ? ulp |
| `nk_dots_symmetric_f16_v128relaxed`  |        14.7 gso/s, ? ulp |        17.1 gso/s, ? ulp |        17.3 gso/s, ? ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`         |        5.95 gso/s, 0 ulp |        5.59 gso/s, 0 ulp |        6.31 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_serial`      |        8.98 gso/s, 0 ulp |        9.09 gso/s, 0 ulp |        10.2 gso/s, 0 ulp |
| `nk_dots_packed_e5m2_v128relaxed`    |        23.0 gso/s, 0 ulp |        25.5 gso/s, 0 ulp |        25.9 gso/s, 0 ulp |
| `nk_dots_symmetric_e5m2_v128relaxed` |        12.3 gso/s, 0 ulp |        13.8 gso/s, 0 ulp |        14.2 gso/s, 0 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.884 gso/s, 0 ulp |       0.840 gso/s, 0 ulp |       0.911 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.868 gso/s, 0 ulp |       0.826 gso/s, 0 ulp |       0.915 gso/s, 0 ulp |
| `nk_dots_packed_e4m3_v128relaxed`    |        19.2 gso/s, 0 ulp |        20.8 gso/s, 0 ulp |        22.5 gso/s, 0 ulp |
| `nk_dots_symmetric_e4m3_v128relaxed` |        10.7 gso/s, 0 ulp |        11.7 gso/s, 0 ulp |        12.1 gso/s, 0 ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`         |        5.89 gso/s, 0 ulp |        5.73 gso/s, 0 ulp |        6.25 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_serial`      |        7.69 gso/s, 0 ulp |        7.45 gso/s, 0 ulp |        8.68 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_v128relaxed`    |        35.2 gso/s, 0 ulp |        38.9 gso/s, 0 ulp |        40.1 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_v128relaxed` |        32.0 gso/s, 0 ulp |        38.1 gso/s, 0 ulp |        39.7 gso/s, 0 ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`         |        5.97 gso/s, 0 ulp |        5.69 gso/s, 0 ulp |        6.32 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`      |        7.65 gso/s, 0 ulp |        7.71 gso/s, 0 ulp |        8.66 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_v128relaxed`    |        35.4 gso/s, 0 ulp |        39.0 gso/s, 0 ulp |        40.1 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_v128relaxed` |        31.6 gso/s, 0 ulp |        37.6 gso/s, 0 ulp |        39.7 gso/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`           |               16.5 gso/s |               16.0 gso/s |               16.7 gso/s |
| `nk_dots_symmetric_i8_serial`        |               12.5 gso/s |               11.8 gso/s |               13.6 gso/s |
| `nk_dots_packed_i8_v128relaxed`      |               44.0 gso/s |               50.0 gso/s |               52.1 gso/s |
| `nk_dots_symmetric_i8_v128relaxed`   |               37.7 gso/s |               45.5 gso/s |               50.6 gso/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               17.2 gso/s |               16.7 gso/s |               17.7 gso/s |
| `nk_dots_symmetric_u8_serial`        |               13.0 gso/s |               12.1 gso/s |               14.1 gso/s |
| `nk_dots_packed_u8_v128relaxed`      |               43.3 gso/s |               47.7 gso/s |               50.8 gso/s |
| `nk_dots_symmetric_u8_v128relaxed`   |               34.6 gso/s |               42.2 gso/s |               48.6 gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               15.0 gso/s |               14.3 gso/s |               15.9 gso/s |
| `nk_dots_symmetric_i4_serial`        |               12.8 gso/s |               12.6 gso/s |               14.0 gso/s |
| `nk_dots_packed_i4_v128relaxed`      |               29.3 gso/s |               26.7 gso/s |               25.8 gso/s |
| `nk_dots_symmetric_i4_v128relaxed`   |               54.0 gso/s |               70.9 gso/s |               80.8 gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               14.6 gso/s |               14.1 gso/s |               15.4 gso/s |
| `nk_dots_symmetric_u4_serial`        |               11.9 gso/s |               11.8 gso/s |               13.0 gso/s |
| `nk_dots_packed_u4_v128relaxed`      |               84.9 gso/s |               92.5 gso/s |               96.2 gso/s |
| `nk_dots_symmetric_u4_v128relaxed`   |               67.4 gso/s |               87.7 gso/s |               93.7 gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`           |                236 gso/s |                265 gso/s |                311 gso/s |
| `nk_dots_symmetric_u1_serial`        |                173 gso/s |                321 gso/s |                443 gso/s |
| `nk_dots_packed_u1_v128relaxed`      |                598 gso/s |                804 gso/s |                871 gso/s |
| `nk_dots_symmetric_u1_v128relaxed`   |                183 gso/s |                390 gso/s |                543 gso/s |
