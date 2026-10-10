# Batched Dot Products in NumKong

NumKong implements batched GEMM computing C = A × Bᵀ (packed) and C = A × Aᵀ (symmetric).
B is pre-packed once and reused across queries.
This is the foundation for the spatials and sets modules.

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

| Input Type | Output Type | Description                                      |
| :--------- | :---------- | :----------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision                 |
| `f32`      | `f64`       | 32-bit IEEE 754 single precision, widened output |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output   |
| `bf16`     | `f32`       | 16-bit brain float, widened output               |
| `e4m3`     | `f32`       | 8-bit Float8: 4 exponent, 3 mantissa bits        |
| `e5m2`     | `f32`       | 8-bit Float8: 5 exponent, 2 mantissa bits        |
| `e2m3`     | `f32`       | 8-bit MX format: 2 exponent, 3 mantissa bits     |
| `e3m2`     | `f32`       | 8-bit MX format: 3 exponent, 2 mantissa bits     |
| `e2m1`     | `f32`       | 4-bit MX format: 2 exponent, 1 mantissa bit      |
| `i8`       | `i32`       | 8-bit signed integers                            |
| `u8`       | `u32`       | 8-bit unsigned integers                          |
| `i4`       | `i32`       | 4-bit signed integers, packed nibble pairs       |
| `u4`       | `u32`       | 4-bit unsigned integers, packed nibble pairs     |
| `u1`       | `u32`       | 1-bit binary packed octets, popcount of AND      |

## Guarantees

Every capability sums each input group the same way, so `nk_dot_error_bound` holds on all of them:

| Inputs                                | Sums              | Holds while        |
| :------------------------------------ | :---------------- | :----------------- |
| `f64`                                 | Compensated, 2⁻⁵¹ | ‖x‖² finite in F64 |
| `f32`                                 | F64, 2⁻⁵¹         | inputs are finite  |
| `f16`, `bf16`, `e4m3`, `e5m2`, `e3m2` | F32, 2⁻²²         | ‖x‖² finite in F32 |
| `e2m3`, `e2m1`, integers              | Exact             | sums fit 32 bits   |
| `nvfp4`, `mxfp*`                      | Rebased F32, 2⁻²² | inputs are finite  |

Block-scaled rows and columns whose exponents spread past the rebasing window take an exact path instead.
A NaN element, a NaN scale code or a non-finite tensor scale gives NaN, a zero scale code contributes zero, and a zero vector's dot is exactly zero.

## Optimizations

### B Matrix Pre-Packing with Stride Breaking

`nk_dots_pack_f32_serial`, `nk_dots_pack_f32_haswell`, `nk_dots_pack_bf16_haswell`, `nk_dots_pack_i8_haswell` pre-pack the B matrix into a contiguous buffer optimized for streaming access during GEMM.
Power-of-2 strides, where `stride & (stride - 1) == 0`, get `depth_simd_dimensions` of padding against cache associativity conflicts.
Type conversion is amortized into the pack step: BFloat16 → Float32, Float16 → Float32, and Float8 → Float32 conversions happen once during packing instead of per-row during GEMM.
A 64-byte header stores the column count, depth, padded depth and the offset of the column norms.
Row grouping (`group_size=16`) zero-pads partial groups at matrix edges for uniform SIMD processing.

### Tiled Register Accumulation

`nk_dots_packed_f32_haswell`, `nk_dots_packed_f32_skylake`, `nk_dots_packed_f32_neon` use a 4×4 tile kernel with 16 accumulators to handle ~80% of the work.
A 1×8 tile kernel with 8 accumulators handles edge rows that don't fill a full 4-row tile.
No depth blocking is used — the kernel relies on hardware prefetch for streaming A/B access patterns.
Row loads are amortized across multiple dot products: each A row is loaded once and multiplied against 4 B columns per tile pass.

### Folding Block Scales into Operands

Block-scaled CPU kernels fold each block scale into the decoded operand wherever the product stays exact in the format the multiply consumes, instead of draining accumulators per block.
An NVFP4 element times its UE4M3 scale has at most 6 significant bits within 2⁻¹⁰ … 2688, an exact F16, and an MX element times 2^(e − base) is an exact BF16 while a row spans at most 32 binades, so the F16 and BF16 multiplies take them as is and the tensor scales and bases apply once in the epilogue.
Rows or columns spanning more take an exact wide sum after the fast loop.
GPU tiles instead multiply each block's sum by both scales, the two scales together first so neither alone flushes the partial, and Metal adds those products into compensated pairs that carry their own exponent.

### Compensated Integer GEMM

`nk_dots_packed_i8_icelake`, `nk_dots_packed_u8_icelake`, `nk_dots_packed_i8_haswell` work around the unsigned×signed operand requirement of integer dot-product instructions.
`VPDPBUSD` (Ice Lake+) computes UInt8×Int8 dot products accumulating directly to Int32 — but requires one unsigned and one signed operand.
For signed×signed (Int8×Int8), one operand is XOR'd with `0x80` to shift to unsigned range, introducing a bias of $128 \cdot \sum_k b_k$ per output element.
Rather than computing the bias correction per-element inside the inner loop (requiring extra registers for running sums), the B column sums $\sum_k b_k$ are pre-computed once during packing and stored in the packed buffer metadata.
The inner loop only needs the `VPDPBUSD` accumulator — the bias subtraction is a single post-loop correction: `result[i][j] -= 128 * b_column_sum[j]`.
This reduces per-accumulator state from 2 registers (dot + running sum) to 1 register (dot only), freeing registers for more accumulators in the 4×4 tile.
Haswell fallback uses `VPMADDUBSW` (UInt8 × Int8 → Int16) + `VPMADDWD` (Int16 → Int32), a two-instruction chain with Int16 intermediate overflow risk — quantization ranges must be tighter ([-79, 79] vs [-127, 127]).

### Finalizers Over One Accumulation Loop

Dots, spatials and sets share one accumulation loop per kernel and differ only in a finalizer, which inlines into every kernel and turns 4 dots into 4 outputs through `nk_b128_vec_t`, a union of `f32[4]`, `i32[4]` and `u32[4]`.
Angular finalizers compute 1 − d · rsqrt(q) · rsqrt(t) with separate reciprocal roots, so two large norms never overflow, and Euclidean ones √(q + t − 2d).
Integer angular and Euclidean finalizers form q · t − d² and q + t − 2d exactly in 64-bit integers before rounding, on every CPU and GPU tier, so identical vectors give exactly 0.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by `NUMWARS_DIMS_HEIGHT`, `NUMWARS_DIMS_WIDTH`, and `NUMWARS_DIMS_DEPTH` environment variables, all set to the same value for products of two square matrices.
Columns show throughput for 256³, 1024³, and 4096³ matrix products.
The throughput is measured in GSO/s as Giga Scalar Operations per Second, with ops = 2 · M · N · K arithmetic complexity for an M × K by K × N product.

### Intel Granite Rapids with RTX PRO 6000 Blackwell

#### Native

| Kernel                                |                   256³ |                  1024³ |                  4096³ |
| :------------------------------------ | ---------------------: | ---------------------: | ---------------------: |
| __f64__                               | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`           |            0.584 gso/s |            0.579 gso/s |            0.557 gso/s |
| `nk_dots_symmetric_f64_serial`        |            0.565 gso/s |            0.539 gso/s |            0.500 gso/s |
| `nk_dots_packed_f64_haswell`          |             6.70 gso/s |             6.88 gso/s |             7.19 gso/s |
| `nk_dots_symmetric_f64_haswell`       |             6.71 gso/s |             6.95 gso/s |             6.73 gso/s |
| `nk_dots_packed_f64_skylake`          |             9.11 gso/s |             10.1 gso/s |             11.0 gso/s |
| `nk_dots_symmetric_f64_skylake`       |             8.89 gso/s |             9.87 gso/s |             9.99 gso/s |
| `nk_dots_packed_f64_cuda`             |      12.7 gso/s, 0 ulp |       102 gso/s, 0 ulp |       149 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_cuda`          |      6.37 gso/s, 0 ulp |      53.2 gso/s, 0 ulp |       131 gso/s, 0 ulp |
| __f32__                               | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`           |             3.51 gso/s |             3.59 gso/s |             3.45 gso/s |
| `nk_dots_symmetric_f32_serial`        |             4.68 gso/s |             4.91 gso/s |             4.84 gso/s |
| `nk_dots_packed_f32_haswell`          |             28.2 gso/s |             32.6 gso/s |             35.6 gso/s |
| `nk_dots_symmetric_f32_haswell`       |             16.4 gso/s |             22.5 gso/s |             22.1 gso/s |
| `nk_dots_packed_f32_skylake`          |             41.0 gso/s |             48.0 gso/s |             49.3 gso/s |
| `nk_dots_symmetric_f32_skylake`       |             23.7 gso/s |             28.4 gso/s |             29.8 gso/s |
| `nk_dots_packed_f32_cuda`             |       130 gso/s, 0 ulp |   1,126 gso/s, 0.1 ulp |   1,652 gso/s, 5.1 ulp |
| `nk_dots_symmetric_f32_cuda`          |      65.2 gso/s, 0 ulp |     568 gso/s, 0.1 ulp |   1,454 gso/s, 3.8 ulp |
| __bf16__                              | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`          |            0.945 gso/s |            0.957 gso/s |            0.952 gso/s |
| `nk_dots_symmetric_bf16_serial`       |            0.873 gso/s |            0.877 gso/s |            0.883 gso/s |
| `nk_dots_packed_bf16_haswell`         |             56.9 gso/s |             65.8 gso/s |             66.6 gso/s |
| `nk_dots_symmetric_bf16_haswell`      |             47.6 gso/s |             58.7 gso/s |             61.6 gso/s |
| `nk_dots_packed_bf16_skylake`         |             80.2 gso/s |              109 gso/s |              118 gso/s |
| `nk_dots_symmetric_bf16_skylake`      |             52.1 gso/s |             82.2 gso/s |             81.4 gso/s |
| `nk_dots_packed_bf16_genoa`           |             76.2 gso/s |             89.8 gso/s |             92.4 gso/s |
| `nk_dots_symmetric_bf16_genoa`        |             54.5 gso/s |             77.7 gso/s |             75.9 gso/s |
| `nk_dots_packed_bf16_sapphireamx`     |              426 gso/s |              845 gso/s |              807 gso/s |
| `nk_dots_symmetric_bf16_sapphireamx`  |             95.1 gso/s |              115 gso/s |              157 gso/s |
| `nk_dots_packed_bf16_ampere`          |   3,237 gso/s, 9.4 ulp | 81,235 gso/s, 54.6 ulp | 311,755 gso/s, 180 ulp |
| `nk_dots_symmetric_bf16_ampere`       |   1,739 gso/s, 3.9 ulp | 40,334 gso/s, 25.4 ulp | 212,595 gso/s, 205 ulp |
| __f16__                               | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_serial`           |             3.67 gso/s |             3.51 gso/s |             3.33 gso/s |
| `nk_dots_symmetric_f16_serial`        |            0.693 gso/s |            0.668 gso/s |            0.674 gso/s |
| `nk_dots_packed_f16_haswell`          |             50.5 gso/s |             52.5 gso/s |             52.5 gso/s |
| `nk_dots_symmetric_f16_haswell`       |             37.2 gso/s |             48.0 gso/s |             52.8 gso/s |
| `nk_dots_packed_f16_skylake`          |             77.6 gso/s |              103 gso/s |              107 gso/s |
| `nk_dots_symmetric_f16_skylake`       |             46.9 gso/s |             62.5 gso/s |             63.5 gso/s |
| `nk_dots_packed_f16_graniteamx`       |              429 gso/s |              824 gso/s |              808 gso/s |
| `nk_dots_symmetric_f16_graniteamx`    |             93.3 gso/s |              117 gso/s |              155 gso/s |
| `nk_dots_packed_f16_ampere`           |   3,220 gso/s, 9.8 ulp | 81,716 gso/s, 76.4 ulp | 305,839 gso/s, 432 ulp |
| `nk_dots_symmetric_f16_ampere`        |   1,739 gso/s, 9.3 ulp | 40,351 gso/s, 41.6 ulp | 211,480 gso/s, 213 ulp |
| __e5m2__                              | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`          |            0.762 gso/s |            0.763 gso/s |            0.750 gso/s |
| `nk_dots_symmetric_e5m2_serial`       |            0.755 gso/s |            0.758 gso/s |            0.756 gso/s |
| `nk_dots_packed_e5m2_haswell`         |             40.5 gso/s |             47.2 gso/s |             48.2 gso/s |
| `nk_dots_symmetric_e5m2_haswell`      |             36.4 gso/s |             48.2 gso/s |             49.8 gso/s |
| `nk_dots_packed_e5m2_skylake`         |             55.4 gso/s |             62.4 gso/s |             66.2 gso/s |
| `nk_dots_symmetric_e5m2_skylake`      |             50.2 gso/s |             67.1 gso/s |             71.2 gso/s |
| `nk_dots_packed_e5m2_genoa`           |             48.6 gso/s |             57.7 gso/s |             65.2 gso/s |
| `nk_dots_symmetric_e5m2_genoa`        |             37.7 gso/s |             43.4 gso/s |             44.5 gso/s |
| `nk_dots_packed_e5m2_sapphireamx`     |              255 gso/s |              547 gso/s |              567 gso/s |
| `nk_dots_symmetric_e5m2_sapphireamx`  |             63.8 gso/s |             85.4 gso/s |             80.7 gso/s |
| `nk_dots_packed_e5m2_graniteamx`      |              399 gso/s |              777 gso/s |              734 gso/s |
| `nk_dots_symmetric_e5m2_graniteamx`   |              110 gso/s |              187 gso/s |              172 gso/s |
| `nk_dots_packed_e5m2_ampere`          |     3,524 gso/s, 0 ulp |  95,138 gso/s, 0.5 ulp | 341,398 gso/s, 4.8 ulp |
| `nk_dots_packed_e5m2_blackwellrtx`    |     4,402 gso/s, 0 ulp | 139,258 gso/s, 0.4 ulp | 576,629 gso/s, 4.7 ulp |
| __e4m3__                              | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`          |            0.572 gso/s |            0.567 gso/s |            0.558 gso/s |
| `nk_dots_symmetric_e4m3_serial`       |            0.599 gso/s |            0.563 gso/s |            0.563 gso/s |
| `nk_dots_packed_e4m3_haswell`         |             19.3 gso/s |             19.5 gso/s |             19.8 gso/s |
| `nk_dots_symmetric_e4m3_haswell`      |             18.2 gso/s |             19.7 gso/s |             20.0 gso/s |
| `nk_dots_packed_e4m3_skylake`         |             45.3 gso/s |             51.6 gso/s |             53.3 gso/s |
| `nk_dots_symmetric_e4m3_skylake`      |             31.6 gso/s |             35.4 gso/s |             35.9 gso/s |
| `nk_dots_packed_e4m3_genoa`           |             44.9 gso/s |             52.1 gso/s |             56.4 gso/s |
| `nk_dots_symmetric_e4m3_genoa`        |             32.3 gso/s |             38.1 gso/s |             38.7 gso/s |
| `nk_dots_packed_e4m3_sapphireamx`     |              225 gso/s |              388 gso/s |              432 gso/s |
| `nk_dots_symmetric_e4m3_sapphireamx`  |             54.3 gso/s |             67.9 gso/s |             65.5 gso/s |
| `nk_dots_packed_e4m3_ampere`          |     3,259 gso/s, 0 ulp |  85,122 gso/s, 0.3 ulp | 302,873 gso/s, 3.5 ulp |
| `nk_dots_packed_e4m3_blackwellrtx`    |     4,403 gso/s, 0 ulp | 139,146 gso/s, 0.2 ulp | 570,903 gso/s, 3.4 ulp |
| `nk_dots_symmetric_e4m3_blackwellrtx` |     2,433 gso/s, 0 ulp |  68,691 gso/s, 0.2 ulp | 399,493 gso/s, 5.2 ulp |
| __e3m2__                              | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`          |            0.909 gso/s |            0.924 gso/s |            0.894 gso/s |
| `nk_dots_symmetric_e3m2_serial`       |            0.956 gso/s |            0.943 gso/s |            0.948 gso/s |
| `nk_dots_packed_e3m2_haswell`         |             30.7 gso/s |             31.4 gso/s |             33.0 gso/s |
| `nk_dots_symmetric_e3m2_haswell`      |             31.2 gso/s |             32.6 gso/s |             33.9 gso/s |
| `nk_dots_packed_e3m2_skylake`         |             46.0 gso/s |             48.0 gso/s |             52.3 gso/s |
| `nk_dots_symmetric_e3m2_skylake`      |             45.1 gso/s |             53.3 gso/s |             57.3 gso/s |
| `nk_dots_packed_e3m2_sapphireamx`     |              337 gso/s |              526 gso/s |              525 gso/s |
| `nk_dots_symmetric_e3m2_sapphireamx`  |             80.2 gso/s |              125 gso/s |              110 gso/s |
| `nk_dots_packed_e3m2_ampere`          |     3,274 gso/s, 0 ulp |    86,167 gso/s, 0 ulp |   309,006 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_blackwellrtx`    |     4,381 gso/s, 0 ulp |   139,472 gso/s, 0 ulp |   585,989 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_blackwellrtx` |     2,434 gso/s, 0 ulp |    66,695 gso/s, 0 ulp |   402,566 gso/s, 0 ulp |
| __e2m3__                              | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`          |            0.813 gso/s |            0.763 gso/s |            0.754 gso/s |
| `nk_dots_symmetric_e2m3_serial`       |            0.751 gso/s |            0.758 gso/s |            0.757 gso/s |
| `nk_dots_packed_e2m3_haswell`         |             42.9 gso/s |             44.5 gso/s |             49.0 gso/s |
| `nk_dots_symmetric_e2m3_haswell`      |             37.5 gso/s |             49.3 gso/s |             48.6 gso/s |
| `nk_dots_packed_e2m3_skylake`         |             76.4 gso/s |             83.8 gso/s |              102 gso/s |
| `nk_dots_symmetric_e2m3_skylake`      |             70.9 gso/s |             92.8 gso/s |             94.4 gso/s |
| `nk_dots_packed_e2m3_sapphireamx`     |              493 gso/s |              935 gso/s |            1,109 gso/s |
| `nk_dots_symmetric_e2m3_sapphireamx`  |              125 gso/s |              262 gso/s |              253 gso/s |
| `nk_dots_packed_e2m3_alder`           |             50.0 gso/s |             52.5 gso/s |             56.3 gso/s |
| `nk_dots_symmetric_e2m3_alder`        |             45.3 gso/s |             54.9 gso/s |             55.9 gso/s |
| `nk_dots_packed_e2m3_ampere`          |     3,858 gso/s, 0 ulp |   113,328 gso/s, 0 ulp |   428,077 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_blackwellrtx`    |     4,402 gso/s, 0 ulp |   139,182 gso/s, 0 ulp |   586,244 gso/s, 0 ulp |
| __e2m1__                              | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m1_serial`          |            0.754 gso/s |            0.741 gso/s |            0.741 gso/s |
| `nk_dots_symmetric_e2m1_serial`       |            0.773 gso/s |            0.752 gso/s |            0.756 gso/s |
| `nk_dots_packed_e2m1_haswell`         |             92.8 gso/s |              105 gso/s |              116 gso/s |
| `nk_dots_symmetric_e2m1_haswell`      |             82.3 gso/s |              112 gso/s |              117 gso/s |
| `nk_dots_packed_e2m1_skylake`         |             99.6 gso/s |              133 gso/s |              147 gso/s |
| `nk_dots_symmetric_e2m1_skylake`      |              103 gso/s |              137 gso/s |              138 gso/s |
| `nk_dots_packed_e2m1_sapphireamx`     |              577 gso/s |            1,074 gso/s |            1,054 gso/s |
| `nk_dots_symmetric_e2m1_sapphireamx`  |              137 gso/s |              251 gso/s |              187 gso/s |
| `nk_dots_packed_e2m1_alder`           |              125 gso/s |              144 gso/s |              170 gso/s |
| `nk_dots_symmetric_e2m1_alder`        |             82.4 gso/s |              145 gso/s |              145 gso/s |
| `nk_dots_packed_e2m1_ampere`          |     4,024 gso/s, 0 ulp |   120,251 gso/s, 0 ulp |   446,458 gso/s, 0 ulp |
| `nk_dots_packed_e2m1_blackwellrtx`    |     5,324 gso/s, 0 ulp |   211,363 gso/s, 0 ulp | 1,043,708 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m1_blackwellrtx` |     3,019 gso/s, 0 ulp |   101,675 gso/s, 0 ulp |   747,117 gso/s, 0 ulp |
| __i8__                                | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`            |             5.82 gso/s |             5.83 gso/s |             5.97 gso/s |
| `nk_dots_symmetric_i8_serial`         |             5.34 gso/s |             5.46 gso/s |             5.49 gso/s |
| `nk_dots_packed_i8_haswell`           |             73.8 gso/s |             77.9 gso/s |             80.9 gso/s |
| `nk_dots_symmetric_i8_haswell`        |             58.8 gso/s |             81.9 gso/s |             85.7 gso/s |
| `nk_dots_packed_i8_icelake`           |              195 gso/s |              231 gso/s |              285 gso/s |
| `nk_dots_symmetric_i8_icelake`        |             97.2 gso/s |              181 gso/s |              187 gso/s |
| `nk_dots_packed_i8_sapphireamx`       |              796 gso/s |            1,527 gso/s |            1,403 gso/s |
| `nk_dots_symmetric_i8_sapphireamx`    |              146 gso/s |              326 gso/s |              294 gso/s |
| `nk_dots_packed_i8_alder`             |              149 gso/s |              151 gso/s |              161 gso/s |
| `nk_dots_symmetric_i8_alder`          |             73.2 gso/s |              150 gso/s |              144 gso/s |
| `nk_dots_packed_i8_ampere`            |            4,403 gso/s |          137,893 gso/s |          588,160 gso/s |
| `nk_dots_symmetric_i8_ampere`         |            2,435 gso/s |           68,976 gso/s |          403,331 gso/s |
| __u8__                                | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`            |             5.80 gso/s |             5.72 gso/s |             5.74 gso/s |
| `nk_dots_symmetric_u8_serial`         |             5.25 gso/s |             5.63 gso/s |             5.60 gso/s |
| `nk_dots_packed_u8_haswell`           |             70.6 gso/s |             77.5 gso/s |             80.8 gso/s |
| `nk_dots_symmetric_u8_haswell`        |             57.9 gso/s |             82.7 gso/s |             85.5 gso/s |
| `nk_dots_packed_u8_icelake`           |              174 gso/s |              229 gso/s |              280 gso/s |
| `nk_dots_symmetric_u8_icelake`        |             90.0 gso/s |              181 gso/s |              188 gso/s |
| `nk_dots_packed_u8_sapphireamx`       |              819 gso/s |            1,531 gso/s |            1,457 gso/s |
| `nk_dots_symmetric_u8_sapphireamx`    |              151 gso/s |              330 gso/s |              296 gso/s |
| `nk_dots_packed_u8_alder`             |              131 gso/s |              138 gso/s |              174 gso/s |
| `nk_dots_symmetric_u8_alder`          |             87.6 gso/s |              154 gso/s |              156 gso/s |
| `nk_dots_packed_u8_ampere`            |            4,407 gso/s |          137,903 gso/s |          591,377 gso/s |
| __i4__                                | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`            |            0.903 gso/s |            0.908 gso/s |            0.906 gso/s |
| `nk_dots_symmetric_i4_serial`         |             1.54 gso/s |             1.54 gso/s |             1.46 gso/s |
| `nk_dots_packed_i4_icelake`           |              140 gso/s |              197 gso/s |              259 gso/s |
| `nk_dots_symmetric_i4_icelake`        |              128 gso/s |              247 gso/s |              235 gso/s |
| `nk_dots_packed_i4_ampere`            |            4,510 gso/s |          149,308 gso/s |          583,289 gso/s |
| __u4__                                | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`            |             2.43 gso/s |             2.39 gso/s |             2.37 gso/s |
| `nk_dots_symmetric_u4_serial`         |             3.85 gso/s |             3.71 gso/s |             3.88 gso/s |
| `nk_dots_packed_u4_icelake`           |              190 gso/s |              246 gso/s |              343 gso/s |
| `nk_dots_symmetric_u4_icelake`        |              110 gso/s |              293 gso/s |              291 gso/s |
| `nk_dots_packed_u4_ampere`            |            4,577 gso/s |          153,345 gso/s |          619,229 gso/s |
| __u1__                                | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`            |             89.1 gso/s |             87.8 gso/s |             93.8 gso/s |
| `nk_dots_symmetric_u1_serial`         |             73.1 gso/s |             93.3 gso/s |             99.0 gso/s |
| `nk_dots_packed_u1_haswell`           |              161 gso/s |              211 gso/s |              377 gso/s |
| `nk_dots_symmetric_u1_haswell`        |             93.9 gso/s |              309 gso/s |              407 gso/s |
| `nk_dots_packed_u1_icelake`           |              249 gso/s |              594 gso/s |            1,034 gso/s |
| `nk_dots_symmetric_u1_icelake`        |              173 gso/s |              772 gso/s |            1,180 gso/s |

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.

#### Native

| Kernel                                    |                     256³ |                    1024³ |                    4096³ |
| :---------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_f64_with_blas` 🧩            |               48.6 gso/s |               68.4 gso/s |               73.7 gso/s |
| `dots_packed_f64_with_mkl` 🧩             |               54.6 gso/s |               68.7 gso/s |               69.2 gso/s |
| `dots_symmetric_f64_with_blas` 🧩         |               49.7 gso/s |               63.6 gso/s |               70.3 gso/s |
| `nk_dots_packed_f64_serial`               |       0.474 gso/s, 0 ulp |       0.402 gso/s, 0 ulp |       0.403 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_serial`            |       0.454 gso/s, 0 ulp |       0.458 gso/s, 0 ulp |       0.453 gso/s, 0 ulp |
| `nk_dots_packed_f64_haswell`              |        5.64 gso/s, 0 ulp |        6.04 gso/s, 0 ulp |        6.12 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_haswell`           |        5.26 gso/s, 0 ulp |        5.63 gso/s, 0 ulp |        5.56 gso/s, 0 ulp |
| `nk_dots_packed_f64_skylake`              |        7.32 gso/s, 0 ulp |        8.52 gso/s, 0 ulp |        8.65 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_skylake`           |        6.79 gso/s, 0 ulp |        7.70 gso/s, 0 ulp |        8.20 gso/s, 0 ulp |
| __f32__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_f32_with_blas` 🧩            |               99.2 gso/s |                134 gso/s |                146 gso/s |
| `dots_symmetric_f32_with_blas` 🧩         |               77.1 gso/s |                122 gso/s |                142 gso/s |
| `nk_dots_packed_f32_serial`               |        4.99 gso/s, 0 ulp |        4.17 gso/s, 0 ulp |      4.69 gso/s, 2.8 ulp |
| `nk_dots_symmetric_f32_serial`            |        3.93 gso/s, 0 ulp |        3.99 gso/s, 0 ulp |      3.95 gso/s, 1.1 ulp |
| `nk_dots_packed_f32_haswell`              |        26.0 gso/s, 0 ulp |        28.0 gso/s, 0 ulp |      29.2 gso/s, 2.8 ulp |
| `nk_dots_symmetric_f32_haswell`           |        19.4 gso/s, 0 ulp |        21.4 gso/s, 0 ulp |      20.9 gso/s, 1.1 ulp |
| `nk_dots_packed_f32_skylake`              |        33.7 gso/s, 0 ulp |        40.4 gso/s, 0 ulp |      38.6 gso/s, 0.6 ulp |
| `nk_dots_symmetric_f32_skylake`           |        28.8 gso/s, 0 ulp |        32.5 gso/s, 0 ulp |      37.2 gso/s, 0.4 ulp |
| __bf16__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_bf16_with_mkl` 🧩            |                267 gso/s |                439 gso/s |                625 gso/s |
| `nk_dots_packed_bf16_serial`              |      5.65 gso/s, 8.7 ulp |      4.29 gso/s, 7.8 ulp |     5.78 gso/s, 13.1 ulp |
| `nk_dots_symmetric_bf16_serial`           |      5.20 gso/s, 2.1 ulp |      5.50 gso/s, 6.1 ulp |                        ⋯ |
| `nk_dots_packed_bf16_haswell`             |      51.9 gso/s, 6.3 ulp |      56.5 gso/s, 9.8 ulp |     61.7 gso/s, 11.4 ulp |
| `nk_dots_symmetric_bf16_haswell`          |        44.3 gso/s, 2 ulp |      54.3 gso/s, 4.1 ulp |     55.2 gso/s, 17.9 ulp |
| `nk_dots_packed_bf16_skylake`             |      77.9 gso/s, 6.2 ulp |       104 gso/s, 5.8 ulp |       107 gso/s, 5.7 ulp |
| `nk_dots_symmetric_bf16_skylake`          |      70.0 gso/s, 1.7 ulp |      92.0 gso/s, 4.2 ulp |     99.7 gso/s, 13.8 ulp |
| `nk_dots_packed_bf16_genoa`               |      63.1 gso/s, 6.2 ulp |      80.0 gso/s, 5.8 ulp |      85.2 gso/s, 6.1 ulp |
| `nk_dots_symmetric_bf16_genoa`            |      61.7 gso/s, 1.7 ulp |      71.6 gso/s, 4.3 ulp |     80.5 gso/s, 10.3 ulp |
| `nk_dots_packed_bf16_sapphireamx`         |       371 gso/s, 2.9 ulp |       594 gso/s, 6.2 ulp |       720 gso/s, 9.2 ulp |
| `nk_dots_symmetric_bf16_sapphireamx`      |       127 gso/s, 1.9 ulp |       215 gso/s, 4.2 ulp |      196 gso/s, 10.1 ulp |
| __f16__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_f16_with_mkl` 🧩             |                254 gso/s |                410 gso/s |                596 gso/s |
| `nk_dots_packed_f16_serial`               |      2.60 gso/s, 6.7 ulp |     1.97 gso/s, 30.2 ulp |     1.95 gso/s, 80.7 ulp |
| `nk_dots_symmetric_f16_serial`            |     0.708 gso/s, 5.3 ulp |    0.714 gso/s, 15.6 ulp |                        ⋯ |
| `nk_dots_packed_f16_haswell`              |      49.2 gso/s, 5.6 ulp |       53.5 gso/s, 15 ulp |     53.1 gso/s, 77.2 ulp |
| `nk_dots_symmetric_f16_haswell`           |        40.2 gso/s, 5 ulp |       51.0 gso/s, 10 ulp |     52.2 gso/s, 32.1 ulp |
| `nk_dots_packed_f16_skylake`              |        73.0 gso/s, 4 ulp |      98.5 gso/s, 9.1 ulp |     90.1 gso/s, 47.5 ulp |
| `nk_dots_symmetric_f16_skylake`           |      52.1 gso/s, 4.4 ulp |      63.0 gso/s, 6.7 ulp |     76.1 gso/s, 18.6 ulp |
| `nk_dots_packed_f16_graniteamx`           |       369 gso/s, 5.2 ulp |       580 gso/s, 8.8 ulp |      238 gso/s, 15.1 ulp |
| `nk_dots_symmetric_f16_graniteamx`        |       124 gso/s, 3.8 ulp |       200 gso/s, 6.6 ulp |     91.1 gso/s, 16.1 ulp |
| __e5m2__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`              |        1.47 gso/s, 0 ulp |      1.14 gso/s, 0.1 ulp |      1.47 gso/s, 1.8 ulp |
| `nk_dots_symmetric_e5m2_serial`           |        1.28 gso/s, 0 ulp |      1.31 gso/s, 0.1 ulp |                        ⋯ |
| `nk_dots_packed_e5m2_haswell`             |        35.2 gso/s, 0 ulp |      39.2 gso/s, 0.1 ulp |      39.3 gso/s, 1.7 ulp |
| `nk_dots_symmetric_e5m2_haswell`          |        35.5 gso/s, 0 ulp |      41.3 gso/s, 0.2 ulp |      41.8 gso/s, 0.8 ulp |
| `nk_dots_packed_e5m2_skylake`             |        44.7 gso/s, 0 ulp |      52.8 gso/s, 0.1 ulp |      52.7 gso/s, 1.7 ulp |
| `nk_dots_symmetric_e5m2_skylake`          |        46.5 gso/s, 0 ulp |      54.0 gso/s, 0.2 ulp |      56.1 gso/s, 0.7 ulp |
| `nk_dots_packed_e5m2_genoa`               |        36.5 gso/s, 0 ulp |      40.9 gso/s, 0.1 ulp |      41.8 gso/s, 1.7 ulp |
| `nk_dots_symmetric_e5m2_genoa`            |        26.7 gso/s, 0 ulp |      29.1 gso/s, 0.1 ulp |      29.4 gso/s, 0.5 ulp |
| `nk_dots_packed_e5m2_sapphireamx`         |         196 gso/s, 0 ulp |       330 gso/s, 0.4 ulp |       359 gso/s, 1.8 ulp |
| `nk_dots_symmetric_e5m2_sapphireamx`      |        85.7 gso/s, 0 ulp |       114 gso/s, 0.4 ulp |       116 gso/s, 1.1 ulp |
| `nk_dots_packed_e5m2_graniteamx`          |         316 gso/s, 0 ulp |       520 gso/s, 0.4 ulp |       235 gso/s, 1.8 ulp |
| `nk_dots_symmetric_e5m2_graniteamx`       |         133 gso/s, 0 ulp |       222 gso/s, 0.4 ulp |      96.5 gso/s, 1.1 ulp |
| __e4m3__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`              |       0.448 gso/s, 0 ulp |     0.377 gso/s, 0.1 ulp |       0.397 gso/s, 2 ulp |
| `nk_dots_symmetric_e4m3_serial`           |       0.446 gso/s, 0 ulp |     0.444 gso/s, 0.1 ulp |                        ⋯ |
| `nk_dots_packed_e4m3_haswell`             |        15.6 gso/s, 0 ulp |      16.2 gso/s, 0.1 ulp |      16.5 gso/s, 0.8 ulp |
| `nk_dots_symmetric_e4m3_haswell`          |        15.8 gso/s, 0 ulp |      16.4 gso/s, 0.1 ulp |      16.4 gso/s, 1.3 ulp |
| `nk_dots_packed_e4m3_skylake`             |        36.2 gso/s, 0 ulp |      40.9 gso/s, 0.1 ulp |      42.3 gso/s, 0.8 ulp |
| `nk_dots_symmetric_e4m3_skylake`          |        25.4 gso/s, 0 ulp |      27.3 gso/s, 0.1 ulp |      27.8 gso/s, 1.3 ulp |
| `nk_dots_packed_e4m3_genoa`               |        37.1 gso/s, 0 ulp |      42.4 gso/s, 0.1 ulp |      43.3 gso/s, 1.8 ulp |
| `nk_dots_symmetric_e4m3_genoa`            |        27.5 gso/s, 0 ulp |      28.7 gso/s, 0.1 ulp |      30.6 gso/s, 2.3 ulp |
| `nk_dots_packed_e4m3_sapphireamx`         |         201 gso/s, 0 ulp |       338 gso/s, 0.1 ulp |       375 gso/s, 2.6 ulp |
| `nk_dots_symmetric_e4m3_sapphireamx`      |        88.2 gso/s, 0 ulp |       118 gso/s, 0.1 ulp |       120 gso/s, 3.4 ulp |
| __e3m2__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`              |        1.45 gso/s, 0 ulp |        1.14 gso/s, 0 ulp |        1.45 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_serial`           |       0.886 gso/s, 0 ulp |       0.886 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_packed_e3m2_haswell`             |        23.0 gso/s, 0 ulp |        25.1 gso/s, 0 ulp |        25.8 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_haswell`          |        24.0 gso/s, 0 ulp |        26.8 gso/s, 0 ulp |        26.9 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_skylake`             |        37.4 gso/s, 0 ulp |        42.3 gso/s, 0 ulp |        43.9 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_skylake`          |        37.3 gso/s, 0 ulp |        43.5 gso/s, 0 ulp |        44.2 gso/s, 0 ulp |
| `nk_dots_packed_e3m2_sapphireamx`         |         255 gso/s, 0 ulp |         423 gso/s, 0 ulp |         400 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_sapphireamx`      |         118 gso/s, 0 ulp |         169 gso/s, 0 ulp |         174 gso/s, 0 ulp |
| __e2m3__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`              |        1.44 gso/s, 0 ulp |        1.15 gso/s, 0 ulp |        1.45 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`           |       0.875 gso/s, 0 ulp |       0.890 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_packed_e2m3_haswell`             |        35.3 gso/s, 0 ulp |        37.6 gso/s, 0 ulp |        39.8 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_haswell`          |        35.1 gso/s, 0 ulp |        40.4 gso/s, 0 ulp |        40.4 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_skylake`             |        63.3 gso/s, 0 ulp |        69.3 gso/s, 0 ulp |        74.2 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_skylake`          |        64.8 gso/s, 0 ulp |        79.6 gso/s, 0 ulp |        82.4 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_sapphireamx`         |         433 gso/s, 0 ulp |       1,002 gso/s, 0 ulp |         868 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_sapphireamx`      |         181 gso/s, 0 ulp |         353 gso/s, 0 ulp |         315 gso/s, 0 ulp |
| `nk_dots_packed_e2m3_alder`               |        42.5 gso/s, 0 ulp |        44.9 gso/s, 0 ulp |        46.5 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_alder`            |        33.1 gso/s, 0 ulp |        43.2 gso/s, 0 ulp |        44.4 gso/s, 0 ulp |
| __i8__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_i8u8_with_mkl` 🧩            |                306 gso/s |                623 gso/s |              1,262 gso/s |
| `nk_dots_packed_i8_serial`                |               4.81 gso/s |               4.83 gso/s |               4.93 gso/s |
| `nk_dots_symmetric_i8_serial`             |               4.56 gso/s |               4.68 gso/s |                        ⋯ |
| `nk_dots_packed_i8_haswell`               |               55.7 gso/s |               62.7 gso/s |               66.0 gso/s |
| `nk_dots_symmetric_i8_haswell`            |               54.9 gso/s |               67.8 gso/s |               71.0 gso/s |
| `nk_dots_packed_i8_icelake`               |                166 gso/s |                242 gso/s |                296 gso/s |
| `nk_dots_symmetric_i8_icelake`            |                113 gso/s |                286 gso/s |                321 gso/s |
| `nk_dots_packed_i8_sapphireamx`           |                626 gso/s |              1,579 gso/s |              1,326 gso/s |
| `nk_dots_symmetric_i8_sapphireamx`        |                191 gso/s |                414 gso/s |                198 gso/s |
| `nk_dots_packed_i8_alder`                 |                114 gso/s |                128 gso/s |                148 gso/s |
| `nk_dots_symmetric_i8_alder`              |               81.8 gso/s |                150 gso/s |                160 gso/s |
| __u8__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`                |               4.86 gso/s |               4.90 gso/s |               5.00 gso/s |
| `nk_dots_symmetric_u8_serial`             |               4.45 gso/s |               4.58 gso/s |                        ⋯ |
| `nk_dots_packed_u8_haswell`               |               56.0 gso/s |               61.9 gso/s |               66.1 gso/s |
| `nk_dots_symmetric_u8_haswell`            |               55.1 gso/s |               66.6 gso/s |               70.6 gso/s |
| `nk_dots_packed_u8_icelake`               |                167 gso/s |                250 gso/s |                296 gso/s |
| `nk_dots_symmetric_u8_icelake`            |                112 gso/s |                286 gso/s |                325 gso/s |
| `nk_dots_packed_u8_sapphireamx`           |                624 gso/s |              1,585 gso/s |              1,285 gso/s |
| `nk_dots_symmetric_u8_sapphireamx`        |                192 gso/s |                410 gso/s |                194 gso/s |
| `nk_dots_packed_u8_alder`                 |                111 gso/s |                127 gso/s |                152 gso/s |
| `nk_dots_symmetric_u8_alder`              |               81.8 gso/s |                151 gso/s |                162 gso/s |
| __i4__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`                |               1.24 gso/s |               1.25 gso/s |               1.24 gso/s |
| `nk_dots_symmetric_i4_serial`             |               1.45 gso/s |               1.45 gso/s |                        ⋯ |
| `nk_dots_packed_i4_icelake`               |               12.2 gso/s |               13.0 gso/s |               13.1 gso/s |
| `nk_dots_symmetric_i4_icelake`            |                105 gso/s |                227 gso/s |                256 gso/s |
| `nk_dots_packed_i4_haswell`               |               11.7 gso/s |               11.9 gso/s |               12.3 gso/s |
| `nk_dots_symmetric_i4_haswell`            |               29.5 gso/s |               33.6 gso/s |               33.9 gso/s |
| __u4__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`                |               1.97 gso/s |               1.98 gso/s |               1.99 gso/s |
| `nk_dots_symmetric_u4_serial`             |               3.17 gso/s |               3.22 gso/s |                        ⋯ |
| `nk_dots_packed_u4_icelake`               |                156 gso/s |                249 gso/s |                316 gso/s |
| `nk_dots_symmetric_u4_icelake`            |                144 gso/s |                311 gso/s |                343 gso/s |
| `nk_dots_packed_u4_haswell`               |               65.9 gso/s |               75.4 gso/s |               79.4 gso/s |
| `nk_dots_symmetric_u4_haswell`            |               59.2 gso/s |               75.1 gso/s |               77.1 gso/s |
| __u1__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_haswell`               |                146 gso/s |                199 gso/s |                333 gso/s |
| `nk_dots_symmetric_u1_haswell`            |               99.7 gso/s |                252 gso/s |                344 gso/s |
| `nk_dots_packed_u1_icelake`               |                224 gso/s |                527 gso/s |              1,134 gso/s |
| `nk_dots_symmetric_u1_icelake`            |                184 gso/s |                815 gso/s |              1,502 gso/s |
| `nk_dots_packed_u1_serial`                |               83.3 gso/s |               91.4 gso/s |                        ⋯ |
| `nk_dots_symmetric_u1_serial`             |               80.9 gso/s |                102 gso/s |                        ⋯ |
| __e2m1__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m1_sapphireamx`         |         428 gso/s, 0 ulp |         884 gso/s, 0 ulp |         944 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m1_sapphireamx`      |         184 gso/s, 0 ulp |         350 gso/s, 0 ulp |         351 gso/s, 0 ulp |
| `nk_dots_packed_e2m1_serial`              |        1.09 gso/s, 0 ulp |        1.00 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_symmetric_e2m1_serial`           |        1.63 gso/s, 0 ulp |        1.65 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_packed_e2m1_haswell`             |        79.9 gso/s, 0 ulp |        94.7 gso/s, 0 ulp |         104 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m1_haswell`          |        75.8 gso/s, 0 ulp |        94.1 gso/s, 0 ulp |        95.6 gso/s, 0 ulp |
| `nk_dots_packed_e2m1_skylake`             |        82.9 gso/s, 0 ulp |         108 gso/s, 0 ulp |         115 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m1_skylake`          |        83.7 gso/s, 0 ulp |         113 gso/s, 0 ulp |         121 gso/s, 0 ulp |
| `nk_dots_packed_e2m1_alder`               |         101 gso/s, 0 ulp |         122 gso/s, 0 ulp |         136 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m1_alder`            |        77.4 gso/s, 0 ulp |         103 gso/s, 0 ulp |         108 gso/s, 0 ulp |
| __nvfp4__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_nvfp4_sapphireamx`        |         152 gso/s, 0 ulp |         236 gso/s, 0 ulp |         248 gso/s, 0 ulp |
| `nk_dots_symmetric_nvfp4_sapphireamx`     |        56.3 gso/s, 0 ulp |        65.9 gso/s, 0 ulp |        50.3 gso/s, 0 ulp |
| `nk_dots_packed_nvfp4_serial`             |       0.202 gso/s, 0 ulp |       0.181 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_symmetric_nvfp4_serial`          |       0.203 gso/s, 0 ulp |       0.205 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_packed_nvfp4_skylake`            |        25.8 gso/s, 0 ulp |        28.5 gso/s, 0 ulp |        28.0 gso/s, 0 ulp |
| `nk_dots_symmetric_nvfp4_skylake`         |        25.1 gso/s, 0 ulp |        27.2 gso/s, 0 ulp |        26.5 gso/s, 0 ulp |
| __mxfp4__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp4_sapphireamx`        |         168 gso/s, 0 ulp |         201 gso/s, 0 ulp |         302 gso/s, 0 ulp |
| `nk_dots_symmetric_mxfp4_sapphireamx`     |        61.0 gso/s, 0 ulp |        69.6 gso/s, 0 ulp |        39.6 gso/s, 0 ulp |
| `nk_dots_packed_mxfp4_serial`             |       0.186 gso/s, 0 ulp |       0.188 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_symmetric_mxfp4_serial`          |       0.193 gso/s, 0 ulp |       0.193 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_packed_mxfp4_skylake`            |        25.9 gso/s, 0 ulp |        27.8 gso/s, 0 ulp |        27.8 gso/s, 0 ulp |
| `nk_dots_symmetric_mxfp4_skylake`         |        24.7 gso/s, 0 ulp |        27.6 gso/s, 0 ulp |        25.9 gso/s, 0 ulp |
| __mxfp8e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp8e4m3_sapphireamx`    |         139 gso/s, 0 ulp |       185 gso/s, 0.1 ulp |       234 gso/s, 2.6 ulp |
| `nk_dots_symmetric_mxfp8e4m3_sapphireamx` |        46.9 gso/s, 0 ulp |      56.8 gso/s, 0.1 ulp |      21.4 gso/s, 3.4 ulp |
| `nk_dots_packed_mxfp8e4m3_serial`         |       0.260 gso/s, 0 ulp |     0.262 gso/s, 0.1 ulp |                        ⋯ |
| `nk_dots_symmetric_mxfp8e4m3_serial`      |       0.255 gso/s, 0 ulp |     0.261 gso/s, 0.1 ulp |                        ⋯ |
| `nk_dots_packed_mxfp8e4m3_skylake`        |        16.3 gso/s, 0 ulp |      17.0 gso/s, 0.1 ulp |      16.8 gso/s, 1.8 ulp |
| `nk_dots_symmetric_mxfp8e4m3_skylake`     |        15.8 gso/s, 0 ulp |      16.9 gso/s, 0.1 ulp |      15.7 gso/s, 3.2 ulp |
| __mxfp8e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp8e5m2_sapphireamx`    |         153 gso/s, 0 ulp |       248 gso/s, 0.4 ulp |       272 gso/s, 1.8 ulp |
| `nk_dots_symmetric_mxfp8e5m2_sapphireamx` |        54.5 gso/s, 0 ulp |      62.9 gso/s, 0.4 ulp |      32.8 gso/s, 1.1 ulp |
| `nk_dots_packed_mxfp8e5m2_serial`         |       0.381 gso/s, 0 ulp |     0.382 gso/s, 0.4 ulp |                        ⋯ |
| `nk_dots_symmetric_mxfp8e5m2_serial`      |       0.378 gso/s, 0 ulp |     0.385 gso/s, 0.4 ulp |                        ⋯ |
| `nk_dots_packed_mxfp8e5m2_skylake`        |        19.8 gso/s, 0 ulp |      20.8 gso/s, 0.1 ulp |      20.5 gso/s, 1.8 ulp |
| `nk_dots_symmetric_mxfp8e5m2_skylake`     |        19.1 gso/s, 0 ulp |      20.4 gso/s, 0.2 ulp |        18.7 gso/s, 1 ulp |
| __mxfp6e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp6e2m3_serial`         |       0.380 gso/s, 0 ulp |       0.259 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_symmetric_mxfp6e2m3_serial`      |       0.380 gso/s, 0 ulp |       0.258 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_packed_mxfp6e2m3_skylake`        |        13.6 gso/s, 0 ulp |        14.3 gso/s, 0 ulp |        14.3 gso/s, 0 ulp |
| `nk_dots_symmetric_mxfp6e2m3_skylake`     |        13.6 gso/s, 0 ulp |        14.2 gso/s, 0 ulp |        13.5 gso/s, 0 ulp |
| __mxfp6e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp6e3m2_serial`         |       0.375 gso/s, 0 ulp |       0.263 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_symmetric_mxfp6e3m2_serial`      |       0.381 gso/s, 0 ulp |       0.263 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_packed_mxfp6e3m2_skylake`        |        13.7 gso/s, 0 ulp |        14.1 gso/s, 0 ulp |        14.2 gso/s, 0 ulp |
| `nk_dots_symmetric_mxfp6e3m2_skylake`     |        13.3 gso/s, 0 ulp |        14.2 gso/s, 0 ulp |        13.5 gso/s, 0 ulp |
| __None__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `dots_packed_i16_with_mkl` 🧩             |                173 gso/s |                257 gso/s |                263 gso/s |
| `dots_packed_f32_with_mkl` 🧩             |                109 gso/s |                135 gso/s |                147 gso/s |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel                               |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`          |       0.331 gso/s, 0 ulp |       0.443 gso/s, 0 ulp |       0.395 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_serial`       |       0.417 gso/s, 0 ulp |       0.414 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_packed_f64_v128relaxed`     |        1.69 gso/s, 0 ulp |        1.57 gso/s, 0 ulp |        1.50 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_v128relaxed`  |        1.04 gso/s, 0 ulp |        1.23 gso/s, 0 ulp |        1.74 gso/s, 0 ulp |
| __f32__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f32_serial`          |        2.28 gso/s, 0 ulp |        3.88 gso/s, 0 ulp |      2.91 gso/s, 1.3 ulp |
| `nk_dots_symmetric_f32_serial`       |        2.98 gso/s, 0 ulp |        4.07 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_packed_f32_v128relaxed`     |        10.1 gso/s, 0 ulp |        9.97 gso/s, 0 ulp |        6.86 gso/s, 2 ulp |
| `nk_dots_symmetric_f32_v128relaxed`  |        4.13 gso/s, 0 ulp |      5.71 gso/s, 0.1 ulp |      8.46 gso/s, 3.8 ulp |
| __bf16__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_serial`         |      1.39 gso/s, 3.7 ulp |      2.02 gso/s, 6.3 ulp |     1.86 gso/s, 42.7 ulp |
| `nk_dots_symmetric_bf16_serial`      |      1.96 gso/s, 3.6 ulp |      2.06 gso/s, 4.3 ulp |                        ⋯ |
| `nk_dots_packed_bf16_v128relaxed`    |      30.0 gso/s, 3.2 ulp |      23.5 gso/s, 5.6 ulp |     20.9 gso/s, 21.4 ulp |
| `nk_dots_symmetric_bf16_v128relaxed` |      15.8 gso/s, 3.9 ulp |      21.1 gso/s, 4.5 ulp |     30.0 gso/s, 41.3 ulp |
| `nk_dots_packed_bf16_v128`           |      16.1 gso/s, 3.2 ulp |      16.1 gso/s, 5.6 ulp |     27.5 gso/s, 21.4 ulp |
| `nk_dots_symmetric_bf16_v128`        |      13.6 gso/s, 3.9 ulp |      18.2 gso/s, 4.5 ulp |     20.4 gso/s, 41.3 ulp |
| __f16__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_serial`          |      2.06 gso/s, 7.6 ulp |     2.90 gso/s, 16.3 ulp |     2.89 gso/s, 35.6 ulp |
| `nk_dots_symmetric_f16_serial`       |      0.575 gso/s, 11 ulp |      0.570 gso/s, 48 ulp |                        ⋯ |
| `nk_dots_packed_f16_v128relaxed`     |        12.0 gso/s, 7 ulp |     11.6 gso/s, 16.3 ulp |     9.79 gso/s, 35.5 ulp |
| `nk_dots_symmetric_f16_v128relaxed`  |     4.34 gso/s, 10.6 ulp |     5.22 gso/s, 47.9 ulp |     7.42 gso/s, 29.1 ulp |
| __e5m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_serial`         |        1.11 gso/s, 0 ulp |      1.09 gso/s, 0.1 ulp |     0.927 gso/s, 0.8 ulp |
| `nk_dots_symmetric_e5m2_serial`      |        1.11 gso/s, 0 ulp |      1.14 gso/s, 0.3 ulp |                        ⋯ |
| `nk_dots_packed_e5m2_v128relaxed`    |        9.23 gso/s, 0 ulp |      7.59 gso/s, 0.1 ulp |      9.62 gso/s, 0.8 ulp |
| `nk_dots_symmetric_e5m2_v128relaxed` |        3.14 gso/s, 0 ulp |      3.77 gso/s, 0.2 ulp |      5.32 gso/s, 4.8 ulp |
| __e4m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_serial`         |       0.277 gso/s, 0 ulp |     0.276 gso/s, 0.1 ulp |     0.262 gso/s, 2.7 ulp |
| `nk_dots_symmetric_e4m3_serial`      |       0.275 gso/s, 0 ulp |     0.198 gso/s, 0.2 ulp |                        ⋯ |
| `nk_dots_packed_e4m3_v128relaxed`    |        10.9 gso/s, 0 ulp |      8.36 gso/s, 0.1 ulp |      11.2 gso/s, 1.6 ulp |
| `nk_dots_symmetric_e4m3_v128relaxed` |        3.80 gso/s, 0 ulp |      3.85 gso/s, 0.3 ulp |      6.45 gso/s, 1.2 ulp |
| __e3m2__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_serial`         |        1.09 gso/s, 0 ulp |        1.12 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_symmetric_e3m2_serial`      |        1.12 gso/s, 0 ulp |        1.02 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_packed_e3m2_v128relaxed`    |     17.0 gso/s, 5.5M ulp |    17.6 gso/s, 6.62M ulp |    17.9 gso/s, 4.01M ulp |
| `nk_dots_symmetric_e3m2_v128relaxed` |    16.7 gso/s, 2.39M ulp |    17.8 gso/s, 11.4M ulp |    17.9 gso/s, 2.46M ulp |
| __e2m3__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_serial`         |        1.09 gso/s, 0 ulp |        1.03 gso/s, 0 ulp |       0.834 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_serial`      |       0.957 gso/s, 0 ulp |       0.811 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_packed_e2m3_v128relaxed`    |        17.0 gso/s, 0 ulp |        15.0 gso/s, 0 ulp |        18.1 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_v128relaxed` |        10.4 gso/s, 0 ulp |        10.8 gso/s, 0 ulp |        17.9 gso/s, 0 ulp |
| __i8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_serial`           |               5.39 gso/s |               5.55 gso/s |               2.79 gso/s |
| `nk_dots_symmetric_i8_serial`        |               3.28 gso/s |               2.59 gso/s |                        ⋯ |
| `nk_dots_packed_i8_v128relaxed`      |               31.5 gso/s |               31.9 gso/s |               37.3 gso/s |
| `nk_dots_symmetric_i8_v128relaxed`   |               18.6 gso/s |               18.5 gso/s |               34.3 gso/s |
| `nk_dots_packed_i8_v128`             |               27.1 gso/s |               32.8 gso/s |               39.0 gso/s |
| `nk_dots_symmetric_i8_v128`          |               23.9 gso/s |               37.5 gso/s |               38.9 gso/s |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               5.43 gso/s |               5.51 gso/s |               5.50 gso/s |
| `nk_dots_symmetric_u8_serial`        |               3.55 gso/s |               2.76 gso/s |                        ⋯ |
| `nk_dots_packed_u8_v128relaxed`      |               29.3 gso/s |               31.4 gso/s |               36.8 gso/s |
| `nk_dots_symmetric_u8_v128relaxed`   |               15.1 gso/s |               14.8 gso/s |               28.3 gso/s |
| `nk_dots_packed_u8_v128`             |               30.3 gso/s |               37.0 gso/s |               42.2 gso/s |
| `nk_dots_symmetric_u8_v128`          |               25.1 gso/s |               42.2 gso/s |               39.7 gso/s |
| __i4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_serial`           |               2.50 gso/s |               2.58 gso/s |               2.60 gso/s |
| `nk_dots_symmetric_i4_serial`        |               2.58 gso/s |               1.83 gso/s |                        ⋯ |
| `nk_dots_packed_i4_v128relaxed`      |               15.5 gso/s |               14.0 gso/s |               18.1 gso/s |
| `nk_dots_symmetric_i4_v128relaxed`   |               22.9 gso/s |               30.6 gso/s |               50.8 gso/s |
| __u4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_serial`           |               3.24 gso/s |               3.25 gso/s |                        ⋯ |
| `nk_dots_symmetric_u4_serial`        |               3.27 gso/s |               3.26 gso/s |                        ⋯ |
| `nk_dots_packed_u4_v128relaxed`      |               36.7 gso/s |               42.2 gso/s |               72.6 gso/s |
| `nk_dots_symmetric_u4_v128relaxed`   |               36.0 gso/s |               38.7 gso/s |               71.6 gso/s |
| __u1__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u1_serial`           |                183 gso/s |                230 gso/s |                        ⋯ |
| `nk_dots_packed_u1_v128`             |                124 gso/s |                186 gso/s |                224 gso/s |
| `nk_dots_symmetric_u1_serial`        |                114 gso/s |                177 gso/s |                        ⋯ |
| `nk_dots_symmetric_u1_v128`          |               79.5 gso/s |                181 gso/s |                206 gso/s |
| __e2m1__                             | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m1_serial`         |        2.26 gso/s, 0 ulp |        2.29 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_symmetric_e2m1_serial`      |        2.36 gso/s, 0 ulp |        2.25 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_packed_e2m1_v128relaxed`    |        39.0 gso/s, 0 ulp |        43.0 gso/s, 0 ulp |        45.0 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m1_v128relaxed` |        37.1 gso/s, 0 ulp |        44.2 gso/s, 0 ulp |        45.1 gso/s, 0 ulp |
| __nvfp4__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_nvfp4_serial`        |       0.288 gso/s, 0 ulp |       0.291 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_symmetric_nvfp4_serial`     |       0.292 gso/s, 0 ulp |       0.280 gso/s, 0 ulp |                        ⋯ |
| __mxfp4__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp4_serial`        |       0.298 gso/s, 0 ulp |       0.297 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_symmetric_mxfp4_serial`     |       0.309 gso/s, 0 ulp |       0.309 gso/s, 0 ulp |                        ⋯ |
| __mxfp8e4m3__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp8e4m3_serial`    |       0.166 gso/s, 0 ulp |     0.164 gso/s, 0.1 ulp |                        ⋯ |
| `nk_dots_symmetric_mxfp8e4m3_serial` |     0.149 gso/s, 0.1 ulp |     0.155 gso/s, 1.3 ulp |                        ⋯ |
| __mxfp8e5m2__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp8e5m2_serial`    |       0.329 gso/s, 0 ulp |     0.309 gso/s, 0.1 ulp |                        ⋯ |
| `nk_dots_symmetric_mxfp8e5m2_serial` |       0.328 gso/s, 0 ulp |     0.332 gso/s, 0.1 ulp |                        ⋯ |
| __mxfp6e2m3__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp6e2m3_serial`    |       0.329 gso/s, 0 ulp |       0.295 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_symmetric_mxfp6e2m3_serial` |       0.337 gso/s, 0 ulp |       0.293 gso/s, 0 ulp |                        ⋯ |
| __mxfp6e3m2__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp6e3m2_serial`    |       0.333 gso/s, 0 ulp |       0.289 gso/s, 0 ulp |                        ⋯ |
| `nk_dots_symmetric_mxfp6e3m2_serial` |       0.336 gso/s, 0 ulp |       0.318 gso/s, 0 ulp |                        ⋯ |

#### CUDA

Rows ran on one `1g.34gb` MIG slice of a B300 with 18 SMs.
The `cublasLtMatmul` rows time cuBLASLt from CUDA 13.2 on the same operands, with unit block scales for block-scaled dtypes.
Cells marked `✗` are dtypes cuBLASLt rejects, with `CUBLAS_STATUS_INVALID_VALUE`, `CUBLAS_STATUS_NOT_SUPPORTED`.

| Kernel                                  |                    4096³ |                    8192³ |
| :-------------------------------------- | -----------------------: | -----------------------: |
| __bf16__                                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_bf16_blackwell`         | 210,000 gso/s, 179.9 ulp | 220,700 gso/s, 294.3 ulp |
| `nk_dots_symmetric_bf16_blackwell`      | 155,200 gso/s, 204.9 ulp |   156,100 gso/s, 315 ulp |
| `cublasLtMatmul`                        | 210,000 gso/s, 179.9 ulp | 216,400 gso/s, 294.3 ulp |
| __f16__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f16_blackwell`          | 209,800 gso/s, 431.7 ulp | 220,600 gso/s, 364.5 ulp |
| `nk_dots_symmetric_f16_blackwell`       | 155,100 gso/s, 212.8 ulp | 156,100 gso/s, 404.2 ulp |
| `cublasLtMatmul`                        | 210,100 gso/s, 431.7 ulp | 216,400 gso/s, 364.5 ulp |
| __e5m2__                                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e5m2_blackwell`         |   390,400 gso/s, 4.7 ulp |   454,400 gso/s, 3.5 ulp |
| `nk_dots_symmetric_e5m2_blackwell`      |   359,600 gso/s, 1.6 ulp |   301,000 gso/s, 2.4 ulp |
| `cublasLtMatmul`                        |                        ✗ |                        ✗ |
| __e4m3__                                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e4m3_blackwell`         |   390,300 gso/s, 3.4 ulp |   454,400 gso/s, 7.8 ulp |
| `nk_dots_symmetric_e4m3_blackwell`      |   347,100 gso/s, 5.2 ulp |     301,200 gso/s, 6 ulp |
| `cublasLtMatmul`                        |   480,100 gso/s, 3.4 ulp |   436,400 gso/s, 7.8 ulp |
| __e3m2__                                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e3m2_blackwell`         |     274,300 gso/s, 0 ulp |     291,500 gso/s, 0 ulp |
| `nk_dots_symmetric_e3m2_blackwell`      |     126,000 gso/s, 0 ulp |     153,000 gso/s, 0 ulp |
| `cublasLtMatmul`                        |                        ✗ |                        ✗ |
| __e2m3__                                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m3_blackwell`         |     274,400 gso/s, 0 ulp |     291,400 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m3_blackwell`      |     126,100 gso/s, 0 ulp |     153,100 gso/s, 0 ulp |
| `cublasLtMatmul`                        |                        ✗ |                        ✗ |
| __e2m1__                                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_e2m1_blackwell`         |     608,500 gso/s, 0 ulp |     811,000 gso/s, 0 ulp |
| `nk_dots_symmetric_e2m1_blackwell`      |     548,900 gso/s, 0 ulp |     788,100 gso/s, 0 ulp |
| `cublasLtMatmul`                        |     718,700 gso/s, 0 ulp |     953,400 gso/s, 0 ulp |
| __nvfp4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_nvfp4_blackwell`        |     460,900 gso/s, 0 ulp |     577,600 gso/s, 0 ulp |
| `nk_dots_symmetric_nvfp4_blackwell`     |     391,500 gso/s, 0 ulp |     534,800 gso/s, 0 ulp |
| `cublasLtMatmul`                        |     720,100 gso/s, 0 ulp |     955,800 gso/s, 0 ulp |
| __mxfp4__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp4_blackwell`        |     493,200 gso/s, 0 ulp |     638,700 gso/s, 0 ulp |
| `nk_dots_symmetric_mxfp4_blackwell`     |     414,100 gso/s, 0 ulp |     597,800 gso/s, 0 ulp |
| `cublasLtMatmul`                        |                        ✗ |                        ✗ |
| __mxfp8e4m3__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp8e4m3_blackwell`    |   315,300 gso/s, 3.4 ulp |   354,600 gso/s, 7.8 ulp |
| `nk_dots_symmetric_mxfp8e4m3_blackwell` |   257,900 gso/s, 5.2 ulp |     263,300 gso/s, 6 ulp |
| `cublasLtMatmul`                        |   455,300 gso/s, 3.4 ulp |   421,600 gso/s, 7.8 ulp |
| __mxfp8e5m2__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_mxfp8e5m2_blackwell`    |   315,400 gso/s, 4.7 ulp |   354,900 gso/s, 3.5 ulp |
| `nk_dots_symmetric_mxfp8e5m2_blackwell` |   258,000 gso/s, 1.6 ulp |   263,300 gso/s, 2.4 ulp |
| `cublasLtMatmul`                        |                        ✗ |                        ✗ |
| __i8__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i8_blackwell`           |     103,100 gso/s, exact |     102,300 gso/s, exact |
| `nk_dots_symmetric_i8_blackwell`        |      60,920 gso/s, exact |      66,040 gso/s, exact |
| `cublasLtMatmul`                        |      18,240 gso/s, exact |      18,280 gso/s, exact |
| __u8__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_blackwell`           |     103,100 gso/s, exact |     103,300 gso/s, exact |
| `nk_dots_symmetric_u8_blackwell`        |      63,510 gso/s, exact |      69,030 gso/s, exact |
| `cublasLtMatmul`                        |                        ✗ |                        ✗ |
| __i4__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_i4_blackwell`           |      68,490 gso/s, exact |      69,250 gso/s, exact |
| `nk_dots_symmetric_i4_blackwell`        |      60,980 gso/s, exact |      64,740 gso/s, exact |
| `cublasLtMatmul`                        |                        ✗ |                        ✗ |
| __u4__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u4_blackwell`           |      69,140 gso/s, exact |      69,840 gso/s, exact |
| `nk_dots_symmetric_u4_blackwell`        |      61,230 gso/s, exact |      65,450 gso/s, exact |
| `cublasLtMatmul`                        |                        ✗ |                        ✗ |

### Apple M5

#### Native

| Kernel                             |                     256³ |                    1024³ |                    4096³ |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_f64_serial`        |        2.49 gso/s, 3 ulp |        2.36 gso/s, 5 ulp |        2.48 gso/s, 6 ulp |
| `nk_dots_symmetric_f64_serial`     |        1.38 gso/s, 0 ulp |        1.36 gso/s, 0 ulp |        1.49 gso/s, 0 ulp |
| `nk_dots_packed_f64_neon`          |        6.31 gso/s, 0 ulp |        6.00 gso/s, 0 ulp |        6.34 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_neon`       |        5.57 gso/s, 0 ulp |        5.41 gso/s, 0 ulp |        5.40 gso/s, 0 ulp |
| `nk_dots_packed_f64_smef64`        |        16.8 gso/s, 0 ulp |        20.4 gso/s, 0 ulp |        21.4 gso/s, 0 ulp |
| `nk_dots_symmetric_f64_smef64`     |        8.11 gso/s, 0 ulp |        9.76 gso/s, 0 ulp |        9.66 gso/s, 0 ulp |
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
| `nk_dots_packed_f16_neon`          |     77.0 gso/s, 16.8 ulp |     79.1 gso/s, 25.5 ulp |      84.2 gso/s, 618 ulp |
| `nk_dots_symmetric_f16_neon`       |     20.5 gso/s, 12.1 ulp |     20.4 gso/s, 25.0 ulp |      22.5 gso/s, 506 ulp |
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
| `nk_dots_packed_bf16_v128`           |                        … |                        … |                        … |
| `nk_dots_symmetric_bf16_v128`        |                        … |                        … |                        … |
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
| `nk_dots_packed_i8_v128`             |                        … |                        … |                        … |
| `nk_dots_symmetric_i8_v128`          |                        … |                        … |                        … |
| __u8__                               | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_dots_packed_u8_serial`           |               17.2 gso/s |               16.7 gso/s |               17.7 gso/s |
| `nk_dots_symmetric_u8_serial`        |               13.0 gso/s |               12.1 gso/s |               14.1 gso/s |
| `nk_dots_packed_u8_v128relaxed`      |               43.3 gso/s |               47.7 gso/s |               50.8 gso/s |
| `nk_dots_symmetric_u8_v128relaxed`   |               34.6 gso/s |               42.2 gso/s |               48.6 gso/s |
| `nk_dots_packed_u8_v128`             |                        … |                        … |                        … |
| `nk_dots_symmetric_u8_v128`          |                        … |                        … |                        … |
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
| `nk_dots_packed_u1_v128`             |                598 gso/s |                804 gso/s |                871 gso/s |
| `nk_dots_symmetric_u1_v128`          |                183 gso/s |                390 gso/s |                543 gso/s |
