# Batched Distance Matrices in NumKong

NumKong implements batched distance matrix computation via pre-packed dot products plus normalization.
Angular distance and Euclidean distance are computed from the packed dot product output without materializing an intermediate C matrix.

Angular distance from pre-packed dot products:

$$
D_{ij} = 1 - \frac{C_{ij}}{\sqrt{\|A_i\|^2 \cdot \|B_j\|^2}}
$$

Euclidean distance from pre-packed dot products:

$$
D_{ij} = \sqrt{\|A_i\|^2 + \|B_j\|^2 - 2 C_{ij}}
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def angulars_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = a @ b.T
    a_norms = np.sum(a ** 2, axis=1, keepdims=True)
    b_norms = np.sum(b ** 2, axis=1, keepdims=True)
    return 1 - dots / np.sqrt(a_norms * b_norms.T)

def euclideans_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = a @ b.T
    a_norms = np.sum(a ** 2, axis=1, keepdims=True)
    b_norms = np.sum(b ** 2, axis=1, keepdims=True)
    return np.sqrt(np.maximum(a_norms + b_norms.T - 2 * dots, 0))
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
| `i8`       | `f32`       | 8-bit signed integers, float output              |
| `u8`       | `f32`       | 8-bit unsigned integers, float output            |
| `i4`       | `f32`       | 4-bit signed integers, float output              |
| `u4`       | `f32`       | 4-bit unsigned integers, float output            |

## Guarantees

Every capability applies one rule to degenerate inputs:

- a NaN dot, from a NaN element or scale code, gives NaN for both metrics;
- angular distance is 0 for two zero norms, 1 for one zero norm or an exactly zero dot, and otherwise max(0, 1 − dot · rsqrt(‖a‖²) · rsqrt(‖b‖²)), where an infinite norm's reciprocal root is 0;
- Euclidean distance is √max(‖a‖² + ‖b‖² − 2 · dot, 0).

A zero vector's dot is exactly zero everywhere, so its angle to any nonzero vector is exactly 1.
Distances finish from the [dots](../dots/README.md#guarantees) and norms, so `nk_angular_error_bound` and `nk_euclidean_error_bound` hold over the same ranges:

| Inputs                                | Distances in                  | Holds while        |
| :------------------------------------ | :---------------------------- | :----------------- |
| `f64`                                 | F64                           | ‖x‖² finite in F64 |
| `f32`                                 | F64                           | inputs are finite  |
| `f16`, `bf16`, `e4m3`, `e5m2`, `e3m2` | F32                           | ‖x‖² finite in F32 |
| `e2m3`, `e2m1`, integers              | F32 after exact integer terms | sums fit 32 bits   |
| `nvfp4`, `mxfp*`                      | F32 from rebased dots, norms  | inputs are finite  |

## Optimizations

### Distance-from-Dot Algebraic Reduction

`nk_angulars_packed_f32_haswell`, `nk_angulars_packed_f32_skylake`, `nk_euclideans_packed_f32_haswell`, `nk_euclideans_packed_f32_skylake` derive distance matrices from pre-packed dot product output without materializing an intermediate result matrix.
Angular distance rewrites as $1 - \text{dot}(a,b) \cdot \text{rsqrt}(\|a\|^2) \cdot \text{rsqrt}(\|b\|^2)$, replacing the division by multiplies, with separate reciprocal roots so the product of two large norms never overflows.
Euclidean distance expands the identity $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2 \cdot \text{dot}(a,b)$, requiring only one final sqrt per output element.
Both formulas decompose into: (1) a batched GEMM for all M×N dot products, (2) per-vector squared norms precomputed once during packing.
The singular `spatial/` kernels compute these three sums ($\sum a_i b_i$, $\sum a_i^2$, $\sum b_i^2$) in a single pass with three interleaved accumulators; the batched `spatials/` kernels separate them — norms are computed once per vector during packing, and dots come from the GEMM — trading register pressure for amortized cost across the full M×N output.

### Serial vs Vectorized Sqrt and Rsqrt Cost

`nk_angular_through_f32_from_dot_serial_` uses the Quake 3 fast inverse square root (magic constant `0x5F375A86`, three Newton-Raphson iterations, ~34.9 correct bits for Float32) to compute `dot * rsqrt(query_norm) * rsqrt(target_norm)`.
`nk_angular_through_f32_from_dot_haswell_` replaces this with hardware `_mm_rsqrt_ps` (~12-bit approximation, 5cy latency, 1/cy on port 0) plus one Newton-Raphson refinement step (~22–24 correct bits).
`nk_euclidean_through_f32_from_dot_serial_` computes `sqrt(x)` as `x * rsqrt(x)` — reusing the same rsqrt path.
`nk_euclidean_through_f32_from_dot_haswell_` uses exact `_mm_sqrt_ps` (11cy latency, 7cy throughput for XMM) instead of the rsqrt approximation — the subtraction $\|a\|^2 + \|b\|^2 - 2 \cdot \text{dot}$ can produce values near zero where rsqrt error would be amplified by the subsequent multiply.
For Float64, the SIMD backends use exact division and sqrt — no fast rsqrt approximation, since reaching 52 mantissa bits of precision would need 4+ Newton-Raphson iterations, negating the speed advantage — while `nk_angular_through_f64_from_dot_serial_` still routes through `nk_f64_rsqrt_serial` and its four iterations.
The 4-wide finalizer batching amortizes these costs: one rsqrt or sqrt call processes 4 output elements simultaneously, hiding the latency behind the GEMM tile's computation.

### Norm Precomputation in Packed Buffers

`nk_dots_pack_f32_serial`, `nk_dots_pack_f32_haswell`, `nk_dots_pack_bf16_haswell` compute per-column squared norms $\|b_j\|^2 = \sum_k b_{jk}^2 = \text{dot}(b_j, b_j)$ during the packing step via `nk_reduce_moments_*` primitives.
The squared norm is a self-dot-product — already a byproduct of touching every element for type conversion and layout transformation.
Angular and Euclidean finalizers read norms from packed buffer metadata, eliminating a separate O(N·K) norm pass over B.

### SME Finishing in Streaming Mode

The SME kernels finish distances in streaming SVE, where lane compares, predicate logic and selects cost more than the arithmetic: applying the zero-norm rule per lane held `nk_angulars_packed_f16_sme` at a third of its Euclidean twin's rate.
Reciprocal roots come once per row and per 256-column chunk instead, with zero norms mapped to zero roots, so a zero column lands on 1 like any exactly zero dot without a per-lane compare.
Symmetric kernels start each row at an aligned column and mask the lanes before it, as vectors straddling cache lines slowed them 1.6x, and reuse the row norms for the columns they cover.
Integer kernels form the exact $ab - d^2$ and $a + b - 2d$ in 64-bit lanes and round once into F32, as F64 runs at half the lanes and a quarter of the rate in streaming mode.
`FSQRT` throughput, about 36 cycles a vector on M5, bounds the Euclidean kernels.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by `NUMWARS_DIMS_HEIGHT`, `NUMWARS_DIMS_WIDTH`, and `NUMWARS_DIMS_DEPTH` environment variables, all set to the same value for batched distance computations over square matrices.
Columns show throughput for 256³, 1024³, and 4096³ configurations.
The throughput is measured in GSO/s as Giga Scalar Operations per Second, with $\text{ops} = 2 \cdot M \cdot N \cdot K$ complexity for computing $M \times N$ pairwise distances over $K$-dimensional vectors.

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.

#### Native

| Kernel                                          |                     256³ |                    1024³ |                    4096³ |
| :---------------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f64_serial`                 |       0.470 gso/s, 0 ulp |       0.478 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_f64_serial`              |       0.454 gso/s, 0 ulp |       0.381 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_f64_serial`               |     0.474 gso/s, 0.4 ulp |     0.407 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_f64_serial`            |     0.455 gso/s, 0.4 ulp |     0.279 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_f64_haswell`                |        5.47 gso/s, 0 ulp |        5.97 gso/s, 0 ulp |        6.08 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_haswell`             |        5.14 gso/s, 0 ulp |        5.49 gso/s, 0 ulp |        5.52 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_haswell`              |      5.52 gso/s, 0.1 ulp |      6.01 gso/s, 0.1 ulp |      6.03 gso/s, 0.1 ulp |
| `nk_euclideans_symmetric_f64_haswell`           |      5.17 gso/s, 0.2 ulp |      5.53 gso/s, 0.1 ulp |      5.50 gso/s, 0.1 ulp |
| `nk_angulars_packed_f64_skylake`                |        6.99 gso/s, 0 ulp |        8.30 gso/s, 0 ulp |        8.74 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_skylake`             |        6.57 gso/s, 0 ulp |        7.63 gso/s, 0 ulp |        7.61 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_skylake`              |      7.10 gso/s, 0.1 ulp |      8.32 gso/s, 0.1 ulp |      8.90 gso/s, 0.1 ulp |
| `nk_euclideans_symmetric_f64_skylake`           |      6.63 gso/s, 0.1 ulp |      7.61 gso/s, 0.1 ulp |      8.12 gso/s, 0.1 ulp |
| __f32__                                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f32_serial`                 |        4.66 gso/s, 0 ulp |        5.06 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_f32_serial`              |        3.49 gso/s, 0 ulp |        3.86 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_f32_serial`               |      4.39 gso/s, 0.4 ulp |      5.06 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_f32_serial`            |      3.51 gso/s, 0.4 ulp |      3.86 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_f32_haswell`                |        23.8 gso/s, 0 ulp |      27.2 gso/s, 0.1 ulp |      28.9 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_f32_haswell`             |        17.5 gso/s, 0 ulp |      20.1 gso/s, 0.1 ulp |      14.8 gso/s, 0.3 ulp |
| `nk_euclideans_packed_f32_haswell`              |      24.5 gso/s, 0.1 ulp |      27.7 gso/s, 0.9 ulp |      27.9 gso/s, 8.6 ulp |
| `nk_euclideans_symmetric_f32_haswell`           |      18.3 gso/s, 0.1 ulp |      20.6 gso/s, 0.9 ulp |      20.4 gso/s, 8.6 ulp |
| `nk_angulars_packed_f32_skylake`                |        30.2 gso/s, 0 ulp |        37.9 gso/s, 0 ulp |      38.3 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f32_skylake`             |        25.8 gso/s, 0 ulp |        31.4 gso/s, 0 ulp |      34.0 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f32_skylake`              |      31.5 gso/s, 0.1 ulp |      39.5 gso/s, 0.1 ulp |      39.8 gso/s, 1.9 ulp |
| `nk_euclideans_symmetric_f32_skylake`           |      26.7 gso/s, 0.1 ulp |      32.2 gso/s, 0.1 ulp |      37.0 gso/s, 1.9 ulp |
| __bf16__                                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_bf16_serial`                |      5.27 gso/s, 0.1 ulp |        5.64 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_bf16_serial`             |        4.45 gso/s, 0 ulp |        5.30 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_bf16_serial`              |      5.06 gso/s, 0.5 ulp |      5.57 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_bf16_serial`           |      4.52 gso/s, 0.5 ulp |      5.32 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_bf16_haswell`               |      46.4 gso/s, 0.1 ulp |        53.4 gso/s, 0 ulp |      60.4 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_bf16_haswell`            |      39.5 gso/s, 0.1 ulp |        50.1 gso/s, 0 ulp |      37.4 gso/s, 0.1 ulp |
| `nk_euclideans_packed_bf16_haswell`             |      49.3 gso/s, 0.3 ulp |      56.1 gso/s, 0.4 ulp |      49.7 gso/s, 2.6 ulp |
| `nk_euclideans_symmetric_bf16_haswell`          |      41.7 gso/s, 0.3 ulp |      51.6 gso/s, 0.4 ulp |      54.2 gso/s, 2.6 ulp |
| `nk_angulars_packed_bf16_skylake`               |      66.1 gso/s, 0.1 ulp |        92.8 gso/s, 0 ulp |         102 gso/s, 0 ulp |
| `nk_angulars_symmetric_bf16_skylake`            |      60.5 gso/s, 0.1 ulp |        84.5 gso/s, 0 ulp |        89.7 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_skylake`             |      72.8 gso/s, 0.2 ulp |      99.8 gso/s, 0.3 ulp |       105 gso/s, 1.4 ulp |
| `nk_euclideans_symmetric_bf16_skylake`          |      64.9 gso/s, 0.3 ulp |      87.2 gso/s, 0.3 ulp |      97.3 gso/s, 1.4 ulp |
| `nk_angulars_packed_bf16_genoa`                 |      56.5 gso/s, 0.1 ulp |        74.3 gso/s, 0 ulp |        83.5 gso/s, 0 ulp |
| `nk_angulars_symmetric_bf16_genoa`              |      53.3 gso/s, 0.1 ulp |        67.6 gso/s, 0 ulp |        72.4 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_genoa`               |      61.0 gso/s, 0.2 ulp |      79.3 gso/s, 0.3 ulp |      70.2 gso/s, 1.4 ulp |
| `nk_euclideans_symmetric_bf16_genoa`            |      56.0 gso/s, 0.3 ulp |      69.4 gso/s, 0.3 ulp |      69.2 gso/s, 1.4 ulp |
| `nk_angulars_packed_bf16_sapphireamx`           |       282 gso/s, 0.1 ulp |         486 gso/s, 0 ulp |       457 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_bf16_sapphireamx`        |         104 gso/s, 0 ulp |        10.8 gso/s, 0 ulp |       147 gso/s, 0.1 ulp |
| `nk_euclideans_packed_bf16_sapphireamx`         |       283 gso/s, 0.2 ulp |       392 gso/s, 0.3 ulp |       675 gso/s, 1.4 ulp |
| `nk_euclideans_symmetric_bf16_sapphireamx`      |       105 gso/s, 0.3 ulp |       117 gso/s, 0.3 ulp |       208 gso/s, 1.4 ulp |
| __f16__                                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f16_serial`                 |      2.48 gso/s, 0.1 ulp |      2.59 gso/s, 0.1 ulp |                        ⋯ |
| `nk_angulars_symmetric_f16_serial`              |     0.643 gso/s, 0.1 ulp |     0.709 gso/s, 0.1 ulp |                        ⋯ |
| `nk_euclideans_packed_f16_serial`               |      2.50 gso/s, 0.5 ulp |      2.57 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_f16_serial`            |     0.687 gso/s, 0.5 ulp |     0.717 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_f16_haswell`                |      44.7 gso/s, 0.1 ulp |      50.7 gso/s, 0.1 ulp |      52.1 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_haswell`             |      37.3 gso/s, 0.1 ulp |      47.4 gso/s, 0.1 ulp |      34.4 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_haswell`              |      46.8 gso/s, 0.4 ulp |      52.3 gso/s, 0.4 ulp |      50.4 gso/s, 1.7 ulp |
| `nk_euclideans_symmetric_f16_haswell`           |      38.5 gso/s, 0.3 ulp |      48.7 gso/s, 0.3 ulp |      50.8 gso/s, 1.7 ulp |
| `nk_angulars_packed_f16_skylake`                |      63.1 gso/s, 0.1 ulp |      89.0 gso/s, 0.1 ulp |        88.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_f16_skylake`             |      46.1 gso/s, 0.1 ulp |      58.4 gso/s, 0.1 ulp |        70.4 gso/s, 0 ulp |
| `nk_euclideans_packed_f16_skylake`              |      69.1 gso/s, 0.3 ulp |      95.7 gso/s, 0.5 ulp |      90.9 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_f16_skylake`           |      49.0 gso/s, 0.3 ulp |      60.6 gso/s, 0.5 ulp |      74.8 gso/s, 0.3 ulp |
| `nk_angulars_packed_f16_graniteamx`             |       239 gso/s, 0.1 ulp |       337 gso/s, 0.1 ulp |         341 gso/s, 0 ulp |
| `nk_angulars_symmetric_f16_graniteamx`          |      93.0 gso/s, 0.1 ulp |       120 gso/s, 0.1 ulp |         118 gso/s, 0 ulp |
| `nk_euclideans_packed_f16_graniteamx`           |       296 gso/s, 0.3 ulp |       346 gso/s, 0.5 ulp |       341 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_f16_graniteamx`        |       106 gso/s, 0.3 ulp |       126 gso/s, 0.5 ulp |       121 gso/s, 0.3 ulp |
| __e5m2__                                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e5m2_serial`                |        1.46 gso/s, 0 ulp |        1.49 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_e5m2_serial`             |        1.25 gso/s, 0 ulp |        1.15 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_e5m2_serial`              |      1.46 gso/s, 0.4 ulp |      1.18 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_e5m2_serial`           |      1.25 gso/s, 0.5 ulp |      1.29 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_e5m2_haswell`               |        33.2 gso/s, 0 ulp |        37.0 gso/s, 0 ulp |        39.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_haswell`            |        31.4 gso/s, 0 ulp |        39.0 gso/s, 0 ulp |        35.8 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_haswell`             |      34.4 gso/s, 0.2 ulp |      38.6 gso/s, 0.2 ulp |      39.3 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e5m2_haswell`          |      33.0 gso/s, 0.2 ulp |      38.9 gso/s, 0.2 ulp |      40.6 gso/s, 0.2 ulp |
| `nk_angulars_packed_e5m2_skylake`               |        40.9 gso/s, 0 ulp |        49.0 gso/s, 0 ulp |        51.9 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_skylake`            |        41.1 gso/s, 0 ulp |        51.1 gso/s, 0 ulp |        55.2 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_skylake`             |      42.6 gso/s, 0.2 ulp |      51.8 gso/s, 0.2 ulp |      52.9 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e5m2_skylake`          |      43.2 gso/s, 0.2 ulp |      52.1 gso/s, 0.2 ulp |      54.6 gso/s, 0.2 ulp |
| `nk_angulars_packed_e5m2_genoa`                 |        33.5 gso/s, 0 ulp |        39.0 gso/s, 0 ulp |        42.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_genoa`              |        24.8 gso/s, 0 ulp |        28.0 gso/s, 0 ulp |        25.7 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_genoa`               |      35.2 gso/s, 0.2 ulp |      40.2 gso/s, 0.2 ulp |      36.6 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e5m2_genoa`            |      25.3 gso/s, 0.2 ulp |      28.5 gso/s, 0.2 ulp |      24.2 gso/s, 0.2 ulp |
| `nk_angulars_packed_e5m2_sapphireamx`           |         164 gso/s, 0 ulp |         290 gso/s, 0 ulp |         246 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_sapphireamx`        |        75.1 gso/s, 0 ulp |        12.2 gso/s, 0 ulp |        75.8 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_sapphireamx`         |       161 gso/s, 0.2 ulp |       223 gso/s, 0.2 ulp |       340 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e5m2_sapphireamx`      |      76.0 gso/s, 0.2 ulp |      77.8 gso/s, 0.2 ulp |       112 gso/s, 0.2 ulp |
| `nk_angulars_packed_e5m2_graniteamx`            |         188 gso/s, 0 ulp |         301 gso/s, 0 ulp |         289 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_graniteamx`         |         102 gso/s, 0 ulp |         134 gso/s, 0 ulp |         124 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_graniteamx`          |       227 gso/s, 0.2 ulp |       313 gso/s, 0.2 ulp |       329 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e5m2_graniteamx`       |       103 gso/s, 0.2 ulp |       136 gso/s, 0.2 ulp |       144 gso/s, 0.2 ulp |
| __e4m3__                                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e4m3_serial`                |       0.441 gso/s, 0 ulp |       0.449 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_e4m3_serial`             |     0.427 gso/s, 0.1 ulp |       0.446 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_e4m3_serial`              |     0.448 gso/s, 0.5 ulp |     0.450 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_e4m3_serial`           |     0.440 gso/s, 0.5 ulp |     0.446 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_e4m3_haswell`               |      15.3 gso/s, 0.1 ulp |        16.0 gso/s, 0 ulp |        16.3 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_haswell`            |        14.8 gso/s, 0 ulp |        16.0 gso/s, 0 ulp |        16.2 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_haswell`             |      15.5 gso/s, 0.2 ulp |      16.2 gso/s, 0.2 ulp |      16.5 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_e4m3_haswell`          |      15.1 gso/s, 0.2 ulp |      16.1 gso/s, 0.2 ulp |      14.0 gso/s, 0.3 ulp |
| `nk_angulars_packed_e4m3_skylake`               |      33.8 gso/s, 0.1 ulp |        38.6 gso/s, 0 ulp |        41.3 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_skylake`            |        23.5 gso/s, 0 ulp |        26.6 gso/s, 0 ulp |        27.3 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_skylake`             |      35.5 gso/s, 0.2 ulp |      40.4 gso/s, 0.2 ulp |      42.1 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e4m3_skylake`          |      24.0 gso/s, 0.2 ulp |      26.8 gso/s, 0.2 ulp |      27.6 gso/s, 0.2 ulp |
| `nk_angulars_packed_e4m3_genoa`                 |        34.9 gso/s, 0 ulp |        39.8 gso/s, 0 ulp |        43.4 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_genoa`              |        25.9 gso/s, 0 ulp |        29.1 gso/s, 0 ulp |        30.1 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_genoa`               |      36.5 gso/s, 0.2 ulp |      41.4 gso/s, 0.2 ulp |      37.8 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e4m3_genoa`            |      26.4 gso/s, 0.2 ulp |      29.5 gso/s, 0.2 ulp |      25.3 gso/s, 0.2 ulp |
| `nk_angulars_packed_e4m3_sapphireamx`           |         164 gso/s, 0 ulp |         290 gso/s, 0 ulp |         248 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_sapphireamx`        |        74.0 gso/s, 0 ulp |        16.1 gso/s, 0 ulp |         116 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_sapphireamx`         |       159 gso/s, 0.2 ulp |       227 gso/s, 0.2 ulp |       337 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e4m3_sapphireamx`      |      74.8 gso/s, 0.2 ulp |      80.0 gso/s, 0.2 ulp |       117 gso/s, 0.2 ulp |
| __e3m2__                                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e3m2_serial`                |        1.38 gso/s, 0 ulp |        1.44 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_e3m2_serial`             |       0.854 gso/s, 0 ulp |       0.925 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_e3m2_serial`              |      1.42 gso/s, 0.4 ulp |      1.45 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_e3m2_serial`           |     0.861 gso/s, 0.4 ulp |     0.872 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_e3m2_haswell`               |        21.9 gso/s, 0 ulp |        23.9 gso/s, 0 ulp |        23.9 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_haswell`            |        21.9 gso/s, 0 ulp |        25.9 gso/s, 0 ulp |        26.3 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_haswell`             |        22.7 gso/s, 0 ulp |      24.6 gso/s, 0.1 ulp |      25.9 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e3m2_haswell`          |        22.0 gso/s, 0 ulp |      25.7 gso/s, 0.2 ulp |      26.1 gso/s, 0.2 ulp |
| `nk_angulars_packed_e3m2_skylake`               |        34.3 gso/s, 0 ulp |        39.7 gso/s, 0 ulp |        42.9 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_skylake`            |        32.3 gso/s, 0 ulp |        41.0 gso/s, 0 ulp |        43.4 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_skylake`             |        36.2 gso/s, 0 ulp |      41.7 gso/s, 0.1 ulp |      43.6 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e3m2_skylake`          |        33.2 gso/s, 0 ulp |      41.7 gso/s, 0.2 ulp |      42.9 gso/s, 0.2 ulp |
| `nk_angulars_packed_e3m2_sapphireamx`           |         179 gso/s, 0 ulp |         360 gso/s, 0 ulp |         307 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_sapphireamx`        |        91.6 gso/s, 0 ulp |        53.1 gso/s, 0 ulp |         167 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_sapphireamx`         |         176 gso/s, 0 ulp |       234 gso/s, 0.1 ulp |       379 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e3m2_sapphireamx`      |        93.5 gso/s, 0 ulp |      77.9 gso/s, 0.2 ulp |       167 gso/s, 0.2 ulp |
| __e2m3__                                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_serial`                |        1.40 gso/s, 0 ulp |        1.45 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_e2m3_serial`             |       0.846 gso/s, 0 ulp |       0.934 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_e2m3_serial`              |      1.41 gso/s, 0.5 ulp |      1.44 gso/s, 0.5 ulp |                        ⋯ |
| `nk_euclideans_symmetric_e2m3_serial`           |     0.861 gso/s, 0.6 ulp |     0.877 gso/s, 0.6 ulp |                        ⋯ |
| `nk_angulars_packed_e2m3_haswell`               |        33.1 gso/s, 0 ulp |        35.9 gso/s, 0 ulp |        34.7 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_haswell`            |        31.1 gso/s, 0 ulp |        37.1 gso/s, 0 ulp |        39.4 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_haswell`             |        34.3 gso/s, 0 ulp |        36.6 gso/s, 0 ulp |        38.7 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_haswell`          |        32.1 gso/s, 0 ulp |        38.7 gso/s, 0 ulp |        39.8 gso/s, 0 ulp |
| `nk_angulars_packed_e2m3_skylake`               |        55.0 gso/s, 0 ulp |        64.1 gso/s, 0 ulp |        71.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_skylake`            |        52.7 gso/s, 0 ulp |        73.5 gso/s, 0 ulp |        79.2 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_skylake`             |        59.3 gso/s, 0 ulp |        67.4 gso/s, 0 ulp |        73.3 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_skylake`          |        56.2 gso/s, 0 ulp |        74.7 gso/s, 0 ulp |        80.2 gso/s, 0 ulp |
| `nk_angulars_packed_e2m3_sapphireamx`           |         224 gso/s, 0 ulp |         683 gso/s, 0 ulp |         598 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_sapphireamx`        |         124 gso/s, 0 ulp |         147 gso/s, 0 ulp |         314 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_sapphireamx`         |         221 gso/s, 0 ulp |         538 gso/s, 0 ulp |         791 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_sapphireamx`      |         129 gso/s, 0 ulp |         170 gso/s, 0 ulp |         323 gso/s, 0 ulp |
| `nk_angulars_packed_e2m3_alder`                 |        39.1 gso/s, 0 ulp |        43.0 gso/s, 0 ulp |        46.5 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_alder`              |        33.7 gso/s, 0 ulp |        41.5 gso/s, 0 ulp |        35.1 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_alder`               |        41.5 gso/s, 0 ulp |        44.3 gso/s, 0 ulp |        36.8 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_alder`            |        34.6 gso/s, 0 ulp |        42.3 gso/s, 0 ulp |        35.1 gso/s, 0 ulp |
| __i8__                                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_serial`                  |        4.39 gso/s, 0 ulp |        3.86 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_i8_serial`               |        4.16 gso/s, 0 ulp |        4.05 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_i8_serial`                |        4.60 gso/s, 0 ulp |        4.77 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_symmetric_i8_serial`             |        4.26 gso/s, 0 ulp |        4.57 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_packed_i8_haswell`                 |      48.2 gso/s, 0.2 ulp |      58.0 gso/s, 0.2 ulp |      60.1 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_i8_haswell`              |      47.4 gso/s, 0.2 ulp |      62.7 gso/s, 0.2 ulp |      69.2 gso/s, 0.3 ulp |
| `nk_euclideans_packed_i8_haswell`               |        53.1 gso/s, 0 ulp |        60.5 gso/s, 0 ulp |      66.0 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_i8_haswell`            |        52.1 gso/s, 0 ulp |        64.2 gso/s, 0 ulp |      69.7 gso/s, 0.2 ulp |
| `nk_angulars_packed_i8_icelake`                 |       115 gso/s, 0.2 ulp |       200 gso/s, 0.2 ulp |       255 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_i8_icelake`              |      85.5 gso/s, 0.2 ulp |       216 gso/s, 0.2 ulp |       294 gso/s, 0.3 ulp |
| `nk_euclideans_packed_i8_icelake`               |         146 gso/s, 0 ulp |         224 gso/s, 0 ulp |       289 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_i8_icelake`            |        98.1 gso/s, 0 ulp |         241 gso/s, 0 ulp |       307 gso/s, 0.2 ulp |
| `nk_angulars_packed_i8_sapphireamx`             |       323 gso/s, 0.2 ulp |       597 gso/s, 0.2 ulp |       868 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_i8_sapphireamx`          |       146 gso/s, 0.2 ulp |       180 gso/s, 0.2 ulp |       372 gso/s, 0.3 ulp |
| `nk_euclideans_packed_i8_sapphireamx`           |         367 gso/s, 0 ulp |         510 gso/s, 0 ulp |     1,218 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_i8_sapphireamx`        |         136 gso/s, 0 ulp |         212 gso/s, 0 ulp |       229 gso/s, 0.2 ulp |
| `nk_angulars_packed_i8_alder`                   |      86.0 gso/s, 0.2 ulp |       111 gso/s, 0.2 ulp |       114 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_i8_alder`                |      66.3 gso/s, 0.2 ulp |       128 gso/s, 0.2 ulp |       127 gso/s, 0.3 ulp |
| `nk_euclideans_packed_i8_alder`                 |         104 gso/s, 0 ulp |         121 gso/s, 0 ulp |       117 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_i8_alder`              |        75.2 gso/s, 0 ulp |         136 gso/s, 0 ulp |       132 gso/s, 0.2 ulp |
| __u8__                                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`                  |        4.54 gso/s, 0 ulp |        3.85 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_u8_serial`               |        4.11 gso/s, 0 ulp |        4.43 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_u8_serial`                |        4.60 gso/s, 0 ulp |        4.86 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_symmetric_u8_serial`             |        4.18 gso/s, 0 ulp |        4.54 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_packed_u8_haswell`                 |      49.3 gso/s, 0.4 ulp |      58.7 gso/s, 0.4 ulp |      65.1 gso/s, 0.4 ulp |
| `nk_angulars_symmetric_u8_haswell`              |      47.7 gso/s, 0.4 ulp |      62.6 gso/s, 0.4 ulp |      69.3 gso/s, 0.4 ulp |
| `nk_euclideans_packed_u8_haswell`               |        53.1 gso/s, 0 ulp |        60.6 gso/s, 0 ulp |      65.6 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_u8_haswell`            |        51.8 gso/s, 0 ulp |        63.9 gso/s, 0 ulp |      69.4 gso/s, 0.2 ulp |
| `nk_angulars_packed_u8_icelake`                 |       121 gso/s, 0.4 ulp |       201 gso/s, 0.4 ulp |       267 gso/s, 0.4 ulp |
| `nk_angulars_symmetric_u8_icelake`              |      89.5 gso/s, 0.4 ulp |       224 gso/s, 0.4 ulp |       301 gso/s, 0.4 ulp |
| `nk_euclideans_packed_u8_icelake`               |         143 gso/s, 0 ulp |         223 gso/s, 0 ulp |       285 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_u8_icelake`            |        99.6 gso/s, 0 ulp |         236 gso/s, 0 ulp |       305 gso/s, 0.2 ulp |
| `nk_angulars_packed_u8_sapphireamx`             |       338 gso/s, 0.4 ulp |       296 gso/s, 0.4 ulp |       846 gso/s, 0.4 ulp |
| `nk_angulars_symmetric_u8_sapphireamx`          |       144 gso/s, 0.4 ulp |       189 gso/s, 0.4 ulp |       376 gso/s, 0.4 ulp |
| `nk_euclideans_packed_u8_sapphireamx`           |         369 gso/s, 0 ulp |         509 gso/s, 0 ulp |     1,239 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_u8_sapphireamx`        |         136 gso/s, 0 ulp |         225 gso/s, 0 ulp |       229 gso/s, 0.2 ulp |
| `nk_angulars_packed_u8_alder`                   |      86.1 gso/s, 0.4 ulp |       113 gso/s, 0.4 ulp |       115 gso/s, 0.4 ulp |
| `nk_angulars_symmetric_u8_alder`                |      68.1 gso/s, 0.4 ulp |       130 gso/s, 0.4 ulp |       132 gso/s, 0.4 ulp |
| `nk_euclideans_packed_u8_alder`                 |        97.7 gso/s, 0 ulp |         122 gso/s, 0 ulp |       117 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_u8_alder`              |        73.6 gso/s, 0 ulp |         136 gso/s, 0 ulp |       132 gso/s, 0.2 ulp |
| __i4__                                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i4_serial`                  |        1.22 gso/s, 0 ulp |        1.22 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_i4_serial`               |        1.41 gso/s, 0 ulp |        1.44 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_i4_serial`                |        1.23 gso/s, 0 ulp |        1.23 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_symmetric_i4_serial`             |        1.40 gso/s, 0 ulp |        1.45 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_packed_i4_icelake`                 |      12.0 gso/s, 0.3 ulp |      12.7 gso/s, 0.3 ulp |      13.1 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_i4_icelake`              |      81.4 gso/s, 0.3 ulp |       180 gso/s, 0.3 ulp |       238 gso/s, 0.3 ulp |
| `nk_euclideans_packed_i4_icelake`               |        12.2 gso/s, 0 ulp |        12.9 gso/s, 0 ulp |        13.2 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i4_icelake`            |        93.9 gso/s, 0 ulp |         201 gso/s, 0 ulp |         244 gso/s, 0 ulp |
| `nk_angulars_packed_i4_haswell`                 |      11.4 gso/s, 0.3 ulp |      11.7 gso/s, 0.3 ulp |      10.7 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_i4_haswell`              |      27.3 gso/s, 0.3 ulp |      32.5 gso/s, 0.3 ulp |      33.4 gso/s, 0.3 ulp |
| `nk_euclideans_packed_i4_haswell`               |        11.6 gso/s, 0 ulp |        11.9 gso/s, 0 ulp |        12.3 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i4_haswell`            |        27.7 gso/s, 0 ulp |        33.0 gso/s, 0 ulp |        33.7 gso/s, 0 ulp |
| __u4__                                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u4_serial`                  |        1.91 gso/s, 0 ulp |        1.92 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_u4_serial`               |        3.02 gso/s, 0 ulp |        2.60 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_u4_serial`                |        1.94 gso/s, 0 ulp |        1.97 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_symmetric_u4_serial`             |        3.02 gso/s, 0 ulp |        3.18 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_packed_u4_icelake`                 |       115 gso/s, 0.4 ulp |       209 gso/s, 0.3 ulp |       289 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u4_icelake`              |       108 gso/s, 0.3 ulp |       237 gso/s, 0.3 ulp |       317 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u4_icelake`               |         134 gso/s, 0 ulp |         230 gso/s, 0 ulp |         308 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u4_icelake`            |         123 gso/s, 0 ulp |         260 gso/s, 0 ulp |         323 gso/s, 0 ulp |
| `nk_angulars_packed_u4_haswell`                 |      58.3 gso/s, 0.4 ulp |      70.2 gso/s, 0.3 ulp |      77.4 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u4_haswell`              |      51.0 gso/s, 0.3 ulp |      70.0 gso/s, 0.3 ulp |      74.7 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u4_haswell`               |        63.5 gso/s, 0 ulp |        73.1 gso/s, 0 ulp |        78.2 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u4_haswell`            |        53.5 gso/s, 0 ulp |        71.1 gso/s, 0 ulp |        75.6 gso/s, 0 ulp |
| __e2m1__                                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m1_sapphireamx`           |         165 gso/s, 0 ulp |         533 gso/s, 0 ulp |         566 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m1_sapphireamx`        |        69.5 gso/s, 0 ulp |         130 gso/s, 0 ulp |         298 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m1_sapphireamx`         |         168 gso/s, 0 ulp |         413 gso/s, 0 ulp |         661 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m1_sapphireamx`      |        68.9 gso/s, 0 ulp |         140 gso/s, 0 ulp |         294 gso/s, 0 ulp |
| `nk_angulars_packed_e2m1_serial`                |        1.09 gso/s, 0 ulp |        1.11 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_e2m1_serial`             |        1.60 gso/s, 0 ulp |        1.56 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_e2m1_serial`              |      1.09 gso/s, 0.4 ulp |      1.11 gso/s, 0.3 ulp |                        ⋯ |
| `nk_euclideans_symmetric_e2m1_serial`           |      1.59 gso/s, 0.4 ulp |      1.64 gso/s, 0.3 ulp |                        ⋯ |
| `nk_angulars_packed_e2m1_haswell`               |        63.5 gso/s, 0 ulp |        85.5 gso/s, 0 ulp |        90.4 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m1_haswell`            |        42.9 gso/s, 0 ulp |        76.8 gso/s, 0 ulp |        91.3 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m1_haswell`             |        67.8 gso/s, 0 ulp |        89.9 gso/s, 0 ulp |         102 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m1_haswell`          |        45.6 gso/s, 0 ulp |        79.0 gso/s, 0 ulp |        90.9 gso/s, 0 ulp |
| `nk_angulars_packed_e2m1_skylake`               |        65.3 gso/s, 0 ulp |        95.0 gso/s, 0 ulp |         112 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m1_skylake`            |        49.6 gso/s, 0 ulp |        88.5 gso/s, 0 ulp |         109 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m1_skylake`             |        69.0 gso/s, 0 ulp |         101 gso/s, 0 ulp |         116 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m1_skylake`          |        51.8 gso/s, 0 ulp |        91.7 gso/s, 0 ulp |         114 gso/s, 0 ulp |
| `nk_angulars_packed_e2m1_alder`                 |        75.0 gso/s, 0 ulp |         105 gso/s, 0 ulp |         129 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m1_alder`              |        43.4 gso/s, 0 ulp |        82.7 gso/s, 0 ulp |        84.4 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m1_alder`               |        84.2 gso/s, 0 ulp |         113 gso/s, 0 ulp |         107 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m1_alder`            |        46.1 gso/s, 0 ulp |        84.5 gso/s, 0 ulp |        85.3 gso/s, 0 ulp |
| __nvfp4__                                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_nvfp4_sapphireamx`          |         164 gso/s, 0 ulp |         331 gso/s, 0 ulp |         499 gso/s, 0 ulp |
| `nk_angulars_symmetric_nvfp4_sapphireamx`       |        46.1 gso/s, 0 ulp |        43.4 gso/s, 0 ulp |        65.2 gso/s, 0 ulp |
| `nk_euclideans_packed_nvfp4_sapphireamx`        |         154 gso/s, 0 ulp |         323 gso/s, 0 ulp |         502 gso/s, 0 ulp |
| `nk_euclideans_symmetric_nvfp4_sapphireamx`     |        45.5 gso/s, 0 ulp |        44.9 gso/s, 0 ulp |        67.5 gso/s, 0 ulp |
| `nk_angulars_packed_nvfp4_serial`               |       0.191 gso/s, 0 ulp |       0.188 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_nvfp4_serial`            |       0.191 gso/s, 0 ulp |       0.189 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_nvfp4_serial`             |     0.186 gso/s, 0.4 ulp |     0.191 gso/s, 0.3 ulp |                        ⋯ |
| `nk_euclideans_symmetric_nvfp4_serial`          |     0.190 gso/s, 0.4 ulp |     0.192 gso/s, 0.3 ulp |                        ⋯ |
| `nk_angulars_packed_nvfp4_skylake`              |        24.9 gso/s, 0 ulp |        27.7 gso/s, 0 ulp |        28.0 gso/s, 0 ulp |
| `nk_angulars_symmetric_nvfp4_skylake`           |        22.9 gso/s, 0 ulp |        25.9 gso/s, 0 ulp |        24.9 gso/s, 0 ulp |
| `nk_euclideans_packed_nvfp4_skylake`            |        24.9 gso/s, 0 ulp |        27.9 gso/s, 0 ulp |        28.2 gso/s, 0 ulp |
| `nk_euclideans_symmetric_nvfp4_skylake`         |        22.8 gso/s, 0 ulp |        26.0 gso/s, 0 ulp |        24.9 gso/s, 0 ulp |
| __mxfp4__                                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp4_sapphireamx`          |         187 gso/s, 0 ulp |         410 gso/s, 0 ulp |         789 gso/s, 0 ulp |
| `nk_angulars_symmetric_mxfp4_sapphireamx`       |        49.2 gso/s, 0 ulp |        35.6 gso/s, 0 ulp |        67.4 gso/s, 0 ulp |
| `nk_euclideans_packed_mxfp4_sapphireamx`        |         177 gso/s, 0 ulp |         412 gso/s, 0 ulp |         822 gso/s, 0 ulp |
| `nk_euclideans_symmetric_mxfp4_sapphireamx`     |        41.3 gso/s, 0 ulp |        47.3 gso/s, 0 ulp |        39.1 gso/s, 0 ulp |
| `nk_angulars_packed_mxfp4_serial`               |       0.189 gso/s, 0 ulp |       0.188 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_mxfp4_serial`            |       0.188 gso/s, 0 ulp |       0.124 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_mxfp4_serial`             |     0.188 gso/s, 0.4 ulp |     0.187 gso/s, 0.3 ulp |                        ⋯ |
| `nk_euclideans_symmetric_mxfp4_serial`          |     0.189 gso/s, 0.4 ulp |     0.189 gso/s, 0.3 ulp |                        ⋯ |
| `nk_angulars_packed_mxfp4_skylake`              |        24.6 gso/s, 0 ulp |        27.9 gso/s, 0 ulp |        28.0 gso/s, 0 ulp |
| `nk_angulars_symmetric_mxfp4_skylake`           |        23.3 gso/s, 0 ulp |        26.3 gso/s, 0 ulp |        24.7 gso/s, 0 ulp |
| `nk_euclideans_packed_mxfp4_skylake`            |        24.4 gso/s, 0 ulp |        27.6 gso/s, 0 ulp |        27.8 gso/s, 0 ulp |
| `nk_euclideans_symmetric_mxfp4_skylake`         |        22.9 gso/s, 0 ulp |        26.1 gso/s, 0 ulp |        24.7 gso/s, 0 ulp |
| __mxfp8e4m3__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp8e4m3_sapphireamx`      |         141 gso/s, 0 ulp |         280 gso/s, 0 ulp |         382 gso/s, 0 ulp |
| `nk_angulars_symmetric_mxfp8e4m3_sapphireamx`   |        39.0 gso/s, 0 ulp |        25.7 gso/s, 0 ulp |        57.7 gso/s, 0 ulp |
| `nk_euclideans_packed_mxfp8e4m3_sapphireamx`    |       133 gso/s, 0.2 ulp |       281 gso/s, 0.2 ulp |       397 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_mxfp8e4m3_sapphireamx` |      36.1 gso/s, 0.2 ulp |      36.9 gso/s, 0.2 ulp |      32.5 gso/s, 0.2 ulp |
| `nk_angulars_packed_mxfp8e4m3_serial`           |       0.265 gso/s, 0 ulp |       0.212 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_mxfp8e4m3_serial`        |       0.271 gso/s, 0 ulp |       0.146 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_mxfp8e4m3_serial`         |     0.264 gso/s, 0.5 ulp |     0.259 gso/s, 0.5 ulp |                        ⋯ |
| `nk_euclideans_symmetric_mxfp8e4m3_serial`      |     0.257 gso/s, 0.5 ulp |     0.262 gso/s, 0.5 ulp |                        ⋯ |
| `nk_angulars_packed_mxfp8e4m3_skylake`          |        15.8 gso/s, 0 ulp |        16.9 gso/s, 0 ulp |        16.8 gso/s, 0 ulp |
| `nk_angulars_symmetric_mxfp8e4m3_skylake`       |        14.6 gso/s, 0 ulp |        16.0 gso/s, 0 ulp |        15.1 gso/s, 0 ulp |
| `nk_euclideans_packed_mxfp8e4m3_skylake`        |      15.6 gso/s, 0.2 ulp |      16.8 gso/s, 0.3 ulp |      16.7 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_mxfp8e4m3_skylake`     |      14.7 gso/s, 0.2 ulp |      16.0 gso/s, 0.2 ulp |      14.9 gso/s, 0.4 ulp |
| __mxfp8e5m2__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp8e5m2_sapphireamx`      |         141 gso/s, 0 ulp |         274 gso/s, 0 ulp |         376 gso/s, 0 ulp |
| `nk_angulars_symmetric_mxfp8e5m2_sapphireamx`   |        44.7 gso/s, 0 ulp |        30.6 gso/s, 0 ulp |        66.4 gso/s, 0 ulp |
| `nk_euclideans_packed_mxfp8e5m2_sapphireamx`    |       134 gso/s, 0.2 ulp |       277 gso/s, 0.2 ulp |       403 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_mxfp8e5m2_sapphireamx` |      38.3 gso/s, 0.2 ulp |      38.2 gso/s, 0.2 ulp |      39.6 gso/s, 0.2 ulp |
| `nk_angulars_packed_mxfp8e5m2_serial`           |       0.379 gso/s, 0 ulp |       0.289 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_mxfp8e5m2_serial`        |       0.378 gso/s, 0 ulp |       0.252 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_mxfp8e5m2_serial`         |     0.386 gso/s, 0.4 ulp |     0.392 gso/s, 0.5 ulp |                        ⋯ |
| `nk_euclideans_symmetric_mxfp8e5m2_serial`      |     0.395 gso/s, 0.5 ulp |     0.403 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_mxfp8e5m2_skylake`          |        18.8 gso/s, 0 ulp |        20.2 gso/s, 0 ulp |        20.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_mxfp8e5m2_skylake`       |        17.7 gso/s, 0 ulp |        19.4 gso/s, 0 ulp |        18.1 gso/s, 0 ulp |
| `nk_euclideans_packed_mxfp8e5m2_skylake`        |      18.9 gso/s, 0.2 ulp |      20.5 gso/s, 0.3 ulp |      20.6 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_mxfp8e5m2_skylake`     |      17.5 gso/s, 0.2 ulp |      19.3 gso/s, 0.2 ulp |      17.6 gso/s, 0.3 ulp |
| __mxfp6e2m3__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp6e2m3_serial`           |       0.377 gso/s, 0 ulp |       0.384 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_mxfp6e2m3_serial`        |       0.375 gso/s, 0 ulp |       0.379 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_mxfp6e2m3_serial`         |     0.398 gso/s, 0.5 ulp |     0.296 gso/s, 0.5 ulp |                        ⋯ |
| `nk_euclideans_symmetric_mxfp6e2m3_serial`      |     0.393 gso/s, 0.6 ulp |     0.286 gso/s, 0.6 ulp |                        ⋯ |
| `nk_angulars_packed_mxfp6e2m3_skylake`          |        13.5 gso/s, 0 ulp |        14.3 gso/s, 0 ulp |        14.3 gso/s, 0 ulp |
| `nk_angulars_symmetric_mxfp6e2m3_skylake`       |        12.7 gso/s, 0 ulp |        13.6 gso/s, 0 ulp |        12.4 gso/s, 0 ulp |
| `nk_euclideans_packed_mxfp6e2m3_skylake`        |        13.4 gso/s, 0 ulp |        14.1 gso/s, 0 ulp |        14.2 gso/s, 0 ulp |
| `nk_euclideans_symmetric_mxfp6e2m3_skylake`     |        12.6 gso/s, 0 ulp |        13.5 gso/s, 0 ulp |        12.7 gso/s, 0 ulp |
| __mxfp6e3m2__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp6e3m2_serial`           |       0.378 gso/s, 0 ulp |       0.283 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_mxfp6e3m2_serial`        |       0.376 gso/s, 0 ulp |       0.382 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_mxfp6e3m2_serial`         |     0.392 gso/s, 0.4 ulp |     0.395 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_mxfp6e3m2_serial`      |     0.398 gso/s, 0.4 ulp |     0.405 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_mxfp6e3m2_skylake`          |        13.5 gso/s, 0 ulp |        14.4 gso/s, 0 ulp |        14.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_mxfp6e3m2_skylake`       |        12.6 gso/s, 0 ulp |        13.6 gso/s, 0 ulp |        12.9 gso/s, 0 ulp |
| `nk_euclideans_packed_mxfp6e3m2_skylake`        |        13.4 gso/s, 0 ulp |      14.3 gso/s, 0.1 ulp |        14.2 gso/s, 1 ulp |
| `nk_euclideans_symmetric_mxfp6e3m2_skylake`     |        12.6 gso/s, 0 ulp |      13.6 gso/s, 0.2 ulp |      12.7 gso/s, 0.4 ulp |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel                                     |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f64_serial`            |       0.440 gso/s, 0 ulp |       0.445 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_f64_serial`         |       0.438 gso/s, 0 ulp |       0.440 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_f64_serial`          |     0.343 gso/s, 0.4 ulp |     0.441 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_f64_serial`       |     0.336 gso/s, 0.4 ulp |     0.441 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_f64_v128relaxed`       |        1.70 gso/s, 0 ulp |        1.74 gso/s, 0 ulp |        1.65 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_v128relaxed`    |        1.68 gso/s, 0 ulp |        1.75 gso/s, 0 ulp |        1.71 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_v128relaxed`     |      1.68 gso/s, 0.4 ulp |      1.74 gso/s, 0.4 ulp |      1.33 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_f64_v128relaxed`  |      1.64 gso/s, 0.4 ulp |      1.73 gso/s, 0.4 ulp |      1.66 gso/s, 0.3 ulp |
| __f32__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f32_serial`            |        3.65 gso/s, 0 ulp |        3.72 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_f32_serial`         |        3.65 gso/s, 0 ulp |        4.01 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_f32_serial`          |      3.62 gso/s, 0.4 ulp |      3.82 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_f32_serial`       |      3.65 gso/s, 0.4 ulp |      4.02 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_f32_v128relaxed`       |        8.90 gso/s, 0 ulp |        10.1 gso/s, 0 ulp |        10.8 gso/s, 0 ulp |
| `nk_angulars_symmetric_f32_v128relaxed`    |        6.19 gso/s, 0 ulp |        7.56 gso/s, 0 ulp |        8.32 gso/s, 0 ulp |
| `nk_euclideans_packed_f32_v128relaxed`     |      8.49 gso/s, 0.4 ulp |      10.1 gso/s, 0.4 ulp |      10.8 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_f32_v128relaxed`  |      6.33 gso/s, 0.4 ulp |      7.63 gso/s, 0.4 ulp |      8.27 gso/s, 0.3 ulp |
| __bf16__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_bf16_serial`           |      1.95 gso/s, 0.1 ulp |        2.00 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_bf16_serial`        |      1.94 gso/s, 0.1 ulp |        2.06 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_bf16_serial`         |      1.96 gso/s, 0.4 ulp |      2.01 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_bf16_serial`      |      1.96 gso/s, 0.5 ulp |      2.05 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_bf16_v128relaxed`      |        28.0 gso/s, 0 ulp |        30.8 gso/s, 0 ulp |        33.8 gso/s, 0 ulp |
| `nk_angulars_symmetric_bf16_v128relaxed`   |        21.2 gso/s, 0 ulp |        28.2 gso/s, 0 ulp |        19.9 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_v128relaxed`    |      28.6 gso/s, 0.2 ulp |      30.9 gso/s, 0.2 ulp |      33.8 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_bf16_v128relaxed` |      21.9 gso/s, 0.2 ulp |      28.0 gso/s, 0.2 ulp |      29.6 gso/s, 0.2 ulp |
| `nk_angulars_packed_bf16_v128`             |        24.3 gso/s, 0 ulp |        25.2 gso/s, 0 ulp |        27.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_bf16_v128`          |        16.8 gso/s, 0 ulp |        24.4 gso/s, 0 ulp |        25.7 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_v128`           |      22.0 gso/s, 0.2 ulp |      25.9 gso/s, 0.2 ulp |      19.6 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_bf16_v128`        |      19.6 gso/s, 0.2 ulp |      24.6 gso/s, 0.2 ulp |      25.3 gso/s, 0.2 ulp |
| __f16__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f16_serial`            |      2.77 gso/s, 0.1 ulp |      1.69 gso/s, 0.1 ulp |                        ⋯ |
| `nk_angulars_symmetric_f16_serial`         |     0.565 gso/s, 0.1 ulp |     0.520 gso/s, 0.1 ulp |                        ⋯ |
| `nk_euclideans_packed_f16_serial`          |      2.75 gso/s, 0.5 ulp |      1.98 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_f16_serial`       |     0.569 gso/s, 0.4 ulp |     0.439 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_f16_v128relaxed`       |      9.93 gso/s, 0.1 ulp |      11.8 gso/s, 0.1 ulp |      12.3 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_v128relaxed`    |      5.77 gso/s, 0.1 ulp |      6.92 gso/s, 0.1 ulp |      7.21 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_v128relaxed`     |      10.1 gso/s, 0.2 ulp |      11.9 gso/s, 0.2 ulp |      12.3 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f16_v128relaxed`  |      4.55 gso/s, 0.2 ulp |      7.24 gso/s, 0.2 ulp |      7.15 gso/s, 0.2 ulp |
| __e5m2__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e5m2_serial`           |        1.09 gso/s, 0 ulp |       0.810 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_e5m2_serial`        |        1.08 gso/s, 0 ulp |        1.13 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_e5m2_serial`         |      1.10 gso/s, 0.4 ulp |      1.10 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_e5m2_serial`      |      1.09 gso/s, 0.5 ulp |      1.14 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_e5m2_v128relaxed`      |        7.95 gso/s, 0 ulp |        9.29 gso/s, 0 ulp |        8.57 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_v128relaxed`   |        4.36 gso/s, 0 ulp |        5.19 gso/s, 0 ulp |        5.23 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_v128relaxed`    |      7.50 gso/s, 0.2 ulp |      9.22 gso/s, 0.2 ulp |      9.54 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e5m2_v128relaxed` |      3.56 gso/s, 0.2 ulp |      5.19 gso/s, 0.2 ulp |      5.24 gso/s, 0.2 ulp |
| __e4m3__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e4m3_serial`           |     0.279 gso/s, 0.1 ulp |       0.242 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_e4m3_serial`        |       0.270 gso/s, 0 ulp |       0.222 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_e4m3_serial`         |     0.277 gso/s, 0.5 ulp |     0.223 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_e4m3_serial`      |     0.276 gso/s, 0.5 ulp |     0.222 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_e4m3_v128relaxed`      |        8.98 gso/s, 0 ulp |        10.8 gso/s, 0 ulp |        8.53 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_v128relaxed`   |        5.01 gso/s, 0 ulp |        6.13 gso/s, 0 ulp |        5.57 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_v128relaxed`    |      9.11 gso/s, 0.2 ulp |      10.9 gso/s, 0.2 ulp |      11.1 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e4m3_v128relaxed` |      4.10 gso/s, 0.2 ulp |      6.14 gso/s, 0.2 ulp |      6.24 gso/s, 0.2 ulp |
| __e3m2__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e3m2_serial`           |        1.09 gso/s, 0 ulp |       0.774 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_e3m2_serial`        |      1.09 gso/s, 0.1 ulp |        1.11 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_e3m2_serial`         |      1.10 gso/s, 0.4 ulp |      1.09 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_e3m2_serial`      |      1.09 gso/s, 0.4 ulp |      1.14 gso/s, 0.4 ulp |                        ⋯ |
| `nk_angulars_packed_e3m2_v128relaxed`      |    14.5 gso/s, 4.01K ulp |    17.3 gso/s, 1.97K ulp |       15.9 gso/s, 1K ulp |
| `nk_angulars_symmetric_e3m2_v128relaxed`   |    11.9 gso/s, 4.01K ulp |    16.7 gso/s, 1.98K ulp |    16.1 gso/s, 989.3 ulp |
| `nk_euclideans_packed_e3m2_v128relaxed`    |    14.7 gso/s, 1.77K ulp |    17.4 gso/s, 757.8 ulp |    18.0 gso/s, 343.4 ulp |
| `nk_euclideans_symmetric_e3m2_v128relaxed` |    9.09 gso/s, 1.78K ulp |    16.9 gso/s, 746.1 ulp |      17.4 gso/s, 340 ulp |
| __e2m3__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_serial`           |        1.08 gso/s, 0 ulp |        1.11 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_e2m3_serial`        |        1.09 gso/s, 0 ulp |        1.13 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_e2m3_serial`         |     0.800 gso/s, 0.6 ulp |      1.12 gso/s, 0.5 ulp |                        ⋯ |
| `nk_euclideans_symmetric_e2m3_serial`      |      1.10 gso/s, 0.6 ulp |      1.14 gso/s, 0.5 ulp |                        ⋯ |
| `nk_angulars_packed_e2m3_v128relaxed`      |        15.8 gso/s, 0 ulp |        17.4 gso/s, 0 ulp |        15.4 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_v128relaxed`   |        13.4 gso/s, 0 ulp |        16.8 gso/s, 0 ulp |        17.7 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_v128relaxed`    |        16.1 gso/s, 0 ulp |        17.6 gso/s, 0 ulp |        18.2 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_v128relaxed` |        13.4 gso/s, 0 ulp |        16.9 gso/s, 0 ulp |        17.7 gso/s, 0 ulp |
| __i8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_serial`             |        4.94 gso/s, 0 ulp |        5.40 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_i8_serial`          |        3.39 gso/s, 0 ulp |        3.62 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_i8_serial`           |        3.94 gso/s, 0 ulp |        5.48 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_symmetric_i8_serial`        |        3.49 gso/s, 0 ulp |        3.65 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_packed_i8_v128relaxed`        |      28.2 gso/s, 0.2 ulp |      33.5 gso/s, 0.2 ulp |      25.3 gso/s, 0.4 ulp |
| `nk_angulars_symmetric_i8_v128relaxed`     |      25.0 gso/s, 0.2 ulp |      31.8 gso/s, 0.2 ulp |      34.1 gso/s, 0.4 ulp |
| `nk_euclideans_packed_i8_v128relaxed`      |        30.4 gso/s, 0 ulp |        34.4 gso/s, 0 ulp |      37.4 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_i8_v128relaxed`   |        26.5 gso/s, 0 ulp |        32.8 gso/s, 0 ulp |      34.3 gso/s, 0.2 ulp |
| `nk_angulars_packed_i8_v128`               |      34.2 gso/s, 0.2 ulp |      43.7 gso/s, 0.2 ulp |      48.8 gso/s, 0.4 ulp |
| `nk_angulars_symmetric_i8_v128`            |      30.5 gso/s, 0.2 ulp |      45.4 gso/s, 0.2 ulp |      48.8 gso/s, 0.4 ulp |
| `nk_euclideans_packed_i8_v128`             |        42.9 gso/s, 0 ulp |        44.9 gso/s, 0 ulp |      35.2 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_i8_v128`          |        37.1 gso/s, 0 ulp |        47.1 gso/s, 0 ulp |      48.6 gso/s, 0.2 ulp |
| __u8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`             |        5.06 gso/s, 0 ulp |        5.42 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_u8_serial`          |        3.28 gso/s, 0 ulp |        3.64 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_u8_serial`           |        3.70 gso/s, 0 ulp |        5.38 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_symmetric_u8_serial`        |        3.36 gso/s, 0 ulp |        3.63 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_packed_u8_v128relaxed`        |      28.1 gso/s, 0.4 ulp |      33.0 gso/s, 0.5 ulp |      36.3 gso/s, 0.5 ulp |
| `nk_angulars_symmetric_u8_v128relaxed`     |      21.6 gso/s, 0.4 ulp |      26.8 gso/s, 0.6 ulp |      27.9 gso/s, 0.5 ulp |
| `nk_euclideans_packed_u8_v128relaxed`      |        29.8 gso/s, 0 ulp |        33.4 gso/s, 0 ulp |      36.3 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_u8_v128relaxed`   |        22.6 gso/s, 0 ulp |        27.0 gso/s, 0 ulp |      27.7 gso/s, 0.2 ulp |
| `nk_angulars_packed_u8_v128`               |      38.0 gso/s, 0.4 ulp |      47.9 gso/s, 0.5 ulp |      54.6 gso/s, 0.5 ulp |
| `nk_angulars_symmetric_u8_v128`            |      31.8 gso/s, 0.4 ulp |      48.5 gso/s, 0.6 ulp |      51.5 gso/s, 0.5 ulp |
| `nk_euclideans_packed_u8_v128`             |        46.0 gso/s, 0 ulp |        50.3 gso/s, 0 ulp |      39.2 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_u8_v128`          |        38.0 gso/s, 0 ulp |        49.1 gso/s, 0 ulp |      51.3 gso/s, 0.2 ulp |
| __i4__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i4_serial`             |        2.41 gso/s, 0 ulp |        2.57 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_i4_serial`          |        1.91 gso/s, 0 ulp |        2.58 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_i4_serial`           |        1.89 gso/s, 0 ulp |        2.63 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_symmetric_i4_serial`        |        2.50 gso/s, 0 ulp |        2.62 gso/s, 0 ulp |                        ⋯ |
| __u4__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u4_serial`             |        3.10 gso/s, 0 ulp |        3.25 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_u4_serial`          |        2.25 gso/s, 0 ulp |        3.34 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_u4_serial`           |        2.22 gso/s, 0 ulp |        3.27 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_symmetric_u4_serial`        |        3.19 gso/s, 0 ulp |        3.36 gso/s, 0 ulp |                        ⋯ |
| __e2m1__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m1_serial`           |        2.16 gso/s, 0 ulp |        2.29 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_e2m1_serial`        |        2.25 gso/s, 0 ulp |        2.31 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_e2m1_serial`         |      2.18 gso/s, 0.3 ulp |      2.28 gso/s, 0.3 ulp |                        ⋯ |
| `nk_euclideans_symmetric_e2m1_serial`      |      2.17 gso/s, 0.4 ulp |      1.68 gso/s, 0.3 ulp |                        ⋯ |
| `nk_angulars_packed_e2m1_v128relaxed`      |        29.7 gso/s, 0 ulp |        40.6 gso/s, 0 ulp |        42.9 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m1_v128relaxed`   |        27.1 gso/s, 0 ulp |        40.5 gso/s, 0 ulp |        40.2 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m1_v128relaxed`    |        25.3 gso/s, 0 ulp |        40.9 gso/s, 0 ulp |        38.6 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m1_v128relaxed` |        18.4 gso/s, 0 ulp |        41.2 gso/s, 0 ulp |        43.6 gso/s, 0 ulp |
| __nvfp4__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_nvfp4_serial`          |       0.290 gso/s, 0 ulp |       0.294 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_nvfp4_serial`       |       0.284 gso/s, 0 ulp |       0.288 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_nvfp4_serial`        |     0.286 gso/s, 0.3 ulp |     0.281 gso/s, 0.3 ulp |                        ⋯ |
| `nk_euclideans_symmetric_nvfp4_serial`     |     0.268 gso/s, 0.4 ulp |     0.272 gso/s, 0.3 ulp |                        ⋯ |
| __mxfp4__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp4_serial`          |       0.296 gso/s, 0 ulp |       0.207 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_mxfp4_serial`       |       0.300 gso/s, 0 ulp |       0.303 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_mxfp4_serial`        |     0.284 gso/s, 0.3 ulp |     0.288 gso/s, 0.3 ulp |                        ⋯ |
| `nk_euclideans_symmetric_mxfp4_serial`     |     0.287 gso/s, 0.4 ulp |     0.294 gso/s, 0.3 ulp |                        ⋯ |
| __mxfp8e4m3__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp8e4m3_serial`      |       0.159 gso/s, 0 ulp |       0.123 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_mxfp8e4m3_serial`   |       0.158 gso/s, 0 ulp |       0.159 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_mxfp8e4m3_serial`    |     0.163 gso/s, 0.5 ulp |     0.164 gso/s, 0.5 ulp |                        ⋯ |
| `nk_euclideans_symmetric_mxfp8e4m3_serial` |     0.150 gso/s, 0.5 ulp |     0.160 gso/s, 0.5 ulp |                        ⋯ |
| __mxfp8e5m2__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp8e5m2_serial`      |       0.330 gso/s, 0 ulp |       0.338 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_mxfp8e5m2_serial`   |       0.332 gso/s, 0 ulp |       0.335 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_mxfp8e5m2_serial`    |     0.316 gso/s, 0.4 ulp |     0.327 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_mxfp8e5m2_serial` |     0.275 gso/s, 0.5 ulp |     0.279 gso/s, 0.5 ulp |                        ⋯ |
| __mxfp6e2m3__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp6e2m3_serial`      |       0.335 gso/s, 0 ulp |       0.201 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_mxfp6e2m3_serial`   |       0.336 gso/s, 0 ulp |       0.201 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_mxfp6e2m3_serial`    |     0.323 gso/s, 0.6 ulp |     0.193 gso/s, 0.5 ulp |                        ⋯ |
| `nk_euclideans_symmetric_mxfp6e2m3_serial` |     0.314 gso/s, 0.6 ulp |     0.194 gso/s, 0.5 ulp |                        ⋯ |
| __mxfp6e3m2__                              | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp6e3m2_serial`      |       0.334 gso/s, 0 ulp |       0.201 gso/s, 0 ulp |                        ⋯ |
| `nk_angulars_symmetric_mxfp6e3m2_serial`   |       0.335 gso/s, 0 ulp |       0.201 gso/s, 0 ulp |                        ⋯ |
| `nk_euclideans_packed_mxfp6e3m2_serial`    |     0.320 gso/s, 0.4 ulp |     0.194 gso/s, 0.4 ulp |                        ⋯ |
| `nk_euclideans_symmetric_mxfp6e3m2_serial` |     0.310 gso/s, 0.4 ulp |     0.194 gso/s, 0.4 ulp |                        ⋯ |

#### CUDA

Rows ran on one `1g.34gb` MIG slice of a B300 with 18 SMs.

| Kernel                                        |                 256³ |                 1024³ |                  4096³ |
| :-------------------------------------------- | -------------------: | --------------------: | ---------------------: |
| __f64__                                       | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f64_cuda`                 |    10.7 gso/s, 0 ulp |     11.6 gso/s, 0 ulp |      12.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_cuda`              |    5.28 gso/s, 0 ulp |     10.7 gso/s, 0 ulp |      11.6 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_cuda`               |    10.7 gso/s, 0 ulp |     11.6 gso/s, 0 ulp |      12.2 gso/s, 0 ulp |
| `nk_euclideans_symmetric_f64_cuda`            |    5.30 gso/s, 0 ulp |     10.7 gso/s, 0 ulp |      11.6 gso/s, 0 ulp |
| __f32__                                       | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f32_cuda`                 |    92.1 gso/s, 0 ulp |      110 gso/s, 0 ulp |       117 gso/s, 0 ulp |
| `nk_angulars_symmetric_f32_cuda`              |    40.2 gso/s, 0 ulp |     89.2 gso/s, 0 ulp |      98.7 gso/s, 0 ulp |
| `nk_euclideans_packed_f32_cuda`               |    94.2 gso/s, 0 ulp |      110 gso/s, 0 ulp |       117 gso/s, 0 ulp |
| `nk_euclideans_symmetric_f32_cuda`            |    41.0 gso/s, 0 ulp |     89.8 gso/s, 0 ulp |      98.8 gso/s, 0 ulp |
| __bf16__                                      | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_bf16_ampere`              | 1,639 gso/s, 0.1 ulp | 25,150 gso/s, 0.2 ulp |  38,830 gso/s, 0.6 ulp |
| `nk_angulars_symmetric_bf16_ampere`           |   737 gso/s, 0.1 ulp | 11,210 gso/s, 0.2 ulp |  30,000 gso/s, 0.6 ulp |
| `nk_euclideans_packed_bf16_ampere`            | 1,834 gso/s, 0.2 ulp | 26,040 gso/s, 0.3 ulp |  38,460 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_bf16_ampere`         |   817 gso/s, 0.3 ulp | 11,400 gso/s, 0.3 ulp |  30,500 gso/s, 0.5 ulp |
| `nk_angulars_packed_bf16_blackwell`           | 2,102 gso/s, 0.1 ulp | 37,330 gso/s, 0.2 ulp | 114,800 gso/s, 0.6 ulp |
| `nk_angulars_symmetric_bf16_blackwell`        |   830 gso/s, 0.1 ulp | 18,160 gso/s, 0.2 ulp |  77,420 gso/s, 0.6 ulp |
| `nk_euclideans_packed_bf16_blackwell`         | 2,332 gso/s, 0.2 ulp | 41,350 gso/s, 0.3 ulp | 115,400 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_bf16_blackwell`      |   964 gso/s, 0.3 ulp | 21,070 gso/s, 0.3 ulp |  75,990 gso/s, 0.4 ulp |
| __f16__                                       | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f16_ampere`               | 1,634 gso/s, 0.2 ulp | 24,640 gso/s, 0.4 ulp |  37,760 gso/s, 0.8 ulp |
| `nk_angulars_symmetric_f16_ampere`            |   734 gso/s, 0.2 ulp | 11,040 gso/s, 0.4 ulp |  29,370 gso/s, 0.7 ulp |
| `nk_euclideans_packed_f16_ampere`             | 1,831 gso/s, 0.3 ulp | 25,540 gso/s, 0.3 ulp |  37,550 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_f16_ampere`          |   813 gso/s, 0.4 ulp | 10,900 gso/s, 0.4 ulp |  29,720 gso/s, 0.6 ulp |
| `nk_angulars_packed_f16_blackwell`            | 2,093 gso/s, 0.2 ulp | 37,180 gso/s, 0.4 ulp | 104,000 gso/s, 0.8 ulp |
| `nk_angulars_symmetric_f16_blackwell`         |   827 gso/s, 0.2 ulp | 18,150 gso/s, 0.4 ulp |  69,930 gso/s, 0.7 ulp |
| `nk_euclideans_packed_f16_blackwell`          | 2,329 gso/s, 0.4 ulp | 40,890 gso/s, 0.3 ulp | 102,800 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_f16_blackwell`       |   963 gso/s, 0.6 ulp | 21,090 gso/s, 0.5 ulp |  70,170 gso/s, 0.6 ulp |
| __e5m2__                                      | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e5m2_ampere`              |   1,730 gso/s, 0 ulp |   26,430 gso/s, 0 ulp |    42,710 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_ampere`           |     768 gso/s, 0 ulp |   11,920 gso/s, 0 ulp |    33,010 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_ampere`            | 1,955 gso/s, 0.2 ulp | 27,660 gso/s, 0.2 ulp |  42,850 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_e5m2_ampere`         |   856 gso/s, 0.2 ulp | 12,550 gso/s, 0.3 ulp |  33,740 gso/s, 0.3 ulp |
| `nk_angulars_packed_e5m2_blackwell`           |   2,157 gso/s, 0 ulp |   39,150 gso/s, 0 ulp |   135,300 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_blackwell`        |     839 gso/s, 0 ulp |   18,790 gso/s, 0 ulp |    87,040 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_blackwell`         | 2,417 gso/s, 0.2 ulp | 44,040 gso/s, 0.2 ulp | 135,200 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e5m2_blackwell`      |   984 gso/s, 0.2 ulp | 21,820 gso/s, 0.2 ulp |  86,320 gso/s, 0.3 ulp |
| __e4m3__                                      | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e4m3_ampere`              |   1,461 gso/s, 0 ulp |   13,830 gso/s, 0 ulp |    19,170 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_ampere`           |     619 gso/s, 0 ulp |    5,935 gso/s, 0 ulp |    13,790 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_ampere`            | 1,617 gso/s, 0.2 ulp | 13,800 gso/s, 0.2 ulp |  19,470 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_e4m3_ampere`         |   679 gso/s, 0.2 ulp |  6,120 gso/s, 0.2 ulp |  13,950 gso/s, 0.3 ulp |
| `nk_angulars_packed_e4m3_blackwell`           |   2,076 gso/s, 0 ulp |   34,310 gso/s, 0 ulp |    78,780 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_blackwell`        |     778 gso/s, 0 ulp |   14,500 gso/s, 0 ulp |    33,260 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_blackwell`         | 2,298 gso/s, 0.2 ulp | 39,290 gso/s, 0.2 ulp |  79,750 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e4m3_blackwell`      |   897 gso/s, 0.2 ulp | 16,260 gso/s, 0.2 ulp |  33,860 gso/s, 0.3 ulp |
| __e3m2__                                      | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e3m2_ampere`              |   1,640 gso/s, 0 ulp |   21,110 gso/s, 0 ulp |    31,130 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_ampere`           |     723 gso/s, 0 ulp |    9,261 gso/s, 0 ulp |    24,040 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_ampere`            |   1,843 gso/s, 0 ulp | 22,060 gso/s, 0.1 ulp |  31,320 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_e3m2_ampere`         |     805 gso/s, 0 ulp |  9,693 gso/s, 0.2 ulp |  24,310 gso/s, 0.5 ulp |
| `nk_angulars_packed_e3m2_blackwell`           |   2,115 gso/s, 0 ulp |   36,110 gso/s, 0 ulp |    93,110 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_blackwell`        |     798 gso/s, 0 ulp |   15,840 gso/s, 0 ulp |    46,180 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_blackwell`         |   2,351 gso/s, 0 ulp | 40,490 gso/s, 0.1 ulp |  93,550 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e3m2_blackwell`      |     915 gso/s, 0 ulp | 17,940 gso/s, 0.2 ulp |  44,800 gso/s, 0.3 ulp |
| __e2m3__                                      | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_ampere`              |   1,290 gso/s, 0 ulp |   11,600 gso/s, 0 ulp |    14,460 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_ampere`           |     566 gso/s, 0 ulp |    5,241 gso/s, 0 ulp |    11,450 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_ampere`            |   1,352 gso/s, 0 ulp |   11,580 gso/s, 0 ulp |    14,290 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_ampere`         |     587 gso/s, 0 ulp |    5,279 gso/s, 0 ulp |    11,360 gso/s, 0 ulp |
| `nk_angulars_packed_e2m3_blackwell`           |   2,121 gso/s, 0 ulp |   35,350 gso/s, 0 ulp |    96,740 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_blackwell`        |     799 gso/s, 0 ulp |   15,890 gso/s, 0 ulp |    48,090 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_blackwell`         |   2,363 gso/s, 0 ulp |   40,150 gso/s, 0 ulp |    96,770 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_blackwell`      |     922 gso/s, 0 ulp |   18,060 gso/s, 0 ulp |    47,630 gso/s, 0 ulp |
| __e2m1__                                      | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m1_ampere`              |   1,358 gso/s, 0 ulp |   12,050 gso/s, 0 ulp |    15,010 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m1_ampere`           |     616 gso/s, 0 ulp |    6,059 gso/s, 0 ulp |    13,370 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m1_ampere`            |   1,423 gso/s, 0 ulp |   12,000 gso/s, 0 ulp |    15,070 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m1_ampere`         |     620 gso/s, 0 ulp |    6,131 gso/s, 0 ulp |    13,400 gso/s, 0 ulp |
| `nk_angulars_packed_e2m1_blackwell`           |   2,266 gso/s, 0 ulp |   41,430 gso/s, 0 ulp |   180,200 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m1_blackwell`        |     880 gso/s, 0 ulp |   20,150 gso/s, 0 ulp |   114,000 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m1_blackwell`         |   2,528 gso/s, 0 ulp |   47,280 gso/s, 0 ulp |   208,200 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m1_blackwell`      |   1,037 gso/s, 0 ulp |   24,120 gso/s, 0 ulp |   139,300 gso/s, 0 ulp |
| __i8__                                        | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_ampere`                |     183 gso/s, 0 ulp |    3,208 gso/s, 0 ulp |    12,440 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_ampere`             |    95.2 gso/s, 0 ulp |    1,594 gso/s, 0 ulp |    10,280 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_ampere`              |     488 gso/s, 0 ulp |    7,633 gso/s, 0 ulp |    14,320 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_ampere`           |     239 gso/s, 0 ulp |    3,752 gso/s, 0 ulp |    12,260 gso/s, 0 ulp |
| `nk_angulars_packed_i8_blackwell`             |     200 gso/s, 0 ulp |    3,220 gso/s, 0 ulp |    13,230 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_blackwell`          |    97.9 gso/s, 0 ulp |    2,098 gso/s, 0 ulp |    11,170 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_blackwell`           |     559 gso/s, 0 ulp |    9,152 gso/s, 0 ulp |    32,750 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_blackwell`        |     269 gso/s, 0 ulp |    5,768 gso/s, 0 ulp |    25,860 gso/s, 0 ulp |
| __i4__                                        | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i4_ampere`                |     182 gso/s, 0 ulp |    3,131 gso/s, 0 ulp |    11,790 gso/s, 0 ulp |
| `nk_angulars_symmetric_i4_ampere`             |    92.5 gso/s, 0 ulp |    1,565 gso/s, 0 ulp |     9,543 gso/s, 0 ulp |
| `nk_euclideans_packed_i4_ampere`              |     488 gso/s, 0 ulp |    7,751 gso/s, 0 ulp |    13,340 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i4_ampere`           |     238 gso/s, 0 ulp |    3,655 gso/s, 0 ulp |    12,370 gso/s, 0 ulp |
| `nk_angulars_packed_i4_blackwell`             |     195 gso/s, 0 ulp |    3,159 gso/s, 0 ulp |    13,980 gso/s, 0 ulp |
| `nk_angulars_symmetric_i4_blackwell`          |    95.8 gso/s, 0 ulp |    2,067 gso/s, 0 ulp |    12,260 gso/s, 0 ulp |
| `nk_euclideans_packed_i4_blackwell`           |     554 gso/s, 0 ulp |    8,984 gso/s, 0 ulp |    38,300 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i4_blackwell`        |     267 gso/s, 0 ulp |    5,736 gso/s, 0 ulp |    32,460 gso/s, 0 ulp |
| __u8__                                        | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_ampere`                |     275 gso/s, 0 ulp |    4,579 gso/s, 0 ulp |    12,730 gso/s, 0 ulp |
| `nk_angulars_symmetric_u8_ampere`             |     136 gso/s, 0 ulp |    2,260 gso/s, 0 ulp |    10,920 gso/s, 0 ulp |
| `nk_euclideans_packed_u8_ampere`              |     488 gso/s, 0 ulp |    7,626 gso/s, 0 ulp |    14,230 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u8_ampere`           |     239 gso/s, 0 ulp |    3,751 gso/s, 0 ulp |    12,370 gso/s, 0 ulp |
| `nk_angulars_packed_u8_blackwell`             |     294 gso/s, 0 ulp |    4,709 gso/s, 0 ulp |    18,620 gso/s, 0 ulp |
| `nk_angulars_symmetric_u8_blackwell`          |     142 gso/s, 0 ulp |    3,048 gso/s, 0 ulp |    15,520 gso/s, 0 ulp |
| `nk_euclideans_packed_u8_blackwell`           |     556 gso/s, 0 ulp |    9,054 gso/s, 0 ulp |    32,520 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u8_blackwell`        |     264 gso/s, 0 ulp |    5,688 gso/s, 0 ulp |    25,760 gso/s, 0 ulp |
| __u4__                                        | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u4_ampere`                |     275 gso/s, 0 ulp |    4,579 gso/s, 0 ulp |    12,420 gso/s, 0 ulp |
| `nk_angulars_symmetric_u4_ampere`             |     136 gso/s, 0 ulp |    2,284 gso/s, 0 ulp |    10,670 gso/s, 0 ulp |
| `nk_euclideans_packed_u4_ampere`              |     487 gso/s, 0 ulp |    7,683 gso/s, 0 ulp |    14,220 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u4_ampere`           |     238 gso/s, 0 ulp |    3,807 gso/s, 0 ulp |    12,370 gso/s, 0 ulp |
| `nk_angulars_packed_u4_blackwell`             |     295 gso/s, 0 ulp |    4,711 gso/s, 0 ulp |    20,550 gso/s, 0 ulp |
| `nk_angulars_symmetric_u4_blackwell`          |     144 gso/s, 0 ulp |    3,074 gso/s, 0 ulp |    18,020 gso/s, 0 ulp |
| `nk_euclideans_packed_u4_blackwell`           |     555 gso/s, 0 ulp |    9,019 gso/s, 0 ulp |    38,730 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u4_blackwell`        |     270 gso/s, 0 ulp |    5,784 gso/s, 0 ulp |    33,310 gso/s, 0 ulp |
| __nvfp4__                                     | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_nvfp4_blackwell`          |   1,564 gso/s, 0 ulp |   14,560 gso/s, 0 ulp |    17,050 gso/s, 0 ulp |
| `nk_angulars_symmetric_nvfp4_blackwell`       |     512 gso/s, 0 ulp |    5,220 gso/s, 0 ulp |     7,695 gso/s, 0 ulp |
| `nk_euclideans_packed_nvfp4_blackwell`        |   1,691 gso/s, 0 ulp |   14,830 gso/s, 0 ulp |    17,160 gso/s, 0 ulp |
| `nk_euclideans_symmetric_nvfp4_blackwell`     |     568 gso/s, 0 ulp |    5,349 gso/s, 0 ulp |     7,816 gso/s, 0 ulp |
| __mxfp4__                                     | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp4_blackwell`          |   1,605 gso/s, 0 ulp |   16,850 gso/s, 0 ulp |    20,840 gso/s, 0 ulp |
| `nk_angulars_symmetric_mxfp4_blackwell`       |     552 gso/s, 0 ulp |    6,200 gso/s, 0 ulp |     9,496 gso/s, 0 ulp |
| `nk_euclideans_packed_mxfp4_blackwell`        |   1,728 gso/s, 0 ulp |   17,180 gso/s, 0 ulp |    20,850 gso/s, 0 ulp |
| `nk_euclideans_symmetric_mxfp4_blackwell`     |     606 gso/s, 0 ulp |    6,248 gso/s, 0 ulp |     9,393 gso/s, 0 ulp |
| __mxfp8e4m3__                                 | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp8e4m3_blackwell`      |   1,744 gso/s, 0 ulp |   21,840 gso/s, 0 ulp |    28,220 gso/s, 0 ulp |
| `nk_angulars_symmetric_mxfp8e4m3_blackwell`   |     619 gso/s, 0 ulp |    8,132 gso/s, 0 ulp |    12,500 gso/s, 0 ulp |
| `nk_euclideans_packed_mxfp8e4m3_blackwell`    | 1,879 gso/s, 0.2 ulp | 22,270 gso/s, 0.3 ulp |  28,260 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_mxfp8e4m3_blackwell` |   686 gso/s, 0.2 ulp |  8,234 gso/s, 0.2 ulp |  12,400 gso/s, 0.3 ulp |
| __mxfp8e5m2__                                 | ░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_mxfp8e5m2_blackwell`      |   1,781 gso/s, 0 ulp |   23,600 gso/s, 0 ulp |    30,680 gso/s, 0 ulp |
| `nk_angulars_symmetric_mxfp8e5m2_blackwell`   |     642 gso/s, 0 ulp |    8,899 gso/s, 0 ulp |    13,750 gso/s, 0 ulp |
| `nk_euclideans_packed_mxfp8e5m2_blackwell`    | 1,937 gso/s, 0.2 ulp | 24,090 gso/s, 0.3 ulp |  30,800 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_mxfp8e5m2_blackwell` |   718 gso/s, 0.2 ulp |  9,214 gso/s, 0.2 ulp |  13,880 gso/s, 0.3 ulp |

### Intel Granite Rapids with RTX PRO 6000 Blackwell

#### Native

| Kernel                                      |                   256³ |                  1024³ |                  4096³ |
| :------------------------------------------ | ---------------------: | ---------------------: | ---------------------: |
| __f64__                                     | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f64_serial`             |            0.565 gso/s |            0.571 gso/s |            0.563 gso/s |
| `nk_angulars_symmetric_f64_serial`          |            0.554 gso/s |            0.544 gso/s |            0.531 gso/s |
| `nk_euclideans_packed_f64_serial`           |            0.618 gso/s |            0.573 gso/s |            0.577 gso/s |
| `nk_euclideans_symmetric_f64_serial`        |            0.569 gso/s |            0.547 gso/s |            0.544 gso/s |
| `nk_angulars_packed_f64_haswell`            |             6.42 gso/s |             6.88 gso/s |             7.09 gso/s |
| `nk_angulars_symmetric_f64_haswell`         |             6.26 gso/s |             6.78 gso/s |             6.66 gso/s |
| `nk_euclideans_packed_f64_haswell`          |             6.30 gso/s |             6.87 gso/s |             7.17 gso/s |
| `nk_euclideans_symmetric_f64_haswell`       |             6.63 gso/s |             6.80 gso/s |             6.49 gso/s |
| `nk_angulars_packed_f64_skylake`            |             8.52 gso/s |             10.0 gso/s |             11.2 gso/s |
| `nk_angulars_symmetric_f64_skylake`         |             8.15 gso/s |             9.63 gso/s |             9.49 gso/s |
| `nk_euclideans_packed_f64_skylake`          |             8.97 gso/s |             10.3 gso/s |             11.0 gso/s |
| `nk_euclideans_symmetric_f64_skylake`       |             8.36 gso/s |             9.64 gso/s |             9.50 gso/s |
| `nk_angulars_packed_f64_cuda`               |      12.5 gso/s, 0 ulp |       101 gso/s, 0 ulp |       147 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_cuda`            |      6.12 gso/s, 0 ulp |      51.4 gso/s, 0 ulp |       127 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_cuda`             |    12.5 gso/s, 0.1 ulp |     101 gso/s, 0.1 ulp |     147 gso/s, 0.1 ulp |
| `nk_euclideans_symmetric_f64_cuda`          |      6.13 gso/s, 0 ulp |      49.7 gso/s, 0 ulp |       126 gso/s, 0 ulp |
| __f32__                                     | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f32_serial`             |             3.44 gso/s |             3.53 gso/s |             3.55 gso/s |
| `nk_angulars_symmetric_f32_serial`          |             4.50 gso/s |             4.78 gso/s |             4.79 gso/s |
| `nk_euclideans_packed_f32_serial`           |             3.44 gso/s |             3.50 gso/s |             3.49 gso/s |
| `nk_euclideans_symmetric_f32_serial`        |             4.51 gso/s |             4.76 gso/s |             4.87 gso/s |
| `nk_angulars_packed_f32_haswell`            |             27.6 gso/s |             31.6 gso/s |             33.7 gso/s |
| `nk_angulars_symmetric_f32_haswell`         |             16.9 gso/s |             20.9 gso/s |             21.8 gso/s |
| `nk_euclideans_packed_f32_haswell`          |             30.0 gso/s |             31.5 gso/s |             34.2 gso/s |
| `nk_euclideans_symmetric_f32_haswell`       |             17.4 gso/s |             20.6 gso/s |             21.8 gso/s |
| `nk_angulars_packed_f32_skylake`            |             34.6 gso/s |             46.2 gso/s |             49.0 gso/s |
| `nk_angulars_symmetric_f32_skylake`         |             21.2 gso/s |             27.2 gso/s |             29.3 gso/s |
| `nk_euclideans_packed_f32_skylake`          |             35.6 gso/s |             46.6 gso/s |             50.1 gso/s |
| `nk_euclideans_symmetric_f32_skylake`       |             23.1 gso/s |             28.7 gso/s |             28.9 gso/s |
| `nk_angulars_packed_f32_cuda`               |       121 gso/s, 0 ulp |     1,091 gso/s, 0 ulp |   1,620 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f32_cuda`            |      59.5 gso/s, 0 ulp |       541 gso/s, 0 ulp |   1,361 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f32_cuda`             |     123 gso/s, 0.1 ulp |   1,095 gso/s, 0.1 ulp |   1,621 gso/s, 1.3 ulp |
| `nk_euclideans_symmetric_f32_cuda`          |    60.0 gso/s, 0.1 ulp |     543 gso/s, 0.1 ulp |   1,359 gso/s, 1.9 ulp |
| __bf16__                                    | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_bf16_serial`            |            0.946 gso/s |            0.956 gso/s |            0.925 gso/s |
| `nk_angulars_symmetric_bf16_serial`         |            0.875 gso/s |            0.878 gso/s |            0.850 gso/s |
| `nk_euclideans_packed_bf16_serial`          |             1.00 gso/s |            0.958 gso/s |            0.955 gso/s |
| `nk_euclideans_symmetric_bf16_serial`       |            0.871 gso/s |            0.876 gso/s |            0.883 gso/s |
| `nk_angulars_packed_bf16_haswell`           |             53.1 gso/s |             63.9 gso/s |             68.1 gso/s |
| `nk_angulars_symmetric_bf16_haswell`        |             37.9 gso/s |             55.6 gso/s |             61.2 gso/s |
| `nk_euclideans_packed_bf16_haswell`         |             58.2 gso/s |             62.5 gso/s |             67.6 gso/s |
| `nk_euclideans_symmetric_bf16_haswell`      |             38.6 gso/s |             54.5 gso/s |             66.8 gso/s |
| `nk_angulars_packed_bf16_skylake`           |             75.5 gso/s |              104 gso/s |              105 gso/s |
| `nk_angulars_symmetric_bf16_skylake`        |             48.4 gso/s |             74.2 gso/s |             88.9 gso/s |
| `nk_euclideans_packed_bf16_skylake`         |             76.5 gso/s |              103 gso/s |              107 gso/s |
| `nk_euclideans_symmetric_bf16_skylake`      |             43.4 gso/s |             75.8 gso/s |             89.1 gso/s |
| `nk_angulars_packed_bf16_genoa`             |             66.8 gso/s |             87.3 gso/s |             92.7 gso/s |
| `nk_angulars_symmetric_bf16_genoa`          |             51.8 gso/s |             69.6 gso/s |             76.0 gso/s |
| `nk_euclideans_packed_bf16_genoa`           |             64.5 gso/s |             86.6 gso/s |             92.8 gso/s |
| `nk_euclideans_symmetric_bf16_genoa`        |             53.2 gso/s |             69.8 gso/s |             76.2 gso/s |
| `nk_angulars_packed_bf16_sapphireamx`       |              364 gso/s |              645 gso/s |              760 gso/s |
| `nk_angulars_symmetric_bf16_sapphireamx`    |             90.7 gso/s |              108 gso/s |              153 gso/s |
| `nk_euclideans_packed_bf16_sapphireamx`     |              352 gso/s |              655 gso/s |              761 gso/s |
| `nk_euclideans_symmetric_bf16_sapphireamx`  |             90.1 gso/s |              110 gso/s |              154 gso/s |
| `nk_angulars_packed_bf16_ampere`            |   1,954 gso/s, 0.1 ulp |  65,415 gso/s, 0.2 ulp | 288,001 gso/s, 0.6 ulp |
| `nk_angulars_symmetric_bf16_ampere`         |     941 gso/s, 0.1 ulp |  30,799 gso/s, 0.2 ulp | 176,244 gso/s, 0.6 ulp |
| `nk_euclideans_packed_bf16_ampere`          |   1,880 gso/s, 0.3 ulp |  64,376 gso/s, 0.3 ulp | 287,180 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_bf16_ampere`       |     862 gso/s, 0.3 ulp |  29,393 gso/s, 0.3 ulp | 183,288 gso/s, 0.5 ulp |
| __f16__                                     | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f16_serial`             |             3.38 gso/s |             3.49 gso/s |             3.53 gso/s |
| `nk_angulars_symmetric_f16_serial`          |            0.675 gso/s |            0.673 gso/s |            0.662 gso/s |
| `nk_euclideans_packed_f16_serial`           |             3.65 gso/s |             3.49 gso/s |             3.50 gso/s |
| `nk_euclideans_symmetric_f16_serial`        |            0.682 gso/s |            0.673 gso/s |            0.671 gso/s |
| `nk_angulars_packed_f16_haswell`            |             47.0 gso/s |             52.0 gso/s |             51.1 gso/s |
| `nk_angulars_symmetric_f16_haswell`         |             32.5 gso/s |             45.2 gso/s |             52.1 gso/s |
| `nk_euclideans_packed_f16_haswell`          |             45.3 gso/s |             51.5 gso/s |             56.4 gso/s |
| `nk_euclideans_symmetric_f16_haswell`       |             35.6 gso/s |             45.8 gso/s |             56.7 gso/s |
| `nk_angulars_packed_f16_skylake`            |             74.3 gso/s |             96.3 gso/s |             95.1 gso/s |
| `nk_angulars_symmetric_f16_skylake`         |             41.2 gso/s |             56.0 gso/s |             63.6 gso/s |
| `nk_euclideans_packed_f16_skylake`          |             80.5 gso/s |             98.7 gso/s |             93.4 gso/s |
| `nk_euclideans_symmetric_f16_skylake`       |             43.0 gso/s |             58.3 gso/s |             63.6 gso/s |
| `nk_angulars_packed_f16_graniteamx`         |              360 gso/s |              644 gso/s |              750 gso/s |
| `nk_angulars_symmetric_f16_graniteamx`      |             93.0 gso/s |              107 gso/s |              151 gso/s |
| `nk_euclideans_packed_f16_graniteamx`       |              349 gso/s |              659 gso/s |              761 gso/s |
| `nk_euclideans_symmetric_f16_graniteamx`    |             88.9 gso/s |              110 gso/s |              151 gso/s |
| `nk_angulars_packed_f16_ampere`             |   1,952 gso/s, 0.2 ulp |  65,402 gso/s, 0.4 ulp | 265,683 gso/s, 0.8 ulp |
| `nk_angulars_symmetric_f16_ampere`          |     936 gso/s, 0.2 ulp |  30,564 gso/s, 0.4 ulp | 181,239 gso/s, 0.7 ulp |
| `nk_euclideans_packed_f16_ampere`           |   1,878 gso/s, 0.3 ulp |  64,254 gso/s, 0.4 ulp | 265,097 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_f16_ampere`        |     858 gso/s, 0.4 ulp |  29,200 gso/s, 0.4 ulp | 177,603 gso/s, 0.6 ulp |
| __e5m2__                                    | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e5m2_serial`            |            0.760 gso/s |            0.761 gso/s |            0.749 gso/s |
| `nk_angulars_symmetric_e5m2_serial`         |            0.747 gso/s |            0.754 gso/s |            0.725 gso/s |
| `nk_euclideans_packed_e5m2_serial`          |            0.746 gso/s |            0.759 gso/s |            0.761 gso/s |
| `nk_euclideans_symmetric_e5m2_serial`       |            0.749 gso/s |            0.749 gso/s |            0.756 gso/s |
| `nk_angulars_packed_e5m2_haswell`           |             39.7 gso/s |             43.7 gso/s |             46.2 gso/s |
| `nk_angulars_symmetric_e5m2_haswell`        |             36.0 gso/s |             45.2 gso/s |             47.5 gso/s |
| `nk_euclideans_packed_e5m2_haswell`         |             43.0 gso/s |             43.7 gso/s |             46.2 gso/s |
| `nk_euclideans_symmetric_e5m2_haswell`      |             36.0 gso/s |             44.9 gso/s |             47.7 gso/s |
| `nk_angulars_packed_e5m2_skylake`           |             50.2 gso/s |             60.6 gso/s |             66.0 gso/s |
| `nk_angulars_symmetric_e5m2_skylake`        |             46.6 gso/s |             62.6 gso/s |             65.7 gso/s |
| `nk_euclideans_packed_e5m2_skylake`         |             52.1 gso/s |             60.5 gso/s |             64.0 gso/s |
| `nk_euclideans_symmetric_e5m2_skylake`      |             50.7 gso/s |             64.4 gso/s |             66.9 gso/s |
| `nk_angulars_packed_e5m2_genoa`             |             47.4 gso/s |             56.3 gso/s |             62.6 gso/s |
| `nk_angulars_symmetric_e5m2_genoa`          |             35.7 gso/s |             41.0 gso/s |             44.0 gso/s |
| `nk_euclideans_packed_e5m2_genoa`           |             49.2 gso/s |             56.8 gso/s |             63.7 gso/s |
| `nk_euclideans_symmetric_e5m2_genoa`        |             34.1 gso/s |             41.6 gso/s |             44.9 gso/s |
| `nk_angulars_packed_e5m2_sapphireamx`       |              216 gso/s |              447 gso/s |              531 gso/s |
| `nk_angulars_symmetric_e5m2_sapphireamx`    |             61.7 gso/s |             80.9 gso/s |             79.0 gso/s |
| `nk_euclideans_packed_e5m2_sapphireamx`     |              207 gso/s |              466 gso/s |              540 gso/s |
| `nk_euclideans_symmetric_e5m2_sapphireamx`  |             59.6 gso/s |             82.4 gso/s |             79.6 gso/s |
| `nk_angulars_packed_e5m2_graniteamx`        |              330 gso/s |              606 gso/s |              685 gso/s |
| `nk_angulars_symmetric_e5m2_graniteamx`     |             97.1 gso/s |              166 gso/s |              167 gso/s |
| `nk_euclideans_packed_e5m2_graniteamx`      |              311 gso/s |              614 gso/s |              686 gso/s |
| `nk_euclideans_symmetric_e5m2_graniteamx`   |             93.2 gso/s |              173 gso/s |              168 gso/s |
| `nk_angulars_packed_e5m2_ampere`            |     2,002 gso/s, 0 ulp |    71,370 gso/s, 0 ulp |   314,398 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_ampere`          |   1,959 gso/s, 0.2 ulp |  69,905 gso/s, 0.2 ulp | 311,196 gso/s, 0.3 ulp |
| `nk_angulars_packed_e5m2_blackwellrtx`      |     2,286 gso/s, 0 ulp |    94,307 gso/s, 0 ulp |   482,991 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_blackwellrtx`    |   2,200 gso/s, 0.2 ulp |  92,207 gso/s, 0.2 ulp | 481,033 gso/s, 0.3 ulp |
| __e4m3__                                    | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e4m3_serial`            |            0.564 gso/s |            0.570 gso/s |            0.559 gso/s |
| `nk_angulars_symmetric_e4m3_serial`         |            0.559 gso/s |            0.561 gso/s |            0.536 gso/s |
| `nk_euclideans_packed_e4m3_serial`          |            0.563 gso/s |            0.569 gso/s |            0.569 gso/s |
| `nk_euclideans_symmetric_e4m3_serial`       |            0.579 gso/s |            0.560 gso/s |            0.558 gso/s |
| `nk_angulars_packed_e4m3_haswell`           |             18.4 gso/s |             19.4 gso/s |             19.8 gso/s |
| `nk_angulars_symmetric_e4m3_haswell`        |             17.7 gso/s |             19.2 gso/s |             19.9 gso/s |
| `nk_euclideans_packed_e4m3_haswell`         |             20.8 gso/s |             19.2 gso/s |             19.7 gso/s |
| `nk_euclideans_symmetric_e4m3_haswell`      |             19.0 gso/s |             19.2 gso/s |             19.8 gso/s |
| `nk_angulars_packed_e4m3_skylake`           |             43.0 gso/s |             49.7 gso/s |             53.4 gso/s |
| `nk_angulars_symmetric_e4m3_skylake`        |             29.8 gso/s |             33.5 gso/s |             35.4 gso/s |
| `nk_euclideans_packed_e4m3_skylake`         |             43.6 gso/s |             49.7 gso/s |             53.2 gso/s |
| `nk_euclideans_symmetric_e4m3_skylake`      |             30.2 gso/s |             33.7 gso/s |             35.4 gso/s |
| `nk_angulars_packed_e4m3_genoa`             |             43.3 gso/s |             50.8 gso/s |             55.4 gso/s |
| `nk_angulars_symmetric_e4m3_genoa`          |             32.0 gso/s |             36.4 gso/s |             38.3 gso/s |
| `nk_euclideans_packed_e4m3_genoa`           |             43.4 gso/s |             50.7 gso/s |             57.2 gso/s |
| `nk_euclideans_symmetric_e4m3_genoa`        |             30.7 gso/s |             36.6 gso/s |             39.4 gso/s |
| `nk_angulars_packed_e4m3_sapphireamx`       |              198 gso/s |              330 gso/s |              415 gso/s |
| `nk_angulars_symmetric_e4m3_sapphireamx`    |             50.7 gso/s |             65.1 gso/s |             64.2 gso/s |
| `nk_euclideans_packed_e4m3_sapphireamx`     |              188 gso/s |              338 gso/s |              418 gso/s |
| `nk_euclideans_symmetric_e4m3_sapphireamx`  |             49.2 gso/s |             66.4 gso/s |             64.5 gso/s |
| `nk_angulars_packed_e4m3_ampere`            |     1,922 gso/s, 0 ulp |    64,337 gso/s, 0 ulp |   276,452 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_ampere`          |   1,897 gso/s, 0.2 ulp |  63,940 gso/s, 0.2 ulp | 275,414 gso/s, 0.3 ulp |
| `nk_angulars_packed_e4m3_blackwellrtx`      |     2,243 gso/s, 0 ulp |    91,652 gso/s, 0 ulp |   468,183 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_blackwellrtx`   |     1,076 gso/s, 0 ulp |    40,085 gso/s, 0 ulp |   286,777 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_blackwellrtx`    |   2,163 gso/s, 0.2 ulp |  88,845 gso/s, 0.2 ulp | 466,218 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_e4m3_blackwellrtx` |     977 gso/s, 0.2 ulp |  37,746 gso/s, 0.2 ulp | 280,475 gso/s, 0.3 ulp |
| __e3m2__                                    | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e3m2_serial`            |            0.912 gso/s |            0.917 gso/s |            0.879 gso/s |
| `nk_angulars_symmetric_e3m2_serial`         |            0.931 gso/s |            0.952 gso/s |            0.909 gso/s |
| `nk_euclideans_packed_e3m2_serial`          |            0.932 gso/s |            0.923 gso/s |            0.922 gso/s |
| `nk_euclideans_symmetric_e3m2_serial`       |            0.936 gso/s |            0.947 gso/s |            0.955 gso/s |
| `nk_angulars_packed_e3m2_haswell`           |             29.0 gso/s |             30.9 gso/s |             32.8 gso/s |
| `nk_angulars_symmetric_e3m2_haswell`        |             30.0 gso/s |             31.2 gso/s |             33.6 gso/s |
| `nk_euclideans_packed_e3m2_haswell`         |             29.9 gso/s |             31.1 gso/s |             32.5 gso/s |
| `nk_euclideans_symmetric_e3m2_haswell`      |             30.4 gso/s |             31.3 gso/s |             33.7 gso/s |
| `nk_angulars_packed_e3m2_skylake`           |             44.6 gso/s |             47.9 gso/s |             53.6 gso/s |
| `nk_angulars_symmetric_e3m2_skylake`        |             44.5 gso/s |             51.8 gso/s |             54.6 gso/s |
| `nk_euclideans_packed_e3m2_skylake`         |             46.4 gso/s |             47.4 gso/s |             53.4 gso/s |
| `nk_euclideans_symmetric_e3m2_skylake`      |             42.7 gso/s |             51.7 gso/s |             54.9 gso/s |
| `nk_angulars_packed_e3m2_sapphireamx`       |              284 gso/s |              421 gso/s |              504 gso/s |
| `nk_angulars_symmetric_e3m2_sapphireamx`    |             76.2 gso/s |              116 gso/s |              108 gso/s |
| `nk_euclideans_packed_e3m2_sapphireamx`     |              288 gso/s |              434 gso/s |              509 gso/s |
| `nk_euclideans_symmetric_e3m2_sapphireamx`  |             71.4 gso/s |              118 gso/s |              109 gso/s |
| `nk_angulars_packed_e3m2_ampere`            |     1,928 gso/s, 0 ulp |    65,070 gso/s, 0 ulp |   282,541 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_ampere`          |     1,906 gso/s, 0 ulp |  64,548 gso/s, 0.1 ulp | 285,442 gso/s, 0.3 ulp |
| `nk_angulars_packed_e3m2_blackwellrtx`      |     2,245 gso/s, 0 ulp |    90,807 gso/s, 0 ulp |   463,370 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_blackwellrtx`   |     1,073 gso/s, 0 ulp |    39,871 gso/s, 0 ulp |   287,090 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_blackwellrtx`    |     2,159 gso/s, 0 ulp |  88,872 gso/s, 0.1 ulp | 459,540 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_e3m2_blackwellrtx` |       974 gso/s, 0 ulp |  37,548 gso/s, 0.2 ulp | 281,467 gso/s, 0.5 ulp |
| __e2m3__                                    | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_serial`            |            0.759 gso/s |            0.765 gso/s |            0.750 gso/s |
| `nk_angulars_symmetric_e2m3_serial`         |            0.749 gso/s |            0.760 gso/s |            0.744 gso/s |
| `nk_euclideans_packed_e2m3_serial`          |            0.762 gso/s |            0.764 gso/s |            0.764 gso/s |
| `nk_euclideans_symmetric_e2m3_serial`       |            0.766 gso/s |            0.755 gso/s |            0.758 gso/s |
| `nk_angulars_packed_e2m3_haswell`           |             41.6 gso/s |             46.2 gso/s |             48.3 gso/s |
| `nk_angulars_symmetric_e2m3_haswell`        |             38.2 gso/s |             46.1 gso/s |             48.8 gso/s |
| `nk_euclideans_packed_e2m3_haswell`         |             41.5 gso/s |             43.8 gso/s |             47.3 gso/s |
| `nk_euclideans_symmetric_e2m3_haswell`      |             39.8 gso/s |             46.6 gso/s |             48.7 gso/s |
| `nk_angulars_packed_e2m3_skylake`           |             72.0 gso/s |             81.8 gso/s |             94.3 gso/s |
| `nk_angulars_symmetric_e2m3_skylake`        |             59.2 gso/s |             85.7 gso/s |             96.2 gso/s |
| `nk_euclideans_packed_e2m3_skylake`         |             71.8 gso/s |             81.9 gso/s |             96.7 gso/s |
| `nk_euclideans_symmetric_e2m3_skylake`      |             66.9 gso/s |             86.1 gso/s |             91.4 gso/s |
| `nk_angulars_packed_e2m3_sapphireamx`       |              373 gso/s |              708 gso/s |            1,000 gso/s |
| `nk_angulars_symmetric_e2m3_sapphireamx`    |              114 gso/s |              228 gso/s |              247 gso/s |
| `nk_euclideans_packed_e2m3_sapphireamx`     |              405 gso/s |              755 gso/s |            1,011 gso/s |
| `nk_euclideans_symmetric_e2m3_sapphireamx`  |              111 gso/s |              241 gso/s |              249 gso/s |
| `nk_angulars_packed_e2m3_alder`             |             46.9 gso/s |             50.3 gso/s |             53.9 gso/s |
| `nk_angulars_symmetric_e2m3_alder`          |             43.8 gso/s |             52.6 gso/s |             56.0 gso/s |
| `nk_euclideans_packed_e2m3_alder`           |             47.7 gso/s |             52.3 gso/s |             54.4 gso/s |
| `nk_euclideans_symmetric_e2m3_alder`        |             46.7 gso/s |             52.5 gso/s |             55.8 gso/s |
| `nk_angulars_packed_e2m3_ampere`            |     2,086 gso/s, 0 ulp |    77,519 gso/s, 0 ulp |   373,218 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_ampere`          |     2,064 gso/s, 0 ulp |    77,049 gso/s, 0 ulp |   372,304 gso/s, 0 ulp |
| `nk_angulars_packed_e2m3_blackwellrtx`      |     2,250 gso/s, 0 ulp |    90,900 gso/s, 0 ulp |   499,278 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_blackwellrtx`    |     2,218 gso/s, 0 ulp |    90,415 gso/s, 0 ulp |   496,651 gso/s, 0 ulp |
| __e2m1__                                    | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m1_serial`            |            0.736 gso/s |            0.740 gso/s |            0.751 gso/s |
| `nk_angulars_symmetric_e2m1_serial`         |            0.764 gso/s |            0.749 gso/s |            0.753 gso/s |
| `nk_euclideans_packed_e2m1_serial`          |            0.737 gso/s |            0.740 gso/s |            0.742 gso/s |
| `nk_euclideans_symmetric_e2m1_serial`       |            0.789 gso/s |            0.751 gso/s |            0.754 gso/s |
| `nk_angulars_packed_e2m1_haswell`           |             81.4 gso/s |              101 gso/s |              122 gso/s |
| `nk_angulars_symmetric_e2m1_haswell`        |             53.0 gso/s |             90.7 gso/s |              114 gso/s |
| `nk_euclideans_packed_e2m1_haswell`         |             82.0 gso/s |             99.5 gso/s |              114 gso/s |
| `nk_euclideans_symmetric_e2m1_haswell`      |             56.3 gso/s |             91.9 gso/s |              111 gso/s |
| `nk_angulars_packed_e2m1_skylake`           |             83.9 gso/s |              120 gso/s |              143 gso/s |
| `nk_angulars_symmetric_e2m1_skylake`        |             58.6 gso/s |              108 gso/s |              129 gso/s |
| `nk_euclideans_packed_e2m1_skylake`         |             89.3 gso/s |              119 gso/s |              147 gso/s |
| `nk_euclideans_symmetric_e2m1_skylake`      |             59.6 gso/s |              109 gso/s |              130 gso/s |
| `nk_angulars_packed_e2m1_sapphireamx`       |              230 gso/s |              604 gso/s |              928 gso/s |
| `nk_angulars_symmetric_e2m1_sapphireamx`    |             73.9 gso/s |              179 gso/s |              174 gso/s |
| `nk_euclideans_packed_e2m1_sapphireamx`     |              218 gso/s |              613 gso/s |              930 gso/s |
| `nk_euclideans_symmetric_e2m1_sapphireamx`  |             73.4 gso/s |              180 gso/s |              175 gso/s |
| `nk_angulars_packed_e2m1_alder`             |             96.0 gso/s |              130 gso/s |              157 gso/s |
| `nk_angulars_symmetric_e2m1_alder`          |             57.4 gso/s |              112 gso/s |              135 gso/s |
| `nk_euclideans_packed_e2m1_alder`           |             97.0 gso/s |              129 gso/s |              161 gso/s |
| `nk_euclideans_symmetric_e2m1_alder`        |             62.8 gso/s |              114 gso/s |              137 gso/s |
| `nk_angulars_packed_e2m1_ampere`            |     2,188 gso/s, 0 ulp |    85,987 gso/s, 0 ulp |   409,559 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m1_ampere`          |     2,173 gso/s, 0 ulp |    87,031 gso/s, 0 ulp |   419,990 gso/s, 0 ulp |
| `nk_angulars_packed_e2m1_blackwellrtx`      |     2,546 gso/s, 0 ulp |   126,737 gso/s, 0 ulp |   930,321 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m1_blackwellrtx`   |     1,224 gso/s, 0 ulp |    58,632 gso/s, 0 ulp |   557,672 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m1_blackwellrtx`    |     2,456 gso/s, 0 ulp |   124,337 gso/s, 0 ulp |   922,602 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m1_blackwellrtx` |     1,137 gso/s, 0 ulp |    54,954 gso/s, 0 ulp |   540,373 gso/s, 0 ulp |
| __i8__                                      | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_serial`              |             4.83 gso/s |             5.63 gso/s |             5.63 gso/s |
| `nk_angulars_symmetric_i8_serial`           |             4.37 gso/s |             5.15 gso/s |             5.36 gso/s |
| `nk_euclideans_packed_i8_serial`            |             5.05 gso/s |             5.69 gso/s |             5.94 gso/s |
| `nk_euclideans_symmetric_i8_serial`         |             4.45 gso/s |             5.17 gso/s |             5.40 gso/s |
| `nk_angulars_packed_i8_haswell`             |             66.7 gso/s |             76.3 gso/s |             83.9 gso/s |
| `nk_angulars_symmetric_i8_haswell`          |             53.4 gso/s |             75.8 gso/s |             83.5 gso/s |
| `nk_euclideans_packed_i8_haswell`           |             67.5 gso/s |             76.9 gso/s |             80.3 gso/s |
| `nk_euclideans_symmetric_i8_haswell`        |             63.2 gso/s |             76.4 gso/s |             85.3 gso/s |
| `nk_angulars_packed_i8_icelake`             |              149 gso/s |              202 gso/s |              289 gso/s |
| `nk_angulars_symmetric_i8_icelake`          |             86.9 gso/s |              164 gso/s |              181 gso/s |
| `nk_euclideans_packed_i8_icelake`           |              171 gso/s |              202 gso/s |              319 gso/s |
| `nk_euclideans_symmetric_i8_icelake`        |              106 gso/s |              157 gso/s |              180 gso/s |
| `nk_angulars_packed_i8_sapphireamx`         |              617 gso/s |              989 gso/s |            1,227 gso/s |
| `nk_angulars_symmetric_i8_sapphireamx`      |              137 gso/s |              267 gso/s |              282 gso/s |
| `nk_euclideans_packed_i8_sapphireamx`       |              519 gso/s |            1,025 gso/s |            1,217 gso/s |
| `nk_euclideans_symmetric_i8_sapphireamx`    |              134 gso/s |              268 gso/s |              284 gso/s |
| `nk_angulars_packed_i8_alder`               |              119 gso/s |              130 gso/s |              162 gso/s |
| `nk_angulars_symmetric_i8_alder`            |             66.9 gso/s |              131 gso/s |              140 gso/s |
| `nk_euclideans_packed_i8_alder`             |              123 gso/s |              133 gso/s |              150 gso/s |
| `nk_euclideans_symmetric_i8_alder`          |             91.3 gso/s |              129 gso/s |              140 gso/s |
| `nk_angulars_packed_i8_ampere`              |            2,285 gso/s |           95,803 gso/s |          528,754 gso/s |
| `nk_angulars_symmetric_i8_ampere`           |            1,131 gso/s |           47,502 gso/s |          346,230 gso/s |
| `nk_euclideans_packed_i8_ampere`            |            2,223 gso/s |           94,228 gso/s |          526,768 gso/s |
| `nk_euclideans_symmetric_i8_ampere`         |            1,038 gso/s |           45,266 gso/s |          339,560 gso/s |
| __u8__                                      | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`              |             4.77 gso/s |             5.53 gso/s |             5.66 gso/s |
| `nk_angulars_symmetric_u8_serial`           |             4.50 gso/s |             5.31 gso/s |             5.56 gso/s |
| `nk_euclideans_packed_u8_serial`            |             5.07 gso/s |             5.49 gso/s |             5.72 gso/s |
| `nk_euclideans_symmetric_u8_serial`         |             4.59 gso/s |             5.35 gso/s |             5.58 gso/s |
| `nk_angulars_packed_u8_haswell`             |             48.1 gso/s |             66.4 gso/s |             86.7 gso/s |
| `nk_angulars_symmetric_u8_haswell`          |             37.9 gso/s |             68.2 gso/s |             81.6 gso/s |
| `nk_euclideans_packed_u8_haswell`           |             48.5 gso/s |             64.7 gso/s |             78.3 gso/s |
| `nk_euclideans_symmetric_u8_haswell`        |             40.7 gso/s |             67.8 gso/s |             81.4 gso/s |
| `nk_angulars_packed_u8_icelake`             |              147 gso/s |              225 gso/s |              277 gso/s |
| `nk_angulars_symmetric_u8_icelake`          |             99.2 gso/s |              163 gso/s |              184 gso/s |
| `nk_euclideans_packed_u8_icelake`           |              175 gso/s |              210 gso/s |              253 gso/s |
| `nk_euclideans_symmetric_u8_icelake`        |              107 gso/s |              158 gso/s |              184 gso/s |
| `nk_angulars_packed_u8_sapphireamx`         |              618 gso/s |              999 gso/s |            1,295 gso/s |
| `nk_angulars_symmetric_u8_sapphireamx`      |              133 gso/s |              270 gso/s |              285 gso/s |
| `nk_euclideans_packed_u8_sapphireamx`       |              567 gso/s |            1,029 gso/s |            1,271 gso/s |
| `nk_euclideans_symmetric_u8_sapphireamx`    |              127 gso/s |              272 gso/s |              287 gso/s |
| `nk_angulars_packed_u8_alder`               |              117 gso/s |              133 gso/s |              165 gso/s |
| `nk_angulars_symmetric_u8_alder`            |             67.0 gso/s |              132 gso/s |              140 gso/s |
| `nk_euclideans_packed_u8_alder`             |              120 gso/s |              132 gso/s |              153 gso/s |
| `nk_euclideans_symmetric_u8_alder`          |             89.0 gso/s |              136 gso/s |              143 gso/s |
| `nk_angulars_packed_u8_ampere`              |            2,286 gso/s |           95,790 gso/s |          529,888 gso/s |
| `nk_euclideans_packed_u8_ampere`            |            2,224 gso/s |           94,177 gso/s |          528,224 gso/s |
| __i4__                                      | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i4_serial`              |            0.873 gso/s |            0.898 gso/s |            0.905 gso/s |
| `nk_angulars_symmetric_i4_serial`           |             1.45 gso/s |             1.51 gso/s |             1.52 gso/s |
| `nk_euclideans_packed_i4_serial`            |            0.874 gso/s |            0.895 gso/s |            0.903 gso/s |
| `nk_euclideans_symmetric_i4_serial`         |             1.48 gso/s |             1.51 gso/s |             1.53 gso/s |
| `nk_angulars_packed_i4_icelake`             |              132 gso/s |              194 gso/s |              267 gso/s |
| `nk_angulars_symmetric_i4_icelake`          |              129 gso/s |              207 gso/s |              230 gso/s |
| `nk_euclideans_packed_i4_icelake`           |              145 gso/s |              190 gso/s |              249 gso/s |
| `nk_euclideans_symmetric_i4_icelake`        |              106 gso/s |              208 gso/s |              232 gso/s |
| `nk_angulars_packed_i4_ampere`              |            2,308 gso/s |          100,007 gso/s |          536,382 gso/s |
| `nk_euclideans_packed_i4_ampere`            |            2,245 gso/s |           97,718 gso/s |          540,531 gso/s |
| __u4__                                      | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u4_serial`              |             2.21 gso/s |             2.35 gso/s |             2.40 gso/s |
| `nk_angulars_symmetric_u4_serial`           |             3.35 gso/s |             3.73 gso/s |             3.84 gso/s |
| `nk_euclideans_packed_u4_serial`            |             2.25 gso/s |             2.35 gso/s |             2.39 gso/s |
| `nk_euclideans_symmetric_u4_serial`         |             3.42 gso/s |             3.74 gso/s |             3.84 gso/s |
| `nk_angulars_packed_u4_icelake`             |              161 gso/s |              230 gso/s |              359 gso/s |
| `nk_angulars_symmetric_u4_icelake`          |              133 gso/s |              227 gso/s |              283 gso/s |
| `nk_euclideans_packed_u4_icelake`           |              179 gso/s |              242 gso/s |              319 gso/s |
| `nk_euclideans_symmetric_u4_icelake`        |              110 gso/s |              228 gso/s |              256 gso/s |
| `nk_angulars_packed_u4_ampere`              |            2,333 gso/s |          102,329 gso/s |          613,437 gso/s |
| `nk_euclideans_packed_u4_ampere`            |            2,270 gso/s |          100,162 gso/s |          605,160 gso/s |

### Apple M5

#### Native

| Kernel                                   |                     256³ |                    1024³ |                    4096³ |
| :--------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f64_serial`          |        2.37 gso/s, 0 ulp |        2.35 gso/s, 0 ulp |        2.67 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_serial`       |     1.36 gso/s, 0.04 ulp |     1.41 gso/s, 0.02 ulp |     1.56 gso/s, 0.01 ulp |
| `nk_euclideans_packed_f64_serial`        |      2.36 gso/s, 0.6 ulp |      2.41 gso/s, 0.6 ulp |      2.67 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_f64_serial`     |      1.44 gso/s, 0.6 ulp |      1.52 gso/s, 0.6 ulp |      1.56 gso/s, 0.6 ulp |
| `nk_angulars_packed_f64_neon`            |    6.05 gso/s, 7,798 ulp |    6.28 gso/s, 3,868 ulp |    6.34 gso/s, 1,720 ulp |
| `nk_angulars_symmetric_f64_neon`         |    5.29 gso/s, 7,660 ulp |    5.39 gso/s, 3,790 ulp |    5.44 gso/s, 1,720 ulp |
| `nk_euclideans_packed_f64_neon`          |      5.97 gso/s, 0.2 ulp |      5.97 gso/s, 0.2 ulp |      6.37 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f64_neon`       |      5.25 gso/s, 0.2 ulp |      5.29 gso/s, 0.2 ulp |      5.48 gso/s, 0.2 ulp |
| `nk_angulars_packed_f64_smef64`          |     40.6 gso/s, 0.02 ulp |     44.7 gso/s, 0.02 ulp |     46.0 gso/s, 0.02 ulp |
| `nk_angulars_symmetric_f64_smef64`       |     19.9 gso/s, 0.02 ulp |     24.1 gso/s, 0.02 ulp |     20.8 gso/s, 0.02 ulp |
| `nk_euclideans_packed_f64_smef64`        |     41.0 gso/s, 0.24 ulp |     44.9 gso/s, 0.24 ulp |     46.1 gso/s, 0.24 ulp |
| `nk_euclideans_symmetric_f64_smef64`     |     20.2 gso/s, 0.28 ulp |     24.1 gso/s, 0.28 ulp |     20.9 gso/s, 0.28 ulp |
| __f32__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f32_serial`          |      10.7 gso/s, 0.1 ulp |      11.7 gso/s, 0.1 ulp |      12.2 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f32_serial`       |      8.31 gso/s, 0.3 ulp |      8.76 gso/s, 0.3 ulp |      9.69 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f32_serial`        |      10.5 gso/s, 0.6 ulp |      11.4 gso/s, 0.5 ulp |      12.5 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_f32_serial`     |      8.89 gso/s, 3.9 ulp |      8.91 gso/s, 7.9 ulp |      9.69 gso/s, 3.4 ulp |
| `nk_angulars_packed_f32_neon`            |        37.6 gso/s, 0 ulp |        40.6 gso/s, 0 ulp |    42.2 gso/s, 1,740 ulp |
| `nk_angulars_symmetric_f32_neon`         |    9.73 gso/s, 7,690 ulp |    10.5 gso/s, 3,830 ulp |    10.8 gso/s, 1,730 ulp |
| `nk_euclideans_packed_f32_neon`          |      37.9 gso/s, 0.2 ulp |      39.7 gso/s, 0.2 ulp |      42.0 gso/s, 3.5 ulp |
| `nk_euclideans_symmetric_f32_neon`       |      10.1 gso/s, 3.8 ulp |      10.3 gso/s, 7.8 ulp |      10.9 gso/s, 3.5 ulp |
| `nk_angulars_packed_f32_smef64`          |      149 gso/s, 0.15 ulp |      230 gso/s, 0.15 ulp |      214 gso/s, 0.15 ulp |
| `nk_angulars_symmetric_f32_smef64`       |     50.7 gso/s, 0.13 ulp |     85.4 gso/s, 0.13 ulp |     54.0 gso/s, 0.13 ulp |
| `nk_euclideans_packed_f32_smef64`        |       151 gso/s, 2.2 ulp |       230 gso/s, 2.2 ulp |       213 gso/s, 2.2 ulp |
| `nk_euclideans_symmetric_f32_smef64`     |      51.7 gso/s, 1.5 ulp |      86.1 gso/s, 1.5 ulp |      54.2 gso/s, 1.5 ulp |
| __bf16__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_bf16_serial`         |        18.0 gso/s, 0 ulp |      19.9 gso/s, 0.1 ulp |        20.7 gso/s, 0 ulp |
| `nk_angulars_symmetric_bf16_serial`      |     15.6 gso/s, 0.04 ulp |      16.9 gso/s, 0.1 ulp |     18.6 gso/s, 0.04 ulp |
| `nk_euclideans_packed_bf16_serial`       |      19.1 gso/s, 0.6 ulp |      19.5 gso/s, 3.1 ulp |      21.8 gso/s, 2.1 ulp |
| `nk_euclideans_symmetric_bf16_serial`    |      15.9 gso/s, 0.6 ulp |      16.9 gso/s, 3.1 ulp |      18.6 gso/s, 2.1 ulp |
| `nk_angulars_packed_bf16_neonbfdot`      |        56.0 gso/s, 0 ulp |      57.4 gso/s, 0.1 ulp |     63.2 gso/s, 0.04 ulp |
| `nk_angulars_symmetric_bf16_neonbfdot`   |        37.5 gso/s, 0 ulp |      39.6 gso/s, 0.1 ulp |     43.4 gso/s, 0.04 ulp |
| `nk_euclideans_packed_bf16_neonbfdot`    |      55.7 gso/s, 0.3 ulp |      56.5 gso/s, 2.9 ulp |      62.1 gso/s, 1.9 ulp |
| `nk_euclideans_symmetric_bf16_neonbfdot` |      39.0 gso/s, 0.3 ulp |      42.1 gso/s, 2.9 ulp |      43.2 gso/s, 1.9 ulp |
| `nk_angulars_packed_bf16_sme`            |      728 gso/s, 0.04 ulp |    1,476 gso/s, 0.04 ulp |    1,451 gso/s, 0.04 ulp |
| `nk_angulars_symmetric_bf16_sme`         |      471 gso/s, 0.03 ulp |    1,011 gso/s, 0.03 ulp |      993 gso/s, 0.03 ulp |
| `nk_euclideans_packed_bf16_sme`          |      468 gso/s, 0.54 ulp |      886 gso/s, 0.54 ulp |    1,109 gso/s, 0.54 ulp |
| `nk_euclideans_symmetric_bf16_sme`       |      412 gso/s, 0.28 ulp |      901 gso/s, 0.28 ulp |      924 gso/s, 0.28 ulp |
| __f16__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f16_serial`          |      12.8 gso/s, 0.1 ulp |      14.4 gso/s, 0.1 ulp |      14.9 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_serial`       |      21.7 gso/s, 0.1 ulp |     25.2 gso/s, 0.09 ulp |      28.2 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_serial`        |      13.1 gso/s, 1.1 ulp |      13.9 gso/s, 0.7 ulp |      15.7 gso/s, 5.6 ulp |
| `nk_euclideans_symmetric_f16_serial`     |      23.6 gso/s, 1.1 ulp |      25.2 gso/s, 0.7 ulp |      28.4 gso/s, 5.6 ulp |
| `nk_angulars_packed_f16_neon`            |      72.2 gso/s, 0.1 ulp |      78.6 gso/s, 0.1 ulp |      83.8 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_neon`         |      19.3 gso/s, 0.1 ulp |      20.9 gso/s, 0.1 ulp |      21.8 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_neon`          |      73.0 gso/s, 0.9 ulp |      76.2 gso/s, 0.7 ulp |      83.7 gso/s, 5.9 ulp |
| `nk_euclideans_symmetric_f16_neon`       |      19.2 gso/s, 0.9 ulp |      20.2 gso/s, 0.6 ulp |      21.9 gso/s, 5.8 ulp |
| `nk_angulars_packed_f16_neonfhm`         |      96.2 gso/s, 0.1 ulp |       107 gso/s, 0.1 ulp |       118 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_neonfhm`      |      35.4 gso/s, 0.1 ulp |      39.1 gso/s, 0.1 ulp |      42.5 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_neonfhm`       |       100 gso/s, 0.9 ulp |       110 gso/s, 0.7 ulp |       119 gso/s, 5.9 ulp |
| `nk_euclideans_symmetric_f16_neonfhm`    |      37.2 gso/s, 0.9 ulp |      39.4 gso/s, 0.6 ulp |      42.0 gso/s, 5.8 ulp |
| `nk_angulars_packed_f16_sme`             |       835 gso/s, 0.1 ulp |     1,531 gso/s, 0.1 ulp |     1,750 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_sme`          |       529 gso/s, 0.1 ulp |     1,081 gso/s, 0.1 ulp |     1,046 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_sme`           |         491 gso/s, 0 ulp |      906 gso/s, 0.06 ulp |     1,118 gso/s, 2.9 ulp |
| `nk_euclideans_symmetric_f16_sme`        |       451 gso/s, 0.3 ulp |       970 gso/s, 0.6 ulp |       969 gso/s, 0.3 ulp |
| __e5m2__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e5m2_serial`         |        15.8 gso/s, 0 ulp |        16.7 gso/s, 0 ulp |        17.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_serial`      |        7.78 gso/s, 0 ulp |        8.37 gso/s, 0 ulp |        8.99 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_serial`       |      15.5 gso/s, 0.5 ulp |      16.7 gso/s, 0.5 ulp |      17.2 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_e5m2_serial`    |      7.93 gso/s, 0.5 ulp |      8.37 gso/s, 0.5 ulp |      8.99 gso/s, 0.5 ulp |
| `nk_angulars_packed_e5m2_neonfhm`        |        84.3 gso/s, 0 ulp |        97.3 gso/s, 0 ulp |         103 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_neonfhm`     |        58.8 gso/s, 0 ulp |        73.2 gso/s, 0 ulp |        79.3 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_neonfhm`      |        88.1 gso/s, 0 ulp |         110 gso/s, 0 ulp |         119 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e5m2_neonfhm`   |        66.1 gso/s, 0 ulp |        60.3 gso/s, 0 ulp |        64.4 gso/s, 0 ulp |
| `nk_angulars_packed_e5m2_sme`            |      685 gso/s, 0.01 ulp |    1,391 gso/s, 0.01 ulp |    1,705 gso/s, 0.01 ulp |
| `nk_angulars_symmetric_e5m2_sme`         |      359 gso/s, 0.01 ulp |      766 gso/s, 0.01 ulp |      893 gso/s, 0.01 ulp |
| `nk_euclideans_packed_e5m2_sme`          |     399 gso/s, 0.005 ulp |     655 gso/s, 0.005 ulp |     762 gso/s, 0.005 ulp |
| `nk_euclideans_symmetric_e5m2_sme`       |     324 gso/s, 0.004 ulp |     706 gso/s, 0.004 ulp |     811 gso/s, 0.004 ulp |
| __e4m3__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e4m3_serial`         |        1.15 gso/s, 0 ulp |        1.20 gso/s, 0 ulp |        1.24 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_serial`      |     1.22 gso/s, 0.03 ulp |     1.24 gso/s, 0.02 ulp |     1.32 gso/s, 0.01 ulp |
| `nk_euclideans_packed_e4m3_serial`       |      1.23 gso/s, 0.5 ulp |      1.20 gso/s, 0.5 ulp |      1.24 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_e4m3_serial`    |      1.25 gso/s, 0.5 ulp |      1.24 gso/s, 0.5 ulp |      1.32 gso/s, 0.3 ulp |
| `nk_angulars_packed_e4m3_neonfhm`        |        29.1 gso/s, 0 ulp |        32.2 gso/s, 0 ulp |        34.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_neonfhm`     |        32.0 gso/s, 0 ulp |        36.6 gso/s, 0 ulp |        38.9 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_neonfhm`      |        30.0 gso/s, 0 ulp |        32.2 gso/s, 0 ulp |      34.1 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e4m3_neonfhm`   |        34.1 gso/s, 0 ulp |        36.6 gso/s, 0 ulp |      38.9 gso/s, 0.2 ulp |
| `nk_angulars_packed_e4m3_sme`            |      621 gso/s, 0.01 ulp |    1,347 gso/s, 0.01 ulp |    1,687 gso/s, 0.01 ulp |
| `nk_angulars_symmetric_e4m3_sme`         |      303 gso/s, 0.01 ulp |      650 gso/s, 0.01 ulp |      728 gso/s, 0.01 ulp |
| `nk_euclideans_packed_e4m3_sme`          |      200 gso/s, 0.11 ulp |      279 gso/s, 0.11 ulp |      310 gso/s, 0.11 ulp |
| `nk_euclideans_symmetric_e4m3_sme`       |      276 gso/s, 0.11 ulp |      600 gso/s, 0.11 ulp |      708 gso/s, 0.11 ulp |
| __e3m2__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e3m2_serial`         |        14.2 gso/s, 0 ulp |        14.6 gso/s, 0 ulp |        15.5 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_serial`      |        7.77 gso/s, 0 ulp |        8.10 gso/s, 0 ulp |        9.05 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_serial`       |      13.9 gso/s, 0.5 ulp |      14.6 gso/s, 0.5 ulp |      15.5 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_e3m2_serial`    |      8.08 gso/s, 0.5 ulp |      8.10 gso/s, 0.5 ulp |      9.05 gso/s, 0.5 ulp |
| `nk_angulars_packed_e3m2_sme`            |      715 gso/s, 0.01 ulp |    1,421 gso/s, 0.01 ulp |    1,727 gso/s, 0.01 ulp |
| `nk_angulars_symmetric_e3m2_sme`         |      382 gso/s, 0.01 ulp |      806 gso/s, 0.01 ulp |      908 gso/s, 0.01 ulp |
| `nk_euclideans_packed_e3m2_sme`          |         379 gso/s, 0 ulp |         604 gso/s, 0 ulp |         702 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e3m2_sme`       |         341 gso/s, 0 ulp |         743 gso/s, 0 ulp |         835 gso/s, 0 ulp |
| __e2m3__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_serial`         |        14.1 gso/s, 0 ulp |        14.8 gso/s, 0 ulp |        15.5 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_serial`      |        7.89 gso/s, 0 ulp |        8.21 gso/s, 0 ulp |        9.09 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_serial`       |      13.6 gso/s, 0.5 ulp |      14.8 gso/s, 0.5 ulp |      15.7 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_e2m3_serial`    |      7.93 gso/s, 0.5 ulp |      8.21 gso/s, 0.5 ulp |      9.09 gso/s, 0.5 ulp |
| `nk_angulars_packed_e2m3_sme`            |    1,005 gso/s, 0.01 ulp |    2,347 gso/s, 0.01 ulp |    3,353 gso/s, 0.01 ulp |
| `nk_angulars_symmetric_e2m3_sme`         |      387 gso/s, 0.01 ulp |      940 gso/s, 0.01 ulp |    1,406 gso/s, 0.01 ulp |
| `nk_euclideans_packed_e2m3_sme`          |         470 gso/s, 0 ulp |       1,011 gso/s, 0 ulp |       1,269 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_sme`       |         345 gso/s, 0 ulp |         851 gso/s, 0 ulp |       1,393 gso/s, 0 ulp |
| __i8__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_serial`           |        18.3 gso/s, 0 ulp |        20.0 gso/s, 0 ulp |        20.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_serial`        |        13.5 gso/s, 0 ulp |        13.9 gso/s, 0 ulp |        14.8 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_serial`         |      18.7 gso/s, 0.4 ulp |      20.0 gso/s, 0.4 ulp |      20.2 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_i8_serial`      |      13.7 gso/s, 0.4 ulp |      13.9 gso/s, 0.4 ulp |      14.8 gso/s, 0.4 ulp |
| `nk_angulars_packed_i8_neonsdot`         |         280 gso/s, 0 ulp |         357 gso/s, 0 ulp |         477 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_neonsdot`      |        74.0 gso/s, 0 ulp |        86.9 gso/s, 0 ulp |        87.2 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_neonsdot`       |         305 gso/s, 0 ulp |         419 gso/s, 0 ulp |         477 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_neonsdot`    |        73.4 gso/s, 0 ulp |        87.0 gso/s, 0 ulp |        87.2 gso/s, 0 ulp |
| `nk_angulars_packed_i8_sme`              |      558 gso/s, 0.31 ulp |    1,612 gso/s, 0.29 ulp |    2,826 gso/s, 0.30 ulp |
| `nk_angulars_symmetric_i8_sme`           |      450 gso/s, 0.32 ulp |    1,467 gso/s, 0.28 ulp |    1,761 gso/s, 0.30 ulp |
| `nk_euclideans_packed_i8_sme`            |         823 gso/s, 0 ulp |       2,096 gso/s, 0 ulp |    3,198 gso/s, 0.15 ulp |
| `nk_euclideans_symmetric_i8_sme`         |         717 gso/s, 0 ulp |       1,738 gso/s, 0 ulp |    2,012 gso/s, 0.15 ulp |
| __u8__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`           |      15.5 gso/s, 0.3 ulp |      16.3 gso/s, 0.3 ulp |      17.4 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_serial`        |      15.7 gso/s, 0.3 ulp |      16.2 gso/s, 0.3 ulp |      17.5 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u8_serial`         |      16.4 gso/s, 0.5 ulp |      16.3 gso/s, 0.5 ulp |      17.4 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_u8_serial`      |      16.1 gso/s, 0.5 ulp |      16.2 gso/s, 0.5 ulp |      17.5 gso/s, 0.6 ulp |
| `nk_angulars_packed_u8_neonsdot`         |       284 gso/s, 0.3 ulp |       369 gso/s, 0.3 ulp |       470 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_neonsdot`      |      72.4 gso/s, 0.3 ulp |      87.4 gso/s, 0.3 ulp |      87.7 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u8_neonsdot`       |         302 gso/s, 0 ulp |         419 gso/s, 0 ulp |         470 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u8_neonsdot`    |        72.0 gso/s, 0 ulp |        87.0 gso/s, 0 ulp |        87.7 gso/s, 0 ulp |
| `nk_angulars_packed_u8_sme`              |      619 gso/s, 0.38 ulp |    1,726 gso/s, 0.41 ulp |    2,909 gso/s, 0.40 ulp |
| `nk_angulars_symmetric_u8_sme`           |      490 gso/s, 0.40 ulp |    1,538 gso/s, 0.43 ulp |    1,930 gso/s, 0.41 ulp |
| `nk_euclideans_packed_u8_sme`            |         817 gso/s, 0 ulp |       2,101 gso/s, 0 ulp |    3,123 gso/s, 0.15 ulp |
| `nk_euclideans_symmetric_u8_sme`         |         706 gso/s, 0 ulp |       1,763 gso/s, 0 ulp |    2,008 gso/s, 0.15 ulp |
| __i4__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i4_serial`           |      17.3 gso/s, 0.3 ulp |      18.2 gso/s, 0.3 ulp |      19.6 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_i4_serial`        |      14.3 gso/s, 0.3 ulp |      14.9 gso/s, 0.3 ulp |      15.6 gso/s, 0.3 ulp |
| `nk_euclideans_packed_i4_serial`         |      17.9 gso/s, 0.5 ulp |      18.2 gso/s, 0.5 ulp |      19.6 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_i4_serial`      |      14.6 gso/s, 0.5 ulp |      14.9 gso/s, 0.5 ulp |      15.6 gso/s, 0.6 ulp |
| `nk_angulars_packed_i4_neonsdot`         |       215 gso/s, 0.3 ulp |       284 gso/s, 0.3 ulp |       291 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_i4_neonsdot`      |       104 gso/s, 0.3 ulp |       162 gso/s, 0.3 ulp |       171 gso/s, 0.3 ulp |
| `nk_euclideans_packed_i4_neonsdot`       |         225 gso/s, 0 ulp |         284 gso/s, 0 ulp |         291 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i4_neonsdot`    |         105 gso/s, 0 ulp |         162 gso/s, 0 ulp |         171 gso/s, 0 ulp |
| `nk_angulars_packed_i4_sme`              |      557 gso/s, 0.34 ulp |    1,599 gso/s, 0.33 ulp |    2,705 gso/s, 0.37 ulp |
| `nk_angulars_symmetric_i4_sme`           |      442 gso/s, 0.32 ulp |    1,417 gso/s, 0.33 ulp |    1,880 gso/s, 0.36 ulp |
| `nk_euclideans_packed_i4_sme`            |         825 gso/s, 0 ulp |       2,095 gso/s, 0 ulp |       3,178 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i4_sme`         |         696 gso/s, 0 ulp |       1,697 gso/s, 0 ulp |       1,990 gso/s, 0 ulp |
| __u4__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u4_serial`           |      18.0 gso/s, 0.3 ulp |      19.4 gso/s, 0.3 ulp |      20.6 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u4_serial`        |      15.5 gso/s, 0.3 ulp |      16.4 gso/s, 0.3 ulp |      17.4 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u4_serial`         |      19.1 gso/s, 0.5 ulp |      19.4 gso/s, 0.5 ulp |      20.6 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_u4_serial`      |      15.7 gso/s, 0.5 ulp |      16.4 gso/s, 0.5 ulp |      17.4 gso/s, 0.6 ulp |
| `nk_angulars_packed_u4_neonsdot`         |       241 gso/s, 0.3 ulp |       319 gso/s, 0.3 ulp |       340 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u4_neonsdot`      |       107 gso/s, 0.3 ulp |       166 gso/s, 0.3 ulp |       173 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u4_neonsdot`       |         250 gso/s, 0 ulp |         340 gso/s, 0 ulp |         340 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u4_neonsdot`    |         105 gso/s, 0 ulp |         173 gso/s, 0 ulp |         173 gso/s, 0 ulp |
| `nk_angulars_packed_u4_sme`              |      622 gso/s, 0.38 ulp |    1,741 gso/s, 0.35 ulp |    2,925 gso/s, 0.34 ulp |
| `nk_angulars_symmetric_u4_sme`           |      496 gso/s, 0.39 ulp |    1,562 gso/s, 0.35 ulp |    2,029 gso/s, 0.33 ulp |
| `nk_euclideans_packed_u4_sme`            |         827 gso/s, 0 ulp |       2,103 gso/s, 0 ulp |       3,165 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u4_sme`         |         729 gso/s, 0 ulp |       1,782 gso/s, 0 ulp |       2,135 gso/s, 0 ulp |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                                     |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f64_serial`            |        2.33 gso/s, 0 ulp |        2.12 gso/s, 0 ulp |        2.19 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_serial`         |        1.35 gso/s, 0 ulp |        1.41 gso/s, 0 ulp |        1.57 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_serial`          |      2.35 gso/s, 0.4 ulp |      2.37 gso/s, 0.4 ulp |      2.50 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_f64_serial`       |      1.40 gso/s, 0.4 ulp |      1.50 gso/s, 0.4 ulp |      1.59 gso/s, 0.4 ulp |
| `nk_angulars_packed_f64_v128relaxed`       |      5.65 gso/s, 0.1 ulp |      5.44 gso/s, 0.1 ulp |      6.22 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f64_v128relaxed`    |      5.29 gso/s, 0.1 ulp |      5.74 gso/s, 0.1 ulp |      6.01 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f64_v128relaxed`     |      5.69 gso/s, 0.4 ulp |      6.05 gso/s, 0.4 ulp |      6.22 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_f64_v128relaxed`  |      5.29 gso/s, 0.4 ulp |      5.89 gso/s, 0.4 ulp |      6.02 gso/s, 0.4 ulp |
| __f32__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f32_serial`            |      10.3 gso/s, 0.1 ulp |      10.1 gso/s, 0.1 ulp |      10.6 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f32_serial`         |      8.18 gso/s, 0.1 ulp |      8.59 gso/s, 0.1 ulp |      9.54 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f32_serial`          |      10.4 gso/s, 0.3 ulp |      10.4 gso/s, 0.3 ulp |      11.0 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_f32_serial`       |      8.58 gso/s, 0.3 ulp |      8.67 gso/s, 0.3 ulp |      9.52 gso/s, 0.3 ulp |
| `nk_angulars_packed_f32_v128relaxed`       |      25.2 gso/s, 0.1 ulp |      30.7 gso/s, 0.1 ulp |      32.3 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f32_v128relaxed`    |      9.91 gso/s, 0.1 ulp |      10.9 gso/s, 0.1 ulp |      11.1 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f32_v128relaxed`     |      26.6 gso/s, 0.2 ulp |      30.7 gso/s, 0.2 ulp |      32.2 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f32_v128relaxed`  |      9.98 gso/s, 0.2 ulp |      10.9 gso/s, 0.2 ulp |      11.1 gso/s, 0.2 ulp |
| __bf16__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_bf16_serial`           |      21.7 gso/s, 0.3 ulp |      21.3 gso/s, 0.3 ulp |      24.3 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_bf16_serial`        |      22.2 gso/s, 0.3 ulp |      24.4 gso/s, 0.3 ulp |      27.8 gso/s, 0.3 ulp |
| `nk_euclideans_packed_bf16_serial`         |      19.1 gso/s, 5.3 ulp |      21.4 gso/s, 5.3 ulp |      24.2 gso/s, 5.3 ulp |
| `nk_euclideans_symmetric_bf16_serial`      |      22.0 gso/s, 5.3 ulp |      24.7 gso/s, 5.3 ulp |      27.7 gso/s, 5.3 ulp |
| `nk_angulars_packed_bf16_v128relaxed`      |      70.2 gso/s, 0.3 ulp |      82.3 gso/s, 0.3 ulp |      89.9 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_bf16_v128relaxed`   |      36.9 gso/s, 0.3 ulp |      44.9 gso/s, 0.3 ulp |      47.3 gso/s, 0.3 ulp |
| `nk_euclideans_packed_bf16_v128relaxed`    |      76.4 gso/s, 5.3 ulp |      87.3 gso/s, 5.3 ulp |      89.7 gso/s, 5.3 ulp |
| `nk_euclideans_symmetric_bf16_v128relaxed` |      37.1 gso/s, 5.3 ulp |      43.1 gso/s, 5.3 ulp |      45.5 gso/s, 5.3 ulp |
| `nk_angulars_packed_bf16_v128`             |                        … |                        … |                        … |
| `nk_angulars_symmetric_bf16_v128`          |                        … |                        … |                        … |
| `nk_euclideans_packed_bf16_v128`           |                        … |                        … |                        … |
| `nk_euclideans_symmetric_bf16_v128`        |                        … |                        … |                        … |
| __e2m3__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_serial`           |        5.78 gso/s, 0 ulp |        5.93 gso/s, 0 ulp |        6.28 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_serial`        |        6.52 gso/s, 0 ulp |        8.09 gso/s, 0 ulp |        8.52 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_serial`         |      5.25 gso/s, 0.3 ulp |      5.94 gso/s, 0.3 ulp |      6.20 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_e2m3_serial`      |      6.99 gso/s, 0.3 ulp |      8.08 gso/s, 0.3 ulp |      8.52 gso/s, 0.3 ulp |
| `nk_angulars_packed_e2m3_v128relaxed`      |        36.7 gso/s, 0 ulp |        38.8 gso/s, 0 ulp |        39.9 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_v128relaxed`   |        31.4 gso/s, 0 ulp |        37.3 gso/s, 0 ulp |        39.5 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_v128relaxed`    |        36.8 gso/s, 0 ulp |        38.9 gso/s, 0 ulp |        40.0 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_v128relaxed` |        31.8 gso/s, 0 ulp |        37.5 gso/s, 0 ulp |        39.5 gso/s, 0 ulp |
| __i8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_serial`             |        13.7 gso/s, 0 ulp |        16.4 gso/s, 0 ulp |        17.3 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_serial`          |        10.5 gso/s, 0 ulp |        12.8 gso/s, 0 ulp |        13.4 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_serial`           |      14.4 gso/s, 0.5 ulp |      16.5 gso/s, 0.5 ulp |      17.3 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_i8_serial`        |      11.3 gso/s, 0.5 ulp |      12.6 gso/s, 0.5 ulp |      13.4 gso/s, 0.5 ulp |
| `nk_angulars_packed_i8_v128relaxed`        |        45.2 gso/s, 0 ulp |        50.0 gso/s, 0 ulp |        52.0 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_v128relaxed`     |        37.7 gso/s, 0 ulp |        47.5 gso/s, 0 ulp |        50.4 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_v128relaxed`      |        45.6 gso/s, 0 ulp |        50.2 gso/s, 0 ulp |        52.0 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_v128relaxed`   |        37.4 gso/s, 0 ulp |        46.8 gso/s, 0 ulp |        50.4 gso/s, 0 ulp |
| `nk_angulars_packed_i8_v128`               |                        … |                        … |                        … |
| `nk_angulars_symmetric_i8_v128`            |                        … |                        … |                        … |
| `nk_euclideans_packed_i8_v128`             |                        … |                        … |                        … |
| `nk_euclideans_symmetric_i8_v128`          |                        … |                        … |                        … |
| __u8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`             |      14.7 gso/s, 0.3 ulp |      17.0 gso/s, 0.3 ulp |      17.8 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_serial`          |      10.9 gso/s, 0.3 ulp |      13.2 gso/s, 0.3 ulp |      13.9 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u8_serial`           |      14.9 gso/s, 0.4 ulp |      17.0 gso/s, 0.4 ulp |      17.8 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_u8_serial`        |      11.7 gso/s, 0.4 ulp |      13.1 gso/s, 0.4 ulp |      13.9 gso/s, 0.4 ulp |
| `nk_angulars_packed_u8_v128relaxed`        |      43.7 gso/s, 0.3 ulp |      49.0 gso/s, 0.3 ulp |      50.7 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_v128relaxed`     |      34.7 gso/s, 0.3 ulp |      45.6 gso/s, 0.3 ulp |      48.4 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u8_v128relaxed`      |        44.8 gso/s, 0 ulp |        49.0 gso/s, 0 ulp |        50.7 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u8_v128relaxed`   |        35.0 gso/s, 0 ulp |        44.2 gso/s, 0 ulp |        48.5 gso/s, 0 ulp |
| `nk_angulars_packed_u8_v128`               |                        … |                        … |                        … |
| `nk_angulars_symmetric_u8_v128`            |                        … |                        … |                        … |
| `nk_euclideans_packed_u8_v128`             |                        … |                        … |                        … |
| `nk_euclideans_symmetric_u8_v128`          |                        … |                        … |                        … |
