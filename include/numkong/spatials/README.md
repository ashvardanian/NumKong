# Batched Distance Matrices in NumKong

NumKong implements batched distance matrix computation via pre-packed dot products plus normalization. Angular distance and Euclidean distance are computed from the packed dot product output without materializing an intermediate C matrix.

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
| ---------- | ----------- | ------------------------------------------------ |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision                 |
| `f32`      | `f64`       | 32-bit IEEE 754 single precision, widened output |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output   |
| `bf16`     | `f32`       | 16-bit brain float, widened output               |
| `e4m3`     | `f32`       | 8-bit Float8: 4 exponent, 3 mantissa bits        |
| `e5m2`     | `f32`       | 8-bit Float8: 5 exponent, 2 mantissa bits        |
| `e2m3`     | `f32`       | 8-bit MX format: 2 exponent, 3 mantissa bits     |
| `e3m2`     | `f32`       | 8-bit MX format: 3 exponent, 2 mantissa bits     |
| `i8`       | `f32`       | 8-bit signed integers, float output              |
| `u8`       | `f32`       | 8-bit unsigned integers, float output            |
| `i4`       | `f32`       | 4-bit signed integers, float output              |
| `u4`       | `f32`       | 4-bit unsigned integers, float output            |

## Optimizations

### Distance-from-Dot Algebraic Reduction

`nk_angulars_packed_f32_haswell`, `nk_angulars_packed_f32_skylake`, `nk_euclideans_packed_f32_haswell`, `nk_euclideans_packed_f32_skylake` derive distance matrices from pre-packed dot product output without materializing an intermediate result matrix.
Angular distance rewrites as $1 - \text{dot}(a,b) \cdot \text{rsqrt}(\|a\|^2 \cdot \|b\|^2)$, converting two separate square roots and a division into one rsqrt and one multiply.
Euclidean distance expands the identity $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2 \cdot \text{dot}(a,b)$, requiring only one final sqrt per output element.
Both formulas decompose into: (1) a batched GEMM for all M×N dot products, (2) per-vector squared norms precomputed once during packing.
The singular `spatial/` kernels compute these three sums ($\sum a_i b_i$, $\sum a_i^2$, $\sum b_i^2$) in a single pass with three interleaved accumulators; the batched `spatials/` kernels separate them — norms are computed once per vector during packing, and dots come from the GEMM — trading register pressure for amortized cost across the full M×N output.

### Serial vs Vectorized Sqrt and Rsqrt Cost

`nk_angular_through_f32_from_dot_serial_` uses the Quake 3 fast inverse square root (magic constant `0x5F375A86`, three Newton-Raphson iterations, ~34.9 correct bits for Float32) to compute `dot * rsqrt(query_norm * target_norm)`.
`nk_angular_through_f32_from_dot_haswell_` replaces this with hardware `_mm_rsqrt_ps` (~12-bit approximation, 5cy latency, 1/cy on port 0) plus one Newton-Raphson refinement step (~22–24 correct bits).
`nk_euclidean_through_f32_from_dot_serial_` computes `sqrt(x)` as `x * rsqrt(x)` — reusing the same rsqrt path.
`nk_euclidean_through_f32_from_dot_haswell_` uses exact `_mm_sqrt_ps` (11cy latency, 7cy throughput for XMM) instead of the rsqrt approximation — the subtraction $\|a\|^2 + \|b\|^2 - 2 \cdot \text{dot}$ can produce values near zero where rsqrt error would be amplified by the subsequent multiply.
For Float64, the SIMD backends use exact division and sqrt — no fast rsqrt approximation, since reaching 52 mantissa bits of precision would need 4+ Newton-Raphson iterations, negating the speed advantage — while `nk_angular_through_f64_from_dot_serial_` still routes through `nk_f64_rsqrt_serial` and its four iterations.
The 4-wide finalizer batching amortizes these costs: one rsqrt or sqrt call processes 4 output elements simultaneously, hiding the latency behind the GEMM tile's computation.

### Norm Precomputation in Packed Buffers

`nk_dots_pack_f32_serial`, `nk_dots_pack_f32_haswell`, `nk_dots_pack_bf16_haswell` compute per-column squared norms $\|b_j\|^2 = \sum_k b_{jk}^2 = \text{dot}(b_j, b_j)$ during the packing step via `nk_reduce_moments_*` primitives.
The squared norm is a self-dot-product — already a byproduct of touching every element for type conversion and layout transformation.
Angular and Euclidean finalizers read norms from packed buffer metadata, eliminating a separate O(N·K) norm pass over B.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by `NK_MATRIX_HEIGHT`, `NK_MATRIX_WIDTH`, and `NK_MATRIX_DEPTH` environment variables, all set to the same value for batched distance computations over square matrices.
Columns show throughput for 256³, 1024³, and 4096³ configurations.
The throughput is measured in GSO/s as Giga Scalar Operations per Second, with $\text{ops} = 2 \cdot M \cdot N \cdot K$ complexity for computing $M \times N$ pairwise distances over $K$-dimensional vectors.
Accuracy is reported as mean ULP (units in last place) unless noted otherwise — the average number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                                     |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f64_serial`            |       0.578 gso/s, 0 ulp |       0.691 gso/s, 0 ulp |       0.787 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_serial`         |       0.477 gso/s, 0 ulp |       0.569 gso/s, 0 ulp |        1.24 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_serial`          |     0.569 gso/s, 0.6 ulp |     0.692 gso/s, 0.6 ulp |     0.775 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_f64_serial`       |     0.477 gso/s, 0.6 ulp |     0.562 gso/s, 0.6 ulp |      1.26 gso/s, 0.3 ulp |
| `nk_angulars_packed_f64_haswell`           |        5.89 gso/s, 0 ulp |        6.04 gso/s, 0 ulp |        6.08 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_haswell`        |        5.17 gso/s, 0 ulp |        5.56 gso/s, 0 ulp |        11.3 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_haswell`         |      5.83 gso/s, 0.2 ulp |      6.21 gso/s, 0.2 ulp |      6.24 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f64_haswell`      |      5.33 gso/s, 0.2 ulp |      5.62 gso/s, 0.2 ulp |      11.7 gso/s, 0.2 ulp |
| `nk_angulars_packed_f64_skylake`           |        7.56 gso/s, 0 ulp |        8.46 gso/s, 0 ulp |        8.92 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_skylake`        |        7.37 gso/s, 0 ulp |        8.66 gso/s, 0 ulp |        17.1 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_skylake`         |      8.06 gso/s, 0.2 ulp |      8.37 gso/s, 0.2 ulp |      8.06 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f64_skylake`      |      7.14 gso/s, 0.2 ulp |      8.43 gso/s, 0.2 ulp |      17.4 gso/s, 0.2 ulp |
| __f32__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f32_serial`            |      15.0 gso/s, 0.1 ulp |      16.3 gso/s, 0.1 ulp |      16.4 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f32_serial`         |      3.86 gso/s, 0.1 ulp |      4.29 gso/s, 0.1 ulp |      8.62 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f32_serial`          |      15.3 gso/s, 0.6 ulp |      17.0 gso/s, 0.5 ulp |      17.0 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_f32_serial`       |      3.97 gso/s, 0.6 ulp |      4.16 gso/s, 0.5 ulp |      8.38 gso/s, 0.3 ulp |
| `nk_angulars_packed_f32_haswell`           |        29.3 gso/s, 0 ulp |        31.6 gso/s, 0 ulp |        31.6 gso/s, 0 ulp |
| `nk_angulars_symmetric_f32_haswell`        |        21.4 gso/s, 0 ulp |        24.8 gso/s, 0 ulp |          52 gso/s, 0 ulp |
| `nk_euclideans_packed_f32_haswell`         |      29.7 gso/s, 0.2 ulp |        32 gso/s, 0.2 ulp |      32.9 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f32_haswell`      |      21.8 gso/s, 0.2 ulp |      25.7 gso/s, 0.2 ulp |        53 gso/s, 0.2 ulp |
| `nk_angulars_packed_f32_skylake`           |        33.3 gso/s, 0 ulp |        39.4 gso/s, 0 ulp |        37.5 gso/s, 0 ulp |
| `nk_angulars_symmetric_f32_skylake`        |        24.8 gso/s, 0 ulp |        25.5 gso/s, 0 ulp |        61.4 gso/s, 0 ulp |
| `nk_euclideans_packed_f32_skylake`         |      34.4 gso/s, 0.2 ulp |      40.3 gso/s, 0.2 ulp |      40.3 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f32_skylake`      |      25.1 gso/s, 0.2 ulp |      29.3 gso/s, 0.2 ulp |      65.9 gso/s, 0.2 ulp |
| __bf16__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_bf16_serial`           |        1.18 gso/s, 0 ulp |        1.21 gso/s, 0 ulp |      1.19 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_bf16_serial`        |        1.19 gso/s, 0 ulp |        1.18 gso/s, 0 ulp |        2.35 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_serial`         |      1.20 gso/s, 0.6 ulp |      1.18 gso/s, 0.6 ulp |      1.16 gso/s, 6.0 ulp |
| `nk_euclideans_symmetric_bf16_serial`      |      1.11 gso/s, 0.6 ulp |      1.14 gso/s, 0.6 ulp |      2.34 gso/s, 0.4 ulp |
| `nk_angulars_packed_bf16_haswell`          |        54.6 gso/s, 0 ulp |        65.7 gso/s, 0 ulp |      66.1 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_bf16_haswell`       |        38.3 gso/s, 0 ulp |        50.1 gso/s, 0 ulp |         106 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_haswell`        |        58 gso/s, 0.2 ulp |      65.7 gso/s, 0.3 ulp |      70.7 gso/s, 5.8 ulp |
| `nk_euclideans_symmetric_bf16_haswell`     |      38.6 gso/s, 0.2 ulp |      49.8 gso/s, 0.3 ulp |       109 gso/s, 0.3 ulp |
| `nk_angulars_packed_bf16_skylake`          |        67.8 gso/s, 0 ulp |        87.7 gso/s, 0 ulp |      86.4 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_bf16_skylake`       |        48.8 gso/s, 0 ulp |        58.7 gso/s, 0 ulp |         125 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_skylake`        |        64 gso/s, 0.2 ulp |      87.4 gso/s, 0.3 ulp |      90.8 gso/s, 5.8 ulp |
| `nk_euclideans_symmetric_bf16_skylake`     |      48.8 gso/s, 0.2 ulp |      58.9 gso/s, 0.3 ulp |       121 gso/s, 0.3 ulp |
| `nk_angulars_packed_bf16_genoa`            |        59.7 gso/s, 0 ulp |        81.9 gso/s, 0 ulp |        87.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_bf16_genoa`         |        54.9 gso/s, 0 ulp |        61.2 gso/s, 0 ulp |         137 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_genoa`          |        63 gso/s, 0.2 ulp |      79.6 gso/s, 0.3 ulp |      87.3 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_bf16_genoa`       |      53.4 gso/s, 0.2 ulp |      60.2 gso/s, 0.3 ulp |       130 gso/s, 0.3 ulp |
| `nk_angulars_packed_bf16_sapphireamx`      |         287 gso/s, 0 ulp |         364 gso/s, 0 ulp |         582 gso/s, 0 ulp |
| `nk_angulars_symmetric_bf16_sapphireamx`   |        75.7 gso/s, 0 ulp |         114 gso/s, 0 ulp |         116 gso/s, 0 ulp |
| `nk_euclideans_packed_bf16_sapphireamx`    |       328 gso/s, 0.3 ulp |       573 gso/s, 0.3 ulp |       632 gso/s, 0.3 ulp |
| `nk_euclideans_symmetric_bf16_sapphireamx` |      76.3 gso/s, 0.3 ulp |       115 gso/s, 0.3 ulp |       123 gso/s, 0.3 ulp |
| __f16__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f16_serial`            |      7.46 gso/s, 0.1 ulp |      7.97 gso/s, 0.1 ulp |      8.12 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_serial`         |      4.04 gso/s, 0.1 ulp |      4.09 gso/s, 0.1 ulp |      8.13 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_serial`          |      7.69 gso/s, 0.7 ulp |      7.73 gso/s, 1.1 ulp |      8.34 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_f16_serial`       |      4.08 gso/s, 0.7 ulp |      4.19 gso/s, 1.1 ulp |      8.23 gso/s, 0.5 ulp |
| `nk_angulars_packed_f16_haswell`           |        62 gso/s, 0.1 ulp |      74.4 gso/s, 0.1 ulp |      70.6 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_haswell`        |      38.3 gso/s, 0.1 ulp |      54.9 gso/s, 0.1 ulp |       121 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_haswell`         |      62.9 gso/s, 0.4 ulp |      75.2 gso/s, 0.9 ulp |      75.7 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_f16_haswell`      |      39.6 gso/s, 0.4 ulp |      54.2 gso/s, 0.9 ulp |       123 gso/s, 0.3 ulp |
| `nk_angulars_packed_f16_skylake`           |      66.6 gso/s, 0.1 ulp |      85.2 gso/s, 0.1 ulp |        88.3 gso/s, 0 ulp |
| `nk_angulars_symmetric_f16_skylake`        |      50.1 gso/s, 0.1 ulp |      57.7 gso/s, 0.1 ulp |         126 gso/s, 0 ulp |
| `nk_euclideans_packed_f16_skylake`         |      69.6 gso/s, 0.4 ulp |      93.3 gso/s, 0.9 ulp |        91 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_f16_skylake`      |      49.4 gso/s, 0.4 ulp |      59.8 gso/s, 0.9 ulp |       134 gso/s, 0.3 ulp |
| __e5m2__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e5m2_serial`           |       0.587 gso/s, 0 ulp |       0.553 gso/s, 0 ulp |       0.563 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_serial`        |       0.446 gso/s, 0 ulp |       0.427 gso/s, 0 ulp |       0.847 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_serial`         |     0.576 gso/s, 0.5 ulp |     0.571 gso/s, 0.5 ulp |     0.557 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e5m2_serial`      |     0.424 gso/s, 0.5 ulp |     0.437 gso/s, 0.5 ulp |     0.836 gso/s, 0.2 ulp |
| `nk_angulars_packed_e5m2_haswell`          |        27.4 gso/s, 0 ulp |        30.4 gso/s, 0 ulp |          31 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_haswell`       |        15.3 gso/s, 0 ulp |        15.7 gso/s, 0 ulp |        32.3 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_haswell`        |          28 gso/s, 0 ulp |        30.8 gso/s, 0 ulp |        30.6 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e5m2_haswell`     |        15.4 gso/s, 0 ulp |        15.9 gso/s, 0 ulp |          32 gso/s, 0 ulp |
| `nk_angulars_packed_e5m2_skylake`          |        32.9 gso/s, 0 ulp |        36.7 gso/s, 0 ulp |        40.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_skylake`       |          19 gso/s, 0 ulp |          21 gso/s, 0 ulp |        42.7 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_skylake`        |        34.1 gso/s, 0 ulp |        37.9 gso/s, 0 ulp |        39.6 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e5m2_skylake`     |          20 gso/s, 0 ulp |        18.4 gso/s, 0 ulp |        41.6 gso/s, 0 ulp |
| `nk_angulars_packed_e5m2_genoa`            |        39.6 gso/s, 0 ulp |        46.8 gso/s, 0 ulp |        47.5 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_genoa`         |          30 gso/s, 0 ulp |        32.5 gso/s, 0 ulp |        66.3 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_genoa`          |        42.3 gso/s, 0 ulp |        49.1 gso/s, 0 ulp |        51.3 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e5m2_genoa`       |        30.1 gso/s, 0 ulp |        32.8 gso/s, 0 ulp |        64.9 gso/s, 0 ulp |
| `nk_angulars_packed_e5m2_sapphireamx`      |         216 gso/s, 0 ulp |         355 gso/s, 0 ulp |         427 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_sapphireamx`   |        48.7 gso/s, 0 ulp |        73.3 gso/s, 0 ulp |        72.3 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_sapphireamx`    |         220 gso/s, 0 ulp |         375 gso/s, 0 ulp |         408 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e5m2_sapphireamx` |        48.3 gso/s, 0 ulp |        73.3 gso/s, 0 ulp |          74 gso/s, 0 ulp |
| __e4m3__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e4m3_serial`           |       0.479 gso/s, 0 ulp |       0.473 gso/s, 0 ulp |       0.485 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_serial`        |       0.395 gso/s, 0 ulp |       0.390 gso/s, 0 ulp |       0.795 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_serial`         |     0.467 gso/s, 0.5 ulp |     0.484 gso/s, 0.5 ulp |     0.480 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_e4m3_serial`      |     0.395 gso/s, 0.5 ulp |     0.395 gso/s, 0.5 ulp |     0.781 gso/s, 0.3 ulp |
| `nk_angulars_packed_e4m3_haswell`          |        20.6 gso/s, 0 ulp |        22.5 gso/s, 0 ulp |        21.8 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_haswell`       |        12.2 gso/s, 0 ulp |        12.1 gso/s, 0 ulp |        24.7 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_haswell`        |        20.7 gso/s, 0 ulp |        22.4 gso/s, 0 ulp |      23.4 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e4m3_haswell`     |        11.2 gso/s, 0 ulp |        11.9 gso/s, 0 ulp |      24.6 gso/s, 0.1 ulp |
| `nk_angulars_packed_e4m3_skylake`          |        28.8 gso/s, 0 ulp |        32.8 gso/s, 0 ulp |        31.3 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_skylake`       |        16.4 gso/s, 0 ulp |        17.4 gso/s, 0 ulp |        35.1 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_skylake`        |        27.8 gso/s, 0 ulp |        31.2 gso/s, 0 ulp |      31.7 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e4m3_skylake`     |        16.1 gso/s, 0 ulp |        16.8 gso/s, 0 ulp |      34.4 gso/s, 0.1 ulp |
| `nk_angulars_packed_e4m3_genoa`            |        40.8 gso/s, 0 ulp |        48.4 gso/s, 0 ulp |        52.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_genoa`         |        30.3 gso/s, 0 ulp |        31.5 gso/s, 0 ulp |        69.2 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_genoa`          |        43.3 gso/s, 0 ulp |        50.9 gso/s, 0 ulp |      48.8 gso/s, 0.1 ulp |
| `nk_euclideans_symmetric_e4m3_genoa`       |        29.9 gso/s, 0 ulp |        31.9 gso/s, 0 ulp |      64.6 gso/s, 0.1 ulp |
| `nk_angulars_packed_e4m3_sapphireamx`      |         212 gso/s, 0 ulp |         325 gso/s, 0 ulp |         418 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_sapphireamx`   |        50.5 gso/s, 0 ulp |        73.4 gso/s, 0 ulp |          72 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_sapphireamx`    |       216 gso/s, 0.1 ulp |       372 gso/s, 0.1 ulp |       394 gso/s, 0.1 ulp |
| `nk_euclideans_symmetric_e4m3_sapphireamx` |      49.3 gso/s, 0.1 ulp |      70.1 gso/s, 0.1 ulp |      73.1 gso/s, 0.1 ulp |
| __e3m2__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e3m2_serial`           |       0.554 gso/s, 0 ulp |       0.524 gso/s, 0 ulp |       0.534 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_serial`        |       0.439 gso/s, 0 ulp |       0.427 gso/s, 0 ulp |       0.839 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_serial`         |     0.556 gso/s, 0.5 ulp |     0.549 gso/s, 0.5 ulp |     0.509 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e3m2_serial`      |     0.413 gso/s, 0.5 ulp |     0.427 gso/s, 0.5 ulp |     0.829 gso/s, 0.2 ulp |
| `nk_angulars_packed_e3m2_haswell`          |        30.3 gso/s, 0 ulp |        32.2 gso/s, 0 ulp |        32.8 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_haswell`       |        27.1 gso/s, 0 ulp |        32.8 gso/s, 0 ulp |        65.7 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_haswell`        |        30.1 gso/s, 0 ulp |        32.3 gso/s, 0 ulp |        33.5 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e3m2_haswell`     |        28.5 gso/s, 0 ulp |        32.6 gso/s, 0 ulp |        66.1 gso/s, 0 ulp |
| `nk_angulars_packed_e3m2_skylake`          |        37.4 gso/s, 0 ulp |        41.4 gso/s, 0 ulp |        44.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_skylake`       |          39 gso/s, 0 ulp |        41.9 gso/s, 0 ulp |        87.3 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_skylake`        |        35.7 gso/s, 0 ulp |        41.3 gso/s, 0 ulp |          43 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e3m2_skylake`     |        36.2 gso/s, 0 ulp |        36.4 gso/s, 0 ulp |        87.8 gso/s, 0 ulp |
| `nk_angulars_packed_e3m2_genoa`            |          48 gso/s, 0 ulp |          56 gso/s, 0 ulp |        59.3 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_genoa`         |          40 gso/s, 0 ulp |        40.8 gso/s, 0 ulp |        87.4 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_genoa`          |        49.8 gso/s, 0 ulp |        58.4 gso/s, 0 ulp |          61 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e3m2_genoa`       |        38.4 gso/s, 0 ulp |        41.6 gso/s, 0 ulp |        87.7 gso/s, 0 ulp |
| `nk_angulars_packed_e3m2_sapphireamx`      |         238 gso/s, 0 ulp |         420 gso/s, 0 ulp |         431 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_sapphireamx`   |        60.7 gso/s, 0 ulp |        96.5 gso/s, 0 ulp |        90.9 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_sapphireamx`    |         224 gso/s, 0 ulp |         426 gso/s, 0 ulp |         443 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e3m2_sapphireamx` |        60.8 gso/s, 0 ulp |        99.2 gso/s, 0 ulp |        92.6 gso/s, 0 ulp |
| __e2m3__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_serial`           |       0.332 gso/s, 0 ulp |       0.325 gso/s, 0 ulp |       0.320 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_serial`        |       0.298 gso/s, 0 ulp |       0.305 gso/s, 0 ulp |       0.568 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_serial`         |     0.324 gso/s, 0.5 ulp |     0.310 gso/s, 0.5 ulp |     0.313 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e2m3_serial`      |     0.293 gso/s, 0.5 ulp |     0.295 gso/s, 0.5 ulp |     0.586 gso/s, 0.2 ulp |
| `nk_angulars_packed_e2m3_haswell`          |        54.2 gso/s, 0 ulp |          61 gso/s, 0 ulp |        66.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_haswell`       |        48.2 gso/s, 0 ulp |          60 gso/s, 0 ulp |         128 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_haswell`        |        55.9 gso/s, 0 ulp |        63.4 gso/s, 0 ulp |        64.8 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_haswell`     |        48.6 gso/s, 0 ulp |        62.1 gso/s, 0 ulp |         128 gso/s, 0 ulp |
| `nk_angulars_packed_e2m3_skylake`          |        65.1 gso/s, 0 ulp |        79.4 gso/s, 0 ulp |        85.4 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_skylake`       |        61.7 gso/s, 0 ulp |        81.1 gso/s, 0 ulp |         163 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_skylake`        |        65.1 gso/s, 0 ulp |        80.4 gso/s, 0 ulp |        80.8 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_skylake`     |        60.8 gso/s, 0 ulp |        62.3 gso/s, 0 ulp |         167 gso/s, 0 ulp |
| `nk_angulars_packed_e2m3_genoa`            |        47.7 gso/s, 0 ulp |        55.4 gso/s, 0 ulp |          60 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_genoa`         |        36.4 gso/s, 0 ulp |        41.5 gso/s, 0 ulp |        86.7 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_genoa`          |          50 gso/s, 0 ulp |        59.1 gso/s, 0 ulp |        58.3 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_genoa`       |          38 gso/s, 0 ulp |        42.3 gso/s, 0 ulp |        85.1 gso/s, 0 ulp |
| `nk_angulars_packed_e2m3_sapphireamx`      |         350 gso/s, 0 ulp |         956 gso/s, 0 ulp |       1,020 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_sapphireamx`   |        88.4 gso/s, 0 ulp |         203 gso/s, 0 ulp |         188 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_sapphireamx`    |         337 gso/s, 0 ulp |         990 gso/s, 0 ulp |         992 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_sapphireamx` |        88.7 gso/s, 0 ulp |         193 gso/s, 0 ulp |         201 gso/s, 0 ulp |
| __i8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_serial`             |        8.84 gso/s, 0 ulp |        9.49 gso/s, 0 ulp |        10.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_serial`          |        4.40 gso/s, 0 ulp |        4.45 gso/s, 0 ulp |        9.58 gso/s, ? ulp |
| `nk_euclideans_packed_i8_serial`           |      8.64 gso/s, 0.4 ulp |      9.84 gso/s, 0.4 ulp |      9.94 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_i8_serial`        |      4.47 gso/s, 0.4 ulp |      4.64 gso/s, 0.4 ulp |        9.15 gso/s, ? ulp |
| `nk_angulars_packed_i8_haswell`            |        79.5 gso/s, 0 ulp |         102 gso/s, 0 ulp |         109 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_haswell`         |        60.6 gso/s, 0 ulp |        77.4 gso/s, 0 ulp |         168 gso/s, ? ulp |
| `nk_euclideans_packed_i8_haswell`          |        82.5 gso/s, 0 ulp |         102 gso/s, 0 ulp |         109 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_haswell`       |          62 gso/s, 0 ulp |        76.5 gso/s, 0 ulp |         166 gso/s, ? ulp |
| `nk_angulars_packed_i8_icelake`            |         155 gso/s, 0 ulp |         206 gso/s, 0 ulp |         402 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_icelake`         |         103 gso/s, 0 ulp |         263 gso/s, 0 ulp |         690 gso/s, ? ulp |
| `nk_euclideans_packed_i8_icelake`          |         169 gso/s, 0 ulp |         313 gso/s, 0 ulp |         393 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_icelake`       |         108 gso/s, 0 ulp |         268 gso/s, 0 ulp |         695 gso/s, ? ulp |
| `nk_angulars_packed_i8_sapphireamx`        |         427 gso/s, 0 ulp |       1,020 gso/s, 0 ulp |       1,170 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_sapphireamx`     |         106 gso/s, 0 ulp |         261 gso/s, 0 ulp |         210 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_sapphireamx`      |         428 gso/s, 0 ulp |       1,240 gso/s, 0 ulp |       1,170 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_sapphireamx`   |         104 gso/s, 0 ulp |         243 gso/s, 0 ulp |         219 gso/s, 0 ulp |
| __u8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`             |      12.2 gso/s, 0.3 ulp |      12.8 gso/s, 0.3 ulp |      13.0 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_serial`          |      4.48 gso/s, 0.3 ulp |      4.73 gso/s, 0.3 ulp |        9.50 gso/s, ? ulp |
| `nk_euclideans_packed_u8_serial`           |      12.0 gso/s, 0.5 ulp |      13.1 gso/s, 0.5 ulp |      13.4 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_u8_serial`        |      4.52 gso/s, 0.5 ulp |      4.69 gso/s, 0.5 ulp |        9.65 gso/s, ? ulp |
| `nk_angulars_packed_u8_haswell`            |      54.6 gso/s, 0.3 ulp |      87.8 gso/s, 0.3 ulp |       104 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_haswell`         |      44.6 gso/s, 0.3 ulp |      70.2 gso/s, 0.3 ulp |         161 gso/s, ? ulp |
| `nk_euclideans_packed_u8_haswell`          |      55.5 gso/s, 0.5 ulp |      87.7 gso/s, 0.5 ulp |       105 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_u8_haswell`       |      45.3 gso/s, 0.5 ulp |      68.4 gso/s, 0.5 ulp |         159 gso/s, ? ulp |
| `nk_angulars_packed_u8_icelake`            |       154 gso/s, 0.3 ulp |       301 gso/s, 0.3 ulp |       404 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_icelake`         |       108 gso/s, 0.3 ulp |       267 gso/s, 0.3 ulp |         699 gso/s, ? ulp |
| `nk_euclideans_packed_u8_icelake`          |         168 gso/s, 0 ulp |         300 gso/s, 0 ulp |         402 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u8_icelake`       |         109 gso/s, 0 ulp |         253 gso/s, 0 ulp |         695 gso/s, ? ulp |
| `nk_angulars_packed_u8_sapphireamx`        |       444 gso/s, 0.2 ulp |     1,210 gso/s, 0.2 ulp |     1,220 gso/s, 0.2 ulp |
| `nk_angulars_symmetric_u8_sapphireamx`     |       103 gso/s, 0.2 ulp |       257 gso/s, 0.2 ulp |       227 gso/s, 0.2 ulp |
| `nk_euclideans_packed_u8_sapphireamx`      |         432 gso/s, 0 ulp |       1,240 gso/s, 0 ulp |       1,200 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u8_sapphireamx`   |         102 gso/s, 0 ulp |         256 gso/s, 0 ulp |         220 gso/s, 0 ulp |
| __i4__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i4_serial`             |        3.79 gso/s, ? ulp |        3.83 gso/s, ? ulp |        4.06 gso/s, ? ulp |
| `nk_angulars_symmetric_i4_serial`          |        3.52 gso/s, ? ulp |        3.58 gso/s, ? ulp |        7.08 gso/s, ? ulp |
| `nk_euclideans_packed_i4_serial`           |        3.69 gso/s, ? ulp |        3.91 gso/s, ? ulp |        3.76 gso/s, ? ulp |
| `nk_euclideans_symmetric_i4_serial`        |        3.45 gso/s, ? ulp |        3.64 gso/s, ? ulp |        6.99 gso/s, ? ulp |
| `nk_angulars_packed_i4_icelake`            |         117 gso/s, ? ulp |         208 gso/s, ? ulp |         249 gso/s, ? ulp |
| `nk_angulars_symmetric_i4_icelake`         |         103 gso/s, ? ulp |         233 gso/s, ? ulp |         561 gso/s, ? ulp |
| `nk_euclideans_packed_i4_icelake`          |         121 gso/s, ? ulp |         173 gso/s, ? ulp |         246 gso/s, ? ulp |
| `nk_euclideans_symmetric_i4_icelake`       |         101 gso/s, ? ulp |         228 gso/s, ? ulp |         572 gso/s, ? ulp |
| __u4__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u4_serial`             |        5.49 gso/s, ? ulp |        5.60 gso/s, ? ulp |        5.78 gso/s, ? ulp |
| `nk_angulars_symmetric_u4_serial`          |        5.18 gso/s, ? ulp |        5.57 gso/s, ? ulp |        11.5 gso/s, ? ulp |
| `nk_euclideans_packed_u4_serial`           |        5.23 gso/s, ? ulp |        5.50 gso/s, ? ulp |        5.64 gso/s, ? ulp |
| `nk_euclideans_symmetric_u4_serial`        |        5.22 gso/s, ? ulp |        5.47 gso/s, ? ulp |        11.1 gso/s, ? ulp |
| `nk_angulars_packed_u4_icelake`            |         153 gso/s, ? ulp |         270 gso/s, ? ulp |         381 gso/s, ? ulp |
| `nk_angulars_symmetric_u4_icelake`         |         122 gso/s, ? ulp |         264 gso/s, ? ulp |         658 gso/s, ? ulp |
| `nk_euclideans_packed_u4_icelake`          |         158 gso/s, ? ulp |         285 gso/s, ? ulp |         385 gso/s, ? ulp |
| `nk_euclideans_symmetric_u4_icelake`       |         120 gso/s, ? ulp |         279 gso/s, ? ulp |         624 gso/s, ? ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                                     |                     256³ |                    1024³ |                    4096³ |
| :----------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f64_serial`            |        1.38 gso/s, 0 ulp |        1.37 gso/s, 0 ulp |        1.36 gso/s, 0 ulp |
| `nk_angulars_symmetric_f64_serial`         |       0.267 gso/s, 0 ulp |       0.268 gso/s, 0 ulp |       0.258 gso/s, 0 ulp |
| `nk_euclideans_packed_f64_serial`          |      1.41 gso/s, 0.6 ulp |      1.37 gso/s, 0.6 ulp |      1.36 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_f64_serial`       |     0.272 gso/s, 0.6 ulp |     0.271 gso/s, 0.5 ulp |     0.161 gso/s, 0.5 ulp |
| `nk_angulars_packed_f64_v128relaxed`       |      10.9 gso/s, 0.1 ulp |      10.9 gso/s, 0.1 ulp |      10.9 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f64_v128relaxed`    |     0.238 gso/s, 0.1 ulp |     0.240 gso/s, 0.1 ulp |     0.271 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f64_v128relaxed`     |      11.0 gso/s, 0.6 ulp |      11.2 gso/s, 0.6 ulp |      11.2 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_f64_v128relaxed`  |    0.0463 gso/s, 0.6 ulp |    0.0465 gso/s, 0.5 ulp |   0.00806 gso/s, 0.5 ulp |
| __f32__                                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f32_serial`            |      4.16 gso/s, 0.1 ulp |      4.26 gso/s, 0.1 ulp |      4.39 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f32_serial`         |      3.08 gso/s, 0.1 ulp |      4.88 gso/s, 0.1 ulp |      5.69 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f32_serial`          |      4.19 gso/s, 0.6 ulp |      4.32 gso/s, 0.6 ulp |      4.33 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_f32_serial`       |      3.05 gso/s, 0.5 ulp |      4.97 gso/s, 0.5 ulp |      5.64 gso/s, 0.5 ulp |
| `nk_angulars_packed_f32_v128relaxed`       |      9.41 gso/s, 0.1 ulp |      10.6 gso/s, 0.1 ulp |      10.7 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f32_v128relaxed`    |      3.64 gso/s, 0.1 ulp |      6.14 gso/s, 0.1 ulp |      7.33 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f32_v128relaxed`     |      9.55 gso/s, 0.2 ulp |      10.6 gso/s, 0.2 ulp |      10.6 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_f32_v128relaxed`  |      3.55 gso/s, 0.2 ulp |      6.15 gso/s, 0.2 ulp |      7.27 gso/s, 0.2 ulp |
| __bf16__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_bf16_serial`           |        4.10 gso/s, 0 ulp |      4.33 gso/s, 0.2 ulp |      4.45 gso/s, 0.6 ulp |
| `nk_angulars_symmetric_bf16_serial`        |        3.74 gso/s, 0 ulp |      6.15 gso/s, 0.2 ulp |      7.39 gso/s, 0.6 ulp |
| `nk_euclideans_packed_bf16_serial`         |      4.26 gso/s, 0.7 ulp |      4.35 gso/s, 6.1 ulp |       4.40 gso/s, 32 ulp |
| `nk_euclideans_symmetric_bf16_serial`      |      3.80 gso/s, 0.6 ulp |      6.16 gso/s, 5.3 ulp |       7.40 gso/s, 28 ulp |
| `nk_angulars_packed_bf16_v128relaxed`      |        22.0 gso/s, 0 ulp |      24.8 gso/s, 0.2 ulp |      24.7 gso/s, 0.6 ulp |
| `nk_angulars_symmetric_bf16_v128relaxed`   |        4.78 gso/s, 0 ulp |      9.61 gso/s, 0.2 ulp |      12.5 gso/s, 0.6 ulp |
| `nk_euclideans_packed_bf16_v128relaxed`    |      22.2 gso/s, 0.7 ulp |      24.1 gso/s, 6.1 ulp |       24.8 gso/s, 32 ulp |
| `nk_euclideans_symmetric_bf16_v128relaxed` |      4.72 gso/s, 0.3 ulp |      9.53 gso/s, 5.1 ulp |       12.4 gso/s, 28 ulp |
| __e2m3__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_serial`           |        2.66 gso/s, 0 ulp |        2.71 gso/s, 0 ulp |        2.63 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_serial`        |      0.0400 gso/s, 0 ulp |      0.0413 gso/s, 0 ulp |       0.238 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_serial`         |      2.74 gso/s, 0.5 ulp |      2.70 gso/s, 0.5 ulp |      2.67 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_e2m3_serial`      |    0.0403 gso/s, 0.5 ulp |    0.0411 gso/s, 0.4 ulp |    0.0401 gso/s, 0.4 ulp |
| `nk_angulars_packed_e2m3_v128relaxed`      |        18.4 gso/s, 0 ulp |        18.6 gso/s, 0 ulp |        18.5 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_v128relaxed`   |      0.0559 gso/s, 0 ulp |      0.0180 gso/s, 0 ulp |       0.131 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_v128relaxed`    |        18.5 gso/s, 0 ulp |        18.7 gso/s, 0 ulp |        18.1 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_v128relaxed` |       0.206 gso/s, 0 ulp |      0.0170 gso/s, 0 ulp |      0.0554 gso/s, 0 ulp |
| __i8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_serial`             |        4.73 gso/s, 0 ulp |        4.81 gso/s, 0 ulp |        4.59 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_serial`          |     0.00447 gso/s, 0 ulp |       0.198 gso/s, 0 ulp |       0.190 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_serial`           |      4.77 gso/s, 0.5 ulp |      4.80 gso/s, 0.4 ulp |      4.65 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_i8_serial`        |     0.201 gso/s, 0.5 ulp |    0.0819 gso/s, 0.4 ulp |    0.0823 gso/s, 0.4 ulp |
| `nk_angulars_packed_i8_v128relaxed`        |        31.6 gso/s, 0 ulp |        31.7 gso/s, 0 ulp |        31.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_v128relaxed`     |      0.0304 gso/s, 0 ulp |      0.0680 gso/s, 0 ulp |       0.298 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_v128relaxed`      |        31.5 gso/s, 0 ulp |        32.3 gso/s, 0 ulp |        30.8 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_v128relaxed`   |       0.224 gso/s, 0 ulp |       0.222 gso/s, 0 ulp |       0.143 gso/s, 0 ulp |
| __u8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`             |      4.26 gso/s, 0.4 ulp |      5.07 gso/s, 0.3 ulp |      5.11 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_serial`          |      2.64 gso/s, 0.4 ulp |      4.02 gso/s, 0.3 ulp |      4.34 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u8_serial`           |      4.35 gso/s, 0.5 ulp |      4.67 gso/s, 0.5 ulp |      5.09 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_u8_serial`        |      2.64 gso/s, 0.5 ulp |      3.97 gso/s, 0.5 ulp |      4.38 gso/s, 0.5 ulp |
| `nk_angulars_packed_u8_v128relaxed`        |      23.7 gso/s, 0.3 ulp |      25.1 gso/s, 0.3 ulp |      25.8 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_v128relaxed`     |      19.6 gso/s, 0.3 ulp |      23.2 gso/s, 0.3 ulp |      24.1 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u8_v128relaxed`      |        23.8 gso/s, 0 ulp |        25.3 gso/s, 0 ulp |        25.8 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u8_v128relaxed`   |        19.5 gso/s, 0 ulp |        23.0 gso/s, 0 ulp |        24.6 gso/s, 0 ulp |
| __i4__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i4_serial`             |     6.22 gso/s, 0.35 ulp |     6.41 gso/s, 0.34 ulp |     6.55 gso/s, 0.35 ulp |
| `nk_angulars_symmetric_i4_serial`          |     2.64 gso/s, 0.34 ulp |     3.69 gso/s, 0.34 ulp |     4.18 gso/s, 0.34 ulp |
| `nk_euclideans_packed_i4_serial`           |     6.00 gso/s, 0.49 ulp |     6.43 gso/s, 0.54 ulp |     6.56 gso/s, 0.64 ulp |
| `nk_euclideans_symmetric_i4_serial`        |     2.61 gso/s, 0.48 ulp |     3.68 gso/s, 0.53 ulp |     4.14 gso/s, 0.63 ulp |
| __u4__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u4_serial`             |     5.38 gso/s, 0.35 ulp |     5.60 gso/s, 0.34 ulp |     5.81 gso/s, 0.35 ulp |
| `nk_angulars_symmetric_u4_serial`          |     2.90 gso/s, 0.34 ulp |     4.28 gso/s, 0.34 ulp |     4.90 gso/s, 0.34 ulp |
| `nk_euclideans_packed_u4_serial`           |     5.25 gso/s, 0.49 ulp |     5.64 gso/s, 0.54 ulp |     5.82 gso/s, 0.64 ulp |
| `nk_euclideans_symmetric_u4_serial`        |     2.89 gso/s, 0.48 ulp |     4.30 gso/s, 0.53 ulp |     4.86 gso/s, 0.63 ulp |

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
| `nk_angulars_packed_bf16_sme`            |      400 gso/s, 0.04 ulp |      821 gso/s, 0.04 ulp |    1,082 gso/s, 0.04 ulp |
| `nk_angulars_symmetric_bf16_sme`         |      218 gso/s, 0.03 ulp |      464 gso/s, 0.03 ulp |      442 gso/s, 0.03 ulp |
| `nk_euclideans_packed_bf16_sme`          |      468 gso/s, 0.54 ulp |      886 gso/s, 0.54 ulp |    1,109 gso/s, 0.54 ulp |
| `nk_euclideans_symmetric_bf16_sme`       |      207 gso/s, 0.28 ulp |      473 gso/s, 0.28 ulp |      445 gso/s, 0.28 ulp |
| __f16__                                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_f16_serial`          |      12.8 gso/s, 0.1 ulp |      14.4 gso/s, 0.1 ulp |      14.9 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_serial`       |      21.7 gso/s, 0.1 ulp |     25.2 gso/s, 0.09 ulp |      28.2 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_serial`        |      13.1 gso/s, 1.1 ulp |      13.9 gso/s, 0.7 ulp |      15.7 gso/s, 5.6 ulp |
| `nk_euclideans_symmetric_f16_serial`     |      23.6 gso/s, 1.1 ulp |      25.2 gso/s, 0.7 ulp |      28.4 gso/s, 5.6 ulp |
| `nk_angulars_packed_f16_neonhalf`        |      72.2 gso/s, 0.1 ulp |      78.6 gso/s, 0.1 ulp |      83.8 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_neonhalf`     |      19.3 gso/s, 0.1 ulp |      20.9 gso/s, 0.1 ulp |      21.8 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_neonhalf`      |      73.0 gso/s, 0.9 ulp |      76.2 gso/s, 0.7 ulp |      83.7 gso/s, 5.9 ulp |
| `nk_euclideans_symmetric_f16_neonhalf`   |      19.2 gso/s, 0.9 ulp |      20.2 gso/s, 0.6 ulp |      21.9 gso/s, 5.8 ulp |
| `nk_angulars_packed_f16_neonfhm`         |      96.2 gso/s, 0.1 ulp |       107 gso/s, 0.1 ulp |       118 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_neonfhm`      |      35.4 gso/s, 0.1 ulp |      39.1 gso/s, 0.1 ulp |      42.5 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_neonfhm`       |       100 gso/s, 0.9 ulp |       110 gso/s, 0.7 ulp |       119 gso/s, 5.9 ulp |
| `nk_euclideans_symmetric_f16_neonfhm`    |      37.2 gso/s, 0.9 ulp |      39.4 gso/s, 0.6 ulp |      42.0 gso/s, 5.8 ulp |
| `nk_angulars_packed_f16_sme`             |       419 gso/s, 0.1 ulp |       839 gso/s, 0.1 ulp |     1,091 gso/s, 0.1 ulp |
| `nk_angulars_symmetric_f16_sme`          |       241 gso/s, 0.1 ulp |       487 gso/s, 0.1 ulp |       450 gso/s, 0.1 ulp |
| `nk_euclideans_packed_f16_sme`           |         491 gso/s, 0 ulp |      906 gso/s, 0.06 ulp |     1,118 gso/s, 2.9 ulp |
| `nk_euclideans_symmetric_f16_sme`        |       227 gso/s, 0.3 ulp |       500 gso/s, 0.6 ulp |       451 gso/s, 0.3 ulp |
| __e5m2__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e5m2_serial`         |        15.8 gso/s, 0 ulp |        16.7 gso/s, 0 ulp |        17.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_serial`      |        7.78 gso/s, 0 ulp |        8.37 gso/s, 0 ulp |        8.99 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_serial`       |      15.5 gso/s, 0.5 ulp |      16.7 gso/s, 0.5 ulp |      17.2 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_e5m2_serial`    |      7.93 gso/s, 0.5 ulp |      8.37 gso/s, 0.5 ulp |      8.99 gso/s, 0.5 ulp |
| `nk_angulars_packed_e5m2_neonfhm`        |        84.3 gso/s, 0 ulp |        97.3 gso/s, 0 ulp |         103 gso/s, 0 ulp |
| `nk_angulars_symmetric_e5m2_neonfhm`     |        58.8 gso/s, 0 ulp |        73.2 gso/s, 0 ulp |        79.3 gso/s, 0 ulp |
| `nk_euclideans_packed_e5m2_neonfhm`      |        88.1 gso/s, 0 ulp |         110 gso/s, 0 ulp |         119 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e5m2_neonfhm`   |        66.1 gso/s, 0 ulp |        60.3 gso/s, 0 ulp |        64.4 gso/s, 0 ulp |
| `nk_angulars_packed_e5m2_sme`            |      350 gso/s, 0.01 ulp |      609 gso/s, 0.01 ulp |      744 gso/s, 0.01 ulp |
| `nk_angulars_symmetric_e5m2_sme`         |      138 gso/s, 0.01 ulp |      204 gso/s, 0.01 ulp |      226 gso/s, 0.01 ulp |
| `nk_euclideans_packed_e5m2_sme`          |     399 gso/s, 0.005 ulp |     655 gso/s, 0.005 ulp |     762 gso/s, 0.005 ulp |
| `nk_euclideans_symmetric_e5m2_sme`       |     132 gso/s, 0.004 ulp |     206 gso/s, 0.004 ulp |     227 gso/s, 0.004 ulp |
| __e4m3__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e4m3_serial`         |        1.15 gso/s, 0 ulp |        1.20 gso/s, 0 ulp |        1.24 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_serial`      |     1.22 gso/s, 0.03 ulp |     1.24 gso/s, 0.02 ulp |     1.32 gso/s, 0.01 ulp |
| `nk_euclideans_packed_e4m3_serial`       |      1.23 gso/s, 0.5 ulp |      1.20 gso/s, 0.5 ulp |      1.24 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_e4m3_serial`    |      1.25 gso/s, 0.5 ulp |      1.24 gso/s, 0.5 ulp |      1.32 gso/s, 0.3 ulp |
| `nk_angulars_packed_e4m3_neonfhm`        |        29.1 gso/s, 0 ulp |        32.2 gso/s, 0 ulp |        34.1 gso/s, 0 ulp |
| `nk_angulars_symmetric_e4m3_neonfhm`     |        32.0 gso/s, 0 ulp |        36.6 gso/s, 0 ulp |        38.9 gso/s, 0 ulp |
| `nk_euclideans_packed_e4m3_neonfhm`      |        30.0 gso/s, 0 ulp |        32.2 gso/s, 0 ulp |      34.1 gso/s, 0.2 ulp |
| `nk_euclideans_symmetric_e4m3_neonfhm`   |        34.1 gso/s, 0 ulp |        36.6 gso/s, 0 ulp |      38.9 gso/s, 0.2 ulp |
| `nk_angulars_packed_e4m3_sme`            |      184 gso/s, 0.01 ulp |      272 gso/s, 0.01 ulp |      307 gso/s, 0.01 ulp |
| `nk_angulars_symmetric_e4m3_sme`         |     56.4 gso/s, 0.01 ulp |     74.6 gso/s, 0.01 ulp |     78.5 gso/s, 0.01 ulp |
| `nk_euclideans_packed_e4m3_sme`          |      200 gso/s, 0.11 ulp |      279 gso/s, 0.11 ulp |      310 gso/s, 0.11 ulp |
| `nk_euclideans_symmetric_e4m3_sme`       |     55.5 gso/s, 0.11 ulp |     75.1 gso/s, 0.11 ulp |     78.4 gso/s, 0.11 ulp |
| __e3m2__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e3m2_serial`         |        14.2 gso/s, 0 ulp |        14.6 gso/s, 0 ulp |        15.5 gso/s, 0 ulp |
| `nk_angulars_symmetric_e3m2_serial`      |        7.77 gso/s, 0 ulp |        8.10 gso/s, 0 ulp |        9.05 gso/s, 0 ulp |
| `nk_euclideans_packed_e3m2_serial`       |      13.9 gso/s, 0.5 ulp |      14.6 gso/s, 0.5 ulp |      15.5 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_e3m2_serial`    |      8.08 gso/s, 0.5 ulp |      8.10 gso/s, 0.5 ulp |      9.05 gso/s, 0.5 ulp |
| `nk_angulars_packed_e3m2_sme`            |      327 gso/s, 0.01 ulp |      573 gso/s, 0.01 ulp |      690 gso/s, 0.01 ulp |
| `nk_angulars_symmetric_e3m2_sme`         |      124 gso/s, 0.01 ulp |      184 gso/s, 0.01 ulp |      204 gso/s, 0.01 ulp |
| `nk_euclideans_packed_e3m2_sme`          |         379 gso/s, 0 ulp |         604 gso/s, 0 ulp |         702 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e3m2_sme`       |         119 gso/s, 0 ulp |         186 gso/s, 0 ulp |         205 gso/s, 0 ulp |
| __e2m3__                                 | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_e2m3_serial`         |        14.1 gso/s, 0 ulp |        14.8 gso/s, 0 ulp |        15.5 gso/s, 0 ulp |
| `nk_angulars_symmetric_e2m3_serial`      |        7.89 gso/s, 0 ulp |        8.21 gso/s, 0 ulp |        9.09 gso/s, 0 ulp |
| `nk_euclideans_packed_e2m3_serial`       |      13.6 gso/s, 0.5 ulp |      14.8 gso/s, 0.5 ulp |      15.7 gso/s, 0.5 ulp |
| `nk_euclideans_symmetric_e2m3_serial`    |      7.93 gso/s, 0.5 ulp |      8.21 gso/s, 0.5 ulp |      9.09 gso/s, 0.5 ulp |
| `nk_angulars_packed_e2m3_sme`            |      415 gso/s, 0.01 ulp |      926 gso/s, 0.01 ulp |    1,216 gso/s, 0.01 ulp |
| `nk_angulars_symmetric_e2m3_sme`         |      170 gso/s, 0.01 ulp |      342 gso/s, 0.01 ulp |      404 gso/s, 0.01 ulp |
| `nk_euclideans_packed_e2m3_sme`          |         470 gso/s, 0 ulp |       1,011 gso/s, 0 ulp |       1,269 gso/s, 0 ulp |
| `nk_euclideans_symmetric_e2m3_sme`       |         163 gso/s, 0 ulp |         348 gso/s, 0 ulp |         408 gso/s, 0 ulp |
| __i8__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i8_serial`           |        18.3 gso/s, 0 ulp |        20.0 gso/s, 0 ulp |        20.2 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_serial`        |        13.5 gso/s, 0 ulp |        13.9 gso/s, 0 ulp |        14.8 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_serial`         |      18.7 gso/s, 0.4 ulp |      20.0 gso/s, 0.4 ulp |      20.2 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_i8_serial`      |      13.7 gso/s, 0.4 ulp |      13.9 gso/s, 0.4 ulp |      14.8 gso/s, 0.4 ulp |
| `nk_angulars_packed_i8_neonsdot`         |         280 gso/s, 0 ulp |         357 gso/s, 0 ulp |         477 gso/s, 0 ulp |
| `nk_angulars_symmetric_i8_neonsdot`      |        74.0 gso/s, 0 ulp |        86.9 gso/s, 0 ulp |        87.2 gso/s, 0 ulp |
| `nk_euclideans_packed_i8_neonsdot`       |         305 gso/s, 0 ulp |         419 gso/s, 0 ulp |         477 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_neonsdot`    |        73.4 gso/s, 0 ulp |        87.0 gso/s, 0 ulp |        87.2 gso/s, 0 ulp |
| `nk_angulars_packed_i8_sme`              |      492 gso/s, 0.01 ulp |    1,356 gso/s, 0.01 ulp |    2,166 gso/s, 0.01 ulp |
| `nk_angulars_symmetric_i8_sme`           |      200 gso/s, 0.01 ulp |      873 gso/s, 0.01 ulp |    1,214 gso/s, 0.01 ulp |
| `nk_euclideans_packed_i8_sme`            |         584 gso/s, 0 ulp |       1,546 gso/s, 0 ulp |       2,263 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i8_sme`         |         201 gso/s, 0 ulp |         917 gso/s, 0 ulp |       1,256 gso/s, 0 ulp |
| __u8__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`           |      15.5 gso/s, 0.3 ulp |      16.3 gso/s, 0.3 ulp |      17.4 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_serial`        |      15.7 gso/s, 0.3 ulp |      16.2 gso/s, 0.3 ulp |      17.5 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u8_serial`         |      16.4 gso/s, 0.5 ulp |      16.3 gso/s, 0.5 ulp |      17.4 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_u8_serial`      |      16.1 gso/s, 0.5 ulp |      16.2 gso/s, 0.5 ulp |      17.5 gso/s, 0.6 ulp |
| `nk_angulars_packed_u8_neonsdot`         |       284 gso/s, 0.3 ulp |       369 gso/s, 0.3 ulp |       470 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_neonsdot`      |      72.4 gso/s, 0.3 ulp |      87.4 gso/s, 0.3 ulp |      87.7 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u8_neonsdot`       |         302 gso/s, 0 ulp |         419 gso/s, 0 ulp |         470 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u8_neonsdot`    |        72.0 gso/s, 0 ulp |        87.0 gso/s, 0 ulp |        87.7 gso/s, 0 ulp |
| `nk_angulars_packed_u8_sme`              |      492 gso/s, 0.32 ulp |    1,369 gso/s, 0.32 ulp |    2,169 gso/s, 0.32 ulp |
| `nk_angulars_symmetric_u8_sme`           |      201 gso/s, 0.32 ulp |      874 gso/s, 0.32 ulp |    1,217 gso/s, 0.32 ulp |
| `nk_euclideans_packed_u8_sme`            |         584 gso/s, 0 ulp |       1,545 gso/s, 0 ulp |       2,260 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u8_sme`         |         199 gso/s, 0 ulp |         905 gso/s, 0 ulp |       1,248 gso/s, 0 ulp |
| __i4__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_i4_serial`           |      17.3 gso/s, 0.3 ulp |      18.2 gso/s, 0.3 ulp |      19.6 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_i4_serial`        |      14.3 gso/s, 0.3 ulp |      14.9 gso/s, 0.3 ulp |      15.6 gso/s, 0.3 ulp |
| `nk_euclideans_packed_i4_serial`         |      17.9 gso/s, 0.5 ulp |      18.2 gso/s, 0.5 ulp |      19.6 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_i4_serial`      |      14.6 gso/s, 0.5 ulp |      14.9 gso/s, 0.5 ulp |      15.6 gso/s, 0.6 ulp |
| `nk_angulars_packed_i4_neonsdot`         |       215 gso/s, 0.3 ulp |       284 gso/s, 0.3 ulp |       291 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_i4_neonsdot`      |       104 gso/s, 0.3 ulp |       162 gso/s, 0.3 ulp |       171 gso/s, 0.3 ulp |
| `nk_euclideans_packed_i4_neonsdot`       |         225 gso/s, 0 ulp |         284 gso/s, 0 ulp |         291 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i4_neonsdot`    |         105 gso/s, 0 ulp |         162 gso/s, 0 ulp |         171 gso/s, 0 ulp |
| `nk_angulars_packed_i4_sme`              |      486 gso/s, 0.32 ulp |    1,309 gso/s, 0.32 ulp |    2,041 gso/s, 0.32 ulp |
| `nk_angulars_symmetric_i4_sme`           |      201 gso/s, 0.32 ulp |      913 gso/s, 0.32 ulp |    1,488 gso/s, 0.32 ulp |
| `nk_euclideans_packed_i4_sme`            |         576 gso/s, 0 ulp |       1,453 gso/s, 0 ulp |       2,126 gso/s, 0 ulp |
| `nk_euclideans_symmetric_i4_sme`         |         200 gso/s, 0 ulp |         948 gso/s, 0 ulp |       1,527 gso/s, 0 ulp |
| __u4__                                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u4_serial`           |      18.0 gso/s, 0.3 ulp |      19.4 gso/s, 0.3 ulp |      20.6 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u4_serial`        |      15.5 gso/s, 0.3 ulp |      16.4 gso/s, 0.3 ulp |      17.4 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u4_serial`         |      19.1 gso/s, 0.5 ulp |      19.4 gso/s, 0.5 ulp |      20.6 gso/s, 0.6 ulp |
| `nk_euclideans_symmetric_u4_serial`      |      15.7 gso/s, 0.5 ulp |      16.4 gso/s, 0.5 ulp |      17.4 gso/s, 0.6 ulp |
| `nk_angulars_packed_u4_neonsdot`         |       241 gso/s, 0.3 ulp |       319 gso/s, 0.3 ulp |       340 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u4_neonsdot`      |       107 gso/s, 0.3 ulp |       166 gso/s, 0.3 ulp |       173 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u4_neonsdot`       |         250 gso/s, 0 ulp |         340 gso/s, 0 ulp |         340 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u4_neonsdot`    |         105 gso/s, 0 ulp |         173 gso/s, 0 ulp |         173 gso/s, 0 ulp |
| `nk_angulars_packed_u4_sme`              |      490 gso/s, 0.32 ulp |    1,322 gso/s, 0.32 ulp |    2,081 gso/s, 0.32 ulp |
| `nk_angulars_symmetric_u4_sme`           |      205 gso/s, 0.32 ulp |      974 gso/s, 0.32 ulp |    1,682 gso/s, 0.32 ulp |
| `nk_euclideans_packed_u4_sme`            |         582 gso/s, 0 ulp |       1,487 gso/s, 0 ulp |       2,162 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u4_sme`         |         205 gso/s, 0 ulp |       1,013 gso/s, 0 ulp |       1,734 gso/s, 0 ulp |

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
| __u8__                                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_angulars_packed_u8_serial`             |      14.7 gso/s, 0.3 ulp |      17.0 gso/s, 0.3 ulp |      17.8 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_serial`          |      10.9 gso/s, 0.3 ulp |      13.2 gso/s, 0.3 ulp |      13.9 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u8_serial`           |      14.9 gso/s, 0.4 ulp |      17.0 gso/s, 0.4 ulp |      17.8 gso/s, 0.4 ulp |
| `nk_euclideans_symmetric_u8_serial`        |      11.7 gso/s, 0.4 ulp |      13.1 gso/s, 0.4 ulp |      13.9 gso/s, 0.4 ulp |
| `nk_angulars_packed_u8_v128relaxed`        |      43.7 gso/s, 0.3 ulp |      49.0 gso/s, 0.3 ulp |      50.7 gso/s, 0.3 ulp |
| `nk_angulars_symmetric_u8_v128relaxed`     |      34.7 gso/s, 0.3 ulp |      45.6 gso/s, 0.3 ulp |      48.4 gso/s, 0.3 ulp |
| `nk_euclideans_packed_u8_v128relaxed`      |        44.8 gso/s, 0 ulp |        49.0 gso/s, 0 ulp |        50.7 gso/s, 0 ulp |
| `nk_euclideans_symmetric_u8_v128relaxed`   |        35.0 gso/s, 0 ulp |        44.2 gso/s, 0 ulp |        48.5 gso/s, 0 ulp |
