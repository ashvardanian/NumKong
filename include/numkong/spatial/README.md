# Spatial Distances in NumKong

NumKong implements spatial distance functions for dense vectors: squared Euclidean distance, Euclidean distance, and angular (cosine) distance.
These metrics are commonly used in nearest-neighbor search, clustering, and dimensionality reduction, and are implemented for every numeric type supported by the library.

Squared Euclidean distance measures the sum of squared element-wise differences:

$$
\text{sqeuclidean}(a, b) = \sum_{i=0}^{n-1} (a_i - b_i)^2
$$

Euclidean distance is the square root of the squared Euclidean distance:

$$
\text{euclidean}(a, b) = \sqrt{\sum_{i=0}^{n-1} (a_i - b_i)^2}
$$

Angular distance (cosine distance) measures the angle between two vectors:

$$
\text{angular}(a, b) = 1 - \frac{\sum_{i=0}^{n-1} a_i \cdot b_i}{\sqrt{\sum_{i=0}^{n-1} a_i^2} \cdot \sqrt{\sum_{i=0}^{n-1} b_i^2}}
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def sqeuclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.sum((a - b) ** 2)

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2))

def angular(a: np.ndarray, b: np.ndarray) -> float:
    ab = np.dot(a, b)
    a2 = np.dot(a, a)
    b2 = np.dot(b, b)
    if a2 == 0 and b2 == 0: return 0
    if ab == 0: return 1
    return 1 - ab / (np.sqrt(a2) * np.sqrt(b2))
```

## Input & Output Types

| Input Type | Output Type | Description                                      |
| ---------- | ----------- | ------------------------------------------------ |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision                 |
| `f32`      | `f64`       | 32-bit IEEE 754 single precision, widened output |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output   |
| `bf16`     | `f32`       | 16-bit brain float, widened output               |
| `e5m2`     | `f32`       | 8-bit Float8: 5 exponent, 2 mantissa bits        |
| `e4m3`     | `f32`       | 8-bit Float8: 4 exponent, 3 mantissa bits        |
| `e3m2`     | `f32`       | 8-bit MX format: 3 exponent, 2 mantissa bits     |
| `e2m3`     | `f32`       | 8-bit MX format: 2 exponent, 3 mantissa bits     |
| `i8`       | `f32`       | 8-bit signed integers                            |
| `u8`       | `f32`       | 8-bit unsigned integers                          |
| `i4`       | `f32`       | 4-bit signed integers, packed nibble pairs       |
| `u4`       | `f32`       | 4-bit unsigned integers, packed nibble pairs     |

The integer `nk_sqeuclidean_*` kernels are the exception: `i8`, `u8`, `i4`, and `u4` return an exact `u32` count of squared differences, while `nk_euclidean_*` and `nk_angular_*` return `f32`.

## Optimizations

### Three-Accumulator Angular Pattern

`nk_angular_f32_haswell`, `nk_angular_f32_skylake`, `nk_angular_f32_neon` compute cosine distance as $1 - ab / (\sqrt{a^2} \cdot \sqrt{b^2})$, requiring three concurrent dot products in a single pass: $\sum a_i b_i$, $\sum a_i^2$, and $\sum b_i^2$.
All spatial angular kernels interleave these three FMA streams so that each vector element is loaded once and immediately contributes to all three accumulators.
This triples register pressure compared to a plain dot product — on Haswell with 16 YMM registers, three independent 4-register accumulator chains leave only 4 registers for temporaries.
The single-pass design is essential because reading two vectors of length $n$ once costs $2n$ cache line fetches, while a three-pass approach would cost $6n$.

### Reciprocal Square Root with Newton-Raphson Refinement

`nk_angular_f16_haswell`, `nk_angular_i8_haswell`, `nk_angular_e4m3_skylake`, `nk_angular_f32_neon`, `nk_angular_f64_neon` compute the final normalization via in-hardware reciprocal square root estimates refined by Newton-Raphson iteration.
The iteration formula is $x_{n+1} = x_n \cdot (3 - d \cdot x_n^2) / 2$, where $d$ is the value whose reciprocal square root is needed.
NEON `vrsqrte_f32` + `vrsqrts_f32` runs two refinement steps for Float32, and `vrsqrteq_f64` + `vrsqrtsq_f64` runs three for Float64.
Haswell `VRSQRTPS` starts from ~12 bits (1.5 x $2^{-12}$ relative error) and one Newton-Raphson step roughly doubles that; Skylake `VRSQRT14PS` starts from $2^{-14}$ and refines the same way.
The Float32 and Float64 angular kernels on x86 do not take this path at all: they accumulate in Float64 and finish with an exact `VSQRTPD`, because a refined 14-bit estimate is still far short of a 52-bit mantissa.

### Absolute Differences for Integer Types

`nk_sqeuclidean_i8_haswell`, `nk_sqeuclidean_u8_haswell`, `nk_sqeuclidean_i8_icelake`, `nk_sqeuclidean_u8_icelake` compute squared Euclidean distance by first obtaining element-wise absolute differences, then squaring and accumulating.
For signed `i8`, XOR with `0x80` converts the range from [-128, 127] to unsigned [0, 255], then saturating subtract in both directions followed by OR gives $|a - b|$:

```
bias_a = _mm256_xor_si256(a, 0x80)
bias_b = _mm256_xor_si256(b, 0x80)
abs_diff = _mm256_or_si256(_mm256_subs_epu8(bias_a, bias_b), _mm256_subs_epu8(bias_b, bias_a))
```

For unsigned `u8`, the same saturating subtract trick works without the XOR bias.
The absolute differences are then zero-extended via `VPUNPCKLBW`/`VPUNPCKHBW` (1 cycle, cheaper than `VPMOVZXBW`) and squared+accumulated via `VPMADDWD`, which computes $d_i^2 + d_{i+1}^2$ in one instruction.

### Masked Dot2 Compensation on Skylake

`nk_angular_f64_skylake` and `nk_angular_f64_haswell` run the Ogita-Rump-Oishi Dot2 error-free transformation on the cross-product accumulator, which is the only one of the three that can suffer cancellation; the two self-products $\|a\|^2$ and $\|b\|^2$ have no negative terms and use a plain FMA.
Each step recovers the rounding error of both the product and the sum, and folds them into a separate compensation accumulator that is added back once at the end.
Tail elements are handled by a masked load that zero-fills the inactive lanes, so the same compensated body runs on the final partial iteration — no scalar tail loop that would lose the error tracking.
`nk_sqeuclidean_f64_skylake` needs none of this: squared differences are non-negative, so a plain `VFMADD231PD` chain suffices.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_DENSE_DIMENSIONS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.
Accuracy is reported as mean ULP (units in last place) unless noted otherwise — the average number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                        |                      256 |                     1024 |                     4096 |
| :---------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`   |       8.00 gb/s, 0.1 ulp |         8.32 gb/s, 0 ulp |         8.13 gb/s, 0 ulp |
| `nk_euclidean_f64_serial`     |       7.81 gb/s, 0.6 ulp |       7.95 gb/s, 0.5 ulp |       8.34 gb/s, 0.5 ulp |
| `nk_angular_f64_serial`       |         2.80 gb/s, 0 ulp |         3.03 gb/s, 0 ulp |         3.18 gb/s, 0 ulp |
| `nk_sqeuclidean_f64_skylake`  |       32.4 gb/s, 0.4 ulp |       30.6 gb/s, 0.7 ulp |       22.2 gb/s, 1.3 ulp |
| `nk_euclidean_f64_skylake`    |       31.7 gb/s, 0.3 ulp |       29.4 gb/s, 0.4 ulp |       22.9 gb/s, 0.7 ulp |
| `nk_angular_f64_skylake`      |         26.5 gb/s, 0 ulp |         26.8 gb/s, 0 ulp |         17.8 gb/s, 0 ulp |
| __f32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`   |         4.01 gb/s, 0 ulp |         4.06 gb/s, 0 ulp |         4.19 gb/s, 0 ulp |
| `nk_euclidean_f32_serial`     |       3.99 gb/s, 0.1 ulp |       4.07 gb/s, 0.1 ulp |       4.11 gb/s, 0.1 ulp |
| `nk_angular_f32_serial`       |         1.29 gb/s, 0 ulp |         1.41 gb/s, 0 ulp |         1.53 gb/s, 0 ulp |
| `nk_sqeuclidean_f32_skylake`  |         36.5 gb/s, 0 ulp |         27.0 gb/s, 0 ulp |         23.2 gb/s, 0 ulp |
| `nk_euclidean_f32_skylake`    |       36.4 gb/s, 0.1 ulp |       28.1 gb/s, 0.1 ulp |       26.7 gb/s, 0.1 ulp |
| `nk_angular_f32_skylake`      |         24.3 gb/s, 0 ulp |         23.2 gb/s, 0 ulp |         22.5 gb/s, 0 ulp |
| __bf16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`  |        0.582 gb/s, 0 ulp |        0.358 gb/s, 0 ulp |        0.390 gb/s, 0 ulp |
| `nk_euclidean_bf16_serial`    |      0.569 gb/s, 0.5 ulp |      0.373 gb/s, 0.5 ulp |      0.372 gb/s, 0.4 ulp |
| `nk_angular_bf16_serial`      |        0.455 gb/s, 0 ulp |        0.241 gb/s, 0 ulp |        0.259 gb/s, 0 ulp |
| `nk_sqeuclidean_bf16_haswell` |       27.7 gb/s, 0.5 ulp |       14.0 gb/s, 7.5 ulp |        11.8 gb/s, 27 ulp |
| `nk_euclidean_bf16_haswell`   |       23.3 gb/s, 0.3 ulp |       13.4 gb/s, 4.1 ulp |        12.0 gb/s, 15 ulp |
| `nk_angular_bf16_haswell`     |         20.1 gb/s, 0 ulp |         13.4 gb/s, 0 ulp |       10.6 gb/s, 0.2 ulp |
| `nk_sqeuclidean_bf16_genoa`   |       50.1 gb/s, 0.3 ulp |       21.0 gb/s, 0.5 ulp |        20.5 gb/s, 10 ulp |
| `nk_euclidean_bf16_genoa`     |       48.3 gb/s, 0.2 ulp |       23.1 gb/s, 0.3 ulp |       20.4 gb/s, 5.8 ulp |
| `nk_angular_bf16_genoa`       |         36.4 gb/s, 0 ulp |         22.4 gb/s, 0 ulp |       21.0 gb/s, 0.1 ulp |
| __f16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`   |      0.950 gb/s, 0.1 ulp |      0.872 gb/s, 0.1 ulp |      0.864 gb/s, 0.1 ulp |
| `nk_euclidean_f16_serial`     |      0.934 gb/s, 0.5 ulp |      0.913 gb/s, 0.5 ulp |      0.906 gb/s, 0.5 ulp |
| `nk_angular_f16_serial`       |        0.881 gb/s, 0 ulp |        0.531 gb/s, 0 ulp |        0.543 gb/s, 0 ulp |
| `nk_sqeuclidean_f16_haswell`  |       29.8 gb/s, 0.4 ulp |       14.8 gb/s, 1.4 ulp |       11.8 gb/s, 5.2 ulp |
| `nk_euclidean_f16_haswell`    |       22.9 gb/s, 0.3 ulp |       12.9 gb/s, 0.8 ulp |       10.6 gb/s, 2.8 ulp |
| `nk_angular_f16_haswell`      |       19.9 gb/s, 0.1 ulp |       17.5 gb/s, 0.1 ulp |       16.1 gb/s, 0.1 ulp |
| __e5m2__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e5m2_serial`  |        0.955 gb/s, 0 ulp |         1.01 gb/s, 0 ulp |         1.02 gb/s, 0 ulp |
| `nk_euclidean_e5m2_serial`    |      0.954 gb/s, 0.5 ulp |      0.985 gb/s, 0.5 ulp |       1.03 gb/s, 0.5 ulp |
| `nk_angular_e5m2_serial`      |        0.336 gb/s, 0 ulp |        0.385 gb/s, 0 ulp |        0.407 gb/s, 0 ulp |
| `nk_sqeuclidean_e5m2_skylake` |         4.44 gb/s, 0 ulp |         4.65 gb/s, 0 ulp |         5.80 gb/s, 0 ulp |
| `nk_euclidean_e5m2_skylake`   |         4.34 gb/s, 0 ulp |         4.65 gb/s, 0 ulp |         5.88 gb/s, 0 ulp |
| `nk_angular_e5m2_skylake`     |         3.83 gb/s, 0 ulp |         4.39 gb/s, 0 ulp |         6.10 gb/s, 0 ulp |
| `nk_sqeuclidean_e5m2_genoa`   |         7.12 gb/s, 0 ulp |         8.07 gb/s, 0 ulp |         8.05 gb/s, 0 ulp |
| `nk_euclidean_e5m2_genoa`     |         7.01 gb/s, 0 ulp |         6.97 gb/s, 0 ulp |         8.16 gb/s, 0 ulp |
| `nk_angular_e5m2_genoa`       |         6.33 gb/s, 0 ulp |         6.79 gb/s, 0 ulp |         7.99 gb/s, 0 ulp |
| __e4m3__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e4m3_serial`  |        0.569 gb/s, 0 ulp |        0.606 gb/s, 0 ulp |        0.609 gb/s, 0 ulp |
| `nk_euclidean_e4m3_serial`    |      0.587 gb/s, 0.5 ulp |      0.602 gb/s, 0.5 ulp |      0.578 gb/s, 0.5 ulp |
| `nk_angular_e4m3_serial`      |        0.326 gb/s, 0 ulp |        0.196 gb/s, 0 ulp |        0.366 gb/s, 0 ulp |
| `nk_sqeuclidean_e4m3_skylake` |         3.84 gb/s, 0 ulp |         3.62 gb/s, 0 ulp |       3.95 gb/s, 0.2 ulp |
| `nk_euclidean_e4m3_skylake`   |         3.48 gb/s, 0 ulp |         3.69 gb/s, 0 ulp |       3.33 gb/s, 0.2 ulp |
| `nk_angular_e4m3_skylake`     |         4.22 gb/s, 0 ulp |         3.38 gb/s, 0 ulp |         4.54 gb/s, 0 ulp |
| `nk_sqeuclidean_e4m3_icelake` |         10.2 gb/s, 0 ulp |         12.0 gb/s, 0 ulp |       12.0 gb/s, 0.2 ulp |
| `nk_euclidean_e4m3_icelake`   |         10.3 gb/s, 0 ulp |         11.8 gb/s, 0 ulp |       11.9 gb/s, 0.2 ulp |
| `nk_angular_e4m3_icelake`     |         8.78 gb/s, 0 ulp |         11.3 gb/s, 0 ulp |         11.9 gb/s, 0 ulp |
| __e3m2__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e3m2_serial`  |         1.01 gb/s, 0 ulp |        0.971 gb/s, 0 ulp |         1.03 gb/s, 0 ulp |
| `nk_euclidean_e3m2_serial`    |      0.997 gb/s, 0.5 ulp |      0.990 gb/s, 0.5 ulp |      0.999 gb/s, 0.4 ulp |
| `nk_angular_e3m2_serial`      |        0.332 gb/s, 0 ulp |        0.361 gb/s, 0 ulp |        0.437 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_skylake` |         4.47 gb/s, 0 ulp |         5.46 gb/s, 0 ulp |         5.04 gb/s, 0 ulp |
| `nk_euclidean_e3m2_skylake`   |         4.34 gb/s, 0 ulp |         6.20 gb/s, 0 ulp |         5.10 gb/s, 0 ulp |
| `nk_angular_e3m2_skylake`     |         3.79 gb/s, 0 ulp |         4.41 gb/s, 0 ulp |         4.82 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_genoa`   |         8.79 gb/s, 0 ulp |         9.52 gb/s, 0 ulp |         10.6 gb/s, 0 ulp |
| `nk_euclidean_e3m2_genoa`     |         8.68 gb/s, 0 ulp |         9.01 gb/s, 0 ulp |         12.8 gb/s, 0 ulp |
| `nk_angular_e3m2_genoa`       |         6.89 gb/s, 0 ulp |         9.30 gb/s, 0 ulp |         10.3 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_icelake` |         21.2 gb/s, 0 ulp |         22.1 gb/s, 0 ulp |         21.9 gb/s, 0 ulp |
| `nk_euclidean_e3m2_icelake`   |         21.2 gb/s, 0 ulp |         22.9 gb/s, 0 ulp |         21.2 gb/s, 0 ulp |
| `nk_angular_e3m2_icelake`     |         14.1 gb/s, 0 ulp |         18.0 gb/s, 0 ulp |         17.6 gb/s, 0 ulp |
| __e2m3__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e2m3_serial`  |        0.964 gb/s, 0 ulp |        0.981 gb/s, 0 ulp |         1.03 gb/s, 0 ulp |
| `nk_euclidean_e2m3_serial`    |      0.979 gb/s, 0.5 ulp |      0.966 gb/s, 0.5 ulp |       1.02 gb/s, 0.5 ulp |
| `nk_angular_e2m3_serial`      |        0.347 gb/s, 0 ulp |        0.389 gb/s, 0 ulp |        0.418 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_skylake` |         4.58 gb/s, 0 ulp |         4.65 gb/s, 0 ulp |         5.08 gb/s, 0 ulp |
| `nk_euclidean_e2m3_skylake`   |         4.48 gb/s, 0 ulp |         4.39 gb/s, 0 ulp |         4.96 gb/s, 0 ulp |
| `nk_angular_e2m3_skylake`     |         3.94 gb/s, 0 ulp |         4.25 gb/s, 0 ulp |         4.90 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_genoa`   |         9.62 gb/s, 0 ulp |         10.9 gb/s, 0 ulp |         10.8 gb/s, 0 ulp |
| `nk_euclidean_e2m3_genoa`     |         8.45 gb/s, 0 ulp |         9.80 gb/s, 0 ulp |         10.3 gb/s, 0 ulp |
| `nk_angular_e2m3_genoa`       |         7.21 gb/s, 0 ulp |         10.1 gb/s, 0 ulp |         10.4 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_icelake` |         50.7 gb/s, 0 ulp |         42.6 gb/s, 0 ulp |         31.0 gb/s, 0 ulp |
| `nk_euclidean_e2m3_icelake`   |         50.2 gb/s, 0 ulp |         44.3 gb/s, 0 ulp |         31.0 gb/s, 0 ulp |
| `nk_angular_e2m3_icelake`     |         27.2 gb/s, 0 ulp |         34.9 gb/s, 0 ulp |         30.5 gb/s, 0 ulp |
| __i8__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`    |                34.0 gb/s |                18.4 gb/s |                16.5 gb/s |
| `nk_euclidean_i8_serial`      |       29.0 gb/s, 0.4 ulp |       18.0 gb/s, 0.4 ulp |       15.6 gb/s, 0.4 ulp |
| `nk_angular_i8_serial`        |         7.88 gb/s, 0 ulp |         6.31 gb/s, 0 ulp |         6.12 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_haswell`   |                38.4 gb/s |                17.9 gb/s |                18.4 gb/s |
| `nk_euclidean_i8_haswell`     |         35.6 gb/s, 0 ulp |         17.0 gb/s, 0 ulp |         15.5 gb/s, 0 ulp |
| `nk_angular_i8_haswell`       |       20.3 gb/s, 0.1 ulp |         12.9 gb/s, 0 ulp |         11.9 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_icelake`   |                60.2 gb/s |                24.5 gb/s |                23.5 gb/s |
| `nk_euclidean_i8_icelake`     |         59.0 gb/s, 0 ulp |         23.0 gb/s, 0 ulp |         22.3 gb/s, 0 ulp |
| `nk_angular_i8_icelake`       |       25.2 gb/s, 0.1 ulp |         18.4 gb/s, 0 ulp |         20.5 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_alder`     |                33.4 gb/s |                17.4 gb/s |                17.6 gb/s |
| `nk_euclidean_i8_alder`       |         31.9 gb/s, 0 ulp |         19.1 gb/s, 0 ulp |         17.8 gb/s, 0 ulp |
| `nk_angular_i8_alder`         |       26.2 gb/s, 0.1 ulp |         17.1 gb/s, 0 ulp |         17.8 gb/s, 0 ulp |
| __u8__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`    |                11.7 gb/s |                8.77 gb/s |                7.07 gb/s |
| `nk_euclidean_u8_serial`      |       11.6 gb/s, 0.5 ulp |       8.31 gb/s, 0.5 ulp |       8.36 gb/s, 0.6 ulp |
| `nk_angular_u8_serial`        |       7.95 gb/s, 0.4 ulp |       6.68 gb/s, 0.4 ulp |       5.88 gb/s, 0.4 ulp |
| `nk_sqeuclidean_u8_haswell`   |                45.4 gb/s |                17.7 gb/s |                18.5 gb/s |
| `nk_euclidean_u8_haswell`     |         38.9 gb/s, 0 ulp |         18.8 gb/s, 0 ulp |         19.3 gb/s, 0 ulp |
| `nk_angular_u8_haswell`       |       21.9 gb/s, 0.7 ulp |       11.7 gb/s, 0.6 ulp |       13.4 gb/s, 0.5 ulp |
| `nk_sqeuclidean_u8_icelake`   |                70.1 gb/s |                28.8 gb/s |                21.0 gb/s |
| `nk_euclidean_u8_icelake`     |         66.4 gb/s, 0 ulp |         27.6 gb/s, 0 ulp |         23.5 gb/s, 0 ulp |
| `nk_angular_u8_icelake`       |       28.9 gb/s, 0.7 ulp |       21.2 gb/s, 0.6 ulp |       21.5 gb/s, 0.5 ulp |
| `nk_sqeuclidean_u8_alder`     |                32.2 gb/s |                17.5 gb/s |                19.0 gb/s |
| `nk_euclidean_u8_alder`       |         31.3 gb/s, 0 ulp |         17.0 gb/s, 0 ulp |         19.6 gb/s, 0 ulp |
| `nk_angular_u8_alder`         |       26.5 gb/s, 0.7 ulp |       17.1 gb/s, 0.6 ulp |       17.5 gb/s, 0.5 ulp |
| __i4__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i4_serial`    |                15.4 gb/s |                16.5 gb/s |                15.6 gb/s |
| `nk_euclidean_i4_serial`      |       12.2 gb/s, 0.5 ulp |       15.6 gb/s, 0.5 ulp |       15.2 gb/s, 0.6 ulp |
| `nk_angular_i4_serial`        |       5.60 gb/s, 0.4 ulp |       6.42 gb/s, 0.4 ulp |       6.69 gb/s, 0.4 ulp |
| `nk_sqeuclidean_i4_icelake`   |                23.6 gb/s |                51.5 gb/s |                29.3 gb/s |
| `nk_euclidean_i4_icelake`     |         20.6 gb/s, 0 ulp |         45.2 gb/s, 0 ulp |         28.9 gb/s, 0 ulp |
| `nk_angular_i4_icelake`       |       5.14 gb/s, 0.7 ulp |       18.0 gb/s, 0.6 ulp |       17.6 gb/s, 0.5 ulp |
| __u4__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u4_serial`    |                15.6 gb/s |                17.3 gb/s |                15.8 gb/s |
| `nk_euclidean_u4_serial`      |       12.0 gb/s, 0.5 ulp |       15.9 gb/s, 0.5 ulp |       15.3 gb/s, 0.6 ulp |
| `nk_angular_u4_serial`        |       5.20 gb/s, 0.4 ulp |       6.63 gb/s, 0.4 ulp |       7.01 gb/s, 0.4 ulp |
| `nk_sqeuclidean_u4_icelake`   |                22.7 gb/s |                23.7 gb/s |                24.5 gb/s |
| `nk_euclidean_u4_icelake`     |         20.9 gb/s, 0 ulp |         18.8 gb/s, 0 ulp |         24.1 gb/s, 0 ulp |
| `nk_angular_u4_icelake`       |       9.32 gb/s, 0.7 ulp |       27.4 gb/s, 0.6 ulp |       24.2 gb/s, 0.5 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                            |                      256 |                     1024 |                     4096 |
| :-------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`       |       2.97 gb/s, 0.1 ulp |         3.16 gb/s, 0 ulp |         0.02 gb/s, 0 ulp |
| `nk_euclidean_f64_serial`         |      0.104 gb/s, 0.6 ulp |       1.06 gb/s, 0.6 ulp |       0.33 gb/s, 0.5 ulp |
| `nk_angular_f64_serial`           |       1.91 gb/s, 0.1 ulp |         1.93 gb/s, 0 ulp |         0.18 gb/s, 0 ulp |
| `nk_sqeuclidean_f64_v128relaxed`  |       1.23 gb/s, 1.3 ulp |       1.87 gb/s, 2.5 ulp |       0.15 gb/s, 5.0 ulp |
| `nk_euclidean_f64_v128relaxed`    |      0.315 gb/s, 0.7 ulp |       2.21 gb/s, 1.4 ulp |       0.03 gb/s, 2.8 ulp |
| `nk_angular_f64_v128relaxed`      |       1.14 gb/s, 0.1 ulp |      0.928 gb/s, 0.1 ulp |       0.26 gb/s, 0.1 ulp |
| __f32__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`       |        0.657 gb/s, 0 ulp |        0.928 gb/s, 0 ulp |         0.06 gb/s, 0 ulp |
| `nk_euclidean_f32_serial`         |      0.757 gb/s, 0.1 ulp |      0.914 gb/s, 0.1 ulp |       0.05 gb/s, 0.1 ulp |
| `nk_angular_f32_serial`           |        0.882 gb/s, 0 ulp |        0.902 gb/s, 0 ulp |         0.26 gb/s, 0 ulp |
| `nk_sqeuclidean_f32_v128relaxed`  |       2.87 gb/s, 0.7 ulp |       3.03 gb/s, 1.3 ulp |       1.77 gb/s, 2.6 ulp |
| `nk_euclidean_f32_v128relaxed`    |       1.83 gb/s, 0.4 ulp |       3.00 gb/s, 0.7 ulp |       0.22 gb/s, 1.4 ulp |
| `nk_angular_f32_v128relaxed`      |         3.37 gb/s, 0 ulp |        0.991 gb/s, 0 ulp |         0.19 gb/s, 0 ulp |
| __bf16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`      |         1.89 gb/s, 0 ulp |         1.09 gb/s, 0 ulp |         0.31 gb/s, 0 ulp |
| `nk_euclidean_bf16_serial`        |       2.02 gb/s, 0.6 ulp |       2.13 gb/s, 0.5 ulp |       0.29 gb/s, 0.5 ulp |
| `nk_angular_bf16_serial`          |        0.399 gb/s, 0 ulp |        0.308 gb/s, 0 ulp |         0.11 gb/s, 0 ulp |
| `nk_sqeuclidean_bf16_v128relaxed` |       2.10 gb/s, 0.9 ulp |      1.94 gb/s, 12.6 ulp |      0.17 gb/s, 20.8 ulp |
| `nk_euclidean_bf16_v128relaxed`   |       2.08 gb/s, 0.5 ulp |       2.22 gb/s, 7.0 ulp |      0.13 gb/s, 11.4 ulp |
| `nk_angular_bf16_v128relaxed`     |         1.08 gb/s, 0 ulp |       2.09 gb/s, 0.2 ulp |       0.20 gb/s, 0.6 ulp |
| __f16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`       |       1.10 gb/s, 0.1 ulp |       1.13 gb/s, 0.1 ulp |       0.20 gb/s, 0.1 ulp |
| `nk_euclidean_f16_serial`         |       1.17 gb/s, 0.6 ulp |       1.16 gb/s, 0.6 ulp |       0.26 gb/s, 0.5 ulp |
| `nk_angular_f16_serial`           |        0.363 gb/s, 0 ulp |        0.372 gb/s, 0 ulp |         0.06 gb/s, 0 ulp |
| `nk_sqeuclidean_f16_v128relaxed`  |       1.12 gb/s, 0.9 ulp |      0.633 gb/s, 3.6 ulp |       0.03 gb/s, 9.7 ulp |
| `nk_euclidean_f16_v128relaxed`    |      0.806 gb/s, 0.5 ulp |      0.991 gb/s, 2.0 ulp |       0.09 gb/s, 5.4 ulp |
| `nk_angular_f16_v128relaxed`      |       1.79 gb/s, 0.1 ulp |      0.976 gb/s, 0.1 ulp |       0.00 gb/s, 0.1 ulp |
| __e5m2__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e5m2_serial`      |        0.713 gb/s, 0 ulp |        0.689 gb/s, 0 ulp |         0.16 gb/s, 0 ulp |
| `nk_euclidean_e5m2_serial`        |      0.637 gb/s, 0.5 ulp |      0.736 gb/s, 0.5 ulp |       0.12 gb/s, 0.5 ulp |
| `nk_angular_e5m2_serial`          |        0.169 gb/s, 0 ulp |        0.162 gb/s, 0 ulp |         0.17 gb/s, 0 ulp |
| __e4m3__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e4m3_serial`      |        0.374 gb/s, 0 ulp |        0.383 gb/s, 0 ulp |         0.09 gb/s, 0 ulp |
| `nk_euclidean_e4m3_serial`        |      0.374 gb/s, 0.5 ulp |      0.360 gb/s, 0.5 ulp |       0.09 gb/s, 0.5 ulp |
| `nk_angular_e4m3_serial`          |        0.162 gb/s, 0 ulp |        0.166 gb/s, 0 ulp |         0.17 gb/s, 0 ulp |
| __e3m2__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e3m2_serial`      |        0.712 gb/s, 0 ulp |        0.744 gb/s, 0 ulp |         0.17 gb/s, 0 ulp |
| `nk_euclidean_e3m2_serial`        |      0.709 gb/s, 0.5 ulp |      0.759 gb/s, 0.5 ulp |       0.17 gb/s, 0.5 ulp |
| `nk_angular_e3m2_serial`          |        0.152 gb/s, 0 ulp |        0.165 gb/s, 0 ulp |         0.17 gb/s, 0 ulp |
| __e2m3__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e2m3_serial`      |        0.702 gb/s, 0 ulp |        0.760 gb/s, 0 ulp |         0.13 gb/s, 0 ulp |
| `nk_euclidean_e2m3_serial`        |      0.650 gb/s, 0.5 ulp |      0.753 gb/s, 0.5 ulp |       0.15 gb/s, 0.5 ulp |
| `nk_angular_e2m3_serial`          |        0.158 gb/s, 0 ulp |        0.168 gb/s, 0 ulp |         0.17 gb/s, 0 ulp |
| __i8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`        |               0.327 gb/s |               0.328 gb/s |                0.09 gb/s |
| `nk_euclidean_i8_serial`          |       2.93 gb/s, 0.5 ulp |      0.174 gb/s, 0.4 ulp |       0.14 gb/s, 0.4 ulp |
| `nk_angular_i8_serial`            |         1.23 gb/s, 0 ulp |        0.946 gb/s, 0 ulp |         0.10 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_v128relaxed`   |                1.84 gb/s |               0.736 gb/s |                0.08 gb/s |
| `nk_euclidean_i8_v128relaxed`     |         1.36 gb/s, 0 ulp |        0.805 gb/s, 0 ulp |         0.21 gb/s, 0 ulp |
| `nk_angular_i8_v128relaxed`       |         1.80 gb/s, 0 ulp |         2.79 gb/s, 0 ulp |         0.14 gb/s, 0 ulp |
| __u8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`        |               0.528 gb/s |               0.496 gb/s |                0.30 gb/s |
| `nk_euclidean_u8_serial`          |    0.00982 gb/s, 0.5 ulp |      0.311 gb/s, 0.5 ulp |       0.04 gb/s, 0.6 ulp |
| `nk_angular_u8_serial`            |      0.813 gb/s, 0.5 ulp |       1.46 gb/s, 0.4 ulp |       0.29 gb/s, 0.5 ulp |
| `nk_sqeuclidean_u8_v128relaxed`   |                3.05 gb/s |                1.68 gb/s |                0.28 gb/s |
| `nk_euclidean_u8_v128relaxed`     |         2.52 gb/s, 0 ulp |         1.70 gb/s, 0 ulp |         0.09 gb/s, 0 ulp |
| `nk_angular_u8_v128relaxed`       |      2.47 gb/s, 526M ulp |      1.91 gb/s, 501M ulp |      0.09 gb/s, 443M ulp |
| __i4__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i4_serial`        |                1.91 gb/s |                1.94 gb/s |                0.30 gb/s |
| `nk_euclidean_i4_serial`          |       1.76 gb/s, 0.5 ulp |       1.90 gb/s, 0.5 ulp |       0.02 gb/s, 0.0 ulp |
| `nk_angular_i4_serial`            |       1.28 gb/s, 0.5 ulp |       1.34 gb/s, 0.5 ulp |       0.10 gb/s, 0.5 ulp |
| __u4__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u4_serial`        |                2.91 gb/s |                3.00 gb/s |                0.09 gb/s |
| `nk_euclidean_u4_serial`          |       2.78 gb/s, 0.5 ulp |       3.01 gb/s, 0.5 ulp |       0.10 gb/s, 0.0 ulp |
| `nk_angular_u4_serial`            |       1.84 gb/s, 0.5 ulp |       2.03 gb/s, 0.5 ulp |       0.21 gb/s, 0.5 ulp |

### Apple M5

#### Native

| Kernel                          |                      256 |                     1024 |                     4096 |
| :------------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`     |       12.4 gb/s, 0.1 ulp |         12.8 gb/s, 0 ulp |         12.8 gb/s, 0 ulp |
| `nk_euclidean_f64_serial`       |       12.7 gb/s, 0.6 ulp |       12.9 gb/s, 0.5 ulp |       12.6 gb/s, 0.5 ulp |
| `nk_angular_f64_serial`         |         8.42 gb/s, 0 ulp |         8.57 gb/s, 0 ulp |         8.30 gb/s, 0 ulp |
| `nk_sqeuclidean_f64_neon`       |       50.6 gb/s, 1.3 ulp |       40.0 gb/s, 2.6 ulp |       36.1 gb/s, 5.1 ulp |
| `nk_euclidean_f64_neon`         |       48.4 gb/s, 0.7 ulp |       38.7 gb/s, 1.4 ulp |       35.1 gb/s, 2.8 ulp |
| `nk_angular_f64_neon`           |       33.3 gb/s, 0.1 ulp |         33.4 gb/s, 0 ulp |         32.4 gb/s, 0 ulp |
| __f32__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`     |         6.32 gb/s, 0 ulp |         6.25 gb/s, 0 ulp |         6.30 gb/s, 0 ulp |
| `nk_euclidean_f32_serial`       |       6.31 gb/s, 0.1 ulp |       6.37 gb/s, 0.1 ulp |       6.41 gb/s, 0.1 ulp |
| `nk_angular_f32_serial`         |         4.03 gb/s, 0 ulp |         4.06 gb/s, 0 ulp |         4.07 gb/s, 0 ulp |
| `nk_sqeuclidean_f32_neon`       |       25.3 gb/s, 0.1 ulp |         19.1 gb/s, 0 ulp |         17.5 gb/s, 0 ulp |
| `nk_euclidean_f32_neon`         |       25.0 gb/s, 0.1 ulp |       20.8 gb/s, 0.1 ulp |       18.6 gb/s, 0.1 ulp |
| `nk_angular_f32_neon`           |         22.2 gb/s, 0 ulp |         17.3 gb/s, 0 ulp |         16.6 gb/s, 0 ulp |
| __bf16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`    |         3.19 gb/s, 0 ulp |         3.16 gb/s, 0 ulp |         3.14 gb/s, 0 ulp |
| `nk_euclidean_bf16_serial`      |       3.16 gb/s, 0.5 ulp |       3.08 gb/s, 0.5 ulp |       3.11 gb/s, 0.5 ulp |
| `nk_angular_bf16_serial`        |         1.88 gb/s, 0 ulp |         1.91 gb/s, 0 ulp |         1.93 gb/s, 0 ulp |
| `nk_sqeuclidean_bf16_neonbfdot` |       35.0 gb/s, 0.9 ulp |        22.7 gb/s, 13 ulp |        18.8 gb/s, 21 ulp |
| `nk_euclidean_bf16_neonbfdot`   |       33.4 gb/s, 0.5 ulp |       23.0 gb/s, 7.0 ulp |        18.6 gb/s, 12 ulp |
| `nk_angular_bf16_neonbfdot`     |         23.8 gb/s, 0 ulp |       32.7 gb/s, 0.1 ulp |         35.9 gb/s, 0 ulp |
| __f16__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`     |       3.09 gb/s, 0.1 ulp |       3.16 gb/s, 0.1 ulp |       3.10 gb/s, 0.1 ulp |
| `nk_euclidean_f16_serial`       |       3.13 gb/s, 0.6 ulp |       3.14 gb/s, 0.5 ulp |       3.11 gb/s, 0.5 ulp |
| `nk_angular_f16_serial`         |         1.84 gb/s, 0 ulp |         1.92 gb/s, 0 ulp |         1.88 gb/s, 0 ulp |
| `nk_sqeuclidean_f16_neonhalf`   |       34.7 gb/s, 0.9 ulp |       21.5 gb/s, 3.6 ulp |       18.3 gb/s, 9.7 ulp |
| `nk_euclidean_f16_neonhalf`     |       32.7 gb/s, 0.5 ulp |       21.7 gb/s, 2.0 ulp |       18.4 gb/s, 5.3 ulp |
| `nk_angular_f16_neonhalf`       |       25.2 gb/s, 0.1 ulp |       19.6 gb/s, 0.1 ulp |       17.3 gb/s, 0.1 ulp |
| __e5m2__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e5m2_serial`    |         2.09 gb/s, 0 ulp |         2.08 gb/s, 0 ulp |         2.10 gb/s, 0 ulp |
| `nk_euclidean_e5m2_serial`      |       2.06 gb/s, 0.5 ulp |       2.10 gb/s, 0.5 ulp |       2.05 gb/s, 0.5 ulp |
| `nk_angular_e5m2_serial`        |        0.921 gb/s, 0 ulp |        0.956 gb/s, 0 ulp |        0.938 gb/s, 0 ulp |
| `nk_sqeuclidean_e5m2_neon`      |         18.2 gb/s, 0 ulp |         12.8 gb/s, 0 ulp |         9.84 gb/s, 0 ulp |
| `nk_euclidean_e5m2_neon`        |       18.0 gb/s, 0.5 ulp |       11.8 gb/s, 0.5 ulp |       9.33 gb/s, 0.5 ulp |
| `nk_angular_e5m2_neon`          |         13.7 gb/s, 0 ulp |         10.9 gb/s, 0 ulp |         9.83 gb/s, 0 ulp |
| __e4m3__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e4m3_serial`    |         1.07 gb/s, 0 ulp |         1.12 gb/s, 0 ulp |         1.11 gb/s, 0 ulp |
| `nk_euclidean_e4m3_serial`      |       1.01 gb/s, 0.5 ulp |       1.12 gb/s, 0.5 ulp |       1.09 gb/s, 0.5 ulp |
| `nk_angular_e4m3_serial`        |        0.711 gb/s, 0 ulp |        0.732 gb/s, 0 ulp |        0.729 gb/s, 0 ulp |
| `nk_sqeuclidean_e4m3_neon`      |       4.29 gb/s, 0.2 ulp |       4.36 gb/s, 0.2 ulp |       4.33 gb/s, 0.2 ulp |
| `nk_euclidean_e4m3_neon`        |       4.20 gb/s, 0.5 ulp |       4.11 gb/s, 0.5 ulp |       4.17 gb/s, 0.5 ulp |
| `nk_angular_e4m3_neon`          |         4.13 gb/s, 0 ulp |         4.21 gb/s, 0 ulp |         4.16 gb/s, 0 ulp |
| __e3m2__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e3m2_serial`    |         1.95 gb/s, 0 ulp |         2.15 gb/s, 0 ulp |         2.08 gb/s, 0 ulp |
| `nk_euclidean_e3m2_serial`      |       1.97 gb/s, 0.5 ulp |       2.18 gb/s, 0.5 ulp |       2.09 gb/s, 0.5 ulp |
| `nk_angular_e3m2_serial`        |        0.900 gb/s, 0 ulp |        0.985 gb/s, 0 ulp |        0.943 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_neon`      |         4.73 gb/s, 0 ulp |         5.19 gb/s, 0 ulp |         5.03 gb/s, 0 ulp |
| `nk_euclidean_e3m2_neon`        |         4.78 gb/s, 0 ulp |         5.23 gb/s, 0 ulp |         5.05 gb/s, 0 ulp |
| `nk_angular_e3m2_neon`          |         4.24 gb/s, 0 ulp |         4.85 gb/s, 0 ulp |         4.73 gb/s, 0 ulp |
| __e2m3__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e2m3_serial`    |         1.98 gb/s, 0 ulp |         2.20 gb/s, 0 ulp |         2.11 gb/s, 0 ulp |
| `nk_euclidean_e2m3_serial`      |       1.91 gb/s, 0.5 ulp |       2.16 gb/s, 0.5 ulp |       2.09 gb/s, 0.4 ulp |
| `nk_angular_e2m3_serial`        |        0.885 gb/s, 0 ulp |        0.985 gb/s, 0 ulp |        0.953 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_neon`      |         4.67 gb/s, 0 ulp |         5.06 gb/s, 0 ulp |         5.07 gb/s, 0 ulp |
| `nk_euclidean_e2m3_neon`        |         4.84 gb/s, 0 ulp |         5.17 gb/s, 0 ulp |         4.98 gb/s, 0 ulp |
| `nk_angular_e2m3_neon`          |         4.45 gb/s, 0 ulp |         4.88 gb/s, 0 ulp |         4.73 gb/s, 0 ulp |
| __i8__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`      |                61.6 gb/s |                75.2 gb/s |                64.9 gb/s |
| `nk_euclidean_i8_serial`        |                43.5 gb/s |                54.0 gb/s |                64.3 gb/s |
| `nk_angular_i8_serial`          |                55.8 gb/s |                63.7 gb/s |                49.6 gb/s |
| `nk_sqeuclidean_i8_neonsdot`    |                89.1 gb/s |                85.9 gb/s |                58.8 gb/s |
| `nk_euclidean_i8_neonsdot`      |                86.9 gb/s |                78.9 gb/s |                57.4 gb/s |
| `nk_angular_i8_neonsdot`        |                66.5 gb/s |                68.9 gb/s |                50.9 gb/s |
| __u8__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`      |                62.9 gb/s |                77.2 gb/s |                66.5 gb/s |
| `nk_euclidean_u8_serial`        |                45.7 gb/s |                52.3 gb/s |                61.4 gb/s |
| `nk_angular_u8_serial`          |                17.8 gb/s |                18.5 gb/s |                16.0 gb/s |
| `nk_sqeuclidean_u8_neonsdot`    |                91.7 gb/s |                83.1 gb/s |                56.6 gb/s |
| `nk_euclidean_u8_neonsdot`      |                87.9 gb/s |                79.3 gb/s |                56.5 gb/s |
| `nk_angular_u8_neonsdot`        |                68.0 gb/s |                64.8 gb/s |                49.5 gb/s |
| __i4__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i4_serial`      |                22.9 gb/s |                25.2 gb/s |                25.1 gb/s |
| `nk_euclidean_i4_serial`        |                20.1 gb/s |                23.6 gb/s |                24.2 gb/s |
| `nk_angular_i4_serial`          |                9.11 gb/s |                10.4 gb/s |                10.4 gb/s |
| __u4__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u4_serial`      |                26.7 gb/s |                26.8 gb/s |                22.4 gb/s |
| `nk_euclidean_u4_serial`        |                20.9 gb/s |                22.5 gb/s |                21.2 gb/s |
| `nk_angular_u4_serial`          |                9.00 gb/s |                9.57 gb/s |                9.62 gb/s |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                            |                      256 |                     1024 |                     4096 |
| :-------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`       |       20.3 gb/s, 0.1 ulp |         19.3 gb/s, 0 ulp |         20.3 gb/s, 0 ulp |
| `nk_euclidean_f64_serial`         |       20.0 gb/s, 0.6 ulp |       19.4 gb/s, 0.6 ulp |       20.3 gb/s, 0.5 ulp |
| `nk_angular_f64_serial`           |         9.28 gb/s, 0 ulp |         8.83 gb/s, 0 ulp |         9.29 gb/s, 0 ulp |
| `nk_sqeuclidean_f64_v128relaxed`  |       48.2 gb/s, 1.3 ulp |       35.7 gb/s, 2.6 ulp |       37.0 gb/s, 5.0 ulp |
| `nk_euclidean_f64_v128relaxed`    |       50.1 gb/s, 0.7 ulp |       36.5 gb/s, 1.4 ulp |       36.7 gb/s, 2.8 ulp |
| `nk_angular_f64_v128relaxed`      |       31.5 gb/s, 0.1 ulp |       22.3 gb/s, 0.1 ulp |       22.4 gb/s, 0.1 ulp |
| __f32__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`       |         8.90 gb/s, 0 ulp |         8.54 gb/s, 0 ulp |         8.86 gb/s, 0 ulp |
| `nk_euclidean_f32_serial`         |       8.86 gb/s, 0.1 ulp |       8.58 gb/s, 0.1 ulp |       8.86 gb/s, 0.1 ulp |
| `nk_angular_f32_serial`           |         4.33 gb/s, 0 ulp |         4.17 gb/s, 0 ulp |         4.30 gb/s, 0 ulp |
| `nk_sqeuclidean_f32_v128relaxed`  |       20.4 gb/s, 0.7 ulp |       17.9 gb/s, 1.3 ulp |       18.3 gb/s, 2.6 ulp |
| `nk_euclidean_f32_v128relaxed`    |       20.3 gb/s, 0.4 ulp |       18.0 gb/s, 0.7 ulp |       18.3 gb/s, 1.4 ulp |
| `nk_angular_f32_v128relaxed`      |         19.7 gb/s, 0 ulp |         17.8 gb/s, 0 ulp |         18.5 gb/s, 0 ulp |
| __bf16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`      |         5.16 gb/s, 0 ulp |         4.88 gb/s, 0 ulp |         5.10 gb/s, 0 ulp |
| `nk_euclidean_bf16_serial`        |       5.12 gb/s, 0.6 ulp |       4.89 gb/s, 0.5 ulp |       5.10 gb/s, 0.5 ulp |
| `nk_angular_bf16_serial`          |         2.24 gb/s, 0 ulp |         2.15 gb/s, 0 ulp |         2.24 gb/s, 0 ulp |
| `nk_sqeuclidean_bf16_v128relaxed` |       39.9 gb/s, 0.9 ulp |        27.0 gb/s, 13 ulp |        20.3 gb/s, 21 ulp |
| `nk_euclidean_bf16_v128relaxed`   |       38.6 gb/s, 0.5 ulp |       27.1 gb/s, 7.0 ulp |        21.2 gb/s, 12 ulp |
| `nk_angular_bf16_v128relaxed`     |         27.9 gb/s, 0 ulp |       22.6 gb/s, 0.2 ulp |       20.5 gb/s, 0.6 ulp |
| __f16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`       |       3.22 gb/s, 0.1 ulp |       3.06 gb/s, 0.1 ulp |       3.09 gb/s, 0.1 ulp |
| `nk_euclidean_f16_serial`         |       3.19 gb/s, 0.6 ulp |       2.92 gb/s, 0.5 ulp |       3.26 gb/s, 0.5 ulp |
| `nk_angular_f16_serial`           |         2.33 gb/s, 0 ulp |         2.21 gb/s, 0 ulp |         2.32 gb/s, 0 ulp |
| `nk_sqeuclidean_f16_v128relaxed`  |       11.2 gb/s, 0.9 ulp |       11.0 gb/s, 3.6 ulp |       11.8 gb/s, 9.6 ulp |
| `nk_euclidean_f16_v128relaxed`    |       11.3 gb/s, 0.5 ulp |       11.0 gb/s, 2.0 ulp |       11.8 gb/s, 5.3 ulp |
| `nk_angular_f16_v128relaxed`      |       9.41 gb/s, 0.1 ulp |       9.56 gb/s, 0.1 ulp |       10.4 gb/s, 0.1 ulp |
| __i8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`        |                14.9 gb/s |                14.7 gb/s |                16.5 gb/s |
| `nk_euclidean_i8_serial`          |       14.7 gb/s, 0.5 ulp |       14.8 gb/s, 0.4 ulp |       16.3 gb/s, 0.4 ulp |
| `nk_angular_i8_serial`            |         8.06 gb/s, 0 ulp |         8.42 gb/s, 0 ulp |         10.7 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_v128relaxed`   |                30.7 gb/s |                22.9 gb/s |                18.0 gb/s |
| `nk_euclidean_i8_v128relaxed`     |         27.4 gb/s, 0 ulp |         22.6 gb/s, 0 ulp |         17.9 gb/s, 0 ulp |
| `nk_angular_i8_v128relaxed`       |         17.2 gb/s, 0 ulp |         18.1 gb/s, 0 ulp |         19.7 gb/s, 0 ulp |
| __u8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`        |                14.8 gb/s |                14.5 gb/s |                16.3 gb/s |
| `nk_euclidean_u8_serial`          |       14.5 gb/s, 0.5 ulp |       14.5 gb/s, 0.5 ulp |       16.0 gb/s, 0.6 ulp |
| `nk_angular_u8_serial`            |       7.86 gb/s, 0.5 ulp |       8.25 gb/s, 0.5 ulp |       10.7 gb/s, 0.4 ulp |
| `nk_sqeuclidean_u8_v128relaxed`   |                33.2 gb/s |                24.6 gb/s |                18.3 gb/s |
| `nk_euclidean_u8_v128relaxed`     |         28.6 gb/s, 0 ulp |         23.7 gb/s, 0 ulp |         18.2 gb/s, 0 ulp |
| `nk_angular_u8_v128relaxed`       |         14.1 gb/s, 0 ulp |         15.0 gb/s, 0 ulp |         16.1 gb/s, 0 ulp |
