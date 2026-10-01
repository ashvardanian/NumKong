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
| :--------- | :---------- | :----------------------------------------------- |
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
Haswell `VRSQRTPS` starts from ~12 bits (1.5 × $2^{-12}$ relative error) and one Newton-Raphson step roughly doubles that; Skylake `VRSQRT14PS` starts from $2^{-14}$ and refines the same way.
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

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by the `NUMWARS_DIMS` environment variable and set to 256, 1024, and 4096 elements.
The throughput is measured in GB/s as the number of input bytes per second.

### Intel Sapphire Rapids

#### Native

| Kernel                        |                      256 |                     1024 |                     4096 |
| :---------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`   |       7.45 gb/s, 0.1 ulp |         7.75 gb/s, 0 ulp |         7.57 gb/s, 0 ulp |
| `nk_euclidean_f64_serial`     |       7.27 gb/s, 0.6 ulp |       7.40 gb/s, 0.5 ulp |       7.77 gb/s, 0.5 ulp |
| `nk_angular_f64_serial`       |         2.61 gb/s, 0 ulp |         2.82 gb/s, 0 ulp |         2.96 gb/s, 0 ulp |
| `nk_sqeuclidean_f64_skylake`  |       30.2 gb/s, 0.4 ulp |       28.5 gb/s, 0.7 ulp |       20.7 gb/s, 1.3 ulp |
| `nk_euclidean_f64_skylake`    |       29.5 gb/s, 0.3 ulp |       27.4 gb/s, 0.4 ulp |       21.3 gb/s, 0.7 ulp |
| `nk_angular_f64_skylake`      |         24.7 gb/s, 0 ulp |         25.0 gb/s, 0 ulp |         16.6 gb/s, 0 ulp |
| __f32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`   |         3.73 gb/s, 0 ulp |         3.78 gb/s, 0 ulp |         3.90 gb/s, 0 ulp |
| `nk_euclidean_f32_serial`     |       3.72 gb/s, 0.1 ulp |       3.79 gb/s, 0.1 ulp |       3.83 gb/s, 0.1 ulp |
| `nk_angular_f32_serial`       |         1.20 gb/s, 0 ulp |         1.31 gb/s, 0 ulp |         1.42 gb/s, 0 ulp |
| `nk_sqeuclidean_f32_skylake`  |         34.0 gb/s, 0 ulp |         25.1 gb/s, 0 ulp |         21.6 gb/s, 0 ulp |
| `nk_euclidean_f32_skylake`    |       33.9 gb/s, 0.1 ulp |       26.2 gb/s, 0.1 ulp |       24.9 gb/s, 0.1 ulp |
| `nk_angular_f32_skylake`      |         22.6 gb/s, 0 ulp |         21.6 gb/s, 0 ulp |         21.0 gb/s, 0 ulp |
| __bf16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`  |        0.542 gb/s, 0 ulp |        0.333 gb/s, 0 ulp |        0.363 gb/s, 0 ulp |
| `nk_euclidean_bf16_serial`    |      0.530 gb/s, 0.5 ulp |      0.347 gb/s, 0.5 ulp |      0.346 gb/s, 0.4 ulp |
| `nk_angular_bf16_serial`      |        0.424 gb/s, 0 ulp |        0.224 gb/s, 0 ulp |        0.241 gb/s, 0 ulp |
| `nk_sqeuclidean_bf16_haswell` |       25.8 gb/s, 0.5 ulp |       13.0 gb/s, 7.5 ulp |        11.0 gb/s, 27 ulp |
| `nk_euclidean_bf16_haswell`   |       21.7 gb/s, 0.3 ulp |       12.5 gb/s, 4.1 ulp |        11.2 gb/s, 15 ulp |
| `nk_angular_bf16_haswell`     |         18.7 gb/s, 0 ulp |         12.5 gb/s, 0 ulp |       9.87 gb/s, 0.2 ulp |
| `nk_sqeuclidean_bf16_genoa`   |       46.7 gb/s, 0.3 ulp |       19.6 gb/s, 0.5 ulp |        19.1 gb/s, 10 ulp |
| `nk_euclidean_bf16_genoa`     |       45.0 gb/s, 0.2 ulp |       21.5 gb/s, 0.3 ulp |       19.0 gb/s, 5.8 ulp |
| `nk_angular_bf16_genoa`       |         33.9 gb/s, 0 ulp |         20.9 gb/s, 0 ulp |       19.6 gb/s, 0.1 ulp |
| __f16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`   |      0.885 gb/s, 0.1 ulp |      0.812 gb/s, 0.1 ulp |      0.805 gb/s, 0.1 ulp |
| `nk_euclidean_f16_serial`     |      0.870 gb/s, 0.5 ulp |      0.850 gb/s, 0.5 ulp |      0.844 gb/s, 0.5 ulp |
| `nk_angular_f16_serial`       |        0.820 gb/s, 0 ulp |        0.495 gb/s, 0 ulp |        0.506 gb/s, 0 ulp |
| `nk_sqeuclidean_f16_haswell`  |       27.8 gb/s, 0.4 ulp |       13.8 gb/s, 1.4 ulp |       11.0 gb/s, 5.2 ulp |
| `nk_euclidean_f16_haswell`    |       21.3 gb/s, 0.3 ulp |       12.0 gb/s, 0.8 ulp |       9.87 gb/s, 2.8 ulp |
| `nk_angular_f16_haswell`      |       18.5 gb/s, 0.1 ulp |       16.3 gb/s, 0.1 ulp |       15.0 gb/s, 0.1 ulp |
| __e5m2__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e5m2_serial`  |        0.889 gb/s, 0 ulp |        0.941 gb/s, 0 ulp |        0.950 gb/s, 0 ulp |
| `nk_euclidean_e5m2_serial`    |      0.888 gb/s, 0.5 ulp |      0.917 gb/s, 0.5 ulp |      0.959 gb/s, 0.5 ulp |
| `nk_angular_e5m2_serial`      |        0.313 gb/s, 0 ulp |        0.359 gb/s, 0 ulp |        0.379 gb/s, 0 ulp |
| `nk_sqeuclidean_e5m2_skylake` |         4.14 gb/s, 0 ulp |         4.33 gb/s, 0 ulp |         5.40 gb/s, 0 ulp |
| `nk_euclidean_e5m2_skylake`   |         4.04 gb/s, 0 ulp |         4.33 gb/s, 0 ulp |         5.48 gb/s, 0 ulp |
| `nk_angular_e5m2_skylake`     |         3.57 gb/s, 0 ulp |         4.09 gb/s, 0 ulp |         5.68 gb/s, 0 ulp |
| __e4m3__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e4m3_serial`  |        0.530 gb/s, 0 ulp |        0.564 gb/s, 0 ulp |        0.567 gb/s, 0 ulp |
| `nk_euclidean_e4m3_serial`    |      0.547 gb/s, 0.5 ulp |      0.561 gb/s, 0.5 ulp |      0.538 gb/s, 0.5 ulp |
| `nk_angular_e4m3_serial`      |        0.304 gb/s, 0 ulp |        0.183 gb/s, 0 ulp |        0.341 gb/s, 0 ulp |
| `nk_sqeuclidean_e4m3_skylake` |         3.58 gb/s, 0 ulp |         3.37 gb/s, 0 ulp |       3.68 gb/s, 0.2 ulp |
| `nk_euclidean_e4m3_skylake`   |         3.24 gb/s, 0 ulp |         3.44 gb/s, 0 ulp |       3.10 gb/s, 0.2 ulp |
| `nk_angular_e4m3_skylake`     |         3.93 gb/s, 0 ulp |         3.15 gb/s, 0 ulp |         4.23 gb/s, 0 ulp |
| `nk_sqeuclidean_e4m3_icelake` |         9.50 gb/s, 0 ulp |         11.2 gb/s, 0 ulp |       11.2 gb/s, 0.2 ulp |
| `nk_euclidean_e4m3_icelake`   |         9.59 gb/s, 0 ulp |         11.0 gb/s, 0 ulp |       11.1 gb/s, 0.2 ulp |
| `nk_angular_e4m3_icelake`     |         8.18 gb/s, 0 ulp |         10.5 gb/s, 0 ulp |         11.1 gb/s, 0 ulp |
| __e3m2__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e3m2_serial`  |        0.941 gb/s, 0 ulp |        0.904 gb/s, 0 ulp |        0.959 gb/s, 0 ulp |
| `nk_euclidean_e3m2_serial`    |      0.929 gb/s, 0.5 ulp |      0.922 gb/s, 0.5 ulp |      0.930 gb/s, 0.4 ulp |
| `nk_angular_e3m2_serial`      |        0.309 gb/s, 0 ulp |        0.336 gb/s, 0 ulp |        0.407 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_skylake` |         4.16 gb/s, 0 ulp |         5.09 gb/s, 0 ulp |         4.69 gb/s, 0 ulp |
| `nk_euclidean_e3m2_skylake`   |         4.04 gb/s, 0 ulp |         5.77 gb/s, 0 ulp |         4.75 gb/s, 0 ulp |
| `nk_angular_e3m2_skylake`     |         3.53 gb/s, 0 ulp |         4.11 gb/s, 0 ulp |         4.49 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_icelake` |         19.7 gb/s, 0 ulp |         20.6 gb/s, 0 ulp |         20.4 gb/s, 0 ulp |
| `nk_euclidean_e3m2_icelake`   |         19.7 gb/s, 0 ulp |         21.3 gb/s, 0 ulp |         19.7 gb/s, 0 ulp |
| `nk_angular_e3m2_icelake`     |         13.1 gb/s, 0 ulp |         16.8 gb/s, 0 ulp |         16.4 gb/s, 0 ulp |
| __e2m3__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e2m3_serial`  |        0.898 gb/s, 0 ulp |        0.914 gb/s, 0 ulp |        0.959 gb/s, 0 ulp |
| `nk_euclidean_e2m3_serial`    |      0.912 gb/s, 0.5 ulp |      0.900 gb/s, 0.5 ulp |      0.950 gb/s, 0.5 ulp |
| `nk_angular_e2m3_serial`      |        0.323 gb/s, 0 ulp |        0.362 gb/s, 0 ulp |        0.389 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_skylake` |         4.27 gb/s, 0 ulp |         4.33 gb/s, 0 ulp |         4.73 gb/s, 0 ulp |
| `nk_euclidean_e2m3_skylake`   |         4.17 gb/s, 0 ulp |         4.09 gb/s, 0 ulp |         4.62 gb/s, 0 ulp |
| `nk_angular_e2m3_skylake`     |         3.67 gb/s, 0 ulp |         3.96 gb/s, 0 ulp |         4.56 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_icelake` |         47.2 gb/s, 0 ulp |         39.7 gb/s, 0 ulp |         28.9 gb/s, 0 ulp |
| `nk_euclidean_e2m3_icelake`   |         46.8 gb/s, 0 ulp |         41.3 gb/s, 0 ulp |         28.9 gb/s, 0 ulp |
| `nk_angular_e2m3_icelake`     |         25.3 gb/s, 0 ulp |         32.5 gb/s, 0 ulp |         28.4 gb/s, 0 ulp |
| __i8__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`    |                31.7 gb/s |                17.1 gb/s |                15.4 gb/s |
| `nk_euclidean_i8_serial`      |       27.0 gb/s, 0.4 ulp |       16.8 gb/s, 0.4 ulp |       14.5 gb/s, 0.4 ulp |
| `nk_angular_i8_serial`        |         7.34 gb/s, 0 ulp |         5.88 gb/s, 0 ulp |         5.70 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_haswell`   |                35.8 gb/s |                16.7 gb/s |                17.1 gb/s |
| `nk_euclidean_i8_haswell`     |         33.2 gb/s, 0 ulp |         15.8 gb/s, 0 ulp |         14.4 gb/s, 0 ulp |
| `nk_angular_i8_haswell`       |       18.9 gb/s, 0.1 ulp |         12.0 gb/s, 0 ulp |         11.1 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_icelake`   |                56.1 gb/s |                22.8 gb/s |                21.9 gb/s |
| `nk_euclidean_i8_icelake`     |         54.9 gb/s, 0 ulp |         21.4 gb/s, 0 ulp |         20.8 gb/s, 0 ulp |
| `nk_angular_i8_icelake`       |       23.5 gb/s, 0.1 ulp |         17.1 gb/s, 0 ulp |         19.1 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_alder`     |                31.1 gb/s |                16.2 gb/s |                16.4 gb/s |
| `nk_euclidean_i8_alder`       |         29.7 gb/s, 0 ulp |         17.8 gb/s, 0 ulp |         16.6 gb/s, 0 ulp |
| `nk_angular_i8_alder`         |       24.4 gb/s, 0.1 ulp |         15.9 gb/s, 0 ulp |         16.6 gb/s, 0 ulp |
| __u8__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`    |                10.9 gb/s |                8.17 gb/s |                6.58 gb/s |
| `nk_euclidean_u8_serial`      |       10.8 gb/s, 0.5 ulp |       7.74 gb/s, 0.5 ulp |       7.79 gb/s, 0.6 ulp |
| `nk_angular_u8_serial`        |       7.40 gb/s, 0.4 ulp |       6.22 gb/s, 0.4 ulp |       5.48 gb/s, 0.4 ulp |
| `nk_sqeuclidean_u8_haswell`   |                42.3 gb/s |                16.5 gb/s |                17.2 gb/s |
| `nk_euclidean_u8_haswell`     |         36.2 gb/s, 0 ulp |         17.5 gb/s, 0 ulp |         18.0 gb/s, 0 ulp |
| `nk_angular_u8_haswell`       |       20.4 gb/s, 0.7 ulp |       10.9 gb/s, 0.6 ulp |       12.5 gb/s, 0.5 ulp |
| `nk_sqeuclidean_u8_icelake`   |                65.3 gb/s |                26.8 gb/s |                19.6 gb/s |
| `nk_euclidean_u8_icelake`     |         61.8 gb/s, 0 ulp |         25.7 gb/s, 0 ulp |         21.9 gb/s, 0 ulp |
| `nk_angular_u8_icelake`       |       26.9 gb/s, 0.7 ulp |       19.7 gb/s, 0.6 ulp |       20.0 gb/s, 0.5 ulp |
| `nk_sqeuclidean_u8_alder`     |                30.0 gb/s |                16.3 gb/s |                17.7 gb/s |
| `nk_euclidean_u8_alder`       |         29.2 gb/s, 0 ulp |         15.8 gb/s, 0 ulp |         18.3 gb/s, 0 ulp |
| `nk_angular_u8_alder`         |       24.7 gb/s, 0.7 ulp |       15.9 gb/s, 0.6 ulp |       16.3 gb/s, 0.5 ulp |
| __i4__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i4_serial`    |                14.3 gb/s |                15.4 gb/s |                14.5 gb/s |
| `nk_euclidean_i4_serial`      |       11.4 gb/s, 0.5 ulp |       14.5 gb/s, 0.5 ulp |       14.2 gb/s, 0.6 ulp |
| `nk_angular_i4_serial`        |       5.22 gb/s, 0.4 ulp |       5.98 gb/s, 0.4 ulp |       6.23 gb/s, 0.4 ulp |
| `nk_sqeuclidean_i4_icelake`   |                22.0 gb/s |                48.0 gb/s |                27.3 gb/s |
| `nk_euclidean_i4_icelake`     |         19.2 gb/s, 0 ulp |         42.1 gb/s, 0 ulp |         26.9 gb/s, 0 ulp |
| `nk_angular_i4_icelake`       |       4.79 gb/s, 0.7 ulp |       16.8 gb/s, 0.6 ulp |       16.4 gb/s, 0.5 ulp |
| __u4__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u4_serial`    |                14.5 gb/s |                16.1 gb/s |                14.7 gb/s |
| `nk_euclidean_u4_serial`      |       11.2 gb/s, 0.5 ulp |       14.8 gb/s, 0.5 ulp |       14.2 gb/s, 0.6 ulp |
| `nk_angular_u4_serial`        |       4.84 gb/s, 0.4 ulp |       6.17 gb/s, 0.4 ulp |       6.53 gb/s, 0.4 ulp |
| `nk_sqeuclidean_u4_icelake`   |                21.1 gb/s |                22.1 gb/s |                22.8 gb/s |
| `nk_euclidean_u4_icelake`     |         19.5 gb/s, 0 ulp |         17.5 gb/s, 0 ulp |         22.4 gb/s, 0 ulp |
| `nk_angular_u4_icelake`       |       8.68 gb/s, 0.7 ulp |       25.5 gb/s, 0.6 ulp |       22.5 gb/s, 0.5 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                            |                      256 |                     1024 |                     4096 |
| :-------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`       |       2.77 gb/s, 0.1 ulp |         2.94 gb/s, 0 ulp |         0.02 gb/s, 0 ulp |
| `nk_euclidean_f64_serial`         |     0.0969 gb/s, 0.6 ulp |      0.987 gb/s, 0.6 ulp |       0.31 gb/s, 0.5 ulp |
| `nk_angular_f64_serial`           |       1.78 gb/s, 0.1 ulp |         1.80 gb/s, 0 ulp |         0.17 gb/s, 0 ulp |
| `nk_sqeuclidean_f64_v128relaxed`  |       1.15 gb/s, 1.3 ulp |       1.74 gb/s, 2.5 ulp |       0.14 gb/s, 5.0 ulp |
| `nk_euclidean_f64_v128relaxed`    |      0.293 gb/s, 0.7 ulp |       2.06 gb/s, 1.4 ulp |       0.03 gb/s, 2.8 ulp |
| `nk_angular_f64_v128relaxed`      |       1.06 gb/s, 0.1 ulp |      0.864 gb/s, 0.1 ulp |       0.24 gb/s, 0.1 ulp |
| __f32__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`       |        0.612 gb/s, 0 ulp |        0.864 gb/s, 0 ulp |         0.06 gb/s, 0 ulp |
| `nk_euclidean_f32_serial`         |      0.705 gb/s, 0.1 ulp |      0.851 gb/s, 0.1 ulp |       0.05 gb/s, 0.1 ulp |
| `nk_angular_f32_serial`           |        0.821 gb/s, 0 ulp |        0.840 gb/s, 0 ulp |         0.24 gb/s, 0 ulp |
| `nk_sqeuclidean_f32_v128relaxed`  |       2.67 gb/s, 0.7 ulp |       2.82 gb/s, 1.3 ulp |       1.65 gb/s, 2.6 ulp |
| `nk_euclidean_f32_v128relaxed`    |       1.70 gb/s, 0.4 ulp |       2.79 gb/s, 0.7 ulp |       0.20 gb/s, 1.4 ulp |
| `nk_angular_f32_v128relaxed`      |         3.14 gb/s, 0 ulp |        0.923 gb/s, 0 ulp |         0.18 gb/s, 0 ulp |
| __bf16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`      |         1.76 gb/s, 0 ulp |         1.02 gb/s, 0 ulp |         0.29 gb/s, 0 ulp |
| `nk_euclidean_bf16_serial`        |       1.88 gb/s, 0.6 ulp |       1.98 gb/s, 0.5 ulp |       0.27 gb/s, 0.5 ulp |
| `nk_angular_bf16_serial`          |        0.372 gb/s, 0 ulp |        0.287 gb/s, 0 ulp |         0.10 gb/s, 0 ulp |
| `nk_sqeuclidean_bf16_v128relaxed` |       1.96 gb/s, 0.9 ulp |      1.81 gb/s, 12.6 ulp |      0.16 gb/s, 20.8 ulp |
| `nk_euclidean_bf16_v128relaxed`   |       1.94 gb/s, 0.5 ulp |       2.07 gb/s, 7.0 ulp |      0.12 gb/s, 11.4 ulp |
| `nk_angular_bf16_v128relaxed`     |         1.01 gb/s, 0 ulp |       1.95 gb/s, 0.2 ulp |       0.19 gb/s, 0.6 ulp |
| `nk_sqeuclidean_bf16_v128`        |                        … |                        … |                        … |
| `nk_euclidean_bf16_v128`          |                        … |                        … |                        … |
| `nk_angular_bf16_v128`            |                        … |                        … |                        … |
| __f16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`       |       1.02 gb/s, 0.1 ulp |       1.05 gb/s, 0.1 ulp |       0.19 gb/s, 0.1 ulp |
| `nk_euclidean_f16_serial`         |       1.09 gb/s, 0.6 ulp |       1.08 gb/s, 0.6 ulp |       0.24 gb/s, 0.5 ulp |
| `nk_angular_f16_serial`           |        0.338 gb/s, 0 ulp |        0.346 gb/s, 0 ulp |         0.06 gb/s, 0 ulp |
| `nk_sqeuclidean_f16_v128relaxed`  |       1.04 gb/s, 0.9 ulp |      0.590 gb/s, 3.6 ulp |       0.03 gb/s, 9.7 ulp |
| `nk_euclidean_f16_v128relaxed`    |      0.751 gb/s, 0.5 ulp |      0.923 gb/s, 2.0 ulp |       0.08 gb/s, 5.4 ulp |
| `nk_angular_f16_v128relaxed`      |       1.67 gb/s, 0.1 ulp |      0.909 gb/s, 0.1 ulp |       0.00 gb/s, 0.1 ulp |
| __e5m2__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e5m2_serial`      |        0.664 gb/s, 0 ulp |        0.642 gb/s, 0 ulp |         0.15 gb/s, 0 ulp |
| `nk_euclidean_e5m2_serial`        |      0.593 gb/s, 0.5 ulp |      0.685 gb/s, 0.5 ulp |       0.11 gb/s, 0.5 ulp |
| `nk_angular_e5m2_serial`          |        0.157 gb/s, 0 ulp |        0.151 gb/s, 0 ulp |         0.16 gb/s, 0 ulp |
| __e4m3__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e4m3_serial`      |        0.348 gb/s, 0 ulp |        0.357 gb/s, 0 ulp |         0.08 gb/s, 0 ulp |
| `nk_euclidean_e4m3_serial`        |      0.348 gb/s, 0.5 ulp |      0.335 gb/s, 0.5 ulp |       0.08 gb/s, 0.5 ulp |
| `nk_angular_e4m3_serial`          |        0.151 gb/s, 0 ulp |        0.155 gb/s, 0 ulp |         0.16 gb/s, 0 ulp |
| __e3m2__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e3m2_serial`      |        0.663 gb/s, 0 ulp |        0.693 gb/s, 0 ulp |         0.16 gb/s, 0 ulp |
| `nk_euclidean_e3m2_serial`        |      0.660 gb/s, 0.5 ulp |      0.707 gb/s, 0.5 ulp |       0.16 gb/s, 0.5 ulp |
| `nk_angular_e3m2_serial`          |        0.142 gb/s, 0 ulp |        0.154 gb/s, 0 ulp |         0.16 gb/s, 0 ulp |
| __e2m3__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e2m3_serial`      |        0.654 gb/s, 0 ulp |        0.708 gb/s, 0 ulp |         0.12 gb/s, 0 ulp |
| `nk_euclidean_e2m3_serial`        |      0.605 gb/s, 0.5 ulp |      0.701 gb/s, 0.5 ulp |       0.14 gb/s, 0.5 ulp |
| `nk_angular_e2m3_serial`          |        0.147 gb/s, 0 ulp |        0.156 gb/s, 0 ulp |         0.16 gb/s, 0 ulp |
| __i8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`        |               0.305 gb/s |               0.305 gb/s |                0.08 gb/s |
| `nk_euclidean_i8_serial`          |       2.73 gb/s, 0.5 ulp |      0.162 gb/s, 0.4 ulp |       0.13 gb/s, 0.4 ulp |
| `nk_angular_i8_serial`            |         1.15 gb/s, 0 ulp |        0.881 gb/s, 0 ulp |        0.093 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_v128relaxed`   |                        … |                        … |                        … |
| `nk_euclidean_i8_v128relaxed`     |                        … |                        … |                        … |
| `nk_angular_i8_v128relaxed`       |         1.68 gb/s, 0 ulp |         2.60 gb/s, 0 ulp |         0.13 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_v128`          |                1.71 gb/s |               0.685 gb/s |                0.07 gb/s |
| `nk_euclidean_i8_v128`            |         1.27 gb/s, 0 ulp |        0.750 gb/s, 0 ulp |         0.20 gb/s, 0 ulp |
| `nk_angular_i8_v128`              |                        … |                        … |                        … |
| __u8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`        |               0.492 gb/s |               0.462 gb/s |                0.28 gb/s |
| `nk_euclidean_u8_serial`          |    0.00915 gb/s, 0.5 ulp |      0.290 gb/s, 0.5 ulp |       0.04 gb/s, 0.6 ulp |
| `nk_angular_u8_serial`            |      0.757 gb/s, 0.5 ulp |       1.36 gb/s, 0.4 ulp |       0.27 gb/s, 0.5 ulp |
| `nk_sqeuclidean_u8_v128relaxed`   |                        … |                        … |                        … |
| `nk_euclidean_u8_v128relaxed`     |                        … |                        … |                        … |
| `nk_angular_u8_v128relaxed`       |      2.30 gb/s, 526M ulp |      1.78 gb/s, 501M ulp |      0.08 gb/s, 443M ulp |
| `nk_sqeuclidean_u8_v128`          |                2.84 gb/s |                1.56 gb/s |                0.26 gb/s |
| `nk_euclidean_u8_v128`            |         2.35 gb/s, 0 ulp |         1.58 gb/s, 0 ulp |         0.08 gb/s, 0 ulp |
| `nk_angular_u8_v128`              |                        … |                        … |                        … |
| __i4__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i4_serial`        |                1.78 gb/s |                1.81 gb/s |                0.28 gb/s |
| `nk_euclidean_i4_serial`          |       1.64 gb/s, 0.5 ulp |       1.77 gb/s, 0.5 ulp |       0.02 gb/s, 0.0 ulp |
| `nk_angular_i4_serial`            |       1.19 gb/s, 0.5 ulp |       1.25 gb/s, 0.5 ulp |      0.093 gb/s, 0.5 ulp |
| __u4__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u4_serial`        |                2.71 gb/s |                2.79 gb/s |                0.08 gb/s |
| `nk_euclidean_u4_serial`          |       2.59 gb/s, 0.5 ulp |       2.80 gb/s, 0.5 ulp |      0.093 gb/s, 0.0 ulp |
| `nk_angular_u4_serial`            |       1.71 gb/s, 0.5 ulp |       1.89 gb/s, 0.5 ulp |       0.20 gb/s, 0.5 ulp |

### Apple M5

#### Native

| Kernel                          |                      256 |                     1024 |                     4096 |
| :------------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`     |       11.5 gb/s, 0.1 ulp |         11.9 gb/s, 0 ulp |         11.9 gb/s, 0 ulp |
| `nk_euclidean_f64_serial`       |       11.8 gb/s, 0.6 ulp |       12.0 gb/s, 0.5 ulp |       11.7 gb/s, 0.5 ulp |
| `nk_angular_f64_serial`         |         7.84 gb/s, 0 ulp |         7.98 gb/s, 0 ulp |         7.73 gb/s, 0 ulp |
| `nk_sqeuclidean_f64_neon`       |       47.1 gb/s, 1.3 ulp |       37.3 gb/s, 2.6 ulp |       33.6 gb/s, 5.1 ulp |
| `nk_euclidean_f64_neon`         |       45.1 gb/s, 0.7 ulp |       36.0 gb/s, 1.4 ulp |       32.7 gb/s, 2.8 ulp |
| `nk_angular_f64_neon`           |       31.0 gb/s, 0.1 ulp |         31.1 gb/s, 0 ulp |         30.2 gb/s, 0 ulp |
| __f32__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`     |         5.89 gb/s, 0 ulp |         5.82 gb/s, 0 ulp |         5.87 gb/s, 0 ulp |
| `nk_euclidean_f32_serial`       |       5.88 gb/s, 0.1 ulp |       5.93 gb/s, 0.1 ulp |       5.97 gb/s, 0.1 ulp |
| `nk_angular_f32_serial`         |         3.75 gb/s, 0 ulp |         3.78 gb/s, 0 ulp |         3.79 gb/s, 0 ulp |
| `nk_sqeuclidean_f32_neon`       |       23.6 gb/s, 0.1 ulp |         17.8 gb/s, 0 ulp |         16.3 gb/s, 0 ulp |
| `nk_euclidean_f32_neon`         |       23.3 gb/s, 0.1 ulp |       19.4 gb/s, 0.1 ulp |       17.3 gb/s, 0.1 ulp |
| `nk_angular_f32_neon`           |         20.7 gb/s, 0 ulp |         16.1 gb/s, 0 ulp |         15.5 gb/s, 0 ulp |
| __bf16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`    |         2.97 gb/s, 0 ulp |         2.94 gb/s, 0 ulp |         2.92 gb/s, 0 ulp |
| `nk_euclidean_bf16_serial`      |       2.94 gb/s, 0.5 ulp |       2.87 gb/s, 0.5 ulp |       2.90 gb/s, 0.5 ulp |
| `nk_angular_bf16_serial`        |         1.75 gb/s, 0 ulp |         1.78 gb/s, 0 ulp |         1.80 gb/s, 0 ulp |
| `nk_sqeuclidean_bf16_neonbfdot` |       32.6 gb/s, 0.9 ulp |        21.1 gb/s, 13 ulp |        17.5 gb/s, 21 ulp |
| `nk_euclidean_bf16_neonbfdot`   |       31.1 gb/s, 0.5 ulp |       21.4 gb/s, 7.0 ulp |        17.3 gb/s, 12 ulp |
| `nk_angular_bf16_neonbfdot`     |         22.2 gb/s, 0 ulp |       30.5 gb/s, 0.1 ulp |         33.4 gb/s, 0 ulp |
| __f16__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`     |       2.88 gb/s, 0.1 ulp |       2.94 gb/s, 0.1 ulp |       2.89 gb/s, 0.1 ulp |
| `nk_euclidean_f16_serial`       |       2.92 gb/s, 0.6 ulp |       2.92 gb/s, 0.5 ulp |       2.90 gb/s, 0.5 ulp |
| `nk_angular_f16_serial`         |         1.71 gb/s, 0 ulp |         1.79 gb/s, 0 ulp |         1.75 gb/s, 0 ulp |
| `nk_sqeuclidean_f16_neon`       |       32.3 gb/s, 0.9 ulp |       20.0 gb/s, 3.6 ulp |       17.0 gb/s, 9.7 ulp |
| `nk_euclidean_f16_neon`         |       30.5 gb/s, 0.5 ulp |       20.2 gb/s, 2.0 ulp |       17.1 gb/s, 5.3 ulp |
| `nk_angular_f16_neon`           |       23.5 gb/s, 0.1 ulp |       18.3 gb/s, 0.1 ulp |       16.1 gb/s, 0.1 ulp |
| __e5m2__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e5m2_serial`    |         1.95 gb/s, 0 ulp |         1.94 gb/s, 0 ulp |         1.96 gb/s, 0 ulp |
| `nk_euclidean_e5m2_serial`      |       1.92 gb/s, 0.5 ulp |       1.96 gb/s, 0.5 ulp |       1.91 gb/s, 0.5 ulp |
| `nk_angular_e5m2_serial`        |        0.858 gb/s, 0 ulp |        0.890 gb/s, 0 ulp |        0.874 gb/s, 0 ulp |
| `nk_sqeuclidean_e5m2_neon`      |         17.0 gb/s, 0 ulp |         11.9 gb/s, 0 ulp |         9.16 gb/s, 0 ulp |
| `nk_euclidean_e5m2_neon`        |       16.8 gb/s, 0.5 ulp |       11.0 gb/s, 0.5 ulp |       8.69 gb/s, 0.5 ulp |
| `nk_angular_e5m2_neon`          |         12.8 gb/s, 0 ulp |         10.2 gb/s, 0 ulp |         9.15 gb/s, 0 ulp |
| __e4m3__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e4m3_serial`    |        0.997 gb/s, 0 ulp |         1.04 gb/s, 0 ulp |         1.03 gb/s, 0 ulp |
| `nk_euclidean_e4m3_serial`      |      0.941 gb/s, 0.5 ulp |       1.04 gb/s, 0.5 ulp |       1.02 gb/s, 0.5 ulp |
| `nk_angular_e4m3_serial`        |        0.662 gb/s, 0 ulp |        0.682 gb/s, 0 ulp |        0.679 gb/s, 0 ulp |
| `nk_sqeuclidean_e4m3_neon`      |       4.00 gb/s, 0.2 ulp |       4.06 gb/s, 0.2 ulp |       4.03 gb/s, 0.2 ulp |
| `nk_euclidean_e4m3_neon`        |       3.91 gb/s, 0.5 ulp |       3.83 gb/s, 0.5 ulp |       3.88 gb/s, 0.5 ulp |
| `nk_angular_e4m3_neon`          |         3.85 gb/s, 0 ulp |         3.92 gb/s, 0 ulp |         3.87 gb/s, 0 ulp |
| __e3m2__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e3m2_serial`    |         1.82 gb/s, 0 ulp |         2.00 gb/s, 0 ulp |         1.94 gb/s, 0 ulp |
| `nk_euclidean_e3m2_serial`      |       1.83 gb/s, 0.5 ulp |       2.03 gb/s, 0.5 ulp |       1.95 gb/s, 0.5 ulp |
| `nk_angular_e3m2_serial`        |        0.838 gb/s, 0 ulp |        0.917 gb/s, 0 ulp |        0.878 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_neon`      |         4.41 gb/s, 0 ulp |         4.83 gb/s, 0 ulp |         4.68 gb/s, 0 ulp |
| `nk_euclidean_e3m2_neon`        |         4.45 gb/s, 0 ulp |         4.87 gb/s, 0 ulp |         4.70 gb/s, 0 ulp |
| `nk_angular_e3m2_neon`          |         3.95 gb/s, 0 ulp |         4.52 gb/s, 0 ulp |         4.41 gb/s, 0 ulp |
| __e2m3__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e2m3_serial`    |         1.84 gb/s, 0 ulp |         2.05 gb/s, 0 ulp |         1.97 gb/s, 0 ulp |
| `nk_euclidean_e2m3_serial`      |       1.78 gb/s, 0.5 ulp |       2.01 gb/s, 0.5 ulp |       1.95 gb/s, 0.4 ulp |
| `nk_angular_e2m3_serial`        |        0.824 gb/s, 0 ulp |        0.917 gb/s, 0 ulp |        0.888 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_neon`      |         4.35 gb/s, 0 ulp |         4.71 gb/s, 0 ulp |         4.72 gb/s, 0 ulp |
| `nk_euclidean_e2m3_neon`        |         4.51 gb/s, 0 ulp |         4.81 gb/s, 0 ulp |         4.64 gb/s, 0 ulp |
| `nk_angular_e2m3_neon`          |         4.14 gb/s, 0 ulp |         4.54 gb/s, 0 ulp |         4.41 gb/s, 0 ulp |
| __i8__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`      |                57.4 gb/s |                70.0 gb/s |                60.4 gb/s |
| `nk_euclidean_i8_serial`        |                40.5 gb/s |                50.3 gb/s |                59.9 gb/s |
| `nk_angular_i8_serial`          |                52.0 gb/s |                59.3 gb/s |                46.2 gb/s |
| `nk_sqeuclidean_i8_neonsdot`    |                83.0 gb/s |                80.0 gb/s |                54.8 gb/s |
| `nk_euclidean_i8_neonsdot`      |                80.9 gb/s |                73.5 gb/s |                53.5 gb/s |
| `nk_angular_i8_neonsdot`        |                61.9 gb/s |                64.2 gb/s |                47.4 gb/s |
| __u8__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`      |                58.6 gb/s |                71.9 gb/s |                61.9 gb/s |
| `nk_euclidean_u8_serial`        |                42.6 gb/s |                48.7 gb/s |                57.2 gb/s |
| `nk_angular_u8_serial`          |                16.6 gb/s |                17.2 gb/s |                14.9 gb/s |
| `nk_sqeuclidean_u8_neonsdot`    |                85.4 gb/s |                77.4 gb/s |                52.7 gb/s |
| `nk_euclidean_u8_neonsdot`      |                81.9 gb/s |                73.9 gb/s |                52.6 gb/s |
| `nk_angular_u8_neonsdot`        |                63.3 gb/s |                60.3 gb/s |                46.1 gb/s |
| __i4__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i4_serial`      |                21.3 gb/s |                23.5 gb/s |                23.4 gb/s |
| `nk_euclidean_i4_serial`        |                18.7 gb/s |                22.0 gb/s |                22.5 gb/s |
| `nk_angular_i4_serial`          |                8.48 gb/s |                9.69 gb/s |                9.69 gb/s |
| __u4__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u4_serial`      |                24.9 gb/s |                25.0 gb/s |                20.9 gb/s |
| `nk_euclidean_u4_serial`        |                19.5 gb/s |                21.0 gb/s |                19.7 gb/s |
| `nk_angular_u4_serial`          |                8.38 gb/s |                8.91 gb/s |                8.96 gb/s |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                            |                      256 |                     1024 |                     4096 |
| :-------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`       |       18.9 gb/s, 0.1 ulp |         18.0 gb/s, 0 ulp |         18.9 gb/s, 0 ulp |
| `nk_euclidean_f64_serial`         |       18.6 gb/s, 0.6 ulp |       18.1 gb/s, 0.6 ulp |       18.9 gb/s, 0.5 ulp |
| `nk_angular_f64_serial`           |         8.64 gb/s, 0 ulp |         8.22 gb/s, 0 ulp |         8.65 gb/s, 0 ulp |
| `nk_sqeuclidean_f64_v128relaxed`  |       44.9 gb/s, 1.3 ulp |       33.2 gb/s, 2.6 ulp |       34.5 gb/s, 5.0 ulp |
| `nk_euclidean_f64_v128relaxed`    |       46.7 gb/s, 0.7 ulp |       34.0 gb/s, 1.4 ulp |       34.2 gb/s, 2.8 ulp |
| `nk_angular_f64_v128relaxed`      |       29.3 gb/s, 0.1 ulp |       20.8 gb/s, 0.1 ulp |       20.9 gb/s, 0.1 ulp |
| __f32__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`       |         8.29 gb/s, 0 ulp |         7.95 gb/s, 0 ulp |         8.25 gb/s, 0 ulp |
| `nk_euclidean_f32_serial`         |       8.25 gb/s, 0.1 ulp |       7.99 gb/s, 0.1 ulp |       8.25 gb/s, 0.1 ulp |
| `nk_angular_f32_serial`           |         4.03 gb/s, 0 ulp |         3.88 gb/s, 0 ulp |         4.00 gb/s, 0 ulp |
| `nk_sqeuclidean_f32_v128relaxed`  |       19.0 gb/s, 0.7 ulp |       16.7 gb/s, 1.3 ulp |       17.0 gb/s, 2.6 ulp |
| `nk_euclidean_f32_v128relaxed`    |       18.9 gb/s, 0.4 ulp |       16.8 gb/s, 0.7 ulp |       17.0 gb/s, 1.4 ulp |
| `nk_angular_f32_v128relaxed`      |         18.3 gb/s, 0 ulp |         16.6 gb/s, 0 ulp |         17.2 gb/s, 0 ulp |
| __bf16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`      |         4.81 gb/s, 0 ulp |         4.54 gb/s, 0 ulp |         4.75 gb/s, 0 ulp |
| `nk_euclidean_bf16_serial`        |       4.77 gb/s, 0.6 ulp |       4.55 gb/s, 0.5 ulp |       4.75 gb/s, 0.5 ulp |
| `nk_angular_bf16_serial`          |         2.09 gb/s, 0 ulp |         2.00 gb/s, 0 ulp |         2.09 gb/s, 0 ulp |
| `nk_sqeuclidean_bf16_v128relaxed` |       37.2 gb/s, 0.9 ulp |        25.1 gb/s, 13 ulp |        18.9 gb/s, 21 ulp |
| `nk_euclidean_bf16_v128relaxed`   |       35.9 gb/s, 0.5 ulp |       25.2 gb/s, 7.0 ulp |        19.7 gb/s, 12 ulp |
| `nk_angular_bf16_v128relaxed`     |         26.0 gb/s, 0 ulp |       21.0 gb/s, 0.2 ulp |       19.1 gb/s, 0.6 ulp |
| `nk_sqeuclidean_bf16_v128`        |                        … |                        … |                        … |
| `nk_euclidean_bf16_v128`          |                        … |                        … |                        … |
| `nk_angular_bf16_v128`            |                        … |                        … |                        … |
| __f16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`       |       3.00 gb/s, 0.1 ulp |       2.85 gb/s, 0.1 ulp |       2.88 gb/s, 0.1 ulp |
| `nk_euclidean_f16_serial`         |       2.97 gb/s, 0.6 ulp |       2.72 gb/s, 0.5 ulp |       3.04 gb/s, 0.5 ulp |
| `nk_angular_f16_serial`           |         2.17 gb/s, 0 ulp |         2.06 gb/s, 0 ulp |         2.16 gb/s, 0 ulp |
| `nk_sqeuclidean_f16_v128relaxed`  |       10.4 gb/s, 0.9 ulp |       10.2 gb/s, 3.6 ulp |       11.0 gb/s, 9.6 ulp |
| `nk_euclidean_f16_v128relaxed`    |       10.5 gb/s, 0.5 ulp |       10.2 gb/s, 2.0 ulp |       11.0 gb/s, 5.3 ulp |
| `nk_angular_f16_v128relaxed`      |       8.76 gb/s, 0.1 ulp |       8.90 gb/s, 0.1 ulp |       9.69 gb/s, 0.1 ulp |
| __i8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`        |                13.9 gb/s |                13.7 gb/s |                15.4 gb/s |
| `nk_euclidean_i8_serial`          |       13.7 gb/s, 0.5 ulp |       13.8 gb/s, 0.4 ulp |       15.2 gb/s, 0.4 ulp |
| `nk_angular_i8_serial`            |         7.51 gb/s, 0 ulp |         7.84 gb/s, 0 ulp |         9.97 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_v128relaxed`   |                        … |                        … |                        … |
| `nk_euclidean_i8_v128relaxed`     |                        … |                        … |                        … |
| `nk_angular_i8_v128relaxed`       |         16.0 gb/s, 0 ulp |         16.9 gb/s, 0 ulp |         18.3 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_v128`          |                28.6 gb/s |                21.3 gb/s |                16.8 gb/s |
| `nk_euclidean_i8_v128`            |         25.5 gb/s, 0 ulp |         21.0 gb/s, 0 ulp |         16.7 gb/s, 0 ulp |
| `nk_angular_i8_v128`              |                        … |                        … |                        … |
| __u8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`        |                13.8 gb/s |                13.5 gb/s |                15.2 gb/s |
| `nk_euclidean_u8_serial`          |       13.5 gb/s, 0.5 ulp |       13.5 gb/s, 0.5 ulp |       14.9 gb/s, 0.6 ulp |
| `nk_angular_u8_serial`            |       7.32 gb/s, 0.5 ulp |       7.68 gb/s, 0.5 ulp |       9.97 gb/s, 0.4 ulp |
| `nk_sqeuclidean_u8_v128relaxed`   |                        … |                        … |                        … |
| `nk_euclidean_u8_v128relaxed`     |                        … |                        … |                        … |
| `nk_angular_u8_v128relaxed`       |         13.1 gb/s, 0 ulp |         14.0 gb/s, 0 ulp |         15.0 gb/s, 0 ulp |
| `nk_sqeuclidean_u8_v128`          |                30.9 gb/s |                22.9 gb/s |                17.0 gb/s |
| `nk_euclidean_u8_v128`            |         26.6 gb/s, 0 ulp |         22.1 gb/s, 0 ulp |         17.0 gb/s, 0 ulp |
| `nk_angular_u8_v128`              |                        … |                        … |                        … |
