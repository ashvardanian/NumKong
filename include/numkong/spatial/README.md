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

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.

#### Native

| Kernel                        |                      256 |                     1024 |                     4096 |
| :---------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`   |       3.56 gb/s, 0.1 ulp |         3.59 gb/s, 0 ulp |         1.85 gb/s, 0 ulp |
| `nk_euclidean_f64_serial`     |       3.60 gb/s, 0.6 ulp |       3.61 gb/s, 0.5 ulp |       1.84 gb/s, 0.5 ulp |
| `nk_angular_f64_serial`       |         1.40 gb/s, 0 ulp |         1.37 gb/s, 0 ulp |         1.22 gb/s, 0 ulp |
| `nk_sqeuclidean_f64_haswell`  |                15.7 gb/s |                23.4 gb/s |                13.2 gb/s |
| `nk_euclidean_f64_haswell`    |                15.5 gb/s |                23.4 gb/s |                19.6 gb/s |
| `nk_angular_f64_haswell`      |                10.5 gb/s |                12.8 gb/s |                3.17 gb/s |
| `nk_sqeuclidean_f64_skylake`  |       18.2 gb/s, 0.4 ulp |       24.5 gb/s, 0.7 ulp |       25.4 gb/s, 1.3 ulp |
| `nk_euclidean_f64_skylake`    |       18.2 gb/s, 0.3 ulp |       23.3 gb/s, 0.4 ulp |       25.2 gb/s, 0.7 ulp |
| `nk_angular_f64_skylake`      |         11.5 gb/s, 0 ulp |         13.0 gb/s, 0 ulp |         13.3 gb/s, 0 ulp |
| __f32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`   |         9.91 gb/s, 0 ulp |         10.5 gb/s, 0 ulp |         9.73 gb/s, 0 ulp |
| `nk_euclidean_f32_serial`     |       9.53 gb/s, 0.1 ulp |       10.6 gb/s, 0.1 ulp |       9.67 gb/s, 0.1 ulp |
| `nk_angular_f32_serial`       |         6.72 gb/s, 0 ulp |         7.63 gb/s, 0 ulp |         6.94 gb/s, 0 ulp |
| `nk_sqeuclidean_f32_haswell`  |                17.2 gb/s |                17.6 gb/s |                17.6 gb/s |
| `nk_euclidean_f32_haswell`    |                18.8 gb/s |                17.9 gb/s |                18.2 gb/s |
| `nk_angular_f32_haswell`      |                16.0 gb/s |                19.8 gb/s |                16.2 gb/s |
| `nk_sqeuclidean_f32_skylake`  |         25.5 gb/s, 0 ulp |         24.3 gb/s, 0 ulp |         26.3 gb/s, 0 ulp |
| `nk_euclidean_f32_skylake`    |       24.9 gb/s, 0.1 ulp |       25.4 gb/s, 0.1 ulp |       26.1 gb/s, 0.1 ulp |
| `nk_angular_f32_skylake`      |         22.5 gb/s, 0 ulp |         22.8 gb/s, 0 ulp |         26.1 gb/s, 0 ulp |
| __bf16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`  |         4.74 gb/s, 0 ulp |         3.27 gb/s, 0 ulp |         4.99 gb/s, 0 ulp |
| `nk_euclidean_bf16_serial`    |       4.68 gb/s, 0.5 ulp |       3.41 gb/s, 0.5 ulp |       4.72 gb/s, 0.4 ulp |
| `nk_angular_bf16_serial`      |         3.55 gb/s, 0 ulp |         2.99 gb/s, 0 ulp |         3.72 gb/s, 0 ulp |
| `nk_sqeuclidean_bf16_haswell` |       30.4 gb/s, 0.5 ulp |       18.7 gb/s, 7.5 ulp |        18.5 gb/s, 27 ulp |
| `nk_euclidean_bf16_haswell`   |       28.9 gb/s, 0.3 ulp |       18.3 gb/s, 4.1 ulp |        18.0 gb/s, 15 ulp |
| `nk_angular_bf16_haswell`     |         21.2 gb/s, 0 ulp |         16.0 gb/s, 0 ulp |       17.5 gb/s, 0.2 ulp |
| `nk_sqeuclidean_bf16_genoa`   |       38.8 gb/s, 0.3 ulp |       21.7 gb/s, 0.5 ulp |        25.4 gb/s, 10 ulp |
| `nk_euclidean_bf16_genoa`     |       36.4 gb/s, 0.2 ulp |       21.7 gb/s, 0.3 ulp |       25.4 gb/s, 5.8 ulp |
| `nk_angular_bf16_genoa`       |         30.3 gb/s, 0 ulp |         20.7 gb/s, 0 ulp |       25.2 gb/s, 0.1 ulp |
| __f16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`   |       1.39 gb/s, 0.1 ulp |      0.984 gb/s, 0.1 ulp |       1.39 gb/s, 0.1 ulp |
| `nk_euclidean_f16_serial`     |       1.40 gb/s, 0.5 ulp |      0.972 gb/s, 0.5 ulp |       1.29 gb/s, 0.5 ulp |
| `nk_angular_f16_serial`       |         1.30 gb/s, 0 ulp |        0.959 gb/s, 0 ulp |         1.16 gb/s, 0 ulp |
| `nk_sqeuclidean_f16_haswell`  |       29.3 gb/s, 0.4 ulp |       13.5 gb/s, 1.4 ulp |       17.9 gb/s, 5.2 ulp |
| `nk_euclidean_f16_haswell`    |       25.1 gb/s, 0.3 ulp |       13.2 gb/s, 0.8 ulp |       17.8 gb/s, 2.8 ulp |
| `nk_angular_f16_haswell`      |       21.5 gb/s, 0.1 ulp |       12.2 gb/s, 0.1 ulp |       17.1 gb/s, 0.1 ulp |
| `nk_sqeuclidean_f16_skylake`  |                39.8 gb/s |                23.1 gb/s |                26.1 gb/s |
| `nk_euclidean_f16_skylake`    |                39.3 gb/s |                22.9 gb/s |                26.0 gb/s |
| `nk_angular_f16_skylake`      |                25.0 gb/s |                20.8 gb/s |                25.6 gb/s |
| __e5m2__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e5m2_serial`  |         1.65 gb/s, 0 ulp |         1.13 gb/s, 0 ulp |         1.72 gb/s, 0 ulp |
| `nk_euclidean_e5m2_serial`    |       1.06 gb/s, 0.5 ulp |       1.13 gb/s, 0.5 ulp |       1.54 gb/s, 0.5 ulp |
| `nk_angular_e5m2_serial`      |         1.35 gb/s, 0 ulp |        0.968 gb/s, 0 ulp |         1.24 gb/s, 0 ulp |
| `nk_sqeuclidean_e5m2_haswell` |                14.6 gb/s |                11.7 gb/s |                13.2 gb/s |
| `nk_euclidean_e5m2_haswell`   |                13.5 gb/s |                11.6 gb/s |                13.7 gb/s |
| `nk_angular_e5m2_haswell`     |                9.05 gb/s |                8.71 gb/s |                9.00 gb/s |
| `nk_sqeuclidean_e5m2_skylake` |         17.7 gb/s, 0 ulp |         14.4 gb/s, 0 ulp |         11.8 gb/s, 0 ulp |
| `nk_euclidean_e5m2_skylake`   |         17.0 gb/s, 0 ulp |         14.5 gb/s, 0 ulp |         11.8 gb/s, 0 ulp |
| `nk_angular_e5m2_skylake`     |         12.6 gb/s, 0 ulp |         12.7 gb/s, 0 ulp |         11.0 gb/s, 0 ulp |
| __e4m3__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e4m3_serial`  |        0.714 gb/s, 0 ulp |        0.591 gb/s, 0 ulp |        0.674 gb/s, 0 ulp |
| `nk_euclidean_e4m3_serial`    |      0.528 gb/s, 0.5 ulp |      0.768 gb/s, 0.5 ulp |      0.684 gb/s, 0.5 ulp |
| `nk_angular_e4m3_serial`      |        0.549 gb/s, 0 ulp |        0.490 gb/s, 0 ulp |        0.607 gb/s, 0 ulp |
| `nk_sqeuclidean_e4m3_haswell` |                4.16 gb/s |                4.19 gb/s |                4.12 gb/s |
| `nk_euclidean_e4m3_haswell`   |                4.10 gb/s |                4.14 gb/s |                4.19 gb/s |
| `nk_angular_e4m3_haswell`     |                3.70 gb/s |                3.91 gb/s |                3.93 gb/s |
| `nk_sqeuclidean_e4m3_skylake` |         6.47 gb/s, 0 ulp |         6.50 gb/s, 0 ulp |       5.55 gb/s, 0.2 ulp |
| `nk_euclidean_e4m3_skylake`   |         6.01 gb/s, 0 ulp |         6.54 gb/s, 0 ulp |       6.45 gb/s, 0.2 ulp |
| `nk_angular_e4m3_skylake`     |         5.69 gb/s, 0 ulp |         6.13 gb/s, 0 ulp |         5.36 gb/s, 0 ulp |
| `nk_sqeuclidean_e4m3_icelake` |         8.72 gb/s, 0 ulp |         7.91 gb/s, 0 ulp |       9.75 gb/s, 0.2 ulp |
| `nk_euclidean_e4m3_icelake`   |         8.35 gb/s, 0 ulp |         7.78 gb/s, 0 ulp |       9.78 gb/s, 0.2 ulp |
| `nk_angular_e4m3_icelake`     |         7.39 gb/s, 0 ulp |         7.44 gb/s, 0 ulp |         9.65 gb/s, 0 ulp |
| __e3m2__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e3m2_serial`  |         1.19 gb/s, 0 ulp |         1.71 gb/s, 0 ulp |         1.74 gb/s, 0 ulp |
| `nk_euclidean_e3m2_serial`    |       1.10 gb/s, 0.5 ulp |       1.67 gb/s, 0.5 ulp |       1.74 gb/s, 0.4 ulp |
| `nk_angular_e3m2_serial`      |        0.919 gb/s, 0 ulp |         1.40 gb/s, 0 ulp |         1.28 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_haswell` |                3.33 gb/s |                2.78 gb/s |                3.54 gb/s |
| `nk_euclidean_e3m2_haswell`   |                3.31 gb/s |                3.04 gb/s |                3.54 gb/s |
| `nk_angular_e3m2_haswell`     |                3.10 gb/s |                3.21 gb/s |                3.44 gb/s |
| `nk_sqeuclidean_e3m2_skylake` |         5.69 gb/s, 0 ulp |         5.57 gb/s, 0 ulp |         5.26 gb/s, 0 ulp |
| `nk_euclidean_e3m2_skylake`   |         5.52 gb/s, 0 ulp |         5.49 gb/s, 0 ulp |         5.41 gb/s, 0 ulp |
| `nk_angular_e3m2_skylake`     |         4.99 gb/s, 0 ulp |         5.38 gb/s, 0 ulp |         5.38 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_icelake` |         19.2 gb/s, 0 ulp |         16.5 gb/s, 0 ulp |         18.3 gb/s, 0 ulp |
| `nk_euclidean_e3m2_icelake`   |         18.4 gb/s, 0 ulp |         16.0 gb/s, 0 ulp |         18.0 gb/s, 0 ulp |
| `nk_angular_e3m2_icelake`     |         12.5 gb/s, 0 ulp |         13.8 gb/s, 0 ulp |         15.9 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_alder`   |                8.13 gb/s |                7.95 gb/s |                8.12 gb/s |
| `nk_euclidean_e3m2_alder`     |                8.00 gb/s |                7.90 gb/s |                8.10 gb/s |
| `nk_angular_e3m2_alder`       |                7.75 gb/s |                8.56 gb/s |                9.07 gb/s |
| __e2m3__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e2m3_serial`  |         1.07 gb/s, 0 ulp |         1.43 gb/s, 0 ulp |         1.74 gb/s, 0 ulp |
| `nk_euclidean_e2m3_serial`    |       1.05 gb/s, 0.5 ulp |       1.49 gb/s, 0.5 ulp |       1.15 gb/s, 0.5 ulp |
| `nk_angular_e2m3_serial`      |        0.904 gb/s, 0 ulp |         1.20 gb/s, 0 ulp |         1.42 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_haswell` |                3.27 gb/s |                3.52 gb/s |                3.54 gb/s |
| `nk_euclidean_e2m3_haswell`   |                3.36 gb/s |                3.51 gb/s |                3.54 gb/s |
| `nk_angular_e2m3_haswell`     |                3.12 gb/s |                3.40 gb/s |                3.45 gb/s |
| `nk_sqeuclidean_e2m3_skylake` |         5.59 gb/s, 0 ulp |         5.27 gb/s, 0 ulp |         5.65 gb/s, 0 ulp |
| `nk_euclidean_e2m3_skylake`   |         5.55 gb/s, 0 ulp |         5.57 gb/s, 0 ulp |         4.44 gb/s, 0 ulp |
| `nk_angular_e2m3_skylake`     |         4.94 gb/s, 0 ulp |         4.39 gb/s, 0 ulp |         5.40 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_icelake` |         37.2 gb/s, 0 ulp |         27.4 gb/s, 0 ulp |         25.6 gb/s, 0 ulp |
| `nk_euclidean_e2m3_icelake`   |         35.9 gb/s, 0 ulp |         26.1 gb/s, 0 ulp |         25.4 gb/s, 0 ulp |
| `nk_angular_e2m3_icelake`     |         24.5 gb/s, 0 ulp |         24.1 gb/s, 0 ulp |         25.4 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_alder`   |                13.3 gb/s |                11.8 gb/s |                13.9 gb/s |
| `nk_euclidean_e2m3_alder`     |                12.6 gb/s |                12.1 gb/s |                13.8 gb/s |
| `nk_angular_e2m3_alder`       |                11.5 gb/s |                11.7 gb/s |                13.7 gb/s |
| __i8__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`    |                3.31 gb/s |                3.56 gb/s |                3.73 gb/s |
| `nk_euclidean_i8_serial`      |       3.17 gb/s, 0.4 ulp |       3.53 gb/s, 0.4 ulp |       3.76 gb/s, 0.4 ulp |
| `nk_angular_i8_serial`        |         1.55 gb/s, 0 ulp |         1.84 gb/s, 0 ulp |         1.87 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_haswell`   |                31.9 gb/s |                17.3 gb/s |                23.1 gb/s |
| `nk_euclidean_i8_haswell`     |         29.9 gb/s, 0 ulp |         17.1 gb/s, 0 ulp |         22.8 gb/s, 0 ulp |
| `nk_angular_i8_haswell`       |       18.4 gb/s, 0.1 ulp |         13.1 gb/s, 0 ulp |         20.1 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_icelake`   |                42.6 gb/s |                29.1 gb/s |                25.1 gb/s |
| `nk_euclidean_i8_icelake`     |         41.6 gb/s, 0 ulp |         27.8 gb/s, 0 ulp |         25.0 gb/s, 0 ulp |
| `nk_angular_i8_icelake`       |       19.3 gb/s, 0.1 ulp |         19.8 gb/s, 0 ulp |         22.7 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_alder`     |                26.1 gb/s |                21.1 gb/s |                24.3 gb/s |
| `nk_euclidean_i8_alder`       |         24.8 gb/s, 0 ulp |         21.2 gb/s, 0 ulp |         22.5 gb/s, 0 ulp |
| `nk_angular_i8_alder`         |       22.1 gb/s, 0.1 ulp |         19.9 gb/s, 0 ulp |         23.8 gb/s, 0 ulp |
| __u8__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`    |                3.23 gb/s |                3.74 gb/s |                3.59 gb/s |
| `nk_euclidean_u8_serial`      |       3.30 gb/s, 0.5 ulp |       3.86 gb/s, 0.5 ulp |       3.50 gb/s, 0.6 ulp |
| `nk_angular_u8_serial`        |       1.75 gb/s, 0.4 ulp |       1.85 gb/s, 0.4 ulp |       1.90 gb/s, 0.4 ulp |
| `nk_sqeuclidean_u8_haswell`   |                36.1 gb/s |                20.4 gb/s |                24.1 gb/s |
| `nk_euclidean_u8_haswell`     |         34.3 gb/s, 0 ulp |         18.4 gb/s, 0 ulp |         24.0 gb/s, 0 ulp |
| `nk_angular_u8_haswell`       |       19.0 gb/s, 0.7 ulp |       14.0 gb/s, 0.6 ulp |       21.0 gb/s, 0.5 ulp |
| `nk_sqeuclidean_u8_icelake`   |                49.2 gb/s |                32.7 gb/s |                25.5 gb/s |
| `nk_euclidean_u8_icelake`     |         47.0 gb/s, 0 ulp |         32.1 gb/s, 0 ulp |         25.4 gb/s, 0 ulp |
| `nk_angular_u8_icelake`       |       23.8 gb/s, 0.7 ulp |       23.8 gb/s, 0.6 ulp |       24.2 gb/s, 0.5 ulp |
| `nk_sqeuclidean_u8_alder`     |                25.6 gb/s |                21.3 gb/s |                23.3 gb/s |
| `nk_euclidean_u8_alder`       |         25.4 gb/s, 0 ulp |         20.8 gb/s, 0 ulp |         24.5 gb/s, 0 ulp |
| `nk_angular_u8_alder`         |       21.7 gb/s, 0.7 ulp |       19.9 gb/s, 0.6 ulp |       23.3 gb/s, 0.5 ulp |
| __i4__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i4_serial`    |               0.986 gb/s |                1.16 gb/s |                1.18 gb/s |
| `nk_euclidean_i4_serial`      |       1.00 gb/s, 0.5 ulp |       1.16 gb/s, 0.5 ulp |       1.16 gb/s, 0.6 ulp |
| `nk_angular_i4_serial`        |      0.645 gb/s, 0.4 ulp |      0.715 gb/s, 0.4 ulp |      0.742 gb/s, 0.4 ulp |
| `nk_sqeuclidean_i4_icelake`   |                24.7 gb/s |                31.4 gb/s |                18.1 gb/s |
| `nk_euclidean_i4_icelake`     |         24.4 gb/s, 0 ulp |         31.6 gb/s, 0 ulp |         17.7 gb/s, 0 ulp |
| `nk_angular_i4_icelake`       |       9.09 gb/s, 0.7 ulp |       15.7 gb/s, 0.6 ulp |       13.6 gb/s, 0.5 ulp |
| __u4__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u4_serial`    |                1.53 gb/s |                1.84 gb/s |                1.91 gb/s |
| `nk_euclidean_u4_serial`      |       1.65 gb/s, 0.5 ulp |       1.84 gb/s, 0.5 ulp |       1.90 gb/s, 0.6 ulp |
| `nk_angular_u4_serial`        |      0.861 gb/s, 0.4 ulp |      0.954 gb/s, 0.4 ulp |      0.969 gb/s, 0.4 ulp |
| `nk_sqeuclidean_u4_icelake`   |                30.8 gb/s |                33.7 gb/s |                19.7 gb/s |
| `nk_euclidean_u4_icelake`     |         29.1 gb/s, 0 ulp |         28.3 gb/s, 0 ulp |         19.4 gb/s, 0 ulp |
| `nk_angular_u4_icelake`       |       13.9 gb/s, 0.7 ulp |       22.4 gb/s, 0.6 ulp |       16.6 gb/s, 0.5 ulp |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel                            |                      256 |                     1024 |                     4096 |
| :-------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f64_serial`       |       3.19 gb/s, 0.1 ulp |         3.35 gb/s, 0 ulp |         2.89 gb/s, 0 ulp |
| `nk_euclidean_f64_serial`         |       3.32 gb/s, 0.6 ulp |       3.41 gb/s, 0.6 ulp |       2.94 gb/s, 0.5 ulp |
| `nk_angular_f64_serial`           |       1.26 gb/s, 0.1 ulp |         1.28 gb/s, 0 ulp |         1.23 gb/s, 0 ulp |
| `nk_sqeuclidean_f64_v128relaxed`  |       8.30 gb/s, 1.3 ulp |       9.56 gb/s, 2.5 ulp |       7.79 gb/s, 5.0 ulp |
| `nk_euclidean_f64_v128relaxed`    |       8.24 gb/s, 0.7 ulp |       9.56 gb/s, 1.4 ulp |       7.75 gb/s, 2.8 ulp |
| `nk_angular_f64_v128relaxed`      |       4.12 gb/s, 0.1 ulp |       4.29 gb/s, 0.1 ulp |       3.58 gb/s, 0.1 ulp |
| __f32__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f32_serial`       |         6.66 gb/s, 0 ulp |         7.84 gb/s, 0 ulp |         6.13 gb/s, 0 ulp |
| `nk_euclidean_f32_serial`         |       6.47 gb/s, 0.1 ulp |       7.21 gb/s, 0.1 ulp |       5.70 gb/s, 0.1 ulp |
| `nk_angular_f32_serial`           |         5.60 gb/s, 0 ulp |         6.08 gb/s, 0 ulp |         4.85 gb/s, 0 ulp |
| `nk_sqeuclidean_f32_v128relaxed`  |       7.37 gb/s, 0.7 ulp |       7.62 gb/s, 1.3 ulp |       5.88 gb/s, 2.6 ulp |
| `nk_euclidean_f32_v128relaxed`    |       7.38 gb/s, 0.4 ulp |       4.77 gb/s, 0.7 ulp |       7.44 gb/s, 1.4 ulp |
| `nk_angular_f32_v128relaxed`      |         6.66 gb/s, 0 ulp |         7.37 gb/s, 0 ulp |         5.31 gb/s, 0 ulp |
| __bf16__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_bf16_serial`      |         4.46 gb/s, 0 ulp |         4.43 gb/s, 0 ulp |         3.54 gb/s, 0 ulp |
| `nk_euclidean_bf16_serial`        |       4.27 gb/s, 0.6 ulp |       4.46 gb/s, 0.5 ulp |       3.71 gb/s, 0.5 ulp |
| `nk_angular_bf16_serial`          |         3.33 gb/s, 0 ulp |         3.42 gb/s, 0 ulp |         2.79 gb/s, 0 ulp |
| `nk_sqeuclidean_bf16_v128relaxed` |       12.8 gb/s, 0.9 ulp |      7.68 gb/s, 12.6 ulp |      9.06 gb/s, 20.8 ulp |
| `nk_euclidean_bf16_v128relaxed`   |       12.4 gb/s, 0.5 ulp |       7.79 gb/s, 7.0 ulp |      9.39 gb/s, 11.4 ulp |
| `nk_angular_bf16_v128relaxed`     |         10.6 gb/s, 0 ulp |       7.22 gb/s, 0.2 ulp |       9.01 gb/s, 0.6 ulp |
| `nk_sqeuclidean_bf16_v128`        |                18.4 gb/s |                9.54 gb/s |                17.6 gb/s |
| `nk_euclidean_bf16_v128`          |                18.3 gb/s |                9.55 gb/s |                19.1 gb/s |
| `nk_angular_bf16_v128`            |                12.8 gb/s |                8.37 gb/s |                13.0 gb/s |
| __f16__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_f16_serial`       |       1.44 gb/s, 0.1 ulp |       1.44 gb/s, 0.1 ulp |       1.29 gb/s, 0.1 ulp |
| `nk_euclidean_f16_serial`         |       1.40 gb/s, 0.6 ulp |       1.41 gb/s, 0.6 ulp |       1.19 gb/s, 0.5 ulp |
| `nk_angular_f16_serial`           |         1.32 gb/s, 0 ulp |         1.36 gb/s, 0 ulp |         1.18 gb/s, 0 ulp |
| `nk_sqeuclidean_f16_v128relaxed`  |       2.74 gb/s, 0.9 ulp |       2.09 gb/s, 3.6 ulp |       2.20 gb/s, 9.7 ulp |
| `nk_euclidean_f16_v128relaxed`    |       2.74 gb/s, 0.5 ulp |       2.02 gb/s, 2.0 ulp |       2.24 gb/s, 5.4 ulp |
| `nk_angular_f16_v128relaxed`      |       2.60 gb/s, 0.1 ulp |       1.86 gb/s, 0.1 ulp |       2.39 gb/s, 0.1 ulp |
| __e5m2__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e5m2_serial`      |         1.12 gb/s, 0 ulp |         1.14 gb/s, 0 ulp |         1.15 gb/s, 0 ulp |
| `nk_euclidean_e5m2_serial`        |       1.10 gb/s, 0.5 ulp |       1.12 gb/s, 0.5 ulp |       1.14 gb/s, 0.5 ulp |
| `nk_angular_e5m2_serial`          |        0.968 gb/s, 0 ulp |        0.993 gb/s, 0 ulp |         1.01 gb/s, 0 ulp |
| `nk_sqeuclidean_e5m2_v128relaxed` |                1.17 gb/s |                1.18 gb/s |                1.19 gb/s |
| `nk_euclidean_e5m2_v128relaxed`   |                1.19 gb/s |                1.19 gb/s |                1.20 gb/s |
| `nk_angular_e5m2_v128relaxed`     |                1.18 gb/s |                1.01 gb/s |                1.22 gb/s |
| __e4m3__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e4m3_serial`      |        0.406 gb/s, 0 ulp |        0.327 gb/s, 0 ulp |        0.323 gb/s, 0 ulp |
| `nk_euclidean_e4m3_serial`        |      0.417 gb/s, 0.5 ulp |      0.360 gb/s, 0.5 ulp |      0.357 gb/s, 0.5 ulp |
| `nk_angular_e4m3_serial`          |        0.400 gb/s, 0 ulp |        0.394 gb/s, 0 ulp |        0.395 gb/s, 0 ulp |
| `nk_sqeuclidean_e4m3_v128relaxed` |                1.06 gb/s |                1.43 gb/s |                1.46 gb/s |
| `nk_euclidean_e4m3_v128relaxed`   |                1.12 gb/s |                1.35 gb/s |                1.44 gb/s |
| `nk_angular_e4m3_v128relaxed`     |                1.40 gb/s |                1.44 gb/s |                1.45 gb/s |
| __e3m2__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e3m2_serial`      |         1.11 gb/s, 0 ulp |        0.823 gb/s, 0 ulp |        0.823 gb/s, 0 ulp |
| `nk_euclidean_e3m2_serial`        |       1.08 gb/s, 0.5 ulp |      0.859 gb/s, 0.5 ulp |       1.14 gb/s, 0.5 ulp |
| `nk_angular_e3m2_serial`          |        0.952 gb/s, 0 ulp |        0.881 gb/s, 0 ulp |        0.754 gb/s, 0 ulp |
| `nk_sqeuclidean_e3m2_v128relaxed` |               0.808 gb/s |               0.824 gb/s |               0.826 gb/s |
| `nk_euclidean_e3m2_v128relaxed`   |               0.811 gb/s |               0.825 gb/s |               0.783 gb/s |
| `nk_angular_e3m2_v128relaxed`     |               0.813 gb/s |               0.825 gb/s |               0.834 gb/s |
| __e2m3__                          | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_e2m3_serial`      |        0.926 gb/s, 0 ulp |         1.08 gb/s, 0 ulp |        0.853 gb/s, 0 ulp |
| `nk_euclidean_e2m3_serial`        |      0.750 gb/s, 0.5 ulp |       1.15 gb/s, 0.5 ulp |      0.841 gb/s, 0.5 ulp |
| `nk_angular_e2m3_serial`          |        0.965 gb/s, 0 ulp |        0.761 gb/s, 0 ulp |        0.741 gb/s, 0 ulp |
| `nk_sqeuclidean_e2m3_v128relaxed` |               0.257 gb/s |               0.259 gb/s |               0.260 gb/s |
| `nk_euclidean_e2m3_v128relaxed`   |               0.260 gb/s |               0.260 gb/s |               0.262 gb/s |
| `nk_angular_e2m3_v128relaxed`     |               0.263 gb/s |               0.265 gb/s |               0.261 gb/s |
| __i8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i8_serial`        |                2.29 gb/s |                3.30 gb/s |                2.62 gb/s |
| `nk_euclidean_i8_serial`          |       2.10 gb/s, 0.5 ulp |       3.31 gb/s, 0.4 ulp |       2.55 gb/s, 0.4 ulp |
| `nk_angular_i8_serial`            |         1.31 gb/s, 0 ulp |         1.87 gb/s, 0 ulp |         1.50 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_v128relaxed`   |                7.90 gb/s |                5.46 gb/s |                7.86 gb/s |
| `nk_euclidean_i8_v128relaxed`     |                7.66 gb/s |                5.69 gb/s |                7.92 gb/s |
| `nk_angular_i8_v128relaxed`       |         6.84 gb/s, 0 ulp |         5.46 gb/s, 0 ulp |         9.70 gb/s, 0 ulp |
| `nk_sqeuclidean_i8_v128`          |                9.45 gb/s |                8.22 gb/s |                9.15 gb/s |
| `nk_euclidean_i8_v128`            |         8.32 gb/s, 0 ulp |         8.20 gb/s, 0 ulp |         8.74 gb/s, 0 ulp |
| `nk_angular_i8_v128`              |                9.87 gb/s |                8.41 gb/s |                12.8 gb/s |
| __u8__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u8_serial`        |                2.01 gb/s |                2.42 gb/s |                2.56 gb/s |
| `nk_euclidean_u8_serial`          |       1.96 gb/s, 0.5 ulp |       3.32 gb/s, 0.5 ulp |       3.44 gb/s, 0.6 ulp |
| `nk_angular_u8_serial`            |       1.25 gb/s, 0.5 ulp |       1.85 gb/s, 0.4 ulp |       1.58 gb/s, 0.5 ulp |
| `nk_sqeuclidean_u8_v128relaxed`   |                6.22 gb/s |                7.08 gb/s |                7.66 gb/s |
| `nk_euclidean_u8_v128relaxed`     |                6.14 gb/s |                7.42 gb/s |                7.72 gb/s |
| `nk_angular_u8_v128relaxed`       |      5.51 gb/s, 526M ulp |      4.45 gb/s, 501M ulp |      7.54 gb/s, 443M ulp |
| `nk_sqeuclidean_u8_v128`          |                8.46 gb/s |                8.54 gb/s |                9.05 gb/s |
| `nk_euclidean_u8_v128`            |         8.29 gb/s, 0 ulp |         8.54 gb/s, 0 ulp |         9.05 gb/s, 0 ulp |
| `nk_angular_u8_v128`              |                7.95 gb/s |                10.6 gb/s |                10.4 gb/s |
| __i4__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_i4_serial`        |               0.962 gb/s |                1.44 gb/s |                1.48 gb/s |
| `nk_euclidean_i4_serial`          |      0.889 gb/s, 0.5 ulp |       1.15 gb/s, 0.5 ulp |       1.48 gb/s, 0.0 ulp |
| `nk_angular_i4_serial`            |      0.634 gb/s, 0.5 ulp |      0.930 gb/s, 0.5 ulp |      0.953 gb/s, 0.5 ulp |
| __u4__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclidean_u4_serial`        |                1.41 gb/s |                1.20 gb/s |                1.55 gb/s |
| `nk_euclidean_u4_serial`          |       1.38 gb/s, 0.5 ulp |       1.14 gb/s, 0.5 ulp |       1.59 gb/s, 0.0 ulp |
| `nk_angular_u4_serial`            |      0.806 gb/s, 0.5 ulp |      0.758 gb/s, 0.5 ulp |      0.959 gb/s, 0.5 ulp |

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
