# Curved Space Distances in NumKong

NumKong implements distance functions for curved metric spaces: bilinear forms compute $a^T C b$ for an arbitrary metric tensor $C$, while Mahalanobis distance generalizes Euclidean distance to account for correlations between dimensions.
Complex bilinear forms extend this to Hermitian inner products.
These operations are central to Gaussian process inference, metric learning, and statistical distance measures.

The bilinear form for real vectors is:

$$
\text{bilinear}(a, b, C) = a^T C b = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} a_i \cdot c_{ij} \cdot b_j
$$

The Mahalanobis distance is:

$$
\text{mahalanobis}(a, b, C) = \sqrt{(a - b)^T C (a - b)}
$$

For complex vectors, the bilinear form uses the conjugate transpose:

$$
\text{bilinear}(a, b, C) = a^H C b = \sum_{i=0}^{n-1} \sum_{j=0}^{n-1} \bar{a_i} \cdot c_{ij} \cdot b_j
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def bilinear(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> float:
    return a @ C @ b

def mahalanobis(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> float:
    diff = a - b
    return np.sqrt(diff @ C @ diff)

def bilinear_complex(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> complex:
    return np.conj(a) @ C @ b
```

## Input & Output Types

Real bilinear and Mahalanobis:

| Input Type | Output Type | Description                                      |
| :--------- | :---------- | :----------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision                 |
| `f32`      | `f64`       | 32-bit IEEE 754 single precision, widened output |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output   |
| `bf16`     | `f32`       | 16-bit brain float, widened output               |

Complex bilinear:

| Input Type | Output Type | Description                                |
| :--------- | :---------- | :----------------------------------------- |
| `f64c`     | `f64c`      | 64-bit complex pairs                       |
| `f32c`     | `f64c`      | 32-bit complex pairs, widened output       |
| `f16c`     | `f32c`      | 16-bit complex pairs, widened output       |
| `bf16c`    | `f32c`      | 16-bit brain complex pairs, widened output |

## Optimizations

### Row-Major Streaming with Nested Dot2

`nk_bilinear_f64_skylake`, `nk_mahalanobis_f64_skylake` decompose the bilinear form $a^T C b$ as $\sum_i a_i \cdot \text{dot}(C_i, b)$ where $C_i$ is the $i$-th row of the metric tensor.
Each inner dot product uses Dot2 compensation — TwoProd via FMA captures the rounding error of each $c_{ij} \cdot b_j$ product exactly, and a TwoSum chain propagates it through the accumulator.
The outer sum over rows uses a second level of compensation, tracking the rounding error of each $a_i \cdot r_i$ accumulation.
This nested structure gives $O(n)$ cache-friendly sequential access to the $n \times n$ matrix $C$, since each row is read once and discarded.
`nk_bilinear_f32_neon`, `nk_bilinear_f32_skylake`, `nk_mahalanobis_f32_neon`, `nk_mahalanobis_f32_skylake` use the same row-major streaming pattern but accumulate in `f64` instead of Dot2, which provides sufficient precision for `f32` inputs.

### SME Outer-Product Accumulation

`nk_bilinear_f32_smef64`, `nk_bilinear_f64_smef64`, `nk_bilinear_f32c_smef64`, `nk_bilinear_f64c_smef64`, `nk_mahalanobis_f32_smef64`, `nk_mahalanobis_f64_smef64` use the Scalable Matrix Extension to compute the bilinear form as an outer-product accumulation.
Each `FMOPA` instruction performs a rank-1 update $a_i \cdot b^T$ into the SME ZA tile array, and the matrix $C$ is streamed row-by-row and multiplied into the accumulator.
This differs from the row-major dot approach — it reformulates $a^T C b$ as a matrix-multiply problem where SME's 2D tile registers use the matrix engine's throughput.
For dimensions that align to the tile size, this approach has high throughput; dimensions that do not align fall back to NEON for cleanup of the residual elements.

### Complex Bilinear Decomposition

`nk_bilinear_f32c_neon`, `nk_bilinear_f32c_skylake`, `nk_bilinear_f64c_skylake` compute $a^H C b$ where each element involves 4 real multiplications from the complex product $\bar{a_i} \cdot c_{ij} \cdot b_j$.
The kernel decomposes this into real and imaginary dot products over rows of $C$: for each row $i$, it computes the real part as $a_{i,re} \cdot \text{dot}(C_i, b)_{re} + a_{i,im} \cdot \text{dot}(C_i, b)_{im}$ and the imaginary part with the conjugation baked in as sign flips.
This fuses the conjugation of $a$ into the sign of the cross terms rather than explicitly negating the imaginary components, saving one negate operation per element.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by the `NUMKONG_CURVED_DIMS` environment variable.
The metric tensor is a square matrix of side $N$, so each bilinear form $\mathbf{x}^\top M \mathbf{x}$ has $O(N^2)$ arithmetic complexity.
Columns show matrix side length: 256², 1024², 4096².
The throughput is measured in GSO/s as Giga Scalar Operations per Second.

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.

#### Native

| Kernel                        |                     256² |                    1024² |                    4096² |
| :---------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `bilinear_f64c_with_blas` 🧩  |               1.51 gso/s |               1.65 gso/s |               1.64 gso/s |
| `nk_bilinear_f64c_serial`     |     0.113 gso/s, 0.5 ulp |     0.113 gso/s, 0.2 ulp |     0.110 gso/s, 0.5 ulp |
| `nk_bilinear_f64c_skylake`    |     0.427 gso/s, 3.5 ulp |     0.605 gso/s, 3.5 ulp |     0.616 gso/s, 3.5 ulp |
| __f32c__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `bilinear_f32c_with_blas` 🧩  |               2.50 gso/s |               3.24 gso/s |               3.26 gso/s |
| `nk_bilinear_f32c_serial`     |       0.414 gso/s, 0 ulp |       0.416 gso/s, 0 ulp |       0.436 gso/s, 0 ulp |
| `nk_bilinear_f32c_skylake`    |        1.29 gso/s, 0 ulp |        2.01 gso/s, 0 ulp |        2.28 gso/s, 0 ulp |
| __bf16c__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_bf16c_serial`    |       0.707 gso/s, 5 ulp |     0.608 gso/s, 5.8 ulp |       0.579 gso/s, 5 ulp |
| `nk_bilinear_bf16c_genoa`     |        4.04 gso/s, 5 ulp |        4.58 gso/s, 5 ulp |        3.20 gso/s, 5 ulp |
| __f16c__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f16c_serial`     |     0.380 gso/s, 7.2 ulp |     0.365 gso/s, 7.2 ulp |     0.357 gso/s, 7.2 ulp |
| __f64__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `bilinear_f64_with_blas` 🧩   |               3.45 gso/s |               3.64 gso/s |               3.46 gso/s |
| `nk_bilinear_f64_serial`      |     0.301 gso/s, 0.7 ulp |     0.293 gso/s, 0.4 ulp |     0.383 gso/s, 0.7 ulp |
| `nk_mahalanobis_f64_serial`   |       0.310 gso/s, 0 ulp |       0.297 gso/s, 0 ulp |       0.384 gso/s, 0 ulp |
| `nk_bilinear_f64_skylake`     |      1.28 gso/s, 1.6 ulp |      1.96 gso/s, 1.3 ulp |        2.16 gso/s, 1 ulp |
| `nk_mahalanobis_f64_skylake`  |        1.85 gso/s, 0 ulp |        1.66 gso/s, 0 ulp |      2.37 gso/s, 0.2 ulp |
| __f32__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `bilinear_f32_with_blas` 🧩   |               4.20 gso/s |               7.31 gso/s |               7.05 gso/s |
| `nk_bilinear_f32_serial`      |        1.07 gso/s, 0 ulp |       0.925 gso/s, 0 ulp |       0.870 gso/s, 0 ulp |
| `nk_mahalanobis_f32_serial`   |       0.935 gso/s, 0 ulp |       0.848 gso/s, 0 ulp |       0.775 gso/s, 0 ulp |
| `nk_bilinear_f32_haswell`     |        2.31 gso/s, 0 ulp |        3.36 gso/s, 0 ulp |        2.98 gso/s, 0 ulp |
| `nk_mahalanobis_f32_haswell`  |        1.98 gso/s, 0 ulp |        2.40 gso/s, 0 ulp |        2.48 gso/s, 0 ulp |
| `nk_bilinear_f32_skylake`     |        2.97 gso/s, 0 ulp |        4.40 gso/s, 0 ulp |        4.40 gso/s, 0 ulp |
| `nk_mahalanobis_f32_skylake`  |        2.55 gso/s, 0 ulp |        4.02 gso/s, 0 ulp |        4.03 gso/s, 0 ulp |
| __bf16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_bf16_serial`     |       1.64 gso/s, 16 ulp |       1.35 gso/s, 13 ulp |       1.25 gso/s, 12 ulp |
| `nk_mahalanobis_bf16_serial`  |      1.62 gso/s, 2.2 ulp |      1.32 gso/s, 2.1 ulp |      1.21 gso/s, 2.3 ulp |
| `nk_bilinear_bf16_haswell`    |       4.11 gso/s, 11 ulp |       5.09 gso/s, 13 ulp |       4.80 gso/s, 13 ulp |
| `nk_mahalanobis_bf16_haswell` |        2.94 gso/s, 1 ulp |        4.44 gso/s, 1 ulp |        4.04 gso/s, 1 ulp |
| `nk_bilinear_bf16_genoa`      |       11.0 gso/s, 18 ulp |       12.3 gso/s, 18 ulp |       10.1 gso/s, 18 ulp |
| `nk_mahalanobis_bf16_genoa`   |    8.14 gso/s, 8.55K ulp |    9.40 gso/s, 8.41K ulp |    6.20 gso/s, 8.41K ulp |
| __f16__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f16_serial`      |      0.785 gso/s, 23 ulp |      0.772 gso/s, 23 ulp |      0.724 gso/s, 23 ulp |
| `nk_mahalanobis_f16_serial`   |     0.546 gso/s, 2.7 ulp |     0.549 gso/s, 3.2 ulp |     0.529 gso/s, 2.7 ulp |
| `nk_bilinear_f16_haswell`     |       4.82 gso/s, 37 ulp |       6.28 gso/s, 37 ulp |       5.60 gso/s, 37 ulp |
| `nk_mahalanobis_f16_haswell`  |        3.69 gso/s, 1 ulp |        5.05 gso/s, 1 ulp |        4.65 gso/s, 1 ulp |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel                       |                     256² |                    1024² |                    4096² |
| :--------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f64c_serial`    |     0.128 gso/s, 1.2 ulp |     0.126 gso/s, 1.2 ulp |     0.130 gso/s, 1.2 ulp |
| __f32c__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f32c_serial`    |       0.932 gso/s, 0 ulp |       0.969 gso/s, 0 ulp |       0.368 gso/s, 0 ulp |
| __bf16c__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_bf16c_serial`   |      1.15 gso/s, 9.8 ulp |      1.19 gso/s, 9.8 ulp |     0.619 gso/s, 9.5 ulp |
| __f16c__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f16c_serial`    |      0.392 gso/s, 39 ulp |      0.390 gso/s, 39 ulp |      0.380 gso/s, 39 ulp |
| __f64__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f64_serial`     |     0.439 gso/s, 0.6 ulp |     0.421 gso/s, 0.6 ulp |     0.380 gso/s, 0.6 ulp |
| `nk_mahalanobis_f64_serial`  |              0.367 gso/s |              0.407 gso/s |              0.333 gso/s |
| __f32__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f32_serial`     |        2.14 gso/s, 0 ulp |        2.14 gso/s, 0 ulp |       0.808 gso/s, 0 ulp |
| `nk_mahalanobis_f32_serial`  |               1.48 gso/s |               1.62 gso/s |              0.701 gso/s |
| __bf16__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_bf16_serial`    |       2.77 gso/s, 27 ulp |       2.83 gso/s, 22 ulp |       1.29 gso/s, 22 ulp |
| `nk_mahalanobis_bf16_serial` |               1.68 gso/s |               1.64 gso/s |               1.12 gso/s |
| __f16__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f16_serial`     |      0.800 gso/s, 74 ulp |      0.760 gso/s, 74 ulp |      0.775 gso/s, 74 ulp |
| `nk_mahalanobis_f16_serial`  |              0.476 gso/s |              0.554 gso/s |              0.552 gso/s |

### Apple M4

#### Native

| Kernel                          |                     256² |                    1024² |                    4096² |
| :------------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f64c_serial`       |     0.368 gso/s, 2.2 ulp |     0.371 gso/s, 2.2 ulp |     0.367 gso/s, 2.2 ulp |
| __f32c__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f32c_serial`       |        2.33 gso/s, 0 ulp |        2.27 gso/s, 0 ulp |        2.28 gso/s, 0 ulp |
| `nk_bilinear_f32c_neon`         |        2.11 gso/s, 0 ulp |        1.89 gso/s, 0 ulp |        1.85 gso/s, 0 ulp |
| __bf16c__                       | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_bf16c_serial`      |     2.83 gso/s, 33.0 ulp |     2.54 gso/s, 34.5 ulp |     2.49 gso/s, 34.5 ulp |
| `nk_bilinear_bf16c_neonbfdot`   |     5.05 gso/s, 17.0 ulp |     4.20 gso/s, 17.0 ulp |     4.04 gso/s, 17.0 ulp |
| __f16c__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f16c_serial`       |     2.81 gso/s, 51.8 ulp |     2.54 gso/s, 51.8 ulp |     2.48 gso/s, 51.8 ulp |
| `nk_bilinear_f16c_neon`         |     5.00 gso/s, 17.3 ulp |     4.16 gso/s, 17.3 ulp |     4.00 gso/s, 16.4 ulp |
| __f64__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f64_serial`        |     0.717 gso/s, 0.4 ulp |     0.711 gso/s, 0.4 ulp |     0.721 gso/s, 0.4 ulp |
| `nk_mahalanobis_f64_serial`     |     0.664 gso/s, 0.5 ulp |     0.667 gso/s, 0.5 ulp |     0.672 gso/s, 0.5 ulp |
| __f32__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f32_serial`        |        3.92 gso/s, 0 ulp |        3.05 gso/s, 0 ulp |        2.87 gso/s, 0 ulp |
| `nk_mahalanobis_f32_serial`     |        3.42 gso/s, 0 ulp |        2.88 gso/s, 0 ulp |        2.74 gso/s, 0 ulp |
| `nk_bilinear_f32_neon`          |        4.90 gso/s, 0 ulp |        3.82 gso/s, 0 ulp |        3.49 gso/s, 0 ulp |
| `nk_mahalanobis_f32_neon`       |        4.68 gso/s, 0 ulp |        3.71 gso/s, 0 ulp |        3.48 gso/s, 0 ulp |
| __bf16__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_bf16_serial`       |     4.17 gso/s, 20.7 ulp |     3.19 gso/s, 21.2 ulp |     2.94 gso/s, 20.7 ulp |
| `nk_mahalanobis_bf16_serial`    |      3.86 gso/s, 2.1 ulp |      2.98 gso/s, 2.2 ulp |      2.79 gso/s, 2.1 ulp |
| `nk_bilinear_bf16_neonbfdot`    |     28.0 gso/s, 28.0 ulp |     23.5 gso/s, 41.2 ulp |     20.4 gso/s, 41.1 ulp |
| `nk_mahalanobis_bf16_neonbfdot` |      9.14 gso/s, 2.2 ulp |      7.93 gso/s, 2.2 ulp |      7.43 gso/s, 2.2 ulp |
| __f16__                         | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f16_serial`        |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_mahalanobis_f16_serial`     |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_bilinear_f16_neon`          |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |
| `nk_mahalanobis_f16_neon`       |           ? gso/s, ? ulp |           ? gso/s, ? ulp |           ? gso/s, ? ulp |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                     |                     256² |                    1024² |                    4096² |
| :------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f64c_serial`  |       0.445 gso/s, ? ulp |       0.445 gso/s, ? ulp |       0.445 gso/s, ? ulp |
| __f32c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f32c_serial`  |        2.83 gso/s, ? ulp |        2.83 gso/s, ? ulp |        2.84 gso/s, ? ulp |
| __bf16c__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_bf16c_serial` |        3.05 gso/s, ? ulp |        3.02 gso/s, ? ulp |        3.03 gso/s, ? ulp |
| __f16c__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f16c_serial`  |       0.984 gso/s, ? ulp |       0.992 gso/s, ? ulp |       0.995 gso/s, ? ulp |
| __f64__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f64_serial`   |       0.998 gso/s, ? ulp |       0.999 gso/s, ? ulp |       0.999 gso/s, ? ulp |
| __f32__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f32_serial`   |        5.00 gso/s, ? ulp |        3.73 gso/s, ? ulp |        3.49 gso/s, ? ulp |
| __bf16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_bf16_serial`  |        4.84 gso/s, ? ulp |        3.83 gso/s, ? ulp |        3.60 gso/s, ? ulp |
| __f16__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_bilinear_f16_serial`   |        1.90 gso/s, ? ulp |        1.75 gso/s, ? ulp |        1.93 gso/s, ? ulp |
