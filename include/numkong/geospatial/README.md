# Geospatial Distances in NumKong

NumKong implements geodesic distance functions for points on Earth's surface: Haversine computes great-circle distance on a perfect sphere, while Vincenty solves the inverse geodesic problem on an oblate spheroid, using the IERS-2003 ellipsoid constants by default.
Both operate on arrays of latitude/longitude pairs in radians and produce distances in meters.

The Haversine formula computes the great-circle distance between two points:

$$
\text{haversine}(\phi_1, \lambda_1, \phi_2, \lambda_2) = 2R \arcsin\sqrt{\sin^2\frac{\phi_2 - \phi_1}{2} + \cos\phi_1 \cos\phi_2 \sin^2\frac{\lambda_2 - \lambda_1}{2}}
$$

where $R$ is the Earth radius from `NK_EARTH_MEDIATORIAL_RADIUS`, 6,335,439 m by default, and $(\phi, \lambda)$ are latitude and longitude in radians.

Vincenty's formula solves the inverse geodesic problem on an oblate spheroid, iteratively refining the reduced latitude difference until convergence:

$$
\text{vincenty}(\phi_1, \lambda_1, \phi_2, \lambda_2) = b \cdot A \cdot (\sigma - \Delta\sigma)
$$

where $a$ and $b$ are the equatorial and polar semi-axes of the configured Earth ellipsoid, $\sigma$ is the angular separation, and $\Delta\sigma$ is the correction term computed through iterative convergence.

Reformulating as Python pseudocode:

```python
import numpy as np

def haversine(lat1, lon1, lat2, lon2, R=6335439):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
```

Input coordinates are in radians, output distances are in meters.

## Input & Output Types

| Input Type | Output Type | Description             |
| ---------- | ----------- | ----------------------- |
| `f64`      | `f64`       | 64-bit double precision |
| `f32`      | `f32`       | 32-bit single precision |

## Optimizations

### Trigonometric Polynomial Approximations

`nk_haversine_f32_haswell`, `nk_haversine_f32_skylake`, `nk_haversine_f32_neon` replace `libm` sin, cos, atan2, and asin with SIMD polynomial approximations achieving approximately 3.5 ULP accuracy.
Range reduction maps the input angle to $[-\pi/2, \pi/2]$ using Cody-Waite extended precision constants, then an odd-degree minimax polynomial evaluates sin, and cos reuses the same polynomial after shifting the argument by $\pi/2$.
The `f32` kernels use 5-term polynomials while `f64` kernels use 10-term polynomials for the extra precision required by double-precision inputs.
This avoids the latency of scalar `libm` calls — each trigonometric evaluation would otherwise serialize through a single execution port, while the polynomial chains pipeline across multiple FMA units.

### Vincenty Iterative Convergence with Masked Lanes

`nk_vincenty_f64_haswell`, `nk_vincenty_f64_skylake`, `nk_vincenty_f64_neon` implement the full Vincenty inverse formula with up to 100 iterations and a convergence threshold of $10^{-12}$ radians (approximately 6 micrometers on Earth's surface).
Each SIMD lane may converge at a different iteration count, so the kernel accumulates a `converged_mask` via `_mm256_or_pd(converged_mask, newly_converged)` and selectively freezes converged lanes with `_mm256_blendv_pd(lambda_new, lambda, converged_mask)`.
Early exit uses `_mm256_movemask_pd` — when all 4 bits (for `f64`) or 8 bits (for `f32`) are set, the loop breaks.
Coincident points and equatorial edge cases are handled by blending safe values (ones) into the intermediate terms to avoid division by zero, without requiring branches that would diverge across SIMD lanes.

### Potential Optimization: Haversine Without Final Arc Conversion

The haversine formula computes $d = 2R \cdot \text{asin}(\sqrt{h})$ where $h = \sin^2(\Delta\phi/2) + \cos\phi_1 \cos\phi_2 \cdot \sin^2(\Delta\lambda/2)$.
Since both `asin` and `sqrt` are monotonically increasing, comparing $h$ values directly produces the same ordering as comparing full haversine distances.
For ranking-only use cases, a future "similarity mode" could skip the final `sqrt`/`atan2` conversion and return $h$ directly, eliminating the two most expensive operations in the pipeline.
Currently, all kernels compute the full distance.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_MAX_COORD_ANGLE` environment variable and set to ≤1°, ≤30°, and ≤180° maximum angular separation between pairs of coordinates.
The larger the angular separation between pairs, the longer the algorithm may take to converge and the higher the error.
The throughput is measured in MP/s as the number of Millions of pairwise point distances computed per second - amortized for a large batch size, with `NK_DENSE_DIMENSIONS=1536` by default.
Current `nk_test` output reports geospatial accuracy in two forms: mean/max absolute error in meters against Vincenty's formula computed at double-double (f118) precision, and mean/max ULP against the matching high-precision implementation of the same formula. The historical tables below use the meter-based summary where it has been remeasured; older x86 rows still retain their original ULP figures until rerun.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Sapphire Rapids

#### Native

| Kernel                     |                      ≤1° |                     ≤30° |                    ≤180° |
| :------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f64_serial`  |       1.95 mp/s, 0.8 ulp |       2.10 mp/s, 0.8 ulp |       2.02 mp/s, 1.7 ulp |
| `nk_vincenty_f64_serial`   |       0.565 mp/s, 82 ulp |      0.481 mp/s, 3.9 ulp |      0.514 mp/s, 1.1 ulp |
| `nk_haversine_f64_haswell` |       73.3 mp/s, 0.6 ulp |       68.3 mp/s, 0.6 ulp |       70.4 mp/s, 1.5 ulp |
| `nk_vincenty_f64_haswell`  |        12.2 mp/s, 80 ulp |       10.2 mp/s, 3.6 ulp |       7.15 mp/s, 1.1 ulp |
| `nk_haversine_f64_skylake` |        106 mp/s, 0.6 ulp |        107 mp/s, 0.6 ulp |       99.8 mp/s, 1.5 ulp |
| `nk_vincenty_f64_skylake`  |      20.4 mp/s, 171K ulp |     17.5 mp/s, 6.57K ulp |     11.2 mp/s, 1.02K ulp |
| __f32__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f32_serial`  |       56.2 mp/s, 3.4 ulp |       62.3 mp/s, 2.9 ulp |        57.2 mp/s, 55 ulp |
| `nk_vincenty_f32_serial`   |     3.25 mp/s, 58.3K ulp |       2.39 mp/s, 306 ulp |       1.79 mp/s, 103 ulp |
| `nk_haversine_f32_haswell` |        247 mp/s, 3.2 ulp |        282 mp/s, 2.7 ulp |         281 mp/s, 54 ulp |
| `nk_vincenty_f32_haswell`  |     53.6 mp/s, 26.2K ulp |       46.4 mp/s, 289 ulp |        16.5 mp/s, 61 ulp |
| `nk_haversine_f32_skylake` |        350 mp/s, 3.1 ulp |        328 mp/s, 2.7 ulp |         356 mp/s, 53 ulp |
| `nk_vincenty_f32_skylake`  |     78.7 mp/s, 7.16K ulp |       73.6 mp/s, 406 ulp |       20.1 mp/s, 105 ulp |

#### WASM

Measured with Wasmtime v42 (Cranelift backend).

| Kernel                         |                      ≤1° |                     ≤30° |                    ≤180° |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f64_serial`      |          ? mp/s, 0.9 ulp |          ? mp/s, 0.9 ulp |          ? mp/s, 1.8 ulp |
| `nk_vincenty_f64_serial`       |          ? mp/s, 102 ulp |          ? mp/s, 3.7 ulp |          ? mp/s, 1.1 ulp |
| `nk_haversine_f64_v128relaxed` |          ? mp/s, 0.6 ulp |          ? mp/s, 0.6 ulp |          ? mp/s, 1.7 ulp |
| `nk_vincenty_f64_v128relaxed`  |          ? mp/s, 104 ulp |          ? mp/s, 3.4 ulp |          ? mp/s, 1.1 ulp |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f32_serial`      |          ? mp/s, 3.5 ulp |          ? mp/s, 2.9 ulp |         ? mp/s, 53.6 ulp |
| `nk_vincenty_f32_serial`       |        ? mp/s, 70.5K ulp |          ? mp/s, 326 ulp |         ? mp/s, 65.5 ulp |
| `nk_haversine_f32_v128relaxed` |          ? mp/s, 6.5 ulp |          ? mp/s, 5.6 ulp |         ? mp/s, 53.3 ulp |
| `nk_vincenty_f32_v128relaxed`  |        ? mp/s, 23.8K ulp |          ? mp/s, 323 ulp |         ? mp/s, 64.0 ulp |

### Apple M4

#### Native

| Kernel                    |                      ≤1° |                     ≤30° |                    ≤180° |
| :------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f64_serial` |       3.47 mp/s, 1.12 km |       3.47 mp/s, 32.8 km |        3.48 mp/s, 150 km |
| `nk_vincenty_f64_serial`  |      0.888 mp/s, 2.20 nm |      0.770 mp/s, 2.79 nm |       0.662 mp/s, 622 nm |
| `nk_haversine_f64_neon`   |       72.8 mp/s, 1.12 km |       72.5 mp/s, 32.8 km |        72.8 mp/s, 150 km |
| `nk_vincenty_f64_neon`    |       9.34 mp/s, 2.12 nm |       7.61 mp/s, 2.33 nm |        5.99 mp/s, 622 nm |
| __f32__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f32_serial` |       14.1 mp/s, 1.12 km |       13.8 mp/s, 32.8 km |        14.0 mp/s, 146 km |
| `nk_vincenty_f32_serial`  |        4.54 mp/s, 12.0 m |        3.25 mp/s, 12.5 m |        2.42 mp/s, 22.0 m |
| `nk_haversine_f32_neon`   |        247 mp/s, 1.12 km |        235 mp/s, 32.8 km |         252 mp/s, 146 km |
| `nk_vincenty_f32_neon`    |        45.7 mp/s, 12.2 m |        37.9 mp/s, 12.8 m |        15.7 mp/s, 22.0 m |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                         |                      ≤1° |                     ≤30° |                    ≤180° |
| :----------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f64_serial`      |       5.12 mp/s, 1.12 km |       5.12 mp/s, 32.8 km |        5.07 mp/s, 148 km |
| `nk_vincenty_f64_serial`       |       1.27 mp/s, 1.86 nm |       1.11 mp/s, 2.33 nm |        1.02 mp/s, 594 nm |
| `nk_haversine_f64_v128relaxed` |        109 mp/s, 1.12 km |        109 mp/s, 32.8 km |         109 mp/s, 148 km |
| `nk_vincenty_f64_v128relaxed`  |       12.8 mp/s, 1.89 nm |       10.4 mp/s, 2.33 nm |        8.28 mp/s, 594 nm |
| __f32__                        | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversine_f32_serial`      |     20.9 mp/s, 20,000 km |       20.9 mp/s, 32.7 km |        21.0 mp/s, 136 km |
| `nk_vincenty_f32_serial`       |     6.11 mp/s, 20,000 km |        4.30 mp/s, 12.0 m |        3.27 mp/s, 22.0 m |
| `nk_haversine_f32_v128relaxed` |      523 mp/s, 20,000 km |        524 mp/s, 32.7 km |         524 mp/s, 153 km |
| `nk_vincenty_f32_v128relaxed`  |        76.9 mp/s, 12.0 m |        68.2 mp/s, 16.2 m |        26.8 mp/s, 18.0 m |
