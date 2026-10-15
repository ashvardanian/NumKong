# Geospatial Distances in NumKong

NumKong implements many-to-many distances between points on and around the Earth, and the terrain visibility they feed.
Every family takes a set of row points and a set of column points and fills a row-major matrix of results:

- `sqeuclideans3d`: squared Euclidean distances between 3D points, exact for `i32` Earth-fixed centimetres and compensated for `f64`.
- `haversines`: great-circle distances on the mean sphere, from latitudes and longitudes packed once per point set.
- `vincentys`: geodesic distances on the WGS-84 ellipsoid, from packs of the same kind.
- `terrain` and `clearances`: heights under points of a grid, and the margin by which every sight line clears it.

Latitudes and longitudes are in radians, distances, heights and clearances in metres.

The Haversine formula computes the great-circle distance between two points:

$$
\text{haversine}(\phi_1, \lambda_1, \phi_2, \lambda_2) = 2R \arcsin\sqrt{\sin^2\frac{\phi_2 - \phi_1}{2} + \cos\phi_1 \cos\phi_2 \sin^2\frac{\lambda_2 - \lambda_1}{2}}
$$

where $R$ is the Earth radius from `NUMKONG_EARTH_MEAN_RADIUS`, the IUGG mean radius of 6,371,008.8 m by default, and $(\phi, \lambda)$ are latitude and longitude in radians.
The kernels evaluate the same angle from unit vectors $\mathbf{a}$ and $\mathbf{b}$, which a pack holds, in a form that stays conditioned from coincident to antipodal points:

$$
\text{haversine}(\mathbf{a}, \mathbf{b}) = 2R \cdot \operatorname{atan2}(\lVert \mathbf{a} - \mathbf{b} \rVert, \lVert \mathbf{a} + \mathbf{b} \rVert)
$$

Vincenty's formula solves the inverse geodesic problem on an oblate spheroid, iteratively refining the reduced latitude difference until convergence:

$$
\text{vincenty}(\phi_1, \lambda_1, \phi_2, \lambda_2) = b \cdot A \cdot (\sigma - \Delta\sigma)
$$

where $a$ and $b$ are the equatorial and polar semi-axes of the configured Earth ellipsoid, $\sigma$ is the angular separation, and $\Delta\sigma$ is the correction term computed through iterative convergence.
The ellipsoid is WGS-84, the one GPS reports on, through `NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS` and `NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING`.

Reformulating as Python pseudocode:

```python
import numpy as np

def haversines(a_latitudes, a_longitudes, b_latitudes, b_longitudes, R=6371008.8):
    def unit(latitudes, longitudes):
        return np.stack([np.cos(latitudes) * np.cos(longitudes), np.cos(latitudes) * np.sin(longitudes), np.sin(latitudes)], axis=-1)

    a = unit(a_latitudes, a_longitudes)[:, None, :]
    b = unit(b_latitudes, b_longitudes)[None, :, :]
    return 2 * R * np.arctan2(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))
```

A clearance runs from a viewpoint to a target over a north-up grid of heights.
The terrain is sampled wherever the sight line crosses a row or column line through cell centres, each sample lowered by the Earth's curvature drop of $(1 - k) \cdot s^2 / 2R$ at ground distance $s$, where $k$ is the refraction, 1/7 for light and 0.25 for radio.
The clearance is the least gap between the sight line and those samples: positive or zero means visible by that margin, negative is the deepest obstruction, NaN means an endpoint lies off the grid, and $-\infty$ means the target lies outside the viewpoint's view cone or range.

## Input & Output Types

| Family                    | Input Type | Output Type | Description                                       |
| :------------------------ | :--------- | :---------- | :------------------------------------------------ |
| `sqeuclideans3d`          | `i32`      | `u64`       | Exact squares of 32-bit integer coordinates       |
| `sqeuclideans3d`          | `f64`      | `f64`       | 64-bit double precision, summed in Dot2           |
| `haversines`, `vincentys` | `f64`      | `f64`       | 64-bit double precision                           |
| `terrain`, `clearances`   | `f32`      | `f32`       | 32-bit heights and positions, computed in 64 bits |

## Optimizations

### Packing Once per Point Set

`nk_haversines_pack_f64` stores each point as a unit vector and `nk_vincentys_pack_f64` stores its longitude with the sine and cosine of its reduced latitude, in planar arrays every capability reads alike.
A rows × columns product then evaluates no trigonometry per point: Haversine reduces to two compensated norms and one `atan2` per pair, and Vincenty starts its iteration from precomputed reduced latitudes instead of an `atan` per pair.
Every CPU capability packs through the serial routine and CUDA through its twin, which rounds the same polynomials one operation at a time, so a pack built once is byte-identical wherever it is read.

### Exact 3D Squares

`nk_sqeuclideans3d_i32` subtracts 32-bit coordinates and squares the differences into 64-bit lanes, which stays exact for Earth-fixed centimetres below 4,359 km of altitude.
NEON de-interleaves rows with `LD3` and squares with `SMULL` and `SMLAL`, x86 with `VPMULDQ` after a permute, WASM with `i64x2.extmul`, and RVV with segmented loads and `vwmul`.
The library build turns auto-vectorization off, so every capability carries its own explicit SIMD body.

### Trigonometric Polynomial Approximations

Vincenty's iteration and every final `atan2` replace `libm` with SIMD polynomial approximations.
Range reduction maps the input angle to $[-\pi/2, \pi/2]$ using Cody-Waite extended precision constants, then an odd-degree minimax polynomial evaluates sin, and cos reuses the same polynomial after shifting the argument by $\pi/2$.
The `f64` kernels use 10-term polynomials.
This avoids the latency of scalar `libm` calls, each of which would otherwise serialize through a single execution port, while the polynomial chains pipeline across multiple FMA units.

### Vincenty Iterative Convergence with Masked Lanes

`nk_vincentys_packed_f64_haswell`, `nk_vincentys_packed_f64_skylake`, `nk_vincentys_packed_f64_neon` implement the full Vincenty inverse formula with up to 100 iterations and a convergence threshold of $10^{-12}$ radians, approximately 6 micrometers on Earth's surface.
Each SIMD lane may converge at a different iteration count, so the kernel accumulates a mask via `_mm256_or_pd(converged_mask_f64x4, newly_converged_f64x4)` and selectively freezes converged lanes with `_mm256_blendv_pd(lambda_next_f64x4, lambda_f64x4, converged_mask_f64x4)`.
Early exit uses `_mm256_movemask_pd`: when all 4 bits are set, the loop breaks.
Coincident points, where $\sin\sigma = 0$, and equatorial pairs, where $\cos^2\alpha = 0$, are detected exactly as the serial kernel detects them, and blend safe values into the intermediate terms to avoid division by zero without branches that would diverge across SIMD lanes.

### Terrain Pyramid

`nk_terrain_pack_f32` keeps the maximum of every $2^k \times 2^k$ block of the grid, one ring wider than the block, beside the caller's heights, which it never copies.
`nk_clearances_packed_f32` walks each sight line through the coarsest blocks first and skips a block whose maximum, lowered by the curvature at its nearest point, cannot undercut the running minimum.
Skipping only drops samples that cannot lower the result, so every clearance equals the one a full visit of every crossing gives.
On CUDA each thread walks one sight line through the serial recursion, unrolled into a stack of frames and rounded one operation at a time, so its clearances equal the serial ones bit for bit.

### Ranking by Chord

On a sphere the great-circle angle grows monotonically with the chord $\lVert \mathbf{a} - \mathbf{b} \rVert$, so ranking by `sqeuclideans3d` over unit vectors or Earth-fixed coordinates orders pairs as Haversine does, without `atan2` or square roots.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by the `NUMKONG_MAX_COORD_ANGLE` environment variable and set to ≤1°, ≤30°, and ≤180° maximum angular separation between pairs of coordinates.
The larger the angular separation between pairs, the longer the algorithm may take to converge and the higher the error.
The throughput is measured in MP/s as the number of Millions of pairwise point distances computed per second, over a `NUMWARS_DIMS_HEIGHT` × `NUMWARS_DIMS_WIDTH` matrix, 1024 × 128 by default.
Current `numkong_cpu_test` output reports geospatial accuracy in two forms: mean/max absolute error in meters against the same formula computed at double-double (f118) precision, and mean/max ULP against the matching high-precision implementation of the same formula.
Every row reads `?` until the many-to-many kernels are remeasured.

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.

#### Native

| Kernel                             |                      ≤1° |                     ≤30° |                    ≤180° |
| :--------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversines_packed_f64_serial`  |                        ? |                        ? |                        ? |
| `nk_vincentys_packed_f64_serial`   |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_f64_serial`     |                        ? |                        ? |                        ? |
| `nk_haversines_packed_f64_haswell` |                        ? |                        ? |                        ? |
| `nk_vincentys_packed_f64_haswell`  |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_f64_haswell`    |                        ? |                        ? |                        ? |
| `nk_haversines_packed_f64_skylake` |                        ? |                        ? |                        ? |
| `nk_vincentys_packed_f64_skylake`  |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_f64_skylake`    |                        ? |                        ? |                        ? |
| __i32__                            | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclideans3d_i32_serial`     |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_i32_haswell`    |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_i32_skylake`    |                        ? |                        ? |                        ? |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel                                 |                      ≤1° |                     ≤30° |                    ≤180° |
| :------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversines_packed_f64_serial`      |                        ? |                        ? |                        ? |
| `nk_vincentys_packed_f64_serial`       |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_f64_serial`         |                        ? |                        ? |                        ? |
| `nk_haversines_packed_f64_v128relaxed` |                        ? |                        ? |                        ? |
| `nk_vincentys_packed_f64_v128relaxed`  |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_f64_v128relaxed`    |                        ? |                        ? |                        ? |
| __i32__                                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclideans3d_i32_serial`         |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_i32_v128relaxed`    |                        ? |                        ? |                        ? |

### Apple M4

#### Native

| Kernel                            |                      ≤1° |                     ≤30° |                    ≤180° |
| :-------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversines_packed_f64_serial` |                        ? |                        ? |                        ? |
| `nk_vincentys_packed_f64_serial`  |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_f64_serial`    |                        ? |                        ? |                        ? |
| `nk_haversines_packed_f64_neon`   |                        ? |                        ? |                        ? |
| `nk_vincentys_packed_f64_neon`    |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_f64_neon`      |                        ? |                        ? |                        ? |
| __i32__                           | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclideans3d_i32_serial`    |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_i32_neon`      |                        ? |                        ? |                        ? |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                                 |                      ≤1° |                     ≤30° |                    ≤180° |
| :------------------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_haversines_packed_f64_serial`      |                        ? |                        ? |                        ? |
| `nk_vincentys_packed_f64_serial`       |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_f64_serial`         |                        ? |                        ? |                        ? |
| `nk_haversines_packed_f64_v128relaxed` |                        ? |                        ? |                        ? |
| `nk_vincentys_packed_f64_v128relaxed`  |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_f64_v128relaxed`    |                        ? |                        ? |                        ? |
| __i32__                                | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_sqeuclideans3d_i32_serial`         |                        ? |                        ? |                        ? |
| `nk_sqeuclideans3d_i32_v128relaxed`    |                        ? |                        ? |                        ? |
