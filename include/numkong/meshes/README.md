# Point Cloud Alignment in NumKong

NumKong implements three algorithms for 3D point cloud comparison and alignment, used in structural biology (protein alignment), robotics (point cloud registration), and computer graphics (mesh registration).

RMSD measures raw point-pair deviation without centering or alignment:

$$
\text{RMSD} = \sqrt{\frac{1}{n}\sum \|a_i - b_i\|^2}
$$

Kabsch finds the optimal rotation $R$ that minimizes RMSD after centering both clouds at their centroids $\bar{a}$, $\bar{b}$, recovering $R$ from the SVD of the cross-covariance matrix $H$:

$$
H = \sum (a_i - \bar{a})(b_i - \bar{b})^T = U \Sigma V^T, \quad R = V U^T
$$

Umeyama extends Kabsch with a uniform scale factor $s$ derived from the singular values and source variance $\sigma_a^2$:

$$
s = \frac{\text{tr}(\Sigma)}{n \cdot \sigma_a^2}, \quad \text{RMSD} = \sqrt{\frac{1}{n}\sum \|s \cdot R(a_i - \bar{a}) - (b_i - \bar{b})\|^2}
$$

Reformulating as Python pseudocode:

```python
import numpy as np

def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_c, b_c = a - a.mean(0), b - b.mean(0)
    H = a_c.T @ b_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R

def umeyama(a: np.ndarray, b: np.ndarray) -> tuple:
    a_c, b_c = a - a.mean(0), b - b.mean(0)
    H = a_c.T @ b_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    scale = S.sum() / (len(a) * np.var(a_c))
    return R, scale

def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))
```

## Input & Output Types

| Input Type | Output Type | Description                                    |
| :--------- | :---------- | :--------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision               |
| `f32`      | `f64`       | 32-bit IEEE 754 single precision, widened RMSD |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output |
| `bf16`     | `f32`       | 16-bit brain float, widened output             |

The output type above is the RMSD result.
The centroid, rotation, and scale outputs use the transform type instead: `f64` for `f64` inputs, `f32` for `f32`, `f16`, and `bf16` inputs.

## Optimizations

### McAdams Branching-Free 3×3 SVD

`nk_kabsch_f32_serial`, `nk_kabsch_f64_haswell`, `nk_umeyama_f32_neon` use a Jacobi eigenanalysis with fixed 16 iterations (no convergence check) for deterministic behavior.
Quaternion-accumulated rotations: each Jacobi sweep updates a 4-element quaternion instead of recomputing eigenvectors.
Approximate Givens angles via `nk_approximate_givens_quaternion_` — a γ-threshold test selects between computed angles and precomputed cos(π/8), sin(π/8) constants.
Cyclic permutation of matrix elements avoids explicit sorting of eigenvalues.

### Stride-3 Deinterleaving

Point clouds are stored interleaved as [x₀,y₀,z₀, x₁,y₁,z₁, ...].
NEON uses `vld3q_f32` to hardware-deinterleave 4 XYZ triplets in one instruction — no gather needed.
Haswell uses `_mm256_i32gather_ps` with indices [0,3,6,9,12,15,18,21] to load 8 x-coordinates from 8 points.
RVV uses indexed loads with dynamic stride to adapt to variable vector length.

### Reflection Correction

`nk_kabsch_f32_haswell`, `nk_kabsch_f64_skylake` check for improper rotations after computing $R = V U^T$ from the SVD of the cross-covariance matrix $H = U \Sigma V^T$.
If $\det(R) = -1$ (a reflection rather than a rotation), the last column of $V$ is negated before recomputing $R$.
This ensures the output is always a proper rotation matrix with $\det(R) = +1$.

### Pre-Scaled Rotation for Umeyama

`nk_umeyama_f32_haswell`, `nk_umeyama_f64_skylake` fold the computed scale factor $s$ into the rotation matrix before applying to points.
The Umeyama transform is $b_i = s R a_i + t$; by precomputing $R' = s R$ once, the per-point operation reduces to $b_i = R' a_i + t$, avoiding a per-point scalar multiply.

### Pivot-Shifted Moments and Residual SSD

Kabsch and Umeyama shift every point by the first one of its cloud before accumulating the sums and products, so the centering correction scales with the spread of the cloud rather than its distance from the origin.
`f64` and `f32` inputs then take a second pass, summing $\|s R (a_i - \bar{a}) - (b_i - \bar{b})\|^2$ directly, as folding it into $\|A\|^2 + \|B\|^2 - 2 s \operatorname{tr}(R H)$ cancels to $\sqrt{\varepsilon}$ once the clouds align.
`f32` inputs run both passes in `f64`.
`f16` and `bf16` inputs keep a single pass with the folded form, clamped at zero, since its $\sqrt{\varepsilon}$ of `f32` stays below their own quantization.
FMLAL shifts `f16` points with FP16 subtraction, while BFDOT and VDPBF16PS shift `bf16` points in `f32` and round them back, each rounding by at most half a ULP of the shifted value.
Clouds with a point over 65504 away from its pivot overflow the FP16 shift and take the widening NEON pass instead.
Tests judge these kernels against a bound derived from that arithmetic: the input type's unit roundoff over the clouds' RMS distance from their pivots, plus $n$ roundings of the `f32` moments.

### Why SME and SVE Were Removed

Historical note: experimental SME variants of RMSD, Kabsch, and Umeyama were implemented in 1,052 lines across `sme.h` and `smef64.h` (commit `0e0bc30c`) and removed 4 days later (commit `f55e9a71`).
The fundamental mismatch: the algorithm computes a 3×3 cross-covariance matrix $H = \sum (a_i - \bar{a})(b_i - \bar{b})^T$ — a sum of outer products of 3D vectors.
SME's `FMOPA` operates on SVL-wide vectors (16+ elements at SVL=512), but the outer products here are 3×3 — the tile is 99.6% wasted (9 useful cells out of 256).
Three approaches were explored in a design document that never landed in the repository:
(1) batched outer products — reformulates as 9 independent dot products but loses SME's outer-product strength, falling back to what NEON already does;
(2) streaming SVE with `svld3` — hardware stride-3 deinterleaving processes 16 points per iteration vs NEON's 4, but `SMSTART`/`SMSTOP` mode transitions cost ~100 cycles and the 3×3 SVD step cannot use streaming mode at all;
(3) SME for SVD — the 3×3 matrix cannot fill even one 16×16 tile.
Performance estimates from the design document: NEON baseline ~2.25N cycles for N points; streaming SVE ~1.2N cycles but with ~100-cycle mode transition overhead — for typical protein alignment workloads (N = 100–500 atoms), the overhead dominates.
Experimental SVE mesh kernels (`sve.h`, `svehalf.h`, 112 lines total) were removed in the same commit — variable vector length added complexity without clear benefit over fixed-width NEON for the 3D point cloud problem.

## Performance

The tables below follow the [benchmark methodology](../../../bench/README.md#methodology).
The input size is controlled by the `NUMKONG_MESH_POINTS` environment variable and set to 256, 1024, and 4096 points.
Each alignment computes centroids, covariance, and a 3×3 SVD over $N$ point pairs, so cost is $O(N)$ per alignment with a large constant.
The throughput is measured in mp/s as millions of 3D points aligned per second.

### Intel Xeon 6 with B300

Rows ran single-threaded on one pinned core of an Intel Xeon 6787P, a Granite Rapids part.
Serial kernels compiled with `-fno-tree-vectorize`.

#### Native

| Kernel                    |                      256 |                     1024 |                     4096 |
| :------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f64_serial`      |       73.1 mp/s, 0.5 ulp |       46.3 mp/s, 0.5 ulp |       47.5 mp/s, 0.5 ulp |
| `nk_kabsch_f64_serial`    |       8.68 mp/s, 0.8 ulp |       8.62 mp/s, 0.8 ulp |       10.7 mp/s, 0.8 ulp |
| `nk_umeyama_f64_serial`   |       7.34 mp/s, 0.3 ulp |       7.84 mp/s, 0.3 ulp |       9.06 mp/s, 0.3 ulp |
| `nk_rmsd_f64_haswell`     |        495 mp/s, 0.3 ulp |        514 mp/s, 0.4 ulp |        268 mp/s, 0.8 ulp |
| `nk_kabsch_f64_haswell`   |       51.7 mp/s, 0.5 ulp |        126 mp/s, 0.9 ulp |        130 mp/s, 1.5 ulp |
| `nk_umeyama_f64_haswell`  |       51.0 mp/s, 0.5 ulp |        106 mp/s, 0.8 ulp |        123 mp/s, 1.5 ulp |
| `nk_rmsd_f64_skylake`     |        502 mp/s, 0.2 ulp |        539 mp/s, 0.3 ulp |        235 mp/s, 0.4 ulp |
| `nk_kabsch_f64_skylake`   |       51.3 mp/s, 0.4 ulp |        117 mp/s, 0.5 ulp |        125 mp/s, 0.8 ulp |
| `nk_umeyama_f64_skylake`  |       51.3 mp/s, 0.3 ulp |        114 mp/s, 0.5 ulp |        126 mp/s, 0.8 ulp |
| __f32__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f32_serial`      |       50.3 mp/s, 0.5 ulp |       49.5 mp/s, 0.5 ulp |       50.0 mp/s, 0.5 ulp |
| `nk_kabsch_f32_serial`    |       7.42 mp/s, 0.8 ulp |       7.73 mp/s, 0.8 ulp |       8.58 mp/s, 0.9 ulp |
| `nk_umeyama_f32_serial`   |       7.18 mp/s, 0.3 ulp |       7.82 mp/s, 0.3 ulp |       8.31 mp/s, 0.4 ulp |
| `nk_rmsd_f32_haswell`     |        582 mp/s, 0.3 ulp |        644 mp/s, 0.5 ulp |        377 mp/s, 0.9 ulp |
| `nk_kabsch_f32_haswell`   |       52.1 mp/s, 0.9 ulp |       94.5 mp/s, 1.3 ulp |        119 mp/s, 7.6 ulp |
| `nk_umeyama_f32_haswell`  |       51.3 mp/s, 0.3 ulp |       91.7 mp/s, 0.4 ulp |        120 mp/s, 0.7 ulp |
| `nk_rmsd_f32_skylake`     |        841 mp/s, 1.2 ulp |        972 mp/s, 1.2 ulp |        697 mp/s, 4.3 ulp |
| `nk_kabsch_f32_skylake`   |       58.3 mp/s, 0.9 ulp |        129 mp/s, 4.1 ulp |        154 mp/s, 3.1 ulp |
| `nk_umeyama_f32_skylake`  |       57.0 mp/s, 0.6 ulp |        126 mp/s, 2.9 ulp |        149 mp/s, 2.1 ulp |
| __bf16__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_bf16_haswell`    |        209 mp/s, 0.3 ulp |        205 mp/s, 3.5 ulp |       193 mp/s, 12.8 ulp |
| `nk_kabsch_bf16_haswell`  |       28.9 mp/s, 0.4 ulp |       79.9 mp/s, 7.6 ulp |       134 mp/s, 33.0 ulp |
| `nk_umeyama_bf16_haswell` |       29.0 mp/s, 0.3 ulp |       75.8 mp/s, 5.3 ulp |       124 mp/s, 23.1 ulp |
| `nk_rmsd_bf16_skylake`    |      1,538 mp/s, 0.4 ulp |      1,922 mp/s, 5.4 ulp |     2,053 mp/s, 11.8 ulp |
| `nk_kabsch_bf16_skylake`  |       30.8 mp/s, 0.3 ulp |        114 mp/s, 3.2 ulp |       260 mp/s, 20.4 ulp |
| `nk_umeyama_bf16_skylake` |       31.1 mp/s, 0.3 ulp |        110 mp/s, 2.2 ulp |       254 mp/s, 14.3 ulp |
| `nk_rmsd_bf16_genoa`      |      1,543 mp/s, 0.3 ulp |      1,859 mp/s, 3.1 ulp |     2,068 mp/s, 20.2 ulp |
| `nk_kabsch_bf16_genoa`    |       30.5 mp/s, 0.3 ulp |        106 mp/s, 3.2 ulp |       188 mp/s, 20.3 ulp |
| `nk_umeyama_bf16_genoa`   |       30.7 mp/s, 0.3 ulp |       98.9 mp/s, 2.2 ulp |       215 mp/s, 14.3 ulp |
| __f16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f16_haswell`     |        204 mp/s, 0.2 ulp |        209 mp/s, 0.7 ulp |        197 mp/s, 2.5 ulp |
| `nk_kabsch_f16_haswell`   |       29.8 mp/s, 0.5 ulp |       82.4 mp/s, 1.8 ulp |        131 mp/s, 8.2 ulp |
| `nk_umeyama_f16_haswell`  |       30.0 mp/s, 0.4 ulp |       80.4 mp/s, 1.2 ulp |        133 mp/s, 5.7 ulp |
| `nk_rmsd_f16_skylake`     |      1,567 mp/s, 0.3 ulp |      1,947 mp/s, 1.3 ulp |      2,006 mp/s, 3.9 ulp |
| `nk_kabsch_f16_skylake`   |       31.6 mp/s, 0.7 ulp |        108 mp/s, 0.5 ulp |        279 mp/s, 4.7 ulp |
| `nk_umeyama_f16_skylake`  |       31.5 mp/s, 0.5 ulp |        109 mp/s, 0.4 ulp |        266 mp/s, 3.3 ulp |

#### WASM

Measured with wasmtime 49.0.2, Cranelift.

| Kernel                       |                      256 |                     1024 |                     4096 |
| :--------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f64_serial`         |       68.2 mp/s, 0.5 ulp |       66.8 mp/s, 0.5 ulp |       64.9 mp/s, 0.5 ulp |
| `nk_rmsd_f64_v128relaxed`    |        546 mp/s, 0.4 ulp |        586 mp/s, 0.7 ulp |        576 mp/s, 1.3 ulp |
| `nk_kabsch_f64_serial`       |       9.37 mp/s, 0.8 ulp |       10.5 mp/s, 0.8 ulp |       10.9 mp/s, 0.9 ulp |
| `nk_kabsch_f64_v128relaxed`  |       48.7 mp/s, 0.9 ulp |       96.2 mp/s, 1.7 ulp |        128 mp/s, 3.1 ulp |
| `nk_umeyama_f64_serial`      |       6.45 mp/s, 0.3 ulp |       6.88 mp/s, 0.3 ulp |       7.19 mp/s, 0.4 ulp |
| `nk_umeyama_f64_v128relaxed` |       47.2 mp/s, 0.8 ulp |       94.1 mp/s, 1.6 ulp |        125 mp/s, 3.2 ulp |
| __f32__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f32_serial`         |       58.2 mp/s, 0.5 ulp |       57.8 mp/s, 0.5 ulp |       54.4 mp/s, 0.5 ulp |
| `nk_rmsd_f32_v128relaxed`    |        356 mp/s, 1.5 ulp |        369 mp/s, 1.3 ulp |        370 mp/s, 4.8 ulp |
| `nk_kabsch_f32_serial`       |       7.84 mp/s, 0.8 ulp |       8.60 mp/s, 0.9 ulp |       8.82 mp/s, 0.8 ulp |
| `nk_kabsch_f32_v128relaxed`  |       43.9 mp/s, 4.2 ulp |       66.2 mp/s, 3.9 ulp |      98.4 mp/s, 14.3 ulp |
| `nk_umeyama_f32_serial`      |       5.67 mp/s, 0.3 ulp |       6.04 mp/s, 0.3 ulp |       6.22 mp/s, 0.3 ulp |
| `nk_umeyama_f32_v128relaxed` |       43.3 mp/s, 2.8 ulp |       65.1 mp/s, 2.8 ulp |      93.8 mp/s, 10.1 ulp |

### Apple M5

#### Native

| Kernel                      |                      256 |                     1024 |                     4096 |
| :-------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f64_serial`        |        279 mp/s, 0.5 ulp |        267 mp/s, 0.5 ulp |        279 mp/s, 0.5 ulp |
| `nk_kabsch_f64_serial`      |       40.4 mp/s, 1.4 ulp |       47.3 mp/s, 2.6 ulp |       50.2 mp/s, 5.4 ulp |
| `nk_umeyama_f64_serial`     |       34.5 mp/s, 1.0 ulp |       39.2 mp/s, 1.9 ulp |       41.6 mp/s, 3.7 ulp |
| `nk_rmsd_f64_neon`          |      1,776 mp/s, 0.4 ulp |      1,536 mp/s, 0.7 ulp |      2,037 mp/s, 1.3 ulp |
| `nk_kabsch_f64_neon`        |        119 mp/s, 0.8 ulp |        222 mp/s, 1.3 ulp |        304 mp/s, 2.2 ulp |
| `nk_umeyama_f64_neon`       |        115 mp/s, 0.4 ulp |        220 mp/s, 0.8 ulp |        296 mp/s, 1.6 ulp |
| __f32__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f32_serial`        |        264 mp/s, 0.5 ulp |        264 mp/s, 0.5 ulp |        261 mp/s, 0.5 ulp |
| `nk_kabsch_f32_serial`      |       39.4 mp/s, 1.4 ulp |       46.0 mp/s, 2.7 ulp |       49.9 mp/s, 5.0 ulp |
| `nk_umeyama_f32_serial`     |       33.6 mp/s, 0.9 ulp |       38.8 mp/s, 1.8 ulp |       41.4 mp/s, 3.5 ulp |
| `nk_rmsd_f32_neon`          |      1,912 mp/s, 1.5 ulp |      2,239 mp/s, 1.3 ulp |      1,966 mp/s, 4.8 ulp |
| `nk_kabsch_f32_neon`        |        135 mp/s, 0.7 ulp |        288 mp/s, 0.9 ulp |        385 mp/s, 1.4 ulp |
| `nk_umeyama_f32_neon`       |        130 mp/s, 0.3 ulp |        272 mp/s, 0.4 ulp |        367 mp/s, 0.8 ulp |
| __bf16__                    | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_bf16_neonbfdot`    |      3,728 mp/s, 0.4 ulp |      3,756 mp/s, 6.0 ulp |     3,769 mp/s, 10.0 ulp |
| `nk_kabsch_bf16_neonbfdot`  |        180 mp/s, 0.7 ulp |        448 mp/s, 0.9 ulp |        726 mp/s, 1.3 ulp |
| `nk_umeyama_bf16_neonbfdot` |        176 mp/s, 0.2 ulp |        433 mp/s, 0.4 ulp |        705 mp/s, 0.8 ulp |
| __f16__                     | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f16_neon`          |      2,998 mp/s, 0.4 ulp |      3,215 mp/s, 1.7 ulp |      3,216 mp/s, 4.6 ulp |
| `nk_kabsch_f16_neon`        |        178 mp/s, 0.9 ulp |        443 mp/s, 1.3 ulp |        711 mp/s, 2.4 ulp |
| `nk_umeyama_f16_neon`       |        175 mp/s, 0.4 ulp |        408 mp/s, 0.8 ulp |        620 mp/s, 1.5 ulp |

#### WASM

Measured with Wasmtime v43 (Cranelift backend).

| Kernel                       |                      256 |                     1024 |                     4096 |
| :--------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f64_serial`         |        137 mp/s, 2.6 ulp |        134 mp/s, 2.6 ulp |        142 mp/s, 2.6 ulp |
| `nk_rmsd_f64_v128relaxed`    |      1,377 mp/s, 0.8 ulp |      1,038 mp/s, 0.8 ulp |      1,566 mp/s, 0.8 ulp |
| `nk_kabsch_f64_serial`       |       42.3 mp/s, 2.7 ulp |       50.4 mp/s, 2.7 ulp |       55.5 mp/s, 2.7 ulp |
| `nk_kabsch_f64_v128relaxed`  |        121 mp/s, 2.2 ulp |        225 mp/s, 2.2 ulp |        345 mp/s, 2.2 ulp |
| `nk_umeyama_f64_serial`      |       36.1 mp/s, 1.8 ulp |       41.3 mp/s, 1.8 ulp |       46.0 mp/s, 1.8 ulp |
| `nk_umeyama_f64_v128relaxed` |        112 mp/s, 1.5 ulp |        207 mp/s, 1.5 ulp |        293 mp/s, 1.5 ulp |
| __f32__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f32_serial`         |        120 mp/s, 2.7 ulp |        120 mp/s, 2.7 ulp |        124 mp/s, 2.7 ulp |
| `nk_rmsd_f32_v128relaxed`    |      1,025 mp/s, 0.5 ulp |      1,038 mp/s, 0.5 ulp |      1,093 mp/s, 0.5 ulp |
| `nk_kabsch_f32_serial`       |       39.6 mp/s, 2.6 ulp |       47.6 mp/s, 2.6 ulp |       51.4 mp/s, 2.6 ulp |
| `nk_kabsch_f32_v128relaxed`  |        125 mp/s, 1.3 ulp |        255 mp/s, 1.3 ulp |        366 mp/s, 1.3 ulp |
| `nk_umeyama_f32_serial`      |       30.5 mp/s, 1.8 ulp |       35.0 mp/s, 1.8 ulp |       38.9 mp/s, 1.8 ulp |
| `nk_umeyama_f32_v128relaxed` |        118 mp/s, 0.8 ulp |        240 mp/s, 0.8 ulp |        338 mp/s, 0.8 ulp |
