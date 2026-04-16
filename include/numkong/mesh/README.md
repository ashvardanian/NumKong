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
| ---------- | ----------- | ---------------------------------------------- |
| `f64`      | `f64`       | 64-bit IEEE 754 double precision               |
| `f32`      | `f32`       | 32-bit IEEE 754 single precision               |
| `f16`      | `f32`       | 16-bit IEEE 754 half precision, widened output |
| `bf16`     | `f32`       | 16-bit brain float, widened output             |

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

### Why SME and SVE Were Removed

Historical note: experimental SME variants of RMSD, Kabsch, and Umeyama were implemented in 1,052 lines across `sme.h` and `smef64.h` (commit `0e0bc30c`) and removed 4 days later (commit `f55e9a71`).
The fundamental mismatch: the algorithm computes a 3×3 cross-covariance matrix $H = \sum (a_i - \bar{a})(b_i - \bar{b})^T$ — a sum of outer products of 3D vectors.
SME's `FMOPA` operates on SVL-wide vectors (16+ elements at SVL=512), but the outer products here are 3×3 — the tile is 99.6% wasted (9 useful cells out of 256).
Three approaches were explored in a design document (`sme_design.h`, 398 lines):
(1) batched outer products — reformulates as 9 independent dot products but loses SME's outer-product strength, falling back to what NEON already does;
(2) streaming SVE with `svld3` — hardware stride-3 deinterleaving processes 16 points per iteration vs NEON's 4, but `SMSTART`/`SMSTOP` mode transitions cost ~100 cycles and the 3×3 SVD step cannot use streaming mode at all;
(3) SME for SVD — the 3×3 matrix cannot fill even one 16×16 tile.
Performance estimates from the design document: NEON baseline ~2.25N cycles for N points; streaming SVE ~1.2N cycles but with ~100-cycle mode transition overhead — for typical protein alignment workloads (N = 100–500 atoms), the overhead dominates.
Experimental SVE mesh kernels (`sve.h`, `svehalf.h`, 112 lines total) were removed in the same commit — variable vector length added complexity without clear benefit over fixed-width NEON for the 3D point cloud problem.

## Performance

The following performance tables are produced by manually re-running `nk_test` and `nk_bench` included internal tools to measure both accuracy and throughput at different input shapes.
The input size is controlled by the `NK_MESH_POINTS` environment variable and set to 256, 1024, and 4096 points.
Each alignment computes centroids, covariance, and a 3×3 SVD over $N$ point pairs, so cost is $O(N)$ per alignment with a large constant.
The throughput is measured in mp/s as millions of 3D points aligned per second.
Accuracy is reported as mean ULP (units in last place) unless noted otherwise — the average number of representable floating-point values between the result and the exact answer.
Each kernel runs for at least 20 seconds per configuration.
Benchmark threads are pinned to specific cores; on machines with heterogeneous core types (e.g., Apple P/E cores), only the fastest cores are used.
Workloads that significantly degrade CPU frequencies (Intel AMX, Apple SME) run in separate passes to avoid affecting throughput measurements of other kernels.

### Intel Granite Rapids

Xeon 6776P, 2.3 GHz base, `cpu_scaling_enabled=false`.
Serial kernels compiled with `-fno-tree-vectorize`.

#### Native

| Kernel                    |                      256 |                     1024 |                     4096 |
| :------------------------ | -----------------------: | -----------------------: | -----------------------: |
| __f64__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f64_serial`      |       93.7 mp/s, 0.5 ulp |       87.4 mp/s, 0.5 ulp |       69.8 mp/s, 0.5 ulp |
| `nk_kabsch_f64_serial`    |       11.8 mp/s, 0.8 ulp |       13.6 mp/s, 0.8 ulp |       12.8 mp/s, 0.8 ulp |
| `nk_umeyama_f64_serial`   |       10.4 mp/s, 0.3 ulp |       11.7 mp/s, 0.3 ulp |       11.5 mp/s, 0.3 ulp |
| `nk_rmsd_f64_haswell`     |        523 mp/s, 0.3 ulp |        564 mp/s, 0.4 ulp |        449 mp/s, 0.8 ulp |
| `nk_kabsch_f64_haswell`   |       65.3 mp/s, 0.5 ulp |        203 mp/s, 0.9 ulp |        326 mp/s, 1.5 ulp |
| `nk_umeyama_f64_haswell`  |       68.0 mp/s, 0.5 ulp |        200 mp/s, 0.8 ulp |        324 mp/s, 1.5 ulp |
| `nk_rmsd_f64_skylake`     |        546 mp/s, 0.2 ulp |        587 mp/s, 0.3 ulp |        583 mp/s, 0.4 ulp |
| `nk_kabsch_f64_skylake`   |       34.5 mp/s, 0.4 ulp |        107 mp/s, 0.5 ulp |        261 mp/s, 0.8 ulp |
| `nk_umeyama_f64_skylake`  |       24.3 mp/s, 0.3 ulp |       82.7 mp/s, 0.5 ulp |        201 mp/s, 0.8 ulp |
| __f32__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f32_serial`      |       68.9 mp/s, 0.5 ulp |       70.7 mp/s, 0.5 ulp |       72.1 mp/s, 0.5 ulp |
| `nk_kabsch_f32_serial`    |       11.2 mp/s, 0.8 ulp |       12.8 mp/s, 0.8 ulp |       14.0 mp/s, 0.9 ulp |
| `nk_umeyama_f32_serial`   |       10.1 mp/s, 0.3 ulp |       11.2 mp/s, 0.3 ulp |       12.1 mp/s, 0.4 ulp |
| `nk_rmsd_f32_haswell`     |        686 mp/s, 0.3 ulp |        848 mp/s, 0.5 ulp |        841 mp/s, 0.9 ulp |
| `nk_kabsch_f32_haswell`   |       90.4 mp/s, 0.9 ulp |        250 mp/s, 1.3 ulp |        455 mp/s, 7.6 ulp |
| `nk_umeyama_f32_haswell`  |       87.7 mp/s, 0.3 ulp |        250 mp/s, 0.4 ulp |        374 mp/s, 0.7 ulp |
| `nk_rmsd_f32_skylake`     |      1,016 mp/s, 1.2 ulp |      1,112 mp/s, 1.2 ulp |      1,042 mp/s, 4.3 ulp |
| `nk_kabsch_f32_skylake`   |       81.8 mp/s, 0.9 ulp |        241 mp/s, 4.1 ulp |        549 mp/s, 3.1 ulp |
| `nk_umeyama_f32_skylake`  |       58.0 mp/s, 0.6 ulp |        168 mp/s, 2.9 ulp |        459 mp/s, 2.1 ulp |
| __bf16__                  | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_bf16_haswell`    |        284 mp/s, 0.3 ulp |        281 mp/s, 3.5 ulp |       273 mp/s, 12.8 ulp |
| `nk_kabsch_bf16_haswell`  |       36.2 mp/s, 0.4 ulp |        106 mp/s, 7.6 ulp |       186 mp/s, 33.0 ulp |
| `nk_umeyama_bf16_haswell` |       34.5 mp/s, 0.3 ulp |        102 mp/s, 5.3 ulp |       186 mp/s, 23.1 ulp |
| `nk_rmsd_bf16_skylake`    |      1,837 mp/s, 0.4 ulp |      2,357 mp/s, 5.4 ulp |     2,422 mp/s, 11.8 ulp |
| `nk_kabsch_bf16_skylake`  |       34.1 mp/s, 0.3 ulp |        131 mp/s, 3.2 ulp |       487 mp/s, 20.4 ulp |
| `nk_umeyama_bf16_skylake` |       34.6 mp/s, 0.3 ulp |        130 mp/s, 2.2 ulp |       394 mp/s, 14.3 ulp |
| `nk_rmsd_bf16_genoa`      |      1,743 mp/s, 0.3 ulp |      2,323 mp/s, 3.1 ulp |     2,066 mp/s, 20.2 ulp |
| `nk_kabsch_bf16_genoa`    |       33.4 mp/s, 0.3 ulp |        133 mp/s, 3.2 ulp |       405 mp/s, 20.3 ulp |
| `nk_umeyama_bf16_genoa`   |       33.2 mp/s, 0.3 ulp |        129 mp/s, 2.2 ulp |       439 mp/s, 14.3 ulp |
| __f16__                   | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f16_haswell`     |        273 mp/s, 0.2 ulp |        274 mp/s, 0.7 ulp |        291 mp/s, 2.5 ulp |
| `nk_kabsch_f16_haswell`   |       34.4 mp/s, 0.5 ulp |       98.0 mp/s, 1.8 ulp |        197 mp/s, 8.2 ulp |
| `nk_umeyama_f16_haswell`  |       35.5 mp/s, 0.4 ulp |       97.9 mp/s, 1.2 ulp |        196 mp/s, 5.7 ulp |
| `nk_rmsd_f16_skylake`     |      1,834 mp/s, 0.3 ulp |      2,341 mp/s, 1.3 ulp |      2,418 mp/s, 3.9 ulp |
| `nk_kabsch_f16_skylake`   |       34.0 mp/s, 0.7 ulp |        132 mp/s, 0.5 ulp |        480 mp/s, 4.7 ulp |
| `nk_umeyama_f16_skylake`  |       33.8 mp/s, 0.5 ulp |        127 mp/s, 0.4 ulp |        481 mp/s, 3.3 ulp |

#### WASM

Measured with Wasmtime v43 (Cranelift backend), WASI-SDK 24, `-msimd128 -mrelaxed-simd`.

| Kernel                       |                      256 |                     1024 |                     4096 |
| :--------------------------- | -----------------------: | -----------------------: | -----------------------: |
| __f64__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f64_serial`         |       89.9 mp/s, 0.5 ulp |       86.1 mp/s, 0.5 ulp |       73.4 mp/s, 0.5 ulp |
| `nk_rmsd_f64_v128relaxed`    |        485 mp/s, 0.4 ulp |        552 mp/s, 0.7 ulp |        412 mp/s, 1.3 ulp |
| `nk_kabsch_f64_serial`       |       12.1 mp/s, 0.8 ulp |       13.9 mp/s, 0.8 ulp |       14.0 mp/s, 0.9 ulp |
| `nk_kabsch_f64_v128relaxed`  |       66.0 mp/s, 0.9 ulp |        188 mp/s, 1.7 ulp |        177 mp/s, 3.1 ulp |
| `nk_umeyama_f64_serial`      |       10.8 mp/s, 0.3 ulp |       12.3 mp/s, 0.3 ulp |       12.2 mp/s, 0.4 ulp |
| `nk_umeyama_f64_v128relaxed` |       64.0 mp/s, 0.8 ulp |        187 mp/s, 1.6 ulp |        178 mp/s, 3.2 ulp |
| __f32__                      | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ | ░░░░░░░░░░░░░░░░░░░░░░░░ |
| `nk_rmsd_f32_serial`         |       80.6 mp/s, 0.5 ulp |       82.7 mp/s, 0.5 ulp |       70.3 mp/s, 0.5 ulp |
| `nk_rmsd_f32_v128relaxed`    |        452 mp/s, 1.5 ulp |        416 mp/s, 1.3 ulp |        399 mp/s, 4.8 ulp |
| `nk_kabsch_f32_serial`       |       11.4 mp/s, 0.8 ulp |       12.8 mp/s, 0.9 ulp |       12.7 mp/s, 0.8 ulp |
| `nk_kabsch_f32_v128relaxed`  |       79.5 mp/s, 4.2 ulp |        132 mp/s, 3.9 ulp |       177 mp/s, 14.3 ulp |
| `nk_umeyama_f32_serial`      |       10.1 mp/s, 0.3 ulp |       11.2 mp/s, 0.3 ulp |       11.2 mp/s, 0.3 ulp |
| `nk_umeyama_f32_v128relaxed` |       79.4 mp/s, 2.8 ulp |        138 mp/s, 2.8 ulp |       194 mp/s, 10.1 ulp |


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
| `nk_rmsd_f16_neonhalf`      |      2,998 mp/s, 0.4 ulp |      3,215 mp/s, 1.7 ulp |      3,216 mp/s, 4.6 ulp |
| `nk_kabsch_f16_neonhalf`    |        178 mp/s, 0.9 ulp |        443 mp/s, 1.3 ulp |        711 mp/s, 2.4 ulp |
| `nk_umeyama_f16_neonhalf`   |        175 mp/s, 0.4 ulp |        408 mp/s, 0.8 ulp |        620 mp/s, 1.5 ulp |

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
