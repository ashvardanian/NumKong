# Many-to-Many Shape Distances in NumKong

NumKong compares 3D point clouds of any size, with no correspondence between their points, scoring every cloud of one list against every cloud of another.
The uses are shape search over molecules and conformers, protein structures, and scanned or sampled surfaces.
Every kind is invariant to rotation, reflection and translation, and between normalized packs to scale as well.

Chamfer charges every point the distance to its nearest partner in the other cloud, in both directions, and Hausdorff keeps the worst of those charges:

$$
\text{Chamfer}(A, B) = \frac{1}{2} \left( \frac{1}{n} \sum_i \min_j \|a_i - b_j\| + \frac{1}{m} \sum_j \min_i \|a_i - b_j\| \right)
$$

$$
\text{Hausdorff}(A, B) = \max\left( \max_i \min_j \|a_i - b_j\|, \max_j \min_i \|a_i - b_j\| \right)
$$

Both are taken over the 8 sign flips of the two clouds' principal axes, keeping the best.
Wasserstein-Procrustes pairs points by an exact optimal transport plan $P$ between uniform weights and superposes them by the best rotation or reflection:

$$
\text{WP}(A, B) = \min_{R \in O(3)} \min_P \sqrt{\sum_{ij} P_{ij} \|a_i - R b_j\|^2}
$$

On equal sizes with a known correspondence it is the Kabsch RMSD.
Gromov-Wasserstein compares the clouds' own distance matrices, so it needs no superposition at all:

$$
\text{GW}(A, B) = \frac{1}{2} \min_P \sqrt{\sum_{ijkl} \left( \|a_i - a_k\| - \|b_j - b_l\| \right)^2 P_{ij} P_{kl}}
$$

Reformulating as Python pseudocode over clouds already centered and rotated onto their principal axes:

```python
import numpy as np
from itertools import product
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

flips = [np.array(signs) for signs in product([1, -1], repeat=3)]

def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    return min(0.5 * (d.min(1).mean() + d.min(0).mean()) for d in (cdist(a, b * s) for s in flips))

def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    return min(max(d.min(1).max(), d.min(0).max()) for d in (cdist(a, b * s) for s in flips))

def transport(cost: np.ndarray) -> np.ndarray:
    n, m = cost.shape
    rows = np.kron(np.eye(n), np.ones(m)); columns = np.kron(np.ones(n), np.eye(m))
    plan = linprog(cost.ravel(), A_eq=np.vstack([rows, columns]),
                   b_eq=np.concatenate([np.full(n, 1 / n), np.full(m, 1 / m)])).x
    return plan.reshape(n, m)

def wasserstein_procrustes(a: np.ndarray, b: np.ndarray) -> float:
    best = np.inf
    for s in flips:
        rotation, value = np.diag(s), np.inf
        while True:
            plan = transport(cdist(a, b @ rotation.T, "sqeuclidean"))
            new = (plan * cdist(a, b @ rotation.T, "sqeuclidean")).sum()
            if not new < value: break
            value = new
            u, _, vt = np.linalg.svd(b.T @ plan.T @ a)
            sign = np.sign(np.linalg.det(vt.T @ u.T)) * np.prod(s)
            rotation = vt.T @ np.diag([1, 1, sign]) @ u.T
        best = min(best, value)
    return np.sqrt(best)
```

## Input & Output Types

| Input Type | Math Type | Distance Type | Pack Header              |
| :--------- | :-------- | :------------ | :----------------------- |
| `f32`      | `f64`     | `f64`         | `nk_shapes_header_f64_t` |
| `f16`      | `f32`     | `f32`         | `nk_shapes_header_f32_t` |

Rotations are `f32` for both, row-major 3 × 3, taking each column cloud onto each row cloud through the centroids in the headers.
The transport solver keeps its duals and Gromov-Wasserstein its line search in `f64` for both, since a near-tie between two plans otherwise lands the search in a different local optimum.

## Packing

A pack is a public header followed by the x, y and z arrays in the input type, each on a 64-byte boundary from the start of the pack and zero-padded to the next one.
Packing centers the cloud, rotates it onto the eigenvectors of its covariance in descending eigenvalue order, and rounds the result once into the input type.
The header keeps the centroid, the frame as a unit quaternion, the scale, the descending spreads of the packed points, and their largest norm.
Packing is one serial routine for every capability, written so no compiler can contract it, so a pack is byte-identical on every machine.

Normalized packing also divides the centered cloud by its RMS distance from the centroid, $s = \sqrt{\operatorname{tr} C}$ over its covariance $C$, before rounding, and keeps $s$ as the scale, which stays 1 for an empty cloud, a single point or coincident points.
An input point comes back as $x = s \, Q^T x' + \bar{x}$, and a rotation between normalized packs takes column cloud $b$ onto row cloud $a$ as $a_i \approx (s_a / s_b) \, R \, (b_j - \bar{b}) + \bar{a}$.
Every kind compares the packed coordinates, so it sees only shape between normalized packs, and mixing absolute and normalized packs in one call is the caller's choice.

The header's constants bound the expensive kinds from below, the way vector norms bound dot products:

- Wasserstein-Procrustes ≥ $\sqrt{\sum_k (\sqrt{\lambda_{a,k}} - \sqrt{\lambda_{b,k}})^2}$ over the descending spreads, from Gelbrich's bound between second moments
- Hausdorff ≥ $|r_a - r_b|$ over the largest radii

On 200 pools of real molecules the first skips 65% of random candidates and 16% of near neighbors before any transport, with no violation.

## Optimizations

### Principal Frames and 8 Flips

Two clouds rotated onto their own principal axes differ by at most a sign per axis, so Chamfer and Hausdorff compare them under the 8 sign patterns instead of searching rotations.
4 patterns are rotations and 4 are reflections, and all 8 are kept, so mirror images score as equals and the determinant of the returned rotation tells them apart.
Each squared distance takes the difference form $(a - b)^2$ or $(a + b)^2$ per axis, never $\|a\|^2 + \|b\|^2 - 2 a \cdot b$, which cancels for the near neighbors that set the minima.
Wasserstein-Procrustes starts from the same 8 frames, since its alternation only descends to the nearest local optimum.

### Exact Transport by Network Simplex

Uniform weights become integer masses, $m$ units per row point and $n$ per column point, so plans stay exact.
The solver starts from the northwest-corner staircase, prices arcs in blocks of $\sqrt{nm}$, picks the leaving arc by Cunningham's strongly feasible rule against cycling, and keeps parent, child and sibling links so a pivot only revisits the moved subtree.
Arc costs come from the coordinates on the fly, so Wasserstein-Procrustes needs $\Theta(n + m)$ workspace.
Each alternation round warm-starts from the previous basis, which stays primal feasible under the new rotation.

### Canonical Pair Order

The transport kinds order each pair by points, spreads, radius and bytes before solving, so swapping the two sides returns the same distance bit for bit and the transposed rotation.
Chamfer and Hausdorff are symmetric by construction.

### Gromov-Wasserstein Starts

From the product plan alone, Gromov-Wasserstein lands more than 1% worse than its best start in 70% of molecule pairs.
It starts from the best proper and the best mirrored Wasserstein-Procrustes plans as well, then takes a fixed budget of Frank-Wolfe steps with exact transport plans and closed-form line searches.

### Why SME and SVE Were Removed

Historical note: experimental SME variants of RMSD, Kabsch, and Umeyama were implemented in 1,052 lines across `sme.h` and `smef64.h` (commit `0e0bc30c`) and removed 4 days later (commit `f55e9a71`).
The fundamental mismatch: the algorithm computes a 3×3 cross-covariance matrix $H = \sum (a_i - \bar{a})(b_i - \bar{b})^T$ — a sum of outer products of 3D vectors.
SME's `FMOPA` operates on SVL-wide vectors (16+ elements at SVL=512), but the outer products here are 3×3 — the tile is 99.6% wasted (9 useful cells out of 256).
Three approaches were explored in a design document that never landed in the repository:
(1) batched outer products — reformulates as 9 independent dot products but loses SME's outer-product strength, falling back to what NEON already does;
(2) streaming SVE with `svld3` — hardware stride-3 deinterleaving processes 16 points per iteration vs NEON's 4, but `SMSTART`/`SMSTOP` mode transitions cost ~100 cycles and the 3×3 SVD step cannot use streaming mode at all;
(3) SME for SVD — the 3×3 matrix cannot fill even one 16×16 tile.
Performance estimates from the design document: NEON baseline ~2.25N cycles for N points; streaming SVE ~1.2N cycles but with ~100-cycle mode transition overhead — for typical protein alignment workloads (N = 100–500 atoms), the overhead dominates.
Experimental SVE shape kernels (`sve.h`, `svehalf.h`, 112 lines total) were removed in the same commit — variable vector length added complexity without clear benefit over fixed-width NEON for the 3D point cloud problem.
