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
Wasserstein-Procrustes pairs points by an exact optimal transport plan $P$ between uniform weights and superposes them by a rotation or reflection $R$, alternating the two while this cost falls:

$$
\text{cost}(P, R) = \sqrt{\sum_{ij} P_{ij} \|a_i - R b_j\|^2}
$$

The alternation descends from each of the 8 flips to a local optimum, and the kernel reports the best of the 8, not a proven minimum over all $P$ and $R \in O(3)$.
On equal sizes that minimum is at most the Kabsch RMSD of a known correspondence, and equals it when that correspondence is an optimal plan.
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

| Input Type | Math Type | Distance Type | Pack Constants |
| :--------- | :-------- | :------------ | :------------- |
| `f32`      | `f64`     | `f64`         | `f64`          |
| `f16`      | `f32`     | `f32`         | `f32`          |

Rotations are `f32` for both, row-major 3 × 3, taking each column cloud onto each row cloud through the centroids of the packs.
The transport solver keeps its duals and Gromov-Wasserstein its line search in `f64` for both, since a near-tie between two plans otherwise lands the search in a different local optimum.

## Packing

A pack is opaque and, like the transport workspace and every NumKong buffer, starts on an `nk_default_alignment_k` boundary.
Packing centers the cloud, rotates it onto the eigenvectors of its covariance in descending eigenvalue order, and rounds the result once into the input type.
The pack keeps the centroid, the frame as a unit quaternion, the scale, the descending spreads of the packed points, and their largest norm, which `nk_shapes_packed_shape_*` reads.
Packing is one serial routine for every capability today, written so no compiler contracts it under the project's ISO C build flags, so a pack is byte-identical on every machine and every capability reads it.

Normalized packing also divides the centered cloud by its RMS distance from the centroid, $s = \sqrt{\operatorname{tr} C}$ over its covariance $C$, before rounding, and keeps $s$ as the scale, which stays 1 for an empty cloud, a single point or coincident points.
`nk_shapes_unpack_*` restores every input point as $x = s \, Q^T x' + \bar{x}$, rounded once into the input type, so an absolute pack returns its inputs within one storage rounding and a normalized pack its cloud at the original scale.
A rotation between normalized packs takes column cloud $b$ onto row cloud $a$ as $a_i \approx (s_a / s_b) \, R \, (b_j - \bar{b}) + \bar{a}$.
Every kind compares the packed coordinates, so it sees only shape between normalized packs, and mixing absolute and normalized packs in one call is the caller's choice.

The spreads and largest radius that `nk_shapes_packed_shape_*` reads bound the expensive kinds from below, the way vector norms bound dot products:

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
Its workspace holds the dense plan, its gradient and a vertex's gradient, each $n \times m$, and the distances within each cloud, $n^2 + m^2$, so it needs $\Theta(n^2 + m^2 + nm)$.
