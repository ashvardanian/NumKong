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
| `f64`      | `f64`     | `f64`         | `f64`          |
| `f32`      | `f64`     | `f64`         | `f64`          |
| `f16`      | `f32`     | `f32`         | `f32`          |

Rotations are `f32` for every input type, row-major 3 × 3, taking each column cloud onto each row cloud through the centroids of the packs.
`f64` inputs sum the pack's moments, Chamfer's roots, each plan's cost and cross-covariance, and every Gromov-Wasserstein reduction with TwoProd and TwoSum, as the one-to-one `f64` kernels do.
The transport solver keeps its duals and Gromov-Wasserstein its line search in `f64` for every input type, since a near-tie between two plans otherwise lands the search in a different local optimum.

## Superposed RMSD

RMSD is the one kind that needs a correspondence, as between the frames of one molecular dynamics trajectory, where every frame holds the same atoms in the same order.
Clustering such a trajectory takes the RMSD of every pair of frames after the best rotation, which with $G = \sum_i \|x_i\|^2$ over a centered frame and $M = \sum_i a_i b_i^T$ is:

$$
\text{RMSD}(A, B) = \sqrt{\frac{1}{n} \min_{R \in SO(3)} \sum_i \|a_i - R b_i\|^2} = \sqrt{\frac{G_a + G_b - 2 \lambda}{n}}
$$

Here $\lambda$ is the largest eigenvalue of Theobald's 4 × 4 quaternion key matrix built from $M$, or $\sigma_1 + \sigma_2 \pm \sigma_3$ over the singular values of $M$.
`nk_rmsds_packed_*` reads the packs of `nk_shapes_pack_*`, which keep the point order, as no proper rotation of either frame onto its principal axes changes the RMSD, and refuses packs of different sizes.
Between normalized packs it is the scale-free RMSD.
Each kernel sums every frame's $G$ from its packed points, once per tile of 64 frames on the CPU and 32 on Blackwell, and the norms bound the RMSD from below: RMSD ≥ $|\sqrt{G_a / n} - \sqrt{G_b / n}|$.

| Input Type | Math Type | Distance Type | Packed Points | Sum of Squares                             |
| :--------- | :-------- | :------------ | :------------ | :----------------------------------------- |
| `f64`      | `f64`     | `f64`         | `f64`         | TwoProd $M$, Kabsch, compensated residuals |
| `f32`      | `f64`     | `f64`         | `f32`         | Kabsch rotation, compensated residual pass |
| `f16`      | `f32`     | `f32`         | `f16`         | $G_a + G_b - 2\lambda$, clamped at zero    |

These are the rules of the one-to-one Kabsch kernels the shapes family replaced.
Folding into $G_a + G_b - 2 \lambda$ cancels to $\sqrt{\varepsilon}$ of the math type once two frames align, which stays below the rounding of `f16` points but not of `f32` or `f64` ones, so those sum $\|a_i - R b_i\|^2$ directly with TwoProd and TwoSum.
`nk_shapes_pack_f64` writes `f64` packs with the layout of `f32` ones and `f64` points.
Packing rounds each rotated point once, which moves an RMSD by at most $\sqrt{3} \, u \, (r_a + r_b)$ for the unit roundoff $u$ of the input type and the largest radii $r$ of the packs.

## Packing

A pack is opaque and, like the transport workspace and every NumKong buffer, starts on an `nk_default_alignment_k` boundary.
Packing centers the cloud, rotates it onto the eigenvectors of its covariance in descending eigenvalue order, and rounds the result once into the input type.
The pack keeps the centroid, the frame as a unit quaternion, the scale, the descending spreads of the packed points, and their largest norm, which `nk_shapes_packed_shape_*` reads.
Packing is one serial routine for every capability today, written so no compiler contracts it under the project's ISO C build flags, so a pack is byte-identical on every machine and every capability reads it.

Eigenvalues within $\tau$ of each other, relative to the largest, leave their eigenvectors to rounding, so packing settles them by higher moments, with $\tau = 2^{-16}$ for `f32` and `f64` inputs and $2^{-7}$ for `f16`, whose own rounding moves the moments by more.
A tied plane around the third axis $c$ turns until the first harmonic of $w = x \cdot a + i \, x \cdot b$ to clear $\tau$ lies along $a$, and three tied axes first point $c$ at a point.
Clouds without ties keep the frame bit for bit, and a copy of a tied cloud packs into the same frame up to the 8 flips, but for about 1 random tied `f16` cloud in 100, whose weakest harmonic lands within rounding of $\tau$.

```
if every eigenvalue ties: c = the point direction u maximizing Σ(x·u)³, or Σ(x·u)⁴ if no cubic clears τ Σ|x|³
                          a, b = any proper completion of c, at a cost of n² for the n points
else:                     a, b = the tied eigenvectors, c = the third one
h = x·c, oriented = |Σh³| > τ Σ|x|³, and if Σh³ < 0 then c, b = −c, −b
for m in 1 … 8, the sum Σwᵐ, then its twin Σh·wᵐ while oriented or m is odd:
    if |sum| > τ Σ|w|ᵐ, or τ Σ|x|ᵐ⁺¹ for a twin:
        turn a, b by an mᵗʰ root of the sum's phase, the root picked by the next sum of order not a multiple of m
        stop
```

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

### Quaternion Characteristic Polynomial

`f16` RMSD finds $\lambda$ by Newton's method on the key matrix's quartic from $(G_a + G_b) / 2$, an upper bound it falls from monotonically, with $M$ divided by that bound so the quartic terms stay in `f32` range.
Near a repeated largest root, as thin, planar-degenerate or linear frames have, Newton stops $\sqrt{\varepsilon}$ short, so two Rayleigh quotients of adjugate columns of $K - \lambda I$ refine it, each squaring the error of its eigenvector.
Exactly linear frames, like 2 points, still lose accuracy, as their two largest roots coincide.

### Blackwell Tensor-Core Products

On `blackwell`, `f16` RMSD is one product $C = A \times B^T$ over the atoms: 32 row frames stack their x, y and z rows as tensor-memory lanes 0-31, 32-63 and 64-95, and 32 column frames their axes as columns, so every pair's $M$ is a 3 × 3 block of $C$, drained by three warps and solved per pair in `f32`.
Chamfer and Hausdorff fold the norms into the depth, $[a, \|a\|^2_{hi}, \|a\|^2_{lo}, 1, 1] \cdot [-2 S b, 1, 1, \|b\|^2_{hi}, \|b\|^2_{lo}]$, and the 8 flips $S$ into the columns, so one product compares 128 points with 32 points under every flip.
That expansion lies within $\delta = 2^{-19} (\|a\|^2 + r_b^2)$ of the exact square, so it only locates the nearest point, as below.
Each thread keeps, per flip, the least expanded value, the group of 8 inner points holding it, and the least value of every other group, one integer minimum per product element.
The nearest point lies in that group unless another group's least comes within $2 \delta$, in which case the thread rescans every inner point, so the distances match the serial ones bit for bit before the square root.

```
for each flip:
    if others ≤ least + 2δ: exact = min over all inner points of ‖a − S b‖²
    else:                   exact = min over the 8 points of the least group
```

### Exact Transport by Network Simplex

Uniform weights become integer masses, $m$ units per row point and $n$ per column point, so plans stay exact.
The solver starts from the northwest-corner staircase, prices arcs in blocks of $\sqrt{nm}$, picks the leaving arc by Cunningham's strongly feasible rule against cycling, and keeps parent, child and sibling links so a pivot only revisits the moved subtree.
Arc costs come from the coordinates on the fly, so Wasserstein-Procrustes needs $\Theta(n + m)$ workspace.
Each alternation round warm-starts from the previous basis, which stays primal feasible under the new rotation.

### Transport on CUDA

On `cuda`, every block solves one flip of one pair with the serial math types, and the last of a pair's 8 blocks picks the best flip in the serial order.
Clouds of equal size make an assignment problem, which successive shortest paths solve: Dijkstra over the reduced costs closes one column per step, each thread scanning its columns, and the potentials stay in `f64` like the serial duals.
Each later round warm-starts from potentials that leave the last plan tight under the new rotation, and returns only the rows a cheaper column now undercuts.
Clouds of unequal size make staircases of small flows, where one simplex pivot ships what takes shortest paths dozens of steps, so they run the serial network simplex, the block pricing spread over the team and the pivot on one thread.
Gromov-Wasserstein runs one block per start of each pair after the flips, with the same solvers on the costs $-G$ and the vertex gradient as a sparse product by $D_y$, then a dense one by $D_x$.
The workspace holds pair slots, each a counter and the records of its flips and starts, plus the dense matrices of Gromov-Wasserstein, and a call walks its pairs in launches of as many pairs as it holds slots.
Pack tables, packs, outputs and workspace are device memory, and each call reads the largest cloud back once to size shared memory, up to 863 points per side for `f32` and 953 for `f16` on 227 KiB.

### Canonical Pair Order

The transport kinds order each pair by points, spreads, radius and bytes before solving, so swapping the two sides returns the same distance bit for bit and the transposed rotation.
Chamfer and Hausdorff are symmetric by construction.

### Gromov-Wasserstein Starts

From the product plan alone, Gromov-Wasserstein lands more than 1% worse than its best start in 70% of molecule pairs.
It starts from the best proper and the best mirrored Wasserstein-Procrustes plans as well, then takes a fixed budget of Frank-Wolfe steps with exact transport plans and closed-form line searches.
Its workspace holds the dense plan, its gradient and a vertex's gradient, each $n \times m$, and the distances within each cloud, $n^2 + m^2$, so it needs $\Theta(n^2 + m^2 + nm)$.
