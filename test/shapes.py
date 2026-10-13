#!/usr/bin/env python3
"""Test the many-to-many shape distances: nk.shapes_pack and the nk.*_packed kinds.

DTypes: float32, float16.
Baselines: SciPy nearest-partner, assignment and linear-program references, RDKit best RMS, POT.
Matches C++ suite: test/shapes.cpp.

File: test/shapes.py
Author: Ash Vardanian
Date: February 27, 2026
"""

from __future__ import annotations

import gc
import math
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    import numpy as np  # static-analysis-only; the runtime try/except below is authoritative

try:
    import numpy as np
except Exception:
    pass

from base import SETTINGS, assert_allclose, collect_errors, numpy_available, timed_call

import numkong as nk


try:
    from scipy.optimize import linear_sum_assignment, linprog
    from scipy.spatial.distance import cdist, directed_hausdorff
    from scipy.spatial.transform import Rotation

    scipy_available = True
except ImportError:
    scipy_available = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    from rdkit.Geometry import Point3D

    rdkit_available = True
except ImportError:
    rdkit_available = False

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # POT's docstrings hold invalid escapes
        import ot

    pot_available = True
except ImportError:
    pot_available = False

requires_numpy = pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
requires_scipy = pytest.mark.skipif(not (numpy_available and scipy_available), reason="NumPy or SciPy is not installed")

KINDS: dict[str, Callable] = {
    "chamfers": nk.chamfers_packed,
    "hausdorffs": nk.hausdorffs_packed,
    "wassersteinprocrustes": nk.wassersteinprocrustes_packed,
    "gromovwassersteins": nk.gromovwassersteins_packed,
}

FLIP_SIGNS = [tuple(-1.0 if flip >> axis & 1 else 1.0 for axis in range(3)) for flip in range(8)]
"""The axis signs of flip `s`, negating principal axis `k` when bit `k` is set."""
PROPER_FLIPS = (0, 3, 5, 6)
"""The flips that are rotations, the rest being reflections."""
INPUT_TOLERANCES = {"float32": 1e-5, "float16": 3e-2}
"""The absolute error of values that also see points rounded to the pack dtype, the inputs worst seen at 1e-6
and 1.6e-2, or the unpacked points that `packed_coordinates` poses back."""
DISTANCE_DTYPES = {"float32": np.float64, "float16": np.float32} if numpy_available else {}


def random_cloud(generator: np.random.Generator, points: int) -> np.ndarray:
    """An anisotropic Gaussian cloud, so its principal axes are well separated."""
    return generator.standard_normal((points, 3)) * np.array([3.0, 2.0, 1.0])


def random_rotation(generator: np.random.Generator) -> np.ndarray:
    """A uniformly random proper rotation."""
    orthogonal, triangular = np.linalg.qr(generator.standard_normal((3, 3)))
    orthogonal *= np.sign(np.diag(triangular))
    if np.linalg.det(orthogonal) < 0:
        orthogonal[:, 0] *= -1
    return orthogonal


def packed_coordinates(packed: nk.ShapesPackedMatrix) -> list[np.ndarray]:
    """The packed points of every cloud in float64, posed back from its unpacked points x as x′ = Q (x − c) / scale."""
    poses = zip(*(np.asarray(array, dtype=np.float64) for array in (packed.frames, packed.centroids, packed.scales)))
    clouds = []
    for index, (frame, centroid, scale) in enumerate(poses):
        points = np.asarray(packed.unpack(index), dtype=np.float64)
        clouds.append((points - centroid) @ Rotation.from_quat(frame).as_matrix().T / scale)
    return clouds


def rotations_buffer(a_packed: nk.ShapesPackedMatrix, b_packed: nk.ShapesPackedMatrix) -> np.ndarray:
    """A NaN-filled `rotations=` output for every pair of `a_packed` and `b_packed`."""
    return np.full((len(a_packed), len(b_packed), 3, 3), np.nan, dtype=np.float32)


def reference_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """½ × (meanᵢ minⱼ ‖aᵢ − S·bⱼ‖ + meanⱼ minᵢ ‖aᵢ − S·bⱼ‖), the best of the 8 flips S."""
    if len(a) == 0 or len(b) == 0:
        return 0.0 if len(a) == len(b) else math.inf
    best = math.inf
    for signs in FLIP_SIGNS:
        distances = cdist(a, b * signs)
        best = min(best, 0.5 * (distances.min(axis=1).mean() + distances.min(axis=0).mean()))
    return best


def reference_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """max(maxᵢ minⱼ ‖aᵢ − S·bⱼ‖, maxⱼ minᵢ ‖aᵢ − S·bⱼ‖), the best of the 8 flips S."""
    if len(a) == 0 or len(b) == 0:
        return 0.0 if len(a) == len(b) else math.inf
    best = math.inf
    for signs in FLIP_SIGNS:
        forward, backward = directed_hausdorff(a, b * signs)[0], directed_hausdorff(b * signs, a)[0]
        best = min(best, max(forward, backward))
    return best


def assignment_plan(costs: np.ndarray) -> np.ndarray:
    """The optimal transport plan between equal uniform weights, a scaled permutation."""
    rows, columns = linear_sum_assignment(costs)
    plan = np.zeros_like(costs)
    plan[rows, columns] = 1.0 / len(costs)
    return plan


def linprog_plan(costs: np.ndarray) -> np.ndarray:
    """The optimal transport plan between uniform weights of any sizes, a vertex of its polytope."""
    rows, columns = costs.shape
    marginals = np.vstack([np.kron(np.eye(rows), np.ones(columns)), np.kron(np.ones(rows), np.eye(columns))])
    weights = np.concatenate([np.full(rows, 1.0 / rows), np.full(columns, 1.0 / columns)])
    solution = linprog(costs.ravel(), A_eq=marginals, b_eq=weights, bounds=(0, None), method="highs-ds")
    assert solution.success, solution.message
    return solution.x.reshape(rows, columns)


def kabsch_rotation(plan: np.ndarray, a: np.ndarray, b: np.ndarray, signs: tuple[float, ...]) -> np.ndarray:
    """The rotation best superposing `b` onto `a` under `plan`, a reflection when the start `signs` is one."""
    left, _, right_transposed = np.linalg.svd(b.T @ plan.T @ a)
    right = right_transposed.T
    if np.linalg.det(right @ left.T) * math.prod(signs) < 0:
        right[:, 2] *= -1
    return right @ left.T


def alternate_from_flips(
    a: np.ndarray, b: np.ndarray, transport: Callable[[np.ndarray], np.ndarray]
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Alternates `transport` with Kabsch from each of the 8 flips while the squared cost falls,
    returning every start's best squared cost, plan and rotation."""
    starts = []
    for signs in FLIP_SIGNS:
        rotation, best, best_plan, best_rotation = np.diag(signs), math.inf, None, None
        while True:
            costs = cdist(a, b @ rotation.T, "sqeuclidean")
            plan = transport(costs)
            value = float(np.sum(plan * costs))
            if not value < best:
                break
            best, best_plan, best_rotation = value, plan, rotation
            rotation = kabsch_rotation(plan, a, b, signs)
        starts.append((best, best_plan, best_rotation))
    return starts


def reference_wassersteinprocrustes(a: np.ndarray, b: np.ndarray, transport: Callable) -> float:
    """√ of the best squared transport cost over the 8 alternations."""
    return math.sqrt(min(cost for cost, _, _ in alternate_from_flips(a, b, transport)))


def reference_gromovwasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """½ · √ of the best square loss that 32 Frank-Wolfe steps with exact linear steps reach from the
    best proper and the best mirrored Wasserstein-Procrustes plans and the product plan."""
    rows, columns = len(a), len(b)
    starts = alternate_from_flips(a, b, linprog_plan)
    proper = min((starts[flip] for flip in PROPER_FLIPS), key=lambda start: start[0])
    mirrored = min((starts[flip] for flip in range(8) if flip not in PROPER_FLIPS), key=lambda start: start[0])
    distances_a, distances_b = cdist(a, a), cdist(b, b)
    constant = np.sum(distances_a**2) / rows**2 + np.sum(distances_b**2) / columns**2
    best = math.inf
    for plan in (proper[1], mirrored[1], np.full((rows, columns), 1.0 / (rows * columns))):
        gradient = distances_a @ plan @ distances_b
        product = np.sum(gradient * plan)
        for _ in range(32):
            vertex = linprog_plan(-gradient)
            vertex_gradient = distances_a @ vertex @ distances_b
            cross = np.sum(gradient * vertex)
            linear, quadratic = cross - product, np.sum(vertex_gradient * vertex) - 2 * cross + product
            curvature, slope = -2 * quadratic, -4 * linear
            if curvature > 0:
                step = min(max(-slope / (2 * curvature), 0.0), 1.0)
            else:
                step = 1.0 if curvature + slope < 0 else 0.0
            if step == 0:
                break
            plan = (1 - step) * plan + step * vertex
            gradient = (1 - step) * gradient + step * vertex_gradient
            product = np.sum(gradient * plan)
        best = min(best, constant - 2 * product)
    return 0.5 * math.sqrt(max(best, 0.0))


def check_against_reference(stats, capabilities, kind: str, dtype: str, a_packed, b_packed, reference) -> None:
    """Scores every cloud pair with `kind` and compares each distance to `reference` on the posed-back packed points."""
    result_ns, distances = timed_call(KINDS[kind], a_packed, b_packed, capabilities=capabilities)
    distances = np.asarray(distances)
    assert distances.dtype == DISTANCE_DTYPES[dtype]
    a_clouds, b_clouds = packed_coordinates(a_packed), packed_coordinates(b_packed)
    accurate_ns, expected = timed_call(lambda: np.array([[reference(a, b) for b in b_clouds] for a in a_clouds]))
    assert_allclose(distances, expected, atol=INPUT_TOLERANCES[dtype], rtol=0, err_msg=kind)
    points = max(a_packed[index].points for index in range(len(a_packed)))
    capability = capabilities.name.lower().replace("|", "+")
    collect_errors(
        kind, points, dtype, expected, accurate_ns, None, 0, distances, result_ns, stats, capability=capability
    )


@requires_scipy
@pytest.mark.repeat(SETTINGS.reduced_repetitions)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_chamfers_and_hausdorffs(stats, capabilities, dtype: str, np_rng: np.random.Generator):
    """Chamfer and Hausdorff against brute-force `cdist` and `directed_hausdorff` over the 8 flips."""
    sizes = [SETTINGS.shape_points, SETTINGS.shape_points // 3 + 1, 17]
    a_packed = nk.shapes_pack([random_cloud(np_rng, size) for size in sizes], dtype=dtype)
    b_packed = nk.shapes_pack([random_cloud(np_rng, size) for size in sizes[1:]], dtype=dtype)
    check_against_reference(stats, capabilities, "chamfers", dtype, a_packed, b_packed, reference_chamfer)
    check_against_reference(stats, capabilities, "hausdorffs", dtype, a_packed, b_packed, reference_hausdorff)


@requires_scipy
@pytest.mark.repeat(SETTINGS.reduced_repetitions)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_wassersteinprocrustes_equal_sizes(stats, capabilities, dtype: str, np_rng: np.random.Generator):
    """Wasserstein-Procrustes against assignments alternated with Kabsch from the same 8 flip starts."""
    a_packed = nk.shapes_pack([random_cloud(np_rng, 16) for _ in range(2)], dtype=dtype)
    b_packed = nk.shapes_pack([random_cloud(np_rng, 16) for _ in range(3)], dtype=dtype)
    check_against_reference(
        stats,
        capabilities,
        "wassersteinprocrustes",
        dtype,
        a_packed,
        b_packed,
        lambda a, b: reference_wassersteinprocrustes(a, b, assignment_plan),
    )


@requires_scipy
@pytest.mark.repeat(SETTINGS.reduced_repetitions)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_wassersteinprocrustes_unequal_sizes(stats, capabilities, dtype: str, np_rng: np.random.Generator):
    """Wasserstein-Procrustes against `linprog` transport alternated with Kabsch from the 8 flip starts."""
    a_packed = nk.shapes_pack([random_cloud(np_rng, points) for points in (7, 12)], dtype=dtype)
    b_packed = nk.shapes_pack([random_cloud(np_rng, points) for points in (11, 5)], dtype=dtype)
    check_against_reference(
        stats,
        capabilities,
        "wassersteinprocrustes",
        dtype,
        a_packed,
        b_packed,
        lambda a, b: reference_wassersteinprocrustes(a, b, linprog_plan),
    )


@requires_scipy
@pytest.mark.repeat(SETTINGS.reduced_repetitions)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_gromovwassersteins(stats, capabilities, dtype: str, np_rng: np.random.Generator):
    """Gromov-Wasserstein against Frank-Wolfe with `linprog` linear steps from the same three starts."""
    a_packed = nk.shapes_pack([random_cloud(np_rng, points) for points in (6, 9)], dtype=dtype)
    b_packed = nk.shapes_pack([random_cloud(np_rng, points) for points in (8, 6)], dtype=dtype)
    check_against_reference(
        stats, capabilities, "gromovwassersteins", dtype, a_packed, b_packed, reference_gromovwasserstein
    )


@requires_scipy
def test_wassersteinprocrustes_equals_kabsch(capabilities, np_rng: np.random.Generator):
    """When the known correspondence is the optimal plan, Wasserstein-Procrustes is the Kabsch RMSD and
    returns its rotation."""
    a = random_cloud(np_rng, 32)
    # Noise far below the spacing between points keeps the correspondence the optimal plan
    b = (a + np_rng.normal(scale=0.01, size=a.shape)) @ random_rotation(np_rng) + np_rng.normal(size=3)
    a_packed, b_packed = nk.shapes_pack(a, dtype="float32"), nk.shapes_pack(b, dtype="float32")
    rotations = rotations_buffer(a_packed, b_packed)
    distances = nk.wassersteinprocrustes_packed(a_packed, b_packed, rotations=rotations, capabilities=capabilities)
    rotation, root_sum_squares = Rotation.align_vectors(a - a.mean(axis=0), b - b.mean(axis=0))
    assert_allclose(float(np.asarray(distances)[0, 0]), root_sum_squares / math.sqrt(len(a)), atol=0, rtol=2e-5)
    assert_allclose(rotations[0, 0], rotation.as_matrix(), atol=1e-6, rtol=0)


@requires_numpy
@pytest.mark.skipif(not rdkit_available, reason="RDKit is not installed")
def test_wassersteinprocrustes_below_rdkit_best_rms(capabilities, np_rng: np.random.Generator):
    """No atom mapping RDKit's symmetry-aware best RMS tries beats the optimal transport plan."""
    molecule = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=42) == 0
    reference = molecule.GetConformer().GetPositions()
    perturbed = reference @ random_rotation(np_rng) + np_rng.normal(size=3)
    perturbed += np_rng.normal(scale=0.05, size=perturbed.shape)
    probe = Chem.Mol(molecule)
    for atom, point in enumerate(perturbed):
        probe.GetConformer().SetAtomPosition(atom, Point3D(*point))
    best_rms = rdMolAlign.GetBestRMS(probe, molecule)
    distances = nk.wassersteinprocrustes_packed(
        nk.shapes_pack(reference, dtype="float32"),
        nk.shapes_pack(perturbed, dtype="float32"),
        capabilities=capabilities,
    )
    assert float(np.asarray(distances)[0, 0]) <= best_rms + 1e-6


@requires_scipy
@pytest.mark.skipif(not pot_available, reason="POT is not installed")
def test_gromovwassersteins_against_pot(capabilities, np_rng: np.random.Generator):
    """On shuffled noisy copies, the Wasserstein-Procrustes starts end no worse than POT's product start."""
    a = random_cloud(np_rng, 12)
    b = (a + np_rng.normal(scale=0.02, size=a.shape))[np_rng.permutation(12)] @ random_rotation(np_rng)
    distances = nk.gromovwassersteins_packed(
        nk.shapes_pack(a, dtype="float32"), nk.shapes_pack(b, dtype="float32"), capabilities=capabilities
    )
    weights = np.full(12, 1.0 / 12)
    loss = ot.gromov.gromov_wasserstein2(cdist(a, a), cdist(b, b), weights, weights, loss_fun="square_loss")
    assert float(np.asarray(distances)[0, 0]) <= 0.5 * math.sqrt(max(loss, 0.0)) + 1e-6


@requires_numpy
@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_swapped_sides_transpose(capabilities, kind: str, dtype: str, np_rng: np.random.Generator):
    """Swapping the sides transposes the distances bit for bit and the rotations with them."""
    a_packed = nk.shapes_pack([random_cloud(np_rng, points) for points in (5, 9, 0)], dtype=dtype)
    b_packed = nk.shapes_pack([random_cloud(np_rng, points) for points in (8, 0, 6, 9)], dtype=dtype)
    forward_rotations, backward_rotations = rotations_buffer(a_packed, b_packed), rotations_buffer(b_packed, a_packed)
    forward = KINDS[kind](a_packed, b_packed, rotations=forward_rotations, capabilities=capabilities)
    backward = KINDS[kind](b_packed, a_packed, rotations=backward_rotations, capabilities=capabilities)
    np.testing.assert_array_equal(np.asarray(forward), np.asarray(backward).T)
    np.testing.assert_array_equal(forward_rotations, backward_rotations.transpose(1, 0, 3, 2))


@requires_numpy
@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_rigid_motions_preserve_distances(capabilities, kind: str, dtype: str, np_rng: np.random.Generator):
    """Rotating, mirroring and moving either cloud before packing leaves every distance unchanged."""
    clouds = [random_cloud(np_rng, points) for points in (10, 7)]
    mirror = np.diag([1.0, -1.0, 1.0])
    moved = [cloud @ random_rotation(np_rng) @ mirror + np_rng.normal(scale=5.0, size=3) for cloud in clouds]
    original = KINDS[kind](nk.shapes_pack(clouds, dtype=dtype), nk.shapes_pack(clouds[::-1], dtype=dtype))
    rigid = KINDS[kind](nk.shapes_pack(moved, dtype=dtype), nk.shapes_pack(moved[::-1], dtype=dtype))
    assert_allclose(np.asarray(rigid), np.asarray(original), atol=INPUT_TOLERANCES[dtype], rtol=0)


@requires_numpy
@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_rotations_superpose_rigid_copies(capabilities, kind: str, dtype: str, np_rng: np.random.Generator):
    """Between a cloud and its shuffled rigid motion, every kind vanishes and its rotation maps b onto a."""
    a = random_cloud(np_rng, 11)
    order = np_rng.permutation(11)
    b = a[order] @ random_rotation(np_rng) + np_rng.normal(scale=3.0, size=3)
    a_packed, b_packed = nk.shapes_pack(a, dtype=dtype), nk.shapes_pack(b, dtype=dtype)
    rotations = rotations_buffer(a_packed, b_packed)
    distances = KINDS[kind](a_packed, b_packed, rotations=rotations, capabilities=capabilities)
    assert_allclose(float(np.asarray(distances)[0, 0]), 0.0, atol=INPUT_TOLERANCES[dtype], rtol=0)
    a_centroid, b_centroid = np.asarray(a_packed.centroids)[0], np.asarray(b_packed.centroids)[0]
    superposed = (b - b_centroid) @ rotations[0, 0].T.astype(np.float64) + a_centroid
    assert_allclose(superposed, a[order], atol=INPUT_TOLERANCES[dtype], rtol=0)


@requires_numpy
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_normalized_packs_ignore_size(capabilities, dtype: str, np_rng: np.random.Generator):
    """Normalized packs keep each RMS radius as the scale, and every kind scores a cloud about zero against
    its turned, moved and 3.7× grown copy between them, but not between plain packs."""
    a = random_cloud(np_rng, 11)
    b = 3.7 * a @ random_rotation(np_rng) + np_rng.normal(scale=3.0, size=3)
    tolerance = INPUT_TOLERANCES[dtype]
    scales = np.asarray(nk.shapes_pack_normalized([a, b], dtype=dtype).scales)
    radii = [math.sqrt(np.mean(np.sum((cloud - cloud.mean(axis=0)) ** 2, axis=1))) for cloud in (a, b)]
    assert scales.dtype == DISTANCE_DTYPES[dtype]
    assert_allclose(scales, radii, atol=0, rtol=tolerance)
    normalized = nk.shapes_pack_normalized(a, dtype=dtype), nk.shapes_pack_normalized(b, dtype=dtype)
    plain = nk.shapes_pack(a, dtype=dtype), nk.shapes_pack(b, dtype=dtype)
    for kind, kernel in KINDS.items():
        assert float(np.asarray(kernel(*normalized, capabilities=capabilities))[0, 0]) <= tolerance, kind
        assert float(np.asarray(kernel(*plain, capabilities=capabilities))[0, 0]) > tolerance * scales[1], kind


@requires_numpy
@pytest.mark.parametrize("kind", list(KINDS))
def test_empty_clouds(capabilities, kind: str, np_rng: np.random.Generator):
    """Two empty clouds are 0 apart, an empty and a non-empty one infinitely, with identity rotations."""
    packed = nk.shapes_pack([np.zeros((0, 3)), random_cloud(np_rng, 4)])
    rotations = rotations_buffer(packed, packed)
    distances = np.asarray(KINDS[kind](packed, packed, rotations=rotations, capabilities=capabilities))
    assert distances[0, 0] == 0.0
    assert math.isinf(distances[0, 1]) and math.isinf(distances[1, 0])
    for row, column in ((0, 0), (0, 1), (1, 0)):
        np.testing.assert_array_equal(rotations[row, column], np.eye(3, dtype=np.float32))


@requires_scipy
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_packed_poses(dtype: str, np_rng: np.random.Generator):
    """The centroids are the input means, the frames unit quaternions, and the spreads and radii the moments
    and norms of the packed points, the spreads also within a relative tolerance as squared lengths."""
    clouds = [random_cloud(np_rng, points) + np_rng.normal(scale=4.0, size=3) for points in (13, 1, 40)]
    packed = nk.shapes_pack(clouds, dtype=dtype)
    assert [packed[index].points for index in range(3)] == [13, 1, 40]
    field_dtype = DISTANCE_DTYPES[dtype]
    centroids, frames, spreads, max_radii = (
        np.asarray(array) for array in (packed.centroids, packed.frames, packed.spreads, packed.max_radii)
    )
    assert (centroids.shape, frames.shape, spreads.shape, max_radii.shape) == ((3, 3), (3, 4), (3, 3), (3,))
    assert centroids.dtype == frames.dtype == spreads.dtype == max_radii.dtype == field_dtype
    tolerance = INPUT_TOLERANCES[dtype]
    for cloud, coordinates, centroid, frame, spread, max_radius in zip(
        clouds, packed_coordinates(packed), centroids, frames, spreads, max_radii
    ):
        assert_allclose(centroid, cloud.mean(axis=0), atol=tolerance, rtol=0)
        assert_allclose(np.linalg.norm(frame), 1.0, atol=1e-12 if dtype == "float32" else 1e-6, rtol=0)
        moments = np.sort(np.linalg.eigvalsh(coordinates.T @ coordinates / len(coordinates)))[::-1]
        assert_allclose(spread, moments, atol=tolerance, rtol=tolerance)
        assert np.all(np.diff(spread) <= 0)
        assert_allclose(max_radius, np.linalg.norm(coordinates, axis=1).max(), atol=tolerance, rtol=0)


@requires_numpy
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_unpack_round_trips(dtype: str, np_rng: np.random.Generator):
    """Unpacking returns every input point in the pack dtype within storage rounding, and normalized packs
    return them at their original scale."""
    clouds = [random_cloud(np_rng, points) + np_rng.normal(scale=4.0, size=3) for points in (13, 1, 0)]
    for pack in (nk.shapes_pack, nk.shapes_pack_normalized):
        packed = pack(clouds, dtype=dtype)
        for index, cloud in enumerate(clouds[:2]):
            unpacked = np.asarray(packed.unpack(index))
            assert unpacked.dtype == np.dtype(dtype)
            assert_allclose(unpacked, cloud, atol=INPUT_TOLERANCES[dtype], rtol=0)
        assert packed.unpack(-1).shape == (0, 3)
        with pytest.raises(IndexError):
            packed.unpack(3)


@requires_numpy
def test_pack_dtypes(np_rng: np.random.Generator):
    """float16 input packs as float16, anything else as float32, unless `dtype` says otherwise."""
    cloud = random_cloud(np_rng, 6)
    assert nk.shapes_pack(cloud).dtype == "float32"
    assert nk.shapes_pack(cloud.astype(np.float16)).dtype == "float16"
    assert nk.shapes_pack(cloud.astype(np.float32), dtype="float16").dtype == "float16"
    assert nk.shapes_pack([cloud.astype(np.float16), cloud], dtype="float32").dtype == "float32"
    strided = np.repeat(cloud, 2, axis=0)[::2]
    strided_packed, contiguous_packed = nk.shapes_pack(strided), nk.shapes_pack(cloud)
    np.testing.assert_array_equal(np.asarray(strided_packed.unpack(0)), np.asarray(contiguous_packed.unpack(0)))
    for dtype, distance_dtype in DISTANCE_DTYPES.items():
        packed = nk.shapes_pack(cloud, dtype=dtype)
        distances = np.asarray(nk.chamfers_packed(packed, packed))
        assert distances.dtype == distance_dtype and distances.shape == (1, 1)


@requires_numpy
def test_packed_views(np_rng: np.random.Generator):
    """Indexing and slicing make views over the same packs that outlive the packed set."""
    clouds = [random_cloud(np_rng, points) for points in (3, 5, 8, 13, 21)]
    packed = nk.shapes_pack(clouds)
    assert len(packed) == packed.clouds == 5
    assert packed.shape == (5, packed.points) == (5, 50)
    assert packed.nbytes == sum(nk.ShapesPackedMatrix.pack_size(len(cloud), dtype="float32") for cloud in clouds)
    assert packed.nbytes == sum(packed[index].nbytes for index in range(5))
    every_other = packed[::2]
    assert every_other.shape == (3, 3 + 8 + 21)
    np.testing.assert_array_equal(np.asarray(packed[-1].unpack(0)), np.asarray(every_other.unpack(-1)))
    expected = np.asarray(nk.chamfers_packed(packed, packed))[::2, 2::2]
    del packed
    gc.collect()
    np.testing.assert_array_equal(np.asarray(nk.chamfers_packed(every_other, every_other[1:])), expected)
    with pytest.raises(IndexError):
        every_other[3]
    with pytest.raises(IndexError):
        every_other[-4]


@requires_numpy
def test_shapes_errors(np_rng: np.random.Generator):
    """Malformed clouds, dtypes, operands and rotation buffers raise before any kernel runs."""
    cloud = random_cloud(np_rng, 4)
    with pytest.raises(ValueError):
        nk.shapes_pack(cloud[:, :2])
    with pytest.raises(ValueError):
        nk.shapes_pack(cloud, dtype="float64")
    with pytest.raises(TypeError):
        nk.shapes_pack([cloud, "points"])
    half, single = nk.shapes_pack(cloud, dtype="float16"), nk.shapes_pack(cloud, dtype="float32")
    with pytest.raises(TypeError):
        nk.chamfers_packed(half, single)
    with pytest.raises(TypeError):
        nk.hausdorffs_packed(single, cloud)
    with pytest.raises(TypeError):
        nk.wassersteinprocrustes_packed(single, single, True)
    with pytest.raises(ValueError):
        nk.chamfers_packed(single, single, rotations=np.zeros((1, 1, 3, 3)))
    with pytest.raises(ValueError):
        nk.chamfers_packed(single, single, rotations=np.zeros((1, 2, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        nk.chamfers_packed(single, single, rotations=np.zeros((1, 1, 3, 6), dtype=np.float32)[..., ::2])
    read_only = np.zeros((1, 1, 3, 3), dtype=np.float32)
    read_only.flags.writeable = False
    with pytest.raises(BufferError):
        nk.chamfers_packed(single, single, rotations=read_only)
