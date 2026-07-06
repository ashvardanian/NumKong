#!/usr/bin/env python3
"""Test mesh alignment: nk.kabsch, nk.umeyama, nk.rmsd.

Dtypes: float64, float32, float16, bfloat16.
Baselines: high-precision Decimal Jacobi SVD, SciPy procrustes, NumPy SVD.
Matches C++ suite: test_mesh.cpp.
"""

import atexit
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    import numpy as np  # static-analysis-only; the runtime try/except below is authoritative

try:
    import numpy as np

    numpy_available = True
except Exception:
    numpy_available = False

from test_base import (
    assert_allclose,
    collect_errors,
    create_stats,
    dense_dimensions,
    downcast_f32_to_dtype,
    keep_one_capability,
    make_nk,
    mesh_points,
    nk_seed,  # noqa: F401 — pytest fixture
    numpy_available,
    possible_capabilities,
    precise_decimal,
    print_stats_report,
    profile,
    reduced_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    tolerances_for_dtype,
)

import numkong as nk


stats = create_stats()
atexit.register(print_stats_report, stats)

try:
    from scipy.spatial import procrustes as scipy_procrustes

    def baseline_kabsch(source, target, dtype=None):
        """Reference Kabsch via SciPy procrustes: returns disparity (should be ~0 for identical).
        Returns None if SVD fails to converge (e.g., near-degenerate random point clouds).
        """
        try:
            _, _, disparity = scipy_procrustes(source.astype(np.float64), target.astype(np.float64))
            return disparity
        except np.linalg.LinAlgError:
            return None

except ImportError:
    baseline_kabsch = None


def baseline_rmsd(source, target, dtype=None):
    """NumPy reference RMSD: raw √(Σ‖aᵢ − bᵢ‖² / n), no centering."""
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    n_points = source.shape[0]
    return np.sqrt(np.sum((source - target) ** 2) / n_points)


def baseline_umeyama(source, target, dtype=None):
    """NumPy reference Umeyama: SVD-based similarity transform returning scale, rotation, RMSD."""
    source = source.astype(np.float64)
    target = target.astype(np.float64)
    n_points = source.shape[0]
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    cross_covariance = source_centered.T @ target_centered / n_points
    left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(cross_covariance)
    determinant = np.linalg.det(right_vectors_transposed.T @ left_vectors.T)
    sign_vector = np.ones(3)
    sign_vector[-1] = np.sign(determinant)
    rotation = right_vectors_transposed.T @ np.diag(sign_vector) @ left_vectors.T
    source_variance = np.sum(source_centered**2) / n_points
    scale = np.sum(singular_values * sign_vector) / source_variance
    transformed = scale * (source_centered @ rotation.T)
    rmsd_valueue = np.sqrt(np.sum((transformed - target_centered) ** 2) / n_points)
    return {
        "rotation": rotation,
        "scale": scale,
        "rmsd": rmsd_valueue,
        "a_centroid": source_centroid,
        "b_centroid": target_centroid,
    }


def _jacobi_svd_3x3(upcast, sqrt):
    """Return a function that computes 3x3 SVD via Jacobi eigendecomposition.

    Uses *upcast* and *sqrt* from ``precise_decimal()`` so the same code works
    for both Decimal and float backends.
    """

    def svd(matrix):
        symmetric = [[sum(matrix[k][i] * matrix[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        right_vectors = [[upcast(1) if i == j else upcast(0) for j in range(3)] for i in range(3)]
        epsilon = upcast(1e-80)

        for _ in range(30):
            for p in range(3):
                for q in range(p + 1, 3):
                    if abs(symmetric[p][q]) < epsilon:
                        continue
                    difference = symmetric[q][q] - symmetric[p][p]
                    if abs(difference) < epsilon:
                        tangent = upcast(1)
                    else:
                        tau = difference / (2 * symmetric[p][q])
                        sign = upcast(1) if tau >= 0 else upcast(-1)
                        tangent = sign / (abs(tau) + sqrt(upcast(1) + tau * tau))
                    cosine = upcast(1) / sqrt(upcast(1) + tangent * tangent)
                    sine = tangent * cosine
                    rotated = [row[:] for row in symmetric]
                    rotated[p][p] = (
                        cosine**2 * symmetric[p][p] - 2 * sine * cosine * symmetric[p][q] + sine**2 * symmetric[q][q]
                    )
                    rotated[q][q] = (
                        sine**2 * symmetric[p][p] + 2 * sine * cosine * symmetric[p][q] + cosine**2 * symmetric[q][q]
                    )
                    rotated[p][q] = rotated[q][p] = upcast(0)
                    for r in range(3):
                        if r != p and r != q:
                            rotated[p][r] = rotated[r][p] = cosine * symmetric[p][r] - sine * symmetric[q][r]
                            rotated[q][r] = rotated[r][q] = sine * symmetric[p][r] + cosine * symmetric[q][r]
                    symmetric = rotated
                    for r in range(3):
                        old_p, old_q = right_vectors[r][p], right_vectors[r][q]
                        right_vectors[r][p] = cosine * old_p - sine * old_q
                        right_vectors[r][q] = sine * old_p + cosine * old_q

        singular_values = [sqrt(symmetric[i][i]) if symmetric[i][i] > 0 else upcast(0) for i in range(3)]
        order = sorted(range(3), key=lambda i: -singular_values[i])
        singular_values = [singular_values[i] for i in order]
        right_vectors = [[right_vectors[r][order[c]] for c in range(3)] for r in range(3)]

        left_vectors = [[upcast(0)] * 3 for _ in range(3)]
        for i in range(3):
            if singular_values[i] > epsilon:
                column = [sum(matrix[r][k] * right_vectors[k][i] for k in range(3)) for r in range(3)]
                for r in range(3):
                    left_vectors[r][i] = column[r] / singular_values[i]

        return left_vectors, singular_values, right_vectors

    return svd


def _center_and_covariance(source, target, upcast):
    """Convert points via *upcast*, center, and compute cross-covariance. Returns
    (source_centered, target_centered, cross_covariance, n_points).
    """
    n_points = len(source)
    source_precise = [[upcast(value) for value in row] for row in source]
    target_precise = [[upcast(value) for value in row] for row in target]
    source_centroid = [sum(source_precise[i][j] for i in range(n_points)) / n_points for j in range(3)]
    target_centroid = [sum(target_precise[i][j] for i in range(n_points)) / n_points for j in range(3)]
    source_centered = [[source_precise[i][j] - source_centroid[j] for j in range(3)] for i in range(n_points)]
    target_centered = [[target_precise[i][j] - target_centroid[j] for j in range(3)] for i in range(n_points)]
    cross_covariance = [
        [sum(source_centered[k][i] * target_centered[k][j] for k in range(n_points)) for j in range(3)]
        for i in range(3)
    ]
    return source_centered, target_centered, cross_covariance, n_points


def _determinant_3x3(matrix):
    """Determinant of a 3x3 list-of-lists matrix."""
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def precise_rmsd(source, target, dtype=None):
    """High-precision raw RMSD: √(Σ‖aᵢ − bᵢ‖² / n). No centering, no SVD."""
    with precise_decimal(dtype) as (upcast, sqrt, _ln):
        n_points = len(source)
        total = upcast(0)
        for i in range(n_points):
            for j in range(3):
                difference = upcast(source[i][j]) - upcast(target[i][j])
                total += difference**2
        return float(sqrt(total / n_points))


def precise_kabsch(source, target, dtype=None):
    """High-precision Kabsch via Jacobi SVD. Returns RMSD after optimal rotation."""
    with precise_decimal(dtype) as (upcast, sqrt, _ln):
        source_centered, target_centered, cross_covariance, n_points = _center_and_covariance(source, target, upcast)
        jacobi_svd = _jacobi_svd_3x3(upcast, sqrt)
        left_vectors, _, right_vectors = jacobi_svd(cross_covariance)
        # R = V Uᵀ, fix reflection if det(R) < 0
        rotation = [
            [sum(right_vectors[r][k] * left_vectors[c][k] for k in range(3)) for c in range(3)] for r in range(3)
        ]
        if _determinant_3x3(rotation) < 0:
            for r in range(3):
                right_vectors[r][2] = -right_vectors[r][2]
            rotation = [
                [sum(right_vectors[r][k] * left_vectors[c][k] for k in range(3)) for c in range(3)] for r in range(3)
            ]
        # RMSD after rotation
        total = upcast(0)
        for i in range(n_points):
            rotated = [sum(rotation[r][k] * source_centered[i][k] for k in range(3)) for r in range(3)]
            for j in range(3):
                total += (rotated[j] - target_centered[i][j]) ** 2
        return float(sqrt(total / n_points))


def precise_umeyama(source, target, dtype=None):
    """High-precision Umeyama via Jacobi SVD. Returns scale factor."""
    with precise_decimal(dtype) as (upcast, sqrt, _ln):
        source_centered, _target_centered, cross_covariance, n_points = _center_and_covariance(source, target, upcast)
        jacobi_svd = _jacobi_svd_3x3(upcast, sqrt)
        left_vectors, singular_values, right_vectors = jacobi_svd(cross_covariance)
        rotation = [
            [sum(right_vectors[r][k] * left_vectors[c][k] for k in range(3)) for c in range(3)] for r in range(3)
        ]
        determinant_sign = upcast(1) if _determinant_3x3(rotation) >= 0 else upcast(-1)
        if determinant_sign < 0:
            for r in range(3):
                right_vectors[r][2] = -right_vectors[r][2]
            rotation = [
                [sum(right_vectors[r][k] * left_vectors[c][k] for k in range(3)) for c in range(3)] for r in range(3)
            ]
        source_variance = sum(source_centered[i][j] ** 2 for i in range(n_points) for j in range(3)) / n_points
        scale = (singular_values[0] + singular_values[1] + determinant_sign * singular_values[2]) / (
            n_points * source_variance
        )
        return float(scale)


KERNELS_MESH: dict[str, tuple[Callable | None, Callable, Callable]] = {
    "kabsch": (baseline_kabsch, nk.kabsch, precise_kabsch),
    "umeyama": (baseline_umeyama if numpy_available else None, nk.umeyama, precise_umeyama),
    "rmsd": (baseline_rmsd if numpy_available else None, nk.rmsd, precise_rmsd),
}


def _make_point_pair(n_points, dtype):
    """Create two distinct (n_points, 3) point clouds as (raw, baseline) pairs.

    Returns (source_raw, source_baseline, target_raw, target_baseline).
    """
    source_f32 = np.random.randn(n_points, 3).astype(np.float32)
    target_f32 = np.random.randn(n_points, 3).astype(np.float32)
    source_raw, source_baseline = downcast_f32_to_dtype(source_f32, dtype)
    target_raw, target_baseline = downcast_f32_to_dtype(target_f32, dtype)
    return source_raw, source_baseline, target_raw, target_baseline


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(reduced_repetitions_count)
@pytest.mark.parametrize("n_points", [mesh_points])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_rmsd_accuracy(n_points: int, dtype: str, capability: str, nk_seed: int):
    """RMSD of random point clouds against high-precision baseline."""
    if n_points < 3:
        pytest.skip("RMSD requires at least 3 points")

    source_raw, source_baseline, target_raw, target_baseline = _make_point_pair(n_points, dtype)
    atol, rtol = tolerances_for_dtype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_MESH["rmsd"]

    # High-precision baseline
    accurate_dt, accurate = profile(precise_kernel or baseline_kernel, source_baseline, target_baseline, dtype=dtype)

    # Native-precision baseline
    if baseline_kernel is not None:
        expected_dt, expected = profile(baseline_kernel, source_baseline, target_baseline, dtype=dtype)
    else:
        expected_dt, expected = 0, None

    # SIMD result
    source_nk = make_nk(source_raw, dtype)
    target_nk = make_nk(target_raw, dtype)
    result_dt, result_obj = profile(simd_kernel, source_nk, target_nk)
    result = float(np.array(result_obj.rmsd))

    assert_allclose(result, accurate, atol=atol, rtol=rtol)
    collect_errors("rmsd", n_points, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(reduced_repetitions_count)
@pytest.mark.parametrize("n_points", [mesh_points])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_kabsch_accuracy(n_points: int, dtype: str, capability: str, nk_seed: int):
    """Kabsch RMSD of random point clouds against high-precision Jacobi SVD baseline."""
    if n_points < 3:
        pytest.skip("Kabsch requires at least 3 non-degenerate points")

    source_raw, source_baseline, target_raw, target_baseline = _make_point_pair(n_points, dtype)
    atol, rtol = tolerances_for_dtype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_MESH["kabsch"]

    # High-precision baseline → scalar RMSD after optimal rotation
    accurate_dt, accurate = profile(precise_kernel or baseline_kernel, source_baseline, target_baseline, dtype=dtype)

    # SIMD result
    source_nk = make_nk(source_raw, dtype)
    target_nk = make_nk(target_raw, dtype)
    result_dt, result_obj = profile(simd_kernel, source_nk, target_nk)
    result = float(np.array(result_obj.rmsd))

    assert_allclose(result, accurate, atol=atol, rtol=rtol)
    collect_errors("kabsch", n_points, dtype, accurate, accurate_dt, None, 0, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(reduced_repetitions_count)
@pytest.mark.parametrize("n_points", [mesh_points])
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_umeyama_accuracy(n_points: int, dtype: str, capability: str, nk_seed: int):
    """Umeyama scale of random point clouds against high-precision baseline."""
    if n_points < 3:
        pytest.skip("Umeyama requires at least 3 non-degenerate points")

    # Generate source and a scaled+noisy version so the scale is non-trivial
    source_f32 = np.random.randn(n_points, 3).astype(np.float32)
    scale_factor = 1.5 + np.random.rand()  # random scale in [1.5, 2.5]
    target_f32 = (source_f32 * scale_factor + np.random.randn(n_points, 3).astype(np.float32) * 0.01).astype(np.float32)
    source_raw, source_baseline = downcast_f32_to_dtype(source_f32, dtype)
    target_raw, target_baseline = downcast_f32_to_dtype(target_f32, dtype)
    atol, rtol = tolerances_for_dtype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_MESH["umeyama"]

    # High-precision baseline → scalar scale factor
    accurate_dt, accurate = profile(precise_kernel or baseline_kernel, source_baseline, target_baseline, dtype=dtype)

    # SIMD result
    source_nk = make_nk(source_raw, dtype)
    target_nk = make_nk(target_raw, dtype)
    result_dt, result_obj = profile(simd_kernel, source_nk, target_nk)
    result = float(np.array(result_obj.scale))

    assert_allclose(result, accurate, atol=atol, rtol=rtol)
    collect_errors("umeyama", n_points, dtype, accurate, accurate_dt, None, 0, result, result_dt, stats)


@pytest.mark.parametrize("n_points", dense_dimensions)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_rmsd_self_zero(n_points: int, capability: str):
    """rmsd(cloud, cloud).rmsd ~ 0 for identical point clouds."""
    if n_points < 3:
        pytest.skip("RMSD requires at least 3 points")
    keep_one_capability(capability)
    point_cloud = nk.ones((n_points, 3), dtype="float64")
    result = nk.rmsd(point_cloud, point_cloud)
    assert_allclose(float(result.rmsd), 0.0)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("n_points", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_kabsch_identity(n_points: int, dtype: str, capability: str):
    """Kabsch on identical points: expects RMSD~0 and scale~1."""
    if n_points < 3:
        pytest.skip("Kabsch requires at least 3 non-degenerate points")
    points_f32 = np.random.randn(n_points, 3).astype(np.float32)
    points_raw, _ = downcast_f32_to_dtype(points_f32, dtype)
    point_cloud = make_nk(points_raw, dtype)
    keep_one_capability(capability)
    result = nk.kabsch(point_cloud, point_cloud)
    atol, rtol = tolerances_for_dtype(dtype)
    assert_allclose(float(np.array(result.rmsd)), 0.0, atol=atol, rtol=rtol)
    assert_allclose(float(np.array(result.scale)), 1.0, atol=atol, rtol=rtol)
