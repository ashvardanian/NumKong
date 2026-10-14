#!/usr/bin/env python3
"""Test the many-to-many geospatial distances: nk.sqeuclideans3d and the packed nk.haversines and nk.vincentys.

DTypes: float64, int32.
Baselines: NumPy unit-vector Haversine, NumPy Vincenty inverse formula, NumPy squared differences.
Matches C++ suite: test/geospatial.cpp.

File: test/geospatial.py
Author: Ash Vardanian
Date: December 22, 2025
"""

from __future__ import annotations

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


requires_numpy = pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")

EARTH_MEAN_RADIUS = 6371008.8
"""The IUGG mean radius R₁ in metres, the sphere of `haversines_packed`."""
EQUATORIAL_RADIUS, FLATTENING = 6378137.0, 1 / 298.257223563
"""The WGS-84 ellipsoid of `vincentys_packed`."""
POLAR_RADIUS = EQUATORIAL_RADIUS * (1 - FLATTENING)
ROWS, COLUMNS = 33, 17
"""Odd sizes, so every SIMD body also runs its tails."""


def random_points(
    generator: np.random.Generator, count: int, latitude_degrees: float, longitude_degrees: float
) -> tuple[np.ndarray, np.ndarray]:
    """`count` latitudes within ±`latitude_degrees` and longitudes within ±`longitude_degrees`, in radians."""
    latitudes = np.radians(generator.uniform(-latitude_degrees, latitude_degrees, count))
    longitudes = np.radians(generator.uniform(-longitude_degrees, longitude_degrees, count))
    return latitudes, longitudes


def unit_vectors(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """The [points,3] unit vectors of points on the sphere."""
    return np.stack(
        [np.cos(latitudes) * np.cos(longitudes), np.cos(latitudes) * np.sin(longitudes), np.sin(latitudes)], axis=-1
    )


def reference_haversines(
    a_latitudes: np.ndarray, a_longitudes: np.ndarray, b_latitudes: np.ndarray, b_longitudes: np.ndarray
) -> np.ndarray:
    """2R · atan2(‖a − b‖, ‖a + b‖) between every row and every column unit vector."""
    a = unit_vectors(a_latitudes, a_longitudes)[:, None, :]
    b = unit_vectors(b_latitudes, b_longitudes)[None, :, :]
    return 2 * EARTH_MEAN_RADIUS * np.arctan2(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))


def reference_vincentys(
    a_latitudes: np.ndarray, a_longitudes: np.ndarray, b_latitudes: np.ndarray, b_longitudes: np.ndarray
) -> np.ndarray:
    """Vincenty's inverse formula on WGS-84 between every row and every column, iterated until every pair settles."""
    a_reduced = np.arctan2((1 - FLATTENING) * np.sin(a_latitudes), np.cos(a_latitudes))[:, None]
    b_reduced = np.arctan2((1 - FLATTENING) * np.sin(b_latitudes), np.cos(b_latitudes))[None, :]
    a_sin, a_cos, b_sin, b_cos = np.sin(a_reduced), np.cos(a_reduced), np.sin(b_reduced), np.cos(b_reduced)
    longitude_difference = b_longitudes[None, :] - a_longitudes[:, None]
    auxiliary_longitude = longitude_difference
    for _ in range(100):
        auxiliary_sin, auxiliary_cos = np.sin(auxiliary_longitude), np.cos(auxiliary_longitude)
        sigma_sin = np.hypot(b_cos * auxiliary_sin, a_cos * b_sin - a_sin * b_cos * auxiliary_cos)
        sigma_cos = a_sin * b_sin + a_cos * b_cos * auxiliary_cos
        sigma = np.arctan2(sigma_sin, sigma_cos)
        azimuth_sin = a_cos * b_cos * auxiliary_sin / sigma_sin
        azimuth_cos_squared = 1 - azimuth_sin**2
        midpoint_cos = sigma_cos - 2 * a_sin * b_sin / azimuth_cos_squared
        correction = FLATTENING / 16 * azimuth_cos_squared * (4 + FLATTENING * (4 - 3 * azimuth_cos_squared))
        previous_longitude = auxiliary_longitude
        auxiliary_longitude = longitude_difference + (1 - correction) * FLATTENING * azimuth_sin * (
            sigma + correction * sigma_sin * (midpoint_cos + correction * sigma_cos * (2 * midpoint_cos**2 - 1))
        )
        if np.all(np.abs(auxiliary_longitude - previous_longitude) < 1e-12):
            break
    u_squared = azimuth_cos_squared * (EQUATORIAL_RADIUS**2 - POLAR_RADIUS**2) / POLAR_RADIUS**2
    big_a = 1 + u_squared / 16384 * (4096 + u_squared * (-768 + u_squared * (320 - 175 * u_squared)))
    big_b = u_squared / 1024 * (256 + u_squared * (-128 + u_squared * (74 - 47 * u_squared)))
    sigma_delta = (
        big_b
        * sigma_sin
        * (
            midpoint_cos
            + big_b
            / 4
            * (
                sigma_cos * (2 * midpoint_cos**2 - 1)
                - big_b / 6 * midpoint_cos * (4 * sigma_sin**2 - 3) * (4 * midpoint_cos**2 - 3)
            )
        )
    )
    return POLAR_RADIUS * big_a * (sigma - sigma_delta)


def record_errors(stats, capabilities, kind: str, dtype: str, expected, accurate_ns, result, result_ns) -> None:
    """Adds one many-to-many call to `stats`, keyed by its rows."""
    capability = capabilities.name.lower().replace("|", "+")
    collect_errors(
        kind, len(expected), dtype, expected, accurate_ns, None, 0, result, result_ns, stats, capability=capability
    )


@requires_numpy
@pytest.mark.repeat(SETTINGS.reduced_repetitions)
def test_haversines_packed(stats, capabilities, np_rng: np.random.Generator):
    """Haversine over the whole sphere against unit vectors in NumPy, down to coincident and antipodal pairs,
    with strided rows and float32 columns cast to float64."""
    angle = SETTINGS.max_coord_angle_degrees
    a_latitudes, a_longitudes = random_points(np_rng, 2 * ROWS, angle / 2, angle)
    a_latitudes, a_longitudes = a_latitudes[::2], a_longitudes[::2]
    b_latitudes, b_longitudes = random_points(np_rng, COLUMNS - 2, angle / 2, angle)
    b_latitudes = np.concatenate([b_latitudes, a_latitudes[:1], -a_latitudes[:1]]).astype(np.float32)
    b_longitudes = np.concatenate([b_longitudes, a_longitudes[:1], a_longitudes[:1] + np.pi]).astype(np.float32)
    a_packed, b_packed = nk.haversines_pack(a_latitudes, a_longitudes), nk.haversines_pack(b_latitudes, b_longitudes)
    result_ns, result = timed_call(nk.haversines_packed, a_packed, b_packed, capabilities=capabilities)
    result = np.asarray(result)
    accurate_ns, expected = timed_call(
        reference_haversines, a_latitudes, a_longitudes, b_latitudes.astype(np.float64), b_longitudes.astype(np.float64)
    )
    assert result.dtype == np.float64 and result.shape == (ROWS, COLUMNS)
    assert_allclose(result, expected, atol=1e-6, rtol=0)
    record_errors(stats, capabilities, "haversines", "float64", expected, accurate_ns, result, result_ns)


@requires_numpy
@pytest.mark.repeat(SETTINGS.reduced_repetitions)
def test_vincentys_packed(stats, capabilities, np_rng: np.random.Generator):
    """Vincenty against its inverse formula in NumPy, the longitudes within 160° of each other, away from the
    antipode where the iteration need not converge."""
    half_span = min(SETTINGS.max_coord_angle_degrees, 160.0) / 2
    a_latitudes, a_longitudes = random_points(np_rng, ROWS, half_span, half_span)
    b_latitudes, b_longitudes = random_points(np_rng, COLUMNS, half_span, half_span)
    a_packed, b_packed = nk.vincentys_pack(a_latitudes, a_longitudes), nk.vincentys_pack(b_latitudes, b_longitudes)
    result_ns, result = timed_call(nk.vincentys_packed, a_packed, b_packed, capabilities=capabilities)
    result = np.asarray(result)
    accurate_ns, expected = timed_call(reference_vincentys, a_latitudes, a_longitudes, b_latitudes, b_longitudes)
    assert result.dtype == np.float64 and result.shape == (ROWS, COLUMNS)
    assert_allclose(result, expected, atol=1e-3, rtol=0)
    record_errors(stats, capabilities, "vincentys", "float64", expected, accurate_ns, result, result_ns)


@requires_numpy
def test_vincentys_flinders_peak_to_buninyong(capabilities):
    """Vincenty's own worked example in the 1975 Survey Review, 54,972.271 m, from the kernel and the reference."""
    latitudes = -np.radians([37 + 57 / 60 + 3.72030 / 3600, 37 + 39 / 60 + 10.15610 / 3600])
    longitudes = np.radians([144 + 25 / 60 + 29.52440 / 3600, 143 + 55 / 60 + 35.38390 / 3600])
    flinders_peak = nk.vincentys_pack(latitudes[:1], longitudes[:1])
    buninyong = nk.vincentys_pack(latitudes[1:], longitudes[1:])
    result = np.asarray(nk.vincentys_packed(flinders_peak, buninyong, capabilities=capabilities))
    expected = reference_vincentys(latitudes[:1], longitudes[:1], latitudes[1:], longitudes[1:])
    assert_allclose(result, [[54972.2711]], atol=1e-3, rtol=0)
    assert_allclose(expected, [[54972.2711]], atol=1e-3, rtol=0)


@requires_numpy
@pytest.mark.repeat(SETTINGS.reduced_repetitions)
@pytest.mark.parametrize("dtype", ["int32", "float64"])
def test_sqeuclideans3d(stats, capabilities, dtype: str, np_rng: np.random.Generator):
    """int32 squares exactly up to Earth-fixed centimetres at 4,359 km of altitude, the axis differences
    filling 31 bits, and float64 within 1e-12 relative, with strided rows."""
    if dtype == "int32":
        a = np_rng.integers(-(2**30), 2**30, size=(2 * ROWS, 3), dtype=np.int32)[::2]
        b = np_rng.integers(-(2**30), 2**30, size=(COLUMNS, 3), dtype=np.int32)
        b[0], b[1] = 2**30 - 1, -(2**30)
    else:
        a = np_rng.normal(scale=EARTH_MEAN_RADIUS, size=(2 * ROWS, 3))[::2]
        b = np_rng.normal(scale=EARTH_MEAN_RADIUS, size=(COLUMNS, 3))
    result_ns, result = timed_call(nk.sqeuclideans3d, a, b, capabilities=capabilities)
    result = np.asarray(result)
    differences = a[:, None, :].astype(np.float64 if dtype == "float64" else np.int64) - b[None, :, :]
    if dtype == "int32":
        magnitudes = np.abs(differences).astype(np.uint64)
        accurate_ns, expected = timed_call(lambda: (magnitudes * magnitudes).sum(axis=-1, dtype=np.uint64))
        assert result.dtype == np.uint64
        np.testing.assert_array_equal(result, expected)
    else:
        accurate_ns, expected = timed_call(lambda: (differences * differences).sum(axis=-1))
        assert result.dtype == np.float64
        assert_allclose(result, expected, atol=0, rtol=1e-12)
    record_errors(stats, capabilities, "sqeuclideans3d", dtype, expected, accurate_ns, result, result_ns)


@requires_numpy
def test_empty_and_single_point(capabilities):
    """An empty side yields an empty matrix, and a single point lies within a nanometre of itself."""
    empty, latitudes, longitudes = np.empty(0), np.radians([51.5007]), np.radians([-0.1246])
    for pack, packed in ((nk.haversines_pack, nk.haversines_packed), (nk.vincentys_pack, nk.vincentys_packed)):
        nothing, single = pack(empty, empty), pack(latitudes, longitudes)
        assert len(nothing) == 0 and nothing.shape == (0, 2)
        assert np.asarray(packed(nothing, single, capabilities=capabilities)).shape == (0, 1)
        assert np.asarray(packed(single, nothing, capabilities=capabilities)).shape == (1, 0)
        assert_allclose(np.asarray(packed(single, single, capabilities=capabilities)), [[0.0]], atol=1e-9, rtol=0)
    point = np.array([[3, -4, 12]], dtype=np.int32)
    assert np.asarray(nk.sqeuclideans3d(point[:0], point, capabilities=capabilities)).shape == (0, 1)
    assert np.asarray(nk.sqeuclideans3d(point, -point, capabilities=capabilities)).tolist() == [[4 * 169]]


@requires_numpy
@pytest.mark.parametrize("kind", ["haversines", "vincentys"])
def test_packed_matrices(kind: str, np_rng: np.random.Generator):
    """A pack reports its point count, shape and size, which its header and planar arrays fix."""
    pack, packed_type = getattr(nk, f"{kind}_pack"), getattr(nk, f"{kind.capitalize()}PackedMatrix")
    latitudes, longitudes = random_points(np_rng, 5, 90.0, 180.0)
    packed, single = pack(latitudes, longitudes), pack(latitudes[:1], longitudes[:1])
    assert type(packed) is packed_type
    assert len(packed) == packed.points == 5 and packed.shape == (5, 2)
    assert packed.nbytes == packed_type.pack_size(5) and single.nbytes == packed_type.pack_size(1)
    assert repr(packed) == f"<numkong.{packed_type.__name__} points=5 nbytes={packed.nbytes}>"


@requires_numpy
def test_geospatial_errors():
    """Malformed coordinates, mixed packs and mismatched dtypes raise before any kernel runs."""
    zeros = np.zeros(3)
    with pytest.raises(ValueError):
        nk.haversines_pack(zeros, zeros[:2])
    with pytest.raises(ValueError):
        nk.vincentys_pack(zeros.reshape(3, 1), zeros.reshape(3, 1))
    haversines, vincentys = nk.haversines_pack(zeros, zeros), nk.vincentys_pack(zeros, zeros)
    with pytest.raises(TypeError):
        nk.haversines_packed(haversines, vincentys)
    with pytest.raises(TypeError):
        nk.vincentys_packed(vincentys, zeros)
    points = np.zeros((2, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        nk.sqeuclideans3d(points[:, :2], points[:, :2])
    with pytest.raises(TypeError):
        nk.sqeuclideans3d(points, points.astype(np.float64))
    with pytest.raises(LookupError):
        nk.sqeuclideans3d(points.astype(np.float32), points.astype(np.float32))
