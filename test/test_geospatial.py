#!/usr/bin/env python3
"""Test geospatial distances: nk.haversine, nk.vincenty.

Dtypes: float64, float32.
Baselines: NumPy great-circle and iterative Vincenty formulas.
Vincenty at float32 shows high relative error near antipodal points.
Matches C++ suite: test_geospatial.cpp.
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
    keep_one_capability,
    max_coord_angle,
    numpy_available,
    possible_capabilities,
    print_stats_report,
    profile,
    randomized_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
)

import numkong as nk


stats = create_stats()
atexit.register(print_stats_report, stats)

_max_angle_rad = np.radians(max_coord_angle) if numpy_available else None

earth_radius_meters = 6335439.0


def baseline_haversine(first_latitude, first_longitude, second_latitude, second_longitude):
    """Haversine distance using NumPy. All inputs in radians, output in meters."""
    latitude_difference = second_latitude - first_latitude
    longitude_difference = second_longitude - first_longitude
    haversine_term = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(first_latitude) * np.cos(second_latitude) * np.sin(longitude_difference / 2) ** 2
    )
    central_angle = 2 * np.arctan2(np.sqrt(haversine_term), np.sqrt(1 - haversine_term))
    return earth_radius_meters * central_angle


def baseline_vincenty(
    first_latitude, first_longitude, second_latitude, second_longitude, max_iterations=100, tolerance=1e-12
):
    """Vincenty distance using NumPy. All inputs in radians, output in meters."""
    equatorial_radius = 6378136.6
    polar_radius = 6356751.9
    flattening = (equatorial_radius - polar_radius) / equatorial_radius

    reduced_latitude_first = np.arctan((1 - flattening) * np.tan(first_latitude))
    reduced_latitude_second = np.arctan((1 - flattening) * np.tan(second_latitude))
    longitude_difference = second_longitude - first_longitude

    sin_reduced_first = np.sin(reduced_latitude_first)
    cos_reduced_first = np.cos(reduced_latitude_first)
    sin_reduced_second = np.sin(reduced_latitude_second)
    cos_reduced_second = np.cos(reduced_latitude_second)

    lambda_current = longitude_difference
    for _ in range(max_iterations):
        sin_lambda = np.sin(lambda_current)
        cos_lambda = np.cos(lambda_current)

        sin_sigma = np.sqrt(
            (cos_reduced_second * sin_lambda) ** 2
            + (cos_reduced_first * sin_reduced_second - sin_reduced_first * cos_reduced_second * cos_lambda) ** 2
        )
        cos_sigma = sin_reduced_first * sin_reduced_second + cos_reduced_first * cos_reduced_second * cos_lambda
        sigma = np.arctan2(sin_sigma, cos_sigma)

        sin_azimuth = cos_reduced_first * cos_reduced_second * sin_lambda / sin_sigma
        cos_squared_azimuth = 1 - sin_azimuth**2
        cos_two_sigma_midpoint = (
            cos_sigma - 2 * sin_reduced_first * sin_reduced_second / cos_squared_azimuth
            if cos_squared_azimuth != 0
            else 0
        )

        correction_term = flattening / 16 * cos_squared_azimuth * (4 + flattening * (4 - 3 * cos_squared_azimuth))
        lambda_next = longitude_difference + (1 - correction_term) * flattening * sin_azimuth * (
            sigma
            + correction_term
            * sin_sigma
            * (cos_two_sigma_midpoint + correction_term * cos_sigma * (-1 + 2 * cos_two_sigma_midpoint**2))
        )

        if np.abs(lambda_next - lambda_current) < tolerance:
            break
        lambda_current = lambda_next

    u_squared = cos_squared_azimuth * (equatorial_radius**2 - polar_radius**2) / polar_radius**2
    coefficient_a = 1 + u_squared / 16384 * (4096 + u_squared * (-768 + u_squared * (320 - 175 * u_squared)))
    coefficient_b = u_squared / 1024 * (256 + u_squared * (-128 + u_squared * (74 - 47 * u_squared)))
    delta_sigma = (
        coefficient_b
        * sin_sigma
        * (
            cos_two_sigma_midpoint
            + coefficient_b
            / 4
            * (
                cos_sigma * (-1 + 2 * cos_two_sigma_midpoint**2)
                - coefficient_b
                / 6
                * cos_two_sigma_midpoint
                * (-3 + 4 * sin_sigma**2)
                * (-3 + 4 * cos_two_sigma_midpoint**2)
            )
        )
    )

    return polar_radius * coefficient_a * (sigma - delta_sigma)


KERNELS_GEOSPATIAL: dict[str, tuple[Callable, Callable, None]] = {
    "haversine": (baseline_haversine, nk.haversine, None),
    "vincenty": (baseline_vincenty, nk.vincenty, None),
}


def _check_geospatial_accuracy(metric, ndim, dtype, coord_scale, atol, rtol):
    """Shared accuracy check for geospatial kernels."""
    baseline_kernel, simd_kernel, _ = KERNELS_GEOSPATIAL[metric]

    lat_scale = min(_max_angle_rad, np.pi) / 2
    lon_scale = min(_max_angle_rad, np.pi)
    first_latitudes = ((np.random.rand(ndim) - 0.5) * 2 * lat_scale * coord_scale).astype(dtype)
    first_longitudes = ((np.random.rand(ndim) - 0.5) * 2 * lon_scale * coord_scale).astype(dtype)
    second_latitudes = ((np.random.rand(ndim) - 0.5) * 2 * lat_scale * coord_scale).astype(dtype)
    second_longitudes = ((np.random.rand(ndim) - 0.5) * 2 * lon_scale * coord_scale).astype(dtype)

    def _baseline_loop(lat1, lon1, lat2, lon2):
        return np.array([baseline_kernel(lat1[i], lon1[i], lat2[i], lon2[i]) for i in range(len(lat1))])

    accurate_dt, accurate = profile(
        _baseline_loop,
        first_latitudes.astype(np.float64),
        first_longitudes.astype(np.float64),
        second_latitudes.astype(np.float64),
        second_longitudes.astype(np.float64),
    )
    expected_dt, expected = profile(
        _baseline_loop, first_latitudes, first_longitudes, second_latitudes, second_longitudes
    )

    result_dt, result = profile(simd_kernel, first_latitudes, first_longitudes, second_latitudes, second_longitudes)
    result = np.asarray(result)

    assert_allclose(result, accurate, atol=atol, rtol=rtol)

    # out= with nk.Tensor buffer
    out_nk = nk.zeros((ndim,), dtype=dtype)
    ret = simd_kernel(first_latitudes, first_longitudes, second_latitudes, second_longitudes, out=out_nk)
    assert ret is None
    assert_allclose(np.asarray(out_nk), result, atol=1e-10, rtol=1e-10)

    collect_errors(metric, ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_haversine_random_accuracy(ndim: int, dtype: str, capability: str):
    """Haversine great-circle distance against baseline for random coordinates."""
    keep_one_capability(capability)
    _check_geospatial_accuracy("haversine", ndim, dtype, coord_scale=1.0, atol=10.0, rtol=1e-2)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_vincenty_random_accuracy(ndim: int, dtype: str, capability: str):
    """Vincenty ellipsoidal geodesic distance against baseline for random coordinates."""
    keep_one_capability(capability)
    rtol = 1.0 if dtype == "float32" else 1e-2
    _check_geospatial_accuracy("vincenty", ndim, dtype, coord_scale=0.9, atol=100.0, rtol=rtol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_haversine_known():
    """Haversine distance for New York to Los Angeles against known reference value."""
    new_york_latitude = np.radians(40.7128)
    new_york_longitude = np.radians(-74.0060)
    los_angeles_latitude = np.radians(34.0522)
    los_angeles_longitude = np.radians(-118.2437)

    first_latitudes = np.array([new_york_latitude], dtype=np.float64)
    first_longitudes = np.array([new_york_longitude], dtype=np.float64)
    second_latitudes = np.array([los_angeles_latitude], dtype=np.float64)
    second_longitudes = np.array([los_angeles_longitude], dtype=np.float64)

    result = np.array(nk.haversine(first_latitudes, first_longitudes, second_latitudes, second_longitudes))
    result_kilometers = float(result[0]) / 1000

    assert 3800 < result_kilometers < 4100, f"Expected ~3940 km, got {result_kilometers:.0f} km"


@pytest.mark.parametrize("capability", possible_capabilities)
def test_haversine_self_zero(capability: str):
    """haversine(lat, lon, lat, lon) ~ 0."""
    keep_one_capability(capability)
    lat = nk.full((1,), 0.5, dtype="float64")
    lon = nk.full((1,), 0.5, dtype="float64")
    result = nk.haversine(lat, lon, lat, lon)
    val = next(iter(result))
    assert abs(val) < 1.0, f"haversine(self) = {val}, expected ~0"


@pytest.mark.parametrize("capability", possible_capabilities)
def test_vincenty_self_zero(capability: str):
    """vincenty(lat, lon, lat, lon) ~ 0."""
    keep_one_capability(capability)
    lat = nk.full((1,), 0.5, dtype="float64")
    lon = nk.full((1,), 0.5, dtype="float64")
    result = nk.vincenty(lat, lon, lat, lon)
    val = next(iter(result))
    assert abs(val) < 1.0, f"vincenty(self) = {val}, expected ~0"
