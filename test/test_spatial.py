#!/usr/bin/env python3
"""Test spatial distances: nk.euclidean, nk.sqeuclidean, nk.angular.

Dtypes: float64, float32, float16, bfloat16, e4m3, e5m2, e2m3, e3m2, int8, uint8.
Baselines: high-precision Decimal accumulation, SciPy spatial.distance.
Matches C++ suite: test_spatial.cpp.
"""

import atexit
import math
import warnings
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
    NATIVE_COMPUTE_DTYPE,
    NK_ATOL,
    NK_RTOL,
    LazyFormat,
    assert_allclose,
    collect_errors,
    create_stats,
    dense_dimensions,
    keep_one_capability,
    make_random,
    make_random_buffer,
    nk_seed,  # noqa: F401 — pytest fixture
    numpy_available,
    possible_capabilities,
    precise_decimal,
    print_stats_report,
    profile,
    randomized_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    tolerances_for_dtype,
)

import numkong as nk


algebraic_dtypes = ["float32", "float64"]
algebraic_ndims = [7, 97]
stats = create_stats()
atexit.register(print_stats_report, stats)

try:
    import scipy.spatial.distance as spd

    def baseline_euclidean(x, y, dtype=None):
        return np.array(spd.euclidean(x, y))

    def baseline_sqeuclidean(x, y, dtype=None):
        return spd.sqeuclidean(x, y)

    def baseline_angular(x, y, dtype=None):
        return spd.cosine(x, y)

except ImportError:

    def baseline_angular(x, y, dtype=None):
        return 1.0 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    def baseline_euclidean(x, y, dtype=None):
        return np.array([np.sqrt(np.sum((x - y) ** 2))])

    def baseline_sqeuclidean(x, y, dtype=None):
        return np.sum((x - y) ** 2)


def precise_sqeuclidean(a, b, dtype=None):
    """High-precision squared Euclidean distance via Python Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        return float(sum((upcast(x) - upcast(y)) ** 2 for x, y in zip(a, b)))


def precise_euclidean(a, b, dtype=None):
    return math.sqrt(precise_sqeuclidean(a, b, dtype=dtype))


def precise_angular(a, b, dtype=None):
    """High-precision angular/cosine distance via Python Decimal."""
    with precise_decimal(dtype) as (upcast, sqrt, _ln):
        dot_product = upcast(0)
        norm_left_squared = upcast(0)
        norm_right_squared = upcast(0)
        for x, y in zip(a, b):
            left_value = upcast(x)
            right_value = upcast(y)
            dot_product += left_value * right_value
            norm_left_squared += left_value * left_value
            norm_right_squared += right_value * right_value
        denominator = sqrt(norm_left_squared * norm_right_squared)
        if norm_left_squared == 0 and norm_right_squared == 0:
            return 0.0
        if denominator == 0:
            return 1.0
        return float(1 - dot_product / denominator)


KERNELS_SPATIAL: dict[str, tuple[Callable | None, Callable, Callable]] = {
    "euclidean": (baseline_euclidean if numpy_available else None, nk.euclidean, precise_euclidean),
    "sqeuclidean": (baseline_sqeuclidean if numpy_available else None, nk.sqeuclidean, precise_sqeuclidean),
    "angular": (baseline_angular if numpy_available else None, nk.angular, precise_angular),
}


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "dtype",
    [
        "float64",
        "float32",
        "float16",
        "bfloat16",
        "e4m3",
        "e5m2",
        "e2m3",
        "e3m2",
        "int8",
        "uint8",
    ],
)
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "angular"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_spatial_random_accuracy(ndim: int, dtype: str, metric: str, capability: str, nk_seed: int):
    """Spatial distances across all numeric dtypes against high-precision Decimal baselines."""
    a_raw, a_baseline = make_random((ndim,), dtype, seed=nk_seed)
    b_raw, b_baseline = make_random((ndim,), dtype, seed=nk_seed + 1)
    atol, rtol = tolerances_for_dtype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_SPATIAL[metric]

    # High-precision baseline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        accurate_dt, accurate = profile(precise_kernel or baseline_kernel, a_baseline, b_baseline, dtype=dtype)

    # Baseline at native precision (for error stats)
    if baseline_kernel is not None:
        native_dt = NATIVE_COMPUTE_DTYPE.get(dtype, np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected_dt, expected = profile(baseline_kernel, a_baseline.astype(native_dt), b_baseline.astype(native_dt))
    else:
        expected_dt, expected = 0, None

    # SIMD result
    result_dt, result = profile(simd_kernel, a_raw, b_raw, dtype)

    err_msg = LazyFormat(lambda: f"\n{metric}({dtype}, ndim={ndim}):\n  Accurate:  {accurate}\n  Got:       {result}")

    assert_allclose(result, accurate, atol=atol, rtol=rtol, err_msg=err_msg)
    collect_errors(metric, ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_angular_zero_vector(ndim: int, dtype: str, capability: str):
    """Tests the nk.angular() function with zero vectors, to catch division by zero errors."""
    a = np.zeros(ndim, dtype=dtype)
    b = (np.random.randn(ndim) + 1).astype(dtype)
    keep_one_capability(capability)

    result = nk.angular(a, b)
    assert result == 1, f"Expected 1, but got {result}"

    result = nk.angular(a, a)
    assert result == 0, f"Expected 0 distance from itself, but got {result}"

    result = nk.angular(b, b)
    assert abs(result) < NK_ATOL, f"Expected 0 distance from itself, but got {result}"

    assert np.all(result >= 0), "Negative result for angular distance"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_spatial_self_distance_zero(ndim: int, dtype: str, capability: str):
    """d(v, v) should be 0 for euclidean, sqeuclidean, and angular."""
    keep_one_capability(capability)
    v = nk.full((ndim,), 1.5, dtype=dtype)
    atol = NK_ATOL
    assert abs(nk.euclidean(v, v)) < atol
    assert abs(nk.sqeuclidean(v, v)) < atol
    assert abs(nk.angular(v, v)) < atol


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_euclidean_known(ndim: int, dtype: str, capability: str):
    """euclidean(ones, zeros) = sqrt(n)."""
    keep_one_capability(capability)
    ones_vector = nk.ones((ndim,), dtype=dtype)
    zeros_vector = nk.zeros((ndim,), dtype=dtype)
    result = nk.euclidean(ones_vector, zeros_vector)
    expected = math.sqrt(ndim)
    assert abs(result - expected) < NK_ATOL + NK_RTOL * expected


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_sqeuclidean_known(ndim: int, dtype: str, capability: str):
    """sqeuclidean(ones, zeros) = n."""
    keep_one_capability(capability)
    ones_vector = nk.ones((ndim,), dtype=dtype)
    zeros_vector = nk.zeros((ndim,), dtype=dtype)
    result = nk.sqeuclidean(ones_vector, zeros_vector)
    assert abs(result - ndim) < NK_ATOL + NK_RTOL * ndim


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_spatial_symmetry(ndim: int, dtype: str, capability: str):
    """Commutativity: d(a, b) = d(b, a) for all spatial metrics."""
    keep_one_capability(capability)
    a = make_random_buffer(ndim, dtype)
    b = make_random_buffer(ndim, dtype)
    for metric_fn in [nk.euclidean, nk.sqeuclidean, nk.angular]:
        d_ab = metric_fn(a, b)
        d_ba = metric_fn(b, a)
        assert abs(d_ab - d_ba) < NK_ATOL, f"{metric_fn.__name__}: d(a,b)={d_ab} != d(b,a)={d_ba}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_spatial_non_negative(ndim: int, dtype: str, capability: str):
    """Non-negativity: d(a, b) >= 0 for all spatial metrics."""
    keep_one_capability(capability)
    a = make_random_buffer(ndim, dtype)
    b = make_random_buffer(ndim, dtype)
    for metric_fn in [nk.euclidean, nk.sqeuclidean, nk.angular]:
        distance = metric_fn(a, b)
        assert distance >= -NK_ATOL, f"{metric_fn.__name__}: d(a,b)={distance} is negative"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_euclidean_triangle_inequality(ndim: int, dtype: str, capability: str):
    """Triangle inequality: euclidean(a, c) <= euclidean(a, b) + euclidean(b, c)."""
    keep_one_capability(capability)
    a = make_random_buffer(ndim, dtype)
    b = make_random_buffer(ndim, dtype)
    c = make_random_buffer(ndim, dtype)
    d_ac = nk.euclidean(a, c)
    d_ab = nk.euclidean(a, b)
    d_bc = nk.euclidean(b, c)
    assert d_ac <= d_ab + d_bc + NK_ATOL, f"Triangle inequality violated: d(a,c)={d_ac} > d(a,b)+d(b,c)={d_ab + d_bc}"
