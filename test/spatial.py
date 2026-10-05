#!/usr/bin/env python3
"""Test spatial distances: nk.euclidean, nk.sqeuclidean, nk.angular.

DTypes: float64, float32, float16, bfloat16, e4m3, e5m2, e2m3, e3m2, int8, uint8.
Baselines: high-precision Decimal accumulation, SciPy spatial.distance.
Matches C++ suite: test/spatial.cpp.

File: test/spatial.py
Author: Ash Vardanian
Date: February 22, 2024
"""

from __future__ import annotations

import math
import random
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

from base import (
    NATIVE_COMPUTE_DTYPE,
    NUMKONG_ATOL,
    NUMKONG_RTOL,
    SETTINGS,
    LazyFormat,
    assert_allclose,
    collect_errors,
    make_random,
    make_random_buffer,
    numpy_available,
    precise_decimal,
    timed_call,
    tolerances_for_dtype,
)

import numkong as nk


algebraic_dtypes = ["float32", "float64"]
algebraic_ndims = [7, 97]

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


@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("ndim", SETTINGS.dims)
@pytest.mark.parametrize(
    "dtype",
    [
        "float64",
        "float32",
        "bfloat16",
        "float16",
        "e5m2",
        "e4m3",
        "e3m2",
        "e2m3",
        "int8",
        "uint8",
    ],
)
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "angular"])
def test_spatial_random_accuracy(stats, capabilities, ndim: int, dtype: str, metric: str, np_rng: np.random.Generator):
    """Spatial distances across all numeric dtypes against high-precision Decimal baselines."""
    a_raw, a_baseline = make_random((ndim,), dtype, np_rng)
    b_raw, b_baseline = make_random((ndim,), dtype, np_rng)
    atol, rtol = tolerances_for_dtype(dtype)

    baseline_kernel, simd_kernel, precise_kernel = KERNELS_SPATIAL[metric]

    # High-precision baseline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        accurate_ns, accurate = timed_call(precise_kernel or baseline_kernel, a_baseline, b_baseline, dtype=dtype)

    # Baseline at native precision (for error stats)
    if baseline_kernel is not None:
        native_dt = NATIVE_COMPUTE_DTYPE.get(dtype, np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected_ns, expected = timed_call(
                baseline_kernel, a_baseline.astype(native_dt), b_baseline.astype(native_dt)
            )
    else:
        expected_ns, expected = 0, None

    # SIMD result
    result_ns, result = timed_call(simd_kernel, a_raw, b_raw, dtype, capabilities=capabilities)

    err_msg = LazyFormat(lambda: f"\n{metric}({dtype}, ndim={ndim}):\n  Accurate:  {accurate}\n  Got:       {result}")

    assert_allclose(result, accurate, atol=atol, rtol=rtol, err_msg=err_msg)
    collect_errors(
        metric,
        ndim,
        dtype,
        accurate,
        accurate_ns,
        expected,
        expected_ns,
        result,
        result_ns,
        stats,
        capability=capabilities.name.lower().replace("|", "+"),
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", SETTINGS.dims)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_angular_zero_vector(capabilities, ndim: int, dtype: str, np_rng: np.random.Generator):
    """Tests the nk.angular() function with zero vectors, to catch division by zero errors."""
    a = np.zeros(ndim, dtype=dtype)
    b = (np_rng.standard_normal(ndim) + 1).astype(dtype)

    result = nk.angular(a, b, capabilities=capabilities)
    assert result == 1, f"Expected 1, but got {result}"

    result = nk.angular(a, a, capabilities=capabilities)
    assert result == 0, f"Expected 0 distance from itself, but got {result}"

    result = nk.angular(b, b, capabilities=capabilities)
    assert abs(result) < NUMKONG_ATOL, f"Expected 0 distance from itself, but got {result}"

    assert np.all(result >= 0), "Negative result for angular distance"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_spatial_self_distance_zero(capabilities, ndim: int, dtype: str):
    """d(v, v) should be 0 for euclidean, sqeuclidean, and angular."""
    v = nk.full((ndim,), 1.5, dtype=dtype)
    atol = NUMKONG_ATOL
    assert abs(nk.euclidean(v, v, capabilities=capabilities)) < atol
    assert abs(nk.sqeuclidean(v, v, capabilities=capabilities)) < atol
    assert abs(nk.angular(v, v, capabilities=capabilities)) < atol


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_euclidean_known(capabilities, ndim: int, dtype: str):
    """euclidean(ones, zeros) = sqrt(n)."""
    ones_vector = nk.ones((ndim,), dtype=dtype)
    zeros_vector = nk.zeros((ndim,), dtype=dtype)
    result = nk.euclidean(ones_vector, zeros_vector, capabilities=capabilities)
    expected = math.sqrt(ndim)
    assert abs(result - expected) < NUMKONG_ATOL + NUMKONG_RTOL * expected


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_sqeuclidean_known(capabilities, ndim: int, dtype: str):
    """sqeuclidean(ones, zeros) = n."""
    ones_vector = nk.ones((ndim,), dtype=dtype)
    zeros_vector = nk.zeros((ndim,), dtype=dtype)
    result = nk.sqeuclidean(ones_vector, zeros_vector, capabilities=capabilities)
    assert abs(result - ndim) < NUMKONG_ATOL + NUMKONG_RTOL * ndim


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_spatial_symmetry(capabilities, ndim: int, dtype: str, rng: random.Random):
    """Commutativity: d(a, b) = d(b, a) for all spatial metrics."""
    a = make_random_buffer(rng, ndim, dtype)
    b = make_random_buffer(rng, ndim, dtype)
    for metric_fn in [nk.euclidean, nk.sqeuclidean, nk.angular]:
        d_ab = metric_fn(a, b, capabilities=capabilities)
        d_ba = metric_fn(b, a, capabilities=capabilities)
        assert abs(d_ab - d_ba) < NUMKONG_ATOL, f"{metric_fn.__name__}: d(a,b)={d_ab} != d(b,a)={d_ba}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_spatial_non_negative(capabilities, ndim: int, dtype: str, rng: random.Random):
    """Non-negativity: d(a, b) >= 0 for all spatial metrics."""
    a = make_random_buffer(rng, ndim, dtype)
    b = make_random_buffer(rng, ndim, dtype)
    for metric_fn in [nk.euclidean, nk.sqeuclidean, nk.angular]:
        distance = metric_fn(a, b, capabilities=capabilities)
        assert distance >= -NUMKONG_ATOL, f"{metric_fn.__name__}: d(a,b)={distance} is negative"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_euclidean_triangle_inequality(capabilities, ndim: int, dtype: str, rng: random.Random):
    """Triangle inequality: euclidean(a, c) <= euclidean(a, b) + euclidean(b, c)."""
    a = make_random_buffer(rng, ndim, dtype)
    b = make_random_buffer(rng, ndim, dtype)
    c = make_random_buffer(rng, ndim, dtype)
    d_ac = nk.euclidean(a, c, capabilities=capabilities)
    d_ab = nk.euclidean(a, b, capabilities=capabilities)
    d_bc = nk.euclidean(b, c, capabilities=capabilities)
    assert d_ac <= d_ab + d_bc + NUMKONG_ATOL, (
        f"Triangle inequality violated: d(a,c)={d_ac} > d(a,b)+d(b,c)={d_ab + d_bc}"
    )
