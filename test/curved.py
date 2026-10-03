#!/usr/bin/env python3
"""Test curved-space distances: nk.bilinear, nk.mahalanobis.

DTypes: float64, float32, float16, bfloat16, e4m3, e5m2, e2m3, e3m2, complex64, complex128.
Baselines: high-precision Decimal quadratic forms, SciPy mahalanobis.
Matches C++ suite: test/curved.cpp.

File: test/curved.py
Author: Ash Vardanian
Date: September 4, 2024
"""

from __future__ import annotations

import atexit
import math
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
    NUMKONG_ATOL,
    NUMKONG_RTOL,
    SETTINGS,
    LazyFormat,
    assert_allclose,
    collect_errors,
    create_stats,
    downcast_f32_to_dtype,
    hex_array,
    numpy_available,
    precise_decimal,
    print_stats_report,
    timed_call,
)

import numkong as nk


stats = create_stats()
atexit.register(print_stats_report, stats)


def baseline_bilinear(x, y, z, dtype=None):
    return x @ z @ y


try:
    import scipy.spatial.distance as spd

    def baseline_mahalanobis(x: np.ndarray, y: np.ndarray, z: np.ndarray, dtype: str | None = None) -> float:
        with np.errstate(invalid="ignore"):
            result = float(spd.mahalanobis(x, y, z))
        if math.isnan(result):
            pytest.skip("SciPy Mahalanobis distance took the `sqrt` of a negative number")
        return result

except ImportError:

    def baseline_mahalanobis(x, y, z, dtype=None):
        diff = x - y
        return np.sqrt(diff @ z @ diff)


def make_positive_semidefinite(data):
    """Make a square matrix positive semi-definite in-place (Gershgorin diagonal dominance).

    Matches C++ make_positive_semidefinite() in test/curved.cpp.
    """
    n = data.shape[0]
    # Symmetrize
    data = (data + data.T) * 0.5
    # Diagonal dominance — each diagonal > absolute row sum of off-diagonals
    for i in range(n):
        row_sum = np.sum(np.abs(data[i])) - np.abs(data[i, i])
        data[i, i] = row_sum + 1.0
    return data


def precise_bilinear(left, right, matrix, dtype=None):
    """High-precision bilinear form left·matrix·right via Python Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        right_values = [upcast(x) for x in right]
        dimensions = len(left)
        total = upcast(0)
        for row in range(dimensions):
            left_element = upcast(left[row])
            for col in range(dimensions):
                total += left_element * upcast(matrix[row, col]) * right_values[col]
        return float(total)


def precise_mahalanobis(left, right, matrix, dtype=None):
    """High-precision Mahalanobis distance √((left−right)·matrix·(left−right)) via Python Decimal."""
    with precise_decimal(dtype) as (upcast, sqrt, _ln):
        differences = [upcast(x) - upcast(y) for x, y in zip(left, right)]
        dimensions = len(differences)
        total = upcast(0)
        for row in range(dimensions):
            for col in range(dimensions):
                total += differences[row] * upcast(matrix[row, col]) * differences[col]
        return float(sqrt(total))


KERNELS_CURVED: dict[str, tuple[Callable, Callable, Callable]] = {
    "bilinear": (baseline_bilinear, nk.bilinear, precise_bilinear),
    "mahalanobis": (baseline_mahalanobis, nk.mahalanobis, precise_mahalanobis),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(SETTINGS.reduced_repetitions)
@pytest.mark.parametrize("ndim", SETTINGS.curved_dims)
@pytest.mark.parametrize(
    "dtypes",
    [
        ("float64", "float64"),
        ("float32", "float32"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
    ],
)
@pytest.mark.parametrize("metric", ["bilinear", "mahalanobis"])
def test_curved_random_accuracy(capabilities, ndim: int, dtypes: str, metric: str, np_rng: np.random.Generator):
    """Bilinear and Mahalanobis for float and bfloat16 dtypes against high-precision baselines."""
    dtype, compute_dtype = dtypes

    # Generate structured data at f32
    a_vector_f32 = np.abs(np_rng.standard_normal(ndim).astype(np.float32))
    b_vector_f32 = np.abs(np_rng.standard_normal(ndim).astype(np.float32))
    a_vector_f32 /= np.sum(a_vector_f32)
    b_vector_f32 /= np.sum(b_vector_f32)
    c_matrix_f32 = make_positive_semidefinite(np_rng.standard_normal((ndim, ndim)).astype(np.float32))

    a_raw, a_baseline = downcast_f32_to_dtype(a_vector_f32, dtype)
    b_raw, b_baseline = downcast_f32_to_dtype(b_vector_f32, dtype)
    c_raw, c_baseline = downcast_f32_to_dtype(c_matrix_f32, dtype)

    baseline_kernel, simd_kernel, precise_kernel = KERNELS_CURVED[metric]

    # High-precision baseline
    accurate_ns, accurate = timed_call(
        precise_kernel or baseline_kernel, a_baseline, b_baseline, c_baseline, dtype=dtype
    )

    # Baseline at native compute precision (for error-stat collection)
    native_dt = np.dtype(compute_dtype).type
    expected_ns, expected = timed_call(
        baseline_kernel,
        a_baseline.astype(native_dt),
        b_baseline.astype(native_dt),
        c_baseline.astype(native_dt),
    )

    # SIMD result
    result_ns, result = timed_call(simd_kernel, a_raw, b_raw, c_raw, dtype, capabilities=capabilities)
    result = np.asarray(result)

    err_msg = LazyFormat(
        lambda: (
            f"\n{metric}({dtype}, ndim={ndim}):"
            f"\n  Accurate:  {accurate}"
            f"\n  Got:       {result}"
            f"\n  Raw a:     {hex_array(a_raw)}"
            f"\n  Raw b:     {hex_array(b_raw)}"
            f"\n  Raw c:     {hex_array(c_raw)}"
        )
    )

    assert_allclose(result, accurate, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL, err_msg=err_msg)
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
@pytest.mark.repeat(SETTINGS.reduced_repetitions)
@pytest.mark.parametrize("ndim", SETTINGS.curved_dims)
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_bilinear_complex_accuracy(capabilities, ndim: int, dtype: str, np_rng: np.random.Generator):
    """Complex bilinear form against NumPy at extended precision."""
    a_vector = (np_rng.standard_normal(ndim) + 1.0j * np_rng.standard_normal(ndim)).astype(dtype)
    b_vector = (np_rng.standard_normal(ndim) + 1.0j * np_rng.standard_normal(ndim)).astype(dtype)
    c_matrix = (np_rng.standard_normal((ndim, ndim)) + 1.0j * np_rng.standard_normal((ndim, ndim))).astype(dtype)

    baseline_kernel, simd_kernel, _ = KERNELS_CURVED["bilinear"]
    precise_dtype = np.clongdouble if dtype == "complex128" else np.complex128
    accurate_ns, accurate = timed_call(
        baseline_kernel,
        a_vector.astype(precise_dtype),
        b_vector.astype(precise_dtype),
        c_matrix.astype(precise_dtype),
    )
    expected_ns, expected = timed_call(baseline_kernel, a_vector, b_vector, c_matrix)
    result_ns, result = timed_call(simd_kernel, a_vector, b_vector, c_matrix, capabilities=capabilities)
    result = np.asarray(result)

    assert_allclose(result, accurate, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)
    collect_errors(
        "bilinear",
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
