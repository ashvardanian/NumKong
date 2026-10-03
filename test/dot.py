#!/usr/bin/env python3
"""Test inner products: nk.inner, nk.dot, nk.vdot.

DTypes: float64, float32, float16, bfloat16, e4m3, e5m2, e2m3, e3m2, int8, uint8, complex64, complex128.
Baselines: high-precision Decimal accumulation, NumPy np.inner.
Matches C++ suite: test/dot.cpp.

File: test/dot.py
Author: Ash Vardanian
Date: September 5, 2024
"""

from __future__ import annotations

import atexit
import random
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
    collect_warnings,
    create_stats,
    make_random,
    make_random_buffer,
    numpy_available,
    precise_decimal,
    print_stats_report,
    timed_call,
    tolerances_for_dtype,
)
from spatial import baseline_angular, baseline_euclidean, baseline_sqeuclidean

import numkong as nk


algebraic_dtypes = ["float32", "float64"]
algebraic_ndims = [7, 97]

stats = create_stats()
atexit.register(print_stats_report, stats)


def baseline_inner(a, b, dtype=None):
    return np.inner(a, b)


baseline_inner = baseline_inner if numpy_available else None


def precise_inner(a, b, dtype=None):
    """High-precision inner product via Python Decimal, exceeding f118 accuracy."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        total = upcast(0)
        for x, y in zip(a, b):
            total += upcast(x) * upcast(y)
        return float(total)


KERNELS_DOT: dict[str, tuple[Callable | None, Callable, Callable]] = {
    "inner": (baseline_inner, nk.inner, precise_inner),
}

KERNELS_OVERFLOW: dict[str, tuple[Callable | None, Callable]] = {
    "inner": (baseline_inner, nk.inner),
    "euclidean": (baseline_euclidean, nk.euclidean),
    "sqeuclidean": (baseline_sqeuclidean, nk.sqeuclidean),
    "angular": (baseline_angular, nk.angular),
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
def test_inner_random_accuracy(capabilities, ndim: int, dtype: str, np_rng: np.random.Generator):
    """Inner product of random vectors across all numeric dtypes, verified against high-precision Decimal baseline."""
    a_raw, a_baseline = make_random((ndim,), dtype, np_rng)
    b_raw, b_baseline = make_random((ndim,), dtype, np_rng)
    atol, rtol = tolerances_for_dtype(dtype)

    baseline_kernel, simd_kernel, precise_kernel = KERNELS_DOT["inner"]

    # High-precision baseline
    accurate_ns, accurate = timed_call(precise_kernel or baseline_kernel, a_baseline, b_baseline, dtype=dtype)

    # Baseline at native precision (for error stats)
    if baseline_kernel is not None:
        native_dt = NATIVE_COMPUTE_DTYPE.get(dtype, np.float64)
        expected_ns, expected = timed_call(baseline_kernel, a_baseline.astype(native_dt), b_baseline.astype(native_dt))
    else:
        expected_ns, expected = 0, None

    # SIMD result — pass dtype for exotic types so the kernel knows the storage format
    result_ns, result = timed_call(simd_kernel, a_raw, b_raw, dtype, capabilities=capabilities)

    err_msg = LazyFormat(lambda: f"\ninner({dtype}, ndim={ndim}):\n  Accurate:  {accurate}\n  Got:       {result}")

    assert_allclose(result, accurate, atol=atol, rtol=rtol, err_msg=err_msg)
    collect_errors(
        "inner",
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
@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("ndim", SETTINGS.dims)
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_dot_vdot_complex_accuracy(capabilities, ndim: int, dtype: str, np_rng: np.random.Generator):
    """Complex dot and vdot products against NumPy for complex64 and complex128 inputs."""
    a_vector, a_baseline = make_random((ndim,), dtype, np_rng)
    b_vector, b_baseline = make_random((ndim,), dtype, np_rng)
    atol, rtol = tolerances_for_dtype(dtype)

    accurate_ns, accurate = timed_call(np.dot, a_baseline, b_baseline)
    expected_ns, expected = timed_call(np.dot, a_vector, b_vector)
    result_ns, result = timed_call(nk.dot, a_vector, b_vector, capabilities=capabilities)
    result = np.asarray(result)

    assert_allclose(result, expected, atol=atol, rtol=rtol)
    collect_errors(
        "dot",
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

    accurate_ns, accurate = timed_call(np.vdot, a_baseline, b_baseline)
    expected_ns, expected = timed_call(np.vdot, a_vector, b_vector)
    result_ns, result = timed_call(nk.vdot, a_vector, b_vector, capabilities=capabilities)
    result = np.asarray(result)

    assert_allclose(result, expected, atol=atol, rtol=rtol)
    collect_errors(
        "vdot",
        ndim,
        dtype + "c",
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
@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("ndim", SETTINGS.dims)
def test_dot_vdot_complex_explicit_dtype(capabilities, ndim: int, np_rng: np.random.Generator):
    """Complex dot and vdot with explicit dtype='complex64' passed to float32 storage."""
    a_real_parts = np_rng.standard_normal(ndim * 2).astype(dtype=np.float32)
    b_real_parts = np_rng.standard_normal(ndim * 2).astype(dtype=np.float32)

    expected = np.dot(a_real_parts.view(np.complex64), b_real_parts.view(np.complex64))
    result = nk.dot(a_real_parts, b_real_parts, "complex64", capabilities=capabilities)

    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    expected = np.vdot(a_real_parts.view(np.complex64), b_real_parts.view(np.complex64))
    result = nk.vdot(a_real_parts, b_real_parts, "complex64", capabilities=capabilities)

    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)


@pytest.mark.skip(reason="Lacks overflow protection: https://github.com/ashvardanian/NumKong/issues/206")
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("ndim", SETTINGS.dims)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("metric", ["inner", "euclidean", "sqeuclidean", "angular"])
def test_inner_float_overflow_detection(capabilities, ndim: int, dtype: str, metric: str, np_rng: np.random.Generator):
    """Tests if the floating-point kernels are capable of detecting overflow yield the same ±inf result."""

    a = np_rng.standard_normal(ndim)
    b = np_rng.standard_normal(ndim)

    # Replace scalar at random position with infinity
    a[np_rng.integers(ndim)] = np.inf
    a = a.astype(dtype)
    b = b.astype(dtype)

    baseline_kernel, simd_kernel = KERNELS_OVERFLOW[metric]
    result = simd_kernel(a, b, capabilities=capabilities)
    assert np.isinf(result), f"Expected ±inf, but got {result}"

    #! In the Euclidean (L2) distance, SciPy raises a `ValueError` from the underlying
    #! NumPy function: `ValueError: array must not contain infs or NaNs`.
    try:
        expected_overflow = baseline_kernel(a, b)
        if not np.isinf(expected_overflow):
            collect_warnings("Overflow not detected in SciPy", stats)
    except Exception as e:
        collect_warnings(f"Arbitrary error raised in SciPy: {e}", stats)


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_inner_known(capabilities, ndim: int, dtype: str):
    """inner(ones, ones) should equal n."""
    ones_vector = nk.ones((ndim,), dtype=dtype)
    result = nk.inner(ones_vector, ones_vector, capabilities=capabilities)
    assert abs(result - ndim) < NUMKONG_ATOL + NUMKONG_RTOL * ndim


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_inner_orthogonal(capabilities, ndim: int, dtype: str):
    """inner(ones, zeros) should equal 0."""
    ones_vector = nk.ones((ndim,), dtype=dtype)
    zeros_vector = nk.zeros((ndim,), dtype=dtype)
    result = nk.inner(ones_vector, zeros_vector, capabilities=capabilities)
    assert abs(result) < NUMKONG_ATOL


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_inner_symmetry(capabilities, ndim: int, dtype: str, rng: random.Random):
    """Commutativity: inner(a, b) = inner(b, a)."""
    a = make_random_buffer(rng, ndim, dtype)
    b = make_random_buffer(rng, ndim, dtype)
    ab = nk.inner(a, b, capabilities=capabilities)
    ba = nk.inner(b, a, capabilities=capabilities)
    assert abs(ab - ba) < NUMKONG_ATOL, f"inner(a,b)={ab} != inner(b,a)={ba}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_inner_cauchy_schwarz(capabilities, ndim: int, dtype: str, rng: random.Random):
    """Cauchy-Schwarz: |inner(a,b)|^2 <= inner(a,a) * inner(b,b)."""
    a = make_random_buffer(rng, ndim, dtype)
    b = make_random_buffer(rng, ndim, dtype)
    ab = nk.inner(a, b, capabilities=capabilities)
    aa = nk.inner(a, a, capabilities=capabilities)
    bb = nk.inner(b, b, capabilities=capabilities)
    assert ab * ab <= aa * bb + NUMKONG_ATOL, (
        f"Cauchy-Schwarz violated: |inner(a,b)|²={ab * ab} > inner(a,a)*inner(b,b)={aa * bb}"
    )


@pytest.mark.skip(reason="Lacks overflow protection: https://github.com/ashvardanian/NumKong/issues/206")
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("ndim", [131072, 262144])
@pytest.mark.parametrize("metric", ["inner", "euclidean", "sqeuclidean", "angular"])
def test_inner_integer_overflow_detection(capabilities, ndim: int, metric: str):
    """Tests if the integral kernels are capable of detecting overflow yield the same ±inf result,
    as with 2^16 elements accumulating "u32(u16(u8)*u16(u8))+u32" products should overflow and the
    same is true for 2^17 elements with "i32(i15(i8))*i32(i15(i8))" products.
    """

    a = np.full(ndim, fill_value=-128, dtype=np.int8)
    b = np.full(ndim, fill_value=-128, dtype=np.int8)

    baseline_kernel, simd_kernel = KERNELS_OVERFLOW[metric]
    _ = baseline_kernel(a, b)
    result = simd_kernel(a, b, capabilities=capabilities)
    assert np.isinf(result), f"Expected ±inf, but got {result}"

    try:
        expected_overflow = baseline_kernel(a, b)
        if not np.isinf(expected_overflow):
            collect_warnings("Overflow not detected in SciPy", stats)
    except Exception as e:
        collect_warnings(f"Arbitrary error raised in SciPy: {e}", stats)
