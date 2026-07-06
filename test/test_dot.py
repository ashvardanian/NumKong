#!/usr/bin/env python3
"""Test inner products: nk.inner, nk.dot, nk.vdot.

Dtypes: float64, float32, float16, bfloat16, e4m3, e5m2, e2m3, e3m2, int8, uint8, complex64, complex128.
Baselines: high-precision Decimal accumulation, NumPy np.inner.
Matches C++ suite: test_dot.cpp.
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
    NATIVE_COMPUTE_DTYPE,
    NK_ATOL,
    NK_RTOL,
    LazyFormat,
    assert_allclose,
    collect_errors,
    collect_warnings,
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
from test_spatial import baseline_angular, baseline_euclidean, baseline_sqeuclidean

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
@pytest.mark.parametrize("capability", possible_capabilities)
def test_inner_random_accuracy(ndim: int, dtype: str, capability: str, nk_seed: int):
    """Inner product of random vectors across all numeric dtypes, verified against high-precision Decimal baseline."""
    a_raw, a_baseline = make_random((ndim,), dtype, seed=nk_seed)
    b_raw, b_baseline = make_random((ndim,), dtype, seed=nk_seed + 1)
    atol, rtol = tolerances_for_dtype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_DOT["inner"]

    # High-precision baseline
    accurate_dt, accurate = profile(precise_kernel or baseline_kernel, a_baseline, b_baseline, dtype=dtype)

    # Baseline at native precision (for error stats)
    if baseline_kernel is not None:
        native_dt = NATIVE_COMPUTE_DTYPE.get(dtype, np.float64)
        expected_dt, expected = profile(baseline_kernel, a_baseline.astype(native_dt), b_baseline.astype(native_dt))
    else:
        expected_dt, expected = 0, None

    # SIMD result — pass dtype for exotic types so the kernel knows the storage format
    result_dt, result = profile(simd_kernel, a_raw, b_raw, dtype)

    err_msg = LazyFormat(lambda: f"\ninner({dtype}, ndim={ndim}):\n  Accurate:  {accurate}\n  Got:       {result}")

    assert_allclose(result, accurate, atol=atol, rtol=rtol, err_msg=err_msg)
    collect_errors("inner", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dot_vdot_complex_accuracy(ndim: int, dtype: str, capability: str, nk_seed: int):
    """Complex dot and vdot products against NumPy for complex64 and complex128 inputs."""
    a_vector, a_baseline = make_random((ndim,), dtype, seed=nk_seed)
    b_vector, b_baseline = make_random((ndim,), dtype, seed=nk_seed + 1)
    atol, rtol = tolerances_for_dtype(dtype)

    keep_one_capability(capability)
    accurate_dt, accurate = profile(np.dot, a_baseline, b_baseline)
    expected_dt, expected = profile(np.dot, a_vector, b_vector)
    result_dt, result = profile(nk.dot, a_vector, b_vector)
    result = np.asarray(result)

    assert_allclose(result, expected, atol=atol, rtol=rtol)
    collect_errors("dot", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)

    accurate_dt, accurate = profile(np.vdot, a_baseline, b_baseline)
    expected_dt, expected = profile(np.vdot, a_vector, b_vector)
    result_dt, result = profile(nk.vdot, a_vector, b_vector)
    result = np.asarray(result)

    assert_allclose(result, expected, atol=atol, rtol=rtol)
    collect_errors("vdot", ndim, dtype + "c", accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dot_vdot_complex_explicit_dtype(ndim: int, capability: str, nk_seed: int):
    """Complex dot and vdot with explicit dtype='complex64' passed to float32 storage."""
    np.random.seed(nk_seed)
    a_real_parts = np.random.randn(ndim * 2).astype(dtype=np.float32)
    b_real_parts = np.random.randn(ndim * 2).astype(dtype=np.float32)

    keep_one_capability(capability)
    expected = np.dot(a_real_parts.view(np.complex64), b_real_parts.view(np.complex64))
    result = nk.dot(a_real_parts, b_real_parts, "complex64")

    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)

    expected = np.vdot(a_real_parts.view(np.complex64), b_real_parts.view(np.complex64))
    result = nk.vdot(a_real_parts, b_real_parts, "complex64")

    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skip(reason="Lacks overflow protection: https://github.com/ashvardanian/NumKong/issues/206")
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("metric", ["inner", "euclidean", "sqeuclidean", "angular"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_inner_float_overflow_detection(ndim: int, dtype: str, metric: str, capability: str):
    """Tests if the floating-point kernels are capable of detecting overflow yield the same ±inf result."""

    a = np.random.randn(ndim)
    b = np.random.randn(ndim)

    # Replace scalar at random position with infinity
    a[np.random.randint(ndim)] = np.inf
    a = a.astype(dtype)
    b = b.astype(dtype)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = KERNELS_OVERFLOW[metric]
    result = simd_kernel(a, b)
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
@pytest.mark.parametrize("capability", possible_capabilities)
def test_inner_known(ndim: int, dtype: str, capability: str):
    """inner(ones, ones) should equal n."""
    keep_one_capability(capability)
    ones_vector = nk.ones((ndim,), dtype=dtype)
    result = nk.inner(ones_vector, ones_vector)
    assert abs(result - ndim) < NK_ATOL + NK_RTOL * ndim


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_inner_orthogonal(ndim: int, dtype: str, capability: str):
    """inner(ones, zeros) should equal 0."""
    keep_one_capability(capability)
    ones_vector = nk.ones((ndim,), dtype=dtype)
    zeros_vector = nk.zeros((ndim,), dtype=dtype)
    result = nk.inner(ones_vector, zeros_vector)
    assert abs(result) < NK_ATOL


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_inner_symmetry(ndim: int, dtype: str, capability: str):
    """Commutativity: inner(a, b) = inner(b, a)."""
    keep_one_capability(capability)
    a = make_random_buffer(ndim, dtype)
    b = make_random_buffer(ndim, dtype)
    ab = nk.inner(a, b)
    ba = nk.inner(b, a)
    assert abs(ab - ba) < NK_ATOL, f"inner(a,b)={ab} != inner(b,a)={ba}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_inner_cauchy_schwarz(ndim: int, dtype: str, capability: str):
    """Cauchy-Schwarz: |inner(a,b)|^2 <= inner(a,a) * inner(b,b)."""
    keep_one_capability(capability)
    a = make_random_buffer(ndim, dtype)
    b = make_random_buffer(ndim, dtype)
    ab = nk.inner(a, b)
    aa = nk.inner(a, a)
    bb = nk.inner(b, b)
    assert ab * ab <= aa * bb + NK_ATOL, (
        f"Cauchy-Schwarz violated: |inner(a,b)|²={ab * ab} > inner(a,a)*inner(b,b)={aa * bb}"
    )


@pytest.mark.skip(reason="Lacks overflow protection: https://github.com/ashvardanian/NumKong/issues/206")
@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", [131072, 262144])
@pytest.mark.parametrize("metric", ["inner", "euclidean", "sqeuclidean", "angular"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_inner_integer_overflow_detection(ndim: int, metric: str, capability: str):
    """Tests if the integral kernels are capable of detecting overflow yield the same ±inf result,
    as with 2^16 elements accumulating "u32(u16(u8)*u16(u8))+u32" products should overflow and the
    same is true for 2^17 elements with "i32(i15(i8))*i32(i15(i8))" products.
    """

    a = np.full(ndim, fill_value=-128, dtype=np.int8)
    b = np.full(ndim, fill_value=-128, dtype=np.int8)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel = KERNELS_OVERFLOW[metric]
    _ = baseline_kernel(a, b)
    result = simd_kernel(a, b)
    assert np.isinf(result), f"Expected ±inf, but got {result}"

    try:
        expected_overflow = baseline_kernel(a, b)
        if not np.isinf(expected_overflow):
            collect_warnings("Overflow not detected in SciPy", stats)
    except Exception as e:
        collect_warnings(f"Arbitrary error raised in SciPy: {e}", stats)
