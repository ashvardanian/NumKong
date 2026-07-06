#!/usr/bin/env python3
"""Test reductions: nk.moments, nk.sum, nk.min, nk.max, nk.argmin, nk.argmax, nk.norm.

Dtypes: float64, float32, float16, int32.
Baselines: high-precision Decimal summation, NumPy reductions.
Matches C++ suite: test_reduce.cpp.
"""

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

from test_base import (
    NK_ATOL,
    NK_RTOL,
    assert_allclose,
    create_stats,
    dense_dimensions,
    keep_one_capability,
    make_nk,
    make_random,
    nk_seed,  # noqa: F401 — pytest fixture
    numpy_available,
    possible_capabilities,
    precise_decimal,
    print_stats_report,
    randomized_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    tolerances_for_dtype,
)

import numkong as nk


algebraic_dtypes = ["float32", "float64"]
algebraic_ndims = [7, 97]
stats = create_stats()
atexit.register(print_stats_report, stats)


def baseline_moments(a, dtype=None):
    """Reference moments: (sum, sum_of_squares) at f64 precision."""
    arr = np.asarray(a).astype(np.float64)
    return (np.sum(arr), np.sum(arr**2))


def baseline_sum(a, dtype=None):
    """Reference sum."""
    return np.sum(np.asarray(a))


def baseline_min(a, dtype=None):
    """Reference min."""
    return np.min(np.asarray(a))


def baseline_max(a, dtype=None):
    """Reference max."""
    return np.max(np.asarray(a))


def baseline_argmin(a, dtype=None):
    """Reference argmin."""
    return np.argmin(np.asarray(a))


def baseline_argmax(a, dtype=None):
    """Reference argmax."""
    return np.argmax(np.asarray(a))


def baseline_norm(a, dtype=None):
    """Reference L2 norm at f64 precision."""
    return np.linalg.norm(np.asarray(a).astype(np.float64))


def precise_sum(a, dtype=None):
    """High-precision sum via Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        return float(sum(upcast(x) for x in a))


def precise_min(a, dtype=None):
    return float(min(float(x) for x in a))


def precise_max(a, dtype=None):
    return float(max(float(x) for x in a))


def precise_argmin(a, dtype=None):
    values = [float(x) for x in a]
    return values.index(min(values))


def precise_argmax(a, dtype=None):
    values = [float(x) for x in a]
    return values.index(max(values))


def precise_norm(a, dtype=None):
    """High-precision L2 norm via Decimal."""
    with precise_decimal(dtype) as (upcast, sqrt, _ln):
        return float(sqrt(sum(upcast(x) ** 2 for x in a)))


KERNELS_REDUCE: dict[str, tuple[Callable | None, Callable, Callable | None]] = {
    "moments": (baseline_moments, nk.moments, None),
    "minmax": (
        lambda a: (np.asarray(a).min(), np.asarray(a).argmin(), np.asarray(a).max(), np.asarray(a).argmax()),
        nk.minmax,
        None,
    ),
    "sum": (baseline_sum, nk.sum, precise_sum),
    "min": (baseline_min, nk.min, precise_min),
    "max": (baseline_max, nk.max, precise_max),
    "argmin": (baseline_argmin, nk.argmin, precise_argmin),
    "argmax": (baseline_argmax, nk.argmax, precise_argmax),
    "norm": (baseline_norm, nk.norm, precise_norm),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32"), pytest.param("float16", id="f16")]
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_moments(ndim: int, dtype: str, capability: str):
    """Test nk.moments() against NumPy sum and sum-of-squares."""
    keep_one_capability(capability)
    np_arr = np.random.randn(ndim).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    baseline_kernel, simd_kernel, _ = KERNELS_REDUCE["moments"]
    nk_sum, nk_sum_sq = simd_kernel(nk_arr)

    expected_sum, expected_sum_sq = baseline_kernel(np_arr)
    atol, rtol = tolerances_for_dtype(dtype)
    assert_allclose(nk_sum, expected_sum, rtol=rtol, atol=atol)
    assert_allclose(nk_sum_sq, expected_sum_sq, rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32"), pytest.param("float16", id="f16")]
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_minmax(ndim: int, dtype: str, capability: str):
    """Test nk.minmax() against NumPy min/argmin/max/argmax."""
    keep_one_capability(capability)
    np_arr = np.random.randn(ndim).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    baseline_kernel, simd_kernel, _ = KERNELS_REDUCE["minmax"]
    nk_min, nk_argmin, nk_max, nk_argmax = simd_kernel(nk_arr)

    ref_min, ref_argmin, ref_max, ref_argmax = baseline_kernel(np_arr)
    atol, rtol = tolerances_for_dtype(dtype)
    assert_allclose(nk_min, ref_min, rtol=rtol, atol=atol)
    assert int(nk_argmin) == ref_argmin
    assert_allclose(nk_max, ref_max, rtol=rtol, atol=atol)
    assert int(nk_argmax) == ref_argmax


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", [1, 7, 16, 97])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("float16", id="f16"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_minmax_all_nan(ndim: int, dtype: str, capability: str):
    """All-NaN input returns None from minmax."""
    keep_one_capability(capability)
    np_arr = np.full(ndim, np.nan, dtype=dtype)
    nk_arr = make_nk(np_arr, dtype)
    assert nk.minmax(nk_arr) is None


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("float16", id="f16"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_minmax_mixed_nan(dtype: str, capability: str):
    """Mixed NaN + valid values returns correct min/max."""
    keep_one_capability(capability)
    np_arr = np.array([np.nan, 3.0, np.nan, 1.0, np.nan, 5.0, np.nan], dtype=dtype)
    nk_arr = make_nk(np_arr, dtype)
    result = nk.minmax(nk_arr)
    assert result is not None
    nk_min, nk_argmin, nk_max, nk_argmax = result
    assert float(nk_min) == pytest.approx(1.0, abs=0.1)
    assert int(nk_argmin) == 3
    assert float(nk_max) == pytest.approx(5.0, abs=0.1)
    assert int(nk_argmax) == 5


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", [1, 7, 16, 97])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("float16", id="f16"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_individual_reductions_all_nan(ndim: int, dtype: str, capability: str):
    """All-NaN input returns None from min, max, argmin, argmax."""
    keep_one_capability(capability)
    np_arr = np.full(ndim, np.nan, dtype=dtype)
    nk_arr = make_nk(np_arr, dtype)
    assert nk.min(nk_arr) is None
    assert nk.max(nk_arr) is None
    assert nk.argmin(nk_arr) is None
    assert nk.argmax(nk_arr) is None


@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("float16", id="f16"),
    ],
)
@pytest.mark.parametrize("metric", ["sum", "min", "max", "norm", "argmin", "argmax"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_module_level_reductions(ndim: int, dtype: str, capability: str, metric: str, nk_seed: int):
    """Test scalar reductions via KERNELS_REDUCE lookup."""
    keep_one_capability(capability)
    raw, baseline = make_random((ndim,), dtype, seed=nk_seed)
    nk_arr = make_nk(raw, dtype) if numpy_available else raw

    baseline_kernel, simd_kernel, precise_kernel = KERNELS_REDUCE[metric]
    accurate = (precise_kernel or baseline_kernel)(baseline, dtype=dtype)
    result = simd_kernel(nk_arr)

    if metric in ("argmin", "argmax"):
        assert result == accurate
    else:
        assert_allclose(result, accurate, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_sum_axis(dtype: str, capability: str):
    """sum(axis=) on 2D and 3D tensors vs NumPy, including out= parameter."""
    keep_one_capability(capability)
    atol, rtol = tolerances_for_dtype(dtype)
    # 2D
    np_arr = np.random.randn(5, 7).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axis in [0, 1]:
        result = np.asarray(nk_arr.sum(axis=axis))
        expected = np_arr.astype(np.float64).sum(axis=axis)
        assert_allclose(result, expected, rtol=rtol, atol=atol)
    # 3D
    np_arr3 = np.random.randn(3, 4, 5).astype(dtype)
    nk_arr3 = make_nk(np_arr3, dtype)
    for axis in [0, 1, 2]:
        result = np.asarray(nk_arr3.sum(axis=axis))
        expected = np_arr3.astype(np.float64).sum(axis=axis)
        assert_allclose(result, expected, rtol=rtol, atol=atol)
    # out= parameter: writes to pre-allocated tensor, returns it
    out = nk.zeros((7,), dtype="float64")
    ret = nk_arr.sum(axis=0, out=out)
    assert np.asarray(ret).ctypes.data == np.asarray(out).ctypes.data
    assert_allclose(np.asarray(out), np_arr.astype(np.float64).sum(axis=0), rtol=rtol, atol=atol)
    # out= shape mismatch raises
    with pytest.raises(ValueError):
        nk_arr.sum(axis=0, out=nk.zeros((999,), dtype="float64"))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_min_max_axis(dtype: str, capability: str):
    """min/max(axis=) on 2D tensors vs NumPy, including out= path."""
    keep_one_capability(capability)
    np_arr = np.random.randn(6, 8).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    atol, rtol = tolerances_for_dtype(dtype)
    for axis in [0, 1]:
        result_min = np.asarray(nk_arr.min(axis=axis))
        result_max = np.asarray(nk_arr.max(axis=axis))
        assert_allclose(result_min, np_arr.min(axis=axis), rtol=rtol, atol=atol)
        assert_allclose(result_max, np_arr.max(axis=axis), rtol=rtol, atol=atol)
        # out= must match the allocated result
        out_min = nk.zeros(result_min.shape, dtype=nk_arr.dtype)
        out_max = nk.zeros(result_max.shape, dtype=nk_arr.dtype)
        nk_arr.min(axis=axis, out=out_min)
        nk_arr.max(axis=axis, out=out_max)
        assert_allclose(np.asarray(out_min), result_min, rtol=rtol, atol=atol)
        assert_allclose(np.asarray(out_max), result_max, rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_argmin_argmax_axis(dtype: str, capability: str):
    """argmin/argmax(axis=) on 2D tensors vs NumPy."""
    keep_one_capability(capability)
    np_arr = np.random.randn(6, 8).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axis in [0, 1]:
        nk_argmin = np.asarray(nk_arr.argmin(axis=axis))
        nk_argmax = np.asarray(nk_arr.argmax(axis=axis))
        np.testing.assert_array_equal(nk_argmin, np_arr.argmin(axis=axis))
        np.testing.assert_array_equal(nk_argmax, np_arr.argmax(axis=axis))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_norm_axis(dtype: str, capability: str):
    """norm(axis=) vs np.linalg.norm(x, axis=), including out= path."""
    keep_one_capability(capability)
    atol, rtol = tolerances_for_dtype(dtype)
    np_arr = np.random.randn(5, 7).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axis in [0, 1]:
        result = np.asarray(nk_arr.norm(axis=axis))
        expected = np.linalg.norm(np_arr.astype(np.float64), axis=axis)
        assert_allclose(result, expected, rtol=rtol, atol=atol)
        # out= must match the allocated result
        out = nk.zeros(result.shape, dtype="float64")
        nk_arr.norm(axis=axis, out=out)
        assert_allclose(np.asarray(out), result, rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_keepdims(dtype: str, capability: str):
    """keepdims=True preserves rank with size-1 at reduced axis."""
    keep_one_capability(capability)
    np_arr = np.random.randn(4, 5).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axis in [0, 1]:
        expected_shape = list(np_arr.shape)
        expected_shape[axis] = 1
        for op in ["sum", "min", "max", "norm"]:
            result = getattr(nk_arr, op)(axis=axis, keepdims=True)
            assert result.ndim == 2, f"{op} keepdims ndim"
            assert list(result.shape) == expected_shape, f"{op} keepdims shape"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_module_level_axis(capability: str):
    """nk.sum(a, axis=0), nk.min(a, axis=1), etc."""
    keep_one_capability(capability)
    np_arr = np.random.randn(4, 5).astype("float64")
    nk_arr = make_nk(np_arr, "float64")
    assert_allclose(np.asarray(nk.sum(nk_arr, axis=0)), np_arr.sum(axis=0))
    assert_allclose(np.asarray(nk.min(nk_arr, axis=1)), np_arr.min(axis=1))
    assert_allclose(np.asarray(nk.max(nk_arr, axis=0)), np_arr.max(axis=0))
    np.testing.assert_array_equal(np.asarray(nk.argmin(nk_arr, axis=1)), np_arr.argmin(axis=1))
    np.testing.assert_array_equal(np.asarray(nk.argmax(nk_arr, axis=0)), np_arr.argmax(axis=0))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_negative_axis(capability: str):
    """axis=-1 on 2D and axis=-2 on 3D reduce the correct dimension."""
    keep_one_capability(capability)
    # 2D, axis=-1
    np_arr = np.random.randn(3, 4).astype("float64")
    nk_arr = make_nk(np_arr, "float64")
    result = np.asarray(nk_arr.sum(axis=-1))
    expected = np_arr.sum(axis=-1)
    assert_allclose(result, expected)
    # 3D, axis=-2
    np_arr3 = np.random.randn(2, 3, 4).astype("float64")
    nk_arr3 = make_nk(np_arr3, "float64")
    result3 = np.asarray(nk_arr3.sum(axis=-2))
    expected3 = np_arr3.sum(axis=-2)
    assert_allclose(result3, expected3)


@pytest.mark.parametrize("capability", possible_capabilities)
def test_axis_error(capability: str):
    """axis out of range raises ValueError."""
    keep_one_capability(capability)
    nk_arr = nk.zeros((3, 4), dtype="float64")
    with pytest.raises(ValueError, match="axis.*out of range"):
        nk_arr.sum(axis=2)
    with pytest.raises(ValueError, match="axis.*out of range"):
        nk_arr.sum(axis=-3)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_integer_axis_reductions(capability: str):
    """sum/min/max/argmin/argmax along axis on int32 tensors."""
    keep_one_capability(capability)
    np_arr = np.array([[10, 3, 7], [1, 8, 5], [4, 9, 2]], dtype=np.int32)
    nk_arr = make_nk(np_arr, "int32")
    for axis in [0, 1]:
        assert_allclose(np.asarray(nk_arr.sum(axis=axis)), np_arr.astype(np.float64).sum(axis=axis))
        np.testing.assert_array_equal(np.asarray(nk_arr.min(axis=axis)), np_arr.min(axis=axis))
        np.testing.assert_array_equal(np.asarray(nk_arr.max(axis=axis)), np_arr.max(axis=axis))
        np.testing.assert_array_equal(np.asarray(nk_arr.argmin(axis=axis)), np_arr.argmin(axis=axis))
        np.testing.assert_array_equal(np.asarray(nk_arr.argmax(axis=axis)), np_arr.argmax(axis=axis))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_norm_integer(capability: str):
    """norm(axis=) on int32 must include sqrt (the fix norm_slice was made for)."""
    keep_one_capability(capability)
    np_arr = np.array([[3, 4], [5, 12]], dtype=np.int32)
    nk_arr = make_nk(np_arr, "int32")
    for axis in [0, 1]:
        result = np.asarray(nk_arr.norm(axis=axis))
        expected = np.linalg.norm(np_arr.astype(np.float64), axis=axis)
        assert_allclose(result, expected)


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_sum_known(ndim: int, dtype: str, capability: str):
    """sum(ones(n)) ~ n."""
    keep_one_capability(capability)
    ones_tensor = nk.ones((ndim,), dtype=dtype)
    result = ones_tensor.sum()
    assert abs(result - ndim) < 0.1 + 0.1 * ndim


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_norm_known(ndim: int, dtype: str, capability: str):
    """norm(ones(n)) ~ sqrt(n)."""
    keep_one_capability(capability)
    ones_tensor = nk.ones((ndim,), dtype=dtype)
    result = nk.norm(ones_tensor)
    expected = math.sqrt(ndim)
    assert abs(result - expected) < 0.1 + 0.1 * expected


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_min_max_known(ndim: int, dtype: str, capability: str):
    """min(full(c)) = max(full(c)) = c."""
    keep_one_capability(capability)
    fill_value = 3.14
    constant_tensor = nk.full((ndim,), fill_value, dtype=dtype)
    assert abs(constant_tensor.min() - fill_value) < 0.01
    assert abs(constant_tensor.max() - fill_value) < 0.01


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_argmin_argmax_constant(ndim: int, dtype: str, capability: str):
    """For a constant tensor, argmin and argmax return valid indices in [0, n)."""
    keep_one_capability(capability)
    constant_tensor = nk.full((ndim,), 2.0, dtype=dtype)
    assert 0 <= constant_tensor.argmin() < ndim
    assert 0 <= constant_tensor.argmax() < ndim


# region Multi-axis reductions


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_sum(dtype: str, capability: str):
    """sum(axis=tuple) on 3D tensor vs NumPy."""
    keep_one_capability(capability)
    atol, rtol = tolerances_for_dtype(dtype)
    np_arr = np.random.randn(3, 4, 5).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axes in [(0, 1), (1, 2), (0, 2), (0, 1, 2)]:
        result = np.asarray(nk_arr.sum(axis=axes))
        expected = np_arr.astype(np.float64).sum(axis=axes)
        assert_allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_min_max(dtype: str, capability: str):
    """min/max(axis=tuple) on 3D tensor vs NumPy."""
    keep_one_capability(capability)
    atol, rtol = tolerances_for_dtype(dtype)
    np_arr = np.random.randn(3, 4, 5).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axes in [(0, 1), (1, 2), (0, 2)]:
        assert_allclose(np.asarray(nk_arr.min(axis=axes)), np_arr.min(axis=axes), rtol=rtol, atol=atol)
        assert_allclose(np.asarray(nk_arr.max(axis=axes)), np_arr.max(axis=axes), rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize(
    "dtype",
    [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_norm(dtype: str, capability: str):
    """norm(axis=tuple) vs manual sqrt(sum(x**2, axis=axes))."""
    keep_one_capability(capability)
    atol, rtol = tolerances_for_dtype(dtype)
    np_arr = np.random.randn(3, 4, 5).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    for axes in [(0, 1), (1, 2), (0, 2)]:
        result = np.asarray(nk_arr.norm(axis=axes))
        expected = np.sqrt(np.sum(np_arr.astype(np.float64) ** 2, axis=axes))
        assert_allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_keepdims(capability: str):
    """keepdims=True preserves rank with size-1 at each reduced axis."""
    keep_one_capability(capability)
    np_arr = np.random.randn(3, 4, 5).astype(np.float64)
    nk_arr = make_nk(np_arr, "float64")
    for axes in [(0, 2), (0, 1), (1, 2), (0, 1, 2)]:
        result = nk_arr.sum(axis=axes, keepdims=True)
        expected = np_arr.sum(axis=axes, keepdims=True)
        assert result.shape == expected.shape, f"axes={axes}: {result.shape} != {expected.shape}"
        assert_allclose(np.asarray(result), expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_negative(capability: str):
    """Negative indices in axis tuple are normalized correctly."""
    keep_one_capability(capability)
    np_arr = np.random.randn(3, 4, 5).astype(np.float64)
    nk_arr = make_nk(np_arr, "float64")
    result = np.asarray(nk_arr.sum(axis=(-1, 0)))
    expected = np_arr.sum(axis=(0, 2))  # -1 -> 2, sorted -> (0, 2)
    assert_allclose(result, expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_single_element_tuple(capability: str):
    """axis=(1,) should behave identically to axis=1."""
    keep_one_capability(capability)
    np_arr = np.random.randn(3, 4, 5).astype(np.float64)
    nk_arr = make_nk(np_arr, "float64")
    result_tuple = np.asarray(nk_arr.sum(axis=(1,)))
    result_int = np.asarray(nk_arr.sum(axis=1))
    assert_allclose(result_tuple, result_int)


@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_errors(capability: str):
    """Error cases for multi-axis reductions."""
    keep_one_capability(capability)
    nk_arr = nk.zeros((3, 4, 5), dtype="float64")

    # Duplicate axes
    with pytest.raises(ValueError, match="duplicate"):
        nk_arr.sum(axis=(1, 1))

    # Out of range
    with pytest.raises(ValueError, match="out of range"):
        nk_arr.sum(axis=(0, 5))

    # Empty tuple
    with pytest.raises(ValueError, match="empty"):
        nk_arr.sum(axis=())

    # argmin/argmax reject tuple axis
    with pytest.raises(TypeError):
        nk_arr.argmin(axis=(0, 1))
    with pytest.raises(TypeError):
        nk_arr.argmax(axis=(0, 1))

    # Non-int in tuple
    with pytest.raises(TypeError):
        nk_arr.sum(axis=(0, 1.5))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_multi_axis_module_level(capability: str):
    """Module-level nk.sum(arr, axis=tuple) works for buffer-protocol inputs."""
    keep_one_capability(capability)
    np_arr = np.random.randn(3, 4, 5).astype(np.float64)
    result = np.asarray(nk.sum(np_arr, axis=(0, 2)))
    expected = np_arr.sum(axis=(0, 2))
    assert_allclose(result, expected)


# endregion Multi-axis reductions

# region N-D tensor contraction tests
#
# These test the stride-collapsing optimization paths: uniform-stride tail detection,
# recursive re-analysis, and axis-reduction fast paths on contiguous non-axis dims.
# Each case is validated against a flattened-then-reduced reference (for global reductions)
# or NumPy (for axis reductions) to isolate the contraction logic from SIMD correctness.


def _nd_global_case(description, np_arr):
    """Verify global reduction on an N-D view matches a flat contiguous copy."""
    flat = np.ascontiguousarray(np_arr).reshape(-1)
    nk_sum, nk_sumsq = nk.moments(np_arr)
    ref_sum, ref_sumsq = nk.moments(nk.Tensor(flat))
    assert_allclose(nk_sum, ref_sum, err_msg=f"{description}: sum")
    assert_allclose(nk_sumsq, ref_sumsq, err_msg=f"{description}: sumsq")
    assert nk.min(np_arr) == nk.min(nk.Tensor(flat)), f"{description}: min mismatch"
    assert nk.max(np_arr) == nk.max(nk.Tensor(flat)), f"{description}: max mismatch"


def _nd_axis_case(description, np_arr, axis):
    """Verify axis reduction matches NumPy."""
    atol, rtol = tolerances_for_dtype(str(np_arr.dtype))
    nk_result = np.asarray(nk.sum(np_arr, axis=axis))
    np_result = np_arr.astype(np.float64).sum(axis=axis)
    assert_allclose(nk_result, np_result, atol=atol, rtol=rtol, err_msg=description)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_nd_contiguous_global_reduction(dtype: str):
    """Global reduction on contiguous N-D tensors: all dims should collapse into one kernel call."""
    np_dtype = np.dtype(dtype)
    rng = np.random.RandomState(42)
    for shape in [(4, 64, 64, 3), (2, 3, 4, 5, 6), (8, 12, 16), (1, 1, 256)]:
        arr = (
            rng.randint(0, 100, shape).astype(np_dtype)
            if np.issubdtype(np_dtype, np.integer)
            else rng.randn(*shape).astype(np_dtype)
        )
        _nd_global_case(f"contiguous {shape} {dtype}", arr)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_nd_strided_global_reduction(dtype: str):
    """Global reduction on non-contiguous but uniformly-strided subviews."""
    np_dtype = np.dtype(dtype)
    rng = np.random.RandomState(42)

    # Channel subview: (H, W, C)[:, :, k] → uniform stride = C * itemsize
    base = (
        rng.randint(0, 100, (64, 64, 3)).astype(np_dtype)
        if np.issubdtype(np_dtype, np.integer)
        else rng.randn(64, 64, 3).astype(np_dtype)
    )
    _nd_global_case(f"channel subview {dtype}", base[:, :, 1])

    # Row skip: (H, W)[::2, :] → first dim breaks, second dim contiguous
    base2d = (
        rng.randint(0, 100, (128, 64)).astype(np_dtype)
        if np.issubdtype(np_dtype, np.integer)
        else rng.randn(128, 64).astype(np_dtype)
    )
    _nd_global_case(f"row skip {dtype}", base2d[::2, :])

    # Batch skip on 4-D: (B, H, W, C)[::2, :, :, :]
    base4d = (
        rng.randint(0, 100, (4, 32, 32, 3)).astype(np_dtype)
        if np.issubdtype(np_dtype, np.integer)
        else rng.randn(4, 32, 32, 3).astype(np_dtype)
    )
    _nd_global_case(f"batch skip {dtype}", base4d[::2, :, :, :])


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_nd_reversed_global_reduction():
    """Global reduction on reversed (negative stride) views."""
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 200, (512,), dtype=np.uint8)
    _nd_global_case("reversed 1D", arr[::-1])
    arr2d = rng.randn(64, 32).astype(np.float32)
    _nd_global_case("reversed last dim", arr2d[:, ::-1])


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("axis", [0, 1, 2, 3, -1])
def test_nd_axis_reduction_contiguous(axis):
    """Axis reductions on contiguous 4-D image-like tensor."""
    rng = np.random.RandomState(42)
    arr = rng.randn(4, 32, 32, 3).astype(np.float32)
    _nd_axis_case(f"axis={axis} on (4,32,32,3)", arr, axis)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_nd_axis_reduction_on_subview():
    """Axis reductions on non-contiguous subviews."""
    rng = np.random.RandomState(42)
    base = rng.randn(8, 64, 64, 3).astype(np.float32)
    # Skip every other batch → non-contiguous outer dim, contiguous inner dims
    sub = base[::2, :, :, :]
    _nd_axis_case("axis=0 on batch-skipped", sub, 0)
    _nd_axis_case("axis=-1 on batch-skipped", sub, -1)
    _nd_axis_case("axis=2 on batch-skipped", sub, 2)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_nd_singleton_dims():
    """Reduction through singleton dimensions (extent=1) that should collapse freely."""
    rng = np.random.RandomState(42)
    arr = rng.randn(1, 64, 1, 64, 1).astype(np.float32)
    _nd_global_case("singleton dims", arr)
    _nd_axis_case("singleton axis=1", arr, 1)
    _nd_axis_case("singleton axis=3", arr, 3)


# endregion N-D tensor contraction tests
