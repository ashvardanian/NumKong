#!/usr/bin/env python3
"""Test elementwise operations: nk.scale, nk.add, nk.blend, nk.fma, nk.multiply.

DTypes: float64, float32, float16, int8, uint8.
Baselines: high-precision Decimal per-element, NumPy at native precision.
Matches C++ suite: test/each.cpp.

File: test/each.py
Author: Ash Vardanian
Date: October 26, 2024
"""

from __future__ import annotations

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
    assert_allclose,
    collect_errors,
    collect_warnings,
    make_nk,
    make_random,
    numpy_available,
    precise_decimal,
    timed_call,
    tolerances_for_dtype,
)

import numkong as nk


algebraic_dtypes = ["float32", "float64"]
algebraic_ndims = [7, 97]


def normalize_elementwise(r, dtype_new):
    """Clips higher-resolution results to the smaller target dtype without overflow."""
    if np.issubdtype(dtype_new, np.integer):
        dtype_new_info = np.iinfo(dtype_new)
        r = np.nan_to_num(r, nan=0.0, posinf=dtype_new_info.max, neginf=dtype_new_info.min)
        r = np.clip(r, dtype_new_info.min, dtype_new_info.max, out=r)
        r = np.rint(r)
    return r.astype(dtype_new)


elementwise_shapes = [(d,) for d in SETTINGS.dims] + ([(6, 8), (4, 5, 3)] if numpy_available else [])
"""Elementwise ops are shape-invariant: sweep a couple of rank-N shapes (NumPy-only — the Decimal
baseline and the flattening below need it) alongside the 1-D `SETTINGS.dims`, so the N-D chunk
walkers are exercised. The baselines run on the flattened data; the SIMD op runs on the real
rank-N tensor and its result is flattened for comparison.
"""


def flatten_for_baseline(x):
    """Flatten to 1-D for per-element baseline comparison; passthrough without NumPy (1-D only)."""
    return np.asarray(x).reshape(-1) if numpy_available else x


def get_computation_dtypes(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    larger_dtype = np.promote_types(x.dtype, y.dtype)
    if larger_dtype == np.uint8:
        return np.uint16, larger_dtype
    elif larger_dtype == np.int8:
        return np.int16, larger_dtype
    if larger_dtype == np.uint16:
        return np.uint32, larger_dtype
    elif larger_dtype == np.int16:
        return np.int32, larger_dtype
    if larger_dtype == np.uint32:
        return np.uint64, larger_dtype
    elif larger_dtype == np.int32:
        return np.int64, larger_dtype
    else:
        return larger_dtype, larger_dtype


def baseline_scale(x, alpha, beta):
    """Scale operation: alpha * x + beta"""
    compute_dtype, _ = get_computation_dtypes(x, alpha)
    result = alpha * x.astype(compute_dtype) + beta
    return normalize_elementwise(result, x.dtype)


def baseline_blend(x, y, alpha, beta):
    """Weighted sum: alpha * x + beta * y"""
    compute_dtype, _ = get_computation_dtypes(x, y)
    result = x.astype(compute_dtype) * alpha + y.astype(compute_dtype) * beta
    return normalize_elementwise(result, x.dtype)


def baseline_fma(x, y, z, alpha, beta):
    """Fused multiply-add: alpha * x * y + beta * z"""
    compute_dtype, _ = get_computation_dtypes(x, y)
    result = x.astype(compute_dtype) * y.astype(compute_dtype) * alpha + z.astype(compute_dtype) * beta
    return normalize_elementwise(result, x.dtype)


def baseline_add(x, y, out=None):
    compute_dtype, final_dtype = get_computation_dtypes(x, y)
    a = x.astype(compute_dtype) if isinstance(x, np.ndarray) else x
    b = y.astype(compute_dtype) if isinstance(y, np.ndarray) else y
    result = np.add(a, b, out=out, casting="unsafe")
    result = normalize_elementwise(result, final_dtype)
    return result


def baseline_multiply(x, y, out=None):
    compute_dtype, final_dtype = get_computation_dtypes(x, y)
    a = x.astype(compute_dtype) if isinstance(x, np.ndarray) else x
    b = y.astype(compute_dtype) if isinstance(y, np.ndarray) else y
    result = np.multiply(a, b, out=out, casting="unsafe")
    result = normalize_elementwise(result, final_dtype)
    return result


def baseline_swiglu(gate, up, gate_scale, output_scale):
    """NumPy float64 reference for SwiGLU / SiLU over the rounded inputs."""
    g = np.asarray(gate, dtype=np.float64) * gate_scale
    y = g / (1.0 + np.exp(-g))  # SiLU
    if up is not None:
        y = y * np.asarray(up, dtype=np.float64)
    return y * output_scale


def baseline_rmsnorm(x, gamma, groups, epsilon):
    """NumPy float64 reference for grouped RMSNorm over the rounded input."""
    rows, width = x.shape
    columns = width // groups
    scaled = np.asarray(x, dtype=np.float64).reshape(rows, groups, columns)
    mean_sq = np.mean(scaled * scaled, axis=2, keepdims=True)
    normalized = scaled / np.sqrt(mean_sq + epsilon)
    if gamma is not None:
        normalized = normalized * np.asarray(gamma, dtype=np.float64).reshape(1, 1, columns)
    return normalized.reshape(rows, width)


_INT_CLIP_RANGES = {
    "int32": (-2147483648, 2147483647),
    "int16": (-32768, 32767),
    "int8": (-128, 127),
    "uint32": (0, 4294967295),
    "uint16": (0, 65535),
    "uint8": (0, 255),
}


def _clip_int(values, dtype):
    """Clip and round values to integer dtype range, mirroring normalize_elementwise."""
    clip_range = _INT_CLIP_RANGES.get(dtype)
    if clip_range is None:
        return values
    lo, hi = clip_range
    return [float(max(lo, min(hi, round(v)))) for v in values]


def precise_scale(a, alpha, beta, dtype=None):
    """High-precision scale: α·x + β via Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        alpha_value, beta_value = upcast(alpha), upcast(beta)
        result = [float(alpha_value * upcast(x) + beta_value) for x in a]
    return _clip_int(result, dtype) if dtype else result


def precise_add(a, b, dtype=None):
    """High-precision elementwise add via Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        result = [float(upcast(x) + upcast(y)) for x, y in zip(a, b)]
    return _clip_int(result, dtype) if dtype else result


def precise_blend(a, b, alpha, beta, dtype=None):
    """High-precision blend: α·x + β·y via Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        alpha_value, beta_value = upcast(alpha), upcast(beta)
        result = [float(alpha_value * upcast(x) + beta_value * upcast(y)) for x, y in zip(a, b)]
    return _clip_int(result, dtype) if dtype else result


def precise_fma(a, b, c, alpha, beta, dtype=None):
    """High-precision FMA: α·x·y + β·z via Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        alpha_value, beta_value = upcast(alpha), upcast(beta)
        result = [float(alpha_value * upcast(x) * upcast(y) + beta_value * upcast(z)) for x, y, z in zip(a, b, c)]
    return _clip_int(result, dtype) if dtype else result


def precise_multiply(a, b, dtype=None):
    """High-precision elementwise multiply via Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        result = [float(upcast(x) * upcast(y)) for x, y in zip(a, b)]
    return _clip_int(result, dtype) if dtype else result


KERNELS_EACH: dict[str, tuple[Callable | None, Callable, Callable]] = {
    "scale": (baseline_scale if numpy_available else None, nk.scale, precise_scale),
    "add": (baseline_add if numpy_available else None, nk.add, precise_add),
    "blend": (baseline_blend if numpy_available else None, nk.blend, precise_blend),
    "fma": (baseline_fma if numpy_available else None, nk.fma, precise_fma),
    "multiply": (baseline_multiply if numpy_available else None, nk.multiply, precise_multiply),
}


def random_coefficients(generator: np.random.Generator, dtype, alpha_div=2, beta_div=2):
    alpha = float(generator.standard_normal())
    beta = float(generator.standard_normal())
    if np.issubdtype(np.dtype(dtype), np.integer):
        alpha, beta = abs(alpha) / alpha_div, abs(beta) / beta_div
    return alpha, beta


@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("shape", elementwise_shapes)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
def test_scale_random_accuracy(stats, capabilities, shape: tuple, dtype: str, np_rng: np.random.Generator):
    """scale(alpha * x + beta) across float/int dtypes and ranks against a high-precision Decimal baseline."""
    input_raw, input_baseline = make_random(shape, dtype, np_rng)

    alpha, beta = random_coefficients(np_rng, dtype)

    baseline_kernel, simd_kernel, precise_kernel = KERNELS_EACH["scale"]

    # High-precision baseline (per-element Decimal on the flattened data)
    accurate_ns, accurate = timed_call(
        precise_kernel, flatten_for_baseline(input_baseline), alpha=alpha, beta=beta, dtype=dtype
    )

    # Native precision baseline, and the SIMD kernel on the real rank-N tensor
    expected_ns, expected = timed_call(baseline_kernel, flatten_for_baseline(input_raw), alpha=alpha, beta=beta)
    result_ns, result = timed_call(simd_kernel, input_raw, alpha=alpha, beta=beta, capabilities=capabilities)
    result = flatten_for_baseline(result)

    assert_allclose(result, accurate)
    collect_errors(
        "scale",
        len(accurate),
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

    # out= into a same-shape buffer returns None and matches the allocated result
    out_nk = nk.zeros(shape, dtype=dtype)
    assert simd_kernel(input_raw, alpha=alpha, beta=beta, out=out_nk, capabilities=capabilities) is None
    assert_allclose(flatten_for_baseline(out_nk), result)


@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("shape", elementwise_shapes)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
def test_add_random_accuracy(stats, capabilities, shape: tuple, dtype: str, np_rng: np.random.Generator):
    """Elementwise addition across float/int dtypes and ranks against a high-precision Decimal baseline."""
    a_raw, a_baseline = make_random(shape, dtype, np_rng)
    b_raw, b_baseline = make_random(shape, dtype, np_rng)

    baseline_kernel, simd_kernel, precise_kernel = KERNELS_EACH["add"]

    # High-precision baseline (per-element Decimal on the flattened data)
    accurate_ns, accurate = timed_call(
        precise_kernel, flatten_for_baseline(a_baseline), flatten_for_baseline(b_baseline), dtype=dtype
    )

    # Native precision baseline, and the SIMD kernel on the real rank-N tensor
    expected_ns, expected = timed_call(baseline_kernel, flatten_for_baseline(a_raw), flatten_for_baseline(b_raw))
    result_ns, result = timed_call(simd_kernel, a_raw, b_raw, capabilities=capabilities)
    result = flatten_for_baseline(result)

    assert_allclose(result, accurate)
    collect_errors(
        "add",
        len(accurate),
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

    # out= into a same-shape buffer returns None and matches the allocated result
    out_nk = nk.zeros(shape, dtype=dtype)
    assert simd_kernel(a_raw, b_raw, out=out_nk, capabilities=capabilities) is None
    assert_allclose(flatten_for_baseline(out_nk), result)


@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("shape", elementwise_shapes)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
def test_blend_random_accuracy(stats, capabilities, shape: tuple, dtype: str, np_rng: np.random.Generator):
    """Weighted sum (alpha * x + beta * y) across float/int dtypes and ranks against a Decimal baseline."""
    a_raw, a_baseline = make_random(shape, dtype, np_rng)
    b_raw, b_baseline = make_random(shape, dtype, np_rng)

    alpha, beta = random_coefficients(np_rng, dtype)

    baseline_kernel, simd_kernel, precise_kernel = KERNELS_EACH["blend"]

    # High-precision baseline (per-element Decimal on the flattened data)
    accurate_ns, accurate = timed_call(
        precise_kernel,
        flatten_for_baseline(a_baseline),
        flatten_for_baseline(b_baseline),
        alpha=alpha,
        beta=beta,
        dtype=dtype,
    )

    # Native precision baseline, and the SIMD kernel on the real rank-N tensor
    expected_ns, expected = timed_call(
        baseline_kernel, flatten_for_baseline(a_raw), flatten_for_baseline(b_raw), alpha=alpha, beta=beta
    )
    result_ns, result = timed_call(simd_kernel, a_raw, b_raw, alpha=alpha, beta=beta, capabilities=capabilities)
    result = flatten_for_baseline(result)

    assert_allclose(result, accurate)
    collect_errors(
        "blend",
        len(accurate),
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

    # out= into a same-shape buffer returns None and matches the allocated result
    out_nk = nk.zeros(shape, dtype=dtype)
    assert simd_kernel(a_raw, b_raw, alpha=alpha, beta=beta, out=out_nk, capabilities=capabilities) is None
    assert_allclose(flatten_for_baseline(out_nk), result)


@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("shape", elementwise_shapes)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "int8", "uint8"])
def test_fma_random_accuracy(stats, capabilities, shape: tuple, dtype: str, np_rng: np.random.Generator):
    """Fused multiply-add (alpha * x * y + beta * z) across float/int dtypes and ranks against a Decimal baseline."""
    a_raw, a_baseline = make_random(shape, dtype, np_rng)
    b_raw, b_baseline = make_random(shape, dtype, np_rng)
    c_raw, c_baseline = make_random(shape, dtype, np_rng)

    alpha, beta = random_coefficients(np_rng, dtype, 512, 3)

    baseline_kernel, simd_kernel, precise_kernel = KERNELS_EACH["fma"]

    # High-precision baseline (per-element Decimal on the flattened data)
    accurate_ns, accurate = timed_call(
        precise_kernel,
        flatten_for_baseline(a_baseline),
        flatten_for_baseline(b_baseline),
        flatten_for_baseline(c_baseline),
        alpha=alpha,
        beta=beta,
        dtype=dtype,
    )

    # Native precision baseline, and the SIMD kernel on the real rank-N tensor
    expected_ns, expected = timed_call(
        baseline_kernel,
        flatten_for_baseline(a_raw),
        flatten_for_baseline(b_raw),
        flatten_for_baseline(c_raw),
        alpha=alpha,
        beta=beta,
    )
    result_ns, result = timed_call(simd_kernel, a_raw, b_raw, c_raw, alpha=alpha, beta=beta, capabilities=capabilities)
    result = flatten_for_baseline(result)

    assert_allclose(result, accurate)
    collect_errors(
        "fma",
        len(accurate),
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

    # out= into a same-shape buffer returns None and matches the allocated result
    out_nk = nk.zeros(shape, dtype=dtype)
    ret = simd_kernel(a_raw, b_raw, c_raw, alpha=alpha, beta=beta, out=out_nk, capabilities=capabilities)
    assert ret is None
    assert_allclose(flatten_for_baseline(out_nk), result)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "first_dtype, second_dtype",
    [
        ("float64", "int16"),
        ("float64", "uint8"),
        ("float32", "int8"),
        ("float32", "int32"),
        ("float16", "int8"),
        ("float16", "uint16"),
        ("float32", "float64"),
        ("int16", "uint16"),
    ],
)
@pytest.mark.parametrize("kernel", ["add", "multiply"])
def test_add_multiply_mixed_dtype_promotion(first_dtype: str, second_dtype: str, kernel):
    """Mixed-dtype add and multiply promote like `np.result_type`."""
    _, simd_kernel, _ = KERNELS_EACH[kernel]
    a = np.ones(4, dtype=first_dtype)
    b = np.ones(4, dtype=second_dtype)
    assert np.asarray(simd_kernel(a, b)).dtype == np.result_type(a, b)
    assert np.asarray(simd_kernel(b, a)).dtype == np.result_type(a, b)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize(
    "dtype",
    [
        ("float64", "float64", "float64"),
        ("float32", "float32", "float32"),
        ("int8", "int8", "int8"),
        ("int16", "int16", "int16"),
        ("int32", "int32", "int32"),
        ("uint8", "uint8", "uint8"),
        ("uint16", "uint16", "uint16"),
        ("uint32", "uint32", "uint32"),
        ("float64", "int16", "uint16"),
        ("float32", "float32", "uint8"),
    ],
)
@pytest.mark.parametrize("kernel", ["add", "multiply"])
def test_add_multiply_noncontiguous(stats, capabilities, dtype: str, kernel, np_rng: np.random.Generator):
    """Add and multiply on non-contiguous, strided, and shape-mismatched arrays."""
    baseline_kernel, simd_kernel, _ = KERNELS_EACH[kernel]
    first_dtype, second_dtype, output_dtype = dtype
    operator = {"add": "+", "multiply": "*"}[kernel]

    def validate(a, b, inplace_numkong):
        result_numpy = baseline_kernel(a, b)
        result_numkong = np.array(simd_kernel(a, b, capabilities=capabilities))
        assert result_numkong.size == result_numpy.size, (
            f"Result sizes differ: {result_numkong.size} vs {result_numpy.size}"
        )
        assert result_numkong.shape == result_numpy.shape, (
            f"Result shapes differ: {result_numkong.shape} vs {result_numpy.shape}"
        )
        assert result_numkong.dtype == result_numpy.dtype, (
            f"Result dtypes differ: {result_numkong.dtype} vs {result_numpy.dtype} for ({a.dtype} {operator} {b.dtype})"
        )

        if not np.allclose(result_numkong, result_numpy, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL):
            assert_allclose(
                result_numkong,
                result_numpy,
                atol=NUMKONG_ATOL,
                rtol=NUMKONG_RTOL,
                err_msg=f"""
                Result mismatch for ({a.dtype} {operator} {b.dtype})
                First descriptor: {a.__array_interface__}
                Second descriptor: {b.__array_interface__}
                First operand: {a}
                Second operand: {b}
                NumKong result: {result_numkong}
                NumPy result: {result_numpy}
                """,
            )

        inplace_numpy = np.empty_like(inplace_numkong)
        simd_kernel(a, b, out=inplace_numkong, capabilities=capabilities)
        baseline_kernel(a, b, out=inplace_numpy)

        assert inplace_numkong.size == inplace_numpy.size
        assert inplace_numkong.shape == inplace_numpy.shape
        assert inplace_numkong.dtype == inplace_numpy.dtype

        mismatch_count = np.sum(~np.isclose(inplace_numkong, inplace_numpy, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL))
        if mismatch_count:
            collect_warnings(f"NumPy overflow in ({a.dtype} {operator} {b.dtype} -> {output_dtype})", stats)

        # out= with nk.Tensor buffer; differing dtypes are covered by test_strided_out_with_dtype_change
        if first_dtype == second_dtype == output_dtype and inplace_numkong.ndim == 1:
            out_nk = nk.zeros(inplace_numkong.shape, dtype=output_dtype)
            ret = simd_kernel(a, b, out=out_nk, capabilities=capabilities)
            assert ret is None
            assert_allclose(np.asarray(out_nk), inplace_numkong, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

        return result_numkong

    # Vector-Vector
    a = make_random((6,), first_dtype, np_rng)[0]
    b = make_random((6,), second_dtype, np_rng)[0]
    o = np.zeros(6).astype(output_dtype)
    validate(a, b, o)

    # Larger Vector-Vector
    a = make_random((47,), first_dtype, np_rng)[0]
    b = make_random((47,), second_dtype, np_rng)[0]
    o = np.zeros(47).astype(output_dtype)
    validate(a, b, o)

    # Much larger Vector-Vector
    a = make_random((247,), first_dtype, np_rng)[0]
    b = make_random((247,), second_dtype, np_rng)[0]
    o = np.zeros(247).astype(output_dtype)
    validate(a, b, o)

    # Vector-Scalar
    first_np_dt = np.dtype(first_dtype)
    second_np_dt = np.dtype(second_dtype)
    first_is_unsigned = np.issubdtype(first_np_dt, np.unsignedinteger)
    second_is_unsigned = np.issubdtype(second_np_dt, np.unsignedinteger)
    validate(a, first_np_dt.type(11 if first_is_unsigned else -11), o)
    validate(a, first_np_dt.type(7), o)

    # Scalar-Vector
    validate(second_np_dt.type(13 if second_is_unsigned else -13), b, o)
    validate(second_np_dt.type(5), b, o)

    # Matrix-Matrix
    a = make_random((10, 47), first_dtype, np_rng)[0]
    b = make_random((10, 47), second_dtype, np_rng)[0]
    o = np.zeros((10, 47)).astype(output_dtype)
    validate(a, b, o)

    # Strided Matrix-Matrix
    a_extended = make_random((10, 47), first_dtype, np_rng)[0]
    b_extended = make_random((10, 47), second_dtype, np_rng)[0]
    a = a_extended[::2, 1:]
    b = b_extended[1::2, :-1]
    o = np.zeros((5, 46)).astype(output_dtype)
    validate(a, b, o)

    # Strided Matrix-Matrix with reverse order
    a_extended = make_random((10, 47), first_dtype, np_rng)[0]
    b_extended = make_random((10, 47), second_dtype, np_rng)[0]
    a = a_extended[::-2, 1:]
    b = b_extended[1::2, -2::-1]
    o = np.zeros((5, 46)).astype(output_dtype)
    validate(a, b, o)

    # Shape mismatch errors
    a = make_random((10, 47), first_dtype, np_rng)[0]
    b = make_random((10, 46), second_dtype, np_rng)[0]
    with pytest.raises(ValueError):
        baseline_kernel(a, b)
    with pytest.raises(ValueError):
        simd_kernel(a, b, capabilities=capabilities)

    a = make_random((6, 2, 3), first_dtype, np_rng)[0]
    b = make_random((6, 6), second_dtype, np_rng)[0]
    with pytest.raises(ValueError):
        baseline_kernel(a, b)
    with pytest.raises(ValueError):
        simd_kernel(a, b, capabilities=capabilities)

    # Broadcasting not supported
    a = make_random((4, 7, 5, 3), first_dtype, np_rng)[0]
    b = make_random((1, 1, 1, 1), second_dtype, np_rng)[0]
    with pytest.raises(ValueError):
        simd_kernel(a, b, capabilities=capabilities)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("ndim", SETTINGS.dims)
@pytest.mark.parametrize(
    "dtype",
    [
        ("float64", "float64", "float64"),
        ("float32", "float32", "float32"),
        ("float16", "float16", "float16"),
        ("float64", "float64", "float32"),
        ("float32", "float32", "float16"),
    ],
)
@pytest.mark.parametrize("kernel", ["add", "multiply"])
def test_add_multiply_broadcast(capabilities, ndim: int, dtype: str, kernel, np_rng: np.random.Generator):
    """Add and multiply with scalar-vector and mixed-dtype broadcasting."""
    first_dtype, second_dtype, output_dtype = dtype

    baseline_kernel, simd_kernel, _ = KERNELS_EACH[kernel]

    # Vector-Vector
    a = np_rng.standard_normal(ndim).astype(first_dtype)
    b = np_rng.standard_normal(ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b, capabilities=capabilities))
    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # Scalar-Vector
    a = np_rng.standard_normal(1).astype(first_dtype)[0]
    b = np_rng.standard_normal(ndim).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b, capabilities=capabilities))
    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # Vector-Scalar
    a = np_rng.standard_normal(ndim).astype(first_dtype)
    b = np_rng.standard_normal(1).astype(second_dtype)[0]
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b, capabilities=capabilities))
    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # Matrix-Matrix
    a = np_rng.standard_normal((10, ndim // 10 if ndim >= 10 else ndim)).astype(first_dtype)
    b = np_rng.standard_normal((10, ndim // 10 if ndim >= 10 else ndim)).astype(second_dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b, capabilities=capabilities))
    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # In-place operation
    a = np_rng.standard_normal(ndim).astype(first_dtype)
    b = np_rng.standard_normal(ndim).astype(second_dtype)
    out_expected = np.zeros(ndim).astype(output_dtype)
    out_result = np.zeros(ndim).astype(output_dtype)
    baseline_kernel(a, b, out=out_expected)
    simd_kernel(a, b, out=out_result, capabilities=capabilities)
    assert_allclose(out_result, out_expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("ndim", SETTINGS.dims)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_scale_edge_cases(capabilities, ndim: int, dtype: str, np_rng: np.random.Generator):
    baseline_kernel, simd_kernel, _ = KERNELS_EACH["scale"]

    a = np_rng.standard_normal(ndim).astype(dtype)

    # Standard alpha and beta
    alpha = np_rng.standard_normal(1).astype(np.float64).item()
    beta = np_rng.standard_normal(1).astype(np.float64).item()
    expected = baseline_kernel(a, alpha=alpha, beta=beta)
    result = np.array(simd_kernel(a, alpha=alpha, beta=beta, capabilities=capabilities))
    assert_allclose(result, expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # Zero alpha
    expected = baseline_kernel(a, alpha=0.0, beta=1.5)
    result = np.array(simd_kernel(a, alpha=0.0, beta=1.5, capabilities=capabilities))
    assert_allclose(result, expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # Zero beta
    expected = baseline_kernel(a, alpha=2.0, beta=0.0)
    result = np.array(simd_kernel(a, alpha=2.0, beta=0.0, capabilities=capabilities))
    assert_allclose(result, expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # Negative alpha and beta
    expected = baseline_kernel(a, alpha=-1.5, beta=-2.0)
    result = np.array(simd_kernel(a, alpha=-1.5, beta=-2.0, capabilities=capabilities))
    assert_allclose(result, expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # out= with NumPy buffer
    out_np = np.zeros(ndim, dtype=dtype)
    ret = simd_kernel(a, alpha=alpha, beta=beta, out=out_np, capabilities=capabilities)
    assert ret is None
    assert_allclose(
        out_np, baseline_kernel(a, alpha=alpha, beta=beta).astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL
    )

    # out= with nk.Tensor buffer
    out_nk = nk.zeros((ndim,), dtype=dtype)
    ret = simd_kernel(a, alpha=alpha, beta=beta, out=out_nk, capabilities=capabilities)
    assert ret is None
    assert_allclose(
        np.asarray(out_nk),
        baseline_kernel(a, alpha=alpha, beta=beta).astype(np.float64),
        atol=NUMKONG_ATOL,
        rtol=NUMKONG_RTOL,
    )

    # out= shape mismatch raises
    with pytest.raises(ValueError):
        simd_kernel(a, alpha=alpha, beta=beta, out=np.zeros(ndim + 1, dtype=dtype), capabilities=capabilities)

    # out= into a non-contiguous 1D buffer is written correctly — the N-D walker honors strides
    # (consistent with add/multiply). Stride only matters when there is more than one element.
    if ndim > 1:
        strided_out = np.zeros(ndim * 2, dtype=dtype)[::2]
        assert simd_kernel(a, alpha=alpha, beta=beta, out=strided_out, capabilities=capabilities) is None
        assert_allclose(
            strided_out,
            baseline_kernel(a, alpha=alpha, beta=beta).astype(np.float64),
            atol=NUMKONG_ATOL,
            rtol=NUMKONG_RTOL,
        )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("ndim", SETTINGS.dims)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
def test_add_edge_cases(capabilities, ndim: int, dtype: str, np_rng: np.random.Generator):
    baseline_kernel, simd_kernel, _ = KERNELS_EACH["add"]

    # Standard random
    a = np_rng.standard_normal(ndim).astype(dtype)
    b = np_rng.standard_normal(ndim).astype(dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b, capabilities=capabilities))
    assert_allclose(result, expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # One vector is zeros
    b = np.zeros(ndim).astype(dtype)
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b, capabilities=capabilities))
    assert_allclose(result, expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # Both vectors the same
    expected = baseline_kernel(a, a)
    result = np.array(simd_kernel(a, a, capabilities=capabilities))
    assert_allclose(result, expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # Negative values
    a = -np.abs(np_rng.standard_normal(ndim).astype(dtype))
    b = -np.abs(np_rng.standard_normal(ndim).astype(dtype))
    expected = baseline_kernel(a, b)
    result = np.array(simd_kernel(a, b, capabilities=capabilities))
    assert_allclose(result, expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_add_numpy_buffer_protocol(dtype: str, np_rng: np.random.Generator):
    """nk.add() accepts NumPy arrays directly via buffer protocol."""
    a = np_rng.standard_normal(50).astype(dtype)
    b = np_rng.standard_normal(50).astype(dtype)

    expected = a + b
    result = nk.add(a, b)
    result_np = np.asarray(result)

    assert_allclose(result_np, expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_multiply_numpy_buffer_protocol(dtype: str, np_rng: np.random.Generator):
    """nk.multiply() accepts NumPy arrays directly via buffer protocol."""
    a = np_rng.standard_normal(50).astype(dtype)
    b = np_rng.standard_normal(50).astype(dtype)

    expected = a * b
    result = nk.multiply(a, b)
    result_np = np.asarray(result)

    assert_allclose(result_np, expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_blend_numpy_buffer_protocol(dtype: str, np_rng: np.random.Generator):
    """nk.blend() accepts NumPy arrays directly via buffer protocol, including out=."""
    a = np_rng.standard_normal(50).astype(dtype)
    b = np_rng.standard_normal(50).astype(dtype)
    alpha = 2.0
    beta = 0.5

    expected = alpha * a + beta * b
    result = nk.blend(a, b, alpha=alpha, beta=beta)
    result_np = np.asarray(result)
    assert_allclose(result_np, expected)

    # out= with NumPy buffer
    out_np = np.zeros(50, dtype=dtype)
    ret = nk.blend(a, b, alpha=alpha, beta=beta, out=out_np)
    assert ret is None
    assert_allclose(out_np, expected)

    # out= with nk.Tensor buffer
    out_nk = nk.zeros((50,), dtype=dtype)
    ret = nk.blend(a, b, alpha=alpha, beta=beta, out=out_nk)
    assert ret is None
    assert_allclose(np.asarray(out_nk), expected)

    # out= shape mismatch raises
    with pytest.raises(ValueError):
        nk.blend(a, b, alpha=alpha, beta=beta, out=np.zeros(49, dtype=dtype))

    # out= into a non-contiguous 1D buffer is written correctly — the N-D walker honors strides
    strided_out = np.zeros(100, dtype=dtype)[::2]
    assert nk.blend(a, b, alpha=alpha, beta=beta, out=strided_out) is None
    assert_allclose(strided_out, expected)


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_add_known(capabilities, ndim: int, dtype: str):
    """add(full(2), full(3)) ~ 5."""
    a = nk.full((ndim,), 2.0, dtype=dtype)
    b = nk.full((ndim,), 3.0, dtype=dtype)
    result = list(nk.add(a, b, capabilities=capabilities))
    for i in range(ndim):
        assert abs(result[i] - 5.0) < NUMKONG_ATOL, f"add(2,3)[{i}] = {result[i]}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_multiply_known(capabilities, ndim: int, dtype: str):
    """multiply(full(2), full(3)) ~ 6."""
    a = nk.full((ndim,), 2.0, dtype=dtype)
    b = nk.full((ndim,), 3.0, dtype=dtype)
    result = list(nk.multiply(a, b, capabilities=capabilities))
    for i in range(ndim):
        assert abs(result[i] - 6.0) < NUMKONG_ATOL, f"multiply(2,3)[{i}] = {result[i]}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_scale_identity(capabilities, ndim: int, dtype: str):
    """scale(v, alpha=1, beta=0) ~ v."""
    input_vector = nk.full((ndim,), 7.5, dtype=dtype)
    result = list(nk.scale(input_vector, alpha=1.0, beta=0.0, capabilities=capabilities))
    for i in range(ndim):
        assert abs(result[i] - 7.5) < NUMKONG_ATOL, f"scale(7.5, 1, 0)[{i}] = {result[i]}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_blend_known(capabilities, ndim: int, dtype: str):
    """blend(full(a), full(b), alpha=2, beta=3) ~ 2a + 3b."""
    a_val, b_val = 4.0, 5.0
    a = nk.full((ndim,), a_val, dtype=dtype)
    b = nk.full((ndim,), b_val, dtype=dtype)
    expected = 2.0 * a_val + 3.0 * b_val  # 23.0
    result = list(nk.blend(a, b, alpha=2.0, beta=3.0, capabilities=capabilities))
    for i in range(ndim):
        assert abs(result[i] - expected) < NUMKONG_ATOL, f"blend[{i}] = {result[i]}, expected {expected}"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("shape", [(1, 8), (3, 100), (8, 128), (5, 63)])
@pytest.mark.parametrize("with_up", [False, True])
@pytest.mark.parametrize(
    "dtype", [pytest.param("float32", id="f32"), pytest.param("bf16", id="bf16"), pytest.param("e4m3", id="e4m3")]
)
def test_swiglu(capabilities, shape, with_up, dtype, np_rng: np.random.Generator):
    """Test nk.swiglu() (and plain SiLU when up=None) against a float64 reference."""
    gate_raw, gate_base = make_random(shape, dtype, np_rng)
    nk_gate = make_nk(gate_raw, dtype)
    if with_up:
        up_raw, up_base = make_random(shape, dtype, np_rng)
        nk_up = make_nk(up_raw, dtype)
    else:
        nk_up, up_base = None, None
    gate_scale, output_scale = (0.75, 0.5) if dtype == "e4m3" else (1.0, 1.0)

    result = nk.swiglu(nk_gate, nk_up, gate_scale=gate_scale, output_scale=output_scale, capabilities=capabilities)
    y = np.asarray(result if dtype == "float32" else result.astype("float32"))

    expected = baseline_swiglu(gate_base, up_base, gate_scale, output_scale)
    if dtype != "float32":  # round the reference through the lossy output dtype (matches what the kernel stores)
        expected = np.asarray(
            nk.Tensor(np.ascontiguousarray(expected.astype(np.float32))).astype(dtype).astype("float32")
        ).astype(np.float64)
    atol, rtol = tolerances_for_dtype(dtype)
    assert_allclose(y, expected, atol=atol, rtol=rtol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("shape", [(1, 8), (3, 100), (8, 128), (7, 63)])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("with_gamma", [False, True])
@pytest.mark.parametrize(
    "dtype", [pytest.param("float32", id="f32"), pytest.param("bf16", id="bf16"), pytest.param("e4m3", id="e4m3")]
)
def test_rmsnorm(capabilities, shape, groups, with_gamma, dtype, np_rng: np.random.Generator):
    """Test nk.rmsnorm() against a float64 reference; low-precision dtypes use dtype-aware tolerances."""
    _rows, width = shape
    if width % groups != 0:
        pytest.skip("width not divisible by groups")
    columns = width // groups
    x_raw, x_base = make_random(shape, dtype, np_rng)
    nk_x = make_nk(x_raw, dtype)
    gamma = make_random((columns,), "float32", np_rng)[0] if with_gamma else None

    result = nk.rmsnorm(nk_x, gamma, groups=groups, epsilon=1e-6, capabilities=capabilities)
    y = np.asarray(result if dtype == "float32" else result.astype("float32"))

    expected = baseline_rmsnorm(x_base, gamma, groups, 1e-6)
    if dtype != "float32":  # round the reference through the lossy output dtype (matches what the kernel stores)
        expected = np.asarray(
            nk.Tensor(np.ascontiguousarray(expected.astype(np.float32))).astype(dtype).astype("float32")
        ).astype(np.float64)
    atol, rtol = tolerances_for_dtype(dtype)
    assert_allclose(y, expected, atol=atol, rtol=rtol)


def test_rmsnorm_strided_qk_norm(np_rng: np.random.Generator):
    """Unit QK-norm shape: a strided [tokens, depth] view of a fused [tokens, 3*hidden] buffer."""
    tokens, hidden = 5, 96
    qkv = np_rng.standard_normal((tokens, 3 * hidden)).astype(np.float32)
    q_view = qkv[:, 0:hidden]  # row stride = 3*hidden*4 bytes
    heads = 3
    out = np.asarray(nk.rmsnorm(q_view, None, groups=heads, epsilon=1e-6, capabilities=nk.Capability.SERIAL))
    expected = baseline_rmsnorm(np.ascontiguousarray(q_view), None, heads, 1e-6)
    assert_allclose(out, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_rmscast_int8_rounds_and_saturates():
    """Values of ±8 make the inverse RMS exactly 1/8, so each int8 output is ±gamma rounded to nearest even."""
    x = np.array([[8, -8, 8, -8, 8, -8, 8, -8]], dtype=np.int32)
    gamma = np.array([0.5, 1.5, 2.5, -2.5, 126.5, 127.5, 300.0, -300.0], dtype=np.float32)
    y = np.asarray(nk.rmscast(x, gamma, dtype="int8"))
    np.testing.assert_array_equal(y, np.array([[0, -2, 2, 2, 126, -128, 127, 127]], dtype=np.int8))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("kernel", ["add", "multiply"])
@pytest.mark.parametrize("second_is_scalar", [True, False], ids=["scalar_b", "array_b"])
def test_strided_out_with_dtype_change(capabilities, kernel: str, second_is_scalar: bool):
    """A strided `out` of a differing dtype must be scattered, not packed over its neighbours.

    The scalar and array forms of `b` select different implementations, and each of
    `add`/`multiply` has its own pair, so this covers all four writeback paths.
    """
    simd_kernel = getattr(nk, kernel)
    first = np.arange(6, dtype=np.int16)
    second = 5 if second_is_scalar else np.arange(6, dtype=np.int16)
    expected = first + second if kernel == "add" else first * second

    # Column 0 of a 2-column array: same shape as the input, stride of two float64s.
    surrounding = np.zeros((6, 2), dtype=np.float64)
    simd_kernel(first, second, out=surrounding[:, 0], capabilities=capabilities)
    assert_allclose(surrounding[:, 0], expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)
    assert not surrounding[:, 1].any(), "writeback spilled into the neighbouring column"

    # Reversed view exercises the negative-stride path.
    reversed_out = np.zeros(6, dtype=np.float64)[::-1]
    simd_kernel(first, second, out=reversed_out, capabilities=capabilities)
    assert_allclose(reversed_out, expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # A packed `out` must still take the single bulk-cast path.
    contiguous_out = np.zeros(6, dtype=np.float64)
    simd_kernel(first, second, out=contiguous_out, capabilities=capabilities)
    assert_allclose(contiguous_out, expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("kernel", ["add", "multiply"])
def test_strided_out_multidimensional(kernel: str):
    """Rank-2 and rank-3 outputs cover both a fully scattered tail and a partially packed one."""
    simd_kernel = getattr(nk, kernel)

    # Rank 2, innermost dimension strided: no contiguous tail at all.
    first = np.arange(12, dtype=np.int16).reshape(3, 4)
    expected = first + 2 if kernel == "add" else first * 2
    surrounding = np.zeros((3, 4, 2), dtype=np.float64)
    simd_kernel(first, 2, out=surrounding[:, :, 0])
    assert_allclose(surrounding[:, :, 0], expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)
    assert not surrounding[:, :, 1].any(), "writeback spilled into the neighbouring column"

    # Rank 3 with a packed innermost pair: recursion plus a bulk slice per row.
    first = np.arange(24, dtype=np.int16).reshape(2, 6, 2)
    expected = first + 3 if kernel == "add" else first * 3
    surrounding = np.zeros((2, 6, 4), dtype=np.float64)
    simd_kernel(first, 3, out=surrounding[:, :, 0:2])
    assert_allclose(surrounding[:, :, 0:2], expected.astype(np.float64), atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)
    assert not surrounding[:, :, 2:].any(), "writeback spilled past the requested columns"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype,distinct_values",
    [pytest.param("uint4", 8, id="uint4"), pytest.param("int4", 8, id="int4"), pytest.param("uint1", 2, id="uint1")],
)
def test_packed_out_requires_contiguity(dtype: str, distinct_values: int):
    """Sub-byte dtypes have no per-value byte stride, so only a packed `out` can be filled."""
    first = np.arange(8, dtype=np.uint8) % distinct_values

    packed_out = nk.zeros(first.shape, dtype=dtype)
    nk.add(first, 0, out=packed_out)
    assert_allclose(np.asarray(nk.astype(packed_out, "uint8")), first, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    rows = np.stack([first, first])
    with pytest.raises(ValueError, match="C-contiguous"):
        nk.add(rows, 0, out=nk.zeros((4, first.size), dtype=dtype)[::2])
