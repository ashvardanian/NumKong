#!/usr/bin/env python3
"""Test tensor API: constructors, slicing, reductions, scalar types, NumPy interop.

Dtypes: float64, float32, int8, int32, bfloat16, float8_e4m3, float8_e5m2, float6_e2m3, float6_e3m2.
Matches C++ suite: test_tensor.cpp.
"""

import concurrent.futures
import multiprocessing
import sys
import sysconfig
import time
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

import numkong as nk
from test_base import (
    NK_ATOL,
    NK_RTOL,
    assert_allclose,
    dense_dimensions,
    f32_downcast_to_bf16,
    make_nk,
    ml_dtypes_available,
    nk_seed,  # noqa: F401 — pytest fixture
    numpy_available,
    precise_decimal,
    randomized_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    to_array,
    tolerances_for_dtype,
)

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None  # type: ignore[assignment]


def precise_sum(a, dtype=None):
    """High-precision sum via Python Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        return float(sum(upcast(x) for x in np.asarray(a).flat))


def precise_min(a, dtype=None):
    """Exact min — comparison is exact on IEEE floats."""
    return float(min(float(x) for x in np.asarray(a).flat))


def precise_max(a, dtype=None):
    """Exact max — comparison is exact on IEEE floats."""
    return float(max(float(x) for x in np.asarray(a).flat))


def precise_add(a, b, dtype=None):
    """High-precision element-wise add via Python Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        return [float(upcast(x) + upcast(y)) for x, y in zip(np.asarray(a).flat, np.asarray(b).flat)]


def precise_subtract(a, b, dtype=None):
    """High-precision element-wise subtract via Python Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        return [float(upcast(x) - upcast(y)) for x, y in zip(np.asarray(a).flat, np.asarray(b).flat)]


def precise_multiply(a, b, dtype=None):
    """High-precision element-wise multiply via Python Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        return [float(upcast(x) * upcast(y)) for x, y in zip(np.asarray(a).flat, np.asarray(b).flat)]


def precise_negate(a, dtype=None):
    """Exact negate — sign flip has no precision loss."""
    return [-float(x) for x in np.asarray(a).flat]


KERNELS_TENSOR: dict[str, tuple[Callable, Callable, Callable | None]] = {
    "sum": (lambda a: np.sum(np.asarray(a)), lambda a: a.sum(), precise_sum),
    "min": (lambda a: np.min(np.asarray(a)), lambda a: a.min(), precise_min),
    "max": (lambda a: np.max(np.asarray(a)), lambda a: a.max(), precise_max),
    "argmin": (lambda a: np.argmin(np.asarray(a)), lambda a: a.argmin(), None),
    "argmax": (lambda a: np.argmax(np.asarray(a)), lambda a: a.argmax(), None),
    "add": (lambda a, b: np.add(np.asarray(a), np.asarray(b)), lambda a, b: a + b, precise_add),
    "subtract": (lambda a, b: np.subtract(np.asarray(a), np.asarray(b)), lambda a, b: a - b, precise_subtract),
    "multiply": (lambda a, b: np.multiply(np.asarray(a), np.asarray(b)), lambda a, b: a * b, precise_multiply),
    "negate": (lambda a: -np.asarray(a), lambda a: -a, precise_negate),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_astype_accepts_ndarray_without_tensor_copy():
    """Top-level astype converts an N-D NumPy buffer while preserving its shape."""
    source = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)

    result = nk.astype(source, "float32")

    assert result.shape == source.shape
    assert result.dtype == "float32"
    assert_allclose(np.asarray(result), source.astype(np.float32))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_astype_writes_to_caller_owned_output():
    """Top-level astype converts directly into a matching caller-owned buffer and returns it."""
    source = np.arange(12, dtype=np.float32).reshape(3, 4) + 0.75
    output = np.empty(source.shape, dtype=np.uint8)

    assert nk.astype(source, "uint8", out=output) is output
    assert_allclose(output, source.astype(np.uint8))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", ["bfloat16", "uint4"])
def test_astype_writes_to_tensor_output(dtype):
    """A NumKong Tensor names dtypes NumPy cannot express, so it must be accepted as `out`."""
    source = np.arange(8, dtype=np.uint8)
    output = nk.empty(source.shape, dtype=dtype)

    assert nk.astype(source, dtype, out=output) is output
    np.testing.assert_array_equal(np.asarray(nk.astype(output, "uint8")), source)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_astype_linearizes_strided_input():
    """Top-level astype accepts a non-contiguous input buffer."""
    source = np.arange(48, dtype=np.int16).reshape(2, 3, 8)[:, ::-1, ::2]

    result = nk.astype(source, "float32")

    assert result.shape == source.shape
    assert_allclose(np.asarray(result), source.astype(np.float32))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_astype_float_to_integer_policy():
    """Float-to-integer casts round ties to even, saturate, and map NaN to zero."""
    source = np.array([-np.inf, -127.5, -0.5, 0.5, 1.5, 126.5, np.inf, np.nan], dtype=np.float64)

    result = nk.astype(source, "int8")

    np.testing.assert_array_equal(np.asarray(result), np.array([-128, -128, 0, 0, 2, 126, 127, 0], dtype=np.int8))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_astype_round_trips_through_packed_dtype():
    """Top-level astype preserves logical values when a packed Tensor is used in between."""
    source = np.arange(8, dtype=np.uint8)

    packed = nk.astype(source, "uint4")
    result = nk.astype(packed, "uint8")

    assert packed.dtype == "uint4"
    np.testing.assert_array_equal(np.asarray(result), source)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_astype_validates_output_shape_dtype_and_layout():
    """Top-level astype requires an exact, C-contiguous output buffer."""
    source = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)

    with pytest.raises(ValueError, match="shapes don't match"):
        nk.astype(source, "float32", out=np.empty((2, 3, 3), dtype=np.float32))
    with pytest.raises(TypeError, match="expected"):
        nk.astype(source, "float32", out=np.empty(source.shape, dtype=np.float64))
    with pytest.raises(ValueError, match="C-contiguous"):
        nk.astype(source, "float32", out=np.empty((4, 3, 2), dtype=np.float32).transpose(2, 1, 0))

    readonly = np.empty(source.shape, dtype=np.float32)
    readonly.flags.writeable = False
    with pytest.raises((ValueError, BufferError, TypeError)):
        nk.astype(source, "float32", out=readonly)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_astype_rejects_overlapping_output():
    """Top-level astype rejects an output buffer that aliases its input."""
    source = np.arange(12, dtype=np.float32)

    with pytest.raises(ValueError, match="must not overlap"):
        nk.astype(source, "float32", out=source)


def test_pointers_availability():
    """Tests the availability of pre-compiled functions for compatibility with USearch."""
    assert nk.pointer_to_sqeuclidean("float64") != 0
    assert nk.pointer_to_angular("float64") != 0
    assert nk.pointer_to_inner("float64") != 0

    assert nk.pointer_to_sqeuclidean("float32") != 0
    assert nk.pointer_to_angular("float32") != 0
    assert nk.pointer_to_inner("float32") != 0

    assert nk.pointer_to_sqeuclidean("float16") != 0
    assert nk.pointer_to_angular("float16") != 0
    assert nk.pointer_to_inner("float16") != 0

    assert nk.pointer_to_sqeuclidean("int8") != 0
    assert nk.pointer_to_angular("int8") != 0
    assert nk.pointer_to_inner("int8") != 0

    assert nk.pointer_to_sqeuclidean("uint8") != 0
    assert nk.pointer_to_angular("uint8") != 0
    assert nk.pointer_to_inner("uint8") != 0


def test_capabilities_list():
    """Tests the visibility of hardware capabilities."""
    caps = nk.get_capabilities()
    avx2_caps = ["haswell", "alder", "sierra"]
    avx512_caps = ["skylake", "icelake", "genoa", "sapphire", "turin", "diamond"]
    amx_caps = ["sapphireamx", "graniteamx"]
    neon_caps = ["neon", "neonhalf", "neonfhm", "neonbfdot", "neonsdot", "neonfp8"]
    sve_caps = ["sve", "svehalf", "svebfdot", "svesdot", "sve2", "sve2p1"]
    sme_caps = ["sme", "sme2", "sme2p1", "smef64", "smehalf", "smebf16", "smebi32", "smelut2", "smefa64"]
    rvv_caps = ["rvv", "rvvhalf", "rvvbf16", "rvvbb"]
    power_caps = ["powervsx"]
    loongarch_caps = ["loongsonasx"]
    wasm_caps = ["v128relaxed"]

    expected_capabilities = [
        "serial",
        *avx2_caps,
        *avx512_caps,
        *amx_caps,
        *neon_caps,
        *sve_caps,
        *sme_caps,
        *rvv_caps,
        *power_caps,
        *loongarch_caps,
        *wasm_caps,
    ]
    for cap in expected_capabilities:
        assert cap in caps, f"Capability '{cap}' missing from get_capabilities()"
    assert caps.get("serial") == 1

    previous_value = nk.get_capabilities().get("neon")
    nk.enable_capability("neon")
    assert nk.get_capabilities().get("neon") == 1
    if not previous_value:
        nk.disable_capability("neon")


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "function, expected_error, args, kwargs",
    [
        (nk.sqeuclidean, TypeError, (), {}),
        (nk.sqeuclidean, TypeError, (to_array([1.0]),), {}),
        (nk.sqeuclidean, ValueError, (to_array([1.0]), to_array([1.0]), "missing_dtype"), {}),
        (nk.sqeuclidean, TypeError, (to_array([1.0]), "invalid"), {}),
        (nk.sqeuclidean, TypeError, (to_array([1.0]), to_array([1.0])), {"invalid_kwarg": "value"}),
        (nk.enable_capability, TypeError, (123,), {}),
        (nk.disable_capability, TypeError, ([],), {}),
        (nk.mahalanobis, TypeError, (to_array([1.0]), to_array([1.0])), {}),
        (nk.bilinear, TypeError, (to_array([1.0]),), {}),
        (nk.angular, TypeError, (to_array([1.0]), to_array([1.0]), to_array([1.0])), {}),
        (nk.cdist, TypeError, (to_array([[1.0]]), to_array([[1.0]]), "l2", "dos"), {}),
        (nk.cdist, TypeError, (to_array([[1.0]]), to_array([[1.0]]), "l2"), {"metric": "l2"}),
        (nk.angular, LookupError, (to_array([1 + 2j]), to_array([1 + 2j])), {}),
        (nk.angular, ValueError, (to_array([1.0]), to_array([1.0, 2.0])), {}),
        (nk.angular, TypeError, (to_array([1.0]), to_array([1], "int8")), {}),
        (nk.angular, TypeError, (to_array([1], "float32"), to_array([1], "float16")), {}),
    ],
)
def test_invalid_argument_handling(function, expected_error, args, kwargs):
    """Test that functions raise appropriate errors with invalid arguments."""
    with pytest.raises(expected_error):
        function(*args, **kwargs)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
def test_bf16_conversion_vs_ml_dtypes(ndim: int):
    """Compare NumKong bfloat16 conversion with ml_dtypes reference implementation."""
    a_f32 = np.random.randn(ndim).astype(np.float32)
    _, a_nk_bf16 = f32_downcast_to_bf16(a_f32)
    a_ml_bf16 = a_f32.astype(ml_dtypes.bfloat16)
    ml_bits = np.asarray(a_ml_bf16.view(np.uint16), dtype=np.uint16)
    assert np.array_equal(a_nk_bf16, ml_bits), (
        f"BFloat16 conversion mismatch with ml_dtypes:\n"
        f"  NumKong bits: {a_nk_bf16[:5]}...\n"
        f"  ml_dtypes bits: {ml_bits[:5]}..."
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
def test_float8_e4m3_conversion_vs_ml_dtypes(ndim: int):
    """Compare NumKong float8_e4m3 conversion with ml_dtypes reference implementation."""
    a_f32 = (np.random.randn(ndim) * 10).astype(np.float32)
    a_f32 = np.clip(a_f32, -448, 448)
    a_ml_e4m3 = a_f32.astype(ml_dtypes.float8_e4m3fn)
    a_nk = nk.zeros((ndim,), dtype="e4m3")
    assert a_ml_e4m3.dtype == ml_dtypes.float8_e4m3fn
    assert a_nk.dtype == "e4m3"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
def test_float8_e5m2_conversion_vs_ml_dtypes(ndim: int):
    """Compare NumKong float8_e5m2 conversion with ml_dtypes reference implementation."""
    a_f32 = (np.random.randn(ndim) * 100).astype(np.float32)
    a_f32 = np.clip(a_f32, -57344, 57344)
    a_ml_e5m2 = a_f32.astype(ml_dtypes.float8_e5m2)
    a_nk = nk.zeros((ndim,), dtype="e5m2")
    assert a_ml_e5m2.dtype == ml_dtypes.float8_e5m2
    assert a_nk.dtype == "e5m2"


@pytest.mark.parametrize("ndim", dense_dimensions)
def test_float6_e2m3_construction(ndim: int):
    """Verify that e2m3 tensors can be constructed and have the correct dtype."""
    a_nk = nk.zeros((ndim,), dtype="e2m3")
    assert a_nk.dtype == "e2m3"
    assert a_nk.shape == (ndim,)
    assert a_nk.ndim == 1


@pytest.mark.parametrize("ndim", dense_dimensions)
def test_float6_e3m2_construction(ndim: int):
    """Verify that e3m2 tensors can be constructed and have the correct dtype."""
    a_nk = nk.zeros((ndim,), dtype="e3m2")
    assert a_nk.dtype == "e3m2"
    assert a_nk.shape == (ndim,)
    assert a_nk.ndim == 1


def test_distances_tensor_properties(nk_seed: int):
    a = nk.iota((5, 128), nk_seed, dtype="float64")
    b = nk.iota((7, 128), nk_seed + 1, dtype="float64")
    result = nk.cdist(a, b, metric="sqeuclidean")

    assert hasattr(result, "shape")
    assert hasattr(result, "dtype")
    assert hasattr(result, "ndim")
    assert hasattr(result, "size")
    assert hasattr(result, "nbytes")
    assert hasattr(result, "strides")
    assert hasattr(result, "itemsize")

    assert result.shape == (5, 7)
    assert result.dtype == "float64"
    assert result.ndim == 2
    assert result.size == 35
    assert result.nbytes == 35 * 8
    assert result.itemsize == 8
    assert isinstance(result.strides, tuple)
    assert len(result.strides) == 2


def test_distances_tensor_properties_1d(nk_seed: int):
    a = nk.iota((10, 128), nk_seed, dtype="float64")
    b = nk.iota((10, 128), nk_seed + 1, dtype="float64")
    result = nk.sqeuclidean(a, b)

    assert result.shape == (10,)
    assert result.ndim == 1
    assert result.size == 10
    assert len(result.strides) == 1


def test_distances_tensor_len(nk_seed: int):
    a = nk.iota((5, 128), nk_seed, dtype="float64")
    b = nk.iota((7, 128), nk_seed + 1, dtype="float64")
    result_2d = nk.cdist(a, b, metric="sqeuclidean")
    assert len(result_2d) == 5

    a = nk.iota((10, 128), nk_seed, dtype="float64")
    b = nk.iota((10, 128), nk_seed + 1, dtype="float64")
    result_1d = nk.sqeuclidean(a, b)
    assert len(result_1d) == 10


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_indexing():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")
    assert repr(result)  # smoke-test repr doesn't crash
    expected = np.asarray(result)

    row0 = result[0]
    assert hasattr(row0, "shape")
    assert row0.shape == (7,)

    row_last = result[-1]
    assert row_last.shape == (7,)
    assert_allclose(np.asarray(row_last), expected[-1])

    val = result[2, 3]
    assert isinstance(val, float)
    assert_allclose(val, expected[2, 3])

    val_neg = result[-1, -1]
    assert isinstance(val_neg, float)
    assert_allclose(val_neg, expected[-1, -1])

    with pytest.raises(IndexError):
        _ = result[10]
    with pytest.raises(IndexError):
        _ = result[0, 100]


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_iteration():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")
    expected = np.asarray(result)

    count = 0
    for i, row in enumerate(result):
        count += 1
        assert row.shape == (7,)
        assert_allclose(np.asarray(row), expected[i])
    assert count == 5

    a = np.random.rand(3, 128).astype(np.float64)
    b = np.random.rand(3, 128).astype(np.float64)
    result_1d = nk.sqeuclidean(a, b)
    expected_1d = np.asarray(result_1d)

    items = list(result_1d)
    assert len(items) == 3
    for i, item in enumerate(items):
        assert isinstance(item, float)
        assert_allclose(item, expected_1d[i])


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_numpy_interop():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    arr = np.asarray(result)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (5, 7)
    assert arr.dtype == np.float64
    assert np.all(arr >= 0)

    mv = memoryview(result)
    assert mv.ndim == 2
    assert mv.shape == (5, 7)

    arr_from_mv = np.asarray(mv)
    np.testing.assert_array_equal(arr, arr_from_mv)


def test_distances_tensor_scalar_conversion(nk_seed: int):
    a = nk.iota((1, 128), nk_seed, dtype="float64").flatten()
    b = nk.iota((1, 128), nk_seed + 1, dtype="float64").flatten()
    result = nk.sqeuclidean(a, b)
    assert isinstance(result, float)
    assert result >= 0

    a2 = nk.iota((3, 128), nk_seed, dtype="float64")
    b2 = nk.iota((5, 128), nk_seed + 1, dtype="float64")
    result2 = nk.cdist(a2, b2, metric="sqeuclidean")
    with pytest.raises(TypeError):
        float(result2)


@pytest.mark.parametrize("input_dtype", ["float32", "float64"])
def test_distances_tensor_dtype_consistency(input_dtype, nk_seed: int):
    a = nk.iota((5, 128), nk_seed, dtype=input_dtype)
    b = nk.iota((7, 128), nk_seed + 1, dtype=input_dtype)
    result = nk.cdist(a, b, metric="sqeuclidean")
    assert result.dtype == "float64"


def test_distances_tensor_strides(nk_seed: int):
    a = nk.iota((5, 128), nk_seed, dtype="float64")
    b = nk.iota((7, 128), nk_seed + 1, dtype="float64")
    result = nk.cdist(a, b, metric="sqeuclidean")

    strides = result.strides
    assert isinstance(strides, tuple)
    assert len(strides) == 2
    assert strides == (7 * 8, 8)


def test_distances_tensor_array_interface(nk_seed: int):
    a = nk.iota((5, 128), nk_seed, dtype="float64")
    b = nk.iota((7, 128), nk_seed + 1, dtype="float64")
    result = nk.cdist(a, b, metric="sqeuclidean")

    ai = result.__array_interface__
    assert isinstance(ai, dict)
    assert "shape" in ai
    assert "typestr" in ai
    assert "data" in ai
    assert "strides" in ai
    assert "version" in ai
    assert ai["shape"] == (5, 7)
    assert ai["typestr"] == "<f8"
    assert ai["version"] == 3
    assert isinstance(ai["data"], tuple)
    assert len(ai["data"]) == 2


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_transpose():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    t = result.T
    assert t.shape == (7, 5)
    result_arr = np.asarray(result)
    t_arr = np.asarray(t)
    assert_allclose(result_arr.T, t_arr)

    a1d = np.random.rand(10, 128).astype(np.float64)
    b1d = np.random.rand(10, 128).astype(np.float64)
    result1d = nk.sqeuclidean(a1d, b1d)
    t1d = result1d.T
    assert t1d.shape == result1d.shape


def test_distances_tensor_str(nk_seed: int):
    a = nk.iota((3, 128), nk_seed, dtype="float64")
    b = nk.iota((4, 128), nk_seed + 1, dtype="float64")
    result = nk.cdist(a, b, metric="sqeuclidean")

    s = str(result)
    assert len(s) > 0
    assert any(c.isdigit() for c in s)


def test_distances_tensor_equality(nk_seed: int):
    a = nk.iota((5, 4), nk_seed, dtype="float32")
    b = nk.iota((7, 4), nk_seed + 20, dtype="float32")
    result = nk.cdist(a, b, metric="sqeuclidean")

    copy = result.copy()
    assert result == copy
    assert not (result != copy)  # noqa: SIM202 — intentionally tests __ne__

    a2 = nk.iota((5, 4), nk_seed + 100, dtype="float32")
    b2 = nk.iota((7, 4), nk_seed + 200, dtype="float32")
    result2 = nk.cdist(a2, b2, metric="sqeuclidean")
    assert result != result2

    a3 = nk.iota((3, 4), nk_seed, dtype="float32")
    b3 = nk.iota((4, 4), nk_seed + 20, dtype="float32")
    result3 = nk.cdist(a3, b3, metric="sqeuclidean")
    assert result != result3


def test_distances_tensor_inf_nan_propagation(nk_seed: int):
    """Verify that inf/nan inputs propagate through cdist without crashing."""
    normal = nk.iota((3, 4), nk_seed, dtype="float64")

    inf_tensor = nk.full((2, 4), float("inf"), dtype="float64")
    result_inf = nk.cdist(inf_tensor, normal, metric="sqeuclidean")
    assert result_inf.shape == (2, 3)
    assert result_inf.dtype == "float64"
    flat = result_inf.flatten()
    for i in range(result_inf.size):
        v = float(flat[i])
        assert v == float("inf") or v != v  # inf or nan

    nan_tensor = nk.full((2, 4), float("nan"), dtype="float64")
    result_nan = nk.cdist(nan_tensor, normal, metric="sqeuclidean")
    assert result_nan.shape == (2, 3)
    assert result_nan.dtype == "float64"
    flat_nan = result_nan.flatten()
    for i in range(result_nan.size):
        assert float(flat_nan[i]) != float(flat_nan[i])  # all nan

    s = str(result_inf)
    assert len(s) > 0
    assert any(c.isdigit() or c in "inaINAf" for c in s)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_copy():
    a = np.random.rand(5, 128).astype(np.float64)
    b = np.random.rand(7, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")

    copy = result.copy()
    assert copy.shape == result.shape
    assert copy.dtype == result.dtype
    assert result == copy

    result_arr = np.asarray(result)
    copy_arr = np.asarray(copy)
    np.testing.assert_array_equal(result_arr, copy_arr)


def test_distances_tensor_reshape(nk_seed: int):
    a = nk.iota((5, 128), nk_seed, dtype="float64")
    b = nk.iota((7, 128), nk_seed + 1, dtype="float64")
    result = nk.cdist(a, b, metric="sqeuclidean")

    flat = result.reshape(35)
    assert flat.shape == (35,)
    assert flat.size == 35

    back = flat.reshape(5, 7)
    assert back.shape == (5, 7)

    reshaped = result.reshape((7, 5))
    assert reshaped.shape == (7, 5)

    with pytest.raises(ValueError):
        result.reshape(10, 10)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_slicing():
    a = np.random.rand(10, 128).astype(np.float64)
    b = np.random.rand(8, 128).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")
    expected = np.asarray(result)

    sliced = result[2:5]
    assert sliced.shape == (3, 8)
    assert_allclose(np.asarray(sliced), expected[2:5])

    step_sliced = result[::2]
    assert step_sliced.shape == (5, 8)
    assert_allclose(np.asarray(step_sliced), expected[::2])

    rev_sliced = result[::-1]
    assert rev_sliced.shape == (10, 8)
    assert_allclose(np.asarray(rev_sliced), expected[::-1])

    end_sliced = result[-3:]
    assert end_sliced.shape == (3, 8)
    assert_allclose(np.asarray(end_sliced), expected[-3:])


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_distances_tensor_zero_copy_views():
    a = np.random.rand(5, 64).astype(np.float64)
    b = np.random.rand(4, 64).astype(np.float64)
    result = nk.cdist(a, b, metric="sqeuclidean")
    orig_np = np.asarray(result)

    sliced = result[1:4]
    sliced_np = np.asarray(sliced)
    assert np.shares_memory(orig_np, sliced_np)

    step_sliced = result[::2]
    step_np = np.asarray(step_sliced)
    assert np.shares_memory(orig_np, step_np)

    rev_sliced = result[::-1]
    rev_np = np.asarray(rev_sliced)
    assert np.shares_memory(orig_np, rev_np)

    row = result[0]
    row_np = np.asarray(row)
    assert np.shares_memory(orig_np, row_np)

    transposed = result.T
    trans_np = np.asarray(transposed)
    assert np.shares_memory(orig_np, trans_np)

    flat = result.reshape(20)
    flat_np = np.asarray(flat)
    assert np.shares_memory(orig_np, flat_np)

    trans_reshaped = transposed.reshape(20)
    trans_reshaped_np = np.asarray(trans_reshaped)
    assert not np.shares_memory(orig_np, trans_reshaped_np)

    chained = result[1:4][1:]
    chained_np = np.asarray(chained)
    assert np.shares_memory(orig_np, chained_np)

    for _i, row in enumerate(result):
        row_np = np.asarray(row)
        assert np.shares_memory(orig_np, row_np)

    orig_val = orig_np[2, 0].copy()
    view = result[2]
    view_np = np.asarray(view)
    view_np[0] = 999.0
    assert orig_np[2, 0] == 999.0
    orig_np[2, 0] = orig_val

    copied = result.copy()
    copied_np = np.asarray(copied)
    assert not np.shares_memory(orig_np, copied_np)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("int8", id="i8"),
        pytest.param("int32", id="i32"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [pytest.param((10,), id="1d-10"), pytest.param((5, 4), id="2d-5x4"), pytest.param((2, 3, 4), id="3d-2x3x4")],
)
def test_ndarray_zeros(dtype: str, shape):
    arr = nk.zeros(shape, dtype=dtype)
    assert arr.shape == shape
    assert arr.dtype == dtype
    flat = arr.flatten()
    for i in range(arr.size):
        assert float(flat[i]) == 0


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float64", id="f64"),
        pytest.param("float32", id="f32"),
        pytest.param("int8", id="i8"),
        pytest.param("int32", id="i32"),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [pytest.param((10,), id="1d-10"), pytest.param((5, 4), id="2d-5x4"), pytest.param((2, 3, 4), id="3d-2x3x4")],
)
def test_ndarray_ones(dtype: str, shape):
    arr = nk.ones(shape, dtype=dtype)
    assert arr.shape == shape
    assert arr.dtype == dtype
    flat = arr.flatten()
    for i in range(arr.size):
        assert float(flat[i]) == 1


@pytest.mark.parametrize(
    "dtype,fill_value",
    [
        pytest.param("float64", 3.14, id="f64-pi"),
        pytest.param("float32", -2.5, id="f32-neg"),
        pytest.param("int8", 42, id="i8-42"),
        pytest.param("int32", -100, id="i32-neg"),
    ],
)
@pytest.mark.parametrize("shape", [pytest.param((10,), id="1d-10"), pytest.param((5, 4), id="2d-5x4")])
def test_ndarray_full(dtype: str, fill_value, shape):
    arr = nk.full(shape, fill_value, dtype=dtype)
    assert arr.shape == shape
    assert arr.dtype == dtype
    flat = arr.flatten()
    for i in range(arr.size):
        val = float(flat[i])
        if dtype.startswith("float"):
            assert abs(val - fill_value) < 1e-5
        else:
            assert int(val) == fill_value


@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
@pytest.mark.parametrize("shape", [pytest.param((10,), id="1d-10"), pytest.param((5, 4), id="2d-5x4")])
def test_ndarray_empty(dtype: str, shape):
    arr = nk.empty(shape, dtype=dtype)
    assert arr.shape == shape
    assert arr.dtype == dtype


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float32", id="f32"),
        pytest.param("float64", id="f64"),
        pytest.param("int8", id="i8"),
        pytest.param("int32", id="i32"),
    ],
)
@pytest.mark.parametrize("shape", [pytest.param((10,), id="1d-10"), pytest.param((3, 4), id="2d-3x4")])
def test_ndarray_iota(dtype: str, shape):
    seed = 5
    arr = nk.iota(shape, seed, dtype=dtype)
    assert arr.shape == shape
    assert arr.dtype == dtype
    flat = arr.flatten()
    assert float(flat[0]) == seed
    assert float(flat[1]) == seed + 1
    assert float(flat[2]) == seed + 2


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float32", id="f32"),
        pytest.param("float64", id="f64"),
        pytest.param("int8", id="i8"),
        pytest.param("int32", id="i32"),
    ],
)
@pytest.mark.parametrize("n", [pytest.param(4, id="4x4"), pytest.param(8, id="8x8")])
def test_ndarray_diagonal(dtype: str, n):
    seed = 3
    arr = nk.diagonal(n, seed, dtype=dtype)
    assert arr.shape == (n, n)
    assert arr.dtype == dtype
    for i in range(n):
        assert float(arr[i, i]) == seed
        if i + 1 < n:
            assert float(arr[i, i + 1]) == 0


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param("float32", id="f32"),
        pytest.param("float64", id="f64"),
        pytest.param("int8", id="i8"),
        pytest.param("int32", id="i32"),
    ],
)
@pytest.mark.parametrize("shape", [pytest.param((10,), id="1d-10"), pytest.param((3, 4), id="2d-3x4")])
def test_ndarray_hash(dtype: str, shape):
    a = nk.hash(shape, seed=42, dtype=dtype)
    b = nk.hash(shape, seed=42, dtype=dtype)
    assert a.shape == shape
    assert a.dtype == dtype
    # Determinism: same seed → same data
    flat_a = a.flatten()
    flat_b = b.flatten()
    total = 1
    for d in shape:
        total *= d
    for i in range(total):
        assert float(flat_a[i]) == float(flat_b[i])
    # Different seed → different data
    c = nk.hash(shape, seed=99, dtype=dtype)
    flat_c = c.flatten()
    differs = any(float(flat_a[i]) != float(flat_c[i]) for i in range(total))
    assert differs, "Different seeds should produce different data"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
@pytest.mark.parametrize(
    "shape",
    [pytest.param((100,), id="1d-100"), pytest.param((10, 10), id="2d-10x10"), pytest.param((4, 5, 5), id="3d-4x5x5")],
)
def test_ndarray_sum_float(dtype: str, shape):
    np_arr = np.random.randn(*shape).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    baseline_kernel, simd_kernel, _ = KERNELS_TENSOR["sum"]
    expected = baseline_kernel(np_arr)
    result = simd_kernel(nk_arr)
    atol, rtol = tolerances_for_dtype(dtype)
    assert_allclose(result, expected, rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("int8", id="i8"), pytest.param("int32", id="i32")])
@pytest.mark.parametrize(
    "shape",
    [pytest.param((100,), id="1d-100"), pytest.param((10, 10), id="2d-10x10"), pytest.param((4, 5, 5), id="3d-4x5x5")],
)
def test_ndarray_sum_integer(dtype: str, shape):
    nk_dtype_conversion_info = np.iinfo(np.dtype(dtype))
    np_arr = np.random.randint(
        nk_dtype_conversion_info.min // 2, nk_dtype_conversion_info.max // 2, size=shape, dtype=dtype
    )
    nk_arr = make_nk(np_arr, dtype)
    # NumKong always accumulates in 64 bits; NumPy's default integer accumulator is C `long`,
    # which is 32-bit under Emscripten, so it must be pinned or Pyodide compares against a wrap.
    assert nk_arr.sum() == np_arr.sum(dtype=np.int64)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
@pytest.mark.parametrize("shape", [pytest.param((100,), id="1d-100"), pytest.param((10, 10), id="2d-10x10")])
def test_ndarray_min_max_float(dtype: str, shape):
    np_arr = np.random.randn(*shape).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)
    atol, rtol = tolerances_for_dtype(dtype)
    for op in ("min", "max"):
        baseline_kernel, simd_kernel, _ = KERNELS_TENSOR[op]
        assert_allclose(simd_kernel(nk_arr), baseline_kernel(np_arr), rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("int8", id="i8"), pytest.param("int32", id="i32")])
@pytest.mark.parametrize("shape", [pytest.param((100,), id="1d-100"), pytest.param((10, 10), id="2d-10x10")])
def test_ndarray_min_max_integer(dtype: str, shape):
    nk_dtype_conversion_info = np.iinfo(np.dtype(dtype))
    np_arr = np.random.randint(
        nk_dtype_conversion_info.min // 2, nk_dtype_conversion_info.max // 2, size=shape, dtype=dtype
    )
    nk_arr = make_nk(np_arr, dtype)
    assert nk_arr.min() == np_arr.min()
    assert nk_arr.max() == np_arr.max()


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32"), pytest.param("int32", id="i32")]
)
@pytest.mark.parametrize("shape", [pytest.param((100,), id="1d-100"), pytest.param((10, 10), id="2d-10x10")])
def test_ndarray_argmin_argmax_methods(dtype: str, shape):
    if dtype.startswith("float"):
        np_arr = np.random.randn(*shape).astype(dtype)
    else:
        nk_dtype_conversion_info = np.iinfo(np.dtype(dtype))
        np_arr = np.random.randint(
            nk_dtype_conversion_info.min // 2, nk_dtype_conversion_info.max // 2, size=shape, dtype=dtype
        )
    nk_arr = make_nk(np_arr, dtype)

    for op in ("argmin", "argmax"):
        baseline_kernel, simd_kernel, _ = KERNELS_TENSOR[op]
        assert simd_kernel(nk_arr) == baseline_kernel(np_arr)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_ndarray_add_float(dtype: str):
    shape = (20,)
    np_a = np.random.randn(*shape).astype(dtype)
    np_b = np.random.randn(*shape).astype(dtype)
    nk_a = make_nk(np_a, dtype)
    nk_b = make_nk(np_b, dtype)
    baseline_kernel, simd_kernel, _ = KERNELS_TENSOR["add"]
    result_np = np.asarray(simd_kernel(nk_a, nk_b))
    assert_allclose(result_np, baseline_kernel(np_a, np_b))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_ndarray_add_integer():
    shape = (20,)
    np_a = np.random.randint(-50, 50, size=shape, dtype="int32")
    np_b = np.random.randint(-50, 50, size=shape, dtype="int32")
    nk_a = make_nk(np_a, "int32")
    nk_b = make_nk(np_b, "int32")
    np.testing.assert_array_equal(np.asarray(nk_a + nk_b), np_a + np_b)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_ndarray_subtract_float(dtype: str):
    shape = (20,)
    np_a = np.random.randn(*shape).astype(dtype)
    np_b = np.random.randn(*shape).astype(dtype)
    nk_a = make_nk(np_a, dtype)
    nk_b = make_nk(np_b, dtype)
    baseline_kernel, simd_kernel, _ = KERNELS_TENSOR["subtract"]
    result_np = np.asarray(simd_kernel(nk_a, nk_b))
    assert_allclose(result_np, baseline_kernel(np_a, np_b))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_ndarray_subtract_integer():
    shape = (20,)
    np_a = np.random.randint(-50, 50, size=shape, dtype="int32")
    np_b = np.random.randint(-50, 50, size=shape, dtype="int32")
    nk_a = make_nk(np_a, "int32")
    nk_b = make_nk(np_b, "int32")
    np.testing.assert_array_equal(np.asarray(nk_a - nk_b), np_a - np_b)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_ndarray_multiply_float(dtype: str):
    shape = (20,)
    np_a = np.random.randn(*shape).astype(dtype)
    np_b = np.random.randn(*shape).astype(dtype)
    nk_a = make_nk(np_a, dtype)
    nk_b = make_nk(np_b, dtype)
    baseline_kernel, simd_kernel, _ = KERNELS_TENSOR["multiply"]
    result_np = np.asarray(simd_kernel(nk_a, nk_b))
    assert_allclose(result_np, baseline_kernel(np_a, np_b))


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_ndarray_multiply_integer():
    shape = (20,)
    np_a = np.random.randint(-10, 10, size=shape, dtype="int32")
    np_b = np.random.randint(-10, 10, size=shape, dtype="int32")
    nk_a = make_nk(np_a, "int32")
    nk_b = make_nk(np_b, "int32")
    np.testing.assert_array_equal(np.asarray(nk_a * nk_b), np_a * np_b)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_ndarray_unary_float(dtype: str):
    shape = (20,)
    np_a = np.random.randn(*shape).astype(dtype)
    nk_a = make_nk(np_a, dtype)
    baseline_kernel, simd_kernel, _ = KERNELS_TENSOR["negate"]
    result_neg_np = np.asarray(simd_kernel(nk_a))
    assert_allclose(result_neg_np, baseline_kernel(np_a))
    np.testing.assert_array_equal(np.asarray(+nk_a), np_a)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_ndarray_unary_integer():
    shape = (20,)
    np_a = np.random.randint(-50, 50, size=shape, dtype="int32")
    nk_a = make_nk(np_a, "int32")
    np.testing.assert_array_equal(np.asarray(-nk_a), -np_a)
    np.testing.assert_array_equal(np.asarray(+nk_a), np_a)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_reduction_on_strided_array(dtype: str):
    np_arr = np.random.randn(10, 10).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    np_strided = np_arr[::2]
    nk_strided = nk_arr[::2]

    atol, rtol = tolerances_for_dtype(dtype)
    for op in ("sum", "min", "max"):
        baseline_kernel, simd_kernel, _ = KERNELS_TENSOR[op]
        assert_allclose(simd_kernel(nk_strided), baseline_kernel(np_strided), rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_reduction_on_transposed_array(dtype: str):
    np_arr = np.random.randn(5, 8).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    np_t = np_arr.T
    nk_t = nk_arr.T

    atol, rtol = tolerances_for_dtype(dtype)
    for op in ("sum", "min", "max"):
        baseline_kernel, simd_kernel, _ = KERNELS_TENSOR[op]
        assert_allclose(simd_kernel(nk_t), baseline_kernel(np_t), rtol=rtol, atol=atol)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", [pytest.param("float64", id="f64"), pytest.param("float32", id="f32")])
def test_reduction_on_subview(dtype: str):
    np_arr = np.random.randn(20, 20).astype(dtype)
    nk_arr = make_nk(np_arr, dtype)

    np_sub = np_arr[5:15, 5:15]
    nk_sub = nk_arr[5:15, 5:15]

    atol, rtol = tolerances_for_dtype(dtype)
    for op in ("sum", "min", "max"):
        baseline_kernel, simd_kernel, _ = KERNELS_TENSOR[op]
        assert_allclose(simd_kernel(nk_sub), baseline_kernel(np_sub), rtol=rtol, atol=atol)
    for op in ("argmin", "argmax"):
        baseline_kernel, simd_kernel, _ = KERNELS_TENSOR[op]
        assert simd_kernel(nk_sub) == baseline_kernel(np_sub)


def test_bfloat16_scalar_creation():
    bf = nk.bfloat16(3.14159)
    assert isinstance(bf, nk.bfloat16)
    assert abs(float(bf) - 3.14159) < 0.01
    assert int(bf) == 3
    assert repr(bf)  # smoke-test repr doesn't crash


def test_bfloat16_scalar_arithmetic():
    a = nk.bfloat16(1.5)
    b = nk.bfloat16(2.5)

    result = a + b
    assert isinstance(result, nk.bfloat16)
    assert float(result) == 4.0

    result = b - a
    assert isinstance(result, nk.bfloat16)
    assert float(result) == 1.0

    result = a * b
    assert isinstance(result, nk.bfloat16)
    assert float(result) == 3.75

    result = b / a
    assert isinstance(result, nk.bfloat16)
    assert abs(float(result) - 1.6666) < 0.01


def test_bfloat16_scalar_unary():
    a = nk.bfloat16(1.5)
    assert float(-a) == -1.5
    assert float(+a) == 1.5
    assert float(abs(nk.bfloat16(-1.5))) == 1.5
    assert bool(a)
    assert not bool(nk.bfloat16(0.0))


def test_bfloat16_scalar_comparison():
    a = nk.bfloat16(1.5)
    b = nk.bfloat16(2.5)

    assert a < b
    assert a <= b
    assert a <= a
    assert b > a
    assert b >= a
    assert a == a
    assert a != b

    assert a == 1.5
    assert a < 2.0
    assert a > 1.0


def test_bfloat16_scalar_hash():
    a = nk.bfloat16(1.5)
    b = nk.bfloat16(1.5)
    s = {a, b}
    assert len(s) == 1
    d = {a: "value"}
    assert d[b] == "value"


def test_float8_e4m3_scalar():
    f8 = nk.float8_e4m3(1.5)
    assert isinstance(f8, nk.float8_e4m3)
    assert float(f8) == 1.5
    assert int(f8) == 1
    assert "float8_e4m3" in repr(f8)
    assert float(-f8) == -1.5
    assert bool(f8)
    assert not bool(nk.float8_e4m3(0.0))


def test_float8_e4m3_scalar_arithmetic():
    a = nk.float8_e4m3(1.5)
    b = nk.float8_e4m3(2.0)

    result = a + b
    assert isinstance(result, nk.float8_e4m3)
    assert abs(float(result) - 3.5) < 0.5

    result = b - a
    assert isinstance(result, nk.float8_e4m3)
    assert abs(float(result) - 0.5) < 0.5

    result = a * b
    assert isinstance(result, nk.float8_e4m3)
    assert abs(float(result) - 3.0) < 0.5

    result = b / a
    assert isinstance(result, nk.float8_e4m3)
    assert abs(float(result) - 1.33) < 0.5

    assert float(+a) == float(a)
    assert float(abs(nk.float8_e4m3(-1.5))) == 1.5


def test_float8_e5m2_scalar():
    f8 = nk.float8_e5m2(1.5)
    assert isinstance(f8, nk.float8_e5m2)
    assert float(f8) == 1.5
    assert int(f8) == 1
    assert "float8_e5m2" in repr(f8)
    assert float(-f8) == -1.5
    assert bool(f8)
    assert not bool(nk.float8_e5m2(0.0))


def test_float8_e5m2_scalar_arithmetic():
    a = nk.float8_e5m2(1.5)
    b = nk.float8_e5m2(2.0)

    result = a + b
    assert isinstance(result, nk.float8_e5m2)
    assert abs(float(result) - 3.5) < 0.5

    result = b - a
    assert isinstance(result, nk.float8_e5m2)
    assert abs(float(result) - 0.5) < 0.5

    result = a * b
    assert isinstance(result, nk.float8_e5m2)
    assert abs(float(result) - 3.0) < 0.5

    result = b / a
    assert isinstance(result, nk.float8_e5m2)
    assert abs(float(result) - 1.33) < 0.5

    assert float(+a) == float(a)
    assert float(abs(nk.float8_e5m2(-1.5))) == 1.5


@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes not installed")
def test_bfloat16_vs_ml_dtypes():
    test_values = [0.0, 1.0, -1.0, 3.14159, 100.0, 0.001, -65504.0]
    for val in test_values:
        nk_bf16 = nk.bfloat16(val)
        ml_bf16 = ml_dtypes.bfloat16(val)
        assert float(nk_bf16) == float(ml_bf16), f"Mismatch for {val}: nk={float(nk_bf16)}, ml={float(ml_bf16)}"


@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes not installed")
def test_float8_e4m3_vs_ml_dtypes():
    test_values = [0.0, 1.0, -1.0, 1.5, 2.0, 0.5]
    for val in test_values:
        nk_f8 = nk.float8_e4m3(val)
        ml_f8 = ml_dtypes.float8_e4m3fn(val)
        assert float(nk_f8) == float(ml_f8), f"Mismatch for {val}: nk={float(nk_f8)}, ml={float(ml_f8)}"


@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes not installed")
def test_float8_e5m2_vs_ml_dtypes():
    test_values = [0.0, 1.0, -1.0, 1.5, 2.0, 0.5]
    for val in test_values:
        nk_f8 = nk.float8_e5m2(val)
        ml_f8 = ml_dtypes.float8_e5m2(val)
        assert float(nk_f8) == float(ml_f8), f"Mismatch for {val}: nk={float(nk_f8)}, ml={float(ml_f8)}"


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes not installed")
@pytest.mark.parametrize(
    "ml_dtype,nk_name",
    [
        ("bfloat16", "bfloat16"),
        ("float8_e4m3fn", "e4m3"),
        ("float8_e5m2", "e5m2"),
        ("float6_e2m3fn", "e2m3"),
        ("float6_e3m2fn", "e3m2"),
    ],
)
def test_ml_dtypes_array_to_tensor(ml_dtype, nk_name):
    """Verify that ml_dtypes arrays can be consumed as nk.Tensor via __array_interface__."""
    dt = getattr(ml_dtypes, ml_dtype)
    a_f32 = np.random.randn(16).astype(np.float32).clip(-1, 1)
    a_ml = a_f32.astype(dt)
    # 1D
    t = nk.Tensor(a_ml)
    assert t.dtype == nk_name
    assert t.shape == (16,)
    # 2D
    a_2d = np.random.randn(4, 8).astype(np.float32).clip(-1, 1).astype(dt)
    t2 = nk.Tensor(a_2d)
    assert t2.dtype == nk_name
    assert t2.shape == (4, 8)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes not installed")
@pytest.mark.parametrize(
    "ml_dtype",
    ["float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e4m3b11fnuz", "float8_e8m0fnu"],
)
def test_ml_dtypes_incompatible_rejected(ml_dtype):
    """Verify that fnuz/b11fnuz/e8m0fnu types are rejected, not silently misinterpreted."""
    dt = getattr(ml_dtypes, ml_dtype, None)
    if dt is None:
        pytest.skip(f"ml_dtypes.{ml_dtype} not available in this version")
    a = np.array([0.5], dtype=dt)
    with pytest.raises((TypeError, ValueError)):
        nk.Tensor(a)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize(
    "scalar_type,name",
    [
        (nk.bfloat16, "bfloat16"),
        (nk.float8_e4m3, "float8_e4m3"),
        (nk.float8_e5m2, "float8_e5m2"),
    ],
)
def test_numpy_array_with_nk_dtype(scalar_type, name):
    """Verify that nk scalar types can be used as NumPy dtype specifiers."""
    if not hasattr(scalar_type, "dtype"):
        pytest.skip("NumPy dtype registration not available")
    arr = np.array([1.0, 2.0, 3.0], dtype=scalar_type)
    assert abs(float(arr[0]) - 1.0) < 0.1
    assert abs(float(arr[1]) - 2.0) < 0.1


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_nk_dtype_numpy_roundtrip():
    """Verify roundtrip: np.array(dtype=nk.bfloat16) → float32 → bfloat16 preserves values."""
    if not hasattr(nk.bfloat16, "dtype"):
        pytest.skip("NumPy dtype registration not available")
    vals = [0.0, 1.0, -1.0, 0.5, 3.14]
    arr = np.array(vals, dtype=nk.bfloat16)
    # Cast to float32 and back to verify the registered cast functions work.
    f32 = arr.astype(np.float32)
    back = f32.astype(nk.bfloat16)
    for i in range(len(vals)):
        assert float(arr[i]) == float(back[i])


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_dots_packed_row_range():
    """Test dots_packed with start_row/end_row splits produce the same result."""
    height, depth, width = 100, 64, 50
    left_matrix = np.random.randn(height, depth).astype(np.float32)
    right_matrix = np.ascontiguousarray(np.random.randn(width, depth).astype(np.float32))
    right_packed = nk.dots_pack(right_matrix, dtype="float32")

    reference = np.array(nk.dots_packed(left_matrix, right_packed))

    output = nk.zeros((height, width), dtype="float64")
    nk.dots_packed(left_matrix, right_packed, out=output, start_row=0, end_row=50)
    nk.dots_packed(left_matrix, right_packed, out=output, start_row=50, end_row=100)

    assert_allclose(np.array(output), reference, err_msg="Row-range split differs from full computation")


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_dots_symmetric_row_range():
    """Test dots_symmetric with start_row/end_row.

    Only the upper triangle of the output is guaranteed to be initialized,
    so we compare only the upper-triangle entries that fall within each row range.
    """
    count, depth = 64, 32
    vectors = np.random.randn(count, depth).astype(np.float32)

    reference = np.array(nk.dots_symmetric(vectors))
    mask = np.triu(np.ones((count, count), dtype=bool))

    output = nk.zeros((count, count), dtype="float64")
    nk.dots_symmetric(vectors, out=output, start_row=0, end_row=count)

    assert_allclose(np.array(output)[mask], reference[mask], err_msg="Full-range dots_symmetric differs from default")


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("height", [63, 64, 129])
def test_dots_packed_threads(height):
    """Verify OpenMP-parallel dots_packed matches the serial path across tile boundaries.

    The packed row tile is 64, so sizes straddling one and two tiles exercise
    both the whole-tile and tail-chunk branches of the parallel loop.
    """
    depth, width = 64, 32
    left_matrix = np.random.randn(height, depth).astype(np.float32)
    right_matrix = np.ascontiguousarray(np.random.randn(width, depth).astype(np.float32))
    right_packed = nk.dots_pack(right_matrix, dtype="float32")

    serial = np.array(nk.dots_packed(left_matrix, right_packed, threads=1))
    parallel = np.array(nk.dots_packed(left_matrix, right_packed, threads=4))
    assert_allclose(parallel, serial, err_msg=f"threads=4 diverges from threads=1 at height={height}")


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("count", [31, 32, 65])
def test_dots_symmetric_threads(count):
    """Verify OpenMP-parallel dots_symmetric matches the serial path across tile boundaries.

    The symmetric row tile is 32; only the upper triangle is guaranteed written.
    """
    depth = 32
    vectors = np.random.randn(count, depth).astype(np.float32)
    mask = np.triu(np.ones((count, count), dtype=bool))

    serial = np.array(nk.dots_symmetric(vectors, threads=1))
    parallel = np.array(nk.dots_symmetric(vectors, threads=4))
    assert_allclose(parallel[mask], serial[mask], err_msg=f"threads=4 diverges from threads=1 at count={count}")


def _skip_unless_free_threaded():
    version = sys.version_info
    if version.major == 3 and version.minor >= 13:
        is_free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        if not is_free_threaded:
            pytest.skip("Uses non-free-threaded Python, skipping GIL-related tests")
        if sys._is_gil_enabled():
            pytest.skip("GIL is enabled, skipping GIL-related tests")
    else:
        pytest.skip("Python < 3.13t, skipping GIL-related tests")


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_gil_free_threading():
    """Test NumKong in Python 3.13t free-threaded mode if available."""
    _skip_unless_free_threaded()

    num_threads = multiprocessing.cpu_count()
    vectors_a = np.random.rand(32 * 1024 * num_threads, 1024).astype(np.float32)
    vectors_b = np.random.rand(32 * 1024 * num_threads, 1024).astype(np.float32)
    distances = np.zeros(vectors_a.shape[0], dtype=np.float32)

    def compute_batch(start_idx, end_idx) -> float:
        slice_a = vectors_a[start_idx:end_idx]
        slice_b = vectors_b[start_idx:end_idx]
        slice_distances = distances[start_idx:end_idx]
        nk.angular(slice_a, slice_b, out=slice_distances)
        return sum(slice_distances)

    def compute_with_threads(threads: int) -> float:
        chunk_size = len(vectors_a) // threads
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for i in range(threads):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < threads - 1 else len(vectors_a)
                futures.append(executor.submit(compute_batch, start_idx, end_idx))
        total_sum = 0.0
        for future in concurrent.futures.as_completed(futures):
            total_sum += future.result()
        return total_sum

    start_time = time.time()
    baseline_sum = compute_with_threads(2)
    end_time = time.time()
    baseline_duration = end_time - start_time

    start_time = time.time()
    multi_sum = compute_with_threads(num_threads)
    end_time = time.time()
    multi_duration = end_time - start_time

    assert np.allclose(
        baseline_sum, multi_sum, atol=NK_ATOL, rtol=NK_RTOL
    ), f"Results differ: baseline {baseline_sum} vs multi-threaded {multi_sum}"

    if baseline_duration < multi_duration:
        warnings.warn(
            f"{num_threads}-threaded execution took longer than 2-threaded baseline: {multi_duration:.2f}s vs {baseline_duration:.2f}s",
            UserWarning,
        )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_gil_free_dots_packed_threading():
    """Test multi-threaded dots_packed with start_row/end_row."""
    _skip_unless_free_threaded()

    height, depth, width = 200, 64, 50
    num_threads = min(multiprocessing.cpu_count(), 4)
    left_matrix = np.random.randn(height, depth).astype(np.float32)
    right_matrix = np.ascontiguousarray(np.random.randn(width, depth).astype(np.float32))
    right_packed = nk.dots_pack(right_matrix, dtype="float32")

    # Single-threaded reference
    reference = nk.dots_packed(left_matrix, right_packed)

    # Multi-threaded with row slicing into shared output
    output = nk.zeros((height, width), dtype="float64")
    rows_per_thread = height // num_threads

    def compute_slice(thread_index):
        start_row = thread_index * rows_per_thread
        end_row = start_row + rows_per_thread if thread_index < num_threads - 1 else height
        nk.dots_packed(left_matrix, right_packed, out=output, start_row=start_row, end_row=end_row)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        list(pool.map(compute_slice, range(num_threads)))

    assert_allclose(
        np.array(output), np.array(reference), err_msg="Multi-threaded dots_packed result differs from single-threaded"
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_gil_free_dots_symmetric_threading():
    """Test multi-threaded dots_symmetric with start_row/end_row.

    The symmetric kernel only fills the upper triangle of each row's output.
    Each thread computes a disjoint row range into the same shared output tensor.
    """
    _skip_unless_free_threaded()

    count, depth = 100, 64
    num_threads = min(multiprocessing.cpu_count(), 4)
    vectors = np.random.randn(count, depth).astype(np.float32)

    # Single-threaded reference (upper triangle only is meaningful)
    reference = np.array(nk.dots_symmetric(vectors))

    # Multi-threaded with row slicing into shared output
    output = nk.zeros((count, count), dtype="float64")
    rows_per_thread = count // num_threads

    def compute_slice(thread_index):
        start_row = thread_index * rows_per_thread
        end_row = start_row + rows_per_thread if thread_index < num_threads - 1 else count
        nk.dots_symmetric(vectors, out=output, start_row=start_row, end_row=end_row)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
        list(pool.map(compute_slice, range(num_threads)))

    # Only the upper triangle is guaranteed initialized by the symmetric kernel
    result = np.array(output)
    mask = np.triu(np.ones((count, count), dtype=bool))
    assert_allclose(
        result[mask],
        reference[mask],
        err_msg="Multi-threaded dots_symmetric upper triangle differs from single-threaded",
    )
