#!/usr/bin/env python3
"""Test batch distance operations: nk.dots_symmetric, nk.dots_packed, nk.cdist.

DTypes: float64, float32, float16, bfloat16, e4m3, e5m2, e2m3, e2m1, e3m2, int8, uint8, complex64, complex128.
Baselines: high-precision Decimal matrix multiplication, NumPy matmul.
Matches C++ suite: test/cross_*.cpp.

File: test/cross.py
Author: Ash Vardanian
Date: February 27, 2026
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

from base import (
    NATIVE_COMPUTE_DTYPE,
    NUMKONG_ATOL,
    NUMKONG_RTOL,
    PACKING_GRANULARITY,
    SETTINGS,
    assert_allclose,
    collect_errors,
    create_stats,
    keep_one_capability,
    make_nk,
    make_random,
    numpy_available,
    possible_capabilities,
    precise_decimal,
    print_stats_report,
    round_up_to,
    scipy_available,
    timed_call,
    tolerances_for_dtype,
)

import numkong as nk


try:
    import scipy.spatial.distance as spd
except ImportError:
    spd = None  # type: ignore[assignment]

stats = create_stats()
atexit.register(print_stats_report, stats)


def baseline_dots_symmetric(vectors, dtype=None):
    return vectors @ vectors.T


def baseline_dots_packed(left, right, dtype=None):
    return left @ right.T


def precise_matmul(left, right_transposed, dtype=None):
    """High-precision left @ right_transposedᵀ via Decimal. Returns 2D numpy array."""
    with precise_decimal(dtype) as (upcast, _sqrt, _ln):
        rows, _depth = left.shape
        cols = right_transposed.shape[0]
        result = np.empty((rows, cols), dtype=np.float64)
        right_rows = [[upcast(x) for x in right_transposed[col]] for col in range(cols)]
        for row in range(rows):
            left_values = [upcast(x) for x in left[row]]
            for col in range(cols):
                result[row, col] = float(
                    sum(left_value * right_value for left_value, right_value in zip(left_values, right_rows[col]))
                )
        return result


def precise_dots_symmetric(vectors, dtype=None):
    """High-precision vectors @ vectors.T via Decimal."""
    return precise_matmul(vectors, vectors, dtype=dtype)


def precise_dots_packed(left, right, dtype=None):
    """High-precision left @ right.T via Decimal."""
    return precise_matmul(left, right, dtype=dtype)


KERNELS_CROSS: dict[str, tuple[Callable | None, Callable, Callable]] = {
    "dots_symmetric": (baseline_dots_symmetric, nk.dots_symmetric, precise_dots_symmetric),
    "dots_packed": (baseline_dots_packed, nk.dots_packed, precise_dots_packed),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", SETTINGS.dims)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_batch_sqeuclidean_broadcasting(ndim: int, dtype: str, capability: str, np_rng: np.random.Generator):
    """Batch sqeuclidean with NxD-vs-NxD, NxD-vs-1xD, strided, transposed, and out_dtype scenarios."""
    keep_one_capability(capability)

    # NxD vs NxD
    a_matrix, _ = make_random((10, ndim), dtype, np_rng)
    b_matrix, _ = make_random((10, ndim), dtype, np_rng)
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # NxD vs 1xD
    b_matrix, _ = make_random((1, ndim), dtype, np_rng)
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix[0]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # 1xD vs NxD
    a_matrix, _ = make_random((1, ndim), dtype, np_rng)
    b_matrix, _ = make_random((10, ndim), dtype, np_rng)
    expected_distances = [spd.sqeuclidean(a_matrix[0], b_matrix[i]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # NxD vs D (1D)
    a_matrix, _ = make_random((10, ndim), dtype, np_rng)
    b_matrix, _ = make_random((ndim), dtype, np_rng)
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # D (1D) vs NxD
    b_matrix, _ = make_random((10, ndim), dtype, np_rng)
    a_matrix, _ = make_random((ndim), dtype, np_rng)
    expected_distances = [spd.sqeuclidean(b_matrix[i], a_matrix) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(b_matrix, a_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # Strided slices of bigger matrices
    a_matrix_extended, _ = make_random((10, ndim + 11), dtype, np_rng)
    b_matrix_extended, _ = make_random((10, ndim + 13), dtype, np_rng)
    a_matrix = a_matrix_extended[:, 1 : 1 + ndim]
    b_matrix = b_matrix_extended[:, 3 : 3 + ndim]
    assert a_matrix.base is a_matrix_extended and b_matrix.base is b_matrix_extended
    assert a_matrix.__array_interface__["strides"] is not None and b_matrix.__array_interface__["strides"] is not None
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # Transposed matrix
    a_matrix, _ = make_random((10, ndim), dtype, np_rng)
    b_matrix = np.ascontiguousarray(make_random((ndim, 10), dtype, np_rng)[0].T)
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)

    # Different output type
    a_matrix, _ = make_random((10, ndim), dtype, np_rng)
    b_matrix, _ = make_random((10, ndim), dtype, np_rng)
    expected_distances = np.array([spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]).astype(np.float32)
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix, out_dtype="float32"))
    assert_allclose(simd_distances, expected_distances, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)
    assert simd_distances.dtype == expected_distances.dtype

    # Supplied output buffer
    a_matrix, _ = make_random((10, ndim), dtype, np_rng)
    b_matrix, _ = make_random((10, ndim), dtype, np_rng)
    expected_distances = np.array([spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]).astype(np.float32)
    output_buffer = np.zeros(10, dtype=np.float32)
    assert nk.sqeuclidean(a_matrix, b_matrix, out=output_buffer) is None
    assert_allclose(output_buffer, expected_distances, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)
    assert output_buffer.dtype == expected_distances.dtype


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(SETTINGS.reduced_repetitions)
@pytest.mark.parametrize("num_vectors", SETTINGS.dims_height)
@pytest.mark.parametrize("vector_depth", SETTINGS.dims_depth)
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
        "e2m1",
        "int8",
        "uint8",
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dots_symmetric(num_vectors: int, vector_depth: int, dtype: str, capability: str, np_rng: np.random.Generator):
    """Test nk.dots_symmetric against high-precision matmul (upper triangle)."""

    vector_depth = round_up_to(vector_depth, PACKING_GRANULARITY.get(dtype, 1))
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_CROSS["dots_symmetric"]
    atol, rtol = tolerances_for_dtype(dtype)
    vectors_raw, vectors_baseline = make_random((num_vectors, vector_depth), dtype, np_rng)

    keep_one_capability(capability)

    accurate_ns, accurate = timed_call(precise_kernel or baseline_kernel, vectors_baseline, dtype=dtype)

    native_dt = NATIVE_COMPUTE_DTYPE.get(dtype, np.float64)
    expected_ns, expected = timed_call(baseline_kernel, vectors_baseline.astype(native_dt))

    result_ns, result = timed_call(simd_kernel, vectors_raw, dtype=dtype)
    result = np.asarray(result)

    mask = np.triu(np.ones((num_vectors, num_vectors), dtype=bool))
    assert_allclose(result[mask], accurate[mask], atol=atol, rtol=rtol)

    # A packed Tensor counts logical dimensions and must match its raw-byte twin
    if dtype in PACKING_GRANULARITY:
        vectors_tensor = make_nk(vectors_raw, dtype)
        assert vectors_tensor.shape == (num_vectors, vector_depth)
        assert_allclose(np.asarray(simd_kernel(vectors_tensor))[mask], result[mask], atol=1e-10, rtol=1e-10)

    # out= must match the allocated result (upper triangle)
    out_dtype = str(result.dtype)  # kernel output dtype depends on input
    out = nk.zeros((num_vectors, num_vectors), dtype=out_dtype)
    simd_kernel(vectors_raw, dtype=dtype, out=out)
    assert_allclose(np.asarray(out)[mask], result[mask], atol=1e-10, rtol=1e-10)

    collect_errors(
        "dots_symmetric",
        num_vectors * vector_depth,
        dtype,
        accurate[mask],
        accurate_ns,
        expected[mask],
        expected_ns,
        result[mask],
        result_ns,
        stats,
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_hammings_symmetric(capability: str, np_rng: np.random.Generator):
    """Test nk.hammings_symmetric against pairwise Hamming (upper triangle)."""
    num_vectors, bit_depth = 16, 128
    bits = np_rng.integers(2, size=(num_vectors, bit_depth)).astype(np.uint8)
    packed = np.packbits(bits, axis=1)

    keep_one_capability(capability)
    result = np.asarray(nk.hammings_symmetric(packed, dtype="uint1"))

    mask = np.triu(np.ones((num_vectors, num_vectors), dtype=bool))
    expected = np.zeros((num_vectors, num_vectors), dtype=np.float64)
    for i in range(num_vectors):
        for j in range(i, num_vectors):
            expected[i, j] = np.logical_xor(bits[i], bits[j]).sum()

    assert_allclose(result[mask], expected[mask], atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(SETTINGS.reduced_repetitions)
@pytest.mark.parametrize("rows", SETTINGS.dims_height)
@pytest.mark.parametrize("columns", SETTINGS.dims_width)
@pytest.mark.parametrize("depth", SETTINGS.dims_depth)
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
        "e2m1",
        "int8",
        "uint8",
    ],
)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dots_pack_and_packed(
    rows: int, columns: int, depth: int, dtype: str, capability: str, np_rng: np.random.Generator
):
    """Test dots_pack + dots_packed against high-precision matmul."""

    depth = round_up_to(depth, PACKING_GRANULARITY.get(dtype, 1))
    _, _, precise_kernel = KERNELS_CROSS["dots_packed"]
    atol, rtol = tolerances_for_dtype(dtype)
    a_raw, a_baseline = make_random((rows, depth), dtype, np_rng)
    b_raw, b_baseline = make_random((columns, depth), dtype, np_rng)

    keep_one_capability(capability)

    # SIMD path — wrap in nk.Tensor so dots_packed can infer dtype; packed Tensors count logical dimensions
    a_tensor, b_tensor = make_nk(a_raw, dtype), make_nk(b_raw, dtype)
    assert a_tensor.shape == (rows, depth) and b_tensor.shape == (columns, depth)
    b_packed = nk.dots_pack(b_tensor, dtype=dtype)
    result_ns, result = timed_call(nk.dots_packed, a_tensor, b_packed)
    result = np.asarray(result)

    accurate_ns, accurate = timed_call(precise_kernel, a_baseline, b_baseline, dtype=dtype)

    native_dt = NATIVE_COMPUTE_DTYPE.get(dtype, np.float64)
    expected_ns, expected = timed_call(baseline_dots_packed, a_baseline.astype(native_dt), b_baseline.astype(native_dt))

    assert_allclose(result, accurate, atol=atol, rtol=rtol)

    # out= must match the allocated result
    out_dtype = str(result.dtype)  # kernel output dtype depends on input
    out = nk.zeros((rows, columns), dtype=out_dtype)
    nk.dots_packed(a_tensor, b_packed, out=out)
    assert_allclose(np.asarray(out), result, atol=1e-10, rtol=1e-10)

    collect_errors(
        "dots_packed", rows * depth, dtype, accurate, accurate_ns, expected, expected_ns, result, result_ns, stats
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("numpy_dtype", ["float64", "float32", "float16"])
def test_dots_pack_infers_dtype(numpy_dtype, np_rng: np.random.Generator):
    """dots_pack() without explicit dtype should infer from the input array."""
    height, width, depth = 4, 8, 32
    a, _ = make_random((height, depth), numpy_dtype, np_rng)
    b, _ = make_random((width, depth), numpy_dtype, np_rng)

    packed = nk.dots_pack(b)  # no dtype= argument
    result = np.asarray(nk.dots_packed(a, packed))

    expected = a.astype(np.float64) @ b.astype(np.float64).T
    assert_allclose(result, expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dots_pack_matmul_operator(capability: str, np_rng: np.random.Generator):
    """Test the @ operator with a PackedMatrix (Tensor @ PackedMatrix)."""
    height, width, depth = 8, 16, 64
    a_matrix, _ = make_random((height, depth), "float32", np_rng)
    b_matrix, _ = make_random((width, depth), "float32", np_rng)

    keep_one_capability(capability)
    a_tensor = nk.zeros((height, depth), dtype="float32")
    a_tensor_view = np.asarray(a_tensor)
    np.copyto(a_tensor_view, a_matrix)

    b_packed = nk.dots_pack(b_matrix, dtype="float32")
    result = np.asarray(a_tensor @ b_packed)
    expected = a_matrix @ b_matrix.T

    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_hammings_pack_and_packed(capability: str, np_rng: np.random.Generator):
    """Test hammings_pack + hammings_packed against pairwise Hamming."""
    num_rows_a, num_rows_b, bit_depth = 8, 16, 128
    a_bits = np_rng.integers(2, size=(num_rows_a, bit_depth)).astype(np.uint8)
    b_bits = np_rng.integers(2, size=(num_rows_b, bit_depth)).astype(np.uint8)
    a_packed = np.packbits(a_bits, axis=1)
    b_packed_raw = np.packbits(b_bits, axis=1)

    keep_one_capability(capability)
    b_packed = nk.hammings_pack(b_packed_raw, dtype="uint1")
    result = np.asarray(nk.hammings_packed(a_packed, b_packed))

    expected = np.zeros((num_rows_a, num_rows_b), dtype=np.float64)
    for i in range(num_rows_a):
        for j in range(num_rows_b):
            expected[i, j] = np.logical_xor(a_bits[i], b_bits[j]).sum()

    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("metric", ["angular", "euclidean"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_spatials_pack_and_packed(metric: str, capability: str, np_rng: np.random.Generator):
    """Test dots_pack + angulars/euclideans_packed against SciPy cdist."""
    num_rows_a, num_rows_b, depth = 8, 16, 64
    a, _ = make_random((num_rows_a, depth), "float32", np_rng)
    b, _ = make_random((num_rows_b, depth), "float32", np_rng)

    keep_one_capability(capability)
    b_packed = nk.dots_pack(b, dtype="float32")
    if metric == "angular":
        result = np.asarray(nk.angulars_packed(a, b_packed))
        expected = spd.cdist(a, b, "cosine")
    else:
        result = np.asarray(nk.euclideans_packed(a, b_packed))
        expected = spd.cdist(a, b, "euclidean")

    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("metric", ["angular", "euclidean"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_spatials_symmetric(metric: str, capability: str, np_rng: np.random.Generator):
    """Test angulars/euclideans_symmetric against SciPy cdist (upper triangle)."""
    num_rows, depth = 16, 64
    vectors, _ = make_random((num_rows, depth), "float32", np_rng)

    keep_one_capability(capability)
    if metric == "angular":
        result = np.asarray(nk.angulars_symmetric(vectors, dtype="float32"))
        expected = spd.cdist(vectors, vectors, "cosine")
    else:
        result = np.asarray(nk.euclideans_symmetric(vectors, dtype="float32"))
        expected = spd.cdist(vectors, vectors, "euclidean")

    mask = np.triu(np.ones((num_rows, num_rows), dtype=bool))
    assert_allclose(result[mask], expected[mask], atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_jaccards_pack_and_packed(capability: str, np_rng: np.random.Generator):
    """Test hammings_pack + jaccards_packed against SciPy cdist."""
    num_rows_a, num_rows_b, bit_depth = 8, 16, 128
    a_bits = np_rng.integers(2, size=(num_rows_a, bit_depth)).astype(np.uint8)
    b_bits = np_rng.integers(2, size=(num_rows_b, bit_depth)).astype(np.uint8)
    a_packed = np.packbits(a_bits, axis=1)
    b_packed_raw = np.packbits(b_bits, axis=1)

    keep_one_capability(capability)
    b_packed = nk.hammings_pack(b_packed_raw, dtype="uint1")
    result = np.asarray(nk.jaccards_packed(a_packed, b_packed))
    expected = spd.cdist(a_bits, b_bits, "jaccard")

    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_jaccards_symmetric(capability: str, np_rng: np.random.Generator):
    """Test jaccards_symmetric against SciPy cdist (upper triangle)."""
    num_rows, bit_depth = 16, 128
    bits = np_rng.integers(2, size=(num_rows, bit_depth)).astype(np.uint8)
    packed = np.packbits(bits, axis=1)

    keep_one_capability(capability)
    result = np.asarray(nk.jaccards_symmetric(packed, dtype="uint1"))
    expected = spd.cdist(bits, bits, "jaccard")

    mask = np.triu(np.ones((num_rows, num_rows), dtype=bool))
    assert_allclose(result[mask], expected[mask], atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_packed_kind_validation(np_rng: np.random.Generator):
    """New packed APIs should enforce the expected packer family."""
    a_float, _ = make_random((4, 8), "float32", np_rng)
    b_float, _ = make_random((5, 8), "float32", np_rng)
    dots_packed = nk.dots_pack(b_float, dtype="float32")

    bits = np_rng.integers(2, size=(5, 64)).astype(np.uint8)
    hamming_packed = nk.hammings_pack(np.packbits(bits, axis=1), dtype="uint1")

    with pytest.raises(TypeError):
        nk.jaccards_packed(np.packbits(np_rng.integers(2, size=(4, 64)).astype(np.uint8), axis=1), dots_packed)
    with pytest.raises(TypeError):
        nk.angulars_packed(a_float, hamming_packed)
