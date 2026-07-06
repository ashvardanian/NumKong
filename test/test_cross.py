#!/usr/bin/env python3
"""Test batch distance operations: nk.dots_symmetric, nk.dots_packed, nk.cdist.

Dtypes: float64, float32, float16, bfloat16, e4m3, e5m2, e2m3, e3m2, int8, uint8, complex64, complex128.
Baselines: high-precision Decimal matrix multiplication, NumPy matmul.
Matches C++ suite: test_cross_*.cpp.
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
    assert_allclose,
    collect_errors,
    create_stats,
    dense_dimensions,
    keep_one_capability,
    make_nk,
    make_random,
    numpy_available,
    possible_capabilities,
    precise_decimal,
    print_stats_report,
    profile,
    randomized_repetitions_count,
    reduced_repetitions_count,
    scipy_available,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    test_depth_dimensions,
    test_height_dimensions,
    test_width_dimensions,
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
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_batch_sqeuclidean_broadcasting(ndim: int, dtype: str, capability: str):
    """Batch sqeuclidean with NxD-vs-NxD, NxD-vs-1xD, strided, transposed, and out_dtype scenarios."""
    keep_one_capability(capability)

    # NxD vs NxD
    a_matrix = np.random.randn(10, ndim).astype(dtype)
    b_matrix = np.random.randn(10, ndim).astype(dtype)
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # NxD vs 1xD
    b_matrix = np.random.randn(1, ndim).astype(dtype)
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix[0]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # 1xD vs NxD
    a_matrix = np.random.randn(1, ndim).astype(dtype)
    b_matrix = np.random.randn(10, ndim).astype(dtype)
    expected_distances = [spd.sqeuclidean(a_matrix[0], b_matrix[i]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # NxD vs D (1D)
    a_matrix = np.random.randn(10, ndim).astype(dtype)
    b_matrix = np.random.randn(ndim).astype(dtype)
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # D (1D) vs NxD
    b_matrix = np.random.randn(10, ndim).astype(dtype)
    a_matrix = np.random.randn(ndim).astype(dtype)
    expected_distances = [spd.sqeuclidean(b_matrix[i], a_matrix) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(b_matrix, a_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # Strided slices of bigger matrices
    a_matrix_extended = np.random.randn(10, ndim + 11).astype(dtype)
    b_matrix_extended = np.random.randn(10, ndim + 13).astype(dtype)
    a_matrix = a_matrix_extended[:, 1 : 1 + ndim]
    b_matrix = b_matrix_extended[:, 3 : 3 + ndim]
    assert a_matrix.base is a_matrix_extended and b_matrix.base is b_matrix_extended
    assert a_matrix.__array_interface__["strides"] is not None and b_matrix.__array_interface__["strides"] is not None
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # Transposed matrix
    a_matrix = np.random.randn(10, ndim).astype(dtype)
    b_matrix = np.ascontiguousarray(np.random.randn(ndim, 10).astype(dtype).T)
    expected_distances = [spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix)).astype(np.float64)
    assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)

    # Different output type
    a_matrix = np.random.randn(10, ndim).astype(dtype)
    b_matrix = np.random.randn(10, ndim).astype(dtype)
    expected_distances = np.array([spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]).astype(np.float32)
    simd_distances = np.array(nk.sqeuclidean(a_matrix, b_matrix, out_dtype="float32"))
    assert_allclose(simd_distances, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)
    assert simd_distances.dtype == expected_distances.dtype

    # Supplied output buffer
    a_matrix = np.random.randn(10, ndim).astype(dtype)
    b_matrix = np.random.randn(10, ndim).astype(dtype)
    expected_distances = np.array([spd.sqeuclidean(a_matrix[i], b_matrix[i]) for i in range(10)]).astype(np.float32)
    output_buffer = np.zeros(10, dtype=np.float32)
    assert nk.sqeuclidean(a_matrix, b_matrix, out=output_buffer) is None
    assert_allclose(output_buffer, expected_distances, atol=NK_ATOL, rtol=NK_RTOL)
    assert output_buffer.dtype == expected_distances.dtype


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(reduced_repetitions_count)
@pytest.mark.parametrize("num_vectors", test_height_dimensions)
@pytest.mark.parametrize("vector_depth", test_depth_dimensions)
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
def test_dots_symmetric(num_vectors: int, vector_depth: int, dtype: str, capability: str):
    """Test nk.dots_symmetric against high-precision matmul (upper triangle)."""

    baseline_kernel, simd_kernel, precise_kernel = KERNELS_CROSS["dots_symmetric"]
    atol, rtol = tolerances_for_dtype(dtype)
    vectors_raw, vectors_baseline = make_random((num_vectors, vector_depth), dtype)

    keep_one_capability(capability)

    accurate_dt, accurate = profile(precise_kernel or baseline_kernel, vectors_baseline, dtype=dtype)

    native_dt = NATIVE_COMPUTE_DTYPE.get(dtype, np.float64)
    expected_dt, expected = profile(baseline_kernel, vectors_baseline.astype(native_dt))

    result_dt, result = profile(simd_kernel, vectors_raw, dtype=dtype)
    result = np.asarray(result)

    mask = np.triu(np.ones((num_vectors, num_vectors), dtype=bool))
    assert_allclose(result[mask], accurate[mask], atol=atol, rtol=rtol)

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
        accurate_dt,
        expected[mask],
        expected_dt,
        result[mask],
        result_dt,
        stats,
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_hammings_symmetric(capability: str):
    """Test nk.hammings_symmetric against pairwise Hamming (upper triangle)."""
    num_vectors, bit_depth = 16, 128
    bits = np.random.randint(2, size=(num_vectors, bit_depth)).astype(np.uint8)
    packed = np.packbits(bits, axis=1)

    keep_one_capability(capability)
    result = np.asarray(nk.hammings_symmetric(packed, dtype="uint1"))

    mask = np.triu(np.ones((num_vectors, num_vectors), dtype=bool))
    expected = np.zeros((num_vectors, num_vectors), dtype=np.float64)
    for i in range(num_vectors):
        for j in range(i, num_vectors):
            expected[i, j] = np.logical_xor(bits[i], bits[j]).sum()

    assert_allclose(result[mask], expected[mask], atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(reduced_repetitions_count)
@pytest.mark.parametrize("rows", test_height_dimensions)
@pytest.mark.parametrize("columns", test_width_dimensions)
@pytest.mark.parametrize("depth", test_depth_dimensions)
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
def test_dots_pack_and_packed(rows: int, columns: int, depth: int, dtype: str, capability: str):
    """Test dots_pack + dots_packed against high-precision matmul."""

    _, _, precise_kernel = KERNELS_CROSS["dots_packed"]
    atol, rtol = tolerances_for_dtype(dtype)
    a_raw, a_baseline = make_random((rows, depth), dtype)
    b_raw, b_baseline = make_random((columns, depth), dtype)

    keep_one_capability(capability)

    # SIMD path — wrap in nk.Tensor so dots_packed can infer dtype
    a_tensor = make_nk(a_raw, dtype)
    b_tensor = make_nk(b_raw, dtype)
    b_packed = nk.dots_pack(b_tensor, dtype=dtype)
    result_dt, result = profile(nk.dots_packed, a_tensor, b_packed)
    result = np.asarray(result)

    accurate_dt, accurate = profile(precise_kernel, a_baseline, b_baseline, dtype=dtype)

    native_dt = NATIVE_COMPUTE_DTYPE.get(dtype, np.float64)
    expected_dt, expected = profile(baseline_dots_packed, a_baseline.astype(native_dt), b_baseline.astype(native_dt))

    assert_allclose(result, accurate, atol=atol, rtol=rtol)

    # out= must match the allocated result
    out_dtype = str(result.dtype)  # kernel output dtype depends on input
    out = nk.zeros((rows, columns), dtype=out_dtype)
    nk.dots_packed(a_tensor, b_packed, out=out)
    assert_allclose(np.asarray(out), result, atol=1e-10, rtol=1e-10)

    collect_errors(
        "dots_packed", rows * depth, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("numpy_dtype", ["float16", "float32", "float64"])
def test_dots_pack_infers_dtype(numpy_dtype):
    """dots_pack() without explicit dtype should infer from the input array."""
    height, width, depth = 4, 8, 32
    a = np.random.randn(height, depth).astype(numpy_dtype)
    b = np.random.randn(width, depth).astype(numpy_dtype)

    packed = nk.dots_pack(b)  # no dtype= argument
    result = np.asarray(nk.dots_packed(a, packed))

    expected = a.astype(np.float64) @ b.astype(np.float64).T
    assert_allclose(result, expected)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_dots_pack_matmul_operator(capability: str):
    """Test the @ operator with a PackedMatrix (Tensor @ PackedMatrix)."""
    height, width, depth = 8, 16, 64
    a_matrix = np.random.randn(height, depth).astype(np.float32)
    b_matrix = np.random.randn(width, depth).astype(np.float32)

    keep_one_capability(capability)
    a_tensor = nk.zeros((height, depth), dtype="float32")
    a_tensor_view = np.asarray(a_tensor)
    np.copyto(a_tensor_view, a_matrix)

    b_packed = nk.dots_pack(b_matrix, dtype="float32")
    result = np.asarray(a_tensor @ b_packed)
    expected = a_matrix @ b_matrix.T

    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_hammings_pack_and_packed(capability: str):
    """Test hammings_pack + hammings_packed against pairwise Hamming."""
    num_rows_a, num_rows_b, bit_depth = 8, 16, 128
    a_bits = np.random.randint(2, size=(num_rows_a, bit_depth)).astype(np.uint8)
    b_bits = np.random.randint(2, size=(num_rows_b, bit_depth)).astype(np.uint8)
    a_packed = np.packbits(a_bits, axis=1)
    b_packed_raw = np.packbits(b_bits, axis=1)

    keep_one_capability(capability)
    b_packed = nk.hammings_pack(b_packed_raw, dtype="uint1")
    result = np.asarray(nk.hammings_packed(a_packed, b_packed))

    expected = np.zeros((num_rows_a, num_rows_b), dtype=np.float64)
    for i in range(num_rows_a):
        for j in range(num_rows_b):
            expected[i, j] = np.logical_xor(a_bits[i], b_bits[j]).sum()

    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("metric", ["angular", "euclidean"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_spatials_pack_and_packed(metric: str, capability: str):
    """Test dots_pack + angulars/euclideans_packed against SciPy cdist."""
    num_rows_a, num_rows_b, depth = 8, 16, 64
    a = np.random.randn(num_rows_a, depth).astype(np.float32)
    b = np.random.randn(num_rows_b, depth).astype(np.float32)

    keep_one_capability(capability)
    b_packed = nk.dots_pack(b, dtype="float32")
    if metric == "angular":
        result = np.asarray(nk.angulars_packed(a, b_packed))
        expected = spd.cdist(a, b, "cosine")
    else:
        result = np.asarray(nk.euclideans_packed(a, b_packed))
        expected = spd.cdist(a, b, "euclidean")

    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("metric", ["angular", "euclidean"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_spatials_symmetric(metric: str, capability: str):
    """Test angulars/euclideans_symmetric against SciPy cdist (upper triangle)."""
    num_rows, depth = 16, 64
    vectors = np.random.randn(num_rows, depth).astype(np.float32)

    keep_one_capability(capability)
    if metric == "angular":
        result = np.asarray(nk.angulars_symmetric(vectors, dtype="float32"))
        expected = spd.cdist(vectors, vectors, "cosine")
    else:
        result = np.asarray(nk.euclideans_symmetric(vectors, dtype="float32"))
        expected = spd.cdist(vectors, vectors, "euclidean")

    mask = np.triu(np.ones((num_rows, num_rows), dtype=bool))
    assert_allclose(result[mask], expected[mask], atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_jaccards_pack_and_packed(capability: str):
    """Test hammings_pack + jaccards_packed against SciPy cdist."""
    num_rows_a, num_rows_b, bit_depth = 8, 16, 128
    a_bits = np.random.randint(2, size=(num_rows_a, bit_depth)).astype(np.uint8)
    b_bits = np.random.randint(2, size=(num_rows_b, bit_depth)).astype(np.uint8)
    a_packed = np.packbits(a_bits, axis=1)
    b_packed_raw = np.packbits(b_bits, axis=1)

    keep_one_capability(capability)
    b_packed = nk.hammings_pack(b_packed_raw, dtype="uint1")
    result = np.asarray(nk.jaccards_packed(a_packed, b_packed))
    expected = spd.cdist(a_bits, b_bits, "jaccard")

    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("capability", possible_capabilities)
def test_jaccards_symmetric(capability: str):
    """Test jaccards_symmetric against SciPy cdist (upper triangle)."""
    num_rows, bit_depth = 16, 128
    bits = np.random.randint(2, size=(num_rows, bit_depth)).astype(np.uint8)
    packed = np.packbits(bits, axis=1)

    keep_one_capability(capability)
    result = np.asarray(nk.jaccards_symmetric(packed, dtype="uint1"))
    expected = spd.cdist(bits, bits, "jaccard")

    mask = np.triu(np.ones((num_rows, num_rows), dtype=bool))
    assert_allclose(result[mask], expected[mask], atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
def test_packed_kind_validation():
    """New packed APIs should enforce the expected packer family."""
    a_float = np.random.randn(4, 8).astype(np.float32)
    b_float = np.random.randn(5, 8).astype(np.float32)
    dots_packed = nk.dots_pack(b_float, dtype="float32")

    bits = np.random.randint(2, size=(5, 64)).astype(np.uint8)
    hamming_packed = nk.hammings_pack(np.packbits(bits, axis=1), dtype="uint1")

    with pytest.raises(TypeError):
        nk.jaccards_packed(np.packbits(np.random.randint(2, size=(4, 64)).astype(np.uint8), axis=1), dots_packed)
    with pytest.raises(TypeError):
        nk.angulars_packed(a_float, hamming_packed)
