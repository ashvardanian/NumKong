#!/usr/bin/env python3
"""Test pairwise cross-distances: nk.cdist for all metric families.

Dtypes: float64, float32, float16, bfloat16, int8, uint8, sub-byte floats.
Baselines: SciPy cdist.
Matches C++ suite: test_cross_*.cpp.
"""

import atexit
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    import numpy as np  # static-analysis-only; the runtime try/except below is authoritative

try:
    import numpy as np

    numpy_available = True
except Exception:
    numpy_available = False

try:
    from scipy.stats import entropy as scipy_entropy
except ImportError:
    scipy_entropy = None  # type: ignore[assignment]

from test_base import (
    NK_ATOL,
    NK_RTOL,
    PACKING_GRANULARITY,
    assert_allclose,
    create_stats,
    dense_dimensions,
    keep_one_capability,
    make_random,
    ml_dtypes_available,
    nk_seed,  # noqa: F401 — pytest fixture
    numpy_available,
    possible_capabilities,
    print_stats_report,
    randomized_repetitions_count,
    round_up_to,
    scipy_available,
    scipy_metric_name,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
)

import numkong as nk


try:
    import scipy.spatial.distance as spd
except ImportError:
    spd = None  # type: ignore[assignment]

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None  # type: ignore[assignment]

stats = create_stats()
atexit.register(print_stats_report, stats)


def round_and_clip_even(values, out_dtype):
    nk_dtype_conversion_info = np.iinfo(out_dtype)
    finite_values = np.nan_to_num(
        values, nan=0.0, posinf=float(nk_dtype_conversion_info.max), neginf=float(nk_dtype_conversion_info.min)
    )
    clipped_values = np.clip(finite_values, nk_dtype_conversion_info.min, nk_dtype_conversion_info.max)
    return np.rint(clipped_values).astype(out_dtype)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["float64", "float32"])
@pytest.mark.parametrize("metric", ["dot", "angular", "euclidean"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_batch_metrics(ndim, input_dtype, metric, capability, nk_seed):
    """Verify the SIMD batch dispatch for dot, angular, and euclidean.

    Sets ``out_dtype`` equal to ``input_dtype`` (float64 or float32) so that the
    kernel's native output type matches and the fast packed/symmetric batch path
    is selected instead of the scalar pairwise fallback.  Uses asymmetric matrix
    sizes (7 x 11) to exercise the general rectangular case.

    Dimensions are inherited from ``NK_DENSE_DIMENSIONS``; capabilities from
    platform auto-detection via ``possible_capabilities``.  Baseline for ``dot``
    is ``np.dot`` (SciPy has no ``cdist`` metric for inner product); other
    metrics use ``scipy.spatial.distance.cdist``.
    """
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 7, 11
    a_matrix, _ = make_random((num_rows_a, ndim), input_dtype, seed=nk_seed)
    b_matrix, _ = make_random((num_rows_b, ndim), input_dtype, seed=nk_seed + 1)

    # Use native out_dtype to force batch path (float64->float64, float32->float32)
    out_dtype = input_dtype
    scipy_metric = scipy_metric_name(metric)

    if metric == "dot":
        expected = np.array(
            [[np.dot(a_matrix[i], b_matrix[j]) for j in range(num_rows_b)] for i in range(num_rows_a)]
        ).astype(out_dtype)
    else:
        expected = spd.cdist(a_matrix, b_matrix, scipy_metric).astype(out_dtype)

    result = nk.cdist(a_matrix, b_matrix, metric=metric, out_dtype=out_dtype)
    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["float64", "float32"])
@pytest.mark.parametrize("metric", ["dot", "angular", "euclidean", "sqeuclidean"])
def test_cdist_self_distance(ndim, input_dtype, metric, nk_seed):
    """Verify ``cdist(A, A)`` produces a complete, correct symmetric matrix.

    When both operands are the same object the C code takes a symmetric batch
    shortcut that only computes the upper triangle.  A post-kernel mirror loop
    must fill the lower triangle.  This test checks:

    1. Default ``out_dtype=float64`` (pairwise fallback on f32 input) — full
       matrix against SciPy.
    2. Native ``out_dtype=input_dtype`` (forces batch symmetric path) — full
       matrix, with an explicit lower-triangle mask assertion to catch missing
       mirror writes.

    No ``capability`` parameter: runs on whatever the default backend is (all
    ISA-specific paths are already covered by ``test_cdist_batch_metrics``).
    Dimensions from ``NK_DENSE_DIMENSIONS``.
    """
    a_matrix, _ = make_random((10, ndim), input_dtype, seed=nk_seed)

    scipy_metric = scipy_metric_name(metric)
    if metric == "dot":
        expected = np.array([[np.dot(a_matrix[i], a_matrix[j]) for j in range(10)] for i in range(10)]).astype(
            np.float64
        )
    else:
        expected = spd.cdist(a_matrix, a_matrix, scipy_metric)

    # Default out_dtype (f64) — may use pairwise fallback
    result_default = np.asarray(nk.cdist(a_matrix, a_matrix, metric=metric))
    assert_allclose(result_default, expected, atol=NK_ATOL, rtol=NK_RTOL)
    # Check lower triangle explicitly
    mask_lower = np.tril(np.ones((10, 10), dtype=bool), k=-1)
    assert_allclose(result_default[mask_lower], expected[mask_lower], atol=NK_ATOL, rtol=NK_RTOL)

    # Native out_dtype — should force batch symmetric path for f32
    native_out_dtype = input_dtype
    if metric == "dot":
        expected_native = np.array([[np.dot(a_matrix[i], a_matrix[j]) for j in range(10)] for i in range(10)]).astype(
            native_out_dtype
        )
    else:
        expected_native = spd.cdist(a_matrix, a_matrix, scipy_metric).astype(native_out_dtype)
    result_native = np.asarray(nk.cdist(a_matrix, a_matrix, metric=metric, out_dtype=native_out_dtype))
    assert_allclose(result_native, expected_native, atol=NK_ATOL, rtol=NK_RTOL)
    assert_allclose(result_native[mask_lower], expected_native[mask_lower], atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["float64", "float32", "float16"])
@pytest.mark.parametrize("out_dtype", [None, "float32", "int32"])
@pytest.mark.parametrize("metric", ["angular", "sqeuclidean", "euclidean", "dot"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_float_accuracy(ndim, input_dtype, out_dtype, metric, capability, nk_seed):
    """Broad coverage of cdist for standard IEEE float inputs.

    Exercises four metrics (angular, sqeuclidean, euclidean, dot) across three
    input dtypes (float64, float32, float16) and three output modes (default
    float64, explicit float32, explicit int32).  Inputs are intentionally
    *strided* — sliced from wider arrays — so the kernel must respect non-
    contiguous memory layouts.

    Additionally tests the ``out=`` buffer path: a strided slice of a wider
    allocation is passed and checked for correct in-place writes.

    Skips:
        * ``angular`` at ndim=1 — degenerate (norm is a single element, 0/0).

    Dimensions from ``NK_DENSE_DIMENSIONS``; capabilities from platform
    auto-detection.  Integer output uses ``atol=1`` (discrete rounding);
    floats use ``NK_ATOL / NK_RTOL``.
    """
    if metric == "angular" and ndim == 1:
        pytest.skip("angular at ndim=1 is degenerate (0/0 from single-element norms)")
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 10, 15
    a_matrix_extended, _ = make_random((num_rows_a, ndim + 1), input_dtype, seed=nk_seed)
    b_matrix_extended, _ = make_random((num_rows_b, ndim + 3), input_dtype, seed=nk_seed + 1)
    a_matrix = a_matrix_extended[:, :ndim]
    b_matrix = b_matrix_extended[:, :ndim]

    is_integer_output = out_dtype in ("int32", "int64", "int16", "int8", "uint32", "uint64", "uint16", "uint8")
    scipy_metric = scipy_metric_name(metric)

    if metric == "dot":
        baseline = np.array(
            [
                [np.dot(a_matrix[i].astype(np.float64), b_matrix[j].astype(np.float64)) for j in range(num_rows_b)]
                for i in range(num_rows_a)
            ]
        )
    else:
        baseline = spd.cdist(a_matrix, b_matrix, scipy_metric)

    if out_dtype is None:
        expected = baseline
        result = nk.cdist(a_matrix, b_matrix, metric)
    else:
        expected = round_and_clip_even(baseline, out_dtype) if is_integer_output else baseline.astype(out_dtype)
        result = nk.cdist(a_matrix, b_matrix, metric, out_dtype=out_dtype)

    atol = 1 if is_integer_output else NK_ATOL
    assert_allclose(result, expected, atol=atol, rtol=NK_RTOL)

    # Test out= buffer with strides
    out_np_dtype = out_dtype if out_dtype else "float64"
    output_buffer_extended = np.zeros((num_rows_a, num_rows_b + 7), dtype=out_np_dtype)
    output_buffer = output_buffer_extended[:, :num_rows_b]
    assert nk.cdist(a_matrix, b_matrix, metric, out=output_buffer) is None
    assert_allclose(output_buffer, expected, atol=atol, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["complex128", "complex64"])
@pytest.mark.parametrize("out_dtype", [None, "complex128", "complex64"])
@pytest.mark.parametrize("metric", ["dot", "vdot"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_complex(ndim, input_dtype, out_dtype, metric, capability):
    """Verify cdist for complex-valued dot and vdot metrics.

    Tests three output modes (default complex128, explicit complex128,
    explicit complex64) for both ``dot`` (Hermitian-unaware) and ``vdot``
    (conjugate dot) metrics.  Exercises:

    * 1D scalar result (single row vs single row).
    * 2D matrix result (10 x 15).
    * ``out=`` buffer path with strided column slice.

    Inputs are strided (sliced from wider allocations).  Dimensions from
    ``NK_DENSE_DIMENSIONS``; capabilities from platform auto-detection.
    """
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 10, 15
    a_matrix_extended = np.random.randn(num_rows_a, ndim + 1).astype(input_dtype)
    b_matrix_extended = np.random.randn(num_rows_b, ndim + 3).astype(input_dtype)
    a_matrix = a_matrix_extended[:, :ndim]
    b_matrix = b_matrix_extended[:, :ndim]
    c_matrix_extended = np.random.randn(num_rows_a, num_rows_b + 7).astype(out_dtype if out_dtype else np.complex128)
    c_matrix = c_matrix_extended[:, :num_rows_b]

    expected = np.zeros((num_rows_a, num_rows_b), dtype=out_dtype if out_dtype else np.complex128)
    baseline_kernel = np.dot if metric == "dot" else np.vdot
    for i in range(num_rows_a):
        for j in range(num_rows_b):
            expected[i, j] = baseline_kernel(a_matrix[i], b_matrix[j])

    if out_dtype is None:
        result1d = nk.cdist(a_matrix[0], b_matrix[0], metric=metric)
        result2d = nk.cdist(a_matrix, b_matrix, metric=metric)
        assert nk.cdist(a_matrix, b_matrix, metric=metric, out=c_matrix) is None
    else:
        expected = expected.astype(out_dtype)
        result1d = nk.cdist(a_matrix[0], b_matrix[0], metric=metric, out_dtype=out_dtype)
        result2d = nk.cdist(a_matrix, b_matrix, metric=metric, out_dtype=out_dtype)
        assert nk.cdist(a_matrix, b_matrix, metric=metric, out_dtype=out_dtype, out=c_matrix) is None

    assert_allclose(result1d, expected[0, 0], atol=NK_ATOL, rtol=NK_RTOL)
    assert_allclose(result2d, expected, atol=NK_ATOL, rtol=NK_RTOL)
    assert_allclose(c_matrix, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("out_dtype", [None, "float32", "float16", "int8"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_hamming(ndim, out_dtype, capability):
    """Verify cdist Hamming distance on packed bit vectors.

    Generates random binary matrices, packs them via ``np.packbits``, and passes
    ``dtype="uint1"`` so the kernel interprets each byte as 8 bits.  Baseline is
    ``scipy.spatial.distance.cdist(bits, bits, "hamming") * ndim`` — SciPy
    normalises by dimension, so we undo that.

    Output dtype coverage: default (float64), float32, float16, int8.  Integer
    output uses standard ``NK_ATOL / NK_RTOL`` (Hamming counts are exact for
    integer types but may round for float16).

    Randomised via ``@pytest.mark.repeat(randomized_repetitions_count)`` (env
    ``NK_RANDOMIZED_REPETITIONS``, default 1).  Dimensions from
    ``NK_DENSE_DIMENSIONS``; capabilities from platform auto-detection.
    """
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 10, 15
    a_bits = np.random.randint(2, size=(num_rows_a, ndim)).astype(np.uint8)
    b_bits = np.random.randint(2, size=(num_rows_b, ndim)).astype(np.uint8)
    a_packed_bits, b_packed_bits = np.packbits(a_bits, axis=1), np.packbits(b_bits, axis=1)

    if out_dtype is None:
        expected = spd.cdist(a_bits, b_bits, "hamming") * ndim
        result = nk.cdist(a_packed_bits, b_packed_bits, metric="hamming", dtype="uint1")
    else:
        raw = spd.cdist(a_bits, b_bits, "hamming") * ndim
        if np.issubdtype(np.dtype(out_dtype), np.integer):
            expected = round_and_clip_even(raw, out_dtype)
        else:
            expected = raw.astype(out_dtype)
        result = nk.cdist(a_packed_bits, b_packed_bits, metric="hamming", dtype="uint1", out_dtype=out_dtype)

    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("out_dtype", [None, "float32"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_jaccard(ndim, out_dtype, capability):
    """Verify cdist Jaccard distance on packed bit vectors.

    Same ``np.packbits`` + ``dtype="uint1"`` pattern as the Hamming test, but
    with ``metric="jaccard"``.  Baseline is ``scipy.spatial.distance.cdist``
    with ``"jaccard"`` (already normalised — ratio of disagreeing bits to bits
    where at least one is set).

    Output dtype coverage: default (float64) and explicit float32.

    Randomised via ``@pytest.mark.repeat``; dimensions from
    ``NK_DENSE_DIMENSIONS``; capabilities from platform auto-detection.
    """
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 10, 15
    a_bits = np.random.randint(2, size=(num_rows_a, ndim)).astype(np.uint8)
    b_bits = np.random.randint(2, size=(num_rows_b, ndim)).astype(np.uint8)
    a_packed_bits, b_packed_bits = np.packbits(a_bits, axis=1), np.packbits(b_bits, axis=1)

    if out_dtype is None:
        expected = spd.cdist(a_bits, b_bits, "jaccard")
        result = nk.cdist(a_packed_bits, b_packed_bits, metric="jaccard", dtype="uint1")
    else:
        expected = spd.cdist(a_bits, b_bits, "jaccard").astype(out_dtype)
        result = nk.cdist(a_packed_bits, b_packed_bits, metric="jaccard", dtype="uint1", out_dtype=out_dtype)

    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("input_dtype", ["float64", "float32"])
@pytest.mark.parametrize("metric", ["kld", "jsd"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_cdist_probability(ndim, input_dtype, metric, capability):
    """Verify cdist for Kullback-Leibler divergence and Jensen-Shannon distance.

    Inputs are positive probability vectors (softmax of randn + epsilon) using
    asymmetric 7 x 11 matrix sizes.

    Baselines:
        * **KLD** — ``scipy_entropy(p, q)`` which computes
          ``sum(p * ln(p/q))`` in natural log (nats), matching ``nk.kld``.
          Note: KLD is asymmetric, so ``cdist(A, B)`` != ``cdist(B, A)``.
        * **JSD** — ``scipy.spatial.distance.jensenshannon`` with default
          ``base=e`` (natural log), returning ``sqrt(JS divergence)`` which
          matches ``nk.jsd`` directly.

    Only float64 and float32 input dtypes are supported.  No ``out_dtype``
    variants — probability divergences always produce float64 output.
    Dimensions from ``NK_DENSE_DIMENSIONS``; capabilities from platform
    auto-detection.
    """
    keep_one_capability(capability)

    num_rows_a, num_rows_b = 7, 11
    # Positive normalized vectors (softmax of randn)
    a_raw = np.abs(np.random.randn(num_rows_a, ndim)).astype(input_dtype) + 1e-6
    b_raw = np.abs(np.random.randn(num_rows_b, ndim)).astype(input_dtype) + 1e-6
    a_matrix = (a_raw / a_raw.sum(axis=1, keepdims=True)).astype(input_dtype)
    b_matrix = (b_raw / b_raw.sum(axis=1, keepdims=True)).astype(input_dtype)

    if metric == "kld":
        # scipy_entropy(p, q) = sum(p * ln(p/q)), natural log, matches nk.kld
        expected = spd.cdist(
            a_matrix.astype(np.float64),
            b_matrix.astype(np.float64),
            lambda u, v: scipy_entropy(u, v),
        )
    else:
        # spd.jensenshannon defaults to base=e (natural log), returns sqrt(JS divergence)
        expected = spd.cdist(a_matrix.astype(np.float64), b_matrix.astype(np.float64), "jensenshannon")

    result = nk.cdist(a_matrix, b_matrix, metric=metric)
    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize(
    "input_dtype",
    ["bfloat16", "e4m3", "e5m2", "e2m3", "e3m2", "int8", "uint8", "int4", "uint4"],
)
@pytest.mark.parametrize("metric", ["dot", "euclidean"])
def test_cdist_exotic_dtypes(ndim, input_dtype, metric):
    """Verify cdist for sub-byte and non-standard types via pairwise fallback.

    Covers bfloat16, e4m3, e5m2, e2m3, e3m2, int8, uint8, int4, and uint4 for
    the ``dot`` and ``euclidean`` metrics.  These dtypes have no batch-path
    kernel, so cdist always falls back to the scalar pairwise loop with float64
    output.

    Baselines are computed from the float64 arrays returned by ``make_random``
    (second return value), avoiding sub-byte row-indexing issues on
    ``nk.Tensor``.  Raw numpy byte-arrays are passed to cdist with an explicit
    ``dtype=`` parameter so the C code knows how to interpret the data.

    Sub-byte types (int4, uint4, uint1) have their ``ndim`` rounded up to
    the packing alignment (2 for nibble types, 8 for bit types) so the
    kernel sees a whole number of packed bytes per row.

    No ``capability`` parameter — only the default backend is tested.
    Dimensions from ``NK_DENSE_DIMENSIONS``.
    """
    ndim = round_up_to(ndim, PACKING_GRANULARITY.get(input_dtype, 1))

    num_rows_a, num_rows_b = 5, 7
    a_raw, a_baseline = make_random((num_rows_a, ndim), input_dtype)
    b_raw, b_baseline = make_random((num_rows_b, ndim), input_dtype)

    # Baseline: compute from float64 baseline arrays (avoids sub-byte row indexing issues)
    if metric == "dot":
        expected = a_baseline @ b_baseline.T
    else:
        expected = np.zeros((num_rows_a, num_rows_b), dtype=np.float64)
        for i in range(num_rows_a):
            for j in range(num_rows_b):
                expected[i, j] = np.sqrt(np.sum((a_baseline[i] - b_baseline[j]) ** 2))

    # Use raw numpy arrays with explicit dtype= for sub-byte/exotic types
    cdist_kwargs = {"metric": metric, "dtype": input_dtype}

    # Test with default out_dtype (f64, pairwise fallback)
    result_f64 = nk.cdist(a_raw, b_raw, **cdist_kwargs)
    assert_allclose(result_f64, expected, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.parametrize(
    "m,n,k",
    [(1, 1, 8), (1, 5, 16), (5, 1, 16), (1, 1, 1), (2, 3, 4096), (13, 17, 97)],
)
def test_cdist_shapes(m, n, k, nk_seed):
    """Verify cdist output shape and correctness for diverse matrix geometries.

    Uses a hardcoded list of ``(m, n, k)`` triples that exercise edge cases
    the dimension-parameterised tests miss:

    * ``(1, 1, 8)`` — minimal single-row × single-row.
    * ``(1, 5, 16)`` and ``(5, 1, 16)`` — 1-row broadcasting in each direction.
    * ``(1, 1, 1)`` — absolute minimum dimensions.
    * ``(2, 3, 4096)`` — large vector depth (exercises SIMD tail handling).
    * ``(13, 17, 97)`` — odd prime dimensions with no alignment.

    Both ``euclidean`` (may use batch path) and ``sqeuclidean`` (pairwise only)
    are tested on float32 input with default float64 output.  Asserts both
    correctness (against SciPy) and output shape ``(m, n)``.

    Not parameterised by capability — shape handling is ISA-independent.
    """
    a_matrix, _ = make_random((m, k), "float32", seed=nk_seed)
    b_matrix, _ = make_random((n, k), "float32", seed=nk_seed + 1)

    # euclidean (may use batch path)
    expected_euc = spd.cdist(a_matrix, b_matrix, "euclidean")
    result_euc = nk.cdist(a_matrix, b_matrix, "euclidean")
    assert result_euc.shape == (m, n), f"Expected shape ({m}, {n}), got {result_euc.shape}"
    assert_allclose(result_euc, expected_euc, atol=NK_ATOL, rtol=NK_RTOL)

    # sqeuclidean (pairwise)
    expected_sqeuc = spd.cdist(a_matrix, b_matrix, "sqeuclidean")
    result_sqeuc = nk.cdist(a_matrix, b_matrix, "sqeuclidean")
    assert result_sqeuc.shape == (m, n), f"Expected shape ({m}, {n}), got {result_sqeuc.shape}"
    assert_allclose(result_sqeuc, expected_sqeuc, atol=NK_ATOL, rtol=NK_RTOL)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("m", [31, 64, 129])
def test_cdist_threads(m, nk_seed):
    """Verify OpenMP-parallel cdist matches the serial path across tile boundaries.

    Exercises both the symmetric batch path (A vs A) and the packed batch path
    (A vs B). Packed tile is 64, symmetric tile is 32, so ``m`` values are
    chosen to straddle one and two tiles.
    """
    k = 32
    a_matrix, _ = make_random((m, k), "float32", seed=nk_seed)
    b_matrix, _ = make_random((m, k), "float32", seed=nk_seed + 1)

    # Symmetric batch path
    serial_symmetric = nk.cdist(a_matrix, a_matrix, "sqeuclidean", threads=1)
    parallel_symmetric = nk.cdist(a_matrix, a_matrix, "sqeuclidean", threads=4)
    assert_allclose(parallel_symmetric, serial_symmetric, atol=NK_ATOL, rtol=NK_RTOL)

    # Packed batch path
    serial_packed = nk.cdist(a_matrix, b_matrix, "euclidean", threads=1)
    parallel_packed = nk.cdist(a_matrix, b_matrix, "euclidean", threads=4)
    assert_allclose(parallel_packed, serial_packed, atol=NK_ATOL, rtol=NK_RTOL)


def test_cdist_edge_cases(nk_seed):
    """Verify cdist edge cases: scalar return, error handling, and kwarg validation.

    Covers four categories:

    1. **Scalar return** — passing two 1D vectors ``(D,)`` must return a scalar
       float, not a ``(1, 1)`` matrix.
    2. **Accepted kwargs** — ``threads=`` controls OpenMP parallelism and must
       produce results identical to the single-threaded path.
    3. **Rejected kwargs** — unknown keywords must raise ``TypeError`` with
       "unexpected keyword" so callers get a clear error.
    4. **Input validation** — mismatched dimensions and empty matrices must
       raise ``ValueError``.

    Not parameterised — uses fixed 16-element vectors on float32.
    """
    ndim = 16
    a_vec = nk.hash((ndim,), seed=nk_seed, dtype="float32")
    b_vec = nk.hash((ndim,), seed=nk_seed + 1, dtype="float32")

    # 1D vectors → scalar float return, not matrix
    result = nk.cdist(a_vec, b_vec, "euclidean")
    assert isinstance(result, (int, float)), f"Expected scalar for 1D inputs, got {type(result)}"

    # threads= is accepted and must match the single-threaded result
    a_matrix = nk.ones((4, ndim), dtype="float32")
    b_matrix = nk.ones((4, ndim), dtype="float32")
    result_serial = nk.cdist(a_matrix, b_matrix, "euclidean", threads=1)
    result_parallel = nk.cdist(a_matrix, b_matrix, "euclidean", threads=2)
    assert_allclose(result_serial, result_parallel, atol=NK_ATOL, rtol=NK_RTOL)

    # Unknown kwargs are rejected
    with pytest.raises(TypeError, match="unexpected keyword"):
        nk.cdist(a_matrix, b_matrix, not_a_real_kwarg=2)

    # Mismatched dimensions → ValueError
    with pytest.raises(ValueError):
        nk.cdist(nk.ones((2, 3), dtype="float32"), nk.ones((2, 5), dtype="float32"), "euclidean")

    # Empty matrix → ValueError
    with pytest.raises(ValueError):
        nk.cdist(nk.ones((0, 3), dtype="float32"), nk.ones((2, 3), dtype="float32"), "euclidean")


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not ml_dtypes_available, reason="ml_dtypes not installed")
@pytest.mark.parametrize("ml_dtype", ["bfloat16", "float8_e4m3fn", "float8_e5m2", "float6_e2m3fn", "float6_e3m2fn"])
def test_cdist_with_ml_dtypes(ml_dtype):
    """Verify cdist accepts ml_dtypes arrays via __array_interface__ fallback."""
    dt = getattr(ml_dtypes, ml_dtype)
    a = np.random.randn(4, 8).astype(np.float32).clip(-1, 1).astype(dt)
    b = np.random.randn(4, 8).astype(np.float32).clip(-1, 1).astype(dt)
    result = nk.cdist(a, b, "dot")
    assert result.shape == (4, 4)
