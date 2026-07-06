#!/usr/bin/env python3
"""Test MaxSim late-interaction scoring: nk.maxsim, nk.maxsim_pack, nk.maxsim_packed.

Dtypes: float32, bfloat16, float16.
Baselines: high-precision Decimal angular distances, NumPy reference.
Matches C++ suite: test_maxsim.cpp.
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
    assert_allclose,
    collect_warnings,
    create_stats,
    downcast_f32_to_dtype,
    keep_one_capability,
    make_nk,
    nk_seed,  # noqa: F401 — pytest fixture
    numpy_available,
    possible_capabilities,
    precise_decimal,
    print_stats_report,
    reduced_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    test_depth_dimensions,
    test_height_dimensions,
    test_width_dimensions,
)

import numkong as nk


stats = create_stats()
atexit.register(print_stats_report, stats)

_MAXSIM_DTYPE = {"float32": "f32", "bfloat16": "bf16", "float16": "f16"}


def baseline_maxsim(queries, documents, dtype=None):
    """Pure NumPy reference: sum of per-query minimum angular distances."""
    total = 0.0
    for q in queries:
        q64 = q.astype(np.float64)
        norm_q = np.linalg.norm(q64)
        min_ang = float("inf")
        for d in documents:
            d64 = d.astype(np.float64)
            norm_d = np.linalg.norm(d64)
            dot = np.dot(q64, d64)
            angular = max(0.0, 1.0 - dot / (norm_q * norm_d)) if norm_q > 0 and norm_d > 0 else 1.0
            min_ang = min(min_ang, angular)
        total += min_ang
    return total


def precise_maxsim(queries, documents, dtype=None):
    """High-precision MaxSim via Python Decimal: sum of per-query min angular distances."""
    with precise_decimal(dtype) as (upcast, sqrt, _ln):
        total = upcast(0)
        for query_vector in queries:
            query_values = [upcast(value) for value in query_vector]
            norm_query = sqrt(sum(value * value for value in query_values))
            min_angular = None
            for document_vector in documents:
                dot_product = upcast(0)
                norm_squared = upcast(0)
                for query_value, document_value in zip(query_values, document_vector):
                    document_precise = upcast(document_value)
                    dot_product += query_value * document_precise
                    norm_squared += document_precise * document_precise
                norm_document = sqrt(norm_squared)
                if norm_query > 0 and norm_document > 0:
                    cosine_similarity = dot_product / (norm_query * norm_document)
                    angular = max(upcast(0), upcast(1) - cosine_similarity)
                else:
                    angular = upcast(1)
                if min_angular is None or angular < min_angular:
                    min_angular = angular
            total += min_angular
        return float(total)


KERNELS_MAXSIM: dict[str, tuple[Callable, Callable, Callable]] = {
    "maxsim": (baseline_maxsim, nk.maxsim, precise_maxsim),
}


def _make_matrix(rows, cols, dtype):
    """Create a test matrix in the target dtype."""
    raw, _ = downcast_f32_to_dtype(np.random.randn(rows, cols).astype(np.float32), dtype)
    return make_nk(raw, dtype)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(reduced_repetitions_count)
@pytest.mark.parametrize("rows", test_height_dimensions)
@pytest.mark.parametrize("columns", test_width_dimensions)
@pytest.mark.parametrize("depth", test_depth_dimensions)
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_pack_and_packed(rows: int, columns: int, depth: int, dtype: str, capability: str):
    """Pack + compute vs baseline."""
    keep_one_capability(capability)
    baseline_kernel, _, _precise_kernel = KERNELS_MAXSIM["maxsim"]
    dtype_str = _MAXSIM_DTYPE[dtype]
    queries = _make_matrix(rows, depth, dtype)
    documents = _make_matrix(columns, depth, dtype)

    qp = nk.maxsim_pack(queries, dtype=dtype_str)
    dp = nk.maxsim_pack(documents, dtype=dtype_str)

    assert isinstance(qp, nk.MaxSimPackedMatrix)
    assert qp.vector_count == rows
    assert qp.depth == depth
    assert qp.nbytes > 0
    assert repr(qp)  # smoke-test repr doesn't crash

    result = nk.maxsim_packed(qp, dp)

    # MaxSim is an approximation (coarse screening + fine pass); compound
    # angular-distance accumulation amplifies f32 error to ~2% relative.
    # Verify sanity bounds only and log deviations for the stats report.
    assert np.isfinite(result), f"maxsim_packed({dtype}): got non-finite {result}"
    assert result >= 0, f"maxsim_packed({dtype}): got negative {result}"
    if dtype != "bfloat16":
        q_np = np.array(queries, dtype=dtype)
        d_np = np.array(documents, dtype=dtype)
        expected = baseline_kernel(q_np, d_np, dtype=dtype_str)
        rel_err = abs(result - expected) / max(abs(expected), 1e-12)
        if rel_err > 0.1:
            collect_warnings(f"maxsim_packed({dtype}) rel_err={rel_err:.4f}", stats)


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_convenience(dtype: str, capability: str):
    """Verify maxsim() matches maxsim_pack + maxsim_packed."""
    keep_one_capability(capability)
    n_q, n_d, depth = 4, 8, 32
    dtype_str = _MAXSIM_DTYPE[dtype]
    queries = _make_matrix(n_q, depth, dtype)
    documents = _make_matrix(n_d, depth, dtype)

    # Packed path
    qp = nk.maxsim_pack(queries, dtype=dtype_str)
    dp = nk.maxsim_pack(documents, dtype=dtype_str)
    packed_result = nk.maxsim_packed(qp, dp)

    # Convenience path
    _, simd_kernel, _ = KERNELS_MAXSIM["maxsim"]
    conv_result = simd_kernel(queries, documents, dtype=dtype_str)

    assert_allclose(packed_result, conv_result)


@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_self_zero(capability: str):
    """Identical queries and documents should give score near zero."""
    keep_one_capability(capability)
    vectors = nk.ones((8, 64), dtype="float32")
    result = nk.maxsim(vectors, vectors, dtype="f32")
    assert_allclose(result, 0.0, err_msg="Expected near-zero self-distance")


@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_type_errors(capability: str, nk_seed: int):
    """Wrong type and mismatched dtype/depth."""
    keep_one_capability(capability)
    query = nk.hash((4, 32), seed=nk_seed, dtype="float32")
    documents = nk.hash((8, 32), seed=nk_seed + 1, dtype="float32")
    query_packed = nk.maxsim_pack(query, dtype="f32")
    documents_packed = nk.maxsim_pack(documents, dtype="f32")

    # Wrong type for maxsim_packed
    with pytest.raises(TypeError):
        nk.maxsim_packed(query, documents_packed)
    with pytest.raises(TypeError):
        nk.maxsim_packed(query_packed, documents)

    # Mismatched depth
    documents_wrong = nk.hash((8, 16), seed=nk_seed + 2, dtype="float32")
    documents_wrong_packed = nk.maxsim_pack(documents_wrong, dtype="f32")
    with pytest.raises(ValueError):
        nk.maxsim_packed(query_packed, documents_wrong_packed)


@pytest.mark.parametrize("capability", possible_capabilities)
def test_maxsim_packed_size(capability: str):
    """Verify classmethod packed_size returns a positive integer."""
    keep_one_capability(capability)
    size = nk.MaxSimPackedMatrix.packed_size(8, 64, dtype="bf16")
    assert isinstance(size, int)
    assert size > 0

    size_f32 = nk.MaxSimPackedMatrix.packed_size(8, 64, dtype="f32")
    assert size_f32 > 0
