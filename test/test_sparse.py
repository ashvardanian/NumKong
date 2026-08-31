#!/usr/bin/env python3
"""Test sparse operations: nk.sparse_dot, nk.intersect.

Dtypes: float32/bfloat16 values with uint32/uint16 indices.
Baselines: manual weighted intersection, NumPy intersect1d.
Matches C++ suite: test_sparse.cpp.
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


import numkong as nk
from test_base import (
    NK_ATOL,
    NK_RTOL,
    assert_allclose,
    collect_errors,
    downcast_f32_to_dtype,
    create_stats,
    keep_one_capability,
    make_nk,
    numpy_available,
    possible_capabilities,
    print_stats_report,
    profile,
    randomized_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    sparse_dimensions,
)

stats = create_stats()
atexit.register(print_stats_report, stats)


def baseline_intersect(x, y, dtype=None):
    return len(np.intersect1d(x, y))


def baseline_sparse_dot(a_idx, a_val, b_idx, b_val):
    common = np.intersect1d(a_idx, b_idx)
    total = 0.0
    for idx in common:
        total += float(a_val[np.searchsorted(a_idx, idx)]) * float(b_val[np.searchsorted(b_idx, idx)])
    return total


KERNELS_SPARSE: dict[str, tuple[Callable, Callable, None]] = {
    "intersect": (baseline_intersect, nk.intersect, None),
    "sparse_dot": (baseline_sparse_dot, nk.sparse_dot, None),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("capability", possible_capabilities)
@pytest.mark.parametrize("index_dtype,weight_dtype", [("uint32", "float32"), ("uint16", "bfloat16")])
def test_sparse_dot(capability: str, index_dtype: str, weight_dtype: str):
    """Test nk.sparse_dot against manual weighted intersection."""
    baseline_kernel, simd_kernel, _ = KERNELS_SPARSE["sparse_dot"]
    sparse_dim = sparse_dimensions[0]
    # Random lengths on both sides, kept above the 64 below which the x86 kernels redirect to serial.
    a_length, b_length = np.random.randint(64, sparse_dim, size=2)
    a_idx = np.sort(np.random.choice(sparse_dim, size=a_length, replace=False)).astype(index_dtype)
    b_idx = np.sort(np.random.choice(sparse_dim, size=b_length, replace=False)).astype(index_dtype)
    a_val, a_f64 = downcast_f32_to_dtype(np.random.randn(len(a_idx)).astype(np.float32), weight_dtype)
    b_val, b_f64 = downcast_f32_to_dtype(np.random.randn(len(b_idx)).astype(np.float32), weight_dtype)

    keep_one_capability(capability)
    result_dt, result = profile(simd_kernel, a_idx, make_nk(a_val, weight_dtype), b_idx, make_nk(b_val, weight_dtype))

    accurate_dt, accurate = profile(baseline_kernel, a_idx, a_f64, b_idx, b_f64)
    expected_dt, expected = profile(baseline_kernel, a_idx, a_f64.astype(np.float32), b_idx, b_f64.astype(np.float32))

    assert_allclose(result, accurate, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(
        "sparse_dot", len(a_idx), weight_dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("dtype", ["uint16", "uint32", "uint64"])
@pytest.mark.parametrize("first_length_bound", [10, 100, 1000])
@pytest.mark.parametrize("second_length_bound", [10, 100, 1000])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_intersect(dtype: str, first_length_bound: int, second_length_bound: int, capability: str):
    """Compares the nk.intersect() function with numpy.intersect1d."""
    a_length = np.random.randint(1, first_length_bound)
    b_length = np.random.randint(1, second_length_bound)
    a = np.random.randint(first_length_bound * 2, size=a_length, dtype=dtype)
    b = np.random.randint(second_length_bound * 2, size=b_length, dtype=dtype)

    a = np.unique(a)
    b = np.unique(b)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_SPARSE["intersect"]
    expected = baseline_kernel(a, b)
    result = simd_kernel(a, b)

    assert round(float(expected)) == round(float(result)), (
        f"Intersection count mismatch: expected {expected}, got {result}. "
        f"Intersection: {np.intersect1d(a, b)}, a={a}, b={b}"
    )
