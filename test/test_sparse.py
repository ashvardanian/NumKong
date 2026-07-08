#!/usr/bin/env python3
"""Test sparse operations: nk.sparse_dot, nk.intersect.

Dtypes: float32 values with uint16/uint32 indices.
Baselines: manual weighted intersection, NumPy intersect1d.
Matches C++ suite: test_sparse.cpp.
"""

import atexit
import platform
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
    collect_errors,
    create_stats,
    is_running_under_qemu,
    keep_one_capability,
    numpy_available,
    possible_capabilities,
    print_stats_report,
    profile,
    randomized_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
    sparse_dimensions,
)

import numkong as nk


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
def test_sparse_dot(capability: str):
    """Test nk.sparse_dot against manual weighted intersection."""
    baseline_kernel, simd_kernel, _ = KERNELS_SPARSE["sparse_dot"]
    sparse_dim = sparse_dimensions[0]
    a_idx = np.unique(np.random.randint(0, sparse_dim, size=min(50, sparse_dim))).astype(np.uint32)
    b_idx = np.unique(np.random.randint(0, sparse_dim, size=min(50, sparse_dim))).astype(np.uint32)
    a_val = np.random.randn(len(a_idx)).astype(np.float32)
    b_val = np.random.randn(len(b_idx)).astype(np.float32)

    keep_one_capability(capability)
    result_dt, result = profile(simd_kernel, a_idx, a_val, b_idx, b_val)

    accurate_dt, accurate = profile(baseline_kernel, a_idx, a_val.astype(np.float64), b_idx, b_val.astype(np.float64))
    expected_dt, expected = profile(baseline_kernel, a_idx, a_val, b_idx, b_val)

    assert_allclose(result, accurate, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(
        "sparse_dot", len(a_idx), "float32", accurate, accurate_dt, expected, expected_dt, result, result_dt, stats
    )


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("dtype", ["uint16", "uint32"])
@pytest.mark.parametrize("first_length_bound", [10, 100, 1000])
@pytest.mark.parametrize("second_length_bound", [10, 100, 1000])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_intersect(dtype: str, first_length_bound: int, second_length_bound: int, capability: str):
    """Compares the nk.intersect() function with numpy.intersect1d."""
    if is_running_under_qemu() and (platform.machine() == "aarch64" or platform.machine() == "arm64"):
        pytest.skip("In QEMU `aarch64` emulation on `x86_64` the `intersect` function is not reliable")

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
