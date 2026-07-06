#!/usr/bin/env python3
"""Test set distances: nk.jaccard, nk.hamming.

Dtypes: packed uint1 bits.
Baselines: SciPy hamming/jaccard, NumPy logical operations.
Matches C++ suite: test_set.cpp.
"""

import array
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
    NK_ATOL,
    NK_RTOL,
    assert_allclose,
    collect_errors,
    create_stats,
    dense_dimensions,
    keep_one_capability,
    numpy_available,
    possible_capabilities,
    print_stats_report,
    profile,
    randomized_repetitions_count,
    scipy_available,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
)

import numkong as nk


algebraic_ndims = [7, 97]
stats = create_stats()
atexit.register(print_stats_report, stats)

try:
    import scipy.spatial.distance as spd

    def baseline_hamming(x, y, dtype=None):
        return spd.hamming(x, y) * len(x)

    def baseline_jaccard(x, y, dtype=None):
        return spd.jaccard(x, y)

except ImportError:

    def baseline_hamming(x, y, dtype=None):
        return np.logical_xor(x, y).sum()

    def baseline_jaccard(x, y, dtype=None):
        intersection = np.logical_and(x, y).sum()
        union = np.logical_or(x, y).sum()
        return 0.0 if union == 0 else 1.0 - float(intersection) / float(union)


KERNELS_SET: dict[str, tuple[Callable, Callable, None]] = {
    "jaccard": (baseline_jaccard, nk.jaccard, None),
    "hamming": (baseline_hamming, nk.hamming, None),
}


@pytest.mark.skipif(not numpy_available, reason="NumPy is not installed")
@pytest.mark.skipif(not scipy_available, reason="SciPy is not installed")
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("metric", ["jaccard", "hamming"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_hamming_jaccard_random_accuracy(ndim: int, metric: str, capability: str):
    """Hamming and Jaccard distances for dense bit arrays against SciPy baselines."""
    a_bits = np.random.randint(2, size=ndim).astype(np.uint8)
    b_bits = np.random.randint(2, size=ndim).astype(np.uint8)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_SET[metric]
    accurate_dt, accurate = profile(baseline_kernel, a_bits.astype(np.uint64), b_bits.astype(np.uint64))
    expected_dt, expected = profile(baseline_kernel, a_bits, b_bits)
    result_dt, result = profile(simd_kernel, np.packbits(a_bits), np.packbits(b_bits), "uint1")
    result = np.asarray(result)

    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(metric, ndim, "uint1", accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)

    # Also verify with boolean view
    result_dt, result = profile(simd_kernel, np.packbits(a_bits).view(np.bool_), np.packbits(b_bits).view(np.bool_))
    result = np.asarray(result)

    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors(metric, ndim, "uint1", accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_hamming_self_zero(ndim: int, capability: str):
    """hamming(v, v) = 0 for identical uint8 vectors."""
    keep_one_capability(capability)
    packed_vector = array.array("B", [0xFF] * ndim)
    result = nk.hamming(packed_vector, packed_vector, "uint1")
    assert result == 0, f"hamming(v,v) = {result}, expected 0"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("capability", possible_capabilities)
def test_jaccard_self_zero(ndim: int, capability: str):
    """jaccard(v, v) = 0 for identical non-zero uint8 vectors."""
    keep_one_capability(capability)
    packed_vector = array.array("B", [0xAA] * ndim)
    result = nk.jaccard(packed_vector, packed_vector, "uint1")
    assert abs(result) < NK_ATOL, f"jaccard(v,v) = {result}, expected 0"
