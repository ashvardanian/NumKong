#!/usr/bin/env python3
"""Test set distances: nk.jaccard, nk.hamming.

DTypes: packed uint1 bits.
Baselines: SciPy hamming/jaccard, NumPy logical operations.
Matches C++ suite: test/set.cpp.

File: test/set.py
Author: Ash Vardanian
Date: February 22, 2024
"""

from __future__ import annotations

import array
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
    PACKING_GRANULARITY,
    SETTINGS,
    assert_allclose,
    collect_errors,
    numpy_available,
    round_up_to,
    scipy_available,
    timed_call,
)

import numkong as nk


algebraic_ndims = [7, 97]

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
@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("ndim", SETTINGS.dims)
@pytest.mark.parametrize("metric", ["jaccard", "hamming"])
def test_hamming_jaccard_random_accuracy(stats, capabilities, ndim: int, metric: str, np_rng: np.random.Generator):
    """Hamming and Jaccard distances for dense bit arrays against SciPy baselines."""
    ndim = round_up_to(ndim, PACKING_GRANULARITY["uint1"])
    a_bits = np_rng.integers(2, size=ndim).astype(np.uint8)
    b_bits = np_rng.integers(2, size=ndim).astype(np.uint8)

    baseline_kernel, simd_kernel, _ = KERNELS_SET[metric]
    accurate_ns, accurate = timed_call(baseline_kernel, a_bits.astype(np.uint64), b_bits.astype(np.uint64))
    expected_ns, expected = timed_call(baseline_kernel, a_bits, b_bits)
    result_ns, result = timed_call(
        simd_kernel, np.packbits(a_bits), np.packbits(b_bits), "uint1", capabilities=capabilities
    )
    result = np.asarray(result)

    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)
    collect_errors(
        metric,
        ndim,
        "uint1",
        accurate,
        accurate_ns,
        expected,
        expected_ns,
        result,
        result_ns,
        stats,
        capability=capabilities.name.lower().replace("|", "+"),
    )

    # Also verify with boolean view
    result_ns, result = timed_call(
        simd_kernel,
        np.packbits(a_bits).view(np.bool_),
        np.packbits(b_bits).view(np.bool_),
        capabilities=capabilities,
    )
    result = np.asarray(result)

    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)
    collect_errors(
        metric,
        ndim,
        "uint1",
        accurate,
        accurate_ns,
        expected,
        expected_ns,
        result,
        result_ns,
        stats,
        capability=capabilities.name.lower().replace("|", "+"),
    )


@pytest.mark.parametrize("ndim", algebraic_ndims)
def test_hamming_self_zero(capabilities, ndim: int):
    """hamming(v, v) = 0 for identical uint8 vectors."""
    packed_vector = array.array("B", [0xFF] * ndim)
    result = nk.hamming(packed_vector, packed_vector, "uint1", capabilities=capabilities)
    assert result == 0, f"hamming(v,v) = {result}, expected 0"


@pytest.mark.parametrize("ndim", algebraic_ndims)
def test_jaccard_self_zero(capabilities, ndim: int):
    """jaccard(v, v) = 0 for identical non-zero uint8 vectors."""
    packed_vector = array.array("B", [0xAA] * ndim)
    result = nk.jaccard(packed_vector, packed_vector, "uint1", capabilities=capabilities)
    assert abs(result) < NUMKONG_ATOL, f"jaccard(v,v) = {result}, expected 0"
