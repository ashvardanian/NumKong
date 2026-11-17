#!/usr/bin/env python3
"""Test probability divergences: nk.jensenshannon, nk.kullbackleibler.

Covers: float32, float16.
Baselines: high-precision Decimal with ln(), SciPy jensenshannon / rel_entr.
Matches C++ suite: test/probability.cpp.
Run: python -m pytest test/probability.py

File: test/probability.py
Author: Ash Vardanian
Date: February 22, 2024
"""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest
from base import (
    ALGEBRAIC_NDIMS,
    NUMKONG_ATOL,
    NUMKONG_RTOL,
    SETTINGS,
    assert_allclose,
    collect_errors,
    make_positive_buffer,
    numpy_available,
    precise_decimal,
    scipy_available,
    timed_call,
)

import numkong as nk


if numpy_available:
    import numpy as np

if scipy_available:
    import scipy.spatial.distance as spd
    from scipy.special import rel_entr

    def baseline_jensenshannon(x, y, dtype=None):
        """Jensen-Shannon distance through SciPy, the reference for `nk.jensenshannon`."""
        return spd.jensenshannon(x, y)

    def baseline_kullbackleibler(p, q, dtype=None):
        """Kullback-Leibler divergence through SciPy, the reference for `nk.kullbackleibler`."""
        return float(np.sum(rel_entr(p, q)))

else:

    def baseline_jensenshannon(p, q, dtype=None):
        """Jensen-Shannon distance (sqrt of JSD) — fallback when SciPy is unavailable."""
        m = 0.5 * (p + q)
        left = np.sum(np.where(p > 0, p * np.log(p / m), 0.0))
        right = np.sum(np.where(q > 0, q * np.log(q / m), 0.0))
        return np.sqrt(np.clip(0.5 * (left + right), 0.0, None))

    def baseline_kullbackleibler(p, q, dtype=None):
        """KL divergence — fallback when SciPy is unavailable."""
        return float(np.sum(np.where(p > 0, p * np.log(p / q), 0.0)))


def precise_jensenshannon(p, q, dtype=None):
    """High-precision Jensen–Shannon distance via Python Decimal."""
    with precise_decimal(dtype) as (upcast, sqrt, ln):
        divergence_first = upcast(0)
        divergence_second = upcast(0)
        for first_value, second_value in zip(p, q):
            first_precise = upcast(first_value)
            second_precise = upcast(second_value)
            midpoint = (first_precise + second_precise) / 2
            if first_precise > 0 and midpoint > 0:
                divergence_first += first_precise * ln(first_precise / midpoint)
            if second_precise > 0 and midpoint > 0:
                divergence_second += second_precise * ln(second_precise / midpoint)
        jensen_shannon_divergence = (divergence_first + divergence_second) / 2
        if jensen_shannon_divergence < 0:
            jensen_shannon_divergence = upcast(0)
        return float(sqrt(jensen_shannon_divergence))


def precise_kullbackleibler(p, q, dtype=None):
    """High-precision KL divergence via Python Decimal."""
    with precise_decimal(dtype) as (upcast, _sqrt, ln):
        total = upcast(0)
        for first_value, second_value in zip(p, q):
            first_precise = upcast(first_value)
            second_precise = upcast(second_value)
            if first_precise > 0 and second_precise > 0:
                total += first_precise * ln(first_precise / second_precise)
        return float(total)


KERNELS_PROBABILITY: dict[str, tuple[Callable, Callable, Callable]] = {
    "jensenshannon": (baseline_jensenshannon, nk.jensenshannon, precise_jensenshannon),
    "kullbackleibler": (baseline_kullbackleibler, nk.kullbackleibler, precise_kullbackleibler),
}


@pytest.mark.skip(reason="Problems inferring the tolerance bounds for numerical errors")
@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("ndim", SETTINGS.dims)
def test_jensenshannon_matches_scipy(stats, capabilities, ndim: int, dtype: str, np_rng: np.random.Generator):
    """Jensen-Shannon distance of random probability distributions against SciPy baseline."""
    a_distribution = np.abs(np_rng.standard_normal(ndim)).astype(dtype)
    b_distribution = np.abs(np_rng.standard_normal(ndim)).astype(dtype)
    a_distribution /= np.sum(a_distribution)
    b_distribution /= np.sum(b_distribution)

    baseline_kernel, simd_kernel, _ = KERNELS_PROBABILITY["jensenshannon"]
    accurate_ns, accurate = timed_call(
        baseline_kernel, a_distribution.astype(np.float64), b_distribution.astype(np.float64)
    )
    expected_ns, expected = timed_call(baseline_kernel, a_distribution, b_distribution)
    result_ns, result = timed_call(simd_kernel, a_distribution, b_distribution, capabilities=capabilities)
    result = np.asarray(result)

    assert_allclose(result, expected, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)
    collect_errors(
        "jensenshannon",
        ndim,
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


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("ndim", ALGEBRAIC_NDIMS)
def test_kullbackleibler_of_a_distribution_with_itself_is_zero(capabilities, ndim: int, dtype: str):
    """KL divergence of a uniform distribution with itself should be ~0."""
    uniform_distribution = nk.full((ndim,), 1.0 / ndim, dtype=dtype)
    result = nk.kullbackleibler(uniform_distribution, uniform_distribution, capabilities=capabilities)
    assert abs(result) < NUMKONG_ATOL, f"KL(p,p) = {result}, expected ~0"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("ndim", ALGEBRAIC_NDIMS)
def test_jensenshannon_of_a_distribution_with_itself_is_zero(capabilities, ndim: int, dtype: str):
    """Jensen-Shannon distance of a uniform distribution with itself should be ~0."""
    uniform_distribution = nk.full((ndim,), 1.0 / ndim, dtype=dtype)
    result = nk.jensenshannon(uniform_distribution, uniform_distribution, capabilities=capabilities)
    assert abs(result) < NUMKONG_ATOL, f"JS(p,p) = {result}, expected ~0"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("ndim", ALGEBRAIC_NDIMS)
def test_jensenshannon_is_symmetric_and_non_negative(capabilities, ndim: int, dtype: str, rng: random.Random):
    """Jensen-Shannon should be symmetric and non-negative."""
    p = make_positive_buffer(rng, ndim, dtype)
    q = make_positive_buffer(rng, ndim, dtype)
    js_pq = nk.jensenshannon(p, q, capabilities=capabilities)
    js_qp = nk.jensenshannon(q, p, capabilities=capabilities)
    assert abs(js_pq - js_qp) < NUMKONG_ATOL, f"JS not symmetric: js(p,q)={js_pq}, js(q,p)={js_qp}"
    assert js_pq >= -NUMKONG_ATOL, f"JS negative: {js_pq}"
