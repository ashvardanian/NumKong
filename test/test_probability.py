#!/usr/bin/env python3
"""Test probability divergences: nk.jensenshannon, nk.kullbackleibler.

Dtypes: float32, float16.
Baselines: high-precision Decimal with ln(), SciPy jensenshannon / rel_entr.
Matches C++ suite: test_probability.cpp.
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
    NK_ATOL,
    NK_RTOL,
    assert_allclose,
    collect_errors,
    create_stats,
    dense_dimensions,
    keep_one_capability,
    make_positive_buffer,
    possible_capabilities,
    precise_decimal,
    print_stats_report,
    profile,
    randomized_repetitions_count,
    seed_rng,  # noqa: F401 — pytest fixture (autouse)
)

import numkong as nk


algebraic_ndims = [7, 97]
stats = create_stats()
atexit.register(print_stats_report, stats)

try:
    import scipy.spatial.distance as spd
    from scipy.special import rel_entr

    def baseline_jensenshannon(x, y, dtype=None):
        return spd.jensenshannon(x, y)

    def baseline_kullbackleibler(p, q, dtype=None):
        return float(np.sum(rel_entr(p, q)))

except ImportError:

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
@pytest.mark.repeat(randomized_repetitions_count)
@pytest.mark.parametrize("ndim", dense_dimensions)
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_jensenshannon_random_accuracy(ndim: int, dtype: str, capability: str):
    """Jensen-Shannon distance of random probability distributions against SciPy baseline."""
    a_distribution = np.abs(np.random.randn(ndim)).astype(dtype)
    b_distribution = np.abs(np.random.randn(ndim)).astype(dtype)
    a_distribution /= np.sum(a_distribution)
    b_distribution /= np.sum(b_distribution)

    keep_one_capability(capability)
    baseline_kernel, simd_kernel, _ = KERNELS_PROBABILITY["jensenshannon"]
    accurate_dt, accurate = profile(
        baseline_kernel, a_distribution.astype(np.float64), b_distribution.astype(np.float64)
    )
    expected_dt, expected = profile(baseline_kernel, a_distribution, b_distribution)
    result_dt, result = profile(simd_kernel, a_distribution, b_distribution)
    result = np.asarray(result)

    assert_allclose(result, expected, atol=NK_ATOL, rtol=NK_RTOL)
    collect_errors("jensenshannon", ndim, dtype, accurate, accurate_dt, expected, expected_dt, result, result_dt, stats)


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_kullbackleibler_self_zero(ndim: int, dtype: str, capability: str):
    """KL divergence of a uniform distribution with itself should be ~0."""
    keep_one_capability(capability)
    uniform_distribution = nk.full((ndim,), 1.0 / ndim, dtype=dtype)
    result = nk.kullbackleibler(uniform_distribution, uniform_distribution)
    assert abs(result) < NK_ATOL, f"KL(p,p) = {result}, expected ~0"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_jensenshannon_self_zero(ndim: int, dtype: str, capability: str):
    """Jensen-Shannon distance of a uniform distribution with itself should be ~0."""
    keep_one_capability(capability)
    uniform_distribution = nk.full((ndim,), 1.0 / ndim, dtype=dtype)
    result = nk.jensenshannon(uniform_distribution, uniform_distribution)
    assert abs(result) < NK_ATOL, f"JS(p,p) = {result}, expected ~0"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("capability", possible_capabilities)
def test_jensenshannon_symmetry_nonneg(ndim: int, dtype: str, capability: str):
    """Jensen-Shannon should be symmetric and non-negative."""
    keep_one_capability(capability)
    p = make_positive_buffer(ndim, dtype)
    q = make_positive_buffer(ndim, dtype)
    js_pq = nk.jensenshannon(p, q)
    js_qp = nk.jensenshannon(q, p)
    assert abs(js_pq - js_qp) < NK_ATOL, f"JS not symmetric: js(p,q)={js_pq}, js(q,p)={js_qp}"
    assert js_pq >= -NK_ATOL, f"JS negative: {js_pq}"
