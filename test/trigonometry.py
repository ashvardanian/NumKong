#!/usr/bin/env python3
"""Test trigonometric functions: nk.sin, nk.cos, nk.atan.

DTypes: float64, float32.
Baselines: math.sin/cos/atan (C libm double precision), NumPy references.
Matches C++ suite: test/trigonometry.cpp.

File: test/trigonometry.py
Author: Ash Vardanian
Date: February 27, 2026
"""

from __future__ import annotations

import atexit
import math
import random
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
    SETTINGS,
    assert_allclose,
    collect_errors,
    create_stats,
    make_random,
    make_random_buffer,
    numpy_available,
    print_stats_report,
    timed_call,
)

import numkong as nk


algebraic_dtypes = ["float32", "float64"]
algebraic_ndims = [7, 97]
stats = create_stats()
atexit.register(print_stats_report, stats)


def baseline_sin(a, dtype=None):
    """Reference sin via NumPy."""
    return np.sin(a)


def baseline_cos(a, dtype=None):
    """Reference cos via NumPy."""
    return np.cos(a)


def baseline_atan(a, dtype=None):
    """Reference arctan via NumPy."""
    return np.arctan(a)


def precise_sin(a, dtype=None):
    """High-precision sin via math.sin (C libm double precision)."""
    return [math.sin(float(x)) for x in a]


def precise_cos(a, dtype=None):
    """High-precision cos via math.cos (C libm double precision)."""
    return [math.cos(float(x)) for x in a]


def precise_atan(a, dtype=None):
    """High-precision atan via math.atan (C libm double precision)."""
    return [math.atan(float(x)) for x in a]


KERNELS_TRIGONOMETRY: dict[str, tuple[Callable, Callable, Callable]] = {
    "sin": (baseline_sin, nk.sin, precise_sin),
    "cos": (baseline_cos, nk.cos, precise_cos),
    "atan": (baseline_atan, nk.atan, precise_atan),
}

trigonometry_shapes = [(d,) for d in SETTINGS.dims] + ([(6, 8), (4, 5, 3)] if numpy_available else [])
"""Trig ops are shape-invariant: sweep a couple of NumPy-only rank-N shapes alongside the 1-D
`SETTINGS.dims` so the N-D chunk walker is exercised; comparison flattens for rank-agnosticism.
"""


@pytest.mark.repeat(SETTINGS.repetitions)
@pytest.mark.parametrize("shape", trigonometry_shapes)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("metric", list(KERNELS_TRIGONOMETRY.keys()))
def test_trigonometry_random_accuracy(capabilities, shape: tuple, dtype: str, metric: str, np_rng: np.random.Generator):
    """sin, cos, atan on random inputs of any rank against high-precision baselines."""
    baseline_kernel, simd_kernel, precise_kernel = KERNELS_TRIGONOMETRY[metric]

    if numpy_available:
        a = np_rng.uniform(-np.pi, np.pi, shape).astype(dtype)
        accurate_ns, accurate = timed_call(baseline_kernel, a.astype(np.float64))
        expected_ns, expected = timed_call(baseline_kernel, a)
    else:
        a, a_baseline = make_random(shape, dtype, np_rng)
        accurate_ns, accurate = timed_call(precise_kernel, a_baseline)
        expected_ns, expected = 0, None

    result_ns, result = timed_call(simd_kernel, a, capabilities=capabilities)

    # Elementwise op is shape-invariant; flatten so a rank-N result matches the baseline.
    if numpy_available:
        result = np.asarray(result).reshape(-1)
        accurate = np.asarray(accurate).reshape(-1)
        expected = np.asarray(expected).reshape(-1)

    assert_allclose(result, accurate, atol=NUMKONG_ATOL, rtol=NUMKONG_RTOL)
    collect_errors(
        metric,
        len(accurate),
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


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_trigonometry_at_zero(capabilities, ndim: int, dtype: str):
    """sin(0)~0, cos(0)~1, atan(0)~0."""
    zeros_vector = nk.zeros((ndim,), dtype=dtype)
    sin_values = list(nk.sin(zeros_vector, capabilities=capabilities))
    cos_values = list(nk.cos(zeros_vector, capabilities=capabilities))
    atan_values = list(nk.atan(zeros_vector, capabilities=capabilities))
    for i in range(ndim):
        assert abs(sin_values[i]) < NUMKONG_ATOL, f"sin(0)[{i}]={sin_values[i]}"
        assert abs(cos_values[i] - 1.0) < NUMKONG_ATOL, f"cos(0)[{i}]={cos_values[i]}"
        assert abs(atan_values[i]) < NUMKONG_ATOL, f"atan(0)[{i}]={atan_values[i]}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_trigonometry_known_values(capabilities, ndim: int, dtype: str):
    """sin(pi/2)~1, cos(pi/2)~0, atan(1)~pi/4 for all elements."""
    half_pi = nk.full((ndim,), math.pi / 2, dtype=dtype)
    ones_vector = nk.ones((ndim,), dtype=dtype)
    sin_values = list(nk.sin(half_pi, capabilities=capabilities))
    cos_values = list(nk.cos(half_pi, capabilities=capabilities))
    atan_one = list(nk.atan(ones_vector, capabilities=capabilities))
    for i in range(ndim):
        assert abs(sin_values[i] - 1.0) < NUMKONG_ATOL, f"sin(pi/2)[{i}]={sin_values[i]}"
        assert abs(cos_values[i]) < NUMKONG_ATOL, f"cos(pi/2)[{i}]={cos_values[i]}"
        assert abs(atan_one[i] - math.pi / 4) < NUMKONG_ATOL, f"atan(1)[{i}]={atan_one[i]}"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_pythagorean_identity(capabilities, ndim: int, dtype: str, rng: random.Random):
    """sin^2(x) + cos^2(x) ~ 1."""
    input_angles = make_random_buffer(rng, ndim, dtype)
    sin_values = list(nk.sin(input_angles, capabilities=capabilities))
    cos_values = list(nk.cos(input_angles, capabilities=capabilities))
    for i in range(ndim):
        identity = sin_values[i] ** 2 + cos_values[i] ** 2
        assert abs(identity - 1.0) < NUMKONG_ATOL, f"sin²+cos²={identity} at [{i}]"


@pytest.mark.parametrize("ndim", algebraic_ndims)
@pytest.mark.parametrize("dtype", algebraic_dtypes)
def test_trigonometry_odd_even(capabilities, ndim: int, dtype: str):
    """sin(-x) ~ -sin(x) (odd), cos(-x) ~ cos(x) (even)."""
    for random_angles in [0.5, 1.0, 2.0]:
        positive_input = nk.full((ndim,), random_angles, dtype=dtype)
        negative_input = nk.full((ndim,), -random_angles, dtype=dtype)
        sin_positive = list(nk.sin(positive_input, capabilities=capabilities))
        sin_negative = list(nk.sin(negative_input, capabilities=capabilities))
        cos_positive = list(nk.cos(positive_input, capabilities=capabilities))
        cos_negative = list(nk.cos(negative_input, capabilities=capabilities))
        for i in range(ndim):
            assert abs(sin_negative[i] + sin_positive[i]) < NUMKONG_ATOL, (
                f"sin(-{random_angles}) + sin({random_angles}) = {sin_negative[i] + sin_positive[i]}"
            )
            assert abs(cos_negative[i] - cos_positive[i]) < NUMKONG_ATOL, (
                f"cos(-{random_angles}) - cos({random_angles}) = {cos_negative[i] - cos_positive[i]}"
            )
