#!/usr/bin/env python3
"""Tests for the stateful NumKong random generator."""

import numkong as nk
import pytest

np = pytest.importorskip("numpy")


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("random", {"size": (3, 4), "dtype": "float32"}),
        ("uniform", {"low": -2.0, "high": 3.0, "size": 12, "dtype": "float64"}),
        ("normal", {"loc": 1.0, "scale": 2.0, "size": 12, "dtype": "float32"}),
        ("standard_normal", {"size": 12, "dtype": "float64"}),
        ("integers", {"low": -3, "high": 7, "size": (3, 4), "dtype": "int32"}),
    ],
)
def test_seed_reproduces_each_distribution(method, kwargs):
    first = getattr(nk.Generator(137), method)(**kwargs)
    second = getattr(nk.Generator(137), method)(**kwargs)

    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_generators_have_independent_state():
    first = nk.Generator(137)
    second = nk.Generator(137)

    first.random(8)
    np.testing.assert_array_equal(np.asarray(second.random(8)), np.asarray(nk.Generator(137).random(8)))


def test_state_can_restore_a_sequence():
    generator = nk.Generator(137)
    generator.random(5)
    state = generator.state
    expected = np.asarray(generator.normal(size=9, dtype="float64"))

    generator.state = state
    np.testing.assert_array_equal(expected, np.asarray(generator.normal(size=9, dtype="float64")))


def test_outputs_cover_shapes_dtypes_and_bounds():
    generator = nk.Generator(137)
    uniform = np.asarray(generator.uniform(-2.0, 3.0, size=(4, 5), dtype="float32"))
    normal = np.asarray(generator.normal(loc=4.0, scale=0.0, size=5, dtype="float64"))
    integers = np.asarray(generator.integers(0, 8, size=100, dtype="uint8"))

    assert uniform.shape == (4, 5)
    assert uniform.dtype == np.float32
    assert np.all(uniform >= -2.0)
    assert np.all(uniform < 3.0)
    np.testing.assert_array_equal(normal, np.full(5, 4.0))
    assert integers.dtype == np.uint8
    assert np.all((integers >= 0) & (integers < 8))


def test_out_buffer_is_filled_in_place():
    generator = nk.Generator(137)
    output = np.empty((2, 3), dtype=np.float32)

    assert generator.random((2, 3), out=output) is None
    assert output.shape == (2, 3)
    assert np.all((output >= 0.0) & (output < 1.0))


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda generator: generator.random(), TypeError),
        (lambda generator: generator.uniform(3.0, 3.0, size=4), ValueError),
        (lambda generator: generator.normal(scale=-1.0, size=4), ValueError),
        (lambda generator: generator.integers(4, 4, size=4), ValueError),
    ],
)
def test_invalid_distribution_arguments_raise(call, error):
    with pytest.raises(error):
        call(nk.Generator(137))


def test_out_requires_matching_c_contiguous_buffer():
    generator = nk.Generator(137)
    non_contiguous = np.empty((3, 2), dtype=np.float32).T
    wrong_dtype = np.empty((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="C-contiguous"):
        generator.random((2, 3), out=non_contiguous)
    with pytest.raises(TypeError, match="dtype"):
        generator.random((2, 3), out=wrong_dtype)
