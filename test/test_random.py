#!/usr/bin/env python3
"""Tests for the stateful NumKong random generator."""

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

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
    np.testing.assert_array_equal(output, np.asarray(nk.Generator(137).random((2, 3))))


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


@pytest.mark.parametrize("method", ["random", "uniform", "normal", "standard_normal", "integers"])
@pytest.mark.skipif(
    sys.platform == "emscripten"
    and not getattr(getattr(sys, "_emscripten_info", None), "pthreads", False),
    reason="Emscripten build has no pthread support",
)
def test_shared_generator_serializes_complete_fills(method):
    generator = nk.Generator(137)
    reference = nk.Generator(137)
    barrier = Barrier(4)
    kwargs = {"size": 262144}
    if method == "integers":
        kwargs.update(low=0, high=1000000)

    def fill(_):
        barrier.wait(timeout=10)
        return np.asarray(getattr(generator, method)(**kwargs))

    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(fill, range(4)))
    expected = [np.asarray(getattr(reference, method)(**kwargs)) for _ in range(4)]

    assert generator.state == reference.state
    for actual, serial in zip(sorted(outputs, key=lambda row: row[0]), sorted(expected, key=lambda row: row[0])):
        np.testing.assert_array_equal(actual, serial)


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"])
def test_integer_dtype_limits(dtype):
    limits = np.iinfo(dtype)
    generator = nk.Generator(137)
    samples = np.asarray(generator.integers(int(limits.min), int(limits.max) + 1, size=1024, dtype=dtype))
    assert samples.dtype == np.dtype(dtype)
    midpoint = (int(limits.min) + int(limits.max) + 1) // 2
    assert np.any(samples < midpoint)
    assert np.any(samples >= midpoint)
    np.testing.assert_array_equal(
        np.asarray(generator.integers(int(limits.max), int(limits.max) + 1, size=4, dtype=dtype)),
        np.full(4, limits.max, dtype=dtype),
    )
    for low, high in [(int(limits.min) - 1, 1), (0, int(limits.max) + 2)]:
        with pytest.raises((ValueError, OverflowError)):
            generator.integers(low, high, size=4, dtype=dtype)


def test_uint64_bounds_above_int64():
    generator = nk.Generator(137)
    samples = np.asarray(generator.integers(1 << 63, 1 << 64, size=1024, dtype="uint64"))
    assert np.all(samples >= np.uint64(1 << 63))
    samples = np.asarray(generator.integers(1 << 64, size=1024, dtype="uint64"))
    assert samples.dtype == np.uint64
    assert np.any(samples >= np.uint64(1 << 63))


@pytest.mark.parametrize(
    ("method", "first_argument"),
    [("random", "size"), ("uniform", "low"), ("normal", "loc"), ("standard_normal", "size"), ("integers", "low")],
)
def test_argument_errors_are_type_errors(method, first_argument):
    function = getattr(nk.Generator(137), method)
    with pytest.raises(TypeError, match="positional arguments"):
        function(1, 2, 3, 4, 5, 6)
    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown'"):
        function(unknown=1)
    kwargs = {"low": 0, "high": 10} if method == "integers" else {}
    with pytest.raises(TypeError, match="size"):
        function(**kwargs)
    with pytest.raises(TypeError, match="multiple values"):
        function(1, **{first_argument: 1})


@pytest.mark.parametrize("dtype", ["float32", "int4", "uint1"])
def test_integers_rejects_unsupported_dtypes(dtype):
    with pytest.raises(TypeError, match="integer dtypes"):
        nk.Generator(137).integers(0, 4, size=4, dtype=dtype)


def test_state_cannot_be_deleted():
    generator = nk.Generator(137)
    with pytest.raises(TypeError, match="state must be an integer"):
        del generator.state
    assert generator.state == 137


def test_generator_new_and_reinitialization():
    generator = nk.Generator.__new__(nk.Generator)
    np.testing.assert_array_equal(np.asarray(generator.random(8)), np.asarray(nk.Generator(0).random(8)))
    generator.__init__(137)
    np.testing.assert_array_equal(np.asarray(generator.random(8)), np.asarray(nk.Generator(137).random(8)))
