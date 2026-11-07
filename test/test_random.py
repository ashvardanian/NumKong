"""Native RNG contracts required by array-based augmentation consumers."""

import copy
import math
import os
import pickle
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numkong as nk
import pytest

np = pytest.importorskip("numpy")

CASES = [
    ("random", {}),
    ("uniform", {"low": -2.0, "high": 3.0}),
    ("normal", {"loc": 1.0, "scale": 2.0}),
    ("standard_normal", {}),
    ("integers", {"low": -3, "high": 7, "dtype": np.int32}),
    ("laplace", {"loc": 1.0, "scale": 2.0}),
    ("beta", {"a": 0.5, "b": 2.0}),
    ("poisson", {"lam": 12.0}),
    ("choice", {"a": 100, "replace": False}),
]


@pytest.mark.parametrize(("method", "kwargs"), CASES)
def test_seed_state_and_copy_preserve_samples(method, kwargs):
    generator = nk.Generator(137, array_factory=np.empty)
    getattr(generator, method)(size=7, **kwargs)
    state = generator.state
    cloned = copy.deepcopy(generator)
    restored = pickle.loads(pickle.dumps(generator))
    expected = getattr(generator, method)(size=(2, 3), **kwargs)
    assert isinstance(expected, np.ndarray)
    for other in (cloned, restored):
        np.testing.assert_array_equal(getattr(other, method)(size=(2, 3), **kwargs), expected)
    generator.state = state
    np.testing.assert_array_equal(getattr(generator, method)(size=(2, 3), **kwargs), expected)


def test_generators_have_independent_state():
    first = nk.Generator(19)
    second = nk.Generator(19)
    first.random(10)
    np.testing.assert_array_equal(np.asarray(second.random(10)), np.asarray(nk.Generator(19).random(10)))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["random", "uniform", "normal", "standard_normal", "laplace"])
def test_scalar_matches_first_array_element(method, dtype):
    scalar = getattr(nk.Generator(137), method)(dtype=dtype)
    array = np.asarray(getattr(nk.Generator(137), method)(size=1, dtype=dtype))
    assert scalar == array[0]


@pytest.mark.parametrize(("method", "kwargs"), CASES)
def test_scalar_empty_and_zero_dimensional_outputs(method, kwargs):
    generator = nk.Generator(137, array_factory=np.empty)
    assert np.isscalar(getattr(generator, method)(**kwargs))
    assert getattr(generator, method)(size=(), **kwargs).shape == ()
    state = generator.state
    assert getattr(generator, method)(size=(2, 0, 3), **kwargs).shape == (2, 0, 3)
    assert generator.state == state


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["random", "standard_normal", "uniform", "normal", "laplace", "beta"])
def test_out_matches_allocated_sequence(method, dtype):
    kwargs = {"a": 2.0, "b": 3.0} if method == "beta" else {}
    output = np.empty((3, 5), dtype=dtype)
    generator = nk.Generator(137)
    assert getattr(generator, method)(out=output, dtype=dtype, **kwargs) is None
    expected = np.asarray(getattr(nk.Generator(137), method)(size=output.shape, dtype=dtype, **kwargs))
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("method", ["uniform", "normal", "laplace", "beta", "poisson"])
def test_broadcasted_and_strided_parameters(method):
    first = np.arange(1, 7, dtype=np.float32).reshape(2, 3)[:, ::-1]
    second = np.array([[2.0], [3.0]])
    names = {"uniform": ("low", "high"), "normal": ("loc", "scale"), "laplace": ("loc", "scale"), "beta": ("a", "b")}
    if method == "uniform":
        second = first + second
    kwargs = {"lam": first} if method == "poisson" else dict(zip(names[method], (first, second)))
    generator = nk.Generator(137, array_factory=np.empty)
    actual = getattr(generator, method)(**kwargs)
    assert actual.shape == (2, 3)
    reference = nk.Generator(137)
    expected = []
    broadcast = np.broadcast_arrays(first, second)
    for a, b in zip(broadcast[0].flat, broadcast[1].flat):
        scalar_kwargs = {"lam": float(a)} if method == "poisson" else dict(zip(names[method], (float(a), float(b))))
        expected.append(getattr(reference, method)(**scalar_kwargs))
    np.testing.assert_array_equal(actual, np.asarray(expected).reshape(2, 3))
    assert generator.state == reference.state


def test_integer_bounds_broadcast_without_float_conversion():
    high = np.array([3, 5, 7], dtype=np.int64)
    generator = nk.Generator(137, array_factory=np.empty)
    result = generator.integers(0, high, size=(1000, 3))
    assert np.all((result >= 0) & (result < high))
    low = np.array([2**63 + 7, 2**64 - 8], dtype=np.uint64)
    high = low + np.uint64(3)
    result = generator.integers(low, high, size=(100, 2), dtype=np.uint64)
    assert np.all((result >= low) & (result < high))
    result = generator.integers(-(2**63), np.array([2**63], dtype=np.uint64), size=2048, dtype=np.int64)
    assert np.any(result < 0) and np.any(result >= 0)


def test_uint8_full_range_fills_tails_without_overwriting_guards():
    generator = nk.Generator(137, array_factory=np.empty)
    storage = np.full(19, 0xA5, dtype=np.uint8)
    generator.integers(0, 256, size=11, dtype=np.uint8, out=storage[3:14])
    np.testing.assert_array_equal(storage[:3], np.full(3, 0xA5))
    np.testing.assert_array_equal(storage[14:], np.full(5, 0xA5))
    np.testing.assert_array_equal(
        storage[3:14], nk.Generator(137, array_factory=np.empty).integers(0, 256, size=11, dtype=np.uint8)
    )
    counts = np.bincount(generator.integers(0, 256, size=1_000_000, dtype=np.uint8), minlength=256)
    expected = 1_000_000 / 256
    assert np.sum((counts - expected) ** 2 / expected) < 370


def test_packed_integer_out_buffer_keeps_tail_guards():
    storage = np.full(19, -424242, dtype=np.int32)
    nk.Generator(137).integers(-1000, 1000, size=11, dtype=np.int32, out=storage[4:15])
    np.testing.assert_array_equal(storage[:4], np.full(4, -424242))
    np.testing.assert_array_equal(storage[15:], np.full(4, -424242))
    assert np.all((storage[4:15] >= -1000) & (storage[4:15] < 1000))


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"])
def test_integer_dtype_limits_and_endpoint(dtype):
    limits = np.iinfo(dtype)
    generator = nk.Generator(137, array_factory=np.empty)
    result = generator.integers(int(limits.min), int(limits.max) + 1, size=2048, dtype=dtype)
    midpoint = (int(limits.min) + int(limits.max) + 1) // 2
    assert np.any(result < midpoint) and np.any(result >= midpoint)
    np.testing.assert_array_equal(
        generator.integers(int(limits.max), int(limits.max), size=7, dtype=dtype, endpoint=True),
        np.full(7, limits.max, dtype=dtype),
    )
    for low, high in [(int(limits.min) - 1, 1), (0, int(limits.max) + 2)]:
        with pytest.raises((ValueError, OverflowError)):
            generator.integers(low, high, size=7, dtype=dtype)


@pytest.mark.parametrize(("dtype", "low", "high"), [(np.int32, -1000, 1000), (np.uint32, 0, 2**31 + 1)])
def test_packed_32bit_integers_remain_unbiased(dtype, low, high):
    values = np.asarray(nk.Generator(137).integers(low, high, size=200_000, dtype=dtype))
    assert np.all((values >= low) & (values < high))
    expected_mean = (low + high - 1) / 2
    expected_variance = ((high - low) ** 2 - 1) / 12
    assert abs(values.mean() - expected_mean) < 7 * np.sqrt(expected_variance / values.size)
    assert abs(values.var() - expected_variance) < 0.035 * expected_variance


def test_choice_handles_weights_strings_and_sparse_large_population():
    generator = nk.Generator(137, array_factory=np.empty)
    assert generator.choice(("r90", "r180"), p=[0, 1]) == "r180"
    population = sys.maxsize
    values = generator.choice(population, size=1000, replace=False)
    assert values.dtype == np.int64
    assert len(np.unique(values)) == len(values)
    assert np.all((values >= 0) & (values < population))
    values = generator.choice([11, 22, 33], size=2, replace=False, p=[0.5, 0, 0.5])
    assert set(values) == {11, 33}
    with pytest.raises(ValueError):
        generator.choice(3, size=3, replace=False, p=[0.5, 0, 0.5])
    frequencies = np.bincount(generator.choice(3, size=100_000, p=[0.1, 0.3, 0.6]), minlength=3) / 100_000
    np.testing.assert_allclose(frequencies, [0.1, 0.3, 0.6], atol=0.007, rtol=0)


def test_shuffle_empty_payload_does_not_advance_state():
    generator = nk.Generator(137)
    generator.shuffle(np.empty((20, 0)))
    assert generator.state == 137


@pytest.mark.parametrize("kind", ["list", "array", "rows"])
def test_shuffle_permutations_are_reproducible(kind):
    first = list(range(60)) if kind == "list" else np.arange(60).reshape((10, 6) if kind == "rows" else (60,))
    second = copy.deepcopy(first)
    generator = nk.Generator(137)
    reference = nk.Generator(137)
    assert generator.shuffle(first) is None
    reference.shuffle(second)
    np.testing.assert_array_equal(first, second)
    assert generator.state == reference.state
    if kind == "rows":
        np.testing.assert_array_equal(np.sort(first[:, 0]), np.arange(0, 60, 6))
        np.testing.assert_array_equal(np.diff(first, axis=1), np.ones((10, 5)))
    else:
        assert sorted(first) == list(range(60))
        assert list(first) != list(range(60))


@pytest.mark.parametrize(
    ("method", "kwargs", "mean", "variance"),
    [
        ("uniform", {"low": -2, "high": 4}, 1, 3),
        ("normal", {"loc": 2, "scale": 3}, 2, 9),
        ("laplace", {"loc": -1, "scale": 2}, -1, 8),
        ("beta", {"a": 0.5, "b": 0.5}, 0.5, 0.125),
        ("beta", {"a": 2, "b": 3}, 0.4, 0.04),
        ("poisson", {"lam": 2}, 2, 2),
        ("poisson", {"lam": 50}, 50, 50),
    ],
)
def test_distribution_moments(method, kwargs, mean, variance):
    result = np.asarray(getattr(nk.Generator(137), method)(size=300_000, **kwargs))
    assert np.isfinite(result).all()
    assert abs(result.mean() - mean) < 7 * np.sqrt(variance / len(result))
    assert abs(result.var() - variance) < 0.035 * variance


def test_normal_tail_beta_underflow_and_unbiased_bounded_integers():
    generator = nk.Generator(137, array_factory=np.empty)
    values = generator.standard_normal(1_000_000)
    assert 25 < np.count_nonzero(np.abs(values) > 4) < 110
    values = generator.beta(1e-200, 1e-200, size=1000)
    assert set(values) == {0, 1}
    counts = np.bincount(generator.integers(0, 7, size=350_000), minlength=7)
    assert np.sum((counts - 50_000) ** 2 / 50_000) < 40


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normal_cdf_matches_reference(dtype):
    values = np.asarray(nk.Generator(137).standard_normal(200_000, dtype=dtype))
    for cutoff in (-3, -2, -1, 0, 1, 2, 3):
        expected = 0.5 * (1 + math.erf(cutoff / math.sqrt(2)))
        error = 7 * math.sqrt(expected * (1 - expected) / len(values)) + 1e-5
        assert abs(np.count_nonzero(values <= cutoff) / len(values) - expected) < error


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.normal(scale=np.array([1.0, -1.0])),
        lambda g: g.poisson(1e20),
        lambda g: g.poisson(np.array([0.0, -1.0])),
        lambda g: g.uniform(2, 1, size=3),
        lambda g: g.beta(0, 1),
        lambda g: g.integers(3, 3),
        lambda g: g.integers(0, np.array([1, 0])),
        lambda g: g.random(-1),
        lambda g: g.normal(np.ones((2, 3)), size=(3, 2)),
    ],
)
def test_invalid_parameters_do_not_advance_state(call):
    generator = nk.Generator(137)
    with pytest.raises((ValueError, OverflowError)):
        call(generator)
    assert generator.state == 137


def test_invalid_outputs_and_factory_are_rejected_before_sampling():
    generator = nk.Generator(137)
    with pytest.raises(ValueError, match="C-contiguous"):
        generator.random(out=np.empty((2, 3)).T)
    with pytest.raises(TypeError, match="dtype"):
        generator.random(out=np.empty(3, dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        generator.random(3, out=np.empty(4))
    unaligned = memoryview(bytearray(25))[1:].cast("d")
    with pytest.raises(ValueError, match="aligned"):
        generator.random(out=unaligned)
    assert generator.state == 137
    with pytest.raises(TypeError, match="callable"):
        nk.Generator(137, array_factory=3)


@pytest.mark.parametrize(("method", "kwargs"), CASES[:-1])
@pytest.mark.skipif(
    sys.platform == "emscripten" and not getattr(getattr(sys, "_emscripten_info", None), "pthreads", False),
    reason="Emscripten build has no pthread support",
)
def test_shared_generator_serializes_complete_fills(method, kwargs):
    generator = nk.Generator(137)
    reference = nk.Generator(137)
    barrier = Barrier(4)

    def fill(_):
        barrier.wait(timeout=10)
        return np.asarray(getattr(generator, method)(size=20_000, **kwargs)).tobytes()

    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(fill, range(4)))
    expected = [np.asarray(getattr(reference, method)(size=20_000, **kwargs)).tobytes() for _ in range(4)]
    assert Counter(actual) == Counter(expected)
    assert generator.state == reference.state


@pytest.mark.skipif(sys.platform in ("emscripten", "wasi"), reason="platform has no subprocess support")
def test_native_sampling_does_not_require_numpy():
    code = """
import sys
class BlockNumpy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("NumPy unavailable")
sys.meta_path.insert(0, BlockNumpy())
import numkong as nk
g = nk.Generator(137)
for method, kwargs in [
    ('random', {}), ('normal', {}), ('laplace', {}),
    ('beta', {'a': 2, 'b': 3}), ('poisson', {}), ('integers', {'low': 9}),
]:
    assert isinstance(getattr(g, method)(size=7, **kwargs), nk.Tensor)
assert 'numpy' not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(nk.__file__).resolve().parents[1])
    subprocess.run([sys.executable, "-S", "-c", code], env=environment, check=True)
