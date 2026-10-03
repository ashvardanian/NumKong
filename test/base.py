#!/usr/bin/env python3
"""Shared test infrastructure for the NumKong test suite.

Provides random data generators, capability detection, high-precision baselines
via ``precise_decimal()``, and assertion helpers. Mirrors ``harness.hpp`` from the
C++ test suite.

File: test/base.py
Author: Ash Vardanian
Date: February 27, 2026
"""

from __future__ import annotations

import array
import collections
import contextlib
import decimal
import faulthandler
import importlib.util
import io
import itertools
import math
import os
import random
import re
import secrets
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, NewType, TypeVar

import numkong as nk


if TYPE_CHECKING:
    import numpy as np  # static-analysis-only; the runtime try/except below is authoritative

faulthandler.enable()

T = TypeVar("T")

Seed = NewType("Seed", int)  # 32-bit; derive streams from it with `stream_key`, never add to it
StreamKey = NewType("StreamKey", int)  # 64-bit, from `stream_key`; seeds one test's generators
Threads = NewType("Threads", int)  # resolved count, never 0: `parse_threads` maps 0 to all cores
Bytes = NewType("Bytes", int)  # a size in bytes, never an element count
Nanoseconds = NewType("Nanoseconds", int)  # a `time.perf_counter_ns` difference

DType = Literal[
    "float64", "float32", "float16", "bfloat16", "bf16", "e5m2", "e4m3", "e3m2", "e2m3", "e2m1",
    "int64", "int32", "int16", "int8", "int4", "uint64", "uint32", "uint16", "uint8", "uint4", "uint1",
    "complex128", "complex64",
]  # fmt: skip
BufferDType = Literal["float64", "float32", "int8", "uint8"]


def env_text(name: str) -> str | None:
    """Reads `name`, or `None` when it is unset or empty."""
    return os.environ.get(name) or None


def env_parsed(name: str, fallback: T, parse: Callable[[str], T | None], expected: str) -> T:
    """Reads `name` through `parse`, or `fallback` when unset or empty; exits with status 1 on bad text."""
    text = env_text(name)
    if text is None:
        return fallback
    try:
        parsed = parse(text)
    except (ValueError, TypeError):
        parsed = None
    if parsed is None:
        raise SystemExit(f'{name}="{text}" does not parse, expected {expected}')
    return parsed


def env_count(name: str, fallback: int) -> int:
    """Reads a positive count like `128`, or `fallback` when unset or empty."""
    return env_parsed(name, fallback, parse_count, "a positive count")


def env_flag(name: str, fallback: bool) -> bool:
    """Reads `0`, `1`, `true` or `false`, or `fallback` when unset or empty."""
    return env_parsed(name, fallback, {"0": False, "false": False, "1": True, "true": True}.get, "0, 1, true or false")


def env_seed(name: str, fallback: Seed) -> Seed:
    """Reads an unsigned 32-bit seed or `random`, or `fallback` when unset or empty."""
    return env_parsed(name, fallback, parse_seed, "an unsigned integer or random")


def parse_count(text: str) -> int | None:
    """Parses a positive whole number in ASCII digits, like `128`; zero is `None`."""
    digits = re.fullmatch(r"[0-9]+", text) is not None
    return (int(text) or None) if digits else None


def parse_dims(text: str) -> list[int] | None:
    """Parses one count like `128` or a comma list like `64,128,256,512`."""
    counts = [count for piece in text.split(",") if (count := parse_count(piece)) is not None]
    return counts if len(counts) == text.count(",") + 1 else None


def parse_seed(text: str) -> Seed | None:
    """Parses an unsigned 32-bit integer like `42`, or `random` to draw one from system entropy."""
    if text == "random":
        return Seed(secrets.randbits(32))
    digits = re.fullmatch(r"[0-9]+", text) is not None
    return Seed(int(text)) if digits and int(text) < 2**32 else None


def parse_angle_degrees(text: str) -> float | None:
    """Parses an angle in degrees within `[0, 180]`, like the C++ harness."""
    angle = float(text)
    return angle if 0 <= angle <= 180 else None


_UINT64_MASK = 2**64 - 1


def mix(value: int) -> int:
    """SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits."""
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def stream_key(seed: Seed, name: str) -> StreamKey:
    """The key of stream `name`, `mix(mix(seed ^ fnv1a64(name)))`, bit-identical to C++ `stream_key`."""
    hashed = 0xCBF29CE484222325
    for byte in name.encode():
        hashed = ((hashed ^ byte) * 0x100000001B3) & _UINT64_MASK
    return StreamKey(mix(mix(seed ^ hashed)))


@dataclass(frozen=True)
class Settings:
    """Every `NUMKONG_*` variable the Python suite reads, parsed once at import."""

    seed: Seed
    filter: str
    filter_pattern: re.Pattern[str] | None
    in_qemu: bool
    repetitions: int
    dims: list[int]
    dims_height: list[int]
    dims_width: list[int]
    dims_depth: list[int]
    curved_dims: list[int]
    sparse_dims: list[int]
    mesh_points: int
    max_coord_angle_degrees: float
    expect_simd: bool

    @property
    def reduced_repetitions(self) -> int:
        """A fifth of `repetitions`, for the costlier matrix and mesh sweeps."""
        return max(1, self.repetitions // 5)

    def selects(self, name: str) -> bool:
        """Whether `NUMKONG_FILTER` selects the test `name`, as a regex or else as a substring."""
        return bool(self.filter_pattern.search(name)) if self.filter_pattern else self.filter in name


def read_settings() -> Settings:
    """Reads every `NUMKONG_*` variable, exiting with status 1 on the first that does not parse."""
    seed = env_seed("NUMKONG_SEED", Seed(42))
    filter = env_text("NUMKONG_FILTER") or ""
    try:
        filter_pattern = re.compile(filter) if filter else None
    except re.error:
        filter_pattern = None
    in_qemu = env_flag("NUMKONG_IN_QEMU", False)
    repetitions = env_count("NUMKONG_REPETITIONS", 3 if in_qemu else 10)
    # Simple cases first, then tiny ones, then the corners around common sizes.
    wide = [4, 8, 16, 64, 128, 1, 2, 3, 5, 7, 9, 15, 17, 31, 32, 33, 63, 65, 97]
    dims = env_parsed(
        "NUMKONG_DIMS", [1, 4, 16, 33, 64] if in_qemu else wide, parse_dims, "positive counts like 64,128"
    )

    def sampled(name: str, sample_key: int, count: int) -> list[int]:
        fallback = sorted(random.Random(sample_key).sample(dims, min(count, len(dims))))
        return env_parsed(name, fallback, parse_dims, "positive counts like 64,128")

    return Settings(
        seed=seed,
        filter=filter,
        filter_pattern=filter_pattern,
        in_qemu=in_qemu,
        repetitions=repetitions,
        dims=dims,
        # Deterministic subsamples for the `[rows, columns, depth]` axes, named as in `dots.h`.
        dims_height=sampled("NUMKONG_DIMS_HEIGHT", 42, 6),
        dims_width=sampled("NUMKONG_DIMS_WIDTH", 43, 6),
        dims_depth=sampled("NUMKONG_DIMS_DEPTH", 44, 6),
        curved_dims=sampled("NUMKONG_CURVED_DIMS", 45, 5),
        sparse_dims=env_parsed("NUMKONG_SPARSE_DIMS", [256], parse_dims, "positive counts like 64,128"),
        mesh_points=env_count("NUMKONG_MESH_POINTS", 1000),
        max_coord_angle_degrees=env_parsed(
            "NUMKONG_MAX_COORD_ANGLE", 180.0, parse_angle_degrees, "a number in [0, 180]"
        ),
        expect_simd=env_flag("NUMKONG_EXPECT_SIMD", True),
    )


SETTINGS = read_settings()

try:
    import numpy as np

    numpy_available = True
except Exception:
    numpy_available = False

scipy_available = importlib.util.find_spec("scipy") is not None
ml_dtypes_available = importlib.util.find_spec("ml_dtypes") is not None


NUMKONG_RTOL = 0.1
NUMKONG_ATOL = 0.1

NATIVE_COMPUTE_DTYPE: dict[str, type[Any]] = (
    {
        "float64": np.float64,
        "float32": np.float32,
        "bfloat16": np.float32,
        "bf16": np.float32,
        "float16": np.float32,
        "e5m2": np.float32,
        "e4m3": np.float32,
        "e3m2": np.float32,
        "e2m3": np.float32,
        "e2m1": np.float32,
        "int64": np.int64,
        "int32": np.int64,
        "int16": np.int64,
        "int8": np.int64,
        "int4": np.int64,
        "uint64": np.int64,
        "uint32": np.int64,
        "uint16": np.int64,
        "uint8": np.int64,
        "uint4": np.int64,
        "complex128": np.complex128,
        "complex64": np.complex128,
    }
    if numpy_available
    else {}
)
"""Map dtype → the NumPy dtype used for "native precision" baseline computation. f64 types compute at
f64; everything else at f32 for floats, or i64 for ints.
"""


PACKING_GRANULARITY: dict[str, int] = {
    "e2m1": 2,
    "int4": 2,
    "uint4": 2,
    "uint1": 8,
}


def round_up_to(value: int, multiple: int) -> int:
    """Round *value* up to the nearest multiple of *multiple*."""
    return (value + multiple - 1) // multiple * multiple


_DTYPES_NEEDING_DECIMAL = {"float32", "float64", "complex64", "complex128"}


@contextlib.contextmanager
def precise_decimal(
    dtype: str | None = None,
) -> Generator[tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]], None, None]:
    """Yield ``(upcast, sqrt, ln)`` helpers for high-precision baselines.

    When *dtype* is a small type (float32, float16, int8, …) native ``float``
    already exceeds its precision, so we skip the Decimal overhead. For
    float64/complex128 we use 120-digit Decimal arithmetic.

    Usage::

        with precise_decimal("float32") as (upcast, sqrt, ln):
            total = upcast(0)
            for x, y in zip(a, b):
                total += upcast(x) * upcast(y)
            return float(sqrt(total))
    """
    if dtype is not None and dtype not in _DTYPES_NEEDING_DECIMAL:
        yield float, math.sqrt, math.log
    else:
        ctx = decimal.Context(prec=120)
        ctx.traps[decimal.InvalidOperation] = False
        with decimal.localcontext(ctx):
            yield decimal.Decimal.from_float, decimal.Decimal.sqrt, decimal.Decimal.ln


def timed_call(function: Callable[..., T] | None, *args: Any, **kwargs: Any) -> tuple[Nanoseconds, T | None]:
    """Calls `function` once, returning its duration and result, or `(0, None)` without one."""
    if function is None:
        return Nanoseconds(0), None
    started = time.perf_counter_ns()
    result = function(*args, **kwargs)
    return Nanoseconds(time.perf_counter_ns() - started), result


def scipy_metric_name(metric: str) -> str:
    """Convert NumKong metric names to SciPy equivalents."""
    if metric == "angular":
        return "cosine"
    return metric


def to_array(x: Any, dtype: str | None = None) -> np.ndarray:
    """Copies `x` into a NumPy array, cast to `dtype` when one is given."""
    array = np.array(x)
    return array if dtype is None else array.astype(dtype)


_DTYPE_TOLERANCES: dict[str, tuple[float, float]] = {
    "float64": (1e-6, 1e-6),
    "float32": (1e-4, 1e-4),
    "bfloat16": (NUMKONG_ATOL, NUMKONG_RTOL),
    "bf16": (NUMKONG_ATOL, NUMKONG_RTOL),
    "float16": (NUMKONG_ATOL, NUMKONG_RTOL),
    "e5m2": (NUMKONG_ATOL, NUMKONG_RTOL),
    "e4m3": (NUMKONG_ATOL, NUMKONG_RTOL),
    "e3m2": (NUMKONG_ATOL, NUMKONG_RTOL),
    "e2m3": (NUMKONG_ATOL, NUMKONG_RTOL),
    "e2m1": (NUMKONG_ATOL, NUMKONG_RTOL),
    "int64": (1, 0),
    "int32": (1, 0),
    "int16": (1, 0),
    "int8": (1, 0),
    "int4": (1, 0),
    "uint64": (1, 0),
    "uint32": (1, 0),
    "uint16": (1, 0),
    "uint8": (1, 0),
    "uint4": (1, 0),
    "complex128": (1e-6, 1e-6),
    "complex64": (1e-4, 1e-4),
}


def tolerances_for_dtype(dtype: str) -> tuple[float, float]:
    """Returns ``(atol, rtol)`` appropriate for assertions on the given dtype."""
    return _DTYPE_TOLERANCES.get(dtype, (NUMKONG_ATOL, NUMKONG_RTOL))


class LazyFormat:
    """Deferred string formatting — only evaluated when str() is called (on assertion failure)."""

    __slots__ = ("_fn",)

    def __init__(self, fn: Callable[[], str]) -> None:
        self._fn = fn

    def __str__(self) -> str:
        return self._fn()


def f32_downcast_to_bf16(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Converts an array of 32-bit floats into 16-bit brain-floats.

    Uses IEEE 754 round-to-nearest-even (banker's rounding) to match
    ml_dtypes.bfloat16 behavior.
    """
    array = np.asarray(array, dtype=np.float32)
    u32 = array.view(np.uint32)
    lower = u32 & np.uint32(0xFFFF)
    # For exact ties (lower 16 bits == 0x8000), round to even:
    # only round up when the bf16 mantissa LSB (bit 16) is odd.
    is_tie = lower == np.uint32(0x8000)
    lsb = (u32 >> np.uint32(16)) & np.uint32(1)
    adjustment = np.where(is_tie, lsb << np.uint32(15), np.uint32(0x8000))
    rounded_u32 = (u32 + adjustment) & np.uint32(0xFFFF0000)
    array_f32_rounded = rounded_u32.view(np.float32)
    array_bf16 = np.right_shift(rounded_u32, 16).astype(np.uint16)
    return array_f32_rounded, array_bf16


def _pack_nibbles(array: Any) -> np.ndarray:
    """Pack pairs of nibbles along the last axis, preserving leading dimensions: high nibble = even index.

    The last axis must hold an even number of nibbles. Returns a uint8 array.
    """
    nibbles = np.asarray(array).astype(np.uint8) & 0x0F
    assert nibbles.shape[-1] % 2 == 0, "last axis must hold an even number of nibbles"
    return ((nibbles[..., 0::2] << 4) | nibbles[..., 1::2]).astype(np.uint8)


def i8_downcast_to_i4(array: Any) -> np.ndarray:
    """Pack signed 8-bit integers into signed 4-bit pairs (2 per byte).

    Layout matches C ``nk_i4x2_t``: high nibble = even index, low nibble = odd index.
    Input values must be in [-8, 7]. Preserves leading dimensions for 2-D+ inputs.
    """
    array = np.asarray(array, dtype=np.int8)
    assert np.all(array >= -8) and np.all(array <= 7), "values must be in [-8, 7]"
    return _pack_nibbles(array)


def u8_downcast_to_u4(array: Any) -> np.ndarray:
    """Pack unsigned 8-bit integers into unsigned 4-bit pairs (2 per byte).

    Layout matches C ``nk_u4x2_t``: high nibble = even index, low nibble = odd index.
    Input values must be in [0, 15]. Preserves leading dimensions for 2-D+ inputs.
    """
    array = np.asarray(array, dtype=np.uint8)
    assert np.all(array <= 15), "values must be in [0, 15]"
    return _pack_nibbles(array)


def e2m1_codes_to_e2m1x2(codes: Any) -> np.ndarray:
    """Pack E2M1 nibble codes along the last axis, matching C ``nk_e2m1x2_t``: high nibble = even index."""
    return _pack_nibbles(codes)


def hex_array(arr: Any) -> str:
    """Converts numerical array into a string of comma-separated hexadecimal values for debugging."""
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.integer):
        # View non-integer data as raw bytes for hex display
        shape = arr.shape
        arr = arr.view(np.uint8).reshape((*shape, -1))
    printer = np.vectorize(hex)
    strings = printer(arr)
    if strings.ndim == 1:
        return ", ".join(strings)
    else:
        return "\n".join(", ".join(row) for row in strings.reshape(-1, strings.shape[-1]))


# Lookup tables for sub-byte float types.
# Built once from the encoding rules in ``include/numkong/cast/serial.h``.
# Each table maps a raw byte value to its float64 representation.
# NaN entries are stored as ``float('nan')``.


class SpecialValues(Enum):
    """What a sub-byte float's top exponent encodes."""

    FINITE = "finite"  # every code is a number: e2m1, e2m3, e3m2
    IEEE = "ieee"  # zero mantissa is ±∞, the rest NaN: e5m2
    NAN_AT_MAX_MANTISSA = "nan-at-max-mantissa"  # only the all-ones code is NaN: e4m3


def build_subbyte_float_lookup_table(
    sign_bit: int, exp_bits: int, mant_bits: int, bias: int, total_bits: int, special: SpecialValues
) -> list[float]:
    """Build a byte→float64 lookup table for a sub-byte float format.

    Args:
        sign_bit: position of the sign bit (counting from bit 0)
        exp_bits: number of exponent bits
        mant_bits: number of mantissa bits
        bias: exponent bias
        total_bits: number of significant bits (6 for float6, 8 for float8)
        special: what the top exponent encodes
    """
    exp_mask = (1 << exp_bits) - 1
    mant_mask = (1 << mant_bits) - 1
    lut = [0.0] * (1 << total_bits)
    for i in range(len(lut)):
        sign = (i >> sign_bit) & 1
        exp = (i >> mant_bits) & exp_mask
        mant = i & mant_mask
        if exp == exp_mask and special is SpecialValues.IEEE:
            lut[i] = (-math.inf if sign else math.inf) if mant == 0 else math.nan
            continue
        if exp == exp_mask and special is SpecialValues.NAN_AT_MAX_MANTISSA and mant == mant_mask:
            lut[i] = math.nan
            continue
        if exp == 0:
            # Subnormal: value = (−1)^s × 2^(1−bias) × (mant / 2^mant_bits)
            val = (mant / (1 << mant_bits)) * (2.0 ** (1 - bias))
        else:
            val = 2.0 ** (exp - bias) * (1.0 + mant / (1 << mant_bits))
        lut[i] = -val if sign else val
    return lut


LOOKUP_TABLE_E2M3 = build_subbyte_float_lookup_table(5, 2, 3, 1, 6, SpecialValues.FINITE)
LOOKUP_TABLE_E2M1 = build_subbyte_float_lookup_table(3, 2, 1, 1, 4, SpecialValues.FINITE)
LOOKUP_TABLE_E3M2 = build_subbyte_float_lookup_table(5, 3, 2, 3, 6, SpecialValues.FINITE)
LOOKUP_TABLE_E4M3 = build_subbyte_float_lookup_table(7, 4, 3, 7, 8, SpecialValues.NAN_AT_MAX_MANTISSA)
LOOKUP_TABLE_E5M2 = build_subbyte_float_lookup_table(7, 5, 2, 15, 8, SpecialValues.IEEE)

SUBBYTE_LOOKUP_TABLES: dict[str, list[float]] = {
    "e5m2": LOOKUP_TABLE_E5M2,
    "e4m3": LOOKUP_TABLE_E4M3,
    "e3m2": LOOKUP_TABLE_E3M2,
    "e2m3": LOOKUP_TABLE_E2M3,
}


def _make_random_numpy(shape: tuple[int, ...], dtype: DType, generator: np.random.Generator) -> tuple[Any, Any]:
    """Draws `shape` values of `dtype` from `generator`, as `(raw, baseline)`."""
    if dtype in ("float64", "float32", "float16"):
        raw = generator.standard_normal(shape).astype(dtype)
        baseline = raw.astype(np.float64)
        return raw, baseline

    if dtype in ("bfloat16", "bf16"):
        f32_arr = generator.standard_normal(shape).astype(np.float32)
        f32_rounded, bf16_raw = f32_downcast_to_bf16(f32_arr)
        baseline = f32_rounded.astype(np.float64)
        return bf16_raw, baseline

    if dtype in ("int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"):
        info = np.iinfo(np.dtype(dtype))
        raw = generator.integers(info.min, info.max, size=shape, dtype=dtype)
        baseline = raw.astype(np.float64)
        return raw, baseline

    if dtype in ("complex64", "complex128"):
        raw = (generator.standard_normal(shape) + 1j * generator.standard_normal(shape)).astype(dtype)
        baseline = raw.astype(np.complex128)
        return raw, baseline

    if dtype in ("e5m2", "e4m3", "e3m2", "e2m3"):
        lut = np.array(SUBBYTE_LOOKUP_TABLES[dtype])
        # Exclude NaN/±∞ entries from random generation
        finite_mask = np.isfinite(lut)
        valid_bytes = np.where(finite_mask)[0].astype(np.uint8)
        # For types whose max representable value can cause FP32 accumulation
        # errors exceeding NUMKONG_ATOL through catastrophic cancellation, restrict
        # to values whose magnitude keeps products within FP32's reliable range.
        # Threshold: T² * eps_f32 < NUMKONG_ATOL/4
        #   =>  T = floor(sqrt(NUMKONG_ATOL / (4 * eps_f32))) ~ 458
        # Only e5m2 (max 57344) is affected; e4m3/e3m2/e2m3 are within bounds.
        eps32 = np.finfo(np.float32).eps
        mag_threshold = np.floor(np.sqrt(NUMKONG_ATOL / (4 * eps32)))
        decoded = lut[valid_bytes.astype(int)]
        magnitude_ok = np.abs(decoded) <= mag_threshold
        if not np.all(magnitude_ok):
            valid_bytes = valid_bytes[magnitude_ok]
        raw = valid_bytes[generator.integers(0, len(valid_bytes), size=shape)]
        baseline = lut[raw.astype(int)]
        return raw, baseline

    if dtype == "e2m1":
        codes = generator.integers(0, 16, size=shape).astype(np.uint8)
        baseline = np.array(LOOKUP_TABLE_E2M1)[codes.astype(int)]
        return e2m1_codes_to_e2m1x2(codes), baseline

    if dtype == "int4":
        values = generator.integers(-8, 8, size=shape, dtype=np.int8)
        baseline = values.astype(np.float64)
        raw = i8_downcast_to_i4(values)
        return raw, baseline

    if dtype == "uint4":
        values = generator.integers(0, 16, size=shape, dtype=np.uint8)
        baseline = values.astype(np.float64)
        raw = u8_downcast_to_u4(values)
        return raw, baseline

    raise ValueError(f"Unsupported dtype for make_random: {dtype}")


def make_random(shape: int | tuple[int, ...], dtype: DType, generator: np.random.Generator) -> tuple[Any, Any]:
    """Unified random-data factory, drawing from the test's `np_rng`.

    Returns ``(raw, baseline)`` where:

    - *raw*: data in the dtype's storage format, suitable for SIMD kernels.
    - *baseline*: ``float64`` (or ``complex128``) array for reference comparison.

    For exotic types the raw array uses a NumPy-native storage dtype
    (``uint16`` for bf16, ``uint8`` for float8/float6).
    """
    if isinstance(shape, int):
        shape = (shape,)
    return _make_random_numpy(shape, dtype, generator)


def make_nk(np_arr: np.ndarray, dtype: str | None = None) -> nk.Tensor:
    """Copy a NumPy array into a NumKong tensor; packed dtypes read the bytes and count logical dimensions."""
    dtype_name: Any = str(np_arr.dtype) if dtype is None else dtype  # a NumPy name, checked by `nk` at run time
    if dtype_name in PACKING_GRANULARITY:
        return nk.Tensor(np.ascontiguousarray(np_arr), dtype=dtype_name)
    nk_arr = nk.zeros(np_arr.shape, dtype=dtype_name)
    dst = np.asarray(nk_arr)
    src = np.ascontiguousarray(np_arr)
    if dst.dtype != src.dtype:
        src = src.view(np.uint8).reshape(dst.shape)
    np.copyto(dst, src)
    return nk_arr


def downcast_f32_to_dtype(f32_arr: np.ndarray, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    """Downcast an f32 array to *dtype*, returning (raw, f64_baseline).

    For native NumPy dtypes (float16/32/64, int*), casts directly.
    For bfloat16, uses round-to-nearest bf16 truncation.
    The baseline is always derived from the *actually stored* values
    (post-quantization), not the original f32.
    """
    if dtype in ("bfloat16", "bf16"):
        f32_rounded, raw = f32_downcast_to_bf16(f32_arr)
        return raw, f32_rounded.astype(np.float64)
    raw = f32_arr.astype(dtype)
    return raw, raw.astype(np.float64)


possible_capabilities: list[str] = [
    str(capability.name).lower() for capability in nk.Capability if capability in nk.cpu_capabilities_enabled()
]
"""Must be `enabled`, not `detected`: parametrizing over a capability this CPU has but this binary
lacks silently tests the serial fallback while claiming to cover the SIMD kernel. A binary holds one
CPU capability group, serial first, so no per-architecture list is needed.
"""


@dataclass(frozen=True)
class ErrorRow:
    """One kernel call checked against its references: errors, timings, and where it ran."""

    metric: str
    dims: int
    dtype: str
    capability: str
    absolute_baseline_error: float
    relative_baseline_error: float
    absolute_nk_error: float
    relative_nk_error: float
    accurate_nanoseconds: Nanoseconds
    baseline_nanoseconds: Nanoseconds
    nk_nanoseconds: Nanoseconds


@dataclass
class Stats:
    """Every `ErrorRow` and warning a module collected, printed once at exit."""

    rows: list[ErrorRow] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)


def create_stats() -> Stats:
    """A fresh, empty collection for one module's error report."""
    return Stats()


def _infer_dtype_name(value: Any) -> str:
    """Extract a dtype name string from a NumPy array, nk.Tensor, or scalar."""
    if hasattr(value, "dtype"):
        return str(getattr(value.dtype, "name", value.dtype))
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "float64"
    return str(np.asarray(value).dtype.name)


def assert_allclose(
    actual: Any, expected: Any, atol: float | None = None, rtol: float | None = None, err_msg: str = ""
) -> None:
    """Drop-in replacement for ``np.testing.assert_allclose`` with dtype-aware defaults.

    When both *atol* and *rtol* are omitted the tolerances are inferred from
    the dtype of *actual* via :func:`tolerances_for_dtype`.
    """
    if atol is None and rtol is None:
        atol, rtol = tolerances_for_dtype(_infer_dtype_name(actual))
    actual_array, expected_array = np.asarray(actual), np.asarray(expected)
    if not np.issubdtype(actual_array.dtype, np.complexfloating):
        actual_array = actual_array.astype(float)
    if not np.issubdtype(expected_array.dtype, np.complexfloating):
        expected_array = expected_array.astype(float)
    np.testing.assert_allclose(
        actual_array, expected_array, atol=atol or 0.0, rtol=1e-7 if rtol is None else rtol, err_msg=err_msg
    )


def _compute_errors(accurate_result: Any, baseline_result: Any, nk_result: Any) -> tuple[float, float, float, float]:
    """The absolute and relative errors of the baseline and of NumKong against the accurate result."""
    accurate_result = np.asarray(accurate_result)
    eps = np.finfo(accurate_result.dtype).resolution if np.issubdtype(accurate_result.dtype, np.inexact) else 1.0
    if baseline_result is None:
        abs_bl, rel_bl = math.nan, math.nan
    else:
        abs_bl = float(np.max(np.abs(baseline_result - accurate_result)))
        rel_bl = float(np.max(np.abs(baseline_result - accurate_result) / (np.abs(accurate_result) + eps)))
    abs_nk = float(np.max(np.abs(nk_result - accurate_result)))
    rel_nk = float(np.max(np.abs(nk_result - accurate_result) / (np.abs(accurate_result) + eps)))
    return abs_bl, rel_bl, abs_nk, rel_nk


def collect_errors(
    metric: str,
    dims: int,
    dtype: str,
    accurate_result: Any,
    accurate_nanoseconds: Nanoseconds,
    baseline_result: Any,
    baseline_nanoseconds: Nanoseconds,
    nk_result: Any,
    nk_nanoseconds: Nanoseconds,
    stats: Stats,
    *,
    capability: str,
) -> None:
    """Calculates the errors of one call and adds them to `stats`."""
    errors = _compute_errors(accurate_result, baseline_result, nk_result)
    row = ErrorRow(metric, dims, dtype, capability, *errors, accurate_nanoseconds, baseline_nanoseconds, nk_nanoseconds)
    stats.rows.append(row)


def collect_warnings(message: str, stats: Stats) -> None:
    """Collects warnings for the final report."""
    full_name = os.environ.get("PYTEST_CURRENT_TEST", "unknown::unknown").split(" ")[0]
    function_name = full_name.split("::")[-1].split("[")[0]
    stats.warnings.append((function_name, message))


def format_scientific(value: float) -> str:
    """Format a float as compact scientific notation (e.g. 7.4e-5). Return '0' for exact zero."""
    if value == 0:
        return "0"
    s = f"{value:.1e}"
    if "e" not in s:
        return s  # inf, nan, etc.
    mantissa, exp = s.split("e")
    exp_sign = exp[0]
    exp_digits = exp[1:].lstrip("0") or "0"
    return f"{mantissa}e{exp_sign}{exp_digits}"


def pad_with_ansi_color(visible: str, width: int, code: str) -> str:
    """Pad visible string to width first, then wrap in ANSI so escape codes don't break alignment."""
    padded = f"{visible:<{width}}"
    return f"\033[{code}m{padded}\033[0m"


@dataclass(frozen=True)
class CapabilityRecord:
    """The mean errors and speed-up of one capability at one `(metric, dims, dtype)`."""

    metric: str
    dims: int
    dtype: str
    capability: str
    baseline_error_mean: float
    nk_error_mean: float
    speedup_mean: float


@dataclass(frozen=True)
class ReportLine:
    """One printed line of the report: the smallest or largest `dims` of a `(metric, dtype)`."""

    metric: str
    dtype: str
    dims: str
    baseline_error: float
    worst_nk_error: float
    worst_capability: str
    best_speedup: float
    best_capability: str
    best_capability_error: float


def print_stats_report(stats: Stats) -> None:
    """Print a condensed error/speedup report: two rows per (metric, dtype) showing min/max dims."""
    if not stats.rows:
        return
    # Windows consoles default to cp1252, which lacks the report's brackets.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(errors="replace")

    # Stage 1: Group rows by (metric, dims, dtype, capability) and compute per-group means.
    def group_key(row: ErrorRow) -> tuple[str, int, str, str]:
        return row.metric, row.dims, row.dtype, row.capability

    cap_records: list[CapabilityRecord] = []
    for (metric, dims, dtype, capability), group in itertools.groupby(sorted(stats.rows, key=group_key), group_key):
        rows = list(group)
        base_err_mean = sum(row.relative_baseline_error for row in rows) / len(rows)
        nk_err_mean = sum(row.relative_nk_error for row in rows) / len(rows)
        speedups = [row.baseline_nanoseconds / row.nk_nanoseconds for row in rows if row.nk_nanoseconds > 0]
        speedup_mean = sum(speedups) / len(speedups) if speedups else 0.0
        cap_records.append(CapabilityRecord(metric, dims, dtype, capability, base_err_mean, nk_err_mean, speedup_mean))

    # Stage 2: Re-aggregate by (metric, dtype): min/max dims and cross-capability aggregates.
    by_metric_dtype: dict[tuple[str, str], list[CapabilityRecord]] = collections.defaultdict(list)
    for rec in cap_records:
        by_metric_dtype[(rec.metric, rec.dtype)].append(rec)

    lines: list[ReportLine] = []
    for (metric, dtype), recs in sorted(by_metric_dtype.items()):
        all_dims = sorted({r.dims for r in recs})
        min_dims, max_dims = all_dims[0], all_dims[-1]
        single_dims = min_dims == max_dims
        target_dims = [min_dims] if single_dims else [min_dims, max_dims]

        for i, target in enumerate(target_dims):
            subset = [r for r in recs if r.dims == target]
            base_err = sum(r.baseline_error_mean for r in subset) / len(subset)
            worst_rec = max(subset, key=lambda r: r.nk_error_mean)
            best_rec = max(subset, key=lambda r: r.speedup_mean)
            if single_dims:
                dims_text = str(target)
            elif i == 0:
                dims_text = f"\u230a{target:>4}\u230b"
            else:
                dims_text = f"\u2308{target:>4}\u2309"
            lines.append(
                ReportLine(
                    metric if i == 0 else "",
                    dtype if i == 0 else "",
                    dims_text,
                    base_err,
                    worst_rec.nk_error_mean,
                    worst_rec.capability,
                    best_rec.speedup_mean,
                    best_rec.capability,
                    best_rec.nk_error_mean,
                )
            )

    # Stage 3: Render
    col_w = {"kernel": 17, "dtype": 12, "dims": 8, "base_err": 14, "worst_nk": 30, "best_spd": 34}
    header = (
        f"{'Kernel':<{col_w['kernel']}}"
        f"{'DType':<{col_w['dtype']}}"
        f"{'Dims':<{col_w['dims']}}"
        f"{'Base Error':<{col_w['base_err']}}"
        f"{'Worst NK Error':<{col_w['worst_nk']}}"
        f"{'Best NK Speedup':<{col_w['best_spd']}}"
    )
    sep = "\u2500" * len(header)
    print(f"\n\n{header}")
    print(sep)

    for line in lines:
        base_err_s = format_scientific(line.baseline_error)
        worst_nk_s = f"{format_scientific(line.worst_nk_error)} \u2039{line.worst_capability}\u203a"
        best_spd_s = f"{line.best_speedup:.1f}x \u2039{line.best_capability}, err {format_scientific(line.best_capability_error)}\u203a"

        # Color for worst NK error: red if NK error > base error
        nk_err_code = "31" if line.worst_nk_error > line.baseline_error else "0"
        # Color for speedup: green >=2x, yellow 1-2x, red <1x
        if line.best_speedup >= 2.0:
            spd_code = "32"
        elif line.best_speedup >= 1.0:
            spd_code = "33"
        else:
            spd_code = "31"

        print(
            f"{line.metric:<{col_w['kernel']}}"
            f"{line.dtype:<{col_w['dtype']}}"
            f"{line.dims:<{col_w['dims']}}"
            f"{base_err_s:<{col_w['base_err']}}"
            f"{pad_with_ansi_color(worst_nk_s, col_w['worst_nk'], nk_err_code)}"
            f"{pad_with_ansi_color(best_spd_s, col_w['best_spd'], spd_code)}"
        )

    warnings_list = [f"{name}: {message}" for name, message in sorted(stats.warnings)]
    if warnings_list:
        print("\nWarnings:")
        for warning, count in sorted(collections.Counter(warnings_list).items()):
            print(f"- {count}x times: {warning}")


ARRAY_TYPECODES: dict[BufferDType, tuple[str, float, float]] = {
    "float64": ("d", -10.0, 10.0),
    "float32": ("f", -10.0, 10.0),
    "int8": ("b", -128, 127),
    "uint8": ("B", 0, 255),
}
"""Map nk dtype to its array.array typecode along with the low and high representable values."""


def make_random_buffer(rng: random.Random, count: int, dtype: BufferDType = "float32") -> array.array[Any]:
    """A random `array.array` of `count` elements drawn from `rng`; no NumPy needed."""
    typecode, low, high = ARRAY_TYPECODES[dtype]
    if typecode in ("f", "d"):
        return array.array(typecode, [rng.uniform(low, high) for _ in range(count)])
    return array.array(typecode, [rng.randint(int(low), int(high)) for _ in range(count)])


def make_positive_buffer(rng: random.Random, count: int, dtype: BufferDType = "float32") -> array.array[float]:
    """A random distribution of `count` probabilities drawn from `rng`, as an `array.array`."""
    typecode = "f" if dtype == "float32" else "d"
    values = [rng.uniform(0.01, 1.0) for _ in range(count)]
    total = sum(values)
    return array.array(typecode, [value / total for value in values])
