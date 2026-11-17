"""Pytest hooks and seeding fixtures for the NumKong test suite, loaded before any test module.

File: test/conftest.py
Author: Ash Vardanian
Date: September 25, 2026
"""

from __future__ import annotations

import collections
import contextlib
import enum
import io
import random
import sys
from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest
from base import (
    BACKEND_AXES,
    MAX_CASES_PER_TEST,
    POSSIBLE_CAPABILITIES,
    SETTINGS,
    Stats,
    print_stats_report,
    stream_key,
)

import numkong as nk


if TYPE_CHECKING:
    import numpy as np


def pytest_report_header() -> list[str]:
    """Echoes every setting in the grammar it parses, where pytest prints its own header."""
    return [
        f"NumKong {nk.__version__} from {nk.__file__}",
        f"- Seed: {SETTINGS.seed}, scale: {SETTINGS.scale:g}, filter: {SETTINGS.filter.pattern or 'none'}",
        f"- Capabilities: {nk.capabilities_enabled_cpu()!r}",
        f"- In QEMU: {str(SETTINGS.in_qemu).lower()}",
        f"- Repetitions: {SETTINGS.repetitions}",
        f"- Dims: {','.join(map(str, SETTINGS.dims))}",
        f"- Dims height: {','.join(map(str, SETTINGS.dims_height))}",
        f"- Dims width: {','.join(map(str, SETTINGS.dims_width))}",
        f"- Dims depth: {','.join(map(str, SETTINGS.dims_depth))}",
        f"- Curved dims: {','.join(map(str, SETTINGS.curved_dims))}",
        f"- Sparse dims: {','.join(map(str, SETTINGS.sparse_dims))}",
        f"- Shape points: {SETTINGS.shape_points}",
        f"- Max coord angle: {SETTINGS.max_coord_angle_degrees:g}",
        f"- Expect SIMD: {str(SETTINGS.expect_simd).lower()}",
    ]


# region Shared Pytest
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Refuses a test whose cases outgrow MAX_CASES_PER_TEST, then deselects what the filter setting leaves out."""
    variants: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for item in items:
        callspec = getattr(item, "callspec", None)
        parameters = dict(callspec.params) if callspec is not None else {}
        parameters.pop("__pytest_repeat_step_number", None)
        kept = {key: value for key, value in parameters.items() if key not in BACKEND_AXES}
        variants[(item.module.__name__, getattr(item, "originalname", item.name))].add(repr(kept))
    oversized = [
        f"{module}::{name} has {len(cases)}"
        for (module, name), cases in variants.items()
        if len(cases) > MAX_CASES_PER_TEST
    ]
    if oversized:
        raise pytest.UsageError(f"over MAX_CASES_PER_TEST={MAX_CASES_PER_TEST}: " + "; ".join(oversized))
    deselected = [item for item in items if not SETTINGS.filter.selects(item.nodeid)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if SETTINGS.filter.selects(item.nodeid)]


def pytest_make_parametrize_id(config: pytest.Config, val: object, argname: str) -> str | None:
    """Names enum parameters as the code spells them, `Backend.LINUX_URING618`, so an id greps to its source."""
    return f"{type(val).__name__}.{val.name}" if isinstance(val, enum.Enum) else None


@pytest.fixture
def seed(request: pytest.FixtureRequest) -> int:
    """Share inputs across backends while keeping other parameters and repeat steps distinct."""
    callspec = getattr(request.node, "callspec", None)
    parameters = dict(callspec.params) if callspec is not None else {}
    repeat_step = parameters.pop("__pytest_repeat_step_number", 0)
    name = request.node.originalname or request.node.name
    name += repr({key: value for key, value in parameters.items() if key not in BACKEND_AXES})
    return stream_key(SETTINGS.seed, name, repeat_step)


@pytest.fixture
def rng(seed: int) -> random.Random:
    """A generator private to this test, so a neighbour's draws cannot shift this one's."""
    return random.Random(seed)


@pytest.fixture
def unraisable(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Collects what the interpreter reports as unraisable, like a finalizer's exception, for the test to check."""
    seen: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", seen.append)
    return seen


# endregion Shared Pytest


@pytest.fixture
def np_rng(seed: int) -> np.random.Generator:
    """A NumPy generator private to this test, so a neighbour's draws cannot shift this one's."""
    numpy = pytest.importorskip("numpy", reason="NumPy is not installed")
    generator: np.random.Generator = numpy.random.default_rng(seed)
    return generator


@pytest.fixture
def cupy():
    """CuPy with a CUDA device it can run kernels on, for the DLPack device tests, or a skip naming what is missing."""
    cupy = pytest.importorskip("cupy", reason="CuPy is not installed")
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:
        pytest.skip(f"no CUDA device: {exc}")
    if device_count == 0:
        pytest.skip("no CUDA device")
    # CuPy wheels target specific compute capabilities and runtimes; a mismatch raises
    # CUDA_ERROR_INVALID_IMAGE
    try:
        cupy.arange(4, dtype=cupy.float32) + 1
    except Exception as exc:
        pytest.skip(f"CuPy kernels are refused on this machine: {exc}")
    return cupy


@pytest.fixture
def pinned_host_floats(cupy) -> np.ndarray:
    """Sixteen ascending float32 values in CuPy's pinned host memory, through a NumPy view."""
    numpy = pytest.importorskip("numpy", reason="NumPy is not installed")
    try:
        pinned = cupy.cuda.alloc_pinned_memory(16 * 4)
    except (AttributeError, RuntimeError) as exc:
        pytest.skip(f"CuPy pinned memory is refused on this machine: {exc}")
    view: np.ndarray = numpy.frombuffer(pinned, dtype=numpy.float32, count=16)
    view[:] = numpy.arange(16, dtype=numpy.float32)
    return view


@pytest.fixture
def pyarrow():
    """PyArrow with a DLPack producer on its arrays, which it gained in 15.0."""
    pyarrow = pytest.importorskip("pyarrow", reason="PyArrow is not installed")
    if not hasattr(pyarrow.array([1]), "__dlpack__"):
        pytest.skip("needs PyArrow 15 or newer for DLPack")
    return pyarrow


@pytest.fixture
def onnxruntime():
    """ONNX Runtime with `OrtValue.to_dlpack`, which builds before microsoft/onnxruntime#23110 lack."""
    onnxruntime = pytest.importorskip("onnxruntime", reason="ONNX Runtime is not installed")
    if not hasattr(onnxruntime.OrtValue, "to_dlpack"):
        pytest.skip("needs ONNX Runtime with DLPack helpers")
    return onnxruntime


@pytest.fixture(params=POSSIBLE_CAPABILITIES, ids=lambda name: name if name == "serial" else name + "+serial")
def capabilities(request: pytest.FixtureRequest) -> nk.Capability:
    """A per-call CPU mask, each compiled capability with a serial fallback for unsupported operation/type pairs."""
    return nk.Capability[request.param.upper()] | nk.Capability.SERIAL


_stats_reports = pytest.StashKey[dict[str, Stats]]()


@pytest.fixture
def stats(request: pytest.FixtureRequest) -> Generator[Stats, None, None]:
    """The error rows of this test, folded into its module's report once the test ends."""
    reports = request.config.stash.setdefault(_stats_reports, {})
    item_stats = Stats(test_name=request.node.nodeid)
    yield item_stats
    report = reports.setdefault(request.node.module.__name__, Stats())
    report.rows.extend(item_stats.rows)
    report.warnings.extend(item_stats.warnings)


def pytest_terminal_summary(terminalreporter, config: pytest.Config) -> None:
    """Prints each module's error and speed-up report after pytest's own summary."""
    for module, report in sorted(config.stash.get(_stats_reports, {}).items()):
        if not report.rows and not report.warnings:
            continue
        terminalreporter.section(module)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_stats_report(report)
        terminalreporter.write(output.getvalue())
