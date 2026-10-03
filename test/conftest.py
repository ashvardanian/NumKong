"""Pytest hooks and seeding fixtures for the NumKong test suite, loaded before any test module.

File: test/conftest.py
Author: Ash Vardanian
Date: September 25, 2026
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest
from base import SETTINGS, StreamKey, possible_capabilities, stream_key

import numkong as nk


if TYPE_CHECKING:
    import numpy as np


def pytest_report_header() -> list[str]:
    """Echoes every setting in the grammar it parses, where pytest prints its own header."""
    return [
        f"- Seed: {SETTINGS.seed}",
        f"- Filter: {SETTINGS.filter or 'none'}",
        f"- In QEMU: {str(SETTINGS.in_qemu).lower()}",
        f"- Repetitions: {SETTINGS.repetitions}",
        f"- Dims: {','.join(map(str, SETTINGS.dims))}",
        f"- Dims height: {','.join(map(str, SETTINGS.dims_height))}",
        f"- Dims width: {','.join(map(str, SETTINGS.dims_width))}",
        f"- Dims depth: {','.join(map(str, SETTINGS.dims_depth))}",
        f"- Curved dims: {','.join(map(str, SETTINGS.curved_dims))}",
        f"- Sparse dims: {','.join(map(str, SETTINGS.sparse_dims))}",
        f"- Mesh points: {SETTINGS.mesh_points}",
        f"- Max coord angle: {SETTINGS.max_coord_angle_degrees:g}",
        f"- Expect SIMD: {str(SETTINGS.expect_simd).lower()}",
    ]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keeps only the tests whose node id `NUMKONG_FILTER` selects, on top of any `-k`."""
    deselected = [item for item in items if not SETTINGS.selects(item.nodeid)]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if SETTINGS.selects(item.nodeid)]


@pytest.fixture
def seed(request: pytest.FixtureRequest) -> StreamKey:
    """Share inputs across CPU masks while keeping other parameters and repeat steps distinct."""
    name = request.node.name
    callspec = getattr(request.node, "callspec", None)
    if callspec is not None and {"capability", "capabilities"} & callspec.params.keys():
        parameters = {key: value for key, value in callspec.params.items() if key not in {"capability", "capabilities"}}
        name = (request.node.originalname or request.node.name) + repr(parameters)
    return stream_key(SETTINGS.seed, name)


@pytest.fixture
def rng(seed: StreamKey) -> random.Random:
    """A generator private to this test, so a neighbour's draws cannot shift this one's."""
    return random.Random(seed)


@pytest.fixture
def np_rng(seed: StreamKey) -> np.random.Generator:
    """A NumPy generator private to this test, so a neighbour's draws cannot shift this one's."""
    numpy = pytest.importorskip("numpy")
    generator: np.random.Generator = numpy.random.default_rng(seed)
    return generator


@pytest.fixture(params=possible_capabilities, ids=lambda name: name if name == "serial" else name + "+serial")
def capability(request: pytest.FixtureRequest) -> str:
    """The requested CPU capability; correctness sweeps permit serial fallback."""
    return request.param


@pytest.fixture
def capabilities(capability: str) -> nk.Capability:
    """A per-call CPU mask with a serial fallback for unsupported operation/type pairs."""
    return nk.Capability[capability.upper()] | nk.Capability.SERIAL
