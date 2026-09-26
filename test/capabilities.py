#!/usr/bin/env python3
"""Test CPU capability reporting and narrowing: nk.capabilities_{detected,compiled,enabled,enable}.

Capabilities are reported along two independent axes — `detected` (what this CPU can execute)
and `compiled` (what the ISA probes baked into this build) — plus `enabled` (what dispatch uses,
their intersection unless narrowed by `capabilities_enable`).

Conflating the axes is a silent performance cliff rather than a build error, which is how
SIMD-free wheels once shipped with every check green: `detected` is true of the machine no
matter what was compiled in.

File: test/capabilities.py
Author: Ash Vardanian
Date: July 16, 2026
"""

import os
import platform
import sys

import pytest

import numkong as nk


BASELINE_BY_MACHINE: dict[tuple[str, ...], nk.Capability] = {
    ("x86_64", "amd64", "x64"): nk.Capability.HASWELL,
    ("arm64", "aarch64"): nk.Capability.NEON,
}
"""The ISA every supported toolchain emits for a given 64-bit architecture. A machine that detects
one of these but did not compile it in has a broken probe, not a slow CPU.
"""


def baseline_for_this_machine() -> nk.Capability | None:
    """The ISA this machine is expected to carry, or None where none is guaranteed."""
    if not sys.maxsize > 2**32:
        return None  # 32-bit targets (i686, armv7) have no guaranteed baseline
    machine = platform.machine().lower()
    for names, baseline in BASELINE_BY_MACHINE.items():
        if machine in names:
            return baseline
    return None


@pytest.fixture(autouse=True)
def restore_enabled_capabilities():
    """Restores the enabled set each test found, which `keep_one_capability` caches across tests."""
    enabled = nk.capabilities_enabled()
    yield
    nk.capabilities_enable(enabled)


def test_capability_members_are_the_cpu_tiers():
    """`Capability` has one member per CPU tier and none for the GPU tiers.

    A name missing here means the names drifted from the `nk_cap_*_k` bits.
    """
    # fmt: off
    expected = [
        "serial",
        "haswell", "alder", "sierra",
        "skylake", "icelake", "genoa", "sapphire", "turin", "diamond",
        "sapphireamx", "graniteamx", "diamondamx",
        "neon", "neonhalf", "neonfhm", "neonbfdot", "neonsdot", "neonfp8",
        "sve", "svehalf", "svebfdot", "svesdot", "sve2", "sve2p1",
        "sme", "sme2", "sme2p1", "smef64", "smehalf", "smebf16", "smebi32", "smelut2", "smefa64",
        "rvv", "rvvhalf", "rvvbf16", "rvvbb",
        "loongsonasx", "powervsx", "v128", "v128relaxed",
    ]
    # fmt: on
    assert sorted(nk.Capability.__members__) == sorted(name.upper() for name in expected)


def test_enabling_everything_keeps_what_runs_here():
    """Asking for every tier leaves exactly the ones both detected and compiled, serial included.

    Without the clamp, enabling an ISA that was compiled in but that this CPU lacks points
    dispatch at instructions the hardware refuses to execute.
    """
    detected, compiled = nk.capabilities_detected(), nk.capabilities_compiled()
    enabled = nk.capabilities_enable(detected | compiled)
    assert enabled == detected & compiled == nk.capabilities_enabled()
    assert nk.Capability.SERIAL in enabled, "the serial fallback is always both detected and compiled in"


def test_compiled_covers_the_baseline_this_machine_detects():
    """A build whose ISA probes failed is scalar, and only `compiled` can see it.

    Skips where there is no SIMD to expect, so genuinely serial targets stay green: a 32-bit
    or exotic arch, or a CPU too old for the baseline. Set `NUMKONG_EXPECT_SIMD=0` to skip a
    deliberately scalar build on a SIMD-capable machine.
    """
    if os.environ.get("NUMKONG_EXPECT_SIMD") == "0":
        pytest.skip("NUMKONG_EXPECT_SIMD=0: this build is deliberately scalar")

    baseline = baseline_for_this_machine()
    if baseline is None:
        pytest.skip(f"no SIMD baseline is guaranteed on {platform.machine()}")
    if baseline not in nk.capabilities_detected():
        pytest.skip(f"this CPU does not report {baseline.name}; nothing to verify")

    assert baseline in nk.capabilities_compiled(), (
        f"this CPU reports {baseline.name} but no {baseline.name} kernels were compiled in — "
        f"the ISA probes failed at build time and this build is scalar"
    )


def test_enable_drops_the_tiers_left_out():
    """`capabilities_enable` makes `wanted` the enabled set, so a tier left out stops dispatching."""
    available = nk.capabilities_detected() & nk.capabilities_compiled()
    tiers = [tier for tier in nk.Capability if tier in available and tier != nk.Capability.SERIAL]
    if not tiers:
        pytest.skip("scalar build: no tier other than serial to toggle")

    enabled = nk.capabilities_enable(available ^ tiers[0])
    assert tiers[0] not in enabled and enabled == nk.capabilities_enabled()
    assert nk.capabilities_enable(available) == available


def test_serial_survives_enabling_nothing():
    """The serial fallback always remains, so a kernel is always found."""
    assert nk.capabilities_enable(nk.Capability(0)) == nk.Capability.SERIAL
