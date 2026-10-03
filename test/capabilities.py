#!/usr/bin/env python3
"""Test capability reporting and narrowing: nk.{cpu,cuda,rocm,metal}_capabilities_{detected,compiled,enabled}.

Capabilities are reported along two independent axes — `detected` (what this CPU can execute)
and `compiled` (what the ISA probes baked into this build) — plus `enabled` (what dispatch uses,
their intersection, which one call can narrow by `capabilities=`).

Conflating the axes is a silent performance cliff rather than a build error, which is how
SIMD-free wheels once shipped with every check green: `detected` is true of the machine no
matter what was compiled in.

File: test/capabilities.py
Author: Ash Vardanian
Date: July 16, 2026
"""

import array
import platform
import sys

import pytest
from base import SETTINGS

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


def test_capability_members_are_the_cpu_capabilities():
    """`Capability` has one member per CPU capability, in bit order, and none for the GPU capabilities.

    A name missing or moved here means the names drifted from the `nk_cap_*_k` bits.
    """
    # fmt: off
    expected = [
        "serial",
        "neon", "neonhalf", "neonbfdot", "neonfhm", "neonsdot", "neonfp8",
        "sve", "svehalf", "svesdot", "svebfdot", "sve2", "sme", "smef64", "smebi32",
        "haswell", "alder", "sierra", "skylake", "icelake", "genoa", "turin", "sapphire", "diamond",
        "sapphireamx", "graniteamx", "diamondamx",
        "rvv", "rvvbf16", "rvvhalf", "rvvbb",
        "v128", "v128relaxed", "powervsx", "loongsonasx",
    ]
    # fmt: on
    assert list(nk.Capability.__members__) == [name.upper() for name in expected], "members follow the bit order"


def test_masks_past_this_cpu_are_clamped():
    """A call asking for every capability runs only what this CPU detects and this build compiled.

    Without the clamp in the library's lookup, an ISA that was compiled in but that this CPU lacks
    would point dispatch at instructions the hardware refuses to execute.
    """
    detected, compiled = nk.cpu_capabilities_detected(), nk.cpu_capabilities_compiled()
    a, b = array.array("f", [0.25] * 64), array.array("f", [0.5] * 64)
    assert nk.dot(a, b, capabilities=detected | compiled) == 8.0
    assert nk.cpu_capabilities_enabled() == detected & compiled
    assert nk.Capability.SERIAL in detected & compiled, "the serial fallback is always both detected and compiled in"


def test_compiled_covers_the_baseline_this_machine_detects():
    """A build whose ISA probes failed is scalar, and only `compiled` can see it.

    Skips where there is no SIMD to expect, so genuinely serial targets stay green: a 32-bit
    or exotic arch, or a CPU too old for the baseline. Set `NUMKONG_EXPECT_SIMD=0` to skip a
    deliberately scalar build on a SIMD-capable machine.
    """
    if not SETTINGS.expect_simd:
        pytest.skip("NUMKONG_EXPECT_SIMD=0: this build is deliberately scalar")

    baseline = baseline_for_this_machine()
    if baseline is None:
        pytest.skip(f"no SIMD baseline is guaranteed on {platform.machine()}")
    if baseline not in nk.cpu_capabilities_detected():
        pytest.skip(f"this CPU does not report {baseline.name}; nothing to verify")

    assert baseline in nk.cpu_capabilities_compiled(), (
        f"this CPU reports {baseline.name} but no {baseline.name} kernels were compiled in — "
        f"the ISA probes failed at build time and this build is scalar"
    )


@pytest.mark.parametrize("vendor", ["cuda", "rocm", "metal"])
def test_gpu_producers_refuse_an_ordinal_they_do_not_see(vendor: str):
    """Each vendor reports the devices it counts, and raises ValueError past them, or for any in a build without it.

    A GPU it counts synchronizes its default stream, and one it does not refuses to.
    """
    count = getattr(nk, f"{vendor}_count_devices")()
    detected = getattr(nk, f"{vendor}_capabilities_detected")
    enabled = getattr(nk, f"{vendor}_capabilities_enabled")
    compiled = getattr(nk, f"{vendor}_capabilities_compiled")()
    for ordinal in (-1, count):
        with pytest.raises(ValueError):
            enabled(ordinal)
    if count:
        assert enabled(count - 1) == detected(count - 1) & compiled
        nk.synchronize(enabled(count - 1))
    else:
        with pytest.raises(RuntimeError):
            nk.synchronize(1 << {"cuda": 48, "rocm": 56, "metal": 60}[vendor])


def test_synchronize_on_the_cpu_returns():
    """The CPU has nothing queued, so its synchronization returns at once, and a stream must be a pointer."""
    nk.synchronize(nk.cpu_capabilities_enabled())
    nk.synchronize(nk.cpu_capabilities_enabled(), stream=None)
    with pytest.raises(TypeError):
        nk.synchronize(nk.cpu_capabilities_enabled(), stream="not a pointer")


def test_capabilities_keyword_narrows_one_call():
    """`capabilities=` picks the capabilities of one call and leaves every other call alone.

    The keyword keeps no serial fallback, so a mask of no capability finds no kernel.
    """
    a, b = array.array("f", [0.25] * 64), array.array("f", [0.5] * 64)
    enabled = nk.cpu_capabilities_enabled()
    assert nk.dot(a, b, capabilities=nk.Capability.SERIAL) == nk.dot(a, b) == 8.0
    assert nk.cpu_capabilities_enabled() == enabled
    with pytest.raises(LookupError):
        nk.dot(a, b, capabilities=nk.Capability(0))
    with pytest.raises(TypeError):
        nk.dot(a, b, stream="not a pointer")


def test_packed_matrix_keeps_the_mask_it_was_packed_with():
    """A packed matrix is read by the capability that packed it, whatever mask later calls default to.

    Pack layouts differ per capability, so another capability's kernel refuses the buffer rather than misreading it.
    """
    vectors = memoryview(array.array("f", [float(i % 7) for i in range(8 * 64)])).cast("B").cast("f", [8, 64])
    packed = nk.dots_pack(vectors)
    expected = nk.dots_packed(vectors, packed)
    serial_packed = nk.dots_pack(vectors, capabilities=nk.Capability.SERIAL)
    assert nk.dots_packed(vectors, packed) == expected
    assert nk.dots_packed(vectors, serial_packed) == expected
    if serial_packed.nbytes != packed.nbytes:
        with pytest.raises(ValueError):
            nk.dots_packed(vectors, packed, capabilities=nk.Capability.SERIAL)
