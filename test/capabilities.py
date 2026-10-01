#!/usr/bin/env python3
"""Test capability reporting and narrowing: nk.Device and its capabilities_{detected,compiled,enabled,enable}.

Capabilities are reported along two independent axes — `detected` (what this CPU can execute)
and `compiled` (what the ISA probes baked into this build) — plus `enabled` (what dispatch uses,
their intersection unless narrowed by `capabilities_enable`, or for one call by `capabilities=`).

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


cpu = nk.Device.cpu()


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
    enabled = cpu.capabilities_enabled()
    yield
    cpu.capabilities_enable(enabled)


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


def test_enabling_everything_keeps_what_runs_here():
    """Asking for every capability leaves exactly the ones both detected and compiled, serial included.

    Without the clamp, enabling an ISA that was compiled in but that this CPU lacks points
    dispatch at instructions the hardware refuses to execute.
    """
    detected, compiled = cpu.capabilities_detected(), cpu.capabilities_compiled()
    enabled = cpu.capabilities_enable(detected | compiled)
    assert enabled == detected & compiled == cpu.capabilities_enabled()
    assert nk.Capability.SERIAL in enabled, "the serial fallback is always both detected and compiled in"


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
    if baseline not in cpu.capabilities_detected():
        pytest.skip(f"this CPU does not report {baseline.name}; nothing to verify")

    assert baseline in cpu.capabilities_compiled(), (
        f"this CPU reports {baseline.name} but no {baseline.name} kernels were compiled in — "
        f"the ISA probes failed at build time and this build is scalar"
    )


def test_enable_drops_the_tiers_left_out():
    """`capabilities_enable` makes `wanted` the enabled set, so a capability left out stops dispatching."""
    available = cpu.capabilities_detected() & cpu.capabilities_compiled()
    capabilities = [
        capability for capability in nk.Capability if capability in available and capability != nk.Capability.SERIAL
    ]
    if not capabilities:
        pytest.skip("scalar build: no capability other than serial to toggle")

    enabled = cpu.capabilities_enable(available ^ capabilities[0])
    assert capabilities[0] not in enabled and enabled == cpu.capabilities_enabled()
    assert cpu.capabilities_enable(available) == available


def test_serial_survives_enabling_nothing():
    """The serial fallback always remains, so a kernel is always found."""
    assert cpu.capabilities_enable(nk.Capability(0)) == nk.Capability.SERIAL


def test_device_names_one_device_it_sees():
    """A `Device` is a kind and an ordinal, compared by value, and refuses an ordinal past the ones this process sees."""
    assert cpu == nk.Device("cpu") == nk.Device(kind="cpu", ordinal=0) and hash(cpu) == hash(nk.Device("cpu"))
    assert (cpu.kind, cpu.ordinal, repr(cpu)) == ("cpu", 0, "Device('cpu', 0)")
    assert nk.zeros((2,), dtype="float32").device == cpu, "host tensors live on the CPU"
    assert nk.Device.count("cpu") == 1
    for kind in ("cpu", "cuda", "rocm", "metal"):
        count = nk.Device.count(kind)
        for ordinal in (-1, count):
            with pytest.raises(ValueError):
                nk.Device(kind, ordinal)
        if count and kind != "cpu":
            device = nk.Device(kind, count - 1)
            assert device.capabilities_enabled() == device.capabilities_detected() & device.capabilities_compiled()
    with pytest.raises(ValueError):
        nk.Device("tpu")


def test_capabilities_keyword_narrows_one_call():
    """`capabilities=` picks the capabilities of one call and leaves the default of every other call alone.

    Unlike `capabilities_enable`, the keyword keeps no serial fallback, so a mask of no capability finds no kernel.
    """
    a, b = array.array("f", [0.25] * 64), array.array("f", [0.5] * 64)
    enabled = cpu.capabilities_enabled()
    assert nk.dot(a, b, capabilities=nk.Capability.SERIAL) == nk.dot(a, b) == 8.0
    assert cpu.capabilities_enabled() == enabled
    with pytest.raises(LookupError):
        nk.dot(a, b, capabilities=nk.Capability(0))
    with pytest.raises(TypeError):
        nk.dot(a, b, stream="not a pointer")


def test_packed_matrix_keeps_the_mask_it_was_packed_with():
    """A packed matrix is read by the capability that packed it, even after the default narrows.

    Pack layouts differ per capability, so another capability's kernel refuses the buffer rather than misreading it.
    """
    vectors = memoryview(array.array("f", [float(i % 7) for i in range(8 * 64)])).cast("B").cast("f", [8, 64])
    packed = nk.dots_pack(vectors)
    expected = nk.dots_packed(vectors, packed)
    cpu.capabilities_enable(nk.Capability.SERIAL)
    serial_packed = nk.dots_pack(vectors)
    assert nk.dots_packed(vectors, packed) == expected
    assert nk.dots_packed(vectors, serial_packed) == expected
    if serial_packed.nbytes != packed.nbytes:
        with pytest.raises(ValueError):
            nk.dots_packed(vectors, packed, capabilities=nk.Capability.SERIAL)
