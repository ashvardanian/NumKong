#!/usr/bin/env python3
"""Test SIMD capability reporting: nk.get_capabilities_{detected,compiled,available,enabled}.

Capabilities are reported along two independent axes — `detected` (what this CPU can execute)
and `compiled` (what the ISA probes baked into this build) — plus `available` (their
intersection) and `enabled` (the subset dispatch is restricted to).

Conflating the axes is a silent performance cliff rather than a build error, which is how
SIMD-free wheels once shipped with every check green: `detected` is true of the machine no
matter what was compiled in.
"""

import platform
import sys

import pytest

import numkong as nk


#: The ISA every supported toolchain emits for a given 64-bit architecture. A machine that
#: detects one of these but did not compile it in has a broken probe, not a slow CPU.
BASELINE_BY_MACHINE: dict[tuple[str, ...], str] = {
    ("x86_64", "amd64", "x64"): "haswell",
    ("arm64", "aarch64"): "neon",
}


def enabled_names(capabilities: dict[str, bool]) -> set[str]:
    """Names of the capabilities set to True."""
    return {name for name, on in capabilities.items() if on}


def baseline_for_this_machine() -> str | None:
    """The ISA this machine is expected to carry, or None where none is guaranteed."""
    if not sys.maxsize > 2**32:
        return None  # 32-bit targets (i686, armv7) have no guaranteed baseline
    machine = platform.machine().lower()
    for names, baseline in BASELINE_BY_MACHINE.items():
        if machine in names:
            return baseline
    return None


def test_capability_names_are_complete():
    """Every accessor reports the same key set.

    A name missing here means the binding's name table drifted from the `nk_cap_*_k` bits.
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
        "loongsonasx", "powervsx", "v128relaxed",
    ]
    # fmt: on
    accessors = (
        nk.get_capabilities_detected,
        nk.get_capabilities_compiled,
        nk.get_capabilities_available,
        nk.get_capabilities_enabled,
    )
    for accessor in accessors:
        reported = accessor()
        for name in expected:
            assert name in reported, f"'{name}' missing from {accessor.__name__}()"


def test_axes_are_independent_and_derived_sets_follow():
    """`available` is exactly the intersection, and `enabled` never escapes it."""
    detected = enabled_names(nk.get_capabilities_detected())
    compiled = enabled_names(nk.get_capabilities_compiled())
    available = enabled_names(nk.get_capabilities_available())
    enabled = enabled_names(nk.get_capabilities_enabled())

    assert available == detected & compiled, "available must be exactly detected & compiled"
    assert enabled <= available, f"dispatch may not reach unavailable kernels: {enabled - available}"
    assert "serial" in available, "the serial fallback is always both detected and compiled in"


def test_compiled_covers_the_baseline_this_machine_detects():
    """A build whose ISA probes failed is scalar, and only `compiled` can see it.

    Skips where there is no SIMD to expect, so genuinely serial targets stay green: a 32-bit
    or exotic arch, or a CPU too old for the baseline. Set `NK_EXPECT_SIMD=0` to skip a
    deliberately scalar build on a SIMD-capable machine.
    """
    import os

    if os.environ.get("NK_EXPECT_SIMD") == "0":
        pytest.skip("NK_EXPECT_SIMD=0: this build is deliberately scalar")

    baseline = baseline_for_this_machine()
    if baseline is None:
        pytest.skip(f"no SIMD baseline is guaranteed on {platform.machine()}")
    if not nk.get_capabilities_detected().get(baseline):
        pytest.skip(f"this CPU does not report {baseline}; nothing to verify")

    assert nk.get_capabilities_compiled().get(baseline), (
        f"this CPU reports {baseline} but no {baseline} kernels were compiled in — "
        f"the ISA probes failed at build time and this build is scalar"
    )


def test_enable_and_disable_move_a_capability_in_and_out():
    """`enable` and `disable` move a capability across the enabled set."""
    candidates = sorted(enabled_names(nk.get_capabilities_available()) - {"serial"})
    if not candidates:
        pytest.skip("scalar build: no capability other than serial to toggle")

    capability = candidates[0]
    nk.disable_capability(capability)
    assert capability not in enabled_names(nk.get_capabilities_enabled())
    nk.enable_capability(capability)
    assert capability in enabled_names(nk.get_capabilities_enabled())


def test_enabling_an_unavailable_capability_is_a_no_op():
    """Enabling what this build cannot run must not reach dispatch.

    Without the clamp, enabling an ISA that was compiled in but that this CPU lacks points
    dispatch at instructions the hardware refuses to execute.
    """
    available = enabled_names(nk.get_capabilities_available())
    unavailable = sorted(enabled_names(nk.get_capabilities_compiled()) - available)
    if not unavailable:
        pytest.skip("every compiled capability is available here; nothing to clamp")

    for capability in unavailable:
        nk.enable_capability(capability)
        assert capability not in enabled_names(nk.get_capabilities_enabled()), (
            f"'{capability}' is not available here, so enabling it must not reach dispatch"
        )


def test_serial_survives_disabling_everything_else():
    """The serial fallback always remains, so a kernel is always found."""
    others = sorted(enabled_names(nk.get_capabilities_available()) - {"serial"})
    try:
        for capability in others:
            nk.disable_capability(capability)
        assert "serial" in enabled_names(nk.get_capabilities_enabled())
    finally:
        for capability in others:
            nk.enable_capability(capability)
