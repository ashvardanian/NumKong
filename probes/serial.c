/* NumKong ISA probe: serial (the always-available fallback).
 * No ISA flags, no intrinsics, no system headers — so a target with no SIMD still compiles
 * this and then rejects the other probes on their own merits. A failure here means the probe
 * toolchain itself is broken, not that the hardware lacks a feature.
 * See `check_probe_toolchain` in setup.py. */
int main(void) { return 0; }
