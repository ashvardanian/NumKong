/**
 *  @file probes/serial.c
 *  @author Ash Vardanian
 *  @date July 16, 2026
 *  @brief NumKong ISA probe for serial, the always-available fallback.
 *
 *  No ISA flags, no intrinsics, no system headers, so a target with no SIMD still compiles this and
 *  then rejects the other probes on their own merits. A failure here means the probe toolchain is
 *  broken, not that the hardware lacks a feature; see @c check_probe_toolchain in setup.py.
 */
int main(void) { return 0; }
