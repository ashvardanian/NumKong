//! Runtime CPU capability detection.
//!
//! Capability bits (`cap::*`) are probed at library load time by examining
//! CPUID / `getauxval` / HWCAP records on the host CPU; [`available`] returns the
//! resulting bitmask so downstream code can select the best kernel without
//! recompiling.
//!
//! This module provides:
//!
//! - [`available`]: Query the bitmask of supported SIMD instruction sets
//! - [`configure_thread`]: Enable optimal SIMD settings for the current thread
//! - [`uses_dynamic_dispatch`]: Check if the library selects kernels at runtime
//! - [`cap`]: Constants for individual capability bits (NEON, SKYLAKE, etc.)

#[link(name = "numkong")]
extern "C" {
    fn nk_configure_thread(capabilities: u64) -> i32;
    fn nk_uses_dynamic_dispatch() -> i32;
    fn nk_capabilities() -> u64;
}

/// Returns the bitmask of available CPU capabilities.
/// Use with `cap::*` constants to check for specific features.
///
/// # Example
/// ```
/// use numkong::{capabilities, cap};
///
/// let caps = capabilities::available();
/// if caps & cap::NEON != 0 {
///     println!("NEON is available");
/// }
/// if caps & cap::SKYLAKE != 0 {
///     println!("AVX-512 (Skylake) is available");
/// }
/// ```
pub fn available() -> u64 { unsafe { nk_capabilities() } }

/// Configures the current thread for optimal SIMD performance.
/// On x86, this enables AMX tile state via `arch_prctl`. On other platforms this is a no-op.
/// Must be called once per thread before using AMX (Advanced Matrix Extensions) operations.
pub fn configure_thread() -> bool {
    // Pass !0 to enable all capabilities including AMX
    unsafe { nk_configure_thread(!0) != 0 }
}

/// Returns `true` if the library uses dynamic dispatch for function selection.
pub fn uses_dynamic_dispatch() -> bool { unsafe { nk_uses_dynamic_dispatch() != 0 } }

/// Capability bit masks in chronological order (by first commercial silicon).
pub mod cap {
    pub const SERIAL: u64 = 1 << 0; // Always: Fallback
    pub const NEON: u64 = 1 << 1; // ARM NEON
    pub const HASWELL: u64 = 1 << 2; // Intel AVX2
    pub const SKYLAKE: u64 = 1 << 3; // Intel AVX-512
    pub const NEONHALF: u64 = 1 << 4; // ARM NEON FP16
    pub const NEONSDOT: u64 = 1 << 5; // ARM NEON i8 dot
    pub const NEONFHM: u64 = 1 << 6; // ARM NEON FP16 FML
    pub const ICELAKE: u64 = 1 << 7; // Intel AVX-512 VNNI
    pub const GENOA: u64 = 1 << 8; // AMD AVX-512 BF16
    pub const NEONBFDOT: u64 = 1 << 9; // ARM NEON BF16
    pub const SVE: u64 = 1 << 10; // ARM SVE
    pub const SVEHALF: u64 = 1 << 11; // ARM SVE FP16
    pub const SVESDOT: u64 = 1 << 12; // ARM SVE i8 dot
    pub const ALDER: u64 = 1 << 13; // Intel AVX2+VNNI
    pub const SVEBFDOT: u64 = 1 << 14; // ARM SVE BF16
    pub const SVE2: u64 = 1 << 15; // ARM SVE2
    pub const V128RELAXED: u64 = 1 << 16; // WASM Relaxed SIMD
    pub const SAPPHIRE: u64 = 1 << 17; // Intel AVX-512 FP16
    pub const SAPPHIREAMX: u64 = 1 << 18; // Intel Sapphire AMX
    pub const RVV: u64 = 1 << 19; // RISC-V Vector
    pub const RVVHALF: u64 = 1 << 20; // RISC-V Zvfh
    pub const RVVBF16: u64 = 1 << 21; // RISC-V Zvfbfwma
    pub const GRANITEAMX: u64 = 1 << 22; // Intel Granite AMX FP16
    pub const TURIN: u64 = 1 << 23; // AMD Turin AVX-512 CD
    pub const SME: u64 = 1 << 24; // ARM SME
    pub const SME2: u64 = 1 << 25; // ARM SME2
    pub const SMEF64: u64 = 1 << 26; // ARM SME F64
    pub const SMEFA64: u64 = 1 << 27; // ARM SME FA64
    pub const SVE2P1: u64 = 1 << 28; // ARM SVE2.1
    pub const SME2P1: u64 = 1 << 29; // ARM SME2.1
    pub const SMEHALF: u64 = 1 << 30; // ARM SME F16F16
    pub const SMEBF16: u64 = 1 << 31; // ARM SME B16B16
    pub const SMELUT2: u64 = 1 << 32; // ARM SME LUTv2
    pub const RVVBB: u64 = 1 << 33; // RISC-V Zvbb
    pub const SIERRA: u64 = 1 << 34; // Intel AVXVNNIINT8
    pub const SMEBI32: u64 = 1 << 35; // ARM SME BI32I32
    pub const LOONGSONASX: u64 = 1 << 36; // LoongArch LASX 256-bit SIMD
    pub const POWERVSX: u64 = 1 << 37; // Power VSX 128-bit SIMD
    pub const DIAMOND: u64 = 1 << 38; // Intel AVX10.2
    pub const NEONFP8: u64 = 1 << 39; // ARM NEON FP8
}
