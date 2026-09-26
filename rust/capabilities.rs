//! SIMD capability reporting, along two independent axes.
//!
//! [`Capabilities`] are reported along two axes that have nothing to do with each other, plus the
//! set dispatch uses:
//!
//! - [`Capabilities::detected`]: what this CPU can execute, from CPUID / `getauxval` / HWCAP
//! - [`Capabilities::compiled`]: what this binary contains, from the ISA probes run at build time
//! - [`Capabilities::enabled`]: what dispatch uses — both axes at once, unless narrowed by
//!   [`Capabilities::enable`]
//!
//! Reach for [`Capabilities::enabled`] unless you specifically mean one of the raw axes.
//! [`Capabilities::detected`] alone describes the machine and says nothing about whether a kernel
//! was compiled in, so selecting on it claims hardware support for code that may not exist in this
//! build.
//!
//! This module also provides:
//!
//! - [`Capability`]: One CPU tier — NEON, Skylake, etc.
//! - [`configure_thread`]: Set up the current thread's SIMD state for the given tiers
//! - [`uses_runtime_dispatch`]: Check if the library selects kernels at runtime
//!
//! File: rust/capabilities.rs
//! Author: Ash Vardanian

use core::fmt;
use core::ops::BitOr;

#[link(name = "numkong")]
extern "C" {
    fn nk_cpu_capabilities_detected() -> u64;
    fn nk_cpu_capabilities_compiled() -> u64;
    fn nk_cpu_capabilities_enabled() -> u64;
    fn nk_cpu_capabilities_enable(wanted: u64) -> u64;
    fn nk_cpu_configure_thread(capabilities: u64) -> i32;
    fn nk_uses_runtime_dispatch() -> i32;
    fn nk_name_capabilities(capabilities: u64, buffer: *mut u8, capacity: usize) -> usize;
}

/// One CPU capability tier, numbered like the C `nk_cap_<tier>_k` bits.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    Serial = 1 << 0,       // Always: Fallback
    Neon = 1 << 1,         // ARM NEON
    Haswell = 1 << 2,      // Intel AVX2
    Skylake = 1 << 3,      // Intel AVX-512
    NeonHalf = 1 << 4,     // ARM NEON FP16
    NeonSdot = 1 << 5,     // ARM NEON i8 dot
    NeonFhm = 1 << 6,      // ARM NEON FP16 FML
    Icelake = 1 << 7,      // Intel AVX-512 VNNI
    Genoa = 1 << 8,        // AMD AVX-512 BF16
    NeonBfdot = 1 << 9,    // ARM NEON BF16
    Sve = 1 << 10,         // ARM SVE
    SveHalf = 1 << 11,     // ARM SVE FP16
    SveSdot = 1 << 12,     // ARM SVE i8 dot
    Alder = 1 << 13,       // Intel AVX2+VNNI
    SveBfdot = 1 << 14,    // ARM SVE BF16
    Sve2 = 1 << 15,        // ARM SVE2
    V128Relaxed = 1 << 16, // WASM Relaxed SIMD
    Sapphire = 1 << 17,    // Intel AVX-512 FP16
    SapphireAmx = 1 << 18, // Intel Sapphire AMX
    Rvv = 1 << 19,         // RISC-V Vector
    RvvHalf = 1 << 20,     // RISC-V Zvfh
    RvvBf16 = 1 << 21,     // RISC-V Zvfbfwma
    GraniteAmx = 1 << 22,  // Intel Granite AMX FP16
    Turin = 1 << 23,       // AMD Turin AVX-512 CD
    Sme = 1 << 24,         // ARM SME
    Sme2 = 1 << 25,        // ARM SME2
    SmeF64 = 1 << 26,      // ARM SME F64
    SmeFa64 = 1 << 27,     // ARM SME FA64
    Sve2p1 = 1 << 28,      // ARM SVE2.1
    Sme2p1 = 1 << 29,      // ARM SME2.1
    SmeHalf = 1 << 30,     // ARM SME F16F16
    SmeBf16 = 1 << 31,     // ARM SME B16B16
    SmeLut2 = 1 << 32,     // ARM SME LUTv2
    RvvBb = 1 << 33,       // RISC-V Zvbb
    Sierra = 1 << 34,      // Intel AVXVNNIINT8
    SmeBi32 = 1 << 35,     // ARM SME BI32I32
    LoongsonAsx = 1 << 36, // LoongArch LASX 256-bit SIMD
    PowerVsx = 1 << 37,    // Power VSX 128-bit SIMD
    Diamond = 1 << 38,     // Intel AVX10.2
    NeonFp8 = 1 << 39,     // ARM NEON FP8
    DiamondAmx = 1 << 40,  // Intel Diamond Rapids AMX
    V128 = 1 << 41,        // WASM SIMD128
}

/// Every [`Capability`], in bit order.
const TIERS: [Capability; 42] = [
    Capability::Serial,
    Capability::Neon,
    Capability::Haswell,
    Capability::Skylake,
    Capability::NeonHalf,
    Capability::NeonSdot,
    Capability::NeonFhm,
    Capability::Icelake,
    Capability::Genoa,
    Capability::NeonBfdot,
    Capability::Sve,
    Capability::SveHalf,
    Capability::SveSdot,
    Capability::Alder,
    Capability::SveBfdot,
    Capability::Sve2,
    Capability::V128Relaxed,
    Capability::Sapphire,
    Capability::SapphireAmx,
    Capability::Rvv,
    Capability::RvvHalf,
    Capability::RvvBf16,
    Capability::GraniteAmx,
    Capability::Turin,
    Capability::Sme,
    Capability::Sme2,
    Capability::SmeF64,
    Capability::SmeFa64,
    Capability::Sve2p1,
    Capability::Sme2p1,
    Capability::SmeHalf,
    Capability::SmeBf16,
    Capability::SmeLut2,
    Capability::RvvBb,
    Capability::Sierra,
    Capability::SmeBi32,
    Capability::LoongsonAsx,
    Capability::PowerVsx,
    Capability::Diamond,
    Capability::NeonFp8,
    Capability::DiamondAmx,
    Capability::V128,
];

/// A set of CPU capability tiers, printed as comma-separated names like `serial,haswell`.
///
/// # Example
/// ```
/// use numkong::{Capabilities, Capability};
///
/// let enabled = Capabilities::enabled();
/// println!("dispatching to {enabled}");
/// if enabled.contains(Capability::SapphireAmx) {
///     println!("AMX is enabled");
/// }
/// let narrowed = enabled.without(Capability::Skylake).enable();
/// assert!(!narrowed.contains(Capability::Skylake));
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Capabilities(u64);

impl Capabilities {
    /// Tiers this CPU supports, whether or not their kernels were compiled in.
    pub fn detected() -> Self { Capabilities(unsafe { nk_cpu_capabilities_detected() }) }

    /// Tiers whose kernels were compiled into this binary, whether or not this CPU supports them.
    pub fn compiled() -> Self { Capabilities(unsafe { nk_cpu_capabilities_compiled() }) }

    /// Tiers dispatch selects kernels from: [`Capabilities::detected`] & [`Capabilities::compiled`]
    /// unless narrowed by [`Capabilities::enable`], always with [`Capability::Serial`].
    pub fn enabled() -> Self { Capabilities(unsafe { nk_cpu_capabilities_enabled() }) }

    /// Makes this set, clamped to [`Capabilities::detected`] & [`Capabilities::compiled`] and with
    /// [`Capability::Serial`] kept, what dispatch selects from, and returns what stuck.
    pub fn enable(self) -> Self { Capabilities(unsafe { nk_cpu_capabilities_enable(self.0) }) }

    /// The raw `nk_capability_t` mask.
    pub const fn bits(self) -> u64 { self.0 }

    /// Whether `tier` is in this set.
    pub const fn contains(self, tier: Capability) -> bool { self.0 & tier as u64 != 0 }

    /// This set with `tier` removed, like `Capabilities::enabled().without(Capability::Skylake)`.
    pub const fn without(self, tier: Capability) -> Self { Capabilities(self.0 & !(tier as u64)) }

    /// The tiers in this set, in bit order.
    pub fn iter(self) -> impl Iterator<Item = Capability> { TIERS.into_iter().filter(move |&tier| self.contains(tier)) }
}

/// Sets up the calling thread for the kernels in `capabilities`, usually [`Capabilities::enabled`]:
/// AMX tile permission on x86 Linux, fused BF16 dots on Arm. Call it once per thread before using
/// those kernels; it is idempotent. Returns `true` on success.
pub fn configure_thread(capabilities: Capabilities) -> bool { unsafe { nk_cpu_configure_thread(capabilities.0) != 0 } }

/// Returns `true` if the library uses runtime dispatch for function selection.
pub fn uses_runtime_dispatch() -> bool { unsafe { nk_uses_runtime_dispatch() != 0 } }

impl From<Capability> for Capabilities {
    fn from(tier: Capability) -> Self { Capabilities(tier as u64) }
}

impl BitOr for Capability {
    type Output = Capabilities;
    fn bitor(self, other: Self) -> Capabilities { Capabilities(self as u64 | other as u64) }
}

impl BitOr<Capability> for Capabilities {
    type Output = Self;
    fn bitor(self, tier: Capability) -> Self { Capabilities(self.0 | tier as u64) }
}

impl BitOr for Capabilities {
    type Output = Self;
    fn bitor(self, other: Self) -> Self { Capabilities(self.0 | other.0) }
}

impl fmt::Display for Capabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buffer = [0u8; 1024]; // NUMKONG_CAPABILITIES_NAME_CAPACITY
        let length = unsafe { nk_name_capabilities(self.0, buffer.as_mut_ptr(), buffer.len()) };
        f.pad(core::str::from_utf8(&buffer[..length]).map_err(|_| fmt::Error)?)
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(&Capabilities::from(*self), f) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_match_the_c_table() {
        let names: [&str; TIERS.len()] = [
            "serial",
            "neon",
            "haswell",
            "skylake",
            "neonhalf",
            "neonsdot",
            "neonfhm",
            "icelake",
            "genoa",
            "neonbfdot",
            "sve",
            "svehalf",
            "svesdot",
            "alder",
            "svebfdot",
            "sve2",
            "v128relaxed",
            "sapphire",
            "sapphireamx",
            "rvv",
            "rvvhalf",
            "rvvbf16",
            "graniteamx",
            "turin",
            "sme",
            "sme2",
            "smef64",
            "smefa64",
            "sve2p1",
            "sme2p1",
            "smehalf",
            "smebf16",
            "smelut2",
            "rvvbb",
            "sierra",
            "smebi32",
            "loongsonasx",
            "powervsx",
            "diamond",
            "neonfp8",
            "diamondamx",
            "v128",
        ];
        for (bit, (tier, name)) in TIERS.into_iter().zip(names).enumerate() {
            assert_eq!(tier as u64, 1 << bit);
            assert_eq!(tier.to_string(), name);
        }
    }
}
