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
//! - [`Capability`]: One CPU capability — NEON, Skylake, etc.
//! - [`configure_thread`]: Set up the current thread's SIMD state for the given capabilities
//!
//! File: rust/capabilities.rs
//! Author: Ash Vardanian

use core::fmt;
use core::ops::BitOr;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::tensor::TensorError;

#[link(name = "numkong")]
extern "C" {
    fn nk_cpu_capabilities_detected(capabilities: *mut u64) -> Status;
    fn nk_cpu_capabilities_compiled(capabilities: *mut u64) -> Status;
    fn nk_cpu_capabilities_enabled(capabilities: *mut u64) -> Status;
    fn nk_cpu_configure_thread(capabilities: u64) -> Status;
    fn nk_name_capabilities(capabilities: u64, buffer: *mut u8, capacity: usize) -> usize;
}

/// What a dispatched kernel reports, C's `nk_status_t`: zero on success, negative when it wrote
/// nothing.
#[repr(transparent)]
#[must_use]
pub(crate) struct Status(i32);

impl Status {
    /// `Some(())` on success, so a method returning `Option` returns `None` on failure through `?`.
    pub(crate) fn ok(self) -> Option<()> { (self.0 == 0).then_some(()) }

    /// `Ok(())` on success, so a method returning `Result` returns the status through `?`.
    pub(crate) fn check(self) -> Result<(), TensorError> {
        match self.0 {
            0 => Ok(()),
            status => Err(TensorError::KernelFailed { status }),
        }
    }

    /// Panics on failure, in methods whose signatures have no failure path.
    #[track_caller]
    pub(crate) fn unwrap(self) { assert!(self.0 == 0, "NumKong kernel failed with status {}", self.0) }
}

/// A failure the workers of a parallel loop report, checked once the loop joins.
#[cfg(feature = "parallel")]
#[derive(Default)]
pub(crate) struct WorkerStatus(core::sync::atomic::AtomicI32);

#[cfg(feature = "parallel")]
impl WorkerStatus {
    /// Keeps the status of a kernel call that failed.
    pub(crate) fn record(&self, result: Result<(), TensorError>) {
        if let Err(TensorError::KernelFailed { status }) = result {
            self.0.store(status, Ordering::Relaxed);
        }
    }

    /// `Ok(())` unless a worker recorded a failure.
    pub(crate) fn check(&self) -> Result<(), TensorError> { Status(self.0.load(Ordering::Relaxed)).check() }
}

/// One CPU capability, numbered like the C `nk_cap_<capability>_k` bits: each capability group in a
/// contiguous run, ascending by dispatch preference.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    Serial = 1 << 0,       // Always: Fallback
    Neon = 1 << 1,         // ARM NEON
    NeonHalf = 1 << 2,     // ARM NEON FP16
    NeonBfdot = 1 << 3,    // ARM NEON BF16
    NeonFhm = 1 << 4,      // ARM NEON FP16 FML
    NeonSdot = 1 << 5,     // ARM NEON i8 dot
    NeonFp8 = 1 << 6,      // ARM NEON FP8
    Sve = 1 << 7,          // ARM SVE
    SveHalf = 1 << 8,      // ARM SVE FP16
    SveSdot = 1 << 9,      // ARM SVE i8 dot
    SveBfdot = 1 << 10,    // ARM SVE BF16
    Sve2 = 1 << 11,        // ARM SVE2
    Sme = 1 << 12,         // ARM SME
    SmeF64 = 1 << 13,      // ARM SME F64
    SmeBi32 = 1 << 14,     // ARM SME BI32I32
    Haswell = 1 << 15,     // Intel AVX2
    Alder = 1 << 16,       // Intel AVX2+VNNI
    Sierra = 1 << 17,      // Intel AVXVNNIINT8
    Skylake = 1 << 18,     // Intel AVX-512
    Icelake = 1 << 19,     // Intel AVX-512 VNNI
    Genoa = 1 << 20,       // AMD AVX-512 BF16
    Turin = 1 << 21,       // AMD Turin AVX-512 CD
    Sapphire = 1 << 22,    // Intel AVX-512 FP16
    Diamond = 1 << 23,     // Intel AVX10.2
    SapphireAmx = 1 << 24, // Intel Sapphire AMX
    GraniteAmx = 1 << 25,  // Intel Granite AMX FP16
    DiamondAmx = 1 << 26,  // Intel Diamond Rapids AMX
    Rvv = 1 << 27,         // RISC-V Vector
    RvvBf16 = 1 << 28,     // RISC-V Zvfbfwma
    RvvHalf = 1 << 29,     // RISC-V Zvfh
    RvvBb = 1 << 30,       // RISC-V Zvbb
    V128 = 1 << 31,        // WASM SIMD128
    V128Relaxed = 1 << 32, // WASM Relaxed SIMD
    PowerVsx = 1 << 33,    // Power VSX 128-bit SIMD
    LoongsonAsx = 1 << 34, // LoongArch LASX 256-bit SIMD
}

/// Every [`Capability`], in bit order.
const CAPABILITIES: [Capability; 35] = [
    Capability::Serial,
    Capability::Neon,
    Capability::NeonHalf,
    Capability::NeonBfdot,
    Capability::NeonFhm,
    Capability::NeonSdot,
    Capability::NeonFp8,
    Capability::Sve,
    Capability::SveHalf,
    Capability::SveSdot,
    Capability::SveBfdot,
    Capability::Sve2,
    Capability::Sme,
    Capability::SmeF64,
    Capability::SmeBi32,
    Capability::Haswell,
    Capability::Alder,
    Capability::Sierra,
    Capability::Skylake,
    Capability::Icelake,
    Capability::Genoa,
    Capability::Turin,
    Capability::Sapphire,
    Capability::Diamond,
    Capability::SapphireAmx,
    Capability::GraniteAmx,
    Capability::DiamondAmx,
    Capability::Rvv,
    Capability::RvvBf16,
    Capability::RvvHalf,
    Capability::RvvBb,
    Capability::V128,
    Capability::V128Relaxed,
    Capability::PowerVsx,
    Capability::LoongsonAsx,
];

/// The mask every kernel call passes, zero until [`Capabilities::enabled`] first reads it.
static ENABLED: AtomicU64 = AtomicU64::new(0);

/// A set of CPU capabilities, printed as comma-separated names like `serial,haswell`.
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
    /// Capabilities this CPU supports, whether or not their kernels were compiled in.
    pub fn detected() -> Self { query(nk_cpu_capabilities_detected) }

    /// Capabilities whose kernels were compiled into this binary, whether or not this CPU supports them.
    pub fn compiled() -> Self { query(nk_cpu_capabilities_compiled) }

    /// Capabilities every kernel call passes: [`Capabilities::detected`] & [`Capabilities::compiled`]
    /// unless narrowed by [`Capabilities::enable`], always with [`Capability::Serial`].
    pub fn enabled() -> Self {
        let mask = ENABLED.load(Ordering::Relaxed);
        if mask != 0 {
            return Capabilities(mask);
        }
        let available = query(nk_cpu_capabilities_enabled).0;
        match ENABLED.compare_exchange(0, available, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => Capabilities(available),
            Err(current) => Capabilities(current),
        }
    }

    /// Makes this set, clamped to [`Capabilities::detected`] & [`Capabilities::compiled`] and with
    /// [`Capability::Serial`] kept, what every kernel call passes, and returns what stuck. Pack
    /// matrices again after the call: packed kernels refuse another capability's layout with
    /// [`TensorError::KernelFailed`].
    pub fn enable(self) -> Self {
        let mask = self.0 & query(nk_cpu_capabilities_enabled).0 | Capability::Serial as u64;
        ENABLED.store(mask, Ordering::Relaxed);
        Capabilities(mask)
    }

    /// The raw `nk_capability_t` mask.
    pub const fn bits(self) -> u64 { self.0 }

    /// Whether `capability` is in this set.
    pub const fn contains(self, capability: Capability) -> bool { self.0 & capability as u64 != 0 }

    /// This set with `capability` removed, like `Capabilities::enabled().without(Capability::Skylake)`.
    pub const fn without(self, capability: Capability) -> Self { Capabilities(self.0 & !(capability as u64)) }

    /// The capabilities in this set, in bit order.
    pub fn iter(self) -> impl Iterator<Item = Capability> {
        CAPABILITIES
            .into_iter()
            .filter(move |&capability| self.contains(capability))
    }
}

/// Reads one of the `nk_cpu_capabilities_*` masks.
fn query(read: unsafe extern "C" fn(*mut u64) -> Status) -> Capabilities {
    let mut mask = Capability::Serial as u64;
    unsafe { read(&mut mask) }.unwrap();
    Capabilities(mask)
}

/// The mask every kernel call passes, [`Capabilities::enabled`] as C's `nk_capability_t`.
pub(crate) fn cpu_capabilities() -> u64 { Capabilities::enabled().0 }

/// Sets up the calling thread for the kernels in `capabilities`, usually [`Capabilities::enabled`]:
/// AMX tile permission on x86 Linux, fused BF16 dots on Arm. Call it once per thread before using
/// those kernels; it is idempotent. Returns `true` on success.
pub fn configure_thread(capabilities: Capabilities) -> bool {
    unsafe { nk_cpu_configure_thread(capabilities.0) }.ok().is_some()
}

impl From<Capability> for Capabilities {
    fn from(capability: Capability) -> Self { Capabilities(capability as u64) }
}

impl BitOr for Capability {
    type Output = Capabilities;
    fn bitor(self, other: Self) -> Capabilities { Capabilities(self as u64 | other as u64) }
}

impl BitOr<Capability> for Capabilities {
    type Output = Self;
    fn bitor(self, capability: Capability) -> Self { Capabilities(self.0 | capability as u64) }
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
        let names: [&str; CAPABILITIES.len()] = [
            "serial",
            "neon",
            "neonhalf",
            "neonbfdot",
            "neonfhm",
            "neonsdot",
            "neonfp8",
            "sve",
            "svehalf",
            "svesdot",
            "svebfdot",
            "sve2",
            "sme",
            "smef64",
            "smebi32",
            "haswell",
            "alder",
            "sierra",
            "skylake",
            "icelake",
            "genoa",
            "turin",
            "sapphire",
            "diamond",
            "sapphireamx",
            "graniteamx",
            "diamondamx",
            "rvv",
            "rvvbf16",
            "rvvhalf",
            "rvvbb",
            "v128",
            "v128relaxed",
            "powervsx",
            "loongsonasx",
        ];
        for (bit, (capability, name)) in CAPABILITIES.into_iter().zip(names).enumerate() {
            assert_eq!(capability as u64, 1 << bit);
            assert_eq!(capability.to_string(), name);
        }
    }

    #[test]
    fn enabled_is_clamped_and_keeps_serial() {
        let enabled = Capabilities::enabled();
        assert!(enabled.contains(Capability::Serial));
        assert_eq!(
            enabled.bits() & !(Capabilities::detected().bits() | Capability::Serial as u64),
            0
        );
        assert_eq!(enabled.bits() & !Capabilities::compiled().bits(), 0);
    }
}
