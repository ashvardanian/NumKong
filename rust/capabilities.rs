//! Devices, the capabilities they report, and the status every C call returns.
//!
//! A [`Device`] is the host CPU or one GPU, and reports its [`Capabilities`] along two axes that
//! have nothing to do with each other, plus the set dispatch uses:
//!
//! - [`Device::capabilities_detected`]: what the device can execute, from CPUID / `getauxval` /
//!   HWCAP on the CPU and from the runtime on a GPU
//! - [`Device::capabilities_compiled`]: what this binary contains, from the build's probes
//! - [`Device::capabilities_enabled`]: what dispatch uses — both axes at once, unless narrowed by
//!   [`Device::capabilities_enable`] on the CPU
//!
//! Reach for [`Device::capabilities_enabled`] unless you specifically mean one of the raw axes.
//! [`Device::capabilities_detected`] alone describes the machine and says nothing about whether a
//! kernel was compiled in, so selecting on it claims hardware support for code that may not exist
//! in this build.
//!
//! This module also provides:
//!
//! - [`Capability`]: One capability — NEON, Skylake, Hopper, etc.
//! - [`DeviceKind`]: The runtime a [`Device`] belongs to
//! - [`Status`]: What a C call reports
//!
//! File: rust/capabilities.rs
//! Author: Ash Vardanian

use core::{
    ffi::{c_char, CStr},
    fmt,
    ops::BitOr,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::tensor::Error;

#[allow(non_camel_case_types)]
pub(crate) type nk_capability_t = u64;
#[allow(non_camel_case_types)]
pub(crate) type nk_size_t = usize;
#[allow(non_camel_case_types)]
pub(crate) type nk_status_t = i32;
#[allow(non_camel_case_types)]
pub(crate) type nk_dtype_t = u32;

#[link(name = "numkong")]
extern "C" {
    fn nk_cpu_capabilities_detected(capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_cpu_capabilities_compiled(capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_cpu_capabilities_enabled(capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_cpu_configure_thread(capabilities: nk_capability_t) -> nk_status_t;
    fn nk_cuda_count_devices(count: *mut nk_size_t) -> nk_status_t;
    fn nk_cuda_capabilities_detected(device: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_cuda_capabilities_compiled(capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_cuda_capabilities_enabled(device: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_rocm_count_devices(count: *mut nk_size_t) -> nk_status_t;
    fn nk_rocm_capabilities_detected(device: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_rocm_capabilities_compiled(capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_rocm_capabilities_enabled(device: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_metal_count_devices(count: *mut nk_size_t) -> nk_status_t;
    fn nk_metal_capabilities_detected(device: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_metal_capabilities_compiled(capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_metal_capabilities_enabled(device: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_capabilities_name(capabilities: nk_capability_t, buffer: *mut c_char, capacity: nk_size_t) -> nk_size_t;
    fn nk_status_name(status: nk_status_t) -> *const c_char;
}

/// What a NumKong call reports, C's `nk_status_t`, each variant holding the header's value.
///
/// Zero is success, and a negative value means the call wrote nothing. A code the header this crate
/// was built from does not list, like the positive values C reserves for results written with a
/// caveat, reads as [`Status::Unrecognized`].
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Status {
    /// `nk_success_k`: scheduled, or finished, without error.
    Success = 0,
    /// `nk_bad_alloc_k`: the queue could not take more memory, or record another block.
    BadAlloc = -10,
    /// `nk_unexpected_dimensions_k`: operand shapes contradict each other.
    UnexpectedDimensions = -15,
    /// `nk_missing_gpu_k`: no GPU of the vendor this build targets answers.
    MissingGpu = -16,
    /// `nk_device_code_mismatch_k`: the device code lacks the kernel, or it failed to run.
    DeviceCodeMismatch = -17,
    /// `nk_device_memory_mismatch_k`: an operand lies in memory the device cannot address.
    DeviceMemoryMismatch = -18,
    /// `nk_missing_kernel_k`: no capability in the capability mask has this kernel.
    MissingKernel = -19,
    /// `nk_misaligned_k`: an operand or stride breaks the alignment contract.
    Misaligned = -20,
    /// `nk_pack_mismatch_k`: the buffer was packed by another capability or layout.
    PackMismatch = -21,
    /// `nk_missing_library_k`: a dispatch point called from a build that links no library.
    MissingLibrary = -22,
    /// A code outside this list, which only a library built from another header returns.
    Unrecognized = i32::MIN,
}

impl Status {
    /// Every variant C defines, which a raw code is matched against.
    const LISTED: [Status; 10] = [
        Status::Success,
        Status::BadAlloc,
        Status::UnexpectedDimensions,
        Status::MissingGpu,
        Status::DeviceCodeMismatch,
        Status::DeviceMemoryMismatch,
        Status::MissingKernel,
        Status::Misaligned,
        Status::PackMismatch,
        Status::MissingLibrary,
    ];

    /// The variant holding `code`, or [`Status::Unrecognized`] for a value C does not list.
    fn from_code(code: nk_status_t) -> Self {
        Self::LISTED
            .into_iter()
            .find(|&status| status as nk_status_t == code)
            .unwrap_or(Status::Unrecognized)
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = unsafe { CStr::from_ptr(nk_status_name(*self as nk_status_t)) };
        f.pad(description.to_str().map_err(|_| fmt::Error)?)
    }
}

/// Reads the `nk_status_t` a C call returns, matching it against [`Status`] rather than
/// transmuting, so a code the header does not list never lands in an enum it isn't a variant of.
pub(crate) trait StatusCode {
    /// `Ok(())` on success, so a method returning `Result` returns the status through `?`.
    fn check(self) -> Result<(), Error>;
}

impl StatusCode for nk_status_t {
    fn check(self) -> Result<(), Error> {
        match Status::from_code(self) {
            Status::Success => Ok(()),
            status => Err(Error::KernelFailed { status }),
        }
    }
}

/// A failure the workers of a parallel loop report, checked once the loop joins.
#[cfg(feature = "parallel")]
#[derive(Default)]
pub(crate) struct WorkerStatus(core::sync::atomic::AtomicI32);

#[cfg(feature = "parallel")]
impl WorkerStatus {
    /// Keeps the status of a kernel call that failed.
    pub(crate) fn record(&self, result: Result<(), Error>) {
        if let Err(Error::KernelFailed { status }) = result {
            self.0.store(status as nk_status_t, Ordering::Relaxed);
        }
    }

    /// `Ok(())` unless a worker recorded a failure.
    pub(crate) fn check(&self) -> Result<(), Error> { self.0.load(Ordering::Relaxed).check() }
}

/// One capability, numbered like the C `nk_cap_<capability>_k` bits: each capability group in a
/// contiguous run, ascending by dispatch preference, with the GPU vendors' groups above bit 47.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    Serial = 1 << 0,        // Always: Fallback
    Neon = 1 << 1,          // ARM NEON
    NeonHalf = 1 << 2,      // ARM NEON FP16
    NeonBfdot = 1 << 3,     // ARM NEON BF16
    NeonFhm = 1 << 4,       // ARM NEON FP16 FML
    NeonSdot = 1 << 5,      // ARM NEON i8 dot
    NeonFp8 = 1 << 6,       // ARM NEON FP8
    Sve = 1 << 7,           // ARM SVE
    SveHalf = 1 << 8,       // ARM SVE FP16
    SveSdot = 1 << 9,       // ARM SVE i8 dot
    SveBfdot = 1 << 10,     // ARM SVE BF16
    Sve2 = 1 << 11,         // ARM SVE2
    Sme = 1 << 12,          // ARM SME
    SmeF64 = 1 << 13,       // ARM SME F64
    SmeBi32 = 1 << 14,      // ARM SME BI32I32
    Haswell = 1 << 15,      // Intel AVX2
    Alder = 1 << 16,        // Intel AVX2+VNNI
    Sierra = 1 << 17,       // Intel AVXVNNIINT8
    Skylake = 1 << 18,      // Intel AVX-512
    Icelake = 1 << 19,      // Intel AVX-512 VNNI
    Genoa = 1 << 20,        // AMD AVX-512 BF16
    Turin = 1 << 21,        // AMD Turin AVX-512 CD
    Sapphire = 1 << 22,     // Intel AVX-512 FP16
    Diamond = 1 << 23,      // Intel AVX10.2
    SapphireAmx = 1 << 24,  // Intel Sapphire AMX
    GraniteAmx = 1 << 25,   // Intel Granite AMX FP16
    DiamondAmx = 1 << 26,   // Intel Diamond Rapids AMX
    Rvv = 1 << 27,          // RISC-V Vector
    RvvBf16 = 1 << 28,      // RISC-V Zvfbfwma
    RvvHalf = 1 << 29,      // RISC-V Zvfh
    RvvBb = 1 << 30,        // RISC-V Zvbb
    V128 = 1 << 31,         // WASM SIMD128
    V128Relaxed = 1 << 32,  // WASM Relaxed SIMD
    PowerVsx = 1 << 33,     // Power VSX 128-bit SIMD
    LoongsonAsx = 1 << 34,  // LoongArch LASX 256-bit SIMD
    Cuda = 1 << 48,         // NVIDIA: every CUDA device
    Ampere = 1 << 49,       // NVIDIA SM 8.0
    Ada = 1 << 50,          // NVIDIA SM 8.9
    Hopper = 1 << 51,       // NVIDIA SM 9.x
    Blackwell = 1 << 52,    // NVIDIA SM 10.x
    BlackwellRtx = 1 << 53, // NVIDIA SM 12.x
    Rocm = 1 << 56,         // AMD: every ROCm device
    Cdna4 = 1 << 57,        // AMD gfx950
    Cdna5 = 1 << 58,        // AMD gfx1250
    Metal = 1 << 60,        // Apple: every Metal device
    Apple9 = 1 << 61,       // Apple GPU family 9
    Apple10 = 1 << 62,      // Apple GPU family 10
}

/// Every [`Capability`], in bit order.
const CAPABILITIES: [Capability; 47] = [
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
    Capability::Cuda,
    Capability::Ampere,
    Capability::Ada,
    Capability::Hopper,
    Capability::Blackwell,
    Capability::BlackwellRtx,
    Capability::Rocm,
    Capability::Cdna4,
    Capability::Cdna5,
    Capability::Metal,
    Capability::Apple9,
    Capability::Apple10,
];

/// A set of capabilities, printed as comma-separated names like `serial,haswell`.
///
/// # Example
/// ```
/// use numkong::{Capability, Device};
///
/// let enabled = Device::cpu().capabilities_enabled()?;
/// println!("dispatching to {enabled}");
/// if enabled.contains(Capability::SapphireAmx) {
///     println!("AMX is enabled");
/// }
/// # Ok::<(), numkong::Error>(())
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Capabilities(nk_capability_t);

impl Capabilities {
    /// Every capability, C's `nk_cap_any_k`.
    pub const ANY: Self = Capabilities(nk_capability_t::MAX);

    /// Every CPU capability, the bits below the first GPU vendor's, C's `nk_cap_cpus_k`.
    pub const CPUS: Self = Capabilities(Capability::Cuda as nk_capability_t - 1);

    /// Every GPU capability, which the CPU never detects or enables, C's `nk_cap_devices_k`.
    pub const DEVICES: Self = Capabilities(
        Capability::Cuda as nk_capability_t
            | Capability::Ampere as nk_capability_t
            | Capability::Ada as nk_capability_t
            | Capability::Hopper as nk_capability_t
            | Capability::Blackwell as nk_capability_t
            | Capability::BlackwellRtx as nk_capability_t
            | Capability::Rocm as nk_capability_t
            | Capability::Cdna4 as nk_capability_t
            | Capability::Cdna5 as nk_capability_t
            | Capability::Metal as nk_capability_t
            | Capability::Apple9 as nk_capability_t
            | Capability::Apple10 as nk_capability_t,
    );

    /// The raw `nk_capability_t` mask.
    pub const fn bits(self) -> u64 { self.0 }

    /// Whether `capability` is in this set.
    pub const fn contains(self, capability: Capability) -> bool { self.0 & capability as nk_capability_t != 0 }

    /// This set less `capability`, like `enabled.without(Capability::Skylake)`.
    pub const fn without(self, capability: Capability) -> Self {
        Capabilities(self.0 & !(capability as nk_capability_t))
    }

    /// The capabilities in this set, in bit order.
    pub fn iter(self) -> impl Iterator<Item = Capability> {
        CAPABILITIES
            .into_iter()
            .filter(move |&capability| self.contains(capability))
    }
}

/// The CPU capability mask every kernel call passes, zero until first read.
static ENABLED: AtomicU64 = AtomicU64::new(0);

/// The raw `nk_capability_t` mask every kernel call passes, [`Device::capabilities_enabled`] of
/// the CPU.
pub(crate) fn enabled_cpu_capabilities_mask() -> nk_capability_t {
    let mask = ENABLED.load(Ordering::Relaxed);
    if mask != 0 {
        return mask;
    }
    let mut available = 0;
    // Only reads CPUID or HWCAP and the build's probes, returning `nk_success_k` on every path.
    let _ = unsafe { nk_cpu_capabilities_enabled(&mut available) };
    match ENABLED.compare_exchange(0, available, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(_) => available,
        Err(current) => current,
    }
}

/// Sets up the calling thread for the enabled CPU capabilities, as every parallel worker must.
#[cfg(any(test, feature = "parallel"))]
pub(crate) fn configure_cpu_thread() -> Result<(), Error> {
    unsafe { nk_cpu_configure_thread(enabled_cpu_capabilities_mask()) }.check()
}

/// Which runtime a device belongs to, as the `nk_<kind>_*` C functions name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Rocm,
    Metal,
}

/// One device NumKong can run kernels on: the host CPU, or a GPU by its runtime's ordinal.
///
/// # Example
/// ```
/// use numkong::{Capability, Device, DeviceKind};
///
/// let cpu = Device::cpu();
/// cpu.configure_thread(cpu.capabilities_enabled()?)?;
/// let narrowed = cpu.capabilities_enable(cpu.capabilities_enabled()?.without(Capability::Skylake))?;
/// assert!(!narrowed.contains(Capability::Skylake));
///
/// for ordinal in 0..Device::count(DeviceKind::Cuda)? {
///     let gpu = Device::new(DeviceKind::Cuda, ordinal)?;
///     println!("CUDA device {ordinal} runs {}", gpu.capabilities_enabled()?);
/// }
/// # Ok::<(), numkong::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Device {
    kind: DeviceKind,
    ordinal: usize,
}

impl Device {
    /// The host CPU, the one device every process has.
    pub const fn cpu() -> Self {
        Device {
            kind: DeviceKind::Cpu,
            ordinal: 0,
        }
    }

    /// How many devices of `kind` this process sees: one CPU, and no GPUs where the runtime finds
    /// none or this build lacks its kernels.
    pub fn count(kind: DeviceKind) -> Result<usize, Error> {
        let count_devices = match kind {
            DeviceKind::Cpu => return Ok(1),
            DeviceKind::Cuda => nk_cuda_count_devices,
            DeviceKind::Rocm => nk_rocm_count_devices,
            DeviceKind::Metal => nk_metal_count_devices,
        };
        let mut count: nk_size_t = 0;
        // C reports an empty runtime as `nk_missing_gpu_k`, which for a count means zero.
        match unsafe { count_devices(&mut count) }.check() {
            Err(Error::KernelFailed {
                status: Status::MissingGpu,
            }) => Ok(0),
            result => result.map(|()| count),
        }
    }

    /// Device `ordinal` of `kind`, as its runtime numbers them; the CPU is ordinal zero.
    pub fn new(kind: DeviceKind, ordinal: usize) -> Result<Self, Error> {
        let count = Self::count(kind)?;
        if ordinal >= count {
            return Err(Error::KernelFailed {
                status: Status::MissingGpu,
            });
        }
        Ok(Device { kind, ordinal })
    }

    /// Which runtime this device belongs to.
    pub const fn kind(&self) -> DeviceKind { self.kind }

    /// This device's ordinal in its runtime, always zero for the CPU.
    pub const fn ordinal(&self) -> usize { self.ordinal }

    /// Capabilities this device supports, whether or not their kernels were compiled in.
    pub fn capabilities_detected(&self) -> Result<Capabilities, Error> {
        let mut mask: nk_capability_t = 0;
        let status = unsafe {
            match self.kind {
                DeviceKind::Cpu => nk_cpu_capabilities_detected(&mut mask),
                DeviceKind::Cuda => nk_cuda_capabilities_detected(self.ordinal, &mut mask),
                DeviceKind::Rocm => nk_rocm_capabilities_detected(self.ordinal, &mut mask),
                DeviceKind::Metal => nk_metal_capabilities_detected(self.ordinal, &mut mask),
            }
        };
        status.check()?;
        Ok(Capabilities(mask))
    }

    /// Capabilities of this device's kind whose kernels were compiled in, whether or not this
    /// device supports them.
    pub fn capabilities_compiled(&self) -> Capabilities {
        let mut mask: nk_capability_t = 0;
        // Each only reads the build's probes, returning `nk_success_k` on every path.
        let _ = unsafe {
            match self.kind {
                DeviceKind::Cpu => nk_cpu_capabilities_compiled(&mut mask),
                DeviceKind::Cuda => nk_cuda_capabilities_compiled(&mut mask),
                DeviceKind::Rocm => nk_rocm_capabilities_compiled(&mut mask),
                DeviceKind::Metal => nk_metal_capabilities_compiled(&mut mask),
            }
        };
        Capabilities(mask)
    }

    /// Capabilities kernels run with: [`Device::capabilities_detected`] &
    /// [`Device::capabilities_compiled`], on the CPU narrowed by [`Device::capabilities_enable`]
    /// and always with [`Capability::Serial`].
    pub fn capabilities_enabled(&self) -> Result<Capabilities, Error> {
        let mut mask: nk_capability_t = 0;
        let status = unsafe {
            match self.kind {
                DeviceKind::Cpu => return Ok(Capabilities(enabled_cpu_capabilities_mask())),
                DeviceKind::Cuda => nk_cuda_capabilities_enabled(self.ordinal, &mut mask),
                DeviceKind::Rocm => nk_rocm_capabilities_enabled(self.ordinal, &mut mask),
                DeviceKind::Metal => nk_metal_capabilities_enabled(self.ordinal, &mut mask),
            }
        };
        status.check()?;
        Ok(Capabilities(mask))
    }

    /// Makes `wanted`, clamped to [`Device::capabilities_detected`] &
    /// [`Device::capabilities_compiled`] and with [`Capability::Serial`] kept, what every CPU
    /// kernel call passes, and returns what stuck.
    ///
    /// This is the one piece of process state the crate keeps: kernel calls take no mask, so every
    /// thread dispatches with the set last enabled here. Pack matrices again after the call:
    /// packed kernels refuse another capability's layout with [`Error::KernelFailed`]. It
    /// applies to the CPU only, and fails with [`Status::MissingKernel`] on a GPU.
    pub fn capabilities_enable(&self, wanted: Capabilities) -> Result<Capabilities, Error> {
        if self.kind != DeviceKind::Cpu {
            return Err(Error::KernelFailed {
                status: Status::MissingKernel,
            });
        }
        let mut available = 0;
        // Only reads CPUID or HWCAP and the build's probes, returning `nk_success_k` on every path.
        let _ = unsafe { nk_cpu_capabilities_enabled(&mut available) };
        let mask = wanted.0 & available | Capability::Serial as nk_capability_t;
        ENABLED.store(mask, Ordering::Relaxed);
        Ok(Capabilities(mask))
    }

    /// Sets up the calling thread for the kernels in `capabilities`, usually
    /// [`Device::capabilities_enabled`]: AMX tile permission on x86 Linux, fused BF16 dots on Arm.
    /// Call it once per thread before using those kernels; it is idempotent. It applies to the CPU
    /// only, and fails with [`Status::MissingKernel`] on a GPU.
    pub fn configure_thread(&self, capabilities: Capabilities) -> Result<(), Error> {
        match self.kind {
            DeviceKind::Cpu => unsafe { nk_cpu_configure_thread(capabilities.0) }.check(),
            DeviceKind::Cuda | DeviceKind::Rocm | DeviceKind::Metal => Err(Error::KernelFailed {
                status: Status::MissingKernel,
            }),
        }
    }
}

impl From<Capability> for Capabilities {
    fn from(capability: Capability) -> Self { Capabilities(capability as nk_capability_t) }
}

impl BitOr for Capability {
    type Output = Capabilities;
    fn bitor(self, other: Self) -> Capabilities { Capabilities(self as nk_capability_t | other as nk_capability_t) }
}

impl BitOr<Capability> for Capabilities {
    type Output = Self;
    fn bitor(self, capability: Capability) -> Self { Capabilities(self.0 | capability as nk_capability_t) }
}

impl BitOr for Capabilities {
    type Output = Self;
    fn bitor(self, other: Self) -> Self { Capabilities(self.0 | other.0) }
}

impl fmt::Display for Capabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buffer = [0u8; 1024]; // NUMKONG_CAPABILITIES_NAME_CAPACITY
        let length = unsafe { nk_capabilities_name(self.0, buffer.as_mut_ptr().cast(), buffer.len()) };
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
            "cuda",
            "ampere",
            "ada",
            "hopper",
            "blackwell",
            "blackwellrtx",
            "rocm",
            "cdna4",
            "cdna5",
            "metal",
            "apple9",
            "apple10",
        ];
        let mut previous = 0;
        for (capability, name) in CAPABILITIES.into_iter().zip(names) {
            assert!(capability as u64 > previous && (capability as u64).is_power_of_two());
            assert_eq!(capability.to_string(), name);
            previous = capability as u64;
        }
        let devices = Capabilities::DEVICES
            .iter()
            .fold(Capabilities::default(), |set, gpu| set | gpu);
        assert_eq!(devices, Capabilities::DEVICES);
        assert_eq!(Capabilities::CPUS.bits() & Capabilities::DEVICES.bits(), 0);
    }

    #[test]
    fn devices_report_clamped_capabilities() {
        let cpu = Device::cpu();
        let enabled = cpu.capabilities_enabled().unwrap();
        assert!(enabled.contains(Capability::Serial));
        let detected = cpu.capabilities_detected().unwrap();
        assert_eq!(enabled.bits() & !(detected.bits() | Capability::Serial as u64), 0);
        assert_eq!(enabled.bits() & !cpu.capabilities_compiled().bits(), 0);
        assert_eq!(Device::count(DeviceKind::Cpu), Ok(1));
        assert!(Device::new(DeviceKind::Cpu, 1).is_err());

        for kind in [DeviceKind::Cuda, DeviceKind::Rocm, DeviceKind::Metal] {
            let count = Device::count(kind).unwrap();
            assert!(Device::new(kind, count).is_err());
            for ordinal in 0..count {
                let device = Device::new(kind, ordinal).unwrap();
                let detected = device.capabilities_detected().unwrap();
                let compiled = device.capabilities_compiled();
                assert_eq!(detected.bits() & Capabilities::CPUS.bits(), 0);
                assert_eq!(
                    device.capabilities_enabled().unwrap().bits(),
                    detected.bits() & compiled.bits()
                );
            }
        }
    }

    #[test]
    fn statuses_read_their_c_codes() {
        let (success, missing_kernel): (nk_status_t, nk_status_t) = (0, -19);
        assert_eq!(success.check(), Ok(()));
        assert_eq!(
            missing_kernel.check(),
            Err(Error::KernelFailed {
                status: Status::MissingKernel
            })
        );
        assert_eq!(Status::PackMismatch.to_string(), "packed by another capability");
        // A code the header does not list, like a positive caveat, is a failure named as such.
        let unlisted: [nk_status_t; 3] = [1, -1, -23];
        for code in unlisted {
            assert_eq!(
                code.check(),
                Err(Error::KernelFailed {
                    status: Status::Unrecognized
                })
            );
        }
        assert_eq!(Status::Unrecognized.to_string(), "an unrecognized status");
    }
}
