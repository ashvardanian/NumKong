//! Capabilities, the unified memory and streams that take them, and the status each C call returns.
//!
//! The CPU and each GPU report their [`Capabilities`] along two axes that have nothing to do with
//! each other, plus the set dispatch uses:
//!
//! - [`Capabilities::cpu_detected`], [`Capabilities::cuda_detected`]: what the CPU or one GPU can
//!   execute, from CPUID / `getauxval` / HWCAP on the CPU and from the runtime on a GPU
//! - [`Capabilities::cpu_compiled`], [`Capabilities::cuda_compiled`]: what this binary contains,
//!   from the build's probes
//! - [`Capabilities::cpu_enabled`], [`Capabilities::cuda_enabled`]: what dispatch runs within —
//!   both axes at once, which the library settles for the CPU as it loads
//!
//! ROCm and Metal have the same functions. Only these and [`Capabilities::stream_init_cuda`] take a
//! GPU's ordinal: everything else takes the mask and a stream, which names the device. Reach for
//! the `enabled` ones unless you specifically mean one of the raw axes. A `detected` mask alone
//! describes the machine and says nothing about whether a kernel was compiled in, so selecting on
//! it claims hardware support for code that may not exist in this build.
//!
//! This module also provides:
//!
//! - [`Capability`]: One capability — NEON, Skylake, Hopper, etc.
//! - [`Stream`]: A device's queue, which [`Stream::scope`] joins so verbs can stay safe
//! - [`Scope`]: The span of such a body, which the safe `_on` verbs queue on
//! - [`UnifiedAllocator`]: Memory the host and the device of a stream both address
//! - [`Status`]: What a C call reports
//!
//! File: rust/capabilities.rs
//! Author: Ash Vardanian
//! Date: March 8, 2026

use core::{
    alloc::Layout,
    ffi::{c_char, c_void, CStr},
    fmt,
    marker::PhantomData,
    ops::BitOr,
    ptr::NonNull,
};

use crate::tensor::{AllocError, Allocator, Error};

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
    fn nk_cuda_capabilities_detected(ordinal: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_cuda_capabilities_compiled(capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_cuda_capabilities_enabled(ordinal: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_stream_init_cuda(ordinal: nk_size_t, stream: *mut *mut c_void) -> nk_status_t;
    fn nk_stream_free_cuda(stream: *mut c_void) -> nk_status_t;
    fn nk_rocm_count_devices(count: *mut nk_size_t) -> nk_status_t;
    fn nk_rocm_capabilities_detected(ordinal: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_rocm_capabilities_compiled(capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_rocm_capabilities_enabled(ordinal: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_stream_init_rocm(ordinal: nk_size_t, stream: *mut *mut c_void) -> nk_status_t;
    fn nk_stream_free_rocm(stream: *mut c_void) -> nk_status_t;
    fn nk_metal_count_devices(count: *mut nk_size_t) -> nk_status_t;
    fn nk_metal_capabilities_detected(ordinal: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_metal_capabilities_compiled(capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_metal_capabilities_enabled(ordinal: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_stream_init_metal(ordinal: nk_size_t, stream: *mut *mut c_void) -> nk_status_t;
    fn nk_stream_free_metal(stream: *mut c_void) -> nk_status_t;
    fn nk_memory_allocate_unified_best(
        bytes: nk_size_t,
        pointer: *mut *mut c_void,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_memory_free_unified_best(
        pointer: *mut c_void,
        bytes: nk_size_t,
        capabilities: nk_capability_t,
        stream: *mut c_void,
    ) -> nk_status_t;
    fn nk_stream_synchronize_best(capabilities: nk_capability_t, stream: *mut c_void) -> nk_status_t;
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
            self.0
                .store(status as nk_status_t, core::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Keeps a dispatch the pool refused, never spawned or running another one, so nothing ran.
    pub(crate) fn record_dispatch(&self, dispatched: forkunion::Result<()>) {
        if dispatched.is_err() {
            self.0
                .store(Status::BadAlloc as nk_status_t, core::sync::atomic::Ordering::Relaxed);
        }
    }

    /// `Ok(())` unless a worker recorded a failure.
    pub(crate) fn check(&self) -> Result<(), Error> { self.0.load(core::sync::atomic::Ordering::Relaxed).check() }
}

/// One capability, numbered like the C `nk_cap_<capability>_k` bits: each capability group in a
/// contiguous run, ascending by dispatch preference, with the GPU vendors' groups above bit 47.
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    /// Serial fallback capability without SIMD. Always available.
    Serial = 1 << 0,
    /// ARM NEON.
    Neon = 1 << 1,
    /// ARM NEON FP16.
    NeonHalf = 1 << 2,
    /// ARM NEON BF16.
    NeonBfDot = 1 << 3,
    /// ARM NEON FP16 FML.
    NeonFhm = 1 << 4,
    /// ARM NEON i8 dot.
    NeonSdot = 1 << 5,
    /// ARM NEON FP8.
    NeonFp8 = 1 << 6,
    /// ARM SVE.
    Sve = 1 << 7,
    /// ARM SVE FP16.
    SveHalf = 1 << 8,
    /// ARM SVE i8 dot.
    SveSdot = 1 << 9,
    /// ARM SVE BF16.
    SveBfDot = 1 << 10,
    /// ARM SVE2.
    Sve2 = 1 << 11,
    /// ARM SME.
    Sme = 1 << 12,
    /// ARM SME F64.
    SmeF64 = 1 << 13,
    /// ARM SME BI32I32.
    SmeBi32 = 1 << 14,
    /// Intel AVX2.
    Haswell = 1 << 15,
    /// Intel AVX2+VNNI.
    Alder = 1 << 16,
    /// Intel AVXVNNIINT8.
    Sierra = 1 << 17,
    /// Intel AVX-512.
    Skylake = 1 << 18,
    /// Intel AVX-512 VNNI.
    IceLake = 1 << 19,
    /// AMD AVX-512 BF16.
    Genoa = 1 << 20,
    /// AMD Turin AVX-512 CD.
    Turin = 1 << 21,
    /// Intel AVX-512 FP16.
    Sapphire = 1 << 22,
    /// Intel AVX10.2.
    Diamond = 1 << 23,
    /// Intel Sapphire AMX.
    SapphireAmx = 1 << 24,
    /// Intel Granite AMX FP16.
    GraniteAmx = 1 << 25,
    /// Intel Diamond Rapids AMX.
    DiamondAmx = 1 << 26,
    /// RISC-V Vector.
    Rvv = 1 << 27,
    /// RISC-V Zvfbfwma.
    RvvBf16 = 1 << 28,
    /// RISC-V Zvfh.
    RvvHalf = 1 << 29,
    /// RISC-V Zvbb.
    RvvBb = 1 << 30,
    /// WASM SIMD128.
    V128 = 1 << 31,
    /// WASM Relaxed SIMD.
    V128Relaxed = 1 << 32,
    /// Power VSX 128-bit SIMD.
    PowerVsx = 1 << 33,
    /// LoongArch LASX 256-bit SIMD.
    LoongsonAsx = 1 << 34,
    /// NVIDIA baseline, running on every CUDA device.
    Cuda = 1 << 48,
    /// NVIDIA SM 8.0.
    Ampere = 1 << 49,
    /// NVIDIA SM 8.9.
    Ada = 1 << 50,
    /// NVIDIA SM 9.x.
    Hopper = 1 << 51,
    /// NVIDIA SM 10.x.
    Blackwell = 1 << 52,
    /// NVIDIA SM 12.x.
    BlackwellRtx = 1 << 53,
    /// NVIDIA SM 10.3.
    BlackwellUltra = 1 << 54,
    /// AMD baseline, running on every ROCm device.
    Rocm = 1 << 56,
    /// AMD gfx942.
    Cdna3 = 1 << 57,
    /// AMD gfx950.
    Cdna4 = 1 << 58,
    /// AMD gfx1250.
    Cdna5 = 1 << 59,
    /// Apple baseline, running on every Metal device.
    Metal = 1 << 60,
    /// Apple GPU family 9.
    Apple9 = 1 << 61,
    /// Apple GPU family 10.
    Apple10 = 1 << 62,
}

/// Every [`Capability`], in bit order.
const CAPABILITIES: [Capability; 49] = [
    Capability::Serial,
    Capability::Neon,
    Capability::NeonHalf,
    Capability::NeonBfDot,
    Capability::NeonFhm,
    Capability::NeonSdot,
    Capability::NeonFp8,
    Capability::Sve,
    Capability::SveHalf,
    Capability::SveSdot,
    Capability::SveBfDot,
    Capability::Sve2,
    Capability::Sme,
    Capability::SmeF64,
    Capability::SmeBi32,
    Capability::Haswell,
    Capability::Alder,
    Capability::Sierra,
    Capability::Skylake,
    Capability::IceLake,
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
    Capability::BlackwellUltra,
    Capability::Rocm,
    Capability::Cdna3,
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
/// use numkong::{Capabilities, Capability};
///
/// let enabled = Capabilities::cpu_enabled();
/// enabled.configure_thread()?;
/// println!("dispatching to {enabled}");
/// if enabled.contains(Capability::SapphireAmx) {
///     println!("AMX is enabled");
/// }
/// for ordinal in 0..Capabilities::cuda_count_devices()? {
///     println!("CUDA device {ordinal} runs {}", Capabilities::cuda_enabled(ordinal)?);
/// }
/// # Ok::<(), numkong::Error>(())
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Capabilities(nk_capability_t);

impl Capabilities {
    /// Every capability, C's `nk_cap_any_k`.
    pub const ANY: Self = Capabilities(nk_capability_t::MAX);

    /// Every CPU capability, the bits below the first GPU vendor's, C's `nk_cap_cpus_k`. Calls
    /// without a mask of their own pass it, as dispatch clamps it to what this CPU runs.
    pub const CPUS: Self = Capabilities(Capability::Cuda as nk_capability_t - 1);

    /// Every GPU capability, which the CPU never detects or enables, C's `nk_cap_gpus_k`.
    pub const GPUS: Self = Capabilities(
        Capability::Cuda as nk_capability_t
            | Capability::Ampere as nk_capability_t
            | Capability::Ada as nk_capability_t
            | Capability::Hopper as nk_capability_t
            | Capability::Blackwell as nk_capability_t
            | Capability::BlackwellRtx as nk_capability_t
            | Capability::BlackwellUltra as nk_capability_t
            | Capability::Rocm as nk_capability_t
            | Capability::Cdna3 as nk_capability_t
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

    /// The GPU vendor whose bit tops this set, which C's dispatch reads as the set's group, or none
    /// for the CPU's.
    fn vendor(self) -> Option<Capability> {
        [Capability::Metal, Capability::Rocm, Capability::Cuda]
            .into_iter()
            .find(|&vendor| self.0 >= vendor as nk_capability_t)
    }

    /// Capabilities this CPU supports, whether or not their kernels were compiled in.
    pub fn cpu_detected() -> Self {
        let mut capabilities: nk_capability_t = 0;
        // The CPU producers only read CPUID, HWCAP or the build's probes, and always succeed.
        let _ = unsafe { nk_cpu_capabilities_detected(&mut capabilities) };
        Capabilities(capabilities)
    }

    /// CPU capabilities whose kernels were compiled in, whether or not this CPU supports them.
    pub fn cpu_compiled() -> Self {
        let mut capabilities: nk_capability_t = 0;
        let _ = unsafe { nk_cpu_capabilities_compiled(&mut capabilities) };
        Capabilities(capabilities)
    }

    /// The CPU capabilities every kernel call runs within: [`Capabilities::cpu_detected`] &
    /// [`Capabilities::cpu_compiled`], always with [`Capability::Serial`].
    pub fn cpu_enabled() -> Self {
        let mut capabilities: nk_capability_t = 0;
        let _ = unsafe { nk_cpu_capabilities_enabled(&mut capabilities) };
        Capabilities(capabilities)
    }

    /// Sets up the calling thread for the CPU kernels in this set, usually
    /// [`Capabilities::cpu_enabled`]: AMX tile permission on x86 Linux, fused BF16 dots on Arm.
    /// Call it once per thread before using those kernels; it is idempotent.
    pub fn configure_thread(self) -> Result<(), Error> { unsafe { nk_cpu_configure_thread(self.0) }.check() }

    /// How many CUDA devices this process sees, zero without the runtime or without its kernels.
    pub fn cuda_count_devices() -> Result<usize, Error> {
        let mut count: nk_size_t = 0;
        // C reports an empty runtime as `nk_missing_gpu_k`, which for a count means zero.
        match unsafe { nk_cuda_count_devices(&mut count) }.check() {
            Err(Error::KernelFailed {
                status: Status::MissingGpu,
            }) => Ok(0),
            result => result.map(|()| count),
        }
    }

    /// Capabilities CUDA device `ordinal` supports, by the runtime's own numbering, whether or not
    /// their kernels were compiled in.
    pub fn cuda_detected(ordinal: usize) -> Result<Self, Error> {
        let mut capabilities: nk_capability_t = 0;
        unsafe { nk_cuda_capabilities_detected(ordinal, &mut capabilities) }.check()?;
        Ok(Capabilities(capabilities))
    }

    /// CUDA capabilities whose kernels were compiled in.
    pub fn cuda_compiled() -> Self {
        let mut capabilities: nk_capability_t = 0;
        let _ = unsafe { nk_cuda_capabilities_compiled(&mut capabilities) };
        Capabilities(capabilities)
    }

    /// The mask CUDA device `ordinal` dispatches with: [`Capabilities::cuda_detected`] &
    /// [`Capabilities::cuda_compiled`].
    pub fn cuda_enabled(ordinal: usize) -> Result<Self, Error> {
        let mut capabilities: nk_capability_t = 0;
        unsafe { nk_cuda_capabilities_enabled(ordinal, &mut capabilities) }.check()?;
        Ok(Capabilities(capabilities))
    }

    /// A new `cudaStream_t` on CUDA device `ordinal`, which names that device to every call it is
    /// passed to; fails with [`Status::MissingGpu`] past the last device or where this build lacks
    /// CUDA.
    ///
    /// # Safety
    /// The caller owns the stream and frees it once, through [`Capabilities::stream_free_cuda`].
    pub unsafe fn stream_init_cuda(ordinal: usize) -> Result<*mut c_void, Error> {
        let mut stream: *mut c_void = core::ptr::null_mut();
        unsafe { nk_stream_init_cuda(ordinal, &mut stream) }.check()?;
        Ok(stream)
    }

    /// Frees a stream [`Capabilities::stream_init_cuda`] made.
    ///
    /// # Safety
    /// `stream` must come from [`Capabilities::stream_init_cuda`] and not be freed yet, and nothing
    /// may use it afterwards, so whatever was queued on it is synchronized first.
    pub unsafe fn stream_free_cuda(stream: *mut c_void) -> Result<(), Error> {
        unsafe { nk_stream_free_cuda(stream) }.check()
    }

    /// [`Capabilities::cuda_count_devices`], for ROCm.
    pub fn rocm_count_devices() -> Result<usize, Error> {
        let mut count: nk_size_t = 0;
        match unsafe { nk_rocm_count_devices(&mut count) }.check() {
            Err(Error::KernelFailed {
                status: Status::MissingGpu,
            }) => Ok(0),
            result => result.map(|()| count),
        }
    }

    /// [`Capabilities::cuda_detected`], for ROCm.
    pub fn rocm_detected(ordinal: usize) -> Result<Self, Error> {
        let mut capabilities: nk_capability_t = 0;
        unsafe { nk_rocm_capabilities_detected(ordinal, &mut capabilities) }.check()?;
        Ok(Capabilities(capabilities))
    }

    /// [`Capabilities::cuda_compiled`], for ROCm.
    pub fn rocm_compiled() -> Self {
        let mut capabilities: nk_capability_t = 0;
        let _ = unsafe { nk_rocm_capabilities_compiled(&mut capabilities) };
        Capabilities(capabilities)
    }

    /// [`Capabilities::cuda_enabled`], for ROCm.
    pub fn rocm_enabled(ordinal: usize) -> Result<Self, Error> {
        let mut capabilities: nk_capability_t = 0;
        unsafe { nk_rocm_capabilities_enabled(ordinal, &mut capabilities) }.check()?;
        Ok(Capabilities(capabilities))
    }

    /// [`Capabilities::stream_init_cuda`], for ROCm, making a `hipStream_t`.
    ///
    /// # Safety
    /// The caller owns the stream and frees it once, through [`Capabilities::stream_free_rocm`].
    pub unsafe fn stream_init_rocm(ordinal: usize) -> Result<*mut c_void, Error> {
        let mut stream: *mut c_void = core::ptr::null_mut();
        unsafe { nk_stream_init_rocm(ordinal, &mut stream) }.check()?;
        Ok(stream)
    }

    /// [`Capabilities::stream_free_cuda`], for ROCm.
    ///
    /// # Safety
    /// `stream` must come from [`Capabilities::stream_init_rocm`], on the same terms.
    pub unsafe fn stream_free_rocm(stream: *mut c_void) -> Result<(), Error> {
        unsafe { nk_stream_free_rocm(stream) }.check()
    }

    /// [`Capabilities::cuda_count_devices`], for Metal, whose devices count in system order.
    pub fn metal_count_devices() -> Result<usize, Error> {
        let mut count: nk_size_t = 0;
        match unsafe { nk_metal_count_devices(&mut count) }.check() {
            Err(Error::KernelFailed {
                status: Status::MissingGpu,
            }) => Ok(0),
            result => result.map(|()| count),
        }
    }

    /// [`Capabilities::cuda_detected`], for Metal.
    pub fn metal_detected(ordinal: usize) -> Result<Self, Error> {
        let mut capabilities: nk_capability_t = 0;
        unsafe { nk_metal_capabilities_detected(ordinal, &mut capabilities) }.check()?;
        Ok(Capabilities(capabilities))
    }

    /// [`Capabilities::cuda_compiled`], for Metal.
    pub fn metal_compiled() -> Self {
        let mut capabilities: nk_capability_t = 0;
        let _ = unsafe { nk_metal_capabilities_compiled(&mut capabilities) };
        Capabilities(capabilities)
    }

    /// [`Capabilities::cuda_enabled`], for Metal.
    pub fn metal_enabled(ordinal: usize) -> Result<Self, Error> {
        let mut capabilities: nk_capability_t = 0;
        unsafe { nk_metal_capabilities_enabled(ordinal, &mut capabilities) }.check()?;
        Ok(Capabilities(capabilities))
    }

    /// [`Capabilities::stream_init_cuda`], for Metal, making an `id<MTLCommandQueue>`.
    ///
    /// # Safety
    /// The caller owns the stream and frees it once, through [`Capabilities::stream_free_metal`].
    pub unsafe fn stream_init_metal(ordinal: usize) -> Result<*mut c_void, Error> {
        let mut stream: *mut c_void = core::ptr::null_mut();
        unsafe { nk_stream_init_metal(ordinal, &mut stream) }.check()?;
        Ok(stream)
    }

    /// [`Capabilities::stream_free_cuda`], for Metal.
    ///
    /// # Safety
    /// `stream` must come from [`Capabilities::stream_init_metal`], on the same terms.
    pub unsafe fn stream_free_metal(stream: *mut c_void) -> Result<(), Error> {
        unsafe { nk_stream_free_metal(stream) }.check()
    }

    /// Waits for everything queued on `stream`, a stream of this mask's group, null for its
    /// default, and null on the CPU.
    ///
    /// # Safety
    /// `stream` must be null or a live `cudaStream_t`, `hipStream_t` or `id<MTLCommandQueue>` of
    /// this mask's group.
    pub unsafe fn synchronize(self, stream: *mut c_void) -> Result<(), Error> {
        unsafe { nk_stream_synchronize_best(self.0, stream) }.check()
    }
}

/// Sets up the calling thread for the enabled CPU capabilities, as every parallel worker must.
#[cfg(any(test, feature = "parallel"))]
pub(crate) fn configure_cpu_thread() -> Result<(), Error> { Capabilities::cpu_enabled().configure_thread() }

/// One GPU vendor's stream producer and the release of what it made.
type StreamProducers = (
    unsafe extern "C" fn(nk_size_t, *mut *mut c_void) -> nk_status_t,
    unsafe extern "C" fn(*mut c_void) -> nk_status_t,
);

/// The stream producers of the vendor `capabilities` belong to, none for the CPU's.
fn stream_producers(capabilities: Capabilities) -> Option<StreamProducers> {
    match capabilities.vendor() {
        Some(Capability::Cuda) => Some((nk_stream_init_cuda, nk_stream_free_cuda)),
        Some(Capability::Rocm) => Some((nk_stream_init_rocm, nk_stream_free_rocm)),
        Some(Capability::Metal) => Some((nk_stream_init_metal, nk_stream_free_metal)),
        _ => None,
    }
}

/// A stream of one capability group's device, where verbs and [`UnifiedAllocator`] blocks queue
/// work: a vendor's default when null, the CPU's always null.
///
/// # Example
/// ```
/// use numkong::{Capabilities, Stream};
///
/// let cpu = Stream::default(Capabilities::cpu_enabled());
/// assert!(cpu.as_raw().is_null());
///
/// // A vendor with no devices fails with `Status::MissingGpu`.
/// for ordinal in 0..Capabilities::cuda_count_devices()? {
///     let stream = Stream::new(Capabilities::cuda_enabled(ordinal)?, ordinal)?;
///     stream.synchronize()?;
/// }
/// # Ok::<(), numkong::Error>(())
/// ```
#[derive(Debug)]
pub struct Stream {
    capabilities: Capabilities,
    handle: *mut c_void,
    owned: bool,
}

impl Stream {
    /// The default stream of the default device of the group `capabilities` name.
    pub fn default(capabilities: Capabilities) -> Self {
        Stream {
            capabilities,
            handle: core::ptr::null_mut(),
            owned: false,
        }
    }

    /// A new stream on device `ordinal` of the vendor `capabilities` name, a `cudaStream_t`, a
    /// `hipStream_t` or an `id<MTLCommandQueue>`, freed on drop. The CPU gets [`Stream::default`].
    ///
    /// Fails with [`Status::MissingGpu`] past the last device or where this build lacks the vendor.
    pub fn new(capabilities: Capabilities, ordinal: usize) -> Result<Self, Error> {
        let Some((init, _)) = stream_producers(capabilities) else {
            return Ok(Self::default(capabilities));
        };
        let mut handle = core::ptr::null_mut();
        unsafe { init(ordinal, &mut handle) }.check()?;
        Ok(Stream {
            capabilities,
            handle,
            owned: true,
        })
    }

    /// Borrows a stream the caller made and keeps owning, which is never freed here.
    ///
    /// # Safety
    /// `handle` must be null or a live stream of the vendor `capabilities` name, a `cudaStream_t`,
    /// a `hipStream_t` or an `id<MTLCommandQueue>`, null on the CPU, and outlive the result.
    pub unsafe fn from_raw(capabilities: Capabilities, handle: *mut c_void) -> Self {
        Stream {
            capabilities,
            handle,
            owned: false,
        }
    }

    /// The capabilities every consumer of this stream dispatches with.
    pub fn capabilities(&self) -> Capabilities { self.capabilities }

    /// The vendor's handle, null for a default stream.
    pub fn as_raw(&self) -> *mut c_void { self.handle }

    /// Waits for everything queued on this stream, after which whatever its verbs wrote is readable
    /// from the host. The CPU has nothing to wait for.
    pub fn synchronize(&self) -> Result<(), Error> {
        unsafe { nk_stream_synchronize_best(self.capabilities.0, self.handle) }.check()
    }

    /// Runs `body`, whose verbs queue on this stream, then joins the stream, even when `body` fails
    /// or unwinds, and reports the first failure, the body's or the join's.
    ///
    /// A verb borrows what it reads and writes for the whole scope, so none of it can be touched
    /// before the join.
    ///
    /// # Example
    /// ```
    /// use numkong::{Capabilities, ShapesPackedMatrix, Stream};
    ///
    /// let triangle = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    /// let packs = ShapesPackedMatrix::new(&[&triangle[..]])?;
    /// let stream = Stream::default(Capabilities::cpu_enabled());
    /// let mut distances = [1.0_f64; 1];
    /// stream.scope(|scope| packs.rmsds_on(scope, &packs, &mut distances))?;
    /// assert!(distances[0] < 1e-6);
    /// # Ok::<(), numkong::Error>(())
    /// ```
    pub fn scope<'env, Output>(
        &'env self,
        body: impl for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> Result<Output, Error>,
    ) -> Result<Output, Error> {
        let join_on_unwind = JoinOnUnwind(self);
        let output = body(&Scope {
            stream: self,
            scope: PhantomData,
        });
        core::mem::forget(join_on_unwind);
        let joined = self.synchronize();
        let output = output?;
        joined.map(|()| output)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Some((_, free)) = stream_producers(self.capabilities).filter(|_| self.owned) {
            // A drop has nowhere to report a failed release.
            let _ = unsafe { free(self.handle) }.check();
        }
    }
}

/// Joins a stream when dropped, which a [`Stream::scope`] body lets happen only by unwinding.
struct JoinOnUnwind<'stream>(&'stream Stream);

impl Drop for JoinOnUnwind<'_> {
    fn drop(&mut self) { let _ = self.0.synchronize(); }
}

/// The span of a [`Stream::scope`] body, which every `_on` verb takes to queue on that stream.
///
/// A verb borrows what it reads and writes for `'scope`, which ends only once the stream is joined.
pub struct Scope<'scope, 'env: 'scope> {
    stream: &'env Stream,
    scope: PhantomData<&'scope mut &'scope ()>,
}

impl Scope<'_, '_> {
    /// The stream every verb of this scope queues on, whose capabilities it dispatches with.
    pub fn stream(&self) -> &Stream { self.stream }
}

/// Memory the host and the devices of one capability group both address, for the packs and outputs
/// of their verbs: managed memory on CUDA and ROCm, a shared buffer on Metal, and the heap on the
/// CPU. Every block lands on the device of the stream it borrows, and a block aligned less strictly
/// than its layout asks is refused.
///
/// # Example
/// ```
/// use numkong::{Capabilities, Stream, Tensor, UnifiedAllocator};
///
/// let capabilities = match Capabilities::cuda_count_devices()? {
///     0 => Capabilities::cpu_enabled(),
///     _ => Capabilities::cuda_enabled(0)?,
/// };
/// let stream = Stream::default(capabilities);
/// let ones = Tensor::<f32, UnifiedAllocator>::ones_in(&[4, 4], UnifiedAllocator::new(&stream))?;
/// stream.synchronize()?;
/// assert_eq!(ones.as_slice()[15], 1.0);
/// # Ok::<(), numkong::Error>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct UnifiedAllocator<'stream> {
    stream: &'stream Stream,
}

impl<'stream> UnifiedAllocator<'stream> {
    /// Unified memory on the device of `stream`, the host heap for a CPU stream.
    pub const fn new(stream: &'stream Stream) -> Self { UnifiedAllocator { stream } }

    /// The capabilities whose memory every block comes from.
    pub fn capabilities(&self) -> Capabilities { self.stream.capabilities }
}

unsafe impl Allocator for UnifiedAllocator<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(
                NonNull::new(core::ptr::without_provenance_mut(layout.align())).ok_or(AllocError)?,
                0,
            ));
        }
        let (capabilities, stream) = (self.stream.capabilities.0, self.stream.handle);
        let mut pointer = core::ptr::null_mut();
        unsafe { nk_memory_allocate_unified_best(layout.size(), &mut pointer, capabilities, stream) }
            .check()
            .map_err(|_| AllocError)?;
        let block = NonNull::new(pointer.cast::<u8>()).ok_or(AllocError)?;
        if block.as_ptr().addr() % layout.align() != 0 {
            unsafe { self.deallocate(block, layout) };
            return Err(AllocError);
        }
        Ok(NonNull::slice_from_raw_parts(block, layout.size()))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() > 0 {
            let (capabilities, stream) = (self.stream.capabilities.0, self.stream.handle);
            let _ = unsafe { nk_memory_free_unified_best(ptr.as_ptr().cast(), layout.size(), capabilities, stream) };
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
            "blackwellultra",
            "rocm",
            "cdna3",
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
        let gpus = Capabilities::GPUS
            .iter()
            .fold(Capabilities::default(), |set, gpu| set | gpu);
        assert_eq!(gpus, Capabilities::GPUS);
        assert_eq!(Capabilities::CPUS.bits() & Capabilities::GPUS.bits(), 0);
    }

    /// One vendor's producers: its count, detected, compiled and enabled queries, and its baseline.
    type Producers = (
        fn() -> Result<usize, Error>,
        fn(usize) -> Result<Capabilities, Error>,
        fn() -> Capabilities,
        fn(usize) -> Result<Capabilities, Error>,
        Capability,
    );

    const GPU_PRODUCERS: [Producers; 3] = [
        (
            Capabilities::cuda_count_devices,
            Capabilities::cuda_detected,
            Capabilities::cuda_compiled,
            Capabilities::cuda_enabled,
            Capability::Cuda,
        ),
        (
            Capabilities::rocm_count_devices,
            Capabilities::rocm_detected,
            Capabilities::rocm_compiled,
            Capabilities::rocm_enabled,
            Capability::Rocm,
        ),
        (
            Capabilities::metal_count_devices,
            Capabilities::metal_detected,
            Capabilities::metal_compiled,
            Capabilities::metal_enabled,
            Capability::Metal,
        ),
    ];

    #[test]
    fn producers_report_clamped_capabilities() {
        let enabled = Capabilities::cpu_enabled();
        assert!(enabled.contains(Capability::Serial));
        let detected = Capabilities::cpu_detected();
        assert_eq!(enabled.bits() & !(detected.bits() | Capability::Serial as u64), 0);
        assert_eq!(enabled.bits() & !Capabilities::cpu_compiled().bits(), 0);

        for (count_devices, detected, compiled, enabled, _) in GPU_PRODUCERS {
            let count = count_devices().unwrap();
            assert!(detected(count).is_err());
            for ordinal in 0..count {
                let reported = detected(ordinal).unwrap();
                assert_eq!(reported.bits() & Capabilities::CPUS.bits(), 0);
                assert_eq!(enabled(ordinal).unwrap().bits(), reported.bits() & compiled().bits());
            }
        }
    }

    #[test]
    fn unified_memory_follows_the_group() {
        use crate::tensor::{Tensor, DEFAULT_ALIGNMENT};

        let cpu = Stream::default(Capabilities::cpu_enabled());
        let allocator = UnifiedAllocator::new(&cpu);
        let empty_layout = Layout::from_size_align(0, 256).unwrap();
        for allocator in [&allocator as &dyn Allocator, &crate::tensor::Global] {
            let empty = allocator.allocate(empty_layout).unwrap();
            assert_eq!(empty.as_ptr().cast::<u8>().addr() % 256, 0);
            let empty = allocator.allocate_zeroed(empty_layout).unwrap();
            assert_eq!(empty.as_ptr().cast::<u8>().addr() % 256, 0);
        }
        let ones = Tensor::<f32, UnifiedAllocator>::ones_in(&[3, 5], UnifiedAllocator::new(&cpu)).unwrap();
        assert!(ones.as_slice().iter().all(|&one| one == 1.0));
        assert_eq!(ones.as_slice().as_ptr() as usize % DEFAULT_ALIGNMENT, 0);
        assert_eq!(cpu.synchronize(), Ok(()));

        // A group without a device, or missing from this build, hands out nothing
        for (count_devices, _, _, _, baseline) in GPU_PRODUCERS {
            let present = count_devices().unwrap() != 0;
            let gpu = Stream::default(Capabilities::from(baseline));
            let zeros = Tensor::<f32, UnifiedAllocator>::zeros_in(&[2], UnifiedAllocator::new(&gpu));
            assert_eq!(zeros.is_ok(), present);
            assert_eq!(gpu.synchronize().is_ok(), present);
        }
    }

    #[test]
    fn streams_follow_their_vendor() {
        let cpu = Stream::new(Capabilities::cpu_enabled(), 7).unwrap();
        assert!(cpu.as_raw().is_null());
        assert_eq!(cpu.scope(|scope| Ok(scope.stream().as_raw().is_null())), Ok(true));

        for (count_devices, _, _, _, baseline) in GPU_PRODUCERS {
            let count = count_devices().unwrap();
            let gpu = Capabilities::from(baseline);
            let past_the_last = Err(Error::KernelFailed {
                status: Status::MissingGpu,
            });
            assert_eq!(Stream::new(gpu, count).map(|_| ()), past_the_last);
            if count == 0 {
                assert_eq!(Stream::default(gpu).scope(|_| Ok(())), past_the_last);
            }
        }

        // The body's failure outranks the join's
        let refused: Result<(), Error> = Err(Error::KernelFailed {
            status: Status::Misaligned,
        });
        let cuda = Stream::default(Capability::Cuda.into());
        if Capabilities::cuda_count_devices().unwrap() == 0 {
            assert_eq!(cuda.scope(|_| refused.clone()), refused);
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
