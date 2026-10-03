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
//! ROCm and Metal have the same functions. Only these and [`Capabilities::cuda_stream_init`] take a
//! GPU's ordinal: everything else takes the mask and a stream, which names the device. Reach for
//! the `enabled` ones unless you specifically mean one of the raw axes. A `detected` mask alone
//! describes the machine and says nothing about whether a kernel was compiled in, so selecting on
//! it claims hardware support for code that may not exist in this build.
//!
//! This module also provides:
//!
//! - [`Capability`]: One capability — NEON, Skylake, Hopper, etc.
//! - [`UnifiedAllocator`]: Memory the host and the GPU of a mask both address
//! - [`Status`]: What a C call reports
//!
//! File: rust/capabilities.rs
//! Author: Ash Vardanian

use core::{
    alloc::Layout,
    ffi::{c_char, c_void, CStr},
    fmt,
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
    fn nk_cuda_stream_init(ordinal: nk_size_t, stream: *mut *mut c_void) -> nk_status_t;
    fn nk_cuda_stream_free(stream: *mut c_void) -> nk_status_t;
    fn nk_rocm_count_devices(count: *mut nk_size_t) -> nk_status_t;
    fn nk_rocm_capabilities_detected(ordinal: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_rocm_capabilities_compiled(capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_rocm_capabilities_enabled(ordinal: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_rocm_stream_init(ordinal: nk_size_t, stream: *mut *mut c_void) -> nk_status_t;
    fn nk_rocm_stream_free(stream: *mut c_void) -> nk_status_t;
    fn nk_metal_count_devices(count: *mut nk_size_t) -> nk_status_t;
    fn nk_metal_capabilities_detected(ordinal: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_metal_capabilities_compiled(capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_metal_capabilities_enabled(ordinal: nk_size_t, capabilities: *mut nk_capability_t) -> nk_status_t;
    fn nk_metal_stream_init(ordinal: nk_size_t, stream: *mut *mut c_void) -> nk_status_t;
    fn nk_metal_stream_free(stream: *mut c_void) -> nk_status_t;
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
    NeonBfDot = 1 << 3,     // ARM NEON BF16
    NeonFhm = 1 << 4,       // ARM NEON FP16 FML
    NeonSdot = 1 << 5,      // ARM NEON i8 dot
    NeonFp8 = 1 << 6,       // ARM NEON FP8
    Sve = 1 << 7,           // ARM SVE
    SveHalf = 1 << 8,       // ARM SVE FP16
    SveSdot = 1 << 9,       // ARM SVE i8 dot
    SveBfDot = 1 << 10,     // ARM SVE BF16
    Sve2 = 1 << 11,         // ARM SVE2
    Sme = 1 << 12,          // ARM SME
    SmeF64 = 1 << 13,       // ARM SME F64
    SmeBi32 = 1 << 14,      // ARM SME BI32I32
    Haswell = 1 << 15,      // Intel AVX2
    Alder = 1 << 16,        // Intel AVX2+VNNI
    Sierra = 1 << 17,       // Intel AVXVNNIINT8
    Skylake = 1 << 18,      // Intel AVX-512
    IceLake = 1 << 19,      // Intel AVX-512 VNNI
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
    /// The caller owns the stream and frees it once, through [`Capabilities::cuda_stream_free`].
    pub unsafe fn cuda_stream_init(ordinal: usize) -> Result<*mut c_void, Error> {
        let mut stream: *mut c_void = core::ptr::null_mut();
        unsafe { nk_cuda_stream_init(ordinal, &mut stream) }.check()?;
        Ok(stream)
    }

    /// Frees a stream [`Capabilities::cuda_stream_init`] made.
    ///
    /// # Safety
    /// `stream` must come from [`Capabilities::cuda_stream_init`] and not be freed yet, and nothing
    /// may use it afterwards, so whatever was queued on it is synchronized first.
    pub unsafe fn cuda_stream_free(stream: *mut c_void) -> Result<(), Error> {
        unsafe { nk_cuda_stream_free(stream) }.check()
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

    /// [`Capabilities::cuda_stream_init`], for ROCm, making a `hipStream_t`.
    ///
    /// # Safety
    /// The caller owns the stream and frees it once, through [`Capabilities::rocm_stream_free`].
    pub unsafe fn rocm_stream_init(ordinal: usize) -> Result<*mut c_void, Error> {
        let mut stream: *mut c_void = core::ptr::null_mut();
        unsafe { nk_rocm_stream_init(ordinal, &mut stream) }.check()?;
        Ok(stream)
    }

    /// [`Capabilities::cuda_stream_free`], for ROCm.
    ///
    /// # Safety
    /// `stream` must come from [`Capabilities::rocm_stream_init`], on the same terms.
    pub unsafe fn rocm_stream_free(stream: *mut c_void) -> Result<(), Error> {
        unsafe { nk_rocm_stream_free(stream) }.check()
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

    /// [`Capabilities::cuda_stream_init`], for Metal, making an `id<MTLCommandQueue>`.
    ///
    /// # Safety
    /// The caller owns the stream and frees it once, through [`Capabilities::metal_stream_free`].
    pub unsafe fn metal_stream_init(ordinal: usize) -> Result<*mut c_void, Error> {
        let mut stream: *mut c_void = core::ptr::null_mut();
        unsafe { nk_metal_stream_init(ordinal, &mut stream) }.check()?;
        Ok(stream)
    }

    /// [`Capabilities::cuda_stream_free`], for Metal.
    ///
    /// # Safety
    /// `stream` must come from [`Capabilities::metal_stream_init`], on the same terms.
    pub unsafe fn metal_stream_free(stream: *mut c_void) -> Result<(), Error> {
        unsafe { nk_metal_stream_free(stream) }.check()
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

/// Memory the host and the device of a capability mask both address, on the null stream: managed
/// memory on CUDA and ROCm, a shared buffer on Metal, and the heap on the CPU.
///
/// # Example
/// ```
/// use numkong::{Capabilities, Tensor, UnifiedAllocator};
///
/// let capabilities = match Capabilities::cuda_count_devices()? {
///     0 => Capabilities::cpu_enabled(),
///     _ => Capabilities::cuda_enabled(0)?,
/// };
/// let ones = Tensor::<f32, UnifiedAllocator>::ones_in(&[4, 4], UnifiedAllocator::new(capabilities))?;
/// unsafe { capabilities.synchronize(core::ptr::null_mut())? };
/// assert_eq!(ones.as_slice()[15], 1.0);
/// # Ok::<(), numkong::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnifiedAllocator(Capabilities);

impl UnifiedAllocator {
    /// Allocates for the group of `capabilities`, like [`Capabilities::cuda_enabled`] reports.
    pub const fn new(capabilities: Capabilities) -> Self { UnifiedAllocator(capabilities) }
}

unsafe impl Allocator for UnifiedAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::slice_from_raw_parts(NonNull::dangling(), 0));
        }
        let mut pointer = core::ptr::null_mut();
        unsafe { nk_memory_allocate_unified_best(layout.size(), &mut pointer, self.0 .0, core::ptr::null_mut()) }
            .check()
            .map_err(|_| AllocError)?;
        let block = NonNull::new(pointer.cast::<u8>()).ok_or(AllocError)?;
        if block.as_ptr() as usize % layout.align() != 0 {
            // SAFETY: the block came from this allocator, with the size of `layout`.
            unsafe { self.deallocate(block, layout) };
            return Err(AllocError);
        }
        Ok(NonNull::slice_from_raw_parts(block, layout.size()))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() > 0 {
            let _ = unsafe {
                nk_memory_free_unified_best(ptr.as_ptr().cast(), layout.size(), self.0 .0, core::ptr::null_mut())
            };
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
        use crate::tensor::{Tensor, SIMD_ALIGNMENT};

        let cpu = Capabilities::cpu_enabled();
        let ones = Tensor::<f32, UnifiedAllocator>::ones_in(&[3, 5], UnifiedAllocator::new(cpu)).unwrap();
        assert!(ones.as_slice().iter().all(|&one| one == 1.0));
        assert_eq!(ones.as_slice().as_ptr() as usize % SIMD_ALIGNMENT, 0);
        assert_eq!(unsafe { cpu.synchronize(core::ptr::null_mut()) }, Ok(()));

        // A group without a device, or missing from this build, hands out nothing
        for (count_devices, _, _, _, baseline) in GPU_PRODUCERS {
            let present = count_devices().unwrap() != 0;
            let gpu = Capabilities::from(baseline);
            let zeros = Tensor::<f32, UnifiedAllocator>::zeros_in(&[2], UnifiedAllocator::new(gpu));
            assert_eq!(zeros.is_ok(), present);
            assert_eq!(unsafe { gpu.synchronize(core::ptr::null_mut()) }.is_ok(), present);
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
