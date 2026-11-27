/**
 *  @file include/numkong/cpu.h
 *  @author Ash Vardanian
 *  @date October 3, 2023
 *  @brief CPU detection as the library and header-only builds run it: what the operating
 *      system reports of each architecture, what the compiler flags imply where it cannot be
 *      asked, and the per-thread and per-process setup some capabilities need.
 *
 *  Only underscore helpers live here. The library calls them once as it loads, from
 *  <tt>src/numkong.c</tt>, and header-only builds on every call, from the tail of
 *  <tt>c/numkong/capabilities.h</tt>.
 */
#ifndef NUMKONG_CPU_H
#define NUMKONG_CPU_H

#include "numkong/capabilities.h" // `nk_cap_haswell_k`, `NUMKONG_OS_LINUX_`

/* On Apple Silicon, @c mrs is not allowed in user-space, so we need to use the @c sysctl API. */
#if NUMKONG_OS_APPLE_
#include <fenv.h>       // `fesetenv` - part of C 99 standard
#include <sys/sysctl.h> // `sysctlbyname`
#endif

/*  On Linux and FreeBSD Arm, the kernel reports through @c AT_HWCAP whether it emulates @c mrs
 *  reads of the ID registers, so nothing has to trap to find out. */
#if (NUMKONG_OS_LINUX_ || NUMKONG_OS_FREEBSD_) && NUMKONG_ARCH_ARM64_
#include <sys/auxv.h> // `getauxval`, `elf_aux_info`, `AT_HWCAP`
#endif

/*  On Linux x86 and RISC-V, AMX permission and hwprobe need `syscall()`. With `-std=c11` glibc
 *  hides it behind @c _GNU_SOURCE, which would change the includer's glibc structs, so we
 *  forward-declare @c syscall, which glibc always has. */
#if NUMKONG_OS_LINUX_ && (NUMKONG_ARCH_X8664_ || NUMKONG_ARCH_RISCV64_)
#include <sys/syscall.h> // `SYS_arch_prctl`, `SYS_riscv_hwprobe`
#ifdef __cplusplus
extern "C" long syscall(long, ...) noexcept;
#else
extern long syscall(long, ...);
#endif
#if NUMKONG_ARCH_RISCV64_
#include <sys/auxv.h> // `getauxval`, `AT_HWCAP`
#endif
#endif

#if NUMKONG_OS_LINUX_ && NUMKONG_ARCH_LOONGARCH64_
#include <sys/auxv.h> // `getauxval`, `AT_HWCAP`
#endif

#if NUMKONG_OS_LINUX_ && NUMKONG_ARCH_PPC64_
#include <sys/auxv.h> // `getauxval`, `AT_HWCAP`
#endif

/* On FreeBSD RISC-V, we use elf_aux_info for capability detection */
#if NUMKONG_OS_FREEBSD_ && NUMKONG_ARCH_RISCV64_
#include <sys/auxv.h> // `elf_aux_info`, `AT_HWCAP`
#endif

/* On Windows ARM, we use IsProcessorFeaturePresent API for capability detection */
#if NUMKONG_OS_WINDOWS_ && NUMKONG_ARCH_ARM64_
#include <processthreadsapi.h> // `IsProcessorFeaturePresent`
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/** What the compiler's own flags guarantee of every CPU running this binary, for platforms whose OS
 *  cannot be asked. The compiled mask would name kits that the CPU may lack. */
NUMKONG_INLINE nk_capability_t nk_capabilities_implied_cpu_(void) {
    nk_capability_t capabilities = nk_cap_serial_k;
#if defined(__ARM_NEON) && defined(__aarch64__)
    capabilities |= nk_cap_neon_k;
#endif
#if defined(__riscv_v) && (__riscv_v >= 1000000)
    capabilities |= nk_cap_rvv_k;
#endif
#if defined(__loongarch_asx)
    capabilities |= nk_cap_loongsonasx_k;
#endif
#if defined(__POWER9_VECTOR__)
    capabilities |= nk_cap_powervsx_k;
#endif
    return capabilities;
}

#if NUMKONG_ARCH_X8664_

NUMKONG_INLINE nk_status_t nk_configure_thread_x86_(nk_capability_t capabilities) {
#if NUMKONG_TARGET_SAPPHIREAMX || NUMKONG_TARGET_GRANITEAMX || NUMKONG_TARGET_DIAMONDAMX
    if (capabilities & (nk_cap_sapphireamx_k | nk_cap_graniteamx_k | nk_cap_diamondamx_k)) {
#if NUMKONG_OS_LINUX_
        // Linux requires explicit permission for AMX tile state via arch_prctl syscall
        int const ARCH_REQ_XCOMP_PERM = 0x1023;
        unsigned long const XFEATURE_XTILEDATA = 18;
        if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) return nk_missing_kernel_k;
#endif
        // On Windows, AMX tile state is automatically enabled by the OS if hardware supports it.
        // On FreeBSD, no explicit request is needed either.
        nk_unused_(capabilities);
    }
#else
    nk_unused_(capabilities);
#endif
    return nk_success_k;
}

NUMKONG_INLINE void nk_cpu_cpuid_x8664_(unsigned leaf, unsigned subleaf, int registers[4]) {
#if defined(_MSC_VER)
    __cpuidex(registers, (int)leaf, (int)subleaf);
#else
    __asm__ __volatile__("cpuid"
                         : "=a"(registers[0]), "=b"(registers[1]), "=c"(registers[2]), "=d"(registers[3])
                         : "a"(leaf), "c"(subleaf));
#endif
}

NUMKONG_INLINE nk_capability_t nk_capabilities_detected_x8664_(void) {
    union four_registers_t {
        int array[4];
        struct separate_t {
            unsigned eax, ebx, ecx, edx;
        } named;
    } info0, info1, info7 = {{0}}, info7sub1 = {{0}}, info_extended;

    nk_cpu_cpuid_x8664_(0, 0, info0.array);
    unsigned max_leaf = info0.named.eax;
    if (max_leaf < 1) return nk_cap_serial_k;
    nk_cpu_cpuid_x8664_(1, 0, info1.array);
    if (max_leaf >= 7) {
        nk_cpu_cpuid_x8664_(7, 0, info7.array);
        if (info7.named.eax >= 1) nk_cpu_cpuid_x8664_(7, 1, info7sub1.array);
    }
    nk_cpu_cpuid_x8664_(0x80000000, 0, info_extended.array);
    unsigned supports_lzcnt = 0;
    if (info_extended.named.eax >= 0x80000001) {
        nk_cpu_cpuid_x8664_(0x80000001, 0, info_extended.array);
        supports_lzcnt = (info_extended.named.ecx & 0x00000020) != 0;
    }

    // XCR0 is readable only when the OS sets OSXSAVE, alongside the AVX and XSAVE bits.
    unsigned xcr0 = 0;
    if ((info1.named.ecx & 0x1C000000) == 0x1C000000) {
#if defined(_MSC_VER)
        xcr0 = (unsigned)_xgetbv(0);
#else
        unsigned xcr0_high;
        __asm__ __volatile__("xgetbv" : "=a"(xcr0), "=d"(xcr0_high) : "c"(0));
#endif
    }
    unsigned supports_avx_os = (xcr0 & 0x6) == 0x6;
    unsigned supports_avx512_os = supports_avx_os && (xcr0 & 0xE0) == 0xE0;
#if NUMKONG_OS_APPLE_
    // Darwin enables AVX-512 register state lazily, so XCR0 alone is insufficient.
    int avx512_enabled = 0;
    size_t avx512_size = sizeof(avx512_enabled);
    supports_avx512_os = supports_avx_os &&
                         sysctlbyname("hw.optional.avx512f", &avx512_enabled, &avx512_size, NUMKONG_NULL, 0) == 0 &&
                         avx512_enabled != 0;
#endif
    unsigned supports_amx_os = supports_avx_os && (xcr0 & 0x60000) == 0x60000;
    unsigned supports_sse41 = (info1.named.ecx & 0x00080000) != 0;
    unsigned supports_popcnt = (info1.named.ecx & 0x00800000) != 0;
    unsigned supports_bmi1 = (info7.named.ebx & 0x00000008) != 0;
    unsigned supports_bmi2 = (info7.named.ebx & 0x00000100) != 0;
    unsigned supports_avx512dq = (info7.named.ebx & 0x00020000) != 0;
    unsigned supports_avx512bw = (info7.named.ebx & 0x40000000) != 0;
    unsigned supports_avx512vl = (info7.named.ebx & 0x80000000) != 0;
    unsigned supports_avx2 = (info7.named.ebx & 0x00000020) != 0;
    unsigned supports_f16c = (info1.named.ecx & 0x20000000) != 0;
    unsigned supports_fma = (info1.named.ecx & 0x00001000) != 0;
    unsigned supports_avx512f = (info7.named.ebx & 0x00010000) != 0;
    unsigned supports_avx512fp16 = (info7.named.edx & 0x00800000) != 0;
    unsigned supports_avx512vnni = (info7.named.ecx & 0x00000800) != 0;
    unsigned supports_avx512ifma = (info7.named.ebx & 0x00200000) != 0;
    unsigned supports_avx512bitalg = (info7.named.ecx & 0x00001000) != 0;
    unsigned supports_avx512vbmi = (info7.named.ecx & 0x00000002) != 0;
    unsigned supports_avx512vbmi2 = (info7.named.ecx & 0x00000040) != 0;
    unsigned supports_avx512vpopcntdq = (info7.named.ecx & 0x00004000) != 0;
    unsigned supports_avx512bf16 = (info7sub1.named.eax & 0x00000020) != 0;
    unsigned supports_avx512vp2intersect = (info7.named.edx & 0x00000100) != 0;
    unsigned supports_amx_tile = (info7.named.edx & 0x01000000) != 0;
    unsigned supports_amx_bf16 = (info7.named.edx & 0x00400000) != 0;
    unsigned supports_amx_int8 = (info7.named.edx & 0x02000000) != 0;
    unsigned supports_amx_fp16 = (info7sub1.named.eax & 0x00200000) != 0;
    unsigned supports_avxvnni = (info7sub1.named.eax & 0x00000010) != 0;
    unsigned supports_avxvnniint8 = (info7sub1.named.edx & 0x00000010) != 0;

    // AVX10.2 detection via CPUID leaf 0x24, subleaf 0, valid only when leaf 7.1 EDX bit 19 flags
    // AVX10. EBX[7:0] holds the AVX10 vector ISA version, and bit 18 its 512-bit support.
    unsigned supports_avx10v2 = 0;
    if (max_leaf >= 0x24 && (info7sub1.named.edx & 0x00080000)) {
        union four_registers_t info24;
        nk_cpu_cpuid_x8664_(0x24, 0, info24.array);
        supports_avx10v2 = (info24.named.ebx & 0xFF) >= 2 && (info24.named.ebx & 0x00040000) != 0;
    }

    unsigned supports_amx_fp8 = 0, supports_amx_avx512 = 0;
    // Diamond Rapids AMX detection via CPUID leaf 0x1E, subleaf 1:
    // EAX bit 4 = AMX-FP8, EAX bit 7 = AMX-AVX512.
    if (max_leaf >= 0x1E) {
        union four_registers_t info1e1;
        nk_cpu_cpuid_x8664_(0x1E, 1, info1e1.array);
        supports_amx_fp8 = (info1e1.named.eax & 0x00000010) != 0;
        supports_amx_avx512 = (info1e1.named.eax & 0x00000080) != 0;
    }

    unsigned supports_haswell = supports_avx_os && supports_sse41 && supports_popcnt && supports_avx2 &&
                                supports_f16c && supports_fma && supports_bmi1 && supports_bmi2;
    unsigned supports_skylake = supports_haswell && supports_avx512_os && supports_avx512f && supports_avx512dq &&
                                supports_avx512bw && supports_avx512vl;
    unsigned supports_icelake = supports_skylake && supports_lzcnt && supports_avx512vnni && supports_avx512ifma &&
                                supports_avx512bitalg && supports_avx512vbmi && supports_avx512vbmi2 &&
                                supports_avx512vpopcntdq;
    unsigned supports_genoa = supports_skylake && supports_avx512bf16 && supports_avx512vnni;
    unsigned supports_sapphire = supports_skylake && supports_avx512fp16;
    unsigned supports_diamond = supports_sapphire && supports_avx10v2;
    unsigned supports_turin = supports_genoa && supports_lzcnt && supports_avx512vbmi2 && supports_avx512vp2intersect;
    unsigned supports_alder = supports_haswell && supports_avxvnni;
    unsigned supports_sierra = supports_alder && supports_avxvnniint8;
    unsigned supports_sapphireamx = supports_sapphire && supports_genoa && supports_avx512vbmi && supports_amx_os &&
                                    supports_amx_tile && supports_amx_bf16 && supports_amx_int8;
    unsigned supports_graniteamx = supports_sapphireamx && supports_amx_fp16;
    unsigned supports_diamondamx = supports_sapphireamx && supports_diamond && supports_amx_fp8 && supports_amx_avx512;

    return (nk_capability_t)((nk_cap_haswell_k * supports_haswell) | (nk_cap_skylake_k * supports_skylake) |
                             (nk_cap_icelake_k * supports_icelake) | (nk_cap_genoa_k * supports_genoa) |
                             (nk_cap_diamond_k * supports_diamond) | (nk_cap_sapphire_k * supports_sapphire) |
                             (nk_cap_turin_k * supports_turin) | (nk_cap_sierra_k * supports_sierra) |
                             (nk_cap_alder_k * supports_alder) | (nk_cap_sapphireamx_k * supports_sapphireamx) |
                             (nk_cap_graniteamx_k * supports_graniteamx) | (nk_cap_diamondamx_k * supports_diamondamx) |
                             (nk_cap_serial_k));
}

#endif // NUMKONG_ARCH_X8664_

#if NUMKONG_ARCH_ARM64_

NUMKONG_INLINE nk_status_t nk_configure_thread_arm64_(nk_capability_t capabilities) {
#if defined(_MSC_VER)
    nk_unused_(capabilities);
    return nk_success_k;
#else
    // FPCR.EBF (bit 13) — requires FEAT_EBF16:
    //   Enables fused BF16 dot-product semantics for BFDOT/BFMOPA/BFMMLA.
    //   Without it, each bf16 × bf16 product is individually rounded (Round-to-Odd) before
    //   summation (3-way rounding). With EBF=1, intermediates are summed before rounding,
    //   matching x86 VDPBF16PS (Genoa/Sapphire Rapids) precision.
    //
    // FPCR.AH (bit 1) — requires FEAT_AFP — is intentionally not set:
    //   It enables alternate floating-point behavior (FEAT_RPRES: 12-bit FRECPE/FRSQRTE),
    //   but also changes sign-bit handling in ways that break CPython's `decimal` module
    //   (`Decimal.from_float()` drops the sign of negative values). The FEAT_RPRES benefit
    //   (saving one Newton-Raphson iteration) is not worth the process-wide side effects.
    //
    // EBF defaults to 0 on process creation (kernel zeroes FPCR). Setting it is ABI-legal per
    // AAPCS64. Writing this bit on hardware without FEAT_EBF16 is unsafe (it is RES0), so we
    // gate on feature detection.
    unsigned long fpcr_desired = 0;

#if NUMKONG_OS_APPLE_
    nk_unused_(capabilities);
    size_t sysctl_size = sizeof(unsigned);
    unsigned has_ebf16 = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_EBF16", &has_ebf16, &sysctl_size, NUMKONG_NULL, 0) != 0) has_ebf16 = 0;
    if (has_ebf16) fpcr_desired |= (1UL << 13);

#elif NUMKONG_OS_LINUX_ || NUMKONG_OS_FREEBSD_
    // Read ID registers via MRS, safe only once MRS is known to work: any capability beyond basic
    // NEON means nk_capabilities_detected_arm64_ saw the kernel emulate it.
    if (capabilities & ~(nk_cap_neon_k | nk_cap_serial_k)) {
        // FEAT_EBF16: ID_AA64ISAR1_EL1.BF16 bits [47:44] >= 0b0010
        unsigned long isar1_value;
        __asm__ __volatile__(".inst 0xD5380620\n\tmov %0, x0" : "=r"(isar1_value) : : "x0"); // ID_AA64ISAR1_EL1
        if (((isar1_value >> 44) & 0xF) >= 2) fpcr_desired |= (1UL << 13);
    }
    else { nk_unused_(capabilities); }
#else
    nk_unused_(capabilities);
#endif

    if (fpcr_desired) {
        unsigned long fpcr_value;
        __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr_value));
        if ((fpcr_value & fpcr_desired) != fpcr_desired) {
            fpcr_value |= fpcr_desired;
            __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr_value));
        }
    }
    return nk_success_k;
#endif // _MSC_VER
}

NUMKONG_INLINE nk_capability_t nk_capabilities_detected_arm64_(void) {
#if NUMKONG_OS_APPLE_
    size_t size = sizeof(unsigned);
    unsigned supports_neon = 0, supports_fp16 = 0, supports_fhm = 0, supports_bf16 = 0, supports_dotprod = 0;
    unsigned supports_sme = 0, supports_smef64 = 0, supports_smebi32 = 0;
    // Every Apple arm64 core has NEON, so a denied sysctl keeps it.
    if (sysctlbyname("hw.optional.neon", &supports_neon, &size, NUMKONG_NULL, 0) != 0) supports_neon = 1;
    if (sysctlbyname("hw.optional.arm.FEAT_FP16", &supports_fp16, &size, NUMKONG_NULL, 0) != 0) supports_fp16 = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_FHM", &supports_fhm, &size, NUMKONG_NULL, 0) != 0) supports_fhm = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_BF16", &supports_bf16, &size, NUMKONG_NULL, 0) != 0) supports_bf16 = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &supports_dotprod, &size, NUMKONG_NULL, 0) != 0)
        supports_dotprod = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &supports_sme, &size, NUMKONG_NULL, 0) != 0) supports_sme = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_SME_F64F64", &supports_smef64, &size, NUMKONG_NULL, 0) != 0)
        supports_smef64 = 0;
    if (sysctlbyname("hw.optional.arm.SME_BI32I32", &supports_smebi32, &size, NUMKONG_NULL, 0) != 0)
        supports_smebi32 = 0;

    // macOS exposes no sysctl for FP8 dot-products, so `neonfp8` stays unset here on purpose.
    return (nk_capability_t)((nk_cap_neon_k * (supports_neon)) |
                             (nk_cap_neonhalf_k * (supports_neon && supports_fp16)) |
                             (nk_cap_neonfhm_k * (supports_neon && supports_fp16 && supports_fhm)) |
                             (nk_cap_neonbfdot_k * (supports_neon && supports_bf16)) |
                             (nk_cap_neonsdot_k * (supports_neon && supports_dotprod)) |
                             (nk_cap_sme_k * (supports_sme)) | (nk_cap_smef64_k * (supports_smef64)) |
                             (nk_cap_smebi32_k * (supports_smebi32)) | (nk_cap_serial_k));

#elif NUMKONG_OS_LINUX_ || NUMKONG_OS_FREEBSD_

    unsigned long const hwcap_cpuid = 1ul << 11; // HWCAP_CPUID: the kernel emulates EL0 reads of the ID registers
#if NUMKONG_OS_LINUX_
    unsigned long hwcap = getauxval(AT_HWCAP);
#else
    unsigned long hwcap = 0;
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
#endif
    if (!(hwcap & hwcap_cpuid)) return nk_capabilities_implied_cpu_();

    unsigned long id_aa64isar0_el1 = 0, id_aa64isar1_el1 = 0, id_aa64pfr0_el1 = 0, id_aa64zfr0_el1 = 0,
                  id_aa64fpfr0_el1 = 0;

    // Each raw MRS writes x0; moving it out spares pinning a variable to x0, which NVCC rejects.
    __asm__ __volatile__(".inst 0xD5380600\n\tmov %0, x0" : "=r"(id_aa64isar0_el1) : : "x0"); // ID_AA64ISAR0_EL1
    unsigned supports_integer_dot_products = ((id_aa64isar0_el1 >> 44) & 0xF) >= 1;
    unsigned supports_fhm = ((id_aa64isar0_el1 >> 48) & 0xF) >= 1;
    __asm__ __volatile__(".inst 0xD5380620\n\tmov %0, x0" : "=r"(id_aa64isar1_el1) : : "x0"); // ID_AA64ISAR1_EL1
    unsigned supports_bf16 = ((id_aa64isar1_el1 >> 44) & 0xF) >= 1;

    __asm__ __volatile__(".inst 0xD5380400\n\tmov %0, x0" : "=r"(id_aa64pfr0_el1) : : "x0"); // ID_AA64PFR0_EL1
    unsigned supports_sve = ((id_aa64pfr0_el1 >> 32) & 0xF) >= 1;
    unsigned supports_fp16 = ((id_aa64pfr0_el1 >> 20) & 0xF) == 0x1;
    unsigned supports_neon = ((id_aa64pfr0_el1 >> 20) & 0xF) != 0xF;

    if (supports_sve)
        __asm__ __volatile__(".inst 0xD5380480\n\tmov %0, x0" : "=r"(id_aa64zfr0_el1) : : "x0"); // ID_AA64ZFR0_EL1
    unsigned supports_svebfdot = ((id_aa64zfr0_el1 >> 20) & 0xF) >= 1;
    unsigned supports_sve2 = ((id_aa64zfr0_el1) & 0xF) >= 1;

    __asm__ __volatile__(".inst 0xD53804E0\n\tmov %0, x0" : "=r"(id_aa64fpfr0_el1) : : "x0"); // ID_AA64FPFR0_EL1
    unsigned supports_fp8dot4 = ((id_aa64fpfr0_el1 >> 29) & 0x1) >= 1;

    unsigned long id_aa64pfr1_el1 = 0, id_aa64smfr0_el1 = 0;
    __asm__ __volatile__(".inst 0xD5380420\n\tmov %0, x0" : "=r"(id_aa64pfr1_el1) : : "x0"); // ID_AA64PFR1_EL1
    unsigned supports_sme = ((id_aa64pfr1_el1 >> 24) & 0xF) >= 1;

    unsigned supports_smef64 = 0, supports_smebi32 = 0;
    if (supports_sme) {
        // MRS x0, ID_AA64SMFR0_EL1 (S3_0_C0_C4_5) — encoded as raw .inst because some
        // assemblers (Clang 21 in Android NDK r29) reject the symbolic register name.
        // Encoding: 0xD53804A0 = MRS x0, op0=3, op1=0, CRn=0, CRm=4, op2=5, Rt=0.
        __asm__ __volatile__(".inst 0xD53804A0\n\tmov %0, x0" : "=r"(id_aa64smfr0_el1) : : "x0");
        supports_smef64 = (id_aa64smfr0_el1 >> 48) & 0x1;
        supports_smebi32 = (id_aa64smfr0_el1 >> 33) & 0x1;
    }

    return (nk_capability_t)((nk_cap_neon_k * (supports_neon)) |
                             (nk_cap_neonhalf_k * (supports_neon && supports_fp16)) |
                             (nk_cap_neonfhm_k * (supports_neon && supports_fp16 && supports_fhm)) |
                             (nk_cap_neonbfdot_k * (supports_neon && supports_bf16)) |
                             (nk_cap_neonsdot_k * (supports_neon && supports_integer_dot_products)) |
                             (nk_cap_neonfp8_k * (supports_neon && supports_fp8dot4)) | //
                             (nk_cap_sve_k * (supports_sve)) | (nk_cap_svehalf_k * (supports_sve && supports_fp16)) |
                             (nk_cap_svebfdot_k * (supports_sve && supports_svebfdot)) |
                             (nk_cap_svesdot_k * (supports_sve && supports_integer_dot_products)) |
                             (nk_cap_sve2_k * (supports_sve2)) | (nk_cap_sme_k * (supports_sme)) |
                             (nk_cap_smef64_k * (supports_smef64)) | (nk_cap_smebi32_k * (supports_smebi32)) |
                             (nk_cap_serial_k));
#elif NUMKONG_OS_WINDOWS_

    unsigned supports_neon = 0, supports_dp = 0, supports_fp16 = 0, supports_bf16 = 0;
    unsigned supports_sve = 0, supports_sve2 = 0, supports_svebfdot = 0;
    unsigned supports_sme = 0, supports_smef64 = 0;

#if defined(PF_ARM_V8_INSTRUCTIONS_AVAILABLE)
    supports_neon = IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE);
#endif
#if defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
    supports_dp = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE);
#endif
#if defined(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE)
    supports_fp16 = IsProcessorFeaturePresent(PF_ARM_V82_FP16_INSTRUCTIONS_AVAILABLE);
#endif
#if defined(PF_ARM_V86_BF16_INSTRUCTIONS_AVAILABLE)
    supports_bf16 = IsProcessorFeaturePresent(PF_ARM_V86_BF16_INSTRUCTIONS_AVAILABLE);
#endif
#if defined(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE)
    supports_sve = IsProcessorFeaturePresent(PF_ARM_SVE_INSTRUCTIONS_AVAILABLE);
#endif
#if defined(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE)
    supports_sve2 = IsProcessorFeaturePresent(PF_ARM_SVE2_INSTRUCTIONS_AVAILABLE);
#endif
#if defined(PF_ARM_SVE_BF16_INSTRUCTIONS_AVAILABLE)
    supports_svebfdot = IsProcessorFeaturePresent(PF_ARM_SVE_BF16_INSTRUCTIONS_AVAILABLE);
#endif
#if defined(PF_ARM_SME_INSTRUCTIONS_AVAILABLE)
    supports_sme = IsProcessorFeaturePresent(PF_ARM_SME_INSTRUCTIONS_AVAILABLE);
#endif
#if defined(PF_ARM_SME_F64F64_INSTRUCTIONS_AVAILABLE)
    supports_smef64 = IsProcessorFeaturePresent(PF_ARM_SME_F64F64_INSTRUCTIONS_AVAILABLE);
#endif

    // Windows exposes no distinct FHM, NEON FP8, or SME BI32 flag.
    // Do not infer them from another extension.
    return (nk_capability_t)((nk_cap_neon_k * supports_neon) | (nk_cap_neonhalf_k * (supports_neon && supports_fp16)) |
                             (nk_cap_neonbfdot_k * (supports_neon && supports_bf16)) |
                             (nk_cap_neonsdot_k * (supports_neon && supports_dp)) | (nk_cap_sve_k * supports_sve) |
                             (nk_cap_svehalf_k * (supports_sve && supports_fp16)) |
                             (nk_cap_svebfdot_k * (supports_sve && supports_svebfdot)) |
                             (nk_cap_svesdot_k * (supports_sve && supports_dp)) |
                             (nk_cap_sve2_k * (supports_sve && supports_sve2)) | (nk_cap_sme_k * supports_sme) |
                             (nk_cap_smef64_k * (supports_sme && supports_smef64)) | nk_cap_serial_k);

#else
    return nk_capabilities_implied_cpu_();
#endif
}

#endif // NUMKONG_ARCH_ARM64_

#if NUMKONG_ARCH_RISCV64_

NUMKONG_INLINE nk_capability_t nk_capabilities_detected_riscv64_(void) {
#if NUMKONG_OS_LINUX_
    unsigned long const hwcap_isa_v = 1ul << ('V' - 'A');   // COMPAT_HWCAP_ISA_V, RVV 1.0
    long const riscv_hwprobe = 258;                         // __NR_riscv_hwprobe, from Linux 6.4
    long const hwprobe_extensions_key = 4;                  // RISCV_HWPROBE_KEY_IMA_EXT_0
    unsigned long long const hwprobe_zvfh = 1ull << 30;     // RISCV_HWPROBE_EXT_ZVFH
    unsigned long long const hwprobe_zvfbfwma = 1ull << 54; // RISCV_HWPROBE_EXT_ZVFBFWMA
    unsigned long long const hwprobe_zvbb = 1ull << 17;     // RISCV_HWPROBE_EXT_ZVBB
    unsigned long hwcap = getauxval(AT_HWCAP);
    nk_capability_t caps = nk_cap_serial_k;
    if (hwcap & hwcap_isa_v) {
        caps |= nk_cap_rvv_k;
        struct {
            long key;
            unsigned long value;
        } pairs[1] = {{hwprobe_extensions_key, 0}};
        if (syscall(riscv_hwprobe, pairs, 1, 0, (void *)0, 0) == 0) {
            if (pairs[0].value & hwprobe_zvfh) caps |= nk_cap_rvvhalf_k;
            if (pairs[0].value & hwprobe_zvfbfwma) caps |= nk_cap_rvvbf16_k;
            if (pairs[0].value & hwprobe_zvbb) caps |= nk_cap_rvvbb_k;
        }
    }
    return caps;
#elif NUMKONG_OS_FREEBSD_
    unsigned long const hwcap_isa_v = 1ul << ('V' - 'A'); // COMPAT_HWCAP_ISA_V, RVV 1.0
    unsigned long hwcap = 0;
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
    nk_capability_t caps = nk_cap_serial_k;
    if (hwcap & hwcap_isa_v) {
        caps |= nk_cap_rvv_k;
        // FreeBSD lacks the Linux hwprobe syscall, so
        // zvfh/zvfbfwma/zvbb are compile-time only.
    }
    return caps;
#else
    return nk_capabilities_implied_cpu_();
#endif
}

#endif // NUMKONG_ARCH_RISCV64_

#if NUMKONG_ARCH_LOONGARCH64_

NUMKONG_INLINE nk_capability_t nk_capabilities_detected_loongarch64_(void) {
#if NUMKONG_OS_LINUX_
    unsigned long const hwcap_loongarch_lasx = 1ul << 5; // HWCAP_LOONGARCH_LASX, 256-bit SIMD
    unsigned long hwcap = getauxval(AT_HWCAP);
    nk_capability_t caps = nk_cap_serial_k;
    if (hwcap & hwcap_loongarch_lasx) caps |= nk_cap_loongsonasx_k;
    return caps;
#else
    return nk_capabilities_implied_cpu_();
#endif
}

#endif // NUMKONG_ARCH_LOONGARCH64_

#if NUMKONG_ARCH_PPC64_

NUMKONG_INLINE nk_capability_t nk_capabilities_detected_power64_(void) {
#if NUMKONG_OS_LINUX_
    unsigned long const hwcap_ppc_vsx = 0x00000080ul;        // PPC_FEATURE_HAS_VSX, in AT_HWCAP
    unsigned long const hwcap2_ppc_arch_3_00 = 0x00800000ul; // PPC_FEATURE2_ARCH_3_00, in AT_HWCAP2: POWER9
    unsigned long const hwcap2_ppc_arch_3_1 = 0x00040000ul;  // PPC_FEATURE2_ARCH_3_1, in AT_HWCAP2: POWER10
    unsigned long const hwcap2_ppc_mma = 0x00020000ul;       // PPC_FEATURE2_MMA, in AT_HWCAP2
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    nk_capability_t caps = nk_cap_serial_k;
    unsigned supports_powervsx = (hwcap & hwcap_ppc_vsx) && (hwcap2 & hwcap2_ppc_arch_3_00);
    unsigned supports_powermma = supports_powervsx && (hwcap2 & hwcap2_ppc_arch_3_1) && (hwcap2 & hwcap2_ppc_mma);
    caps |= (nk_cap_powervsx_k * supports_powervsx) | (nk_cap_powermma_k * supports_powermma);
    return caps;
#else
    return nk_capabilities_implied_cpu_();
#endif
}

#endif // NUMKONG_ARCH_PPC64_

#if NUMKONG_ARCH_WASM_

/** An engine validates a module whole before running it, so the compiled capabilities are the
 *  detected ones. */
NUMKONG_INLINE nk_capability_t nk_capabilities_detected_wasm_(void) {
    return nk_cap_serial_k | (nk_cap_v128_k * NUMKONG_TARGET_V128) |
           (nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED);
}

#endif // NUMKONG_ARCH_WASM_

NUMKONG_INLINE nk_status_t nk_configure_thread_cpu_(nk_capability_t capabilities) {
#if NUMKONG_ARCH_X8664_
    return nk_configure_thread_x86_(capabilities);
#endif
#if NUMKONG_ARCH_ARM64_
    return nk_configure_thread_arm64_(capabilities);
#endif
    nk_unused_(capabilities);
    return nk_success_k; // no other architecture needs thread configuration
}

NUMKONG_INLINE nk_capability_t nk_capabilities_detected_cpu_(void) {
#if NUMKONG_ARCH_X8664_
    return nk_capabilities_detected_x8664_();
#elif NUMKONG_ARCH_ARM64_
    return nk_capabilities_detected_arm64_();
#elif NUMKONG_ARCH_RISCV64_
    return nk_capabilities_detected_riscv64_();
#elif NUMKONG_ARCH_LOONGARCH64_
    return nk_capabilities_detected_loongarch64_();
#elif NUMKONG_ARCH_PPC64_
    return nk_capabilities_detected_power64_();
#elif NUMKONG_ARCH_WASM_
    return nk_capabilities_detected_wasm_();
#else
    return nk_cap_serial_k;
#endif
}

/** Drops the AMX capabilities from @p capabilities when the process cannot hold their tile state,
 *  requesting it for the whole process on Linux, as the library does once as it loads. */
NUMKONG_INLINE nk_capability_t nk_capabilities_enable_amx_(nk_capability_t capabilities) {
#if NUMKONG_ARCH_X8664_
    if (nk_configure_thread_x86_(capabilities) != nk_success_k)
        capabilities &= ~(nk_cap_sapphireamx_k | nk_cap_graniteamx_k | nk_cap_diamondamx_k);
#endif
    return capabilities;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_CPU_H
