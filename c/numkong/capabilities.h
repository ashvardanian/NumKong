/**
 *  @file c/numkong/capabilities.h
 *  @author Ash Vardanian
 *  @date October 3, 2023
 *  @brief Capabilities and their groups, kernel kinds and signatures, the pick of a kernel, and CPU
 *      and GPU detection.
 *
 *  A capability is one bit, like @c nk_cap_haswell_k or @c nk_cap_ampere_k, and a capability group
 *  is the range of bits of one kind of hardware: the CPU's, CUDA's, ROCm's or Metal's. Every batch
 *  kernel, on every capability, returns an @c nk_status_t and takes a trailing @c stream: null on
 *  the CPU, a @c cudaStream_t, @c hipStream_t or @c id<MTLCommandQueue> on a GPU. A dispatch
 *  point, like @c nk_dot_f32_best, holds its kernels of every capability group and runs the best
 *  capability its mask shares with the group that mask describes.
 *
 *  @section gpu_devices GPU Devices
 *
 *  Each vendor numbers its own devices; a stream names one. Only the functions reporting a
 *  device's capabilities or making a stream on it take its ordinal, and everything else takes that
 *  capability mask and a stream of that vendor:
 *
 *  - CUDA: a @c cudaStream_t, or null for the legacy default stream of the current device.
 *  - ROCm: a @c hipStream_t, or null for the null stream of the current device.
 *  - Metal: an @c id<MTLCommandQueue>, or null for the library's queue on the default device.
 *
 *  A GPU entry queues its work on the stream and returns without waiting, and
 *  @ref nk_stream_synchronize_best waits for it.
 *
 *  @section x86_targets Choosing x86 Target Generations
 *
 *  Fine-grained controls over AVX512 families matter, since they are very fragmented:
 *
 *  - Intel Skylake servers: F, CD, VL, DQ, BW
 *  - Intel Cascade Lake workstations: F, CD, VL, DQ, BW, VNNI, extending Skylake with VNNI support.
 *  - Intel Sunny Cove (Ice Lake) servers:
 *    F, CD, VL, DQ, BW, VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ
 *  - AMD Zen4 (Genoa): F, CD, VL, DQ, BW, VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG,
 *    VPCLMULQDQ, BF16, extending Sunny Cove with BF16 support.
 *  - Intel Golden Cove (Sapphire Rapids): extends Zen4 and Sunny Cove with FP16 support
 *  - AMD Zen5 (Turin): makes VP2INTERSECT cool again
 *
 *  Intel Palm Cove was an irrelevant intermediate release extending Skylake with IFMA and VBMI.
 *  Intel Willow Cove was an irrelevant intermediate release extending Sunny Cove with VP2INTERSECT,
 *  which are not supported by other CPUs to date and are only available in Tiger Lake laptops.
 *  Intel Cooper Lake was the only intermediary platform, that supported BF16, but not FP16.
 *  It's mostly used in 4-socket and 8-socket high-memory configurations.
 *
 *  For us, it makes sense to differentiate only these AVX512 generations:
 *
 *  1. Intel Skylake (pre 2019): supports single-precision dot-products.
 *  2. Intel Ice Lake (2019-2021): advanced integer algorithms.
 *  3. AMD Genoa (2023+): brain-floating point support.
 *  4. Intel Sapphire Rapids (2023+): advanced mixed-precision float processing.
 *  5. AMD Turin (2024+): advanced sparse algorithms.
 *
 *  Beyond those, AVX2 serves Haswell, AVX2+VNNI serves 12th and 13th gen Alder Lake, and Sierra
 *  Forest gets AVX2+VNNI-INT8, whose 8-bit dot products natively take signed × signed and
 *  unsigned × unsigned operands.
 *
 *  To list all available macros for x86, take a recent compiler, like GCC 12 and run:
 *
 *  @code{.sh}
 *  gcc-12 -march=sapphirerapids -dM -E - < /dev/null | egrep "SSE|AVX" | sort
 *  @endcode
 *
 *  On Arm machines you may want to check for other flags:
 *
 *  @code{.sh}
 *  gcc-12 -march=native -dM -E - < /dev/null | egrep "NEON|SVE|FP16|FMA" | sort
 *  @endcode
 *
 *  @section arm_targets Choosing Arm Target Generations
 *
 *  Arm CPUs share design IP, but are produced by different vendors, potentially making the platform
 *  even more fragmented than x86. There are 2 important families of SIMD extensions - NEON and SVE.
 *
 *  - Armv8-A: +fp, +simd
 *  - Armv8.1-A: armv8-a, +crc, +lse, +rdma
 *  - Armv8.2-A: armv8.1-a
 *  - Armv8.3-A: armv8.2-a, +pauth
 *  - Armv8.4-A: armv8.3-a, +flagm, +fp16fml, +dotprod
 *  - Armv8.5-A: armv8.4-a, +sb, +ssbs, +predres
 *  - Armv8.6-A: armv8.5-a, +bf16, +i8mm
 *  - Armv8.7-A: armv8.6-a, +ls64
 *  - Armv8.8-A: armv8.7-a, +mops
 *  - Armv8.9-A: armv8.8-a
 *  - Armv9-A: armv8.5-a, +sve, +sve2
 *  - Armv9.1-A: armv9-a, +bf16, +i8mm
 *  - Armv9.2-A: armv9.1-a, +ls64
 *  - Armv9.3-A: armv9.2-a, +mops
 *  - Armv9.4-A: armv9.3-a
 *
 *  SVE has been optional since Armv8.2-A, but it's a requirement for Armv9.0-A.
 *  A 512-bit SVE variant has already been implemented on the Fugaku supercomputer.
 *  A more flexible version, 2x256 SVE, was implemented by the AWS Graviton3 ARM processor.
 *  Here are the most important recent families of CPU cores designed by Arm:
 *
 *  Neoverse N1 implements armv8.2-a, extended with Armv8.4 "dotprod" instructions, and is used in
 *  AWS @b Graviton2 and Ampere @b Altra.
 *
 *  @see Neoverse N1: https://developer.arm.com/Processors/Neoverse%20N1
 *
 *  Neoverse V1 implements armv8.4-a, extended with Armv8.6 bfloat/int8 "matmul" instructions, and
 *  is used in AWS @b Graviton3, which also enables @c sve, @c svebf16, and @c svei8mm.
 *
 *  @see Neoverse V1: https://developer.arm.com/Processors/Neoverse%20V1
 *
 *  Neoverse V2 implements armv9.0 with SVE2 and SVE bit-permutes, and is used in AWS @b Graviton4,
 *  NVIDIA @b Grace, and Google @b Axion.
 *
 *  @see Neoverse V2: https://developer.arm.com/Processors/Neoverse%20V2
 *
 *  The N2 core is very similar to V2, extending it with the same design, and is used by Microsoft
 *  @b Cobalt.
 *
 *  @see Neoverse N2: https://developer.arm.com/Processors/Neoverse%20N2
 *
 *  On the consumer side, Apple is the biggest player with mobile @b A chips and desktop @b M chips.
 *  M1 implements Armv8.5-A, M2 and M3 implement Armv8.6-A, and M4 is expected to have Armv9.1-A.
 *
 *  @section capabilities_references References
 *
 *  @see x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics
 *  @see Detecting target CPU features at compile time: https://stackoverflow.com/a/28939692/2766161
 */

#ifndef NUMKONG_CAPABILITIES_H
#define NUMKONG_CAPABILITIES_H

#include "numkong/types.h" // `nk_u64_t`, `NUMKONG_OS_LINUX_`

#ifdef __cplusplus
extern "C" {
#endif

/** 64-bit bitmask representing SIMD capabilities of the target architecture. */
typedef nk_u64_t nk_capability_t;

/** Serial (non-SIMD) fallback capability. Always available. */
#define nk_cap_serial_index_k 0
#define nk_cap_serial_k       ((nk_capability_t)1 << nk_cap_serial_index_k)

/** Mask representing any capability. */
#define nk_cap_any_k ((nk_capability_t)NUMKONG_U64_MAX)

/*  CPU capabilities, each architecture's in a contiguous run of bits ascending by preference, so
 *  the highest bit a mask shares with a dispatch point's capabilities names the one to run. */

/** Arm NEON, the AArch64 Advanced SIMD baseline. */
#define nk_cap_neon_index_k 1
#define nk_cap_neon_k       ((nk_capability_t)1 << nk_cap_neon_index_k)

/** NEON with half-precision arithmetic, FEAT_FP16. */
#define nk_cap_neonhalf_index_k 2
#define nk_cap_neonhalf_k       ((nk_capability_t)1 << nk_cap_neonhalf_index_k)

/** NEON with BFloat16 dot products, FEAT_BF16. */
#define nk_cap_neonbfdot_index_k 3
#define nk_cap_neonbfdot_k       ((nk_capability_t)1 << nk_cap_neonbfdot_index_k)

/** NEON with half-precision multiply-accumulate into single precision, FEAT_FHM. */
#define nk_cap_neonfhm_index_k 4
#define nk_cap_neonfhm_k       ((nk_capability_t)1 << nk_cap_neonfhm_index_k)

/** NEON with 8-bit integer dot products, FEAT_DotProd. */
#define nk_cap_neonsdot_index_k 5
#define nk_cap_neonsdot_k       ((nk_capability_t)1 << nk_cap_neonsdot_index_k)

/** NEON with FP8 dot products, FEAT_FP8DOT4. */
#define nk_cap_neonfp8_index_k 6
#define nk_cap_neonfp8_k       ((nk_capability_t)1 << nk_cap_neonfp8_index_k)

/** Arm Scalable Vector Extension. */
#define nk_cap_sve_index_k 7
#define nk_cap_sve_k       ((nk_capability_t)1 << nk_cap_sve_index_k)

/** SVE with half-precision arithmetic. */
#define nk_cap_svehalf_index_k 8
#define nk_cap_svehalf_k       ((nk_capability_t)1 << nk_cap_svehalf_index_k)

/** SVE with 8-bit integer matrix multiplies, FEAT_I8MM. */
#define nk_cap_svesdot_index_k 9
#define nk_cap_svesdot_k       ((nk_capability_t)1 << nk_cap_svesdot_index_k)

/** SVE with BFloat16 dot products. */
#define nk_cap_svebfdot_index_k 10
#define nk_cap_svebfdot_k       ((nk_capability_t)1 << nk_cap_svebfdot_index_k)

/** Arm SVE2. */
#define nk_cap_sve2_index_k 11
#define nk_cap_sve2_k       ((nk_capability_t)1 << nk_cap_sve2_index_k)

/** Arm Scalable Matrix Extension. */
#define nk_cap_sme_index_k 12
#define nk_cap_sme_k       ((nk_capability_t)1 << nk_cap_sme_index_k)

/** SME with double-precision outer products, FEAT_SME_F64F64. */
#define nk_cap_smef64_index_k 13
#define nk_cap_smef64_k       ((nk_capability_t)1 << nk_cap_smef64_index_k)

/** SME with binary outer products into 32-bit integers, SME_BI32I32. */
#define nk_cap_smebi32_index_k 14
#define nk_cap_smebi32_k       ((nk_capability_t)1 << nk_cap_smebi32_index_k)

/** x86 AVX2, FMA, F16C, BMI1 and BMI2, as on Intel Haswell. */
#define nk_cap_haswell_index_k 15
#define nk_cap_haswell_k       ((nk_capability_t)1 << nk_cap_haswell_index_k)

/** @c nk_cap_haswell_k plus AVX-VNNI, as on Intel Alder Lake. */
#define nk_cap_alder_index_k 16
#define nk_cap_alder_k       ((nk_capability_t)1 << nk_cap_alder_index_k)

/** @c nk_cap_alder_k plus AVX-VNNI-INT8, as on Intel Sierra Forest. */
#define nk_cap_sierra_index_k 17
#define nk_cap_sierra_k       ((nk_capability_t)1 << nk_cap_sierra_index_k)

/** @c nk_cap_haswell_k plus AVX-512 F, DQ, BW and VL, as on Intel Skylake-X. */
#define nk_cap_skylake_index_k 18
#define nk_cap_skylake_k       ((nk_capability_t)1 << nk_cap_skylake_index_k)

/** @c nk_cap_skylake_k plus AVX-512 VNNI, IFMA, BITALG, VBMI, VBMI2 and VPOPCNTDQ. */
#define nk_cap_icelake_index_k 19
#define nk_cap_icelake_k       ((nk_capability_t)1 << nk_cap_icelake_index_k)

/** @c nk_cap_skylake_k plus AVX-512 BF16 and VNNI, as on AMD Genoa. */
#define nk_cap_genoa_index_k 20
#define nk_cap_genoa_k       ((nk_capability_t)1 << nk_cap_genoa_index_k)

/** @c nk_cap_genoa_k plus AVX-512 VBMI2 and VP2INTERSECT, as on AMD Turin. */
#define nk_cap_turin_index_k 21
#define nk_cap_turin_k       ((nk_capability_t)1 << nk_cap_turin_index_k)

/** @c nk_cap_skylake_k plus AVX-512 FP16, as on Intel Sapphire Rapids. */
#define nk_cap_sapphire_index_k 22
#define nk_cap_sapphire_k       ((nk_capability_t)1 << nk_cap_sapphire_index_k)

/** @c nk_cap_sapphire_k plus AVX10.2, as on Intel Diamond Rapids. */
#define nk_cap_diamond_index_k 23
#define nk_cap_diamond_k       ((nk_capability_t)1 << nk_cap_diamond_index_k)

/** @c nk_cap_sapphire_k and @c nk_cap_genoa_k plus AMX tiles with BF16 and INT8. */
#define nk_cap_sapphireamx_index_k 24
#define nk_cap_sapphireamx_k       ((nk_capability_t)1 << nk_cap_sapphireamx_index_k)

/** @c nk_cap_sapphireamx_k plus AMX-FP16, as on Intel Granite Rapids. */
#define nk_cap_graniteamx_index_k 25
#define nk_cap_graniteamx_k       ((nk_capability_t)1 << nk_cap_graniteamx_index_k)

/** @c nk_cap_sapphireamx_k plus AMX-FP8 and AMX-AVX512, as on Intel Diamond Rapids. */
#define nk_cap_diamondamx_index_k 26
#define nk_cap_diamondamx_k       ((nk_capability_t)1 << nk_cap_diamondamx_index_k)

/** RISC-V Vector extension. */
#define nk_cap_rvv_index_k 27
#define nk_cap_rvv_k       ((nk_capability_t)1 << nk_cap_rvv_index_k)

/** RVV with BFloat16 widening multiply-accumulate, Zvfbfwma. */
#define nk_cap_rvvbf16_index_k 28
#define nk_cap_rvvbf16_k       ((nk_capability_t)1 << nk_cap_rvvbf16_index_k)

/** RVV with half-precision arithmetic, Zvfh. */
#define nk_cap_rvvhalf_index_k 29
#define nk_cap_rvvhalf_k       ((nk_capability_t)1 << nk_cap_rvvhalf_index_k)

/** RVV with basic bit manipulation, Zvbb. */
#define nk_cap_rvvbb_index_k 30
#define nk_cap_rvvbb_k       ((nk_capability_t)1 << nk_cap_rvvbb_index_k)

/** WebAssembly 128-bit SIMD. */
#define nk_cap_v128_index_k 31
#define nk_cap_v128_k       ((nk_capability_t)1 << nk_cap_v128_index_k)

/** WebAssembly Relaxed SIMD. */
#define nk_cap_v128relaxed_index_k 32
#define nk_cap_v128relaxed_k       ((nk_capability_t)1 << nk_cap_v128relaxed_index_k)

/** IBM Power VSX at ISA 3.0, as on POWER9. */
#define nk_cap_powervsx_index_k 33
#define nk_cap_powervsx_k       ((nk_capability_t)1 << nk_cap_powervsx_index_k)

/** LoongArch LASX 256-bit SIMD. */
#define nk_cap_loongsonasx_index_k 34
#define nk_cap_loongsonasx_k       ((nk_capability_t)1 << nk_cap_loongsonasx_index_k)

/** @c nk_cap_powervsx_k plus the ISA 3.1 Matrix-Multiply Assist accumulators, as on POWER10. */
#define nk_cap_powermma_index_k 35
#define nk_cap_powermma_k       ((nk_capability_t)1 << nk_cap_powermma_index_k)

/*  GPU capabilities, as each vendor's <tt>nk_capabilities_*_\<vendor\></tt> functions report them for one
 *  device, grouped by vendor above bit 47 to leave room for more CPU capabilities and more per
 *  vendor. Each vendor's baseline runs on all its devices, as @c serial runs on every CPU. */

/** NVIDIA baseline, which runs on every CUDA device. */
#define nk_cap_cuda_index_k 48
#define nk_cap_cuda_k       ((nk_capability_t)1 << nk_cap_cuda_index_k)

/** NVIDIA @c mma.sync tensor cores, as on A100, RTX 30 and every later GPU. */
#define nk_cap_ampere_index_k 49
#define nk_cap_ampere_k       ((nk_capability_t)1 << nk_cap_ampere_index_k)

/** NVIDIA FP8 @c mma.sync, as on L4, L40S, RTX 40 and every later GPU. */
#define nk_cap_ada_index_k 50
#define nk_cap_ada_k       ((nk_capability_t)1 << nk_cap_ada_index_k)

/** NVIDIA Hopper @c wgmma, as on H100, H200 and GH200. */
#define nk_cap_hopper_index_k 51
#define nk_cap_hopper_k       ((nk_capability_t)1 << nk_cap_hopper_index_k)

/** NVIDIA datacenter Blackwell @c tcgen05, as on B200, GB200, B300 and GB300. */
#define nk_cap_blackwell_index_k 52
#define nk_cap_blackwell_k       ((nk_capability_t)1 << nk_cap_blackwell_index_k)

/** NVIDIA FP6 and FP4 @c mma.sync of compute capability 12.x, as on RTX 50 and DGX Spark. */
#define nk_cap_blackwellrtx_index_k 53
#define nk_cap_blackwellrtx_k       ((nk_capability_t)1 << nk_cap_blackwellrtx_index_k)

/** NVIDIA Blackwell Ultra @c tcgen05.ld.red, as on B300 and GB300. */
#define nk_cap_blackwellultra_index_k 54
#define nk_cap_blackwellultra_k       ((nk_capability_t)1 << nk_cap_blackwellultra_index_k)

/** AMD baseline, which runs on every ROCm device. */
#define nk_cap_rocm_index_k 56
#define nk_cap_rocm_k       ((nk_capability_t)1 << nk_cap_rocm_index_k)

/** AMD MFMA over BF16, F16 and I8, as on MI300X, MI300A and MI325X. */
#define nk_cap_cdna3_index_k 57
#define nk_cap_cdna3_k       ((nk_capability_t)1 << nk_cap_cdna3_index_k)

/** AMD MFMA over FP8, FP6 and FP4, as on MI350X and MI355X. */
#define nk_cap_cdna4_index_k 58
#define nk_cap_cdna4_k       ((nk_capability_t)1 << nk_cap_cdna4_index_k)

/** AMD WMMA over FP8, FP6 and FP4, as on MI450X, MI455X and MI430X. */
#define nk_cap_cdna5_index_k 59
#define nk_cap_cdna5_k       ((nk_capability_t)1 << nk_cap_cdna5_index_k)

/** Apple baseline, which runs on every Metal device. */
#define nk_cap_metal_index_k 60
#define nk_cap_metal_k       ((nk_capability_t)1 << nk_cap_metal_index_k)

/** Apple GPU family 9, as on M3 and M4. */
#define nk_cap_apple9_index_k 61
#define nk_cap_apple9_k       ((nk_capability_t)1 << nk_cap_apple9_index_k)

/** Apple GPU family 10, as on M5. */
#define nk_cap_apple10_index_k 62
#define nk_cap_apple10_k       ((nk_capability_t)1 << nk_cap_apple10_index_k)

/** Every GPU capability above, which the CPU dispatch never detects, compiles or enables. */
#define nk_cap_gpus_k                                                                                                \
    (nk_cap_cuda_k | nk_cap_ampere_k | nk_cap_ada_k | nk_cap_hopper_k | nk_cap_blackwell_k | nk_cap_blackwellrtx_k | \
     nk_cap_blackwellultra_k | nk_cap_rocm_k | nk_cap_cdna3_k | nk_cap_cdna4_k | nk_cap_cdna5_k | nk_cap_metal_k |   \
     nk_cap_apple9_k | nk_cap_apple10_k)

/** Every CPU capability, the bits below the first GPU vendor's. */
#define nk_cap_cpus_k (nk_cap_cuda_k - 1)

/** Buffer size @c nk_capabilities_name never overruns, including its null terminator. */
#define NUMKONG_CAPABILITIES_NAME_CAPACITY 1024

/** Every named capability in bit order, spelled as the bindings parse them; a null name ends the
 *  table. */
static struct {
    char const *name;
    nk_capability_t flag;
} const nk_capability_names_[] = {
    {"serial", nk_cap_serial_k},
    {"neon", nk_cap_neon_k},
    {"neonhalf", nk_cap_neonhalf_k},
    {"neonbfdot", nk_cap_neonbfdot_k},
    {"neonfhm", nk_cap_neonfhm_k},
    {"neonsdot", nk_cap_neonsdot_k},
    {"neonfp8", nk_cap_neonfp8_k},
    {"sve", nk_cap_sve_k},
    {"svehalf", nk_cap_svehalf_k},
    {"svesdot", nk_cap_svesdot_k},
    {"svebfdot", nk_cap_svebfdot_k},
    {"sve2", nk_cap_sve2_k},
    {"sme", nk_cap_sme_k},
    {"smef64", nk_cap_smef64_k},
    {"smebi32", nk_cap_smebi32_k},
    {"haswell", nk_cap_haswell_k},
    {"alder", nk_cap_alder_k},
    {"sierra", nk_cap_sierra_k},
    {"skylake", nk_cap_skylake_k},
    {"icelake", nk_cap_icelake_k},
    {"genoa", nk_cap_genoa_k},
    {"turin", nk_cap_turin_k},
    {"sapphire", nk_cap_sapphire_k},
    {"diamond", nk_cap_diamond_k},
    {"sapphireamx", nk_cap_sapphireamx_k},
    {"graniteamx", nk_cap_graniteamx_k},
    {"diamondamx", nk_cap_diamondamx_k},
    {"rvv", nk_cap_rvv_k},
    {"rvvbf16", nk_cap_rvvbf16_k},
    {"rvvhalf", nk_cap_rvvhalf_k},
    {"rvvbb", nk_cap_rvvbb_k},
    {"v128", nk_cap_v128_k},
    {"v128relaxed", nk_cap_v128relaxed_k},
    {"powervsx", nk_cap_powervsx_k},
    {"loongsonasx", nk_cap_loongsonasx_k},
    {"powermma", nk_cap_powermma_k},
    {"cuda", nk_cap_cuda_k},
    {"ampere", nk_cap_ampere_k},
    {"ada", nk_cap_ada_k},
    {"hopper", nk_cap_hopper_k},
    {"blackwell", nk_cap_blackwell_k},
    {"blackwellrtx", nk_cap_blackwellrtx_k},
    {"blackwellultra", nk_cap_blackwellultra_k},
    {"rocm", nk_cap_rocm_k},
    {"cdna3", nk_cap_cdna3_k},
    {"cdna4", nk_cap_cdna4_k},
    {"cdna5", nk_cap_cdna5_k},
    {"metal", nk_cap_metal_k},
    {"apple9", nk_cap_apple9_k},
    {"apple10", nk_cap_apple10_k},
    {0, 0},
};

/** Writes the names of @p capabilities into @p buffer, behind @c nk_capabilities_name. */
NUMKONG_CONSTEXPR nk_size_t nk_capabilities_name_(nk_capability_t capabilities, char *buffer, nk_size_t capacity) {
    if (!capacity) return 0;
    nk_size_t length = 0;
    for (nk_size_t entry = 0; nk_capability_names_[entry].name; ++entry) {
        if (!(capabilities & nk_capability_names_[entry].flag)) continue;
        char const *character = length ? "," : "";
        for (; *character && length + 1 < capacity; ++character) buffer[length++] = *character;
        for (character = nk_capability_names_[entry].name; *character && length + 1 < capacity; ++character)
            buffer[length++] = *character;
    }
    buffer[length] = '\0';
    return length;
}

/** Enumeration of supported metric kinds. Some have aliases for convenience/discoverability. */
typedef enum {

    /** Unknown kernel kind. */
    nk_kernel_unknown_k = 0,

    /** Inner product. */
    nk_kernel_dot_k = 'i',

    /** Complex inner product. */
    nk_kernel_vdot_k = 'v',

    /** Angular (cosine) distance. */
    nk_kernel_angular_k = 'a',

    /** Euclidean distance. */
    nk_kernel_euclidean_k = 'e',

    /** Squared Euclidean distance. */
    nk_kernel_sqeuclidean_k = '2',

    /** Hamming (or Manhattan) distance. */
    nk_kernel_hamming_k = 'h',

    /** Jaccard (or Tanimoto) coefficient. */
    nk_kernel_jaccard_k = 'j',

    /** Bilinear form. */
    nk_kernel_bilinear_k = 'b',

    /** Mahalanobis distance. */
    nk_kernel_mahalanobis_k = 'm',

    /** Kullback-Leibler divergence. */
    nk_kernel_kld_k = 'k',

    /** Jensen-Shannon divergence. */
    nk_kernel_jsd_k = 's',

    /** Sparse dot product with weighted indices. */
    nk_kernel_sparse_dot_k = 'd',

    /** Equivalent to unnormalized Jaccard. */
    nk_kernel_sparse_intersect_k = 'x',

    /** Element-wise Scale. */
    nk_kernel_each_scale_k = '*',

    /** Element-wise Sum. */
    nk_kernel_each_sum_k = '+',

    /** Element-wise Weighted Sum. */
    nk_kernel_each_blend_k = 'w',

    /** Element-wise Fused Multiply-Add. */
    nk_kernel_each_fma_k = 'f',

    /** Fused SwiGLU: y = silu(gate) ⊙ up (up = NULL → SiLU). */
    nk_kernel_each_swiglu_k = 'W',

    /** Grouped RMSNorm: y = x · rsqrt(mean(x²) + epsilon) · γ. */
    nk_kernel_each_rmsnorm_k = 'H',

    /** Grouped RMSNorm of F32 dot products into the narrower type they came from. */
    nk_kernel_each_rmscast_k = 'u',

    /** Element-wise sine. */
    nk_kernel_trig_sin_k = 'S',

    /** Element-wise cosine. */
    nk_kernel_trig_cos_k = 'C',

    /** Element-wise arctangent. */
    nk_kernel_trig_atan_k = 'A',

    /** Horizontal moments reduction (sum + sum-of-squares). */
    nk_kernel_reduce_moments_k = 'R',

    /** Horizontal minmax reduction (min + argmin + max + argmax). */
    nk_kernel_reduce_minmax_k = 'X',

    /** GEMM packed buffer size. */
    nk_kernel_dots_pack_size_k = 'P',

    /** GEMM B matrix packing. */
    nk_kernel_dots_pack_k = 'Q',

    /** GEMM computation. */
    nk_kernel_dots_packed_k = 'G',

    /** GEMM packed shape read (width, depth). */
    nk_kernel_dots_packed_shape_k = 'J',

    /** Symmetric Gram matrix A × Aᵀ. */
    nk_kernel_dots_symmetric_k = 'y',

    /** Workspace bytes of the GEMM computation. */
    nk_kernel_dots_packed_workspace_size_k = '!',

    /** Workspace bytes of the symmetric Gram matrix. */
    nk_kernel_dots_symmetric_workspace_size_k = '#',

    /** Workspace bytes of batched angular distances (packed B). */
    nk_kernel_angulars_packed_workspace_size_k = '$',

    /** Workspace bytes of the symmetric angular distance matrix. */
    nk_kernel_angulars_symmetric_workspace_size_k = '%',

    /** Workspace bytes of batched euclidean distances (packed B). */
    nk_kernel_euclideans_packed_workspace_size_k = '&',

    /** Workspace bytes of the symmetric euclidean distance matrix. */
    nk_kernel_euclideans_symmetric_workspace_size_k = ',',

    /** Hamming distance computation. */
    nk_kernel_hammings_packed_k = 'M',

    /** Symmetric Hamming distance matrix A × Aᵀ. */
    nk_kernel_hammings_symmetric_k = 'Y',

    /** Jaccard distance computation. */
    nk_kernel_jaccards_packed_k = 'p',

    /** Symmetric Jaccard distance matrix. */
    nk_kernel_jaccards_symmetric_k = 'Z',

    /** Workspace bytes of the Hamming distance computation. */
    nk_kernel_hammings_packed_workspace_size_k = '.',

    /** Workspace bytes of the symmetric Hamming distance matrix. */
    nk_kernel_hammings_symmetric_workspace_size_k = ':',

    /** Workspace bytes of the Jaccard distance computation. */
    nk_kernel_jaccards_packed_workspace_size_k = ';',

    /** Workspace bytes of the symmetric Jaccard distance matrix. */
    nk_kernel_jaccards_symmetric_workspace_size_k = '=',

    /** Batched angular distances (packed B). */
    nk_kernel_angulars_packed_k = 'N',

    /** Symmetric angular distance matrix. */
    nk_kernel_angulars_symmetric_k = 'n',

    /** Batched euclidean distances (packed B). */
    nk_kernel_euclideans_packed_k = 'E',

    /** Symmetric euclidean distance matrix. */
    nk_kernel_euclideans_symmetric_k = 'D',

    /** MaxSim packed buffer size. */
    nk_kernel_maxsim_pack_size_k = 'L',

    /** MaxSim vector packing. */
    nk_kernel_maxsim_pack_k = 'l',

    /** MaxSim late-interaction computation. */
    nk_kernel_maxsim_packed_k = 'T',

    /** MaxSim packed shape read (vectors, depth). */
    nk_kernel_maxsim_packed_shape_k = 'q',

    /** Packed cloud buffer size in bytes. */
    nk_kernel_shapes_pack_size_k = '3',

    /** Cloud packing: centering, principal axes and the pose header. */
    nk_kernel_shapes_pack_k = '4',

    /** Cloud packing that also divides each cloud by its RMS distance from the centroid. */
    nk_kernel_shapes_pack_normalized_k = '7',

    /** Packed cloud read (points, centroid, frame, scale, spreads, largest radius). */
    nk_kernel_shapes_packed_shape_k = '0',

    /** Packed cloud restoration into the input type, the inverse of packing. */
    nk_kernel_shapes_unpack_k = '1',

    /** Workspace bytes of many-to-many Chamfer. */
    nk_kernel_chamfers_workspace_size_k = '~',

    /** Many-to-many Chamfer over principal-frame flips. */
    nk_kernel_chamfers_packed_k = '8',

    /** Many-to-many Hausdorff over principal-frame flips. */
    nk_kernel_hausdorffs_packed_k = '9',

    /** Workspace bytes of many-to-many Wasserstein-Procrustes. */
    nk_kernel_wassersteinprocrustes_workspace_size_k = '5',

    /** Many-to-many Wasserstein-Procrustes: exact transport alternated with Kabsch. */
    nk_kernel_wassersteinprocrustes_packed_k = 't',

    /** Workspace bytes of many-to-many Gromov-Wasserstein. */
    nk_kernel_gromovwassersteins_workspace_size_k = '6',

    /** Many-to-many Gromov-Wasserstein over plain distances within each cloud. */
    nk_kernel_gromovwassersteins_packed_k = 'c',

    /** Many-to-many superposed RMSD between frames matched point by point. */
    nk_kernel_rmsds_packed_k = '@',

    /** Attention packed KV-cache Buffer size in Bytes. */
    nk_kernel_attention_pack_size_k = 'B',

    /** Attention K/V packing (KV cache → the capability's layout). */
    nk_kernel_attention_pack_k = 'V',

    /** Fused, Flash-style, attention under a band of visible keys. */
    nk_kernel_attention_packed_k = 'F',

    /** Gradients of banded attention. */
    nk_kernel_attention_packed_gradients_k = 'g',

    /** Workspace bytes of banded attention. */
    nk_kernel_attention_packed_workspace_size_k = '[',

    /** Workspace bytes of the gradients of banded attention. */
    nk_kernel_attention_packed_gradients_workspace_size_k = ']',

    /** Attention packed shape read (heads, depth, segments). */
    nk_kernel_attention_packed_shape_k = 'z',

    /** NeoX split-half rotary position embedding (in place). */
    nk_kernel_attention_rope_k = 'I',

    /** Many-to-many squared Euclidean distances between 3D points. */
    nk_kernel_sqeuclideans3d_k = 'r',

    /** Haversine pack buffer size in bytes. */
    nk_kernel_haversines_pack_size_k = '(',

    /** Haversine packing: latitudes and longitudes into unit vectors. */
    nk_kernel_haversines_pack_k = '<',

    /** Many-to-many great-circle distances on the mean sphere. */
    nk_kernel_haversines_packed_k = 'o',

    /** Vincenty pack buffer size in bytes. */
    nk_kernel_vincentys_pack_size_k = ')',

    /** Vincenty packing: longitudes and reduced latitudes. */
    nk_kernel_vincentys_pack_k = '>',

    /** Many-to-many geodesic distances on the ellipsoid. */
    nk_kernel_vincentys_packed_k = 'O',

    /** Terrain pack buffer size in bytes. */
    nk_kernel_terrain_pack_size_k = 'U',

    /** Terrain packing: the maxima pyramid of a height grid. */
    nk_kernel_terrain_pack_k = 'K',

    /** Bilinear terrain heights under points. */
    nk_kernel_terrain_heights_packed_k = '^',

    /** Many-to-many terrain clearances of sight lines. */
    nk_kernel_clearances_packed_k = '/',

    /** Type casting from one type to another. */
    nk_kernel_cast_k = '-',
} nk_kernel_kind_t;

/** Canonical name of a kernel kind - the spelling bindings parse and interchange formats carry;
 *  "unknown" for unrecognized values. */
NUMKONG_CONSTEXPR char const *nk_kernel_name(nk_kernel_kind_t kind) {
    switch (kind) {
    case nk_kernel_unknown_k: return "unknown";
    case nk_kernel_dot_k: return "dot";
    case nk_kernel_vdot_k: return "vdot";
    case nk_kernel_angular_k: return "angular";
    case nk_kernel_euclidean_k: return "euclidean";
    case nk_kernel_sqeuclidean_k: return "sqeuclidean";
    case nk_kernel_hamming_k: return "hamming";
    case nk_kernel_jaccard_k: return "jaccard";
    case nk_kernel_bilinear_k: return "bilinear";
    case nk_kernel_mahalanobis_k: return "mahalanobis";
    case nk_kernel_kld_k: return "kld";
    case nk_kernel_jsd_k: return "jsd";
    case nk_kernel_sparse_dot_k: return "sparse_dot";
    case nk_kernel_sparse_intersect_k: return "sparse_intersect";
    case nk_kernel_each_scale_k: return "each_scale";
    case nk_kernel_each_sum_k: return "each_sum";
    case nk_kernel_each_blend_k: return "each_blend";
    case nk_kernel_each_fma_k: return "each_fma";
    case nk_kernel_each_swiglu_k: return "each_swiglu";
    case nk_kernel_each_rmsnorm_k: return "each_rmsnorm";
    case nk_kernel_each_rmscast_k: return "each_rmscast";
    case nk_kernel_trig_sin_k: return "trig_sin";
    case nk_kernel_trig_cos_k: return "trig_cos";
    case nk_kernel_trig_atan_k: return "trig_atan";
    case nk_kernel_reduce_moments_k: return "reduce_moments";
    case nk_kernel_reduce_minmax_k: return "reduce_minmax";
    case nk_kernel_dots_pack_size_k: return "dots_pack_size";
    case nk_kernel_dots_pack_k: return "dots_pack";
    case nk_kernel_dots_packed_k: return "dots_packed";
    case nk_kernel_dots_packed_shape_k: return "dots_packed_shape";
    case nk_kernel_dots_symmetric_k: return "dots_symmetric";
    case nk_kernel_dots_packed_workspace_size_k: return "dots_packed_workspace_size";
    case nk_kernel_dots_symmetric_workspace_size_k: return "dots_symmetric_workspace_size";
    case nk_kernel_angulars_packed_workspace_size_k: return "angulars_packed_workspace_size";
    case nk_kernel_angulars_symmetric_workspace_size_k: return "angulars_symmetric_workspace_size";
    case nk_kernel_euclideans_packed_workspace_size_k: return "euclideans_packed_workspace_size";
    case nk_kernel_euclideans_symmetric_workspace_size_k: return "euclideans_symmetric_workspace_size";
    case nk_kernel_hammings_packed_k: return "hammings_packed";
    case nk_kernel_hammings_symmetric_k: return "hammings_symmetric";
    case nk_kernel_jaccards_packed_k: return "jaccards_packed";
    case nk_kernel_jaccards_symmetric_k: return "jaccards_symmetric";
    case nk_kernel_hammings_packed_workspace_size_k: return "hammings_packed_workspace_size";
    case nk_kernel_hammings_symmetric_workspace_size_k: return "hammings_symmetric_workspace_size";
    case nk_kernel_jaccards_packed_workspace_size_k: return "jaccards_packed_workspace_size";
    case nk_kernel_jaccards_symmetric_workspace_size_k: return "jaccards_symmetric_workspace_size";
    case nk_kernel_angulars_packed_k: return "angulars_packed";
    case nk_kernel_angulars_symmetric_k: return "angulars_symmetric";
    case nk_kernel_euclideans_packed_k: return "euclideans_packed";
    case nk_kernel_euclideans_symmetric_k: return "euclideans_symmetric";
    case nk_kernel_maxsim_pack_size_k: return "maxsim_pack_size";
    case nk_kernel_maxsim_pack_k: return "maxsim_pack";
    case nk_kernel_maxsim_packed_k: return "maxsim_packed";
    case nk_kernel_maxsim_packed_shape_k: return "maxsim_packed_shape";
    case nk_kernel_shapes_pack_size_k: return "shapes_pack_size";
    case nk_kernel_shapes_pack_k: return "shapes_pack";
    case nk_kernel_shapes_pack_normalized_k: return "shapes_pack_normalized";
    case nk_kernel_shapes_packed_shape_k: return "shapes_packed_shape";
    case nk_kernel_shapes_unpack_k: return "shapes_unpack";
    case nk_kernel_chamfers_workspace_size_k: return "chamfers_workspace_size";
    case nk_kernel_chamfers_packed_k: return "chamfers_packed";
    case nk_kernel_hausdorffs_packed_k: return "hausdorffs_packed";
    case nk_kernel_wassersteinprocrustes_workspace_size_k: return "wassersteinprocrustes_workspace_size";
    case nk_kernel_wassersteinprocrustes_packed_k: return "wassersteinprocrustes_packed";
    case nk_kernel_gromovwassersteins_workspace_size_k: return "gromovwassersteins_workspace_size";
    case nk_kernel_gromovwassersteins_packed_k: return "gromovwassersteins_packed";
    case nk_kernel_rmsds_packed_k: return "rmsds_packed";
    case nk_kernel_attention_pack_size_k: return "attention_pack_size";
    case nk_kernel_attention_pack_k: return "attention_pack";
    case nk_kernel_attention_packed_k: return "attention_packed";
    case nk_kernel_attention_packed_gradients_k: return "attention_packed_gradients";
    case nk_kernel_attention_packed_workspace_size_k: return "attention_packed_workspace_size";
    case nk_kernel_attention_packed_gradients_workspace_size_k: return "attention_packed_gradients_workspace_size";
    case nk_kernel_attention_packed_shape_k: return "attention_packed_shape";
    case nk_kernel_attention_rope_k: return "attention_rope";
    case nk_kernel_sqeuclideans3d_k: return "sqeuclideans3d";
    case nk_kernel_haversines_pack_size_k: return "haversines_pack_size";
    case nk_kernel_haversines_pack_k: return "haversines_pack";
    case nk_kernel_haversines_packed_k: return "haversines_packed";
    case nk_kernel_vincentys_pack_size_k: return "vincentys_pack_size";
    case nk_kernel_vincentys_pack_k: return "vincentys_pack";
    case nk_kernel_vincentys_packed_k: return "vincentys_packed";
    case nk_kernel_terrain_pack_size_k: return "terrain_pack_size";
    case nk_kernel_terrain_pack_k: return "terrain_pack";
    case nk_kernel_terrain_heights_packed_k: return "terrain_heights_packed";
    case nk_kernel_clearances_packed_k: return "clearances_packed";
    case nk_kernel_cast_k: return "cast";
    default: return "unknown";
    }
}

/** Inverse of @c nk_kernel_name over an explicit-length string; @c nk_kernel_unknown_k for
 *  unrecognized names. */
NUMKONG_CONSTEXPR nk_kernel_kind_t nk_kernel_named(char const *name, nk_size_t length) {
    if (nk_same_literal_(name, length, "dot")) return nk_kernel_dot_k;
    if (nk_same_literal_(name, length, "vdot")) return nk_kernel_vdot_k;
    if (nk_same_literal_(name, length, "angular")) return nk_kernel_angular_k;
    if (nk_same_literal_(name, length, "euclidean")) return nk_kernel_euclidean_k;
    if (nk_same_literal_(name, length, "sqeuclidean")) return nk_kernel_sqeuclidean_k;
    if (nk_same_literal_(name, length, "hamming")) return nk_kernel_hamming_k;
    if (nk_same_literal_(name, length, "jaccard")) return nk_kernel_jaccard_k;
    if (nk_same_literal_(name, length, "bilinear")) return nk_kernel_bilinear_k;
    if (nk_same_literal_(name, length, "mahalanobis")) return nk_kernel_mahalanobis_k;
    if (nk_same_literal_(name, length, "kld")) return nk_kernel_kld_k;
    if (nk_same_literal_(name, length, "jsd")) return nk_kernel_jsd_k;
    if (nk_same_literal_(name, length, "sparse_dot")) return nk_kernel_sparse_dot_k;
    if (nk_same_literal_(name, length, "sparse_intersect")) return nk_kernel_sparse_intersect_k;
    if (nk_same_literal_(name, length, "each_scale")) return nk_kernel_each_scale_k;
    if (nk_same_literal_(name, length, "each_sum")) return nk_kernel_each_sum_k;
    if (nk_same_literal_(name, length, "each_blend")) return nk_kernel_each_blend_k;
    if (nk_same_literal_(name, length, "each_fma")) return nk_kernel_each_fma_k;
    if (nk_same_literal_(name, length, "each_swiglu")) return nk_kernel_each_swiglu_k;
    if (nk_same_literal_(name, length, "each_rmsnorm")) return nk_kernel_each_rmsnorm_k;
    if (nk_same_literal_(name, length, "each_rmscast")) return nk_kernel_each_rmscast_k;
    if (nk_same_literal_(name, length, "trig_sin")) return nk_kernel_trig_sin_k;
    if (nk_same_literal_(name, length, "trig_cos")) return nk_kernel_trig_cos_k;
    if (nk_same_literal_(name, length, "trig_atan")) return nk_kernel_trig_atan_k;
    if (nk_same_literal_(name, length, "reduce_moments")) return nk_kernel_reduce_moments_k;
    if (nk_same_literal_(name, length, "reduce_minmax")) return nk_kernel_reduce_minmax_k;
    if (nk_same_literal_(name, length, "dots_pack_size")) return nk_kernel_dots_pack_size_k;
    if (nk_same_literal_(name, length, "dots_pack")) return nk_kernel_dots_pack_k;
    if (nk_same_literal_(name, length, "dots_packed")) return nk_kernel_dots_packed_k;
    if (nk_same_literal_(name, length, "dots_packed_shape")) return nk_kernel_dots_packed_shape_k;
    if (nk_same_literal_(name, length, "dots_symmetric")) return nk_kernel_dots_symmetric_k;
    if (nk_same_literal_(name, length, "dots_packed_workspace_size")) return nk_kernel_dots_packed_workspace_size_k;
    if (nk_same_literal_(name, length, "dots_symmetric_workspace_size"))
        return nk_kernel_dots_symmetric_workspace_size_k;
    if (nk_same_literal_(name, length, "angulars_packed_workspace_size"))
        return nk_kernel_angulars_packed_workspace_size_k;
    if (nk_same_literal_(name, length, "angulars_symmetric_workspace_size"))
        return nk_kernel_angulars_symmetric_workspace_size_k;
    if (nk_same_literal_(name, length, "euclideans_packed_workspace_size"))
        return nk_kernel_euclideans_packed_workspace_size_k;
    if (nk_same_literal_(name, length, "euclideans_symmetric_workspace_size"))
        return nk_kernel_euclideans_symmetric_workspace_size_k;
    if (nk_same_literal_(name, length, "hammings_packed")) return nk_kernel_hammings_packed_k;
    if (nk_same_literal_(name, length, "hammings_symmetric")) return nk_kernel_hammings_symmetric_k;
    if (nk_same_literal_(name, length, "jaccards_packed")) return nk_kernel_jaccards_packed_k;
    if (nk_same_literal_(name, length, "jaccards_symmetric")) return nk_kernel_jaccards_symmetric_k;
    if (nk_same_literal_(name, length, "hammings_packed_workspace_size"))
        return nk_kernel_hammings_packed_workspace_size_k;
    if (nk_same_literal_(name, length, "hammings_symmetric_workspace_size"))
        return nk_kernel_hammings_symmetric_workspace_size_k;
    if (nk_same_literal_(name, length, "jaccards_packed_workspace_size"))
        return nk_kernel_jaccards_packed_workspace_size_k;
    if (nk_same_literal_(name, length, "jaccards_symmetric_workspace_size"))
        return nk_kernel_jaccards_symmetric_workspace_size_k;
    if (nk_same_literal_(name, length, "angulars_packed")) return nk_kernel_angulars_packed_k;
    if (nk_same_literal_(name, length, "angulars_symmetric")) return nk_kernel_angulars_symmetric_k;
    if (nk_same_literal_(name, length, "euclideans_packed")) return nk_kernel_euclideans_packed_k;
    if (nk_same_literal_(name, length, "euclideans_symmetric")) return nk_kernel_euclideans_symmetric_k;
    if (nk_same_literal_(name, length, "maxsim_pack_size")) return nk_kernel_maxsim_pack_size_k;
    if (nk_same_literal_(name, length, "maxsim_pack")) return nk_kernel_maxsim_pack_k;
    if (nk_same_literal_(name, length, "maxsim_packed")) return nk_kernel_maxsim_packed_k;
    if (nk_same_literal_(name, length, "maxsim_packed_shape")) return nk_kernel_maxsim_packed_shape_k;
    if (nk_same_literal_(name, length, "shapes_pack_size")) return nk_kernel_shapes_pack_size_k;
    if (nk_same_literal_(name, length, "shapes_pack")) return nk_kernel_shapes_pack_k;
    if (nk_same_literal_(name, length, "shapes_pack_normalized")) return nk_kernel_shapes_pack_normalized_k;
    if (nk_same_literal_(name, length, "shapes_packed_shape")) return nk_kernel_shapes_packed_shape_k;
    if (nk_same_literal_(name, length, "shapes_unpack")) return nk_kernel_shapes_unpack_k;
    if (nk_same_literal_(name, length, "chamfers_workspace_size")) return nk_kernel_chamfers_workspace_size_k;
    if (nk_same_literal_(name, length, "chamfers_packed")) return nk_kernel_chamfers_packed_k;
    if (nk_same_literal_(name, length, "hausdorffs_packed")) return nk_kernel_hausdorffs_packed_k;
    if (nk_same_literal_(name, length, "wassersteinprocrustes_workspace_size"))
        return nk_kernel_wassersteinprocrustes_workspace_size_k;
    if (nk_same_literal_(name, length, "wassersteinprocrustes_packed")) return nk_kernel_wassersteinprocrustes_packed_k;
    if (nk_same_literal_(name, length, "gromovwassersteins_workspace_size"))
        return nk_kernel_gromovwassersteins_workspace_size_k;
    if (nk_same_literal_(name, length, "gromovwassersteins_packed")) return nk_kernel_gromovwassersteins_packed_k;
    if (nk_same_literal_(name, length, "rmsds_packed")) return nk_kernel_rmsds_packed_k;
    if (nk_same_literal_(name, length, "attention_pack_size")) return nk_kernel_attention_pack_size_k;
    if (nk_same_literal_(name, length, "attention_pack")) return nk_kernel_attention_pack_k;
    if (nk_same_literal_(name, length, "attention_packed")) return nk_kernel_attention_packed_k;
    if (nk_same_literal_(name, length, "attention_packed_gradients")) return nk_kernel_attention_packed_gradients_k;
    if (nk_same_literal_(name, length, "attention_packed_workspace_size"))
        return nk_kernel_attention_packed_workspace_size_k;
    if (nk_same_literal_(name, length, "attention_packed_gradients_workspace_size"))
        return nk_kernel_attention_packed_gradients_workspace_size_k;
    if (nk_same_literal_(name, length, "attention_packed_shape")) return nk_kernel_attention_packed_shape_k;
    if (nk_same_literal_(name, length, "attention_rope")) return nk_kernel_attention_rope_k;
    if (nk_same_literal_(name, length, "sqeuclideans3d")) return nk_kernel_sqeuclideans3d_k;
    if (nk_same_literal_(name, length, "haversines_pack_size")) return nk_kernel_haversines_pack_size_k;
    if (nk_same_literal_(name, length, "haversines_pack")) return nk_kernel_haversines_pack_k;
    if (nk_same_literal_(name, length, "haversines_packed")) return nk_kernel_haversines_packed_k;
    if (nk_same_literal_(name, length, "vincentys_pack_size")) return nk_kernel_vincentys_pack_size_k;
    if (nk_same_literal_(name, length, "vincentys_pack")) return nk_kernel_vincentys_pack_k;
    if (nk_same_literal_(name, length, "vincentys_packed")) return nk_kernel_vincentys_packed_k;
    if (nk_same_literal_(name, length, "terrain_pack_size")) return nk_kernel_terrain_pack_size_k;
    if (nk_same_literal_(name, length, "terrain_pack")) return nk_kernel_terrain_pack_k;
    if (nk_same_literal_(name, length, "terrain_heights_packed")) return nk_kernel_terrain_heights_packed_k;
    if (nk_same_literal_(name, length, "clearances_packed")) return nk_kernel_clearances_packed_k;
    if (nk_same_literal_(name, length, "cast")) return nk_kernel_cast_k;
    return nk_kernel_unknown_k;
}

typedef nk_status_t (*nk_metric_dense_punned_t)(void const *a, void const *b, nk_size_t dimensions, void *result,
                                                nk_stream_t stream);

typedef nk_status_t (*nk_sparse_intersect_punned_t)(void const *a, void const *b, nk_size_t a_length,
                                                    nk_size_t b_length, void *result, nk_size_t *count,
                                                    nk_stream_t stream);

typedef nk_status_t (*nk_sparse_dot_punned_t)(void const *a, void const *b, void const *a_weights,
                                              void const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                              void *product, nk_stream_t stream);

typedef nk_status_t (*nk_metric_curved_punned_t)(void const *a, void const *b, void const *c, nk_size_t dimensions,
                                                 void *result, nk_stream_t stream);

typedef nk_status_t (*nk_sqeuclideans3d_punned_t)(void const *a, nk_size_t rows, void const *b, nk_size_t columns,
                                                  void *results, nk_size_t results_stride, nk_stream_t stream);

/** Pack sizes are host arithmetic, so they take no stream. */
typedef nk_status_t (*nk_geospatial_pack_size_punned_t)(nk_size_t points, nk_size_t *bytes);

typedef nk_status_t (*nk_geospatial_pack_punned_t)(void const *latitudes, void const *longitudes, nk_size_t count,
                                                   void *packed, nk_stream_t stream);

typedef nk_status_t (*nk_geospatial_packed_punned_t)(void const *a_packed, void const *b_packed, void *results,
                                                     nk_size_t results_stride, nk_stream_t stream);

typedef nk_status_t (*nk_terrain_pack_size_punned_t)(nk_size_t rows, nk_size_t columns, nk_size_t *bytes);

typedef nk_status_t (*nk_terrain_pack_punned_t)(void const *heights, nk_size_t rows, nk_size_t columns,
                                                nk_size_t heights_stride, nk_f64_t row_metres, nk_f64_t column_metres,
                                                void *packed, nk_stream_t stream);

typedef nk_status_t (*nk_terrain_heights_packed_punned_t)(void const *heights, void const *packed, void const *points,
                                                          nk_size_t count, void *results, nk_stream_t stream);

typedef nk_status_t (*nk_clearances_packed_punned_t)(void const *heights, void const *packed, void const *viewpoints,
                                                     void const *views, nk_size_t rows, void const *targets,
                                                     nk_size_t columns, nk_f64_t refraction, void *results,
                                                     nk_size_t results_stride, nk_stream_t stream);

typedef nk_status_t (*nk_each_scale_punned_t)(void const *a, nk_size_t count, void const *alpha, void const *beta,
                                              void *y, nk_stream_t stream);

typedef nk_status_t (*nk_each_sum_punned_t)(void const *a, void const *b, nk_size_t count, void *y, nk_stream_t stream);

typedef nk_status_t (*nk_each_blend_punned_t)(void const *a, void const *b, nk_size_t count, void const *alpha,
                                              void const *beta, void *y, nk_stream_t stream);

typedef nk_status_t (*nk_each_fma_punned_t)(void const *a, void const *b, void const *c, nk_size_t count,
                                            void const *alpha, void const *beta, void *y, nk_stream_t stream);

typedef nk_status_t (*nk_kernel_trig_punned_t)(void const *x, nk_size_t count, void *y, nk_stream_t stream);

typedef nk_status_t (*nk_reduce_moments_punned_t)(void const *data, nk_size_t count, nk_size_t stride, void *sum,
                                                  void *sumsq, nk_stream_t stream);

typedef nk_status_t (*nk_reduce_minmax_punned_t)(void const *data, nk_size_t count, nk_size_t stride, void *min_value,
                                                 nk_size_t *min_index, void *max_value, nk_size_t *max_index,
                                                 nk_stream_t stream);

typedef nk_status_t (*nk_each_rmsnorm_punned_t)(void const *x, void const *gamma, void *y, nk_size_t rows,
                                                nk_size_t groups, nk_size_t columns, nk_size_t x_stride,
                                                nk_size_t y_stride, nk_f32_t epsilon, nk_stream_t stream);

typedef nk_status_t (*nk_each_swiglu_punned_t)(void const *gate, void const *up, void *y, nk_size_t rows,
                                               nk_size_t columns, nk_size_t gate_stride, nk_size_t up_stride,
                                               nk_size_t y_stride, nk_f32_t gate_scale, nk_f32_t output_scale,
                                               nk_stream_t stream);

/** Pack sizes are host arithmetic, so they take no stream. */
typedef nk_status_t (*nk_dots_pack_size_punned_t)(nk_size_t columns, nk_size_t depth, nk_size_t *bytes);

typedef nk_status_t (*nk_dots_packed_shape_punned_t)(void const *packed, nk_size_t *columns, nk_size_t *depth,
                                                     nk_stream_t stream);

typedef nk_status_t (*nk_dots_pack_punned_t)(void const *b, nk_size_t columns, nk_size_t depth, nk_size_t b_stride,
                                             void *b_packed, nk_size_t columns_begin, nk_size_t columns_end,
                                             nk_stream_t stream);

typedef nk_status_t (*nk_maxsim_pack_punned_t)(void const *vectors, nk_size_t vector_count, nk_size_t depth,
                                               nk_size_t stride, void *packed, nk_stream_t stream);

typedef nk_status_t (*nk_dots_packed_punned_t)(void const *a, void const *b_packed, void *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, void *workspace, nk_size_t workspace_bytes,
                                               nk_stream_t stream);

typedef nk_status_t (*nk_dots_packed_workspace_size_punned_t)(nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                              nk_size_t *bytes);

typedef nk_status_t (*nk_dots_symmetric_punned_t)(void const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, void *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, void *workspace,
                                                  nk_size_t workspace_bytes, nk_stream_t stream);

typedef nk_status_t (*nk_dots_symmetric_workspace_size_punned_t)(nk_size_t vector_count, nk_size_t depth,
                                                                 nk_size_t rows_begin, nk_size_t rows_end,
                                                                 nk_size_t *bytes);

/** Bit vectors have no scaled form, so the set batches keep their unscaled shapes. */
typedef nk_status_t (*nk_sets_packed_punned_t)(void const *a, void const *b_packed, void *c, nk_size_t rows,
                                               nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                               nk_size_t c_stride, void *workspace, nk_size_t workspace_bytes,
                                               nk_stream_t stream);
typedef nk_status_t (*nk_sets_symmetric_punned_t)(void const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                  nk_size_t stride, void *result, nk_size_t result_stride,
                                                  nk_size_t rows_begin, nk_size_t rows_end, void *workspace,
                                                  nk_size_t workspace_bytes, nk_stream_t stream);

typedef nk_sets_packed_punned_t nk_hammings_packed_punned_t;
typedef nk_sets_symmetric_punned_t nk_hammings_symmetric_punned_t;
typedef nk_sets_packed_punned_t nk_jaccards_packed_punned_t;
typedef nk_sets_symmetric_punned_t nk_jaccards_symmetric_punned_t;
typedef nk_dots_packed_workspace_size_punned_t nk_hammings_packed_workspace_size_punned_t;
typedef nk_dots_symmetric_workspace_size_punned_t nk_hammings_symmetric_workspace_size_punned_t;
typedef nk_dots_packed_workspace_size_punned_t nk_jaccards_packed_workspace_size_punned_t;
typedef nk_dots_symmetric_workspace_size_punned_t nk_jaccards_symmetric_workspace_size_punned_t;
typedef nk_dots_packed_punned_t nk_angulars_packed_punned_t;
typedef nk_dots_symmetric_punned_t nk_angulars_symmetric_punned_t;
typedef nk_dots_packed_punned_t nk_euclideans_packed_punned_t;
typedef nk_dots_symmetric_punned_t nk_euclideans_symmetric_punned_t;
typedef nk_dots_packed_workspace_size_punned_t nk_angulars_packed_workspace_size_punned_t;
typedef nk_dots_symmetric_workspace_size_punned_t nk_angulars_symmetric_workspace_size_punned_t;
typedef nk_dots_packed_workspace_size_punned_t nk_euclideans_packed_workspace_size_punned_t;
typedef nk_dots_symmetric_workspace_size_punned_t nk_euclideans_symmetric_workspace_size_punned_t;

typedef nk_status_t (*nk_maxsim_packed_punned_t)(void const *query_packed, void const *document_packed,
                                                 nk_size_t query_count, nk_size_t document_count, nk_size_t depth,
                                                 void *result, nk_stream_t stream);

typedef nk_status_t (*nk_shapes_pack_size_punned_t)(nk_size_t points, nk_size_t *bytes);

typedef nk_status_t (*nk_shapes_pack_punned_t)(void const *points, nk_size_t count, void *packed, nk_stream_t stream);

typedef nk_status_t (*nk_shapes_packed_shape_punned_t)(void const *packed, nk_size_t *points, void *centroid,
                                                       void *frame, void *scale, void *spreads, void *max_radius,
                                                       nk_stream_t stream);

typedef nk_status_t (*nk_shapes_unpack_punned_t)(void const *packed, void *points, nk_stream_t stream);

/** Workspace sizes are host arithmetic, like pack sizes, so they take no stream. */
typedef nk_status_t (*nk_shapes_workspace_size_punned_t)(nk_size_t row_points, nk_size_t column_points,
                                                         nk_size_t *bytes);

typedef nk_status_t (*nk_chamfers_packed_punned_t)(void const *const *a_packed, nk_size_t rows,
                                                   void const *const *b_packed, nk_size_t columns, void *results,
                                                   nk_size_t results_stride, void *rotations, void const *limit,
                                                   void *workspace, nk_size_t workspace_bytes, nk_stream_t stream);

typedef nk_status_t (*nk_hausdorffs_packed_punned_t)(void const *const *a_packed, nk_size_t rows,
                                                     void const *const *b_packed, nk_size_t columns, void *results,
                                                     nk_size_t results_stride, void *rotations, void const *limit,
                                                     nk_stream_t stream);

typedef nk_status_t (*nk_rmsds_packed_punned_t)(void const *const *a_packed, nk_size_t rows,
                                                void const *const *b_packed, nk_size_t columns, void *results,
                                                nk_size_t results_stride, nk_stream_t stream);

typedef nk_status_t (*nk_wassersteinprocrustes_packed_punned_t)(void const *const *a_packed, nk_size_t rows,
                                                                void const *const *b_packed, nk_size_t columns,
                                                                void *results, nk_size_t results_stride,
                                                                void *rotations, void *workspace,
                                                                nk_size_t workspace_bytes, nk_stream_t stream);

typedef nk_wassersteinprocrustes_packed_punned_t nk_gromovwassersteins_packed_punned_t;

typedef nk_status_t (*nk_attention_packed_shape_punned_t)(void const *packed, nk_size_t *key_value_head_count,
                                                          nk_size_t *depth, nk_size_t *segments, nk_stream_t stream);

/** Pack sizes are host arithmetic, so they take no stream. */
typedef nk_status_t (*nk_attention_pack_size_punned_t)(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_size_t token_count, nk_size_t segment_count,
                                                       nk_size_t *bytes);

typedef nk_status_t (*nk_attention_pack_punned_t)(nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                  nk_size_t segment_count, void const *keys, nk_size_t key_stride,
                                                  void const *values, nk_size_t value_stride, void *key_value_packed,
                                                  nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream);

typedef nk_status_t (*nk_attention_packed_punned_t)(nk_size_t head_count, nk_size_t key_value_head_count,
                                                    nk_size_t depth, nk_u32_t const *query_offsets,
                                                    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                    nk_size_t keys_after, void const *queries, nk_size_t query_stride,
                                                    void const *key_value_packed, void *output, nk_size_t output_stride,
                                                    nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                    void *workspace, nk_size_t workspace_bytes, nk_stream_t stream);

typedef nk_status_t (*nk_attention_packed_gradients_punned_t)(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after, void const *queries,
    nk_size_t query_stride, void const *key_value_packed, nk_f32_t const *output, nk_f32_t const *output_gradient,
    nk_size_t output_stride, nk_f32_t const *log_sum_exp, nk_f32_t *query_gradient, nk_size_t query_gradient_stride,
    nk_f32_t *key_gradient, nk_f32_t *value_gradient, nk_size_t key_value_gradient_stride, nk_size_t tasks_begin,
    nk_size_t tasks_end, void *workspace, nk_size_t workspace_bytes, nk_stream_t stream);

/** Workspace sizes of attention and its gradients, host arithmetic like pack sizes. */
typedef nk_status_t (*nk_attention_packed_workspace_size_punned_t)(nk_size_t head_count, nk_size_t key_value_head_count,
                                                                   nk_size_t depth, nk_size_t query_token_count,
                                                                   nk_size_t keys_before, nk_size_t keys_after,
                                                                   nk_size_t *bytes);

typedef nk_status_t (*nk_attention_rope_punned_t)(void const *x, void const *cos, void const *sin, void *y,
                                                  nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                  nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream);

typedef nk_status_t (*nk_kernel_cast_punned_t)(void const *from, nk_dtype_t from_dtype, void *to, nk_dtype_t to_dtype,
                                               nk_size_t count, nk_stream_t stream);

/** Any kernel, cast back to its kind's signature before the call. */
typedef void (*nk_kernel_punned_t)(void);

/** The capabilities the <tt>NUMKONG_TARGET_*</tt> macros of the including unit compile, as a
 *  constant for consumers picking kits at compile time; in the library, the mask that
 *  @c nk_capabilities_compiled_cpu reports. Says nothing about the current CPU. */
NUMKONG_CONSTEXPR nk_capability_t nk_capabilities_compiled_cpu_mask(void) {
    nk_capability_t caps = nk_cap_serial_k;
    caps |= nk_cap_haswell_k * NUMKONG_TARGET_HASWELL;
    caps |= nk_cap_skylake_k * NUMKONG_TARGET_SKYLAKE;
    caps |= nk_cap_icelake_k * NUMKONG_TARGET_ICELAKE;
    caps |= nk_cap_genoa_k * NUMKONG_TARGET_GENOA;
    caps |= nk_cap_sapphire_k * NUMKONG_TARGET_SAPPHIRE;
    caps |= nk_cap_sapphireamx_k * NUMKONG_TARGET_SAPPHIREAMX;
    caps |= nk_cap_graniteamx_k * NUMKONG_TARGET_GRANITEAMX;
    caps |= nk_cap_diamondamx_k * NUMKONG_TARGET_DIAMONDAMX;
    caps |= nk_cap_diamond_k * NUMKONG_TARGET_DIAMOND;
    caps |= nk_cap_turin_k * NUMKONG_TARGET_TURIN;
    caps |= nk_cap_alder_k * NUMKONG_TARGET_ALDER;
    caps |= nk_cap_sierra_k * NUMKONG_TARGET_SIERRA;
    caps |= nk_cap_neon_k * NUMKONG_TARGET_NEON;
    caps |= nk_cap_neonhalf_k * NUMKONG_TARGET_NEONHALF;
    caps |= nk_cap_neonsdot_k * NUMKONG_TARGET_NEONSDOT;
    caps |= nk_cap_neonbfdot_k * NUMKONG_TARGET_NEONBFDOT;
    caps |= nk_cap_neonfhm_k * NUMKONG_TARGET_NEONFHM;
    caps |= nk_cap_neonfp8_k * NUMKONG_TARGET_NEONFP8;
    caps |= nk_cap_sve_k * NUMKONG_TARGET_SVE;
    caps |= nk_cap_svehalf_k * NUMKONG_TARGET_SVEHALF;
    caps |= nk_cap_svesdot_k * NUMKONG_TARGET_SVESDOT;
    caps |= nk_cap_svebfdot_k * NUMKONG_TARGET_SVEBFDOT;
    caps |= nk_cap_sve2_k * NUMKONG_TARGET_SVE2;
    caps |= nk_cap_sme_k * NUMKONG_TARGET_SME;
    caps |= nk_cap_smef64_k * NUMKONG_TARGET_SMEF64;
    caps |= nk_cap_smebi32_k * NUMKONG_TARGET_SMEBI32;
    caps |= nk_cap_rvv_k * NUMKONG_TARGET_RVV;
    caps |= nk_cap_rvvhalf_k * NUMKONG_TARGET_RVVHALF;
    caps |= nk_cap_rvvbf16_k * NUMKONG_TARGET_RVVBF16;
    caps |= nk_cap_rvvbb_k * NUMKONG_TARGET_RVVBB;
    caps |= nk_cap_loongsonasx_k * NUMKONG_TARGET_LOONGSONASX;
    caps |= nk_cap_powervsx_k * NUMKONG_TARGET_POWERVSX;
    caps |= nk_cap_powermma_k * NUMKONG_TARGET_POWERMMA;
    caps |= nk_cap_v128_k * NUMKONG_TARGET_V128;
    caps |= nk_cap_v128relaxed_k * NUMKONG_TARGET_V128RELAXED;
    return caps;
}

/*  CPU capabilities, reported along two independent axes and the mask dispatch runs within:
 *
 *  - @b nk_capabilities_detected_cpu() — what this CPU can execute, from CPUID or the OS, or from
 *    what the compiler flags guarantee where the OS cannot be asked.
 *  - @b nk_capabilities_compiled_cpu() — what this binary contains, from the `NUMKONG_TARGET_*`
 *    macros the ISA probes set at build time.
 *  - @b nk_capabilities_enabled_cpu() — both axes at once, which the library settles as it loads:
 *    what every dispatch point on the CPU runs within. Always retains @b nk_cap_serial_k.
 *
 *  The two axes are independent, and conflating them is a silent performance cliff rather than a
 *  build error: a binary whose ISA probes failed still reports this machine's full @b detected
 *  mask while containing no SIMD kernels at all. Dispatch clamps every CPU mask to @b enabled(),
 *  so @b nk_cap_cpus_k runs the best kernel, and a narrower mask narrows dispatch. */

/** Writes the CPU capabilities this machine can execute. */
NUMKONG_API nk_status_t nk_capabilities_detected_cpu(nk_capability_t *capabilities);

/** Writes the CPU capabilities this binary holds kernels for. */
NUMKONG_API nk_status_t nk_capabilities_compiled_cpu(nk_capability_t *capabilities);

/** Writes the CPU capabilities dispatch runs within: detected and compiled together. */
NUMKONG_API nk_status_t nk_capabilities_enabled_cpu(nk_capability_t *capabilities);

/**
 *  @brief Prepares the calling thread for the kernels of @p capabilities, and only those.
 *  @param[in] capabilities The capabilities to prepare for, like @c nk_capabilities_enabled_cpu.
 *
 *  Most capabilities need nothing. The ones that do, and what they cost:
 *  - @c nk_cap_sapphireamx_k on Linux: one @c arch_prctl syscall for AMX tile state, which the
 *    kernel grants to the whole process, and which the library already makes as it loads.
 *    Windows and FreeBSD need no request.
 *  - Arm with FEAT_EBF16: one write of the thread's @c FPCR, making BF16 dot products fused.
 *    Finding FEAT_EBF16 costs a @c sysctl on Apple systems, and one ID-register read on Linux and
 *    FreeBSD when @p capabilities holds more than NEON.
 */
NUMKONG_API nk_status_t nk_configure_thread_cpu(nk_capability_t capabilities);

/**
 *  @brief Writes @p capabilities as a comma-separated name list such as "serial,haswell,skylake".
 *  @param[in] capabilities The mask to name, like @c nk_capabilities_detected_cpu() reports.
 *  @param[out] buffer Destination, always null-terminated; the list is truncated to fit.
 *  @param[in] capacity Size of @p buffer, like @c NUMKONG_CAPABILITIES_NAME_CAPACITY; a zero
 *      capacity writes nothing.
 *  @return Bytes written, excluding the null terminator.
 */
NUMKONG_API nk_size_t nk_capabilities_name(nk_capability_t capabilities, char *buffer, nk_size_t capacity);

/**
 *  @brief Waits for everything queued on @p stream.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k
 */
NUMKONG_API nk_status_t nk_stream_synchronize_best(nk_capability_t capabilities, nk_stream_t stream);

/** @copydoc nk_stream_synchronize_best */
NUMKONG_API nk_status_t nk_stream_synchronize_serial(nk_stream_t stream);

/** @copydoc nk_stream_synchronize_best */
NUMKONG_API nk_status_t nk_stream_synchronize_cuda(nk_stream_t stream);

/** @copydoc nk_stream_synchronize_best */
NUMKONG_API nk_status_t nk_stream_synchronize_rocm(nk_stream_t stream);

/** @copydoc nk_stream_synchronize_best */
NUMKONG_API nk_status_t nk_stream_synchronize_metal(nk_stream_t stream);

/*  GPU capabilities, per vendor, of one device named by its runtime's own ordinal: the index that
 *  @c cudaSetDevice or @c hipSetDevice takes, or the position in Metal's device list. Each vendor
 *  has four queries:
 *
 *  - @b nk_count_devices_cuda() — how many devices the runtime sees, zero without its kernels.
 *  - @b nk_capabilities_detected_cuda() — what one device runs.
 *  - @b nk_capabilities_compiled_cuda() — what this binary holds code for that one device runs.
 *  - @b nk_capabilities_enabled_cuda() — both at once: the mask for its dispatch points.
 *
 *  ROCm and Metal have the same four. Nothing is cached, as the runtimes answer from their own
 *  state: ask once per device and keep the mask. Each vendor also makes a stream on one device with
 *  @b nk_stream_init_cuda() and frees it with @b nk_stream_free_cuda(), for a caller without the
 *  vendor's runtime at hand.
 *
 *  A CUDA or ROCm unit holds code for the architectures CMake compiled it for, and a device runs
 *  a capability only where the runtime loads one of its unit's kernels there, which
 *  @b nk_capabilities_compiled_cuda() and @b nk_capabilities_compiled_rocm() ask of
 *  @c cudaFuncGetAttributes and @c hipFuncGetAttributes. */

/**
 *  @brief Counts the CUDA devices the process sees.
 *  @param[out] count The number of devices, zero without one.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k Without one.
 */
NUMKONG_API nk_status_t nk_count_devices_cuda(nk_size_t *count);

/**
 *  @brief Reports the capabilities CUDA device @p ordinal runs.
 *  @param[in] ordinal The CUDA runtime's ordinal, like @c cudaSetDevice takes.
 *  @param[out] capabilities The CUDA baseline and what the device runs, zero on failure.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k Past the last device.
 */
NUMKONG_API nk_status_t nk_capabilities_detected_cuda(nk_size_t ordinal, nk_capability_t *capabilities);

/**
 *  @brief Reports the CUDA capabilities this binary holds code for that device @p ordinal runs.
 *  @param[in] ordinal The CUDA runtime's ordinal, like @c cudaSetDevice takes.
 *  @param[out] capabilities The CUDA capabilities compiled for the device, zero on failure.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k Past the last device.
 */
NUMKONG_API nk_status_t nk_capabilities_compiled_cuda(nk_size_t ordinal, nk_capability_t *capabilities);

/**
 *  @brief Reports the capabilities CUDA device @p ordinal runs that this binary holds kernels for.
 *  @param[in] ordinal The CUDA runtime's ordinal, like @c cudaSetDevice takes.
 *  @param[out] capabilities The CUDA baseline and what the device runs, zero on failure.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k Past the last device.
 */
NUMKONG_API nk_status_t nk_capabilities_enabled_cuda(nk_size_t ordinal, nk_capability_t *capabilities);

/** Creates a stream on CUDA device @p ordinal with @c cudaStreamCreate, for any consumer to use. */
NUMKONG_API nk_status_t nk_stream_init_cuda(nk_size_t ordinal, nk_stream_t *stream);

/** Destroys a stream of @ref nk_stream_init_cuda with @c cudaStreamDestroy, once its work ends. */
NUMKONG_API nk_status_t nk_stream_free_cuda(nk_stream_t stream);

/**
 *  @brief Counts the ROCm devices the process sees.
 *  @param[out] count The number of devices, zero without one.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k Without one.
 */
NUMKONG_API nk_status_t nk_count_devices_rocm(nk_size_t *count);

/**
 *  @brief Reports the capabilities ROCm device @p ordinal runs.
 *  @param[in] ordinal The HIP runtime's ordinal, like @c hipSetDevice takes.
 *  @param[out] capabilities The ROCm baseline and what the device runs, zero on failure.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k Past the last device.
 */
NUMKONG_API nk_status_t nk_capabilities_detected_rocm(nk_size_t ordinal, nk_capability_t *capabilities);

/**
 *  @brief Reports the ROCm capabilities this binary holds code for that device @p ordinal runs.
 *  @param[in] ordinal The HIP runtime's ordinal, like @c hipSetDevice takes.
 *  @param[out] capabilities The ROCm capabilities compiled for the device, zero on failure.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k Past the last device.
 */
NUMKONG_API nk_status_t nk_capabilities_compiled_rocm(nk_size_t ordinal, nk_capability_t *capabilities);

/**
 *  @brief Reports the capabilities ROCm device @p ordinal runs that this binary holds kernels for.
 *  @param[in] ordinal The HIP runtime's ordinal, like @c hipSetDevice takes.
 *  @param[out] capabilities The ROCm baseline and what the device runs, zero on failure.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k Past the last device.
 */
NUMKONG_API nk_status_t nk_capabilities_enabled_rocm(nk_size_t ordinal, nk_capability_t *capabilities);

/** Creates a stream on ROCm device @p ordinal with @c hipStreamCreate, for any consumer to use. */
NUMKONG_API nk_status_t nk_stream_init_rocm(nk_size_t ordinal, nk_stream_t *stream);

/** Destroys a stream of @ref nk_stream_init_rocm with @c hipStreamDestroy, once its work ends. */
NUMKONG_API nk_status_t nk_stream_free_rocm(nk_stream_t stream);

/**
 *  @brief Counts the Metal devices the system lists.
 *  @param[out] count The number of devices, zero without one.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k Without one.
 */
NUMKONG_API nk_status_t nk_count_devices_metal(nk_size_t *count);

/**
 *  @brief Reports the capabilities Metal device @p ordinal runs.
 *  @param[in] ordinal The device's position in the order the system lists them.
 *  @param[out] capabilities The Metal baseline when the device supports it, zero otherwise.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k Past the last device.
 */
NUMKONG_API nk_status_t nk_capabilities_detected_metal(nk_size_t ordinal, nk_capability_t *capabilities);

/**
 *  @brief Reports the Metal capabilities this binary holds code for that device @p ordinal runs.
 *  @param[in] ordinal The device's position in the order the system lists them.
 *  @param[out] capabilities The Metal capabilities compiled for the device, zero on failure.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k Past the last device.
 */
NUMKONG_API nk_status_t nk_capabilities_compiled_metal(nk_size_t ordinal, nk_capability_t *capabilities);

/**
 *  @brief Reports the capabilities Metal device @p ordinal runs that this binary holds kernels for.
 *  @param[in] ordinal The device's position in the order the system lists them.
 *  @param[out] capabilities The Metal baseline when the device supports it, zero otherwise.
 *
 *  @par Statuses:
 *  - @c nk_missing_gpu_k Past the last device.
 */
NUMKONG_API nk_status_t nk_capabilities_enabled_metal(nk_size_t ordinal, nk_capability_t *capabilities);

/** Opens an @c id<MTLCommandQueue> stream on Metal device @p ordinal with @c newCommandQueue. */
NUMKONG_API nk_status_t nk_stream_init_metal(nk_size_t ordinal, nk_stream_t *stream);

/** Waits for a queue of @ref nk_stream_init_metal, then releases it. */
NUMKONG_API nk_status_t nk_stream_free_metal(nk_stream_t stream);

/**
 *  @brief Finds the kernel of @p kind for @p dtype that the best of @p capabilities runs.
 *  @param[in] kind The kernel kind, like @c nk_kernel_dot_k.
 *  @param[in] dtype The input type, like @c nk_f32_k.
 *  @param[in] capabilities One device's mask, like @c nk_capabilities_enabled_cpu reports.
 *  @param[out] kernel The kernel, to cast back to its kind's signature, or null.
 *  @param[out] capability The single capability the kernel belongs to, or zero.
 *
 *  @par Statuses:
 *  - @c nk_missing_kernel_k No capability in the mask has it.
 */
NUMKONG_API nk_status_t nk_find_kernel_punned(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability);

#ifdef __cplusplus
} // extern "C"
#endif

/*  The library detects once per process; header-only builds ask on every call, and make the same
 *  process-wide AMX request the library makes as it loads. GPU runtime services live only in the
 *  library, so a header-only unit answers `nk_missing_library_k` for each vendor, even in nvcc. */
#if NUMKONG_HEADER_ONLY
#include "numkong/cpu.h"
#include "numkong/cuda.cuh"
#include "numkong/rocm.cuh"
#include "numkong/metal.h"

NUMKONG_API nk_status_t nk_capabilities_detected_cpu(nk_capability_t *capabilities) {
    *capabilities = nk_capabilities_detected_cpu_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_capabilities_compiled_cpu(nk_capability_t *capabilities) {
    *capabilities = nk_capabilities_compiled_cpu_mask();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_capabilities_enabled_cpu(nk_capability_t *capabilities) {
    *capabilities = nk_capabilities_enable_amx_(nk_capabilities_detected_cpu_() & nk_capabilities_compiled_cpu_mask());
    return nk_success_k;
}

/** @copydoc nk_configure_thread_cpu */
NUMKONG_API nk_status_t nk_configure_thread_cpu(nk_capability_t capabilities) {
    return nk_configure_thread_cpu_(capabilities);
}

/** @copydoc nk_capabilities_name */
NUMKONG_API nk_size_t nk_capabilities_name(nk_capability_t capabilities, char *buffer, nk_size_t capacity) {
    return nk_capabilities_name_(capabilities, buffer, capacity);
}

NUMKONG_API nk_status_t nk_count_devices_cuda(nk_size_t *count) {
    *count = 0;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_capabilities_detected_cuda(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_capabilities_compiled_cuda(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_capabilities_enabled_cuda(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_stream_init_cuda(nk_size_t ordinal, nk_stream_t *stream) {
    nk_unused_(ordinal);
    *stream = NUMKONG_NULL;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_stream_free_cuda(nk_stream_t stream) {
    nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_stream_synchronize_cuda(nk_stream_t stream) {
    nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_count_devices_rocm(nk_size_t *count) {
    *count = 0;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_capabilities_detected_rocm(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_capabilities_compiled_rocm(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_capabilities_enabled_rocm(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_stream_init_rocm(nk_size_t ordinal, nk_stream_t *stream) {
    nk_unused_(ordinal);
    *stream = NUMKONG_NULL;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_stream_free_rocm(nk_stream_t stream) {
    nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_stream_synchronize_rocm(nk_stream_t stream) {
    nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_count_devices_metal(nk_size_t *count) {
    *count = 0;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_capabilities_detected_metal(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_capabilities_compiled_metal(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_capabilities_enabled_metal(nk_size_t ordinal, nk_capability_t *capabilities) {
    nk_unused_(ordinal);
    *capabilities = 0;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_stream_init_metal(nk_size_t ordinal, nk_stream_t *stream) {
    nk_unused_(ordinal);
    *stream = NUMKONG_NULL;
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_stream_free_metal(nk_stream_t stream) {
    nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_stream_synchronize_metal(nk_stream_t stream) {
    nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_stream_synchronize_best(nk_capability_t capabilities, nk_stream_t stream) {
    nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_find_kernel_punned(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#endif // NUMKONG_HEADER_ONLY

#endif // NUMKONG_CAPABILITIES_H
