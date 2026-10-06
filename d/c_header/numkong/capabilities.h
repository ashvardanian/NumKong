/**
 *  @brief SIMD capability detection and thread configuration.
 *  @file include/numkong/capabilities.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @section x86_targets Choosing x86 Target Generations
 *
 *  It's important to provide fine-grained controls over AVX512 families, as they are very fragmented:
 *
 *  - Intel Skylake servers: F, CD, VL, DQ, BW
 *  - Intel Cascade Lake workstations: F, CD, VL, DQ, BW, VNNI
 *       > In other words, it extends Skylake with VNNI support
 *  - Intel Sunny Cove (Ice Lake) servers:
 *         F, CD, VL, DQ, BW, VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ
 *  - AMD Zen4 (Genoa):
 *         F, CD, VL, DQ, BW, VNNI, VPOPCNTDQ, IFMA, VBMI, VAES, GFNI, VBMI2, BITALG, VPCLMULQDQ, BF16
 *       > In other words, it extends Sunny Cove with BF16 support
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
 *  Beyond those, we support AVX2 for old Haswell generation CPUs, AVX2+VNNI for Alder Lake (12th/13th gen),
 *  and AVX2+VNNI-INT8 for Sierra Forest (adds native signed×signed and unsigned×unsigned 8-bit dot products).
 *
 *  To list all available macros for x86, take a recent compiler, like GCC 12 and run:
 *       gcc-12 -march=sapphirerapids -dM -E - < /dev/null | egrep "SSE|AVX" | sort
 *  On Arm machines you may want to check for other flags:
 *       gcc-12 -march=native -dM -E - < /dev/null | egrep "NEON|SVE|FP16|FMA" | sort
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
 *  - Neoverse N1: armv8.2-a, extended with Armv8.4 "dotprod" instructions.
 *    Used in AWS @b Graviton2 and Ampere @b Altra.
 *    https://developer.arm.com/Processors/Neoverse%20N1
 *  - Neoverse V1: armv8.4-a, extended with Armv8.6 bfloat/int8 "matmul" instructions.
 *    Used in AWS @b Graviton3, which also enables `sve`, `svebf16`, and `svei8mm`.
 *    https://developer.arm.com/Processors/Neoverse%20V1
 *  - Neoverse V2: armv9.0 with SVE2 and SVE bit-permutes
 *    Used in AWS @b Graviton4, NVIDIA @b Grace, Google @b Axion.
 *    https://developer.arm.com/Processors/Neoverse%20V2
 *    The N2 core is very similar to V2 and is used by Microsoft @b Cobalt.
 *    https://developer.arm.com/Processors/Neoverse%20N2
 *
 *  On the consumer side, Apple is the biggest player with mobile @b A chips and desktop @b M chips.
 *  The M1 implements Armv8.5-A, both M2 and M3 implement Armv8.6-A, and M4 is expected to have Armv9.1-A.
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics
 *  - Detecting target CPU features at compile time: https://stackoverflow.com/a/28939692/2766161
 */

#ifndef NK_CAPABILITIES_H
#define NK_CAPABILITIES_H

#include "numkong/types.h" // `nk_u64_t`, `NK_DEFINED_LINUX_`

#define NK_VERSION_MAJOR 7
#define NK_VERSION_MINOR 1
#define NK_VERSION_PATCH 0

/**
 *  @brief  Removes compile-time dispatching, and replaces it with runtime dispatching.
 *          So the `nk_dot_f32` function will invoke the most advanced backend supported by the CPU,
 *          that runs the program, rather than the most advanced backend supported by the CPU
 *          used to compile the library or the downstream application.
 */
#if !defined(NK_DYNAMIC_DISPATCH)
#define NK_DYNAMIC_DISPATCH (0) // true or false
#endif

// On Apple Silicon, `mrs` is not allowed in user-space, so we need to use the `sysctl` API.
#if defined(NK_DEFINED_APPLE_)
#include <fenv.h>       // `fesetenv` - part of C 99 standard
#include <sys/sysctl.h> // `sysctlbyname`
#endif

// Detect POSIX extensions availability for signal handling.
// POSIX extensions provide `sigaction`, `sigjmp_buf`, and `sigsetjmp` for safe signal handling.
// These are needed on Linux ARM for safely testing `mrs` instruction availability.
#if defined(NK_DEFINED_LINUX_) || defined(NK_DEFINED_FREEBSD_)
#include <unistd.h> // `_POSIX_VERSION`
#endif
#if (defined(NK_DEFINED_LINUX_) || defined(NK_DEFINED_FREEBSD_)) && defined(_POSIX_VERSION)
#include <setjmp.h> // `sigjmp_buf`, `sigsetjmp`, `siglongjmp`
#include <signal.h> // `sigaction`, `SIGILL`
#define NK_HAS_POSIX_EXTENSIONS_ 1
#else
#define NK_HAS_POSIX_EXTENSIONS_ 0
#endif

// On Linux x86/RISC-V, we need `syscall()` for AMX permission and hwprobe.
// With `-std=c11` glibc hides `syscall()` behind `_GNU_SOURCE`, but if any
// system header was included before us, `<features.h>` is already locked.
// Forward-declare `syscall` directly — it always exists in glibc.
#if defined(NK_DEFINED_LINUX_) && (NK_TARGET_X86_ || NK_TARGET_RISCV_)
#include <sys/syscall.h> // `SYS_arch_prctl`, `SYS_riscv_hwprobe`
#ifdef __cplusplus
extern "C" long syscall(long, ...) noexcept;
#else
extern long syscall(long, ...);
#endif
#if NK_TARGET_RISCV_
#include <sys/auxv.h> // `getauxval`, `AT_HWCAP`
#endif
#endif

// On FreeBSD RISC-V, we use elf_aux_info for capability detection
#if defined(NK_DEFINED_FREEBSD_) && NK_TARGET_RISCV_
#include <sys/auxv.h> // `elf_aux_info`, `AT_HWCAP`
#endif

// On Windows ARM, we use IsProcessorFeaturePresent API for capability detection
#if defined(NK_DEFINED_WINDOWS_) && NK_TARGET_ARM_
#include <processthreadsapi.h> // `IsProcessorFeaturePresent`
#endif

// On WASM with Emscripten, we use EM_JS for runtime capability detection
#if NK_TARGET_WASM_ && defined(__EMSCRIPTEN__)
#include <emscripten.h> // `EM_JS`
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  Enumeration of supported metric kinds.
 *          Some have aliases for convenience/discoverability.
 */
typedef enum {
    nk_kernel_unknown_k = 0, ///< Unknown kernel kind

    // Classics:
    nk_kernel_dot_k = 'i',         ///< Inner product
    nk_kernel_vdot_k = 'v',        ///< Complex inner product
    nk_kernel_angular_k = 'a',     ///< Angular (cosine) distance
    nk_kernel_euclidean_k = 'e',   ///< Euclidean distance
    nk_kernel_sqeuclidean_k = '2', ///< Squared Euclidean distance

    // Binary:
    nk_kernel_hamming_k = 'h', ///< Hamming (or Manhattan) distance
    nk_kernel_jaccard_k = 'j', ///< Jaccard (or Tanimoto) coefficient

    // Curved Spaces:
    nk_kernel_bilinear_k = 'b',    ///< Bilinear form
    nk_kernel_mahalanobis_k = 'm', ///< Mahalanobis distance

    // Geospatial:
    nk_kernel_haversine_k = 'o', ///< Haversine distance
    nk_kernel_vincenty_k = 'O',  ///< Vincenty distance (ellipsoidal geodesic)

    // Probability:
    nk_kernel_kld_k = 'k', ///< Kullback-Leibler divergence
    nk_kernel_jsd_k = 's', ///< Jensen-Shannon divergence

    // Mesh superposition:
    nk_kernel_rmsd_k = 'r',    ///< RMSD without optimal superposition
    nk_kernel_kabsch_k = 'K',  ///< Kabsch RMSD with optimal rotation
    nk_kernel_umeyama_k = 'U', ///< Umeyama RMSD with optimal rotation and scale

    // Sparse Sets:
    nk_kernel_sparse_dot_k = 'd',       ///< Sparse dot product with weighted indices
    nk_kernel_sparse_intersect_k = 'x', ///< Equivalent to unnormalized Jaccard

    // BLAS-like operations:
    nk_kernel_each_scale_k = '*', ///< Element-wise Scale
    nk_kernel_each_sum_k = '+',   ///< Element-wise Sum
    nk_kernel_each_blend_k = 'w', ///< Element-wise Weighted Sum
    nk_kernel_each_fma_k = 'f',   ///< Element-wise Fused Multiply-Add

    // Trigonometric functions:
    nk_kernel_each_sin_k = 'S',  ///< Element-wise sine
    nk_kernel_each_cos_k = 'C',  ///< Element-wise cosine
    nk_kernel_each_atan_k = 'A', ///< Element-wise arctangent

    // Horizontal reductions:
    nk_kernel_reduce_moments_k = 'R', ///< Horizontal moments reduction (sum + sum-of-squares)
    nk_kernel_reduce_minmax_k = 'X',  ///< Horizontal minmax reduction (min + argmin + max + argmax)

    // GEMM-like batched dot products:
    nk_kernel_dots_packed_size_k = 'P', ///< GEMM packed buffer size
    nk_kernel_dots_pack_k = 'Q',        ///< GEMM B matrix packing
    nk_kernel_dots_packed_k = 'G',      ///< GEMM computation
    nk_kernel_dots_symmetric_k = 'y',   ///< Symmetric Gram matrix (A x At)

    // GEMM-like batched set similarity functions:
    nk_kernel_hammings_packed_k = 'M',    ///< Hamming distance computation
    nk_kernel_hammings_symmetric_k = 'Y', ///< Symmetric Hamming distance matrix (A x At)
    nk_kernel_jaccards_packed_k = 'p',    ///< Jaccard distance computation
    nk_kernel_jaccards_symmetric_k = 'Z', ///< Symmetric Jaccard distance matrix

    // GEMM-like batched spatial distances functions:
    nk_kernel_angulars_packed_k = 'N',      ///< Batched angular distances (packed B)
    nk_kernel_angulars_symmetric_k = 'n',   ///< Symmetric angular distance matrix
    nk_kernel_euclideans_packed_k = 'E',    ///< Batched euclidean distances (packed B)
    nk_kernel_euclideans_symmetric_k = 'D', ///< Symmetric euclidean distance matrix

    // MaxSim late-interaction functions:
    nk_kernel_maxsim_packed_size_k = 'L', ///< MaxSim packed buffer size
    nk_kernel_maxsim_pack_k = 'l',        ///< MaxSim vector packing
    nk_kernel_maxsim_packed_k = 'T',      ///< MaxSim late-interaction computation

    nk_kernel_cast_k = '-', ///< Type casting from one type to another

} nk_kernel_kind_t;

/**
 *  @brief  64-bit bitmask representing SIMD capabilities of the target architecture.
 */
typedef nk_u64_t nk_capability_t;

/** @brief  Serial (non-SIMD) fallback capability. Always available. */
#define nk_cap_serial_k ((nk_capability_t)1)

/** @brief  Mask representing any capability. */
#define nk_cap_any_k ((nk_capability_t)NK_U64_MAX)

#define nk_cap_neon_k        ((nk_capability_t)1 << 1)
#define nk_cap_haswell_k     ((nk_capability_t)1 << 2)
#define nk_cap_skylake_k     ((nk_capability_t)1 << 3)
#define nk_cap_neonhalf_k    ((nk_capability_t)1 << 4)
#define nk_cap_neonsdot_k    ((nk_capability_t)1 << 5)
#define nk_cap_neonfhm_k     ((nk_capability_t)1 << 6)
#define nk_cap_icelake_k     ((nk_capability_t)1 << 7)
#define nk_cap_genoa_k       ((nk_capability_t)1 << 8)
#define nk_cap_neonbfdot_k   ((nk_capability_t)1 << 9)
#define nk_cap_sve_k         ((nk_capability_t)1 << 10)
#define nk_cap_svehalf_k     ((nk_capability_t)1 << 11)
#define nk_cap_svesdot_k     ((nk_capability_t)1 << 12)
#define nk_cap_alder_k       ((nk_capability_t)1 << 13)
#define nk_cap_svebfdot_k    ((nk_capability_t)1 << 14)
#define nk_cap_sve2_k        ((nk_capability_t)1 << 15)
#define nk_cap_v128relaxed_k ((nk_capability_t)1 << 16)
#define nk_cap_sapphire_k    ((nk_capability_t)1 << 17)
#define nk_cap_sapphireamx_k ((nk_capability_t)1 << 18)
#define nk_cap_rvv_k         ((nk_capability_t)1 << 19)
#define nk_cap_rvvhalf_k     ((nk_capability_t)1 << 20)
#define nk_cap_rvvbf16_k     ((nk_capability_t)1 << 21)
#define nk_cap_graniteamx_k  ((nk_capability_t)1 << 22)
#define nk_cap_turin_k       ((nk_capability_t)1 << 23)
#define nk_cap_sme_k         ((nk_capability_t)1 << 24)
#define nk_cap_sme2_k        ((nk_capability_t)1 << 25)
#define nk_cap_smef64_k      ((nk_capability_t)1 << 26)
#define nk_cap_smefa64_k     ((nk_capability_t)1 << 27)
#define nk_cap_sve2p1_k      ((nk_capability_t)1 << 28)
#define nk_cap_sme2p1_k      ((nk_capability_t)1 << 29)
#define nk_cap_smehalf_k     ((nk_capability_t)1 << 30)
#define nk_cap_smebf16_k     ((nk_capability_t)1 << 31)
#define nk_cap_smelut2_k     ((nk_capability_t)1 << 32)
#define nk_cap_rvvbb_k       ((nk_capability_t)1 << 33)
#define nk_cap_sierra_k      ((nk_capability_t)1 << 34)

typedef void (*nk_metric_dense_punned_t)(void const *a, void const *b, nk_size_t n, void *d);

typedef void (*nk_sparse_intersect_punned_t)(void const *a, void const *b, nk_size_t a_length, nk_size_t b_length,
                                             void *result, nk_size_t *count);

typedef void (*nk_sparse_dot_punned_t)(void const *a, void const *b, void const *a_weights, void const *b_weights,
                                       nk_size_t a_length, nk_size_t b_length, void *product);

typedef void (*nk_metric_curved_punned_t)(void const *a, void const *b, void const *c, nk_size_t n, void *d);

typedef void (*nk_metric_geospatial_punned_t)(void const *a_lats, void const *a_lons, void const *b_lats,
                                              void const *b_lons, nk_size_t n, void *results);

typedef void (*nk_each_scale_punned_t)(void const *a, nk_size_t n, void const *alpha, void const *beta, void *y);

typedef void (*nk_each_sum_punned_t)(void const *a, void const *b, nk_size_t n, void *y);

typedef void (*nk_each_blend_punned_t)(void const *a, void const *b, nk_size_t n, void const *alpha, void const *beta,
                                       void *y);

typedef void (*nk_each_fma_punned_t)(void const *a, void const *b, void const *c, nk_size_t n, void const *alpha,
                                     void const *beta, void *y);

typedef void (*nk_kernel_trigonometry_punned_t)(void const *x, nk_size_t n, void *y);

typedef void (*nk_metric_mesh_punned_t)(void const *a, void const *b, nk_size_t n, void *a_centroid, void *b_centroid,
                                        void *rotation, void *scale, void *d);

typedef void (*nk_kernel_reduce_moments_punned_t)(void const *data, nk_size_t count, nk_size_t stride_bytes,
                                                  void *sum_ptr, void *sumsq_ptr);

typedef void (*nk_kernel_reduce_minmax_punned_t)(void const *data, nk_size_t count, nk_size_t stride_bytes,
                                                 void *min_value, nk_size_t *min_index, void *max_value,
                                                 nk_size_t *max_index);

typedef nk_size_t (*nk_dots_packed_size_punned_t)(nk_size_t width, nk_size_t depth);

typedef void (*nk_dots_pack_punned_t)(void const *b, nk_size_t width, nk_size_t depth, nk_size_t b_stride,
                                      void *b_packed);

typedef void (*nk_dots_packed_punned_t)(void const *a, void const *b_packed, void *c, nk_size_t height, nk_size_t width,
                                        nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);

typedef void (*nk_dots_symmetric_punned_t)(void const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride,
                                           void *result, nk_size_t result_stride, nk_size_t row_start,
                                           nk_size_t row_count);

typedef void (*nk_hammings_packed_punned_t)(void const *a, void const *b_packed, void *c, nk_size_t height,
                                            nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);

typedef void (*nk_hammings_symmetric_punned_t)(void const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                               nk_size_t stride, void *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);

typedef void (*nk_jaccards_packed_punned_t)(void const *a, void const *b_packed, void *c, nk_size_t height,
                                            nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);

typedef void (*nk_jaccards_symmetric_punned_t)(void const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                               nk_size_t stride, void *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);

typedef void (*nk_angulars_packed_punned_t)(void const *a, void const *b_packed, void *c, nk_size_t height,
                                            nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
typedef void (*nk_angulars_symmetric_punned_t)(void const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                               nk_size_t stride, void *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);
typedef void (*nk_euclideans_packed_punned_t)(void const *a, void const *b_packed, void *c, nk_size_t height,
                                              nk_size_t width, nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride);
typedef void (*nk_euclideans_symmetric_punned_t)(void const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                                 nk_size_t stride, void *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);

typedef void (*nk_maxsim_packed_punned_t)(void const *q_packed, void const *d_packed, nk_size_t query_count,
                                          nk_size_t document_count, nk_size_t depth, void *result);

typedef void (*nk_kernel_cast_punned_t)(void const *from, nk_dtype_t from_type, nk_size_t n, void *to,
                                        nk_dtype_t to_type);

typedef void (*nk_kernel_punned_t)(void *);

#if NK_TARGET_X86_

NK_PUBLIC int nk_configure_thread_x86_(nk_capability_t capabilities) {
#if defined(_MSC_VER)
    unsigned int mxcsr = _mm_getcsr();
    mxcsr |= 1 << 15;
    mxcsr |= 1 << 6;
    _mm_setcsr(mxcsr);
#else
    unsigned int mxcsr;
    __asm__ __volatile__("stmxcsr %0" : "=m"(mxcsr));
    mxcsr |= 1 << 15;
    mxcsr |= 1 << 6;
    __asm__ __volatile__("ldmxcsr %0" : : "m"(mxcsr));
#endif

#if NK_TARGET_SAPPHIREAMX
    if (capabilities & nk_cap_sapphireamx_k) {
#if defined(NK_DEFINED_LINUX_)
        // Linux requires explicit permission for AMX tile state via arch_prctl syscall
        int const ARCH_REQ_XCOMP_PERM = 0x1023;
        unsigned long const XFEATURE_XTILEDATA = 18;
        syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
#endif
        // On Windows, AMX tile state is automatically enabled by the OS if hardware supports it.
        // On FreeBSD, no explicit request is needed either.
        nk_unused_(capabilities);
    }
#else
    nk_unused_(capabilities);
#endif
    return 1;
}

NK_PUBLIC nk_capability_t nk_capabilities_x86_(void) {
    union four_registers_t {
        int array[4];
        struct separate_t {
            unsigned eax, ebx, ecx, edx;
        } named;
    } info1, info7, info7sub1;

#if defined(_MSC_VER)
    __cpuidex(info1.array, 1, 0);
    __cpuidex(info7.array, 7, 0);
    __cpuidex(info7sub1.array, 7, 1);
#else
    __asm__ __volatile__("cpuid"
                         : "=a"(info1.named.eax), "=b"(info1.named.ebx), "=c"(info1.named.ecx), "=d"(info1.named.edx)
                         : "a"(1), "c"(0));
    __asm__ __volatile__("cpuid"
                         : "=a"(info7.named.eax), "=b"(info7.named.ebx), "=c"(info7.named.ecx), "=d"(info7.named.edx)
                         : "a"(7), "c"(0));
    __asm__ __volatile__("cpuid"
                         : "=a"(info7sub1.named.eax), "=b"(info7sub1.named.ebx), "=c"(info7sub1.named.ecx),
                           "=d"(info7sub1.named.edx)
                         : "a"(7), "c"(1));
#endif

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

    unsigned supports_haswell = supports_avx2 && supports_f16c && supports_fma;
    unsigned supports_skylake = supports_avx512f;
    unsigned supports_icelake = supports_avx512vnni && supports_avx512ifma && supports_avx512bitalg &&
                                supports_avx512vbmi && supports_avx512vbmi2 && supports_avx512vpopcntdq;
    unsigned supports_genoa = supports_avx512bf16;
    unsigned supports_sapphire = supports_avx512fp16;
    unsigned supports_turin = supports_avx512vp2intersect && supports_avx512bf16;
    unsigned supports_sierra = supports_haswell && supports_avxvnniint8;
    unsigned supports_alder = supports_haswell && supports_avxvnni;
    unsigned supports_sapphireamx = supports_amx_tile && supports_amx_bf16 && supports_amx_int8;
    unsigned supports_graniteamx = supports_sapphireamx && supports_amx_fp16;

    return (nk_capability_t)((nk_cap_haswell_k * supports_haswell) | (nk_cap_skylake_k * supports_skylake) |
                             (nk_cap_icelake_k * supports_icelake) | (nk_cap_genoa_k * supports_genoa) |
                             (nk_cap_sapphire_k * supports_sapphire) | (nk_cap_turin_k * supports_turin) |
                             (nk_cap_sierra_k * supports_sierra) | (nk_cap_alder_k * supports_alder) |
                             (nk_cap_sapphireamx_k * supports_sapphireamx) |
                             (nk_cap_graniteamx_k * supports_graniteamx) | (nk_cap_serial_k));
}

#endif // NK_TARGET_X86_

#if NK_TARGET_ARM_

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.5-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.5-a+sve")
#endif

#if NK_HAS_POSIX_EXTENSIONS_
static sigjmp_buf nk_mrs_test_jump_buffer_;
static void nk_mrs_test_sigill_handler_(int sig) {
    nk_unused_(sig);
    siglongjmp(nk_mrs_test_jump_buffer_, 1);
}
#endif

NK_PUBLIC int nk_configure_thread_arm_(nk_capability_t capabilities) {
    nk_unused_(capabilities);
#if defined(NK_DEFINED_APPLE_)
    int is_success = fesetenv(FE_DFL_DISABLE_DENORMS_ENV) == 0;
    return is_success;
#elif defined(NK_DEFINED_LINUX_) || defined(NK_DEFINED_FREEBSD_)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1 << 19);
    fpcr |= (1 << 24);
    fpcr |= (1 << 25);
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    return 1;
#else
    return 0;
#endif
}

NK_PUBLIC nk_capability_t nk_capabilities_arm_(void) {
#if defined(NK_DEFINED_APPLE_)
    size_t size = sizeof(unsigned);
    unsigned supports_neon = 0, supports_fp16 = 0, supports_fhm = 0, supports_bf16 = 0, supports_i8mm = 0;
    unsigned supports_sme = 0, supports_sme2 = 0, supports_smef64 = 0, supports_smehalf = 0, supports_sme2p1 = 0;
    if (sysctlbyname("hw.optional.neon", &supports_neon, &size, NULL, 0) != 0) supports_neon = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_FP16", &supports_fp16, &size, NULL, 0) != 0) supports_fp16 = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_FHM", &supports_fhm, &size, NULL, 0) != 0) supports_fhm = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_BF16", &supports_bf16, &size, NULL, 0) != 0) supports_bf16 = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &supports_i8mm, &size, NULL, 0) != 0) supports_i8mm = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_SME", &supports_sme, &size, NULL, 0) != 0) supports_sme = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &supports_sme2, &size, NULL, 0) != 0) supports_sme2 = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_SME_F64F64", &supports_smef64, &size, NULL, 0) != 0) supports_smef64 = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_SME_F16F16", &supports_smehalf, &size, NULL, 0) != 0) supports_smehalf = 0;
    if (sysctlbyname("hw.optional.arm.FEAT_SME2p1", &supports_sme2p1, &size, NULL, 0) != 0) supports_sme2p1 = 0;

    return (nk_capability_t)((nk_cap_neon_k * (supports_neon)) |
                             (nk_cap_neonhalf_k * (supports_neon && supports_fp16)) |
                             (nk_cap_neonfhm_k * (supports_neon && supports_fhm)) |
                             (nk_cap_neonbfdot_k * (supports_neon && supports_bf16)) |
                             (nk_cap_neonsdot_k * (supports_neon && supports_i8mm)) | (nk_cap_sme_k * (supports_sme)) |
                             (nk_cap_sme2_k * (supports_sme2)) | (nk_cap_sme2p1_k * (supports_sme2p1)) |
                             (nk_cap_smef64_k * (supports_smef64)) | (nk_cap_smehalf_k * (supports_smehalf)) |
                             (nk_cap_smebf16_k * (supports_sme)) | (nk_cap_serial_k));

#elif defined(NK_DEFINED_LINUX_) || defined(NK_DEFINED_FREEBSD_)

#if NK_HAS_POSIX_EXTENSIONS_
    struct sigaction action_new, action_old;
    action_new.sa_handler = nk_mrs_test_sigill_handler_;
    sigemptyset(&action_new.sa_mask);
    action_new.sa_flags = 0;

    int mrs_works = 0;
    if (sigaction(SIGILL, &action_new, &action_old) == 0) {
        if (sigsetjmp(nk_mrs_test_jump_buffer_, 1) == 0) {
            unsigned long midr_value;
            __asm__ __volatile__("mrs %0, MIDR_EL1" : "=r"(midr_value));
            mrs_works = 1;
        }
        sigaction(SIGILL, &action_old, NULL);
    }

    if (!mrs_works) return (nk_capability_t)(nk_cap_neon_k | nk_cap_serial_k);
#else
    return (nk_capability_t)(nk_cap_neon_k | nk_cap_serial_k);
#endif

    unsigned long id_aa64isar0_el1 = 0, id_aa64isar1_el1 = 0, id_aa64pfr0_el1 = 0, id_aa64zfr0_el1 = 0;

    __asm__ __volatile__("mrs %0, ID_AA64ISAR0_EL1" : "=r"(id_aa64isar0_el1));
    unsigned supports_integer_dot_products = ((id_aa64isar0_el1 >> 44) & 0xF) >= 1;
    unsigned supports_fhm = ((id_aa64isar0_el1 >> 48) & 0xF) >= 1;
    __asm__ __volatile__("mrs %0, ID_AA64ISAR1_EL1" : "=r"(id_aa64isar1_el1));
    unsigned supports_i8mm = ((id_aa64isar1_el1 >> 52) & 0xF) >= 1;
    unsigned supports_bf16 = ((id_aa64isar1_el1 >> 44) & 0xF) >= 1;

    __asm__ __volatile__("mrs %0, ID_AA64PFR0_EL1" : "=r"(id_aa64pfr0_el1));
    unsigned supports_sve = ((id_aa64pfr0_el1 >> 32) & 0xF) >= 1;
    unsigned supports_fp16 = ((id_aa64pfr0_el1 >> 20) & 0xF) == 0x1;
    unsigned supports_neon = ((id_aa64pfr0_el1 >> 20) & 0xF) != 0xF;

    if (supports_sve) __asm__ __volatile__("mrs %0, ID_AA64ZFR0_EL1" : "=r"(id_aa64zfr0_el1));
    unsigned supports_svesdotmm = ((id_aa64zfr0_el1 >> 44) & 0xF) >= 1;
    unsigned supports_svebfdot = ((id_aa64zfr0_el1 >> 20) & 0xF) >= 1;
    unsigned supports_sve2 = ((id_aa64zfr0_el1) & 0xF) >= 1;
    unsigned supports_sve2p1 = ((id_aa64zfr0_el1) & 0xF) >= 2;

    unsigned long id_aa64pfr1_el1 = 0, id_aa64smfr0_el1 = 0;
    __asm__ __volatile__("mrs %0, ID_AA64PFR1_EL1" : "=r"(id_aa64pfr1_el1));
    unsigned supports_sme = ((id_aa64pfr1_el1 >> 24) & 0xF) >= 1;

    unsigned supports_sme2 = 0, supports_sme2p1 = 0;
    unsigned supports_smef64 = 0, supports_smehalf = 0, supports_smebf16 = 0;
    unsigned supports_smelut2 = 0, supports_smefa64 = 0;
    if (supports_sme) {
        // MRS x0, ID_AA64SMFR0_EL1 (S3_0_C0_C4_5) — encoded as raw .inst because some
        // assemblers (Clang 21 in Android NDK r29) reject the symbolic register name.
        // Encoding: 0xD53804A0 = MRS x0, op0=3, op1=0, CRn=0, CRm=4, op2=5, Rt=0.
        register unsigned long __smfr0 __asm__("x0");
        __asm__ __volatile__(".inst 0xD53804A0" : "=r"(__smfr0));
        id_aa64smfr0_el1 = __smfr0;
        unsigned sme_version = (id_aa64smfr0_el1 >> 56) & 0xF;
        supports_sme2 = sme_version >= 1;
        supports_sme2p1 = sme_version >= 2;
        supports_smef64 = (id_aa64smfr0_el1 >> 48) & 0x1;
        supports_smehalf = (id_aa64smfr0_el1 >> 42) & 0x1;
        supports_smebf16 = (id_aa64smfr0_el1 >> 44) & 0x1;
        supports_smefa64 = (id_aa64smfr0_el1 >> 63) & 0x1;
    }

    return (nk_capability_t)((nk_cap_neon_k * (supports_neon)) |
                             (nk_cap_neonhalf_k * (supports_neon && supports_fp16)) |
                             (nk_cap_neonfhm_k * (supports_neon && supports_fhm)) |
                             (nk_cap_neonbfdot_k * (supports_neon && supports_bf16)) |
                             (nk_cap_neonsdot_k * (supports_neon && supports_i8mm && supports_integer_dot_products)) |
                             (nk_cap_sve_k * (supports_sve)) | (nk_cap_svehalf_k * (supports_sve && supports_fp16)) |
                             (nk_cap_svebfdot_k * (supports_sve && supports_svebfdot)) |
                             (nk_cap_svesdot_k * (supports_sve && supports_svesdotmm)) |
                             (nk_cap_sve2_k * (supports_sve2)) | (nk_cap_sve2p1_k * (supports_sve2p1)) |
                             (nk_cap_sme_k * (supports_sme)) | (nk_cap_sme2_k * (supports_sme2)) |
                             (nk_cap_sme2p1_k * (supports_sme2p1)) | (nk_cap_smef64_k * (supports_smef64)) |
                             (nk_cap_smehalf_k * (supports_smehalf)) | (nk_cap_smebf16_k * (supports_smebf16)) |
                             (nk_cap_smefa64_k * (supports_smefa64)) | (nk_cap_serial_k));
#elif defined(NK_DEFINED_WINDOWS_)

    unsigned supports_neon = 0, supports_dp = 0;

#if defined(PF_ARM_V8_INSTRUCTIONS_AVAILABLE)
    supports_neon = IsProcessorFeaturePresent(PF_ARM_V8_INSTRUCTIONS_AVAILABLE);
#endif
#if defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
    supports_dp = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE);
#endif

    return (nk_capability_t)((nk_cap_neon_k * (supports_neon)) | (nk_cap_neonsdot_k * (supports_neon && supports_dp)) |
                             (nk_cap_serial_k));

#else
    return (nk_capability_t)(nk_cap_neon_k | nk_cap_serial_k);
#endif
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_ARM_

#if NK_TARGET_RISCV_

NK_PUBLIC nk_capability_t nk_capabilities_riscv_(void) {
#if defined(NK_DEFINED_LINUX_)
    unsigned long hwcap = getauxval(AT_HWCAP);
    nk_capability_t caps = nk_cap_serial_k;
    if (hwcap & (1UL << 21)) {
        caps |= nk_cap_rvv_k;
        struct {
            long key;
            unsigned long value;
        } pairs[1] = {{4, 0}};
        if (syscall(258, pairs, 1, 0, (void *)0, 0) == 0) {
            if (pairs[0].value & (1ULL << 30)) caps |= nk_cap_rvvhalf_k;
            if (pairs[0].value & (1ULL << 54)) caps |= nk_cap_rvvbf16_k;
            if (pairs[0].value & (1ULL << 48)) caps |= nk_cap_rvvbb_k; // Zvbb
        }
    }
    return caps;
#elif defined(NK_DEFINED_FREEBSD_)
    unsigned long hwcap = 0;
    elf_aux_info(AT_HWCAP, &hwcap, sizeof(hwcap));
    nk_capability_t caps = nk_cap_serial_k;
    if (hwcap & (1UL << 21)) {
        caps |= nk_cap_rvv_k;
        // FreeBSD lacks the Linux hwprobe syscall (258),
        // so zvfh/zvfbfwma/zvbb are compile-time only.
    }
    return caps;
#else
    return nk_cap_serial_k;
#endif
}

#endif // NK_TARGET_RISCV_

#if NK_TARGET_WASM_

#if defined(__EMSCRIPTEN__) && NK_DYNAMIC_DISPATCH && !defined(NK_PYODIDE_SIDE_MODULE)
// Standalone Emscripten dynamic dispatch: EM_JS probes defined in c/numkong.c.
extern int nk_has_v128(void);
extern int nk_has_relaxed(void);
#elif defined(__wasi__) && NK_DEFINED_WASI_
// WASI hosted (NK_WASI_HOSTED=ON): the host provides capability probes via imports.
__attribute__((__import_module__("env"), __import_name__("nk_has_v128"))) extern int nk_has_v128(void);
__attribute__((__import_module__("env"), __import_name__("nk_has_relaxed"))) extern int nk_has_relaxed(void);
#endif

NK_PUBLIC nk_capability_t nk_capabilities_v128relaxed_(void) {
#if ((defined(__EMSCRIPTEN__) && NK_DYNAMIC_DISPATCH) || (defined(__wasi__) && NK_DEFINED_WASI_)) && \
    !defined(NK_PYODIDE_SIDE_MODULE)
    // Hosted environment (Emscripten or WASI with NK_WASI_HOSTED): the host provides
    // runtime probes.  Compile-time flags only mean the *compiler* emitted relaxed-SIMD
    // opcodes, not that the current runtime can execute them.
    int has_relaxed = nk_has_relaxed();
    return has_relaxed ? (nk_cap_serial_k | nk_cap_v128relaxed_k) : nk_cap_serial_k;
#elif defined(__wasm_relaxed_simd__) || defined(__wasm_simd128__)
    // Static WASM or Pyodide side module: if the compiler targeted relaxed SIMD,
    // the runtime must support it (modules with relaxed opcodes fail validation otherwise).
    return nk_cap_serial_k | nk_cap_v128relaxed_k;
#else
    return nk_cap_serial_k;
#endif
}

#endif // NK_TARGET_WASM_

NK_PUBLIC int nk_configure_thread_(nk_capability_t capabilities) {
#if NK_TARGET_X86_
    return nk_configure_thread_x86_(capabilities);
#endif
#if NK_TARGET_ARM_
    return nk_configure_thread_arm_(capabilities);
#endif
    nk_unused_(capabilities);
    return 1; // success — no platform-specific thread configuration needed
}

NK_PUBLIC nk_capability_t nk_capabilities_(void) {
#if NK_TARGET_X86_
    return nk_capabilities_x86_();
#endif
#if NK_TARGET_ARM_
    return nk_capabilities_arm_();
#endif
#if NK_TARGET_RISCV_
    return nk_capabilities_riscv_();
#endif
#if NK_TARGET_WASM_
    return nk_capabilities_v128relaxed_();
#endif
    return nk_cap_serial_k;
}

#if NK_DYNAMIC_DISPATCH

NK_DYNAMIC nk_capability_t nk_capabilities(void);
NK_DYNAMIC int nk_configure_thread(nk_capability_t);
NK_DYNAMIC int nk_uses_dynamic_dispatch(void);
NK_DYNAMIC void nk_dispatch_table_update(nk_capability_t);
NK_DYNAMIC void nk_find_kernel_punned(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t viable,
                                      nk_kernel_punned_t *kernel_output, nk_capability_t *capability_output);

#else

NK_PUBLIC int nk_uses_dynamic_dispatch(void) { return 0; }
NK_PUBLIC int nk_configure_thread(nk_capability_t c) { return nk_configure_thread_(c); }
NK_PUBLIC nk_capability_t nk_capabilities(void) { return nk_capabilities_(); }
NK_PUBLIC void nk_dispatch_table_update(nk_capability_t caps) { nk_unused_(caps); }

#endif

#ifdef __cplusplus
}

#endif
#endif // NK_CAPABILITIES_H
