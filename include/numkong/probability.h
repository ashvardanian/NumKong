/**
 *  @file include/numkong/probability.h
 *  @author Ash Vardanian
 *  @date October 20, 2023
 *  @brief SIMD-accelerated similarity measures for probability distributions.
 *
 *  Contains following similarity measures:
 *
 *  - Kullback-Leibler Divergence (KLD)
 *  - Jensen-Shannon Distance (JSD)
 *
 *  For dtypes:
 *
 *  - 64-bit floating point numbers → 64-bit
 *  - 32-bit floating point numbers → 64-bit
 *  - 16-bit floating point numbers → 32-bit
 *  - 16-bit brain-floating point numbers → 32-bit
 *
 *  Precision policy:
 *
 *  - For @c f32 inputs, the per-element vertical path stays in @c f32 to preserve the fast
 *    ratio/log approximations and SIMD throughput.
 *  - The horizontal reduction over those per-element contributions widens to @c f64, and public
 *    @c f32 results are exposed as @c f64.
 *  - For @c f64 inputs, both the vertical path and the horizontal reduction stay in @c f64, with
 *    stable summation in the serial kernels.
 *  - For @c f16 and @c bf16 inputs, the kernels still widen to @c f32.
 *  - Both operands of every ratio are clamped to at least ε, which is
 *    @c NUMKONG_F32_DIVISION_EPSILON or @c NUMKONG_F64_DIVISION_EPSILON, so terms at or above ε
 *    follow the exact formula.
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON
 *  - x86: Haswell, Skylake, Sapphire
 *
 *  @section probability_x86_instructions Relevant x86 Instructions
 *
 *  KL/JS divergence requires log2(x) which decomposes into exponent extraction (VGETEXP) plus
 *  mantissa polynomial (using VGETMANT + FMA chain). This approach is faster than scalar log()
 *  calls. Division (for p/q ratio) uses either VDIVPS directly or VRCP14PS with Newton-Raphson
 *  refinement when ~14-bit precision suffices. Genoa's VGETEXP/VGETMANT are 25% faster than Ice.
 *
 *  @verbatim
 *  Intrinsic          Instruction                  Icelake           Genoa
 *  _mm512_getexp_ps   VGETEXPPS (ZMM, ZMM)         4cy @ p0          3cy @ p23
 *  _mm512_getexp_pd   VGETEXPPD (ZMM, ZMM)         4cy @ p0          3cy @ p23
 *  _mm512_getmant_ps  VGETMANTPS (ZMM, ZMM, I8)    4cy @ p0          3cy @ p23
 *  _mm512_getmant_pd  VGETMANTPD (ZMM, ZMM, I8)    4cy @ p0          3cy @ p23
 *  _mm512_rcp14_ps    VRCP14PS (ZMM, ZMM)          7cy @ p0+p0+p05   5cy @ p01
 *  _mm512_div_ps      VDIVPS (ZMM, ZMM, ZMM)       17cy @ p0+p0+p05  11cy @ p01
 *  _mm512_fmadd_ps    VFMADD231PS (ZMM, ZMM, ZMM)  4cy @ p0          4cy @ p01
 *  @endverbatim
 *
 *  @section probability_arm_instructions Relevant ARM NEON/SVE Instructions
 *
 *  ARM lacks direct exponent/mantissa extraction, so log2 uses integer reinterpretation of the
 *  float bits followed by polynomial refinement. FRECPE provides ~8-bit reciprocal approximation
 *  for division, refined with FRECPS Newton-Raphson steps to ~22-bit precision.
 *
 *  @verbatim
 *  Intrinsic    Instruction   M1 Firestorm  Graviton 3   Graviton 4
 *  vfmaq_f32    FMLA.S (vec)  4cy @ V0123   4cy @ V0123  4cy @ V0123
 *  vrecpeq_f32  FRECPE.S      3cy @ V02     3cy @ V02    3cy @ V02
 *  vrecpsq_f32  FRECPS.S      4cy @ V0123   4cy @ V0123  4cy @ V0123
 *  @endverbatim
 *
 *  @section probability_references References
 *
 *  @see x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NUMKONG_PROBABILITY_H
#define NUMKONG_PROBABILITY_H

#include "numkong/capabilities.h"

#if !defined(NUMKONG_F64_DIVISION_EPSILON)
#define NUMKONG_F64_DIVISION_EPSILON (1e-15)
#endif

#if !defined(NUMKONG_F32_DIVISION_EPSILON)
#define NUMKONG_F32_DIVISION_EPSILON (1e-7f)
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Kullback-Leibler divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
NUMKONG_API nk_status_t nk_kld_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                        nk_capability_t capabilities, void *stream);

/**
 *  @brief Kullback-Leibler divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
NUMKONG_API nk_status_t nk_kld_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                         nk_capability_t capabilities, void *stream);

/**
 *  @brief Kullback-Leibler divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
NUMKONG_API nk_status_t nk_kld_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_capability_t capabilities, void *stream);

/**
 *  @brief Kullback-Leibler divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
NUMKONG_API nk_status_t nk_kld_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_capability_t capabilities, void *stream);

/**
 *  @brief Jensen-Shannon distance between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two distributions are identical.
 */
NUMKONG_API nk_status_t nk_jsd_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                        nk_capability_t capabilities, void *stream);

/**
 *  @brief Jensen-Shannon distance between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two distributions are identical.
 */
NUMKONG_API nk_status_t nk_jsd_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                         nk_capability_t capabilities, void *stream);

/**
 *  @brief Jensen-Shannon distance between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two distributions are identical.
 */
NUMKONG_API nk_status_t nk_jsd_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_capability_t capabilities, void *stream);

/**
 *  @brief Jensen-Shannon distance between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two distributions are identical.
 */
NUMKONG_API nk_status_t nk_jsd_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_capability_t capabilities, void *stream);

/** @copydoc nk_kld_f64_best */
NUMKONG_API nk_status_t nk_kld_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                          void *stream);
/** @copydoc nk_jsd_f64_best */
NUMKONG_API nk_status_t nk_jsd_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                          void *stream);
/** @copydoc nk_kld_f32_best */
NUMKONG_API nk_status_t nk_kld_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                          void *stream);
/** @copydoc nk_jsd_f32_best */
NUMKONG_API nk_status_t nk_jsd_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                          void *stream);
/** @copydoc nk_kld_f16_best */
NUMKONG_API nk_status_t nk_kld_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                          void *stream);
/** @copydoc nk_jsd_f16_best */
NUMKONG_API nk_status_t nk_jsd_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                          void *stream);
/** @copydoc nk_kld_bf16_best */
NUMKONG_API nk_status_t nk_kld_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                           void *stream);
/** @copydoc nk_jsd_bf16_best */
NUMKONG_API nk_status_t nk_jsd_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                           void *stream);

#if NUMKONG_TARGET_NEON
/** @copydoc nk_kld_f32_best */
NUMKONG_API nk_status_t nk_kld_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                        void *stream);
/** @copydoc nk_jsd_f32_best */
NUMKONG_API nk_status_t nk_jsd_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                        void *stream);
/** @copydoc nk_kld_f16_best */
NUMKONG_API nk_status_t nk_kld_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                        void *stream);
/** @copydoc nk_jsd_f16_best */
NUMKONG_API nk_status_t nk_jsd_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                        void *stream);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_kld_f64_best */
NUMKONG_API nk_status_t nk_kld_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                           void *stream);
/** @copydoc nk_jsd_f64_best */
NUMKONG_API nk_status_t nk_jsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                           void *stream);
/** @copydoc nk_kld_f16_best */
NUMKONG_API nk_status_t nk_kld_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                           void *stream);
/** @copydoc nk_jsd_f16_best */
NUMKONG_API nk_status_t nk_jsd_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                           void *stream);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_kld_f64_best */
NUMKONG_API nk_status_t nk_kld_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                           void *stream);
/** @copydoc nk_jsd_f64_best */
NUMKONG_API nk_status_t nk_jsd_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                           void *stream);
/** @copydoc nk_kld_f32_best */
NUMKONG_API nk_status_t nk_kld_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                           void *stream);
/** @copydoc nk_jsd_f32_best */
NUMKONG_API nk_status_t nk_jsd_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                           void *stream);
/** @copydoc nk_kld_f16_best */
NUMKONG_API nk_status_t nk_kld_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                           void *stream);
/** @copydoc nk_jsd_f16_best */
NUMKONG_API nk_status_t nk_jsd_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                           void *stream);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_RVV
/** @copydoc nk_kld_f32_best */
NUMKONG_API nk_status_t nk_kld_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                       void *stream);
/** @copydoc nk_jsd_f32_best */
NUMKONG_API nk_status_t nk_jsd_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                       void *stream);
/** @copydoc nk_kld_f64_best */
NUMKONG_API nk_status_t nk_kld_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                       void *stream);
/** @copydoc nk_jsd_f64_best */
NUMKONG_API nk_status_t nk_jsd_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                       void *stream);
/** @copydoc nk_kld_f16_best */
NUMKONG_API nk_status_t nk_kld_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                       void *stream);
/** @copydoc nk_jsd_f16_best */
NUMKONG_API nk_status_t nk_jsd_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                       void *stream);
/** @copydoc nk_kld_bf16_best */
NUMKONG_API nk_status_t nk_kld_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                        void *stream);
/** @copydoc nk_jsd_bf16_best */
NUMKONG_API nk_status_t nk_jsd_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                        void *stream);
#endif // NUMKONG_TARGET_RVV

/** Returns the output dtype for probability measures (KLD, JSD). */
NUMKONG_INLINE nk_dtype_t nk_probability_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief Finds the probability kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when no capability in @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k when header-only.
 */
NUMKONG_API nk_status_t nk_probability_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype,
                                                   nk_capability_t capabilities, nk_kernel_punned_t *kernel,
                                                   nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/probability/serial.h"
#include "numkong/probability/neon.h"
#include "numkong/probability/haswell.h"
#include "numkong/probability/skylake.h"
#include "numkong/probability/rvv.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_kld_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_jsd_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_kld_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_jsd_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result,
                                        nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_kld_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                         nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_jsd_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result,
                                         nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_kld_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                        nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_jsd_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result,
                                        nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_probability_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype,
                                                   nk_capability_t capabilities, nk_kernel_punned_t *kernel,
                                                   nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = (nk_kernel_punned_t)NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif
