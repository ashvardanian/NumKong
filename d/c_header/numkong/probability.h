/**
 *  @brief SIMD-accelerated Similarity Measures for Probability Distributions.
 *  @file include/numkong/probability.h
 *  @author Ash Vardanian
 *  @date October 20, 2023
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
 *  - For `f32` inputs, the per-element vertical path stays in `f32` to preserve the fast ratio/log
 *    approximations and SIMD throughput.
 *  - The horizontal reduction over those per-element contributions widens to `f64`, and public
 *    `f32` results are exposed as `f64`.
 *  - For `f64` inputs, both the vertical path and the horizontal reduction stay in `f64`, with
 *    stable summation in the serial kernels.
 *  - For `f16` and `bf16` inputs, the kernels still widen to `f32`.
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON
 *  - x86: Haswell, Skylake, Sapphire
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  KL/JS divergence requires log2(x) which decomposes into exponent extraction (VGETEXP) plus
 *  mantissa polynomial (using VGETMANT + FMA chain). This approach is faster than scalar log()
 *  calls. Division (for p/q ratio) uses either VDIVPS directly or VRCP14PS with Newton-Raphson
 *  refinement when ~14-bit precision suffices. Genoa's VGETEXP/VGETMANT are 25% faster than Ice.
 *
 *      Intrinsic               Instruction                     Ice         Genoa
 *      _mm512_getexp_ps        VGETEXPPS (ZMM, ZMM)            4c @ p0     3c @ p23
 *      _mm512_getexp_pd        VGETEXPPD (ZMM, ZMM)            4c @ p0     3c @ p23
 *      _mm512_getmant_ps       VGETMANTPS (ZMM, ZMM, I8)       4c @ p0     3c @ p23
 *      _mm512_getmant_pd       VGETMANTPD (ZMM, ZMM, I8)       4c @ p0     3c @ p23
 *      _mm512_rcp14_ps         VRCP14PS (ZMM, ZMM)             7c @ p05    5c @ p01
 *      _mm512_div_ps           VDIVPS (ZMM, ZMM, ZMM)          17c @ p05   11c @ p01
 *      _mm512_fmadd_ps         VFMADD231PS (ZMM, ZMM, ZMM)     4c @ p0     4c @ p01
 *
 *  @section arm_instructions Relevant ARM NEON/SVE Instructions
 *
 *  ARM lacks direct exponent/mantissa extraction, so log2 uses integer reinterpretation of the
 *  float bits followed by polynomial refinement. FRECPE provides ~8-bit reciprocal approximation
 *  for division, refined with FRECPS Newton-Raphson steps to ~22-bit precision.
 *
 *      Intrinsic               Instruction     M1 Firestorm    Graviton 3      Graviton 4
 *      vfmaq_f32               FMLA.S (vec)    4c @ V0123      4c @ V0123      4c @ V0123
 *      vrecpeq_f32             FRECPE.S        3c @ V02        3c @ V02        3c @ V02
 *      vrecpsq_f32             FRECPS.S        4c @ V0123      4c @ V0123      4c @ V0123
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *
 */
#ifndef NK_PROBABILITY_H
#define NK_PROBABILITY_H

#include "numkong/types.h"
#include "numkong/reduce.h" // For horizontal reduction helpers

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
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
NK_DYNAMIC void nk_kld_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/**
 *  @brief Kullback-Leibler divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
NK_DYNAMIC void nk_kld_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/**
 *  @brief Kullback-Leibler divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
NK_DYNAMIC void nk_kld_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/**
 *  @brief Kullback-Leibler divergence between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output divergence value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output divergence value is non-negative.
 *  @note The output divergence value is zero if and only if the two distributions are identical.
 */
NK_DYNAMIC void nk_kld_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/**
 *  @brief Jensen-Shannon distance between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output distance value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two distributions are identical.
 */
NK_DYNAMIC void nk_jsd_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/**
 *  @brief Jensen-Shannon distance between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output distance value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two distributions are identical.
 */
NK_DYNAMIC void nk_jsd_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/**
 *  @brief Jensen-Shannon distance between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output distance value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two distributions are identical.
 */
NK_DYNAMIC void nk_jsd_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/**
 *  @brief Jensen-Shannon distance between two discrete probability distributions.
 *
 *  @param[in] a The first discrete probability distribution.
 *  @param[in] b The second discrete probability distribution.
 *  @param[in] n The number of elements in the distributions.
 *  @param[out] result The output distance value.
 *
 *  @note The distributions are assumed to be normalized.
 *  @note The output distance value is non-negative.
 *  @note The output distance value is zero if and only if the two distributions are identical.
 */
NK_DYNAMIC void nk_jsd_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);

/** @copydoc nk_kld_f64 */
NK_PUBLIC void nk_kld_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_jsd_f64 */
NK_PUBLIC void nk_jsd_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_kld_f32 */
NK_PUBLIC void nk_kld_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_jsd_f32 */
NK_PUBLIC void nk_jsd_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_kld_f16 */
NK_PUBLIC void nk_kld_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_f16 */
NK_PUBLIC void nk_jsd_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_kld_bf16 */
NK_PUBLIC void nk_kld_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_bf16 */
NK_PUBLIC void nk_jsd_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);

#if NK_TARGET_NEON
/** @copydoc nk_kld_f32 */
NK_PUBLIC void nk_kld_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_jsd_f32 */
NK_PUBLIC void nk_jsd_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
/** @copydoc nk_kld_f16 */
NK_PUBLIC void nk_kld_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_f16 */
NK_PUBLIC void nk_jsd_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_HASWELL
/** @copydoc nk_kld_f64 */
NK_PUBLIC void nk_kld_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_jsd_f64 */
NK_PUBLIC void nk_jsd_f64_haswell(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_kld_f16 */
NK_PUBLIC void nk_kld_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_f16 */
NK_PUBLIC void nk_jsd_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_kld_f64 */
NK_PUBLIC void nk_kld_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_jsd_f64 */
NK_PUBLIC void nk_jsd_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_kld_f32 */
NK_PUBLIC void nk_kld_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_jsd_f32 */
NK_PUBLIC void nk_jsd_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_kld_f16 */
NK_PUBLIC void nk_kld_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_f16 */
NK_PUBLIC void nk_jsd_f16_skylake(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_RVV
/** @copydoc nk_kld_f32 */
NK_PUBLIC void nk_kld_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_jsd_f32 */
NK_PUBLIC void nk_jsd_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_kld_f64 */
NK_PUBLIC void nk_kld_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_jsd_f64 */
NK_PUBLIC void nk_jsd_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_kld_f16 */
NK_PUBLIC void nk_kld_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_f16 */
NK_PUBLIC void nk_jsd_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_kld_bf16 */
NK_PUBLIC void nk_kld_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_jsd_bf16 */
NK_PUBLIC void nk_jsd_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result);
#endif // NK_TARGET_RVV

/** @brief Returns the output dtype for probability measures (KLD, JSD). */
NK_INTERNAL nk_dtype_t nk_probability_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/probability/serial.h"
#include "numkong/probability/neon.h"
#include "numkong/probability/haswell.h"
#include "numkong/probability/skylake.h"
#include "numkong/probability/rvv.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_kld_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_NEONHALF
    nk_kld_f16_neonhalf(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_kld_f16_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_kld_f16_haswell(a, b, n, result);
#elif NK_TARGET_RVV
    nk_kld_f16_rvv(a, b, n, result);
#else
    nk_kld_f16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_kld_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_RVV
    nk_kld_bf16_rvv(a, b, n, result);
#else
    nk_kld_bf16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_kld_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_NEON
    nk_kld_f32_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_kld_f32_skylake(a, b, n, result);
#elif NK_TARGET_RVV
    nk_kld_f32_rvv(a, b, n, result);
#else
    nk_kld_f32_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_kld_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_kld_f64_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_kld_f64_haswell(a, b, n, result);
#elif NK_TARGET_RVV
    nk_kld_f64_rvv(a, b, n, result);
#else
    nk_kld_f64_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_jsd_f16(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_NEONHALF
    nk_jsd_f16_neonhalf(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_jsd_f16_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_jsd_f16_haswell(a, b, n, result);
#elif NK_TARGET_RVV
    nk_jsd_f16_rvv(a, b, n, result);
#else
    nk_jsd_f16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_jsd_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_RVV
    nk_jsd_bf16_rvv(a, b, n, result);
#else
    nk_jsd_bf16_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_jsd_f32(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_NEON
    nk_jsd_f32_neon(a, b, n, result);
#elif NK_TARGET_SKYLAKE
    nk_jsd_f32_skylake(a, b, n, result);
#elif NK_TARGET_RVV
    nk_jsd_f32_rvv(a, b, n, result);
#else
    nk_jsd_f32_serial(a, b, n, result);
#endif
}

NK_PUBLIC void nk_jsd_f64(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_jsd_f64_skylake(a, b, n, result);
#elif NK_TARGET_HASWELL
    nk_jsd_f64_haswell(a, b, n, result);
#elif NK_TARGET_RVV
    nk_jsd_f64_rvv(a, b, n, result);
#else
    nk_jsd_f64_serial(a, b, n, result);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif
