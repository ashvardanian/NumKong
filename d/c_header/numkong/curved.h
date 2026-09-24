/**
 *  @brief SIMD-accelerated Similarity Measures for Curved Spaces.
 *  @file include/numkong/curved.h
 *  @author Ash Vardanian
 *  @date August 27, 2024
 *
 *  Contains following similarity measures:
 *
 *  - Mahalanobis distance: √((a-b)ᵀ × C × (a-b))
 *  - Bilinear form: aᵀ × C × b
 *  - Bilinear form over complex numbers
 *
 *  For dtypes:
 *
 *  - 64-bit floating point numbers → 64-bit floats
 *  - 32-bit floating point numbers → 64-bit floats
 *  - 16-bit floating point numbers → 32-bit floats
 *  - 16-bit brain-floating point numbers → 32-bit floats
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, NEON+F16, NEON+BF16, SME+F64
 *  - x86: Haswell, Skylake, Genoa
 *  - RISC-V: RVV
 *
 *  @section numerical_stability Numerical Stability
 *
 *  To minimize catastrophic cancellation in large-magnitude sums:
 *  - f32 kernels widen public outputs to f64/f64c and accumulate in f64 precision where possible
 *  - f64 kernels use Dot2 algorithm (Ogita-Rump-Oishi 2005) in SIMD paths
 *  - Serial kernels use Neumaier compensated summation for all types
 *
 *  @section usage Usage and Benefits
 *
 *  These kernels target BLAS level 2 patterns where vectors are combined with a metric
 *  tensor or covariance matrix. Using raw bilinear and Mahalanobis forms avoids constructing
 *  intermediates and keeps memory traffic low, which is often faster than a full GEMM path
 *  for small and medium sizes. Complex bilinear forms return a complex scalar as two reals,
 *  serving complex-valued signals without extra packing or unpacking.
 *
 *  @section references References
 *
 *  - x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  - Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  - Neumaier, A. (1974). "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen"
 *  - Ogita, T., Rump, S.M., Oishi, S. (2005). "Accurate Sum and Dot Product"
 *
 */
#ifndef NK_CURVED_H
#define NK_CURVED_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Bilinear form between vectors a and b under metric tensor C.
 *
 *  Computes aᵀ × C × b = Σᵢ Σⱼ aᵢ × cᵢⱼ × bⱼ
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] c The metric tensor or covariance matrix, stored row-major as an n×n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] result The output bilinear form value.
 *
 *  @note The output value can be negative.
 */
NK_DYNAMIC void nk_bilinear_f64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_bilinear_f64 */
NK_DYNAMIC void nk_bilinear_f32(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n, nk_f64_t *result);
/** @copydoc nk_bilinear_f64 */
NK_DYNAMIC void nk_bilinear_f16(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n, nk_f32_t *result);
/** @copydoc nk_bilinear_f64 */
NK_DYNAMIC void nk_bilinear_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                 nk_f32_t *result);

/**
 *  @brief Mahalanobis distance between vectors a and b under metric tensor C.
 *
 *  Computes √((a-b)ᵀ × C × (a-b)) = √(Σᵢ Σⱼ (aᵢ-bᵢ) × cᵢⱼ × (aⱼ-bⱼ))
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] c The Positive Semi-Definite (PSD) matrix, stored row-major as an n×n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] result The output distance value.
 *
 *  @note The output value is non-negative when C is PSD.
 *  @note The output value is zero if and only if the two vectors are identical.
 *  @note The matrix C must be positive semi-definite. If C is not PSD, the quadratic form
 *        (a-b)ᵀ C (a-b) may be negative, and the square root will produce NaN.
 */
NK_DYNAMIC void nk_mahalanobis_f64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                   nk_f64_t *result);
/** @copydoc nk_mahalanobis_f64 */
NK_DYNAMIC void nk_mahalanobis_f32(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                   nk_f64_t *result);
/** @copydoc nk_mahalanobis_f64 */
NK_DYNAMIC void nk_mahalanobis_f16(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                   nk_f32_t *result);
/** @copydoc nk_mahalanobis_f64 */
NK_DYNAMIC void nk_mahalanobis_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                    nk_f32_t *result);

/**
 *  @brief Complex bilinear form between vectors a and b under metric tensor C.
 *
 *  @param[in] a The first complex vector.
 *  @param[in] b The second complex vector.
 *  @param[in] c The complex metric tensor, stored row-major as an n×n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] results The output complex value with real and imaginary parts.
 */
NK_DYNAMIC void nk_bilinear_f64c(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                 nk_f64c_t *results);
/** @copydoc nk_bilinear_f64c */
NK_DYNAMIC void nk_bilinear_f32c(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                 nk_f64c_t *results);
/** @copydoc nk_bilinear_f64c */
NK_DYNAMIC void nk_bilinear_f16c(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                 nk_f32c_t *results);
/** @copydoc nk_bilinear_f64c */
NK_DYNAMIC void nk_bilinear_bf16c(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                  nk_f32c_t *results);

/** @copydoc nk_bilinear_f64 */
NK_PUBLIC void nk_bilinear_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                      nk_f64_t *result);
/** @copydoc nk_bilinear_f64c */
NK_PUBLIC void nk_bilinear_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                       nk_f64c_t *results);
/** @copydoc nk_mahalanobis_f64 */
NK_PUBLIC void nk_mahalanobis_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                         nk_f64_t *result);
/** @copydoc nk_bilinear_f32 */
NK_PUBLIC void nk_bilinear_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                      nk_f64_t *result);
/** @copydoc nk_bilinear_f32c */
NK_PUBLIC void nk_bilinear_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                       nk_f64c_t *results);
/** @copydoc nk_mahalanobis_f32 */
NK_PUBLIC void nk_mahalanobis_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                         nk_f64_t *result);
/** @copydoc nk_bilinear_f16 */
NK_PUBLIC void nk_bilinear_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                      nk_f32_t *result);
/** @copydoc nk_bilinear_f16c */
NK_PUBLIC void nk_bilinear_f16c_serial(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                       nk_f32c_t *results);
/** @copydoc nk_mahalanobis_f16 */
NK_PUBLIC void nk_mahalanobis_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                         nk_f32_t *result);
/** @copydoc nk_bilinear_bf16 */
NK_PUBLIC void nk_bilinear_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                       nk_f32_t *result);
/** @copydoc nk_bilinear_bf16c */
NK_PUBLIC void nk_bilinear_bf16c_serial(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                        nk_f32c_t *results);
/** @copydoc nk_mahalanobis_bf16 */
NK_PUBLIC void nk_mahalanobis_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                          nk_f32_t *result);

#if NK_TARGET_NEON
/** @copydoc nk_bilinear_f32 */
NK_PUBLIC void nk_bilinear_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                    nk_f64_t *result);
/** @copydoc nk_bilinear_f32c */
NK_PUBLIC void nk_bilinear_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                     nk_f64c_t *results);
/** @copydoc nk_mahalanobis_f32 */
NK_PUBLIC void nk_mahalanobis_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                       nk_f64_t *result);
#endif // NK_TARGET_NEON

#if NK_TARGET_NEONHALF
/** @copydoc nk_bilinear_f16 */
NK_PUBLIC void nk_bilinear_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                        nk_f32_t *result);
/** @copydoc nk_bilinear_f16c */
NK_PUBLIC void nk_bilinear_f16c_neonhalf(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                         nk_f32c_t *results);
/** @copydoc nk_mahalanobis_f16 */
NK_PUBLIC void nk_mahalanobis_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                           nk_f32_t *result);
#endif // NK_TARGET_NEONHALF

#if NK_TARGET_NEONBFDOT
/** @copydoc nk_bilinear_bf16 */
NK_PUBLIC void nk_bilinear_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                          nk_f32_t *result);
/** @copydoc nk_bilinear_bf16c */
NK_PUBLIC void nk_bilinear_bf16c_neonbfdot(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                           nk_f32c_t *results);
/** @copydoc nk_mahalanobis_bf16 */
NK_PUBLIC void nk_mahalanobis_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                             nk_f32_t *result);
#endif // NK_TARGET_NEONBFDOT

#if NK_TARGET_SMEF64
/** @copydoc nk_bilinear_f32 */
NK_PUBLIC void nk_bilinear_f32_smef64(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                      nk_f64_t *result);
/** @copydoc nk_bilinear_f32c */
NK_PUBLIC void nk_bilinear_f32c_smef64(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                       nk_f64c_t *result);
/** @copydoc nk_mahalanobis_f32 */
NK_PUBLIC void nk_mahalanobis_f32_smef64(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                         nk_f64_t *result);
/** @copydoc nk_bilinear_f64 */
NK_PUBLIC void nk_bilinear_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                      nk_f64_t *result);
/** @copydoc nk_bilinear_f64c */
NK_PUBLIC void nk_bilinear_f64c_smef64(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                       nk_f64c_t *result);
/** @copydoc nk_mahalanobis_f64 */
NK_PUBLIC void nk_mahalanobis_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                         nk_f64_t *result);
#endif // NK_TARGET_SMEF64

#if NK_TARGET_HASWELL
/** @copydoc nk_bilinear_f32 */
NK_PUBLIC void nk_bilinear_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                       nk_f64_t *result);
/** @copydoc nk_mahalanobis_f32 */
NK_PUBLIC void nk_mahalanobis_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                          nk_f64_t *result);
/** @copydoc nk_bilinear_f16 */
NK_PUBLIC void nk_bilinear_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                       nk_f32_t *result);
/** @copydoc nk_mahalanobis_f16 */
NK_PUBLIC void nk_mahalanobis_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                          nk_f32_t *result);
/** @copydoc nk_bilinear_bf16 */
NK_PUBLIC void nk_bilinear_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                        nk_f32_t *result);
/** @copydoc nk_mahalanobis_bf16 */
NK_PUBLIC void nk_mahalanobis_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                           nk_f32_t *result);
#endif // NK_TARGET_HASWELL

#if NK_TARGET_SKYLAKE
/** @copydoc nk_bilinear_f64 */
NK_PUBLIC void nk_bilinear_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                       nk_f64_t *result);
/** @copydoc nk_bilinear_f64c */
NK_PUBLIC void nk_bilinear_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                        nk_f64c_t *results);
/** @copydoc nk_mahalanobis_f64 */
NK_PUBLIC void nk_mahalanobis_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                          nk_f64_t *result);
/** @copydoc nk_bilinear_f32 */
NK_PUBLIC void nk_bilinear_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                       nk_f64_t *result);
/** @copydoc nk_bilinear_f32c */
NK_PUBLIC void nk_bilinear_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                        nk_f64c_t *results);
/** @copydoc nk_mahalanobis_f32 */
NK_PUBLIC void nk_mahalanobis_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                          nk_f64_t *result);
#endif // NK_TARGET_SKYLAKE

#if NK_TARGET_GENOA
/** @copydoc nk_bilinear_bf16 */
NK_PUBLIC void nk_bilinear_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                      nk_f32_t *result);
/** @copydoc nk_bilinear_bf16c */
NK_PUBLIC void nk_bilinear_bf16c_genoa(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                       nk_f32c_t *results);
/** @copydoc nk_mahalanobis_bf16 */
NK_PUBLIC void nk_mahalanobis_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                         nk_f32_t *result);
#endif // NK_TARGET_GENOA

#if NK_TARGET_RVV
/** @copydoc nk_bilinear_f64 */
NK_PUBLIC void nk_bilinear_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                   nk_f64_t *result);
/** @copydoc nk_mahalanobis_f64 */
NK_PUBLIC void nk_mahalanobis_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                      nk_f64_t *result);
/** @copydoc nk_bilinear_f32 */
NK_PUBLIC void nk_bilinear_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                   nk_f64_t *result);
/** @copydoc nk_mahalanobis_f32 */
NK_PUBLIC void nk_mahalanobis_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                      nk_f64_t *result);
/** @copydoc nk_bilinear_f16 */
NK_PUBLIC void nk_bilinear_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                   nk_f32_t *result);
/** @copydoc nk_mahalanobis_f16 */
NK_PUBLIC void nk_mahalanobis_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                      nk_f32_t *result);
/** @copydoc nk_bilinear_bf16 */
NK_PUBLIC void nk_bilinear_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                    nk_f32_t *result);
/** @copydoc nk_mahalanobis_bf16 */
NK_PUBLIC void nk_mahalanobis_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                       nk_f32_t *result);
#endif // NK_TARGET_RVV

/**
 *  @brief  Returns the output dtype for bilinear forms.
 */
NK_INTERNAL nk_dtype_t nk_bilinear_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    case nk_f64c_k: return nk_f64c_k;
    case nk_f32c_k: return nk_f64c_k;
    case nk_f16c_k: return nk_f32c_k;
    case nk_bf16c_k: return nk_f32c_k;
    default: return nk_dtype_unknown_k;
    }
}

/**
 *  @brief  Returns the output dtype for Mahalanobis metrics.
 */
NK_INTERNAL nk_dtype_t nk_mahalanobis_output_dtype(nk_dtype_t dtype) {
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

#include "numkong/curved/serial.h"
#include "numkong/curved/neon.h"
#include "numkong/curved/neonhalf.h"
#include "numkong/curved/neonbfdot.h"
#include "numkong/curved/smef64.h"
#include "numkong/curved/haswell.h"
#include "numkong/curved/skylake.h"
#include "numkong/curved/genoa.h"
#include "numkong/curved/rvv.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_bilinear_f64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_bilinear_f64_skylake(a, b, c, n, result);
#elif NK_TARGET_SMEF64
    nk_bilinear_f64_smef64(a, b, c, n, result);
#elif NK_TARGET_RVV
    nk_bilinear_f64_rvv(a, b, c, n, result);
#else
    nk_bilinear_f64_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_bilinear_f32(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n, nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_bilinear_f32_skylake(a, b, c, n, result);
#elif NK_TARGET_SMEF64
    nk_bilinear_f32_smef64(a, b, c, n, result);
#elif NK_TARGET_HASWELL
    nk_bilinear_f32_haswell(a, b, c, n, result);
#elif NK_TARGET_NEON
    nk_bilinear_f32_neon(a, b, c, n, result);
#elif NK_TARGET_RVV
    nk_bilinear_f32_rvv(a, b, c, n, result);
#else
    nk_bilinear_f32_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_bilinear_f16(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n, nk_f32_t *result) {
#if NK_TARGET_HASWELL
    nk_bilinear_f16_haswell(a, b, c, n, result);
#elif NK_TARGET_NEONHALF
    nk_bilinear_f16_neonhalf(a, b, c, n, result);
#elif NK_TARGET_RVV
    nk_bilinear_f16_rvv(a, b, c, n, result);
#else
    nk_bilinear_f16_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_bilinear_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_bilinear_bf16_genoa(a, b, c, n, result);
#elif NK_TARGET_HASWELL
    nk_bilinear_bf16_haswell(a, b, c, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_bilinear_bf16_neonbfdot(a, b, c, n, result);
#elif NK_TARGET_RVV
    nk_bilinear_bf16_rvv(a, b, c, n, result);
#else
    nk_bilinear_bf16_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_bilinear_f64c(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                nk_f64c_t *results) {
#if NK_TARGET_SKYLAKE
    nk_bilinear_f64c_skylake(a, b, c, n, results);
#elif NK_TARGET_SMEF64
    nk_bilinear_f64c_smef64(a, b, c, n, results);
#else
    nk_bilinear_f64c_serial(a, b, c, n, results);
#endif
}

NK_PUBLIC void nk_bilinear_f32c(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                nk_f64c_t *results) {
#if NK_TARGET_SKYLAKE
    nk_bilinear_f32c_skylake(a, b, c, n, results);
#elif NK_TARGET_SMEF64
    nk_bilinear_f32c_smef64(a, b, c, n, results);
#elif NK_TARGET_NEON
    nk_bilinear_f32c_neon(a, b, c, n, results);
#else
    nk_bilinear_f32c_serial(a, b, c, n, results);
#endif
}

NK_PUBLIC void nk_bilinear_f16c(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                nk_f32c_t *results) {
#if NK_TARGET_NEONHALF
    nk_bilinear_f16c_neonhalf(a, b, c, n, results);
#else
    nk_bilinear_f16c_serial(a, b, c, n, results);
#endif
}

NK_PUBLIC void nk_bilinear_bf16c(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c, nk_size_t n,
                                 nk_f32c_t *results) {
#if NK_TARGET_GENOA
    nk_bilinear_bf16c_genoa(a, b, c, n, results);
#elif NK_TARGET_NEONBFDOT
    nk_bilinear_bf16c_neonbfdot(a, b, c, n, results);
#else
    nk_bilinear_bf16c_serial(a, b, c, n, results);
#endif
}

NK_PUBLIC void nk_mahalanobis_f64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                  nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_mahalanobis_f64_skylake(a, b, c, n, result);
#elif NK_TARGET_SMEF64
    nk_mahalanobis_f64_smef64(a, b, c, n, result);
#elif NK_TARGET_RVV
    nk_mahalanobis_f64_rvv(a, b, c, n, result);
#else
    nk_mahalanobis_f64_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_mahalanobis_f32(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                  nk_f64_t *result) {
#if NK_TARGET_SKYLAKE
    nk_mahalanobis_f32_skylake(a, b, c, n, result);
#elif NK_TARGET_SMEF64
    nk_mahalanobis_f32_smef64(a, b, c, n, result);
#elif NK_TARGET_HASWELL
    nk_mahalanobis_f32_haswell(a, b, c, n, result);
#elif NK_TARGET_NEON
    nk_mahalanobis_f32_neon(a, b, c, n, result);
#elif NK_TARGET_RVV
    nk_mahalanobis_f32_rvv(a, b, c, n, result);
#else
    nk_mahalanobis_f32_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_mahalanobis_f16(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                  nk_f32_t *result) {
#if NK_TARGET_HASWELL
    nk_mahalanobis_f16_haswell(a, b, c, n, result);
#elif NK_TARGET_NEONHALF
    nk_mahalanobis_f16_neonhalf(a, b, c, n, result);
#elif NK_TARGET_RVV
    nk_mahalanobis_f16_rvv(a, b, c, n, result);
#else
    nk_mahalanobis_f16_serial(a, b, c, n, result);
#endif
}

NK_PUBLIC void nk_mahalanobis_bf16(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                   nk_f32_t *result) {
#if NK_TARGET_GENOA
    nk_mahalanobis_bf16_genoa(a, b, c, n, result);
#elif NK_TARGET_HASWELL
    nk_mahalanobis_bf16_haswell(a, b, c, n, result);
#elif NK_TARGET_NEONBFDOT
    nk_mahalanobis_bf16_neonbfdot(a, b, c, n, result);
#elif NK_TARGET_RVV
    nk_mahalanobis_bf16_rvv(a, b, c, n, result);
#else
    nk_mahalanobis_bf16_serial(a, b, c, n, result);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_CURVED_H
