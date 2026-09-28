/**
 *  @file include/numkong/curved.h
 *  @author Ash Vardanian
 *  @date August 27, 2024
 *  @brief SIMD-accelerated similarity measures for curved spaces.
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
 *  @section curved_numerical_stability Numerical Stability
 *
 *  To minimize catastrophic cancellation in large-magnitude sums:
 *  - f32 kernels widen public outputs to f64/f64c and accumulate in f64 precision where possible
 *  - f64 kernels use Dot2 algorithm (Ogita-Rump-Oishi 2005) in SIMD paths
 *  - Serial kernels use Neumaier compensated summation for all types
 *
 *  @section curved_usage Usage and Benefits
 *
 *  These kernels target BLAS level 2 patterns where vectors are combined with a metric tensor or
 *  covariance matrix. Using raw bilinear and Mahalanobis forms avoids constructing intermediates
 *  and keeps memory traffic low, which is often faster than a full GEMM path for small and medium
 *  sizes. Complex bilinear forms return a complex scalar as two reals, serving complex-valued
 *  signals without extra packing or unpacking.
 *
 *  @section curved_references References
 *
 *  - Neumaier, A. (1974). "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher Summen"
 *  - Ogita, T., Rump, S.M., Oishi, S. (2005). "Accurate Sum and Dot Product"
 *
 *  @see x86 intrinsics: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm intrinsics: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 */
#ifndef NUMKONG_CURVED_H
#define NUMKONG_CURVED_H

#include "numkong/capabilities.h"

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
 *  @param[in] c The metric tensor or covariance matrix, stored row-major as an n × n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] result The output bilinear form value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The output value can be negative.
 */
NUMKONG_API nk_status_t nk_bilinear_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                             nk_f64_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_bilinear_f64_best */
NUMKONG_API nk_status_t nk_bilinear_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                             nk_f64_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_bilinear_f64_best */
NUMKONG_API nk_status_t nk_bilinear_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                             nk_f32_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_bilinear_f64_best */
NUMKONG_API nk_status_t nk_bilinear_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                              nk_f32_t *result, nk_capability_t capabilities, void *stream);

/**
 *  @brief Mahalanobis distance between vectors a and b under metric tensor C.
 *
 *  Computes √((a-b)ᵀ × C × (a-b)) = √(Σᵢ Σⱼ (aᵢ-bᵢ) × cᵢⱼ × (aⱼ-bⱼ))
 *
 *  @param[in] a The first vector.
 *  @param[in] b The second vector.
 *  @param[in] c The Positive Semi-Definite (PSD) matrix, stored row-major as an n × n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] result The output distance value.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note The output value is non-negative when C is PSD.
 *  @note The output value is zero if and only if the two vectors are identical.
 *  @note The matrix C must be positive semi-definite. If C is not PSD, the quadratic form (a-b)ᵀ C
 *      (a-b) may be negative, and the square root will produce NaN.
 */
NUMKONG_API nk_status_t nk_mahalanobis_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                                nk_f64_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_mahalanobis_f64_best */
NUMKONG_API nk_status_t nk_mahalanobis_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                                nk_f64_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_mahalanobis_f64_best */
NUMKONG_API nk_status_t nk_mahalanobis_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                                nk_f32_t *result, nk_capability_t capabilities, void *stream);
/** @copydoc nk_mahalanobis_f64_best */
NUMKONG_API nk_status_t nk_mahalanobis_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                 nk_size_t n, nk_f32_t *result, nk_capability_t capabilities,
                                                 void *stream);

/**
 *  @brief Complex bilinear form between vectors a and b under metric tensor C.
 *
 *  @param[in] a The first complex vector.
 *  @param[in] b The second complex vector.
 *  @param[in] c The complex metric tensor, stored row-major as an n × n matrix.
 *  @param[in] n The number of dimensions in the vectors.
 *  @param[out] results The output complex value with real and imaginary parts.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_bilinear_f64c_best(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                              nk_f64c_t *results, nk_capability_t capabilities, void *stream);
/** @copydoc nk_bilinear_f64c_best */
NUMKONG_API nk_status_t nk_bilinear_f32c_best(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                              nk_f64c_t *results, nk_capability_t capabilities, void *stream);
/** @copydoc nk_bilinear_f64c_best */
NUMKONG_API nk_status_t nk_bilinear_f16c_best(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                              nk_f32c_t *results, nk_capability_t capabilities, void *stream);
/** @copydoc nk_bilinear_f64c_best */
NUMKONG_API nk_status_t nk_bilinear_bf16c_best(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c,
                                               nk_size_t n, nk_f32c_t *results, nk_capability_t capabilities,
                                               void *stream);

/** @copydoc nk_bilinear_f64_best */
NUMKONG_API nk_status_t nk_bilinear_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                               nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f64c_best */
NUMKONG_API nk_status_t nk_bilinear_f64c_serial(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                                nk_f64c_t *results, void *stream);
/** @copydoc nk_mahalanobis_f64_best */
NUMKONG_API nk_status_t nk_mahalanobis_f64_serial(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                                  nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f32_best */
NUMKONG_API nk_status_t nk_bilinear_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                               nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f32c_best */
NUMKONG_API nk_status_t nk_bilinear_f32c_serial(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                                nk_f64c_t *results, void *stream);
/** @copydoc nk_mahalanobis_f32_best */
NUMKONG_API nk_status_t nk_mahalanobis_f32_serial(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                                  nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f16_best */
NUMKONG_API nk_status_t nk_bilinear_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                               nk_f32_t *result, void *stream);
/** @copydoc nk_bilinear_f16c_best */
NUMKONG_API nk_status_t nk_bilinear_f16c_serial(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                                nk_f32c_t *results, void *stream);
/** @copydoc nk_mahalanobis_f16_best */
NUMKONG_API nk_status_t nk_mahalanobis_f16_serial(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                                  nk_f32_t *result, void *stream);
/** @copydoc nk_bilinear_bf16_best */
NUMKONG_API nk_status_t nk_bilinear_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                                nk_f32_t *result, void *stream);
/** @copydoc nk_bilinear_bf16c_best */
NUMKONG_API nk_status_t nk_bilinear_bf16c_serial(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c,
                                                 nk_size_t n, nk_f32c_t *results, void *stream);
/** @copydoc nk_mahalanobis_bf16_best */
NUMKONG_API nk_status_t nk_mahalanobis_bf16_serial(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                   nk_size_t n, nk_f32_t *result, void *stream);

#if NUMKONG_TARGET_NEON
/** @copydoc nk_bilinear_f32_best */
NUMKONG_API nk_status_t nk_bilinear_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                             nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f32c_best */
NUMKONG_API nk_status_t nk_bilinear_f32c_neon(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                              nk_f64c_t *results, void *stream);
/** @copydoc nk_mahalanobis_f32_best */
NUMKONG_API nk_status_t nk_mahalanobis_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                                nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f16_best */
NUMKONG_API nk_status_t nk_bilinear_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                             nk_f32_t *result, void *stream);
/** @copydoc nk_bilinear_f16c_best */
NUMKONG_API nk_status_t nk_bilinear_f16c_neon(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                              nk_f32c_t *results, void *stream);
/** @copydoc nk_mahalanobis_f16_best */
NUMKONG_API nk_status_t nk_mahalanobis_f16_neon(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                                nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_NEONBFDOT
/** @copydoc nk_bilinear_bf16_best */
NUMKONG_API nk_status_t nk_bilinear_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                   nk_size_t n, nk_f32_t *result, void *stream);
/** @copydoc nk_bilinear_bf16c_best */
NUMKONG_API nk_status_t nk_bilinear_bf16c_neonbfdot(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c,
                                                    nk_size_t n, nk_f32c_t *results, void *stream);
/** @copydoc nk_mahalanobis_bf16_best */
NUMKONG_API nk_status_t nk_mahalanobis_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                      nk_size_t n, nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_NEONBFDOT

#if NUMKONG_TARGET_SMEF64
/** @copydoc nk_bilinear_f32_best */
NUMKONG_API nk_status_t nk_bilinear_f32_smef64(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                               nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f32c_best */
NUMKONG_API nk_status_t nk_bilinear_f32c_smef64(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                                nk_f64c_t *result, void *stream);
/** @copydoc nk_mahalanobis_f32_best */
NUMKONG_API nk_status_t nk_mahalanobis_f32_smef64(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                                  nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f64_best */
NUMKONG_API nk_status_t nk_bilinear_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                               nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f64c_best */
NUMKONG_API nk_status_t nk_bilinear_f64c_smef64(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                                nk_f64c_t *result, void *stream);
/** @copydoc nk_mahalanobis_f64_best */
NUMKONG_API nk_status_t nk_mahalanobis_f64_smef64(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                                  nk_f64_t *result, void *stream);
#endif // NUMKONG_TARGET_SMEF64

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_bilinear_f32_best */
NUMKONG_API nk_status_t nk_bilinear_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                                nk_f64_t *result, void *stream);
/** @copydoc nk_mahalanobis_f32_best */
NUMKONG_API nk_status_t nk_mahalanobis_f32_haswell(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                                   nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f16_best */
NUMKONG_API nk_status_t nk_bilinear_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                                nk_f32_t *result, void *stream);
/** @copydoc nk_mahalanobis_f16_best */
NUMKONG_API nk_status_t nk_mahalanobis_f16_haswell(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                                   nk_f32_t *result, void *stream);
/** @copydoc nk_bilinear_bf16_best */
NUMKONG_API nk_status_t nk_bilinear_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                 nk_size_t n, nk_f32_t *result, void *stream);
/** @copydoc nk_mahalanobis_bf16_best */
NUMKONG_API nk_status_t nk_mahalanobis_bf16_haswell(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                    nk_size_t n, nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_bilinear_f64_best */
NUMKONG_API nk_status_t nk_bilinear_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                                nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f64c_best */
NUMKONG_API nk_status_t nk_bilinear_f64c_skylake(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c,
                                                 nk_size_t n, nk_f64c_t *results, void *stream);
/** @copydoc nk_mahalanobis_f64_best */
NUMKONG_API nk_status_t nk_mahalanobis_f64_skylake(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                                   nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f32_best */
NUMKONG_API nk_status_t nk_bilinear_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                                nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f32c_best */
NUMKONG_API nk_status_t nk_bilinear_f32c_skylake(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c,
                                                 nk_size_t n, nk_f64c_t *results, void *stream);
/** @copydoc nk_mahalanobis_f32_best */
NUMKONG_API nk_status_t nk_mahalanobis_f32_skylake(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                                   nk_f64_t *result, void *stream);
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_GENOA
/** @copydoc nk_bilinear_bf16_best */
NUMKONG_API nk_status_t nk_bilinear_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                               nk_f32_t *result, void *stream);
/** @copydoc nk_bilinear_bf16c_best */
NUMKONG_API nk_status_t nk_bilinear_bf16c_genoa(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c,
                                                nk_size_t n, nk_f32c_t *results, void *stream);
/** @copydoc nk_mahalanobis_bf16_best */
NUMKONG_API nk_status_t nk_mahalanobis_bf16_genoa(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                  nk_size_t n, nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_RVV
/** @copydoc nk_bilinear_f64_best */
NUMKONG_API nk_status_t nk_bilinear_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                            nk_f64_t *result, void *stream);
/** @copydoc nk_mahalanobis_f64_best */
NUMKONG_API nk_status_t nk_mahalanobis_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                               nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f32_best */
NUMKONG_API nk_status_t nk_bilinear_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                            nk_f64_t *result, void *stream);
/** @copydoc nk_mahalanobis_f32_best */
NUMKONG_API nk_status_t nk_mahalanobis_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                               nk_f64_t *result, void *stream);
/** @copydoc nk_bilinear_f16_best */
NUMKONG_API nk_status_t nk_bilinear_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                            nk_f32_t *result, void *stream);
/** @copydoc nk_mahalanobis_f16_best */
NUMKONG_API nk_status_t nk_mahalanobis_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                               nk_f32_t *result, void *stream);
/** @copydoc nk_bilinear_bf16_best */
NUMKONG_API nk_status_t nk_bilinear_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                             nk_f32_t *result, void *stream);
/** @copydoc nk_mahalanobis_bf16_best */
NUMKONG_API nk_status_t nk_mahalanobis_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                                nk_f32_t *result, void *stream);
#endif // NUMKONG_TARGET_RVV

/** Returns the output dtype for bilinear forms. */
NUMKONG_INLINE nk_dtype_t nk_bilinear_output_dtype(nk_dtype_t dtype) {
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

/** Returns the output dtype for Mahalanobis metrics. */
NUMKONG_INLINE nk_dtype_t nk_mahalanobis_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f64_k: return nk_f64_k;
    case nk_f32_k: return nk_f64_k;
    case nk_f16_k: return nk_f32_k;
    case nk_bf16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the error bound of bilinear forms, per @c nk_accumulation_error_bound of their
 *  output. */
NUMKONG_INLINE nk_f64_t nk_bilinear_error_bound(nk_dtype_t dtype) {
    return nk_accumulation_error_bound(nk_bilinear_output_dtype(dtype));
}

/** Returns the error bound of Mahalanobis metrics, per @c nk_accumulation_error_bound of their
 *  output. */
NUMKONG_INLINE nk_f64_t nk_mahalanobis_error_bound(nk_dtype_t dtype) {
    return nk_accumulation_error_bound(nk_mahalanobis_output_dtype(dtype));
}

/**
 *  @brief Finds the curved kernel of @p kind for @p dtype from the best capability in @p capabilities.
 *  @param[out] kernel The kernel, or null when no capability in @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k in header-only builds.
 */
NUMKONG_API nk_status_t nk_curved_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/curved/serial.h"
#include "numkong/curved/neon.h"
#include "numkong/curved/neonbfdot.h"
#include "numkong/curved/smef64.h"
#include "numkong/curved/haswell.h"
#include "numkong/curved/skylake.h"
#include "numkong/curved/genoa.h"
#include "numkong/curved/rvv.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_bilinear_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                             nk_f64_t *result, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(c), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_mahalanobis_f64_best(nk_f64_t const *a, nk_f64_t const *b, nk_f64_t const *c, nk_size_t n,
                                                nk_f64_t *result, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(c), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_bilinear_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                             nk_f64_t *result, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(c), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_mahalanobis_f32_best(nk_f32_t const *a, nk_f32_t const *b, nk_f32_t const *c, nk_size_t n,
                                                nk_f64_t *result, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(c), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_bilinear_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c, nk_size_t n,
                                              nk_f32_t *result, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(c), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_mahalanobis_bf16_best(nk_bf16_t const *a, nk_bf16_t const *b, nk_bf16_t const *c,
                                                 nk_size_t n, nk_f32_t *result, nk_capability_t capabilities,
                                                 void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(c), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_bilinear_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                             nk_f32_t *result, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(c), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_mahalanobis_f16_best(nk_f16_t const *a, nk_f16_t const *b, nk_f16_t const *c, nk_size_t n,
                                                nk_f32_t *result, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(c), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_bilinear_f64c_best(nk_f64c_t const *a, nk_f64c_t const *b, nk_f64c_t const *c, nk_size_t n,
                                              nk_f64c_t *result, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(c), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_bilinear_f32c_best(nk_f32c_t const *a, nk_f32c_t const *b, nk_f32c_t const *c, nk_size_t n,
                                              nk_f64c_t *result, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(c), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_bilinear_bf16c_best(nk_bf16c_t const *a, nk_bf16c_t const *b, nk_bf16c_t const *c,
                                               nk_size_t n, nk_f32c_t *result, nk_capability_t capabilities,
                                               void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(c), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_bilinear_f16c_best(nk_f16c_t const *a, nk_f16c_t const *b, nk_f16c_t const *c, nk_size_t n,
                                              nk_f32c_t *result, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(c), nk_unused_(n), nk_unused_(result), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_curved_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = (nk_kernel_punned_t)NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif // NUMKONG_CURVED_H
