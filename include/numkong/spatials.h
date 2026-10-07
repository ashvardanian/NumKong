/**
 *  @file include/numkong/spatials.h
 *  @author Ash Vardanian
 *  @date February 22, 2026
 *  @brief SIMD-accelerated batched spatial distances, angular and euclidean.
 *
 *  This module provides efficient batched computation of angular and euclidean distances via a
 *  two-pass approach: compute dot products first, then post-process with spatial distance formulas
 *  using pre-computed norms stored in the packed buffer.
 *
 *  For dtypes:
 *
 *  - f64: 64-bit IEEE floating point numbers → 64-bit floats
 *  - f32: 32-bit IEEE floating point numbers → 64-bit floats
 *  - f16: 16-bit IEEE floating point numbers → 32-bit floats
 *  - bf16: 16-bit brain floating point numbers → 32-bit floats
 *  - e4m3: 8-bit e4m3 floating point numbers → 32-bit floats
 *  - e5m2: 8-bit e5m2 floating point numbers → 32-bit floats
 *  - e2m3: 8-bit e2m3 floating point numbers (MX) → 32-bit floats
 *  - e3m2: 8-bit e3m2 floating point numbers (MX) → 32-bit floats
 *  - e2m1: 4-bit e2m1 floating point numbers (packed pairs) → 32-bit floats
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, NEON+HALF, NEON+FHM, NEON+BF16, NEON+SDOT, SME, SME+F64
 *  - x86: Haswell, Skylake, Ice Lake, Genoa, Sapphire Rapids (AMX), Sierra Forest
 *  - RISC-V: RVV
 *
 *  @section spatials_numerical_stability Numerical Stability
 *
 *  Inherits dot-product precision from nk_dots_packed_* and keeps packed payloads narrow. @c f32
 *  batched spatial kernels now normalize from widened @c f64 dots and norms, storing the @c f64
 *  results directly, without narrowing back to @c f32.
 *
 *  @section spatials_approach Two-Pass Approach
 *
 *  1. Pack B matrix using `nk_dots_pack_*`, which stores the norms in the packed buffer footer.
 *  2. Compute `nk_angulars_packed_*` or `nk_euclideans_packed_*`:
 *     - Internally calls `nk_dots_packed_*` to fill the result buffer with dot products.
 *     - Post-processes each result cell by the angular or euclidean formula with precomputed norms.
 *
 *  @section spatials_math Mathematical Foundation
 *
 *  Angular distance:  1 - dot(a,b) / sqrt(sumsq(a) * sumsq(b))
 *  Euclidean distance: sqrt(max(0, sumsq(a) + sumsq(b) - 2*dot(a,b)))
 *
 *  @section spatials_packing Packing
 *
 *  Uses the same pack functions as dot products, nk_dots_pack_size_*, nk_dots_pack_*. The packed
 *  buffer includes norms appended after the data.
 */

#ifndef NUMKONG_SPATIALS_H
#define NUMKONG_SPATIALS_H

#include "numkong/dots.h"
#include "numkong/types.h"
#include "numkong/capabilities.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Computes batched angular distances using a packed second matrix.
 *  @param[in] a Input A matrix in row-major order.
 *  @param[in] b_packed Packed B with norms from `nk_dots_pack_*_best` on the same @p capabilities.
 *  @param[out] result Output matrix (rows x columns) of angular distances.
 *  @param[in] rows Number of rows in A.
 *  @param[in] columns Number of columns in B (packed).
 *  @param[in] depth Shared inner dimension in dimensions, a multiple of the values per byte.
 *  @param[in] a_stride Row stride in bytes for A.
 *  @param[in] r_stride Row stride in bytes for the result matrix.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_angulars_packed_f32_best(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride,
                                                    nk_capability_t capabilities, void *stream);

/**
 *  @brief Computes symmetric angular distance matrix (Gram-style) for a set of vectors.
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] vectors_count Number of vectors (rows) in the input matrix.
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] stride Row stride in bytes for the input matrix.
 *  @param[out] result Output symmetric matrix of @p vectors_count × @p vectors_count.
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] row_start Starting row offset of results to compute (for parallelism).
 *  @param[in] row_count Number of rows of results to compute (for parallelism).
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_angulars_symmetric_f32_best(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, nk_capability_t capabilities, void *stream);

/**
 *  @brief Computes batched euclidean distances using a packed second matrix.
 *  @param[in] a Input A matrix in row-major order.
 *  @param[in] b_packed Packed B with norms from `nk_dots_pack_*_best` on the same @p capabilities.
 *  @param[out] result Output matrix (rows x columns) of euclidean distances.
 *  @param[in] rows Number of rows in A.
 *  @param[in] columns Number of columns in B (packed).
 *  @param[in] depth Shared inner dimension in dimensions, a multiple of the values per byte.
 *  @param[in] a_stride Row stride in bytes for A.
 *  @param[in] r_stride Row stride in bytes for the result matrix.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_euclideans_packed_f32_best(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride,
                                                      nk_capability_t capabilities, void *stream);

/**
 *  @brief Computes symmetric euclidean distance matrix (Gram-style) for a set of vectors.
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] vectors_count Number of vectors (rows) in the input matrix.
 *  @param[in] depth Counts dimensions, a multiple of the values per byte.
 *  @param[in] stride Row stride in bytes for the input matrix.
 *  @param[out] result Output symmetric matrix of @p vectors_count × @p vectors_count.
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] row_start Starting row offset of results to compute (for parallelism).
 *  @param[in] row_count Number of rows of results to compute (for parallelism).
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_best(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f64_best(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f64_best(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f64_best(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride,
                                                      nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_best(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_best(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride,
                                                    nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_best(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_best(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride,
                                                      nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_best(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_best(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride,
                                                     nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_best(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_best(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_best(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_best(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride,
                                                     nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_best(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_best(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_best(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_best(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride,
                                                     nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_best(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_best(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_best(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_best(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride,
                                                     nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_best(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride,
                                                     nk_capability_t capabilities, void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_best(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride,
                                                      nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_best(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride,
                                                      nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e2m3_best(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e3m2_best(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_best(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_best(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_best(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_best(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream);

/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_best(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_best(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_best(nk_mxfp6e2m3_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count,
                                                             nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e3m2_best(nk_mxfp6e3m2_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count,
                                                             nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_best(nk_mxfp8e4m3_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count,
                                                             nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_best(nk_mxfp8e5m2_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count,
                                                             nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_best(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_best(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride,
                                                       nk_capability_t capabilities, void *stream);

/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_best(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                        nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                        nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_best(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                        nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                        nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                        nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_best(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e3m2_best(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_best(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_best(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_best(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_best(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream);

/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_best(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, nk_capability_t capabilities,
                                                           void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_best(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, nk_capability_t capabilities,
                                                           void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_best(
    nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e3m2_best(
    nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_best(
    nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_best(
    nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, nk_capability_t capabilities, void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_best(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride,
                                                     nk_capability_t capabilities, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_best(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_best(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride,
                                                       nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_best(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_best(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t r_stride, nk_capability_t capabilities,
                                                   void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_best(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_best(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride,
                                                     nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_best(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_best(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t r_stride, nk_capability_t capabilities,
                                                   void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_best(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_best(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride,
                                                     nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_best(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_best(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t r_stride, nk_capability_t capabilities,
                                                   void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_best(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_best(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride,
                                                     nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_best(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_best(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t r_stride, nk_capability_t capabilities,
                                                   void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_best(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_best(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride,
                                                     nk_capability_t capabilities, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_best(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream);

/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f32_serial(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f32_serial(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f32_serial(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_serial(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_f64_best */
NUMKONG_API nk_status_t nk_angulars_packed_f64_serial(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f64_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f64_serial(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f64_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f64_serial(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f64_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_serial(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_serial(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_serial(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_serial(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_serial(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_serial(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_serial(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_serial(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_serial(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_serial(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_serial(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_serial(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_serial(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_serial(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_serial(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_serial(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_serial(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_serial(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_serial(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_serial(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                        nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                        nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                        void *stream);
/** @copydoc nk_angulars_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_serial(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                        nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                        nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                        void *stream);
/** @copydoc nk_angulars_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e2m3_serial(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_angulars_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e3m2_serial(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_angulars_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_serial(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_angulars_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_serial(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_serial(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_serial(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_serial(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_serial(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_serial(nk_mxfp6e2m3_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e3m2_serial(nk_mxfp6e3m2_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_serial(nk_mxfp8e4m3_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_serial(nk_mxfp8e5m2_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_serial(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_serial(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_serial(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          void *stream);
/** @copydoc nk_euclideans_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_serial(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_serial(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e3m2_serial(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_serial(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_serial(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_serial(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_serial(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_serial(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_serial(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_serial(nk_mxfp6e2m3_cref_t const *vectors,
                                                                 nk_size_t vectors_count, nk_size_t depth,
                                                                 nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e3m2_serial(nk_mxfp6e3m2_cref_t const *vectors,
                                                                 nk_size_t vectors_count, nk_size_t depth,
                                                                 nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_serial(nk_mxfp8e4m3_cref_t const *vectors,
                                                                 nk_size_t vectors_count, nk_size_t depth,
                                                                 nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_serial(nk_mxfp8e5m2_cref_t const *vectors,
                                                                 nk_size_t vectors_count, nk_size_t depth,
                                                                 nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_serial(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_serial(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_serial(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_serial(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_serial(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_serial(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_serial(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_serial(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_serial(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_serial(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_serial(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_serial(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_serial(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_serial(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_serial(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_serial(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_serial(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_serial(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_serial(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_serial(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);

/*  Genoa backends using AVX-512 with BF16 extensions.
 *  These use VDPBF16PS for BF16 dot products.
 *  Packing interleaves elements for SIMD broadcast patterns. */
#if NUMKONG_TARGET_GENOA
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_genoa(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_genoa(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_genoa(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_genoa(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_genoa(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_genoa(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_genoa(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_genoa(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_genoa(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_genoa(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_genoa(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_genoa(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_GENOA

#if NUMKONG_TARGET_DIAMOND
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_diamond(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_diamond(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_diamond(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_diamond(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_diamond(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_diamond(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_diamond(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_diamond(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_DIAMOND

/*  Sapphire Rapids backends using Intel AMX (Advanced Matrix Extensions).
 *  AMX provides 8 tile registers (TMM0-TMM7), each holding up to 1KB of data.
 *  Tiles are configured as 16 rows x 64 bytes, enabling (16 x 32) BF16 or (16 x 64) INT8 tiles.
 *  Packing arranges data into AMX-native tile layout with pair interleaving for TDPBF16PS. */
#if NUMKONG_TARGET_SAPPHIREAMX
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_sapphireamx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_sapphireamx(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_sapphireamx(nk_bf16_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_sapphireamx(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_sapphireamx(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_sapphireamx(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_sapphireamx(nk_e4m3_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_sapphireamx(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_sapphireamx(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_sapphireamx(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_sapphireamx(nk_e5m2_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_sapphireamx(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_sapphireamx(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_sapphireamx(nk_e2m1x2_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_sapphireamx(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_angulars_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_sapphireamx(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_angulars_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                                 nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                                 nk_size_t depth, nk_size_t a_stride,
                                                                 nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                                 nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                                 nk_size_t depth, nk_size_t a_stride,
                                                                 nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_sapphireamx(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_sapphireamx(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_sapphireamx(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_sapphireamx(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *vectors,
                                                                    nk_size_t vectors_count, nk_size_t depth,
                                                                    nk_size_t stride, nk_f32_t *result,
                                                                    nk_size_t result_stride, nk_size_t row_start,
                                                                    nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *vectors,
                                                                    nk_size_t vectors_count, nk_size_t depth,
                                                                    nk_size_t stride, nk_f32_t *result,
                                                                    nk_size_t result_stride, nk_size_t row_start,
                                                                    nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_sapphireamx(nk_e2m3_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_sapphireamx(nk_e2m1x2_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_sapphireamx(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                               void *stream);
/** @copydoc nk_euclideans_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_sapphireamx(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                               void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                                   nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                                   nk_size_t depth, nk_size_t a_stride,
                                                                   nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                                   nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                                   nk_size_t depth, nk_size_t a_stride,
                                                                   nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_sapphireamx(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_sapphireamx(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_sapphireamx(nk_nvfp4_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_sapphireamx(nk_mxfp4_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_sapphireamx(nk_mxfp8e4m3_cref_t const *vectors,
                                                                      nk_size_t vectors_count, nk_size_t depth,
                                                                      nk_size_t stride, nk_f32_t *result,
                                                                      nk_size_t result_stride, nk_size_t row_start,
                                                                      nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_sapphireamx(nk_mxfp8e5m2_cref_t const *vectors,
                                                                      nk_size_t vectors_count, nk_size_t depth,
                                                                      nk_size_t stride, nk_f32_t *result,
                                                                      nk_size_t result_stride, nk_size_t row_start,
                                                                      nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_sapphireamx(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_sapphireamx(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_sapphireamx(nk_e3m2_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_sapphireamx(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_sapphireamx(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_sapphireamx(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_sapphireamx(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_sapphireamx(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_sapphireamx(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_sapphireamx(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_sapphireamx(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_sapphireamx(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_SAPPHIREAMX

/*  Granite Rapids backends using Intel AMX-FP16.
 *  Native FP16 spatial kernels. */
#if NUMKONG_TARGET_GRANITEAMX
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_graniteamx(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_graniteamx(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_graniteamx(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_graniteamx(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_graniteamx(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_graniteamx(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_graniteamx(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_graniteamx(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_GRANITEAMX

/*  ARM SME backends using Scalable Matrix Extension.
 *  SME provides ZA tile registers for outer product operations.
 *  F16/BF16/I8/U8/E4M3 use ZA32 tiles, F32/F64 use ZA64 tiles (FEAT_SME_F64F64). */
#if NUMKONG_TARGET_SME
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_sme(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_sme(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_sme(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_sme(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_sme(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_sme(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_sme(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_sme(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_sme(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_sme(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_sme(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_sme(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_sme(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_sme(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_sme(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_sme(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_sme(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_sme(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_sme(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_sme(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_sme(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_sme(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_sme(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_sme(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_sme(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_sme(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_sme(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_sme(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_sme(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_sme(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_sme(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_sme(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t row_start, nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_sme(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_sme(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_sme(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_sme(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t row_start, nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_sme(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_sme(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_sme(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_sme(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_sme(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_sme(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_sme(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_sme(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_sme(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_sme(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_sme(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_sme(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_sme(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_sme(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_sme(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_sme(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                         void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                           void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *vectors,
                                                              nk_size_t vectors_count, nk_size_t depth,
                                                              nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                         void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                           void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *vectors,
                                                              nk_size_t vectors_count, nk_size_t depth,
                                                              nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                         void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                           void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *vectors,
                                                              nk_size_t vectors_count, nk_size_t depth,
                                                              nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                         void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                           void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *vectors,
                                                              nk_size_t vectors_count, nk_size_t depth,
                                                              nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);

#endif // NUMKONG_TARGET_SME

/*  ARM SME with FEAT_SME_F64F64 (F32/F64 with F64 accumulators).
 *  Requires Apple M4 or equivalent with F64 outer product support. */
#if NUMKONG_TARGET_SMEF64
/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f32_smef64(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f32_smef64(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f32_smef64(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_smef64(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_f64_best */
NUMKONG_API nk_status_t nk_angulars_packed_f64_smef64(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f64_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f64_smef64(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f64_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f64_smef64(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f64_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_smef64(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_SMEF64

/*  Haswell backends using AVX2 (Intel Core 4th gen).
 *  Supports F32/F64 via FMA, F16/BF16/FP8 via software emulation, I8/U8 via VPMADDUBSW+VPADDD. */
#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f32_haswell(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f32_haswell(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f32_haswell(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_haswell(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_f64_best */
NUMKONG_API nk_status_t nk_angulars_packed_f64_haswell(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f64_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f64_haswell(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f64_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f64_haswell(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f64_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_haswell(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_haswell(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_haswell(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_haswell(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_haswell(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_haswell(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_haswell(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_haswell(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_haswell(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_haswell(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_haswell(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_haswell(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_haswell(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_haswell(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_haswell(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_haswell(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_haswell(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_haswell(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_haswell(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_haswell(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_haswell(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_haswell(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_haswell(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_haswell(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_haswell(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_haswell(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_haswell(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_haswell(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_haswell(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_haswell(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_haswell(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_haswell(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_haswell(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_haswell(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_haswell(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_haswell(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_haswell(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_haswell(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_haswell(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_haswell(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_haswell(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_haswell(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_haswell(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_haswell(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_haswell(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_HASWELL

/*  Skylake backends using AVX-512 (Intel Core 6th gen+).
 *  Provides 512-bit vectors (16x f32, 8x f64), supporting F32/F64/F16/BF16/FP8 with FMA. */
#if NUMKONG_TARGET_SKYLAKE
/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f32_skylake(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f32_skylake(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f32_skylake(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_skylake(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_f64_best */
NUMKONG_API nk_status_t nk_angulars_packed_f64_skylake(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f64_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f64_skylake(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f64_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f64_skylake(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f64_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_skylake(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_skylake(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_skylake(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_skylake(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_skylake(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_skylake(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_skylake(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_skylake(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_skylake(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_skylake(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_skylake(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_skylake(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_skylake(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_skylake(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_skylake(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_skylake(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_skylake(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_skylake(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_skylake(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_skylake(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                         void *stream);
/** @copydoc nk_angulars_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_skylake(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                         void *stream);
/** @copydoc nk_angulars_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_skylake(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_angulars_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_skylake(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_skylake(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_skylake(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_skylake(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_skylake(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_skylake(nk_mxfp8e4m3_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_skylake(nk_mxfp8e5m2_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_skylake(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_skylake(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_skylake(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                           void *stream);
/** @copydoc nk_euclideans_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_skylake(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                           void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_skylake(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                               void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_skylake(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                               void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_skylake(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_skylake(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_skylake(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_skylake(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_skylake(nk_mxfp8e4m3_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_skylake(nk_mxfp8e5m2_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_skylake(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_skylake(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_skylake(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_skylake(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e2m3_skylake(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_skylake(nk_mxfp6e2m3_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_skylake(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                               void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_skylake(nk_mxfp6e2m3_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e3m2_skylake(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e3m2_skylake(nk_mxfp6e3m2_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e3m2_skylake(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                               void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e3m2_skylake(nk_mxfp6e3m2_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_SKYLAKE

/*  Ice Lake backends using AVX-512 with VNNI (Vector Neural Network Instructions).
 *  Adds VPDPBUSD for I8/U8, VPDPWSSD for I4/U4 with efficient dot products. */
#if NUMKONG_TARGET_ICELAKE
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_icelake(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_icelake(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_icelake(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_icelake(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_icelake(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_icelake(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_icelake(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_icelake(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_icelake(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_icelake(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_icelake(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_icelake(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_icelake(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_icelake(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_icelake(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_icelake(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_ALDER
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_alder(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_alder(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_alder(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_alder(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_alder(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_alder(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_alder(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_alder(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_alder(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_alder(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_alder(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_alder(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_alder(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_alder(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_alder(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_alder(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_ALDER

/*  Sierra backends using AVX10.2 with VMPSADBW.
 *  Optimized for I8/U8 via VMPSADBW (vector multiply-sum of absolute differences). */
#if NUMKONG_TARGET_SIERRA
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_sierra(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_sierra(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_sierra(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_sierra(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_sierra(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_sierra(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_sierra(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_sierra(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_sierra(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_sierra(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_sierra(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_sierra(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_sierra(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_sierra(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_sierra(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_sierra(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_SIERRA

/*  WASM Relaxed SIMD backends for angular/euclidean distances.
 *  Covers I8/U8/E2M3/BF16/F32/F64 spatial distance operations. */
#if NUMKONG_TARGET_V128RELAXED
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_v128relaxed(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_v128relaxed(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_v128relaxed(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_v128relaxed(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_v128relaxed(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_v128relaxed(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_v128relaxed(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_v128relaxed(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_v128relaxed(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_v128relaxed(nk_e2m1x2_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_v128relaxed(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_v128relaxed(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_v128relaxed(nk_e2m3_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_v128relaxed(nk_e2m1x2_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_v128relaxed(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_v128relaxed(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_v128relaxed(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_v128relaxed(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_v128relaxed(nk_e4m3_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_v128relaxed(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_v128relaxed(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_v128relaxed(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_v128relaxed(nk_e5m2_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_v128relaxed(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_v128relaxed(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_v128relaxed(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_v128relaxed(nk_bf16_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_v128relaxed(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_v128relaxed(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_v128relaxed(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_v128relaxed(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_v128relaxed(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_v128relaxed(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_v128relaxed(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_v128relaxed(nk_e3m2_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_v128relaxed(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f32_v128relaxed(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f32_v128relaxed(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f32_v128relaxed(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_v128relaxed(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f64_best */
NUMKONG_API nk_status_t nk_angulars_packed_f64_v128relaxed(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f64_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f64_v128relaxed(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f64_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f64_v128relaxed(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f64_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_v128relaxed(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_v128relaxed(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_v128relaxed(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_v128relaxed(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_v128relaxed(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_v128relaxed(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_v128relaxed(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_v128relaxed(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_v128relaxed(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_V128RELAXED

#if NUMKONG_TARGET_V128
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_v128(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_v128(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_v128(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_v128(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_v128(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_v128(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_v128(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_v128(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_v128(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_v128(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_v128(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_v128(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_V128

/*  ARM NEON backends (base NEON with F32/F64 support).
 *  Uses FMLA for F32 dots, FMLA (scalar) for F64. */
#if NUMKONG_TARGET_NEON
/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f32_neon(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f32_neon(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f32_neon(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_neon(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_f64_best */
NUMKONG_API nk_status_t nk_angulars_packed_f64_neon(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f64_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f64_neon(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f64_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f64_neon(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f64_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_neon(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_neon(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_neon(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_neon(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_neon(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_neon(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_neon(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_neon(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_neon(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e2m3_neon(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_neon(nk_mxfp6e2m3_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_neon(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_neon(nk_mxfp6e2m3_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e3m2_neon(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e3m2_neon(nk_mxfp6e3m2_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e3m2_neon(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e3m2_neon(nk_mxfp6e3m2_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_neon(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_neon(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_neon(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                        nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                        nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                        void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_neon(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_neon(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_neon(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_neon(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                        nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                        nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                        void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_neon(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_neon(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_neon(nk_mxfp8e4m3_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_neon(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_neon(nk_mxfp8e4m3_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_neon(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_neon(nk_mxfp8e5m2_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_neon(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_neon(nk_mxfp8e5m2_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);

#endif // NUMKONG_TARGET_NEON

/*  ARM NEON with BF16 dot product (ARMv8.6-A BF16).
 *  Uses BFDOT/BFMMLA for efficient BF16 matrix operations. */
#if NUMKONG_TARGET_NEONBFDOT
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_neonbfdot(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_neonbfdot(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_neonbfdot(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_neonbfdot(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_NEONBFDOT

/*  ARM NEON with signed/unsigned dot product (ARMv8.2-A DotProd).
 *  Provides SDOT/UDOT for I8/U8 vector dot products. */
#if NUMKONG_TARGET_NEONSDOT
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_neonsdot(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_neonsdot(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_neonsdot(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_neonsdot(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_neonsdot(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_neonsdot(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_neonsdot(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_neonsdot(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_neonsdot(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_neonsdot(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_neonsdot(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_neonsdot(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_neonsdot(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_neonsdot(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_neonsdot(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_neonsdot(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_neonsdot(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_neonsdot(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_neonsdot(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_neonsdot(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_neonsdot(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_neonsdot(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_neonsdot(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_neonsdot(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_neonsdot(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_neonsdot(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_neonsdot(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_neonsdot(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_neonsdot(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_neonsdot(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_neonsdot(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_neonsdot(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_neonsdot(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          void *stream);
/** @copydoc nk_angulars_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_neonsdot(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_neonsdot(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_neonsdot(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e2m3_neonsdot(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_neonsdot(nk_mxfp6e2m3_cref_t const *vectors,
                                                                 nk_size_t vectors_count, nk_size_t depth,
                                                                 nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_neonsdot(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                                nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                                nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                                void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_neonsdot(nk_mxfp6e2m3_cref_t const *vectors,
                                                                   nk_size_t vectors_count, nk_size_t depth,
                                                                   nk_size_t stride, nk_f32_t *result,
                                                                   nk_size_t result_stride, nk_size_t row_start,
                                                                   nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_NEONSDOT

/*  ARM NEON with FP16 FML (fused multiply-long, ARMv8.2-A FP16FML).
 *  Uses FMLAL/FMLSL for F16 and custom FP8 (E2M3/E3M2) operations. */
#if NUMKONG_TARGET_NEONFHM
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_neonfhm(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_neonfhm(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_neonfhm(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_neonfhm(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_neonfhm(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_neonfhm(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_neonfhm(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_neonfhm(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_neonfhm(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_neonfhm(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_neonfhm(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_neonfhm(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_neonfhm(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_neonfhm(nk_mxfp8e4m3_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_neonfhm(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                               void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_neonfhm(nk_mxfp8e4m3_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_neonfhm(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_neonfhm(nk_mxfp8e5m2_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_neonfhm(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                               void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_neonfhm(nk_mxfp8e5m2_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_NEONFHM

/*  ARM NEON with FP8 (ARMv9.2-A FP8).
 *  Uses native FP8 dot-product instructions for E4M3/E5M2/E2M3/E3M2 operations. */
#if NUMKONG_TARGET_NEONFP8
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_neonfp8(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_neonfp8(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_neonfp8(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_neonfp8(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_neonfp8(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_neonfp8(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_neonfp8(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_neonfp8(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_neonfp8(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_neonfp8(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_neonfp8(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_neonfp8(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_neonfp8(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_neonfp8(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_neonfp8(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_neonfp8(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_neonfp8(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_neonfp8(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_neonfp8(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_neonfp8(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_NEONFP8

#if NUMKONG_TARGET_RVV
/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f32_rvv(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f32_rvv(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f32_rvv(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_rvv(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_f64_best */
NUMKONG_API nk_status_t nk_angulars_packed_f64_rvv(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f64_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f64_rvv(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f64_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f64_rvv(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f64_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_rvv(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_rvv(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_rvv(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_rvv(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_rvv(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_rvv(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_rvv(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_rvv(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_rvv(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_rvv(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_rvv(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_rvv(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_rvv(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_rvv(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_rvv(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_rvv(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_rvv(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_rvv(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_rvv(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_rvv(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_rvv(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_rvv(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_rvv(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_rvv(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_rvv(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_rvv(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_rvv(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_rvv(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_rvv(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_rvv(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_rvv(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_rvv(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_rvv(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t row_start, nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_rvv(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                  nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_rvv(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                     nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_rvv(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_rvv(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t row_start, nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_RVV

/*  Power VSX backends, POWER9 and newer. */
#if NUMKONG_TARGET_POWERVSX
/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f32_powervsx(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f32_powervsx(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f32_powervsx(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_powervsx(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f64_best */
NUMKONG_API nk_status_t nk_angulars_packed_f64_powervsx(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f64_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f64_powervsx(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f64_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f64_powervsx(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f64_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_powervsx(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_powervsx(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_powervsx(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_powervsx(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_powervsx(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_powervsx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_powervsx(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_powervsx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_powervsx(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_powervsx(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_powervsx(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_powervsx(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_powervsx(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_powervsx(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_powervsx(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_powervsx(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_powervsx(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_POWERVSX

/*  LoongArch LASX backends, 256-bit vectors. */
#if NUMKONG_TARGET_LOONGSONASX
/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f32_loongsonasx(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f32_loongsonasx(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f32_loongsonasx(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_loongsonasx(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f64_best */
NUMKONG_API nk_status_t nk_angulars_packed_f64_loongsonasx(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f64_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f64_loongsonasx(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f64_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f64_loongsonasx(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f64_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_loongsonasx(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_loongsonasx(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_loongsonasx(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_loongsonasx(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_loongsonasx(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_loongsonasx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_loongsonasx(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_loongsonasx(nk_bf16_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_loongsonasx(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                                 nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_loongsonasx(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_loongsonasx(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_loongsonasx(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_loongsonasx(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_loongsonasx(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_loongsonasx(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_loongsonasx(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_loongsonasx(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_LOONGSONASX

/*  GPU kernels take their CPU counterparts' arguments and return without waiting on the device,
 *  each reading a B packed by its own capability's `nk_dots_pack_*`. Apple GPUs have none.
 *
 *  NVIDIA backends on every device, reusing the baseline dots packs and scalar-FMA tile with
 *  the metric applied in the epilogue. Products and norms accumulate in F64, F64 inputs in
 *  Dot2; outputs are F64 for F64 and F32 inputs, F32 for the rest. */
#if NUMKONG_TARGET_CUDA
/** @copydoc nk_angulars_packed_f64_best */
NUMKONG_API nk_status_t nk_angulars_packed_f64_cuda(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f64_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f64_cuda(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f64_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f64_cuda(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f64_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_cuda(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f32_cuda(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f32_cuda(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f32_cuda(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_cuda(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_cuda(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_cuda(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_cuda(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_cuda(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_cuda(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_cuda(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_cuda(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_cuda(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_cuda(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_cuda(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_cuda(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_cuda(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_cuda(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_cuda(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_cuda(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_cuda(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_cuda(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_cuda(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_cuda(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_cuda(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_cuda(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_cuda(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_cuda(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_cuda(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_cuda(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_cuda(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_cuda(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_cuda(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                          void *stream);
/** @copydoc nk_angulars_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_cuda(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                          void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_cuda(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_cuda(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_cuda(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_cuda(nk_mxfp8e4m3_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_cuda(nk_mxfp8e5m2_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_cuda(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_cuda(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_cuda(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_cuda(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                            void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_cuda(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                            void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_cuda(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_cuda(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_cuda(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_cuda(nk_mxfp8e4m3_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_cuda(nk_mxfp8e5m2_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_cuda(nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_cuda(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_cuda(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_cuda(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_cuda(nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_cuda(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_cuda(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_cuda(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_cuda(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_cuda(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_cuda(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_cuda(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_cuda(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_cuda(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_cuda(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_cuda(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_CUDA

/*  NVIDIA backends from Ampere on, reusing the CUDA dots packs and tiles with the metric applied
 *  in the epilogue, every output F32. */
#if NUMKONG_TARGET_AMPERE
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_ampere(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_ampere(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_ampere(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_ampere(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_ampere(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_ampere(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_ampere(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_ampere(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_ampere(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_ampere(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_ampere(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_ampere(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_ampere(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_ampere(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_ampere(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_ampere(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_ampere(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_ampere(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_ampere(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_ampere(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_ampere(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_ampere(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_ampere(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_ampere(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_ampere(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_ampere(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_ampere(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_ampere(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_ampere(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_ampere(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_ampere(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_ampere(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_ampere(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_ampere(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_ampere(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_ampere(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_ampere(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_ampere(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_ampere(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_ampere(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_ampere(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_ampere(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_ampere(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_ampere(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_AMPERE

/*  NVIDIA Hopper backends, compute capability 9.0, through warpgroup @c wgmma over shared-memory
 *  descriptors. */
#if NUMKONG_TARGET_HOPPER
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_hopper(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_hopper(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_hopper(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_hopper(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_hopper(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_hopper(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_hopper(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_hopper(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_hopper(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_hopper(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_hopper(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_hopper(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_hopper(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_hopper(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_hopper(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_hopper(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_hopper(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_hopper(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_hopper(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_hopper(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_hopper(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_hopper(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_hopper(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_hopper(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_hopper(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_hopper(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_hopper(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_hopper(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_hopper(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_hopper(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_hopper(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_hopper(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_HOPPER

/*  NVIDIA datacenter Blackwell backends, the compute capability 10.x family, through single-thread
 *  @c tcgen05 MMAs into tensor memory. */
#if NUMKONG_TARGET_BLACKWELL
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_blackwell(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_blackwell(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_blackwell(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_blackwell(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_blackwell(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_blackwell(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_blackwell(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_blackwell(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_blackwell(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_blackwell(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_blackwell(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_blackwell(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_blackwell(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_blackwell(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_blackwell(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_blackwell(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_blackwell(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_blackwell(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_blackwell(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_blackwell(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_blackwell(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_blackwell(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_blackwell(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_blackwell(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_blackwell(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_blackwell(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_blackwell(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_blackwell(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                               void *stream);
/** @copydoc nk_angulars_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_blackwell(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                               void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_blackwell(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_blackwell(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_blackwell(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_blackwell(nk_mxfp8e4m3_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_blackwell(nk_mxfp8e5m2_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_blackwell(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_blackwell(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                             void *stream);
/** @copydoc nk_euclideans_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_blackwell(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                             void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_blackwell(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                                 nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                 nk_size_t depth, nk_size_t a_stride,
                                                                 nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_blackwell(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                                 nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                 nk_size_t depth, nk_size_t a_stride,
                                                                 nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_blackwell(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_blackwell(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_blackwell(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_blackwell(nk_mxfp8e4m3_cref_t const *vectors,
                                                                    nk_size_t vectors_count, nk_size_t depth,
                                                                    nk_size_t stride, nk_f32_t *result,
                                                                    nk_size_t result_stride, nk_size_t row_start,
                                                                    nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_blackwell(nk_mxfp8e5m2_cref_t const *vectors,
                                                                    nk_size_t vectors_count, nk_size_t depth,
                                                                    nk_size_t stride, nk_f32_t *result,
                                                                    nk_size_t result_stride, nk_size_t row_start,
                                                                    nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_blackwell(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_blackwell(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_blackwell(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_blackwell(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_blackwell(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_blackwell(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_blackwell(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_blackwell(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_blackwell(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_blackwell(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_blackwell(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_blackwell(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_blackwell(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_blackwell(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_blackwell(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_blackwell(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_BLACKWELL

/*  NVIDIA backends for the compute capability 12.x family, with Float8, Float6 and Float4 products
 *  on the tensor cores natively. */
#if NUMKONG_TARGET_BLACKWELLRTX
/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_blackwellrtx(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_blackwellrtx(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_blackwellrtx(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_blackwellrtx(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                                  nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_blackwellrtx(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_blackwellrtx(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_blackwellrtx(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_blackwellrtx(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                                  nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_blackwellrtx(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_blackwellrtx(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_blackwellrtx(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_blackwellrtx(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                                  nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_blackwellrtx(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_blackwellrtx(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_blackwellrtx(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_blackwellrtx(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                                  nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_blackwellrtx(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_blackwellrtx(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_blackwellrtx(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_blackwellrtx(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                                  nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_BLACKWELLRTX

/*  AMD backends on every device: the CUDA baseline's source, compiled by HIP. */
#if NUMKONG_TARGET_ROCM
/** @copydoc nk_angulars_packed_f64_best */
NUMKONG_API nk_status_t nk_angulars_packed_f64_rocm(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f64_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f64_rocm(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f64_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f64_rocm(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f64_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_rocm(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f32_best */
NUMKONG_API nk_status_t nk_angulars_packed_f32_rocm(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f32_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f32_rocm(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f32_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f32_rocm(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f32_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_rocm(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_rocm(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_rocm(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_rocm(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_rocm(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_rocm(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_rocm(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_rocm(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_rocm(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_rocm(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_rocm(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_rocm(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_rocm(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_rocm(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_rocm(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_rocm(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_rocm(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_rocm(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_rocm(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_rocm(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_rocm(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_rocm(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_rocm(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_rocm(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_rocm(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_rocm(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_rocm(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_rocm(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_rocm(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_rocm(nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_rocm(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_rocm(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_rocm(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_rocm(nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_rocm(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_rocm(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_rocm(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_rocm(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_rocm(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_rocm(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_rocm(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_rocm(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_rocm(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_rocm(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_rocm(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_ROCM

/*  AMD Instinct MI350 backends, gfx950, through its matrix cores. */
#if NUMKONG_TARGET_CDNA4
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_cdna4(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_cdna4(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_cdna4(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_cdna4(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_cdna4(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_cdna4(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_cdna4(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_cdna4(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_cdna4(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_cdna4(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_cdna4(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_cdna4(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_cdna4(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_cdna4(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_cdna4(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_cdna4(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_cdna4(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_cdna4(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_cdna4(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_cdna4(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_cdna4(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_cdna4(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_cdna4(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_cdna4(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_cdna4(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_cdna4(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_cdna4(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_cdna4(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_cdna4(nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                    nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                    nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_cdna4(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_cdna4(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_cdna4(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_cdna4(nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                    nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                    nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_cdna4(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_cdna4(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_cdna4(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_cdna4(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_cdna4(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_cdna4(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_cdna4(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_cdna4(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_cdna4(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_cdna4(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_cdna4(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_CDNA4

/*  AMD Instinct MI400 backends, through its matrix cores. */
#if NUMKONG_TARGET_CDNA5
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_cdna5(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_cdna5(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_cdna5(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_cdna5(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_cdna5(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_cdna5(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_cdna5(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_cdna5(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_cdna5(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_cdna5(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_cdna5(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_cdna5(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_cdna5(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_cdna5(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_cdna5(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_cdna5(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_cdna5(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_cdna5(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_cdna5(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_cdna5(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_cdna5(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_cdna5(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_cdna5(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_cdna5(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_cdna5(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_cdna5(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_cdna5(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_cdna5(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_cdna5(nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                    nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                    nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_cdna5(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_cdna5(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_cdna5(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_cdna5(nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                    nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                    nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_cdna5(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_cdna5(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_cdna5(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_cdna5(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_cdna5(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_cdna5(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_cdna5(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_cdna5(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_cdna5(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_cdna5(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_cdna5(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_CDNA5

#if NUMKONG_TARGET_METAL
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_metal(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_metal(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_metal(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_metal(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_metal(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_metal(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                       nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                       nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_metal(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_metal(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_metal(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_metal(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_metal(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_metal(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_metal(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_metal(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_metal(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_metal(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_metal(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_metal(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_metal(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_metal(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_metal(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_metal(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_metal(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_metal(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_metal(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_metal(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_metal(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_metal(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_metal(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_metal(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_metal(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_metal(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_metal(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_metal(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_metal(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_metal(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_metal(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_metal(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_metal(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_metal(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_metal(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_metal(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_metal(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_metal(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_metal(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_metal(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_metal(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                         void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_metal(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_metal(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_metal(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_metal(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                         void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_metal(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e2m3_metal(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                           void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_metal(nk_mxfp6e2m3_cref_t const *vectors,
                                                              nk_size_t vectors_count, nk_size_t depth,
                                                              nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_metal(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_metal(nk_mxfp6e2m3_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e3m2_metal(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                           void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e3m2_metal(nk_mxfp6e3m2_cref_t const *vectors,
                                                              nk_size_t vectors_count, nk_size_t depth,
                                                              nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e3m2_metal(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e3m2_metal(nk_mxfp6e3m2_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_metal(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                           void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_metal(nk_mxfp8e4m3_cref_t const *vectors,
                                                              nk_size_t vectors_count, nk_size_t depth,
                                                              nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_metal(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_metal(nk_mxfp8e4m3_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_metal(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                           void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_metal(nk_mxfp8e5m2_cref_t const *vectors,
                                                              nk_size_t vectors_count, nk_size_t depth,
                                                              nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_metal(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_metal(nk_mxfp8e5m2_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);

#endif // NUMKONG_TARGET_METAL

#if NUMKONG_TARGET_APPLE9
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_apple9(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_apple9(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_apple9(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_apple9(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_apple9(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_apple9(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_apple9(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_apple9(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_apple9(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_apple9(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_apple9(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_apple9(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_apple9(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_apple9(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_apple9(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_apple9(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_apple9(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_apple9(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_apple9(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_apple9(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_apple9(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_apple9(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_apple9(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_apple9(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_apple9(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_apple9(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_apple9(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_apple9(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_apple9(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                        nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                        nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                        void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_apple9(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_apple9(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_apple9(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_apple9(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                        nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                        nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                        void *stream);
/** @copydoc nk_angulars_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_apple9(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_apple9(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                          void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_apple9(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e2m3_apple9(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_apple9(nk_mxfp6e2m3_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_apple9(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_apple9(nk_mxfp6e2m3_cref_t const *vectors,
                                                                 nk_size_t vectors_count, nk_size_t depth,
                                                                 nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e3m2_apple9(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e3m2_apple9(nk_mxfp6e3m2_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e3m2_apple9(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e3m2_apple9(nk_mxfp6e3m2_cref_t const *vectors,
                                                                 nk_size_t vectors_count, nk_size_t depth,
                                                                 nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_apple9(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_apple9(nk_mxfp8e4m3_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_apple9(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_apple9(nk_mxfp8e4m3_cref_t const *vectors,
                                                                 nk_size_t vectors_count, nk_size_t depth,
                                                                 nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_apple9(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                            void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_apple9(nk_mxfp8e5m2_cref_t const *vectors,
                                                               nk_size_t vectors_count, nk_size_t depth,
                                                               nk_size_t stride, nk_f32_t *result,
                                                               nk_size_t result_stride, nk_size_t row_start,
                                                               nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_apple9(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                              nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                              nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                              void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_apple9(nk_mxfp8e5m2_cref_t const *vectors,
                                                                 nk_size_t vectors_count, nk_size_t depth,
                                                                 nk_size_t stride, nk_f32_t *result,
                                                                 nk_size_t result_stride, nk_size_t row_start,
                                                                 nk_size_t row_count, void *stream);

#endif // NUMKONG_TARGET_APPLE9

#if NUMKONG_TARGET_APPLE10
/** @copydoc nk_angulars_packed_i8_best */
NUMKONG_API nk_status_t nk_angulars_packed_i8_apple10(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i8_apple10(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i8_apple10(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_apple10(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u8_best */
NUMKONG_API nk_status_t nk_angulars_packed_u8_apple10(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u8_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u8_apple10(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u8_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u8_apple10(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u8_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_apple10(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_f16_best */
NUMKONG_API nk_status_t nk_angulars_packed_f16_apple10(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_f16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_f16_apple10(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_f16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_f16_apple10(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_f16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_apple10(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_bf16_best */
NUMKONG_API nk_status_t nk_angulars_packed_bf16_apple10(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_apple10(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_packed_bf16_apple10(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_bf16_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_apple10(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e4m3_apple10(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_apple10(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_apple10(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_apple10(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e5m2_apple10(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_apple10(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_apple10(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_apple10(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_e3m2_apple10(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_apple10(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_apple10(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_apple10(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m3_apple10(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_apple10(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_apple10(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_apple10(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_packed_e2m1_apple10(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_apple10(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_apple10(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_e2m1_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_apple10(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_apple10(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                         void *stream);
/** @copydoc nk_angulars_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_apple10(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_apple10(nk_nvfp4_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                           void *stream);
/** @copydoc nk_euclideans_symmetric_nvfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_apple10(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_apple10(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                         void *stream);
/** @copydoc nk_angulars_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_apple10(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_apple10(nk_mxfp4_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                           void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_apple10(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e2m3_apple10(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_apple10(nk_mxfp6e2m3_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_apple10(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                               void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e2m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_apple10(nk_mxfp6e2m3_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e3m2_apple10(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_angulars_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e3m2_apple10(nk_mxfp6e3m2_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e3m2_apple10(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                               void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp6e3m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e3m2_apple10(nk_mxfp6e3m2_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_apple10(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_apple10(nk_mxfp8e4m3_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_apple10(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                               void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e4m3_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_apple10(nk_mxfp8e4m3_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_apple10(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                             nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                             nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                             void *stream);
/** @copydoc nk_angulars_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_apple10(nk_mxfp8e5m2_cref_t const *vectors,
                                                                nk_size_t vectors_count, nk_size_t depth,
                                                                nk_size_t stride, nk_f32_t *result,
                                                                nk_size_t result_stride, nk_size_t row_start,
                                                                nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_apple10(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                               nk_f32_t *result, nk_size_t rows, nk_size_t columns,
                                                               nk_size_t depth, nk_size_t a_stride, nk_size_t r_stride,
                                                               void *stream);
/** @copydoc nk_euclideans_symmetric_mxfp8e5m2_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_apple10(nk_mxfp8e5m2_cref_t const *vectors,
                                                                  nk_size_t vectors_count, nk_size_t depth,
                                                                  nk_size_t stride, nk_f32_t *result,
                                                                  nk_size_t result_stride, nk_size_t row_start,
                                                                  nk_size_t row_count, void *stream);

/** @copydoc nk_angulars_packed_i4_best */
NUMKONG_API nk_status_t nk_angulars_packed_i4_apple10(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_i4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_i4_apple10(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_i4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_i4_apple10(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_i4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_apple10(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);
/** @copydoc nk_angulars_packed_u4_best */
NUMKONG_API nk_status_t nk_angulars_packed_u4_apple10(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_angulars_symmetric_u4_best */
NUMKONG_API nk_status_t nk_angulars_symmetric_u4_apple10(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, void *stream);
/** @copydoc nk_euclideans_packed_u4_best */
NUMKONG_API nk_status_t nk_euclideans_packed_u4_apple10(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t r_stride, void *stream);
/** @copydoc nk_euclideans_symmetric_u4_best */
NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_apple10(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, void *stream);

#endif // NUMKONG_TARGET_APPLE10

/**
 *  @brief Finds the spatials kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k when header-only.
 */
NUMKONG_API nk_status_t nk_spatials_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                                nk_kernel_punned_t *kernel, nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/spatials/serial.h"
#include "numkong/spatials/neon.h"
#include "numkong/spatials/neonfhm.h"
#include "numkong/spatials/neonfp8.h"
#include "numkong/spatials/neonbfdot.h"
#include "numkong/spatials/neonsdot.h"
#include "numkong/spatials/haswell.h"
#include "numkong/spatials/skylake.h"
#include "numkong/spatials/genoa.h"
#include "numkong/spatials/diamond.h"
#include "numkong/spatials/icelake.h"
#include "numkong/spatials/alder.h"
#include "numkong/spatials/sierra.h"
#include "numkong/spatials/sapphireamx.h"
#include "numkong/spatials/graniteamx.h"
#include "numkong/spatials/rvv.h"
#include "numkong/spatials/v128.h"
#include "numkong/spatials/v128relaxed.h"
#include "numkong/spatials/sme.h"
#include "numkong/spatials/smef64.h"
#include "numkong/spatials/powervsx.h"
#include "numkong/spatials/loongsonasx.h"
#include "numkong/spatials/cuda.cuh"
#include "numkong/spatials/rocm.cuh"
#include "numkong/spatials/ampere.cuh"
#include "numkong/spatials/hopper.cuh"
#include "numkong/spatials/blackwell.cuh"
#include "numkong/spatials/blackwellrtx.cuh"
#include "numkong/spatials/cdna4.cuh"
#include "numkong/spatials/cdna5.cuh"
#include "numkong/spatials/metal.h"
#include "numkong/spatials/apple9.h"
#include "numkong/spatials/apple10.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_angulars_packed_f64_best(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride,
                                                    nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f64_best(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, nk_capability_t capabilities,
                                                       void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_f64_best(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_best(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_f32_best(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride,
                                                    nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f32_best(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, nk_capability_t capabilities,
                                                       void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_f32_best(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_best(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_bf16_best(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_best(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_bf16_best(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride,
                                                       nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_best(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_f16_best(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                    nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                    nk_size_t a_stride, nk_size_t c_stride,
                                                    nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f16_best(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count, nk_capability_t capabilities,
                                                       void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_f16_best(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_best(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_e5m2_best(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_best(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_best(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride,
                                                       nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_best(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_e4m3_best(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_best(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_best(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride,
                                                       nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_best(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_e3m2_best(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_best(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_best(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride,
                                                       nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_best(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_e2m3_best(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_best(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_best(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride,
                                                       nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_best(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_e2m1_best(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_best(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_best(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride, nk_size_t c_stride,
                                                      nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_best(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                          nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e2m3_best(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                          nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e3m2_best(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                          nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_best(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                          nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                          nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_best(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_best(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_best(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                         nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                         nk_size_t result_stride, nk_size_t row_start,
                                                         nk_size_t row_count, nk_capability_t capabilities,
                                                         void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_best(nk_mxfp8e4m3_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count,
                                                             nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_best(nk_mxfp6e2m3_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count,
                                                             nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e3m2_best(nk_mxfp6e3m2_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count,
                                                             nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_best(nk_mxfp8e5m2_cref_t const *vectors,
                                                             nk_size_t vectors_count, nk_size_t depth, nk_size_t stride,
                                                             nk_f32_t *result, nk_size_t result_stride,
                                                             nk_size_t row_start, nk_size_t row_count,
                                                             nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_best(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride,
                                                       nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_best(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride,
                                                        nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_best(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride, nk_size_t c_stride,
                                                        nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_best(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                            nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_best(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                            nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e3m2_best(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                            nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_best(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                            nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                            nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_best(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, nk_capability_t capabilities,
                                                          void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_best(nk_nvfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, nk_capability_t capabilities,
                                                           void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_best(nk_mxfp4_cref_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                           nk_size_t result_stride, nk_size_t row_start,
                                                           nk_size_t row_count, nk_capability_t capabilities,
                                                           void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_best(
    nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_best(
    nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e3m2_best(
    nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_best(
    nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_i8_best(nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_i8_best(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_i8_best(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_best(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_i4_best(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_i4_best(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_i4_best(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_best(nk_i4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_u8_best(nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                   nk_size_t columns, nk_size_t depth, nk_size_t a_stride,
                                                   nk_size_t c_stride, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_u8_best(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_u8_best(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_best(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_u4_best(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_u4_best(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_u4_best(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(rows), nk_unused_(columns), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_best(nk_u4x2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, nk_capability_t capabilities,
                                                        void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_spatials_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                                nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif // NUMKONG_SPATIALS_H
