/**
 *  @brief SIMD-accelerated Batched Spatial Distances (Angular & Euclidean).
 *  @file include/numkong/spatials.h
 *  @author Ash Vardanian
 *  @date February 22, 2026
 *
 *  This module provides efficient batched computation of angular and euclidean distances
 *  via a two-pass approach: compute dot products first, then post-process with spatial
 *  distance formulas using pre-computed norms stored in the packed buffer.
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
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, NEON+HALF, NEON+FHM, NEON+BF16, NEON+SDOT, SME, SME+F64
 *  - x86: Haswell, Skylake, Ice Lake, Genoa, Sapphire Rapids (AMX), Sierra Forest
 *  - RISC-V: RVV
 *
 *  @section numerical_stability Numerical Stability
 *
 *  Inherits dot-product precision from nk_dots_packed_* and keeps packed payloads narrow. `f32` batched spatial
 *  kernels now normalize from widened `f64` dots and norms and store `f64` results directly.
 *
 *  @section approach Two-Pass Approach
 *
 *  1. Pack B matrix using nk_dots_pack_* (norms are stored in the packed buffer footer)
 *  2. Compute nk_angulars_packed_* or nk_euclideans_packed_*:
 *     a. Internally calls nk_dots_packed_* to fill result buffer with dot products
 *     b. Post-processes each result cell using angular/euclidean formula with pre-computed norms
 *
 *  @section math Mathematical Foundation
 *
 *  Angular distance:  1 - dot(a,b) / sqrt(sumsq(a) * sumsq(b))
 *  Euclidean distance: sqrt(max(0, sumsq(a) + sumsq(b) - 2*dot(a,b)))
 *
 *  @section packing Packing
 *
 *  Uses the SAME pack functions as dot products (nk_dots_packed_size_*, nk_dots_pack_*).
 *  The packed buffer includes norms appended after the data.
 */

#ifndef NK_SPATIALS_H
#define NK_SPATIALS_H

#include "numkong/dots.h"
#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Computes batched angular distances using a packed second matrix.
 *  @param[in] a Input A matrix in row-major order.
 *  @param[in] b_packed Packed B matrix (produced by nk_dots_pack_*), with norms in footer.
 *  @param[out] result Output matrix (rows x cols) of angular distances.
 *  @param[in] rows Number of rows in A.
 *  @param[in] cols Number of columns in B (packed).
 *  @param[in] depth Shared inner dimension (vector length).
 *  @param[in] a_stride_in_bytes Row stride in bytes for A.
 *  @param[in] r_stride_in_bytes Row stride in bytes for the result matrix.
 */
NK_DYNAMIC void nk_angulars_packed_f32(nk_f32_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                       nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                       nk_size_t r_stride_in_bytes);

/**
 *  @brief Computes symmetric angular distance matrix (Gram-style) for a set of vectors.
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] vectors_count Number of vectors (rows) in the input matrix.
 *  @param[in] depth Dimension of each vector (columns).
 *  @param[in] stride Row stride in bytes for the input matrix.
 *  @param[out] result Output symmetric matrix (vectors_count x vectors_count).
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] row_start Starting row offset of results to compute (for parallelism).
 *  @param[in] row_count Number of rows of results to compute (for parallelism).
 */
NK_DYNAMIC void nk_angulars_symmetric_f32(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);

/**
 *  @brief Computes batched euclidean distances using a packed second matrix.
 *  @param[in] a Input A matrix in row-major order.
 *  @param[in] b_packed Packed B matrix (produced by nk_dots_pack_*), with norms in footer.
 *  @param[out] result Output matrix (rows x cols) of euclidean distances.
 *  @param[in] rows Number of rows in A.
 *  @param[in] cols Number of columns in B (packed).
 *  @param[in] depth Shared inner dimension (vector length).
 *  @param[in] a_stride_in_bytes Row stride in bytes for A.
 *  @param[in] r_stride_in_bytes Row stride in bytes for the result matrix.
 */
NK_DYNAMIC void nk_euclideans_packed_f32(nk_f32_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes);

/**
 *  @brief Computes symmetric euclidean distance matrix (Gram-style) for a set of vectors.
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] vectors_count Number of vectors (rows) in the input matrix.
 *  @param[in] depth Dimension of each vector (columns).
 *  @param[in] stride Row stride in bytes for the input matrix.
 *  @param[out] result Output symmetric matrix (vectors_count x vectors_count).
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] row_start Starting row offset of results to compute (for parallelism).
 *  @param[in] row_count Number of rows of results to compute (for parallelism).
 */
NK_DYNAMIC void nk_euclideans_symmetric_f32(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f32 */
NK_DYNAMIC void nk_angulars_packed_f64(nk_f64_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                       nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                       nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_DYNAMIC void nk_angulars_symmetric_f64(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_DYNAMIC void nk_euclideans_packed_f64(nk_f64_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_DYNAMIC void nk_euclideans_symmetric_f64(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f32 */
NK_DYNAMIC void nk_angulars_packed_f16(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                       nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                       nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_DYNAMIC void nk_angulars_symmetric_f16(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_DYNAMIC void nk_euclideans_packed_f16(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_DYNAMIC void nk_euclideans_symmetric_f16(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f32 */
NK_DYNAMIC void nk_angulars_packed_bf16(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                        nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                        nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_DYNAMIC void nk_angulars_symmetric_bf16(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_DYNAMIC void nk_euclideans_packed_bf16(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                          nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                          nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_DYNAMIC void nk_euclideans_symmetric_bf16(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f32 */
NK_DYNAMIC void nk_angulars_packed_e4m3(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                        nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                        nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_DYNAMIC void nk_angulars_symmetric_e4m3(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_DYNAMIC void nk_euclideans_packed_e4m3(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                          nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                          nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_DYNAMIC void nk_euclideans_symmetric_e4m3(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f32 */
NK_DYNAMIC void nk_angulars_packed_e5m2(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                        nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                        nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_DYNAMIC void nk_angulars_symmetric_e5m2(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_DYNAMIC void nk_euclideans_packed_e5m2(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                          nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                          nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_DYNAMIC void nk_euclideans_symmetric_e5m2(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f32 */
NK_DYNAMIC void nk_angulars_packed_e2m3(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                        nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                        nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_DYNAMIC void nk_angulars_symmetric_e2m3(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_DYNAMIC void nk_euclideans_packed_e2m3(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                          nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                          nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_DYNAMIC void nk_euclideans_symmetric_e2m3(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f32 */
NK_DYNAMIC void nk_angulars_packed_e3m2(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                        nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                        nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_DYNAMIC void nk_angulars_symmetric_e3m2(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_DYNAMIC void nk_euclideans_packed_e3m2(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                          nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                          nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_DYNAMIC void nk_euclideans_symmetric_e3m2(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f32 */
NK_DYNAMIC void nk_angulars_packed_i8(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                      nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                      nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_DYNAMIC void nk_angulars_symmetric_i8(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_DYNAMIC void nk_euclideans_packed_i8(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                        nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                        nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_DYNAMIC void nk_euclideans_symmetric_i8(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f32 */
NK_DYNAMIC void nk_angulars_packed_u8(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                      nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                      nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_DYNAMIC void nk_angulars_symmetric_u8(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_DYNAMIC void nk_euclideans_packed_u8(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                        nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                        nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_DYNAMIC void nk_euclideans_symmetric_u8(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f32 */
NK_DYNAMIC void nk_angulars_packed_i4(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                      nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                      nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_DYNAMIC void nk_angulars_symmetric_i4(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_DYNAMIC void nk_euclideans_packed_i4(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                        nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                        nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_DYNAMIC void nk_euclideans_symmetric_i4(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f32 */
NK_DYNAMIC void nk_angulars_packed_u4(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                      nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                      nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_DYNAMIC void nk_angulars_symmetric_u4(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_DYNAMIC void nk_euclideans_packed_u4(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                        nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                        nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_DYNAMIC void nk_euclideans_symmetric_u4(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f32 */
NK_PUBLIC void nk_angulars_packed_f32_serial(nk_f32_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_PUBLIC void nk_angulars_symmetric_f32_serial(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_PUBLIC void nk_euclideans_packed_f32_serial(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_PUBLIC void nk_euclideans_symmetric_f32_serial(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f64 */
NK_PUBLIC void nk_angulars_packed_f64_serial(nk_f64_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f64 */
NK_PUBLIC void nk_angulars_symmetric_f64_serial(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f64 */
NK_PUBLIC void nk_euclideans_packed_f64_serial(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f64 */
NK_PUBLIC void nk_euclideans_symmetric_f64_serial(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f16 */
NK_PUBLIC void nk_angulars_packed_f16_serial(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f16 */
NK_PUBLIC void nk_angulars_symmetric_f16_serial(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f16 */
NK_PUBLIC void nk_euclideans_packed_f16_serial(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f16 */
NK_PUBLIC void nk_euclideans_symmetric_f16_serial(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_bf16 */
NK_PUBLIC void nk_angulars_packed_bf16_serial(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                              nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                              nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_bf16 */
NK_PUBLIC void nk_angulars_symmetric_bf16_serial(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_bf16 */
NK_PUBLIC void nk_euclideans_packed_bf16_serial(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_bf16 */
NK_PUBLIC void nk_euclideans_symmetric_bf16_serial(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e4m3 */
NK_PUBLIC void nk_angulars_packed_e4m3_serial(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                              nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                              nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e4m3 */
NK_PUBLIC void nk_angulars_symmetric_e4m3_serial(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e4m3 */
NK_PUBLIC void nk_euclideans_packed_e4m3_serial(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e4m3 */
NK_PUBLIC void nk_euclideans_symmetric_e4m3_serial(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e5m2 */
NK_PUBLIC void nk_angulars_packed_e5m2_serial(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                              nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                              nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e5m2 */
NK_PUBLIC void nk_angulars_symmetric_e5m2_serial(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e5m2 */
NK_PUBLIC void nk_euclideans_packed_e5m2_serial(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e5m2 */
NK_PUBLIC void nk_euclideans_symmetric_e5m2_serial(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e2m3 */
NK_PUBLIC void nk_angulars_packed_e2m3_serial(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                              nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                              nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e2m3 */
NK_PUBLIC void nk_angulars_symmetric_e2m3_serial(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e2m3 */
NK_PUBLIC void nk_euclideans_packed_e2m3_serial(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e2m3 */
NK_PUBLIC void nk_euclideans_symmetric_e2m3_serial(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e3m2 */
NK_PUBLIC void nk_angulars_packed_e3m2_serial(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                              nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                              nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e3m2 */
NK_PUBLIC void nk_angulars_symmetric_e3m2_serial(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e3m2 */
NK_PUBLIC void nk_euclideans_packed_e3m2_serial(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e3m2 */
NK_PUBLIC void nk_euclideans_symmetric_e3m2_serial(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_i8 */
NK_PUBLIC void nk_angulars_packed_i8_serial(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i8 */
NK_PUBLIC void nk_angulars_symmetric_i8_serial(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                               nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i8 */
NK_PUBLIC void nk_euclideans_packed_i8_serial(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i8 */
NK_PUBLIC void nk_euclideans_symmetric_i8_serial(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_u8 */
NK_PUBLIC void nk_angulars_packed_u8_serial(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u8 */
NK_PUBLIC void nk_angulars_symmetric_u8_serial(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                               nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u8 */
NK_PUBLIC void nk_euclideans_packed_u8_serial(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u8 */
NK_PUBLIC void nk_euclideans_symmetric_u8_serial(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_i4 */
NK_PUBLIC void nk_angulars_packed_i4_serial(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i4 */
NK_PUBLIC void nk_angulars_symmetric_i4_serial(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                               nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i4 */
NK_PUBLIC void nk_euclideans_packed_i4_serial(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                              nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                              nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i4 */
NK_PUBLIC void nk_euclideans_symmetric_i4_serial(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_u4 */
NK_PUBLIC void nk_angulars_packed_u4_serial(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u4 */
NK_PUBLIC void nk_angulars_symmetric_u4_serial(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                               nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u4 */
NK_PUBLIC void nk_euclideans_packed_u4_serial(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                              nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                              nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u4 */
NK_PUBLIC void nk_euclideans_symmetric_u4_serial(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);

/*  Genoa backends using AVX-512 with BF16 extensions.
 *  These use VDPBF16PS for BF16 dot products.
 *  Packing interleaves elements for SIMD broadcast patterns.
 */
#if NK_TARGET_GENOA
/** @copydoc nk_angulars_packed_bf16 */
NK_PUBLIC void nk_angulars_packed_bf16_genoa(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_bf16 */
NK_PUBLIC void nk_angulars_symmetric_bf16_genoa(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_bf16 */
NK_PUBLIC void nk_euclideans_packed_bf16_genoa(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_bf16 */
NK_PUBLIC void nk_euclideans_symmetric_bf16_genoa(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e4m3 */
NK_PUBLIC void nk_angulars_packed_e4m3_genoa(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e4m3 */
NK_PUBLIC void nk_angulars_symmetric_e4m3_genoa(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e4m3 */
NK_PUBLIC void nk_euclideans_packed_e4m3_genoa(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e4m3 */
NK_PUBLIC void nk_euclideans_symmetric_e4m3_genoa(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e5m2 */
NK_PUBLIC void nk_angulars_packed_e5m2_genoa(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e5m2 */
NK_PUBLIC void nk_angulars_symmetric_e5m2_genoa(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e5m2 */
NK_PUBLIC void nk_euclideans_packed_e5m2_genoa(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e5m2 */
NK_PUBLIC void nk_euclideans_symmetric_e5m2_genoa(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

#endif // NK_TARGET_GENOA

#if NK_TARGET_DIAMOND
/** @copydoc nk_angulars_packed_e4m3 */
NK_PUBLIC void nk_angulars_packed_e4m3_diamond(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e4m3 */
NK_PUBLIC void nk_angulars_symmetric_e4m3_diamond(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e4m3 */
NK_PUBLIC void nk_euclideans_packed_e4m3_diamond(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e4m3 */
NK_PUBLIC void nk_euclideans_symmetric_e4m3_diamond(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e5m2 */
NK_PUBLIC void nk_angulars_packed_e5m2_diamond(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e5m2 */
NK_PUBLIC void nk_angulars_symmetric_e5m2_diamond(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e5m2 */
NK_PUBLIC void nk_euclideans_packed_e5m2_diamond(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e5m2 */
NK_PUBLIC void nk_euclideans_symmetric_e5m2_diamond(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_DIAMOND

/*  Sapphire Rapids backends using Intel AMX (Advanced Matrix Extensions).
 *  AMX provides 8 tile registers (TMM0-TMM7), each holding up to 1KB of data.
 *  Tiles are configured as 16 rows x 64 bytes, enabling (16 x 32) BF16 or (16 x 64) INT8 tiles.
 *  Packing arranges data into AMX-native tile layout with pair interleaving for TDPBF16PS.
 */
#if NK_TARGET_SAPPHIREAMX
/** @copydoc nk_angulars_packed_bf16 */
NK_PUBLIC void nk_angulars_packed_bf16_sapphireamx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_bf16 */
NK_PUBLIC void nk_angulars_symmetric_bf16_sapphireamx(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start,
                                                      nk_size_t row_count);
/** @copydoc nk_euclideans_packed_bf16 */
NK_PUBLIC void nk_euclideans_packed_bf16_sapphireamx(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                     nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_bf16 */
NK_PUBLIC void nk_euclideans_symmetric_bf16_sapphireamx(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count);

/** @copydoc nk_angulars_packed_e4m3 */
NK_PUBLIC void nk_angulars_packed_e4m3_sapphireamx(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e4m3 */
NK_PUBLIC void nk_angulars_symmetric_e4m3_sapphireamx(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start,
                                                      nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e4m3 */
NK_PUBLIC void nk_euclideans_packed_e4m3_sapphireamx(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                     nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e4m3 */
NK_PUBLIC void nk_euclideans_symmetric_e4m3_sapphireamx(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count);

/** @copydoc nk_angulars_packed_e5m2 */
NK_PUBLIC void nk_angulars_packed_e5m2_sapphireamx(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e5m2 */
NK_PUBLIC void nk_angulars_symmetric_e5m2_sapphireamx(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start,
                                                      nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e5m2 */
NK_PUBLIC void nk_euclideans_packed_e5m2_sapphireamx(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                     nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e5m2 */
NK_PUBLIC void nk_euclideans_symmetric_e5m2_sapphireamx(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count);

/** @copydoc nk_angulars_packed_e2m3 */
NK_PUBLIC void nk_angulars_packed_e2m3_sapphireamx(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e2m3 */
NK_PUBLIC void nk_angulars_symmetric_e2m3_sapphireamx(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start,
                                                      nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e2m3 */
NK_PUBLIC void nk_euclideans_packed_e2m3_sapphireamx(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                     nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e2m3 */
NK_PUBLIC void nk_euclideans_symmetric_e2m3_sapphireamx(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count);

/** @copydoc nk_angulars_packed_e3m2 */
NK_PUBLIC void nk_angulars_packed_e3m2_sapphireamx(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e3m2 */
NK_PUBLIC void nk_angulars_symmetric_e3m2_sapphireamx(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start,
                                                      nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e3m2 */
NK_PUBLIC void nk_euclideans_packed_e3m2_sapphireamx(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                     nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e3m2 */
NK_PUBLIC void nk_euclideans_symmetric_e3m2_sapphireamx(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count);

/** @copydoc nk_angulars_packed_i8 */
NK_PUBLIC void nk_angulars_packed_i8_sapphireamx(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i8 */
NK_PUBLIC void nk_angulars_symmetric_i8_sapphireamx(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i8 */
NK_PUBLIC void nk_euclideans_packed_i8_sapphireamx(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i8 */
NK_PUBLIC void nk_euclideans_symmetric_i8_sapphireamx(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_u8 */
NK_PUBLIC void nk_angulars_packed_u8_sapphireamx(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u8 */
NK_PUBLIC void nk_angulars_symmetric_u8_sapphireamx(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u8 */
NK_PUBLIC void nk_euclideans_packed_u8_sapphireamx(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u8 */
NK_PUBLIC void nk_euclideans_symmetric_u8_sapphireamx(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SAPPHIREAMX

/*  Granite Rapids backends using Intel AMX-FP16.
 *  Native FP16 spatial kernels.
 */
#if NK_TARGET_GRANITEAMX
/** @copydoc nk_angulars_packed_f16 */
NK_PUBLIC void nk_angulars_packed_f16_graniteamx(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f16 */
NK_PUBLIC void nk_angulars_symmetric_f16_graniteamx(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f16 */
NK_PUBLIC void nk_euclideans_packed_f16_graniteamx(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f16 */
NK_PUBLIC void nk_euclideans_symmetric_f16_graniteamx(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_GRANITEAMX

/*  ARM SME backends using Scalable Matrix Extension.
 *  SME provides ZA tile registers for outer product operations.
 *  F16/BF16/I8/U8/E4M3 use ZA32 tiles, F32/F64 use ZA64 tiles (FEAT_SME_F64F64).
 */
#if NK_TARGET_SME
/** @copydoc nk_angulars_packed_f16 */
NK_PUBLIC void nk_angulars_packed_f16_sme(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                          nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                          nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f16 */
NK_PUBLIC void nk_angulars_symmetric_f16_sme(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f16 */
NK_PUBLIC void nk_euclideans_packed_f16_sme(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f16 */
NK_PUBLIC void nk_euclideans_symmetric_f16_sme(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                               nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_bf16 */
NK_PUBLIC void nk_angulars_packed_bf16_sme(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_bf16 */
NK_PUBLIC void nk_angulars_symmetric_bf16_sme(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_bf16 */
NK_PUBLIC void nk_euclideans_packed_bf16_sme(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_bf16 */
NK_PUBLIC void nk_euclideans_symmetric_bf16_sme(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e4m3 */
NK_PUBLIC void nk_angulars_packed_e4m3_sme(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e4m3 */
NK_PUBLIC void nk_angulars_symmetric_e4m3_sme(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e4m3 */
NK_PUBLIC void nk_euclideans_packed_e4m3_sme(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e4m3 */
NK_PUBLIC void nk_euclideans_symmetric_e4m3_sme(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e5m2 */
NK_PUBLIC void nk_angulars_packed_e5m2_sme(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e5m2 */
NK_PUBLIC void nk_angulars_symmetric_e5m2_sme(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e5m2 */
NK_PUBLIC void nk_euclideans_packed_e5m2_sme(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e5m2 */
NK_PUBLIC void nk_euclideans_symmetric_e5m2_sme(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e2m3 */
NK_PUBLIC void nk_angulars_packed_e2m3_sme(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e2m3 */
NK_PUBLIC void nk_angulars_symmetric_e2m3_sme(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e2m3 */
NK_PUBLIC void nk_euclideans_packed_e2m3_sme(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e2m3 */
NK_PUBLIC void nk_euclideans_symmetric_e2m3_sme(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e3m2 */
NK_PUBLIC void nk_angulars_packed_e3m2_sme(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e3m2 */
NK_PUBLIC void nk_angulars_symmetric_e3m2_sme(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e3m2 */
NK_PUBLIC void nk_euclideans_packed_e3m2_sme(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e3m2 */
NK_PUBLIC void nk_euclideans_symmetric_e3m2_sme(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_i8 */
NK_PUBLIC void nk_angulars_packed_i8_sme(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i8 */
NK_PUBLIC void nk_angulars_symmetric_i8_sme(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i8 */
NK_PUBLIC void nk_euclideans_packed_i8_sme(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i8 */
NK_PUBLIC void nk_euclideans_symmetric_i8_sme(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_u8 */
NK_PUBLIC void nk_angulars_packed_u8_sme(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u8 */
NK_PUBLIC void nk_angulars_symmetric_u8_sme(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u8 */
NK_PUBLIC void nk_euclideans_packed_u8_sme(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u8 */
NK_PUBLIC void nk_euclideans_symmetric_u8_sme(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_i4 */
NK_PUBLIC void nk_angulars_packed_i4_sme(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i4 */
NK_PUBLIC void nk_angulars_symmetric_i4_sme(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i4 */
NK_PUBLIC void nk_euclideans_packed_i4_sme(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i4 */
NK_PUBLIC void nk_euclideans_symmetric_i4_sme(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_u4 */
NK_PUBLIC void nk_angulars_packed_u4_sme(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u4 */
NK_PUBLIC void nk_angulars_symmetric_u4_sme(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u4 */
NK_PUBLIC void nk_euclideans_packed_u4_sme(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u4 */
NK_PUBLIC void nk_euclideans_symmetric_u4_sme(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SME

/*  ARM SME with FEAT_SME_F64F64 (F32/F64 with F64 accumulators).
 *  Requires Apple M4 or equivalent with F64 outer product support.
 */
#if NK_TARGET_SMEF64
/** @copydoc nk_angulars_packed_f32 */
NK_PUBLIC void nk_angulars_packed_f32_smef64(nk_f32_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_PUBLIC void nk_angulars_symmetric_f32_smef64(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_PUBLIC void nk_euclideans_packed_f32_smef64(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_PUBLIC void nk_euclideans_symmetric_f32_smef64(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f64 */
NK_PUBLIC void nk_angulars_packed_f64_smef64(nk_f64_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f64 */
NK_PUBLIC void nk_angulars_symmetric_f64_smef64(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f64 */
NK_PUBLIC void nk_euclideans_packed_f64_smef64(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f64 */
NK_PUBLIC void nk_euclideans_symmetric_f64_smef64(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SMEF64

/*  Haswell backends using AVX2 (Intel Core 4th gen).
 *  Supports F32/F64 via FMA, F16/BF16/FP8 via software emulation, I8/U8 via VPMADDUBSW+VPADDD.
 */
#if NK_TARGET_HASWELL
/** @copydoc nk_angulars_packed_f32 */
NK_PUBLIC void nk_angulars_packed_f32_haswell(nk_f32_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_PUBLIC void nk_angulars_symmetric_f32_haswell(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_PUBLIC void nk_euclideans_packed_f32_haswell(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_PUBLIC void nk_euclideans_symmetric_f32_haswell(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f64 */
NK_PUBLIC void nk_angulars_packed_f64_haswell(nk_f64_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f64 */
NK_PUBLIC void nk_angulars_symmetric_f64_haswell(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f64 */
NK_PUBLIC void nk_euclideans_packed_f64_haswell(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f64 */
NK_PUBLIC void nk_euclideans_symmetric_f64_haswell(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f16 */
NK_PUBLIC void nk_angulars_packed_f16_haswell(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f16 */
NK_PUBLIC void nk_angulars_symmetric_f16_haswell(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f16 */
NK_PUBLIC void nk_euclideans_packed_f16_haswell(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f16 */
NK_PUBLIC void nk_euclideans_symmetric_f16_haswell(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_bf16 */
NK_PUBLIC void nk_angulars_packed_bf16_haswell(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_bf16 */
NK_PUBLIC void nk_angulars_symmetric_bf16_haswell(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_bf16 */
NK_PUBLIC void nk_euclideans_packed_bf16_haswell(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_bf16 */
NK_PUBLIC void nk_euclideans_symmetric_bf16_haswell(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e4m3 */
NK_PUBLIC void nk_angulars_packed_e4m3_haswell(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e4m3 */
NK_PUBLIC void nk_angulars_symmetric_e4m3_haswell(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e4m3 */
NK_PUBLIC void nk_euclideans_packed_e4m3_haswell(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e4m3 */
NK_PUBLIC void nk_euclideans_symmetric_e4m3_haswell(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e5m2 */
NK_PUBLIC void nk_angulars_packed_e5m2_haswell(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e5m2 */
NK_PUBLIC void nk_angulars_symmetric_e5m2_haswell(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e5m2 */
NK_PUBLIC void nk_euclideans_packed_e5m2_haswell(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e5m2 */
NK_PUBLIC void nk_euclideans_symmetric_e5m2_haswell(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e2m3 */
NK_PUBLIC void nk_angulars_packed_e2m3_haswell(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e2m3 */
NK_PUBLIC void nk_angulars_symmetric_e2m3_haswell(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e2m3 */
NK_PUBLIC void nk_euclideans_packed_e2m3_haswell(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e2m3 */
NK_PUBLIC void nk_euclideans_symmetric_e2m3_haswell(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e3m2 */
NK_PUBLIC void nk_angulars_packed_e3m2_haswell(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e3m2 */
NK_PUBLIC void nk_angulars_symmetric_e3m2_haswell(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e3m2 */
NK_PUBLIC void nk_euclideans_packed_e3m2_haswell(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e3m2 */
NK_PUBLIC void nk_euclideans_symmetric_e3m2_haswell(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_angulars_packed_i8 */
NK_PUBLIC void nk_angulars_packed_i8_haswell(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i8 */
NK_PUBLIC void nk_angulars_symmetric_i8_haswell(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i8 */
NK_PUBLIC void nk_euclideans_packed_i8_haswell(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                               nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                               nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i8 */
NK_PUBLIC void nk_euclideans_symmetric_i8_haswell(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_angulars_packed_u8 */
NK_PUBLIC void nk_angulars_packed_u8_haswell(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u8 */
NK_PUBLIC void nk_angulars_symmetric_u8_haswell(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u8 */
NK_PUBLIC void nk_euclideans_packed_u8_haswell(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                               nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                               nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u8 */
NK_PUBLIC void nk_euclideans_symmetric_u8_haswell(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_HASWELL

/*  Skylake backends using AVX-512 (Intel Core 6th gen+).
 *  Provides 512-bit vectors (16x f32, 8x f64), supporting F32/F64/F16/BF16/FP8 with FMA.
 */
#if NK_TARGET_SKYLAKE
/** @copydoc nk_angulars_packed_f32 */
NK_PUBLIC void nk_angulars_packed_f32_skylake(nk_f32_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_PUBLIC void nk_angulars_symmetric_f32_skylake(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_PUBLIC void nk_euclideans_packed_f32_skylake(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_PUBLIC void nk_euclideans_symmetric_f32_skylake(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f64 */
NK_PUBLIC void nk_angulars_packed_f64_skylake(nk_f64_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f64 */
NK_PUBLIC void nk_angulars_symmetric_f64_skylake(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f64 */
NK_PUBLIC void nk_euclideans_packed_f64_skylake(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f64 */
NK_PUBLIC void nk_euclideans_symmetric_f64_skylake(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f16 */
NK_PUBLIC void nk_angulars_packed_f16_skylake(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f16 */
NK_PUBLIC void nk_angulars_symmetric_f16_skylake(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f16 */
NK_PUBLIC void nk_euclideans_packed_f16_skylake(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f16 */
NK_PUBLIC void nk_euclideans_symmetric_f16_skylake(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_bf16 */
NK_PUBLIC void nk_angulars_packed_bf16_skylake(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_bf16 */
NK_PUBLIC void nk_angulars_symmetric_bf16_skylake(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_bf16 */
NK_PUBLIC void nk_euclideans_packed_bf16_skylake(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_bf16 */
NK_PUBLIC void nk_euclideans_symmetric_bf16_skylake(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e4m3 */
NK_PUBLIC void nk_angulars_packed_e4m3_skylake(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e4m3 */
NK_PUBLIC void nk_angulars_symmetric_e4m3_skylake(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e4m3 */
NK_PUBLIC void nk_euclideans_packed_e4m3_skylake(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e4m3 */
NK_PUBLIC void nk_euclideans_symmetric_e4m3_skylake(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e5m2 */
NK_PUBLIC void nk_angulars_packed_e5m2_skylake(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e5m2 */
NK_PUBLIC void nk_angulars_symmetric_e5m2_skylake(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e5m2 */
NK_PUBLIC void nk_euclideans_packed_e5m2_skylake(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e5m2 */
NK_PUBLIC void nk_euclideans_symmetric_e5m2_skylake(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e2m3 */
NK_PUBLIC void nk_angulars_packed_e2m3_skylake(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e2m3 */
NK_PUBLIC void nk_angulars_symmetric_e2m3_skylake(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e2m3 */
NK_PUBLIC void nk_euclideans_packed_e2m3_skylake(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e2m3 */
NK_PUBLIC void nk_euclideans_symmetric_e2m3_skylake(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e3m2 */
NK_PUBLIC void nk_angulars_packed_e3m2_skylake(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e3m2 */
NK_PUBLIC void nk_angulars_symmetric_e3m2_skylake(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e3m2 */
NK_PUBLIC void nk_euclideans_packed_e3m2_skylake(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e3m2 */
NK_PUBLIC void nk_euclideans_symmetric_e3m2_skylake(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SKYLAKE

/*  Ice Lake backends using AVX-512 with VNNI (Vector Neural Network Instructions).
 *  Adds VPDPBUSD for I8/U8, VPDPWSSD for I4/U4 with efficient dot products.
 */
#if NK_TARGET_ICELAKE
/** @copydoc nk_angulars_packed_i8 */
NK_PUBLIC void nk_angulars_packed_i8_icelake(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i8 */
NK_PUBLIC void nk_angulars_symmetric_i8_icelake(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i8 */
NK_PUBLIC void nk_euclideans_packed_i8_icelake(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                               nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                               nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i8 */
NK_PUBLIC void nk_euclideans_symmetric_i8_icelake(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_u8 */
NK_PUBLIC void nk_angulars_packed_u8_icelake(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u8 */
NK_PUBLIC void nk_angulars_symmetric_u8_icelake(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u8 */
NK_PUBLIC void nk_euclideans_packed_u8_icelake(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                               nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                               nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u8 */
NK_PUBLIC void nk_euclideans_symmetric_u8_icelake(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_i4 */
NK_PUBLIC void nk_angulars_packed_i4_icelake(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i4 */
NK_PUBLIC void nk_angulars_symmetric_i4_icelake(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i4 */
NK_PUBLIC void nk_euclideans_packed_i4_icelake(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i4 */
NK_PUBLIC void nk_euclideans_symmetric_i4_icelake(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_u4 */
NK_PUBLIC void nk_angulars_packed_u4_icelake(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u4 */
NK_PUBLIC void nk_angulars_symmetric_u4_icelake(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u4 */
NK_PUBLIC void nk_euclideans_packed_u4_icelake(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u4 */
NK_PUBLIC void nk_euclideans_symmetric_u4_icelake(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_ALDER
/** @copydoc nk_angulars_packed_i8 */
NK_PUBLIC void nk_angulars_packed_i8_alder(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i8 */
NK_PUBLIC void nk_angulars_symmetric_i8_alder(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i8 */
NK_PUBLIC void nk_euclideans_packed_i8_alder(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i8 */
NK_PUBLIC void nk_euclideans_symmetric_i8_alder(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_angulars_packed_u8 */
NK_PUBLIC void nk_angulars_packed_u8_alder(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u8 */
NK_PUBLIC void nk_angulars_symmetric_u8_alder(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u8 */
NK_PUBLIC void nk_euclideans_packed_u8_alder(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u8 */
NK_PUBLIC void nk_euclideans_symmetric_u8_alder(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_angulars_packed_e2m3 */
NK_PUBLIC void nk_angulars_packed_e2m3_alder(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e2m3 */
NK_PUBLIC void nk_angulars_symmetric_e2m3_alder(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e2m3 */
NK_PUBLIC void nk_euclideans_packed_e2m3_alder(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e2m3 */
NK_PUBLIC void nk_euclideans_symmetric_e2m3_alder(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_ALDER

/*  Sierra backends using AVX10.2 with VMPSADBW.
 *  Optimized for I8/U8 via VMPSADBW (vector multiply-sum of absolute differences).
 */
#if NK_TARGET_SIERRA
/** @copydoc nk_angulars_packed_i8 */
NK_PUBLIC void nk_angulars_packed_i8_sierra(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i8 */
NK_PUBLIC void nk_angulars_symmetric_i8_sierra(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                               nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i8 */
NK_PUBLIC void nk_euclideans_packed_i8_sierra(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i8 */
NK_PUBLIC void nk_euclideans_symmetric_i8_sierra(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_angulars_packed_u8 */
NK_PUBLIC void nk_angulars_packed_u8_sierra(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u8 */
NK_PUBLIC void nk_angulars_symmetric_u8_sierra(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                               nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u8 */
NK_PUBLIC void nk_euclideans_packed_u8_sierra(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u8 */
NK_PUBLIC void nk_euclideans_symmetric_u8_sierra(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_angulars_packed_e2m3 */
NK_PUBLIC void nk_angulars_packed_e2m3_sierra(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                              nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                              nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e2m3 */
NK_PUBLIC void nk_angulars_symmetric_e2m3_sierra(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e2m3 */
NK_PUBLIC void nk_euclideans_packed_e2m3_sierra(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e2m3 */
NK_PUBLIC void nk_euclideans_symmetric_e2m3_sierra(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SIERRA

/*  WASM Relaxed SIMD backends for angular/euclidean distances.
 *  Covers I8/U8/E2M3/BF16/F32/F64 spatial distance operations.
 */
#if NK_TARGET_V128RELAXED
/** @copydoc nk_angulars_packed_i8 */
NK_PUBLIC void nk_angulars_packed_i8_v128relaxed(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i8 */
NK_PUBLIC void nk_angulars_symmetric_i8_v128relaxed(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i8 */
NK_PUBLIC void nk_euclideans_packed_i8_v128relaxed(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i8 */
NK_PUBLIC void nk_euclideans_symmetric_i8_v128relaxed(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_angulars_packed_u8 */
NK_PUBLIC void nk_angulars_packed_u8_v128relaxed(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u8 */
NK_PUBLIC void nk_angulars_symmetric_u8_v128relaxed(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u8 */
NK_PUBLIC void nk_euclideans_packed_u8_v128relaxed(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u8 */
NK_PUBLIC void nk_euclideans_symmetric_u8_v128relaxed(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_angulars_packed_e2m3 */
NK_PUBLIC void nk_angulars_packed_e2m3_v128relaxed(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e2m3 */
NK_PUBLIC void nk_angulars_symmetric_e2m3_v128relaxed(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start,
                                                      nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e2m3 */
NK_PUBLIC void nk_euclideans_packed_e2m3_v128relaxed(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                     nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e2m3 */
NK_PUBLIC void nk_euclideans_symmetric_e2m3_v128relaxed(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count);
/** @copydoc nk_angulars_packed_e4m3 */
NK_PUBLIC void nk_angulars_packed_e4m3_v128relaxed(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e4m3 */
NK_PUBLIC void nk_angulars_symmetric_e4m3_v128relaxed(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start,
                                                      nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e4m3 */
NK_PUBLIC void nk_euclideans_packed_e4m3_v128relaxed(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                     nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e4m3 */
NK_PUBLIC void nk_euclideans_symmetric_e4m3_v128relaxed(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count);
/** @copydoc nk_angulars_packed_e5m2 */
NK_PUBLIC void nk_angulars_packed_e5m2_v128relaxed(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e5m2 */
NK_PUBLIC void nk_angulars_symmetric_e5m2_v128relaxed(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start,
                                                      nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e5m2 */
NK_PUBLIC void nk_euclideans_packed_e5m2_v128relaxed(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                     nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e5m2 */
NK_PUBLIC void nk_euclideans_symmetric_e5m2_v128relaxed(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count);
/** @copydoc nk_angulars_packed_bf16 */
NK_PUBLIC void nk_angulars_packed_bf16_v128relaxed(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_bf16 */
NK_PUBLIC void nk_angulars_symmetric_bf16_v128relaxed(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start,
                                                      nk_size_t row_count);
/** @copydoc nk_euclideans_packed_bf16 */
NK_PUBLIC void nk_euclideans_packed_bf16_v128relaxed(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                     nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_bf16 */
NK_PUBLIC void nk_euclideans_symmetric_bf16_v128relaxed(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count);
/** @copydoc nk_angulars_packed_f32 */
NK_PUBLIC void nk_angulars_packed_f32_v128relaxed(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                  nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                  nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_PUBLIC void nk_angulars_symmetric_f32_v128relaxed(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                     nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_PUBLIC void nk_euclideans_packed_f32_v128relaxed(nk_f32_t const *a, void const *b_packed, nk_f64_t *result,
                                                    nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                    nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_PUBLIC void nk_euclideans_symmetric_f32_v128relaxed(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count);
/** @copydoc nk_angulars_packed_f64 */
NK_PUBLIC void nk_angulars_packed_f64_v128relaxed(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                  nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                  nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f64 */
NK_PUBLIC void nk_angulars_symmetric_f64_v128relaxed(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                     nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                     nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f64 */
NK_PUBLIC void nk_euclideans_packed_f64_v128relaxed(nk_f64_t const *a, void const *b_packed, nk_f64_t *result,
                                                    nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                    nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f64 */
NK_PUBLIC void nk_euclideans_symmetric_f64_v128relaxed(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                       nk_size_t depth, nk_size_t stride, nk_f64_t *result,
                                                       nk_size_t result_stride, nk_size_t row_start,
                                                       nk_size_t row_count);
#endif // NK_TARGET_V128RELAXED

/*  ARM NEON backends (base NEON with F32/F64 support).
 *  Uses FMLA for F32 dots, FMLA (scalar) for F64.
 */
#if NK_TARGET_NEON
/** @copydoc nk_angulars_packed_f32 */
NK_PUBLIC void nk_angulars_packed_f32_neon(nk_f32_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_PUBLIC void nk_angulars_symmetric_f32_neon(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_PUBLIC void nk_euclideans_packed_f32_neon(nk_f32_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_PUBLIC void nk_euclideans_symmetric_f32_neon(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f64 */
NK_PUBLIC void nk_angulars_packed_f64_neon(nk_f64_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f64 */
NK_PUBLIC void nk_angulars_symmetric_f64_neon(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f64 */
NK_PUBLIC void nk_euclideans_packed_f64_neon(nk_f64_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f64 */
NK_PUBLIC void nk_euclideans_symmetric_f64_neon(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_angulars_packed_bf16 */
NK_PUBLIC void nk_angulars_packed_bf16_neon(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_bf16 */
NK_PUBLIC void nk_angulars_symmetric_bf16_neon(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                               nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_bf16 */
NK_PUBLIC void nk_euclideans_packed_bf16_neon(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                              nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                              nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_bf16 */
NK_PUBLIC void nk_euclideans_symmetric_bf16_neon(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_angulars_packed_f16 */
NK_PUBLIC void nk_angulars_packed_f16_neon(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f16 */
NK_PUBLIC void nk_angulars_symmetric_f16_neon(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f16 */
NK_PUBLIC void nk_euclideans_packed_f16_neon(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f16 */
NK_PUBLIC void nk_euclideans_symmetric_f16_neon(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEON

/*  ARM NEON with BF16 dot product (ARMv8.6-A BF16).
 *  Uses BFDOT/BFMMLA for efficient BF16 matrix operations.
 */
#if NK_TARGET_NEONBFDOT
/** @copydoc nk_angulars_packed_bf16 */
NK_PUBLIC void nk_angulars_packed_bf16_neonbfdot(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_bf16 */
NK_PUBLIC void nk_angulars_symmetric_bf16_neonbfdot(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_bf16 */
NK_PUBLIC void nk_euclideans_packed_bf16_neonbfdot(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                   nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_bf16 */
NK_PUBLIC void nk_euclideans_symmetric_bf16_neonbfdot(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start,
                                                      nk_size_t row_count);
#endif // NK_TARGET_NEONBFDOT

/*  ARM NEON with signed/unsigned dot product (ARMv8.2-A DotProd).
 *  Provides SDOT/UDOT for I8/U8 vector dot products.
 */
#if NK_TARGET_NEONSDOT
/** @copydoc nk_angulars_packed_i8 */
NK_PUBLIC void nk_angulars_packed_i8_neonsdot(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i8 */
NK_PUBLIC void nk_angulars_symmetric_i8_neonsdot(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i8 */
NK_PUBLIC void nk_euclideans_packed_i8_neonsdot(nk_i8_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i8 */
NK_PUBLIC void nk_euclideans_symmetric_i8_neonsdot(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_u8 */
NK_PUBLIC void nk_angulars_packed_u8_neonsdot(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u8 */
NK_PUBLIC void nk_angulars_symmetric_u8_neonsdot(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u8 */
NK_PUBLIC void nk_euclideans_packed_u8_neonsdot(nk_u8_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u8 */
NK_PUBLIC void nk_euclideans_symmetric_u8_neonsdot(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_i4 */
NK_PUBLIC void nk_angulars_packed_i4_neonsdot(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                              nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                              nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i4 */
NK_PUBLIC void nk_angulars_symmetric_i4_neonsdot(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i4 */
NK_PUBLIC void nk_euclideans_packed_i4_neonsdot(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i4 */
NK_PUBLIC void nk_euclideans_symmetric_i4_neonsdot(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_u4 */
NK_PUBLIC void nk_angulars_packed_u4_neonsdot(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                              nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                              nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u4 */
NK_PUBLIC void nk_angulars_symmetric_u4_neonsdot(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u4 */
NK_PUBLIC void nk_euclideans_packed_u4_neonsdot(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u4 */
NK_PUBLIC void nk_euclideans_symmetric_u4_neonsdot(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEONSDOT

/*  ARM NEON with FP16 FML (fused multiply-long, ARMv8.2-A FP16FML).
 *  Uses FMLAL/FMLSL for F16 and custom FP8 (E2M3/E3M2) operations.
 */
#if NK_TARGET_NEONFHM
/** @copydoc nk_angulars_packed_f16 */
NK_PUBLIC void nk_angulars_packed_f16_neonfhm(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                              nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                              nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f16 */
NK_PUBLIC void nk_angulars_symmetric_f16_neonfhm(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                 nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                 nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f16 */
NK_PUBLIC void nk_euclideans_packed_f16_neonfhm(nk_f16_t const *a, void const *b_packed, nk_f32_t *result,
                                                nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f16 */
NK_PUBLIC void nk_euclideans_symmetric_f16_neonfhm(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                   nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                   nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e4m3 */
NK_PUBLIC void nk_angulars_packed_e4m3_neonfhm(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e4m3 */
NK_PUBLIC void nk_angulars_symmetric_e4m3_neonfhm(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e4m3 */
NK_PUBLIC void nk_euclideans_packed_e4m3_neonfhm(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e4m3 */
NK_PUBLIC void nk_euclideans_symmetric_e4m3_neonfhm(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e5m2 */
NK_PUBLIC void nk_angulars_packed_e5m2_neonfhm(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e5m2 */
NK_PUBLIC void nk_angulars_symmetric_e5m2_neonfhm(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e5m2 */
NK_PUBLIC void nk_euclideans_packed_e5m2_neonfhm(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e5m2 */
NK_PUBLIC void nk_euclideans_symmetric_e5m2_neonfhm(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

#endif // NK_TARGET_NEONFHM

/*  ARM NEON with FP8 (ARMv9.2-A FP8).
 *  Uses native FP8 dot-product instructions for E4M3/E5M2/E2M3/E3M2 operations.
 */
#if NK_TARGET_NEONFP8
/** @copydoc nk_angulars_packed_e4m3 */
NK_PUBLIC void nk_angulars_packed_e4m3_neonfp8(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e4m3 */
NK_PUBLIC void nk_angulars_symmetric_e4m3_neonfp8(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e4m3 */
NK_PUBLIC void nk_euclideans_packed_e4m3_neonfp8(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e4m3 */
NK_PUBLIC void nk_euclideans_symmetric_e4m3_neonfp8(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e5m2 */
NK_PUBLIC void nk_angulars_packed_e5m2_neonfp8(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e5m2 */
NK_PUBLIC void nk_angulars_symmetric_e5m2_neonfp8(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e5m2 */
NK_PUBLIC void nk_euclideans_packed_e5m2_neonfp8(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e5m2 */
NK_PUBLIC void nk_euclideans_symmetric_e5m2_neonfp8(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e2m3 */
NK_PUBLIC void nk_angulars_packed_e2m3_neonfp8(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e2m3 */
NK_PUBLIC void nk_angulars_symmetric_e2m3_neonfp8(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e2m3 */
NK_PUBLIC void nk_euclideans_packed_e2m3_neonfp8(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e2m3 */
NK_PUBLIC void nk_euclideans_symmetric_e2m3_neonfp8(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e3m2 */
NK_PUBLIC void nk_angulars_packed_e3m2_neonfp8(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                               nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                               nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e3m2 */
NK_PUBLIC void nk_angulars_symmetric_e3m2_neonfp8(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                  nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                  nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e3m2 */
NK_PUBLIC void nk_euclideans_packed_e3m2_neonfp8(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t depth,
                                                 nk_size_t a_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e3m2 */
NK_PUBLIC void nk_euclideans_symmetric_e3m2_neonfp8(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);

#endif // NK_TARGET_NEONFP8

#if NK_TARGET_RVV
/** @copydoc nk_angulars_packed_f32 */
NK_PUBLIC void nk_angulars_packed_f32_rvv(nk_f32_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                          nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                          nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f32 */
NK_PUBLIC void nk_angulars_symmetric_f32_rvv(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f32 */
NK_PUBLIC void nk_euclideans_packed_f32_rvv(nk_f32_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f32 */
NK_PUBLIC void nk_euclideans_symmetric_f32_rvv(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                               nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f64 */
NK_PUBLIC void nk_angulars_packed_f64_rvv(nk_f64_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                          nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                          nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f64 */
NK_PUBLIC void nk_angulars_symmetric_f64_rvv(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f64 */
NK_PUBLIC void nk_euclideans_packed_f64_rvv(nk_f64_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f64 */
NK_PUBLIC void nk_euclideans_symmetric_f64_rvv(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                               nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_f16 */
NK_PUBLIC void nk_angulars_packed_f16_rvv(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                          nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                          nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_f16 */
NK_PUBLIC void nk_angulars_symmetric_f16_rvv(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_f16 */
NK_PUBLIC void nk_euclideans_packed_f16_rvv(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_f16 */
NK_PUBLIC void nk_euclideans_symmetric_f16_rvv(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                               nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_bf16 */
NK_PUBLIC void nk_angulars_packed_bf16_rvv(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_bf16 */
NK_PUBLIC void nk_angulars_symmetric_bf16_rvv(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_bf16 */
NK_PUBLIC void nk_euclideans_packed_bf16_rvv(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_bf16 */
NK_PUBLIC void nk_euclideans_symmetric_bf16_rvv(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e4m3 */
NK_PUBLIC void nk_angulars_packed_e4m3_rvv(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e4m3 */
NK_PUBLIC void nk_angulars_symmetric_e4m3_rvv(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e4m3 */
NK_PUBLIC void nk_euclideans_packed_e4m3_rvv(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e4m3 */
NK_PUBLIC void nk_euclideans_symmetric_e4m3_rvv(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e5m2 */
NK_PUBLIC void nk_angulars_packed_e5m2_rvv(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e5m2 */
NK_PUBLIC void nk_angulars_symmetric_e5m2_rvv(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e5m2 */
NK_PUBLIC void nk_euclideans_packed_e5m2_rvv(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e5m2 */
NK_PUBLIC void nk_euclideans_symmetric_e5m2_rvv(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e2m3 */
NK_PUBLIC void nk_angulars_packed_e2m3_rvv(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e2m3 */
NK_PUBLIC void nk_angulars_symmetric_e2m3_rvv(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e2m3 */
NK_PUBLIC void nk_euclideans_packed_e2m3_rvv(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e2m3 */
NK_PUBLIC void nk_euclideans_symmetric_e2m3_rvv(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_e3m2 */
NK_PUBLIC void nk_angulars_packed_e3m2_rvv(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_e3m2 */
NK_PUBLIC void nk_angulars_symmetric_e3m2_rvv(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_e3m2 */
NK_PUBLIC void nk_euclideans_packed_e3m2_rvv(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_e3m2 */
NK_PUBLIC void nk_euclideans_symmetric_e3m2_rvv(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_i8 */
NK_PUBLIC void nk_angulars_packed_i8_rvv(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_i8 */
NK_PUBLIC void nk_angulars_symmetric_i8_rvv(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_i8 */
NK_PUBLIC void nk_euclideans_packed_i8_rvv(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_i8 */
NK_PUBLIC void nk_euclideans_symmetric_i8_rvv(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);

/** @copydoc nk_angulars_packed_u8 */
NK_PUBLIC void nk_angulars_packed_u8_rvv(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes);
/** @copydoc nk_angulars_symmetric_u8 */
NK_PUBLIC void nk_angulars_symmetric_u8_rvv(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_euclideans_packed_u8 */
NK_PUBLIC void nk_euclideans_packed_u8_rvv(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                           nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                           nk_size_t r_stride_in_bytes);
/** @copydoc nk_euclideans_symmetric_u8 */
NK_PUBLIC void nk_euclideans_symmetric_u8_rvv(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                              nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                              nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_RVV

#if defined(__cplusplus)
} // extern "C"
#endif

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
#include "numkong/spatials/v128relaxed.h"
#include "numkong/spatials/sme.h"
#include "numkong/spatials/smef64.h"
#include "numkong/spatials/powervsx.h"
#include "numkong/spatials/loongsonasx.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_angulars_packed_f64(nk_f64_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                      nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                      nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SMEF64
    nk_angulars_packed_f64_smef64(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEON
    nk_angulars_packed_f64_neon(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_angulars_packed_f64_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_angulars_packed_f64_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_POWERVSX
    nk_angulars_packed_f64_powervsx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_angulars_packed_f64_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_angulars_packed_f64_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_angulars_packed_f64_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_angulars_symmetric_f64(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                         nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SMEF64
    nk_angulars_symmetric_f64_smef64(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_NEON
    nk_angulars_symmetric_f64_neon(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_angulars_symmetric_f64_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_HASWELL
    nk_angulars_symmetric_f64_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_POWERVSX
    nk_angulars_symmetric_f64_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_RVV
    nk_angulars_symmetric_f64_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_angulars_symmetric_f64_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                          row_count);
#else
    nk_angulars_symmetric_f64_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#endif
}
NK_PUBLIC void nk_euclideans_packed_f64(nk_f64_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                        nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                        nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SMEF64
    nk_euclideans_packed_f64_smef64(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEON
    nk_euclideans_packed_f64_neon(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_packed_f64_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_euclideans_packed_f64_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_POWERVSX
    nk_euclideans_packed_f64_powervsx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_euclideans_packed_f64_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_packed_f64_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_euclideans_packed_f64_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_euclideans_symmetric_f64(nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SMEF64
    nk_euclideans_symmetric_f64_smef64(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_NEON
    nk_euclideans_symmetric_f64_neon(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_symmetric_f64_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_HASWELL
    nk_euclideans_symmetric_f64_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_POWERVSX
    nk_euclideans_symmetric_f64_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_RVV
    nk_euclideans_symmetric_f64_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_symmetric_f64_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                            row_count);
#else
    nk_euclideans_symmetric_f64_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#endif
}

NK_PUBLIC void nk_angulars_packed_f32(nk_f32_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                      nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                      nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SMEF64
    nk_angulars_packed_f32_smef64(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEON
    nk_angulars_packed_f32_neon(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_angulars_packed_f32_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_angulars_packed_f32_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_POWERVSX
    nk_angulars_packed_f32_powervsx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_angulars_packed_f32_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_angulars_packed_f32_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_angulars_packed_f32_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_angulars_symmetric_f32(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                         nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SMEF64
    nk_angulars_symmetric_f32_smef64(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_NEON
    nk_angulars_symmetric_f32_neon(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_angulars_symmetric_f32_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_HASWELL
    nk_angulars_symmetric_f32_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_POWERVSX
    nk_angulars_symmetric_f32_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_RVV
    nk_angulars_symmetric_f32_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_angulars_symmetric_f32_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                          row_count);
#else
    nk_angulars_symmetric_f32_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#endif
}
NK_PUBLIC void nk_euclideans_packed_f32(nk_f32_t const *a, void const *b_packed, nk_f64_t *result, nk_size_t rows,
                                        nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                        nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SMEF64
    nk_euclideans_packed_f32_smef64(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEON
    nk_euclideans_packed_f32_neon(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_packed_f32_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_euclideans_packed_f32_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_POWERVSX
    nk_euclideans_packed_f32_powervsx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_euclideans_packed_f32_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_packed_f32_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_euclideans_packed_f32_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_euclideans_symmetric_f32(nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f64_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SMEF64
    nk_euclideans_symmetric_f32_smef64(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_NEON
    nk_euclideans_symmetric_f32_neon(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_symmetric_f32_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_HASWELL
    nk_euclideans_symmetric_f32_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_POWERVSX
    nk_euclideans_symmetric_f32_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_RVV
    nk_euclideans_symmetric_f32_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_symmetric_f32_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                            row_count);
#else
    nk_euclideans_symmetric_f32_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#endif
}

NK_PUBLIC void nk_angulars_packed_f16(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                      nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                      nk_size_t r_stride_in_bytes) {
#if NK_TARGET_GRANITEAMX
    nk_angulars_packed_f16_graniteamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SME
    nk_angulars_packed_f16_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFHM
    nk_angulars_packed_f16_neonfhm(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEON
    nk_angulars_packed_f16_neon(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_angulars_packed_f16_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_angulars_packed_f16_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_POWERVSX
    nk_angulars_packed_f16_powervsx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_angulars_packed_f16_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_angulars_packed_f16_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_angulars_symmetric_f16(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                         nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_GRANITEAMX
    nk_angulars_symmetric_f16_graniteamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_SME
    nk_angulars_symmetric_f16_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONFHM
    nk_angulars_symmetric_f16_neonfhm(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_NEON
    nk_angulars_symmetric_f16_neon(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SKYLAKE
    nk_angulars_symmetric_f16_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_HASWELL
    nk_angulars_symmetric_f16_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_POWERVSX
    nk_angulars_symmetric_f16_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_RVV
    nk_angulars_symmetric_f16_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_angulars_symmetric_f16_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#endif
}
NK_PUBLIC void nk_euclideans_packed_f16(nk_f16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                        nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                        nk_size_t r_stride_in_bytes) {
#if NK_TARGET_GRANITEAMX
    nk_euclideans_packed_f16_graniteamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SME
    nk_euclideans_packed_f16_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFHM
    nk_euclideans_packed_f16_neonfhm(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEON
    nk_euclideans_packed_f16_neon(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_packed_f16_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_euclideans_packed_f16_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_POWERVSX
    nk_euclideans_packed_f16_powervsx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_euclideans_packed_f16_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_euclideans_packed_f16_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_euclideans_symmetric_f16(nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                           nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                           nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_GRANITEAMX
    nk_euclideans_symmetric_f16_graniteamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#elif NK_TARGET_SME
    nk_euclideans_symmetric_f16_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONFHM
    nk_euclideans_symmetric_f16_neonfhm(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_NEON
    nk_euclideans_symmetric_f16_neon(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_symmetric_f16_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_HASWELL
    nk_euclideans_symmetric_f16_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_POWERVSX
    nk_euclideans_symmetric_f16_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_RVV
    nk_euclideans_symmetric_f16_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_euclideans_symmetric_f16_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#endif
}

NK_PUBLIC void nk_angulars_packed_bf16(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                       nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                       nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_angulars_packed_bf16_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONBFDOT
    nk_angulars_packed_bf16_neonbfdot(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_packed_bf16_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_GENOA
    nk_angulars_packed_bf16_genoa(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_angulars_packed_bf16_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_angulars_packed_bf16_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_POWERVSX
    nk_angulars_packed_bf16_powervsx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_angulars_packed_bf16_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_angulars_packed_bf16_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_angulars_packed_bf16_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_angulars_symmetric_bf16(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_angulars_symmetric_bf16_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONBFDOT
    nk_angulars_symmetric_bf16_neonbfdot(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_symmetric_bf16_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#elif NK_TARGET_GENOA
    nk_angulars_symmetric_bf16_genoa(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_SKYLAKE
    nk_angulars_symmetric_bf16_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_HASWELL
    nk_angulars_symmetric_bf16_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_POWERVSX
    nk_angulars_symmetric_bf16_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_RVV
    nk_angulars_symmetric_bf16_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_angulars_symmetric_bf16_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#else
    nk_angulars_symmetric_bf16_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#endif
}
NK_PUBLIC void nk_euclideans_packed_bf16(nk_bf16_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_euclideans_packed_bf16_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONBFDOT
    nk_euclideans_packed_bf16_neonbfdot(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_packed_bf16_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_GENOA
    nk_euclideans_packed_bf16_genoa(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_packed_bf16_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_euclideans_packed_bf16_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_POWERVSX
    nk_euclideans_packed_bf16_powervsx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_euclideans_packed_bf16_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_packed_bf16_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_euclideans_packed_bf16_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_euclideans_symmetric_bf16(nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_euclideans_symmetric_bf16_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_NEONBFDOT
    nk_euclideans_symmetric_bf16_neonbfdot(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_symmetric_bf16_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                             row_count);
#elif NK_TARGET_GENOA
    nk_euclideans_symmetric_bf16_genoa(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_symmetric_bf16_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_HASWELL
    nk_euclideans_symmetric_bf16_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_POWERVSX
    nk_euclideans_symmetric_bf16_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                          row_count);
#elif NK_TARGET_RVV
    nk_euclideans_symmetric_bf16_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_symmetric_bf16_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                             row_count);
#else
    nk_euclideans_symmetric_bf16_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#endif
}

NK_PUBLIC void nk_angulars_packed_e4m3(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                       nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                       nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_angulars_packed_e4m3_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFP8
    nk_angulars_packed_e4m3_neonfp8(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFHM
    nk_angulars_packed_e4m3_neonfhm(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_packed_e4m3_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_DIAMOND
    nk_angulars_packed_e4m3_diamond(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_GENOA
    nk_angulars_packed_e4m3_genoa(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_angulars_packed_e4m3_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_angulars_packed_e4m3_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_angulars_packed_e4m3_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_angulars_packed_e4m3_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_angulars_packed_e4m3_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_angulars_symmetric_e4m3(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_angulars_symmetric_e4m3_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONFP8
    nk_angulars_symmetric_e4m3_neonfp8(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_NEONFHM
    nk_angulars_symmetric_e4m3_neonfhm(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_symmetric_e4m3_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#elif NK_TARGET_DIAMOND
    nk_angulars_symmetric_e4m3_diamond(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_GENOA
    nk_angulars_symmetric_e4m3_genoa(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_SKYLAKE
    nk_angulars_symmetric_e4m3_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_HASWELL
    nk_angulars_symmetric_e4m3_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_RVV
    nk_angulars_symmetric_e4m3_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_angulars_symmetric_e4m3_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#else
    nk_angulars_symmetric_e4m3_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#endif
}
NK_PUBLIC void nk_euclideans_packed_e4m3(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_euclideans_packed_e4m3_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFP8
    nk_euclideans_packed_e4m3_neonfp8(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFHM
    nk_euclideans_packed_e4m3_neonfhm(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_packed_e4m3_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_DIAMOND
    nk_euclideans_packed_e4m3_diamond(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_GENOA
    nk_euclideans_packed_e4m3_genoa(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_packed_e4m3_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_euclideans_packed_e4m3_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_euclideans_packed_e4m3_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_packed_e4m3_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_euclideans_packed_e4m3_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_euclideans_symmetric_e4m3(nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_euclideans_symmetric_e4m3_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_NEONFP8
    nk_euclideans_symmetric_e4m3_neonfp8(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_NEONFHM
    nk_euclideans_symmetric_e4m3_neonfhm(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_symmetric_e4m3_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                             row_count);
#elif NK_TARGET_DIAMOND
    nk_euclideans_symmetric_e4m3_diamond(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_GENOA
    nk_euclideans_symmetric_e4m3_genoa(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_symmetric_e4m3_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_HASWELL
    nk_euclideans_symmetric_e4m3_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_RVV
    nk_euclideans_symmetric_e4m3_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_symmetric_e4m3_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                             row_count);
#else
    nk_euclideans_symmetric_e4m3_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#endif
}

NK_PUBLIC void nk_angulars_packed_e5m2(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                       nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                       nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_angulars_packed_e5m2_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFP8
    nk_angulars_packed_e5m2_neonfp8(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFHM
    nk_angulars_packed_e5m2_neonfhm(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_packed_e5m2_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_DIAMOND
    nk_angulars_packed_e5m2_diamond(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_GENOA
    nk_angulars_packed_e5m2_genoa(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_angulars_packed_e5m2_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_angulars_packed_e5m2_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_angulars_packed_e5m2_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_angulars_packed_e5m2_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_angulars_packed_e5m2_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_angulars_symmetric_e5m2(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_angulars_symmetric_e5m2_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONFP8
    nk_angulars_symmetric_e5m2_neonfp8(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_NEONFHM
    nk_angulars_symmetric_e5m2_neonfhm(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_symmetric_e5m2_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#elif NK_TARGET_DIAMOND
    nk_angulars_symmetric_e5m2_diamond(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_GENOA
    nk_angulars_symmetric_e5m2_genoa(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_SKYLAKE
    nk_angulars_symmetric_e5m2_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_HASWELL
    nk_angulars_symmetric_e5m2_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_RVV
    nk_angulars_symmetric_e5m2_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_angulars_symmetric_e5m2_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#else
    nk_angulars_symmetric_e5m2_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#endif
}
NK_PUBLIC void nk_euclideans_packed_e5m2(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_euclideans_packed_e5m2_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFP8
    nk_euclideans_packed_e5m2_neonfp8(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFHM
    nk_euclideans_packed_e5m2_neonfhm(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_packed_e5m2_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_DIAMOND
    nk_euclideans_packed_e5m2_diamond(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_GENOA
    nk_euclideans_packed_e5m2_genoa(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_packed_e5m2_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_euclideans_packed_e5m2_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_euclideans_packed_e5m2_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_packed_e5m2_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_euclideans_packed_e5m2_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_euclideans_symmetric_e5m2(nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_euclideans_symmetric_e5m2_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_NEONFP8
    nk_euclideans_symmetric_e5m2_neonfp8(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_NEONFHM
    nk_euclideans_symmetric_e5m2_neonfhm(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_symmetric_e5m2_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                             row_count);
#elif NK_TARGET_DIAMOND
    nk_euclideans_symmetric_e5m2_diamond(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_GENOA
    nk_euclideans_symmetric_e5m2_genoa(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_symmetric_e5m2_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_HASWELL
    nk_euclideans_symmetric_e5m2_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_RVV
    nk_euclideans_symmetric_e5m2_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_symmetric_e5m2_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                             row_count);
#else
    nk_euclideans_symmetric_e5m2_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#endif
}

NK_PUBLIC void nk_angulars_packed_e2m3(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                       nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                       nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_angulars_packed_e2m3_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_packed_e2m3_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFP8
    nk_angulars_packed_e2m3_neonfp8(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_angulars_packed_e2m3_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SIERRA
    nk_angulars_packed_e2m3_sierra(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ALDER
    nk_angulars_packed_e2m3_alder(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_angulars_packed_e2m3_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_angulars_packed_e2m3_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_angulars_packed_e2m3_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_angulars_packed_e2m3_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_angulars_symmetric_e2m3(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_angulars_symmetric_e2m3_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_symmetric_e2m3_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#elif NK_TARGET_NEONFP8
    nk_angulars_symmetric_e2m3_neonfp8(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_SKYLAKE
    nk_angulars_symmetric_e2m3_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_SIERRA
    nk_angulars_symmetric_e2m3_sierra(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_ALDER
    nk_angulars_symmetric_e2m3_alder(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_HASWELL
    nk_angulars_symmetric_e2m3_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_RVV
    nk_angulars_symmetric_e2m3_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_angulars_symmetric_e2m3_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#else
    nk_angulars_symmetric_e2m3_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#endif
}
NK_PUBLIC void nk_euclideans_packed_e2m3(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_euclideans_packed_e2m3_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_packed_e2m3_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFP8
    nk_euclideans_packed_e2m3_neonfp8(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_packed_e2m3_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SIERRA
    nk_euclideans_packed_e2m3_sierra(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ALDER
    nk_euclideans_packed_e2m3_alder(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_euclideans_packed_e2m3_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_euclideans_packed_e2m3_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_packed_e2m3_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_euclideans_packed_e2m3_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_euclideans_symmetric_e2m3(nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_euclideans_symmetric_e2m3_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_symmetric_e2m3_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                             row_count);
#elif NK_TARGET_NEONFP8
    nk_euclideans_symmetric_e2m3_neonfp8(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_symmetric_e2m3_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_SIERRA
    nk_euclideans_symmetric_e2m3_sierra(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_ALDER
    nk_euclideans_symmetric_e2m3_alder(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_HASWELL
    nk_euclideans_symmetric_e2m3_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_RVV
    nk_euclideans_symmetric_e2m3_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_symmetric_e2m3_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                             row_count);
#else
    nk_euclideans_symmetric_e2m3_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#endif
}

NK_PUBLIC void nk_angulars_packed_e3m2(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                       nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                       nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_angulars_packed_e3m2_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_packed_e3m2_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFP8
    nk_angulars_packed_e3m2_neonfp8(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_angulars_packed_e3m2_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_angulars_packed_e3m2_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_angulars_packed_e3m2_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_angulars_packed_e3m2_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_angulars_symmetric_e3m2(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_angulars_symmetric_e3m2_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_symmetric_e3m2_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#elif NK_TARGET_NEONFP8
    nk_angulars_symmetric_e3m2_neonfp8(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_SKYLAKE
    nk_angulars_symmetric_e3m2_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_HASWELL
    nk_angulars_symmetric_e3m2_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_RVV
    nk_angulars_symmetric_e3m2_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#else
    nk_angulars_symmetric_e3m2_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#endif
}
NK_PUBLIC void nk_euclideans_packed_e3m2(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                         nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                         nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_euclideans_packed_e3m2_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_packed_e3m2_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONFP8
    nk_euclideans_packed_e3m2_neonfp8(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_packed_e3m2_skylake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_euclideans_packed_e3m2_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_euclideans_packed_e3m2_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_euclideans_packed_e3m2_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_euclideans_symmetric_e3m2(nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                            nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                            nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_euclideans_symmetric_e3m2_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_symmetric_e3m2_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                             row_count);
#elif NK_TARGET_NEONFP8
    nk_euclideans_symmetric_e3m2_neonfp8(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_SKYLAKE
    nk_euclideans_symmetric_e3m2_skylake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_HASWELL
    nk_euclideans_symmetric_e3m2_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_RVV
    nk_euclideans_symmetric_e3m2_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#else
    nk_euclideans_symmetric_e3m2_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#endif
}

NK_PUBLIC void nk_angulars_packed_i8(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                     nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                     nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_angulars_packed_i8_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONSDOT
    nk_angulars_packed_i8_neonsdot(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_packed_i8_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ICELAKE
    nk_angulars_packed_i8_icelake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SIERRA
    nk_angulars_packed_i8_sierra(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ALDER
    nk_angulars_packed_i8_alder(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_angulars_packed_i8_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_POWERVSX
    nk_angulars_packed_i8_powervsx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_angulars_packed_i8_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_angulars_packed_i8_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_angulars_packed_i8_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_angulars_symmetric_i8(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                        nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_angulars_symmetric_i8_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_angulars_symmetric_i8_neonsdot(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_symmetric_i8_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_ICELAKE
    nk_angulars_symmetric_i8_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_SIERRA
    nk_angulars_symmetric_i8_sierra(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ALDER
    nk_angulars_symmetric_i8_alder(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_angulars_symmetric_i8_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_POWERVSX
    nk_angulars_symmetric_i8_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_RVV
    nk_angulars_symmetric_i8_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_angulars_symmetric_i8_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#else
    nk_angulars_symmetric_i8_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}
NK_PUBLIC void nk_euclideans_packed_i8(nk_i8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                       nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                       nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_euclideans_packed_i8_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONSDOT
    nk_euclideans_packed_i8_neonsdot(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_packed_i8_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ICELAKE
    nk_euclideans_packed_i8_icelake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SIERRA
    nk_euclideans_packed_i8_sierra(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ALDER
    nk_euclideans_packed_i8_alder(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_euclideans_packed_i8_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_POWERVSX
    nk_euclideans_packed_i8_powervsx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_euclideans_packed_i8_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_packed_i8_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_euclideans_packed_i8_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_euclideans_symmetric_i8(nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_euclideans_symmetric_i8_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_euclideans_symmetric_i8_neonsdot(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_symmetric_i8_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#elif NK_TARGET_ICELAKE
    nk_euclideans_symmetric_i8_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_SIERRA
    nk_euclideans_symmetric_i8_sierra(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_ALDER
    nk_euclideans_symmetric_i8_alder(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_HASWELL
    nk_euclideans_symmetric_i8_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_POWERVSX
    nk_euclideans_symmetric_i8_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_RVV
    nk_euclideans_symmetric_i8_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_symmetric_i8_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#else
    nk_euclideans_symmetric_i8_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#endif
}

NK_PUBLIC void nk_angulars_packed_u8(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                     nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                     nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_angulars_packed_u8_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONSDOT
    nk_angulars_packed_u8_neonsdot(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_packed_u8_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ICELAKE
    nk_angulars_packed_u8_icelake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SIERRA
    nk_angulars_packed_u8_sierra(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ALDER
    nk_angulars_packed_u8_alder(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_angulars_packed_u8_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_POWERVSX
    nk_angulars_packed_u8_powervsx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_angulars_packed_u8_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_angulars_packed_u8_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_angulars_packed_u8_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_angulars_symmetric_u8(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                        nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_angulars_symmetric_u8_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_angulars_symmetric_u8_neonsdot(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_angulars_symmetric_u8_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#elif NK_TARGET_ICELAKE
    nk_angulars_symmetric_u8_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_SIERRA
    nk_angulars_symmetric_u8_sierra(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ALDER
    nk_angulars_symmetric_u8_alder(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_angulars_symmetric_u8_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_POWERVSX
    nk_angulars_symmetric_u8_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_RVV
    nk_angulars_symmetric_u8_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_angulars_symmetric_u8_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                         row_count);
#else
    nk_angulars_symmetric_u8_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}
NK_PUBLIC void nk_euclideans_packed_u8(nk_u8_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                       nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                       nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_euclideans_packed_u8_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONSDOT
    nk_euclideans_packed_u8_neonsdot(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_packed_u8_sapphireamx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ICELAKE
    nk_euclideans_packed_u8_icelake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_SIERRA
    nk_euclideans_packed_u8_sierra(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ALDER
    nk_euclideans_packed_u8_alder(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_euclideans_packed_u8_haswell(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_POWERVSX
    nk_euclideans_packed_u8_powervsx(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_RVV
    nk_euclideans_packed_u8_rvv(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_packed_u8_v128relaxed(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_euclideans_packed_u8_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_euclideans_symmetric_u8(nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_euclideans_symmetric_u8_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_euclideans_symmetric_u8_neonsdot(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_SAPPHIREAMX
    nk_euclideans_symmetric_u8_sapphireamx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#elif NK_TARGET_ICELAKE
    nk_euclideans_symmetric_u8_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_SIERRA
    nk_euclideans_symmetric_u8_sierra(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_ALDER
    nk_euclideans_symmetric_u8_alder(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#elif NK_TARGET_HASWELL
    nk_euclideans_symmetric_u8_haswell(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#elif NK_TARGET_POWERVSX
    nk_euclideans_symmetric_u8_powervsx(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_RVV
    nk_euclideans_symmetric_u8_rvv(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_euclideans_symmetric_u8_v128relaxed(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                           row_count);
#else
    nk_euclideans_symmetric_u8_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#endif
}

NK_PUBLIC void nk_angulars_packed_i4(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                     nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                     nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_angulars_packed_i4_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONSDOT
    nk_angulars_packed_i4_neonsdot(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ICELAKE
    nk_angulars_packed_i4_icelake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_angulars_packed_i4_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_angulars_symmetric_i4(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                        nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_angulars_symmetric_i4_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_angulars_symmetric_i4_neonsdot(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_ICELAKE
    nk_angulars_symmetric_i4_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#else
    nk_angulars_symmetric_i4_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}
NK_PUBLIC void nk_euclideans_packed_i4(nk_i4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                       nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                       nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_euclideans_packed_i4_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONSDOT
    nk_euclideans_packed_i4_neonsdot(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ICELAKE
    nk_euclideans_packed_i4_icelake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_euclideans_packed_i4_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_euclideans_symmetric_i4(nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_euclideans_symmetric_i4_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_euclideans_symmetric_i4_neonsdot(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_ICELAKE
    nk_euclideans_symmetric_i4_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#else
    nk_euclideans_symmetric_i4_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#endif
}

NK_PUBLIC void nk_angulars_packed_u4(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                     nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                     nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_angulars_packed_u4_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONSDOT
    nk_angulars_packed_u4_neonsdot(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ICELAKE
    nk_angulars_packed_u4_icelake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_angulars_packed_u4_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_angulars_symmetric_u4(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                        nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                        nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_angulars_symmetric_u4_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_angulars_symmetric_u4_neonsdot(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#elif NK_TARGET_ICELAKE
    nk_angulars_symmetric_u4_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
#else
    nk_angulars_symmetric_u4_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#endif
}
NK_PUBLIC void nk_euclideans_packed_u4(nk_u4x2_t const *a, void const *b_packed, nk_f32_t *result, nk_size_t rows,
                                       nk_size_t cols, nk_size_t depth, nk_size_t a_stride_in_bytes,
                                       nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SME
    nk_euclideans_packed_u4_sme(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEONSDOT
    nk_euclideans_packed_u4_neonsdot(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ICELAKE
    nk_euclideans_packed_u4_icelake(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#else
    nk_euclideans_packed_u4_serial(a, b_packed, result, rows, cols, depth, a_stride_in_bytes, r_stride_in_bytes);
#endif
}
NK_PUBLIC void nk_euclideans_symmetric_u4(nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
                                          nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                          nk_size_t row_start, nk_size_t row_count) {
#if NK_TARGET_SME
    nk_euclideans_symmetric_u4_sme(vectors, vectors_count, depth, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEONSDOT
    nk_euclideans_symmetric_u4_neonsdot(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                        row_count);
#elif NK_TARGET_ICELAKE
    nk_euclideans_symmetric_u4_icelake(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                       row_count);
#else
    nk_euclideans_symmetric_u4_serial(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SPATIALS_H
