/**
 *  @brief SIMD-accelerated Batched Set Distances.
 *  @file include/numkong/sets.h
 *  @author Ash Vardanian
 *
 *  This module provides efficient batched computation of Hamming and Jaccard distances
 *  between large collections of sets. Unlike the single-vector `set.h` module, this module
 *  is optimized for matrix-style operations where you compute distances between:
 *
 *  - All pairs of rows in a query matrix Q against rows in values matrix V
 *  - All pairs within a single values matrix V (symmetric kernel)
 *
 *  For dtypes:
 *
 *  - u1: 1-bit binary (packed octets) → u32 Hamming / f32 Jaccard
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, SME+BI32
 *  - x86: Haswell, Ice Lake
 *
 *  @section numerical_stability Numerical Stability
 *
 *  Hamming u1: u32 popcount accumulator. Overflows at n_bits > 2^32.
 *  Jaccard u1: u32 intersection count, f32 division. Popcount values above 2^24 lose
 *  precision in f32 cast. Streaming variants use u64 accumulation internally.
 *
 *  @section use_cases Use Cases
 *
 *  - Binary similarity search: Find nearest neighbors in Hamming/Jaccard space
 *  - MinHash/SimHash: Compute Jaccard similarity for document fingerprints
 *  - Locality-sensitive hashing (LSH): Build similarity graphs
 *  - Binary neural network inference: Compute distances for BNN outputs
 *
 *  @section math Mathematical Background
 *
 *  Hamming distance: Number of positions where bits differ
 *    hamming(a, b) = popcount(a XOR b)
 *
 *  Jaccard distance: 1 minus the Jaccard similarity
 *    jaccard(a, b) = 1 - |a ∩ b| / |a ∪ b|
 *                  = 1 - popcount(a AND b) / popcount(a OR b)
 *
 *  For Jaccard, we use the identity: |a ∪ b| = |a| + |b| - |a ∩ b|
 *  This allows precomputing |a| and |b| (population counts) during packing.
 */

#ifndef NK_SETS_H
#define NK_SETS_H

#include "numkong/types.h"
#include "numkong/dots.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Compute Hamming distances between V rows and packed Q rows.
 *  @param[in] v Input values matrix
 *  @param[in] q_packed Packed queries matrix
 *  @param[out] result Row-major results matrix
 *  @param[in] rows Number of rows in the results matrix
 *  @param[in] cols Number of columns in the results matrix
 *  @param[in] d Number of dimensions (depth) per vector
 *  @param[in] v_stride_in_bytes Byte stride between rows of A
 *  @param[in] r_stride_in_bytes Byte stride between rows of C
 */
NK_DYNAMIC void nk_hammings_packed_u1(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result, nk_size_t rows,
                                      nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                      nk_size_t r_stride_in_bytes);

/**
 *  @brief Computes C = A × Aᵀ symmetric Gram matrix of Hamming distances.
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] n_vectors Number of vectors (rows) in the input matrix.
 *  @param[in] d Dimension of each vector (columns).
 *  @param[in] stride Row stride in bytes for the input matrix.
 *  @param[out] result Output symmetric matrix (n_vectors × n_vectors).
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] row_start Starting row offset of results to compute (needed for parallelism).
 *  @param[in] row_count Number of rows of results to compute (needed for parallelism).
 */
NK_DYNAMIC void nk_hammings_symmetric_u1(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d, nk_size_t stride,
                                         nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                         nk_size_t row_count);

/**
 *  @brief Compute Jaccard distances between V rows and packed Q rows.
 *  @param[in] v Input values matrix
 *  @param[in] q_packed Packed queries matrix (with norms)
 *  @param[out] result Row-major f32 results matrix
 *  @param[in] rows Number of rows in the results matrix
 *  @param[in] cols Number of columns in the results matrix
 *  @param[in] d Number of dimensions (depth) per vector
 *  @param[in] v_stride_in_bytes Byte stride between rows of A
 *  @param[in] r_stride_in_bytes Byte stride between rows of C
 */
NK_DYNAMIC void nk_jaccards_packed_u1(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result, nk_size_t rows,
                                      nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                      nk_size_t r_stride_in_bytes);

/**
 *  @brief Computes C = f(A, Aᵀ) symmetric Gram matrix of Jaccard distances.
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] n_vectors Number of vectors (rows).
 *  @param[in] d Dimension of each vector (columns).
 *  @param[in] stride Row stride in bytes.
 *  @param[out] result Output symmetric f32 matrix (n_vectors × n_vectors).
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] row_start Starting row offset (for parallelism).
 *  @param[in] row_count Number of rows to compute (for parallelism).
 */
NK_DYNAMIC void nk_jaccards_symmetric_u1(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d, nk_size_t stride,
                                         nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                         nk_size_t row_count);

/** @copydoc nk_hammings_packed_u1 */
NK_PUBLIC void nk_hammings_packed_u1_serial(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_hammings_symmetric_u1 */
NK_PUBLIC void nk_hammings_symmetric_u1_serial(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d,
                                               nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_jaccards_packed_u1 */
NK_PUBLIC void nk_jaccards_packed_u1_serial(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result, nk_size_t rows,
                                            nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                            nk_size_t r_stride_in_bytes);
/** @copydoc nk_jaccards_symmetric_u1 */
NK_PUBLIC void nk_jaccards_symmetric_u1_serial(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d,
                                               nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                               nk_size_t row_start, nk_size_t row_count);

/*  ARM SME with BI32 (binary integer outer products).
 *  Uses BMOPA/BMOPS for efficient popcount-based set distances.
 */
#if NK_TARGET_SMEBI32
/** @copydoc nk_hammings_packed_u1 */
NK_PUBLIC void nk_hammings_packed_u1_smebi32(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_hammings_symmetric_u1 */
NK_PUBLIC void nk_hammings_symmetric_u1_smebi32(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d,
                                                nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_jaccards_packed_u1 */
NK_PUBLIC void nk_jaccards_packed_u1_smebi32(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_jaccards_symmetric_u1 */
NK_PUBLIC void nk_jaccards_symmetric_u1_smebi32(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_SMEBI32

/*  Haswell backends using AVX2 (Intel Core 4th gen).
 *  Supports F32/F64 via FMA, F16/BF16/FP8 via software emulation, I8/U8 via VPMADDUBSW+VPADDD.
 */
#if NK_TARGET_HASWELL
/** @copydoc nk_hammings_packed_u1 */
NK_PUBLIC void nk_hammings_packed_u1_haswell(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_hammings_symmetric_u1 */
NK_PUBLIC void nk_hammings_symmetric_u1_haswell(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d,
                                                nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_jaccards_packed_u1 */
NK_PUBLIC void nk_jaccards_packed_u1_haswell(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_jaccards_symmetric_u1 */
NK_PUBLIC void nk_jaccards_symmetric_u1_haswell(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_HASWELL

/*  Ice Lake backends using AVX-512 with VNNI (Vector Neural Network Instructions).
 *  Adds VPDPBUSD for I8/U8, VPDPWSSD for I4/U4 with efficient dot products.
 */
#if NK_TARGET_ICELAKE
/** @copydoc nk_hammings_packed_u1 */
NK_PUBLIC void nk_hammings_packed_u1_icelake(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_hammings_symmetric_u1 */
NK_PUBLIC void nk_hammings_symmetric_u1_icelake(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d,
                                                nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_jaccards_packed_u1 */
NK_PUBLIC void nk_jaccards_packed_u1_icelake(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result, nk_size_t rows,
                                             nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                             nk_size_t r_stride_in_bytes);
/** @copydoc nk_jaccards_symmetric_u1 */
NK_PUBLIC void nk_jaccards_symmetric_u1_icelake(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d,
                                                nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_ICELAKE

/*  ARM NEON backends (base NEON with F32/F64 support).
 *  Uses FMLA for F32 dots, FMLA (scalar) for F64.
 */
#if NK_TARGET_NEON
/** @copydoc nk_hammings_packed_u1 */
NK_PUBLIC void nk_hammings_packed_u1_neon(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result, nk_size_t rows,
                                          nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                          nk_size_t r_stride_in_bytes);
/** @copydoc nk_hammings_symmetric_u1 */
NK_PUBLIC void nk_hammings_symmetric_u1_neon(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d,
                                             nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_jaccards_packed_u1 */
NK_PUBLIC void nk_jaccards_packed_u1_neon(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result, nk_size_t rows,
                                          nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                          nk_size_t r_stride_in_bytes);
/** @copydoc nk_jaccards_symmetric_u1 */
NK_PUBLIC void nk_jaccards_symmetric_u1_neon(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d,
                                             nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                             nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_NEON

/*  WASM Relaxed SIMD backends using wasm_i8x16_popcnt for popcount-based set distances.
 */
#if NK_TARGET_V128RELAXED
/** @copydoc nk_hammings_packed_u1 */
NK_PUBLIC void nk_hammings_packed_u1_v128relaxed(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                 nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_hammings_symmetric_u1 */
NK_PUBLIC void nk_hammings_symmetric_u1_v128relaxed(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d,
                                                    nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);
/** @copydoc nk_jaccards_packed_u1 */
NK_PUBLIC void nk_jaccards_packed_u1_v128relaxed(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result,
                                                 nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                 nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes);
/** @copydoc nk_jaccards_symmetric_u1 */
NK_PUBLIC void nk_jaccards_symmetric_u1_v128relaxed(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d,
                                                    nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                    nk_size_t row_start, nk_size_t row_count);
#endif // NK_TARGET_V128RELAXED

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/sets/serial.h"
#include "numkong/sets/neon.h"
#include "numkong/sets/icelake.h"
#include "numkong/sets/haswell.h"
#include "numkong/sets/smebi32.h"
#include "numkong/sets/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_hammings_packed_u1(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result, nk_size_t rows,
                                     nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                     nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SMEBI32
    nk_hammings_packed_u1_smebi32(v, q_packed, result, rows, cols, d, v_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEON
    nk_hammings_packed_u1_neon(v, q_packed, result, rows, cols, d, v_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ICELAKE
    nk_hammings_packed_u1_icelake(v, q_packed, result, rows, cols, d, v_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_hammings_packed_u1_haswell(v, q_packed, result, rows, cols, d, v_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_hammings_packed_u1_v128relaxed(v, q_packed, result, rows, cols, d, v_stride_in_bytes, r_stride_in_bytes);
#else
    nk_hammings_packed_u1_serial(v, q_packed, result, rows, cols, d, v_stride_in_bytes, r_stride_in_bytes);
#endif
}

NK_PUBLIC void nk_hammings_symmetric_u1(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d, nk_size_t stride,
                                        nk_u32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                        nk_size_t row_count) {
#if NK_TARGET_SMEBI32
    nk_hammings_symmetric_u1_smebi32(vectors, n_vectors, d, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEON
    nk_hammings_symmetric_u1_neon(vectors, n_vectors, d, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ICELAKE
    nk_hammings_symmetric_u1_icelake(vectors, n_vectors, d, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_hammings_symmetric_u1_haswell(vectors, n_vectors, d, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_hammings_symmetric_u1_v128relaxed(vectors, n_vectors, d, stride, result, result_stride, row_start, row_count);
#else
    nk_hammings_symmetric_u1_serial(vectors, n_vectors, d, stride, result, result_stride, row_start, row_count);
#endif
}

NK_PUBLIC void nk_jaccards_packed_u1(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result, nk_size_t rows,
                                     nk_size_t cols, nk_size_t d, nk_size_t v_stride_in_bytes,
                                     nk_size_t r_stride_in_bytes) {
#if NK_TARGET_SMEBI32
    nk_jaccards_packed_u1_smebi32(v, q_packed, result, rows, cols, d, v_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_NEON
    nk_jaccards_packed_u1_neon(v, q_packed, result, rows, cols, d, v_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_ICELAKE
    nk_jaccards_packed_u1_icelake(v, q_packed, result, rows, cols, d, v_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_HASWELL
    nk_jaccards_packed_u1_haswell(v, q_packed, result, rows, cols, d, v_stride_in_bytes, r_stride_in_bytes);
#elif NK_TARGET_V128RELAXED
    nk_jaccards_packed_u1_v128relaxed(v, q_packed, result, rows, cols, d, v_stride_in_bytes, r_stride_in_bytes);
#else
    nk_jaccards_packed_u1_serial(v, q_packed, result, rows, cols, d, v_stride_in_bytes, r_stride_in_bytes);
#endif
}

NK_PUBLIC void nk_jaccards_symmetric_u1(nk_u1x8_t const *vectors, nk_size_t n_vectors, nk_size_t d, nk_size_t stride,
                                        nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start,
                                        nk_size_t row_count) {
#if NK_TARGET_SMEBI32
    nk_jaccards_symmetric_u1_smebi32(vectors, n_vectors, d, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_NEON
    nk_jaccards_symmetric_u1_neon(vectors, n_vectors, d, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_ICELAKE
    nk_jaccards_symmetric_u1_icelake(vectors, n_vectors, d, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_HASWELL
    nk_jaccards_symmetric_u1_haswell(vectors, n_vectors, d, stride, result, result_stride, row_start, row_count);
#elif NK_TARGET_V128RELAXED
    nk_jaccards_symmetric_u1_v128relaxed(vectors, n_vectors, d, stride, result, result_stride, row_start, row_count);
#else
    nk_jaccards_symmetric_u1_serial(vectors, n_vectors, d, stride, result, result_stride, row_start, row_count);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_SETS_H
