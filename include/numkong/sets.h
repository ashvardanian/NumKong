/**
 *  @file include/numkong/sets.h
 *  @author Ash Vardanian
 *  @date January 22, 2026
 *  @brief SIMD-accelerated batched set distances.
 *
 *  This module provides efficient batched computation of Hamming and Jaccard distances between
 *  large collections of sets. Unlike the single-vector `set.h` module, this module is optimized for
 *  matrix-style operations where you compute distances between:
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
 *  @section sets_numerical_stability Numerical Stability
 *
 *  - Hamming u1: u32 popcount accumulator, overflowing at n_bits > 2^32.
 *  - Jaccard u1: u32 intersection count and f32 division, so popcounts above 2^24 lose precision.
 *  - Streaming variants: u64 accumulation internally.
 *
 *  @section use_cases Use Cases
 *
 *  - Binary similarity search: Find nearest neighbors in Hamming/Jaccard space
 *  - MinHash/SimHash: Compute Jaccard similarity for document fingerprints
 *  - Locality-sensitive hashing (LSH): Build similarity graphs
 *  - Binary neural network inference: Compute distances for BNN outputs
 *
 *  @section sets_math Mathematical Background
 *
 *  Hamming distance counts the positions where bits differ, hamming(a, b) = popcount(a XOR b).
 *  Jaccard distance is 1 minus the Jaccard similarity:
 *
 *  @verbatim
 *  jaccard(a, b) = 1 - |a ∩ b| / |a ∪ b|
 *                = 1 - popcount(a AND b) / popcount(a OR b)
 *  @endverbatim
 *
 *  For Jaccard, we use the identity |a ∪ b| = |a| + |b| - |a ∩ b|, which allows precomputing |a|
 *  and |b|, population counts, during packing.
 */

#ifndef NUMKONG_SETS_H
#define NUMKONG_SETS_H

#include "numkong/types.h"
#include "numkong/capabilities.h"
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
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[in] v_stride_in_bytes Byte stride between rows of A
 *  @param[in] r_stride_in_bytes Byte stride between rows of C
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_hammings_packed_u1_best(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                   nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                   nk_capability_t capabilities, void *stream);

/**
 *  @brief Computes C = A × Aᵀ symmetric Gram matrix of Hamming distances.
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] vectors_count Number of vectors (rows) in the input matrix.
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[in] stride Row stride in bytes for the input matrix.
 *  @param[out] result Output symmetric matrix of @p vectors_count × @p vectors_count.
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] row_start Starting row offset of results to compute (needed for parallelism).
 *  @param[in] row_count Number of rows of results to compute (needed for parallelism).
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_hammings_symmetric_u1_best(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                      nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream);

/**
 *  @brief Compute Jaccard distances between V rows and packed Q rows.
 *  @param[in] v Input values matrix
 *  @param[in] q_packed Packed queries matrix (with norms)
 *  @param[out] result Row-major f32 results matrix
 *  @param[in] rows Number of rows in the results matrix
 *  @param[in] cols Number of columns in the results matrix
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[in] v_stride_in_bytes Byte stride between rows of A
 *  @param[in] r_stride_in_bytes Byte stride between rows of C
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_jaccards_packed_u1_best(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                   nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                   nk_capability_t capabilities, void *stream);

/**
 *  @brief Computes C = f(A, Aᵀ) symmetric Gram matrix of Jaccard distances.
 *  @param[in] vectors Input matrix of row vectors in row-major order.
 *  @param[in] vectors_count Number of vectors (rows).
 *  @param[in] d Counts dimensions, a multiple of the values per byte.
 *  @param[in] stride Row stride in bytes.
 *  @param[out] result Output symmetric f32 matrix of @p vectors_count × @p vectors_count.
 *  @param[in] result_stride Row stride in bytes for the result matrix.
 *  @param[in] row_start Starting row offset (for parallelism).
 *  @param[in] row_count Number of rows to compute (for parallelism).
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 */
NUMKONG_API nk_status_t nk_jaccards_symmetric_u1_best(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream);

/** @copydoc nk_hammings_packed_u1_best */
NUMKONG_API nk_status_t nk_hammings_packed_u1_serial(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result,
                                                     nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                     nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                     void *stream);
/** @copydoc nk_hammings_symmetric_u1_best */
NUMKONG_API nk_status_t nk_hammings_symmetric_u1_serial(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                        nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                        nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_jaccards_packed_u1_best */
NUMKONG_API nk_status_t nk_jaccards_packed_u1_serial(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result,
                                                     nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                     nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                     void *stream);
/** @copydoc nk_jaccards_symmetric_u1_best */
NUMKONG_API nk_status_t nk_jaccards_symmetric_u1_serial(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                        nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                        nk_size_t row_start, nk_size_t row_count, void *stream);

/*  ARM SME with BI32 (binary integer outer products).
 *  Uses BMOPA/BMOPS for efficient popcount-based set distances. */
#if NUMKONG_TARGET_SMEBI32
/** @copydoc nk_hammings_packed_u1_best */
NUMKONG_API nk_status_t nk_hammings_packed_u1_smebi32(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result,
                                                      nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                      nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                      void *stream);
/** @copydoc nk_hammings_symmetric_u1_best */
NUMKONG_API nk_status_t nk_hammings_symmetric_u1_smebi32(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                         nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                         nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_jaccards_packed_u1_best */
NUMKONG_API nk_status_t nk_jaccards_packed_u1_smebi32(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                      nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                      void *stream);
/** @copydoc nk_jaccards_symmetric_u1_best */
NUMKONG_API nk_status_t nk_jaccards_symmetric_u1_smebi32(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                         nk_size_t row_start, nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_SMEBI32

/*  Haswell backends using AVX2 (Intel Core 4th gen).
 *  Supports F32/F64 via FMA, F16/BF16/FP8 via software emulation, I8/U8 via VPMADDUBSW+VPADDD. */
#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_hammings_packed_u1_best */
NUMKONG_API nk_status_t nk_hammings_packed_u1_haswell(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result,
                                                      nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                      nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                      void *stream);
/** @copydoc nk_hammings_symmetric_u1_best */
NUMKONG_API nk_status_t nk_hammings_symmetric_u1_haswell(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                         nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                         nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_jaccards_packed_u1_best */
NUMKONG_API nk_status_t nk_jaccards_packed_u1_haswell(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                      nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                      void *stream);
/** @copydoc nk_jaccards_symmetric_u1_best */
NUMKONG_API nk_status_t nk_jaccards_symmetric_u1_haswell(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                         nk_size_t row_start, nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_HASWELL

/*  Ice Lake backends using AVX-512 with VNNI (Vector Neural Network Instructions).
 *  Adds VPDPBUSD for I8/U8, VPDPWSSD for I4/U4 with efficient dot products. */
#if NUMKONG_TARGET_ICELAKE
/** @copydoc nk_hammings_packed_u1_best */
NUMKONG_API nk_status_t nk_hammings_packed_u1_icelake(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result,
                                                      nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                      nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                      void *stream);
/** @copydoc nk_hammings_symmetric_u1_best */
NUMKONG_API nk_status_t nk_hammings_symmetric_u1_icelake(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                         nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                         nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_jaccards_packed_u1_best */
NUMKONG_API nk_status_t nk_jaccards_packed_u1_icelake(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result,
                                                      nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                      nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                      void *stream);
/** @copydoc nk_jaccards_symmetric_u1_best */
NUMKONG_API nk_status_t nk_jaccards_symmetric_u1_icelake(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                         nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                         nk_size_t row_start, nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_ICELAKE

/*  ARM NEON backends (base NEON with F32/F64 support).
 *  Uses FMLA for F32 dots, FMLA (scalar) for F64. */
#if NUMKONG_TARGET_NEON
/** @copydoc nk_hammings_packed_u1_best */
NUMKONG_API nk_status_t nk_hammings_packed_u1_neon(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                   nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                   void *stream);
/** @copydoc nk_hammings_symmetric_u1_best */
NUMKONG_API nk_status_t nk_hammings_symmetric_u1_neon(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                      nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_jaccards_packed_u1_best */
NUMKONG_API nk_status_t nk_jaccards_packed_u1_neon(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                   nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                   void *stream);
/** @copydoc nk_jaccards_symmetric_u1_best */
NUMKONG_API nk_status_t nk_jaccards_symmetric_u1_neon(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_NEON

/*  WASM Relaxed SIMD backends using wasm_i8x16_popcnt for popcount-based set distances. */
#if NUMKONG_TARGET_V128
/** @copydoc nk_hammings_packed_u1_best */
NUMKONG_API nk_status_t nk_hammings_packed_u1_v128(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                   nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                   void *stream);
/** @copydoc nk_hammings_symmetric_u1_best */
NUMKONG_API nk_status_t nk_hammings_symmetric_u1_v128(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                      nk_size_t stride, nk_u32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
/** @copydoc nk_jaccards_packed_u1_best */
NUMKONG_API nk_status_t nk_jaccards_packed_u1_v128(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result,
                                                   nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                   nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                   void *stream);
/** @copydoc nk_jaccards_symmetric_u1_best */
NUMKONG_API nk_status_t nk_jaccards_symmetric_u1_v128(nk_u1x8_t const *vectors, nk_size_t vectors_count, nk_size_t d,
                                                      nk_size_t stride, nk_f32_t *result, nk_size_t result_stride,
                                                      nk_size_t row_start, nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_V128

/*  IBM Power VSX backends using VPOPCNTD for popcount-based set distances. */
#if NUMKONG_TARGET_POWERVSX
/** @copydoc nk_hammings_packed_u1_best */
NUMKONG_API nk_status_t nk_hammings_packed_u1_powervsx(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result,
                                                       nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                       nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                       void *stream);
/** @copydoc nk_hammings_symmetric_u1_best */
NUMKONG_API nk_status_t nk_hammings_symmetric_u1_powervsx(nk_u1x8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t d, nk_size_t stride, nk_u32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
/** @copydoc nk_jaccards_packed_u1_best */
NUMKONG_API nk_status_t nk_jaccards_packed_u1_powervsx(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result,
                                                       nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                       nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                       void *stream);
/** @copydoc nk_jaccards_symmetric_u1_best */
NUMKONG_API nk_status_t nk_jaccards_symmetric_u1_powervsx(nk_u1x8_t const *vectors, nk_size_t vectors_count,
                                                          nk_size_t d, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_POWERVSX

/*  Loongson LASX backends using 256-bit SIMD with XVPCNT.W for popcount-based set distances. */
#if NUMKONG_TARGET_LOONGSONASX
/** @copydoc nk_hammings_packed_u1_best */
NUMKONG_API nk_status_t nk_hammings_packed_u1_loongsonasx(nk_u1x8_t const *v, void const *q_packed, nk_u32_t *result,
                                                          nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                          nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                          void *stream);
/** @copydoc nk_hammings_symmetric_u1_best */
NUMKONG_API nk_status_t nk_hammings_symmetric_u1_loongsonasx(nk_u1x8_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t d, nk_size_t stride, nk_u32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
/** @copydoc nk_jaccards_packed_u1_best */
NUMKONG_API nk_status_t nk_jaccards_packed_u1_loongsonasx(nk_u1x8_t const *v, void const *q_packed, nk_f32_t *result,
                                                          nk_size_t rows, nk_size_t cols, nk_size_t d,
                                                          nk_size_t v_stride_in_bytes, nk_size_t r_stride_in_bytes,
                                                          void *stream);
/** @copydoc nk_jaccards_symmetric_u1_best */
NUMKONG_API nk_status_t nk_jaccards_symmetric_u1_loongsonasx(nk_u1x8_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t d, nk_size_t stride, nk_f32_t *result,
                                                             nk_size_t result_stride, nk_size_t row_start,
                                                             nk_size_t row_count, void *stream);
#endif // NUMKONG_TARGET_LOONGSONASX

/**
 *  @brief Finds the batched set distance kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k in header-only builds.
 */
NUMKONG_API nk_status_t nk_sets_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                            nk_kernel_punned_t *kernel, nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/sets/serial.h"
#include "numkong/sets/neon.h"
#include "numkong/sets/icelake.h"
#include "numkong/sets/haswell.h"
#include "numkong/sets/smebi32.h"
#include "numkong/sets/v128.h"
#include "numkong/sets/powervsx.h"
#include "numkong/sets/loongsonasx.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_hammings_packed_u1_best(nk_u1x8_t const *a, void const *b_packed, nk_u32_t *c,
                                                   nk_size_t height, nk_size_t width, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(height), nk_unused_(width), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_hammings_symmetric_u1_best(nk_u1x8_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_u32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_jaccards_packed_u1_best(nk_u1x8_t const *a, void const *b_packed, nk_f32_t *c,
                                                   nk_size_t height, nk_size_t width, nk_size_t depth,
                                                   nk_size_t a_stride, nk_size_t c_stride, nk_capability_t capabilities,
                                                   void *stream) {
    nk_unused_(a), nk_unused_(b_packed), nk_unused_(c), nk_unused_(height), nk_unused_(width), nk_unused_(depth),
        nk_unused_(a_stride), nk_unused_(c_stride), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_jaccards_symmetric_u1_best(nk_u1x8_t const *vectors, nk_size_t vectors_count,
                                                      nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                      nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count,
                                                      nk_capability_t capabilities, void *stream) {
    nk_unused_(vectors), nk_unused_(vectors_count), nk_unused_(depth), nk_unused_(stride), nk_unused_(result),
        nk_unused_(result_stride), nk_unused_(row_start), nk_unused_(row_count), nk_unused_(capabilities),
        nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sets_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                            nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif // NUMKONG_SETS_H
