/**
 *  @brief SIMD-accelerated Sparse Vector Dot Products.
 *  @file include/numkong/sparse.h
 *  @author Ash Vardanian
 *  @date March 21, 2024
 *
 *  Contains:
 *
 *  - Set intersection for sorted unique arrays → `u32` count
 *  - Sparse dot products for weighted sparse vectors
 *
 *  For dtypes:
 *
 *  - `u16`: indices for vocabularies under 64 thousand tokens
 *  - `u32`: indices for vocabularies under 4 billion tokens
 *  - `u64`: indices for trillion-scale combinatorics and graphs
 *  - `u16` indices + `bf16` weights → `f32` product
 *  - `u32` indices + `f32` weights → `f64` product
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, SVE2
 *  - x86: Ice Lake, Turin
 *
 *  @section intersection_algorithm Intersection by Merge
 *
 *  The core primitive is analogous to `std::set_intersection`, taking two sorted arrays
 *  of unique values and producing the intersection size:
 *
 *      std::size_t intersection_size = 0;
 *      while (i != a_length && j != b_length) {
 *          scalar_t ai = a[i], bj = b[j];
 *          intersection_size += ai == bj;
 *          i += ai < bj;
 *          j += ai ≥ bj;
 *      }
 *
 *  Weighted sparse dot-products follow the same merge loop, but accumulate a product
 *  for matching indices. For the `u32+f32` family the matched products are widened before
 *  accumulation, matching the widened `f64` public result.
 *
 *      double product = 0;
 *      while (i != a_length && j != b_length) {
 *          scalar_t ai = a[i], bj = b[j];
 *          product += ai == bj ? a_weights[i] * b_weights[j] : 0;
 *          i += ai < bj;
 *          j += ai ≥ bj;
 *      }
 *
 *  @section galloping_search Galloping vs Linear
 *
 *  When the arrays are highly imbalanced, linear merge wastes cycles skipping elements.
 *  The serial implementation switches to a galloping search to jump over large gaps.
 *
 *  @section x86_instructions Relevant x86 Instructions
 *
 *  The Ice Lake kernels are shuffle/compare heavy; their throughput is often gated by port 5.
 *  On Genoa, many integer ops dual-issue on FP ports, often improving throughput despite higher latency.
 *
 *      Intrinsic                       Instruction                      Ice           Genoa
 *      _mm512_shuffle_epi32            VPSHUFD (ZMM, ZMM, I8)           1c @ p5       1c @ p123
 *      _mm512_mask_cmpneq_epi32_mask   VPCMPD (K, ZMM, ZMM, I8)         3c @ p5       5c @ p01
 *      _mm512_alignr_epi32             VALIGND (ZMM, ZMM, ZMM, I8)      3c @ p5       6c @ p12
 *      _mm512_conflict_epi32           VPCONFLICTD (ZMM, ZMM)           26c @ p0/5    7c @ p01/12
 *      _mm256_maskz_compress_epi16     VPCOMPRESSW (YMM, K, YMM)        3-6c @ p5     4-8c @ p01/12
 *      _mm256_dpwssds_epi32            VPDPWSSDS (YMM, K, YMM, YMM)     4-5c @ p01    4c @ p01
 *      _mm256_dpbf16_ps                VDPBF16PS (YMM, YMM, YMM)        n/a           6c @ p01
 *
 *  VP2INTERSECTD is unsupported on Ice Lake and not yet covered by uops.info for Zen5/Turin.
 *  Tiger Lake measures ~36-41c @ p5 for ZMM variants, which is why we always avoid it on Intel.
 *
 *  @section references References
 *
 *  - uops.info: https://uops.info/
 *  - Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  - Arm Intrinsics Reference: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  - vp2intersect experiments: https://github.com/mozonaut/vp2intersect
 *  - Diez-Canas "Faster-Than-Native Alternatives for x86 VP2INTERSECT Instructions":
 *    https://arxiv.org/pdf/2112.06342.pdf
 *
 */
#ifndef NK_SPARSE_H
#define NK_SPARSE_H

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Set intersection between two sorted u16 arrays.
 *
 *  @param[in] a The first sorted array of indices.
 *  @param[in] b The second sorted array of indices.
 *  @param[in] a_length The number of elements in the first array.
 *  @param[in] b_length The number of elements in the second array.
 *  @param[out] result Output buffer for intersection elements, or NULL to count only.
 *  @param[out] count The output intersection count.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NK_DYNAMIC void nk_sparse_intersect_u16( //
    nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length, nk_size_t b_length, nk_u16_t *result, nk_size_t *count);

/**
 *  @brief Set intersection between two sorted u32 arrays.
 *
 *  @param[in] a The first sorted array of indices.
 *  @param[in] b The second sorted array of indices.
 *  @param[in] a_length The number of elements in the first array.
 *  @param[in] b_length The number of elements in the second array.
 *  @param[out] result Output buffer for intersection elements, or NULL to count only.
 *  @param[out] count The output intersection count.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NK_DYNAMIC void nk_sparse_intersect_u32( //
    nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length, nk_size_t b_length, nk_u32_t *result, nk_size_t *count);

/**
 *  @brief Set intersection between two sorted u64 arrays.
 *
 *  @param[in] a The first sorted array of indices.
 *  @param[in] b The second sorted array of indices.
 *  @param[in] a_length The number of elements in the first array.
 *  @param[in] b_length The number of elements in the second array.
 *  @param[out] result Output buffer for intersection elements, or NULL to count only.
 *  @param[out] count The output intersection count.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NK_DYNAMIC void nk_sparse_intersect_u64( //
    nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length, nk_size_t b_length, nk_u64_t *result, nk_size_t *count);

/**
 *  @brief Sparse dot-product over u16 indices with bf16 weights.
 *
 *  @param[in] a The first sorted array of indices.
 *  @param[in] b The second sorted array of indices.
 *  @param[in] a_weights The bf16 weights for the first array.
 *  @param[in] b_weights The bf16 weights for the second array.
 *  @param[in] a_length The number of elements in the first array.
 *  @param[in] b_length The number of elements in the second array.
 *  @param[out] product The output dot product.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NK_DYNAMIC void nk_sparse_dot_u16bf16( //
    nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights, nk_bf16_t const *b_weights, nk_size_t a_length,
    nk_size_t b_length, nk_f32_t *product);

/**
 *  @brief Sparse dot-product over u32 indices with f32 weights.
 *
 *  @param[in] a The first sorted array of indices.
 *  @param[in] b The second sorted array of indices.
 *  @param[in] a_weights The f32 weights for the first array.
 *  @param[in] b_weights The f32 weights for the second array.
 *  @param[in] a_length The number of elements in the first array.
 *  @param[in] b_length The number of elements in the second array.
 *  @param[out] product The output dot product.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NK_DYNAMIC void nk_sparse_dot_u32f32( //
    nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights, nk_f32_t const *b_weights, nk_size_t a_length,
    nk_size_t b_length, nk_f64_t *product);

/** @copydoc nk_sparse_intersect_u16 */
NK_PUBLIC void nk_sparse_intersect_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                              nk_size_t b_length, nk_u16_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u32 */
NK_PUBLIC void nk_sparse_intersect_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                              nk_size_t b_length, nk_u32_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u64 */
NK_PUBLIC void nk_sparse_intersect_u64_serial(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                              nk_size_t b_length, nk_u64_t *result, nk_size_t *count);
/** @copydoc nk_sparse_dot_u16bf16 */
NK_PUBLIC void nk_sparse_dot_u16bf16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                            nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                            nk_f32_t *product);
/** @copydoc nk_sparse_dot_u32f32 */
NK_PUBLIC void nk_sparse_dot_u32f32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                           nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                           nk_f64_t *product);

#if NK_TARGET_NEON
/** @copydoc nk_sparse_intersect_u16 */
NK_PUBLIC void nk_sparse_intersect_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                            nk_size_t b_length, nk_u16_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u32 */
NK_PUBLIC void nk_sparse_intersect_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                            nk_size_t b_length, nk_u32_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u64 */
NK_PUBLIC void nk_sparse_intersect_u64_neon(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                            nk_size_t b_length, nk_u64_t *result, nk_size_t *count);
#endif // NK_TARGET_NEON

#if NK_TARGET_SVE2
/** @copydoc nk_sparse_intersect_u16 */
NK_PUBLIC void nk_sparse_intersect_u16_sve2(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                            nk_size_t b_length, nk_u16_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u32 */
NK_PUBLIC void nk_sparse_intersect_u32_sve2(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                            nk_size_t b_length, nk_u32_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u64 */
NK_PUBLIC void nk_sparse_intersect_u64_sve2(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                            nk_size_t b_length, nk_u64_t *result, nk_size_t *count);
/** @copydoc nk_sparse_dot_u32f32 */
NK_PUBLIC void nk_sparse_dot_u32f32_sve2(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                         nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                         nk_f64_t *product);
#endif // NK_TARGET_SVE2

#if NK_TARGET_SVE2 && NK_TARGET_SVEBFDOT
/** @copydoc nk_sparse_dot_u16bf16 */
NK_PUBLIC void nk_sparse_dot_u16bf16_sve2(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                          nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                          nk_f32_t *product);
#endif // NK_TARGET_SVE2 && NK_TARGET_SVEBFDOT

#if NK_TARGET_ICELAKE
/** @copydoc nk_sparse_intersect_u16 */
NK_PUBLIC void nk_sparse_intersect_u16_icelake(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                               nk_size_t b_length, nk_u16_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u32 */
NK_PUBLIC void nk_sparse_intersect_u32_icelake(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                               nk_size_t b_length, nk_u32_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u64 */
NK_PUBLIC void nk_sparse_intersect_u64_icelake(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                               nk_size_t b_length, nk_u64_t *result, nk_size_t *count);
/** @copydoc nk_sparse_dot_u32f32 */
NK_PUBLIC void nk_sparse_dot_u32f32_icelake(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                            nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                            nk_f64_t *product);
#endif // NK_TARGET_ICELAKE

#if NK_TARGET_TURIN
/** @copydoc nk_sparse_intersect_u16 */
NK_PUBLIC void nk_sparse_intersect_u16_turin(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                             nk_size_t b_length, nk_u16_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u32 */
NK_PUBLIC void nk_sparse_intersect_u32_turin(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                             nk_size_t b_length, nk_u32_t *result, nk_size_t *count);
/** @copydoc nk_sparse_intersect_u64 */
NK_PUBLIC void nk_sparse_intersect_u64_turin(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                             nk_size_t b_length, nk_u64_t *result, nk_size_t *count);
/** @copydoc nk_sparse_dot_u16bf16 */
NK_PUBLIC void nk_sparse_dot_u16bf16_turin(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                           nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                           nk_f32_t *product);
/** @copydoc nk_sparse_dot_u32f32 */
NK_PUBLIC void nk_sparse_dot_u32f32_turin(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                          nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                          nk_f64_t *product);
#endif // NK_TARGET_TURIN

/**
 *  @brief  Returns the output dtype for sparse dot products.
 */
NK_INTERNAL nk_dtype_t nk_sparse_dot_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f32_k: return nk_f64_k;
    case nk_bf16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#include "numkong/sparse/serial.h"
#include "numkong/sparse/neon.h"
#include "numkong/sparse/sve2.h"
#include "numkong/sparse/icelake.h"
#include "numkong/sparse/turin.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if !NK_DYNAMIC_DISPATCH

NK_PUBLIC void nk_sparse_intersect_u16(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length, nk_size_t b_length,
                                       nk_u16_t *result, nk_size_t *count) {
#if NK_TARGET_SVE2
    nk_sparse_intersect_u16_sve2(a, b, a_length, b_length, result, count);
#elif NK_TARGET_NEON
    nk_sparse_intersect_u16_neon(a, b, a_length, b_length, result, count);
#elif NK_TARGET_TURIN
    nk_sparse_intersect_u16_turin(a, b, a_length, b_length, result, count);
#elif NK_TARGET_ICELAKE
    nk_sparse_intersect_u16_icelake(a, b, a_length, b_length, result, count);
#else
    nk_sparse_intersect_u16_serial(a, b, a_length, b_length, result, count);
#endif
}

NK_PUBLIC void nk_sparse_intersect_u32(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length, nk_size_t b_length,
                                       nk_u32_t *result, nk_size_t *count) {
#if NK_TARGET_SVE2
    nk_sparse_intersect_u32_sve2(a, b, a_length, b_length, result, count);
#elif NK_TARGET_NEON
    nk_sparse_intersect_u32_neon(a, b, a_length, b_length, result, count);
#elif NK_TARGET_TURIN
    nk_sparse_intersect_u32_turin(a, b, a_length, b_length, result, count);
#elif NK_TARGET_ICELAKE
    nk_sparse_intersect_u32_icelake(a, b, a_length, b_length, result, count);
#else
    nk_sparse_intersect_u32_serial(a, b, a_length, b_length, result, count);
#endif
}

NK_PUBLIC void nk_sparse_intersect_u64(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length, nk_size_t b_length,
                                       nk_u64_t *result, nk_size_t *count) {
#if NK_TARGET_SVE2
    nk_sparse_intersect_u64_sve2(a, b, a_length, b_length, result, count);
#elif NK_TARGET_NEON
    nk_sparse_intersect_u64_neon(a, b, a_length, b_length, result, count);
#elif NK_TARGET_TURIN
    nk_sparse_intersect_u64_turin(a, b, a_length, b_length, result, count);
#elif NK_TARGET_ICELAKE
    nk_sparse_intersect_u64_icelake(a, b, a_length, b_length, result, count);
#else
    nk_sparse_intersect_u64_serial(a, b, a_length, b_length, result, count);
#endif
}

NK_PUBLIC void nk_sparse_dot_u16bf16(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                     nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                     nk_f32_t *product) {
#if NK_TARGET_SVE2 && NK_TARGET_SVEBFDOT
    nk_sparse_dot_u16bf16_sve2(a, b, a_weights, b_weights, a_length, b_length, product);
#elif NK_TARGET_TURIN
    nk_sparse_dot_u16bf16_turin(a, b, a_weights, b_weights, a_length, b_length, product);
#else
    nk_sparse_dot_u16bf16_serial(a, b, a_weights, b_weights, a_length, b_length, product);
#endif
}

NK_PUBLIC void nk_sparse_dot_u32f32(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                    nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                    nk_f64_t *product) {
#if NK_TARGET_SVE2
    nk_sparse_dot_u32f32_sve2(a, b, a_weights, b_weights, a_length, b_length, product);
#elif NK_TARGET_TURIN
    nk_sparse_dot_u32f32_turin(a, b, a_weights, b_weights, a_length, b_length, product);
#elif NK_TARGET_ICELAKE
    nk_sparse_dot_u32f32_icelake(a, b, a_weights, b_weights, a_length, b_length, product);
#else
    nk_sparse_dot_u32f32_serial(a, b, a_weights, b_weights, a_length, b_length, product);
#endif
}

#endif // !NK_DYNAMIC_DISPATCH

#if defined(__cplusplus)
} // extern "C"
#endif

#endif
