/**
 *  @file include/numkong/sparse.h
 *  @author Ash Vardanian
 *  @date March 21, 2024
 *  @brief SIMD-accelerated sparse vector dot products.
 *
 *  Contains:
 *
 *  - Set intersection for sorted unique arrays → @c u32 count
 *  - Sparse dot products for weighted sparse vectors
 *
 *  For dtypes:
 *
 *  - @c u16: indices for vocabularies under 64 thousand tokens
 *  - @c u32: indices for vocabularies under 4 billion tokens
 *  - @c u64: indices for trillion-scale combinatorics and graphs
 *  - @c u16 indices + @c bf16 weights → @c f32 product
 *  - @c u32 indices + @c f32 weights → @c f64 product
 *
 *  For hardware architectures:
 *
 *  - Arm: NEON, SVE2
 *  - x86: Haswell, Ice Lake, Turin
 *
 *  @section intersection_algorithm Intersection by Merge
 *
 *  The core primitive is analogous to @c std::set_intersection, taking two sorted arrays of unique
 *  values and producing the intersection size:
 *
 *  @code{.cpp}
 *  std::size_t intersection_size = 0;
 *  while (i != a_length && j != b_length) {
 *      scalar_t ai = a[i], bj = b[j];
 *      intersection_size += ai == bj;
 *      i += ai < bj;
 *      j += ai ≥ bj;
 *  }
 *  @endcode
 *
 *  Weighted sparse dot-products follow the same merge loop, but accumulate a product for matching
 *  indices. For the `u32+f32` family the matched products are widened before accumulation, matching
 *  the widened @c f64 public result.
 *
 *  @code{.cpp}
 *  double product = 0;
 *  while (i != a_length && j != b_length) {
 *      scalar_t ai = a[i], bj = b[j];
 *      product += ai == bj ? a_weights[i] * b_weights[j] : 0;
 *      i += ai < bj;
 *      j += ai ≥ bj;
 *  }
 *  @endcode
 *
 *  @section galloping_search Galloping vs Linear
 *
 *  When the arrays are highly imbalanced, linear merge wastes cycles skipping elements.
 *  The serial implementation switches to a galloping search to jump over large gaps.
 *
 *  @section sparse_x86_instructions Relevant x86 Instructions
 *
 *  The Ice Lake kernels are shuffle/compare heavy; their throughput is often gated by port 5. On
 *  Genoa, many integer ops dual-issue on FP ports, often improving throughput even though each op
 *  runs with higher latency.
 *
 *  @verbatim
 *  Intrinsic                     Instruction                  Icelake          Genoa
 *  _mm512_shuffle_epi32          VPSHUFD (ZMM, ZMM, I8)       1cy @ p5         1cy @ p123
 *  _mm512_mask_cmpneq_epi32_mask VPCMPD (K, ZMM, ZMM, I8)     3cy @ p5         5cy @ p01
 *  _mm512_alignr_epi32           VALIGND (ZMM, ZMM, ZMM, I8)  3cy @ p5         6cy @ p12
 *  _mm512_conflict_epi32         VPCONFLICTD (ZMM, ZMM)       26cy @ p0+p05+p5 7cy @ p01+p12
 *  _mm256_maskz_compress_epi16   VPCOMPRESSW (YMM, K, YMM)    3-6cy @ p5+p5    4-8cy @ p01+p12
 *  _mm256_dpwssds_epi32          VPDPWSSDS (YMM, K, YMM, YMM) 4-5cy @ p01      4cy @ p01
 *  _mm256_dpbf16_ps              VDPBF16PS (YMM, YMM, YMM)    n/a              6cy @ p01
 *  @endverbatim
 *
 *  VP2INTERSECTD is unsupported on Ice Lake and not yet covered by uops.info for Zen5/Turin.
 *  Tiger Lake measures ~36-41cy @ p5 for ZMM variants, which is why we always avoid it on Intel.
 *
 *  @section sparse_references References
 *
 *  @see uops.info: https://uops.info/
 *  @see Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 *  @see Arm Intrinsics Reference: https://developer.arm.com/architectures/instruction-sets/intrinsics/
 *  @see vp2intersect experiments: https://github.com/mozonaut/vp2intersect
 *  @see Diez-Canas "Faster-Than-Native Alternatives for x86 VP2INTERSECT Instructions": https://arxiv.org/pdf/2112.06342.pdf
 *
 */
#ifndef NUMKONG_SPARSE_H
#define NUMKONG_SPARSE_H

#include "numkong/capabilities.h"

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
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NUMKONG_API nk_status_t nk_sparse_intersect_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u16_t *result, nk_size_t *count,
                                                     nk_capability_t capabilities, void *stream);

/**
 *  @brief Set intersection between two sorted u32 arrays.
 *
 *  @param[in] a The first sorted array of indices.
 *  @param[in] b The second sorted array of indices.
 *  @param[in] a_length The number of elements in the first array.
 *  @param[in] b_length The number of elements in the second array.
 *  @param[out] result Output buffer for intersection elements, or NULL to count only.
 *  @param[out] count The output intersection count.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NUMKONG_API nk_status_t nk_sparse_intersect_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u32_t *result, nk_size_t *count,
                                                     nk_capability_t capabilities, void *stream);

/**
 *  @brief Set intersection between two sorted u64 arrays.
 *
 *  @param[in] a The first sorted array of indices.
 *  @param[in] b The second sorted array of indices.
 *  @param[in] a_length The number of elements in the first array.
 *  @param[in] b_length The number of elements in the second array.
 *  @param[out] result Output buffer for intersection elements, or NULL to count only.
 *  @param[out] count The output intersection count.
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NUMKONG_API nk_status_t nk_sparse_intersect_u64_best(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u64_t *result, nk_size_t *count,
                                                     nk_capability_t capabilities, void *stream);

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
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NUMKONG_API nk_status_t nk_sparse_dot_u16bf16_best(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                                   nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                   nk_f32_t *product, nk_capability_t capabilities, void *stream);

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
 *  @param[in] capabilities One device's capabilities, like @c nk_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c nk_success_k, or @c nk_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note Inputs must be sorted in ascending order and contain unique elements.
 */
NUMKONG_API nk_status_t nk_sparse_dot_u32f32_best(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                                  nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                  nk_f64_t *product, nk_capability_t capabilities, void *stream);

/** @copydoc nk_sparse_intersect_u16_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                                       nk_size_t b_length, nk_u16_t *result, nk_size_t *count,
                                                       void *stream);
/** @copydoc nk_sparse_intersect_u32_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                                       nk_size_t b_length, nk_u32_t *result, nk_size_t *count,
                                                       void *stream);
/** @copydoc nk_sparse_intersect_u64_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u64_serial(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                                       nk_size_t b_length, nk_u64_t *result, nk_size_t *count,
                                                       void *stream);
/** @copydoc nk_sparse_dot_u16bf16_best */
NUMKONG_API nk_status_t nk_sparse_dot_u16bf16_serial(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                                     nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                     nk_f32_t *product, void *stream);
/** @copydoc nk_sparse_dot_u32f32_best */
NUMKONG_API nk_status_t nk_sparse_dot_u32f32_serial(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                                    nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                    nk_f64_t *product, void *stream);

#if NUMKONG_TARGET_NEON
/** @copydoc nk_sparse_intersect_u16_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u16_neon(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u16_t *result, nk_size_t *count,
                                                     void *stream);
/** @copydoc nk_sparse_intersect_u32_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u32_neon(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u32_t *result, nk_size_t *count,
                                                     void *stream);
/** @copydoc nk_sparse_intersect_u64_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u64_neon(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u64_t *result, nk_size_t *count,
                                                     void *stream);
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_SVE2
/** @copydoc nk_sparse_intersect_u16_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u16_sve2(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u16_t *result, nk_size_t *count,
                                                     void *stream);
/** @copydoc nk_sparse_intersect_u32_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u32_sve2(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u32_t *result, nk_size_t *count,
                                                     void *stream);
/** @copydoc nk_sparse_intersect_u64_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u64_sve2(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u64_t *result, nk_size_t *count,
                                                     void *stream);
/** @copydoc nk_sparse_dot_u32f32_best */
NUMKONG_API nk_status_t nk_sparse_dot_u32f32_sve2(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                                  nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                  nk_f64_t *product, void *stream);
/** @copydoc nk_sparse_dot_u16bf16_best */
NUMKONG_API nk_status_t nk_sparse_dot_u16bf16_sve2(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                                   nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                   nk_f32_t *product, void *stream);
#endif // NUMKONG_TARGET_SVE2

#if NUMKONG_TARGET_ICELAKE
/** @copydoc nk_sparse_intersect_u16_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u16_icelake(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                                        nk_size_t b_length, nk_u16_t *result, nk_size_t *count,
                                                        void *stream);
/** @copydoc nk_sparse_intersect_u32_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u32_icelake(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                                        nk_size_t b_length, nk_u32_t *result, nk_size_t *count,
                                                        void *stream);
/** @copydoc nk_sparse_intersect_u64_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u64_icelake(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                                        nk_size_t b_length, nk_u64_t *result, nk_size_t *count,
                                                        void *stream);
/** @copydoc nk_sparse_dot_u32f32_best */
NUMKONG_API nk_status_t nk_sparse_dot_u32f32_icelake(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                                     nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                     nk_f64_t *product, void *stream);
#endif // NUMKONG_TARGET_ICELAKE

#if NUMKONG_TARGET_HASWELL
/** @copydoc nk_sparse_dot_u32f32_best */
NUMKONG_API nk_status_t nk_sparse_dot_u32f32_haswell(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                                     nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                     nk_f64_t *product, void *stream);
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_TURIN
/** @copydoc nk_sparse_intersect_u16_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u16_turin(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                                      nk_size_t b_length, nk_u16_t *result, nk_size_t *count,
                                                      void *stream);
/** @copydoc nk_sparse_intersect_u32_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u32_turin(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                                      nk_size_t b_length, nk_u32_t *result, nk_size_t *count,
                                                      void *stream);
/** @copydoc nk_sparse_intersect_u64_best */
NUMKONG_API nk_status_t nk_sparse_intersect_u64_turin(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                                      nk_size_t b_length, nk_u64_t *result, nk_size_t *count,
                                                      void *stream);
/** @copydoc nk_sparse_dot_u16bf16_best */
NUMKONG_API nk_status_t nk_sparse_dot_u16bf16_turin(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                                    nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                    nk_f32_t *product, void *stream);
/** @copydoc nk_sparse_dot_u32f32_best */
NUMKONG_API nk_status_t nk_sparse_dot_u32f32_turin(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                                   nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                   nk_f64_t *product, void *stream);
#endif // NUMKONG_TARGET_TURIN

/** Returns the output dtype for sparse dot products. */
NUMKONG_INLINE nk_dtype_t nk_sparse_dot_output_dtype(nk_dtype_t dtype) {
    switch (dtype) {
    case nk_f32_k: return nk_f64_k;
    case nk_bf16_k: return nk_f32_k;
    default: return nk_dtype_unknown_k;
    }
}

/** Returns the error bound of sparse dot products, per @c nk_accumulation_error_bound of their
 *  output. */
NUMKONG_INLINE nk_f64_t nk_sparse_dot_error_bound(nk_dtype_t dtype) {
    return nk_accumulation_error_bound(nk_sparse_dot_output_dtype(dtype));
}

/**
 *  @brief Finds the sparse kernel of @p kind for @p dtype, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c nk_success_k, @c nk_missing_kernel_k, or @c nk_missing_library_k in header-only builds.
 */
NUMKONG_API nk_status_t nk_sparse_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability);

#if defined(__cplusplus)
} // extern "C"
#endif

#if NUMKONG_HEADER_ONLY
#include "numkong/sparse/serial.h"
#include "numkong/sparse/neon.h"
#include "numkong/sparse/sve2.h"
#include "numkong/sparse/haswell.h"
#include "numkong/sparse/icelake.h"
#include "numkong/sparse/turin.h"

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_API nk_status_t nk_sparse_dot_u32f32_best(nk_u32_t const *a, nk_u32_t const *b, nk_f32_t const *a_weights,
                                                  nk_f32_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                  nk_f64_t *product, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(a_weights), nk_unused_(b_weights), nk_unused_(a_length),
        nk_unused_(b_length), nk_unused_(product), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sparse_dot_u16bf16_best(nk_u16_t const *a, nk_u16_t const *b, nk_bf16_t const *a_weights,
                                                   nk_bf16_t const *b_weights, nk_size_t a_length, nk_size_t b_length,
                                                   nk_f32_t *product, nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(a_weights), nk_unused_(b_weights), nk_unused_(a_length),
        nk_unused_(b_length), nk_unused_(product), nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sparse_intersect_u64_best(nk_u64_t const *a, nk_u64_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u64_t *result, nk_size_t *count,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(a_length), nk_unused_(b_length), nk_unused_(result), nk_unused_(count),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sparse_intersect_u32_best(nk_u32_t const *a, nk_u32_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u32_t *result, nk_size_t *count,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(a_length), nk_unused_(b_length), nk_unused_(result), nk_unused_(count),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sparse_intersect_u16_best(nk_u16_t const *a, nk_u16_t const *b, nk_size_t a_length,
                                                     nk_size_t b_length, nk_u16_t *result, nk_size_t *count,
                                                     nk_capability_t capabilities, void *stream) {
    nk_unused_(a), nk_unused_(b), nk_unused_(a_length), nk_unused_(b_length), nk_unused_(result), nk_unused_(count),
        nk_unused_(capabilities), nk_unused_(stream);
    return nk_missing_library_k;
}

NUMKONG_API nk_status_t nk_sparse_find_kernel(nk_kernel_kind_t kind, nk_dtype_t dtype, nk_capability_t capabilities,
                                              nk_kernel_punned_t *kernel, nk_capability_t *capability) {
    nk_unused_(kind), nk_unused_(dtype), nk_unused_(capabilities);
    *kernel = NUMKONG_NULL, *capability = 0;
    return nk_missing_library_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // NUMKONG_HEADER_ONLY

#endif
