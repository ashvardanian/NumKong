/**
 *  @file include/numkong/maxsim/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 5, 2026
 *  @brief SIMD-accelerated MaxSim, angular distance late-interaction, for WASM Relaxed SIMD.
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Uses wasm_i32x4_relaxed_dot_i8x16_i7x16_add for coarse i8 screening. Codes stay in [-63, 63],
 *  yet engines read a negative i7 operand differently, Wasmtime on x86 as unsigned, so documents
 *  are biased by 64 into [1, 127] and 64 times the query's code sum is subtracted, like the Haswell
 *  and Alder XOR-0x80 approach. Packing routines live in `maxsim/v128.h`.
 *
 *  1Q × 1D tiling, simpler than x86 4x4. Depth steps at 16 bytes, the v128 width in bytes.
 */
#ifndef NUMKONG_MAXSIM_V128RELAXED_H
#define NUMKONG_MAXSIM_V128RELAXED_H

#if NUMKONG_ARCH_WASM_
#if NUMKONG_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/maxsim/serial.h"   // `nk_maxsim_packed_regions_t`
#include "numkong/maxsim/v128.h"     // `nk_maxsim_pack_bf16_v128_`
#include "numkong/dot/v128relaxed.h" // `nk_dot_bf16_widened_v128relaxed_`, `nk_dot_f32_widened_v128relaxed_`
#include "numkong/reduce/v128.h"     // `nk_reduce_add_i32x4_v128_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/** Coarse i8 kernel for WASM Relaxed SIMD as an @c nk_maxsim_coarse_dots_t. Uses
 *  relaxed_dot_i8x16_i7x16_add on documents biased by 64 into [1, 127], where every engine agrees,
 *  and subtracts 64 times each query's code sum. */
NUMKONG_INLINE void nk_maxsim_coarse_dots_v128relaxed_(   //
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,  //
    nk_maxsim_vector_metadata_t const *document_metadata, //
    nk_size_t query_count, nk_size_t document_count,      //
    nk_size_t depth_i8_padded, nk_i32_t *dots) {
    nk_unused_(document_metadata);
    v128_t const bias_i8x16 = wasm_i8x16_splat(64);

    for (nk_size_t query_index = 0; query_index < query_count; query_index++) {
        nk_i8_t const *query_i8_row = query_i8 + query_index * depth_i8_padded;
        v128_t query_sum_i32x4 = wasm_i32x4_splat(0);
        for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 16)
            query_sum_i32x4 = wasm_i32x4_add(
                query_sum_i32x4, wasm_i32x4_extadd_pairwise_i16x8(
                                     wasm_i16x8_extadd_pairwise_i8x16(wasm_v128_load(query_i8_row + depth_index))));
        nk_i32_t const bias_correction = 64 * nk_reduce_add_i32x4_v128_(query_sum_i32x4);

        for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            v128_t accumulator_i32x4 = wasm_i32x4_splat(0);

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 16) {
                v128_t query_i8x16 = wasm_v128_load(query_i8_row + depth_index);
                v128_t document_u8x16 = wasm_i8x16_add(wasm_v128_load(document_i8_row + depth_index), bias_i8x16);
                accumulator_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(query_i8x16, document_u8x16,
                                                                           accumulator_i32x4);
            }

            dots[query_index * document_count + document_index] = nk_reduce_add_i32x4_v128_(accumulator_i32x4) -
                                                                  bias_correction;
        }
    }
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_bf16_v128relaxed_(void const *query, void const *document, nk_size_t depth) {
    return nk_dot_bf16_widened_v128relaxed_((nk_bf16_t const *)query, (nk_bf16_t const *)document, depth);
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f32_v128relaxed_(void const *query, void const *document, nk_size_t depth) {
    return nk_dot_f32_widened_v128relaxed_((nk_f32_t const *)query, (nk_f32_t const *)document, depth);
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f16_v128relaxed_(void const *query, void const *document, nk_size_t depth) {
    return nk_dot_f16_widened_v128relaxed_((nk_f16_t const *)query, (nk_f16_t const *)document, depth);
}

NUMKONG_API nk_status_t nk_maxsim_packed_bf16_v128relaxed( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_v128relaxed_k) ||
        !nk_maxsim_packed_by_(document_packed, nk_cap_v128relaxed_k))
        return nk_pack_mismatch_k;

    *result = (nk_f32_t)nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                                  nk_maxsim_coarse_dots_v128relaxed_,
                                                  nk_maxsim_refine_bf16_v128relaxed_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f32_v128relaxed( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_v128relaxed_k) ||
        !nk_maxsim_packed_by_(document_packed, nk_cap_v128relaxed_k))
        return nk_pack_mismatch_k;

    *result = nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                        nk_maxsim_coarse_dots_v128relaxed_, nk_maxsim_refine_f32_v128relaxed_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f16_v128relaxed( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_v128relaxed_k) ||
        !nk_maxsim_packed_by_(document_packed, nk_cap_v128relaxed_k))
        return nk_pack_mismatch_k;

    *result = (nk_f32_t)nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                                  nk_maxsim_coarse_dots_v128relaxed_,
                                                  nk_maxsim_refine_f16_v128relaxed_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_v128relaxed(nk_size_t vector_count, nk_size_t depth,
                                                             nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_bf16_t), 16);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_v128relaxed(void const *packed, nk_size_t *vectors,
                                                                nk_size_t *depth, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_v128relaxed_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_bf16_v128relaxed(nk_bf16_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, void *packed,
                                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_maxsim_pack_bf16_v128_(vectors, vector_count, depth, stride, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_v128relaxed(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_f32_t), 16);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_v128relaxed(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_v128relaxed_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f32_v128relaxed(nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride, void *packed, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_maxsim_pack_f32_v128_(vectors, vector_count, depth, stride, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_v128relaxed(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_f16_t), 16);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_v128relaxed(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_v128relaxed_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f16_v128relaxed(nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
                                                       nk_size_t stride, void *packed, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_maxsim_pack_f16_v128_(vectors, vector_count, depth, stride, packed);
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_V128RELAXED
#endif // NUMKONG_ARCH_WASM_
#endif // NUMKONG_MAXSIM_V128RELAXED_H
