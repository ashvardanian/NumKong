/**
 *  @file include/numkong/maxsim/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 5, 2026
 *  @brief SIMD-accelerated MaxSim, angular distance late-interaction, for WASM Relaxed SIMD.
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Uses wasm_i32x4_relaxed_dot_i8x16_i7x16_add for coarse i8 screening. Both operands stay within
 *  i7 range [-63, 63] for native signed × signed arithmetic, so no bias correction is needed,
 *  unlike the Haswell and Alder XOR-0x80 approach. Packing routines live in `maxsim/v128.h`.
 *
 *  1Q × 1D tiling, simpler than x86 4x4, with scalar running argmax. Depth steps at 16 bytes, the
 *  v128 width in bytes.
 */
#ifndef NUMKONG_MAXSIM_V128RELAXED_H
#define NUMKONG_MAXSIM_V128RELAXED_H

#if NUMKONG_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/maxsim/serial.h"   // `nk_maxsim_packed_regions_t`
#include "numkong/dot/v128relaxed.h" // `nk_dot_bf16_v128relaxed`, `nk_dot_f32_v128relaxed`, `nk_dot_f16_v128relaxed`
#include "numkong/reduce/v128.h"     // `nk_reduce_add_i32x4_v128_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/** Coarse i8 argmax kernel for WASM Relaxed SIMD. Uses relaxed_dot_i8x16_i7x16_add with both
 *  operands in [-63, 63], so native signed × signed arithmetic needs no bias correction. Simple
 *  1Q × 1D tiling with a scalar running argmax over screening-weighted dots. */
NUMKONG_HELPER_INLINE void nk_maxsim_coarse_argmax_v128relaxed_( //
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,         //
    nk_maxsim_vector_metadata_t const *document_metadata,        //
    nk_size_t query_count, nk_size_t document_count,             //
    nk_size_t depth_i8_padded, nk_u32_t *best_document_indices) {

    for (nk_size_t query_index = 0; query_index < query_count; query_index++) {
        nk_i8_t const *query_i8_row = query_i8 + query_index * depth_i8_padded;
        nk_f32_t running_max_f32 = NUMKONG_F32_MIN;
        nk_u32_t running_argmax_u32 = 0;

        for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            v128_t accumulator_i32x4 = wasm_i32x4_splat(0);

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 16) {
                v128_t query_i8x16 = wasm_v128_load(query_i8_row + depth_index);
                v128_t document_i8x16 = wasm_v128_load(document_i8_row + depth_index);
                accumulator_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(query_i8x16, document_i8x16,
                                                                           accumulator_i32x4);
            }

            // Horizontal i32x4 reduce → scalar
            nk_i32_t coarse_dot_i32 = nk_reduce_add_i32x4_v128_(accumulator_i32x4);
            nk_f32_t coarse_score_f32 = (nk_f32_t)coarse_dot_i32 * document_metadata[document_index].screen_weight_f32;

            if (coarse_score_f32 > running_max_f32) {
                running_max_f32 = coarse_score_f32;
                running_argmax_u32 = (nk_u32_t)document_index;
            }
        }

        best_document_indices[query_index] = running_argmax_u32;
    }
}

NUMKONG_API_COMPTIME void nk_maxsim_packed_bf16_v128relaxed( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_v128relaxed_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                             regions.document_quantized, regions.document_metadata, chunk_size,
                                             document_count, regions.depth_i8_padded, best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f32_t dot_result;
            nk_dot_bf16_v128relaxed((nk_bf16_t const *)(regions.query_originals +
                                                        (chunk_start + query_index) * regions.query_original_stride),
                                    (nk_bf16_t const *)(regions.document_originals +
                                                        best_document_index * regions.document_original_stride),
                                    depth, &dot_result);
            nk_f32_t cosine = dot_result * regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                              regions.document_metadata[best_document_index].inverse_norm_f32;
            nk_f32_t angular = 1.0f - cosine;
            if (angular < 0.0f) angular = 0.0f;
            total_angular_distance += (nk_f64_t)angular;
        }
    }

    *result = (nk_f32_t)total_angular_distance;
}

NUMKONG_API_COMPTIME void nk_maxsim_packed_f32_v128relaxed( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_v128relaxed_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                             regions.document_quantized, regions.document_metadata, chunk_size,
                                             document_count, regions.depth_i8_padded, best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f64_t dot_result;
            nk_dot_f32_v128relaxed(
                (nk_f32_t const *)(regions.query_originals +
                                   (chunk_start + query_index) * regions.query_original_stride),
                (nk_f32_t const *)(regions.document_originals + best_document_index * regions.document_original_stride),
                depth, &dot_result);
            nk_f64_t cosine = dot_result *
                              (nk_f64_t)regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                              (nk_f64_t)regions.document_metadata[best_document_index].inverse_norm_f32;
            nk_f64_t angular = 1.0 - cosine;
            if (angular < 0.0) angular = 0.0;
            total_angular_distance += angular;
        }
    }

    *result = total_angular_distance;
}

NUMKONG_API_COMPTIME void nk_maxsim_packed_f16_v128relaxed( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_v128relaxed_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                             regions.document_quantized, regions.document_metadata, chunk_size,
                                             document_count, regions.depth_i8_padded, best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f32_t dot_result;
            nk_dot_f16_v128relaxed(
                (nk_f16_t const *)(regions.query_originals +
                                   (chunk_start + query_index) * regions.query_original_stride),
                (nk_f16_t const *)(regions.document_originals + best_document_index * regions.document_original_stride),
                depth, &dot_result);
            nk_f32_t cosine = dot_result * regions.query_metadata[chunk_start + query_index].inverse_norm_f32 *
                              regions.document_metadata[best_document_index].inverse_norm_f32;
            nk_f32_t angular = 1.0f - cosine;
            if (angular < 0.0f) angular = 0.0f;
            total_angular_distance += (nk_f64_t)angular;
        }
    }

    *result = (nk_f32_t)total_angular_distance;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_V128RELAXED
#endif // NUMKONG_MAXSIM_V128RELAXED_H
