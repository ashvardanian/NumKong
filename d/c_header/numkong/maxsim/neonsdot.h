/**
 *  @brief SIMD-accelerated MaxSim (angular distance late-interaction) for ARM NEONSDOT.
 *  @file include/numkong/maxsim/neonsdot.h
 *  @author Ash Vardanian
 *  @date February 28, 2026
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Uses ARM SDOT (vdotq_s32) for coarse i8 screening — signed×signed natively, no bias correction.
 *  4x4 register tiling: 4 queries x 4 documents = 16 int32x4_t accumulators per depth loop.
 *  Depth steps at 16 bytes (128-bit NEON = 16 i8 lanes).
 */
#ifndef NK_MAXSIM_NEONSDOT_H
#define NK_MAXSIM_NEONSDOT_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONSDOT

#include "numkong/types.h"
#include "numkong/maxsim/serial.h" // `nk_maxsim_packed_header_t`
#include "numkong/dot.h"           // `nk_dot_bf16`, `nk_dot_f32`, `nk_dot_f16`
#include "numkong/cast/neon.h"     // `nk_f16_to_f32_neon`
#include "numkong/spatial/neon.h"  // `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+dotprod")
#endif

NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_neonsdot(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_bf16_t), 16);
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_neonsdot(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_f32_t), 16);
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_neonsdot(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_f16_t), 16);
}

NK_PUBLIC void nk_maxsim_pack_bf16_neonsdot( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_bf16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, element_bytes);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_f32_t norm_sq;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f,
                                   (nk_maxsim_to_f32_t)nk_bf16_to_f32_serial,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index], &norm_sq);
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? (1.0f / nk_f32_sqrt_neon(norm_sq)) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

NK_PUBLIC void nk_maxsim_pack_f32_neonsdot( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_f32_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, element_bytes);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_f32_t norm_sq;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f, nk_f32_to_f32_,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index], &norm_sq);
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? (1.0f / nk_f32_sqrt_neon(norm_sq)) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

NK_PUBLIC void nk_maxsim_pack_f16_neonsdot( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_f16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, element_bytes);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_f32_t norm_sq;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f,
                                   (nk_maxsim_to_f32_t)nk_f16_to_f32_neon,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index], &norm_sq);
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? (1.0f / nk_f32_sqrt_neon(norm_sq)) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

/**
 *  @brief Factored coarse i8 argmax kernel for NEONSDOT.
 *  Uses vdotq_s32 (signed×signed) — no XOR bias, no metadata parameter.
 *  4Q×4D register tiling with 16 int32x4_t accumulators.
 */
NK_INTERNAL void nk_maxsim_coarse_argmax_neonsdot_(                                                       //
    nk_i8_t const *query_i8, nk_i8_t const *document_i8, nk_size_t query_count, nk_size_t document_count, //
    nk_size_t depth_i8_padded, nk_u32_t *best_document_indices) {

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {
        nk_i32_t running_max_i32[4] = {NK_I32_MIN, NK_I32_MIN, NK_I32_MIN, NK_I32_MIN};
        nk_u32_t running_argmax_u32[4] = {0, 0, 0, 0};

        // 4Q×4D document blocking
        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= document_count; document_block_start_index += 4) {
            // 16 accumulators: [query_idx][doc_idx]
            int32x4_t accumulator_tiles_i32x4[4][4];
            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++)
                for (nk_size_t document_tile_index = 0; document_tile_index < 4; document_tile_index++)
                    accumulator_tiles_i32x4[query_tile_index][document_tile_index] = vdupq_n_s32(0);

            // Depth loop: 16 bytes per step
            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 16) {
                int8x16_t query_i8x16_0 = vld1q_s8(
                    (int8_t const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index));
                int8x16_t query_i8x16_1 = vld1q_s8(
                    (int8_t const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index));
                int8x16_t query_i8x16_2 = vld1q_s8(
                    (int8_t const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index));
                int8x16_t query_i8x16_3 = vld1q_s8(
                    (int8_t const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index));

                int8x16_t document_i8x16;

                document_i8x16 = vld1q_s8(
                    (int8_t const *)(document_i8 + (document_block_start_index + 0) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x4[0][0] = vdotq_s32(accumulator_tiles_i32x4[0][0], query_i8x16_0, document_i8x16);
                accumulator_tiles_i32x4[1][0] = vdotq_s32(accumulator_tiles_i32x4[1][0], query_i8x16_1, document_i8x16);
                accumulator_tiles_i32x4[2][0] = vdotq_s32(accumulator_tiles_i32x4[2][0], query_i8x16_2, document_i8x16);
                accumulator_tiles_i32x4[3][0] = vdotq_s32(accumulator_tiles_i32x4[3][0], query_i8x16_3, document_i8x16);

                document_i8x16 = vld1q_s8(
                    (int8_t const *)(document_i8 + (document_block_start_index + 1) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x4[0][1] = vdotq_s32(accumulator_tiles_i32x4[0][1], query_i8x16_0, document_i8x16);
                accumulator_tiles_i32x4[1][1] = vdotq_s32(accumulator_tiles_i32x4[1][1], query_i8x16_1, document_i8x16);
                accumulator_tiles_i32x4[2][1] = vdotq_s32(accumulator_tiles_i32x4[2][1], query_i8x16_2, document_i8x16);
                accumulator_tiles_i32x4[3][1] = vdotq_s32(accumulator_tiles_i32x4[3][1], query_i8x16_3, document_i8x16);

                document_i8x16 = vld1q_s8(
                    (int8_t const *)(document_i8 + (document_block_start_index + 2) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x4[0][2] = vdotq_s32(accumulator_tiles_i32x4[0][2], query_i8x16_0, document_i8x16);
                accumulator_tiles_i32x4[1][2] = vdotq_s32(accumulator_tiles_i32x4[1][2], query_i8x16_1, document_i8x16);
                accumulator_tiles_i32x4[2][2] = vdotq_s32(accumulator_tiles_i32x4[2][2], query_i8x16_2, document_i8x16);
                accumulator_tiles_i32x4[3][2] = vdotq_s32(accumulator_tiles_i32x4[3][2], query_i8x16_3, document_i8x16);

                document_i8x16 = vld1q_s8(
                    (int8_t const *)(document_i8 + (document_block_start_index + 3) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x4[0][3] = vdotq_s32(accumulator_tiles_i32x4[0][3], query_i8x16_0, document_i8x16);
                accumulator_tiles_i32x4[1][3] = vdotq_s32(accumulator_tiles_i32x4[1][3], query_i8x16_1, document_i8x16);
                accumulator_tiles_i32x4[2][3] = vdotq_s32(accumulator_tiles_i32x4[2][3], query_i8x16_2, document_i8x16);
                accumulator_tiles_i32x4[3][3] = vdotq_s32(accumulator_tiles_i32x4[3][3], query_i8x16_3, document_i8x16);
            }

            // Reduce and update argmax for each of 4 queries × 4 documents
            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++) {
                for (nk_size_t document_tile_index = 0; document_tile_index < 4; document_tile_index++) {
                    nk_i32_t dot = vaddvq_s32(accumulator_tiles_i32x4[query_tile_index][document_tile_index]);
                    if (dot > running_max_i32[query_tile_index]) {
                        running_max_i32[query_tile_index] = dot;
                        running_argmax_u32[query_tile_index] = (nk_u32_t)(document_block_start_index +
                                                                          document_tile_index);
                    }
                }
            }
        }

        // Document tail: 4Q×1D
        for (nk_size_t document_index = document_block_start_index; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;

            int32x4_t accumulator_i32x4_0 = vdupq_n_s32(0);
            int32x4_t accumulator_i32x4_1 = vdupq_n_s32(0);
            int32x4_t accumulator_i32x4_2 = vdupq_n_s32(0);
            int32x4_t accumulator_i32x4_3 = vdupq_n_s32(0);

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 16) {
                int8x16_t document_i8x16 = vld1q_s8((int8_t const *)(document_i8_row + depth_index));

                accumulator_i32x4_0 = vdotq_s32(
                    accumulator_i32x4_0,
                    vld1q_s8(
                        (int8_t const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                    document_i8x16);
                accumulator_i32x4_1 = vdotq_s32(
                    accumulator_i32x4_1,
                    vld1q_s8(
                        (int8_t const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                    document_i8x16);
                accumulator_i32x4_2 = vdotq_s32(
                    accumulator_i32x4_2,
                    vld1q_s8(
                        (int8_t const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                    document_i8x16);
                accumulator_i32x4_3 = vdotq_s32(
                    accumulator_i32x4_3,
                    vld1q_s8(
                        (int8_t const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                    document_i8x16);
            }

            nk_i32_t dots[4] = {vaddvq_s32(accumulator_i32x4_0), vaddvq_s32(accumulator_i32x4_1),
                                vaddvq_s32(accumulator_i32x4_2), vaddvq_s32(accumulator_i32x4_3)};
            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++) {
                if (dots[query_tile_index] > running_max_i32[query_tile_index]) {
                    running_max_i32[query_tile_index] = dots[query_tile_index];
                    running_argmax_u32[query_tile_index] = (nk_u32_t)document_index;
                }
            }
        }

        for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++)
            best_document_indices[query_block_start_index + query_tile_index] = running_argmax_u32[query_tile_index];
    }

    // Query tail: 1Q×1D
    for (nk_size_t query_index = query_block_start_index; query_index < query_count; query_index++) {
        nk_i8_t const *query_i8_row = query_i8 + query_index * depth_i8_padded;
        nk_i32_t running_max_i32 = NK_I32_MIN;
        nk_u32_t running_argmax_u32 = 0;

        for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            int32x4_t accumulator_i32x4 = vdupq_n_s32(0);

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 16) {
                int8x16_t query_i8x16 = vld1q_s8((int8_t const *)(query_i8_row + depth_index));
                int8x16_t document_i8x16 = vld1q_s8((int8_t const *)(document_i8_row + depth_index));
                accumulator_i32x4 = vdotq_s32(accumulator_i32x4, query_i8x16, document_i8x16);
            }

            nk_i32_t coarse_dot_i32 = vaddvq_s32(accumulator_i32x4);
            if (coarse_dot_i32 > running_max_i32) {
                running_max_i32 = coarse_dot_i32;
                running_argmax_u32 = (nk_u32_t)document_index;
            }
        }

        best_document_indices[query_index] = running_argmax_u32;
    }
}

NK_PUBLIC void nk_maxsim_packed_bf16_neonsdot( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_neonsdot_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                          regions.document_quantized, chunk_size, document_count,
                                          regions.depth_i8_padded, best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f32_t dot_result;
            nk_dot_bf16((nk_bf16_t const *)(regions.query_originals +
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

NK_PUBLIC void nk_maxsim_packed_f32_neonsdot( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_neonsdot_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                          regions.document_quantized, chunk_size, document_count,
                                          regions.depth_i8_padded, best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f64_t dot_result;
            nk_dot_f32(
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

NK_PUBLIC void nk_maxsim_packed_f16_neonsdot( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_neonsdot_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                          regions.document_quantized, chunk_size, document_count,
                                          regions.depth_i8_padded, best_document_indices);

        for (nk_size_t query_index = 0; query_index < chunk_size; query_index++) {
            nk_u32_t best_document_index = best_document_indices[query_index];
            nk_f32_t dot_result;
            nk_dot_f16(
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
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONSDOT
#endif // NK_TARGET_ARM_
#endif // NK_MAXSIM_NEONSDOT_H
