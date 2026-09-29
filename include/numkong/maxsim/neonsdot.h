/**
 *  @file include/numkong/maxsim/neonsdot.h
 *  @author Ash Vardanian
 *  @date February 28, 2026
 *  @brief SIMD-accelerated MaxSim, angular distance late-interaction, for ARM NEONSDOT.
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Uses ARM SDOT, vdotq_s32, for coarse i8 screening — signed × signed natively, no bias
 *  correction. 4x4 register tiling: 4 queries × 4 documents = 16 int32x4_t accumulators per depth
 *  loop. Depth steps at 16 bytes, the 128-bit NEON width of 16 i8 lanes.
 */
#ifndef NUMKONG_MAXSIM_NEONSDOT_H
#define NUMKONG_MAXSIM_NEONSDOT_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONSDOT

#include "numkong/types.h"
#include "numkong/maxsim/serial.h" // `nk_maxsim_packed_header_t`
#include "numkong/dot/neon.h"      // `nk_dot_bf16_through_f32_neon_`, `nk_dot_f32_through_f64_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+dotprod")
#endif

NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_neonsdot(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_bf16_t), 16);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_neonsdot(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                             void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_neonsdot_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_neonsdot(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_f32_t), 16);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_neonsdot(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                            void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_neonsdot_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_neonsdot(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_f16_t), 16);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_neonsdot(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                            void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_neonsdot_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_bf16_neonsdot( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_in_bytes, void *packed,
    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_bf16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, element_bytes,
                                                               nk_cap_neonsdot_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride_in_bytes;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f,
                                   (nk_maxsim_to_f32_t)nk_bf16_to_f32_, &quantized_i8[vector_index * depth_i8_padded],
                                   &metadata[vector_index]);
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f32_neonsdot( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_in_bytes, void *packed,
    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_f32_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, element_bytes,
                                                               nk_cap_neonsdot_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride_in_bytes;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f, nk_f32_to_f32_,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index]);
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f16_neonsdot( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_in_bytes, void *packed,
    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_f16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, element_bytes,
                                                               nk_cap_neonsdot_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride_in_bytes;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f,
                                   (nk_maxsim_to_f32_t)nk_f16_to_f32_, &quantized_i8[vector_index * depth_i8_padded],
                                   &metadata[vector_index]);
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
    return nk_success_k;
}

/** Coarse i8 kernel for NEONSDOT as an @c nk_maxsim_coarse_dots_t. Uses vdotq_s32 (signed ×
 *  signed), so no XOR bias. 4Q × 4D register tiling with 16 int32x4_t accumulators. */
NUMKONG_INLINE void nk_maxsim_coarse_dots_neonsdot_(      //
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,  //
    nk_maxsim_vector_metadata_t const *document_metadata, //
    nk_size_t query_count, nk_size_t document_count,      //
    nk_size_t depth_i8_padded, nk_i32_t *dots) {
    nk_unused_(document_metadata);

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {
        // 4Q × 4D document blocking
        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= document_count; document_block_start_index += 4) {
            // 16 accumulators: [query_idx][doc_idx]
            int32x4_t accumulator_tiles_i32x4[4][4];
            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++)
                for (nk_size_t document_tile_index = 0; document_tile_index < 4; document_tile_index++)
                    accumulator_tiles_i32x4[query_tile_index][document_tile_index] = vdupq_n_s32(0);

            // Depth loop: 16 bytes per step
            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 16) {
                int8x16_t query_0_i8x16 = vld1q_s8(
                    (nk_i8_t const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index));
                int8x16_t query_1_i8x16 = vld1q_s8(
                    (nk_i8_t const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index));
                int8x16_t query_2_i8x16 = vld1q_s8(
                    (nk_i8_t const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index));
                int8x16_t query_3_i8x16 = vld1q_s8(
                    (nk_i8_t const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index));

                int8x16_t document_i8x16;

                document_i8x16 = vld1q_s8(
                    (nk_i8_t const *)(document_i8 + (document_block_start_index + 0) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x4[0][0] = vdotq_s32(accumulator_tiles_i32x4[0][0], query_0_i8x16, document_i8x16);
                accumulator_tiles_i32x4[1][0] = vdotq_s32(accumulator_tiles_i32x4[1][0], query_1_i8x16, document_i8x16);
                accumulator_tiles_i32x4[2][0] = vdotq_s32(accumulator_tiles_i32x4[2][0], query_2_i8x16, document_i8x16);
                accumulator_tiles_i32x4[3][0] = vdotq_s32(accumulator_tiles_i32x4[3][0], query_3_i8x16, document_i8x16);

                document_i8x16 = vld1q_s8(
                    (nk_i8_t const *)(document_i8 + (document_block_start_index + 1) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x4[0][1] = vdotq_s32(accumulator_tiles_i32x4[0][1], query_0_i8x16, document_i8x16);
                accumulator_tiles_i32x4[1][1] = vdotq_s32(accumulator_tiles_i32x4[1][1], query_1_i8x16, document_i8x16);
                accumulator_tiles_i32x4[2][1] = vdotq_s32(accumulator_tiles_i32x4[2][1], query_2_i8x16, document_i8x16);
                accumulator_tiles_i32x4[3][1] = vdotq_s32(accumulator_tiles_i32x4[3][1], query_3_i8x16, document_i8x16);

                document_i8x16 = vld1q_s8(
                    (nk_i8_t const *)(document_i8 + (document_block_start_index + 2) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x4[0][2] = vdotq_s32(accumulator_tiles_i32x4[0][2], query_0_i8x16, document_i8x16);
                accumulator_tiles_i32x4[1][2] = vdotq_s32(accumulator_tiles_i32x4[1][2], query_1_i8x16, document_i8x16);
                accumulator_tiles_i32x4[2][2] = vdotq_s32(accumulator_tiles_i32x4[2][2], query_2_i8x16, document_i8x16);
                accumulator_tiles_i32x4[3][2] = vdotq_s32(accumulator_tiles_i32x4[3][2], query_3_i8x16, document_i8x16);

                document_i8x16 = vld1q_s8(
                    (nk_i8_t const *)(document_i8 + (document_block_start_index + 3) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x4[0][3] = vdotq_s32(accumulator_tiles_i32x4[0][3], query_0_i8x16, document_i8x16);
                accumulator_tiles_i32x4[1][3] = vdotq_s32(accumulator_tiles_i32x4[1][3], query_1_i8x16, document_i8x16);
                accumulator_tiles_i32x4[2][3] = vdotq_s32(accumulator_tiles_i32x4[2][3], query_2_i8x16, document_i8x16);
                accumulator_tiles_i32x4[3][3] = vdotq_s32(accumulator_tiles_i32x4[3][3], query_3_i8x16, document_i8x16);
            }

            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++)
                for (nk_size_t document_tile_index = 0; document_tile_index < 4; document_tile_index++)
                    dots[(query_block_start_index + query_tile_index) * document_count + document_block_start_index +
                         document_tile_index] =
                        vaddvq_s32(accumulator_tiles_i32x4[query_tile_index][document_tile_index]);
        }

        // Document tail: 4Q × 1D
        for (nk_size_t document_index = document_block_start_index; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;

            int32x4_t accumulator_0_i32x4 = vdupq_n_s32(0);
            int32x4_t accumulator_1_i32x4 = vdupq_n_s32(0);
            int32x4_t accumulator_2_i32x4 = vdupq_n_s32(0);
            int32x4_t accumulator_3_i32x4 = vdupq_n_s32(0);

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 16) {
                int8x16_t document_i8x16 = vld1q_s8((nk_i8_t const *)(document_i8_row + depth_index));

                accumulator_0_i32x4 = vdotq_s32(
                    accumulator_0_i32x4,
                    vld1q_s8(
                        (nk_i8_t const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                    document_i8x16);
                accumulator_1_i32x4 = vdotq_s32(
                    accumulator_1_i32x4,
                    vld1q_s8(
                        (nk_i8_t const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                    document_i8x16);
                accumulator_2_i32x4 = vdotq_s32(
                    accumulator_2_i32x4,
                    vld1q_s8(
                        (nk_i8_t const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                    document_i8x16);
                accumulator_3_i32x4 = vdotq_s32(
                    accumulator_3_i32x4,
                    vld1q_s8(
                        (nk_i8_t const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                    document_i8x16);
            }

            dots[(query_block_start_index + 0) * document_count + document_index] = vaddvq_s32(accumulator_0_i32x4);
            dots[(query_block_start_index + 1) * document_count + document_index] = vaddvq_s32(accumulator_1_i32x4);
            dots[(query_block_start_index + 2) * document_count + document_index] = vaddvq_s32(accumulator_2_i32x4);
            dots[(query_block_start_index + 3) * document_count + document_index] = vaddvq_s32(accumulator_3_i32x4);
        }
    }

    // Query tail: 1Q × 1D
    for (nk_size_t query_index = query_block_start_index; query_index < query_count; query_index++) {
        nk_i8_t const *query_i8_row = query_i8 + query_index * depth_i8_padded;

        for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            int32x4_t accumulator_i32x4 = vdupq_n_s32(0);

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 16) {
                int8x16_t query_i8x16 = vld1q_s8((nk_i8_t const *)(query_i8_row + depth_index));
                int8x16_t document_i8x16 = vld1q_s8((nk_i8_t const *)(document_i8_row + depth_index));
                accumulator_i32x4 = vdotq_s32(accumulator_i32x4, query_i8x16, document_i8x16);
            }

            dots[query_index * document_count + document_index] = vaddvq_s32(accumulator_i32x4);
        }
    }
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_bf16_neonsdot_(void const *query, void const *document, nk_size_t depth) {
    nk_f32_t dot;
    nk_dot_bf16_through_f32_neon_((nk_bf16_t const *)query, (nk_bf16_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f32_neonsdot_(void const *query, void const *document, nk_size_t depth) {
    nk_f64_t dot;
    nk_dot_f32_through_f64_neon_((nk_f32_t const *)query, (nk_f32_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f16_neonsdot_(void const *query, void const *document, nk_size_t depth) {
    nk_f32_t dot;
    nk_dot_f16_through_f32_neon_((nk_f16_t const *)query, (nk_f16_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_API nk_status_t nk_maxsim_packed_bf16_neonsdot( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_neonsdot_k) ||
        !nk_maxsim_packed_by_(document_packed, nk_cap_neonsdot_k))
        return nk_pack_mismatch_k;

    *result = (nk_f32_t)nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                                  nk_maxsim_coarse_dots_neonsdot_, nk_maxsim_refine_bf16_neonsdot_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f32_neonsdot( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_neonsdot_k) ||
        !nk_maxsim_packed_by_(document_packed, nk_cap_neonsdot_k))
        return nk_pack_mismatch_k;

    *result = nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                        nk_maxsim_coarse_dots_neonsdot_, nk_maxsim_refine_f32_neonsdot_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f16_neonsdot( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_neonsdot_k) ||
        !nk_maxsim_packed_by_(document_packed, nk_cap_neonsdot_k))
        return nk_pack_mismatch_k;

    *result = (nk_f32_t)nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                                  nk_maxsim_coarse_dots_neonsdot_, nk_maxsim_refine_f16_neonsdot_);
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_NEONSDOT
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_MAXSIM_NEONSDOT_H
