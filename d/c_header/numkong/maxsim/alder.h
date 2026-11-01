/**
 *  @brief SIMD-accelerated MaxSim (angular distance late-interaction) for Alder Lake (AVX-VNNI).
 *  @file include/numkong/maxsim/alder.h
 *  @author Ash Vardanian
 *  @date March 5, 2026
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Uses AVX-VNNI VPDPBUSD (u8×i8→i32 with accumulate) for coarse i8 screening.
 *  Unlike Haswell's VPMADDUBSW+VPMADDWD, DPBUSD has no i16 intermediate, so no saturation concern.
 *  Quantization range [-127, 127] (vs Haswell's [-79, 79]) for better precision.
 *  Bias correction via XOR-0x80 converts signed queries to unsigned, then subtracts 128 × sum_quantized.
 *
 *  4x4 register tiling: 4 queries × 4 documents = 16 YMM accumulators per depth loop.
 *  Depth steps at 32 bytes (YMM width in bytes).
 */
#ifndef NK_MAXSIM_ALDER_H
#define NK_MAXSIM_ALDER_H

#if NK_TARGET_X86_
#if NK_TARGET_ALDER

#include "numkong/types.h"
#include "numkong/maxsim/serial.h"   // `nk_maxsim_packed_header_t`
#include "numkong/maxsim/haswell.h"  // `nk_maxsim_reduce_i32x8x4_haswell_`
#include "numkong/dot.h"             // `nk_dot_bf16`, `nk_dot_f32`, `nk_dot_f16`
#include "numkong/cast/haswell.h"    // `nk_f16_to_f32_haswell`
#include "numkong/spatial/haswell.h" // `nk_f32_sqrt_haswell`

// On GCC/Clang, VEX encoding is handled by target attributes.
// Alias the MSVC-specific _avx intrinsic names to standard names.
#if !defined(_MSC_VER) && !defined(_mm256_dpbusd_avx_epi32)
#define _mm256_dpbusd_avx_epi32 _mm256_dpbusd_epi32
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2,avxvnni"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2", "avxvnni")
#endif

NK_PUBLIC nk_size_t nk_maxsim_packed_size_bf16_alder(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_bf16_t), 32);
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f32_alder(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_f32_t), 32);
}

NK_PUBLIC nk_size_t nk_maxsim_packed_size_f16_alder(nk_size_t vector_count, nk_size_t depth) {
    return nk_maxsim_packed_size_(vector_count, depth, sizeof(nk_f16_t), 32);
}

NK_PUBLIC void nk_maxsim_pack_bf16_alder( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_bf16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 32, element_bytes);

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
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? (1.0f / nk_f32_sqrt_haswell(norm_sq)) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

NK_PUBLIC void nk_maxsim_pack_f32_alder( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_f32_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 32, element_bytes);

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
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? (1.0f / nk_f32_sqrt_haswell(norm_sq)) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

NK_PUBLIC void nk_maxsim_pack_f16_alder( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed) {

    nk_size_t const element_bytes = sizeof(nk_f16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 32, element_bytes);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride_bytes;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        char const *source_row = (char const *)vectors + vector_index * stride;
        nk_f32_t norm_sq;
        nk_maxsim_quantize_vector_(source_row, element_bytes, depth, depth_i8_padded, 127.0f,
                                   (nk_maxsim_to_f32_t)nk_f16_to_f32_haswell,
                                   &quantized_i8[vector_index * depth_i8_padded], &metadata[vector_index], &norm_sq);
        metadata[vector_index].inverse_norm_f32 = norm_sq > 0.0f ? (1.0f / nk_f32_sqrt_haswell(norm_sq)) : 0.0f;
        char *destination_original = originals + vector_index * original_stride;
        nk_copy_bytes_(destination_original, source_row, depth * element_bytes);
        for (nk_size_t byte_index = depth * element_bytes; byte_index < original_stride; byte_index++)
            destination_original[byte_index] = 0;
    }
}

/**
 *  @brief Factored coarse i8 argmax kernel for Alder Lake using DPBUSD.
 *  Uses single VPDPBUSD instruction per query×doc pair (no i16 intermediate).
 *  4Q×4D register tiling with 16 YMM accumulators.
 */
NK_INTERNAL void nk_maxsim_coarse_argmax_alder_(          //
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,  //
    nk_maxsim_vector_metadata_t const *document_metadata, //
    nk_size_t query_count, nk_size_t document_count,      //
    nk_size_t depth_i8_padded, nk_u32_t *best_document_indices) {

    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {
        __m128i running_max_i32x4 = _mm_set1_epi32(NK_I32_MIN);
        __m128i running_argmax_i32x4 = _mm_setzero_si128();

        // 4Q×4D document blocking
        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= document_count; document_block_start_index += 4) {
            __m256i accumulator_tiles_i32x8[4][4];
            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++)
                for (nk_size_t document_tile_index = 0; document_tile_index < 4; document_tile_index++)
                    accumulator_tiles_i32x8[query_tile_index][document_tile_index] = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i query_biased_u8x32_0 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);
                __m256i query_biased_u8x32_1 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);
                __m256i query_biased_u8x32_2 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);
                __m256i query_biased_u8x32_3 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);

                __m256i document_i8x32;

                // Document 0
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 0) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x8[0][0] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[0][0],
                                                                        query_biased_u8x32_0, document_i8x32);
                accumulator_tiles_i32x8[1][0] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[1][0],
                                                                        query_biased_u8x32_1, document_i8x32);
                accumulator_tiles_i32x8[2][0] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[2][0],
                                                                        query_biased_u8x32_2, document_i8x32);
                accumulator_tiles_i32x8[3][0] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[3][0],
                                                                        query_biased_u8x32_3, document_i8x32);

                // Document 1
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 1) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x8[0][1] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[0][1],
                                                                        query_biased_u8x32_0, document_i8x32);
                accumulator_tiles_i32x8[1][1] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[1][1],
                                                                        query_biased_u8x32_1, document_i8x32);
                accumulator_tiles_i32x8[2][1] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[2][1],
                                                                        query_biased_u8x32_2, document_i8x32);
                accumulator_tiles_i32x8[3][1] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[3][1],
                                                                        query_biased_u8x32_3, document_i8x32);

                // Document 2
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 2) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x8[0][2] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[0][2],
                                                                        query_biased_u8x32_0, document_i8x32);
                accumulator_tiles_i32x8[1][2] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[1][2],
                                                                        query_biased_u8x32_1, document_i8x32);
                accumulator_tiles_i32x8[2][2] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[2][2],
                                                                        query_biased_u8x32_2, document_i8x32);
                accumulator_tiles_i32x8[3][2] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[3][2],
                                                                        query_biased_u8x32_3, document_i8x32);

                // Document 3
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 3) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x8[0][3] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[0][3],
                                                                        query_biased_u8x32_0, document_i8x32);
                accumulator_tiles_i32x8[1][3] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[1][3],
                                                                        query_biased_u8x32_1, document_i8x32);
                accumulator_tiles_i32x8[2][3] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[2][3],
                                                                        query_biased_u8x32_2, document_i8x32);
                accumulator_tiles_i32x8[3][3] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[3][3],
                                                                        query_biased_u8x32_3, document_i8x32);
            }

            // Reduce each query's 4 doc accumulators -> __m128i
            __m128i query_0_coarse_dots_i32x4 = nk_maxsim_reduce_i32x8x4_haswell_(
                accumulator_tiles_i32x8[0][0], accumulator_tiles_i32x8[0][1], accumulator_tiles_i32x8[0][2],
                accumulator_tiles_i32x8[0][3]);
            __m128i query_1_coarse_dots_i32x4 = nk_maxsim_reduce_i32x8x4_haswell_(
                accumulator_tiles_i32x8[1][0], accumulator_tiles_i32x8[1][1], accumulator_tiles_i32x8[1][2],
                accumulator_tiles_i32x8[1][3]);
            __m128i query_2_coarse_dots_i32x4 = nk_maxsim_reduce_i32x8x4_haswell_(
                accumulator_tiles_i32x8[2][0], accumulator_tiles_i32x8[2][1], accumulator_tiles_i32x8[2][2],
                accumulator_tiles_i32x8[2][3]);
            __m128i query_3_coarse_dots_i32x4 = nk_maxsim_reduce_i32x8x4_haswell_(
                accumulator_tiles_i32x8[3][0], accumulator_tiles_i32x8[3][1], accumulator_tiles_i32x8[3][2],
                accumulator_tiles_i32x8[3][3]);

            // Bias correction: subtract 128 × sum_quantized for each document
            __m128i bias_correction_i32x4 = _mm_set_epi32(
                128 * document_metadata[document_block_start_index + 3].sum_i8_i32,
                128 * document_metadata[document_block_start_index + 2].sum_i8_i32,
                128 * document_metadata[document_block_start_index + 1].sum_i8_i32,
                128 * document_metadata[document_block_start_index + 0].sum_i8_i32);
            query_0_coarse_dots_i32x4 = _mm_sub_epi32(query_0_coarse_dots_i32x4, bias_correction_i32x4);
            query_1_coarse_dots_i32x4 = _mm_sub_epi32(query_1_coarse_dots_i32x4, bias_correction_i32x4);
            query_2_coarse_dots_i32x4 = _mm_sub_epi32(query_2_coarse_dots_i32x4, bias_correction_i32x4);
            query_3_coarse_dots_i32x4 = _mm_sub_epi32(query_3_coarse_dots_i32x4, bias_correction_i32x4);

            // 4x4 transpose: [query][doc] -> [doc][query] for vectorized argmax
            __m128i transpose_queries_01_low_i32x4 = _mm_unpacklo_epi32(query_0_coarse_dots_i32x4,
                                                                        query_1_coarse_dots_i32x4);
            __m128i transpose_queries_23_low_i32x4 = _mm_unpacklo_epi32(query_2_coarse_dots_i32x4,
                                                                        query_3_coarse_dots_i32x4);
            __m128i transpose_queries_01_high_i32x4 = _mm_unpackhi_epi32(query_0_coarse_dots_i32x4,
                                                                         query_1_coarse_dots_i32x4);
            __m128i transpose_queries_23_high_i32x4 = _mm_unpackhi_epi32(query_2_coarse_dots_i32x4,
                                                                         query_3_coarse_dots_i32x4);
            __m128i document_0_dots_i32x4 = _mm_unpacklo_epi64(transpose_queries_01_low_i32x4,
                                                               transpose_queries_23_low_i32x4);
            __m128i document_1_dots_i32x4 = _mm_unpackhi_epi64(transpose_queries_01_low_i32x4,
                                                               transpose_queries_23_low_i32x4);
            __m128i document_2_dots_i32x4 = _mm_unpacklo_epi64(transpose_queries_01_high_i32x4,
                                                               transpose_queries_23_high_i32x4);
            __m128i document_3_dots_i32x4 = _mm_unpackhi_epi64(transpose_queries_01_high_i32x4,
                                                               transpose_queries_23_high_i32x4);

            // Branchless SIMD argmax
            __m128i comparison_mask_i32x4, document_index_i32x4;

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_0_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 0));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_0_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_1_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 1));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_1_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_2_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 2));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_2_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);

            comparison_mask_i32x4 = _mm_cmpgt_epi32(document_3_dots_i32x4, running_max_i32x4);
            document_index_i32x4 = _mm_set1_epi32((int)(document_block_start_index + 3));
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, document_3_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);
        }

        // Document tail: 4Q×1D
        for (nk_size_t document_index = document_block_start_index; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;

            __m256i accumulator_i32x8_0 = _mm256_setzero_si256();
            __m256i accumulator_i32x8_1 = _mm256_setzero_si256();
            __m256i accumulator_i32x8_2 = _mm256_setzero_si256();
            __m256i accumulator_i32x8_3 = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i document_i8x32 = _mm256_loadu_si256((__m256i const *)(document_i8_row + depth_index));

                accumulator_i32x8_0 = _mm256_dpbusd_avx_epi32(
                    accumulator_i32x8_0,
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);

                accumulator_i32x8_1 = _mm256_dpbusd_avx_epi32(
                    accumulator_i32x8_1,
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);

                accumulator_i32x8_2 = _mm256_dpbusd_avx_epi32(
                    accumulator_i32x8_2,
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);

                accumulator_i32x8_3 = _mm256_dpbusd_avx_epi32(
                    accumulator_i32x8_3,
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);
            }

            __m128i reduced_i32x4 = nk_maxsim_reduce_i32x8x4_haswell_(accumulator_i32x8_0, accumulator_i32x8_1,
                                                                      accumulator_i32x8_2, accumulator_i32x8_3);
            nk_i32_t bias_correction_i32 = 128 * document_metadata[document_index].sum_i8_i32;
            __m128i coarse_dots_i32x4 = _mm_sub_epi32(reduced_i32x4, _mm_set1_epi32(bias_correction_i32));

            __m128i comparison_mask_i32x4 = _mm_cmpgt_epi32(coarse_dots_i32x4, running_max_i32x4);
            __m128i document_index_i32x4 = _mm_set1_epi32((int)document_index);
            running_max_i32x4 = _mm_blendv_epi8(running_max_i32x4, coarse_dots_i32x4, comparison_mask_i32x4);
            running_argmax_i32x4 = _mm_blendv_epi8(running_argmax_i32x4, document_index_i32x4, comparison_mask_i32x4);
        }

        best_document_indices[query_block_start_index + 0] = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 0);
        best_document_indices[query_block_start_index + 1] = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 1);
        best_document_indices[query_block_start_index + 2] = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 2);
        best_document_indices[query_block_start_index + 3] = (nk_u32_t)_mm_extract_epi32(running_argmax_i32x4, 3);
    }

    // Query tail: 1Q×1D
    for (nk_size_t query_index = query_block_start_index; query_index < query_count; query_index++) {
        nk_i8_t const *query_i8_row = query_i8 + query_index * depth_i8_padded;
        nk_i32_t running_max_i32 = NK_I32_MIN;
        nk_u32_t running_argmax_u32 = 0;

        for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            __m256i accumulator_i32x8 = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i document_i8x32 = _mm256_loadu_si256((__m256i const *)(document_i8_row + depth_index));
                __m256i query_biased_u8x32 = _mm256_xor_si256(
                    _mm256_loadu_si256((__m256i const *)(query_i8_row + depth_index)), xor_mask_u8x32);
                accumulator_i32x8 = _mm256_dpbusd_avx_epi32(accumulator_i32x8, query_biased_u8x32, document_i8x32);
            }

            // Horizontal sum of 8 i32 lanes
            __m128i sum_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(accumulator_i32x8),
                                              _mm256_extracti128_si256(accumulator_i32x8, 1));
            sum_i32x4 = _mm_add_epi32(sum_i32x4, _mm_shuffle_epi32(sum_i32x4, 0x4E)); // 01001110
            sum_i32x4 = _mm_add_epi32(sum_i32x4, _mm_shuffle_epi32(sum_i32x4, 0xB1)); // 10110001
            nk_i32_t coarse_dot_i32 = _mm_extract_epi32(sum_i32x4, 0) -
                                      128 * document_metadata[document_index].sum_i8_i32;

            if (coarse_dot_i32 > running_max_i32) {
                running_max_i32 = coarse_dot_i32;
                running_argmax_u32 = (nk_u32_t)document_index;
            }
        }

        best_document_indices[query_index] = running_argmax_u32;
    }
}

NK_PUBLIC void nk_maxsim_packed_bf16_alder( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_alder_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                       regions.document_quantized, regions.document_metadata, chunk_size,
                                       document_count, regions.depth_i8_padded, best_document_indices);

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

NK_PUBLIC void nk_maxsim_packed_f32_alder( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_alder_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                       regions.document_quantized, regions.document_metadata, chunk_size,
                                       document_count, regions.depth_i8_padded, best_document_indices);

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

NK_PUBLIC void nk_maxsim_packed_f16_alder( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result) {

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_(query_packed, document_packed);
    nk_f64_t total_angular_distance = 0.0;

    for (nk_size_t chunk_start = 0; chunk_start < query_count; chunk_start += 256) {
        nk_size_t chunk_size = query_count - chunk_start < 256 ? query_count - chunk_start : 256;
        nk_u32_t best_document_indices[256];

        nk_maxsim_coarse_argmax_alder_(regions.query_quantized + chunk_start * regions.depth_i8_padded,
                                       regions.document_quantized, regions.document_metadata, chunk_size,
                                       document_count, regions.depth_i8_padded, best_document_indices);

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

#endif // NK_TARGET_ALDER
#endif // NK_TARGET_X86_
#endif // NK_MAXSIM_ALDER_H
