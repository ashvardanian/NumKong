/**
 *  @file include/numkong/maxsim/alder.h
 *  @author Ash Vardanian
 *  @date March 5, 2026
 *  @brief SIMD-accelerated MaxSim, angular distance late-interaction, for Alder Lake, AVX-VNNI.
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Uses AVX-VNNI VPDPBUSD, u8 × i8 → i32 with accumulate, for coarse i8 screening. Unlike Haswell's
 *  VPMADDUBSW+VPMADDWD, DPBUSD has no i16 intermediate, so no saturation concern. Quantization
 *  range [-127, 127], versus Haswell's [-79, 79], gives better precision. Bias correction via
 *  XOR-0x80 converts signed queries to unsigned, then subtracts 128 × sum_quantized.
 *
 *  4x4 register tiling: 4 queries × 4 documents = 16 YMM accumulators per depth loop. Depth steps
 *  at 32 bytes, the YMM width in bytes.
 */
#ifndef NUMKONG_MAXSIM_ALDER_H
#define NUMKONG_MAXSIM_ALDER_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_ALDER

#include "numkong/types.h"
#include "numkong/maxsim/serial.h"  // `nk_maxsim_packed_header_t`
#include "numkong/maxsim/haswell.h" // `nk_maxsim_reduce_i32x8x4_haswell_`
#include "numkong/dot/haswell.h"    // `nk_dot_bf16_through_f32_haswell_`, `nk_dot_f32_through_f64_haswell_`

/** On GCC/Clang, VEX encoding is handled by target attributes. Alias the MSVC-specific _avx
 *  intrinsic names to standard names. */
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

NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_alder(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_bf16_t), 32);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_alder(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                          void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_alder_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_alder(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_f32_t), 32);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_alder(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                         void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_alder_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_alder(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_f16_t), 32);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_alder(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                         void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_alder_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_bf16_alder( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_bf16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 32, element_bytes,
                                                               nk_cap_alder_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++)
        nk_maxsim_pack_vector_bf16_haswell_((nk_bf16_t const *)((char const *)vectors + vector_index * stride), depth,
                                            127.0f, (nk_bf16_t *)(originals + vector_index * original_stride),
                                            quantized_i8 + vector_index * depth_i8_padded, metadata + vector_index);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f32_alder( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_f32_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 32, element_bytes,
                                                               nk_cap_alder_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++)
        nk_maxsim_pack_vector_f32_haswell_((nk_f32_t const *)((char const *)vectors + vector_index * stride), depth,
                                           127.0f, (nk_f32_t *)(originals + vector_index * original_stride),
                                           quantized_i8 + vector_index * depth_i8_padded, metadata + vector_index);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f16_alder( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_f16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 32, element_bytes,
                                                               nk_cap_alder_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++)
        nk_maxsim_pack_vector_f16_haswell_((nk_f16_t const *)((char const *)vectors + vector_index * stride), depth,
                                           127.0f, (nk_f16_t *)(originals + vector_index * original_stride),
                                           quantized_i8 + vector_index * depth_i8_padded, metadata + vector_index);
    return nk_success_k;
}

/** Coarse i8 kernel for Alder Lake as an @c nk_maxsim_coarse_dots_t. One VPDPBUSD per query × doc
 *  pair, with no i16 intermediate and an XOR-0x80 bias, tiled 4Q × 4D over 16 YMM accumulators. */
NUMKONG_INLINE void nk_maxsim_coarse_dots_alder_(         //
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,  //
    nk_maxsim_vector_metadata_t const *document_metadata, //
    nk_size_t query_count, nk_size_t document_count,      //
    nk_size_t depth_i8_padded, nk_i32_t *dots) {

    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {

        // 4Q × 4D document blocking
        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= document_count; document_block_start_index += 4) {
            __m256i accumulator_tiles_i32x8[4][4];
            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++)
                for (nk_size_t document_tile_index = 0; document_tile_index < 4; document_tile_index++)
                    accumulator_tiles_i32x8[query_tile_index][document_tile_index] = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i query0_biased_u8x32 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);
                __m256i query1_biased_u8x32 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);
                __m256i query2_biased_u8x32 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);
                __m256i query3_biased_u8x32 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);

                __m256i document_i8x32;

                // Document 0
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 0) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x8[0][0] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[0][0],
                                                                        query0_biased_u8x32, document_i8x32);
                accumulator_tiles_i32x8[1][0] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[1][0],
                                                                        query1_biased_u8x32, document_i8x32);
                accumulator_tiles_i32x8[2][0] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[2][0],
                                                                        query2_biased_u8x32, document_i8x32);
                accumulator_tiles_i32x8[3][0] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[3][0],
                                                                        query3_biased_u8x32, document_i8x32);

                // Document 1
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 1) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x8[0][1] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[0][1],
                                                                        query0_biased_u8x32, document_i8x32);
                accumulator_tiles_i32x8[1][1] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[1][1],
                                                                        query1_biased_u8x32, document_i8x32);
                accumulator_tiles_i32x8[2][1] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[2][1],
                                                                        query2_biased_u8x32, document_i8x32);
                accumulator_tiles_i32x8[3][1] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[3][1],
                                                                        query3_biased_u8x32, document_i8x32);

                // Document 2
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 2) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x8[0][2] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[0][2],
                                                                        query0_biased_u8x32, document_i8x32);
                accumulator_tiles_i32x8[1][2] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[1][2],
                                                                        query1_biased_u8x32, document_i8x32);
                accumulator_tiles_i32x8[2][2] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[2][2],
                                                                        query2_biased_u8x32, document_i8x32);
                accumulator_tiles_i32x8[3][2] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[3][2],
                                                                        query3_biased_u8x32, document_i8x32);

                // Document 3
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 3) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x8[0][3] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[0][3],
                                                                        query0_biased_u8x32, document_i8x32);
                accumulator_tiles_i32x8[1][3] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[1][3],
                                                                        query1_biased_u8x32, document_i8x32);
                accumulator_tiles_i32x8[2][3] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[2][3],
                                                                        query2_biased_u8x32, document_i8x32);
                accumulator_tiles_i32x8[3][3] = _mm256_dpbusd_avx_epi32(accumulator_tiles_i32x8[3][3],
                                                                        query3_biased_u8x32, document_i8x32);
            }

            // Reduce each query's 4 doc accumulators → __m128i
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

            _mm_storeu_si128(
                (__m128i *)(dots + (query_block_start_index + 0) * document_count + document_block_start_index),
                query_0_coarse_dots_i32x4);
            _mm_storeu_si128(
                (__m128i *)(dots + (query_block_start_index + 1) * document_count + document_block_start_index),
                query_1_coarse_dots_i32x4);
            _mm_storeu_si128(
                (__m128i *)(dots + (query_block_start_index + 2) * document_count + document_block_start_index),
                query_2_coarse_dots_i32x4);
            _mm_storeu_si128(
                (__m128i *)(dots + (query_block_start_index + 3) * document_count + document_block_start_index),
                query_3_coarse_dots_i32x4);
        }

        // Document tail: 4Q × 1D
        for (nk_size_t document_index = document_block_start_index; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;

            __m256i accumulator0_i32x8 = _mm256_setzero_si256();
            __m256i accumulator1_i32x8 = _mm256_setzero_si256();
            __m256i accumulator2_i32x8 = _mm256_setzero_si256();
            __m256i accumulator3_i32x8 = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i document_i8x32 = _mm256_loadu_si256((__m256i const *)(document_i8_row + depth_index));

                accumulator0_i32x8 = _mm256_dpbusd_avx_epi32(
                    accumulator0_i32x8,
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);

                accumulator1_i32x8 = _mm256_dpbusd_avx_epi32(
                    accumulator1_i32x8,
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);

                accumulator2_i32x8 = _mm256_dpbusd_avx_epi32(
                    accumulator2_i32x8,
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);

                accumulator3_i32x8 = _mm256_dpbusd_avx_epi32(
                    accumulator3_i32x8,
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);
            }

            __m128i reduced_i32x4 = nk_maxsim_reduce_i32x8x4_haswell_(accumulator0_i32x8, accumulator1_i32x8,
                                                                      accumulator2_i32x8, accumulator3_i32x8);
            nk_i32_t bias_correction_i32 = 128 * document_metadata[document_index].sum_i8_i32;
            __m128i coarse_dots_i32x4 = _mm_sub_epi32(reduced_i32x4, _mm_set1_epi32(bias_correction_i32));
            dots[(query_block_start_index + 0) * document_count + document_index] = _mm_extract_epi32(coarse_dots_i32x4,
                                                                                                      0);
            dots[(query_block_start_index + 1) * document_count + document_index] = _mm_extract_epi32(coarse_dots_i32x4,
                                                                                                      1);
            dots[(query_block_start_index + 2) * document_count + document_index] = _mm_extract_epi32(coarse_dots_i32x4,
                                                                                                      2);
            dots[(query_block_start_index + 3) * document_count + document_index] = _mm_extract_epi32(coarse_dots_i32x4,
                                                                                                      3);
        }
    }

    // Query tail: 1Q × 1D
    for (nk_size_t query_index = query_block_start_index; query_index < query_count; query_index++) {
        nk_i8_t const *query_i8_row = query_i8 + query_index * depth_i8_padded;

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
            dots[query_index * document_count + document_index] = _mm_extract_epi32(sum_i32x4, 0) -
                                                                  128 * document_metadata[document_index].sum_i8_i32;
        }
    }
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_bf16_alder_(void const *query, void const *document, nk_size_t depth) {
    nk_f32_t dot;
    nk_dot_bf16_through_f32_haswell_((nk_bf16_t const *)query, (nk_bf16_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f32_alder_(void const *query, void const *document, nk_size_t depth) {
    nk_f64_t dot;
    nk_dot_f32_through_f64_haswell_((nk_f32_t const *)query, (nk_f32_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f16_alder_(void const *query, void const *document, nk_size_t depth) {
    nk_f32_t dot;
    nk_dot_f16_through_f32_haswell_((nk_f16_t const *)query, (nk_f16_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_API nk_status_t nk_maxsim_packed_bf16_alder( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_alder_k) || !nk_maxsim_packed_by_(document_packed, nk_cap_alder_k))
        return nk_pack_mismatch_k;

    *result = (nk_f32_t)nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                                  nk_maxsim_coarse_dots_alder_, nk_maxsim_refine_bf16_alder_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f32_alder( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_alder_k) || !nk_maxsim_packed_by_(document_packed, nk_cap_alder_k))
        return nk_pack_mismatch_k;

    *result = nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                        nk_maxsim_coarse_dots_alder_, nk_maxsim_refine_f32_alder_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f16_alder( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_alder_k) || !nk_maxsim_packed_by_(document_packed, nk_cap_alder_k))
        return nk_pack_mismatch_k;

    *result = (nk_f32_t)nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                                  nk_maxsim_coarse_dots_alder_, nk_maxsim_refine_f16_alder_);
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

#endif // NUMKONG_TARGET_ALDER
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_MAXSIM_ALDER_H
