/**
 *  @file include/numkong/maxsim/haswell.h
 *  @author Ash Vardanian
 *  @date February 28, 2026
 *  @brief SIMD-accelerated MaxSim, angular distance late-interaction, for Haswell, AVX2.
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Uses AVX2 VPMADDUBSW, u8 × i8 → i16, plus VPMADDWD, i16 → i32, for coarse i8 screening.
 *  Quantization range [-79, 79] ensures no i16 saturation, as the worst pair sum is 2 × 207 × 79 =
 *  32706, under the 32767 i16 ceiling. Bias correction via XOR-0x80 converts signed queries to
 *  unsigned, then subtracts 128 × sum_quantized.
 *
 *  4x4 register tiling: 4 queries × 4 documents = 16 YMM accumulators per depth loop. Depth steps
 *  at 32 bytes, the YMM width in bytes.
 */
#ifndef NUMKONG_MAXSIM_HASWELL_H
#define NUMKONG_MAXSIM_HASWELL_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_HASWELL_

#include "numkong/types.h"
#include "numkong/maxsim/serial.h" // `nk_maxsim_packed_header_t`
#include "numkong/dot/haswell.h"   // `nk_dot_bf16_through_f32_haswell_`, `nk_dot_f32_through_f64_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

/** Reduces 4 YMM i32x8 accumulators to a single __m128i with 4 horizontal sums. */
NUMKONG_INLINE __m128i nk_maxsim_reduce_i32x8x4_haswell_(     //
    __m256i accumulator_a_i32x8, __m256i accumulator_b_i32x8, //
    __m256i accumulator_c_i32x8, __m256i accumulator_d_i32x8) {
    // 8 → 4 (extract high 128-bit half and add to low half)
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(accumulator_a_i32x8),
                                        _mm256_extracti128_si256(accumulator_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(accumulator_b_i32x8),
                                        _mm256_extracti128_si256(accumulator_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(accumulator_c_i32x8),
                                        _mm256_extracti128_si256(accumulator_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(accumulator_d_i32x8),
                                        _mm256_extracti128_si256(accumulator_d_i32x8, 1));
    // 4x4 transpose + reduce → [sum_a, sum_b, sum_c, sum_d]
    __m128i transpose_ab_low_i32x4 = _mm_unpacklo_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_low_i32x4 = _mm_unpacklo_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i transpose_ab_high_i32x4 = _mm_unpackhi_epi32(sum_a_i32x4, sum_b_i32x4);
    __m128i transpose_cd_high_i32x4 = _mm_unpackhi_epi32(sum_c_i32x4, sum_d_i32x4);
    __m128i sum_lane_0_i32x4 = _mm_unpacklo_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane_1_i32x4 = _mm_unpackhi_epi64(transpose_ab_low_i32x4, transpose_cd_low_i32x4);
    __m128i sum_lane_2_i32x4 = _mm_unpacklo_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    __m128i sum_lane_3_i32x4 = _mm_unpackhi_epi64(transpose_ab_high_i32x4, transpose_cd_high_i32x4);
    return _mm_add_epi32(_mm_add_epi32(sum_lane_0_i32x4, sum_lane_1_i32x4),
                         _mm_add_epi32(sum_lane_2_i32x4, sum_lane_3_i32x4));
}

/** Coarse i8 kernel for Haswell: AVX2 VPMADDUBSW (u8 × i8 → i16) and VPMADDWD (i16 × 1 → i32) with
 *  an XOR-0x80 bias, tiled 4Q × 4D over 16 YMM accumulators. */
NUMKONG_INLINE void nk_maxsim_coarse_dots_haswell_(       //
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,  //
    nk_maxsim_vector_metadata_t const *document_metadata, //
    nk_size_t query_count, nk_size_t document_count,      //
    nk_size_t depth_i8_padded, nk_i32_t *dots) {

    __m256i const xor_mask_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const ones_i16x16 = _mm256_set1_epi16(1);

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {
        nk_i32_t *query_dots = dots + query_block_start_index * document_count;

        // 4Q × 4D document blocking
        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= document_count; document_block_start_index += 4) {
            __m256i accumulator_tiles_i32x8[4][4];
            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++)
                for (nk_size_t document_tile_index = 0; document_tile_index < 4; document_tile_index++)
                    accumulator_tiles_i32x8[query_tile_index][document_tile_index] = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i query_biased_0_u8x32 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);
                __m256i query_biased_1_u8x32 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);
                __m256i query_biased_2_u8x32 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);
                __m256i query_biased_3_u8x32 = _mm256_xor_si256(
                    _mm256_loadu_si256(
                        (__m256i const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                    xor_mask_u8x32);

                __m256i document_i8x32, products_i16x16, products_i32x8;

                // Document 0
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 0) * depth_i8_padded + depth_index));
                products_i16x16 = _mm256_maddubs_epi16(query_biased_0_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[0][0] = _mm256_add_epi32(accumulator_tiles_i32x8[0][0], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_1_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[1][0] = _mm256_add_epi32(accumulator_tiles_i32x8[1][0], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_2_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[2][0] = _mm256_add_epi32(accumulator_tiles_i32x8[2][0], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_3_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[3][0] = _mm256_add_epi32(accumulator_tiles_i32x8[3][0], products_i32x8);

                // Document 1
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 1) * depth_i8_padded + depth_index));
                products_i16x16 = _mm256_maddubs_epi16(query_biased_0_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[0][1] = _mm256_add_epi32(accumulator_tiles_i32x8[0][1], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_1_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[1][1] = _mm256_add_epi32(accumulator_tiles_i32x8[1][1], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_2_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[2][1] = _mm256_add_epi32(accumulator_tiles_i32x8[2][1], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_3_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[3][1] = _mm256_add_epi32(accumulator_tiles_i32x8[3][1], products_i32x8);

                // Document 2
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 2) * depth_i8_padded + depth_index));
                products_i16x16 = _mm256_maddubs_epi16(query_biased_0_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[0][2] = _mm256_add_epi32(accumulator_tiles_i32x8[0][2], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_1_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[1][2] = _mm256_add_epi32(accumulator_tiles_i32x8[1][2], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_2_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[2][2] = _mm256_add_epi32(accumulator_tiles_i32x8[2][2], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_3_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[3][2] = _mm256_add_epi32(accumulator_tiles_i32x8[3][2], products_i32x8);

                // Document 3
                document_i8x32 = _mm256_loadu_si256(
                    (__m256i const *)(document_i8 + (document_block_start_index + 3) * depth_i8_padded + depth_index));
                products_i16x16 = _mm256_maddubs_epi16(query_biased_0_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[0][3] = _mm256_add_epi32(accumulator_tiles_i32x8[0][3], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_1_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[1][3] = _mm256_add_epi32(accumulator_tiles_i32x8[1][3], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_2_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[2][3] = _mm256_add_epi32(accumulator_tiles_i32x8[2][3], products_i32x8);
                products_i16x16 = _mm256_maddubs_epi16(query_biased_3_u8x32, document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_tiles_i32x8[3][3] = _mm256_add_epi32(accumulator_tiles_i32x8[3][3], products_i32x8);
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

            _mm_storeu_si128((__m128i *)(query_dots + 0 * document_count + document_block_start_index),
                             query_0_coarse_dots_i32x4);
            _mm_storeu_si128((__m128i *)(query_dots + 1 * document_count + document_block_start_index),
                             query_1_coarse_dots_i32x4);
            _mm_storeu_si128((__m128i *)(query_dots + 2 * document_count + document_block_start_index),
                             query_2_coarse_dots_i32x4);
            _mm_storeu_si128((__m128i *)(query_dots + 3 * document_count + document_block_start_index),
                             query_3_coarse_dots_i32x4);
        }

        // Document tail: 4Q × 1D
        for (nk_size_t document_index = document_block_start_index; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;

            __m256i accumulator_0_i32x8 = _mm256_setzero_si256();
            __m256i accumulator_1_i32x8 = _mm256_setzero_si256();
            __m256i accumulator_2_i32x8 = _mm256_setzero_si256();
            __m256i accumulator_3_i32x8 = _mm256_setzero_si256();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 32) {
                __m256i document_i8x32 = _mm256_loadu_si256((__m256i const *)(document_i8_row + depth_index));
                __m256i products_i16x16, products_i32x8;

                products_i16x16 = _mm256_maddubs_epi16(
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_0_i32x8 = _mm256_add_epi32(accumulator_0_i32x8, products_i32x8);

                products_i16x16 = _mm256_maddubs_epi16(
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_1_i32x8 = _mm256_add_epi32(accumulator_1_i32x8, products_i32x8);

                products_i16x16 = _mm256_maddubs_epi16(
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_2_i32x8 = _mm256_add_epi32(accumulator_2_i32x8, products_i32x8);

                products_i16x16 = _mm256_maddubs_epi16(
                    _mm256_xor_si256(
                        _mm256_loadu_si256((
                            __m256i const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                        xor_mask_u8x32),
                    document_i8x32);
                products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_3_i32x8 = _mm256_add_epi32(accumulator_3_i32x8, products_i32x8);
            }

            __m128i reduced_i32x4 = nk_maxsim_reduce_i32x8x4_haswell_(accumulator_0_i32x8, accumulator_1_i32x8,
                                                                      accumulator_2_i32x8, accumulator_3_i32x8);
            nk_i32_t bias_correction_i32 = 128 * document_metadata[document_index].sum_i8_i32;
            __m128i coarse_dots_i32x4 = _mm_sub_epi32(reduced_i32x4, _mm_set1_epi32(bias_correction_i32));
            query_dots[0 * document_count + document_index] = _mm_extract_epi32(coarse_dots_i32x4, 0);
            query_dots[1 * document_count + document_index] = _mm_extract_epi32(coarse_dots_i32x4, 1);
            query_dots[2 * document_count + document_index] = _mm_extract_epi32(coarse_dots_i32x4, 2);
            query_dots[3 * document_count + document_index] = _mm_extract_epi32(coarse_dots_i32x4, 3);
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
                __m256i products_i16x16 = _mm256_maddubs_epi16(query_biased_u8x32, document_i8x32);
                __m256i products_i32x8 = _mm256_madd_epi16(products_i16x16, ones_i16x16);
                accumulator_i32x8 = _mm256_add_epi32(accumulator_i32x8, products_i32x8);
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

/*  Pack helpers: pass 1 reads the source once, copies it into the original row and measures it;
 *  pass 2 quantizes from that original row. Codes are summed with VPSADBW on biased bytes, and
 *  @p scale_limit maps the absolute maximum onto the largest code. */

/** Quantizes 8 F32 values by true division with @p scale_f32x8, rounding half away from zero and
 *  clamping magnitudes to @p limit_i32x8. */
NUMKONG_INLINE __m256i nk_maxsim_quantize_f32x8_haswell_(__m256 values_f32x8, __m256 scale_f32x8, __m256i limit_i32x8) {
    __m256 const scaled_f32x8 = _mm256_div_ps(values_f32x8, scale_f32x8);
    __m256 const half_f32x8 = _mm256_or_ps(_mm256_and_ps(scaled_f32x8, _mm256_set1_ps(-0.0f)), _mm256_set1_ps(0.5f));
    __m256i const rounded_i32x8 = _mm256_cvttps_epi32(_mm256_add_ps(scaled_f32x8, half_f32x8));
    return _mm256_max_epi32(_mm256_min_epi32(rounded_i32x8, limit_i32x8),
                            _mm256_sub_epi32(_mm256_setzero_si256(), limit_i32x8));
}

/** Accumulates 8 F32 values into the running absolute maximum, skipping NaNs, and into an F64
 *  sum of squares, each square exact in F64. */
NUMKONG_INLINE void nk_maxsim_measure_f32x8_haswell_(__m256 values_f32x8, __m256 *absmax_f32x8,
                                                     __m256d *sumsq_low_f64x4, __m256d *sumsq_high_f64x4) {
    __m256 const magnitudes_f32x8 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), values_f32x8);
    *absmax_f32x8 = _mm256_max_ps(magnitudes_f32x8, *absmax_f32x8);
    __m256d const low_f64x4 = _mm256_cvtps_pd(_mm256_castps256_ps128(values_f32x8));
    __m256d const high_f64x4 = _mm256_cvtps_pd(_mm256_extractf128_ps(values_f32x8, 1));
    *sumsq_low_f64x4 = _mm256_fmadd_pd(low_f64x4, low_f64x4, *sumsq_low_f64x4);
    *sumsq_high_f64x4 = _mm256_fmadd_pd(high_f64x4, high_f64x4, *sumsq_high_f64x4);
}

/** Narrows four vectors of I32 codes, in order, to 32 I8 codes. */
NUMKONG_INLINE __m256i nk_maxsim_narrow_i32x8x4_haswell_(__m256i first_i32x8, __m256i second_i32x8, __m256i third_i32x8,
                                                         __m256i fourth_i32x8) {
    // Packing spreads each vector's dwords across both 128-bit halves, which the permute undoes
    __m256i const codes_i8x32 = _mm256_packs_epi16(_mm256_packs_epi32(first_i32x8, second_i32x8),
                                                   _mm256_packs_epi32(third_i32x8, fourth_i32x8));
    return _mm256_permutevar8x32_epi32(codes_i8x32, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

/** Packs one BF16 vector into @p original and @p quantized, filling @p metadata. */
NUMKONG_INLINE void nk_maxsim_pack_vector_bf16_haswell_(nk_bf16_t const *source, nk_size_t depth, nk_f32_t scale_limit,
                                                        nk_bf16_t *original, nk_i8_t *quantized,
                                                        nk_maxsim_vector_metadata_t *metadata) {
    __m256 absmax_f32x8 = _mm256_setzero_ps();
    __m256d sumsq_first_f64x4 = _mm256_setzero_pd(), sumsq_second_f64x4 = _mm256_setzero_pd();
    __m256d sumsq_third_f64x4 = _mm256_setzero_pd(), sumsq_fourth_f64x4 = _mm256_setzero_pd();
    for (nk_size_t index = 0; index < depth; index += 16) {
        nk_size_t const chunk = depth - index < 16 ? depth - index : 16;
        nk_b256_vec_t raw_vec;
        nk_partial_load_b16x16_haswell_(source + index, &raw_vec, chunk);
        nk_partial_store_b16x16_haswell_(&raw_vec, original + index, chunk);
        nk_maxsim_measure_f32x8_haswell_(nk_bf16x8_to_f32x8_haswell_(raw_vec.xmms[0]), &absmax_f32x8,
                                         &sumsq_first_f64x4, &sumsq_second_f64x4);
        nk_maxsim_measure_f32x8_haswell_(nk_bf16x8_to_f32x8_haswell_(raw_vec.xmms[1]), &absmax_f32x8,
                                         &sumsq_third_f64x4, &sumsq_fourth_f64x4);
    }
    nk_f64_t const norm_squared = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(
        _mm256_add_pd(sumsq_first_f64x4, sumsq_second_f64x4), _mm256_add_pd(sumsq_third_f64x4, sumsq_fourth_f64x4)));
    nk_f32_t const scale = nk_maxsim_scale_f32_serial_(nk_reduce_max_f32x8_haswell_(absmax_f32x8), scale_limit);

    // Original rows span multiples of 32 values, zero past the depth, so whole loads stay inside
    __m256 const scale_f32x8 = _mm256_set1_ps(scale);
    __m256i const limit_i32x8 = _mm256_set1_epi32((int)scale_limit);
    __m256i const bias_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i sum_u64x4 = _mm256_setzero_si256();
    for (nk_size_t index = 0; index < depth; index += 32) {
        __m256i const first_i32x8 = nk_maxsim_quantize_f32x8_haswell_(
            nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(original + index))), scale_f32x8,
            limit_i32x8);
        __m256i const second_i32x8 = nk_maxsim_quantize_f32x8_haswell_(
            nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(original + index + 8))), scale_f32x8,
            limit_i32x8);
        __m256i const third_i32x8 = nk_maxsim_quantize_f32x8_haswell_(
            nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(original + index + 16))), scale_f32x8,
            limit_i32x8);
        __m256i const fourth_i32x8 = nk_maxsim_quantize_f32x8_haswell_(
            nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)(original + index + 24))), scale_f32x8,
            limit_i32x8);
        nk_b256_vec_t codes_vec;
        codes_vec.ymm = nk_maxsim_narrow_i32x8x4_haswell_(first_i32x8, second_i32x8, third_i32x8, fourth_i32x8);
        nk_partial_store_b8x32_haswell_(&codes_vec, quantized + index, depth - index < 32 ? depth - index : 32);
        sum_u64x4 = _mm256_add_epi64(
            sum_u64x4, _mm256_sad_epu8(_mm256_xor_si256(codes_vec.ymm, bias_u8x32), _mm256_setzero_si256()));
    }
    // Every lane summed carries the 128 bias, the zero codes past the depth included
    nk_i64_t const lanes = (nk_i64_t)nk_size_round_up_to_multiple_(depth, 32);
    nk_maxsim_vector_metadata_serial_(scale, norm_squared,
                                      (nk_i32_t)(nk_reduce_add_i64x4_haswell_(sum_u64x4) - 128 * lanes), metadata);
}

/** Packs one F16 vector into @p original and @p quantized, filling @p metadata. */
NUMKONG_INLINE void nk_maxsim_pack_vector_f16_haswell_(nk_f16_t const *source, nk_size_t depth, nk_f32_t scale_limit,
                                                       nk_f16_t *original, nk_i8_t *quantized,
                                                       nk_maxsim_vector_metadata_t *metadata) {
    __m256 absmax_f32x8 = _mm256_setzero_ps();
    __m256d sumsq_first_f64x4 = _mm256_setzero_pd(), sumsq_second_f64x4 = _mm256_setzero_pd();
    __m256d sumsq_third_f64x4 = _mm256_setzero_pd(), sumsq_fourth_f64x4 = _mm256_setzero_pd();
    for (nk_size_t index = 0; index < depth; index += 16) {
        nk_size_t const chunk = depth - index < 16 ? depth - index : 16;
        nk_b256_vec_t raw_vec;
        nk_partial_load_b16x16_haswell_(source + index, &raw_vec, chunk);
        nk_partial_store_b16x16_haswell_(&raw_vec, original + index, chunk);
        nk_maxsim_measure_f32x8_haswell_(_mm256_cvtph_ps(raw_vec.xmms[0]), &absmax_f32x8, &sumsq_first_f64x4,
                                         &sumsq_second_f64x4);
        nk_maxsim_measure_f32x8_haswell_(_mm256_cvtph_ps(raw_vec.xmms[1]), &absmax_f32x8, &sumsq_third_f64x4,
                                         &sumsq_fourth_f64x4);
    }
    nk_f64_t const norm_squared = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(
        _mm256_add_pd(sumsq_first_f64x4, sumsq_second_f64x4), _mm256_add_pd(sumsq_third_f64x4, sumsq_fourth_f64x4)));
    nk_f32_t const scale = nk_maxsim_scale_f32_serial_(nk_reduce_max_f32x8_haswell_(absmax_f32x8), scale_limit);

    // Original rows span multiples of 32 values, zero past the depth, so whole loads stay inside
    __m256 const scale_f32x8 = _mm256_set1_ps(scale);
    __m256i const limit_i32x8 = _mm256_set1_epi32((int)scale_limit);
    __m256i const bias_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i sum_u64x4 = _mm256_setzero_si256();
    for (nk_size_t index = 0; index < depth; index += 32) {
        __m256i const first_i32x8 = nk_maxsim_quantize_f32x8_haswell_(
            _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(original + index))), scale_f32x8, limit_i32x8);
        __m256i const second_i32x8 = nk_maxsim_quantize_f32x8_haswell_(
            _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(original + index + 8))), scale_f32x8, limit_i32x8);
        __m256i const third_i32x8 = nk_maxsim_quantize_f32x8_haswell_(
            _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(original + index + 16))), scale_f32x8, limit_i32x8);
        __m256i const fourth_i32x8 = nk_maxsim_quantize_f32x8_haswell_(
            _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(original + index + 24))), scale_f32x8, limit_i32x8);
        nk_b256_vec_t codes_vec;
        codes_vec.ymm = nk_maxsim_narrow_i32x8x4_haswell_(first_i32x8, second_i32x8, third_i32x8, fourth_i32x8);
        nk_partial_store_b8x32_haswell_(&codes_vec, quantized + index, depth - index < 32 ? depth - index : 32);
        sum_u64x4 = _mm256_add_epi64(
            sum_u64x4, _mm256_sad_epu8(_mm256_xor_si256(codes_vec.ymm, bias_u8x32), _mm256_setzero_si256()));
    }
    // Every lane summed carries the 128 bias, the zero codes past the depth included
    nk_i64_t const lanes = (nk_i64_t)nk_size_round_up_to_multiple_(depth, 32);
    nk_maxsim_vector_metadata_serial_(scale, norm_squared,
                                      (nk_i32_t)(nk_reduce_add_i64x4_haswell_(sum_u64x4) - 128 * lanes), metadata);
}

/** Packs one F32 vector into @p original and @p quantized, filling @p metadata. */
NUMKONG_INLINE void nk_maxsim_pack_vector_f32_haswell_(nk_f32_t const *source, nk_size_t depth, nk_f32_t scale_limit,
                                                       nk_f32_t *original, nk_i8_t *quantized,
                                                       nk_maxsim_vector_metadata_t *metadata) {
    __m256 absmax_f32x8 = _mm256_setzero_ps();
    __m256d sumsq_low_f64x4 = _mm256_setzero_pd(), sumsq_high_f64x4 = _mm256_setzero_pd();
    for (nk_size_t index = 0; index < depth; index += 8) {
        nk_size_t const chunk = depth - index < 8 ? depth - index : 8;
        nk_b256_vec_t raw_vec;
        nk_partial_load_b32x8_haswell_(source + index, &raw_vec, chunk);
        nk_partial_store_b32x8_haswell_(&raw_vec, original + index, chunk);
        nk_maxsim_measure_f32x8_haswell_(raw_vec.ymm_ps, &absmax_f32x8, &sumsq_low_f64x4, &sumsq_high_f64x4);
    }
    nk_f64_t const norm_squared = nk_reduce_add_f64x4_haswell_(_mm256_add_pd(sumsq_low_f64x4, sumsq_high_f64x4));
    nk_f32_t const scale = nk_maxsim_scale_f32_serial_(nk_reduce_max_f32x8_haswell_(absmax_f32x8), scale_limit);

    // Original rows span multiples of 16 values, zero past the depth, so whole loads stay inside
    __m256 const scale_f32x8 = _mm256_set1_ps(scale);
    __m256i const limit_i32x8 = _mm256_set1_epi32((int)scale_limit);
    __m128i const bias_u8x16 = _mm_set1_epi8((char)0x80);
    __m128i sum_u64x2 = _mm_setzero_si128();
    for (nk_size_t index = 0; index < depth; index += 16) {
        __m256i const low_i32x8 = nk_maxsim_quantize_f32x8_haswell_(_mm256_loadu_ps(original + index), scale_f32x8,
                                                                    limit_i32x8);
        __m256i const high_i32x8 = nk_maxsim_quantize_f32x8_haswell_(_mm256_loadu_ps(original + index + 8), scale_f32x8,
                                                                     limit_i32x8);
        // Packing interleaves the 128-bit halves, which the permute puts back in order
        __m256i const codes_i16x16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(low_i32x8, high_i32x8), 0xD8);
        nk_b128_vec_t codes_vec;
        codes_vec.xmm = _mm_packs_epi16(_mm256_castsi256_si128(codes_i16x16),
                                        _mm256_extracti128_si256(codes_i16x16, 1));
        nk_partial_store_b8x16_haswell_(&codes_vec, quantized + index, depth - index < 16 ? depth - index : 16);
        sum_u64x2 = _mm_add_epi64(sum_u64x2,
                                  _mm_sad_epu8(_mm_xor_si128(codes_vec.xmm, bias_u8x16), _mm_setzero_si128()));
    }
    // Every lane summed carries the 128 bias, the zero codes past the depth included
    nk_i64_t const lanes = (nk_i64_t)nk_size_round_up_to_multiple_(depth, 16);
    nk_i64_t const biased_sum = (nk_i64_t)_mm_cvtsi128_si64(sum_u64x2) + (nk_i64_t)_mm_extract_epi64(sum_u64x2, 1);
    nk_maxsim_vector_metadata_serial_(scale, norm_squared, (nk_i32_t)(biased_sum - 128 * lanes), metadata);
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_bf16_haswell_(void const *query, void const *document, nk_size_t depth) {
    nk_f32_t dot;
    nk_dot_bf16_through_f32_haswell_((nk_bf16_t const *)query, (nk_bf16_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f32_haswell_(void const *query, void const *document, nk_size_t depth) {
    nk_f64_t dot;
    nk_dot_f32_through_f64_haswell_((nk_f32_t const *)query, (nk_f32_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f16_haswell_(void const *query, void const *document, nk_size_t depth) {
    nk_f32_t dot;
    nk_dot_f16_through_f32_haswell_((nk_f16_t const *)query, (nk_f16_t const *)document, depth, &dot);
    return dot;
}

#if NUMKONG_TARGET_HASWELL

NUMKONG_API nk_status_t nk_maxsim_pack_size_bf16_haswell(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_serial_(vector_count, depth, sizeof(nk_bf16_t), 32);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_bf16_haswell(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_serial_(packed, nk_cap_haswell_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_serial_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_haswell(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_serial_(vector_count, depth, sizeof(nk_f32_t), 32);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_haswell(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_serial_(packed, nk_cap_haswell_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_serial_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_haswell(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_serial_(vector_count, depth, sizeof(nk_f16_t), 32);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_haswell(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_serial_(packed, nk_cap_haswell_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_serial_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_bf16_haswell( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed,
    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_bf16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_serial_(packed, vector_count, depth, 32, element_bytes,
                                                                      nk_cap_haswell_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++)
        nk_maxsim_pack_vector_bf16_haswell_((nk_bf16_t const *)((char const *)vectors + vector_index * stride), depth,
                                            79.0f, (nk_bf16_t *)(originals + vector_index * original_stride),
                                            quantized_i8 + vector_index * depth_i8_padded, metadata + vector_index);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f32_haswell( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed,
    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_f32_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_serial_(packed, vector_count, depth, 32, element_bytes,
                                                                      nk_cap_haswell_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++)
        nk_maxsim_pack_vector_f32_haswell_((nk_f32_t const *)((char const *)vectors + vector_index * stride), depth,
                                           79.0f, (nk_f32_t *)(originals + vector_index * original_stride),
                                           quantized_i8 + vector_index * depth_i8_padded, metadata + vector_index);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f16_haswell( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed,
    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_f16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_serial_(packed, vector_count, depth, 32, element_bytes,
                                                                      nk_cap_haswell_k);

    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized_i8 = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    nk_size_t const original_stride = header->original_stride;

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++)
        nk_maxsim_pack_vector_f16_haswell_((nk_f16_t const *)((char const *)vectors + vector_index * stride), depth,
                                           79.0f, (nk_f16_t *)(originals + vector_index * original_stride),
                                           quantized_i8 + vector_index * depth_i8_padded, metadata + vector_index);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_bf16_haswell( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_serial_(query_packed, nk_cap_haswell_k) ||
        !nk_maxsim_packed_by_serial_(document_packed, nk_cap_haswell_k))
        return nk_pack_mismatch_k;

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_serial_(query_packed, document_packed);
    nk_f64_t const poison = nk_maxsim_packed_poison_serial_(&regions, query_count, document_count);
    if (poison != 0) {
        *result = (nk_f32_t)poison;
        return nk_success_k;
    }
    nk_size_t const weights_stride = sizeof(nk_maxsim_vector_metadata_t) / sizeof(nk_f32_t);
    nk_f32_t const residue = 0.5f * nk_sqrt_f32_serial_((nk_f32_t)depth);
    nk_i32_t dots[32 * 128];
    nk_u32_t candidates[128];
    nk_f64_t total_angular_distance = 0.0, total_compensation = 0.0;

    for (nk_size_t query_start = 0; query_start < query_count; query_start += 32) {
        nk_size_t const query_tile = query_count - query_start < 32 ? query_count - query_start : 32;
        nk_maxsim_screen_error_t errors[32];
        nk_f32_t lower_bounds[32];
        nk_f64_t best_cosines[32];
        nk_maxsim_query_tile_begin_serial_(&regions, query_start, query_tile, residue, depth, errors, lower_bounds,
                                           best_cosines);

        for (nk_size_t document_start = 0; document_start < document_count; document_start += 128) {
            nk_size_t const document_tile = document_count - document_start < 128 ? document_count - document_start
                                                                                  : 128;
            nk_maxsim_coarse_dots_haswell_(regions.query_quantized + query_start * regions.depth_i8_padded,
                                           regions.document_quantized + document_start * regions.depth_i8_padded,
                                           regions.document_metadata + document_start, query_tile, document_tile,
                                           regions.depth_i8_padded, dots);
            nk_f32_t const *weights = &regions.document_metadata[document_start].screen_weight_f32;

            for (nk_size_t query_index = 0; query_index < query_tile; query_index++) {
                nk_size_t const query_global_index = query_start + query_index;
                nk_size_t const candidate_count = nk_maxsim_screen_query_serial_(
                    dots + query_index * document_tile, weights, weights_stride, document_tile, errors[query_index],
                    &lower_bounds[query_index], candidates);
                for (nk_size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
                    nk_size_t const document_index = document_start + candidates[candidate_index];
                    nk_f64_t const dot = nk_maxsim_refine_bf16_haswell_(
                        regions.query_originals + query_global_index * regions.query_original_stride,
                        regions.document_originals + document_index * regions.document_original_stride, depth);
                    nk_maxsim_best_cosine_update_serial_(
                        dot, regions.query_metadata[query_global_index].inverse_norm_f64,
                        regions.document_metadata[document_index].inverse_norm_f64, &best_cosines[query_index]);
                }
            }
        }

        nk_maxsim_angular_accumulate_serial_(best_cosines, query_tile, &total_angular_distance, &total_compensation);
    }
    *result = (nk_f32_t)(total_angular_distance + total_compensation);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f32_haswell( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_serial_(query_packed, nk_cap_haswell_k) ||
        !nk_maxsim_packed_by_serial_(document_packed, nk_cap_haswell_k))
        return nk_pack_mismatch_k;

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_serial_(query_packed, document_packed);
    nk_f64_t const poison = nk_maxsim_packed_poison_serial_(&regions, query_count, document_count);
    if (poison != 0) {
        *result = poison;
        return nk_success_k;
    }
    nk_size_t const weights_stride = sizeof(nk_maxsim_vector_metadata_t) / sizeof(nk_f32_t);
    nk_f32_t const residue = 0.5f * nk_sqrt_f32_serial_((nk_f32_t)depth);
    nk_i32_t dots[32 * 128];
    nk_u32_t candidates[128];
    nk_f64_t total_angular_distance = 0.0, total_compensation = 0.0;

    for (nk_size_t query_start = 0; query_start < query_count; query_start += 32) {
        nk_size_t const query_tile = query_count - query_start < 32 ? query_count - query_start : 32;
        nk_maxsim_screen_error_t errors[32];
        nk_f32_t lower_bounds[32];
        nk_f64_t best_cosines[32];
        nk_maxsim_query_tile_begin_serial_(&regions, query_start, query_tile, residue, depth, errors, lower_bounds,
                                           best_cosines);

        for (nk_size_t document_start = 0; document_start < document_count; document_start += 128) {
            nk_size_t const document_tile = document_count - document_start < 128 ? document_count - document_start
                                                                                  : 128;
            nk_maxsim_coarse_dots_haswell_(regions.query_quantized + query_start * regions.depth_i8_padded,
                                           regions.document_quantized + document_start * regions.depth_i8_padded,
                                           regions.document_metadata + document_start, query_tile, document_tile,
                                           regions.depth_i8_padded, dots);
            nk_f32_t const *weights = &regions.document_metadata[document_start].screen_weight_f32;

            for (nk_size_t query_index = 0; query_index < query_tile; query_index++) {
                nk_size_t const query_global_index = query_start + query_index;
                nk_size_t const candidate_count = nk_maxsim_screen_query_serial_(
                    dots + query_index * document_tile, weights, weights_stride, document_tile, errors[query_index],
                    &lower_bounds[query_index], candidates);
                for (nk_size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
                    nk_size_t const document_index = document_start + candidates[candidate_index];
                    nk_f64_t const dot = nk_maxsim_refine_f32_haswell_(
                        regions.query_originals + query_global_index * regions.query_original_stride,
                        regions.document_originals + document_index * regions.document_original_stride, depth);
                    nk_maxsim_best_cosine_update_serial_(
                        dot, regions.query_metadata[query_global_index].inverse_norm_f64,
                        regions.document_metadata[document_index].inverse_norm_f64, &best_cosines[query_index]);
                }
            }
        }

        nk_maxsim_angular_accumulate_serial_(best_cosines, query_tile, &total_angular_distance, &total_compensation);
    }
    *result = (total_angular_distance + total_compensation);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f16_haswell( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_serial_(query_packed, nk_cap_haswell_k) ||
        !nk_maxsim_packed_by_serial_(document_packed, nk_cap_haswell_k))
        return nk_pack_mismatch_k;

    nk_maxsim_packed_regions_t regions = nk_maxsim_extract_packed_regions_serial_(query_packed, document_packed);
    nk_f64_t const poison = nk_maxsim_packed_poison_serial_(&regions, query_count, document_count);
    if (poison != 0) {
        *result = (nk_f32_t)poison;
        return nk_success_k;
    }
    nk_size_t const weights_stride = sizeof(nk_maxsim_vector_metadata_t) / sizeof(nk_f32_t);
    nk_f32_t const residue = 0.5f * nk_sqrt_f32_serial_((nk_f32_t)depth);
    nk_i32_t dots[32 * 128];
    nk_u32_t candidates[128];
    nk_f64_t total_angular_distance = 0.0, total_compensation = 0.0;

    for (nk_size_t query_start = 0; query_start < query_count; query_start += 32) {
        nk_size_t const query_tile = query_count - query_start < 32 ? query_count - query_start : 32;
        nk_maxsim_screen_error_t errors[32];
        nk_f32_t lower_bounds[32];
        nk_f64_t best_cosines[32];
        nk_maxsim_query_tile_begin_serial_(&regions, query_start, query_tile, residue, depth, errors, lower_bounds,
                                           best_cosines);

        for (nk_size_t document_start = 0; document_start < document_count; document_start += 128) {
            nk_size_t const document_tile = document_count - document_start < 128 ? document_count - document_start
                                                                                  : 128;
            nk_maxsim_coarse_dots_haswell_(regions.query_quantized + query_start * regions.depth_i8_padded,
                                           regions.document_quantized + document_start * regions.depth_i8_padded,
                                           regions.document_metadata + document_start, query_tile, document_tile,
                                           regions.depth_i8_padded, dots);
            nk_f32_t const *weights = &regions.document_metadata[document_start].screen_weight_f32;

            for (nk_size_t query_index = 0; query_index < query_tile; query_index++) {
                nk_size_t const query_global_index = query_start + query_index;
                nk_size_t const candidate_count = nk_maxsim_screen_query_serial_(
                    dots + query_index * document_tile, weights, weights_stride, document_tile, errors[query_index],
                    &lower_bounds[query_index], candidates);
                for (nk_size_t candidate_index = 0; candidate_index < candidate_count; candidate_index++) {
                    nk_size_t const document_index = document_start + candidates[candidate_index];
                    nk_f64_t const dot = nk_maxsim_refine_f16_haswell_(
                        regions.query_originals + query_global_index * regions.query_original_stride,
                        regions.document_originals + document_index * regions.document_original_stride, depth);
                    nk_maxsim_best_cosine_update_serial_(
                        dot, regions.query_metadata[query_global_index].inverse_norm_f64,
                        regions.document_metadata[document_index].inverse_norm_f64, &best_cosines[query_index]);
                }
            }
        }

        nk_maxsim_angular_accumulate_serial_(best_cosines, query_tile, &total_angular_distance, &total_compensation);
    }
    *result = (nk_f32_t)(total_angular_distance + total_compensation);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_HASWELL

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_HASWELL_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_MAXSIM_HASWELL_H
