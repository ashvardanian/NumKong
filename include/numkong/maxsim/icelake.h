/**
 *  @file include/numkong/maxsim/icelake.h
 *  @author Ash Vardanian
 *  @date February 28, 2026
 *  @brief SIMD-accelerated MaxSim, ColBERT late-interaction, for Ice Lake.
 *
 *  @sa include/numkong/maxsim.h
 *
 *  Uses AVX-512 VNNI (VPDPBUSD) for coarse i8 screening. The coarse dots kernel and reduce helper
 *  are shared with genoa.h — genoa.h imports them from this file for its bf16 compute path.
 *
 *  VPDPBUSD computes 4 groups of (u8 × i8) → i32 per 128-bit lane, processing 64 i8 pairs per ZMM
 *  register operation. Bias correction via XOR with 0x80 converts signed queries to unsigned, then
 *  subtracts 128 * sum(document_i8) after the depth loop.
 *
 *  4x4 register tiling: 4 queries × 4 documents = 16 ZMM accumulators per depth loop.
 *  Each document load is amortized across 4 VPDPBUSDs, and each query load across 4 documents.
 *
 *  @verbatim
 *  Intrinsic            Instruction  Icelake   Genoa
 *  _mm512_dpbusd_epi32  VPDPBUSD     5cy @ p0  4cy @ p01
 *  @endverbatim
 */
#ifndef NUMKONG_MAXSIM_ICELAKE_H
#define NUMKONG_MAXSIM_ICELAKE_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_ICELAKE_

#include "numkong/types.h"
#include "numkong/maxsim/serial.h"  // `nk_maxsim_packed_header_t`
#include "numkong/dot/skylake.h"    // `nk_dot_f32_through_f64_skylake_`, `nk_dot_f16_through_f32_skylake_`
#include "numkong/reduce/skylake.h" // `nk_reduce_add_f32x16_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "f16c", "fma", "bmi", "bmi2")
#endif

#pragma region F32 Floats

#if NUMKONG_TARGET_ICELAKE
NUMKONG_API nk_status_t nk_maxsim_pack_size_f32_icelake(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_f32_t), 64);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f32_icelake(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                           void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_icelake_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f32_icelake( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_in_bytes, void *packed,
    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_f32_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 64, element_bytes,
                                                               nk_cap_icelake_k);

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

#pragma endregion F32 Floats

#pragma region F16 Floats

NUMKONG_API nk_status_t nk_maxsim_pack_size_f16_icelake(nk_size_t vector_count, nk_size_t depth, nk_size_t *bytes) {
    *bytes = nk_maxsim_pack_size_(vector_count, depth, sizeof(nk_f16_t), 64);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_shape_f16_icelake(void const *packed, nk_size_t *vectors, nk_size_t *depth,
                                                           void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_icelake_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f16_icelake( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_in_bytes, void *packed,
    void *stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const element_bytes = sizeof(nk_f16_t);
    nk_size_t depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 64, element_bytes,
                                                               nk_cap_icelake_k);

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
#endif // NUMKONG_TARGET_ICELAKE

#pragma endregion F16 Floats

#pragma region Coarse Dots

/** Reduces 4 ZMM i32x16 accumulators to a single __m128i with 4 horizontal sums. */
NUMKONG_INLINE __m128i nk_maxsim_reduce_i32x16x4_icelake_(      //
    __m512i accumulator_a_i32x16, __m512i accumulator_b_i32x16, //
    __m512i accumulator_c_i32x16, __m512i accumulator_d_i32x16) {
    // 16 → 8 (extract high 256-bit half and add to low half)
    __m256i sum_a_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(accumulator_a_i32x16),
                                           _mm512_extracti32x8_epi32(accumulator_a_i32x16, 1));
    __m256i sum_b_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(accumulator_b_i32x16),
                                           _mm512_extracti32x8_epi32(accumulator_b_i32x16, 1));
    __m256i sum_c_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(accumulator_c_i32x16),
                                           _mm512_extracti32x8_epi32(accumulator_c_i32x16, 1));
    __m256i sum_d_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(accumulator_d_i32x16),
                                           _mm512_extracti32x8_epi32(accumulator_d_i32x16, 1));
    // 8 → 4 (extract high 128-bit half and add to low half)
    __m128i sum_a_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_a_i32x8), _mm256_extracti128_si256(sum_a_i32x8, 1));
    __m128i sum_b_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_b_i32x8), _mm256_extracti128_si256(sum_b_i32x8, 1));
    __m128i sum_c_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_c_i32x8), _mm256_extracti128_si256(sum_c_i32x8, 1));
    __m128i sum_d_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum_d_i32x8), _mm256_extracti128_si256(sum_d_i32x8, 1));
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

/** Coarse i8 kernel for Ice Lake / Genoa as an @c nk_maxsim_coarse_dots_t. Uses AVX-512 VNNI VPDPBUSD
 *  with XOR-0x80 bias and 128*sum_quantized correction. */
NUMKONG_INLINE void nk_maxsim_coarse_dots_icelake_(       //
    nk_i8_t const *query_i8, nk_i8_t const *document_i8,  //
    nk_maxsim_vector_metadata_t const *document_metadata, //
    nk_size_t query_count, nk_size_t document_count,      //
    nk_size_t depth_i8_padded, nk_i32_t *dots) {

    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);

    // Primary path: 4-query grouping
    nk_size_t query_block_start_index = 0;
    for (; query_block_start_index + 4 <= query_count; query_block_start_index += 4) {
        // 4x4 document blocking
        nk_size_t document_block_start_index = 0;
        for (; document_block_start_index + 4 <= document_count; document_block_start_index += 4) {
            __m512i accumulator_tiles_i32x16[4][4];
            for (nk_size_t query_tile_index = 0; query_tile_index < 4; query_tile_index++)
                for (nk_size_t document_tile_index = 0; document_tile_index < 4; document_tile_index++)
                    accumulator_tiles_i32x16[query_tile_index][document_tile_index] = _mm512_setzero_si512();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 64) {
                __m512i query_biased_0_u8x64 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                __m512i query_biased_1_u8x64 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                __m512i query_biased_2_u8x64 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);
                __m512i query_biased_3_u8x64 = _mm512_xor_si512(
                    _mm512_loadu_si512(
                        (__m512i const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                    xor_mask_u8x64);

                __m512i document_i8x64;

                document_i8x64 = _mm512_loadu_si512(
                    (__m512i const *)(document_i8 + (document_block_start_index + 0) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x16[0][0] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[0][0],
                                                                     query_biased_0_u8x64, document_i8x64);
                accumulator_tiles_i32x16[1][0] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[1][0],
                                                                     query_biased_1_u8x64, document_i8x64);
                accumulator_tiles_i32x16[2][0] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[2][0],
                                                                     query_biased_2_u8x64, document_i8x64);
                accumulator_tiles_i32x16[3][0] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[3][0],
                                                                     query_biased_3_u8x64, document_i8x64);

                document_i8x64 = _mm512_loadu_si512(
                    (__m512i const *)(document_i8 + (document_block_start_index + 1) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x16[0][1] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[0][1],
                                                                     query_biased_0_u8x64, document_i8x64);
                accumulator_tiles_i32x16[1][1] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[1][1],
                                                                     query_biased_1_u8x64, document_i8x64);
                accumulator_tiles_i32x16[2][1] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[2][1],
                                                                     query_biased_2_u8x64, document_i8x64);
                accumulator_tiles_i32x16[3][1] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[3][1],
                                                                     query_biased_3_u8x64, document_i8x64);

                document_i8x64 = _mm512_loadu_si512(
                    (__m512i const *)(document_i8 + (document_block_start_index + 2) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x16[0][2] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[0][2],
                                                                     query_biased_0_u8x64, document_i8x64);
                accumulator_tiles_i32x16[1][2] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[1][2],
                                                                     query_biased_1_u8x64, document_i8x64);
                accumulator_tiles_i32x16[2][2] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[2][2],
                                                                     query_biased_2_u8x64, document_i8x64);
                accumulator_tiles_i32x16[3][2] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[3][2],
                                                                     query_biased_3_u8x64, document_i8x64);

                document_i8x64 = _mm512_loadu_si512(
                    (__m512i const *)(document_i8 + (document_block_start_index + 3) * depth_i8_padded + depth_index));
                accumulator_tiles_i32x16[0][3] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[0][3],
                                                                     query_biased_0_u8x64, document_i8x64);
                accumulator_tiles_i32x16[1][3] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[1][3],
                                                                     query_biased_1_u8x64, document_i8x64);
                accumulator_tiles_i32x16[2][3] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[2][3],
                                                                     query_biased_2_u8x64, document_i8x64);
                accumulator_tiles_i32x16[3][3] = _mm512_dpbusd_epi32(accumulator_tiles_i32x16[3][3],
                                                                     query_biased_3_u8x64, document_i8x64);
            }

            __m128i query_0_coarse_dots_i32x4 = nk_maxsim_reduce_i32x16x4_icelake_(
                accumulator_tiles_i32x16[0][0], accumulator_tiles_i32x16[0][1], accumulator_tiles_i32x16[0][2],
                accumulator_tiles_i32x16[0][3]);
            __m128i query_1_coarse_dots_i32x4 = nk_maxsim_reduce_i32x16x4_icelake_(
                accumulator_tiles_i32x16[1][0], accumulator_tiles_i32x16[1][1], accumulator_tiles_i32x16[1][2],
                accumulator_tiles_i32x16[1][3]);
            __m128i query_2_coarse_dots_i32x4 = nk_maxsim_reduce_i32x16x4_icelake_(
                accumulator_tiles_i32x16[2][0], accumulator_tiles_i32x16[2][1], accumulator_tiles_i32x16[2][2],
                accumulator_tiles_i32x16[2][3]);
            __m128i query_3_coarse_dots_i32x4 = nk_maxsim_reduce_i32x16x4_icelake_(
                accumulator_tiles_i32x16[3][0], accumulator_tiles_i32x16[3][1], accumulator_tiles_i32x16[3][2],
                accumulator_tiles_i32x16[3][3]);

            __m128i bias_correction_i32x4 = _mm_set_epi32(
                128 * document_metadata[document_block_start_index + 3].sum_i8_i32,
                128 * document_metadata[document_block_start_index + 2].sum_i8_i32,
                128 * document_metadata[document_block_start_index + 1].sum_i8_i32,
                128 * document_metadata[document_block_start_index + 0].sum_i8_i32);
            query_0_coarse_dots_i32x4 = _mm_sub_epi32(query_0_coarse_dots_i32x4, bias_correction_i32x4);
            query_1_coarse_dots_i32x4 = _mm_sub_epi32(query_1_coarse_dots_i32x4, bias_correction_i32x4);
            query_2_coarse_dots_i32x4 = _mm_sub_epi32(query_2_coarse_dots_i32x4, bias_correction_i32x4);
            query_3_coarse_dots_i32x4 = _mm_sub_epi32(query_3_coarse_dots_i32x4, bias_correction_i32x4);

            nk_i32_t *dots_row = dots + query_block_start_index * document_count + document_block_start_index;
            _mm_storeu_si128((__m128i *)(dots_row + 0 * document_count), query_0_coarse_dots_i32x4);
            _mm_storeu_si128((__m128i *)(dots_row + 1 * document_count), query_1_coarse_dots_i32x4);
            _mm_storeu_si128((__m128i *)(dots_row + 2 * document_count), query_2_coarse_dots_i32x4);
            _mm_storeu_si128((__m128i *)(dots_row + 3 * document_count), query_3_coarse_dots_i32x4);
        }

        // Document tail: 4Q × 1D
        for (nk_size_t document_index = document_block_start_index; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;

            __m512i accumulator_0_i32x16 = _mm512_setzero_si512();
            __m512i accumulator_1_i32x16 = _mm512_setzero_si512();
            __m512i accumulator_2_i32x16 = _mm512_setzero_si512();
            __m512i accumulator_3_i32x16 = _mm512_setzero_si512();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 64) {
                __m512i document_i8x64 = _mm512_loadu_si512((__m512i const *)(document_i8_row + depth_index));

                accumulator_0_i32x16 = _mm512_dpbusd_epi32(
                    accumulator_0_i32x16,
                    _mm512_xor_si512(
                        _mm512_loadu_si512((
                            __m512i const *)(query_i8 + (query_block_start_index + 0) * depth_i8_padded + depth_index)),
                        xor_mask_u8x64),
                    document_i8x64);
                accumulator_1_i32x16 = _mm512_dpbusd_epi32(
                    accumulator_1_i32x16,
                    _mm512_xor_si512(
                        _mm512_loadu_si512((
                            __m512i const *)(query_i8 + (query_block_start_index + 1) * depth_i8_padded + depth_index)),
                        xor_mask_u8x64),
                    document_i8x64);
                accumulator_2_i32x16 = _mm512_dpbusd_epi32(
                    accumulator_2_i32x16,
                    _mm512_xor_si512(
                        _mm512_loadu_si512((
                            __m512i const *)(query_i8 + (query_block_start_index + 2) * depth_i8_padded + depth_index)),
                        xor_mask_u8x64),
                    document_i8x64);
                accumulator_3_i32x16 = _mm512_dpbusd_epi32(
                    accumulator_3_i32x16,
                    _mm512_xor_si512(
                        _mm512_loadu_si512((
                            __m512i const *)(query_i8 + (query_block_start_index + 3) * depth_i8_padded + depth_index)),
                        xor_mask_u8x64),
                    document_i8x64);
            }

            nk_i32_t bias_correction_i32 = 128 * document_metadata[document_index].sum_i8_i32;
            nk_i32_t *dots_column = dots + query_block_start_index * document_count + document_index;
            dots_column[0 * document_count] = _mm512_reduce_add_epi32(accumulator_0_i32x16) - bias_correction_i32;
            dots_column[1 * document_count] = _mm512_reduce_add_epi32(accumulator_1_i32x16) - bias_correction_i32;
            dots_column[2 * document_count] = _mm512_reduce_add_epi32(accumulator_2_i32x16) - bias_correction_i32;
            dots_column[3 * document_count] = _mm512_reduce_add_epi32(accumulator_3_i32x16) - bias_correction_i32;
        }
    }

    // Query tail: 1Q × 1D
    for (nk_size_t query_index = query_block_start_index; query_index < query_count; query_index++) {
        nk_i8_t const *query_i8_row = query_i8 + query_index * depth_i8_padded;

        for (nk_size_t document_index = 0; document_index < document_count; document_index++) {
            nk_i8_t const *document_i8_row = document_i8 + document_index * depth_i8_padded;
            __m512i accumulator_i32x16 = _mm512_setzero_si512();

            for (nk_size_t depth_index = 0; depth_index < depth_i8_padded; depth_index += 64) {
                __m512i document_i8x64 = _mm512_loadu_si512((__m512i const *)(document_i8_row + depth_index));
                __m512i query_biased_u8x64 = _mm512_xor_si512(
                    _mm512_loadu_si512((__m512i const *)(query_i8_row + depth_index)), xor_mask_u8x64);
                accumulator_i32x16 = _mm512_dpbusd_epi32(accumulator_i32x16, query_biased_u8x64, document_i8x64);
            }

            dots[query_index * document_count + document_index] = _mm512_reduce_add_epi32(accumulator_i32x16) -
                                                                  128 * document_metadata[document_index].sum_i8_i32;
        }
    }
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f32_icelake_(void const *query, void const *document, nk_size_t depth) {
    nk_f64_t dot;
    nk_dot_f32_through_f64_skylake_((nk_f32_t const *)query, (nk_f32_t const *)document, depth, &dot);
    return dot;
}

NUMKONG_INLINE nk_f64_t nk_maxsim_refine_f16_icelake_(void const *query, void const *document, nk_size_t depth) {
    nk_f32_t dot;
    nk_dot_f16_through_f32_skylake_((nk_f16_t const *)query, (nk_f16_t const *)document, depth, &dot);
    return dot;
}

#pragma endregion Coarse Dots

#pragma region Compute Functions

#if NUMKONG_TARGET_ICELAKE
NUMKONG_API nk_status_t nk_maxsim_packed_f32_icelake( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f64_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_icelake_k) ||
        !nk_maxsim_packed_by_(document_packed, nk_cap_icelake_k))
        return nk_pack_mismatch_k;

    *result = nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                        nk_maxsim_coarse_dots_icelake_, nk_maxsim_refine_f32_icelake_);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_packed_f16_icelake( //
    void const *query_packed, void const *document_packed, nk_size_t query_count, nk_size_t document_count,
    nk_size_t depth, nk_f32_t *result, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(query_packed, nk_cap_icelake_k) ||
        !nk_maxsim_packed_by_(document_packed, nk_cap_icelake_k))
        return nk_pack_mismatch_k;

    *result = (nk_f32_t)nk_maxsim_packed_angular_(query_packed, document_packed, query_count, document_count, depth,
                                                  nk_maxsim_coarse_dots_icelake_, nk_maxsim_refine_f16_icelake_);
    return nk_success_k;
}
#endif // NUMKONG_TARGET_ICELAKE

#pragma endregion Compute Functions

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_ICELAKE_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_MAXSIM_ICELAKE_H
