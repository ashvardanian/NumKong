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
#include "numkong/cast/neon.h"     // `nk_load_b128_neon_`
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
                                                             nk_stream_t stream) {
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
                                                            nk_stream_t stream) {
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
                                                            nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_maxsim_packed_by_(packed, nk_cap_neonsdot_k)) return nk_pack_mismatch_k;
    nk_maxsim_packed_shape_(packed, vectors, depth);
    return nk_success_k;
}

/*  The packs below quantize each vector in two passes. The first reads the source once, stores it
 *  into the originals row, and folds in the abs-max and the F64 sum of squares; the second reads
 *  that row back to quantize. The rows end at most a chunk short of the 64-byte multiples the
 *  originals are padded to, so whole-chunk stores and reads stay inside them. */

/** Divides @p values_f32x4 by @p scale_f32x4, biases by ±0.5 along the sign, truncates and clamps
 *  to ±127, as @c nk_maxsim_quantize_f32_ does lane by lane. */
NUMKONG_INLINE int32x4_t nk_maxsim_quantize_f32x4_neonsdot_(float32x4_t values_f32x4, float32x4_t scale_f32x4) {
    float32x4_t const scaled_f32x4 = vdivq_f32(values_f32x4, scale_f32x4);
    float32x4_t const bias_f32x4 = vbslq_f32(vcgezq_f32(scaled_f32x4), vdupq_n_f32(0.5f), vdupq_n_f32(-0.5f));
    int32x4_t const codes_i32x4 = vcvtq_s32_f32(vaddq_f32(scaled_f32x4, bias_f32x4));
    return vmaxq_s32(vminq_s32(codes_i32x4, vdupq_n_s32(127)), vdupq_n_s32(-127));
}

/** Narrows four vectors of codes within ±127 into one, in order. */
NUMKONG_INLINE int8x16_t nk_maxsim_narrow_i32x4x4_neonsdot_(int32x4_t first_i32x4, int32x4_t second_i32x4,
                                                            int32x4_t third_i32x4, int32x4_t fourth_i32x4) {
    int16x8_t const low_i16x8 = vcombine_s16(vmovn_s32(first_i32x4), vmovn_s32(second_i32x4));
    int16x8_t const high_i16x8 = vcombine_s16(vmovn_s32(third_i32x4), vmovn_s32(fourth_i32x4));
    return vcombine_s8(vmovn_s16(low_i16x8), vmovn_s16(high_i16x8));
}

/** Writes the metadata of a vector from its abs-max scale, F64 sum of squares and code sum, as
 *  @c nk_maxsim_vector_metadata_ does. */
NUMKONG_INLINE void nk_maxsim_metadata_neonsdot_(nk_f32_t scale, nk_f64_t sumsq, nk_i32_t codes_sum,
                                                 nk_maxsim_vector_metadata_t *metadata) {
    metadata->inverse_norm_f64 = sumsq > 0.0 ? nk_f64_rsqrt_(sumsq) : 0.0;
    metadata->screen_weight_f32 = scale * (nk_f32_t)metadata->inverse_norm_f64;
    metadata->sum_i8_i32 = codes_sum;
}

NUMKONG_API nk_status_t nk_maxsim_pack_bf16_neonsdot( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed,
    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, sizeof(nk_bf16_t),
                                                                     nk_cap_neonsdot_k);
    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    int8x16_t const ones_i8x16 = vdupq_n_s8(1);

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        nk_bf16_t const *source = (nk_bf16_t const *)((char const *)vectors + vector_index * stride);
        nk_u16_t *original = (nk_u16_t *)(originals + vector_index * header->original_stride);
        float32x4_t absmax_f32x4 = vdupq_n_f32(0);
        float64x2_t sumsq_f64x2 = vdupq_n_f64(0);
        for (nk_size_t index = 0; index < depth; index += 8) {
            nk_b128_vec_t raw_vec;
            if (index + 8 <= depth) nk_load_b128_neon_(source + index, &raw_vec);
            else nk_partial_load_b16x8_serial_(source + index, &raw_vec, depth - index);
            vst1q_u16(original + index, raw_vec.u16x8);
            float32x4_t const low_f32x4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(raw_vec.u16x8), 16));
            float32x4_t const high_f32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(raw_vec.u16x8, 16));
            absmax_f32x4 = vmaxnmq_f32(absmax_f32x4, vmaxnmq_f32(vabsq_f32(low_f32x4), vabsq_f32(high_f32x4)));
            float64x2_t const first_f64x2 = vcvt_f64_f32(vget_low_f32(low_f32x4));
            float64x2_t const second_f64x2 = vcvt_high_f64_f32(low_f32x4);
            float64x2_t const third_f64x2 = vcvt_f64_f32(vget_low_f32(high_f32x4));
            float64x2_t const fourth_f64x2 = vcvt_high_f64_f32(high_f32x4);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, first_f64x2, first_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, second_f64x2, second_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, third_f64x2, third_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, fourth_f64x2, fourth_f64x2);
        }
        nk_f32_t scale = vmaxnmvq_f32(absmax_f32x4) / 127.0f;
        if (scale == 0.0f) scale = 1.0f;

        float32x4_t const scale_f32x4 = vdupq_n_f32(scale);
        int32x4_t codes_sum_i32x4 = vdupq_n_s32(0);
        nk_i8_t *codes = quantized + vector_index * depth_i8_padded;
        for (nk_size_t index = 0; index < depth; index += 16) {
            uint16x8_t const first_u16x8 = vld1q_u16(original + index), second_u16x8 = vld1q_u16(original + index + 8);
            int8x16_t const codes_i8x16 = nk_maxsim_narrow_i32x4x4_neonsdot_(
                nk_maxsim_quantize_f32x4_neonsdot_(vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(first_u16x8), 16)),
                                                   scale_f32x4),
                nk_maxsim_quantize_f32x4_neonsdot_(vreinterpretq_f32_u32(vshll_high_n_u16(first_u16x8, 16)),
                                                   scale_f32x4),
                nk_maxsim_quantize_f32x4_neonsdot_(vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(second_u16x8), 16)),
                                                   scale_f32x4),
                nk_maxsim_quantize_f32x4_neonsdot_(vreinterpretq_f32_u32(vshll_high_n_u16(second_u16x8, 16)),
                                                   scale_f32x4));
            vst1q_s8(codes + index, codes_i8x16);
            codes_sum_i32x4 = vdotq_s32(codes_sum_i32x4, codes_i8x16, ones_i8x16);
        }
        nk_maxsim_metadata_neonsdot_(scale, vaddvq_f64(sumsq_f64x2), vaddvq_s32(codes_sum_i32x4),
                                     &metadata[vector_index]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f32_neonsdot( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed,
    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, sizeof(nk_f32_t),
                                                                     nk_cap_neonsdot_k);
    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    int8x16_t const ones_i8x16 = vdupq_n_s8(1);

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        nk_f32_t const *source = (nk_f32_t const *)((char const *)vectors + vector_index * stride);
        nk_f32_t *original = (nk_f32_t *)(originals + vector_index * header->original_stride);
        float32x4_t absmax_f32x4 = vdupq_n_f32(0);
        float64x2_t sumsq_f64x2 = vdupq_n_f64(0);
        for (nk_size_t index = 0; index < depth; index += 4) {
            nk_b128_vec_t raw_vec;
            if (index + 4 <= depth) nk_load_b128_neon_(source + index, &raw_vec);
            else nk_partial_load_b32x4_serial_(source + index, &raw_vec, depth - index);
            vst1q_f32(original + index, raw_vec.f32x4);
            absmax_f32x4 = vmaxnmq_f32(absmax_f32x4, vabsq_f32(raw_vec.f32x4));
            float64x2_t const low_f64x2 = vcvt_f64_f32(vget_low_f32(raw_vec.f32x4));
            float64x2_t const high_f64x2 = vcvt_high_f64_f32(raw_vec.f32x4);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, low_f64x2, low_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, high_f64x2, high_f64x2);
        }
        nk_f32_t scale = vmaxnmvq_f32(absmax_f32x4) / 127.0f;
        if (scale == 0.0f) scale = 1.0f;

        float32x4_t const scale_f32x4 = vdupq_n_f32(scale);
        int32x4_t codes_sum_i32x4 = vdupq_n_s32(0);
        nk_i8_t *codes = quantized + vector_index * depth_i8_padded;
        for (nk_size_t index = 0; index < depth; index += 16) {
            int8x16_t const codes_i8x16 = nk_maxsim_narrow_i32x4x4_neonsdot_(
                nk_maxsim_quantize_f32x4_neonsdot_(vld1q_f32(original + index), scale_f32x4),
                nk_maxsim_quantize_f32x4_neonsdot_(vld1q_f32(original + index + 4), scale_f32x4),
                nk_maxsim_quantize_f32x4_neonsdot_(vld1q_f32(original + index + 8), scale_f32x4),
                nk_maxsim_quantize_f32x4_neonsdot_(vld1q_f32(original + index + 12), scale_f32x4));
            vst1q_s8(codes + index, codes_i8x16);
            codes_sum_i32x4 = vdotq_s32(codes_sum_i32x4, codes_i8x16, ones_i8x16);
        }
        nk_maxsim_metadata_neonsdot_(scale, vaddvq_f64(sumsq_f64x2), vaddvq_s32(codes_sum_i32x4),
                                     &metadata[vector_index]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_maxsim_pack_f16_neonsdot( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, void *packed,
    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);

    nk_size_t const depth_i8_padded = nk_maxsim_packed_header_setup_(packed, vector_count, depth, 16, sizeof(nk_f16_t),
                                                                     nk_cap_neonsdot_k);
    nk_maxsim_packed_header_t const *header = (nk_maxsim_packed_header_t const *)packed;
    nk_i8_t *quantized = (nk_i8_t *)((char *)packed + header->offset_i8_data);
    nk_maxsim_vector_metadata_t *metadata = (nk_maxsim_vector_metadata_t *)((char *)packed + header->offset_metadata);
    char *originals = (char *)packed + header->offset_original_data;
    int8x16_t const ones_i8x16 = vdupq_n_s8(1);

    for (nk_size_t vector_index = 0; vector_index < vector_count; vector_index++) {
        nk_f16_t const *source = (nk_f16_t const *)((char const *)vectors + vector_index * stride);
        nk_u16_t *original = (nk_u16_t *)(originals + vector_index * header->original_stride);
        float32x4_t absmax_f32x4 = vdupq_n_f32(0);
        float64x2_t sumsq_f64x2 = vdupq_n_f64(0);
        for (nk_size_t index = 0; index < depth; index += 8) {
            nk_b128_vec_t raw_vec;
            if (index + 8 <= depth) nk_load_b128_neon_(source + index, &raw_vec);
            else nk_partial_load_b16x8_serial_(source + index, &raw_vec, depth - index);
            vst1q_u16(original + index, raw_vec.u16x8);
            float32x4_t const low_f32x4 = vcvt_f32_f16(vget_low_f16(raw_vec.f16x8));
            float32x4_t const high_f32x4 = vcvt_high_f32_f16(raw_vec.f16x8);
            absmax_f32x4 = vmaxnmq_f32(absmax_f32x4, vmaxnmq_f32(vabsq_f32(low_f32x4), vabsq_f32(high_f32x4)));
            float64x2_t const first_f64x2 = vcvt_f64_f32(vget_low_f32(low_f32x4));
            float64x2_t const second_f64x2 = vcvt_high_f64_f32(low_f32x4);
            float64x2_t const third_f64x2 = vcvt_f64_f32(vget_low_f32(high_f32x4));
            float64x2_t const fourth_f64x2 = vcvt_high_f64_f32(high_f32x4);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, first_f64x2, first_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, second_f64x2, second_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, third_f64x2, third_f64x2);
            sumsq_f64x2 = vfmaq_f64(sumsq_f64x2, fourth_f64x2, fourth_f64x2);
        }
        nk_f32_t scale = vmaxnmvq_f32(absmax_f32x4) / 127.0f;
        if (scale == 0.0f) scale = 1.0f;

        float32x4_t const scale_f32x4 = vdupq_n_f32(scale);
        int32x4_t codes_sum_i32x4 = vdupq_n_s32(0);
        nk_i8_t *codes = quantized + vector_index * depth_i8_padded;
        for (nk_size_t index = 0; index < depth; index += 16) {
            float16x8_t const first_f16x8 = vreinterpretq_f16_u16(vld1q_u16(original + index));
            float16x8_t const second_f16x8 = vreinterpretq_f16_u16(vld1q_u16(original + index + 8));
            int8x16_t const codes_i8x16 = nk_maxsim_narrow_i32x4x4_neonsdot_(
                nk_maxsim_quantize_f32x4_neonsdot_(vcvt_f32_f16(vget_low_f16(first_f16x8)), scale_f32x4),
                nk_maxsim_quantize_f32x4_neonsdot_(vcvt_high_f32_f16(first_f16x8), scale_f32x4),
                nk_maxsim_quantize_f32x4_neonsdot_(vcvt_f32_f16(vget_low_f16(second_f16x8)), scale_f32x4),
                nk_maxsim_quantize_f32x4_neonsdot_(vcvt_high_f32_f16(second_f16x8), scale_f32x4));
            vst1q_s8(codes + index, codes_i8x16);
            codes_sum_i32x4 = vdotq_s32(codes_sum_i32x4, codes_i8x16, ones_i8x16);
        }
        nk_maxsim_metadata_neonsdot_(scale, vaddvq_f64(sumsq_f64x2), vaddvq_s32(codes_sum_i32x4),
                                     &metadata[vector_index]);
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
    nk_size_t depth, nk_f32_t *result, nk_stream_t stream) {
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
    nk_size_t depth, nk_f64_t *result, nk_stream_t stream) {
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
    nk_size_t depth, nk_f32_t *result, nk_stream_t stream) {
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
