/**
 *  @file include/numkong/attention/neon.h
 *  @author Ash Vardanian
 *  @date October 5, 2026
 *  @brief Arm NEON ragged attention backend and rotary position embeddings.
 *
 *  @sa include/numkong/attention.h
 *
 *  Baseline backend for Armv8 cores without the dot-product extensions, in the @c haswell shape:
 *  BF16 and F16 planes keep their source encoding, E4M3 widens exactly to F16 at pack, and every
 *  value widens to F32 inside the hot loops, four lanes per FMA. Scores keep four KV rows in flight
 *  over panels of 512 keys under the family's base-2 streaming softmax. I8 scores stay exact in I32
 *  through widening multiplies, and the family's U8 weights times I8 values add exactly in I32
 *  before one F32 conversion per panel. Channels pad to 8 for floats and 16 for I8, and heads
 *  deeper than 256 route to the serial kernels under this capability.
 */
#ifndef NUMKONG_ATTENTION_NEON_H
#define NUMKONG_ATTENTION_NEON_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include <arm_neon.h>

#include "numkong/types.h"
#include "numkong/attention/serial.h" // `nk_attention_packed_header_t`, `nk_attention_pack_directory_serial_`
#include "numkong/cast/neon.h"        // `nk_bf16x4_to_f32x4_neon_`, `nk_e4m3x8_to_f16x8_neon_`
#include "numkong/each/neon.h"        // `nk_exp2_f32x4_neon_`, `nk_exp2_u8_i32x4_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

enum {

    /** KV panel width in positions; the F32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_neon_k_ = 512,

    /** Deepest head this backend handles in scratch; deeper heads route to the serial kernel. */
    nk_attention_max_depth_neon_k_ = 256,
};

/** Widens @p count BF16 query elements to F32 into @p destination, zero-filling to @p padded. */
NUMKONG_INLINE void nk_attention_widen_bf16_neon_(void const *source, nk_f32_t *destination, nk_size_t count,
                                                  nk_size_t padded) {
    nk_bf16_t const *elements = (nk_bf16_t const *)source;
    nk_size_t channel_idx = 0;
    for (; channel_idx + 4 <= count; channel_idx += 4)
        vst1q_f32(destination + channel_idx, nk_load_bf16x4_as_f32_neon_(elements + channel_idx));
    if (channel_idx < count) {
        nk_b64_vec_t elements_vec;
        nk_partial_load_b16x4_serial_(elements + channel_idx, &elements_vec, count - channel_idx);
        vst1q_f32(destination + channel_idx, nk_bf16x4_to_f32x4_neon_(elements_vec.u16x4));
        channel_idx += 4;
    }
    for (; channel_idx < padded; channel_idx += 4) vst1q_f32(destination + channel_idx, vdupq_n_f32(0));
}

/** Widens @p count F16 query elements to F32 into @p destination, zero-filling to @p padded. */
NUMKONG_INLINE void nk_attention_widen_f16_neon_(void const *source, nk_f32_t *destination, nk_size_t count,
                                                 nk_size_t padded) {
    nk_f16_t const *elements = (nk_f16_t const *)source;
    nk_size_t channel_idx = 0;
    for (; channel_idx + 4 <= count; channel_idx += 4)
        vst1q_f32(destination + channel_idx, nk_load_f16x4_as_f32_neon_(elements + channel_idx));
    if (channel_idx < count) {
        nk_b64_vec_t elements_vec;
        nk_partial_load_b16x4_serial_(elements + channel_idx, &elements_vec, count - channel_idx);
        vst1q_f32(destination + channel_idx, vcvt_f32_f16(vreinterpret_f16_u16(elements_vec.u16x4)));
        channel_idx += 4;
    }
    for (; channel_idx < padded; channel_idx += 4) vst1q_f32(destination + channel_idx, vdupq_n_f32(0));
}

/** Widens @p count E4M3 query elements to F32 into @p destination, zero-filling to @p padded. */
NUMKONG_INLINE void nk_attention_widen_e4m3_neon_(void const *source, nk_f32_t *destination, nk_size_t count,
                                                  nk_size_t padded) {
    nk_e4m3_t const *elements = (nk_e4m3_t const *)source;
    nk_size_t channel_idx = 0;
    for (; channel_idx + 4 <= count; channel_idx += 4)
        vst1q_f32(destination + channel_idx, nk_load_e4m3x4_as_f32_neon_(elements + channel_idx));
    if (channel_idx < count) {
        vst1q_f32(destination + channel_idx,
                  nk_e4m3x4_to_f32x4_neon_(nk_partial_load_b8x4_serial_(elements + channel_idx, count - channel_idx)));
        channel_idx += 4;
    }
    for (; channel_idx < padded; channel_idx += 4) vst1q_f32(destination + channel_idx, vdupq_n_f32(0));
}

/** Copies @p count 16-bit values into @p destination, zeroed up to @p padded, a multiple of 8. */
NUMKONG_INLINE void nk_attention_copy_row_b16_neon_(void const *source, nk_u16_t *destination, nk_size_t count,
                                                    nk_size_t padded) {
    nk_u16_t const *elements = (nk_u16_t const *)source;
    for (nk_size_t channel_idx = 0; channel_idx < padded; channel_idx += 8) {
        nk_b128_vec_t elements_vec;
        if (channel_idx + 8 <= count) elements_vec.u16x8 = vld1q_u16(elements + channel_idx);
        else
            nk_partial_load_b16x8_serial_(elements + channel_idx, &elements_vec,
                                          channel_idx < count ? count - channel_idx : 0);
        vst1q_u16(destination + channel_idx, elements_vec.u16x8);
    }
}

/** Copies @p count bytes into @p destination, zeroing it up to @p padded, a multiple of 16. */
NUMKONG_INLINE void nk_attention_copy_row_b8_neon_(void const *source, nk_u8_t *destination, nk_size_t count,
                                                   nk_size_t padded) {
    nk_u8_t const *bytes = (nk_u8_t const *)source;
    for (nk_size_t byte_idx = 0; byte_idx < padded; byte_idx += 16) {
        nk_b128_vec_t bytes_vec;
        if (byte_idx + 16 <= count) bytes_vec.u8x16 = vld1q_u8(bytes + byte_idx);
        else nk_partial_load_b8x16_serial_(bytes + byte_idx, &bytes_vec, byte_idx < count ? count - byte_idx : 0);
        vst1q_u8(destination + byte_idx, bytes_vec.u8x16);
    }
}

/** Widens @p count E4M3 values exactly to F16 into @p destination, zeroing it up to @p padded, a
 *  multiple of 8. */
NUMKONG_INLINE void nk_attention_e4m3_row_to_f16_neon_(void const *source, nk_u16_t *destination, nk_size_t count,
                                                       nk_size_t padded) {
    nk_u8_t const *codes = (nk_u8_t const *)source;
    for (nk_size_t channel_idx = 0; channel_idx < padded; channel_idx += 8) {
        nk_b64_vec_t codes_vec;
        if (channel_idx + 8 <= count) codes_vec.u8x8 = vld1_u8(codes + channel_idx);
        else
            nk_partial_load_b8x8_serial_(codes + channel_idx, &codes_vec,
                                         channel_idx < count ? count - channel_idx : 0);
        vst1q_u16(destination + channel_idx, vreinterpretq_u16_f16(nk_e4m3x8_to_f16x8_neon_(codes_vec.u8x8)));
    }
}

/** Panel max, online correction and exp2 in place for one query row's raw scores, scaled by
 *  @p scale2; returns the correction 2^(m_old − m_new) the caller applies to its output row. */
NUMKONG_INLINE nk_f32_t nk_attention_softmax_panel_neon_(nk_f32_t *scores, nk_size_t panel_length, nk_f32_t scale2,
                                                         nk_f32_t *running_max2, nk_f32_t *running_sum) {
    nk_u32_t const lane_indices_u32[4] = {0, 1, 2, 3};
    uint32x4_t const tail_mask_u32x4 = vcltq_u32(vld1q_u32(lane_indices_u32), vdupq_n_u32(panel_length % 4));
    nk_size_t const tail_start = panel_length - panel_length % 4;
    float32x4_t const scale2_f32x4 = vdupq_n_f32(scale2);
    float32x4_t const lowest_f32x4 = vdupq_n_f32(NUMKONG_F32_MIN);
    float32x4_t max_f32x4 = lowest_f32x4;
    nk_size_t position_idx = 0;
    for (; position_idx < tail_start; position_idx += 4)
        max_f32x4 = vmaxq_f32(max_f32x4, vmulq_f32(vld1q_f32(scores + position_idx), scale2_f32x4));
    nk_b128_vec_t tail_vec;
    nk_partial_load_b32x4_serial_(scores + tail_start, &tail_vec, panel_length - tail_start);
    max_f32x4 = vmaxq_f32(max_f32x4, vbslq_f32(tail_mask_u32x4, vmulq_f32(tail_vec.f32x4, scale2_f32x4), lowest_f32x4));
    nk_f32_t const panel_max2 = vmaxvq_f32(max_f32x4);
    nk_f32_t const new_max2 = *running_max2 > panel_max2 ? *running_max2 : panel_max2;
    nk_f32_t const correction = vgetq_lane_f32(nk_exp2_f32x4_neon_(vdupq_n_f32(*running_max2 - new_max2)), 0);
    *running_max2 = new_max2;

    float32x4_t const max2_f32x4 = vdupq_n_f32(new_max2);
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    for (position_idx = 0; position_idx < tail_start; position_idx += 4) {
        float32x4_t const weights_f32x4 = nk_exp2_f32x4_neon_(
            vsubq_f32(vmulq_f32(vld1q_f32(scores + position_idx), scale2_f32x4), max2_f32x4));
        sum_f32x4 = vaddq_f32(sum_f32x4, weights_f32x4);
        vst1q_f32(scores + position_idx, weights_f32x4);
    }
    // The masked tail keeps the vector exp2 end-to-end
    tail_vec.f32x4 = vreinterpretq_f32_u32(vandq_u32(
        vreinterpretq_u32_f32(nk_exp2_f32x4_neon_(vsubq_f32(vmulq_f32(tail_vec.f32x4, scale2_f32x4), max2_f32x4))),
        tail_mask_u32x4));
    sum_f32x4 = vaddq_f32(sum_f32x4, tail_vec.f32x4);
    nk_partial_store_b32x4_serial_(&tail_vec, scores + tail_start, panel_length - tail_start);
    *running_sum = *running_sum * correction + vaddvq_f32(sum_f32x4);
    return correction;
}

NUMKONG_INLINE void nk_attention_zero_row_neon_(nk_f32_t *output_row, nk_size_t depth_padded) {
    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
        vst1q_f32(output_row + channel_idx, vdupq_n_f32(0));
}

NUMKONG_INLINE void nk_attention_scale_row_neon_(nk_f32_t *output_row, nk_size_t depth_padded, nk_f32_t factor) {
    float32x4_t const factor_f32x4 = vdupq_n_f32(factor);
    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
        vst1q_f32(output_row + channel_idx, vmulq_f32(vld1q_f32(output_row + channel_idx), factor_f32x4));
}

/** Writes @p depth channels of @p output_row, padded to a multiple of 4 and scaled by @p factor, to
 *  @p destination. */
NUMKONG_INLINE void nk_attention_store_row_neon_(nk_f32_t *destination, nk_f32_t const *output_row, nk_size_t depth,
                                                 nk_f32_t factor) {
    float32x4_t const factor_f32x4 = vdupq_n_f32(factor);
    nk_size_t channel_idx = 0;
    for (; channel_idx + 4 <= depth; channel_idx += 4)
        vst1q_f32(destination + channel_idx, vmulq_f32(vld1q_f32(output_row + channel_idx), factor_f32x4));
    if (channel_idx == depth) return;
    nk_b128_vec_t output_vec;
    output_vec.f32x4 = vmulq_f32(vld1q_f32(output_row + channel_idx), factor_f32x4);
    nk_partial_store_b32x4_serial_(&output_vec, destination + channel_idx, depth - channel_idx);
}

/** Scores of one panel of BF16 keys against the widened query, four KV rows in flight. */
NUMKONG_INLINE void nk_attention_scores_panel_bf16_neon_(nk_f32_t const *query_row, nk_f32_t *scores,
                                                         char const *keys_rows, nk_size_t panel_length,
                                                         nk_size_t depth_padded, nk_size_t plane_row_bytes) {
    nk_size_t position_idx = 0;
    for (; position_idx + 4 <= panel_length; position_idx += 4) {
        nk_u16_t const *keys_row0 = (nk_u16_t const *)(keys_rows + (position_idx + 0) * plane_row_bytes);
        nk_u16_t const *keys_row1 = (nk_u16_t const *)(keys_rows + (position_idx + 1) * plane_row_bytes);
        nk_u16_t const *keys_row2 = (nk_u16_t const *)(keys_rows + (position_idx + 2) * plane_row_bytes);
        nk_u16_t const *keys_row3 = (nk_u16_t const *)(keys_rows + (position_idx + 3) * plane_row_bytes);
        float32x4_t sum0_f32x4 = vdupq_n_f32(0), sum1_f32x4 = vdupq_n_f32(0);
        float32x4_t sum2_f32x4 = vdupq_n_f32(0), sum3_f32x4 = vdupq_n_f32(0);
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
            float32x4_t const query_low_f32x4 = vld1q_f32(query_row + channel_idx);
            float32x4_t const query_high_f32x4 = vld1q_f32(query_row + channel_idx + 4);
            uint16x8_t const keys0_u16x8 = vld1q_u16(keys_row0 + channel_idx);
            uint16x8_t const keys1_u16x8 = vld1q_u16(keys_row1 + channel_idx);
            uint16x8_t const keys2_u16x8 = vld1q_u16(keys_row2 + channel_idx);
            uint16x8_t const keys3_u16x8 = vld1q_u16(keys_row3 + channel_idx);
            sum0_f32x4 = vfmaq_f32(sum0_f32x4, query_low_f32x4,
                                   vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(keys0_u16x8), 16)));
            sum1_f32x4 = vfmaq_f32(sum1_f32x4, query_low_f32x4,
                                   vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(keys1_u16x8), 16)));
            sum2_f32x4 = vfmaq_f32(sum2_f32x4, query_low_f32x4,
                                   vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(keys2_u16x8), 16)));
            sum3_f32x4 = vfmaq_f32(sum3_f32x4, query_low_f32x4,
                                   vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(keys3_u16x8), 16)));
            sum0_f32x4 = vfmaq_f32(sum0_f32x4, query_high_f32x4,
                                   vreinterpretq_f32_u32(vshll_high_n_u16(keys0_u16x8, 16)));
            sum1_f32x4 = vfmaq_f32(sum1_f32x4, query_high_f32x4,
                                   vreinterpretq_f32_u32(vshll_high_n_u16(keys1_u16x8, 16)));
            sum2_f32x4 = vfmaq_f32(sum2_f32x4, query_high_f32x4,
                                   vreinterpretq_f32_u32(vshll_high_n_u16(keys2_u16x8, 16)));
            sum3_f32x4 = vfmaq_f32(sum3_f32x4, query_high_f32x4,
                                   vreinterpretq_f32_u32(vshll_high_n_u16(keys3_u16x8, 16)));
        }
        vst1q_f32(scores + position_idx,
                  vpaddq_f32(vpaddq_f32(sum0_f32x4, sum1_f32x4), vpaddq_f32(sum2_f32x4, sum3_f32x4)));
    }
    for (; position_idx < panel_length; position_idx++) {
        nk_u16_t const *keys_row = (nk_u16_t const *)(keys_rows + position_idx * plane_row_bytes);
        float32x4_t sum_f32x4 = vdupq_n_f32(0);
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
            uint16x8_t const keys_u16x8 = vld1q_u16(keys_row + channel_idx);
            sum_f32x4 = vfmaq_f32(sum_f32x4, vld1q_f32(query_row + channel_idx),
                                  vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(keys_u16x8), 16)));
            sum_f32x4 = vfmaq_f32(sum_f32x4, vld1q_f32(query_row + channel_idx + 4),
                                  vreinterpretq_f32_u32(vshll_high_n_u16(keys_u16x8, 16)));
        }
        scores[position_idx] = vaddvq_f32(sum_f32x4);
    }
}

/** Scores of one panel of F16 keys against the widened query, four KV rows in flight. */
NUMKONG_INLINE void nk_attention_scores_panel_f16_neon_(nk_f32_t const *query_row, nk_f32_t *scores,
                                                        char const *keys_rows, nk_size_t panel_length,
                                                        nk_size_t depth_padded, nk_size_t plane_row_bytes) {
    nk_size_t position_idx = 0;
    for (; position_idx + 4 <= panel_length; position_idx += 4) {
        nk_u16_t const *keys_row0 = (nk_u16_t const *)(keys_rows + (position_idx + 0) * plane_row_bytes);
        nk_u16_t const *keys_row1 = (nk_u16_t const *)(keys_rows + (position_idx + 1) * plane_row_bytes);
        nk_u16_t const *keys_row2 = (nk_u16_t const *)(keys_rows + (position_idx + 2) * plane_row_bytes);
        nk_u16_t const *keys_row3 = (nk_u16_t const *)(keys_rows + (position_idx + 3) * plane_row_bytes);
        float32x4_t sum0_f32x4 = vdupq_n_f32(0), sum1_f32x4 = vdupq_n_f32(0);
        float32x4_t sum2_f32x4 = vdupq_n_f32(0), sum3_f32x4 = vdupq_n_f32(0);
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
            float32x4_t const query_low_f32x4 = vld1q_f32(query_row + channel_idx);
            float32x4_t const query_high_f32x4 = vld1q_f32(query_row + channel_idx + 4);
            float16x8_t const keys0_f16x8 = vreinterpretq_f16_u16(vld1q_u16(keys_row0 + channel_idx));
            float16x8_t const keys1_f16x8 = vreinterpretq_f16_u16(vld1q_u16(keys_row1 + channel_idx));
            float16x8_t const keys2_f16x8 = vreinterpretq_f16_u16(vld1q_u16(keys_row2 + channel_idx));
            float16x8_t const keys3_f16x8 = vreinterpretq_f16_u16(vld1q_u16(keys_row3 + channel_idx));
            sum0_f32x4 = vfmaq_f32(sum0_f32x4, query_low_f32x4, vcvt_f32_f16(vget_low_f16(keys0_f16x8)));
            sum1_f32x4 = vfmaq_f32(sum1_f32x4, query_low_f32x4, vcvt_f32_f16(vget_low_f16(keys1_f16x8)));
            sum2_f32x4 = vfmaq_f32(sum2_f32x4, query_low_f32x4, vcvt_f32_f16(vget_low_f16(keys2_f16x8)));
            sum3_f32x4 = vfmaq_f32(sum3_f32x4, query_low_f32x4, vcvt_f32_f16(vget_low_f16(keys3_f16x8)));
            sum0_f32x4 = vfmaq_f32(sum0_f32x4, query_high_f32x4, vcvt_high_f32_f16(keys0_f16x8));
            sum1_f32x4 = vfmaq_f32(sum1_f32x4, query_high_f32x4, vcvt_high_f32_f16(keys1_f16x8));
            sum2_f32x4 = vfmaq_f32(sum2_f32x4, query_high_f32x4, vcvt_high_f32_f16(keys2_f16x8));
            sum3_f32x4 = vfmaq_f32(sum3_f32x4, query_high_f32x4, vcvt_high_f32_f16(keys3_f16x8));
        }
        vst1q_f32(scores + position_idx,
                  vpaddq_f32(vpaddq_f32(sum0_f32x4, sum1_f32x4), vpaddq_f32(sum2_f32x4, sum3_f32x4)));
    }
    for (; position_idx < panel_length; position_idx++) {
        nk_u16_t const *keys_row = (nk_u16_t const *)(keys_rows + position_idx * plane_row_bytes);
        float32x4_t sum_f32x4 = vdupq_n_f32(0);
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
            float16x8_t const keys_f16x8 = vreinterpretq_f16_u16(vld1q_u16(keys_row + channel_idx));
            sum_f32x4 = vfmaq_f32(sum_f32x4, vld1q_f32(query_row + channel_idx),
                                  vcvt_f32_f16(vget_low_f16(keys_f16x8)));
            sum_f32x4 = vfmaq_f32(sum_f32x4, vld1q_f32(query_row + channel_idx + 4), vcvt_high_f32_f16(keys_f16x8));
        }
        scores[position_idx] = vaddvq_f32(sum_f32x4);
    }
}

/** Adds the panel's weights times its BF16 V rows into @p output_row. */
NUMKONG_INLINE void nk_attention_weighted_sum_bf16_neon_(nk_f32_t *output_row, nk_f32_t const *weights,
                                                         char const *values_rows, nk_size_t panel_length,
                                                         nk_size_t depth_padded, nk_size_t plane_row_bytes) {
    for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx++) {
        float32x4_t const weight_f32x4 = vdupq_n_f32(weights[position_idx]);
        nk_u16_t const *values_row = (nk_u16_t const *)(values_rows + position_idx * plane_row_bytes);
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
            uint16x8_t const values_u16x8 = vld1q_u16(values_row + channel_idx);
            vst1q_f32(output_row + channel_idx,
                      vfmaq_f32(vld1q_f32(output_row + channel_idx), weight_f32x4,
                                vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(values_u16x8), 16))));
            vst1q_f32(output_row + channel_idx + 4,
                      vfmaq_f32(vld1q_f32(output_row + channel_idx + 4), weight_f32x4,
                                vreinterpretq_f32_u32(vshll_high_n_u16(values_u16x8, 16))));
        }
    }
}

/** Adds the panel's weights times its F16 V rows into @p output_row. */
NUMKONG_INLINE void nk_attention_weighted_sum_f16_neon_(nk_f32_t *output_row, nk_f32_t const *weights,
                                                        char const *values_rows, nk_size_t panel_length,
                                                        nk_size_t depth_padded, nk_size_t plane_row_bytes) {
    for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx++) {
        float32x4_t const weight_f32x4 = vdupq_n_f32(weights[position_idx]);
        nk_u16_t const *values_row = (nk_u16_t const *)(values_rows + position_idx * plane_row_bytes);
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
            float16x8_t const values_f16x8 = vreinterpretq_f16_u16(vld1q_u16(values_row + channel_idx));
            vst1q_f32(output_row + channel_idx, vfmaq_f32(vld1q_f32(output_row + channel_idx), weight_f32x4,
                                                          vcvt_f32_f16(vget_low_f16(values_f16x8))));
            vst1q_f32(output_row + channel_idx + 4, vfmaq_f32(vld1q_f32(output_row + channel_idx + 4), weight_f32x4,
                                                              vcvt_high_f32_f16(values_f16x8)));
        }
    }
}

/** Exact I32 scores of one panel of I8 keys against the query, four KV rows in flight. */
NUMKONG_INLINE void nk_attention_scores_panel_i8_neon_(nk_i8_t const *query_row, nk_i32_t *scores,
                                                       char const *keys_rows, nk_size_t panel_length,
                                                       nk_size_t depth_padded) {
    nk_size_t position_idx = 0;
    for (; position_idx + 4 <= panel_length; position_idx += 4) {
        int8_t const *keys_row0 = (int8_t const *)(keys_rows + (position_idx + 0) * depth_padded);
        int8_t const *keys_row1 = (int8_t const *)(keys_rows + (position_idx + 1) * depth_padded);
        int8_t const *keys_row2 = (int8_t const *)(keys_rows + (position_idx + 2) * depth_padded);
        int8_t const *keys_row3 = (int8_t const *)(keys_rows + (position_idx + 3) * depth_padded);
        int32x4_t sum0_i32x4 = vdupq_n_s32(0), sum1_i32x4 = vdupq_n_s32(0);
        int32x4_t sum2_i32x4 = vdupq_n_s32(0), sum3_i32x4 = vdupq_n_s32(0);
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16) {
            int8x16_t const query_i8x16 = vld1q_s8((int8_t const *)query_row + channel_idx);
            int8x16_t const keys0_i8x16 = vld1q_s8(keys_row0 + channel_idx);
            int8x16_t const keys1_i8x16 = vld1q_s8(keys_row1 + channel_idx);
            int8x16_t const keys2_i8x16 = vld1q_s8(keys_row2 + channel_idx);
            int8x16_t const keys3_i8x16 = vld1q_s8(keys_row3 + channel_idx);
            // Each I16 product holds at most 2^14, so the pairwise sums widen before they add
            sum0_i32x4 = vpadalq_s16(sum0_i32x4, vmull_s8(vget_low_s8(query_i8x16), vget_low_s8(keys0_i8x16)));
            sum1_i32x4 = vpadalq_s16(sum1_i32x4, vmull_s8(vget_low_s8(query_i8x16), vget_low_s8(keys1_i8x16)));
            sum2_i32x4 = vpadalq_s16(sum2_i32x4, vmull_s8(vget_low_s8(query_i8x16), vget_low_s8(keys2_i8x16)));
            sum3_i32x4 = vpadalq_s16(sum3_i32x4, vmull_s8(vget_low_s8(query_i8x16), vget_low_s8(keys3_i8x16)));
            sum0_i32x4 = vpadalq_s16(sum0_i32x4, vmull_high_s8(query_i8x16, keys0_i8x16));
            sum1_i32x4 = vpadalq_s16(sum1_i32x4, vmull_high_s8(query_i8x16, keys1_i8x16));
            sum2_i32x4 = vpadalq_s16(sum2_i32x4, vmull_high_s8(query_i8x16, keys2_i8x16));
            sum3_i32x4 = vpadalq_s16(sum3_i32x4, vmull_high_s8(query_i8x16, keys3_i8x16));
        }
        vst1q_s32(scores + position_idx,
                  vpaddq_s32(vpaddq_s32(sum0_i32x4, sum1_i32x4), vpaddq_s32(sum2_i32x4, sum3_i32x4)));
    }
    for (; position_idx < panel_length; position_idx++) {
        int8_t const *keys_row = (int8_t const *)(keys_rows + position_idx * depth_padded);
        int32x4_t sum_i32x4 = vdupq_n_s32(0);
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16) {
            int8x16_t const query_i8x16 = vld1q_s8((int8_t const *)query_row + channel_idx);
            int8x16_t const keys_i8x16 = vld1q_s8(keys_row + channel_idx);
            sum_i32x4 = vpadalq_s16(sum_i32x4, vmull_s8(vget_low_s8(query_i8x16), vget_low_s8(keys_i8x16)));
            sum_i32x4 = vpadalq_s16(sum_i32x4, vmull_high_s8(query_i8x16, keys_i8x16));
        }
        scores[position_idx] = vaddvq_s32(sum_i32x4);
    }
}

/**
 *  @brief Integer panel softmax for one I8 query row, in place of its exact I32 scores.
 *
 *  Weights quantize to trunc(2^(s₂ − m₂) · 255 + 0.5) like the whole I8 family, through the Q15
 *  @p scale_fixed and the @p delta_floor below which every weight is zero. Returns the correction
 *  2^(m_old − m_new) the caller applies to its output row.
 */
NUMKONG_INLINE nk_f32_t nk_attention_softmax_panel_i8_neon_(nk_i32_t *scores, nk_size_t panel_length, nk_f32_t scale2,
                                                            nk_i32_t scale_fixed, nk_i32_t delta_floor,
                                                            nk_i32_t *running_max, nk_f32_t *running_sum) {
    nk_u32_t const lane_indices_u32[4] = {0, 1, 2, 3};
    uint32x4_t const tail_mask_u32x4 = vcltq_u32(vld1q_u32(lane_indices_u32), vdupq_n_u32(panel_length % 4));
    nk_size_t const tail_start = panel_length - panel_length % 4;
    int32x4_t const lowest_i32x4 = vdupq_n_s32(NUMKONG_I32_MIN);
    int32x4_t max_i32x4 = lowest_i32x4;
    nk_size_t position_idx = 0;
    for (; position_idx < tail_start; position_idx += 4)
        max_i32x4 = vmaxq_s32(max_i32x4, vld1q_s32(scores + position_idx));
    nk_b128_vec_t tail_vec;
    nk_partial_load_b32x4_serial_(scores + tail_start, &tail_vec, panel_length - tail_start);
    max_i32x4 = vmaxq_s32(max_i32x4, vbslq_s32(tail_mask_u32x4, tail_vec.i32x4, lowest_i32x4));
    nk_i32_t const panel_max = vmaxvq_s32(max_i32x4);
    nk_i32_t const new_max = *running_max > panel_max ? *running_max : panel_max;
    nk_f32_t const correction = vgetq_lane_f32(
        nk_exp2_f32x4_neon_(vdupq_n_f32(((nk_f32_t)*running_max - (nk_f32_t)new_max) * scale2)), 0);
    *running_max = new_max;

    int32x4_t const new_max_i32x4 = vdupq_n_s32(new_max);
    int32x4_t const scale_fixed_i32x4 = vdupq_n_s32(scale_fixed);
    int32x4_t const delta_floor_i32x4 = vdupq_n_s32(delta_floor);
    uint32x4_t sum_u32x4 = vdupq_n_u32(0); // weights are U8 over at most 512 positions
    for (position_idx = 0; position_idx < tail_start; position_idx += 4) {
        int32x4_t const delta_i32x4 = vmaxq_s32(vsubq_s32(vld1q_s32(scores + position_idx), new_max_i32x4),
                                                delta_floor_i32x4);
        int32x4_t const weights_i32x4 = nk_exp2_u8_i32x4_neon_(vmulq_s32(delta_i32x4, scale_fixed_i32x4));
        sum_u32x4 = vaddq_u32(sum_u32x4, vreinterpretq_u32_s32(weights_i32x4));
        vst1q_s32(scores + position_idx, weights_i32x4);
    }
    int32x4_t const delta_i32x4 = vmaxq_s32(vsubq_s32(tail_vec.i32x4, new_max_i32x4), delta_floor_i32x4);
    tail_vec.u32x4 = vandq_u32(vreinterpretq_u32_s32(nk_exp2_u8_i32x4_neon_(vmulq_s32(delta_i32x4, scale_fixed_i32x4))),
                               tail_mask_u32x4);
    sum_u32x4 = vaddq_u32(sum_u32x4, tail_vec.u32x4);
    nk_partial_store_b32x4_serial_(&tail_vec, scores + tail_start, panel_length - tail_start);
    *running_sum = *running_sum * correction + (nk_f32_t)vaddvq_u32(sum_u32x4);
    return correction;
}

/** Adds the panel's U8 weights times its I8 V rows into the exact I32 @p totals: each panel total
 *  stays under 512 · 255 · 128 < 2^24, so it converts to F32 exactly. */
NUMKONG_INLINE void nk_attention_weighted_sum_i8_neon_(nk_i32_t *totals, nk_i32_t const *weights,
                                                       char const *values_rows, nk_size_t panel_length,
                                                       nk_size_t depth_padded) {
    for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx++) {
        nk_i16_t const weight = (nk_i16_t)weights[position_idx];
        if (weight == 0) continue; // U8 softmax weights are sparse
        int8_t const *values_row = (int8_t const *)(values_rows + position_idx * depth_padded);
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16) {
            int8x16_t const values_i8x16 = vld1q_s8(values_row + channel_idx);
            int16x8_t const values_low_i16x8 = vmovl_s8(vget_low_s8(values_i8x16));
            int16x8_t const values_high_i16x8 = vmovl_high_s8(values_i8x16);
            nk_i32_t *totals_chunk = totals + channel_idx;
            vst1q_s32(totals_chunk, vmlal_n_s16(vld1q_s32(totals_chunk), vget_low_s16(values_low_i16x8), weight));
            vst1q_s32(totals_chunk + 4, vmlal_high_n_s16(vld1q_s32(totals_chunk + 4), values_low_i16x8, weight));
            vst1q_s32(totals_chunk + 8,
                      vmlal_n_s16(vld1q_s32(totals_chunk + 8), vget_low_s16(values_high_i16x8), weight));
            vst1q_s32(totals_chunk + 12, vmlal_high_n_s16(vld1q_s32(totals_chunk + 12), values_high_i16x8, weight));
        }
    }
}

#if NUMKONG_TARGET_NEON

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes) {
    // Planes keep the raw BF16 encoding, past the NEON depths in serial F32
    *bytes = depth > nk_attention_max_depth_neon_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   depth * sizeof(nk_f32_t))
                 : nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   nk_size_round_up_to_multiple_(depth, 8) * sizeof(nk_bf16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_f16_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes) {
    // Planes keep the raw F16 encoding, past the NEON depths in serial F32
    *bytes = depth > nk_attention_max_depth_neon_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   depth * sizeof(nk_f32_t))
                 : nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   nk_size_round_up_to_multiple_(depth, 8) * sizeof(nk_f16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes) {
    // Planes store the exact F16 widening of the E4M3 inputs, past the NEON depths in serial F32
    *bytes = depth > nk_attention_max_depth_neon_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   depth * sizeof(nk_f32_t))
                 : nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   nk_size_round_up_to_multiple_(depth, 8) * sizeof(nk_f16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_i8_neon(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_size_t token_count, nk_size_t segment_count,
                                                       nk_size_t *bytes) {
    // Planes keep the raw I8 rows, channels padded to the 16-byte loads
    *bytes = depth > nk_attention_max_depth_neon_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1, depth)
                 : nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   nk_size_round_up_to_multiple_(depth, 16));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_neon(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neon_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_f16_neon(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neon_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_neon(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neon_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_neon(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                          nk_size_t *depth, nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neon_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_bf16_neon(                       //
    nk_bf16_t const *keys, nk_bf16_t const *values,                        //
    nk_size_t key_value_head_count, nk_size_t depth,                       //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,              //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, //
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_neon_k_) {
        nk_attention_pack_bf16_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                       segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                       tasks_end, nk_cap_neon_k);
        return nk_success_k;
    }
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const padded_row_bytes = depth_padded * sizeof(nk_bf16_t);
    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, padded_row_bytes, nk_cap_neon_k);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return nk_success_k;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count;
        nk_size_t const key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                padded_row_bytes);
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * padded_row_bytes;
        char *keys_plane = payload_base + payload_offset + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride +
                                   key_value_head_idx * depth * sizeof(nk_bf16_t);
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride +
                                     key_value_head_idx * depth * sizeof(nk_bf16_t);
            nk_attention_copy_row_b16_neon_(keys_row, (nk_u16_t *)(keys_plane + position_idx * padded_row_bytes), depth,
                                            depth_padded);
            nk_attention_copy_row_b16_neon_(values_row, (nk_u16_t *)(values_plane + position_idx * padded_row_bytes),
                                            depth, depth_padded);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_f16_neon(                        //
    nk_f16_t const *keys, nk_f16_t const *values,                          //
    nk_size_t key_value_head_count, nk_size_t depth,                       //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,              //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, //
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_neon_k_) {
        nk_attention_pack_f16_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                      segment_count, key_stride, value_stride, key_value_packed, tasks_begin, tasks_end,
                                      nk_cap_neon_k);
        return nk_success_k;
    }
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const padded_row_bytes = depth_padded * sizeof(nk_f16_t);
    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, padded_row_bytes, nk_cap_neon_k);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return nk_success_k;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count;
        nk_size_t const key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                padded_row_bytes);
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * padded_row_bytes;
        char *keys_plane = payload_base + payload_offset + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride +
                                   key_value_head_idx * depth * sizeof(nk_f16_t);
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride +
                                     key_value_head_idx * depth * sizeof(nk_f16_t);
            nk_attention_copy_row_b16_neon_(keys_row, (nk_u16_t *)(keys_plane + position_idx * padded_row_bytes), depth,
                                            depth_padded);
            nk_attention_copy_row_b16_neon_(values_row, (nk_u16_t *)(values_plane + position_idx * padded_row_bytes),
                                            depth, depth_padded);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_neon(                       //
    nk_e4m3_t const *keys, nk_e4m3_t const *values,                        //
    nk_size_t key_value_head_count, nk_size_t depth,                       //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,              //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, //
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_neon_k_) {
        nk_attention_pack_e4m3_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                       segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                       tasks_end, nk_cap_neon_k);
        return nk_success_k;
    }
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const padded_row_bytes = depth_padded * sizeof(nk_f16_t);
    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, padded_row_bytes, nk_cap_neon_k);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return nk_success_k;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count;
        nk_size_t const key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                padded_row_bytes);
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * padded_row_bytes;
        char *keys_plane = payload_base + payload_offset + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride +
                                   key_value_head_idx * depth;
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride +
                                     key_value_head_idx * depth;
            nk_attention_e4m3_row_to_f16_neon_(keys_row, (nk_u16_t *)(keys_plane + position_idx * padded_row_bytes),
                                               depth, depth_padded);
            nk_attention_e4m3_row_to_f16_neon_(values_row, (nk_u16_t *)(values_plane + position_idx * padded_row_bytes),
                                               depth, depth_padded);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_i8_neon(                         //
    nk_i8_t const *keys, nk_i8_t const *values,                            //
    nk_size_t key_value_head_count, nk_size_t depth,                       //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,              //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, //
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_neon_k_) {
        nk_attention_pack_i8_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                     key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, nk_cap_neon_k);
        return nk_success_k;
    }
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 16);
    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, depth_padded, nk_cap_neon_k);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return nk_success_k;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count;
        nk_size_t const key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 1,
                depth_padded);
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * depth_padded;
        char *keys_plane = payload_base + payload_offset + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride +
                                   key_value_head_idx * depth;
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride +
                                     key_value_head_idx * depth;
            nk_attention_copy_row_b8_neon_(keys_row, (nk_u8_t *)(keys_plane + position_idx * depth_padded), depth,
                                           depth_padded);
            nk_attention_copy_row_b8_neon_(values_row, (nk_u8_t *)(values_plane + position_idx * depth_padded), depth,
                                           depth_padded);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_bf16_neon(                           //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neon_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_neon_k_) {
        nk_attention_packed_bf16_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                         key_value_head_count, depth, query_offsets, query_stride, output_stride, scale,
                                         band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const plane_row_bytes = depth_padded * sizeof(nk_bf16_t);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)
    nk_size_t const panel_width = nk_attention_panel_neon_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_f32_t query_row[nk_attention_max_depth_neon_k_];
    nk_align_(64) nk_f32_t output_row[nk_attention_max_depth_neon_k_];
    nk_align_(64) nk_f32_t scores[nk_attention_panel_neon_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const plane_bytes = position_count * plane_row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                     (head_idx / head_group_size) * plane_bytes;
            char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_attention_widen_bf16_neon_((char const *)queries + (query_first + row_idx) * query_stride +
                                                  head_idx * depth * sizeof(nk_bf16_t),
                                              query_row, depth, depth_padded);
                nk_attention_zero_row_neon_(output_row, depth_padded);
                nk_f32_t running_max2 = NUMKONG_F32_MIN, running_sum = 0;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    nk_attention_scores_panel_bf16_neon_(query_row, scores, keys_plane + panel_start * plane_row_bytes,
                                                         panel_length, depth_padded, plane_row_bytes);
                    nk_f32_t const correction = nk_attention_softmax_panel_neon_(scores, panel_length, scale2,
                                                                                 &running_max2, &running_sum);
                    nk_attention_scale_row_neon_(output_row, depth_padded, correction);
                    nk_attention_weighted_sum_bf16_neon_(output_row, scores,
                                                         values_plane + panel_start * plane_row_bytes, panel_length,
                                                         depth_padded, plane_row_bytes);
                }

                nk_size_t const token = query_first + row_idx;
                nk_attention_store_row_neon_(output + token * output_stride_floats + head_idx * depth, output_row,
                                             depth, running_sum > 0 ? 1 / running_sum : 0);
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(running_max2,
                                                                                                  running_sum);
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_f16_neon(                            //
    nk_f16_t const *queries, void const *key_value_packed, nk_f32_t *output,     //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neon_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_neon_k_) {
        nk_attention_packed_f16_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                        key_value_head_count, depth, query_offsets, query_stride, output_stride, scale,
                                        band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const plane_row_bytes = depth_padded * sizeof(nk_f16_t);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)
    nk_size_t const panel_width = nk_attention_panel_neon_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_f32_t query_row[nk_attention_max_depth_neon_k_];
    nk_align_(64) nk_f32_t output_row[nk_attention_max_depth_neon_k_];
    nk_align_(64) nk_f32_t scores[nk_attention_panel_neon_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const plane_bytes = position_count * plane_row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                     (head_idx / head_group_size) * plane_bytes;
            char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_attention_widen_f16_neon_((char const *)queries + (query_first + row_idx) * query_stride +
                                                 head_idx * depth * sizeof(nk_f16_t),
                                             query_row, depth, depth_padded);
                nk_attention_zero_row_neon_(output_row, depth_padded);
                nk_f32_t running_max2 = NUMKONG_F32_MIN, running_sum = 0;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    nk_attention_scores_panel_f16_neon_(query_row, scores, keys_plane + panel_start * plane_row_bytes,
                                                        panel_length, depth_padded, plane_row_bytes);
                    nk_f32_t const correction = nk_attention_softmax_panel_neon_(scores, panel_length, scale2,
                                                                                 &running_max2, &running_sum);
                    nk_attention_scale_row_neon_(output_row, depth_padded, correction);
                    nk_attention_weighted_sum_f16_neon_(output_row, scores,
                                                        values_plane + panel_start * plane_row_bytes, panel_length,
                                                        depth_padded, plane_row_bytes);
                }

                nk_size_t const token = query_first + row_idx;
                nk_attention_store_row_neon_(output + token * output_stride_floats + head_idx * depth, output_row,
                                             depth, running_sum > 0 ? 1 / running_sum : 0);
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(running_max2,
                                                                                                  running_sum);
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_e4m3_neon(                           //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neon_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_neon_k_) {
        nk_attention_packed_e4m3_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                         key_value_head_count, depth, query_offsets, query_stride, output_stride, scale,
                                         band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const plane_row_bytes = depth_padded * sizeof(nk_f16_t);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)
    nk_size_t const panel_width = nk_attention_panel_neon_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_f32_t query_row[nk_attention_max_depth_neon_k_];
    nk_align_(64) nk_f32_t output_row[nk_attention_max_depth_neon_k_];
    nk_align_(64) nk_f32_t scores[nk_attention_panel_neon_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const plane_bytes = position_count * plane_row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                     (head_idx / head_group_size) * plane_bytes;
            char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_attention_widen_e4m3_neon_(
                    (char const *)queries + (query_first + row_idx) * query_stride + head_idx * depth, query_row, depth,
                    depth_padded);
                nk_attention_zero_row_neon_(output_row, depth_padded);
                nk_f32_t running_max2 = NUMKONG_F32_MIN, running_sum = 0;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    nk_attention_scores_panel_f16_neon_(query_row, scores, keys_plane + panel_start * plane_row_bytes,
                                                        panel_length, depth_padded, plane_row_bytes);
                    nk_f32_t const correction = nk_attention_softmax_panel_neon_(scores, panel_length, scale2,
                                                                                 &running_max2, &running_sum);
                    nk_attention_scale_row_neon_(output_row, depth_padded, correction);
                    nk_attention_weighted_sum_f16_neon_(output_row, scores,
                                                        values_plane + panel_start * plane_row_bytes, panel_length,
                                                        depth_padded, plane_row_bytes);
                }

                nk_size_t const token = query_first + row_idx;
                nk_attention_store_row_neon_(output + token * output_stride_floats + head_idx * depth, output_row,
                                             depth, running_sum > 0 ? 1 / running_sum : 0);
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(running_max2,
                                                                                                  running_sum);
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_i8_neon(                             //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,      //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neon_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_neon_k_) {
        nk_attention_packed_i8_serial_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                       depth, query_offsets, query_stride, output_stride, scale, band, tasks_begin,
                                       tasks_end);
        return nk_success_k;
    }
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 16);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;                // softmax(x) = softmax₂(x · log₂e)
    nk_i32_t const scale_fixed = (nk_i32_t)(scale2 * 32768.0f + 0.5f); // Q15 scale for the integer exponential
    nk_i32_t const delta_floor = // the score delta below which every weight quantizes to zero (2^t · 255 + 0.5 < 1)
        scale_fixed > 0 ? -(nk_i32_t)((10u << 15) / (nk_u32_t)scale_fixed) - 1 : 0;
    nk_size_t const panel_width = nk_attention_panel_neon_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_i8_t query_row[nk_attention_max_depth_neon_k_];
    nk_align_(64) nk_f32_t output_row[nk_attention_max_depth_neon_k_];
    nk_align_(64) nk_i32_t totals[nk_attention_max_depth_neon_k_];
    nk_align_(64) nk_i32_t scores[nk_attention_panel_neon_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const plane_bytes = position_count * depth_padded;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                     (head_idx / head_group_size) * plane_bytes;
            char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_attention_copy_row_b8_neon_(
                    (char const *)queries + (query_first + row_idx) * query_stride + head_idx * depth,
                    (nk_u8_t *)query_row, depth, depth_padded);
                nk_attention_zero_row_neon_(output_row, depth_padded);
                nk_i32_t running_max = NUMKONG_I32_MIN;
                nk_f32_t running_sum = 0;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    nk_attention_scores_panel_i8_neon_(query_row, scores, keys_plane + panel_start * depth_padded,
                                                       panel_length, depth_padded);
                    nk_f32_t const correction = nk_attention_softmax_panel_i8_neon_(
                        scores, panel_length, scale2, scale_fixed, delta_floor, &running_max, &running_sum);
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                        vst1q_s32(totals + channel_idx, vdupq_n_s32(0));
                    nk_attention_weighted_sum_i8_neon_(totals, scores, values_plane + panel_start * depth_padded,
                                                       panel_length, depth_padded);
                    float32x4_t const correction_f32x4 = vdupq_n_f32(correction);
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                        vst1q_f32(output_row + channel_idx,
                                  vfmaq_f32(vcvtq_f32_s32(vld1q_s32(totals + channel_idx)),
                                            vld1q_f32(output_row + channel_idx), correction_f32x4));
                }

                nk_size_t const token = query_first + row_idx;
                nk_attention_store_row_neon_(output + token * output_stride_floats + head_idx * depth, output_row,
                                             depth, running_sum > 0 ? 1 / running_sum : 0);
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(
                        (nk_f32_t)running_max * scale2, running_sum / 255.0f);
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_rope_f32_neon(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                   nk_f32_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                   nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(depth % 2 == 0);
    nk_size_t const half_depth = depth / 2;
    for (nk_size_t row_idx = 0; row_idx != rows; ++row_idx) {
        nk_f32_t const *cos_row = cos + row_idx * half_depth;
        nk_f32_t const *sin_row = sin + row_idx * half_depth;
        nk_f32_t const *x_row = (nk_f32_t const *)((char const *)x + row_idx * x_stride);
        nk_f32_t *y_row = (nk_f32_t *)((char *)y + row_idx * y_stride);
        for (nk_size_t head_idx = 0; head_idx != head_count; ++head_idx) {
            nk_f32_t const *x_head = x_row + head_idx * depth;
            nk_f32_t *y_head = y_row + head_idx * depth;
            nk_size_t channel_idx = 0;
            for (; channel_idx + 4 <= half_depth; channel_idx += 4) {
                float32x4_t const low_f32x4 = vld1q_f32(x_head + channel_idx);
                float32x4_t const high_f32x4 = vld1q_f32(x_head + half_depth + channel_idx);
                float32x4_t const cos_f32x4 = vld1q_f32(cos_row + channel_idx);
                float32x4_t const sin_f32x4 = vld1q_f32(sin_row + channel_idx);
                vst1q_f32(y_head + channel_idx,
                          vsubq_f32(vmulq_f32(low_f32x4, cos_f32x4), vmulq_f32(high_f32x4, sin_f32x4)));
                vst1q_f32(y_head + half_depth + channel_idx,
                          vaddq_f32(vmulq_f32(low_f32x4, sin_f32x4), vmulq_f32(high_f32x4, cos_f32x4)));
            }
            if (channel_idx == half_depth) continue;
            nk_size_t const count = half_depth - channel_idx;
            nk_b128_vec_t low_vec, high_vec, cos_vec, sin_vec;
            nk_partial_load_b32x4_serial_(x_head + channel_idx, &low_vec, count);
            nk_partial_load_b32x4_serial_(x_head + half_depth + channel_idx, &high_vec, count);
            nk_partial_load_b32x4_serial_(cos_row + channel_idx, &cos_vec, count);
            nk_partial_load_b32x4_serial_(sin_row + channel_idx, &sin_vec, count);
            nk_b128_vec_t rotated_low_vec, rotated_high_vec;
            rotated_low_vec.f32x4 = vsubq_f32(vmulq_f32(low_vec.f32x4, cos_vec.f32x4),
                                              vmulq_f32(high_vec.f32x4, sin_vec.f32x4));
            rotated_high_vec.f32x4 = vaddq_f32(vmulq_f32(low_vec.f32x4, sin_vec.f32x4),
                                               vmulq_f32(high_vec.f32x4, cos_vec.f32x4));
            nk_partial_store_b32x4_serial_(&rotated_low_vec, y_head + channel_idx, count);
            nk_partial_store_b32x4_serial_(&rotated_high_vec, y_head + half_depth + channel_idx, count);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_rope_bf16_neon(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_bf16_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(depth % 2 == 0);
    nk_size_t const half_depth = depth / 2;
    for (nk_size_t row_idx = 0; row_idx != rows; ++row_idx) {
        nk_f32_t const *cos_row = cos + row_idx * half_depth;
        nk_f32_t const *sin_row = sin + row_idx * half_depth;
        nk_bf16_t const *x_row = (nk_bf16_t const *)((char const *)x + row_idx * x_stride);
        nk_bf16_t *y_row = (nk_bf16_t *)((char *)y + row_idx * y_stride);
        for (nk_size_t head_idx = 0; head_idx != head_count; ++head_idx) {
            nk_bf16_t const *x_head = x_row + head_idx * depth;
            nk_bf16_t *y_head = y_row + head_idx * depth;
            nk_size_t channel_idx = 0;
            for (; channel_idx + 4 <= half_depth; channel_idx += 4) {
                float32x4_t const low_f32x4 = nk_load_bf16x4_as_f32_neon_(x_head + channel_idx);
                float32x4_t const high_f32x4 = nk_load_bf16x4_as_f32_neon_(x_head + half_depth + channel_idx);
                float32x4_t const cos_f32x4 = vld1q_f32(cos_row + channel_idx);
                float32x4_t const sin_f32x4 = vld1q_f32(sin_row + channel_idx);
                nk_store_f32x4_as_bf16_neon_(
                    y_head + channel_idx, vsubq_f32(vmulq_f32(low_f32x4, cos_f32x4), vmulq_f32(high_f32x4, sin_f32x4)));
                nk_store_f32x4_as_bf16_neon_(
                    y_head + half_depth + channel_idx,
                    vaddq_f32(vmulq_f32(low_f32x4, sin_f32x4), vmulq_f32(high_f32x4, cos_f32x4)));
            }
            if (channel_idx == half_depth) continue;
            nk_size_t const count = half_depth - channel_idx;
            nk_b64_vec_t low_vec, high_vec;
            nk_b128_vec_t cos_vec, sin_vec;
            nk_partial_load_b16x4_serial_(x_head + channel_idx, &low_vec, count);
            nk_partial_load_b16x4_serial_(x_head + half_depth + channel_idx, &high_vec, count);
            nk_partial_load_b32x4_serial_(cos_row + channel_idx, &cos_vec, count);
            nk_partial_load_b32x4_serial_(sin_row + channel_idx, &sin_vec, count);
            float32x4_t const low_f32x4 = nk_bf16x4_to_f32x4_neon_(low_vec.u16x4);
            float32x4_t const high_f32x4 = nk_bf16x4_to_f32x4_neon_(high_vec.u16x4);
            low_vec.u16x4 = nk_f32x4_to_bf16x4_neon_(
                vsubq_f32(vmulq_f32(low_f32x4, cos_vec.f32x4), vmulq_f32(high_f32x4, sin_vec.f32x4)));
            high_vec.u16x4 = nk_f32x4_to_bf16x4_neon_(
                vaddq_f32(vmulq_f32(low_f32x4, sin_vec.f32x4), vmulq_f32(high_f32x4, cos_vec.f32x4)));
            nk_partial_store_b16x4_serial_(y_head + channel_idx, &low_vec, count);
            nk_partial_store_b16x4_serial_(y_head + half_depth + channel_idx, &high_vec, count);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_rope_e4m3_neon(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                    nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count, nk_size_t depth,
                                                    nk_size_t x_stride, nk_size_t y_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(depth % 2 == 0);
    nk_size_t const half_depth = depth / 2;
    for (nk_size_t row_idx = 0; row_idx != rows; ++row_idx) {
        nk_f32_t const *cos_row = cos + row_idx * half_depth;
        nk_f32_t const *sin_row = sin + row_idx * half_depth;
        nk_e4m3_t const *x_row = (nk_e4m3_t const *)((char const *)x + row_idx * x_stride);
        nk_e4m3_t *y_row = (nk_e4m3_t *)((char *)y + row_idx * y_stride);
        for (nk_size_t head_idx = 0; head_idx != head_count; ++head_idx) {
            nk_e4m3_t const *x_head = x_row + head_idx * depth;
            nk_e4m3_t *y_head = y_row + head_idx * depth;
            nk_size_t channel_idx = 0;
            for (; channel_idx + 4 <= half_depth; channel_idx += 4) {
                float32x4_t const low_f32x4 = nk_load_e4m3x4_as_f32_neon_(x_head + channel_idx);
                float32x4_t const high_f32x4 = nk_load_e4m3x4_as_f32_neon_(x_head + half_depth + channel_idx);
                float32x4_t const cos_f32x4 = vld1q_f32(cos_row + channel_idx);
                float32x4_t const sin_f32x4 = vld1q_f32(sin_row + channel_idx);
                nk_store_f32x4_as_e4m3_neon_(
                    y_head + channel_idx, vsubq_f32(vmulq_f32(low_f32x4, cos_f32x4), vmulq_f32(high_f32x4, sin_f32x4)));
                nk_store_f32x4_as_e4m3_neon_(
                    y_head + half_depth + channel_idx,
                    vaddq_f32(vmulq_f32(low_f32x4, sin_f32x4), vmulq_f32(high_f32x4, cos_f32x4)));
            }
            if (channel_idx == half_depth) continue;
            nk_size_t const count = half_depth - channel_idx;
            nk_b128_vec_t cos_vec, sin_vec;
            float32x4_t const low_f32x4 = nk_e4m3x4_to_f32x4_neon_(
                nk_partial_load_b8x4_serial_(x_head + channel_idx, count));
            float32x4_t const high_f32x4 = nk_e4m3x4_to_f32x4_neon_(
                nk_partial_load_b8x4_serial_(x_head + half_depth + channel_idx, count));
            nk_partial_load_b32x4_serial_(cos_row + channel_idx, &cos_vec, count);
            nk_partial_load_b32x4_serial_(sin_row + channel_idx, &sin_vec, count);
            nk_b32_vec_t const rotated_low_vec = nk_f32x4_to_e4m3x4_neon_(
                vsubq_f32(vmulq_f32(low_f32x4, cos_vec.f32x4), vmulq_f32(high_f32x4, sin_vec.f32x4)));
            nk_b32_vec_t const rotated_high_vec = nk_f32x4_to_e4m3x4_neon_(
                vaddq_f32(vmulq_f32(low_f32x4, sin_vec.f32x4), vmulq_f32(high_f32x4, cos_vec.f32x4)));
            nk_partial_store_b8x4_serial_(&rotated_low_vec, y_head + channel_idx, count);
            nk_partial_store_b8x4_serial_(&rotated_high_vec, y_head + half_depth + channel_idx, count);
        }
    }
    return nk_success_k;
}

#endif // NUMKONG_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_NEON_
#endif // NUMKONG_ARCH_ARM64_

#endif // NUMKONG_ATTENTION_NEON_H
