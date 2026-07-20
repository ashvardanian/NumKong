/**
 *  @brief Arm NEON ragged attention backend for BF16, using `BFDOT`.
 *  @file include/numkong/attention/neonbfdot.h
 *  @author Ash Vardanian
 *  @date July 8, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  Mirrors the `v128relaxed` panel-flash shape with the family-shared packed header, segment
 *  directory, base-2 streaming softmax, and `(first_task, task_count)` windows. Scores run
 *  four KV rows in flight through `BFDOT` (one query-vector load feeds four dot steps), the
 *  softmax stays in F32, and the weighted V accumulation widens BF16 rows with one `SHLL`
 *  pair per eight channels. K/V planes keep the raw BF16 encoding, channels zero-padded to
 *  eight for the dot lanes.
 */
#ifndef NK_ATTENTION_NEONBFDOT_H
#define NK_ATTENTION_NEONBFDOT_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEONBFDOT

#include <arm_neon.h>

#include "numkong/types.h"
#include "numkong/attention/serial.h" // `nk_attention_packed_header_t`, `nk_attention_pack_directory_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#endif

enum {
    /** KV panel width in positions; the F32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_neonbfdot_k_ = 512,
    /** Deepest head this backend handles in scratch; deeper heads route to the serial tier. */
    nk_attention_max_depth_neonbfdot_k_ = 256,
};

/** @brief Fast vectorized 2^x: exact range reduction + the family's shared degree-4 polynomial. */
NK_HELPER_INLINE float32x4_t nk_attention_exp2_f32x4_neonbfdot_(float32x4_t x_f32x4) {
    x_f32x4 = vmaxq_f32(vminq_f32(x_f32x4, vdupq_n_f32(127.0f)), vdupq_n_f32(-125.0f));
    float32x4_t const whole_f32x4 = vrndnq_f32(x_f32x4);
    float32x4_t const reduced_f32x4 = vsubq_f32(x_f32x4, whole_f32x4);
    float32x4_t poly_f32x4 = vdupq_n_f32(9.61812910e-3f);
    poly_f32x4 = vfmaq_f32(vdupq_n_f32(5.55041087e-2f), poly_f32x4, reduced_f32x4);
    poly_f32x4 = vfmaq_f32(vdupq_n_f32(2.40226507e-1f), poly_f32x4, reduced_f32x4);
    poly_f32x4 = vfmaq_f32(vdupq_n_f32(6.93147181e-1f), poly_f32x4, reduced_f32x4);
    poly_f32x4 = vfmaq_f32(vdupq_n_f32(1.0f), poly_f32x4, reduced_f32x4);
    int32x4_t const whole_i32x4 = vcvtq_s32_f32(whole_f32x4); // integral and clamped, so exact
    float32x4_t const power_f32x4 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(whole_i32x4, vdupq_n_s32(127)), 23));
    return vmulq_f32(poly_f32x4, power_f32x4);
}

NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_neonbfdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_u32_t const *segment_lengths,
                                                                nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_neonbfdot_k_)
        return nk_attention_pack_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count);
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t payload_bytes = 0; // planes keep the raw BF16 encoding
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count * (nk_size_t)segment_lengths[segment_idx] * depth_padded *
                         sizeof(nk_bf16_t);
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_API_COMPTIME void nk_attention_packed_shape_bf16_neonbfdot(void const *key_value_packed, nk_size_t *heads,
                                                              nk_size_t *depth, nk_size_t *segments) {
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
}

NK_API_COMPTIME void nk_attention_pack_bf16_neonbfdot(                                 //
    nk_bf16_t const *keys, nk_bf16_t const *values,                                    //
    nk_size_t key_value_head_count, nk_size_t depth,                                   //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                  //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, //
    void *key_value_packed, nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_neonbfdot_k_) {
        nk_attention_pack_bf16_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const padded_row_bytes = depth_padded * sizeof(nk_bf16_t);
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, 1, padded_row_bytes);
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *payload_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count;
        nk_size_t const key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * padded_row_bytes;
        char *keys_plane = payload_base + payload_offsets[segment_idx] + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        nk_size_t const row_bytes = depth * sizeof(nk_bf16_t);
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride_bytes +
                                   key_value_head_idx * row_bytes;
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride_bytes +
                                     key_value_head_idx * row_bytes;
            char *keys_destination = keys_plane + position_idx * padded_row_bytes;
            char *values_destination = values_plane + position_idx * padded_row_bytes;
            nk_size_t byte_idx = 0;
            for (; byte_idx + 16 <= row_bytes; byte_idx += 16)
                vst1q_u8((uint8_t *)(keys_destination + byte_idx), vld1q_u8((uint8_t const *)(keys_row + byte_idx)));
            for (; byte_idx < row_bytes; byte_idx++) keys_destination[byte_idx] = keys_row[byte_idx];
            for (; byte_idx < padded_row_bytes; byte_idx++) keys_destination[byte_idx] = 0;
            for (byte_idx = 0; byte_idx + 16 <= row_bytes; byte_idx += 16)
                vst1q_u8((uint8_t *)(values_destination + byte_idx),
                         vld1q_u8((uint8_t const *)(values_row + byte_idx)));
            for (; byte_idx < row_bytes; byte_idx++) values_destination[byte_idx] = values_row[byte_idx];
            for (; byte_idx < padded_row_bytes; byte_idx++) values_destination[byte_idx] = 0;
        }
    }
}

NK_API_COMPTIME void nk_attention_packed_bf16_neonbfdot(                         //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_neonbfdot_k_) {
        nk_attention_packed_bf16_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->heads != key_value_head_count) return;
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_(segment_count);
    nk_size_t const output_stride_floats = output_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const plane_row_bytes = depth_padded * sizeof(nk_bf16_t);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // softmax(x) = softmax₂(x·log₂e)
    nk_size_t const panel_width = nk_attention_panel_neonbfdot_k_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_u16_t query_row[nk_attention_max_depth_neonbfdot_k_];
    NK_ALIGN64 nk_f32_t output_row[nk_attention_max_depth_neonbfdot_k_];
    NK_ALIGN64 nk_f32_t scores[nk_attention_panel_neonbfdot_k_];

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const plane_bytes = position_count * plane_row_bytes;
        char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                 (head_idx / head_group_size) * plane_bytes;
        char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

        for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
            nk_u16_t const *query_source = (nk_u16_t const *)((char const *)queries +
                                                              (query_offsets[segment_idx] + row_idx) *
                                                                  query_stride_bytes) +
                                           head_idx * depth;
            nk_size_t channel_idx = 0;
            for (; channel_idx < depth; channel_idx++) query_row[channel_idx] = query_source[channel_idx];
            for (; channel_idx < depth_padded; channel_idx++) query_row[channel_idx] = 0;
            for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                vst1q_f32(output_row + channel_idx, vdupq_n_f32(0.0f));
            nk_f32_t running_max2 = NK_F32_MIN, running_sum = 0;

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_length = (panel_start + panel_width <= position_count)
                                                   ? panel_width
                                                   : (position_count - panel_start);

                nk_size_t position_idx = 0;
                float32x4_t const scale2_f32x4 = vdupq_n_f32(scale2); // folded into the score store
                float32x4_t max_f32x4 = vdupq_n_f32(NK_F32_MIN);
                // Score sweep: four KV rows in flight so each query-vector load feeds four BFDOTs.
                for (; position_idx + 4 <= panel_length; position_idx += 4) {
                    char const *keys_row0 = keys_plane + (panel_start + position_idx + 0) * plane_row_bytes;
                    char const *keys_row1 = keys_plane + (panel_start + position_idx + 1) * plane_row_bytes;
                    char const *keys_row2 = keys_plane + (panel_start + position_idx + 2) * plane_row_bytes;
                    char const *keys_row3 = keys_plane + (panel_start + position_idx + 3) * plane_row_bytes;
                    float32x4_t sum0_f32x4 = vdupq_n_f32(0.0f), sum1_f32x4 = vdupq_n_f32(0.0f);
                    float32x4_t sum2_f32x4 = vdupq_n_f32(0.0f), sum3_f32x4 = vdupq_n_f32(0.0f);
                    for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
                        bfloat16x8_t const query_bf16x8 = vreinterpretq_bf16_u16(vld1q_u16(query_row + channel_idx));
                        nk_size_t const chunk_bytes = channel_idx * sizeof(nk_bf16_t);
                        sum0_f32x4 = vbfdotq_f32(
                            sum0_f32x4, query_bf16x8,
                            vreinterpretq_bf16_u16(vld1q_u16((nk_u16_t const *)(keys_row0 + chunk_bytes))));
                        sum1_f32x4 = vbfdotq_f32(
                            sum1_f32x4, query_bf16x8,
                            vreinterpretq_bf16_u16(vld1q_u16((nk_u16_t const *)(keys_row1 + chunk_bytes))));
                        sum2_f32x4 = vbfdotq_f32(
                            sum2_f32x4, query_bf16x8,
                            vreinterpretq_bf16_u16(vld1q_u16((nk_u16_t const *)(keys_row2 + chunk_bytes))));
                        sum3_f32x4 = vbfdotq_f32(
                            sum3_f32x4, query_bf16x8,
                            vreinterpretq_bf16_u16(vld1q_u16((nk_u16_t const *)(keys_row3 + chunk_bytes))));
                    }
                    float32x4_t const sums_f32x4 = vpaddq_f32(vpaddq_f32(sum0_f32x4, sum1_f32x4),
                                                              vpaddq_f32(sum2_f32x4, sum3_f32x4));
                    float32x4_t const scores2_f32x4 = vmulq_f32(sums_f32x4, scale2_f32x4);
                    max_f32x4 = vmaxq_f32(max_f32x4, scores2_f32x4);
                    vst1q_f32(scores + position_idx, scores2_f32x4);
                }
                nk_f32_t panel_max2 = vmaxvq_f32(max_f32x4);
                for (; position_idx < panel_length; position_idx++) {
                    char const *keys_row = keys_plane + (panel_start + position_idx) * plane_row_bytes;
                    float32x4_t sum_f32x4 = vdupq_n_f32(0.0f);
                    for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
                        sum_f32x4 = vbfdotq_f32(sum_f32x4, vreinterpretq_bf16_u16(vld1q_u16(query_row + channel_idx)),
                                                vreinterpretq_bf16_u16(vld1q_u16(
                                                    (nk_u16_t const *)(keys_row + channel_idx * sizeof(nk_bf16_t)))));
                    scores[position_idx] = vaddvq_f32(sum_f32x4) * scale2;
                    if (scores[position_idx] > panel_max2) panel_max2 = scores[position_idx];
                }

                nk_f32_t const new_max2 = running_max2 > panel_max2 ? running_max2 : panel_max2;
                nk_f32_t const correction = vgetq_lane_f32(
                    nk_attention_exp2_f32x4_neonbfdot_(vdupq_n_f32(running_max2 - new_max2)), 0);
                running_max2 = new_max2;

                float32x4_t const new_max2_f32x4 = vdupq_n_f32(new_max2);
                float32x4_t panel_sum_f32x4 = vdupq_n_f32(0.0f);
                nk_f32_t panel_sum = 0;
                for (position_idx = 0; position_idx + 4 <= panel_length; position_idx += 4) {
                    float32x4_t const weight_f32x4 = nk_attention_exp2_f32x4_neonbfdot_(
                        vsubq_f32(vld1q_f32(scores + position_idx), new_max2_f32x4));
                    panel_sum_f32x4 = vaddq_f32(panel_sum_f32x4, weight_f32x4);
                    vst1q_f32(scores + position_idx, weight_f32x4);
                }
                panel_sum = vaddvq_f32(panel_sum_f32x4);
                for (; position_idx < panel_length; position_idx++) { // scalar tail keeps the family exp2 end-to-end
                    nk_f32_t const weight = nk_f32_exp2_serial_(scores[position_idx] - new_max2);
                    scores[position_idx] = weight;
                    panel_sum += weight;
                }
                running_sum = running_sum * correction + panel_sum;

                float32x4_t const correction_f32x4 = vdupq_n_f32(correction);
                for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                    vst1q_f32(output_row + channel_idx,
                              vmulq_f32(vld1q_f32(output_row + channel_idx), correction_f32x4));
                for (position_idx = 0; position_idx < panel_length; position_idx++) {
                    float32x4_t const weight_f32x4 = vdupq_n_f32(scores[position_idx]);
                    char const *values_row = values_plane + (panel_start + position_idx) * plane_row_bytes;
                    for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
                        uint16x8_t const values_u16x8 = vld1q_u16(
                            (nk_u16_t const *)(values_row + channel_idx * sizeof(nk_bf16_t)));
                        float32x4_t const values_low_f32x4 = vreinterpretq_f32_u32(
                            vshll_n_u16(vget_low_u16(values_u16x8), 16));
                        float32x4_t const values_high_f32x4 = vreinterpretq_f32_u32(vshll_high_n_u16(values_u16x8, 16));
                        vst1q_f32(output_row + channel_idx,
                                  vfmaq_f32(vld1q_f32(output_row + channel_idx), weight_f32x4, values_low_f32x4));
                        vst1q_f32(output_row + channel_idx + 4,
                                  vfmaq_f32(vld1q_f32(output_row + channel_idx + 4), weight_f32x4, values_high_f32x4));
                    }
                }
            }

            nk_f32_t const inverse_sum = 1.0f / running_sum;
            float32x4_t const inverse_sum_f32x4 = vdupq_n_f32(inverse_sum);
            nk_f32_t *destination = output + (query_offsets[segment_idx] + row_idx) * output_stride_floats +
                                    head_idx * depth;
            for (channel_idx = 0; channel_idx + 4 <= depth; channel_idx += 4)
                vst1q_f32(destination + channel_idx, vmulq_f32(vld1q_f32(output_row + channel_idx), inverse_sum_f32x4));
            for (; channel_idx < depth; channel_idx++) destination[channel_idx] = output_row[channel_idx] * inverse_sum;
        }
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM64_

#endif // NK_ATTENTION_NEONBFDOT_H
