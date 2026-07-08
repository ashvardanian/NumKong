/**
 *  @brief Arm NEON ragged attention backend for I8, using `SDOT`.
 *  @file include/numkong/attention/neonsdot.h
 *  @author Ash Vardanian
 *  @date July 8, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  Mirrors the `v128relaxed` panel-flash shape with the family-shared packed header, segment
 *  directory, base-2 streaming softmax, and `(first_task, task_count)` windows. Scores stay
 *  exact in I32: four KV rows in flight through `SDOT` over row-major I8 planes, with one
 *  lane-wise reduction per score. Softmax weights quantize to `trunc(2^(s₂−m₂)·255 + 0.5)`
 *  like the whole I8 family — the maximum position lands on exactly 255, so the weight sum
 *  never vanishes and the 255 cancels in normalization. The weighted V accumulation widens
 *  I8 rows to F32, so the backend needs only the baseline `dotprod` extension.
 */
#ifndef NK_ATTENTION_NEONSDOT_H
#define NK_ATTENTION_NEONSDOT_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEONSDOT

#include <arm_neon.h>

#include "numkong/types.h"
#include "numkong/attention/serial.h" // `nk_attention_packed_header_t`, `nk_attention_pack_directory_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+dotprod")
#endif

enum {
    /** KV panel width in positions; the F32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_neonsdot_k_ = 512,
    /** Deepest head this backend handles in scratch; deeper heads route to the serial tier. */
    nk_attention_max_depth_neonsdot_k_ = 256,
};

/** @brief Fast vectorized 2^x: exact range reduction + the family's shared degree-4 polynomial. */
NK_INTERNAL float32x4_t nk_attention_exp2_f32x4_neonsdot_(float32x4_t x_f32x4) {
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

NK_PUBLIC nk_size_t nk_attention_packed_size_i8_neonsdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_neonsdot_k_)
        return nk_attention_packed_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count);
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 16);
    nk_size_t payload_bytes = 0; // raw I8 planes, channels zero-padded to the SDOT chunk
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count * (nk_size_t)segment_lengths[segment_idx] * depth_padded;
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_PUBLIC void nk_attention_pack_i8_neonsdot(                                          //
    nk_i8_t const *keys, nk_i8_t const *values,                                        //
    nk_size_t key_value_head_count, nk_size_t depth,                                   //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                  //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, //
    void *key_value_packed, nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_neonsdot_k_) {
        nk_attention_pack_i8_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                    segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                    task_count);
        return;
    }

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 16);
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, 1, depth_padded);
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
        nk_size_t const plane_bytes = position_count * depth_padded;
        char *keys_plane = payload_base + payload_offsets[segment_idx] + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride_bytes +
                                   key_value_head_idx * depth;
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride_bytes +
                                     key_value_head_idx * depth;
            char *keys_destination = keys_plane + position_idx * depth_padded;
            char *values_destination = values_plane + position_idx * depth_padded;
            nk_size_t byte_idx = 0;
            for (; byte_idx + 16 <= depth; byte_idx += 16)
                vst1q_u8((uint8_t *)(keys_destination + byte_idx), vld1q_u8((uint8_t const *)(keys_row + byte_idx)));
            for (; byte_idx < depth; byte_idx++) keys_destination[byte_idx] = keys_row[byte_idx];
            for (; byte_idx < depth_padded; byte_idx++) keys_destination[byte_idx] = 0;
            for (byte_idx = 0; byte_idx + 16 <= depth; byte_idx += 16)
                vst1q_u8((uint8_t *)(values_destination + byte_idx),
                         vld1q_u8((uint8_t const *)(values_row + byte_idx)));
            for (; byte_idx < depth; byte_idx++) values_destination[byte_idx] = values_row[byte_idx];
            for (; byte_idx < depth_padded; byte_idx++) values_destination[byte_idx] = 0;
        }
    }
}

NK_PUBLIC void nk_attention_packed_i8_neonsdot(                                  //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,      //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_neonsdot_k_) {
        nk_attention_packed_i8_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                      query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                      task_count);
        return;
    }

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->key_value_head_count != key_value_head_count) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_(segment_count);
    nk_size_t const output_stride_floats = output_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 16);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // softmax(x) = softmax₂(x·log₂e)
    nk_size_t const panel_width = nk_attention_panel_neonsdot_k_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_i8_t query_row[nk_attention_max_depth_neonsdot_k_];
    NK_ALIGN64 nk_f32_t output_row[nk_attention_max_depth_neonsdot_k_];
    NK_ALIGN64 nk_f32_t scores[nk_attention_panel_neonsdot_k_];

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const plane_bytes = position_count * depth_padded;
        char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                 (head_idx / head_group_size) * plane_bytes;
        char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

        for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
            nk_i8_t const *query_source =
                (nk_i8_t const *)((char const *)queries + (query_offsets[segment_idx] + row_idx) * query_stride_bytes) +
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
                // Score sweep: exact I32 dots, four KV rows in flight per query-vector load.
                for (; position_idx + 4 <= panel_length; position_idx += 4) {
                    char const *keys_row0 = keys_plane + (panel_start + position_idx + 0) * depth_padded;
                    char const *keys_row1 = keys_plane + (panel_start + position_idx + 1) * depth_padded;
                    char const *keys_row2 = keys_plane + (panel_start + position_idx + 2) * depth_padded;
                    char const *keys_row3 = keys_plane + (panel_start + position_idx + 3) * depth_padded;
                    int32x4_t sum0_i32x4 = vdupq_n_s32(0), sum1_i32x4 = vdupq_n_s32(0);
                    int32x4_t sum2_i32x4 = vdupq_n_s32(0), sum3_i32x4 = vdupq_n_s32(0);
                    for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 16) {
                        int8x16_t const query_i8x16 = vld1q_s8((int8_t const *)(query_row + channel_idx));
                        sum0_i32x4 = vdotq_s32(sum0_i32x4, query_i8x16,
                                               vld1q_s8((int8_t const *)(keys_row0 + channel_idx)));
                        sum1_i32x4 = vdotq_s32(sum1_i32x4, query_i8x16,
                                               vld1q_s8((int8_t const *)(keys_row1 + channel_idx)));
                        sum2_i32x4 = vdotq_s32(sum2_i32x4, query_i8x16,
                                               vld1q_s8((int8_t const *)(keys_row2 + channel_idx)));
                        sum3_i32x4 = vdotq_s32(sum3_i32x4, query_i8x16,
                                               vld1q_s8((int8_t const *)(keys_row3 + channel_idx)));
                    }
                    int32x4_t const sums_i32x4 = vpaddq_s32(vpaddq_s32(sum0_i32x4, sum1_i32x4),
                                                            vpaddq_s32(sum2_i32x4, sum3_i32x4));
                    float32x4_t const scores2_f32x4 = vmulq_f32(vcvtq_f32_s32(sums_i32x4), scale2_f32x4);
                    max_f32x4 = vmaxq_f32(max_f32x4, scores2_f32x4);
                    vst1q_f32(scores + position_idx, scores2_f32x4);
                }
                nk_f32_t panel_max2 = vmaxvq_f32(max_f32x4);
                for (; position_idx < panel_length; position_idx++) {
                    char const *keys_row = keys_plane + (panel_start + position_idx) * depth_padded;
                    int32x4_t sum_i32x4 = vdupq_n_s32(0);
                    for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 16)
                        sum_i32x4 = vdotq_s32(sum_i32x4, vld1q_s8((int8_t const *)(query_row + channel_idx)),
                                              vld1q_s8((int8_t const *)(keys_row + channel_idx)));
                    scores[position_idx] = (nk_f32_t)vaddvq_s32(sum_i32x4) * scale2;
                    if (scores[position_idx] > panel_max2) panel_max2 = scores[position_idx];
                }

                nk_f32_t const new_max2 = running_max2 > panel_max2 ? running_max2 : panel_max2;
                nk_f32_t const correction = vgetq_lane_f32(
                    nk_attention_exp2_f32x4_neonsdot_(vdupq_n_f32(running_max2 - new_max2)), 0);
                running_max2 = new_max2;

                float32x4_t const new_max2_f32x4 = vdupq_n_f32(new_max2);
                float32x4_t panel_sum_f32x4 = vdupq_n_f32(0.0f);
                nk_f32_t panel_sum = 0;
                for (position_idx = 0; position_idx + 4 <= panel_length; position_idx += 4) {
                    float32x4_t const exponent_f32x4 = nk_attention_exp2_f32x4_neonsdot_(
                        vsubq_f32(vld1q_f32(scores + position_idx), new_max2_f32x4));
                    float32x4_t const weight_f32x4 = vcvtq_f32_u32( // mul then add matches the serial rounding

                        vcvtq_u32_f32(vaddq_f32(vmulq_f32(exponent_f32x4, vdupq_n_f32(255.0f)), vdupq_n_f32(0.5f))));
                    panel_sum_f32x4 = vaddq_f32(panel_sum_f32x4, weight_f32x4);
                    vst1q_f32(scores + position_idx, weight_f32x4);
                }
                panel_sum = vaddvq_f32(panel_sum_f32x4);
                for (; position_idx < panel_length; position_idx++) { // scalar tail keeps the family exp2 end-to-end
                    nk_f32_t const weight =
                        (nk_f32_t)(nk_u32_t)(nk_f32_exp2_serial_(scores[position_idx] - new_max2) * 255.0f + 0.5f);
                    scores[position_idx] = weight;
                    panel_sum += weight;
                }
                running_sum = running_sum * correction + panel_sum;

                float32x4_t const correction_f32x4 = vdupq_n_f32(correction);
                for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                    vst1q_f32(output_row + channel_idx,
                              vmulq_f32(vld1q_f32(output_row + channel_idx), correction_f32x4));
                for (position_idx = 0; position_idx < panel_length; position_idx++) {
                    if (scores[position_idx] == 0.0f) continue; // zero weights add nothing
                    float32x4_t const weight_f32x4 = vdupq_n_f32(scores[position_idx]);
                    char const *values_row = values_plane + (panel_start + position_idx) * depth_padded;
                    for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
                        int16x8_t const values_i16x8 = vmovl_s8(vld1_s8((int8_t const *)(values_row + channel_idx)));
                        float32x4_t const values_low_f32x4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(values_i16x8)));
                        float32x4_t const values_high_f32x4 = vcvtq_f32_s32(vmovl_high_s16(values_i16x8));
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

#endif // NK_TARGET_NEONSDOT
#endif // NK_TARGET_ARM64_

#endif // NK_ATTENTION_NEONSDOT_H
