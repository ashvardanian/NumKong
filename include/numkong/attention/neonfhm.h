/**
 *  @file include/numkong/attention/neonfhm.h
 *  @author Ash Vardanian
 *  @date July 8, 2026
 *  @brief Arm NEON ragged attention for F16, E4M3 and NVFP4, using @c FMLAL over F16 planes.
 *
 *  @sa include/numkong/attention.h
 *
 *  Mirrors the @c v128relaxed panel-flash shape with the family-shared packed header, segment
 *  directory, base-2 streaming softmax, and [tasks_begin, tasks_end) windows. F16 planes keep the
 *  source encoding, and every E4M3 value converts exactly to F16 at the pack boundary, the Skylake
 *  precedent. NVFP4 planes hold each element times its UE4M3 block scale, which F16 keeps exactly:
 *  E2M1 and UE4M3 carry six significant bits between them, from 2⁻¹⁰ up to 2688. So the hot loops
 *  run pure half-precision: scores keep four KV rows in flight through widening @c FMLAL and
 *  @c FMLAL2 pairs into F32 accumulators, and the weighted V accumulation shares the F16
 *  weight-broadcast path. The NVFP4 tensor scales fold into the score multiplier and into the
 *  normalization of every output row.
 */
#ifndef NUMKONG_ATTENTION_NEONFHM_H
#define NUMKONG_ATTENTION_NEONFHM_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONFHM

#include <arm_neon.h>

#include "numkong/types.h"
#include "numkong/attention/serial.h" // `nk_attention_packed_header_t`, `nk_attention_pack_directory_serial_`
#include "numkong/attention/neon.h"   // `nk_attention_softmax_panel_neon_`, `nk_attention_e4m3_row_to_f16_neon_`
#include "numkong/each/neon.h"        // `nk_exp2_f32x4_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

enum {

    /** KV panel width in positions; the F32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_neonfhm_k_ = 512,

    /** Deepest head this backend handles in scratch; deeper heads route to the serial kernel. */
    nk_attention_max_depth_neonfhm_k_ = 256,
};

/** Scores of one panel of F16 keys against the F16 query, four KV rows in flight through widening
 *  @c FMLAL and @c FMLAL2 pairs into F32. */
NUMKONG_INLINE void nk_attention_scores_panel_f16_neonfhm_(nk_u16_t const *query_row, nk_f32_t *scores,
                                                           char const *keys_rows, nk_size_t panel_length,
                                                           nk_size_t depth_padded, nk_size_t plane_row_bytes) {
    nk_size_t position_idx = 0;
    for (; position_idx + 4 <= panel_length; position_idx += 4) {
        nk_u16_t const *keys_row0 = (nk_u16_t const *)(keys_rows + (position_idx + 0) * plane_row_bytes);
        nk_u16_t const *keys_row1 = (nk_u16_t const *)(keys_rows + (position_idx + 1) * plane_row_bytes);
        nk_u16_t const *keys_row2 = (nk_u16_t const *)(keys_rows + (position_idx + 2) * plane_row_bytes);
        nk_u16_t const *keys_row3 = (nk_u16_t const *)(keys_rows + (position_idx + 3) * plane_row_bytes);
        float32x4_t sum0_f32x4 = vdupq_n_f32(0.0f), sum1_f32x4 = vdupq_n_f32(0.0f);
        float32x4_t sum2_f32x4 = vdupq_n_f32(0.0f), sum3_f32x4 = vdupq_n_f32(0.0f);
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
            float16x8_t const query_f16x8 = vreinterpretq_f16_u16(vld1q_u16(query_row + channel_idx));
            float16x8_t const keys0_f16x8 = vreinterpretq_f16_u16(vld1q_u16(keys_row0 + channel_idx));
            float16x8_t const keys1_f16x8 = vreinterpretq_f16_u16(vld1q_u16(keys_row1 + channel_idx));
            float16x8_t const keys2_f16x8 = vreinterpretq_f16_u16(vld1q_u16(keys_row2 + channel_idx));
            float16x8_t const keys3_f16x8 = vreinterpretq_f16_u16(vld1q_u16(keys_row3 + channel_idx));
            sum0_f32x4 = vfmlalq_low_f16(sum0_f32x4, query_f16x8, keys0_f16x8);
            sum0_f32x4 = vfmlalq_high_f16(sum0_f32x4, query_f16x8, keys0_f16x8);
            sum1_f32x4 = vfmlalq_low_f16(sum1_f32x4, query_f16x8, keys1_f16x8);
            sum1_f32x4 = vfmlalq_high_f16(sum1_f32x4, query_f16x8, keys1_f16x8);
            sum2_f32x4 = vfmlalq_low_f16(sum2_f32x4, query_f16x8, keys2_f16x8);
            sum2_f32x4 = vfmlalq_high_f16(sum2_f32x4, query_f16x8, keys2_f16x8);
            sum3_f32x4 = vfmlalq_low_f16(sum3_f32x4, query_f16x8, keys3_f16x8);
            sum3_f32x4 = vfmlalq_high_f16(sum3_f32x4, query_f16x8, keys3_f16x8);
        }
        vst1q_f32(scores + position_idx,
                  vpaddq_f32(vpaddq_f32(sum0_f32x4, sum1_f32x4), vpaddq_f32(sum2_f32x4, sum3_f32x4)));
    }
    for (; position_idx < panel_length; position_idx++) {
        nk_u16_t const *keys_row = (nk_u16_t const *)(keys_rows + position_idx * plane_row_bytes);
        float32x4_t sum_f32x4 = vdupq_n_f32(0.0f);
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
            float16x8_t const query_f16x8 = vreinterpretq_f16_u16(vld1q_u16(query_row + channel_idx));
            float16x8_t const keys_f16x8 = vreinterpretq_f16_u16(vld1q_u16(keys_row + channel_idx));
            sum_f32x4 = vfmlalq_low_f16(sum_f32x4, query_f16x8, keys_f16x8);
            sum_f32x4 = vfmlalq_high_f16(sum_f32x4, query_f16x8, keys_f16x8);
        }
        scores[position_idx] = vaddvq_f32(sum_f32x4);
    }
}

/** Adds the panel's weights, rounded to F16, times its F16 V rows into @p output_row through
 *  @c FMLAL and @c FMLAL2. */
NUMKONG_INLINE void nk_attention_weighted_sum_f16_neonfhm_(nk_f32_t *output_row, nk_f32_t const *weights,
                                                           char const *values_rows, nk_size_t panel_length,
                                                           nk_size_t depth_padded, nk_size_t plane_row_bytes) {
    for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx++) {
        float16x8_t const weight_f16x8 = vdupq_n_f16((float16_t)weights[position_idx]);
        nk_u16_t const *values_row = (nk_u16_t const *)(values_rows + position_idx * plane_row_bytes);
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
            float16x8_t const values_f16x8 = vreinterpretq_f16_u16(vld1q_u16(values_row + channel_idx));
            vst1q_f32(output_row + channel_idx,
                      vfmlalq_low_f16(vld1q_f32(output_row + channel_idx), values_f16x8, weight_f16x8));
            vst1q_f32(output_row + channel_idx + 4,
                      vfmlalq_high_f16(vld1q_f32(output_row + channel_idx + 4), values_f16x8, weight_f16x8));
        }
    }
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_neonfhm(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes) {
    // Planes store the exact F16 widening of the E4M3 inputs, past the NEON depths in serial F32
    *bytes = depth > nk_attention_max_depth_neonfhm_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   depth * sizeof(nk_f32_t))
                 : nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   nk_size_round_up_to_multiple_(depth, 8) * sizeof(nk_f16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_neonfhm(void const *key_value_packed,
                                                               nk_size_t *key_value_head_count, nk_size_t *depth,
                                                               nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neonfhm_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_neonfhm(                    //
    nk_e4m3_t const *keys, nk_e4m3_t const *values,                        //
    nk_size_t key_value_head_count, nk_size_t depth,                       //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,              //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, //
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_neonfhm_k_) {
        nk_attention_pack_e4m3_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                       segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                       tasks_end, nk_cap_neonfhm_k);
        return nk_success_k;
    }
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const padded_row_bytes = depth_padded * sizeof(nk_f16_t);
    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, padded_row_bytes, nk_cap_neonfhm_k);
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

NUMKONG_API nk_status_t nk_attention_packed_e4m3_neonfhm(                        //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neonfhm_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_neonfhm_k_) {
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
    nk_size_t const panel_width = nk_attention_panel_neonfhm_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_u16_t query_row[nk_attention_max_depth_neonfhm_k_];
    nk_align_(64) nk_f32_t output_row[nk_attention_max_depth_neonfhm_k_];
    nk_align_(64) nk_f32_t scores[nk_attention_panel_neonfhm_k_];

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
                nk_attention_e4m3_row_to_f16_neon_(
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
                    nk_attention_scores_panel_f16_neonfhm_(query_row, scores,
                                                           keys_plane + panel_start * plane_row_bytes, panel_length,
                                                           depth_padded, plane_row_bytes);
                    nk_f32_t const correction = nk_attention_softmax_panel_neon_(scores, panel_length, scale2,
                                                                                 &running_max2, &running_sum);
                    nk_attention_scale_row_neon_(output_row, depth_padded, correction);
                    nk_attention_weighted_sum_f16_neonfhm_(output_row, scores,
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

NUMKONG_API nk_status_t nk_attention_pack_size_f16_neonfhm(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes) {
    // Planes keep the raw F16 encoding, past the NEON depths in serial F32
    *bytes = depth > nk_attention_max_depth_neonfhm_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   depth * sizeof(nk_f32_t))
                 : nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   nk_size_round_up_to_multiple_(depth, 8) * sizeof(nk_f16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_f16_neonfhm(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neonfhm_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_f16_neonfhm(                     //
    nk_f16_t const *keys, nk_f16_t const *values,                          //
    nk_size_t key_value_head_count, nk_size_t depth,                       //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,              //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, //
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_neonfhm_k_) {
        nk_attention_pack_f16_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                      segment_count, key_stride, value_stride, key_value_packed, tasks_begin, tasks_end,
                                      nk_cap_neonfhm_k);
        return nk_success_k;
    }
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const padded_row_bytes = depth_padded * sizeof(nk_f16_t);
    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 1, padded_row_bytes, nk_cap_neonfhm_k);
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

NUMKONG_API nk_status_t nk_attention_packed_f16_neonfhm(                         //
    nk_f16_t const *queries, void const *key_value_packed, nk_f32_t *output,     //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neonfhm_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_neonfhm_k_) {
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
    nk_size_t const panel_width = nk_attention_panel_neonfhm_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_u16_t query_row[nk_attention_max_depth_neonfhm_k_];
    nk_align_(64) nk_f32_t output_row[nk_attention_max_depth_neonfhm_k_];
    nk_align_(64) nk_f32_t scores[nk_attention_panel_neonfhm_k_];

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
                nk_attention_copy_row_b16_neon_((char const *)queries + (query_first + row_idx) * query_stride +
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
                    nk_attention_scores_panel_f16_neonfhm_(query_row, scores,
                                                           keys_plane + panel_start * plane_row_bytes, panel_length,
                                                           depth_padded, plane_row_bytes);
                    nk_f32_t const correction = nk_attention_softmax_panel_neon_(scores, panel_length, scale2,
                                                                                 &running_max2, &running_sum);
                    nk_attention_scale_row_neon_(output_row, depth_padded, correction);
                    nk_attention_weighted_sum_f16_neonfhm_(output_row, scores,
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

NUMKONG_API nk_status_t nk_attention_pack_size_nvfp4_neonfhm(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_size_t token_count, nk_size_t segment_count,
                                                             nk_size_t *bytes) {
    // Exact F16 products of elements and block scales, past the NEON depths in serial F32
    *bytes = depth > nk_attention_max_depth_neonfhm_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   depth * sizeof(nk_f32_t))
                 : nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   nk_size_round_up_to_multiple_(depth, 8) * sizeof(nk_f16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_nvfp4_neonfhm(void const *key_value_packed,
                                                                nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neonfhm_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_nvfp4_neonfhm(                   //
    nk_nvfp4_cref_t const *keys, nk_nvfp4_cref_t const *values,            //
    nk_size_t key_value_head_count, nk_size_t depth,                       //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,              //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, //
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth % 16 != 0) return nk_unexpected_dimensions_k;
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_neonfhm_k_) {
        nk_attention_pack_nvfp4_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                        tasks_end, nk_cap_neonfhm_k);
        return nk_success_k;
    }
    nk_attention_pack_nvfp4_neon_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                  key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, nk_cap_neonfhm_k);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_nvfp4_neonfhm(                          //
    nk_nvfp4_cref_t const *queries, void const *key_value_packed, nk_f32_t *output, //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count,    //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,         //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,                 //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neonfhm_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_neonfhm_k_) {
        nk_attention_packed_nvfp4_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                          key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                          scale, band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 16 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_u8_t const *query_codes = (nk_u8_t const *)queries->elements, *query_scales = (nk_u8_t const *)queries->scales;
    nk_size_t const query_scales_stride = query_stride / 8;
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const plane_row_bytes = depth_padded * sizeof(nk_f16_t);
    nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_serial_(
        nk_cross_tensor_scale_serial_(queries->tensor_scale), header->key_tensor_scale);
    // The power of two folds in exactly, so scores round as serial ones do unless subnormal
    nk_f32_t const scale2 = nk_scale_f32_serial_(scale * NUMKONG_F32_LOG2E_ * factor.mantissa, factor.exponent);
    nk_size_t const panel_width = nk_attention_panel_neonfhm_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_u16_t query_row[nk_attention_max_depth_neonfhm_k_];
    nk_align_(64) nk_f32_t output_row[nk_attention_max_depth_neonfhm_k_];
    nk_align_(64) nk_f32_t scores[nk_attention_panel_neonfhm_k_];

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
                nk_size_t const token = query_first + row_idx;
                nk_attention_nvfp4_row_to_f16_neon_(query_codes + token * query_stride + head_idx * depth / 2,
                                                    query_scales + token * query_scales_stride + head_idx * depth / 16,
                                                    query_row, depth);
                nk_attention_zero_row_neon_(output_row, depth_padded);
                nk_f32_t running_max2 = NUMKONG_F32_MIN, running_sum = 0;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    nk_attention_scores_panel_f16_neonfhm_(query_row, scores,
                                                           keys_plane + panel_start * plane_row_bytes, panel_length,
                                                           depth_padded, plane_row_bytes);
                    nk_f32_t const correction = nk_attention_softmax_panel_neon_(scores, panel_length, scale2,
                                                                                 &running_max2, &running_sum);
                    nk_attention_scale_row_neon_(output_row, depth_padded, correction);
                    nk_attention_weighted_sum_f16_neonfhm_(output_row, scores,
                                                           values_plane + panel_start * plane_row_bytes, panel_length,
                                                           depth_padded, plane_row_bytes);
                }

                nk_attention_store_row_neon_(output + token * output_stride_floats + head_idx * depth, output_row,
                                             depth, running_sum > 0 ? header->value_tensor_scale / running_sum : 0);
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(running_max2,
                                                                                                  running_sum);
            }
        }
    }
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

#endif // NUMKONG_TARGET_NEONFHM
#endif // NUMKONG_ARCH_ARM64_

#endif // NUMKONG_ATTENTION_NEONFHM_H
