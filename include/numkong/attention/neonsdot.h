/**
 *  @file include/numkong/attention/neonsdot.h
 *  @author Ash Vardanian
 *  @date July 8, 2026
 *  @brief Arm NEON ragged attention backend for I8, using @c SDOT.
 *
 *  @sa include/numkong/attention.h
 *
 *  Mirrors the @c v128relaxed panel-flash shape with the family-shared packed header, segment
 *  directory, base-2 streaming softmax, and [tasks_begin, tasks_end) windows. Scores stay exact in
 *  I32: four KV rows in flight through @c SDOT over row-major I8 planes, with one lane-wise
 *  reduction per score. Softmax weights quantize to trunc(2^(s₂−m₂) · 255 + 0.5) like the whole I8
 *  family — the maximum position lands on exactly 255, so the weight sum never vanishes and the 255
 *  cancels in normalization. The weighted V accumulation runs as @c UDOT over V tiles packed
 *  4-positions × 4-channels with a +128 offset. Since the weights are U8, the identity holds
 *  exactly: Σ w · (v+128) − 128 · Σw = Σ w · v. The bias subtracts in integer before the single F32
 *  conversion, and every panel total stays under 2^24 — bit-exact with serial at a 6× faster inner
 *  loop, still on the baseline @c dotprod extension.
 */
#ifndef NUMKONG_ATTENTION_NEONSDOT_H
#define NUMKONG_ATTENTION_NEONSDOT_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONSDOT

#include <arm_neon.h>

#include "numkong/types.h"
#include "numkong/attention/serial.h" // `nk_attention_packed_header_t`, `nk_attention_pack_directory_serial_`
#include "numkong/cast/neon.h"        // `nk_load_b128_neon_`
#include "numkong/each/neon.h"        // `nk_exp2_f32x4_neon_`, `nk_exp2_u8_i32x4_neon_`

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

    /** Deepest head this backend handles in scratch; deeper heads route to the serial kernel. */
    nk_attention_max_depth_neonsdot_k_ = 256,
};

NUMKONG_API nk_status_t nk_attention_pack_size_i8_neonsdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_size_t token_count, nk_size_t segment_count,
                                                           nk_size_t *bytes) {
    // K rows plus quad-tiled V, positions padded to the UDOT quad
    *bytes = depth > nk_attention_max_depth_neonsdot_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1, depth)
                 : nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 4,
                                                   nk_size_round_up_to_multiple_(depth, 16));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_neonsdot(void const *key_value_packed,
                                                              nk_size_t *key_value_head_count, nk_size_t *depth,
                                                              nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neonsdot_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_i8_neonsdot(                     //
    nk_i8_t const *keys, nk_i8_t const *values,                            //
    nk_size_t key_value_head_count, nk_size_t depth,                       //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,              //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, //
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_neonsdot_k_) {
        nk_attention_pack_i8_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                     key_stride, value_stride, key_value_packed, tasks_begin, tasks_end,
                                     nk_cap_neonsdot_k);
        return nk_success_k;
    }
    static nk_u8_t const lane_indices_u8x16[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8x16_t const lanes_u8x16 = vld1q_u8(lane_indices_u8x16);

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 16);
    nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, tasks_begin, 4, depth_padded, nk_cap_neonsdot_k);
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
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count, 4,
                depth_padded);
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 4);
        nk_size_t const plane_bytes = position_count_padded * depth_padded;
        char *keys_plane = payload_base + payload_offset + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count_padded; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride +
                                   key_value_head_idx * depth;
            nk_u8_t *keys_destination = (nk_u8_t *)(keys_plane + position_idx * depth_padded);
            for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16) {
                nk_b128_vec_t keys_vec;
                if (position_idx < position_count && channel_idx + 16 <= depth)
                    nk_load_b128_neon_(keys_row + channel_idx, &keys_vec);
                else {
                    keys_vec.u8x16 = vdupq_n_u8(0);
                    if (position_idx < position_count)
                        for (nk_size_t lane = 0; channel_idx + lane < depth; lane++)
                            keys_vec.u8s[lane] = (nk_u8_t)keys_row[channel_idx + lane];
                }
                vst1q_u8(keys_destination + channel_idx, keys_vec.u8x16);
            }
        }
        // V tiles as [position_quad][channel][4 positions] with a +128 offset to U8, one XOR of
        // the sign bit on the live channels, so one UDOT covers 4 positions × 4 channels;
        // `vst4q_u8` interleaves four source rows per 16 channels. Padded positions and channels
        // carry zero weights and only need deterministic content, zeros here.
        for (nk_size_t quad_idx = 0; quad_idx < position_count_padded / 4; quad_idx++) {
            nk_u8_t *tile = (nk_u8_t *)(values_plane + quad_idx * depth_padded * 4);
            for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16) {
                uint8x16_t const flips_u8x16 = vandq_u8(
                    vcltq_u8(lanes_u8x16, vdupq_n_u8((nk_u8_t)nk_min_of_two(16, depth - channel_idx))),
                    vdupq_n_u8(0x80));
                uint8x16x4_t rows_u8x16x4;
                for (nk_size_t lane = 0; lane < 4; lane++) {
                    nk_size_t const position_idx = quad_idx * 4 + lane;
                    char const *values_row = (char const *)values + (position_first + position_idx) * value_stride +
                                             key_value_head_idx * depth + channel_idx;
                    nk_b128_vec_t values_vec;
                    int const live = position_idx < position_count;
                    if (live && channel_idx + 16 <= depth) nk_load_b128_neon_(values_row, &values_vec);
                    else {
                        values_vec.u8x16 = vdupq_n_u8(0);
                        if (live)
                            for (nk_size_t lane = 0; channel_idx + lane < depth; lane++)
                                values_vec.u8s[lane] = (nk_u8_t)values_row[lane];
                    }
                    rows_u8x16x4.val[lane] = live ? veorq_u8(values_vec.u8x16, flips_u8x16) : values_vec.u8x16;
                }
                vst4q_u8(tile + channel_idx * 4, rows_u8x16x4);
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_i8_neonsdot(                         //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,      //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_neonsdot_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_neonsdot_k_) {
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
    nk_size_t const panel_width = nk_attention_panel_neonsdot_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_i8_t query_row[nk_attention_max_depth_neonsdot_k_];
    nk_align_(64) nk_f32_t output_row[nk_attention_max_depth_neonsdot_k_];
    nk_align_(64) nk_u32_t output_totals[nk_attention_max_depth_neonsdot_k_];
    // Raw I32 QK dots and U8 weights; 4 slack slots absorb the quad tail of a row starting mid-quad.
    nk_align_(64) nk_i32_t scores[nk_attention_panel_neonsdot_k_ + 4];
    nk_align_(64) nk_u8_t weights[nk_attention_panel_neonsdot_k_ + 4];

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
            nk_size_t const plane_bytes = nk_size_round_up_to_multiple_(position_count, 4) * depth_padded;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                     (head_idx / head_group_size) * plane_bytes;
            char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_i8_t const *query_source = (nk_i8_t const *)((char const *)queries +
                                                                (query_first + row_idx) * query_stride) +
                                              head_idx * depth;
                nk_size_t channel_idx = 0;
                for (; channel_idx < depth; channel_idx++) query_row[channel_idx] = query_source[channel_idx];
                for (; channel_idx < depth_padded; channel_idx++) query_row[channel_idx] = 0;
                for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                    vst1q_f32(output_row + channel_idx, vdupq_n_f32(0.0f));
                nk_i32_t running_max = NUMKONG_I32_MIN;
                nk_f32_t running_sum = 0;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                // Panels start on a V quad; the first skips the up-to-3 keys before `key_begin`.
                for (nk_size_t panel_start = key_begin & ~(nk_size_t)3; panel_start < key_end;
                     panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    nk_size_t const range_begin = key_begin > panel_start ? key_begin - panel_start : 0;
                    for (nk_size_t position_idx = 0; position_idx < range_begin; position_idx++)
                        weights[position_idx] = 0;

                    nk_size_t position_idx = range_begin;
                    int32x4_t max_i32x4 = vdupq_n_s32(NUMKONG_I32_MIN); // exact integer row max over raw I32 scores
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
                        max_i32x4 = vmaxq_s32(max_i32x4, sums_i32x4);
                        vst1q_s32(scores + position_idx, sums_i32x4);
                    }
                    nk_i32_t panel_max = vmaxvq_s32(max_i32x4);
                    for (; position_idx < panel_length; position_idx++) {
                        char const *keys_row = keys_plane + (panel_start + position_idx) * depth_padded;
                        int32x4_t sum_i32x4 = vdupq_n_s32(0);
                        for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 16)
                            sum_i32x4 = vdotq_s32(sum_i32x4, vld1q_s8((int8_t const *)(query_row + channel_idx)),
                                                  vld1q_s8((int8_t const *)(keys_row + channel_idx)));
                        nk_i32_t const score = vaddvq_s32(sum_i32x4);
                        scores[position_idx] = score;
                        if (score > panel_max) panel_max = score;
                    }

                    nk_i32_t const new_max = running_max > panel_max ? running_max : panel_max;
                    nk_f32_t const correction = vgetq_lane_f32(
                        nk_exp2_f32x4_neon_(vdupq_n_f32(((nk_f32_t)running_max - (nk_f32_t)new_max) * scale2)), 0);
                    running_max = new_max;

                    int32x4_t const new_max_i32x4 = vdupq_n_s32(new_max);
                    int32x4_t const scale_fixed_i32x4 = vdupq_n_s32(scale_fixed);
                    int32x4_t const delta_floor_i32x4 = vdupq_n_s32(delta_floor);
                    nk_f32_t panel_sum = 0;
                    uint32x4_t panel_sum_u32x4 = vdupq_n_u32(0); // weights are U8 over <= 512 positions
                    for (position_idx = range_begin; position_idx + 4 <= panel_length; position_idx += 4) {
                        int32x4_t const delta_i32x4 = vmaxq_s32(
                            vsubq_s32(vld1q_s32(scores + position_idx), new_max_i32x4), delta_floor_i32x4);
                        uint32x4_t const weight_u32x4 = vreinterpretq_u32_s32(
                            nk_exp2_u8_i32x4_neon_(vmulq_s32(delta_i32x4, scale_fixed_i32x4)));
                        panel_sum_u32x4 = vaddq_u32(panel_sum_u32x4, weight_u32x4);
                        uint8x8_t const weight_u8x8 = vmovn_u16(vcombine_u16(vmovn_u32(weight_u32x4), vdup_n_u16(0)));
                        vst1_lane_u32((nk_u32_t *)(weights + position_idx), vreinterpret_u32_u8(weight_u8x8), 0);
                    }
                    if (position_idx < panel_length) { // masked vector tail — no scalar exp2, padded lanes forced to 0
                        nk_u32_t const lane_indices_u32[4] = {0, 1, 2, 3};
                        uint32x4_t const lane_index_u32x4 = vld1q_u32(lane_indices_u32);
                        uint32x4_t const tail_mask_u32x4 = vcltq_u32(
                            lane_index_u32x4, vdupq_n_u32((nk_u32_t)(panel_length - position_idx)));
                        int32x4_t const delta_i32x4 = vmaxq_s32(
                            vsubq_s32(vld1q_s32(scores + position_idx), new_max_i32x4), delta_floor_i32x4);
                        uint32x4_t const weight_u32x4 = vandq_u32(
                            vreinterpretq_u32_s32(nk_exp2_u8_i32x4_neon_(vmulq_s32(delta_i32x4, scale_fixed_i32x4))),
                            tail_mask_u32x4);
                        panel_sum_u32x4 = vaddq_u32(panel_sum_u32x4, weight_u32x4);
                        uint8x8_t const weight_u8x8 = vmovn_u16(vcombine_u16(vmovn_u32(weight_u32x4), vdup_n_u16(0)));
                        vst1_lane_u32((nk_u32_t *)(weights + position_idx), vreinterpret_u32_u8(weight_u8x8), 0);
                        position_idx += 4;
                    }
                    nk_u32_t panel_sum_u32 = vaddvq_u32(panel_sum_u32x4);
                    for (; position_idx % 4; position_idx++) weights[position_idx] = 0; // zero any last-quad padding
                    panel_sum = (nk_f32_t)panel_sum_u32;
                    running_sum = running_sum * correction + panel_sum;

                    // P × V as UDOT over U8 quad tiles
                    nk_size_t const panel_quads = nk_size_divide_round_up_(panel_length, 4);
                    char const *panel_tiles = values_plane + (panel_start / 4) * depth_padded * 4;
                    for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                        vst1q_u32(output_totals + channel_idx, vdupq_n_u32(0));
                    for (nk_size_t quad_idx = 0; quad_idx < panel_quads; quad_idx++) {
                        nk_u32_t const weights_word = *(nk_u32_t const *)(weights + quad_idx * 4);
                        if (weights_word == 0) continue; // U8 softmax weights are sparse: whole quads vanish
                        uint8x16_t const weights_u8x16 = vreinterpretq_u8_u32(vdupq_n_u32(weights_word));
                        uint8_t const *tile = (uint8_t const *)(panel_tiles + quad_idx * depth_padded * 4);
                        for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 16) {
                            vst1q_u32(output_totals + channel_idx,
                                      vdotq_u32(vld1q_u32(output_totals + channel_idx), weights_u8x16,
                                                vld1q_u8(tile + channel_idx * 4)));
                            vst1q_u32(output_totals + channel_idx + 4,
                                      vdotq_u32(vld1q_u32(output_totals + channel_idx + 4), weights_u8x16,
                                                vld1q_u8(tile + channel_idx * 4 + 16)));
                            vst1q_u32(output_totals + channel_idx + 8,
                                      vdotq_u32(vld1q_u32(output_totals + channel_idx + 8), weights_u8x16,
                                                vld1q_u8(tile + channel_idx * 4 + 32)));
                            vst1q_u32(output_totals + channel_idx + 12,
                                      vdotq_u32(vld1q_u32(output_totals + channel_idx + 12), weights_u8x16,
                                                vld1q_u8(tile + channel_idx * 4 + 48)));
                        }
                    }
                    // 128 · Σw subtracts in integer; |Σ w · v| < 2^24 converts to F32 exactly.
                    uint32x4_t const bias_u32x4 = vdupq_n_u32(panel_sum_u32 << 7);
                    float32x4_t const correction_f32x4 = vdupq_n_f32(correction);
                    for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4) {
                        int32x4_t const total_i32x4 = vreinterpretq_s32_u32(
                            vsubq_u32(vld1q_u32(output_totals + channel_idx), bias_u32x4));
                        vst1q_f32(output_row + channel_idx,
                                  vfmaq_f32(vcvtq_f32_s32(total_i32x4), vld1q_f32(output_row + channel_idx),
                                            correction_f32x4));
                    }
                }

                nk_f32_t const inverse_sum = running_sum > 0 ? 1 / running_sum : 0;
                float32x4_t const inverse_sum_f32x4 = vdupq_n_f32(inverse_sum);
                nk_size_t const token = query_first + row_idx;
                nk_f32_t *destination = output + token * output_stride_floats + head_idx * depth;
                for (channel_idx = 0; channel_idx + 4 <= depth; channel_idx += 4)
                    vst1q_f32(destination + channel_idx,
                              vmulq_f32(vld1q_f32(output_row + channel_idx), inverse_sum_f32x4));
                for (; channel_idx < depth; channel_idx++)
                    destination[channel_idx] = output_row[channel_idx] * inverse_sum;
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(
                        (nk_f32_t)running_max * scale2, running_sum / 255.0f);
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

#endif // NUMKONG_TARGET_NEONSDOT
#endif // NUMKONG_ARCH_ARM64_

#endif // NUMKONG_ATTENTION_NEONSDOT_H
