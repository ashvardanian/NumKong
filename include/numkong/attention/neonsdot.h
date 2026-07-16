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
 *  never vanishes and the 255 cancels in normalization. The weighted V accumulation runs as
 *  `UDOT` over V tiles packed 4-positions × 4-channels with a +128 offset: since the weights
 *  are U8, `Σ w·(v+128) − 128·Σw = Σ w·v` exactly, the bias subtracts in integer before the
 *  single F32 conversion, and every panel total stays under 2^24 — bit-exact with serial at
 *  a 6× faster inner loop, still on the baseline `dotprod` extension.
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
NK_HELPER_INLINE float32x4_t nk_attention_exp2_f32x4_neonsdot_(float32x4_t x_f32x4) {
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

/**
 *  @brief I-BERT-style integer exponential for the I8 weight path: takes the base-2 argument as a
 *         Q15 fixed-point value in `[−10·2^15, 0]` and returns `round(2^t · 255)` as a U8 weight in
 *         the low byte of each I32 lane — the NEON mirror of the SME/x86 helper, bit-identical math.
 *         A degree-3 fixed-point polynomial covers the fraction; a lane-variable shift (`vshlq_s32`
 *         with a negated count = arithmetic right shift) applies the integer part. No floating point.
 */
NK_HELPER_INLINE int32x4_t nk_attention_iexp2_weight_i32x4_neonsdot_(int32x4_t t_q15_i32x4) {
    int32x4_t const whole_i32x4 = vshrq_n_s32(t_q15_i32x4, 15); // floor, in [-10, 0]
    int32x4_t const fraction_i32x4 = vandq_s32(t_q15_i32x4, vdupq_n_s32(0x7FFF));
    int32x4_t poly_i32x4 = vdupq_n_s32(1296); // Chebyshev-fit 2^r coefficients in Q14, degree 3
    poly_i32x4 = vaddq_s32(vshrq_n_s32(vmulq_s32(fraction_i32x4, poly_i32x4), 15), vdupq_n_s32(3678));
    poly_i32x4 = vaddq_s32(vshrq_n_s32(vmulq_s32(fraction_i32x4, poly_i32x4), 15), vdupq_n_s32(11410));
    poly_i32x4 = vaddq_s32(vshrq_n_s32(vmulq_s32(fraction_i32x4, poly_i32x4), 15), vdupq_n_s32(16382));
    int32x4_t const scaled_i32x4 = vsubq_s32(vshlq_n_s32(poly_i32x4, 8), poly_i32x4); // (poly<<8)−poly = poly·255
    int32x4_t const shift_i32x4 = vsubq_s32(vdupq_n_s32(14), whole_i32x4);            // in [14, 24]
    int32x4_t const bias_i32x4 = vshlq_s32(vdupq_n_s32(1), vsubq_s32(vdupq_n_s32(13), whole_i32x4)); // 1 << (13−whole)
    return vshlq_s32(vaddq_s32(scaled_i32x4, bias_i32x4), vnegq_s32(shift_i32x4)); // round-half-up, then ≫ shift
}

NK_API_COMPTIME nk_size_t nk_attention_packed_size_i8_neonsdot(nk_size_t key_value_head_count, nk_size_t depth,
                                                               nk_u32_t const *segment_lengths,
                                                               nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_neonsdot_k_)
        return nk_attention_packed_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count);
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 16);
    nk_size_t payload_bytes = 0; // K rows plus quad-tiled V, positions padded to the UDOT quad
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count *
                         nk_size_round_up_to_multiple_((nk_size_t)segment_lengths[segment_idx], 4) * depth_padded;
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_API_COMPTIME void nk_attention_pack_i8_neonsdot(                                    //
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
                                 first_task, 4, depth_padded);
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
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 4);
        nk_size_t const plane_bytes = position_count_padded * depth_padded;
        char *keys_plane = payload_base + payload_offsets[segment_idx] + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride_bytes +
                                   key_value_head_idx * depth;
            char *keys_destination = keys_plane + position_idx * depth_padded;
            nk_size_t byte_idx = 0;
            for (; byte_idx + 16 <= depth; byte_idx += 16)
                vst1q_u8((uint8_t *)(keys_destination + byte_idx), vld1q_u8((uint8_t const *)(keys_row + byte_idx)));
            for (; byte_idx < depth; byte_idx++) keys_destination[byte_idx] = keys_row[byte_idx];
            for (; byte_idx < depth_padded; byte_idx++) keys_destination[byte_idx] = 0;
        }
        for (nk_size_t position_idx = position_count; position_idx < position_count_padded; position_idx++)
            for (nk_size_t byte_idx = 0; byte_idx < depth_padded; byte_idx += 16)
                vst1q_u8((uint8_t *)(keys_plane + position_idx * depth_padded + byte_idx), vdupq_n_u8(0));
        // V tiles as [position_quad][channel][4 positions] with a +128 offset to U8 — one XOR
        // of the sign bit — so one UDOT covers 4 positions × 4 channels; `vst4q_u8` interleaves
        // four source rows per 16 channels with no staging. Full quads run branch-free; the
        // ragged last quad and any channel tail fall to scalar code, since padded positions
        // carry zero weights and only need deterministic content.
        for (nk_size_t quad_idx = 0; quad_idx < position_count / 4; quad_idx++) {
            nk_u8_t const *row0 = (nk_u8_t const *)values + (position_first + quad_idx * 4 + 0) * value_stride_bytes +
                                  key_value_head_idx * depth;
            nk_u8_t const *row1 = (nk_u8_t const *)values + (position_first + quad_idx * 4 + 1) * value_stride_bytes +
                                  key_value_head_idx * depth;
            nk_u8_t const *row2 = (nk_u8_t const *)values + (position_first + quad_idx * 4 + 2) * value_stride_bytes +
                                  key_value_head_idx * depth;
            nk_u8_t const *row3 = (nk_u8_t const *)values + (position_first + quad_idx * 4 + 3) * value_stride_bytes +
                                  key_value_head_idx * depth;
            nk_u8_t *tile = (nk_u8_t *)(values_plane + quad_idx * depth_padded * 4);
            nk_size_t channel_idx = 0;
            for (; channel_idx + 16 <= depth; channel_idx += 16) {
                uint8x16x4_t rows_u8x16x4;
                rows_u8x16x4.val[0] = veorq_u8(vld1q_u8(row0 + channel_idx), vdupq_n_u8(0x80));
                rows_u8x16x4.val[1] = veorq_u8(vld1q_u8(row1 + channel_idx), vdupq_n_u8(0x80));
                rows_u8x16x4.val[2] = veorq_u8(vld1q_u8(row2 + channel_idx), vdupq_n_u8(0x80));
                rows_u8x16x4.val[3] = veorq_u8(vld1q_u8(row3 + channel_idx), vdupq_n_u8(0x80));
                vst4q_u8(tile + channel_idx * 4, rows_u8x16x4);
            }
            for (; channel_idx < depth_padded; channel_idx++) {
                tile[channel_idx * 4 + 0] = channel_idx < depth ? (nk_u8_t)(row0[channel_idx] ^ 0x80) : 0;
                tile[channel_idx * 4 + 1] = channel_idx < depth ? (nk_u8_t)(row1[channel_idx] ^ 0x80) : 0;
                tile[channel_idx * 4 + 2] = channel_idx < depth ? (nk_u8_t)(row2[channel_idx] ^ 0x80) : 0;
                tile[channel_idx * 4 + 3] = channel_idx < depth ? (nk_u8_t)(row3[channel_idx] ^ 0x80) : 0;
            }
        }
        for (nk_size_t position_idx = position_count / 4 * 4; position_idx < position_count_padded; position_idx++) {
            nk_u8_t const *row = (nk_u8_t const *)values + (position_first + position_idx) * value_stride_bytes +
                                 key_value_head_idx * depth;
            nk_u8_t *tile = (nk_u8_t *)(values_plane + (position_idx / 4) * depth_padded * 4) + position_idx % 4;
            for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx++)
                tile[channel_idx * 4] = position_idx < position_count && channel_idx < depth
                                            ? (nk_u8_t)(row[channel_idx] ^ 0x80)
                                            : 0;
        }
    }
}

NK_API_COMPTIME void nk_attention_packed_i8_neonsdot(                            //
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
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_;                     // softmax(x) = softmax₂(x·log₂e)
    nk_i32_t const scale_fixed = (nk_i32_t)(scale2 * 32768.0f + 0.5f); // Q15 scale for the integer exponential
    nk_i32_t const delta_floor = // the score delta below which every weight quantizes to zero (2^t·255 + 0.5 < 1)
        scale_fixed > 0 ? -(nk_i32_t)((10u << 15) / (nk_u32_t)scale_fixed) - 1 : 0;
    nk_size_t const panel_width = nk_attention_panel_neonsdot_k_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_i8_t query_row[nk_attention_max_depth_neonsdot_k_];
    NK_ALIGN64 nk_f32_t output_row[nk_attention_max_depth_neonsdot_k_];
    NK_ALIGN64 nk_u32_t output_totals[nk_attention_max_depth_neonsdot_k_];
    NK_ALIGN64 nk_i32_t scores[nk_attention_panel_neonsdot_k_]; // raw I32 QK dots; softmax stays integer
    NK_ALIGN64 nk_u8_t weights[nk_attention_panel_neonsdot_k_];

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const plane_bytes = nk_size_round_up_to_multiple_(position_count, 4) * depth_padded;
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
            nk_i32_t running_max = NK_I32_MIN;
            nk_f32_t running_sum = 0;

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_length = (panel_start + panel_width <= position_count)
                                                   ? panel_width
                                                   : (position_count - panel_start);

                nk_size_t position_idx = 0;
                int32x4_t max_i32x4 = vdupq_n_s32(NK_I32_MIN); // exact integer row max over raw I32 scores
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
                nk_f32_t const correction = vgetq_lane_f32(nk_attention_exp2_f32x4_neonsdot_(vdupq_n_f32(
                                                               ((nk_f32_t)running_max - (nk_f32_t)new_max) * scale2)),
                                                           0);
                running_max = new_max;

                int32x4_t const new_max_i32x4 = vdupq_n_s32(new_max);
                int32x4_t const scale_fixed_i32x4 = vdupq_n_s32(scale_fixed);
                int32x4_t const delta_floor_i32x4 = vdupq_n_s32(delta_floor);
                nk_f32_t panel_sum = 0;
                uint32x4_t panel_sum_u32x4 = vdupq_n_u32(0); // weights are U8 over <= 512 positions
                for (position_idx = 0; position_idx + 4 <= panel_length; position_idx += 4) {
                    int32x4_t const delta_i32x4 = vmaxq_s32(vsubq_s32(vld1q_s32(scores + position_idx), new_max_i32x4),
                                                            delta_floor_i32x4);
                    uint32x4_t const weight_u32x4 = vreinterpretq_u32_s32(
                        nk_attention_iexp2_weight_i32x4_neonsdot_(vmulq_s32(delta_i32x4, scale_fixed_i32x4)));
                    panel_sum_u32x4 = vaddq_u32(panel_sum_u32x4, weight_u32x4);
                    uint8x8_t const weight_u8x8 = vmovn_u16(vcombine_u16(vmovn_u32(weight_u32x4), vdup_n_u16(0)));
                    vst1_lane_u32((nk_u32_t *)(weights + position_idx), vreinterpret_u32_u8(weight_u8x8), 0);
                }
                if (position_idx < panel_length) { // masked vector tail — no scalar exp2, padded lanes forced to 0
                    uint32x4_t const lane_index_u32x4 = {0, 1, 2, 3};
                    uint32x4_t const tail_mask_u32x4 = vcltq_u32(lane_index_u32x4,
                                                                 vdupq_n_u32((nk_u32_t)(panel_length - position_idx)));
                    int32x4_t const delta_i32x4 = vmaxq_s32(vsubq_s32(vld1q_s32(scores + position_idx), new_max_i32x4),
                                                            delta_floor_i32x4);
                    uint32x4_t const weight_u32x4 = vandq_u32(
                        vreinterpretq_u32_s32(
                            nk_attention_iexp2_weight_i32x4_neonsdot_(vmulq_s32(delta_i32x4, scale_fixed_i32x4))),
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

                nk_size_t const panel_quads = (panel_length + 3) / 4; // P×V as UDOT over U8 quad tiles
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
                uint32x4_t const bias_u32x4 = vdupq_n_u32(panel_sum_u32 << 7); // 128·Σw subtracts in integer,
                float32x4_t const correction_f32x4 = vdupq_n_f32(correction);  // so |Σ w·v| <= 512·255·128 < 2^24
                                                                               // converts to F32 exactly
                for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4) {
                    int32x4_t const total_i32x4 = vreinterpretq_s32_u32(
                        vsubq_u32(vld1q_u32(output_totals + channel_idx), bias_u32x4));
                    vst1q_f32(
                        output_row + channel_idx,
                        vfmaq_f32(vcvtq_f32_s32(total_i32x4), vld1q_f32(output_row + channel_idx), correction_f32x4));
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
