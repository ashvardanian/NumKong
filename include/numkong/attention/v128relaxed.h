/**
 *  @file include/numkong/attention/v128relaxed.h
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *  @brief Ragged attention for WASM with Relaxed SIMD.
 *
 *  @sa include/numkong/attention.h
 *
 *  Portable 128-bit backend for WebAssembly engines with the Relaxed SIMD proposal. Storage follows
 *  the `attention/haswell.h` conventions exactly: BF16, E4M3, and I8 stay in their source encoding
 *  at rest — packing is a raw strided-row copy with channels zero-padded to a multiple of 8, done
 *  by `attention/v128.h` — and every value widens to F32 on the fly inside the compute loops
 *  through the `cast/v128.h` and `cast/v128relaxed.h` helpers. Per query row, KV is swept in
 *  512-position panels with an exact online running-max correction; the F32 score row, 2 KB, stays
 *  L1-resident. `depth > 256` routes to the width-agnostic serial kernel.
 *
 *  The base-2 exponent is the family's shared degree-7 polynomial, evaluated 4-wide with
 *  @c wasm_f32x4_relaxed_madd after a @c wasm_f32x4_nearest range reduction and the same
 *  denormal-avoiding [−125, 127] clamps as @c nk_f32_exp2_serial_; scalar panel tails call the
 *  serial helper directly to keep the family polynomial end-to-end.
 *
 *  The I8 path keeps QK scores exact in I32: @c wasm_i32x4_relaxed_dot_i8x16_i7x16_add requires a
 *  7-bit second operand, so K is bit-split as k = k₇ − 128 · [k < 0] — the dot runs on the low 7
 *  bits while an I16 pairwise correction — Σq over K-negative lanes, × 128 — is folded into the I32
 *  sum vector before the single horizontal reduce per position. Attention weights quantize to U8
 *  exactly like serial, trunc(2^(s₂−m₂) · 255 + 0.5), vectorized 4-wide with the same separate
 *  multiply and add so the rounding matches the scalar reference bit-for-bit. In the PV
 *  accumulation @c wasm_f32x4_relaxed_madd is safe even under that bit-exactness contract: a U8
 *  weight (≤ 255) times an I8 plane value (|v| ≤ 128) is at most 32640 < 2^24, so every product is
 *  exactly representable in F32, and fusing multiply into add cannot change one accumulated bit.
 */
#ifndef NUMKONG_ATTENTION_V128RELAXED_H
#define NUMKONG_ATTENTION_V128RELAXED_H

#if NUMKONG_TARGET_V128RELAXED

#include <wasm_simd128.h>

#include "numkong/attention/v128.h"   // `nk_attention_load_bf16x4_v128_`
#include "numkong/attention/serial.h" // shared packed-KV header/directory, width-agnostic fallback
#include "numkong/cast/v128relaxed.h" // `nk_e4m3x4_to_f32x4_v128relaxed_`
#include "numkong/each/v128.h"        // `nk_exp2_u8_i32x4_v128_`
#include "numkong/each/v128relaxed.h" // `nk_exp2_f32x4_v128relaxed_`
#include "numkong/reduce/v128.h"      // `nk_reduce_add_f32x4_v128_`, `nk_reduce_max_f32x4_v128_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_v128relaxed(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes) {
    *bytes =
        depth > nk_attention_max_depth_v128_k_
            ? nk_attention_pack_bound_(key_value_head_count, token_count, segment_count, 1, depth * sizeof(nk_f32_t))
            : nk_attention_pack_size_v128_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_bf16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_v128relaxed(void const *key_value_packed, nk_size_t *heads,
                                                                   nk_size_t *depth, nk_size_t *segments,
                                                                   void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_v128relaxed_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_v128relaxed(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_size_t token_count, nk_size_t segment_count,
                                                                nk_size_t *bytes) {
    *bytes =
        depth > nk_attention_max_depth_v128_k_
            ? nk_attention_pack_bound_(key_value_head_count, token_count, segment_count, 1, depth * sizeof(nk_f32_t))
            : nk_attention_pack_size_v128_(key_value_head_count, depth, token_count, segment_count, sizeof(nk_e4m3_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_v128relaxed(void const *key_value_packed, nk_size_t *heads,
                                                                   nk_size_t *depth, nk_size_t *segments,
                                                                   void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_v128relaxed_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_i8_v128relaxed(nk_size_t key_value_head_count, nk_size_t depth,
                                                              nk_size_t token_count, nk_size_t segment_count,
                                                              nk_size_t *bytes) {
    *bytes = depth > nk_attention_max_depth_v128_k_
                 ? nk_attention_pack_bound_(key_value_head_count, token_count, segment_count, 1, depth)
                 : nk_attention_pack_size_v128_(key_value_head_count, depth, token_count, segment_count, 1);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_v128relaxed(void const *key_value_packed, nk_size_t *heads,
                                                                 nk_size_t *depth, nk_size_t *segments, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_v128relaxed_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_bf16_v128relaxed(                //
    nk_bf16_t const *keys, nk_bf16_t const *values,                        //
    nk_size_t key_value_head_count, nk_size_t depth,                       //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,      //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, //
    void *key_value_packed, nk_size_t task_begin, nk_size_t task_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth > nk_attention_max_depth_v128_k_) {
        nk_attention_pack_serial_(keys, values, sizeof(nk_bf16_t), &nk_attention_load_bf16_serial_,
                                  key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                                  key_stride, value_stride, key_value_packed, task_begin, task_end,
                                  nk_cap_v128relaxed_k);
        return nk_success_k;
    }
    nk_attention_pack_v128_(keys, values, sizeof(nk_bf16_t), key_value_head_count, depth, segment_offsets,
                            segment_lengths, segment_count, key_stride, value_stride, key_value_packed, task_begin,
                            task_end, nk_cap_v128relaxed_k);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_v128relaxed(                //
    nk_e4m3_t const *keys, nk_e4m3_t const *values,                        //
    nk_size_t key_value_head_count, nk_size_t depth,                       //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,      //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, //
    void *key_value_packed, nk_size_t task_begin, nk_size_t task_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth > nk_attention_max_depth_v128_k_) {
        nk_attention_pack_serial_(keys, values, sizeof(nk_e4m3_t), &nk_attention_load_e4m3_serial_,
                                  key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                                  key_stride, value_stride, key_value_packed, task_begin, task_end,
                                  nk_cap_v128relaxed_k);
        return nk_success_k;
    }
    nk_attention_pack_v128_(keys, values, sizeof(nk_e4m3_t), key_value_head_count, depth, segment_offsets,
                            segment_lengths, segment_count, key_stride, value_stride, key_value_packed, task_begin,
                            task_end, nk_cap_v128relaxed_k);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_i8_v128relaxed(                  //
    nk_i8_t const *keys, nk_i8_t const *values,                            //
    nk_size_t key_value_head_count, nk_size_t depth,                       //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,      //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, //
    void *key_value_packed, nk_size_t task_begin, nk_size_t task_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth > nk_attention_max_depth_v128_k_) {
        nk_attention_pack_i8_serial_(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                     segment_count, key_stride, value_stride, key_value_packed, task_begin, task_end,
                                     nk_cap_v128relaxed_k);
        return nk_success_k;
    }
    nk_attention_pack_v128_(keys, values, 1, key_value_head_count, depth, segment_offsets, segment_lengths,
                            segment_count, key_stride, value_stride, key_value_packed, task_begin, task_end,
                            nk_cap_v128relaxed_k);
    return nk_success_k;
}

/** Widens 4 raw plane scalars (BF16 or E4M3 at rest) to F32 inside the hot loops. */
typedef v128_t (*nk_attention_load_v128relaxed_t_)(void const *plane_chunk);

NUMKONG_INLINE v128_t nk_attention_load_e4m3x4_v128relaxed_(void const *plane_chunk) {
    nk_b32_vec_t raw_vec;
    raw_vec.u32 = *(nk_u32_t const *)plane_chunk;
    return nk_e4m3x4_to_f32x4_v128relaxed_(raw_vec).v128;
}

/** Shared attention core over raw-encoded planes: per query row, panel-flash with an exact online
 *  correction; queries widen once per row, planes widen in-loop. */
NUMKONG_INLINE void nk_attention_packed_float_v128relaxed_(                //
    void const *queries, nk_size_t element_bytes,                          //
    nk_attention_load_f32_serial_t_ load_f32,                              //
    nk_attention_load_v128relaxed_t_ load,                                 //
    void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *query_offsets,                                         //
    nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale,       //
    nk_diagonal_band_t band, nk_size_t task_begin, nk_size_t task_end) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->heads == key_value_head_count && key_value_head_count != 0 &&
               head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const plane_row_bytes = depth_padded * element_bytes;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)
    nk_size_t const panel_width = nk_attention_panel_v128_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (task_begin < grid_begin) task_begin = grid_begin;
    if (task_end > grid_end) task_end = grid_end;

    nk_align_(64) nk_f32_t query_row[nk_attention_max_depth_v128_k_];
    nk_align_(64) nk_f32_t output_row[nk_attention_max_depth_v128_k_];
    nk_align_(64) nk_f32_t scores[nk_attention_panel_v128_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && task_begin < task_end; head_idx++) {
        nk_size_t const token_first = (task_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (task_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = segment_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_(query_end - query_first, position_count);
            nk_size_t const plane_bytes = position_count * plane_row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                     (head_idx / head_group_size) * plane_bytes;
            char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                char const *query_source = (char const *)queries + (query_first + row_idx) * query_stride +
                                           head_idx * depth * element_bytes;
                nk_size_t channel_idx = 0;
                for (; channel_idx < depth; channel_idx++)
                    query_row[channel_idx] = load_f32(query_source + channel_idx * element_bytes);
                for (; channel_idx < depth_padded; channel_idx++) query_row[channel_idx] = 0.0f;
                for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                    wasm_v128_store(output_row + channel_idx, wasm_f32x4_splat(0.0f));
                nk_f32_t running_max2 = NUMKONG_F32_MIN, running_sum = 0;

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    nk_size_t position_idx;
                    for (position_idx = 0; position_idx < panel_length; position_idx++) {
                        char const *keys_row = keys_plane + (panel_start + position_idx) * plane_row_bytes;
                        v128_t sum0_f32x4 = wasm_f32x4_splat(0.0f), sum1_f32x4 = wasm_f32x4_splat(0.0f);
                        for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
                            nk_size_t const chunk_bytes = channel_idx * element_bytes;
                            sum0_f32x4 = wasm_f32x4_relaxed_madd(wasm_v128_load(query_row + channel_idx),
                                                                 load(keys_row + chunk_bytes), sum0_f32x4);
                            sum1_f32x4 = wasm_f32x4_relaxed_madd(wasm_v128_load(query_row + channel_idx + 4),
                                                                 load(keys_row + chunk_bytes + 4 * element_bytes),
                                                                 sum1_f32x4);
                        }
                        scores[position_idx] = nk_reduce_add_f32x4_v128_(wasm_f32x4_add(sum0_f32x4, sum1_f32x4));
                    }

                    v128_t const scale2_f32x4 = wasm_f32x4_splat(scale2);
                    v128_t max_f32x4 = wasm_f32x4_splat(NUMKONG_F32_MIN);
                    for (position_idx = 0; position_idx + 4 <= panel_length; position_idx += 4)
                        max_f32x4 = wasm_f32x4_max(max_f32x4,
                                                   wasm_f32x4_mul(wasm_v128_load(scores + position_idx), scale2_f32x4));
                    nk_f32_t panel_max2 = nk_reduce_max_f32x4_v128_(max_f32x4);
                    for (; position_idx < panel_length; position_idx++) {
                        nk_f32_t const scaled2 = scores[position_idx] * scale2;
                        if (scaled2 > panel_max2) panel_max2 = scaled2;
                    }
                    nk_f32_t const new_max2 = running_max2 > panel_max2 ? running_max2 : panel_max2;
                    nk_f32_t const correction = wasm_f32x4_extract_lane(
                        nk_exp2_f32x4_v128relaxed_(wasm_f32x4_splat(running_max2 - new_max2)), 0);
                    running_max2 = new_max2;

                    v128_t const new_max2_f32x4 = wasm_f32x4_splat(new_max2);
                    v128_t panel_sum_f32x4 = wasm_f32x4_splat(0.0f);
                    nk_f32_t panel_sum = 0;
                    for (position_idx = 0; position_idx + 4 <= panel_length; position_idx += 4) {
                        v128_t weight_f32x4 = nk_exp2_f32x4_v128relaxed_(wasm_f32x4_sub(
                            wasm_f32x4_mul(wasm_v128_load(scores + position_idx), scale2_f32x4), new_max2_f32x4));
                        panel_sum_f32x4 = wasm_f32x4_add(panel_sum_f32x4, weight_f32x4);
                        wasm_v128_store(scores + position_idx, weight_f32x4);
                    }
                    panel_sum = nk_reduce_add_f32x4_v128_(panel_sum_f32x4);
                    // The scalar tail keeps the family exp2 end-to-end.
                    for (; position_idx < panel_length; position_idx++) {
                        nk_f32_t const weight = nk_f32_exp2_serial_(scores[position_idx] * scale2 - new_max2);
                        scores[position_idx] = weight;
                        panel_sum += weight;
                    }
                    running_sum = running_sum * correction + panel_sum;

                    v128_t const correction_f32x4 = wasm_f32x4_splat(correction);
                    for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                        wasm_v128_store(output_row + channel_idx,
                                        wasm_f32x4_mul(wasm_v128_load(output_row + channel_idx), correction_f32x4));
                    for (position_idx = 0; position_idx < panel_length; position_idx++) {
                        v128_t const weight_f32x4 = wasm_f32x4_splat(scores[position_idx]);
                        char const *values_row = values_plane + (panel_start + position_idx) * plane_row_bytes;
                        for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
                            nk_size_t const chunk_bytes = channel_idx * element_bytes;
                            wasm_v128_store(output_row + channel_idx,
                                            wasm_f32x4_relaxed_madd(weight_f32x4, load(values_row + chunk_bytes),
                                                                    wasm_v128_load(output_row + channel_idx)));
                            wasm_v128_store(output_row + channel_idx + 4,
                                            wasm_f32x4_relaxed_madd(weight_f32x4,
                                                                    load(values_row + chunk_bytes + 4 * element_bytes),
                                                                    wasm_v128_load(output_row + channel_idx + 4)));
                        }
                    }
                }

                nk_f32_t const inverse_sum = running_sum > 0 ? 1 / running_sum : 0.0f;
                v128_t const inverse_sum_f32x4 = wasm_f32x4_splat(inverse_sum);
                nk_size_t const token = query_first + row_idx;
                nk_f32_t *destination = output + token * output_stride_floats + head_idx * depth;
                for (channel_idx = 0; channel_idx + 4 <= depth; channel_idx += 4)
                    wasm_v128_store(destination + channel_idx,
                                    wasm_f32x4_mul(wasm_v128_load(output_row + channel_idx), inverse_sum_f32x4));
                for (; channel_idx < depth; channel_idx++)
                    destination[channel_idx] = output_row[channel_idx] * inverse_sum;
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_(running_max2, running_sum);
            }
        }
    }
}

NUMKONG_API nk_status_t nk_attention_packed_bf16_v128relaxed(                    //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t task_begin, nk_size_t task_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_v128relaxed_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_v128_k_)
        nk_attention_serial_(queries, sizeof(nk_bf16_t), &nk_attention_load_bf16_serial_, key_value_packed, output,
                             log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_stride,
                             output_stride, scale, band, task_begin, task_end);
    else
        nk_attention_packed_float_v128relaxed_(queries, sizeof(nk_bf16_t), &nk_attention_load_bf16_serial_,
                                               &nk_attention_load_bf16x4_v128_, key_value_packed, output, log_sum_exp,
                                               head_count, key_value_head_count, depth, query_offsets, query_stride,
                                               output_stride, scale, band, task_begin, task_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_e4m3_v128relaxed(                    //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t task_begin, nk_size_t task_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_v128relaxed_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_v128_k_)
        nk_attention_serial_(queries, sizeof(nk_e4m3_t), &nk_attention_load_e4m3_serial_, key_value_packed, output,
                             log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_stride,
                             output_stride, scale, band, task_begin, task_end);
    else
        nk_attention_packed_float_v128relaxed_(queries, sizeof(nk_e4m3_t), &nk_attention_load_e4m3_serial_,
                                               &nk_attention_load_e4m3x4_v128relaxed_, key_value_packed, output,
                                               log_sum_exp, head_count, key_value_head_count, depth, query_offsets,
                                               query_stride, output_stride, scale, band, task_begin, task_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_i8_v128relaxed(                      //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,      //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t task_begin, nk_size_t task_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_v128relaxed_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_v128_k_) {
        nk_attention_packed_i8_serial_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                       depth, query_offsets, query_stride, output_stride, scale, band, task_begin,
                                       task_end);
        return nk_success_k;
    }
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->heads == key_value_head_count && key_value_head_count != 0 &&
               head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const depth_full16 = depth_padded & ~(nk_size_t)15;
    nk_size_t const depth_padded16 = nk_size_round_up_to_multiple_(depth_padded, 16);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)
    nk_size_t const panel_width = nk_attention_panel_v128_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (task_begin < grid_begin) task_begin = grid_begin;
    if (task_end > grid_end) task_end = grid_end;

    nk_align_(64) nk_i8_t query_i8[nk_attention_max_depth_v128_k_];
    nk_align_(64) nk_f32_t output_row[nk_attention_max_depth_v128_k_];
    nk_align_(64) nk_i32_t scores[nk_attention_panel_v128_k_];
    nk_align_(64) nk_u8_t weights[nk_attention_panel_v128_k_];
    nk_align_(64) nk_i32_t panel_acc[nk_attention_max_depth_v128_k_]; // per-panel integer P × V accumulator

    for (nk_size_t head_idx = 0; head_idx < head_count && task_begin < task_end; head_idx++) {
        nk_size_t const token_first = (task_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (task_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = segment_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_(query_end - query_first, position_count);
            nk_size_t const plane_bytes = position_count * depth_padded;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                     (head_idx / head_group_size) * plane_bytes;
            char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);
                nk_i8_t const *query_source = (nk_i8_t const *)((char const *)queries +
                                                                (query_first + row_idx) * query_stride) +
                                              head_idx * depth;
                nk_size_t channel_idx = 0;
                for (; channel_idx + 16 <= depth; channel_idx += 16)
                    wasm_v128_store(query_i8 + channel_idx, wasm_v128_load(query_source + channel_idx));
                for (; channel_idx < depth; channel_idx++) query_i8[channel_idx] = query_source[channel_idx];
                for (; channel_idx < depth_padded16; channel_idx++) query_i8[channel_idx] = 0;
                for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                    wasm_v128_store(output_row + channel_idx, wasm_f32x4_splat(0.0f));
                nk_i32_t running_max = NUMKONG_I32_MIN;
                nk_f32_t running_sum = 0;
                nk_i32_t const scale_fixed = (nk_i32_t)(scale2 * 32768.0f + 0.5f); // Q15 scale for the integer exp
                nk_i32_t const delta_floor = // score delta below which every weight quantizes to zero
                    scale_fixed > 0 ? -(nk_i32_t)((10u << 15) / (nk_u32_t)scale_fixed) - 1 : 0;

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    nk_size_t position_idx;
                    for (position_idx = 0; position_idx < panel_length; position_idx++) {
                        char const *keys_row = keys_plane + (panel_start + position_idx) * depth_padded;
                        v128_t sum_i32x4 = wasm_i32x4_splat(0);
                        v128_t correction_i16x8 = wasm_i16x8_splat(0);
                        for (channel_idx = 0; channel_idx < depth_full16; channel_idx += 16) {
                            v128_t query_i8x16 = wasm_v128_load(query_i8 + channel_idx);
                            v128_t keys_i8x16 = wasm_v128_load(keys_row + channel_idx);
                            v128_t keys_neg_mask_i8x16 = wasm_i8x16_lt(keys_i8x16, wasm_i8x16_splat(0));
                            v128_t keys_7bit_i8x16 = wasm_v128_and(keys_i8x16, wasm_i8x16_splat(0x7F));
                            sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(query_i8x16, keys_7bit_i8x16, sum_i32x4);
                            correction_i16x8 = wasm_i16x8_add(
                                correction_i16x8,
                                wasm_i16x8_extadd_pairwise_i8x16(wasm_v128_and(query_i8x16, keys_neg_mask_i8x16)));
                        }
                        if (channel_idx < depth_padded) { // trailing 8-channel chunk; upper I8 lanes zero on both sides
                            v128_t query_i8x16 = wasm_v128_load64_zero(query_i8 + channel_idx);
                            v128_t keys_i8x16 = wasm_v128_load64_zero(keys_row + channel_idx);
                            v128_t keys_neg_mask_i8x16 = wasm_i8x16_lt(keys_i8x16, wasm_i8x16_splat(0));
                            v128_t keys_7bit_i8x16 = wasm_v128_and(keys_i8x16, wasm_i8x16_splat(0x7F));
                            sum_i32x4 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(query_i8x16, keys_7bit_i8x16, sum_i32x4);
                            correction_i16x8 = wasm_i16x8_add(
                                correction_i16x8,
                                wasm_i16x8_extadd_pairwise_i8x16(wasm_v128_and(query_i8x16, keys_neg_mask_i8x16)));
                        }
                        v128_t correction_i32x4 = wasm_i32x4_extadd_pairwise_i16x8(correction_i16x8);
                        v128_t score_i32x4 = wasm_i32x4_sub(sum_i32x4,
                                                            wasm_i32x4_mul(correction_i32x4, wasm_i32x4_splat(128)));
                        scores[position_idx] = nk_reduce_add_i32x4_v128_(score_i32x4);
                    }

                    v128_t max_i32x4 = wasm_i32x4_splat(NUMKONG_I32_MIN);
                    // exact integer panel max over the raw I32 scores
                    for (position_idx = 0; position_idx + 4 <= panel_length; position_idx += 4)
                        max_i32x4 = wasm_i32x4_max(max_i32x4, wasm_v128_load(scores + position_idx));
                    v128_t hmax_i32x4 = wasm_i32x4_max(max_i32x4, wasm_i32x4_shuffle(max_i32x4, max_i32x4, 2, 3, 0, 1));
                    hmax_i32x4 = wasm_i32x4_max(hmax_i32x4, wasm_i32x4_shuffle(hmax_i32x4, hmax_i32x4, 1, 0, 3, 2));
                    nk_i32_t panel_max = wasm_i32x4_extract_lane(hmax_i32x4, 0);
                    for (; position_idx < panel_length; position_idx++)
                        if (scores[position_idx] > panel_max) panel_max = scores[position_idx];
                    nk_i32_t const new_max = running_max > panel_max ? running_max : panel_max;
                    nk_f32_t const correction = wasm_f32x4_extract_lane(
                        nk_exp2_f32x4_v128relaxed_(
                            wasm_f32x4_splat(((nk_f32_t)running_max - (nk_f32_t)new_max) * scale2)),
                        0);
                    running_max = new_max;

                    v128_t const new_max_i32x4 = wasm_i32x4_splat(new_max);
                    v128_t const scale_fixed_i32x4 = wasm_i32x4_splat(scale_fixed);
                    v128_t const delta_floor_i32x4 = wasm_i32x4_splat(delta_floor);
                    v128_t panel_sum_i32x4 = wasm_i32x4_splat(0);
                    for (position_idx = 0; position_idx + 4 <= panel_length; position_idx += 4) {
                        v128_t delta_i32x4 = wasm_i32x4_max(
                            wasm_i32x4_sub(wasm_v128_load(scores + position_idx), new_max_i32x4), delta_floor_i32x4);
                        v128_t weight_i32x4 = nk_exp2_u8_i32x4_v128_(wasm_i32x4_mul(delta_i32x4, scale_fixed_i32x4));
                        panel_sum_i32x4 = wasm_i32x4_add(panel_sum_i32x4, weight_i32x4);
                        v128_t weight_i16x8 = wasm_i16x8_narrow_i32x4(weight_i32x4, weight_i32x4);
                        wasm_v128_store32_lane(weights + position_idx,
                                               wasm_u8x16_narrow_i16x8(weight_i16x8, weight_i16x8), 0);
                    }
                    if (position_idx < panel_length) { // masked vector tail — inactive lanes forced to a zero weight
                        v128_t const lane_index_i32x4 = wasm_i32x4_make(0, 1, 2, 3);
                        v128_t const tail_mask_i32x4 = wasm_i32x4_lt(
                            lane_index_i32x4, wasm_i32x4_splat((nk_i32_t)(panel_length - position_idx)));
                        v128_t delta_i32x4 = wasm_i32x4_max(
                            wasm_i32x4_sub(wasm_v128_load(scores + position_idx), new_max_i32x4), delta_floor_i32x4);
                        v128_t weight_i32x4 = wasm_v128_and(
                            nk_exp2_u8_i32x4_v128_(wasm_i32x4_mul(delta_i32x4, scale_fixed_i32x4)), tail_mask_i32x4);
                        panel_sum_i32x4 = wasm_i32x4_add(panel_sum_i32x4, weight_i32x4);
                        v128_t weight_i16x8 = wasm_i16x8_narrow_i32x4(weight_i32x4, weight_i32x4);
                        wasm_v128_store32_lane(weights + position_idx,
                                               wasm_u8x16_narrow_i16x8(weight_i16x8, weight_i16x8), 0);
                    }
                    running_sum = running_sum * correction + (nk_f32_t)nk_reduce_add_i32x4_v128_(panel_sum_i32x4);

                    // Integer P × V: zero the panel accumulator, sum U8 · I8 products per non-zero
                    // position, then drain.
                    for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                        wasm_v128_store(panel_acc + channel_idx, wasm_i32x4_splat(0));
                    for (position_idx = 0; position_idx < panel_length; position_idx++) {
                        nk_u8_t const weight_u8 = weights[position_idx];
                        // Sparse U8 weights: skipping dead positions is the kernel's lever.
                        if (weight_u8 == 0) continue;
                        v128_t const weight_i32x4 = wasm_i32x4_splat((nk_i32_t)weight_u8);
                        char const *values_row = values_plane + (panel_start + position_idx) * depth_padded;
                        channel_idx = 0;
                        // One 16-byte V load, widened to four i32x4.
                        for (; channel_idx < depth_full16; channel_idx += 16) {
                            v128_t values_i8x16 = wasm_v128_load(values_row + channel_idx);
                            v128_t low_i16x8 = wasm_i16x8_extend_low_i8x16(values_i8x16);
                            v128_t high_i16x8 = wasm_i16x8_extend_high_i8x16(values_i8x16);
                            v128_t v0_i32x4 = wasm_i32x4_extend_low_i16x8(low_i16x8);
                            v128_t v1_i32x4 = wasm_i32x4_extend_high_i16x8(low_i16x8);
                            v128_t v2_i32x4 = wasm_i32x4_extend_low_i16x8(high_i16x8);
                            v128_t v3_i32x4 = wasm_i32x4_extend_high_i16x8(high_i16x8);
                            wasm_v128_store(panel_acc + channel_idx + 0,
                                            wasm_i32x4_add(wasm_v128_load(panel_acc + channel_idx + 0),
                                                           wasm_i32x4_mul(weight_i32x4, v0_i32x4)));
                            wasm_v128_store(panel_acc + channel_idx + 4,
                                            wasm_i32x4_add(wasm_v128_load(panel_acc + channel_idx + 4),
                                                           wasm_i32x4_mul(weight_i32x4, v1_i32x4)));
                            wasm_v128_store(panel_acc + channel_idx + 8,
                                            wasm_i32x4_add(wasm_v128_load(panel_acc + channel_idx + 8),
                                                           wasm_i32x4_mul(weight_i32x4, v2_i32x4)));
                            wasm_v128_store(panel_acc + channel_idx + 12,
                                            wasm_i32x4_add(wasm_v128_load(panel_acc + channel_idx + 12),
                                                           wasm_i32x4_mul(weight_i32x4, v3_i32x4)));
                        }
                        if (channel_idx < depth_padded) { // trailing 8-channel chunk; upper V lanes unused
                            v128_t values_i8x16 = wasm_v128_load64_zero(values_row + channel_idx);
                            v128_t low_i16x8 = wasm_i16x8_extend_low_i8x16(values_i8x16);
                            v128_t v0_i32x4 = wasm_i32x4_extend_low_i16x8(low_i16x8);
                            v128_t v1_i32x4 = wasm_i32x4_extend_high_i16x8(low_i16x8);
                            wasm_v128_store(panel_acc + channel_idx + 0,
                                            wasm_i32x4_add(wasm_v128_load(panel_acc + channel_idx + 0),
                                                           wasm_i32x4_mul(weight_i32x4, v0_i32x4)));
                            wasm_v128_store(panel_acc + channel_idx + 4,
                                            wasm_i32x4_add(wasm_v128_load(panel_acc + channel_idx + 4),
                                                           wasm_i32x4_mul(weight_i32x4, v1_i32x4)));
                        }
                    }
                    v128_t const correction_f32x4 = wasm_f32x4_splat(correction);
                    // drain: O = O · correction + panel_acc; products < 2^24 are exact in F32
                    for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                        wasm_v128_store(
                            output_row + channel_idx,
                            wasm_f32x4_add(wasm_f32x4_mul(wasm_v128_load(output_row + channel_idx), correction_f32x4),
                                           wasm_f32x4_convert_i32x4(wasm_v128_load(panel_acc + channel_idx))));
                }

                nk_f32_t const inverse_sum = running_sum > 0 ? 1 / running_sum : 0.0f;
                v128_t const inverse_sum_f32x4 = wasm_f32x4_splat(inverse_sum);
                nk_size_t const token = query_first + row_idx;
                nk_f32_t *destination = output + token * output_stride_floats + head_idx * depth;
                for (channel_idx = 0; channel_idx + 4 <= depth; channel_idx += 4)
                    wasm_v128_store(destination + channel_idx,
                                    wasm_f32x4_mul(wasm_v128_load(output_row + channel_idx), inverse_sum_f32x4));
                for (; channel_idx < depth; channel_idx++)
                    destination[channel_idx] = output_row[channel_idx] * inverse_sum;
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_(
                        (nk_f32_t)running_max * scale2, running_sum / 255.0f);
            }
        }
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_V128RELAXED
#endif // NUMKONG_ATTENTION_V128RELAXED_H
