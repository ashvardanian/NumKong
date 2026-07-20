/**
 *  @brief Ragged attention for WASM with Relaxed SIMD.
 *  @file include/numkong/attention/v128relaxed.h
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  Portable 128-bit backend for WebAssembly engines with the Relaxed SIMD proposal.
 *  Storage follows the `attention/haswell.h` conventions exactly: BF16, E4M3, and I8 stay
 *  in their source encoding at rest — packing is a raw strided-row copy with channels
 *  zero-padded to a multiple of 8 — and every value widens to F32 on the fly inside the
 *  compute loops through the `cast/v128relaxed.h` helpers. Per query row, KV is swept in
 *  512-position panels with an exact online running-max correction; the F32 score row
 *  (2 KB) stays L1-resident. `depth > 256` routes to the width-agnostic serial tier.
 *
 *  The base-2 exponent is the family's shared degree-4 polynomial, evaluated 4-wide with
 *  `wasm_f32x4_relaxed_madd` after a `wasm_f32x4_nearest` range reduction and the same
 *  denormal-avoiding [−125, 127] clamps as `nk_f32_exp2_serial_`; scalar panel tails call
 *  the serial helper directly to keep the family polynomial end-to-end.
 *
 *  The I8 path keeps QK scores exact in I32: `wasm_i32x4_relaxed_dot_i8x16_i7x16_add`
 *  requires a 7-bit second operand, so K is bit-split as `k = k₇ − 128·[k < 0]` — the dot
 *  runs on the low 7 bits while an I16 pairwise correction (Σq over K-negative lanes, ×128)
 *  is folded into the I32 sum vector before the single horizontal reduce per position.
 *  Attention weights quantize to U8 exactly like serial, `trunc(2^(s₂−m₂) · 255 + 0.5)`,
 *  vectorized 4-wide with the same separate multiply and add so the rounding matches the
 *  scalar reference bit-for-bit. In the PV accumulation `wasm_f32x4_relaxed_madd` is safe
 *  even under that bit-exactness contract: a U8 weight (≤ 255) times an I8 plane value
 *  (|v| ≤ 128) is at most 32640 < 2^24, so every product is exactly representable in F32
 *  and fusing the multiply into the add cannot change a single bit of the accumulation.
 */
#ifndef NK_ATTENTION_V128RELAXED_H
#define NK_ATTENTION_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include <wasm_simd128.h>

#include "numkong/attention/serial.h"   // shared packed-KV header/directory, width-agnostic fallback
#include "numkong/cast/v128relaxed.h"   // widening helpers like `nk_bf16x4_to_f32x4_v128relaxed_`
#include "numkong/reduce/v128relaxed.h" // `nk_reduce_add_f32x4_v128relaxed_`, `nk_reduce_max_f32x4_v128relaxed_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

enum {
    /** @brief KV panel width in positions; the F32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_v128relaxed_k_ = 512,
    /** @brief Deepest head this backend handles in scratch; deeper heads route to the serial tier. */
    nk_attention_max_depth_v128relaxed_k_ = 256,
};

/** @brief Fast vectorized 2^x: exact range reduction + the family's shared degree-4 polynomial. */
NK_HELPER_INLINE v128_t nk_attention_exp2_f32x4_v128relaxed_(v128_t x_f32x4) {
    x_f32x4 = wasm_f32x4_max(wasm_f32x4_min(x_f32x4, wasm_f32x4_splat(127.0f)), wasm_f32x4_splat(-125.0f));
    v128_t whole_f32x4 = wasm_f32x4_nearest(x_f32x4);
    v128_t reduced_f32x4 = wasm_f32x4_sub(x_f32x4, whole_f32x4);
    v128_t poly_f32x4 = wasm_f32x4_splat(9.61812910e-3f);
    poly_f32x4 = wasm_f32x4_relaxed_madd(poly_f32x4, reduced_f32x4, wasm_f32x4_splat(5.55041087e-2f));
    poly_f32x4 = wasm_f32x4_relaxed_madd(poly_f32x4, reduced_f32x4, wasm_f32x4_splat(2.40226507e-1f));
    poly_f32x4 = wasm_f32x4_relaxed_madd(poly_f32x4, reduced_f32x4, wasm_f32x4_splat(6.93147181e-1f));
    poly_f32x4 = wasm_f32x4_relaxed_madd(poly_f32x4, reduced_f32x4, wasm_f32x4_splat(1.0f));
    v128_t whole_i32x4 = wasm_i32x4_trunc_sat_f32x4(whole_f32x4); // integral and clamped, so exact
    v128_t power_f32x4 = wasm_i32x4_shl(wasm_i32x4_add(whole_i32x4, wasm_i32x4_splat(127)), 23);
    return wasm_f32x4_mul(poly_f32x4, power_f32x4);
}

/**
 *  @brief I-BERT-style integer exponential for the I8 weight path: takes the base-2 argument as a Q15
 *         fixed-point value in `[−10·2^15, 0]` and returns `round(2^t · 255)` as a U8 weight in each i32
 *         lane. Pure integer — SIMD128 has no per-lane variable shift, so the `1 << (13−whole)` bias and
 *         the `>> (14−whole)` are synthesized with a 4-stage `bitselect` barrel network keyed on
 *         `nlz = −whole ∈ [0,10]`. Mirrors the validated SME/Ice Lake helper (no floating point).
 */
NK_HELPER_INLINE v128_t nk_attention_iexp2_weight_i32x4_v128relaxed_(v128_t t_q15_i32x4) {
    v128_t const zero_i32x4 = wasm_i32x4_splat(0);
    v128_t const whole_i32x4 = wasm_i32x4_shr(t_q15_i32x4, 15); // arithmetic floor, in [-10,0]
    v128_t const fraction_i32x4 = wasm_v128_and(t_q15_i32x4, wasm_i32x4_splat(0x7FFF));
    v128_t poly_i32x4 = wasm_i32x4_splat(1296); // Q14 Chebyshev coefficients, degree 3
    poly_i32x4 = wasm_i32x4_add(wasm_i32x4_shr(wasm_i32x4_mul(fraction_i32x4, poly_i32x4), 15), wasm_i32x4_splat(3678));
    poly_i32x4 = wasm_i32x4_add(wasm_i32x4_shr(wasm_i32x4_mul(fraction_i32x4, poly_i32x4), 15),
                                wasm_i32x4_splat(11410));
    poly_i32x4 = wasm_i32x4_add(wasm_i32x4_shr(wasm_i32x4_mul(fraction_i32x4, poly_i32x4), 15),
                                wasm_i32x4_splat(16382));
    v128_t const scaled_i32x4 = wasm_i32x4_sub(wasm_i32x4_shl(poly_i32x4, 8), poly_i32x4); // poly·255
    v128_t const nlz_i32x4 = wasm_i32x4_sub(zero_i32x4, whole_i32x4);                      // -whole, in [0,10]
    v128_t const mask1_i32x4 = wasm_i32x4_ne(wasm_v128_and(nlz_i32x4, wasm_i32x4_splat(1)), zero_i32x4);
    v128_t const mask2_i32x4 = wasm_i32x4_ne(wasm_v128_and(nlz_i32x4, wasm_i32x4_splat(2)), zero_i32x4);
    v128_t const mask4_i32x4 = wasm_i32x4_ne(wasm_v128_and(nlz_i32x4, wasm_i32x4_splat(4)), zero_i32x4);
    v128_t const mask8_i32x4 = wasm_i32x4_ne(wasm_v128_and(nlz_i32x4, wasm_i32x4_splat(8)), zero_i32x4);
    v128_t bias_i32x4 = wasm_i32x4_splat(1 << 13);
    // bias = 1 << (13 + nlz) = (1<<13) << nlz — round-half-up bias, added before the full shift
    bias_i32x4 = wasm_v128_bitselect(wasm_i32x4_shl(bias_i32x4, 1), bias_i32x4, mask1_i32x4);
    bias_i32x4 = wasm_v128_bitselect(wasm_i32x4_shl(bias_i32x4, 2), bias_i32x4, mask2_i32x4);
    bias_i32x4 = wasm_v128_bitselect(wasm_i32x4_shl(bias_i32x4, 4), bias_i32x4, mask4_i32x4);
    bias_i32x4 = wasm_v128_bitselect(wasm_i32x4_shl(bias_i32x4, 8), bias_i32x4, mask8_i32x4);
    v128_t r_i32x4 = wasm_i32x4_shr(wasm_i32x4_add(scaled_i32x4, bias_i32x4), 14);
    // result = (scaled + bias) >> (14 + nlz) = ((scaled+bias) >> 14) >> nlz (exact for non-negatives)
    r_i32x4 = wasm_v128_bitselect(wasm_i32x4_shr(r_i32x4, 1), r_i32x4, mask1_i32x4);
    r_i32x4 = wasm_v128_bitselect(wasm_i32x4_shr(r_i32x4, 2), r_i32x4, mask2_i32x4);
    r_i32x4 = wasm_v128_bitselect(wasm_i32x4_shr(r_i32x4, 4), r_i32x4, mask4_i32x4);
    r_i32x4 = wasm_v128_bitselect(wasm_i32x4_shr(r_i32x4, 8), r_i32x4, mask8_i32x4);
    return r_i32x4; // U8 weight (0..255) per i32 lane
}

/** @brief Widens 4 raw plane scalars (BF16 or E4M3 at rest) to F32 inside the hot loops. */
typedef v128_t (*nk_attention_load_v128relaxed_t_)(void const *plane_chunk);

NK_HELPER_INLINE v128_t nk_attention_load_bf16x4_v128relaxed_(void const *plane_chunk) {
    nk_b64_vec_t raw_vec;
    raw_vec.u64 = *(nk_u64_t const *)plane_chunk;
    return nk_bf16x4_to_f32x4_v128relaxed_(raw_vec).v128;
}

NK_HELPER_INLINE v128_t nk_attention_load_e4m3x4_v128relaxed_(void const *plane_chunk) {
    nk_b32_vec_t raw_vec;
    raw_vec.u32 = *(nk_u32_t const *)plane_chunk;
    return nk_e4m3x4_to_f32x4_v128relaxed_(raw_vec).v128;
}

NK_HELPER_INLINE nk_size_t nk_attention_pack_size_v128relaxed_(nk_size_t key_value_head_count, nk_size_t depth,
                                                               nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                               nk_size_t element_bytes) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t payload_bytes = 0; // planes keep the source encoding, like the dots family
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count * (nk_size_t)segment_lengths[segment_idx] * depth_padded *
                         element_bytes;
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_v128relaxed(nk_size_t key_value_head_count, nk_size_t depth,
                                                                  nk_u32_t const *segment_lengths,
                                                                  nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_v128relaxed_k_)
        return nk_attention_pack_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_pack_size_v128relaxed_(key_value_head_count, depth, segment_lengths, segment_count,
                                               sizeof(nk_bf16_t));
}

NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_v128relaxed(nk_size_t key_value_head_count, nk_size_t depth,
                                                                  nk_u32_t const *segment_lengths,
                                                                  nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_v128relaxed_k_)
        return nk_attention_pack_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_pack_size_v128relaxed_(key_value_head_count, depth, segment_lengths, segment_count,
                                               sizeof(nk_e4m3_t));
}

NK_API_COMPTIME nk_size_t nk_attention_pack_size_i8_v128relaxed(nk_size_t key_value_head_count, nk_size_t depth,
                                                                nk_u32_t const *segment_lengths,
                                                                nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_v128relaxed_k_)
        return nk_attention_pack_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_pack_size_v128relaxed_(key_value_head_count, depth, segment_lengths, segment_count, 1);
}

/** @brief Raw strided-row repack: source encoding is preserved, tails zero-padded, 16-byte chunks. */
NK_HELPER_INLINE void nk_attention_pack_v128relaxed_(                                  //
    void const *keys, void const *values, nk_size_t element_bytes,                     //
    nk_size_t key_value_head_count, nk_size_t depth,                                   //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                  //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, //
    void *key_value_packed, nk_size_t first_task, nk_size_t task_count) {

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const row_bytes = depth * element_bytes;
    nk_size_t const padded_row_bytes = depth_padded * element_bytes;
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
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride_bytes +
                                   key_value_head_idx * row_bytes;
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride_bytes +
                                     key_value_head_idx * row_bytes;
            char *keys_destination = keys_plane + position_idx * padded_row_bytes;
            char *values_destination = values_plane + position_idx * padded_row_bytes;
            nk_size_t byte_idx = 0;
            for (; byte_idx + 16 <= row_bytes; byte_idx += 16)
                wasm_v128_store(keys_destination + byte_idx, wasm_v128_load(keys_row + byte_idx));
            for (; byte_idx < row_bytes; byte_idx++) keys_destination[byte_idx] = keys_row[byte_idx];
            for (; byte_idx < padded_row_bytes; byte_idx++) keys_destination[byte_idx] = 0;
            for (byte_idx = 0; byte_idx + 16 <= row_bytes; byte_idx += 16)
                wasm_v128_store(values_destination + byte_idx, wasm_v128_load(values_row + byte_idx));
            for (; byte_idx < row_bytes; byte_idx++) values_destination[byte_idx] = values_row[byte_idx];
            for (; byte_idx < padded_row_bytes; byte_idx++) values_destination[byte_idx] = 0;
        }
    }
}

NK_API_COMPTIME void nk_attention_pack_bf16_v128relaxed(                               //
    nk_bf16_t const *keys, nk_bf16_t const *values,                                    //
    nk_size_t key_value_head_count, nk_size_t depth,                                   //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                  //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, //
    void *key_value_packed, nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_v128relaxed_k_) {
        nk_attention_pack_bf16_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_v128relaxed_(keys, values, sizeof(nk_bf16_t), key_value_head_count, depth, segment_offsets,
                                   segment_lengths, segment_count, key_stride_bytes, value_stride_bytes,
                                   key_value_packed, first_task, task_count);
}

NK_API_COMPTIME void nk_attention_pack_e4m3_v128relaxed(                               //
    nk_e4m3_t const *keys, nk_e4m3_t const *values,                                    //
    nk_size_t key_value_head_count, nk_size_t depth,                                   //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                  //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, //
    void *key_value_packed, nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_v128relaxed_k_) {
        nk_attention_pack_e4m3_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_v128relaxed_(keys, values, sizeof(nk_e4m3_t), key_value_head_count, depth, segment_offsets,
                                   segment_lengths, segment_count, key_stride_bytes, value_stride_bytes,
                                   key_value_packed, first_task, task_count);
}

NK_API_COMPTIME void nk_attention_pack_i8_v128relaxed(                                 //
    nk_i8_t const *keys, nk_i8_t const *values,                                        //
    nk_size_t key_value_head_count, nk_size_t depth,                                   //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                  //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, //
    void *key_value_packed, nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_v128relaxed_k_) {
        nk_attention_pack_i8_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                    segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                    task_count);
        return;
    }
    nk_attention_pack_v128relaxed_(keys, values, 1, key_value_head_count, depth, segment_offsets, segment_lengths,
                                   segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                   task_count);
}

/**
 *  @brief Shared attention core over raw-encoded planes: per query row, panel-flash with
 *         an exact online correction; queries widen once per row, planes widen in-loop.
 */
NK_HELPER_INLINE void nk_attention_packed_float_v128relaxed_(                    //
    void const *queries, nk_size_t element_bytes,                                //
    nk_attention_load_f32_serial_t_ load_f32,                                    //
    nk_attention_load_v128relaxed_t_ load,                                       //
    void const *key_value_packed, nk_f32_t *output,                              //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->key_value_head_count != key_value_head_count) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_(segment_count);
    nk_size_t const output_stride_floats = output_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const plane_row_bytes = depth_padded * element_bytes;
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // softmax(x) = softmax₂(x·log₂e)
    nk_size_t const panel_width = nk_attention_panel_v128relaxed_k_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_f32_t query_row[nk_attention_max_depth_v128relaxed_k_];
    NK_ALIGN64 nk_f32_t output_row[nk_attention_max_depth_v128relaxed_k_];
    NK_ALIGN64 nk_f32_t scores[nk_attention_panel_v128relaxed_k_];

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
            char const *query_source = (char const *)queries +
                                       (query_offsets[segment_idx] + row_idx) * query_stride_bytes +
                                       head_idx * depth * element_bytes;
            nk_size_t channel_idx = 0;
            for (; channel_idx < depth; channel_idx++)
                query_row[channel_idx] = load_f32(query_source + channel_idx * element_bytes);
            for (; channel_idx < depth_padded; channel_idx++) query_row[channel_idx] = 0.0f;
            for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                wasm_v128_store(output_row + channel_idx, wasm_f32x4_splat(0.0f));
            nk_f32_t running_max2 = NK_F32_MIN, running_sum = 0;

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_length = (panel_start + panel_width <= position_count)
                                                   ? panel_width
                                                   : (position_count - panel_start);
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
                    scores[position_idx] = nk_reduce_add_f32x4_v128relaxed_(wasm_f32x4_add(sum0_f32x4, sum1_f32x4));
                }

                v128_t const scale2_f32x4 = wasm_f32x4_splat(scale2);
                v128_t max_f32x4 = wasm_f32x4_splat(NK_F32_MIN);
                for (position_idx = 0; position_idx + 4 <= panel_length; position_idx += 4)
                    max_f32x4 = wasm_f32x4_max(max_f32x4,
                                               wasm_f32x4_mul(wasm_v128_load(scores + position_idx), scale2_f32x4));
                nk_f32_t panel_max2 = nk_reduce_max_f32x4_v128relaxed_(max_f32x4);
                for (; position_idx < panel_length; position_idx++) {
                    nk_f32_t const scaled2 = scores[position_idx] * scale2;
                    if (scaled2 > panel_max2) panel_max2 = scaled2;
                }
                nk_f32_t const new_max2 = running_max2 > panel_max2 ? running_max2 : panel_max2;
                nk_f32_t const correction = wasm_f32x4_extract_lane(
                    nk_attention_exp2_f32x4_v128relaxed_(wasm_f32x4_splat(running_max2 - new_max2)), 0);
                running_max2 = new_max2;

                v128_t const new_max2_f32x4 = wasm_f32x4_splat(new_max2);
                v128_t panel_sum_f32x4 = wasm_f32x4_splat(0.0f);
                nk_f32_t panel_sum = 0;
                for (position_idx = 0; position_idx + 4 <= panel_length; position_idx += 4) {
                    v128_t weight_f32x4 = nk_attention_exp2_f32x4_v128relaxed_(wasm_f32x4_sub(
                        wasm_f32x4_mul(wasm_v128_load(scores + position_idx), scale2_f32x4), new_max2_f32x4));
                    panel_sum_f32x4 = wasm_f32x4_add(panel_sum_f32x4, weight_f32x4);
                    wasm_v128_store(scores + position_idx, weight_f32x4);
                }
                panel_sum = nk_reduce_add_f32x4_v128relaxed_(panel_sum_f32x4);
                for (; position_idx < panel_length; position_idx++) { // scalar tail keeps the family exp2 end-to-end
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
                        wasm_v128_store(
                            output_row + channel_idx + 4,
                            wasm_f32x4_relaxed_madd(weight_f32x4, load(values_row + chunk_bytes + 4 * element_bytes),
                                                    wasm_v128_load(output_row + channel_idx + 4)));
                    }
                }
            }

            nk_f32_t const inverse_sum = 1.0f / running_sum;
            v128_t const inverse_sum_f32x4 = wasm_f32x4_splat(inverse_sum);
            nk_f32_t *destination = output + (query_offsets[segment_idx] + row_idx) * output_stride_floats +
                                    head_idx * depth;
            for (channel_idx = 0; channel_idx + 4 <= depth; channel_idx += 4)
                wasm_v128_store(destination + channel_idx,
                                wasm_f32x4_mul(wasm_v128_load(output_row + channel_idx), inverse_sum_f32x4));
            for (; channel_idx < depth; channel_idx++) destination[channel_idx] = output_row[channel_idx] * inverse_sum;
        }
    }
}

NK_API_COMPTIME void nk_attention_packed_bf16_v128relaxed(                       //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_v128relaxed_k_) {
        nk_attention_packed_bf16_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_attention_packed_float_v128relaxed_(queries, sizeof(nk_bf16_t), &nk_attention_load_bf16_serial_,
                                           &nk_attention_load_bf16x4_v128relaxed_, key_value_packed, output, head_count,
                                           key_value_head_count, depth, query_offsets, query_stride_bytes,
                                           output_stride_bytes, scale, first_task, task_count);
}

NK_API_COMPTIME void nk_attention_packed_e4m3_v128relaxed(                       //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_v128relaxed_k_) {
        nk_attention_packed_e4m3_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_attention_packed_float_v128relaxed_(queries, sizeof(nk_e4m3_t), &nk_attention_load_e4m3_serial_,
                                           &nk_attention_load_e4m3x4_v128relaxed_, key_value_packed, output, head_count,
                                           key_value_head_count, depth, query_offsets, query_stride_bytes,
                                           output_stride_bytes, scale, first_task, task_count);
}

NK_API_COMPTIME void nk_attention_packed_i8_v128relaxed(                         //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,      //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_v128relaxed_k_) {
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
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const depth_full16 = depth_padded & ~(nk_size_t)15;
    nk_size_t const depth_padded16 = nk_size_round_up_to_multiple_(depth_padded, 16);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // softmax(x) = softmax₂(x·log₂e)
    nk_size_t const panel_width = nk_attention_panel_v128relaxed_k_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_i8_t query_i8[nk_attention_max_depth_v128relaxed_k_];
    NK_ALIGN64 nk_f32_t output_row[nk_attention_max_depth_v128relaxed_k_];
    NK_ALIGN64 nk_i32_t scores[nk_attention_panel_v128relaxed_k_];
    NK_ALIGN64 nk_u8_t weights[nk_attention_panel_v128relaxed_k_];
    NK_ALIGN64 nk_i32_t panel_acc[nk_attention_max_depth_v128relaxed_k_]; // per-panel integer P×V accumulator

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
            for (; channel_idx + 16 <= depth; channel_idx += 16)
                wasm_v128_store(query_i8 + channel_idx, wasm_v128_load(query_source + channel_idx));
            for (; channel_idx < depth; channel_idx++) query_i8[channel_idx] = query_source[channel_idx];
            for (; channel_idx < depth_padded16; channel_idx++) query_i8[channel_idx] = 0;
            for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                wasm_v128_store(output_row + channel_idx, wasm_f32x4_splat(0.0f));
            nk_i32_t running_max = NK_I32_MIN;
            nk_f32_t running_sum = 0;
            nk_i32_t const scale_fixed = (nk_i32_t)(scale2 * 32768.0f + 0.5f); // Q15 scale for the integer exp
            nk_i32_t const delta_floor = // score delta below which every weight quantizes to zero
                scale_fixed > 0 ? -(nk_i32_t)((10u << 15) / (nk_u32_t)scale_fixed) - 1 : 0;

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_length = (panel_start + panel_width <= position_count)
                                                   ? panel_width
                                                   : (position_count - panel_start);
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
                    scores[position_idx] = nk_reduce_add_i32x4_v128relaxed_(score_i32x4);
                }

                v128_t max_i32x4 = wasm_i32x4_splat(NK_I32_MIN);
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
                    nk_attention_exp2_f32x4_v128relaxed_(
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
                    v128_t weight_i32x4 = nk_attention_iexp2_weight_i32x4_v128relaxed_(
                        wasm_i32x4_mul(delta_i32x4, scale_fixed_i32x4));
                    panel_sum_i32x4 = wasm_i32x4_add(panel_sum_i32x4, weight_i32x4);
                    v128_t weight_i16x8 = wasm_i16x8_narrow_i32x4(weight_i32x4, weight_i32x4);
                    wasm_v128_store32_lane(weights + position_idx, wasm_u8x16_narrow_i16x8(weight_i16x8, weight_i16x8),
                                           0);
                }
                if (position_idx < panel_length) { // masked vector tail — inactive lanes forced to a zero weight
                    v128_t const lane_index_i32x4 = wasm_i32x4_make(0, 1, 2, 3);
                    v128_t const tail_mask_i32x4 = wasm_i32x4_lt(
                        lane_index_i32x4, wasm_i32x4_splat((nk_i32_t)(panel_length - position_idx)));
                    v128_t delta_i32x4 = wasm_i32x4_max(
                        wasm_i32x4_sub(wasm_v128_load(scores + position_idx), new_max_i32x4), delta_floor_i32x4);
                    v128_t weight_i32x4 = wasm_v128_and(
                        nk_attention_iexp2_weight_i32x4_v128relaxed_(wasm_i32x4_mul(delta_i32x4, scale_fixed_i32x4)),
                        tail_mask_i32x4);
                    panel_sum_i32x4 = wasm_i32x4_add(panel_sum_i32x4, weight_i32x4);
                    v128_t weight_i16x8 = wasm_i16x8_narrow_i32x4(weight_i32x4, weight_i32x4);
                    wasm_v128_store32_lane(weights + position_idx, wasm_u8x16_narrow_i16x8(weight_i16x8, weight_i16x8),
                                           0);
                }
                running_sum = running_sum * correction + (nk_f32_t)nk_reduce_add_i32x4_v128relaxed_(panel_sum_i32x4);

                // Integer P×V: zero the panel accumulator, sum U8·I8 products per non-zero position, then drain.
                for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                    wasm_v128_store(panel_acc + channel_idx, wasm_i32x4_splat(0));
                for (position_idx = 0; position_idx < panel_length; position_idx++) {
                    nk_u8_t const weight_u8 = weights[position_idx];
                    if (weight_u8 == 0) continue; // sparse U8 weights: skipping dead positions is the kernel's lever
                    v128_t const weight_i32x4 = wasm_i32x4_splat((nk_i32_t)weight_u8);
                    char const *values_row = values_plane + (panel_start + position_idx) * depth_padded;
                    channel_idx = 0;
                    for (; channel_idx < depth_full16; channel_idx += 16) { // one 16-byte V load, widen to four i32x4
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
                // drain: O = O·correction + panel_acc; integer products < 2^24 convert to F32 exactly
                for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 4)
                    wasm_v128_store(
                        output_row + channel_idx,
                        wasm_f32x4_add(wasm_f32x4_mul(wasm_v128_load(output_row + channel_idx), correction_f32x4),
                                       wasm_f32x4_convert_i32x4(wasm_v128_load(panel_acc + channel_idx))));
            }

            nk_f32_t const inverse_sum = 1.0f / running_sum;
            v128_t const inverse_sum_f32x4 = wasm_f32x4_splat(inverse_sum);
            nk_f32_t *destination = output + (query_offsets[segment_idx] + row_idx) * output_stride_floats +
                                    head_idx * depth;
            for (channel_idx = 0; channel_idx + 4 <= depth; channel_idx += 4)
                wasm_v128_store(destination + channel_idx,
                                wasm_f32x4_mul(wasm_v128_load(output_row + channel_idx), inverse_sum_f32x4));
            for (; channel_idx < depth; channel_idx++) destination[channel_idx] = output_row[channel_idx] * inverse_sum;
        }
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_ATTENTION_V128RELAXED_H
