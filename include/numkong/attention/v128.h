/**
 *  @file include/numkong/attention/v128.h
 *  @author Ash Vardanian
 *  @date September 12, 2026
 *  @brief Ragged attention packing for WASM with SIMD128.
 *
 *  @sa include/numkong/attention.h
 *
 *  Portable 128-bit packing side of the WebAssembly backend: the packed-KV layout, its size and
 *  shape queries, and the raw strided-row repack. Storage follows the `attention/haswell.h`
 *  conventions exactly: BF16, E4M3, and I8 stay in their source encoding at rest — packing is a raw
 *  strided-row copy with channels zero-padded to a multiple of 8. The compute kernels over that
 *  layout live in `attention/v128relaxed.h`.
 */
#ifndef NK_ATTENTION_V128_H
#define NK_ATTENTION_V128_H

#if NK_TARGET_V128

#include <wasm_simd128.h>

#include "numkong/attention/serial.h" // shared packed-KV header/directory, width-agnostic fallback
#include "numkong/cast/v128.h"        // `nk_bf16x4_to_f32x4_v128_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

enum {

    /** KV panel width in positions; the F32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_v128_k_ = 512,

    /** Deepest head this backend handles in scratch; deeper heads route to the serial tier. */
    nk_attention_max_depth_v128_k_ = 256,
};

NK_HELPER_INLINE v128_t nk_attention_load_bf16x4_v128_(void const *plane_chunk) {
    nk_b64_vec_t raw_vec;
    raw_vec.u64 = *(nk_u64_t const *)plane_chunk;
    return nk_bf16x4_to_f32x4_v128_(raw_vec).v128;
}

NK_HELPER_INLINE nk_size_t nk_attention_pack_size_v128_(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                        nk_size_t element_bytes) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t payload_bytes = 0; // planes keep the source encoding, like the dots family
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count * (nk_size_t)segment_lengths[segment_idx] * depth_padded *
                         element_bytes;
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_v128(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_v128_k_)
        return nk_attention_pack_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_pack_size_v128_(key_value_head_count, depth, segment_lengths, segment_count, sizeof(nk_bf16_t));
}

NK_API_COMPTIME void nk_attention_packed_shape_bf16_v128(void const *key_value_packed, nk_size_t *heads,
                                                         nk_size_t *depth, nk_size_t *segments) {
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
}

NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_v128(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_v128_k_)
        return nk_attention_pack_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_pack_size_v128_(key_value_head_count, depth, segment_lengths, segment_count, sizeof(nk_e4m3_t));
}

NK_API_COMPTIME void nk_attention_packed_shape_e4m3_v128(void const *key_value_packed, nk_size_t *heads,
                                                         nk_size_t *depth, nk_size_t *segments) {
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
}

NK_API_COMPTIME nk_size_t nk_attention_pack_size_i8_v128(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_v128_k_)
        return nk_attention_pack_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_pack_size_v128_(key_value_head_count, depth, segment_lengths, segment_count, 1);
}

NK_API_COMPTIME void nk_attention_packed_shape_i8_v128(void const *key_value_packed, nk_size_t *heads, nk_size_t *depth,
                                                       nk_size_t *segments) {
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
}

/** Raw strided-row repack: source encoding is preserved, tails zero-padded, 16-byte chunks. */
NK_HELPER_INLINE void nk_attention_pack_v128_(                                         //
    void const *keys, void const *values, nk_size_t element_bytes,                     //
    nk_size_t key_value_head_count, nk_size_t depth,                                   //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                  //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, //
    void *key_value_packed, nk_size_t task_begin, nk_size_t task_end) {

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const row_bytes = depth * element_bytes;
    nk_size_t const padded_row_bytes = depth_padded * element_bytes;
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 task_begin, 1, padded_row_bytes);
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *payload_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (task_begin >= total_tasks) return;
    if (task_end > total_tasks) task_end = total_tasks;

    for (nk_size_t task_idx = task_begin; task_idx < task_end; task_idx++) {
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

NK_API_COMPTIME void nk_attention_pack_bf16_v128(                                      //
    nk_bf16_t const *keys, nk_bf16_t const *values,                                    //
    nk_size_t key_value_head_count, nk_size_t depth,                                   //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                  //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, //
    void *key_value_packed, nk_size_t task_begin, nk_size_t task_end) {
    if (depth > nk_attention_max_depth_v128_k_) {
        nk_attention_pack_bf16_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                      task_end);
        return;
    }
    nk_attention_pack_v128_(keys, values, sizeof(nk_bf16_t), key_value_head_count, depth, segment_offsets,
                            segment_lengths, segment_count, key_stride_bytes, value_stride_bytes, key_value_packed,
                            task_begin, task_end);
}

NK_API_COMPTIME void nk_attention_pack_e4m3_v128(                                      //
    nk_e4m3_t const *keys, nk_e4m3_t const *values,                                    //
    nk_size_t key_value_head_count, nk_size_t depth,                                   //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                  //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, //
    void *key_value_packed, nk_size_t task_begin, nk_size_t task_end) {
    if (depth > nk_attention_max_depth_v128_k_) {
        nk_attention_pack_e4m3_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                      task_end);
        return;
    }
    nk_attention_pack_v128_(keys, values, sizeof(nk_e4m3_t), key_value_head_count, depth, segment_offsets,
                            segment_lengths, segment_count, key_stride_bytes, value_stride_bytes, key_value_packed,
                            task_begin, task_end);
}

NK_API_COMPTIME void nk_attention_pack_i8_v128(                                        //
    nk_i8_t const *keys, nk_i8_t const *values,                                        //
    nk_size_t key_value_head_count, nk_size_t depth,                                   //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                  //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, //
    void *key_value_packed, nk_size_t task_begin, nk_size_t task_end) {
    if (depth > nk_attention_max_depth_v128_k_) {
        nk_attention_pack_i8_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                    segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                    task_end);
        return;
    }
    nk_attention_pack_v128_(keys, values, 1, key_value_head_count, depth, segment_offsets, segment_lengths,
                            segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                            task_end);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128
#endif // NK_ATTENTION_V128_H
