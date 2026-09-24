/**
 *  @file include/numkong/attention/serial.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief Serial (SIMD-free) ragged attention baseline.
 *
 *  @sa include/numkong/attention.h
 *
 *  Width-agnostic reference implementation of the ragged scaled-dot-product attention family: any
 *  `depth ≥ 1`, any segment lengths, GQA/MQA, the same base-2 softmax formulation and the same
 *  [task_start, task_start + task_count) windows over the flat @b [segment,head] grid as the SIMD
 *  backends, in both the bidirectional and the causal mode.
 *
 *  @section attention_serial_roles Roles in the Family
 *
 *  1. Ground truth for every SIMD backend's conformance tests.
 *  2. Final runtime-dispatch fallback on CPUs without any compiled SIMD target.
 *  3. Fallback for shapes outside a SIMD backend's fast-path envelope (`depth > 256`), invoked from
 *     those backends so pack and attention always agree on the packed-buffer format.
 *
 *  @section attention_serial_layout Packed Layout
 *
 *  This file also owns the family-shared packed-KV header and offsets layout:
 *
 *  @verbatim
 *  [64 B header]
 *  [offsets: u64 payload_offsets[segments+1] + u32 segment_lengths[segments], 64-byte padded]
 *  [per-backend payload]
 *  @endverbatim
 *
 *  The payload is backend-opaque; serial stores K and V as plain F32 row-major planes
 *  `[key_value_head][position][channel]` per segment, so both input dtypes — BF16, E4M3 — share one
 *  compute path after per-element conversion at pack.
 *
 *  Like the rest of the serial tier, no libm: the base-2 exponent uses the same degree-4 polynomial
 *  and the same denormal-avoiding clamp as the AVX-512 helper, so serial and vector paths agree to
 *  polynomial precision.
 *
 *  @sa nk_f32_exp2_serial_
 *
 *  @section attention_serial_i8 I8 Weight Quantization
 *
 *  The I8 path keeps scores exact in I32 integer arithmetic and quantizes softmax weights to U8 as
 *  round(255 · 2^(s₂ − m₂)); the max-scoring position always lands on 255, so the weight sum can
 *  never be zero. Normalizing by that sum cancels the 255, so no descale constant remains.
 */
#ifndef NK_ATTENTION_SERIAL_H
#define NK_ATTENTION_SERIAL_H

#include "numkong/types.h"
#include "numkong/scalar/serial.h" // `nk_f32_exp2_serial_`, `NK_F32_LOG2E_`
#include "numkong/cast/serial.h"   // `nk_bf16_to_f32_serial`, `nk_e4m3_to_f32_serial`

#if defined(__cplusplus)
extern "C" {
#endif

/*  GCC inlines a helper only into callers whose targets include its own, so serial code builds at
 *  the Armv8-A floor. */
#if defined(__GNUC__) && !defined(__clang__) && NK_TARGET_ARM64_
#pragma GCC push_options
#pragma GCC target("arch=armv8-a")
#endif

/** Packed ragged KV cache header (64 bytes), shared by all attention backends. Followed by the
 *  segment offsets table; the payload beyond it is backend-specific. */
typedef struct {

    /** Number of K/V heads, ≤ query heads for GQA. */
    nk_u32_t heads;

    /** Head dimension the buffer was packed for. */
    nk_u32_t depth;

    /** Number of independent segments packed. */
    nk_u32_t segments;

    /** Zeroed; pads the header to 64 bytes. */
    nk_u32_t reserved[13];
} nk_attention_packed_header_t;

/** Offsets table size in bytes, 64-byte padded: @p segment_count + 1 u64 payload offsets, then
 *  @p segment_count u32 lengths. */
NK_HELPER_INLINE nk_size_t nk_attention_pack_directory_size_(nk_size_t segment_count) {
    return nk_size_round_up_to_multiple_((segment_count + 1) * sizeof(nk_u64_t) + segment_count * sizeof(nk_u32_t), 64);
}

/**
 *  @brief Writes the family-shared packed-KV header and segment offsets table.
 *
 *  Only the window covering task 0 writes — later windows require the offsets table present
 *  (the race-free parallel-pack contract). Per-segment payload bytes follow a formula that
 *  covers every backend:
 *
 *  @verbatim
 *  2 · key_value_head_count · round_up(length, position_multiple) · unit_bytes
 *  @endverbatim
 */
NK_HELPER_INLINE void nk_attention_pack_directory_(void *key_value_packed, nk_size_t key_value_head_count,
                                                   nk_size_t depth, nk_u32_t const *segment_lengths,
                                                   nk_size_t segment_count, nk_size_t task_begin,
                                                   nk_size_t position_multiple, nk_size_t unit_bytes) {
    if (task_begin != 0) return;
    // Zero the whole directory — header plus the offsets table including its 64-byte-aligned tail —
    // so the packed blob is a pure function of its inputs, with no allocator garbage in the unwritten
    // slack. The per-segment payload planes are zero-filled by each backend, so this makes the pack
    // hermetic and the caller need not pre-zero the buffer.

    nk_size_t const directory_bytes = sizeof(nk_attention_packed_header_t) +
                                      nk_attention_pack_directory_size_(segment_count);
    for (nk_size_t byte_index = 0; byte_index < directory_bytes; byte_index++)
        ((char *)key_value_packed)[byte_index] = 0;
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    header->heads = (nk_u32_t)key_value_head_count;
    header->depth = (nk_u32_t)depth;
    header->segments = (nk_u32_t)segment_count;
    nk_u64_t *payload_offsets = (nk_u64_t *)((char *)key_value_packed + sizeof(*header));
    nk_u32_t *lengths_copy = (nk_u32_t *)(payload_offsets + segment_count + 1);
    nk_u64_t running = 0;
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++) {
        payload_offsets[segment_idx] = running;
        lengths_copy[segment_idx] = segment_lengths[segment_idx];
        running += 2 * key_value_head_count *
                   (nk_u64_t)nk_size_round_up_to_multiple_(segment_lengths[segment_idx], position_multiple) *
                   unit_bytes;
    }
    payload_offsets[segment_count] = running;
}

/**
 *  @brief Reads a packed KV cache's shape from its header.
 *
 *  Shared by every per-(dtype, ISA) nk_attention_packed_shape_* accessor.
 */
NK_HELPER_INLINE void nk_attention_packed_shape_(void const *key_value_packed, nk_size_t *heads, nk_size_t *depth,
                                                 nk_size_t *segments) {
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    *heads = header->heads;
    *depth = header->depth;
    *segments = header->segments;
}

/** Writes the half-open range of keys visible to the query at @p position, in a segment of
 *  @p length keys, into @p key_begin and @p key_end. @p window counts the visible keys including
 *  the query itself; a negative @p position or a zero @p window is empty. */
NK_HELPER_INLINE void nk_attention_row_range_(nk_i64_t position, nk_size_t window, nk_size_t length,
                                              nk_size_t *key_begin, nk_size_t *key_end) {
    if (position < 0 || window == 0) {
        *key_begin = *key_end = 0;
        return;
    }
    nk_size_t const query_position = (nk_size_t)position;
    *key_end = query_position < length ? query_position + 1 : length;
    *key_begin = window > query_position ? 0 : query_position - window + 1;
    if (*key_begin > *key_end) *key_begin = *key_end;
}

/** Exclusive end of the window of @p task_count tasks from @p task_start, clipped to
 *  @p total_tasks. */
NK_HELPER_INLINE nk_size_t nk_attention_task_end_(nk_size_t task_start, nk_size_t task_count, nk_size_t total_tasks) {
    if (task_start >= total_tasks) return task_start;
    return task_count < total_tasks - task_start ? task_start + task_count : total_tasks;
}

/** Per-element widening converter, `maxsim/serial.h`-style dtype abstraction. */
typedef nk_f32_t (*nk_attention_load_f32_serial_t_)(void const *element);

NK_HELPER_INLINE nk_f32_t nk_attention_load_bf16_serial_(void const *element) {
    nk_f32_t result;
    nk_bf16_to_f32_serial((nk_bf16_t const *)element, &result);
    return result;
}

NK_HELPER_INLINE nk_f32_t nk_attention_load_e4m3_serial_(void const *element) {
    nk_f32_t result;
    nk_e4m3_to_f32_serial((nk_e4m3_t const *)element, &result);
    return result;
}

NK_HELPER_INLINE nk_size_t nk_attention_pack_size_serial_(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    nk_size_t payload_bytes = 0;
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count * (nk_size_t)segment_lengths[segment_idx] * depth *
                         sizeof(nk_f32_t); // K + V
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

/*  Keep the serial instantiations below actually scalar, regardless of build type.
 *  See dots/serial.h for rationale. */
#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NK_API_COMPTIME nk_size_t nk_attention_pack_size_bf16_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    return nk_attention_pack_size_serial_(key_value_head_count, depth, segment_lengths, segment_count);
}

NK_API_COMPTIME void nk_attention_packed_shape_bf16_serial(void const *key_value_packed, nk_size_t *heads,
                                                           nk_size_t *depth, nk_size_t *segments) {
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
}

NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    return nk_attention_pack_size_serial_(key_value_head_count, depth, segment_lengths, segment_count);
}

NK_API_COMPTIME void nk_attention_packed_shape_e4m3_serial(void const *key_value_packed, nk_size_t *heads,
                                                           nk_size_t *depth, nk_size_t *segments) {
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/** Shared packing core: widen K and V rows to F32 planes `[key_value_head][position][channel]`. The
 *  header and offsets table are deterministic functions of the arguments, so concurrent packing
 *  tasks may rewrite them with identical bytes. */
NK_HELPER_INLINE void nk_attention_pack_serial_(                                                               //
    void const *keys, void const *values, nk_size_t element_bytes,                                             //
    nk_attention_load_f32_serial_t_ load_f32,                                                                  //
    nk_size_t key_value_head_count, nk_size_t depth,                                                           //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                                          //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t task_begin, nk_size_t task_end) {

    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 task_begin, 1, depth * sizeof(nk_f32_t));
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *payload_offsets_ro = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *payload_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (task_begin >= total_tasks) return;
    if (task_end > total_tasks) task_end = total_tasks;

    for (nk_size_t task_idx = task_begin; task_idx < task_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t *keys_plane = (nk_f32_t *)(payload_base + payload_offsets_ro[segment_idx]) +
                               key_value_head_idx * plane_floats;
        nk_f32_t *values_plane = keys_plane + key_value_head_count * plane_floats;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride_bytes +
                                   key_value_head_idx * depth * element_bytes;
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride_bytes +
                                     key_value_head_idx * depth * element_bytes;
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                keys_plane[position_idx * depth + channel_idx] = load_f32(keys_row + channel_idx * element_bytes);
                values_plane[position_idx * depth + channel_idx] = load_f32(values_row + channel_idx * element_bytes);
            }
        }
    }
}

#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NK_API_COMPTIME void nk_attention_pack_bf16_serial(                                                  //
    nk_bf16_t const *keys, nk_bf16_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t task_begin,
    nk_size_t task_end) {
    nk_attention_pack_serial_(keys, values, sizeof(nk_bf16_t), &nk_attention_load_bf16_serial_, key_value_head_count,
                              depth, segment_offsets, segment_lengths, segment_count, key_stride_bytes,
                              value_stride_bytes, key_value_packed, task_begin, task_end);
}

NK_API_COMPTIME void nk_attention_pack_e4m3_serial(                                                  //
    nk_e4m3_t const *keys, nk_e4m3_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t task_begin,
    nk_size_t task_end) {
    nk_attention_pack_serial_(keys, values, sizeof(nk_e4m3_t), &nk_attention_load_e4m3_serial_, key_value_head_count,
                              depth, segment_offsets, segment_lengths, segment_count, key_stride_bytes,
                              value_stride_bytes, key_value_packed, task_begin, task_end);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/**
 *  @brief Shared attention core: exact two-sweep softmax attention per (segment, head) task.
 *
 *  Per query row: sweep 1 finds the row maximum of score · scale₂; sweep 2 recomputes the scores,
 *  accumulating V rows weighted by 2^(score · scale₂ − max₂) straight into the output row, used as
 *  the accumulator, then normalizes by the accumulated sum. With no scratch, @p depth and
 *  @c position_count are unbounded. Recomputing scores costs ~1.5× the arithmetic of a buffered
 *  implementation and buys exact width-agnosticism with zero allocations. Row @c r reads only the
 *  keys that @c nk_attention_row_range_ admits for position r + @p diagonal_offset and @p window.
 */
NK_HELPER_INLINE void nk_attention_serial_(                                                                     //
    void const *queries, nk_size_t element_bytes, nk_attention_load_f32_serial_t_ load_f32,                     //
    void const *key_value_packed, nk_f32_t *output,                                                             //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start, nk_size_t task_count) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->heads != key_value_head_count) return;
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_(segment_count);
    nk_size_t const output_stride_floats = output_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)

    nk_size_t const task_end = nk_attention_task_end_(task_start, task_count, segment_count * head_count);
    for (nk_size_t task_idx = task_start; task_idx < task_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        if (row_count == 0) continue;
        nk_size_t const plane_floats = position_count * depth;
        nk_f32_t const *keys_plane = (nk_f32_t const *)(payload_base + payload_offsets[segment_idx]) +
                                     (head_idx / head_group_size) * plane_floats;
        nk_f32_t const *values_plane = keys_plane + key_value_head_count * plane_floats;

        for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
            char const *query_row = (char const *)queries +
                                    (query_offsets[segment_idx] + row_idx) * query_stride_bytes +
                                    head_idx * depth * element_bytes;
            nk_f32_t *output_row = output + (query_offsets[segment_idx] + row_idx) * output_stride_floats +
                                   head_idx * depth;
            nk_size_t key_begin, key_end;
            nk_attention_row_range_((nk_i64_t)row_idx + diagonal_offset, window, position_count, &key_begin, &key_end);

            nk_f32_t max2 = NK_F32_MIN;
            for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                nk_f32_t score = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    score += load_f32(query_row + channel_idx * element_bytes) *
                             keys_plane[position_idx * depth + channel_idx];
                nk_f32_t const scaled2 = score * scale2;
                if (scaled2 > max2) max2 = scaled2;
            }
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] = 0;
            nk_f32_t weights_sum = 0;
            for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                nk_f32_t score = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    score += load_f32(query_row + channel_idx * element_bytes) *
                             keys_plane[position_idx * depth + channel_idx];
                nk_f32_t const weight = nk_f32_exp2_serial_(score * scale2 - max2);
                weights_sum += weight;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    output_row[channel_idx] += weight * values_plane[position_idx * depth + channel_idx];
            }
            nk_f32_t const inverse_sum = weights_sum > 0 ? 1.0f / weights_sum : 0;
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] *= inverse_sum;
        }
    }
}

/** I8 attention core: exact I32 scores, U8-quantized weights, same row ranges as
 *  @c nk_attention_serial_. */
NK_HELPER_INLINE void nk_attention_packed_i8_serial_(                                                           //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,                                     //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start, nk_size_t task_count) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->heads != key_value_head_count) return;
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_(segment_count);
    nk_size_t const output_stride_floats = output_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)

    nk_size_t const task_end = nk_attention_task_end_(task_start, task_count, segment_count * head_count);
    for (nk_size_t task_idx = task_start; task_idx < task_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        if (row_count == 0) continue;
        nk_size_t const plane_bytes = position_count * depth;
        nk_i8_t const *keys_plane = (nk_i8_t const *)(payload_base + payload_offsets[segment_idx]) +
                                    (head_idx / head_group_size) * plane_bytes;
        nk_i8_t const *values_plane = keys_plane + key_value_head_count * plane_bytes;

        for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
            nk_i8_t const *query_row = (nk_i8_t const *)((char const *)queries +
                                                         (query_offsets[segment_idx] + row_idx) * query_stride_bytes) +
                                       head_idx * depth;
            nk_f32_t *output_row = output + (query_offsets[segment_idx] + row_idx) * output_stride_floats +
                                   head_idx * depth;
            nk_size_t key_begin, key_end;
            nk_attention_row_range_((nk_i64_t)row_idx + diagonal_offset, window, position_count, &key_begin, &key_end);

            nk_f32_t max2 = NK_F32_MIN; // scores are exact I32 integer dots; row max found before quantizing
            for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                nk_i32_t score = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    score += (nk_i32_t)query_row[channel_idx] *
                             (nk_i32_t)keys_plane[position_idx * depth + channel_idx];
                nk_f32_t const scaled2 = (nk_f32_t)score * scale2;
                if (scaled2 > max2) max2 = scaled2;
            }
            nk_f32_t sum_weights = 0;
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] = 0;
            for (nk_size_t position_idx = key_begin; position_idx < key_end; position_idx++) {
                nk_i32_t score = 0;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    score += (nk_i32_t)query_row[channel_idx] *
                             (nk_i32_t)keys_plane[position_idx * depth + channel_idx];
                nk_u32_t const weight_u8 = (nk_u32_t)(nk_f32_exp2_serial_((nk_f32_t)score * scale2 - max2) * 255.0f +
                                                      0.5f);
                if (weight_u8 == 0) continue;
                sum_weights += (nk_f32_t)weight_u8;
                nk_f32_t const weight = (nk_f32_t)weight_u8;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++)
                    output_row[channel_idx] += weight * (nk_f32_t)values_plane[position_idx * depth + channel_idx];
            }
            nk_f32_t const inverse_sum = sum_weights > 0 ? 1.0f / sum_weights : 0;
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) output_row[channel_idx] *= inverse_sum;
        }
    }
}

#if defined(__clang__)
#pragma clang attribute push(__attribute__((noinline)), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize", "no-ipa-cp-clone", "no-inline")
#endif

NK_API_COMPTIME void nk_attention_bidirectional_packed_bf16_serial(                                             //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,                                   //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t task_start, nk_size_t task_count) {
    nk_attention_serial_(queries, sizeof(nk_bf16_t), &nk_attention_load_bf16_serial_, key_value_packed, output,
                         head_count, key_value_head_count, depth, query_offsets, query_stride_bytes,
                         output_stride_bytes, scale, NK_I64_MAX / 2, NK_SIZE_MAX, task_start, task_count);
}

NK_API_COMPTIME void nk_attention_causal_packed_bf16_serial(                                                    //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,                                   //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start, nk_size_t task_count) {
    nk_attention_serial_(queries, sizeof(nk_bf16_t), &nk_attention_load_bf16_serial_, key_value_packed, output,
                         head_count, key_value_head_count, depth, query_offsets, query_stride_bytes,
                         output_stride_bytes, scale, diagonal_offset, window, task_start, task_count);
}

NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3_serial(                                             //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,                                   //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t task_start, nk_size_t task_count) {
    nk_attention_serial_(queries, sizeof(nk_e4m3_t), &nk_attention_load_e4m3_serial_, key_value_packed, output,
                         head_count, key_value_head_count, depth, query_offsets, query_stride_bytes,
                         output_stride_bytes, scale, NK_I64_MAX / 2, NK_SIZE_MAX, task_start, task_count);
}

NK_API_COMPTIME void nk_attention_causal_packed_e4m3_serial(                                                    //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,                                   //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start, nk_size_t task_count) {
    nk_attention_serial_(queries, sizeof(nk_e4m3_t), &nk_attention_load_e4m3_serial_, key_value_packed, output,
                         head_count, key_value_head_count, depth, query_offsets, query_stride_bytes,
                         output_stride_bytes, scale, diagonal_offset, window, task_start, task_count);
}
NK_API_COMPTIME nk_size_t nk_attention_pack_size_i8_serial(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    nk_size_t payload_bytes = 0; // raw I8 planes: scores stay exact in I32 integer arithmetic
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count * (nk_size_t)segment_lengths[segment_idx] * depth;
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_API_COMPTIME void nk_attention_packed_shape_i8_serial(void const *key_value_packed, nk_size_t *heads,
                                                         nk_size_t *depth, nk_size_t *segments) {
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
}

NK_API_COMPTIME void nk_attention_pack_i8_serial(                                                              //
    nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count, nk_size_t depth,               //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                                          //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t task_begin, nk_size_t task_end) {

    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 task_begin, 1, depth);
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *payload_offsets_ro = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *payload_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (task_begin >= total_tasks) return;
    if (task_end > total_tasks) task_end = total_tasks;

    for (nk_size_t task_idx = task_begin; task_idx < task_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * depth;
        nk_i8_t *keys_plane = (nk_i8_t *)(payload_base + payload_offsets_ro[segment_idx]) +
                              key_value_head_idx * plane_bytes;
        nk_i8_t *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride_bytes +
                                   key_value_head_idx * depth;
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride_bytes +
                                     key_value_head_idx * depth;
            for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx++) {
                keys_plane[position_idx * depth + channel_idx] = (nk_i8_t)keys_row[channel_idx];
                values_plane[position_idx * depth + channel_idx] = (nk_i8_t)values_row[channel_idx];
            }
        }
    }
}

NK_API_COMPTIME void nk_attention_bidirectional_packed_i8_serial(                                               //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,                                     //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t task_start, nk_size_t task_count) {
    nk_attention_packed_i8_serial_(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                   query_offsets, query_stride_bytes, output_stride_bytes, scale, NK_I64_MAX / 2,
                                   NK_SIZE_MAX, task_start, task_count);
}

NK_API_COMPTIME void nk_attention_causal_packed_i8_serial(                                                      //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,                                     //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start, nk_size_t task_count) {
    nk_attention_packed_i8_serial_(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                   query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                   window, task_start, task_count);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__GNUC__) && !defined(__clang__) && NK_TARGET_ARM64_
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_ATTENTION_SERIAL_H
