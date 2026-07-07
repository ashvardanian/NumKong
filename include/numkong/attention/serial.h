/**
 *  @brief Serial (SIMD-free) ragged attention baseline.
 *  @file include/numkong/attention/serial.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  Width-agnostic reference implementation of the ragged scaled-dot-product attention
 *  family: any `head_dim ≥ 1`, any segment lengths, GQA/MQA, the same base-2 softmax
 *  formulation and the same `(first_task, task_count)` windows over the segment × head
 *  grid as the SIMD backends.
 *
 *  @section attention_serial_roles Roles in the Family
 *
 *  Serves three roles:
 *
 *  1. Ground truth for every SIMD backend's conformance tests.
 *  2. Final runtime-dispatch fallback on CPUs without any compiled SIMD target.
 *  3. Fallback for shapes outside a SIMD backend's fast-path envelope (`head_dim > 256`),
 *     invoked from those backends' own entry points so pack and attention always agree
 *     on the packed-buffer format.
 *
 *  @section attention_serial_layout Packed Layout
 *
 *  This file also owns the family-shared packed-KV header and directory layout:
 *  `[64 B header][directory: u64 tile_offsets[segments+1] + u32 segment_lengths[segments],
 *  64-byte padded][per-backend payload]`.  The payload is backend-opaque; serial stores
 *  K and V as plain F32 row-major planes `[kv_head][token][channel]` per segment, so both
 *  input dtypes (BF16, E4M3) share one compute path after per-element conversion at pack.
 *
 *  Like the rest of the serial tier, no libm: the base-2 exponent uses the same degree-4
 *  polynomial and the same denormal-avoiding clamp as the AVX-512 helper, so serial and
 *  vector paths agree to polynomial precision (@see nk_f32_exp2_serial_).
 */
#ifndef NK_ATTENTION_SERIAL_H
#define NK_ATTENTION_SERIAL_H

#include "numkong/types.h"
#include "numkong/scalar/serial.h" // `nk_f32_exp2_serial_`, `NK_LOG2E_`
#include "numkong/cast/serial.h"   // `nk_bf16_to_f32_serial`, `nk_e4m3_to_f32_serial`

#if defined(__cplusplus)
extern "C" {
#endif

/**
 *  @brief Packed ragged KV cache header (64 bytes), shared by all attention backends.
 *  Followed by the segment directory; the payload beyond it is backend-specific.
 */
typedef struct {
    nk_u32_t num_kv_heads;  ///< Number of K/V heads (≤ query heads for GQA)
    nk_u32_t head_dim;      ///< Head dimension the buffer was packed for
    nk_u32_t segment_count; ///< Number of independent segments packed
    nk_u32_t reserved[13];  ///< Zeroed; pads the header to 64 bytes
} nk_attention_packed_header_t;

/** @brief Directory size in bytes: u64 payload offsets [count+1] + u32 lengths [count], 64-byte padded. */
NK_INTERNAL nk_size_t nk_attention_packed_directory_size_(nk_size_t segment_count) {
    return nk_size_round_up_to_multiple_((segment_count + 1) * sizeof(nk_u64_t) + segment_count * sizeof(nk_u32_t), 64);
}

/**
 *  @brief Writes the family-shared packed-KV header and segment directory.
 *  Only the window covering task 0 writes — later windows require the directory present
 *  (the race-free parallel-pack contract).  Per-segment payload bytes follow
 *  `2 · num_kv_heads · round_up(len, seq_multiple) · unit_bytes`, which covers every backend.
 */
NK_INTERNAL void nk_attention_pack_directory_(void *kv_packed, nk_size_t num_kv_heads, nk_size_t head_dim,
                                              nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                              nk_size_t first_task, nk_size_t seq_multiple, nk_size_t unit_bytes) {
    if (first_task != 0) return;
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)kv_packed;
    for (nk_size_t i = 0; i < sizeof(*header) / sizeof(nk_u32_t); i++) ((nk_u32_t *)header)[i] = 0;
    header->num_kv_heads = (nk_u32_t)num_kv_heads;
    header->head_dim = (nk_u32_t)head_dim;
    header->segment_count = (nk_u32_t)segment_count;
    nk_u64_t *payload_offsets = (nk_u64_t *)((char *)kv_packed + sizeof(*header));
    nk_u32_t *lengths_copy = (nk_u32_t *)(payload_offsets + segment_count + 1);
    nk_u64_t running = 0;
    for (nk_size_t s = 0; s < segment_count; s++) {
        payload_offsets[s] = running;
        lengths_copy[s] = segment_lengths[s];
        running += 2 * num_kv_heads * (nk_u64_t)nk_size_round_up_to_multiple_(segment_lengths[s], seq_multiple) *
                   unit_bytes;
    }
    payload_offsets[segment_count] = running;
}

NK_PUBLIC nk_size_t nk_attention_packed_segments(void const *kv_packed) {
    return ((nk_attention_packed_header_t const *)kv_packed)->segment_count;
}

NK_PUBLIC nk_size_t nk_attention_packed_head_dim(void const *kv_packed) {
    return ((nk_attention_packed_header_t const *)kv_packed)->head_dim;
}

NK_PUBLIC nk_size_t nk_attention_packed_heads(void const *kv_packed) {
    return ((nk_attention_packed_header_t const *)kv_packed)->num_kv_heads;
}

/** @brief Per-element widening converter, `maxsim/serial.h`-style dtype abstraction. */
typedef nk_f32_t (*nk_attention_load_f32_serial_t_)(void const *element);

NK_INTERNAL nk_f32_t nk_attention_load_bf16_serial_(void const *element) {
    nk_f32_t result;
    nk_bf16_to_f32_serial((nk_bf16_t const *)element, &result);
    return result;
}

NK_INTERNAL nk_f32_t nk_attention_load_e4m3_serial_(void const *element) {
    nk_f32_t result;
    nk_e4m3_to_f32_serial((nk_e4m3_t const *)element, &result);
    return result;
}

NK_INTERNAL nk_size_t nk_attention_packed_size_serial_(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                       nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    nk_size_t payload_bytes = 0;
    for (nk_size_t s = 0; s < segment_count; s++)
        payload_bytes += 2 * num_kv_heads * (nk_size_t)segment_lengths[s] * head_dim * sizeof(nk_f32_t); // K + V
    return sizeof(nk_attention_packed_header_t) + nk_attention_packed_directory_size_(segment_count) + payload_bytes;
}

NK_PUBLIC nk_size_t nk_attention_packed_size_bf16_serial(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    return nk_attention_packed_size_serial_(num_kv_heads, head_dim, segment_lengths, segment_count);
}

NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3_serial(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                         nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    return nk_attention_packed_size_serial_(num_kv_heads, head_dim, segment_lengths, segment_count);
}

/**
 *  @brief Shared packing core: widen K and V rows to F32 planes `[kv_head][token][channel]`.
 *  The header and directory are deterministic functions of the arguments, so concurrent
 *  packing tasks may rewrite them with identical bytes.
 */
NK_INTERNAL void nk_attention_pack_serial_(                                           //
    void const *k, void const *v, nk_size_t element_bytes,                            //
    nk_attention_load_f32_serial_t_ load_f32,                                         //
    nk_size_t num_kv_heads, nk_size_t head_dim,                                       //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                 //
    nk_size_t segment_count, nk_size_t k_stride, nk_size_t v_stride, void *kv_packed, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_attention_pack_directory_(kv_packed, num_kv_heads, head_dim, segment_lengths, segment_count, first_task, 1,
                                 head_dim * sizeof(nk_f32_t));
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)kv_packed;
    nk_u64_t const *payload_offsets_ro = (nk_u64_t const *)((char *)kv_packed + sizeof(*header));
    char *payload_base = (char *)kv_packed + sizeof(*header) + nk_attention_packed_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * num_kv_heads;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    for (nk_size_t task = first_task; task < first_task + task_count; task++) {
        nk_size_t const segment = task / num_kv_heads, kv_head = task % num_kv_heads;
        nk_size_t const seq_len = segment_lengths[segment];
        if (seq_len == 0) continue;
        nk_size_t const token_first = segment_offsets[segment];
        nk_size_t const plane_floats = seq_len * head_dim;
        nk_f32_t *k_plane = (nk_f32_t *)(payload_base + payload_offsets_ro[segment]) + kv_head * plane_floats;
        nk_f32_t *v_plane = k_plane + num_kv_heads * plane_floats;
        for (nk_size_t token = 0; token < seq_len; token++) {
            char const *k_row = (char const *)k + (token_first + token) * k_stride + kv_head * head_dim * element_bytes;
            char const *v_row = (char const *)v + (token_first + token) * v_stride + kv_head * head_dim * element_bytes;
            for (nk_size_t channel = 0; channel < head_dim; channel++) {
                k_plane[token * head_dim + channel] = load_f32(k_row + channel * element_bytes);
                v_plane[token * head_dim + channel] = load_f32(v_row + channel * element_bytes);
            }
        }
    }
}

NK_PUBLIC void nk_attention_pack_bf16_serial(                                           //
    nk_bf16_t const *k, nk_bf16_t const *v, nk_size_t num_kv_heads, nk_size_t head_dim, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t k_stride,
    nk_size_t v_stride, void *kv_packed, nk_size_t first_task, nk_size_t task_count) {
    nk_attention_pack_serial_(k, v, sizeof(nk_bf16_t), &nk_attention_load_bf16_serial_, num_kv_heads, head_dim,
                              segment_offsets, segment_lengths, segment_count, k_stride, v_stride, kv_packed,
                              first_task, task_count);
}

NK_PUBLIC void nk_attention_pack_e4m3_serial(                                           //
    nk_e4m3_t const *k, nk_e4m3_t const *v, nk_size_t num_kv_heads, nk_size_t head_dim, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t k_stride,
    nk_size_t v_stride, void *kv_packed, nk_size_t first_task, nk_size_t task_count) {
    nk_attention_pack_serial_(k, v, sizeof(nk_e4m3_t), &nk_attention_load_e4m3_serial_, num_kv_heads, head_dim,
                              segment_offsets, segment_lengths, segment_count, k_stride, v_stride, kv_packed,
                              first_task, task_count);
}

/**
 *  @brief Shared attention core: exact two-sweep softmax attention per (segment, head) task.
 *
 *  Per query row: sweep 1 finds the row maximum of `score · scale₂`; sweep 2 recomputes the
 *  scores, accumulating `2^(score·scale₂ − max₂)`-weighted V rows straight into the output
 *  row (used as the accumulator — no scratch, so `head_dim` and `kv_len` are unbounded),
 *  then normalizes by the accumulated sum.  Recomputing scores costs ~1.5× the arithmetic
 *  of a buffered implementation and buys exact width-agnosticism with zero allocations.
 */
NK_INTERNAL void nk_attention_serial_(                                                     //
    void const *q, nk_size_t element_bytes, nk_attention_load_f32_serial_t_ load_f32,      //
    void const *kv_packed, nk_f32_t *output,                                               //
    nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,                       //
    nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)kv_packed;
    if (header->head_dim != head_dim || header->num_kv_heads != num_kv_heads) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)kv_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)kv_packed + sizeof(*header) +
                               nk_attention_packed_directory_size_(segment_count);
    nk_size_t const o_stride_floats = o_stride / sizeof(nk_f32_t);
    nk_size_t const gqa_ratio = num_heads / num_kv_heads;
    nk_f32_t const scale2 = scale * NK_LOG2E_; // softmax(x) = softmax₂(x·log₂e)

    nk_size_t const total_tasks = segment_count * num_heads;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    for (nk_size_t task = first_task; task < first_task + task_count; task++) {
        nk_size_t const segment = task / num_heads, head = task % num_heads;
        nk_size_t const kv_len = segment_lengths[segment];
        nk_size_t const query_count = query_offsets[segment + 1] - query_offsets[segment];
        if (kv_len == 0 || query_count == 0) continue;
        nk_size_t const plane_floats = kv_len * head_dim;
        nk_f32_t const *k_plane = (nk_f32_t const *)(payload_base + payload_offsets[segment]) +
                                  (head / gqa_ratio) * plane_floats;
        nk_f32_t const *v_plane = k_plane + num_kv_heads * plane_floats;

        for (nk_size_t row = 0; row < query_count; row++) {
            char const *q_row = (char const *)q + (query_offsets[segment] + row) * q_stride +
                                head * head_dim * element_bytes;
            nk_f32_t *o_row = output + (query_offsets[segment] + row) * o_stride_floats + head * head_dim;

            nk_f32_t max2 = NK_F32_MIN;
            for (nk_size_t position = 0; position < kv_len; position++) {
                nk_f32_t score = 0;
                for (nk_size_t channel = 0; channel < head_dim; channel++)
                    score += load_f32(q_row + channel * element_bytes) * k_plane[position * head_dim + channel];
                nk_f32_t const scaled2 = score * scale2;
                if (scaled2 > max2) max2 = scaled2;
            }
            for (nk_size_t channel = 0; channel < head_dim; channel++) o_row[channel] = 0;
            nk_f32_t weights_sum = 0;
            for (nk_size_t position = 0; position < kv_len; position++) {
                nk_f32_t score = 0;
                for (nk_size_t channel = 0; channel < head_dim; channel++)
                    score += load_f32(q_row + channel * element_bytes) * k_plane[position * head_dim + channel];
                nk_f32_t const weight = nk_f32_exp2_serial_(score * scale2 - max2);
                weights_sum += weight;
                for (nk_size_t channel = 0; channel < head_dim; channel++)
                    o_row[channel] += weight * v_plane[position * head_dim + channel];
            }
            nk_f32_t const inverse_sum = 1.0f / weights_sum;
            for (nk_size_t channel = 0; channel < head_dim; channel++) o_row[channel] *= inverse_sum;
        }
    }
}

NK_PUBLIC void nk_attention_packed_bf16_serial(                                            //
    nk_bf16_t const *q, void const *kv_packed, nk_f32_t *output,                           //
    nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,                       //
    nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    nk_attention_serial_(q, sizeof(nk_bf16_t), &nk_attention_load_bf16_serial_, kv_packed, output, num_heads,
                         num_kv_heads, head_dim, query_offsets, q_stride, o_stride, scale, first_task, task_count);
}

NK_PUBLIC void nk_attention_packed_e4m3_serial(                                            //
    nk_e4m3_t const *q, void const *kv_packed, nk_f32_t *output,                           //
    nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,                       //
    nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    nk_attention_serial_(q, sizeof(nk_e4m3_t), &nk_attention_load_e4m3_serial_, kv_packed, output, num_heads,
                         num_kv_heads, head_dim, query_offsets, q_stride, o_stride, scale, first_task, task_count);
}

NK_PUBLIC nk_size_t nk_attention_packed_size_i8_serial(nk_size_t num_kv_heads, nk_size_t head_dim,
                                                       nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    nk_size_t payload_bytes = 0; // raw I8 planes: scores stay exact in I32 integer arithmetic
    for (nk_size_t s = 0; s < segment_count; s++)
        payload_bytes += 2 * num_kv_heads * (nk_size_t)segment_lengths[s] * head_dim;
    return sizeof(nk_attention_packed_header_t) + nk_attention_packed_directory_size_(segment_count) + payload_bytes;
}

NK_PUBLIC void nk_attention_pack_i8_serial(                                           //
    nk_i8_t const *k, nk_i8_t const *v, nk_size_t num_kv_heads, nk_size_t head_dim,   //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                 //
    nk_size_t segment_count, nk_size_t k_stride, nk_size_t v_stride, void *kv_packed, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_attention_pack_directory_(kv_packed, num_kv_heads, head_dim, segment_lengths, segment_count, first_task, 1,
                                 head_dim);
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)kv_packed;
    nk_u64_t const *payload_offsets_ro = (nk_u64_t const *)((char *)kv_packed + sizeof(*header));
    char *payload_base = (char *)kv_packed + sizeof(*header) + nk_attention_packed_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * num_kv_heads;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    for (nk_size_t task = first_task; task < first_task + task_count; task++) {
        nk_size_t const segment = task / num_kv_heads, kv_head = task % num_kv_heads;
        nk_size_t const seq_len = segment_lengths[segment];
        if (seq_len == 0) continue;
        nk_size_t const token_first = segment_offsets[segment];
        nk_size_t const plane_bytes = seq_len * head_dim;
        nk_i8_t *k_plane = (nk_i8_t *)(payload_base + payload_offsets_ro[segment]) + kv_head * plane_bytes;
        nk_i8_t *v_plane = k_plane + num_kv_heads * plane_bytes;
        for (nk_size_t token = 0; token < seq_len; token++) {
            char const *k_row = (char const *)k + (token_first + token) * k_stride + kv_head * head_dim;
            char const *v_row = (char const *)v + (token_first + token) * v_stride + kv_head * head_dim;
            for (nk_size_t channel = 0; channel < head_dim; channel++) {
                k_plane[token * head_dim + channel] = (nk_i8_t)k_row[channel];
                v_plane[token * head_dim + channel] = (nk_i8_t)v_row[channel];
            }
        }
    }
}

NK_PUBLIC void nk_attention_packed_i8_serial(                                              //
    nk_i8_t const *q, void const *kv_packed, nk_f32_t *output,                             //
    nk_size_t num_heads, nk_size_t num_kv_heads, nk_size_t head_dim,                       //
    nk_u32_t const *query_offsets, nk_size_t q_stride, nk_size_t o_stride, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)kv_packed;
    if (header->head_dim != head_dim || header->num_kv_heads != num_kv_heads) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)kv_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)kv_packed + sizeof(*header) +
                               nk_attention_packed_directory_size_(segment_count);
    nk_size_t const o_stride_floats = o_stride / sizeof(nk_f32_t);
    nk_size_t const gqa_ratio = num_heads / num_kv_heads;
    nk_f32_t const scale2 = scale * NK_LOG2E_; // softmax(x) = softmax₂(x·log₂e)

    nk_size_t const total_tasks = segment_count * num_heads;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    for (nk_size_t task = first_task; task < first_task + task_count; task++) {
        nk_size_t const segment = task / num_heads, head = task % num_heads;
        nk_size_t const kv_len = segment_lengths[segment];
        nk_size_t const query_count = query_offsets[segment + 1] - query_offsets[segment];
        if (kv_len == 0 || query_count == 0) continue;
        nk_size_t const plane_bytes = kv_len * head_dim;
        nk_i8_t const *k_plane = (nk_i8_t const *)(payload_base + payload_offsets[segment]) +
                                 (head / gqa_ratio) * plane_bytes;
        nk_i8_t const *v_plane = k_plane + num_kv_heads * plane_bytes;

        for (nk_size_t row = 0; row < query_count; row++) {
            nk_i8_t const *q_row = (nk_i8_t const *)((char const *)q + (query_offsets[segment] + row) * q_stride) +
                                   head * head_dim;
            nk_f32_t *o_row = output + (query_offsets[segment] + row) * o_stride_floats + head * head_dim;

            nk_f32_t max2 = NK_F32_MIN; // scores are exact I32 integer dots; row max found before quantizing
            for (nk_size_t position = 0; position < kv_len; position++) {
                nk_i32_t score = 0;
                for (nk_size_t channel = 0; channel < head_dim; channel++)
                    score += (nk_i32_t)q_row[channel] * (nk_i32_t)k_plane[position * head_dim + channel];
                nk_f32_t const scaled2 = (nk_f32_t)score * scale2;
                if (scaled2 > max2) max2 = scaled2;
            }
            // Weights quantize to U8 as `round(255 · 2^(s₂ − m₂))`; the max-scoring position
            // always lands on 255, so the sum below can never be zero.  Normalizing by the
            // sum of the quantized weights makes the 255 cancel — no descale constant remains.

            nk_f32_t sum_weights = 0;
            for (nk_size_t channel = 0; channel < head_dim; channel++) o_row[channel] = 0;
            for (nk_size_t position = 0; position < kv_len; position++) {
                nk_i32_t score = 0;
                for (nk_size_t channel = 0; channel < head_dim; channel++)
                    score += (nk_i32_t)q_row[channel] * (nk_i32_t)k_plane[position * head_dim + channel];
                nk_u32_t const weight_u8 = (nk_u32_t)(nk_f32_exp2_serial_((nk_f32_t)score * scale2 - max2) * 255.0f +
                                                      0.5f);
                if (weight_u8 == 0) continue;
                sum_weights += (nk_f32_t)weight_u8;
                nk_f32_t const weight = (nk_f32_t)weight_u8;
                for (nk_size_t channel = 0; channel < head_dim; channel++)
                    o_row[channel] += weight * (nk_f32_t)v_plane[position * head_dim + channel];
            }
            nk_f32_t const inverse_sum = 1.0f / sum_weights;
            for (nk_size_t channel = 0; channel < head_dim; channel++) o_row[channel] *= inverse_sum;
        }
    }
}

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_ATTENTION_SERIAL_H
