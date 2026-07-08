/**
 *  @brief Ragged attention for AVX2 Haswell generation CPUs.
 *  @file include/numkong/attention/haswell.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  Compatibility backend for AVX2/FMA machines. Storage follows the `dots/haswell.h`
 *  conventions exactly: both BF16 and E4M3 stay in their source encoding at rest —
 *  packing is a raw strided-row copy — and every value widens to F32 on the fly inside
 *  the compute loops, 8-wide on the FMA ports. That keeps the packed KV cache at 2 or
 *  even 1 byte per scalar, trading a couple of unpacking ops per FMA for halved-to-
 *  quartered streaming traffic, the same call the GEMM family made for this ISA.
 *
 *  The panel structure matches the family design — per query row, KV is swept in panels
 *  with an exact online correction and the family's base-2 degree-4 softmax polynomial;
 *  scores keep four KV rows in flight. Packed payload per segment: K planes then V
 *  planes, `[key_value_head][position][channel]` in the source dtype with channels zero-padded to
 *  a multiple of 8. `depth > 256` routes to the width-agnostic serial tier.
 */
#ifndef NK_ATTENTION_HASWELL_H
#define NK_ATTENTION_HASWELL_H

#if NK_TARGET_X8664_
#if NK_TARGET_HASWELL

#include "numkong/attention/serial.h" // shared packed-KV header/offsets, width-agnostic fallback
#include "numkong/cast/haswell.h"     // widening helpers like `nk_bf16x8_to_f32x8_haswell_`
#include "numkong/reduce/haswell.h"   // `nk_reduce_add_f32x8_haswell_`, `nk_reduce_max_f32x8_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

enum {
    /** KV panel width in positions; the F32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_haswell_k_ = 512,
    /** Widest head this backend handles in registers; larger heads route to the serial tier. */
    nk_attention_max_depth_haswell_k_ = 256,
};

/** @brief Fast vectorized 2^x: exact range reduction + the family's shared degree-4 polynomial. */
NK_INTERNAL __m256 nk_attention_exp2_f32x8_haswell_(__m256 x_f32x8) {
    x_f32x8 = _mm256_max_ps(_mm256_min_ps(x_f32x8, _mm256_set1_ps(127.0f)), _mm256_set1_ps(-125.0f));
    __m256 n_f32x8 = _mm256_round_ps(x_f32x8, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r_f32x8 = _mm256_sub_ps(x_f32x8, n_f32x8);
    __m256 p_f32x8 = _mm256_set1_ps(9.61812910e-3f);
    p_f32x8 = _mm256_fmadd_ps(p_f32x8, r_f32x8, _mm256_set1_ps(5.55041087e-2f));
    p_f32x8 = _mm256_fmadd_ps(p_f32x8, r_f32x8, _mm256_set1_ps(2.40226507e-1f));
    p_f32x8 = _mm256_fmadd_ps(p_f32x8, r_f32x8, _mm256_set1_ps(6.93147181e-1f));
    p_f32x8 = _mm256_fmadd_ps(p_f32x8, r_f32x8, _mm256_set1_ps(1.0f));
    __m256i n_i32x8 = _mm256_cvtps_epi32(n_f32x8);
    n_i32x8 = _mm256_slli_epi32(_mm256_add_epi32(n_i32x8, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p_f32x8, _mm256_castsi256_ps(n_i32x8));
}

/** @brief Widens 8 raw plane scalars (BF16 or E4M3 at rest) to F32 inside the hot loops. */
typedef __m256 (*nk_attention_load_haswell_t_)(void const *plane_chunk);

NK_INTERNAL __m256 nk_attention_load_bf16x8_haswell_(void const *plane_chunk) {
    nk_b256_vec_t widened;
    nk_load_bf16x8_to_f32x8_haswell_(plane_chunk, &widened);
    return widened.ymm_ps;
}

NK_INTERNAL __m256 nk_attention_load_e4m3x8_haswell_(void const *plane_chunk) {
    nk_b256_vec_t widened;
    nk_load_e4m3x8_to_f32x8_haswell_(plane_chunk, &widened);
    return widened.ymm_ps;
}

/** @brief Widens `count` raw query elements to F32 into `destination`, zero-filling to `padded`. */
typedef void (*nk_attention_widen_haswell_t_)(void const *source, nk_f32_t *destination, nk_size_t count,
                                              nk_size_t padded);

NK_INTERNAL void nk_attention_widen_bf16_haswell_(void const *source, nk_f32_t *destination, nk_size_t count,
                                                  nk_size_t padded) {
    nk_size_t channel_idx = 0;
    nk_b256_vec_t widened;
    for (; channel_idx + 8 <= count; channel_idx += 8) {
        nk_load_bf16x8_to_f32x8_haswell_((nk_bf16_t const *)source + channel_idx, &widened);
        _mm256_storeu_ps(destination + channel_idx, widened.ymm_ps);
    }
    if (channel_idx < count) {
        nk_partial_load_bf16x8_to_f32x8_haswell_((nk_bf16_t const *)source + channel_idx, &widened,
                                                 count - channel_idx);
        _mm256_storeu_ps(destination + channel_idx, widened.ymm_ps);
        channel_idx += 8;
    }
    for (; channel_idx < padded; channel_idx += 8) _mm256_storeu_ps(destination + channel_idx, _mm256_setzero_ps());
}

NK_INTERNAL void nk_attention_widen_e4m3_haswell_(void const *source, nk_f32_t *destination, nk_size_t count,
                                                  nk_size_t padded) {
    nk_size_t channel_idx = 0;
    nk_b256_vec_t widened;
    for (; channel_idx + 8 <= count; channel_idx += 8) {
        nk_load_e4m3x8_to_f32x8_haswell_((nk_e4m3_t const *)source + channel_idx, &widened);
        _mm256_storeu_ps(destination + channel_idx, widened.ymm_ps);
    }
    if (channel_idx < count) {
        nk_partial_load_e4m3x8_to_f32x8_haswell_((nk_e4m3_t const *)source + channel_idx, &widened,
                                                 count - channel_idx);
        _mm256_storeu_ps(destination + channel_idx, widened.ymm_ps);
        channel_idx += 8;
    }
    for (; channel_idx < padded; channel_idx += 8) _mm256_storeu_ps(destination + channel_idx, _mm256_setzero_ps());
}

NK_INTERNAL nk_size_t nk_attention_packed_size_haswell_(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count,
                                                        nk_size_t element_bytes) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t payload_bytes = 0; // planes keep the source encoding, like the dots family
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count * (nk_size_t)segment_lengths[segment_idx] * depth_padded *
                         element_bytes;
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_PUBLIC nk_size_t nk_attention_packed_size_bf16_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_haswell_k_)
        return nk_attention_packed_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_haswell_(key_value_head_count, depth, segment_lengths, segment_count,
                                             sizeof(nk_bf16_t));
}

NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_haswell_k_)
        return nk_attention_packed_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_haswell_(key_value_head_count, depth, segment_lengths, segment_count,
                                             sizeof(nk_e4m3_t));
}

/** @brief Raw strided-row repack: source encoding is preserved, tails zero-padded. */
NK_INTERNAL void nk_attention_pack_haswell_(                                                                   //
    void const *keys, void const *values, nk_size_t element_bytes,                                             //
    nk_size_t key_value_head_count, nk_size_t depth,                                                           //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                                          //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t first_task, nk_size_t task_count) {

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const row_bytes = depth * element_bytes;
    nk_size_t const padded_row_bytes = depth_padded * element_bytes;
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, 1, padded_row_bytes);
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *payload_offsets_ro = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *payload_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * padded_row_bytes;
        char *keys_plane = payload_base + payload_offsets_ro[segment_idx] + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride_bytes +
                                   key_value_head_idx * row_bytes;
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride_bytes +
                                     key_value_head_idx * row_bytes;
            char *keys_destination = keys_plane + position_idx * padded_row_bytes;
            char *values_destination = values_plane + position_idx * padded_row_bytes;
            for (nk_size_t byte = 0; byte < row_bytes; byte++) keys_destination[byte] = keys_row[byte];
            for (nk_size_t byte = row_bytes; byte < padded_row_bytes; byte++) keys_destination[byte] = 0;
            for (nk_size_t byte = 0; byte < row_bytes; byte++) values_destination[byte] = values_row[byte];
            for (nk_size_t byte = row_bytes; byte < padded_row_bytes; byte++) values_destination[byte] = 0;
        }
    }
}

NK_PUBLIC void nk_attention_pack_bf16_haswell(                                                       //
    nk_bf16_t const *keys, nk_bf16_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t first_task,
    nk_size_t task_count) {
    if (depth > nk_attention_max_depth_haswell_k_) {
        nk_attention_pack_bf16_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_haswell_(keys, values, sizeof(nk_bf16_t), key_value_head_count, depth, segment_offsets,
                               segment_lengths, segment_count, key_stride_bytes, value_stride_bytes, key_value_packed,
                               first_task, task_count);
}

NK_PUBLIC void nk_attention_pack_e4m3_haswell(                                                       //
    nk_e4m3_t const *keys, nk_e4m3_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t first_task,
    nk_size_t task_count) {
    if (depth > nk_attention_max_depth_haswell_k_) {
        nk_attention_pack_e4m3_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_haswell_(keys, values, sizeof(nk_e4m3_t), key_value_head_count, depth, segment_offsets,
                               segment_lengths, segment_count, key_stride_bytes, value_stride_bytes, key_value_packed,
                               first_task, task_count);
}

/**
 *  @brief Shared attention core over raw-encoded planes: per query row, panel-flash with
 *         an exact online correction; scores keep four KV rows in flight, widening in-loop.
 */
NK_INTERNAL void nk_attention_packed_haswell_(                                                                  //
    void const *queries, nk_size_t element_bytes, nk_attention_widen_haswell_t_ widen,                          //
    nk_attention_load_haswell_t_ load,                                                                          //
    void const *key_value_packed, nk_f32_t *output,                                                             //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
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
    nk_size_t const panel_width = nk_attention_panel_haswell_k_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_f32_t query_row[nk_attention_max_depth_haswell_k_];
    NK_ALIGN64 nk_f32_t output_row[nk_attention_max_depth_haswell_k_];
    NK_ALIGN64 nk_f32_t scores[nk_attention_panel_haswell_k_];

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
            widen((char const *)queries + (query_offsets[segment_idx] + row_idx) * query_stride_bytes +
                      head_idx * depth * element_bytes,
                  query_row, depth, depth_padded);
            for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
                _mm256_store_ps(output_row + channel_idx, _mm256_setzero_ps());
            nk_f32_t running_max2 = NK_F32_MIN, running_sum = 0;

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_length = (panel_start + panel_width <= position_count)
                                                   ? panel_width
                                                   : (position_count - panel_start);
                // Scores: four KV rows in flight, raw plane scalars widened in-loop.

                nk_size_t position_idx = 0;
                for (; position_idx + 4 <= panel_length; position_idx += 4) {
                    char const *keys_row = keys_plane + (panel_start + position_idx) * plane_row_bytes;
                    __m256 accumulator0_f32x8 = _mm256_setzero_ps(), accumulator1_f32x8 = _mm256_setzero_ps();
                    __m256 accumulator2_f32x8 = _mm256_setzero_ps(), accumulator3_f32x8 = _mm256_setzero_ps();
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
                        __m256 const query_f32x8 = _mm256_load_ps(query_row + channel_idx);
                        nk_size_t const chunk_bytes = channel_idx * element_bytes;
                        accumulator0_f32x8 = _mm256_fmadd_ps(query_f32x8, load(keys_row + chunk_bytes),
                                                             accumulator0_f32x8);
                        accumulator1_f32x8 = _mm256_fmadd_ps(
                            query_f32x8, load(keys_row + plane_row_bytes + chunk_bytes), accumulator1_f32x8);
                        accumulator2_f32x8 = _mm256_fmadd_ps(
                            query_f32x8, load(keys_row + 2 * plane_row_bytes + chunk_bytes), accumulator2_f32x8);
                        accumulator3_f32x8 = _mm256_fmadd_ps(
                            query_f32x8, load(keys_row + 3 * plane_row_bytes + chunk_bytes), accumulator3_f32x8);
                    }
                    scores[position_idx + 0] = nk_reduce_add_f32x8_haswell_(accumulator0_f32x8);
                    scores[position_idx + 1] = nk_reduce_add_f32x8_haswell_(accumulator1_f32x8);
                    scores[position_idx + 2] = nk_reduce_add_f32x8_haswell_(accumulator2_f32x8);
                    scores[position_idx + 3] = nk_reduce_add_f32x8_haswell_(accumulator3_f32x8);
                }
                for (; position_idx < panel_length; position_idx++) {
                    char const *keys_row = keys_plane + (panel_start + position_idx) * plane_row_bytes;
                    __m256 accumulator_f32x8 = _mm256_setzero_ps();
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
                        accumulator_f32x8 = _mm256_fmadd_ps(_mm256_load_ps(query_row + channel_idx),
                                                            load(keys_row + channel_idx * element_bytes),
                                                            accumulator_f32x8);
                    scores[position_idx] = nk_reduce_add_f32x8_haswell_(accumulator_f32x8);
                }

                // Panel max, online correction, exp2 in place (scalar tail via the serial exp2).

                __m256 max_f32x8 = _mm256_set1_ps(NK_F32_MIN);
                position_idx = 0;
                for (; position_idx + 8 <= panel_length; position_idx += 8)
                    max_f32x8 = _mm256_max_ps(max_f32x8, _mm256_loadu_ps(scores + position_idx));
                nk_f32_t panel_max2 = nk_reduce_max_f32x8_haswell_(max_f32x8) * scale2;
                for (; position_idx < panel_length; position_idx++) {
                    nk_f32_t const scaled2 = scores[position_idx] * scale2;
                    if (scaled2 > panel_max2) panel_max2 = scaled2;
                }
                nk_f32_t const new_max2 = running_max2 > panel_max2 ? running_max2 : panel_max2;
                nk_f32_t const correction = _mm256_cvtss_f32(
                    nk_attention_exp2_f32x8_haswell_(_mm256_set1_ps(running_max2 - new_max2)));
                running_max2 = new_max2;

                __m256 const scale2_f32x8 = _mm256_set1_ps(scale2);
                __m256 const max2_f32x8 = _mm256_set1_ps(new_max2);
                __m256 sum_f32x8 = _mm256_setzero_ps();
                for (position_idx = 0; position_idx + 8 <= panel_length; position_idx += 8) {
                    __m256 weights_f32x8 = nk_attention_exp2_f32x8_haswell_(
                        _mm256_fmsub_ps(_mm256_loadu_ps(scores + position_idx), scale2_f32x8, max2_f32x8));
                    sum_f32x8 = _mm256_add_ps(sum_f32x8, weights_f32x8);
                    _mm256_storeu_ps(scores + position_idx, weights_f32x8);
                }
                if (position_idx < panel_length) { // masked tail keeps the vector rounding mode end-to-end
                    __m256i const lane_index_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                    __m256i const tail_mask_i32x8 = _mm256_cmpgt_epi32(
                        _mm256_set1_epi32((int)(panel_length - position_idx)), lane_index_i32x8);
                    __m256 weights_f32x8 = nk_attention_exp2_f32x8_haswell_(_mm256_fmsub_ps(
                        _mm256_maskload_ps(scores + position_idx, tail_mask_i32x8), scale2_f32x8, max2_f32x8));
                    weights_f32x8 = _mm256_and_ps(weights_f32x8, _mm256_castsi256_ps(tail_mask_i32x8));
                    sum_f32x8 = _mm256_add_ps(sum_f32x8, weights_f32x8);
                    _mm256_maskstore_ps(scores + position_idx, tail_mask_i32x8, weights_f32x8);
                }
                running_sum = running_sum * correction + nk_reduce_add_f32x8_haswell_(sum_f32x8);

                // O = O · correction + Σ weight · widened V-row over the panel.

                __m256 const correction_f32x8 = _mm256_set1_ps(correction);
                for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
                    _mm256_store_ps(output_row + channel_idx,
                                    _mm256_mul_ps(_mm256_load_ps(output_row + channel_idx), correction_f32x8));
                for (position_idx = 0; position_idx < panel_length; position_idx++) {
                    __m256 const weight_f32x8 = _mm256_set1_ps(scores[position_idx]);
                    char const *values_row = values_plane + (panel_start + position_idx) * plane_row_bytes;
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
                        _mm256_store_ps(output_row + channel_idx,
                                        _mm256_fmadd_ps(weight_f32x8, load(values_row + channel_idx * element_bytes),
                                                        _mm256_load_ps(output_row + channel_idx)));
                }
            }

            nk_f32_t const inverse_sum = 1.0f / running_sum;
            __m256 const inverse_sum_f32x8 = _mm256_set1_ps(inverse_sum);
            nk_f32_t *destination = output + (query_offsets[segment_idx] + row_idx) * output_stride_floats +
                                    head_idx * depth;
            nk_size_t channel_idx = 0;
            for (; channel_idx + 8 <= depth; channel_idx += 8)
                _mm256_storeu_ps(destination + channel_idx,
                                 _mm256_mul_ps(_mm256_load_ps(output_row + channel_idx), inverse_sum_f32x8));
            for (; channel_idx < depth; channel_idx++) destination[channel_idx] = output_row[channel_idx] * inverse_sum;
        }
    }
}

NK_PUBLIC void nk_attention_packed_bf16_haswell(                                 //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_haswell_k_) {
        nk_attention_packed_bf16_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_attention_packed_haswell_(queries, sizeof(nk_bf16_t), &nk_attention_widen_bf16_haswell_,
                                 &nk_attention_load_bf16x8_haswell_, key_value_packed, output, head_count,
                                 key_value_head_count, depth, query_offsets, query_stride_bytes, output_stride_bytes,
                                 scale, first_task, task_count);
}

NK_PUBLIC void nk_attention_packed_e4m3_haswell(                                 //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_haswell_k_) {
        nk_attention_packed_e4m3_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_attention_packed_haswell_(queries, sizeof(nk_e4m3_t), &nk_attention_widen_e4m3_haswell_,
                                 &nk_attention_load_e4m3x8_haswell_, key_value_packed, output, head_count,
                                 key_value_head_count, depth, query_offsets, query_stride_bytes, output_stride_bytes,
                                 scale, first_task, task_count);
}

/**
 *  @brief Register 8×8 transpose of 16-bit elements (128-bit hierarchical unpack). Given 8 rows
 *         (each 8 words), returns the 8 columns. Used at pack time to turn 8 position rows into the
 *         drain-free K tiles: each column is one depth pair (two channels) of eight KV positions.
 *         The 128-bit unpacks never cross lanes, so the columns emerge in natural pair order.
 */
NK_INTERNAL void nk_attention_transpose_i16x8x8_haswell_(__m128i const rows_i16x8[8], __m128i columns_i16x8[8]) {
    __m128i const stage01_low_i16x8 = _mm_unpacklo_epi16(rows_i16x8[0], rows_i16x8[1]);
    __m128i const stage23_low_i16x8 = _mm_unpacklo_epi16(rows_i16x8[2], rows_i16x8[3]);
    __m128i const stage45_low_i16x8 = _mm_unpacklo_epi16(rows_i16x8[4], rows_i16x8[5]);
    __m128i const stage67_low_i16x8 = _mm_unpacklo_epi16(rows_i16x8[6], rows_i16x8[7]);
    __m128i const stage01_high_i16x8 = _mm_unpackhi_epi16(rows_i16x8[0], rows_i16x8[1]);
    __m128i const stage23_high_i16x8 = _mm_unpackhi_epi16(rows_i16x8[2], rows_i16x8[3]);
    __m128i const stage45_high_i16x8 = _mm_unpackhi_epi16(rows_i16x8[4], rows_i16x8[5]);
    __m128i const stage67_high_i16x8 = _mm_unpackhi_epi16(rows_i16x8[6], rows_i16x8[7]);
    __m128i const quad0123_ll_i16x8 = _mm_unpacklo_epi32(stage01_low_i16x8, stage23_low_i16x8);
    __m128i const quad4567_ll_i16x8 = _mm_unpacklo_epi32(stage45_low_i16x8, stage67_low_i16x8);
    __m128i const quad0123_lh_i16x8 = _mm_unpackhi_epi32(stage01_low_i16x8, stage23_low_i16x8);
    __m128i const quad4567_lh_i16x8 = _mm_unpackhi_epi32(stage45_low_i16x8, stage67_low_i16x8);
    __m128i const quad0123_hl_i16x8 = _mm_unpacklo_epi32(stage01_high_i16x8, stage23_high_i16x8);
    __m128i const quad4567_hl_i16x8 = _mm_unpacklo_epi32(stage45_high_i16x8, stage67_high_i16x8);
    __m128i const quad0123_hh_i16x8 = _mm_unpackhi_epi32(stage01_high_i16x8, stage23_high_i16x8);
    __m128i const quad4567_hh_i16x8 = _mm_unpackhi_epi32(stage45_high_i16x8, stage67_high_i16x8);
    columns_i16x8[0] = _mm_unpacklo_epi64(quad0123_ll_i16x8, quad4567_ll_i16x8);
    columns_i16x8[1] = _mm_unpackhi_epi64(quad0123_ll_i16x8, quad4567_ll_i16x8);
    columns_i16x8[2] = _mm_unpacklo_epi64(quad0123_lh_i16x8, quad4567_lh_i16x8);
    columns_i16x8[3] = _mm_unpackhi_epi64(quad0123_lh_i16x8, quad4567_lh_i16x8);
    columns_i16x8[4] = _mm_unpacklo_epi64(quad0123_hl_i16x8, quad4567_hl_i16x8);
    columns_i16x8[5] = _mm_unpackhi_epi64(quad0123_hl_i16x8, quad4567_hl_i16x8);
    columns_i16x8[6] = _mm_unpacklo_epi64(quad0123_hh_i16x8, quad4567_hh_i16x8);
    columns_i16x8[7] = _mm_unpackhi_epi64(quad0123_hh_i16x8, quad4567_hh_i16x8);
}

/**
 *  @brief Drain-free exact I32 scores for a query block over one panel. K is packed `[tile of 8
 *         positions][depth pair][8 lanes × 2 channels]`, so a 128-bit load widened to I16 holds two
 *         channels of eight KV positions — one score per lane. Each query's two channels broadcast as
 *         one dword and one VPMADDWD advances eight KV positions by two channels; accumulating over the
 *         depth pairs leaves eight exact scores per lane with no transpose. Eight queries share each K
 *         load. Writes a `block_rows × panel` score block, rows `nk_attention_panel_haswell_k_` apart.
 */
NK_INTERNAL void nk_attention_score_block_i8_haswell_(nk_i16_t const *queries_i16, nk_size_t block_rows,
                                                      char const *keys_plane, nk_size_t panel_start,
                                                      nk_size_t panel_length, nk_size_t depth_padded,
                                                      nk_i32_t *scores) {
    nk_size_t const depth_pairs = depth_padded / 2;
    nk_size_t const tile_first = panel_start / 8;
    nk_size_t const tile_count = (panel_length + 7) / 8;
    for (nk_size_t tile_idx = 0; tile_idx < tile_count; tile_idx++) {
        __m256i accumulators_i32x8[8];
        for (nk_size_t query_idx = 0; query_idx < 8; query_idx++)
            accumulators_i32x8[query_idx] = _mm256_setzero_si256();
        char const *keys_tile = keys_plane + (tile_first + tile_idx) * depth_pairs * 16;
        for (nk_size_t pair_idx = 0; pair_idx < depth_pairs; pair_idx++) {
            __m256i const keys_i16x16 = _mm256_cvtepi8_epi16(
                _mm_loadu_si128((__m128i const *)(keys_tile + pair_idx * 16)));
            for (nk_size_t query_idx = 0; query_idx < block_rows; query_idx++) {
                __m256i const query_pair_i16x16 = _mm256_broadcastd_epi32(
                    _mm_loadu_si32(queries_i16 + query_idx * nk_attention_max_depth_haswell_k_ + pair_idx * 2));
                accumulators_i32x8[query_idx] = _mm256_add_epi32(accumulators_i32x8[query_idx],
                                                                 _mm256_madd_epi16(query_pair_i16x16, keys_i16x16));
            }
        }
        for (nk_size_t query_idx = 0; query_idx < block_rows; query_idx++)
            _mm256_storeu_si256((__m256i *)(scores + query_idx * nk_attention_panel_haswell_k_ + tile_idx * 8),
                                accumulators_i32x8[query_idx]);
    }
}

/**
 *  @brief Streaming base-2 softmax over one panel for a single query row: row max over live columns,
 *         U8 weight quantization `round(255 · 2^(s₂ − m₂))` — separate multiply and add to match the
 *         serial reference's rounding — and the online-correction bookkeeping. Returns `2^(m_old − m_new)`.
 */
NK_INTERNAL nk_f32_t nk_attention_softmax_panel_i8_haswell_(nk_i32_t const *scores, nk_size_t panel_length,
                                                            nk_f32_t scale2, nk_f32_t *running_max2,
                                                            nk_f32_t *running_sum, nk_u8_t *weights) {
    __m256 const scale2_f32x8 = _mm256_set1_ps(scale2);
    __m256 max_f32x8 = _mm256_set1_ps(NK_F32_MIN);
    nk_size_t position_idx = 0;
    for (; position_idx + 8 <= panel_length; position_idx += 8)
        max_f32x8 = _mm256_max_ps(
            max_f32x8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const *)(scores + position_idx))),
                                     scale2_f32x8));
    nk_f32_t panel_max2 = nk_reduce_max_f32x8_haswell_(max_f32x8);
    for (; position_idx < panel_length; position_idx++) {
        nk_f32_t const scaled2 = (nk_f32_t)scores[position_idx] * scale2;
        if (scaled2 > panel_max2) panel_max2 = scaled2;
    }
    nk_f32_t const new_max2 = *running_max2 > panel_max2 ? *running_max2 : panel_max2;
    nk_f32_t const correction = _mm256_cvtss_f32(
        nk_attention_exp2_f32x8_haswell_(_mm256_set1_ps(*running_max2 - new_max2)));
    *running_max2 = new_max2;

    __m256 const max2_f32x8 = _mm256_set1_ps(new_max2);
    __m256 const amplitude_f32x8 = _mm256_set1_ps(255.0f);
    __m256 const half_f32x8 = _mm256_set1_ps(0.5f);
    __m128i const zero_u8x16 = _mm_setzero_si128();
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (position_idx = 0; position_idx + 8 <= panel_length; position_idx += 8) {
        __m256 const exp_f32x8 = nk_attention_exp2_f32x8_haswell_(
            _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const *)(scores + position_idx))),
                            scale2_f32x8, max2_f32x8));
        __m256i const weight_i32x8 = _mm256_cvttps_epi32(
            _mm256_add_ps(_mm256_mul_ps(exp_f32x8, amplitude_f32x8), half_f32x8));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, _mm256_cvtepi32_ps(weight_i32x8));
        __m128i const weight_i16x8 = _mm_packus_epi32(_mm256_castsi256_si128(weight_i32x8),
                                                      _mm256_extracti128_si256(weight_i32x8, 1));
        _mm_storel_epi64((__m128i *)(weights + position_idx), _mm_packus_epi16(weight_i16x8, zero_u8x16));
    }
    if (position_idx < panel_length) { // masked tail keeps the vector rounding mode end-to-end
        __m256i const lane_index_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i const tail_mask_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)(panel_length - position_idx)),
                                                           lane_index_i32x8);
        __m256 exp_f32x8 = nk_attention_exp2_f32x8_haswell_(_mm256_fmsub_ps(
            _mm256_cvtepi32_ps(_mm256_maskload_epi32((int const *)(scores + position_idx), tail_mask_i32x8)),
            scale2_f32x8, max2_f32x8));
        exp_f32x8 = _mm256_and_ps(exp_f32x8, _mm256_castsi256_ps(tail_mask_i32x8));
        __m256i const weight_i32x8 = _mm256_cvttps_epi32(
            _mm256_add_ps(_mm256_mul_ps(exp_f32x8, amplitude_f32x8), half_f32x8));
        sum_f32x8 = _mm256_add_ps(sum_f32x8, _mm256_cvtepi32_ps(weight_i32x8));
        __m128i const weight_i16x8 = _mm_packus_epi32(_mm256_castsi256_si128(weight_i32x8),
                                                      _mm256_extracti128_si256(weight_i32x8, 1));
        _mm_storel_epi64((__m128i *)(weights + position_idx), _mm_packus_epi16(weight_i16x8, zero_u8x16));
    }
    *running_sum = *running_sum * correction + nk_reduce_add_f32x8_haswell_(sum_f32x8);
    return correction;
}

/**
 *  @brief `O = O · correction + Σ w̃ · widened V-row` for one query row over one panel; V stays
 *         token-major and widens on the fly, exact-zero weights skipped like the serial reference.
 */
NK_INTERNAL void nk_attention_weighted_sum_panel_i8_haswell_(nk_u8_t const *weights, char const *values_plane,
                                                             nk_size_t panel_start, nk_size_t panel_length,
                                                             nk_size_t depth_padded, nk_f32_t correction,
                                                             nk_f32_t *output_row) {
    __m256 const correction_f32x8 = _mm256_set1_ps(correction);
    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
        _mm256_store_ps(output_row + channel_idx,
                        _mm256_mul_ps(_mm256_load_ps(output_row + channel_idx), correction_f32x8));
    for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx++) {
        nk_u8_t const weight_u8 = weights[position_idx];
        if (weight_u8 == 0) continue;
        __m256 const weight_f32x8 = _mm256_set1_ps((nk_f32_t)weight_u8);
        char const *values_row = values_plane + (panel_start + position_idx) * depth_padded;
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
            _mm256_store_ps(output_row + channel_idx,
                            _mm256_fmadd_ps(weight_f32x8,
                                            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                                                _mm_loadl_epi64((__m128i const *)(values_row + channel_idx)))),
                                            _mm256_load_ps(output_row + channel_idx)));
    }
}

NK_PUBLIC nk_size_t nk_attention_packed_size_i8_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_haswell_k_)
        return nk_attention_packed_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count);
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t payload_bytes = 0; // K in drain-free tiles, V token-major; both pad positions to the 8-tile
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count *
                         (nk_size_t)nk_size_round_up_to_multiple_(segment_lengths[segment_idx], 8) * depth_padded;
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_PUBLIC void nk_attention_pack_i8_haswell(                                                                   //
    nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count, nk_size_t depth,               //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                                          //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_haswell_k_) {
        nk_attention_pack_i8_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                    segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                    task_count);
        return;
    }

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const depth_pairs = depth_padded / 2;
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, 8, depth_padded);
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *payload_offsets_ro = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *payload_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 8);
        nk_size_t const plane_bytes = position_count_padded * depth_padded;
        char *keys_plane = payload_base + payload_offsets_ro[segment_idx] + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;

        // K is drain-free `[tile of 8 positions][depth pair][8 lanes × 2 channels]`. Each 8-position tile
        // stages token-major into a zeroed buffer (AVX2 has no masked byte load), then a register 8×8 word
        // transpose emits one 16-byte column per depth pair — no scatter.
        NK_ALIGN64 nk_i8_t stage[8 * nk_attention_max_depth_haswell_k_];
        for (nk_size_t tile_idx = 0; tile_idx * 8 < position_count_padded; tile_idx++) {
            for (nk_size_t lane_idx = 0; lane_idx < 8; lane_idx++) {
                nk_size_t const position_idx = tile_idx * 8 + lane_idx;
                nk_i8_t *stage_row = stage + lane_idx * depth_padded;
                if (position_idx < position_count) {
                    nk_i8_t const *keys_row = (nk_i8_t const *)((char const *)keys +
                                                                (position_first + position_idx) * key_stride_bytes) +
                                              key_value_head_idx * depth;
                    nk_size_t channel_idx = 0;
                    for (; channel_idx < depth; channel_idx++) stage_row[channel_idx] = keys_row[channel_idx];
                    for (; channel_idx < depth_padded; channel_idx++) stage_row[channel_idx] = 0;
                }
                else {
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx++)
                        stage_row[channel_idx] = 0;
                }
            }
            char *keys_tile = keys_plane + tile_idx * depth_pairs * 16;
            for (nk_size_t pair_block = 0; pair_block < depth_pairs; pair_block += 8) {
                __m128i rows_i16x8[8], columns_i16x8[8];
                for (nk_size_t lane_idx = 0; lane_idx < 8; lane_idx++)
                    rows_i16x8[lane_idx] = _mm_loadu_si128(
                        (__m128i const *)(stage + lane_idx * depth_padded + pair_block * 2));
                nk_attention_transpose_i16x8x8_haswell_(rows_i16x8, columns_i16x8);
                for (nk_size_t pair_idx = 0; pair_idx < 8 && pair_block + pair_idx < depth_pairs; pair_idx++)
                    _mm_storeu_si128((__m128i *)(keys_tile + (pair_block + pair_idx) * 16), columns_i16x8[pair_idx]);
            }
        }

        // V is token-major, zero-padded through the 8-position tile so the scalar-broadcast P×V reads clean rows.
        for (nk_size_t position_idx = 0; position_idx < position_count_padded; position_idx++) {
            char *values_destination = values_plane + position_idx * depth_padded;
            if (position_idx < position_count) {
                nk_i8_t const *values_row = (nk_i8_t const *)((char const *)values +
                                                              (position_first + position_idx) * value_stride_bytes) +
                                            key_value_head_idx * depth;
                nk_size_t channel_idx = 0;
                for (; channel_idx < depth; channel_idx++)
                    values_destination[channel_idx] = (char)values_row[channel_idx];
                for (; channel_idx < depth_padded; channel_idx++) values_destination[channel_idx] = 0;
            }
            else {
                for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx++)
                    values_destination[channel_idx] = 0;
            }
        }
    }
}

NK_PUBLIC void nk_attention_packed_i8_haswell(                                                                  //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,                                     //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_haswell_k_) {
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
    nk_size_t const depth_padded16 = nk_size_round_up_to_multiple_(depth_padded, 16);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // softmax(x) = softmax₂(x·log₂e)
    nk_size_t const panel_width = nk_attention_panel_haswell_k_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_i16_t queries_i16[8 * nk_attention_max_depth_haswell_k_];
    NK_ALIGN64 nk_f32_t output_rows[8 * nk_attention_max_depth_haswell_k_];
    NK_ALIGN64 nk_i32_t scores[8 * nk_attention_panel_haswell_k_];
    NK_ALIGN64 nk_u8_t weights[nk_attention_panel_haswell_k_];

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 8);
        nk_size_t const plane_bytes = position_count_padded * depth_padded;
        char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                 (head_idx / head_group_size) * plane_bytes;
        char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

        for (nk_size_t query_block = 0; query_block < row_count; query_block += 8) {
            nk_size_t const block_rows = (query_block + 8 <= row_count) ? 8 : (row_count - query_block);
            // Widen up to eight query rows to I16 once; zero the block tail so its scores stay harmless.
            for (nk_size_t block_row = 0; block_row < 8; block_row++) {
                nk_i16_t *query_i16 = queries_i16 + block_row * nk_attention_max_depth_haswell_k_;
                nk_size_t channel_idx = 0;
                if (block_row < block_rows) {
                    nk_i8_t const *query_row =
                        (nk_i8_t const *)((char const *)queries +
                                          (query_offsets[segment_idx] + query_block + block_row) * query_stride_bytes) +
                        head_idx * depth;
                    for (; channel_idx < depth; channel_idx++)
                        query_i16[channel_idx] = (nk_i16_t)query_row[channel_idx];
                }
                for (; channel_idx < depth_padded16; channel_idx++) query_i16[channel_idx] = 0;
            }
            for (nk_size_t block_row = 0; block_row < block_rows; block_row++)
                for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
                    _mm256_store_ps(output_rows + block_row * nk_attention_max_depth_haswell_k_ + channel_idx,
                                    _mm256_setzero_ps());
            nk_f32_t running_max2[8], running_sum[8];
            for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                running_max2[block_row] = NK_F32_MIN;
                running_sum[block_row] = 0;
            }

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_length = (panel_start + panel_width <= position_count)
                                                   ? panel_width
                                                   : (position_count - panel_start);
                nk_attention_score_block_i8_haswell_(queries_i16, block_rows, keys_plane, panel_start, panel_length,
                                                     depth_padded, scores);
                for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                    nk_f32_t const correction = nk_attention_softmax_panel_i8_haswell_(
                        scores + block_row * panel_width, panel_length, scale2, &running_max2[block_row],
                        &running_sum[block_row], weights);
                    nk_attention_weighted_sum_panel_i8_haswell_(
                        weights, values_plane, panel_start, panel_length, depth_padded, correction,
                        output_rows + block_row * nk_attention_max_depth_haswell_k_);
                }
            }

            for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                nk_f32_t const *output_row = output_rows + block_row * nk_attention_max_depth_haswell_k_;
                nk_f32_t const inverse_sum = 1.0f / running_sum[block_row];
                __m256 const inverse_sum_f32x8 = _mm256_set1_ps(inverse_sum);
                nk_f32_t *destination = output +
                                        (query_offsets[segment_idx] + query_block + block_row) * output_stride_floats +
                                        head_idx * depth;
                nk_size_t channel_idx = 0;
                for (; channel_idx + 8 <= depth; channel_idx += 8)
                    _mm256_storeu_ps(destination + channel_idx,
                                     _mm256_mul_ps(_mm256_load_ps(output_row + channel_idx), inverse_sum_f32x8));
                for (; channel_idx < depth; channel_idx++)
                    destination[channel_idx] = output_row[channel_idx] * inverse_sum;
            }
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

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X8664_
#endif // NK_ATTENTION_HASWELL_H
