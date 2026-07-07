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
NK_INTERNAL __m256 nk_attention_exp2_ps_haswell_(__m256 x_f32x8) {
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
typedef __m256 (*nk_attention_plane_widen_haswell_t_)(void const *plane_chunk);

NK_INTERNAL __m256 nk_attention_plane_widen_bf16_haswell_(void const *plane_chunk) {
    return nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)plane_chunk));
}

NK_INTERNAL __m256 nk_attention_plane_widen_e4m3_haswell_(void const *plane_chunk) {
    return nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)plane_chunk));
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
    nk_attention_plane_widen_haswell_t_ plane_widen,                                                            //
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
                    __m256 acc0_f32x8 = _mm256_setzero_ps(), acc1_f32x8 = _mm256_setzero_ps();
                    __m256 acc2_f32x8 = _mm256_setzero_ps(), acc3_f32x8 = _mm256_setzero_ps();
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8) {
                        __m256 const query_f32x8 = _mm256_load_ps(query_row + channel_idx);
                        nk_size_t const chunk_bytes = channel_idx * element_bytes;
                        acc0_f32x8 = _mm256_fmadd_ps(query_f32x8, plane_widen(keys_row + chunk_bytes), acc0_f32x8);
                        acc1_f32x8 = _mm256_fmadd_ps(query_f32x8, plane_widen(keys_row + plane_row_bytes + chunk_bytes),
                                                     acc1_f32x8);
                        acc2_f32x8 = _mm256_fmadd_ps(
                            query_f32x8, plane_widen(keys_row + 2 * plane_row_bytes + chunk_bytes), acc2_f32x8);
                        acc3_f32x8 = _mm256_fmadd_ps(
                            query_f32x8, plane_widen(keys_row + 3 * plane_row_bytes + chunk_bytes), acc3_f32x8);
                    }
                    scores[position_idx + 0] = nk_reduce_add_f32x8_haswell_(acc0_f32x8);
                    scores[position_idx + 1] = nk_reduce_add_f32x8_haswell_(acc1_f32x8);
                    scores[position_idx + 2] = nk_reduce_add_f32x8_haswell_(acc2_f32x8);
                    scores[position_idx + 3] = nk_reduce_add_f32x8_haswell_(acc3_f32x8);
                }
                for (; position_idx < panel_length; position_idx++) {
                    char const *keys_row = keys_plane + (panel_start + position_idx) * plane_row_bytes;
                    __m256 acc_f32x8 = _mm256_setzero_ps();
                    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
                        acc_f32x8 = _mm256_fmadd_ps(_mm256_load_ps(query_row + channel_idx),
                                                    plane_widen(keys_row + channel_idx * element_bytes), acc_f32x8);
                    scores[position_idx] = nk_reduce_add_f32x8_haswell_(acc_f32x8);
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
                    nk_attention_exp2_ps_haswell_(_mm256_set1_ps(running_max2 - new_max2)));
                running_max2 = new_max2;

                __m256 const scale2_f32x8 = _mm256_set1_ps(scale2);
                __m256 const max2_f32x8 = _mm256_set1_ps(new_max2);
                __m256 sum_f32x8 = _mm256_setzero_ps();
                for (position_idx = 0; position_idx + 8 <= panel_length; position_idx += 8) {
                    __m256 weights_f32x8 = nk_attention_exp2_ps_haswell_(
                        _mm256_fmsub_ps(_mm256_loadu_ps(scores + position_idx), scale2_f32x8, max2_f32x8));
                    sum_f32x8 = _mm256_add_ps(sum_f32x8, weights_f32x8);
                    _mm256_storeu_ps(scores + position_idx, weights_f32x8);
                }
                if (position_idx < panel_length) { // masked tail keeps the vector rounding mode end-to-end
                    __m256i const lane_idx_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                    __m256i const tail_mask_i32x8 = _mm256_cmpgt_epi32(
                        _mm256_set1_epi32((int)(panel_length - position_idx)), lane_idx_i32x8);
                    __m256 weights_f32x8 = nk_attention_exp2_ps_haswell_(_mm256_fmsub_ps(
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
                        _mm256_store_ps(
                            output_row + channel_idx,
                            _mm256_fmadd_ps(weight_f32x8, plane_widen(values_row + channel_idx * element_bytes),
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
                                 &nk_attention_plane_widen_bf16_haswell_, key_value_packed, output, head_count,
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
                                 &nk_attention_plane_widen_e4m3_haswell_, key_value_packed, output, head_count,
                                 key_value_head_count, depth, query_offsets, query_stride_bytes, output_stride_bytes,
                                 scale, first_task, task_count);
}

NK_PUBLIC nk_size_t nk_attention_packed_size_i8_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_haswell_k_)
        return nk_attention_packed_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_haswell_(key_value_head_count, depth, segment_lengths, segment_count, 1);
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
    nk_attention_pack_haswell_(keys, values, 1, key_value_head_count, depth, segment_offsets, segment_lengths,
                               segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                               task_count);
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
    nk_size_t const depth_full16 = depth_padded & ~(nk_size_t)15;
    nk_size_t const depth_padded16 = nk_size_round_up_to_multiple_(depth_padded, 16);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // softmax(x) = softmax₂(x·log₂e)
    nk_size_t const panel_width = nk_attention_panel_haswell_k_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_i16_t query_i16[nk_attention_max_depth_haswell_k_];
    NK_ALIGN64 nk_f32_t output_row[nk_attention_max_depth_haswell_k_];
    NK_ALIGN64 nk_i32_t scores[nk_attention_panel_haswell_k_];
    NK_ALIGN64 nk_u8_t weights[nk_attention_panel_haswell_k_];

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
            nk_i8_t const *query_row = (nk_i8_t const *)((char const *)queries +
                                                         (query_offsets[segment_idx] + row_idx) * query_stride_bytes) +
                                       head_idx * depth;
            nk_size_t channel_idx = 0;
            // Widen Q to I16 once per row; zero-pad through the 16-channel score-loop granularity.
            for (; channel_idx < depth; channel_idx++) query_i16[channel_idx] = (nk_i16_t)query_row[channel_idx];
            for (; channel_idx < depth_padded16; channel_idx++) query_i16[channel_idx] = 0;
            for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
                _mm256_store_ps(output_row + channel_idx, _mm256_setzero_ps());
            nk_f32_t running_max2 = NK_F32_MIN, running_sum = 0;

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_length = (panel_start + panel_width <= position_count)
                                                   ? panel_width
                                                   : (position_count - panel_start);
                // Exact I32 scores: four KV rows in flight, K sign-extended to I16 in-loop.

                nk_size_t position_idx = 0;
                for (; position_idx + 4 <= panel_length; position_idx += 4) {
                    char const *keys_row = keys_plane + (panel_start + position_idx) * depth_padded;
                    __m256i acc0_i32x8 = _mm256_setzero_si256(), acc1_i32x8 = _mm256_setzero_si256();
                    __m256i acc2_i32x8 = _mm256_setzero_si256(), acc3_i32x8 = _mm256_setzero_si256();
                    for (channel_idx = 0; channel_idx < depth_full16; channel_idx += 16) {
                        __m256i const query_i16x16 = _mm256_load_si256((__m256i const *)(query_i16 + channel_idx));
                        acc0_i32x8 = _mm256_add_epi32(
                            acc0_i32x8,
                            _mm256_madd_epi16(query_i16x16, _mm256_cvtepi8_epi16(_mm_loadu_si128(
                                                                (__m128i const *)(keys_row + channel_idx)))));
                        acc1_i32x8 = _mm256_add_epi32(
                            acc1_i32x8,
                            _mm256_madd_epi16(query_i16x16,
                                              _mm256_cvtepi8_epi16(_mm_loadu_si128(
                                                  (__m128i const *)(keys_row + depth_padded + channel_idx)))));
                        acc2_i32x8 = _mm256_add_epi32(
                            acc2_i32x8,
                            _mm256_madd_epi16(query_i16x16,
                                              _mm256_cvtepi8_epi16(_mm_loadu_si128(
                                                  (__m128i const *)(keys_row + 2 * depth_padded + channel_idx)))));
                        acc3_i32x8 = _mm256_add_epi32(
                            acc3_i32x8,
                            _mm256_madd_epi16(query_i16x16,
                                              _mm256_cvtepi8_epi16(_mm_loadu_si128(
                                                  (__m128i const *)(keys_row + 3 * depth_padded + channel_idx)))));
                    }
                    if (channel_idx < depth_padded) { // final 8-channel chunk; upper I16 lanes are zero on both sides
                        __m256i const query_i16x16 = _mm256_load_si256((__m256i const *)(query_i16 + channel_idx));
                        acc0_i32x8 = _mm256_add_epi32(
                            acc0_i32x8,
                            _mm256_madd_epi16(query_i16x16, _mm256_cvtepi8_epi16(_mm_loadl_epi64(
                                                                (__m128i const *)(keys_row + channel_idx)))));
                        acc1_i32x8 = _mm256_add_epi32(
                            acc1_i32x8,
                            _mm256_madd_epi16(query_i16x16,
                                              _mm256_cvtepi8_epi16(_mm_loadl_epi64(
                                                  (__m128i const *)(keys_row + depth_padded + channel_idx)))));
                        acc2_i32x8 = _mm256_add_epi32(
                            acc2_i32x8,
                            _mm256_madd_epi16(query_i16x16,
                                              _mm256_cvtepi8_epi16(_mm_loadl_epi64(
                                                  (__m128i const *)(keys_row + 2 * depth_padded + channel_idx)))));
                        acc3_i32x8 = _mm256_add_epi32(
                            acc3_i32x8,
                            _mm256_madd_epi16(query_i16x16,
                                              _mm256_cvtepi8_epi16(_mm_loadl_epi64(
                                                  (__m128i const *)(keys_row + 3 * depth_padded + channel_idx)))));
                    }
                    __m128i const sum0_i32x4 = _mm_add_epi32( // 4-way transpose: 4 accumulators to 4 scores
                        _mm256_castsi256_si128(acc0_i32x8), _mm256_extracti128_si256(acc0_i32x8, 1));
                    __m128i const sum1_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(acc1_i32x8),
                                                             _mm256_extracti128_si256(acc1_i32x8, 1));
                    __m128i const sum2_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(acc2_i32x8),
                                                             _mm256_extracti128_si256(acc2_i32x8, 1));
                    __m128i const sum3_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(acc3_i32x8),
                                                             _mm256_extracti128_si256(acc3_i32x8, 1));
                    __m128i const transpose01_lo_i32x4 = _mm_unpacklo_epi32(sum0_i32x4, sum1_i32x4);
                    __m128i const transpose23_lo_i32x4 = _mm_unpacklo_epi32(sum2_i32x4, sum3_i32x4);
                    __m128i const transpose01_hi_i32x4 = _mm_unpackhi_epi32(sum0_i32x4, sum1_i32x4);
                    __m128i const transpose23_hi_i32x4 = _mm_unpackhi_epi32(sum2_i32x4, sum3_i32x4);
                    __m128i const scores_i32x4 = _mm_add_epi32(
                        _mm_add_epi32(_mm_unpacklo_epi64(transpose01_lo_i32x4, transpose23_lo_i32x4),
                                      _mm_unpackhi_epi64(transpose01_lo_i32x4, transpose23_lo_i32x4)),
                        _mm_add_epi32(_mm_unpacklo_epi64(transpose01_hi_i32x4, transpose23_hi_i32x4),
                                      _mm_unpackhi_epi64(transpose01_hi_i32x4, transpose23_hi_i32x4)));
                    _mm_storeu_si128((__m128i *)(scores + position_idx), scores_i32x4);
                }
                for (; position_idx < panel_length; position_idx++) {
                    char const *keys_row = keys_plane + (panel_start + position_idx) * depth_padded;
                    __m256i acc_i32x8 = _mm256_setzero_si256();
                    for (channel_idx = 0; channel_idx < depth_full16; channel_idx += 16)
                        acc_i32x8 = _mm256_add_epi32(
                            acc_i32x8,
                            _mm256_madd_epi16(
                                _mm256_load_si256((__m256i const *)(query_i16 + channel_idx)),
                                _mm256_cvtepi8_epi16(_mm_loadu_si128((__m128i const *)(keys_row + channel_idx)))));
                    if (channel_idx < depth_padded)
                        acc_i32x8 = _mm256_add_epi32(
                            acc_i32x8,
                            _mm256_madd_epi16(
                                _mm256_load_si256((__m256i const *)(query_i16 + channel_idx)),
                                _mm256_cvtepi8_epi16(_mm_loadl_epi64((__m128i const *)(keys_row + channel_idx)))));
                    scores[position_idx] = nk_reduce_add_i32x8_haswell_(acc_i32x8);
                }

                // Row max over live columns, online correction, U8 weight quantization.

                __m256 const scale2_f32x8 = _mm256_set1_ps(scale2);
                __m256 max_f32x8 = _mm256_set1_ps(NK_F32_MIN);
                position_idx = 0;
                for (; position_idx + 8 <= panel_length; position_idx += 8)
                    max_f32x8 = _mm256_max_ps(
                        max_f32x8,
                        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const *)(scores + position_idx))),
                                      scale2_f32x8));
                nk_f32_t panel_max2 = nk_reduce_max_f32x8_haswell_(max_f32x8);
                for (; position_idx < panel_length; position_idx++) {
                    nk_f32_t const scaled2 = (nk_f32_t)scores[position_idx] * scale2;
                    if (scaled2 > panel_max2) panel_max2 = scaled2;
                }
                nk_f32_t const new_max2 = running_max2 > panel_max2 ? running_max2 : panel_max2;
                nk_f32_t const correction = _mm256_cvtss_f32(
                    nk_attention_exp2_ps_haswell_(_mm256_set1_ps(running_max2 - new_max2)));
                running_max2 = new_max2;

                __m256 const max2_f32x8 = _mm256_set1_ps(new_max2);
                __m256 const amplitude_f32x8 = _mm256_set1_ps(255.0f);
                __m256 const half_f32x8 = _mm256_set1_ps(0.5f);
                __m128i const zero_u8x16 = _mm_setzero_si128();
                __m256 sum_f32x8 = _mm256_setzero_ps();
                for (position_idx = 0; position_idx + 8 <= panel_length; position_idx += 8) {
                    __m256 const exp_f32x8 = nk_attention_exp2_ps_haswell_(_mm256_fmsub_ps(
                        _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i const *)(scores + position_idx))), scale2_f32x8,
                        max2_f32x8));
                    __m256i const weight_i32x8 = _mm256_cvttps_epi32(
                        _mm256_add_ps(_mm256_mul_ps(exp_f32x8, amplitude_f32x8), half_f32x8));
                    sum_f32x8 = _mm256_add_ps(sum_f32x8, _mm256_cvtepi32_ps(weight_i32x8));
                    __m128i const weight_i16x8 = _mm_packus_epi32(_mm256_castsi256_si128(weight_i32x8),
                                                                  _mm256_extracti128_si256(weight_i32x8, 1));
                    _mm_storel_epi64((__m128i *)(weights + position_idx), _mm_packus_epi16(weight_i16x8, zero_u8x16));
                }
                if (position_idx < panel_length) { // masked tail keeps the vector rounding mode end-to-end
                    __m256i const lane_idx_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                    __m256i const tail_mask_i32x8 = _mm256_cmpgt_epi32(
                        _mm256_set1_epi32((int)(panel_length - position_idx)), lane_idx_i32x8);
                    __m256 exp_f32x8 = nk_attention_exp2_ps_haswell_(
                        _mm256_fmsub_ps(_mm256_cvtepi32_ps(_mm256_maskload_epi32((int const *)(scores + position_idx),
                                                                                 tail_mask_i32x8)),
                                        scale2_f32x8, max2_f32x8));
                    exp_f32x8 = _mm256_and_ps(exp_f32x8, _mm256_castsi256_ps(tail_mask_i32x8));
                    __m256i const weight_i32x8 = _mm256_cvttps_epi32(
                        _mm256_add_ps(_mm256_mul_ps(exp_f32x8, amplitude_f32x8), half_f32x8));
                    sum_f32x8 = _mm256_add_ps(sum_f32x8, _mm256_cvtepi32_ps(weight_i32x8));
                    __m128i const weight_i16x8 = _mm_packus_epi32(_mm256_castsi256_si128(weight_i32x8),
                                                                  _mm256_extracti128_si256(weight_i32x8, 1));
                    _mm_storel_epi64((__m128i *)(weights + position_idx), _mm_packus_epi16(weight_i16x8, zero_u8x16));
                }
                running_sum = running_sum * correction + nk_reduce_add_f32x8_haswell_(sum_f32x8);

                // O = O · correction + Σ w̃ · widened V-row; exact-zero weights skipped like serial.

                __m256 const correction_f32x8 = _mm256_set1_ps(correction);
                for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
                    _mm256_store_ps(output_row + channel_idx,
                                    _mm256_mul_ps(_mm256_load_ps(output_row + channel_idx), correction_f32x8));
                for (position_idx = 0; position_idx < panel_length; position_idx++) {
                    nk_u8_t const weight_u8 = weights[position_idx];
                    if (weight_u8 == 0) continue;
                    __m256 const weight_f32x8 = _mm256_set1_ps((nk_f32_t)weight_u8);
                    char const *values_row = values_plane + (panel_start + position_idx) * depth_padded;
                    for (channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
                        _mm256_store_ps(output_row + channel_idx,
                                        _mm256_fmadd_ps(weight_f32x8,
                                                        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(
                                                            (__m128i const *)(values_row + channel_idx)))),
                                                        _mm256_load_ps(output_row + channel_idx)));
                }
            }

            nk_f32_t const inverse_sum = 1.0f / running_sum;
            __m256 const inverse_sum_f32x8 = _mm256_set1_ps(inverse_sum);
            nk_f32_t *destination = output + (query_offsets[segment_idx] + row_idx) * output_stride_floats +
                                    head_idx * depth;
            for (channel_idx = 0; channel_idx + 8 <= depth; channel_idx += 8)
                _mm256_storeu_ps(destination + channel_idx,
                                 _mm256_mul_ps(_mm256_load_ps(output_row + channel_idx), inverse_sum_f32x8));
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

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X8664_
#endif // NK_ATTENTION_HASWELL_H
