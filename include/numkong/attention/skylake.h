/**
 *  @file include/numkong/attention/skylake.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief Ragged attention for AVX-512 Skylake-X generation CPUs.
 *
 *  @sa include/numkong/attention.h
 *
 *  Compatibility backend for AVX-512F machines without BF16 or AMX ISA extensions. Storage follows
 *  the `dots/skylake.h` conventions exactly: BF16 inputs stay BF16 at rest and widen to F32 inside
 *  the compute loops, shift-based, two ops per 16 lanes; E4M3 converts once to F16 during packing,
 *  so the in-loop widening is a single hardware @c VCVTPH2PS — the same asymmetry the GEMM family
 *  chose, trading one cheap pack-time pass for halved KV streaming traffic against F32 planes.
 *
 *  The panel structure matches the family design — per query row, KV is swept in panels with an
 *  exact online correction, a base-2 streaming softmax sharing the family's degree-7 polynomial,
 *  and a score core with four KV rows in flight on the dual FMA ports. Packed payload per segment:
 *  K planes then V planes, `[key_value_head][position][channel]` in 16-bit scalars with channels
 *  zero-padded to a multiple of 16. Heads deeper than 256 channels widen the query and accumulate
 *  scores in 256-channel chunks; the output accumulates in place.
 */
#ifndef NUMKONG_ATTENTION_SKYLAKE_H
#define NUMKONG_ATTENTION_SKYLAKE_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_SKYLAKE_

#include "numkong/attention/serial.h" // shared packed-KV header/offsets, width-agnostic fallback
#include "numkong/cast/skylake.h"     // widening loaders like `nk_load_bf16x16_to_f32x16_skylake_`
#include "numkong/each/skylake.h"     // `nk_exp2_f32x16_skylake_`
#include "numkong/reduce/skylake.h"   // `nk_reduce_add_f32x16_skylake_`, `nk_reduce_max_f32x16_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

enum {

    /** KV panel width in positions; the F32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_skylake_k_ = 512,

    /** Channels of the query row widened at once; deeper heads accumulate scores across chunks. */
    nk_attention_depth_chunk_skylake_k_ = 256,
};

/** One panel of the streaming base-2 softmax: merges the panel maximum into the running
 *  one, exponentiates the scores in place into weights, and folds the panel weight-sum
 *  into the running sum. Returns the correction 2^(m_old − m_new) the caller applies to
 *  its output accumulators. */
NUMKONG_INLINE nk_f32_t nk_attention_softmax_panel_skylake_(nk_f32_t *scores, nk_size_t panel_length, nk_f32_t scale2,
                                                            nk_f32_t *running_max2, nk_f32_t *running_sum) {
    __m512 max_f32x16 = _mm512_set1_ps(NUMKONG_F32_MIN);
    nk_size_t position_index = 0;
    for (; position_index + 16 <= panel_length; position_index += 16)
        max_f32x16 = _mm512_max_ps(max_f32x16, _mm512_loadu_ps(scores + position_index));
    if (position_index < panel_length) {
        __mmask16 const tail_m16 = (__mmask16)((1u << (panel_length - position_index)) - 1);
        max_f32x16 = _mm512_mask_max_ps(max_f32x16, tail_m16, max_f32x16,
                                        _mm512_maskz_loadu_ps(tail_m16, scores + position_index));
    }
    nk_f32_t const panel_max2 = nk_reduce_max_f32x16_skylake_(max_f32x16) * scale2;
    nk_f32_t const new_max2 = *running_max2 > panel_max2 ? *running_max2 : panel_max2;
    nk_f32_t const correction = _mm512_cvtss_f32(nk_exp2_f32x16_skylake_(_mm512_set1_ps(*running_max2 - new_max2)));
    *running_max2 = new_max2;

    __m512 const scale2_f32x16 = _mm512_set1_ps(scale2);
    __m512 const max2_f32x16 = _mm512_set1_ps(new_max2);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    nk_size_t const panel_full = panel_length & ~(nk_size_t)15;
    __mmask16 const panel_tail_m16 = (__mmask16)((1u << (panel_length - panel_full)) - 1);
    for (position_index = 0; position_index < panel_full; position_index += 16) {
        __m512 weights_f32x16 = nk_exp2_f32x16_skylake_(
            _mm512_fmsub_ps(_mm512_loadu_ps(scores + position_index), scale2_f32x16, max2_f32x16));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, weights_f32x16);
        _mm512_storeu_ps(scores + position_index, weights_f32x16);
    }
    if (position_index < panel_length) {
        __m512 weights_f32x16 = _mm512_maskz_mov_ps(
            panel_tail_m16, nk_exp2_f32x16_skylake_(
                                _mm512_fmsub_ps(_mm512_loadu_ps(scores + position_index), scale2_f32x16, max2_f32x16)));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, weights_f32x16);
        _mm512_storeu_ps(scores + position_index, weights_f32x16);
    }
    *running_sum = *running_sum * correction + nk_reduce_add_f32x16_skylake_(sum_f32x16);
    return correction;
}

/** Widens 16 packed-plane scalars (BF16 or F16 at rest) to F32 inside the hot loops. */
typedef __m512 (*nk_attention_load_skylake_t_)(void const *plane_chunk);

NUMKONG_INLINE __m512 nk_attention_load_bf16x16_skylake_(void const *plane_chunk) {
    nk_b512_vec_t widened;
    nk_load_bf16x16_to_f32x16_skylake_(plane_chunk, &widened);
    return widened.zmm_ps;
}

NUMKONG_INLINE __m512 nk_attention_load_f16x16_skylake_(void const *plane_chunk) {
    nk_b512_vec_t widened;
    nk_load_f16x16_to_f32x16_skylake_(plane_chunk, &widened);
    return widened.zmm_ps;
}

/** Narrows @p count contiguous elements into 16-bit plane scalars, zero-filling to @p padded. */
typedef void (*nk_attention_narrow_skylake_t_)(void const *source, void *destination, nk_size_t count,
                                               nk_size_t padded);

NUMKONG_INLINE void nk_attention_narrow_bf16_skylake_(void const *source, void *destination, nk_size_t count,
                                                      nk_size_t padded) {
    nk_b256_vec_t chunk_vec; // BF16 stays BF16 at rest, like the dots family
    for (nk_size_t channel_index = 0; channel_index < padded; channel_index += 16) {
        nk_size_t const chunk = channel_index < count ? (count - channel_index < 16 ? count - channel_index : 16) : 0;
        nk_partial_load_b16x16_skylake_((nk_bf16_t const *)source + channel_index, &chunk_vec, chunk);
        _mm256_storeu_si256((__m256i *)((nk_bf16_t *)destination + channel_index), chunk_vec.ymm);
    }
}

NUMKONG_INLINE void nk_attention_narrow_e4m3_skylake_(void const *source, void *destination, nk_size_t count,
                                                      nk_size_t padded) {
    nk_b256_vec_t converted_vec; // E4M3 converts once to F16, so the hot loops widen with one VCVTPH2PS
    for (nk_size_t channel_index = 0; channel_index < padded; channel_index += 16) {
        nk_size_t const chunk = channel_index < count ? (count - channel_index < 16 ? count - channel_index : 16) : 0;
        nk_partial_load_e4m3x16_to_f16x16_skylake_((char const *)source + channel_index, &converted_vec, chunk);
        _mm256_storeu_si256((__m256i *)((nk_f16_t *)destination + channel_index), converted_vec.ymm);
    }
}

/** Widens @p count raw query elements to F32 into @p destination, zero-filling to @p padded. */
typedef void (*nk_attention_widen_skylake_t_)(void const *source, nk_f32_t *destination, nk_size_t count,
                                              nk_size_t padded);

NUMKONG_INLINE void nk_attention_widen_bf16_skylake_(void const *source, nk_f32_t *destination, nk_size_t count,
                                                     nk_size_t padded) {
    nk_size_t channel_index = 0;
    nk_b512_vec_t widened;
    for (; channel_index + 16 <= count; channel_index += 16) {
        nk_load_bf16x16_to_f32x16_skylake_((nk_bf16_t const *)source + channel_index, &widened);
        _mm512_storeu_ps(destination + channel_index, widened.zmm_ps);
    }
    if (channel_index < count) {
        nk_partial_load_bf16x16_to_f32x16_skylake_((nk_bf16_t const *)source + channel_index, &widened,
                                                   count - channel_index);
        _mm512_storeu_ps(destination + channel_index, widened.zmm_ps);
        channel_index += 16;
    }
    for (; channel_index < padded; channel_index += 16)
        _mm512_storeu_ps(destination + channel_index, _mm512_setzero_ps());
}

NUMKONG_INLINE void nk_attention_widen_e4m3_skylake_(void const *source, nk_f32_t *destination, nk_size_t count,
                                                     nk_size_t padded) {
    nk_size_t channel_index = 0;
    nk_b512_vec_t widened;
    for (; channel_index + 16 <= count; channel_index += 16) {
        nk_load_e4m3x16_to_f32x16_skylake_((nk_e4m3_t const *)source + channel_index, &widened);
        _mm512_storeu_ps(destination + channel_index, widened.zmm_ps);
    }
    if (channel_index < count) {
        nk_partial_load_e4m3x16_to_f32x16_skylake_((nk_e4m3_t const *)source + channel_index, &widened,
                                                   count - channel_index);
        _mm512_storeu_ps(destination + channel_index, widened.zmm_ps);
        channel_index += 16;
    }
    for (; channel_index < padded; channel_index += 16)
        _mm512_storeu_ps(destination + channel_index, _mm512_setzero_ps());
}

/** Bytes of a pack whose planes hold 16-bit scalars for both dtypes, BF16 or F16. */
NUMKONG_INLINE nk_size_t nk_attention_pack_size_skylake_(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count) {
    nk_size_t const unit_bytes = nk_size_round_up_to_multiple_(depth, 16) * sizeof(nk_bf16_t);
    return nk_attention_pack_bound_(key_value_head_count, token_count, segment_count, 1, unit_bytes);
}

NUMKONG_INLINE void nk_attention_pack_skylake_(                                                //
    void const *keys, void const *values, nk_size_t element_bytes,                             //
    nk_attention_narrow_skylake_t_ narrow,                                                     //
    nk_size_t key_value_head_count, nk_size_t depth,                                           //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, //
    nk_size_t key_stride, nk_size_t value_stride, void *key_value_packed,                      //
    nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 16);
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 tasks_begin, 1, depth_padded * sizeof(nk_bf16_t), nk_cap_skylake_k);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_index = tasks_begin; task_index < tasks_end; task_index++) {
        nk_size_t const segment_index = task_index / key_value_head_count,
                        key_value_head_index = task_index % key_value_head_count;
        for (; payload_segment < segment_index; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_(segment_lengths[payload_segment], key_value_head_count,
                                                               1, depth_padded * sizeof(nk_bf16_t));
        nk_size_t const position_count = segment_lengths[segment_index];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_index];
        nk_size_t const plane_bytes = position_count * depth_padded * sizeof(nk_bf16_t);
        char *keys_plane = payload_base + payload_offset + key_value_head_index * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_index = 0; position_index < position_count; position_index++) {
            narrow((char const *)keys + (position_first + position_index) * key_stride +
                       key_value_head_index * depth * element_bytes,
                   keys_plane + position_index * depth_padded * sizeof(nk_bf16_t), depth, depth_padded);
            narrow((char const *)values + (position_first + position_index) * value_stride +
                       key_value_head_index * depth * element_bytes,
                   values_plane + position_index * depth_padded * sizeof(nk_bf16_t), depth, depth_padded);
        }
    }
}

/** Shared attention core over 16-bit planes: per query row, panel-flash with an exact online
 *  correction; scores keep four KV rows in flight, widening in-loop. */
NUMKONG_INLINE void nk_attention_packed_skylake_(                                                   //
    void const *queries, nk_size_t element_bytes, nk_attention_widen_skylake_t_ widen,              //
    nk_attention_load_skylake_t_ load,                                                              //
    void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp,                          //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                          //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) {

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
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 16);
    nk_size_t const plane_row_bytes = depth_padded * sizeof(nk_bf16_t);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)
    nk_size_t const panel_width = nk_attention_panel_skylake_k_;

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_f32_t query_row[nk_attention_depth_chunk_skylake_k_];
    nk_align_(64) nk_f32_t scores[nk_attention_panel_skylake_k_];
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const depth_tail_m16 = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t head_index = 0; head_index < head_count && tasks_begin < tasks_end; head_index++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_index) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_index) / head_count;
        for (nk_size_t segment_index = nk_attention_segment_of_(query_offsets, segment_count, token_first);
             segment_index < segment_count && query_offsets[segment_index] < token_end; segment_index++) {
            nk_size_t const query_first = query_offsets[segment_index], query_end = query_offsets[segment_index + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = segment_lengths[segment_index];
            nk_i64_t const first_position = nk_attention_first_position_(query_end - query_first, position_count);
            nk_size_t const plane_bytes = position_count * plane_row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_index] +
                                     (head_index / head_group_size) * plane_bytes;
            char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_index = row_begin; row_index < row_end; row_index++) {
                char const *query_source = (char const *)queries + (query_first + row_index) * query_stride +
                                           head_index * depth * element_bytes;
                nk_size_t const token = query_first + row_index;
                nk_f32_t *output_row = output + token * output_stride_floats + head_index * depth;
                for (nk_size_t channel_index = 0; channel_index < depth; channel_index += 16)
                    _mm512_mask_storeu_ps(output_row + channel_index,
                                          channel_index + 16 <= depth ? (__mmask16)0xFFFF : depth_tail_m16,
                                          _mm512_setzero_ps());
                nk_f32_t running_max2 = NUMKONG_F32_MIN, running_sum = 0;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_index, position_count, &key_begin,
                                            &key_end);

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    // Scores: four KV rows in flight, plane scalars widened in-loop.

                    nk_size_t position_index = 0;
                    for (; position_index < panel_length; position_index += 16)
                        _mm512_store_ps(scores + position_index, _mm512_setzero_ps());
                    for (nk_size_t chunk_start = 0; chunk_start < depth_padded;
                         chunk_start += nk_attention_depth_chunk_skylake_k_) {
                        nk_size_t const chunk_padded = depth_padded - chunk_start < nk_attention_depth_chunk_skylake_k_
                                                           ? depth_padded - chunk_start
                                                           : nk_attention_depth_chunk_skylake_k_;
                        widen(query_source + chunk_start * element_bytes, query_row,
                              depth - chunk_start < chunk_padded ? depth - chunk_start : chunk_padded, chunk_padded);
                        for (position_index = 0; position_index < panel_length; position_index += 4) {
                            // Rows past the panel repeat the last live one; scores go unread.
                            nk_size_t const live_last = panel_length - position_index - 1;
                            char const *keys_row0 = keys_plane + (panel_start + position_index) * plane_row_bytes +
                                                    chunk_start * sizeof(nk_bf16_t);
                            char const *keys_row1 = keys_row0 + (live_last < 1 ? live_last : 1) * plane_row_bytes;
                            char const *keys_row2 = keys_row0 + (live_last < 2 ? live_last : 2) * plane_row_bytes;
                            char const *keys_row3 = keys_row0 + (live_last < 3 ? live_last : 3) * plane_row_bytes;
                            __m512 accumulator0_f32x16 = _mm512_setzero_ps(), accumulator1_f32x16 = _mm512_setzero_ps();
                            __m512 accumulator2_f32x16 = _mm512_setzero_ps(), accumulator3_f32x16 = _mm512_setzero_ps();
                            for (nk_size_t channel_index = 0; channel_index < chunk_padded; channel_index += 16) {
                                __m512 const query_f32x16 = _mm512_load_ps(query_row + channel_index);
                                nk_size_t const chunk_bytes = channel_index * sizeof(nk_bf16_t);
                                accumulator0_f32x16 = _mm512_fmadd_ps(query_f32x16, load(keys_row0 + chunk_bytes),
                                                                      accumulator0_f32x16);
                                accumulator1_f32x16 = _mm512_fmadd_ps(query_f32x16, load(keys_row1 + chunk_bytes),
                                                                      accumulator1_f32x16);
                                accumulator2_f32x16 = _mm512_fmadd_ps(query_f32x16, load(keys_row2 + chunk_bytes),
                                                                      accumulator2_f32x16);
                                accumulator3_f32x16 = _mm512_fmadd_ps(query_f32x16, load(keys_row3 + chunk_bytes),
                                                                      accumulator3_f32x16);
                            }
                            scores[position_index + 0] += nk_reduce_add_f32x16_skylake_(accumulator0_f32x16);
                            scores[position_index + 1] += nk_reduce_add_f32x16_skylake_(accumulator1_f32x16);
                            scores[position_index + 2] += nk_reduce_add_f32x16_skylake_(accumulator2_f32x16);
                            scores[position_index + 3] += nk_reduce_add_f32x16_skylake_(accumulator3_f32x16);
                        }
                    }

                    nk_f32_t const correction = nk_attention_softmax_panel_skylake_(scores, panel_length, scale2,
                                                                                    &running_max2, &running_sum);

                    // O = O · correction + Σ weight · widened V-row over the panel.

                    __m512 const correction_f32x16 = _mm512_set1_ps(correction);
                    for (nk_size_t channel_index = 0; channel_index < depth; channel_index += 16) {
                        __mmask16 const channel_m16 = channel_index + 16 <= depth ? (__mmask16)0xFFFF : depth_tail_m16;
                        _mm512_mask_storeu_ps(
                            output_row + channel_index, channel_m16,
                            _mm512_mul_ps(_mm512_maskz_loadu_ps(channel_m16, output_row + channel_index),
                                          correction_f32x16));
                    }
                    for (position_index = 0; position_index < panel_length; position_index++) {
                        __m512 const weight_f32x16 = _mm512_set1_ps(scores[position_index]);
                        char const *values_row = values_plane + (panel_start + position_index) * plane_row_bytes;
                        for (nk_size_t channel_index = 0; channel_index < depth; channel_index += 16) {
                            __mmask16 const channel_m16 = channel_index + 16 <= depth ? (__mmask16)0xFFFF
                                                                                      : depth_tail_m16;
                            _mm512_mask_storeu_ps(
                                output_row + channel_index, channel_m16,
                                _mm512_fmadd_ps(weight_f32x16, load(values_row + channel_index * sizeof(nk_bf16_t)),
                                                _mm512_maskz_loadu_ps(channel_m16, output_row + channel_index)));
                        }
                    }
                }

                __m512 const inverse_sum_f32x16 = _mm512_set1_ps(running_sum > 0 ? 1 / running_sum : 0);
                for (nk_size_t channel_index = 0; channel_index < depth; channel_index += 16) {
                    __mmask16 const channel_m16 = channel_index + 16 <= depth ? (__mmask16)0xFFFF : depth_tail_m16;
                    _mm512_mask_storeu_ps(output_row + channel_index, channel_m16,
                                          _mm512_mul_ps(_mm512_maskz_loadu_ps(channel_m16, output_row + channel_index),
                                                        inverse_sum_f32x16));
                }
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_index] = nk_attention_log_sum_exp_(running_max2, running_sum);
            }
        }
    }
}

#if NUMKONG_TARGET_SKYLAKE

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_skylake(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes) {
    *bytes = nk_attention_pack_size_skylake_(key_value_head_count, depth, token_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_skylake(void const *key_value_packed, nk_size_t *heads,
                                                               nk_size_t *depth, nk_size_t *segments,
                                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_skylake_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_skylake(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes) {
    *bytes = nk_attention_pack_size_skylake_(key_value_head_count, depth, token_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_skylake(void const *key_value_packed, nk_size_t *heads,
                                                               nk_size_t *depth, nk_size_t *segments,
                                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_skylake_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_bf16_skylake(                                              //
    nk_bf16_t const *keys, nk_bf16_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_attention_pack_skylake_(keys, values, sizeof(nk_bf16_t), &nk_attention_narrow_bf16_skylake_,
                               key_value_head_count, depth, segment_offsets, segment_lengths, segment_count, key_stride,
                               value_stride, key_value_packed, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_skylake(                                              //
    nk_e4m3_t const *keys, nk_e4m3_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_attention_pack_skylake_(keys, values, sizeof(nk_e4m3_t), &nk_attention_narrow_e4m3_skylake_,
                               key_value_head_count, depth, segment_offsets, segment_lengths, segment_count, key_stride,
                               value_stride, key_value_packed, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_bf16_skylake(                        //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_skylake_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_skylake_(queries, sizeof(nk_bf16_t), &nk_attention_widen_bf16_skylake_,
                                 &nk_attention_load_bf16x16_skylake_, key_value_packed, output, log_sum_exp, head_count,
                                 key_value_head_count, depth, query_offsets, query_stride, output_stride, scale, band,
                                 tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_e4m3_skylake(                        //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_skylake_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_skylake_(queries, sizeof(nk_e4m3_t), &nk_attention_widen_e4m3_skylake_,
                                 &nk_attention_load_f16x16_skylake_, key_value_packed, output, log_sum_exp, head_count,
                                 key_value_head_count, depth, query_offsets, query_stride, output_stride, scale, band,
                                 tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_rope_f32_skylake(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                      nk_f32_t *y, nk_size_t rows, nk_size_t head_count,
                                                      nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                      nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(depth % 2 == 0);
    nk_size_t const half_depth = depth / 2;
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_f32_t const *cos_row = cos + r * half_depth;
        nk_f32_t const *sin_row = sin + r * half_depth;
        nk_f32_t const *x_row = (nk_f32_t const *)((unsigned char const *)x + r * x_stride);
        nk_f32_t *y_row = (nk_f32_t *)((unsigned char *)y + r * y_stride);
        for (nk_size_t h = 0; h != head_count; ++h) {
            nk_f32_t const *x_base = x_row + h * depth;
            nk_f32_t *y_base = y_row + h * depth;
            nk_size_t i = 0;
            for (; i + 16 <= half_depth; i += 16) {
                __m512 low_f32x16 = _mm512_loadu_ps(x_base + i);
                __m512 high_f32x16 = _mm512_loadu_ps(x_base + i + half_depth);
                __m512 cos_f32x16 = _mm512_loadu_ps(cos_row + i), sin_f32x16 = _mm512_loadu_ps(sin_row + i);
                _mm512_storeu_ps(y_base + i,
                                 _mm512_fmsub_ps(low_f32x16, cos_f32x16, _mm512_mul_ps(high_f32x16, sin_f32x16)));
                _mm512_storeu_ps(y_base + i + half_depth,
                                 _mm512_fmadd_ps(low_f32x16, sin_f32x16, _mm512_mul_ps(high_f32x16, cos_f32x16)));
            }
            if (i < half_depth) {
                __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFFu, (unsigned)(half_depth - i));
                __m512 low_f32x16 = _mm512_maskz_loadu_ps(mask_m16, x_base + i);
                __m512 high_f32x16 = _mm512_maskz_loadu_ps(mask_m16, x_base + i + half_depth);
                __m512 cos_f32x16 = _mm512_maskz_loadu_ps(mask_m16, cos_row + i);
                __m512 sin_f32x16 = _mm512_maskz_loadu_ps(mask_m16, sin_row + i);
                _mm512_mask_storeu_ps(y_base + i, mask_m16,
                                      _mm512_fmsub_ps(low_f32x16, cos_f32x16, _mm512_mul_ps(high_f32x16, sin_f32x16)));
                _mm512_mask_storeu_ps(y_base + i + half_depth, mask_m16,
                                      _mm512_fmadd_ps(low_f32x16, sin_f32x16, _mm512_mul_ps(high_f32x16, cos_f32x16)));
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_rope_bf16_skylake(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                       nk_bf16_t *y, nk_size_t rows, nk_size_t head_count,
                                                       nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                       nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(depth % 2 == 0);
    nk_size_t const half_depth = depth / 2;
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_f32_t const *cos_row = cos + r * half_depth;
        nk_f32_t const *sin_row = sin + r * half_depth;
        nk_bf16_t const *x_row = (nk_bf16_t const *)((unsigned char const *)x + r * x_stride);
        nk_bf16_t *y_row = (nk_bf16_t *)((unsigned char *)y + r * y_stride);
        for (nk_size_t h = 0; h != head_count; ++h) {
            nk_bf16_t const *x_base = x_row + h * depth;
            nk_bf16_t *y_base = y_row + h * depth;
            nk_size_t i = 0;
            for (; i + 16 <= half_depth; i += 16) {
                nk_b512_vec_t low_vec, high_vec;
                nk_load_bf16x16_to_f32x16_skylake_(x_base + i, &low_vec);
                nk_load_bf16x16_to_f32x16_skylake_(x_base + i + half_depth, &high_vec);
                __m512 low_f32x16 = low_vec.zmm_ps;
                __m512 high_f32x16 = high_vec.zmm_ps;
                __m512 cos_f32x16 = _mm512_loadu_ps(cos_row + i), sin_f32x16 = _mm512_loadu_ps(sin_row + i);
                _mm256_storeu_si256((__m256i *)(y_base + i),
                                    nk_f32x16_to_bf16x16_skylake_(_mm512_fmsub_ps(
                                        low_f32x16, cos_f32x16, _mm512_mul_ps(high_f32x16, sin_f32x16))));
                _mm256_storeu_si256((__m256i *)(y_base + i + half_depth),
                                    nk_f32x16_to_bf16x16_skylake_(_mm512_fmadd_ps(
                                        low_f32x16, sin_f32x16, _mm512_mul_ps(high_f32x16, cos_f32x16))));
            }
            if (i < half_depth) {
                nk_size_t remaining = half_depth - i;
                __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFFu, (unsigned)remaining);
                nk_b512_vec_t low_vec, high_vec;
                nk_partial_load_bf16x16_to_f32x16_skylake_(x_base + i, &low_vec, remaining);
                nk_partial_load_bf16x16_to_f32x16_skylake_(x_base + i + half_depth, &high_vec, remaining);
                __m512 low_f32x16 = low_vec.zmm_ps;
                __m512 high_f32x16 = high_vec.zmm_ps;
                __m512 cos_f32x16 = _mm512_maskz_loadu_ps(mask_m16, cos_row + i);
                __m512 sin_f32x16 = _mm512_maskz_loadu_ps(mask_m16, sin_row + i);
                _mm256_mask_storeu_epi16((void *)(y_base + i), mask_m16,
                                         nk_f32x16_to_bf16x16_skylake_(_mm512_fmsub_ps(
                                             low_f32x16, cos_f32x16, _mm512_mul_ps(high_f32x16, sin_f32x16))));
                _mm256_mask_storeu_epi16((void *)(y_base + i + half_depth), mask_m16,
                                         nk_f32x16_to_bf16x16_skylake_(_mm512_fmadd_ps(
                                             low_f32x16, sin_f32x16, _mm512_mul_ps(high_f32x16, cos_f32x16))));
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_rope_e4m3_skylake(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
                                                       nk_e4m3_t *y, nk_size_t rows, nk_size_t head_count,
                                                       nk_size_t depth, nk_size_t x_stride, nk_size_t y_stride,
                                                       nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(depth % 2 == 0);
    nk_size_t const half_depth = depth / 2;
    for (nk_size_t r = 0; r != rows; ++r) {
        nk_f32_t const *cos_row = cos + r * half_depth;
        nk_f32_t const *sin_row = sin + r * half_depth;
        nk_e4m3_t const *x_row = (nk_e4m3_t const *)((unsigned char const *)x + r * x_stride);
        nk_e4m3_t *y_row = (nk_e4m3_t *)((unsigned char *)y + r * y_stride);
        for (nk_size_t h = 0; h != head_count; ++h) {
            nk_e4m3_t const *x_base = x_row + h * depth;
            nk_e4m3_t *y_base = y_row + h * depth;
            nk_size_t i = 0;
            for (; i + 16 <= half_depth; i += 16) {
                nk_b512_vec_t low_vec, high_vec;
                nk_load_e4m3x16_to_f32x16_skylake_(x_base + i, &low_vec);
                nk_load_e4m3x16_to_f32x16_skylake_(x_base + i + half_depth, &high_vec);
                __m512 low_f32x16 = low_vec.zmm_ps;
                __m512 high_f32x16 = high_vec.zmm_ps;
                __m512 cos_f32x16 = _mm512_loadu_ps(cos_row + i), sin_f32x16 = _mm512_loadu_ps(sin_row + i);
                _mm_storeu_si128((__m128i *)(y_base + i),
                                 nk_f32x16_to_e4m3x16_skylake_(
                                     _mm512_fmsub_ps(low_f32x16, cos_f32x16, _mm512_mul_ps(high_f32x16, sin_f32x16))));
                _mm_storeu_si128((__m128i *)(y_base + i + half_depth),
                                 nk_f32x16_to_e4m3x16_skylake_(
                                     _mm512_fmadd_ps(low_f32x16, sin_f32x16, _mm512_mul_ps(high_f32x16, cos_f32x16))));
            }
            if (i < half_depth) {
                nk_size_t remaining = half_depth - i;
                __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFFu, (unsigned)remaining);
                nk_b512_vec_t low_vec, high_vec;
                nk_partial_load_e4m3x16_to_f32x16_skylake_(x_base + i, &low_vec, remaining);
                nk_partial_load_e4m3x16_to_f32x16_skylake_(x_base + i + half_depth, &high_vec, remaining);
                __m512 low_f32x16 = low_vec.zmm_ps;
                __m512 high_f32x16 = high_vec.zmm_ps;
                __m512 cos_f32x16 = _mm512_maskz_loadu_ps(mask_m16, cos_row + i);
                __m512 sin_f32x16 = _mm512_maskz_loadu_ps(mask_m16, sin_row + i);
                _mm_mask_storeu_epi8((void *)(y_base + i), mask_m16,
                                     nk_f32x16_to_e4m3x16_skylake_(_mm512_fmsub_ps(
                                         low_f32x16, cos_f32x16, _mm512_mul_ps(high_f32x16, sin_f32x16))));
                _mm_mask_storeu_epi8((void *)(y_base + i + half_depth), mask_m16,
                                     nk_f32x16_to_e4m3x16_skylake_(_mm512_fmadd_ps(
                                         low_f32x16, sin_f32x16, _mm512_mul_ps(high_f32x16, cos_f32x16))));
            }
        }
    }
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SKYLAKE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_SKYLAKE_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_ATTENTION_SKYLAKE_H
