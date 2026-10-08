/**
 *  @file include/numkong/attention/genoa.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief Ragged attention for AVX-512 BF16-capable Genoa generation CPUs.
 *
 *  @sa include/numkong/attention.h
 *
 *  Backend for AVX512_BF16 machines without AMX: scores accumulate with @c vdpbf16ps straight from
 *  BF16 planes, doubling the per-instruction throughput over the widened-F32 Skylake kernels while
 *  halving the packed-KV footprint. The panel structure, online correction, and the base-2 softmax
 *  polynomial are shared with the rest of the family; the softmax and weighted-sum stages reuse the
 *  Skylake helpers — Genoa always implies the Skylake feature set, matching `dots/genoa.h`.
 *
 *  Packed payload per segment: K planes then V planes, `[key_value_head][position][channel]` in
 *  BF16 with channels zero-padded to a multiple of 32 — @c vdpbf16ps consumes value pairs, so
 *  full-width loops need no masks. E4M3 widens to BF16 during packing and Q staging via the Ice
 *  Lake converters, exactly like `dots/genoa.h`. Heads deeper than 256 channels narrow the query
 *  and accumulate scores in 256-channel chunks; the output accumulates in place.
 */
#ifndef NUMKONG_ATTENTION_GENOA_H
#define NUMKONG_ATTENTION_GENOA_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_GENOA

#include "numkong/attention/serial.h"  // shared packed-KV header/offsets, width-agnostic fallback
#include "numkong/attention/skylake.h" // `nk_attention_softmax_panel_skylake_`; Genoa implies Skylake
#include "numkong/cast/icelake.h"      // `nk_load_e4m3x32_to_bf16x32_icelake_`
#include "numkong/cast/skylake.h"      // `nk_bf16x16_to_f32x16_skylake_`
#include "numkong/reduce/skylake.h"    // `nk_reduce_add_f32x16_skylake_`, `nk_reduce_max_f32x16_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512bf16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512bf16", "f16c", "fma", "bmi", "bmi2")
#endif

enum {

    /** KV panel width in positions; the F32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_genoa_k_ = 512,

    /** Channels of the query row narrowed at once; deeper heads accumulate scores across chunks. */
    nk_attention_depth_chunk_genoa_k_ = 256,
};

/** Converts @p count contiguous elements to BF16 at @p destination, zero-filling to @p padded. */
typedef void (*nk_attention_narrow_genoa_t_)(void const *source, nk_bf16_t *destination, nk_size_t count,
                                             nk_size_t padded);

NUMKONG_INLINE void nk_attention_narrow_bf16_genoa_(void const *source, nk_bf16_t *destination, nk_size_t count,
                                                    nk_size_t padded) {
    nk_b512_vec_t chunk_vec;
    for (nk_size_t channel_idx = 0; channel_idx < padded; channel_idx += 32) {
        nk_size_t const chunk = channel_idx < count ? (count - channel_idx < 32 ? count - channel_idx : 32) : 0;
        nk_partial_load_b16x32_skylake_((nk_bf16_t const *)source + channel_idx, &chunk_vec, chunk);
        _mm512_storeu_si512(destination + channel_idx, chunk_vec.zmm);
    }
}

NUMKONG_INLINE void nk_attention_narrow_e4m3_genoa_(void const *source, nk_bf16_t *destination, nk_size_t count,
                                                    nk_size_t padded) {
    nk_b512_vec_t converted_vec;
    for (nk_size_t channel_idx = 0; channel_idx < padded; channel_idx += 32) {
        nk_size_t const chunk = channel_idx < count ? (count - channel_idx < 32 ? count - channel_idx : 32) : 0;
        nk_partial_load_e4m3x32_to_bf16x32_icelake_((char const *)source + channel_idx, &converted_vec, chunk);
        _mm512_storeu_si512(destination + channel_idx, converted_vec.zmm);
    }
}

/** Bytes of a pack of BF16 planes. */
NUMKONG_INLINE nk_size_t nk_attention_pack_size_genoa_(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_size_t token_count, nk_size_t segment_count) {
    nk_size_t const unit_bytes = nk_size_round_up_to_multiple_(depth, 32) * sizeof(nk_bf16_t);
    return nk_attention_pack_bound_(key_value_head_count, token_count, segment_count, 1, unit_bytes);
}

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_genoa(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes) {
    *bytes = nk_attention_pack_size_genoa_(key_value_head_count, depth, token_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_genoa(void const *key_value_packed, nk_size_t *heads,
                                                             nk_size_t *depth, nk_size_t *segments,
                                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_genoa_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_genoa(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes) {
    *bytes = nk_attention_pack_size_genoa_(key_value_head_count, depth, token_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_genoa(void const *key_value_packed, nk_size_t *heads,
                                                             nk_size_t *depth, nk_size_t *segments,
                                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_genoa_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
    return nk_success_k;
}

NUMKONG_INLINE void nk_attention_pack_genoa_(                                                  //
    void const *keys, void const *values, nk_size_t element_bytes,                             //
    nk_attention_narrow_genoa_t_ narrow,                                                       //
    nk_size_t key_value_head_count, nk_size_t depth,                                           //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, //
    nk_size_t key_stride, nk_size_t value_stride, void *key_value_packed,                      //
    nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 tasks_begin, 1, depth_padded * sizeof(nk_bf16_t), nk_cap_genoa_k);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_(segment_lengths[payload_segment], key_value_head_count,
                                                               1, depth_padded * sizeof(nk_bf16_t));
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const plane_values = position_count * depth_padded;
        nk_bf16_t *keys_plane = (nk_bf16_t *)(payload_base + payload_offset) + key_value_head_idx * plane_values;
        nk_bf16_t *values_plane = keys_plane + key_value_head_count * plane_values;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            narrow((char const *)keys + (position_first + position_idx) * key_stride +
                       key_value_head_idx * depth * element_bytes,
                   keys_plane + position_idx * depth_padded, depth, depth_padded);
            narrow((char const *)values + (position_first + position_idx) * value_stride +
                       key_value_head_idx * depth * element_bytes,
                   values_plane + position_idx * depth_padded, depth, depth_padded);
        }
    }
}

NUMKONG_API nk_status_t nk_attention_pack_bf16_genoa(                                                //
    nk_bf16_t const *keys, nk_bf16_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_attention_pack_genoa_(keys, values, sizeof(nk_bf16_t), &nk_attention_narrow_bf16_genoa_, key_value_head_count,
                             depth, segment_offsets, segment_lengths, segment_count, key_stride, value_stride,
                             key_value_packed, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_genoa(                                                //
    nk_e4m3_t const *keys, nk_e4m3_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_attention_pack_genoa_(keys, values, sizeof(nk_e4m3_t), &nk_attention_narrow_e4m3_genoa_, key_value_head_count,
                             depth, segment_offsets, segment_lengths, segment_count, key_stride, value_stride,
                             key_value_packed, tasks_begin, tasks_end);
    return nk_success_k;
}

/** Shared attention core over BF16 planes: per query row, panel-flash with an exact online
 *  correction; scores use @c vdpbf16ps with four KV rows in flight. */
NUMKONG_INLINE void nk_attention_packed_genoa_(                                                     //
    void const *queries, nk_size_t element_bytes, nk_attention_narrow_genoa_t_ narrow,              //
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
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)
    nk_size_t const panel_width = nk_attention_panel_genoa_k_;

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_bf16_t query_row[nk_attention_depth_chunk_genoa_k_];
    nk_align_(64) nk_f32_t scores[nk_attention_panel_genoa_k_];
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const depth_tail_m16 = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = segment_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_(query_end - query_first, position_count);
            nk_size_t const plane_values = position_count * depth_padded;
            nk_bf16_t const *keys_plane = (nk_bf16_t const *)(payload_base + payload_offsets[segment_idx]) +
                                          (head_idx / head_group_size) * plane_values;
            nk_bf16_t const *values_plane = keys_plane + key_value_head_count * plane_values;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                char const *query_source = (char const *)queries + (query_first + row_idx) * query_stride +
                                           head_idx * depth * element_bytes;
                nk_size_t const token = query_first + row_idx;
                nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx += 16)
                    _mm512_mask_storeu_ps(output_row + channel_idx,
                                          channel_idx + 16 <= depth ? (__mmask16)0xFFFF : depth_tail_m16,
                                          _mm512_setzero_ps());
                nk_f32_t running_max2 = NUMKONG_F32_MIN, running_sum = 0;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
                    // Scores: `vdpbf16ps` accumulation, four KV rows in flight.

                    nk_size_t position_idx = 0;
                    for (; position_idx < panel_length; position_idx += 16)
                        _mm512_store_ps(scores + position_idx, _mm512_setzero_ps());
                    for (nk_size_t chunk_start = 0; chunk_start < depth_padded;
                         chunk_start += nk_attention_depth_chunk_genoa_k_) {
                        nk_size_t const chunk_padded = depth_padded - chunk_start < nk_attention_depth_chunk_genoa_k_
                                                           ? depth_padded - chunk_start
                                                           : nk_attention_depth_chunk_genoa_k_;
                        narrow(query_source + chunk_start * element_bytes, query_row,
                               depth - chunk_start < chunk_padded ? depth - chunk_start : chunk_padded, chunk_padded);
                        for (position_idx = 0; position_idx < panel_length; position_idx += 4) {
                            // Rows past the panel repeat the last live one; scores go unread.
                            nk_size_t const live_last = panel_length - position_idx - 1;
                            nk_bf16_t const *keys_row0 = keys_plane + (panel_start + position_idx) * depth_padded +
                                                         chunk_start;
                            nk_bf16_t const *keys_row1 = keys_row0 + (live_last < 1 ? live_last : 1) * depth_padded;
                            nk_bf16_t const *keys_row2 = keys_row0 + (live_last < 2 ? live_last : 2) * depth_padded;
                            nk_bf16_t const *keys_row3 = keys_row0 + (live_last < 3 ? live_last : 3) * depth_padded;
                            __m512 accumulator0_f32x16 = _mm512_setzero_ps(), accumulator1_f32x16 = _mm512_setzero_ps();
                            __m512 accumulator2_f32x16 = _mm512_setzero_ps(), accumulator3_f32x16 = _mm512_setzero_ps();
                            for (nk_size_t channel_idx = 0; channel_idx < chunk_padded; channel_idx += 32) {
                                __m512bh const query_bf16x32 = (__m512bh)_mm512_load_si512(query_row + channel_idx);
                                accumulator0_f32x16 = _mm512_dpbf16_ps(
                                    accumulator0_f32x16, query_bf16x32,
                                    (__m512bh)_mm512_loadu_si512(keys_row0 + channel_idx));
                                accumulator1_f32x16 = _mm512_dpbf16_ps(
                                    accumulator1_f32x16, query_bf16x32,
                                    (__m512bh)_mm512_loadu_si512(keys_row1 + channel_idx));
                                accumulator2_f32x16 = _mm512_dpbf16_ps(
                                    accumulator2_f32x16, query_bf16x32,
                                    (__m512bh)_mm512_loadu_si512(keys_row2 + channel_idx));
                                accumulator3_f32x16 = _mm512_dpbf16_ps(
                                    accumulator3_f32x16, query_bf16x32,
                                    (__m512bh)_mm512_loadu_si512(keys_row3 + channel_idx));
                            }
                            scores[position_idx + 0] += nk_reduce_add_f32x16_skylake_(accumulator0_f32x16);
                            scores[position_idx + 1] += nk_reduce_add_f32x16_skylake_(accumulator1_f32x16);
                            scores[position_idx + 2] += nk_reduce_add_f32x16_skylake_(accumulator2_f32x16);
                            scores[position_idx + 3] += nk_reduce_add_f32x16_skylake_(accumulator3_f32x16);
                        }
                    }

                    nk_f32_t const correction = nk_attention_softmax_panel_skylake_(scores, panel_length, scale2,
                                                                                    &running_max2, &running_sum);

                    // O = O · correction + Σ weight · widened V-row over the panel.

                    __m512 const correction_f32x16 = _mm512_set1_ps(correction);
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx += 16) {
                        __mmask16 const channel_m16 = channel_idx + 16 <= depth ? (__mmask16)0xFFFF : depth_tail_m16;
                        _mm512_mask_storeu_ps(
                            output_row + channel_idx, channel_m16,
                            _mm512_mul_ps(_mm512_maskz_loadu_ps(channel_m16, output_row + channel_idx),
                                          correction_f32x16));
                    }
                    for (position_idx = 0; position_idx < panel_length; position_idx++) {
                        __m512 const weight_f32x16 = _mm512_set1_ps(scores[position_idx]);
                        nk_bf16_t const *values_row = values_plane + (panel_start + position_idx) * depth_padded;
                        for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx += 16) {
                            __mmask16 const channel_m16 = channel_idx + 16 <= depth ? (__mmask16)0xFFFF
                                                                                    : depth_tail_m16;
                            __m512 const v_f32x16 = nk_bf16x16_to_f32x16_skylake_(
                                _mm256_loadu_si256((__m256i const *)(values_row + channel_idx)));
                            _mm512_mask_storeu_ps(
                                output_row + channel_idx, channel_m16,
                                _mm512_fmadd_ps(weight_f32x16, v_f32x16,
                                                _mm512_maskz_loadu_ps(channel_m16, output_row + channel_idx)));
                        }
                    }
                }

                __m512 const inverse_sum_f32x16 = _mm512_set1_ps(running_sum > 0 ? 1 / running_sum : 0);
                for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx += 16) {
                    __mmask16 const channel_m16 = channel_idx + 16 <= depth ? (__mmask16)0xFFFF : depth_tail_m16;
                    _mm512_mask_storeu_ps(output_row + channel_idx, channel_m16,
                                          _mm512_mul_ps(_mm512_maskz_loadu_ps(channel_m16, output_row + channel_idx),
                                                        inverse_sum_f32x16));
                }
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_(running_max2, running_sum);
            }
        }
    }
}

NUMKONG_API nk_status_t nk_attention_packed_bf16_genoa(                          //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_genoa_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_genoa_(queries, sizeof(nk_bf16_t), &nk_attention_narrow_bf16_genoa_, key_value_packed, output,
                               log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_stride,
                               output_stride, scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_e4m3_genoa(                          //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_genoa_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_genoa_(queries, sizeof(nk_e4m3_t), &nk_attention_narrow_e4m3_genoa_, key_value_packed, output,
                               log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_stride,
                               output_stride, scale, band, tasks_begin, tasks_end);
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

#endif // NUMKONG_TARGET_GENOA
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_ATTENTION_GENOA_H
