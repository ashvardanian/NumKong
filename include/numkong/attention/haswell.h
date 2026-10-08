/**
 *  @file include/numkong/attention/haswell.h
 *  @author Ash Vardanian
 *  @date July 6, 2026
 *  @brief Ragged attention for AVX2 Haswell generation CPUs.
 *
 *  @sa include/numkong/attention.h
 *
 *  Compatibility backend for AVX2/FMA machines. Storage follows the `dots/haswell.h` conventions
 *  exactly: both BF16 and E4M3 stay in their source encoding at rest — packing is a raw strided-row
 *  copy — and every value widens to F32 on the fly inside the compute loops, 8-wide on the FMA
 *  ports. That keeps the packed KV cache at 2 or even 1 byte per scalar, trading a couple of
 *  unpacking ops per FMA for halved-to-quartered streaming traffic, the same call the GEMM family
 *  made for this ISA.
 *
 *  The panel structure matches the family design — per query row, KV is swept in panels with an
 *  exact online correction and the family's base-2 degree-7 softmax polynomial; scores keep four KV
 *  rows in flight. Packed payload per segment: K planes then V planes,
 *  `[key_value_head][position][channel]` in the source dtype with channels zero-padded to a
 *  multiple of 8. `depth > 256` routes to the width-agnostic serial kernel.
 */
#ifndef NUMKONG_ATTENTION_HASWELL_H
#define NUMKONG_ATTENTION_HASWELL_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_HASWELL_

#include "numkong/attention/serial.h" // shared packed-KV header/offsets, width-agnostic fallback
#include "numkong/cast/haswell.h"     // widening helpers like `nk_bf16x8_to_f32x8_haswell_`
#include "numkong/each/haswell.h"     // `nk_exp2_f32x8_haswell_`, `nk_exp2_u8_i32x8_haswell_`
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

    /** Widest head this backend handles in registers; larger heads route to the serial kernel. */
    nk_attention_max_depth_haswell_k_ = 256,
};

/** Widens 8 raw plane scalars (BF16 or E4M3 at rest) to F32 inside the hot loops. */
typedef __m256 (*nk_attention_load_haswell_t_)(void const *plane_chunk);

NUMKONG_INLINE __m256 nk_attention_load_bf16x8_haswell_(void const *plane_chunk) {
    nk_b256_vec_t widened;
    nk_load_bf16x8_to_f32x8_haswell_(plane_chunk, &widened);
    return widened.ymm_ps;
}

NUMKONG_INLINE __m256 nk_attention_load_e4m3x8_haswell_(void const *plane_chunk) {
    nk_b256_vec_t widened;
    nk_load_e4m3x8_to_f32x8_haswell_(plane_chunk, &widened);
    return widened.ymm_ps;
}

/** Widens @p count raw query elements to F32 into @p destination, zero-filling to @p padded. */
typedef void (*nk_attention_widen_haswell_t_)(void const *source, nk_f32_t *destination, nk_size_t count,
                                              nk_size_t padded);

NUMKONG_INLINE void nk_attention_widen_bf16_haswell_(void const *source, nk_f32_t *destination, nk_size_t count,
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

NUMKONG_INLINE void nk_attention_widen_e4m3_haswell_(void const *source, nk_f32_t *destination, nk_size_t count,
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

/** Bytes of a pack whose planes keep the source encoding, like the dots family, past the serial
 *  depths in F32. */
NUMKONG_INLINE nk_size_t nk_attention_pack_size_haswell_(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t element_bytes) {
    nk_size_t const unit_bytes = depth > nk_attention_max_depth_haswell_k_
                                     ? depth * sizeof(nk_f32_t)
                                     : nk_size_round_up_to_multiple_(depth, 8) * element_bytes;
    return nk_attention_pack_bound_(key_value_head_count, token_count, segment_count, 1, unit_bytes);
}

/** Copies @p count contiguous 16-bit values into @p destination, zeroing it up to @p padded. */
NUMKONG_INLINE void nk_attention_copy_row_b16_haswell_(char const *source, void *destination, nk_size_t count,
                                                       nk_size_t padded) {
    nk_b256_vec_t chunk_vec;
    for (nk_size_t index = 0; index < padded; index += 16) {
        nk_size_t const chunk = index < count ? (count - index < 16 ? count - index : 16) : 0;
        nk_partial_load_b16x16_haswell_(source + index * sizeof(nk_u16_t), &chunk_vec, chunk);
        nk_partial_store_b16x16_haswell_(&chunk_vec, (nk_u16_t *)destination + index,
                                         padded - index < 16 ? padded - index : 16);
    }
}

/** Copies @p count contiguous bytes into @p destination, zeroing it up to @p padded. */
NUMKONG_INLINE void nk_attention_copy_row_b8_haswell_(char const *source, void *destination, nk_size_t count,
                                                      nk_size_t padded) {
    nk_b256_vec_t chunk_vec;
    for (nk_size_t index = 0; index < padded; index += 32) {
        nk_size_t const chunk = index < count ? (count - index < 32 ? count - index : 32) : 0;
        nk_partial_load_b8x32_haswell_(source + index, &chunk_vec, chunk);
        nk_partial_store_b8x32_haswell_(&chunk_vec, (nk_u8_t *)destination + index,
                                        padded - index < 32 ? padded - index : 32);
    }
}

/** Raw row repack: source encoding is preserved, tails zero-padded. */
NUMKONG_INLINE void nk_attention_pack_haswell_(                                                //
    void const *keys, void const *values, nk_size_t element_bytes,                             //
    nk_size_t key_value_head_count, nk_size_t depth,                                           //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, //
    nk_size_t key_stride, nk_size_t value_stride, void *key_value_packed,                      //
    nk_size_t tasks_begin, nk_size_t tasks_end) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const padded_row_bytes = depth_padded * element_bytes;
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 tasks_begin, 1, padded_row_bytes, nk_cap_haswell_k);
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
                                                               1, padded_row_bytes);
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const plane_bytes = position_count * padded_row_bytes;
        char *keys_plane = payload_base + payload_offset + key_value_head_idx * plane_bytes;
        char *values_plane = keys_plane + key_value_head_count * plane_bytes;
        for (nk_size_t position_idx = 0; position_idx < position_count; position_idx++) {
            char const *keys_row = (char const *)keys + (position_first + position_idx) * key_stride +
                                   key_value_head_idx * depth * element_bytes;
            char const *values_row = (char const *)values + (position_first + position_idx) * value_stride +
                                     key_value_head_idx * depth * element_bytes;
            char *keys_destination = keys_plane + position_idx * padded_row_bytes;
            char *values_destination = values_plane + position_idx * padded_row_bytes;
            if (element_bytes == sizeof(nk_u16_t)) {
                nk_attention_copy_row_b16_haswell_(keys_row, keys_destination, depth, depth_padded);
                nk_attention_copy_row_b16_haswell_(values_row, values_destination, depth, depth_padded);
            }
            else {
                nk_attention_copy_row_b8_haswell_(keys_row, keys_destination, depth, depth_padded);
                nk_attention_copy_row_b8_haswell_(values_row, values_destination, depth, depth_padded);
            }
        }
    }
}

/** Shared attention core over raw-encoded planes: per query row, panel-flash with an exact online
 *  correction; scores keep four KV rows in flight, widening in-loop. */
NUMKONG_INLINE void nk_attention_packed_haswell_(                                                   //
    void const *queries, nk_size_t element_bytes, nk_attention_widen_haswell_t_ widen,              //
    nk_attention_load_haswell_t_ load,                                                              //
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
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const plane_row_bytes = depth_padded * element_bytes;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // softmax(x) = softmax₂(x · log₂e)
    nk_size_t const panel_width = nk_attention_panel_haswell_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_f32_t query_row[nk_attention_max_depth_haswell_k_];
    nk_align_(64) nk_f32_t output_row[nk_attention_max_depth_haswell_k_];
    nk_align_(64) nk_f32_t scores[nk_attention_panel_haswell_k_];

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
            nk_size_t const plane_bytes = position_count * plane_row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                     (head_idx / head_group_size) * plane_bytes;
            char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            for (nk_size_t row_idx = row_begin; row_idx < row_end; row_idx++) {
                widen((char const *)queries + (query_first + row_idx) * query_stride + head_idx * depth * element_bytes,
                      query_row, depth, depth_padded);
                for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
                    _mm256_store_ps(output_row + channel_idx, _mm256_setzero_ps());
                nk_f32_t running_max2 = NUMKONG_F32_MIN, running_sum = 0;
                nk_size_t key_begin, key_end;
                nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row_idx, position_count, &key_begin,
                                            &key_end);

                for (nk_size_t panel_start = key_begin; panel_start < key_end; panel_start += panel_width) {
                    nk_size_t const panel_length = (panel_start + panel_width <= key_end) ? panel_width
                                                                                          : (key_end - panel_start);
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

                    // Panel max, online correction, exp2 in place, scalar tail on serial exp2.

                    __m256 max_f32x8 = _mm256_set1_ps(NUMKONG_F32_MIN);
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
                        nk_exp2_f32x8_haswell_(_mm256_set1_ps(running_max2 - new_max2)));
                    running_max2 = new_max2;

                    __m256 const scale2_f32x8 = _mm256_set1_ps(scale2);
                    __m256 const max2_f32x8 = _mm256_set1_ps(new_max2);
                    __m256 sum_f32x8 = _mm256_setzero_ps();
                    for (position_idx = 0; position_idx + 8 <= panel_length; position_idx += 8) {
                        __m256 weights_f32x8 = nk_exp2_f32x8_haswell_(
                            _mm256_fmsub_ps(_mm256_loadu_ps(scores + position_idx), scale2_f32x8, max2_f32x8));
                        sum_f32x8 = _mm256_add_ps(sum_f32x8, weights_f32x8);
                        _mm256_storeu_ps(scores + position_idx, weights_f32x8);
                    }
                    if (position_idx < panel_length) { // masked tail keeps the vector rounding mode end-to-end
                        __m256i const lane_index_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                        __m256i const tail_mask_i32x8 = _mm256_cmpgt_epi32(
                            _mm256_set1_epi32((int)(panel_length - position_idx)), lane_index_i32x8);
                        __m256 weights_f32x8 = nk_exp2_f32x8_haswell_(_mm256_fmsub_ps(
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
                                _mm256_fmadd_ps(weight_f32x8, load(values_row + channel_idx * element_bytes),
                                                _mm256_load_ps(output_row + channel_idx)));
                    }
                }

                nk_f32_t const inverse_sum = running_sum > 0 ? 1 / running_sum : 0;
                __m256 const inverse_sum_f32x8 = _mm256_set1_ps(inverse_sum);
                nk_size_t const token = query_first + row_idx;
                nk_f32_t *destination = output + token * output_stride_floats + head_idx * depth;
                nk_size_t channel_idx = 0;
                for (; channel_idx + 8 <= depth; channel_idx += 8)
                    _mm256_storeu_ps(destination + channel_idx,
                                     _mm256_mul_ps(_mm256_load_ps(output_row + channel_idx), inverse_sum_f32x8));
                for (; channel_idx < depth; channel_idx++)
                    destination[channel_idx] = output_row[channel_idx] * inverse_sum;
                if (log_sum_exp)
                    log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_(running_max2, running_sum);
            }
        }
    }
}

/** Register 8×8 transpose of 16-bit elements (128-bit hierarchical unpack). Given 8 rows (each 8
 *  words), returns the 8 columns. Used at pack time to turn 8 position rows into the drain-free K
 *  tiles: each column is one depth pair (two channels) of eight KV positions. The 128-bit unpacks
 *  never cross lanes, so the columns emerge in natural pair order. */
NUMKONG_INLINE void nk_attention_transpose_i16x8x8_haswell_(__m128i const rows_i16x8[8], __m128i columns_i16x8[8]) {
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
 *  @brief Drain-free exact I32 scores for a query block over one panel.
 *
 *  K is packed in tiles of 8 positions, each holding depth pairs of 8 lanes by 2 channels, so a
 *  128-bit load widened to I16 holds two channels of eight KV positions — one score per lane. Each
 *  query's two channels broadcast as one dword and one VPMADDWD advances eight KV positions by two
 *  channels; accumulating over the depth pairs leaves eight exact scores per lane with no
 *  transpose. Eight queries share each K load. Writes a @b [block_rows,panel] score block, rows
 *  @c nk_attention_panel_haswell_k_ apart.
 */
NUMKONG_INLINE void nk_attention_score_block_i8_haswell_(nk_i16_t const *queries_i16, nk_size_t block_rows,
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

/** Streaming base-2 softmax over one panel for a single query row, entirely in integer arithmetic:
 *  the row max is an exact @c _mm256_max_epi32 over live columns, weights come from the integer
 *  i-exp over (score − max) · scale₂ in Q15, and the weight sum accumulates in I32. Only the online
 *  correction 2^((m_old − m_new) · scale₂) stays in F32. Returns that correction. */
NUMKONG_INLINE nk_f32_t nk_attention_softmax_panel_i8_haswell_(nk_i32_t const *scores, nk_size_t panel_length,
                                                               nk_f32_t scale2, nk_i32_t scale_fixed,
                                                               nk_i32_t delta_floor, nk_i32_t *running_max,
                                                               nk_f32_t *running_sum, nk_u8_t *weights) {
    __m256i max_i32x8 = _mm256_set1_epi32(NUMKONG_I32_MIN);
    nk_size_t position_idx = 0;
    for (; position_idx + 8 <= panel_length; position_idx += 8)
        max_i32x8 = _mm256_max_epi32(max_i32x8, _mm256_loadu_si256((__m256i const *)(scores + position_idx)));
    nk_i32_t panel_max = nk_reduce_max_i32x8_haswell_(max_i32x8);
    for (; position_idx < panel_length; position_idx++)
        if (scores[position_idx] > panel_max) panel_max = scores[position_idx];
    nk_i32_t const new_max = *running_max > panel_max ? *running_max : panel_max;
    nk_f32_t const correction = _mm256_cvtss_f32(
        nk_exp2_f32x8_haswell_(_mm256_set1_ps(((nk_f32_t)*running_max - (nk_f32_t)new_max) * scale2)));
    *running_max = new_max;

    __m256i const new_max_i32x8 = _mm256_set1_epi32(new_max);
    __m256i const scale_fixed_i32x8 = _mm256_set1_epi32(scale_fixed);
    __m256i const delta_floor_i32x8 = _mm256_set1_epi32(delta_floor);
    __m128i const zero_u8x16 = _mm_setzero_si128();
    __m256i sum_i32x8 = _mm256_setzero_si256();
    __m256i sum2_i32x8 = _mm256_setzero_si256();
    position_idx = 0;
    // Two independent 8-lane groups per iteration keep the vpmulld-heavy i-exp chains off each other's
    // critical path (port-0 latency hiding); a trailing single group + masked tail follow unchanged.
    for (; position_idx + 16 <= panel_length; position_idx += 16) {
        __m256i const delta0_i32x8 = _mm256_max_epi32(
            _mm256_sub_epi32(_mm256_loadu_si256((__m256i const *)(scores + position_idx)), new_max_i32x8),
            delta_floor_i32x8);
        __m256i const delta1_i32x8 = _mm256_max_epi32(
            _mm256_sub_epi32(_mm256_loadu_si256((__m256i const *)(scores + position_idx + 8)), new_max_i32x8),
            delta_floor_i32x8);
        __m256i const weight0_i32x8 = nk_exp2_u8_i32x8_haswell_(_mm256_mullo_epi32(delta0_i32x8, scale_fixed_i32x8));
        __m256i const weight1_i32x8 = nk_exp2_u8_i32x8_haswell_(_mm256_mullo_epi32(delta1_i32x8, scale_fixed_i32x8));
        sum_i32x8 = _mm256_add_epi32(sum_i32x8, weight0_i32x8);
        sum2_i32x8 = _mm256_add_epi32(sum2_i32x8, weight1_i32x8);
        __m128i const weight0_i16x8 = _mm_packus_epi32(_mm256_castsi256_si128(weight0_i32x8),
                                                       _mm256_extracti128_si256(weight0_i32x8, 1));
        _mm_storel_epi64((__m128i *)(weights + position_idx), _mm_packus_epi16(weight0_i16x8, zero_u8x16));
        __m128i const weight1_i16x8 = _mm_packus_epi32(_mm256_castsi256_si128(weight1_i32x8),
                                                       _mm256_extracti128_si256(weight1_i32x8, 1));
        _mm_storel_epi64((__m128i *)(weights + position_idx + 8), _mm_packus_epi16(weight1_i16x8, zero_u8x16));
    }
    sum_i32x8 = _mm256_add_epi32(sum_i32x8, sum2_i32x8);
    for (; position_idx + 8 <= panel_length; position_idx += 8) {
        __m256i const delta_i32x8 = _mm256_max_epi32(
            _mm256_sub_epi32(_mm256_loadu_si256((__m256i const *)(scores + position_idx)), new_max_i32x8),
            delta_floor_i32x8);
        __m256i const weight_i32x8 = nk_exp2_u8_i32x8_haswell_(_mm256_mullo_epi32(delta_i32x8, scale_fixed_i32x8));
        sum_i32x8 = _mm256_add_epi32(sum_i32x8, weight_i32x8);
        __m128i const weight_i16x8 = _mm_packus_epi32(_mm256_castsi256_si128(weight_i32x8),
                                                      _mm256_extracti128_si256(weight_i32x8, 1));
        _mm_storel_epi64((__m128i *)(weights + position_idx), _mm_packus_epi16(weight_i16x8, zero_u8x16));
    }
    if (position_idx < panel_length) { // masked tail: padded lanes forced to a zero weight
        __m256i const lane_index_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        __m256i const tail_mask_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32((int)(panel_length - position_idx)),
                                                           lane_index_i32x8);
        __m256i const scores_i32x8 = _mm256_maskload_epi32((int const *)(scores + position_idx), tail_mask_i32x8);
        __m256i const delta_i32x8 = _mm256_max_epi32(_mm256_sub_epi32(scores_i32x8, new_max_i32x8), delta_floor_i32x8);
        __m256i weight_i32x8 = nk_exp2_u8_i32x8_haswell_(_mm256_mullo_epi32(delta_i32x8, scale_fixed_i32x8));
        weight_i32x8 = _mm256_and_si256(weight_i32x8, tail_mask_i32x8);
        sum_i32x8 = _mm256_add_epi32(sum_i32x8, weight_i32x8);
        __m128i const weight_i16x8 = _mm_packus_epi32(_mm256_castsi256_si128(weight_i32x8),
                                                      _mm256_extracti128_si256(weight_i32x8, 1));
        _mm_storel_epi64((__m128i *)(weights + position_idx), _mm_packus_epi16(weight_i16x8, zero_u8x16));
    }
    *running_sum = *running_sum * correction + (nk_f32_t)nk_reduce_add_i32x8_haswell_(sum_i32x8);
    return correction;
}

/** O = O · correction + Σ w̃ · widened V-row for one query row over one panel; V stays token-major
 *  and widens on the fly, exact-zero weights skipped like the serial reference. The skip is
 *  load-bearing here: this P × V is a scalar-broadcast FP FMA per position, and U8 softmax weights
 *  are sparse, most positions quantizing to zero, so the full dense sweep would run ~3× slower. */
NUMKONG_INLINE void nk_attention_weighted_sum_panel_i8_haswell_(nk_u8_t const *weights, char const *values_plane,
                                                                nk_size_t panel_start, nk_size_t panel_length,
                                                                nk_size_t depth_padded, nk_f32_t correction,
                                                                nk_f32_t *output_row) {
    __m256 const correction_f32x8 = _mm256_set1_ps(correction);
    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 8)
        _mm256_store_ps(output_row + channel_idx,
                        _mm256_mul_ps(_mm256_load_ps(output_row + channel_idx), correction_f32x8));
    for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx++) {
        nk_u8_t const weight_u8 = weights[position_idx];
        if (weight_u8 == 0) continue; // sparse U8 weights: skipping dead positions is the kernel's main lever
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

/** I8 attention core: 8-row query blocks share each K panel, panels outside @p band are skipped,
 *  and rows of a panel the band's edge crosses softmax only their own visible slice. */
NUMKONG_INLINE void nk_attention_packed_i8_haswell_(                                                //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp,  //
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
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
    nk_size_t const depth_padded16 = nk_size_round_up_to_multiple_(depth_padded, 16);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;                // softmax(x) = softmax₂(x · log₂e)
    nk_i32_t const scale_fixed = (nk_i32_t)(scale2 * 32768.0f + 0.5f); // Q15 scale for the integer exponential
    nk_i32_t const delta_floor = // the score delta below which every weight quantizes to zero (2^t · 255 + 0.5 < 1)
        scale_fixed > 0 ? -(nk_i32_t)((10u << 15) / (nk_u32_t)scale_fixed) - 1 : 0;
    nk_size_t const panel_width = nk_attention_panel_haswell_k_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_i16_t queries_i16[8 * nk_attention_max_depth_haswell_k_];
    nk_align_(64) nk_f32_t output_rows[8 * nk_attention_max_depth_haswell_k_];
    nk_align_(64) nk_i32_t scores[8 * nk_attention_panel_haswell_k_];
    nk_align_(64) nk_u8_t weights[nk_attention_panel_haswell_k_];

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
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 8);
            nk_size_t const plane_bytes = position_count_padded * depth_padded;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] +
                                     (head_idx / head_group_size) * plane_bytes;
            char const *values_plane = keys_plane + key_value_head_count * plane_bytes;

            // Blocks keep the segment's 8-row grid, as panels start at each block's first keys.
            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t query_block = row_begin / 8 * 8; query_block < row_end; query_block += 8) {
                nk_size_t const block_rows = (query_block + 8 <= row_count) ? 8 : (row_count - query_block);
                // Widen up to eight query rows to I16; zero the block tail to keep scores harmless.
                for (nk_size_t block_row = 0; block_row < 8; block_row++) {
                    nk_i16_t *query_i16 = queries_i16 + block_row * nk_attention_max_depth_haswell_k_;
                    nk_size_t channel_idx = 0;
                    if (block_row < block_rows) {
                        nk_i8_t const *query_row = (nk_i8_t const *)((char const *)queries +
                                                                     (query_first + query_block + block_row) *
                                                                         query_stride) +
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
                nk_i32_t running_max[8];
                nk_f32_t running_sum[8];
                nk_size_t key_begins[8], key_ends[8];
                nk_i64_t const block_position = first_position + (nk_i64_t)query_block;
                for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                    running_max[block_row] = NUMKONG_I32_MIN;
                    running_sum[block_row] = 0;
                    nk_diagonal_band_row_range_(band, block_position + (nk_i64_t)block_row, position_count,
                                                &key_begins[block_row], &key_ends[block_row]);
                }
                nk_size_t const block_key_begin = key_begins[0] / 8 * 8, block_key_end = key_ends[block_rows - 1];

                // Row ranges grow monotonically, so the block spans the first begin to last end.
                for (nk_size_t panel_start = block_key_begin; panel_start < block_key_end; panel_start += panel_width) {
                    nk_size_t const panel_end = (panel_start + panel_width <= block_key_end) ? panel_start + panel_width
                                                                                             : block_key_end;
                    nk_diagonal_band_coverage_t const coverage = nk_diagonal_band_tile_coverage_(
                        band, block_position, block_rows, panel_start, panel_end - panel_start);
                    if (coverage == nk_diagonal_band_outside_k) continue;
                    nk_attention_score_block_i8_haswell_(queries_i16, block_rows, keys_plane, panel_start,
                                                         panel_end - panel_start, depth_padded, scores);
                    for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                        nk_size_t visible_begin = panel_start, visible_end = panel_end;
                        if (coverage == nk_diagonal_band_crossing_k) {
                            if (key_begins[block_row] > visible_begin) visible_begin = key_begins[block_row];
                            if (key_ends[block_row] < visible_end) visible_end = key_ends[block_row];
                            if (visible_begin >= visible_end) continue; // running state unchanged
                        }
                        nk_f32_t const correction = nk_attention_softmax_panel_i8_haswell_(
                            scores + block_row * panel_width + (visible_begin - panel_start),
                            visible_end - visible_begin, scale2, scale_fixed, delta_floor, &running_max[block_row],
                            &running_sum[block_row], weights);
                        nk_attention_weighted_sum_panel_i8_haswell_(
                            weights, values_plane, visible_begin, visible_end - visible_begin, depth_padded, correction,
                            output_rows + block_row * nk_attention_max_depth_haswell_k_);
                    }
                }

                for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                    if (query_block + block_row < row_begin || query_block + block_row >= row_end) continue;
                    nk_f32_t const *output_row = output_rows + block_row * nk_attention_max_depth_haswell_k_;
                    nk_f32_t const inverse_sum = running_sum[block_row] > 0 ? 1 / running_sum[block_row] : 0;
                    __m256 const inverse_sum_f32x8 = _mm256_set1_ps(inverse_sum);
                    nk_size_t const token = query_first + query_block + block_row;
                    nk_f32_t *destination = output + token * output_stride_floats + head_idx * depth;
                    nk_size_t channel_idx = 0;
                    for (; channel_idx + 8 <= depth; channel_idx += 8)
                        _mm256_storeu_ps(destination + channel_idx,
                                         _mm256_mul_ps(_mm256_load_ps(output_row + channel_idx), inverse_sum_f32x8));
                    for (; channel_idx < depth; channel_idx++)
                        destination[channel_idx] = output_row[channel_idx] * inverse_sum;
                    if (log_sum_exp)
                        log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_(
                            (nk_f32_t)running_max[block_row] * scale2, running_sum[block_row] / 255.0f);
                }
            }
        }
    }
}

#if NUMKONG_TARGET_HASWELL

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes) {
    *bytes = nk_attention_pack_size_haswell_(key_value_head_count, depth, token_count, segment_count,
                                             sizeof(nk_bf16_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_haswell(void const *key_value_packed, nk_size_t *heads,
                                                               nk_size_t *depth, nk_size_t *segments,
                                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_haswell_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                            nk_size_t token_count, nk_size_t segment_count,
                                                            nk_size_t *bytes) {
    *bytes = nk_attention_pack_size_haswell_(key_value_head_count, depth, token_count, segment_count,
                                             sizeof(nk_e4m3_t));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_haswell(void const *key_value_packed, nk_size_t *heads,
                                                               nk_size_t *depth, nk_size_t *segments,
                                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_haswell_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_bf16_haswell(                                              //
    nk_bf16_t const *keys, nk_bf16_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth > nk_attention_max_depth_haswell_k_)
        nk_attention_pack_serial_(keys, values, sizeof(nk_bf16_t), &nk_attention_load_bf16_serial_,
                                  key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                                  key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, nk_cap_haswell_k);
    else
        nk_attention_pack_haswell_(keys, values, sizeof(nk_bf16_t), key_value_head_count, depth, segment_offsets,
                                   segment_lengths, segment_count, key_stride, value_stride, key_value_packed,
                                   tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_haswell(                                              //
    nk_e4m3_t const *keys, nk_e4m3_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth > nk_attention_max_depth_haswell_k_)
        nk_attention_pack_serial_(keys, values, sizeof(nk_e4m3_t), &nk_attention_load_e4m3_serial_,
                                  key_value_head_count, depth, segment_offsets, segment_lengths, segment_count,
                                  key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, nk_cap_haswell_k);
    else
        nk_attention_pack_haswell_(keys, values, sizeof(nk_e4m3_t), key_value_head_count, depth, segment_offsets,
                                   segment_lengths, segment_count, key_stride, value_stride, key_value_packed,
                                   tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_bf16_haswell(                        //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_haswell_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_haswell_k_)
        nk_attention_serial_(queries, sizeof(nk_bf16_t), &nk_attention_load_bf16_serial_, key_value_packed, output,
                             log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_stride,
                             output_stride, scale, band, tasks_begin, tasks_end);
    else
        nk_attention_packed_haswell_(queries, sizeof(nk_bf16_t), &nk_attention_widen_bf16_haswell_,
                                     &nk_attention_load_bf16x8_haswell_, key_value_packed, output, log_sum_exp,
                                     head_count, key_value_head_count, depth, query_offsets, query_stride,
                                     output_stride, scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_e4m3_haswell(                        //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_haswell_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_haswell_k_)
        nk_attention_serial_(queries, sizeof(nk_e4m3_t), &nk_attention_load_e4m3_serial_, key_value_packed, output,
                             log_sum_exp, head_count, key_value_head_count, depth, query_offsets, query_stride,
                             output_stride, scale, band, tasks_begin, tasks_end);
    else
        nk_attention_packed_haswell_(queries, sizeof(nk_e4m3_t), &nk_attention_widen_e4m3_haswell_,
                                     &nk_attention_load_e4m3x8_haswell_, key_value_packed, output, log_sum_exp,
                                     head_count, key_value_head_count, depth, query_offsets, query_stride,
                                     output_stride, scale, band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_i8_haswell(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes) {
    // K in drain-free tiles and V token-major both pad positions to the 8-tile
    *bytes = depth > nk_attention_max_depth_haswell_k_
                 ? nk_attention_pack_bound_(key_value_head_count, token_count, segment_count, 1, depth)
                 : nk_attention_pack_bound_(key_value_head_count, token_count, segment_count, 8,
                                            nk_size_round_up_to_multiple_(depth, 8));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_haswell(void const *key_value_packed, nk_size_t *heads,
                                                             nk_size_t *depth, nk_size_t *segments,
                                                             nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_haswell_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_i8_haswell(                                              //
    nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count, nk_size_t depth,   //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                              //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride, void *key_value_packed, //
    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth > nk_attention_max_depth_haswell_k_) {
        nk_attention_pack_i8_serial_(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                     segment_count, key_stride, value_stride, key_value_packed, tasks_begin, tasks_end,
                                     nk_cap_haswell_k);
    }
    else {
        nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 8);
        nk_size_t const depth_pairs = depth_padded / 2;
        nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                     tasks_begin, 8, depth_padded, nk_cap_haswell_k);
        char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                             nk_attention_pack_directory_size_(segment_count);

        nk_size_t const total_tasks = segment_count * key_value_head_count;
        if (tasks_begin >= total_tasks) return nk_success_k;
        if (tasks_end > total_tasks) tasks_end = total_tasks;

        nk_size_t payload_segment = 0;
        nk_u64_t payload_offset = 0;
        for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
            nk_size_t const segment_idx = task_idx / key_value_head_count,
                            key_value_head_idx = task_idx % key_value_head_count;
            for (; payload_segment < segment_idx; payload_segment++)
                payload_offset += nk_attention_pack_segment_bytes_(segment_lengths[payload_segment],
                                                                   key_value_head_count, 8, depth_padded);
            nk_size_t const position_count = segment_lengths[segment_idx];
            if (position_count == 0) continue;
            nk_size_t const position_first = segment_offsets[segment_idx];
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 8);
            nk_size_t const plane_bytes = position_count_padded * depth_padded;
            char *keys_plane = payload_base + payload_offset + key_value_head_idx * plane_bytes;
            char *values_plane = keys_plane + key_value_head_count * plane_bytes;

            // Eight zero-padded rows land in an aligned stage for each tile's register transpose.
            nk_align_(64) nk_i8_t stage[8 * nk_attention_max_depth_haswell_k_];
            for (nk_size_t tile_idx = 0; tile_idx * 8 < position_count_padded; tile_idx++) {
                for (nk_size_t lane_idx = 0; lane_idx < 8; lane_idx++) {
                    nk_size_t const position_idx = tile_idx * 8 + lane_idx;
                    nk_attention_copy_row_b8_haswell_(
                        (char const *)keys + (position_first + position_idx) * key_stride + key_value_head_idx * depth,
                        stage + lane_idx * depth_padded, position_idx < position_count ? depth : 0, depth_padded);
                }
                char *keys_tile = keys_plane + tile_idx * depth_pairs * 16;
                for (nk_size_t pair_block = 0; pair_block < depth_pairs; pair_block += 8) {
                    __m128i rows_i16x8[8], columns_i16x8[8];
                    for (nk_size_t lane_idx = 0; lane_idx < 8; lane_idx++)
                        rows_i16x8[lane_idx] = _mm_loadu_si128(
                            (__m128i const *)(stage + lane_idx * depth_padded + pair_block * 2));
                    nk_attention_transpose_i16x8x8_haswell_(rows_i16x8, columns_i16x8);
                    for (nk_size_t pair_idx = 0; pair_idx < 8 && pair_block + pair_idx < depth_pairs; pair_idx++)
                        _mm_storeu_si128((__m128i *)(keys_tile + (pair_block + pair_idx) * 16),
                                         columns_i16x8[pair_idx]);
                }
            }

            // Pad the whole eight-position V tile so scalar broadcasts read initialized rows.
            for (nk_size_t position_idx = 0; position_idx < position_count_padded; position_idx++)
                nk_attention_copy_row_b8_haswell_(
                    (char const *)values + (position_first + position_idx) * value_stride + key_value_head_idx * depth,
                    values_plane + position_idx * depth_padded, position_idx < position_count ? depth : 0,
                    depth_padded);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_i8_haswell(                          //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,      //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_haswell_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_haswell_k_)
        nk_attention_packed_i8_serial_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                       depth, query_offsets, query_stride, output_stride, scale, band, tasks_begin,
                                       tasks_end);
    else
        nk_attention_packed_i8_haswell_(queries, key_value_packed, output, log_sum_exp, head_count,
                                        key_value_head_count, depth, query_offsets, query_stride, output_stride, scale,
                                        band, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_rope_f32_haswell(nk_f32_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
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
            for (; i + 8 <= half_depth; i += 8) {
                __m256 low_f32x8 = _mm256_loadu_ps(x_base + i);
                __m256 high_f32x8 = _mm256_loadu_ps(x_base + i + half_depth);
                __m256 cos_f32x8 = _mm256_loadu_ps(cos_row + i), sin_f32x8 = _mm256_loadu_ps(sin_row + i);
                _mm256_storeu_ps(y_base + i,
                                 _mm256_fmsub_ps(low_f32x8, cos_f32x8, _mm256_mul_ps(high_f32x8, sin_f32x8)));
                _mm256_storeu_ps(y_base + i + half_depth,
                                 _mm256_fmadd_ps(low_f32x8, sin_f32x8, _mm256_mul_ps(high_f32x8, cos_f32x8)));
            }
            for (; i != half_depth; ++i) {
                nk_f32_t low = x_base[i], high = x_base[i + half_depth];
                nk_f32_t cosine = cos_row[i], sine = sin_row[i];
                y_base[i] = low * cosine - high * sine;
                y_base[i + half_depth] = low * sine + high * cosine;
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_rope_bf16_haswell(nk_bf16_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
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
            for (; i + 8 <= half_depth; i += 8) {
                nk_b256_vec_t low_vec, high_vec;
                nk_load_bf16x8_to_f32x8_haswell_(x_base + i, &low_vec);
                nk_load_bf16x8_to_f32x8_haswell_(x_base + i + half_depth, &high_vec);
                __m256 low_f32x8 = low_vec.ymm_ps, high_f32x8 = high_vec.ymm_ps;
                __m256 cos_f32x8 = _mm256_loadu_ps(cos_row + i), sin_f32x8 = _mm256_loadu_ps(sin_row + i);
                _mm_storeu_si128((__m128i *)(y_base + i),
                                 nk_f32x8_to_bf16x8_haswell_(
                                     _mm256_fmsub_ps(low_f32x8, cos_f32x8, _mm256_mul_ps(high_f32x8, sin_f32x8))));
                _mm_storeu_si128((__m128i *)(y_base + i + half_depth),
                                 nk_f32x8_to_bf16x8_haswell_(
                                     _mm256_fmadd_ps(low_f32x8, sin_f32x8, _mm256_mul_ps(high_f32x8, cos_f32x8))));
            }
            for (; i != half_depth; ++i) {
                nk_f32_t low, high;
                nk_bf16_to_f32_(x_base + i, &low);
                nk_bf16_to_f32_(x_base + i + half_depth, &high);
                nk_f32_t cosine = cos_row[i], sine = sin_row[i], rotated_low = low * cosine - high * sine,
                         rotated_high = low * sine + high * cosine;
                nk_f32_to_bf16_(&rotated_low, y_base + i);
                nk_f32_to_bf16_(&rotated_high, y_base + i + half_depth);
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_rope_e4m3_haswell(nk_e4m3_t const *x, nk_f32_t const *cos, nk_f32_t const *sin,
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
            for (; i + 8 <= half_depth; i += 8) {
                nk_b256_vec_t low_vec, high_vec;
                nk_load_e4m3x8_to_f32x8_haswell_(x_base + i, &low_vec);
                nk_load_e4m3x8_to_f32x8_haswell_(x_base + i + half_depth, &high_vec);
                __m256 low_f32x8 = low_vec.ymm_ps, high_f32x8 = high_vec.ymm_ps;
                __m256 cos_f32x8 = _mm256_loadu_ps(cos_row + i), sin_f32x8 = _mm256_loadu_ps(sin_row + i);
                _mm_storel_epi64((__m128i *)(y_base + i),
                                 nk_f32x8_to_e4m3x8_haswell_(
                                     _mm256_fmsub_ps(low_f32x8, cos_f32x8, _mm256_mul_ps(high_f32x8, sin_f32x8))));
                _mm_storel_epi64((__m128i *)(y_base + i + half_depth),
                                 nk_f32x8_to_e4m3x8_haswell_(
                                     _mm256_fmadd_ps(low_f32x8, sin_f32x8, _mm256_mul_ps(high_f32x8, cos_f32x8))));
            }
            for (; i != half_depth; ++i) {
                nk_f32_t low, high;
                nk_e4m3_to_f32_(x_base + i, &low);
                nk_e4m3_to_f32_(x_base + i + half_depth, &high);
                nk_f32_t cosine = cos_row[i], sine = sin_row[i], rotated_low = low * cosine - high * sine,
                         rotated_high = low * sine + high * cosine;
                nk_f32_to_e4m3_(&rotated_low, y_base + i);
                nk_f32_to_e4m3_(&rotated_high, y_base + i + half_depth);
            }
        }
    }
    return nk_success_k;
}

#endif // NUMKONG_TARGET_HASWELL

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_HASWELL_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_ATTENTION_HASWELL_H
