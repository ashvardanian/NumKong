/**
 *  @brief Ragged attention for AVX-512 VNNI (Ice Lake) generation CPUs.
 *  @file include/numkong/attention/icelake.h
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  VNNI backend for the INT8 attention triple. Q, K and V arrive caller-quantized to I8,
 *  the Q·K descale already folded into `scale`. Scores are exact I32 integer dot products,
 *  the base-2 softmax reuses the Skylake helpers (Ice Lake caps imply Skylake), weights
 *  quantize to U8 as `round(255 · 2^(s₂ − m₂))`, and the P×V contraction runs natively on
 *  `_mm512_dpbusd_epi32` with the U8 weights as the unsigned operand.
 *
 *  @section attention_icelake_qk Q×Kᵀ correction placement
 *
 *  DPBUSD multiplies an unsigned byte by a signed byte, so Q is shifted into the unsigned
 *  domain once per query row (`q' = q ⊕ 0x80 = q + 128`). `dpbusd(q', k)` then yields
 *  `Σ(q+128)·k = Σq·k + 128·Σk`, so the exact score is `dpbusd(q', k) − 128·Σk`. The
 *  per-position `Σk = Σ_channel k[pos][channel]` depends only on K, so it is precomputed
 *  once at pack time and stored as one I32 per KV position in a table riding beside the K
 *  plane (see the payload layout below). The hot score loop then collapses to one load +
 *  one DPBUSD per K chunk, and the `128·Σk` correction folds into the finalize as a single
 *  `xmm ≪ 7` subtract across a position quad. Zero-padded channels are exact: a padded
 *  `k = 0` contributes `(q+128)·0 = 0` to the product and `0` to `Σk`.
 *
 *  @section attention_icelake_pv P×V layout
 *
 *  DPBUSD contracts four adjacent bytes per I32 lane, so the P×V contraction over KV
 *  positions needs four consecutive positions of one channel adjacent in memory. V is
 *  therefore packed position-quad-interleaved at pack time: byte `[group][channel][pos%4]`
 *  holds `v[4·group + pos%4][channel]`. A single 64-byte load then covers 16 channels × 4
 *  positions, the U8 weights of those four positions broadcast into every I32 lane, and one
 *  DPBUSD advances 16 channels by four positions with no shift and no correction. The I32
 *  accumulators drain to F32 once per panel and fold into the online `O = O·2^(m_old−m_new)
 *  + panel` correction. K stays token-major `[token][channel]`; both planes zero-pad
 *  channels to a multiple of 64 and positions to a multiple of 4. The row-max sweep covers
 *  live columns only: a zero-padded position's score of 0 could otherwise raise the max and
 *  zero out an all-negative row's weight sum. `depth > 256` routes to the width-agnostic
 *  serial tier from every entry point.
 *
 *  Per-segment payload is `[K planes][V planes][Σk tables]` across `num_kv_heads` heads: the
 *  K and V planes as above (`round_up(len, 4) · dim_padded` bytes each), then one I32 `Σk`
 *  per padded KV position per head. Two extra payload bytes per position per plane pair
 *  equal exactly one I32 per position, so the directory keeps its single closed form with
 *  `unit_bytes = dim_padded + 2` — `2 · num_kv_heads · round_up(len, 4) · (dim_padded + 2)`.
 */
#ifndef NK_ATTENTION_ICELAKE_H
#define NK_ATTENTION_ICELAKE_H

#if NK_TARGET_X8664_
#if NK_TARGET_ICELAKE

#include "numkong/attention/serial.h"  // shared packed-KV header/directory, width-agnostic fallback
#include "numkong/attention/skylake.h" // `nk_attention_exp2_f32x16_skylake_`, panel constants
#include "numkong/reduce/skylake.h"    // `nk_reduce_add_f32x16_skylake_`, `nk_reduce_max_f32x16_skylake_`
#include "numkong/dot/icelake.h"       // VNNI DPBUSD + SAD correction precedent

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                        \
    __attribute__((                                                                                                  \
        target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512vnni,avx512vbmi,avx512vpopcntdq,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vnni", "avx512vbmi", \
                   "avx512vpopcntdq", "f16c", "fma", "bmi", "bmi2")
#endif

enum {
    /** KV panel width in positions; the I32 score row (2 KB) stays L1-resident. */
    nk_attention_panel_icelake_k_ = 512,
    /** Widest head this backend handles in registers; larger heads route to the serial tier. */
    nk_attention_max_depth_icelake_k_ = 256,
};

NK_INTERNAL nk_size_t nk_attention_packed_size_icelake_(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_size_t payload_bytes = 0; // raw I8 K/V planes plus one I32 Σk per KV position, folded as `+ 2` per plane pair
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count *
                         (nk_size_t)nk_size_round_up_to_multiple_(segment_lengths[segment_idx], 4) * (depth_padded + 2);
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_PUBLIC nk_size_t nk_attention_packed_size_i8_icelake(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_icelake_k_)
        return nk_attention_packed_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_icelake_(key_value_head_count, depth, segment_lengths, segment_count);
}

NK_PUBLIC void nk_attention_pack_i8_icelake(                                                                   //
    nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count, nk_size_t depth,               //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                                          //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_icelake_k_) {
        nk_attention_pack_i8_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                    segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                    task_count);
        return;
    }

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, 4, depth_padded + 2);
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *payload_offsets_ro = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *payload_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment = task_idx / key_value_head_count, key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 4);
        nk_size_t const plane_bytes = position_count_padded * depth_padded;
        nk_i8_t *keys_plane = (nk_i8_t *)(payload_base + payload_offsets_ro[segment]) +
                              key_value_head_idx * plane_bytes;
        nk_i8_t *values_plane = keys_plane + key_value_head_count * plane_bytes;
        // `Σk` table rides after both plane blocks: one I32 per padded KV position per head.
        nk_i32_t *key_sums_plane = (nk_i32_t *)(payload_base + payload_offsets_ro[segment] +
                                                2 * key_value_head_count * plane_bytes) +
                                   key_value_head_idx * position_count_padded;
        __m512i const ones_u8x64 = _mm512_set1_epi8(1);

        // K stays token-major `[token][channel]`, channels zero-padded to `dim_padded`; `Σk` folds in per chunk.
        for (nk_size_t position_idx = 0; position_idx < position_count_padded; position_idx++) {
            nk_i8_t *keys_destination = keys_plane + position_idx * depth_padded;
            if (position_idx < position_count) {
                nk_i8_t const *keys_row = (nk_i8_t const *)((char const *)keys +
                                                            (position_first + position_idx) * key_stride_bytes) +
                                          key_value_head_idx * depth;
                __m512i ksum_acc_i32x16 = _mm512_setzero_si512();
                for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 64) {
                    __mmask64 const mask = (channel_idx + 64 <= depth) ? ~(__mmask64)0
                                           : (channel_idx < depth)
                                               ? (__mmask64)_bzhi_u64(~(nk_u64_t)0, depth - channel_idx)
                                               : (__mmask64)0;
                    __m512i const k_i8x64 = _mm512_maskz_loadu_epi8(mask, keys_row + channel_idx);
                    _mm512_storeu_si512(keys_destination + channel_idx, k_i8x64);
                    ksum_acc_i32x16 = _mm512_dpbusd_epi32(ksum_acc_i32x16, ones_u8x64, k_i8x64);
                }
                key_sums_plane[position_idx] = nk_reduce_add_i32x16_skylake_(ksum_acc_i32x16);
            }
            else {
                for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 64)
                    _mm512_storeu_si512(keys_destination + channel_idx, _mm512_setzero_si512());
                key_sums_plane[position_idx] = 0;
            }
        }

        // V is position-quad-interleaved: byte `[group][channel][pos%4]` = `v[4·group+pos%4][channel]`.
        for (nk_size_t quad_idx = 0; quad_idx < position_count_padded / 4; quad_idx++) {
            nk_i8_t *group_destination = values_plane + quad_idx * depth_padded * 4;
            nk_i8_t const *v_rows[4];
            for (nk_size_t lane_idx = 0; lane_idx < 4; lane_idx++) {
                nk_size_t const position_idx = quad_idx * 4 + lane_idx;
                v_rows[lane_idx] = (position_idx < position_count)
                                       ? (nk_i8_t const *)((char const *)values +
                                                           (position_first + position_idx) * value_stride_bytes) +
                                             key_value_head_idx * depth
                                       : (nk_i8_t const *)0;
            }
            for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx++)
                for (nk_size_t lane_idx = 0; lane_idx < 4; lane_idx++)
                    group_destination[channel_idx * 4 + lane_idx] = (v_rows[lane_idx] && channel_idx < depth)
                                                                        ? v_rows[lane_idx][channel_idx]
                                                                        : (nk_i8_t)0;
        }
    }
}

/**
 *  @brief Exact I32 score panel for one query row: `dpbusd(q', k) − 128·Σk` per position,
 *         four KV rows in flight. The hot loop is load + DPBUSD only; the quad drains through
 *         the 4-way unpack transpose (four scores at once) and folds the four pack-time `Σk`
 *         values as a single `xmm ≪ 7` subtract.
 */
NK_INTERNAL void nk_attention_score_panel_icelake_(nk_u8_t const *query_biased, nk_i8_t const *keys_plane,
                                                   nk_i32_t const *key_sums_plane, nk_size_t panel_start,
                                                   nk_size_t panel_len, nk_size_t depth_padded, nk_i32_t *scores) {
    nk_size_t position = 0;
    for (; position + 4 <= panel_len; position += 4) {
        nk_i8_t const *keys_row = keys_plane + (panel_start + position) * depth_padded;
        __m512i acc0_i32x16 = _mm512_setzero_si512(), acc1_i32x16 = _mm512_setzero_si512();
        __m512i acc2_i32x16 = _mm512_setzero_si512(), acc3_i32x16 = _mm512_setzero_si512();
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 64) {
            __m512i const qb_u8x64 = _mm512_load_si512(query_biased + channel_idx);
            acc0_i32x16 = _mm512_dpbusd_epi32(acc0_i32x16, qb_u8x64, _mm512_loadu_si512(keys_row + channel_idx));
            acc1_i32x16 = _mm512_dpbusd_epi32(acc1_i32x16, qb_u8x64,
                                              _mm512_loadu_si512(keys_row + depth_padded + channel_idx));
            acc2_i32x16 = _mm512_dpbusd_epi32(acc2_i32x16, qb_u8x64,
                                              _mm512_loadu_si512(keys_row + 2 * depth_padded + channel_idx));
            acc3_i32x16 = _mm512_dpbusd_epi32(acc3_i32x16, qb_u8x64,
                                              _mm512_loadu_si512(keys_row + 3 * depth_padded + channel_idx));
        }
        __m256i sum0_i32x8 = // zmm → ymm → xmm per accumulator, then 4-way transpose to one score per lane
            _mm256_add_epi32(_mm512_castsi512_si256(acc0_i32x16), _mm512_extracti32x8_epi32(acc0_i32x16, 1));
        __m256i sum1_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(acc1_i32x16),
                                              _mm512_extracti32x8_epi32(acc1_i32x16, 1));
        __m256i sum2_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(acc2_i32x16),
                                              _mm512_extracti32x8_epi32(acc2_i32x16, 1));
        __m256i sum3_i32x8 = _mm256_add_epi32(_mm512_castsi512_si256(acc3_i32x16),
                                              _mm512_extracti32x8_epi32(acc3_i32x16, 1));
        __m128i sum0_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum0_i32x8), _mm256_extracti128_si256(sum0_i32x8, 1));
        __m128i sum1_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum1_i32x8), _mm256_extracti128_si256(sum1_i32x8, 1));
        __m128i sum2_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum2_i32x8), _mm256_extracti128_si256(sum2_i32x8, 1));
        __m128i sum3_i32x4 = _mm_add_epi32(_mm256_castsi256_si128(sum3_i32x8), _mm256_extracti128_si256(sum3_i32x8, 1));
        __m128i transpose01_lo_i32x4 = _mm_unpacklo_epi32(sum0_i32x4, sum1_i32x4);
        __m128i transpose23_lo_i32x4 = _mm_unpacklo_epi32(sum2_i32x4, sum3_i32x4);
        __m128i transpose01_hi_i32x4 = _mm_unpackhi_epi32(sum0_i32x4, sum1_i32x4);
        __m128i transpose23_hi_i32x4 = _mm_unpackhi_epi32(sum2_i32x4, sum3_i32x4);
        __m128i biased_i32x4 = _mm_add_epi32(
            _mm_add_epi32(_mm_unpacklo_epi64(transpose01_lo_i32x4, transpose23_lo_i32x4),
                          _mm_unpackhi_epi64(transpose01_lo_i32x4, transpose23_lo_i32x4)),
            _mm_add_epi32(_mm_unpacklo_epi64(transpose01_hi_i32x4, transpose23_hi_i32x4),
                          _mm_unpackhi_epi64(transpose01_hi_i32x4, transpose23_hi_i32x4)));
        __m128i const correction_i32x4 = _mm_slli_epi32(
            _mm_loadu_si128((__m128i const *)(key_sums_plane + panel_start + position)), 7); // × 128
        _mm_storeu_si128((__m128i *)(scores + position), _mm_sub_epi32(biased_i32x4, correction_i32x4));
    }
    for (; position < panel_len; position++) {
        nk_i8_t const *keys_row = keys_plane + (panel_start + position) * depth_padded;
        __m512i acc_i32x16 = _mm512_setzero_si512();
        for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 64)
            acc_i32x16 = _mm512_dpbusd_epi32(acc_i32x16, _mm512_load_si512(query_biased + channel_idx),
                                             _mm512_loadu_si512(keys_row + channel_idx));
        scores[position] = nk_reduce_add_i32x16_skylake_(acc_i32x16) - 128 * key_sums_plane[panel_start + position];
    }
}

/**
 *  @brief Streaming base-2 softmax over one panel: row max over live columns only, U8 weight
 *         quantization `round(255 · 2^(s₂ − m₂))` — separate multiply and add to match the
 *         serial reference's rounding exactly — and the online-correction bookkeeping.
 *         Returns `2^(m_old − m_new)` for the caller to apply to its output accumulators.
 */
NK_INTERNAL nk_f32_t nk_attention_softmax_panel_icelake_(nk_i32_t const *scores, nk_u8_t *weights, nk_size_t panel_len,
                                                         nk_f32_t scale2, nk_f32_t *running_max2,
                                                         nk_f32_t *running_sum) {
    __m512 const scale2_f32x16 = _mm512_set1_ps(scale2);
    nk_size_t const full = panel_len & ~(nk_size_t)15;
    __mmask16 const tail_mask = (__mmask16)((1u << (panel_len - full)) - 1);

    __m512 max_f32x16 = _mm512_set1_ps(NK_F32_MIN);
    nk_size_t position_idx = 0;
    for (; position_idx < full; position_idx += 16)
        max_f32x16 = _mm512_max_ps(
            max_f32x16, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_load_si512(scores + position_idx)), scale2_f32x16));
    if (position_idx < panel_len)
        max_f32x16 = _mm512_max_ps(
            max_f32x16,
            _mm512_mask_mul_ps(_mm512_set1_ps(NK_F32_MIN), tail_mask,
                               _mm512_cvtepi32_ps(_mm512_load_si512(scores + position_idx)), scale2_f32x16));
    nk_f32_t const panel_max2 = nk_reduce_max_f32x16_skylake_(max_f32x16);
    nk_f32_t const new_max2 = *running_max2 > panel_max2 ? *running_max2 : panel_max2;
    nk_f32_t const correction = _mm512_cvtss_f32(
        nk_attention_exp2_f32x16_skylake_(_mm512_set1_ps(*running_max2 - new_max2)));
    *running_max2 = new_max2;

    __m512 const max2_f32x16 = _mm512_set1_ps(new_max2);
    __m512 const amplitude_f32x16 = _mm512_set1_ps(255.0f);
    __m512 const half_f32x16 = _mm512_set1_ps(0.5f);
    __m512 sum_f32x16 = _mm512_setzero_ps();
    for (position_idx = 0; position_idx < full; position_idx += 16) {
        __m512 const exp_f32x16 = nk_attention_exp2_f32x16_skylake_(
            _mm512_fmsub_ps(_mm512_cvtepi32_ps(_mm512_load_si512(scores + position_idx)), scale2_f32x16, max2_f32x16));
        __m512i const weight_u32x16 = _mm512_cvttps_epu32(
            _mm512_add_ps(_mm512_mul_ps(exp_f32x16, amplitude_f32x16), half_f32x16));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, _mm512_cvtepu32_ps(weight_u32x16));
        _mm_storeu_si128((__m128i *)(weights + position_idx), _mm512_cvtusepi32_epi8(weight_u32x16));
    }
    if (position_idx < panel_len) {
        __m512 const exp_f32x16 = _mm512_maskz_mov_ps(
            tail_mask, nk_attention_exp2_f32x16_skylake_(_mm512_fmsub_ps(
                           _mm512_cvtepi32_ps(_mm512_load_si512(scores + position_idx)), scale2_f32x16, max2_f32x16)));
        __m512i const weight_u32x16 = _mm512_maskz_cvttps_epu32(
            tail_mask, _mm512_add_ps(_mm512_mul_ps(exp_f32x16, amplitude_f32x16), half_f32x16));
        sum_f32x16 = _mm512_add_ps(sum_f32x16, _mm512_cvtepu32_ps(weight_u32x16));
        _mm_storeu_si128((__m128i *)(weights + position_idx), _mm512_cvtusepi32_epi8(weight_u32x16));
        position_idx += 16;
    }
    // Zero any leftover bytes up to the next quad boundary so padded V positions get zero weight.
    for (position_idx = panel_len; position_idx < nk_size_round_up_to_multiple_(panel_len, 4); position_idx++)
        weights[position_idx] = 0;
    *running_sum = *running_sum * correction + nk_reduce_add_f32x16_skylake_(sum_f32x16);
    return correction;
}

/**
 *  @brief P×V for one panel: `dpbusd(weight_quad, v_quad)` over the quad-interleaved V plane,
 *         I32 accumulators drained to F32 and folded into `O = O·correction + panel`.
 */
NK_INTERNAL void nk_attention_weighted_sum_panel_icelake_(nk_u8_t const *weights, nk_i8_t const *values_plane,
                                                          nk_size_t panel_start, nk_size_t panel_len,
                                                          nk_size_t depth_padded, nk_f32_t correction,
                                                          nk_f32_t *output_row) {
    nk_size_t const quad_count = (panel_len + 3) / 4;
    nk_size_t const quad_start = panel_start / 4;
    __m512 const correction_f32x16 = _mm512_set1_ps(correction);
    for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16) {
        __m512i acc_i32x16 = _mm512_setzero_si512();
        for (nk_size_t quad_idx = 0; quad_idx < quad_count; quad_idx++) {
            nk_u32_t weight_quad; // four consecutive positions' U8 weights, broadcast into every lane
            nk_copy_bytes_(&weight_quad, weights + quad_idx * 4, sizeof(weight_quad));
            __m512i const weights_u8x64 = _mm512_set1_epi32((int)weight_quad);
            __m512i const v_quad_i8x64 = _mm512_loadu_si512(values_plane + (quad_start + quad_idx) * depth_padded * 4 +
                                                            channel_idx * 4);
            acc_i32x16 = _mm512_dpbusd_epi32(acc_i32x16, weights_u8x64, v_quad_i8x64);
        }
        __m512 const scaled_f32x16 = _mm512_mul_ps(_mm512_load_ps(output_row + channel_idx), correction_f32x16);
        _mm512_store_ps(output_row + channel_idx, _mm512_add_ps(scaled_f32x16, _mm512_cvtepi32_ps(acc_i32x16)));
    }
}

NK_PUBLIC void nk_attention_packed_i8_icelake(                                                                  //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,                                     //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_icelake_k_) {
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
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // softmax(x) = softmax₂(x·log₂e)
    nk_size_t const panel_width = nk_attention_panel_icelake_k_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    NK_ALIGN64 nk_u8_t query_biased[nk_attention_max_depth_icelake_k_];
    NK_ALIGN64 nk_f32_t output_row[nk_attention_max_depth_icelake_k_];
    NK_ALIGN64 nk_i32_t scores[nk_attention_panel_icelake_k_];
    NK_ALIGN64 nk_u8_t weights[nk_attention_panel_icelake_k_];
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const dim_tail_mask = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment = task_idx / head_count, head = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment];
        nk_size_t const row_count = query_offsets[segment + 1] - query_offsets[segment];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 4);
        nk_size_t const plane_bytes = position_count_padded * depth_padded;
        nk_i8_t const *keys_plane = (nk_i8_t const *)(payload_base + payload_offsets[segment]) +
                                    (head / head_group_size) * plane_bytes;
        nk_i8_t const *values_plane = keys_plane + key_value_head_count * plane_bytes;
        nk_i32_t const *key_sums_plane = (nk_i32_t const *)(payload_base + payload_offsets[segment] +
                                                            2 * key_value_head_count * plane_bytes) +
                                         (head / head_group_size) * position_count_padded;

        for (nk_size_t row_idx = 0; row_idx < row_count; row_idx++) {
            nk_i8_t const *query_row = (nk_i8_t const *)((char const *)queries +
                                                         (query_offsets[segment] + row_idx) * query_stride_bytes) +
                                       head * depth;
            // Bias Q into the unsigned domain once per row_idx; padded channels become 0x80 = q of 0.
            for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 64) {
                __mmask64 const mask = (channel_idx + 64 <= depth) ? ~(__mmask64)0
                                       : (channel_idx < depth) ? (__mmask64)_bzhi_u64(~(nk_u64_t)0, depth - channel_idx)
                                                               : (__mmask64)0;
                _mm512_store_si512(
                    query_biased + channel_idx,
                    _mm512_xor_si512(_mm512_maskz_loadu_epi8(mask, query_row + channel_idx), xor_mask_u8x64));
            }
            for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16)
                _mm512_store_ps(output_row + channel_idx, _mm512_setzero_ps());
            nk_f32_t running_max2 = NK_F32_MIN, running_sum = 0;

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_len = (panel_start + panel_width <= position_count)
                                                ? panel_width
                                                : (position_count - panel_start);
                nk_attention_score_panel_icelake_(query_biased, keys_plane, key_sums_plane, panel_start, panel_len,
                                                  depth_padded, scores);
                nk_f32_t const correction = nk_attention_softmax_panel_icelake_(scores, weights, panel_len, scale2,
                                                                                &running_max2, &running_sum);
                nk_attention_weighted_sum_panel_icelake_(weights, values_plane, panel_start, panel_len, depth_padded,
                                                         correction, output_row);
            }

            __m512 const inverse_sum_f32x16 = _mm512_set1_ps(1.0f / running_sum);
            nk_f32_t *destination = output + (query_offsets[segment] + row_idx) * output_stride_floats + head * depth;
            nk_size_t channel_idx = 0;
            for (; channel_idx < depth_full; channel_idx += 16)
                _mm512_storeu_ps(destination + channel_idx,
                                 _mm512_mul_ps(_mm512_load_ps(output_row + channel_idx), inverse_sum_f32x16));
            if (channel_idx < depth)
                _mm512_mask_storeu_ps(destination + channel_idx, dim_tail_mask,
                                      _mm512_mul_ps(_mm512_load_ps(output_row + channel_idx), inverse_sum_f32x16));
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

#endif // NK_TARGET_ICELAKE
#endif // NK_TARGET_X8664_
#endif // NK_ATTENTION_ICELAKE_H
