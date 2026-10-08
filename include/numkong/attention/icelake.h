/**
 *  @file include/numkong/attention/icelake.h
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *  @brief Ragged attention for AVX-512 VNNI Ice Lake generation CPUs.
 *
 *  @sa include/numkong/attention.h
 *
 *  VNNI backend for the INT8 attention triple. Q, K and V arrive caller-quantized to I8, the Q · K
 *  descale already folded into @c scale. Scores are exact I32 integer dot products, the base-2
 *  softmax reuses the Skylake helpers — Ice Lake caps imply Skylake — weights quantize to U8 as
 *  round(255 · 2^(s₂ − m₂)), and the P × V contraction runs natively on @c _mm512_dpbusd_epi32 with
 *  the U8 weights as the unsigned operand.
 *
 *  @section attention_icelake_qk Q × Kᵀ correction placement
 *
 *  DPBUSD multiplies an unsigned byte by a signed byte, so Q is shifted into the unsigned domain
 *  once per query row, q' = q ⊕ 0x80 = q + 128. dpbusd(q', k) then yields Σ(q+128) · k = Σq · k +
 *  128 · Σk, so the exact score is dpbusd(q', k) − 128 · Σk. The per-position Σk = Σ_channel
 *  k[pos][channel] depends only on K, so it is precomputed once at pack time and stored as one I32
 *  per KV position in a table riding beside the K plane, see the payload layout below. To keep the
 *  score loop drain-free the score kernel never reduces across the 16 lanes: K is packed
 *  VNNI-interleaved so a 64-byte load holds four channels of sixteen consecutive KV positions, one
 *  score per lane. A query's four channels broadcast as one dword, @c _mm512_set1_epi32, and one
 *  DPBUSD advances sixteen KV positions by four channels; accumulating over the depth quads leaves
 *  sixteen exact biased scores with no transpose. Sixteen queries share each K load — hold sixteen
 *  accumulators and issue sixteen broadcast DPBUSDs per K vector — so the score cost scales flat
 *  with KV length. The 128 · Σk correction is one sixteen-wide `zmm ≪ 7` subtract per KV tile.
 *  Zero-padded channels stay exact: a padded k=0 adds (q+128) · 0=0 to the product and 0 to Σk.
 *
 *  @section attention_icelake_pv P × V layout
 *
 *  DPBUSD contracts four adjacent bytes per I32 lane, so the P × V contraction over KV positions
 *  needs four consecutive positions of one channel adjacent in memory. At pack time V is therefore
 *  position-quad-interleaved: byte [group][channel][pos%4] holds v[4 · group + pos%4][channel]. A
 *  single 64-byte load then covers 16 channels × 4 positions, the U8 weights of those four
 *  positions broadcast into every I32 lane, and one DPBUSD advances 16 channels by four positions
 *  with no shift and no correction. The I32 accumulators drain to F32 once per panel and fold into
 *  the online O = O · 2^(m_old−m_new) + panel correction. K is VNNI-interleaved as tiles of 16
 *  positions, each a depth quad of 16 lanes by 4 channels; both planes zero-pad channels to a
 *  multiple of 64, K pads positions to a multiple of 16, one 16-lane score tile, and V to a
 *  multiple of 4. The row-max sweep covers live columns only: a zero-padded position's score of 0
 *  could otherwise raise the max and zero out an all-negative row's weight sum. Heads deeper than
 *  256 channels accumulate scores over 256-channel chunks and the output in place.
 *
 *  Per-segment payload is [K planes][V planes][Σk tables] across @c key_value_head_count heads: the
 *  K and V planes as above, round_up(length, 16) · dim_padded bytes each, then one I32 Σk per
 *  padded KV position per head. Two extra payload bytes per position per plane pair equal exactly
 *  one I32 per position, so the directory keeps its single closed form with unit_bytes = dim_padded
 *  + 2 — 2 · key_value_head_count · round_up(length, 16) · (dim_padded + 2).
 */
#ifndef NUMKONG_ATTENTION_ICELAKE_H
#define NUMKONG_ATTENTION_ICELAKE_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_ICELAKE_

#include "numkong/attention/serial.h" // shared packed-KV header/directory, width-agnostic fallback
#include "numkong/each/skylake.h"     // `nk_exp2_f32x16_skylake_`, `nk_exp2_u8_i32x16_skylake_`
#include "numkong/reduce/skylake.h"   // `nk_reduce_add_f32x16_skylake_`, `nk_reduce_max_f32x16_skylake_`
#include "numkong/dot/icelake.h"      // VNNI DPBUSD + SAD correction precedent

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

    /** Channels of the biased queries held at once; deeper heads accumulate scores in chunks. */
    nk_attention_depth_chunk_icelake_k_ = 256,
};

/** Register 16×16 transpose of 32-bit elements (hierarchical unpack + lane shuffle). Given 16
 *  rows (each 16 dwords), returns the 16 columns. Used at pack time to turn 16 position rows into
 *  the VNNI depth-quad tiles. The @c shuffle_i32x4 stages emit rows in the group order
 *  `groups[i]` holds group `{0,1,2,3,8,9,10,11,4,5,6,7,12,13,14,15}[i]`; the caller restores
 *  natural order at store. */
NUMKONG_INLINE void nk_attention_transpose_i32x16x16_icelake_(__m512i const rows_i32x16[16],
                                                              __m512i groups_i32x16[16]) {
    __m512i t01_low_i32x16 = _mm512_unpacklo_epi32(rows_i32x16[0], rows_i32x16[1]),
            t01_high_i32x16 = _mm512_unpackhi_epi32(rows_i32x16[0], rows_i32x16[1]);
    __m512i t23_low_i32x16 = _mm512_unpacklo_epi32(rows_i32x16[2], rows_i32x16[3]),
            t23_high_i32x16 = _mm512_unpackhi_epi32(rows_i32x16[2], rows_i32x16[3]);
    __m512i t45_low_i32x16 = _mm512_unpacklo_epi32(rows_i32x16[4], rows_i32x16[5]),
            t45_high_i32x16 = _mm512_unpackhi_epi32(rows_i32x16[4], rows_i32x16[5]);
    __m512i t67_low_i32x16 = _mm512_unpacklo_epi32(rows_i32x16[6], rows_i32x16[7]),
            t67_high_i32x16 = _mm512_unpackhi_epi32(rows_i32x16[6], rows_i32x16[7]);
    __m512i t89_low_i32x16 = _mm512_unpacklo_epi32(rows_i32x16[8], rows_i32x16[9]),
            t89_high_i32x16 = _mm512_unpackhi_epi32(rows_i32x16[8], rows_i32x16[9]);
    __m512i tab_low_i32x16 = _mm512_unpacklo_epi32(rows_i32x16[10], rows_i32x16[11]),
            tab_high_i32x16 = _mm512_unpackhi_epi32(rows_i32x16[10], rows_i32x16[11]);
    __m512i tcd_low_i32x16 = _mm512_unpacklo_epi32(rows_i32x16[12], rows_i32x16[13]),
            tcd_high_i32x16 = _mm512_unpackhi_epi32(rows_i32x16[12], rows_i32x16[13]);
    __m512i tef_low_i32x16 = _mm512_unpacklo_epi32(rows_i32x16[14], rows_i32x16[15]),
            tef_high_i32x16 = _mm512_unpackhi_epi32(rows_i32x16[14], rows_i32x16[15]);
    __m512i u0123_ll_i32x16 = _mm512_unpacklo_epi64(t01_low_i32x16, t23_low_i32x16),
            u0123_lh_i32x16 = _mm512_unpackhi_epi64(t01_low_i32x16, t23_low_i32x16);
    __m512i u0123_hl_i32x16 = _mm512_unpacklo_epi64(t01_high_i32x16, t23_high_i32x16),
            u0123_hh_i32x16 = _mm512_unpackhi_epi64(t01_high_i32x16, t23_high_i32x16);
    __m512i u4567_ll_i32x16 = _mm512_unpacklo_epi64(t45_low_i32x16, t67_low_i32x16),
            u4567_lh_i32x16 = _mm512_unpackhi_epi64(t45_low_i32x16, t67_low_i32x16);
    __m512i u4567_hl_i32x16 = _mm512_unpacklo_epi64(t45_high_i32x16, t67_high_i32x16),
            u4567_hh_i32x16 = _mm512_unpackhi_epi64(t45_high_i32x16, t67_high_i32x16);
    __m512i u89ab_ll_i32x16 = _mm512_unpacklo_epi64(t89_low_i32x16, tab_low_i32x16),
            u89ab_lh_i32x16 = _mm512_unpackhi_epi64(t89_low_i32x16, tab_low_i32x16);
    __m512i u89ab_hl_i32x16 = _mm512_unpacklo_epi64(t89_high_i32x16, tab_high_i32x16),
            u89ab_hh_i32x16 = _mm512_unpackhi_epi64(t89_high_i32x16, tab_high_i32x16);
    __m512i ucdef_ll_i32x16 = _mm512_unpacklo_epi64(tcd_low_i32x16, tef_low_i32x16),
            ucdef_lh_i32x16 = _mm512_unpackhi_epi64(tcd_low_i32x16, tef_low_i32x16);
    __m512i ucdef_hl_i32x16 = _mm512_unpacklo_epi64(tcd_high_i32x16, tef_high_i32x16),
            ucdef_hh_i32x16 = _mm512_unpackhi_epi64(tcd_high_i32x16, tef_high_i32x16);
    __m512i v0_a_i32x16 = _mm512_shuffle_i32x4(u0123_ll_i32x16, u4567_ll_i32x16, 0x88),
            v0_b_i32x16 = _mm512_shuffle_i32x4(u0123_ll_i32x16, u4567_ll_i32x16, 0xDD);
    __m512i v1_a_i32x16 = _mm512_shuffle_i32x4(u0123_lh_i32x16, u4567_lh_i32x16, 0x88),
            v1_b_i32x16 = _mm512_shuffle_i32x4(u0123_lh_i32x16, u4567_lh_i32x16, 0xDD);
    __m512i v2_a_i32x16 = _mm512_shuffle_i32x4(u0123_hl_i32x16, u4567_hl_i32x16, 0x88),
            v2_b_i32x16 = _mm512_shuffle_i32x4(u0123_hl_i32x16, u4567_hl_i32x16, 0xDD);
    __m512i v3_a_i32x16 = _mm512_shuffle_i32x4(u0123_hh_i32x16, u4567_hh_i32x16, 0x88),
            v3_b_i32x16 = _mm512_shuffle_i32x4(u0123_hh_i32x16, u4567_hh_i32x16, 0xDD);
    __m512i v4_a_i32x16 = _mm512_shuffle_i32x4(u89ab_ll_i32x16, ucdef_ll_i32x16, 0x88),
            v4_b_i32x16 = _mm512_shuffle_i32x4(u89ab_ll_i32x16, ucdef_ll_i32x16, 0xDD);
    __m512i v5_a_i32x16 = _mm512_shuffle_i32x4(u89ab_lh_i32x16, ucdef_lh_i32x16, 0x88),
            v5_b_i32x16 = _mm512_shuffle_i32x4(u89ab_lh_i32x16, ucdef_lh_i32x16, 0xDD);
    __m512i v6_a_i32x16 = _mm512_shuffle_i32x4(u89ab_hl_i32x16, ucdef_hl_i32x16, 0x88),
            v6_b_i32x16 = _mm512_shuffle_i32x4(u89ab_hl_i32x16, ucdef_hl_i32x16, 0xDD);
    __m512i v7_a_i32x16 = _mm512_shuffle_i32x4(u89ab_hh_i32x16, ucdef_hh_i32x16, 0x88),
            v7_b_i32x16 = _mm512_shuffle_i32x4(u89ab_hh_i32x16, ucdef_hh_i32x16, 0xDD);
    groups_i32x16[0] = _mm512_shuffle_i32x4(v0_a_i32x16, v4_a_i32x16, 0x88),
    groups_i32x16[1] = _mm512_shuffle_i32x4(v1_a_i32x16, v5_a_i32x16, 0x88);
    groups_i32x16[2] = _mm512_shuffle_i32x4(v2_a_i32x16, v6_a_i32x16, 0x88),
    groups_i32x16[3] = _mm512_shuffle_i32x4(v3_a_i32x16, v7_a_i32x16, 0x88);
    groups_i32x16[4] = _mm512_shuffle_i32x4(v0_a_i32x16, v4_a_i32x16, 0xDD),
    groups_i32x16[5] = _mm512_shuffle_i32x4(v1_a_i32x16, v5_a_i32x16, 0xDD);
    groups_i32x16[6] = _mm512_shuffle_i32x4(v2_a_i32x16, v6_a_i32x16, 0xDD),
    groups_i32x16[7] = _mm512_shuffle_i32x4(v3_a_i32x16, v7_a_i32x16, 0xDD);
    groups_i32x16[8] = _mm512_shuffle_i32x4(v0_b_i32x16, v4_b_i32x16, 0x88),
    groups_i32x16[9] = _mm512_shuffle_i32x4(v1_b_i32x16, v5_b_i32x16, 0x88);
    groups_i32x16[10] = _mm512_shuffle_i32x4(v2_b_i32x16, v6_b_i32x16, 0x88),
    groups_i32x16[11] = _mm512_shuffle_i32x4(v3_b_i32x16, v7_b_i32x16, 0x88);
    groups_i32x16[12] = _mm512_shuffle_i32x4(v0_b_i32x16, v4_b_i32x16, 0xDD),
    groups_i32x16[13] = _mm512_shuffle_i32x4(v1_b_i32x16, v5_b_i32x16, 0xDD);
    groups_i32x16[14] = _mm512_shuffle_i32x4(v2_b_i32x16, v6_b_i32x16, 0xDD),
    groups_i32x16[15] = _mm512_shuffle_i32x4(v3_b_i32x16, v7_b_i32x16, 0xDD);
}

NUMKONG_INLINE nk_size_t nk_attention_pack_size_icelake_(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count) {
    // Raw I8 K/V planes plus one I32 Σk per KV position, folded as `+ 2` per plane pair
    nk_size_t const unit_bytes = nk_size_round_up_to_multiple_(depth, 64) + 2;
    return nk_attention_pack_bound_(key_value_head_count, token_count, segment_count, 16, unit_bytes);
}

/** Drain-free exact I32 scores for a 16-query block over one panel: for each 16-KV tile it holds 16
 *  lane-parallel accumulators (one score per KV position), broadcasts each query's four-channel
 *  dword and issues 16 DPBUSDs reusing one K load, then accumulates over the depth quads. No lane
 *  reduction and no transpose: Σq · k lands directly per lane. The 128 · Σk correction is one
 *  16-wide `zmm ≪ 7` subtract per tile. Writes a @b [16,panel_width] score block whose query rows
 *  sit @c panel_width apart. */
NUMKONG_INLINE void nk_attention_score_block_icelake_(nk_u8_t const *queries_biased, nk_i8_t const *keys_plane,
                                                      nk_i32_t const *key_sums_plane, nk_size_t panel_start,
                                                      nk_size_t panel_len, nk_size_t depth_padded,
                                                      nk_size_t group_begin, nk_size_t group_count, nk_i32_t *scores) {
    nk_size_t const depth_groups = depth_padded / 4;
    nk_size_t const tile_first = panel_start / 16;
    nk_size_t const tile_count = (panel_len + 15) / 16;
    for (nk_size_t tile_idx = 0; tile_idx < tile_count; tile_idx++) {
        __m512i accumulator_i32x16[16];
        __m512i const correction_i32x16 = _mm512_slli_epi32(
            _mm512_loadu_si512(key_sums_plane + panel_start + tile_idx * 16), 7); // × 128
        for (nk_size_t query_idx = 0; query_idx < 16; query_idx++)
            accumulator_i32x16[query_idx] =
                group_begin == 0
                    ? _mm512_sub_epi32(_mm512_setzero_si512(), correction_i32x16)
                    : _mm512_loadu_si512(scores + query_idx * nk_attention_panel_icelake_k_ + tile_idx * 16);
        nk_i8_t const *keys_tile = keys_plane + (tile_first + tile_idx) * depth_groups * 64;
        for (nk_size_t group_idx = 0; group_idx < group_count; group_idx++) {
            // 16 positions × 4 channels
            __m512i const k_i8x64 = _mm512_loadu_si512(keys_tile + (group_begin + group_idx) * 64);
            for (nk_size_t query_idx = 0; query_idx < 16; query_idx++) {
                __m512i const query_quad_u8x64 = _mm512_broadcastd_epi32(
                    _mm_loadu_si32(queries_biased + query_idx * nk_attention_depth_chunk_icelake_k_ + group_idx * 4));
                accumulator_i32x16[query_idx] = _mm512_dpbusd_epi32(accumulator_i32x16[query_idx], query_quad_u8x64,
                                                                    k_i8x64);
            }
        }
        for (nk_size_t query_idx = 0; query_idx < 16; query_idx++)
            _mm512_storeu_si512(scores + query_idx * nk_attention_panel_icelake_k_ + tile_idx * 16,
                                accumulator_i32x16[query_idx]);
    }
}

/** Streaming base-2 softmax over the panel-relative live range from @p range_begin inclusive to
 *  @p range_end exclusive, all in integer arithmetic: the row max is an exact @c _mm512_max_epi32
 *  over that range only, weights come from the integer i-exp over (score − max) · scale₂ in Q15,
 *  and the weight sum accumulates in I32. Weights outside the range, from its 16-aligned start to
 *  its quad-rounded end, are zero. Only the online correction, 2^((m_old − m_new) · scale₂) per
 *  panel, stays in F32, where it scales the F32 output accumulators anyway, and is returned. */
NUMKONG_INLINE nk_f32_t nk_attention_softmax_panel_icelake_(nk_i32_t const *scores, nk_u8_t *weights,
                                                            nk_size_t range_begin, nk_size_t range_end, nk_f32_t scale2,
                                                            nk_i32_t scale_fixed, nk_i32_t delta_floor,
                                                            nk_i32_t *running_max, nk_f32_t *running_sum) {
    nk_size_t const panel_len = range_end;
    nk_size_t const full = panel_len & ~(nk_size_t)15;
    __mmask16 const tail_m16 = (__mmask16)((1u << (panel_len - full)) - 1);
    nk_size_t const head_start = range_begin & ~(nk_size_t)15; // masks lanes outside a mid-group range
    __mmask16 const head_m16 = (__mmask16)((0xFFFFu << (range_begin - head_start)) &
                                           (panel_len - head_start < 16 ? tail_m16 : 0xFFFFu));

    __m512i max_i32x16 = _mm512_set1_epi32(NUMKONG_I32_MIN);
    nk_size_t position_idx = head_start;
    if (head_start != range_begin) {
        max_i32x16 = _mm512_mask_max_epi32(max_i32x16, head_m16, max_i32x16, _mm512_load_si512(scores + head_start));
        position_idx += 16;
    }
    for (; position_idx < full; position_idx += 16)
        max_i32x16 = _mm512_max_epi32(max_i32x16, _mm512_load_si512(scores + position_idx));
    if (position_idx < panel_len)
        max_i32x16 = _mm512_mask_max_epi32(max_i32x16, tail_m16, max_i32x16, _mm512_load_si512(scores + position_idx));
    nk_i32_t const panel_max = _mm512_reduce_max_epi32(max_i32x16);
    nk_i32_t const new_max = *running_max > panel_max ? *running_max : panel_max;
    nk_f32_t const correction = _mm512_cvtss_f32(
        nk_exp2_f32x16_skylake_(_mm512_set1_ps(((nk_f32_t)*running_max - (nk_f32_t)new_max) * scale2)));
    *running_max = new_max;

    __m512i const new_max_i32x16 = _mm512_set1_epi32(new_max);
    __m512i const scale_fixed_i32x16 = _mm512_set1_epi32(scale_fixed);
    __m512i const delta_floor_i32x16 = _mm512_set1_epi32(delta_floor);
    __m512i sum_a_i32x16 = _mm512_setzero_si512();
    __m512i sum_b_i32x16 = _mm512_setzero_si512();
    position_idx = head_start;
    if (head_start != range_begin) {
        __m512i const delta_i32x16 = _mm512_max_epi32(
            _mm512_sub_epi32(_mm512_load_si512(scores + head_start), new_max_i32x16), delta_floor_i32x16);
        __m512i const weight_i32x16 = _mm512_maskz_mov_epi32(
            head_m16, nk_exp2_u8_i32x16_skylake_(_mm512_mullo_epi32(delta_i32x16, scale_fixed_i32x16)));
        sum_a_i32x16 = weight_i32x16;
        _mm_storeu_si128((__m128i *)(weights + head_start), _mm512_cvtusepi32_epi8(weight_i32x16));
        position_idx += 16;
    }
    // Two independent 16-lane groups per iteration: group B's i-exp fills the port-0 vpmulld latency left
    // by group A's dependency chain; the separate sum accumulators recombine after the loop, and each
    // group's delta/iexp2/store is byte-identical to the scalar path.
    for (; position_idx + 32 <= full; position_idx += 32) {
        __m512i const delta_a_i32x16 = _mm512_max_epi32(
            _mm512_sub_epi32(_mm512_load_si512(scores + position_idx), new_max_i32x16), delta_floor_i32x16);
        __m512i const weight_a_i32x16 = nk_exp2_u8_i32x16_skylake_(
            _mm512_mullo_epi32(delta_a_i32x16, scale_fixed_i32x16));
        __m512i const delta_b_i32x16 = _mm512_max_epi32(
            _mm512_sub_epi32(_mm512_load_si512(scores + position_idx + 16), new_max_i32x16), delta_floor_i32x16);
        __m512i const weight_b_i32x16 = nk_exp2_u8_i32x16_skylake_(
            _mm512_mullo_epi32(delta_b_i32x16, scale_fixed_i32x16));
        sum_a_i32x16 = _mm512_add_epi32(sum_a_i32x16, weight_a_i32x16);
        sum_b_i32x16 = _mm512_add_epi32(sum_b_i32x16, weight_b_i32x16);
        _mm_storeu_si128((__m128i *)(weights + position_idx), _mm512_cvtusepi32_epi8(weight_a_i32x16));
        _mm_storeu_si128((__m128i *)(weights + position_idx + 16), _mm512_cvtusepi32_epi8(weight_b_i32x16));
    }
    __m512i sum_i32x16 = _mm512_add_epi32(sum_a_i32x16, sum_b_i32x16);
    for (; position_idx < full; position_idx += 16) { // trailing odd 16-group, if any
        __m512i const delta_i32x16 = _mm512_max_epi32(
            _mm512_sub_epi32(_mm512_load_si512(scores + position_idx), new_max_i32x16), delta_floor_i32x16);
        __m512i const weight_i32x16 = nk_exp2_u8_i32x16_skylake_(_mm512_mullo_epi32(delta_i32x16, scale_fixed_i32x16));
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, weight_i32x16);
        _mm_storeu_si128((__m128i *)(weights + position_idx), _mm512_cvtusepi32_epi8(weight_i32x16));
    }
    if (position_idx < panel_len) {
        __m512i const delta_i32x16 = _mm512_max_epi32(
            _mm512_sub_epi32(_mm512_load_si512(scores + position_idx), new_max_i32x16), delta_floor_i32x16);
        __m512i const weight_i32x16 = _mm512_maskz_mov_epi32(
            tail_m16, nk_exp2_u8_i32x16_skylake_(_mm512_mullo_epi32(delta_i32x16, scale_fixed_i32x16)));
        sum_i32x16 = _mm512_add_epi32(sum_i32x16, weight_i32x16);
        _mm_storeu_si128((__m128i *)(weights + position_idx), _mm512_cvtusepi32_epi8(weight_i32x16));
        position_idx += 16;
    }
    *running_sum = *running_sum * correction + (nk_f32_t)_mm512_reduce_add_epi32(sum_i32x16);
    return correction;
}

/** P × V over the quads covering the panel-relative range from @p range_begin inclusive to
 *  @p range_end exclusive: `dpbusd(weight_quad, v_quad)` over the quad-interleaved V plane, I32
 *  accumulators drained to F32 and folded into O = O · correction + panel. */
NUMKONG_INLINE void nk_attention_weighted_sum_panel_icelake_(nk_u8_t const *weights, nk_i8_t const *values_plane,
                                                             nk_size_t panel_start, nk_size_t range_begin,
                                                             nk_size_t range_end, nk_size_t depth_padded,
                                                             nk_size_t depth, nk_f32_t correction,
                                                             nk_f32_t *output_row) {
    nk_size_t const quad_first = range_begin / 4;
    nk_size_t const quad_count = (range_end + 3) / 4;
    nk_size_t const quad_start = panel_start / 4;
    __m512 const correction_f32x16 = _mm512_set1_ps(correction);
    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx += 16) {
        __mmask16 const channel_m16 = channel_idx + 16 <= depth ? (__mmask16)0xFFFF
                                                                : (__mmask16)((1u << (depth - channel_idx)) - 1);
        __m512i accumulator_i32x16 = _mm512_setzero_si512();
        for (nk_size_t quad_idx = quad_first; quad_idx < quad_count; quad_idx++) {
            __m512i const weights_u8x64 = _mm512_broadcastd_epi32(_mm_loadu_si32(weights + quad_idx * 4));
            __m512i const v_quad_i8x64 = _mm512_loadu_si512(values_plane + (quad_start + quad_idx) * depth_padded * 4 +
                                                            channel_idx * 4);
            accumulator_i32x16 = _mm512_dpbusd_epi32(accumulator_i32x16, weights_u8x64, v_quad_i8x64);
        }
        __m512 const scaled_f32x16 = _mm512_mul_ps(_mm512_maskz_loadu_ps(channel_m16, output_row + channel_idx),
                                                   correction_f32x16);
        _mm512_mask_storeu_ps(output_row + channel_idx, channel_m16,
                              _mm512_add_ps(scaled_f32x16, _mm512_cvtepi32_ps(accumulator_i32x16)));
    }
}

/** Shared I8 body: 16-row query blocks skip the panels outside @p band, and rows of a panel the
 *  band's edge crosses read only the keys @c nk_diagonal_band_row_range_ shows them. */
NUMKONG_INLINE void nk_attention_packed_i8_icelake_(                                                //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp,  //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                          //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, //
    nk_diagonal_band_t band, nk_size_t task_begin, nk_size_t task_end) {
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
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;                // softmax(x) = softmax₂(x · log₂e)
    nk_i32_t const scale_fixed = (nk_i32_t)(scale2 * 32768.0f + 0.5f); // Q15 scale for the integer exponential
    nk_i32_t const delta_floor = // the score delta below which every weight quantizes to zero (2^t · 255 + 0.5 < 1)
        scale_fixed > 0 ? -(nk_i32_t)((10u << 15) / (nk_u32_t)scale_fixed) - 1 : 0;
    nk_size_t const panel_width = nk_attention_panel_icelake_k_;

    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (task_begin < grid_begin) task_begin = grid_begin;
    if (task_end > grid_end) task_end = grid_end;

    // 16-query blocks share each K load; each query keeps its own running state and output.
    nk_align_(64) nk_u8_t queries_biased[16 * nk_attention_depth_chunk_icelake_k_];
    nk_align_(64) nk_i32_t scores[16 * nk_attention_panel_icelake_k_];
    nk_align_(64) nk_u8_t weights[nk_attention_panel_icelake_k_];
    nk_i32_t running_max[16];
    nk_f32_t running_sum[16];
    nk_size_t key_begins[16], key_ends[16];
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const dim_tail_m16 = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t head = 0; head < head_count && task_begin < task_end; head++) {
        nk_size_t const token_first = (task_begin + head_count - 1 - head) / head_count;
        nk_size_t const token_end = (task_end + head_count - 1 - head) / head_count;
        for (nk_size_t segment = nk_attention_segment_of_(query_offsets, segment_count, token_first);
             segment < segment_count && query_offsets[segment] < token_end; segment++) {
            nk_size_t const query_first = query_offsets[segment], query_end = query_offsets[segment + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = segment_lengths[segment];
            nk_i64_t const first_position = nk_attention_first_position_(query_end - query_first, position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 16);
            nk_size_t const plane_bytes = position_count_padded * depth_padded;
            nk_i8_t const *keys_plane = (nk_i8_t const *)(payload_base + payload_offsets[segment]) +
                                        (head / head_group_size) * plane_bytes;
            nk_i8_t const *values_plane = keys_plane + key_value_head_count * plane_bytes;
            nk_i32_t const *key_sums_plane = (nk_i32_t const *)(payload_base + payload_offsets[segment] +
                                                                2 * key_value_head_count * plane_bytes) +
                                             (head / head_group_size) * position_count_padded;

            // Blocks keep the segment's 16-row grid, as panels start at each block's first keys.
            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block = row_begin / 16 * 16; row_block < row_end; row_block += 16) {
                nk_size_t const block_rows = (row_count - row_block < 16) ? (row_count - row_block) : 16;

                for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                    running_max[block_row] = NUMKONG_I32_MIN;
                    running_sum[block_row] = 0;
                    if (row_block + block_row < row_begin || row_block + block_row >= row_end) continue;
                    nk_f32_t *output_row = output + (query_first + row_block + block_row) * output_stride_floats +
                                           head * depth;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx += 16)
                        _mm512_mask_storeu_ps(output_row + channel_idx,
                                              channel_idx + 16 <= depth ? (__mmask16)0xFFFF : dim_tail_m16,
                                              _mm512_setzero_ps());
                }
                nk_size_t block_begin = position_count, block_end = 0;
                nk_i64_t const block_position = first_position + (nk_i64_t)row_block;
                // The block's key union bounds the panel sweep; empty rows add nothing to it.
                for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                    nk_diagonal_band_row_range_(band, block_position + (nk_i64_t)block_row, position_count,
                                                &key_begins[block_row], &key_ends[block_row]);
                    if (key_begins[block_row] == key_ends[block_row]) continue;
                    if (key_begins[block_row] < block_begin) block_begin = key_begins[block_row];
                    if (key_ends[block_row] > block_end) block_end = key_ends[block_row];
                }
                // Panels start on a 16-position K tile; rows clip to their own range per panel.
                for (nk_size_t panel_start = block_begin & ~(nk_size_t)15; panel_start < block_end;
                     panel_start += panel_width) {
                    nk_size_t const panel_len = (panel_start + panel_width <= block_end) ? panel_width
                                                                                         : (block_end - panel_start);
                    nk_diagonal_band_coverage_t const coverage = nk_diagonal_band_tile_coverage_(
                        band, block_position, block_rows, panel_start, panel_len);
                    if (coverage == nk_diagonal_band_outside_k) continue;
                    for (nk_size_t chunk_start = 0; chunk_start < depth_padded;
                         chunk_start += nk_attention_depth_chunk_icelake_k_) {
                        nk_size_t const chunk_padded = depth_padded - chunk_start < nk_attention_depth_chunk_icelake_k_
                                                           ? depth_padded - chunk_start
                                                           : nk_attention_depth_chunk_icelake_k_;
                        // Bias each block query into the unsigned domain; padded channels become
                        // 0x80 and meet zero K bytes. Slots past the block load nothing, unread.
                        for (nk_size_t block_row = 0; block_row < 16; block_row++) {
                            nk_i8_t const *query_row = block_row < block_rows
                                                           ? (nk_i8_t const *)((char const *)queries +
                                                                               (query_first + row_block + block_row) *
                                                                                   query_stride) +
                                                                 head * depth + chunk_start
                                                           : queries;
                            for (nk_size_t channel_idx = 0; channel_idx < chunk_padded; channel_idx += 64) {
                                nk_size_t const live = chunk_start + channel_idx < depth
                                                           ? depth - chunk_start - channel_idx
                                                           : 0;
                                __mmask64 const load_m64 = block_row >= block_rows ? (__mmask64)0
                                                           : live >= 64 ? ~(__mmask64)0
                                                                        : (__mmask64)_bzhi_u64(~(nk_u64_t)0, live);
                                _mm512_store_si512(
                                    queries_biased + block_row * nk_attention_depth_chunk_icelake_k_ + channel_idx,
                                    _mm512_xor_si512(_mm512_maskz_loadu_epi8(load_m64, query_row + channel_idx),
                                                     xor_mask_u8x64));
                            }
                        }
                        nk_attention_score_block_icelake_(queries_biased, keys_plane, key_sums_plane, panel_start,
                                                          panel_len, depth_padded, chunk_start / 4, chunk_padded / 4,
                                                          scores);
                    }
                    for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                        if (row_block + block_row < row_begin || row_block + block_row >= row_end) continue;
                        nk_size_t range_begin = 0, range_end = panel_len;
                        if (coverage == nk_diagonal_band_crossing_k) {
                            if (key_begins[block_row] > panel_start) range_begin = key_begins[block_row] - panel_start;
                            if (key_ends[block_row] < panel_start + panel_len)
                                range_end = key_ends[block_row] > panel_start ? key_ends[block_row] - panel_start : 0;
                            if (range_begin >= range_end) continue; // no key of this panel: running state unchanged
                        }
                        nk_f32_t const correction = nk_attention_softmax_panel_icelake_(
                            scores + block_row * nk_attention_panel_icelake_k_, weights, range_begin, range_end, scale2,
                            scale_fixed, delta_floor, &running_max[block_row], &running_sum[block_row]);
                        nk_attention_weighted_sum_panel_icelake_(
                            weights, values_plane, panel_start, range_begin, range_end, depth_padded, depth, correction,
                            output + (query_first + row_block + block_row) * output_stride_floats + head * depth);
                    }
                }

                for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                    if (row_block + block_row < row_begin || row_block + block_row >= row_end) continue;
                    __m512 const inverse_sum_f32x16 = _mm512_set1_ps(
                        running_sum[block_row] > 0 ? 1 / running_sum[block_row] : 0);
                    nk_size_t const token = query_first + row_block + block_row;
                    nk_f32_t *destination = output + token * output_stride_floats + head * depth;
                    for (nk_size_t channel_idx = 0; channel_idx < depth; channel_idx += 16) {
                        __mmask16 const channel_m16 = channel_idx + 16 <= depth ? (__mmask16)0xFFFF : dim_tail_m16;
                        _mm512_mask_storeu_ps(
                            destination + channel_idx, channel_m16,
                            _mm512_mul_ps(_mm512_maskz_loadu_ps(channel_m16, destination + channel_idx),
                                          inverse_sum_f32x16));
                    }
                    if (log_sum_exp)
                        log_sum_exp[token * head_count + head] = nk_attention_log_sum_exp_(
                            (nk_f32_t)running_max[block_row] * scale2, running_sum[block_row] / 255.0f);
                }
            }
        }
    }
}

#if NUMKONG_TARGET_ICELAKE

NUMKONG_API nk_status_t nk_attention_pack_size_i8_icelake(nk_size_t key_value_head_count, nk_size_t depth,
                                                          nk_size_t token_count, nk_size_t segment_count,
                                                          nk_size_t *bytes) {
    *bytes = nk_attention_pack_size_icelake_(key_value_head_count, depth, token_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_icelake(void const *key_value_packed, nk_size_t *heads,
                                                             nk_size_t *depth, nk_size_t *segments, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_icelake_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_i8_icelake(                                            //
    nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count, nk_size_t key_stride,
    nk_size_t value_stride, void *key_value_packed, nk_size_t task_begin, nk_size_t task_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 task_begin, 16, depth_padded + 2, nk_cap_icelake_k);
    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (task_begin >= total_tasks) return nk_success_k;
    if (task_end > total_tasks) task_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = task_begin; task_idx < task_end; task_idx++) {
        nk_size_t const segment = task_idx / key_value_head_count, key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_(segment_lengths[payload_segment], key_value_head_count,
                                                               16, depth_padded + 2);
        nk_size_t const position_count = segment_lengths[segment];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 16);
        nk_size_t const plane_bytes = position_count_padded * depth_padded;
        nk_i8_t *keys_plane = (nk_i8_t *)(payload_base + payload_offset) + key_value_head_idx * plane_bytes;
        nk_i8_t *values_plane = keys_plane + key_value_head_count * plane_bytes;
        // Σk table rides after both plane blocks: one I32 per padded KV position per head.
        nk_i32_t *key_sums_plane = (nk_i32_t *)(payload_base + payload_offset +
                                                2 * key_value_head_count * plane_bytes) +
                                   key_value_head_idx * position_count_padded;
        __m512i const ones_u8x64 = _mm512_set1_epi8(1);
        nk_size_t const depth_groups = depth_padded / 4;
        nk_size_t const depth_blocks = depth_padded / 64; // one 64-channel transpose block = 16 depth quads
        // Restore natural depth-group order after the middle-quad swap.
        static nk_i8_t const group_slot[16] = {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15};

        // Transpose each tile in registers and accumulate its key sums before storing.
        for (nk_size_t tile_idx = 0; tile_idx * 16 < position_count_padded; tile_idx++) {
            nk_i8_t *keys_tile = keys_plane + tile_idx * depth_groups * 64;
            __m512i ksum_accumulator_i32x16 = _mm512_setzero_si512();
            char const *keys_rows[16];
            for (nk_size_t lane_idx = 0; lane_idx < 16; lane_idx++) {
                nk_size_t const position_idx = tile_idx * 16 + lane_idx;
                keys_rows[lane_idx] = position_idx < position_count
                                          ? (char const *)keys + (position_first + position_idx) * key_stride +
                                                key_value_head_idx * depth
                                          : (char const *)0;
            }
            for (nk_size_t block_idx = 0; block_idx < depth_blocks; block_idx++) {
                __m512i rows_i8x64[16];
                nk_size_t const channels_remaining = depth - block_idx * 64;
                nk_size_t const channels = channels_remaining < 64 ? channels_remaining : 64;
                for (nk_size_t lane_idx = 0; lane_idx < 16; lane_idx++) {
                    nk_b512_vec_t row_vec;
                    if (!keys_rows[lane_idx]) row_vec.zmm = _mm512_setzero_si512();
                    else nk_partial_load_b8x64_skylake_(keys_rows[lane_idx] + block_idx * 64, &row_vec, channels);
                    rows_i8x64[lane_idx] = row_vec.zmm;
                }
                __m512i groups_i8x64[16];
                nk_attention_transpose_i32x16x16_icelake_(rows_i8x64, groups_i8x64);
                for (nk_size_t group_idx = 0; group_idx < 16; group_idx++) {
                    _mm512_storeu_si512(keys_tile + (block_idx * 16 + group_slot[group_idx]) * 64,
                                        groups_i8x64[group_idx]);
                    ksum_accumulator_i32x16 = _mm512_dpbusd_epi32(ksum_accumulator_i32x16, ones_u8x64,
                                                                  groups_i8x64[group_idx]);
                }
            }
            _mm512_storeu_si512((__m512i *)(key_sums_plane + tile_idx * 16), ksum_accumulator_i32x16);
        }

        // Interleave four positions per channel for VNNI.
        for (nk_size_t quad_idx = 0; quad_idx < position_count_padded / 4; quad_idx++) {
            nk_i8_t *group_destination = values_plane + quad_idx * depth_padded * 4;
            char const *v_rows[4];
            for (nk_size_t lane_idx = 0; lane_idx < 4; lane_idx++) {
                nk_size_t const position_idx = quad_idx * 4 + lane_idx;
                v_rows[lane_idx] = (position_idx < position_count)
                                       ? (char const *)values + (position_first + position_idx) * value_stride +
                                             key_value_head_idx * depth
                                       : (char const *)0;
            }
            // Pair bytes, then words, to interleave four positions within each 128-bit lane.
            for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16) {
                __m128i rows_i8x16[4];
                nk_size_t const channels_remaining = channel_idx < depth ? depth - channel_idx : 0;
                nk_size_t const channels = channels_remaining < 16 ? channels_remaining : 16;
                for (nk_size_t lane_idx = 0; lane_idx < 4; lane_idx++) {
                    nk_b128_vec_t row_vec;
                    if (!v_rows[lane_idx] || channels == 0) row_vec.xmm = _mm_setzero_si128();
                    else nk_partial_load_b8x16_skylake_(v_rows[lane_idx] + channel_idx, &row_vec, channels);
                    rows_i8x16[lane_idx] = row_vec.xmm;
                }
                __m128i const pair01_low_i8x16 = _mm_unpacklo_epi8(rows_i8x16[0], rows_i8x16[1]);
                __m128i const pair01_high_i8x16 = _mm_unpackhi_epi8(rows_i8x16[0], rows_i8x16[1]);
                __m128i const pair23_low_i8x16 = _mm_unpacklo_epi8(rows_i8x16[2], rows_i8x16[3]);
                __m128i const pair23_high_i8x16 = _mm_unpackhi_epi8(rows_i8x16[2], rows_i8x16[3]);
                nk_i8_t *quad_out = group_destination + channel_idx * 4;
                _mm_storeu_si128((__m128i *)(quad_out + 0), _mm_unpacklo_epi16(pair01_low_i8x16, pair23_low_i8x16));
                _mm_storeu_si128((__m128i *)(quad_out + 16), _mm_unpackhi_epi16(pair01_low_i8x16, pair23_low_i8x16));
                _mm_storeu_si128((__m128i *)(quad_out + 32), _mm_unpacklo_epi16(pair01_high_i8x16, pair23_high_i8x16));
                _mm_storeu_si128((__m128i *)(quad_out + 48), _mm_unpackhi_epi16(pair01_high_i8x16, pair23_high_i8x16));
            }
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_i8_icelake(                          //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,      //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *query_offsets, nk_size_t query_stride,      //
    nk_size_t output_stride, nk_f32_t scale, nk_size_t keys_before,              //
    nk_size_t keys_after, nk_size_t task_begin, nk_size_t task_end, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_(key_value_packed, nk_cap_icelake_k)) return nk_pack_mismatch_k;
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_attention_packed_i8_icelake_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                    depth, query_offsets, query_stride, output_stride, scale, band, task_begin,
                                    task_end);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_ICELAKE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_ICELAKE_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_ATTENTION_ICELAKE_H
