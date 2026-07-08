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
 *  plane (see the payload layout below). To keep the score loop drain-free the score kernel
 *  never reduces across the 16 lanes: K is packed VNNI-interleaved so a 64-byte load holds
 *  four channels of sixteen consecutive KV positions, one score per lane. A query's four
 *  channels broadcast as one dword (`_mm512_set1_epi32`) and one DPBUSD advances sixteen KV
 *  positions by four channels; accumulating over the depth quads leaves sixteen exact
 *  biased scores with no transpose. Sixteen queries share each K load — hold sixteen
 *  accumulators and issue sixteen broadcast DPBUSDs per K vector — so the score cost scales
 *  flat with KV length. The `128·Σk` correction is one sixteen-wide `zmm ≪ 7` subtract per
 *  KV tile. Zero-padded channels are exact: a padded `k = 0` contributes `(q+128)·0 = 0` to
 *  the product and `0` to `Σk`.
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
 *  + panel` correction. K is VNNI-interleaved `[tile of 16 positions][depth quad][16 lanes ×
 *  4 channels]`; both planes zero-pad channels to a multiple of 64, K pads positions to a
 *  multiple of 16 (one 16-lane score tile) and V to a multiple of 4. The row-max sweep covers
 *  live columns only: a zero-padded position's score of 0 could otherwise raise the max and
 *  zero out an all-negative row's weight sum. `depth > 256` routes to the width-agnostic
 *  serial tier from every entry point.
 *
 *  Per-segment payload is `[K planes][V planes][Σk tables]` across `num_kv_heads` heads: the
 *  K and V planes as above (`round_up(len, 16) · dim_padded` bytes each), then one I32 `Σk`
 *  per padded KV position per head. Two extra payload bytes per position per plane pair
 *  equal exactly one I32 per position, so the directory keeps its single closed form with
 *  `unit_bytes = dim_padded + 2` — `2 · num_kv_heads · round_up(len, 16) · (dim_padded + 2)`.
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

/**
 *  @brief Register 16×16 transpose of 32-bit elements (hierarchical unpack + lane shuffle). Given 16 rows
 *         (each 16 dwords), returns the 16 columns. Used at pack time to turn 16 position rows into the
 *         VNNI depth-quad tiles. The `shuffle_i32x4` stages emit rows in the group order `groups[i]` holds
 *         group `{0,1,2,3,8,9,10,11,4,5,6,7,12,13,14,15}[i]`; the caller restores natural order at store.
 */
NK_INTERNAL void nk_attention_transpose_i32x16x16_icelake_(__m512i const rows_i32x16[16], __m512i groups_i32x16[16]) {
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

NK_INTERNAL nk_size_t nk_attention_packed_size_icelake_(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_size_t payload_bytes = 0; // raw I8 K/V planes plus one I32 Σk per KV position, folded as `+ 2` per plane pair
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++)
        payload_bytes += 2 * key_value_head_count *
                         (nk_size_t)nk_size_round_up_to_multiple_(segment_lengths[segment_idx], 16) *
                         (depth_padded + 2);
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
                                 first_task, 16, depth_padded + 2);
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
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 16);
        nk_size_t const plane_bytes = position_count_padded * depth_padded;
        nk_i8_t *keys_plane = (nk_i8_t *)(payload_base + payload_offsets_ro[segment]) +
                              key_value_head_idx * plane_bytes;
        nk_i8_t *values_plane = keys_plane + key_value_head_count * plane_bytes;
        // `Σk` table rides after both plane blocks: one I32 per padded KV position per head.
        nk_i32_t *key_sums_plane = (nk_i32_t *)(payload_base + payload_offsets_ro[segment] +
                                                2 * key_value_head_count * plane_bytes) +
                                   key_value_head_idx * position_count_padded;
        __m512i const ones_u8x64 = _mm512_set1_epi8(1);
        nk_size_t const depth_groups = depth_padded / 4;
        nk_size_t const depth_blocks = depth_padded / 64; // one 64-channel transpose block = 16 depth quads
        // Transposed rows emerge middle-quad-swapped; this maps output row → its natural depth-group slot.
        static nk_i8_t const group_slot[16] = {0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15};

        // K is VNNI-interleaved `[tile of 16 positions][depth quad][lane · 4 + channel % 4]`. Each 16-position
        // tile is a register 16×16 dword transpose per 64-channel block — no scatter — and `Σk` for all 16
        // positions accumulates batched from the transposed groups, one store per tile.
        for (nk_size_t tile_idx = 0; tile_idx * 16 < position_count_padded; tile_idx++) {
            nk_i8_t *keys_tile = keys_plane + tile_idx * depth_groups * 64;
            __m512i ksum_accumulator_i32x16 = _mm512_setzero_si512();
            for (nk_size_t block_idx = 0; block_idx < depth_blocks; block_idx++) {
                __m512i rows_i8x64[16];
                for (nk_size_t lane_idx = 0; lane_idx < 16; lane_idx++) {
                    nk_size_t const position_idx = tile_idx * 16 + lane_idx;
                    if (position_idx < position_count) {
                        nk_i8_t const *keys_row =
                            (nk_i8_t const *)((char const *)keys + (position_first + position_idx) * key_stride_bytes) +
                            key_value_head_idx * depth + block_idx * 64;
                        nk_size_t const channels_remaining = depth - block_idx * 64;
                        __mmask64 const load_m64 = channels_remaining >= 64
                                                       ? ~(__mmask64)0
                                                       : (__mmask64)_bzhi_u64(~(nk_u64_t)0, channels_remaining);
                        rows_i8x64[lane_idx] = _mm512_maskz_loadu_epi8(load_m64, keys_row);
                    }
                    else { rows_i8x64[lane_idx] = _mm512_setzero_si512(); }
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
            // 128-bit 4-way byte interleave, 16 channels per iteration — no lane crossing, no scalar loop:
            // `unpacklo/hi_epi8` pairs the positions, `unpacklo/hi_epi16` groups all four per channel.
            for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16) {
                __m128i rows_i8x16[4];
                for (nk_size_t lane_idx = 0; lane_idx < 4; lane_idx++) {
                    nk_size_t const channels_remaining = channel_idx < depth ? depth - channel_idx : 0;
                    __mmask16 const load_m16 = channels_remaining >= 16
                                                   ? (__mmask16)0xFFFF
                                                   : (__mmask16)_bzhi_u32(0xFFFFu, channels_remaining);
                    rows_i8x16[lane_idx] = v_rows[lane_idx]
                                               ? _mm_maskz_loadu_epi8(load_m16, v_rows[lane_idx] + channel_idx)
                                               : _mm_setzero_si128();
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
}

/**
 *  @brief Drain-free exact I32 scores for a 16-query block over one panel: for each 16-KV
 *         tile it holds 16 lane-parallel accumulators (one score per KV position), broadcasts
 *         each query's four-channel dword and issues 16 DPBUSDs reusing one K load, then
 *         accumulates over the depth quads. No lane reduction and no transpose: `Σq·k` lands
 *         directly per lane. The `128·Σk` correction is one 16-wide `zmm ≪ 7` subtract per
 *         tile. Writes a `16 × panel_width` score block, each query row `panel_width` apart.
 */
NK_INTERNAL void nk_attention_score_block_icelake_(nk_u8_t const *queries_biased, nk_i8_t const *keys_plane,
                                                   nk_i32_t const *key_sums_plane, nk_size_t panel_start,
                                                   nk_size_t panel_len, nk_size_t depth_padded, nk_i32_t *scores) {
    nk_size_t const depth_groups = depth_padded / 4;
    nk_size_t const tile_first = panel_start / 16;
    nk_size_t const tile_count = (panel_len + 15) / 16;
    for (nk_size_t tile_idx = 0; tile_idx < tile_count; tile_idx++) {
        __m512i accumulator_i32x16[16];
        for (nk_size_t query_idx = 0; query_idx < 16; query_idx++)
            accumulator_i32x16[query_idx] = _mm512_setzero_si512();
        nk_i8_t const *keys_tile = keys_plane + (tile_first + tile_idx) * depth_groups * 64;
        for (nk_size_t group_idx = 0; group_idx < depth_groups; group_idx++) {
            __m512i const k_i8x64 = _mm512_loadu_si512(keys_tile + group_idx * 64); // 16 positions × 4 channels
            for (nk_size_t query_idx = 0; query_idx < 16; query_idx++) {
                __m512i const query_quad_u8x64 = _mm512_broadcastd_epi32(
                    _mm_loadu_si32(queries_biased + query_idx * nk_attention_max_depth_icelake_k_ + group_idx * 4));
                accumulator_i32x16[query_idx] = _mm512_dpbusd_epi32(accumulator_i32x16[query_idx], query_quad_u8x64,
                                                                    k_i8x64);
            }
        }
        __m512i const correction_i32x16 = _mm512_slli_epi32(
            _mm512_loadu_si512(key_sums_plane + panel_start + tile_idx * 16), 7); // × 128
        for (nk_size_t query_idx = 0; query_idx < 16; query_idx++)
            _mm512_storeu_si512(scores + query_idx * nk_attention_panel_icelake_k_ + tile_idx * 16,
                                _mm512_sub_epi32(accumulator_i32x16[query_idx], correction_i32x16));
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
    __mmask16 const tail_m16 = (__mmask16)((1u << (panel_len - full)) - 1);

    __m512 max_f32x16 = _mm512_set1_ps(NK_F32_MIN);
    nk_size_t position_idx = 0;
    for (; position_idx < full; position_idx += 16)
        max_f32x16 = _mm512_max_ps(
            max_f32x16, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_load_si512(scores + position_idx)), scale2_f32x16));
    if (position_idx < panel_len)
        max_f32x16 = _mm512_max_ps(
            max_f32x16,
            _mm512_mask_mul_ps(_mm512_set1_ps(NK_F32_MIN), tail_m16,
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
            tail_m16, nk_attention_exp2_f32x16_skylake_(_mm512_fmsub_ps(
                          _mm512_cvtepi32_ps(_mm512_load_si512(scores + position_idx)), scale2_f32x16, max2_f32x16)));
        __m512i const weight_u32x16 = _mm512_maskz_cvttps_epu32(
            tail_m16, _mm512_add_ps(_mm512_mul_ps(exp_f32x16, amplitude_f32x16), half_f32x16));
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
        __m512i accumulator_i32x16 = _mm512_setzero_si512();
        for (nk_size_t quad_idx = 0; quad_idx < quad_count; quad_idx++) {
            __m512i const weights_u8x64 = _mm512_broadcastd_epi32(_mm_loadu_si32(weights + quad_idx * 4));
            __m512i const v_quad_i8x64 = _mm512_loadu_si512(values_plane + (quad_start + quad_idx) * depth_padded * 4 +
                                                            channel_idx * 4);
            accumulator_i32x16 = _mm512_dpbusd_epi32(accumulator_i32x16, weights_u8x64, v_quad_i8x64);
        }
        __m512 const scaled_f32x16 = _mm512_mul_ps(_mm512_load_ps(output_row + channel_idx), correction_f32x16);
        _mm512_store_ps(output_row + channel_idx, _mm512_add_ps(scaled_f32x16, _mm512_cvtepi32_ps(accumulator_i32x16)));
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

    // 16-query blocks share each K load; each query keeps its own biased Q, running state and F32 output row.
    NK_ALIGN64 nk_u8_t queries_biased[16 * nk_attention_max_depth_icelake_k_];
    NK_ALIGN64 nk_f32_t output_rows[16 * nk_attention_max_depth_icelake_k_];
    NK_ALIGN64 nk_i32_t scores[16 * nk_attention_panel_icelake_k_];
    NK_ALIGN64 nk_u8_t weights[nk_attention_panel_icelake_k_];
    nk_f32_t running_max2[16], running_sum[16];
    __m512i const xor_mask_u8x64 = _mm512_set1_epi8((char)0x80);
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const dim_tail_m16 = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment = task_idx / head_count, head = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment];
        nk_size_t const row_count = query_offsets[segment + 1] - query_offsets[segment];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 16);
        nk_size_t const plane_bytes = position_count_padded * depth_padded;
        nk_i8_t const *keys_plane = (nk_i8_t const *)(payload_base + payload_offsets[segment]) +
                                    (head / head_group_size) * plane_bytes;
        nk_i8_t const *values_plane = keys_plane + key_value_head_count * plane_bytes;
        nk_i32_t const *key_sums_plane = (nk_i32_t const *)(payload_base + payload_offsets[segment] +
                                                            2 * key_value_head_count * plane_bytes) +
                                         (head / head_group_size) * position_count_padded;

        for (nk_size_t row_block = 0; row_block < row_count; row_block += 16) {
            nk_size_t const block_rows = (row_count - row_block < 16) ? (row_count - row_block) : 16;

            // Bias each query in the block into the unsigned domain; padded channels become 0x80 = q of 0.
            for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                nk_i8_t const *query_row = (nk_i8_t const *)((char const *)queries +
                                                             (query_offsets[segment] + row_block + block_row) *
                                                                 query_stride_bytes) +
                                           head * depth;
                nk_u8_t *query_biased = queries_biased + block_row * nk_attention_max_depth_icelake_k_;
                for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 64) {
                    __mmask64 const load_m64 = (channel_idx + 64 <= depth) ? ~(__mmask64)0
                                               : (channel_idx < depth)
                                                   ? (__mmask64)_bzhi_u64(~(nk_u64_t)0, depth - channel_idx)
                                                   : (__mmask64)0;
                    _mm512_store_si512(
                        query_biased + channel_idx,
                        _mm512_xor_si512(_mm512_maskz_loadu_epi8(load_m64, query_row + channel_idx), xor_mask_u8x64));
                }
                for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 16)
                    _mm512_store_ps(output_rows + block_row * nk_attention_max_depth_icelake_k_ + channel_idx,
                                    _mm512_setzero_ps());
                running_max2[block_row] = NK_F32_MIN;
                running_sum[block_row] = 0;
            }
            // Zero the unused query slots so the fixed-16 score kernel reads defined biased bytes.
            for (nk_size_t block_row = block_rows; block_row < 16; block_row++)
                for (nk_size_t channel_idx = 0; channel_idx < depth_padded; channel_idx += 64)
                    _mm512_store_si512(queries_biased + block_row * nk_attention_max_depth_icelake_k_ + channel_idx,
                                       _mm512_setzero_si512());

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_len = (panel_start + panel_width <= position_count)
                                                ? panel_width
                                                : (position_count - panel_start);
                nk_attention_score_block_icelake_(queries_biased, keys_plane, key_sums_plane, panel_start, panel_len,
                                                  depth_padded, scores);
                for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                    nk_f32_t const correction = nk_attention_softmax_panel_icelake_(
                        scores + block_row * nk_attention_panel_icelake_k_, weights, panel_len, scale2,
                        &running_max2[block_row], &running_sum[block_row]);
                    nk_attention_weighted_sum_panel_icelake_(
                        weights, values_plane, panel_start, panel_len, depth_padded, correction,
                        output_rows + block_row * nk_attention_max_depth_icelake_k_);
                }
            }

            for (nk_size_t block_row = 0; block_row < block_rows; block_row++) {
                __m512 const inverse_sum_f32x16 = _mm512_set1_ps(1.0f / running_sum[block_row]);
                nk_f32_t const *output_row = output_rows + block_row * nk_attention_max_depth_icelake_k_;
                nk_f32_t *destination = output +
                                        (query_offsets[segment] + row_block + block_row) * output_stride_floats +
                                        head * depth;
                nk_size_t channel_idx = 0;
                for (; channel_idx < depth_full; channel_idx += 16)
                    _mm512_storeu_ps(destination + channel_idx,
                                     _mm512_mul_ps(_mm512_load_ps(output_row + channel_idx), inverse_sum_f32x16));
                if (channel_idx < depth)
                    _mm512_mask_storeu_ps(destination + channel_idx, dim_tail_m16,
                                          _mm512_mul_ps(_mm512_load_ps(output_row + channel_idx), inverse_sum_f32x16));
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

#endif // NK_TARGET_ICELAKE
#endif // NK_TARGET_X8664_
#endif // NK_ATTENTION_ICELAKE_H
