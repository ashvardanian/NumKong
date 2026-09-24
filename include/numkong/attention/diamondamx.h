/**
 *  @file include/numkong/attention/diamondamx.h
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *  @brief Ragged attention for Diamond Rapids AMX — the FP8 E4M3 variant only.
 *
 *  @sa include/numkong/attention.h
 *
 *  Diamond Rapids AMX debuts ~2027. This backend provides only the E4M3 variant: Diamond's
 *  differentiator over Sapphire Rapids is native FP8, via @c _tile_dphf8ps, whereas its I8/BF16
 *  tile pipelines are near-clones of `attention/sapphireamx.h`, the same TDPBSSD/TDPBUSD/TDPBF16PS,
 *  and are served there. It keeps the Sapphire Rapids panel-flash structure — 2×2 register
 *  blocking, KV-reuse chunking, and the base-2 streaming softmax reused from the Skylake tier — and
 *  layers on Diamond's two upgrades:
 *
 *  - Accumulator tiles drain straight into ZMMs with a tile-row move instead of a @c _tile_stored
 *    memory round-trip: @c _tile_movrow casts an FP32-accumulator row into a ZMM,
 *    @c _tile_cvtrowd2ps for the INT32 → FP32 case.
 *  - E4M3 runs natively through @c _tile_dphf8ps on raw E4M3 tiles — quad-interleaved like the I8
 *    layout, so the packed KV blob is half the Sapphire Rapids BF16-widened size — with softmax
 *    probabilities quantized to E4M3.
 *
 *  Hardware debuts ~2027: correctness is SDE-validated; performance claims await silicon.
 */
#ifndef NK_ATTENTION_DIAMONDAMX_H
#define NK_ATTENTION_DIAMONDAMX_H

#if NK_TARGET_X8664_
#if NK_TARGET_DIAMONDAMX

#include "numkong/attention/serial.h" // shared packed-KV offsets, width-agnostic fallback
#include "numkong/each/skylake.h"     // `nk_exp2_f32x16_skylake_`
#include "numkong/reduce/skylake.h"   // `nk_reduce_add_f32x16_skylake_`, `nk_reduce_max_f32x16_skylake_`
#include "numkong/dots/sapphireamx.h" // tile config, BF16/I8 tile structs, load_a, transposers

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__) && __clang_major__ >= 22
#pragma clang attribute push(                                                                                                            \
    __attribute__((target(                                                                                                               \
        "avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx10.2,f16c,fma,bmi,bmi2,amx-tile,amx-bf16,amx-int8,amx-fp8,amx-avx512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                                                \
    __attribute__((target(                                                                                                                   \
        "avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx10.2-512,f16c,fma,bmi,bmi2,amx-tile,amx-bf16,amx-int8,amx-fp8,amx-avx512"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx10.2", "f16c", "fma", \
                   "bmi", "bmi2", "amx-tile", "amx-bf16", "amx-int8", "amx-fp8", "amx-avx512")
#endif

enum {

    /** KV panel width in positions; the F32 score row stays L1-resident. */
    nk_attention_panel_diamondamx_k_ = 512,

    /** Widest head this backend handles in registers; larger heads route to the serial tier. */
    nk_attention_max_depth_diamondamx_k_ = 256,
};

/** Drains the four FP32 accumulator tiles (TDPBF16PS / TDPHF8PS) of one 2×2 register block to
 *  the score panel, one ZMM row per tile per iteration. Tiles 0/1 fill the top 16 rows, 2/3 the
 *  bottom 16; @c _tile_movrow takes the tile as a compile-time immediate, so all four are
 *  written out explicitly. */
NK_HELPER_INLINE void nk_attention_f32_store_grid_diamondamx_(nk_f32_t *scores_panel, nk_size_t pair_idx,
                                                              nk_size_t panel_width) {
    for (unsigned row_idx = 0; row_idx < 16; row_idx++) {
        nk_f32_t *top_row = scores_panel + pair_idx * 32 + row_idx * panel_width;
        nk_f32_t *bottom_row = scores_panel + 16 * panel_width + pair_idx * 32 + row_idx * panel_width;
        _mm512_storeu_ps(top_row, _mm512_castsi512_ps(_tile_movrow(0, row_idx)));
        _mm512_storeu_ps(top_row + 16, _mm512_castsi512_ps(_tile_movrow(1, row_idx)));
        _mm512_storeu_ps(bottom_row, _mm512_castsi512_ps(_tile_movrow(2, row_idx)));
        _mm512_storeu_ps(bottom_row + 16, _mm512_castsi512_ps(_tile_movrow(3, row_idx)));
    }
}

/** Fuses the four FP32 accumulator tiles of one 2×2 register block into @p o_acc row-wise as o = o
 *  · correction + drained. Tiles 0/1 use the top row-tile's corrections, 2/3 the bottom's;
 *  @c _tile_movrow needs a compile-time tile immediate, so all four are written out explicitly. */
NK_HELPER_INLINE void nk_attention_f32_accumulate_grid_diamondamx_(nk_size_t channel_start, nk_f32_t *o_acc,
                                                                   nk_size_t output_stride_floats,
                                                                   nk_f32_t const (*corrections)[16]) {
    for (unsigned row_idx = 0; row_idx < 16; row_idx++) {
        nk_f32_t *top_row = o_acc + row_idx * output_stride_floats + channel_start;
        nk_f32_t *bottom_row = o_acc + (16 + row_idx) * output_stride_floats + channel_start;
        __m512 const top_correction = _mm512_set1_ps(corrections[0][row_idx]);
        __m512 const bottom_correction = _mm512_set1_ps(corrections[1][row_idx]);
        _mm512_store_ps(top_row, _mm512_fmadd_ps(_mm512_load_ps(top_row), top_correction,
                                                 _mm512_castsi512_ps(_tile_movrow(0, row_idx))));
        _mm512_store_ps(top_row + 16, _mm512_fmadd_ps(_mm512_load_ps(top_row + 16), top_correction,
                                                      _mm512_castsi512_ps(_tile_movrow(1, row_idx))));
        _mm512_store_ps(bottom_row, _mm512_fmadd_ps(_mm512_load_ps(bottom_row), bottom_correction,
                                                    _mm512_castsi512_ps(_tile_movrow(2, row_idx))));
        _mm512_store_ps(bottom_row + 16, _mm512_fmadd_ps(_mm512_load_ps(bottom_row + 16), bottom_correction,
                                                         _mm512_castsi512_ps(_tile_movrow(3, row_idx))));
    }
}

/** Quantizes 16 F32 probabilities to E4M3: F32 → F16 (VCVTPS2PHX) then F16 → E4M3 (VCVT2PH2HF8).
 *  The second VCVT2PH2HF8 operand fills the result's low half, so the payload rides there. */
NK_HELPER_INLINE __m128i nk_attention_quantize_e4m3x16_diamondamx_(__m512 weights_f32x16) {
    __m512h const weights_f16x32 = _mm512_castph256_ph512(_mm512_cvtxps_ph(weights_f32x16));
    return _mm512_castsi512_si128(_mm512_cvts_2ph_hf8(_mm512_setzero_ph(), weights_f16x32));
}

/** Widens 16 E4M3 probabilities back to F32 (VCVTHF82PH + VCVTPH2PSX) for a consistent sum. */
NK_HELPER_INLINE __m512 nk_attention_dequantize_e4m3x16_diamondamx_(__m128i weights_e4m3x16) {
    return _mm512_cvtxph_ps(_mm256_cvthf8_ph(weights_e4m3x16));
}

/** E4M3 native and I8 share a raw 1-byte, 64-deep, quad-interleaved tile layout. */
NK_HELPER_INLINE nk_size_t nk_attention_pack_size_quad_diamondamx_(nk_size_t key_value_head_count, nk_size_t depth,
                                                                   nk_u32_t const *segment_lengths,
                                                                   nk_size_t segment_count) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_size_t total_tile_bytes = 0;
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++) {
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(segment_lengths[segment_idx], 64);
        total_tile_bytes += 2 * key_value_head_count * position_count_padded * depth_padded; // one byte per element
    }
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + total_tile_bytes;
}

NK_API_COMPTIME nk_size_t nk_attention_pack_size_e4m3_diamondamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                                 nk_u32_t const *segment_lengths,
                                                                 nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_diamondamx_k_)
        return nk_attention_pack_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_pack_size_quad_diamondamx_(key_value_head_count, depth, segment_lengths, segment_count);
}

NK_API_COMPTIME void nk_attention_packed_shape_e4m3_diamondamx(void const *key_value_packed, nk_size_t *heads,
                                                               nk_size_t *depth, nk_size_t *segments) {
    nk_attention_packed_shape_(key_value_packed, heads, depth, segments);
}

/** Raw 1-byte packing shared by E4M3-native and I8: K transposed, V quad-interleaved. */
NK_HELPER_INLINE void nk_attention_pack_quad_diamondamx_(                                        //
    nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t task_begin,
    nk_size_t task_end) {

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_size_t const depth_blocks = depth_padded / 64;
    nk_size_t const channel_tiles = depth_padded / 16;
    nk_size_t const tile_bytes = 1024;

    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 task_begin, 64, depth_padded);
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *tile_offsets_ro = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *tiles_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (task_begin >= total_tasks) return;
    if (task_end > total_tasks) task_end = total_tasks;

    __m512i const quad_interleave_index_u8x64 = _mm512_setr_epi32( //
        0x30201000, 0x31211101, 0x32221202, 0x33231303,            //
        0x34241404, 0x35251505, 0x36261606, 0x37271707,            //
        0x38281808, 0x39291909, 0x3A2A1A0A, 0x3B2B1B0B,            //
        0x3C2C1C0C, 0x3D2D1D0D, 0x3E2E1E0E, 0x3F2F1F0F);

    for (nk_size_t task_idx = task_begin; task_idx < task_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count, h = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 64);
        nk_size_t const bytes_per_head = position_count_padded * depth_padded;

        nk_i8_t *keys_head_tiles = (nk_i8_t *)(tiles_base + tile_offsets_ro[segment_idx] + h * bytes_per_head);
        for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / 16; position_tile_idx++) {
            nk_size_t const row_start = position_tile_idx * 16;
            nk_size_t const valid_rows = (row_start + 16 <= position_count) ? 16
                                         : (row_start < position_count)     ? position_count - row_start
                                                                            : 0;
            for (nk_size_t depth_block_idx = 0; depth_block_idx < depth_blocks; depth_block_idx++) {
                nk_size_t const channel_start = depth_block_idx * 64;
                nk_size_t const valid_columns = (channel_start + 64 <= depth) ? 64
                                                : (channel_start < depth)     ? depth - channel_start
                                                                              : 0;
                nk_i8_t const *source = (nk_i8_t const *)((char const *)keys +
                                                          (position_first + row_start) * key_stride_bytes) +
                                        h * depth + channel_start;
                nk_dots_i8_a16x64_sapphireamx_t source_tile;
                nk_dots_i8_b64x16_sapphireamx_t transposed_tile;
                nk_dots_i8_load_a_sapphireamx_(&source_tile, source, key_stride_bytes, valid_rows, valid_columns);
                nk_dots_pack_i8_transposed_sapphireamx_(&source_tile, &transposed_tile);
                nk_i8_t *tile_output = keys_head_tiles +
                                       (position_tile_idx * depth_blocks + depth_block_idx) * tile_bytes;
                for (nk_size_t i = 0; i < tile_bytes; i += 64)
                    _mm512_storeu_si512(tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
            }
        }

        nk_i8_t *values_head_tiles = (nk_i8_t *)(tiles_base + tile_offsets_ro[segment_idx] +
                                                 (key_value_head_count + h) * bytes_per_head);
        for (nk_size_t channel_tile_idx = 0; channel_tile_idx < channel_tiles; channel_tile_idx++) {
            nk_size_t const channel_start = channel_tile_idx * 16;
            nk_size_t const valid_columns = (channel_start + 16 <= depth) ? 16
                                            : (channel_start < depth)     ? depth - channel_start
                                                                          : 0;
            __mmask16 const columns_mask = (__mmask16)((1u << valid_columns) - 1);
            for (nk_size_t position_block_idx = 0; position_block_idx < position_count_padded / 64;
                 position_block_idx++) {
                nk_i8_t *tile_output = values_head_tiles +
                                       (channel_tile_idx * (position_count_padded / 64) + position_block_idx) *
                                           tile_bytes;
                for (nk_size_t quad_idx = 0; quad_idx < 16; quad_idx++) {
                    nk_size_t const quad_start = position_block_idx * 64 + quad_idx * 4;
                    __m512i quad_i8x64 = _mm512_setzero_si512();
                    for (nk_size_t lane_idx = 0; lane_idx < 4; lane_idx++) {
                        if (quad_start + lane_idx >= position_count) continue;
                        char const *row_ptr = (char const *)values +
                                              (position_first + quad_start + lane_idx) * value_stride_bytes +
                                              h * depth + channel_start;
                        __m128i const row_i8x16 = _mm_maskz_loadu_epi8(columns_mask, row_ptr);
                        switch (lane_idx) {
                        case 0: quad_i8x64 = _mm512_inserti32x4(quad_i8x64, row_i8x16, 0); break;
                        case 1: quad_i8x64 = _mm512_inserti32x4(quad_i8x64, row_i8x16, 1); break;
                        case 2: quad_i8x64 = _mm512_inserti32x4(quad_i8x64, row_i8x16, 2); break;
                        case 3: quad_i8x64 = _mm512_inserti32x4(quad_i8x64, row_i8x16, 3); break;
                        }
                    }
                    _mm512_storeu_si512(tile_output + quad_idx * 64,
                                        _mm512_permutexvar_epi8(quad_interleave_index_u8x64, quad_i8x64));
                }
            }
        }
    }
    nk_compiler_barrier_sapphireamx_();
}

NK_API_COMPTIME void nk_attention_pack_e4m3_diamondamx( //
    nk_e4m3_t const *keys, nk_e4m3_t const *values,     //
    nk_size_t key_value_head_count, nk_size_t depth,    //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t task_begin,
    nk_size_t task_end) {
    if (depth > nk_attention_max_depth_diamondamx_k_) {
        nk_attention_pack_e4m3_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, task_begin,
                                      task_end);
        return;
    }
    nk_attention_pack_quad_diamondamx_((nk_i8_t const *)keys, (nk_i8_t const *)values, key_value_head_count, depth,
                                       segment_offsets, segment_lengths, segment_count, key_stride_bytes,
                                       value_stride_bytes, key_value_packed, task_begin, task_end);
}

/** Lanes of the 16 columns starting at @p position_idx that fall between @p column_begin inclusive
 *  and @p column_end exclusive. */
NK_HELPER_INLINE __mmask16 nk_attention_columns_mask_diamondamx_(nk_size_t position_idx, nk_size_t column_begin,
                                                                 nk_size_t column_end) {
    nk_u32_t const below = column_begin > position_idx ? (nk_u32_t)(column_begin - position_idx) : 0;
    nk_u32_t const above = column_end - position_idx >= 16 ? 16 : (nk_u32_t)(column_end - position_idx);
    return (__mmask16)(((1u << above) - 1) & ~((1u << below) - 1));
}

/** Row maxima of an FP32 score panel, row @c r over its columns from @p column_begins at @c r
 *  inclusive to @p column_ends at @c r exclusive. */
NK_HELPER_INLINE void nk_attention_panel_rowmax_diamondamx_(nk_f32_t const *scores_panel, nk_size_t panel_width,
                                                            nk_size_t const *column_begins,
                                                            nk_size_t const *column_ends, nk_f32_t (*panel_max)[16]) {
    for (nk_size_t row_idx = 0; row_idx < 32; row_idx++) {
        nk_f32_t const *scores_row = scores_panel + row_idx * panel_width;
        nk_size_t const column_begin = column_begins[row_idx], column_end = column_ends[row_idx];
        __m512 max_f32x16 = _mm512_set1_ps(NK_F32_MIN);
        for (nk_size_t position_idx = column_begin & ~(nk_size_t)15; position_idx < column_end; position_idx += 16)
            max_f32x16 = _mm512_mask_max_ps(
                max_f32x16, nk_attention_columns_mask_diamondamx_(position_idx, column_begin, column_end), max_f32x16,
                _mm512_loadu_ps(scores_row + position_idx));
        panel_max[row_idx / 16][row_idx % 16] = nk_reduce_max_f32x16_skylake_(max_f32x16);
    }
}

/** E4M3 native Q × Kᵀ: @c _tile_dphf8ps over 64-deep raw E4M3 tiles, drained to FP32. */
NK_HELPER_INLINE void nk_attention_score_panel_e4m3_diamondamx_(
    nk_dots_i8_a16x64_sapphireamx_t const (*queries_tiles)[4], nk_i8_t const *keys_head_tiles,
    nk_size_t panel_first_tile, nk_size_t panel_pairs, nk_size_t depth_blocks, nk_f32_t *scores_panel,
    nk_size_t panel_width) {
    nk_size_t const keys_tile_stride = depth_blocks * 1024;
    for (nk_size_t pair_idx = 0; pair_idx < panel_pairs; pair_idx++) {
        nk_i8_t const *keys_tile0 = keys_head_tiles + (panel_first_tile + pair_idx * 2) * keys_tile_stride;
        nk_i8_t const *keys_tile1 = keys_tile0 + keys_tile_stride;
        _tile_zero(0);
        _tile_zero(1);
        _tile_zero(2);
        _tile_zero(3);
        for (nk_size_t depth_block_idx = 0; depth_block_idx < depth_blocks; depth_block_idx++) {
            _tile_loadd(4, queries_tiles[0][depth_block_idx].data, 64);
            _tile_loadd(5, queries_tiles[1][depth_block_idx].data, 64);
            _tile_loadd(6, keys_tile0 + depth_block_idx * 1024, 64);
            _tile_loadd(7, keys_tile1 + depth_block_idx * 1024, 64);
            _tile_dphf8ps(0, 4, 6);
            _tile_dphf8ps(1, 4, 7);
            _tile_dphf8ps(2, 5, 6);
            _tile_dphf8ps(3, 5, 7);
        }
        nk_attention_f32_store_grid_diamondamx_(scores_panel, pair_idx, panel_width);
    }
}

/** Streaming base-2 softmax → E4M3 weights, stored as e4m3(256 · 2^(s₂ − m₂)). The × 256 amplitude
 *  (the U8 tier's 255, rounded to a power of two so it divides out exactly) keeps every live weight
 *  down to 2⁻¹⁴ of the row maximum in E4M3's @b normal range: TDPHF8PS treats subnormal inputs as
 *  zero, so unscaled sub-2⁻⁶ weights would vanish from the P × V numerator while the dequantized
 *  sum kept them — a shrink-toward-zero bias that grows with context length. The sum accumulates
 *  the dequantized scaled weights, so the 256 cancels in normalization. */
NK_HELPER_INLINE void nk_attention_exp_panel_e4m3_diamondamx_(nk_f32_t const *scores_panel, nk_e4m3_t *weights_panel,
                                                              nk_size_t panel_length, nk_size_t const *column_begins,
                                                              nk_size_t const *column_ends, nk_size_t panel_width,
                                                              nk_f32_t scale2, nk_f32_t const (*new_max)[16],
                                                              nk_f32_t (*panel_sums)[16]) {
    __m512 const scale_f32x16 = _mm512_set1_ps(scale2);
    __m512 const amplitude_f32x16 = _mm512_set1_ps(256.0f);
    for (nk_size_t row_idx = 0; row_idx < 32; row_idx++) {
        nk_f32_t const *scores_row = scores_panel + row_idx * panel_width;
        nk_e4m3_t *weights_row = weights_panel + row_idx * panel_width;
        nk_size_t const column_begin = column_begins[row_idx], column_end = column_ends[row_idx];
        __m512 const max_f32x16 = _mm512_set1_ps(new_max[row_idx / 16][row_idx % 16]);
        __m512 sum_f32x16 = _mm512_setzero_ps();
        for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx += 16) {
            if (position_idx + 16 <= column_begin || position_idx >= column_end) {
                _mm_store_si128((__m128i *)(weights_row + position_idx), _mm_setzero_si128());
                continue;
            }
            __mmask16 const columns_mask = nk_attention_columns_mask_diamondamx_(position_idx, column_begin,
                                                                                 column_end);
            __m512 const exp_f32x16 = _mm512_maskz_mov_ps(
                columns_mask, _mm512_mul_ps(nk_exp2_f32x16_skylake_(_mm512_fmsub_ps(
                                                _mm512_loadu_ps(scores_row + position_idx), scale_f32x16, max_f32x16)),
                                            amplitude_f32x16));
            __m128i const weight_e4m3x16 = nk_attention_quantize_e4m3x16_diamondamx_(exp_f32x16);
            sum_f32x16 = _mm512_add_ps(sum_f32x16, nk_attention_dequantize_e4m3x16_diamondamx_(weight_e4m3x16));
            _mm_store_si128((__m128i *)(weights_row + position_idx), weight_e4m3x16);
        }
        panel_sums[row_idx / 16][row_idx % 16] = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    }
}

/** E4M3 native P × V: @c _tile_dphf8ps (E4M3 weights × E4M3 values → FP32), fused on drain. */
NK_HELPER_INLINE void nk_attention_weighted_sum_panel_e4m3_diamondamx_(
    nk_e4m3_t const *weights_panel, nk_size_t panel_width, nk_i8_t const *values_head_tiles,
    nk_size_t position_blocks_total, nk_size_t panel_first_block, nk_size_t panel_blocks, nk_size_t depth_tiles,
    nk_size_t output_stride_floats, nk_f32_t const (*corrections)[16], nk_f32_t *o_acc) {
    int const weights_stride_bytes = (int)panel_width;
    for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles; depth_tile_idx++) {
        nk_i8_t const *values_tile0 = values_head_tiles +
                                      ((depth_tile_idx * 2 + 0) * position_blocks_total + panel_first_block) * 1024;
        nk_i8_t const *values_tile1 = values_head_tiles +
                                      ((depth_tile_idx * 2 + 1) * position_blocks_total + panel_first_block) * 1024;
        _tile_zero(0);
        _tile_zero(1);
        _tile_zero(2);
        _tile_zero(3);
        for (nk_size_t position_block_idx = 0; position_block_idx < panel_blocks; position_block_idx++) {
            _tile_loadd(4, weights_panel + position_block_idx * 64, weights_stride_bytes);
            _tile_loadd(5, weights_panel + 16 * panel_width + position_block_idx * 64, weights_stride_bytes);
            _tile_loadd(6, values_tile0 + position_block_idx * 1024, 64);
            _tile_loadd(7, values_tile1 + position_block_idx * 1024, 64);
            _tile_dphf8ps(0, 4, 6);
            _tile_dphf8ps(1, 4, 7);
            _tile_dphf8ps(2, 5, 6);
            _tile_dphf8ps(3, 5, 7);
        }
        nk_attention_f32_accumulate_grid_diamondamx_(depth_tile_idx * 32, o_acc, output_stride_floats, corrections);
    }
}

/** E4M3 native per-call scratch: FP32 scores, E4M3 weights, output accumulators, raw Q tiles. */
typedef struct {
    NK_ALIGN64 nk_f32_t scores_panel[32 * nk_attention_panel_diamondamx_k_];
    NK_ALIGN64 nk_e4m3_t weights_panel[32 * nk_attention_panel_diamondamx_k_];
    NK_ALIGN64 nk_f32_t o_acc[4][32 * nk_attention_max_depth_diamondamx_k_];
    nk_dots_i8_a16x64_sapphireamx_t queries_tiles[4][2][4];
} nk_attention_scratch_e4m3_diamondamx_t_;

/** E4M3 native (segment, head) task: @c _tile_dphf8ps scores and P × V with E4M3-quantized weights.
 *  Row @c r reads only the keys that @c nk_attention_row_range_ admits for position r +
 *  @p diagonal_offset and @p window. */
NK_HELPER_INLINE void nk_attention_task_e4m3_diamondamx_(
    nk_e4m3_t const *queries, nk_f32_t *output, nk_i8_t const *keys_head_tiles, nk_i8_t const *values_head_tiles,
    nk_size_t head_idx, nk_size_t depth, nk_size_t position_count, nk_size_t position_count_padded,
    nk_size_t query_first, nk_size_t row_count, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes,
    nk_f32_t scale2, nk_i64_t diagonal_offset, nk_size_t window, nk_attention_scratch_e4m3_diamondamx_t_ *scratch) {
    nk_size_t const panel_width = nk_attention_panel_diamondamx_k_;
    nk_size_t const row_blocks = 4;
    nk_size_t const kv_padded = nk_size_round_up_to_multiple_(position_count, 64);
    nk_size_t const position_blocks_total = position_count_padded / 64;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_size_t const depth_blocks = depth_padded / 64;
    nk_size_t const depth_tiles = depth_padded / 32;
    nk_size_t const output_stride_floats = depth_padded;
    nk_size_t const output_stride_out = output_stride_bytes / sizeof(nk_f32_t);

    NK_ALIGN64 nk_f32_t panel_max[2][16], corrections[2][16], new_max_arr[2][16], panel_sums[2][16];
    __m512 row_max2_f32x16[4][2], row_sum_f32x16[4][2];
    nk_size_t key_begins[4][32], key_ends[4][32], block_key_begin[4], block_key_end[4];
    nk_size_t column_begins[32], column_ends[32];
    __m512 const zero_f32x16 = _mm512_setzero_ps();
    __m512 const scale2_f32x16 = _mm512_set1_ps(scale2);
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const depth_tail_mask = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t row_block_start = 0; row_block_start < row_count; row_block_start += 32 * row_blocks) {
        nk_size_t const row_block_count = ((row_count - row_block_start + 31) / 32 < row_blocks)
                                              ? (row_count - row_block_start + 31) / 32
                                              : row_blocks;
        nk_size_t group_key_begin = position_count, group_key_end = 0;
        for (nk_size_t row_block_idx = 0; row_block_idx < row_block_count; row_block_idx++) {
            // Block range is the union of its rows' ranges; padding rows past `row_count` stay empty
            block_key_begin[row_block_idx] = position_count, block_key_end[row_block_idx] = 0;
            for (nk_size_t row_idx = 0; row_idx < 32; row_idx++) {
                nk_size_t const row = row_block_start + row_block_idx * 32 + row_idx;
                nk_size_t key_begin = 0, key_end = 0;
                if (row < row_count)
                    nk_attention_row_range_((nk_i64_t)row + diagonal_offset, window, position_count, &key_begin,
                                            &key_end);
                key_begins[row_block_idx][row_idx] = key_begin, key_ends[row_block_idx][row_idx] = key_end;
                if (key_begin >= key_end) continue;
                if (key_begin < block_key_begin[row_block_idx]) block_key_begin[row_block_idx] = key_begin;
                if (key_end > block_key_end[row_block_idx]) block_key_end[row_block_idx] = key_end;
            }
            if (block_key_begin[row_block_idx] < group_key_begin) group_key_begin = block_key_begin[row_block_idx];
            if (block_key_end[row_block_idx] > group_key_end) group_key_end = block_key_end[row_block_idx];
            for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                nk_size_t const row_start = row_block_start + row_block_idx * 32 + row_tile_idx * 16;
                nk_size_t const valid_rows = row_start >= row_count          ? 0
                                             : (row_count - row_start >= 16) ? 16
                                                                             : row_count - row_start;
                for (nk_size_t depth_block_idx = 0; depth_block_idx < depth_blocks; depth_block_idx++) {
                    nk_size_t const channel_start = depth_block_idx * 64;
                    nk_size_t const valid_columns = (channel_start + 64 <= depth) ? 64 : depth - channel_start;
                    nk_dots_i8_load_a_sapphireamx_(
                        &scratch->queries_tiles[row_block_idx][row_tile_idx][depth_block_idx],
                        (nk_i8_t const *)((char const *)queries + (query_first + row_start) * query_stride_bytes) +
                            head_idx * depth + channel_start,
                        query_stride_bytes, valid_rows, valid_columns);
                }
                row_max2_f32x16[row_block_idx][row_tile_idx] = _mm512_set1_ps(NK_F32_MIN);
                row_sum_f32x16[row_block_idx][row_tile_idx] = _mm512_setzero_ps();
            }
        }

        nk_size_t const panels_start = group_key_begin < group_key_end ? group_key_begin / panel_width * panel_width
                                                                       : kv_padded;
        // Panels stay aligned to `panel_width`, and only those intersecting the group's key range run
        for (nk_size_t panel_start = panels_start; panel_start < group_key_end; panel_start += panel_width) {
            nk_size_t const panel_length = (panel_start + panel_width <= kv_padded) ? panel_width
                                                                                    : (kv_padded - panel_start);
            nk_size_t const valid_cols = (panel_start + panel_length <= position_count)
                                             ? panel_length
                                             : (position_count - panel_start);
            nk_size_t const panel_end = panel_start + valid_cols;
            for (nk_size_t row_block_idx = 0; row_block_idx < row_block_count; row_block_idx++) {
                if (block_key_begin[row_block_idx] >= panel_end || block_key_end[row_block_idx] <= panel_start)
                    continue;
                __mmask16 live_rows[2] = {0, 0};
                for (nk_size_t row_idx = 0; row_idx < 32; row_idx++) {
                    nk_size_t const key_begin = key_begins[row_block_idx][row_idx],
                                    key_end = key_ends[row_block_idx][row_idx];
                    nk_size_t const column_begin = key_begin > panel_start ? key_begin - panel_start : 0;
                    nk_size_t const column_end = key_end < panel_end ? key_end - panel_start : valid_cols;
                    if (key_end <= panel_start || key_begin >= panel_end || column_begin >= column_end) {
                        column_begins[row_idx] = column_ends[row_idx] = 0;
                        continue;
                    }
                    column_begins[row_idx] = column_begin, column_ends[row_idx] = column_end;
                    live_rows[row_idx / 16] |= (__mmask16)(1u << (row_idx % 16));
                }
                nk_attention_score_panel_e4m3_diamondamx_(
                    (nk_dots_i8_a16x64_sapphireamx_t const(*)[4])scratch->queries_tiles[row_block_idx], keys_head_tiles,
                    panel_start / 16, panel_length / 32, depth_blocks, scratch->scores_panel, panel_width);
                nk_attention_panel_rowmax_diamondamx_(scratch->scores_panel, panel_width, column_begins, column_ends,
                                                      panel_max);
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    __m512 const old_max_f32x16 = row_max2_f32x16[row_block_idx][row_tile_idx];
                    __m512 const new_max_f32x16 = _mm512_mask_max_ps(
                        old_max_f32x16, live_rows[row_tile_idx], old_max_f32x16, // rows without keys keep their max
                        _mm512_mul_ps(_mm512_load_ps(panel_max[row_tile_idx]), scale2_f32x16));
                    __m512 const corr_f32x16 = nk_exp2_f32x16_skylake_(_mm512_sub_ps(old_max_f32x16, new_max_f32x16));
                    row_max2_f32x16[row_block_idx][row_tile_idx] = new_max_f32x16;
                    _mm512_store_ps(corrections[row_tile_idx], corr_f32x16);
                    _mm512_store_ps(new_max_arr[row_tile_idx], new_max_f32x16);
                }
                nk_attention_exp_panel_e4m3_diamondamx_(scratch->scores_panel, scratch->weights_panel, panel_length,
                                                        column_begins, column_ends, panel_width, scale2,
                                                        (nk_f32_t const(*)[16])new_max_arr, panel_sums);
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++)
                    row_sum_f32x16[row_block_idx][row_tile_idx] = _mm512_fmadd_ps(
                        row_sum_f32x16[row_block_idx][row_tile_idx], _mm512_load_ps(corrections[row_tile_idx]),
                        _mm512_load_ps(panel_sums[row_tile_idx]));
                nk_attention_weighted_sum_panel_e4m3_diamondamx_(
                    scratch->weights_panel, panel_width, values_head_tiles, position_blocks_total, panel_start / 64,
                    panel_length / 64, depth_tiles, output_stride_floats, (nk_f32_t const(*)[16])corrections,
                    scratch->o_acc[row_block_idx]);
            }
        }

        for (nk_size_t row_block_idx = 0; row_block_idx < row_block_count; row_block_idx++)
            for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                nk_size_t const row_start = row_block_start + row_block_idx * 32 + row_tile_idx * 16;
                if (row_start >= row_count) break;
                nk_size_t const valid_rows = (row_count - row_start >= 16) ? 16 : row_count - row_start;
                NK_ALIGN64 nk_f32_t row_sums[16];
                _mm512_store_ps(row_sums, row_sum_f32x16[row_block_idx][row_tile_idx]);
                for (nk_size_t row_idx = 0; row_idx < valid_rows; row_idx++) {
                    __m512 const inv_sum_f32x16 = _mm512_set1_ps(row_sums[row_idx] > 0 ? 1.0f / row_sums[row_idx]
                                                                                       : 0.0f);
                    nk_f32_t const *accumulator_row =
                        &scratch->o_acc[row_block_idx][(row_tile_idx * 16 + row_idx) * output_stride_floats];
                    nk_f32_t *output_row = output + (query_first + row_start + row_idx) * output_stride_out +
                                           head_idx * depth;
                    nk_size_t channel_idx = 0;
                    for (; channel_idx < depth_full; channel_idx += 16)
                        _mm512_storeu_ps(output_row + channel_idx,
                                         _mm512_mul_ps(_mm512_load_ps(accumulator_row + channel_idx), inv_sum_f32x16));
                    if (channel_idx < depth)
                        _mm512_mask_storeu_ps(
                            output_row + channel_idx, depth_tail_mask,
                            _mm512_mul_ps(_mm512_load_ps(accumulator_row + channel_idx), inv_sum_f32x16));
                }
                for (nk_size_t row_idx = 0; row_idx < 16; row_idx++)
                    for (nk_size_t channel_idx = 0; channel_idx < output_stride_floats; channel_idx += 16)
                        _mm512_store_ps(
                            &scratch->o_acc[row_block_idx]
                                           [(row_tile_idx * 16 + row_idx) * output_stride_floats + channel_idx],
                            zero_f32x16);
            }
    }
}

/** E4M3 attention over [task_start, task_start + task_count), reading only the keys each row's
 *  range admits. */
NK_HELPER_INLINE void nk_attention_packed_e4m3_diamondamx_(                      //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_diamondamx_k_) {
        nk_attention_causal_packed_e4m3_serial(queries, key_value_packed, output, head_count, key_value_head_count,
                                               depth, query_offsets, query_stride_bytes, output_stride_bytes, scale,
                                               diagonal_offset, window, task_start, task_count);
        return;
    }
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->heads != key_value_head_count) return;
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *tile_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(tile_offsets + segment_count + 1);
    char const *tiles_base = (char const *)key_value_packed + sizeof(*header) +
                             nk_attention_pack_directory_size_(segment_count);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_;

    nk_size_t const task_end = nk_attention_task_end_(task_start, task_count, segment_count * head_count);
    if (task_start >= task_end) return;

    nk_amx_tile_configure_sapphireamx_();
    nk_attention_scratch_e4m3_diamondamx_t_ scratch;
    __m512 const zero_f32x16 = _mm512_setzero_ps();
    for (nk_size_t row_block_idx = 0; row_block_idx < 4; row_block_idx++)
        for (nk_size_t i = 0; i < 32 * nk_attention_max_depth_diamondamx_k_; i += 16)
            _mm512_store_ps(&scratch.o_acc[row_block_idx][i], zero_f32x16);

    for (nk_size_t task_idx = task_start; task_idx < task_end; task_idx++) {
        nk_size_t const segment = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment];
        nk_size_t const row_count = query_offsets[segment + 1] - query_offsets[segment];
        if (row_count == 0) continue;
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 64);
        nk_size_t const bytes_per_head = position_count_padded * depth_padded; // one byte per element
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        nk_i8_t const *keys_head_tiles = (nk_i8_t const *)(tiles_base + tile_offsets[segment] +
                                                           key_value_head_idx * bytes_per_head);
        nk_i8_t const *values_head_tiles = (nk_i8_t const *)(tiles_base + tile_offsets[segment] +
                                                             (key_value_head_count + key_value_head_idx) *
                                                                 bytes_per_head);
        nk_attention_task_e4m3_diamondamx_(queries, output, keys_head_tiles, values_head_tiles, head_idx, depth,
                                           position_count, position_count_padded, query_offsets[segment], row_count,
                                           query_stride_bytes, output_stride_bytes, scale2, diagonal_offset, window,
                                           &scratch);
    }
    _tile_release();
}

NK_API_COMPTIME void nk_attention_bidirectional_packed_e4m3_diamondamx(          //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t task_start, nk_size_t task_count) {
    nk_attention_packed_e4m3_diamondamx_(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                         query_offsets, query_stride_bytes, output_stride_bytes, scale, NK_I64_MAX / 2,
                                         NK_SIZE_MAX, task_start, task_count);
}

NK_API_COMPTIME void nk_attention_causal_packed_e4m3_diamondamx(                 //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_i64_t diagonal_offset, nk_size_t window, nk_size_t task_start, nk_size_t task_count) {
    nk_attention_packed_e4m3_diamondamx_(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                         query_offsets, query_stride_bytes, output_stride_bytes, scale, diagonal_offset,
                                         window, task_start, task_count);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_DIAMONDAMX
#endif // NK_TARGET_X8664_
#endif // NK_ATTENTION_DIAMONDAMX_H
