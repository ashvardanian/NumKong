/**
 *  @brief Ragged attention for Diamond Rapids AMX (FP8 tiles, tile-row drains) targets.
 *  @file include/numkong/attention/diamondamx.h
 *  @author Ash Vardanian
 *  @date July 7, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  Diamond Rapids AMX debuts ~2027. This backend keeps the Sapphire Rapids panel-flash
 *  structure (`attention/sapphireamx.h`): 2×2 register blocking, KV-reuse chunking, and the
 *  base-2 streaming softmax reused from the Skylake tier. It layers on three Diamond Rapids
 *  upgrades:
 *
 *  - Accumulator tiles drain straight into ZMMs with a tile-row move instead of a
 *    `_tile_stored` memory round-trip: `_tile_movrow` casts an FP32-accumulator row into a
 *    ZMM, `_tile_cvtrowd2ps` converts an INT32-accumulator row to FP32 in the same move.
 *  - E4M3 runs natively through `_tile_dphf8ps` on raw E4M3 tiles — quad-interleaved exactly
 *    like the I8 pipeline, so the packed KV blob is half the Sapphire Rapids BF16-widened
 *    size — with softmax probabilities quantized to E4M3.
 *  - I8 keeps the exact TDPBSSD/TDPBUSD pipeline with the new drains.
 *
 *  Hardware debuts ~2027: correctness is SDE-validated; performance claims await silicon.
 */
#ifndef NK_ATTENTION_DIAMONDAMX_H
#define NK_ATTENTION_DIAMONDAMX_H

#if NK_TARGET_X8664_
#if NK_TARGET_DIAMONDAMX

#include "numkong/attention/serial.h"  // shared packed-KV offsets, width-agnostic fallback
#include "numkong/attention/skylake.h" // `nk_attention_exp2_f32x16_skylake_`, reduces
#include "numkong/dots/sapphireamx.h"  // tile config, BF16/I8 tile structs, load_a, transposers

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                                                \
    __attribute__((target(                                                                                                                   \
        "avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx10.2-512,f16c,fma,bmi,bmi2,amx-tile,amx-bf16,amx-int8,amx-fp8,amx-avx512"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx10.2-512", "f16c", "fma", \
                   "bmi", "bmi2", "amx-tile", "amx-bf16", "amx-int8", "amx-fp8", "amx-avx512")
#endif

enum {
    /** KV panel width in positions; the F32 score row stays L1-resident. */
    nk_attention_panel_diamondamx_k_ = 512,
    /** Widest head this backend handles in registers; larger heads route to the serial tier. */
    nk_attention_max_depth_diamondamx_k_ = 256,
};

/** @brief Drains the four FP32 accumulator tiles (TDPBF16PS / TDPHF8PS) of one 2×2 register block to the
 *  score panel, one ZMM row per tile per iteration. Tiles 0/1 fill the top 16 rows, 2/3 the bottom 16;
 *  `_tile_movrow` takes the tile as a compile-time immediate, so all four are written out explicitly. */
NK_INTERNAL void nk_attention_f32_store_grid_diamondamx_(nk_f32_t *scores_panel, nk_size_t pair_idx,
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

/** @brief Drains the four INT32 accumulator tiles (TDPBSSD) of one 2×2 register block to the score panel as
 *  FP32, converting on drain via `_tile_cvtrowd2ps`; all four tile immediates are written out explicitly. */
NK_INTERNAL void nk_attention_i32_store_grid_diamondamx_(nk_f32_t *scores_panel, nk_size_t pair_idx,
                                                         nk_size_t panel_width) {
    for (unsigned row_idx = 0; row_idx < 16; row_idx++) {
        nk_f32_t *top_row = scores_panel + pair_idx * 32 + row_idx * panel_width;
        nk_f32_t *bottom_row = scores_panel + 16 * panel_width + pair_idx * 32 + row_idx * panel_width;
        _mm512_storeu_ps(top_row, _tile_cvtrowd2ps(0, row_idx));
        _mm512_storeu_ps(top_row + 16, _tile_cvtrowd2ps(1, row_idx));
        _mm512_storeu_ps(bottom_row, _tile_cvtrowd2ps(2, row_idx));
        _mm512_storeu_ps(bottom_row + 16, _tile_cvtrowd2ps(3, row_idx));
    }
}

/** @brief Fuses the four FP32 accumulator tiles of one 2×2 register block into `o_acc` row-wise as
 *  `o = o·correction + drained`. Tiles 0/1 use the top row-tile's corrections, 2/3 the bottom's;
 *  `_tile_movrow` needs a compile-time tile immediate, so all four are written out explicitly. */
NK_INTERNAL void nk_attention_f32_accumulate_grid_diamondamx_(nk_size_t channel_start, nk_f32_t *o_acc,
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

/** @brief Fuses the four INT32 accumulator tiles (TDPBUSD) of one 2×2 register block into `o_acc` row-wise,
 *  converting on drain via `_tile_cvtrowd2ps`; all four tile immediates are written out explicitly. */
NK_INTERNAL void nk_attention_i32_accumulate_grid_diamondamx_(nk_size_t channel_start, nk_f32_t *o_acc,
                                                              nk_size_t output_stride_floats,
                                                              nk_f32_t const (*corrections)[16]) {
    for (unsigned row_idx = 0; row_idx < 16; row_idx++) {
        nk_f32_t *top_row = o_acc + row_idx * output_stride_floats + channel_start;
        nk_f32_t *bottom_row = o_acc + (16 + row_idx) * output_stride_floats + channel_start;
        __m512 const top_correction = _mm512_set1_ps(corrections[0][row_idx]);
        __m512 const bottom_correction = _mm512_set1_ps(corrections[1][row_idx]);
        _mm512_store_ps(top_row,
                        _mm512_fmadd_ps(_mm512_load_ps(top_row), top_correction, _tile_cvtrowd2ps(0, row_idx)));
        _mm512_store_ps(top_row + 16,
                        _mm512_fmadd_ps(_mm512_load_ps(top_row + 16), top_correction, _tile_cvtrowd2ps(1, row_idx)));
        _mm512_store_ps(bottom_row,
                        _mm512_fmadd_ps(_mm512_load_ps(bottom_row), bottom_correction, _tile_cvtrowd2ps(2, row_idx)));
        _mm512_store_ps(bottom_row + 16, _mm512_fmadd_ps(_mm512_load_ps(bottom_row + 16), bottom_correction,
                                                         _tile_cvtrowd2ps(3, row_idx)));
    }
}

/** @brief Quantizes 16 F32 probabilities to E4M3: F32→F16 (VCVTPS2PHX) then F16→E4M3 (VCVT2PH2HF8).
 *  The second VCVT2PH2HF8 operand fills the result's low half, so the payload rides there. */
NK_INTERNAL __m128i nk_attention_quantize_e4m3x16_diamondamx_(__m512 weights_f32x16) {
    __m512h const weights_f16x32 = _mm512_castph256_ph512(_mm512_cvtxps_ph(weights_f32x16));
    return _mm512_castsi512_si128(_mm512_cvts_2ph_hf8(_mm512_setzero_ph(), weights_f16x32));
}

/** @brief Widens 16 E4M3 probabilities back to F32 (VCVTHF82PH + VCVTPH2PSX) for a consistent sum. */
NK_INTERNAL __m512 nk_attention_dequantize_e4m3x16_diamondamx_(__m128i weights_e4m3x16) {
    return _mm512_cvtxph_ps(_mm256_cvthf8_ph(weights_e4m3x16));
}

NK_PUBLIC nk_size_t nk_attention_packed_size_bf16_diamondamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_diamondamx_k_)
        return nk_attention_packed_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count);
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
    nk_size_t total_tile_bytes = 0;
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++) {
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(segment_lengths[segment_idx], 32);
        total_tile_bytes += 2 * key_value_head_count * position_count_padded * depth_padded * sizeof(nk_bf16_t);
    }
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + total_tile_bytes;
}

/** @brief E4M3 native and I8 share a raw 1-byte, 64-deep, quad-interleaved tile layout. */
NK_INTERNAL nk_size_t nk_attention_packed_size_quad_diamondamx_(nk_size_t key_value_head_count, nk_size_t depth,
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

NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3_diamondamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_diamondamx_k_)
        return nk_attention_packed_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_quad_diamondamx_(key_value_head_count, depth, segment_lengths, segment_count);
}

NK_PUBLIC nk_size_t nk_attention_packed_size_i8_diamondamx(nk_size_t key_value_head_count, nk_size_t depth,
                                                           nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_diamondamx_k_)
        return nk_attention_packed_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_quad_diamondamx_(key_value_head_count, depth, segment_lengths, segment_count);
}

/** @brief BF16 packing: K transposed pair-interleaved B-tiles, V dim-major pair-interleaved. */
NK_INTERNAL void nk_attention_pack_bf16_core_diamondamx_( //
    nk_bf16_t const *keys, nk_bf16_t const *values, nk_size_t key_value_head_count, nk_size_t depth,
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t first_task,
    nk_size_t task_count) {

    nk_size_t const tile_elements = 512;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
    nk_size_t const depth_tiles = depth_padded / 32;
    nk_size_t const channel_tiles = depth_padded / 16;

    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, 32, depth_padded * sizeof(nk_bf16_t));
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *tile_offsets_ro = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *tiles_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    __m256i const interleave_idx_u16x16 = _mm256_setr_epi16( //
        0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    __m256i const interleave_idx_hi_u16x16 = _mm256_setr_epi16( //
        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count, h = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 32);
        nk_size_t const bytes_per_head = position_count_padded * depth_padded * sizeof(nk_bf16_t);

        nk_bf16_t *keys_head_tiles = (nk_bf16_t *)(tiles_base + tile_offsets_ro[segment_idx] + h * bytes_per_head);
        for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / 16; position_tile_idx++) {
            nk_size_t const row_start = position_tile_idx * 16;
            nk_size_t const valid_rows = (row_start + 16 <= position_count) ? 16
                                         : (row_start < position_count)     ? position_count - row_start
                                                                            : 0;
            for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles; depth_tile_idx++) {
                nk_size_t const channel_start = depth_tile_idx * 32;
                nk_size_t const valid_columns = (channel_start + 32 <= depth) ? 32
                                                : (channel_start < depth)     ? depth - channel_start
                                                                              : 0;
                nk_dots_bf16_a16x32_sapphireamx_t source_tile;
                nk_dots_bf16_b32x16_sapphireamx_t transposed_tile;
                nk_dots_bf16_load_a_sapphireamx_(
                    &source_tile,
                    (nk_bf16_t const *)((char const *)keys + (position_first + row_start) * key_stride_bytes) +
                        h * depth + channel_start,
                    key_stride_bytes / sizeof(nk_bf16_t), valid_rows, valid_columns);
                nk_dots_pack_bf16_transposed_sapphireamx_(&source_tile, &transposed_tile);
                nk_bf16_t *tile_output = keys_head_tiles +
                                         (position_tile_idx * depth_tiles + depth_tile_idx) * tile_elements;
                for (nk_size_t i = 0; i < tile_elements * sizeof(nk_bf16_t); i += 64)
                    _mm512_storeu_si512((char *)tile_output + i, _mm512_load_si512((char const *)&transposed_tile + i));
            }
        }

        nk_bf16_t *values_head_tiles = (nk_bf16_t *)(tiles_base + tile_offsets_ro[segment_idx] +
                                                     (key_value_head_count + h) * bytes_per_head);
        for (nk_size_t channel_tile_idx = 0; channel_tile_idx < channel_tiles; channel_tile_idx++) {
            nk_size_t const channel_start = channel_tile_idx * 16;
            nk_size_t const valid_columns = (channel_start + 16 <= depth) ? 16
                                            : (channel_start < depth)     ? depth - channel_start
                                                                          : 0;
            __mmask16 const columns_mask = (__mmask16)((1u << valid_columns) - 1);
            for (nk_size_t position_block_idx = 0; position_block_idx < position_count_padded / 32;
                 position_block_idx++) {
                nk_bf16_t *tile_output = values_head_tiles +
                                         (channel_tile_idx * (position_count_padded / 32) + position_block_idx) *
                                             tile_elements;
                for (nk_size_t pair_idx = 0; pair_idx < 16; pair_idx++) {
                    nk_size_t const row_a = position_block_idx * 32 + pair_idx * 2, row_b = row_a + 1;
                    char const *row_a_ptr = (char const *)values + (position_first + row_a) * value_stride_bytes +
                                            (h * depth + channel_start) * sizeof(nk_bf16_t);
                    char const *row_b_ptr = (char const *)values + (position_first + row_b) * value_stride_bytes +
                                            (h * depth + channel_start) * sizeof(nk_bf16_t);
                    __m256i a_bf16x16 = row_a < position_count ? _mm256_maskz_loadu_epi16(columns_mask, row_a_ptr)
                                                               : _mm256_setzero_si256();
                    __m256i b_bf16x16 = row_b < position_count ? _mm256_maskz_loadu_epi16(columns_mask, row_b_ptr)
                                                               : _mm256_setzero_si256();
                    __m256i lo_bf16x16 = _mm256_permutex2var_epi16(a_bf16x16, interleave_idx_u16x16, b_bf16x16);
                    __m256i hi_bf16x16 = _mm256_permutex2var_epi16(a_bf16x16, interleave_idx_hi_u16x16, b_bf16x16);
                    _mm256_storeu_si256((__m256i *)(tile_output + pair_idx * 32), lo_bf16x16);
                    _mm256_storeu_si256((__m256i *)(tile_output + pair_idx * 32 + 16), hi_bf16x16);
                }
            }
        }
    }
    nk_compiler_barrier_sapphireamx_();
}

/** @brief Raw 1-byte packing shared by E4M3-native and I8: K transposed, V quad-interleaved. */
NK_INTERNAL void nk_attention_pack_quad_diamondamx_(                                             //
    nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t first_task,
    nk_size_t task_count) {

    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_size_t const depth_blocks = depth_padded / 64;
    nk_size_t const channel_tiles = depth_padded / 16;
    nk_size_t const tile_bytes = 1024;

    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, 64, depth_padded);
    nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
    nk_u64_t const *tile_offsets_ro = (nk_u64_t const *)((char *)key_value_packed + sizeof(*header));
    char *tiles_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    __m512i const quad_interleave_idx_u8x64 = _mm512_setr_epi32( //
        0x30201000, 0x31211101, 0x32221202, 0x33231303,          //
        0x34241404, 0x35251505, 0x36261606, 0x37271707,          //
        0x38281808, 0x39291909, 0x3A2A1A0A, 0x3B2B1B0B,          //
        0x3C2C1C0C, 0x3D2D1D0D, 0x3E2E1E0E, 0x3F2F1F0F);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
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
                                        _mm512_permutexvar_epi8(quad_interleave_idx_u8x64, quad_i8x64));
                }
            }
        }
    }
    nk_compiler_barrier_sapphireamx_();
}

NK_PUBLIC void nk_attention_pack_bf16_diamondamx(    //
    nk_bf16_t const *keys, nk_bf16_t const *values,  //
    nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t first_task,
    nk_size_t task_count) {
    if (depth > nk_attention_max_depth_diamondamx_k_) {
        nk_attention_pack_bf16_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_bf16_core_diamondamx_(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                            segment_count, key_stride_bytes, value_stride_bytes, key_value_packed,
                                            first_task, task_count);
}

NK_PUBLIC void nk_attention_pack_e4m3_diamondamx(    //
    nk_e4m3_t const *keys, nk_e4m3_t const *values,  //
    nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t first_task,
    nk_size_t task_count) {
    if (depth > nk_attention_max_depth_diamondamx_k_) {
        nk_attention_pack_e4m3_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_quad_diamondamx_((nk_i8_t const *)keys, (nk_i8_t const *)values, key_value_head_count, depth,
                                       segment_offsets, segment_lengths, segment_count, key_stride_bytes,
                                       value_stride_bytes, key_value_packed, first_task, task_count);
}

NK_PUBLIC void nk_attention_pack_i8_diamondamx(      //
    nk_i8_t const *keys, nk_i8_t const *values,      //
    nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, nk_size_t segment_count,
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, nk_size_t first_task,
    nk_size_t task_count) {
    if (depth > nk_attention_max_depth_diamondamx_k_) {
        nk_attention_pack_i8_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                    segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                    task_count);
        return;
    }
    nk_attention_pack_quad_diamondamx_(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                       segment_count, key_stride_bytes, value_stride_bytes, key_value_packed,
                                       first_task, task_count);
}

/** @brief Row maxima over the live columns of an FP32 score panel; padded columns excluded. */
NK_INTERNAL void nk_attention_panel_rowmax_diamondamx_(nk_f32_t const *scores_panel, nk_size_t panel_width,
                                                       nk_size_t valid_cols, nk_f32_t (*panel_max)[16]) {
    nk_size_t const full_cols = valid_cols & ~(nk_size_t)15;
    __mmask16 const tail_mask = (__mmask16)((1u << (valid_cols - full_cols)) - 1);
    for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++)
        for (nk_size_t row_idx = 0; row_idx < 16; row_idx++) {
            nk_f32_t const *scores_row = scores_panel + (row_tile_idx * 16 + row_idx) * panel_width;
            __m512 max_f32x16 = _mm512_set1_ps(NK_F32_MIN);
            nk_size_t position_idx = 0;
            for (; position_idx < full_cols; position_idx += 16)
                max_f32x16 = _mm512_max_ps(max_f32x16, _mm512_loadu_ps(scores_row + position_idx));
            if (position_idx < valid_cols)
                max_f32x16 = _mm512_max_ps(max_f32x16, _mm512_mask_mov_ps(_mm512_set1_ps(NK_F32_MIN), tail_mask,
                                                                          _mm512_loadu_ps(scores_row + position_idx)));
            panel_max[row_tile_idx][row_idx] = nk_reduce_max_f32x16_skylake_(max_f32x16);
        }
}

/** @brief BF16 Q×Kᵀ: 2×2 register blocking over 32-deep tiles, drained to FP32 via `_tile_movrow`. */
NK_INTERNAL void nk_attention_score_panel_bf16_diamondamx_(nk_dots_bf16_a16x32_sapphireamx_t const (*queries_tiles)[8],
                                                           nk_bf16_t const *keys_head_tiles, nk_size_t panel_first_tile,
                                                           nk_size_t panel_pairs, nk_size_t depth_tiles,
                                                           nk_f32_t *scores_panel, nk_size_t panel_width) {
    nk_size_t const keys_tile_stride = depth_tiles * 512;
    for (nk_size_t pair_idx = 0; pair_idx < panel_pairs; pair_idx++) {
        nk_bf16_t const *keys_tile0 = keys_head_tiles + (panel_first_tile + pair_idx * 2) * keys_tile_stride;
        nk_bf16_t const *keys_tile1 = keys_tile0 + keys_tile_stride;
        _tile_zero(0);
        _tile_zero(1);
        _tile_zero(2);
        _tile_zero(3);
        for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles; depth_tile_idx++) {
            _tile_loadd(4, queries_tiles[0][depth_tile_idx].data, 64);
            _tile_loadd(5, queries_tiles[1][depth_tile_idx].data, 64);
            _tile_loadd(6, keys_tile0 + depth_tile_idx * 512, 64);
            _tile_loadd(7, keys_tile1 + depth_tile_idx * 512, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);
        }
        nk_attention_f32_store_grid_diamondamx_(scores_panel, pair_idx, panel_width);
    }
}

/** @brief E4M3 native Q×Kᵀ: `_tile_dphf8ps` over 64-deep raw E4M3 tiles, drained to FP32. */
NK_INTERNAL void nk_attention_score_panel_e4m3_diamondamx_(nk_dots_i8_a16x64_sapphireamx_t const (*queries_tiles)[4],
                                                           nk_i8_t const *keys_head_tiles, nk_size_t panel_first_tile,
                                                           nk_size_t panel_pairs, nk_size_t depth_blocks,
                                                           nk_f32_t *scores_panel, nk_size_t panel_width) {
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

/** @brief I8 Q×Kᵀ: exact-integer TDPBSSD over 64-deep tiles, drained to FP32 via `_tile_cvtrowd2ps`. */
NK_INTERNAL void nk_attention_score_panel_i8_diamondamx_(nk_dots_i8_a16x64_sapphireamx_t const (*queries_tiles)[4],
                                                         nk_i8_t const *keys_head_tiles, nk_size_t panel_first_tile,
                                                         nk_size_t panel_pairs, nk_size_t depth_blocks,
                                                         nk_f32_t *scores_panel, nk_size_t panel_width) {
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
            _tile_dpbssd(0, 4, 6);
            _tile_dpbssd(1, 4, 7);
            _tile_dpbssd(2, 5, 6);
            _tile_dpbssd(3, 5, 7);
        }
        nk_attention_i32_store_grid_diamondamx_(scores_panel, pair_idx, panel_width);
    }
}

/** @brief Streaming base-2 softmax → BF16 weights (F32 exp summed), Sapphire's contract verbatim. */
NK_INTERNAL void nk_attention_exp_panel_bf16_diamondamx_(nk_f32_t const *scores_panel, nk_bf16_t *weights_panel,
                                                         nk_size_t panel_length, nk_size_t valid_cols,
                                                         nk_size_t panel_width, nk_f32_t scale2,
                                                         nk_f32_t const (*new_max)[16], nk_f32_t (*panel_sums)[16]) {
    __m512 const scale_f32x16 = _mm512_set1_ps(scale2);
    nk_size_t const full_cols = valid_cols & ~(nk_size_t)15;
    __mmask16 const tail_mask = (__mmask16)((1u << (valid_cols - full_cols)) - 1);
    for (nk_size_t row_idx = 0; row_idx < 32; row_idx++) {
        nk_f32_t const *scores_row = scores_panel + row_idx * panel_width;
        nk_bf16_t *weights_row = weights_panel + row_idx * panel_width;
        __m512 const max_f32x16 = _mm512_set1_ps(new_max[row_idx / 16][row_idx % 16]);
        __m512 sum_f32x16 = _mm512_setzero_ps();
        nk_size_t position_idx = 0;
        for (; position_idx < full_cols; position_idx += 16) {
            __m512 exp_f32x16 = nk_attention_exp2_f32x16_skylake_(
                _mm512_fmsub_ps(_mm512_loadu_ps(scores_row + position_idx), scale_f32x16, max_f32x16));
            sum_f32x16 = _mm512_add_ps(sum_f32x16, exp_f32x16);
            _mm256_store_si256((__m256i *)(weights_row + position_idx), (__m256i)_mm512_cvtneps_pbh(exp_f32x16));
        }
        if (position_idx < valid_cols) {
            __m512 exp_f32x16 = _mm512_maskz_mov_ps(
                tail_mask, nk_attention_exp2_f32x16_skylake_(
                               _mm512_fmsub_ps(_mm512_loadu_ps(scores_row + position_idx), scale_f32x16, max_f32x16)));
            sum_f32x16 = _mm512_add_ps(sum_f32x16, exp_f32x16);
            _mm256_store_si256((__m256i *)(weights_row + position_idx), (__m256i)_mm512_cvtneps_pbh(exp_f32x16));
            position_idx += 16;
        }
        for (; position_idx < panel_length; position_idx += 16)
            _mm256_store_si256((__m256i *)(weights_row + position_idx), _mm256_setzero_si256());
        panel_sums[row_idx / 16][row_idx % 16] = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    }
}

/**
 *  @brief Streaming base-2 softmax → E4M3 weights, stored as `e4m3(256 · 2^(s₂ − m₂))`.
 *  The ×256 amplitude (the U8 tier's 255, rounded to a power of two so it divides out
 *  exactly) keeps every live weight down to 2⁻¹⁴ of the row maximum in E4M3's @b normal
 *  range: TDPHF8PS treats subnormal inputs as zero, so unscaled sub-2⁻⁶ weights would
 *  vanish from the P×V numerator while the dequantized sum kept them — a shrink-toward-zero
 *  bias that grows with context length. The sum accumulates the dequantized scaled
 *  weights, so the 256 cancels in normalization.
 */
NK_INTERNAL void nk_attention_exp_panel_e4m3_diamondamx_(nk_f32_t const *scores_panel, nk_e4m3_t *weights_panel,
                                                         nk_size_t panel_length, nk_size_t valid_cols,
                                                         nk_size_t panel_width, nk_f32_t scale2,
                                                         nk_f32_t const (*new_max)[16], nk_f32_t (*panel_sums)[16]) {
    __m512 const scale_f32x16 = _mm512_set1_ps(scale2);
    __m512 const amplitude_f32x16 = _mm512_set1_ps(256.0f);
    nk_size_t const full_cols = valid_cols & ~(nk_size_t)15;
    __mmask16 const tail_mask = (__mmask16)((1u << (valid_cols - full_cols)) - 1);
    for (nk_size_t row_idx = 0; row_idx < 32; row_idx++) {
        nk_f32_t const *scores_row = scores_panel + row_idx * panel_width;
        nk_e4m3_t *weights_row = weights_panel + row_idx * panel_width;
        __m512 const max_f32x16 = _mm512_set1_ps(new_max[row_idx / 16][row_idx % 16]);
        __m512 sum_f32x16 = _mm512_setzero_ps();
        nk_size_t position_idx = 0;
        for (; position_idx < full_cols; position_idx += 16) {
            __m512 const exp_f32x16 = _mm512_mul_ps(
                nk_attention_exp2_f32x16_skylake_(
                    _mm512_fmsub_ps(_mm512_loadu_ps(scores_row + position_idx), scale_f32x16, max_f32x16)),
                amplitude_f32x16);
            __m128i const weight_e4m3x16 = nk_attention_quantize_e4m3x16_diamondamx_(exp_f32x16);
            sum_f32x16 = _mm512_add_ps(sum_f32x16, nk_attention_dequantize_e4m3x16_diamondamx_(weight_e4m3x16));
            _mm_store_si128((__m128i *)(weights_row + position_idx), weight_e4m3x16);
        }
        if (position_idx < valid_cols) {
            __m512 const exp_f32x16 = _mm512_maskz_mov_ps(
                tail_mask, _mm512_mul_ps(nk_attention_exp2_f32x16_skylake_(_mm512_fmsub_ps(
                                             _mm512_loadu_ps(scores_row + position_idx), scale_f32x16, max_f32x16)),
                                         amplitude_f32x16));
            __m128i const weight_e4m3x16 = nk_attention_quantize_e4m3x16_diamondamx_(exp_f32x16);
            sum_f32x16 = _mm512_add_ps(sum_f32x16, nk_attention_dequantize_e4m3x16_diamondamx_(weight_e4m3x16));
            _mm_store_si128((__m128i *)(weights_row + position_idx), weight_e4m3x16);
            position_idx += 16;
        }
        for (; position_idx < panel_length; position_idx += 16)
            _mm_store_si128((__m128i *)(weights_row + position_idx), _mm_setzero_si128());
        panel_sums[row_idx / 16][row_idx % 16] = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    }
}

/** @brief Streaming base-2 softmax → U8 weights, `round(255·2^(s₂−m₂))`; the max position lands on 255. */
NK_INTERNAL void nk_attention_exp_panel_i8_diamondamx_(nk_f32_t const *scores_panel, nk_u8_t *weights_panel,
                                                       nk_size_t panel_length, nk_size_t valid_cols,
                                                       nk_size_t panel_width, nk_f32_t scale2,
                                                       nk_f32_t const (*new_max)[16], nk_f32_t (*panel_sums)[16]) {
    __m512 const scale_f32x16 = _mm512_set1_ps(scale2);
    __m512 const half_f32x16 = _mm512_set1_ps(0.5f);
    __m512 const amplitude_f32x16 = _mm512_set1_ps(255.0f);
    nk_size_t const full_cols = valid_cols & ~(nk_size_t)15;
    __mmask16 const tail_mask = (__mmask16)((1u << (valid_cols - full_cols)) - 1);
    for (nk_size_t row_idx = 0; row_idx < 32; row_idx++) {
        nk_f32_t const *scores_row = scores_panel + row_idx * panel_width;
        nk_u8_t *weights_row = weights_panel + row_idx * panel_width;
        __m512 const max_f32x16 = _mm512_set1_ps(new_max[row_idx / 16][row_idx % 16]);
        __m512 sum_f32x16 = _mm512_setzero_ps();
        nk_size_t position_idx = 0;
        for (; position_idx < full_cols; position_idx += 16) {
            __m512 const exp_f32x16 = nk_attention_exp2_f32x16_skylake_(
                _mm512_fmsub_ps(_mm512_loadu_ps(scores_row + position_idx), scale_f32x16, max_f32x16));
            __m512i const weight_u32x16 = _mm512_cvttps_epu32(
                _mm512_add_ps(_mm512_mul_ps(exp_f32x16, amplitude_f32x16), half_f32x16));
            sum_f32x16 = _mm512_add_ps(sum_f32x16, _mm512_cvtepu32_ps(weight_u32x16));
            _mm_store_si128((__m128i *)(weights_row + position_idx), _mm512_cvtusepi32_epi8(weight_u32x16));
        }
        if (position_idx < valid_cols) {
            __m512 const exp_f32x16 = _mm512_maskz_mov_ps(
                tail_mask, nk_attention_exp2_f32x16_skylake_(
                               _mm512_fmsub_ps(_mm512_loadu_ps(scores_row + position_idx), scale_f32x16, max_f32x16)));
            __m512i const weight_u32x16 = _mm512_cvttps_epu32(
                _mm512_add_ps(_mm512_mul_ps(exp_f32x16, amplitude_f32x16), half_f32x16));
            sum_f32x16 = _mm512_add_ps(sum_f32x16, _mm512_cvtepu32_ps(weight_u32x16));
            _mm_store_si128((__m128i *)(weights_row + position_idx), _mm512_cvtusepi32_epi8(weight_u32x16));
            position_idx += 16;
        }
        for (; position_idx < panel_length; position_idx += 16)
            _mm_store_si128((__m128i *)(weights_row + position_idx), _mm_setzero_si128());
        panel_sums[row_idx / 16][row_idx % 16] = nk_reduce_add_f32x16_skylake_(sum_f32x16);
    }
}

/** @brief BF16 P×V: TDPBF16PS over 32-deep steps, each accumulator fused into `o_acc` on drain. */
NK_INTERNAL void nk_attention_weighted_sum_panel_bf16_diamondamx_(nk_bf16_t const *weights_panel, nk_size_t panel_width,
                                                                  nk_bf16_t const *values_head_tiles,
                                                                  nk_size_t position_blocks_total,
                                                                  nk_size_t panel_first_block, nk_size_t panel_blocks,
                                                                  nk_size_t depth_tiles, nk_size_t output_stride_floats,
                                                                  nk_f32_t const (*corrections)[16], nk_f32_t *o_acc) {
    int const weights_stride_bytes = (int)(panel_width * sizeof(nk_bf16_t));
    for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles; depth_tile_idx++) {
        nk_bf16_t const *values_tile0 = values_head_tiles +
                                        ((depth_tile_idx * 2 + 0) * position_blocks_total + panel_first_block) * 512;
        nk_bf16_t const *values_tile1 = values_head_tiles +
                                        ((depth_tile_idx * 2 + 1) * position_blocks_total + panel_first_block) * 512;
        _tile_zero(0);
        _tile_zero(1);
        _tile_zero(2);
        _tile_zero(3);
        for (nk_size_t position_block_idx = 0; position_block_idx < panel_blocks; position_block_idx++) {
            _tile_loadd(4, weights_panel + position_block_idx * 32, weights_stride_bytes);
            _tile_loadd(5, weights_panel + 16 * panel_width + position_block_idx * 32, weights_stride_bytes);
            _tile_loadd(6, values_tile0 + position_block_idx * 512, 64);
            _tile_loadd(7, values_tile1 + position_block_idx * 512, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);
        }
        nk_attention_f32_accumulate_grid_diamondamx_(depth_tile_idx * 32, o_acc, output_stride_floats, corrections);
    }
}

/** @brief E4M3 native P×V: `_tile_dphf8ps` (E4M3 weights × E4M3 values → FP32), fused on drain. */
NK_INTERNAL void nk_attention_weighted_sum_panel_e4m3_diamondamx_(nk_e4m3_t const *weights_panel, nk_size_t panel_width,
                                                                  nk_i8_t const *values_head_tiles,
                                                                  nk_size_t position_blocks_total,
                                                                  nk_size_t panel_first_block, nk_size_t panel_blocks,
                                                                  nk_size_t depth_tiles, nk_size_t output_stride_floats,
                                                                  nk_f32_t const (*corrections)[16], nk_f32_t *o_acc) {
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

/** @brief I8 P×V: TDPBUSD (U8 weights × I8 values → I32), converted to FP32 on drain. */
NK_INTERNAL void nk_attention_weighted_sum_panel_i8_diamondamx_(nk_u8_t const *weights_panel, nk_size_t panel_width,
                                                                nk_i8_t const *values_head_tiles,
                                                                nk_size_t position_blocks_total,
                                                                nk_size_t panel_first_block, nk_size_t panel_blocks,
                                                                nk_size_t depth_tiles, nk_size_t output_stride_floats,
                                                                nk_f32_t const (*corrections)[16], nk_f32_t *o_acc) {
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
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(1, 4, 7);
            _tile_dpbusd(2, 5, 6);
            _tile_dpbusd(3, 5, 7);
        }
        nk_attention_i32_accumulate_grid_diamondamx_(depth_tile_idx * 32, o_acc, output_stride_floats, corrections);
    }
}

/** @brief BF16 per-call scratch: FP32 score panel, BF16 weight panel, output accumulators, Q tiles. */
typedef struct {
    NK_ALIGN64 nk_f32_t scores_panel[32 * nk_attention_panel_diamondamx_k_];
    NK_ALIGN64 nk_bf16_t weights_panel[32 * nk_attention_panel_diamondamx_k_];
    NK_ALIGN64 nk_f32_t o_acc[4][32 * nk_attention_max_depth_diamondamx_k_];
    nk_dots_bf16_a16x32_sapphireamx_t queries_tiles[4][2][8];
} nk_attention_scratch_bf16_diamondamx_t_;

/** @brief E4M3 native per-call scratch: FP32 scores, E4M3 weights, output accumulators, raw Q tiles. */
typedef struct {
    NK_ALIGN64 nk_f32_t scores_panel[32 * nk_attention_panel_diamondamx_k_];
    NK_ALIGN64 nk_e4m3_t weights_panel[32 * nk_attention_panel_diamondamx_k_];
    NK_ALIGN64 nk_f32_t o_acc[4][32 * nk_attention_max_depth_diamondamx_k_];
    nk_dots_i8_a16x64_sapphireamx_t queries_tiles[4][2][4];
} nk_attention_scratch_e4m3_diamondamx_t_;

/** @brief I8 per-call scratch: FP32 scores, U8 weights, output accumulators, raw Q tiles. */
typedef struct {
    NK_ALIGN64 nk_f32_t scores_panel[32 * nk_attention_panel_diamondamx_k_];
    NK_ALIGN64 nk_u8_t weights_panel[32 * nk_attention_panel_diamondamx_k_];
    NK_ALIGN64 nk_f32_t o_acc[4][32 * nk_attention_max_depth_diamondamx_k_];
    nk_dots_i8_a16x64_sapphireamx_t queries_tiles[4][2][4];
} nk_attention_scratch_i8_diamondamx_t_;

/** @brief BF16 (segment, head) task: panel-flash with KV-reuse chunking and tile-row drains. */
NK_INTERNAL void nk_attention_task_bf16_diamondamx_(
    nk_bf16_t const *queries, nk_f32_t *output, nk_bf16_t const *keys_head_tiles, nk_bf16_t const *values_head_tiles,
    nk_size_t head_idx, nk_size_t depth, nk_size_t position_count, nk_size_t position_count_padded,
    nk_size_t query_first, nk_size_t row_count, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes,
    nk_f32_t scale2, nk_attention_scratch_bf16_diamondamx_t_ *scratch) {
    nk_size_t const panel_width = nk_attention_panel_diamondamx_k_;
    nk_size_t const row_blocks = 4;
    nk_size_t const kv_padded = nk_size_round_up_to_multiple_(position_count, 32);
    nk_size_t const position_blocks_total = position_count_padded / 32;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
    nk_size_t const depth_tiles = depth_padded / 32;
    nk_size_t const output_stride_floats = depth_padded;
    nk_size_t const output_stride_out = output_stride_bytes / sizeof(nk_f32_t);

    NK_ALIGN64 nk_f32_t panel_max[2][16], corrections[2][16], new_max_arr[2][16], panel_sums[2][16];
    __m512 row_max2_f32x16[4][2], row_sum_f32x16[4][2];
    __m512 const zero_f32x16 = _mm512_setzero_ps();
    __m512 const scale2_f32x16 = _mm512_set1_ps(scale2);
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const depth_tail_mask = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t row_block_start = 0; row_block_start < row_count; row_block_start += 32 * row_blocks) {
        nk_size_t const row_block_count = ((row_count - row_block_start + 31) / 32 < row_blocks)
                                              ? (row_count - row_block_start + 31) / 32
                                              : row_blocks;
        for (nk_size_t row_block_idx = 0; row_block_idx < row_block_count; row_block_idx++)
            for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                nk_size_t const row_start = row_block_start + row_block_idx * 32 + row_tile_idx * 16;
                nk_size_t const valid_rows = row_start >= row_count          ? 0
                                             : (row_count - row_start >= 16) ? 16
                                                                             : row_count - row_start;
                for (nk_size_t depth_tile_idx = 0; depth_tile_idx < depth_tiles; depth_tile_idx++) {
                    nk_size_t const channel_start = depth_tile_idx * 32;
                    nk_size_t const valid_columns = (channel_start + 32 <= depth) ? 32 : depth - channel_start;
                    nk_dots_bf16_load_a_sapphireamx_(
                        &scratch->queries_tiles[row_block_idx][row_tile_idx][depth_tile_idx],
                        (nk_bf16_t const *)((char const *)queries + (query_first + row_start) * query_stride_bytes) +
                            head_idx * depth + channel_start,
                        query_stride_bytes / sizeof(nk_bf16_t), valid_rows, valid_columns);
                }
                row_max2_f32x16[row_block_idx][row_tile_idx] = _mm512_set1_ps(NK_F32_MIN);
                row_sum_f32x16[row_block_idx][row_tile_idx] = _mm512_setzero_ps();
            }

        for (nk_size_t panel_start = 0; panel_start < kv_padded; panel_start += panel_width) {
            nk_size_t const panel_length = (panel_start + panel_width <= kv_padded) ? panel_width
                                                                                    : (kv_padded - panel_start);
            nk_size_t const valid_cols = (panel_start + panel_length <= position_count)
                                             ? panel_length
                                             : (position_count - panel_start);
            for (nk_size_t row_block_idx = 0; row_block_idx < row_block_count; row_block_idx++) {
                nk_attention_score_panel_bf16_diamondamx_(
                    (nk_dots_bf16_a16x32_sapphireamx_t const(*)[8])scratch->queries_tiles[row_block_idx],
                    keys_head_tiles, panel_start / 16, panel_length / 32, depth_tiles, scratch->scores_panel,
                    panel_width);
                nk_attention_panel_rowmax_diamondamx_(scratch->scores_panel, panel_width, valid_cols, panel_max);
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    __m512 const old_max_f32x16 = row_max2_f32x16[row_block_idx][row_tile_idx];
                    __m512 const new_max_f32x16 = _mm512_max_ps(
                        old_max_f32x16, _mm512_mul_ps(_mm512_load_ps(panel_max[row_tile_idx]), scale2_f32x16));
                    __m512 const corr_f32x16 = nk_attention_exp2_f32x16_skylake_(
                        _mm512_sub_ps(old_max_f32x16, new_max_f32x16));
                    row_max2_f32x16[row_block_idx][row_tile_idx] = new_max_f32x16;
                    _mm512_store_ps(corrections[row_tile_idx], corr_f32x16);
                    _mm512_store_ps(new_max_arr[row_tile_idx], new_max_f32x16);
                }
                nk_attention_exp_panel_bf16_diamondamx_(scratch->scores_panel, scratch->weights_panel, panel_length,
                                                        valid_cols, panel_width, scale2,
                                                        (nk_f32_t const(*)[16])new_max_arr, panel_sums);
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++)
                    row_sum_f32x16[row_block_idx][row_tile_idx] = _mm512_fmadd_ps(
                        row_sum_f32x16[row_block_idx][row_tile_idx], _mm512_load_ps(corrections[row_tile_idx]),
                        _mm512_load_ps(panel_sums[row_tile_idx]));
                nk_attention_weighted_sum_panel_bf16_diamondamx_(
                    scratch->weights_panel, panel_width, values_head_tiles, position_blocks_total, panel_start / 32,
                    panel_length / 32, depth_tiles, output_stride_floats, (nk_f32_t const(*)[16])corrections,
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
                    __m512 const inv_sum_f32x16 = _mm512_set1_ps(1.0f / row_sums[row_idx]);
                    nk_f32_t const *acc_row =
                        &scratch->o_acc[row_block_idx][(row_tile_idx * 16 + row_idx) * output_stride_floats];
                    nk_f32_t *output_row = output + (query_first + row_start + row_idx) * output_stride_out +
                                           head_idx * depth;
                    nk_size_t channel_idx = 0;
                    for (; channel_idx < depth_full; channel_idx += 16)
                        _mm512_storeu_ps(output_row + channel_idx,
                                         _mm512_mul_ps(_mm512_load_ps(acc_row + channel_idx), inv_sum_f32x16));
                    if (channel_idx < depth)
                        _mm512_mask_storeu_ps(output_row + channel_idx, depth_tail_mask,
                                              _mm512_mul_ps(_mm512_load_ps(acc_row + channel_idx), inv_sum_f32x16));
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

/** @brief E4M3 native (segment, head) task: `_tile_dphf8ps` scores and P×V with E4M3-quantized weights. */
NK_INTERNAL void nk_attention_task_e4m3_diamondamx_(nk_e4m3_t const *queries, nk_f32_t *output,
                                                    nk_i8_t const *keys_head_tiles, nk_i8_t const *values_head_tiles,
                                                    nk_size_t head_idx, nk_size_t depth, nk_size_t position_count,
                                                    nk_size_t position_count_padded, nk_size_t query_first,
                                                    nk_size_t row_count, nk_size_t query_stride_bytes,
                                                    nk_size_t output_stride_bytes, nk_f32_t scale2,
                                                    nk_attention_scratch_e4m3_diamondamx_t_ *scratch) {
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
    __m512 const zero_f32x16 = _mm512_setzero_ps();
    __m512 const scale2_f32x16 = _mm512_set1_ps(scale2);
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const depth_tail_mask = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t row_block_start = 0; row_block_start < row_count; row_block_start += 32 * row_blocks) {
        nk_size_t const row_block_count = ((row_count - row_block_start + 31) / 32 < row_blocks)
                                              ? (row_count - row_block_start + 31) / 32
                                              : row_blocks;
        for (nk_size_t row_block_idx = 0; row_block_idx < row_block_count; row_block_idx++)
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

        for (nk_size_t panel_start = 0; panel_start < kv_padded; panel_start += panel_width) {
            nk_size_t const panel_length = (panel_start + panel_width <= kv_padded) ? panel_width
                                                                                    : (kv_padded - panel_start);
            nk_size_t const valid_cols = (panel_start + panel_length <= position_count)
                                             ? panel_length
                                             : (position_count - panel_start);
            for (nk_size_t row_block_idx = 0; row_block_idx < row_block_count; row_block_idx++) {
                nk_attention_score_panel_e4m3_diamondamx_(
                    (nk_dots_i8_a16x64_sapphireamx_t const(*)[4])scratch->queries_tiles[row_block_idx], keys_head_tiles,
                    panel_start / 16, panel_length / 32, depth_blocks, scratch->scores_panel, panel_width);
                nk_attention_panel_rowmax_diamondamx_(scratch->scores_panel, panel_width, valid_cols, panel_max);
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    __m512 const old_max_f32x16 = row_max2_f32x16[row_block_idx][row_tile_idx];
                    __m512 const new_max_f32x16 = _mm512_max_ps(
                        old_max_f32x16, _mm512_mul_ps(_mm512_load_ps(panel_max[row_tile_idx]), scale2_f32x16));
                    __m512 const corr_f32x16 = nk_attention_exp2_f32x16_skylake_(
                        _mm512_sub_ps(old_max_f32x16, new_max_f32x16));
                    row_max2_f32x16[row_block_idx][row_tile_idx] = new_max_f32x16;
                    _mm512_store_ps(corrections[row_tile_idx], corr_f32x16);
                    _mm512_store_ps(new_max_arr[row_tile_idx], new_max_f32x16);
                }
                nk_attention_exp_panel_e4m3_diamondamx_(scratch->scores_panel, scratch->weights_panel, panel_length,
                                                        valid_cols, panel_width, scale2,
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
                    __m512 const inv_sum_f32x16 = _mm512_set1_ps(1.0f / row_sums[row_idx]);
                    nk_f32_t const *acc_row =
                        &scratch->o_acc[row_block_idx][(row_tile_idx * 16 + row_idx) * output_stride_floats];
                    nk_f32_t *output_row = output + (query_first + row_start + row_idx) * output_stride_out +
                                           head_idx * depth;
                    nk_size_t channel_idx = 0;
                    for (; channel_idx < depth_full; channel_idx += 16)
                        _mm512_storeu_ps(output_row + channel_idx,
                                         _mm512_mul_ps(_mm512_load_ps(acc_row + channel_idx), inv_sum_f32x16));
                    if (channel_idx < depth)
                        _mm512_mask_storeu_ps(output_row + channel_idx, depth_tail_mask,
                                              _mm512_mul_ps(_mm512_load_ps(acc_row + channel_idx), inv_sum_f32x16));
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

/** @brief I8 (segment, head) task: exact TDPBSSD scores and U8×I8 P×V, bit-identical to the serial tier. */
NK_INTERNAL void nk_attention_task_i8_diamondamx_(nk_i8_t const *queries, nk_f32_t *output,
                                                  nk_i8_t const *keys_head_tiles, nk_i8_t const *values_head_tiles,
                                                  nk_size_t head_idx, nk_size_t depth, nk_size_t position_count,
                                                  nk_size_t position_count_padded, nk_size_t query_first,
                                                  nk_size_t row_count, nk_size_t query_stride_bytes,
                                                  nk_size_t output_stride_bytes, nk_f32_t scale2,
                                                  nk_attention_scratch_i8_diamondamx_t_ *scratch) {
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
    __m512 const zero_f32x16 = _mm512_setzero_ps();
    __m512 const scale2_f32x16 = _mm512_set1_ps(scale2);
    nk_size_t const depth_full = depth & ~(nk_size_t)15;
    __mmask16 const depth_tail_mask = (__mmask16)((1u << (depth - depth_full)) - 1);

    for (nk_size_t row_block_start = 0; row_block_start < row_count; row_block_start += 32 * row_blocks) {
        nk_size_t const row_block_count = ((row_count - row_block_start + 31) / 32 < row_blocks)
                                              ? (row_count - row_block_start + 31) / 32
                                              : row_blocks;
        for (nk_size_t row_block_idx = 0; row_block_idx < row_block_count; row_block_idx++)
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

        for (nk_size_t panel_start = 0; panel_start < kv_padded; panel_start += panel_width) {
            nk_size_t const panel_length = (panel_start + panel_width <= kv_padded) ? panel_width
                                                                                    : (kv_padded - panel_start);
            nk_size_t const valid_cols = (panel_start + panel_length <= position_count)
                                             ? panel_length
                                             : (position_count - panel_start);
            for (nk_size_t row_block_idx = 0; row_block_idx < row_block_count; row_block_idx++) {
                nk_attention_score_panel_i8_diamondamx_(
                    (nk_dots_i8_a16x64_sapphireamx_t const(*)[4])scratch->queries_tiles[row_block_idx], keys_head_tiles,
                    panel_start / 16, panel_length / 32, depth_blocks, scratch->scores_panel, panel_width);
                nk_attention_panel_rowmax_diamondamx_(scratch->scores_panel, panel_width, valid_cols, panel_max);
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    __m512 const old_max_f32x16 = row_max2_f32x16[row_block_idx][row_tile_idx];
                    __m512 const new_max_f32x16 = _mm512_max_ps(
                        old_max_f32x16, _mm512_mul_ps(_mm512_load_ps(panel_max[row_tile_idx]), scale2_f32x16));
                    __m512 const corr_f32x16 = nk_attention_exp2_f32x16_skylake_(
                        _mm512_sub_ps(old_max_f32x16, new_max_f32x16));
                    row_max2_f32x16[row_block_idx][row_tile_idx] = new_max_f32x16;
                    _mm512_store_ps(corrections[row_tile_idx], corr_f32x16);
                    _mm512_store_ps(new_max_arr[row_tile_idx], new_max_f32x16);
                }
                nk_attention_exp_panel_i8_diamondamx_(scratch->scores_panel, scratch->weights_panel, panel_length,
                                                      valid_cols, panel_width, scale2,
                                                      (nk_f32_t const(*)[16])new_max_arr, panel_sums);
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++)
                    row_sum_f32x16[row_block_idx][row_tile_idx] = _mm512_fmadd_ps(
                        row_sum_f32x16[row_block_idx][row_tile_idx], _mm512_load_ps(corrections[row_tile_idx]),
                        _mm512_load_ps(panel_sums[row_tile_idx]));
                nk_attention_weighted_sum_panel_i8_diamondamx_(
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
                    __m512 const inv_sum_f32x16 = _mm512_set1_ps(1.0f / row_sums[row_idx]);
                    nk_f32_t const *acc_row =
                        &scratch->o_acc[row_block_idx][(row_tile_idx * 16 + row_idx) * output_stride_floats];
                    nk_f32_t *output_row = output + (query_first + row_start + row_idx) * output_stride_out +
                                           head_idx * depth;
                    nk_size_t channel_idx = 0;
                    for (; channel_idx < depth_full; channel_idx += 16)
                        _mm512_storeu_ps(output_row + channel_idx,
                                         _mm512_mul_ps(_mm512_load_ps(acc_row + channel_idx), inv_sum_f32x16));
                    if (channel_idx < depth)
                        _mm512_mask_storeu_ps(output_row + channel_idx, depth_tail_mask,
                                              _mm512_mul_ps(_mm512_load_ps(acc_row + channel_idx), inv_sum_f32x16));
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

NK_PUBLIC void nk_attention_packed_bf16_diamondamx(                              //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_diamondamx_k_) {
        nk_attention_packed_bf16_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->key_value_head_count != key_value_head_count) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *tile_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(tile_offsets + segment_count + 1);
    char const *tiles_base = (char const *)key_value_packed + sizeof(*header) +
                             nk_attention_pack_directory_size_(segment_count);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 32);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    nk_amx_tile_configure_sapphireamx_();
    nk_attention_scratch_bf16_diamondamx_t_ scratch;
    __m512 const zero_f32x16 = _mm512_setzero_ps();
    for (nk_size_t row_block_idx = 0; row_block_idx < 4; row_block_idx++)
        for (nk_size_t i = 0; i < 32 * nk_attention_max_depth_diamondamx_k_; i += 16)
            _mm512_store_ps(&scratch.o_acc[row_block_idx][i], zero_f32x16);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment];
        nk_size_t const row_count = query_offsets[segment + 1] - query_offsets[segment];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 32);
        nk_size_t const bytes_per_head = position_count_padded * depth_padded * sizeof(nk_bf16_t);
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        nk_bf16_t const *keys_head_tiles = (nk_bf16_t const *)(tiles_base + tile_offsets[segment] +
                                                               key_value_head_idx * bytes_per_head);
        nk_bf16_t const *values_head_tiles = (nk_bf16_t const *)(tiles_base + tile_offsets[segment] +
                                                                 (key_value_head_count + key_value_head_idx) *
                                                                     bytes_per_head);
        nk_attention_task_bf16_diamondamx_(queries, output, keys_head_tiles, values_head_tiles, head_idx, depth,
                                           position_count, position_count_padded, query_offsets[segment], row_count,
                                           query_stride_bytes, output_stride_bytes, scale2, &scratch);
    }
    _tile_release();
}

NK_PUBLIC void nk_attention_packed_e4m3_diamondamx(                              //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_diamondamx_k_) {
        nk_attention_packed_e4m3_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->key_value_head_count != key_value_head_count) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *tile_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(tile_offsets + segment_count + 1);
    char const *tiles_base = (char const *)key_value_packed + sizeof(*header) +
                             nk_attention_pack_directory_size_(segment_count);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    nk_amx_tile_configure_sapphireamx_();
    nk_attention_scratch_e4m3_diamondamx_t_ scratch;
    __m512 const zero_f32x16 = _mm512_setzero_ps();
    for (nk_size_t row_block_idx = 0; row_block_idx < 4; row_block_idx++)
        for (nk_size_t i = 0; i < 32 * nk_attention_max_depth_diamondamx_k_; i += 16)
            _mm512_store_ps(&scratch.o_acc[row_block_idx][i], zero_f32x16);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment];
        nk_size_t const row_count = query_offsets[segment + 1] - query_offsets[segment];
        if (position_count == 0 || row_count == 0) continue;
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
                                           query_stride_bytes, output_stride_bytes, scale2, &scratch);
    }
    _tile_release();
}

NK_PUBLIC void nk_attention_packed_i8_diamondamx(                                //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,      //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_diamondamx_k_) {
        nk_attention_packed_i8_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                      query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                      task_count);
        return;
    }
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->key_value_head_count != key_value_head_count) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *tile_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(tile_offsets + segment_count + 1);
    char const *tiles_base = (char const *)key_value_packed + sizeof(*header) +
                             nk_attention_pack_directory_size_(segment_count);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, 64);
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_;

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    nk_amx_tile_configure_sapphireamx_();
    nk_attention_scratch_i8_diamondamx_t_ scratch;
    __m512 const zero_f32x16 = _mm512_setzero_ps();
    for (nk_size_t row_block_idx = 0; row_block_idx < 4; row_block_idx++)
        for (nk_size_t i = 0; i < 32 * nk_attention_max_depth_diamondamx_k_; i += 16)
            _mm512_store_ps(&scratch.o_acc[row_block_idx][i], zero_f32x16);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment];
        nk_size_t const row_count = query_offsets[segment + 1] - query_offsets[segment];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 64);
        nk_size_t const bytes_per_head = position_count_padded * depth_padded;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        nk_i8_t const *keys_head_tiles = (nk_i8_t const *)(tiles_base + tile_offsets[segment] +
                                                           key_value_head_idx * bytes_per_head);
        nk_i8_t const *values_head_tiles = (nk_i8_t const *)(tiles_base + tile_offsets[segment] +
                                                             (key_value_head_count + key_value_head_idx) *
                                                                 bytes_per_head);
        nk_attention_task_i8_diamondamx_(queries, output, keys_head_tiles, values_head_tiles, head_idx, depth,
                                         position_count, position_count_padded, query_offsets[segment], row_count,
                                         query_stride_bytes, output_stride_bytes, scale2, &scratch);
    }
    _tile_release();
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
