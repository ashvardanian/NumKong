/**
 *  @file include/numkong/attention/sme.h
 *  @author Ash Vardanian
 *  @date July 8, 2026
 *  @brief Arm SME ragged attention backend.
 *
 *  @sa include/numkong/attention.h
 *
 *  FlashAttention-style panel sweep on the SME outer-product engine, mirroring the @c sapphireamx
 *  skeleton with the shared packed header, segment directory, base-2 streaming softmax, and
 *  [tasks_begin, tasks_end) windows — but restructured around three Arm-specific properties
 *  measured on Apple M5.
 *
 *  Streaming mode is entered once per public call and never left: matrix work runs as widening MOPA
 *  outer products into ZA32 tiles, and the softmax stays on the streaming SVE vector unit, so there
 *  are no SMSTART/SMSTOP round-trips and no NEON excursions inside the hot loop.
 *
 *  ZA slices move in both directions: vertical reads give a free transpose, so the score panel is
 *  stored position-major with one query per lane — the running maximum, corrections, weight sums,
 *  output rescaling, and normalization are all plain lane-parallel vector ops with no horizontal
 *  reductions or per-row scalar broadcasts anywhere in the pipeline.
 *
 *  Probabilities cross from F32 scores to MOPA-ready pair-interleaved BF16 operands in registers,
 *  round + @c TRN2, replacing the AMX tile-loader's mandatory memory round-trip with a single
 *  in-register shuffle per position pair.
 *
 *  ZA tile roles per stage (SVL = 512: four 16×16 F32 tiles):
 *
 *  - Q staging & output transpose: ZA0 horizontal-write / vertical-read (pair interleave)
 *  - Q × Kᵀ scores: ZA0-ZA3 = (two query row-tiles) × (two K position-tiles)
 *  - P × V: ZA0-ZA3 = (two probability row-tiles) × (two V channel-tiles)
 *
 *  Widening MOPA keeps every reduction in F32 accumulators: the non-widening ZA16 forms measure ~2×
 *  faster but lose ~12% relative accuracy on signed depth-256 reductions, which fails the family's
 *  F32-accumulator contract.
 *
 *  F16 and NVFP4 run the same sweep on widening F16 MOPAs with F16 weights. NVFP4 elements times
 *  their UE4M3 block scales are exact F16 tiles; both tensor scales fold into the score multiplier
 *  and the value one into the normalization.
 *
 *  The MX formats run the BF16 sweep over rebased planes, exact in BF16, with the plane-exponent
 *  table after them. Each query block rebases by its own largest block exponent. A block whose
 *  planes are raw, whose exponents span past the window, or whose score multiplier is not a normal
 *  F32 is skipped in streaming mode and recomputed after it by the serial exact helpers.
 */
#ifndef NUMKONG_ATTENTION_SME_H
#define NUMKONG_ATTENTION_SME_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_SME_

#include <arm_sme.h>

#include "numkong/types.h"
#include "numkong/attention/serial.h" // `nk_attention_packed_header_t`, `nk_attention_pack_directory_serial_`
#include "numkong/dots/sme.h"         // `nk_sme_zero_za32_tile_0_k` and the ZA transpose pack idiom
#include "numkong/each/sme.h"         // `nk_exp2_f32x_sme_`, `nk_exp2_polynomial_f16x_sme_`, `nk_exp2_u8_i32x_sme_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

enum {

    /** KV panel width in positions; the position-major F32 score panel (64
     *  KB) stays L2-resident. */
    nk_attention_panel_sme_k_ = 512,

    /** Widest head this backend handles in tiles; larger heads route to the serial kernel. */
    nk_attention_max_depth_sme_k_ = 256,

    /** Widest ZA32 tile dimension the stack scratch is sized for (SVL ≤ 512); larger
     *  routes to serial. */
    nk_attention_max_tile_sme_k_ = 16,
};

/** Rounds two F32 weight vectors to BF16 and interleaves them pair-wise in one @c TRN2: lane `2i`
 *  gets `even[i]`, lane `2i + 1` gets `odd[i]` — the exact widening-BFMOPA operand layout, produced
 *  without any memory round-trip. */
NUMKONG_INLINE svuint16_t nk_attention_bf16_pair_sme_(svfloat32_t even_f32x, svfloat32_t odd_f32x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svuint32_t even_u32x = svreinterpret_u32_f32(even_f32x);
    svuint32_t odd_u32x = svreinterpret_u32_f32(odd_f32x);
    // Round half up: the weights are positive finite and already carry ~1e-3 polynomial
    // error, so the sub-ULP tie direction of round-to-nearest-even buys nothing here.
    even_u32x = svadd_n_u32_x(predicate_all_b32x, even_u32x, 0x8000);
    odd_u32x = svadd_n_u32_x(predicate_all_b32x, odd_u32x, 0x8000);
    return svtrn2_u16(svreinterpret_u16_u32(even_u32x), svreinterpret_u16_u32(odd_u32x));
}

/** The F16 twin of @c nk_attention_bf16_pair_sme_, rounding to nearest even. */
NUMKONG_INLINE svuint16_t nk_attention_f16_pair_sme_(svfloat32_t even_f32x, svfloat32_t odd_f32x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    return svreinterpret_u16_f16(
        svcvtnt_f16_f32_m(svcvt_f16_f32_x(predicate_all_b32x, even_f32x), predicate_all_b32x, odd_f32x));
}

/** Lanes whose visible key range `[key_begins, key_ends)` contains @p position. */
NUMKONG_INLINE svbool_t nk_attention_visible_sme_(svuint32_t key_begins_u32x, svuint32_t key_ends_u32x,
                                                  nk_size_t position) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    return svand_b_z(predicate_all_b32x, svcmple_n_u32(predicate_all_b32x, key_begins_u32x, (uint32_t)position),
                     svcmpgt_n_u32(predicate_all_b32x, key_ends_u32x, (uint32_t)position));
}

/** Interleaves the leading quarters of four rows of bytes into 32-bit quads, the SMOPA layout. */
NUMKONG_INLINE svuint8_t nk_interleave_quads_u8x_sme_(svuint8_t first_u8x, svuint8_t second_u8x, svuint8_t third_u8x,
                                                      svuint8_t fourth_u8x) NUMKONG_STREAMING_ {
    svuint16_t first_pairs_u16x = svreinterpret_u16_u8(svzip1_u8(first_u8x, second_u8x));
    svuint16_t second_pairs_u16x = svreinterpret_u16_u8(svzip1_u8(third_u8x, fourth_u8x));
    return svreinterpret_u8_u16(svzip1_u16(first_pairs_u16x, second_pairs_u16x));
}

/** Widens E4M3 bytes to their exact BF16 representations: every E4M3 value (3-bit mantissa, ±448
 *  range) is exactly representable in BF16, so the F16 hop through the @c dots converter and the
 *  final narrowing round are lossless. */
NUMKONG_INLINE svuint16_t nk_attention_e4m3_to_bf16_sme_(svbool_t predicate_b16x,
                                                         svuint8_t bytes_u8x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svfloat16_t const halves_f16x = nk_e4m3x_to_f16x_sme_streaming_(predicate_b16x, bytes_u8x);
    svfloat32_t const even_f32x = svcvt_f32_f16_x(predicate_all_b32x, halves_f16x);
    svfloat32_t const odd_f32x = svcvt_f32_f16_x(
        predicate_all_b32x,
        svreinterpret_f16_u32(svlsr_n_u32_x(predicate_all_b32x, svreinterpret_u32_f16(halves_f16x), 16)));
    return nk_attention_bf16_pair_sme_(even_f32x, odd_f32x);
}

/** Bytes of a pack of 16-bit planes, positions padded to one vector of them. */
NUMKONG_INLINE nk_size_t nk_attention_pack_size_b16_sme_(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, nk_cntw_sme_());
    return nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, nk_cnth_sme_(),
                                           depth_padded * sizeof(nk_u16_t));
}

/**
 *  @brief Streaming pack core for 16-bit K/V planes.
 *
 *  K becomes pair-interleaved MOPA operand vectors `[position_tile][depth_pair]` through the
 *  ZA0 horizontal-write / vertical-read transpose (the @c dots packer idiom); V becomes
 *  transposed pair-interleaved vectors `[channel_tile][position_pair]` through one @c ZIP1 per
 *  position pair, so P × V runs as outer products over positions. Rows beyond the segment and
 *  channels beyond the head are zero-filled by the predicated loads and the ZA0 pre-zeroing.
 */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_pack_b16_sme_streaming_(   //
    void const *keys, void const *values, nk_size_t element_bytes,             //
    nk_size_t key_value_head_count,                                            //
    nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride,     //
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2;
    int const widened = element_bytes != sizeof(nk_u16_t);

    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    svbool_t const predicate_all_b32x = svptrue_b32();

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count,
                vector_elements, depth_padded * sizeof(nk_u16_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
        nk_size_t const plane_bytes = position_count_padded * depth_padded * sizeof(nk_u16_t);
        nk_u16_t *keys_plane = (nk_u16_t *)(payload_base + payload_offset + key_value_head_idx * plane_bytes);
        nk_u16_t *values_plane = (nk_u16_t *)(payload_base + payload_offset +
                                              (key_value_head_count + key_value_head_idx) * plane_bytes);

        // K: pair-interleave each position tile via the ZA0 transpose; fully padded tiles
        // still run and store the zeros the score stage expects.
        for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / tile_dimension;
             position_tile_idx++) {
            nk_size_t const position_start = position_tile_idx * tile_dimension;
            nk_size_t const rows_to_pack = (position_start + tile_dimension <= position_count) ? tile_dimension
                                           : (position_start < position_count) ? position_count - position_start
                                                                               : 0;
            char const *source = (char const *)keys + (position_first + position_start) * key_stride +
                                 key_value_head_idx * depth * element_bytes;
            svbool_t const row_predicate_b32x = svwhilelt_b32_u64(0u, rows_to_pack);
            nk_u16_t *tile_output = keys_plane + position_tile_idx * depth_pairs * vector_elements;
            for (nk_size_t step = 0; step < depth_pairs; step++) {
                nk_size_t const slice = step % tile_dimension;
                if (slice == 0)
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_to_pack; row_in_tile++) {
                        char const *row = source + row_in_tile * key_stride + 2 * step * element_bytes;
                        svbool_t const depth_predicate_b16x = svwhilelt_b16_u64(2 * step, depth);
                        svuint16_t const row_u16x =
                            widened ? nk_attention_e4m3_to_bf16_sme_(
                                          depth_predicate_b16x,
                                          svld1_u8(svwhilelt_b8_u64(2 * step,
                                                                    nk_min_of_two(depth, 2 * step + vector_elements)),
                                                   (nk_u8_t const *)row))
                                    : svld1_u16(depth_predicate_b16x, (nk_u16_t const *)row);
                        svwrite_hor_za32_u32_m(0, (uint32_t)row_in_tile, predicate_all_b32x,
                                               svreinterpret_u32_u16(row_u16x));
                    }
                svuint16_t const packed_u16x = svreinterpret_u16_u32(
                    svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32x, 0, (uint32_t)slice));
                svst1_u16(svptrue_b16(), tile_output + step * vector_elements, packed_u16x);
            }
        }

        // V: one ZIP1 turns two position rows of a channel tile into the pair-interleaved
        // MOPA operand; dead positions and channels arrive as zeros from the predicates.
        for (nk_size_t channel_tile_idx = 0; channel_tile_idx < depth_padded / tile_dimension; channel_tile_idx++) {
            nk_size_t const channel_start = channel_tile_idx * tile_dimension;
            char const *source = (char const *)values + position_first * value_stride +
                                 (key_value_head_idx * depth + channel_start) * element_bytes;
            nk_u16_t *tile_output = values_plane + channel_tile_idx * (position_count_padded / 2) * vector_elements;
            svbool_t const channel_predicate_b16x = svwhilelt_b16_u64(channel_start, depth);
            for (nk_size_t position_pair_idx = 0; position_pair_idx < position_count_padded / 2; position_pair_idx++) {
                nk_size_t const position_even = position_pair_idx * 2, position_odd = position_even + 1;
                char const *even_row = source + position_even * value_stride;
                char const *odd_row = source + position_odd * value_stride;
                svuint16_t even_u16x = svdup_u16(0), odd_u16x = svdup_u16(0);
                if (widened) {
                    svbool_t const channel_bytes_b8x = svwhilelt_b8_u64(channel_start, depth);
                    if (position_even < position_count)
                        even_u16x = nk_attention_e4m3_to_bf16_sme_(
                            channel_predicate_b16x, svld1_u8(channel_bytes_b8x, (nk_u8_t const *)even_row));
                    if (position_odd < position_count)
                        odd_u16x = nk_attention_e4m3_to_bf16_sme_(
                            channel_predicate_b16x, svld1_u8(channel_bytes_b8x, (nk_u8_t const *)odd_row));
                }
                else {
                    if (position_even < position_count)
                        even_u16x = svld1_u16(channel_predicate_b16x, (nk_u16_t const *)even_row);
                    if (position_odd < position_count)
                        odd_u16x = svld1_u16(channel_predicate_b16x, (nk_u16_t const *)odd_row);
                }
                svst1_u16(svptrue_b16(), tile_output + position_pair_idx * vector_elements,
                          svzip1_u16(even_u16x, odd_u16x));
            }
        }
    }
}

/** Zeroes ZA and accumulates the 2×2 widening BFMOPA scores of the staged query row-tiles at
 *  @p queries_low and @p queries_high against the key position-tiles at @p keys_low and
 *  @p keys_high, over @p depth_pairs pair-interleaved steps. */
NUMKONG_INLINE void nk_attention_scores_bf16_sme_streaming_(nk_u16_t const *queries_low, nk_u16_t const *queries_high,
                                                            nk_u16_t const *keys_low, nk_u16_t const *keys_high,
                                                            nk_size_t depth_pairs) NUMKONG_STREAMING_
    __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b16x = svptrue_b16();
    svzero_za();
    for (nk_size_t depth_pair_idx = 0; depth_pair_idx < depth_pairs; depth_pair_idx++) {
        nk_size_t const step_offset = depth_pair_idx * vector_elements;
        svbfloat16_t const queries_low_bf16x = svreinterpret_bf16_u16(
            svld1_u16(predicate_all_b16x, queries_low + step_offset));
        svbfloat16_t const queries_high_bf16x = svreinterpret_bf16_u16(
            svld1_u16(predicate_all_b16x, queries_high + step_offset));
        svbfloat16_t const keys_low_bf16x = svreinterpret_bf16_u16(
            svld1_u16(predicate_all_b16x, keys_low + step_offset));
        svbfloat16_t const keys_high_bf16x = svreinterpret_bf16_u16(
            svld1_u16(predicate_all_b16x, keys_high + step_offset));
        svmopa_za32_bf16_m(0, predicate_all_b16x, predicate_all_b16x, queries_low_bf16x, keys_low_bf16x);
        svmopa_za32_bf16_m(1, predicate_all_b16x, predicate_all_b16x, queries_low_bf16x, keys_high_bf16x);
        svmopa_za32_bf16_m(2, predicate_all_b16x, predicate_all_b16x, queries_high_bf16x, keys_low_bf16x);
        svmopa_za32_bf16_m(3, predicate_all_b16x, predicate_all_b16x, queries_high_bf16x, keys_high_bf16x);
    }
}

/** The F16 twin of @c nk_attention_scores_bf16_sme_streaming_. */
NUMKONG_INLINE void nk_attention_scores_f16_sme_streaming_(nk_u16_t const *queries_low, nk_u16_t const *queries_high,
                                                           nk_u16_t const *keys_low, nk_u16_t const *keys_high,
                                                           nk_size_t depth_pairs) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b16x = svptrue_b16();
    svzero_za();
    for (nk_size_t depth_pair_idx = 0; depth_pair_idx < depth_pairs; depth_pair_idx++) {
        nk_size_t const step_offset = depth_pair_idx * vector_elements;
        svfloat16_t const queries_low_f16x = svreinterpret_f16_u16(
            svld1_u16(predicate_all_b16x, queries_low + step_offset));
        svfloat16_t const queries_high_f16x = svreinterpret_f16_u16(
            svld1_u16(predicate_all_b16x, queries_high + step_offset));
        svfloat16_t const keys_low_f16x = svreinterpret_f16_u16(svld1_u16(predicate_all_b16x, keys_low + step_offset));
        svfloat16_t const keys_high_f16x = svreinterpret_f16_u16(
            svld1_u16(predicate_all_b16x, keys_high + step_offset));
        svmopa_za32_f16_m(0, predicate_all_b16x, predicate_all_b16x, queries_low_f16x, keys_low_f16x);
        svmopa_za32_f16_m(1, predicate_all_b16x, predicate_all_b16x, queries_low_f16x, keys_high_f16x);
        svmopa_za32_f16_m(2, predicate_all_b16x, predicate_all_b16x, queries_high_f16x, keys_low_f16x);
        svmopa_za32_f16_m(3, predicate_all_b16x, predicate_all_b16x, queries_high_f16x, keys_high_f16x);
    }
}

/** Drains the 2×2 score tiles of the chunk at key @p first_position into @p chunk_scores,
 *  position-major with one query per lane, folding the scores into the panel maxima, hidden keys
 *  excluded when @p panel_masked. Padded positions contribute exact zeros, which may only raise
 *  the maximum, always numerically safe and cancelling in normalization. */
NUMKONG_INLINE void nk_attention_drain_scores_sme_streaming_(                                      //
    nk_f32_t *chunk_scores, nk_size_t first_position, int panel_masked,                            //
    svuint32_t key_begins_low_u32x, svuint32_t key_begins_high_u32x, svuint32_t key_ends_low_u32x, //
    svuint32_t key_ends_high_u32x, svfloat32_t *panel_max_low_f32x,                                //
    svfloat32_t *panel_max_high_f32x) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw(), block_rows_capacity = 2 * tile_dimension;
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t slice_idx = 0; slice_idx < tile_dimension; slice_idx++) {
        svfloat32_t const column0_low_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 0,
                                                                   (uint32_t)slice_idx);
        svfloat32_t const column0_high_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 2,
                                                                    (uint32_t)slice_idx);
        svfloat32_t const column1_low_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 1,
                                                                   (uint32_t)slice_idx);
        svfloat32_t const column1_high_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 3,
                                                                    (uint32_t)slice_idx);
        svfloat32_t maximum0_low_f32x = column0_low_f32x, maximum0_high_f32x = column0_high_f32x;
        svfloat32_t maximum1_low_f32x = column1_low_f32x, maximum1_high_f32x = column1_high_f32x;
        if (panel_masked) { // hidden keys never raise the maximum
            nk_size_t const position0 = first_position + slice_idx;
            nk_size_t const position1 = position0 + tile_dimension;
            svfloat32_t const hidden_f32x = svdup_f32(NUMKONG_F32_MIN);
            maximum0_low_f32x = svsel_f32(nk_attention_visible_sme_(key_begins_low_u32x, key_ends_low_u32x, position0),
                                          column0_low_f32x, hidden_f32x);
            maximum0_high_f32x = svsel_f32(
                nk_attention_visible_sme_(key_begins_high_u32x, key_ends_high_u32x, position0), column0_high_f32x,
                hidden_f32x);
            maximum1_low_f32x = svsel_f32(nk_attention_visible_sme_(key_begins_low_u32x, key_ends_low_u32x, position1),
                                          column1_low_f32x, hidden_f32x);
            maximum1_high_f32x = svsel_f32(
                nk_attention_visible_sme_(key_begins_high_u32x, key_ends_high_u32x, position1), column1_high_f32x,
                hidden_f32x);
        }
        *panel_max_low_f32x = svmax_f32_x(predicate_all_b32x,
                                          svmax_f32_x(predicate_all_b32x, *panel_max_low_f32x, maximum0_low_f32x),
                                          maximum1_low_f32x);
        *panel_max_high_f32x = svmax_f32_x(predicate_all_b32x,
                                           svmax_f32_x(predicate_all_b32x, *panel_max_high_f32x, maximum0_high_f32x),
                                           maximum1_high_f32x);
        nk_f32_t *column0_scores = chunk_scores + slice_idx * block_rows_capacity;
        nk_f32_t *column1_scores = chunk_scores + (tile_dimension + slice_idx) * block_rows_capacity;
        svst1_f32(predicate_all_b32x, (float32_t *)column0_scores, column0_low_f32x);
        svst1_f32(predicate_all_b32x, (float32_t *)(column0_scores + tile_dimension), column0_high_f32x);
        svst1_f32(predicate_all_b32x, (float32_t *)column1_scores, column1_low_f32x);
        svst1_f32(predicate_all_b32x, (float32_t *)(column1_scores + tile_dimension), column1_high_f32x);
    }
}

/** The F32 weights 2^(s · scale₂ − m₂) of position pair @p pair_idx of the score panel at key
 *  @p panel_start, for both row-tiles; positions past @p panel_length and keys hidden when
 *  @p panel_masked take a sentinel that decays to zero weight without NaN. */
NUMKONG_INLINE void nk_attention_pair_weights_sme_streaming_(                                        //
    nk_f32_t const *scores_panel, nk_size_t pair_idx, nk_size_t panel_start, nk_size_t panel_length, //
    int panel_masked, svfloat32_t scale2_f32x, svfloat32_t negated_max2_low_f32x,                    //
    svfloat32_t negated_max2_high_f32x, svuint32_t key_begins_low_u32x,                              //
    svuint32_t key_begins_high_u32x, svuint32_t key_ends_low_u32x, svuint32_t key_ends_high_u32x,    //
    svfloat32_t *weight_even_low_f32x, svfloat32_t *weight_even_high_f32x,                           //
    svfloat32_t *weight_odd_low_f32x, svfloat32_t *weight_odd_high_f32x) NUMKONG_STREAMING_ {
    nk_size_t const tile_dimension = svcntw(), block_rows_capacity = 2 * tile_dimension;
    svbool_t const predicate_all_b32x = svptrue_b32();
    nk_size_t const position_even = pair_idx * 2, position_odd = position_even + 1;
    nk_f32_t const *even_scores = scores_panel + position_even * block_rows_capacity;
    nk_f32_t const *odd_scores = scores_panel + position_odd * block_rows_capacity;
    svfloat32_t even_low_f32x = svld1_f32(predicate_all_b32x, (float32_t const *)even_scores);
    svfloat32_t even_high_f32x = svld1_f32(predicate_all_b32x, (float32_t const *)(even_scores + tile_dimension));
    svfloat32_t odd_low_f32x = svdup_f32(NUMKONG_F32_MIN);
    svfloat32_t odd_high_f32x = svdup_f32(NUMKONG_F32_MIN);
    if (position_odd < panel_length) {
        odd_low_f32x = svld1_f32(predicate_all_b32x, (float32_t const *)odd_scores);
        odd_high_f32x = svld1_f32(predicate_all_b32x, (float32_t const *)(odd_scores + tile_dimension));
        odd_low_f32x = svmla_f32_x(predicate_all_b32x, negated_max2_low_f32x, odd_low_f32x, scale2_f32x);
        odd_high_f32x = svmla_f32_x(predicate_all_b32x, negated_max2_high_f32x, odd_high_f32x, scale2_f32x);
    }
    even_low_f32x = svmla_f32_x(predicate_all_b32x, negated_max2_low_f32x, even_low_f32x, scale2_f32x);
    even_high_f32x = svmla_f32_x(predicate_all_b32x, negated_max2_high_f32x, even_high_f32x, scale2_f32x);
    if (panel_masked) {
        nk_size_t const position_absolute = panel_start + position_even;
        svfloat32_t const hidden_f32x = svdup_f32(NUMKONG_F32_MIN);
        even_low_f32x = svsel_f32(nk_attention_visible_sme_(key_begins_low_u32x, key_ends_low_u32x, position_absolute),
                                  even_low_f32x, hidden_f32x);
        even_high_f32x = svsel_f32(
            nk_attention_visible_sme_(key_begins_high_u32x, key_ends_high_u32x, position_absolute), even_high_f32x,
            hidden_f32x);
        odd_low_f32x = svsel_f32(
            nk_attention_visible_sme_(key_begins_low_u32x, key_ends_low_u32x, position_absolute + 1), odd_low_f32x,
            hidden_f32x);
        odd_high_f32x = svsel_f32(
            nk_attention_visible_sme_(key_begins_high_u32x, key_ends_high_u32x, position_absolute + 1), odd_high_f32x,
            hidden_f32x);
    }
    svfloat32_t const whole_even_low_f32x = svrintn_f32_x(predicate_all_b32x, even_low_f32x);
    svfloat32_t const whole_even_high_f32x = svrintn_f32_x(predicate_all_b32x, even_high_f32x);
    svfloat32_t const whole_odd_low_f32x = svrintn_f32_x(predicate_all_b32x, odd_low_f32x);
    svfloat32_t const whole_odd_high_f32x = svrintn_f32_x(predicate_all_b32x, odd_high_f32x);
    // Split precision: only the fraction's polynomial runs in F16
    svfloat16_t const fraction_even_f16x = svcvtnt_f16_f32_m(
        svcvt_f16_f32_x(predicate_all_b32x, svsub_f32_x(predicate_all_b32x, even_low_f32x, whole_even_low_f32x)),
        predicate_all_b32x, svsub_f32_x(predicate_all_b32x, even_high_f32x, whole_even_high_f32x));
    svfloat16_t const fraction_odd_f16x = svcvtnt_f16_f32_m(
        svcvt_f16_f32_x(predicate_all_b32x, svsub_f32_x(predicate_all_b32x, odd_low_f32x, whole_odd_low_f32x)),
        predicate_all_b32x, svsub_f32_x(predicate_all_b32x, odd_high_f32x, whole_odd_high_f32x));
    svfloat16_t const poly_even_f16x = nk_exp2_polynomial_f16x_sme_(fraction_even_f16x);
    svfloat16_t const poly_odd_f16x = nk_exp2_polynomial_f16x_sme_(fraction_odd_f16x);
    *weight_even_low_f32x = svscale_f32_x(predicate_all_b32x, svcvt_f32_f16_x(predicate_all_b32x, poly_even_f16x),
                                          svcvt_s32_f32_x(predicate_all_b32x, whole_even_low_f32x));
    *weight_even_high_f32x = svscale_f32_x(predicate_all_b32x, svcvtlt_f32_f16_x(predicate_all_b32x, poly_even_f16x),
                                           svcvt_s32_f32_x(predicate_all_b32x, whole_even_high_f32x));
    *weight_odd_low_f32x = svscale_f32_x(predicate_all_b32x, svcvt_f32_f16_x(predicate_all_b32x, poly_odd_f16x),
                                         svcvt_s32_f32_x(predicate_all_b32x, whole_odd_low_f32x));
    *weight_odd_high_f32x = svscale_f32_x(predicate_all_b32x, svcvtlt_f32_f16_x(predicate_all_b32x, poly_odd_f16x),
                                          svcvt_s32_f32_x(predicate_all_b32x, whole_odd_high_f32x));
}

/** Zeroes ZA and accumulates the widening BFMOPA products of the pair-interleaved weights of
 *  @p panel_pairs position pairs against the value channel-tile at @p values_low, and the one at
 *  @p values_high when @p has_second_tile. */
NUMKONG_INLINE void nk_attention_values_bf16_sme_streaming_(nk_u16_t const *weights_panel, nk_u16_t const *values_low,
                                                            nk_u16_t const *values_high, nk_size_t panel_pairs,
                                                            int has_second_tile) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b16x = svptrue_b16();
    svzero_za();
    for (nk_size_t pair_idx = 0; pair_idx < panel_pairs; pair_idx++) {
        svbfloat16_t const weights_low_bf16x = svreinterpret_bf16_u16(
            svld1_u16(predicate_all_b16x, weights_panel + (pair_idx * 2 + 0) * vector_elements));
        svbfloat16_t const weights_high_bf16x = svreinterpret_bf16_u16(
            svld1_u16(predicate_all_b16x, weights_panel + (pair_idx * 2 + 1) * vector_elements));
        svbfloat16_t const values_low_bf16x = svreinterpret_bf16_u16(
            svld1_u16(predicate_all_b16x, values_low + pair_idx * vector_elements));
        svmopa_za32_bf16_m(0, predicate_all_b16x, predicate_all_b16x, weights_low_bf16x, values_low_bf16x);
        svmopa_za32_bf16_m(2, predicate_all_b16x, predicate_all_b16x, weights_high_bf16x, values_low_bf16x);
        if (has_second_tile) {
            svbfloat16_t const values_high_bf16x = svreinterpret_bf16_u16(
                svld1_u16(predicate_all_b16x, values_high + pair_idx * vector_elements));
            svmopa_za32_bf16_m(1, predicate_all_b16x, predicate_all_b16x, weights_low_bf16x, values_high_bf16x);
            svmopa_za32_bf16_m(3, predicate_all_b16x, predicate_all_b16x, weights_high_bf16x, values_high_bf16x);
        }
    }
}

/** The F16 twin of @c nk_attention_values_bf16_sme_streaming_. */
NUMKONG_INLINE void nk_attention_values_f16_sme_streaming_(nk_u16_t const *weights_panel, nk_u16_t const *values_low,
                                                           nk_u16_t const *values_high, nk_size_t panel_pairs,
                                                           int has_second_tile) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b16x = svptrue_b16();
    svzero_za();
    for (nk_size_t pair_idx = 0; pair_idx < panel_pairs; pair_idx++) {
        svfloat16_t const weights_low_f16x = svreinterpret_f16_u16(
            svld1_u16(predicate_all_b16x, weights_panel + (pair_idx * 2 + 0) * vector_elements));
        svfloat16_t const weights_high_f16x = svreinterpret_f16_u16(
            svld1_u16(predicate_all_b16x, weights_panel + (pair_idx * 2 + 1) * vector_elements));
        svfloat16_t const values_low_f16x = svreinterpret_f16_u16(
            svld1_u16(predicate_all_b16x, values_low + pair_idx * vector_elements));
        svmopa_za32_f16_m(0, predicate_all_b16x, predicate_all_b16x, weights_low_f16x, values_low_f16x);
        svmopa_za32_f16_m(2, predicate_all_b16x, predicate_all_b16x, weights_high_f16x, values_low_f16x);
        if (has_second_tile) {
            svfloat16_t const values_high_f16x = svreinterpret_f16_u16(
                svld1_u16(predicate_all_b16x, values_high + pair_idx * vector_elements));
            svmopa_za32_f16_m(1, predicate_all_b16x, predicate_all_b16x, weights_low_f16x, values_high_f16x);
            svmopa_za32_f16_m(3, predicate_all_b16x, predicate_all_b16x, weights_high_f16x, values_high_f16x);
        }
    }
}

/** Drains the P × V tiles of channel tile @p channel_tile_idx, and of the next one when
 *  @p has_second_tile, into the channel-major accumulator @p o_acc, rescaled by the corrections. */
NUMKONG_INLINE void nk_attention_drain_values_sme_streaming_(nk_f32_t *o_acc, nk_size_t channel_tile_idx,
                                                             int has_second_tile, svfloat32_t correction_low_f32x,
                                                             svfloat32_t correction_high_f32x) NUMKONG_STREAMING_
    __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw(), block_rows_capacity = 2 * tile_dimension;
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t slice_idx = 0; slice_idx < tile_dimension; slice_idx++) {
        nk_f32_t *accumulator_tile0 = o_acc + (channel_tile_idx * tile_dimension + slice_idx) * block_rows_capacity;
        svfloat32_t o_tile0_low_f32x = svld1_f32(predicate_all_b32x, (float32_t const *)accumulator_tile0);
        svfloat32_t o_tile0_high_f32x = svld1_f32(predicate_all_b32x,
                                                  (float32_t const *)(accumulator_tile0 + tile_dimension));
        o_tile0_low_f32x = svmad_f32_x(
            predicate_all_b32x, o_tile0_low_f32x, correction_low_f32x,
            svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 0, (uint32_t)slice_idx));
        o_tile0_high_f32x = svmad_f32_x(
            predicate_all_b32x, o_tile0_high_f32x, correction_high_f32x,
            svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 2, (uint32_t)slice_idx));
        svst1_f32(predicate_all_b32x, (float32_t *)accumulator_tile0, o_tile0_low_f32x);
        svst1_f32(predicate_all_b32x, (float32_t *)(accumulator_tile0 + tile_dimension), o_tile0_high_f32x);
        if (has_second_tile) {
            nk_f32_t *accumulator_tile1 = accumulator_tile0 + tile_dimension * block_rows_capacity;
            svfloat32_t o_tile1_low_f32x = svld1_f32(predicate_all_b32x, (float32_t const *)accumulator_tile1);
            svfloat32_t o_tile1_high_f32x = svld1_f32(predicate_all_b32x,
                                                      (float32_t const *)(accumulator_tile1 + tile_dimension));
            o_tile1_low_f32x = svmad_f32_x(
                predicate_all_b32x, o_tile1_low_f32x, correction_low_f32x,
                svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 1, (uint32_t)slice_idx));
            o_tile1_high_f32x = svmad_f32_x(
                predicate_all_b32x, o_tile1_high_f32x, correction_high_f32x,
                svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 3, (uint32_t)slice_idx));
            svst1_f32(predicate_all_b32x, (float32_t *)accumulator_tile1, o_tile1_low_f32x);
            svst1_f32(predicate_all_b32x, (float32_t *)(accumulator_tile1 + tile_dimension), o_tile1_high_f32x);
        }
    }
}

/**
 *  @brief Finishes a block of @p block_rows query rows: scales the channel-major accumulator
 *      lane-wise by @p value_scale over the weight sums and by two to the @p value_exponent, then
 *      transposes it back to output rows through ZA0, one row-tile × channel-tile at a time.
 *
 *  Only the rows from @p store_begin up to @p store_end are stored: to @p output rows,
 *  @p output_stride_floats apart, and when @p log_sum_exp is not null, to it, @p log_sum_exp_stride
 *  apart. Rows that saw no key keep a zero sum and emit zeros.
 */
NUMKONG_INLINE void nk_attention_finish_block_sme_streaming_(                                         //
    nk_f32_t const *o_acc, svfloat32_t running_max2_low_f32x, svfloat32_t running_max2_high_f32x,     //
    svfloat32_t running_sum_low_f32x, svfloat32_t running_sum_high_f32x, nk_size_t depth,             //
    nk_size_t block_rows, nk_size_t store_begin, nk_size_t store_end, nk_f32_t value_scale,           //
    nk_i32_t value_exponent, nk_f32_t *output, nk_size_t output_stride_floats, nk_f32_t *log_sum_exp, //
    nk_size_t log_sum_exp_stride) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const tile_dimension = svcntw(), block_rows_capacity = 2 * tile_dimension;
    nk_size_t const channel_tiles = nk_size_divide_round_up_(depth, tile_dimension);
    svbool_t const predicate_all_b32x = svptrue_b32();
    svint32_t const value_exponent_i32x = svdup_s32(value_exponent);
    svfloat32_t const inverse_sum_low_f32x = svsel_f32(
        svcmpgt_n_f32(predicate_all_b32x, running_sum_low_f32x, 0.0f),
        svdiv_f32_x(predicate_all_b32x, svdup_f32(value_scale), running_sum_low_f32x), svdup_f32(0.0f));
    svfloat32_t const inverse_sum_high_f32x = svsel_f32(
        svcmpgt_n_f32(predicate_all_b32x, running_sum_high_f32x, 0.0f),
        svdiv_f32_x(predicate_all_b32x, svdup_f32(value_scale), running_sum_high_f32x), svdup_f32(0.0f));
    for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
        nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
        if (tile_row_start >= block_rows) break;
        nk_size_t const rows_valid = (block_rows - tile_row_start < tile_dimension) ? block_rows - tile_row_start
                                                                                    : tile_dimension;
        svfloat32_t const inverse_sum_f32x = row_tile_idx == 0 ? inverse_sum_low_f32x : inverse_sum_high_f32x;
        if (log_sum_exp) {
            nk_f32_t row_maxima2[nk_attention_max_tile_sme_k_], row_sums[nk_attention_max_tile_sme_k_];
            svst1_f32(predicate_all_b32x, row_maxima2,
                      row_tile_idx == 0 ? running_max2_low_f32x : running_max2_high_f32x);
            svst1_f32(predicate_all_b32x, row_sums, row_tile_idx == 0 ? running_sum_low_f32x : running_sum_high_f32x);
            for (nk_size_t row_in_tile = 0; row_in_tile < rows_valid; row_in_tile++) {
                nk_size_t const row = tile_row_start + row_in_tile;
                if (row < store_begin || row >= store_end) continue;
                log_sum_exp[row * log_sum_exp_stride] = nk_attention_log_sum_exp_serial_(row_maxima2[row_in_tile],
                                                                                         row_sums[row_in_tile]);
            }
        }
        for (nk_size_t channel_tile_idx = 0; channel_tile_idx < channel_tiles; channel_tile_idx++) {
            nk_size_t const channel_start = channel_tile_idx * tile_dimension;
            svbool_t const channel_predicate_b32x = svwhilelt_b32_u64(channel_start, depth);
            svzero_mask_za(nk_sme_zero_za32_tile_0_k);
            for (nk_size_t channel_in_tile = 0; channel_in_tile < tile_dimension; channel_in_tile++) {
                svfloat32_t const accumulated_f32x = svld1_f32(
                    predicate_all_b32x,
                    (float32_t const *)(o_acc + (channel_start + channel_in_tile) * block_rows_capacity +
                                        tile_row_start));
                // Scaling the product rounds once, where the power of two alone may be subnormal
                svfloat32_t const normalized_f32x = svscale_f32_x(
                    predicate_all_b32x, svmul_f32_x(predicate_all_b32x, accumulated_f32x, inverse_sum_f32x),
                    value_exponent_i32x);
                svwrite_hor_za32_f32_m(0, (uint32_t)channel_in_tile, predicate_all_b32x, normalized_f32x);
            }
            for (nk_size_t row_in_tile = 0; row_in_tile < rows_valid; row_in_tile++) {
                nk_size_t const row = tile_row_start + row_in_tile;
                if (row < store_begin || row >= store_end) continue;
                svst1_ver_za32(0, (uint32_t)row_in_tile, channel_predicate_b32x,
                               output + row * output_stride_floats + channel_start);
            }
        }
    }
}

/**
 *  @brief Attends one block of @p block_rows query rows, staged as pair-interleaved BF16 at
 *      @p queries_low and @p queries_high, over BF16 K and V tile planes.
 *
 *  1. Scores: 2×2 widening BFMOPA blocking (two Q row-tiles × two K position-tiles), one vector
 *     load per MOPA; tiles drain through vertical stores into a position-major F32 panel with one
 *     query per lane.
 *  2. Softmax: lane-parallel running maximum, correction 2^(m_old − m_new), and weight sums;
 *     each position pair's weights convert to pair-interleaved BF16 in registers.
 *  3. P × V: 2×2 widening BFMOPA over position pairs into (row-tile × channel-tile) accumulators;
 *     vertical-slice drains fuse the correction FMA into the channel-major output accumulator,
 *     keeping queries in lanes end to end.
 *
 *  Scores multiply by @p scale2, and @c nk_attention_finish_block_sme_streaming_ stores the rows.
 *  The block's first row sits at key position @p block_first_position under @p band.
 */
NUMKONG_INLINE void nk_attention_block_bf16_sme_streaming_(                                           //
    nk_u16_t const *queries_low, nk_u16_t const *queries_high, nk_u16_t const *keys_plane,            //
    nk_u16_t const *values_plane, nk_size_t depth, nk_size_t position_count, nk_diagonal_band_t band, //
    nk_i64_t block_first_position, nk_size_t block_rows, nk_size_t store_begin, nk_size_t store_end,  //
    nk_f32_t scale2, nk_f32_t value_scale, nk_i32_t value_exponent, nk_f32_t *output,                 //
    nk_size_t output_stride_floats, nk_f32_t *log_sum_exp, nk_size_t log_sum_exp_stride) NUMKONG_STREAMING_
    __arm_inout("za") {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const panel_width = nk_attention_panel_sme_k_;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2;
    nk_size_t const channel_tiles = depth_padded / tile_dimension;
    nk_size_t const position_pairs_total = nk_size_round_up_to_multiple_(position_count, vector_elements) / 2;

    // Queries stay in lanes, so the accumulator is channel-major: `o_acc[channel][query lane]`
    nk_align_(64) nk_f32_t scores_panel[nk_attention_panel_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    nk_align_(64) nk_u16_t weights_panel[nk_attention_panel_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    nk_align_(64) nk_f32_t o_acc[nk_attention_max_depth_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    nk_align_(64) nk_u32_t key_begins[2 * nk_attention_max_tile_sme_k_]; // visible key range per query lane
    nk_align_(64) nk_u32_t key_ends[2 * nk_attention_max_tile_sme_k_];

    svbool_t const predicate_all_b32x = svptrue_b32();
    svbool_t const predicate_all_b16x = svptrue_b16();
    svfloat32_t const scale2_f32x = svdup_f32(scale2);

    // Lanes past the block see no keys; ranges grow monotonically with the row, so the first and
    // last rows bound the block's.
    for (nk_size_t lane_idx = 0; lane_idx < block_rows_capacity; lane_idx++) {
        nk_size_t key_begin = 0, key_end = 0;
        if (lane_idx < block_rows)
            nk_diagonal_band_row_range_(band, block_first_position + (nk_i64_t)lane_idx, position_count, &key_begin,
                                        &key_end);
        key_begins[lane_idx] = (nk_u32_t)key_begin, key_ends[lane_idx] = (nk_u32_t)key_end;
    }
    nk_size_t const block_key_begin = key_begins[0], block_key_end = key_ends[block_rows - 1];
    svuint32_t const key_begins_low_u32x = svld1_u32(predicate_all_b32x, key_begins);
    svuint32_t const key_begins_high_u32x = svld1_u32(predicate_all_b32x, key_begins + tile_dimension);
    svuint32_t const key_ends_low_u32x = svld1_u32(predicate_all_b32x, key_ends);
    svuint32_t const key_ends_high_u32x = svld1_u32(predicate_all_b32x, key_ends + tile_dimension);

    svfloat32_t running_max2_low_f32x = svdup_f32(NUMKONG_F32_MIN); // row-tile 0 queries, one per lane
    svfloat32_t running_max2_high_f32x = svdup_f32(NUMKONG_F32_MIN);
    svfloat32_t running_sum_low_f32x = svdup_f32(0.0f);
    svfloat32_t running_sum_high_f32x = svdup_f32(0.0f);
    for (nk_size_t element_idx = 0; element_idx < depth_padded * block_rows_capacity; element_idx += tile_dimension)
        svst1_f32(predicate_all_b32x, (float32_t *)(o_acc + element_idx), svdup_f32(0.0f));

    // Only panels crossing some row's range boundary pay for per-lane predicates
    for (nk_size_t panel_start = block_key_begin / panel_width * panel_width; panel_start < block_key_end;
         panel_start += panel_width) {
        nk_size_t const panel_length = (block_key_end - panel_start < panel_width) ? block_key_end - panel_start
                                                                                   : panel_width;
        nk_diagonal_band_coverage_t const coverage = nk_diagonal_band_tile_coverage_(
            band, block_first_position, block_rows, panel_start, panel_length);
        if (coverage == nk_diagonal_band_outside_k) continue;
        int const panel_masked = coverage == nk_diagonal_band_crossing_k;
        nk_size_t const panel_pairs = nk_size_divide_round_up_(panel_length, 2);

        svfloat32_t panel_max_low_f32x = svdup_f32(NUMKONG_F32_MIN);
        svfloat32_t panel_max_high_f32x = svdup_f32(NUMKONG_F32_MIN);
        for (nk_size_t chunk_start = 0; chunk_start < panel_length; chunk_start += block_rows_capacity) {
            nk_u16_t const *keys_tile0 = keys_plane +
                                         (panel_start + chunk_start) / tile_dimension * depth_pairs * vector_elements;
            nk_attention_scores_bf16_sme_streaming_(queries_low, queries_high, keys_tile0,
                                                    keys_tile0 + depth_pairs * vector_elements, depth_pairs);
            nk_attention_drain_scores_sme_streaming_(scores_panel + chunk_start * block_rows_capacity,
                                                     panel_start + chunk_start, panel_masked, key_begins_low_u32x,
                                                     key_begins_high_u32x, key_ends_low_u32x, key_ends_high_u32x,
                                                     &panel_max_low_f32x, &panel_max_high_f32x);
        }

        svfloat32_t const new_max2_low_f32x = svmax_f32_x(
            predicate_all_b32x, running_max2_low_f32x,
            svmul_f32_x(predicate_all_b32x, panel_max_low_f32x, scale2_f32x));
        svfloat32_t const new_max2_high_f32x = svmax_f32_x(
            predicate_all_b32x, running_max2_high_f32x,
            svmul_f32_x(predicate_all_b32x, panel_max_high_f32x, scale2_f32x));
        svfloat32_t const correction_low_f32x = nk_exp2_f32x_sme_(
            svsub_f32_x(predicate_all_b32x, running_max2_low_f32x, new_max2_low_f32x));
        svfloat32_t const correction_high_f32x = nk_exp2_f32x_sme_(
            svsub_f32_x(predicate_all_b32x, running_max2_high_f32x, new_max2_high_f32x));
        running_max2_low_f32x = new_max2_low_f32x;
        running_max2_high_f32x = new_max2_high_f32x;
        svfloat32_t const negated_max2_low_f32x = svneg_f32_x(predicate_all_b32x, new_max2_low_f32x);
        svfloat32_t const negated_max2_high_f32x = svneg_f32_x(predicate_all_b32x, new_max2_high_f32x);

        svfloat32_t panel_sum_low_f32x = svdup_f32(0.0f);
        svfloat32_t panel_sum_high_f32x = svdup_f32(0.0f);
        for (nk_size_t pair_idx = 0; pair_idx < panel_pairs; pair_idx++) {
            svfloat32_t weight_even_low_f32x, weight_even_high_f32x, weight_odd_low_f32x, weight_odd_high_f32x;
            nk_attention_pair_weights_sme_streaming_(scores_panel, pair_idx, panel_start, panel_length, panel_masked,
                                                     scale2_f32x, negated_max2_low_f32x, negated_max2_high_f32x,
                                                     key_begins_low_u32x, key_begins_high_u32x, key_ends_low_u32x,
                                                     key_ends_high_u32x, &weight_even_low_f32x, &weight_even_high_f32x,
                                                     &weight_odd_low_f32x, &weight_odd_high_f32x);
            panel_sum_low_f32x = svadd_f32_x(
                predicate_all_b32x, panel_sum_low_f32x,
                svadd_f32_x(predicate_all_b32x, weight_even_low_f32x, weight_odd_low_f32x));
            panel_sum_high_f32x = svadd_f32_x(
                predicate_all_b32x, panel_sum_high_f32x,
                svadd_f32_x(predicate_all_b32x, weight_even_high_f32x, weight_odd_high_f32x));
            svst1_u16(predicate_all_b16x, weights_panel + (pair_idx * 2 + 0) * vector_elements,
                      nk_attention_bf16_pair_sme_(weight_even_low_f32x, weight_odd_low_f32x));
            svst1_u16(predicate_all_b16x, weights_panel + (pair_idx * 2 + 1) * vector_elements,
                      nk_attention_bf16_pair_sme_(weight_even_high_f32x, weight_odd_high_f32x));
        }
        running_sum_low_f32x = svmad_f32_x(predicate_all_b32x, running_sum_low_f32x, correction_low_f32x,
                                           panel_sum_low_f32x);
        running_sum_high_f32x = svmad_f32_x(predicate_all_b32x, running_sum_high_f32x, correction_high_f32x,
                                            panel_sum_high_f32x);

        for (nk_size_t channel_tile_idx = 0; channel_tile_idx < channel_tiles; channel_tile_idx += 2) {
            int const has_second_tile = channel_tile_idx + 1 < channel_tiles;
            nk_u16_t const *values_tile0 = values_plane + (channel_tile_idx * position_pairs_total + panel_start / 2) *
                                                              vector_elements;
            nk_attention_values_bf16_sme_streaming_(weights_panel, values_tile0,
                                                    values_tile0 + position_pairs_total * vector_elements, panel_pairs,
                                                    has_second_tile);
            nk_attention_drain_values_sme_streaming_(o_acc, channel_tile_idx, has_second_tile, correction_low_f32x,
                                                     correction_high_f32x);
        }
    }
    nk_attention_finish_block_sme_streaming_(o_acc, running_max2_low_f32x, running_max2_high_f32x, running_sum_low_f32x,
                                             running_sum_high_f32x, depth, block_rows, store_begin, store_end,
                                             value_scale, value_exponent, output, output_stride_floats, log_sum_exp,
                                             log_sum_exp_stride);
}

/** @c nk_attention_block_bf16_sme_streaming_ over F16 queries, tiles and weights. */
NUMKONG_INLINE void nk_attention_block_f16_sme_streaming_(                                            //
    nk_u16_t const *queries_low, nk_u16_t const *queries_high, nk_u16_t const *keys_plane,            //
    nk_u16_t const *values_plane, nk_size_t depth, nk_size_t position_count, nk_diagonal_band_t band, //
    nk_i64_t block_first_position, nk_size_t block_rows, nk_size_t store_begin, nk_size_t store_end,  //
    nk_f32_t scale2, nk_f32_t value_scale, nk_i32_t value_exponent, nk_f32_t *output,                 //
    nk_size_t output_stride_floats, nk_f32_t *log_sum_exp, nk_size_t log_sum_exp_stride) NUMKONG_STREAMING_
    __arm_inout("za") {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const panel_width = nk_attention_panel_sme_k_;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2;
    nk_size_t const channel_tiles = depth_padded / tile_dimension;
    nk_size_t const position_pairs_total = nk_size_round_up_to_multiple_(position_count, vector_elements) / 2;

    nk_align_(64) nk_f32_t scores_panel[nk_attention_panel_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    nk_align_(64) nk_u16_t weights_panel[nk_attention_panel_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    nk_align_(64) nk_f32_t o_acc[nk_attention_max_depth_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    nk_align_(64) nk_u32_t key_begins[2 * nk_attention_max_tile_sme_k_];
    nk_align_(64) nk_u32_t key_ends[2 * nk_attention_max_tile_sme_k_];

    svbool_t const predicate_all_b32x = svptrue_b32();
    svbool_t const predicate_all_b16x = svptrue_b16();
    svfloat32_t const scale2_f32x = svdup_f32(scale2);

    for (nk_size_t lane_idx = 0; lane_idx < block_rows_capacity; lane_idx++) {
        nk_size_t key_begin = 0, key_end = 0;
        if (lane_idx < block_rows)
            nk_diagonal_band_row_range_(band, block_first_position + (nk_i64_t)lane_idx, position_count, &key_begin,
                                        &key_end);
        key_begins[lane_idx] = (nk_u32_t)key_begin, key_ends[lane_idx] = (nk_u32_t)key_end;
    }
    nk_size_t const block_key_begin = key_begins[0], block_key_end = key_ends[block_rows - 1];
    svuint32_t const key_begins_low_u32x = svld1_u32(predicate_all_b32x, key_begins);
    svuint32_t const key_begins_high_u32x = svld1_u32(predicate_all_b32x, key_begins + tile_dimension);
    svuint32_t const key_ends_low_u32x = svld1_u32(predicate_all_b32x, key_ends);
    svuint32_t const key_ends_high_u32x = svld1_u32(predicate_all_b32x, key_ends + tile_dimension);

    svfloat32_t running_max2_low_f32x = svdup_f32(NUMKONG_F32_MIN);
    svfloat32_t running_max2_high_f32x = svdup_f32(NUMKONG_F32_MIN);
    svfloat32_t running_sum_low_f32x = svdup_f32(0.0f);
    svfloat32_t running_sum_high_f32x = svdup_f32(0.0f);
    for (nk_size_t element_idx = 0; element_idx < depth_padded * block_rows_capacity; element_idx += tile_dimension)
        svst1_f32(predicate_all_b32x, (float32_t *)(o_acc + element_idx), svdup_f32(0.0f));

    for (nk_size_t panel_start = block_key_begin / panel_width * panel_width; panel_start < block_key_end;
         panel_start += panel_width) {
        nk_size_t const panel_length = (block_key_end - panel_start < panel_width) ? block_key_end - panel_start
                                                                                   : panel_width;
        nk_diagonal_band_coverage_t const coverage = nk_diagonal_band_tile_coverage_(
            band, block_first_position, block_rows, panel_start, panel_length);
        if (coverage == nk_diagonal_band_outside_k) continue;
        int const panel_masked = coverage == nk_diagonal_band_crossing_k;
        nk_size_t const panel_pairs = nk_size_divide_round_up_(panel_length, 2);

        svfloat32_t panel_max_low_f32x = svdup_f32(NUMKONG_F32_MIN);
        svfloat32_t panel_max_high_f32x = svdup_f32(NUMKONG_F32_MIN);
        for (nk_size_t chunk_start = 0; chunk_start < panel_length; chunk_start += block_rows_capacity) {
            nk_u16_t const *keys_tile0 = keys_plane +
                                         (panel_start + chunk_start) / tile_dimension * depth_pairs * vector_elements;
            nk_attention_scores_f16_sme_streaming_(queries_low, queries_high, keys_tile0,
                                                   keys_tile0 + depth_pairs * vector_elements, depth_pairs);
            nk_attention_drain_scores_sme_streaming_(scores_panel + chunk_start * block_rows_capacity,
                                                     panel_start + chunk_start, panel_masked, key_begins_low_u32x,
                                                     key_begins_high_u32x, key_ends_low_u32x, key_ends_high_u32x,
                                                     &panel_max_low_f32x, &panel_max_high_f32x);
        }

        svfloat32_t const new_max2_low_f32x = svmax_f32_x(
            predicate_all_b32x, running_max2_low_f32x,
            svmul_f32_x(predicate_all_b32x, panel_max_low_f32x, scale2_f32x));
        svfloat32_t const new_max2_high_f32x = svmax_f32_x(
            predicate_all_b32x, running_max2_high_f32x,
            svmul_f32_x(predicate_all_b32x, panel_max_high_f32x, scale2_f32x));
        svfloat32_t const correction_low_f32x = nk_exp2_f32x_sme_(
            svsub_f32_x(predicate_all_b32x, running_max2_low_f32x, new_max2_low_f32x));
        svfloat32_t const correction_high_f32x = nk_exp2_f32x_sme_(
            svsub_f32_x(predicate_all_b32x, running_max2_high_f32x, new_max2_high_f32x));
        running_max2_low_f32x = new_max2_low_f32x;
        running_max2_high_f32x = new_max2_high_f32x;
        svfloat32_t const negated_max2_low_f32x = svneg_f32_x(predicate_all_b32x, new_max2_low_f32x);
        svfloat32_t const negated_max2_high_f32x = svneg_f32_x(predicate_all_b32x, new_max2_high_f32x);

        svfloat32_t panel_sum_low_f32x = svdup_f32(0.0f);
        svfloat32_t panel_sum_high_f32x = svdup_f32(0.0f);
        for (nk_size_t pair_idx = 0; pair_idx < panel_pairs; pair_idx++) {
            svfloat32_t weight_even_low_f32x, weight_even_high_f32x, weight_odd_low_f32x, weight_odd_high_f32x;
            nk_attention_pair_weights_sme_streaming_(scores_panel, pair_idx, panel_start, panel_length, panel_masked,
                                                     scale2_f32x, negated_max2_low_f32x, negated_max2_high_f32x,
                                                     key_begins_low_u32x, key_begins_high_u32x, key_ends_low_u32x,
                                                     key_ends_high_u32x, &weight_even_low_f32x, &weight_even_high_f32x,
                                                     &weight_odd_low_f32x, &weight_odd_high_f32x);
            panel_sum_low_f32x = svadd_f32_x(
                predicate_all_b32x, panel_sum_low_f32x,
                svadd_f32_x(predicate_all_b32x, weight_even_low_f32x, weight_odd_low_f32x));
            panel_sum_high_f32x = svadd_f32_x(
                predicate_all_b32x, panel_sum_high_f32x,
                svadd_f32_x(predicate_all_b32x, weight_even_high_f32x, weight_odd_high_f32x));
            svst1_u16(predicate_all_b16x, weights_panel + (pair_idx * 2 + 0) * vector_elements,
                      nk_attention_f16_pair_sme_(weight_even_low_f32x, weight_odd_low_f32x));
            svst1_u16(predicate_all_b16x, weights_panel + (pair_idx * 2 + 1) * vector_elements,
                      nk_attention_f16_pair_sme_(weight_even_high_f32x, weight_odd_high_f32x));
        }
        running_sum_low_f32x = svmad_f32_x(predicate_all_b32x, running_sum_low_f32x, correction_low_f32x,
                                           panel_sum_low_f32x);
        running_sum_high_f32x = svmad_f32_x(predicate_all_b32x, running_sum_high_f32x, correction_high_f32x,
                                            panel_sum_high_f32x);

        for (nk_size_t channel_tile_idx = 0; channel_tile_idx < channel_tiles; channel_tile_idx += 2) {
            int const has_second_tile = channel_tile_idx + 1 < channel_tiles;
            nk_u16_t const *values_tile0 = values_plane + (channel_tile_idx * position_pairs_total + panel_start / 2) *
                                                              vector_elements;
            nk_attention_values_f16_sme_streaming_(weights_panel, values_tile0,
                                                   values_tile0 + position_pairs_total * vector_elements, panel_pairs,
                                                   has_second_tile);
            nk_attention_drain_values_sme_streaming_(o_acc, channel_tile_idx, has_second_tile, correction_low_f32x,
                                                     correction_high_f32x);
        }
    }
    nk_attention_finish_block_sme_streaming_(o_acc, running_max2_low_f32x, running_max2_high_f32x, running_sum_low_f32x,
                                             running_sum_high_f32x, depth, block_rows, store_begin, store_end,
                                             value_scale, value_exponent, output, output_stride_floats, log_sum_exp,
                                             log_sum_exp_stride);
}

/**
 *  @brief Streaming attention core for the BF16-compute dtypes (raw BF16, E4M3 widened at the query
 *      loads): the whole task loop runs inside one ZA context.
 *
 *  Per head, per segment the window's query rows reach, and per query block of two row-tiles, the
 *  ZA0 transpose pair-interleaves the block's query rows once into a scratch of MOPA operand
 *  vectors, reused across every KV panel of @c nk_attention_block_bf16_sme_streaming_.
 */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_packed_b16_sme_streaming_(                      //
    void const *queries, nk_size_t element_bytes, void const *key_value_packed, nk_f32_t *output,   //
    nk_f32_t *log_sum_exp, nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,   //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2;

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // fold log2e: softmax(x) = softmax₂(x · log₂e)
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    // Scratch sized for SVL ≤ 512 (tile dimension ≤ 16) and depth ≤ 256; the entry points
    // route larger shapes to serial.
    nk_align_(64) nk_u16_t queries_packed[2][(nk_attention_max_depth_sme_k_ / 2) * 2 * nk_attention_max_tile_sme_k_];
    svbool_t const predicate_all_b32x = svptrue_b32();

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
            nk_size_t const plane_bytes = position_count_padded * depth_padded * sizeof(nk_u16_t);
            nk_size_t const key_value_head_idx = head_idx / head_group_size;
            nk_u16_t const *keys_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                            key_value_head_idx * plane_bytes);
            nk_u16_t const *values_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                              (key_value_head_count + key_value_head_idx) *
                                                                  plane_bytes);

            // Blocks keep the segment's own grid, as their panels end at the last row's keys.
            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;

                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
                    nk_size_t const rows_to_stage = (block_rows > tile_row_start)
                                                        ? ((block_rows - tile_row_start < tile_dimension)
                                                               ? block_rows - tile_row_start
                                                               : tile_dimension)
                                                        : 0;
                    for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_pairs;
                         depth_batch_start += tile_dimension) {
                        svbool_t const batch_predicate_b32x = svwhilelt_b32_u64(depth_batch_start, depth_pairs);
                        nk_size_t const batch_size = svcntp_b32(svptrue_b32(), batch_predicate_b32x);
                        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_to_stage; row_in_tile++) {
                            char const *row_ptr = (char const *)queries +
                                                  (query_first + row_block_start + tile_row_start + row_in_tile) *
                                                      query_stride +
                                                  (head_idx * depth + depth_batch_start * 2) * element_bytes;
                            svbool_t const depth_predicate_b16x = svwhilelt_b16_u64(depth_batch_start * 2, depth);
                            svuint16_t row_u16x;
                            if (element_bytes == sizeof(nk_u16_t))
                                row_u16x = svld1_u16(depth_predicate_b16x, (nk_u16_t const *)row_ptr);
                            else
                                row_u16x = nk_attention_e4m3_to_bf16_sme_(
                                    depth_predicate_b16x,
                                    svld1_u8(svwhilelt_b8_u64(depth_batch_start * 2, depth), (nk_u8_t const *)row_ptr));
                            svwrite_hor_za32_f32_m(0, (uint32_t)row_in_tile, batch_predicate_b32x,
                                                   svreinterpret_f32_u16(row_u16x));
                        }
                        for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                            svfloat32_t column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 0,
                                                                            (uint32_t)depth_step);
                            svst1_f32(predicate_all_b32x,
                                      (float32_t *)(queries_packed[row_tile_idx] +
                                                    (depth_batch_start + depth_step) * vector_elements),
                                      column_f32x);
                        }
                    }
                }

                nk_size_t const block_token = query_first + row_block_start;
                nk_attention_block_bf16_sme_streaming_(
                    queries_packed[0], queries_packed[1], keys_plane, values_plane, depth, position_count, band,
                    first_position + (nk_i64_t)row_block_start, block_rows,
                    row_begin > row_block_start ? row_begin - row_block_start : 0, row_end - row_block_start, scale2,
                    1.0f, 0, output + block_token * output_stride_floats + head_idx * depth, output_stride_floats,
                    log_sum_exp ? log_sum_exp + block_token * head_count + head_idx : NUMKONG_NULL, head_count);
            }
        }
    }
}

/** The F16 twin of @c nk_attention_packed_b16_sme_streaming_, staging raw F16 query rows for
 *  @c nk_attention_block_f16_sme_streaming_. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_packed_f16_sme_streaming_(                      //
    nk_f16_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                          //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2, staged_pairs = nk_size_divide_round_up_(depth, 2);

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_u16_t queries_packed[2][(nk_attention_max_depth_sme_k_ / 2) * 2 * nk_attention_max_tile_sme_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
            nk_size_t const plane_bytes = position_count_padded * depth_padded * sizeof(nk_u16_t);
            nk_size_t const key_value_head_idx = head_idx / head_group_size;
            nk_u16_t const *keys_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                            key_value_head_idx * plane_bytes);
            nk_u16_t const *values_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                              (key_value_head_count + key_value_head_idx) *
                                                                  plane_bytes);

            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;
                nk_size_t const block_token = query_first + row_block_start;
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
                    nk_size_t const rows_to_stage = block_rows <= tile_row_start ? 0
                                                    : block_rows - tile_row_start < tile_dimension
                                                        ? block_rows - tile_row_start
                                                        : tile_dimension;
                    nk_stage_panel_b16_sme_((nk_u16_t const *)queries, query_stride, block_token + tile_row_start,
                                            rows_to_stage, head_idx * depth, depth, queries_packed[row_tile_idx]);
                    for (nk_size_t step = staged_pairs; step < depth_pairs; step++)
                        svst1_u16(svptrue_b16(), queries_packed[row_tile_idx] + step * vector_elements, svdup_u16(0));
                }
                nk_attention_block_f16_sme_streaming_(
                    queries_packed[0], queries_packed[1], keys_plane, values_plane, depth, position_count, band,
                    first_position + (nk_i64_t)row_block_start, block_rows,
                    row_begin > row_block_start ? row_begin - row_block_start : 0, row_end - row_block_start, scale2,
                    1.0f, 0, output + block_token * output_stride_floats + head_idx * depth, output_stride_floats,
                    log_sum_exp ? log_sum_exp + block_token * head_count + head_idx : NUMKONG_NULL, head_count);
            }
        }
    }
}

/**
 *  @brief Streaming pack core for NVFP4 K/V planes: each element times its UE4M3 block scale, an
 *      exact F16, in the tile layout of @c nk_attention_pack_b16_sme_streaming_.
 *
 *  K position tiles decode through the ZA0 panel stager; V position pairs decode per channel tile
 *  and interleave through one @c ZIP1. Tensor scales stay in the header, which the caller writes.
 */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_pack_nvfp4_sme_streaming_(                           //
    nk_cross_operand_t keys, nk_cross_operand_t values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count,                   //
    nk_size_t key_stride, nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin,         //
    nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2, staged_pairs = nk_size_divide_round_up_(depth, 2);

    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count,
                vector_elements, depth_padded * sizeof(nk_u16_t));
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
        nk_size_t const plane_bytes = position_count_padded * depth_padded * sizeof(nk_u16_t);
        nk_u16_t *keys_plane = (nk_u16_t *)(payload_base + payload_offset + key_value_head_idx * plane_bytes);
        nk_u16_t *values_plane = (nk_u16_t *)(payload_base + payload_offset +
                                              (key_value_head_count + key_value_head_idx) * plane_bytes);

        // Fully padded tiles still run and store the zeros the score stage expects
        for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / tile_dimension;
             position_tile_idx++) {
            nk_size_t const position_start = position_tile_idx * tile_dimension;
            nk_size_t const rows_to_pack = position_start >= position_count ? 0
                                           : position_count - position_start < tile_dimension
                                               ? position_count - position_start
                                               : tile_dimension;
            nk_u16_t *tile_output = keys_plane + position_tile_idx * depth_pairs * vector_elements;
            nk_stage_panel_nvfp4_sme_(keys, key_stride, position_first + position_start, rows_to_pack,
                                      key_value_head_idx * depth, depth, tile_output);
            for (nk_size_t step = staged_pairs; step < depth_pairs; step++)
                svst1_u16(svptrue_b16(), tile_output + step * vector_elements, svdup_u16(0));
        }

        for (nk_size_t channel_tile_idx = 0; channel_tile_idx < depth_padded / tile_dimension; channel_tile_idx++) {
            nk_size_t const channel_start = channel_tile_idx * tile_dimension;
            nk_size_t const channels = depth - channel_start < tile_dimension ? depth - channel_start : tile_dimension;
            nk_size_t const channel_first = key_value_head_idx * depth + channel_start;
            nk_u16_t *tile_output = values_plane + channel_tile_idx * (position_count_padded / 2) * vector_elements;
            for (nk_size_t position_pair_idx = 0; position_pair_idx < position_count_padded / 2; position_pair_idx++) {
                nk_size_t const position_even = position_pair_idx * 2, position_odd = position_even + 1;
                svuint16_t even_u16x = svdup_u16(0), odd_u16x = svdup_u16(0);
                if (position_even < position_count) {
                    nk_size_t const row = position_first + position_even;
                    even_u16x = nk_decode_row_nvfp4_sme_((nk_u8_t const *)values.elements + row * value_stride,
                                                         values.scales + row * values.scales_stride, channel_first,
                                                         channels);
                }
                if (position_odd < position_count) {
                    nk_size_t const row = position_first + position_odd;
                    odd_u16x = nk_decode_row_nvfp4_sme_((nk_u8_t const *)values.elements + row * value_stride,
                                                        values.scales + row * values.scales_stride, channel_first,
                                                        channels);
                }
                svst1_u16(svptrue_b16(), tile_output + position_pair_idx * vector_elements,
                          svzip1_u16(even_u16x, odd_u16x));
            }
        }
    }
}

/** The NVFP4 twin of @c nk_attention_packed_f16_sme_streaming_: query rows stage times their block
 *  scales as exact F16, scores multiply by @p scale2 with both tensor scales folded in, and the
 *  normalization carries the @p value_scale tensor scale. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_packed_nvfp4_sme_streaming_(                       //
    nk_cross_operand_t queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                             //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale2,   //
    nk_f32_t value_scale, nk_diagonal_band_t band, nk_size_t tasks_begin,                              //
    nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2, staged_pairs = nk_size_divide_round_up_(depth, 2);

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 16 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_u16_t queries_packed[2][(nk_attention_max_depth_sme_k_ / 2) * 2 * nk_attention_max_tile_sme_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
            nk_size_t const plane_bytes = position_count_padded * depth_padded * sizeof(nk_u16_t);
            nk_size_t const key_value_head_idx = head_idx / head_group_size;
            nk_u16_t const *keys_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                            key_value_head_idx * plane_bytes);
            nk_u16_t const *values_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                              (key_value_head_count + key_value_head_idx) *
                                                                  plane_bytes);

            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;
                nk_size_t const block_token = query_first + row_block_start;
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
                    nk_size_t const rows_to_stage = block_rows <= tile_row_start ? 0
                                                    : block_rows - tile_row_start < tile_dimension
                                                        ? block_rows - tile_row_start
                                                        : tile_dimension;
                    nk_stage_panel_nvfp4_sme_(queries, query_stride, block_token + tile_row_start, rows_to_stage,
                                              head_idx * depth, depth, queries_packed[row_tile_idx]);
                    for (nk_size_t step = staged_pairs; step < depth_pairs; step++)
                        svst1_u16(svptrue_b16(), queries_packed[row_tile_idx] + step * vector_elements, svdup_u16(0));
                }
                nk_attention_block_f16_sme_streaming_(
                    queries_packed[0], queries_packed[1], keys_plane, values_plane, depth, position_count, band,
                    first_position + (nk_i64_t)row_block_start, block_rows,
                    row_begin > row_block_start ? row_begin - row_block_start : 0, row_end - row_block_start, scale2,
                    value_scale, 0, output + block_token * output_stride_floats + head_idx * depth,
                    output_stride_floats,
                    log_sum_exp ? log_sum_exp + block_token * head_count + head_idx : NUMKONG_NULL, head_count);
            }
        }
    }
}

/** BF16 bits of @p count E2M1 elements from dim @p first of one row of @p codes, moved by
 *  @p exponent_shift binades either way: the shift adds to the nonzero table entries, where the
 *  saturating decrement of @c nk_load_mx_e2m1_sme_ would zero the blocks a plane base raises. */
NUMKONG_INLINE svuint16_t nk_attention_load_mx_e2m1_sme_(nk_u8_t const *codes, nk_size_t first, nk_size_t count,
                                                         nk_i32_t exponent_shift) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b8x = svptrue_b8(), predicate_all_b16x = svptrue_b16();
    nk_u16_t const shift_bits = (nk_u16_t)(exponent_shift * 128);
    svuint16x2_t const table_u16x = nk_table16_u16_sme_(nk_mxfp4_table_data_sme_);
    svuint16_t const first_entries_u16x = svget2_u16(table_u16x, 0);
    svuint16_t const second_entries_u16x = svget2_u16(table_u16x, 1);
    svuint16x2_t const shifted_u16x = svcreate2_u16(
        svadd_n_u16_m(svcmpne_n_u16(predicate_all_b16x, first_entries_u16x, 0), first_entries_u16x, shift_bits),
        svadd_n_u16_m(svcmpne_n_u16(predicate_all_b16x, second_entries_u16x, 0), second_entries_u16x, shift_bits));
    svuint8_t const pairs_u8x = svld1_u8(svwhilelt_b8_u64(0, count / 2), codes + first / 2);
    svuint8_t const nibbles_u8x = svzip1_u8(svlsr_n_u8_x(predicate_all_b8x, pairs_u8x, 4),
                                            svand_n_u8_x(predicate_all_b8x, pairs_u8x, 0x0F));
    return svtbl2_u16(shifted_u16x, svunpklo_u16(nibbles_u8x));
}

/** MXFP4 dims as BF16 rebased by @p base, as @c nk_decode_row_mxfp4_sme_ decodes them through
 *  @c nk_attention_load_mx_e2m1_sme_. */
NUMKONG_INLINE svuint16_t nk_attention_decode_row_mxfp4_sme_(nk_u8_t const *codes, nk_u8_t const *scales,
                                                             nk_size_t first, nk_size_t count,
                                                             nk_i32_t base) NUMKONG_STREAMING_ {
    svuint16_t row_u16x = svdup_n_u16(0);
    for (nk_size_t piece = 0; piece < count; piece += 32) {
        nk_size_t const piece_count = count - piece < 32 ? count - piece : 32;
        nk_u8_t const scale = scales[(first + piece) / 32];
        svuint16_t const piece_u16x = nk_mx_block_sme_(
            nk_attention_load_mx_e2m1_sme_(codes, first + piece, piece_count, nk_mx_shift_sme_(scale, base)), scale,
            piece_count);
        row_u16x = svsplice_u16(svwhilelt_b16_u64(0, piece), row_u16x, piece_u16x);
    }
    return row_u16x;
}

/** Stages MXFP4 rows as @c nk_stage_panel_mxfp4_sme_ does, decoding through
 *  @c nk_attention_decode_row_mxfp4_sme_. */
NUMKONG_OUTLINED_ void nk_attention_stage_panel_mxfp4_sme_(nk_cross_operand_t operand, nk_size_t row_stride,
                                                           nk_size_t row_first, nk_size_t rows, nk_size_t depth_first,
                                                           nk_size_t depth, nk_i32_t const *bases,
                                                           nk_u16_t *panel) NUMKONG_STREAMING_ __arm_inout("za") {
    nk_size_t const vector_elements = svcnth();
    svbool_t const predicate_all_b32x = svptrue_b32();
    for (nk_size_t dims = 0; dims < depth; dims += vector_elements) {
        nk_size_t const count = depth - dims < vector_elements ? depth - dims : vector_elements;
        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
        for (nk_size_t row = 0; row < rows; ++row) {
            nk_size_t const row_index = row_first + row;
            svuint16_t const row_u16x = nk_attention_decode_row_mxfp4_sme_(
                (nk_u8_t const *)operand.elements + row_index * row_stride,
                operand.scales + row_index * operand.scales_stride, depth_first + dims, count, bases[row]);
            svwrite_hor_za32_u32_m(0, (uint32_t)row, predicate_all_b32x, svreinterpret_u32_u16(row_u16x));
        }
        nk_store_panel_steps_sme_(panel, dims / 2, nk_size_divide_round_up_(count, 2));
    }
}

/** Writes @p positions_padded rows of a raw MX plane, each @p row_bytes long, as
 *  @c nk_attention_raw_row_serial_ fills them: each of the first @p positions rows takes
 *  @p codes_bytes codes from @p codes, then @p blocks scale codes from @p scales, sources
 *  @p codes_stride and @p scales_stride bytes apart, and every other byte is zero. */
NUMKONG_INLINE void nk_attention_raw_plane_sme_streaming_(nk_u8_t const *codes, nk_size_t codes_stride,
                                                          nk_u8_t const *scales, nk_size_t scales_stride,
                                                          nk_size_t codes_bytes, nk_size_t blocks, nk_size_t positions,
                                                          nk_size_t positions_padded, nk_size_t row_bytes,
                                                          nk_u8_t *plane) NUMKONG_STREAMING_ {
    nk_size_t const vector_bytes = svcntb();
    for (nk_size_t position_idx = 0; position_idx < positions_padded; position_idx++) {
        nk_u8_t *row = plane + position_idx * row_bytes;
        for (nk_size_t byte_index = 0; byte_index < row_bytes; byte_index += vector_bytes)
            svst1_u8(svwhilelt_b8_u64(byte_index, row_bytes), row + byte_index, svdup_n_u8(0));
        if (position_idx >= positions) continue;
        for (nk_size_t byte_index = 0; byte_index < codes_bytes; byte_index += vector_bytes) {
            svbool_t const predicate_b8x = svwhilelt_b8_u64(byte_index, codes_bytes);
            svst1_u8(predicate_b8x, row + byte_index,
                     svld1_u8(predicate_b8x, codes + position_idx * codes_stride + byte_index));
        }
        for (nk_size_t block = 0; block < blocks; block += vector_bytes) {
            svbool_t const predicate_b8x = svwhilelt_b8_u64(block, blocks);
            svst1_u8(predicate_b8x, row + codes_bytes + block,
                     svld1_u8(predicate_b8x, scales + position_idx * scales_stride + block));
        }
    }
}

/** The base-2 score multiplier of @p block_rows MX query rows, of @p blocks UE8M0 scale codes each,
 *  @p scales_stride bytes apart, over planes at @p key_exponent and @p value_exponent: @p scale2
 *  times 2 to the block's base, stored to @p query_base, plus the key plane's. Zero sends the block
 *  to the exact path, for a raw plane, a block wider than the window or an abnormal multiplier. */
NUMKONG_INLINE nk_f32_t nk_attention_block_scale2_sme_(nk_u8_t const *scales, nk_size_t scales_stride,
                                                       nk_size_t block_rows, nk_size_t blocks, nk_i8_t key_exponent,
                                                       nk_i8_t value_exponent, nk_f32_t scale2,
                                                       nk_i32_t *query_base) NUMKONG_STREAMABLE_ {
    if (key_exponent == nk_attention_raw_plane_k_ || value_exponent == nk_attention_raw_plane_k_) return 0;
    nk_i8_t const query_exponent = nk_attention_plane_exponent_ue8m0_serial_(scales, scales_stride, block_rows, blocks);
    if (query_exponent == nk_attention_raw_plane_k_) return 0;
    *query_base = nk_attention_plane_base_serial_(query_exponent);
    nk_fui32_t multiplier;
    multiplier.f = nk_scale_f32_serial_(scale2, *query_base + nk_attention_plane_base_serial_(key_exponent));
    nk_u32_t const biased_exponent = (multiplier.u >> 23) & 0xFF;
    return biased_exponent != 0 && biased_exponent != 0xFF ? multiplier.f : 0;
}

/** Row @p position of an MX K plane at @p key_exponent, as the serial score helpers read it: raw
 *  rows in place, tile rows of @c nk_attention_pack_b16_sme_streaming_ widened into @p scratch as
 *  F32. Runs outside streaming mode, so @p tile_dimension stands in for `svcntw()`. */
NUMKONG_INLINE nk_u8_t const *nk_attention_key_row_sme_(char const *keys_plane, nk_i8_t key_exponent,
                                                        nk_size_t tile_dimension, nk_size_t depth, nk_size_t position,
                                                        nk_f32_t *scratch) {
    if (key_exponent == nk_attention_raw_plane_k_)
        return (nk_u8_t const *)keys_plane + position * depth * sizeof(nk_bf16_t);
    nk_size_t const vector_elements = 2 * tile_dimension;
    nk_bf16_t const *pairs = (nk_bf16_t const *)keys_plane + (position / tile_dimension * depth / 2) * vector_elements +
                             position % tile_dimension * 2;
    for (nk_size_t dimension = 0; dimension < depth; dimension++)
        nk_bf16_to_f32_serial_(pairs + dimension / 2 * vector_elements + dimension % 2, scratch + dimension);
    return (nk_u8_t const *)scratch;
}

/** Row @p position of an MX V plane at @p value_exponent, as the serial plane helpers read it, like
 *  @c nk_attention_key_row_sme_: tile rows come from the transposed layout of
 *  @c nk_attention_pack_b16_sme_streaming_, @p position_pairs pairs per channel tile. */
NUMKONG_INLINE nk_u8_t const *nk_attention_value_row_sme_(char const *values_plane, nk_i8_t value_exponent,
                                                          nk_size_t tile_dimension, nk_size_t depth,
                                                          nk_size_t position_pairs, nk_size_t position,
                                                          nk_f32_t *scratch) {
    if (value_exponent == nk_attention_raw_plane_k_)
        return (nk_u8_t const *)values_plane + position * depth * sizeof(nk_bf16_t);
    nk_size_t const vector_elements = 2 * tile_dimension;
    nk_bf16_t const *pairs = (nk_bf16_t const *)values_plane + position / 2 * vector_elements + position % 2;
    for (nk_size_t channel = 0; channel < depth; channel++)
        nk_bf16_to_f32_serial_(
            pairs + channel / tile_dimension * position_pairs * vector_elements + channel % tile_dimension * 2,
            scratch + channel);
    return (nk_u8_t const *)scratch;
}

/**
 *  @brief Streaming pack core for MXFP4 K/V planes: elements times their block scales under the
 *      plane's base as exact BF16, in the tile layout of @c nk_attention_pack_b16_sme_streaming_,
 *      and each plane's entry in the plane-exponent table.
 *
 *  K position tiles stage through ZA0 and V position pairs decode per channel tile, as for NVFP4.
 *  Raw planes keep serial's rows of codes, then scale codes, which the exact path reads in place.
 */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_pack_mxfp4_sme_streaming_(                           //
    nk_cross_operand_t keys, nk_cross_operand_t values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count,                   //
    nk_size_t key_stride, nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin,         //
    nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2, row_bytes = depth_padded * sizeof(nk_u16_t);
    nk_size_t const blocks = depth / 32, codes_bytes = depth / 2;

    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t *plane_exponents = (nk_i8_t *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                 key_value_head_count, vector_elements, row_bytes,
                                                                 key_offsets, key_lengths, segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const plane_exponents_size = nk_attention_plane_exponents_size_serial_(key_value_head_count,
                                                                                     segment_count);
    if (tasks_begin == 0)
        for (nk_size_t entry = 2 * total_tasks; entry < plane_exponents_size; entry++) plane_exponents[entry] = 0;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_i32_t bases[nk_attention_max_tile_sme_k_];
    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count,
                vector_elements, row_bytes);
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
        nk_size_t const plane_bytes = position_count_padded * row_bytes;
        nk_u16_t *keys_plane = (nk_u16_t *)(payload_base + payload_offset + key_value_head_idx * plane_bytes);
        nk_u16_t *values_plane = (nk_u16_t *)(payload_base + payload_offset +
                                              (key_value_head_count + key_value_head_idx) * plane_bytes);
        nk_u8_t const *key_scales = keys.scales + position_first * keys.scales_stride + key_value_head_idx * blocks;
        nk_u8_t const *value_scales = values.scales + position_first * values.scales_stride +
                                      key_value_head_idx * blocks;
        nk_i8_t const key_exponent = nk_attention_plane_exponent_ue8m0_serial_(key_scales, keys.scales_stride,
                                                                               position_count, blocks);
        nk_i8_t const value_exponent = nk_attention_plane_exponent_ue8m0_serial_(value_scales, values.scales_stride,
                                                                                 position_count, blocks);
        plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx] = key_exponent;
        plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx] = value_exponent;

        if (key_exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_plane_sme_streaming_(
                (nk_u8_t const *)keys.elements + position_first * key_stride + key_value_head_idx * codes_bytes,
                key_stride, key_scales, keys.scales_stride, codes_bytes, blocks, position_count, position_count_padded,
                row_bytes, (nk_u8_t *)keys_plane);
        else {
            nk_i32_t const key_base = nk_attention_plane_base_serial_(key_exponent);
            for (nk_size_t lane_idx = 0; lane_idx < tile_dimension; lane_idx++) bases[lane_idx] = key_base;
            // Fully padded tiles still run and store the zeros the score stage expects
            for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / tile_dimension;
                 position_tile_idx++) {
                nk_size_t const position_start = position_tile_idx * tile_dimension;
                nk_size_t const rows_to_pack = position_start >= position_count ? 0
                                               : position_count - position_start < tile_dimension
                                                   ? position_count - position_start
                                                   : tile_dimension;
                nk_attention_stage_panel_mxfp4_sme_(keys, key_stride, position_first + position_start, rows_to_pack,
                                                    key_value_head_idx * depth, depth, bases,
                                                    keys_plane + position_tile_idx * depth_pairs * vector_elements);
            }
        }

        if (value_exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_plane_sme_streaming_(
                (nk_u8_t const *)values.elements + position_first * value_stride + key_value_head_idx * codes_bytes,
                value_stride, value_scales, values.scales_stride, codes_bytes, blocks, position_count,
                position_count_padded, row_bytes, (nk_u8_t *)values_plane);
        else {
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);
            for (nk_size_t channel_tile_idx = 0; channel_tile_idx < depth_padded / tile_dimension; channel_tile_idx++) {
                nk_size_t const channel_start = channel_tile_idx * tile_dimension;
                nk_size_t const channels = depth - channel_start < tile_dimension ? depth - channel_start
                                                                                  : tile_dimension;
                nk_size_t const channel_first = key_value_head_idx * depth + channel_start;
                nk_u16_t *tile_output = values_plane + channel_tile_idx * (position_count_padded / 2) * vector_elements;
                for (nk_size_t position_pair_idx = 0; position_pair_idx < position_count_padded / 2;
                     position_pair_idx++) {
                    nk_size_t const position_even = position_pair_idx * 2, position_odd = position_even + 1;
                    svuint16_t even_u16x = svdup_u16(0), odd_u16x = svdup_u16(0);
                    if (position_even < position_count) {
                        nk_size_t const row = position_first + position_even;
                        even_u16x = nk_attention_decode_row_mxfp4_sme_(
                            (nk_u8_t const *)values.elements + row * value_stride,
                            values.scales + row * values.scales_stride, channel_first, channels, value_base);
                    }
                    if (position_odd < position_count) {
                        nk_size_t const row = position_first + position_odd;
                        odd_u16x = nk_attention_decode_row_mxfp4_sme_(
                            (nk_u8_t const *)values.elements + row * value_stride,
                            values.scales + row * values.scales_stride, channel_first, channels, value_base);
                    }
                    svst1_u16(svptrue_b16(), tile_output + position_pair_idx * vector_elements,
                              svzip1_u16(even_u16x, odd_u16x));
                }
            }
        }
    }
}

/** The MXFP4 twin of @c nk_attention_packed_nvfp4_sme_streaming_: each query block stages as BF16
 *  under its own base, and @c nk_attention_block_scale2_sme_ folds both bases into the score
 *  multiplier, or leaves the block to @c nk_attention_packed_exact_mxfp4_sme_; the value base
 *  scales the normalization. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_packed_mxfp4_sme_streaming_(                       //
    nk_cross_operand_t queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                             //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale2,   //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const row_bytes = depth_padded * sizeof(nk_u16_t), blocks = depth / 32;

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, vector_elements,
                                                                                 row_bytes, key_offsets, key_lengths,
                                                                                 segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_u16_t queries_packed[2][(nk_attention_max_depth_sme_k_ / 2) * 2 * nk_attention_max_tile_sme_k_];
    nk_i32_t bases[nk_attention_max_tile_sme_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
            nk_size_t const plane_bytes = position_count_padded * row_bytes;
            nk_u16_t const *keys_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                            key_value_head_idx * plane_bytes);
            nk_u16_t const *values_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                              (key_value_head_count + key_value_head_idx) *
                                                                  plane_bytes);
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;
                nk_size_t const block_token = query_first + row_block_start;
                nk_i32_t query_base = 0;
                nk_f32_t const block_scale2 = nk_attention_block_scale2_sme_(
                    queries.scales + block_token * queries.scales_stride + head_idx * blocks, queries.scales_stride,
                    block_rows, blocks, key_exponent, value_exponent, scale2, &query_base);
                if (block_scale2 == 0) continue;
                for (nk_size_t lane_idx = 0; lane_idx < tile_dimension; lane_idx++) bases[lane_idx] = query_base;
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
                    nk_size_t const rows_to_stage = block_rows <= tile_row_start ? 0
                                                    : block_rows - tile_row_start < tile_dimension
                                                        ? block_rows - tile_row_start
                                                        : tile_dimension;
                    nk_attention_stage_panel_mxfp4_sme_(queries, query_stride, block_token + tile_row_start,
                                                        rows_to_stage, head_idx * depth, depth, bases,
                                                        queries_packed[row_tile_idx]);
                }
                nk_attention_block_bf16_sme_streaming_(
                    queries_packed[0], queries_packed[1], keys_plane, values_plane, depth, position_count, band,
                    first_position + (nk_i64_t)row_block_start, block_rows,
                    row_begin > row_block_start ? row_begin - row_block_start : 0, row_end - row_block_start,
                    block_scale2, 1.0f, value_base, output + block_token * output_stride_floats + head_idx * depth,
                    output_stride_floats,
                    log_sum_exp ? log_sum_exp + block_token * head_count + head_idx : NUMKONG_NULL, head_count);
            }
        }
    }
}

/**
 *  @brief Recomputes outside streaming mode the MXFP4 query blocks that
 *      @c nk_attention_block_scale2_sme_ sends to the exact path, each row as
 *      @c nk_attention_packed_mxfp4_serial_ computes it.
 *
 *  Raw planes serve their rows in place and tile planes through @c nk_attention_key_row_sme_ and
 *  @c nk_attention_value_row_sme_, so the serial score and plane helpers read serial's rows.
 */
NUMKONG_OUTLINED_ void nk_attention_packed_exact_mxfp4_sme_(                                           //
    nk_cross_operand_t queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                             //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale2,   //
    nk_diagonal_band_t band, nk_size_t tile_dimension, nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const row_bytes = depth * sizeof(nk_bf16_t), blocks = depth / 32;
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                             key_value_head_count, block_rows_capacity,
                                                                             row_bytes, key_offsets, key_lengths,
                                                                             segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_f32_t key_row[nk_attention_max_depth_sme_k_];
    nk_align_(64) nk_f32_t value_row[nk_attention_max_depth_sme_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, block_rows_capacity);
            nk_size_t const plane_bytes = position_count_padded * row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] + key_value_head_idx * plane_bytes;
            char const *values_plane = payload_base + payload_offsets[segment_idx] +
                                       (key_value_head_count + key_value_head_idx) * plane_bytes;
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;
                nk_i32_t block_base = 0;
                if (nk_attention_block_scale2_sme_(
                        queries.scales + (query_first + row_block_start) * queries.scales_stride + head_idx * blocks,
                        queries.scales_stride, block_rows, blocks, key_exponent, value_exponent, scale2,
                        &block_base) != 0)
                    continue;
                nk_size_t const block_end = row_block_start + block_rows < row_end ? row_block_start + block_rows
                                                                                   : row_end;
                for (nk_size_t row = row_begin > row_block_start ? row_begin : row_block_start; row < block_end;
                     row++) {
                    nk_size_t const token = query_first + row;
                    nk_u8_t const *query_row = (nk_u8_t const *)queries.elements + token * query_stride +
                                               head_idx * depth / 2;
                    nk_u8_t const *query_row_scales = queries.scales + token * queries.scales_stride +
                                                      head_idx * blocks;
                    nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                    nk_size_t key_begin, key_end;
                    nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row, position_count, &key_begin,
                                                &key_end);

                    nk_f32_t max2 = NUMKONG_F32_MIN;
                    for (nk_size_t position = key_begin; position < key_end; position++) {
                        nk_f32_t const scaled2 = nk_attention_score_mxfp4_serial_(
                            query_row, query_row_scales,
                            nk_attention_key_row_sme_(keys_plane, key_exponent, tile_dimension, depth, position,
                                                      key_row),
                            key_exponent, depth, scale2);
                        if (scaled2 > max2) max2 = scaled2;
                    }
                    for (nk_size_t channel = 0; channel < depth; channel++) output_row[channel] = 0;
                    nk_f32_t weights_sum = 0;
                    for (nk_size_t position = key_begin; position < key_end; position++) {
                        nk_f32_t const weight = nk_f32_exp2_serial_(
                            nk_attention_score_mxfp4_serial_(
                                query_row, query_row_scales,
                                nk_attention_key_row_sme_(keys_plane, key_exponent, tile_dimension, depth, position,
                                                          key_row),
                                key_exponent, depth, scale2) -
                            max2);
                        nk_u8_t const *value_bytes = nk_attention_value_row_sme_(
                            values_plane, value_exponent, tile_dimension, depth, position_count_padded / 2, position,
                            value_row);
                        weights_sum += weight;
                        for (nk_size_t channel = 0; channel < depth; channel++)
                            output_row[channel] += weight * nk_attention_plane_mxfp4_serial_(
                                                                value_bytes, value_exponent, depth, channel);
                    }
                    nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                    for (nk_size_t channel = 0; channel < depth; channel++)
                        output_row[channel] = nk_scale_f32_serial_(output_row[channel] * inverse_sum, value_base);
                    if (log_sum_exp)
                        log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(max2,
                                                                                                      weights_sum);
                }
            }
        }
    }
}

/** @c nk_attention_pack_mxfp4_sme_streaming_ for MXFP6 E2M3, over the dots stager and decoder. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_pack_mxfp6e2m3_sme_streaming_(                       //
    nk_cross_operand_t keys, nk_cross_operand_t values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count,                   //
    nk_size_t key_stride, nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin,         //
    nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2, row_bytes = depth_padded * sizeof(nk_u16_t);
    nk_size_t const blocks = depth / 32, codes_bytes = depth;

    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t *plane_exponents = (nk_i8_t *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                 key_value_head_count, vector_elements, row_bytes,
                                                                 key_offsets, key_lengths, segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const plane_exponents_size = nk_attention_plane_exponents_size_serial_(key_value_head_count,
                                                                                     segment_count);
    if (tasks_begin == 0)
        for (nk_size_t entry = 2 * total_tasks; entry < plane_exponents_size; entry++) plane_exponents[entry] = 0;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_i32_t bases[nk_attention_max_tile_sme_k_];
    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count,
                vector_elements, row_bytes);
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
        nk_size_t const plane_bytes = position_count_padded * row_bytes;
        nk_u16_t *keys_plane = (nk_u16_t *)(payload_base + payload_offset + key_value_head_idx * plane_bytes);
        nk_u16_t *values_plane = (nk_u16_t *)(payload_base + payload_offset +
                                              (key_value_head_count + key_value_head_idx) * plane_bytes);
        nk_u8_t const *key_scales = keys.scales + position_first * keys.scales_stride + key_value_head_idx * blocks;
        nk_u8_t const *value_scales = values.scales + position_first * values.scales_stride +
                                      key_value_head_idx * blocks;
        nk_i8_t const key_exponent = nk_attention_plane_exponent_ue8m0_serial_(key_scales, keys.scales_stride,
                                                                               position_count, blocks);
        nk_i8_t const value_exponent = nk_attention_plane_exponent_ue8m0_serial_(value_scales, values.scales_stride,
                                                                                 position_count, blocks);
        plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx] = key_exponent;
        plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx] = value_exponent;

        if (key_exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_plane_sme_streaming_(
                (nk_u8_t const *)keys.elements + position_first * key_stride + key_value_head_idx * codes_bytes,
                key_stride, key_scales, keys.scales_stride, codes_bytes, blocks, position_count, position_count_padded,
                row_bytes, (nk_u8_t *)keys_plane);
        else {
            nk_i32_t const key_base = nk_attention_plane_base_serial_(key_exponent);
            for (nk_size_t lane_idx = 0; lane_idx < tile_dimension; lane_idx++) bases[lane_idx] = key_base;
            // Fully padded tiles still run and store the zeros the score stage expects
            for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / tile_dimension;
                 position_tile_idx++) {
                nk_size_t const position_start = position_tile_idx * tile_dimension;
                nk_size_t const rows_to_pack = position_start >= position_count ? 0
                                               : position_count - position_start < tile_dimension
                                                   ? position_count - position_start
                                                   : tile_dimension;
                nk_stage_panel_mxfp6e2m3_sme_(keys, key_stride, position_first + position_start, rows_to_pack,
                                              key_value_head_idx * depth, depth, bases,
                                              keys_plane + position_tile_idx * depth_pairs * vector_elements);
            }
        }

        if (value_exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_plane_sme_streaming_(
                (nk_u8_t const *)values.elements + position_first * value_stride + key_value_head_idx * codes_bytes,
                value_stride, value_scales, values.scales_stride, codes_bytes, blocks, position_count,
                position_count_padded, row_bytes, (nk_u8_t *)values_plane);
        else {
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);
            for (nk_size_t channel_tile_idx = 0; channel_tile_idx < depth_padded / tile_dimension; channel_tile_idx++) {
                nk_size_t const channel_start = channel_tile_idx * tile_dimension;
                nk_size_t const channels = depth - channel_start < tile_dimension ? depth - channel_start
                                                                                  : tile_dimension;
                nk_size_t const channel_first = key_value_head_idx * depth + channel_start;
                nk_u16_t *tile_output = values_plane + channel_tile_idx * (position_count_padded / 2) * vector_elements;
                for (nk_size_t position_pair_idx = 0; position_pair_idx < position_count_padded / 2;
                     position_pair_idx++) {
                    nk_size_t const position_even = position_pair_idx * 2, position_odd = position_even + 1;
                    svuint16_t even_u16x = svdup_u16(0), odd_u16x = svdup_u16(0);
                    if (position_even < position_count) {
                        nk_size_t const row = position_first + position_even;
                        even_u16x = nk_decode_row_mxfp6e2m3_sme_((nk_u8_t const *)values.elements + row * value_stride,
                                                                 values.scales + row * values.scales_stride,
                                                                 channel_first, channels, value_base);
                    }
                    if (position_odd < position_count) {
                        nk_size_t const row = position_first + position_odd;
                        odd_u16x = nk_decode_row_mxfp6e2m3_sme_((nk_u8_t const *)values.elements + row * value_stride,
                                                                values.scales + row * values.scales_stride,
                                                                channel_first, channels, value_base);
                    }
                    svst1_u16(svptrue_b16(), tile_output + position_pair_idx * vector_elements,
                              svzip1_u16(even_u16x, odd_u16x));
                }
            }
        }
    }
}

/** @c nk_attention_packed_mxfp4_sme_streaming_ for MXFP6 E2M3, over the dots stager. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_packed_mxfp6e2m3_sme_streaming_(                   //
    nk_cross_operand_t queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                             //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale2,   //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const row_bytes = depth_padded * sizeof(nk_u16_t), blocks = depth / 32;

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, vector_elements,
                                                                                 row_bytes, key_offsets, key_lengths,
                                                                                 segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_u16_t queries_packed[2][(nk_attention_max_depth_sme_k_ / 2) * 2 * nk_attention_max_tile_sme_k_];
    nk_i32_t bases[nk_attention_max_tile_sme_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
            nk_size_t const plane_bytes = position_count_padded * row_bytes;
            nk_u16_t const *keys_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                            key_value_head_idx * plane_bytes);
            nk_u16_t const *values_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                              (key_value_head_count + key_value_head_idx) *
                                                                  plane_bytes);
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;
                nk_size_t const block_token = query_first + row_block_start;
                nk_i32_t query_base = 0;
                nk_f32_t const block_scale2 = nk_attention_block_scale2_sme_(
                    queries.scales + block_token * queries.scales_stride + head_idx * blocks, queries.scales_stride,
                    block_rows, blocks, key_exponent, value_exponent, scale2, &query_base);
                if (block_scale2 == 0) continue;
                for (nk_size_t lane_idx = 0; lane_idx < tile_dimension; lane_idx++) bases[lane_idx] = query_base;
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
                    nk_size_t const rows_to_stage = block_rows <= tile_row_start ? 0
                                                    : block_rows - tile_row_start < tile_dimension
                                                        ? block_rows - tile_row_start
                                                        : tile_dimension;
                    nk_stage_panel_mxfp6e2m3_sme_(queries, query_stride, block_token + tile_row_start, rows_to_stage,
                                                  head_idx * depth, depth, bases, queries_packed[row_tile_idx]);
                }
                nk_attention_block_bf16_sme_streaming_(
                    queries_packed[0], queries_packed[1], keys_plane, values_plane, depth, position_count, band,
                    first_position + (nk_i64_t)row_block_start, block_rows,
                    row_begin > row_block_start ? row_begin - row_block_start : 0, row_end - row_block_start,
                    block_scale2, 1.0f, value_base, output + block_token * output_stride_floats + head_idx * depth,
                    output_stride_floats,
                    log_sum_exp ? log_sum_exp + block_token * head_count + head_idx : NUMKONG_NULL, head_count);
            }
        }
    }
}

/** @c nk_attention_packed_exact_mxfp4_sme_ for MXFP6 E2M3. */
NUMKONG_OUTLINED_ void nk_attention_packed_exact_mxfp6e2m3_sme_(                                       //
    nk_cross_operand_t queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                             //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale2,   //
    nk_diagonal_band_t band, nk_size_t tile_dimension, nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const row_bytes = depth * sizeof(nk_bf16_t), blocks = depth / 32;
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                             key_value_head_count, block_rows_capacity,
                                                                             row_bytes, key_offsets, key_lengths,
                                                                             segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_f32_t key_row[nk_attention_max_depth_sme_k_];
    nk_align_(64) nk_f32_t value_row[nk_attention_max_depth_sme_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, block_rows_capacity);
            nk_size_t const plane_bytes = position_count_padded * row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] + key_value_head_idx * plane_bytes;
            char const *values_plane = payload_base + payload_offsets[segment_idx] +
                                       (key_value_head_count + key_value_head_idx) * plane_bytes;
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;
                nk_i32_t block_base = 0;
                if (nk_attention_block_scale2_sme_(
                        queries.scales + (query_first + row_block_start) * queries.scales_stride + head_idx * blocks,
                        queries.scales_stride, block_rows, blocks, key_exponent, value_exponent, scale2,
                        &block_base) != 0)
                    continue;
                nk_size_t const block_end = row_block_start + block_rows < row_end ? row_block_start + block_rows
                                                                                   : row_end;
                for (nk_size_t row = row_begin > row_block_start ? row_begin : row_block_start; row < block_end;
                     row++) {
                    nk_size_t const token = query_first + row;
                    nk_u8_t const *query_row = (nk_u8_t const *)queries.elements + token * query_stride +
                                               head_idx * depth;
                    nk_u8_t const *query_row_scales = queries.scales + token * queries.scales_stride +
                                                      head_idx * blocks;
                    nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                    nk_size_t key_begin, key_end;
                    nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row, position_count, &key_begin,
                                                &key_end);

                    nk_f32_t max2 = NUMKONG_F32_MIN;
                    for (nk_size_t position = key_begin; position < key_end; position++) {
                        nk_f32_t const scaled2 = nk_attention_score_mxfp6e2m3_serial_(
                            query_row, query_row_scales,
                            nk_attention_key_row_sme_(keys_plane, key_exponent, tile_dimension, depth, position,
                                                      key_row),
                            key_exponent, depth, scale2);
                        if (scaled2 > max2) max2 = scaled2;
                    }
                    for (nk_size_t channel = 0; channel < depth; channel++) output_row[channel] = 0;
                    nk_f32_t weights_sum = 0;
                    for (nk_size_t position = key_begin; position < key_end; position++) {
                        nk_f32_t const weight = nk_f32_exp2_serial_(
                            nk_attention_score_mxfp6e2m3_serial_(
                                query_row, query_row_scales,
                                nk_attention_key_row_sme_(keys_plane, key_exponent, tile_dimension, depth, position,
                                                          key_row),
                                key_exponent, depth, scale2) -
                            max2);
                        nk_u8_t const *value_bytes = nk_attention_value_row_sme_(
                            values_plane, value_exponent, tile_dimension, depth, position_count_padded / 2, position,
                            value_row);
                        weights_sum += weight;
                        for (nk_size_t channel = 0; channel < depth; channel++)
                            output_row[channel] += weight * nk_attention_plane_mxfp6e2m3_serial_(
                                                                value_bytes, value_exponent, depth, channel);
                    }
                    nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                    for (nk_size_t channel = 0; channel < depth; channel++)
                        output_row[channel] = nk_scale_f32_serial_(output_row[channel] * inverse_sum, value_base);
                    if (log_sum_exp)
                        log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(max2,
                                                                                                      weights_sum);
                }
            }
        }
    }
}

/** @c nk_attention_pack_mxfp4_sme_streaming_ for MXFP6 E3M2, over the dots stager and decoder. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_pack_mxfp6e3m2_sme_streaming_(                       //
    nk_cross_operand_t keys, nk_cross_operand_t values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count,                   //
    nk_size_t key_stride, nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin,         //
    nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2, row_bytes = depth_padded * sizeof(nk_u16_t);
    nk_size_t const blocks = depth / 32, codes_bytes = depth;

    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t *plane_exponents = (nk_i8_t *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                 key_value_head_count, vector_elements, row_bytes,
                                                                 key_offsets, key_lengths, segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const plane_exponents_size = nk_attention_plane_exponents_size_serial_(key_value_head_count,
                                                                                     segment_count);
    if (tasks_begin == 0)
        for (nk_size_t entry = 2 * total_tasks; entry < plane_exponents_size; entry++) plane_exponents[entry] = 0;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_i32_t bases[nk_attention_max_tile_sme_k_];
    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count,
                vector_elements, row_bytes);
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
        nk_size_t const plane_bytes = position_count_padded * row_bytes;
        nk_u16_t *keys_plane = (nk_u16_t *)(payload_base + payload_offset + key_value_head_idx * plane_bytes);
        nk_u16_t *values_plane = (nk_u16_t *)(payload_base + payload_offset +
                                              (key_value_head_count + key_value_head_idx) * plane_bytes);
        nk_u8_t const *key_scales = keys.scales + position_first * keys.scales_stride + key_value_head_idx * blocks;
        nk_u8_t const *value_scales = values.scales + position_first * values.scales_stride +
                                      key_value_head_idx * blocks;
        nk_i8_t const key_exponent = nk_attention_plane_exponent_ue8m0_serial_(key_scales, keys.scales_stride,
                                                                               position_count, blocks);
        nk_i8_t const value_exponent = nk_attention_plane_exponent_ue8m0_serial_(value_scales, values.scales_stride,
                                                                                 position_count, blocks);
        plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx] = key_exponent;
        plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx] = value_exponent;

        if (key_exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_plane_sme_streaming_(
                (nk_u8_t const *)keys.elements + position_first * key_stride + key_value_head_idx * codes_bytes,
                key_stride, key_scales, keys.scales_stride, codes_bytes, blocks, position_count, position_count_padded,
                row_bytes, (nk_u8_t *)keys_plane);
        else {
            nk_i32_t const key_base = nk_attention_plane_base_serial_(key_exponent);
            for (nk_size_t lane_idx = 0; lane_idx < tile_dimension; lane_idx++) bases[lane_idx] = key_base;
            // Fully padded tiles still run and store the zeros the score stage expects
            for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / tile_dimension;
                 position_tile_idx++) {
                nk_size_t const position_start = position_tile_idx * tile_dimension;
                nk_size_t const rows_to_pack = position_start >= position_count ? 0
                                               : position_count - position_start < tile_dimension
                                                   ? position_count - position_start
                                                   : tile_dimension;
                nk_stage_panel_mxfp6e3m2_sme_(keys, key_stride, position_first + position_start, rows_to_pack,
                                              key_value_head_idx * depth, depth, bases,
                                              keys_plane + position_tile_idx * depth_pairs * vector_elements);
            }
        }

        if (value_exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_plane_sme_streaming_(
                (nk_u8_t const *)values.elements + position_first * value_stride + key_value_head_idx * codes_bytes,
                value_stride, value_scales, values.scales_stride, codes_bytes, blocks, position_count,
                position_count_padded, row_bytes, (nk_u8_t *)values_plane);
        else {
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);
            for (nk_size_t channel_tile_idx = 0; channel_tile_idx < depth_padded / tile_dimension; channel_tile_idx++) {
                nk_size_t const channel_start = channel_tile_idx * tile_dimension;
                nk_size_t const channels = depth - channel_start < tile_dimension ? depth - channel_start
                                                                                  : tile_dimension;
                nk_size_t const channel_first = key_value_head_idx * depth + channel_start;
                nk_u16_t *tile_output = values_plane + channel_tile_idx * (position_count_padded / 2) * vector_elements;
                for (nk_size_t position_pair_idx = 0; position_pair_idx < position_count_padded / 2;
                     position_pair_idx++) {
                    nk_size_t const position_even = position_pair_idx * 2, position_odd = position_even + 1;
                    svuint16_t even_u16x = svdup_u16(0), odd_u16x = svdup_u16(0);
                    if (position_even < position_count) {
                        nk_size_t const row = position_first + position_even;
                        even_u16x = nk_decode_row_mxfp6e3m2_sme_((nk_u8_t const *)values.elements + row * value_stride,
                                                                 values.scales + row * values.scales_stride,
                                                                 channel_first, channels, value_base);
                    }
                    if (position_odd < position_count) {
                        nk_size_t const row = position_first + position_odd;
                        odd_u16x = nk_decode_row_mxfp6e3m2_sme_((nk_u8_t const *)values.elements + row * value_stride,
                                                                values.scales + row * values.scales_stride,
                                                                channel_first, channels, value_base);
                    }
                    svst1_u16(svptrue_b16(), tile_output + position_pair_idx * vector_elements,
                              svzip1_u16(even_u16x, odd_u16x));
                }
            }
        }
    }
}

/** @c nk_attention_packed_mxfp4_sme_streaming_ for MXFP6 E3M2, over the dots stager. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_packed_mxfp6e3m2_sme_streaming_(                   //
    nk_cross_operand_t queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                             //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale2,   //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const row_bytes = depth_padded * sizeof(nk_u16_t), blocks = depth / 32;

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, vector_elements,
                                                                                 row_bytes, key_offsets, key_lengths,
                                                                                 segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_u16_t queries_packed[2][(nk_attention_max_depth_sme_k_ / 2) * 2 * nk_attention_max_tile_sme_k_];
    nk_i32_t bases[nk_attention_max_tile_sme_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
            nk_size_t const plane_bytes = position_count_padded * row_bytes;
            nk_u16_t const *keys_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                            key_value_head_idx * plane_bytes);
            nk_u16_t const *values_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                              (key_value_head_count + key_value_head_idx) *
                                                                  plane_bytes);
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;
                nk_size_t const block_token = query_first + row_block_start;
                nk_i32_t query_base = 0;
                nk_f32_t const block_scale2 = nk_attention_block_scale2_sme_(
                    queries.scales + block_token * queries.scales_stride + head_idx * blocks, queries.scales_stride,
                    block_rows, blocks, key_exponent, value_exponent, scale2, &query_base);
                if (block_scale2 == 0) continue;
                for (nk_size_t lane_idx = 0; lane_idx < tile_dimension; lane_idx++) bases[lane_idx] = query_base;
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
                    nk_size_t const rows_to_stage = block_rows <= tile_row_start ? 0
                                                    : block_rows - tile_row_start < tile_dimension
                                                        ? block_rows - tile_row_start
                                                        : tile_dimension;
                    nk_stage_panel_mxfp6e3m2_sme_(queries, query_stride, block_token + tile_row_start, rows_to_stage,
                                                  head_idx * depth, depth, bases, queries_packed[row_tile_idx]);
                }
                nk_attention_block_bf16_sme_streaming_(
                    queries_packed[0], queries_packed[1], keys_plane, values_plane, depth, position_count, band,
                    first_position + (nk_i64_t)row_block_start, block_rows,
                    row_begin > row_block_start ? row_begin - row_block_start : 0, row_end - row_block_start,
                    block_scale2, 1.0f, value_base, output + block_token * output_stride_floats + head_idx * depth,
                    output_stride_floats,
                    log_sum_exp ? log_sum_exp + block_token * head_count + head_idx : NUMKONG_NULL, head_count);
            }
        }
    }
}

/** @c nk_attention_packed_exact_mxfp4_sme_ for MXFP6 E3M2. */
NUMKONG_OUTLINED_ void nk_attention_packed_exact_mxfp6e3m2_sme_(                                       //
    nk_cross_operand_t queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                             //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale2,   //
    nk_diagonal_band_t band, nk_size_t tile_dimension, nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const row_bytes = depth * sizeof(nk_bf16_t), blocks = depth / 32;
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                             key_value_head_count, block_rows_capacity,
                                                                             row_bytes, key_offsets, key_lengths,
                                                                             segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_f32_t key_row[nk_attention_max_depth_sme_k_];
    nk_align_(64) nk_f32_t value_row[nk_attention_max_depth_sme_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, block_rows_capacity);
            nk_size_t const plane_bytes = position_count_padded * row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] + key_value_head_idx * plane_bytes;
            char const *values_plane = payload_base + payload_offsets[segment_idx] +
                                       (key_value_head_count + key_value_head_idx) * plane_bytes;
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;
                nk_i32_t block_base = 0;
                if (nk_attention_block_scale2_sme_(
                        queries.scales + (query_first + row_block_start) * queries.scales_stride + head_idx * blocks,
                        queries.scales_stride, block_rows, blocks, key_exponent, value_exponent, scale2,
                        &block_base) != 0)
                    continue;
                nk_size_t const block_end = row_block_start + block_rows < row_end ? row_block_start + block_rows
                                                                                   : row_end;
                for (nk_size_t row = row_begin > row_block_start ? row_begin : row_block_start; row < block_end;
                     row++) {
                    nk_size_t const token = query_first + row;
                    nk_u8_t const *query_row = (nk_u8_t const *)queries.elements + token * query_stride +
                                               head_idx * depth;
                    nk_u8_t const *query_row_scales = queries.scales + token * queries.scales_stride +
                                                      head_idx * blocks;
                    nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                    nk_size_t key_begin, key_end;
                    nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row, position_count, &key_begin,
                                                &key_end);

                    nk_f32_t max2 = NUMKONG_F32_MIN;
                    for (nk_size_t position = key_begin; position < key_end; position++) {
                        nk_f32_t const scaled2 = nk_attention_score_mxfp6e3m2_serial_(
                            query_row, query_row_scales,
                            nk_attention_key_row_sme_(keys_plane, key_exponent, tile_dimension, depth, position,
                                                      key_row),
                            key_exponent, depth, scale2);
                        if (scaled2 > max2) max2 = scaled2;
                    }
                    for (nk_size_t channel = 0; channel < depth; channel++) output_row[channel] = 0;
                    nk_f32_t weights_sum = 0;
                    for (nk_size_t position = key_begin; position < key_end; position++) {
                        nk_f32_t const weight = nk_f32_exp2_serial_(
                            nk_attention_score_mxfp6e3m2_serial_(
                                query_row, query_row_scales,
                                nk_attention_key_row_sme_(keys_plane, key_exponent, tile_dimension, depth, position,
                                                          key_row),
                                key_exponent, depth, scale2) -
                            max2);
                        nk_u8_t const *value_bytes = nk_attention_value_row_sme_(
                            values_plane, value_exponent, tile_dimension, depth, position_count_padded / 2, position,
                            value_row);
                        weights_sum += weight;
                        for (nk_size_t channel = 0; channel < depth; channel++)
                            output_row[channel] += weight * nk_attention_plane_mxfp6e3m2_serial_(
                                                                value_bytes, value_exponent, depth, channel);
                    }
                    nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                    for (nk_size_t channel = 0; channel < depth; channel++)
                        output_row[channel] = nk_scale_f32_serial_(output_row[channel] * inverse_sum, value_base);
                    if (log_sum_exp)
                        log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(max2,
                                                                                                      weights_sum);
                }
            }
        }
    }
}

/** @c nk_attention_pack_mxfp4_sme_streaming_ for MXFP8 E4M3, over the dots stager and decoder. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_pack_mxfp8e4m3_sme_streaming_(                       //
    nk_cross_operand_t keys, nk_cross_operand_t values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count,                   //
    nk_size_t key_stride, nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin,         //
    nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2, row_bytes = depth_padded * sizeof(nk_u16_t);
    nk_size_t const blocks = depth / 32, codes_bytes = depth;

    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t *plane_exponents = (nk_i8_t *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                 key_value_head_count, vector_elements, row_bytes,
                                                                 key_offsets, key_lengths, segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const plane_exponents_size = nk_attention_plane_exponents_size_serial_(key_value_head_count,
                                                                                     segment_count);
    if (tasks_begin == 0)
        for (nk_size_t entry = 2 * total_tasks; entry < plane_exponents_size; entry++) plane_exponents[entry] = 0;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_i32_t bases[nk_attention_max_tile_sme_k_];
    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count,
                vector_elements, row_bytes);
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
        nk_size_t const plane_bytes = position_count_padded * row_bytes;
        nk_u16_t *keys_plane = (nk_u16_t *)(payload_base + payload_offset + key_value_head_idx * plane_bytes);
        nk_u16_t *values_plane = (nk_u16_t *)(payload_base + payload_offset +
                                              (key_value_head_count + key_value_head_idx) * plane_bytes);
        nk_u8_t const *key_scales = keys.scales + position_first * keys.scales_stride + key_value_head_idx * blocks;
        nk_u8_t const *value_scales = values.scales + position_first * values.scales_stride +
                                      key_value_head_idx * blocks;
        nk_i8_t const key_exponent = nk_attention_plane_exponent_ue8m0_serial_(key_scales, keys.scales_stride,
                                                                               position_count, blocks);
        nk_i8_t const value_exponent = nk_attention_plane_exponent_ue8m0_serial_(value_scales, values.scales_stride,
                                                                                 position_count, blocks);
        plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx] = key_exponent;
        plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx] = value_exponent;

        if (key_exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_plane_sme_streaming_(
                (nk_u8_t const *)keys.elements + position_first * key_stride + key_value_head_idx * codes_bytes,
                key_stride, key_scales, keys.scales_stride, codes_bytes, blocks, position_count, position_count_padded,
                row_bytes, (nk_u8_t *)keys_plane);
        else {
            nk_i32_t const key_base = nk_attention_plane_base_serial_(key_exponent);
            for (nk_size_t lane_idx = 0; lane_idx < tile_dimension; lane_idx++) bases[lane_idx] = key_base;
            // Fully padded tiles still run and store the zeros the score stage expects
            for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / tile_dimension;
                 position_tile_idx++) {
                nk_size_t const position_start = position_tile_idx * tile_dimension;
                nk_size_t const rows_to_pack = position_start >= position_count ? 0
                                               : position_count - position_start < tile_dimension
                                                   ? position_count - position_start
                                                   : tile_dimension;
                nk_stage_panel_mxfp8e4m3_sme_(keys, key_stride, position_first + position_start, rows_to_pack,
                                              key_value_head_idx * depth, depth, bases,
                                              keys_plane + position_tile_idx * depth_pairs * vector_elements);
            }
        }

        if (value_exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_plane_sme_streaming_(
                (nk_u8_t const *)values.elements + position_first * value_stride + key_value_head_idx * codes_bytes,
                value_stride, value_scales, values.scales_stride, codes_bytes, blocks, position_count,
                position_count_padded, row_bytes, (nk_u8_t *)values_plane);
        else {
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);
            for (nk_size_t channel_tile_idx = 0; channel_tile_idx < depth_padded / tile_dimension; channel_tile_idx++) {
                nk_size_t const channel_start = channel_tile_idx * tile_dimension;
                nk_size_t const channels = depth - channel_start < tile_dimension ? depth - channel_start
                                                                                  : tile_dimension;
                nk_size_t const channel_first = key_value_head_idx * depth + channel_start;
                nk_u16_t *tile_output = values_plane + channel_tile_idx * (position_count_padded / 2) * vector_elements;
                for (nk_size_t position_pair_idx = 0; position_pair_idx < position_count_padded / 2;
                     position_pair_idx++) {
                    nk_size_t const position_even = position_pair_idx * 2, position_odd = position_even + 1;
                    svuint16_t even_u16x = svdup_u16(0), odd_u16x = svdup_u16(0);
                    if (position_even < position_count) {
                        nk_size_t const row = position_first + position_even;
                        even_u16x = nk_decode_row_mxfp8e4m3_sme_((nk_u8_t const *)values.elements + row * value_stride,
                                                                 values.scales + row * values.scales_stride,
                                                                 channel_first, channels, value_base);
                    }
                    if (position_odd < position_count) {
                        nk_size_t const row = position_first + position_odd;
                        odd_u16x = nk_decode_row_mxfp8e4m3_sme_((nk_u8_t const *)values.elements + row * value_stride,
                                                                values.scales + row * values.scales_stride,
                                                                channel_first, channels, value_base);
                    }
                    svst1_u16(svptrue_b16(), tile_output + position_pair_idx * vector_elements,
                              svzip1_u16(even_u16x, odd_u16x));
                }
            }
        }
    }
}

/** @c nk_attention_packed_mxfp4_sme_streaming_ for MXFP8 E4M3, over the dots stager. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_packed_mxfp8e4m3_sme_streaming_(                   //
    nk_cross_operand_t queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                             //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale2,   //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const row_bytes = depth_padded * sizeof(nk_u16_t), blocks = depth / 32;

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, vector_elements,
                                                                                 row_bytes, key_offsets, key_lengths,
                                                                                 segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_u16_t queries_packed[2][(nk_attention_max_depth_sme_k_ / 2) * 2 * nk_attention_max_tile_sme_k_];
    nk_i32_t bases[nk_attention_max_tile_sme_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
            nk_size_t const plane_bytes = position_count_padded * row_bytes;
            nk_u16_t const *keys_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                            key_value_head_idx * plane_bytes);
            nk_u16_t const *values_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                              (key_value_head_count + key_value_head_idx) *
                                                                  plane_bytes);
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;
                nk_size_t const block_token = query_first + row_block_start;
                nk_i32_t query_base = 0;
                nk_f32_t const block_scale2 = nk_attention_block_scale2_sme_(
                    queries.scales + block_token * queries.scales_stride + head_idx * blocks, queries.scales_stride,
                    block_rows, blocks, key_exponent, value_exponent, scale2, &query_base);
                if (block_scale2 == 0) continue;
                for (nk_size_t lane_idx = 0; lane_idx < tile_dimension; lane_idx++) bases[lane_idx] = query_base;
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
                    nk_size_t const rows_to_stage = block_rows <= tile_row_start ? 0
                                                    : block_rows - tile_row_start < tile_dimension
                                                        ? block_rows - tile_row_start
                                                        : tile_dimension;
                    nk_stage_panel_mxfp8e4m3_sme_(queries, query_stride, block_token + tile_row_start, rows_to_stage,
                                                  head_idx * depth, depth, bases, queries_packed[row_tile_idx]);
                }
                nk_attention_block_bf16_sme_streaming_(
                    queries_packed[0], queries_packed[1], keys_plane, values_plane, depth, position_count, band,
                    first_position + (nk_i64_t)row_block_start, block_rows,
                    row_begin > row_block_start ? row_begin - row_block_start : 0, row_end - row_block_start,
                    block_scale2, 1.0f, value_base, output + block_token * output_stride_floats + head_idx * depth,
                    output_stride_floats,
                    log_sum_exp ? log_sum_exp + block_token * head_count + head_idx : NUMKONG_NULL, head_count);
            }
        }
    }
}

/** @c nk_attention_packed_exact_mxfp4_sme_ for MXFP8 E4M3. */
NUMKONG_OUTLINED_ void nk_attention_packed_exact_mxfp8e4m3_sme_(                                       //
    nk_cross_operand_t queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                             //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale2,   //
    nk_diagonal_band_t band, nk_size_t tile_dimension, nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const row_bytes = depth * sizeof(nk_bf16_t), blocks = depth / 32;
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                             key_value_head_count, block_rows_capacity,
                                                                             row_bytes, key_offsets, key_lengths,
                                                                             segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_f32_t key_row[nk_attention_max_depth_sme_k_];
    nk_align_(64) nk_f32_t value_row[nk_attention_max_depth_sme_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, block_rows_capacity);
            nk_size_t const plane_bytes = position_count_padded * row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] + key_value_head_idx * plane_bytes;
            char const *values_plane = payload_base + payload_offsets[segment_idx] +
                                       (key_value_head_count + key_value_head_idx) * plane_bytes;
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;
                nk_i32_t block_base = 0;
                if (nk_attention_block_scale2_sme_(
                        queries.scales + (query_first + row_block_start) * queries.scales_stride + head_idx * blocks,
                        queries.scales_stride, block_rows, blocks, key_exponent, value_exponent, scale2,
                        &block_base) != 0)
                    continue;
                nk_size_t const block_end = row_block_start + block_rows < row_end ? row_block_start + block_rows
                                                                                   : row_end;
                for (nk_size_t row = row_begin > row_block_start ? row_begin : row_block_start; row < block_end;
                     row++) {
                    nk_size_t const token = query_first + row;
                    nk_u8_t const *query_row = (nk_u8_t const *)queries.elements + token * query_stride +
                                               head_idx * depth;
                    nk_u8_t const *query_row_scales = queries.scales + token * queries.scales_stride +
                                                      head_idx * blocks;
                    nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                    nk_size_t key_begin, key_end;
                    nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row, position_count, &key_begin,
                                                &key_end);

                    nk_f32_t max2 = NUMKONG_F32_MIN;
                    for (nk_size_t position = key_begin; position < key_end; position++) {
                        nk_f32_t const scaled2 = nk_attention_score_mxfp8e4m3_serial_(
                            query_row, query_row_scales,
                            nk_attention_key_row_sme_(keys_plane, key_exponent, tile_dimension, depth, position,
                                                      key_row),
                            key_exponent, depth, scale2);
                        if (scaled2 > max2) max2 = scaled2;
                    }
                    for (nk_size_t channel = 0; channel < depth; channel++) output_row[channel] = 0;
                    nk_f32_t weights_sum = 0;
                    for (nk_size_t position = key_begin; position < key_end; position++) {
                        nk_f32_t const weight = nk_f32_exp2_serial_(
                            nk_attention_score_mxfp8e4m3_serial_(
                                query_row, query_row_scales,
                                nk_attention_key_row_sme_(keys_plane, key_exponent, tile_dimension, depth, position,
                                                          key_row),
                                key_exponent, depth, scale2) -
                            max2);
                        nk_u8_t const *value_bytes = nk_attention_value_row_sme_(
                            values_plane, value_exponent, tile_dimension, depth, position_count_padded / 2, position,
                            value_row);
                        weights_sum += weight;
                        for (nk_size_t channel = 0; channel < depth; channel++)
                            output_row[channel] += weight * nk_attention_plane_mxfp8e4m3_serial_(
                                                                value_bytes, value_exponent, depth, channel);
                    }
                    nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                    for (nk_size_t channel = 0; channel < depth; channel++)
                        output_row[channel] = nk_scale_f32_serial_(output_row[channel] * inverse_sum, value_base);
                    if (log_sum_exp)
                        log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(max2,
                                                                                                      weights_sum);
                }
            }
        }
    }
}

/** @c nk_attention_pack_mxfp4_sme_streaming_ for MXFP8 E5M2, over the dots stager and decoder. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_pack_mxfp8e5m2_sme_streaming_(                       //
    nk_cross_operand_t keys, nk_cross_operand_t values, nk_size_t key_value_head_count, nk_size_t depth, //
    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths, nk_size_t segment_count,                   //
    nk_size_t key_stride, nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin,         //
    nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2, row_bytes = depth_padded * sizeof(nk_u16_t);
    nk_size_t const blocks = depth / 32, codes_bytes = depth;

    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t *plane_exponents = (nk_i8_t *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                 key_value_head_count, vector_elements, row_bytes,
                                                                 key_offsets, key_lengths, segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    nk_size_t const plane_exponents_size = nk_attention_plane_exponents_size_serial_(key_value_head_count,
                                                                                     segment_count);
    if (tasks_begin == 0)
        for (nk_size_t entry = 2 * total_tasks; entry < plane_exponents_size; entry++) plane_exponents[entry] = 0;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    nk_i32_t bases[nk_attention_max_tile_sme_k_];
    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count,
                vector_elements, row_bytes);
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
        nk_size_t const plane_bytes = position_count_padded * row_bytes;
        nk_u16_t *keys_plane = (nk_u16_t *)(payload_base + payload_offset + key_value_head_idx * plane_bytes);
        nk_u16_t *values_plane = (nk_u16_t *)(payload_base + payload_offset +
                                              (key_value_head_count + key_value_head_idx) * plane_bytes);
        nk_u8_t const *key_scales = keys.scales + position_first * keys.scales_stride + key_value_head_idx * blocks;
        nk_u8_t const *value_scales = values.scales + position_first * values.scales_stride +
                                      key_value_head_idx * blocks;
        nk_i8_t const key_exponent = nk_attention_plane_exponent_ue8m0_serial_(key_scales, keys.scales_stride,
                                                                               position_count, blocks);
        nk_i8_t const value_exponent = nk_attention_plane_exponent_ue8m0_serial_(value_scales, values.scales_stride,
                                                                                 position_count, blocks);
        plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx] = key_exponent;
        plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx] = value_exponent;

        if (key_exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_plane_sme_streaming_(
                (nk_u8_t const *)keys.elements + position_first * key_stride + key_value_head_idx * codes_bytes,
                key_stride, key_scales, keys.scales_stride, codes_bytes, blocks, position_count, position_count_padded,
                row_bytes, (nk_u8_t *)keys_plane);
        else {
            nk_i32_t const key_base = nk_attention_plane_base_serial_(key_exponent);
            for (nk_size_t lane_idx = 0; lane_idx < tile_dimension; lane_idx++) bases[lane_idx] = key_base;
            // Fully padded tiles still run and store the zeros the score stage expects
            for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / tile_dimension;
                 position_tile_idx++) {
                nk_size_t const position_start = position_tile_idx * tile_dimension;
                nk_size_t const rows_to_pack = position_start >= position_count ? 0
                                               : position_count - position_start < tile_dimension
                                                   ? position_count - position_start
                                                   : tile_dimension;
                nk_stage_panel_mxfp8e5m2_sme_(keys, key_stride, position_first + position_start, rows_to_pack,
                                              key_value_head_idx * depth, depth, bases,
                                              keys_plane + position_tile_idx * depth_pairs * vector_elements);
            }
        }

        if (value_exponent == nk_attention_raw_plane_k_)
            nk_attention_raw_plane_sme_streaming_(
                (nk_u8_t const *)values.elements + position_first * value_stride + key_value_head_idx * codes_bytes,
                value_stride, value_scales, values.scales_stride, codes_bytes, blocks, position_count,
                position_count_padded, row_bytes, (nk_u8_t *)values_plane);
        else {
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);
            for (nk_size_t channel_tile_idx = 0; channel_tile_idx < depth_padded / tile_dimension; channel_tile_idx++) {
                nk_size_t const channel_start = channel_tile_idx * tile_dimension;
                nk_size_t const channels = depth - channel_start < tile_dimension ? depth - channel_start
                                                                                  : tile_dimension;
                nk_size_t const channel_first = key_value_head_idx * depth + channel_start;
                nk_u16_t *tile_output = values_plane + channel_tile_idx * (position_count_padded / 2) * vector_elements;
                for (nk_size_t position_pair_idx = 0; position_pair_idx < position_count_padded / 2;
                     position_pair_idx++) {
                    nk_size_t const position_even = position_pair_idx * 2, position_odd = position_even + 1;
                    svuint16_t even_u16x = svdup_u16(0), odd_u16x = svdup_u16(0);
                    if (position_even < position_count) {
                        nk_size_t const row = position_first + position_even;
                        even_u16x = nk_decode_row_mxfp8e5m2_sme_((nk_u8_t const *)values.elements + row * value_stride,
                                                                 values.scales + row * values.scales_stride,
                                                                 channel_first, channels, value_base);
                    }
                    if (position_odd < position_count) {
                        nk_size_t const row = position_first + position_odd;
                        odd_u16x = nk_decode_row_mxfp8e5m2_sme_((nk_u8_t const *)values.elements + row * value_stride,
                                                                values.scales + row * values.scales_stride,
                                                                channel_first, channels, value_base);
                    }
                    svst1_u16(svptrue_b16(), tile_output + position_pair_idx * vector_elements,
                              svzip1_u16(even_u16x, odd_u16x));
                }
            }
        }
    }
}

/** @c nk_attention_packed_mxfp4_sme_streaming_ for MXFP8 E5M2, over the dots stager. */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_packed_mxfp8e5m2_sme_streaming_(                   //
    nk_cross_operand_t queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                             //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale2,   //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const row_bytes = depth_padded * sizeof(nk_u16_t), blocks = depth / 32;

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0 && depth % 32 == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed +
                                     nk_attention_plane_exponents_offset_serial_(key_value_head_count, vector_elements,
                                                                                 row_bytes, key_offsets, key_lengths,
                                                                                 segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_u16_t queries_packed[2][(nk_attention_max_depth_sme_k_ / 2) * 2 * nk_attention_max_tile_sme_k_];
    nk_i32_t bases[nk_attention_max_tile_sme_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
            nk_size_t const plane_bytes = position_count_padded * row_bytes;
            nk_u16_t const *keys_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                            key_value_head_idx * plane_bytes);
            nk_u16_t const *values_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                              (key_value_head_count + key_value_head_idx) *
                                                                  plane_bytes);
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;
                nk_size_t const block_token = query_first + row_block_start;
                nk_i32_t query_base = 0;
                nk_f32_t const block_scale2 = nk_attention_block_scale2_sme_(
                    queries.scales + block_token * queries.scales_stride + head_idx * blocks, queries.scales_stride,
                    block_rows, blocks, key_exponent, value_exponent, scale2, &query_base);
                if (block_scale2 == 0) continue;
                for (nk_size_t lane_idx = 0; lane_idx < tile_dimension; lane_idx++) bases[lane_idx] = query_base;
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
                    nk_size_t const rows_to_stage = block_rows <= tile_row_start ? 0
                                                    : block_rows - tile_row_start < tile_dimension
                                                        ? block_rows - tile_row_start
                                                        : tile_dimension;
                    nk_stage_panel_mxfp8e5m2_sme_(queries, query_stride, block_token + tile_row_start, rows_to_stage,
                                                  head_idx * depth, depth, bases, queries_packed[row_tile_idx]);
                }
                nk_attention_block_bf16_sme_streaming_(
                    queries_packed[0], queries_packed[1], keys_plane, values_plane, depth, position_count, band,
                    first_position + (nk_i64_t)row_block_start, block_rows,
                    row_begin > row_block_start ? row_begin - row_block_start : 0, row_end - row_block_start,
                    block_scale2, 1.0f, value_base, output + block_token * output_stride_floats + head_idx * depth,
                    output_stride_floats,
                    log_sum_exp ? log_sum_exp + block_token * head_count + head_idx : NUMKONG_NULL, head_count);
            }
        }
    }
}

/** @c nk_attention_packed_exact_mxfp4_sme_ for MXFP8 E5M2. */
NUMKONG_OUTLINED_ void nk_attention_packed_exact_mxfp8e5m2_sme_(                                       //
    nk_cross_operand_t queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp, //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                             //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale2,   //
    nk_diagonal_band_t band, nk_size_t tile_dimension, nk_size_t tasks_begin, nk_size_t tasks_end) {

    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const row_bytes = depth * sizeof(nk_bf16_t), blocks = depth / 32;
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_offsets = nk_attention_packed_key_offsets_serial_(key_value_packed, segment_count);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_i8_t const *plane_exponents = (nk_i8_t const *)key_value_packed + nk_attention_plane_exponents_offset_serial_(
                                                                             key_value_head_count, block_rows_capacity,
                                                                             row_bytes, key_offsets, key_lengths,
                                                                             segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_align_(64) nk_f32_t key_row[nk_attention_max_depth_sme_k_];
    nk_align_(64) nk_f32_t value_row[nk_attention_max_depth_sme_k_];

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, block_rows_capacity);
            nk_size_t const plane_bytes = position_count_padded * row_bytes;
            char const *keys_plane = payload_base + payload_offsets[segment_idx] + key_value_head_idx * plane_bytes;
            char const *values_plane = payload_base + payload_offsets[segment_idx] +
                                       (key_value_head_count + key_value_head_idx) * plane_bytes;
            nk_i8_t const key_exponent = plane_exponents[2 * segment_idx * key_value_head_count + key_value_head_idx];
            nk_i8_t const value_exponent =
                plane_exponents[(2 * segment_idx + 1) * key_value_head_count + key_value_head_idx];
            nk_i32_t const value_base = nk_attention_plane_base_serial_(value_exponent);

            nk_size_t const row_count = query_end - query_first;
            for (nk_size_t row_block_start = row_begin / block_rows_capacity * block_rows_capacity;
                 row_block_start < row_end; row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                                 ? row_count - row_block_start
                                                 : block_rows_capacity;
                nk_i32_t block_base = 0;
                if (nk_attention_block_scale2_sme_(
                        queries.scales + (query_first + row_block_start) * queries.scales_stride + head_idx * blocks,
                        queries.scales_stride, block_rows, blocks, key_exponent, value_exponent, scale2,
                        &block_base) != 0)
                    continue;
                nk_size_t const block_end = row_block_start + block_rows < row_end ? row_block_start + block_rows
                                                                                   : row_end;
                for (nk_size_t row = row_begin > row_block_start ? row_begin : row_block_start; row < block_end;
                     row++) {
                    nk_size_t const token = query_first + row;
                    nk_u8_t const *query_row = (nk_u8_t const *)queries.elements + token * query_stride +
                                               head_idx * depth;
                    nk_u8_t const *query_row_scales = queries.scales + token * queries.scales_stride +
                                                      head_idx * blocks;
                    nk_f32_t *output_row = output + token * output_stride_floats + head_idx * depth;
                    nk_size_t key_begin, key_end;
                    nk_diagonal_band_row_range_(band, first_position + (nk_i64_t)row, position_count, &key_begin,
                                                &key_end);

                    nk_f32_t max2 = NUMKONG_F32_MIN;
                    for (nk_size_t position = key_begin; position < key_end; position++) {
                        nk_f32_t const scaled2 = nk_attention_score_mxfp8e5m2_serial_(
                            query_row, query_row_scales,
                            nk_attention_key_row_sme_(keys_plane, key_exponent, tile_dimension, depth, position,
                                                      key_row),
                            key_exponent, depth, scale2);
                        if (scaled2 > max2) max2 = scaled2;
                    }
                    for (nk_size_t channel = 0; channel < depth; channel++) output_row[channel] = 0;
                    nk_f32_t weights_sum = 0;
                    for (nk_size_t position = key_begin; position < key_end; position++) {
                        nk_f32_t const weight = nk_f32_exp2_serial_(
                            nk_attention_score_mxfp8e5m2_serial_(
                                query_row, query_row_scales,
                                nk_attention_key_row_sme_(keys_plane, key_exponent, tile_dimension, depth, position,
                                                          key_row),
                                key_exponent, depth, scale2) -
                            max2);
                        nk_u8_t const *value_bytes = nk_attention_value_row_sme_(
                            values_plane, value_exponent, tile_dimension, depth, position_count_padded / 2, position,
                            value_row);
                        weights_sum += weight;
                        for (nk_size_t channel = 0; channel < depth; channel++)
                            output_row[channel] += weight * nk_attention_plane_mxfp8e5m2_serial_(
                                                                value_bytes, value_exponent, depth, channel);
                    }
                    nk_f32_t const inverse_sum = weights_sum > 0 ? 1 / weights_sum : 0;
                    for (nk_size_t channel = 0; channel < depth; channel++)
                        output_row[channel] = nk_scale_f32_serial_(output_row[channel] * inverse_sum, value_base);
                    if (log_sum_exp)
                        log_sum_exp[token * head_count + head_idx] = nk_attention_log_sum_exp_serial_(max2,
                                                                                                      weights_sum);
                }
            }
        }
    }
}

/** Bytes of a pack of raw I8 planes, positions padded like the 16-bit ones. */
NUMKONG_INLINE nk_size_t nk_attention_pack_size_b8_sme_(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count) {
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, nk_cntw_sme_());
    return nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, nk_cnth_sme_(),
                                           depth_padded);
}

/**
 *  @brief Streaming pack core for I8 K/V planes.
 *
 *  K becomes quad-interleaved SMOPA operand vectors `[position_tile][depth_quad]` through the ZA0
 *  horizontal-write / vertical-read transpose; V becomes transposed quad-interleaved vectors
 *  `[channel_tile][position_quad]` through a two-level @c ZIP1, so P × V runs as USMOPA outer
 *  products over positions with the U8 probabilities.
 */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_pack_i8_sme_streaming_(     //
    nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count, //
    nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,  //
    nk_size_t segment_count, nk_size_t key_stride, nk_size_t value_stride,      //
    void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_bytes = svcntb();
    nk_size_t const position_multiple = svcnth();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_quads = depth_padded / 4;

    char *payload_base = (char *)key_value_packed + sizeof(nk_attention_packed_header_t) +
                         nk_attention_pack_directory_size_serial_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (tasks_begin >= total_tasks) return;
    if (tasks_end > total_tasks) tasks_end = total_tasks;

    svbool_t const predicate_all_b32x = svptrue_b32();

    nk_size_t payload_segment = 0;
    nk_u64_t payload_offset = 0;
    for (nk_size_t task_idx = tasks_begin; task_idx < tasks_end; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        for (; payload_segment < segment_idx; payload_segment++)
            payload_offset += nk_attention_pack_segment_bytes_serial_(
                nk_attention_pack_key_count_serial_(key_offsets, key_lengths, payload_segment), key_value_head_count,
                position_multiple, depth_padded);
        nk_size_t const position_count = nk_attention_pack_key_count_serial_(key_offsets, key_lengths, segment_idx);
        if (position_count == 0) continue;
        nk_size_t const position_first = key_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, position_multiple);
        nk_size_t const plane_bytes = position_count_padded * depth_padded;
        nk_u8_t *keys_plane = (nk_u8_t *)(payload_base + payload_offset + key_value_head_idx * plane_bytes);
        nk_u8_t *values_plane = (nk_u8_t *)(payload_base + payload_offset +
                                            (key_value_head_count + key_value_head_idx) * plane_bytes);

        // K: quad-interleave each position tile via the ZA0 transpose; fully padded tiles
        // still run and store the zeros the score stage expects.
        for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / tile_dimension;
             position_tile_idx++) {
            nk_size_t const position_start = position_tile_idx * tile_dimension;
            nk_size_t const rows_to_pack = (position_start + tile_dimension <= position_count) ? tile_dimension
                                           : (position_start < position_count) ? position_count - position_start
                                                                               : 0;
            char const *source = (char const *)keys + (position_first + position_start) * key_stride +
                                 key_value_head_idx * depth;
            svbool_t const row_predicate_b32x = svwhilelt_b32_u64(0u, rows_to_pack);
            nk_u8_t *tile_output = keys_plane + position_tile_idx * depth_quads * vector_bytes;
            for (nk_size_t step = 0; step < depth_quads; step++) {
                nk_size_t const slice = step % tile_dimension;
                if (slice == 0)
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_to_pack; row_in_tile++)
                        svwrite_hor_za32_u32_m(0, (uint32_t)row_in_tile, predicate_all_b32x,
                                               svreinterpret_u32_u8(svld1_u8(
                                                   svwhilelt_b8_u64(4 * step, depth),
                                                   (nk_u8_t const *)(source + row_in_tile * key_stride) + 4 * step)));
                svuint8_t const packed_u8x = svreinterpret_u8_u32(
                    svread_ver_za32_u32_m(svdup_u32(0), row_predicate_b32x, 0, (uint32_t)slice));
                svst1_u8(svptrue_b8(), tile_output + step * vector_bytes, packed_u8x);
            }
        }

        // V: two-level ZIP1 turns four position rows of a channel tile into the quad-interleaved
        // operand; dead positions and channels arrive as zeros from the predicates.
        for (nk_size_t channel_tile_idx = 0; channel_tile_idx < depth_padded / tile_dimension; channel_tile_idx++) {
            nk_size_t const channel_start = channel_tile_idx * tile_dimension;
            char const *source = (char const *)values + position_first * value_stride + key_value_head_idx * depth +
                                 channel_start;
            nk_u8_t *tile_output = values_plane + channel_tile_idx * (position_count_padded / 4) * vector_bytes;
            svbool_t const channel_predicate_b8x = svwhilelt_b8_u64(channel_start, depth);
            for (nk_size_t position_quad_idx = 0; position_quad_idx < position_count_padded / 4; position_quad_idx++) {
                nk_size_t const position_base = position_quad_idx * 4;
                nk_u8_t const *row = (nk_u8_t const *)(source + position_base * value_stride);
                svbool_t const first_b8x = position_base < position_count ? channel_predicate_b8x : svpfalse_b();
                svbool_t const second_b8x = position_base + 1 < position_count ? channel_predicate_b8x : svpfalse_b();
                svbool_t const third_b8x = position_base + 2 < position_count ? channel_predicate_b8x : svpfalse_b();
                svbool_t const fourth_b8x = position_base + 3 < position_count ? channel_predicate_b8x : svpfalse_b();
                svuint8_t const packed_u8x = nk_interleave_quads_u8x_sme_(
                    svld1_u8(first_b8x, row), svld1_u8(second_b8x, row + value_stride),
                    svld1_u8(third_b8x, row + 2 * value_stride), svld1_u8(fourth_b8x, row + 3 * value_stride));
                svst1_u8(svptrue_b8(), tile_output + position_quad_idx * vector_bytes, packed_u8x);
            }
        }
    }
}

/**
 *  @brief Streaming attention core for I8: exact I32 scores through SMOPA, U8-quantized
 *      probabilities, and USMOPA for the probability × value product.
 *
 *  Mirrors the B16 core's structure — ZA0 quad-interleaving Q staging, 2×2 score blocking
 *  with position-major F32 drains, lane-parallel softmax, channel-major F32 accumulator —
 *  with two I8 twists: weights quantize to trunc(2^(s₂ − m₂) · 255 + 0.5) and assemble into
 *  quad-interleaved U8 operands with three shift-ors per vector, and the P × V drain converts
 *  the exact I32 outer products to F32 before the correction FMA.
 */
__arm_new("za") NUMKONG_OUTLINED_ void nk_attention_packed_i8_sme_streaming_(                       //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output, nk_f32_t *log_sum_exp,  //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                          //
    nk_u32_t const *query_offsets, nk_size_t query_stride, nk_size_t output_stride, nk_f32_t scale, //
    nk_diagonal_band_t band, nk_size_t tasks_begin, nk_size_t tasks_end) NUMKONG_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_bytes = svcntb();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const panel_width = nk_attention_panel_sme_k_;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_quads = depth_padded / 4;
    nk_size_t const channel_tiles = depth_padded / tile_dimension;

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_assert_(header->depth == depth && header->key_value_head_count == key_value_head_count &&
               key_value_head_count != 0 && head_count % key_value_head_count == 0);
    nk_size_t const segment_count = header->segments;
    nk_u64_t const *payload_offsets = nk_attention_packed_payload_offsets_serial_(key_value_packed);
    nk_u32_t const *key_lengths = nk_attention_packed_key_lengths_serial_(key_value_packed, segment_count);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_serial_(segment_count);
    nk_size_t const output_stride_floats = output_stride / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_; // fold log2e: softmax(x) = softmax₂(x · log₂e)
    nk_size_t const grid_begin = query_offsets[0] * head_count, grid_end = query_offsets[segment_count] * head_count;
    if (tasks_begin < grid_begin) tasks_begin = grid_begin;
    if (tasks_end > grid_end) tasks_end = grid_end;

    nk_i32_t const scale_fixed = (nk_i32_t)(scale2 * 32768.0f + 0.5f); // Q15 scale for the integer exponential
    nk_i32_t const delta_floor = // the score delta below which every weight quantizes to zero (2^t · 255 + 0.5 < 1)
        scale_fixed > 0 ? -(nk_i32_t)((10u << 15) / (nk_u32_t)scale_fixed) - 1 : 0;

    nk_align_(64) nk_u8_t queries_packed[2][(nk_attention_max_depth_sme_k_ / 4) * 4 * nk_attention_max_tile_sme_k_];
    nk_align_(64) nk_i32_t scores_panel[nk_attention_panel_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    nk_align_(64) nk_u8_t weights_panel[nk_attention_panel_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    nk_align_(64) nk_f32_t o_acc[nk_attention_max_depth_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    nk_align_(64) nk_u32_t key_begins[2 * nk_attention_max_tile_sme_k_]; // visible key range per query lane
    nk_align_(64) nk_u32_t key_ends[2 * nk_attention_max_tile_sme_k_];

    svbool_t const predicate_all_b32x = svptrue_b32();
    svbool_t const predicate_all_b8x = svptrue_b8();
    svfloat32_t const scale2_f32x = svdup_f32(scale2);

    for (nk_size_t head_idx = 0; head_idx < head_count && tasks_begin < tasks_end; head_idx++) {
        nk_size_t const token_first = (tasks_begin + head_count - 1 - head_idx) / head_count;
        nk_size_t const token_end = (tasks_end + head_count - 1 - head_idx) / head_count;
        for (nk_size_t segment_idx = nk_attention_segment_of_serial_(query_offsets, segment_count, token_first);
             segment_idx < segment_count && query_offsets[segment_idx] < token_end; segment_idx++) {
            nk_size_t const query_first = query_offsets[segment_idx], query_end = query_offsets[segment_idx + 1];
            nk_size_t const row_begin = token_first > query_first ? token_first - query_first : 0;
            nk_size_t const row_end = (token_end < query_end ? token_end : query_end) - query_first;
            if (row_begin >= row_end) continue;
            nk_size_t const position_count = key_lengths[segment_idx];
            nk_i64_t const first_position = nk_attention_first_position_serial_(query_end - query_first,
                                                                                position_count);
            nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 2 * tile_dimension);
            nk_size_t const position_quads_total = position_count_padded / 4;
            nk_size_t const plane_bytes = position_count_padded * depth_padded;
            nk_size_t const key_value_head_idx = head_idx / head_group_size;
            nk_u8_t const *keys_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx] +
                                                          key_value_head_idx * plane_bytes);
            nk_u8_t const *values_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx] +
                                                            (key_value_head_count + key_value_head_idx) * plane_bytes);

            for (nk_size_t row_block_start = row_begin; row_block_start < row_end;
                 row_block_start += block_rows_capacity) {
                nk_size_t const block_rows = (row_end - row_block_start < block_rows_capacity)
                                                 ? row_end - row_block_start
                                                 : block_rows_capacity;

                // Stage 1: quad-interleave the block's Q rows through the ZA0 transpose, per block.
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
                    nk_size_t const rows_to_stage = (block_rows > tile_row_start)
                                                        ? ((block_rows - tile_row_start < tile_dimension)
                                                               ? block_rows - tile_row_start
                                                               : tile_dimension)
                                                        : 0;
                    for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_quads;
                         depth_batch_start += tile_dimension) {
                        svbool_t const batch_predicate_b32x = svwhilelt_b32_u64(depth_batch_start, depth_quads);
                        nk_size_t const batch_size = svcntp_b32(svptrue_b32(), batch_predicate_b32x);
                        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_to_stage; row_in_tile++) {
                            char const *row_ptr = (char const *)queries +
                                                  (query_first + row_block_start + tile_row_start + row_in_tile) *
                                                      query_stride +
                                                  head_idx * depth + depth_batch_start * 4;
                            svbool_t const depth_predicate_b8x = svwhilelt_b8_u64(depth_batch_start * 4, depth);
                            svint8_t row_i8x = svld1_s8(depth_predicate_b8x, (int8_t const *)row_ptr);
                            svwrite_hor_za32_s32_m(0, (uint32_t)row_in_tile, batch_predicate_b32x,
                                                   svreinterpret_s32_s8(row_i8x));
                        }
                        for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                            svint32_t column_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 0,
                                                                          (uint32_t)depth_step);
                            svst1_s32(predicate_all_b32x,
                                      (int32_t *)(queries_packed[row_tile_idx] +
                                                  (depth_batch_start + depth_step) * vector_bytes),
                                      column_i32x);
                        }
                    }
                }

                // Per-lane visible key ranges; lanes past the block see no keys.
                // Ranges grow monotonically with the row, so the first and last rows bound it.
                nk_i64_t const block_first_position = first_position + (nk_i64_t)row_block_start;
                for (nk_size_t lane_idx = 0; lane_idx < block_rows_capacity; lane_idx++) {
                    nk_size_t key_begin = 0, key_end = 0;
                    if (lane_idx < block_rows)
                        nk_diagonal_band_row_range_(band, block_first_position + (nk_i64_t)lane_idx, position_count,
                                                    &key_begin, &key_end);
                    key_begins[lane_idx] = (nk_u32_t)key_begin, key_ends[lane_idx] = (nk_u32_t)key_end;
                }
                nk_size_t const block_key_begin = key_begins[0], block_key_end = key_ends[block_rows - 1];
                svuint32_t const key_begins_low_u32x = svld1_u32(predicate_all_b32x, key_begins);
                svuint32_t const key_begins_high_u32x = svld1_u32(predicate_all_b32x, key_begins + tile_dimension);
                svuint32_t const key_ends_low_u32x = svld1_u32(predicate_all_b32x, key_ends);
                svuint32_t const key_ends_high_u32x = svld1_u32(predicate_all_b32x, key_ends + tile_dimension);

                svint32_t running_max_low_i32x = svdup_s32(-2147483647 - 1); // raw scores, one query per lane
                svint32_t running_max_high_i32x = svdup_s32(-2147483647 - 1);
                svfloat32_t running_sum_low_f32x = svdup_f32(0.0f);
                svfloat32_t running_sum_high_f32x = svdup_f32(0.0f);
                for (nk_size_t element_idx = 0; element_idx < depth_padded * block_rows_capacity;
                     element_idx += tile_dimension)
                    svst1_f32(predicate_all_b32x, (float32_t *)(o_acc + element_idx), svdup_f32(0.0f));

                // Only panels crossing some row's range boundary pay for per-lane predicates.
                for (nk_size_t panel_start = block_key_begin / panel_width * panel_width; panel_start < block_key_end;
                     panel_start += panel_width) {
                    nk_size_t const panel_length = (block_key_end - panel_start < panel_width)
                                                       ? block_key_end - panel_start
                                                       : panel_width;
                    nk_diagonal_band_coverage_t const coverage = nk_diagonal_band_tile_coverage_(
                        band, block_first_position, block_rows, panel_start, panel_length);
                    if (coverage == nk_diagonal_band_outside_k) continue;
                    int const panel_masked = coverage == nk_diagonal_band_crossing_k;
                    nk_size_t const panel_quads = nk_size_divide_round_up_(panel_length, 4);

                    svint32_t panel_max_low_i32x = svdup_s32(-2147483647 - 1);
                    svint32_t panel_max_high_i32x = svdup_s32(-2147483647 - 1);
                    // Stage 2: 2×2 SMOPA scores over depth quads, exact in I32, drained
                    // position-major as F32 with one query per lane.
                    for (nk_size_t chunk_start = 0; chunk_start < panel_length; chunk_start += block_rows_capacity) {
                        nk_size_t const position_tile_first = (panel_start + chunk_start) / tile_dimension;
                        nk_u8_t const *keys_tile0 = keys_plane + position_tile_first * depth_quads * vector_bytes;
                        nk_u8_t const *keys_tile1 = keys_tile0 + depth_quads * vector_bytes;
                        svzero_za();
                        for (nk_size_t depth_quad_idx = 0; depth_quad_idx < depth_quads; depth_quad_idx++) {
                            svint8_t const queries_low_i8x = svreinterpret_s8_u8(
                                svld1_u8(predicate_all_b8x, queries_packed[0] + depth_quad_idx * vector_bytes));
                            svint8_t const queries_high_i8x = svreinterpret_s8_u8(
                                svld1_u8(predicate_all_b8x, queries_packed[1] + depth_quad_idx * vector_bytes));
                            svint8_t const keys_low_i8x = svreinterpret_s8_u8(
                                svld1_u8(predicate_all_b8x, keys_tile0 + depth_quad_idx * vector_bytes));
                            svint8_t const keys_high_i8x = svreinterpret_s8_u8(
                                svld1_u8(predicate_all_b8x, keys_tile1 + depth_quad_idx * vector_bytes));
                            svmopa_za32_s8_m(0, predicate_all_b8x, predicate_all_b8x, queries_low_i8x, keys_low_i8x);
                            svmopa_za32_s8_m(1, predicate_all_b8x, predicate_all_b8x, queries_low_i8x, keys_high_i8x);
                            svmopa_za32_s8_m(2, predicate_all_b8x, predicate_all_b8x, queries_high_i8x, keys_low_i8x);
                            svmopa_za32_s8_m(3, predicate_all_b8x, predicate_all_b8x, queries_high_i8x, keys_high_i8x);
                        }
                        nk_i32_t *chunk_scores = scores_panel + chunk_start * block_rows_capacity;
                        for (nk_size_t slice_idx = 0; slice_idx < tile_dimension; slice_idx++) {
                            svint32_t const column0_low_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x,
                                                                                     0, (uint32_t)slice_idx);
                            svint32_t const column0_high_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x,
                                                                                      2, (uint32_t)slice_idx);
                            svint32_t const column1_low_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x,
                                                                                     1, (uint32_t)slice_idx);
                            svint32_t const column1_high_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x,
                                                                                      3, (uint32_t)slice_idx);
                            // Unlike the B16 core, only valid positions may raise the maximum: a
                            // padded zero score above every real one would quantize all U8 weights
                            // to zero and break the weight-sum-never-zero invariant.
                            svint32_t maximum0_low_i32x = column0_low_i32x, maximum0_high_i32x = column0_high_i32x;
                            svint32_t maximum1_low_i32x = column1_low_i32x, maximum1_high_i32x = column1_high_i32x;
                            if (panel_masked) { // hidden keys never raise the maximum
                                nk_size_t const position0 = panel_start + chunk_start + slice_idx;
                                nk_size_t const position1 = position0 + tile_dimension;
                                svint32_t const hidden_i32x = svdup_s32(-2147483647 - 1);
                                maximum0_low_i32x = svsel_s32(
                                    nk_attention_visible_sme_(key_begins_low_u32x, key_ends_low_u32x, position0),
                                    column0_low_i32x, hidden_i32x);
                                maximum0_high_i32x = svsel_s32(
                                    nk_attention_visible_sme_(key_begins_high_u32x, key_ends_high_u32x, position0),
                                    column0_high_i32x, hidden_i32x);
                                maximum1_low_i32x = svsel_s32(
                                    nk_attention_visible_sme_(key_begins_low_u32x, key_ends_low_u32x, position1),
                                    column1_low_i32x, hidden_i32x);
                                maximum1_high_i32x = svsel_s32(
                                    nk_attention_visible_sme_(key_begins_high_u32x, key_ends_high_u32x, position1),
                                    column1_high_i32x, hidden_i32x);
                            }
                            if (chunk_start + slice_idx < panel_length) {
                                panel_max_low_i32x = svmax_s32_x(predicate_all_b32x, panel_max_low_i32x,
                                                                 maximum0_low_i32x);
                                panel_max_high_i32x = svmax_s32_x(predicate_all_b32x, panel_max_high_i32x,
                                                                  maximum0_high_i32x);
                            }
                            if (chunk_start + tile_dimension + slice_idx < panel_length) {
                                panel_max_low_i32x = svmax_s32_x(predicate_all_b32x, panel_max_low_i32x,
                                                                 maximum1_low_i32x);
                                panel_max_high_i32x = svmax_s32_x(predicate_all_b32x, panel_max_high_i32x,
                                                                  maximum1_high_i32x);
                            }
                            svst1_s32(predicate_all_b32x, (int32_t *)(chunk_scores + slice_idx * block_rows_capacity),
                                      column0_low_i32x);
                            svst1_s32(predicate_all_b32x,
                                      (int32_t *)(chunk_scores + slice_idx * block_rows_capacity + tile_dimension),
                                      column0_high_i32x);
                            svst1_s32(predicate_all_b32x,
                                      (int32_t *)(chunk_scores + (tile_dimension + slice_idx) * block_rows_capacity),
                                      column1_low_i32x);
                            svst1_s32(predicate_all_b32x,
                                      (int32_t *)(chunk_scores + (tile_dimension + slice_idx) * block_rows_capacity +
                                                  tile_dimension),
                                      column1_high_i32x);
                        }
                    }

                    // Stage 3: lane-parallel integer softmax; scores, maxima, and the exponential
                    // all stay in integer arithmetic, and the maximum position lands on exactly
                    // 255, so the weight sum can never be zero.
                    svint32_t const new_max_low_i32x = svmax_s32_x(predicate_all_b32x, running_max_low_i32x,
                                                                   panel_max_low_i32x);
                    svint32_t const new_max_high_i32x = svmax_s32_x(predicate_all_b32x, running_max_high_i32x,
                                                                    panel_max_high_i32x);
                    svfloat32_t const correction_low_f32x = nk_exp2_f32x_sme_(svmul_f32_x( // exact I32 → F32

                        predicate_all_b32x,
                        svsub_f32_x(predicate_all_b32x, svcvt_f32_s32_x(predicate_all_b32x, running_max_low_i32x),
                                    svcvt_f32_s32_x(predicate_all_b32x, new_max_low_i32x)),
                        scale2_f32x));
                    svfloat32_t const correction_high_f32x = nk_exp2_f32x_sme_(svmul_f32_x(
                        predicate_all_b32x,
                        svsub_f32_x(predicate_all_b32x, svcvt_f32_s32_x(predicate_all_b32x, running_max_high_i32x),
                                    svcvt_f32_s32_x(predicate_all_b32x, new_max_high_i32x)),
                        scale2_f32x));
                    running_max_low_i32x = new_max_low_i32x;
                    running_max_high_i32x = new_max_high_i32x;

                    svuint32_t panel_sum_low_u32x = svdup_u32(0); // weights are u8 over <= 512 positions
                    svuint32_t panel_sum_high_u32x = svdup_u32(0);
                    for (nk_size_t quad_idx = 0; quad_idx < panel_quads; quad_idx++) {
                        svuint32_t quantized_low_u32x = svdup_u32(0), quantized_high_u32x = svdup_u32(0);
                        nk_size_t const quad_length = // padded slots keep their zero bytes: weight zero
                            panel_length - quad_idx * 4 < 4 ? panel_length - quad_idx * 4 : 4;
                        for (nk_size_t position_in_quad = 0; position_in_quad < quad_length; position_in_quad++) {
                            nk_size_t const position_idx = quad_idx * 4 + position_in_quad;
                            nk_i32_t const *position_scores = scores_panel + position_idx * block_rows_capacity;
                            svint32_t const delta_low_i32x = svmax_n_s32_x(
                                predicate_all_b32x,
                                svsub_s32_x(predicate_all_b32x,
                                            svld1_s32(predicate_all_b32x, (int32_t const *)position_scores),
                                            new_max_low_i32x),
                                delta_floor);
                            svint32_t const delta_high_i32x = svmax_n_s32_x(
                                predicate_all_b32x,
                                svsub_s32_x(
                                    predicate_all_b32x,
                                    svld1_s32(predicate_all_b32x, (int32_t const *)(position_scores + tile_dimension)),
                                    new_max_high_i32x),
                                delta_floor);
                            svuint32_t weight_low_u32x = svreinterpret_u32_s32(
                                nk_exp2_u8_i32x_sme_(svmul_n_s32_x(predicate_all_b32x, delta_low_i32x, scale_fixed)));
                            svuint32_t weight_high_u32x = svreinterpret_u32_s32(
                                nk_exp2_u8_i32x_sme_(svmul_n_s32_x(predicate_all_b32x, delta_high_i32x, scale_fixed)));
                            if (panel_masked) { // hidden keys weigh zero; their possibly wrapped deltas are discarded
                                nk_size_t const position_absolute = panel_start + position_idx;
                                weight_low_u32x = svsel_u32(
                                    nk_attention_visible_sme_(key_begins_low_u32x, key_ends_low_u32x,
                                                              position_absolute),
                                    weight_low_u32x, svdup_u32(0));
                                weight_high_u32x = svsel_u32(
                                    nk_attention_visible_sme_(key_begins_high_u32x, key_ends_high_u32x,
                                                              position_absolute),
                                    weight_high_u32x, svdup_u32(0));
                            }
                            panel_sum_low_u32x = svadd_u32_x(predicate_all_b32x, panel_sum_low_u32x, weight_low_u32x);
                            panel_sum_high_u32x = svadd_u32_x(predicate_all_b32x, panel_sum_high_u32x,
                                                              weight_high_u32x);
                            quantized_low_u32x = svorr_u32_x(
                                predicate_all_b32x, quantized_low_u32x,
                                svlsl_n_u32_x(predicate_all_b32x, weight_low_u32x, (uint64_t)(position_in_quad * 8)));
                            quantized_high_u32x = svorr_u32_x(
                                predicate_all_b32x, quantized_high_u32x,
                                svlsl_n_u32_x(predicate_all_b32x, weight_high_u32x, (uint64_t)(position_in_quad * 8)));
                        }
                        svst1_u32(predicate_all_b32x, (nk_u32_t *)(weights_panel + (quad_idx * 2 + 0) * vector_bytes),
                                  quantized_low_u32x);
                        svst1_u32(predicate_all_b32x, (nk_u32_t *)(weights_panel + (quad_idx * 2 + 1) * vector_bytes),
                                  quantized_high_u32x);
                    }
                    running_sum_low_f32x = svmad_f32_x(predicate_all_b32x, running_sum_low_f32x, correction_low_f32x,
                                                       svcvt_f32_u32_x(predicate_all_b32x, panel_sum_low_u32x));
                    running_sum_high_f32x = svmad_f32_x(predicate_all_b32x, running_sum_high_f32x, correction_high_f32x,
                                                        svcvt_f32_u32_x(predicate_all_b32x, panel_sum_high_u32x));

                    nk_size_t const panel_quad_first = panel_start / 4;
                    // Stage 4: P × V as USMOPA outer products over position quads; drains convert
                    // the exact I32 totals to F32 and fuse the correction FMA.
                    for (nk_size_t channel_tile_idx = 0; channel_tile_idx < channel_tiles; channel_tile_idx += 2) {
                        int const has_second_tile = channel_tile_idx + 1 < channel_tiles;
                        nk_u8_t const *values_tile0 =
                            values_plane + (channel_tile_idx * position_quads_total + panel_quad_first) * vector_bytes;
                        nk_u8_t const *values_tile1 = values_tile0 + position_quads_total * vector_bytes;
                        svzero_za();
                        for (nk_size_t quad_idx = 0; quad_idx < panel_quads; quad_idx++) {
                            svuint8_t const weights_low_u8x = svld1_u8(
                                predicate_all_b8x, weights_panel + (quad_idx * 2 + 0) * vector_bytes);
                            svuint8_t const weights_high_u8x = svld1_u8(
                                predicate_all_b8x, weights_panel + (quad_idx * 2 + 1) * vector_bytes);
                            svint8_t const values_low_i8x = svreinterpret_s8_u8(
                                svld1_u8(predicate_all_b8x, values_tile0 + quad_idx * vector_bytes));
                            svusmopa_za32_u8_m(0, predicate_all_b8x, predicate_all_b8x, weights_low_u8x,
                                               values_low_i8x);
                            svusmopa_za32_u8_m(2, predicate_all_b8x, predicate_all_b8x, weights_high_u8x,
                                               values_low_i8x);
                            if (has_second_tile) {
                                svint8_t const values_high_i8x = svreinterpret_s8_u8(
                                    svld1_u8(predicate_all_b8x, values_tile1 + quad_idx * vector_bytes));
                                svusmopa_za32_u8_m(1, predicate_all_b8x, predicate_all_b8x, weights_low_u8x,
                                                   values_high_i8x);
                                svusmopa_za32_u8_m(3, predicate_all_b8x, predicate_all_b8x, weights_high_u8x,
                                                   values_high_i8x);
                            }
                        }
                        for (nk_size_t slice_idx = 0; slice_idx < tile_dimension; slice_idx++) {
                            nk_size_t const channel_tile0 = channel_tile_idx * tile_dimension + slice_idx;
                            nk_f32_t *accumulator_tile0 = o_acc + channel_tile0 * block_rows_capacity;
                            svfloat32_t o_tile0_low_f32x = svld1_f32(predicate_all_b32x,
                                                                     (float32_t const *)accumulator_tile0);
                            svfloat32_t o_tile0_high_f32x = svld1_f32(
                                predicate_all_b32x, (float32_t const *)(accumulator_tile0 + tile_dimension));
                            o_tile0_low_f32x = svmad_f32_x(
                                predicate_all_b32x, o_tile0_low_f32x, correction_low_f32x,
                                svcvt_f32_s32_x(
                                    predicate_all_b32x,
                                    svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 0, (uint32_t)slice_idx)));
                            o_tile0_high_f32x = svmad_f32_x(
                                predicate_all_b32x, o_tile0_high_f32x, correction_high_f32x,
                                svcvt_f32_s32_x(
                                    predicate_all_b32x,
                                    svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 2, (uint32_t)slice_idx)));
                            svst1_f32(predicate_all_b32x, (float32_t *)accumulator_tile0, o_tile0_low_f32x);
                            svst1_f32(predicate_all_b32x, (float32_t *)(accumulator_tile0 + tile_dimension),
                                      o_tile0_high_f32x);
                            if (has_second_tile) {
                                nk_size_t const channel_tile1 = (channel_tile_idx + 1) * tile_dimension + slice_idx;
                                nk_f32_t *accumulator_tile1 = o_acc + channel_tile1 * block_rows_capacity;
                                svfloat32_t o_tile1_low_f32x = svld1_f32(predicate_all_b32x,
                                                                         (float32_t const *)accumulator_tile1);
                                svfloat32_t o_tile1_high_f32x = svld1_f32(
                                    predicate_all_b32x, (float32_t const *)(accumulator_tile1 + tile_dimension));
                                o_tile1_low_f32x = svmad_f32_x(
                                    predicate_all_b32x, o_tile1_low_f32x, correction_low_f32x,
                                    svcvt_f32_s32_x(predicate_all_b32x,
                                                    svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 1,
                                                                          (uint32_t)slice_idx)));
                                o_tile1_high_f32x = svmad_f32_x(
                                    predicate_all_b32x, o_tile1_high_f32x, correction_high_f32x,
                                    svcvt_f32_s32_x(predicate_all_b32x,
                                                    svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 3,
                                                                          (uint32_t)slice_idx)));
                                svst1_f32(predicate_all_b32x, (float32_t *)accumulator_tile1, o_tile1_low_f32x);
                                svst1_f32(predicate_all_b32x, (float32_t *)(accumulator_tile1 + tile_dimension),
                                          o_tile1_high_f32x);
                            }
                        }
                    }
                }

                svfloat32_t const inverse_sum_low_f32x = svsel_f32(
                    svcmpgt_n_f32(predicate_all_b32x, running_sum_low_f32x, 0.0f),
                    svdiv_f32_x(predicate_all_b32x, svdup_f32(1.0f), running_sum_low_f32x), svdup_f32(0.0f));
                svfloat32_t const inverse_sum_high_f32x = svsel_f32(
                    svcmpgt_n_f32(predicate_all_b32x, running_sum_high_f32x, 0.0f),
                    svdiv_f32_x(predicate_all_b32x, svdup_f32(1.0f), running_sum_high_f32x), svdup_f32(0.0f));
                // Rows that saw no key keep a zero sum and emit zeros.
                // Finalize: normalize lane-wise, then transpose the channel-major accumulator back
                // to output rows through ZA0, one row-tile × channel-tile block at a time.
                for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                    nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
                    if (tile_row_start >= block_rows) break;
                    nk_size_t const rows_valid = (block_rows - tile_row_start < tile_dimension)
                                                     ? block_rows - tile_row_start
                                                     : tile_dimension;
                    svfloat32_t const inverse_sum_f32x = row_tile_idx == 0 ? inverse_sum_low_f32x
                                                                           : inverse_sum_high_f32x;
                    nk_size_t const first_token = query_first + row_block_start + tile_row_start;
                    if (log_sum_exp) {
                        nk_i32_t row_maxima[nk_attention_max_tile_sme_k_];
                        nk_f32_t row_sums[nk_attention_max_tile_sme_k_];
                        svst1_s32(predicate_all_b32x, row_maxima,
                                  row_tile_idx == 0 ? running_max_low_i32x : running_max_high_i32x);
                        svst1_f32(predicate_all_b32x, row_sums,
                                  row_tile_idx == 0 ? running_sum_low_f32x : running_sum_high_f32x);
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_valid; row_in_tile++)
                            log_sum_exp[(first_token + row_in_tile) * head_count + head_idx] =
                                nk_attention_log_sum_exp_serial_((nk_f32_t)row_maxima[row_in_tile] * scale2,
                                                                 row_sums[row_in_tile] / 255.0f);
                    }
                    for (nk_size_t channel_tile_idx = 0; channel_tile_idx < channel_tiles; channel_tile_idx++) {
                        nk_size_t const channel_start = channel_tile_idx * tile_dimension;
                        svbool_t const channel_predicate_b32x = svwhilelt_b32_u64(channel_start, depth);
                        svzero_mask_za(nk_sme_zero_za32_tile_0_k);
                        for (nk_size_t channel_in_tile = 0; channel_in_tile < tile_dimension; channel_in_tile++) {
                            svfloat32_t normalized_f32x = svmul_f32_x(
                                predicate_all_b32x,
                                svld1_f32(predicate_all_b32x,
                                          (float32_t const *)(o_acc +
                                                              (channel_start + channel_in_tile) * block_rows_capacity +
                                                              tile_row_start)),
                                inverse_sum_f32x);
                            svwrite_hor_za32_f32_m(0, (uint32_t)channel_in_tile, predicate_all_b32x, normalized_f32x);
                        }
                        for (nk_size_t row_in_tile = 0; row_in_tile < rows_valid; row_in_tile++) {
                            nk_f32_t *output_row = output + (first_token + row_in_tile) * output_stride_floats +
                                                   head_idx * depth + channel_start;
                            svst1_ver_za32(0, (uint32_t)row_in_tile, channel_predicate_b32x, output_row);
                        }
                    }
                }
            }
        }
    }
}

#if NUMKONG_TARGET_SME

NUMKONG_API nk_status_t nk_attention_pack_size_bf16_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes) {
    // Shapes outside the tile fast-path envelope route to the width-agnostic serial kernel;
    // the rule is a pure function of the arguments and the machine, so pack and attention agree.
    *bytes = depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   depth * sizeof(nk_f32_t))
                 : nk_attention_pack_size_b16_sme_(key_value_head_count, depth, token_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_bf16_sme(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_bf16_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_bf16_t const *keys, nk_size_t key_stride,
                                                   nk_bf16_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_bf16_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                       segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                       tasks_end, nk_cap_sme_k);
    }
    else {
        nk_attention_pack_directory_serial_(
            key_value_packed, key_value_head_count, depth, key_offsets, key_lengths, segment_count, tasks_begin,
            nk_cnth_sme_(), nk_size_round_up_to_multiple_(depth, nk_cntw_sme_()) * sizeof(nk_u16_t), nk_cap_sme_k);
        nk_start_sme_streaming_();
        nk_attention_pack_b16_sme_streaming_(keys, values, sizeof(nk_bf16_t), key_value_head_count, depth, key_offsets,
                                             key_lengths, segment_count, key_stride, value_stride, key_value_packed,
                                             tasks_begin, tasks_end);
        nk_stop_sme_streaming_();
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_e4m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_size_t token_count, nk_size_t segment_count,
                                                        nk_size_t *bytes) {
    *bytes = depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   depth * sizeof(nk_f32_t))
                 : nk_attention_pack_size_b16_sme_(key_value_head_count, depth, token_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_e4m3_sme(void const *key_value_packed,
                                                           nk_size_t *key_value_head_count, nk_size_t *depth,
                                                           nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_e4m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                   nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                   nk_size_t segment_count, nk_e4m3_t const *keys, nk_size_t key_stride,
                                                   nk_e4m3_t const *values, nk_size_t value_stride,
                                                   void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_e4m3_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                       segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                       tasks_end, nk_cap_sme_k);
    }
    else {
        nk_attention_pack_directory_serial_(
            key_value_packed, key_value_head_count, depth, key_offsets, key_lengths, segment_count, tasks_begin,
            nk_cnth_sme_(), nk_size_round_up_to_multiple_(depth, nk_cntw_sme_()) * sizeof(nk_u16_t), nk_cap_sme_k);
        nk_start_sme_streaming_();
        nk_attention_pack_b16_sme_streaming_(keys, values, sizeof(nk_e4m3_t), key_value_head_count, depth, key_offsets,
                                             key_lengths, segment_count, key_stride, value_stride, key_value_packed,
                                             tasks_begin, tasks_end);
        nk_stop_sme_streaming_();
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_bf16_sme(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_bf16_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_bf16_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                         key_value_head_count, depth, query_offsets, query_stride, output_stride, scale,
                                         band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_start_sme_streaming_();
    nk_attention_packed_b16_sme_streaming_(queries, sizeof(nk_bf16_t), key_value_packed, output, log_sum_exp,
                                           head_count, key_value_head_count, depth, query_offsets, query_stride,
                                           output_stride, scale, band, tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_e4m3_sme(nk_size_t head_count, nk_size_t key_value_head_count,
                                                     nk_size_t depth, nk_u32_t const *query_offsets,
                                                     nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                     nk_size_t keys_after, nk_e4m3_t const *queries,
                                                     nk_size_t query_stride, void const *key_value_packed,
                                                     nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                     nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_e4m3_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                         key_value_head_count, depth, query_offsets, query_stride, output_stride, scale,
                                         band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_start_sme_streaming_();
    nk_attention_packed_b16_sme_streaming_(queries, sizeof(nk_e4m3_t), key_value_packed, output, log_sum_exp,
                                           head_count, key_value_head_count, depth, query_offsets, query_stride,
                                           output_stride, scale, band, tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_f16_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_size_t token_count, nk_size_t segment_count,
                                                       nk_size_t *bytes) {
    *bytes = depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   depth * sizeof(nk_f32_t))
                 : nk_attention_pack_size_b16_sme_(key_value_head_count, depth, token_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_f16_sme(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                          nk_size_t *depth, nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_f16_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                  nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                  nk_size_t segment_count, nk_f16_t const *keys, nk_size_t key_stride,
                                                  nk_f16_t const *values, nk_size_t value_stride,
                                                  void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end,
                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_f16_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                      segment_count, key_stride, value_stride, key_value_packed, tasks_begin, tasks_end,
                                      nk_cap_sme_k);
    }
    else {
        nk_attention_pack_directory_serial_(
            key_value_packed, key_value_head_count, depth, key_offsets, key_lengths, segment_count, tasks_begin,
            nk_cnth_sme_(), nk_size_round_up_to_multiple_(depth, nk_cntw_sme_()) * sizeof(nk_u16_t), nk_cap_sme_k);
        nk_start_sme_streaming_();
        nk_attention_pack_b16_sme_streaming_(keys, values, sizeof(nk_f16_t), key_value_head_count, depth, key_offsets,
                                             key_lengths, segment_count, key_stride, value_stride, key_value_packed,
                                             tasks_begin, tasks_end);
        nk_stop_sme_streaming_();
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_f16_sme(nk_size_t head_count, nk_size_t key_value_head_count,
                                                    nk_size_t depth, nk_u32_t const *query_offsets,
                                                    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                    nk_size_t keys_after, nk_f16_t const *queries,
                                                    nk_size_t query_stride, void const *key_value_packed,
                                                    nk_f32_t *output, nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_f16_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                        key_value_head_count, depth, query_offsets, query_stride, output_stride, scale,
                                        band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_start_sme_streaming_();
    nk_attention_packed_f16_sme_streaming_(queries, key_value_packed, output, log_sum_exp, head_count,
                                           key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                           scale, band, tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_nvfp4_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes) {
    *bytes = depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                   depth * sizeof(nk_f32_t))
                 : nk_attention_pack_size_b16_sme_(key_value_head_count, depth, token_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_nvfp4_sme(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_nvfp4_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_nvfp4_cref_t const *keys,
                                                    nk_size_t key_stride, nk_nvfp4_cref_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_nvfp4_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                        tasks_end, nk_cap_sme_k);
        return nk_success_k;
    }
    nk_attention_pack_directory_serial_(
        key_value_packed, key_value_head_count, depth, key_offsets, key_lengths, segment_count, tasks_begin,
        nk_cnth_sme_(), nk_size_round_up_to_multiple_(depth, nk_cntw_sme_()) * sizeof(nk_u16_t), nk_cap_sme_k);
    if (tasks_begin == 0) {
        nk_attention_packed_header_t *header = (nk_attention_packed_header_t *)key_value_packed;
        header->key_tensor_scale = nk_cross_tensor_scale_serial_(keys->tensor_scale);
        header->value_tensor_scale = nk_cross_tensor_scale_serial_(values->tensor_scale);
    }
    nk_cross_operand_t const key_operand = nk_cross_operand_serial_(nk_nvfp4_k, keys, key_stride);
    nk_cross_operand_t const value_operand = nk_cross_operand_serial_(nk_nvfp4_k, values, value_stride);
    nk_start_sme_streaming_();
    nk_attention_pack_nvfp4_sme_streaming_(key_operand, value_operand, key_value_head_count, depth, key_offsets,
                                           key_lengths, segment_count, key_stride, value_stride, key_value_packed,
                                           tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_nvfp4_sme(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_nvfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_nvfp4_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                          key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                          scale, band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_cross_tensor_factor_t const factor = nk_cross_tensor_factor_serial_(
        nk_cross_tensor_scale_serial_(queries->tensor_scale), header->key_tensor_scale);
    nk_f32_t const scale2 = nk_scale_f32_serial_(scale * NUMKONG_F32_LOG2E_ * factor.mantissa, factor.exponent);
    nk_cross_operand_t const query_operand = nk_cross_operand_serial_(nk_nvfp4_k, queries, query_stride);
    nk_start_sme_streaming_();
    nk_attention_packed_nvfp4_sme_streaming_(query_operand, key_value_packed, output, log_sum_exp, head_count,
                                             key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                             scale2, header->value_tensor_scale, band, tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp4_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                         nk_size_t token_count, nk_size_t segment_count,
                                                         nk_size_t *bytes) {
    // Tile planes and their exponent table, or serial's F32 planes and table past the tile envelope
    *bytes = (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_
                  ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                    depth * sizeof(nk_f32_t))
                  : nk_attention_pack_size_b16_sme_(key_value_head_count, depth, token_count, segment_count)) +
             nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp4_sme(void const *key_value_packed,
                                                            nk_size_t *key_value_head_count, nk_size_t *depth,
                                                            nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp4_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                    nk_size_t segment_count, nk_mxfp4_cref_t const *keys,
                                                    nk_size_t key_stride, nk_mxfp4_cref_t const *values,
                                                    nk_size_t value_stride, void *key_value_packed,
                                                    nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_mxfp4_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                        segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                        tasks_end, nk_cap_sme_k);
        return nk_success_k;
    }
    nk_attention_pack_directory_serial_(
        key_value_packed, key_value_head_count, depth, key_offsets, key_lengths, segment_count, tasks_begin,
        nk_cnth_sme_(), nk_size_round_up_to_multiple_(depth, nk_cntw_sme_()) * sizeof(nk_u16_t), nk_cap_sme_k);
    nk_cross_operand_t const key_operand = nk_cross_operand_serial_(nk_mxfp4_k, keys, key_stride);
    nk_cross_operand_t const value_operand = nk_cross_operand_serial_(nk_mxfp4_k, values, value_stride);
    nk_start_sme_streaming_();
    nk_attention_pack_mxfp4_sme_streaming_(key_operand, value_operand, key_value_head_count, depth, key_offsets,
                                           key_lengths, segment_count, key_stride, value_stride, key_value_packed,
                                           tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp4_sme(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp4_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_size_t const tile_dimension = nk_cntw_sme_();
    if (depth > nk_attention_max_depth_sme_k_ || tile_dimension > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_mxfp4_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                          key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                          scale, band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;
    nk_cross_operand_t const query_operand = nk_cross_operand_serial_(nk_mxfp4_k, queries, query_stride);
    nk_start_sme_streaming_();
    nk_attention_packed_mxfp4_sme_streaming_(query_operand, key_value_packed, output, log_sum_exp, head_count,
                                             key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                             scale2, band, tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    nk_attention_packed_exact_mxfp4_sme_(query_operand, key_value_packed, output, log_sum_exp, head_count,
                                         key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                         scale2, band, tile_dimension, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e2m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_size_t token_count, nk_size_t segment_count,
                                                             nk_size_t *bytes) {
    *bytes = (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_
                  ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                    depth * sizeof(nk_f32_t))
                  : nk_attention_pack_size_b16_sme_(key_value_head_count, depth, token_count, segment_count)) +
             nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e2m3_sme(void const *key_value_packed,
                                                                nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp6e2m3_sme(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp6e2m3_cref_t const *keys, nk_size_t key_stride, nk_mxfp6e2m3_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_mxfp6e2m3_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                            segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                            tasks_end, nk_cap_sme_k);
        return nk_success_k;
    }
    nk_attention_pack_directory_serial_(
        key_value_packed, key_value_head_count, depth, key_offsets, key_lengths, segment_count, tasks_begin,
        nk_cnth_sme_(), nk_size_round_up_to_multiple_(depth, nk_cntw_sme_()) * sizeof(nk_u16_t), nk_cap_sme_k);
    nk_cross_operand_t const key_operand = nk_cross_operand_serial_(nk_mxfp6e2m3_k, keys, key_stride);
    nk_cross_operand_t const value_operand = nk_cross_operand_serial_(nk_mxfp6e2m3_k, values, value_stride);
    nk_start_sme_streaming_();
    nk_attention_pack_mxfp6e2m3_sme_streaming_(key_operand, value_operand, key_value_head_count, depth, key_offsets,
                                               key_lengths, segment_count, key_stride, value_stride, key_value_packed,
                                               tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp6e2m3_sme(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e2m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_size_t const tile_dimension = nk_cntw_sme_();
    if (depth > nk_attention_max_depth_sme_k_ || tile_dimension > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_mxfp6e2m3_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                              key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                              scale, band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;
    nk_cross_operand_t const query_operand = nk_cross_operand_serial_(nk_mxfp6e2m3_k, queries, query_stride);
    nk_start_sme_streaming_();
    nk_attention_packed_mxfp6e2m3_sme_streaming_(query_operand, key_value_packed, output, log_sum_exp, head_count,
                                                 key_value_head_count, depth, query_offsets, query_stride,
                                                 output_stride, scale2, band, tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    nk_attention_packed_exact_mxfp6e2m3_sme_(query_operand, key_value_packed, output, log_sum_exp, head_count,
                                             key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                             scale2, band, tile_dimension, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp6e3m2_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_size_t token_count, nk_size_t segment_count,
                                                             nk_size_t *bytes) {
    *bytes = (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_
                  ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                    depth * sizeof(nk_f32_t))
                  : nk_attention_pack_size_b16_sme_(key_value_head_count, depth, token_count, segment_count)) +
             nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp6e3m2_sme(void const *key_value_packed,
                                                                nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp6e3m2_sme(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp6e3m2_cref_t const *keys, nk_size_t key_stride, nk_mxfp6e3m2_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_mxfp6e3m2_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                            segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                            tasks_end, nk_cap_sme_k);
        return nk_success_k;
    }
    nk_attention_pack_directory_serial_(
        key_value_packed, key_value_head_count, depth, key_offsets, key_lengths, segment_count, tasks_begin,
        nk_cnth_sme_(), nk_size_round_up_to_multiple_(depth, nk_cntw_sme_()) * sizeof(nk_u16_t), nk_cap_sme_k);
    nk_cross_operand_t const key_operand = nk_cross_operand_serial_(nk_mxfp6e3m2_k, keys, key_stride);
    nk_cross_operand_t const value_operand = nk_cross_operand_serial_(nk_mxfp6e3m2_k, values, value_stride);
    nk_start_sme_streaming_();
    nk_attention_pack_mxfp6e3m2_sme_streaming_(key_operand, value_operand, key_value_head_count, depth, key_offsets,
                                               key_lengths, segment_count, key_stride, value_stride, key_value_packed,
                                               tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp6e3m2_sme(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp6e3m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_size_t const tile_dimension = nk_cntw_sme_();
    if (depth > nk_attention_max_depth_sme_k_ || tile_dimension > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_mxfp6e3m2_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                              key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                              scale, band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;
    nk_cross_operand_t const query_operand = nk_cross_operand_serial_(nk_mxfp6e3m2_k, queries, query_stride);
    nk_start_sme_streaming_();
    nk_attention_packed_mxfp6e3m2_sme_streaming_(query_operand, key_value_packed, output, log_sum_exp, head_count,
                                                 key_value_head_count, depth, query_offsets, query_stride,
                                                 output_stride, scale2, band, tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    nk_attention_packed_exact_mxfp6e3m2_sme_(query_operand, key_value_packed, output, log_sum_exp, head_count,
                                             key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                             scale2, band, tile_dimension, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e4m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_size_t token_count, nk_size_t segment_count,
                                                             nk_size_t *bytes) {
    *bytes = (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_
                  ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                    depth * sizeof(nk_f32_t))
                  : nk_attention_pack_size_b16_sme_(key_value_head_count, depth, token_count, segment_count)) +
             nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e4m3_sme(void const *key_value_packed,
                                                                nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp8e4m3_sme(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp8e4m3_cref_t const *keys, nk_size_t key_stride, nk_mxfp8e4m3_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_mxfp8e4m3_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                            segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                            tasks_end, nk_cap_sme_k);
        return nk_success_k;
    }
    nk_attention_pack_directory_serial_(
        key_value_packed, key_value_head_count, depth, key_offsets, key_lengths, segment_count, tasks_begin,
        nk_cnth_sme_(), nk_size_round_up_to_multiple_(depth, nk_cntw_sme_()) * sizeof(nk_u16_t), nk_cap_sme_k);
    nk_cross_operand_t const key_operand = nk_cross_operand_serial_(nk_mxfp8e4m3_k, keys, key_stride);
    nk_cross_operand_t const value_operand = nk_cross_operand_serial_(nk_mxfp8e4m3_k, values, value_stride);
    nk_start_sme_streaming_();
    nk_attention_pack_mxfp8e4m3_sme_streaming_(key_operand, value_operand, key_value_head_count, depth, key_offsets,
                                               key_lengths, segment_count, key_stride, value_stride, key_value_packed,
                                               tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp8e4m3_sme(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e4m3_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_size_t const tile_dimension = nk_cntw_sme_();
    if (depth > nk_attention_max_depth_sme_k_ || tile_dimension > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_mxfp8e4m3_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                              key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                              scale, band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;
    nk_cross_operand_t const query_operand = nk_cross_operand_serial_(nk_mxfp8e4m3_k, queries, query_stride);
    nk_start_sme_streaming_();
    nk_attention_packed_mxfp8e4m3_sme_streaming_(query_operand, key_value_packed, output, log_sum_exp, head_count,
                                                 key_value_head_count, depth, query_offsets, query_stride,
                                                 output_stride, scale2, band, tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    nk_attention_packed_exact_mxfp8e4m3_sme_(query_operand, key_value_packed, output, log_sum_exp, head_count,
                                             key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                             scale2, band, tile_dimension, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_mxfp8e5m2_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                             nk_size_t token_count, nk_size_t segment_count,
                                                             nk_size_t *bytes) {
    *bytes = (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_
                  ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1,
                                                    depth * sizeof(nk_f32_t))
                  : nk_attention_pack_size_b16_sme_(key_value_head_count, depth, token_count, segment_count)) +
             nk_attention_plane_exponents_size_serial_(key_value_head_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_mxfp8e5m2_sme(void const *key_value_packed,
                                                                nk_size_t *key_value_head_count, nk_size_t *depth,
                                                                nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_mxfp8e5m2_sme(
    nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
    nk_size_t segment_count, nk_mxfp8e5m2_cref_t const *keys, nk_size_t key_stride, nk_mxfp8e5m2_cref_t const *values,
    nk_size_t value_stride, void *key_value_packed, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (depth % 32 != 0) return nk_unexpected_dimensions_k;
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_mxfp8e5m2_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                            segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                            tasks_end, nk_cap_sme_k);
        return nk_success_k;
    }
    nk_attention_pack_directory_serial_(
        key_value_packed, key_value_head_count, depth, key_offsets, key_lengths, segment_count, tasks_begin,
        nk_cnth_sme_(), nk_size_round_up_to_multiple_(depth, nk_cntw_sme_()) * sizeof(nk_u16_t), nk_cap_sme_k);
    nk_cross_operand_t const key_operand = nk_cross_operand_serial_(nk_mxfp8e5m2_k, keys, key_stride);
    nk_cross_operand_t const value_operand = nk_cross_operand_serial_(nk_mxfp8e5m2_k, values, value_stride);
    nk_start_sme_streaming_();
    nk_attention_pack_mxfp8e5m2_sme_streaming_(key_operand, value_operand, key_value_head_count, depth, key_offsets,
                                               key_lengths, segment_count, key_stride, value_stride, key_value_packed,
                                               tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_mxfp8e5m2_sme(
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth, nk_u32_t const *query_offsets,
    nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before, nk_size_t keys_after,
    nk_mxfp8e5m2_cref_t const *queries, nk_size_t query_stride, void const *key_value_packed, nk_f32_t *output,
    nk_size_t output_stride, nk_f32_t *log_sum_exp, nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    nk_size_t const tile_dimension = nk_cntw_sme_();
    if (depth > nk_attention_max_depth_sme_k_ || tile_dimension > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_mxfp8e5m2_serial_(queries, key_value_packed, output, log_sum_exp, head_count,
                                              key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                              scale, band, tasks_begin, tasks_end);
        return nk_success_k;
    }
    nk_f32_t const scale2 = scale * NUMKONG_F32_LOG2E_;
    nk_cross_operand_t const query_operand = nk_cross_operand_serial_(nk_mxfp8e5m2_k, queries, query_stride);
    nk_start_sme_streaming_();
    nk_attention_packed_mxfp8e5m2_sme_streaming_(query_operand, key_value_packed, output, log_sum_exp, head_count,
                                                 key_value_head_count, depth, query_offsets, query_stride,
                                                 output_stride, scale2, band, tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    nk_attention_packed_exact_mxfp8e5m2_sme_(query_operand, key_value_packed, output, log_sum_exp, head_count,
                                             key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                             scale2, band, tile_dimension, tasks_begin, tasks_end);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_size_i8_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_size_t token_count, nk_size_t segment_count,
                                                      nk_size_t *bytes) {
    *bytes = depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_
                 ? nk_attention_pack_bound_serial_(key_value_head_count, token_count, segment_count, 1, depth)
                 : nk_attention_pack_size_b8_sme_(key_value_head_count, depth, token_count, segment_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_shape_i8_sme(void const *key_value_packed, nk_size_t *key_value_head_count,
                                                         nk_size_t *depth, nk_size_t *segments, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_attention_packed_shape_serial_(key_value_packed, key_value_head_count, depth, segments);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_pack_i8_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                 nk_u32_t const *key_offsets, nk_u32_t const *key_lengths,
                                                 nk_size_t segment_count, nk_i8_t const *keys, nk_size_t key_stride,
                                                 nk_i8_t const *values, nk_size_t value_stride, void *key_value_packed,
                                                 nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const validation = nk_attention_pack_validate_serial_(key_value_head_count, depth, key_offsets,
                                                                      key_lengths, segment_count);
    if (validation != nk_success_k) return validation;
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_i8_serial_(keys, values, key_value_head_count, depth, key_offsets, key_lengths, segment_count,
                                     key_stride, value_stride, key_value_packed, tasks_begin, tasks_end, nk_cap_sme_k);
    }
    else {
        nk_attention_pack_directory_serial_(key_value_packed, key_value_head_count, depth, key_offsets, key_lengths,
                                            segment_count, tasks_begin, nk_cnth_sme_(),
                                            nk_size_round_up_to_multiple_(depth, nk_cntw_sme_()), nk_cap_sme_k);
        nk_start_sme_streaming_();
        nk_attention_pack_i8_sme_streaming_(keys, values, key_value_head_count, depth, key_offsets, key_lengths,
                                            segment_count, key_stride, value_stride, key_value_packed, tasks_begin,
                                            tasks_end);
        nk_stop_sme_streaming_();
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_attention_packed_i8_sme(nk_size_t head_count, nk_size_t key_value_head_count,
                                                   nk_size_t depth, nk_u32_t const *query_offsets,
                                                   nk_size_t query_token_count, nk_f32_t scale, nk_size_t keys_before,
                                                   nk_size_t keys_after, nk_i8_t const *queries, nk_size_t query_stride,
                                                   void const *key_value_packed, nk_f32_t *output,
                                                   nk_size_t output_stride, nk_f32_t *log_sum_exp,
                                                   nk_size_t tasks_begin, nk_size_t tasks_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!nk_attention_packed_by_serial_(key_value_packed, nk_cap_sme_k)) return nk_pack_mismatch_k;
    nk_assert_(query_token_count == nk_attention_query_end_serial_(key_value_packed, query_offsets));
    nk_diagonal_band_t const band = {keys_before, keys_after};
    if (depth > nk_attention_max_depth_sme_k_ || nk_cntw_sme_() > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_i8_serial_(queries, key_value_packed, output, log_sum_exp, head_count, key_value_head_count,
                                       depth, query_offsets, query_stride, output_stride, scale, band, tasks_begin,
                                       tasks_end);
        return nk_success_k;
    }
    nk_start_sme_streaming_();
    nk_attention_packed_i8_sme_streaming_(queries, key_value_packed, output, log_sum_exp, head_count,
                                          key_value_head_count, depth, query_offsets, query_stride, output_stride,
                                          scale, band, tasks_begin, tasks_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SME

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_SME_
#endif // NUMKONG_ARCH_ARM64_

#endif // NUMKONG_ATTENTION_SME_H
