/**
 *  @brief Arm SME ragged attention backend.
 *  @file include/numkong/attention/sme.h
 *  @author Ash Vardanian
 *  @date July 8, 2026
 *
 *  @sa include/numkong/attention.h
 *
 *  FlashAttention-style panel sweep on the SME outer-product engine, mirroring the
 *  `sapphireamx` skeleton with the shared packed header, segment directory, base-2
 *  streaming softmax, and `(first_task, task_count)` windows — but restructured around
 *  three Arm-specific properties measured on Apple M5:
 *
 *  1. Streaming mode is entered once per public call and never left: matrix work runs
 *      as widening MOPA outer products into ZA32 tiles, and the softmax stays on the
 *      streaming SVE vector unit, so there are no SMSTART/SMSTOP round-trips and no
 *      NEON excursions inside the hot loop.
 *  2. ZA slices move in both directions: vertical reads give a free transpose, so the
 *      score panel is stored position-major with one query per lane — the running
 *      maximum, corrections, weight sums, output rescaling, and normalization are all
 *      plain lane-parallel vector ops with no horizontal reductions or per-row scalar
 *      broadcasts anywhere in the pipeline.
 *  3. Probabilities cross from F32 scores to MOPA-ready pair-interleaved BF16 operands
 *      in registers (round + `TRN2`), replacing the AMX tile-loader's mandatory memory
 *      round-trip with a single in-register shuffle per position pair.
 *
 *  ZA tile roles per stage (SVL = 512: four 16×16 F32 tiles):
 *
 *  - Q staging & output transpose: ZA0 horizontal-write / vertical-read (pair interleave)
 *  - Q×Kᵀ scores: ZA0-ZA3 = (two query row-tiles) × (two K position-tiles)
 *  - P×V: ZA0-ZA3 = (two probability row-tiles) × (two V channel-tiles)
 *
 *  Widening MOPA keeps every reduction in F32 accumulators: the non-widening ZA16
 *  forms measure ~2× faster but lose ~12% relative accuracy on signed depth-256
 *  reductions, which fails the family's F32-accumulator contract.
 */
#ifndef NK_ATTENTION_SME_H
#define NK_ATTENTION_SME_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SME

#include <arm_sme.h>

#include "numkong/types.h"
#include "numkong/attention/serial.h" // `nk_attention_packed_header_t`, `nk_attention_pack_directory_`
#include "numkong/dots/sme.h"         // `nk_sme_zero_za32_tile_0_` and the ZA transpose pack idiom

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
    /** KV panel width in positions; the position-major F32 score panel (64 KB) stays L2-resident. */
    nk_attention_panel_sme_k_ = 512,
    /** Widest head this backend handles in tiles; larger heads route to the serial tier. */
    nk_attention_max_depth_sme_k_ = 256,
    /** Widest ZA32 tile dimension the stack scratch is sized for (SVL ≤ 512); larger routes to serial. */
    nk_attention_max_tile_sme_k_ = 16,
};

/** @brief Fast vectorized 2^x: exact range reduction + the family's shared degree-4 polynomial. */
NK_INTERNAL svfloat32_t nk_attention_exp2_f32x_sme_(svfloat32_t x_f32x) NK_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    x_f32x = svmax_f32_x(predicate_all_b32x, svmin_f32_x(predicate_all_b32x, x_f32x, svdup_f32(127.0f)),
                         svdup_f32(-125.0f));
    svfloat32_t const n_f32x = svrintn_f32_x(predicate_all_b32x, x_f32x);
    svfloat32_t const r_f32x = svsub_f32_x(predicate_all_b32x, x_f32x, n_f32x);
    svfloat32_t p_f32x = svdup_f32(9.61812910e-3f);
    p_f32x = svmad_f32_x(predicate_all_b32x, p_f32x, r_f32x, svdup_f32(5.55041087e-2f));
    p_f32x = svmad_f32_x(predicate_all_b32x, p_f32x, r_f32x, svdup_f32(2.40226507e-1f));
    p_f32x = svmad_f32_x(predicate_all_b32x, p_f32x, r_f32x, svdup_f32(6.93147181e-1f));
    p_f32x = svmad_f32_x(predicate_all_b32x, p_f32x, r_f32x, svdup_f32(1.0f));
    svint32_t n_i32x = svcvt_s32_f32_x(predicate_all_b32x, n_f32x);
    n_i32x = svlsl_n_s32_x(predicate_all_b32x, svadd_n_s32_x(predicate_all_b32x, n_i32x, 127), 23);
    return svmul_f32_x(predicate_all_b32x, p_f32x, svreinterpret_f32_s32(n_i32x));
}

/**
 *  @brief Rounds two F32 weight vectors to BF16 and interleaves them pair-wise in one `TRN2`:
 *         lane `2i` gets `even[i]`, lane `2i + 1` gets `odd[i]` — the exact widening-BFMOPA
 *         operand layout, produced without any memory round-trip.
 */
NK_INTERNAL svuint16_t nk_attention_bf16_pair_sme_(svfloat32_t even_f32x, svfloat32_t odd_f32x) NK_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svuint32_t even_u32x = svreinterpret_u32_f32(even_f32x);
    svuint32_t odd_u32x = svreinterpret_u32_f32(odd_f32x);
    // Round to nearest even: add 0x7FFF plus the lowest surviving mantissa bit; weights are
    // positive finite, so no NaN patching is needed.
    even_u32x = svadd_u32_x(predicate_all_b32x, svadd_n_u32_x(predicate_all_b32x, even_u32x, 0x7FFF),
                            svand_n_u32_x(predicate_all_b32x, svlsr_n_u32_x(predicate_all_b32x, even_u32x, 16), 1));
    odd_u32x = svadd_u32_x(predicate_all_b32x, svadd_n_u32_x(predicate_all_b32x, odd_u32x, 0x7FFF),
                           svand_n_u32_x(predicate_all_b32x, svlsr_n_u32_x(predicate_all_b32x, odd_u32x, 16), 1));
    return svtrn2_u16(svreinterpret_u16_u32(even_u32x), svreinterpret_u16_u32(odd_u32x));
}

/**
 *  @brief Widens E4M3 bytes to their exact BF16 representations: every E4M3 value (3-bit
 *         mantissa, ±448 range) is exactly representable in BF16, so the F16 hop through the
 *         `dots` converter and the final narrowing round are lossless.
 */
NK_INTERNAL svuint16_t nk_attention_e4m3_to_bf16_sme_(svbool_t predicate_b16x, svuint8_t bytes_u8x) NK_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svfloat16_t const halves_f16x = nk_e4m3x_to_f16x_ssve_(predicate_b16x, bytes_u8x);
    svfloat32_t const even_f32x = svcvt_f32_f16_x(predicate_all_b32x, halves_f16x);
    svfloat32_t const odd_f32x = svcvt_f32_f16_x(
        predicate_all_b32x,
        svreinterpret_f16_u32(svlsr_n_u32_x(predicate_all_b32x, svreinterpret_u32_f16(halves_f16x), 16)));
    return nk_attention_bf16_pair_sme_(even_f32x, odd_f32x);
}

/**
 *  @brief Degree-3 evaluation of `2^r` over the reduced fraction `r ∈ [-0.5, 0.5]`, 32 lanes
 *         at a time: the family coefficients with the degree-4 term dropped, which falls
 *         below the F16 resolution of the probability quantization this feeds. Callers keep
 *         the range reduction and the power-of-two application in F32, so the F16 rounding
 *         touches only the bounded fraction and the weight error stays near 1e-3 regardless
 *         of the argument magnitude.
 */
NK_INTERNAL svfloat16_t nk_attention_exp2_fraction_f16x_sme_(svfloat16_t reduced_f16x) NK_STREAMING_ {
    svbool_t const predicate_all_b16x = svptrue_b16();
    svfloat16_t poly_f16x = svdup_f16((__fp16)5.55041087e-2f);
    poly_f16x = svmad_f16_x(predicate_all_b16x, poly_f16x, reduced_f16x, svdup_f16((__fp16)2.40226507e-1f));
    poly_f16x = svmad_f16_x(predicate_all_b16x, poly_f16x, reduced_f16x, svdup_f16((__fp16)6.93147181e-1f));
    poly_f16x = svmad_f16_x(predicate_all_b16x, poly_f16x, reduced_f16x, svdup_f16((__fp16)1.0f));
    return poly_f16x;
}

/**
 *  @brief I-BERT-style integer exponential for the I8 weight path: takes the base-2 argument
 *         as a Q15 fixed-point value in `[-10·2^15, 0]` and returns `round(2^t · 255)` as U8
 *         weights in I32 lanes. A degree-3 fixed-point polynomial covers the fraction (error
 *         ~0.03 of a weight step) and a lane-variable shift applies the integer part, so no
 *         floating-point instruction touches the weights at all.
 */
NK_INTERNAL svint32_t nk_attention_iexp2_weight_i32x_sme_(svint32_t t_q15_i32x) NK_STREAMING_ {
    svbool_t const predicate_all_b32x = svptrue_b32();
    svint32_t const whole_i32x = svasr_n_s32_x(predicate_all_b32x, t_q15_i32x, 15); // floor, in [-10, 0]
    svint32_t const fraction_i32x = svand_n_s32_x(predicate_all_b32x, t_q15_i32x, 0x7FFF);
    svint32_t poly_i32x = svdup_s32(1296); // Chebyshev-fit 2^r coefficients in Q14, degree 3
    poly_i32x = svadd_n_s32_x(
        predicate_all_b32x,
        svasr_n_s32_x(predicate_all_b32x, svmul_s32_x(predicate_all_b32x, fraction_i32x, poly_i32x), 15), 3678);
    poly_i32x = svadd_n_s32_x(
        predicate_all_b32x,
        svasr_n_s32_x(predicate_all_b32x, svmul_s32_x(predicate_all_b32x, fraction_i32x, poly_i32x), 15), 11410);
    poly_i32x = svadd_n_s32_x(
        predicate_all_b32x,
        svasr_n_s32_x(predicate_all_b32x, svmul_s32_x(predicate_all_b32x, fraction_i32x, poly_i32x), 15), 16382);
    svint32_t const scaled_i32x = svmul_n_s32_x(predicate_all_b32x, poly_i32x, 255); // Q14 of 2^r · 255
    svuint32_t const shift_u32x = svreinterpret_u32_s32(svsubr_n_s32_x(predicate_all_b32x, whole_i32x, 14));
    svint32_t const bias_i32x = svlsl_s32_x(predicate_all_b32x, svdup_s32(1),
                                            svreinterpret_u32_s32(svsubr_n_s32_x(predicate_all_b32x, whole_i32x, 13)));
    return svasr_s32_x(predicate_all_b32x, svadd_s32_x(predicate_all_b32x, scaled_i32x, bias_i32x), shift_u32x);
}

NK_INTERNAL nk_size_t nk_attention_packed_size_b16_sme_(nk_size_t key_value_head_count, nk_size_t depth,
                                                        nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    nk_size_t const tile_dimension = nk_sme_cntw_();
    nk_size_t const vector_elements = nk_sme_cnth_();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t payload_bytes = 0;
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++) {
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(segment_lengths[segment_idx],
                                                                              vector_elements);
        payload_bytes += 2 * key_value_head_count * position_count_padded * depth_padded * sizeof(nk_u16_t); // K + V
    }
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_PUBLIC nk_size_t nk_attention_packed_size_bf16_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    // Shapes outside the tile fast-path envelope route to the width-agnostic serial tier;
    // the rule is a pure function of the arguments and the machine, so pack and attention agree.
    if (depth > nk_attention_max_depth_sme_k_ || nk_sme_cntw_() > nk_attention_max_tile_sme_k_)
        return nk_attention_packed_size_bf16_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_b16_sme_(key_value_head_count, depth, segment_lengths, segment_count);
}

/**
 *  @brief Streaming pack core for 16-bit K/V planes.
 *
 *  K becomes pair-interleaved MOPA operand vectors `[position_tile][depth_pair]` through the
 *  ZA0 horizontal-write / vertical-read transpose (the `dots` packer idiom); V becomes
 *  transposed pair-interleaved vectors `[channel_tile][position_pair]` through one `ZIP1` per
 *  position pair, so P×V runs as outer products over positions. Rows beyond the segment and
 *  channels beyond the head are zero-filled by the predicated loads and the ZA0 pre-zeroing.
 */
__arm_new("za") static void nk_attention_pack_b16_sme_streaming_(                      //
    void const *keys, void const *values, nk_size_t element_bytes,                     //
    nk_size_t key_value_head_count,                                                    //
    nk_size_t depth, nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, //
    void *key_value_packed, nk_size_t first_task, nk_size_t task_count) NK_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2;

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    char *payload_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    svbool_t const predicate_all_b32x = svptrue_b32();

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
        nk_size_t const plane_bytes = position_count_padded * depth_padded * sizeof(nk_u16_t);
        nk_u16_t *keys_plane = (nk_u16_t *)(payload_base + payload_offsets[segment_idx] +
                                            key_value_head_idx * plane_bytes);
        nk_u16_t *values_plane = (nk_u16_t *)(payload_base + payload_offsets[segment_idx] +
                                              (key_value_head_count + key_value_head_idx) * plane_bytes);

        // K: pair-interleave each position tile via the ZA0 transpose; fully padded tiles
        // still run and store the zeros the score stage expects.
        for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / tile_dimension;
             position_tile_idx++) {
            nk_size_t const position_start = position_tile_idx * tile_dimension;
            nk_size_t const rows_to_pack = (position_start + tile_dimension <= position_count) ? tile_dimension
                                           : (position_start < position_count) ? position_count - position_start
                                                                               : 0;
            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_pairs;
                 depth_batch_start += tile_dimension) {
                svbool_t const batch_predicate_b32x = svwhilelt_b32_u64(depth_batch_start, depth_pairs);
                nk_size_t const batch_size = svcntp_b32(svptrue_b32(), batch_predicate_b32x);
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_to_pack; row_in_tile++) {
                    char const *row_ptr = (char const *)keys +
                                          (position_first + position_start + row_in_tile) * key_stride_bytes +
                                          (key_value_head_idx * depth + depth_batch_start * 2) * element_bytes;
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
                    nk_u16_t *vector_output = keys_plane +
                                              (position_tile_idx * depth_pairs + depth_batch_start + depth_step) *
                                                  vector_elements;
                    svfloat32_t column_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 0,
                                                                    (uint32_t)depth_step);
                    svst1_f32(predicate_all_b32x, (float32_t *)vector_output, column_f32x);
                }
            }
        }

        // V: one ZIP1 turns two position rows of a channel tile into the pair-interleaved
        // MOPA operand; dead positions and channels arrive as zeros from the predicates.
        for (nk_size_t channel_tile_idx = 0; channel_tile_idx < depth_padded / tile_dimension; channel_tile_idx++) {
            nk_size_t const channel_start = channel_tile_idx * tile_dimension;
            svbool_t const channel_predicate_b16x = svwhilelt_b16_u64(channel_start, depth);
            svbool_t const channel_bytes_b8x = svwhilelt_b8_u64(channel_start, depth);
            for (nk_size_t position_pair_idx = 0; position_pair_idx < position_count_padded / 2; position_pair_idx++) {
                nk_size_t const position_even = position_pair_idx * 2, position_odd = position_even + 1;
                char const *even_ptr = (char const *)values + (position_first + position_even) * value_stride_bytes +
                                       (key_value_head_idx * depth + channel_start) * element_bytes;
                char const *odd_ptr = (char const *)values + (position_first + position_odd) * value_stride_bytes +
                                      (key_value_head_idx * depth + channel_start) * element_bytes;
                svuint16_t even_u16x = svdup_u16(0), odd_u16x = svdup_u16(0);
                if (element_bytes == sizeof(nk_u16_t)) {
                    if (position_even < position_count)
                        even_u16x = svld1_u16(channel_predicate_b16x, (nk_u16_t const *)even_ptr);
                    if (position_odd < position_count)
                        odd_u16x = svld1_u16(channel_predicate_b16x, (nk_u16_t const *)odd_ptr);
                }
                else {
                    if (position_even < position_count)
                        even_u16x = nk_attention_e4m3_to_bf16_sme_(
                            channel_predicate_b16x, svld1_u8(channel_bytes_b8x, (nk_u8_t const *)even_ptr));
                    if (position_odd < position_count)
                        odd_u16x = nk_attention_e4m3_to_bf16_sme_(
                            channel_predicate_b16x, svld1_u8(channel_bytes_b8x, (nk_u8_t const *)odd_ptr));
                }
                nk_u16_t *vector_output = values_plane +
                                          (channel_tile_idx * (position_count_padded / 2) + position_pair_idx) *
                                              vector_elements;
                svst1_u16(svptrue_b16(), vector_output, svzip1_u16(even_u16x, odd_u16x));
            }
        }
    }
}

NK_PUBLIC void nk_attention_pack_bf16_sme(                                            //
    nk_bf16_t const *keys, nk_bf16_t const *values,                                   //
    nk_size_t key_value_head_count, nk_size_t depth,                                  //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                 //
    nk_size_t segment_count,                                                          //
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_sme_k_ || nk_sme_cntw_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_bf16_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, nk_sme_cnth_(),
                                 nk_size_round_up_to_multiple_(depth, nk_sme_cntw_()) * sizeof(nk_u16_t));
    nk_sme_start_streaming_();
    nk_attention_pack_b16_sme_streaming_(keys, values, sizeof(nk_bf16_t), key_value_head_count, depth, segment_offsets,
                                         segment_lengths, segment_count, key_stride_bytes, value_stride_bytes,
                                         key_value_packed, first_task, task_count);
    nk_sme_stop_streaming_();
}

NK_PUBLIC nk_size_t nk_attention_packed_size_e4m3_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                      nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_sme_k_ || nk_sme_cntw_() > nk_attention_max_tile_sme_k_)
        return nk_attention_packed_size_e4m3_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_b16_sme_(key_value_head_count, depth, segment_lengths, segment_count);
}

NK_PUBLIC void nk_attention_pack_e4m3_sme(                                            //
    nk_e4m3_t const *keys, nk_e4m3_t const *values,                                   //
    nk_size_t key_value_head_count, nk_size_t depth,                                  //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                 //
    nk_size_t segment_count,                                                          //
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_sme_k_ || nk_sme_cntw_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_e4m3_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                      segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                      task_count);
        return;
    }
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, nk_sme_cnth_(),
                                 nk_size_round_up_to_multiple_(depth, nk_sme_cntw_()) * sizeof(nk_u16_t));
    nk_sme_start_streaming_();
    nk_attention_pack_b16_sme_streaming_(keys, values, sizeof(nk_e4m3_t), key_value_head_count, depth, segment_offsets,
                                         segment_lengths, segment_count, key_stride_bytes, value_stride_bytes,
                                         key_value_packed, first_task, task_count);
    nk_sme_stop_streaming_();
}

/**
 *  @brief Streaming attention core for the BF16-compute dtypes (raw BF16, E4M3 widened at
 *         the query loads): the whole task loop runs inside one ZA context.
 *
 *  Per (segment, head) task and per query block of two row-tiles:
 *
 *  1. Q staging: the ZA0 transpose pair-interleaves the block's query rows once into a
 *     scratch of MOPA operand vectors, reused across every KV panel.
 *  2. Scores: 2×2 widening BFMOPA blocking (two Q row-tiles × two K position-tiles), one
 *     vector load per MOPA; tiles drain through vertical stores into a position-major
 *     F32 panel with one query per lane.
 *  3. Softmax: lane-parallel running maximum, correction `2^(m_old − m_new)`, and weight
 *     sums; each position pair's weights convert to pair-interleaved BF16 in registers.
 *  4. P×V: 2×2 widening BFMOPA over position pairs into (row-tile × channel-tile)
 *     accumulators; vertical-slice drains fuse the correction FMA into the channel-major
 *     output accumulator, keeping queries in lanes end to end.
 *
 *  The channel-major accumulator transposes back to output rows through ZA0 once per block,
 *  with the `1 / Σweights` normalization applied on the way in.
 */
__arm_new("za") static void nk_attention_packed_b16_sme_streaming_(                                             //
    void const *queries, nk_size_t element_bytes, void const *key_value_packed, nk_f32_t *output,               //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) NK_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_elements = svcnth();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const panel_width = nk_attention_panel_sme_k_;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_pairs = depth_padded / 2;
    nk_size_t const channel_tiles = depth_padded / tile_dimension;

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->key_value_head_count != key_value_head_count) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_(segment_count);
    nk_size_t const output_stride_floats = output_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // fold log2e: softmax(x) = softmax₂(x·log₂e)

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    // Scratch sized for SVL ≤ 512 (tile dimension ≤ 16) and depth ≤ 256; the entry points
    // route larger shapes to serial. Queries live in lanes throughout, so the output
    // accumulator is channel-major: `o_acc[channel][query lane]`.

    NK_ALIGN64 nk_u16_t queries_packed[2][(nk_attention_max_depth_sme_k_ / 2) * 2 * nk_attention_max_tile_sme_k_];
    NK_ALIGN64 nk_f32_t scores_panel[nk_attention_panel_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    NK_ALIGN64 nk_u16_t weights_panel[nk_attention_panel_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    NK_ALIGN64 nk_f32_t o_acc[nk_attention_max_depth_sme_k_ * 2 * nk_attention_max_tile_sme_k_];

    svbool_t const predicate_all_b32x = svptrue_b32();
    svbool_t const predicate_all_b16x = svptrue_b16();
    svfloat32_t const scale2_f32x = svdup_f32(scale2);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const query_first = query_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, vector_elements);
        nk_size_t const position_pairs_total = position_count_padded / 2;
        nk_size_t const plane_bytes = position_count_padded * depth_padded * sizeof(nk_u16_t);
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        nk_u16_t const *keys_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                        key_value_head_idx * plane_bytes);
        nk_u16_t const *values_plane = (nk_u16_t const *)(payload_base + payload_offsets[segment_idx] +
                                                          (key_value_head_count + key_value_head_idx) * plane_bytes);

        for (nk_size_t row_block_start = 0; row_block_start < row_count; row_block_start += block_rows_capacity) {
            nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                             ? row_count - row_block_start
                                             : block_rows_capacity;

            // Stage 1: pair-interleave the block's Q rows through the ZA0 transpose, once per block.
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
                    svzero_mask_za(nk_sme_zero_za32_tile_0_);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_to_stage; row_in_tile++) {
                        char const *row_ptr = (char const *)queries +
                                              (query_first + row_block_start + tile_row_start + row_in_tile) *
                                                  query_stride_bytes +
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

            svfloat32_t running_max2_low_f32x = svdup_f32(NK_F32_MIN); // row-tile 0 queries, one per lane
            svfloat32_t running_max2_high_f32x = svdup_f32(NK_F32_MIN);
            svfloat32_t running_sum_low_f32x = svdup_f32(0.0f);
            svfloat32_t running_sum_high_f32x = svdup_f32(0.0f);
            for (nk_size_t element_idx = 0; element_idx < depth_padded * block_rows_capacity;
                 element_idx += tile_dimension)
                svst1_f32(predicate_all_b32x, (float32_t *)(o_acc + element_idx), svdup_f32(0.0f));

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_length = (position_count - panel_start < panel_width)
                                                   ? position_count - panel_start
                                                   : panel_width;
                nk_size_t const panel_pairs = (panel_length + 1) / 2;

                svfloat32_t panel_max_low_f32x = svdup_f32(NK_F32_MIN);
                svfloat32_t panel_max_high_f32x = svdup_f32(NK_F32_MIN);
                // Stage 2: 2×2 widening BFMOPA scores, drained position-major with the running
                // maximum folded into the drain so the score panel is read once. Padded positions
                // contribute exact zeros, which may only raise the maximum — always numerically
                // safe, and it cancels in normalization like the x86 tiers.
                for (nk_size_t chunk_start = 0; chunk_start < panel_length; chunk_start += block_rows_capacity) {
                    nk_size_t const position_tile_first = (panel_start + chunk_start) / tile_dimension;
                    nk_u16_t const *keys_tile0 = keys_plane + position_tile_first * depth_pairs * vector_elements;
                    nk_u16_t const *keys_tile1 = keys_tile0 + depth_pairs * vector_elements;
                    svzero_za();
                    for (nk_size_t depth_pair_idx = 0; depth_pair_idx < depth_pairs; depth_pair_idx++) {
                        svbfloat16_t const queries_low_bf16x = svreinterpret_bf16_u16(
                            svld1_u16(predicate_all_b16x, queries_packed[0] + depth_pair_idx * vector_elements));
                        svbfloat16_t const queries_high_bf16x = svreinterpret_bf16_u16(
                            svld1_u16(predicate_all_b16x, queries_packed[1] + depth_pair_idx * vector_elements));
                        svbfloat16_t const keys_low_bf16x = svreinterpret_bf16_u16(
                            svld1_u16(predicate_all_b16x, keys_tile0 + depth_pair_idx * vector_elements));
                        svbfloat16_t const keys_high_bf16x = svreinterpret_bf16_u16(
                            svld1_u16(predicate_all_b16x, keys_tile1 + depth_pair_idx * vector_elements));
                        svmopa_za32_bf16_m(0, predicate_all_b16x, predicate_all_b16x, queries_low_bf16x,
                                           keys_low_bf16x);
                        svmopa_za32_bf16_m(1, predicate_all_b16x, predicate_all_b16x, queries_low_bf16x,
                                           keys_high_bf16x);
                        svmopa_za32_bf16_m(2, predicate_all_b16x, predicate_all_b16x, queries_high_bf16x,
                                           keys_low_bf16x);
                        svmopa_za32_bf16_m(3, predicate_all_b16x, predicate_all_b16x, queries_high_bf16x,
                                           keys_high_bf16x);
                    }
                    nk_f32_t *chunk_scores = scores_panel + chunk_start * block_rows_capacity;
                    for (nk_size_t slice_idx = 0; slice_idx < tile_dimension; slice_idx++) {
                        svfloat32_t const column0_low_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x,
                                                                                   0, (uint32_t)slice_idx);
                        svfloat32_t const column0_high_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x,
                                                                                    2, (uint32_t)slice_idx);
                        svfloat32_t const column1_low_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x,
                                                                                   1, (uint32_t)slice_idx);
                        svfloat32_t const column1_high_f32x = svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x,
                                                                                    3, (uint32_t)slice_idx);
                        panel_max_low_f32x = svmax_f32_x(predicate_all_b32x, panel_max_low_f32x, column0_low_f32x);
                        panel_max_high_f32x = svmax_f32_x(predicate_all_b32x, panel_max_high_f32x, column0_high_f32x);
                        panel_max_low_f32x = svmax_f32_x(predicate_all_b32x, panel_max_low_f32x, column1_low_f32x);
                        panel_max_high_f32x = svmax_f32_x(predicate_all_b32x, panel_max_high_f32x, column1_high_f32x);
                        svst1_f32(predicate_all_b32x, (float32_t *)(chunk_scores + slice_idx * block_rows_capacity),
                                  column0_low_f32x);
                        svst1_f32(predicate_all_b32x,
                                  (float32_t *)(chunk_scores + slice_idx * block_rows_capacity + tile_dimension),
                                  column0_high_f32x);
                        svst1_f32(predicate_all_b32x,
                                  (float32_t *)(chunk_scores + (tile_dimension + slice_idx) * block_rows_capacity),
                                  column1_low_f32x);
                        svst1_f32(predicate_all_b32x,
                                  (float32_t *)(chunk_scores + (tile_dimension + slice_idx) * block_rows_capacity +
                                                tile_dimension),
                                  column1_high_f32x);
                    }
                }

                // Stage 3: lane-parallel softmax — every query owns one lane, so the maxima,
                // corrections, and sums are plain vector ops with no horizontal reductions.
                svfloat32_t const new_max2_low_f32x = svmax_f32_x(
                    predicate_all_b32x, running_max2_low_f32x,
                    svmul_f32_x(predicate_all_b32x, panel_max_low_f32x, scale2_f32x));
                svfloat32_t const new_max2_high_f32x = svmax_f32_x(
                    predicate_all_b32x, running_max2_high_f32x,
                    svmul_f32_x(predicate_all_b32x, panel_max_high_f32x, scale2_f32x));
                svfloat32_t const correction_low_f32x = nk_attention_exp2_f32x_sme_(
                    svsub_f32_x(predicate_all_b32x, running_max2_low_f32x, new_max2_low_f32x));
                svfloat32_t const correction_high_f32x = nk_attention_exp2_f32x_sme_(
                    svsub_f32_x(predicate_all_b32x, running_max2_high_f32x, new_max2_high_f32x));
                running_max2_low_f32x = new_max2_low_f32x;
                running_max2_high_f32x = new_max2_high_f32x;
                svfloat32_t const negated_max2_low_f32x = svneg_f32_x(predicate_all_b32x, new_max2_low_f32x);
                svfloat32_t const negated_max2_high_f32x = svneg_f32_x(predicate_all_b32x, new_max2_high_f32x);

                svfloat32_t panel_sum_low_f32x = svdup_f32(0.0f);
                svfloat32_t panel_sum_high_f32x = svdup_f32(0.0f);
                for (nk_size_t pair_idx = 0; pair_idx < panel_pairs; pair_idx++) {
                    nk_size_t const position_even = pair_idx * 2, position_odd = position_even + 1;
                    nk_f32_t const *even_scores = scores_panel + position_even * block_rows_capacity;
                    nk_f32_t const *odd_scores = scores_panel + position_odd * block_rows_capacity;
                    svfloat32_t even_low_f32x = svld1_f32(predicate_all_b32x, (float32_t const *)even_scores);
                    svfloat32_t even_high_f32x = svld1_f32(predicate_all_b32x,
                                                           (float32_t const *)(even_scores + tile_dimension));
                    svfloat32_t odd_low_f32x = svdup_f32(NK_F32_MIN); // padded positions decay to zero weight
                    svfloat32_t odd_high_f32x = svdup_f32(NK_F32_MIN);
                    if (position_odd < panel_length) {
                        odd_low_f32x = svld1_f32(predicate_all_b32x, (float32_t const *)odd_scores);
                        odd_high_f32x = svld1_f32(predicate_all_b32x, (float32_t const *)(odd_scores + tile_dimension));
                        odd_low_f32x = svmla_f32_x(predicate_all_b32x, negated_max2_low_f32x, odd_low_f32x,
                                                   scale2_f32x);
                        odd_high_f32x = svmla_f32_x(predicate_all_b32x, negated_max2_high_f32x, odd_high_f32x,
                                                    scale2_f32x);
                    }
                    even_low_f32x = svmla_f32_x(predicate_all_b32x, negated_max2_low_f32x, even_low_f32x, scale2_f32x);
                    even_high_f32x = svmla_f32_x(predicate_all_b32x, negated_max2_high_f32x, even_high_f32x,
                                                 scale2_f32x);
                    svfloat32_t const whole_even_low_f32x = svrintn_f32_x(predicate_all_b32x, even_low_f32x);
                    svfloat32_t const whole_even_high_f32x = svrintn_f32_x(predicate_all_b32x, even_high_f32x);
                    svfloat32_t const whole_odd_low_f32x = svrintn_f32_x(predicate_all_b32x, odd_low_f32x);
                    svfloat32_t const whole_odd_high_f32x = svrintn_f32_x(predicate_all_b32x, odd_high_f32x);
                    svfloat16_t const fraction_even_f16x = svcvtnt_f16_f32_m( // split-precision: F16 only here

                        svcvt_f16_f32_x(predicate_all_b32x,
                                        svsub_f32_x(predicate_all_b32x, even_low_f32x, whole_even_low_f32x)),
                        predicate_all_b32x, svsub_f32_x(predicate_all_b32x, even_high_f32x, whole_even_high_f32x));
                    svfloat16_t const fraction_odd_f16x = svcvtnt_f16_f32_m(
                        svcvt_f16_f32_x(predicate_all_b32x,
                                        svsub_f32_x(predicate_all_b32x, odd_low_f32x, whole_odd_low_f32x)),
                        predicate_all_b32x, svsub_f32_x(predicate_all_b32x, odd_high_f32x, whole_odd_high_f32x));
                    svfloat16_t const poly_even_f16x = nk_attention_exp2_fraction_f16x_sme_(fraction_even_f16x);
                    svfloat16_t const poly_odd_f16x = nk_attention_exp2_fraction_f16x_sme_(fraction_odd_f16x);
                    svfloat32_t const weight_even_low_f32x = svscale_f32_x(
                        predicate_all_b32x, svcvt_f32_f16_x(predicate_all_b32x, poly_even_f16x),
                        svcvt_s32_f32_x(predicate_all_b32x, whole_even_low_f32x));
                    svfloat32_t const weight_even_high_f32x = svscale_f32_x(
                        predicate_all_b32x, svcvtlt_f32_f16_x(predicate_all_b32x, poly_even_f16x),
                        svcvt_s32_f32_x(predicate_all_b32x, whole_even_high_f32x));
                    svfloat32_t const weight_odd_low_f32x = svscale_f32_x(
                        predicate_all_b32x, svcvt_f32_f16_x(predicate_all_b32x, poly_odd_f16x),
                        svcvt_s32_f32_x(predicate_all_b32x, whole_odd_low_f32x));
                    svfloat32_t const weight_odd_high_f32x = svscale_f32_x(
                        predicate_all_b32x, svcvtlt_f32_f16_x(predicate_all_b32x, poly_odd_f16x),
                        svcvt_s32_f32_x(predicate_all_b32x, whole_odd_high_f32x));
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

                nk_size_t const panel_pair_first = panel_start / 2;
                // Stage 4: P×V outer products over position pairs; vertical drains fuse the
                // correction FMA into the channel-major accumulator.
                for (nk_size_t channel_tile_idx = 0; channel_tile_idx < channel_tiles; channel_tile_idx += 2) {
                    int const has_second_tile = channel_tile_idx + 1 < channel_tiles;
                    nk_u16_t const *values_tile0 =
                        values_plane + (channel_tile_idx * position_pairs_total + panel_pair_first) * vector_elements;
                    nk_u16_t const *values_tile1 = values_tile0 + position_pairs_total * vector_elements;
                    svzero_za();
                    for (nk_size_t pair_idx = 0; pair_idx < panel_pairs; pair_idx++) {
                        svbfloat16_t const weights_low_bf16x = svreinterpret_bf16_u16(
                            svld1_u16(predicate_all_b16x, weights_panel + (pair_idx * 2 + 0) * vector_elements));
                        svbfloat16_t const weights_high_bf16x = svreinterpret_bf16_u16(
                            svld1_u16(predicate_all_b16x, weights_panel + (pair_idx * 2 + 1) * vector_elements));
                        svbfloat16_t const values_low_bf16x = svreinterpret_bf16_u16(
                            svld1_u16(predicate_all_b16x, values_tile0 + pair_idx * vector_elements));
                        svmopa_za32_bf16_m(0, predicate_all_b16x, predicate_all_b16x, weights_low_bf16x,
                                           values_low_bf16x);
                        svmopa_za32_bf16_m(2, predicate_all_b16x, predicate_all_b16x, weights_high_bf16x,
                                           values_low_bf16x);
                        if (has_second_tile) {
                            svbfloat16_t const values_high_bf16x = svreinterpret_bf16_u16(
                                svld1_u16(predicate_all_b16x, values_tile1 + pair_idx * vector_elements));
                            svmopa_za32_bf16_m(1, predicate_all_b16x, predicate_all_b16x, weights_low_bf16x,
                                               values_high_bf16x);
                            svmopa_za32_bf16_m(3, predicate_all_b16x, predicate_all_b16x, weights_high_bf16x,
                                               values_high_bf16x);
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
                            svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 0, (uint32_t)slice_idx));
                        o_tile0_high_f32x = svmad_f32_x(
                            predicate_all_b32x, o_tile0_high_f32x, correction_high_f32x,
                            svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 2, (uint32_t)slice_idx));
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
                                svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 1, (uint32_t)slice_idx));
                            o_tile1_high_f32x = svmad_f32_x(
                                predicate_all_b32x, o_tile1_high_f32x, correction_high_f32x,
                                svread_ver_za32_f32_m(svdup_f32(0.0f), predicate_all_b32x, 3, (uint32_t)slice_idx));
                            svst1_f32(predicate_all_b32x, (float32_t *)accumulator_tile1, o_tile1_low_f32x);
                            svst1_f32(predicate_all_b32x, (float32_t *)(accumulator_tile1 + tile_dimension),
                                      o_tile1_high_f32x);
                        }
                    }
                }
            }

            svfloat32_t const inverse_sum_low_f32x = svdiv_f32_x(predicate_all_b32x, svdup_f32(1.0f),
                                                                 running_sum_low_f32x);
            svfloat32_t const inverse_sum_high_f32x = svdiv_f32_x(predicate_all_b32x, svdup_f32(1.0f),
                                                                  running_sum_high_f32x);
            // Finalize: normalize lane-wise, then transpose the channel-major accumulator back
            // to output rows through ZA0, one row-tile × channel-tile block at a time.
            for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
                if (tile_row_start >= block_rows) break;
                nk_size_t const rows_valid = (block_rows - tile_row_start < tile_dimension)
                                                 ? block_rows - tile_row_start
                                                 : tile_dimension;
                svfloat32_t const inverse_sum_f32x = row_tile_idx == 0 ? inverse_sum_low_f32x : inverse_sum_high_f32x;
                for (nk_size_t channel_tile_idx = 0; channel_tile_idx < channel_tiles; channel_tile_idx++) {
                    nk_size_t const channel_start = channel_tile_idx * tile_dimension;
                    svbool_t const channel_predicate_b32x = svwhilelt_b32_u64(channel_start, depth);
                    svzero_mask_za(nk_sme_zero_za32_tile_0_);
                    for (nk_size_t channel_in_tile = 0; channel_in_tile < tile_dimension; channel_in_tile++) {
                        svfloat32_t normalized_f32x = svmul_f32_x(
                            predicate_all_b32x,
                            svld1_f32(
                                predicate_all_b32x,
                                (float32_t const *)(o_acc + (channel_start + channel_in_tile) * block_rows_capacity +
                                                    tile_row_start)),
                            inverse_sum_f32x);
                        svwrite_hor_za32_f32_m(0, (uint32_t)channel_in_tile, predicate_all_b32x, normalized_f32x);
                    }
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_valid; row_in_tile++) {
                        nk_f32_t *output_row = output +
                                               (query_first + row_block_start + tile_row_start + row_in_tile) *
                                                   output_stride_floats +
                                               head_idx * depth + channel_start;
                        svst1_ver_za32(0, (uint32_t)row_in_tile, channel_predicate_b32x, output_row);
                    }
                }
            }
        }
    }
}

NK_PUBLIC void nk_attention_packed_bf16_sme(                                     //
    nk_bf16_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_sme_k_ || nk_sme_cntw_() > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_bf16_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_sme_start_streaming_();
    nk_attention_packed_b16_sme_streaming_(queries, sizeof(nk_bf16_t), key_value_packed, output, head_count,
                                           key_value_head_count, depth, query_offsets, query_stride_bytes,
                                           output_stride_bytes, scale, first_task, task_count);
    nk_sme_stop_streaming_();
}

NK_PUBLIC void nk_attention_packed_e4m3_sme(                                     //
    nk_e4m3_t const *queries, void const *key_value_packed, nk_f32_t *output,    //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_sme_k_ || nk_sme_cntw_() > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_e4m3_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                        query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                        task_count);
        return;
    }
    nk_sme_start_streaming_();
    nk_attention_packed_b16_sme_streaming_(queries, sizeof(nk_e4m3_t), key_value_packed, output, head_count,
                                           key_value_head_count, depth, query_offsets, query_stride_bytes,
                                           output_stride_bytes, scale, first_task, task_count);
    nk_sme_stop_streaming_();
}

NK_INTERNAL nk_size_t nk_attention_packed_size_b8_sme_(nk_size_t key_value_head_count, nk_size_t depth,
                                                       nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    nk_size_t const tile_dimension = nk_sme_cntw_();
    nk_size_t const position_multiple = nk_sme_cnth_();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t payload_bytes = 0;
    for (nk_size_t segment_idx = 0; segment_idx < segment_count; segment_idx++) {
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(segment_lengths[segment_idx],
                                                                              position_multiple);
        payload_bytes += 2 * key_value_head_count * position_count_padded * depth_padded; // K + V, raw bytes
    }
    return sizeof(nk_attention_packed_header_t) + nk_attention_pack_directory_size_(segment_count) + payload_bytes;
}

NK_PUBLIC nk_size_t nk_attention_packed_size_i8_sme(nk_size_t key_value_head_count, nk_size_t depth,
                                                    nk_u32_t const *segment_lengths, nk_size_t segment_count) {
    if (depth > nk_attention_max_depth_sme_k_ || nk_sme_cntw_() > nk_attention_max_tile_sme_k_)
        return nk_attention_packed_size_i8_serial(key_value_head_count, depth, segment_lengths, segment_count);
    return nk_attention_packed_size_b8_sme_(key_value_head_count, depth, segment_lengths, segment_count);
}

/**
 *  @brief Streaming pack core for I8 K/V planes.
 *
 *  K becomes quad-interleaved SMOPA operand vectors `[position_tile][depth_quad]` through the
 *  ZA0 horizontal-write / vertical-read transpose; V becomes transposed quad-interleaved
 *  vectors `[channel_tile][position_quad]` through a two-level `ZIP1`, so P×V runs as
 *  USMOPA outer products over positions with the U8 probabilities.
 */
__arm_new("za") static void nk_attention_pack_i8_sme_streaming_(                       //
    nk_i8_t const *keys, nk_i8_t const *values, nk_size_t key_value_head_count,        //
    nk_size_t depth, nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths, //
    nk_size_t segment_count, nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, //
    void *key_value_packed, nk_size_t first_task, nk_size_t task_count) NK_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_bytes = svcntb();
    nk_size_t const position_multiple = svcnth();
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_quads = depth_padded / 4;

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    char *payload_base = (char *)key_value_packed + sizeof(*header) + nk_attention_pack_directory_size_(segment_count);

    nk_size_t const total_tasks = segment_count * key_value_head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    svbool_t const predicate_all_b32x = svptrue_b32();

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / key_value_head_count,
                        key_value_head_idx = task_idx % key_value_head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        if (position_count == 0) continue;
        nk_size_t const position_first = segment_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, position_multiple);
        nk_size_t const plane_bytes = position_count_padded * depth_padded;
        nk_u8_t *keys_plane = (nk_u8_t *)(payload_base + payload_offsets[segment_idx] +
                                          key_value_head_idx * plane_bytes);
        nk_u8_t *values_plane = (nk_u8_t *)(payload_base + payload_offsets[segment_idx] +
                                            (key_value_head_count + key_value_head_idx) * plane_bytes);

        // K: quad-interleave each position tile via the ZA0 transpose; fully padded tiles
        // still run and store the zeros the score stage expects.
        for (nk_size_t position_tile_idx = 0; position_tile_idx < position_count_padded / tile_dimension;
             position_tile_idx++) {
            nk_size_t const position_start = position_tile_idx * tile_dimension;
            nk_size_t const rows_to_pack = (position_start + tile_dimension <= position_count) ? tile_dimension
                                           : (position_start < position_count) ? position_count - position_start
                                                                               : 0;
            for (nk_size_t depth_batch_start = 0; depth_batch_start < depth_quads;
                 depth_batch_start += tile_dimension) {
                svbool_t const batch_predicate_b32x = svwhilelt_b32_u64(depth_batch_start, depth_quads);
                nk_size_t const batch_size = svcntp_b32(svptrue_b32(), batch_predicate_b32x);
                svzero_mask_za(nk_sme_zero_za32_tile_0_);
                for (nk_size_t row_in_tile = 0; row_in_tile < rows_to_pack; row_in_tile++) {
                    char const *row_ptr = (char const *)keys +
                                          (position_first + position_start + row_in_tile) * key_stride_bytes +
                                          key_value_head_idx * depth + depth_batch_start * 4;
                    svbool_t const depth_predicate_b8x = svwhilelt_b8_u64(depth_batch_start * 4, depth);
                    svint8_t row_i8x = svld1_s8(depth_predicate_b8x, (int8_t const *)row_ptr);
                    svwrite_hor_za32_s32_m(0, (uint32_t)row_in_tile, batch_predicate_b32x,
                                           svreinterpret_s32_s8(row_i8x));
                }
                for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                    nk_u8_t *vector_output =
                        keys_plane + (position_tile_idx * depth_quads + depth_batch_start + depth_step) * vector_bytes;
                    svint32_t column_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 0,
                                                                  (uint32_t)depth_step);
                    svst1_s32(predicate_all_b32x, (int32_t *)vector_output, column_i32x);
                }
            }
        }

        // V: two ZIP1 levels turn four position rows of a channel tile into the
        // quad-interleaved operand; dead positions and channels arrive as zeros.
        for (nk_size_t channel_tile_idx = 0; channel_tile_idx < depth_padded / tile_dimension; channel_tile_idx++) {
            nk_size_t const channel_start = channel_tile_idx * tile_dimension;
            svbool_t const channel_predicate_b8x = svwhilelt_b8_u64(channel_start, depth);
            for (nk_size_t position_quad_idx = 0; position_quad_idx < position_count_padded / 4; position_quad_idx++) {
                nk_size_t const position_base = position_quad_idx * 4;
                nk_u8_t const *values_rows = (nk_u8_t const *)values + key_value_head_idx * depth + channel_start;
                svuint8_t const row_0_u8x = position_base + 0 < position_count
                                                ? svld1_u8(channel_predicate_b8x,
                                                           values_rows + (position_first + position_base + 0) *
                                                                             value_stride_bytes)
                                                : svdup_u8(0);
                svuint8_t const row_1_u8x = position_base + 1 < position_count
                                                ? svld1_u8(channel_predicate_b8x,
                                                           values_rows + (position_first + position_base + 1) *
                                                                             value_stride_bytes)
                                                : svdup_u8(0);
                svuint8_t const row_2_u8x = position_base + 2 < position_count
                                                ? svld1_u8(channel_predicate_b8x,
                                                           values_rows + (position_first + position_base + 2) *
                                                                             value_stride_bytes)
                                                : svdup_u8(0);
                svuint8_t const row_3_u8x = position_base + 3 < position_count
                                                ? svld1_u8(channel_predicate_b8x,
                                                           values_rows + (position_first + position_base + 3) *
                                                                             value_stride_bytes)
                                                : svdup_u8(0);
                svuint8_t const even_pairs_u8x = svzip1_u8(row_0_u8x, row_1_u8x);
                svuint8_t const odd_pairs_u8x = svzip1_u8(row_2_u8x, row_3_u8x);
                svuint16_t const quads_u16x = svzip1_u16(svreinterpret_u16_u8(even_pairs_u8x),
                                                         svreinterpret_u16_u8(odd_pairs_u8x));
                nk_u8_t *vector_output =
                    values_plane + (channel_tile_idx * (position_count_padded / 4) + position_quad_idx) * vector_bytes;
                svst1_u16(svptrue_b16(), (nk_u16_t *)vector_output, quads_u16x);
            }
        }
    }
}

NK_PUBLIC void nk_attention_pack_i8_sme(                                              //
    nk_i8_t const *keys, nk_i8_t const *values,                                       //
    nk_size_t key_value_head_count, nk_size_t depth,                                  //
    nk_u32_t const *segment_offsets, nk_u32_t const *segment_lengths,                 //
    nk_size_t segment_count,                                                          //
    nk_size_t key_stride_bytes, nk_size_t value_stride_bytes, void *key_value_packed, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_sme_k_ || nk_sme_cntw_() > nk_attention_max_tile_sme_k_) {
        nk_attention_pack_i8_serial(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                    segment_count, key_stride_bytes, value_stride_bytes, key_value_packed, first_task,
                                    task_count);
        return;
    }
    nk_attention_pack_directory_(key_value_packed, key_value_head_count, depth, segment_lengths, segment_count,
                                 first_task, nk_sme_cnth_(), nk_size_round_up_to_multiple_(depth, nk_sme_cntw_()));
    nk_sme_start_streaming_();
    nk_attention_pack_i8_sme_streaming_(keys, values, key_value_head_count, depth, segment_offsets, segment_lengths,
                                        segment_count, key_stride_bytes, value_stride_bytes, key_value_packed,
                                        first_task, task_count);
    nk_sme_stop_streaming_();
}

/**
 *  @brief Streaming attention core for I8: exact I32 scores through SMOPA, U8-quantized
 *         probabilities, and USMOPA for the probability × value product.
 *
 *  Mirrors the B16 core's structure — ZA0 quad-interleaving Q staging, 2×2 score blocking
 *  with position-major F32 drains, lane-parallel softmax, channel-major F32 accumulator —
 *  with two I8 twists: weights quantize to `trunc(2^(s₂−m₂)·255 + 0.5)` and assemble into
 *  quad-interleaved U8 operands with three shift-ors per vector, and the P×V drain converts
 *  the exact I32 outer products to F32 before the correction FMA.
 */
__arm_new("za") static void nk_attention_packed_i8_sme_streaming_(                                              //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,                                     //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,                                      //
    nk_u32_t const *query_offsets, nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) NK_STREAMING_ {

    nk_size_t const tile_dimension = svcntw();
    nk_size_t const vector_bytes = svcntb();
    nk_size_t const block_rows_capacity = 2 * tile_dimension;
    nk_size_t const panel_width = nk_attention_panel_sme_k_;
    nk_size_t const depth_padded = nk_size_round_up_to_multiple_(depth, tile_dimension);
    nk_size_t const depth_quads = depth_padded / 4;
    nk_size_t const channel_tiles = depth_padded / tile_dimension;

    nk_attention_packed_header_t const *header = (nk_attention_packed_header_t const *)key_value_packed;
    if (header->depth != depth || header->key_value_head_count != key_value_head_count) return;
    nk_size_t const segment_count = header->segment_count;
    nk_u64_t const *payload_offsets = (nk_u64_t const *)((char const *)key_value_packed + sizeof(*header));
    nk_u32_t const *segment_lengths = (nk_u32_t const *)(payload_offsets + segment_count + 1);
    char const *payload_base = (char const *)key_value_packed + sizeof(*header) +
                               nk_attention_pack_directory_size_(segment_count);
    nk_size_t const output_stride_floats = output_stride_bytes / sizeof(nk_f32_t);
    nk_size_t const head_group_size = head_count / key_value_head_count;
    nk_f32_t const scale2 = scale * NK_F32_LOG2E_; // fold log2e: softmax(x) = softmax₂(x·log₂e)

    nk_size_t const total_tasks = segment_count * head_count;
    if (first_task >= total_tasks) return;
    if (task_count == 0 || first_task + task_count > total_tasks) task_count = total_tasks - first_task;

    nk_i32_t const scale_fixed = (nk_i32_t)(scale2 * 32768.0f + 0.5f); // Q15 scale for the integer exponential
    nk_i32_t const delta_floor = // the score delta below which every weight quantizes to zero (2^t·255 + 0.5 < 1)
        scale_fixed > 0 ? -(nk_i32_t)((10u << 15) / (nk_u32_t)scale_fixed) - 1 : 0;

    NK_ALIGN64 nk_u8_t queries_packed[2][(nk_attention_max_depth_sme_k_ / 4) * 4 * nk_attention_max_tile_sme_k_];
    NK_ALIGN64 nk_i32_t scores_panel[nk_attention_panel_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    NK_ALIGN64 nk_u8_t weights_panel[nk_attention_panel_sme_k_ * 2 * nk_attention_max_tile_sme_k_];
    NK_ALIGN64 nk_f32_t o_acc[nk_attention_max_depth_sme_k_ * 2 * nk_attention_max_tile_sme_k_];

    svbool_t const predicate_all_b32x = svptrue_b32();
    svbool_t const predicate_all_b8x = svptrue_b8();
    svfloat32_t const scale2_f32x = svdup_f32(scale2);

    for (nk_size_t task_idx = first_task; task_idx < first_task + task_count; task_idx++) {
        nk_size_t const segment_idx = task_idx / head_count, head_idx = task_idx % head_count;
        nk_size_t const position_count = segment_lengths[segment_idx];
        nk_size_t const row_count = query_offsets[segment_idx + 1] - query_offsets[segment_idx];
        if (position_count == 0 || row_count == 0) continue;
        nk_size_t const query_first = query_offsets[segment_idx];
        nk_size_t const position_count_padded = nk_size_round_up_to_multiple_(position_count, 2 * tile_dimension);
        nk_size_t const position_quads_total = position_count_padded / 4;
        nk_size_t const plane_bytes = position_count_padded * depth_padded;
        nk_size_t const key_value_head_idx = head_idx / head_group_size;
        nk_u8_t const *keys_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx] +
                                                      key_value_head_idx * plane_bytes);
        nk_u8_t const *values_plane = (nk_u8_t const *)(payload_base + payload_offsets[segment_idx] +
                                                        (key_value_head_count + key_value_head_idx) * plane_bytes);

        for (nk_size_t row_block_start = 0; row_block_start < row_count; row_block_start += block_rows_capacity) {
            nk_size_t const block_rows = (row_count - row_block_start < block_rows_capacity)
                                             ? row_count - row_block_start
                                             : block_rows_capacity;

            // Stage 1: quad-interleave the block's Q rows through the ZA0 transpose, once per block.
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
                    svzero_mask_za(nk_sme_zero_za32_tile_0_);
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_to_stage; row_in_tile++) {
                        char const *row_ptr = (char const *)queries +
                                              (query_first + row_block_start + tile_row_start + row_in_tile) *
                                                  query_stride_bytes +
                                              head_idx * depth + depth_batch_start * 4;
                        svbool_t const depth_predicate_b8x = svwhilelt_b8_u64(depth_batch_start * 4, depth);
                        svint8_t row_i8x = svld1_s8(depth_predicate_b8x, (int8_t const *)row_ptr);
                        svwrite_hor_za32_s32_m(0, (uint32_t)row_in_tile, batch_predicate_b32x,
                                               svreinterpret_s32_s8(row_i8x));
                    }
                    for (nk_size_t depth_step = 0; depth_step < batch_size; depth_step++) {
                        svint32_t column_i32x = svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 0,
                                                                      (uint32_t)depth_step);
                        svst1_s32(
                            predicate_all_b32x,
                            (int32_t *)(queries_packed[row_tile_idx] + (depth_batch_start + depth_step) * vector_bytes),
                            column_i32x);
                    }
                }
            }

            svint32_t running_max_low_i32x = svdup_s32(-2147483647 - 1); // raw scores, one query per lane
            svint32_t running_max_high_i32x = svdup_s32(-2147483647 - 1);
            svfloat32_t running_sum_low_f32x = svdup_f32(0.0f);
            svfloat32_t running_sum_high_f32x = svdup_f32(0.0f);
            for (nk_size_t element_idx = 0; element_idx < depth_padded * block_rows_capacity;
                 element_idx += tile_dimension)
                svst1_f32(predicate_all_b32x, (float32_t *)(o_acc + element_idx), svdup_f32(0.0f));

            for (nk_size_t panel_start = 0; panel_start < position_count; panel_start += panel_width) {
                nk_size_t const panel_length = (position_count - panel_start < panel_width)
                                                   ? position_count - panel_start
                                                   : panel_width;
                nk_size_t const panel_quads = (panel_length + 3) / 4;

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
                        svst1_ver_za32(0, (uint32_t)slice_idx, predicate_all_b32x,
                                       chunk_scores + slice_idx * block_rows_capacity);
                        svst1_ver_za32(2, (uint32_t)slice_idx, predicate_all_b32x,
                                       chunk_scores + slice_idx * block_rows_capacity + tile_dimension);
                        svst1_ver_za32(1, (uint32_t)slice_idx, predicate_all_b32x,
                                       chunk_scores + (tile_dimension + slice_idx) * block_rows_capacity);
                        svst1_ver_za32(
                            3, (uint32_t)slice_idx, predicate_all_b32x,
                            chunk_scores + (tile_dimension + slice_idx) * block_rows_capacity + tile_dimension);
                    }
                }

                svint32_t panel_max_low_i32x = svdup_s32(-2147483647 - 1);
                svint32_t panel_max_high_i32x = svdup_s32(-2147483647 - 1);
                // Stage 3: lane-parallel integer softmax; scores, maxima, and the exponential
                // all stay in integer arithmetic, and the maximum position lands on exactly
                // 255, so the weight sum can never be zero.
                for (nk_size_t position_idx = 0; position_idx < panel_length; position_idx++) {
                    panel_max_low_i32x = svmax_s32_x(
                        predicate_all_b32x, panel_max_low_i32x,
                        svld1_s32(predicate_all_b32x,
                                  (int32_t const *)(scores_panel + position_idx * block_rows_capacity)));
                    panel_max_high_i32x = svmax_s32_x(
                        predicate_all_b32x, panel_max_high_i32x,
                        svld1_s32(
                            predicate_all_b32x,
                            (int32_t const *)(scores_panel + position_idx * block_rows_capacity + tile_dimension)));
                }
                svint32_t const new_max_low_i32x = svmax_s32_x(predicate_all_b32x, running_max_low_i32x,
                                                               panel_max_low_i32x);
                svint32_t const new_max_high_i32x = svmax_s32_x(predicate_all_b32x, running_max_high_i32x,
                                                                panel_max_high_i32x);
                svfloat32_t const correction_low_f32x = nk_attention_exp2_f32x_sme_(svmul_f32_x( // exact I32→F32

                    predicate_all_b32x,
                    svsub_f32_x(predicate_all_b32x, svcvt_f32_s32_x(predicate_all_b32x, running_max_low_i32x),
                                svcvt_f32_s32_x(predicate_all_b32x, new_max_low_i32x)),
                    scale2_f32x));
                svfloat32_t const correction_high_f32x = nk_attention_exp2_f32x_sme_(svmul_f32_x(
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
                    for (nk_size_t position_in_quad = 0; position_in_quad < 4; position_in_quad++) {
                        nk_size_t const position_idx = quad_idx * 4 + position_in_quad;
                        svint32_t delta_low_i32x = svdup_s32(delta_floor); // padded positions quantize to zero
                        svint32_t delta_high_i32x = svdup_s32(delta_floor);
                        if (position_idx < panel_length) {
                            nk_i32_t const *position_scores = scores_panel + position_idx * block_rows_capacity;
                            delta_low_i32x = svmax_n_s32_x(
                                predicate_all_b32x,
                                svsub_s32_x(predicate_all_b32x,
                                            svld1_s32(predicate_all_b32x, (int32_t const *)position_scores),
                                            new_max_low_i32x),
                                delta_floor);
                            delta_high_i32x = svmax_n_s32_x(
                                predicate_all_b32x,
                                svsub_s32_x(
                                    predicate_all_b32x,
                                    svld1_s32(predicate_all_b32x, (int32_t const *)(position_scores + tile_dimension)),
                                    new_max_high_i32x),
                                delta_floor);
                        }
                        svuint32_t const weight_low_u32x = svreinterpret_u32_s32(nk_attention_iexp2_weight_i32x_sme_(
                            svmul_n_s32_x(predicate_all_b32x, delta_low_i32x, scale_fixed)));
                        svuint32_t const weight_high_u32x = svreinterpret_u32_s32(nk_attention_iexp2_weight_i32x_sme_(
                            svmul_n_s32_x(predicate_all_b32x, delta_high_i32x, scale_fixed)));
                        panel_sum_low_u32x = svadd_u32_x(predicate_all_b32x, panel_sum_low_u32x, weight_low_u32x);
                        panel_sum_high_u32x = svadd_u32_x(predicate_all_b32x, panel_sum_high_u32x, weight_high_u32x);
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
                // Stage 4: P×V as USMOPA outer products over position quads; drains convert
                // the exact I32 totals to F32 and fuse the correction FMA.
                for (nk_size_t channel_tile_idx = 0; channel_tile_idx < channel_tiles; channel_tile_idx += 2) {
                    int const has_second_tile = channel_tile_idx + 1 < channel_tiles;
                    nk_u8_t const *values_tile0 =
                        values_plane + (channel_tile_idx * position_quads_total + panel_quad_first) * vector_bytes;
                    nk_u8_t const *values_tile1 = values_tile0 + position_quads_total * vector_bytes;
                    svzero_za();
                    for (nk_size_t quad_idx = 0; quad_idx < panel_quads; quad_idx++) {
                        svuint8_t const weights_low_u8x = svld1_u8(predicate_all_b8x,
                                                                   weights_panel + (quad_idx * 2 + 0) * vector_bytes);
                        svuint8_t const weights_high_u8x = svld1_u8(predicate_all_b8x,
                                                                    weights_panel + (quad_idx * 2 + 1) * vector_bytes);
                        svint8_t const values_low_i8x = svreinterpret_s8_u8(
                            svld1_u8(predicate_all_b8x, values_tile0 + quad_idx * vector_bytes));
                        svusmopa_za32_u8_m(0, predicate_all_b8x, predicate_all_b8x, weights_low_u8x, values_low_i8x);
                        svusmopa_za32_u8_m(2, predicate_all_b8x, predicate_all_b8x, weights_high_u8x, values_low_i8x);
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
                            svcvt_f32_s32_x(predicate_all_b32x, svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x,
                                                                                      0, (uint32_t)slice_idx)));
                        o_tile0_high_f32x = svmad_f32_x(
                            predicate_all_b32x, o_tile0_high_f32x, correction_high_f32x,
                            svcvt_f32_s32_x(predicate_all_b32x, svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x,
                                                                                      2, (uint32_t)slice_idx)));
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
                                svcvt_f32_s32_x(
                                    predicate_all_b32x,
                                    svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 1, (uint32_t)slice_idx)));
                            o_tile1_high_f32x = svmad_f32_x(
                                predicate_all_b32x, o_tile1_high_f32x, correction_high_f32x,
                                svcvt_f32_s32_x(
                                    predicate_all_b32x,
                                    svread_ver_za32_s32_m(svdup_s32(0), predicate_all_b32x, 3, (uint32_t)slice_idx)));
                            svst1_f32(predicate_all_b32x, (float32_t *)accumulator_tile1, o_tile1_low_f32x);
                            svst1_f32(predicate_all_b32x, (float32_t *)(accumulator_tile1 + tile_dimension),
                                      o_tile1_high_f32x);
                        }
                    }
                }
            }

            svfloat32_t const inverse_sum_low_f32x = svdiv_f32_x(predicate_all_b32x, svdup_f32(1.0f),
                                                                 running_sum_low_f32x);
            svfloat32_t const inverse_sum_high_f32x = svdiv_f32_x(predicate_all_b32x, svdup_f32(1.0f),
                                                                  running_sum_high_f32x);
            // Finalize: normalize lane-wise, then transpose the channel-major accumulator back
            // to output rows through ZA0, one row-tile × channel-tile block at a time.
            for (nk_size_t row_tile_idx = 0; row_tile_idx < 2; row_tile_idx++) {
                nk_size_t const tile_row_start = row_tile_idx * tile_dimension;
                if (tile_row_start >= block_rows) break;
                nk_size_t const rows_valid = (block_rows - tile_row_start < tile_dimension)
                                                 ? block_rows - tile_row_start
                                                 : tile_dimension;
                svfloat32_t const inverse_sum_f32x = row_tile_idx == 0 ? inverse_sum_low_f32x : inverse_sum_high_f32x;
                for (nk_size_t channel_tile_idx = 0; channel_tile_idx < channel_tiles; channel_tile_idx++) {
                    nk_size_t const channel_start = channel_tile_idx * tile_dimension;
                    svbool_t const channel_predicate_b32x = svwhilelt_b32_u64(channel_start, depth);
                    svzero_mask_za(nk_sme_zero_za32_tile_0_);
                    for (nk_size_t channel_in_tile = 0; channel_in_tile < tile_dimension; channel_in_tile++) {
                        svfloat32_t normalized_f32x = svmul_f32_x(
                            predicate_all_b32x,
                            svld1_f32(
                                predicate_all_b32x,
                                (float32_t const *)(o_acc + (channel_start + channel_in_tile) * block_rows_capacity +
                                                    tile_row_start)),
                            inverse_sum_f32x);
                        svwrite_hor_za32_f32_m(0, (uint32_t)channel_in_tile, predicate_all_b32x, normalized_f32x);
                    }
                    for (nk_size_t row_in_tile = 0; row_in_tile < rows_valid; row_in_tile++) {
                        nk_f32_t *output_row = output +
                                               (query_first + row_block_start + tile_row_start + row_in_tile) *
                                                   output_stride_floats +
                                               head_idx * depth + channel_start;
                        svst1_ver_za32(0, (uint32_t)row_in_tile, channel_predicate_b32x, output_row);
                    }
                }
            }
        }
    }
}

NK_PUBLIC void nk_attention_packed_i8_sme(                                       //
    nk_i8_t const *queries, void const *key_value_packed, nk_f32_t *output,      //
    nk_size_t head_count, nk_size_t key_value_head_count, nk_size_t depth,       //
    nk_u32_t const *query_offsets,                                               //
    nk_size_t query_stride_bytes, nk_size_t output_stride_bytes, nk_f32_t scale, //
    nk_size_t first_task, nk_size_t task_count) {
    if (depth > nk_attention_max_depth_sme_k_ || nk_sme_cntw_() > nk_attention_max_tile_sme_k_) {
        nk_attention_packed_i8_serial(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                      query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                      task_count);
        return;
    }
    nk_sme_start_streaming_();
    nk_attention_packed_i8_sme_streaming_(queries, key_value_packed, output, head_count, key_value_head_count, depth,
                                          query_offsets, query_stride_bytes, output_stride_bytes, scale, first_task,
                                          task_count);
    nk_sme_stop_streaming_();
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM64_

#endif // NK_ATTENTION_SME_H
