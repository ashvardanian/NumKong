/**
 *  @file include/numkong/dots/cdna4.cuh
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief SIMD-accelerated Batched Dot Products for AMD Instinct MI350, gfx950.
 *
 *  @sa include/numkong/dots.h
 *  @sa include/numkong/dots/rocm.cuh
 *
 *  The CDNA3 tile, its staging and its epilogue, with the dtypes differing only in how a lane's 32
 *  bytes of fragment are multiplied: BF16, F16 and I8 take two 16 × 16 MFMAs per slab, and the
 *  Float8, Float6 and Float4 codes go to @c v_mfma_scale_f32_16x16x128_f8f6f4 as they are, with
 *  every block scale at 2⁰. Float6 codes are packed from their bytes into the dense 6-bit stream
 *  the MFMA reads. U8 is offset into I8 and restored from the row and column byte sums, and I4 and
 *  U4 widen into I8. Packs keep rows as they are, as on the @c rocm capability. The Float norms
 *  serve the CDNA5 tile too.
 */
#ifndef NUMKONG_DOTS_CDNA4_CUH
#define NUMKONG_DOTS_CDNA4_CUH

#if NUMKONG_ARCH_ROCM_
#if NUMKONG_ARCH_ROCM_CDNA4_

#include "numkong/dots/cdna3.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Instructions

typedef int nk_i32x8_cdna4_t __attribute__((vector_size(32)));
typedef __bf16 nk_bf16x8_cdna4_t __attribute__((vector_size(16)));
typedef _Float16 nk_f16x8_cdna4_t __attribute__((vector_size(16)));

/** Every E8M0 block scale at 127, 2⁰, whichever byte the MFMA selects. */
enum { nk_unit_scales_cdna4_k = 0x7F7F7F7F };

NUMKONG_DEVICE nk_i32x4_cdna3_t nk_i32x4_load_cdna4_(nk_u32_t const words[4]) {
    nk_i32x4_cdna3_t const vector = {(int)words[0], (int)words[1], (int)words[2], (int)words[3]};
    return vector;
}

/** Up to 8 words of Float8, Float6 or Float4 codes; the MFMA reads as many as the format needs. */
NUMKONG_DEVICE nk_i32x8_cdna4_t nk_i32x8_load_cdna4_(nk_u32_t const *words, unsigned count) {
    nk_i32x8_cdna4_t vector = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (unsigned index = 0; index < 8; ++index)
        if (index < count) vector[index] = (int)words[index];
    return vector;
}

NUMKONG_DEVICE void nk_mfma_bf16_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t const b[4]) {
    nk_f32x4_store_cdna3_(accumulator, __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                                           __builtin_bit_cast(nk_bf16x8_cdna4_t, nk_i32x4_load_cdna4_(a)),
                                           __builtin_bit_cast(nk_bf16x8_cdna4_t, nk_i32x4_load_cdna4_(b)),
                                           nk_f32x4_load_cdna3_(accumulator), 0, 0, 0));
}

NUMKONG_DEVICE void nk_mfma_f16_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t const b[4]) {
    nk_f32x4_store_cdna3_(accumulator, __builtin_amdgcn_mfma_f32_16x16x32_f16(
                                           __builtin_bit_cast(nk_f16x8_cdna4_t, nk_i32x4_load_cdna4_(a)),
                                           __builtin_bit_cast(nk_f16x8_cdna4_t, nk_i32x4_load_cdna4_(b)),
                                           nk_f32x4_load_cdna3_(accumulator), 0, 0, 0));
}

NUMKONG_DEVICE void nk_mfma_i8_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t const b[4]) {
    nk_i32x4_cdna3_t const sum = {accumulator[0].i, accumulator[1].i, accumulator[2].i, accumulator[3].i};
    nk_i32x4_cdna3_t const result = __builtin_amdgcn_mfma_i32_16x16x64_i8(nk_i32x4_load_cdna4_(a),
                                                                          nk_i32x4_load_cdna4_(b), sum, 0, 0, 0);
    accumulator[0].i = result[0], accumulator[1].i = result[1], accumulator[2].i = result[2];
    accumulator[3].i = result[3];
}

/*  One 16 × 16 × 128 step per format: @c cbsz and @c blgp select E4M3 as 0, E5M2 as 1, E2M3 as 2,
 *  E3M2 as 3 and E2M1 as 4, for 8, 8, 6, 6 and 4 words a lane. */

NUMKONG_DEVICE void nk_mfma_e4m3_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[8], nk_u32_t const b[8]) {
    nk_f32x4_store_cdna3_(
        accumulator, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                         nk_i32x8_load_cdna4_(a, 8), nk_i32x8_load_cdna4_(b, 8), nk_f32x4_load_cdna3_(accumulator), 0,
                         0, 0, nk_unit_scales_cdna4_k, 0, nk_unit_scales_cdna4_k));
}

NUMKONG_DEVICE void nk_mfma_e5m2_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[8], nk_u32_t const b[8]) {
    nk_f32x4_store_cdna3_(
        accumulator, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                         nk_i32x8_load_cdna4_(a, 8), nk_i32x8_load_cdna4_(b, 8), nk_f32x4_load_cdna3_(accumulator), 1,
                         1, 0, nk_unit_scales_cdna4_k, 0, nk_unit_scales_cdna4_k));
}

NUMKONG_DEVICE void nk_mfma_e2m3_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[6], nk_u32_t const b[6]) {
    nk_f32x4_store_cdna3_(
        accumulator, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                         nk_i32x8_load_cdna4_(a, 6), nk_i32x8_load_cdna4_(b, 6), nk_f32x4_load_cdna3_(accumulator), 2,
                         2, 0, nk_unit_scales_cdna4_k, 0, nk_unit_scales_cdna4_k));
}

NUMKONG_DEVICE void nk_mfma_e3m2_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[6], nk_u32_t const b[6]) {
    nk_f32x4_store_cdna3_(
        accumulator, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                         nk_i32x8_load_cdna4_(a, 6), nk_i32x8_load_cdna4_(b, 6), nk_f32x4_load_cdna3_(accumulator), 3,
                         3, 0, nk_unit_scales_cdna4_k, 0, nk_unit_scales_cdna4_k));
}

NUMKONG_DEVICE void nk_mfma_e2m1_cdna4_(nk_fui32_t accumulator[4], nk_u32_t const a[4], nk_u32_t const b[4]) {
    nk_f32x4_store_cdna3_(
        accumulator, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                         nk_i32x8_load_cdna4_(a, 4), nk_i32x8_load_cdna4_(b, 4), nk_f32x4_load_cdna3_(accumulator), 4,
                         4, 0, nk_unit_scales_cdna4_k, 0, nk_unit_scales_cdna4_k));
}

#pragma endregion Instructions

#pragma region Conversions

/** Four 6-bit codes, one in the low bits of each byte, as 24 dense bits, code @c i at bit 6 × i. */
NUMKONG_DEVICE nk_u32_t nk_b8x4_to_b6x4_cdna4_(nk_u32_t codes) {
    return (codes & 0x3Fu) | ((codes >> 2) & 0xFC0u) | ((codes >> 4) & 0x3F000u) | ((codes >> 6) & 0xFC0000u);
}

/** Thirty-two 6-bit codes, one per byte, as the 192-bit stream a Float6 MFMA operand holds. */
NUMKONG_DEVICE void nk_b8x32_to_b6x32_cdna4_(nk_u32_t const codes[8], nk_u32_t packed[6]) {
    nk_u32_t dense[8];
#pragma unroll
    for (unsigned word = 0; word < 8; ++word) dense[word] = nk_b8x4_to_b6x4_cdna4_(codes[word]);
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        nk_u32_t const *quad = dense + half * 4;
        packed[half * 3 + 0] = quad[0] | (quad[1] << 24);
        packed[half * 3 + 1] = (quad[1] >> 8) | (quad[2] << 16);
        packed[half * 3 + 2] = (quad[2] >> 16) | (quad[3] << 8);
    }
}

#pragma endregion Conversions

/*  Each adds the squares of one 16-byte chunk of a staged row. Float8 and E3M2 square their scaled
 *  F16 widenings, and E2M3 and E2M1 their scaled integer ones, so the tile scales them back. */
#pragma region Norms

NUMKONG_DEVICE void nk_e5m2_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned byte = 0; byte < 16; ++byte) {
        nk_e5m2_t const code = (nk_e5m2_t)(words[byte / 4] >> (byte % 4 * 8));
        nk_f32_t value;
        nk_e5m2_to_f32_simt_(&code, &value);
        *real_sum = __fmaf_rn(value, value, *real_sum);
    }
}

NUMKONG_DEVICE void nk_e4m3_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned byte = 0; byte < 16; ++byte) {
        nk_u32_t const code = (words[byte / 4] >> (byte % 4 * 8)) & 0xFFu;
        // S.1111.111 is E4M3's NaN, which the F16 widening would read as 480.
        nk_f32_t const value = (code & 0x7Fu) == 0x7Fu ? __int_as_float(0x7FC00000) : nk_e4m3_to_scaled_f32_(code);
        *real_sum = __fmaf_rn(value, value, *real_sum);
    }
}

NUMKONG_DEVICE void nk_e3m2_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned byte = 0; byte < 16; ++byte) {
        nk_f32_t const value = nk_e3m2_to_scaled_f32_((words[byte / 4] >> (byte % 4 * 8)) & 0xFFu);
        *real_sum = __fmaf_rn(value, value, *real_sum);
    }
}

NUMKONG_DEVICE void nk_e2m3_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t const magnitudes = nk_e2m3x4_to_u8x4_magnitudes_(words[word]);
        *integer_sum = nk_dot_u8x4_rocm_(magnitudes, magnitudes, *integer_sum);
    }
}

NUMKONG_DEVICE void nk_e2m1_norm_update_cdna4_(nk_u32_t const words[4], nk_u32_t *integer_sum, nk_f32_t *real_sum) {
    // Squares of twice each magnitude, {0, 1, 4, 9, 16, 36, 64, 144}, 4 nibbles a lookup.
#pragma unroll
    for (unsigned word = 0; word < 4; ++word) {
        nk_u32_t const low = nk_byte_permute_rocm_(0x90402410u, 0x09040100u, words[word] & 0x07070707u);
        nk_u32_t const high = nk_byte_permute_rocm_(0x90402410u, 0x09040100u, (words[word] >> 4) & 0x07070707u);
        *integer_sum = nk_dot_u8x4_rocm_(low, 0x01010101u, *integer_sum);
        *integer_sum = nk_dot_u8x4_rocm_(high, 0x01010101u, *integer_sum);
    }
}

#pragma endregion Norms

#if NUMKONG_TARGET_CDNA4

/*  Each folds one 128-byte slab. Two MFMAs that read words 0-3 and 4-7 of every fragment cover
 *  the slab once, since A and B share each lane's depth mapping; the outer loop over those halves
 *  keeps dependent MFMAs 16 apart. */
#pragma region Multiplies

NUMKONG_DEVICE void nk_dots_bf16_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_bf16_cdna4_(accumulators[row_tile][column_tile], a[row_tile] + half * 4,
                                    b[column_tile] + half * 4);
}

NUMKONG_DEVICE void nk_dots_f16_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_f16_cdna4_(accumulators[row_tile][column_tile], a[row_tile] + half * 4,
                                   b[column_tile] + half * 4);
}

NUMKONG_DEVICE void nk_dots_e5m2_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
            nk_mfma_e5m2_cdna4_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile]);
}

NUMKONG_DEVICE void nk_dots_e4m3_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
            nk_mfma_e4m3_cdna4_(accumulators[row_tile][column_tile], a[row_tile], b[column_tile]);
}

NUMKONG_DEVICE void nk_dots_e3m2_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
    nk_u32_t a_packed[4][6], b_packed[4][6];
#pragma unroll
    for (unsigned tile = 0; tile < 4; ++tile)
        nk_b8x32_to_b6x32_cdna4_(a[tile], a_packed[tile]), nk_b8x32_to_b6x32_cdna4_(b[tile], b_packed[tile]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
            nk_mfma_e3m2_cdna4_(accumulators[row_tile][column_tile], a_packed[row_tile], b_packed[column_tile]);
}

NUMKONG_DEVICE void nk_dots_e2m3_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
    nk_u32_t a_packed[4][6], b_packed[4][6];
#pragma unroll
    for (unsigned tile = 0; tile < 4; ++tile)
        nk_b8x32_to_b6x32_cdna4_(a[tile], a_packed[tile]), nk_b8x32_to_b6x32_cdna4_(b[tile], b_packed[tile]);
#pragma unroll
    for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
        for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
            nk_mfma_e2m3_cdna4_(accumulators[row_tile][column_tile], a_packed[row_tile], b_packed[column_tile]);
}

NUMKONG_DEVICE void nk_dots_e2m1_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                                 nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_e2m1_cdna4_(accumulators[row_tile][column_tile], a[row_tile] + half * 4,
                                    b[column_tile] + half * 4);
}

NUMKONG_DEVICE void nk_dots_i8_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                               nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half)
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_i8_cdna4_(accumulators[row_tile][column_tile], a[row_tile] + half * 4,
                                  b[column_tile] + half * 4);
}

/** U8 codes offset by −128 into I8 by flipping each byte's top bit, which the epilogue undoes. */
NUMKONG_DEVICE void nk_dots_u8_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                               nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
        nk_u32_t a_offset[4][4], b_offset[4][4];
#pragma unroll
        for (unsigned tile = 0; tile < 4; ++tile)
#pragma unroll
            for (unsigned word = 0; word < 4; ++word)
                a_offset[tile][word] = a[tile][half * 4 + word] ^ 0x80808080u,
                b_offset[tile][word] = b[tile][half * 4 + word] ^ 0x80808080u;
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_i8_cdna4_(accumulators[row_tile][column_tile], a_offset[row_tile], b_offset[column_tile]);
    }
}

/** Widens nibble fragments into I8, 2 words of nibbles becoming the 4 words of one MFMA. */
NUMKONG_DEVICE void nk_dots_widened_i8_multiply_cdna4_(nk_cross_widen_t widen, nk_fui32_t accumulators[4][4][4],
                                                       nk_u32_t const a[4][8], nk_u32_t const b[4][8]) {
#pragma unroll
    for (unsigned step = 0; step < 4; ++step) {
        nk_u32_t a_widened[4][4], b_widened[4][4];
#pragma unroll
        for (unsigned tile = 0; tile < 4; ++tile) {
            widen(a[tile][step * 2], &a_widened[tile][0], &a_widened[tile][1]);
            widen(a[tile][step * 2 + 1], &a_widened[tile][2], &a_widened[tile][3]);
            widen(b[tile][step * 2], &b_widened[tile][0], &b_widened[tile][1]);
            widen(b[tile][step * 2 + 1], &b_widened[tile][2], &b_widened[tile][3]);
        }
#pragma unroll
        for (unsigned row_tile = 0; row_tile < 4; ++row_tile)
#pragma unroll
            for (unsigned column_tile = 0; column_tile < 4; ++column_tile)
                nk_mfma_i8_cdna4_(accumulators[row_tile][column_tile], a_widened[row_tile], b_widened[column_tile]);
    }
}

NUMKONG_DEVICE void nk_dots_i4_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                               nk_u32_t const b[4][8]) {
    nk_dots_widened_i8_multiply_cdna4_(nk_i4x8_to_i8x8_, accumulators, a, b);
}

NUMKONG_DEVICE void nk_dots_u4_multiply_cdna4_(nk_fui32_t accumulators[4][4][4], nk_u32_t const a[4][8],
                                               nk_u32_t const b[4][8]) {
    nk_dots_widened_i8_multiply_cdna4_(nk_u4x8_to_u8x8_, accumulators, a, b);
}

#pragma endregion Multiplies

#pragma region BF16

nk_define_cross_pack_rocm_(bf16, cdna4, bf16, bf16, nk_load_b8_, /*norm_value_type=*/f32, nk_bf16_lane_sumsq_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, bf16, cdna4, cdna3, bf16, bf16, f32, /*depth_simd_dimensions=*/8,
                      /*dimensions_per_value=*/1, nk_dots_bf16_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion BF16

#pragma region F16

nk_define_cross_pack_rocm_(f16, cdna4, f16, f16, nk_load_b8_, /*norm_value_type=*/f32, nk_f16_lane_sumsq_,
                           /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, f16, cdna4, cdna3, f16, f16, f32, /*depth_simd_dimensions=*/8, /*dimensions_per_value=*/1,
                      nk_dots_f16_multiply_cdna4_, nk_cross_epilogue_f32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                      NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion F16

#pragma region E5M2

nk_define_cross_pack_rocm_(e5m2, cdna4, e5m2, e5m2, nk_load_b8_, /*norm_value_type=*/f32, nk_e5m2_lane_sumsq_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e5m2, cdna4, cdna3, e5m2, e5m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e5m2_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E5M2

#pragma region E4M3

nk_define_cross_pack_rocm_(e4m3, cdna4, e4m3, e4m3, nk_load_b8_, /*norm_value_type=*/f32, nk_e4m3_lane_sumsq_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e4m3, cdna4, cdna3, e4m3, e4m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e4m3_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E4M3

#pragma region E3M2

nk_define_cross_pack_rocm_(e3m2, cdna4, e3m2, e3m2, nk_load_b8_, /*norm_value_type=*/f32, nk_e3m2_lane_sumsq_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e3m2, cdna4, cdna3, e3m2, e3m2, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e3m2_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E3M2

#pragma region E2M3

nk_define_cross_pack_rocm_(e2m3, cdna4, e2m3, e2m3, nk_load_b8_, /*norm_value_type=*/f32, nk_e2m3_lane_sumsq_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, e2m3, cdna4, cdna3, e2m3, e2m3, f32, /*depth_simd_dimensions=*/16,
                      /*dimensions_per_value=*/1, nk_dots_e2m3_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E2M3

#pragma region E2M1

nk_define_cross_pack_rocm_(e2m1, cdna4, e2m1x2, e2m1x2, nk_load_b8_, /*norm_value_type=*/f32, nk_e2m1_lane_sumsq_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, e2m1, cdna4, cdna3, e2m1x2, e2m1x2, f32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_e2m1_multiply_cdna4_, nk_cross_epilogue_f32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion E2M1

#pragma region I8

nk_define_cross_pack_rocm_(i8, cdna4, i8, i8, nk_load_b8_, /*norm_value_type=*/u32, nk_i8_lane_sumsq_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, i8, cdna4, cdna3, i8, i8, i32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1,
                      nk_dots_i8_multiply_cdna4_, nk_cross_epilogue_i32_k, /*output_scale=*/1.0f, nk_cross_norm_f32_k,
                      NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion I8

#pragma region I4

nk_define_cross_pack_rocm_(i4, cdna4, i4x2, i4x2, nk_load_b8_, /*norm_value_type=*/u32, nk_i4_lane_sumsq_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, i4, cdna4, cdna3, i4x2, i4x2, i32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_i4_multiply_cdna4_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion I4

#pragma region U8

nk_define_cross_pack_rocm_(u8, cdna4, u8, u8, nk_load_b8_, /*norm_value_type=*/u32, nk_u8_lane_sumsq_,
                           /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1)
nk_define_cross_rocm_(dot, u8, cdna4, cdna3, u8, u8, u32, /*depth_simd_dimensions=*/16, /*dimensions_per_value=*/1,
                      nk_dots_u8_multiply_cdna4_, nk_cross_epilogue_offset_u32_k, /*output_scale=*/1.0f,
                      nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion U8

#pragma region U4

nk_define_cross_pack_rocm_(u4, cdna4, u4x2, u4x2, nk_load_b8_, /*norm_value_type=*/u32, nk_u4_lane_sumsq_,
                           /*depth_simd_dimensions=*/32, /*dimensions_per_value=*/2)
nk_define_cross_rocm_(dot, u4, cdna4, cdna3, u4x2, u4x2, u32, /*depth_simd_dimensions=*/32,
                      /*dimensions_per_value=*/2, nk_dots_u4_multiply_cdna4_, nk_cross_epilogue_i32_k,
                      /*output_scale=*/1.0f, nk_cross_norm_f32_k, NUMKONG_NULL, /*norm_scale=*/1.0f)

#pragma endregion U4

#endif // NUMKONG_TARGET_CDNA4

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ROCM_CDNA4_
#endif // NUMKONG_ARCH_ROCM_
#endif // NUMKONG_DOTS_CDNA4_CUH
