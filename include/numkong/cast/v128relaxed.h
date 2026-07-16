/**
 *  @brief SIMD-accelerated Type Conversions for WASM.
 *  @file include/numkong/cast/v128relaxed.h
 */

#ifndef NK_CAST_V128RELAXED_H
#define NK_CAST_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/cast/serial.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/** @brief Native WASM SIMD 128-bit load. */
NK_HELPER_INLINE void nk_load_b128_v128relaxed_(void const *src, nk_b128_vec_t *dst) {
    dst->v128 = wasm_v128_load(src);
}
/** @brief Native WASM SIMD 256-bit load using two v128 loads. */
NK_HELPER_INLINE void nk_load_b256_v128relaxed_(void const *src, nk_b256_vec_t *dst) {
    dst->v128s[0] = wasm_v128_load(src);
    dst->v128s[1] = wasm_v128_load((char const *)src + 16);
}
/** @brief Native WASM SIMD 128-bit store. */
NK_HELPER_INLINE void nk_store_b128_v128relaxed_(nk_b128_vec_t const *src, void *dst) {
    wasm_v128_store(dst, src->v128);
}
/** @brief Native WASM SIMD 256-bit store using two v128 stores. */
NK_HELPER_INLINE void nk_store_b256_v128relaxed_(nk_b256_vec_t const *src, void *dst) {
    wasm_v128_store(dst, src->v128s[0]);
    wasm_v128_store((char *)dst + 16, src->v128s[1]);
}

/** @brief BF16 is the upper 16 bits of F32, so zero-extend to u32 and shift left by 16. */
NK_HELPER_INLINE nk_b128_vec_t nk_bf16x4_to_f32x4_v128relaxed_(nk_b64_vec_t bf16_vec) {
    v128_t bf16_i64x2 = wasm_i64x2_splat(bf16_vec.u64);
    v128_t bf16_low_u32x4 = wasm_u32x4_extend_low_u16x8(bf16_i64x2);
    nk_b128_vec_t result;
    result.v128 = wasm_i32x4_shl(bf16_low_u32x4, 16);
    return result;
}

/**
 *  @brief F16→F32 via Giesen's magic-number multiply trick.
 *  @see https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/
 *
 *  Shifts the 15-bit magnitude into F32 exponent+mantissa position, then multiplies
 *  by 2^112 (magic = 0x77800000) to rebias the exponent. This single multiply also
 *  correctly normalizes F16 subnormals into F32 normals — no branching or FPU
 *  integer-to-float conversion needed. Inf/NaN (exp=31) overflows the multiply and
 *  is fixed with a comparison + blend.
 */
NK_HELPER_INLINE nk_b128_vec_t nk_f16x4_to_f32x4_v128relaxed_(nk_b64_vec_t f16_vec) {
    v128_t raw_i64x2 = wasm_i64x2_splat(f16_vec.u64);
    v128_t raw_u32x4 = wasm_u32x4_extend_low_u16x8(raw_i64x2);

    // Extract sign and unsigned magnitude
    v128_t sign_u32x4 = wasm_v128_and(raw_u32x4, wasm_i32x4_splat(0x8000));
    v128_t sign_f32_u32x4 = wasm_i32x4_shl(sign_u32x4, 16);
    v128_t magnitude_u32x4 = wasm_v128_and(raw_u32x4, wasm_i32x4_splat(0x7FFF));

    // Shift mantissa+exponent into F32 position and multiply by magic 2^112
    v128_t shifted_u32x4 = wasm_i32x4_shl(magnitude_u32x4, 13);
    v128_t magic_f32x4 = wasm_i32x4_splat(0x77800000);
    v128_t rebiased_f32x4 = wasm_f32x4_mul((v128_t)shifted_u32x4, (v128_t)magic_f32x4);

    // Fix inf/NaN: exp=31 after shift becomes 0x1F<<13 = 0x000F8000, ×2^112 overflows.
    // Detect via threshold on shifted magnitude and apply direct rebias instead.
    v128_t infnan_threshold_u32x4 = wasm_i32x4_splat(0x38800000);
    v128_t infnan_mask_u32x4 = wasm_u32x4_ge(shifted_u32x4, infnan_threshold_u32x4);
    v128_t direct_u32x4 = wasm_v128_or(shifted_u32x4, wasm_i32x4_splat(0x70000000));
    v128_t result_u32x4 = wasm_i32x4_relaxed_laneselect(direct_u32x4, rebiased_f32x4, infnan_mask_u32x4);

    // Apply sign
    result_u32x4 = wasm_v128_or(result_u32x4, sign_f32_u32x4);

    nk_b128_vec_t result;
    result.v128 = result_u32x4;
    return result;
}

/**
 *  @brief E4M3→F32 via Giesen's magic multiply (×2^120).
 *  Shift 7-bit magnitude left by 20 into f32 position, multiply by 2^120 to rebias exponent.
 *  The multiply also normalizes subnormals. NaN fixup for magnitude 0x7F only.
 */
NK_HELPER_INLINE nk_b128_vec_t nk_e4m3x4_to_f32x4_v128relaxed_(nk_b32_vec_t e4m3_vec) {
    v128_t raw_u32x4 = wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(wasm_i32x4_splat(e4m3_vec.u32)));
    v128_t sign_u32x4 = wasm_i32x4_shl(wasm_v128_and(raw_u32x4, wasm_i32x4_splat(0x80)), 24);
    v128_t nonsign_u32x4 = wasm_v128_and(raw_u32x4, wasm_i32x4_splat(0x7F));
    v128_t shifted_u32x4 = wasm_i32x4_shl(nonsign_u32x4, 20);
    v128_t rebiased_f32x4 = wasm_f32x4_mul((v128_t)shifted_u32x4, (v128_t)wasm_i32x4_splat(0x7B800000)); // 2^120
    v128_t is_nan_u32x4 = wasm_i32x4_eq(nonsign_u32x4, wasm_i32x4_splat(0x7F));
    v128_t nan_u32x4 = wasm_v128_or(sign_u32x4, wasm_i32x4_splat(0x7FC00000));
    v128_t result_u32x4 = wasm_i32x4_relaxed_laneselect(nan_u32x4, rebiased_f32x4, is_nan_u32x4);
    nk_b128_vec_t result_vec;
    result_vec.v128 = wasm_v128_or(result_u32x4, sign_u32x4);
    return result_vec;
}

/**
 *  @brief E5M2→F32 via Giesen's magic multiply (×2^112).
 *  Same exponent encoding as F16 (5-bit, bias=15). Shift 7-bit magnitude left by 21,
 *  multiply by 2^112 to rebias. Inf/NaN fixup for exp=31 (nonsign > 123).
 */
NK_HELPER_INLINE nk_b128_vec_t nk_e5m2x4_to_f32x4_v128relaxed_(nk_b32_vec_t e5m2_vec) {
    v128_t raw_u32x4 = wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(wasm_i32x4_splat(e5m2_vec.u32)));
    v128_t sign_u32x4 = wasm_i32x4_shl(wasm_v128_and(raw_u32x4, wasm_i32x4_splat(0x80)), 24);
    v128_t nonsign_u32x4 = wasm_v128_and(raw_u32x4, wasm_i32x4_splat(0x7F));
    v128_t shifted_u32x4 = wasm_i32x4_shl(nonsign_u32x4, 21);
    v128_t rebiased_f32x4 = wasm_f32x4_mul((v128_t)shifted_u32x4, (v128_t)wasm_i32x4_splat(0x77800000)); // 2^112
    v128_t is_infnan_u32x4 = wasm_u32x4_gt(nonsign_u32x4, wasm_i32x4_splat(123));
    v128_t result_u32x4 = wasm_v128_or(rebiased_f32x4, wasm_v128_and(is_infnan_u32x4, wasm_i32x4_splat(0x7F800000)));
    nk_b128_vec_t result_vec;
    result_vec.v128 = wasm_v128_or(result_u32x4, sign_u32x4);
    return result_vec;
}

/**
 *  @brief E2M3→F32 via Giesen's magic multiply (×2^126).
 *  S EE MMM (bias=1). Shift 5-bit magnitude left by 20, multiply by 2^126 to rebias.
 *  No inf/NaN in E2M3FN format, so no fixup needed.
 */
NK_HELPER_INLINE nk_b128_vec_t nk_e2m3x4_to_f32x4_v128relaxed_(nk_b32_vec_t e2m3_vec) {
    v128_t raw_u32x4 = wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(wasm_i32x4_splat(e2m3_vec.u32)));
    v128_t sign_u32x4 = wasm_i32x4_shl(wasm_v128_and(raw_u32x4, wasm_i32x4_splat(0x20)), 26);
    v128_t nonsign_u32x4 = wasm_v128_and(raw_u32x4, wasm_i32x4_splat(0x1F));
    v128_t shifted_u32x4 = wasm_i32x4_shl(nonsign_u32x4, 20);
    v128_t rebiased_f32x4 = wasm_f32x4_mul((v128_t)shifted_u32x4, (v128_t)wasm_i32x4_splat(0x7E800000)); // 2^126
    nk_b128_vec_t result_vec;
    result_vec.v128 = wasm_v128_or(rebiased_f32x4, sign_u32x4);
    return result_vec;
}

/**
 *  @brief E3M2→F32 via Giesen's magic multiply (×2^124).
 *  S EEE MM (bias=3). Shift 5-bit magnitude left by 21, multiply by 2^124 to rebias.
 *  No inf/NaN in E3M2FN format, so no fixup needed.
 */
NK_HELPER_INLINE nk_b128_vec_t nk_e3m2x4_to_f32x4_v128relaxed_(nk_b32_vec_t e3m2_vec) {
    v128_t raw_u32x4 = wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(wasm_i32x4_splat(e3m2_vec.u32)));
    v128_t sign_u32x4 = wasm_i32x4_shl(wasm_v128_and(raw_u32x4, wasm_i32x4_splat(0x20)), 26);
    v128_t nonsign_u32x4 = wasm_v128_and(raw_u32x4, wasm_i32x4_splat(0x1F));
    v128_t shifted_u32x4 = wasm_i32x4_shl(nonsign_u32x4, 21);
    v128_t rebiased_f32x4 = wasm_f32x4_mul((v128_t)shifted_u32x4, (v128_t)wasm_i32x4_splat(0x7D800000)); // 2^124
    nk_b128_vec_t result_vec;
    result_vec.v128 = wasm_v128_or(rebiased_f32x4, sign_u32x4);
    return result_vec;
}

/** @brief Convert 4x i8 → f32x4 (WASM). Widen i8→i16→i32, convert to f32. */
NK_HELPER_INLINE nk_b128_vec_t nk_i8x4_to_f32x4_v128relaxed_(nk_b32_vec_t in_vec) {
    v128_t in_i8x16 = wasm_i32x4_splat(in_vec.u32);
    v128_t in_i16x8 = wasm_i16x8_extend_low_i8x16(in_i8x16);
    v128_t in_i32x4 = wasm_i32x4_extend_low_i16x8(in_i16x8);
    nk_b128_vec_t result_vec;
    result_vec.v128 = wasm_f32x4_convert_i32x4(in_i32x4);
    return result_vec;
}

/** @brief Convert 4x u8 → f32x4 (WASM). Widen u8→u16→u32, convert to f32. */
NK_HELPER_INLINE nk_b128_vec_t nk_u8x4_to_f32x4_v128relaxed_(nk_b32_vec_t in_vec) {
    v128_t in_u8x16 = wasm_i32x4_splat(in_vec.u32);
    v128_t in_u16x8 = wasm_u16x8_extend_low_u8x16(in_u8x16);
    v128_t in_u32x4 = wasm_u32x4_extend_low_u16x8(in_u16x8);
    nk_b128_vec_t result_vec;
    result_vec.v128 = wasm_f32x4_convert_u32x4(in_u32x4);
    return result_vec;
}

/** @brief Convert f32x4 → 4x bf16 via RNE rounding (WASM). */
NK_HELPER_INLINE nk_b64_vec_t nk_f32x4_to_bf16x4_v128relaxed_(nk_b128_vec_t hub_vec) {
    v128_t bits_u32x4 = hub_vec.v128;
    v128_t lsb_u32x4 = wasm_v128_and(wasm_u32x4_shr(bits_u32x4, 16), wasm_i32x4_splat(1));
    v128_t rounded_u32x4 = wasm_i32x4_add(bits_u32x4, wasm_i32x4_add(wasm_i32x4_splat(0x7FFF), lsb_u32x4));
    v128_t bf16_u32x4 = wasm_u32x4_shr(rounded_u32x4, 16);
    v128_t packed_u16x8 = wasm_u16x8_narrow_i32x4(bf16_u32x4, bf16_u32x4);
    nk_b64_vec_t result_vec;
    result_vec.u64 = (nk_u64_t)wasm_i64x2_extract_lane(packed_u16x8, 0);
    return result_vec;
}

/**
 *  @brief F32→F16 via bit manipulation with RNE (WASM).
 *  Handles normal, subnormal, overflow (→inf), and inf/NaN cases.
 */
NK_HELPER_INLINE nk_b64_vec_t nk_f32x4_to_f16x4_v128relaxed_(nk_b128_vec_t hub_vec) {
    v128_t bits_u32x4 = hub_vec.v128;
    v128_t sign_u32x4 = wasm_i32x4_shl(wasm_u32x4_shr(bits_u32x4, 31), 15);
    v128_t f32_exp_u32x4 = wasm_v128_and(wasm_u32x4_shr(bits_u32x4, 23), wasm_i32x4_splat(0xFF));
    v128_t f32_mant_u32x4 = wasm_v128_and(bits_u32x4, wasm_i32x4_splat(0x007FFFFF));

    // Normal path: rebias exponent (127→15), RNE round mantissa 23→10 bits
    v128_t f16_exp_i32x4 = wasm_i32x4_sub(f32_exp_u32x4, wasm_i32x4_splat(112));
    v128_t significand_u32x4 = wasm_v128_or(f32_mant_u32x4, wasm_i32x4_splat(0x00800000));
    v128_t lsb_u32x4 = wasm_v128_and(wasm_u32x4_shr(significand_u32x4, 13), wasm_i32x4_splat(1));
    v128_t rounded_u32x4 = wasm_i32x4_add(significand_u32x4, wasm_i32x4_add(wasm_i32x4_splat(0x0FFF), lsb_u32x4));
    v128_t carry_u32x4 = wasm_u32x4_shr(rounded_u32x4, 24);
    v128_t f16_mant_u32x4 = wasm_v128_and(wasm_u32x4_shr(rounded_u32x4, 13), wasm_i32x4_splat(0x3FF));
    v128_t carry_mask_u32x4 = wasm_i32x4_eq(carry_u32x4, wasm_i32x4_splat(1));
    f16_mant_u32x4 = wasm_v128_andnot(f16_mant_u32x4, carry_mask_u32x4);
    f16_exp_i32x4 = wasm_i32x4_add(f16_exp_i32x4, carry_u32x4);

    // Clamp exponent and assemble normal result
    v128_t clamped_exp_i32x4 = wasm_i32x4_max(f16_exp_i32x4, wasm_i32x4_splat(1));
    clamped_exp_i32x4 = wasm_i32x4_min(clamped_exp_i32x4, wasm_i32x4_splat(30));
    v128_t normal_result_u32x4 = wasm_v128_or(sign_u32x4,
                                              wasm_v128_or(wasm_i32x4_shl(clamped_exp_i32x4, 10), f16_mant_u32x4));

    // Overflow → infinity
    v128_t overflow_mask_u32x4 = wasm_i32x4_gt(f16_exp_i32x4, wasm_i32x4_splat(30));
    v128_t inf_result_u32x4 = wasm_v128_or(sign_u32x4, wasm_i32x4_splat(0x7C00));
    normal_result_u32x4 = wasm_i32x4_relaxed_laneselect(inf_result_u32x4, normal_result_u32x4, overflow_mask_u32x4);

    // Underflow → zero (exp <= 0 after rebias, ignoring subnormals for simplicity)
    v128_t underflow_mask_u32x4 = wasm_i32x4_lt(f16_exp_i32x4, wasm_i32x4_splat(1));
    normal_result_u32x4 = wasm_i32x4_relaxed_laneselect(sign_u32x4, normal_result_u32x4, underflow_mask_u32x4);

    // Inf/NaN passthrough: f32 exp=255
    v128_t infnan_mask_u32x4 = wasm_i32x4_eq(f32_exp_u32x4, wasm_i32x4_splat(255));
    v128_t nan_payload_u32x4 = wasm_v128_or(wasm_u32x4_shr(f32_mant_u32x4, 13), wasm_i32x4_splat(1));
    v128_t mant_nonzero_u32x4 = wasm_i32x4_ne(f32_mant_u32x4, wasm_i32x4_splat(0));
    v128_t nan_result_u32x4 = wasm_v128_or(
        sign_u32x4, wasm_v128_or(wasm_i32x4_splat(0x7C00), wasm_v128_and(nan_payload_u32x4, mant_nonzero_u32x4)));
    normal_result_u32x4 = wasm_i32x4_relaxed_laneselect(nan_result_u32x4, normal_result_u32x4, infnan_mask_u32x4);

    // F32 zero/denorm → f16 zero
    v128_t f32_zero_mask_u32x4 = wasm_i32x4_eq(f32_exp_u32x4, wasm_i32x4_splat(0));
    normal_result_u32x4 = wasm_i32x4_relaxed_laneselect(sign_u32x4, normal_result_u32x4, f32_zero_mask_u32x4);

    // Pack 4x u32 → 4x u16
    v128_t packed_u16x8 = wasm_u16x8_narrow_i32x4(normal_result_u32x4, normal_result_u32x4);
    nk_b64_vec_t result_vec;
    result_vec.u64 = (nk_u64_t)wasm_i64x2_extract_lane(packed_u16x8, 0);
    return result_vec;
}

/** @brief Convert f32x4 → 4x e4m3 via bit manipulation with RNE (WASM). */
NK_HELPER_INLINE nk_b32_vec_t nk_f32x4_to_e4m3x4_v128relaxed_(nk_b128_vec_t hub_vec) {
    v128_t bits_u32x4 = hub_vec.v128;
    v128_t sign_u32x4 = wasm_u32x4_shr(bits_u32x4, 31);
    v128_t f32_exp_u32x4 = wasm_v128_and(wasm_u32x4_shr(bits_u32x4, 23), wasm_i32x4_splat(0xFF));

    // RNE mantissa rounding from 23 to 3 bits
    v128_t significand_u32x4 = wasm_v128_or(wasm_v128_and(bits_u32x4, wasm_i32x4_splat(0x007FFFFF)),
                                            wasm_i32x4_splat(0x00800000));
    v128_t lsb_u32x4 = wasm_v128_and(wasm_u32x4_shr(significand_u32x4, 20), wasm_i32x4_splat(1));
    v128_t rounded_u32x4 = wasm_i32x4_add(significand_u32x4, wasm_i32x4_add(wasm_i32x4_splat(0x0007FFFF), lsb_u32x4));
    v128_t carry_u32x4 = wasm_u32x4_shr(rounded_u32x4, 24);
    v128_t f32_mant_u32x4 = wasm_v128_and(wasm_u32x4_shr(rounded_u32x4, 20), wasm_i32x4_splat(0x07));
    v128_t carry_mask_u32x4 = wasm_i32x4_eq(carry_u32x4, wasm_i32x4_splat(1));
    f32_mant_u32x4 = wasm_v128_andnot(f32_mant_u32x4, carry_mask_u32x4);
    v128_t e4m3_exp_i32x4 = wasm_i32x4_sub(wasm_i32x4_add(f32_exp_u32x4, carry_u32x4), wasm_i32x4_splat(120));

    v128_t is_subnormal_u32x4 = wasm_i32x4_lt(e4m3_exp_i32x4, wasm_i32x4_splat(1));
    v128_t overflow_u32x4 = wasm_i32x4_gt(e4m3_exp_i32x4, wasm_i32x4_splat(15));

    // Normal path
    v128_t clamped_exp_i32x4 = wasm_i32x4_max(e4m3_exp_i32x4, wasm_i32x4_splat(1));
    clamped_exp_i32x4 = wasm_i32x4_min(clamped_exp_i32x4, wasm_i32x4_splat(15));
    v128_t is_max_exp_u32x4 = wasm_i32x4_eq(clamped_exp_i32x4, wasm_i32x4_splat(15));
    v128_t max_mant_u32x4 = wasm_i32x4_relaxed_laneselect(wasm_i32x4_splat(6), wasm_i32x4_splat(7), is_max_exp_u32x4);
    v128_t normal_mant_u32x4 = wasm_i32x4_min(f32_mant_u32x4, max_mant_u32x4);
    normal_mant_u32x4 = wasm_i32x4_relaxed_laneselect(wasm_i32x4_splat(0x06), normal_mant_u32x4, overflow_u32x4);
    v128_t normal_u32x4 = wasm_v128_or(wasm_i32x4_shl(sign_u32x4, 7),
                                       wasm_v128_or(wasm_i32x4_shl(clamped_exp_i32x4, 3), normal_mant_u32x4));

    // Subnormal path
    v128_t abs_f32x4 = wasm_v128_and(hub_vec.v128, wasm_i32x4_splat(0x7FFFFFFF));
    v128_t scaled_f32x4 = wasm_f32x4_mul((v128_t)abs_f32x4, wasm_f32x4_splat(512.0f));
    v128_t sub_mant_i32x4 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(scaled_f32x4));
    v128_t promotes_u32x4 = wasm_i32x4_gt(sub_mant_i32x4, wasm_i32x4_splat(7));
    sub_mant_i32x4 = wasm_i32x4_min(sub_mant_i32x4, wasm_i32x4_splat(7));
    sub_mant_i32x4 = wasm_i32x4_max(sub_mant_i32x4, wasm_i32x4_splat(0));
    v128_t subnormal_u32x4 = wasm_v128_or(wasm_i32x4_shl(sign_u32x4, 7), sub_mant_i32x4);
    v128_t first_normal_u32x4 = wasm_v128_or(wasm_i32x4_shl(sign_u32x4, 7), wasm_i32x4_splat(0x08));
    subnormal_u32x4 = wasm_i32x4_relaxed_laneselect(first_normal_u32x4, subnormal_u32x4, promotes_u32x4);

    v128_t e4m3_u32x4 = wasm_i32x4_relaxed_laneselect(subnormal_u32x4, normal_u32x4, is_subnormal_u32x4);

    // Pack 4x u32 → 4x u8
    v128_t packed_u16x8 = wasm_u16x8_narrow_i32x4(e4m3_u32x4, e4m3_u32x4);
    v128_t packed_u8x16 = wasm_u8x16_narrow_i16x8(packed_u16x8, packed_u16x8);
    nk_b32_vec_t result_vec;
    result_vec.u32 = (nk_u32_t)wasm_i32x4_extract_lane(packed_u8x16, 0);
    return result_vec;
}

/** @brief Convert f32x4 → 4x e5m2 via bit manipulation with RNE (WASM). */
NK_HELPER_INLINE nk_b32_vec_t nk_f32x4_to_e5m2x4_v128relaxed_(nk_b128_vec_t hub_vec) {
    v128_t bits_u32x4 = hub_vec.v128;
    v128_t sign_u32x4 = wasm_u32x4_shr(bits_u32x4, 31);
    v128_t f32_exp_u32x4 = wasm_v128_and(wasm_u32x4_shr(bits_u32x4, 23), wasm_i32x4_splat(0xFF));

    // RNE mantissa rounding from 23 to 2 bits
    v128_t significand_u32x4 = wasm_v128_or(wasm_v128_and(bits_u32x4, wasm_i32x4_splat(0x007FFFFF)),
                                            wasm_i32x4_splat(0x00800000));
    v128_t lsb_u32x4 = wasm_v128_and(wasm_u32x4_shr(significand_u32x4, 21), wasm_i32x4_splat(1));
    v128_t rounded_u32x4 = wasm_i32x4_add(significand_u32x4, wasm_i32x4_add(wasm_i32x4_splat(0x000FFFFF), lsb_u32x4));
    v128_t carry_u32x4 = wasm_u32x4_shr(rounded_u32x4, 24);
    v128_t f32_mant_u32x4 = wasm_v128_and(wasm_u32x4_shr(rounded_u32x4, 21), wasm_i32x4_splat(0x03));
    v128_t carry_mask_u32x4 = wasm_i32x4_eq(carry_u32x4, wasm_i32x4_splat(1));
    f32_mant_u32x4 = wasm_v128_andnot(f32_mant_u32x4, carry_mask_u32x4);
    v128_t e5m2_exp_i32x4 = wasm_i32x4_sub(wasm_i32x4_add(f32_exp_u32x4, carry_u32x4), wasm_i32x4_splat(112));

    v128_t is_subnormal_u32x4 = wasm_i32x4_lt(e5m2_exp_i32x4, wasm_i32x4_splat(1));
    v128_t overflow_u32x4 = wasm_i32x4_gt(e5m2_exp_i32x4, wasm_i32x4_splat(31));

    // Normal path: overflow → infinity (exp=31, mant=0)
    v128_t clamped_exp_i32x4 = wasm_i32x4_max(e5m2_exp_i32x4, wasm_i32x4_splat(1));
    clamped_exp_i32x4 = wasm_i32x4_min(clamped_exp_i32x4, wasm_i32x4_splat(31));
    v128_t normal_mant_u32x4 = wasm_i32x4_relaxed_laneselect(wasm_i32x4_splat(0), f32_mant_u32x4, overflow_u32x4);
    v128_t normal_u32x4 = wasm_v128_or(wasm_i32x4_shl(sign_u32x4, 7),
                                       wasm_v128_or(wasm_i32x4_shl(clamped_exp_i32x4, 2), normal_mant_u32x4));

    // Subnormal path
    v128_t abs_f32x4 = wasm_v128_and(hub_vec.v128, wasm_i32x4_splat(0x7FFFFFFF));
    v128_t scaled_f32x4 = wasm_f32x4_mul((v128_t)abs_f32x4, wasm_f32x4_splat(65536.0f));
    v128_t sub_mant_i32x4 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(scaled_f32x4));
    v128_t promotes_u32x4 = wasm_i32x4_gt(sub_mant_i32x4, wasm_i32x4_splat(3));
    sub_mant_i32x4 = wasm_i32x4_min(sub_mant_i32x4, wasm_i32x4_splat(3));
    sub_mant_i32x4 = wasm_i32x4_max(sub_mant_i32x4, wasm_i32x4_splat(0));
    v128_t subnormal_u32x4 = wasm_v128_or(wasm_i32x4_shl(sign_u32x4, 7), sub_mant_i32x4);
    v128_t first_normal_u32x4 = wasm_v128_or(wasm_i32x4_shl(sign_u32x4, 7), wasm_i32x4_splat(0x04));
    subnormal_u32x4 = wasm_i32x4_relaxed_laneselect(first_normal_u32x4, subnormal_u32x4, promotes_u32x4);

    v128_t e5m2_u32x4 = wasm_i32x4_relaxed_laneselect(subnormal_u32x4, normal_u32x4, is_subnormal_u32x4);

    v128_t packed_u16x8 = wasm_u16x8_narrow_i32x4(e5m2_u32x4, e5m2_u32x4);
    v128_t packed_u8x16 = wasm_u8x16_narrow_i16x8(packed_u16x8, packed_u16x8);
    nk_b32_vec_t result_vec;
    result_vec.u32 = (nk_u32_t)wasm_i32x4_extract_lane(packed_u8x16, 0);
    return result_vec;
}

/** @brief Convert f32x4 → 4x e2m3 via bit manipulation with RNE (WASM). */
NK_HELPER_INLINE nk_b32_vec_t nk_f32x4_to_e2m3x4_v128relaxed_(nk_b128_vec_t hub_vec) {
    v128_t bits_u32x4 = hub_vec.v128;
    v128_t sign_u32x4 = wasm_u32x4_shr(bits_u32x4, 31);
    v128_t f32_exp_u32x4 = wasm_v128_and(wasm_u32x4_shr(bits_u32x4, 23), wasm_i32x4_splat(0xFF));

    v128_t significand_u32x4 = wasm_v128_or(wasm_v128_and(bits_u32x4, wasm_i32x4_splat(0x007FFFFF)),
                                            wasm_i32x4_splat(0x00800000));
    v128_t lsb_u32x4 = wasm_v128_and(wasm_u32x4_shr(significand_u32x4, 20), wasm_i32x4_splat(1));
    v128_t rounded_u32x4 = wasm_i32x4_add(significand_u32x4, wasm_i32x4_add(wasm_i32x4_splat(0x0007FFFF), lsb_u32x4));
    v128_t carry_u32x4 = wasm_u32x4_shr(rounded_u32x4, 24);
    v128_t f32_mant_u32x4 = wasm_v128_and(wasm_u32x4_shr(rounded_u32x4, 20), wasm_i32x4_splat(0x07));
    v128_t carry_mask_u32x4 = wasm_i32x4_eq(carry_u32x4, wasm_i32x4_splat(1));
    f32_mant_u32x4 = wasm_v128_andnot(f32_mant_u32x4, carry_mask_u32x4);
    v128_t e2m3_exp_i32x4 = wasm_i32x4_sub(wasm_i32x4_add(f32_exp_u32x4, carry_u32x4), wasm_i32x4_splat(126));

    v128_t is_subnormal_u32x4 = wasm_i32x4_lt(e2m3_exp_i32x4, wasm_i32x4_splat(1));
    v128_t overflow_u32x4 = wasm_i32x4_gt(e2m3_exp_i32x4, wasm_i32x4_splat(3));

    v128_t clamped_exp_i32x4 = wasm_i32x4_max(e2m3_exp_i32x4, wasm_i32x4_splat(1));
    clamped_exp_i32x4 = wasm_i32x4_min(clamped_exp_i32x4, wasm_i32x4_splat(3));
    v128_t normal_mant_u32x4 = wasm_i32x4_relaxed_laneselect(wasm_i32x4_splat(0x07), f32_mant_u32x4, overflow_u32x4);
    v128_t normal_u32x4 = wasm_v128_or(wasm_i32x4_shl(sign_u32x4, 5),
                                       wasm_v128_or(wasm_i32x4_shl(clamped_exp_i32x4, 3), normal_mant_u32x4));

    v128_t abs_f32x4 = wasm_v128_and(hub_vec.v128, wasm_i32x4_splat(0x7FFFFFFF));
    v128_t scaled_f32x4 = wasm_f32x4_mul((v128_t)abs_f32x4, wasm_f32x4_splat(8.0f));
    v128_t sub_mant_i32x4 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(scaled_f32x4));
    v128_t promotes_u32x4 = wasm_i32x4_gt(sub_mant_i32x4, wasm_i32x4_splat(7));
    sub_mant_i32x4 = wasm_i32x4_min(sub_mant_i32x4, wasm_i32x4_splat(7));
    sub_mant_i32x4 = wasm_i32x4_max(sub_mant_i32x4, wasm_i32x4_splat(0));
    v128_t subnormal_u32x4 = wasm_v128_or(wasm_i32x4_shl(sign_u32x4, 5), sub_mant_i32x4);
    v128_t first_normal_u32x4 = wasm_v128_or(wasm_i32x4_shl(sign_u32x4, 5), wasm_i32x4_splat(0x08));
    subnormal_u32x4 = wasm_i32x4_relaxed_laneselect(first_normal_u32x4, subnormal_u32x4, promotes_u32x4);

    v128_t e2m3_u32x4 = wasm_i32x4_relaxed_laneselect(subnormal_u32x4, normal_u32x4, is_subnormal_u32x4);

    v128_t packed_u16x8 = wasm_u16x8_narrow_i32x4(e2m3_u32x4, e2m3_u32x4);
    v128_t packed_u8x16 = wasm_u8x16_narrow_i16x8(packed_u16x8, packed_u16x8);
    nk_b32_vec_t result_vec;
    result_vec.u32 = (nk_u32_t)wasm_i32x4_extract_lane(packed_u8x16, 0);
    return result_vec;
}

/** @brief Convert f32x4 → 4x e3m2 via bit manipulation with RNE (WASM). */
NK_HELPER_INLINE nk_b32_vec_t nk_f32x4_to_e3m2x4_v128relaxed_(nk_b128_vec_t hub_vec) {
    v128_t bits_u32x4 = hub_vec.v128;
    v128_t sign_u32x4 = wasm_u32x4_shr(bits_u32x4, 31);
    v128_t f32_exp_u32x4 = wasm_v128_and(wasm_u32x4_shr(bits_u32x4, 23), wasm_i32x4_splat(0xFF));

    v128_t significand_u32x4 = wasm_v128_or(wasm_v128_and(bits_u32x4, wasm_i32x4_splat(0x007FFFFF)),
                                            wasm_i32x4_splat(0x00800000));
    v128_t lsb_u32x4 = wasm_v128_and(wasm_u32x4_shr(significand_u32x4, 21), wasm_i32x4_splat(1));
    v128_t rounded_u32x4 = wasm_i32x4_add(significand_u32x4, wasm_i32x4_add(wasm_i32x4_splat(0x000FFFFF), lsb_u32x4));
    v128_t carry_u32x4 = wasm_u32x4_shr(rounded_u32x4, 24);
    v128_t f32_mant_u32x4 = wasm_v128_and(wasm_u32x4_shr(rounded_u32x4, 21), wasm_i32x4_splat(0x03));
    v128_t carry_mask_u32x4 = wasm_i32x4_eq(carry_u32x4, wasm_i32x4_splat(1));
    f32_mant_u32x4 = wasm_v128_andnot(f32_mant_u32x4, carry_mask_u32x4);
    v128_t e3m2_exp_i32x4 = wasm_i32x4_sub(wasm_i32x4_add(f32_exp_u32x4, carry_u32x4), wasm_i32x4_splat(124));

    v128_t is_subnormal_u32x4 = wasm_i32x4_lt(e3m2_exp_i32x4, wasm_i32x4_splat(1));
    v128_t overflow_u32x4 = wasm_i32x4_gt(e3m2_exp_i32x4, wasm_i32x4_splat(7));

    v128_t clamped_exp_i32x4 = wasm_i32x4_max(e3m2_exp_i32x4, wasm_i32x4_splat(1));
    clamped_exp_i32x4 = wasm_i32x4_min(clamped_exp_i32x4, wasm_i32x4_splat(7));
    v128_t normal_mant_u32x4 = wasm_i32x4_relaxed_laneselect(wasm_i32x4_splat(0x03), f32_mant_u32x4, overflow_u32x4);
    v128_t normal_u32x4 = wasm_v128_or(wasm_i32x4_shl(sign_u32x4, 5),
                                       wasm_v128_or(wasm_i32x4_shl(clamped_exp_i32x4, 2), normal_mant_u32x4));

    v128_t abs_f32x4 = wasm_v128_and(hub_vec.v128, wasm_i32x4_splat(0x7FFFFFFF));
    v128_t scaled_f32x4 = wasm_f32x4_mul((v128_t)abs_f32x4, wasm_f32x4_splat(16.0f));
    v128_t sub_mant_i32x4 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(scaled_f32x4));
    v128_t promotes_u32x4 = wasm_i32x4_gt(sub_mant_i32x4, wasm_i32x4_splat(3));
    sub_mant_i32x4 = wasm_i32x4_min(sub_mant_i32x4, wasm_i32x4_splat(3));
    sub_mant_i32x4 = wasm_i32x4_max(sub_mant_i32x4, wasm_i32x4_splat(0));
    v128_t subnormal_u32x4 = wasm_v128_or(wasm_i32x4_shl(sign_u32x4, 5), sub_mant_i32x4);
    v128_t first_normal_u32x4 = wasm_v128_or(wasm_i32x4_shl(sign_u32x4, 5), wasm_i32x4_splat(0x04));
    subnormal_u32x4 = wasm_i32x4_relaxed_laneselect(first_normal_u32x4, subnormal_u32x4, promotes_u32x4);

    v128_t e3m2_u32x4 = wasm_i32x4_relaxed_laneselect(subnormal_u32x4, normal_u32x4, is_subnormal_u32x4);

    v128_t packed_u16x8 = wasm_u16x8_narrow_i32x4(e3m2_u32x4, e3m2_u32x4);
    v128_t packed_u8x16 = wasm_u8x16_narrow_i16x8(packed_u16x8, packed_u16x8);
    nk_b32_vec_t result_vec;
    result_vec.u32 = (nk_u32_t)wasm_i32x4_extract_lane(packed_u8x16, 0);
    return result_vec;
}

/** @brief Convert f32x4 → 4x i8 with saturation (WASM). */
NK_HELPER_INLINE nk_b32_vec_t nk_f32x4_to_i8x4_v128relaxed_(nk_b128_vec_t hub_vec) {
    v128_t clamped_f32x4 = wasm_f32x4_min(wasm_f32x4_max(hub_vec.v128, wasm_f32x4_splat(-128.0f)),
                                          wasm_f32x4_splat(127.0f));
    v128_t result_i32x4 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(clamped_f32x4));
    v128_t result_i16x8 = wasm_i16x8_narrow_i32x4(result_i32x4, result_i32x4);
    v128_t result_i8x16 = wasm_i8x16_narrow_i16x8(result_i16x8, result_i16x8);
    nk_b32_vec_t result_vec;
    result_vec.u32 = (nk_u32_t)wasm_i32x4_extract_lane(result_i8x16, 0);
    return result_vec;
}

/** @brief Convert f32x4 → 4x u8 with saturation (WASM). */
NK_HELPER_INLINE nk_b32_vec_t nk_f32x4_to_u8x4_v128relaxed_(nk_b128_vec_t hub_vec) {
    v128_t clamped_f32x4 = wasm_f32x4_min(wasm_f32x4_max(hub_vec.v128, wasm_f32x4_splat(0.0f)),
                                          wasm_f32x4_splat(255.0f));
    v128_t result_u32x4 = wasm_u32x4_trunc_sat_f32x4(wasm_f32x4_nearest(clamped_f32x4));
    v128_t result_u16x8 = wasm_u16x8_narrow_i32x4(result_u32x4, result_u32x4);
    v128_t result_u8x16 = wasm_u8x16_narrow_i16x8(result_u16x8, result_u16x8);
    nk_b32_vec_t result_vec;
    result_vec.u32 = (nk_u32_t)wasm_i32x4_extract_lane(result_u8x16, 0);
    return result_vec;
}

NK_API_COMPTIME void nk_cast_v128relaxed(void const *from, nk_dtype_t from_type, nk_size_t n, void *to,
                                         nk_dtype_t to_type) {
    // Same-type fast path
    if (from_type == to_type) {
        nk_size_t size_bits = nk_dtype_bits(from_type);
        if (size_bits > 0) nk_copy_bytes_(to, from, nk_size_divide_round_up_(n * size_bits, 8));
        return;
    }

    // Validate supported types
    int from_ok = (from_type == nk_f32_k || from_type == nk_f16_k || from_type == nk_bf16_k || from_type == nk_e4m3_k ||
                   from_type == nk_e5m2_k || from_type == nk_e2m3_k || from_type == nk_e3m2_k || from_type == nk_i8_k ||
                   from_type == nk_u8_k);
    int to_ok = (to_type == nk_f32_k || to_type == nk_f16_k || to_type == nk_bf16_k || to_type == nk_e4m3_k ||
                 to_type == nk_e5m2_k || to_type == nk_e2m3_k || to_type == nk_e3m2_k || to_type == nk_i8_k ||
                 to_type == nk_u8_k);

    if (!from_ok || !to_ok) {
        nk_cast_serial(from, from_type, n, to, to_type);
        return;
    }

    // F32 hub: 4 elements per iteration
    nk_size_t batches = n / 4;
    nk_size_t tail = n % 4;
    nk_size_t from_step = nk_size_divide_round_up_(4 * nk_dtype_bits(from_type), NK_BITS_PER_BYTE);
    nk_size_t to_step = nk_size_divide_round_up_(4 * nk_dtype_bits(to_type), NK_BITS_PER_BYTE);
    nk_u8_t const *from_ptr = (nk_u8_t const *)from;
    nk_u8_t *to_ptr = (nk_u8_t *)to;

    for (nk_size_t i = 0; i < batches; ++i, from_ptr += from_step, to_ptr += to_step) {
        nk_b128_vec_t hub_vec;

        // Upcast to f32x4 hub using size-appropriate loads
        if (from_step == 16) { hub_vec.v128 = wasm_v128_load(from_ptr); }
        else if (from_step == 8) {
            nk_b64_vec_t raw64_vec;
            raw64_vec.u64 = (nk_u64_t)wasm_i64x2_extract_lane(wasm_v128_load64_zero(from_ptr), 0);
            switch (from_type) {
            case nk_f16_k: hub_vec = nk_f16x4_to_f32x4_v128relaxed_(raw64_vec); break;
            case nk_bf16_k: hub_vec = nk_bf16x4_to_f32x4_v128relaxed_(raw64_vec); break;
            default: break;
            }
        }
        else if (from_step == 4) {
            nk_b32_vec_t raw32_vec;
            raw32_vec.u32 = (nk_u32_t)wasm_i32x4_extract_lane(wasm_v128_load32_zero(from_ptr), 0);
            switch (from_type) {
            case nk_e4m3_k: hub_vec = nk_e4m3x4_to_f32x4_v128relaxed_(raw32_vec); break;
            case nk_e5m2_k: hub_vec = nk_e5m2x4_to_f32x4_v128relaxed_(raw32_vec); break;
            case nk_e2m3_k: hub_vec = nk_e2m3x4_to_f32x4_v128relaxed_(raw32_vec); break;
            case nk_e3m2_k: hub_vec = nk_e3m2x4_to_f32x4_v128relaxed_(raw32_vec); break;
            case nk_i8_k: hub_vec = nk_i8x4_to_f32x4_v128relaxed_(raw32_vec); break;
            case nk_u8_k: hub_vec = nk_u8x4_to_f32x4_v128relaxed_(raw32_vec); break;
            default: break;
            }
        }
        else hub_vec.v128 = wasm_f32x4_splat(0);

        // Downcast from f32x4 hub and store using half-register stores
        switch (to_type) {
        case nk_f32_k: wasm_v128_store(to_ptr, hub_vec.v128); break;
        case nk_f16_k: *(nk_u64_t *)to_ptr = nk_f32x4_to_f16x4_v128relaxed_(hub_vec).u64; break;
        case nk_bf16_k: *(nk_u64_t *)to_ptr = nk_f32x4_to_bf16x4_v128relaxed_(hub_vec).u64; break;
        case nk_e4m3_k: *(nk_u32_t *)to_ptr = nk_f32x4_to_e4m3x4_v128relaxed_(hub_vec).u32; break;
        case nk_e5m2_k: *(nk_u32_t *)to_ptr = nk_f32x4_to_e5m2x4_v128relaxed_(hub_vec).u32; break;
        case nk_e2m3_k: *(nk_u32_t *)to_ptr = nk_f32x4_to_e2m3x4_v128relaxed_(hub_vec).u32; break;
        case nk_e3m2_k: *(nk_u32_t *)to_ptr = nk_f32x4_to_e3m2x4_v128relaxed_(hub_vec).u32; break;
        case nk_i8_k: *(nk_u32_t *)to_ptr = nk_f32x4_to_i8x4_v128relaxed_(hub_vec).u32; break;
        case nk_u8_k: *(nk_u32_t *)to_ptr = nk_f32x4_to_u8x4_v128relaxed_(hub_vec).u32; break;
        default: break;
        }
    }

    // Handle tail elements with serial fallback
    if (tail) nk_cast_serial(from_ptr, from_type, tail, to_ptr, to_type);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_CAST_V128RELAXED_H
