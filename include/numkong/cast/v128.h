/**
 *  @brief SIMD-accelerated Type Conversions for WASM.
 *  @file include/numkong/cast/v128.h
 *  @author Ash Vardanian
 *  @date September 12, 2026
 */

#ifndef NK_CAST_V128_H
#define NK_CAST_V128_H

#if NK_TARGET_V128

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

/** @brief Native WASM SIMD 128-bit load. */
NK_HELPER_INLINE void nk_load_b128_v128_(void const *src, nk_b128_vec_t *dst) { dst->v128 = wasm_v128_load(src); }
/** @brief Native WASM SIMD 256-bit load using two v128 loads. */
NK_HELPER_INLINE void nk_load_b256_v128_(void const *src, nk_b256_vec_t *dst) {
    dst->v128s[0] = wasm_v128_load(src);
    dst->v128s[1] = wasm_v128_load((char const *)src + 16);
}
/** @brief Native WASM SIMD 128-bit store. */
NK_HELPER_INLINE void nk_store_b128_v128_(nk_b128_vec_t const *src, void *dst) { wasm_v128_store(dst, src->v128); }
/** @brief Native WASM SIMD 256-bit store using two v128 stores. */
NK_HELPER_INLINE void nk_store_b256_v128_(nk_b256_vec_t const *src, void *dst) {
    wasm_v128_store(dst, src->v128s[0]);
    wasm_v128_store((char *)dst + 16, src->v128s[1]);
}

/** @brief BF16 is the upper 16 bits of F32, so zero-extend to u32 and shift left by 16. */
NK_HELPER_INLINE nk_b128_vec_t nk_bf16x4_to_f32x4_v128_(nk_b64_vec_t bf16_vec) {
    v128_t bf16_i64x2 = wasm_i64x2_splat(bf16_vec.u64);
    v128_t bf16_low_u32x4 = wasm_u32x4_extend_low_u16x8(bf16_i64x2);
    nk_b128_vec_t result;
    result.v128 = wasm_i32x4_shl(bf16_low_u32x4, 16);
    return result;
}

/**
 *  @brief E5M2→F32 via Giesen's magic multiply (×2^112).
 *  Same exponent encoding as F16 (5-bit, bias=15). Shift 7-bit magnitude left by 21,
 *  multiply by 2^112 to rebias. Inf/NaN fixup for exp=31 (nonsign > 123).
 */
NK_HELPER_INLINE nk_b128_vec_t nk_e5m2x4_to_f32x4_v128_(nk_b32_vec_t e5m2_vec) {
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
NK_HELPER_INLINE nk_b128_vec_t nk_e2m3x4_to_f32x4_v128_(nk_b32_vec_t e2m3_vec) {
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
NK_HELPER_INLINE nk_b128_vec_t nk_e3m2x4_to_f32x4_v128_(nk_b32_vec_t e3m2_vec) {
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
NK_HELPER_INLINE nk_b128_vec_t nk_i8x4_to_f32x4_v128_(nk_b32_vec_t in_vec) {
    v128_t in_i8x16 = wasm_i32x4_splat(in_vec.u32);
    v128_t in_i16x8 = wasm_i16x8_extend_low_i8x16(in_i8x16);
    v128_t in_i32x4 = wasm_i32x4_extend_low_i16x8(in_i16x8);
    nk_b128_vec_t result_vec;
    result_vec.v128 = wasm_f32x4_convert_i32x4(in_i32x4);
    return result_vec;
}

/** @brief Convert 4x u8 → f32x4 (WASM). Widen u8→u16→u32, convert to f32. */
NK_HELPER_INLINE nk_b128_vec_t nk_u8x4_to_f32x4_v128_(nk_b32_vec_t in_vec) {
    v128_t in_u8x16 = wasm_i32x4_splat(in_vec.u32);
    v128_t in_u16x8 = wasm_u16x8_extend_low_u8x16(in_u8x16);
    v128_t in_u32x4 = wasm_u32x4_extend_low_u16x8(in_u16x8);
    nk_b128_vec_t result_vec;
    result_vec.v128 = wasm_f32x4_convert_u32x4(in_u32x4);
    return result_vec;
}

/** @brief Convert f32x4 → 4x bf16 via RNE rounding (WASM). */
NK_HELPER_INLINE nk_b64_vec_t nk_f32x4_to_bf16x4_v128_(nk_b128_vec_t hub_vec) {
    v128_t bits_u32x4 = hub_vec.v128;
    v128_t lsb_u32x4 = wasm_v128_and(wasm_u32x4_shr(bits_u32x4, 16), wasm_i32x4_splat(1));
    v128_t rounded_u32x4 = wasm_i32x4_add(bits_u32x4, wasm_i32x4_add(wasm_i32x4_splat(0x7FFF), lsb_u32x4));
    v128_t bf16_u32x4 = wasm_u32x4_shr(rounded_u32x4, 16);
    v128_t packed_u16x8 = wasm_u16x8_narrow_i32x4(bf16_u32x4, bf16_u32x4);
    nk_b64_vec_t result_vec;
    result_vec.u64 = (nk_u64_t)wasm_i64x2_extract_lane(packed_u16x8, 0);
    return result_vec;
}

/** @brief Convert f32x4 → 4x i8 with saturation (WASM). */
NK_HELPER_INLINE nk_b32_vec_t nk_f32x4_to_i8x4_v128_(nk_b128_vec_t hub_vec) {
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
NK_HELPER_INLINE nk_b32_vec_t nk_f32x4_to_u8x4_v128_(nk_b128_vec_t hub_vec) {
    v128_t clamped_f32x4 = wasm_f32x4_min(wasm_f32x4_max(hub_vec.v128, wasm_f32x4_splat(0.0f)),
                                          wasm_f32x4_splat(255.0f));
    v128_t result_u32x4 = wasm_u32x4_trunc_sat_f32x4(wasm_f32x4_nearest(clamped_f32x4));
    v128_t result_u16x8 = wasm_u16x8_narrow_i32x4(result_u32x4, result_u32x4);
    v128_t result_u8x16 = wasm_u8x16_narrow_i16x8(result_u16x8, result_u16x8);
    nk_b32_vec_t result_vec;
    result_vec.u32 = (nk_u32_t)wasm_i32x4_extract_lane(result_u8x16, 0);
    return result_vec;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128
#endif // NK_CAST_V128_H
