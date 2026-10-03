/**
 *  @brief SIMD-accelerated Type Conversions for NEON.
 *  @file include/numkong/cast/neon.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/cast.h
 *
 *  @section neon_cast_instructions ARM NEON Conversion Instructions
 *
 *  Float ↔ integer conversions (Cortex-A76 class):
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      vcvtq_f32_s32               SCVTF (V.4S, V.4S)              3cy         2/cy
 *      vcvtq_f32_u32               UCVTF (V.4S, V.4S)              3cy         2/cy
 *      vcvtq_s32_f32               FCVTZS (V.4S, V.4S)             3cy         2/cy
 *      vcvtq_u32_f32               FCVTZU (V.4S, V.4S)             3cy         2/cy
 *
 *  Float precision conversions:
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      vcvt_f32_f16                FCVTL (V.4S, V.4H)              3cy         2/cy
 *      vcvt_f16_f32                FCVTN (V.4H, V.4S)              3cy         2/cy
 *      vcvt_f64_f32                FCVTL (V.2D, V.2S)              3cy         2/cy
 *      vcvt_f32_f64                FCVTN (V.2S, V.2D)              3cy         2/cy
 *
 *  Integer narrowing with saturation:
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      vqmovn_s32                  SQXTN (V.4H, V.4S)              3cy         2/cy
 *      vqmovn_u32                  UQXTN (V.4H, V.4S)              3cy         2/cy
 *      vqmovun_s32                 SQXTUN (V.4H, V.4S)             3cy         2/cy
 *
 *  BF16 support (ARMv8.6-A+):
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *      vcvtq_low_bf16_f32          BFCVTN (V.4H, V.4S)             3cy         1/cy
 *      vcvtq_high_bf16_f32         BFCVTN2 (V.8H, V.4S)            3cy         1/cy
 *
 *  BF16 conversions on baseline NEON (emulated via bit shifts):
 *  - bf16 → f32: vmovl_u16 + vshlq_n_u32 by 16
 *  - f32 → bf16: round-to-nearest + vshrn_n_u32 by 16
 *
 *  FP8 (E4M3/E5M2) conversions use NEON bit manipulation:
 *  - Field extraction: vandq, vshrq, vshlq
 *  - Blending: vbslq for conditional selection
 *  - Subnormal handling: vmulq_n_f32 with scale factors (1/512, 1/65536)
 */
#ifndef NK_CAST_NEON_H
#define NK_CAST_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"
#include "numkong/cast/serial.h"   // `nk_cast_serial`, `nk_dtype_bits`
#include "numkong/reduce/serial.h" // `nk_reduce_moments_f32_serial`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

NK_PUBLIC void nk_f16_to_f32_neon(nk_f16_t const *src, nk_f32_t *dest) {
    float16x4_t f16vec = vreinterpret_f16_u16(vld1_dup_u16((nk_u16_t const *)src));
    float32x4_t f32vec = vcvt_f32_f16(f16vec);
    *dest = vgetq_lane_f32(f32vec, 0);
}

NK_PUBLIC void nk_f32_to_f16_neon(nk_f32_t const *src, nk_f16_t *dest) {
    float32x4_t f32vec = vdupq_n_f32(*src);
    float16x4_t f16vec = vcvt_f16_f32(f32vec);
    vst1_lane_u16((nk_u16_t *)dest, vreinterpret_u16_f16(f16vec), 0);
}

#pragma region - Type Punned Loads and Stores

/** @brief Type-agnostic 128-bit full load (NEON). */
NK_INTERNAL void nk_load_b128_neon_(void const *src, nk_b128_vec_t *dst) {
    dst->u8x16 = vld1q_u8((nk_u8_t const *)src);
}

/** @brief Type-agnostic 256-bit full load (NEON). */
NK_INTERNAL void nk_load_b256_neon_(void const *src, nk_b256_vec_t *dst) {
    dst->u8x16s[0] = vld1q_u8((nk_u8_t const *)src);
    dst->u8x16s[1] = vld1q_u8((nk_u8_t const *)src + 16);
}

/** @brief Type-agnostic 128-bit full store (NEON). */
NK_INTERNAL void nk_store_b128_neon_(nk_b128_vec_t const *src, void *dst) { vst1q_u8((nk_u8_t *)dst, src->u8x16); }

/** @brief Type-agnostic 256-bit full store (NEON). */
NK_INTERNAL void nk_store_b256_neon_(nk_b256_vec_t const *src, void *dst) {
    vst1q_u8((nk_u8_t *)dst, src->u8x16s[0]);
    vst1q_u8((nk_u8_t *)dst + 16, src->u8x16s[1]);
}

/** @brief Type-agnostic 64-bit full load (NEON). */
NK_INTERNAL void nk_load_b64_neon_(void const *src, nk_b64_vec_t *dst) { dst->u8x8 = vld1_u8((nk_u8_t const *)src); }

#pragma endregion - Type Punned Loads and Stores

#pragma region - Vectorized Conversions

/** @brief Convert 4x e4m3 → f32x4 via bit manipulation (NEON).
 *  E4M3FN format: S EEEE MMM (bias=7). No ∞ representation.
 *  Only exp=15, mant=7 (0x7F) is NaN; exp=15, mant ∈ [0,6] are valid normals (max=448). */
NK_INTERNAL float32x4_t nk_e4m3x4_to_f32x4_neon_(nk_b32_vec_t src) {
    uint8x8_t e4m3_u8x8 = vcreate_u8(src.u32);
    uint16x8_t e4m3_u16x8 = vmovl_u8(e4m3_u8x8);
    uint32x4_t e4m3_u32x4 = vmovl_u16(vget_low_u16(e4m3_u16x8));
    uint32x4_t sign_u32x4 = vshlq_n_u32(vandq_u32(e4m3_u32x4, vdupq_n_u32(0x80)), 24);
    uint32x4_t exp_u32x4 = vandq_u32(vshrq_n_u32(e4m3_u32x4, 3), vdupq_n_u32(0x0F));
    uint32x4_t mant_u32x4 = vandq_u32(e4m3_u32x4, vdupq_n_u32(0x07));

    // Normal path: f32 = sign | ((exp+120)<<23) | (mant<<20)
    uint32x4_t f32_exp_u32x4 = vshlq_n_u32(vaddq_u32(exp_u32x4, vdupq_n_u32(120)), 23);
    uint32x4_t f32_mant_u32x4 = vshlq_n_u32(mant_u32x4, 20);
    uint32x4_t normal_u32x4 = vorrq_u32(sign_u32x4, vorrq_u32(f32_exp_u32x4, f32_mant_u32x4));

    // Subnormal path (exp=0, mant ≠ 0): value = ±mantissa × 2⁻⁹
    float32x4_t subnormal_f32x4 = vmulq_n_f32(vcvtq_f32_u32(mant_u32x4), 1.0f / 512.0f);
    uint32x4_t subnormal_u32x4 = vorrq_u32(vreinterpretq_u32_f32(subnormal_f32x4), sign_u32x4);

    // NaN path: E4M3FN only has NaN when exp=15 AND mant=7 (0x7F or 0xFF)
    uint32x4_t nan_u32x4 = vorrq_u32(sign_u32x4, vdupq_n_u32(0x7FC00000));
    uint32x4_t is_nan_mask = vandq_u32(vceqq_u32(exp_u32x4, vdupq_n_u32(15)), vceqq_u32(mant_u32x4, vdupq_n_u32(7)));

    // Blend paths: subnormal when exp=0, NaN when exp=15 && mant=7, else normal
    uint32x4_t exp_zero_mask = vceqq_u32(exp_u32x4, vdupq_n_u32(0));
    uint32x4_t result_u32x4 = vbslq_u32(exp_zero_mask, subnormal_u32x4, normal_u32x4);
    result_u32x4 = vbslq_u32(is_nan_mask, nan_u32x4, result_u32x4);
    return vreinterpretq_f32_u32(result_u32x4);
}

/** @brief Convert 4x e5m2 → f32x4 via bit manipulation (NEON).
 *  E5M2 format: S EEEEE MM (bias=15). F32: sign<<31, (exp+112)<<23, mant<<21.
 *  Handles subnormals (exp=0, mant ≠ 0), inf (exp=31, mant=0), and nan (exp=31, mant ≠ 0). */
NK_INTERNAL float32x4_t nk_e5m2x4_to_f32x4_neon_(nk_b32_vec_t src) {
    uint8x8_t e5m2_u8x8 = vcreate_u8(src.u32);
    uint16x8_t e5m2_u16x8 = vmovl_u8(e5m2_u8x8);
    uint32x4_t e5m2_u32x4 = vmovl_u16(vget_low_u16(e5m2_u16x8));
    uint32x4_t sign_u32x4 = vshlq_n_u32(vandq_u32(e5m2_u32x4, vdupq_n_u32(0x80)), 24);
    uint32x4_t exp_u32x4 = vandq_u32(vshrq_n_u32(e5m2_u32x4, 2), vdupq_n_u32(0x1F));
    uint32x4_t mant_u32x4 = vandq_u32(e5m2_u32x4, vdupq_n_u32(0x03));

    // Normal path: f32 = sign | ((exp+112)<<23) | (mant<<21)
    uint32x4_t f32_exp_u32x4 = vshlq_n_u32(vaddq_u32(exp_u32x4, vdupq_n_u32(112)), 23);
    uint32x4_t f32_mant_u32x4 = vshlq_n_u32(mant_u32x4, 21);
    uint32x4_t normal_u32x4 = vorrq_u32(sign_u32x4, vorrq_u32(f32_exp_u32x4, f32_mant_u32x4));

    // Subnormal path (exp=0, mant ≠ 0): value = ±mantissa × 2⁻¹⁶
    float32x4_t subnormal_f32x4 = vmulq_n_f32(vcvtq_f32_u32(mant_u32x4), 1.0f / 65536.0f);
    uint32x4_t subnormal_u32x4 = vorrq_u32(vreinterpretq_u32_f32(subnormal_f32x4), sign_u32x4);

    // Special path (exp=31): inf (mant=0) or nan (mant≠0)
    uint32x4_t infinity_u32x4 = vorrq_u32(sign_u32x4, vdupq_n_u32(0x7F800000));
    uint32x4_t nan_u32x4 = vorrq_u32(sign_u32x4, vdupq_n_u32(0x7FC00000));
    uint32x4_t mant_zero_mask = vceqq_u32(mant_u32x4, vdupq_n_u32(0));
    uint32x4_t special_u32x4 = vbslq_u32(mant_zero_mask, infinity_u32x4, nan_u32x4);

    // Blend paths based on exponent value
    uint32x4_t exp_zero_mask = vceqq_u32(exp_u32x4, vdupq_n_u32(0));
    uint32x4_t exp_max_mask = vceqq_u32(exp_u32x4, vdupq_n_u32(31));
    uint32x4_t result_u32x4 = vbslq_u32(exp_zero_mask, subnormal_u32x4, normal_u32x4);
    result_u32x4 = vbslq_u32(exp_max_mask, special_u32x4, result_u32x4);
    return vreinterpretq_f32_u32(result_u32x4);
}

/** @brief Convert 8x e4m3 → f16x8 via bit manipulation (NEON).
 *  E4M3FN format: S EEEE MMM (bias=7). F16: S EEEEE MMMMMMMMMM (bias=15).
 *  E4M3FN has no ∞; only exp=15, mant=7 is NaN. exp=15, mant ∈ [0,6] are valid normals. */
NK_INTERNAL float16x8_t nk_e4m3x8_to_f16x8_neon_(uint8x8_t e4m3_u8x8) {
    uint16x8_t e4m3_u16x8 = vmovl_u8(e4m3_u8x8);
    uint16x8_t sign_u16x8 = vshlq_n_u16(vandq_u16(e4m3_u16x8, vdupq_n_u16(0x80)), 8); // sign << 15
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(e4m3_u16x8, 3), vdupq_n_u16(0x0F));
    uint16x8_t mant_u16x8 = vandq_u16(e4m3_u16x8, vdupq_n_u16(0x07));

    // Normal path: F16_exp = E4M3_exp + 8, F16_mant = E4M3_mant << 7
    uint16x8_t f16_exp_u16x8 = vshlq_n_u16(vaddq_u16(exp_u16x8, vdupq_n_u16(8)), 10);
    uint16x8_t f16_mant_u16x8 = vshlq_n_u16(mant_u16x8, 7);
    uint16x8_t normal_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(f16_exp_u16x8, f16_mant_u16x8));

    // Subnormal path (exp=0, mant ≠ 0): E4M3 subnormal value = mant × 2⁻⁹ = mant ÷ 512
    // Compute arithmetically: mant → f32 → multiply → f16
    float32x4_t subnormal_low_f32x4 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(mant_u16x8))), 1.0f / 512.0f);
    float32x4_t subnormal_high_f32x4 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(mant_u16x8))), 1.0f / 512.0f);
    uint16x8_t subnormal_abs_u16x8 = vreinterpretq_u16_f16(
        vcombine_f16(vcvt_f16_f32(subnormal_low_f32x4), vcvt_f16_f32(subnormal_high_f32x4)));
    uint16x8_t subnormal_u16x8 = vorrq_u16(subnormal_abs_u16x8, sign_u16x8);

    // NaN path: E4M3FN only has NaN when exp=15 AND mant=7 (0x7F or 0xFF)
    uint16x8_t nan_u16x8 = vorrq_u16(sign_u16x8, vdupq_n_u16(0x7E00)); // F16 quiet NaN
    uint16x8_t is_nan_mask = vandq_u16(vceqq_u16(exp_u16x8, vdupq_n_u16(15)), vceqq_u16(mant_u16x8, vdupq_n_u16(7)));

    // Blend paths: subnormal when exp=0, NaN when exp=15 && mant=7, else normal
    uint16x8_t exp_zero_mask = vceqq_u16(exp_u16x8, vdupq_n_u16(0));
    uint16x8_t result_u16x8 = vbslq_u16(exp_zero_mask, subnormal_u16x8, normal_u16x8);
    result_u16x8 = vbslq_u16(is_nan_mask, nan_u16x8, result_u16x8);
    return vreinterpretq_f16_u16(result_u16x8);
}

/** @brief Convert 16x e4m3 → 2x f16x8 via TBL lookup (NEON).
 *  E4M3FN format: S EEEE MMM (bias=7) → F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Uses sign symmetry: negative LUT entries = positive XOR 0x80, so we strip sign,
 *  lookup 7-bit absolute value in 2× VQTBL4 (128 bytes), then OR sign back.
 *  Arithmetic for the low byte: (has_exp && !nan) ? (lsb << 7) : 0.
 *  Exact for all 256 input values including subnormals and NaN.
 *
 *  Performance (per 16 elements): ~10-12 instructions vs ~40 for 2× nk_e4m3x8_to_f16x8_neon_ */
NK_INTERNAL void nk_e4m3x16_to_f16x8x2_neon_(uint8x16_t input_u8x16, float16x8_t *result_low_f16x8,
                                             float16x8_t *result_high_f16x8) {
    // Precomputed LUT: F16 high byte for unsigned 7-bit e4m3 values (sign handled separately).
    // Quadrant 0: |e4m3| bytes 0x00..0x3F (exp=0..7)
    static nk_u8_t const table_q0_u8x64[64] = {
        0x00, 0x18, 0x1C, 0x1E, 0x20, 0x21, 0x22, 0x23, 0x24, 0x24, 0x25, 0x25, 0x26, 0x26, 0x27, 0x27,
        0x28, 0x28, 0x29, 0x29, 0x2A, 0x2A, 0x2B, 0x2B, 0x2C, 0x2C, 0x2D, 0x2D, 0x2E, 0x2E, 0x2F, 0x2F,
        0x30, 0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33, 0x34, 0x34, 0x35, 0x35, 0x36, 0x36, 0x37, 0x37,
        0x38, 0x38, 0x39, 0x39, 0x3A, 0x3A, 0x3B, 0x3B, 0x3C, 0x3C, 0x3D, 0x3D, 0x3E, 0x3E, 0x3F, 0x3F,
    };
    // Quadrant 1: |e4m3| bytes 0x40..0x7F (exp=8..15)
    static nk_u8_t const table_q1_u8x64[64] = {
        0x40, 0x40, 0x41, 0x41, 0x42, 0x42, 0x43, 0x43, 0x44, 0x44, 0x45, 0x45, 0x46, 0x46, 0x47, 0x47,
        0x48, 0x48, 0x49, 0x49, 0x4A, 0x4A, 0x4B, 0x4B, 0x4C, 0x4C, 0x4D, 0x4D, 0x4E, 0x4E, 0x4F, 0x4F,
        0x50, 0x50, 0x51, 0x51, 0x52, 0x52, 0x53, 0x53, 0x54, 0x54, 0x55, 0x55, 0x56, 0x56, 0x57, 0x57,
        0x58, 0x58, 0x59, 0x59, 0x5A, 0x5A, 0x5B, 0x5B, 0x5C, 0x5C, 0x5D, 0x5D, 0x5E, 0x5E, 0x5F, 0x7E,
    };

    uint8x16x4_t lut_q0 = vld1q_u8_x4(table_q0_u8x64);
    uint8x16x4_t lut_q1 = vld1q_u8_x4(table_q1_u8x64);

    // Strip sign bit, work with 7-bit absolute value
    uint8x16_t sign_u8x16 = vandq_u8(input_u8x16, vdupq_n_u8(0x80));
    uint8x16_t abs_u8x16 = vandq_u8(input_u8x16, vdupq_n_u8(0x7F));

    // High byte via 2× VQTBL4 on unsigned index, then OR sign back.
    // VQTBL4 returns 0 for out-of-range indices (>= 64), so results OR together cleanly.
    uint8x16_t high_q0_u8x16 = vqtbl4q_u8(lut_q0, abs_u8x16);
    uint8x16_t offset_q1_u8x16 = vsubq_u8(abs_u8x16, vdupq_n_u8(64));
    uint8x16_t high_q1_u8x16 = vqtbl4q_u8(lut_q1, offset_q1_u8x16);
    uint8x16_t high_bytes_u8x16 = vorrq_u8(vorrq_u8(high_q0_u8x16, high_q1_u8x16), sign_u8x16);

    // Low byte: (lsb << 7), masked to 0 for subnormals (exp=0) and NaN (exp=15, mant=7)
    uint8x16_t lsb_shifted_u8x16 = vshlq_n_u8(input_u8x16, 7);
    uint8x16_t exponent_bits_u8x16 = vandq_u8(input_u8x16, vdupq_n_u8(0x78));
    uint8x16_t has_exp_u8x16 = vcgtq_u8(exponent_bits_u8x16, vdupq_n_u8(0));
    // Exclude NaN: |input| == 0x7F means exp=15, mant=7
    uint8x16_t abs_input_u8x16 = vandq_u8(input_u8x16, vdupq_n_u8(0x7F));
    uint8x16_t not_nan_u8x16 = vmvnq_u8(vceqq_u8(abs_input_u8x16, vdupq_n_u8(0x7F)));
    uint8x16_t low_bytes_u8x16 = vandq_u8(lsb_shifted_u8x16, vandq_u8(has_exp_u8x16, not_nan_u8x16));

    // ZIP to interleave: [l0,l1..l15] + [h0,h1..h15] → [l0,h0,l1,h1..] (little-endian f16)
    uint8x16x2_t interleaved_u8x16x2 = vzipq_u8(low_bytes_u8x16, high_bytes_u8x16);
    *result_low_f16x8 = vreinterpretq_f16_u8(interleaved_u8x16x2.val[0]);
    *result_high_f16x8 = vreinterpretq_f16_u8(interleaved_u8x16x2.val[1]);
}

/** @brief Convert 8x e5m2 → f16x8 via bit shift (NEON).
 *  E5M2 (bias=15) and F16 (bias=15) share the same exponent bias, so conversion is trivial.
 *  E5M2: S EEEEE MM → F16: S EEEEE MM 00000000. Works for all: zero, subnormal, normal, inf, nan. */
NK_INTERNAL float16x8_t nk_e5m2x8_to_f16x8_neon_(uint8x8_t e5m2_u8x8) {
    uint16x8_t e5m2_u16x8 = vmovl_u8(e5m2_u8x8);
    return vreinterpretq_f16_u16(vshlq_n_u16(e5m2_u16x8, 8));
}

/** @brief Convert 8x e2m3 → f16x8 via direct bit manipulation (NEON).
 *  E2M3FN (FP6): S EE MMM (bias=1) → F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Handles subnormals (exp=0) via arithmetic conversion. No Inf/NaN in E2M3FN. */
NK_INTERNAL float16x8_t nk_e2m3x8_to_f16x8_neon_(uint8x8_t e2m3_u8x8) {
    // Widen to 16-bit for NEON operations
    uint16x8_t e2m3_u16x8 = vmovl_u8(e2m3_u8x8);

    // Extract fields: format is 0b00SEEMMM (6 bits used)
    uint16x8_t sign_u16x8 = vshlq_n_u16(vandq_u16(e2m3_u16x8, vdupq_n_u16(0x20)), 10); // sign << 15
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(e2m3_u16x8, 3), vdupq_n_u16(0x03));   // 2-bit exp
    uint16x8_t mant_u16x8 = vandq_u16(e2m3_u16x8, vdupq_n_u16(0x07));                  // 3-bit mant

    // Normal path: F16_exp = E2M3_exp + 14, F16_mant = E2M3_mant << 7
    uint16x8_t exp_rebiased_u16x8 = vaddq_u16(exp_u16x8, vdupq_n_u16(14));
    uint16x8_t exp_positioned_u16x8 = vshlq_n_u16(exp_rebiased_u16x8, 10);
    uint16x8_t mant_positioned_u16x8 = vshlq_n_u16(mant_u16x8, 7);
    uint16x8_t normal_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(exp_positioned_u16x8, mant_positioned_u16x8));

    // Subnormal path (exp=0): E2M3 subnormal = mant / 8
    // Compute via f32: mant → f32 → multiply → f16
    float32x4_t subnormal_low_f32x4 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(mant_u16x8))), 1.0f / 8.0f);
    float32x4_t subnormal_high_f32x4 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(mant_u16x8))), 1.0f / 8.0f);
    uint16x8_t subnormal_abs_u16x8 = vreinterpretq_u16_f16(
        vcombine_f16(vcvt_f16_f32(subnormal_low_f32x4), vcvt_f16_f32(subnormal_high_f32x4)));
    uint16x8_t subnormal_u16x8 = vorrq_u16(subnormal_abs_u16x8, sign_u16x8);

    // Blend: use subnormal result when exp=0, else normal
    uint16x8_t exp_zero_mask = vceqq_u16(exp_u16x8, vdupq_n_u16(0));
    uint16x8_t result_u16x8 = vbslq_u16(exp_zero_mask, subnormal_u16x8, normal_u16x8);

    return vreinterpretq_f16_u16(result_u16x8);
}

/** @brief Convert 8x e3m2 → f16x8 via direct bit manipulation (NEON).
 *  E3M2FN (FP6): S EEE MM (bias=3) → F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Handles subnormals (exp=0) via arithmetic conversion. No Inf/NaN in E3M2FN. */
NK_INTERNAL float16x8_t nk_e3m2x8_to_f16x8_neon_(uint8x8_t e3m2_u8x8) {
    // Widen to 16-bit for NEON operations
    uint16x8_t e3m2_u16x8 = vmovl_u8(e3m2_u8x8);

    // Extract fields: format is 0b00SEEEMM (6 bits used)
    uint16x8_t sign_u16x8 = vshlq_n_u16(vandq_u16(e3m2_u16x8, vdupq_n_u16(0x20)), 10); // sign << 15
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(e3m2_u16x8, 2), vdupq_n_u16(0x07));   // 3-bit exp
    uint16x8_t mant_u16x8 = vandq_u16(e3m2_u16x8, vdupq_n_u16(0x03));                  // 2-bit mant

    // Normal path: F16_exp = E3M2_exp + 12, F16_mant = E3M2_mant << 8
    uint16x8_t exp_rebiased_u16x8 = vaddq_u16(exp_u16x8, vdupq_n_u16(12));
    uint16x8_t exp_positioned_u16x8 = vshlq_n_u16(exp_rebiased_u16x8, 10);
    uint16x8_t mant_positioned_u16x8 = vshlq_n_u16(mant_u16x8, 8);
    uint16x8_t normal_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(exp_positioned_u16x8, mant_positioned_u16x8));

    // Subnormal path (exp=0): E3M2 subnormal = mant × 2^(-2) × (1/4) = mant / 16
    // Compute via f32: mant → f32 → multiply → f16
    float32x4_t subnormal_low_f32x4 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(mant_u16x8))), 1.0f / 16.0f);
    float32x4_t subnormal_high_f32x4 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(mant_u16x8))), 1.0f / 16.0f);
    uint16x8_t subnormal_abs_u16x8 = vreinterpretq_u16_f16(
        vcombine_f16(vcvt_f16_f32(subnormal_low_f32x4), vcvt_f16_f32(subnormal_high_f32x4)));
    uint16x8_t subnormal_u16x8 = vorrq_u16(subnormal_abs_u16x8, sign_u16x8);

    // Blend: use subnormal result when exp=0, else normal
    uint16x8_t exp_zero_mask = vceqq_u16(exp_u16x8, vdupq_n_u16(0));
    uint16x8_t result_u16x8 = vbslq_u16(exp_zero_mask, subnormal_u16x8, normal_u16x8);

    return vreinterpretq_f16_u16(result_u16x8);
}

/** @brief Convert 16x e2m3 → 2x f16x8 via TBL lookup (NEON).
 *  E2M3FN (FP6): S EE MMM (bias=1) → F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Uses precomputed lookup tables for 64 possible 6-bit values.
 *  VQTBL4 byte shuffle (p01) + VZIP interleave (p01) (~6 instructions, parallel execution) */
NK_INTERNAL void nk_e2m3x16_to_f16x8x2_neon_(uint8x16_t input_u8x16, float16x8_t *result_low_f16x8,
                                             float16x8_t *result_high_f16x8) {
    // E2M3FN → F16 conversion using TBL for high byte, arithmetic for low byte.
    // Sign symmetry: negative half = positive XOR 0x80, so we strip sign (bit 5),
    // lookup unsigned 5-bit value in VQTBL2 (32 bytes), then OR sign back.
    // E2M3FN: sign(1) exp(2) mant(3), bias=1
    // F16: sign(1) exp(5) mant(10), bias=15
    // Normal (exp!=0): f16 = (sign << 15) | ((exp + 14) << 10) | (mant << 7)
    // Subnormal (exp=0): f16 = mant/8 converted to f16
    //
    // Low byte pattern: E2M3 has 3 mantissa bits → f16 bits 9-7, so low byte (bits 7-0) is:
    //   - Subnormals (exp=0): always 0x00
    //   - Normals (exp≠0): (mant & 1) << 7 = 0x00 or 0x80
    // This simple pattern can be computed arithmetically, saving table registers!
    static nk_u8_t const table_high_u8x32[32] = {
        0x00, 0x30, 0x34, 0x36, 0x38, 0x39, 0x3A, 0x3B, // exp=0 (subnormals)
        0x3C, 0x3C, 0x3D, 0x3D, 0x3E, 0x3E, 0x3F, 0x3F, // exp=1 → f16_exp=15 (0x3C-0x3F)
        0x40, 0x40, 0x41, 0x41, 0x42, 0x42, 0x43, 0x43, // exp=2 → f16_exp=16 (0x40-0x43)
        0x44, 0x44, 0x45, 0x45, 0x46, 0x46, 0x47, 0x47, // exp=3 → f16_exp=17 (0x44-0x47)
    };

    // Load unsigned high byte table (2 registers instead of 4)
    uint8x16x2_t table_high_u8x16x2 = vld1q_u8_x2(table_high_u8x32);

    // Extract sign (bit 5 → bit 7 for F16 high byte) and 5-bit unsigned index
    uint8x16_t sign_u8x16 = vshlq_n_u8(vandq_u8(input_u8x16, vdupq_n_u8(0x20)), 2);
    uint8x16_t abs_indices_u8x16 = vandq_u8(input_u8x16, vdupq_n_u8(0x1F));

    // High bytes via VQTBL2 on unsigned index, then OR sign back
    uint8x16_t high_bytes_u8x16 = vorrq_u8(vqtbl2q_u8(table_high_u8x16x2, abs_indices_u8x16), sign_u8x16);

    // Low bytes via arithmetic: (exp != 0) ? (bit0 << 7) : 0
    // This uses shift/logic ports instead of permute port, and frees 4 registers
    uint8x16_t shifted_u8x16 = vshlq_n_u8(input_u8x16, 7);                 // bit 0 → bit 7
    uint8x16_t exp_bits_u8x16 = vandq_u8(input_u8x16, vdupq_n_u8(0x18));   // isolate exp (bits 4-3)
    uint8x16_t is_normal_u8x16 = vcgtq_u8(exp_bits_u8x16, vdupq_n_u8(0));  // 0xFF if exp≠0
    uint8x16_t low_bytes_u8x16 = vandq_u8(shifted_u8x16, is_normal_u8x16); // mask off subnormals

    // ZIP to interleave bytes into uint16 values: [l0,l1...l15] + [h0,h1...h15] → [l0,h0,l1,h1...]
    uint8x16x2_t interleaved_u8x16x2 = vzipq_u8(low_bytes_u8x16, high_bytes_u8x16);

    *result_low_f16x8 = vreinterpretq_f16_u8(interleaved_u8x16x2.val[0]);  // elements 0-7
    *result_high_f16x8 = vreinterpretq_f16_u8(interleaved_u8x16x2.val[1]); // elements 8-15
}

/** @brief Convert 16x e3m2 → 2x f16x8 via TBL lookup (NEON).
 *  E3M2FN (FP6): S EEE MM (bias=3) → F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Uses precomputed lookup tables for 64 possible 6-bit values.
 *
 *  Performance (per 16 elements):
 *    VLD1Q x4 (4 loads)         4 cy latency, 1 cy throughput each
 *    VAND (mask)                2 cy latency, 0.5 cy throughput
 *    VQTBL4 (table lookup)      3 cy latency, 1 cy throughput
 *    VZIP (interleave)          2 cy latency, 1 cy throughput
 *  Total: ~6-8 cy latency, ~2.5 cy amortized throughput (dominated by table lookup + zip) */
NK_INTERNAL void nk_e3m2x16_to_f16x8x2_neon_(uint8x16_t input_u8x16, float16x8_t *result_low_f16x8,
                                             float16x8_t *result_high_f16x8) {
    // Precomputed lookup table for E3M2FN → F16 conversion (high byte only, unsigned).
    // Sign symmetry: negative half = positive XOR 0x80, so we strip sign (bit 5),
    // lookup unsigned 5-bit value in VQTBL2 (32 bytes), then OR sign back.
    // E3M2FN: sign(1) exp(3) mant(2), bias=3
    // F16: sign(1) exp(5) mant(10), bias=15
    // Normal (exp!=0): f16 = (sign << 15) | ((exp + 12) << 10) | (mant << 8)
    // Subnormal (exp=0): f16 = mant/16 converted to f16
    // E3M2 mantissa (2 bits) maps to F16 bits [9:8]; F16 bits [7:0] = 0
    static nk_u8_t const table_high_u8x32[32] = {
        0x00, 0x2C, 0x30, 0x32, // exp=0 (subnormals): 0, 1/16, 2/16, 3/16
        0x34, 0x35, 0x36, 0x37, // exp=1 → f16_exp=13 (0x34-0x37)
        0x38, 0x39, 0x3A, 0x3B, // exp=2 → f16_exp=14 (0x38-0x3B)
        0x3C, 0x3D, 0x3E, 0x3F, // exp=3 → f16_exp=15 (0x3C-0x3F)
        0x40, 0x41, 0x42, 0x43, // exp=4 → f16_exp=16 (0x40-0x43)
        0x44, 0x45, 0x46, 0x47, // exp=5 → f16_exp=17 (0x44-0x47)
        0x48, 0x49, 0x4A, 0x4B, // exp=6 → f16_exp=18 (0x48-0x4B)
        0x4C, 0x4D, 0x4E, 0x4F, // exp=7 → f16_exp=19 (0x4C-0x4F)
    };

    // Load unsigned high byte table (2 registers instead of 4 - low bytes are always zero!)
    uint8x16x2_t table_high_u8x16x2 = vld1q_u8_x2(table_high_u8x32);

    // Extract sign (bit 5 → bit 7 for F16 high byte) and 5-bit unsigned index
    uint8x16_t sign_u8x16 = vshlq_n_u8(vandq_u8(input_u8x16, vdupq_n_u8(0x20)), 2);
    uint8x16_t abs_indices_u8x16 = vandq_u8(input_u8x16, vdupq_n_u8(0x1F));

    // Table lookup for high bytes on unsigned index, then OR sign back
    uint8x16_t high_bytes_u8x16 = vorrq_u8(vqtbl2q_u8(table_high_u8x16x2, abs_indices_u8x16), sign_u8x16);

    // ZIP zeros with high bytes: [0,0...0] + [h0,h1...h15] → [0,h0,0,h1...]
    uint8x16x2_t interleaved_u8x16x2 = vzipq_u8(vdupq_n_u8(0), high_bytes_u8x16);

    *result_low_f16x8 = vreinterpretq_f16_u8(interleaved_u8x16x2.val[0]);  // elements 0-7
    *result_high_f16x8 = vreinterpretq_f16_u8(interleaved_u8x16x2.val[1]); // elements 8-15
}

/** @brief Convert f16x8 → 8x e4m3 with RNE rounding (NEON).
 *  F16: S EEEEE MMMMMMMMMM (bias=15) → E4M3: S EEEE MMM (bias=7).
 *  Handles subnormals (exp < 9 → E4M3 subnormal), overflow (> 448 → clamp), inf → max, nan → nan. */
NK_INTERNAL uint8x8_t nk_f16x8_to_e4m3x8_neon_(float16x8_t f16x8) {
    uint16x8_t bits_u16x8 = vreinterpretq_u16_f16(f16x8);
    uint16x8_t sign_byte_u16x8 = vshrq_n_u16(vandq_u16(bits_u16x8, vdupq_n_u16(0x8000)), 8);
    uint16x8_t f16_exp_u16x8 = vandq_u16(vshrq_n_u16(bits_u16x8, 10), vdupq_n_u16(0x1F));
    uint16x8_t f16_mant_u16x8 = vandq_u16(bits_u16x8, vdupq_n_u16(0x03FF));

    // Rebias exponent: F16 bias=15 → E4M3 bias=7, subtract 8
    int16x8_t e4m3_exp_s16x8 = vsubq_s16(vreinterpretq_s16_u16(f16_exp_u16x8), vdupq_n_s16(8));

    // Detect special cases
    uint16x8_t is_f16_zero = vceqq_u16(vandq_u16(bits_u16x8, vdupq_n_u16(0x7FFF)), vdupq_n_u16(0));
    uint16x8_t is_f16_special = vceqq_u16(f16_exp_u16x8, vdupq_n_u16(31)); // inf or nan
    uint16x8_t is_f16_nan = vandq_u16(is_f16_special, vcgtq_u16(f16_mant_u16x8, vdupq_n_u16(0)));
    uint16x8_t is_underflow = vcltq_s16(e4m3_exp_s16x8, vdupq_n_s16(1)); // exp < 1 → subnormal/zero
    uint16x8_t is_overflow = vcgtq_s16(e4m3_exp_s16x8, vdupq_n_s16(15)); // exp > 15 → overflow

    // Normal path with RNE rounding: round mantissa from 10 to 3 bits
    // RNE: add (0x3F + lsb) where lsb = bit 7 of mantissa
    uint16x8_t lsb_u16x8 = vandq_u16(vshrq_n_u16(f16_mant_u16x8, 7), vdupq_n_u16(1));
    uint16x8_t rounded_mant_u16x8 = vaddq_u16(f16_mant_u16x8, vaddq_u16(vdupq_n_u16(0x3F), lsb_u16x8));
    uint16x8_t carry_u16x8 = vshrq_n_u16(rounded_mant_u16x8, 10); // Mantissa overflow → carry to exponent
    e4m3_exp_s16x8 = vaddq_s16(e4m3_exp_s16x8, vreinterpretq_s16_u16(carry_u16x8));
    uint16x8_t e4m3_mant_u16x8 = vandq_u16(vshrq_n_u16(rounded_mant_u16x8, 7), vdupq_n_u16(0x07));
    e4m3_mant_u16x8 = vbicq_u16(e4m3_mant_u16x8, vceqq_u16(carry_u16x8, vdupq_n_u16(1))); // Clear mant if carry

    // Recheck overflow after rounding (carry might have pushed us over)
    is_overflow = vorrq_u16(is_overflow, vcgtq_s16(e4m3_exp_s16x8, vdupq_n_s16(15)));

    // Clamp exponent to [1, 15] for normal values
    int16x8_t clamped_exp_s16x8 = vmaxq_s16(e4m3_exp_s16x8, vdupq_n_s16(1));
    clamped_exp_s16x8 = vminq_s16(clamped_exp_s16x8, vdupq_n_s16(15));

    // E4M3FN quirk: exp=15, mant=7 is NaN, so clamp mantissa to 6 when exp=15
    uint16x8_t is_max_exp = vceqq_s16(clamped_exp_s16x8, vdupq_n_s16(15));
    e4m3_mant_u16x8 = vbslq_u16(is_max_exp, vminq_u16(e4m3_mant_u16x8, vdupq_n_u16(6)), e4m3_mant_u16x8);

    // Assemble normal result
    uint16x8_t normal_result_u16x8 = vorrq_u16(
        sign_byte_u16x8, vorrq_u16(vshlq_n_u16(vreinterpretq_u16_s16(clamped_exp_s16x8), 3), e4m3_mant_u16x8));

    // Subnormal path: E4M3 subnormal = mant × 2⁻⁹
    // Use float conversion for correctness: abs(f16) × 512, round to int, clamp to [0,7]
    float32x4_t abs_low_f32x4 = vabsq_f32(vcvt_f32_f16(vget_low_f16(f16x8)));
    float32x4_t abs_high_f32x4 = vabsq_f32(vcvt_f32_f16(vget_high_f16(f16x8)));
    float32x4_t scaled_low_f32x4 = vmulq_n_f32(abs_low_f32x4, 512.0f);
    float32x4_t scaled_high_f32x4 = vmulq_n_f32(abs_high_f32x4, 512.0f);
    int32x4_t subnormal_mantissa_low_i32x4 = vcvtnq_s32_f32(scaled_low_f32x4); // Round to nearest even
    int32x4_t subnormal_mantissa_high_i32x4 = vcvtnq_s32_f32(scaled_high_f32x4);
    subnormal_mantissa_low_i32x4 = vmaxq_s32(vminq_s32(subnormal_mantissa_low_i32x4, vdupq_n_s32(7)), vdupq_n_s32(0));
    subnormal_mantissa_high_i32x4 = vmaxq_s32(vminq_s32(subnormal_mantissa_high_i32x4, vdupq_n_s32(7)), vdupq_n_s32(0));
    int16x4_t subnormal_mantissa_low_i16x4 = vmovn_s32(subnormal_mantissa_low_i32x4);
    int16x4_t subnormal_mantissa_high_i16x4 = vmovn_s32(subnormal_mantissa_high_i32x4);
    uint16x8_t subnormal_mant_u16x8 = vreinterpretq_u16_s16(
        vcombine_s16(subnormal_mantissa_low_i16x4, subnormal_mantissa_high_i16x4));
    uint16x8_t subnormal_result_u16x8 = vorrq_u16(sign_byte_u16x8, subnormal_mant_u16x8);

    // Special values: E4M3FN has no ∞, max normal = 0x7E (exp=15, mant=6 = 448)
    uint16x8_t e4m3_max = vorrq_u16(sign_byte_u16x8, vdupq_n_u16(0x7E)); // ±448 (exp=15, mant=6)
    uint16x8_t e4m3_nan = vorrq_u16(sign_byte_u16x8, vdupq_n_u16(0x7F)); // ±NaN (exp=15, mant=7)
    uint16x8_t e4m3_zero = sign_byte_u16x8;                              // ±0

    // Blend results (order matters: later conditions override earlier)
    uint16x8_t result_u16x8 = normal_result_u16x8;
    result_u16x8 = vbslq_u16(is_underflow, subnormal_result_u16x8, result_u16x8);
    result_u16x8 = vbslq_u16(is_overflow, e4m3_max, result_u16x8);
    result_u16x8 = vbslq_u16(is_f16_special, e4m3_max, result_u16x8); // F16 inf → E4M3 max (no inf in E4M3FN)
    result_u16x8 = vbslq_u16(is_f16_nan, e4m3_nan, result_u16x8);     // F16 nan → E4M3 nan
    result_u16x8 = vbslq_u16(is_f16_zero, e4m3_zero, result_u16x8);   // Preserve ±0

    return vmovn_u16(result_u16x8);
}

/** @brief Convert f16x8 → 8x e5m2 with RNE rounding (NEON).
 *  F16 (bias=15) and E5M2 (bias=15) share the same bias, so conversion is truncation with RNE rounding.
 *  F16: S EEEEE MMMMMMMMMM → E5M2: S EEEEE MM. Mantissa overflow carries into exponent. */
NK_INTERNAL uint8x8_t nk_f16x8_to_e5m2x8_neon_(float16x8_t f16x8) {
    uint16x8_t bits_u16x8 = vreinterpretq_u16_f16(f16x8);

    // Detect inf/nan (exp=31) - these should not be rounded, just truncated
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(bits_u16x8, 10), vdupq_n_u16(0x1F));
    uint16x8_t is_special_mask = vceqq_u16(exp_u16x8, vdupq_n_u16(31));

    // RNE rounding: add (0x7F + lsb) where lsb = bit 8 of F16
    // This rounds the lower 8 bits correctly and may carry into exponent
    uint16x8_t lsb_u16x8 = vandq_u16(vshrq_n_u16(bits_u16x8, 8), vdupq_n_u16(1));
    uint16x8_t rounding_bias_u16x8 = vaddq_u16(vdupq_n_u16(0x7F), lsb_u16x8);
    uint16x8_t rounded_bits_u16x8 = vaddq_u16(bits_u16x8, rounding_bias_u16x8);

    // For special values (inf/nan), use original bits without rounding
    uint16x8_t final_bits_u16x8 = vbslq_u16(is_special_mask, bits_u16x8, rounded_bits_u16x8);

    // Shift right by 8 to get E5M2 format
    uint16x8_t e5m2_u16x8 = vshrq_n_u16(final_bits_u16x8, 8);
    return vmovn_u16(e5m2_u16x8);
}

/** @brief Convert 4x bf16 → f32x4 via bit shift (NEON).
 *  BF16 format: S EEEEEEEE MMMMMMM (bias=127, same as f32 but truncated mantissa).
 *  F32 = bf16 << 16. */
NK_INTERNAL float32x4_t nk_bf16x4_to_f32x4_neon_(uint16x4_t bf16_u16x4) {
    uint32x4_t bits_u32x4 = vshlq_n_u32(vmovl_u16(bf16_u16x4), 16);
    return vreinterpretq_f32_u32(bits_u32x4);
}

/** @brief Convert 4x f16 (as u16 bits) → f32x4 via integer bit manipulation (NEON).
 *  F16 format: S EEEEE MMMMMMMMMM (bias=15, 5-bit exponent, 10-bit mantissa).
 *  Works on ARMv8.0 without the FP16 arithmetic extension. Treats denormals as zero. */
NK_INTERNAL float32x4_t nk_f16x4_to_f32x4_neon_(uint16x4_t half_u16x4) {
    // Widen u16 to u32
    uint32x4_t bits_u32x4 = vmovl_u16(half_u16x4);
    // Extract sign, exponent, mantissa
    uint32x4_t sign_u32x4 = vshlq_n_u32(vandq_u32(bits_u32x4, vdupq_n_u32(0x8000)), 16);
    uint32x4_t exponent_u32x4 = vandq_u32(bits_u32x4, vdupq_n_u32(0x7C00));
    uint32x4_t mantissa_u32x4 = vandq_u32(bits_u32x4, vdupq_n_u32(0x03FF));
    // Normal path: ((exponent + mantissa) << 13) + rebias(112 << 23 = 0x38000000)
    uint32x4_t exponent_mantissa_u32x4 = vandq_u32(bits_u32x4, vdupq_n_u32(0x7FFF));
    uint32x4_t normal_u32x4 = vaddq_u32(vshlq_n_u32(exponent_mantissa_u32x4, 13), vdupq_n_u32(0x38000000));
    // Inf/NaN path (exponent == 0x7C00): 0x7F800000 | (mantissa << 13)
    uint32x4_t inf_nan_u32x4 = vorrq_u32(vdupq_n_u32(0x7F800000), vshlq_n_u32(mantissa_u32x4, 13));
    // Select inf/NaN where exponent == 31 (0x7C00)
    uint32x4_t is_inf_nan_u32x4 = vceqq_u32(exponent_u32x4, vdupq_n_u32(0x7C00));
    uint32x4_t result_u32x4 = vbslq_u32(is_inf_nan_u32x4, inf_nan_u32x4, normal_u32x4);
    // Zero path (exponent == 0): treat denormals as zero for simplicity
    uint32x4_t is_zero_u32x4 = vceqq_u32(exponent_u32x4, vdupq_n_u32(0));
    result_u32x4 = vbslq_u32(is_zero_u32x4, vdupq_n_u32(0), result_u32x4);
    // OR sign back
    result_u32x4 = vorrq_u32(result_u32x4, sign_u32x4);
    return vreinterpretq_f32_u32(result_u32x4);
}

/** @brief Convert f32x4 → 4x bf16 with RNE rounding (NEON).
 *  Round-to-nearest-even: add (0x7FFF + lsb) before truncation. */
NK_INTERNAL uint16x4_t nk_f32x4_to_bf16x4_neon_(float32x4_t f32x4) {
    uint32x4_t bits_u32x4 = vreinterpretq_u32_f32(f32x4);
    uint32x4_t lsb_u32x4 = vandq_u32(vshrq_n_u32(bits_u32x4, 16), vdupq_n_u32(1));
    uint32x4_t rounding_u32x4 = vaddq_u32(vdupq_n_u32(0x7FFF), lsb_u32x4);
    bits_u32x4 = vaddq_u32(bits_u32x4, rounding_u32x4);
    return vmovn_u32(vshrq_n_u32(bits_u32x4, 16));
}

/** @brief Convert 8x e4m3 → bf16x8 via direct bit manipulation (NEON).
 *  E4M3FN format: S EEEE MMM (bias=7). BF16: S EEEEEEEE MMMMMMM (bias=127).
 *  Direct conversion without F16 ÷ F32 intermediate for hot loop efficiency. */
NK_INTERNAL uint16x8_t nk_e4m3x8_to_bf16x8_neon_(uint8x8_t e4m3_u8x8) {
    uint16x8_t e4m3_u16x8 = vmovl_u8(e4m3_u8x8);
    uint16x8_t sign_u16x8 = vshlq_n_u16(vandq_u16(e4m3_u16x8, vdupq_n_u16(0x80)), 8); // sign << 15
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(e4m3_u16x8, 3), vdupq_n_u16(0x0F));
    uint16x8_t mant_u16x8 = vandq_u16(e4m3_u16x8, vdupq_n_u16(0x07));

    // Normal path: BF16_exp = E4M3_exp + 120, BF16_mant = E4M3_mant << 4
    uint16x8_t bf16_exp_u16x8 = vshlq_n_u16(vaddq_u16(exp_u16x8, vdupq_n_u16(120)), 7);
    uint16x8_t bf16_mant_u16x8 = vshlq_n_u16(mant_u16x8, 4);
    uint16x8_t normal_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(bf16_exp_u16x8, bf16_mant_u16x8));

    // Subnormal path (exp=0): E4M3 subnormal = mant × 2⁻⁹ = mant ÷ 512 → BF16
    // Compute via f32: mant → f32 → multiply → truncate to bf16
    float32x4_t subnormal_low_f32x4 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(mant_u16x8))), 1.0f / 512.0f);
    float32x4_t subnormal_high_f32x4 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(mant_u16x8))), 1.0f / 512.0f);
    uint16x8_t subnormal_abs_u16x8 = vcombine_u16(nk_f32x4_to_bf16x4_neon_(subnormal_low_f32x4),
                                                  nk_f32x4_to_bf16x4_neon_(subnormal_high_f32x4));
    uint16x8_t subnormal_u16x8 = vorrq_u16(subnormal_abs_u16x8, sign_u16x8);

    // NaN path: E4M3FN only has NaN when exp=15 AND mant=7 (0x7F or 0xFF)
    uint16x8_t nan_u16x8 = vorrq_u16(sign_u16x8, vdupq_n_u16(0x7FC0)); // BF16 quiet NaN
    uint16x8_t is_nan_mask = vandq_u16(vceqq_u16(exp_u16x8, vdupq_n_u16(15)), vceqq_u16(mant_u16x8, vdupq_n_u16(7)));

    // Blend paths: subnormal when exp=0, NaN when exp=15 && mant=7, else normal
    uint16x8_t exp_zero_mask = vceqq_u16(exp_u16x8, vdupq_n_u16(0));
    uint16x8_t result_u16x8 = vbslq_u16(exp_zero_mask, subnormal_u16x8, normal_u16x8);
    result_u16x8 = vbslq_u16(is_nan_mask, nan_u16x8, result_u16x8);
    return result_u16x8;
}

/** @brief Convert 8x e5m2 → bf16x8 via direct bit manipulation (NEON).
 *  E5M2 format: S EEEEE MM (bias=15). BF16: S EEEEEEEE MMMMMMM (bias=127).
 *  Direct conversion without F16 ÷ F32 intermediate for hot loop efficiency. */
NK_INTERNAL uint16x8_t nk_e5m2x8_to_bf16x8_neon_(uint8x8_t e5m2_u8x8) {
    uint16x8_t e5m2_u16x8 = vmovl_u8(e5m2_u8x8);
    uint16x8_t sign_u16x8 = vshlq_n_u16(vandq_u16(e5m2_u16x8, vdupq_n_u16(0x80)), 8); // sign << 15
    uint16x8_t exp_u16x8 = vandq_u16(vshrq_n_u16(e5m2_u16x8, 2), vdupq_n_u16(0x1F));
    uint16x8_t mant_u16x8 = vandq_u16(e5m2_u16x8, vdupq_n_u16(0x03));

    // Normal path: BF16_exp = E5M2_exp + 112, BF16_mant = E5M2_mant << 5
    uint16x8_t bf16_exp_u16x8 = vshlq_n_u16(vaddq_u16(exp_u16x8, vdupq_n_u16(112)), 7);
    uint16x8_t bf16_mant_u16x8 = vshlq_n_u16(mant_u16x8, 5);
    uint16x8_t normal_u16x8 = vorrq_u16(sign_u16x8, vorrq_u16(bf16_exp_u16x8, bf16_mant_u16x8));

    // Subnormal path (exp=0): E5M2 subnormal = mant × 2⁻¹⁶ = mant ÷ 65536 → BF16
    // Compute via f32: mant → f32 → multiply → truncate to bf16
    float32x4_t subnormal_low_f32x4 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(mant_u16x8))), 1.0f / 65536.0f);
    float32x4_t subnormal_high_f32x4 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(mant_u16x8))),
                                                   1.0f / 65536.0f);
    uint16x8_t subnormal_abs_u16x8 = vcombine_u16(nk_f32x4_to_bf16x4_neon_(subnormal_low_f32x4),
                                                  nk_f32x4_to_bf16x4_neon_(subnormal_high_f32x4));
    uint16x8_t subnormal_u16x8 = vorrq_u16(subnormal_abs_u16x8, sign_u16x8);

    // Special path (exp=31): inf (mant=0) or nan (mant≠0)
    uint16x8_t infinity_u16x8 = vorrq_u16(sign_u16x8, vdupq_n_u16(0x7F80));
    uint16x8_t nan_u16x8 = vorrq_u16(sign_u16x8, vdupq_n_u16(0x7FC0));
    uint16x8_t mant_zero_mask = vceqq_u16(mant_u16x8, vdupq_n_u16(0));
    uint16x8_t special_u16x8 = vbslq_u16(mant_zero_mask, infinity_u16x8, nan_u16x8);

    // Blend paths based on exponent value
    uint16x8_t exp_zero_mask = vceqq_u16(exp_u16x8, vdupq_n_u16(0));
    uint16x8_t exp_max_mask = vceqq_u16(exp_u16x8, vdupq_n_u16(31));
    uint16x8_t result_u16x8 = vbslq_u16(exp_zero_mask, subnormal_u16x8, normal_u16x8);
    result_u16x8 = vbslq_u16(exp_max_mask, special_u16x8, result_u16x8);
    return result_u16x8;
}

/** @brief Convert 4x i16 → f32x4 (NEON). Widen to i32, then convert. */
NK_INTERNAL float32x4_t nk_i16x4_to_f32x4_neon_(int16x4_t i16x4) { return vcvtq_f32_s32(vmovl_s16(i16x4)); }

/** @brief Convert 4x u16 → f32x4 (NEON). Widen to u32, then convert. */
NK_INTERNAL float32x4_t nk_u16x4_to_f32x4_neon_(uint16x4_t u16x4) { return vcvtq_f32_u32(vmovl_u16(u16x4)); }

/** @brief Convert 4x i8 → f32x4 (NEON). Loads exactly 4 bytes via nk_b32_vec_t to avoid overread. */
NK_INTERNAL float32x4_t nk_i8x4_to_f32x4_neon_(nk_b32_vec_t in_vec) {
    int8x8_t i8x8 = vcreate_s8((nk_u64_t)in_vec.u32);
    int16x8_t i16x8 = vmovl_s8(i8x8);
    return vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16x8)));
}

/** @brief Convert 4x u8 → f32x4 (NEON). Loads exactly 4 bytes via nk_b32_vec_t to avoid overread. */
NK_INTERNAL float32x4_t nk_u8x4_to_f32x4_neon_(nk_b32_vec_t in_vec) {
    uint8x8_t u8x8 = vcreate_u8((nk_u64_t)in_vec.u32);
    uint16x8_t u16x8 = vmovl_u8(u8x8);
    return vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16x8)));
}

/** @brief Convert f32x4 → 4x i16 with saturation (NEON). Convert to i32, narrow. */
NK_INTERNAL int16x4_t nk_f32x4_to_i16x4_neon_(float32x4_t f32x4) {
    int32x4_t i32x4 = vcvtnq_s32_f32(f32x4);
    return vqmovn_s32(i32x4);
}

/** @brief Convert f32x4 → 4x u16 with saturation (NEON). Convert to u32, narrow. */
NK_INTERNAL uint16x4_t nk_f32x4_to_u16x4_neon_(float32x4_t f32x4) {
    uint32x4_t u32x4 = vcvtnq_u32_f32(f32x4);
    return vqmovn_u32(u32x4);
}

/** @brief Convert f32x4 → 4x i8 with saturation (NEON). Convert to i32, narrow twice. */
NK_INTERNAL void nk_f32x4_to_i8x4_neon_(float32x4_t f32x4, nk_i8_t *dst) {
    int32x4_t i32x4 = vcvtnq_s32_f32(f32x4);
    int16x4_t i16x4 = vqmovn_s32(i32x4);
    int8x8_t i8x8 = vqmovn_s16(vcombine_s16(i16x4, i16x4));
    // Reinterpret as s32x2, store lane 0 (4 bytes in one instruction)
    vst1_lane_s32((int32_t *)dst, vreinterpret_s32_s8(i8x8), 0);
}

/** @brief Convert f32x4 → 4x u8 with saturation (NEON). Convert to u32, narrow twice. */
NK_INTERNAL void nk_f32x4_to_u8x4_neon_(float32x4_t f32x4, nk_u8_t *dst) {
    uint32x4_t u32x4 = vcvtnq_u32_f32(f32x4);
    uint16x4_t u16x4 = vqmovn_u32(u32x4);
    uint8x8_t u8x8 = vqmovn_u16(vcombine_u16(u16x4, u16x4));
    // Reinterpret as u32x2, store lane 0 (4 bytes in one instruction)
    vst1_lane_u32((uint32_t *)dst, vreinterpret_u32_u8(u8x8), 0);
}

/** @brief Convert f32x4 → 4x e4m3 via bit manipulation (NEON).
 *  E4M3 format: S EEEE MMM (bias=7). Handles normal, subnormal, and overflow cases.
 *  Uses RNE (round to nearest even) for mantissa rounding. Returns packed result in nk_b32_vec_t. */
NK_INTERNAL nk_b32_vec_t nk_f32x4_to_e4m3x4_neon_(float32x4_t f32x4) {
    uint32x4_t bits_u32x4 = vreinterpretq_u32_f32(f32x4);
    uint32x4_t sign_u32x4 = vshrq_n_u32(bits_u32x4, 31);
    uint32x4_t f32_exp_u32x4 = vandq_u32(vshrq_n_u32(bits_u32x4, 23), vdupq_n_u32(0xFF));

    // Round mantissa from 23 to 3 bits using RNE (round to nearest, ties to even)
    // RNE trick: add (half - 1 + lsb) where lsb is the bit that will become the new lsb after shift
    uint32x4_t significand_u32x4 = vorrq_u32(vandq_u32(bits_u32x4, vdupq_n_u32(0x007FFFFF)),
                                             vdupq_n_u32(0x00800000)); // Add implicit 1 bit
    uint32x4_t lsb_u32x4 = vandq_u32(vshrq_n_u32(significand_u32x4, 20), vdupq_n_u32(1));
    uint32x4_t rounding_bias_u32x4 = vaddq_u32(vdupq_n_u32(0x0007FFFF), lsb_u32x4);
    uint32x4_t rounded_sig_u32x4 = vaddq_u32(significand_u32x4, rounding_bias_u32x4);
    uint32x4_t carry_u32x4 = vshrq_n_u32(rounded_sig_u32x4, 24); // Carry into exponent if bit 24 set
    uint32x4_t f32_mantissa_u32x4 = vandq_u32(vshrq_n_u32(rounded_sig_u32x4, 20), vdupq_n_u32(0x07));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    uint32x4_t carry_mask_u32x4 = vceqq_u32(carry_u32x4, vdupq_n_u32(1));
    f32_mantissa_u32x4 = vbicq_u32(f32_mantissa_u32x4, carry_mask_u32x4);

    // Rebias exponent: f32 bias 127 → e4m3 bias 7 (subtract 120)
    int32x4_t e4m3_exp_i32x4 = vsubq_s32(
        vaddq_s32(vreinterpretq_s32_u32(f32_exp_u32x4), vreinterpretq_s32_u32(carry_u32x4)), vdupq_n_s32(120));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 15)
    uint32x4_t is_subnormal_u32x4 = vcltq_s32(e4m3_exp_i32x4, vdupq_n_s32(1));
    uint32x4_t overflow_u32x4 = vcgtq_s32(e4m3_exp_i32x4, vdupq_n_s32(15));

    // Normal path: clamp exp to [1,15], extract mantissa bits
    // e4m3FN quirk: exp=15 with mantissa=7 is NaN (0x7F), so clamp mantissa to 6 when exp=15.
    int32x4_t clamped_exp_i32x4 = vmaxq_s32(e4m3_exp_i32x4, vdupq_n_s32(1));
    clamped_exp_i32x4 = vminq_s32(clamped_exp_i32x4, vdupq_n_s32(15));
    uint32x4_t is_max_exp_u32x4 = vceqq_s32(clamped_exp_i32x4, vdupq_n_s32(15));
    uint32x4_t max_mantissa_u32x4 = vbslq_u32(is_max_exp_u32x4, vdupq_n_u32(6), vdupq_n_u32(7));
    uint32x4_t normal_mantissa_u32x4 = vminq_u32(f32_mantissa_u32x4, max_mantissa_u32x4);
    normal_mantissa_u32x4 = vbslq_u32(overflow_u32x4, vdupq_n_u32(0x06), normal_mantissa_u32x4);
    uint32x4_t normal_e4m3_u32x4 = vorrq_u32(
        vshlq_n_u32(sign_u32x4, 7),
        vorrq_u32(vshlq_n_u32(vreinterpretq_u32_s32(clamped_exp_i32x4), 3), normal_mantissa_u32x4));

    // Subnormal path: mantissa = round(abs_f32 * 512)
    // If mantissa rounds to 8 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x08
    float32x4_t abs_f32x4 = vabsq_f32(f32x4);
    float32x4_t scaled_f32x4 = vmulq_n_f32(abs_f32x4, 512.0f);
    int32x4_t subnormal_mantissa_i32x4 = vcvtnq_s32_f32(scaled_f32x4);
    uint32x4_t promotes_to_normal_u32x4 = vcgtq_s32(subnormal_mantissa_i32x4, vdupq_n_s32(7));
    subnormal_mantissa_i32x4 = vminq_s32(subnormal_mantissa_i32x4, vdupq_n_s32(7));
    subnormal_mantissa_i32x4 = vmaxq_s32(subnormal_mantissa_i32x4, vdupq_n_s32(0));
    uint32x4_t subnormal_e4m3_u32x4 = vorrq_u32(vshlq_n_u32(sign_u32x4, 7),
                                                vreinterpretq_u32_s32(subnormal_mantissa_i32x4));
    // When mantissa rounds to 8, use first normal value (0x08) instead of clamped subnormal
    uint32x4_t first_normal_e4m3_u32x4 = vorrq_u32(vshlq_n_u32(sign_u32x4, 7), vdupq_n_u32(0x08));
    subnormal_e4m3_u32x4 = vbslq_u32(promotes_to_normal_u32x4, first_normal_e4m3_u32x4, subnormal_e4m3_u32x4);

    // Blend: use subnormal result when exp <= 0, else normal
    uint32x4_t e4m3_u32x4 = vbslq_u32(is_subnormal_u32x4, subnormal_e4m3_u32x4, normal_e4m3_u32x4);

    // Pack 4 u32s to 4 u8s
    uint16x4_t e4m3_u16x4 = vmovn_u32(e4m3_u32x4);
    uint8x8_t e4m3_u8x8 = vmovn_u16(vcombine_u16(e4m3_u16x4, e4m3_u16x4));
    nk_b32_vec_t result;
    result.u32 = vget_lane_u32(vreinterpret_u32_u8(e4m3_u8x8), 0);
    return result;
}

/** @brief Convert f32x4 → 4x e5m2 via bit manipulation (NEON).
 *  E5M2 format: S EEEEE MM (bias=15). Handles normal, subnormal, and overflow cases.
 *  Uses RNE (round to nearest even) for mantissa rounding. Returns packed result in nk_b32_vec_t. */
NK_INTERNAL nk_b32_vec_t nk_f32x4_to_e5m2x4_neon_(float32x4_t f32x4) {
    uint32x4_t bits_u32x4 = vreinterpretq_u32_f32(f32x4);
    uint32x4_t sign_u32x4 = vshrq_n_u32(bits_u32x4, 31);
    uint32x4_t f32_exp_u32x4 = vandq_u32(vshrq_n_u32(bits_u32x4, 23), vdupq_n_u32(0xFF));

    // Round mantissa from 23 to 2 bits using RNE (round to nearest, ties to even)
    // RNE trick: add (half - 1 + lsb) where lsb is the bit that will become the new lsb after shift
    uint32x4_t significand_u32x4 = vorrq_u32(vandq_u32(bits_u32x4, vdupq_n_u32(0x007FFFFF)),
                                             vdupq_n_u32(0x00800000)); // Add implicit 1 bit
    uint32x4_t lsb_u32x4 = vandq_u32(vshrq_n_u32(significand_u32x4, 21), vdupq_n_u32(1));
    uint32x4_t rounding_bias_u32x4 = vaddq_u32(vdupq_n_u32(0x000FFFFF), lsb_u32x4); // half = 0x100000
    uint32x4_t rounded_sig_u32x4 = vaddq_u32(significand_u32x4, rounding_bias_u32x4);
    uint32x4_t carry_u32x4 = vshrq_n_u32(rounded_sig_u32x4, 24); // Carry into exponent if bit 24 set
    uint32x4_t f32_mantissa_u32x4 = vandq_u32(vshrq_n_u32(rounded_sig_u32x4, 21), vdupq_n_u32(0x03));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    uint32x4_t carry_mask_u32x4 = vceqq_u32(carry_u32x4, vdupq_n_u32(1));
    f32_mantissa_u32x4 = vbicq_u32(f32_mantissa_u32x4, carry_mask_u32x4);

    // Rebias exponent: f32 bias 127 → e5m2 bias 15 (subtract 112)
    int32x4_t e5m2_exp_i32x4 = vsubq_s32(
        vaddq_s32(vreinterpretq_s32_u32(f32_exp_u32x4), vreinterpretq_s32_u32(carry_u32x4)), vdupq_n_s32(112));

    // Detect subnormal (exp <= 0) and overflow (exp > 31)
    uint32x4_t is_subnormal_u32x4 = vcltq_s32(e5m2_exp_i32x4, vdupq_n_s32(1));
    uint32x4_t overflow_u32x4 = vcgtq_s32(e5m2_exp_i32x4, vdupq_n_s32(31));

    // Normal path: clamp exp to [1,31], on overflow return infinity (exp=31, mantissa=0 = 0x7C)
    int32x4_t clamped_exp_i32x4 = vmaxq_s32(e5m2_exp_i32x4, vdupq_n_s32(1));
    clamped_exp_i32x4 = vminq_s32(clamped_exp_i32x4, vdupq_n_s32(31));
    uint32x4_t normal_mantissa_u32x4 = vbslq_u32(overflow_u32x4, vdupq_n_u32(0), f32_mantissa_u32x4);
    uint32x4_t normal_e5m2_u32x4 = vorrq_u32(
        vshlq_n_u32(sign_u32x4, 7),
        vorrq_u32(vshlq_n_u32(vreinterpretq_u32_s32(clamped_exp_i32x4), 2), normal_mantissa_u32x4));

    // Subnormal path: mantissa = round(abs_f32 * 65536)
    // If mantissa rounds to 4 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x04
    float32x4_t abs_f32x4 = vabsq_f32(f32x4);
    float32x4_t scaled_f32x4 = vmulq_n_f32(abs_f32x4, 65536.0f);
    int32x4_t subnormal_mantissa_i32x4 = vcvtnq_s32_f32(scaled_f32x4);
    uint32x4_t promotes_to_normal_u32x4 = vcgtq_s32(subnormal_mantissa_i32x4, vdupq_n_s32(3));
    subnormal_mantissa_i32x4 = vminq_s32(subnormal_mantissa_i32x4, vdupq_n_s32(3));
    subnormal_mantissa_i32x4 = vmaxq_s32(subnormal_mantissa_i32x4, vdupq_n_s32(0));
    uint32x4_t subnormal_e5m2_u32x4 = vorrq_u32(vshlq_n_u32(sign_u32x4, 7),
                                                vreinterpretq_u32_s32(subnormal_mantissa_i32x4));
    // When mantissa rounds to 4, use first normal value (0x04) instead of clamped subnormal
    uint32x4_t first_normal_e5m2_u32x4 = vorrq_u32(vshlq_n_u32(sign_u32x4, 7), vdupq_n_u32(0x04));
    subnormal_e5m2_u32x4 = vbslq_u32(promotes_to_normal_u32x4, first_normal_e5m2_u32x4, subnormal_e5m2_u32x4);

    // Blend: use subnormal result when exp <= 0
    uint32x4_t e5m2_u32x4 = vbslq_u32(is_subnormal_u32x4, subnormal_e5m2_u32x4, normal_e5m2_u32x4);

    // Pack 4 u32s to 4 u8s
    uint16x4_t e5m2_u16x4 = vmovn_u32(e5m2_u32x4);
    uint8x8_t e5m2_u8x8 = vmovn_u16(vcombine_u16(e5m2_u16x4, e5m2_u16x4));
    nk_b32_vec_t result;
    result.u32 = vget_lane_u32(vreinterpret_u32_u8(e5m2_u8x8), 0);
    return result;
}

/** @brief Convert 4x e2m3 → f32x4 via bit manipulation (NEON).
 *  E2M3 format: S EE MMM (bias=1). F32: sign<<31, (exp+126)<<23, mantissa<<20.
 *  Handles subnormals (exp=0, mant ≠ 0). */
NK_INTERNAL float32x4_t nk_e2m3x4_to_f32x4_neon_(nk_b32_vec_t src) {
    uint8x8_t e2m3_u8x8 = vcreate_u8(src.u32);
    uint16x8_t e2m3_u16x8 = vmovl_u8(e2m3_u8x8);
    uint32x4_t e2m3_u32x4 = vmovl_u16(vget_low_u16(e2m3_u16x8));
    uint32x4_t sign_u32x4 = vshlq_n_u32(vandq_u32(e2m3_u32x4, vdupq_n_u32(0x20)), 26);
    uint32x4_t exp_u32x4 = vandq_u32(vshrq_n_u32(e2m3_u32x4, 3), vdupq_n_u32(0x03));
    uint32x4_t mant_u32x4 = vandq_u32(e2m3_u32x4, vdupq_n_u32(0x07));

    // Normal path: f32 = sign | ((exp+126)<<23) | (mant<<20)
    uint32x4_t f32_exp_u32x4 = vshlq_n_u32(vaddq_u32(exp_u32x4, vdupq_n_u32(126)), 23);
    uint32x4_t f32_mant_u32x4 = vshlq_n_u32(mant_u32x4, 20);
    uint32x4_t normal_u32x4 = vorrq_u32(sign_u32x4, vorrq_u32(f32_exp_u32x4, f32_mant_u32x4));

    // Subnormal path (exp=0, mant ≠ 0): value = ±mantissa × 2⁻³
    float32x4_t subnormal_f32x4 = vmulq_n_f32(vcvtq_f32_u32(mant_u32x4), 1.0f / 8.0f);
    uint32x4_t subnormal_u32x4 = vorrq_u32(vreinterpretq_u32_f32(subnormal_f32x4), sign_u32x4);

    // Blend paths: subnormal when exp=0, else normal
    uint32x4_t exp_zero_mask = vceqq_u32(exp_u32x4, vdupq_n_u32(0));
    uint32x4_t result_u32x4 = vbslq_u32(exp_zero_mask, subnormal_u32x4, normal_u32x4);
    return vreinterpretq_f32_u32(result_u32x4);
}

/** @brief Convert 4x e3m2 → f32x4 via bit manipulation (NEON).
 *  E3M2 format: S EEE MM (bias=3). F32: sign<<31, (exp+124)<<23, mantissa<<21.
 *  Handles subnormals (exp=0, mant ≠ 0). */
NK_INTERNAL float32x4_t nk_e3m2x4_to_f32x4_neon_(nk_b32_vec_t src) {
    uint8x8_t e3m2_u8x8 = vcreate_u8(src.u32);
    uint16x8_t e3m2_u16x8 = vmovl_u8(e3m2_u8x8);
    uint32x4_t e3m2_u32x4 = vmovl_u16(vget_low_u16(e3m2_u16x8));
    uint32x4_t sign_u32x4 = vshlq_n_u32(vandq_u32(e3m2_u32x4, vdupq_n_u32(0x20)), 26);
    uint32x4_t exp_u32x4 = vandq_u32(vshrq_n_u32(e3m2_u32x4, 2), vdupq_n_u32(0x07));
    uint32x4_t mant_u32x4 = vandq_u32(e3m2_u32x4, vdupq_n_u32(0x03));

    // Normal path: f32 = sign | ((exp+124)<<23) | (mant<<21)
    uint32x4_t f32_exp_u32x4 = vshlq_n_u32(vaddq_u32(exp_u32x4, vdupq_n_u32(124)), 23);
    uint32x4_t f32_mant_u32x4 = vshlq_n_u32(mant_u32x4, 21);
    uint32x4_t normal_u32x4 = vorrq_u32(sign_u32x4, vorrq_u32(f32_exp_u32x4, f32_mant_u32x4));

    // Subnormal path (exp=0, mant ≠ 0): value = ±mantissa × 2⁻⁴
    float32x4_t subnormal_f32x4 = vmulq_n_f32(vcvtq_f32_u32(mant_u32x4), 1.0f / 16.0f);
    uint32x4_t subnormal_u32x4 = vorrq_u32(vreinterpretq_u32_f32(subnormal_f32x4), sign_u32x4);

    // Blend paths: subnormal when exp=0, else normal
    uint32x4_t exp_zero_mask = vceqq_u32(exp_u32x4, vdupq_n_u32(0));
    uint32x4_t result_u32x4 = vbslq_u32(exp_zero_mask, subnormal_u32x4, normal_u32x4);
    return vreinterpretq_f32_u32(result_u32x4);
}

/** @brief Convert f32x4 → 4x e2m3 via bit manipulation (NEON).
 *  E2M3 format: S EE MMM (bias=1). Handles normal, subnormal, and overflow cases.
 *  Uses RNE (round to nearest even) for mantissa rounding. Returns packed result in nk_b32_vec_t. */
NK_INTERNAL nk_b32_vec_t nk_f32x4_to_e2m3x4_neon_(float32x4_t f32x4) {
    uint32x4_t bits_u32x4 = vreinterpretq_u32_f32(f32x4);
    uint32x4_t sign_u32x4 = vshrq_n_u32(bits_u32x4, 31);
    uint32x4_t f32_exp_u32x4 = vandq_u32(vshrq_n_u32(bits_u32x4, 23), vdupq_n_u32(0xFF));

    // Round mantissa from 23 to 3 bits using RNE (round to nearest, ties to even)
    uint32x4_t significand_u32x4 = vorrq_u32(vandq_u32(bits_u32x4, vdupq_n_u32(0x007FFFFF)),
                                             vdupq_n_u32(0x00800000)); // Add implicit 1 bit
    uint32x4_t lsb_u32x4 = vandq_u32(vshrq_n_u32(significand_u32x4, 20), vdupq_n_u32(1));
    uint32x4_t rounding_bias_u32x4 = vaddq_u32(vdupq_n_u32(0x0007FFFF), lsb_u32x4);
    uint32x4_t rounded_sig_u32x4 = vaddq_u32(significand_u32x4, rounding_bias_u32x4);
    uint32x4_t carry_u32x4 = vshrq_n_u32(rounded_sig_u32x4, 24); // Carry into exponent if bit 24 set
    uint32x4_t f32_mantissa_u32x4 = vandq_u32(vshrq_n_u32(rounded_sig_u32x4, 20), vdupq_n_u32(0x07));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    uint32x4_t carry_mask_u32x4 = vceqq_u32(carry_u32x4, vdupq_n_u32(1));
    f32_mantissa_u32x4 = vbicq_u32(f32_mantissa_u32x4, carry_mask_u32x4);

    // Rebias exponent: f32 bias 127 → e2m3 bias 1 (subtract 126)
    int32x4_t e2m3_exp_i32x4 = vsubq_s32(
        vaddq_s32(vreinterpretq_s32_u32(f32_exp_u32x4), vreinterpretq_s32_u32(carry_u32x4)), vdupq_n_s32(126));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 3)
    uint32x4_t is_subnormal_u32x4 = vcltq_s32(e2m3_exp_i32x4, vdupq_n_s32(1));
    uint32x4_t overflow_u32x4 = vcgtq_s32(e2m3_exp_i32x4, vdupq_n_s32(3));

    // Normal path: clamp exp to [1,3], extract mantissa bits
    int32x4_t clamped_exp_i32x4 = vmaxq_s32(e2m3_exp_i32x4, vdupq_n_s32(1));
    clamped_exp_i32x4 = vminq_s32(clamped_exp_i32x4, vdupq_n_s32(3));
    uint32x4_t normal_mantissa_u32x4 = vbslq_u32(overflow_u32x4, vdupq_n_u32(0x07), f32_mantissa_u32x4);
    uint32x4_t normal_e2m3_u32x4 = vorrq_u32(
        vshlq_n_u32(sign_u32x4, 5),
        vorrq_u32(vshlq_n_u32(vreinterpretq_u32_s32(clamped_exp_i32x4), 3), normal_mantissa_u32x4));

    // Subnormal path: mantissa = round(abs_f32 * 8)
    // If mantissa rounds to 8 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x08
    float32x4_t abs_f32x4 = vabsq_f32(f32x4);
    float32x4_t scaled_f32x4 = vmulq_n_f32(abs_f32x4, 8.0f);
    int32x4_t subnormal_mantissa_i32x4 = vcvtnq_s32_f32(scaled_f32x4);
    uint32x4_t promotes_to_normal_u32x4 = vcgtq_s32(subnormal_mantissa_i32x4, vdupq_n_s32(7));
    subnormal_mantissa_i32x4 = vminq_s32(subnormal_mantissa_i32x4, vdupq_n_s32(7));
    subnormal_mantissa_i32x4 = vmaxq_s32(subnormal_mantissa_i32x4, vdupq_n_s32(0));
    uint32x4_t subnormal_e2m3_u32x4 = vorrq_u32(vshlq_n_u32(sign_u32x4, 5),
                                                vreinterpretq_u32_s32(subnormal_mantissa_i32x4));
    // When mantissa rounds to 8, use first normal value (0x08) instead of clamped subnormal
    uint32x4_t first_normal_e2m3_u32x4 = vorrq_u32(vshlq_n_u32(sign_u32x4, 5), vdupq_n_u32(0x08));
    subnormal_e2m3_u32x4 = vbslq_u32(promotes_to_normal_u32x4, first_normal_e2m3_u32x4, subnormal_e2m3_u32x4);

    // Blend: use subnormal result when exp <= 0, else normal
    uint32x4_t e2m3_u32x4 = vbslq_u32(is_subnormal_u32x4, subnormal_e2m3_u32x4, normal_e2m3_u32x4);

    // Pack 4 u32s to 4 u8s
    uint16x4_t e2m3_u16x4 = vmovn_u32(e2m3_u32x4);
    uint8x8_t e2m3_u8x8 = vmovn_u16(vcombine_u16(e2m3_u16x4, e2m3_u16x4));
    nk_b32_vec_t result;
    result.u32 = vget_lane_u32(vreinterpret_u32_u8(e2m3_u8x8), 0);
    return result;
}

/** @brief Convert f32x4 → 4x e3m2 via bit manipulation (NEON).
 *  E3M2 format: S EEE MM (bias=3). Handles normal, subnormal, and overflow cases.
 *  Uses RNE (round to nearest even) for mantissa rounding. Returns packed result in nk_b32_vec_t. */
NK_INTERNAL nk_b32_vec_t nk_f32x4_to_e3m2x4_neon_(float32x4_t f32x4) {
    uint32x4_t bits_u32x4 = vreinterpretq_u32_f32(f32x4);
    uint32x4_t sign_u32x4 = vshrq_n_u32(bits_u32x4, 31);
    uint32x4_t f32_exp_u32x4 = vandq_u32(vshrq_n_u32(bits_u32x4, 23), vdupq_n_u32(0xFF));

    // Round mantissa from 23 to 2 bits using RNE (round to nearest, ties to even)
    uint32x4_t significand_u32x4 = vorrq_u32(vandq_u32(bits_u32x4, vdupq_n_u32(0x007FFFFF)),
                                             vdupq_n_u32(0x00800000)); // Add implicit 1 bit
    uint32x4_t lsb_u32x4 = vandq_u32(vshrq_n_u32(significand_u32x4, 21), vdupq_n_u32(1));
    uint32x4_t rounding_bias_u32x4 = vaddq_u32(vdupq_n_u32(0x000FFFFF), lsb_u32x4);
    uint32x4_t rounded_sig_u32x4 = vaddq_u32(significand_u32x4, rounding_bias_u32x4);
    uint32x4_t carry_u32x4 = vshrq_n_u32(rounded_sig_u32x4, 24); // Carry into exponent if bit 24 set
    uint32x4_t f32_mantissa_u32x4 = vandq_u32(vshrq_n_u32(rounded_sig_u32x4, 21), vdupq_n_u32(0x03));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    uint32x4_t carry_mask_u32x4 = vceqq_u32(carry_u32x4, vdupq_n_u32(1));
    f32_mantissa_u32x4 = vbicq_u32(f32_mantissa_u32x4, carry_mask_u32x4);

    // Rebias exponent: f32 bias 127 → e3m2 bias 3 (subtract 124)
    int32x4_t e3m2_exp_i32x4 = vsubq_s32(
        vaddq_s32(vreinterpretq_s32_u32(f32_exp_u32x4), vreinterpretq_s32_u32(carry_u32x4)), vdupq_n_s32(124));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 7)
    uint32x4_t is_subnormal_u32x4 = vcltq_s32(e3m2_exp_i32x4, vdupq_n_s32(1));
    uint32x4_t overflow_u32x4 = vcgtq_s32(e3m2_exp_i32x4, vdupq_n_s32(7));

    // Normal path: clamp exp to [1,7], extract mantissa bits
    int32x4_t clamped_exp_i32x4 = vmaxq_s32(e3m2_exp_i32x4, vdupq_n_s32(1));
    clamped_exp_i32x4 = vminq_s32(clamped_exp_i32x4, vdupq_n_s32(7));
    uint32x4_t normal_mantissa_u32x4 = vbslq_u32(overflow_u32x4, vdupq_n_u32(0x03), f32_mantissa_u32x4);
    uint32x4_t normal_e3m2_u32x4 = vorrq_u32(
        vshlq_n_u32(sign_u32x4, 5),
        vorrq_u32(vshlq_n_u32(vreinterpretq_u32_s32(clamped_exp_i32x4), 2), normal_mantissa_u32x4));

    // Subnormal path: mantissa = round(abs_f32 * 16)
    // If mantissa rounds to 4 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x04
    float32x4_t abs_f32x4 = vabsq_f32(f32x4);
    float32x4_t scaled_f32x4 = vmulq_n_f32(abs_f32x4, 16.0f);
    int32x4_t subnormal_mantissa_i32x4 = vcvtnq_s32_f32(scaled_f32x4);
    uint32x4_t promotes_to_normal_u32x4 = vcgtq_s32(subnormal_mantissa_i32x4, vdupq_n_s32(3));
    subnormal_mantissa_i32x4 = vminq_s32(subnormal_mantissa_i32x4, vdupq_n_s32(3));
    subnormal_mantissa_i32x4 = vmaxq_s32(subnormal_mantissa_i32x4, vdupq_n_s32(0));
    uint32x4_t subnormal_e3m2_u32x4 = vorrq_u32(vshlq_n_u32(sign_u32x4, 5),
                                                vreinterpretq_u32_s32(subnormal_mantissa_i32x4));
    // When mantissa rounds to 4, use first normal value (0x04) instead of clamped subnormal
    uint32x4_t first_normal_e3m2_u32x4 = vorrq_u32(vshlq_n_u32(sign_u32x4, 5), vdupq_n_u32(0x04));
    subnormal_e3m2_u32x4 = vbslq_u32(promotes_to_normal_u32x4, first_normal_e3m2_u32x4, subnormal_e3m2_u32x4);

    // Blend: use subnormal result when exp <= 0
    uint32x4_t e3m2_u32x4 = vbslq_u32(is_subnormal_u32x4, subnormal_e3m2_u32x4, normal_e3m2_u32x4);

    // Pack 4 u32s to 4 u8s
    uint16x4_t e3m2_u16x4 = vmovn_u32(e3m2_u32x4);
    uint8x8_t e3m2_u8x8 = vmovn_u16(vcombine_u16(e3m2_u16x4, e3m2_u16x4));
    nk_b32_vec_t result;
    result.u32 = vget_lane_u32(vreinterpret_u32_u8(e3m2_u8x8), 0);
    return result;
}

#pragma endregion - Vectorized Conversions

#pragma region - Public API

NK_PUBLIC void nk_cast_neon(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
    // Same-type fast path
    if (from_type == to_type) {
        nk_size_t size_bits = nk_dtype_bits(from_type);
        if (size_bits > 0) nk_copy_bytes_(to, from, nk_size_divide_round_up_(n * size_bits, 8));
        return;
    }

    // Validate supported types (f32 and smaller)
    int from_ok = (from_type == nk_f32_k || from_type == nk_f16_k || from_type == nk_bf16_k || from_type == nk_e4m3_k ||
                   from_type == nk_e5m2_k || from_type == nk_e2m3_k || from_type == nk_e3m2_k || from_type == nk_i8_k ||
                   from_type == nk_u8_k || from_type == nk_i16_k || from_type == nk_u16_k || from_type == nk_i32_k ||
                   from_type == nk_u32_k);
    int to_ok = (to_type == nk_f32_k || to_type == nk_f16_k || to_type == nk_bf16_k || to_type == nk_e4m3_k ||
                 to_type == nk_e5m2_k || to_type == nk_e2m3_k || to_type == nk_e3m2_k || to_type == nk_i8_k ||
                 to_type == nk_u8_k || to_type == nk_i16_k || to_type == nk_u16_k || to_type == nk_i32_k ||
                 to_type == nk_u32_k);

    // Fall back to serial for unsupported or i32<->u32 (loses precision through f32)
    if (!from_ok || !to_ok || (from_type == nk_i32_k && to_type == nk_u32_k) ||
        (from_type == nk_u32_k && to_type == nk_i32_k)) {
        nk_cast_serial(from, from_type, n, to, to_type);
        return;
    }

    // Check if F16 hub is applicable (FP8/F16/BF16 conversions, 8 elements/iter)
    // Exception: BF16 ↔ F16 skips F16 hub since it needs F32 intermediate anyway
    int from_f16_hub = (from_type == nk_e4m3_k || from_type == nk_e5m2_k || from_type == nk_e2m3_k ||
                        from_type == nk_e3m2_k || from_type == nk_f16_k || from_type == nk_bf16_k);
    int to_f16_hub = (to_type == nk_e4m3_k || to_type == nk_e5m2_k || to_type == nk_f16_k || to_type == nk_bf16_k ||
                      to_type == nk_f32_k);
    int is_bf16_f16 = (from_type == nk_bf16_k && to_type == nk_f16_k) ||
                      (from_type == nk_f16_k && to_type == nk_bf16_k);

    if (from_f16_hub && to_f16_hub && !is_bf16_f16) {
        // F16 hub: 8 elements per iteration (float16x8_t intermediate)
        nk_size_t batches = n / 8;
        nk_size_t from_step = 8 * nk_dtype_bits(from_type) / 8;
        nk_size_t to_step = 8 * nk_dtype_bits(to_type) / 8;
        nk_u8_t const *from_ptr = (nk_u8_t const *)from;
        nk_u8_t *to_ptr = (nk_u8_t *)to;

        for (nk_size_t idx = 0; idx < batches; ++idx, from_ptr += from_step, to_ptr += to_step) {
            // Upcast to f16x8 hub
            float16x8_t hub_f16x8;
            switch (from_type) {
            case nk_e4m3_k: hub_f16x8 = nk_e4m3x8_to_f16x8_neon_(vld1_u8(from_ptr)); break;
            case nk_e5m2_k: hub_f16x8 = nk_e5m2x8_to_f16x8_neon_(vld1_u8(from_ptr)); break;
            case nk_e2m3_k: hub_f16x8 = nk_e2m3x8_to_f16x8_neon_(vld1_u8(from_ptr)); break;
            case nk_e3m2_k: hub_f16x8 = nk_e3m2x8_to_f16x8_neon_(vld1_u8(from_ptr)); break;
            case nk_f16_k: hub_f16x8 = vreinterpretq_f16_u16(vld1q_u16((nk_u16_t const *)from_ptr)); break;
            case nk_bf16_k: {
                uint16x4_t brain_low_u16x4 = vld1_u16((nk_u16_t const *)from_ptr);
                uint16x4_t brain_high_u16x4 = vld1_u16((nk_u16_t const *)(from_ptr + 8));
                float32x4_t ieee_low_f32x4 = nk_bf16x4_to_f32x4_neon_(brain_low_u16x4);
                float32x4_t ieee_high_f32x4 = nk_bf16x4_to_f32x4_neon_(brain_high_u16x4);
                hub_f16x8 = vcombine_f16(vcvt_f16_f32(ieee_low_f32x4), vcvt_f16_f32(ieee_high_f32x4));
            } break;
            default: hub_f16x8 = vreinterpretq_f16_u16(vdupq_n_u16(0)); break;
            }

            // Downcast from f16x8 hub
            switch (to_type) {
            case nk_e4m3_k: vst1_u8(to_ptr, nk_f16x8_to_e4m3x8_neon_(hub_f16x8)); break;
            case nk_e5m2_k: vst1_u8(to_ptr, nk_f16x8_to_e5m2x8_neon_(hub_f16x8)); break;
            case nk_f16_k: vst1q_u16((nk_u16_t *)to_ptr, vreinterpretq_u16_f16(hub_f16x8)); break;
            case nk_bf16_k: {
                float32x4_t ieee_low_f32x4 = vcvt_f32_f16(vget_low_f16(hub_f16x8));
                float32x4_t ieee_high_f32x4 = vcvt_f32_f16(vget_high_f16(hub_f16x8));
                vst1_u16((nk_u16_t *)to_ptr, nk_f32x4_to_bf16x4_neon_(ieee_low_f32x4));
                vst1_u16((nk_u16_t *)(to_ptr + 8), nk_f32x4_to_bf16x4_neon_(ieee_high_f32x4));
            } break;
            case nk_f32_k: {
                vst1q_f32((nk_f32_t *)to_ptr, vcvt_f32_f16(vget_low_f16(hub_f16x8)));
                vst1q_f32((nk_f32_t *)(to_ptr + 16), vcvt_f32_f16(vget_high_f16(hub_f16x8)));
            } break;
            default: break;
            }
        }

        // Handle remaining elements (0-7) with F32 hub or serial
        n = n % 8;
        from = from_ptr;
        to = to_ptr;
        if (n == 0) return;
    }

    // F32 hub: 4 elements per iteration (f32x4 intermediate)
    nk_size_t batches = n / 4;
    nk_size_t tail = n % 4;
    nk_size_t from_step = 4 * nk_dtype_bits(from_type) / 8;
    nk_size_t to_step = 4 * nk_dtype_bits(to_type) / 8;
    nk_u8_t const *from_ptr = (nk_u8_t const *)from;
    nk_u8_t *to_ptr = (nk_u8_t *)to;

    for (nk_size_t idx = 0; idx < batches; ++idx, from_ptr += from_step, to_ptr += to_step) {
        // Load and upcast to f32x4
        float32x4_t hub_f32x4;
        switch (from_type) {
        case nk_f32_k: hub_f32x4 = vld1q_f32((nk_f32_t const *)from_ptr); break;
        case nk_f16_k: hub_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16((nk_u16_t const *)from_ptr))); break;
        case nk_bf16_k: hub_f32x4 = nk_bf16x4_to_f32x4_neon_(vld1_u16((nk_u16_t const *)from_ptr)); break;
        case nk_e4m3_k: {
            nk_b32_vec_t in_vec;
            nk_load_b32_serial_(from_ptr, &in_vec);
            hub_f32x4 = nk_e4m3x4_to_f32x4_neon_(in_vec);
        } break;
        case nk_e5m2_k: {
            nk_b32_vec_t in_vec;
            nk_load_b32_serial_(from_ptr, &in_vec);
            hub_f32x4 = nk_e5m2x4_to_f32x4_neon_(in_vec);
        } break;
        case nk_e2m3_k: {
            nk_b32_vec_t in_vec;
            nk_load_b32_serial_(from_ptr, &in_vec);
            hub_f32x4 = nk_e2m3x4_to_f32x4_neon_(in_vec);
        } break;
        case nk_e3m2_k: {
            nk_b32_vec_t in_vec;
            nk_load_b32_serial_(from_ptr, &in_vec);
            hub_f32x4 = nk_e3m2x4_to_f32x4_neon_(in_vec);
        } break;
        case nk_i32_k: hub_f32x4 = vcvtq_f32_s32(vld1q_s32((nk_i32_t const *)from_ptr)); break;
        case nk_u32_k: hub_f32x4 = vcvtq_f32_u32(vld1q_u32((nk_u32_t const *)from_ptr)); break;
        case nk_i16_k: hub_f32x4 = nk_i16x4_to_f32x4_neon_(vld1_s16((nk_i16_t const *)from_ptr)); break;
        case nk_u16_k: hub_f32x4 = nk_u16x4_to_f32x4_neon_(vld1_u16((nk_u16_t const *)from_ptr)); break;
        case nk_i8_k: {
            nk_b32_vec_t in_vec;
            nk_load_b32_serial_(from_ptr, &in_vec);
            hub_f32x4 = nk_i8x4_to_f32x4_neon_(in_vec);
        } break;
        case nk_u8_k: {
            nk_b32_vec_t in_vec;
            nk_load_b32_serial_(from_ptr, &in_vec);
            hub_f32x4 = nk_u8x4_to_f32x4_neon_(in_vec);
        } break;
        default: hub_f32x4 = vdupq_n_f32(0); break;
        }

        // Downcast from f32x4 and store
        switch (to_type) {
        case nk_f32_k: vst1q_f32((nk_f32_t *)to_ptr, hub_f32x4); break;
        case nk_f16_k: vst1_u16((nk_u16_t *)to_ptr, vreinterpret_u16_f16(vcvt_f16_f32(hub_f32x4))); break;
        case nk_bf16_k: vst1_u16((nk_u16_t *)to_ptr, nk_f32x4_to_bf16x4_neon_(hub_f32x4)); break;
        case nk_e4m3_k: {
            nk_b32_vec_t out_vec = nk_f32x4_to_e4m3x4_neon_(hub_f32x4);
            *(nk_u32_t *)to_ptr = out_vec.u32;
        } break;
        case nk_e5m2_k: {
            nk_b32_vec_t out_vec = nk_f32x4_to_e5m2x4_neon_(hub_f32x4);
            *(nk_u32_t *)to_ptr = out_vec.u32;
        } break;
        case nk_e2m3_k: {
            nk_b32_vec_t out_vec = nk_f32x4_to_e2m3x4_neon_(hub_f32x4);
            nk_copy_bytes_(to_ptr, &out_vec, sizeof(nk_b32_vec_t));
        } break;
        case nk_e3m2_k: {
            nk_b32_vec_t out_vec = nk_f32x4_to_e3m2x4_neon_(hub_f32x4);
            nk_copy_bytes_(to_ptr, &out_vec, sizeof(nk_b32_vec_t));
        } break;
        case nk_i32_k: vst1q_s32((nk_i32_t *)to_ptr, vcvtnq_s32_f32(hub_f32x4)); break;
        case nk_u32_k: vst1q_u32((nk_u32_t *)to_ptr, vcvtnq_u32_f32(hub_f32x4)); break;
        case nk_i16_k: vst1_s16((nk_i16_t *)to_ptr, nk_f32x4_to_i16x4_neon_(hub_f32x4)); break;
        case nk_u16_k: vst1_u16((nk_u16_t *)to_ptr, nk_f32x4_to_u16x4_neon_(hub_f32x4)); break;
        case nk_i8_k: nk_f32x4_to_i8x4_neon_(hub_f32x4, (nk_i8_t *)to_ptr); break;
        case nk_u8_k: nk_f32x4_to_u8x4_neon_(hub_f32x4, (nk_u8_t *)to_ptr); break;
        default: break;
        }
    }

    // Handle tail elements with serial fallback
    if (tail) nk_cast_serial(from_ptr, from_type, tail, to_ptr, to_type);
}

#pragma endregion - Public API

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_CAST_NEON_H
