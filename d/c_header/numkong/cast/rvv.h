/**
 *  @brief SIMD-accelerated Type Conversions for RISC-V.
 *  @file include/numkong/cast/rvv.h
 *  @author Ash Vardanian
 *  @date January 13, 2026
 *
 *  @sa include/numkong/cast.h
 *
 *  SpacemiT K1 and similar chips implement RVA22 profile with base RVV 1.0.
 *  This file provides vectorized type conversions for:
 *  - BF16 ↔ F32 (bit manipulation, no hardware support)
 *  - F16 ↔ F32 (bit manipulation, no hardware support)
 *  - E4M3 ↔ F32 (FP8 format for ML inference)
 *  - E5M2 ↔ F32 (FP8 format for ML training)
 *  - i4/u4 unpacking to i8/u8
 *
 *  Mini-float conversions use sign-symmetric magnitude LUTs: every mini-float
 *  format is sign|magnitude, so we store only the positive-half (magnitude)
 *  entries and extract the sign bit separately. This cuts LUT memory by 50-87%
 *  and fixes the E2M3FN NaN bug (E2M3FN has NO NaN; index 31 is +7.5, not NaN).
 *
 *  8-bit formats (e4m3, e5m2): sign = bit 7, magnitude = bits 6:0 (128 entries)
 *  6-bit formats (e2m3, e3m2): sign = bit 5, magnitude = bits 4:0 (32 entries)
 *
 *  @section rvv_cast_instructions Key RVV Cast Instructions
 *
 *      Intrinsic                       Purpose
 *      vzext_vf4_u32m4                 Zero-extend u8 → u32 (4x widening)
 *      vsext_vf4_i32m4                 Sign-extend i8 → i32 (4x widening)
 *      vsll_vx / vsrl_vx               Bit shifts for field extraction
 *      vand_vx                         Bit masking
 *      vor_vv                          Combining bit fields
 *      vfcvt_f_xu_v                    Unsigned int → float
 *      vmseq_vx                        Compare for conditional selection
 *      vmerge_vvm                      Conditional select (blend)
 */
#ifndef NK_CAST_RVV_H
#define NK_CAST_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_cast_serial`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region - Register-to-Register Helpers

/**
 *  @brief Convert bf16 (m1) to f32 (m2) register-to-register.
 *
 *  BF16 is the upper 16 bits of F32 (same sign + exponent + top 7 mantissa bits).
 *  Conversion is simply: f32_bits = bf16_bits << 16.
 */
NK_INTERNAL vfloat32m2_t nk_bf16m1_to_f32m2_rvv_(vuint16m1_t bf16_u16m1, nk_size_t vector_length) {
    vuint32m2_t bits_u32m2 = __riscv_vzext_vf2_u32m2(bf16_u16m1, vector_length);
    bits_u32m2 = __riscv_vsll_vx_u32m2(bits_u32m2, 16, vector_length);
    return __riscv_vreinterpret_v_u32m2_f32m2(bits_u32m2);
}

/**
 *  @brief Convert f32 (m2) to bf16 (m1) register-to-register.
 *
 *  Conversion with round-to-nearest-even (RNE): add (0x7FFF + lsb) to match hardware BF16 behavior.
 */
NK_INTERNAL vuint16m1_t nk_f32m2_to_bf16m1_rvv_(vfloat32m2_t f32_f32m2, nk_size_t vector_length) {
    vuint32m2_t bits_u32m2 = __riscv_vreinterpret_v_f32m2_u32m2(f32_f32m2);
    // Extract LSB of result (bit 16) for round-to-nearest-even
    vuint32m2_t lsb_u32m2 = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(bits_u32m2, 16, vector_length), 1,
                                                  vector_length);
    vuint32m2_t rounding_u32m2 = __riscv_vadd_vx_u32m2(lsb_u32m2, 0x7FFF, vector_length);
    vuint32m2_t rounded_u32m2 = __riscv_vadd_vv_u32m2(bits_u32m2, rounding_u32m2, vector_length);
    vuint32m2_t shifted_u32m2 = __riscv_vsrl_vx_u32m2(rounded_u32m2, 16, vector_length);
    return __riscv_vncvt_x_x_w_u16m1(shifted_u32m2, vector_length);
}

/**
 *  @brief Convert f16 (m1) to f32 (m2) register-to-register.
 *
 *  F16 format: S EEEEE MMMMMMMMMM (1 sign, 5 exponent bits with bias=15, 10 mantissa bits)
 *  F32 format: S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM (1 sign, 8 exponent bits with bias=127, 23 mantissa bits)
 *
 *  Handles all IEEE-754 edge cases: ±zero, denormals, normals, ±inf, NaN.
 */
NK_INTERNAL vfloat32m2_t nk_f16m1_to_f32m2_rvv_(vuint16m1_t f16_u16m1, nk_size_t vector_length) {
    // Widen to 32-bit for manipulation
    vuint32m2_t bits_u32m2 = __riscv_vzext_vf2_u32m2(f16_u16m1, vector_length);
    // Extract sign: (raw >> 15) << 31
    vuint32m2_t sign_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(bits_u32m2, 15, vector_length), 31,
                                                   vector_length);
    // Extract exponent: (raw >> 10) & 0x1F
    vuint32m2_t exponent_u32m2 = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(bits_u32m2, 10, vector_length), 0x1F,
                                                       vector_length);
    // Extract mantissa: raw & 0x3FF
    vuint32m2_t mantissa_u32m2 = __riscv_vand_vx_u32m2(bits_u32m2, 0x3FF, vector_length);

    // Normal path: rebias exponent (15 → 127): add 112, combine
    vuint32m2_t f32_exponent_u32m2 = __riscv_vadd_vx_u32m2(exponent_u32m2, 112, vector_length);
    vuint32m2_t normal_u32m2 = __riscv_vor_vv_u32m2(
        sign_u32m2,
        __riscv_vor_vv_u32m2(__riscv_vsll_vx_u32m2(f32_exponent_u32m2, 23, vector_length),
                             __riscv_vsll_vx_u32m2(mantissa_u32m2, 13, vector_length), vector_length),
        vector_length);

    // Special case: exponent == 0 (zero or denormal)
    // Zero: sign | 0. Denormal: mantissa × 2^(-24), handled via FPU normalization trick.
    // For denormals, convert mantissa to float and subtract 0x0C000000 (24 from exponent),
    // matching the serial implementation. For zeros (mantissa==0), (float)0 - bias = 0.
    vbool16_t is_exp_zero = __riscv_vmseq_vx_u32m2_b16(exponent_u32m2, 0, vector_length);
    vfloat32m2_t mantissa_f32m2 = __riscv_vfcvt_f_xu_v_f32m2(mantissa_u32m2, vector_length);
    vuint32m2_t denorm_bits_u32m2 = __riscv_vsub_vx_u32m2(__riscv_vreinterpret_v_f32m2_u32m2(mantissa_f32m2),
                                                          0x0C000000, vector_length);
    vuint32m2_t zero_or_denorm_u32m2 = __riscv_vor_vv_u32m2(sign_u32m2, denorm_bits_u32m2, vector_length);
    // For true zeros (mantissa==0), the FPU converts 0 to 0x00000000, minus bias wraps,
    // so force to sign-only.
    vbool16_t is_true_zero = __riscv_vmand_mm_b16(
        is_exp_zero, __riscv_vmseq_vx_u32m2_b16(mantissa_u32m2, 0, vector_length), vector_length);
    zero_or_denorm_u32m2 = __riscv_vmerge_vvm_u32m2(zero_or_denorm_u32m2, sign_u32m2, is_true_zero, vector_length);

    // Special case: exponent == 31 (infinity or NaN)
    // sign | 0x7F800000 | (mantissa << 13)
    vbool16_t is_exp_max = __riscv_vmseq_vx_u32m2_b16(exponent_u32m2, 31, vector_length);
    vuint32m2_t inf_nan_u32m2 = __riscv_vor_vv_u32m2(__riscv_vor_vx_u32m2(sign_u32m2, 0x7F800000, vector_length),
                                                     __riscv_vsll_vx_u32m2(mantissa_u32m2, 13, vector_length),
                                                     vector_length);

    // Select: exp==0 → zero_or_denorm, exp==31 → inf_nan, else → normal
    vuint32m2_t result_u32m2 = __riscv_vmerge_vvm_u32m2(normal_u32m2, zero_or_denorm_u32m2, is_exp_zero, vector_length);
    result_u32m2 = __riscv_vmerge_vvm_u32m2(result_u32m2, inf_nan_u32m2, is_exp_max, vector_length);
    return __riscv_vreinterpret_v_u32m2_f32m2(result_u32m2);
}

/**
 *  @brief Convert f32 (m2) to f16 (m1) register-to-register.
 *
 *  Conversion: Rebias exponent from 127 to 15, truncate mantissa from 23 to 10 bits with rounding.
 */
NK_INTERNAL vuint16m1_t nk_f32m2_to_f16m1_rvv_(vfloat32m2_t f32_f32m2, nk_size_t vector_length) {
    vuint32m2_t bits_u32m2 = __riscv_vreinterpret_v_f32m2_u32m2(f32_f32m2);
    // Extract sign: (raw >> 31) << 15
    vuint32m2_t sign_u32m2 = __riscv_vsll_vx_u32m2(__riscv_vsrl_vx_u32m2(bits_u32m2, 31, vector_length), 15,
                                                   vector_length);
    // Extract exponent: (raw >> 23) & 0xFF
    vuint32m2_t exponent_u32m2 = __riscv_vand_vx_u32m2(__riscv_vsrl_vx_u32m2(bits_u32m2, 23, vector_length), 0xFF,
                                                       vector_length);
    // Extract mantissa: raw & 0x7FFFFF
    vuint32m2_t mantissa_u32m2 = __riscv_vand_vx_u32m2(bits_u32m2, 0x7FFFFF, vector_length);
    // Rebias exponent (127 → 15): subtract 112, clamp to [0, 31]
    // Note: This is a simplified conversion that doesn't handle subnormals or overflow properly
    vint32m2_t exponent_i32m2 = __riscv_vsub_vx_i32m2(__riscv_vreinterpret_v_u32m2_i32m2(exponent_u32m2), 112,
                                                      vector_length);
    exponent_i32m2 = __riscv_vmax_vx_i32m2(exponent_i32m2, 0, vector_length);
    vuint32m2_t f16_exponent_u32m2 = __riscv_vreinterpret_v_i32m2_u32m2(
        __riscv_vmin_vx_i32m2(exponent_i32m2, 31, vector_length));
    // Round mantissa: add 0x1000 (half of truncated bits) then shift.
    // If rounding overflows the mantissa (bit 23 set), carry into exponent.
    vuint32m2_t rounded_mantissa_u32m2 = __riscv_vadd_vx_u32m2(mantissa_u32m2, 0x1000, vector_length);
    vbool16_t mantissa_overflow_b16 = __riscv_vmsne_vx_u32m2_b16(
        __riscv_vand_vx_u32m2(rounded_mantissa_u32m2, 0x800000, vector_length), 0, vector_length);
    f16_exponent_u32m2 = __riscv_vadd_vx_u32m2_mu(mantissa_overflow_b16, f16_exponent_u32m2, f16_exponent_u32m2, 1,
                                                  vector_length);
    vuint32m2_t f16_mantissa_u32m2 = __riscv_vsrl_vx_u32m2(rounded_mantissa_u32m2, 13, vector_length);
    f16_mantissa_u32m2 = __riscv_vand_vx_u32m2(f16_mantissa_u32m2, 0x3FF, vector_length);
    // Combine: sign | (exponent << 10) | mantissa
    vuint32m2_t result_u32m2 = __riscv_vor_vv_u32m2(
        sign_u32m2,
        __riscv_vor_vv_u32m2(__riscv_vsll_vx_u32m2(f16_exponent_u32m2, 10, vector_length), f16_mantissa_u32m2,
                             vector_length),
        vector_length);
    return __riscv_vncvt_x_x_w_u16m1(result_u32m2, vector_length);
}

/**
 *  @brief Convert e4m3 (m1) to f32 (m4) via sign-symmetric magnitude LUT.
 *  E4M3FN: sign = bit 7, magnitude = bits 6:0 (128 entries). Sign bit 7 → f32 bit 31 (<<24).
 */
NK_INTERNAL vfloat32m4_t nk_e4m3m1_to_f32m4_rvv_(vuint8m1_t e4m3_u8m1, nk_size_t vector_length) {
    static nk_u32_t const nk_e4m3_mag_to_f32_lut_[128] = {
        0x00000000u, 0x3B000000u, 0x3B800000u, 0x3BC00000u,
        0x3C000000u, 0x3C200000u, 0x3C400000u, 0x3C600000u, /* [  0..  7] */
        0x3C800000u, 0x3C900000u, 0x3CA00000u, 0x3CB00000u,
        0x3CC00000u, 0x3CD00000u, 0x3CE00000u, 0x3CF00000u, /* [  8.. 15] */
        0x3D000000u, 0x3D100000u, 0x3D200000u, 0x3D300000u,
        0x3D400000u, 0x3D500000u, 0x3D600000u, 0x3D700000u, /* [ 16.. 23] */
        0x3D800000u, 0x3D900000u, 0x3DA00000u, 0x3DB00000u,
        0x3DC00000u, 0x3DD00000u, 0x3DE00000u, 0x3DF00000u, /* [ 24.. 31] */
        0x3E000000u, 0x3E100000u, 0x3E200000u, 0x3E300000u,
        0x3E400000u, 0x3E500000u, 0x3E600000u, 0x3E700000u, /* [ 32.. 39] */
        0x3E800000u, 0x3E900000u, 0x3EA00000u, 0x3EB00000u,
        0x3EC00000u, 0x3ED00000u, 0x3EE00000u, 0x3EF00000u, /* [ 40.. 47] */
        0x3F000000u, 0x3F100000u, 0x3F200000u, 0x3F300000u,
        0x3F400000u, 0x3F500000u, 0x3F600000u, 0x3F700000u, /* [ 48.. 55] */
        0x3F800000u, 0x3F900000u, 0x3FA00000u, 0x3FB00000u,
        0x3FC00000u, 0x3FD00000u, 0x3FE00000u, 0x3FF00000u, /* [ 56.. 63] */
        0x40000000u, 0x40100000u, 0x40200000u, 0x40300000u,
        0x40400000u, 0x40500000u, 0x40600000u, 0x40700000u, /* [ 64.. 71] */
        0x40800000u, 0x40900000u, 0x40A00000u, 0x40B00000u,
        0x40C00000u, 0x40D00000u, 0x40E00000u, 0x40F00000u, /* [ 72.. 79] */
        0x41000000u, 0x41100000u, 0x41200000u, 0x41300000u,
        0x41400000u, 0x41500000u, 0x41600000u, 0x41700000u, /* [ 80.. 87] */
        0x41800000u, 0x41900000u, 0x41A00000u, 0x41B00000u,
        0x41C00000u, 0x41D00000u, 0x41E00000u, 0x41F00000u, /* [ 88.. 95] */
        0x42000000u, 0x42100000u, 0x42200000u, 0x42300000u,
        0x42400000u, 0x42500000u, 0x42600000u, 0x42700000u, /* [ 96..103] */
        0x42800000u, 0x42900000u, 0x42A00000u, 0x42B00000u,
        0x42C00000u, 0x42D00000u, 0x42E00000u, 0x42F00000u, /* [104..111] */
        0x43000000u, 0x43100000u, 0x43200000u, 0x43300000u,
        0x43400000u, 0x43500000u, 0x43600000u, 0x43700000u, /* [112..119] */
        0x43800000u, 0x43900000u, 0x43A00000u, 0x43B00000u,
        0x43C00000u, 0x43D00000u, 0x43E00000u, 0x7FC00000u /* [120..127] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e4m3_u8m1, 0x80, vector_length);
    vuint8m1_t mag_u8m1 = __riscv_vand_vx_u8m1(e4m3_u8m1, 0x7F, vector_length);
    vuint32m4_t offsets_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(mag_u8m1, vector_length), 2,
                                                      vector_length);
    vuint32m4_t result_u32m4 = __riscv_vluxei32_v_u32m4(nk_e4m3_mag_to_f32_lut_, offsets_u32m4, vector_length);
    vuint32m4_t sign_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(sign_u8m1, vector_length), 24,
                                                   vector_length);
    return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vv_u32m4(result_u32m4, sign_u32m4, vector_length));
}

/**
 *  @brief Convert e5m2 (m1) to f32 (m4) via sign-symmetric magnitude LUT.
 *  E5M2: sign = bit 7, magnitude = bits 6:0 (128 entries). Sign bit 7 → f32 bit 31 (<<24).
 */
NK_INTERNAL vfloat32m4_t nk_e5m2m1_to_f32m4_rvv_(vuint8m1_t e5m2_u8m1, nk_size_t vector_length) {
    static nk_u32_t const nk_e5m2_mag_to_f32_lut_[128] = {
        0x00000000u, 0x37800000u, 0x38000000u, 0x38400000u,
        0x38800000u, 0x38A00000u, 0x38C00000u, 0x38E00000u, /* [  0..  7] */
        0x39000000u, 0x39200000u, 0x39400000u, 0x39600000u,
        0x39800000u, 0x39A00000u, 0x39C00000u, 0x39E00000u, /* [  8.. 15] */
        0x3A000000u, 0x3A200000u, 0x3A400000u, 0x3A600000u,
        0x3A800000u, 0x3AA00000u, 0x3AC00000u, 0x3AE00000u, /* [ 16.. 23] */
        0x3B000000u, 0x3B200000u, 0x3B400000u, 0x3B600000u,
        0x3B800000u, 0x3BA00000u, 0x3BC00000u, 0x3BE00000u, /* [ 24.. 31] */
        0x3C000000u, 0x3C200000u, 0x3C400000u, 0x3C600000u,
        0x3C800000u, 0x3CA00000u, 0x3CC00000u, 0x3CE00000u, /* [ 32.. 39] */
        0x3D000000u, 0x3D200000u, 0x3D400000u, 0x3D600000u,
        0x3D800000u, 0x3DA00000u, 0x3DC00000u, 0x3DE00000u, /* [ 40.. 47] */
        0x3E000000u, 0x3E200000u, 0x3E400000u, 0x3E600000u,
        0x3E800000u, 0x3EA00000u, 0x3EC00000u, 0x3EE00000u, /* [ 48.. 55] */
        0x3F000000u, 0x3F200000u, 0x3F400000u, 0x3F600000u,
        0x3F800000u, 0x3FA00000u, 0x3FC00000u, 0x3FE00000u, /* [ 56.. 63] */
        0x40000000u, 0x40200000u, 0x40400000u, 0x40600000u,
        0x40800000u, 0x40A00000u, 0x40C00000u, 0x40E00000u, /* [ 64.. 71] */
        0x41000000u, 0x41200000u, 0x41400000u, 0x41600000u,
        0x41800000u, 0x41A00000u, 0x41C00000u, 0x41E00000u, /* [ 72.. 79] */
        0x42000000u, 0x42200000u, 0x42400000u, 0x42600000u,
        0x42800000u, 0x42A00000u, 0x42C00000u, 0x42E00000u, /* [ 80.. 87] */
        0x43000000u, 0x43200000u, 0x43400000u, 0x43600000u,
        0x43800000u, 0x43A00000u, 0x43C00000u, 0x43E00000u, /* [ 88.. 95] */
        0x44000000u, 0x44200000u, 0x44400000u, 0x44600000u,
        0x44800000u, 0x44A00000u, 0x44C00000u, 0x44E00000u, /* [ 96..103] */
        0x45000000u, 0x45200000u, 0x45400000u, 0x45600000u,
        0x45800000u, 0x45A00000u, 0x45C00000u, 0x45E00000u, /* [104..111] */
        0x46000000u, 0x46200000u, 0x46400000u, 0x46600000u,
        0x46800000u, 0x46A00000u, 0x46C00000u, 0x46E00000u, /* [112..119] */
        0x47000000u, 0x47200000u, 0x47400000u, 0x47600000u,
        0x7F800000u, 0x7FC00000u, 0x7FC00000u, 0x7FC00000u /* [120..127] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e5m2_u8m1, 0x80, vector_length);
    vuint8m1_t mag_u8m1 = __riscv_vand_vx_u8m1(e5m2_u8m1, 0x7F, vector_length);
    vuint32m4_t offsets_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(mag_u8m1, vector_length), 2,
                                                      vector_length);
    vuint32m4_t result_u32m4 = __riscv_vluxei32_v_u32m4(nk_e5m2_mag_to_f32_lut_, offsets_u32m4, vector_length);
    vuint32m4_t sign_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(sign_u8m1, vector_length), 24,
                                                   vector_length);
    return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vv_u32m4(result_u32m4, sign_u32m4, vector_length));
}

/**
 *  @brief Convert e2m3 (m1) to f32 (m4) via sign-symmetric magnitude LUT.
 *  E2M3FN: sign = bit 5, magnitude = bits 4:0 (32 entries). Sign bit 5 → f32 bit 31 (<<26).
 */
NK_INTERNAL vfloat32m4_t nk_e2m3m1_to_f32m4_rvv_(vuint8m1_t e2m3_u8m1, nk_size_t vector_length) {
    static nk_u32_t const nk_e2m3_mag_to_f32_lut_[32] = {
        0x00000000u, 0x3E000000u, 0x3E800000u, 0x3EC00000u,
        0x3F000000u, 0x3F200000u, 0x3F400000u, 0x3F600000u, /* [  0..  7] */
        0x3F800000u, 0x3F900000u, 0x3FA00000u, 0x3FB00000u,
        0x3FC00000u, 0x3FD00000u, 0x3FE00000u, 0x3FF00000u, /* [  8.. 15] */
        0x40000000u, 0x40100000u, 0x40200000u, 0x40300000u,
        0x40400000u, 0x40500000u, 0x40600000u, 0x40700000u, /* [ 16.. 23] */
        0x40800000u, 0x40900000u, 0x40A00000u, 0x40B00000u,
        0x40C00000u, 0x40D00000u, 0x40E00000u, 0x40F00000u /* [ 24.. 31] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e2m3_u8m1, 0x20, vector_length);
    vuint8m1_t mag_u8m1 = __riscv_vand_vx_u8m1(e2m3_u8m1, 0x1F, vector_length);
    vuint32m4_t offsets_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(mag_u8m1, vector_length), 2,
                                                      vector_length);
    vuint32m4_t result_u32m4 = __riscv_vluxei32_v_u32m4(nk_e2m3_mag_to_f32_lut_, offsets_u32m4, vector_length);
    vuint32m4_t sign_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(sign_u8m1, vector_length), 26,
                                                   vector_length);
    return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vv_u32m4(result_u32m4, sign_u32m4, vector_length));
}

/**
 *  @brief Convert e3m2 (m1) to f32 (m4) via sign-symmetric magnitude LUT.
 *  E3M2FN: sign = bit 5, magnitude = bits 4:0 (32 entries). Sign bit 5 → f32 bit 31 (<<26).
 */
NK_INTERNAL vfloat32m4_t nk_e3m2m1_to_f32m4_rvv_(vuint8m1_t e3m2_u8m1, nk_size_t vector_length) {
    static nk_u32_t const nk_e3m2_mag_to_f32_lut_[32] = {
        0x00000000u, 0x3D800000u, 0x3E000000u, 0x3E400000u,
        0x3E800000u, 0x3EA00000u, 0x3EC00000u, 0x3EE00000u, /* [  0..  7] */
        0x3F000000u, 0x3F200000u, 0x3F400000u, 0x3F600000u,
        0x3F800000u, 0x3FA00000u, 0x3FC00000u, 0x3FE00000u, /* [  8.. 15] */
        0x40000000u, 0x40200000u, 0x40400000u, 0x40600000u,
        0x40800000u, 0x40A00000u, 0x40C00000u, 0x40E00000u, /* [ 16.. 23] */
        0x41000000u, 0x41200000u, 0x41400000u, 0x41600000u,
        0x41800000u, 0x41A00000u, 0x41C00000u, 0x41E00000u /* [ 24.. 31] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e3m2_u8m1, 0x20, vector_length);
    vuint8m1_t mag_u8m1 = __riscv_vand_vx_u8m1(e3m2_u8m1, 0x1F, vector_length);
    vuint32m4_t offsets_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(mag_u8m1, vector_length), 2,
                                                      vector_length);
    vuint32m4_t result_u32m4 = __riscv_vluxei32_v_u32m4(nk_e3m2_mag_to_f32_lut_, offsets_u32m4, vector_length);
    vuint32m4_t sign_u32m4 = __riscv_vsll_vx_u32m4(__riscv_vzext_vf4_u32m4(sign_u8m1, vector_length), 26,
                                                   vector_length);
    return __riscv_vreinterpret_v_u32m4_f32m4(__riscv_vor_vv_u32m4(result_u32m4, sign_u32m4, vector_length));
}

/** @brief Convert e4m3 (m1) to bf16 (m2) via sign-symmetric magnitude LUT. Sign bit 7 → bf16 bit 15 (<<8). */
NK_INTERNAL vuint16m2_t nk_e4m3m1_to_bf16m2_rvv_(vuint8m1_t e4m3_u8m1, nk_size_t vector_length) {
    static nk_u16_t const nk_e4m3_mag_to_bf16_lut_[128] = {
        0x0000u, 0x3B00u, 0x3B80u, 0x3BC0u, 0x3C00u, 0x3C20u, 0x3C40u, 0x3C60u, /* [  0..  7] */
        0x3C80u, 0x3C90u, 0x3CA0u, 0x3CB0u, 0x3CC0u, 0x3CD0u, 0x3CE0u, 0x3CF0u, /* [  8.. 15] */
        0x3D00u, 0x3D10u, 0x3D20u, 0x3D30u, 0x3D40u, 0x3D50u, 0x3D60u, 0x3D70u, /* [ 16.. 23] */
        0x3D80u, 0x3D90u, 0x3DA0u, 0x3DB0u, 0x3DC0u, 0x3DD0u, 0x3DE0u, 0x3DF0u, /* [ 24.. 31] */
        0x3E00u, 0x3E10u, 0x3E20u, 0x3E30u, 0x3E40u, 0x3E50u, 0x3E60u, 0x3E70u, /* [ 32.. 39] */
        0x3E80u, 0x3E90u, 0x3EA0u, 0x3EB0u, 0x3EC0u, 0x3ED0u, 0x3EE0u, 0x3EF0u, /* [ 40.. 47] */
        0x3F00u, 0x3F10u, 0x3F20u, 0x3F30u, 0x3F40u, 0x3F50u, 0x3F60u, 0x3F70u, /* [ 48.. 55] */
        0x3F80u, 0x3F90u, 0x3FA0u, 0x3FB0u, 0x3FC0u, 0x3FD0u, 0x3FE0u, 0x3FF0u, /* [ 56.. 63] */
        0x4000u, 0x4010u, 0x4020u, 0x4030u, 0x4040u, 0x4050u, 0x4060u, 0x4070u, /* [ 64.. 71] */
        0x4080u, 0x4090u, 0x40A0u, 0x40B0u, 0x40C0u, 0x40D0u, 0x40E0u, 0x40F0u, /* [ 72.. 79] */
        0x4100u, 0x4110u, 0x4120u, 0x4130u, 0x4140u, 0x4150u, 0x4160u, 0x4170u, /* [ 80.. 87] */
        0x4180u, 0x4190u, 0x41A0u, 0x41B0u, 0x41C0u, 0x41D0u, 0x41E0u, 0x41F0u, /* [ 88.. 95] */
        0x4200u, 0x4210u, 0x4220u, 0x4230u, 0x4240u, 0x4250u, 0x4260u, 0x4270u, /* [ 96..103] */
        0x4280u, 0x4290u, 0x42A0u, 0x42B0u, 0x42C0u, 0x42D0u, 0x42E0u, 0x42F0u, /* [104..111] */
        0x4300u, 0x4310u, 0x4320u, 0x4330u, 0x4340u, 0x4350u, 0x4360u, 0x4370u, /* [112..119] */
        0x4380u, 0x4390u, 0x43A0u, 0x43B0u, 0x43C0u, 0x43D0u, 0x43E0u, 0x7FC0u  /* [120..127] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e4m3_u8m1, 0x80, vector_length);
    vuint8m1_t mag_u8m1 = __riscv_vand_vx_u8m1(e4m3_u8m1, 0x7F, vector_length);
    vuint16m2_t offsets_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(mag_u8m1, vector_length), 1,
                                                      vector_length);
    vuint16m2_t result_u16m2 = __riscv_vluxei16_v_u16m2(nk_e4m3_mag_to_bf16_lut_, offsets_u16m2, vector_length);
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 8, vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/** @brief Convert e5m2 (m1) to bf16 (m2) via sign-symmetric magnitude LUT. Sign bit 7 → bf16 bit 15 (<<8). */
NK_INTERNAL vuint16m2_t nk_e5m2m1_to_bf16m2_rvv_(vuint8m1_t e5m2_u8m1, nk_size_t vector_length) {
    static nk_u16_t const nk_e5m2_mag_to_bf16_lut_[128] = {
        0x0000u, 0x3780u, 0x3800u, 0x3840u, 0x3880u, 0x38A0u, 0x38C0u, 0x38E0u, /* [  0..  7] */
        0x3900u, 0x3920u, 0x3940u, 0x3960u, 0x3980u, 0x39A0u, 0x39C0u, 0x39E0u, /* [  8.. 15] */
        0x3A00u, 0x3A20u, 0x3A40u, 0x3A60u, 0x3A80u, 0x3AA0u, 0x3AC0u, 0x3AE0u, /* [ 16.. 23] */
        0x3B00u, 0x3B20u, 0x3B40u, 0x3B60u, 0x3B80u, 0x3BA0u, 0x3BC0u, 0x3BE0u, /* [ 24.. 31] */
        0x3C00u, 0x3C20u, 0x3C40u, 0x3C60u, 0x3C80u, 0x3CA0u, 0x3CC0u, 0x3CE0u, /* [ 32.. 39] */
        0x3D00u, 0x3D20u, 0x3D40u, 0x3D60u, 0x3D80u, 0x3DA0u, 0x3DC0u, 0x3DE0u, /* [ 40.. 47] */
        0x3E00u, 0x3E20u, 0x3E40u, 0x3E60u, 0x3E80u, 0x3EA0u, 0x3EC0u, 0x3EE0u, /* [ 48.. 55] */
        0x3F00u, 0x3F20u, 0x3F40u, 0x3F60u, 0x3F80u, 0x3FA0u, 0x3FC0u, 0x3FE0u, /* [ 56.. 63] */
        0x4000u, 0x4020u, 0x4040u, 0x4060u, 0x4080u, 0x40A0u, 0x40C0u, 0x40E0u, /* [ 64.. 71] */
        0x4100u, 0x4120u, 0x4140u, 0x4160u, 0x4180u, 0x41A0u, 0x41C0u, 0x41E0u, /* [ 72.. 79] */
        0x4200u, 0x4220u, 0x4240u, 0x4260u, 0x4280u, 0x42A0u, 0x42C0u, 0x42E0u, /* [ 80.. 87] */
        0x4300u, 0x4320u, 0x4340u, 0x4360u, 0x4380u, 0x43A0u, 0x43C0u, 0x43E0u, /* [ 88.. 95] */
        0x4400u, 0x4420u, 0x4440u, 0x4460u, 0x4480u, 0x44A0u, 0x44C0u, 0x44E0u, /* [ 96..103] */
        0x4500u, 0x4520u, 0x4540u, 0x4560u, 0x4580u, 0x45A0u, 0x45C0u, 0x45E0u, /* [104..111] */
        0x4600u, 0x4620u, 0x4640u, 0x4660u, 0x4680u, 0x46A0u, 0x46C0u, 0x46E0u, /* [112..119] */
        0x4700u, 0x4720u, 0x4740u, 0x4760u, 0x7F80u, 0x7FC0u, 0x7FC0u, 0x7FC0u  /* [120..127] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e5m2_u8m1, 0x80, vector_length);
    vuint8m1_t mag_u8m1 = __riscv_vand_vx_u8m1(e5m2_u8m1, 0x7F, vector_length);
    vuint16m2_t offsets_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(mag_u8m1, vector_length), 1,
                                                      vector_length);
    vuint16m2_t result_u16m2 = __riscv_vluxei16_v_u16m2(nk_e5m2_mag_to_bf16_lut_, offsets_u16m2, vector_length);
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 8, vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/** @brief Convert e2m3 (m1) to bf16 (m2) via sign-symmetric magnitude LUT. Sign bit 5 → bf16 bit 15 (<<10). */
NK_INTERNAL vuint16m2_t nk_e2m3m1_to_bf16m2_rvv_(vuint8m1_t e2m3_u8m1, nk_size_t vector_length) {
    static nk_u16_t const nk_e2m3_mag_to_bf16_lut_[32] = {
        0x0000u, 0x3E00u, 0x3E80u, 0x3EC0u, 0x3F00u, 0x3F20u, 0x3F40u, 0x3F60u, /* [  0..  7] */
        0x3F80u, 0x3F90u, 0x3FA0u, 0x3FB0u, 0x3FC0u, 0x3FD0u, 0x3FE0u, 0x3FF0u, /* [  8.. 15] */
        0x4000u, 0x4010u, 0x4020u, 0x4030u, 0x4040u, 0x4050u, 0x4060u, 0x4070u, /* [ 16.. 23] */
        0x4080u, 0x4090u, 0x40A0u, 0x40B0u, 0x40C0u, 0x40D0u, 0x40E0u, 0x40F0u  /* [ 24.. 31] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e2m3_u8m1, 0x20, vector_length);
    vuint8m1_t mag_u8m1 = __riscv_vand_vx_u8m1(e2m3_u8m1, 0x1F, vector_length);
    vuint16m2_t offsets_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(mag_u8m1, vector_length), 1,
                                                      vector_length);
    vuint16m2_t result_u16m2 = __riscv_vluxei16_v_u16m2(nk_e2m3_mag_to_bf16_lut_, offsets_u16m2, vector_length);
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 10,
                                                   vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/** @brief Convert e3m2 (m1) to bf16 (m2) via sign-symmetric magnitude LUT. Sign bit 5 → bf16 bit 15 (<<10). */
NK_INTERNAL vuint16m2_t nk_e3m2m1_to_bf16m2_rvv_(vuint8m1_t e3m2_u8m1, nk_size_t vector_length) {
    static nk_u16_t const nk_e3m2_mag_to_bf16_lut_[32] = {
        0x0000u, 0x3D80u, 0x3E00u, 0x3E40u, 0x3E80u, 0x3EA0u, 0x3EC0u, 0x3EE0u, /* [  0..  7] */
        0x3F00u, 0x3F20u, 0x3F40u, 0x3F60u, 0x3F80u, 0x3FA0u, 0x3FC0u, 0x3FE0u, /* [  8.. 15] */
        0x4000u, 0x4020u, 0x4040u, 0x4060u, 0x4080u, 0x40A0u, 0x40C0u, 0x40E0u, /* [ 16.. 23] */
        0x4100u, 0x4120u, 0x4140u, 0x4160u, 0x4180u, 0x41A0u, 0x41C0u, 0x41E0u  /* [ 24.. 31] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e3m2_u8m1, 0x20, vector_length);
    vuint8m1_t mag_u8m1 = __riscv_vand_vx_u8m1(e3m2_u8m1, 0x1F, vector_length);
    vuint16m2_t offsets_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(mag_u8m1, vector_length), 1,
                                                      vector_length);
    vuint16m2_t result_u16m2 = __riscv_vluxei16_v_u16m2(nk_e3m2_mag_to_bf16_lut_, offsets_u16m2, vector_length);
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 10,
                                                   vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/** @brief Convert e4m3 (m1) to f16 (m2) via sign-symmetric magnitude LUT. Sign bit 7 → f16 bit 15 (<<8). */
NK_INTERNAL vuint16m2_t nk_e4m3m1_to_f16m2_rvv_(vuint8m1_t e4m3_u8m1, nk_size_t vector_length) {
    static nk_u16_t const nk_e4m3_mag_to_f16_lut_[128] = {
        0x0000u, 0x1800u, 0x1C00u, 0x1E00u, 0x2000u, 0x2100u, 0x2200u, 0x2300u, /* [  0..  7] */
        0x2400u, 0x2480u, 0x2500u, 0x2580u, 0x2600u, 0x2680u, 0x2700u, 0x2780u, /* [  8.. 15] */
        0x2800u, 0x2880u, 0x2900u, 0x2980u, 0x2A00u, 0x2A80u, 0x2B00u, 0x2B80u, /* [ 16.. 23] */
        0x2C00u, 0x2C80u, 0x2D00u, 0x2D80u, 0x2E00u, 0x2E80u, 0x2F00u, 0x2F80u, /* [ 24.. 31] */
        0x3000u, 0x3080u, 0x3100u, 0x3180u, 0x3200u, 0x3280u, 0x3300u, 0x3380u, /* [ 32.. 39] */
        0x3400u, 0x3480u, 0x3500u, 0x3580u, 0x3600u, 0x3680u, 0x3700u, 0x3780u, /* [ 40.. 47] */
        0x3800u, 0x3880u, 0x3900u, 0x3980u, 0x3A00u, 0x3A80u, 0x3B00u, 0x3B80u, /* [ 48.. 55] */
        0x3C00u, 0x3C80u, 0x3D00u, 0x3D80u, 0x3E00u, 0x3E80u, 0x3F00u, 0x3F80u, /* [ 56.. 63] */
        0x4000u, 0x4080u, 0x4100u, 0x4180u, 0x4200u, 0x4280u, 0x4300u, 0x4380u, /* [ 64.. 71] */
        0x4400u, 0x4480u, 0x4500u, 0x4580u, 0x4600u, 0x4680u, 0x4700u, 0x4780u, /* [ 72.. 79] */
        0x4800u, 0x4880u, 0x4900u, 0x4980u, 0x4A00u, 0x4A80u, 0x4B00u, 0x4B80u, /* [ 80.. 87] */
        0x4C00u, 0x4C80u, 0x4D00u, 0x4D80u, 0x4E00u, 0x4E80u, 0x4F00u, 0x4F80u, /* [ 88.. 95] */
        0x5000u, 0x5080u, 0x5100u, 0x5180u, 0x5200u, 0x5280u, 0x5300u, 0x5380u, /* [ 96..103] */
        0x5400u, 0x5480u, 0x5500u, 0x5580u, 0x5600u, 0x5680u, 0x5700u, 0x5780u, /* [104..111] */
        0x5800u, 0x5880u, 0x5900u, 0x5980u, 0x5A00u, 0x5A80u, 0x5B00u, 0x5B80u, /* [112..119] */
        0x5C00u, 0x5C80u, 0x5D00u, 0x5D80u, 0x5E00u, 0x5E80u, 0x5F00u, 0x7E00u  /* [120..127] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e4m3_u8m1, 0x80, vector_length);
    vuint8m1_t mag_u8m1 = __riscv_vand_vx_u8m1(e4m3_u8m1, 0x7F, vector_length);
    vuint16m2_t offsets_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(mag_u8m1, vector_length), 1,
                                                      vector_length);
    vuint16m2_t result_u16m2 = __riscv_vluxei16_v_u16m2(nk_e4m3_mag_to_f16_lut_, offsets_u16m2, vector_length);
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 8, vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/** @brief Convert e2m3 (m1) to f16 (m2) via sign-symmetric magnitude LUT. Sign bit 5 → f16 bit 15 (<<10). */
NK_INTERNAL vuint16m2_t nk_e2m3m1_to_f16m2_rvv_(vuint8m1_t e2m3_u8m1, nk_size_t vector_length) {
    static nk_u16_t const nk_e2m3_mag_to_f16_lut_[32] = {
        0x0000u, 0x3000u, 0x3400u, 0x3600u, 0x3800u, 0x3900u, 0x3A00u, 0x3B00u, /* [  0..  7] */
        0x3C00u, 0x3C80u, 0x3D00u, 0x3D80u, 0x3E00u, 0x3E80u, 0x3F00u, 0x3F80u, /* [  8.. 15] */
        0x4000u, 0x4080u, 0x4100u, 0x4180u, 0x4200u, 0x4280u, 0x4300u, 0x4380u, /* [ 16.. 23] */
        0x4400u, 0x4480u, 0x4500u, 0x4580u, 0x4600u, 0x4680u, 0x4700u, 0x4780u  /* [ 24.. 31] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e2m3_u8m1, 0x20, vector_length);
    vuint8m1_t mag_u8m1 = __riscv_vand_vx_u8m1(e2m3_u8m1, 0x1F, vector_length);
    vuint16m2_t offsets_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(mag_u8m1, vector_length), 1,
                                                      vector_length);
    vuint16m2_t result_u16m2 = __riscv_vluxei16_v_u16m2(nk_e2m3_mag_to_f16_lut_, offsets_u16m2, vector_length);
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 10,
                                                   vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/** @brief Convert e3m2 (m1) to f16 (m2) via sign-symmetric magnitude LUT. Sign bit 5 → f16 bit 15 (<<10). */
NK_INTERNAL vuint16m2_t nk_e3m2m1_to_f16m2_rvv_(vuint8m1_t e3m2_u8m1, nk_size_t vector_length) {
    static nk_u16_t const nk_e3m2_mag_to_f16_lut_[32] = {
        0x0000u, 0x2C00u, 0x3000u, 0x3200u, 0x3400u, 0x3500u, 0x3600u, 0x3700u, /* [  0..  7] */
        0x3800u, 0x3900u, 0x3A00u, 0x3B00u, 0x3C00u, 0x3D00u, 0x3E00u, 0x3F00u, /* [  8.. 15] */
        0x4000u, 0x4100u, 0x4200u, 0x4300u, 0x4400u, 0x4500u, 0x4600u, 0x4700u, /* [ 16.. 23] */
        0x4800u, 0x4900u, 0x4A00u, 0x4B00u, 0x4C00u, 0x4D00u, 0x4E00u, 0x4F00u  /* [ 24.. 31] */
    };
    vuint8m1_t sign_u8m1 = __riscv_vand_vx_u8m1(e3m2_u8m1, 0x20, vector_length);
    vuint8m1_t mag_u8m1 = __riscv_vand_vx_u8m1(e3m2_u8m1, 0x1F, vector_length);
    vuint16m2_t offsets_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(mag_u8m1, vector_length), 1,
                                                      vector_length);
    vuint16m2_t result_u16m2 = __riscv_vluxei16_v_u16m2(nk_e3m2_mag_to_f16_lut_, offsets_u16m2, vector_length);
    vuint16m2_t sign_u16m2 = __riscv_vsll_vx_u16m2(__riscv_vzext_vf2_u16m2(sign_u8m1, vector_length), 10,
                                                   vector_length);
    return __riscv_vor_vv_u16m2(result_u16m2, sign_u16m2, vector_length);
}

/**
 *  @brief Unpack i4 (m1) nibbles to i8 (m2) register-to-register.
 *
 *  Packed format: byte[i] contains two nibbles:
 *  - High nibble (bits [7:4]) → output[i*2]
 *  - Low nibble (bits [3:0]) → output[i*2+1]
 *
 *  Sign extension: 4-bit signed value [-8,7] extended to 8-bit.
 *  Trick: (x ^ 8) - 8 sign-extends a 4-bit value to larger type.
 *
 *  Returns a tuple of two m1 vectors (high nibbles, low nibbles) for segment store.
 */
NK_INTERNAL vint8m1x2_t nk_i4m1_to_i8m2_rvv_(vuint8m1_t packed_u8m1, nk_size_t vector_length) {
    // Extract high nibble (even indices in output)
    vuint8m1_t hi_u8m1 = __riscv_vsrl_vx_u8m1(packed_u8m1, 4, vector_length);
    // Sign extend: (x ^ 8) - 8
    vint8m1_t hi_i8m1 = __riscv_vsub_vx_i8m1(
        __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(hi_u8m1), 8, vector_length), 8, vector_length);

    // Extract low nibble (odd indices in output)
    vuint8m1_t lo_u8m1 = __riscv_vand_vx_u8m1(packed_u8m1, 0x0F, vector_length);
    // Sign extend: (x ^ 8) - 8
    vint8m1_t lo_i8m1 = __riscv_vsub_vx_i8m1(
        __riscv_vxor_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(lo_u8m1), 8, vector_length), 8, vector_length);

    return __riscv_vcreate_v_i8m1x2(hi_i8m1, lo_i8m1);
}

/**
 *  @brief Unpack u4 (m1) nibbles to u8 (m2) register-to-register.
 *
 *  Returns a tuple of two m1 vectors (high nibbles, low nibbles) for segment store.
 */
NK_INTERNAL vuint8m1x2_t nk_u4m1_to_u8m2_rvv_(vuint8m1_t packed_u8m1, nk_size_t vector_length) {
    // Extract high nibble (even indices in output)
    vuint8m1_t hi_u8m1 = __riscv_vsrl_vx_u8m1(packed_u8m1, 4, vector_length);

    // Extract low nibble (odd indices in output)
    vuint8m1_t lo_u8m1 = __riscv_vand_vx_u8m1(packed_u8m1, 0x0F, vector_length);

    return __riscv_vcreate_v_u8m1x2(hi_u8m1, lo_u8m1);
}

/**
 *  @brief Pack i8 (m2) to i4 (m1) nibbles register-to-register.
 *
 *  Takes a tuple of two m1 vectors (high nibbles, low nibbles from segment load).
 *  Values are clamped to [-8, 7] before packing.
 */
NK_INTERNAL vuint8m1_t nk_i8m2_to_i4m1_rvv_(vint8m1_t hi_i8m1, vint8m1_t lo_i8m1, nk_size_t vector_length) {
    // Clamp to [-8, 7]
    hi_i8m1 = __riscv_vmax_vx_i8m1(__riscv_vmin_vx_i8m1(hi_i8m1, 7, vector_length), -8, vector_length);
    lo_i8m1 = __riscv_vmax_vx_i8m1(__riscv_vmin_vx_i8m1(lo_i8m1, 7, vector_length), -8, vector_length);

    // Convert to unsigned nibbles: value & 0x0F
    vuint8m1_t hi_u4m1 = __riscv_vand_vx_u8m1(__riscv_vreinterpret_v_i8m1_u8m1(hi_i8m1), 0x0F, vector_length);
    vuint8m1_t lo_u4m1 = __riscv_vand_vx_u8m1(__riscv_vreinterpret_v_i8m1_u8m1(lo_i8m1), 0x0F, vector_length);

    // Pack: (hi << 4) | lo
    return __riscv_vor_vv_u8m1(__riscv_vsll_vx_u8m1(hi_u4m1, 4, vector_length), lo_u4m1, vector_length);
}

/**
 *  @brief Pack u8 (m2) to u4 (m1) nibbles register-to-register.
 *
 *  Takes a tuple of two m1 vectors (high nibbles, low nibbles from segment load).
 *  Values are clamped to [0, 15] before packing.
 */
NK_INTERNAL vuint8m1_t nk_u8m2_to_u4m1_rvv_(vuint8m1_t hi_u8m1, vuint8m1_t lo_u8m1, nk_size_t vector_length) {
    // Clamp to [0, 15]
    hi_u8m1 = __riscv_vminu_vx_u8m1(hi_u8m1, 15, vector_length);
    lo_u8m1 = __riscv_vminu_vx_u8m1(lo_u8m1, 15, vector_length);

    // Pack: (hi << 4) | lo
    return __riscv_vor_vv_u8m1(__riscv_vsll_vx_u8m1(hi_u8m1, 4, vector_length), lo_u8m1, vector_length);
}

/**
 *  @brief Convert f32 (m4) to e4m3 (m1) register-to-register.
 *
 *  E4M3FN format: S EEEE MMM (1 sign, 4 exponent bits with bias=7, 3 mantissa bits)
 *  Handles normal, subnormal, overflow, and NaN. Uses RNE mantissa rounding.
 *  E4M3FN quirk: exp=15 with mant=7 is NaN (0x7F), so max finite is 0x7E (exp=15, mant=6).
 */
NK_INTERNAL vuint8m1_t nk_f32m4_to_e4m3m1_rvv_(vfloat32m4_t f32_f32m4, nk_size_t vector_length) {
    vuint32m4_t bits_u32m4 = __riscv_vreinterpret_v_f32m4_u32m4(f32_f32m4);
    vuint32m4_t sign_u32m4 = __riscv_vsrl_vx_u32m4(bits_u32m4, 31, vector_length);
    vuint32m4_t abs_bits_u32m4 = __riscv_vand_vx_u32m4(bits_u32m4, 0x7FFFFFFF, vector_length);
    vuint32m4_t f32_exp_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(bits_u32m4, 23, vector_length), 0xFF,
                                                      vector_length);

    // Round mantissa from 23 to 3 bits using RNE (round to nearest, ties to even)
    vuint32m4_t significand_u32m4 = __riscv_vor_vx_u32m4(__riscv_vand_vx_u32m4(bits_u32m4, 0x007FFFFF, vector_length),
                                                         0x00800000, vector_length);
    vuint32m4_t lsb_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(significand_u32m4, 20, vector_length), 1,
                                                  vector_length);
    vuint32m4_t rounding_bias_u32m4 = __riscv_vadd_vx_u32m4(lsb_u32m4, 0x0007FFFF, vector_length);
    vuint32m4_t rounded_sig_u32m4 = __riscv_vadd_vv_u32m4(significand_u32m4, rounding_bias_u32m4, vector_length);
    vuint32m4_t carry_u32m4 = __riscv_vsrl_vx_u32m4(rounded_sig_u32m4, 24, vector_length);
    vuint32m4_t f32_mantissa_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(rounded_sig_u32m4, 20, vector_length),
                                                           0x07, vector_length);
    // If carry, mantissa becomes 0 (rounded up to next power of 2)
    vbool8_t has_carry_b8 = __riscv_vmsne_vx_u32m4_b8(carry_u32m4, 0, vector_length);
    f32_mantissa_u32m4 = __riscv_vmerge_vxm_u32m4(f32_mantissa_u32m4, 0, has_carry_b8, vector_length);

    // e4m3_exp = f32_exp + carry - 120
    vint32m4_t e4m3_exp_i32m4 = __riscv_vsub_vx_i32m4(
        __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vadd_vv_u32m4(f32_exp_u32m4, carry_u32m4, vector_length)), 120,
        vector_length);

    // Detect subnormal (exp <= 0) and overflow (exp > 15)
    vbool8_t is_subnormal_b8 = __riscv_vmsle_vx_i32m4_b8(e4m3_exp_i32m4, 0, vector_length);
    vbool8_t is_overflow_b8 = __riscv_vmsgt_vx_i32m4_b8(e4m3_exp_i32m4, 15, vector_length);

    // Normal path: clamp exp to [1,15]
    vint32m4_t clamped_exp_i32m4 = __riscv_vmax_vx_i32m4(e4m3_exp_i32m4, 1, vector_length);
    clamped_exp_i32m4 = __riscv_vmin_vx_i32m4(clamped_exp_i32m4, 15, vector_length);
    // E4M3FN quirk: exp=15 with mant=7 is NaN, so cap mantissa to 6 when exp=15
    vbool8_t is_max_exp_b8 = __riscv_vmseq_vx_i32m4_b8(clamped_exp_i32m4, 15, vector_length);
    vuint32m4_t max_mant_u32m4 = __riscv_vmerge_vxm_u32m4(__riscv_vmv_v_x_u32m4(7, vector_length), 6, is_max_exp_b8,
                                                          vector_length);
    vuint32m4_t normal_mant_u32m4 = __riscv_vminu_vv_u32m4(f32_mantissa_u32m4, max_mant_u32m4, vector_length);
    // On overflow, saturate to max finite (exp=15, mant=6 = 0x7E with sign)
    normal_mant_u32m4 = __riscv_vmerge_vxm_u32m4(normal_mant_u32m4, 0x06, is_overflow_b8, vector_length);
    vuint32m4_t normal_u32m4 = __riscv_vor_vv_u32m4(
        __riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length),
        __riscv_vor_vv_u32m4(
            __riscv_vsll_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(clamped_exp_i32m4), 3, vector_length),
            normal_mant_u32m4, vector_length),
        vector_length);

    // Subnormal path: mantissa = round(|f32| * 512)
    vfloat32m4_t abs_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(abs_bits_u32m4);
    vfloat32m4_t scaled_f32m4 = __riscv_vfmul_vf_f32m4(abs_f32m4, 512.0f, vector_length);
    vint32m4_t subnorm_mant_i32m4 = __riscv_vfcvt_x_f_v_i32m4(scaled_f32m4, vector_length); // RNE rounding
    // If rounds to 8+, promote to first normal (exp=1, mant=0 = 0x08)
    vbool8_t promotes_b8 = __riscv_vmsgt_vx_i32m4_b8(subnorm_mant_i32m4, 7, vector_length);
    subnorm_mant_i32m4 = __riscv_vmin_vx_i32m4(subnorm_mant_i32m4, 7, vector_length);
    subnorm_mant_i32m4 = __riscv_vmax_vx_i32m4(subnorm_mant_i32m4, 0, vector_length);
    vuint32m4_t subnorm_u32m4 = __riscv_vor_vv_u32m4(__riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length),
                                                     __riscv_vreinterpret_v_i32m4_u32m4(subnorm_mant_i32m4),
                                                     vector_length);
    vuint32m4_t first_normal_u32m4 = __riscv_vor_vx_u32m4(__riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length), 0x08,
                                                          vector_length);
    subnorm_u32m4 = __riscv_vmerge_vvm_u32m4(subnorm_u32m4, first_normal_u32m4, promotes_b8, vector_length);

    // Select: subnormal when exp <= 0, else normal
    vuint32m4_t result_u32m4 = __riscv_vmerge_vvm_u32m4(normal_u32m4, subnorm_u32m4, is_subnormal_b8, vector_length);

    // Handle NaN: f32 NaN (abs_bits > 0x7F800000) → e4m3 NaN (sign | 0x7F)
    vbool8_t is_nan_b8 = __riscv_vmsgtu_vx_u32m4_b8(abs_bits_u32m4, 0x7F800000, vector_length);
    vuint32m4_t nan_u32m4 = __riscv_vor_vx_u32m4(__riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length), 0x7F,
                                                 vector_length);
    result_u32m4 = __riscv_vmerge_vvm_u32m4(result_u32m4, nan_u32m4, is_nan_b8, vector_length);

    // Narrow u32m4 → u16m2 → u8m1
    vuint16m2_t result_u16m2 = __riscv_vncvt_x_x_w_u16m2(result_u32m4, vector_length);
    return __riscv_vncvt_x_x_w_u8m1(result_u16m2, vector_length);
}

/**
 *  @brief Convert f32 (m4) to e5m2 (m1) register-to-register.
 *
 *  E5M2 format: S EEEEE MM (1 sign, 5 exponent bits with bias=15, 2 mantissa bits)
 *  Handles normal, subnormal, overflow (→ infinity), and NaN. Uses RNE mantissa rounding.
 */
NK_INTERNAL vuint8m1_t nk_f32m4_to_e5m2m1_rvv_(vfloat32m4_t f32_f32m4, nk_size_t vector_length) {
    vuint32m4_t bits_u32m4 = __riscv_vreinterpret_v_f32m4_u32m4(f32_f32m4);
    vuint32m4_t sign_u32m4 = __riscv_vsrl_vx_u32m4(bits_u32m4, 31, vector_length);
    vuint32m4_t abs_bits_u32m4 = __riscv_vand_vx_u32m4(bits_u32m4, 0x7FFFFFFF, vector_length);
    vuint32m4_t f32_exp_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(bits_u32m4, 23, vector_length), 0xFF,
                                                      vector_length);

    // Round mantissa from 23 to 2 bits using RNE
    vuint32m4_t significand_u32m4 = __riscv_vor_vx_u32m4(__riscv_vand_vx_u32m4(bits_u32m4, 0x007FFFFF, vector_length),
                                                         0x00800000, vector_length);
    vuint32m4_t lsb_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(significand_u32m4, 21, vector_length), 1,
                                                  vector_length);
    vuint32m4_t rounding_bias_u32m4 = __riscv_vadd_vx_u32m4(lsb_u32m4, 0x000FFFFF, vector_length);
    vuint32m4_t rounded_sig_u32m4 = __riscv_vadd_vv_u32m4(significand_u32m4, rounding_bias_u32m4, vector_length);
    vuint32m4_t carry_u32m4 = __riscv_vsrl_vx_u32m4(rounded_sig_u32m4, 24, vector_length);
    vuint32m4_t f32_mantissa_u32m4 = __riscv_vand_vx_u32m4(__riscv_vsrl_vx_u32m4(rounded_sig_u32m4, 21, vector_length),
                                                           0x03, vector_length);
    vbool8_t has_carry_b8 = __riscv_vmsne_vx_u32m4_b8(carry_u32m4, 0, vector_length);
    f32_mantissa_u32m4 = __riscv_vmerge_vxm_u32m4(f32_mantissa_u32m4, 0, has_carry_b8, vector_length);

    // e5m2_exp = f32_exp + carry - 112
    vint32m4_t e5m2_exp_i32m4 = __riscv_vsub_vx_i32m4(
        __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vadd_vv_u32m4(f32_exp_u32m4, carry_u32m4, vector_length)), 112,
        vector_length);

    // Detect subnormal (exp <= 0) and overflow (exp > 31)
    vbool8_t is_subnormal_b8 = __riscv_vmsle_vx_i32m4_b8(e5m2_exp_i32m4, 0, vector_length);
    vbool8_t is_overflow_b8 = __riscv_vmsgt_vx_i32m4_b8(e5m2_exp_i32m4, 31, vector_length);

    // Normal path: clamp exp to [1,31], on overflow return infinity (exp=31, mant=0)
    vint32m4_t clamped_exp_i32m4 = __riscv_vmax_vx_i32m4(e5m2_exp_i32m4, 1, vector_length);
    clamped_exp_i32m4 = __riscv_vmin_vx_i32m4(clamped_exp_i32m4, 31, vector_length);
    vuint32m4_t normal_mant_u32m4 = __riscv_vmerge_vxm_u32m4(f32_mantissa_u32m4, 0, is_overflow_b8, vector_length);
    vuint32m4_t normal_u32m4 = __riscv_vor_vv_u32m4(
        __riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length),
        __riscv_vor_vv_u32m4(
            __riscv_vsll_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(clamped_exp_i32m4), 2, vector_length),
            normal_mant_u32m4, vector_length),
        vector_length);

    // Subnormal path: mantissa = round(|f32| * 65536)
    vfloat32m4_t abs_f32m4 = __riscv_vreinterpret_v_u32m4_f32m4(abs_bits_u32m4);
    vfloat32m4_t scaled_f32m4 = __riscv_vfmul_vf_f32m4(abs_f32m4, 65536.0f, vector_length);
    vint32m4_t subnorm_mant_i32m4 = __riscv_vfcvt_x_f_v_i32m4(scaled_f32m4, vector_length);
    vbool8_t promotes_b8 = __riscv_vmsgt_vx_i32m4_b8(subnorm_mant_i32m4, 3, vector_length);
    subnorm_mant_i32m4 = __riscv_vmin_vx_i32m4(subnorm_mant_i32m4, 3, vector_length);
    subnorm_mant_i32m4 = __riscv_vmax_vx_i32m4(subnorm_mant_i32m4, 0, vector_length);
    vuint32m4_t subnorm_u32m4 = __riscv_vor_vv_u32m4(__riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length),
                                                     __riscv_vreinterpret_v_i32m4_u32m4(subnorm_mant_i32m4),
                                                     vector_length);
    vuint32m4_t first_normal_u32m4 = __riscv_vor_vx_u32m4(__riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length), 0x04,
                                                          vector_length);
    subnorm_u32m4 = __riscv_vmerge_vvm_u32m4(subnorm_u32m4, first_normal_u32m4, promotes_b8, vector_length);

    // Select: subnormal when exp <= 0, else normal
    vuint32m4_t result_u32m4 = __riscv_vmerge_vvm_u32m4(normal_u32m4, subnorm_u32m4, is_subnormal_b8, vector_length);

    // Handle NaN: f32 NaN (abs_bits > 0x7F800000) → e5m2 NaN (sign | 0x7D)
    vbool8_t is_nan_b8 = __riscv_vmsgtu_vx_u32m4_b8(abs_bits_u32m4, 0x7F800000, vector_length);
    vuint32m4_t nan_u32m4 = __riscv_vor_vx_u32m4(__riscv_vsll_vx_u32m4(sign_u32m4, 7, vector_length), 0x7D,
                                                 vector_length);
    result_u32m4 = __riscv_vmerge_vvm_u32m4(result_u32m4, nan_u32m4, is_nan_b8, vector_length);

    // Narrow u32m4 → u16m2 → u8m1
    vuint16m2_t result_u16m2 = __riscv_vncvt_x_x_w_u16m2(result_u32m4, vector_length);
    return __riscv_vncvt_x_x_w_u8m1(result_u16m2, vector_length);
}

#pragma endregion - Register - to - Register Helpers

#pragma region - Unified Cast Dispatcher

NK_PUBLIC void nk_cast_rvv(void const *from, nk_dtype_t from_type, nk_size_t count, void *to, nk_dtype_t to_type) {
    // bf16 → f32
    if (from_type == nk_bf16_k && to_type == nk_f32_k) {
        nk_bf16_t const *source = (nk_bf16_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(count);
            vuint16m1_t bf16_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)source, vector_length);
            vfloat32m2_t f32_f32m2 = nk_bf16m1_to_f32m2_rvv_(bf16_u16m1, vector_length);
            __riscv_vse32_v_f32m2(destination, f32_f32m2, vector_length);
        }
        return;
    }

    // f32 → bf16
    if (from_type == nk_f32_k && to_type == nk_bf16_k) {
        nk_f32_t const *source = (nk_f32_t const *)from;
        nk_bf16_t *destination = (nk_bf16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e32m2(count);
            vfloat32m2_t f32_f32m2 = __riscv_vle32_v_f32m2(source, vector_length);
            vuint16m1_t bf16_u16m1 = nk_f32m2_to_bf16m1_rvv_(f32_f32m2, vector_length);
            __riscv_vse16_v_u16m1((nk_u16_t *)destination, bf16_u16m1, vector_length);
        }
        return;
    }

    // f16 → f32
    if (from_type == nk_f16_k && to_type == nk_f32_k) {
        nk_f16_t const *source = (nk_f16_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(count);
            vuint16m1_t f16_u16m1 = __riscv_vle16_v_u16m1((nk_u16_t const *)source, vector_length);
            vfloat32m2_t f32_f32m2 = nk_f16m1_to_f32m2_rvv_(f16_u16m1, vector_length);
            __riscv_vse32_v_f32m2(destination, f32_f32m2, vector_length);
        }
        return;
    }

    // f32 → f16
    if (from_type == nk_f32_k && to_type == nk_f16_k) {
        nk_f32_t const *source = (nk_f32_t const *)from;
        nk_f16_t *destination = (nk_f16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e32m2(count);
            vfloat32m2_t f32_f32m2 = __riscv_vle32_v_f32m2(source, vector_length);
            vuint16m1_t f16_u16m1 = nk_f32m2_to_f16m1_rvv_(f32_f32m2, vector_length);
            __riscv_vse16_v_u16m1((nk_u16_t *)destination, f16_u16m1, vector_length);
        }
        return;
    }

    // e4m3 → f32
    if (from_type == nk_e4m3_k && to_type == nk_f32_k) {
        nk_e4m3_t const *source = (nk_e4m3_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e4m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vfloat32m4_t f32_f32m4 = nk_e4m3m1_to_f32m4_rvv_(e4m3_u8m1, vector_length);
            __riscv_vse32_v_f32m4(destination, f32_f32m4, vector_length);
        }
        return;
    }

    // e5m2 → f32
    if (from_type == nk_e5m2_k && to_type == nk_f32_k) {
        nk_e5m2_t const *source = (nk_e5m2_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e5m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vfloat32m4_t f32_f32m4 = nk_e5m2m1_to_f32m4_rvv_(e5m2_u8m1, vector_length);
            __riscv_vse32_v_f32m4(destination, f32_f32m4, vector_length);
        }
        return;
    }

    // e2m3 → f32
    if (from_type == nk_e2m3_k && to_type == nk_f32_k) {
        nk_e2m3_t const *source = (nk_e2m3_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e2m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vfloat32m4_t f32_f32m4 = nk_e2m3m1_to_f32m4_rvv_(e2m3_u8m1, vector_length);
            __riscv_vse32_v_f32m4(destination, f32_f32m4, vector_length);
        }
        return;
    }

    // e3m2 → f32
    if (from_type == nk_e3m2_k && to_type == nk_f32_k) {
        nk_e3m2_t const *source = (nk_e3m2_t const *)from;
        nk_f32_t *destination = (nk_f32_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e3m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vfloat32m4_t f32_f32m4 = nk_e3m2m1_to_f32m4_rvv_(e3m2_u8m1, vector_length);
            __riscv_vse32_v_f32m4(destination, f32_f32m4, vector_length);
        }
        return;
    }

    // e4m3 → bf16
    if (from_type == nk_e4m3_k && to_type == nk_bf16_k) {
        nk_e4m3_t const *source = (nk_e4m3_t const *)from;
        nk_bf16_t *destination = (nk_bf16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e4m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t bf16_u16m2 = nk_e4m3m1_to_bf16m2_rvv_(e4m3_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, bf16_u16m2, vector_length);
        }
        return;
    }

    // e5m2 → bf16
    if (from_type == nk_e5m2_k && to_type == nk_bf16_k) {
        nk_e5m2_t const *source = (nk_e5m2_t const *)from;
        nk_bf16_t *destination = (nk_bf16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e5m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t bf16_u16m2 = nk_e5m2m1_to_bf16m2_rvv_(e5m2_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, bf16_u16m2, vector_length);
        }
        return;
    }

    // e2m3 → bf16
    if (from_type == nk_e2m3_k && to_type == nk_bf16_k) {
        nk_e2m3_t const *source = (nk_e2m3_t const *)from;
        nk_bf16_t *destination = (nk_bf16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e2m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t bf16_u16m2 = nk_e2m3m1_to_bf16m2_rvv_(e2m3_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, bf16_u16m2, vector_length);
        }
        return;
    }

    // e3m2 → bf16
    if (from_type == nk_e3m2_k && to_type == nk_bf16_k) {
        nk_e3m2_t const *source = (nk_e3m2_t const *)from;
        nk_bf16_t *destination = (nk_bf16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e3m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t bf16_u16m2 = nk_e3m2m1_to_bf16m2_rvv_(e3m2_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, bf16_u16m2, vector_length);
        }
        return;
    }

    // e4m3 → f16
    if (from_type == nk_e4m3_k && to_type == nk_f16_k) {
        nk_e4m3_t const *source = (nk_e4m3_t const *)from;
        nk_f16_t *destination = (nk_f16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e4m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t f16_u16m2 = nk_e4m3m1_to_f16m2_rvv_(e4m3_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, f16_u16m2, vector_length);
        }
        return;
    }

    // e2m3 → f16
    if (from_type == nk_e2m3_k && to_type == nk_f16_k) {
        nk_e2m3_t const *source = (nk_e2m3_t const *)from;
        nk_f16_t *destination = (nk_f16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e2m3_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t f16_u16m2 = nk_e2m3m1_to_f16m2_rvv_(e2m3_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, f16_u16m2, vector_length);
        }
        return;
    }

    // e3m2 → f16
    if (from_type == nk_e3m2_k && to_type == nk_f16_k) {
        nk_e3m2_t const *source = (nk_e3m2_t const *)from;
        nk_f16_t *destination = (nk_f16_t *)to;
        for (nk_size_t vector_length; count > 0;
             count -= vector_length, source += vector_length, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(count);
            vuint8m1_t e3m2_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint16m2_t f16_u16m2 = nk_e3m2m1_to_f16m2_rvv_(e3m2_u8m1, vector_length);
            __riscv_vse16_v_u16m2((nk_u16_t *)destination, f16_u16m2, vector_length);
        }
        return;
    }

    // i4 → i8
    if (from_type == nk_i4_k && to_type == nk_i8_k) {
        nk_i4x2_t const *source = (nk_i4x2_t const *)from;
        nk_i8_t *destination = (nk_i8_t *)to;
        nk_size_t n_bytes = count / 2;
        for (nk_size_t vector_length; n_bytes > 0;
             n_bytes -= vector_length, source += vector_length, destination += vector_length * 2) {
            vector_length = __riscv_vsetvl_e8m1(n_bytes);
            vuint8m1_t packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vint8m1x2_t unpacked_i8m1x2 = nk_i4m1_to_i8m2_rvv_(packed_u8m1, vector_length);
            __riscv_vsseg2e8_v_i8m1x2(destination, unpacked_i8m1x2, vector_length);
        }
        return;
    }

    // u4 → u8
    if (from_type == nk_u4_k && to_type == nk_u8_k) {
        nk_u4x2_t const *source = (nk_u4x2_t const *)from;
        nk_u8_t *destination = (nk_u8_t *)to;
        nk_size_t n_bytes = count / 2;
        for (nk_size_t vector_length; n_bytes > 0;
             n_bytes -= vector_length, source += vector_length, destination += vector_length * 2) {
            vector_length = __riscv_vsetvl_e8m1(n_bytes);
            vuint8m1_t packed_u8m1 = __riscv_vle8_v_u8m1((nk_u8_t const *)source, vector_length);
            vuint8m1x2_t unpacked_u8m1x2 = nk_u4m1_to_u8m2_rvv_(packed_u8m1, vector_length);
            __riscv_vsseg2e8_v_u8m1x2(destination, unpacked_u8m1x2, vector_length);
        }
        return;
    }

    // i8 → i4
    if (from_type == nk_i8_k && to_type == nk_i4_k) {
        nk_i8_t const *source = (nk_i8_t const *)from;
        nk_i4x2_t *destination = (nk_i4x2_t *)to;
        nk_size_t n_bytes = count / 2;
        for (nk_size_t vector_length; n_bytes > 0;
             n_bytes -= vector_length, source += vector_length * 2, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(n_bytes);
            vint8m1x2_t loaded_i8m1x2 = __riscv_vlseg2e8_v_i8m1x2(source, vector_length);
            vint8m1_t hi_i8m1 = __riscv_vget_v_i8m1x2_i8m1(loaded_i8m1x2, 0);
            vint8m1_t lo_i8m1 = __riscv_vget_v_i8m1x2_i8m1(loaded_i8m1x2, 1);
            vuint8m1_t packed_u8m1 = nk_i8m2_to_i4m1_rvv_(hi_i8m1, lo_i8m1, vector_length);
            __riscv_vse8_v_u8m1((nk_u8_t *)destination, packed_u8m1, vector_length);
        }
        return;
    }

    // u8 → u4
    if (from_type == nk_u8_k && to_type == nk_u4_k) {
        nk_u8_t const *source = (nk_u8_t const *)from;
        nk_u4x2_t *destination = (nk_u4x2_t *)to;
        nk_size_t n_bytes = count / 2;
        for (nk_size_t vector_length; n_bytes > 0;
             n_bytes -= vector_length, source += vector_length * 2, destination += vector_length) {
            vector_length = __riscv_vsetvl_e8m1(n_bytes);
            vuint8m1x2_t loaded_u8m1x2 = __riscv_vlseg2e8_v_u8m1x2(source, vector_length);
            vuint8m1_t hi_u8m1 = __riscv_vget_v_u8m1x2_u8m1(loaded_u8m1x2, 0);
            vuint8m1_t lo_u8m1 = __riscv_vget_v_u8m1x2_u8m1(loaded_u8m1x2, 1);
            vuint8m1_t packed_u8m1 = nk_u8m2_to_u4m1_rvv_(hi_u8m1, lo_u8m1, vector_length);
            __riscv_vse8_v_u8m1((nk_u8_t *)destination, packed_u8m1, vector_length);
        }
        return;
    }

    // Fallback to serial for unimplemented conversions
    nk_cast_serial(from, from_type, count, to, to_type);
}

#pragma endregion - Unified Cast Dispatcher

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_CAST_RVV_H
