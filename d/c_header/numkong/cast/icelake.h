/**
 *  @brief SIMD-accelerated Type Conversions for Ice Lake.
 *  @file include/numkong/cast/icelake.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *
 *  @section ice_cast_instructions AVX-512 VBMI2 Instructions
 *
 *      Intrinsic                   Instruction                     Ice         Genoa
 *      _mm512_permutex2var_epi16   VPERMI2W (ZMM, ZMM, ZMM)        3cy @ p5    2cy @ p12
 *      _mm512_test_epi16_mask      VPTESTMW (k, ZMM, ZMM)          3cy @ p5    2cy @ p01
 *      _mm512_mask_mov_epi16       VMOVDQU16 (ZMM{k}, ZMM)         1cy @ p05   1cy @ p05
 *      _mm512_cvtepi16_epi8        VPMOVWB (YMM, ZMM)              3cy @ p5    2cy @ p12
 *
 *  Ice Lake's AVX-512 VBMI2 enables efficient 128-entry LUT lookups via dual VPERMI2W operations.
 *  FP8-to-BF16/F16 conversions use 4 ZMM LUT registers with VPTESTMW for range selection, achieving
 *  ~6 cycles for 32 FP8 conversions. E5M2-to-F16 simplifies to VPSLLW due to matching exponent bias.
 */
#ifndef NK_CAST_ICELAKE_H
#define NK_CAST_ICELAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_ICELAKE

#include "numkong/types.h"
#include "numkong/cast/skylake.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

#pragma region - Vectorized Conversions

/** @brief Convert 32x e4m3 → 32x bf16 via arithmetic + 8-entry subnormal LUT (AVX-512BW).
 *  E4M3 format: S EEEE MMM (bias=7). BF16: S EEEEEEEE MMMMMMM (bias=127).
 *  Normal values (exp != 0): BF16 = sign | ((lower7 << 4) + 0x3C00).
 *  Subnormals (exp == 0, 8 values): looked up from 8-entry LUT via permutexvar.
 *  Memory: 16 bytes (8 × 16-bit entries) vs 256 bytes (128-entry LUT). OCP FP8 v1.0. */
NK_INTERNAL __m512i nk_e4m3x32_to_bf16x32_icelake_(__m256i e4m3x32) {
    __m512i e4m3_i16x32 = _mm512_cvtepu8_epi16(e4m3x32);
    __m512i sign_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16((short)0x80));
    __m512i lower7_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16(0x7F));

    // Normal path: BF16 = ((lower7 << 4) + 0x3C00) | (sign << 8)
    // Formula: E4M3 exp=e, mant=m → BF16 exp = e+120 (bias 7→127), mant = m<<4
    __m512i normal_abs_i16x32 = _mm512_add_epi16(_mm512_slli_epi16(lower7_i16x32, 4), _mm512_set1_epi16(0x3C00));

    // Subnormal LUT (8 entries, repeated 4x for all lanes): E4M3 subnormals are mant × 2^(-9)
    // Values: 0, 1/512, 2/512, 3/512, 4/512, 5/512, 6/512, 7/512
    __m512i subn_lut_i16x32 = _mm512_set_epi16(                          //
        0x3C60, 0x3C40, 0x3C20, 0x3C00, 0x3BC0, 0x3B80, 0x3B00, 0x0000,  // lane 3
        0x3C60, 0x3C40, 0x3C20, 0x3C00, 0x3BC0, 0x3B80, 0x3B00, 0x0000,  // lane 2
        0x3C60, 0x3C40, 0x3C20, 0x3C00, 0x3BC0, 0x3B80, 0x3B00, 0x0000,  // lane 1
        0x3C60, 0x3C40, 0x3C20, 0x3C00, 0x3BC0, 0x3B80, 0x3B00, 0x0000); // lane 0

    // Lookup subnormals via permutexvar (use lower 3 bits of mantissa as index)
    __m512i mant_idx_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16(0x07));
    __m512i subnorm_abs_i16x32 = _mm512_permutexvar_epi16(mant_idx_i16x32, subn_lut_i16x32);

    // Blend: if exponent == 0, use subnormal; else use normal
    __m512i exp_bits_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16(0x78));
    __mmask32 is_subnormal = _mm512_cmpeq_epi16_mask(exp_bits_i16x32, _mm512_setzero_si512());
    __m512i result_abs_i16x32 = _mm512_mask_blend_epi16(is_subnormal, normal_abs_i16x32, subnorm_abs_i16x32);

    // Apply sign: shift E4M3 bit 7 to BF16 bit 15
    sign_i16x32 = _mm512_slli_epi16(sign_i16x32, 8);
    return _mm512_or_si512(result_abs_i16x32, sign_i16x32);
}

/** @brief Convert 32x e5m2 → 32x bf16 via arithmetic + 4-entry subnormal LUT (AVX-512BW).
 *  E5M2 format: S EEEEE MM (bias=15). BF16: S EEEEEEEE MMMMMMM (bias=127).
 *  Normal values (exp != 0): BF16 = sign | ((lower7 << 5) + 0x3800).
 *  Subnormals (exp == 0, 4 values): looked up from 4-entry LUT via permutexvar.
 *  Memory: 8 bytes (4 × 16-bit entries) vs 256 bytes (128-entry LUT). OCP FP8 v1.0. */
NK_INTERNAL __m512i nk_e5m2x32_to_bf16x32_icelake_(__m256i e5m2x32) {
    __m512i e5m2_i16x32 = _mm512_cvtepu8_epi16(e5m2x32);
    __m512i sign_i16x32 = _mm512_and_si512(e5m2_i16x32, _mm512_set1_epi16((short)0x80));
    __m512i lower7_i16x32 = _mm512_and_si512(e5m2_i16x32, _mm512_set1_epi16(0x7F));

    // Normal path: BF16 = ((lower7 << 5) + 0x3800) | (sign << 8)
    // Formula: E5M2 exp=e, mant=m → BF16 exp = e+112 (bias 15→127), mant = m<<5
    __m512i normal_abs_i16x32 = _mm512_add_epi16(_mm512_slli_epi16(lower7_i16x32, 5), _mm512_set1_epi16(0x3800));

    // Subnormal LUT (4 entries, repeated 8x for all lanes): E5M2 subnormals are mant × 2^(-16)
    // Values: 0, 1/65536, 2/65536, 3/65536 (4 entries, then zeros for padding to 8)
    __m512i subn_lut_i16x32 = _mm512_set_epi16(                          //
        0x0000, 0x0000, 0x0000, 0x0000, 0x3840, 0x3800, 0x3780, 0x0000,  // lanes 3-2 (16 entries)
        0x0000, 0x0000, 0x0000, 0x0000, 0x3840, 0x3800, 0x3780, 0x0000,  // lanes 1-0 (16 entries)
        0x0000, 0x0000, 0x0000, 0x0000, 0x3840, 0x3800, 0x3780, 0x0000,  // repeat for remaining
        0x0000, 0x0000, 0x0000, 0x0000, 0x3840, 0x3800, 0x3780, 0x0000); // all 32 entries

    // Lookup subnormals via permutexvar (use lower 2 bits of mantissa as index)
    __m512i mant_idx_i16x32 = _mm512_and_si512(e5m2_i16x32, _mm512_set1_epi16(0x03));
    __m512i subnorm_abs_i16x32 = _mm512_permutexvar_epi16(mant_idx_i16x32, subn_lut_i16x32);

    // Blend: if exponent == 0, use subnormal; else use normal
    __m512i exp_bits_i16x32 = _mm512_and_si512(e5m2_i16x32, _mm512_set1_epi16(0x7C));
    __mmask32 is_subnormal = _mm512_cmpeq_epi16_mask(exp_bits_i16x32, _mm512_setzero_si512());
    __m512i result_abs_i16x32 = _mm512_mask_blend_epi16(is_subnormal, normal_abs_i16x32, subnorm_abs_i16x32);

    // Apply sign: shift E5M2 bit 7 to BF16 bit 15
    sign_i16x32 = _mm512_slli_epi16(sign_i16x32, 8);
    return _mm512_or_si512(result_abs_i16x32, sign_i16x32);
}

/** @brief Convert 32x e2m3 → 32x bf16 via 32-entry LUT lookup (AVX-512BW).
 *  E2M3 format: S EE MMM (bias=1, 6 bits total: sign at bit 5, magnitude bits 4-0).
 *  BF16: S EEEEEEEE MMMMMMM (bias=127). Uses single permutexvar; sign handled separately.
 *  Subnormals (exp=0): value = mant/8. OCP Microscaling Formats v1.0. */
NK_INTERNAL __m512i nk_e2m3x32_to_bf16x32_icelake_(__m256i e2m3x32) {
    __m512i e2m3_i16x32 = _mm512_cvtepu8_epi16(e2m3x32);
    __m512i sign_i16x32 = _mm512_and_si512(e2m3_i16x32, _mm512_set1_epi16(0x20)); // E2M3 sign at bit 5
    __m512i idx_i16x32 = _mm512_and_si512(e2m3_i16x32, _mm512_set1_epi16(0x1F));

    // 32-entry LUT for E2M3 magnitude (5 bits: bits [4:3]=exp, bits [2:0]=mant)
    // E2M3: bias=1, range [0, 7.5] for positive, subnormals = mant/8 (OCP MX v1.0)
    // BF16 = (bf16_exp << 7) | (bf16_mant), where bf16_exp = e2m3_exp + 126, bf16_mant = e2m3_mant << 4
    __m512i const lut_i16x32 = _mm512_set_epi16(                         //
        0x40F0, 0x40E0, 0x40D0, 0x40C0, 0x40B0, 0x40A0, 0x4090, 0x4080,  // [31-24] exp=3: bf16_exp=129
        0x4070, 0x4060, 0x4050, 0x4040, 0x4030, 0x4020, 0x4010, 0x4000,  // [23-16] exp=2: bf16_exp=128
        0x3FF0, 0x3FE0, 0x3FD0, 0x3FC0, 0x3FB0, 0x3FA0, 0x3F90, 0x3F80,  // [15-8] exp=1: bf16_exp=127
        0x3F60, 0x3F40, 0x3F20, 0x3F00, 0x3EC0, 0x3E80, 0x3E00, 0x0000); // [7-0] exp=0: subnormals 7/8..1/8, 0

    // Single permutexvar for 32-entry lookup
    __m512i result_i16x32 = _mm512_permutexvar_epi16(idx_i16x32, lut_i16x32);

    // Apply sign: shift E2M3 bit 5 to BF16 bit 15, then OR
    sign_i16x32 = _mm512_slli_epi16(sign_i16x32, 10);
    return _mm512_or_si512(result_i16x32, sign_i16x32);
}

/** @brief Convert 32x e3m2 → 32x bf16 via 32-entry LUT lookup (AVX-512BW).
 *  E3M2 format: S EEE MM (bias=3, 6 bits total: sign at bit 7, magnitude bits 4-0).
 *  BF16: S EEEEEEEE MMMMMMM (bias=127). Uses single permutexvar; sign handled separately. */
NK_INTERNAL __m512i nk_e3m2x32_to_bf16x32_icelake_(__m256i e3m2x32) {
    __m512i e3m2_i16x32 = _mm512_cvtepu8_epi16(e3m2x32);
    __m512i sign_i16x32 = _mm512_and_si512(e3m2_i16x32, _mm512_set1_epi16(0x20)); // E3M2 sign at bit 5
    __m512i idx_i16x32 = _mm512_and_si512(e3m2_i16x32, _mm512_set1_epi16(0x1F));

    // 32-entry LUT for E3M2 magnitude (5 bits: bits [4:2]=exp, bits [1:0]=mant)
    // E3M2: bias=3, range [0, 28] for positive, subnormals = mant/16 (OCP Microscaling v1.0)
    // BF16 = (bf16_exp << 7) | (bf16_mant), where bf16_exp = e3m2_exp + 124, bf16_mant = e3m2_mant << 5
    __m512i const lut_i16x32 = _mm512_set_epi16( //
        0x41E0, 0x41C0, 0x41A0, 0x4180,          // [31-28] exp=7, mant=3-0: bf16_exp=131
        0x4160, 0x4140, 0x4120, 0x4100,          // [27-24] exp=6, mant=3-0: bf16_exp=130
        0x40E0, 0x40C0, 0x40A0, 0x4080,          // [23-20] exp=5, mant=3-0: bf16_exp=129
        0x4060, 0x4040, 0x4020, 0x4000,          // [19-16] exp=4, mant=3-0: bf16_exp=128
        0x3FE0, 0x3FC0, 0x3FA0, 0x3F80,          // [15-12] exp=3, mant=3-0: bf16_exp=127
        0x3F60, 0x3F40, 0x3F20, 0x3F00,          // [11-8] exp=2, mant=3-0: bf16_exp=126
        0x3EE0, 0x3EC0, 0x3EA0, 0x3E80,          // [7-4] exp=1, mant=3-0: bf16_exp=125
        0x3E40, 0x3E00, 0x3D80, 0x0000);         // [3-0] exp=0: subnormals 3/16, 2/16, 1/16, 0

    // Single permutexvar for 32-entry lookup
    __m512i result_i16x32 = _mm512_permutexvar_epi16(idx_i16x32, lut_i16x32);

    // Apply sign: shift E3M2 bit 5 to BF16 bit 15, then OR
    sign_i16x32 = _mm512_slli_epi16(sign_i16x32, 10);
    return _mm512_or_si512(result_i16x32, sign_i16x32);
}

/** @brief Convert 32x e4m3 → 32x f16 via 128-entry LUT lookup (AVX-512BW).
 *  E4M3 format: S EEEE MMM (bias=7). F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Uses permutex2var for fast LUT lookup; sign handled separately via shift+OR.
 *  Handles all corner cases: zero, subnormals, normals, and NaN. */
NK_INTERNAL __m512i nk_e4m3x32_to_f16x32_icelake_(__m256i e4m3x32) {
    __m512i e4m3_i16x32 = _mm512_cvtepu8_epi16(e4m3x32);
    __m512i sign_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16((short)0x80));
    __m512i idx_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16(0x7F));

    // 128-entry LUT for E4M3 absolute values to F16, split into 4x32 chunks
    // Subnormals (idx 0-7): 0, 1/512, ..., 7/512 mapped to F16
    // Normals (idx 8-126): F16 = (lower7 << 7) + 0x2000
    // NaN (idx 127): 0x7E00
    __m512i const lut0_i16x32 = _mm512_set_epi16(                        // indices 0-31
        0x2F80, 0x2F00, 0x2E80, 0x2E00, 0x2D80, 0x2D00, 0x2C80, 0x2C00,  // idx 31-24
        0x2B80, 0x2B00, 0x2A80, 0x2A00, 0x2980, 0x2900, 0x2880, 0x2800,  // idx 23-16
        0x2780, 0x2700, 0x2680, 0x2600, 0x2580, 0x2500, 0x2480, 0x2400,  // idx 15-8
        0x2300, 0x2200, 0x2100, 0x2000, 0x1E00, 0x1C00, 0x1800, 0x0000); // idx 7-0
    __m512i const lut1_i16x32 = _mm512_set_epi16(                        // indices 32-63
        0x3F80, 0x3F00, 0x3E80, 0x3E00, 0x3D80, 0x3D00, 0x3C80, 0x3C00,  // idx 63-56
        0x3B80, 0x3B00, 0x3A80, 0x3A00, 0x3980, 0x3900, 0x3880, 0x3800,  // idx 55-48
        0x3780, 0x3700, 0x3680, 0x3600, 0x3580, 0x3500, 0x3480, 0x3400,  // idx 47-40
        0x3380, 0x3300, 0x3280, 0x3200, 0x3180, 0x3100, 0x3080, 0x3000); // idx 39-32
    __m512i const lut2_i16x32 = _mm512_set_epi16(                        // indices 64-95
        0x4F80, 0x4F00, 0x4E80, 0x4E00, 0x4D80, 0x4D00, 0x4C80, 0x4C00,  // idx 95-88
        0x4B80, 0x4B00, 0x4A80, 0x4A00, 0x4980, 0x4900, 0x4880, 0x4800,  // idx 87-80
        0x4780, 0x4700, 0x4680, 0x4600, 0x4580, 0x4500, 0x4480, 0x4400,  // idx 79-72
        0x4380, 0x4300, 0x4280, 0x4200, 0x4180, 0x4100, 0x4080, 0x4000); // idx 71-64
    __m512i const lut3_i16x32 = _mm512_set_epi16(                        // indices 96-127
        0x7E00, 0x5F00, 0x5E80, 0x5E00, 0x5D80, 0x5D00, 0x5C80, 0x5C00,  // idx 127-120
        0x5B80, 0x5B00, 0x5A80, 0x5A00, 0x5980, 0x5900, 0x5880, 0x5800,  // idx 119-112
        0x5780, 0x5700, 0x5680, 0x5600, 0x5580, 0x5500, 0x5480, 0x5400,  // idx 111-104
        0x5380, 0x5300, 0x5280, 0x5200, 0x5180, 0x5100, 0x5080, 0x5000); // idx 103-96

    // 2x permutex2var for 64-entry lookup each, then select based on bit 6
    __m512i result_low_i16x32 = _mm512_permutex2var_epi16(lut0_i16x32, idx_i16x32, lut1_i16x32);
    __m512i result_high_i16x32 = _mm512_permutex2var_epi16(lut2_i16x32, idx_i16x32, lut3_i16x32);

    // Select between low (idx 0-63) and high (idx 64-127) based on bit 6
    __mmask32 use_high_mask = _mm512_test_epi16_mask(idx_i16x32, _mm512_set1_epi16(0x40));
    __m512i result_i16x32 = _mm512_mask_mov_epi16(result_low_i16x32, use_high_mask, result_high_i16x32);

    // Apply sign: shift sign bit to bit 15, then OR
    sign_i16x32 = _mm512_slli_epi16(sign_i16x32, 8);
    return _mm512_or_si512(result_i16x32, sign_i16x32);
}

/** @brief Convert 32x e5m2 → 32x f16 via simple bit shift (AVX-512BW).
 *  E5M2 format: S EEEEE MM (bias=15). F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Same exponent bias means F16 = (lower7 << 8) | (sign << 15).
 *  Handles all corner cases: zero, subnormals, normals, infinity, and NaN. */
NK_INTERNAL __m512i nk_e5m2x32_to_f16x32_icelake_(__m256i e5m2x32) {
    __m512i e5m2_i16x32 = _mm512_cvtepu8_epi16(e5m2x32);
    __m512i sign_i16x32 = _mm512_and_si512(e5m2_i16x32, _mm512_set1_epi16((short)0x80));
    __m512i lower7_i16x32 = _mm512_and_si512(e5m2_i16x32, _mm512_set1_epi16(0x7F));

    // F16 = (lower7 << 8) | (sign << 15)
    // Works for all cases: subnormals, normals, infinity, and NaN
    __m512i result_i16x32 = _mm512_slli_epi16(lower7_i16x32, 8);
    sign_i16x32 = _mm512_slli_epi16(sign_i16x32, 8);
    return _mm512_or_si512(result_i16x32, sign_i16x32);
}

/** @brief Convert 32x bf16 → 32x e4m3 via bit manipulation (AVX-512BW).
 *  BF16: S EEEEEEEE MMMMMMM (bias=127). E4M3: S EEEE MMM (bias=7).
 *  Handles normal, subnormal, and overflow cases with RNE rounding. */
NK_INTERNAL __m256i nk_bf16x32_to_e4m3x32_icelake_(__m512i bf16x32) {
    __m512i sign_i16x32 = _mm512_srli_epi16(bf16x32, 15);
    __m512i bf16_exp_i16x32 = _mm512_and_si512(_mm512_srli_epi16(bf16x32, 7), _mm512_set1_epi16(0xFF));

    // Round mantissa from 7 to 3 bits using RNE (round to nearest, ties to even)
    __m512i significand_i16x32 = _mm512_or_si512(_mm512_and_si512(bf16x32, _mm512_set1_epi16(0x7F)),
                                                 _mm512_set1_epi16(0x80)); // Add implicit 1 bit
    __m512i lsb_i16x32 = _mm512_and_si512(_mm512_srli_epi16(significand_i16x32, 4), _mm512_set1_epi16(1));
    __m512i rounding_bias_i16x32 = _mm512_add_epi16(_mm512_set1_epi16(0x07), lsb_i16x32);
    __m512i rounded_sig_i16x32 = _mm512_add_epi16(significand_i16x32, rounding_bias_i16x32);
    __m512i carry_i16x32 = _mm512_srli_epi16(rounded_sig_i16x32, 8); // Carry into exponent if bit 8 set
    __m512i bf16_mantissa_i16x32 = _mm512_and_si512(_mm512_srli_epi16(rounded_sig_i16x32, 4), _mm512_set1_epi16(0x07));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    bf16_mantissa_i16x32 = _mm512_andnot_si512(_mm512_slli_epi16(carry_i16x32, 15), bf16_mantissa_i16x32);
    __m512i e4m3_exp_i16x32 = _mm512_sub_epi16(_mm512_add_epi16(bf16_exp_i16x32, carry_i16x32), _mm512_set1_epi16(120));

    // Detect underflow (exp <= 0) and overflow (exp > 15)
    __mmask32 is_subnormal = _mm512_cmpgt_epi16_mask(_mm512_set1_epi16(1), e4m3_exp_i16x32);
    __mmask32 overflow = _mm512_cmpgt_epi16_mask(e4m3_exp_i16x32, _mm512_set1_epi16(15));

    // Normal path: clamp exp to [1,15]
    // e4m3FN quirk: exp=15 with mantissa=7 is NaN (0x7F), so clamp mantissa to 6 when exp=15.
    __m512i clamped_exp_i16x32 = _mm512_max_epi16(e4m3_exp_i16x32, _mm512_set1_epi16(1));
    clamped_exp_i16x32 = _mm512_min_epi16(clamped_exp_i16x32, _mm512_set1_epi16(15));
    __mmask32 is_max_exp = _mm512_cmpeq_epi16_mask(clamped_exp_i16x32, _mm512_set1_epi16(15));
    __m512i max_mantissa_i16x32 = _mm512_mask_blend_epi16(is_max_exp, _mm512_set1_epi16(7), _mm512_set1_epi16(6));
    __m512i normal_mantissa_i16x32 = _mm512_min_epi16(bf16_mantissa_i16x32, max_mantissa_i16x32);
    normal_mantissa_i16x32 = _mm512_mask_blend_epi16(overflow, normal_mantissa_i16x32, _mm512_set1_epi16(0x06));
    __m512i normal_e4m3_i16x32 = _mm512_or_si512(
        _mm512_slli_epi16(sign_i16x32, 7),
        _mm512_or_si512(_mm512_slli_epi16(clamped_exp_i16x32, 3), normal_mantissa_i16x32));

    // Subnormal path: compute via f32 to get correct rounding
    // bf16 to f32 is just left shift by 16
    __m512i bf16_low_i32x16 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(bf16x32));
    __m512i bf16_high_i32x16 = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(bf16x32, 1));
    __m512 f32_low = _mm512_castsi512_ps(_mm512_slli_epi32(bf16_low_i32x16, 16));
    __m512 f32_high = _mm512_castsi512_ps(_mm512_slli_epi32(bf16_high_i32x16, 16));
    __m512 abs_f32_low = _mm512_and_ps(f32_low, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 abs_f32_high = _mm512_and_ps(f32_high, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 scaled_low = _mm512_mul_ps(abs_f32_low, _mm512_set1_ps(512.0f));
    __m512 scaled_high = _mm512_mul_ps(abs_f32_high, _mm512_set1_ps(512.0f));
    __m512i subnorm_mant_low_i32x16 = _mm512_cvtps_epi32(scaled_low);
    __m512i subnorm_mant_high_i32x16 = _mm512_cvtps_epi32(scaled_high);
    __m256i subnorm_mant_low_i16x16 = _mm512_cvtepi32_epi16(subnorm_mant_low_i32x16);
    __m256i subnorm_mant_high_i16x16 = _mm512_cvtepi32_epi16(subnorm_mant_high_i32x16);
    __m512i subnorm_mantissa_i16x32 = _mm512_inserti64x4(_mm512_castsi256_si512(subnorm_mant_low_i16x16),
                                                         subnorm_mant_high_i16x16, 1);
    __mmask32 promotes_to_normal = _mm512_cmpgt_epi16_mask(subnorm_mantissa_i16x32, _mm512_set1_epi16(7));
    subnorm_mantissa_i16x32 = _mm512_min_epi16(subnorm_mantissa_i16x32, _mm512_set1_epi16(7));
    subnorm_mantissa_i16x32 = _mm512_max_epi16(subnorm_mantissa_i16x32, _mm512_setzero_si512());
    __m512i subnorm_e4m3_i16x32 = _mm512_or_si512(_mm512_slli_epi16(sign_i16x32, 7), subnorm_mantissa_i16x32);
    __m512i first_normal_e4m3_i16x32 = _mm512_or_si512(_mm512_slli_epi16(sign_i16x32, 7), _mm512_set1_epi16(0x08));
    subnorm_e4m3_i16x32 = _mm512_mask_blend_epi16(promotes_to_normal, subnorm_e4m3_i16x32, first_normal_e4m3_i16x32);

    // Blend: use subnormal result when exp <= 0
    __m512i e4m3_i16x32 = _mm512_mask_blend_epi16(is_subnormal, normal_e4m3_i16x32, subnorm_e4m3_i16x32);

    // Pack 32 i16s to 32 unsigned i8s via AVX-512BW
    return _mm512_cvtepi16_epi8(e4m3_i16x32);
}

/** @brief Convert 32x bf16 → 32x e5m2 via bit manipulation (AVX-512BW).
 *  BF16: S EEEEEEEE MMMMMMM (bias=127). E5M2: S EEEEE MM (bias=15).
 *  Handles normal, subnormal, and overflow cases with RNE rounding. */
NK_INTERNAL __m256i nk_bf16x32_to_e5m2x32_icelake_(__m512i bf16x32) {
    __m512i sign_i16x32 = _mm512_srli_epi16(bf16x32, 15);
    __m512i bf16_exp_i16x32 = _mm512_and_si512(_mm512_srli_epi16(bf16x32, 7), _mm512_set1_epi16(0xFF));

    // Round mantissa from 7 to 2 bits using RNE (round to nearest, ties to even)
    __m512i significand_i16x32 = _mm512_or_si512(_mm512_and_si512(bf16x32, _mm512_set1_epi16(0x7F)),
                                                 _mm512_set1_epi16(0x80)); // Add implicit 1 bit
    __m512i lsb_i16x32 = _mm512_and_si512(_mm512_srli_epi16(significand_i16x32, 5), _mm512_set1_epi16(1));
    __m512i rounding_bias_i16x32 = _mm512_add_epi16(_mm512_set1_epi16(0x0F), lsb_i16x32);
    __m512i rounded_sig_i16x32 = _mm512_add_epi16(significand_i16x32, rounding_bias_i16x32);
    __m512i carry_i16x32 = _mm512_srli_epi16(rounded_sig_i16x32, 8); // Carry into exponent if bit 8 set
    __m512i bf16_mantissa_i16x32 = _mm512_and_si512(_mm512_srli_epi16(rounded_sig_i16x32, 5), _mm512_set1_epi16(0x03));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    bf16_mantissa_i16x32 = _mm512_andnot_si512(_mm512_slli_epi16(carry_i16x32, 15), bf16_mantissa_i16x32);
    __m512i e5m2_exp_i16x32 = _mm512_sub_epi16(_mm512_add_epi16(bf16_exp_i16x32, carry_i16x32), _mm512_set1_epi16(112));

    // Detect subnormal (exp <= 0) and overflow (exp > 31)
    __mmask32 is_subnormal = _mm512_cmpgt_epi16_mask(_mm512_set1_epi16(1), e5m2_exp_i16x32);
    __mmask32 overflow = _mm512_cmpgt_epi16_mask(e5m2_exp_i16x32, _mm512_set1_epi16(31));

    // Normal path: clamp exp to [1,31], on overflow return infinity (exp=31, mantissa=0 = 0x7C)
    __m512i clamped_exp_i16x32 = _mm512_max_epi16(e5m2_exp_i16x32, _mm512_set1_epi16(1));
    clamped_exp_i16x32 = _mm512_min_epi16(clamped_exp_i16x32, _mm512_set1_epi16(31));
    __m512i normal_mantissa_i16x32 = _mm512_mask_blend_epi16(overflow, bf16_mantissa_i16x32, _mm512_setzero_si512());
    __m512i normal_e5m2_i16x32 = _mm512_or_si512(
        _mm512_slli_epi16(sign_i16x32, 7),
        _mm512_or_si512(_mm512_slli_epi16(clamped_exp_i16x32, 2), normal_mantissa_i16x32));

    // Subnormal path: compute via f32 to get correct rounding
    __m512i bf16_low_i32x16 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(bf16x32));
    __m512i bf16_high_i32x16 = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(bf16x32, 1));
    __m512 f32_low = _mm512_castsi512_ps(_mm512_slli_epi32(bf16_low_i32x16, 16));
    __m512 f32_high = _mm512_castsi512_ps(_mm512_slli_epi32(bf16_high_i32x16, 16));
    __m512 abs_f32_low = _mm512_and_ps(f32_low, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 abs_f32_high = _mm512_and_ps(f32_high, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 scaled_low = _mm512_mul_ps(abs_f32_low, _mm512_set1_ps(65536.0f));
    __m512 scaled_high = _mm512_mul_ps(abs_f32_high, _mm512_set1_ps(65536.0f));
    __m512i subnorm_mant_low_i32x16 = _mm512_cvtps_epi32(scaled_low);
    __m512i subnorm_mant_high_i32x16 = _mm512_cvtps_epi32(scaled_high);
    __m256i subnorm_mant_low_i16x16 = _mm512_cvtepi32_epi16(subnorm_mant_low_i32x16);
    __m256i subnorm_mant_high_i16x16 = _mm512_cvtepi32_epi16(subnorm_mant_high_i32x16);
    __m512i subnorm_mantissa_i16x32 = _mm512_inserti64x4(_mm512_castsi256_si512(subnorm_mant_low_i16x16),
                                                         subnorm_mant_high_i16x16, 1);
    __mmask32 promotes_to_normal = _mm512_cmpgt_epi16_mask(subnorm_mantissa_i16x32, _mm512_set1_epi16(3));
    subnorm_mantissa_i16x32 = _mm512_min_epi16(subnorm_mantissa_i16x32, _mm512_set1_epi16(3));
    subnorm_mantissa_i16x32 = _mm512_max_epi16(subnorm_mantissa_i16x32, _mm512_setzero_si512());
    __m512i subnorm_e5m2_i16x32 = _mm512_or_si512(_mm512_slli_epi16(sign_i16x32, 7), subnorm_mantissa_i16x32);
    __m512i first_normal_e5m2_i16x32 = _mm512_or_si512(_mm512_slli_epi16(sign_i16x32, 7), _mm512_set1_epi16(0x04));
    subnorm_e5m2_i16x32 = _mm512_mask_blend_epi16(promotes_to_normal, subnorm_e5m2_i16x32, first_normal_e5m2_i16x32);

    // Blend: use subnormal result when exp <= 0
    __m512i e5m2_i16x32 = _mm512_mask_blend_epi16(is_subnormal, normal_e5m2_i16x32, subnorm_e5m2_i16x32);

    // Pack 32 i16s to 32 unsigned i8s via AVX-512BW
    return _mm512_cvtepi16_epi8(e5m2_i16x32);
}

/** @brief Load 32x e4m3 from memory and convert to 32x bf16 (Ice Lake AVX-512BW). */
NK_INTERNAL void nk_load_e4m3x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm = nk_e4m3x32_to_bf16x32_icelake_(_mm256_loadu_si256((__m256i const *)src));
}

/** @brief Partial load n e4m3 elements from memory and convert to bf16 (Ice Lake AVX-512BW). */
NK_INTERNAL void nk_partial_load_e4m3x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
    __m256i e4m3_partial = _mm256_maskz_loadu_epi8(mask, src);
    dst->zmm = nk_e4m3x32_to_bf16x32_icelake_(e4m3_partial);
}

/** @brief Load 32x e5m2 from memory and convert to 32x bf16 (Ice Lake AVX-512BW). */
NK_INTERNAL void nk_load_e5m2x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm = nk_e5m2x32_to_bf16x32_icelake_(_mm256_loadu_si256((__m256i const *)src));
}

/** @brief Partial load n e5m2 elements from memory and convert to bf16 (Ice Lake AVX-512BW). */
NK_INTERNAL void nk_partial_load_e5m2x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
    __m256i e5m2_partial = _mm256_maskz_loadu_epi8(mask, src);
    dst->zmm = nk_e5m2x32_to_bf16x32_icelake_(e5m2_partial);
}

/** @brief Load 32x e2m3 from memory and convert to 32x bf16 (Ice Lake AVX-512BW). */
NK_INTERNAL void nk_load_e2m3x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm = nk_e2m3x32_to_bf16x32_icelake_(_mm256_loadu_si256((__m256i const *)src));
}

/** @brief Partial load n e2m3 elements from memory and convert to bf16 (Ice Lake AVX-512BW). */
NK_INTERNAL void nk_partial_load_e2m3x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
    __m256i e2m3_partial = _mm256_maskz_loadu_epi8(mask, src);
    dst->zmm = nk_e2m3x32_to_bf16x32_icelake_(e2m3_partial);
}

/** @brief Load 32x e3m2 from memory and convert to 32x bf16 (Ice Lake AVX-512BW). */
NK_INTERNAL void nk_load_e3m2x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm = nk_e3m2x32_to_bf16x32_icelake_(_mm256_loadu_si256((__m256i const *)src));
}

/** @brief Partial load n e3m2 elements from memory and convert to bf16 (Ice Lake AVX-512BW). */
NK_INTERNAL void nk_partial_load_e3m2x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
    __m256i e3m2_partial = _mm256_maskz_loadu_epi8(mask, src);
    dst->zmm = nk_e3m2x32_to_bf16x32_icelake_(e3m2_partial);
}

#pragma endregion - Vectorized Conversions

#pragma region - Public API

NK_PUBLIC void nk_cast_icelake(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
    // Group 1: Conversions to bf16 (e4m3 → bf16, e5m2 → bf16)
    if (to_type == nk_bf16_k && (from_type == nk_e4m3_k || from_type == nk_e5m2_k)) {
        nk_e4m3_t const *from_ptr = (nk_e4m3_t const *)from;
        nk_bf16_t *to_ptr = (nk_bf16_t *)to;
        for (nk_size_t idx = 0; idx < n; idx += 32) {
            nk_size_t remaining = n - idx;
            __mmask32 mask = (remaining >= 32) ? 0xFFFFFFFF : _bzhi_u32(0xFFFFFFFF, (unsigned)remaining);
            __m256i in_f8x32 = _mm256_maskz_loadu_epi8(mask, from_ptr + idx);
            __m512i out_bf16x32 = (from_type == nk_e4m3_k) ? nk_e4m3x32_to_bf16x32_icelake_(in_f8x32)
                                                           : nk_e5m2x32_to_bf16x32_icelake_(in_f8x32);
            _mm512_mask_storeu_epi16(to_ptr + idx, mask, out_bf16x32);
        }
    }

    // Group 2: Conversions FROM bf16 (bf16 → e4m3, bf16 → e5m2)
    else if (from_type == nk_bf16_k && (to_type == nk_e4m3_k || to_type == nk_e5m2_k)) {
        nk_bf16_t const *from_ptr = (nk_bf16_t const *)from;
        nk_e4m3_t *to_ptr = (nk_e4m3_t *)to;
        for (nk_size_t idx = 0; idx < n; idx += 32) {
            nk_size_t remaining = n - idx;
            __mmask32 mask = (remaining >= 32) ? 0xFFFFFFFF : _bzhi_u32(0xFFFFFFFF, (unsigned)remaining);
            __m512i in_bf16x32 = _mm512_maskz_loadu_epi16(mask, from_ptr + idx);
            __m256i out_f8x32 = (to_type == nk_e4m3_k) ? nk_bf16x32_to_e4m3x32_icelake_(in_bf16x32)
                                                       : nk_bf16x32_to_e5m2x32_icelake_(in_bf16x32);
            _mm256_mask_storeu_epi8(to_ptr + idx, mask, out_f8x32);
        }
    }

    // Group 3: Conversions to f16 (e4m3 → f16, e5m2 → f16)
    else if (to_type == nk_f16_k && (from_type == nk_e4m3_k || from_type == nk_e5m2_k)) {
        nk_e4m3_t const *from_ptr = (nk_e4m3_t const *)from;
        nk_f16_t *to_ptr = (nk_f16_t *)to;
        for (nk_size_t idx = 0; idx < n; idx += 32) {
            nk_size_t remaining = n - idx;
            __mmask32 mask = (remaining >= 32) ? 0xFFFFFFFF : _bzhi_u32(0xFFFFFFFF, (unsigned)remaining);
            __m256i in_f8x32 = _mm256_maskz_loadu_epi8(mask, from_ptr + idx);
            __m512i out_f16x32 = (from_type == nk_e4m3_k) ? nk_e4m3x32_to_f16x32_icelake_(in_f8x32)
                                                          : nk_e5m2x32_to_f16x32_icelake_(in_f8x32);
            _mm512_mask_storeu_epi16(to_ptr + idx, mask, out_f16x32);
        }
    }

    // Default: delegate to Skylake for all other conversions
    else nk_cast_skylake(from, from_type, n, to, to_type);
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

#endif // NK_TARGET_ICELAKE
#endif // NK_TARGET_X86_
#endif // NK_CAST_ICELAKE_H
