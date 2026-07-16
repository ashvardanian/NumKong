/**
 *  @brief SIMD-accelerated Type Conversions for Ice Lake.
 *  @file include/numkong/cast/icelake.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *
 *  @section ice_cast_instructions AVX-512 VBMI2 Instructions
 *
 *      Intrinsic                  Instruction               Icelake    Genoa
 *      _mm512_permutex2var_epi16  VPERMI2W (ZMM, ZMM, ZMM)  3cy @ p5   2cy @ p12
 *      _mm512_test_epi16_mask     VPTESTMW (k, ZMM, ZMM)    3cy @ p5   2cy @ p01
 *      _mm512_mask_mov_epi16      VMOVDQU16 (ZMM{k}, ZMM)   1cy @ p05  1cy @ p05
 *      _mm512_cvtepi16_epi8       VPMOVWB (YMM, ZMM)        3cy @ p5   2cy @ p12
 *
 *  Ice Lake's AVX-512 VBMI2 enables efficient 128-entry LUT lookups via dual VPERMI2W operations.
 *  FP8-to-BF16/F16 conversions use 4 ZMM LUT registers with VPTESTMW for range selection, achieving
 *  ~6 cycles for 32 FP8 conversions. E5M2-to-F16 simplifies to VPSLLW due to matching exponent bias.
 */
#ifndef NK_CAST_ICELAKE_H
#define NK_CAST_ICELAKE_H

#if NK_TARGET_X8664_
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

#pragma region Vectorized Conversions

/** @brief Convert 32x e4m3 → 32x bf16 via arithmetic + 8-entry subnormal LUT (AVX-512BW).
 *  E4M3 format: S EEEE MMM (bias=7). BF16: S EEEEEEEE MMMMMMM (bias=127).
 *  Normal values (exp != 0): BF16 = sign | ((lower7 << 4) + 0x3C00).
 *  Subnormals (exp == 0, 8 values): looked up from 8-entry LUT via permutexvar.
 *  Memory: 16 bytes (8 × 16-bit entries) vs 256 bytes (128-entry LUT). OCP FP8 v1.0. */
NK_HELPER_INLINE __m512i nk_e4m3x32_to_bf16x32_icelake_(__m256i e4m3x32) {
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
    __m512i mantissa_index_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16(0x07));
    __m512i subnorm_abs_i16x32 = _mm512_permutexvar_epi16(mantissa_index_i16x32, subn_lut_i16x32);

    // Blend: if exponent == 0, use subnormal; else use normal
    __m512i exponent_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16(0x78));
    __mmask32 is_subnormal_m32 = _mm512_cmpeq_epi16_mask(exponent_i16x32, _mm512_setzero_si512());
    __m512i result_abs_i16x32 = _mm512_mask_blend_epi16(is_subnormal_m32, normal_abs_i16x32, subnorm_abs_i16x32);

    // Apply sign: shift E4M3 bit 7 to BF16 bit 15
    sign_i16x32 = _mm512_slli_epi16(sign_i16x32, 8);
    __m512i result_i16x32 = _mm512_or_si512(result_abs_i16x32, sign_i16x32);

    // NaN: E4M3FN has NaN only at magnitude 0x7F → BF16 quiet NaN (0x7FC0)
    __mmask32 is_nan_m32 = _mm512_cmpeq_epi16_mask(lower7_i16x32, _mm512_set1_epi16(0x7F));
    __m512i nan_i16x32 = _mm512_or_si512(sign_i16x32, _mm512_set1_epi16(0x7FC0));
    return _mm512_mask_blend_epi16(is_nan_m32, result_i16x32, nan_i16x32);
}

/** @brief Convert 32x e5m2 → 32x bf16 via arithmetic + 4-entry subnormal LUT (AVX-512BW).
 *  E5M2 format: S EEEEE MM (bias=15). BF16: S EEEEEEEE MMMMMMM (bias=127).
 *  Normal values (exp != 0): BF16 = sign | ((lower7 << 5) + 0x3800).
 *  Subnormals (exp == 0, 4 values): looked up from 4-entry LUT via permutexvar.
 *  Memory: 8 bytes (4 × 16-bit entries) vs 256 bytes (128-entry LUT). OCP FP8 v1.0. */
NK_HELPER_INLINE __m512i nk_e5m2x32_to_bf16x32_icelake_(__m256i e5m2x32) {
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
    __m512i mantissa_index_i16x32 = _mm512_and_si512(e5m2_i16x32, _mm512_set1_epi16(0x03));
    __m512i subnorm_abs_i16x32 = _mm512_permutexvar_epi16(mantissa_index_i16x32, subn_lut_i16x32);

    // Blend: if exponent == 0, use subnormal; else use normal
    __m512i exponent_i16x32 = _mm512_and_si512(e5m2_i16x32, _mm512_set1_epi16(0x7C));
    __mmask32 is_subnormal_m32 = _mm512_cmpeq_epi16_mask(exponent_i16x32, _mm512_setzero_si512());
    __m512i result_abs_i16x32 = _mm512_mask_blend_epi16(is_subnormal_m32, normal_abs_i16x32, subnorm_abs_i16x32);

    // Apply sign: shift E5M2 bit 7 to BF16 bit 15
    sign_i16x32 = _mm512_slli_epi16(sign_i16x32, 8);
    return _mm512_or_si512(result_abs_i16x32, sign_i16x32);
}

/** @brief Convert 32x e2m3 → 32x bf16 via 32-entry LUT lookup (AVX-512BW).
 *  E2M3 format: S EE MMM (bias=1, 6 bits total: sign at bit 5, magnitude bits 4-0).
 *  BF16: S EEEEEEEE MMMMMMM (bias=127). Uses single permutexvar; sign handled separately.
 *  Subnormals (exp=0): value = mant/8. OCP Microscaling Formats v1.0. */
NK_HELPER_INLINE __m512i nk_e2m3x32_to_bf16x32_icelake_(__m256i e2m3x32) {
    __m512i e2m3_i16x32 = _mm512_cvtepu8_epi16(e2m3x32);
    __m512i sign_i16x32 = _mm512_and_si512(e2m3_i16x32, _mm512_set1_epi16(0x20)); // E2M3 sign at bit 5
    __m512i index_i16x32 = _mm512_and_si512(e2m3_i16x32, _mm512_set1_epi16(0x1F));

    // 32-entry LUT for E2M3 magnitude (5 bits: bits [4:3]=exp, bits [2:0]=mant)
    // E2M3: bias=1, range [0, 7.5] for positive, subnormals = mant/8 (OCP MX v1.0)
    // BF16 = (bf16_exp << 7) | (bf16_mant), where bf16_exp = e2m3_exp + 126, bf16_mant = e2m3_mant << 4
    __m512i const lut_i16x32 = _mm512_set_epi16(                         //
        0x40F0, 0x40E0, 0x40D0, 0x40C0, 0x40B0, 0x40A0, 0x4090, 0x4080,  // [31-24] exp=3: bf16_exp=129
        0x4070, 0x4060, 0x4050, 0x4040, 0x4030, 0x4020, 0x4010, 0x4000,  // [23-16] exp=2: bf16_exp=128
        0x3FF0, 0x3FE0, 0x3FD0, 0x3FC0, 0x3FB0, 0x3FA0, 0x3F90, 0x3F80,  // [15-8] exp=1: bf16_exp=127
        0x3F60, 0x3F40, 0x3F20, 0x3F00, 0x3EC0, 0x3E80, 0x3E00, 0x0000); // [7-0] exp=0: subnormals 7/8..1/8, 0

    // Single permutexvar for 32-entry lookup
    __m512i result_i16x32 = _mm512_permutexvar_epi16(index_i16x32, lut_i16x32);

    // Apply sign: shift E2M3 bit 5 to BF16 bit 15, then OR
    sign_i16x32 = _mm512_slli_epi16(sign_i16x32, 10);
    return _mm512_or_si512(result_i16x32, sign_i16x32);
}

/** @brief Convert 32x e3m2 → 32x bf16 via 32-entry LUT lookup (AVX-512BW).
 *  E3M2 format: S EEE MM (bias=3, 6 bits total: sign at bit 7, magnitude bits 4-0).
 *  BF16: S EEEEEEEE MMMMMMM (bias=127). Uses single permutexvar; sign handled separately. */
NK_HELPER_INLINE __m512i nk_e3m2x32_to_bf16x32_icelake_(__m256i e3m2x32) {
    __m512i e3m2_i16x32 = _mm512_cvtepu8_epi16(e3m2x32);
    __m512i sign_i16x32 = _mm512_and_si512(e3m2_i16x32, _mm512_set1_epi16(0x20)); // E3M2 sign at bit 5
    __m512i index_i16x32 = _mm512_and_si512(e3m2_i16x32, _mm512_set1_epi16(0x1F));

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
    __m512i result_i16x32 = _mm512_permutexvar_epi16(index_i16x32, lut_i16x32);

    // Apply sign: shift E3M2 bit 5 to BF16 bit 15, then OR
    sign_i16x32 = _mm512_slli_epi16(sign_i16x32, 10);
    return _mm512_or_si512(result_i16x32, sign_i16x32);
}

/** @brief Convert 32x e4m3 → 32x f16 via 128-entry LUT lookup (AVX-512BW).
 *  E4M3 format: S EEEE MMM (bias=7). F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Uses permutex2var for fast LUT lookup; sign handled separately via shift+OR.
 *  Handles all corner cases: zero, subnormals, normals, and NaN. */
NK_HELPER_INLINE __m512i nk_e4m3x32_to_f16x32_icelake_(__m256i e4m3x32) {
    __m512i e4m3_i16x32 = _mm512_cvtepu8_epi16(e4m3x32);
    __m512i sign_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16((short)0x80));
    __m512i index_i16x32 = _mm512_and_si512(e4m3_i16x32, _mm512_set1_epi16(0x7F));

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
    __m512i result_low_i16x32 = _mm512_permutex2var_epi16(lut0_i16x32, index_i16x32, lut1_i16x32);
    __m512i result_high_i16x32 = _mm512_permutex2var_epi16(lut2_i16x32, index_i16x32, lut3_i16x32);

    // Select between low (idx 0-63) and high (idx 64-127) based on bit 6
    __mmask32 use_high_m32 = _mm512_test_epi16_mask(index_i16x32, _mm512_set1_epi16(0x40));
    __m512i result_i16x32 = _mm512_mask_mov_epi16(result_low_i16x32, use_high_m32, result_high_i16x32);

    // Apply sign: shift sign bit to bit 15, then OR
    sign_i16x32 = _mm512_slli_epi16(sign_i16x32, 8);
    return _mm512_or_si512(result_i16x32, sign_i16x32);
}

/** @brief Convert 32x e5m2 → 32x f16 via simple bit shift (AVX-512BW).
 *  E5M2 format: S EEEEE MM (bias=15). F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Same exponent bias means F16 = (lower7 << 8) | (sign << 15).
 *  Handles all corner cases: zero, subnormals, normals, infinity, and NaN. */
NK_HELPER_INLINE __m512i nk_e5m2x32_to_f16x32_icelake_(__m256i e5m2x32) {
    __m512i e5m2_i16x32 = _mm512_cvtepu8_epi16(e5m2x32);
    __m512i sign_i16x32 = _mm512_and_si512(e5m2_i16x32, _mm512_set1_epi16((short)0x80));
    __m512i lower7_i16x32 = _mm512_and_si512(e5m2_i16x32, _mm512_set1_epi16(0x7F));

    // F16 = (lower7 << 8) | (sign << 15)
    // Works for all cases: subnormals, normals, infinity, and NaN
    __m512i result_i16x32 = _mm512_slli_epi16(lower7_i16x32, 8);
    sign_i16x32 = _mm512_slli_epi16(sign_i16x32, 8);
    return _mm512_or_si512(result_i16x32, sign_i16x32);
}

/** @brief Widen 32x bf16 → 64 bytes of 32x f32 by shifting each half-word into the high f32 bits.
 *  Exact for any value whose f32 round-trips through bf16 losslessly (true for all FP4/FP6 magnitudes,
 *  whose mantissas fit in BF16's 7 bits), so callers stay byte-identical to the serial f32 decode. */
NK_HELPER_INLINE void nk_bf16x32_to_f32x32_icelake_(__m512i bf16x32, __m512 *low_f32x16, __m512 *high_f32x16) {
    __m512i lo_i32x16 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(bf16x32)), 16);
    __m512i hi_i32x16 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(bf16x32, 1)), 16);
    *low_f32x16 = _mm512_castsi512_ps(lo_i32x16);
    *high_f32x16 = _mm512_castsi512_ps(hi_i32x16);
}

/** @brief Convert 32x e2m1 (16 packed bytes) → 32x f32 via a 16-entry BF16 LUT widened to f32.
 *  Input nibble order matches the serial codec: byte i holds element 2i in the high nibble and
 *  element 2i+1 in the low nibble. The LUT bakes both magnitude and sign into BF16 half-words
 *  {0,0.5,1,1.5,2,3,4,6} × {+,−}; widening to f32 is exact because every FP4 magnitude round-trips
 *  through BF16. Faster than Skylake's per-32-bit permute: one VPERMW covers all 32 elements. */
NK_HELPER_INLINE void nk_e2m1x32_to_f32x32_icelake_(__m128i packed, __m512 *low_f32x16, __m512 *high_f32x16) {
    // Expand 16 packed bytes to 32 nibble bytes via shift + mask + unpack interleave.
    __m128i low_nibbles_b8x16 = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
    __m128i high_nibbles_b8x16 = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0F));
    __m128i nibbles_low_b8x16 = _mm_unpacklo_epi8(high_nibbles_b8x16, low_nibbles_b8x16);  // elements 0..15
    __m128i nibbles_high_b8x16 = _mm_unpackhi_epi8(high_nibbles_b8x16, low_nibbles_b8x16); // elements 16..31
    __m256i nibbles_b8x32 = _mm256_set_m128i(nibbles_high_b8x16, nibbles_low_b8x16);
    __m512i nibble_index_i16x32 = _mm512_cvtepu8_epi16(nibbles_b8x32);

    // 16-entry BF16 LUT (bits 2..0 → magnitude, bit 3 → sign), replicated across both 256-bit halves
    // so a single VPERMW resolves all 32 lanes regardless of which half each index lands in.
    __m512i const lut_i16x32 = _mm512_set_epi16(                         //
        (short)0xC0C0, (short)0xC080, (short)0xC040, (short)0xC000,      // -6, -4, -3, -2
        (short)0xBFC0, (short)0xBF80, (short)0xBF00, (short)0x8000,      // -1.5, -1, -0.5, -0
        0x40C0, 0x4080, 0x4040, 0x4000, 0x3FC0, 0x3F80, 0x3F00, 0x0000,  // +6..+0
        (short)0xC0C0, (short)0xC080, (short)0xC040, (short)0xC000,      // (replicated half)
        (short)0xBFC0, (short)0xBF80, (short)0xBF00, (short)0x8000,      //
        0x40C0, 0x4080, 0x4040, 0x4000, 0x3FC0, 0x3F80, 0x3F00, 0x0000); //
    __m512i bf16_i16x32 = _mm512_permutexvar_epi16(nibble_index_i16x32, lut_i16x32);
    nk_bf16x32_to_f32x32_icelake_(bf16_i16x32, low_f32x16, high_f32x16);
}

/** @brief Compute 16x e2m1 nibbles (each in the low 4 bits of an i32 lane) from 16x f32 via the
 *  Skylake RNE bit-manipulation. Shared by the x32 packer; identical arithmetic to the Skylake codec. */
NK_HELPER_INLINE __m512i nk_f32x16_to_e2m1_nibbles_icelake_(__m512 f32x16) {
    __m512i bits_i32x16 = _mm512_castps_si512(f32x16);
    __m512i sign_i32x16 = _mm512_srli_epi32(bits_i32x16, 31);
    __m512i f32_exponent_i32x16 = _mm512_and_si512(_mm512_srli_epi32(bits_i32x16, 23), _mm512_set1_epi32(0xFF));

    // Normal path: round 23-bit mantissa to 1 bit using RNE (cut at bit 22).
    __m512i significand_i32x16 = _mm512_or_si512(_mm512_and_si512(bits_i32x16, _mm512_set1_epi32(0x007FFFFF)),
                                                 _mm512_set1_epi32(0x00800000));
    __m512i lsb_i32x16 = _mm512_and_si512(_mm512_srli_epi32(significand_i32x16, 22), _mm512_set1_epi32(1));
    __m512i rounding_bias_i32x16 = _mm512_add_epi32(_mm512_set1_epi32(0x001FFFFF), lsb_i32x16);
    __m512i rounded_sig_i32x16 = _mm512_add_epi32(significand_i32x16, rounding_bias_i32x16);
    __m512i carry_i32x16 = _mm512_srli_epi32(rounded_sig_i32x16, 24);
    __m512i normal_mantissa_i32x16 = _mm512_and_si512(_mm512_srli_epi32(rounded_sig_i32x16, 22),
                                                      _mm512_set1_epi32(0x01));
    __m512i e2m1_exponent_i32x16 = _mm512_sub_epi32(_mm512_add_epi32(f32_exponent_i32x16, carry_i32x16),
                                                    _mm512_set1_epi32(126));

    __mmask16 is_subnormal_m16 = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(1), e2m1_exponent_i32x16);
    __mmask16 overflow_m16 = _mm512_cmpgt_epi32_mask(e2m1_exponent_i32x16, _mm512_set1_epi32(3));

    __m512i clamped_exponent_i32x16 = _mm512_max_epi32(e2m1_exponent_i32x16, _mm512_set1_epi32(1));
    clamped_exponent_i32x16 = _mm512_min_epi32(clamped_exponent_i32x16, _mm512_set1_epi32(3));
    normal_mantissa_i32x16 = _mm512_mask_blend_epi32(overflow_m16, normal_mantissa_i32x16, _mm512_set1_epi32(0x01));
    __m512i normal_nibble_i32x16 = _mm512_ternarylogic_epi32(
        _mm512_slli_epi32(sign_i32x16, 3), _mm512_slli_epi32(clamped_exponent_i32x16, 1), normal_mantissa_i32x16, 0xFE);

    // Subnormal path: round(|x| * 2), clamp to {0, 1}; promote to first normal (0x02) when it rounds to 2.
    __m512 abs_f32x16 = _mm512_and_ps(f32x16, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 scaled_f32x16 = _mm512_mul_ps(abs_f32x16, _mm512_set1_ps(2.0f));
    __m512i subnorm_mantissa_i32x16 = _mm512_cvtps_epi32(scaled_f32x16);
    __mmask16 promotes_to_normal_m16 = _mm512_cmpgt_epi32_mask(subnorm_mantissa_i32x16, _mm512_set1_epi32(1));
    subnorm_mantissa_i32x16 = _mm512_max_epi32(_mm512_min_epi32(subnorm_mantissa_i32x16, _mm512_set1_epi32(1)),
                                               _mm512_setzero_si512());
    __m512i subnorm_nibble_i32x16 = _mm512_or_si512(_mm512_slli_epi32(sign_i32x16, 3), subnorm_mantissa_i32x16);
    __m512i first_normal_nibble_i32x16 = _mm512_or_si512(_mm512_slli_epi32(sign_i32x16, 3), _mm512_set1_epi32(0x02));
    subnorm_nibble_i32x16 = _mm512_mask_blend_epi32(promotes_to_normal_m16, subnorm_nibble_i32x16,
                                                    first_normal_nibble_i32x16);

    return _mm512_mask_blend_epi32(is_subnormal_m16, normal_nibble_i32x16, subnorm_nibble_i32x16);
}

/** @brief Convert 32x f32 → 32x e2m1 packed into 16 bytes via the Skylake RNE bit-manipulation
 *  widened to x32, then a byte pack. Lane ordering matches `nk_f32x2_to_e2m1x2_serial`:
 *  element 2i → high nibble of byte i, element 2i+1 → low nibble. Byte-identical to serial. */
NK_HELPER_INLINE __m128i nk_f32x32_to_e2m1x32_icelake_(__m512 low_f32x16, __m512 high_f32x16) {
    __m128i nibble_low_b8x16 = _mm512_cvtepi32_epi8(nk_f32x16_to_e2m1_nibbles_icelake_(low_f32x16));
    __m128i nibble_high_b8x16 = _mm512_cvtepi32_epi8(nk_f32x16_to_e2m1_nibbles_icelake_(high_f32x16));
    __m256i nibbles_b8x32 = _mm256_set_m128i(nibble_high_b8x16, nibble_low_b8x16);
    // Pack pairs: byte i = (elem[2i] << 4) | elem[2i+1] via maddubs with coefficients {0x10, 0x01}.
    __m256i pack_coeff_i16x16 = _mm256_set1_epi16(0x0110);
    __m256i packed_i16x16 = _mm256_maddubs_epi16(nibbles_b8x32, pack_coeff_i16x16);
    return _mm256_cvtepi16_epi8(packed_i16x16);
}

/** @brief Convert 32x e2m3 → 32x f32 by reusing the BF16 LUT decode then widening to f32.
 *  Exact: every E2M3 magnitude (≤ 3 mantissa bits) round-trips through BF16, so f32 = bf16 << 16. */
NK_HELPER_INLINE void nk_e2m3x32_to_f32x32_icelake_(__m256i e2m3x32, __m512 *low_f32x16, __m512 *high_f32x16) {
    nk_bf16x32_to_f32x32_icelake_(nk_e2m3x32_to_bf16x32_icelake_(e2m3x32), low_f32x16, high_f32x16);
}

/** @brief Convert 32x e3m2 → 32x f32 by reusing the BF16 LUT decode then widening to f32.
 *  Exact: every E3M2 magnitude (≤ 2 mantissa bits) round-trips through BF16, so f32 = bf16 << 16. */
NK_HELPER_INLINE void nk_e3m2x32_to_f32x32_icelake_(__m256i e3m2x32, __m512 *low_f32x16, __m512 *high_f32x16) {
    nk_bf16x32_to_f32x32_icelake_(nk_e3m2x32_to_bf16x32_icelake_(e3m2x32), low_f32x16, high_f32x16);
}

/** @brief Convert 32x bf16 → 32x e4m3 via bit manipulation (AVX-512BW).
 *  BF16: S EEEEEEEE MMMMMMM (bias=127). E4M3: S EEEE MMM (bias=7).
 *  Handles normal, subnormal, and overflow cases with RNE rounding. */
NK_HELPER_INLINE __m256i nk_bf16x32_to_e4m3x32_icelake_(__m512i bf16x32) {
    __m512i sign_i16x32 = _mm512_srli_epi16(bf16x32, 15);
    __m512i bf16_exponent_i16x32 = _mm512_and_si512(_mm512_srli_epi16(bf16x32, 7), _mm512_set1_epi16(0xFF));

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
    __m512i e4m3_exponent_i16x32 = _mm512_sub_epi16(_mm512_add_epi16(bf16_exponent_i16x32, carry_i16x32),
                                                    _mm512_set1_epi16(120));

    // Detect underflow (exp <= 0) and overflow (exp > 15)
    __mmask32 is_subnormal_m32 = _mm512_cmpgt_epi16_mask(_mm512_set1_epi16(1), e4m3_exponent_i16x32);
    __mmask32 overflow_m32 = _mm512_cmpgt_epi16_mask(e4m3_exponent_i16x32, _mm512_set1_epi16(15));

    // Normal path: clamp exp to [1,15]
    // e4m3FN quirk: exp=15 with mantissa=7 is NaN (0x7F), so clamp mantissa to 6 when exp=15.
    __m512i clamped_exponent_i16x32 = _mm512_max_epi16(e4m3_exponent_i16x32, _mm512_set1_epi16(1));
    clamped_exponent_i16x32 = _mm512_min_epi16(clamped_exponent_i16x32, _mm512_set1_epi16(15));
    __mmask32 is_max_exponent_m32 = _mm512_cmpeq_epi16_mask(clamped_exponent_i16x32, _mm512_set1_epi16(15));
    __m512i max_mantissa_i16x32 = _mm512_mask_blend_epi16(is_max_exponent_m32, _mm512_set1_epi16(7),
                                                          _mm512_set1_epi16(6));
    __m512i normal_mantissa_i16x32 = _mm512_min_epi16(bf16_mantissa_i16x32, max_mantissa_i16x32);
    normal_mantissa_i16x32 = _mm512_mask_blend_epi16(overflow_m32, normal_mantissa_i16x32, _mm512_set1_epi16(0x06));
    __m512i normal_e4m3_i16x32 = _mm512_or_si512(
        _mm512_slli_epi16(sign_i16x32, 7),
        _mm512_or_si512(_mm512_slli_epi16(clamped_exponent_i16x32, 3), normal_mantissa_i16x32));

    // Subnormal path: compute via f32 to get correct rounding
    // bf16 to f32 is just left shift by 16
    __m512i bf16_low_i32x16 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(bf16x32));
    __m512i bf16_high_i32x16 = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(bf16x32, 1));
    __m512 f32_low_f32x16 = _mm512_castsi512_ps(_mm512_slli_epi32(bf16_low_i32x16, 16));
    __m512 f32_high_f32x16 = _mm512_castsi512_ps(_mm512_slli_epi32(bf16_high_i32x16, 16));
    __m512 abs_f32_low_f32x16 = _mm512_and_ps(f32_low_f32x16, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 abs_f32_high_f32x16 = _mm512_and_ps(f32_high_f32x16, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 scaled_low_f32x16 = _mm512_mul_ps(abs_f32_low_f32x16, _mm512_set1_ps(512.0f));
    __m512 scaled_high_f32x16 = _mm512_mul_ps(abs_f32_high_f32x16, _mm512_set1_ps(512.0f));
    __m512i subnorm_mantissa_low_i32x16 = _mm512_cvtps_epi32(scaled_low_f32x16);
    __m512i subnorm_mantissa_high_i32x16 = _mm512_cvtps_epi32(scaled_high_f32x16);
    __m256i subnorm_mantissa_low_i16x16 = _mm512_cvtepi32_epi16(subnorm_mantissa_low_i32x16);
    __m256i subnorm_mantissa_high_i16x16 = _mm512_cvtepi32_epi16(subnorm_mantissa_high_i32x16);
    __m512i subnorm_mantissa_i16x32 = _mm512_inserti64x4(_mm512_castsi256_si512(subnorm_mantissa_low_i16x16),
                                                         subnorm_mantissa_high_i16x16, 1);
    __mmask32 promotes_to_normal_m32 = _mm512_cmpgt_epi16_mask(subnorm_mantissa_i16x32, _mm512_set1_epi16(7));
    subnorm_mantissa_i16x32 = _mm512_min_epi16(subnorm_mantissa_i16x32, _mm512_set1_epi16(7));
    subnorm_mantissa_i16x32 = _mm512_max_epi16(subnorm_mantissa_i16x32, _mm512_setzero_si512());
    __m512i subnorm_e4m3_i16x32 = _mm512_or_si512(_mm512_slli_epi16(sign_i16x32, 7), subnorm_mantissa_i16x32);
    __m512i first_normal_e4m3_i16x32 = _mm512_or_si512(_mm512_slli_epi16(sign_i16x32, 7), _mm512_set1_epi16(0x08));
    subnorm_e4m3_i16x32 = _mm512_mask_blend_epi16(promotes_to_normal_m32, subnorm_e4m3_i16x32,
                                                  first_normal_e4m3_i16x32);

    // Blend: use subnormal result when exp <= 0
    __m512i e4m3_i16x32 = _mm512_mask_blend_epi16(is_subnormal_m32, normal_e4m3_i16x32, subnorm_e4m3_i16x32);

    // Pack 32 i16s to 32 unsigned i8s via AVX-512BW
    return _mm512_cvtepi16_epi8(e4m3_i16x32);
}

/** @brief Convert 32x bf16 → 32x e5m2 via bit manipulation (AVX-512BW).
 *  BF16: S EEEEEEEE MMMMMMM (bias=127). E5M2: S EEEEE MM (bias=15).
 *  Handles normal, subnormal, and overflow cases with RNE rounding. */
NK_HELPER_INLINE __m256i nk_bf16x32_to_e5m2x32_icelake_(__m512i bf16x32) {
    __m512i sign_i16x32 = _mm512_srli_epi16(bf16x32, 15);
    __m512i bf16_exponent_i16x32 = _mm512_and_si512(_mm512_srli_epi16(bf16x32, 7), _mm512_set1_epi16(0xFF));

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
    __m512i e5m2_exponent_i16x32 = _mm512_sub_epi16(_mm512_add_epi16(bf16_exponent_i16x32, carry_i16x32),
                                                    _mm512_set1_epi16(112));

    // Detect subnormal (exp <= 0) and overflow (exp > 31)
    __mmask32 is_subnormal_m32 = _mm512_cmpgt_epi16_mask(_mm512_set1_epi16(1), e5m2_exponent_i16x32);
    __mmask32 overflow_m32 = _mm512_cmpgt_epi16_mask(e5m2_exponent_i16x32, _mm512_set1_epi16(31));

    // Normal path: clamp exp to [1,31], on overflow return infinity (exp=31, mantissa=0 = 0x7C)
    __m512i clamped_exponent_i16x32 = _mm512_max_epi16(e5m2_exponent_i16x32, _mm512_set1_epi16(1));
    clamped_exponent_i16x32 = _mm512_min_epi16(clamped_exponent_i16x32, _mm512_set1_epi16(31));
    __m512i normal_mantissa_i16x32 = _mm512_mask_blend_epi16(overflow_m32, bf16_mantissa_i16x32,
                                                             _mm512_setzero_si512());
    __m512i normal_e5m2_i16x32 = _mm512_or_si512(
        _mm512_slli_epi16(sign_i16x32, 7),
        _mm512_or_si512(_mm512_slli_epi16(clamped_exponent_i16x32, 2), normal_mantissa_i16x32));

    // Subnormal path: compute via f32 to get correct rounding
    __m512i bf16_low_i32x16 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(bf16x32));
    __m512i bf16_high_i32x16 = _mm512_cvtepu16_epi32(_mm512_extracti32x8_epi32(bf16x32, 1));
    __m512 f32_low_f32x16 = _mm512_castsi512_ps(_mm512_slli_epi32(bf16_low_i32x16, 16));
    __m512 f32_high_f32x16 = _mm512_castsi512_ps(_mm512_slli_epi32(bf16_high_i32x16, 16));
    __m512 abs_f32_low_f32x16 = _mm512_and_ps(f32_low_f32x16, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 abs_f32_high_f32x16 = _mm512_and_ps(f32_high_f32x16, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 scaled_low_f32x16 = _mm512_mul_ps(abs_f32_low_f32x16, _mm512_set1_ps(65536.0f));
    __m512 scaled_high_f32x16 = _mm512_mul_ps(abs_f32_high_f32x16, _mm512_set1_ps(65536.0f));
    __m512i subnorm_mantissa_low_i32x16 = _mm512_cvtps_epi32(scaled_low_f32x16);
    __m512i subnorm_mantissa_high_i32x16 = _mm512_cvtps_epi32(scaled_high_f32x16);
    __m256i subnorm_mantissa_low_i16x16 = _mm512_cvtepi32_epi16(subnorm_mantissa_low_i32x16);
    __m256i subnorm_mantissa_high_i16x16 = _mm512_cvtepi32_epi16(subnorm_mantissa_high_i32x16);
    __m512i subnorm_mantissa_i16x32 = _mm512_inserti64x4(_mm512_castsi256_si512(subnorm_mantissa_low_i16x16),
                                                         subnorm_mantissa_high_i16x16, 1);
    __mmask32 promotes_to_normal_m32 = _mm512_cmpgt_epi16_mask(subnorm_mantissa_i16x32, _mm512_set1_epi16(3));
    subnorm_mantissa_i16x32 = _mm512_min_epi16(subnorm_mantissa_i16x32, _mm512_set1_epi16(3));
    subnorm_mantissa_i16x32 = _mm512_max_epi16(subnorm_mantissa_i16x32, _mm512_setzero_si512());
    __m512i subnorm_e5m2_i16x32 = _mm512_or_si512(_mm512_slli_epi16(sign_i16x32, 7), subnorm_mantissa_i16x32);
    __m512i first_normal_e5m2_i16x32 = _mm512_or_si512(_mm512_slli_epi16(sign_i16x32, 7), _mm512_set1_epi16(0x04));
    subnorm_e5m2_i16x32 = _mm512_mask_blend_epi16(promotes_to_normal_m32, subnorm_e5m2_i16x32,
                                                  first_normal_e5m2_i16x32);

    // Blend: use subnormal result when exp <= 0
    __m512i e5m2_i16x32 = _mm512_mask_blend_epi16(is_subnormal_m32, normal_e5m2_i16x32, subnorm_e5m2_i16x32);

    // Pack 32 i16s to 32 unsigned i8s via AVX-512BW
    return _mm512_cvtepi16_epi8(e5m2_i16x32);
}

/** @brief Load 32x e4m3 from memory and convert to 32x bf16 (Ice Lake AVX-512BW). */
NK_HELPER_INLINE void nk_load_e4m3x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm = nk_e4m3x32_to_bf16x32_icelake_(_mm256_loadu_si256((__m256i const *)src));
}

/** @brief Partial load n e4m3 elements from memory and convert to bf16 (Ice Lake AVX-512BW). */
NK_HELPER_INLINE void nk_partial_load_e4m3x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    __mmask32 mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
    __m256i e4m3_partial_i8x32 = _mm256_maskz_loadu_epi8(mask_m32, src);
    dst->zmm = nk_e4m3x32_to_bf16x32_icelake_(e4m3_partial_i8x32);
}

/** @brief Load 32x e5m2 from memory and convert to 32x bf16 (Ice Lake AVX-512BW). */
NK_HELPER_INLINE void nk_load_e5m2x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm = nk_e5m2x32_to_bf16x32_icelake_(_mm256_loadu_si256((__m256i const *)src));
}

/** @brief Partial load n e5m2 elements from memory and convert to bf16 (Ice Lake AVX-512BW). */
NK_HELPER_INLINE void nk_partial_load_e5m2x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    __mmask32 mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
    __m256i e5m2_partial_i8x32 = _mm256_maskz_loadu_epi8(mask_m32, src);
    dst->zmm = nk_e5m2x32_to_bf16x32_icelake_(e5m2_partial_i8x32);
}

/** @brief Load 32x e2m3 from memory and convert to 32x bf16 (Ice Lake AVX-512BW). */
NK_HELPER_INLINE void nk_load_e2m3x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm = nk_e2m3x32_to_bf16x32_icelake_(_mm256_loadu_si256((__m256i const *)src));
}

/** @brief Partial load n e2m3 elements from memory and convert to bf16 (Ice Lake AVX-512BW). */
NK_HELPER_INLINE void nk_partial_load_e2m3x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    __mmask32 mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
    __m256i e2m3_partial_i8x32 = _mm256_maskz_loadu_epi8(mask_m32, src);
    dst->zmm = nk_e2m3x32_to_bf16x32_icelake_(e2m3_partial_i8x32);
}

/** @brief Load 32x e3m2 from memory and convert to 32x bf16 (Ice Lake AVX-512BW). */
NK_HELPER_INLINE void nk_load_e3m2x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm = nk_e3m2x32_to_bf16x32_icelake_(_mm256_loadu_si256((__m256i const *)src));
}

/** @brief Partial load n e3m2 elements from memory and convert to bf16 (Ice Lake AVX-512BW). */
NK_HELPER_INLINE void nk_partial_load_e3m2x32_to_bf16x32_icelake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    __mmask32 mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
    __m256i e3m2_partial_i8x32 = _mm256_maskz_loadu_epi8(mask_m32, src);
    dst->zmm = nk_e3m2x32_to_bf16x32_icelake_(e3m2_partial_i8x32);
}

#pragma endregion Vectorized Conversions

#pragma region Public API

NK_API_COMPTIME void nk_cast_icelake(void const *from, nk_dtype_t from_type, nk_size_t n, void *to,
                                     nk_dtype_t to_type) {
    // Group 1: Conversions to bf16 (e4m3 → bf16, e5m2 → bf16)
    if (to_type == nk_bf16_k && (from_type == nk_e4m3_k || from_type == nk_e5m2_k)) {
        nk_e4m3_t const *from_ptr = (nk_e4m3_t const *)from;
        nk_bf16_t *to_ptr = (nk_bf16_t *)to;
        for (nk_size_t i = 0; i < n; i += 32) {
            nk_size_t remaining = n - i;
            __mmask32 mask_m32 = (remaining >= 32) ? 0xFFFFFFFF : _bzhi_u32(0xFFFFFFFF, (unsigned)remaining);
            __m256i in_i8x32 = _mm256_maskz_loadu_epi8(mask_m32, from_ptr + i);
            __m512i out_bf16x32 = (from_type == nk_e4m3_k) ? nk_e4m3x32_to_bf16x32_icelake_(in_i8x32)
                                                           : nk_e5m2x32_to_bf16x32_icelake_(in_i8x32);
            _mm512_mask_storeu_epi16(to_ptr + i, mask_m32, out_bf16x32);
        }
    }

    // Group 2: Conversions FROM bf16 (bf16 → e4m3, bf16 → e5m2)
    else if (from_type == nk_bf16_k && (to_type == nk_e4m3_k || to_type == nk_e5m2_k)) {
        nk_bf16_t const *from_ptr = (nk_bf16_t const *)from;
        nk_e4m3_t *to_ptr = (nk_e4m3_t *)to;
        for (nk_size_t i = 0; i < n; i += 32) {
            nk_size_t remaining = n - i;
            __mmask32 mask_m32 = (remaining >= 32) ? 0xFFFFFFFF : _bzhi_u32(0xFFFFFFFF, (unsigned)remaining);
            __m512i in_bf16x32_i16x32 = _mm512_maskz_loadu_epi16(mask_m32, from_ptr + i);
            __m256i out_i8x32 = (to_type == nk_e4m3_k) ? nk_bf16x32_to_e4m3x32_icelake_(in_bf16x32_i16x32)
                                                       : nk_bf16x32_to_e5m2x32_icelake_(in_bf16x32_i16x32);
            _mm256_mask_storeu_epi8(to_ptr + i, mask_m32, out_i8x32);
        }
    }

    // Group 3: Conversions to f16 (e4m3 → f16, e5m2 → f16)
    else if (to_type == nk_f16_k && (from_type == nk_e4m3_k || from_type == nk_e5m2_k)) {
        nk_e4m3_t const *from_ptr = (nk_e4m3_t const *)from;
        nk_f16_t *to_ptr = (nk_f16_t *)to;
        for (nk_size_t i = 0; i < n; i += 32) {
            nk_size_t remaining = n - i;
            __mmask32 mask_m32 = (remaining >= 32) ? 0xFFFFFFFF : _bzhi_u32(0xFFFFFFFF, (unsigned)remaining);
            __m256i in_i8x32 = _mm256_maskz_loadu_epi8(mask_m32, from_ptr + i);
            __m512i out_f16x32 = (from_type == nk_e4m3_k) ? nk_e4m3x32_to_f16x32_icelake_(in_i8x32)
                                                          : nk_e5m2x32_to_f16x32_icelake_(in_i8x32);
            _mm512_mask_storeu_epi16(to_ptr + i, mask_m32, out_f16x32);
        }
    }

    // Group 4: E2M1 (FP4) ↔ f32. Sub-byte, so it rides a dedicated 32-wide loop instead of the
    // byte-stride hub; only the f32 peer is vectorised here (matches the block-scaled element codec).
    else if ((from_type == nk_e2m1_k && to_type == nk_f32_k) || (from_type == nk_f32_k && to_type == nk_e2m1_k)) {
        // 32 e2m1 = 16 bytes; f32 steps 128 bytes. Process whole groups of 32 with SIMD.
        nk_u8_t const *from_ptr = (nk_u8_t const *)from;
        nk_u8_t *to_ptr = (nk_u8_t *)to;
        nk_size_t from_step = nk_size_divide_round_up_(32 * nk_dtype_bits(from_type), NK_BITS_PER_BYTE);
        nk_size_t to_step = nk_size_divide_round_up_(32 * nk_dtype_bits(to_type), NK_BITS_PER_BYTE);
        nk_size_t batches = n / 32;
        for (nk_size_t i = 0; i < batches; ++i, from_ptr += from_step, to_ptr += to_step) {
            __m512 low_f32x16, high_f32x16;
            if (from_type == nk_e2m1_k) {
                nk_e2m1x32_to_f32x32_icelake_(_mm_loadu_si128((__m128i const *)from_ptr), &low_f32x16, &high_f32x16);
                _mm512_storeu_ps((float *)to_ptr, low_f32x16);
                _mm512_storeu_ps((float *)to_ptr + 16, high_f32x16);
            }
            else {
                low_f32x16 = _mm512_loadu_ps((float const *)from_ptr);
                high_f32x16 = _mm512_loadu_ps((float const *)from_ptr + 16);
                _mm_storeu_si128((__m128i *)to_ptr, nk_f32x32_to_e2m1x32_icelake_(low_f32x16, high_f32x16));
            }
        }
        // Tail (< 32): serial keeps packed-nibble writes byte-identical to the reference.
        nk_size_t tail = n % 32;
        if (tail) nk_cast_serial(from_ptr, from_type, tail, to_ptr, to_type);
    }

    // Group 5: E2M3 / E3M2 (FP6) → f32 via the BF16 LUT decode widened to f32 (one VPERMW per 32).
    else if (to_type == nk_f32_k && (from_type == nk_e2m3_k || from_type == nk_e3m2_k)) {
        nk_u8_t const *from_ptr = (nk_u8_t const *)from;
        nk_f32_t *to_ptr = (nk_f32_t *)to;
        nk_size_t i = 0;
        for (; i + 32 <= n; i += 32, from_ptr += 32, to_ptr += 32) {
            __m256i in_i8x32 = _mm256_loadu_si256((__m256i const *)from_ptr);
            __m512 low_f32x16, high_f32x16;
            if (from_type == nk_e2m3_k) nk_e2m3x32_to_f32x32_icelake_(in_i8x32, &low_f32x16, &high_f32x16);
            else nk_e3m2x32_to_f32x32_icelake_(in_i8x32, &low_f32x16, &high_f32x16);
            _mm512_storeu_ps(to_ptr, low_f32x16);
            _mm512_storeu_ps(to_ptr + 16, high_f32x16);
        }
        nk_size_t tail = n - i;
        if (tail) nk_cast_skylake(from_ptr, from_type, tail, to_ptr, to_type);
    }

    // Default: delegate to Skylake for all other conversions (FP8/integer codecs, f32→FP6, etc.)
    else nk_cast_skylake(from, from_type, n, to, to_type);
}

/** @brief Reduce a block of `block_count` f32s to `amax = max(|x|)`. `block_count` ≤ 32.
 *  Reuses the Skylake AVX-512 reduction (already 16-wide with NaN→sentinel propagation). */
NK_HELPER_INLINE nk_f32_t nk_block_amax_f32_icelake_(nk_f32_t const *block, nk_size_t block_count) {
    return nk_block_amax_f32_skylake_(block, block_count);
}

/** @brief IceLake block-scaled cast. Mirrors `nk_cast_block_scaled_skylake` but routes the element
 *  codec through `nk_cast_icelake`, whose 32-wide BF16-LUT decodes (FP4/FP6 → f32) replace Skylake's
 *  per-32-bit permutes; the f32 scale-derivation reuses the Skylake amax + reciprocal multiply. */
NK_API_COMPTIME void nk_cast_block_scaled_icelake(                                                             //
    void const *from, void const *from_scales, nk_scalar_buffer_t const *from_tensor_scale,                    //
    nk_block_scaled_format_t const *from_format,                                                               //
    void *to, void *to_scales, nk_scalar_buffer_t *to_tensor_scale, nk_block_scaled_format_t const *to_format, //
    nk_size_t count) {

    int from_plain = (from_format->scale_dtype == nk_dtype_unknown_k || from_format->block_size == 0);
    int to_plain = (to_format->scale_dtype == nk_dtype_unknown_k || to_format->block_size == 0);

    if (from_plain && to_plain) {
        nk_cast_icelake(from, from_format->element_dtype, count, to, to_format->element_dtype);
        return;
    }

    nk_size_t from_block = from_plain ? 1u : from_format->block_size;
    nk_size_t to_block = to_plain ? 1u : to_format->block_size;
    nk_size_t chunk = from_block > to_block ? from_block : to_block;

    nk_f32_t from_tensor_scale_f32 = 1.0f;
    if (from_tensor_scale != NK_NULL && !from_plain && from_format->tensor_scale_dtype == nk_f32_k)
        from_tensor_scale_f32 = from_tensor_scale->f32;

    nk_f32_t to_tensor_scale_f32 = 1.0f;
    int to_has_tensor_scale = (!to_plain && to_tensor_scale != NK_NULL && to_format->tensor_scale_dtype == nk_f32_k);
    if (to_has_tensor_scale) {
        to_tensor_scale_f32 = to_tensor_scale->f32;
        if (to_tensor_scale_f32 == 0.0f) {
            // Fall back to serial for auto-derive (needs a full tensor scan; rare calibration path).
            nk_cast_block_scaled_serial(from, from_scales, from_tensor_scale, from_format, to, to_scales,
                                        to_tensor_scale, to_format, count);
            return;
        }
    }

    nk_f32_t scratch[32];
    nk_size_t from_bits_per_element = nk_dtype_bits(from_format->element_dtype);
    nk_size_t to_bits_per_element = nk_dtype_bits(to_format->element_dtype);
    nk_u8_t const *from_scales_bytes = (nk_u8_t const *)from_scales;
    nk_u8_t *to_scales_bytes = (nk_u8_t *)to_scales;

    for (nk_size_t chunk_start = 0; chunk_start < count; chunk_start += chunk) {
        nk_size_t chunk_count = (chunk_start + chunk <= count) ? chunk : (count - chunk_start);

        // Decode source chunk into f32 scratch.
        if (from_plain) {
            void const *src = (nk_u8_t const *)from + (chunk_start * from_bits_per_element / NK_BITS_PER_BYTE);
            nk_cast_icelake(src, from_format->element_dtype, chunk_count, scratch, nk_f32_k);
        }
        else {
            for (nk_size_t b = 0; b < chunk_count; b += from_block) {
                nk_size_t valid = (chunk_count - b) < from_block ? (chunk_count - b) : from_block;
                nk_size_t block_idx = (chunk_start + b) / from_block;
                nk_u8_t raw = from_scales_bytes[block_idx];
                nk_f32_t scale_f32 = nk_block_scaled_decode_scale_serial_(raw, from_format->scale_dtype) *
                                     from_tensor_scale_f32;
                void const *src = (nk_u8_t const *)from +
                                  ((chunk_start + b) * from_bits_per_element / NK_BITS_PER_BYTE);
                nk_cast_icelake(src, from_format->element_dtype, valid, scratch + b, nk_f32_k);
                __m512 scale_bcast_f32x16 = _mm512_set1_ps(scale_f32);
                __m512 v_low_f32x16 = _mm512_maskz_loadu_ps(valid >= 16 ? 0xFFFF : (1u << valid) - 1u, scratch + b);
                _mm512_mask_storeu_ps(scratch + b, valid >= 16 ? 0xFFFF : (1u << valid) - 1u,
                                      _mm512_mul_ps(v_low_f32x16, scale_bcast_f32x16));
                if (valid > 16) {
                    __m512 v_high_f32x16 = _mm512_maskz_loadu_ps((1u << (valid - 16)) - 1u, scratch + b + 16);
                    _mm512_mask_storeu_ps(scratch + b + 16, (1u << (valid - 16)) - 1u,
                                          _mm512_mul_ps(v_high_f32x16, scale_bcast_f32x16));
                }
            }
        }

        // Encode f32 scratch into destination chunk.
        if (to_plain) {
            void *dst = (nk_u8_t *)to + (chunk_start * to_bits_per_element / NK_BITS_PER_BYTE);
            nk_cast_icelake(scratch, nk_f32_k, chunk_count, dst, to_format->element_dtype);
        }
        else {
            nk_f32_t element_max = nk_element_max_representable_(to_format->element_dtype);
            for (nk_size_t b = 0; b < chunk_count; b += to_block) {
                nk_size_t valid = (chunk_count - b) < to_block ? (chunk_count - b) : to_block;
                nk_f32_t block_amax = nk_block_amax_f32_icelake_(scratch + b, valid);
                nk_u8_t raw = nk_block_scaled_encode_scale_serial_(block_amax, element_max, to_tensor_scale_f32,
                                                                   to_format->scale_dtype);
                nk_size_t block_idx = (chunk_start + b) / to_block;
                to_scales_bytes[block_idx] = raw;
                nk_f32_t effective_scale = nk_block_scaled_decode_scale_serial_(raw, to_format->scale_dtype) *
                                           to_tensor_scale_f32;
                nk_f32_t reciprocal = effective_scale > 0 ? (1.0f / effective_scale) : 0.0f;
                __m512 reciprocal_bcast_f32x16 = _mm512_set1_ps(reciprocal);
                nk_f32_t encoded_scratch[32];
                __m512 v_low_f32x16 = _mm512_maskz_loadu_ps(valid >= 16 ? 0xFFFF : (1u << valid) - 1u, scratch + b);
                _mm512_mask_storeu_ps(encoded_scratch, valid >= 16 ? 0xFFFF : (1u << valid) - 1u,
                                      _mm512_mul_ps(v_low_f32x16, reciprocal_bcast_f32x16));
                if (valid > 16) {
                    __m512 v_high_f32x16 = _mm512_maskz_loadu_ps((1u << (valid - 16)) - 1u, scratch + b + 16);
                    _mm512_mask_storeu_ps(encoded_scratch + 16, (1u << (valid - 16)) - 1u,
                                          _mm512_mul_ps(v_high_f32x16, reciprocal_bcast_f32x16));
                }
                void *dst = (nk_u8_t *)to + ((chunk_start + b) * to_bits_per_element / NK_BITS_PER_BYTE);
                // Write only valid elements: dst is sized for `count` (bytes), not whole blocks.
                /* Saturate to element_max: finite inputs must not overflow to +/-inf (E5M2 has inf; OCP SAT). */
                for (nk_size_t saturate_index = 0; saturate_index < valid; ++saturate_index) {
                    if (encoded_scratch[saturate_index] > element_max) encoded_scratch[saturate_index] = element_max;
                    else if (encoded_scratch[saturate_index] < -element_max)
                        encoded_scratch[saturate_index] = -element_max;
                }
                nk_cast_icelake(encoded_scratch, nk_f32_k, valid, dst, to_format->element_dtype);
            }
        }
    }
}

#pragma endregion Public API

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_ICELAKE
#endif // NK_TARGET_X8664_
#endif // NK_CAST_ICELAKE_H
