/**
 *  @brief SIMD-accelerated Type Conversions for Haswell.
 *  @file include/numkong/cast/haswell.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *
 *  @section haswell_cast_instructions Key F16C/AVX2 Conversion Instructions
 *
 *      Intrinsic              Instruction                     Haswell     Genoa
 *      _mm256_cvtph_ps        VCVTPH2PS (YMM, XMM)            5cy @ p01   4cy @ p12+p23
 *      _mm256_cvtps_ph        VCVTPS2PH (XMM, YMM, I8)        5cy @ p01   4cy @ p12+p23
 *      _mm256_cvtepi16_epi32  VPMOVSXWD (YMM, XMM)            1cy @ p5    2cy @ p12
 *      _mm256_slli_epi32      VPSLLD (YMM, YMM, I8)           1cy @ p0    1cy @ p23
 *      _mm256_blendv_ps       VBLENDVPS (YMM, YMM, YMM, YMM)  2cy @ p015  1cy @ p01
 *
 *  F16C provides hardware F16<->F32 conversion. BF16 lacks hardware support and is emulated via
 *  bit manipulation (shift upper 16 bits). FP8 formats (E4M3/E5M2) use lookup tables for subnormal
 *  handling combined with arithmetic for normal values. All conversions hub through F32.
 */
#ifndef NK_CAST_HASWELL_H
#define NK_CAST_HASWELL_H

#if NK_TARGET_X8664_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_partial_load_b16x16_serial_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_f32_to_f16_haswell(nk_f32_t const *from, nk_f16_t *to) {
    *(nk_u16_t *)to = (nk_u16_t)_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(*from), _MM_FROUND_TO_NEAREST_INT));
}

NK_PUBLIC void nk_f16_to_f32_haswell(nk_f16_t const *from, nk_f32_t *to) {
    *to = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(*(nk_u16_t const *)from)));
}

#pragma region Type Punned Loads and Stores

/** @brief Type-agnostic 256-bit full load (Haswell AVX2). */
NK_INTERNAL void nk_load_b256_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm = _mm256_loadu_si256((const __m256i *)src);
}

/** @brief Type-agnostic 256-bit full store (Haswell AVX2). */
NK_INTERNAL void nk_store_b256_haswell_(nk_b256_vec_t const *src, void *dst) {
    _mm256_storeu_si256((__m256i *)dst, src->ymm);
}

/** @brief Type-agnostic 128-bit full load (Haswell AVX2). */
NK_INTERNAL void nk_load_b128_haswell_(void const *src, nk_b128_vec_t *dst) {
    dst->xmm = _mm_loadu_si128((const __m128i *)src);
}

/** @brief Type-agnostic 128-bit full store (SSE2). */
NK_INTERNAL void nk_store_b128_haswell_(nk_b128_vec_t const *src, void *dst) {
    _mm_storeu_si128((__m128i *)dst, src->xmm);
}

/** @brief Type-agnostic 128-bit partial load with AVX maskload. */
NK_INTERNAL void nk_partial_load_b32x4_haswell_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    __m128i idx_i32x4 = _mm_setr_epi32(0, 1, 2, 3);
    __m128i limit_i32x4 = _mm_set1_epi32((int)n);
    __m128i mask_i32x4 = _mm_cmpgt_epi32(limit_i32x4, idx_i32x4);
    dst->xmm = _mm_castps_si128(_mm_maskload_ps((float const *)src, mask_i32x4));
}

/** @brief Type-agnostic 128-bit partial store with AVX maskstore. */
NK_INTERNAL void nk_partial_store_b32x4_haswell_(nk_b128_vec_t const *src, void *dst, nk_size_t n) {
    __m128i idx_i32x4 = _mm_setr_epi32(0, 1, 2, 3);
    __m128i limit_i32x4 = _mm_set1_epi32((int)n);
    __m128i mask_i32x4 = _mm_cmpgt_epi32(limit_i32x4, idx_i32x4);
    _mm_maskstore_ps((float *)dst, mask_i32x4, _mm_castsi128_ps(src->xmm));
}

/** @brief Type-agnostic 256-bit partial load with AVX2 maskload. */
NK_INTERNAL void nk_partial_load_b64x4_haswell_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    __m256i idx_i64x4 = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i limit_i64x4 = _mm256_set1_epi64x((long long)n);
    __m256i mask_i64x4 = _mm256_cmpgt_epi64(limit_i64x4, idx_i64x4);
    dst->ymm = _mm256_castpd_si256(_mm256_maskload_pd((double const *)src, mask_i64x4));
}

/** @brief Type-agnostic 256-bit partial store with AVX2 maskstore. */
NK_INTERNAL void nk_partial_store_b64x4_haswell_(nk_b256_vec_t const *src, void *dst, nk_size_t n) {
    __m256i idx_i64x4 = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i limit_i64x4 = _mm256_set1_epi64x((long long)n);
    __m256i mask_i64x4 = _mm256_cmpgt_epi64(limit_i64x4, idx_i64x4);
    _mm256_maskstore_pd((double *)dst, mask_i64x4, _mm256_castsi256_pd(src->ymm));
}

#pragma endregion Type Punned Loads and Stores

#pragma region Vectorized Conversions

/** @brief Convert 8x bf16 → 8x f32 by shifting left 16 bits (AVX2). */
NK_INTERNAL __m256 nk_bf16x8_to_f32x8_haswell_(__m128i bf16_i16x8) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_i16x8), 16));
}

/** @brief Convert 8x f32 → 8x bf16 by truncating with RNE rounding (AVX2). */
NK_INTERNAL __m128i nk_f32x8_to_bf16x8_haswell_(__m256 f32x8) {
    __m256i bits_i32x8 = _mm256_castps_si256(f32x8);
    // RNE rounding: add (0x7FFF + lsb) where lsb is bit 16
    __m256i lsb_i32x8 = _mm256_and_si256(_mm256_srli_epi32(bits_i32x8, 16), _mm256_set1_epi32(1));
    __m256i rounded_i32x8 = _mm256_add_epi32(bits_i32x8, _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb_i32x8));
    __m256i bf16_i32x8 = _mm256_srli_epi32(rounded_i32x8, 16);
    // Pack 8x i32 to 8x i16
    __m128i low_i32x4 = _mm256_castsi256_si128(bf16_i32x8);
    __m128i high_i32x4 = _mm256_extracti128_si256(bf16_i32x8, 1);
    return _mm_packus_epi32(low_i32x4, high_i32x4);
}

/** @brief Integer upcasts to f32x8 (AVX2). */
NK_INTERNAL __m256 nk_i8x8_to_f32x8_haswell_(__m128i i8x8) { return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(i8x8)); }
NK_INTERNAL __m256 nk_u8x8_to_f32x8_haswell_(__m128i u8x8) { return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8x8)); }
NK_INTERNAL __m256 nk_i16x8_to_f32x8_haswell_(__m128i i16x8) {
    return _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(i16x8));
}
NK_INTERNAL __m256 nk_u16x8_to_f32x8_haswell_(__m128i u16x8) {
    return _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(u16x8));
}
NK_INTERNAL __m256 nk_i32x8_to_f32x8_haswell_(__m256i i32x8) { return _mm256_cvtepi32_ps(i32x8); }
NK_INTERNAL __m256 nk_u32x8_to_f32x8_haswell_(__m256i u32x8) {
    __m256i low_i32x8 = _mm256_and_si256(u32x8, _mm256_set1_epi32(0xFFFF));
    __m256i high_i32x8 = _mm256_srli_epi32(u32x8, 16);
    return _mm256_add_ps(_mm256_cvtepi32_ps(low_i32x8),
                         _mm256_mul_ps(_mm256_cvtepi32_ps(high_i32x8), _mm256_set1_ps(65536.0f)));
}

/** @brief Saturating f32x8 downcasts to integers (AVX2). */
NK_INTERNAL __m256i nk_f32x8_to_i32x8_haswell_(__m256 f32x8) { return _mm256_cvtps_epi32(f32x8); }
NK_INTERNAL __m256i nk_f32x8_to_u32x8_haswell_(__m256 f32x8) {
    __m256 clamped_f32x8 = _mm256_max_ps(_mm256_min_ps(f32x8, _mm256_set1_ps((float)NK_U32_MAX)), _mm256_setzero_ps());
    __m256 threshold_f32x8 = _mm256_set1_ps(2147483648.0f);
    __m256i mask_i32x8 = _mm256_castps_si256(_mm256_cmp_ps(clamped_f32x8, threshold_f32x8, _CMP_GE_OQ));
    __m256 adjusted_f32x8 = _mm256_sub_ps(clamped_f32x8,
                                          _mm256_and_ps(_mm256_castsi256_ps(mask_i32x8), threshold_f32x8));
    return _mm256_add_epi32(_mm256_cvtps_epi32(adjusted_f32x8),
                            _mm256_and_si256(mask_i32x8, _mm256_set1_epi32((int)0x80000000)));
}
NK_INTERNAL __m128i nk_f32x8_to_i16x8_haswell_(__m256 f32x8) {
    __m256 clamped_f32x8 = _mm256_min_ps(_mm256_max_ps(f32x8, _mm256_set1_ps(-32768.0f)), _mm256_set1_ps(32767.0f));
    __m256i i32x8 = _mm256_cvtps_epi32(clamped_f32x8);
    return _mm_packs_epi32(_mm256_castsi256_si128(i32x8), _mm256_extracti128_si256(i32x8, 1));
}
NK_INTERNAL __m128i nk_f32x8_to_u16x8_haswell_(__m256 f32x8) {
    __m256 clamped_f32x8 = _mm256_min_ps(_mm256_max_ps(f32x8, _mm256_setzero_ps()), _mm256_set1_ps(65535.0f));
    __m256i i32x8 = _mm256_cvtps_epi32(clamped_f32x8);
    return _mm_packus_epi32(_mm256_castsi256_si128(i32x8), _mm256_extracti128_si256(i32x8, 1));
}
NK_INTERNAL __m128i nk_f32x8_to_i8x8_haswell_(__m256 f32x8) {
    __m256 clamped_f32x8 = _mm256_min_ps(_mm256_max_ps(f32x8, _mm256_set1_ps(-128.0f)), _mm256_set1_ps(127.0f));
    __m256i i32x8 = _mm256_cvtps_epi32(clamped_f32x8);
    __m128i i16x8 = _mm_packs_epi32(_mm256_castsi256_si128(i32x8), _mm256_extracti128_si256(i32x8, 1));
    return _mm_packs_epi16(i16x8, _mm_setzero_si128());
}
NK_INTERNAL __m128i nk_f32x8_to_u8x8_haswell_(__m256 f32x8) {
    __m256 clamped_f32x8 = _mm256_min_ps(_mm256_max_ps(f32x8, _mm256_setzero_ps()), _mm256_set1_ps(255.0f));
    __m256i i32x8 = _mm256_cvtps_epi32(clamped_f32x8);
    __m128i u16x8 = _mm_packus_epi32(_mm256_castsi256_si128(i32x8), _mm256_extracti128_si256(i32x8, 1));
    return _mm_packus_epi16(u16x8, _mm_setzero_si128());
}

/** @brief Convert 8x e4m3 → 8x f32 via Giesen-style fake-F16 cast (AVX2 + F16C).
 *  E4M3 `byte = S EEEE MMM` (bias 7). Shifting the magnitude into F16 positions
 *  `((byte & 0x7F) << 7) | ((byte & 0x80) << 8)` yields a fake F16 whose F16 value
 *  differs from the true E4M3 magnitude by exactly 2⁸ (bias delta 15 − 7). The
 *  fake F16 is widened via `vcvtph2ps` and corrected by ×256 in F32. Subnormal
 *  handling falls out for free via F16 subnormal semantics. NaN (|byte|==0x7F)
 *  is blended explicitly with F16 quiet-NaN bits. */
NK_INTERNAL __m256 nk_e4m3x8_to_f32x8_haswell_(__m128i e4m3_i8x8) {
    __m128i const magnitude_mask_u16x8 = _mm_set1_epi16(0x7F);
    __m128i const sign_mask_u16x8 = _mm_set1_epi16((short)0x80);
    __m128i const f16_nan_u16x8 = _mm_set1_epi16(0x7E00);
    __m128i word_u16x8 = _mm_cvtepu8_epi16(e4m3_i8x8);
    __m128i magnitude_u16x8 = _mm_and_si128(word_u16x8, magnitude_mask_u16x8);
    __m128i is_nan_u16x8 = _mm_cmpeq_epi16(magnitude_u16x8, magnitude_mask_u16x8);
    __m128i shifted_magnitude_u16x8 = _mm_slli_epi16(magnitude_u16x8, 7);
    __m128i shifted_sign_u16x8 = _mm_slli_epi16(_mm_and_si128(word_u16x8, sign_mask_u16x8), 8);
    __m128i f16_bits_u16x8 = _mm_or_si128(shifted_magnitude_u16x8, shifted_sign_u16x8);
    f16_bits_u16x8 = _mm_blendv_epi8(f16_bits_u16x8, f16_nan_u16x8, is_nan_u16x8);
    __m256 fake_f32x8 = _mm256_cvtph_ps(f16_bits_u16x8);
    return _mm256_mul_ps(fake_f32x8, _mm256_set1_ps(256.0f));
}

/** @brief Convert 8x e5m2 → 8x f32 via free-shift widen (AVX2 + F16C).
 *  E5M2 shares F16's exponent bias (15): `(byte << 8)` is the matching F16 bit
 *  pattern for every E5M2 value (normals, subnormals, zero, ±Inf, NaN — all
 *  bit-exact). Widen u8 → u16, shift, then VCVTPH2PS to F32. Three ops total. */
NK_INTERNAL __m256 nk_e5m2x8_to_f32x8_haswell_(__m128i e5m2_i8x8) {
    __m128i e5m2_u16x8 = _mm_cvtepu8_epi16(e5m2_i8x8);
    __m128i f16_bits_u16x8 = _mm_slli_epi16(e5m2_u16x8, 8);
    return _mm256_cvtph_ps(f16_bits_u16x8);
}

/** @brief Convert 8x f32 → 8x e4m3 via bit manipulation (AVX2).
 *  E4M3 format: S EEEE MMM (bias=7). Handles normal, subnormal, and overflow cases.
 *  Subnormals (f32_exp ≤ 120): mantissa = round(abs_f32 * 512), clamped to [0,7]. */
NK_INTERNAL __m128i nk_f32x8_to_e4m3x8_haswell_(__m256 f32x8) {
    __m256i bits_i32x8 = _mm256_castps_si256(f32x8);
    __m256i sign_i32x8 = _mm256_srli_epi32(bits_i32x8, 31);
    __m256i f32_exp_i32x8 = _mm256_and_si256(_mm256_srli_epi32(bits_i32x8, 23), _mm256_set1_epi32(0xFF));

    // Round mantissa from 23 to 3 bits using RNE (round to nearest, ties to even)
    // RNE trick: add (half - 1 + lsb) where lsb is the bit that will become the new lsb after shift
    __m256i significand_i32x8 = _mm256_or_si256(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x007FFFFF)),
                                                _mm256_set1_epi32(0x00800000)); // Add implicit 1 bit
    __m256i lsb_i32x8 = _mm256_and_si256(_mm256_srli_epi32(significand_i32x8, 20), _mm256_set1_epi32(1));
    __m256i rounding_bias_i32x8 = _mm256_add_epi32(_mm256_set1_epi32(0x0007FFFF), lsb_i32x8);
    __m256i rounded_sig_i32x8 = _mm256_add_epi32(significand_i32x8, rounding_bias_i32x8);
    __m256i carry_i32x8 = _mm256_srli_epi32(rounded_sig_i32x8, 24); // Carry into exponent if bit 24 set
    __m256i f32_mantissa_i32x8 = _mm256_and_si256(_mm256_srli_epi32(rounded_sig_i32x8, 20), _mm256_set1_epi32(0x07));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    f32_mantissa_i32x8 = _mm256_andnot_si256(_mm256_slli_epi32(carry_i32x8, 31), f32_mantissa_i32x8);
    __m256i e4m3_exp_i32x8 = _mm256_sub_epi32(_mm256_add_epi32(f32_exp_i32x8, carry_i32x8), _mm256_set1_epi32(120));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 15)
    __m256i is_subnormal_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e4m3_exp_i32x8);
    __m256i overflow_i32x8 = _mm256_cmpgt_epi32(e4m3_exp_i32x8, _mm256_set1_epi32(15));

    // Normal path: clamp exp to [1,15], extract mantissa bits
    // e4m3FN quirk: exp=15 with mantissa=7 is NaN (0x7F), so clamp mantissa to 6 when exp=15.
    __m256i clamped_exp_i32x8 = _mm256_max_epi32(e4m3_exp_i32x8, _mm256_set1_epi32(1));
    clamped_exp_i32x8 = _mm256_min_epi32(clamped_exp_i32x8, _mm256_set1_epi32(15));
    __m256i is_max_exp_i32x8 = _mm256_cmpeq_epi32(clamped_exp_i32x8, _mm256_set1_epi32(15));
    __m256i max_mantissa_i32x8 = _mm256_blendv_epi8(_mm256_set1_epi32(7), _mm256_set1_epi32(6), is_max_exp_i32x8);
    __m256i normal_mantissa_i32x8 = _mm256_min_epi32(f32_mantissa_i32x8, max_mantissa_i32x8);
    normal_mantissa_i32x8 = _mm256_blendv_epi8(normal_mantissa_i32x8, _mm256_set1_epi32(0x06), overflow_i32x8);
    __m256i normal_e4m3_i32x8 = _mm256_or_si256(
        _mm256_slli_epi32(sign_i32x8, 7),
        _mm256_or_si256(_mm256_slli_epi32(clamped_exp_i32x8, 3), normal_mantissa_i32x8));

    // Subnormal path: mantissa = round(abs_f32 * 512)
    // If mantissa rounds to 8 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x08
    __m256 abs_f32x8 = _mm256_and_ps(f32x8, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    __m256 scaled_f32x8 = _mm256_mul_ps(abs_f32x8, _mm256_set1_ps(512.0f));
    __m256i subnorm_mantissa_i32x8 = _mm256_cvtps_epi32(scaled_f32x8);
    __m256i promotes_to_normal_i32x8 = _mm256_cmpgt_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(7));
    subnorm_mantissa_i32x8 = _mm256_min_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(7));
    subnorm_mantissa_i32x8 = _mm256_max_epi32(subnorm_mantissa_i32x8, _mm256_setzero_si256());
    __m256i subnorm_e4m3_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 7), subnorm_mantissa_i32x8);
    // When mantissa rounds to 8, use first normal value (0x08) instead of clamped subnormal
    __m256i first_normal_e4m3_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 7), _mm256_set1_epi32(0x08));
    subnorm_e4m3_i32x8 = _mm256_blendv_epi8(subnorm_e4m3_i32x8, first_normal_e4m3_i32x8, promotes_to_normal_i32x8);

    // Blend: use subnormal result when exp <= 0, else normal
    __m256i e4m3_i32x8 = _mm256_blendv_epi8(normal_e4m3_i32x8, subnorm_e4m3_i32x8, is_subnormal_i32x8);

    // Pack 8 i32s to 8 unsigned i8s (use unsigned saturation to preserve values 128-255)
    __m128i low_i32x4 = _mm256_castsi256_si128(e4m3_i32x8);
    __m128i high_i32x4 = _mm256_extracti128_si256(e4m3_i32x8, 1);
    __m128i packed_i16x8 = _mm_packus_epi32(low_i32x4, high_i32x4);
    __m128i packed_i8x8 = _mm_packus_epi16(packed_i16x8, packed_i16x8);
    return packed_i8x8;
}

/** @brief Convert 8x f32 → 8x e5m2 via bit manipulation (AVX2).
 *  E5M2 format: S EEEEE MM (bias=15). Handles normal, subnormal, and overflow cases.
 *  Uses RNE (round to nearest even) for mantissa rounding. */
NK_INTERNAL __m128i nk_f32x8_to_e5m2x8_haswell_(__m256 f32x8) {
    __m256i bits_i32x8 = _mm256_castps_si256(f32x8);
    __m256i sign_i32x8 = _mm256_srli_epi32(bits_i32x8, 31);
    __m256i f32_exp_i32x8 = _mm256_and_si256(_mm256_srli_epi32(bits_i32x8, 23), _mm256_set1_epi32(0xFF));

    // Round mantissa from 23 to 2 bits using RNE (round to nearest, ties to even)
    // RNE trick: add (half - 1 + lsb) where lsb is the bit that will become the new lsb after shift
    __m256i significand_i32x8 = _mm256_or_si256(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x007FFFFF)),
                                                _mm256_set1_epi32(0x00800000)); // Add implicit 1 bit
    __m256i lsb_i32x8 = _mm256_and_si256(_mm256_srli_epi32(significand_i32x8, 21), _mm256_set1_epi32(1));
    __m256i rounding_bias_i32x8 = _mm256_add_epi32(_mm256_set1_epi32(0x000FFFFF), lsb_i32x8); // half = 0x100000
    __m256i rounded_sig_i32x8 = _mm256_add_epi32(significand_i32x8, rounding_bias_i32x8);
    __m256i carry_i32x8 = _mm256_srli_epi32(rounded_sig_i32x8, 24); // Carry into exponent if bit 24 set
    __m256i f32_mantissa_i32x8 = _mm256_and_si256(_mm256_srli_epi32(rounded_sig_i32x8, 21), _mm256_set1_epi32(0x03));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    f32_mantissa_i32x8 = _mm256_andnot_si256(_mm256_slli_epi32(carry_i32x8, 31), f32_mantissa_i32x8);
    __m256i e5m2_exp_i32x8 = _mm256_sub_epi32(_mm256_add_epi32(f32_exp_i32x8, carry_i32x8), _mm256_set1_epi32(112));

    // Detect subnormal (exp <= 0) and overflow (exp > 31)
    __m256i is_subnormal_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e5m2_exp_i32x8);
    __m256i overflow_i32x8 = _mm256_cmpgt_epi32(e5m2_exp_i32x8, _mm256_set1_epi32(31));

    // Normal path: clamp exp to [1,31], on overflow return infinity (exp=31, mantissa=0 = 0x7C)
    __m256i clamped_exp_i32x8 = _mm256_max_epi32(e5m2_exp_i32x8, _mm256_set1_epi32(1));
    clamped_exp_i32x8 = _mm256_min_epi32(clamped_exp_i32x8, _mm256_set1_epi32(31));
    __m256i normal_mantissa_i32x8 = _mm256_blendv_epi8(f32_mantissa_i32x8, _mm256_setzero_si256(), overflow_i32x8);
    __m256i normal_e5m2_i32x8 = _mm256_or_si256(
        _mm256_slli_epi32(sign_i32x8, 7),
        _mm256_or_si256(_mm256_slli_epi32(clamped_exp_i32x8, 2), normal_mantissa_i32x8));

    // Subnormal path: mantissa = round(abs_f32 * 65536)
    // If mantissa rounds to 4 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x04
    __m256 abs_f32x8 = _mm256_and_ps(f32x8, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    __m256 scaled_f32x8 = _mm256_mul_ps(abs_f32x8, _mm256_set1_ps(65536.0f));
    __m256i subnorm_mantissa_i32x8 = _mm256_cvtps_epi32(scaled_f32x8);
    __m256i promotes_to_normal_i32x8 = _mm256_cmpgt_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(3));
    subnorm_mantissa_i32x8 = _mm256_min_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(3));
    subnorm_mantissa_i32x8 = _mm256_max_epi32(subnorm_mantissa_i32x8, _mm256_setzero_si256());
    __m256i subnorm_e5m2_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 7), subnorm_mantissa_i32x8);
    // When mantissa rounds to 4, use first normal value (0x04) instead of clamped subnormal
    __m256i first_normal_e5m2_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 7), _mm256_set1_epi32(0x04));
    subnorm_e5m2_i32x8 = _mm256_blendv_epi8(subnorm_e5m2_i32x8, first_normal_e5m2_i32x8, promotes_to_normal_i32x8);

    // Blend: use subnormal result when exp <= 0
    __m256i e5m2_i32x8 = _mm256_blendv_epi8(normal_e5m2_i32x8, subnorm_e5m2_i32x8, is_subnormal_i32x8);

    // Pack 8 i32s to 8 unsigned i8s (use unsigned saturation to preserve values 128-255)
    __m128i low_i32x4 = _mm256_castsi256_si128(e5m2_i32x8);
    __m128i high_i32x4 = _mm256_extracti128_si256(e5m2_i32x8, 1);
    __m128i packed_i16x8 = _mm_packus_epi32(low_i32x4, high_i32x4);
    __m128i packed_i8x8 = _mm_packus_epi16(packed_i16x8, packed_i16x8);
    return packed_i8x8;
}

/** @brief Convert 8x e2m3 → 8x f32 via bit manipulation (AVX2).
 *  E2M3 format: S EE MMM (bias=1). F32: sign<<31, (exp+126)<<23, mantissa<<20.
 *  Subnormals (exp=0): value = mantissa × 2⁽¹⁻¹⁾ × 2⁻³ = mantissa ÷ 8. */
NK_INTERNAL __m256 nk_e2m3x8_to_f32x8_haswell_(__m128i e2m3_i8x8) {
    __m256i e2m3_i32x8 = _mm256_cvtepu8_epi32(e2m3_i8x8);

    // Extract fields (only 6 bits used: S EE MMM)
    __m256i exp_i32x8 = _mm256_and_si256(_mm256_srli_epi32(e2m3_i32x8, 3), _mm256_set1_epi32(0x03));
    __m256i mant_i32x8 = _mm256_and_si256(e2m3_i32x8, _mm256_set1_epi32(0x07));

    // Build F32 sign bit
    __m256i f32_sign_i32x8 = _mm256_slli_epi32(_mm256_srli_epi32(e2m3_i32x8, 5), 31);

    // Normal path: sign | ((exp+126)<<23) | (mant<<20)
    __m256i f32_exp_i32x8 = _mm256_slli_epi32(_mm256_add_epi32(exp_i32x8, _mm256_set1_epi32(126)), 23);
    __m256i f32_mant_i32x8 = _mm256_slli_epi32(mant_i32x8, 20);
    __m256i normal_bits_i32x8 = _mm256_or_si256(f32_sign_i32x8, _mm256_or_si256(f32_exp_i32x8, f32_mant_i32x8));

    // Subnormal path: value = mantissa / 8.0f, then apply sign
    __m256 subnorm_abs_f32x8 = _mm256_mul_ps(_mm256_cvtepi32_ps(mant_i32x8), _mm256_set1_ps(1.0f / 8.0f));
    __m256 subnorm_f32x8 = _mm256_or_ps(subnorm_abs_f32x8, _mm256_castsi256_ps(f32_sign_i32x8));

    // Blend: if exp==0, use subnormal result; otherwise use normal bits
    __m256i exp_zero_mask = _mm256_cmpeq_epi32(exp_i32x8, _mm256_setzero_si256());
    return _mm256_blendv_ps(_mm256_castsi256_ps(normal_bits_i32x8), subnorm_f32x8, _mm256_castsi256_ps(exp_zero_mask));
}

/** @brief Convert 8x e3m2 → 8x f32 via bit manipulation (AVX2).
 *  E3M2 format: S EEE MM (bias=3). F32: sign<<31, (exp+124)<<23, mantissa<<21.
 *  Subnormals (exp=0): value = mantissa × 2⁽¹⁻³⁾ × 2⁻² = mantissa ÷ 16. */
NK_INTERNAL __m256 nk_e3m2x8_to_f32x8_haswell_(__m128i e3m2_i8x8) {
    __m256i e3m2_i32x8 = _mm256_cvtepu8_epi32(e3m2_i8x8);

    // Extract fields (only 6 bits used: S EEE MM)
    __m256i exp_i32x8 = _mm256_and_si256(_mm256_srli_epi32(e3m2_i32x8, 2), _mm256_set1_epi32(0x07));
    __m256i mant_i32x8 = _mm256_and_si256(e3m2_i32x8, _mm256_set1_epi32(0x03));

    // Build F32 sign bit
    __m256i f32_sign_i32x8 = _mm256_slli_epi32(_mm256_srli_epi32(e3m2_i32x8, 5), 31);

    // Normal path: sign | ((exp+124)<<23) | (mant<<21)
    __m256i f32_exp_i32x8 = _mm256_slli_epi32(_mm256_add_epi32(exp_i32x8, _mm256_set1_epi32(124)), 23);
    __m256i f32_mant_i32x8 = _mm256_slli_epi32(mant_i32x8, 21);
    __m256i normal_bits_i32x8 = _mm256_or_si256(f32_sign_i32x8, _mm256_or_si256(f32_exp_i32x8, f32_mant_i32x8));

    // Subnormal path: value = mantissa / 16.0f, then apply sign
    __m256 subnorm_abs_f32x8 = _mm256_mul_ps(_mm256_cvtepi32_ps(mant_i32x8), _mm256_set1_ps(1.0f / 16.0f));
    __m256 subnorm_f32x8 = _mm256_or_ps(subnorm_abs_f32x8, _mm256_castsi256_ps(f32_sign_i32x8));

    // Blend: if exp==0, use subnormal result; otherwise use normal bits
    __m256i exp_zero_mask = _mm256_cmpeq_epi32(exp_i32x8, _mm256_setzero_si256());
    return _mm256_blendv_ps(_mm256_castsi256_ps(normal_bits_i32x8), subnorm_f32x8, _mm256_castsi256_ps(exp_zero_mask));
}

/** @brief Convert 8x f32 → 8x e2m3 via bit manipulation (AVX2).
 *  E2M3 format: S EE MMM (bias=1). Handles normal, subnormal, and overflow cases.
 *  Subnormals (f32_exp ≤ 126): mantissa = round(abs_f32 * 8), clamped to [0,7]. */
NK_INTERNAL __m128i nk_f32x8_to_e2m3x8_haswell_(__m256 f32x8) {
    __m256i bits_i32x8 = _mm256_castps_si256(f32x8);
    __m256i sign_i32x8 = _mm256_srli_epi32(bits_i32x8, 31);
    __m256i f32_exp_i32x8 = _mm256_and_si256(_mm256_srli_epi32(bits_i32x8, 23), _mm256_set1_epi32(0xFF));

    // Round mantissa from 23 to 3 bits using RNE (round to nearest, ties to even)
    __m256i significand_i32x8 = _mm256_or_si256(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x007FFFFF)),
                                                _mm256_set1_epi32(0x00800000)); // Add implicit 1 bit
    __m256i lsb_i32x8 = _mm256_and_si256(_mm256_srli_epi32(significand_i32x8, 20), _mm256_set1_epi32(1));
    __m256i rounding_bias_i32x8 = _mm256_add_epi32(_mm256_set1_epi32(0x0007FFFF), lsb_i32x8);
    __m256i rounded_sig_i32x8 = _mm256_add_epi32(significand_i32x8, rounding_bias_i32x8);
    __m256i carry_i32x8 = _mm256_srli_epi32(rounded_sig_i32x8, 24); // Carry into exponent if bit 24 set
    __m256i f32_mantissa_i32x8 = _mm256_and_si256(_mm256_srli_epi32(rounded_sig_i32x8, 20), _mm256_set1_epi32(0x07));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    f32_mantissa_i32x8 = _mm256_andnot_si256(_mm256_slli_epi32(carry_i32x8, 31), f32_mantissa_i32x8);
    __m256i e2m3_exp_i32x8 = _mm256_sub_epi32(_mm256_add_epi32(f32_exp_i32x8, carry_i32x8), _mm256_set1_epi32(126));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 3)
    __m256i is_subnormal_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e2m3_exp_i32x8);
    __m256i overflow_i32x8 = _mm256_cmpgt_epi32(e2m3_exp_i32x8, _mm256_set1_epi32(3));

    // Normal path: clamp exp to [1,3], extract mantissa bits
    __m256i clamped_exp_i32x8 = _mm256_max_epi32(e2m3_exp_i32x8, _mm256_set1_epi32(1));
    clamped_exp_i32x8 = _mm256_min_epi32(clamped_exp_i32x8, _mm256_set1_epi32(3));
    __m256i normal_mantissa_i32x8 = _mm256_blendv_epi8(f32_mantissa_i32x8, _mm256_set1_epi32(0x07), overflow_i32x8);
    __m256i normal_e2m3_i32x8 = _mm256_or_si256(
        _mm256_slli_epi32(sign_i32x8, 5),
        _mm256_or_si256(_mm256_slli_epi32(clamped_exp_i32x8, 3), normal_mantissa_i32x8));

    // Subnormal path: mantissa = round(abs_f32 * 8)
    // If mantissa rounds to 8 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x08
    __m256 abs_f32x8 = _mm256_and_ps(f32x8, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    __m256 scaled_f32x8 = _mm256_mul_ps(abs_f32x8, _mm256_set1_ps(8.0f));
    __m256i subnorm_mantissa_i32x8 = _mm256_cvtps_epi32(scaled_f32x8);
    __m256i promotes_to_normal_i32x8 = _mm256_cmpgt_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(7));
    subnorm_mantissa_i32x8 = _mm256_min_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(7));
    subnorm_mantissa_i32x8 = _mm256_max_epi32(subnorm_mantissa_i32x8, _mm256_setzero_si256());
    __m256i subnorm_e2m3_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 5), subnorm_mantissa_i32x8);
    // When mantissa rounds to 8, use first normal value (0x08) instead of clamped subnormal
    __m256i first_normal_e2m3_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 5), _mm256_set1_epi32(0x08));
    subnorm_e2m3_i32x8 = _mm256_blendv_epi8(subnorm_e2m3_i32x8, first_normal_e2m3_i32x8, promotes_to_normal_i32x8);

    // Blend: use subnormal result when exp <= 0, else normal
    __m256i e2m3_i32x8 = _mm256_blendv_epi8(normal_e2m3_i32x8, subnorm_e2m3_i32x8, is_subnormal_i32x8);

    // Pack 8 i32s to 8 unsigned i8s (use unsigned saturation to preserve values 128-255)
    __m128i low_i32x4 = _mm256_castsi256_si128(e2m3_i32x8);
    __m128i high_i32x4 = _mm256_extracti128_si256(e2m3_i32x8, 1);
    __m128i packed_i16x8 = _mm_packus_epi32(low_i32x4, high_i32x4);
    __m128i packed_i8x8 = _mm_packus_epi16(packed_i16x8, packed_i16x8);
    return packed_i8x8;
}

/** @brief Convert 8x f32 → 8x e3m2 via bit manipulation (AVX2).
 *  E3M2 format: S EEE MM (bias=3). Handles normal, subnormal, and overflow cases.
 *  Subnormals (f32_exp ≤ 124): mantissa = round(abs_f32 * 16), clamped to [0,3]. */
NK_INTERNAL __m128i nk_f32x8_to_e3m2x8_haswell_(__m256 f32x8) {
    __m256i bits_i32x8 = _mm256_castps_si256(f32x8);
    __m256i sign_i32x8 = _mm256_srli_epi32(bits_i32x8, 31);
    __m256i f32_exp_i32x8 = _mm256_and_si256(_mm256_srli_epi32(bits_i32x8, 23), _mm256_set1_epi32(0xFF));

    // Round mantissa from 23 to 2 bits using RNE (round to nearest, ties to even)
    __m256i significand_i32x8 = _mm256_or_si256(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x007FFFFF)),
                                                _mm256_set1_epi32(0x00800000)); // Add implicit 1 bit
    __m256i lsb_i32x8 = _mm256_and_si256(_mm256_srli_epi32(significand_i32x8, 21), _mm256_set1_epi32(1));
    __m256i rounding_bias_i32x8 = _mm256_add_epi32(_mm256_set1_epi32(0x000FFFFF), lsb_i32x8);
    __m256i rounded_sig_i32x8 = _mm256_add_epi32(significand_i32x8, rounding_bias_i32x8);
    __m256i carry_i32x8 = _mm256_srli_epi32(rounded_sig_i32x8, 24); // Carry into exponent if bit 24 set
    __m256i f32_mantissa_i32x8 = _mm256_and_si256(_mm256_srli_epi32(rounded_sig_i32x8, 21), _mm256_set1_epi32(0x03));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    f32_mantissa_i32x8 = _mm256_andnot_si256(_mm256_slli_epi32(carry_i32x8, 31), f32_mantissa_i32x8);
    __m256i e3m2_exp_i32x8 = _mm256_sub_epi32(_mm256_add_epi32(f32_exp_i32x8, carry_i32x8), _mm256_set1_epi32(124));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 7)
    __m256i is_subnormal_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e3m2_exp_i32x8);
    __m256i overflow_i32x8 = _mm256_cmpgt_epi32(e3m2_exp_i32x8, _mm256_set1_epi32(7));

    // Normal path: clamp exp to [1,7], extract mantissa bits
    __m256i clamped_exp_i32x8 = _mm256_max_epi32(e3m2_exp_i32x8, _mm256_set1_epi32(1));
    clamped_exp_i32x8 = _mm256_min_epi32(clamped_exp_i32x8, _mm256_set1_epi32(7));
    __m256i normal_mantissa_i32x8 = _mm256_blendv_epi8(f32_mantissa_i32x8, _mm256_set1_epi32(0x03), overflow_i32x8);
    __m256i normal_e3m2_i32x8 = _mm256_or_si256(
        _mm256_slli_epi32(sign_i32x8, 5),
        _mm256_or_si256(_mm256_slli_epi32(clamped_exp_i32x8, 2), normal_mantissa_i32x8));

    // Subnormal path: mantissa = round(abs_f32 * 16)
    // If mantissa rounds to 4 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x04
    __m256 abs_f32x8 = _mm256_and_ps(f32x8, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    __m256 scaled_f32x8 = _mm256_mul_ps(abs_f32x8, _mm256_set1_ps(16.0f));
    __m256i subnorm_mantissa_i32x8 = _mm256_cvtps_epi32(scaled_f32x8);
    __m256i promotes_to_normal_i32x8 = _mm256_cmpgt_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(3));
    subnorm_mantissa_i32x8 = _mm256_min_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(3));
    subnorm_mantissa_i32x8 = _mm256_max_epi32(subnorm_mantissa_i32x8, _mm256_setzero_si256());
    __m256i subnorm_e3m2_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 5), subnorm_mantissa_i32x8);
    // When mantissa rounds to 4, use first normal value (0x04) instead of clamped subnormal
    __m256i first_normal_e3m2_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 5), _mm256_set1_epi32(0x04));
    subnorm_e3m2_i32x8 = _mm256_blendv_epi8(subnorm_e3m2_i32x8, first_normal_e3m2_i32x8, promotes_to_normal_i32x8);

    // Blend: use subnormal result when exp <= 0
    __m256i e3m2_i32x8 = _mm256_blendv_epi8(normal_e3m2_i32x8, subnorm_e3m2_i32x8, is_subnormal_i32x8);

    // Pack 8 i32s to 8 unsigned i8s (use unsigned saturation to preserve values 128-255)
    __m128i low_i32x4 = _mm256_castsi256_si128(e3m2_i32x8);
    __m128i high_i32x4 = _mm256_extracti128_si256(e3m2_i32x8, 1);
    __m128i packed_i16x8 = _mm_packus_epi32(low_i32x4, high_i32x4);
    __m128i packed_i8x8 = _mm_packus_epi16(packed_i16x8, packed_i16x8);
    return packed_i8x8;
}

#pragma endregion Vectorized Conversions

#pragma region Converting Loads and Stores

/** @brief Full load for f16 elements (8) with conversion to f32 via F16C. */
NK_INTERNAL void nk_load_f16x8_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm_ps = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)src));
}

/** @brief Partial load for f16 elements (up to 8) with conversion to f32 via F16C. */
NK_INTERNAL void nk_partial_load_f16x8_to_f32x8_haswell_(nk_f16_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t vec;
    nk_partial_load_b16x8_serial_(src, &vec, n);
    dst->ymm_ps = _mm256_cvtph_ps(vec.xmm);
}

/** @brief Full load for bf16 elements (8) with conversion to f32. */
NK_INTERNAL void nk_load_bf16x8_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm_ps = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)src));
}

/** @brief Partial load for bf16 elements (up to 8) with conversion to f32. */
NK_INTERNAL void nk_partial_load_bf16x8_to_f32x8_haswell_(nk_bf16_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t vec;
    nk_partial_load_b16x8_serial_(src, &vec, n);
    dst->ymm_ps = nk_bf16x8_to_f32x8_haswell_(vec.xmm);
}

/** @brief Full load for e4m3 elements (8) with conversion to f32. */
NK_INTERNAL void nk_load_e4m3x8_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm_ps = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)src));
}

/** @brief Partial load for e4m3 elements (up to 8) with conversion to f32. */
NK_INTERNAL void nk_partial_load_e4m3x8_to_f32x8_haswell_(nk_e4m3_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b64_vec_t vec;
    nk_partial_load_b8x8_serial_(src, &vec, n);
    dst->ymm_ps = nk_e4m3x8_to_f32x8_haswell_(_mm_cvtsi64_si128(vec.u64));
}

/** @brief Full load for e5m2 elements (8) with conversion to f32. */
NK_INTERNAL void nk_load_e5m2x8_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm_ps = nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)src));
}

/** @brief Partial load for e5m2 elements (up to 8) with conversion to f32. */
NK_INTERNAL void nk_partial_load_e5m2x8_to_f32x8_haswell_(nk_e5m2_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b64_vec_t vec;
    nk_partial_load_b8x8_serial_(src, &vec, n);
    dst->ymm_ps = nk_e5m2x8_to_f32x8_haswell_(_mm_cvtsi64_si128(vec.u64));
}

/** @brief Full load for e2m3 elements (8) with conversion to f32. */
NK_INTERNAL void nk_load_e2m3x8_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm_ps = nk_e2m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)src));
}

/** @brief Partial load for e2m3 elements (up to 8) with conversion to f32. */
NK_INTERNAL void nk_partial_load_e2m3x8_to_f32x8_haswell_(nk_e2m3_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b64_vec_t vec;
    nk_partial_load_b8x8_serial_(src, &vec, n);
    dst->ymm_ps = nk_e2m3x8_to_f32x8_haswell_(_mm_cvtsi64_si128(vec.u64));
}

/** @brief Full load for e3m2 elements (8) with conversion to f32. */
NK_INTERNAL void nk_load_e3m2x8_to_f32x8_haswell_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm_ps = nk_e3m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)src));
}

/** @brief Partial load for e3m2 elements (up to 8) with conversion to f32. */
NK_INTERNAL void nk_partial_load_e3m2x8_to_f32x8_haswell_(nk_e3m2_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b64_vec_t vec;
    nk_partial_load_b8x8_serial_(src, &vec, n);
    dst->ymm_ps = nk_e3m2x8_to_f32x8_haswell_(_mm_cvtsi64_si128(vec.u64));
}

/** @brief Partial load for i8 elements (up to 8) with conversion to f32. */
NK_INTERNAL void nk_partial_load_i8x8_to_f32x8_haswell_(nk_i8_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b64_vec_t vec;
    nk_partial_load_b8x8_serial_(src, &vec, n);
    dst->ymm_ps = nk_i8x8_to_f32x8_haswell_(_mm_cvtsi64_si128(vec.u64));
}

/** @brief Partial load for u8 elements (up to 8) with conversion to f32. */
NK_INTERNAL void nk_partial_load_u8x8_to_f32x8_haswell_(nk_u8_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b64_vec_t vec;
    nk_partial_load_b8x8_serial_(src, &vec, n);
    dst->ymm_ps = nk_u8x8_to_f32x8_haswell_(_mm_cvtsi64_si128(vec.u64));
}

/** @brief Partial load for i16 elements (up to 8) with conversion to f32. */
NK_INTERNAL void nk_partial_load_i16x8_to_f32x8_haswell_(nk_i16_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t vec;
    nk_partial_load_b16x8_serial_(src, &vec, n);
    dst->ymm_ps = nk_i16x8_to_f32x8_haswell_(vec.xmm);
}

/** @brief Partial load for u16 elements (up to 8) with conversion to f32. */
NK_INTERNAL void nk_partial_load_u16x8_to_f32x8_haswell_(nk_u16_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t vec;
    nk_partial_load_b16x8_serial_(src, &vec, n);
    dst->ymm_ps = nk_u16x8_to_f32x8_haswell_(vec.xmm);
}

/** @brief Partial load for i32 elements (up to 8) with conversion to f32. */
NK_INTERNAL void nk_partial_load_i32x8_to_f32x8_haswell_(nk_i32_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b256_vec_t vec;
    nk_partial_load_b32x8_serial_(src, &vec, n);
    dst->ymm_ps = nk_i32x8_to_f32x8_haswell_(vec.ymm);
}

/** @brief Partial load for u32 elements (up to 8) with conversion to f32. */
NK_INTERNAL void nk_partial_load_u32x8_to_f32x8_haswell_(nk_u32_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b256_vec_t vec;
    nk_partial_load_b32x8_serial_(src, &vec, n);
    dst->ymm_ps = nk_u32x8_to_f32x8_haswell_(vec.ymm);
}

#pragma endregion Converting Loads and Stores

#pragma region Public API

NK_PUBLIC void nk_cast_haswell(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
    // Same-type fast path
    if (from_type == to_type) {
        nk_size_t size_bits = nk_dtype_bits(from_type);
        if (size_bits > 0) nk_copy_bytes_(to, from, nk_size_divide_round_up_(n * size_bits, NK_BITS_PER_BYTE));
        return;
    }

    // Supported types: floats (f32, f16, bf16, e4m3, e5m2, e2m3, e3m2) and integers (i8, u8, i16, u16, i32, u32)
    int from_supported = (from_type == nk_f32_k || from_type == nk_f16_k || from_type == nk_bf16_k ||
                          from_type == nk_e4m3_k || from_type == nk_e5m2_k || from_type == nk_e2m3_k ||
                          from_type == nk_e3m2_k || from_type == nk_i8_k || from_type == nk_u8_k ||
                          from_type == nk_i16_k || from_type == nk_u16_k || from_type == nk_i32_k ||
                          from_type == nk_u32_k);
    int to_supported = (to_type == nk_f32_k || to_type == nk_f16_k || to_type == nk_bf16_k || to_type == nk_e4m3_k ||
                        to_type == nk_e5m2_k || to_type == nk_e2m3_k || to_type == nk_e3m2_k || to_type == nk_i8_k ||
                        to_type == nk_u8_k || to_type == nk_i16_k || to_type == nk_u16_k || to_type == nk_i32_k ||
                        to_type == nk_u32_k);
    if (!from_supported || !to_supported) {
        nk_cast_serial(from, from_type, n, to, to_type);
        return;
    }

    // Fall back to serial for i32/u32↔i32/u32 (f32 intermediate loses precision for large values)
    int from_32bit_int = (from_type == nk_i32_k || from_type == nk_u32_k);
    int to_32bit_int = (to_type == nk_i32_k || to_type == nk_u32_k);
    if (from_32bit_int && to_32bit_int) {
        nk_cast_serial(from, from_type, n, to, to_type);
        return;
    }

    // Byte steps per 8 elements
    nk_size_t from_step = 8 * nk_dtype_bits(from_type) / NK_BITS_PER_BYTE;
    nk_size_t to_step = 8 * nk_dtype_bits(to_type) / NK_BITS_PER_BYTE;

    nk_u8_t const *from_ptr = (nk_u8_t const *)from;
    nk_u8_t *to_ptr = (nk_u8_t *)to;
    nk_size_t batches = n / 8;
    nk_size_t tail = n % 8;
    nk_b256_vec_t hub;

    for (nk_size_t idx = 0; idx < batches; ++idx, from_ptr += from_step, to_ptr += to_step) {
        // Upcast to f32x8
        if (from_type == nk_f32_k) hub.ymm_ps = _mm256_loadu_ps((float const *)from_ptr);
        else if (from_type == nk_f16_k) hub.ymm_ps = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)from_ptr));
        else if (from_type == nk_bf16_k)
            hub.ymm_ps = nk_bf16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)from_ptr));
        else if (from_type == nk_e4m3_k)
            hub.ymm_ps = nk_e4m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)from_ptr));
        else if (from_type == nk_e5m2_k)
            hub.ymm_ps = nk_e5m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)from_ptr));
        else if (from_type == nk_e2m3_k)
            hub.ymm_ps = nk_e2m3x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)from_ptr));
        else if (from_type == nk_e3m2_k)
            hub.ymm_ps = nk_e3m2x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)from_ptr));
        else if (from_type == nk_i8_k)
            hub.ymm_ps = nk_i8x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)from_ptr));
        else if (from_type == nk_u8_k)
            hub.ymm_ps = nk_u8x8_to_f32x8_haswell_(_mm_loadl_epi64((__m128i const *)from_ptr));
        else if (from_type == nk_i16_k)
            hub.ymm_ps = nk_i16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)from_ptr));
        else if (from_type == nk_u16_k)
            hub.ymm_ps = nk_u16x8_to_f32x8_haswell_(_mm_loadu_si128((__m128i const *)from_ptr));
        else if (from_type == nk_i32_k)
            hub.ymm_ps = nk_i32x8_to_f32x8_haswell_(_mm256_loadu_si256((__m256i const *)from_ptr));
        else if (from_type == nk_u32_k)
            hub.ymm_ps = nk_u32x8_to_f32x8_haswell_(_mm256_loadu_si256((__m256i const *)from_ptr));

        // Downcast from f32x8
        if (to_type == nk_f32_k) _mm256_storeu_ps((float *)to_ptr, hub.ymm_ps);
        else if (to_type == nk_f16_k)
            _mm_storeu_si128((__m128i *)to_ptr, _mm256_cvtps_ph(hub.ymm_ps, _MM_FROUND_TO_NEAREST_INT));
        else if (to_type == nk_bf16_k) _mm_storeu_si128((__m128i *)to_ptr, nk_f32x8_to_bf16x8_haswell_(hub.ymm_ps));
        else if (to_type == nk_e4m3_k) _mm_storel_epi64((__m128i *)to_ptr, nk_f32x8_to_e4m3x8_haswell_(hub.ymm_ps));
        else if (to_type == nk_e5m2_k) _mm_storel_epi64((__m128i *)to_ptr, nk_f32x8_to_e5m2x8_haswell_(hub.ymm_ps));
        else if (to_type == nk_e2m3_k) _mm_storel_epi64((__m128i *)to_ptr, nk_f32x8_to_e2m3x8_haswell_(hub.ymm_ps));
        else if (to_type == nk_e3m2_k) _mm_storel_epi64((__m128i *)to_ptr, nk_f32x8_to_e3m2x8_haswell_(hub.ymm_ps));
        else if (to_type == nk_i8_k) _mm_storel_epi64((__m128i *)to_ptr, nk_f32x8_to_i8x8_haswell_(hub.ymm_ps));
        else if (to_type == nk_u8_k) _mm_storel_epi64((__m128i *)to_ptr, nk_f32x8_to_u8x8_haswell_(hub.ymm_ps));
        else if (to_type == nk_i16_k) _mm_storeu_si128((__m128i *)to_ptr, nk_f32x8_to_i16x8_haswell_(hub.ymm_ps));
        else if (to_type == nk_u16_k) _mm_storeu_si128((__m128i *)to_ptr, nk_f32x8_to_u16x8_haswell_(hub.ymm_ps));
        else if (to_type == nk_i32_k) _mm256_storeu_si256((__m256i *)to_ptr, nk_f32x8_to_i32x8_haswell_(hub.ymm_ps));
        else if (to_type == nk_u32_k) _mm256_storeu_si256((__m256i *)to_ptr, nk_f32x8_to_u32x8_haswell_(hub.ymm_ps));
    }

    // Handle tail with partial loads/stores
    if (tail) {
        // Upcast tail to f32x8
        if (from_type == nk_f32_k) nk_partial_load_b32x8_serial_(from_ptr, &hub, tail);
        else if (from_type == nk_f16_k) nk_partial_load_f16x8_to_f32x8_haswell_((nk_f16_t const *)from_ptr, &hub, tail);
        else if (from_type == nk_bf16_k)
            nk_partial_load_bf16x8_to_f32x8_haswell_((nk_bf16_t const *)from_ptr, &hub, tail);
        else if (from_type == nk_e4m3_k)
            nk_partial_load_e4m3x8_to_f32x8_haswell_((nk_e4m3_t const *)from_ptr, &hub, tail);
        else if (from_type == nk_e5m2_k)
            nk_partial_load_e5m2x8_to_f32x8_haswell_((nk_e5m2_t const *)from_ptr, &hub, tail);
        else if (from_type == nk_e2m3_k)
            nk_partial_load_e2m3x8_to_f32x8_haswell_((nk_e2m3_t const *)from_ptr, &hub, tail);
        else if (from_type == nk_e3m2_k)
            nk_partial_load_e3m2x8_to_f32x8_haswell_((nk_e3m2_t const *)from_ptr, &hub, tail);
        else if (from_type == nk_i8_k) nk_partial_load_i8x8_to_f32x8_haswell_((nk_i8_t const *)from_ptr, &hub, tail);
        else if (from_type == nk_u8_k) nk_partial_load_u8x8_to_f32x8_haswell_((nk_u8_t const *)from_ptr, &hub, tail);
        else if (from_type == nk_i16_k) nk_partial_load_i16x8_to_f32x8_haswell_((nk_i16_t const *)from_ptr, &hub, tail);
        else if (from_type == nk_u16_k) nk_partial_load_u16x8_to_f32x8_haswell_((nk_u16_t const *)from_ptr, &hub, tail);
        else if (from_type == nk_i32_k) nk_partial_load_i32x8_to_f32x8_haswell_((nk_i32_t const *)from_ptr, &hub, tail);
        else if (from_type == nk_u32_k) nk_partial_load_u32x8_to_f32x8_haswell_((nk_u32_t const *)from_ptr, &hub, tail);

        // Downcast and store tail
        if (to_type == nk_f32_k) nk_partial_store_b32x8_serial_(&hub, to_ptr, tail);
        else if (to_type == nk_f16_k) {
            hub.xmms[0] = _mm256_cvtps_ph(hub.ymm_ps, _MM_FROUND_TO_NEAREST_INT);
            nk_partial_store_b16x8_serial_((nk_b128_vec_t *)&hub, to_ptr, tail);
        }
        else if (to_type == nk_bf16_k) {
            hub.xmms[0] = nk_f32x8_to_bf16x8_haswell_(hub.ymm_ps);
            nk_partial_store_b16x8_serial_((nk_b128_vec_t *)&hub, to_ptr, tail);
        }
        else if (to_type == nk_e4m3_k) {
            hub.xmms[0] = nk_f32x8_to_e4m3x8_haswell_(hub.ymm_ps);
            nk_partial_store_b8x8_serial_((nk_b64_vec_t *)&hub, to_ptr, tail);
        }
        else if (to_type == nk_e5m2_k) {
            hub.xmms[0] = nk_f32x8_to_e5m2x8_haswell_(hub.ymm_ps);
            nk_partial_store_b8x8_serial_((nk_b64_vec_t *)&hub, to_ptr, tail);
        }
        else if (to_type == nk_e2m3_k) {
            hub.xmms[0] = nk_f32x8_to_e2m3x8_haswell_(hub.ymm_ps);
            nk_partial_store_b8x8_serial_((nk_b64_vec_t *)&hub, to_ptr, tail);
        }
        else if (to_type == nk_e3m2_k) {
            hub.xmms[0] = nk_f32x8_to_e3m2x8_haswell_(hub.ymm_ps);
            nk_partial_store_b8x8_serial_((nk_b64_vec_t *)&hub, to_ptr, tail);
        }
        else if (to_type == nk_i8_k) {
            hub.xmms[0] = nk_f32x8_to_i8x8_haswell_(hub.ymm_ps);
            nk_partial_store_b8x8_serial_((nk_b64_vec_t *)&hub, to_ptr, tail);
        }
        else if (to_type == nk_u8_k) {
            hub.xmms[0] = nk_f32x8_to_u8x8_haswell_(hub.ymm_ps);
            nk_partial_store_b8x8_serial_((nk_b64_vec_t *)&hub, to_ptr, tail);
        }
        else if (to_type == nk_i16_k) {
            hub.xmms[0] = nk_f32x8_to_i16x8_haswell_(hub.ymm_ps);
            nk_partial_store_b16x8_serial_((nk_b128_vec_t *)&hub, to_ptr, tail);
        }
        else if (to_type == nk_u16_k) {
            hub.xmms[0] = nk_f32x8_to_u16x8_haswell_(hub.ymm_ps);
            nk_partial_store_b16x8_serial_((nk_b128_vec_t *)&hub, to_ptr, tail);
        }
        else if (to_type == nk_i32_k) {
            hub.ymm = nk_f32x8_to_i32x8_haswell_(hub.ymm_ps);
            nk_partial_store_b32x8_serial_(&hub, to_ptr, tail);
        }
        else if (to_type == nk_u32_k) {
            hub.ymm = nk_f32x8_to_u32x8_haswell_(hub.ymm_ps);
            nk_partial_store_b32x8_serial_(&hub, to_ptr, tail);
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

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X8664_
#endif // NK_CAST_HASWELL_H
