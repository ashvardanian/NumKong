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
    __m128i index_i32x4 = _mm_setr_epi32(0, 1, 2, 3);
    __m128i limit_i32x4 = _mm_set1_epi32((int)n);
    __m128i mask_i32x4 = _mm_cmpgt_epi32(limit_i32x4, index_i32x4);
    dst->xmm = _mm_castps_si128(_mm_maskload_ps((float const *)src, mask_i32x4));
}

/** @brief Type-agnostic 128-bit partial store with AVX maskstore. */
NK_INTERNAL void nk_partial_store_b32x4_haswell_(nk_b128_vec_t const *src, void *dst, nk_size_t n) {
    __m128i index_i32x4 = _mm_setr_epi32(0, 1, 2, 3);
    __m128i limit_i32x4 = _mm_set1_epi32((int)n);
    __m128i mask_i32x4 = _mm_cmpgt_epi32(limit_i32x4, index_i32x4);
    _mm_maskstore_ps((float *)dst, mask_i32x4, _mm_castsi128_ps(src->xmm));
}

/** @brief Type-agnostic 256-bit partial load with AVX2 maskload. */
NK_INTERNAL void nk_partial_load_b64x4_haswell_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    __m256i index_i64x4 = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i limit_i64x4 = _mm256_set1_epi64x((long long)n);
    __m256i mask_i64x4 = _mm256_cmpgt_epi64(limit_i64x4, index_i64x4);
    dst->ymm = _mm256_castpd_si256(_mm256_maskload_pd((double const *)src, mask_i64x4));
}

/** @brief Type-agnostic 256-bit partial store with AVX2 maskstore. */
NK_INTERNAL void nk_partial_store_b64x4_haswell_(nk_b256_vec_t const *src, void *dst, nk_size_t n) {
    __m256i index_i64x4 = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i limit_i64x4 = _mm256_set1_epi64x((long long)n);
    __m256i mask_i64x4 = _mm256_cmpgt_epi64(limit_i64x4, index_i64x4);
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
    __m256i rounded_i32x8 = _mm256_cvtps_epi32(clamped_f32x8);
    return _mm_packs_epi32(_mm256_castsi256_si128(rounded_i32x8), _mm256_extracti128_si256(rounded_i32x8, 1));
}
NK_INTERNAL __m128i nk_f32x8_to_u16x8_haswell_(__m256 f32x8) {
    __m256 clamped_f32x8 = _mm256_min_ps(_mm256_max_ps(f32x8, _mm256_setzero_ps()), _mm256_set1_ps(65535.0f));
    __m256i rounded_i32x8 = _mm256_cvtps_epi32(clamped_f32x8);
    return _mm_packus_epi32(_mm256_castsi256_si128(rounded_i32x8), _mm256_extracti128_si256(rounded_i32x8, 1));
}
NK_INTERNAL __m128i nk_f32x8_to_i8x8_haswell_(__m256 f32x8) {
    __m256 clamped_f32x8 = _mm256_min_ps(_mm256_max_ps(f32x8, _mm256_set1_ps(-128.0f)), _mm256_set1_ps(127.0f));
    __m256i rounded_i32x8 = _mm256_cvtps_epi32(clamped_f32x8);
    __m128i packed_i16x8 = _mm_packs_epi32(_mm256_castsi256_si128(rounded_i32x8),
                                           _mm256_extracti128_si256(rounded_i32x8, 1));
    return _mm_packs_epi16(packed_i16x8, _mm_setzero_si128());
}
NK_INTERNAL __m128i nk_f32x8_to_u8x8_haswell_(__m256 f32x8) {
    __m256 clamped_f32x8 = _mm256_min_ps(_mm256_max_ps(f32x8, _mm256_setzero_ps()), _mm256_set1_ps(255.0f));
    __m256i rounded_i32x8 = _mm256_cvtps_epi32(clamped_f32x8);
    __m128i packed_u16x8 = _mm_packus_epi32(_mm256_castsi256_si128(rounded_i32x8),
                                            _mm256_extracti128_si256(rounded_i32x8, 1));
    return _mm_packus_epi16(packed_u16x8, _mm_setzero_si128());
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
    __m128i f16_u16x8 = _mm_or_si128(shifted_magnitude_u16x8, shifted_sign_u16x8);
    f16_u16x8 = _mm_blendv_epi8(f16_u16x8, f16_nan_u16x8, is_nan_u16x8);
    __m256 fake_f32x8 = _mm256_cvtph_ps(f16_u16x8);
    return _mm256_mul_ps(fake_f32x8, _mm256_set1_ps(256.0f));
}

/** @brief Convert 8x e5m2 → 8x f32 via free-shift widen (AVX2 + F16C).
 *  E5M2 shares F16's exponent bias (15): `(byte << 8)` is the matching F16 bit
 *  pattern for every E5M2 value (normals, subnormals, zero, ±Inf, NaN — all
 *  bit-exact). Widen u8 → u16, shift, then VCVTPH2PS to F32. Three ops total. */
NK_INTERNAL __m256 nk_e5m2x8_to_f32x8_haswell_(__m128i e5m2_i8x8) {
    __m128i e5m2_u16x8 = _mm_cvtepu8_epi16(e5m2_i8x8);
    __m128i f16_u16x8 = _mm_slli_epi16(e5m2_u16x8, 8);
    return _mm256_cvtph_ps(f16_u16x8);
}

/** @brief Convert 8x f32 → 8x e4m3 via bit manipulation (AVX2).
 *  E4M3 format: S EEEE MMM (bias=7). Handles normal, subnormal, and overflow cases.
 *  Subnormals (f32_exp ≤ 120): mantissa = round(abs_f32 * 512), clamped to [0,7]. */
NK_INTERNAL __m128i nk_f32x8_to_e4m3x8_haswell_(__m256 f32x8) {
    __m256i bits_i32x8 = _mm256_castps_si256(f32x8);
    __m256i sign_i32x8 = _mm256_srli_epi32(bits_i32x8, 31);
    __m256i f32_exponent_i32x8 = _mm256_and_si256(_mm256_srli_epi32(bits_i32x8, 23), _mm256_set1_epi32(0xFF));

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
    __m256i e4m3_exponent_i32x8 = _mm256_sub_epi32(_mm256_add_epi32(f32_exponent_i32x8, carry_i32x8),
                                                   _mm256_set1_epi32(120));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 15)
    __m256i is_subnormal_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e4m3_exponent_i32x8);
    __m256i overflow_i32x8 = _mm256_cmpgt_epi32(e4m3_exponent_i32x8, _mm256_set1_epi32(15));

    // Normal path: clamp exp to [1,15], extract mantissa bits
    // e4m3FN quirk: exp=15 with mantissa=7 is NaN (0x7F), so clamp mantissa to 6 when exp=15.
    __m256i clamped_exponent_i32x8 = _mm256_max_epi32(e4m3_exponent_i32x8, _mm256_set1_epi32(1));
    clamped_exponent_i32x8 = _mm256_min_epi32(clamped_exponent_i32x8, _mm256_set1_epi32(15));
    __m256i is_max_exponent_i32x8 = _mm256_cmpeq_epi32(clamped_exponent_i32x8, _mm256_set1_epi32(15));
    __m256i max_mantissa_i32x8 = _mm256_blendv_epi8(_mm256_set1_epi32(7), _mm256_set1_epi32(6), is_max_exponent_i32x8);
    __m256i normal_mantissa_i32x8 = _mm256_min_epi32(f32_mantissa_i32x8, max_mantissa_i32x8);
    normal_mantissa_i32x8 = _mm256_blendv_epi8(normal_mantissa_i32x8, _mm256_set1_epi32(0x06), overflow_i32x8);
    __m256i normal_e4m3_i32x8 = _mm256_or_si256(
        _mm256_slli_epi32(sign_i32x8, 7),
        _mm256_or_si256(_mm256_slli_epi32(clamped_exponent_i32x8, 3), normal_mantissa_i32x8));

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
    __m256i f32_exponent_i32x8 = _mm256_and_si256(_mm256_srli_epi32(bits_i32x8, 23), _mm256_set1_epi32(0xFF));

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
    __m256i e5m2_exponent_i32x8 = _mm256_sub_epi32(_mm256_add_epi32(f32_exponent_i32x8, carry_i32x8),
                                                   _mm256_set1_epi32(112));

    // Detect subnormal (exp <= 0) and overflow (exp > 31)
    __m256i is_subnormal_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e5m2_exponent_i32x8);
    __m256i overflow_i32x8 = _mm256_cmpgt_epi32(e5m2_exponent_i32x8, _mm256_set1_epi32(31));

    // Normal path: clamp exp to [1,31], on overflow return infinity (exp=31, mantissa=0 = 0x7C)
    __m256i clamped_exponent_i32x8 = _mm256_max_epi32(e5m2_exponent_i32x8, _mm256_set1_epi32(1));
    clamped_exponent_i32x8 = _mm256_min_epi32(clamped_exponent_i32x8, _mm256_set1_epi32(31));
    __m256i normal_mantissa_i32x8 = _mm256_blendv_epi8(f32_mantissa_i32x8, _mm256_setzero_si256(), overflow_i32x8);
    __m256i normal_e5m2_i32x8 = _mm256_or_si256(
        _mm256_slli_epi32(sign_i32x8, 7),
        _mm256_or_si256(_mm256_slli_epi32(clamped_exponent_i32x8, 2), normal_mantissa_i32x8));

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
    __m256i exponent_i32x8 = _mm256_and_si256(_mm256_srli_epi32(e2m3_i32x8, 3), _mm256_set1_epi32(0x03));
    __m256i mantissa_i32x8 = _mm256_and_si256(e2m3_i32x8, _mm256_set1_epi32(0x07));

    // Build F32 sign bit
    __m256i f32_sign_i32x8 = _mm256_slli_epi32(_mm256_srli_epi32(e2m3_i32x8, 5), 31);

    // Normal path: sign | ((exp+126)<<23) | (mant<<20)
    __m256i f32_exponent_i32x8 = _mm256_slli_epi32(_mm256_add_epi32(exponent_i32x8, _mm256_set1_epi32(126)), 23);
    __m256i f32_mantissa_i32x8 = _mm256_slli_epi32(mantissa_i32x8, 20);
    __m256i normal_i32x8 = _mm256_or_si256(f32_sign_i32x8, _mm256_or_si256(f32_exponent_i32x8, f32_mantissa_i32x8));

    // Subnormal path: value = mantissa / 8.0f, then apply sign
    __m256 subnorm_abs_f32x8 = _mm256_mul_ps(_mm256_cvtepi32_ps(mantissa_i32x8), _mm256_set1_ps(1.0f / 8.0f));
    __m256 subnorm_f32x8 = _mm256_or_ps(subnorm_abs_f32x8, _mm256_castsi256_ps(f32_sign_i32x8));

    // Blend: if exp==0, use subnormal result; otherwise use normal bits
    __m256i exponent_zero_b32x8 = _mm256_cmpeq_epi32(exponent_i32x8, _mm256_setzero_si256());
    return _mm256_blendv_ps(_mm256_castsi256_ps(normal_i32x8), subnorm_f32x8, _mm256_castsi256_ps(exponent_zero_b32x8));
}

/** @brief Convert 8x e3m2 → 8x f32 via bit manipulation (AVX2).
 *  E3M2 format: S EEE MM (bias=3). F32: sign<<31, (exp+124)<<23, mantissa<<21.
 *  Subnormals (exp=0): value = mantissa × 2⁽¹⁻³⁾ × 2⁻² = mantissa ÷ 16. */
NK_INTERNAL __m256 nk_e3m2x8_to_f32x8_haswell_(__m128i e3m2_i8x8) {
    __m256i e3m2_i32x8 = _mm256_cvtepu8_epi32(e3m2_i8x8);

    // Extract fields (only 6 bits used: S EEE MM)
    __m256i exponent_i32x8 = _mm256_and_si256(_mm256_srli_epi32(e3m2_i32x8, 2), _mm256_set1_epi32(0x07));
    __m256i mantissa_i32x8 = _mm256_and_si256(e3m2_i32x8, _mm256_set1_epi32(0x03));

    // Build F32 sign bit
    __m256i f32_sign_i32x8 = _mm256_slli_epi32(_mm256_srli_epi32(e3m2_i32x8, 5), 31);

    // Normal path: sign | ((exp+124)<<23) | (mant<<21)
    __m256i f32_exponent_i32x8 = _mm256_slli_epi32(_mm256_add_epi32(exponent_i32x8, _mm256_set1_epi32(124)), 23);
    __m256i f32_mantissa_i32x8 = _mm256_slli_epi32(mantissa_i32x8, 21);
    __m256i normal_i32x8 = _mm256_or_si256(f32_sign_i32x8, _mm256_or_si256(f32_exponent_i32x8, f32_mantissa_i32x8));

    // Subnormal path: value = mantissa / 16.0f, then apply sign
    __m256 subnorm_abs_f32x8 = _mm256_mul_ps(_mm256_cvtepi32_ps(mantissa_i32x8), _mm256_set1_ps(1.0f / 16.0f));
    __m256 subnorm_f32x8 = _mm256_or_ps(subnorm_abs_f32x8, _mm256_castsi256_ps(f32_sign_i32x8));

    // Blend: if exp==0, use subnormal result; otherwise use normal bits
    __m256i exponent_zero_b32x8 = _mm256_cmpeq_epi32(exponent_i32x8, _mm256_setzero_si256());
    return _mm256_blendv_ps(_mm256_castsi256_ps(normal_i32x8), subnorm_f32x8, _mm256_castsi256_ps(exponent_zero_b32x8));
}

/** @brief Convert 8x f32 → 8x e2m3 via bit manipulation (AVX2).
 *  E2M3 format: S EE MMM (bias=1). Handles normal, subnormal, and overflow cases.
 *  Subnormals (f32_exp ≤ 126): mantissa = round(abs_f32 * 8), clamped to [0,7]. */
NK_INTERNAL __m128i nk_f32x8_to_e2m3x8_haswell_(__m256 f32x8) {
    __m256i bits_i32x8 = _mm256_castps_si256(f32x8);
    __m256i sign_i32x8 = _mm256_srli_epi32(bits_i32x8, 31);
    __m256i f32_exponent_i32x8 = _mm256_and_si256(_mm256_srli_epi32(bits_i32x8, 23), _mm256_set1_epi32(0xFF));

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
    __m256i e2m3_exponent_i32x8 = _mm256_sub_epi32(_mm256_add_epi32(f32_exponent_i32x8, carry_i32x8),
                                                   _mm256_set1_epi32(126));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 3)
    __m256i is_subnormal_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e2m3_exponent_i32x8);
    __m256i overflow_i32x8 = _mm256_cmpgt_epi32(e2m3_exponent_i32x8, _mm256_set1_epi32(3));

    // Normal path: clamp exp to [1,3], extract mantissa bits
    __m256i clamped_exponent_i32x8 = _mm256_max_epi32(e2m3_exponent_i32x8, _mm256_set1_epi32(1));
    clamped_exponent_i32x8 = _mm256_min_epi32(clamped_exponent_i32x8, _mm256_set1_epi32(3));
    __m256i normal_mantissa_i32x8 = _mm256_blendv_epi8(f32_mantissa_i32x8, _mm256_set1_epi32(0x07), overflow_i32x8);
    __m256i normal_e2m3_i32x8 = _mm256_or_si256(
        _mm256_slli_epi32(sign_i32x8, 5),
        _mm256_or_si256(_mm256_slli_epi32(clamped_exponent_i32x8, 3), normal_mantissa_i32x8));

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
    __m256i f32_exponent_i32x8 = _mm256_and_si256(_mm256_srli_epi32(bits_i32x8, 23), _mm256_set1_epi32(0xFF));

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
    __m256i e3m2_exponent_i32x8 = _mm256_sub_epi32(_mm256_add_epi32(f32_exponent_i32x8, carry_i32x8),
                                                   _mm256_set1_epi32(124));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 7)
    __m256i is_subnormal_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e3m2_exponent_i32x8);
    __m256i overflow_i32x8 = _mm256_cmpgt_epi32(e3m2_exponent_i32x8, _mm256_set1_epi32(7));

    // Normal path: clamp exp to [1,7], extract mantissa bits
    __m256i clamped_exponent_i32x8 = _mm256_max_epi32(e3m2_exponent_i32x8, _mm256_set1_epi32(1));
    clamped_exponent_i32x8 = _mm256_min_epi32(clamped_exponent_i32x8, _mm256_set1_epi32(7));
    __m256i normal_mantissa_i32x8 = _mm256_blendv_epi8(f32_mantissa_i32x8, _mm256_set1_epi32(0x03), overflow_i32x8);
    __m256i normal_e3m2_i32x8 = _mm256_or_si256(
        _mm256_slli_epi32(sign_i32x8, 5),
        _mm256_or_si256(_mm256_slli_epi32(clamped_exponent_i32x8, 2), normal_mantissa_i32x8));

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

/** @brief Convert 8× e2m1 → 8× f32 via 8-magnitude LUT + sign flip (AVX2).
 *  Input: 4 bytes (low 32 bits of @p packed) holding 8 nibbles, high nibble of byte → even lane.
 *  E2M1 magnitudes {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} indexed by nibble bits 2..0; bit 3 → sign. */
NK_INTERNAL __m256 nk_e2m1x8_to_f32x8_haswell_(__m128i packed) {
    // Expand 4 packed bytes to 8 nibble bytes via shift + mask + unpack interleave
    __m128i low_nibbles_b8x16 = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
    __m128i high_nibbles_b8x16 = _mm_and_si128(_mm_srli_epi32(packed, 4), _mm_set1_epi8(0x0F));
    __m128i nibbles_b8x8 = _mm_unpacklo_epi8(high_nibbles_b8x16, low_nibbles_b8x16);
    __m256i nibbles_i32x8 = _mm256_cvtepu8_epi32(nibbles_b8x8);

    // Magnitude LUT indexed by bits 2..0 (8 entries, gathered via permutevar within a 256-bit lane).
    __m256 magnitude_lut_f32x8 = _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f);
    __m256i magnitude_index_i32x8 = _mm256_and_si256(nibbles_i32x8, _mm256_set1_epi32(0x07));
    __m256 magnitudes_f32x8 = _mm256_permutevar8x32_ps(magnitude_lut_f32x8, magnitude_index_i32x8);

    // Sign: bit 3 of the nibble → bit 31 of the f32
    __m256i sign_f32_i32x8 = _mm256_slli_epi32(_mm256_and_si256(nibbles_i32x8, _mm256_set1_epi32(0x08)), 28);
    return _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(magnitudes_f32x8), sign_f32_i32x8));
}

/** @brief Convert 8× f32 → 8× e2m1 via bit manipulation, packed to 4 bytes (AVX2).
 *  Output: 4 bytes in the low 32 bits of the result. Lane 0 → high nibble of byte 0, lane 1 → low nibble. */
NK_INTERNAL __m128i nk_f32x8_to_e2m1x8_haswell_(__m256 f32x8) {
    __m256i bits_i32x8 = _mm256_castps_si256(f32x8);
    __m256i sign_i32x8 = _mm256_srli_epi32(bits_i32x8, 31);
    __m256i f32_exponent_i32x8 = _mm256_and_si256(_mm256_srli_epi32(bits_i32x8, 23), _mm256_set1_epi32(0xFF));

    // Normal path: round 23-bit mantissa to 1 bit using RNE (cut at bit 22).
    __m256i significand_i32x8 = _mm256_or_si256(_mm256_and_si256(bits_i32x8, _mm256_set1_epi32(0x007FFFFF)),
                                                _mm256_set1_epi32(0x00800000));
    __m256i lsb_i32x8 = _mm256_and_si256(_mm256_srli_epi32(significand_i32x8, 22), _mm256_set1_epi32(1));
    __m256i rounding_bias_i32x8 = _mm256_add_epi32(_mm256_set1_epi32(0x001FFFFF), lsb_i32x8);
    __m256i rounded_sig_i32x8 = _mm256_add_epi32(significand_i32x8, rounding_bias_i32x8);
    __m256i carry_i32x8 = _mm256_srli_epi32(rounded_sig_i32x8, 24);
    __m256i normal_mantissa_i32x8 = _mm256_and_si256(_mm256_srli_epi32(rounded_sig_i32x8, 22), _mm256_set1_epi32(0x01));
    __m256i e2m1_exponent_i32x8 = _mm256_sub_epi32(_mm256_add_epi32(f32_exponent_i32x8, carry_i32x8),
                                                   _mm256_set1_epi32(126));

    __m256i is_subnormal_i32x8 = _mm256_cmpgt_epi32(_mm256_set1_epi32(1), e2m1_exponent_i32x8);
    __m256i overflow_i32x8 = _mm256_cmpgt_epi32(e2m1_exponent_i32x8, _mm256_set1_epi32(3));

    __m256i clamped_exponent_i32x8 = _mm256_max_epi32(e2m1_exponent_i32x8, _mm256_set1_epi32(1));
    clamped_exponent_i32x8 = _mm256_min_epi32(clamped_exponent_i32x8, _mm256_set1_epi32(3));
    normal_mantissa_i32x8 = _mm256_blendv_epi8(normal_mantissa_i32x8, _mm256_set1_epi32(0x01), overflow_i32x8);
    __m256i normal_nibble_i32x8 = _mm256_or_si256(
        _mm256_slli_epi32(sign_i32x8, 3),
        _mm256_or_si256(_mm256_slli_epi32(clamped_exponent_i32x8, 1), normal_mantissa_i32x8));

    // Subnormal path: round(|x| * 2), clamp to {0, 1}. Promotion to first normal (0x02) when it rounds up to 2.
    __m256 abs_f32x8 = _mm256_and_ps(f32x8, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
    __m256 scaled_f32x8 = _mm256_mul_ps(abs_f32x8, _mm256_set1_ps(2.0f));
    __m256i subnorm_mantissa_i32x8 = _mm256_cvtps_epi32(scaled_f32x8);
    __m256i promotes_to_normal_i32x8 = _mm256_cmpgt_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(1));
    subnorm_mantissa_i32x8 = _mm256_max_epi32(_mm256_min_epi32(subnorm_mantissa_i32x8, _mm256_set1_epi32(1)),
                                              _mm256_setzero_si256());
    __m256i subnorm_nibble_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 3), subnorm_mantissa_i32x8);
    __m256i first_normal_nibble_i32x8 = _mm256_or_si256(_mm256_slli_epi32(sign_i32x8, 3), _mm256_set1_epi32(0x02));
    subnorm_nibble_i32x8 = _mm256_blendv_epi8(subnorm_nibble_i32x8, first_normal_nibble_i32x8,
                                              promotes_to_normal_i32x8);

    __m256i nibble_i32x8 = _mm256_blendv_epi8(normal_nibble_i32x8, subnorm_nibble_i32x8, is_subnormal_i32x8);

    // Pack 8 nibbles (each in low 4 bits of a byte) to 4 bytes: even idx → high nibble, odd → low.
    __m128i low_i32x4 = _mm256_castsi256_si128(nibble_i32x8);
    __m128i high_i32x4 = _mm256_extracti128_si256(nibble_i32x8, 1);
    __m128i nibble_i16x8 = _mm_packus_epi32(low_i32x4, high_i32x4);
    __m128i nibble_b8x8 = _mm_packus_epi16(nibble_i16x8, _mm_setzero_si128());
    __m128i pack_coeff_i16x8 = _mm_set1_epi16(0x0110); // byte 0 coefficient 0x10, byte 1 coefficient 0x01
    __m128i packed_i16x4 = _mm_maddubs_epi16(nibble_b8x8, pack_coeff_i16x8);
    return _mm_packus_epi16(packed_i16x4, _mm_setzero_si128());
}

/** @brief Reduce a block of `block_count` f32s to `amax = max(|x|)`. `block_count` ≤ 32.
 *  Propagates NaN (returns a NaN when any lane is NaN) so the block scale becomes the NaN sentinel. */
NK_INTERNAL nk_f32_t nk_block_amax_f32_haswell_(nk_f32_t const *block, nk_size_t block_count) {
    __m256 const abs_mask_f32x8 = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    nk_fui32_t qnan;
    qnan.u = 0x7FC00000u; // NaN in any lane → propagate so the block scale becomes the NaN sentinel
    __m256 amax_f32x8 = _mm256_setzero_ps();
    __m256 nan_accumulator_f32x8 = _mm256_setzero_ps(); // accumulates unordered (NaN) comparisons
    nk_size_t i = 0;
    for (; i + 8 <= block_count; i += 8) {
        __m256 v_f32x8 = _mm256_loadu_ps(block + i);
        nan_accumulator_f32x8 = _mm256_or_ps(nan_accumulator_f32x8, _mm256_cmp_ps(v_f32x8, v_f32x8, _CMP_UNORD_Q));
        amax_f32x8 = _mm256_max_ps(amax_f32x8, _mm256_and_ps(v_f32x8, abs_mask_f32x8));
    }
    if (_mm256_movemask_ps(nan_accumulator_f32x8)) return qnan.f;

    // Horizontal max of the 8 lanes: fold high half into low, then shuffle-reduce.
    __m128 low_f32x4 = _mm256_castps256_ps128(amax_f32x8);
    __m128 high_f32x4 = _mm256_extractf128_ps(amax_f32x8, 1);
    __m128 max_f32x4 = _mm_max_ps(low_f32x4, high_f32x4);
    max_f32x4 = _mm_max_ps(max_f32x4, _mm_movehl_ps(max_f32x4, max_f32x4));
    max_f32x4 = _mm_max_ss(max_f32x4, _mm_shuffle_ps(max_f32x4, max_f32x4, 0x1));
    nk_f32_t amax = _mm_cvtss_f32(max_f32x4);

    // Scalar tail (block_count not a multiple of 8).
    for (; i < block_count; ++i) {
        nk_f32_t x = block[i];
        if (x != x) return qnan.f;
        nk_f32_t a = x < 0 ? -x : x;
        if (a > amax) amax = a;
    }
    return amax;
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

    // E2M1 (4-bit) ↔ f32 — sub-byte, so it rides a dedicated 8-wide loop instead of the byte-stride
    // hub; only the f32 peer is vectorised here (matches the block-scaled element codec). Any other
    // e2m1 pairing falls through to the unsupported-type check below and routes to serial.
    if ((from_type == nk_e2m1_k && to_type == nk_f32_k) || (from_type == nk_f32_k && to_type == nk_e2m1_k)) {
        // 8 e2m1 = 4 bytes; f32 steps 32 bytes. Process whole groups of 8 with SIMD.
        nk_size_t from_step = nk_size_divide_round_up_(8 * nk_dtype_bits(from_type), NK_BITS_PER_BYTE);
        nk_size_t to_step = nk_size_divide_round_up_(8 * nk_dtype_bits(to_type), NK_BITS_PER_BYTE);
        nk_u8_t const *from_ptr = (nk_u8_t const *)from;
        nk_u8_t *to_ptr = (nk_u8_t *)to;
        nk_size_t batches = n / 8;
        for (nk_size_t i = 0; i < batches; ++i, from_ptr += from_step, to_ptr += to_step) {
            __m256 value_f32x8;
            if (from_type == nk_e2m1_k)
                value_f32x8 = nk_e2m1x8_to_f32x8_haswell_(_mm_cvtsi32_si128(*(int const *)from_ptr));
            else value_f32x8 = _mm256_loadu_ps((float const *)from_ptr);
            if (to_type == nk_e2m1_k) *(int *)to_ptr = _mm_cvtsi128_si32(nk_f32x8_to_e2m1x8_haswell_(value_f32x8));
            else _mm256_storeu_ps((float *)to_ptr, value_f32x8);
        }
        // Tail (< 8): delegate to serial so packed-nibble writes match the serial reference byte-for-byte.
        nk_size_t tail = n % 8;
        if (tail) nk_cast_serial(from_ptr, from_type, tail, to_ptr, to_type);
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
    nk_size_t from_step = nk_size_divide_round_up_(8 * nk_dtype_bits(from_type), NK_BITS_PER_BYTE);
    nk_size_t to_step = nk_size_divide_round_up_(8 * nk_dtype_bits(to_type), NK_BITS_PER_BYTE);

    nk_u8_t const *from_ptr = (nk_u8_t const *)from;
    nk_u8_t *to_ptr = (nk_u8_t *)to;
    nk_size_t batches = n / 8;
    nk_size_t tail = n % 8;
    nk_b256_vec_t hub;

    for (nk_size_t i = 0; i < batches; ++i, from_ptr += from_step, to_ptr += to_step) {
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

/** @brief Build an AVX2 lane mask selecting the low @p valid (≤ 8) f32 lanes. */
NK_INTERNAL __m256i nk_lane_mask_f32x8_haswell_(nk_size_t valid) {
    __m256i index_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    return _mm256_cmpgt_epi32(_mm256_set1_epi32((int)valid), index_i32x8);
}

/** @brief Haswell-optimised block-scaled cast. Uses AVX2 amax + broadcast reciprocal multiply
 *  around the serial element codec hub, mirroring `nk_cast_block_scaled_skylake` at x8 width. */
NK_PUBLIC void nk_cast_block_scaled_haswell(                                                                   //
    void const *from, void const *from_scales, nk_scalar_buffer_t const *from_tensor_scale,                    //
    nk_block_scaled_format_t const *from_format,                                                               //
    void *to, void *to_scales, nk_scalar_buffer_t *to_tensor_scale, nk_block_scaled_format_t const *to_format, //
    nk_size_t count) {

    int from_plain = (from_format->scale_dtype == nk_dtype_unknown_k || from_format->block_size == 0);
    int to_plain = (to_format->scale_dtype == nk_dtype_unknown_k || to_format->block_size == 0);

    if (from_plain && to_plain) {
        nk_cast_haswell(from, from_format->element_dtype, count, to, to_format->element_dtype);
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
            nk_cast_haswell(src, from_format->element_dtype, chunk_count, scratch, nk_f32_k);
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
                nk_cast_haswell(src, from_format->element_dtype, valid, scratch + b, nk_f32_k);
                __m256 scale_bcast_f32x8 = _mm256_set1_ps(scale_f32);
                for (nk_size_t e = 0; e < valid; e += 8) {
                    nk_size_t lanes = (valid - e) < 8 ? (valid - e) : 8;
                    __m256i mask_b32x8 = nk_lane_mask_f32x8_haswell_(lanes);
                    __m256 v_f32x8 = _mm256_maskload_ps(scratch + b + e, mask_b32x8);
                    _mm256_maskstore_ps(scratch + b + e, mask_b32x8, _mm256_mul_ps(v_f32x8, scale_bcast_f32x8));
                }
            }
        }

        // Encode f32 scratch into destination chunk.
        if (to_plain) {
            void *dst = (nk_u8_t *)to + (chunk_start * to_bits_per_element / NK_BITS_PER_BYTE);
            nk_cast_haswell(scratch, nk_f32_k, chunk_count, dst, to_format->element_dtype);
        }
        else {
            nk_f32_t element_max = nk_element_max_representable_(to_format->element_dtype);
            for (nk_size_t b = 0; b < chunk_count; b += to_block) {
                nk_size_t valid = (chunk_count - b) < to_block ? (chunk_count - b) : to_block;
                nk_f32_t block_amax = nk_block_amax_f32_haswell_(scratch + b, valid);
                nk_u8_t raw = nk_block_scaled_encode_scale_serial_(block_amax, element_max, to_tensor_scale_f32,
                                                                   to_format->scale_dtype);
                nk_size_t block_idx = (chunk_start + b) / to_block;
                to_scales_bytes[block_idx] = raw;
                nk_f32_t effective_scale = nk_block_scaled_decode_scale_serial_(raw, to_format->scale_dtype) *
                                           to_tensor_scale_f32;
                nk_f32_t reciprocal = effective_scale > 0 ? (1.0f / effective_scale) : 0.0f;
                __m256 reciprocal_bcast_f32x8 = _mm256_set1_ps(reciprocal);
                nk_f32_t encoded_scratch[32];
                for (nk_size_t e = 0; e < valid; e += 8) {
                    nk_size_t lanes = (valid - e) < 8 ? (valid - e) : 8;
                    __m256i mask_b32x8 = nk_lane_mask_f32x8_haswell_(lanes);
                    __m256 v_f32x8 = _mm256_maskload_ps(scratch + b + e, mask_b32x8);
                    _mm256_maskstore_ps(encoded_scratch + e, mask_b32x8,
                                        _mm256_mul_ps(v_f32x8, reciprocal_bcast_f32x8));
                }
                void *dst = (nk_u8_t *)to + ((chunk_start + b) * to_bits_per_element / NK_BITS_PER_BYTE);
                // Write only valid elements: dst is sized for `count` (bytes), not whole blocks.
                /* Saturate to element_max: finite inputs must not overflow to +/-inf (E5M2 has inf; OCP SAT). */
                for (nk_size_t saturate_index = 0; saturate_index < valid; ++saturate_index) {
                    if (encoded_scratch[saturate_index] > element_max) encoded_scratch[saturate_index] = element_max;
                    else if (encoded_scratch[saturate_index] < -element_max)
                        encoded_scratch[saturate_index] = -element_max;
                }
                nk_cast_haswell(encoded_scratch, nk_f32_k, valid, dst, to_format->element_dtype);
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

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X8664_
#endif // NK_CAST_HASWELL_H
