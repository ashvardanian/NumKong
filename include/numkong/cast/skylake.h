/**
 *  @brief SIMD-accelerated Type Conversions for Skylake.
 *  @file include/numkong/cast/skylake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/cast.h
 *
 *  @section skylake_cast_instructions AVX-512 Conversion Instructions
 *
 *      Intrinsic              Instruction                SKL        ICL        Genoa
 *      _mm512_cvtph_ps        VCVTPH2PS (ZMM, YMM)       5cy @ p05  5cy @ p05  4cy @ p01
 *      _mm512_cvtps_ph        VCVTPS2PH (YMM, ZMM, imm)  5cy @ p05  5cy @ p05  4cy @ p01
 *      _mm512_cvtps_epi32     VCVTPS2DQ (ZMM, ZMM)       4cy @ p0   4cy @ p0   3cy @ p01
 *      _mm512_cvtepi32_ps     VCVTDQ2PS (ZMM, ZMM)       4cy @ p0   4cy @ p0   3cy @ p01
 *      _mm512_cvtepi32_epi16  VPMOVDW (YMM, ZMM)         3cy @ p5   3cy @ p5   2cy @ p12
 *      _mm512_cvtsepi32_epi8  VPMOVSDB (XMM, ZMM)        3cy @ p5   3cy @ p5   2cy @ p12
 *
 *  F16 conversions use hardware F16C via VCVTPH2PS/VCVTPS2PH. BF16 lacks hardware support on Skylake,
 *  requiring emulation via VPMOVZXWD + VPSLLD for bf16-to-f32, achieving ~4cy total. FP8 (E4M3/E5M2)
 *  conversions use bit manipulation with VPTERNLOGD for sign/exp/mantissa composition.
 */
#ifndef NK_CAST_SKYLAKE_H
#define NK_CAST_SKYLAKE_H

#if NK_TARGET_X8664_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
#include "numkong/cast/serial.h" // `nk_dtype_bits`

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

#pragma region Type Punned Loads and Stores

/** @brief Type-agnostic 512-bit full load (Skylake AVX-512). */
NK_INTERNAL void nk_load_b512_skylake_(void const *src, nk_b512_vec_t *dst) { dst->zmm = _mm512_loadu_si512(src); }

/** @brief Type-agnostic partial load for 64-bit elements (8 elements max) into 512-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_b64x8_skylake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)n);
    dst->zmm = _mm512_maskz_loadu_epi64(mask, src);
}

/** @brief Type-agnostic partial load for 32-bit elements (16 elements max) into 512-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_b32x16_skylake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)n);
    dst->zmm = _mm512_maskz_loadu_epi32(mask, src);
}

/** @brief Type-agnostic partial load for 16-bit elements (32 elements max) into 512-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_b16x32_skylake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
    dst->zmm = _mm512_maskz_loadu_epi16(mask, src);
}

/** @brief Partial load for 8-bit elements (64 max) into 512-bit vector (zeros in remaining slots). */
NK_INTERNAL void nk_partial_load_b8x64_skylake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    __mmask64 mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)n);
    dst->zmm = _mm512_maskz_loadu_epi8(mask, src);
}

/** @brief Partial load for 4-bit nibbles (128 max = 64 bytes) into 512-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_b4x128_skylake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n, 2);
    __mmask64 mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)n_bytes);
    dst->zmm = _mm512_maskz_loadu_epi8(mask, src);
}

/** @brief Type-agnostic partial load for 32-bit elements (8 elements max) into 256-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_b32x8_skylake_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)n);
    dst->ymm = _mm256_maskz_loadu_epi32(mask, src);
}

/** @brief Type-agnostic partial load for 16-bit elements (16 elements max) into 256-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_b16x16_skylake_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)n);
    dst->ymm = _mm256_maskz_loadu_epi16(mask, src);
}

/** @brief Type-agnostic partial load for 8-bit elements (16 elements max) into 128-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_b8x16_skylake_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)n);
    dst->xmm = _mm_maskz_loadu_epi8(mask, src);
}

/** @brief Partial load for 1-bit elements (512 max bits = 64 bytes) into 512-bit vector (Skylake AVX-512).
 *  Wrapper that converts bit count to byte count and delegates to byte-level masked load. */
NK_INTERNAL void nk_partial_load_b1x512_skylake_(void const *src, nk_b512_vec_t *dst, nk_size_t n_bits) {
    nk_size_t n_bytes = nk_size_divide_round_up_(n_bits, 8);
    nk_partial_load_b8x64_skylake_(src, dst, n_bytes);
}

/** @brief Type-agnostic partial load for 32-bit elements (4 elements max) into 128-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_b32x4_skylake_(void const *src, nk_b128_vec_t *dst, nk_size_t n) {
    __mmask8 mask = (__mmask8)_bzhi_u32(0xF, (unsigned int)n);
    dst->xmm = _mm_maskz_loadu_epi32(mask, src);
}

/** @brief Type-agnostic partial load for 64-bit elements (4 elements max) into 256-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_b64x4_skylake_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    __mmask8 mask = (__mmask8)_bzhi_u32(0xF, (unsigned int)n);
    dst->ymm = _mm256_maskz_loadu_epi64(mask, src);
}

/** @brief Type-agnostic partial store for 32-bit elements (16 elements max) from 512-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_store_b32x16_skylake_(nk_b512_vec_t const *src, void *dst, nk_size_t n) {
    __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)n);
    _mm512_mask_storeu_epi32(dst, mask, src->zmm);
}

/** @brief Type-agnostic partial store for 32-bit elements (4 elements max) from 128-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_store_b32x4_skylake_(nk_b128_vec_t const *src, void *dst, nk_size_t n) {
    __mmask8 mask = (__mmask8)_bzhi_u32(0xF, (unsigned int)n);
    _mm_mask_storeu_epi32(dst, mask, src->xmm);
}

/** @brief Type-agnostic partial store for 64-bit elements (4 elements max) from 256-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_store_b64x4_skylake_(nk_b256_vec_t const *src, void *dst, nk_size_t n) {
    __mmask8 mask = (__mmask8)_bzhi_u32(0xF, (unsigned int)n);
    _mm256_mask_storeu_epi64(dst, mask, src->ymm);
}

/** @brief Type-agnostic full store for 512-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_store_b512_skylake_(nk_b512_vec_t const *src, void *dst) {
    _mm512_storeu_si512((__m512i *)dst, src->zmm);
}

/** @brief Type-agnostic partial store for 16-bit elements (32 elements max) from 512-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_store_b16x32_skylake_(nk_b512_vec_t const *src, void *dst, nk_size_t n) {
    __mmask32 mask = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned int)n);
    _mm512_mask_storeu_epi16(dst, mask, src->zmm);
}

/** @brief Type-agnostic partial store for 8-bit elements (64 elements max) from 512-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_store_b8x64_skylake_(nk_b512_vec_t const *src, void *dst, nk_size_t n) {
    __mmask64 mask = _bzhi_u64(0xFFFFFFFFFFFFFFFFULL, (unsigned int)n);
    _mm512_mask_storeu_epi8(dst, mask, src->zmm);
}

/** @brief Type-agnostic partial store for 64-bit elements (8 elements max) from 512-bit vector (Skylake AVX-512). */
NK_INTERNAL void nk_partial_store_b64x8_skylake_(nk_b512_vec_t const *src, void *dst, nk_size_t n) {
    __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)n);
    _mm512_mask_storeu_epi64(dst, mask, src->zmm);
}

#pragma endregion Type Punned Loads and Stores

#pragma region Vectorized Conversions

/** @brief Convert 16x bf16 → 16x f32 (Skylake AVX-512). */
NK_INTERNAL __m512 nk_bf16x16_to_f32x16_skylake_(__m256i a) {
    // Upcasting from `bf16` to `f32` is done by shifting the `bf16` values by 16 bits to the left, like:
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(a), 16));
}

/** @brief Convert 16x f32 → 16x bf16 (Skylake AVX-512). */
NK_INTERNAL __m256i nk_f32x16_to_bf16x16_skylake_(__m512 a) {
    // Round-to-nearest-even: add (0x7FFF + lsb) to match hardware BF16 behavior
    __m512i bits = _mm512_castps_si512(a);
    __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
    __m512i rounded = _mm512_add_epi32(bits, _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb));
    __m512i x = _mm512_srli_epi32(rounded, 16);
    return _mm512_cvtepi32_epi16(x);
}

/** @brief Convert 16x e4m3 → 16x f32 via Giesen-style fake-F16 cast (AVX-512 + F16C).
 *  E4M3 `byte = S EEEE MMM` (bias 7). Shifting the magnitude into F16 positions
 *  `((byte & 0x7F) << 7) | ((byte & 0x80) << 8)` yields a fake F16 whose F16 value
 *  differs from the true E4M3 magnitude by exactly 2⁸ (bias delta 15 − 7). The
 *  fake F16 is widened via `vcvtph2ps` and corrected by ×256 in F32. Subnormal
 *  handling falls out for free via F16 subnormal semantics. NaN (|byte|==0x7F) is
 *  the sole E4M3 special value that would misinterpret as finite; blended
 *  explicitly with F32 quiet NaN bits. */
NK_INTERNAL __m512 nk_e4m3x16_to_f32x16_skylake_(__m128i e4m3_i8x16) {
    __m256i const magnitude_mask_u16x16 = _mm256_set1_epi16(0x7F);
    __m256i const sign_mask_u16x16 = _mm256_set1_epi16((short)0x80);
    __m256i const f16_nan_u16x16 = _mm256_set1_epi16(0x7E00);
    __m256i word_u16x16 = _mm256_cvtepu8_epi16(e4m3_i8x16);
    __m256i magnitude_u16x16 = _mm256_and_si256(word_u16x16, magnitude_mask_u16x16);
    __mmask16 is_nan = _mm256_cmpeq_epi16_mask(magnitude_u16x16, magnitude_mask_u16x16);
    __m256i shifted_magnitude_u16x16 = _mm256_slli_epi16(magnitude_u16x16, 7);
    __m256i shifted_sign_u16x16 = _mm256_slli_epi16(_mm256_and_si256(word_u16x16, sign_mask_u16x16), 8);
    __m256i f16_bits_u16x16 = _mm256_or_si256(shifted_magnitude_u16x16, shifted_sign_u16x16);
    f16_bits_u16x16 = _mm256_mask_mov_epi16(f16_bits_u16x16, is_nan, f16_nan_u16x16);
    __m512 fake_f32x16 = _mm512_cvtph_ps(f16_bits_u16x16);
    return _mm512_mul_ps(fake_f32x16, _mm512_set1_ps(256.0f));
}

/** @brief Convert 16x e4m3 → 16x f16 via arithmetic + 8-entry subnormal LUT (AVX-512BW + AVX-512VL).
 *  E4M3: S EEEE MMM (bias=7). F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Normal (exp != 0): F16 = ((lower7 << 7) + 0x2000) | (sign << 8) — bias delta 8 added at the
 *  exp-position (8 << 10 = 0x2000) after placing magnitude bits at F16 positions 13..7.
 *  Subnormal (exp == 0): looked up from 8-entry F16 LUT — values 0, 1/512, 2/512, …, 7/512 encoded as
 *  F16 normals (the smallest E4M3 subnormal 1/512 = 2⁻⁹ is well within F16 normal range).
 *  NaN (|byte| == 0x7F): blended in as F16 quiet NaN with original sign. */
NK_INTERNAL __m256i nk_e4m3x16_to_f16x16_skylake_(__m128i e4m3_u8x16) {
    __m256i e4m3_i16x16 = _mm256_cvtepu8_epi16(e4m3_u8x16);
    __m256i sign_i16x16 = _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi16((short)0x80));
    __m256i lower7_i16x16 = _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi16(0x7F));
    __m256i normal_abs_i16x16 = _mm256_add_epi16(_mm256_slli_epi16(lower7_i16x16, 7), _mm256_set1_epi16(0x2000));
    __m256i subn_lut_i16x16 = _mm256_set_epi16( //
        0x2300, 0x2200, 0x2100, 0x2000, 0x1E00, 0x1C00, 0x1800, 0x0000, 0x2300, 0x2200, 0x2100, 0x2000, 0x1E00, 0x1C00,
        0x1800, 0x0000);
    __m256i mant_idx_i16x16 = _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi16(0x07));
    __m256i subn_abs_i16x16 = _mm256_permutexvar_epi16(mant_idx_i16x16, subn_lut_i16x16);
    __mmask16 is_subnormal = _mm256_testn_epi16_mask(e4m3_i16x16, _mm256_set1_epi16(0x78));
    __m256i abs_i16x16 = _mm256_mask_blend_epi16(is_subnormal, normal_abs_i16x16, subn_abs_i16x16);
    __m256i shifted_sign_i16x16 = _mm256_slli_epi16(sign_i16x16, 8);
    __m256i result_i16x16 = _mm256_or_si256(abs_i16x16, shifted_sign_i16x16);
    __mmask16 is_nan = _mm256_cmpeq_epi16_mask(lower7_i16x16, _mm256_set1_epi16(0x7F));
    __m256i nan_i16x16 = _mm256_or_si256(shifted_sign_i16x16, _mm256_set1_epi16(0x7E00));
    return _mm256_mask_blend_epi16(is_nan, result_i16x16, nan_i16x16);
}

/** @brief Convert 16x e5m2 → 16x f32 via free-shift widen (AVX-512 + F16C).
 *  E5M2 shares F16's exponent bias (15): `(byte << 8)` is the matching F16 bit
 *  pattern for every E5M2 value (normals, subnormals, zero, ±Inf, NaN — all
 *  bit-exact). Widen u8 → u16, shift, then VCVTPH2PS to F32. Three ops total. */
NK_INTERNAL __m512 nk_e5m2x16_to_f32x16_skylake_(__m128i e5m2_i8x16) {
    __m256i e5m2_u16x16 = _mm256_cvtepu8_epi16(e5m2_i8x16);
    __m256i f16_bits_u16x16 = _mm256_slli_epi16(e5m2_u16x16, 8);
    return _mm512_cvtph_ps(f16_bits_u16x16);
}

/** @brief Convert 16x e2m3 → 16x f32 via bit manipulation (AVX-512).
 *  E2M3 format: S EE MMM (bias=1, only 6 bits used). F32: sign<<31, (exp+126)<<23, mantissa<<20.
 *  Subnormals (exp=0): value = mantissa × 2⁽¹⁻¹⁾ × 2⁻³ = mantissa ÷ 8. */
NK_INTERNAL __m512 nk_e2m3x16_to_f32x16_skylake_(__m128i e2m3_i8x16) {
    __m512i e2m3_i32x16 = _mm512_cvtepu8_epi32(e2m3_i8x16);

    // Extract fields (only 6 bits used: S EE MMM)
    __m512i exp_i32x16 = _mm512_and_si512(_mm512_srli_epi32(e2m3_i32x16, 3), _mm512_set1_epi32(0x03));
    __m512i mantissa_i32x16 = _mm512_and_si512(e2m3_i32x16, _mm512_set1_epi32(0x07));
    __m512i sign_i32x16 = _mm512_slli_epi32(_mm512_srli_epi32(e2m3_i32x16, 5), 31);

    // Normal path: sign | ((exp+126)<<23) | (mantissa<<20)
    __m512i f32_exp_i32x16 = _mm512_slli_epi32(_mm512_add_epi32(exp_i32x16, _mm512_set1_epi32(126)), 23);
    __m512i f32_mantissa_i32x16 = _mm512_slli_epi32(mantissa_i32x16, 20);
    __m512 result_f32x16 = _mm512_castsi512_ps(
        _mm512_ternarylogic_epi32(sign_i32x16, f32_exp_i32x16, f32_mantissa_i32x16, 0xFE));

    // Subnormal fix: for exp==0 lanes, replace with (mantissa / 8) | sign using masked OR
    __mmask16 is_subnormal = _mm512_testn_epi32_mask(e2m3_i32x16, _mm512_set1_epi32(0x18));
    __m512 subnorm_abs_f32x16 = _mm512_mul_ps(_mm512_cvtepi32_ps(mantissa_i32x16), _mm512_set1_ps(1.0f / 8.0f));
    return _mm512_mask_or_ps(result_f32x16, is_subnormal, subnorm_abs_f32x16, _mm512_castsi512_ps(sign_i32x16));
}

/** @brief Convert 16x e3m2 → 16x f32 via bit manipulation (AVX-512).
 *  E3M2 format: S EEE MM (bias=3, only 6 bits used). F32: sign<<31, (exp+124)<<23, mantissa<<21.
 *  Subnormals (exp=0): value = mantissa × 2⁽¹⁻³⁾ × 2⁻² = mantissa ÷ 16. */
NK_INTERNAL __m512 nk_e3m2x16_to_f32x16_skylake_(__m128i e3m2_i8x16) {
    __m512i e3m2_i32x16 = _mm512_cvtepu8_epi32(e3m2_i8x16);

    // Extract fields (only 6 bits used: S EEE MM)
    __m512i exp_i32x16 = _mm512_and_si512(_mm512_srli_epi32(e3m2_i32x16, 2), _mm512_set1_epi32(0x07));
    __m512i mantissa_i32x16 = _mm512_and_si512(e3m2_i32x16, _mm512_set1_epi32(0x03));
    __m512i sign_i32x16 = _mm512_slli_epi32(_mm512_srli_epi32(e3m2_i32x16, 5), 31);

    // Normal path: sign | ((exp+124)<<23) | (mantissa<<21)
    __m512i f32_exp_i32x16 = _mm512_slli_epi32(_mm512_add_epi32(exp_i32x16, _mm512_set1_epi32(124)), 23);
    __m512i f32_mantissa_i32x16 = _mm512_slli_epi32(mantissa_i32x16, 21);
    __m512 result_f32x16 = _mm512_castsi512_ps(
        _mm512_ternarylogic_epi32(sign_i32x16, f32_exp_i32x16, f32_mantissa_i32x16, 0xFE));

    // Subnormal fix: for exp==0 lanes, replace with (mantissa / 16) | sign using masked OR
    __mmask16 is_subnormal = _mm512_testn_epi32_mask(e3m2_i32x16, _mm512_set1_epi32(0x1C));
    __m512 subnorm_abs_f32x16 = _mm512_mul_ps(_mm512_cvtepi32_ps(mantissa_i32x16), _mm512_set1_ps(1.0f / 16.0f));
    return _mm512_mask_or_ps(result_f32x16, is_subnormal, subnorm_abs_f32x16, _mm512_castsi512_ps(sign_i32x16));
}

/** @brief Convert 16x f32 → 16x e2m3 via bit manipulation (AVX-512).
 *  E2M3 format: S EE MMM (bias=1). Handles normal, subnormal, and overflow cases.
 *  Subnormals (f32_exp ≤ 126): mantissa = round(abs_f32 * 8), clamped to [0,7]. */
NK_INTERNAL __m128i nk_f32x16_to_e2m3x16_skylake_(__m512 f32x16) {
    __m512i bits_i32x16 = _mm512_castps_si512(f32x16);
    __m512i sign_i32x16 = _mm512_srli_epi32(bits_i32x16, 31);
    __m512i f32_exp_i32x16 = _mm512_and_si512(_mm512_srli_epi32(bits_i32x16, 23), _mm512_set1_epi32(0xFF));

    // Round mantissa from 23 to 3 bits using RNE (round to nearest, ties to even)
    __m512i significand_i32x16 = _mm512_or_si512(_mm512_and_si512(bits_i32x16, _mm512_set1_epi32(0x007FFFFF)),
                                                 _mm512_set1_epi32(0x00800000)); // (a & mask) | implicit_one
    __m512i lsb_i32x16 = _mm512_and_si512(_mm512_srli_epi32(significand_i32x16, 20), _mm512_set1_epi32(1));
    __m512i rounding_bias_i32x16 = _mm512_add_epi32(_mm512_set1_epi32(0x0007FFFF), lsb_i32x16);
    __m512i rounded_sig_i32x16 = _mm512_add_epi32(significand_i32x16, rounding_bias_i32x16);
    __m512i carry_i32x16 = _mm512_srli_epi32(rounded_sig_i32x16, 24); // Carry into exponent if bit 24 set
    __m512i f32_mantissa_i32x16 = _mm512_and_si512(_mm512_srli_epi32(rounded_sig_i32x16, 20), _mm512_set1_epi32(0x07));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    f32_mantissa_i32x16 = _mm512_andnot_si512(_mm512_slli_epi32(carry_i32x16, 31), f32_mantissa_i32x16);
    __m512i e2m3_exp_i32x16 = _mm512_sub_epi32(_mm512_add_epi32(f32_exp_i32x16, carry_i32x16), _mm512_set1_epi32(126));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 3)
    __mmask16 is_subnormal = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(1), e2m3_exp_i32x16);
    __mmask16 overflow = _mm512_cmpgt_epi32_mask(e2m3_exp_i32x16, _mm512_set1_epi32(3));

    // Normal path: clamp exp to [1,3], extract mantissa bits
    __m512i clamped_exp_i32x16 = _mm512_max_epi32(e2m3_exp_i32x16, _mm512_set1_epi32(1));
    clamped_exp_i32x16 = _mm512_min_epi32(clamped_exp_i32x16, _mm512_set1_epi32(3));
    __m512i normal_mantissa_i32x16 = _mm512_mask_blend_epi32(overflow, f32_mantissa_i32x16, _mm512_set1_epi32(0x07));
    __m512i normal_e2m3_i32x16 = _mm512_ternarylogic_epi32(_mm512_slli_epi32(sign_i32x16, 5),
                                                           _mm512_slli_epi32(clamped_exp_i32x16, 3),
                                                           normal_mantissa_i32x16, 0xFE); // a | b | c

    // Subnormal path: mantissa = round(abs_f32 * 8)
    // If mantissa rounds to 8 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x08
    __m512 abs_f32x16 = _mm512_and_ps(f32x16, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 scaled_f32x16 = _mm512_mul_ps(abs_f32x16, _mm512_set1_ps(8.0f));
    __m512i subnorm_mantissa_i32x16 = _mm512_cvtps_epi32(scaled_f32x16);
    __mmask16 promotes_to_normal = _mm512_cmpgt_epi32_mask(subnorm_mantissa_i32x16, _mm512_set1_epi32(7));
    subnorm_mantissa_i32x16 = _mm512_min_epi32(subnorm_mantissa_i32x16, _mm512_set1_epi32(7));
    subnorm_mantissa_i32x16 = _mm512_max_epi32(subnorm_mantissa_i32x16, _mm512_setzero_si512());
    __m512i subnorm_e2m3_i32x16 = _mm512_or_si512(_mm512_slli_epi32(sign_i32x16, 5), subnorm_mantissa_i32x16);
    // When mantissa rounds to 8, use first normal value (0x08) instead of clamped subnormal
    __m512i first_normal_e2m3_i32x16 = _mm512_or_si512(_mm512_slli_epi32(sign_i32x16, 5), _mm512_set1_epi32(0x08));
    subnorm_e2m3_i32x16 = _mm512_mask_blend_epi32(promotes_to_normal, subnorm_e2m3_i32x16, first_normal_e2m3_i32x16);

    // Blend: use subnormal result when exp <= 0, else normal
    __m512i e2m3_i32x16 = _mm512_mask_blend_epi32(is_subnormal, normal_e2m3_i32x16, subnorm_e2m3_i32x16);

    // Pack 16 i32s to 16 unsigned i8s via AVX-512 cvtepi32_epi8
    return _mm512_cvtepi32_epi8(e2m3_i32x16);
}

/** @brief Convert 16x f32 → 16x e3m2 via bit manipulation (AVX-512).
 *  E3M2 format: S EEE MM (bias=3). Handles normal, subnormal, and overflow cases.
 *  Subnormals (f32_exp ≤ 124): mantissa = round(abs_f32 * 16), clamped to [0,3]. */
NK_INTERNAL __m128i nk_f32x16_to_e3m2x16_skylake_(__m512 f32x16) {
    __m512i bits_i32x16 = _mm512_castps_si512(f32x16);
    __m512i sign_i32x16 = _mm512_srli_epi32(bits_i32x16, 31);
    __m512i f32_exp_i32x16 = _mm512_and_si512(_mm512_srli_epi32(bits_i32x16, 23), _mm512_set1_epi32(0xFF));

    // Round mantissa from 23 to 2 bits using RNE (round to nearest, ties to even)
    __m512i significand_i32x16 = _mm512_or_si512(_mm512_and_si512(bits_i32x16, _mm512_set1_epi32(0x007FFFFF)),
                                                 _mm512_set1_epi32(0x00800000)); // (a & mask) | implicit_one
    __m512i lsb_i32x16 = _mm512_and_si512(_mm512_srli_epi32(significand_i32x16, 21), _mm512_set1_epi32(1));
    __m512i rounding_bias_i32x16 = _mm512_add_epi32(_mm512_set1_epi32(0x000FFFFF), lsb_i32x16);
    __m512i rounded_sig_i32x16 = _mm512_add_epi32(significand_i32x16, rounding_bias_i32x16);
    __m512i carry_i32x16 = _mm512_srli_epi32(rounded_sig_i32x16, 24); // Carry into exponent if bit 24 set
    __m512i f32_mantissa_i32x16 = _mm512_and_si512(_mm512_srli_epi32(rounded_sig_i32x16, 21), _mm512_set1_epi32(0x03));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    f32_mantissa_i32x16 = _mm512_andnot_si512(_mm512_slli_epi32(carry_i32x16, 31), f32_mantissa_i32x16);
    __m512i e3m2_exp_i32x16 = _mm512_sub_epi32(_mm512_add_epi32(f32_exp_i32x16, carry_i32x16), _mm512_set1_epi32(124));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 7)
    __mmask16 is_subnormal = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(1), e3m2_exp_i32x16);
    __mmask16 overflow = _mm512_cmpgt_epi32_mask(e3m2_exp_i32x16, _mm512_set1_epi32(7));

    // Normal path: clamp exp to [1,7], extract mantissa bits
    __m512i clamped_exp_i32x16 = _mm512_max_epi32(e3m2_exp_i32x16, _mm512_set1_epi32(1));
    clamped_exp_i32x16 = _mm512_min_epi32(clamped_exp_i32x16, _mm512_set1_epi32(7));
    __m512i normal_mantissa_i32x16 = _mm512_mask_blend_epi32(overflow, f32_mantissa_i32x16, _mm512_set1_epi32(0x03));
    __m512i normal_e3m2_i32x16 = _mm512_ternarylogic_epi32(_mm512_slli_epi32(sign_i32x16, 5),
                                                           _mm512_slli_epi32(clamped_exp_i32x16, 2),
                                                           normal_mantissa_i32x16, 0xFE); // a | b | c

    // Subnormal path: mantissa = round(abs_f32 * 16)
    // If mantissa rounds to 4 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x04
    __m512 abs_f32x16 = _mm512_and_ps(f32x16, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 scaled_f32x16 = _mm512_mul_ps(abs_f32x16, _mm512_set1_ps(16.0f));
    __m512i subnorm_mantissa_i32x16 = _mm512_cvtps_epi32(scaled_f32x16);
    __mmask16 promotes_to_normal = _mm512_cmpgt_epi32_mask(subnorm_mantissa_i32x16, _mm512_set1_epi32(3));
    subnorm_mantissa_i32x16 = _mm512_min_epi32(subnorm_mantissa_i32x16, _mm512_set1_epi32(3));
    subnorm_mantissa_i32x16 = _mm512_max_epi32(subnorm_mantissa_i32x16, _mm512_setzero_si512());
    __m512i subnorm_e3m2_i32x16 = _mm512_or_si512(_mm512_slli_epi32(sign_i32x16, 5), subnorm_mantissa_i32x16);
    // When mantissa rounds to 4, use first normal value (0x04) instead of clamped subnormal
    __m512i first_normal_e3m2_i32x16 = _mm512_or_si512(_mm512_slli_epi32(sign_i32x16, 5), _mm512_set1_epi32(0x04));
    subnorm_e3m2_i32x16 = _mm512_mask_blend_epi32(promotes_to_normal, subnorm_e3m2_i32x16, first_normal_e3m2_i32x16);

    // Blend: use subnormal result when exp <= 0, else normal
    __m512i e3m2_i32x16 = _mm512_mask_blend_epi32(is_subnormal, normal_e3m2_i32x16, subnorm_e3m2_i32x16);

    // Pack 16 i32s to 16 unsigned i8s via AVX-512 cvtepi32_epi8
    return _mm512_cvtepi32_epi8(e3m2_i32x16);
}

/** @brief Convert 16x f32 → 16x e4m3 via bit manipulation (AVX-512).
 *  E4M3 format: S EEEE MMM (bias=7). Handles normal, subnormal, and overflow cases.
 *  Subnormals (f32_exp ≤ 120): mantissa = round(abs_f32 * 512), clamped to [0,7]. */
NK_INTERNAL __m128i nk_f32x16_to_e4m3x16_skylake_(__m512 f32x16) {
    __m512i bits_i32x16 = _mm512_castps_si512(f32x16);
    __m512i sign_i32x16 = _mm512_srli_epi32(bits_i32x16, 31);
    __m512i f32_exp_i32x16 = _mm512_and_si512(_mm512_srli_epi32(bits_i32x16, 23), _mm512_set1_epi32(0xFF));

    // Round mantissa from 23 to 3 bits using RNE (round to nearest, ties to even)
    // RNE trick: add (half - 1 + lsb) where lsb is the bit that will become the new lsb after shift
    __m512i significand_i32x16 = _mm512_or_si512(_mm512_and_si512(bits_i32x16, _mm512_set1_epi32(0x007FFFFF)),
                                                 _mm512_set1_epi32(0x00800000)); // (a & mask) | implicit_one
    __m512i lsb_i32x16 = _mm512_and_si512(_mm512_srli_epi32(significand_i32x16, 20), _mm512_set1_epi32(1));
    __m512i rounding_bias_i32x16 = _mm512_add_epi32(_mm512_set1_epi32(0x0007FFFF), lsb_i32x16);
    __m512i rounded_sig_i32x16 = _mm512_add_epi32(significand_i32x16, rounding_bias_i32x16);
    __m512i carry_i32x16 = _mm512_srli_epi32(rounded_sig_i32x16, 24); // Carry into exponent if bit 24 set
    __m512i f32_mantissa_i32x16 = _mm512_and_si512(_mm512_srli_epi32(rounded_sig_i32x16, 20), _mm512_set1_epi32(0x07));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    f32_mantissa_i32x16 = _mm512_andnot_si512(_mm512_slli_epi32(carry_i32x16, 31), f32_mantissa_i32x16);
    __m512i e4m3_exp_i32x16 = _mm512_sub_epi32(_mm512_add_epi32(f32_exp_i32x16, carry_i32x16), _mm512_set1_epi32(120));

    // Detect underflow (exp <= 0, maps to subnormal/zero) and overflow (exp > 15)
    __mmask16 is_subnormal = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(1), e4m3_exp_i32x16);
    __mmask16 overflow = _mm512_cmpgt_epi32_mask(e4m3_exp_i32x16, _mm512_set1_epi32(15));

    // Normal path: clamp exp to [1,15], extract mantissa bits
    // e4m3FN quirk: exp=15 with mantissa=7 is NaN (0x7F), so clamp mantissa to 6 when exp=15.
    __m512i clamped_exp_i32x16 = _mm512_max_epi32(e4m3_exp_i32x16, _mm512_set1_epi32(1));
    clamped_exp_i32x16 = _mm512_min_epi32(clamped_exp_i32x16, _mm512_set1_epi32(15));
    __mmask16 is_max_exp = _mm512_cmpeq_epi32_mask(clamped_exp_i32x16, _mm512_set1_epi32(15));
    __m512i max_mantissa_i32x16 = _mm512_mask_blend_epi32(is_max_exp, _mm512_set1_epi32(7), _mm512_set1_epi32(6));
    __m512i normal_mantissa_i32x16 = _mm512_min_epi32(f32_mantissa_i32x16, max_mantissa_i32x16);
    normal_mantissa_i32x16 = _mm512_mask_blend_epi32(overflow, normal_mantissa_i32x16, _mm512_set1_epi32(0x06));
    __m512i normal_e4m3_i32x16 = _mm512_ternarylogic_epi32(_mm512_slli_epi32(sign_i32x16, 7),
                                                           _mm512_slli_epi32(clamped_exp_i32x16, 3),
                                                           normal_mantissa_i32x16, 0xFE); // a | b | c

    // Subnormal path: mantissa = round(abs_f32 * 512)
    // If mantissa rounds to 8 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x08
    __m512 abs_f32x16 = _mm512_and_ps(f32x16, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 scaled_f32x16 = _mm512_mul_ps(abs_f32x16, _mm512_set1_ps(512.0f));
    __m512i subnorm_mantissa_i32x16 = _mm512_cvtps_epi32(scaled_f32x16);
    __mmask16 promotes_to_normal = _mm512_cmpgt_epi32_mask(subnorm_mantissa_i32x16, _mm512_set1_epi32(7));
    subnorm_mantissa_i32x16 = _mm512_min_epi32(subnorm_mantissa_i32x16, _mm512_set1_epi32(7));
    subnorm_mantissa_i32x16 = _mm512_max_epi32(subnorm_mantissa_i32x16, _mm512_setzero_si512());
    __m512i subnorm_e4m3_i32x16 = _mm512_or_si512(_mm512_slli_epi32(sign_i32x16, 7), subnorm_mantissa_i32x16);
    // When mantissa rounds to 8, use first normal value (0x08) instead of clamped subnormal
    __m512i first_normal_e4m3_i32x16 = _mm512_or_si512(_mm512_slli_epi32(sign_i32x16, 7), _mm512_set1_epi32(0x08));
    subnorm_e4m3_i32x16 = _mm512_mask_blend_epi32(promotes_to_normal, subnorm_e4m3_i32x16, first_normal_e4m3_i32x16);

    // Blend: use subnormal result when exp <= 0, else normal
    __m512i e4m3_i32x16 = _mm512_mask_blend_epi32(is_subnormal, normal_e4m3_i32x16, subnorm_e4m3_i32x16);

    // Pack 16 i32s to 16 unsigned i8s via AVX-512 cvtepi32_epi8
    return _mm512_cvtepi32_epi8(e4m3_i32x16);
}

/** @brief Convert 16x f32 → 16x e5m2 via bit manipulation (AVX-512).
 *  E5M2 format: S EEEEE MM (bias=15). Handles normal, subnormal, and overflow cases.
 *  Uses RNE (round to nearest even) for mantissa rounding. */
NK_INTERNAL __m128i nk_f32x16_to_e5m2x16_skylake_(__m512 f32x16) {
    __m512i bits_i32x16 = _mm512_castps_si512(f32x16);
    __m512i sign_i32x16 = _mm512_srli_epi32(bits_i32x16, 31);
    __m512i f32_exp_i32x16 = _mm512_and_si512(_mm512_srli_epi32(bits_i32x16, 23), _mm512_set1_epi32(0xFF));

    // Round mantissa from 23 to 2 bits using RNE (round to nearest, ties to even)
    // RNE trick: add (half - 1 + lsb) where lsb is the bit that will become the new lsb after shift
    __m512i significand_i32x16 = _mm512_or_si512(_mm512_and_si512(bits_i32x16, _mm512_set1_epi32(0x007FFFFF)),
                                                 _mm512_set1_epi32(0x00800000)); // (a & mask) | implicit_one
    __m512i lsb_i32x16 = _mm512_and_si512(_mm512_srli_epi32(significand_i32x16, 21), _mm512_set1_epi32(1));
    __m512i rounding_bias_i32x16 = _mm512_add_epi32(_mm512_set1_epi32(0x000FFFFF), lsb_i32x16); // half = 0x100000
    __m512i rounded_sig_i32x16 = _mm512_add_epi32(significand_i32x16, rounding_bias_i32x16);
    __m512i carry_i32x16 = _mm512_srli_epi32(rounded_sig_i32x16, 24); // Carry into exponent if bit 24 set
    __m512i f32_mantissa_i32x16 = _mm512_and_si512(_mm512_srli_epi32(rounded_sig_i32x16, 21), _mm512_set1_epi32(0x03));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    f32_mantissa_i32x16 = _mm512_andnot_si512(_mm512_slli_epi32(carry_i32x16, 31), f32_mantissa_i32x16);
    __m512i e5m2_exp_i32x16 = _mm512_sub_epi32(_mm512_add_epi32(f32_exp_i32x16, carry_i32x16), _mm512_set1_epi32(112));

    // Detect subnormal (exp <= 0) and overflow (exp > 31)
    __mmask16 is_subnormal = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(1), e5m2_exp_i32x16);
    __mmask16 overflow = _mm512_cmpgt_epi32_mask(e5m2_exp_i32x16, _mm512_set1_epi32(31));

    // Normal path: clamp exp to [1,31], on overflow return infinity (exp=31, mantissa=0 = 0x7C)
    __m512i clamped_exp_i32x16 = _mm512_max_epi32(e5m2_exp_i32x16, _mm512_set1_epi32(1));
    clamped_exp_i32x16 = _mm512_min_epi32(clamped_exp_i32x16, _mm512_set1_epi32(31));
    __m512i normal_mantissa_i32x16 = _mm512_mask_blend_epi32(overflow, f32_mantissa_i32x16, _mm512_setzero_si512());
    __m512i normal_e5m2_i32x16 = _mm512_ternarylogic_epi32(_mm512_slli_epi32(sign_i32x16, 7),
                                                           _mm512_slli_epi32(clamped_exp_i32x16, 2),
                                                           normal_mantissa_i32x16, 0xFE); // a | b | c

    // Subnormal path: mantissa = round(abs_f32 * 65536)
    // If mantissa rounds to 4 or higher, promote to first normal (exp_field=1, mantissa=0) = 0x04
    __m512 abs_f32x16 = _mm512_and_ps(f32x16, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 scaled_f32x16 = _mm512_mul_ps(abs_f32x16, _mm512_set1_ps(65536.0f));
    __m512i subnorm_mantissa_i32x16 = _mm512_cvtps_epi32(scaled_f32x16);
    __mmask16 promotes_to_normal = _mm512_cmpgt_epi32_mask(subnorm_mantissa_i32x16, _mm512_set1_epi32(3));
    subnorm_mantissa_i32x16 = _mm512_min_epi32(subnorm_mantissa_i32x16, _mm512_set1_epi32(3));
    subnorm_mantissa_i32x16 = _mm512_max_epi32(subnorm_mantissa_i32x16, _mm512_setzero_si512());
    __m512i subnorm_e5m2_i32x16 = _mm512_or_si512(_mm512_slli_epi32(sign_i32x16, 7), subnorm_mantissa_i32x16);
    // When mantissa rounds to 4, use first normal value (0x04) instead of clamped subnormal
    __m512i first_normal_e5m2_i32x16 = _mm512_or_si512(_mm512_slli_epi32(sign_i32x16, 7), _mm512_set1_epi32(0x04));
    subnorm_e5m2_i32x16 = _mm512_mask_blend_epi32(promotes_to_normal, subnorm_e5m2_i32x16, first_normal_e5m2_i32x16);

    // Blend: use subnormal result when exp <= 0
    __m512i e5m2_i32x16 = _mm512_mask_blend_epi32(is_subnormal, normal_e5m2_i32x16, subnorm_e5m2_i32x16);

    // Pack 16 i32s to 16 unsigned i8s via AVX-512 cvtepi32_epi8
    return _mm512_cvtepi32_epi8(e5m2_i32x16);
}

NK_INTERNAL __m512 nk_i8x16_to_f32x16_skylake_(__m128i i8x16) {
    return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(i8x16));
}
NK_INTERNAL __m512 nk_u8x16_to_f32x16_skylake_(__m128i u8x16) {
    return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u8x16));
}
NK_INTERNAL __m512 nk_i16x16_to_f32x16_skylake_(__m256i i16x16) {
    return _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(i16x16));
}
NK_INTERNAL __m512 nk_u16x16_to_f32x16_skylake_(__m256i u16x16) {
    return _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(u16x16));
}

NK_INTERNAL __m128i nk_f32x16_to_i8x16_skylake_(__m512 f32x16) {
    __m512 clamped = _mm512_min_ps(_mm512_max_ps(f32x16, _mm512_set1_ps(-128.0f)), _mm512_set1_ps(127.0f));
    return _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(clamped));
}
NK_INTERNAL __m128i nk_f32x16_to_u8x16_skylake_(__m512 f32x16) {
    __m512 clamped = _mm512_min_ps(_mm512_max_ps(f32x16, _mm512_setzero_ps()), _mm512_set1_ps(255.0f));
    return _mm512_cvtusepi32_epi8(_mm512_cvtps_epu32(clamped));
}
NK_INTERNAL __m256i nk_f32x16_to_i16x16_skylake_(__m512 f32x16) {
    __m512 clamped = _mm512_min_ps(_mm512_max_ps(f32x16, _mm512_set1_ps(-32768.0f)), _mm512_set1_ps(32767.0f));
    return _mm512_cvtsepi32_epi16(_mm512_cvtps_epi32(clamped));
}
NK_INTERNAL __m256i nk_f32x16_to_u16x16_skylake_(__m512 f32x16) {
    __m512 clamped = _mm512_min_ps(_mm512_max_ps(f32x16, _mm512_setzero_ps()), _mm512_set1_ps(65535.0f));
    return _mm512_cvtusepi32_epi16(_mm512_cvtps_epu32(clamped));
}

NK_INTERNAL __m512i nk_u8x8_to_u64x8_skylake_(__m128i u8x8) { return _mm512_cvtepu8_epi64(u8x8); }
NK_INTERNAL __m512i nk_u16x8_to_u64x8_skylake_(__m128i u16x8) { return _mm512_cvtepu16_epi64(u16x8); }
NK_INTERNAL __m512i nk_u32x8_to_u64x8_skylake_(__m256i u32x8) { return _mm512_cvtepu32_epi64(u32x8); }

NK_INTERNAL __m128i nk_u64x8_to_u8x8_skylake_(__m512i u64x8) {
    __m512i clamped = _mm512_min_epu64(u64x8, _mm512_set1_epi64(255));
    return _mm512_cvtepi64_epi8(clamped);
}
NK_INTERNAL __m128i nk_u64x8_to_u16x8_skylake_(__m512i u64x8) {
    __m512i clamped = _mm512_min_epu64(u64x8, _mm512_set1_epi64(65535));
    return _mm512_cvtepi64_epi16(clamped);
}
NK_INTERNAL __m256i nk_u64x8_to_u32x8_skylake_(__m512i u64x8) {
    __m512i clamped = _mm512_min_epu64(u64x8, _mm512_set1_epi64(0xFFFFFFFFULL));
    return _mm512_cvtepi64_epi32(clamped);
}

NK_INTERNAL __m512i nk_i8x8_to_i64x8_skylake_(__m128i i8x8) { return _mm512_cvtepi8_epi64(i8x8); }
NK_INTERNAL __m512i nk_i16x8_to_i64x8_skylake_(__m128i i16x8) { return _mm512_cvtepi16_epi64(i16x8); }
NK_INTERNAL __m512i nk_i32x8_to_i64x8_skylake_(__m256i i32x8) { return _mm512_cvtepi32_epi64(i32x8); }
NK_INTERNAL __m512i nk_u8x8_to_i64x8_skylake_(__m128i u8x8) { return _mm512_cvtepu8_epi64(u8x8); }
NK_INTERNAL __m512i nk_u16x8_to_i64x8_skylake_(__m128i u16x8) { return _mm512_cvtepu16_epi64(u16x8); }
NK_INTERNAL __m512i nk_u32x8_to_i64x8_skylake_(__m256i u32x8) { return _mm512_cvtepu32_epi64(u32x8); }

NK_INTERNAL __m128i nk_i64x8_to_i8x8_skylake_(__m512i i64x8) {
    __m512i clamped = _mm512_max_epi64(_mm512_min_epi64(i64x8, _mm512_set1_epi64(127)), _mm512_set1_epi64(-128));
    return _mm512_cvtepi64_epi8(clamped);
}
NK_INTERNAL __m128i nk_i64x8_to_u8x8_skylake_(__m512i i64x8) {
    __m512i clamped = _mm512_max_epi64(_mm512_min_epi64(i64x8, _mm512_set1_epi64(255)), _mm512_setzero_si512());
    return _mm512_cvtepi64_epi8(clamped);
}
NK_INTERNAL __m128i nk_i64x8_to_i16x8_skylake_(__m512i i64x8) {
    __m512i clamped = _mm512_max_epi64(_mm512_min_epi64(i64x8, _mm512_set1_epi64(32767)), _mm512_set1_epi64(-32768));
    return _mm512_cvtepi64_epi16(clamped);
}
NK_INTERNAL __m128i nk_i64x8_to_u16x8_skylake_(__m512i i64x8) {
    __m512i clamped = _mm512_max_epi64(_mm512_min_epi64(i64x8, _mm512_set1_epi64(65535)), _mm512_setzero_si512());
    return _mm512_cvtepi64_epi16(clamped);
}
NK_INTERNAL __m256i nk_i64x8_to_i32x8_skylake_(__m512i i64x8) {
    __m512i clamped = _mm512_max_epi64(_mm512_min_epi64(i64x8, _mm512_set1_epi64(NK_I32_MAX)),
                                       _mm512_set1_epi64(NK_I32_MIN));
    return _mm512_cvtepi64_epi32(clamped);
}
NK_INTERNAL __m256i nk_i64x8_to_u32x8_skylake_(__m512i i64x8) {
    __m512i clamped = _mm512_max_epi64(_mm512_min_epi64(i64x8, _mm512_set1_epi64(NK_U32_MAX)), _mm512_setzero_si512());
    return _mm512_cvtepi64_epi32(clamped);
}

NK_INTERNAL __m512d nk_f32x8_to_f64x8_skylake_(__m256 f32x8) { return _mm512_cvtps_pd(f32x8); }
NK_INTERNAL __m512d nk_i32x8_to_f64x8_skylake_(__m256i i32x8) { return _mm512_cvtepi32_pd(i32x8); }
NK_INTERNAL __m512d nk_u32x8_to_f64x8_skylake_(__m256i u32x8) { return _mm512_cvtepu32_pd(u32x8); }

NK_INTERNAL __m256 nk_f64x8_to_f32x8_skylake_(__m512d f64x8) { return _mm512_cvtpd_ps(f64x8); }
NK_INTERNAL __m256i nk_f64x8_to_i32x8_skylake_(__m512d f64x8) {
    __m512d clamped = _mm512_min_pd(_mm512_max_pd(f64x8, _mm512_set1_pd((double)NK_I32_MIN)),
                                    _mm512_set1_pd((double)NK_I32_MAX));
    return _mm512_cvtpd_epi32(clamped);
}
NK_INTERNAL __m256i nk_f64x8_to_u32x8_skylake_(__m512d f64x8) {
    __m512d clamped = _mm512_min_pd(_mm512_max_pd(f64x8, _mm512_setzero_pd()), _mm512_set1_pd((double)NK_U32_MAX));
    return _mm512_cvtpd_epu32(clamped);
}

/**
 *  @brief Convert 64x E2M3 → 64x I8 using VPSHUFB LUT (Skylake AVX-512).
 *
 *  E2M3 format: [sign:1][magnitude:5] where magnitude indexes a 32-entry LUT
 *  that produces the scaled integer value. Sign bit negates the result.
 *  The 32-entry LUT is split into two 16-entry halves for VPSHUFB (which
 *  indexes within 16-byte lanes). Bit 4 of the magnitude selects the half.
 */
NK_INTERNAL __m512i nk_e2m3x64_to_i8x64_skylake_(__m512i raw_i8x64) {
    // lut_magnitude[0..15]  = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}
    // lut_magnitude[16..31] = {32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120}
    // _mm512_set4_epi32(d3, d2, d1, d0) fills bytes [0..3]=d0, [4..7]=d1, [8..11]=d2, [12..15]=d3
    // per 128-bit lane, matching VPSHUFB's per-lane indexing.
    __m512i lut_low_i8x64 = _mm512_set4_epi32( //
        0x1E1C1A18, 0x16141210, 0x0E0C0A08, 0x06040200);
    __m512i lut_high_i8x64 = _mm512_set4_epi32( //
        0x78706860, 0x58504840, 0x3C383430, 0x2C282420);

    __m512i magnitude_i8x64 = _mm512_and_si512(raw_i8x64, _mm512_set1_epi8(0x1F));
    __m512i index_i8x64 = _mm512_and_si512(magnitude_i8x64, _mm512_set1_epi8(0x0F));

    __m512i val_low_i8x64 = _mm512_shuffle_epi8(lut_low_i8x64, index_i8x64);
    __m512i val_high_i8x64 = _mm512_shuffle_epi8(lut_high_i8x64, index_i8x64);

    // Select high half when bit 4 of magnitude is set (magnitude >= 16)
    __mmask64 use_high_mask = _mm512_test_epi8_mask(magnitude_i8x64, _mm512_set1_epi8(0x10));
    __m512i val_i8x64 = _mm512_mask_blend_epi8(use_high_mask, val_low_i8x64, val_high_i8x64);

    // Negate if sign bit (bit 5) is set
    __mmask64 sign_mask = _mm512_test_epi8_mask(raw_i8x64, _mm512_set1_epi8(0x20));
    __m512i negated_i8x64 = _mm512_sub_epi8(_mm512_setzero_si512(), val_i8x64);
    return _mm512_mask_blend_epi8(sign_mask, val_i8x64, negated_i8x64);
}

#pragma endregion Vectorized Conversions

#pragma region Converting Loads and Stores

/** @brief Load 16 f16 values and convert to 16 f32 (Skylake AVX-512). */
NK_INTERNAL void nk_load_f16x16_to_f32x16_skylake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm_ps = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const *)src));
}

/** @brief Partial load of up to 16 f16 values with conversion to f32 (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_f16x16_to_f32x16_skylake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    nk_b256_vec_t f16_partial;
    nk_partial_load_b16x16_skylake_(src, &f16_partial, n);
    dst->zmm_ps = _mm512_cvtph_ps(f16_partial.ymm);
}

/** @brief Load 16 bf16 values and convert to 16 f32 (Skylake AVX-512). */
NK_INTERNAL void nk_load_bf16x16_to_f32x16_skylake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm_ps = nk_bf16x16_to_f32x16_skylake_(_mm256_loadu_si256((__m256i const *)src));
}

/** @brief Partial load of up to 16 bf16 values with conversion to f32 (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_bf16x16_to_f32x16_skylake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    nk_b256_vec_t bf16_partial;
    nk_partial_load_b16x16_skylake_(src, &bf16_partial, n);
    dst->zmm_ps = nk_bf16x16_to_f32x16_skylake_(bf16_partial.ymm);
}

/** @brief Load 16 e4m3 values and convert to 16 f32 (Skylake AVX-512). */
NK_INTERNAL void nk_load_e4m3x16_to_f32x16_skylake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm_ps = nk_e4m3x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)src));
}

/** @brief Partial load of up to 16 e4m3 values with conversion to f32 (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_e4m3x16_to_f32x16_skylake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t e4m3_partial;
    nk_partial_load_b8x16_skylake_(src, &e4m3_partial, n);
    dst->zmm_ps = nk_e4m3x16_to_f32x16_skylake_(e4m3_partial.xmm);
}

/** @brief Load 16 e4m3 values and convert to 16 f16 (Skylake AVX-512BW). */
NK_INTERNAL void nk_load_e4m3x16_to_f16x16_skylake_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm = nk_e4m3x16_to_f16x16_skylake_(_mm_loadu_si128((__m128i const *)src));
}

/** @brief Partial load of up to 16 e4m3 values with conversion to f16 (Skylake AVX-512BW). */
NK_INTERNAL void nk_partial_load_e4m3x16_to_f16x16_skylake_(void const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t e4m3_partial;
    nk_partial_load_b8x16_skylake_(src, &e4m3_partial, n);
    dst->ymm = nk_e4m3x16_to_f16x16_skylake_(e4m3_partial.xmm);
}

/** @brief Load 16 e5m2 values and convert to 16 f32 (Skylake AVX-512). */
NK_INTERNAL void nk_load_e5m2x16_to_f32x16_skylake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm_ps = nk_e5m2x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)src));
}

/** @brief Partial load of up to 16 e5m2 values with conversion to f32 (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_e5m2x16_to_f32x16_skylake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t e5m2_partial;
    nk_partial_load_b8x16_skylake_(src, &e5m2_partial, n);
    dst->zmm_ps = nk_e5m2x16_to_f32x16_skylake_(e5m2_partial.xmm);
}

/** @brief Load 16 e2m3 values and convert to 16 f32 (Skylake AVX-512). */
NK_INTERNAL void nk_load_e2m3x16_to_f32x16_skylake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm_ps = nk_e2m3x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)src));
}

/** @brief Partial load of up to 16 e2m3 values with conversion to f32 (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_e2m3x16_to_f32x16_skylake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t e2m3_partial;
    nk_partial_load_b8x16_skylake_(src, &e2m3_partial, n);
    dst->zmm_ps = nk_e2m3x16_to_f32x16_skylake_(e2m3_partial.xmm);
}

/** @brief Load 16 e3m2 values and convert to 16 f32 (Skylake AVX-512). */
NK_INTERNAL void nk_load_e3m2x16_to_f32x16_skylake_(void const *src, nk_b512_vec_t *dst) {
    dst->zmm_ps = nk_e3m2x16_to_f32x16_skylake_(_mm_loadu_si128((__m128i const *)src));
}

/** @brief Partial load of up to 16 e3m2 values with conversion to f32 (Skylake AVX-512). */
NK_INTERNAL void nk_partial_load_e3m2x16_to_f32x16_skylake_(void const *src, nk_b512_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t e3m2_partial;
    nk_partial_load_b8x16_skylake_(src, &e3m2_partial, n);
    dst->zmm_ps = nk_e3m2x16_to_f32x16_skylake_(e3m2_partial.xmm);
}

#pragma endregion Converting Loads and Stores

#pragma region Public API

NK_PUBLIC void nk_cast_skylake(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
    // Same-type fast path
    if (from_type == to_type) {
        nk_size_t size_bits = nk_dtype_bits(from_type);
        if (size_bits > 0) nk_copy_bytes_(to, from, nk_size_divide_round_up_(n * size_bits, 8));
        return;
    }

    // Type classification for hub selection
    int from_f32_hub = (from_type == nk_f32_k || from_type == nk_f16_k || from_type == nk_bf16_k ||
                        from_type == nk_e4m3_k || from_type == nk_e5m2_k || from_type == nk_e2m3_k ||
                        from_type == nk_e3m2_k || from_type == nk_i8_k || from_type == nk_u8_k ||
                        from_type == nk_i16_k || from_type == nk_u16_k);
    int to_f32_hub = (to_type == nk_f32_k || to_type == nk_f16_k || to_type == nk_bf16_k || to_type == nk_e4m3_k ||
                      to_type == nk_e5m2_k || to_type == nk_e2m3_k || to_type == nk_e3m2_k || to_type == nk_i8_k ||
                      to_type == nk_u8_k || to_type == nk_i16_k || to_type == nk_u16_k);
    int from_unsigned = (from_type == nk_u8_k || from_type == nk_u16_k || from_type == nk_u32_k ||
                         from_type == nk_u64_k);
    int to_unsigned = (to_type == nk_u8_k || to_type == nk_u16_k || to_type == nk_u32_k || to_type == nk_u64_k);
    int from_signed = (from_type == nk_i8_k || from_type == nk_i16_k || from_type == nk_i32_k || from_type == nk_i64_k);
    int to_signed = (to_type == nk_i8_k || to_type == nk_i16_k || to_type == nk_i32_k || to_type == nk_i64_k);
    int from_f64 = (from_type == nk_f64_k);
    int to_f64 = (to_type == nk_f64_k);

    nk_u8_t const *from_ptr = (nk_u8_t const *)from;
    nk_u8_t *to_ptr = (nk_u8_t *)to;

    // Hub 1: f32x16 - float types + small integers (16 elements/batch)
    if (from_f32_hub && to_f32_hub) {
        nk_size_t from_bytes = nk_dtype_bits(from_type) / NK_BITS_PER_BYTE;
        nk_size_t to_bytes = nk_dtype_bits(to_type) / NK_BITS_PER_BYTE;
        while (n > 0) {
            nk_size_t batch = n < 16 ? n : 16;
            __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)batch);
            __m512 hub_f32x16;

            // Upcast to f32x16
            if (from_type == nk_f32_k) hub_f32x16 = _mm512_maskz_loadu_ps(mask, from_ptr);
            else if (from_type == nk_f16_k) hub_f32x16 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, from_ptr));
            else if (from_type == nk_bf16_k)
                hub_f32x16 = nk_bf16x16_to_f32x16_skylake_(_mm256_maskz_loadu_epi16(mask, from_ptr));
            else if (from_type == nk_e4m3_k)
                hub_f32x16 = nk_e4m3x16_to_f32x16_skylake_(_mm_maskz_loadu_epi8(mask, from_ptr));
            else if (from_type == nk_e5m2_k)
                hub_f32x16 = nk_e5m2x16_to_f32x16_skylake_(_mm_maskz_loadu_epi8(mask, from_ptr));
            else if (from_type == nk_e2m3_k)
                hub_f32x16 = nk_e2m3x16_to_f32x16_skylake_(_mm_maskz_loadu_epi8(mask, from_ptr));
            else if (from_type == nk_e3m2_k)
                hub_f32x16 = nk_e3m2x16_to_f32x16_skylake_(_mm_maskz_loadu_epi8(mask, from_ptr));
            else if (from_type == nk_i8_k)
                hub_f32x16 = nk_i8x16_to_f32x16_skylake_(_mm_maskz_loadu_epi8(mask, from_ptr));
            else if (from_type == nk_u8_k)
                hub_f32x16 = nk_u8x16_to_f32x16_skylake_(_mm_maskz_loadu_epi8(mask, from_ptr));
            else if (from_type == nk_i16_k)
                hub_f32x16 = nk_i16x16_to_f32x16_skylake_(_mm256_maskz_loadu_epi16(mask, from_ptr));
            else if (from_type == nk_u16_k)
                hub_f32x16 = nk_u16x16_to_f32x16_skylake_(_mm256_maskz_loadu_epi16(mask, from_ptr));
            else hub_f32x16 = _mm512_setzero_ps();

            // Downcast from f32x16
            if (to_type == nk_f32_k) _mm512_mask_storeu_ps(to_ptr, mask, hub_f32x16);
            else if (to_type == nk_f16_k)
                _mm256_mask_storeu_epi16(to_ptr, mask, _mm512_cvtps_ph(hub_f32x16, _MM_FROUND_TO_NEAREST_INT));
            else if (to_type == nk_bf16_k)
                _mm256_mask_storeu_epi16(to_ptr, mask, nk_f32x16_to_bf16x16_skylake_(hub_f32x16));
            else if (to_type == nk_e4m3_k)
                _mm_mask_storeu_epi8(to_ptr, mask, nk_f32x16_to_e4m3x16_skylake_(hub_f32x16));
            else if (to_type == nk_e5m2_k)
                _mm_mask_storeu_epi8(to_ptr, mask, nk_f32x16_to_e5m2x16_skylake_(hub_f32x16));
            else if (to_type == nk_e2m3_k)
                _mm_mask_storeu_epi8(to_ptr, mask, nk_f32x16_to_e2m3x16_skylake_(hub_f32x16));
            else if (to_type == nk_e3m2_k)
                _mm_mask_storeu_epi8(to_ptr, mask, nk_f32x16_to_e3m2x16_skylake_(hub_f32x16));
            else if (to_type == nk_i8_k) _mm_mask_storeu_epi8(to_ptr, mask, nk_f32x16_to_i8x16_skylake_(hub_f32x16));
            else if (to_type == nk_u8_k) _mm_mask_storeu_epi8(to_ptr, mask, nk_f32x16_to_u8x16_skylake_(hub_f32x16));
            else if (to_type == nk_i16_k)
                _mm256_mask_storeu_epi16(to_ptr, mask, nk_f32x16_to_i16x16_skylake_(hub_f32x16));
            else if (to_type == nk_u16_k)
                _mm256_mask_storeu_epi16(to_ptr, mask, nk_f32x16_to_u16x16_skylake_(hub_f32x16));

            from_ptr += batch * from_bytes;
            to_ptr += batch * to_bytes;
            n -= batch;
        }
        return;
    }

    // Hub 2: u64x8 - unsigned ↔ unsigned integers (8 elements/batch)
    if (from_unsigned && to_unsigned) {
        nk_size_t from_bytes = nk_dtype_bits(from_type) / NK_BITS_PER_BYTE;
        nk_size_t to_bytes = nk_dtype_bits(to_type) / NK_BITS_PER_BYTE;
        while (n > 0) {
            nk_size_t batch = n < 8 ? n : 8;
            __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)batch);
            __m512i hub_u64x8;

            // Upcast to u64x8
            if (from_type == nk_u8_k) hub_u64x8 = nk_u8x8_to_u64x8_skylake_(_mm_maskz_loadu_epi8(mask, from_ptr));
            else if (from_type == nk_u16_k)
                hub_u64x8 = nk_u16x8_to_u64x8_skylake_(_mm_maskz_loadu_epi16(mask, from_ptr));
            else if (from_type == nk_u32_k)
                hub_u64x8 = nk_u32x8_to_u64x8_skylake_(_mm256_maskz_loadu_epi32(mask, from_ptr));
            else if (from_type == nk_u64_k) hub_u64x8 = _mm512_maskz_loadu_epi64(mask, from_ptr);
            else hub_u64x8 = _mm512_setzero_si512();

            // Downcast from u64x8
            if (to_type == nk_u8_k) _mm_mask_storeu_epi8(to_ptr, mask, nk_u64x8_to_u8x8_skylake_(hub_u64x8));
            else if (to_type == nk_u16_k) _mm_mask_storeu_epi16(to_ptr, mask, nk_u64x8_to_u16x8_skylake_(hub_u64x8));
            else if (to_type == nk_u32_k) _mm256_mask_storeu_epi32(to_ptr, mask, nk_u64x8_to_u32x8_skylake_(hub_u64x8));
            else if (to_type == nk_u64_k) _mm512_mask_storeu_epi64(to_ptr, mask, hub_u64x8);

            from_ptr += batch * from_bytes;
            to_ptr += batch * to_bytes;
            n -= batch;
        }
        return;
    }

    // Hub 3: i64x8 - signed/mixed integer conversions (8 elements/batch)
    if ((from_signed || from_unsigned) && (to_signed || to_unsigned)) {
        nk_size_t from_bytes = nk_dtype_bits(from_type) / NK_BITS_PER_BYTE;
        nk_size_t to_bytes = nk_dtype_bits(to_type) / NK_BITS_PER_BYTE;
        while (n > 0) {
            nk_size_t batch = n < 8 ? n : 8;
            __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)batch);
            __m512i hub_i64x8;

            // Upcast to i64x8
            if (from_type == nk_i8_k) hub_i64x8 = nk_i8x8_to_i64x8_skylake_(_mm_maskz_loadu_epi8(mask, from_ptr));
            else if (from_type == nk_u8_k) hub_i64x8 = nk_u8x8_to_i64x8_skylake_(_mm_maskz_loadu_epi8(mask, from_ptr));
            else if (from_type == nk_i16_k)
                hub_i64x8 = nk_i16x8_to_i64x8_skylake_(_mm_maskz_loadu_epi16(mask, from_ptr));
            else if (from_type == nk_u16_k)
                hub_i64x8 = nk_u16x8_to_i64x8_skylake_(_mm_maskz_loadu_epi16(mask, from_ptr));
            else if (from_type == nk_i32_k)
                hub_i64x8 = nk_i32x8_to_i64x8_skylake_(_mm256_maskz_loadu_epi32(mask, from_ptr));
            else if (from_type == nk_u32_k)
                hub_i64x8 = nk_u32x8_to_i64x8_skylake_(_mm256_maskz_loadu_epi32(mask, from_ptr));
            else if (from_type == nk_i64_k || from_type == nk_u64_k)
                hub_i64x8 = _mm512_maskz_loadu_epi64(mask, from_ptr);
            else hub_i64x8 = _mm512_setzero_si512();

            // Downcast from i64x8
            if (to_type == nk_i8_k) _mm_mask_storeu_epi8(to_ptr, mask, nk_i64x8_to_i8x8_skylake_(hub_i64x8));
            else if (to_type == nk_u8_k) _mm_mask_storeu_epi8(to_ptr, mask, nk_i64x8_to_u8x8_skylake_(hub_i64x8));
            else if (to_type == nk_i16_k) _mm_mask_storeu_epi16(to_ptr, mask, nk_i64x8_to_i16x8_skylake_(hub_i64x8));
            else if (to_type == nk_u16_k) _mm_mask_storeu_epi16(to_ptr, mask, nk_i64x8_to_u16x8_skylake_(hub_i64x8));
            else if (to_type == nk_i32_k) _mm256_mask_storeu_epi32(to_ptr, mask, nk_i64x8_to_i32x8_skylake_(hub_i64x8));
            else if (to_type == nk_u32_k) _mm256_mask_storeu_epi32(to_ptr, mask, nk_i64x8_to_u32x8_skylake_(hub_i64x8));
            else if (to_type == nk_i64_k || to_type == nk_u64_k) _mm512_mask_storeu_epi64(to_ptr, mask, hub_i64x8);

            from_ptr += batch * from_bytes;
            to_ptr += batch * to_bytes;
            n -= batch;
        }
        return;
    }

    // Hub 4: f64x8 - f64 conversions (8 elements/batch)
    // Only enter when both sides are types we can actually handle: f64, f32, i32, u32.
    // Unsupported pairs (e.g. i8→f64, f16→f64) fall through to serial fallback.
    if ((from_f64 || to_f64) &&                                                                               //
        (from_type == nk_f64_k || from_type == nk_f32_k || from_type == nk_i32_k || from_type == nk_u32_k) && //
        (to_type == nk_f64_k || to_type == nk_f32_k || to_type == nk_i32_k || to_type == nk_u32_k)) {
        nk_size_t from_bytes = nk_dtype_bits(from_type) / NK_BITS_PER_BYTE;
        nk_size_t to_bytes = nk_dtype_bits(to_type) / NK_BITS_PER_BYTE;
        while (n > 0) {
            nk_size_t batch = n < 8 ? n : 8;
            __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, (unsigned int)batch);
            __m512d hub_f64x8;

            // Upcast to f64x8
            if (from_type == nk_f64_k) hub_f64x8 = _mm512_maskz_loadu_pd(mask, from_ptr);
            else if (from_type == nk_f32_k)
                hub_f64x8 = nk_f32x8_to_f64x8_skylake_(_mm256_maskz_loadu_ps(mask, from_ptr));
            else if (from_type == nk_i32_k)
                hub_f64x8 = nk_i32x8_to_f64x8_skylake_(_mm256_maskz_loadu_epi32(mask, from_ptr));
            else if (from_type == nk_u32_k)
                hub_f64x8 = nk_u32x8_to_f64x8_skylake_(_mm256_maskz_loadu_epi32(mask, from_ptr));
            else hub_f64x8 = _mm512_setzero_pd();

            // Downcast from f64x8
            if (to_type == nk_f64_k) _mm512_mask_storeu_pd(to_ptr, mask, hub_f64x8);
            else if (to_type == nk_f32_k) _mm256_mask_storeu_ps(to_ptr, mask, nk_f64x8_to_f32x8_skylake_(hub_f64x8));
            else if (to_type == nk_i32_k) _mm256_mask_storeu_epi32(to_ptr, mask, nk_f64x8_to_i32x8_skylake_(hub_f64x8));
            else if (to_type == nk_u32_k) _mm256_mask_storeu_epi32(to_ptr, mask, nk_f64x8_to_u32x8_skylake_(hub_f64x8));

            from_ptr += batch * from_bytes;
            to_ptr += batch * to_bytes;
            n -= batch;
        }
        return;
    }

    // Fallback: complex types, i4/u4/u1, unsupported combinations
    nk_cast_serial(from, from_type, n, to, to_type);
}

/** @brief Convert 16× e2m1 → 16× f32 via 8-magnitude LUT + sign flip (AVX-512).
 *  Input: 8 bytes (low 64 bits of @p packed) holding 16 nibbles, high nibble of byte → even lane.
 *  E2M1 magnitudes {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} indexed by nibble bits 2..0; bit 3 → sign. */
NK_INTERNAL __m512 nk_e2m1x16_to_f32x16_skylake_(__m128i packed) {
    // Expand 8 packed bytes to 16 nibble bytes via shift + mask + unpack interleave
    __m128i low_nibbles = _mm_and_si128(packed, _mm_set1_epi8(0x0F));
    __m128i high_nibbles = _mm_and_si128(_mm_srli_epi32(packed, 4), _mm_set1_epi8(0x0F));
    __m128i nibbles_b8x16 = _mm_unpacklo_epi8(high_nibbles, low_nibbles);
    __m512i nibbles_i32x16 = _mm512_cvtepu8_epi32(nibbles_b8x16);

    // Magnitude LUT indexed by bits 2..0 (8 entries, broadcast to low half of __m512 via permutexvar)
    __m512 magnitude_lut = _mm512_set_ps(0, 0, 0, 0, 0, 0, 0, 0, //
                                         6.0f, 4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f, 0.0f);
    __m512i magnitude_idx_i32x16 = _mm512_and_si512(nibbles_i32x16, _mm512_set1_epi32(0x07));
    __m512 magnitudes_f32x16 = _mm512_permutexvar_ps(magnitude_idx_i32x16, magnitude_lut);

    // Sign: bit 3 of the nibble → bit 31 of the f32
    __m512i sign_f32_bits_i32x16 = _mm512_slli_epi32(_mm512_and_si512(nibbles_i32x16, _mm512_set1_epi32(0x08)), 28);
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(magnitudes_f32x16), sign_f32_bits_i32x16));
}

/** @brief Convert 16× f32 → 16× e2m1 via bit manipulation, packed to 8 bytes (AVX-512).
 *  Output: 8 bytes in the low 64 bits of the result. Lane 0 → high nibble of byte 0, lane 1 → low nibble. */
NK_INTERNAL __m128i nk_f32x16_to_e2m1x16_skylake_(__m512 f32x16) {
    __m512i bits_i32x16 = _mm512_castps_si512(f32x16);
    __m512i sign_i32x16 = _mm512_srli_epi32(bits_i32x16, 31);
    __m512i f32_exp_i32x16 = _mm512_and_si512(_mm512_srli_epi32(bits_i32x16, 23), _mm512_set1_epi32(0xFF));

    // Normal path: round 23-bit mantissa to 1 bit using RNE (cut at bit 22).
    __m512i significand_i32x16 = _mm512_or_si512(_mm512_and_si512(bits_i32x16, _mm512_set1_epi32(0x007FFFFF)),
                                                 _mm512_set1_epi32(0x00800000));
    __m512i lsb_i32x16 = _mm512_and_si512(_mm512_srli_epi32(significand_i32x16, 22), _mm512_set1_epi32(1));
    __m512i rounding_bias_i32x16 = _mm512_add_epi32(_mm512_set1_epi32(0x001FFFFF), lsb_i32x16);
    __m512i rounded_sig_i32x16 = _mm512_add_epi32(significand_i32x16, rounding_bias_i32x16);
    __m512i carry_i32x16 = _mm512_srli_epi32(rounded_sig_i32x16, 24);
    __m512i normal_mantissa_i32x16 = _mm512_and_si512(_mm512_srli_epi32(rounded_sig_i32x16, 22),
                                                      _mm512_set1_epi32(0x01));
    __m512i e2m1_exp_i32x16 = _mm512_sub_epi32(_mm512_add_epi32(f32_exp_i32x16, carry_i32x16), _mm512_set1_epi32(126));

    __mmask16 is_subnormal = _mm512_cmpgt_epi32_mask(_mm512_set1_epi32(1), e2m1_exp_i32x16);
    __mmask16 overflow = _mm512_cmpgt_epi32_mask(e2m1_exp_i32x16, _mm512_set1_epi32(3));

    __m512i clamped_exp_i32x16 = _mm512_max_epi32(e2m1_exp_i32x16, _mm512_set1_epi32(1));
    clamped_exp_i32x16 = _mm512_min_epi32(clamped_exp_i32x16, _mm512_set1_epi32(3));
    normal_mantissa_i32x16 = _mm512_mask_blend_epi32(overflow, normal_mantissa_i32x16, _mm512_set1_epi32(0x01));
    __m512i normal_nibble_i32x16 = _mm512_ternarylogic_epi32(
        _mm512_slli_epi32(sign_i32x16, 3), _mm512_slli_epi32(clamped_exp_i32x16, 1), normal_mantissa_i32x16, 0xFE);

    // Subnormal path: round(|x| * 2), clamp to {0, 1}. Promotion to first normal (0x02) when it rounds up to 2.
    __m512 abs_f32x16 = _mm512_and_ps(f32x16, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)));
    __m512 scaled_f32x16 = _mm512_mul_ps(abs_f32x16, _mm512_set1_ps(2.0f));
    __m512i subnorm_mantissa_i32x16 = _mm512_cvtps_epi32(scaled_f32x16);
    __mmask16 promotes_to_normal = _mm512_cmpgt_epi32_mask(subnorm_mantissa_i32x16, _mm512_set1_epi32(1));
    subnorm_mantissa_i32x16 = _mm512_max_epi32(_mm512_min_epi32(subnorm_mantissa_i32x16, _mm512_set1_epi32(1)),
                                               _mm512_setzero_si512());
    __m512i subnorm_nibble_i32x16 = _mm512_or_si512(_mm512_slli_epi32(sign_i32x16, 3), subnorm_mantissa_i32x16);
    __m512i first_normal_nibble_i32x16 = _mm512_or_si512(_mm512_slli_epi32(sign_i32x16, 3), _mm512_set1_epi32(0x02));
    subnorm_nibble_i32x16 = _mm512_mask_blend_epi32(promotes_to_normal, subnorm_nibble_i32x16,
                                                    first_normal_nibble_i32x16);

    __m512i nibble_i32x16 = _mm512_mask_blend_epi32(is_subnormal, normal_nibble_i32x16, subnorm_nibble_i32x16);

    // Pack 16 nibbles (each in low 4 bits of a byte) to 8 bytes: even idx → high nibble, odd → low.
    __m128i nibble_b8x16 = _mm512_cvtepi32_epi8(nibble_i32x16);
    __m128i pack_coeff = _mm_set1_epi16(0x0110); // byte 0 coefficient 0x10, byte 1 coefficient 0x01
    __m128i packed_i16x8 = _mm_maddubs_epi16(nibble_b8x16, pack_coeff);
    return _mm_packus_epi16(packed_i16x8, _mm_setzero_si128());
}

/** @brief Reduce a block of `block_count` f32s to `amax = max(|x|)`. `block_count` ≤ 32. */
NK_INTERNAL nk_f32_t nk_block_amax_f32_skylake_(nk_f32_t const *block, nk_size_t block_count) {
    __m512i abs_mask = _mm512_set1_epi32(0x7FFFFFFF);
    __mmask16 const full_mask = 0xFFFF;
    __m512 block_low = _mm512_maskz_loadu_ps(block_count >= 16 ? full_mask : ((1u << block_count) - 1u), block);
    __m512 abs_low = _mm512_and_ps(block_low, _mm512_castsi512_ps(abs_mask));
    if (block_count <= 16) return _mm512_reduce_max_ps(abs_low);
    __m512 block_high = _mm512_maskz_loadu_ps((1u << (block_count - 16)) - 1u, block + 16);
    __m512 abs_high = _mm512_and_ps(block_high, _mm512_castsi512_ps(abs_mask));
    return _mm512_reduce_max_ps(_mm512_max_ps(abs_low, abs_high));
}

/** @brief Skylake-optimised block-scaled cast. Uses AVX-512 amax + broadcast reciprocal multiply
 *  around the serial element codec hub, which vectorises the dominant scale-derivation cost for
 *  MXFP8 / MXFP6 / MXFP4 / MXINT8 / NVFP4 without duplicating per-format packing logic. */
NK_PUBLIC void nk_cast_block_scaled_skylake(                                                             //
    void const *from, void const *from_scales, nk_scalar_buffer_t const *from_global,                    //
    nk_block_scaled_format_t const *from_format,                                                         //
    void *to, void *to_scales, nk_scalar_buffer_t *to_global, nk_block_scaled_format_t const *to_format, //
    nk_size_t count) {

    int from_plain = (from_format->scale_dtype == nk_dtype_unknown_k || from_format->block_size == 0);
    int to_plain = (to_format->scale_dtype == nk_dtype_unknown_k || to_format->block_size == 0);

    if (from_plain && to_plain) {
        nk_cast_skylake(from, from_format->element_dtype, count, to, to_format->element_dtype);
        return;
    }

    nk_size_t from_block = from_plain ? 1u : from_format->block_size;
    nk_size_t to_block = to_plain ? 1u : to_format->block_size;
    nk_size_t chunk = from_block > to_block ? from_block : to_block;

    nk_f32_t from_global_f32 = 1.0f;
    if (from_global != NULL && !from_plain && from_format->global_dtype == nk_f32_k) from_global_f32 = from_global->f32;

    nk_f32_t to_global_f32 = 1.0f;
    int to_has_global = (!to_plain && to_global != NULL && to_format->global_dtype == nk_f32_k);
    if (to_has_global) {
        to_global_f32 = to_global->f32;
        if (to_global_f32 == 0.0f) {
            // Fall back to serial for auto-derive (needs a full tensor scan; rare calibration path).
            nk_cast_block_scaled_serial(from, from_scales, from_global, from_format, to, to_scales, to_global,
                                        to_format, count);
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
            nk_cast_skylake(src, from_format->element_dtype, chunk_count, scratch, nk_f32_k);
        }
        else {
            for (nk_size_t b = 0; b < chunk_count; b += from_block) {
                nk_size_t block_idx = (chunk_start + b) / from_block;
                nk_u8_t raw = from_scales_bytes[block_idx];
                nk_f32_t scale_f32 = nk_block_scaled_decode_scale_serial_(raw, from_format->scale_dtype) *
                                     from_global_f32;
                void const *src = (nk_u8_t const *)from +
                                  ((chunk_start + b) * from_bits_per_element / NK_BITS_PER_BYTE);
                nk_cast_skylake(src, from_format->element_dtype, from_block, scratch + b, nk_f32_k);
                __m512 scale_bcast = _mm512_set1_ps(scale_f32);
                __m512 v_low = _mm512_maskz_loadu_ps(from_block >= 16 ? 0xFFFF : (1u << from_block) - 1u, scratch + b);
                _mm512_mask_storeu_ps(scratch + b, from_block >= 16 ? 0xFFFF : (1u << from_block) - 1u,
                                      _mm512_mul_ps(v_low, scale_bcast));
                if (from_block > 16) {
                    __m512 v_high = _mm512_maskz_loadu_ps((1u << (from_block - 16)) - 1u, scratch + b + 16);
                    _mm512_mask_storeu_ps(scratch + b + 16, (1u << (from_block - 16)) - 1u,
                                          _mm512_mul_ps(v_high, scale_bcast));
                }
            }
        }

        // Encode f32 scratch into destination chunk.
        if (to_plain) {
            void *dst = (nk_u8_t *)to + (chunk_start * to_bits_per_element / NK_BITS_PER_BYTE);
            nk_cast_skylake(scratch, nk_f32_k, chunk_count, dst, to_format->element_dtype);
        }
        else {
            nk_f32_t element_max = nk_element_max_representable_(to_format->element_dtype);
            for (nk_size_t b = 0; b < chunk_count; b += to_block) {
                nk_f32_t block_amax = nk_block_amax_f32_skylake_(scratch + b, to_block);
                nk_f32_t scale_target = block_amax / element_max / to_global_f32;
                nk_u8_t raw = nk_block_scaled_encode_scale_serial_(scale_target, to_format->scale_dtype);
                nk_size_t block_idx = (chunk_start + b) / to_block;
                to_scales_bytes[block_idx] = raw;
                nk_f32_t effective_scale = nk_block_scaled_decode_scale_serial_(raw, to_format->scale_dtype) *
                                           to_global_f32;
                nk_f32_t reciprocal = effective_scale > 0 ? (1.0f / effective_scale) : 0.0f;
                __m512 reciprocal_bcast = _mm512_set1_ps(reciprocal);
                nk_f32_t encoded_scratch[32];
                __m512 v_low = _mm512_maskz_loadu_ps(to_block >= 16 ? 0xFFFF : (1u << to_block) - 1u, scratch + b);
                _mm512_mask_storeu_ps(encoded_scratch, to_block >= 16 ? 0xFFFF : (1u << to_block) - 1u,
                                      _mm512_mul_ps(v_low, reciprocal_bcast));
                if (to_block > 16) {
                    __m512 v_high = _mm512_maskz_loadu_ps((1u << (to_block - 16)) - 1u, scratch + b + 16);
                    _mm512_mask_storeu_ps(encoded_scratch + 16, (1u << (to_block - 16)) - 1u,
                                          _mm512_mul_ps(v_high, reciprocal_bcast));
                }
                void *dst = (nk_u8_t *)to + ((chunk_start + b) * to_bits_per_element / NK_BITS_PER_BYTE);
                nk_cast_skylake(encoded_scratch, nk_f32_k, to_block, dst, to_format->element_dtype);
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

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X8664_
#endif // NK_CAST_SKYLAKE_H
