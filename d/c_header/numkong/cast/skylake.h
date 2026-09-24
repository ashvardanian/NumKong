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
 *      Intrinsic                   Instruction                     SKL         ICL         Genoa
 *      _mm512_cvtph_ps             VCVTPH2PS (ZMM, YMM)            5cy @ p05   5cy @ p05   4cy @ p01
 *      _mm512_cvtps_ph             VCVTPS2PH (YMM, ZMM, imm)       5cy @ p05   5cy @ p05   4cy @ p01
 *      _mm512_cvtps_epi32          VCVTPS2DQ (ZMM, ZMM)            4cy @ p0    4cy @ p0    3cy @ p01
 *      _mm512_cvtepi32_ps          VCVTDQ2PS (ZMM, ZMM)            4cy @ p0    4cy @ p0    3cy @ p01
 *      _mm512_cvtepi32_epi16       VPMOVDW (YMM, ZMM)              3cy @ p5    3cy @ p5    2cy @ p12
 *      _mm512_cvtsepi32_epi8       VPMOVSDB (XMM, ZMM)             3cy @ p5    3cy @ p5    2cy @ p12
 *
 *  F16 conversions use hardware F16C via VCVTPH2PS/VCVTPS2PH. BF16 lacks hardware support on Skylake,
 *  requiring emulation via VPMOVZXWD + VPSLLD for bf16-to-f32, achieving ~4cy total. FP8 (E4M3/E5M2)
 *  conversions use bit manipulation with VPTERNLOGD for sign/exp/mantissa composition.
 */
#ifndef NK_CAST_SKYLAKE_H
#define NK_CAST_SKYLAKE_H

#if NK_TARGET_X86_
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

#pragma region - Type Punned Loads and Stores

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

#pragma endregion - Type Punned Loads and Stores

#pragma region - Vectorized Conversions

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

/** @brief Convert 16x e4m3 → 16x f32 via bit manipulation (AVX-512).
 *  E4M3 format: S EEEE MMM (bias=7). F32: sign<<31, (exp+120)<<23, mantissa<<20.
 *  Subnormals (exp=0): value = mantissa × 2⁽¹⁻⁷⁾ × 2⁻³ = mantissa ÷ 512. */
NK_INTERNAL __m512 nk_e4m3x16_to_f32x16_skylake_(__m128i e4m3_i8x16) {
    __m512i e4m3_i32x16 = _mm512_cvtepu8_epi32(e4m3_i8x16);

    // Extract fields
    __m512i exp_i32x16 = _mm512_and_si512(_mm512_srli_epi32(e4m3_i32x16, 3), _mm512_set1_epi32(0x0F));
    __m512i mantissa_i32x16 = _mm512_and_si512(e4m3_i32x16, _mm512_set1_epi32(0x07));
    __m512i sign_i32x16 = _mm512_slli_epi32(_mm512_srli_epi32(e4m3_i32x16, 7), 31);

    // Normal path: sign | ((exp+120)<<23) | (mantissa<<20)
    __m512i f32_exp_i32x16 = _mm512_slli_epi32(_mm512_add_epi32(exp_i32x16, _mm512_set1_epi32(120)), 23);
    __m512i f32_mantissa_i32x16 = _mm512_slli_epi32(mantissa_i32x16, 20);
    __m512 result_f32x16 = _mm512_castsi512_ps(
        _mm512_ternarylogic_epi32(sign_i32x16, f32_exp_i32x16, f32_mantissa_i32x16, 0xFE));

    // Subnormal fix: for exp==0 lanes, replace with (mantissa / 512) | sign using masked OR
    __mmask16 is_subnormal = _mm512_testn_epi32_mask(e4m3_i32x16, _mm512_set1_epi32(0x78));
    __m512 subnorm_abs_f32x16 = _mm512_mul_ps(_mm512_cvtepi32_ps(mantissa_i32x16), _mm512_set1_ps(1.0f / 512.0f));
    result_f32x16 = _mm512_mask_or_ps(result_f32x16, is_subnormal, subnorm_abs_f32x16,
                                      _mm512_castsi512_ps(sign_i32x16));

    // NaN path: E4M3FN has NaN only when exp=15 AND mant=7 (0x7F or 0xFF)
    __mmask16 is_nan = _mm512_mask_cmpeq_epi32_mask(                                //
        _mm512_cmpeq_epi32_mask(exp_i32x16, _mm512_set1_epi32(15)),                 //
        mantissa_i32x16, _mm512_set1_epi32(7));                                     //
    __m512i nan_bits = _mm512_or_si512(sign_i32x16, _mm512_set1_epi32(0x7FC00000)); // F32 quiet NaN
    return _mm512_mask_blend_ps(is_nan, result_f32x16, _mm512_castsi512_ps(nan_bits));
}

/** @brief Convert 16x e5m2 → 16x f32 via bit manipulation (AVX-512).
 *  E5M2 format: S EEEEE MM (bias=15). F32: sign<<31, (exp+112)<<23, mantissa<<21.
 *  Subnormals (exp=0): value = mantissa × 2⁽¹⁻¹⁵⁾ × 2⁻² = mantissa ÷ 65536. */
NK_INTERNAL __m512 nk_e5m2x16_to_f32x16_skylake_(__m128i e5m2_i8x16) {
    __m512i e5m2_i32x16 = _mm512_cvtepu8_epi32(e5m2_i8x16);

    // Extract fields
    __m512i exp_i32x16 = _mm512_and_si512(_mm512_srli_epi32(e5m2_i32x16, 2), _mm512_set1_epi32(0x1F));
    __m512i mantissa_i32x16 = _mm512_and_si512(e5m2_i32x16, _mm512_set1_epi32(0x03));
    __m512i sign_i32x16 = _mm512_slli_epi32(_mm512_srli_epi32(e5m2_i32x16, 7), 31);

    // Normal path: sign | ((exp+112)<<23) | (mantissa<<21)
    __m512i f32_exp_i32x16 = _mm512_slli_epi32(_mm512_add_epi32(exp_i32x16, _mm512_set1_epi32(112)), 23);
    __m512i f32_mantissa_i32x16 = _mm512_slli_epi32(mantissa_i32x16, 21);
    __m512 result_f32x16 = _mm512_castsi512_ps(
        _mm512_ternarylogic_epi32(sign_i32x16, f32_exp_i32x16, f32_mantissa_i32x16, 0xFE));

    // Subnormal fix: for exp==0 lanes, replace with (mantissa / 65536) | sign using masked OR
    __mmask16 is_subnormal = _mm512_testn_epi32_mask(e5m2_i32x16, _mm512_set1_epi32(0x7C));
    __m512 subnorm_abs_f32x16 = _mm512_mul_ps(_mm512_cvtepi32_ps(mantissa_i32x16), _mm512_set1_ps(1.0f / 65536.0f));
    return _mm512_mask_or_ps(result_f32x16, is_subnormal, subnorm_abs_f32x16, _mm512_castsi512_ps(sign_i32x16));
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

#pragma endregion - Vectorized Conversions

#pragma region - Converting Loads and Stores

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

#pragma endregion - Converting Loads and Stores

#pragma region - Public API

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

#pragma endregion - Public API

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X86_
#endif // NK_CAST_SKYLAKE_H
