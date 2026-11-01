/**
 *  @brief SIMD-accelerated Type Conversions for Sapphire Rapids.
 *  @file include/numkong/cast/sapphire.h
 *  @author Ash Vardanian
 *  @date January 2, 2026
 *
 *  @section sapphire_cast_instructions Relevant Instructions
 *
 *      Intrinsic                   Instruction                     Sapphire    Genoa
 *      _mm_cvtss_sh                VCVTSS2SH (XMM, XMM, XMM)       5cy @ p05   5cy @ p01
 *      _mm_cvtsh_ss                VCVTSH2SS (XMM, XMM, XMM)       5cy @ p05   5cy @ p01
 *      _mm256_cvtepu8_epi16        VPMOVZXBW (YMM, XMM)            3cy @ p5    3cy @ p12
 *      _mm256_mul_ph               VMULPH (YMM, YMM, YMM)          4cy @ p05   3cy @ p01
 *      _mm256_cvtepi16_ph          VCVTW2PH (YMM, YMM)             4cy @ p05   4cy @ p01
 *      _mm256_cvtph_epi16          VCVTPH2W (YMM, YMM)             4cy @ p05   4cy @ p01
 *      _mm256_mask_blend_epi16     VPBLENDMW (YMM, K, YMM, YMM)    1cy @ p05   1cy @ p0123
 *      _mm256_testn_epi16_mask     VPTESTNMW (K, YMM, YMM)         3cy @ p5    3cy @ p0
 *      _mm256_cvtepi16_epi8        VPMOVWB (XMM, YMM)              4cy @ p5    4cy @ p12
 *      _mm_maskz_loadu_epi8        VMOVDQU8 (XMM {K}, M128)        7cy @ p23   7cy @ p23
 *      _mm256_mask_storeu_epi16    VMOVDQU16 (M256 {K}, YMM)       4cy @ p4    4cy @ p4
 */
#ifndef NK_CAST_SAPPHIRE_H
#define NK_CAST_SAPPHIRE_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE

#include "numkong/types.h"
#include "numkong/cast/icelake.h" // `nk_cast_icelake`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512fp16,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512fp16", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC void nk_f32_to_f16_sapphire(nk_f32_t const *from, nk_f16_t *to) {
    *to = _mm_cvtsi128_si32(_mm_castph_si128(_mm_cvtss_sh(_mm_setzero_ph(), _mm_set_ss(*from))));
}

NK_PUBLIC void nk_f16_to_f32_sapphire(nk_f16_t const *from, nk_f32_t *to) {
    *to = _mm_cvtss_f32(_mm_cvtsh_ss(_mm_setzero_ps(), _mm_castsi128_ph(_mm_cvtsi32_si128(*from))));
}

#pragma region - Vectorized Conversions

/** @brief Convert 16x e4m3 → 16x f16 via bit manipulation (AVX-512 FP16).
 *  E4M3 format: S EEEE MMM (bias=7). F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Normal: sign | ((exp+8)<<10) | (mant<<7).
 *  Subnormals (exp=0): value = mantissa ÷ 512, computed via f16 arithmetic. */
NK_INTERNAL __m256h nk_e4m3x16_to_f16x16_sapphire_(__m128i e4m3_i8x16) {
    __m256i e4m3_i16x16 = _mm256_cvtepu8_epi16(e4m3_i8x16);

    // Extract fields
    __m256i mantissa_i16x16 = _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi16(0x07));
    __m256i sign_i16x16 = _mm256_and_si256(_mm256_slli_epi16(e4m3_i16x16, 8), _mm256_set1_epi16((short)0x8000));

    // Normal path: sign | ((exp+8)<<10) | (mantissa<<7) via single shift + bias add
    __m256i exp_mantissa_i16x16 = _mm256_slli_epi16(_mm256_and_si256(e4m3_i16x16, _mm256_set1_epi16(0x7F)), 7);
    __m256i exp_mantissa_biased_i16x16 = _mm256_add_epi16(exp_mantissa_i16x16, _mm256_set1_epi16(0x2000));
    __m256i normal_i16x16 = _mm256_or_si256(sign_i16x16, exp_mantissa_biased_i16x16);

    // Subnormal fix: for exp==0 lanes, use (subnorm_abs | sign); else keep normal
    __mmask16 is_subnormal = _mm256_testn_epi16_mask(e4m3_i16x16, _mm256_set1_epi16(0x78));
    __m256h subnorm_abs_f16x16 = _mm256_mul_ph(_mm256_cvtepi16_ph(mantissa_i16x16),
                                               _mm256_castsi256_ph(_mm256_set1_epi16(0x1800))); // 1/512
    __m256i subnorm_signed_i16x16 = _mm256_or_si256(_mm256_castph_si256(subnorm_abs_f16x16), sign_i16x16);
    __m256i result_i16x16 = _mm256_mask_blend_epi16(is_subnormal, normal_i16x16, subnorm_signed_i16x16);

    // NaN path: E4M3FN has NaN only when exp=15 AND mant=7 (lower 7 bits == 0x7F)
    __mmask16 is_nan = _mm256_cmpeq_epi16_mask(                                 //
        _mm256_and_si256(e4m3_i16x16, _mm256_set1_epi16(0x7F)),                 //
        _mm256_set1_epi16(0x7F));                                               //
    __m256i nan_bits = _mm256_or_si256(sign_i16x16, _mm256_set1_epi16(0x7E00)); // F16 quiet NaN
    return _mm256_castsi256_ph(_mm256_mask_blend_epi16(is_nan, result_i16x16, nan_bits));
}

/** @brief Convert 16x e5m2 → 16x f16 via bit manipulation (AVX-512 FP16).
 *  E5M2 format: S EEEEE MM (bias=15). F16: S EEEEE MMMMMMMMMM (bias=15).
 *  Normal: sign | (exp<<10) | (mant<<8) (same exponent bias).
 *  Subnormals (exp=0): value = mantissa ÷ 65536, computed via f16 arithmetic. */
NK_INTERNAL __m256h nk_e5m2x16_to_f16x16_sapphire_(__m128i e5m2_i8x16) {
    __m256i e5m2_i16x16 = _mm256_cvtepu8_epi16(e5m2_i8x16);

    // Extract fields
    __m256i mantissa_i16x16 = _mm256_and_si256(e5m2_i16x16, _mm256_set1_epi16(0x03));
    __m256i sign_i16x16 = _mm256_and_si256(_mm256_slli_epi16(e5m2_i16x16, 8), _mm256_set1_epi16((short)0x8000));

    // Normal path: sign | (exp<<10) | (mant<<8) - same exponent bias so just shift lower7 by 8
    __m256i exp_mantissa_i16x16 = _mm256_slli_epi16(_mm256_and_si256(e5m2_i16x16, _mm256_set1_epi16(0x7F)), 8);
    __m256i normal_i16x16 = _mm256_or_si256(sign_i16x16, exp_mantissa_i16x16);

    // Subnormal fix: for exp==0 lanes, use (subnorm_abs | sign); else keep normal
    __mmask16 is_subnormal = _mm256_testn_epi16_mask(e5m2_i16x16, _mm256_set1_epi16(0x7C));
    __m256h subnorm_abs_f16x16 = _mm256_mul_ph(_mm256_cvtepi16_ph(mantissa_i16x16),
                                               _mm256_castsi256_ph(_mm256_set1_epi16(0x0100))); // 1/65536
    __m256i subnorm_signed_i16x16 = _mm256_or_si256(_mm256_castph_si256(subnorm_abs_f16x16), sign_i16x16);
    return _mm256_castsi256_ph(_mm256_mask_blend_epi16(is_subnormal, normal_i16x16, subnorm_signed_i16x16));
}

/** @brief Convert 16x f16 → 16x e4m3 via bit manipulation (AVX-512 FP16).
 *  F16: S EEEEE MMMMMMMMMM (bias=15). E4M3: S EEEE MMM (bias=7).
 *  Handles normal, subnormal, and overflow cases with RNE rounding. */
NK_INTERNAL __m128i nk_f16x16_to_e4m3x16_sapphire_(__m256h f16x16) {
    __m256i bits_i16x16 = _mm256_castph_si256(f16x16);
    __m256i sign_i16x16 = _mm256_srli_epi16(bits_i16x16, 15);
    __m256i f16_exp_i16x16 = _mm256_and_si256(_mm256_srli_epi16(bits_i16x16, 10), _mm256_set1_epi16(0x1F));

    // Round mantissa from 10 to 3 bits using RNE (round to nearest, ties to even)
    __m256i significand_i16x16 = _mm256_or_si256(_mm256_and_si256(bits_i16x16, _mm256_set1_epi16(0x03FF)),
                                                 _mm256_set1_epi16(0x0400)); // Add implicit 1 bit
    __m256i lsb_i16x16 = _mm256_and_si256(_mm256_srli_epi16(significand_i16x16, 7), _mm256_set1_epi16(1));
    __m256i rounding_bias_i16x16 = _mm256_add_epi16(_mm256_set1_epi16(0x003F), lsb_i16x16);
    __m256i rounded_sig_i16x16 = _mm256_add_epi16(significand_i16x16, rounding_bias_i16x16);
    __m256i carry_i16x16 = _mm256_srli_epi16(rounded_sig_i16x16, 11); // Carry into exponent if bit 11 set
    __m256i f16_mantissa_i16x16 = _mm256_and_si256(_mm256_srli_epi16(rounded_sig_i16x16, 7), _mm256_set1_epi16(0x07));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    f16_mantissa_i16x16 = _mm256_andnot_si256(_mm256_slli_epi16(carry_i16x16, 15), f16_mantissa_i16x16);
    __m256i e4m3_exp_i16x16 = _mm256_sub_epi16(_mm256_add_epi16(f16_exp_i16x16, carry_i16x16), _mm256_set1_epi16(8));

    // Detect underflow (exp <= 0) and overflow (exp > 15)
    __mmask16 is_subnormal = _mm256_cmpgt_epi16_mask(_mm256_set1_epi16(1), e4m3_exp_i16x16);
    __mmask16 overflow = _mm256_cmpgt_epi16_mask(e4m3_exp_i16x16, _mm256_set1_epi16(15));

    // Normal path: clamp exp to [1,15]
    // e4m3FN quirk: exp=15 with mantissa=7 is NaN (0x7F), so clamp mantissa to 6 when exp=15.
    __m256i clamped_exp_i16x16 = _mm256_max_epi16(e4m3_exp_i16x16, _mm256_set1_epi16(1));
    clamped_exp_i16x16 = _mm256_min_epi16(clamped_exp_i16x16, _mm256_set1_epi16(15));
    __mmask16 is_max_exp = _mm256_cmpeq_epi16_mask(clamped_exp_i16x16, _mm256_set1_epi16(15));
    __m256i max_mantissa_i16x16 = _mm256_mask_blend_epi16(is_max_exp, _mm256_set1_epi16(7), _mm256_set1_epi16(6));
    __m256i normal_mantissa_i16x16 = _mm256_min_epi16(f16_mantissa_i16x16, max_mantissa_i16x16);
    normal_mantissa_i16x16 = _mm256_mask_blend_epi16(overflow, normal_mantissa_i16x16, _mm256_set1_epi16(0x06));
    __m256i normal_e4m3_i16x16 = _mm256_or_si256(
        _mm256_slli_epi16(sign_i16x16, 7),
        _mm256_or_si256(_mm256_slli_epi16(clamped_exp_i16x16, 3), normal_mantissa_i16x16));

    // Subnormal path: mantissa = round(abs_f16 * 512)
    __m256h abs_f16x16 = _mm256_castsi256_ph(_mm256_and_si256(_mm256_castph_si256(f16x16), _mm256_set1_epi16(0x7FFF)));
    __m256h scaled_f16x16 = _mm256_mul_ph(abs_f16x16, _mm256_castsi256_ph(_mm256_set1_epi16(0x6000))); // 512
    __m256i subnorm_mantissa_i16x16 = _mm256_cvtph_epi16(scaled_f16x16);
    __mmask16 promotes_to_normal = _mm256_cmpgt_epi16_mask(subnorm_mantissa_i16x16, _mm256_set1_epi16(7));
    subnorm_mantissa_i16x16 = _mm256_min_epi16(subnorm_mantissa_i16x16, _mm256_set1_epi16(7));
    subnorm_mantissa_i16x16 = _mm256_max_epi16(subnorm_mantissa_i16x16, _mm256_setzero_si256());
    __m256i subnorm_e4m3_i16x16 = _mm256_or_si256(_mm256_slli_epi16(sign_i16x16, 7), subnorm_mantissa_i16x16);
    __m256i first_normal_e4m3_i16x16 = _mm256_or_si256(_mm256_slli_epi16(sign_i16x16, 7), _mm256_set1_epi16(0x08));
    subnorm_e4m3_i16x16 = _mm256_mask_blend_epi16(promotes_to_normal, subnorm_e4m3_i16x16, first_normal_e4m3_i16x16);

    // Blend: use subnormal result when exp <= 0
    __m256i e4m3_i16x16 = _mm256_mask_blend_epi16(is_subnormal, normal_e4m3_i16x16, subnorm_e4m3_i16x16);

    // Pack 16 i16s to 16 unsigned i8s via AVX-512BW
    return _mm256_cvtepi16_epi8(e4m3_i16x16);
}

/** @brief Convert 16x f16 → 16x e5m2 via bit manipulation (AVX-512 FP16).
 *  F16: S EEEEE MMMMMMMMMM (bias=15). E5M2: S EEEEE MM (bias=15).
 *  Same exponent bias, so just round mantissa from 10 to 2 bits. */
NK_INTERNAL __m128i nk_f16x16_to_e5m2x16_sapphire_(__m256h f16x16) {
    __m256i bits_i16x16 = _mm256_castph_si256(f16x16);
    __m256i sign_i16x16 = _mm256_srli_epi16(bits_i16x16, 15);
    __m256i f16_exp_i16x16 = _mm256_and_si256(_mm256_srli_epi16(bits_i16x16, 10), _mm256_set1_epi16(0x1F));

    // Round mantissa from 10 to 2 bits using RNE (round to nearest, ties to even)
    __m256i significand_i16x16 = _mm256_or_si256(_mm256_and_si256(bits_i16x16, _mm256_set1_epi16(0x03FF)),
                                                 _mm256_set1_epi16(0x0400)); // Add implicit 1 bit
    __m256i lsb_i16x16 = _mm256_and_si256(_mm256_srli_epi16(significand_i16x16, 8), _mm256_set1_epi16(1));
    __m256i rounding_bias_i16x16 = _mm256_add_epi16(_mm256_set1_epi16(0x007F), lsb_i16x16);
    __m256i rounded_sig_i16x16 = _mm256_add_epi16(significand_i16x16, rounding_bias_i16x16);
    __m256i carry_i16x16 = _mm256_srli_epi16(rounded_sig_i16x16, 11); // Carry into exponent if bit 11 set
    __m256i f16_mantissa_i16x16 = _mm256_and_si256(_mm256_srli_epi16(rounded_sig_i16x16, 8), _mm256_set1_epi16(0x03));
    // If carry, mantissa becomes 0 (we rounded up to next power of 2)
    f16_mantissa_i16x16 = _mm256_andnot_si256(_mm256_slli_epi16(carry_i16x16, 15), f16_mantissa_i16x16);
    __m256i e5m2_exp_i16x16 = _mm256_add_epi16(f16_exp_i16x16, carry_i16x16);

    // Detect subnormal (exp <= 0) and overflow (exp > 31)
    __mmask16 is_subnormal = _mm256_cmpeq_epi16_mask(f16_exp_i16x16, _mm256_setzero_si256());
    __mmask16 overflow = _mm256_cmpgt_epi16_mask(e5m2_exp_i16x16, _mm256_set1_epi16(31));

    // Normal path: clamp exp to [1,31], on overflow return infinity
    __m256i clamped_exp_i16x16 = _mm256_max_epi16(e5m2_exp_i16x16, _mm256_set1_epi16(1));
    clamped_exp_i16x16 = _mm256_min_epi16(clamped_exp_i16x16, _mm256_set1_epi16(31));
    __m256i normal_mantissa_i16x16 = _mm256_mask_blend_epi16(overflow, f16_mantissa_i16x16, _mm256_setzero_si256());
    __m256i normal_e5m2_i16x16 = _mm256_or_si256(
        _mm256_slli_epi16(sign_i16x16, 7),
        _mm256_or_si256(_mm256_slli_epi16(clamped_exp_i16x16, 2), normal_mantissa_i16x16));

    // Subnormal path: mantissa = round(abs_f16 * 65536)
    __m256h abs_f16x16 = _mm256_castsi256_ph(_mm256_and_si256(_mm256_castph_si256(f16x16), _mm256_set1_epi16(0x7FFF)));
    __m256h scaled_f16x16 = _mm256_mul_ph(abs_f16x16, _mm256_castsi256_ph(_mm256_set1_epi16(0x7C00))); // 65536 (inf)
    __m256i subnorm_mantissa_i16x16 = _mm256_cvtph_epi16(scaled_f16x16);
    __mmask16 promotes_to_normal = _mm256_cmpgt_epi16_mask(subnorm_mantissa_i16x16, _mm256_set1_epi16(3));
    subnorm_mantissa_i16x16 = _mm256_min_epi16(subnorm_mantissa_i16x16, _mm256_set1_epi16(3));
    subnorm_mantissa_i16x16 = _mm256_max_epi16(subnorm_mantissa_i16x16, _mm256_setzero_si256());
    __m256i subnorm_e5m2_i16x16 = _mm256_or_si256(_mm256_slli_epi16(sign_i16x16, 7), subnorm_mantissa_i16x16);
    __m256i first_normal_e5m2_i16x16 = _mm256_or_si256(_mm256_slli_epi16(sign_i16x16, 7), _mm256_set1_epi16(0x04));
    subnorm_e5m2_i16x16 = _mm256_mask_blend_epi16(promotes_to_normal, subnorm_e5m2_i16x16, first_normal_e5m2_i16x16);

    // Blend: use subnormal result when exp == 0
    __m256i e5m2_i16x16 = _mm256_mask_blend_epi16(is_subnormal, normal_e5m2_i16x16, subnorm_e5m2_i16x16);

    // Pack 16 i16s to 16 unsigned i8s via AVX-512BW
    return _mm256_cvtepi16_epi8(e5m2_i16x16);
}

#pragma endregion - Vectorized Conversions

#pragma region - Public API

NK_PUBLIC void nk_cast_sapphire(void const *from, nk_dtype_t from_type, nk_size_t n, void *to, nk_dtype_t to_type) {
    // Group 1: Conversions to f16 (e4m3 → f16, e5m2 → f16)
    if (to_type == nk_f16_k && (from_type == nk_e4m3_k || from_type == nk_e5m2_k)) {
        nk_e4m3_t const *from_ptr = (nk_e4m3_t const *)from;
        nk_f16_t *to_ptr = (nk_f16_t *)to;
        for (nk_size_t idx = 0; idx < n; idx += 16) {
            nk_size_t remaining = n - idx;
            __mmask16 mask = (remaining >= 16) ? 0xFFFF : (unsigned short)_bzhi_u32(0xFFFF, (unsigned)remaining);
            __m128i in_f8x16 = _mm_maskz_loadu_epi8(mask, from_ptr + idx);
            __m256h out_f16x16 = (from_type == nk_e4m3_k) ? nk_e4m3x16_to_f16x16_sapphire_(in_f8x16)
                                                          : nk_e5m2x16_to_f16x16_sapphire_(in_f8x16);
            _mm256_mask_storeu_epi16(to_ptr + idx, mask, _mm256_castph_si256(out_f16x16));
        }
    }

    // Group 2: Conversions from f16 (f16 → e4m3, f16 → e5m2)
    else if (from_type == nk_f16_k && (to_type == nk_e4m3_k || to_type == nk_e5m2_k)) {
        nk_f16_t const *from_ptr = (nk_f16_t const *)from;
        nk_e4m3_t *to_ptr = (nk_e4m3_t *)to;
        for (nk_size_t idx = 0; idx < n; idx += 16) {
            nk_size_t remaining = n - idx;
            __mmask16 mask = (remaining >= 16) ? 0xFFFF : (unsigned short)_bzhi_u32(0xFFFF, (unsigned)remaining);
            __m256h in_f16x16 = _mm256_castsi256_ph(_mm256_maskz_loadu_epi16(mask, from_ptr + idx));
            __m128i out_f8x16 = (to_type == nk_e4m3_k) ? nk_f16x16_to_e4m3x16_sapphire_(in_f16x16)
                                                       : nk_f16x16_to_e5m2x16_sapphire_(in_f16x16);
            _mm_mask_storeu_epi8(to_ptr + idx, mask, out_f8x16);
        }
    }

    // Default: delegate to Ice for all other conversions
    else nk_cast_icelake(from, from_type, n, to, to_type);
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

#endif // NK_TARGET_SAPPHIRE
#endif // NK_TARGET_X86_
#endif // NK_CAST_SAPPHIRE_H
