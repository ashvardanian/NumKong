/**
 *  @file include/numkong/trigonometry/sapphire.h
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief SIMD-accelerated trigonometric functions for Sapphire Rapids.
 *
 *  @sa include/numkong/trigonometry.h
 *
 *  @section trigonometry_sapphire_instructions AVX-512 FP16 Instructions
 *
 *  @verbatim
 *  Intrinsic             Instruction
 *  _mm512_fmadd_ph       VFMADD231PH (ZMM, ZMM, ZMM)
 *  _mm512_fnmadd_ph      VFNMADD231PH (ZMM, ZMM, ZMM)
 *  _mm512_roundscale_ph  VRNDSCALEPH (ZMM, ZMM, I8)
 *  _mm512_div_ph         VDIVPH (ZMM, ZMM, ZMM)
 *  _mm512_cmp_ph_mask    VCMPPH (K, ZMM, ZMM, I8)
 *  _mm512_cvttph_epi16   VCVTTPH2W (ZMM, ZMM)
 *  @endverbatim
 *
 *  Sine, cosine and arctangent evaluate in F16 on 32 lanes, with polynomials fitted for F16.
 *  Sine and cosine reduce in F16 while every lane stays within |x| ≤ 256, and in F32 otherwise.
 */
#ifndef NUMKONG_TRIGONOMETRY_SAPPHIRE_H
#define NUMKONG_TRIGONOMETRY_SAPPHIRE_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_SAPPHIRE

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                        \
    __attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,f16c,fma,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "f16c", "fma", "bmi", "bmi2")
#endif

/** Broadcasts an F16 bit pattern without requiring a native scalar half type. */
NUMKONG_INLINE __m512h nk_set1_f16x32_sapphire_(nk_u16_t bits) {
    return nk_m512h_from_m512i_(_mm512_set1_epi16((short)bits));
}

/** Sine of 32 F16 angles already reduced to about [-π/2, π/2], odd in the reduced angle. */
NUMKONG_INLINE __m512h nk_sin_reduced_f16x32_sapphire_(__m512h reduced_f16x32) {
    // Degree-7 odd polynomial with a unit linear term, coefficients searched in F16 arithmetic
    __m512h const squared_f16x32 = _mm512_mul_ph(reduced_f16x32, reduced_f16x32);
    __m512h polynomial_f16x32 = _mm512_fmadd_ph(nk_set1_f16x32_sapphire_(0x8A00 /* -0.00018310546875 */),
                                                squared_f16x32,
                                                nk_set1_f16x32_sapphire_(0x2039 /* +0.00824737548828125 */));
    polynomial_f16x32 = _mm512_fmadd_ph(polynomial_f16x32, squared_f16x32,
                                        nk_set1_f16x32_sapphire_(0xB154 /* -0.16650390625 */));
    return _mm512_fmadd_ph(_mm512_mul_ph(reduced_f16x32, squared_f16x32), polynomial_f16x32, reduced_f16x32);
}

/** Subtracts @p multiples_f16x32 of π from @p angles_f16x32, for F16 angles up to 256. */
NUMKONG_INLINE __m512h nk_reduce_pi_f16x32_sapphire_(__m512h angles_f16x32, __m512h multiples_f16x32) {
    // π in three F16 parts, the last scaled by 2¹² to stay clear of F16 subnormals
    __m512h reduced_f16x32 = _mm512_fnmadd_ph(multiples_f16x32, nk_set1_f16x32_sapphire_(0x4248 /* 3.140625 */),
                                              angles_f16x32);
    reduced_f16x32 = _mm512_fnmadd_ph(multiples_f16x32, nk_set1_f16x32_sapphire_(0x13ED /* 0.0009675025939941406 */),
                                      reduced_f16x32);
    __m512h const scaled_multiples_f16x32 = _mm512_mul_ph(multiples_f16x32,
                                                          nk_set1_f16x32_sapphire_(0x0C00 /* 0.000244140625 */));
    return _mm512_fnmadd_ph(scaled_multiples_f16x32, nk_set1_f16x32_sapphire_(0x1111 /* 0.0006184577941894531 */),
                            reduced_f16x32);
}

/** Subtracts @p multiples_f32x16 of π from @p angles_f32x16, for F16 angles of any magnitude. */
NUMKONG_INLINE __m512 nk_reduce_pi_f32x16_sapphire_(__m512 angles_f32x16, __m512 multiples_f32x16) {
    __m512 const reduced_f32x16 = _mm512_fnmadd_ps(multiples_f32x16, _mm512_set1_ps(3.140625f), angles_f32x16);
    return _mm512_fnmadd_ps(multiples_f32x16, _mm512_set1_ps(9.676535897e-4f), reduced_f32x16);
}

/** Flips the sign of @p values_f32x16 in the lanes where @p flips_i32x16 is odd. */
NUMKONG_INLINE __m512 nk_flip_odd_f32x16_sapphire_(__m512 values_f32x16, __m512i flips_i32x16) {
    __m512i const signs_u32x16 = _mm512_slli_epi32(flips_i32x16, 31);
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(values_f32x16), signs_u32x16));
}

/** Flips the sign of @p values_f16x32 in the lanes where @p flips_i16x32 is odd. */
NUMKONG_INLINE __m512h nk_flip_odd_f16x32_sapphire_(__m512h values_f16x32, __m512i flips_i16x32) {
    __m512i const signs_u16x32 = _mm512_slli_epi16(flips_i16x32, 15);
    return _mm512_castsi512_ph(_mm512_xor_si512(_mm512_castph_si512(values_f16x32), signs_u16x32));
}

/** Reduces 16 F32 angles by the nearest multiple of π, negated where that multiple is odd. */
NUMKONG_INLINE __m512 nk_sin_reduce_f32x16_sapphire_(__m512 angles_f32x16) {
    __m512 const quotients_f32x16 = _mm512_mul_ps(angles_f32x16, _mm512_set1_ps(0.31830988618379067154f));
    __m512i const multiples_i32x16 = _mm512_cvt_roundps_epi32(quotients_f32x16,
                                                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 const reduced_f32x16 = nk_reduce_pi_f32x16_sapphire_(angles_f32x16, _mm512_cvtepi32_ps(multiples_i32x16));
    return nk_flip_odd_f32x16_sapphire_(reduced_f32x16, multiples_i32x16);
}

/** Reduces 16 angles by the nearest odd multiple of π/2, signed so its sine is their cosine. */
NUMKONG_INLINE __m512 nk_cos_reduce_f32x16_sapphire_(__m512 angles_f32x16) {
    __m512 const quotients_f32x16 = _mm512_fmsub_ps(angles_f32x16, _mm512_set1_ps(0.31830988618379067154f),
                                                    _mm512_set1_ps(0.5f));
    __m512i const multiples_i32x16 = _mm512_cvt_roundps_epi32(quotients_f32x16,
                                                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 const offsets_f32x16 = _mm512_add_ps(_mm512_cvtepi32_ps(multiples_i32x16), _mm512_set1_ps(0.5f));
    __m512 const reduced_f32x16 = nk_reduce_pi_f32x16_sapphire_(angles_f32x16, offsets_f32x16);
    return nk_flip_odd_f32x16_sapphire_(reduced_f32x16, _mm512_add_epi32(multiples_i32x16, _mm512_set1_epi32(1)));
}

/** Sine of 32 F16 angles within one F16 ULP. */
NUMKONG_INLINE __m512h nk_sin_f16x32_sapphire_(__m512h angles_f16x32) {
    __m512h reduced_f16x32;
    // The F16 reduction holds one ULP only up to |x| ≤ 256
    __m512h const magnitudes_f16x32 = _mm512_abs_ph(angles_f16x32);
    if (_mm512_cmp_ph_mask(magnitudes_f16x32, nk_set1_f16x32_sapphire_(0x5C00 /* 256.0 */), _CMP_GT_OQ)) {
        __m512i const angles_b16x32 = _mm512_castph_si512(angles_f16x32);
        __m512 const low_f32x16 = nk_sin_reduce_f32x16_sapphire_(
            _mm512_cvtph_ps(_mm512_castsi512_si256(angles_b16x32)));
        __m512 const high_f32x16 = nk_sin_reduce_f32x16_sapphire_(
            _mm512_cvtph_ps(_mm512_extracti64x4_epi64(angles_b16x32, 1)));
        __m256i const low_b16x16 = _mm512_cvtps_ph(low_f32x16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i const high_b16x16 = _mm512_cvtps_ph(high_f32x16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        reduced_f16x32 = _mm512_castsi512_ph(_mm512_inserti64x4(_mm512_castsi256_si512(low_b16x16), high_b16x16, 1));
    }
    else {
        __m512h const quotients_f16x32 = _mm512_mul_ph(angles_f16x32,
                                                       nk_set1_f16x32_sapphire_(0x3518 /* 0.31830988618379067154 */));
        __m512h const multiples_f16x32 = _mm512_roundscale_ph(quotients_f16x32,
                                                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        reduced_f16x32 = nk_reduce_pi_f16x32_sapphire_(angles_f16x32, multiples_f16x32);
        reduced_f16x32 = nk_flip_odd_f16x32_sapphire_(reduced_f16x32, _mm512_cvttph_epi16(multiples_f16x32));
    }
    return nk_sin_reduced_f16x32_sapphire_(reduced_f16x32);
}

/** Cosine of 32 F16 angles within one F16 ULP. */
NUMKONG_INLINE __m512h nk_cos_f16x32_sapphire_(__m512h angles_f16x32) {
    __m512h reduced_f16x32;
    // The F16 reduction holds one ULP only up to |x| ≤ 256
    __m512h const magnitudes_f16x32 = _mm512_abs_ph(angles_f16x32);
    if (_mm512_cmp_ph_mask(magnitudes_f16x32, nk_set1_f16x32_sapphire_(0x5C00 /* 256.0 */), _CMP_GT_OQ)) {
        __m512i const angles_b16x32 = _mm512_castph_si512(angles_f16x32);
        __m512 const low_f32x16 = nk_cos_reduce_f32x16_sapphire_(
            _mm512_cvtph_ps(_mm512_castsi512_si256(angles_b16x32)));
        __m512 const high_f32x16 = nk_cos_reduce_f32x16_sapphire_(
            _mm512_cvtph_ps(_mm512_extracti64x4_epi64(angles_b16x32, 1)));
        __m256i const low_b16x16 = _mm512_cvtps_ph(low_f32x16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256i const high_b16x16 = _mm512_cvtps_ph(high_f32x16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        reduced_f16x32 = _mm512_castsi512_ph(_mm512_inserti64x4(_mm512_castsi256_si512(low_b16x16), high_b16x16, 1));
    }
    else {
        __m512h const quotients_f16x32 = _mm512_fmadd_ph(angles_f16x32,
                                                         nk_set1_f16x32_sapphire_(0x3518 /* 0.31830988618379067154 */),
                                                         nk_set1_f16x32_sapphire_(0xB800 /* -0.5 */));
        __m512h const multiples_f16x32 = _mm512_roundscale_ph(quotients_f16x32,
                                                              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512h const offsets_f16x32 = _mm512_add_ph(multiples_f16x32, nk_set1_f16x32_sapphire_(0x3800 /* 0.5 */));
        __m512i const flips_i16x32 = _mm512_add_epi16(_mm512_cvttph_epi16(multiples_f16x32), _mm512_set1_epi16(1));
        reduced_f16x32 = nk_reduce_pi_f16x32_sapphire_(angles_f16x32, offsets_f16x32);
        reduced_f16x32 = nk_flip_odd_f16x32_sapphire_(reduced_f16x32, flips_i16x32);
    }
    return nk_sin_reduced_f16x32_sapphire_(reduced_f16x32);
}

/** Arctangent of 32 F16 values within one F16 ULP. */
NUMKONG_INLINE __m512h nk_atan_f16x32_sapphire_(__m512h values_f16x32) {
    __m512h const one_f16x32 = nk_set1_f16x32_sapphire_(0x3C00 /* 1.0 */);

    // Fold |x| > 1 into [0, 1] through atan(x) = π/2 - atan(1/x)
    __m512h const magnitudes_f16x32 = _mm512_abs_ph(values_f16x32);
    __mmask32 const folded_m32 = _mm512_cmp_ph_mask(magnitudes_f16x32, one_f16x32, _CMP_GT_OQ);
    __m512h const reduced_f16x32 = _mm512_mask_div_ph(magnitudes_f16x32, folded_m32, one_f16x32, magnitudes_f16x32);

    // Folded lanes add the low part of π/2 first, as its high part alone is half an F16 ULP off
    __m512h const signed_f16x32 = _mm512_mask_sub_ph(reduced_f16x32, folded_m32, _mm512_setzero_ph(), reduced_f16x32);
    __m512h const bases_f16x32 = _mm512_mask_sub_ph(
        reduced_f16x32, folded_m32, nk_set1_f16x32_sapphire_(0x0FED /* 0.0004837512969970703 */), reduced_f16x32);

    // Degree-7 odd polynomial with a unit linear term, coefficients searched in F16 arithmetic
    __m512h const squared_f16x32 = _mm512_mul_ph(reduced_f16x32, reduced_f16x32);
    __m512h polynomial_f16x32 = _mm512_fmadd_ph(nk_set1_f16x32_sapphire_(0xA9FB /* -0.046722412109375 */),
                                                squared_f16x32,
                                                nk_set1_f16x32_sapphire_(0x311F /* +0.1600341796875 */));
    polynomial_f16x32 = _mm512_fmadd_ph(polynomial_f16x32, squared_f16x32,
                                        nk_set1_f16x32_sapphire_(0xB540 /* -0.328125 */));
    __m512h results_f16x32 = _mm512_fmadd_ph(_mm512_mul_ph(signed_f16x32, squared_f16x32), polynomial_f16x32,
                                             bases_f16x32);
    results_f16x32 = _mm512_mask_add_ph(results_f16x32, folded_m32, results_f16x32,
                                        nk_set1_f16x32_sapphire_(0x3E48 /* 1.5703125 */));

    // Results are non-negative, so XOR-ing the input's sign bit copies it
    __m512i const signs_b16x32 = _mm512_and_si512(_mm512_castph_si512(values_f16x32), _mm512_set1_epi16(-0x8000));
    return _mm512_castsi512_ph(_mm512_xor_si512(_mm512_castph_si512(results_f16x32), signs_b16x32));
}

NUMKONG_API nk_status_t nk_trig_sin_f16_sapphire(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512h angles_f16x32 = _mm512_castsi512_ph(_mm512_loadu_si512(ins + i));
        _mm512_storeu_si512(outs + i, _mm512_castph_si512(nk_sin_f16x32_sapphire_(angles_f16x32)));
    }
    if (i < n) {
        __mmask32 mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned)(n - i));
        __m512h angles_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask_m32, ins + i));
        _mm512_mask_storeu_epi16(outs + i, mask_m32, _mm512_castph_si512(nk_sin_f16x32_sapphire_(angles_f16x32)));
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_trig_cos_f16_sapphire(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512h angles_f16x32 = _mm512_castsi512_ph(_mm512_loadu_si512(ins + i));
        _mm512_storeu_si512(outs + i, _mm512_castph_si512(nk_cos_f16x32_sapphire_(angles_f16x32)));
    }
    if (i < n) {
        __mmask32 mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned)(n - i));
        __m512h angles_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask_m32, ins + i));
        _mm512_mask_storeu_epi16(outs + i, mask_m32, _mm512_castph_si512(nk_cos_f16x32_sapphire_(angles_f16x32)));
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_trig_atan_f16_sapphire(nk_f16_t const *ins, nk_size_t n, nk_f16_t *outs,
                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m512h values_f16x32 = _mm512_castsi512_ph(_mm512_loadu_si512(ins + i));
        _mm512_storeu_si512(outs + i, _mm512_castph_si512(nk_atan_f16x32_sapphire_(values_f16x32)));
    }
    if (i < n) {
        __mmask32 mask_m32 = (__mmask32)_bzhi_u32(0xFFFFFFFF, (unsigned)(n - i));
        __m512h values_f16x32 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask_m32, ins + i));
        _mm512_mask_storeu_epi16(outs + i, mask_m32, _mm512_castph_si512(nk_atan_f16x32_sapphire_(values_f16x32)));
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_SAPPHIRE
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_TRIGONOMETRY_SAPPHIRE_H
