/**
 *  @brief SIMD-accelerated Scalar Math Helpers for Haswell.
 *  @file include/numkong/scalar/haswell.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalar.h
 *
 *  @section scalars_haswell_instructions Key AVX2/FMA Scalar Instructions
 *
 *      Intrinsic                   Instruction                     Latency     Throughput  Ports
 *      _mm_sqrt_ps                 VSQRTPS (XMM, XMM)              11cy        7cy         p0
 *      _mm_sqrt_pd                 VSQRTPD (XMM, XMM)              16cy        12cy        p0
 *      _mm_fmadd_ss                VFMADD (XMM, XMM, XMM)          5cy         0.5/cy      p01
 *      _mm_fmadd_sd                VFMADD (XMM, XMM, XMM)          5cy         0.5/cy      p01
 *      _mm_cvtps_ph                VCVTPS2PH (XMM, XMM, I8)        4cy         1/cy        p01+p5
 *      _mm_cvtph_ps                VCVTPH2PS (XMM, XMM)            5cy         1/cy        p01
 */
#ifndef NK_SCALAR_HASWELL_H
#define NK_SCALAR_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

NK_PUBLIC nk_f32_t nk_f32_sqrt_haswell(nk_f32_t x) { return _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(x))); }
NK_PUBLIC nk_f64_t nk_f64_sqrt_haswell(nk_f64_t x) { return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(x))); }
NK_PUBLIC nk_f32_t nk_f32_rsqrt_haswell(nk_f32_t x) {
    __m128 x_f32x4 = _mm_set_ss(x);
    __m128 estimate_f32x4 = _mm_rsqrt_ss(x_f32x4);
    __m128 refinement_f32x4 = _mm_mul_ss(_mm_mul_ss(x_f32x4, estimate_f32x4), estimate_f32x4);
    refinement_f32x4 = _mm_sub_ss(_mm_set_ss(3.0f), refinement_f32x4);
    return _mm_cvtss_f32(_mm_mul_ss(_mm_mul_ss(_mm_set_ss(0.5f), estimate_f32x4), refinement_f32x4));
}
NK_PUBLIC nk_f64_t nk_f64_rsqrt_haswell(nk_f64_t x) { return 1.0 / nk_f64_sqrt_haswell(x); }
NK_PUBLIC nk_f32_t nk_f32_fma_haswell(nk_f32_t a, nk_f32_t b, nk_f32_t c) {
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}
NK_PUBLIC nk_f64_t nk_f64_fma_haswell(nk_f64_t a, nk_f64_t b, nk_f64_t c) {
    return _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)));
}
NK_PUBLIC nk_f16_t nk_f16_sqrt_haswell(nk_f16_t x) {
    __m128 x_f32x4 = _mm_cvtph_ps(_mm_cvtsi32_si128(x));
    return (nk_f16_t)_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_sqrt_ps(x_f32x4), _MM_FROUND_TO_NEAREST_INT));
}
NK_PUBLIC nk_f16_t nk_f16_rsqrt_haswell(nk_f16_t x) {
    __m128 x_f32x4 = _mm_cvtph_ps(_mm_cvtsi32_si128(x));
    __m128 estimate_f32x4 = _mm_rsqrt_ss(x_f32x4);
    __m128 refinement_f32x4 = _mm_mul_ss(_mm_mul_ss(x_f32x4, estimate_f32x4), estimate_f32x4);
    refinement_f32x4 = _mm_sub_ss(_mm_set_ss(3.0f), refinement_f32x4);
    estimate_f32x4 = _mm_mul_ss(_mm_mul_ss(_mm_set_ss(0.5f), estimate_f32x4), refinement_f32x4);
    return (nk_f16_t)_mm_cvtsi128_si32(_mm_cvtps_ph(estimate_f32x4, _MM_FROUND_TO_NEAREST_INT));
}
NK_PUBLIC nk_f16_t nk_f16_fma_haswell(nk_f16_t a, nk_f16_t b, nk_f16_t c) {
    __m128 a_f32x4 = _mm_cvtph_ps(_mm_cvtsi32_si128(a));
    __m128 b_f32x4 = _mm_cvtph_ps(_mm_cvtsi32_si128(b));
    __m128 c_f32x4 = _mm_cvtph_ps(_mm_cvtsi32_si128(c));
    return (nk_f16_t)_mm_cvtsi128_si32(
        _mm_cvtps_ph(_mm_fmadd_ss(a_f32x4, b_f32x4, c_f32x4), _MM_FROUND_TO_NEAREST_INT));
}
NK_PUBLIC nk_u8_t nk_u8_saturating_add_haswell(nk_u8_t a, nk_u8_t b) {
    return (nk_u8_t)_mm_cvtsi128_si32(_mm_adds_epu8(_mm_cvtsi32_si128(a), _mm_cvtsi32_si128(b)));
}
NK_PUBLIC nk_i8_t nk_i8_saturating_add_haswell(nk_i8_t a, nk_i8_t b) {
    return (nk_i8_t)_mm_cvtsi128_si32(_mm_adds_epi8(_mm_cvtsi32_si128(a), _mm_cvtsi32_si128(b)));
}
NK_PUBLIC nk_u16_t nk_u16_saturating_add_haswell(nk_u16_t a, nk_u16_t b) {
    return (nk_u16_t)_mm_cvtsi128_si32(_mm_adds_epu16(_mm_cvtsi32_si128(a), _mm_cvtsi32_si128(b)));
}
NK_PUBLIC nk_i16_t nk_i16_saturating_add_haswell(nk_i16_t a, nk_i16_t b) {
    return (nk_i16_t)_mm_cvtsi128_si32(_mm_adds_epi16(_mm_cvtsi32_si128(a), _mm_cvtsi32_si128(b)));
}
NK_PUBLIC nk_u64_t nk_u64_saturating_mul_haswell(nk_u64_t a, nk_u64_t b) {
    unsigned long long high;
    unsigned long long low = _mulx_u64(a, b, &high);
    return high ? 18446744073709551615ull : low;
}
NK_PUBLIC nk_i64_t nk_i64_saturating_mul_haswell(nk_i64_t a, nk_i64_t b) {
    int sign = (a < 0) ^ (b < 0);
    nk_u64_t abs_a = a < 0 ? -(nk_u64_t)a : (nk_u64_t)a;
    nk_u64_t abs_b = b < 0 ? -(nk_u64_t)b : (nk_u64_t)b;
    unsigned long long high;
    unsigned long long low = _mulx_u64(abs_a, abs_b, &high);
    if (high || (sign && low > 9223372036854775808ull) || (!sign && low > 9223372036854775807ull))
        return sign ? (-9223372036854775807ll - 1ll) : 9223372036854775807ll;
    return sign ? -(nk_i64_t)low : (nk_i64_t)low;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_SCALAR_HASWELL_H
