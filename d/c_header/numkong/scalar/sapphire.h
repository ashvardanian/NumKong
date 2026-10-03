/**
 *  @brief SIMD-accelerated Scalar Math Helpers for Sapphire Rapids.
 *  @file include/numkong/scalar/sapphire.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalar.h
 *
 *  Provides native AVX-512 FP16 scalar ordering via `VCOMISH`.
 */
#ifndef NK_SCALAR_SAPPHIRE_H
#define NK_SCALAR_SAPPHIRE_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIRE

#include "numkong/types.h"

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

NK_PUBLIC int nk_f16_order_sapphire(nk_f16_t a, nk_f16_t b) {
    __m128h a_f16x8 = _mm_castsi128_ph(_mm_cvtsi32_si128(a));
    __m128h b_f16x8 = _mm_castsi128_ph(_mm_cvtsi32_si128(b));
    return _mm_comigt_sh(a_f16x8, b_f16x8) - _mm_comilt_sh(a_f16x8, b_f16x8);
}
NK_PUBLIC nk_f16_t nk_f16_sqrt_sapphire(nk_f16_t x) {
    __m128h x_f16x8 = _mm_castsi128_ph(_mm_cvtsi32_si128(x));
    return (nk_f16_t)_mm_cvtsi128_si32(_mm_castph_si128(_mm_sqrt_sh(x_f16x8, x_f16x8)));
}
NK_PUBLIC nk_f16_t nk_f16_rsqrt_sapphire(nk_f16_t x) {
    __m128h x_f16x8 = _mm_castsi128_ph(_mm_cvtsi32_si128(x));
    return (nk_f16_t)_mm_cvtsi128_si32(_mm_castph_si128(_mm_rsqrt_sh(x_f16x8, x_f16x8)));
}
NK_PUBLIC nk_f16_t nk_f16_fma_sapphire(nk_f16_t a, nk_f16_t b, nk_f16_t c) {
    __m128h a_f16x8 = _mm_castsi128_ph(_mm_cvtsi32_si128(a));
    __m128h b_f16x8 = _mm_castsi128_ph(_mm_cvtsi32_si128(b));
    __m128h c_f16x8 = _mm_castsi128_ph(_mm_cvtsi32_si128(c));
    return (nk_f16_t)_mm_cvtsi128_si32(_mm_castph_si128(_mm_fmadd_sh(a_f16x8, b_f16x8, c_f16x8)));
}

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
#endif // NK_SCALAR_SAPPHIRE_H
