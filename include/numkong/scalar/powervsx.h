/**
 *  @brief SIMD-accelerated Scalar Math Helpers for Power VSX.
 *  @file include/numkong/scalar/powervsx.h
 *  @author Ash Vardanian
 *  @date March 24, 2026
 *
 *  @sa include/numkong/scalar.h
 *
 *  @section scalars_powervsx_instructions Key Power VSX Scalar Instructions
 *
 *      Instruction     Description                  Latency
 *      xssqrtsp        Scalar √ (f32)               26cy
 *      xssqrtdp        Scalar √ (f64)               33cy
 *      xsrsqrtesp      Scalar 1/√ estimate (f32)    6cy
 *      xsrsqrtedp      Scalar 1/√ estimate (f64)    6cy
 *      xsmaddadp       Scalar FMA (f64)             5cy
 *      xsmaddasp       Scalar FMA (f32)             5cy
 */
#ifndef NK_SCALAR_POWERVSX_H
#define NK_SCALAR_POWERVSX_H

#if NK_TARGET_POWER64_
#if NK_TARGET_POWERVSX

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

NK_API_COMPTIME nk_f32_t nk_f32_sqrt_powervsx(nk_f32_t x) {
    nk_f32_t result;
    __asm__("xssqrtsp %0, %1" : "=f"(result) : "f"(x));
    return result;
}
NK_API_COMPTIME nk_f64_t nk_f64_sqrt_powervsx(nk_f64_t x) {
    nk_f64_t result;
    __asm__("xssqrtdp %0, %1" : "=d"(result) : "d"(x));
    return result;
}
NK_API_COMPTIME nk_f32_t nk_f32_rsqrt_powervsx(nk_f32_t x) {
    // xsrsqrtesp → ~12-bit estimate, then 2 Newton→Raphson iterations → ~24-bit precision
    nk_f32_t r;
    __asm__("xsrsqrtesp %0, %1" : "=f"(r) : "f"(x));
    // Newton→Raphson: r = r * (3 - x * r * r) / 2
    nk_f32_t half_x = x * 0.5f;
    nk_f32_t three_half = 1.5f;
    r = r * (three_half - half_x * r * r);
    r = r * (three_half - half_x * r * r);
    return r;
}
NK_API_COMPTIME nk_f64_t nk_f64_rsqrt_powervsx(nk_f64_t x) {
    // xsrsqrtedp → ~14-bit estimate, then 3 Newton→Raphson iterations → ~48-bit precision
    nk_f64_t r;
    __asm__("xsrsqrtedp %0, %1" : "=d"(r) : "d"(x));
    // Newton→Raphson: r = r * (3 - x * r * r) / 2
    nk_f64_t half_x = x * 0.5;
    nk_f64_t three_half = 1.5;
    r = r * (three_half - half_x * r * r);
    r = r * (three_half - half_x * r * r);
    r = r * (three_half - half_x * r * r);
    return r;
}
NK_API_COMPTIME nk_f32_t nk_f32_fma_powervsx(nk_f32_t a, nk_f32_t b, nk_f32_t c) {
    // xsmaddasp: result = a * b + c (scalar f32 FMA)
    nk_f32_t r = c;
    __asm__("xsmaddasp %0, %1, %2" : "+f"(r) : "f"(a), "f"(b));
    return r;
}
NK_API_COMPTIME nk_f64_t nk_f64_fma_powervsx(nk_f64_t a, nk_f64_t b, nk_f64_t c) {
    // xsmaddadp: result = a * b + c (scalar f64 FMA)
    nk_f64_t r = c;
    __asm__("xsmaddadp %0, %1, %2" : "+d"(r) : "d"(a), "d"(b));
    return r;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_POWERVSX
#endif // NK_TARGET_POWER64_
#endif // NK_SCALAR_POWERVSX_H
