/**
 *  @brief SIMD-accelerated Scalar Math Helpers for LoongArch LASX.
 *  @file include/numkong/scalar/loongsonasx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/scalar.h
 *
 *  LASX provides `xvfrsqrt` (full-precision reciprocal sqrt) and `xvfsqrt`
 *  (full-precision sqrt). No Newton-Raphson refinement needed.
 *  Full-precision sqrt uses the hardware `xvfsqrt` instruction.
 *  Broadcast via `xvreplgr2vr`, extract via `xvpickve2gr` — no memory round-trips.
 */
#ifndef NK_SCALAR_LOONGSONASX_H
#define NK_SCALAR_LOONGSONASX_H

#if NK_TARGET_LOONGARCH64_
#if NK_TARGET_LOONGSONASX

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Broadcast f32 scalar into all 4 lanes of a 128-bit register (GCC/Clang portable). */
NK_HELPER_INLINE __m128 nk_xvreplgr2vr_s_128_(float x) {
    nk_fui32_t c;
    c.f = x;
    return (__m128)__lsx_vreplgr2vr_w((int)c.u);
}

/** @brief Broadcast f32 scalar into all 8 lanes of a 256-bit register (GCC/Clang portable). */
NK_HELPER_INLINE __m256 nk_xvfreplgr2vr_s_(float x) {
    nk_fui32_t c;
    c.f = x;
    return (__m256)__lasx_xvreplgr2vr_w((int)c.u);
}

/** @brief Broadcast f64 scalar into all 4 lanes of a 256-bit register (GCC/Clang portable). */
NK_HELPER_INLINE __m256d nk_xvfreplgr2vr_d_(double x) {
    nk_fui64_t c;
    c.f = x;
    return (__m256d)__lasx_xvreplgr2vr_d((long long)c.u);
}

NK_API_COMPTIME nk_f32_t nk_f32_rsqrt_loongsonasx(nk_f32_t x) {
    // xvfrsqrt.s is full precision — no Newton-Raphson needed
    __m256 x_f32x8 = nk_xvfreplgr2vr_s_(x);
    __m256 result_f32x8 = __lasx_xvfrsqrt_s(x_f32x8);
    nk_fui32_t c;
    c.u = (nk_u32_t)__lasx_xvpickve2gr_w((__m256i)result_f32x8, 0);
    return c.f;
}

NK_API_COMPTIME nk_f32_t nk_f32_sqrt_loongsonasx(nk_f32_t x) { return x > 0 ? x * nk_f32_rsqrt_loongsonasx(x) : 0; }

NK_API_COMPTIME nk_f64_t nk_f64_sqrt_loongsonasx(nk_f64_t x) {
    __m256d x_f64x4 = nk_xvfreplgr2vr_d_(x);
    __m256d result_f64x4 = __lasx_xvfsqrt_d(x_f64x4);
    nk_fui64_t c;
    c.u = (nk_u64_t)__lasx_xvpickve2gr_du((__m256i)result_f64x4, 0);
    return c.f;
}

NK_API_COMPTIME nk_f64_t nk_f64_rsqrt_loongsonasx(nk_f64_t x) { return 1.0 / nk_f64_sqrt_loongsonasx(x); }

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_LOONGSONASX
#endif // NK_TARGET_LOONGARCH64_
#endif // NK_SCALAR_LOONGSONASX_H
