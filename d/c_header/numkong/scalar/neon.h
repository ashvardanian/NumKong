/**
 *  @brief SIMD-accelerated Scalar Math Helpers for NEON.
 *  @file include/numkong/scalar/neon.h
 *  @author Ash Vardanian
 *  @date March 1, 2026
 *
 *  @sa include/numkong/scalar.h
 *
 *  @section scalars_neon_instructions Key NEON Scalar Instructions
 *
 *      Intrinsic           Instruction                     Latency     Throughput
 *      vsqrt_f32           FSQRT (S)                       9-12cy      0.25/cy
 *      vsqrt_f64           FSQRT (D)                       12-18cy     0.25/cy
 *      vfmas_f32           FMADD (S, S, S, S)              4cy         2/cy
 *      vfmad_f64           FMADD (D, D, D, D)              4cy         2/cy
 *      vqaddb_u8           UQADD (B)                       1cy         4/cy
 *      vqaddb_s8           SQADD (B)                       1cy         4/cy
 */
#ifndef NK_SCALAR_NEON_H
#define NK_SCALAR_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

NK_PUBLIC nk_f32_t nk_f32_sqrt_neon(nk_f32_t x) { return vget_lane_f32(vsqrt_f32(vdup_n_f32(x)), 0); }
NK_PUBLIC nk_f64_t nk_f64_sqrt_neon(nk_f64_t x) { return vget_lane_f64(vsqrt_f64(vdup_n_f64(x)), 0); }
NK_PUBLIC nk_f32_t nk_f32_rsqrt_neon(nk_f32_t x) {
    nk_f32_t r = vrsqrtes_f32(x);
    r *= vrsqrtss_f32(x * r, r);
    r *= vrsqrtss_f32(x * r, r);
    return r;
}
NK_PUBLIC nk_f64_t nk_f64_rsqrt_neon(nk_f64_t x) {
    nk_f64_t r = vrsqrted_f64(x);
    r *= vrsqrtsd_f64(x * r, r);
    r *= vrsqrtsd_f64(x * r, r);
    r *= vrsqrtsd_f64(x * r, r);
    return r;
}
NK_PUBLIC nk_f32_t nk_f32_fma_neon(nk_f32_t a, nk_f32_t b, nk_f32_t c) {
    // MSVC lacks both GCC inline asm and scalar ACLE FMA intrinsics (vfmas_f32/vfmad_f64).
    // GCC/Clang: use inline asm for scalar FMADD.
    // MSVC: use vector FMA + lane extract (compiler may optimize to scalar FMADD).
#if defined(_MSC_VER)
    return vget_lane_f32(vfma_f32(vdup_n_f32(c), vdup_n_f32(a), vdup_n_f32(b)), 0);
#else
    nk_f32_t r;
    __asm__("fmadd %s0, %s1, %s2, %s3" : "=w"(r) : "w"(a), "w"(b), "w"(c));
    return r;
#endif
}
NK_PUBLIC nk_f64_t nk_f64_fma_neon(nk_f64_t a, nk_f64_t b, nk_f64_t c) {
    // MSVC lacks both GCC inline asm and scalar ACLE FMA intrinsics (vfmas_f32/vfmad_f64).
    // GCC/Clang: use inline asm for scalar FMADD.
    // MSVC: use vector FMA + lane extract (compiler may optimize to scalar FMADD).
#if defined(_MSC_VER)
    return vget_lane_f64(vfma_f64(vdup_n_f64(c), vdup_n_f64(a), vdup_n_f64(b)), 0);
#else
    nk_f64_t r;
    __asm__("fmadd %d0, %d1, %d2, %d3" : "=w"(r) : "w"(a), "w"(b), "w"(c));
    return r;
#endif
}

NK_PUBLIC nk_u8_t nk_u8_saturating_add_neon(nk_u8_t a, nk_u8_t b) { return vqaddb_u8(a, b); }
NK_PUBLIC nk_i8_t nk_i8_saturating_add_neon(nk_i8_t a, nk_i8_t b) { return vqaddb_s8(a, b); }
NK_PUBLIC nk_u16_t nk_u16_saturating_add_neon(nk_u16_t a, nk_u16_t b) { return vqaddh_u16(a, b); }
NK_PUBLIC nk_i16_t nk_i16_saturating_add_neon(nk_i16_t a, nk_i16_t b) { return vqaddh_s16(a, b); }
NK_PUBLIC nk_u32_t nk_u32_saturating_add_neon(nk_u32_t a, nk_u32_t b) { return vqadds_u32(a, b); }
NK_PUBLIC nk_i32_t nk_i32_saturating_add_neon(nk_i32_t a, nk_i32_t b) { return vqadds_s32(a, b); }
NK_PUBLIC nk_u64_t nk_u64_saturating_add_neon(nk_u64_t a, nk_u64_t b) { return vqaddd_u64(a, b); }
NK_PUBLIC nk_i64_t nk_i64_saturating_add_neon(nk_i64_t a, nk_i64_t b) { return vqaddd_s64(a, b); }

NK_INTERNAL nk_u64_t nk_u64_mulhigh_neon_(nk_u64_t a, nk_u64_t b) {
#if defined(_MSC_VER)
    return __umulh(a, b);
#else
    nk_u64_t high;
    __asm__("umulh %0, %1, %2" : "=r"(high) : "r"(a), "r"(b));
    return high;
#endif
}
NK_PUBLIC nk_u64_t nk_u64_saturating_mul_neon(nk_u64_t a, nk_u64_t b) {
    return nk_u64_mulhigh_neon_(a, b) ? 18446744073709551615ull : (a * b);
}
NK_PUBLIC nk_i64_t nk_i64_saturating_mul_neon(nk_i64_t a, nk_i64_t b) {
    int sign = (a < 0) ^ (b < 0);
    nk_u64_t abs_a = a < 0 ? -(nk_u64_t)a : (nk_u64_t)a;
    nk_u64_t abs_b = b < 0 ? -(nk_u64_t)b : (nk_u64_t)b;
    nk_u64_t high = nk_u64_mulhigh_neon_(abs_a, abs_b);
    nk_u64_t low = abs_a * abs_b;
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

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_SCALAR_NEON_H
