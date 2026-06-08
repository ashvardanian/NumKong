/**
 *  @brief SIMD-accelerated Type Conversions and Load/Store Helpers for LoongArch LASX (256-bit).
 *  @file include/numkong/cast/loongsonasx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/cast.h
 *
 *  @section loongsonasx_cast_instructions Key LASX Load/Store Instructions
 *
 *      Intrinsic                      Instruction       Description
 *      __lasx_xvld(ptr, 0)            XVLD              256-bit aligned/unaligned load
 *      __lasx_xvst(v, ptr, 0)         XVST              256-bit aligned/unaligned store
 *      __lasx_xvreplgr2vr_w(bits)     XVREPLGR2VR.W     Broadcast i32 to 8 lanes
 *      __lasx_xvreplgr2vr_d(bits)     XVREPLGR2VR.D     Broadcast i64 to 4 lanes
 *      __lasx_xvffint_s_w(v)          XVFFINT.S.W       4x i32 -> f32 (per 128-bit lane)
 *      __lasx_xvfrsqrt_s(v)           XVFRSQRT.S        f32 full-precision reciprocal sqrt
 *      __lasx_xvfsqrt_s(v)            XVFSQRT.S         f32 full-precision sqrt
 *      __lasx_xvfsqrt_d(v)            XVFSQRT.D         f64 full-precision sqrt
 *
 *  LASX is a 256-bit extension; all vector registers are 256-bit `__m256i`. For 128-bit
 *  `nk_b128_vec_t` operations, `__lasx_xvld` safely loads into the low 128 bits (the high
 *  128 bits are zeroed or undefined depending on context). For 128-bit stores we use `memcpy`
 *  to avoid writing beyond the intended 16 bytes. Partial loads/stores delegate to serial
 *  helpers since LASX lacks masked load/store instructions.
 */
#ifndef NK_CAST_LOONGSONASX_H
#define NK_CAST_LOONGSONASX_H

#if NK_TARGET_LOONGARCH64_
#if NK_TARGET_LOONGSONASX

#include "numkong/types.h"
#include "numkong/cast/serial.h"        // `nk_partial_load_b32x4_serial_`, `nk_partial_load_b64x4_serial_`
#include "numkong/scalar/loongsonasx.h" // `nk_xvreplgr2vr_s_128_`, `nk_xvfreplgr2vr_s_`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Type Punned Loads and Stores

/**
 *  LSX and LASX share the same physical register file, so widening __m128i → __m256i and
 *  extracting __m256i → __m128i are no-ops on hardware. Empty inline asm with "f" constraints
 *  avoids the stack round-trip that union punning causes on GCC 14.
 *  Named after x86 `_mm256_castsi128_si256` / `_mm256_castsi256_si128` / `_mm256_castps256_ps128`.
 */
NK_INTERNAL __m256i nk_lasx_castsi128_si256_(__m128i low_i64x2) {
    __m256i wide_i64x4;
    __asm__("" : "=f"(wide_i64x4) : "f"(low_i64x2));
    return wide_i64x4;
}
NK_INTERNAL __m128i nk_lasx_castsi256_si128_(__m256i wide_i64x4) {
    __m128i low_i64x2;
    __asm__("" : "=f"(low_i64x2) : "f"(wide_i64x4));
    return low_i64x2;
}
NK_INTERNAL __m128 nk_lasx_castps256_ps128_(__m256 wide_f32x8) {
    __m128 low_f32x4;
    __asm__("" : "=f"(low_f32x4) : "f"(wide_f32x8));
    return low_f32x4;
}

/** @brief Type-agnostic 256-bit full load (LASX). */
NK_INTERNAL void nk_load_b256_loongsonasx_(void const *src, nk_b256_vec_t *dst) { dst->ymm = __lasx_xvld(src, 0); }

/** @brief Type-agnostic 256-bit full store (LASX). */
NK_INTERNAL void nk_store_b256_loongsonasx_(nk_b256_vec_t const *src, void *dst) { __lasx_xvst(src->ymm, dst, 0); }

/** @brief Type-agnostic 128-bit full load (LSX subset of LASX). */
NK_INTERNAL void nk_load_b128_loongsonasx_(void const *src, nk_b128_vec_t *dst) { dst->xmm = __lsx_vld(src, 0); }

/** @brief Type-agnostic 128-bit full store (LSX subset of LASX). */
NK_INTERNAL void nk_store_b128_loongsonasx_(nk_b128_vec_t const *src, void *dst) { __lsx_vst(src->xmm, dst, 0); }

/** @brief Convert 8 × bf16 → 8 × f32 by interleaving with zero so bf16 lands in upper 16 bits (LASX).
 *
 *  Duplicates the 128-bit input into both lanes, then `xvilvl_h(bf16, zero)` places each bf16
 *  value in the high 16 bits of a 32-bit slot — which is valid f32 with no shift needed.
 *  `xvpermi_q` combines the low-element and high-element halves into a single register.
 */
NK_INTERNAL __m256i nk_bf16x8_to_f32x8_loongsonasx_(__m128i bf16_i16x8) {
    __m256i duped_bf16x16 = __lasx_xvpermi_q(nk_lasx_castsi128_si256_(bf16_i16x8), nk_lasx_castsi128_si256_(bf16_i16x8),
                                             0x00);
    __m256i zero_i16x16 = __lasx_xvreplgr2vr_h(0);
    __m256i low_f32x8 = __lasx_xvilvl_h(duped_bf16x16, zero_i16x16);
    __m256i high_f32x8 = __lasx_xvilvh_h(duped_bf16x16, zero_i16x16);
    return __lasx_xvpermi_q(high_f32x8, low_f32x8, 0x20);
}

/** @brief Load 8 × bf16 from memory, convert to 8 × f32, store in 256-bit vector (LASX). */
NK_INTERNAL void nk_load_bf16x8_to_f32x8_loongsonasx_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm = nk_bf16x8_to_f32x8_loongsonasx_(__lsx_vld(src, 0));
}

/** @brief Partial load for bf16 elements (up to 8) with conversion to f32 (LASX). */
NK_INTERNAL void nk_partial_load_bf16x8_to_f32x8_loongsonasx_(nk_bf16_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t vec;
    nk_partial_load_b16x8_serial_(src, &vec, n);
    dst->ymm = nk_bf16x8_to_f32x8_loongsonasx_(vec.xmm);
}

/** @brief Convert 8 × f16 → 8 × f32 via native LASX hardware conversion. */
NK_INTERNAL __m256i nk_f16x8_to_f32x8_loongsonasx_(__m128i f16_i16x8) {
    __m256i duped_f16x16 = __lasx_xvpermi_q(nk_lasx_castsi128_si256_(f16_i16x8), nk_lasx_castsi128_si256_(f16_i16x8),
                                            0x00);
    __m256i low_f32x8 = (__m256i)__lasx_xvfcvtl_s_h(duped_f16x16);
    __m256i high_f32x8 = (__m256i)__lasx_xvfcvth_s_h(duped_f16x16);
    return __lasx_xvpermi_q(high_f32x8, low_f32x8, 0x20);
}

/** @brief Load 8 × f16 from memory, convert to 8 × f32 via native LASX conversion. */
NK_INTERNAL void nk_load_f16x8_to_f32x8_loongsonasx_(void const *src, nk_b256_vec_t *dst) {
    dst->ymm = nk_f16x8_to_f32x8_loongsonasx_(__lsx_vld(src, 0));
}

/** @brief Partial load for f16 elements (up to 8) with conversion to f32 (LASX). */
NK_INTERNAL void nk_partial_load_f16x8_to_f32x8_loongsonasx_(nk_f16_t const *src, nk_b256_vec_t *dst, nk_size_t n) {
    nk_b128_vec_t vec;
    nk_partial_load_b16x8_serial_(src, &vec, n);
    dst->ymm = nk_f16x8_to_f32x8_loongsonasx_(vec.xmm);
}

#pragma endregion Type Punned Loads and Stores

#pragma region Vectorized From Dot Helpers

/** @brief Safe square root of 8 floats with zero-clamping for numerical stability (LASX 256-bit). */
NK_INTERNAL __m256 nk_sqrt_f32x8_loongsonasx_(__m256 x_f32x8) {
    __m256 zero_f32x8 = (__m256)__lasx_xvreplgr2vr_w(0);
    return __lasx_xvfsqrt_s(__lasx_xvfmax_s(x_f32x8, zero_f32x8));
}

/** @brief Safe square root of 4 floats with zero-clamping for numerical stability (LSX 128-bit). */
NK_INTERNAL __m128 nk_sqrt_f32x4_loongsonasx_(__m128 x_f32x4) {
    __m128 zero_f32x4 = (__m128)__lsx_vreplgr2vr_w(0);
    return __lsx_vfsqrt_s(__lsx_vfmax_s(x_f32x4, zero_f32x4));
}

/** @brief Angular from_dot: computes 1 − dot × rsqrt(query_sumsq × target_sumsq) for 4 pairs (LSX 128-bit f32). */
NK_INTERNAL void nk_angular_through_f32_from_dot_loongsonasx_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    __m128 dots_f32x4 = dots_vec->xmm_ps;
    __m128 query_sumsq_f32x4 = nk_xvreplgr2vr_s_128_(query_sumsq);
    __m128 products_f32x4 = __lsx_vfmul_s(query_sumsq_f32x4, target_sumsqs_vec->xmm_ps);
    __m128 rsqrt_f32x4 = __lsx_vfrsqrt_s(products_f32x4);
    __m128 normalized_f32x4 = __lsx_vfmul_s(dots_f32x4, rsqrt_f32x4);
    __m128 one_f32x4 = nk_xvreplgr2vr_s_128_(1.0f);
    __m128 angular_f32x4 = __lsx_vfsub_s(one_f32x4, normalized_f32x4);
    __m128 zero_f32x4 = (__m128)__lsx_vreplgr2vr_w(0);
    result_vec->xmm_ps = __lsx_vfmax_s(angular_f32x4, zero_f32x4);
}

/** @brief Euclidean from_dot: computes √(query_sumsq + target_sumsq − 2 × dot) for 4 pairs (LSX 128-bit f32). */
NK_INTERNAL void nk_euclidean_through_f32_from_dot_loongsonasx_(nk_b128_vec_t const *dots_vec, nk_f32_t query_sumsq,
                                                                nk_b128_vec_t const *target_sumsqs_vec,
                                                                nk_b128_vec_t *result_vec) {
    __m128 dots_f32x4 = dots_vec->xmm_ps;
    __m128 query_sumsq_f32x4 = nk_xvreplgr2vr_s_128_(query_sumsq);
    __m128 sum_sq_f32x4 = __lsx_vfadd_s(query_sumsq_f32x4, target_sumsqs_vec->xmm_ps);
    __m128 two_f32x4 = nk_xvreplgr2vr_s_128_(2.0f);
    // dist_sq = sum_sq − 2 × dots = -(2 × dots − sum_sq)
    __m128 dist_sq_f32x4 = __lsx_vfnmsub_s(two_f32x4, dots_f32x4, sum_sq_f32x4);
    result_vec->xmm_ps = nk_sqrt_f32x4_loongsonasx_(dist_sq_f32x4);
}

/** @brief Angular from_dot for native f64: 1 − dot / √(query_sumsq × target_sumsq) for 4 pairs (LASX 256-bit). */
NK_INTERNAL void nk_angular_through_f64_from_dot_loongsonasx_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                              nk_b256_vec_t const *target_sumsqs_vec,
                                                              nk_b256_vec_t *result_vec) {
    __m256d dots_f64x4 = dots_vec->ymm_pd;
    __m256d query_sumsq_f64x4 = nk_xvfreplgr2vr_d_(query_sumsq);
    __m256d products_f64x4 = __lasx_xvfmul_d(query_sumsq_f64x4, target_sumsqs_vec->ymm_pd);
    __m256d sqrt_products_f64x4 = __lasx_xvfsqrt_d(products_f64x4);
    __m256d normalized_f64x4 = __lasx_xvfdiv_d(dots_f64x4, sqrt_products_f64x4);
    __m256d one_f64x4 = nk_xvfreplgr2vr_d_(1.0);
    __m256d angular_f64x4 = __lasx_xvfsub_d(one_f64x4, normalized_f64x4);
    __m256d zero_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    result_vec->ymm_pd = __lasx_xvfmax_d(angular_f64x4, zero_f64x4);
}

/** @brief Euclidean from_dot for native f64: √(query_sumsq + target_sumsq − 2 × dot) for 4 pairs (LASX 256-bit). */
NK_INTERNAL void nk_euclidean_through_f64_from_dot_loongsonasx_(nk_b256_vec_t const *dots_vec, nk_f64_t query_sumsq,
                                                                nk_b256_vec_t const *target_sumsqs_vec,
                                                                nk_b256_vec_t *result_vec) {
    __m256d dots_f64x4 = dots_vec->ymm_pd;
    __m256d query_sumsq_f64x4 = nk_xvfreplgr2vr_d_(query_sumsq);
    __m256d sum_sq_f64x4 = __lasx_xvfadd_d(query_sumsq_f64x4, target_sumsqs_vec->ymm_pd);
    __m256d two_f64x4 = nk_xvfreplgr2vr_d_(2.0);
    // dist_sq = sum_sq − 2 × dots = -(2 × dots − sum_sq)
    __m256d dist_sq_f64x4 = __lasx_xvfnmsub_d(two_f64x4, dots_f64x4, sum_sq_f64x4);
    __m256d zero_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    result_vec->ymm_pd = __lasx_xvfsqrt_d(__lasx_xvfmax_d(dist_sq_f64x4, zero_f64x4));
}

/** @brief Angular from_dot for i32 accumulators: cast i32 → f32, rsqrt+NR, clamp. 4 pairs (LSX 128-bit). */
NK_INTERNAL void nk_angular_through_i32_from_dot_loongsonasx_(nk_b128_vec_t const *dots_vec, nk_i32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    __m128 dots_f32x4 = __lsx_vffint_s_w(dots_vec->xmm);
    __m128 query_sumsq_f32x4 = nk_xvreplgr2vr_s_128_((nk_f32_t)query_sumsq);
    __m128 products_f32x4 = __lsx_vfmul_s(query_sumsq_f32x4, __lsx_vffint_s_w(target_sumsqs_vec->xmm));
    __m128 rsqrt_f32x4 = __lsx_vfrsqrt_s(products_f32x4);
    __m128 normalized_f32x4 = __lsx_vfmul_s(dots_f32x4, rsqrt_f32x4);
    __m128 one_f32x4 = nk_xvreplgr2vr_s_128_(1.0f);
    __m128 angular_f32x4 = __lsx_vfsub_s(one_f32x4, normalized_f32x4);
    __m128 zero_f32x4 = (__m128)__lsx_vreplgr2vr_w(0);
    result_vec->xmm_ps = __lsx_vfmax_s(angular_f32x4, zero_f32x4);
}

/** @brief Euclidean from_dot for i32 accumulators: cast i32 → f32, then √(a² + b² − 2ab). 4 pairs (LSX 128-bit). */
NK_INTERNAL void nk_euclidean_through_i32_from_dot_loongsonasx_(nk_b128_vec_t const *dots_vec, nk_i32_t query_sumsq,
                                                                nk_b128_vec_t const *target_sumsqs_vec,
                                                                nk_b128_vec_t *result_vec) {
    __m128 dots_f32x4 = __lsx_vffint_s_w(dots_vec->xmm);
    __m128 query_sumsq_f32x4 = nk_xvreplgr2vr_s_128_((nk_f32_t)query_sumsq);
    __m128 sum_sq_f32x4 = __lsx_vfadd_s(query_sumsq_f32x4, __lsx_vffint_s_w(target_sumsqs_vec->xmm));
    __m128 two_f32x4 = nk_xvreplgr2vr_s_128_(2.0f);
    __m128 dist_sq_f32x4 = __lsx_vfnmsub_s(two_f32x4, dots_f32x4, sum_sq_f32x4);
    result_vec->xmm_ps = nk_sqrt_f32x4_loongsonasx_(dist_sq_f32x4);
}

/** @brief Angular from_dot for u32 accumulators: cast u32 → f32, rsqrt+NR, clamp. 4 pairs (LSX 128-bit). */
NK_INTERNAL void nk_angular_through_u32_from_dot_loongsonasx_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                              nk_b128_vec_t const *target_sumsqs_vec,
                                                              nk_b128_vec_t *result_vec) {
    __m128 dots_f32x4 = __lsx_vffint_s_w(dots_vec->xmm);
    __m128 query_sumsq_f32x4 = nk_xvreplgr2vr_s_128_((nk_f32_t)query_sumsq);
    __m128 products_f32x4 = __lsx_vfmul_s(query_sumsq_f32x4, __lsx_vffint_s_w(target_sumsqs_vec->xmm));
    __m128 rsqrt_f32x4 = __lsx_vfrsqrt_s(products_f32x4);
    __m128 normalized_f32x4 = __lsx_vfmul_s(dots_f32x4, rsqrt_f32x4);
    __m128 one_f32x4 = nk_xvreplgr2vr_s_128_(1.0f);
    __m128 angular_f32x4 = __lsx_vfsub_s(one_f32x4, normalized_f32x4);
    __m128 zero_f32x4 = (__m128)__lsx_vreplgr2vr_w(0);
    result_vec->xmm_ps = __lsx_vfmax_s(angular_f32x4, zero_f32x4);
}

/** @brief Euclidean from_dot for u32 accumulators: cast u32 → f32, then √(a² + b² − 2ab). 4 pairs (LSX 128-bit). */
NK_INTERNAL void nk_euclidean_through_u32_from_dot_loongsonasx_(nk_b128_vec_t const *dots_vec, nk_u32_t query_sumsq,
                                                                nk_b128_vec_t const *target_sumsqs_vec,
                                                                nk_b128_vec_t *result_vec) {
    __m128 dots_f32x4 = __lsx_vffint_s_w(dots_vec->xmm);
    __m128 query_sumsq_f32x4 = nk_xvreplgr2vr_s_128_((nk_f32_t)query_sumsq);
    __m128 sum_sq_f32x4 = __lsx_vfadd_s(query_sumsq_f32x4, __lsx_vffint_s_w(target_sumsqs_vec->xmm));
    __m128 two_f32x4 = nk_xvreplgr2vr_s_128_(2.0f);
    __m128 dist_sq_f32x4 = __lsx_vfnmsub_s(two_f32x4, dots_f32x4, sum_sq_f32x4);
    result_vec->xmm_ps = nk_sqrt_f32x4_loongsonasx_(dist_sq_f32x4);
}

#pragma endregion Vectorized From Dot Helpers

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_LOONGSONASX
#endif // NK_TARGET_LOONGARCH64_
#endif // NK_CAST_LOONGSONASX_H
