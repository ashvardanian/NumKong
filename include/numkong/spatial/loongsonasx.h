/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for LoongArch LASX (256-bit).
 *  @file include/numkong/spatial/loongsonasx.h
 *  @author Ash Vardanian
 *  @date March 23, 2026
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_loongsonasx_instructions Key LASX Spatial Instructions
 *
 *  LASX provides 256-bit SIMD operations using __m256i as the universal vector type.
 *  All intrinsics are prefixed with __lasx_. Float operations reinterpret __m256i as
 *  f32x8 or f64x4. Integer widening multiply-accumulate chains handle i8/u8 distances.
 *
 *  For F32 spatial distances, upcasting to F64 and downcasting back is faster than stable
 *  summation algorithms. For F64 angular we use the Dot2 algorithm (Ogita-Rump-Oishi, 2005)
 *  for the cross-product accumulation, while self-products use simple FMA since all terms
 *  are non-negative and don't suffer from cancellation.
 */
#ifndef NK_SPATIAL_LOONGSONASX_H
#define NK_SPATIAL_LOONGSONASX_H

#if NK_TARGET_LOONGARCH64_
#if NK_TARGET_LOONGSONASX

#include "numkong/types.h"
#include "numkong/spatial/serial.h"
#include "numkong/dot/loongsonasx.h"    //
#include "numkong/cast/loongsonasx.h"   // `nk_bf16x8_to_f32x8_loongsonasx_`
#include "numkong/scalar/loongsonasx.h" // `nk_f32_sqrt_loongsonasx`, `nk_f64_sqrt_loongsonasx`

#if defined(__cplusplus)
extern "C" {
#endif

#pragma region Angular Normalize Helpers

NK_HELPER_INLINE nk_f64_t nk_angular_normalize_f64_loongsonasx_(nk_f64_t ab, nk_f64_t a2, nk_f64_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    else if (ab == 0) return 1;
    nk_f64_t result = 1 - ab / (nk_f64_sqrt_loongsonasx(a2) * nk_f64_sqrt_loongsonasx(b2));
    return result > 0 ? result : 0;
}

NK_HELPER_INLINE nk_f32_t nk_angular_normalize_i32_loongsonasx_(nk_i32_t ab, nk_i32_t a2, nk_i32_t b2) {
    if (a2 == 0 && b2 == 0) return 0;
    else if (ab == 0) return 1;
    nk_f32_t result = 1.0f -
                      (nk_f32_t)ab * nk_f32_rsqrt_loongsonasx((nk_f32_t)a2) * nk_f32_rsqrt_loongsonasx((nk_f32_t)b2);
    return result > 0 ? result : 0;
}

#pragma endregion Angular Normalize Helpers

#pragma region I8 and U8 Integers

NK_API_COMPTIME void nk_sqeuclidean_i8_loongsonasx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {
    __m256i sum_i32x8 = __lasx_xvreplgr2vr_w(0);
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = __lasx_xvld(a + i, 0);
        __m256i b_i8x32 = __lasx_xvld(b + i, 0);
        __m256i diff_i8x32 = __lasx_xvsub_b(a_i8x32, b_i8x32);
        __m256i sq_i16x16 = __lasx_xvreplgr2vr_h(0);
        sq_i16x16 = __lasx_xvmaddwev_h_b(sq_i16x16, diff_i8x32, diff_i8x32);
        sq_i16x16 = __lasx_xvmaddwod_h_b(sq_i16x16, diff_i8x32, diff_i8x32);
        sum_i32x8 = __lasx_xvadd_w(sum_i32x8, __lasx_xvhaddw_w_h(sq_i16x16, sq_i16x16));
    }
    nk_i32_t sum = nk_reduce_add_i32x8_loongsonasx_(sum_i32x8);
    for (; i < n; ++i) {
        nk_i32_t diff = (nk_i32_t)a[i] - b[i];
        sum += diff * diff;
    }
    *result = (nk_u32_t)sum;
}

NK_API_COMPTIME void nk_euclidean_i8_loongsonasx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_sqeuclidean_i8_loongsonasx(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_loongsonasx((nk_f32_t)distance_sq_u32);
}

NK_API_COMPTIME void nk_angular_i8_loongsonasx(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256i dot_i32x8 = __lasx_xvreplgr2vr_w(0);
    __m256i a_sq_i32x8 = __lasx_xvreplgr2vr_w(0);
    __m256i b_sq_i32x8 = __lasx_xvreplgr2vr_w(0);
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_i8x32 = __lasx_xvld(a + i, 0);
        __m256i b_i8x32 = __lasx_xvld(b + i, 0);
        // dot(a, b)
        __m256i ab_i16x16 = __lasx_xvreplgr2vr_h(0);
        ab_i16x16 = __lasx_xvmaddwev_h_b(ab_i16x16, a_i8x32, b_i8x32);
        ab_i16x16 = __lasx_xvmaddwod_h_b(ab_i16x16, a_i8x32, b_i8x32);
        dot_i32x8 = __lasx_xvadd_w(dot_i32x8, __lasx_xvhaddw_w_h(ab_i16x16, ab_i16x16));
        // norm_sq(a)
        __m256i aa_i16x16 = __lasx_xvreplgr2vr_h(0);
        aa_i16x16 = __lasx_xvmaddwev_h_b(aa_i16x16, a_i8x32, a_i8x32);
        aa_i16x16 = __lasx_xvmaddwod_h_b(aa_i16x16, a_i8x32, a_i8x32);
        a_sq_i32x8 = __lasx_xvadd_w(a_sq_i32x8, __lasx_xvhaddw_w_h(aa_i16x16, aa_i16x16));
        // norm_sq(b)
        __m256i bb_i16x16 = __lasx_xvreplgr2vr_h(0);
        bb_i16x16 = __lasx_xvmaddwev_h_b(bb_i16x16, b_i8x32, b_i8x32);
        bb_i16x16 = __lasx_xvmaddwod_h_b(bb_i16x16, b_i8x32, b_i8x32);
        b_sq_i32x8 = __lasx_xvadd_w(b_sq_i32x8, __lasx_xvhaddw_w_h(bb_i16x16, bb_i16x16));
    }
    nk_i32_t dot = nk_reduce_add_i32x8_loongsonasx_(dot_i32x8);
    nk_i32_t a_sq = nk_reduce_add_i32x8_loongsonasx_(a_sq_i32x8);
    nk_i32_t b_sq = nk_reduce_add_i32x8_loongsonasx_(b_sq_i32x8);
    for (; i < n; ++i) {
        nk_i32_t a_val = a[i], b_val = b[i];
        dot += a_val * b_val;
        a_sq += a_val * a_val;
        b_sq += b_val * b_val;
    }
    *result = nk_angular_normalize_i32_loongsonasx_(dot, a_sq, b_sq);
}

NK_API_COMPTIME void nk_sqeuclidean_u8_loongsonasx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    __m256i sum_i32x8 = __lasx_xvreplgr2vr_w(0);
    __m256i zeros_i8x32 = __lasx_xvreplgr2vr_b(0);
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = __lasx_xvld(a + i, 0);
        __m256i b_u8x32 = __lasx_xvld(b + i, 0);
        __m256i a_low_u16x16 = __lasx_xvilvl_b(zeros_i8x32, a_u8x32);
        __m256i a_high_u16x16 = __lasx_xvilvh_b(zeros_i8x32, a_u8x32);
        __m256i b_low_u16x16 = __lasx_xvilvl_b(zeros_i8x32, b_u8x32);
        __m256i b_high_u16x16 = __lasx_xvilvh_b(zeros_i8x32, b_u8x32);
        __m256i diff_low_i16x16 = __lasx_xvsub_h(a_low_u16x16, b_low_u16x16);
        __m256i diff_high_i16x16 = __lasx_xvsub_h(a_high_u16x16, b_high_u16x16);
        __m256i sq_ev_low_i32x8 = __lasx_xvmulwev_w_h(diff_low_i16x16, diff_low_i16x16);
        __m256i sq_od_low_i32x8 = __lasx_xvmulwod_w_h(diff_low_i16x16, diff_low_i16x16);
        __m256i sq_ev_high_i32x8 = __lasx_xvmulwev_w_h(diff_high_i16x16, diff_high_i16x16);
        __m256i sq_od_high_i32x8 = __lasx_xvmulwod_w_h(diff_high_i16x16, diff_high_i16x16);
        sum_i32x8 = __lasx_xvadd_w(sum_i32x8, __lasx_xvadd_w(sq_ev_low_i32x8, sq_od_low_i32x8));
        sum_i32x8 = __lasx_xvadd_w(sum_i32x8, __lasx_xvadd_w(sq_ev_high_i32x8, sq_od_high_i32x8));
    }
    nk_i32_t sum = nk_reduce_add_i32x8_loongsonasx_(sum_i32x8);
    for (; i < n; ++i) {
        nk_i32_t diff = (nk_i32_t)a[i] - b[i];
        sum += diff * diff;
    }
    *result = (nk_u32_t)sum;
}

NK_API_COMPTIME void nk_euclidean_u8_loongsonasx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_sqeuclidean_u8_loongsonasx(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_loongsonasx((nk_f32_t)distance_sq_u32);
}

NK_API_COMPTIME void nk_angular_u8_loongsonasx(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256i dot_i32x8 = __lasx_xvreplgr2vr_w(0);
    __m256i a_sq_i32x8 = __lasx_xvreplgr2vr_w(0);
    __m256i b_sq_i32x8 = __lasx_xvreplgr2vr_w(0);
    __m256i zeros_i8x32 = __lasx_xvreplgr2vr_b(0);
    nk_size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i a_u8x32 = __lasx_xvld(a + i, 0);
        __m256i b_u8x32 = __lasx_xvld(b + i, 0);
        __m256i a_low_u16x16 = __lasx_xvilvl_b(zeros_i8x32, a_u8x32);
        __m256i a_high_u16x16 = __lasx_xvilvh_b(zeros_i8x32, a_u8x32);
        __m256i b_low_u16x16 = __lasx_xvilvl_b(zeros_i8x32, b_u8x32);
        __m256i b_high_u16x16 = __lasx_xvilvh_b(zeros_i8x32, b_u8x32);
        // dot(a, b)
        __m256i ab_ev_low_i32x8 = __lasx_xvmulwev_w_h(a_low_u16x16, b_low_u16x16);
        __m256i ab_od_low_i32x8 = __lasx_xvmulwod_w_h(a_low_u16x16, b_low_u16x16);
        __m256i ab_ev_high_i32x8 = __lasx_xvmulwev_w_h(a_high_u16x16, b_high_u16x16);
        __m256i ab_od_high_i32x8 = __lasx_xvmulwod_w_h(a_high_u16x16, b_high_u16x16);
        dot_i32x8 = __lasx_xvadd_w(dot_i32x8, __lasx_xvadd_w(ab_ev_low_i32x8, ab_od_low_i32x8));
        dot_i32x8 = __lasx_xvadd_w(dot_i32x8, __lasx_xvadd_w(ab_ev_high_i32x8, ab_od_high_i32x8));
        // norm_sq(a)
        __m256i aa_ev_low_i32x8 = __lasx_xvmulwev_w_h(a_low_u16x16, a_low_u16x16);
        __m256i aa_od_low_i32x8 = __lasx_xvmulwod_w_h(a_low_u16x16, a_low_u16x16);
        __m256i aa_ev_high_i32x8 = __lasx_xvmulwev_w_h(a_high_u16x16, a_high_u16x16);
        __m256i aa_od_high_i32x8 = __lasx_xvmulwod_w_h(a_high_u16x16, a_high_u16x16);
        a_sq_i32x8 = __lasx_xvadd_w(a_sq_i32x8, __lasx_xvadd_w(aa_ev_low_i32x8, aa_od_low_i32x8));
        a_sq_i32x8 = __lasx_xvadd_w(a_sq_i32x8, __lasx_xvadd_w(aa_ev_high_i32x8, aa_od_high_i32x8));
        // norm_sq(b)
        __m256i bb_ev_low_i32x8 = __lasx_xvmulwev_w_h(b_low_u16x16, b_low_u16x16);
        __m256i bb_od_low_i32x8 = __lasx_xvmulwod_w_h(b_low_u16x16, b_low_u16x16);
        __m256i bb_ev_high_i32x8 = __lasx_xvmulwev_w_h(b_high_u16x16, b_high_u16x16);
        __m256i bb_od_high_i32x8 = __lasx_xvmulwod_w_h(b_high_u16x16, b_high_u16x16);
        b_sq_i32x8 = __lasx_xvadd_w(b_sq_i32x8, __lasx_xvadd_w(bb_ev_low_i32x8, bb_od_low_i32x8));
        b_sq_i32x8 = __lasx_xvadd_w(b_sq_i32x8, __lasx_xvadd_w(bb_ev_high_i32x8, bb_od_high_i32x8));
    }
    nk_i32_t dot = nk_reduce_add_i32x8_loongsonasx_(dot_i32x8);
    nk_i32_t a_sq = nk_reduce_add_i32x8_loongsonasx_(a_sq_i32x8);
    nk_i32_t b_sq = nk_reduce_add_i32x8_loongsonasx_(b_sq_i32x8);
    for (; i < n; ++i) {
        nk_i32_t a_val = a[i], b_val = b[i];
        dot += a_val * b_val;
        a_sq += a_val * a_val;
        b_sq += b_val * b_val;
    }
    *result = nk_angular_normalize_i32_loongsonasx_(dot, a_sq, b_sq);
}

#pragma endregion I8 and U8 Integers

#pragma region F32 and F64 Floats

NK_API_COMPTIME void nk_sqeuclidean_f32_loongsonasx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,
                                                    nk_f64_t *result) {
    __m256d sum_low_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    __m256d sum_high_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i a_f32x8 = __lasx_xvld(a + i, 0);
        __m256i b_f32x8 = __lasx_xvld(b + i, 0);
        __m256d a_low_f64x4 = __lasx_xvfcvtl_d_s((__m256)a_f32x8);
        __m256d b_low_f64x4 = __lasx_xvfcvtl_d_s((__m256)b_f32x8);
        __m256d a_high_f64x4 = __lasx_xvfcvth_d_s((__m256)a_f32x8);
        __m256d b_high_f64x4 = __lasx_xvfcvth_d_s((__m256)b_f32x8);
        __m256d diff_low_f64x4 = __lasx_xvfsub_d(a_low_f64x4, b_low_f64x4);
        __m256d diff_high_f64x4 = __lasx_xvfsub_d(a_high_f64x4, b_high_f64x4);
        sum_low_f64x4 = __lasx_xvfmadd_d(diff_low_f64x4, diff_low_f64x4, sum_low_f64x4);
        sum_high_f64x4 = __lasx_xvfmadd_d(diff_high_f64x4, diff_high_f64x4, sum_high_f64x4);
    }
    __m256d combined_f64x4 = __lasx_xvfadd_d(sum_low_f64x4, sum_high_f64x4);
    nk_f64_t sum = nk_reduce_add_f64x4_loongsonasx_(combined_f64x4);
    for (; i < n; ++i) {
        nk_f64_t diff = (nk_f64_t)a[i] - b[i];
        sum += diff * diff;
    }
    *result = sum;
}

NK_API_COMPTIME void nk_euclidean_f32_loongsonasx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f32_loongsonasx(a, b, n, result);
    *result = nk_f64_sqrt_loongsonasx(*result);
}

NK_API_COMPTIME void nk_angular_f32_loongsonasx(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t *result) {
    __m256d dot_low_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    __m256d dot_high_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    __m256d a_sq_low_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    __m256d a_sq_high_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    __m256d b_sq_low_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    __m256d b_sq_high_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i a_f32x8 = __lasx_xvld(a + i, 0);
        __m256i b_f32x8 = __lasx_xvld(b + i, 0);
        __m256d a_low_f64x4 = __lasx_xvfcvtl_d_s((__m256)a_f32x8);
        __m256d b_low_f64x4 = __lasx_xvfcvtl_d_s((__m256)b_f32x8);
        __m256d a_high_f64x4 = __lasx_xvfcvth_d_s((__m256)a_f32x8);
        __m256d b_high_f64x4 = __lasx_xvfcvth_d_s((__m256)b_f32x8);
        dot_low_f64x4 = __lasx_xvfmadd_d(a_low_f64x4, b_low_f64x4, dot_low_f64x4);
        dot_high_f64x4 = __lasx_xvfmadd_d(a_high_f64x4, b_high_f64x4, dot_high_f64x4);
        a_sq_low_f64x4 = __lasx_xvfmadd_d(a_low_f64x4, a_low_f64x4, a_sq_low_f64x4);
        a_sq_high_f64x4 = __lasx_xvfmadd_d(a_high_f64x4, a_high_f64x4, a_sq_high_f64x4);
        b_sq_low_f64x4 = __lasx_xvfmadd_d(b_low_f64x4, b_low_f64x4, b_sq_low_f64x4);
        b_sq_high_f64x4 = __lasx_xvfmadd_d(b_high_f64x4, b_high_f64x4, b_sq_high_f64x4);
    }
    nk_f64_t dot = nk_reduce_add_f64x4_loongsonasx_(__lasx_xvfadd_d(dot_low_f64x4, dot_high_f64x4));
    nk_f64_t a_sq = nk_reduce_add_f64x4_loongsonasx_(__lasx_xvfadd_d(a_sq_low_f64x4, a_sq_high_f64x4));
    nk_f64_t b_sq = nk_reduce_add_f64x4_loongsonasx_(__lasx_xvfadd_d(b_sq_low_f64x4, b_sq_high_f64x4));
    for (; i < n; ++i) {
        nk_f64_t a_val = a[i], b_val = b[i];
        dot += a_val * b_val;
        a_sq += a_val * a_val;
        b_sq += b_val * b_val;
    }
    *result = nk_angular_normalize_f64_loongsonasx_(dot, a_sq, b_sq);
}

NK_API_COMPTIME void nk_sqeuclidean_f64_loongsonasx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,
                                                    nk_f64_t *result) {
    __m256d sum_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = (__m256d)__lasx_xvld(a + i, 0);
        __m256d b_f64x4 = (__m256d)__lasx_xvld(b + i, 0);
        __m256d diff_f64x4 = __lasx_xvfsub_d(a_f64x4, b_f64x4);
        sum_f64x4 = __lasx_xvfmadd_d(diff_f64x4, diff_f64x4, sum_f64x4);
    }
    nk_f64_t sum = nk_reduce_add_f64x4_loongsonasx_(sum_f64x4);
    for (; i < n; ++i) {
        nk_f64_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    *result = sum;
}

NK_API_COMPTIME void nk_euclidean_f64_loongsonasx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    nk_sqeuclidean_f64_loongsonasx(a, b, n, result);
    *result = nk_f64_sqrt_loongsonasx(*result);
}

NK_API_COMPTIME void nk_angular_f64_loongsonasx(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *result) {
    __m256d dot_sum_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    __m256d dot_compensation_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    __m256d a_norm_sq_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    __m256d b_norm_sq_f64x4 = (__m256d)__lasx_xvreplgr2vr_d(0);
    nk_size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d a_f64x4 = (__m256d)__lasx_xvld(a + i, 0);
        __m256d b_f64x4 = (__m256d)__lasx_xvld(b + i, 0);

        __m256d product_f64x4 = __lasx_xvfmul_d(a_f64x4, b_f64x4);
        __m256d product_error_f64x4 = __lasx_xvfmsub_d(a_f64x4, b_f64x4, product_f64x4);

        __m256d tentative_sum_f64x4 = __lasx_xvfadd_d(dot_sum_f64x4, product_f64x4);
        __m256d virtual_addend_f64x4 = __lasx_xvfsub_d(tentative_sum_f64x4, dot_sum_f64x4);
        __m256d sum_error_f64x4 = __lasx_xvfadd_d(
            __lasx_xvfsub_d(dot_sum_f64x4, __lasx_xvfsub_d(tentative_sum_f64x4, virtual_addend_f64x4)),
            __lasx_xvfsub_d(product_f64x4, virtual_addend_f64x4));

        dot_sum_f64x4 = tentative_sum_f64x4;
        dot_compensation_f64x4 = __lasx_xvfadd_d(dot_compensation_f64x4,
                                                 __lasx_xvfadd_d(sum_error_f64x4, product_error_f64x4));

        a_norm_sq_f64x4 = __lasx_xvfmadd_d(a_f64x4, a_f64x4, a_norm_sq_f64x4);
        b_norm_sq_f64x4 = __lasx_xvfmadd_d(b_f64x4, b_f64x4, b_norm_sq_f64x4);
    }

    nk_f64_t dot = nk_dot_stable_sum_f64x4_loongsonasx_(dot_sum_f64x4, dot_compensation_f64x4);
    nk_f64_t a_sq = nk_reduce_add_f64x4_loongsonasx_(a_norm_sq_f64x4);
    nk_f64_t b_sq = nk_reduce_add_f64x4_loongsonasx_(b_norm_sq_f64x4);
    for (; i < n; ++i) {
        nk_f64_t a_val = a[i], b_val = b[i];
        dot += a_val * b_val;
        a_sq += a_val * a_val;
        b_sq += b_val * b_val;
    }
    *result = nk_angular_normalize_f64_loongsonasx_(dot, a_sq, b_sq);
}

#pragma endregion F32 and F64 Floats

#pragma region F16 and BF16 Floats

NK_HELPER_INLINE nk_f32_t nk_angular_normalize_f32_loongsonasx_(nk_f32_t ab, nk_f32_t a2, nk_f32_t b2) {
    if (a2 == 0.0f && b2 == 0.0f) return 0.0f;
    else if (ab == 0.0f) return 1.0f;
    nk_f32_t result = 1.0f - ab * nk_f32_rsqrt_loongsonasx(a2) * nk_f32_rsqrt_loongsonasx(b2);
    return result > 0.0f ? result : 0.0f;
}

/** @brief Horizontal sum of 8 × f32 lanes in a 256-bit LASX register. */
NK_HELPER_INLINE nk_f32_t nk_reduce_add_f32x8_loongsonasx_(__m256 sum_f32x8) {
    // Add high 128-bit lane to low 128-bit lane
    __m256 high_f32x4 = (__m256)__lasx_xvpermi_q((__m256i)sum_f32x8, (__m256i)sum_f32x8, 0x11);
    __m256 sum_f32x4 = __lasx_xvfadd_s(sum_f32x8, high_f32x4);
    __m256 swapped_f32x4 = (__m256)__lasx_xvshuf4i_w((__m256i)sum_f32x4, 0b01001110);
    __m256 reduced_f32x4 = __lasx_xvfadd_s(sum_f32x4, swapped_f32x4);
    __m256 swapped_f32x2 = (__m256)__lasx_xvshuf4i_w((__m256i)reduced_f32x4, 0b10110001);
    __m256 reduced_f32x2 = __lasx_xvfadd_s(reduced_f32x4, swapped_f32x2);
    nk_fui32_t c;
    c.u = (nk_u32_t)__lasx_xvpickve2gr_w((__m256i)reduced_f32x2, 0);
    return c.f;
}

NK_API_COMPTIME void nk_sqeuclidean_bf16_loongsonasx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                     nk_f32_t *result) {
    __m256 sum_f32x8 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256i mask_high_u32x8 = __lasx_xvreplgr2vr_w((int)0xFFFF0000);
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a_bf16x16 = __lasx_xvld(a + i, 0);
        __m256i b_bf16x16 = __lasx_xvld(b + i, 0);
        __m256 a_even_f32x8 = (__m256)__lasx_xvslli_w(a_bf16x16, 16);
        __m256 b_even_f32x8 = (__m256)__lasx_xvslli_w(b_bf16x16, 16);
        __m256 diff_even_f32x8 = __lasx_xvfsub_s(a_even_f32x8, b_even_f32x8);
        sum_f32x8 = __lasx_xvfmadd_s(diff_even_f32x8, diff_even_f32x8, sum_f32x8);
        __m256 a_odd_f32x8 = (__m256)__lasx_xvand_v(a_bf16x16, mask_high_u32x8);
        __m256 b_odd_f32x8 = (__m256)__lasx_xvand_v(b_bf16x16, mask_high_u32x8);
        __m256 diff_odd_f32x8 = __lasx_xvfsub_s(a_odd_f32x8, b_odd_f32x8);
        sum_f32x8 = __lasx_xvfmadd_s(diff_odd_f32x8, diff_odd_f32x8, sum_f32x8);
    }
    nk_f32_t sum = nk_reduce_add_f32x8_loongsonasx_(sum_f32x8);
    for (; i < n; ++i) {
        nk_f32_t a_val, b_val;
        nk_bf16_to_f32_serial(&a[i], &a_val);
        nk_bf16_to_f32_serial(&b[i], &b_val);
        nk_f32_t diff = a_val - b_val;
        sum += diff * diff;
    }
    *result = sum;
}

NK_API_COMPTIME void nk_euclidean_bf16_loongsonasx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                   nk_f32_t *result) {
    nk_sqeuclidean_bf16_loongsonasx(a, b, n, result);
    *result = nk_f32_sqrt_loongsonasx(*result);
}

NK_API_COMPTIME void nk_angular_bf16_loongsonasx(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                 nk_f32_t *result) {
    __m256 dot_f32x8 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 a_sq_f32x8 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 b_sq_f32x8 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256i mask_high_u32x8 = __lasx_xvreplgr2vr_w((int)0xFFFF0000);
    nk_size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a_bf16x16 = __lasx_xvld(a + i, 0);
        __m256i b_bf16x16 = __lasx_xvld(b + i, 0);
        __m256 a_even_f32x8 = (__m256)__lasx_xvslli_w(a_bf16x16, 16);
        __m256 b_even_f32x8 = (__m256)__lasx_xvslli_w(b_bf16x16, 16);
        dot_f32x8 = __lasx_xvfmadd_s(a_even_f32x8, b_even_f32x8, dot_f32x8);
        a_sq_f32x8 = __lasx_xvfmadd_s(a_even_f32x8, a_even_f32x8, a_sq_f32x8);
        b_sq_f32x8 = __lasx_xvfmadd_s(b_even_f32x8, b_even_f32x8, b_sq_f32x8);
        __m256 a_odd_f32x8 = (__m256)__lasx_xvand_v(a_bf16x16, mask_high_u32x8);
        __m256 b_odd_f32x8 = (__m256)__lasx_xvand_v(b_bf16x16, mask_high_u32x8);
        dot_f32x8 = __lasx_xvfmadd_s(a_odd_f32x8, b_odd_f32x8, dot_f32x8);
        a_sq_f32x8 = __lasx_xvfmadd_s(a_odd_f32x8, a_odd_f32x8, a_sq_f32x8);
        b_sq_f32x8 = __lasx_xvfmadd_s(b_odd_f32x8, b_odd_f32x8, b_sq_f32x8);
    }
    nk_f32_t dot = nk_reduce_add_f32x8_loongsonasx_(dot_f32x8);
    nk_f32_t a_sq = nk_reduce_add_f32x8_loongsonasx_(a_sq_f32x8);
    nk_f32_t b_sq = nk_reduce_add_f32x8_loongsonasx_(b_sq_f32x8);
    for (; i < n; ++i) {
        nk_f32_t a_val, b_val;
        nk_bf16_to_f32_serial(&a[i], &a_val);
        nk_bf16_to_f32_serial(&b[i], &b_val);
        dot += a_val * b_val;
        a_sq += a_val * a_val;
        b_sq += b_val * b_val;
    }
    *result = nk_angular_normalize_f32_loongsonasx_(dot, a_sq, b_sq);
}

NK_API_COMPTIME void nk_sqeuclidean_f16_loongsonasx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                    nk_f32_t *result) {
    __m256 sum_f32x8 = (__m256)__lasx_xvreplgr2vr_w(0);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16x8 = __lsx_vld(a + i, 0);
        __m128i b_f16x8 = __lsx_vld(b + i, 0);
        __m256 a_f32x8 = (__m256)nk_f16x8_to_f32x8_loongsonasx_(a_f16x8);
        __m256 b_f32x8 = (__m256)nk_f16x8_to_f32x8_loongsonasx_(b_f16x8);
        __m256 diff_f32x8 = __lasx_xvfsub_s(a_f32x8, b_f32x8);
        sum_f32x8 = __lasx_xvfmadd_s(diff_f32x8, diff_f32x8, sum_f32x8);
    }
    nk_f32_t sum = nk_reduce_add_f32x8_loongsonasx_(sum_f32x8);
    for (; i < n; ++i) {
        nk_f32_t a_val, b_val;
        nk_f16_to_f32_serial(&a[i], &a_val);
        nk_f16_to_f32_serial(&b[i], &b_val);
        nk_f32_t diff = a_val - b_val;
        sum += diff * diff;
    }
    *result = sum;
}

NK_API_COMPTIME void nk_euclidean_f16_loongsonasx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_sqeuclidean_f16_loongsonasx(a, b, n, result);
    *result = nk_f32_sqrt_loongsonasx(*result);
}

NK_API_COMPTIME void nk_angular_f16_loongsonasx(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *result) {
    __m256 dot_f32x8 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 a_sq_f32x8 = (__m256)__lasx_xvreplgr2vr_w(0);
    __m256 b_sq_f32x8 = (__m256)__lasx_xvreplgr2vr_w(0);
    nk_size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i a_f16x8 = __lsx_vld(a + i, 0);
        __m128i b_f16x8 = __lsx_vld(b + i, 0);
        __m256 a_f32x8 = (__m256)nk_f16x8_to_f32x8_loongsonasx_(a_f16x8);
        __m256 b_f32x8 = (__m256)nk_f16x8_to_f32x8_loongsonasx_(b_f16x8);
        dot_f32x8 = __lasx_xvfmadd_s(a_f32x8, b_f32x8, dot_f32x8);
        a_sq_f32x8 = __lasx_xvfmadd_s(a_f32x8, a_f32x8, a_sq_f32x8);
        b_sq_f32x8 = __lasx_xvfmadd_s(b_f32x8, b_f32x8, b_sq_f32x8);
    }
    nk_f32_t dot = nk_reduce_add_f32x8_loongsonasx_(dot_f32x8);
    nk_f32_t a_sq = nk_reduce_add_f32x8_loongsonasx_(a_sq_f32x8);
    nk_f32_t b_sq = nk_reduce_add_f32x8_loongsonasx_(b_sq_f32x8);
    for (; i < n; ++i) {
        nk_f32_t a_val, b_val;
        nk_f16_to_f32_serial(&a[i], &a_val);
        nk_f16_to_f32_serial(&b[i], &b_val);
        dot += a_val * b_val;
        a_sq += a_val * a_val;
        b_sq += b_val * b_val;
    }
    *result = nk_angular_normalize_f32_loongsonasx_(dot, a_sq, b_sq);
}

#pragma endregion F16 and BF16 Floats

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_LOONGSONASX
#endif // NK_TARGET_LOONGARCH64_
#endif // NK_SPATIAL_LOONGSONASX_H
