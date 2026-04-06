/**
 *  @brief SIMD-accelerated Spatial Similarity Measures for SVE SDOT.
 *  @file include/numkong/spatial/svesdot.h
 *  @author Ash Vardanian
 *  @date April 3, 2026
 *
 *  @sa include/numkong/spatial.h
 *
 *  @section spatial_svesdot_instructions ARM SVE+DotProd Instructions
 *
 *      Intrinsic      Instruction              V1
 *      svld1_s8       LD1B (Z.B, P/Z, [Xn])    4-6cy @ 2p
 *      svld1_u8       LD1B (Z.B, P/Z, [Xn])    4-6cy @ 2p
 *      svdot_s32      SDOT (Z.S, Z.B, Z.B)     3cy @ 2p
 *      svdot_u32      UDOT (Z.S, Z.B, Z.B)     3cy @ 2p
 *      svabd_s8_x     SABD (Z.B, P/M, Z.B)     3cy @ 2p
 *      svabd_u8_x     UABD (Z.B, P/M, Z.B)     3cy @ 2p
 *      svaddv_s32     SADDV (D, P, Z.S)        6cy @ 1p
 *      svaddv_u32     UADDV (D, P, Z.S)        6cy @ 1p
 *      svwhilelt_b8   WHILELT (P.B, Xn, Xm)    2cy @ 1p
 *      svcntb         CNTB (Xd)                1cy @ 2p
 *
 *  SVE vector widths vary across implementations: Graviton3 uses 256-bit, while Graviton4/5
 *  and Apple M4+ use 128-bit. Code using svcntb() adapts automatically, but wider vectors
 *  process more elements per iteration with identical latencies.
 *
 *  For L2 distance, SABD/UABD computes |a-b| per byte, then UDOT squares and accumulates.
 *  Angular distance uses SDOT/UDOT directly for dot product and norm computations.
 */
#ifndef NK_SPATIAL_SVESDOT_H
#define NK_SPATIAL_SVESDOT_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SVESDOT

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_angular_normalize_f32_neon_`, `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve+dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve+dotprod")
#endif

NK_PUBLIC void nk_sqeuclidean_i8_svesdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t i = 0;
    svuint32_t distance_sq_u32x = svdup_u32(0);
    do {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, n);
        svint8_t a_i8x = svld1_s8(predicate_b8x, a + i);
        svint8_t b_i8x = svld1_s8(predicate_b8x, b + i);
        svuint8_t diff_u8x = svreinterpret_u8_s8(svabd_s8_x(predicate_b8x, a_i8x, b_i8x));
        distance_sq_u32x = svdot_u32(distance_sq_u32x, diff_u8x, diff_u8x);
        i += svcntb();
    } while (i < n);
    *result = (nk_u32_t)svaddv_u32(svptrue_b32(), distance_sq_u32x);
    NK_UNPOISON(result, sizeof(*result));
}
NK_PUBLIC void nk_euclidean_i8_svesdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_sqeuclidean_i8_svesdot(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_neon((nk_f32_t)distance_sq_u32);
}

NK_PUBLIC void nk_angular_i8_svesdot(nk_i8_t const *a, nk_i8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    svint32_t ab_i32x = svdup_s32(0);
    svint32_t a2_i32x = svdup_s32(0);
    svint32_t b2_i32x = svdup_s32(0);
    do {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, n);
        svint8_t a_i8x = svld1_s8(predicate_b8x, a + i);
        svint8_t b_i8x = svld1_s8(predicate_b8x, b + i);
        ab_i32x = svdot_s32(ab_i32x, a_i8x, b_i8x);
        a2_i32x = svdot_s32(a2_i32x, a_i8x, a_i8x);
        b2_i32x = svdot_s32(b2_i32x, b_i8x, b_i8x);
        i += svcntb();
    } while (i < n);

    nk_i32_t ab = (nk_i32_t)svaddv_s32(svptrue_b32(), ab_i32x);
    nk_i32_t a2 = (nk_i32_t)svaddv_s32(svptrue_b32(), a2_i32x);
    nk_i32_t b2 = (nk_i32_t)svaddv_s32(svptrue_b32(), b2_i32x);
    NK_UNPOISON(&ab, sizeof(ab));
    NK_UNPOISON(&a2, sizeof(a2));
    NK_UNPOISON(&b2, sizeof(b2));
    *result = nk_angular_normalize_f32_neon_((nk_f32_t)ab, (nk_f32_t)a2, (nk_f32_t)b2);
}

NK_PUBLIC void nk_sqeuclidean_u8_svesdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_u32_t *result) {
    nk_size_t i = 0;
    svuint32_t distance_sq_u32x = svdup_u32(0);
    do {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, n);
        svuint8_t a_u8x = svld1_u8(predicate_b8x, a + i);
        svuint8_t b_u8x = svld1_u8(predicate_b8x, b + i);
        svuint8_t diff_u8x = svabd_u8_x(predicate_b8x, a_u8x, b_u8x);
        distance_sq_u32x = svdot_u32(distance_sq_u32x, diff_u8x, diff_u8x);
        i += svcntb();
    } while (i < n);
    *result = (nk_u32_t)svaddv_u32(svptrue_b32(), distance_sq_u32x);
    NK_UNPOISON(result, sizeof(*result));
}
NK_PUBLIC void nk_euclidean_u8_svesdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_u32_t distance_sq_u32;
    nk_sqeuclidean_u8_svesdot(a, b, n, &distance_sq_u32);
    *result = nk_f32_sqrt_neon((nk_f32_t)distance_sq_u32);
}

NK_PUBLIC void nk_angular_u8_svesdot(nk_u8_t const *a, nk_u8_t const *b, nk_size_t n, nk_f32_t *result) {
    nk_size_t i = 0;
    svuint32_t ab_u32x = svdup_u32(0);
    svuint32_t a2_u32x = svdup_u32(0);
    svuint32_t b2_u32x = svdup_u32(0);
    do {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, n);
        svuint8_t a_u8x = svld1_u8(predicate_b8x, a + i);
        svuint8_t b_u8x = svld1_u8(predicate_b8x, b + i);
        ab_u32x = svdot_u32(ab_u32x, a_u8x, b_u8x);
        a2_u32x = svdot_u32(a2_u32x, a_u8x, a_u8x);
        b2_u32x = svdot_u32(b2_u32x, b_u8x, b_u8x);
        i += svcntb();
    } while (i < n);

    nk_u32_t ab = (nk_u32_t)svaddv_u32(svptrue_b32(), ab_u32x);
    nk_u32_t a2 = (nk_u32_t)svaddv_u32(svptrue_b32(), a2_u32x);
    nk_u32_t b2 = (nk_u32_t)svaddv_u32(svptrue_b32(), b2_u32x);
    NK_UNPOISON(&ab, sizeof(ab));
    NK_UNPOISON(&a2, sizeof(a2));
    NK_UNPOISON(&b2, sizeof(b2));
    *result = nk_angular_normalize_f32_neon_((nk_f32_t)ab, (nk_f32_t)a2, (nk_f32_t)b2);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SVESDOT
#endif // NK_TARGET_ARM64_
#endif // NK_SPATIAL_SVESDOT_H
