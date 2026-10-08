/**
 *  @file include/numkong/spatials/sme.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *  @brief Batched spatial distances for ARM SME.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_SME_H
#define NUMKONG_SPATIALS_SME_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_SME_

#include "numkong/dots/serial.h"
#include "numkong/reduce/sve.h" // `nk_svaddv_f64_`
#include "numkong/dots/sme.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_f16_ssve_(nk_f16_t const *data, nk_size_t count) NUMKONG_STREAMING_ {
    svfloat32_t accumulator_even_f32x = svdup_f32(0.0f);
    svfloat32_t accumulator_odd_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    // Lanes past `count` load as zeros, so every widened lane may accumulate.
    svbool_t const widened_b32x = svptrue_b32();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b16x = svwhilelt_b16_u64(i, count);
        svfloat16_t values_f16x = svld1_f16(predicate_b16x, (nk_f16_for_arm_simd_t const *)(data + i));

        svfloat32_t values_even_f32x = svcvt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_even_f32x = svmla_f32_m(widened_b32x, accumulator_even_f32x, values_even_f32x, values_even_f32x);

        svfloat32_t values_odd_f32x = svcvtlt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_odd_f32x = svmla_f32_m(widened_b32x, accumulator_odd_f32x, values_odd_f32x, values_odd_f32x);
    }
    return nk_svaddv_f32_(svptrue_b32(), accumulator_even_f32x) + nk_svaddv_f32_(svptrue_b32(), accumulator_odd_f32x);
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_bf16_ssve_(nk_bf16_t const *data, nk_size_t count) NUMKONG_STREAMING_ {
    svfloat32_t accumulator_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b16x = svwhilelt_b16_u64(i, count);
        svbfloat16_t values_bf16x = svld1_bf16(predicate_b16x, (nk_bf16_for_arm_simd_t const *)(data + i));
        accumulator_f32x = svbfdot_f32(accumulator_f32x, values_bf16x, values_bf16x);
    }
    return nk_svaddv_f32_(svptrue_b32(), accumulator_f32x);
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e4m3_ssve_(nk_e4m3_t const *data, nk_size_t count) NUMKONG_STREAMING_ {
    svfloat32_t accumulator_even_f32x = svdup_f32(0.0f);
    svfloat32_t accumulator_odd_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    // Lanes past `count` load as zeros, so every widened lane may accumulate.
    svbool_t const widened_b32x = svptrue_b32();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        nk_size_t const batch_size = (i + vector_length < count) ? vector_length : (count - i);
        svbool_t predicate_b8x = svwhilelt_b8_u64(0u, batch_size);
        svbool_t predicate_b16x = svwhilelt_b16_u64(0u, batch_size);
        svuint8_t raw_u8x = svld1_u8(predicate_b8x, (nk_u8_t const *)data + i);
        svfloat16_t values_f16x = nk_e4m3x_to_f16x_ssve_(predicate_b16x, raw_u8x);

        svfloat32_t values_even_f32x = svcvt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_even_f32x = svmla_f32_m(widened_b32x, accumulator_even_f32x, values_even_f32x, values_even_f32x);

        svfloat32_t values_odd_f32x = svcvtlt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_odd_f32x = svmla_f32_m(widened_b32x, accumulator_odd_f32x, values_odd_f32x, values_odd_f32x);
    }
    return nk_svaddv_f32_(svptrue_b32(), accumulator_even_f32x) + nk_svaddv_f32_(svptrue_b32(), accumulator_odd_f32x);
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e5m2_ssve_(nk_e5m2_t const *data, nk_size_t count) NUMKONG_STREAMING_ {
    svfloat32_t accumulator_even_f32x = svdup_f32(0.0f);
    svfloat32_t accumulator_odd_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    // Lanes past `count` load as zeros, so every widened lane may accumulate.
    svbool_t const widened_b32x = svptrue_b32();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        nk_size_t const batch_size = (i + vector_length < count) ? vector_length : (count - i);
        svbool_t predicate_b8x = svwhilelt_b8_u64(0u, batch_size);
        svbool_t predicate_b16x = svwhilelt_b16_u64(0u, batch_size);
        svuint8_t raw_u8x = svld1_u8(predicate_b8x, (nk_u8_t const *)data + i);
        svfloat16_t values_f16x = nk_e5m2x_to_f16x_ssve_(predicate_b16x, raw_u8x);

        svfloat32_t values_even_f32x = svcvt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_even_f32x = svmla_f32_m(widened_b32x, accumulator_even_f32x, values_even_f32x, values_even_f32x);

        svfloat32_t values_odd_f32x = svcvtlt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_odd_f32x = svmla_f32_m(widened_b32x, accumulator_odd_f32x, values_odd_f32x, values_odd_f32x);
    }
    return nk_svaddv_f32_(svptrue_b32(), accumulator_even_f32x) + nk_svaddv_f32_(svptrue_b32(), accumulator_odd_f32x);
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m3_ssve_(nk_e2m3_t const *data, nk_size_t count) NUMKONG_STREAMING_ {
    svint32_t accumulator_i32x = svdup_s32(0);
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, count);
        svuint8_t raw_u8x = svld1_u8(predicate_b8x, (nk_u8_t const *)data + i);
        svint8_t values_i8x = nk_e2m3x_to_i8x_ssve_(predicate_b8x, raw_u8x);
        accumulator_i32x = svdot_s32(accumulator_i32x, values_i8x, values_i8x);
    }
    return (nk_f32_t)nk_svaddv_s32_(svptrue_b32(), accumulator_i32x) / 256.0f;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m1_ssve_(nk_e2m1x2_t const *data, nk_size_t count) NUMKONG_STREAMING_ {
    svint32_t accumulator_i32x = svdup_s32(0);
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svint8_t values_i8x = nk_e2m1x_to_i8x_ssve_(data + i / 2, count - i);
        accumulator_i32x = svdot_s32(accumulator_i32x, values_i8x, values_i8x);
    }
    return (nk_f32_t)nk_svaddv_s32_(svptrue_b32(), accumulator_i32x) * 0.25f;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e3m2_ssve_(nk_e3m2_t const *data, nk_size_t count) NUMKONG_STREAMING_ {
    svfloat32_t accumulator_even_f32x = svdup_f32(0.0f);
    svfloat32_t accumulator_odd_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    // Lanes past `count` load as zeros, so every widened lane may accumulate.
    svbool_t const widened_b32x = svptrue_b32();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        nk_size_t const batch_size = (i + vector_length < count) ? vector_length : (count - i);
        svbool_t predicate_b8x = svwhilelt_b8_u64(0u, batch_size);
        svbool_t predicate_b16x = svwhilelt_b16_u64(0u, batch_size);
        svuint8_t raw_u8x = svld1_u8(predicate_b8x, (nk_u8_t const *)data + i);
        svfloat16_t values_f16x = nk_e3m2x_to_f16x_ssve_(predicate_b16x, raw_u8x);

        svfloat32_t values_even_f32x = svcvt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_even_f32x = svmla_f32_m(widened_b32x, accumulator_even_f32x, values_even_f32x, values_even_f32x);

        svfloat32_t values_odd_f32x = svcvtlt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_odd_f32x = svmla_f32_m(widened_b32x, accumulator_odd_f32x, values_odd_f32x, values_odd_f32x);
    }
    return nk_svaddv_f32_(svptrue_b32(), accumulator_even_f32x) + nk_svaddv_f32_(svptrue_b32(), accumulator_odd_f32x);
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_i8_ssve_(nk_i8_t const *data, nk_size_t count) NUMKONG_STREAMING_ {
    svint32_t accumulator_i32x = svdup_s32(0);
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, count);
        svint8_t loaded_i8x = svld1_s8(predicate_b8x, data + i);
        accumulator_i32x = svdot_s32(accumulator_i32x, loaded_i8x, loaded_i8x);
    }
    return (nk_u32_t)nk_svaddv_s32_(svptrue_b32(), accumulator_i32x);
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_u8_ssve_(nk_u8_t const *data, nk_size_t count) NUMKONG_STREAMING_ {
    svuint32_t accumulator_u32x = svdup_u32(0);
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, count);
        svuint8_t loaded_u8x = svld1_u8(predicate_b8x, data + i);
        accumulator_u32x = svdot_u32(accumulator_u32x, loaded_u8x, loaded_u8x);
    }
    return (nk_u32_t)nk_svaddv_u32_(svptrue_b32(), accumulator_u32x);
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_i4_ssve_(nk_i4x2_t const *data, nk_size_t count) NUMKONG_STREAMING_ {
    svint32_t accumulator_i32x = svdup_s32(0);
    nk_u8_t const *bytes = (nk_u8_t const *)data;
    nk_size_t const byte_count = count / NUMKONG_NIBBLES_PER_BYTE;
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < byte_count; i += vector_length) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, byte_count);
        svuint8_t packed_u8x = svld1_u8(predicate_b8x, bytes + i);
        svuint8_t low_u8x = svand_n_u8_x(predicate_b8x, packed_u8x, 0x0F);
        svuint8_t high_u8x = svlsr_n_u8_x(predicate_b8x, packed_u8x, 4);
        // Sign-extend 4-bit to 8-bit: shift left 4, arithmetic shift right 4
        svint8_t low_i8x = svasr_n_s8_x(predicate_b8x, svreinterpret_s8_u8(svlsl_n_u8_x(predicate_b8x, low_u8x, 4)), 4);
        svint8_t high_i8x = svasr_n_s8_x(predicate_b8x, svreinterpret_s8_u8(svlsl_n_u8_x(predicate_b8x, high_u8x, 4)),
                                         4);
        accumulator_i32x = svdot_s32(accumulator_i32x, low_i8x, low_i8x);
        accumulator_i32x = svdot_s32(accumulator_i32x, high_i8x, high_i8x);
    }
    return (nk_u32_t)nk_svaddv_s32_(svptrue_b32(), accumulator_i32x);
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_u4_ssve_(nk_u4x2_t const *data, nk_size_t count) NUMKONG_STREAMING_ {
    svuint32_t accumulator_u32x = svdup_u32(0);
    nk_u8_t const *bytes = (nk_u8_t const *)data;
    nk_size_t const byte_count = count / NUMKONG_NIBBLES_PER_BYTE;
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < byte_count; i += vector_length) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, byte_count);
        svuint8_t packed_u8x = svld1_u8(predicate_b8x, bytes + i);
        svuint8_t low_u8x = svand_n_u8_x(predicate_b8x, packed_u8x, 0x0F);
        svuint8_t high_u8x = svlsr_n_u8_x(predicate_b8x, packed_u8x, 4);
        accumulator_u32x = svdot_u32(accumulator_u32x, low_u8x, low_u8x);
        accumulator_u32x = svdot_u32(accumulator_u32x, high_u8x, high_u8x);
    }
    return (nk_u32_t)nk_svaddv_u32_(svptrue_b32(), accumulator_u32x);
}

NUMKONG_INLINE svfloat32_t nk_angulars_from_dot_f32x_ssve_(svbool_t predicate_b32x, svfloat32_t dots_f32x,
                                                           svfloat32_t query_norm_sq_f32x,
                                                           svfloat32_t target_norms_sq_f32x) NUMKONG_STREAMING_ {
    // Separate roots avoid overflowing the product of two finite norms.
    svfloat32_t const query_rsqrt_f32x = svdiv_f32_x(predicate_b32x, svdup_f32(1),
                                                     svsqrt_f32_x(predicate_b32x, query_norm_sq_f32x));
    svfloat32_t const target_rsqrt_f32x = svdiv_f32_x(predicate_b32x, svdup_f32(1),
                                                      svsqrt_f32_x(predicate_b32x, target_norms_sq_f32x));
    svfloat32_t const rsqrt_f32x = svmul_f32_x(predicate_b32x, query_rsqrt_f32x, target_rsqrt_f32x);
    svfloat32_t const angular_f32x = svmax_n_f32_x(
        predicate_b32x, svsub_f32_x(predicate_b32x, svdup_f32(1), svmul_f32_x(predicate_b32x, dots_f32x, rsqrt_f32x)),
        0);
    svbool_t const query_zero_b32x = svcmpeq_n_f32(predicate_b32x, query_norm_sq_f32x, 0);
    svbool_t const target_zero_b32x = svcmpeq_n_f32(predicate_b32x, target_norms_sq_f32x, 0);
    // One zero norm or an exactly zero dot gives 1, two zero norms 0, and a NaN dot stays NaN
    svbool_t const ones_b32x = svorr_b_z(predicate_b32x, svorr_b_z(predicate_b32x, query_zero_b32x, target_zero_b32x),
                                         svcmpeq_n_f32(predicate_b32x, dots_f32x, 0));
    svfloat32_t const ruled_f32x = svsel_f32(svand_b_z(predicate_b32x, query_zero_b32x, target_zero_b32x), svdup_f32(0),
                                             svsel_f32(ones_b32x, svdup_f32(1), angular_f32x));
    return svsel_f32(svcmpuo_f32(predicate_b32x, dots_f32x, dots_f32x), dots_f32x, ruled_f32x);
}

NUMKONG_INLINE svfloat32_t nk_euclideans_from_dot_f32x_ssve_(svbool_t predicate_b32x, svfloat32_t dots_f32x,
                                                             svfloat32_t query_norm_sq_f32x,
                                                             svfloat32_t target_norms_sq_f32x) NUMKONG_STREAMING_ {
    svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_b32x, query_norm_sq_f32x, target_norms_sq_f32x);
    svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_b32x, sum_sq_f32x,
                                           svmul_f32_x(predicate_b32x, svdup_n_f32(2.0f), dots_f32x));
    // FMAX keeps a NaN, which FMAXNM would replace with zero
    dist_sq_f32x = svmax_n_f32_x(predicate_b32x, dist_sq_f32x, 0);
    return svsqrt_f32_x(predicate_b32x, dist_sq_f32x);
}

/** Angular distances of one half of the lanes from 64-bit integer dots d, their squares, the
 *  products ab of the query norm @p query_sumsq and the target norms b, and those target norms, by
 *  the rule of @c nk_angular_through_i32_from_dot_serial_: a positive dot takes the exact
 *  numerator of (ab − d²) / (ab + d · √(ab)), so equal vectors give exactly 0. */
NUMKONG_INLINE svfloat64_t nk_angulars_from_dot_i64x_ssve_(svint64_t dots_i64x, svuint64_t squares_u64x,
                                                           svuint64_t products_u64x, svuint64_t targets_u64x,
                                                           nk_u32_t query_sumsq) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b64x = svptrue_b64();
    svfloat64_t const products_f64x = svcvt_f64_u64_x(predicate_all_b64x, products_u64x);
    svfloat64_t const roots_f64x = svsqrt_f64_x(predicate_all_b64x, products_f64x);
    svfloat64_t const dots_f64x = svcvt_f64_s64_x(predicate_all_b64x, dots_i64x);
    svfloat64_t const positive_f64x = svdiv_f64_x(
        predicate_all_b64x,
        svcvt_f64_u64_x(predicate_all_b64x, svsub_u64_x(predicate_all_b64x, products_u64x, squares_u64x)),
        svmla_f64_x(predicate_all_b64x, products_f64x, dots_f64x, roots_f64x));
    svfloat64_t const other_f64x = svsub_f64_x(predicate_all_b64x, svdup_f64(1),
                                               svdiv_f64_x(predicate_all_b64x, dots_f64x, roots_f64x));
    svfloat64_t const angular_f64x = svsel_f64(svcmpgt_n_s64(predicate_all_b64x, dots_i64x, 0), positive_f64x,
                                               other_f64x);
    // A zero product means a zero norm: 0 when both are zero, else 1
    svbool_t const any_norm_b64x = svcmpne_n_u64(predicate_all_b64x,
                                                 svorr_n_u64_x(predicate_all_b64x, targets_u64x, query_sumsq), 0);
    return svsel_f64(svcmpeq_n_u64(predicate_all_b64x, products_u64x, 0),
                     svsel_f64(any_norm_b64x, svdup_f64(1), svdup_f64(0)), angular_f64x);
}

/** Angular distances of i32 dots against the u32 query norm @p query_sumsq and u32 target norms,
 *  forming ab and d² exactly with SVE2 widening multiplies and rounding once into F32. */
NUMKONG_INLINE svfloat32_t nk_angulars_from_dot_i32x_ssve_(svint32_t dots_i32x, nk_u32_t query_sumsq,
                                                           svuint32_t targets_u32x) NUMKONG_STREAMING_ {
    svfloat64_t const even_f64x = nk_angulars_from_dot_i64x_ssve_(
        svmovlb_s64(dots_i32x), svreinterpret_u64_s64(svmullb_s64(dots_i32x, dots_i32x)),
        svmullb_n_u64(targets_u32x, query_sumsq), svmovlb_u64(targets_u32x), query_sumsq);
    svfloat64_t const odd_f64x = nk_angulars_from_dot_i64x_ssve_(
        svmovlt_s64(dots_i32x), svreinterpret_u64_s64(svmullt_s64(dots_i32x, dots_i32x)),
        svmullt_n_u64(targets_u32x, query_sumsq), svmovlt_u64(targets_u32x), query_sumsq);
    svbool_t const predicate_all_b64x = svptrue_b64();
    return svcvtnt_f32_f64_m(svcvt_f32_f64_x(predicate_all_b64x, even_f64x), predicate_all_b64x, odd_f64x);
}

/** Angular distances of u32 dots against the u32 query norm @p query_sumsq and u32 target norms,
 *  as @c nk_angulars_from_dot_i32x_ssve_ forms them; a zero dot gives exactly 1. */
NUMKONG_INLINE svfloat32_t nk_angulars_from_dot_u32x_ssve_(svuint32_t dots_u32x, nk_u32_t query_sumsq,
                                                           svuint32_t targets_u32x) NUMKONG_STREAMING_ {
    svfloat64_t const even_f64x = nk_angulars_from_dot_i64x_ssve_(
        svreinterpret_s64_u64(svmovlb_u64(dots_u32x)), svmullb_u64(dots_u32x, dots_u32x),
        svmullb_n_u64(targets_u32x, query_sumsq), svmovlb_u64(targets_u32x), query_sumsq);
    svfloat64_t const odd_f64x = nk_angulars_from_dot_i64x_ssve_(
        svreinterpret_s64_u64(svmovlt_u64(dots_u32x)), svmullt_u64(dots_u32x, dots_u32x),
        svmullt_n_u64(targets_u32x, query_sumsq), svmovlt_u64(targets_u32x), query_sumsq);
    svbool_t const predicate_all_b64x = svptrue_b64();
    return svcvtnt_f32_f64_m(svcvt_f32_f64_x(predicate_all_b64x, even_f64x), predicate_all_b64x, odd_f64x);
}

/** Euclidean distances of 64-bit integer dots d against the query norm @p query_sumsq and target
 *  norms b: a + b − 2d is exact in 64 bits and its root rounds once. */
NUMKONG_INLINE svfloat64_t nk_euclideans_from_dot_i64x_ssve_(svint64_t dots_i64x, svuint64_t targets_u64x,
                                                             nk_u32_t query_sumsq) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b64x = svptrue_b64();
    svint64_t const distances_i64x = svsub_s64_x(
        predicate_all_b64x, svadd_n_s64_x(predicate_all_b64x, svreinterpret_s64_u64(targets_u64x), query_sumsq),
        svlsl_n_s64_x(predicate_all_b64x, dots_i64x, 1));
    return svsqrt_f64_x(predicate_all_b64x, svcvt_f64_s64_x(predicate_all_b64x, distances_i64x));
}

/** Euclidean distances of i32 dots against u32 norms, exact before one rounding into F32. */
NUMKONG_INLINE svfloat32_t nk_euclideans_from_dot_i32x_ssve_(svint32_t dots_i32x, nk_u32_t query_sumsq,
                                                             svuint32_t targets_u32x) NUMKONG_STREAMING_ {
    svfloat64_t const even_f64x = nk_euclideans_from_dot_i64x_ssve_(svmovlb_s64(dots_i32x), svmovlb_u64(targets_u32x),
                                                                    query_sumsq);
    svfloat64_t const odd_f64x = nk_euclideans_from_dot_i64x_ssve_(svmovlt_s64(dots_i32x), svmovlt_u64(targets_u32x),
                                                                   query_sumsq);
    svbool_t const predicate_all_b64x = svptrue_b64();
    return svcvtnt_f32_f64_m(svcvt_f32_f64_x(predicate_all_b64x, even_f64x), predicate_all_b64x, odd_f64x);
}

/** Euclidean distances of u32 dots against u32 norms, exact before one rounding into F32. */
NUMKONG_INLINE svfloat32_t nk_euclideans_from_dot_u32x_ssve_(svuint32_t dots_u32x, nk_u32_t query_sumsq,
                                                             svuint32_t targets_u32x) NUMKONG_STREAMING_ {
    svfloat64_t const even_f64x = nk_euclideans_from_dot_i64x_ssve_(svreinterpret_s64_u64(svmovlb_u64(dots_u32x)),
                                                                    svmovlb_u64(targets_u32x), query_sumsq);
    svfloat64_t const odd_f64x = nk_euclideans_from_dot_i64x_ssve_(svreinterpret_s64_u64(svmovlt_u64(dots_u32x)),
                                                                   svmovlt_u64(targets_u32x), query_sumsq);
    svbool_t const predicate_all_b64x = svptrue_b64();
    return svcvtnt_f32_f64_m(svcvt_f32_f64_x(predicate_all_b64x, even_f64x), predicate_all_b64x, odd_f64x);
}

/** Angular distances in place of @p count relative dots of one row, times @p mantissa, against
 *  the relative squared norms @p row_norm and @p column_norms, whose exponents cancel. */
NUMKONG_INLINE void nk_angulars_from_relative_ssve_(nk_f32_t *values, nk_size_t count, nk_f32_t mantissa,
                                                    nk_f32_t row_norm,
                                                    nk_f32_t const *column_norms) NUMKONG_STREAMING_ {
    for (nk_size_t column = 0; column < count; column += svcntw()) {
        svbool_t const predicate_b32x = svwhilelt_b32_u64(column, count);
        svfloat32_t const dots_f32x = svmul_n_f32_x(predicate_b32x, svld1_f32(predicate_b32x, values + column),
                                                    mantissa);
        svst1_f32(predicate_b32x, values + column,
                  nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, svdup_n_f32(row_norm),
                                                  svld1_f32(predicate_b32x, column_norms + column)));
    }
}

/** Euclidean distances in place of @p count relative dots of one row, times @p mantissa, the row's
 *  squared norm being @p row_norm times 4 to the @p row_exponent and each column's its norm times
 *  4^E; every term scales to the larger exponent before the root, which scales back once. */
NUMKONG_INLINE void nk_euclideans_from_relative_ssve_(nk_f32_t *values, nk_size_t count, nk_f32_t mantissa,
                                                      nk_f32_t row_norm, nk_i32_t row_exponent,
                                                      nk_f32_t const *column_norms,
                                                      nk_i32_t const *column_exponents) NUMKONG_STREAMING_ {
    for (nk_size_t column = 0; column < count; column += svcntw()) {
        svbool_t const predicate_b32x = svwhilelt_b32_u64(column, count);
        svint32_t const exponents_i32x = svld1_s32(predicate_b32x, column_exponents + column);
        svint32_t const top_i32x = svmax_n_s32_x(predicate_b32x, exponents_i32x, row_exponent);
        svfloat32_t const row_f32x = svscale_f32_x(
            predicate_b32x, svdup_n_f32(row_norm),
            svlsl_n_s32_x(predicate_b32x, svsubr_n_s32_x(predicate_b32x, top_i32x, row_exponent), 1));
        svfloat32_t const column_f32x = svscale_f32_x(
            predicate_b32x, svld1_f32(predicate_b32x, column_norms + column),
            svlsl_n_s32_x(predicate_b32x, svsub_s32_x(predicate_b32x, exponents_i32x, top_i32x), 1));
        svfloat32_t const dots_f32x = svscale_f32_x(
            predicate_b32x, svmul_n_f32_x(predicate_b32x, svld1_f32(predicate_b32x, values + column), 2 * mantissa),
            svsub_s32_x(predicate_b32x, svadd_n_s32_x(predicate_b32x, exponents_i32x, row_exponent),
                        svlsl_n_s32_x(predicate_b32x, top_i32x, 1)));
        // FMAX keeps a NaN, which FMAXNM would replace with zero
        svfloat32_t const squares_f32x = svmax_n_f32_x(
            predicate_b32x, svsub_f32_x(predicate_b32x, svadd_f32_x(predicate_b32x, row_f32x, column_f32x), dots_f32x),
            0);
        svst1_f32(predicate_b32x, values + column,
                  svscale_f32_x(predicate_b32x, svsqrt_f32_x(predicate_b32x, squares_f32x), top_i32x));
    }
}

#pragma region F16 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_f16_sme_finalize_ssve_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_f16_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

#if NUMKONG_TARGET_SME
NUMKONG_API nk_status_t nk_angulars_packed_f16_sme( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_f16_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_f16_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                              c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_f16_sme_finalize_ssve_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_f16_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_f16_sme( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_f16_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_f16_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_f16_sme_finalize_ssve_( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f16_sme( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_f16_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                         rows_end);
    nk_angulars_symmetric_f16_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                 result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_f16_sme_finalize_ssve_( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_sme( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_f16_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                         rows_end);
    nk_euclideans_symmetric_f16_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                   result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion F16 Floats

#pragma region BF16 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_bf16_sme_finalize_ssve_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_bf16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_bf16_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_bf16_sme( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_bf16_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_bf16_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_bf16_sme_finalize_ssve_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_bf16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_bf16_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_bf16_sme( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_bf16_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_bf16_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_bf16_sme_finalize_ssve_( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_sme( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_bf16_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_angulars_symmetric_bf16_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                  result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_bf16_sme_finalize_ssve_( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_sme( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_bf16_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_euclideans_symmetric_bf16_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                    result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion BF16 Floats

#pragma region E4M3 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_e4m3_sme_finalize_ssve_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e4m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e4m3_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_e4m3_sme( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_e4m3_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_e4m3_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_e4m3_sme_finalize_ssve_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e4m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e4m3_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_sme( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_e4m3_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_e4m3_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_e4m3_sme_finalize_ssve_( //
    nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_sme( //
    nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_e4m3_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_angulars_symmetric_e4m3_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                  result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_e4m3_sme_finalize_ssve_( //
    nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_sme( //
    nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_e4m3_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_euclideans_symmetric_e4m3_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                    result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion E4M3 Floats

#pragma region E5M2 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_e5m2_sme_finalize_ssve_( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e5m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e5m2_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_e5m2_sme( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_e5m2_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_e5m2_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_e5m2_sme_finalize_ssve_( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e5m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e5m2_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_sme( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_e5m2_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_e5m2_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_e5m2_sme_finalize_ssve_( //
    nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_sme( //
    nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_e5m2_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_angulars_symmetric_e5m2_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                  result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_e5m2_sme_finalize_ssve_( //
    nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_sme( //
    nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_e5m2_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_euclideans_symmetric_e5m2_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                    result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion E5M2 Floats

#pragma region E2M3 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_e2m3_sme_finalize_ssve_( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m3_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_e2m3_sme( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_e2m3_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_e2m3_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_e2m3_sme_finalize_ssve_( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m3_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_sme( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_e2m3_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_e2m3_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_e2m3_sme_finalize_ssve_( //
    nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_sme( //
    nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_e2m3_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_angulars_symmetric_e2m3_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                  result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_e2m3_sme_finalize_ssve_( //
    nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_sme( //
    nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_e2m3_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_euclideans_symmetric_e2m3_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                    result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion E2M3 Floats

#pragma region E2M1 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_e2m1_sme_finalize_ssve_( //
    nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m1x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m1_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_e2m1_sme( //
    nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m1x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_e2m1_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_e2m1_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_e2m1_sme_finalize_ssve_( //
    nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m1x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m1_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_sme( //
    nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m1x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_e2m1_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_e2m1_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_e2m1_sme_finalize_ssve_( //
    nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m1_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m1_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_sme( //
    nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_e2m1x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_e2m1_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_angulars_symmetric_e2m1_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                  result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_e2m1_sme_finalize_ssve_( //
    nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m1_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m1_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_sme( //
    nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_e2m1x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_e2m1_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_euclideans_symmetric_e2m1_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                    result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion E2M1 Floats

#pragma region E3M2 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_e3m2_sme_finalize_ssve_( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e3m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e3m2_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_e3m2_sme( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_e3m2_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_e3m2_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_e3m2_sme_finalize_ssve_( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e3m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e3m2_ssve_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_sme( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_e3m2_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_e3m2_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_e3m2_sme_finalize_ssve_( //
    nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_sme( //
    nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_e3m2_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_angulars_symmetric_e3m2_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                  result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_e3m2_sme_finalize_ssve_( //
    nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_sme( //
    nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_e3m2_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_euclideans_symmetric_e3m2_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                    result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion E3M2 Floats
#pragma region I8 Integers

NUMKONG_OUTLINED_ void nk_angulars_packed_i8_sme_finalize_ssve_( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_i8_ssve_(a_row, depth);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svint32_t dots_i32x = svld1_s32(predicate_b32x, (nk_i32_t const *)(result_row + col_index));
            svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_angulars_from_dot_i32x_ssve_(dots_i32x, query_norm_sq_u32, target_norms_sq_u32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_i8_sme( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_i8_sme_streaming_(a, a_stride, b_packed, (nk_i32_t *)c, rows, columns, depth, c_stride);
    nk_angulars_packed_i8_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                             c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_i8_sme_finalize_ssve_( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_i8_ssve_(a_row, depth);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svint32_t dots_i32x = svld1_s32(predicate_b32x, (nk_i32_t const *)(result_row + col_index));
            svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_euclideans_from_dot_i32x_ssve_(dots_i32x, query_norm_sq_u32, target_norms_sq_u32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_i8_sme( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_i8_sme_streaming_(a, a_stride, b_packed, (nk_i32_t *)c, rows, columns, depth, c_stride);
    nk_euclideans_packed_i8_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_i8_sme_finalize_ssve_( //
    nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svint32_t dots_i32x = svld1_s32(predicate_b32x, (nk_i32_t *)(result_row + col_index));
                svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_i32x_ssve_(dots_i32x, query_sumsq_u32, target_norms_sq_u32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_i8_sme( //
    nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_i8_sme_streaming_(vectors, stride, vector_count, depth, (nk_i32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_angulars_symmetric_i8_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_i8_sme_finalize_ssve_( //
    nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svint32_t dots_i32x = svld1_s32(predicate_b32x, (nk_i32_t *)(result_row + col_index));
                svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_i32x_ssve_(dots_i32x, query_sumsq_u32, target_norms_sq_u32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_sme( //
    nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_i8_sme_streaming_(vectors, stride, vector_count, depth, (nk_i32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_euclideans_symmetric_i8_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                  result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion I8 Integers

#pragma region U8 Integers

NUMKONG_OUTLINED_ void nk_angulars_packed_u8_sme_finalize_ssve_( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_u8_ssve_(a_row, depth);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svuint32_t dots_u32x = svld1_u32(predicate_b32x, (nk_u32_t const *)(result_row + col_index));
            svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_angulars_from_dot_u32x_ssve_(dots_u32x, query_norm_sq_u32, target_norms_sq_u32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_u8_sme( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_u8_sme_streaming_(a, a_stride, b_packed, (nk_u32_t *)c, rows, columns, depth, c_stride);
    nk_angulars_packed_u8_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                             c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_u8_sme_finalize_ssve_( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_u8_ssve_(a_row, depth);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svuint32_t dots_u32x = svld1_u32(predicate_b32x, (nk_u32_t const *)(result_row + col_index));
            svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_euclideans_from_dot_u32x_ssve_(dots_u32x, query_norm_sq_u32, target_norms_sq_u32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_u8_sme( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_u8_sme_streaming_(a, a_stride, b_packed, (nk_u32_t *)c, rows, columns, depth, c_stride);
    nk_euclideans_packed_u8_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_u8_sme_finalize_ssve_( //
    nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svuint32_t dots_u32x = svld1_u32(predicate_b32x, (nk_u32_t *)(result_row + col_index));
                svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_u32x_ssve_(dots_u32x, query_sumsq_u32, target_norms_sq_u32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_u8_sme( //
    nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_u8_sme_streaming_(vectors, stride, vector_count, depth, (nk_u32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_angulars_symmetric_u8_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_u8_sme_finalize_ssve_( //
    nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svuint32_t dots_u32x = svld1_u32(predicate_b32x, (nk_u32_t *)(result_row + col_index));
                svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_u32x_ssve_(dots_u32x, query_sumsq_u32, target_norms_sq_u32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_sme( //
    nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_u8_sme_streaming_(vectors, stride, vector_count, depth, (nk_u32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_euclideans_symmetric_u8_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                  result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion U8 Integers

#pragma region I4 Integers

NUMKONG_OUTLINED_ void nk_angulars_packed_i4_sme_finalize_ssve_( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i4x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_i4_ssve_(a_row, depth);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svint32_t dots_i32x = svld1_s32(predicate_b32x, (nk_i32_t const *)(result_row + col_index));
            svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_angulars_from_dot_i32x_ssve_(dots_i32x, query_norm_sq_u32, target_norms_sq_u32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_i4_sme( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_i4_sme_streaming_(a, a_stride, b_packed, (nk_i32_t *)c, rows, columns, depth, c_stride);
    nk_angulars_packed_i4_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                             c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_i4_sme_finalize_ssve_( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i4x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_i4_ssve_(a_row, depth);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svint32_t dots_i32x = svld1_s32(predicate_b32x, (nk_i32_t const *)(result_row + col_index));
            svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_euclideans_from_dot_i32x_ssve_(dots_i32x, query_norm_sq_u32, target_norms_sq_u32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_i4_sme( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_i4_sme_streaming_(a, a_stride, b_packed, (nk_i32_t *)c, rows, columns, depth, c_stride);
    nk_euclideans_packed_i4_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_i4_sme_finalize_ssve_( //
    nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i4_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svint32_t dots_i32x = svld1_s32(predicate_b32x, (nk_i32_t *)(result_row + col_index));
                svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_i32x_ssve_(dots_i32x, query_sumsq_u32, target_norms_sq_u32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_i4_sme( //
    nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_i4_sme_streaming_(vectors, stride, vector_count, depth, (nk_i32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_angulars_symmetric_i4_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_i4_sme_finalize_ssve_( //
    nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i4_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svint32_t dots_i32x = svld1_s32(predicate_b32x, (nk_i32_t *)(result_row + col_index));
                svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_i32x_ssve_(dots_i32x, query_sumsq_u32, target_norms_sq_u32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_sme( //
    nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_i4_sme_streaming_(vectors, stride, vector_count, depth, (nk_i32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_euclideans_symmetric_i4_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                  result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion Signed Integers

#pragma region U4 Integers

NUMKONG_OUTLINED_ void nk_angulars_packed_u4_sme_finalize_ssve_( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u4x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_u4_ssve_(a_row, depth);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svuint32_t dots_u32x = svld1_u32(predicate_b32x, (nk_u32_t const *)(result_row + col_index));
            svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_angulars_from_dot_u32x_ssve_(dots_u32x, query_norm_sq_u32, target_norms_sq_u32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_u4_sme( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_u4_sme_streaming_(a, a_stride, b_packed, (nk_u32_t *)c, rows, columns, depth, c_stride);
    nk_angulars_packed_u4_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                             c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_u4_sme_finalize_ssve_( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u4x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_u4_ssve_(a_row, depth);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svuint32_t dots_u32x = svld1_u32(predicate_b32x, (nk_u32_t const *)(result_row + col_index));
            svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_euclideans_from_dot_u32x_ssve_(dots_u32x, query_norm_sq_u32, target_norms_sq_u32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_u4_sme( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_u4_sme_streaming_(a, a_stride, b_packed, (nk_u32_t *)c, rows, columns, depth, c_stride);
    nk_euclideans_packed_u4_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_u4_sme_finalize_ssve_( //
    nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u4_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svuint32_t dots_u32x = svld1_u32(predicate_b32x, (nk_u32_t *)(result_row + col_index));
                svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_u32x_ssve_(dots_u32x, query_sumsq_u32, target_norms_sq_u32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_u4_sme( //
    nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_u4_sme_streaming_(vectors, stride, vector_count, depth, (nk_u32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_angulars_symmetric_u4_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_u4_sme_finalize_ssve_( //
    nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u4_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svuint32_t dots_u32x = svld1_u32(predicate_b32x, (nk_u32_t *)(result_row + col_index));
                svuint32_t target_norms_sq_u32x = svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_u32x_ssve_(dots_u32x, query_sumsq_u32, target_norms_sq_u32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_sme( //
    nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_u4_sme_streaming_(vectors, stride, vector_count, depth, (nk_u32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_euclideans_symmetric_u4_sme_finalize_ssve_(vectors, vector_count, depth, stride_elements, result,
                                                  result_stride_elements, rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion Unsigned Integers

/** Turns packed NVFP4 relative dots into angular distances. */
NUMKONG_OUTLINED_ void nk_angulars_packed_nvfp4_sme_finalize_ssve_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                   void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                                   nk_size_t columns, nk_size_t depth,
                                                                   nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_sme_columns_(b_packed, 16, 4);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(a.tensor_scale), &tensor_exponent) * b.mantissa;
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_nvfp4_sme_(a, a_stride, row, depth, &exponent);
        nk_angulars_from_relative_ssve_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, norm, b.norms);
    }
}

/** Turns symmetric NVFP4 relative dots into angular distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_angulars_symmetric_nvfp4_sme_finalize_ssve_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                      nk_size_t vector_count, nk_size_t depth,
                                                                      nk_f32_t *result, nk_size_t result_stride,
                                                                      nk_size_t rows_begin,
                                                                      nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(vectors.tensor_scale), &tensor_exponent);
    nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column)
            norms[column - chunk] = nk_dots_scaled_row_nvfp4_sme_(vectors, stride, column, depth,
                                                                  &exponents[column - chunk]);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_nvfp4_sme_(vectors, stride, row, depth, &exponent);
            nk_angulars_from_relative_ssve_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                            chunk_end - first, mantissa * mantissa, norm, norms + (first - chunk));
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

/** Turns packed NVFP4 relative dots into Euclidean distances. */
NUMKONG_OUTLINED_ void nk_euclideans_packed_nvfp4_sme_finalize_ssve_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                     void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                                     nk_size_t columns, nk_size_t depth,
                                                                     nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_sme_columns_(b_packed, 16, 4);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(a.tensor_scale), &tensor_exponent) * b.mantissa;
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_nvfp4_sme_(a, a_stride, row, depth, &exponent);
        nk_euclideans_from_relative_ssve_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, norm, exponent,
                                          b.norms, b.exponents);
    }
}

/** Turns symmetric NVFP4 relative dots into Euclidean distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_euclideans_symmetric_nvfp4_sme_finalize_ssve_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                        nk_size_t vector_count, nk_size_t depth,
                                                                        nk_f32_t *result, nk_size_t result_stride,
                                                                        nk_size_t rows_begin,
                                                                        nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(vectors.tensor_scale), &tensor_exponent);
    nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column)
            norms[column - chunk] = nk_dots_scaled_row_nvfp4_sme_(vectors, stride, column, depth,
                                                                  &exponents[column - chunk]);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_nvfp4_sme_(vectors, stride, row, depth, &exponent);
            nk_euclideans_from_relative_ssve_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                              chunk_end - first, mantissa * mantissa, norm, exponent,
                                              norms + (first - chunk), exponents + (first - chunk));
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_sme(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_nvfp4_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_dots_packed_nvfp4_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_nvfp4_sme_finalize_ssve_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_sme(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_(nk_nvfp4_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_nvfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                           rows_end);
    nk_angulars_symmetric_nvfp4_sme_finalize_ssve_(operand, stride, vector_count, depth, result, result_stride,
                                                   rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_sme(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_nvfp4_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_dots_packed_nvfp4_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_nvfp4_sme_finalize_ssve_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_sme(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_(nk_nvfp4_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_nvfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                           rows_end);
    nk_euclideans_symmetric_nvfp4_sme_finalize_ssve_(operand, stride, vector_count, depth, result, result_stride,
                                                     rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

/** Turns packed MXFP4 relative dots into angular distances. */
NUMKONG_OUTLINED_ void nk_angulars_packed_mxfp4_sme_finalize_ssve_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                   void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                                   nk_size_t columns, nk_size_t depth,
                                                                   nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_sme_columns_(b_packed, 32, 4);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(a.tensor_scale), &tensor_exponent) * b.mantissa;
    nk_dots_scaled_exact_mxfp4_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                    depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp4_sme_(a, a_stride, row, depth, &exponent);
        nk_angulars_from_relative_ssve_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, norm, b.norms);
    }
}

/** Turns symmetric MXFP4 relative dots into angular distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_angulars_symmetric_mxfp4_sme_finalize_ssve_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                      nk_size_t vector_count, nk_size_t depth,
                                                                      nk_f32_t *result, nk_size_t result_stride,
                                                                      nk_size_t rows_begin,
                                                                      nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(vectors.tensor_scale), &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_sme_mx_base_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp4_sme_(vectors, stride, column, depth,
                                                                  &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp4_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end, bases,
                                        exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp4_sme_(vectors, stride, row, depth, &exponent);
            nk_angulars_from_relative_ssve_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                            chunk_end - first, mantissa * mantissa, norm, norms + (first - chunk));
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

/** Turns packed MXFP4 relative dots into Euclidean distances. */
NUMKONG_OUTLINED_ void nk_euclideans_packed_mxfp4_sme_finalize_ssve_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                     void const *b_packed, nk_f32_t *c, nk_size_t rows,
                                                                     nk_size_t columns, nk_size_t depth,
                                                                     nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_sme_columns_(b_packed, 32, 4);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(a.tensor_scale), &tensor_exponent) * b.mantissa;
    nk_dots_scaled_exact_mxfp4_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                    depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp4_sme_(a, a_stride, row, depth, &exponent);
        nk_euclideans_from_relative_ssve_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, norm, exponent,
                                          b.norms, b.exponents);
    }
}

/** Turns symmetric MXFP4 relative dots into Euclidean distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_euclideans_symmetric_mxfp4_sme_finalize_ssve_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                        nk_size_t vector_count, nk_size_t depth,
                                                                        nk_f32_t *result, nk_size_t result_stride,
                                                                        nk_size_t rows_begin,
                                                                        nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(vectors.tensor_scale), &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_sme_mx_base_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp4_sme_(vectors, stride, column, depth,
                                                                  &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp4_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end, bases,
                                        exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp4_sme_(vectors, stride, row, depth, &exponent);
            nk_euclideans_from_relative_ssve_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                              chunk_end - first, mantissa * mantissa, norm, exponent,
                                              norms + (first - chunk), exponents + (first - chunk));
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_sme(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp4_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_dots_packed_mxfp4_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_mxfp4_sme_finalize_ssve_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_sme(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp4_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_mxfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                           rows_end);
    nk_angulars_symmetric_mxfp4_sme_finalize_ssve_(operand, stride, vector_count, depth, result, result_stride,
                                                   rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_sme(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp4_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_dots_packed_mxfp4_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_mxfp4_sme_finalize_ssve_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_sme(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp4_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_mxfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                           rows_end);
    nk_euclideans_symmetric_mxfp4_sme_finalize_ssve_(operand, stride, vector_count, depth, result, result_stride,
                                                     rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

/** Turns packed MXFP6 E2M3 relative dots into angular distances. */
NUMKONG_OUTLINED_ void nk_angulars_packed_mxfp6e2m3_sme_finalize_ssve_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                       void const *b_packed, nk_f32_t *c,
                                                                       nk_size_t rows, nk_size_t columns,
                                                                       nk_size_t depth,
                                                                       nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_sme_columns_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(a.tensor_scale), &tensor_exponent) * b.mantissa;
    nk_dots_scaled_exact_mxfp6e2m3_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp6e2m3_sme_(a, a_stride, row, depth, &exponent);
        nk_angulars_from_relative_ssve_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, norm, b.norms);
    }
}

/** Turns symmetric MXFP6 E2M3 relative dots into angular distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_angulars_symmetric_mxfp6e2m3_sme_finalize_ssve_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                          nk_size_t vector_count, nk_size_t depth,
                                                                          nk_f32_t *result, nk_size_t result_stride,
                                                                          nk_size_t rows_begin,
                                                                          nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(vectors.tensor_scale), &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_sme_mx_base_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp6e2m3_sme_(vectors, stride, column, depth,
                                                                      &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp6e2m3_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp6e2m3_sme_(vectors, stride, row, depth, &exponent);
            nk_angulars_from_relative_ssve_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                            chunk_end - first, mantissa * mantissa, norm, norms + (first - chunk));
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

/** Turns packed MXFP6 E2M3 relative dots into Euclidean distances. */
NUMKONG_OUTLINED_ void nk_euclideans_packed_mxfp6e2m3_sme_finalize_ssve_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                         void const *b_packed, nk_f32_t *c,
                                                                         nk_size_t rows, nk_size_t columns,
                                                                         nk_size_t depth,
                                                                         nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_sme_columns_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(a.tensor_scale), &tensor_exponent) * b.mantissa;
    nk_dots_scaled_exact_mxfp6e2m3_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp6e2m3_sme_(a, a_stride, row, depth, &exponent);
        nk_euclideans_from_relative_ssve_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, norm, exponent,
                                          b.norms, b.exponents);
    }
}

/** Turns symmetric MXFP6 E2M3 relative dots into Euclidean distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_euclideans_symmetric_mxfp6e2m3_sme_finalize_ssve_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(vectors.tensor_scale), &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_sme_mx_base_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp6e2m3_sme_(vectors, stride, column, depth,
                                                                      &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp6e2m3_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp6e2m3_sme_(vectors, stride, row, depth, &exponent);
            nk_euclideans_from_relative_ssve_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                              chunk_end - first, mantissa * mantissa, norm, exponent,
                                              norms + (first - chunk), exponents + (first - chunk));
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e2m3_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_dots_packed_mxfp6e2m3_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_mxfp6e2m3_sme_finalize_ssve_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e2m3_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_mxfp6e2m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_angulars_symmetric_mxfp6e2m3_sme_finalize_ssve_(operand, stride, vector_count, depth, result, result_stride,
                                                       rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e2m3_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_dots_packed_mxfp6e2m3_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_mxfp6e2m3_sme_finalize_ssve_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *vectors,
                                                              nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                                              nk_f32_t *result, nk_size_t result_stride,
                                                              nk_size_t rows_begin, nk_size_t rows_end,
                                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e2m3_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_mxfp6e2m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_euclideans_symmetric_mxfp6e2m3_sme_finalize_ssve_(operand, stride, vector_count, depth, result, result_stride,
                                                         rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

/** Turns packed MXFP6 E3M2 relative dots into angular distances. */
NUMKONG_OUTLINED_ void nk_angulars_packed_mxfp6e3m2_sme_finalize_ssve_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                       void const *b_packed, nk_f32_t *c,
                                                                       nk_size_t rows, nk_size_t columns,
                                                                       nk_size_t depth,
                                                                       nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_sme_columns_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(a.tensor_scale), &tensor_exponent) * b.mantissa;
    nk_dots_scaled_exact_mxfp6e3m2_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp6e3m2_sme_(a, a_stride, row, depth, &exponent);
        nk_angulars_from_relative_ssve_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, norm, b.norms);
    }
}

/** Turns symmetric MXFP6 E3M2 relative dots into angular distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_angulars_symmetric_mxfp6e3m2_sme_finalize_ssve_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                          nk_size_t vector_count, nk_size_t depth,
                                                                          nk_f32_t *result, nk_size_t result_stride,
                                                                          nk_size_t rows_begin,
                                                                          nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(vectors.tensor_scale), &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_sme_mx_base_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp6e3m2_sme_(vectors, stride, column, depth,
                                                                      &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp6e3m2_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp6e3m2_sme_(vectors, stride, row, depth, &exponent);
            nk_angulars_from_relative_ssve_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                            chunk_end - first, mantissa * mantissa, norm, norms + (first - chunk));
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

/** Turns packed MXFP6 E3M2 relative dots into Euclidean distances. */
NUMKONG_OUTLINED_ void nk_euclideans_packed_mxfp6e3m2_sme_finalize_ssve_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                         void const *b_packed, nk_f32_t *c,
                                                                         nk_size_t rows, nk_size_t columns,
                                                                         nk_size_t depth,
                                                                         nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_sme_columns_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(a.tensor_scale), &tensor_exponent) * b.mantissa;
    nk_dots_scaled_exact_mxfp6e3m2_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp6e3m2_sme_(a, a_stride, row, depth, &exponent);
        nk_euclideans_from_relative_ssve_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, norm, exponent,
                                          b.norms, b.exponents);
    }
}

/** Turns symmetric MXFP6 E3M2 relative dots into Euclidean distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_euclideans_symmetric_mxfp6e3m2_sme_finalize_ssve_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(vectors.tensor_scale), &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_sme_mx_base_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp6e3m2_sme_(vectors, stride, column, depth,
                                                                      &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp6e3m2_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp6e3m2_sme_(vectors, stride, row, depth, &exponent);
            nk_euclideans_from_relative_ssve_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                              chunk_end - first, mantissa * mantissa, norm, exponent,
                                              norms + (first - chunk), exponents + (first - chunk));
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e3m2_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_dots_packed_mxfp6e3m2_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_mxfp6e3m2_sme_finalize_ssve_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e3m2_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_mxfp6e3m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_angulars_symmetric_mxfp6e3m2_sme_finalize_ssve_(operand, stride, vector_count, depth, result, result_stride,
                                                       rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e3m2_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_dots_packed_mxfp6e3m2_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_mxfp6e3m2_sme_finalize_ssve_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *vectors,
                                                              nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                                              nk_f32_t *result, nk_size_t result_stride,
                                                              nk_size_t rows_begin, nk_size_t rows_end,
                                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e3m2_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_mxfp6e3m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_euclideans_symmetric_mxfp6e3m2_sme_finalize_ssve_(operand, stride, vector_count, depth, result, result_stride,
                                                         rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

/** Turns packed MXFP8 E4M3 relative dots into angular distances. */
NUMKONG_OUTLINED_ void nk_angulars_packed_mxfp8e4m3_sme_finalize_ssve_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                       void const *b_packed, nk_f32_t *c,
                                                                       nk_size_t rows, nk_size_t columns,
                                                                       nk_size_t depth,
                                                                       nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_sme_columns_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(a.tensor_scale), &tensor_exponent) * b.mantissa;
    nk_dots_scaled_exact_mxfp8e4m3_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp8e4m3_sme_(a, a_stride, row, depth, &exponent);
        nk_angulars_from_relative_ssve_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, norm, b.norms);
    }
}

/** Turns symmetric MXFP8 E4M3 relative dots into angular distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_angulars_symmetric_mxfp8e4m3_sme_finalize_ssve_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                          nk_size_t vector_count, nk_size_t depth,
                                                                          nk_f32_t *result, nk_size_t result_stride,
                                                                          nk_size_t rows_begin,
                                                                          nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(vectors.tensor_scale), &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_sme_mx_base_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp8e4m3_sme_(vectors, stride, column, depth,
                                                                      &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp8e4m3_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp8e4m3_sme_(vectors, stride, row, depth, &exponent);
            nk_angulars_from_relative_ssve_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                            chunk_end - first, mantissa * mantissa, norm, norms + (first - chunk));
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

/** Turns packed MXFP8 E4M3 relative dots into Euclidean distances. */
NUMKONG_OUTLINED_ void nk_euclideans_packed_mxfp8e4m3_sme_finalize_ssve_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                         void const *b_packed, nk_f32_t *c,
                                                                         nk_size_t rows, nk_size_t columns,
                                                                         nk_size_t depth,
                                                                         nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_sme_columns_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(a.tensor_scale), &tensor_exponent) * b.mantissa;
    nk_dots_scaled_exact_mxfp8e4m3_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp8e4m3_sme_(a, a_stride, row, depth, &exponent);
        nk_euclideans_from_relative_ssve_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, norm, exponent,
                                          b.norms, b.exponents);
    }
}

/** Turns symmetric MXFP8 E4M3 relative dots into Euclidean distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_euclideans_symmetric_mxfp8e4m3_sme_finalize_ssve_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(vectors.tensor_scale), &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_sme_mx_base_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp8e4m3_sme_(vectors, stride, column, depth,
                                                                      &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp8e4m3_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp8e4m3_sme_(vectors, stride, row, depth, &exponent);
            nk_euclideans_from_relative_ssve_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                              chunk_end - first, mantissa * mantissa, norm, exponent,
                                              norms + (first - chunk), exponents + (first - chunk));
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e4m3_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_dots_packed_mxfp8e4m3_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_mxfp8e4m3_sme_finalize_ssve_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e4m3_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_mxfp8e4m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_angulars_symmetric_mxfp8e4m3_sme_finalize_ssve_(operand, stride, vector_count, depth, result, result_stride,
                                                       rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e4m3_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_dots_packed_mxfp8e4m3_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_mxfp8e4m3_sme_finalize_ssve_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *vectors,
                                                              nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                                              nk_f32_t *result, nk_size_t result_stride,
                                                              nk_size_t rows_begin, nk_size_t rows_end,
                                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e4m3_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_mxfp8e4m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_euclideans_symmetric_mxfp8e4m3_sme_finalize_ssve_(operand, stride, vector_count, depth, result, result_stride,
                                                         rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

/** Turns packed MXFP8 E5M2 relative dots into angular distances. */
NUMKONG_OUTLINED_ void nk_angulars_packed_mxfp8e5m2_sme_finalize_ssve_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                       void const *b_packed, nk_f32_t *c,
                                                                       nk_size_t rows, nk_size_t columns,
                                                                       nk_size_t depth,
                                                                       nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_sme_columns_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(a.tensor_scale), &tensor_exponent) * b.mantissa;
    nk_dots_scaled_exact_mxfp8e5m2_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp8e5m2_sme_(a, a_stride, row, depth, &exponent);
        nk_angulars_from_relative_ssve_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, norm, b.norms);
    }
}

/** Turns symmetric MXFP8 E5M2 relative dots into angular distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_angulars_symmetric_mxfp8e5m2_sme_finalize_ssve_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                          nk_size_t vector_count, nk_size_t depth,
                                                                          nk_f32_t *result, nk_size_t result_stride,
                                                                          nk_size_t rows_begin,
                                                                          nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(vectors.tensor_scale), &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_sme_mx_base_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp8e5m2_sme_(vectors, stride, column, depth,
                                                                      &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp8e5m2_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp8e5m2_sme_(vectors, stride, row, depth, &exponent);
            nk_angulars_from_relative_ssve_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                            chunk_end - first, mantissa * mantissa, norm, norms + (first - chunk));
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

/** Turns packed MXFP8 E5M2 relative dots into Euclidean distances. */
NUMKONG_OUTLINED_ void nk_euclideans_packed_mxfp8e5m2_sme_finalize_ssve_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                         void const *b_packed, nk_f32_t *c,
                                                                         nk_size_t rows, nk_size_t columns,
                                                                         nk_size_t depth,
                                                                         nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_sme_columns_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(a.tensor_scale), &tensor_exponent) * b.mantissa;
    nk_dots_scaled_exact_mxfp8e5m2_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp8e5m2_sme_(a, a_stride, row, depth, &exponent);
        nk_euclideans_from_relative_ssve_((nk_f32_t *)((char *)c + row * c_stride), columns, mantissa, norm, exponent,
                                          b.norms, b.exponents);
    }
}

/** Turns symmetric MXFP8 E5M2 relative dots into Euclidean distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_euclideans_symmetric_mxfp8e5m2_sme_finalize_ssve_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_f32_split_(nk_cross_tensor_scale_(vectors.tensor_scale), &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_sme_mx_base_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp8e5m2_sme_(vectors, stride, column, depth,
                                                                      &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp8e5m2_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp8e5m2_sme_(vectors, stride, row, depth, &exponent);
            nk_euclideans_from_relative_ssve_((nk_f32_t *)((char *)result + row * result_stride) + first,
                                              chunk_end - first, mantissa * mantissa, norm, exponent,
                                              norms + (first - chunk), exponents + (first - chunk));
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e5m2_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_dots_packed_mxfp8e5m2_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_mxfp8e5m2_sme_finalize_ssve_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e5m2_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_mxfp8e5m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_angulars_symmetric_mxfp8e5m2_sme_finalize_ssve_(operand, stride, vector_count, depth, result, result_stride,
                                                       rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_sme_check_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e5m2_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_dots_packed_mxfp8e5m2_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_mxfp8e5m2_sme_finalize_ssve_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *vectors,
                                                              nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                                              nk_f32_t *result, nk_size_t result_stride,
                                                              nk_size_t rows_begin, nk_size_t rows_end,
                                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e5m2_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_dots_symmetric_mxfp8e5m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_euclideans_symmetric_mxfp8e5m2_sme_finalize_ssve_(operand, stride, vector_count, depth, result, result_stride,
                                                         rows_begin, rows_end);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
#endif // NUMKONG_TARGET_SME

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_SME_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SPATIALS_SME_H
