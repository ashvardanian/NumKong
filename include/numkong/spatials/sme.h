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

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_f16_sme_streaming_(nk_f16_t const *data,
                                                                nk_size_t count) NUMKONG_STREAMING_ {
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

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_bf16_sme_streaming_(nk_bf16_t const *data,
                                                                 nk_size_t count) NUMKONG_STREAMING_ {
    svfloat32_t accumulator_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b16x = svwhilelt_b16_u64(i, count);
        svbfloat16_t values_bf16x = svld1_bf16(predicate_b16x, (nk_bf16_for_arm_simd_t const *)(data + i));
        accumulator_f32x = svbfdot_f32(accumulator_f32x, values_bf16x, values_bf16x);
    }
    return nk_svaddv_f32_(svptrue_b32(), accumulator_f32x);
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e4m3_sme_streaming_(nk_e4m3_t const *data,
                                                                 nk_size_t count) NUMKONG_STREAMING_ {
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
        svfloat16_t values_f16x = nk_e4m3x_to_f16x_sme_streaming_(predicate_b16x, raw_u8x);

        svfloat32_t values_even_f32x = svcvt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_even_f32x = svmla_f32_m(widened_b32x, accumulator_even_f32x, values_even_f32x, values_even_f32x);

        svfloat32_t values_odd_f32x = svcvtlt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_odd_f32x = svmla_f32_m(widened_b32x, accumulator_odd_f32x, values_odd_f32x, values_odd_f32x);
    }
    return nk_svaddv_f32_(svptrue_b32(), accumulator_even_f32x) + nk_svaddv_f32_(svptrue_b32(), accumulator_odd_f32x);
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e5m2_sme_streaming_(nk_e5m2_t const *data,
                                                                 nk_size_t count) NUMKONG_STREAMING_ {
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
        svfloat16_t values_f16x = nk_e5m2x_to_f16x_sme_streaming_(predicate_b16x, raw_u8x);

        svfloat32_t values_even_f32x = svcvt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_even_f32x = svmla_f32_m(widened_b32x, accumulator_even_f32x, values_even_f32x, values_even_f32x);

        svfloat32_t values_odd_f32x = svcvtlt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_odd_f32x = svmla_f32_m(widened_b32x, accumulator_odd_f32x, values_odd_f32x, values_odd_f32x);
    }
    return nk_svaddv_f32_(svptrue_b32(), accumulator_even_f32x) + nk_svaddv_f32_(svptrue_b32(), accumulator_odd_f32x);
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m3_sme_streaming_(nk_e2m3_t const *data,
                                                                 nk_size_t count) NUMKONG_STREAMING_ {
    svint32_t accumulator_i32x = svdup_s32(0);
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, count);
        svuint8_t raw_u8x = svld1_u8(predicate_b8x, (nk_u8_t const *)data + i);
        svint8_t values_i8x = nk_e2m3x_to_i8x_sme_streaming_(predicate_b8x, raw_u8x);
        accumulator_i32x = svdot_s32(accumulator_i32x, values_i8x, values_i8x);
    }
    return (nk_f32_t)nk_svaddv_s32_(svptrue_b32(), accumulator_i32x) / 256.0f;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e2m1_sme_streaming_(nk_e2m1x2_t const *data,
                                                                 nk_size_t count) NUMKONG_STREAMING_ {
    svint32_t accumulator_i32x = svdup_s32(0);
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svint8_t values_i8x = nk_e2m1x_to_i8x_sme_streaming_(data + i / 2, count - i);
        accumulator_i32x = svdot_s32(accumulator_i32x, values_i8x, values_i8x);
    }
    return (nk_f32_t)nk_svaddv_s32_(svptrue_b32(), accumulator_i32x) * 0.25f;
}

NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_e3m2_sme_streaming_(nk_e3m2_t const *data,
                                                                 nk_size_t count) NUMKONG_STREAMING_ {
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
        svfloat16_t values_f16x = nk_e3m2x_to_f16x_sme_streaming_(predicate_b16x, raw_u8x);

        svfloat32_t values_even_f32x = svcvt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_even_f32x = svmla_f32_m(widened_b32x, accumulator_even_f32x, values_even_f32x, values_even_f32x);

        svfloat32_t values_odd_f32x = svcvtlt_f32_f16_x(widened_b32x, values_f16x);
        accumulator_odd_f32x = svmla_f32_m(widened_b32x, accumulator_odd_f32x, values_odd_f32x, values_odd_f32x);
    }
    return nk_svaddv_f32_(svptrue_b32(), accumulator_even_f32x) + nk_svaddv_f32_(svptrue_b32(), accumulator_odd_f32x);
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_i8_sme_streaming_(nk_i8_t const *data,
                                                               nk_size_t count) NUMKONG_STREAMING_ {
    svint32_t accumulator_i32x = svdup_s32(0);
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, count);
        svint8_t loaded_i8x = svld1_s8(predicate_b8x, data + i);
        accumulator_i32x = svdot_s32(accumulator_i32x, loaded_i8x, loaded_i8x);
    }
    return (nk_u32_t)nk_svaddv_s32_(svptrue_b32(), accumulator_i32x);
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_u8_sme_streaming_(nk_u8_t const *data,
                                                               nk_size_t count) NUMKONG_STREAMING_ {
    svuint32_t accumulator_u32x = svdup_u32(0);
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, count);
        svuint8_t loaded_u8x = svld1_u8(predicate_b8x, data + i);
        accumulator_u32x = svdot_u32(accumulator_u32x, loaded_u8x, loaded_u8x);
    }
    return (nk_u32_t)nk_svaddv_u32_(svptrue_b32(), accumulator_u32x);
}

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_i4_sme_streaming_(nk_i4x2_t const *data,
                                                               nk_size_t count) NUMKONG_STREAMING_ {
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

NUMKONG_INLINE nk_u32_t nk_dots_reduce_sumsq_u4_sme_streaming_(nk_u4x2_t const *data,
                                                               nk_size_t count) NUMKONG_STREAMING_ {
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

/** @p numerator over the square roots of squared norms, 0 for zero norms. Streaming roots and
 *  divisions cost about 36 cycles a vector on M5, so finalizers take them once per row and
 *  column, and block-scaled rows fold their tensor mantissa in as the @p numerator. */
NUMKONG_INLINE svfloat32_t nk_rsqrt_f32x_sme_streaming_(svbool_t predicate_b32x, svfloat32_t norms_f32x,
                                                        nk_f32_t numerator) NUMKONG_STREAMING_ {
    svfloat32_t const rsqrts_f32x = svdiv_f32_x(predicate_b32x, svdup_f32(numerator),
                                                svsqrt_f32_x(predicate_b32x, norms_f32x));
    return svsel_f32(svcmpeq_n_f32(predicate_b32x, norms_f32x, 0), svdup_f32(0), rsqrts_f32x);
}

/** @c nk_rsqrt_f32x_sme_streaming_ of @p count squared norms into @p rsqrts. */
NUMKONG_INLINE void nk_rsqrts_f32_sme_streaming_(nk_f32_t const *norms, nk_f32_t numerator, nk_f32_t *rsqrts,
                                                 nk_size_t count) NUMKONG_STREAMING_ {
    for (nk_size_t index = 0; index < count; index += svcntw()) {
        svbool_t const predicate_b32x = svwhilelt_b32_u64(index, count);
        svst1_f32(predicate_b32x, rsqrts + index,
                  nk_rsqrt_f32x_sme_streaming_(predicate_b32x, svld1_f32(predicate_b32x, norms + index), numerator));
    }
}

/*  Row finishers rewrite the dots in [first, end) of a vector-aligned @p row in place, reading
 *  target norms at the same columns. A row starting mid-vector masks its first vector rather than
 *  store across cache lines, which made symmetric finishing 1.6x slower. */

/** The lanes of the vector at @p column from @p first up to @p end. */
NUMKONG_INLINE svbool_t nk_row_lanes_b32x_sme_streaming_(nk_size_t column, nk_size_t first,
                                                         nk_size_t end) NUMKONG_STREAMING_ {
    svbool_t const lanes_b32x = svwhilelt_b32_u64(column, end);
    return column < first ? nk_diagonal_cut_b32x_sme_(lanes_b32x, column, first) : lanes_b32x;
}

/** Angular distances from @c nk_rsqrts_f32_sme_streaming_ roots: two zero norms give 0, one zero
 *  norm or an exactly zero dot 1, a NaN dot NaN, and otherwise max(0, 1 − dot · rsqrt(a²) ·
 *  rsqrt(b²)). A zero column's dot and root are both zero, so it lands on 1 without a compare. A
 *  zero row, found by its bits as scalar FP compares cost about 30 ns in streaming mode, takes its
 *  own pass. The row's root comes in a register, as reloading a streaming store's data stalls. */
NUMKONG_INLINE void nk_angulars_from_rsqrts_sme_streaming_(nk_f32_t *row, nk_size_t first, nk_size_t end,
                                                           nk_f32_t query_norm, svfloat32_t query_rsqrt_f32x,
                                                           nk_f32_t const *target_norms,
                                                           nk_f32_t const *target_rsqrts) NUMKONG_STREAMING_ {
    nk_size_t const aligned_first = first & ~(svcntw() - 1);
    nk_fui32_t query_bits;
    query_bits.f = query_norm;
    if (query_bits.u == 0) {
        for (nk_size_t column = aligned_first; column < end; column += svcntw()) {
            svbool_t const predicate_b32x = nk_row_lanes_b32x_sme_streaming_(column, first, end);
            svfloat32_t const dots_f32x = svld1_f32(predicate_b32x, row + column);
            svfloat32_t const ruled_f32x = svsel_f32(
                svcmpeq_n_f32(predicate_b32x, svld1_f32(predicate_b32x, target_norms + column), 0), svdup_f32(0),
                svdup_f32(1));
            svst1_f32(predicate_b32x, row + column,
                      svsel_f32(svcmpuo_f32(predicate_b32x, dots_f32x, dots_f32x), dots_f32x, ruled_f32x));
        }
        return;
    }
    for (nk_size_t column = aligned_first; column < end; column += svcntw()) {
        svbool_t const predicate_b32x = nk_row_lanes_b32x_sme_streaming_(column, first, end);
        svfloat32_t const rsqrt_f32x = svmul_f32_x(predicate_b32x, query_rsqrt_f32x,
                                                   svld1_f32(predicate_b32x, target_rsqrts + column));
        svfloat32_t const scaled_f32x = svmul_f32_x(predicate_b32x, svld1_f32(predicate_b32x, row + column),
                                                    rsqrt_f32x);
        // FMAX keeps a NaN dot, which FMAXNM would turn into 0
        svst1_f32(predicate_b32x, row + column,
                  svmax_n_f32_x(predicate_b32x, svsubr_n_f32_x(predicate_b32x, scaled_f32x, 1), 0));
    }
}

NUMKONG_INLINE svfloat32_t nk_euclideans_from_dot_f32x_sme_streaming_(
    svbool_t predicate_b32x, svfloat32_t dots_f32x, svfloat32_t query_norm_sq_f32x,
    svfloat32_t target_norms_sq_f32x) NUMKONG_STREAMING_ {
    svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_b32x, query_norm_sq_f32x, target_norms_sq_f32x);
    svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_b32x, sum_sq_f32x,
                                           svmul_f32_x(predicate_b32x, svdup_n_f32(2.0f), dots_f32x));
    // FMAX keeps a NaN, which FMAXNM would replace with zero
    dist_sq_f32x = svmax_n_f32_x(predicate_b32x, dist_sq_f32x, 0);
    return svsqrt_f32_x(predicate_b32x, dist_sq_f32x);
}

/** Euclidean distances √max(0, a² + b² − 2 · dot) against the squared query norm in every lane of
 *  @p query_norm_f32x; FSQRT throughput bounds them at about 36 cycles a vector. */
NUMKONG_INLINE void nk_euclideans_from_norms_sme_streaming_(nk_f32_t *row, nk_size_t first, nk_size_t end,
                                                            svfloat32_t query_norm_f32x,
                                                            nk_f32_t const *target_norms) NUMKONG_STREAMING_ {
    for (nk_size_t column = first & ~(svcntw() - 1); column < end; column += svcntw()) {
        svbool_t const predicate_b32x = nk_row_lanes_b32x_sme_streaming_(column, first, end);
        svst1_f32(predicate_b32x, row + column,
                  nk_euclideans_from_dot_f32x_sme_streaming_(predicate_b32x, svld1_f32(predicate_b32x, row + column),
                                                             query_norm_f32x,
                                                             svld1_f32(predicate_b32x, target_norms + column)));
    }
}

/** Square roots of @p count u32 norms in F32. */
NUMKONG_INLINE void nk_roots_u32_sme_streaming_(nk_u32_t const *norms, nk_f32_t *roots,
                                                nk_size_t count) NUMKONG_STREAMING_ {
    for (nk_size_t index = 0; index < count; index += svcntw()) {
        svbool_t const predicate_b32x = svwhilelt_b32_u64(index, count);
        svst1_f32(
            predicate_b32x, roots + index,
            svsqrt_f32_x(predicate_b32x, svcvt_f32_u32_x(predicate_b32x, svld1_u32(predicate_b32x, norms + index))));
    }
}

/** 1 for each of @p count zero u32 norms and 0 for the rest: FMAXNM against these floors turns a
 *  zero column's 0 / 0 into its angle of 1, and one minus them gives a zero row's angles. */
NUMKONG_INLINE void nk_floors_u32_sme_streaming_(nk_u32_t const *norms, nk_f32_t *floors,
                                                 nk_size_t count) NUMKONG_STREAMING_ {
    for (nk_size_t index = 0; index < count; index += svcntw()) {
        svbool_t const predicate_b32x = svwhilelt_b32_u64(index, count);
        svst1_f32(predicate_b32x, floors + index,
                  svsel_f32(svcmpeq_n_u32(predicate_b32x, svld1_u32(predicate_b32x, norms + index), 0), svdup_f32(1),
                            svdup_f32(0)));
    }
}

/** F32 of the exact U64 values of the even and odd 32-bit lanes, back in those lanes. */
NUMKONG_INLINE svfloat32_t nk_f32x_from_u64x2_sme_streaming_(svuint64_t even_u64x,
                                                             svuint64_t odd_u64x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b64x = svptrue_b64();
    return svtrn1_f32(svcvt_f32_u64_x(predicate_all_b64x, even_u64x), svcvt_f32_u64_x(predicate_all_b64x, odd_u64x));
}

/** F32 of the exact I64 values of the even and odd 32-bit lanes, back in those lanes. */
NUMKONG_INLINE svfloat32_t nk_f32x_from_i64x2_sme_streaming_(svint64_t even_i64x,
                                                             svint64_t odd_i64x) NUMKONG_STREAMING_ {
    svbool_t const predicate_all_b64x = svptrue_b64();
    return svtrn1_f32(svcvt_f32_s64_x(predicate_all_b64x, even_i64x), svcvt_f32_s64_x(predicate_all_b64x, odd_i64x));
}

/** Angular distances from i32 dots d, the u32 query norm a with its F32 root, and u32 target norms
 *  b with their roots and @c nk_floors_u32_sme_streaming_ floors. With s = √a · √b, a positive dot
 *  takes 1 − d / s as (ab − d²) / (ab + d · s), whose numerator is exact in 64-bit lanes, so equal
 *  vectors give exactly 0 and near-parallel ones keep their angle; any other dot takes (ab − d · s)
 *  / ab, as the NEON helpers do. */
NUMKONG_INLINE void nk_angulars_from_i32_sme_streaming_(nk_f32_t *row, nk_size_t first, nk_size_t end,
                                                        nk_u32_t query_norm, svfloat32_t query_root_f32x,
                                                        nk_u32_t const *target_norms, nk_f32_t const *target_roots,
                                                        nk_f32_t const *target_floors) NUMKONG_STREAMING_ {
    nk_size_t const aligned_first = first & ~(svcntw() - 1);
    if (query_norm == 0) {
        for (nk_size_t column = aligned_first; column < end; column += svcntw()) {
            svbool_t const predicate_b32x = nk_row_lanes_b32x_sme_streaming_(column, first, end);
            svst1_f32(predicate_b32x, row + column,
                      svsubr_n_f32_x(predicate_b32x, svld1_f32(predicate_b32x, target_floors + column), 1));
        }
        return;
    }
    svbool_t const predicate_all_b64x = svptrue_b64();
    for (nk_size_t column = aligned_first; column < end; column += svcntw()) {
        svbool_t const predicate_b32x = nk_row_lanes_b32x_sme_streaming_(column, first, end);
        svint32_t const dots_i32x = svld1_s32(predicate_b32x, (nk_i32_t const *)row + column);
        svuint32_t const targets_u32x = svld1_u32(predicate_b32x, target_norms + column);
        svuint64_t const products_even_u64x = svmullb_n_u64(targets_u32x, query_norm);
        svuint64_t const products_odd_u64x = svmullt_n_u64(targets_u32x, query_norm);
        svfloat32_t const gaps_f32x = nk_f32x_from_u64x2_sme_streaming_(
            svsub_u64_x(predicate_all_b64x, products_even_u64x,
                        svreinterpret_u64_s64(svmullb_s64(dots_i32x, dots_i32x))),
            svsub_u64_x(predicate_all_b64x, products_odd_u64x,
                        svreinterpret_u64_s64(svmullt_s64(dots_i32x, dots_i32x))));
        svfloat32_t const products_f32x = nk_f32x_from_u64x2_sme_streaming_(products_even_u64x, products_odd_u64x);
        svfloat32_t const dots_f32x = svcvt_f32_s32_x(predicate_b32x, dots_i32x);
        svfloat32_t const roots_f32x = svmul_f32_x(predicate_b32x, svld1_f32(predicate_b32x, target_roots + column),
                                                   query_root_f32x);
        svfloat32_t const numerators_f32x = svsel_f32(
            svcmpgt_n_s32(predicate_b32x, dots_i32x, 0), gaps_f32x,
            svmls_f32_x(predicate_b32x, products_f32x, dots_f32x, roots_f32x));
        svfloat32_t const denominators_f32x = svmla_f32_x(predicate_b32x, products_f32x,
                                                          svmax_n_f32_x(predicate_b32x, dots_f32x, 0), roots_f32x);
        svst1_f32(predicate_b32x, row + column,
                  svmaxnm_f32_x(predicate_b32x, svdiv_f32_x(predicate_b32x, numerators_f32x, denominators_f32x),
                                svld1_f32(predicate_b32x, target_floors + column)));
    }
}

/** Angular distances from u32 dots, as @c nk_angulars_from_i32_sme_streaming_ takes positive ones;
 *  a zero dot gives (ab − 0) / ab, exactly 1. */
NUMKONG_INLINE void nk_angulars_from_u32_sme_streaming_(nk_f32_t *row, nk_size_t first, nk_size_t end,
                                                        nk_u32_t query_norm, svfloat32_t query_root_f32x,
                                                        nk_u32_t const *target_norms, nk_f32_t const *target_roots,
                                                        nk_f32_t const *target_floors) NUMKONG_STREAMING_ {
    nk_size_t const aligned_first = first & ~(svcntw() - 1);
    if (query_norm == 0) {
        for (nk_size_t column = aligned_first; column < end; column += svcntw()) {
            svbool_t const predicate_b32x = nk_row_lanes_b32x_sme_streaming_(column, first, end);
            svst1_f32(predicate_b32x, row + column,
                      svsubr_n_f32_x(predicate_b32x, svld1_f32(predicate_b32x, target_floors + column), 1));
        }
        return;
    }
    svbool_t const predicate_all_b64x = svptrue_b64();
    for (nk_size_t column = aligned_first; column < end; column += svcntw()) {
        svbool_t const predicate_b32x = nk_row_lanes_b32x_sme_streaming_(column, first, end);
        svuint32_t const dots_u32x = svld1_u32(predicate_b32x, (nk_u32_t const *)row + column);
        svuint32_t const targets_u32x = svld1_u32(predicate_b32x, target_norms + column);
        svuint64_t const products_even_u64x = svmullb_n_u64(targets_u32x, query_norm);
        svuint64_t const products_odd_u64x = svmullt_n_u64(targets_u32x, query_norm);
        svfloat32_t const gaps_f32x = nk_f32x_from_u64x2_sme_streaming_(
            svsub_u64_x(predicate_all_b64x, products_even_u64x, svmullb_u64(dots_u32x, dots_u32x)),
            svsub_u64_x(predicate_all_b64x, products_odd_u64x, svmullt_u64(dots_u32x, dots_u32x)));
        svfloat32_t const products_f32x = nk_f32x_from_u64x2_sme_streaming_(products_even_u64x, products_odd_u64x);
        svfloat32_t const roots_f32x = svmul_f32_x(predicate_b32x, svld1_f32(predicate_b32x, target_roots + column),
                                                   query_root_f32x);
        svfloat32_t const denominators_f32x = svmla_f32_x(predicate_b32x, products_f32x,
                                                          svcvt_f32_u32_x(predicate_b32x, dots_u32x), roots_f32x);
        svst1_f32(predicate_b32x, row + column,
                  svmaxnm_f32_x(predicate_b32x, svdiv_f32_x(predicate_b32x, gaps_f32x, denominators_f32x),
                                svld1_f32(predicate_b32x, target_floors + column)));
    }
}

/** Euclidean distances from i32 dots d and u32 norms a, b: a + b − 2d is exact in 64-bit lanes
 *  and rounds once into F32 before the root. */
NUMKONG_INLINE void nk_euclideans_from_i32_sme_streaming_(nk_f32_t *row, nk_size_t first, nk_size_t end,
                                                          nk_u32_t query_norm,
                                                          nk_u32_t const *target_norms) NUMKONG_STREAMING_ {
    for (nk_size_t column = first & ~(svcntw() - 1); column < end; column += svcntw()) {
        svbool_t const predicate_b32x = nk_row_lanes_b32x_sme_streaming_(column, first, end);
        svint32_t const dots_i32x = svld1_s32(predicate_b32x, (nk_i32_t const *)row + column);
        svuint32_t const targets_u32x = svld1_u32(predicate_b32x, target_norms + column);
        svfloat32_t const squares_f32x = nk_f32x_from_i64x2_sme_streaming_(
            svmlslb_n_s64(svreinterpret_s64_u64(svaddlb_n_u64(targets_u32x, query_norm)), dots_i32x, 2),
            svmlslt_n_s64(svreinterpret_s64_u64(svaddlt_n_u64(targets_u32x, query_norm)), dots_i32x, 2));
        svst1_f32(predicate_b32x, row + column, svsqrt_f32_x(predicate_b32x, squares_f32x));
    }
}

/** Euclidean distances from u32 dots, as @c nk_euclideans_from_i32_sme_streaming_ forms them. */
NUMKONG_INLINE void nk_euclideans_from_u32_sme_streaming_(nk_f32_t *row, nk_size_t first, nk_size_t end,
                                                          nk_u32_t query_norm,
                                                          nk_u32_t const *target_norms) NUMKONG_STREAMING_ {
    for (nk_size_t column = first & ~(svcntw() - 1); column < end; column += svcntw()) {
        svbool_t const predicate_b32x = nk_row_lanes_b32x_sme_streaming_(column, first, end);
        svuint32_t const dots_u32x = svld1_u32(predicate_b32x, (nk_u32_t const *)row + column);
        svuint32_t const targets_u32x = svld1_u32(predicate_b32x, target_norms + column);
        svfloat32_t const squares_f32x = nk_f32x_from_u64x2_sme_streaming_(
            svmlslb_n_u64(svaddlb_n_u64(targets_u32x, query_norm), dots_u32x, 2),
            svmlslt_n_u64(svaddlt_n_u64(targets_u32x, query_norm), dots_u32x, 2));
        svst1_f32(predicate_b32x, row + column, svsqrt_f32_x(predicate_b32x, squares_f32x));
    }
}

/** Euclidean distances from relative dots times @p mantissa, the row's squared norm being
 *  @p row_norm times 4 to the @p row_exponent and each column's its norm times 4^E; every term
 *  scales to the larger exponent before the root, which scales back once. */
NUMKONG_INLINE void nk_euclideans_from_relative_sme_streaming_(nk_f32_t *row, nk_size_t first, nk_size_t end,
                                                               nk_f32_t mantissa, nk_f32_t row_norm,
                                                               nk_i32_t row_exponent, nk_f32_t const *column_norms,
                                                               nk_i32_t const *column_exponents) NUMKONG_STREAMING_ {
    for (nk_size_t column = first & ~(svcntw() - 1); column < end; column += svcntw()) {
        svbool_t const predicate_b32x = nk_row_lanes_b32x_sme_streaming_(column, first, end);
        svint32_t const exponents_i32x = svld1_s32(predicate_b32x, column_exponents + column);
        svint32_t const top_i32x = svmax_n_s32_x(predicate_b32x, exponents_i32x, row_exponent);
        svfloat32_t const row_f32x = svscale_f32_x(
            predicate_b32x, svdup_n_f32(row_norm),
            svlsl_n_s32_x(predicate_b32x, svsubr_n_s32_x(predicate_b32x, top_i32x, row_exponent), 1));
        svfloat32_t const column_f32x = svscale_f32_x(
            predicate_b32x, svld1_f32(predicate_b32x, column_norms + column),
            svlsl_n_s32_x(predicate_b32x, svsub_s32_x(predicate_b32x, exponents_i32x, top_i32x), 1));
        svfloat32_t const dots_f32x = svscale_f32_x(
            predicate_b32x, svmul_n_f32_x(predicate_b32x, svld1_f32(predicate_b32x, row + column), 2 * mantissa),
            svsub_s32_x(predicate_b32x, svadd_n_s32_x(predicate_b32x, exponents_i32x, row_exponent),
                        svlsl_n_s32_x(predicate_b32x, top_i32x, 1)));
        // FMAX keeps a NaN, which FMAXNM would replace with zero
        svfloat32_t const squares_f32x = svmax_n_f32_x(
            predicate_b32x, svsub_f32_x(predicate_b32x, svadd_f32_x(predicate_b32x, row_f32x, column_f32x), dots_f32x),
            0);
        svst1_f32(predicate_b32x, row + column,
                  svscale_f32_x(predicate_b32x, svsqrt_f32_x(predicate_b32x, squares_f32x), top_i32x));
    }
}

#pragma region F16 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_f16_sme_streaming_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row)
            row_norms[row] = nk_dots_reduce_sumsq_f16_sme_streaming_(a + (row_first + row) * a_stride_elements, depth);
        nk_rsqrts_f32_sme_streaming_(row_norms, 1, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b_norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(c + (row_first + row) * c_stride_elements + column_first, 0,
                                                       block_columns, row_norms[row], svdup_n_f32(row_rsqrts[row]),
                                                       b_norms + column_first, column_rsqrts);
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
    nk_start_sme_streaming_();
    nk_dots_packed_f16_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_f16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                   c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_f16_sme_streaming_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_f16_sme_streaming_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_euclideans_from_dot_f32x_sme_streaming_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                                 target_norms_sq_f32x));
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
    nk_start_sme_streaming_();
    nk_dots_packed_f16_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_f16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                     c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_f16_sme_streaming_( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_f16_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        nk_align_(64) nk_f32_t rsqrts_cache[256];
        nk_rsqrts_f32_sme_streaming_(norms_cache, 1, rsqrts_cache, chunk_end - chunk_start);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_angulars_from_rsqrts_sme_streaming_(
                result_row + chunk_start, first - chunk_start, chunk_end - chunk_start, result_row[row_index],
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(result_row[row_index]), 1), norms_cache,
                rsqrts_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_f16_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                         rows_end);
    nk_angulars_symmetric_finalize_f16_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                      result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_f16_sme_streaming_( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_f16_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_euclideans_from_norms_sme_streaming_(result_row + chunk_start, first - chunk_start,
                                                    chunk_end - chunk_start, svdup_n_f32(result_row[row_index]),
                                                    norms_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_f16_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                         rows_end);
    nk_euclideans_symmetric_finalize_f16_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                        result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion F16 Floats

#pragma region BF16 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_bf16_sme_streaming_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row)
            row_norms[row] = nk_dots_reduce_sumsq_bf16_sme_streaming_(a + (row_first + row) * a_stride_elements, depth);
        nk_rsqrts_f32_sme_streaming_(row_norms, 1, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b_norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(c + (row_first + row) * c_stride_elements + column_first, 0,
                                                       block_columns, row_norms[row], svdup_n_f32(row_rsqrts[row]),
                                                       b_norms + column_first, column_rsqrts);
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
    nk_start_sme_streaming_();
    nk_dots_packed_bf16_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_bf16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_bf16_sme_streaming_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_bf16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_bf16_sme_streaming_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_euclideans_from_dot_f32x_sme_streaming_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                                 target_norms_sq_f32x));
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
    nk_start_sme_streaming_();
    nk_dots_packed_bf16_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_bf16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_bf16_sme_streaming_( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_bf16_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        nk_align_(64) nk_f32_t rsqrts_cache[256];
        nk_rsqrts_f32_sme_streaming_(norms_cache, 1, rsqrts_cache, chunk_end - chunk_start);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_angulars_from_rsqrts_sme_streaming_(
                result_row + chunk_start, first - chunk_start, chunk_end - chunk_start, result_row[row_index],
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(result_row[row_index]), 1), norms_cache,
                rsqrts_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_bf16_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_angulars_symmetric_finalize_bf16_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                       result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_bf16_sme_streaming_( //
    nk_bf16_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_bf16_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_euclideans_from_norms_sme_streaming_(result_row + chunk_start, first - chunk_start,
                                                    chunk_end - chunk_start, svdup_n_f32(result_row[row_index]),
                                                    norms_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_bf16_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_euclideans_symmetric_finalize_bf16_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                         result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion BF16 Floats

#pragma region E4M3 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_e4m3_sme_streaming_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row)
            row_norms[row] = nk_dots_reduce_sumsq_e4m3_sme_streaming_(a + (row_first + row) * a_stride_elements, depth);
        nk_rsqrts_f32_sme_streaming_(row_norms, 1, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b_norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(c + (row_first + row) * c_stride_elements + column_first, 0,
                                                       block_columns, row_norms[row], svdup_n_f32(row_rsqrts[row]),
                                                       b_norms + column_first, column_rsqrts);
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
    nk_start_sme_streaming_();
    nk_dots_packed_e4m3_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_e4m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_e4m3_sme_streaming_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e4m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e4m3_sme_streaming_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_euclideans_from_dot_f32x_sme_streaming_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                                 target_norms_sq_f32x));
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
    nk_start_sme_streaming_();
    nk_dots_packed_e4m3_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_e4m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_e4m3_sme_streaming_( //
    nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_e4m3_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        nk_align_(64) nk_f32_t rsqrts_cache[256];
        nk_rsqrts_f32_sme_streaming_(norms_cache, 1, rsqrts_cache, chunk_end - chunk_start);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_angulars_from_rsqrts_sme_streaming_(
                result_row + chunk_start, first - chunk_start, chunk_end - chunk_start, result_row[row_index],
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(result_row[row_index]), 1), norms_cache,
                rsqrts_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_e4m3_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_angulars_symmetric_finalize_e4m3_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                       result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_e4m3_sme_streaming_( //
    nk_e4m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_e4m3_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_euclideans_from_norms_sme_streaming_(result_row + chunk_start, first - chunk_start,
                                                    chunk_end - chunk_start, svdup_n_f32(result_row[row_index]),
                                                    norms_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_e4m3_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_euclideans_symmetric_finalize_e4m3_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                         result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion E4M3 Floats

#pragma region E5M2 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_e5m2_sme_streaming_( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row)
            row_norms[row] = nk_dots_reduce_sumsq_e5m2_sme_streaming_(a + (row_first + row) * a_stride_elements, depth);
        nk_rsqrts_f32_sme_streaming_(row_norms, 1, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b_norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(c + (row_first + row) * c_stride_elements + column_first, 0,
                                                       block_columns, row_norms[row], svdup_n_f32(row_rsqrts[row]),
                                                       b_norms + column_first, column_rsqrts);
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
    nk_start_sme_streaming_();
    nk_dots_packed_e5m2_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_e5m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_e5m2_sme_streaming_( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e5m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e5m2_sme_streaming_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_euclideans_from_dot_f32x_sme_streaming_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                                 target_norms_sq_f32x));
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
    nk_start_sme_streaming_();
    nk_dots_packed_e5m2_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_e5m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_e5m2_sme_streaming_( //
    nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_e5m2_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        nk_align_(64) nk_f32_t rsqrts_cache[256];
        nk_rsqrts_f32_sme_streaming_(norms_cache, 1, rsqrts_cache, chunk_end - chunk_start);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_angulars_from_rsqrts_sme_streaming_(
                result_row + chunk_start, first - chunk_start, chunk_end - chunk_start, result_row[row_index],
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(result_row[row_index]), 1), norms_cache,
                rsqrts_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_e5m2_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_angulars_symmetric_finalize_e5m2_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                       result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_e5m2_sme_streaming_( //
    nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_e5m2_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_euclideans_from_norms_sme_streaming_(result_row + chunk_start, first - chunk_start,
                                                    chunk_end - chunk_start, svdup_n_f32(result_row[row_index]),
                                                    norms_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_e5m2_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_euclideans_symmetric_finalize_e5m2_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                         result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion E5M2 Floats

#pragma region E2M3 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_e2m3_sme_streaming_( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row)
            row_norms[row] = nk_dots_reduce_sumsq_e2m3_sme_streaming_(a + (row_first + row) * a_stride_elements, depth);
        nk_rsqrts_f32_sme_streaming_(row_norms, 1, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b_norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(c + (row_first + row) * c_stride_elements + column_first, 0,
                                                       block_columns, row_norms[row], svdup_n_f32(row_rsqrts[row]),
                                                       b_norms + column_first, column_rsqrts);
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
    nk_start_sme_streaming_();
    nk_dots_packed_e2m3_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_e2m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_e2m3_sme_streaming_( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m3_sme_streaming_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_euclideans_from_dot_f32x_sme_streaming_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                                 target_norms_sq_f32x));
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
    nk_start_sme_streaming_();
    nk_dots_packed_e2m3_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_e2m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_e2m3_sme_streaming_( //
    nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_e2m3_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        nk_align_(64) nk_f32_t rsqrts_cache[256];
        nk_rsqrts_f32_sme_streaming_(norms_cache, 1, rsqrts_cache, chunk_end - chunk_start);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_angulars_from_rsqrts_sme_streaming_(
                result_row + chunk_start, first - chunk_start, chunk_end - chunk_start, result_row[row_index],
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(result_row[row_index]), 1), norms_cache,
                rsqrts_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_e2m3_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_angulars_symmetric_finalize_e2m3_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                       result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_e2m3_sme_streaming_( //
    nk_e2m3_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_e2m3_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_euclideans_from_norms_sme_streaming_(result_row + chunk_start, first - chunk_start,
                                                    chunk_end - chunk_start, svdup_n_f32(result_row[row_index]),
                                                    norms_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_e2m3_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_euclideans_symmetric_finalize_e2m3_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                         result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion E2M3 Floats

#pragma region E2M1 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_e2m1_sme_streaming_( //
    nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row)
            row_norms[row] = nk_dots_reduce_sumsq_e2m1_sme_streaming_(a + (row_first + row) * a_stride_elements, depth);
        nk_rsqrts_f32_sme_streaming_(row_norms, 1, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b_norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(c + (row_first + row) * c_stride_elements + column_first, 0,
                                                       block_columns, row_norms[row], svdup_n_f32(row_rsqrts[row]),
                                                       b_norms + column_first, column_rsqrts);
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
    nk_start_sme_streaming_();
    nk_dots_packed_e2m1_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_e2m1_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_e2m1_sme_streaming_( //
    nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m1x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m1_sme_streaming_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_euclideans_from_dot_f32x_sme_streaming_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                                 target_norms_sq_f32x));
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
    nk_start_sme_streaming_();
    nk_dots_packed_e2m1_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_e2m1_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_e2m1_sme_streaming_( //
    nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m1_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_e2m1_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        nk_align_(64) nk_f32_t rsqrts_cache[256];
        nk_rsqrts_f32_sme_streaming_(norms_cache, 1, rsqrts_cache, chunk_end - chunk_start);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_angulars_from_rsqrts_sme_streaming_(
                result_row + chunk_start, first - chunk_start, chunk_end - chunk_start, result_row[row_index],
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(result_row[row_index]), 1), norms_cache,
                rsqrts_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_e2m1_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_angulars_symmetric_finalize_e2m1_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                       result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_e2m1_sme_streaming_( //
    nk_e2m1x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m1_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_e2m1_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_euclideans_from_norms_sme_streaming_(result_row + chunk_start, first - chunk_start,
                                                    chunk_end - chunk_start, svdup_n_f32(result_row[row_index]),
                                                    norms_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_e2m1_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_euclideans_symmetric_finalize_e2m1_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                         result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion E2M1 Floats

#pragma region E3M2 Floats

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_e3m2_sme_streaming_( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row)
            row_norms[row] = nk_dots_reduce_sumsq_e3m2_sme_streaming_(a + (row_first + row) * a_stride_elements, depth);
        nk_rsqrts_f32_sme_streaming_(row_norms, 1, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b_norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(c + (row_first + row) * c_stride_elements + column_first, 0,
                                                       block_columns, row_norms[row], svdup_n_f32(row_rsqrts[row]),
                                                       b_norms + column_first, column_rsqrts);
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
    nk_start_sme_streaming_();
    nk_dots_packed_e3m2_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_e3m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_e3m2_sme_streaming_( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e3m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e3m2_sme_streaming_(a_row, depth);
        svfloat32_t query_norm_sq_f32x = svdup_n_f32(query_norm_sq_f32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_b32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_b32x, b_norms + col_index);
            svst1_f32(predicate_b32x, result_row + col_index,
                      nk_euclideans_from_dot_f32x_sme_streaming_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                                 target_norms_sq_f32x));
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
    nk_start_sme_streaming_();
    nk_dots_packed_e3m2_sme_streaming_(a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_e3m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_e3m2_sme_streaming_( //
    nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_e3m2_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        nk_align_(64) nk_f32_t rsqrts_cache[256];
        nk_rsqrts_f32_sme_streaming_(norms_cache, 1, rsqrts_cache, chunk_end - chunk_start);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_angulars_from_rsqrts_sme_streaming_(
                result_row + chunk_start, first - chunk_start, chunk_end - chunk_start, result_row[row_index],
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(result_row[row_index]), 1), norms_cache,
                rsqrts_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_e3m2_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_angulars_symmetric_finalize_e3m2_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                       result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_e3m2_sme_streaming_( //
    nk_e3m2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_sme_streaming_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_align_(64) nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? result[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_e3m2_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_euclideans_from_norms_sme_streaming_(result_row + chunk_start, first - chunk_start,
                                                    chunk_end - chunk_start, svdup_n_f32(result_row[row_index]),
                                                    norms_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_e3m2_sme_streaming_(vectors, stride, vector_count, depth, result, result_stride, rows_begin,
                                          rows_end);
    nk_euclideans_symmetric_finalize_e3m2_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                         result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion E3M2 Floats
#pragma region I8 Integers

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_i8_sme_streaming_( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    nk_u32_t row_norms[64];
    nk_align_(64) nk_f32_t row_roots[64], column_roots[256], column_floors[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row)
            row_norms[row] = nk_dots_reduce_sumsq_i8_sme_streaming_(a + (row_first + row) * a_stride_elements, depth);
        nk_roots_u32_sme_streaming_(row_norms, row_roots, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_roots_u32_sme_streaming_(b_norms + column_first, column_roots, block_columns);
            nk_floors_u32_sme_streaming_(b_norms + column_first, column_floors, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_i32_sme_streaming_(c + (row_first + row) * c_stride_elements + column_first, 0,
                                                    block_columns, row_norms[row], svdup_n_f32(row_roots[row]),
                                                    b_norms + column_first, column_roots, column_floors);
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
    nk_start_sme_streaming_();
    nk_dots_packed_i8_sme_streaming_(a, a_stride, b_packed, (nk_i32_t *)c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_i8_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_i8_sme_streaming_( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_i8_sme_streaming_(a_row, depth);
        nk_euclideans_from_i32_sme_streaming_(result_row, 0, columns, query_norm_sq_u32, b_norms);
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_i8_sme( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_start_sme_streaming_();
    nk_dots_packed_i8_sme_streaming_(a, a_stride, b_packed, (nk_i32_t *)c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_i8_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_i8_sme_streaming_( //
    nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i8_sme_streaming_(vectors + row_index * stride_elements, depth);
        nk_u32_t *result_row_norms = (nk_u32_t *)(result + row_index * result_stride_elements);
        result_row_norms[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_align_(64) nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? ((nk_u32_t const *)result)[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_i8_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        nk_align_(64) nk_f32_t roots_cache[256], floors_cache[256];
        nk_roots_u32_sme_streaming_(norms_cache, roots_cache, chunk_end - chunk_start);
        nk_floors_u32_sme_streaming_(norms_cache, floors_cache, chunk_end - chunk_start);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t const query_norm = ((nk_u32_t const *)result_row)[row_index];
            svfloat32_t const query_root_f32x = svsqrt_f32_x(svptrue_b32(),
                                                             svcvt_f32_u32_x(svptrue_b32(), svdup_n_u32(query_norm)));
            nk_angulars_from_i32_sme_streaming_(result_row + chunk_start, first - chunk_start, chunk_end - chunk_start,
                                                query_norm, query_root_f32x, norms_cache, roots_cache, floors_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_i8_sme_streaming_(vectors, stride, vector_count, depth, (nk_i32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_angulars_symmetric_finalize_i8_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                     result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_i8_sme_streaming_( //
    nk_i8_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i8_sme_streaming_(vectors + row_index * stride_elements, depth);
        nk_u32_t *result_row_norms = (nk_u32_t *)(result + row_index * result_stride_elements);
        result_row_norms[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_align_(64) nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? ((nk_u32_t const *)result)[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_i8_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_euclideans_from_i32_sme_streaming_(result_row + chunk_start, first - chunk_start,
                                                  chunk_end - chunk_start, ((nk_u32_t const *)result_row)[row_index],
                                                  norms_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_i8_sme_streaming_(vectors, stride, vector_count, depth, (nk_i32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_euclideans_symmetric_finalize_i8_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                       result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion I8 Integers

#pragma region U8 Integers

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_u8_sme_streaming_( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    nk_u32_t row_norms[64];
    nk_align_(64) nk_f32_t row_roots[64], column_roots[256], column_floors[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row)
            row_norms[row] = nk_dots_reduce_sumsq_u8_sme_streaming_(a + (row_first + row) * a_stride_elements, depth);
        nk_roots_u32_sme_streaming_(row_norms, row_roots, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_roots_u32_sme_streaming_(b_norms + column_first, column_roots, block_columns);
            nk_floors_u32_sme_streaming_(b_norms + column_first, column_floors, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_u32_sme_streaming_(c + (row_first + row) * c_stride_elements + column_first, 0,
                                                    block_columns, row_norms[row], svdup_n_f32(row_roots[row]),
                                                    b_norms + column_first, column_roots, column_floors);
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
    nk_start_sme_streaming_();
    nk_dots_packed_u8_sme_streaming_(a, a_stride, b_packed, (nk_u32_t *)c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_u8_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_u8_sme_streaming_( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_u8_sme_streaming_(a_row, depth);
        nk_euclideans_from_u32_sme_streaming_(result_row, 0, columns, query_norm_sq_u32, b_norms);
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_u8_sme( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_start_sme_streaming_();
    nk_dots_packed_u8_sme_streaming_(a, a_stride, b_packed, (nk_u32_t *)c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_u8_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_u8_sme_streaming_( //
    nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u8_sme_streaming_(vectors + row_index * stride_elements, depth);
        nk_u32_t *result_row_norms = (nk_u32_t *)(result + row_index * result_stride_elements);
        result_row_norms[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_align_(64) nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? ((nk_u32_t const *)result)[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_u8_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        nk_align_(64) nk_f32_t roots_cache[256], floors_cache[256];
        nk_roots_u32_sme_streaming_(norms_cache, roots_cache, chunk_end - chunk_start);
        nk_floors_u32_sme_streaming_(norms_cache, floors_cache, chunk_end - chunk_start);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t const query_norm = ((nk_u32_t const *)result_row)[row_index];
            svfloat32_t const query_root_f32x = svsqrt_f32_x(svptrue_b32(),
                                                             svcvt_f32_u32_x(svptrue_b32(), svdup_n_u32(query_norm)));
            nk_angulars_from_u32_sme_streaming_(result_row + chunk_start, first - chunk_start, chunk_end - chunk_start,
                                                query_norm, query_root_f32x, norms_cache, roots_cache, floors_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_u8_sme_streaming_(vectors, stride, vector_count, depth, (nk_u32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_angulars_symmetric_finalize_u8_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                     result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_u8_sme_streaming_( //
    nk_u8_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u8_sme_streaming_(vectors + row_index * stride_elements, depth);
        nk_u32_t *result_row_norms = (nk_u32_t *)(result + row_index * result_stride_elements);
        result_row_norms[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_align_(64) nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? ((nk_u32_t const *)result)[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_u8_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_euclideans_from_u32_sme_streaming_(result_row + chunk_start, first - chunk_start,
                                                  chunk_end - chunk_start, ((nk_u32_t const *)result_row)[row_index],
                                                  norms_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_u8_sme_streaming_(vectors, stride, vector_count, depth, (nk_u32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_euclideans_symmetric_finalize_u8_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                       result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion U8 Integers

#pragma region I4 Integers

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_i4_sme_streaming_( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    nk_u32_t row_norms[64];
    nk_align_(64) nk_f32_t row_roots[64], column_roots[256], column_floors[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row)
            row_norms[row] = nk_dots_reduce_sumsq_i4_sme_streaming_(a + (row_first + row) * a_stride_elements, depth);
        nk_roots_u32_sme_streaming_(row_norms, row_roots, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_roots_u32_sme_streaming_(b_norms + column_first, column_roots, block_columns);
            nk_floors_u32_sme_streaming_(b_norms + column_first, column_floors, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_i32_sme_streaming_(c + (row_first + row) * c_stride_elements + column_first, 0,
                                                    block_columns, row_norms[row], svdup_n_f32(row_roots[row]),
                                                    b_norms + column_first, column_roots, column_floors);
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
    nk_start_sme_streaming_();
    nk_dots_packed_i4_sme_streaming_(a, a_stride, b_packed, (nk_i32_t *)c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_i4_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_i4_sme_streaming_( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i4x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_i4_sme_streaming_(a_row, depth);
        nk_euclideans_from_i32_sme_streaming_(result_row, 0, columns, query_norm_sq_u32, b_norms);
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_i4_sme( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_start_sme_streaming_();
    nk_dots_packed_i4_sme_streaming_(a, a_stride, b_packed, (nk_i32_t *)c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_i4_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_i4_sme_streaming_( //
    nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i4_sme_streaming_(vectors + row_index * stride_elements, depth);
        nk_u32_t *result_row_norms = (nk_u32_t *)(result + row_index * result_stride_elements);
        result_row_norms[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_align_(64) nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? ((nk_u32_t const *)result)[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_i4_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        nk_align_(64) nk_f32_t roots_cache[256], floors_cache[256];
        nk_roots_u32_sme_streaming_(norms_cache, roots_cache, chunk_end - chunk_start);
        nk_floors_u32_sme_streaming_(norms_cache, floors_cache, chunk_end - chunk_start);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t const query_norm = ((nk_u32_t const *)result_row)[row_index];
            svfloat32_t const query_root_f32x = svsqrt_f32_x(svptrue_b32(),
                                                             svcvt_f32_u32_x(svptrue_b32(), svdup_n_u32(query_norm)));
            nk_angulars_from_i32_sme_streaming_(result_row + chunk_start, first - chunk_start, chunk_end - chunk_start,
                                                query_norm, query_root_f32x, norms_cache, roots_cache, floors_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_i4_sme_streaming_(vectors, stride, vector_count, depth, (nk_i32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_angulars_symmetric_finalize_i4_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                     result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_i4_sme_streaming_( //
    nk_i4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i4_sme_streaming_(vectors + row_index * stride_elements, depth);
        nk_u32_t *result_row_norms = (nk_u32_t *)(result + row_index * result_stride_elements);
        result_row_norms[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_align_(64) nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? ((nk_u32_t const *)result)[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_i4_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_euclideans_from_i32_sme_streaming_(result_row + chunk_start, first - chunk_start,
                                                  chunk_end - chunk_start, ((nk_u32_t const *)result_row)[row_index],
                                                  norms_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_i4_sme_streaming_(vectors, stride, vector_count, depth, (nk_i32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_euclideans_symmetric_finalize_i4_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                       result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion Signed Integers

#pragma region U4 Integers

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_u4_sme_streaming_( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    nk_u32_t row_norms[64];
    nk_align_(64) nk_f32_t row_roots[64], column_roots[256], column_floors[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row)
            row_norms[row] = nk_dots_reduce_sumsq_u4_sme_streaming_(a + (row_first + row) * a_stride_elements, depth);
        nk_roots_u32_sme_streaming_(row_norms, row_roots, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_roots_u32_sme_streaming_(b_norms + column_first, column_roots, block_columns);
            nk_floors_u32_sme_streaming_(b_norms + column_first, column_floors, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_u32_sme_streaming_(c + (row_first + row) * c_stride_elements + column_first, 0,
                                                    block_columns, row_norms[row], svdup_n_f32(row_roots[row]),
                                                    b_norms + column_first, column_roots, column_floors);
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
    nk_start_sme_streaming_();
    nk_dots_packed_u4_sme_streaming_(a, a_stride, b_packed, (nk_u32_t *)c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_u4_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_u4_sme_streaming_( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_offset);
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u4x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq_u32 = nk_dots_reduce_sumsq_u4_sme_streaming_(a_row, depth);
        nk_euclideans_from_u32_sme_streaming_(result_row, 0, columns, query_norm_sq_u32, b_norms);
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_u4_sme( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_start_sme_streaming_();
    nk_dots_packed_u4_sme_streaming_(a, a_stride, b_packed, (nk_u32_t *)c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_u4_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_u4_sme_streaming_( //
    nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u4_sme_streaming_(vectors + row_index * stride_elements, depth);
        nk_u32_t *result_row_norms = (nk_u32_t *)(result + row_index * result_stride_elements);
        result_row_norms[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_align_(64) nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? ((nk_u32_t const *)result)[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_u4_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        nk_align_(64) nk_f32_t roots_cache[256], floors_cache[256];
        nk_roots_u32_sme_streaming_(norms_cache, roots_cache, chunk_end - chunk_start);
        nk_floors_u32_sme_streaming_(norms_cache, floors_cache, chunk_end - chunk_start);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t const query_norm = ((nk_u32_t const *)result_row)[row_index];
            svfloat32_t const query_root_f32x = svsqrt_f32_x(svptrue_b32(),
                                                             svcvt_f32_u32_x(svptrue_b32(), svdup_n_u32(query_norm)));
            nk_angulars_from_u32_sme_streaming_(result_row + chunk_start, first - chunk_start, chunk_end - chunk_start,
                                                query_norm, query_root_f32x, norms_cache, roots_cache, floors_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_u4_sme_streaming_(vectors, stride, vector_count, depth, (nk_u32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_angulars_symmetric_finalize_u4_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                     result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_u4_sme_streaming_( //
    nk_u4x2_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u4_sme_streaming_(vectors + row_index * stride_elements, depth);
        nk_u32_t *result_row_norms = (nk_u32_t *)(result + row_index * result_stride_elements);
        result_row_norms[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_align_(64) nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = col >= rows_begin && col < rows_end
                                                 ? ((nk_u32_t const *)result)[col * result_stride_elements + col]
                                                 : nk_dots_reduce_sumsq_u4_sme_streaming_(
                                                       vectors + col * stride_elements, depth);
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t const first = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (first >= chunk_end) break;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_euclideans_from_u32_sme_streaming_(result_row + chunk_start, first - chunk_start,
                                                  chunk_end - chunk_start, ((nk_u32_t const *)result_row)[row_index],
                                                  norms_cache);
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
    nk_start_sme_streaming_();
    nk_dots_symmetric_u4_sme_streaming_(vectors, stride, vector_count, depth, (nk_u32_t *)result, result_stride,
                                        rows_begin, rows_end);
    nk_euclideans_symmetric_finalize_u4_sme_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                       result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion Unsigned Integers

/** Turns packed NVFP4 relative dots into angular distances. */
NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_nvfp4_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                        void const *b_packed, nk_f32_t *c,
                                                                        nk_size_t rows, nk_size_t columns,
                                                                        nk_size_t depth,
                                                                        nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 16, 4);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &tensor_exponent) *
                              b.mantissa;
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row) {
            nk_i32_t exponent;
            row_norms[row] = nk_dots_scaled_row_nvfp4_sme_(a, a_stride, row_first + row, depth, &exponent);
        }
        nk_rsqrts_f32_sme_streaming_(row_norms, mantissa, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b.norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(
                    (nk_f32_t *)((char *)c + (row_first + row) * c_stride) + column_first, 0, block_columns,
                    row_norms[row], svdup_n_f32(row_rsqrts[row]), b.norms + column_first, column_rsqrts);
        }
    }
}

/** Turns symmetric NVFP4 relative dots into angular distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_nvfp4_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                           nk_size_t vector_count, nk_size_t depth,
                                                                           nk_f32_t *result, nk_size_t result_stride,
                                                                           nk_size_t rows_begin,
                                                                           nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale),
                                                   &tensor_exponent);
    nk_align_(64) nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_align_(64) nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin / 64 * 64; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column)
            norms[column - chunk] = nk_dots_scaled_row_nvfp4_sme_(vectors, stride, column, depth,
                                                                  &exponents[column - chunk]);
        nk_f32_t const squared_mantissa = mantissa * mantissa;
        nk_align_(64) nk_f32_t rsqrts[nk_sme_finish_columns_k];
        nk_rsqrts_f32_sme_streaming_(norms, 1, rsqrts, chunk_end - chunk);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_nvfp4_sme_(vectors, stride, row, depth, &exponent);
            nk_angulars_from_rsqrts_sme_streaming_(
                (nk_f32_t *)((char *)result + row * result_stride) + chunk, first - chunk, chunk_end - chunk, norm,
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(norm), squared_mantissa), norms, rsqrts);
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

/** Turns packed NVFP4 relative dots into Euclidean distances. */
NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_nvfp4_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                          void const *b_packed, nk_f32_t *c,
                                                                          nk_size_t rows, nk_size_t columns,
                                                                          nk_size_t depth,
                                                                          nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 16, 4);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &tensor_exponent) *
                              b.mantissa;
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_nvfp4_sme_(a, a_stride, row, depth, &exponent);
        nk_euclideans_from_relative_sme_streaming_((nk_f32_t *)((char *)c + row * c_stride), 0, columns, mantissa, norm,
                                                   exponent, b.norms, b.exponents);
    }
}

/** Turns symmetric NVFP4 relative dots into Euclidean distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_nvfp4_sme_streaming_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale),
                                                   &tensor_exponent);
    nk_align_(64) nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_align_(64) nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin / 64 * 64; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column)
            norms[column - chunk] = nk_dots_scaled_row_nvfp4_sme_(vectors, stride, column, depth,
                                                                  &exponents[column - chunk]);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_nvfp4_sme_(vectors, stride, row, depth, &exponent);
            nk_euclideans_from_relative_sme_streaming_((nk_f32_t *)((char *)result + row * result_stride) + chunk,
                                                       first - chunk, chunk_end - chunk, mantissa * mantissa, norm,
                                                       exponent, norms, exponents);
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_sme(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_nvfp4_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_nvfp4_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_nvfp4_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_sme(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_nvfp4_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_nvfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                           rows_end);
    nk_angulars_symmetric_finalize_nvfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                        rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_sme(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_nvfp4_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_nvfp4_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_nvfp4_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_sme(nk_nvfp4_cref_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_nvfp4_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_nvfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                           rows_end);
    nk_euclideans_symmetric_finalize_nvfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                          rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

/** Turns packed MXFP4 relative dots into angular distances. */
NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_mxfp4_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                        void const *b_packed, nk_f32_t *c,
                                                                        nk_size_t rows, nk_size_t columns,
                                                                        nk_size_t depth,
                                                                        nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 4);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &tensor_exponent) *
                              b.mantissa;
    nk_dots_scaled_exact_mxfp4_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                    depth, c, c_stride);
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row) {
            nk_i32_t exponent;
            row_norms[row] = nk_dots_scaled_row_mxfp4_sme_(a, a_stride, row_first + row, depth, &exponent);
        }
        nk_rsqrts_f32_sme_streaming_(row_norms, mantissa, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b.norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(
                    (nk_f32_t *)((char *)c + (row_first + row) * c_stride) + column_first, 0, block_columns,
                    row_norms[row], svdup_n_f32(row_rsqrts[row]), b.norms + column_first, column_rsqrts);
        }
    }
}

/** Turns symmetric MXFP4 relative dots into angular distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_mxfp4_sme_streaming_(nk_cross_operand_t vectors, nk_size_t stride,
                                                                           nk_size_t vector_count, nk_size_t depth,
                                                                           nk_f32_t *result, nk_size_t result_stride,
                                                                           nk_size_t rows_begin,
                                                                           nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale),
                                                   &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_align_(64) nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_align_(64) nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin / 64 * 64; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp4_sme_(vectors, stride, column, depth,
                                                                  &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp4_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end, bases,
                                        exponents, 1, depth, result, result_stride);
        nk_f32_t const squared_mantissa = mantissa * mantissa;
        nk_align_(64) nk_f32_t rsqrts[nk_sme_finish_columns_k];
        nk_rsqrts_f32_sme_streaming_(norms, 1, rsqrts, chunk_end - chunk);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp4_sme_(vectors, stride, row, depth, &exponent);
            nk_angulars_from_rsqrts_sme_streaming_(
                (nk_f32_t *)((char *)result + row * result_stride) + chunk, first - chunk, chunk_end - chunk, norm,
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(norm), squared_mantissa), norms, rsqrts);
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

/** Turns packed MXFP4 relative dots into Euclidean distances. */
NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_mxfp4_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                          void const *b_packed, nk_f32_t *c,
                                                                          nk_size_t rows, nk_size_t columns,
                                                                          nk_size_t depth,
                                                                          nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 4);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &tensor_exponent) *
                              b.mantissa;
    nk_dots_scaled_exact_mxfp4_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                    depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp4_sme_(a, a_stride, row, depth, &exponent);
        nk_euclideans_from_relative_sme_streaming_((nk_f32_t *)((char *)c + row * c_stride), 0, columns, mantissa, norm,
                                                   exponent, b.norms, b.exponents);
    }
}

/** Turns symmetric MXFP4 relative dots into Euclidean distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_mxfp4_sme_streaming_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale),
                                                   &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_align_(64) nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_align_(64) nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin / 64 * 64; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
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
            nk_euclideans_from_relative_sme_streaming_((nk_f32_t *)((char *)result + row * result_stride) + chunk,
                                                       first - chunk, chunk_end - chunk, mantissa * mantissa, norm,
                                                       exponent, norms, exponents);
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_sme(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp4_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp4_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_mxfp4_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_sme(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t rows_begin,
                                                        nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp4_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                           rows_end);
    nk_angulars_symmetric_finalize_mxfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                        rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_sme(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp4_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp4_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_mxfp4_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_sme(nk_mxfp4_cref_t const *vectors, nk_size_t vector_count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t rows_begin,
                                                          nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp4_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                           rows_end);
    nk_euclideans_symmetric_finalize_mxfp4_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                          rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

/** Turns packed MXFP6 E2M3 relative dots into angular distances. */
NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_mxfp6e2m3_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                            void const *b_packed, nk_f32_t *c,
                                                                            nk_size_t rows, nk_size_t columns,
                                                                            nk_size_t depth,
                                                                            nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &tensor_exponent) *
                              b.mantissa;
    nk_dots_scaled_exact_mxfp6e2m3_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row) {
            nk_i32_t exponent;
            row_norms[row] = nk_dots_scaled_row_mxfp6e2m3_sme_(a, a_stride, row_first + row, depth, &exponent);
        }
        nk_rsqrts_f32_sme_streaming_(row_norms, mantissa, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b.norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(
                    (nk_f32_t *)((char *)c + (row_first + row) * c_stride) + column_first, 0, block_columns,
                    row_norms[row], svdup_n_f32(row_rsqrts[row]), b.norms + column_first, column_rsqrts);
        }
    }
}

/** Turns symmetric MXFP6 E2M3 relative dots into angular distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_mxfp6e2m3_sme_streaming_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale),
                                                   &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_align_(64) nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_align_(64) nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin / 64 * 64; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp6e2m3_sme_(vectors, stride, column, depth,
                                                                      &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp6e2m3_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        nk_f32_t const squared_mantissa = mantissa * mantissa;
        nk_align_(64) nk_f32_t rsqrts[nk_sme_finish_columns_k];
        nk_rsqrts_f32_sme_streaming_(norms, 1, rsqrts, chunk_end - chunk);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp6e2m3_sme_(vectors, stride, row, depth, &exponent);
            nk_angulars_from_rsqrts_sme_streaming_(
                (nk_f32_t *)((char *)result + row * result_stride) + chunk, first - chunk, chunk_end - chunk, norm,
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(norm), squared_mantissa), norms, rsqrts);
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

/** Turns packed MXFP6 E2M3 relative dots into Euclidean distances. */
NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_mxfp6e2m3_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                              void const *b_packed, nk_f32_t *c,
                                                                              nk_size_t rows, nk_size_t columns,
                                                                              nk_size_t depth,
                                                                              nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &tensor_exponent) *
                              b.mantissa;
    nk_dots_scaled_exact_mxfp6e2m3_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp6e2m3_sme_(a, a_stride, row, depth, &exponent);
        nk_euclideans_from_relative_sme_streaming_((nk_f32_t *)((char *)c + row * c_stride), 0, columns, mantissa, norm,
                                                   exponent, b.norms, b.exponents);
    }
}

/** Turns symmetric MXFP6 E2M3 relative dots into Euclidean distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_mxfp6e2m3_sme_streaming_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale),
                                                   &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_align_(64) nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_align_(64) nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin / 64 * 64; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
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
            nk_euclideans_from_relative_sme_streaming_((nk_f32_t *)((char *)result + row * result_stride) + chunk,
                                                       first - chunk, chunk_end - chunk, mantissa * mantissa, norm,
                                                       exponent, norms, exponents);
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
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e2m3_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp6e2m3_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_mxfp6e2m3_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth,
                                                         c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e2m3_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp6e2m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_angulars_symmetric_finalize_mxfp6e2m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                            rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e2m3_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp6e2m3_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_mxfp6e2m3_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth,
                                                           c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *vectors,
                                                              nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                                              nk_f32_t *result, nk_size_t result_stride,
                                                              nk_size_t rows_begin, nk_size_t rows_end,
                                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e2m3_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp6e2m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_euclideans_symmetric_finalize_mxfp6e2m3_sme_streaming_(operand, stride, vector_count, depth, result,
                                                              result_stride, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

/** Turns packed MXFP6 E3M2 relative dots into angular distances. */
NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_mxfp6e3m2_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                            void const *b_packed, nk_f32_t *c,
                                                                            nk_size_t rows, nk_size_t columns,
                                                                            nk_size_t depth,
                                                                            nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &tensor_exponent) *
                              b.mantissa;
    nk_dots_scaled_exact_mxfp6e3m2_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row) {
            nk_i32_t exponent;
            row_norms[row] = nk_dots_scaled_row_mxfp6e3m2_sme_(a, a_stride, row_first + row, depth, &exponent);
        }
        nk_rsqrts_f32_sme_streaming_(row_norms, mantissa, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b.norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(
                    (nk_f32_t *)((char *)c + (row_first + row) * c_stride) + column_first, 0, block_columns,
                    row_norms[row], svdup_n_f32(row_rsqrts[row]), b.norms + column_first, column_rsqrts);
        }
    }
}

/** Turns symmetric MXFP6 E3M2 relative dots into angular distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_mxfp6e3m2_sme_streaming_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale),
                                                   &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_align_(64) nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_align_(64) nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin / 64 * 64; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp6e3m2_sme_(vectors, stride, column, depth,
                                                                      &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp6e3m2_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        nk_f32_t const squared_mantissa = mantissa * mantissa;
        nk_align_(64) nk_f32_t rsqrts[nk_sme_finish_columns_k];
        nk_rsqrts_f32_sme_streaming_(norms, 1, rsqrts, chunk_end - chunk);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp6e3m2_sme_(vectors, stride, row, depth, &exponent);
            nk_angulars_from_rsqrts_sme_streaming_(
                (nk_f32_t *)((char *)result + row * result_stride) + chunk, first - chunk, chunk_end - chunk, norm,
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(norm), squared_mantissa), norms, rsqrts);
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

/** Turns packed MXFP6 E3M2 relative dots into Euclidean distances. */
NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_mxfp6e3m2_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                              void const *b_packed, nk_f32_t *c,
                                                                              nk_size_t rows, nk_size_t columns,
                                                                              nk_size_t depth,
                                                                              nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &tensor_exponent) *
                              b.mantissa;
    nk_dots_scaled_exact_mxfp6e3m2_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp6e3m2_sme_(a, a_stride, row, depth, &exponent);
        nk_euclideans_from_relative_sme_streaming_((nk_f32_t *)((char *)c + row * c_stride), 0, columns, mantissa, norm,
                                                   exponent, b.norms, b.exponents);
    }
}

/** Turns symmetric MXFP6 E3M2 relative dots into Euclidean distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_mxfp6e3m2_sme_streaming_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale),
                                                   &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_align_(64) nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_align_(64) nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin / 64 * 64; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
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
            nk_euclideans_from_relative_sme_streaming_((nk_f32_t *)((char *)result + row * result_stride) + chunk,
                                                       first - chunk, chunk_end - chunk, mantissa * mantissa, norm,
                                                       exponent, norms, exponents);
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
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e3m2_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp6e3m2_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_mxfp6e3m2_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth,
                                                         c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e3m2_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp6e3m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_angulars_symmetric_finalize_mxfp6e3m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                            rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e3m2_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp6e3m2_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_mxfp6e3m2_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth,
                                                           c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *vectors,
                                                              nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                                              nk_f32_t *result, nk_size_t result_stride,
                                                              nk_size_t rows_begin, nk_size_t rows_end,
                                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp6e3m2_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp6e3m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_euclideans_symmetric_finalize_mxfp6e3m2_sme_streaming_(operand, stride, vector_count, depth, result,
                                                              result_stride, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

/** Turns packed MXFP8 E4M3 relative dots into angular distances. */
NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_mxfp8e4m3_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                            void const *b_packed, nk_f32_t *c,
                                                                            nk_size_t rows, nk_size_t columns,
                                                                            nk_size_t depth,
                                                                            nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &tensor_exponent) *
                              b.mantissa;
    nk_dots_scaled_exact_mxfp8e4m3_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row) {
            nk_i32_t exponent;
            row_norms[row] = nk_dots_scaled_row_mxfp8e4m3_sme_(a, a_stride, row_first + row, depth, &exponent);
        }
        nk_rsqrts_f32_sme_streaming_(row_norms, mantissa, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b.norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(
                    (nk_f32_t *)((char *)c + (row_first + row) * c_stride) + column_first, 0, block_columns,
                    row_norms[row], svdup_n_f32(row_rsqrts[row]), b.norms + column_first, column_rsqrts);
        }
    }
}

/** Turns symmetric MXFP8 E4M3 relative dots into angular distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_mxfp8e4m3_sme_streaming_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale),
                                                   &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_align_(64) nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_align_(64) nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin / 64 * 64; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp8e4m3_sme_(vectors, stride, column, depth,
                                                                      &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp8e4m3_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        nk_f32_t const squared_mantissa = mantissa * mantissa;
        nk_align_(64) nk_f32_t rsqrts[nk_sme_finish_columns_k];
        nk_rsqrts_f32_sme_streaming_(norms, 1, rsqrts, chunk_end - chunk);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp8e4m3_sme_(vectors, stride, row, depth, &exponent);
            nk_angulars_from_rsqrts_sme_streaming_(
                (nk_f32_t *)((char *)result + row * result_stride) + chunk, first - chunk, chunk_end - chunk, norm,
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(norm), squared_mantissa), norms, rsqrts);
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

/** Turns packed MXFP8 E4M3 relative dots into Euclidean distances. */
NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_mxfp8e4m3_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                              void const *b_packed, nk_f32_t *c,
                                                                              nk_size_t rows, nk_size_t columns,
                                                                              nk_size_t depth,
                                                                              nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &tensor_exponent) *
                              b.mantissa;
    nk_dots_scaled_exact_mxfp8e4m3_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp8e4m3_sme_(a, a_stride, row, depth, &exponent);
        nk_euclideans_from_relative_sme_streaming_((nk_f32_t *)((char *)c + row * c_stride), 0, columns, mantissa, norm,
                                                   exponent, b.norms, b.exponents);
    }
}

/** Turns symmetric MXFP8 E4M3 relative dots into Euclidean distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_mxfp8e4m3_sme_streaming_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale),
                                                   &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_align_(64) nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_align_(64) nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin / 64 * 64; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
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
            nk_euclideans_from_relative_sme_streaming_((nk_f32_t *)((char *)result + row * result_stride) + chunk,
                                                       first - chunk, chunk_end - chunk, mantissa * mantissa, norm,
                                                       exponent, norms, exponents);
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
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e4m3_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp8e4m3_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_mxfp8e4m3_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth,
                                                         c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e4m3_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp8e4m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_angulars_symmetric_finalize_mxfp8e4m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                            rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e4m3_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp8e4m3_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_mxfp8e4m3_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth,
                                                           c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *vectors,
                                                              nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                                              nk_f32_t *result, nk_size_t result_stride,
                                                              nk_size_t rows_begin, nk_size_t rows_end,
                                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e4m3_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp8e4m3_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_euclideans_symmetric_finalize_mxfp8e4m3_sme_streaming_(operand, stride, vector_count, depth, result,
                                                              result_stride, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

/** Turns packed MXFP8 E5M2 relative dots into angular distances. */
NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_mxfp8e5m2_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                            void const *b_packed, nk_f32_t *c,
                                                                            nk_size_t rows, nk_size_t columns,
                                                                            nk_size_t depth,
                                                                            nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &tensor_exponent) *
                              b.mantissa;
    nk_dots_scaled_exact_mxfp8e5m2_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    nk_f32_t row_norms[64], row_rsqrts[64], column_rsqrts[256];
    for (nk_size_t row_first = 0; row_first < rows; row_first += 64) {
        nk_size_t const block_rows = rows - row_first < 64 ? rows - row_first : 64;
        for (nk_size_t row = 0; row < block_rows; ++row) {
            nk_i32_t exponent;
            row_norms[row] = nk_dots_scaled_row_mxfp8e5m2_sme_(a, a_stride, row_first + row, depth, &exponent);
        }
        nk_rsqrts_f32_sme_streaming_(row_norms, mantissa, row_rsqrts, block_rows);
        for (nk_size_t column_first = 0; column_first < columns; column_first += 256) {
            nk_size_t const block_columns = columns - column_first < 256 ? columns - column_first : 256;
            nk_rsqrts_f32_sme_streaming_(b.norms + column_first, 1, column_rsqrts, block_columns);
            for (nk_size_t row = 0; row < block_rows; ++row)
                nk_angulars_from_rsqrts_sme_streaming_(
                    (nk_f32_t *)((char *)c + (row_first + row) * c_stride) + column_first, 0, block_columns,
                    row_norms[row], svdup_n_f32(row_rsqrts[row]), b.norms + column_first, column_rsqrts);
        }
    }
}

/** Turns symmetric MXFP8 E5M2 relative dots into angular distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_mxfp8e5m2_sme_streaming_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale),
                                                   &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_align_(64) nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_align_(64) nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin / 64 * 64; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
            bases[column - chunk] = (nk_i8_t)(wide ? -128 : 0);
            norms[column - chunk] = nk_dots_scaled_row_mxfp8e5m2_sme_(vectors, stride, column, depth,
                                                                      &exponents[column - chunk]);
        }
        nk_dots_scaled_exact_mxfp8e5m2_sme_(vectors, stride, rows_begin, rows_end, vectors, stride, chunk, chunk_end,
                                            bases, exponents, 1, depth, result, result_stride);
        nk_f32_t const squared_mantissa = mantissa * mantissa;
        nk_align_(64) nk_f32_t rsqrts[nk_sme_finish_columns_k];
        nk_rsqrts_f32_sme_streaming_(norms, 1, rsqrts, chunk_end - chunk);
        for (nk_size_t row = rows_begin; row < rows_end && row < chunk_end; ++row) {
            nk_size_t const first = row > chunk ? row : chunk;
            nk_i32_t exponent;
            nk_f32_t const norm = nk_dots_scaled_row_mxfp8e5m2_sme_(vectors, stride, row, depth, &exponent);
            nk_angulars_from_rsqrts_sme_streaming_(
                (nk_f32_t *)((char *)result + row * result_stride) + chunk, first - chunk, chunk_end - chunk, norm,
                nk_rsqrt_f32x_sme_streaming_(svptrue_b32(), svdup_n_f32(norm), squared_mantissa), norms, rsqrts);
        }
    }
    for (nk_size_t row = rows_begin; row < rows_end; ++row)
        ((nk_f32_t *)((char *)result + row * result_stride))[row] = 0;
}

/** Turns packed MXFP8 E5M2 relative dots into Euclidean distances. */
NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_mxfp8e5m2_sme_streaming_(nk_cross_operand_t a, nk_size_t a_stride,
                                                                              void const *b_packed, nk_f32_t *c,
                                                                              nk_size_t rows, nk_size_t columns,
                                                                              nk_size_t depth,
                                                                              nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_scaled_sme_columns_t const b = nk_dots_scaled_columns_sme_(b_packed, 32, 8);
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(a.tensor_scale), &tensor_exponent) *
                              b.mantissa;
    nk_dots_scaled_exact_mxfp8e5m2_sme_(a, a_stride, 0, rows, b.raw, b.raw_stride, 0, columns, b.bases, b.exponents, 0,
                                        depth, c, c_stride);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_i32_t exponent;
        nk_f32_t const norm = nk_dots_scaled_row_mxfp8e5m2_sme_(a, a_stride, row, depth, &exponent);
        nk_euclideans_from_relative_sme_streaming_((nk_f32_t *)((char *)c + row * c_stride), 0, columns, mantissa, norm,
                                                   exponent, b.norms, b.exponents);
    }
}

/** Turns symmetric MXFP8 E5M2 relative dots into Euclidean distances with a zero diagonal. */
NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_mxfp8e5m2_sme_streaming_(
    nk_cross_operand_t vectors, nk_size_t stride, nk_size_t vector_count, nk_size_t depth, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    nk_size_t const blocks = depth / 32;
    nk_i32_t tensor_exponent;
    nk_f32_t const mantissa = nk_split_f32_serial_(nk_cross_tensor_scale_serial_(vectors.tensor_scale),
                                                   &tensor_exponent);
    nk_i8_t bases[nk_sme_finish_columns_k];
    nk_align_(64) nk_i32_t exponents[nk_sme_finish_columns_k];
    nk_align_(64) nk_f32_t norms[nk_sme_finish_columns_k];
    for (nk_size_t chunk = rows_begin / 64 * 64; chunk < vector_count; chunk += nk_sme_finish_columns_k) {
        nk_size_t const chunk_end = nk_min_of_two(chunk + nk_sme_finish_columns_k, vector_count);
        for (nk_size_t column = chunk; column < chunk_end; ++column) {
            int wide;
            nk_mx_base_sme_(vectors.scales + column * vectors.scales_stride, blocks, &wide);
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
            nk_euclideans_from_relative_sme_streaming_((nk_f32_t *)((char *)result + row * result_stride) + chunk,
                                                       first - chunk, chunk_end - chunk, mantissa * mantissa, norm,
                                                       exponent, norms, exponents);
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
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e5m2_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp8e5m2_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_angulars_packed_finalize_mxfp8e5m2_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth,
                                                         c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t vector_count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t rows_begin,
                                                            nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e5m2_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp8e5m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_angulars_symmetric_finalize_mxfp8e5m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride,
                                                            rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (nk_dots_check_sme_(b_packed) != nk_success_k) return nk_pack_mismatch_k;
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e5m2_k, a, a_stride);
    nk_start_sme_streaming_();
    nk_dots_packed_mxfp8e5m2_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_finalize_mxfp8e5m2_sme_streaming_(operand, a_stride, b_packed, c, rows, columns, depth,
                                                           c_stride);
    nk_stop_sme_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *vectors,
                                                              nk_size_t vector_count, nk_size_t depth, nk_size_t stride,
                                                              nk_f32_t *result, nk_size_t result_stride,
                                                              nk_size_t rows_begin, nk_size_t rows_end,
                                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_cross_operand_t const operand = nk_cross_operand_serial_(nk_mxfp8e5m2_k, vectors, stride);
    nk_start_sme_streaming_();
    nk_dots_symmetric_mxfp8e5m2_sme_streaming_(operand, stride, vector_count, depth, result, result_stride, rows_begin,
                                               rows_end);
    nk_euclideans_symmetric_finalize_mxfp8e5m2_sme_streaming_(operand, stride, vector_count, depth, result,
                                                              result_stride, rows_begin, rows_end);
    nk_stop_sme_streaming_();
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
