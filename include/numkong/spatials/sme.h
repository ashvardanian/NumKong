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
    svfloat32_t const angular_f32x = svsub_f32_x(predicate_b32x, svdup_f32(1),
                                                 svmul_f32_x(predicate_b32x, dots_f32x, rsqrt_f32x));
    svbool_t const normed_b32x = svand_b_z(predicate_b32x, svcmpgt_n_f32(predicate_b32x, query_norm_sq_f32x, 0),
                                           svcmpgt_n_f32(predicate_b32x, target_norms_sq_f32x, 0));
    return svsel_f32(normed_b32x, svmaxnm_n_f32_x(predicate_b32x, angular_f32x, 0),
                     svsel_f32(svcmpeq_n_f32(predicate_b32x, dots_f32x, 0), svdup_f32(0), svdup_f32(1)));
}

NUMKONG_INLINE svfloat32_t nk_euclideans_from_dot_f32x_ssve_(svbool_t predicate_b32x, svfloat32_t dots_f32x,
                                                             svfloat32_t query_norm_sq_f32x,
                                                             svfloat32_t target_norms_sq_f32x) NUMKONG_STREAMING_ {
    svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_b32x, query_norm_sq_f32x, target_norms_sq_f32x);
    svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_b32x, sum_sq_f32x,
                                           svmul_f32_x(predicate_b32x, svdup_n_f32(2.0f), dots_f32x));
    dist_sq_f32x = svmaxnm_n_f32_x(predicate_b32x, dist_sq_f32x, 0);
    return svsqrt_f32_x(predicate_b32x, dist_sq_f32x);
}

/** Squared norm of block-scaled row @p row of @p operand with its tensor scale squared, as F32 sums
 *  of the exact squares of the values the dots fold, rebased like them for MX; rows wider than the
 *  fold covers take the exact serial sum. */
NUMKONG_INLINE nk_f32_t nk_dots_reduce_sumsq_scaled_ssve_(nk_dtype_t dtype, nk_cross_operand_t operand,
                                                          nk_size_t row_stride, nk_size_t row,
                                                          nk_size_t depth) NUMKONG_STREAMING_ {
    int const nvfp4 = dtype == nk_nvfp4_k;
    nk_size_t const blocks = depth / (nvfp4 ? 16 : 32);
    nk_f64_t const tensor_scale = nk_cross_tensor_scale_(operand.tensor_scale);
    nk_i32_t base = 0;
    if (!nvfp4) {
        nk_i32_t minimum, maximum;
        nk_sme_exponent_range_(operand.scales + row * operand.scales_stride, blocks, &minimum, &maximum);
        if (maximum - minimum > nk_sme_exponent_spread_k)
            return nk_cross_scaled_sumsq_serial_(dtype, (nk_u8_t const *)operand.elements + row * row_stride,
                                                 operand.scales + row * operand.scales_stride, tensor_scale, depth);
        base = dtype == nk_mxfp4_k ? maximum : minimum;
    }
    svbool_t const predicate_all_b32x = svptrue_b32();
    nk_size_t const vector_elements = svcnth();
    svfloat32_t sum_even_f32x = svdup_n_f32(0), sum_odd_f32x = svdup_n_f32(0);
    for (nk_size_t first = 0; first < depth; first += vector_elements) {
        nk_size_t const count = depth - first < vector_elements ? depth - first : vector_elements;
        svuint32_t const bits_u32x = svreinterpret_u32_u16(
            nk_sme_decode_row_b16_(dtype, nk_f16_k, operand, row_stride, row, first, count, base));
        svfloat32_t even_f32x, odd_f32x;
        if (nvfp4) {
            svfloat16_t const halves_f16x = svreinterpret_f16_u32(bits_u32x);
            even_f32x = svcvt_f32_f16_x(predicate_all_b32x, halves_f16x);
            odd_f32x = svcvtlt_f32_f16_x(predicate_all_b32x, halves_f16x);
        }
        else {
            even_f32x = svreinterpret_f32_u32(svlsl_n_u32_x(predicate_all_b32x, bits_u32x, 16));
            odd_f32x = svreinterpret_f32_u32(svand_n_u32_x(predicate_all_b32x, bits_u32x, 0xFFFF0000u));
        }
        sum_even_f32x = svmla_f32_x(predicate_all_b32x, sum_even_f32x, even_f32x, even_f32x);
        sum_odd_f32x = svmla_f32_x(predicate_all_b32x, sum_odd_f32x, odd_f32x, odd_f32x);
    }
    nk_f64_t const sum = svaddv_f32(predicate_all_b32x, svadd_f32_x(predicate_all_b32x, sum_even_f32x, sum_odd_f32x));
    nk_fui64_t rebase;
    rebase.u = (nk_u64_t)(1023 + 2 * base) << 52;
    return (nk_f32_t)(sum * rebase.f * tensor_scale * tensor_scale);
}

/** Whether all @p count squared norms are positive, so none of their angles needs the F64 recheck.
 *  Scalar FP compares cost ~30 ns each in streaming mode, so rows test lanes only if this fails. */
NUMKONG_INLINE int nk_spatials_norms_positive_ssve_(nk_f32_t const *norms, nk_size_t count) NUMKONG_STREAMING_ {
    svbool_t unnormed_b32x = svpfalse_b();
    for (nk_size_t index = 0; index < count; index += svcntw()) {
        svbool_t const predicate_b32x = svwhilelt_b32_u64(index, count);
        svfloat32_t const norms_f32x = svld1_f32(predicate_b32x, norms + index);
        unnormed_b32x = svorr_b_z(svptrue_b32(), unnormed_b32x,
                                  svnot_b_z(predicate_b32x, svcmpgt_n_f32(predicate_b32x, norms_f32x, 0)));
    }
    return !svptest_any(svptrue_b32(), unnormed_b32x);
}

/** Whether any of the next @p count angles, up to one vector, is 1 without both its row's squared
 *  @p norm and its column's squared @p norms positive. */
NUMKONG_INLINE int nk_spatials_recheck_any_ssve_(nk_f32_t const *angles, nk_f32_t const *norms, nk_f32_t norm,
                                                 nk_size_t count) NUMKONG_STREAMING_ {
    svbool_t const predicate_b32x = svwhilelt_b32_u64(0, count);
    svbool_t const normed_b32x = svand_b_z(predicate_b32x,
                                           svcmpgt_n_f32(predicate_b32x, svld1_f32(predicate_b32x, norms), 0),
                                           svcmpgt_n_f32(predicate_b32x, svdup_f32(norm), 0));
    svbool_t const ones_b32x = svcmpeq_n_f32(predicate_b32x, svld1_f32(predicate_b32x, angles), 1);
    return svptest_any(predicate_b32x, svbic_b_z(predicate_b32x, ones_b32x, normed_b32x));
}

NUMKONG_OUTLINED_ void nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_kind_t metric, nk_dtype_t dtype,
                                                                    nk_cross_operand_t a, void const *b_packed,
                                                                    nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                    nk_size_t depth, nk_size_t a_stride,
                                                                    nk_size_t c_stride) NUMKONG_STREAMING_ {
    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((nk_u8_t const *)b_packed + header->norms_offset);
    nk_cross_operand_t const unpacked = {NUMKONG_NULL, NUMKONG_NULL, 0, NUMKONG_NULL};
    int const columns_normed = nk_spatials_norms_positive_ssve_(b_norms, columns);
    for (nk_size_t row = 0; row < rows; ++row) {
        nk_f32_t const norm = nk_dots_reduce_sumsq_scaled_ssve_(dtype, a, a_stride, row, depth);
        nk_f32_t *output = (nk_f32_t *)((nk_u8_t *)c + row * c_stride);
        for (nk_size_t column = 0; column < columns; column += svcntw()) {
            svbool_t const predicate_b32x = svwhilelt_b32_u64(column, columns);
            svfloat32_t const dots_f32x = svld1_f32(predicate_b32x, output + column);
            svfloat32_t const norms_f32x = svld1_f32(predicate_b32x, b_norms + column);
            svfloat32_t const distances_f32x =
                metric == nk_kernel_angular_k
                    ? nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, svdup_f32(norm), norms_f32x)
                    : nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, svdup_f32(norm), norms_f32x);
            svst1_f32(predicate_b32x, output + column, distances_f32x);
        }
        // Without both norms the angle is 1 only for a nonzero dot, and F32 sums of wide blocks can
        // leave a residue where the exact dot cancels, so those lanes recheck it in F64
        if (metric != nk_kernel_angular_k || (columns_normed && norm > 0)) continue;
        for (nk_size_t column = 0; column < columns; column += svcntw()) {
            if (!nk_spatials_recheck_any_ssve_(output + column, b_norms + column, norm, columns - column)) continue;
            for (nk_size_t lane = column; lane < columns && lane < column + svcntw(); ++lane)
                if (output[lane] == 1 && !(norm > 0 && b_norms[lane] > 0) &&
                    nk_dots_scaled_sme_exact_row_(dtype, a, a_stride, row, b_packed, unpacked, 0, lane, depth) == 0)
                    output[lane] = 0;
        }
    }
}

NUMKONG_OUTLINED_ void nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_kind_t metric, nk_dtype_t dtype,
                                                                       nk_cross_operand_t vectors, nk_size_t count,
                                                                       nk_size_t depth, nk_size_t stride,
                                                                       nk_f32_t *result, nk_size_t result_stride,
                                                                       nk_size_t row_start,
                                                                       nk_size_t row_count) NUMKONG_STREAMING_ {
    for (nk_size_t row = row_start; row < row_start + row_count; ++row) {
        nk_f32_t *output = (nk_f32_t *)((nk_u8_t *)result + row * result_stride);
        output[row] = nk_dots_reduce_sumsq_scaled_ssve_(dtype, vectors, stride, row, depth);
    }
    nk_f32_t column_norms[256];
    for (nk_size_t chunk = row_start; chunk < count; chunk += 256) {
        nk_size_t const column_end = nk_min_of_two(chunk + 256, count);
        for (nk_size_t column = chunk; column < column_end; ++column)
            column_norms[column - chunk] = nk_dots_reduce_sumsq_scaled_ssve_(dtype, vectors, stride, column, depth);
        int const columns_normed = nk_spatials_norms_positive_ssve_(column_norms, column_end - chunk);
        for (nk_size_t row = row_start; row < row_start + row_count; ++row) {
            nk_f32_t *output = (nk_f32_t *)((nk_u8_t *)result + row * result_stride);
            nk_f32_t const norm = output[row];
            for (nk_size_t column = nk_max_of_two(row + 1, chunk); column < column_end; column += svcntw()) {
                svbool_t const predicate_b32x = svwhilelt_b32_u64(column, column_end);
                svfloat32_t const dots_f32x = svld1_f32(predicate_b32x, output + column);
                svfloat32_t const norms_f32x = svld1_f32(predicate_b32x, column_norms + column - chunk);
                svfloat32_t const distances_f32x =
                    metric == nk_kernel_angular_k
                        ? nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, svdup_f32(norm), norms_f32x)
                        : nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, svdup_f32(norm), norms_f32x);
                svst1_f32(predicate_b32x, output + column, distances_f32x);
            }
            // Lanes without both norms recheck a nonzero dot in F64, as the packed finalize does
            if (metric != nk_kernel_angular_k || (columns_normed && norm > 0)) continue;
            for (nk_size_t column = nk_max_of_two(row + 1, chunk); column < column_end; column += svcntw()) {
                if (!nk_spatials_recheck_any_ssve_(output + column, column_norms + column - chunk, norm,
                                                   column_end - column))
                    continue;
                for (nk_size_t lane = column; lane < column_end && lane < column + svcntw(); ++lane)
                    if (output[lane] == 1 && !(norm > 0 && column_norms[lane - chunk] > 0) &&
                        nk_dots_scaled_sme_exact_row_(dtype, vectors, stride, row, NUMKONG_NULL, vectors, stride, lane,
                                                      depth) == 0)
                        output[lane] = 0;
            }
        }
    }
    for (nk_size_t row = row_start; row < row_start + row_count; ++row)
        ((nk_f32_t *)((nk_u8_t *)result + row * result_stride))[row] = 0;
}

#if NUMKONG_TARGET_SME
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

NUMKONG_API nk_status_t nk_angulars_packed_f16_sme( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {a, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_packed_b16_sme_streaming_(nk_f16_k, operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {a, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_packed_b16_sme_streaming_(nk_f16_k, operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_f16_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_f16_sme_finalize_ssve_( //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f16_sme( //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {vectors, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_symmetric_b16_sme_streaming_(nk_f16_k, operand, stride, vectors_count, depth, result, result_stride,
                                         row_start, row_count);
    nk_angulars_symmetric_f16_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                 result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_f16_sme_finalize_ssve_( //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_sme( //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {vectors, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_symmetric_b16_sme_streaming_(nk_f16_k, operand, stride, vectors_count, depth, result, result_stride,
                                         row_start, row_count);
    nk_euclideans_symmetric_f16_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                   result_stride_elements, row_start, row_count);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {a, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_packed_b16_sme_streaming_(nk_bf16_k, operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {a, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_packed_b16_sme_streaming_(nk_bf16_k, operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_bf16_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_bf16_sme_finalize_ssve_( //
    nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_sme( //
    nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {vectors, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_symmetric_b16_sme_streaming_(nk_bf16_k, operand, stride, vectors_count, depth, result, result_stride,
                                         row_start, row_count);
    nk_angulars_symmetric_bf16_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                  result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_bf16_sme_finalize_ssve_( //
    nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_sme( //
    nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {vectors, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_symmetric_b16_sme_streaming_(nk_bf16_k, operand, stride, vectors_count, depth, result, result_stride,
                                         row_start, row_count);
    nk_euclideans_symmetric_bf16_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                    result_stride_elements, row_start, row_count);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {a, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_packed_b16_sme_streaming_(nk_e4m3_k, operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {a, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_packed_b16_sme_streaming_(nk_e4m3_k, operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_e4m3_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_e4m3_sme_finalize_ssve_( //
    nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_sme( //
    nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {vectors, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_symmetric_b16_sme_streaming_(nk_e4m3_k, operand, stride, vectors_count, depth, result, result_stride,
                                         row_start, row_count);
    nk_angulars_symmetric_e4m3_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                  result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_e4m3_sme_finalize_ssve_( //
    nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_sme( //
    nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {vectors, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_symmetric_b16_sme_streaming_(nk_e4m3_k, operand, stride, vectors_count, depth, result, result_stride,
                                         row_start, row_count);
    nk_euclideans_symmetric_e4m3_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                    result_stride_elements, row_start, row_count);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {a, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_packed_b16_sme_streaming_(nk_e5m2_k, operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {a, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_packed_b16_sme_streaming_(nk_e5m2_k, operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_e5m2_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_e5m2_sme_finalize_ssve_( //
    nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_sme( //
    nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {vectors, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_symmetric_b16_sme_streaming_(nk_e5m2_k, operand, stride, vectors_count, depth, result, result_stride,
                                         row_start, row_count);
    nk_angulars_symmetric_e5m2_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                  result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_e5m2_sme_finalize_ssve_( //
    nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_sme( //
    nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {vectors, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_symmetric_b16_sme_streaming_(nk_e5m2_k, operand, stride, vectors_count, depth, result, result_stride,
                                         row_start, row_count);
    nk_euclideans_symmetric_e5m2_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                    result_stride_elements, row_start, row_count);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(nk_e2m3_k, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(nk_e2m3_k, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_e2m3_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_e2m3_sme_finalize_ssve_( //
    nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_sme( //
    nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(nk_e2m3_k, vectors, stride, vectors_count, depth, result, result_stride,
                                        row_start, row_count);
    nk_angulars_symmetric_e2m3_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                  result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_e2m3_sme_finalize_ssve_( //
    nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_sme( //
    nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(nk_e2m3_k, vectors, stride, vectors_count, depth, result, result_stride,
                                        row_start, row_count);
    nk_euclideans_symmetric_e2m3_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                    result_stride_elements, row_start, row_count);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m1x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(nk_e2m1_k, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m1x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(nk_e2m1_k, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_e2m1_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_e2m1_sme_finalize_ssve_( //
    nk_e2m1x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m1_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m1_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_sme( //
    nk_e2m1x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e2m1x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(nk_e2m1_k, vectors, stride, vectors_count, depth, result, result_stride,
                                        row_start, row_count);
    nk_angulars_symmetric_e2m1_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                  result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_e2m1_sme_finalize_ssve_( //
    nk_e2m1x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m1_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m1_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_sme( //
    nk_e2m1x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e2m1x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(nk_e2m1_k, vectors, stride, vectors_count, depth, result, result_stride,
                                        row_start, row_count);
    nk_euclideans_symmetric_e2m1_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                    result_stride_elements, row_start, row_count);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {a, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_packed_b16_sme_streaming_(nk_e3m2_k, operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
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
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {a, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_packed_b16_sme_streaming_(nk_e3m2_k, operand, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_e3m2_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_e3m2_sme_finalize_ssve_( //
    nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_sme( //
    nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {vectors, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_symmetric_b16_sme_streaming_(nk_e3m2_k, operand, stride, vectors_count, depth, result, result_stride,
                                         row_start, row_count);
    nk_angulars_symmetric_e3m2_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                  result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_e3m2_sme_finalize_ssve_( //
    nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
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
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_sme( //
    nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_cross_operand_t const operand = {vectors, NUMKONG_NULL, 0, NUMKONG_NULL};
    nk_sme_start_streaming_();
    nk_dots_symmetric_b16_sme_streaming_(nk_e3m2_k, operand, stride, vectors_count, depth, result, result_stride,
                                         row_start, row_count);
    nk_euclideans_symmetric_e3m2_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                    result_stride_elements, row_start, row_count);
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
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_s32_x(
                predicate_b32x, svld1_s32(predicate_b32x, (nk_i32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_b32x,
                                                               svld1_u32(predicate_b32x, b_norms + col_index));
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_i8_sme( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(nk_i8_k, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
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
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_s32_x(
                predicate_b32x, svld1_s32(predicate_b32x, (nk_i32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_b32x,
                                                               svld1_u32(predicate_b32x, b_norms + col_index));
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_i8_sme( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(nk_i8_k, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_i8_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_i8_sme_finalize_ssve_( //
    nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_s32_x(
                    predicate_b32x, svld1_s32(predicate_b32x, (nk_i32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_b32x, svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_i8_sme( //
    nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(nk_i8_k, vectors, stride, vectors_count, depth, result, result_stride,
                                        row_start, row_count);
    nk_angulars_symmetric_i8_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_i8_sme_finalize_ssve_( //
    nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_s32_x(
                    predicate_b32x, svld1_s32(predicate_b32x, (nk_i32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_b32x, svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_sme( //
    nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(nk_i8_k, vectors, stride, vectors_count, depth, result, result_stride,
                                        row_start, row_count);
    nk_euclideans_symmetric_i8_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                  result_stride_elements, row_start, row_count);
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
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_u32_x(
                predicate_b32x, svld1_u32(predicate_b32x, (nk_u32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_b32x,
                                                               svld1_u32(predicate_b32x, b_norms + col_index));
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_u8_sme( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(nk_u8_k, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
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
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_u32_x(
                predicate_b32x, svld1_u32(predicate_b32x, (nk_u32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_b32x,
                                                               svld1_u32(predicate_b32x, b_norms + col_index));
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_u8_sme( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(nk_u8_k, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_u8_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_u8_sme_finalize_ssve_( //
    nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_u32_x(
                    predicate_b32x, svld1_u32(predicate_b32x, (nk_u32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_b32x, svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_u8_sme( //
    nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(nk_u8_k, vectors, stride, vectors_count, depth, result, result_stride,
                                        row_start, row_count);
    nk_angulars_symmetric_u8_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_u8_sme_finalize_ssve_( //
    nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_u32_x(
                    predicate_b32x, svld1_u32(predicate_b32x, (nk_u32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_b32x, svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_sme( //
    nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(nk_u8_k, vectors, stride, vectors_count, depth, result, result_stride,
                                        row_start, row_count);
    nk_euclideans_symmetric_u8_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                  result_stride_elements, row_start, row_count);
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
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_s32_x(
                predicate_b32x, svld1_s32(predicate_b32x, (nk_i32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_b32x,
                                                               svld1_u32(predicate_b32x, b_norms + col_index));
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_i4_sme( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(nk_i4_k, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
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
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_s32_x(
                predicate_b32x, svld1_s32(predicate_b32x, (nk_i32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_b32x,
                                                               svld1_u32(predicate_b32x, b_norms + col_index));
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_i4_sme( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(nk_i4_k, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_i4_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_i4_sme_finalize_ssve_( //
    nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i4_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_s32_x(
                    predicate_b32x, svld1_s32(predicate_b32x, (nk_i32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_b32x, svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_i4_sme( //
    nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(nk_i4_k, vectors, stride, vectors_count, depth, result, result_stride,
                                        row_start, row_count);
    nk_angulars_symmetric_i4_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_i4_sme_finalize_ssve_( //
    nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i4_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_s32_x(
                    predicate_b32x, svld1_s32(predicate_b32x, (nk_i32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_b32x, svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_i4_sme( //
    nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(nk_i4_k, vectors, stride, vectors_count, depth, result, result_stride,
                                        row_start, row_count);
    nk_euclideans_symmetric_i4_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                  result_stride_elements, row_start, row_count);
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
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_u32_x(
                predicate_b32x, svld1_u32(predicate_b32x, (nk_u32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_b32x,
                                                               svld1_u32(predicate_b32x, b_norms + col_index));
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_u4_sme( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(nk_u4_k, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
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
        svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_norm_sq_u32);
        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntw()) {
            svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_u32_x(
                predicate_b32x, svld1_u32(predicate_b32x, (nk_u32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_b32x,
                                                               svld1_u32(predicate_b32x, b_norms + col_index));
            svst1_f32(
                predicate_b32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_u4_sme( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_sme_k) return nk_pack_mismatch_k;
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u4x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_packed_b8_sme_streaming_(nk_u4_k, a, a_stride, b_packed, c, rows, columns, depth, c_stride);
    nk_euclideans_packed_u4_sme_finalize_ssve_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                               c_stride_elements);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_angulars_symmetric_u4_sme_finalize_ssve_( //
    nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u4_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_u32_x(
                    predicate_b32x, svld1_u32(predicate_b32x, (nk_u32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_b32x, svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_u4_sme( //
    nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(nk_u4_k, vectors, stride, vectors_count, depth, result, result_stride,
                                        row_start, row_count);
    nk_angulars_symmetric_u4_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_u4_sme_finalize_ssve_( //
    nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
    nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) NUMKONG_STREAMING_ {
    // cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u4_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_b32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_u32_x(
                    predicate_b32x, svld1_u32(predicate_b32x, (nk_u32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_b32x, svld1_u32(predicate_b32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_b32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_b32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_u4_sme( //
    nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride, nk_f32_t *result,
    nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_sme_start_streaming_();
    nk_dots_symmetric_b8_sme_streaming_(nk_u4_k, vectors, stride, vectors_count, depth, result, result_stride,
                                        row_start, row_count);
    nk_euclideans_symmetric_u4_sme_finalize_ssve_(vectors, vectors_count, depth, stride_elements, result,
                                                  result_stride_elements, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

#pragma endregion Unsigned Integers

NUMKONG_API nk_status_t nk_angulars_packed_nvfp4_sme(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const status = nk_dots_scaled_packed_sme_(nk_nvfp4_k, a, b_packed, c, rows, columns, depth, a_stride,
                                                          c_stride);
    if (status != nk_success_k) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_nvfp4_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_angular_k, nk_nvfp4_k, operand, b_packed, c, rows, columns,
                                                 depth, a_stride, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_nvfp4_sme(nk_nvfp4_cref_t const *vectors, nk_size_t count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < count ? nk_min_of_two(row_count, count - row_start) : 0;
    nk_status_t const status = nk_dots_scaled_symmetric_sme_(nk_nvfp4_k, vectors, count, depth, stride, result,
                                                             result_stride, row_start, row_count);
    if (status != nk_success_k || !row_count) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_nvfp4_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_angular_k, nk_nvfp4_k, operand, count, depth, stride,
                                                    result, result_stride, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_nvfp4_sme(nk_nvfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const status = nk_dots_scaled_packed_sme_(nk_nvfp4_k, a, b_packed, c, rows, columns, depth, a_stride,
                                                          c_stride);
    if (status != nk_success_k) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_nvfp4_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_euclidean_k, nk_nvfp4_k, operand, b_packed, c, rows, columns,
                                                 depth, a_stride, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_nvfp4_sme(nk_nvfp4_cref_t const *vectors, nk_size_t count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < count ? nk_min_of_two(row_count, count - row_start) : 0;
    nk_status_t const status = nk_dots_scaled_symmetric_sme_(nk_nvfp4_k, vectors, count, depth, stride, result,
                                                             result_stride, row_start, row_count);
    if (status != nk_success_k || !row_count) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_nvfp4_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_euclidean_k, nk_nvfp4_k, operand, count, depth, stride,
                                                    result, result_stride, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp4_sme(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const status = nk_dots_scaled_packed_sme_(nk_mxfp4_k, a, b_packed, c, rows, columns, depth, a_stride,
                                                          c_stride);
    if (status != nk_success_k) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp4_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_angular_k, nk_mxfp4_k, operand, b_packed, c, rows, columns,
                                                 depth, a_stride, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp4_sme(nk_mxfp4_cref_t const *vectors, nk_size_t count,
                                                        nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                        nk_size_t result_stride, nk_size_t row_start,
                                                        nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < count ? nk_min_of_two(row_count, count - row_start) : 0;
    nk_status_t const status = nk_dots_scaled_symmetric_sme_(nk_mxfp4_k, vectors, count, depth, stride, result,
                                                             result_stride, row_start, row_count);
    if (status != nk_success_k || !row_count) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp4_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_angular_k, nk_mxfp4_k, operand, count, depth, stride,
                                                    result, result_stride, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp4_sme(nk_mxfp4_cref_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const status = nk_dots_scaled_packed_sme_(nk_mxfp4_k, a, b_packed, c, rows, columns, depth, a_stride,
                                                          c_stride);
    if (status != nk_success_k) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp4_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_euclidean_k, nk_mxfp4_k, operand, b_packed, c, rows, columns,
                                                 depth, a_stride, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp4_sme(nk_mxfp4_cref_t const *vectors, nk_size_t count,
                                                          nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                          nk_size_t result_stride, nk_size_t row_start,
                                                          nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < count ? nk_min_of_two(row_count, count - row_start) : 0;
    nk_status_t const status = nk_dots_scaled_symmetric_sme_(nk_mxfp4_k, vectors, count, depth, stride, result,
                                                             result_stride, row_start, row_count);
    if (status != nk_success_k || !row_count) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp4_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_euclidean_k, nk_mxfp4_k, operand, count, depth, stride,
                                                    result, result_stride, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const status = nk_dots_scaled_packed_sme_(nk_mxfp6e2m3_k, a, b_packed, c, rows, columns, depth,
                                                          a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e2m3_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_angular_k, nk_mxfp6e2m3_k, operand, b_packed, c, rows,
                                                 columns, depth, a_stride, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < count ? nk_min_of_two(row_count, count - row_start) : 0;
    nk_status_t const status = nk_dots_scaled_symmetric_sme_(nk_mxfp6e2m3_k, vectors, count, depth, stride, result,
                                                             result_stride, row_start, row_count);
    if (status != nk_success_k || !row_count) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e2m3_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_angular_k, nk_mxfp6e2m3_k, operand, count, depth, stride,
                                                    result, result_stride, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const status = nk_dots_scaled_packed_sme_(nk_mxfp6e2m3_k, a, b_packed, c, rows, columns, depth,
                                                          a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e2m3_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_euclidean_k, nk_mxfp6e2m3_k, operand, b_packed, c, rows,
                                                 columns, depth, a_stride, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e2m3_sme(nk_mxfp6e2m3_cref_t const *vectors, nk_size_t count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < count ? nk_min_of_two(row_count, count - row_start) : 0;
    nk_status_t const status = nk_dots_scaled_symmetric_sme_(nk_mxfp6e2m3_k, vectors, count, depth, stride, result,
                                                             result_stride, row_start, row_count);
    if (status != nk_success_k || !row_count) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e2m3_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_euclidean_k, nk_mxfp6e2m3_k, operand, count, depth,
                                                    stride, result, result_stride, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const status = nk_dots_scaled_packed_sme_(nk_mxfp6e3m2_k, a, b_packed, c, rows, columns, depth,
                                                          a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e3m2_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_angular_k, nk_mxfp6e3m2_k, operand, b_packed, c, rows,
                                                 columns, depth, a_stride, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < count ? nk_min_of_two(row_count, count - row_start) : 0;
    nk_status_t const status = nk_dots_scaled_symmetric_sme_(nk_mxfp6e3m2_k, vectors, count, depth, stride, result,
                                                             result_stride, row_start, row_count);
    if (status != nk_success_k || !row_count) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e3m2_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_angular_k, nk_mxfp6e3m2_k, operand, count, depth, stride,
                                                    result, result_stride, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const status = nk_dots_scaled_packed_sme_(nk_mxfp6e3m2_k, a, b_packed, c, rows, columns, depth,
                                                          a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e3m2_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_euclidean_k, nk_mxfp6e3m2_k, operand, b_packed, c, rows,
                                                 columns, depth, a_stride, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp6e3m2_sme(nk_mxfp6e3m2_cref_t const *vectors, nk_size_t count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < count ? nk_min_of_two(row_count, count - row_start) : 0;
    nk_status_t const status = nk_dots_scaled_symmetric_sme_(nk_mxfp6e3m2_k, vectors, count, depth, stride, result,
                                                             result_stride, row_start, row_count);
    if (status != nk_success_k || !row_count) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp6e3m2_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_euclidean_k, nk_mxfp6e3m2_k, operand, count, depth,
                                                    stride, result, result_stride, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const status = nk_dots_scaled_packed_sme_(nk_mxfp8e4m3_k, a, b_packed, c, rows, columns, depth,
                                                          a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e4m3_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_angular_k, nk_mxfp8e4m3_k, operand, b_packed, c, rows,
                                                 columns, depth, a_stride, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < count ? nk_min_of_two(row_count, count - row_start) : 0;
    nk_status_t const status = nk_dots_scaled_symmetric_sme_(nk_mxfp8e4m3_k, vectors, count, depth, stride, result,
                                                             result_stride, row_start, row_count);
    if (status != nk_success_k || !row_count) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e4m3_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_angular_k, nk_mxfp8e4m3_k, operand, count, depth, stride,
                                                    result, result_stride, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const status = nk_dots_scaled_packed_sme_(nk_mxfp8e4m3_k, a, b_packed, c, rows, columns, depth,
                                                          a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e4m3_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_euclidean_k, nk_mxfp8e4m3_k, operand, b_packed, c, rows,
                                                 columns, depth, a_stride, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e4m3_sme(nk_mxfp8e4m3_cref_t const *vectors, nk_size_t count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < count ? nk_min_of_two(row_count, count - row_start) : 0;
    nk_status_t const status = nk_dots_scaled_symmetric_sme_(nk_mxfp8e4m3_k, vectors, count, depth, stride, result,
                                                             result_stride, row_start, row_count);
    if (status != nk_success_k || !row_count) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e4m3_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_euclidean_k, nk_mxfp8e4m3_k, operand, count, depth,
                                                    stride, result, result_stride, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_packed_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                         nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                         nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                         void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const status = nk_dots_scaled_packed_sme_(nk_mxfp8e5m2_k, a, b_packed, c, rows, columns, depth,
                                                          a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e5m2_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_angular_k, nk_mxfp8e5m2_k, operand, b_packed, c, rows,
                                                 columns, depth, a_stride, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_angulars_symmetric_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t count,
                                                            nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                            nk_size_t result_stride, nk_size_t row_start,
                                                            nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < count ? nk_min_of_two(row_count, count - row_start) : 0;
    nk_status_t const status = nk_dots_scaled_symmetric_sme_(nk_mxfp8e5m2_k, vectors, count, depth, stride, result,
                                                             result_stride, row_start, row_count);
    if (status != nk_success_k || !row_count) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e5m2_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_angular_k, nk_mxfp8e5m2_k, operand, count, depth, stride,
                                                    result, result_stride, row_start, row_count);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_packed_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *a, void const *b_packed,
                                                           nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t depth, nk_size_t a_stride, nk_size_t c_stride,
                                                           void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_status_t const status = nk_dots_scaled_packed_sme_(nk_mxfp8e5m2_k, a, b_packed, c, rows, columns, depth,
                                                          a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e5m2_k, a, a_stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_packed_sme_finalize_ssve_(nk_kernel_euclidean_k, nk_mxfp8e5m2_k, operand, b_packed, c, rows,
                                                 columns, depth, a_stride, c_stride);
    nk_sme_stop_streaming_();
    return nk_success_k;
}
NUMKONG_API nk_status_t nk_euclideans_symmetric_mxfp8e5m2_sme(nk_mxfp8e5m2_cref_t const *vectors, nk_size_t count,
                                                              nk_size_t depth, nk_size_t stride, nk_f32_t *result,
                                                              nk_size_t result_stride, nk_size_t row_start,
                                                              nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    row_count = row_start < count ? nk_min_of_two(row_count, count - row_start) : 0;
    nk_status_t const status = nk_dots_scaled_symmetric_sme_(nk_mxfp8e5m2_k, vectors, count, depth, stride, result,
                                                             result_stride, row_start, row_count);
    if (status != nk_success_k || !row_count) return status;
    nk_cross_operand_t const operand = nk_cross_operand_(nk_mxfp8e5m2_k, vectors, stride);
    nk_sme_start_streaming_();
    nk_spatials_scaled_symmetric_sme_finalize_ssve_(nk_kernel_euclidean_k, nk_mxfp8e5m2_k, operand, count, depth,
                                                    stride, result, result_stride, row_start, row_count);
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
