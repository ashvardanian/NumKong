/**
 *  @brief Batched Spatial Distances for ARM SME.
 *  @file include/numkong/spatials/sme.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_SME_H
#define NK_SPATIALS_SME_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SME

#include "numkong/dots/serial.h"
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

NK_PUBLIC nk_f32_t nk_dots_reduce_sumsq_f16_ssve_(nk_f16_t const *data, nk_size_t count) NK_STREAMING_ {
    svfloat32_t accumulator_even_f32x = svdup_f32(0.0f);
    svfloat32_t accumulator_odd_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    nk_size_t const half_vector_length = svcntw();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b16x = svwhilelt_b16_u64(i, count);
        svfloat16_t values_f16x = svld1_f16(predicate_b16x, (nk_f16_for_arm_simd_t const *)(data + i));

        svbool_t predicate_even_b32x = svwhilelt_b32_u64(i, count);
        svfloat32_t values_even_f32x = svcvt_f32_f16_x(predicate_even_b32x, values_f16x);
        accumulator_even_f32x = svmla_f32_m(predicate_even_b32x, accumulator_even_f32x, values_even_f32x,
                                            values_even_f32x);

        svbool_t predicate_odd_b32x = svwhilelt_b32_u64(i + half_vector_length, count);
        svfloat32_t values_odd_f32x = svcvtlt_f32_f16_x(predicate_odd_b32x, values_f16x);
        accumulator_odd_f32x = svmla_f32_m(predicate_odd_b32x, accumulator_odd_f32x, values_odd_f32x, values_odd_f32x);
    }
    nk_f32_t sum_even = svaddv_f32(svptrue_b32(), accumulator_even_f32x);
    nk_f32_t sum_odd = svaddv_f32(svptrue_b32(), accumulator_odd_f32x);
    NK_UNPOISON(&sum_even, sizeof(sum_even));
    NK_UNPOISON(&sum_odd, sizeof(sum_odd));
    return sum_even + sum_odd;
}

NK_PUBLIC nk_f32_t nk_dots_reduce_sumsq_bf16_ssve_(nk_bf16_t const *data, nk_size_t count) NK_STREAMING_ {
    svfloat32_t accumulator_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b16x = svwhilelt_b16_u64(i, count);
        svbfloat16_t values_bf16x = svld1_bf16(predicate_b16x, (nk_bf16_for_arm_simd_t const *)(data + i));
        accumulator_f32x = svbfdot_f32(accumulator_f32x, values_bf16x, values_bf16x);
    }
    nk_f32_t sum = svaddv_f32(svptrue_b32(), accumulator_f32x);
    NK_UNPOISON(&sum, sizeof(sum));
    return sum;
}

NK_PUBLIC nk_f32_t nk_dots_reduce_sumsq_e4m3_ssve_(nk_e4m3_t const *data, nk_size_t count) NK_STREAMING_ {
    svfloat32_t accumulator_even_f32x = svdup_f32(0.0f);
    svfloat32_t accumulator_odd_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    nk_size_t const half_vector_length = svcntw();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        nk_size_t const batch_size = (i + vector_length < count) ? vector_length : (count - i);
        svbool_t predicate_b8x = svwhilelt_b8_u64(0u, batch_size);
        svbool_t predicate_b16x = svwhilelt_b16_u64(0u, batch_size);
        svuint8_t raw_u8x = svld1_u8(predicate_b8x, (nk_u8_t const *)data + i);
        svfloat16_t values_f16x = nk_e4m3x_to_f16x_ssve_(predicate_b16x, raw_u8x);

        svbool_t predicate_even_b32x = svwhilelt_b32_u64(0u, batch_size);
        svfloat32_t values_even_f32x = svcvt_f32_f16_x(predicate_even_b32x, values_f16x);
        accumulator_even_f32x = svmla_f32_m(predicate_even_b32x, accumulator_even_f32x, values_even_f32x,
                                            values_even_f32x);

        svbool_t predicate_odd_b32x = svwhilelt_b32_u64(half_vector_length, batch_size);
        svfloat32_t values_odd_f32x = svcvtlt_f32_f16_x(predicate_odd_b32x, values_f16x);
        accumulator_odd_f32x = svmla_f32_m(predicate_odd_b32x, accumulator_odd_f32x, values_odd_f32x, values_odd_f32x);
    }
    nk_f32_t sum_even = svaddv_f32(svptrue_b32(), accumulator_even_f32x);
    nk_f32_t sum_odd = svaddv_f32(svptrue_b32(), accumulator_odd_f32x);
    NK_UNPOISON(&sum_even, sizeof(sum_even));
    NK_UNPOISON(&sum_odd, sizeof(sum_odd));
    return sum_even + sum_odd;
}

NK_PUBLIC nk_f32_t nk_dots_reduce_sumsq_e5m2_ssve_(nk_e5m2_t const *data, nk_size_t count) NK_STREAMING_ {
    svfloat32_t accumulator_even_f32x = svdup_f32(0.0f);
    svfloat32_t accumulator_odd_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    nk_size_t const half_vector_length = svcntw();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        nk_size_t const batch_size = (i + vector_length < count) ? vector_length : (count - i);
        svbool_t predicate_b8x = svwhilelt_b8_u64(0u, batch_size);
        svbool_t predicate_b16x = svwhilelt_b16_u64(0u, batch_size);
        svuint8_t raw_u8x = svld1_u8(predicate_b8x, (nk_u8_t const *)data + i);
        svfloat16_t values_f16x = nk_e5m2x_to_f16x_ssve_(predicate_b16x, raw_u8x);

        svbool_t predicate_even_b32x = svwhilelt_b32_u64(0u, batch_size);
        svfloat32_t values_even_f32x = svcvt_f32_f16_x(predicate_even_b32x, values_f16x);
        accumulator_even_f32x = svmla_f32_m(predicate_even_b32x, accumulator_even_f32x, values_even_f32x,
                                            values_even_f32x);

        svbool_t predicate_odd_b32x = svwhilelt_b32_u64(half_vector_length, batch_size);
        svfloat32_t values_odd_f32x = svcvtlt_f32_f16_x(predicate_odd_b32x, values_f16x);
        accumulator_odd_f32x = svmla_f32_m(predicate_odd_b32x, accumulator_odd_f32x, values_odd_f32x, values_odd_f32x);
    }
    nk_f32_t sum_even = svaddv_f32(svptrue_b32(), accumulator_even_f32x);
    nk_f32_t sum_odd = svaddv_f32(svptrue_b32(), accumulator_odd_f32x);
    NK_UNPOISON(&sum_even, sizeof(sum_even));
    NK_UNPOISON(&sum_odd, sizeof(sum_odd));
    return sum_even + sum_odd;
}

NK_PUBLIC nk_f32_t nk_dots_reduce_sumsq_e2m3_ssve_(nk_e2m3_t const *data, nk_size_t count) NK_STREAMING_ {
    svint32_t accumulator_i32x = svdup_s32(0);
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, count);
        svuint8_t raw_u8x = svld1_u8(predicate_b8x, (nk_u8_t const *)data + i);
        svint8_t values_i8x = nk_e2m3x_to_i8x_ssve_(predicate_b8x, raw_u8x);
        accumulator_i32x = svdot_s32(accumulator_i32x, values_i8x, values_i8x);
    }
    nk_i64_t sum_i64 = svaddv_s32(svptrue_b32(), accumulator_i32x);
    NK_UNPOISON(&sum_i64, sizeof(sum_i64));
    return (nk_f32_t)sum_i64 / 256.0f;
}

NK_PUBLIC nk_f32_t nk_dots_reduce_sumsq_e3m2_ssve_(nk_e3m2_t const *data, nk_size_t count) NK_STREAMING_ {
    svfloat32_t accumulator_even_f32x = svdup_f32(0.0f);
    svfloat32_t accumulator_odd_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    nk_size_t const half_vector_length = svcntw();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        nk_size_t const batch_size = (i + vector_length < count) ? vector_length : (count - i);
        svbool_t predicate_b8x = svwhilelt_b8_u64(0u, batch_size);
        svbool_t predicate_b16x = svwhilelt_b16_u64(0u, batch_size);
        svuint8_t raw_u8x = svld1_u8(predicate_b8x, (nk_u8_t const *)data + i);
        svfloat16_t values_f16x = nk_e3m2x_to_f16x_ssve_(predicate_b16x, raw_u8x);

        svbool_t predicate_even_b32x = svwhilelt_b32_u64(0u, batch_size);
        svfloat32_t values_even_f32x = svcvt_f32_f16_x(predicate_even_b32x, values_f16x);
        accumulator_even_f32x = svmla_f32_m(predicate_even_b32x, accumulator_even_f32x, values_even_f32x,
                                            values_even_f32x);

        svbool_t predicate_odd_b32x = svwhilelt_b32_u64(half_vector_length, batch_size);
        svfloat32_t values_odd_f32x = svcvtlt_f32_f16_x(predicate_odd_b32x, values_f16x);
        accumulator_odd_f32x = svmla_f32_m(predicate_odd_b32x, accumulator_odd_f32x, values_odd_f32x, values_odd_f32x);
    }
    nk_f32_t sum_even = svaddv_f32(svptrue_b32(), accumulator_even_f32x);
    nk_f32_t sum_odd = svaddv_f32(svptrue_b32(), accumulator_odd_f32x);
    NK_UNPOISON(&sum_even, sizeof(sum_even));
    NK_UNPOISON(&sum_odd, sizeof(sum_odd));
    return sum_even + sum_odd;
}

NK_PUBLIC nk_u32_t nk_dots_reduce_sumsq_i8_ssve_(nk_i8_t const *data, nk_size_t count) NK_STREAMING_ {
    svint32_t accumulator_i32x = svdup_s32(0);
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, count);
        svint8_t loaded_i8x = svld1_s8(predicate_b8x, data + i);
        accumulator_i32x = svdot_s32(accumulator_i32x, loaded_i8x, loaded_i8x);
    }
    nk_i64_t sum_i64 = svaddv_s32(svptrue_b32(), accumulator_i32x);
    NK_UNPOISON(&sum_i64, sizeof(sum_i64));
    return (nk_u32_t)sum_i64;
}

NK_PUBLIC nk_u32_t nk_dots_reduce_sumsq_u8_ssve_(nk_u8_t const *data, nk_size_t count) NK_STREAMING_ {
    svuint32_t accumulator_u32x = svdup_u32(0);
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, count);
        svuint8_t loaded_u8x = svld1_u8(predicate_b8x, data + i);
        accumulator_u32x = svdot_u32(accumulator_u32x, loaded_u8x, loaded_u8x);
    }
    nk_u64_t sum_u64 = svaddv_u32(svptrue_b32(), accumulator_u32x);
    NK_UNPOISON(&sum_u64, sizeof(sum_u64));
    return (nk_u32_t)sum_u64;
}

NK_PUBLIC nk_u32_t nk_dots_reduce_sumsq_i4_ssve_(nk_i4x2_t const *data, nk_size_t count) NK_STREAMING_ {
    svint32_t accumulator_i32x = svdup_s32(0);
    nk_u8_t const *bytes = (nk_u8_t const *)data;
    nk_size_t const byte_count = (count + 1) / 2;
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
    nk_i64_t sum_i64 = svaddv_s32(svptrue_b32(), accumulator_i32x);
    NK_UNPOISON(&sum_i64, sizeof(sum_i64));
    return (nk_u32_t)sum_i64;
}

NK_PUBLIC nk_u32_t nk_dots_reduce_sumsq_u4_ssve_(nk_u4x2_t const *data, nk_size_t count) NK_STREAMING_ {
    svuint32_t accumulator_u32x = svdup_u32(0);
    nk_u8_t const *bytes = (nk_u8_t const *)data;
    nk_size_t const byte_count = (count + 1) / 2;
    nk_size_t const vector_length = svcntb();
    for (nk_size_t i = 0; i < byte_count; i += vector_length) {
        svbool_t predicate_b8x = svwhilelt_b8_u64(i, byte_count);
        svuint8_t packed_u8x = svld1_u8(predicate_b8x, bytes + i);
        svuint8_t low_u8x = svand_n_u8_x(predicate_b8x, packed_u8x, 0x0F);
        svuint8_t high_u8x = svlsr_n_u8_x(predicate_b8x, packed_u8x, 4);
        accumulator_u32x = svdot_u32(accumulator_u32x, low_u8x, low_u8x);
        accumulator_u32x = svdot_u32(accumulator_u32x, high_u8x, high_u8x);
    }
    nk_u64_t sum_u64 = svaddv_u32(svptrue_b32(), accumulator_u32x);
    NK_UNPOISON(&sum_u64, sizeof(sum_u64));
    return (nk_u32_t)sum_u64;
}

NK_PUBLIC svfloat32_t nk_angulars_from_dot_f32x_ssve_(svbool_t predicate_b32x, svfloat32_t dots_f32x,
                                                      svfloat32_t query_norm_sq_f32x,
                                                      svfloat32_t target_norms_sq_f32x) NK_STREAMING_ {
    svfloat32_t norms_product_f32x = svmul_f32_x(predicate_b32x, query_norm_sq_f32x, target_norms_sq_f32x);
    svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
    rsqrt_f32x = svmul_f32_x(predicate_b32x, rsqrt_f32x,
                             svrsqrts_f32(svmul_f32_x(predicate_b32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
    rsqrt_f32x = svmul_f32_x(predicate_b32x, rsqrt_f32x,
                             svrsqrts_f32(svmul_f32_x(predicate_b32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
    svfloat32_t angular_f32x = svsub_f32_x(predicate_b32x, svdup_n_f32(1.0f),
                                           svmul_f32_x(predicate_b32x, dots_f32x, rsqrt_f32x));
    return svmax_f32_x(predicate_b32x, angular_f32x, svdup_n_f32(0.0f));
}

NK_PUBLIC svfloat32_t nk_euclideans_from_dot_f32x_ssve_(svbool_t predicate_b32x, svfloat32_t dots_f32x,
                                                        svfloat32_t query_norm_sq_f32x,
                                                        svfloat32_t target_norms_sq_f32x) NK_STREAMING_ {
    svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_b32x, query_norm_sq_f32x, target_norms_sq_f32x);
    svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_b32x, sum_sq_f32x,
                                           svmul_f32_x(predicate_b32x, svdup_n_f32(2.0f), dots_f32x));
    dist_sq_f32x = svmax_f32_x(predicate_b32x, dist_sq_f32x, svdup_n_f32(0.0f));
    return svsqrt_f32_x(predicate_b32x, dist_sq_f32x);
}

#pragma region F16 Floats

__arm_locally_streaming static void nk_angulars_packed_f16_sme_finalize_streaming_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                             //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_angulars_packed_f16_sme(                //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_f16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_f16_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                   c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_f16_sme_finalize_streaming_( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                               //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_euclideans_packed_f16_sme(              //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_f16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_f16_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                     c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_f16_sme_finalize_streaming_(            //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_f16_sme(                                                     //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_sme_streaming_(vectors, vectors_count, depth, stride_elements, result, result_stride_elements,
                                         row_start, row_count);
    nk_angulars_symmetric_f16_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                      result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_f16_sme_finalize_streaming_(          //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_f16_sme(                                                   //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_sme_streaming_(vectors, vectors_count, depth, stride_elements, result, result_stride_elements,
                                         row_start, row_count);
    nk_euclideans_symmetric_f16_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                        result_stride_elements, row_start, row_count);
}

#pragma endregion F16 Floats

#pragma region BF16 Floats

__arm_locally_streaming static void nk_angulars_packed_bf16_sme_finalize_streaming_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_angulars_packed_bf16_sme(                //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_bf16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_bf16_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_bf16_sme_finalize_streaming_( //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_euclideans_packed_bf16_sme(              //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_bf16_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_bf16_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_bf16_sme_finalize_streaming_(            //
    nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_bf16_sme(                                                     //
    nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_sme_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                          result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_bf16_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_bf16_sme_finalize_streaming_(          //
    nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_bf16_sme(                                                   //
    nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_sme_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                          result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_bf16_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion BF16 Floats

#pragma region E4M3 Floats

__arm_locally_streaming static void nk_angulars_packed_e4m3_sme_finalize_streaming_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_angulars_packed_e4m3_sme(                //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e4m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_e4m3_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_e4m3_sme_finalize_streaming_( //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_euclideans_packed_e4m3_sme(              //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e4m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_e4m3_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_e4m3_sme_finalize_streaming_(            //
    nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e4m3_sme(                                                     //
    nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_sme_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                          result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_e4m3_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_e4m3_sme_finalize_streaming_(          //
    nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e4m3_sme(                                                   //
    nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_sme_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                          result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_e4m3_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion E4M3 Floats

#pragma region E5M2 Floats

__arm_locally_streaming static void nk_angulars_packed_e5m2_sme_finalize_streaming_( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_angulars_packed_e5m2_sme(                //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e5m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_e5m2_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_e5m2_sme_finalize_streaming_( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_euclideans_packed_e5m2_sme(              //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e5m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_e5m2_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_e5m2_sme_finalize_streaming_(            //
    nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e5m2_sme(                                                     //
    nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_sme_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                          result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_e5m2_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_e5m2_sme_finalize_streaming_(          //
    nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e5m2_sme(                                                   //
    nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_sme_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                          result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_e5m2_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion E5M2 Floats

#pragma region E2M3 Floats

__arm_locally_streaming static void nk_angulars_packed_e2m3_sme_finalize_streaming_( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_angulars_packed_e2m3_sme(                //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e2m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_e2m3_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_e2m3_sme_finalize_streaming_( //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_euclideans_packed_e2m3_sme(              //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e2m3_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_e2m3_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_e2m3_sme_finalize_streaming_(            //
    nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e2m3_sme(                                                     //
    nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_sme_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                          result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_e2m3_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_e2m3_sme_finalize_streaming_(          //
    nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e2m3_sme(                                                   //
    nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_sme_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                          result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_e2m3_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion E2M3 Floats

#pragma region E3M2 Floats

__arm_locally_streaming static void nk_angulars_packed_e3m2_sme_finalize_streaming_( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_angulars_packed_e3m2_sme(                //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e3m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_e3m2_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_e3m2_sme_finalize_streaming_( //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_euclideans_packed_e3m2_sme(              //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e3m2_sme_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_e3m2_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_e3m2_sme_finalize_streaming_(            //
    nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e3m2_sme(                                                     //
    nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_sme_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                          result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_e3m2_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_e3m2_sme_finalize_streaming_(          //
    nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e3m2_sme(                                                   //
    nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_sme_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                          result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_e3m2_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion E3M2 Floats
#pragma region I8 Integers

__arm_locally_streaming static void nk_angulars_packed_i8_sme_finalize_streaming_( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_angulars_packed_i8_sme(                //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_i8_sme_streaming_(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_angulars_packed_i8_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_i8_sme_finalize_streaming_( //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_euclideans_packed_i8_sme(              //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_i8_sme_streaming_(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_euclideans_packed_i8_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_i8_sme_finalize_streaming_(            //
    nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_i8_sme(                                                     //
    nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_i8_sme_streaming_(vectors, vectors_count, depth, stride_elements, (nk_i32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_i8_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_i8_sme_finalize_streaming_(          //
    nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_i8_sme(                                                   //
    nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_i8_sme_streaming_(vectors, vectors_count, depth, stride_elements, (nk_i32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_i8_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion I8 Integers

#pragma region U8 Integers

__arm_locally_streaming static void nk_angulars_packed_u8_sme_finalize_streaming_( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_angulars_packed_u8_sme(                //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_u8_sme_streaming_(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_angulars_packed_u8_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_u8_sme_finalize_streaming_( //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c,                             //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_euclideans_packed_u8_sme(              //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_u8_sme_streaming_(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_euclideans_packed_u8_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_u8_sme_finalize_streaming_(            //
    nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_u8_sme(                                                     //
    nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_u8_sme_streaming_(vectors, vectors_count, depth, stride_elements, (nk_u32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_u8_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_u8_sme_finalize_streaming_(          //
    nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_u8_sme(                                                   //
    nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_u8_sme_streaming_(vectors, vectors_count, depth, stride_elements, (nk_u32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_u8_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion U8 Integers

#pragma region I4 Integers

__arm_locally_streaming static void nk_angulars_packed_i4_sme_finalize_streaming_( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,                         //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_angulars_packed_i4_sme(                  //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i4x2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_i4_sme_streaming_(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_angulars_packed_i4_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_i4_sme_finalize_streaming_( //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_euclideans_packed_i4_sme(                //
    nk_i4x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i4x2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_i4_sme_streaming_(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_euclideans_packed_i4_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_i4_sme_finalize_streaming_(              //
    nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_i4_sme(                                                       //
    nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_i4_sme_streaming_(vectors, vectors_count, depth, stride_elements, (nk_i32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_i4_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_i4_sme_finalize_streaming_(            //
    nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_i4_sme(                                                     //
    nk_i4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_i4_sme_streaming_(vectors, vectors_count, depth, stride_elements, (nk_i32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_i4_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion Signed Integers

#pragma region U4 Integers

__arm_locally_streaming static void nk_angulars_packed_u4_sme_finalize_streaming_( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,                         //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_angulars_packed_u4_sme(                  //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u4x2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_u4_sme_streaming_(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_angulars_packed_u4_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

__arm_locally_streaming static void nk_euclideans_packed_u4_sme_finalize_streaming_( //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c,                           //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                              //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
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

NK_PUBLIC void nk_euclideans_packed_u4_sme(                //
    nk_u4x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u4x2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_u4_sme_streaming_(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride_elements,
                                     c_stride_elements);
    nk_euclideans_packed_u4_sme_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

__arm_locally_streaming static void nk_angulars_symmetric_u4_sme_finalize_streaming_(              //
    nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_u4_sme(                                                       //
    nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_u4_sme_streaming_(vectors, vectors_count, depth, stride_elements, (nk_u32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_u4_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_u4_sme_finalize_streaming_(            //
    nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
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
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_u4_sme(                                                     //
    nk_u4x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_u4_sme_streaming_(vectors, vectors_count, depth, stride_elements, (nk_u32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_u4_sme_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion Unsigned Integers

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM64_
#endif // NK_SPATIALS_SME_H
