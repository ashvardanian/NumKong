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

#if NK_TARGET_ARM_
#if NK_TARGET_SME

#include "numkong/dots/serial.h"
#include "numkong/dots/sme.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme,sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme")
#endif

NK_PUBLIC nk_f32_t nk_dots_reduce_sumsq_f16_ssve_(nk_f16_t const *data, nk_size_t count) NK_STREAMING_COMPATIBLE_ {
    svfloat32_t accumulator_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcntw();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_f32x = svwhilelt_b32_u64(i, count);
        svfloat32_t values_f32x = svcvt_f32_f16_x(
            predicate_f32x, svld1_f16(svwhilelt_b16_u64(i, count), (nk_f16_for_arm_simd_t const *)(data + i)));
        accumulator_f32x = svmla_f32_x(predicate_f32x, accumulator_f32x, values_f32x, values_f32x);
    }
    return svaddv_f32(svptrue_b32(), accumulator_f32x);
}

NK_PUBLIC nk_f32_t nk_dots_reduce_sumsq_bf16_ssve_(nk_bf16_t const *data, nk_size_t count) NK_STREAMING_COMPATIBLE_ {
    svfloat32_t accumulator_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcntw();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_f32x = svwhilelt_b32_u64(i, count);
        svuint16_t raw_u16x = svld1_u16(svwhilelt_b16_u64(i, count), (nk_u16_t const *)data + i);
        svfloat32_t values_f32x = svreinterpret_f32_u32(svlsl_n_u32_x(predicate_f32x, svunpklo_u32(raw_u16x), 16));
        accumulator_f32x = svmla_f32_x(predicate_f32x, accumulator_f32x, values_f32x, values_f32x);
    }
    return svaddv_f32(svptrue_b32(), accumulator_f32x);
}

NK_PUBLIC nk_f32_t nk_dots_reduce_sumsq_e4m3_ssve_(nk_e4m3_t const *data, nk_size_t count) NK_STREAMING_ {
    svfloat32_t accumulator_lo_f32x = svdup_f32(0.0f);
    svfloat32_t accumulator_hi_f32x = svdup_f32(0.0f);
    svuint16_t subnorm_lut_u16x = svld1_u16(svwhilelt_b16(0u, 8u), nk_e4m3_subnorm_f16_lut_);
    nk_size_t const vector_length = svcnth();
    nk_size_t const half_vector_length = svcntw();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        nk_size_t const batch_size = (i + vector_length < count) ? vector_length : (count - i);
        svbool_t predicate_i8x = svwhilelt_b8_u64(0u, batch_size);
        svbool_t predicate_f16x = svwhilelt_b16_u64(0u, batch_size);
        svuint8_t raw_u8x = svld1_u8(predicate_i8x, (nk_u8_t const *)data + i);
        svfloat16_t values_f16x = nk_e4m3x_to_f16x_ssve_(predicate_f16x, raw_u8x, subnorm_lut_u16x);

        svbool_t predicate_lo_f32x = svwhilelt_b32_u64(0u, batch_size);
        svfloat32_t values_lo_f32x = svcvt_f32_f16_x(predicate_lo_f32x, values_f16x);
        accumulator_lo_f32x = svmla_f32_m(predicate_lo_f32x, accumulator_lo_f32x, values_lo_f32x, values_lo_f32x);

        svbool_t predicate_hi_f32x = svwhilelt_b32_u64(half_vector_length, batch_size);
        svfloat32_t values_hi_f32x = svcvtlt_f32_f16_x(predicate_hi_f32x, values_f16x);
        accumulator_hi_f32x = svmla_f32_m(predicate_hi_f32x, accumulator_hi_f32x, values_hi_f32x, values_hi_f32x);
    }
    return svaddv_f32(svptrue_b32(), accumulator_lo_f32x) + svaddv_f32(svptrue_b32(), accumulator_hi_f32x);
}

NK_PUBLIC nk_f32_t nk_dots_reduce_sumsq_e5m2_ssve_(nk_e5m2_t const *data, nk_size_t count) NK_STREAMING_ {
    svfloat32_t accumulator_lo_f32x = svdup_f32(0.0f);
    svfloat32_t accumulator_hi_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    nk_size_t const half_vector_length = svcntw();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        nk_size_t const batch_size = (i + vector_length < count) ? vector_length : (count - i);
        svbool_t predicate_i8x = svwhilelt_b8_u64(0u, batch_size);
        svbool_t predicate_f16x = svwhilelt_b16_u64(0u, batch_size);
        svuint8_t raw_u8x = svld1_u8(predicate_i8x, (nk_u8_t const *)data + i);
        svfloat16_t values_f16x = nk_e5m2x_to_f16x_ssve_(predicate_f16x, raw_u8x);

        svbool_t predicate_lo_f32x = svwhilelt_b32_u64(0u, batch_size);
        svfloat32_t values_lo_f32x = svcvt_f32_f16_x(predicate_lo_f32x, values_f16x);
        accumulator_lo_f32x = svmla_f32_m(predicate_lo_f32x, accumulator_lo_f32x, values_lo_f32x, values_lo_f32x);

        svbool_t predicate_hi_f32x = svwhilelt_b32_u64(half_vector_length, batch_size);
        svfloat32_t values_hi_f32x = svcvtlt_f32_f16_x(predicate_hi_f32x, values_f16x);
        accumulator_hi_f32x = svmla_f32_m(predicate_hi_f32x, accumulator_hi_f32x, values_hi_f32x, values_hi_f32x);
    }
    return svaddv_f32(svptrue_b32(), accumulator_lo_f32x) + svaddv_f32(svptrue_b32(), accumulator_hi_f32x);
}

NK_PUBLIC nk_f32_t nk_dots_reduce_sumsq_e2m3_ssve_(nk_e2m3_t const *data, nk_size_t count) NK_STREAMING_COMPATIBLE_ {
    svint64_t accumulator_i64x = svdup_s64(0);
    nk_size_t const vector_length = svcntd();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_i64x = svwhilelt_b64_u64(i, count);
        svuint8_t raw_u8x = svld1_u8(svwhilelt_b8_u64(i, count), (nk_u8_t const *)data + i);
        svint8_t values_i8x = nk_e2m3x_to_i8x_ssve_(svwhilelt_b8_u64(i, count), raw_u8x);
        svint16_t values_i16x = svunpklo_s16(values_i8x);
        svint16_t squares_i16x = svmul_s16_z(svwhilelt_b16_u64(i, count), values_i16x, values_i16x);
        svint64_t squares_i64x = svunpklo_s64(svunpklo_s32(squares_i16x));
        accumulator_i64x = svadd_s64_m(predicate_i64x, accumulator_i64x, squares_i64x);
    }
    return (nk_f32_t)svaddv_s64(svptrue_b64(), accumulator_i64x) / 256.0f;
}

NK_PUBLIC nk_f32_t nk_dots_reduce_sumsq_e3m2_ssve_(nk_e3m2_t const *data, nk_size_t count) NK_STREAMING_ {
    svfloat32_t accumulator_lo_f32x = svdup_f32(0.0f);
    svfloat32_t accumulator_hi_f32x = svdup_f32(0.0f);
    nk_size_t const vector_length = svcnth();
    nk_size_t const half_vector_length = svcntw();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        nk_size_t const batch_size = (i + vector_length < count) ? vector_length : (count - i);
        svbool_t predicate_i8x = svwhilelt_b8_u64(0u, batch_size);
        svbool_t predicate_f16x = svwhilelt_b16_u64(0u, batch_size);
        svuint8_t raw_u8x = svld1_u8(predicate_i8x, (nk_u8_t const *)data + i);
        svfloat16_t values_f16x = nk_e3m2x_to_f16x_ssve_(predicate_f16x, raw_u8x);

        svbool_t predicate_lo_f32x = svwhilelt_b32_u64(0u, batch_size);
        svfloat32_t values_lo_f32x = svcvt_f32_f16_x(predicate_lo_f32x, values_f16x);
        accumulator_lo_f32x = svmla_f32_m(predicate_lo_f32x, accumulator_lo_f32x, values_lo_f32x, values_lo_f32x);

        svbool_t predicate_hi_f32x = svwhilelt_b32_u64(half_vector_length, batch_size);
        svfloat32_t values_hi_f32x = svcvtlt_f32_f16_x(predicate_hi_f32x, values_f16x);
        accumulator_hi_f32x = svmla_f32_m(predicate_hi_f32x, accumulator_hi_f32x, values_hi_f32x, values_hi_f32x);
    }
    return svaddv_f32(svptrue_b32(), accumulator_lo_f32x) + svaddv_f32(svptrue_b32(), accumulator_hi_f32x);
}

NK_PUBLIC nk_u32_t nk_dots_reduce_sumsq_i8_ssve_(nk_i8_t const *data, nk_size_t count) NK_STREAMING_COMPATIBLE_ {
    svint64_t accumulator_i64x = svdup_s64(0);
    nk_size_t const vector_length = svcntd();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_i64x = svwhilelt_b64_u64(i, count);
        svint8_t loaded_i8x = svld1_s8(svwhilelt_b8_u64(i, count), data + i);
        svint16_t values_i16x = svunpklo_s16(loaded_i8x);
        svint16_t squares_i16x = svmul_s16_z(svwhilelt_b16_u64(i, count), values_i16x, values_i16x);
        svint64_t squares_i64x = svunpklo_s64(svunpklo_s32(squares_i16x));
        accumulator_i64x = svadd_s64_m(predicate_i64x, accumulator_i64x, squares_i64x);
    }
    return (nk_u32_t)svaddv_s64(svptrue_b64(), accumulator_i64x);
}

NK_PUBLIC nk_u32_t nk_dots_reduce_sumsq_u8_ssve_(nk_u8_t const *data, nk_size_t count) NK_STREAMING_COMPATIBLE_ {
    svuint64_t accumulator_u64x = svdup_u64(0);
    nk_size_t const vector_length = svcntd();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_u64x = svwhilelt_b64_u64(i, count);
        svuint8_t raw_u8x = svld1_u8(svwhilelt_b8_u64(i, count), data + i);
        svuint16_t values_u16x = svunpklo_u16(raw_u8x);
        svuint16_t squares_u16x = svmul_u16_z(svwhilelt_b16_u64(i, count), values_u16x, values_u16x);
        svuint64_t squares_u64x = svunpklo_u64(svunpklo_u32(squares_u16x));
        accumulator_u64x = svadd_u64_m(predicate_u64x, accumulator_u64x, squares_u64x);
    }
    return (nk_u32_t)svaddv_u64(svptrue_b64(), accumulator_u64x);
}

NK_PUBLIC nk_u32_t nk_dots_reduce_sumsq_i4_ssve_(nk_i4x2_t const *data, nk_size_t count) NK_STREAMING_COMPATIBLE_ {
    svint64_t accumulator_i64x = svdup_s64(0);
    nk_u8_t const *bytes = (nk_u8_t const *)data;
    nk_size_t const byte_count = (count + 1) / 2;
    nk_size_t const vector_length = svcntd();
    for (nk_size_t i = 0; i < byte_count; i += vector_length) {
        svbool_t predicate_u8x = svwhilelt_b8_u64(i, byte_count);
        svuint8_t packed_u8x = svld1_u8(predicate_u8x, bytes + i);
        svuint8_t low_u8x = svand_n_u8_x(predicate_u8x, packed_u8x, 0x0F);
        svuint8_t high_u8x = svlsr_n_u8_x(predicate_u8x, packed_u8x, 4);
        // Sign-extend 4-bit to 8-bit: shift left 4, arithmetic shift right 4
        svint8_t low_i8x = svasr_n_s8_x(predicate_u8x, svreinterpret_s8_u8(svlsl_n_u8_x(predicate_u8x, low_u8x, 4)), 4);
        svint8_t high_i8x = svasr_n_s8_x(predicate_u8x, svreinterpret_s8_u8(svlsl_n_u8_x(predicate_u8x, high_u8x, 4)),
                                         4);
        // Widen to i16, square, sum per byte
        svbool_t predicate_i16x = svwhilelt_b16_u64(i, byte_count);
        svint16_t low_i16x = svunpklo_s16(low_i8x);
        svint16_t high_i16x = svunpklo_s16(high_i8x);
        svint16_t squares_low_i16x = svmul_s16_z(predicate_i16x, low_i16x, low_i16x);
        svint16_t squares_high_i16x = svmul_s16_z(predicate_i16x, high_i16x, high_i16x);
        svint16_t sum_i16x = svadd_s16_z(predicate_i16x, squares_low_i16x, squares_high_i16x);
        svbool_t predicate_i64x = svwhilelt_b64_u64(i, byte_count);
        svint64_t sum_i64x = svunpklo_s64(svunpklo_s32(sum_i16x));
        accumulator_i64x = svadd_s64_m(predicate_i64x, accumulator_i64x, sum_i64x);
    }
    return (nk_u32_t)svaddv_s64(svptrue_b64(), accumulator_i64x);
}

NK_PUBLIC nk_u32_t nk_dots_reduce_sumsq_u4_ssve_(nk_u4x2_t const *data, nk_size_t count) NK_STREAMING_COMPATIBLE_ {
    svuint64_t accumulator_u64x = svdup_u64(0);
    nk_u8_t const *bytes = (nk_u8_t const *)data;
    nk_size_t const byte_count = (count + 1) / 2;
    nk_size_t const vector_length = svcntd();
    for (nk_size_t i = 0; i < byte_count; i += vector_length) {
        svbool_t predicate_u8x = svwhilelt_b8_u64(i, byte_count);
        svuint8_t packed_u8x = svld1_u8(predicate_u8x, bytes + i);
        svuint8_t low_u8x = svand_n_u8_x(predicate_u8x, packed_u8x, 0x0F);
        svuint8_t high_u8x = svlsr_n_u8_x(predicate_u8x, packed_u8x, 4);
        // Widen to u16, square, sum per byte
        svbool_t predicate_u16x = svwhilelt_b16_u64(i, byte_count);
        svuint16_t low_u16x = svunpklo_u16(low_u8x);
        svuint16_t high_u16x = svunpklo_u16(high_u8x);
        svuint16_t squares_low_u16x = svmul_u16_z(predicate_u16x, low_u16x, low_u16x);
        svuint16_t squares_high_u16x = svmul_u16_z(predicate_u16x, high_u16x, high_u16x);
        svuint16_t sum_u16x = svadd_u16_z(predicate_u16x, squares_low_u16x, squares_high_u16x);
        svbool_t predicate_u64x = svwhilelt_b64_u64(i, byte_count);
        svuint64_t sum_u64x = svunpklo_u64(svunpklo_u32(sum_u16x));
        accumulator_u64x = svadd_u64_m(predicate_u64x, accumulator_u64x, sum_u64x);
    }
    return (nk_u32_t)svaddv_u64(svptrue_b64(), accumulator_u64x);
}

NK_PUBLIC svfloat32_t nk_angulars_from_dot_f32x_ssve_(svbool_t predicate_f32x, svfloat32_t dots_f32x,
                                                      svfloat32_t query_norm_sq_f32x,
                                                      svfloat32_t target_norms_sq_f32x) NK_STREAMING_COMPATIBLE_ {
    svfloat32_t norms_product_f32x = svmul_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
    svfloat32_t rsqrt_f32x = svrsqrte_f32(norms_product_f32x);
    rsqrt_f32x = svmul_f32_x(predicate_f32x, rsqrt_f32x,
                             svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
    rsqrt_f32x = svmul_f32_x(predicate_f32x, rsqrt_f32x,
                             svrsqrts_f32(svmul_f32_x(predicate_f32x, norms_product_f32x, rsqrt_f32x), rsqrt_f32x));
    svfloat32_t angular_f32x = svsub_f32_x(predicate_f32x, svdup_n_f32(1.0f),
                                           svmul_f32_x(predicate_f32x, dots_f32x, rsqrt_f32x));
    return svmax_f32_x(predicate_f32x, angular_f32x, svdup_n_f32(0.0f));
}

NK_PUBLIC svfloat32_t nk_euclideans_from_dot_f32x_ssve_(svbool_t predicate_f32x, svfloat32_t dots_f32x,
                                                        svfloat32_t query_norm_sq_f32x,
                                                        svfloat32_t target_norms_sq_f32x) NK_STREAMING_COMPATIBLE_ {
    svfloat32_t sum_sq_f32x = svadd_f32_x(predicate_f32x, query_norm_sq_f32x, target_norms_sq_f32x);
    svfloat32_t dist_sq_f32x = svsub_f32_x(predicate_f32x, sum_sq_f32x,
                                           svmul_f32_x(predicate_f32x, svdup_n_f32(2.0f), dots_f32x));
    dist_sq_f32x = svmax_f32_x(predicate_f32x, dist_sq_f32x, svdup_n_f32(0.0f));
    return svsqrt_f32_x(predicate_f32x, dist_sq_f32x);
}

#pragma region Half Precision Floats

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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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

__arm_locally_streaming static void nk_angulars_symmetric_f16_sme_finalize_streaming_(        //
    nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_f16_sme(                                        //
    nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                         row_start, row_count);
    nk_angulars_symmetric_f16_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                      result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_f16_sme_finalize_streaming_(      //
    nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_f16_sme(                                      //
    nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                         row_start, row_count);
    nk_euclideans_symmetric_f16_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                        result_stride_elements, row_start, row_count);
}

#pragma endregion // Half Precision Floats

#pragma region Brain Float 16

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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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

__arm_locally_streaming static void nk_angulars_symmetric_bf16_sme_finalize_streaming_(        //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_bf16_sme(                                        //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_angulars_symmetric_bf16_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_bf16_sme_finalize_streaming_(      //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_bf16_sme(                                      //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_euclideans_symmetric_bf16_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion // Brain Float 16

#pragma region Quarter Precision E4M3

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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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

__arm_locally_streaming static void nk_angulars_symmetric_e4m3_sme_finalize_streaming_(        //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e4m3_sme(                                        //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_angulars_symmetric_e4m3_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_e4m3_sme_finalize_streaming_(      //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e4m3_sme(                                      //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_euclideans_symmetric_e4m3_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion // Quarter Precision E4M3

#pragma region Quarter Precision E5M2

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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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

__arm_locally_streaming static void nk_angulars_symmetric_e5m2_sme_finalize_streaming_(        //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e5m2_sme(                                        //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_angulars_symmetric_e5m2_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_e5m2_sme_finalize_streaming_(      //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e5m2_sme(                                      //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_euclideans_symmetric_e5m2_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion // Quarter Precision E5M2

#pragma region Micro Precision E2M3

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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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

__arm_locally_streaming static void nk_angulars_symmetric_e2m3_sme_finalize_streaming_(        //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e2m3_sme(                                        //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_angulars_symmetric_e2m3_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_e2m3_sme_finalize_streaming_(      //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e2m3_sme(                                      //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_euclideans_symmetric_e2m3_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion // Micro Precision E2M3

#pragma region Micro Precision E3M2

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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
            svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, b_norms + col_index);
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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

__arm_locally_streaming static void nk_angulars_symmetric_e3m2_sme_finalize_streaming_(        //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e3m2_sme(                                        //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_angulars_symmetric_e3m2_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_e3m2_sme_finalize_streaming_(      //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + row_index * stride_elements, depth);
    }
    // Phase 2: column-first post-processing
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            svfloat32_t query_norm_sq_f32x = svdup_n_f32(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svld1_f32(predicate_f32x, result_row + col_index);
                svfloat32_t target_norms_sq_f32x = svld1_f32(predicate_f32x, norms_cache + (col_index - chunk_start));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e3m2_sme(                                      //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_sme_streaming_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                          row_start, row_count);
    nk_euclideans_symmetric_e3m2_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion // Micro Precision E3M2
#pragma region Signed 8-bit Integers

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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_s32_x(
                predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_s32_x(
                predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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

__arm_locally_streaming static void nk_angulars_symmetric_i8_sme_finalize_streaming_(        //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_s32_x(
                    predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_i8_sme(                                        //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i8_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_i32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_i8_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_i8_sme_finalize_streaming_(      //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_s32_x(
                    predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_i8_sme(                                      //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i8_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_i32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_i8_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // Signed 8-bit Integers

#pragma region Unsigned 8-bit Integers

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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_u32_x(
                predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_u32_x(
                predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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

__arm_locally_streaming static void nk_angulars_symmetric_u8_sme_finalize_streaming_(        //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_u8_sme(                                        //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u8_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_u32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_u8_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_u8_sme_finalize_streaming_(      //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u8_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_u8_sme(                                      //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u8_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_u32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_u8_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // Unsigned 8-bit Integers

#pragma region Nibble Signed Integers

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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_s32_x(
                predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_s32_x(
                predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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

__arm_locally_streaming static void nk_angulars_symmetric_i4_sme_finalize_streaming_(          //
    nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i4_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_s32_x(
                    predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_i4_sme(                                          //
    nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i4_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_i32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_i4_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_i4_sme_finalize_streaming_(        //
    nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_i4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i4_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_s32_x(
                    predicate_f32x, svld1_s32(predicate_f32x, (nk_i32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_i4_sme(                                        //
    nk_i4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_i4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i4_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_i32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_i4_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // Nibble Signed Integers

#pragma region Nibble Unsigned Integers

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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_u32_x(
                predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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
            svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, columns);
            svfloat32_t dots_f32x = svcvt_f32_u32_x(
                predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t const *)(result_row + col_index)));
            svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(predicate_f32x,
                                                               svld1_u32(predicate_f32x, b_norms + col_index));
            svst1_f32(
                predicate_f32x, result_row + col_index,
                nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x, target_norms_sq_f32x));
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

__arm_locally_streaming static void nk_angulars_symmetric_u4_sme_finalize_streaming_(          //
    nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u4_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_angulars_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                          target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_u4_sme(                                          //
    nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u4_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_u32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_u4_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

__arm_locally_streaming static void nk_euclideans_symmetric_u4_sme_finalize_streaming_(        //
    nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride_elements, //
    nk_f32_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal (store as u32 in f32 slot)
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t row_sumsq_u32 = nk_dots_reduce_sumsq_u4_ssve_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = row_sumsq_u32;
    }
    // Phase 2: column-first post-processing
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u4_ssve_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_sumsq_u32 = ((nk_u32_t *)result_row)[row_index];
            svfloat32_t query_norm_sq_f32x = svdup_n_f32((nk_f32_t)query_sumsq_u32);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntw()) {
                svbool_t predicate_f32x = svwhilelt_b32_u64(col_index, chunk_end);
                svfloat32_t dots_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, (nk_u32_t *)(result_row + col_index)));
                svfloat32_t target_norms_sq_f32x = svcvt_f32_u32_x(
                    predicate_f32x, svld1_u32(predicate_f32x, norms_cache + (col_index - chunk_start)));
                svst1_f32(predicate_f32x, result_row + col_index,
                          nk_euclideans_from_dot_f32x_ssve_(predicate_f32x, dots_f32x, query_norm_sq_f32x,
                                                            target_norms_sq_f32x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_u4_sme(                                        //
    nk_u4x2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_u4x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u4_sme_streaming_(vectors, n_vectors, depth, stride_elements, (nk_u32_t *)result,
                                        result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_u4_sme_finalize_streaming_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // Nibble Unsigned Integers

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SME
#endif // NK_TARGET_ARM_
#endif // NK_SPATIALS_SME_H
