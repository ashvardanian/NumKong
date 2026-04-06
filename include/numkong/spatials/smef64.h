/**
 *  @brief Batched Spatial Distances for ARM SME-F64.
 *  @file include/numkong/spatials/smef64.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_SMEF64_H
#define NK_SPATIALS_SMEF64_H

#if NK_TARGET_ARM64_
#if NK_TARGET_SME

#include "numkong/dots/serial.h"
#include "numkong/dots/smef64.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme,sme-f64f64"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme+sme-f64f64")
#endif

NK_PUBLIC nk_f64_t nk_dots_reduce_sumsq_f32_ssve_(nk_f32_t const *data, nk_size_t count) NK_STREAMING_ {
    svfloat64_t accumulator_even_f64x = svdup_f64(0.0);
    svfloat64_t accumulator_odd_f64x = svdup_f64(0.0);
    nk_size_t const vector_length = svcntw();
    nk_size_t const half_vector_length = svcntd();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b32x = svwhilelt_b32_u64(i, count);
        svfloat32_t values_f32x = svld1_f32(predicate_b32x, data + i);

        svbool_t predicate_even_b64x = svwhilelt_b64_u64(i, count);
        svfloat64_t values_even_f64x = svcvt_f64_f32_x(predicate_even_b64x, values_f32x);
        accumulator_even_f64x = svmla_f64_m(predicate_even_b64x, accumulator_even_f64x, values_even_f64x,
                                            values_even_f64x);

        svbool_t predicate_odd_b64x = svwhilelt_b64_u64(i + half_vector_length, count);
        svfloat64_t values_odd_f64x = svcvtlt_f64_f32_x(predicate_odd_b64x, values_f32x);
        accumulator_odd_f64x = svmla_f64_m(predicate_odd_b64x, accumulator_odd_f64x, values_odd_f64x, values_odd_f64x);
    }
    nk_f64_t sum_even = svaddv_f64(svptrue_b64(), accumulator_even_f64x);
    nk_f64_t sum_odd = svaddv_f64(svptrue_b64(), accumulator_odd_f64x);
    NK_UNPOISON(&sum_even, sizeof(sum_even));
    NK_UNPOISON(&sum_odd, sizeof(sum_odd));
    return sum_even + sum_odd;
}

NK_PUBLIC nk_f64_t nk_dots_reduce_sumsq_f64_ssve_(nk_f64_t const *data, nk_size_t count) NK_STREAMING_ {
    svfloat64_t accumulator_f64x = svdup_f64(0.0);
    nk_size_t const vector_length = svcntd();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b64x = svwhilelt_b64_u64(i, count);
        svfloat64_t values_f64x = svld1_f64(predicate_b64x, data + i);
        accumulator_f64x = svmla_f64_m(predicate_b64x, accumulator_f64x, values_f64x, values_f64x);
    }
    nk_f64_t sum = svaddv_f64(svptrue_b64(), accumulator_f64x);
    NK_UNPOISON(&sum, sizeof(sum));
    return sum;
}

NK_PUBLIC svfloat64_t nk_angulars_from_dot_f64x_ssvef64_(svbool_t predicate_b64x, svfloat64_t dots_f64x,
                                                         svfloat64_t query_norm_sq_f64x,
                                                         svfloat64_t target_norms_sq_f64x) NK_STREAMING_ {
    svfloat64_t norms_product_f64x = svmul_f64_x(predicate_b64x, query_norm_sq_f64x, target_norms_sq_f64x);
    svbool_t positive_norms_b64x = svcmpgt_n_f64(predicate_b64x, norms_product_f64x, 0.0);
    svfloat64_t denom_f64x = svsqrt_f64_x(positive_norms_b64x, norms_product_f64x);
    svfloat64_t safe_denom_f64x = svsel_f64(positive_norms_b64x, denom_f64x, svdup_n_f64(1.0));
    svfloat64_t normalized_f64x = svdiv_f64_x(predicate_b64x, dots_f64x, safe_denom_f64x);
    svfloat64_t angular_f64x = svsub_f64_x(predicate_b64x, svdup_n_f64(1.0), normalized_f64x);
    angular_f64x = svsel_f64(
        positive_norms_b64x, angular_f64x,
        svsel_f64(svcmpeq_n_f64(predicate_b64x, dots_f64x, 0.0), svdup_n_f64(0.0), svdup_n_f64(1.0)));
    return svmax_f64_x(predicate_b64x, angular_f64x, svdup_n_f64(0.0));
}

NK_PUBLIC svfloat64_t nk_euclideans_from_dot_f64x_ssvef64_(svbool_t predicate_b64x, svfloat64_t dots_f64x,
                                                           svfloat64_t query_norm_sq_f64x,
                                                           svfloat64_t target_norms_sq_f64x) NK_STREAMING_ {
    svfloat64_t sum_sq_f64x = svadd_f64_x(predicate_b64x, query_norm_sq_f64x, target_norms_sq_f64x);
    svfloat64_t dist_sq_f64x = svsub_f64_x(predicate_b64x, sum_sq_f64x,
                                           svmul_f64_x(predicate_b64x, svdup_n_f64(2.0), dots_f64x));
    dist_sq_f64x = svmax_f64_x(predicate_b64x, dist_sq_f64x, svdup_n_f64(0.0));
    return svsqrt_f64_x(predicate_b64x, dist_sq_f64x);
}

#pragma region F32 Packed Angular

__arm_locally_streaming static void nk_angulars_packed_f32_smef64_finalize_streaming_( //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c,                              //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f64_t const *b_norms = (nk_f64_t const *)((char const *)b_packed + header->norms_offset);

    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f32_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *c_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f32_ssve_(a_row, depth);
        svfloat64_t query_norm_sq_f64x = svdup_n_f64(query_norm_sq_f64);

        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntd()) {
            svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, columns);
            svfloat64_t dots_f64x = svld1_f64(predicate_b64x, c_row + col_index);
            svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, b_norms + col_index);
            svst1_f64(predicate_b64x, c_row + col_index,
                      nk_angulars_from_dot_f64x_ssvef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                         target_norms_sq_f64x));
        }
    }
}

NK_PUBLIC void nk_angulars_packed_f32_smef64(             //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {

    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f64_t);

    nk_dots_packed_f32_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_f32_smef64_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
}

#pragma endregion F32 Packed Angular
#pragma region F32 Packed Euclidean

__arm_locally_streaming static void nk_euclideans_packed_f32_smef64_finalize_streaming_( //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c,                                //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                  //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f64_t const *b_norms = (nk_f64_t const *)((char const *)b_packed + header->norms_offset);

    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f32_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *c_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f32_ssve_(a_row, depth);
        svfloat64_t query_norm_sq_f64x = svdup_n_f64(query_norm_sq_f64);

        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntd()) {
            svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, columns);
            svfloat64_t dots_f64x = svld1_f64(predicate_b64x, c_row + col_index);
            svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, b_norms + col_index);
            svst1_f64(predicate_b64x, c_row + col_index,
                      nk_euclideans_from_dot_f64x_ssvef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                           target_norms_sq_f64x));
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_f32_smef64(           //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {

    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f64_t);

    nk_dots_packed_f32_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_f32_smef64_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                        c_stride_elements);
}

#pragma endregion F32 Packed Euclidean
#pragma region F32 Symmetric Angular

__arm_locally_streaming static void nk_angulars_symmetric_f32_smef64_finalize_streaming_(         //
    nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f64_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t const *row_vector = vectors + row_index * stride_elements;
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f32_ssve_(row_vector, depth);
    }
    // Phase 2: column-chunked post-processing
    nk_f64_t column_norms[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col) {
            nk_f32_t const *col_vector = vectors + col * stride_elements;
            column_norms[col - chunk_start] = nk_dots_reduce_sumsq_f32_ssve_(col_vector, depth);
        }
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f64_t *result_row = result + row_index * result_stride_elements;
            svfloat64_t query_norm_sq_f64x = svdup_n_f64(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntd()) {
                svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, chunk_end);
                svfloat64_t dots_f64x = svld1_f64(predicate_b64x, result_row + col_index);
                svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, column_norms + (col_index - chunk_start));
                svst1_f64(predicate_b64x, result_row + col_index,
                          nk_angulars_from_dot_f64x_ssvef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                             target_norms_sq_f64x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_f32_smef64(                                                  //
    nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f64_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f64_t);

    nk_dots_symmetric_f32_smef64_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                            result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_f32_smef64_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion F32 Symmetric Angular
#pragma region F32 Symmetric Euclidean

__arm_locally_streaming static void nk_euclideans_symmetric_f32_smef64_finalize_streaming_(       //
    nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f64_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t const *row_vector = vectors + row_index * stride_elements;
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f32_ssve_(row_vector, depth);
    }
    // Phase 2: column-chunked post-processing
    nk_f64_t column_norms[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col) {
            nk_f32_t const *col_vector = vectors + col * stride_elements;
            column_norms[col - chunk_start] = nk_dots_reduce_sumsq_f32_ssve_(col_vector, depth);
        }
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f64_t *result_row = result + row_index * result_stride_elements;
            svfloat64_t query_norm_sq_f64x = svdup_n_f64(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntd()) {
                svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, chunk_end);
                svfloat64_t dots_f64x = svld1_f64(predicate_b64x, result_row + col_index);
                svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, column_norms + (col_index - chunk_start));
                svst1_f64(predicate_b64x, result_row + col_index,
                          nk_euclideans_from_dot_f64x_ssvef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                               target_norms_sq_f64x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_f32_smef64(                                                //
    nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f64_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f64_t);

    nk_dots_symmetric_f32_smef64_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                            result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_f32_smef64_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                           result_stride_elements, row_start, row_count);
}

#pragma endregion F32 Symmetric Euclidean
#pragma region F64 Packed Angular

__arm_locally_streaming static void nk_angulars_packed_f64_smef64_finalize_streaming_( //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c,                              //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f64_t const *b_norms = (nk_f64_t const *)((char const *)b_packed + header->norms_offset);

    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f64_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *c_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f64_ssve_(a_row, depth);
        svfloat64_t query_norm_sq_f64x = svdup_n_f64(query_norm_sq_f64);

        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntd()) {
            svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, columns);
            svfloat64_t dots_f64x = svld1_f64(predicate_b64x, c_row + col_index);
            svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, b_norms + col_index);
            svst1_f64(predicate_b64x, c_row + col_index,
                      nk_angulars_from_dot_f64x_ssvef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                         target_norms_sq_f64x));
        }
    }
}

NK_PUBLIC void nk_angulars_packed_f64_smef64(             //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {

    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f64_t);

    nk_dots_packed_f64_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_f64_smef64_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
}

#pragma endregion F64 Packed Angular
#pragma region F64 Packed Euclidean

__arm_locally_streaming static void nk_euclideans_packed_f64_smef64_finalize_streaming_( //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c,                                //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                                  //
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f64_t const *b_norms = (nk_f64_t const *)((char const *)b_packed + header->norms_offset);

    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f64_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *c_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f64_ssve_(a_row, depth);
        svfloat64_t query_norm_sq_f64x = svdup_n_f64(query_norm_sq_f64);

        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntd()) {
            svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, columns);
            svfloat64_t dots_f64x = svld1_f64(predicate_b64x, c_row + col_index);
            svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, b_norms + col_index);
            svst1_f64(predicate_b64x, c_row + col_index,
                      nk_euclideans_from_dot_f64x_ssvef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                           target_norms_sq_f64x));
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_f64_smef64(           //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {

    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f64_t);

    nk_dots_packed_f64_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_f64_smef64_finalize_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                        c_stride_elements);
}

#pragma endregion F64 Packed Euclidean
#pragma region F64 Symmetric Angular

__arm_locally_streaming static void nk_angulars_symmetric_f64_smef64_finalize_streaming_(         //
    nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f64_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f64_t const *row_vector = vectors + row_index * stride_elements;
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f64_ssve_(row_vector, depth);
    }
    // Phase 2: column-chunked post-processing
    nk_f64_t column_norms[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col) {
            nk_f64_t const *col_vector = vectors + col * stride_elements;
            column_norms[col - chunk_start] = nk_dots_reduce_sumsq_f64_ssve_(col_vector, depth);
        }
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f64_t *result_row = result + row_index * result_stride_elements;
            svfloat64_t query_norm_sq_f64x = svdup_n_f64(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntd()) {
                svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, chunk_end);
                svfloat64_t dots_f64x = svld1_f64(predicate_b64x, result_row + col_index);
                svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, column_norms + (col_index - chunk_start));
                svst1_f64(predicate_b64x, result_row + col_index,
                          nk_angulars_from_dot_f64x_ssvef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                             target_norms_sq_f64x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_f64_smef64(                                                  //
    nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f64_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f64_t);

    nk_dots_symmetric_f64_smef64_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                            result_stride_elements, row_start, row_count);
    nk_angulars_symmetric_f64_smef64_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                         result_stride_elements, row_start, row_count);
}

#pragma endregion F64 Symmetric Angular
#pragma region F64 Symmetric Euclidean

__arm_locally_streaming static void nk_euclideans_symmetric_f64_smef64_finalize_streaming_(       //
    nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_elements, //
    nk_f64_t *result, nk_size_t result_stride_elements, nk_size_t row_start, nk_size_t row_count) {
    // Phase 1: cache row norms on diagonal
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f64_t const *row_vector = vectors + row_index * stride_elements;
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f64_ssve_(row_vector, depth);
    }
    // Phase 2: column-chunked post-processing
    nk_f64_t column_norms[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col) {
            nk_f64_t const *col_vector = vectors + col * stride_elements;
            column_norms[col - chunk_start] = nk_dots_reduce_sumsq_f64_ssve_(col_vector, depth);
        }
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f64_t *result_row = result + row_index * result_stride_elements;
            svfloat64_t query_norm_sq_f64x = svdup_n_f64(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntd()) {
                svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, chunk_end);
                svfloat64_t dots_f64x = svld1_f64(predicate_b64x, result_row + col_index);
                svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, column_norms + (col_index - chunk_start));
                svst1_f64(predicate_b64x, result_row + col_index,
                          nk_euclideans_from_dot_f64x_ssvef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                               target_norms_sq_f64x));
            }
        }
    }
    // Phase 3: zero diagonals
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_f64_smef64(                                                //
    nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f64_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {

    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f64_t);

    nk_dots_symmetric_f64_smef64_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                            result_stride_elements, row_start, row_count);
    nk_euclideans_symmetric_f64_smef64_finalize_streaming_(vectors, vectors_count, depth, stride_elements, result,
                                                           result_stride_elements, row_start, row_count);
}

#pragma endregion F64 Symmetric Euclidean
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
#endif // NK_SPATIALS_SMEF64_H
