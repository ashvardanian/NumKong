/**
 *  @file include/numkong/spatials/smef64.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *  @brief Batched spatial distances for ARM SME-F64.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_SMEF64_H
#define NUMKONG_SPATIALS_SMEF64_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_SMEF64

#include "numkong/dots/serial.h"
#include "numkong/reduce/sve.h" // `nk_svaddv_f64_`
#include "numkong/dots/smef64.h"
#include "numkong/dots/sme.h" // `nk_start_sme_streaming_`, `nk_stop_sme_streaming_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("sme,sme-f64f64"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sme+sme-f64f64")
#endif

NUMKONG_INLINE nk_f64_t nk_dots_reduce_sumsq_f32_smef64_streaming_(nk_f32_t const *data,
                                                                   nk_size_t count) NUMKONG_STREAMING_ {
    svfloat64_t accumulator_even_f64x = svdup_f64(0.0);
    svfloat64_t accumulator_odd_f64x = svdup_f64(0.0);
    nk_size_t const vector_length = svcntw();
    // Lanes past `count` load as zeros, so every widened lane may accumulate.
    svbool_t const widened_b64x = svptrue_b64();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b32x = svwhilelt_b32_u64(i, count);
        svfloat32_t values_f32x = svld1_f32(predicate_b32x, data + i);

        svfloat64_t values_even_f64x = svcvt_f64_f32_x(widened_b64x, values_f32x);
        accumulator_even_f64x = svmla_f64_m(widened_b64x, accumulator_even_f64x, values_even_f64x, values_even_f64x);

        svfloat64_t values_odd_f64x = svcvtlt_f64_f32_x(widened_b64x, values_f32x);
        accumulator_odd_f64x = svmla_f64_m(widened_b64x, accumulator_odd_f64x, values_odd_f64x, values_odd_f64x);
    }
    return nk_svaddv_f64_(svptrue_b64(), accumulator_even_f64x) + nk_svaddv_f64_(svptrue_b64(), accumulator_odd_f64x);
}

NUMKONG_INLINE nk_f64_t nk_dots_reduce_sumsq_f64_smef64_streaming_(nk_f64_t const *data,
                                                                   nk_size_t count) NUMKONG_STREAMING_ {
    svfloat64_t sum_f64x = svdup_f64(0.0), compensation_f64x = svdup_f64(0.0);
    nk_size_t const vector_length = svcntd();
    for (nk_size_t i = 0; i < count; i += vector_length) {
        svbool_t predicate_b64x = svwhilelt_b64_u64(i, count);
        svfloat64_t values_f64x = svld1_f64(predicate_b64x, data + i);
        nk_dot2_accumulate_f64_smef64_streaming_(predicate_b64x, &sum_f64x, &compensation_f64x, values_f64x,
                                                 values_f64x);
    }
    return nk_svaddv_f64_(svptrue_b64(), sum_f64x) + nk_svaddv_f64_(svptrue_b64(), compensation_f64x);
}

NUMKONG_INLINE svfloat64_t nk_angulars_from_dot_ssvef64_f64x_smef64_(
    svbool_t predicate_b64x, svfloat64_t dots_f64x, svfloat64_t query_norm_sq_f64x,
    svfloat64_t target_norms_sq_f64x) NUMKONG_STREAMING_ {
    // Separate reciprocal square roots avoid overflowing the product of two large norms.
    svfloat64_t one_f64x = svdup_n_f64(1.0);
    svfloat64_t query_rsqrt_f64x = svdiv_f64_x(predicate_b64x, one_f64x,
                                               svsqrt_f64_x(predicate_b64x, query_norm_sq_f64x));
    svfloat64_t target_rsqrt_f64x = svdiv_f64_x(predicate_b64x, one_f64x,
                                                svsqrt_f64_x(predicate_b64x, target_norms_sq_f64x));
    svfloat64_t scaled_f64x = svmul_f64_x(predicate_b64x, svmul_f64_x(predicate_b64x, dots_f64x, query_rsqrt_f64x),
                                          target_rsqrt_f64x);
    svfloat64_t angular_f64x = svsub_f64_x(predicate_b64x, one_f64x, scaled_f64x);
    svbool_t query_zero_b64x = svcmpeq_n_f64(predicate_b64x, query_norm_sq_f64x, 0.0);
    svbool_t target_zero_b64x = svcmpeq_n_f64(predicate_b64x, target_norms_sq_f64x, 0.0);
    svbool_t one_b64x = svorr_b_z(predicate_b64x, svorr_b_z(predicate_b64x, query_zero_b64x, target_zero_b64x),
                                  svcmpeq_n_f64(predicate_b64x, dots_f64x, 0.0));
    angular_f64x = svsel_f64(one_b64x, one_f64x, angular_f64x);
    angular_f64x = svsel_f64(svand_b_z(predicate_b64x, query_zero_b64x, target_zero_b64x), svdup_n_f64(0.0),
                             angular_f64x);
    angular_f64x = svsel_f64(svcmpuo_f64(predicate_b64x, dots_f64x, dots_f64x), dots_f64x, angular_f64x);
    // `svmax` is FMAX, which keeps a NaN
    return svmax_f64_x(predicate_b64x, angular_f64x, svdup_n_f64(0.0));
}

NUMKONG_INLINE svfloat64_t nk_euclideans_from_dot_ssvef64_f64x_smef64_(
    svbool_t predicate_b64x, svfloat64_t dots_f64x, svfloat64_t query_norm_sq_f64x,
    svfloat64_t target_norms_sq_f64x) NUMKONG_STREAMING_ {
    svfloat64_t sum_sq_f64x = svadd_f64_x(predicate_b64x, query_norm_sq_f64x, target_norms_sq_f64x);
    svfloat64_t dist_sq_f64x = svsub_f64_x(predicate_b64x, sum_sq_f64x,
                                           svmul_f64_x(predicate_b64x, svdup_n_f64(2.0), dots_f64x));
    dist_sq_f64x = svmax_f64_x(predicate_b64x, dist_sq_f64x, svdup_n_f64(0.0));
    return svsqrt_f64_x(predicate_b64x, dist_sq_f64x);
}

#pragma region F32 Packed Angular

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_f32_smef64_streaming_( //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f64_t const *b_norms = (nk_f64_t const *)((char const *)b_packed + header->norms_offset);

    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f32_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *c_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f32_smef64_streaming_(a_row, depth);
        svfloat64_t query_norm_sq_f64x = svdup_n_f64(query_norm_sq_f64);

        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntd()) {
            svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, columns);
            svfloat64_t dots_f64x = svld1_f64(predicate_b64x, c_row + col_index);
            svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, b_norms + col_index);
            svst1_f64(predicate_b64x, c_row + col_index,
                      nk_angulars_from_dot_ssvef64_f64x_smef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                                target_norms_sq_f64x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_f32_smef64( //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_smef64_k) return nk_pack_mismatch_k;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);

    nk_start_sme_streaming_();
    nk_dots_packed_f32_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_finalize_f32_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion F32 Packed Angular
#pragma region F32 Packed Euclidean

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_f32_smef64_streaming_( //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f64_t const *b_norms = (nk_f64_t const *)((char const *)b_packed + header->norms_offset);

    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f32_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *c_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f32_smef64_streaming_(a_row, depth);
        svfloat64_t query_norm_sq_f64x = svdup_n_f64(query_norm_sq_f64);

        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntd()) {
            svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, columns);
            svfloat64_t dots_f64x = svld1_f64(predicate_b64x, c_row + col_index);
            svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, b_norms + col_index);
            svst1_f64(predicate_b64x, c_row + col_index,
                      nk_euclideans_from_dot_ssvef64_f64x_smef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                                  target_norms_sq_f64x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_f32_smef64( //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_smef64_k) return nk_pack_mismatch_k;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);

    nk_start_sme_streaming_();
    nk_dots_packed_f32_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_finalize_f32_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                        c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion F32 Packed Euclidean
#pragma region F32 Symmetric Angular

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_f32_smef64_streaming_( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f64_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t const *row_vector = vectors + row_index * stride_elements;
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f32_smef64_streaming_(row_vector, depth);
    }
    // column-chunked post-processing
    nk_f64_t column_norms[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col) {
            nk_f32_t const *col_vector = vectors + col * stride_elements;
            column_norms[col - chunk_start] = nk_dots_reduce_sumsq_f32_smef64_streaming_(col_vector, depth);
        }
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f64_t *result_row = result + row_index * result_stride_elements;
            svfloat64_t query_norm_sq_f64x = svdup_n_f64(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntd()) {
                svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, chunk_end);
                svfloat64_t dots_f64x = svld1_f64(predicate_b64x, result_row + col_index);
                svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, column_norms + (col_index - chunk_start));
                svst1_f64(predicate_b64x, result_row + col_index,
                          nk_angulars_from_dot_ssvef64_f64x_smef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                                    target_norms_sq_f64x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f32_smef64( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f64_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);

    nk_size_t const stride_elements = stride / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);

    nk_start_sme_streaming_();
    nk_dots_symmetric_f32_smef64_streaming_(vectors, vector_count, depth, stride_elements, result,
                                            result_stride_elements, rows_begin, rows_end);
    nk_angulars_symmetric_finalize_f32_smef64_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                         result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion F32 Symmetric Angular
#pragma region F32 Symmetric Euclidean

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_f32_smef64_streaming_( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f64_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f32_t const *row_vector = vectors + row_index * stride_elements;
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f32_smef64_streaming_(row_vector, depth);
    }
    // column-chunked post-processing
    nk_f64_t column_norms[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col) {
            nk_f32_t const *col_vector = vectors + col * stride_elements;
            column_norms[col - chunk_start] = nk_dots_reduce_sumsq_f32_smef64_streaming_(col_vector, depth);
        }
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f64_t *result_row = result + row_index * result_stride_elements;
            svfloat64_t query_norm_sq_f64x = svdup_n_f64(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntd()) {
                svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, chunk_end);
                svfloat64_t dots_f64x = svld1_f64(predicate_b64x, result_row + col_index);
                svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, column_norms + (col_index - chunk_start));
                svst1_f64(predicate_b64x, result_row + col_index,
                          nk_euclideans_from_dot_ssvef64_f64x_smef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                                      target_norms_sq_f64x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_smef64( //
    nk_f32_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f64_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);

    nk_size_t const stride_elements = stride / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);

    nk_start_sme_streaming_();
    nk_dots_symmetric_f32_smef64_streaming_(vectors, vector_count, depth, stride_elements, result,
                                            result_stride_elements, rows_begin, rows_end);
    nk_euclideans_symmetric_finalize_f32_smef64_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                           result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion F32 Symmetric Euclidean
#pragma region F64 Packed Angular

NUMKONG_OUTLINED_ void nk_angulars_packed_finalize_f64_smef64_streaming_( //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f64_t const *b_norms = (nk_f64_t const *)((char const *)b_packed + header->norms_offset);

    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f64_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *c_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f64_smef64_streaming_(a_row, depth);
        svfloat64_t query_norm_sq_f64x = svdup_n_f64(query_norm_sq_f64);

        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntd()) {
            svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, columns);
            svfloat64_t dots_f64x = svld1_f64(predicate_b64x, c_row + col_index);
            svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, b_norms + col_index);
            svst1_f64(predicate_b64x, c_row + col_index,
                      nk_angulars_from_dot_ssvef64_f64x_smef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                                target_norms_sq_f64x));
        }
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_f64_smef64( //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_smef64_k) return nk_pack_mismatch_k;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);

    nk_start_sme_streaming_();
    nk_dots_packed_f64_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_angulars_packed_finalize_f64_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                      c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion F64 Packed Angular
#pragma region F64 Packed Euclidean

NUMKONG_OUTLINED_ void nk_euclideans_packed_finalize_f64_smef64_streaming_( //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride_elements, nk_size_t c_stride_elements) NUMKONG_STREAMING_ {

    nk_dots_sme_packed_header_t const *header = (nk_dots_sme_packed_header_t const *)b_packed;
    nk_f64_t const *b_norms = (nk_f64_t const *)((char const *)b_packed + header->norms_offset);

    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f64_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *c_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f64_smef64_streaming_(a_row, depth);
        svfloat64_t query_norm_sq_f64x = svdup_n_f64(query_norm_sq_f64);

        for (nk_size_t col_index = 0; col_index < columns; col_index += svcntd()) {
            svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, columns);
            svfloat64_t dots_f64x = svld1_f64(predicate_b64x, c_row + col_index);
            svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, b_norms + col_index);
            svst1_f64(predicate_b64x, c_row + col_index,
                      nk_euclideans_from_dot_ssvef64_f64x_smef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                                  target_norms_sq_f64x));
        }
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_f64_smef64( //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, nk_size_t rows, nk_size_t columns, nk_size_t depth,
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (((nk_dots_sme_packed_header_t const *)b_packed)->capability != nk_cap_smef64_k) return nk_pack_mismatch_k;

    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);

    nk_start_sme_streaming_();
    nk_dots_packed_f64_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    nk_euclideans_packed_finalize_f64_smef64_streaming_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                        c_stride_elements);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion F64 Packed Euclidean
#pragma region F64 Symmetric Angular

NUMKONG_OUTLINED_ void nk_angulars_symmetric_finalize_f64_smef64_streaming_( //
    nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f64_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f64_t const *row_vector = vectors + row_index * stride_elements;
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f64_smef64_streaming_(row_vector, depth);
    }
    // column-chunked post-processing
    nk_f64_t column_norms[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col) {
            nk_f64_t const *col_vector = vectors + col * stride_elements;
            column_norms[col - chunk_start] = nk_dots_reduce_sumsq_f64_smef64_streaming_(col_vector, depth);
        }
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f64_t *result_row = result + row_index * result_stride_elements;
            svfloat64_t query_norm_sq_f64x = svdup_n_f64(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntd()) {
                svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, chunk_end);
                svfloat64_t dots_f64x = svld1_f64(predicate_b64x, result_row + col_index);
                svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, column_norms + (col_index - chunk_start));
                svst1_f64(predicate_b64x, result_row + col_index,
                          nk_angulars_from_dot_ssvef64_f64x_smef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                                    target_norms_sq_f64x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f64_smef64( //
    nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f64_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);

    nk_size_t const stride_elements = stride / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);

    nk_start_sme_streaming_();
    nk_dots_symmetric_f64_smef64_streaming_(vectors, vector_count, depth, stride_elements, result,
                                            result_stride_elements, rows_begin, rows_end);
    nk_angulars_symmetric_finalize_f64_smef64_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                         result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
}

#pragma endregion F64 Symmetric Angular
#pragma region F64 Symmetric Euclidean

NUMKONG_OUTLINED_ void nk_euclideans_symmetric_finalize_f64_smef64_streaming_( //
    nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride_elements, nk_f64_t *result,
    nk_size_t result_stride_elements, nk_size_t rows_begin, nk_size_t rows_end) NUMKONG_STREAMING_ {
    // cache row norms on diagonal
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
        nk_f64_t const *row_vector = vectors + row_index * stride_elements;
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f64_smef64_streaming_(row_vector, depth);
    }
    // column-chunked post-processing
    nk_f64_t column_norms[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col) {
            nk_f64_t const *col_vector = vectors + col * stride_elements;
            column_norms[col - chunk_start] = nk_dots_reduce_sumsq_f64_smef64_streaming_(col_vector, depth);
        }
        for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f64_t *result_row = result + row_index * result_stride_elements;
            svfloat64_t query_norm_sq_f64x = svdup_n_f64(result_row[row_index]);
            for (nk_size_t col_index = col_start; col_index < chunk_end; col_index += svcntd()) {
                svbool_t predicate_b64x = svwhilelt_b64_u64(col_index, chunk_end);
                svfloat64_t dots_f64x = svld1_f64(predicate_b64x, result_row + col_index);
                svfloat64_t target_norms_sq_f64x = svld1_f64(predicate_b64x, column_norms + (col_index - chunk_start));
                svst1_f64(predicate_b64x, result_row + col_index,
                          nk_euclideans_from_dot_ssvef64_f64x_smef64_(predicate_b64x, dots_f64x, query_norm_sq_f64x,
                                                                      target_norms_sq_f64x));
            }
        }
    }
    // zero diagonals
    for (nk_size_t row_index = rows_begin; row_index < rows_end; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_smef64( //
    nk_f64_t const *vectors, nk_size_t vector_count, nk_size_t depth, nk_size_t stride, nk_f64_t *result,
    nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);

    nk_size_t const stride_elements = stride / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);

    nk_start_sme_streaming_();
    nk_dots_symmetric_f64_smef64_streaming_(vectors, vector_count, depth, stride_elements, result,
                                            result_stride_elements, rows_begin, rows_end);
    nk_euclideans_symmetric_finalize_f64_smef64_streaming_(vectors, vector_count, depth, stride_elements, result,
                                                           result_stride_elements, rows_begin, rows_end);
    nk_stop_sme_streaming_();
    return nk_success_k;
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

#endif // NUMKONG_TARGET_SMEF64
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SPATIALS_SMEF64_H
