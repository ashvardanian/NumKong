/**
 *  @file include/numkong/spatials/graniteamx.h
 *  @author Ash Vardanian
 *  @date April 9, 2026
 *  @brief Batched spatial distances for Granite Rapids, AMX-FP16, with AVX-512 finalization.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_GRANITEAMX_H
#define NUMKONG_SPATIALS_GRANITEAMX_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_TARGET_GRANITEAMX

#include "numkong/spatial/skylake.h"
#include "numkong/spatial/serial.h"
#include "numkong/dots/serial.h"
#include "numkong/dots/graniteamx.h"
#include "numkong/dots/sapphireamx.h"     // `nk_compiler_barrier_sapphireamx_`
#include "numkong/spatials/sapphireamx.h" // `nk_angulars_row_f32dots_sapphireamx_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                                     \
    __attribute__((target(                                                                                                        \
        "avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx512vbmi,f16c,fma,bmi,bmi2,amx-tile,amx-bf16,amx-int8,amx-fp16"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx512vbmi", "f16c", "fma", \
                   "bmi", "bmi2", "amx-tile", "amx-bf16", "amx-int8", "amx-fp16")
#endif

#pragma region F16 Packed

NUMKONG_INLINE void nk_angulars_packed_f16_graniteamx_finalize_(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                                nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                nk_size_t a_stride_elements,
                                                                nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_f16_skylake_(a + row * a_stride_elements, depth,
                                                                   sizeof(nk_f16_t));
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_f16_graniteamx( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c,  //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_f16_graniteamx_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_angulars_packed_f16_graniteamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                c_stride_elements);
    return nk_success_k;
}

NUMKONG_INLINE void nk_euclideans_packed_f16_graniteamx_finalize_(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_f16_skylake_(a + row * a_stride_elements, depth,
                                                                   sizeof(nk_f16_t));
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_f16_graniteamx( //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c,    //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,      //
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_f16_graniteamx_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_f16_graniteamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

#pragma endregion F16 Packed

#pragma region F16 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_f16_graniteamx_finalize_(nk_f16_t const *vectors, nk_size_t vector_count,
                                                                   nk_size_t depth, nk_size_t stride_elements,
                                                                   nk_f32_t *result, nk_size_t result_stride_elements,
                                                                   nk_size_t rows_begin, nk_size_t rows_end) {

    for (nk_size_t row = rows_begin; row < rows_end; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_f16_skylake_(vectors + row * stride_elements,
                                                                                       depth, sizeof(nk_f16_t));

    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_skylake_(vectors + col * stride_elements,
                                                                                      depth, sizeof(nk_f16_t));

        for (nk_size_t row = rows_begin; row < rows_end; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    for (nk_size_t row = rows_begin; row < rows_end; row++) result[row * result_stride_elements + row] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f16_graniteamx( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_f16_graniteamx_(vectors, vector_count, depth, stride, result, result_stride,
                                                       rows_begin, rows_end);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_f16_graniteamx_finalize_(vectors, vector_count, depth, stride_elements, result,
                                                   result_stride_elements, rows_begin, rows_end);
    return nk_success_k;
}

NUMKONG_INLINE void nk_euclideans_symmetric_f16_graniteamx_finalize_(nk_f16_t const *vectors, nk_size_t vector_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t rows_begin, nk_size_t rows_end) {

    for (nk_size_t row = rows_begin; row < rows_end; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_f16_skylake_(vectors + row * stride_elements,
                                                                                       depth, sizeof(nk_f16_t));

    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_skylake_(vectors + col * stride_elements,
                                                                                      depth, sizeof(nk_f16_t));

        for (nk_size_t row = rows_begin; row < rows_end; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    for (nk_size_t row = rows_begin; row < rows_end; row++) result[row * result_stride_elements + row] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_graniteamx( //
    nk_f16_t const *vectors, nk_size_t vector_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_f16_graniteamx_(vectors, vector_count, depth, stride, result, result_stride,
                                                       rows_begin, rows_end);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_f16_graniteamx_finalize_(vectors, vector_count, depth, stride_elements, result,
                                                     result_stride_elements, rows_begin, rows_end);
    return nk_success_k;
}

#pragma endregion F16 Symmetric

#pragma region E5M2 Packed

NUMKONG_INLINE void nk_angulars_packed_e5m2_graniteamx_finalize_(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                 nk_size_t a_stride_elements,
                                                                 nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e5m2_skylake_(a + row * a_stride_elements, depth,
                                                                    sizeof(nk_e5m2_t));
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NUMKONG_API nk_status_t nk_angulars_packed_e5m2_graniteamx( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,  //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,     //
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride;
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e5m2_graniteamx_(a, b_packed, c, rows, columns, depth, a_stride,
                                                               c_stride);
    if (status != nk_success_k) return status;
    nk_angulars_packed_e5m2_graniteamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                 c_stride_elements);
    return nk_success_k;
}

NUMKONG_INLINE void nk_euclideans_packed_e5m2_graniteamx_finalize_(nk_e5m2_t const *a, void const *b_packed,
                                                                   nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                   nk_size_t depth, nk_size_t a_stride_elements,
                                                                   nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e5m2_skylake_(a + row * a_stride_elements, depth,
                                                                    sizeof(nk_e5m2_t));
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_graniteamx( //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,    //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,       //
    nk_size_t a_stride, nk_size_t c_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride;
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e5m2_graniteamx_(a, b_packed, c, rows, columns, depth, a_stride,
                                                               c_stride);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_e5m2_graniteamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                   c_stride_elements);
    return nk_success_k;
}

#pragma endregion E5M2 Packed

#pragma region E5M2 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_e5m2_graniteamx_finalize_(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                                    nk_size_t depth, nk_size_t stride_elements,
                                                                    nk_f32_t *result, nk_size_t result_stride_elements,
                                                                    nk_size_t rows_begin, nk_size_t rows_end) {

    for (nk_size_t row = rows_begin; row < rows_end; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e5m2_skylake_(vectors + row * stride_elements,
                                                                                        depth, sizeof(nk_e5m2_t));

    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_skylake_(vectors + col * stride_elements,
                                                                                       depth, sizeof(nk_e5m2_t));

        for (nk_size_t row = rows_begin; row < rows_end; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    for (nk_size_t row = rows_begin; row < rows_end; row++) result[row * result_stride_elements + row] = 0;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_graniteamx( //
    nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride;
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e5m2_graniteamx_(vectors, vector_count, depth, stride, result, result_stride,
                                                        rows_begin, rows_end);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_e5m2_graniteamx_finalize_(vectors, vector_count, depth, stride_elements, result,
                                                    result_stride_elements, rows_begin, rows_end);
    return nk_success_k;
}

NUMKONG_INLINE void nk_euclideans_symmetric_e5m2_graniteamx_finalize_(nk_e5m2_t const *vectors, nk_size_t vector_count,
                                                                      nk_size_t depth, nk_size_t stride_elements,
                                                                      nk_f32_t *result,
                                                                      nk_size_t result_stride_elements,
                                                                      nk_size_t rows_begin, nk_size_t rows_end) {

    for (nk_size_t row = rows_begin; row < rows_end; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e5m2_skylake_(vectors + row * stride_elements,
                                                                                        depth, sizeof(nk_e5m2_t));

    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vector_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vector_count ? chunk_start + 256 : vector_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_skylake_(vectors + col * stride_elements,
                                                                                       depth, sizeof(nk_e5m2_t));

        for (nk_size_t row = rows_begin; row < rows_end; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    for (nk_size_t row = rows_begin; row < rows_end; row++) result[row * result_stride_elements + row] = 0;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_graniteamx( //
    nk_e5m2_t const *vectors, nk_size_t vector_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t rows_begin, nk_size_t rows_end, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    rows_end = nk_min_of_two(rows_end, vector_count);
    nk_size_t const stride_elements = stride;
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e5m2_graniteamx_(vectors, vector_count, depth, stride, result, result_stride,
                                                        rows_begin, rows_end);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_e5m2_graniteamx_finalize_(vectors, vector_count, depth, stride_elements, result,
                                                      result_stride_elements, rows_begin, rows_end);
    return nk_success_k;
}

#pragma endregion E5M2 Symmetric

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_GRANITEAMX
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_SPATIALS_GRANITEAMX_H
