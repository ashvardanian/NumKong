/**
 *  @brief Batched Spatial Distances for Granite Rapids (AMX-FP16) with AVX-512 Finalization.
 *  @file include/numkong/spatials/graniteamx.h
 *  @author Ash Vardanian
 *  @date April 9, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_GRANITEAMX_H
#define NK_SPATIALS_GRANITEAMX_H

#if NK_TARGET_X8664_
#if NK_TARGET_GRANITEAMX

#include "numkong/spatial/skylake.h"
#include "numkong/spatial/serial.h"
#include "numkong/dots/graniteamx.h"

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

NK_INTERNAL void nk_angulars_packed_f16_graniteamx_finalize_(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_f16_(a + row * a_stride_elements, depth);
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_angulars_packed_f16_graniteamx(         //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_f16_graniteamx(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_f16_graniteamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_f16_graniteamx_finalize_(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride_elements,
                                                               nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_f16_(a + row * a_stride_elements, depth);
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_euclideans_packed_f16_graniteamx(       //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_f16_graniteamx(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_f16_graniteamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

#pragma endregion F16 Packed

#pragma region F16 Symmetric

NK_INTERNAL void nk_angulars_symmetric_f16_graniteamx_finalize_(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                                nk_size_t depth, nk_size_t stride_elements,
                                                                nk_f32_t *result, nk_size_t result_stride_elements,
                                                                nk_size_t row_start, nk_size_t row_count) {

    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_f16_(vectors + row * stride_elements, depth);

    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_f16_graniteamx(                                              //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_graniteamx(vectors, vectors_count, depth, stride_in_bytes, result, result_stride_in_bytes,
                                     row_start, row_count);
    nk_angulars_symmetric_f16_graniteamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                   result_stride_elements, row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_f16_graniteamx_finalize_(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                                  nk_size_t depth, nk_size_t stride_elements,
                                                                  nk_f32_t *result, nk_size_t result_stride_elements,
                                                                  nk_size_t row_start, nk_size_t row_count) {

    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_f16_(vectors + row * stride_elements, depth);

    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_f16_graniteamx(                                            //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth, nk_size_t stride_in_bytes, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_graniteamx(vectors, vectors_count, depth, stride_in_bytes, result, result_stride_in_bytes,
                                     row_start, row_count);
    nk_euclideans_symmetric_f16_graniteamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

#pragma endregion F16 Symmetric

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_GRANITEAMX
#endif // NK_TARGET_X8664_
#endif // NK_SPATIALS_GRANITEAMX_H
