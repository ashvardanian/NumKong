/**
 *  @brief Batched Spatial Distances for Sapphire Rapids (AMX) with AVX-512 Finalization.
 *  @file include/numkong/spatials/sapphireamx.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_SAPPHIREAMX_H
#define NK_SPATIALS_SAPPHIREAMX_H

#if NK_TARGET_X86_
#if NK_TARGET_SAPPHIREAMX

#include "numkong/spatial/skylake.h"
#include "numkong/spatial/serial.h"
#include "numkong/dots/sapphireamx.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                                            \
    __attribute__((target(                                                                                               \
        "avx2,avx512f,avx512vl,avx512bw,avx512dq,avx512fp16,avx512vbmi,f16c,fma,bmi,bmi2,amx-tile,amx-bf16,amx-int8"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512fp16", "avx512vbmi", "f16c", "fma", \
                   "bmi", "bmi2", "amx-tile", "amx-bf16", "amx-int8")
#endif

#pragma region Row Finalize Helpers

NK_INTERNAL void nk_angulars_row_f32dots_sapphireamx_(nk_f32_t *results, nk_f32_t const *norms, nk_f32_t query_norm_sq,
                                                      nk_size_t count) {
    __m512 query_norm_sq_f32x16 = _mm512_set1_ps(query_norm_sq);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 dots_f32x16 = _mm512_loadu_ps(results + i);
        __m512 norms_f32x16 = _mm512_loadu_ps(norms + i);
        __m512 products_f32x16 = _mm512_mul_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(products_f32x16);
        __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
        __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
        _mm512_storeu_ps(results + i, _mm512_max_ps(angular_f32x16, _mm512_setzero_ps()));
    }
    if (i < count) {
        __mmask16 tail = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_maskz_loadu_ps(tail, results + i);
        __m512 norms_f32x16 = _mm512_maskz_loadu_ps(tail, norms + i);
        __m512 products_f32x16 = _mm512_mul_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(products_f32x16);
        __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
        __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
        _mm512_mask_storeu_ps(results + i, tail, _mm512_max_ps(angular_f32x16, _mm512_setzero_ps()));
    }
}

NK_INTERNAL void nk_euclideans_row_f32dots_sapphireamx_(nk_f32_t *results, nk_f32_t const *norms,
                                                        nk_f32_t query_norm_sq, nk_size_t count) {
    __m512 query_norm_sq_f32x16 = _mm512_set1_ps(query_norm_sq);
    __m512 two_f32x16 = _mm512_set1_ps(2.0f);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 dots_f32x16 = _mm512_loadu_ps(results + i);
        __m512 norms_f32x16 = _mm512_loadu_ps(norms + i);
        __m512 sum_norms_f32x16 = _mm512_add_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 dist_sq_f32x16 = _mm512_fnmadd_ps(two_f32x16, dots_f32x16, sum_norms_f32x16);
        dist_sq_f32x16 = _mm512_max_ps(dist_sq_f32x16, _mm512_setzero_ps());
        _mm512_storeu_ps(results + i, _mm512_sqrt_ps(dist_sq_f32x16));
    }
    if (i < count) {
        __mmask16 tail = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_maskz_loadu_ps(tail, results + i);
        __m512 norms_f32x16 = _mm512_maskz_loadu_ps(tail, norms + i);
        __m512 sum_norms_f32x16 = _mm512_add_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 dist_sq_f32x16 = _mm512_fnmadd_ps(two_f32x16, dots_f32x16, sum_norms_f32x16);
        dist_sq_f32x16 = _mm512_max_ps(dist_sq_f32x16, _mm512_setzero_ps());
        _mm512_mask_storeu_ps(results + i, tail, _mm512_sqrt_ps(dist_sq_f32x16));
    }
}

NK_INTERNAL void nk_angulars_row_i32dots_sapphireamx_(nk_f32_t *results, nk_u32_t const *norms, nk_f32_t query_norm_sq,
                                                      nk_size_t count) {
    nk_i32_t *results_i32 = (nk_i32_t *)results;
    __m512 query_norm_sq_f32x16 = _mm512_set1_ps(query_norm_sq);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 dots_f32x16 = _mm512_cvtepi32_ps(_mm512_loadu_si512(results_i32 + i));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_loadu_si512((__m512i const *)(norms + i)));
        __m512 products_f32x16 = _mm512_mul_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(products_f32x16);
        __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
        __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
        _mm512_storeu_ps(results + i, _mm512_max_ps(angular_f32x16, _mm512_setzero_ps()));
    }
    if (i < count) {
        __mmask16 tail = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(tail, results_i32 + i));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_maskz_loadu_epi32(tail, norms + i));
        __m512 products_f32x16 = _mm512_mul_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(products_f32x16);
        __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
        __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
        _mm512_mask_storeu_ps(results + i, tail, _mm512_max_ps(angular_f32x16, _mm512_setzero_ps()));
    }
}

NK_INTERNAL void nk_euclideans_row_i32dots_sapphireamx_(nk_f32_t *results, nk_u32_t const *norms,
                                                        nk_f32_t query_norm_sq, nk_size_t count) {
    nk_i32_t *results_i32 = (nk_i32_t *)results;
    __m512 query_norm_sq_f32x16 = _mm512_set1_ps(query_norm_sq);
    __m512 two_f32x16 = _mm512_set1_ps(2.0f);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 dots_f32x16 = _mm512_cvtepi32_ps(_mm512_loadu_si512(results_i32 + i));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_loadu_si512((__m512i const *)(norms + i)));
        __m512 sum_norms_f32x16 = _mm512_add_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 dist_sq_f32x16 = _mm512_fnmadd_ps(two_f32x16, dots_f32x16, sum_norms_f32x16);
        dist_sq_f32x16 = _mm512_max_ps(dist_sq_f32x16, _mm512_setzero_ps());
        _mm512_storeu_ps(results + i, _mm512_sqrt_ps(dist_sq_f32x16));
    }
    if (i < count) {
        __mmask16 tail = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(tail, results_i32 + i));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_maskz_loadu_epi32(tail, norms + i));
        __m512 sum_norms_f32x16 = _mm512_add_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 dist_sq_f32x16 = _mm512_fnmadd_ps(two_f32x16, dots_f32x16, sum_norms_f32x16);
        dist_sq_f32x16 = _mm512_max_ps(dist_sq_f32x16, _mm512_setzero_ps());
        _mm512_mask_storeu_ps(results + i, tail, _mm512_sqrt_ps(dist_sq_f32x16));
    }
}

NK_INTERNAL void nk_angulars_row_u32dots_sapphireamx_(nk_f32_t *results, nk_u32_t const *norms, nk_f32_t query_norm_sq,
                                                      nk_size_t count) {
    nk_u32_t *results_u32 = (nk_u32_t *)results;
    __m512 query_norm_sq_f32x16 = _mm512_set1_ps(query_norm_sq);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 dots_f32x16 = _mm512_cvtepu32_ps(_mm512_loadu_si512((__m512i const *)(results_u32 + i)));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_loadu_si512((__m512i const *)(norms + i)));
        __m512 products_f32x16 = _mm512_mul_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(products_f32x16);
        __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
        __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
        _mm512_storeu_ps(results + i, _mm512_max_ps(angular_f32x16, _mm512_setzero_ps()));
    }
    if (i < count) {
        __mmask16 tail = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_cvtepu32_ps(_mm512_maskz_loadu_epi32(tail, results_u32 + i));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_maskz_loadu_epi32(tail, norms + i));
        __m512 products_f32x16 = _mm512_mul_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(products_f32x16);
        __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
        __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
        _mm512_mask_storeu_ps(results + i, tail, _mm512_max_ps(angular_f32x16, _mm512_setzero_ps()));
    }
}

NK_INTERNAL void nk_euclideans_row_u32dots_sapphireamx_(nk_f32_t *results, nk_u32_t const *norms,
                                                        nk_f32_t query_norm_sq, nk_size_t count) {
    nk_u32_t *results_u32 = (nk_u32_t *)results;
    __m512 query_norm_sq_f32x16 = _mm512_set1_ps(query_norm_sq);
    __m512 two_f32x16 = _mm512_set1_ps(2.0f);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 dots_f32x16 = _mm512_cvtepu32_ps(_mm512_loadu_si512((__m512i const *)(results_u32 + i)));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_loadu_si512((__m512i const *)(norms + i)));
        __m512 sum_norms_f32x16 = _mm512_add_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 dist_sq_f32x16 = _mm512_fnmadd_ps(two_f32x16, dots_f32x16, sum_norms_f32x16);
        dist_sq_f32x16 = _mm512_max_ps(dist_sq_f32x16, _mm512_setzero_ps());
        _mm512_storeu_ps(results + i, _mm512_sqrt_ps(dist_sq_f32x16));
    }
    if (i < count) {
        __mmask16 tail = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_cvtepu32_ps(_mm512_maskz_loadu_epi32(tail, results_u32 + i));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_maskz_loadu_epi32(tail, norms + i));
        __m512 sum_norms_f32x16 = _mm512_add_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 dist_sq_f32x16 = _mm512_fnmadd_ps(two_f32x16, dots_f32x16, sum_norms_f32x16);
        dist_sq_f32x16 = _mm512_max_ps(dist_sq_f32x16, _mm512_setzero_ps());
        _mm512_mask_storeu_ps(results + i, tail, _mm512_sqrt_ps(dist_sq_f32x16));
    }
}

#pragma endregion // Row Finalize Helpers

#pragma region BF16 Packed

NK_INTERNAL void nk_angulars_packed_bf16_sapphireamx_finalize_(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride_elements,
                                                               nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_bf16_(a + row * a_stride_elements, depth);
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_angulars_packed_bf16_sapphireamx(        //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_bf16_sapphireamx(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_bf16_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_bf16_sapphireamx_finalize_(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                 nk_size_t a_stride_elements,
                                                                 nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_bf16_(a + row * a_stride_elements, depth);
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_euclideans_packed_bf16_sapphireamx(      //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_bf16_sapphireamx(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_bf16_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

#pragma endregion // BF16 Packed

#pragma region BF16 Symmetric

NK_INTERNAL void nk_angulars_symmetric_bf16_sapphireamx_finalize_(nk_bf16_t const *vectors, nk_size_t n_vectors,
                                                                  nk_size_t depth, nk_size_t stride_elements,
                                                                  nk_f32_t *result, nk_size_t result_stride_elements,
                                                                  nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_bf16_(vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_bf16_sapphireamx(                                //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_sapphireamx(vectors, n_vectors, depth, stride, (nk_f32_t *)result, result_stride, row_start,
                                       row_count);
    nk_angulars_symmetric_bf16_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_bf16_sapphireamx_finalize_(nk_bf16_t const *vectors, nk_size_t n_vectors,
                                                                    nk_size_t depth, nk_size_t stride_elements,
                                                                    nk_f32_t *result, nk_size_t result_stride_elements,
                                                                    nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_bf16_(vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_bf16_sapphireamx(                              //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_sapphireamx(vectors, n_vectors, depth, stride, (nk_f32_t *)result, result_stride, row_start,
                                       row_count);
    nk_euclideans_symmetric_bf16_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // BF16 Symmetric

#pragma region Signed 8-bit Integer Packed

NK_INTERNAL void nk_angulars_packed_i8_sapphireamx_finalize_(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = (nk_f32_t)nk_dots_reduce_sumsq_i8_(a + row * a_stride_elements, depth);
        nk_angulars_row_i32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_angulars_packed_i8_sapphireamx(        //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_i8_sapphireamx(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride_in_bytes,
                                  c_stride_in_bytes);
    nk_angulars_packed_i8_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_i8_sapphireamx_finalize_(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride_elements,
                                                               nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = (nk_f32_t)nk_dots_reduce_sumsq_i8_(a + row * a_stride_elements, depth);
        nk_euclideans_row_i32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_euclideans_packed_i8_sapphireamx(      //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_i8_sapphireamx(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride_in_bytes,
                                  c_stride_in_bytes);
    nk_euclideans_packed_i8_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

#pragma endregion // Signed 8-bit Integer Packed

#pragma region Signed 8-bit Integer Symmetric

NK_INTERNAL void nk_angulars_symmetric_i8_sapphireamx_finalize_(nk_i8_t const *vectors, nk_size_t n_vectors,
                                                                nk_size_t depth, nk_size_t stride_elements,
                                                                nk_f32_t *result, nk_size_t result_stride_elements,
                                                                nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal (stored as u32 reinterpreted in f32 slot)
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        ((nk_u32_t *)(result + row * result_stride_elements))[row] = nk_dots_reduce_sumsq_i8_(
            vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_u32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_f32_t query_norm_sq_f32 = (nk_f32_t)((nk_u32_t *)r_row)[row];
            nk_angulars_row_i32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 query_norm_sq_f32, chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_i8_sapphireamx(                                //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i8_sapphireamx(vectors, n_vectors, depth, stride, (nk_i32_t *)result, result_stride, row_start,
                                     row_count);
    nk_angulars_symmetric_i8_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                   result_stride_elements, row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_i8_sapphireamx_finalize_(nk_i8_t const *vectors, nk_size_t n_vectors,
                                                                  nk_size_t depth, nk_size_t stride_elements,
                                                                  nk_f32_t *result, nk_size_t result_stride_elements,
                                                                  nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal (stored as u32 reinterpreted in f32 slot)
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        ((nk_u32_t *)(result + row * result_stride_elements))[row] = nk_dots_reduce_sumsq_i8_(
            vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_u32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_f32_t query_norm_sq_f32 = (nk_f32_t)((nk_u32_t *)r_row)[row];
            nk_euclideans_row_i32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   query_norm_sq_f32, chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_i8_sapphireamx(                              //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i8_sapphireamx(vectors, n_vectors, depth, stride, (nk_i32_t *)result, result_stride, row_start,
                                     row_count);
    nk_euclideans_symmetric_i8_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

#pragma endregion // Signed 8-bit Integer Symmetric

#pragma region Unsigned 8-bit Integer Packed

NK_INTERNAL void nk_angulars_packed_u8_sapphireamx_finalize_(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                             nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                             nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = (nk_f32_t)nk_dots_reduce_sumsq_u8_(a + row * a_stride_elements, depth);
        nk_angulars_row_u32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_angulars_packed_u8_sapphireamx(        //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_u8_sapphireamx(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride_in_bytes,
                                  c_stride_in_bytes);
    nk_angulars_packed_u8_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_u8_sapphireamx_finalize_(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride_elements,
                                                               nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = (nk_f32_t)nk_dots_reduce_sumsq_u8_(a + row * a_stride_elements, depth);
        nk_euclideans_row_u32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_euclideans_packed_u8_sapphireamx(      //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_u8_sapphireamx(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride_in_bytes,
                                  c_stride_in_bytes);
    nk_euclideans_packed_u8_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

#pragma endregion // Unsigned 8-bit Integer Packed

#pragma region Unsigned 8-bit Integer Symmetric

NK_INTERNAL void nk_angulars_symmetric_u8_sapphireamx_finalize_(nk_u8_t const *vectors, nk_size_t n_vectors,
                                                                nk_size_t depth, nk_size_t stride_elements,
                                                                nk_f32_t *result, nk_size_t result_stride_elements,
                                                                nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal (stored as u32 reinterpreted in f32 slot)
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        ((nk_u32_t *)(result + row * result_stride_elements))[row] = nk_dots_reduce_sumsq_u8_(
            vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_u32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_f32_t query_norm_sq_f32 = (nk_f32_t)((nk_u32_t *)r_row)[row];
            nk_angulars_row_u32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 query_norm_sq_f32, chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_u8_sapphireamx(                                //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u8_sapphireamx(vectors, n_vectors, depth, stride, (nk_u32_t *)result, result_stride, row_start,
                                     row_count);
    nk_angulars_symmetric_u8_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                   result_stride_elements, row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_u8_sapphireamx_finalize_(nk_u8_t const *vectors, nk_size_t n_vectors,
                                                                  nk_size_t depth, nk_size_t stride_elements,
                                                                  nk_f32_t *result, nk_size_t result_stride_elements,
                                                                  nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal (stored as u32 reinterpreted in f32 slot)
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        ((nk_u32_t *)(result + row * result_stride_elements))[row] = nk_dots_reduce_sumsq_u8_(
            vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_u32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_f32_t query_norm_sq_f32 = (nk_f32_t)((nk_u32_t *)r_row)[row];
            nk_euclideans_row_u32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   query_norm_sq_f32, chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_u8_sapphireamx(                              //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u8_sapphireamx(vectors, n_vectors, depth, stride, (nk_u32_t *)result, result_stride, row_start,
                                     row_count);
    nk_euclideans_symmetric_u8_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

#pragma endregion // Unsigned 8-bit Integer Symmetric

#pragma region E4M3 Packed

NK_INTERNAL void nk_angulars_packed_e4m3_sapphireamx_finalize_(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride_elements,
                                                               nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e4m3_(a + row * a_stride_elements, depth);
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_angulars_packed_e4m3_sapphireamx(        //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e4m3_sapphireamx(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_e4m3_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_e4m3_sapphireamx_finalize_(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                 nk_size_t a_stride_elements,
                                                                 nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e4m3_(a + row * a_stride_elements, depth);
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_euclideans_packed_e4m3_sapphireamx(      //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e4m3_sapphireamx(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_e4m3_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

#pragma endregion // E4M3 Packed

#pragma region E5M2 Packed

NK_INTERNAL void nk_angulars_packed_e5m2_sapphireamx_finalize_(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride_elements,
                                                               nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e5m2_(a + row * a_stride_elements, depth);
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_angulars_packed_e5m2_sapphireamx(        //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e5m2_sapphireamx(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_e5m2_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_e5m2_sapphireamx_finalize_(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                 nk_size_t a_stride_elements,
                                                                 nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e5m2_(a + row * a_stride_elements, depth);
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_euclideans_packed_e5m2_sapphireamx(      //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e5m2_sapphireamx(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_e5m2_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

#pragma endregion // E5M2 Packed

#pragma region E5M2 Symmetric

NK_INTERNAL void nk_angulars_symmetric_e5m2_sapphireamx_finalize_(nk_e5m2_t const *vectors, nk_size_t n_vectors,
                                                                  nk_size_t depth, nk_size_t stride_elements,
                                                                  nk_f32_t *result, nk_size_t result_stride_elements,
                                                                  nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e5m2_(vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e5m2_sapphireamx(                                //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_sapphireamx(vectors, n_vectors, depth, stride, (nk_f32_t *)result, result_stride, row_start,
                                       row_count);
    nk_angulars_symmetric_e5m2_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_e5m2_sapphireamx_finalize_(nk_e5m2_t const *vectors, nk_size_t n_vectors,
                                                                    nk_size_t depth, nk_size_t stride_elements,
                                                                    nk_f32_t *result, nk_size_t result_stride_elements,
                                                                    nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e5m2_(vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e5m2_sapphireamx(                              //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_sapphireamx(vectors, n_vectors, depth, stride, (nk_f32_t *)result, result_stride, row_start,
                                       row_count);
    nk_euclideans_symmetric_e5m2_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // E5M2 Symmetric

#pragma region E4M3 Symmetric

NK_INTERNAL void nk_angulars_symmetric_e4m3_sapphireamx_finalize_(nk_e4m3_t const *vectors, nk_size_t n_vectors,
                                                                  nk_size_t depth, nk_size_t stride_elements,
                                                                  nk_f32_t *result, nk_size_t result_stride_elements,
                                                                  nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e4m3_(vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e4m3_sapphireamx(                                //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_sapphireamx(vectors, n_vectors, depth, stride, (nk_f32_t *)result, result_stride, row_start,
                                       row_count);
    nk_angulars_symmetric_e4m3_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_e4m3_sapphireamx_finalize_(nk_e4m3_t const *vectors, nk_size_t n_vectors,
                                                                    nk_size_t depth, nk_size_t stride_elements,
                                                                    nk_f32_t *result, nk_size_t result_stride_elements,
                                                                    nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e4m3_(vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e4m3_sapphireamx(                              //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_sapphireamx(vectors, n_vectors, depth, stride, (nk_f32_t *)result, result_stride, row_start,
                                       row_count);
    nk_euclideans_symmetric_e4m3_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // E4M3 Symmetric

#pragma region E2M3 Packed

NK_INTERNAL void nk_angulars_packed_e2m3_sapphireamx_finalize_(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride_elements,
                                                               nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e2m3_(a + row * a_stride_elements, depth);
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_angulars_packed_e2m3_sapphireamx(        //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e2m3_sapphireamx(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_e2m3_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_e2m3_sapphireamx_finalize_(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                 nk_size_t a_stride_elements,
                                                                 nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e2m3_(a + row * a_stride_elements, depth);
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_euclideans_packed_e2m3_sapphireamx(      //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e2m3_sapphireamx(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_e2m3_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

#pragma endregion // E2M3 Packed

#pragma region E2M3 Symmetric

NK_INTERNAL void nk_angulars_symmetric_e2m3_sapphireamx_finalize_(nk_e2m3_t const *vectors, nk_size_t n_vectors,
                                                                  nk_size_t depth, nk_size_t stride_elements,
                                                                  nk_f32_t *result, nk_size_t result_stride_elements,
                                                                  nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e2m3_(vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e2m3_sapphireamx(                                //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_sapphireamx(vectors, n_vectors, depth, stride, (nk_f32_t *)result, result_stride, row_start,
                                       row_count);
    nk_angulars_symmetric_e2m3_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_e2m3_sapphireamx_finalize_(nk_e2m3_t const *vectors, nk_size_t n_vectors,
                                                                    nk_size_t depth, nk_size_t stride_elements,
                                                                    nk_f32_t *result, nk_size_t result_stride_elements,
                                                                    nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e2m3_(vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e2m3_sapphireamx(                              //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_sapphireamx(vectors, n_vectors, depth, stride, (nk_f32_t *)result, result_stride, row_start,
                                       row_count);
    nk_euclideans_symmetric_e2m3_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // E2M3 Symmetric

#pragma region E3M2 Packed

NK_INTERNAL void nk_angulars_packed_e3m2_sapphireamx_finalize_(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                               nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                               nk_size_t a_stride_elements,
                                                               nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e3m2_(a + row * a_stride_elements, depth);
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_angulars_packed_e3m2_sapphireamx(        //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e3m2_sapphireamx(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_e3m2_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_e3m2_sapphireamx_finalize_(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                                 nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                 nk_size_t a_stride_elements,
                                                                 nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e3m2_(a + row * a_stride_elements, depth);
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

NK_PUBLIC void nk_euclideans_packed_e3m2_sapphireamx(      //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e3m2_sapphireamx(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_e3m2_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
}

#pragma endregion // E3M2 Packed

#pragma region E3M2 Symmetric

NK_INTERNAL void nk_angulars_symmetric_e3m2_sapphireamx_finalize_(nk_e3m2_t const *vectors, nk_size_t n_vectors,
                                                                  nk_size_t depth, nk_size_t stride_elements,
                                                                  nk_f32_t *result, nk_size_t result_stride_elements,
                                                                  nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e3m2_(vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e3m2_sapphireamx(                                //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_sapphireamx(vectors, n_vectors, depth, stride, (nk_f32_t *)result, result_stride, row_start,
                                       row_count);
    nk_angulars_symmetric_e3m2_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_e3m2_sapphireamx_finalize_(nk_e3m2_t const *vectors, nk_size_t n_vectors,
                                                                    nk_size_t depth, nk_size_t stride_elements,
                                                                    nk_f32_t *result, nk_size_t result_stride_elements,
                                                                    nk_size_t row_start, nk_size_t row_count) {

    // Phase 1: Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e3m2_(vectors + row * stride_elements, depth);

    // Phase 2: 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_(vectors + col * stride_elements, depth);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Phase 3: Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e3m2_sapphireamx(                              //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_sapphireamx(vectors, n_vectors, depth, stride, (nk_f32_t *)result, result_stride, row_start,
                                       row_count);
    nk_euclideans_symmetric_e3m2_sapphireamx_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
}

#pragma endregion // E3M2 Symmetric

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SAPPHIREAMX
#endif // NK_TARGET_X86_
#endif // NK_SPATIALS_SAPPHIREAMX_H
