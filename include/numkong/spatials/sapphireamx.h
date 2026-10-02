/**
 *  @file include/numkong/spatials/sapphireamx.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *  @brief Batched spatial distances for Sapphire Rapids, AMX, with AVX-512 finalization.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_SAPPHIREAMX_H
#define NUMKONG_SPATIALS_SAPPHIREAMX_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_SAPPHIREAMX_

#include "numkong/spatial/skylake.h"
#include "numkong/spatial/serial.h"
#include "numkong/dots/serial.h"
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

NUMKONG_INLINE void nk_angulars_row_f32dots_sapphireamx_(nk_f32_t *results, nk_f32_t const *norms,
                                                         nk_f32_t query_norm_sq, nk_size_t count) {
    __m512 query_norm_sq_f32x16 = _mm512_set1_ps(query_norm_sq);
    // Separate reciprocal square roots avoid overflowing the product of two finite-but-large norms.
    __m512 query_rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(query_norm_sq_f32x16);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 dots_f32x16 = _mm512_loadu_ps(results + i);
        __m512 norms_f32x16 = _mm512_loadu_ps(norms + i);
        __m512 target_rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(norms_f32x16);
        __m512 rsqrt_f32x16 = _mm512_mul_ps(query_rsqrt_f32x16, target_rsqrt_f32x16);
        __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
        __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
        _mm512_storeu_ps(results + i, _mm512_max_ps(angular_f32x16, _mm512_setzero_ps()));
    }
    if (i < count) {
        __mmask16 tail_m16 = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_maskz_loadu_ps(tail_m16, results + i);
        __m512 norms_f32x16 = _mm512_maskz_loadu_ps(tail_m16, norms + i);
        __m512 target_rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(norms_f32x16);
        __m512 rsqrt_f32x16 = _mm512_mul_ps(query_rsqrt_f32x16, target_rsqrt_f32x16);
        __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
        __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
        _mm512_mask_storeu_ps(results + i, tail_m16, _mm512_max_ps(angular_f32x16, _mm512_setzero_ps()));
    }
}

NUMKONG_INLINE void nk_euclideans_row_f32dots_sapphireamx_(nk_f32_t *results, nk_f32_t const *norms,
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
        __mmask16 tail_m16 = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_maskz_loadu_ps(tail_m16, results + i);
        __m512 norms_f32x16 = _mm512_maskz_loadu_ps(tail_m16, norms + i);
        __m512 sum_norms_f32x16 = _mm512_add_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 dist_sq_f32x16 = _mm512_fnmadd_ps(two_f32x16, dots_f32x16, sum_norms_f32x16);
        dist_sq_f32x16 = _mm512_max_ps(dist_sq_f32x16, _mm512_setzero_ps());
        _mm512_mask_storeu_ps(results + i, tail_m16, _mm512_sqrt_ps(dist_sq_f32x16));
    }
}

NUMKONG_INLINE void nk_angulars_row_i32dots_sapphireamx_(nk_f32_t *results, nk_u32_t const *norms,
                                                         nk_f32_t query_norm_sq, nk_size_t count) {
    nk_i32_t *results_i32 = (nk_i32_t *)results;
    __m512 query_norm_sq_f32x16 = _mm512_set1_ps(query_norm_sq);
    // Separate reciprocal square roots avoid overflowing the product of two finite-but-large norms.
    __m512 query_rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(query_norm_sq_f32x16);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 dots_f32x16 = _mm512_cvtepi32_ps(_mm512_loadu_si512(results_i32 + i));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_loadu_si512((__m512i const *)(norms + i)));
        __m512 target_rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(norms_f32x16);
        __m512 rsqrt_f32x16 = _mm512_mul_ps(query_rsqrt_f32x16, target_rsqrt_f32x16);
        __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
        __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
        _mm512_storeu_ps(results + i, _mm512_max_ps(angular_f32x16, _mm512_setzero_ps()));
    }
    if (i < count) {
        __mmask16 tail_m16 = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(tail_m16, results_i32 + i));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_maskz_loadu_epi32(tail_m16, norms + i));
        __m512 target_rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(norms_f32x16);
        __m512 rsqrt_f32x16 = _mm512_mul_ps(query_rsqrt_f32x16, target_rsqrt_f32x16);
        __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
        __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
        _mm512_mask_storeu_ps(results + i, tail_m16, _mm512_max_ps(angular_f32x16, _mm512_setzero_ps()));
    }
}

NUMKONG_INLINE void nk_euclideans_row_i32dots_sapphireamx_(nk_f32_t *results, nk_u32_t const *norms,
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
        __mmask16 tail_m16 = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(tail_m16, results_i32 + i));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_maskz_loadu_epi32(tail_m16, norms + i));
        __m512 sum_norms_f32x16 = _mm512_add_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 dist_sq_f32x16 = _mm512_fnmadd_ps(two_f32x16, dots_f32x16, sum_norms_f32x16);
        dist_sq_f32x16 = _mm512_max_ps(dist_sq_f32x16, _mm512_setzero_ps());
        _mm512_mask_storeu_ps(results + i, tail_m16, _mm512_sqrt_ps(dist_sq_f32x16));
    }
}

NUMKONG_INLINE void nk_angulars_row_u32dots_sapphireamx_(nk_f32_t *results, nk_u32_t const *norms,
                                                         nk_f32_t query_norm_sq, nk_size_t count) {
    nk_u32_t *results_u32 = (nk_u32_t *)results;
    __m512 query_norm_sq_f32x16 = _mm512_set1_ps(query_norm_sq);
    // Separate reciprocal square roots avoid overflowing the product of two finite-but-large norms.
    __m512 query_rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(query_norm_sq_f32x16);
    nk_size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 dots_f32x16 = _mm512_cvtepu32_ps(_mm512_loadu_si512((__m512i const *)(results_u32 + i)));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_loadu_si512((__m512i const *)(norms + i)));
        __m512 target_rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(norms_f32x16);
        __m512 rsqrt_f32x16 = _mm512_mul_ps(query_rsqrt_f32x16, target_rsqrt_f32x16);
        __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
        __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
        _mm512_storeu_ps(results + i, _mm512_max_ps(angular_f32x16, _mm512_setzero_ps()));
    }
    if (i < count) {
        __mmask16 tail_m16 = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_cvtepu32_ps(_mm512_maskz_loadu_epi32(tail_m16, results_u32 + i));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_maskz_loadu_epi32(tail_m16, norms + i));
        __m512 target_rsqrt_f32x16 = nk_rsqrt_f32x16_skylake_(norms_f32x16);
        __m512 rsqrt_f32x16 = _mm512_mul_ps(query_rsqrt_f32x16, target_rsqrt_f32x16);
        __m512 normalized_f32x16 = _mm512_mul_ps(dots_f32x16, rsqrt_f32x16);
        __m512 angular_f32x16 = _mm512_sub_ps(_mm512_set1_ps(1.0f), normalized_f32x16);
        _mm512_mask_storeu_ps(results + i, tail_m16, _mm512_max_ps(angular_f32x16, _mm512_setzero_ps()));
    }
}

NUMKONG_INLINE void nk_euclideans_row_u32dots_sapphireamx_(nk_f32_t *results, nk_u32_t const *norms,
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
        __mmask16 tail_m16 = (__mmask16)((1u << (count - i)) - 1);
        __m512 dots_f32x16 = _mm512_cvtepu32_ps(_mm512_maskz_loadu_epi32(tail_m16, results_u32 + i));
        __m512 norms_f32x16 = _mm512_cvtepu32_ps(_mm512_maskz_loadu_epi32(tail_m16, norms + i));
        __m512 sum_norms_f32x16 = _mm512_add_ps(query_norm_sq_f32x16, norms_f32x16);
        __m512 dist_sq_f32x16 = _mm512_fnmadd_ps(two_f32x16, dots_f32x16, sum_norms_f32x16);
        dist_sq_f32x16 = _mm512_max_ps(dist_sq_f32x16, _mm512_setzero_ps());
        _mm512_mask_storeu_ps(results + i, tail_m16, _mm512_sqrt_ps(dist_sq_f32x16));
    }
}

#pragma endregion Row Finalize Helpers

#pragma region BF16 Packed

NUMKONG_INLINE void nk_angulars_packed_bf16_sapphireamx_finalize_(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_bf16_(a + row * a_stride_elements, depth, nk_cap_sapphireamx_k);
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_packed_bf16_sapphireamx(                     //
    nk_bf16_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                          //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_bf16_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride_in_bytes,
                                                                c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_angulars_packed_bf16_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_packed_bf16_sapphireamx_finalize_(nk_bf16_t const *a, void const *b_packed,
                                                                    nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                    nk_size_t depth, nk_size_t a_stride_elements,
                                                                    nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_bf16_(a + row * a_stride_elements, depth, nk_cap_sapphireamx_k);
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_packed_bf16_sapphireamx(                   //
    nk_bf16_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                          //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_bf16_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride_in_bytes,
                                                                c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_bf16_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion BF16 Packed

#pragma region BF16 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_bf16_sapphireamx_finalize_(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_bf16_(vectors + row * stride_elements, depth,
                                                                                nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_(vectors + col * stride_elements, depth,
                                                                               nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_sapphireamx( //
    nk_bf16_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_bf16_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                         (nk_f32_t *)result, result_stride_in_bytes, row_start,
                                                         row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_bf16_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_symmetric_bf16_sapphireamx_finalize_(nk_bf16_t const *vectors,
                                                                       nk_size_t vectors_count, nk_size_t depth,
                                                                       nk_size_t stride_elements, nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_bf16_(vectors + row * stride_elements, depth,
                                                                                nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_(vectors + col * stride_elements, depth,
                                                                               nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_sapphireamx( //
    nk_bf16_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_bf16_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                         (nk_f32_t *)result, result_stride_in_bytes, row_start,
                                                         row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_bf16_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion BF16 Symmetric

#pragma region I8 Packed

NUMKONG_INLINE void nk_angulars_packed_i8_sapphireamx_finalize_(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                                nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                nk_size_t a_stride_elements,
                                                                nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = (nk_f32_t)nk_dots_reduce_sumsq_i8_(a + row * a_stride_elements, depth,
                                                                    nk_cap_sapphireamx_k);
        nk_angulars_row_i32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_packed_i8_sapphireamx(                     //
    nk_i8_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                        //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_i8_sapphireamx_(a, b_packed, (nk_i32_t *)c, rows, columns, depth,
                                                              a_stride_in_bytes, c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_angulars_packed_i8_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_packed_i8_sapphireamx_finalize_(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = (nk_f32_t)nk_dots_reduce_sumsq_i8_(a + row * a_stride_elements, depth,
                                                                    nk_cap_sapphireamx_k);
        nk_euclideans_row_i32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_packed_i8_sapphireamx(                   //
    nk_i8_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                        //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_i8_sapphireamx_(a, b_packed, (nk_i32_t *)c, rows, columns, depth,
                                                              a_stride_in_bytes, c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_i8_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion I8 Packed

#pragma region I8 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_i8_sapphireamx_finalize_(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                                   nk_size_t depth, nk_size_t stride_elements,
                                                                   nk_f32_t *result, nk_size_t result_stride_elements,
                                                                   nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal (stored as u32 reinterpreted in f32 slot)
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        ((nk_u32_t *)(result + row * result_stride_elements))[row] = nk_dots_reduce_sumsq_i8_(
            vectors + row * stride_elements, depth, nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_u32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_(vectors + col * stride_elements, depth,
                                                                             nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_f32_t query_norm_sq_f32 = (nk_f32_t)((nk_u32_t *)r_row)[row];
            nk_angulars_row_i32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 query_norm_sq_f32, chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_symmetric_i8_sapphireamx( //
    nk_i8_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_i8_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                       (nk_i32_t *)result, result_stride_in_bytes, row_start,
                                                       row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_i8_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                   result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_symmetric_i8_sapphireamx_finalize_(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal (stored as u32 reinterpreted in f32 slot)
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        ((nk_u32_t *)(result + row * result_stride_elements))[row] = nk_dots_reduce_sumsq_i8_(
            vectors + row * stride_elements, depth, nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_u32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_(vectors + col * stride_elements, depth,
                                                                             nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_f32_t query_norm_sq_f32 = (nk_f32_t)((nk_u32_t *)r_row)[row];
            nk_euclideans_row_i32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   query_norm_sq_f32, chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_sapphireamx( //
    nk_i8_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_i8_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                       (nk_i32_t *)result, result_stride_in_bytes, row_start,
                                                       row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_i8_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion I8 Symmetric

#pragma region U8 Packed

NUMKONG_INLINE void nk_angulars_packed_u8_sapphireamx_finalize_(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                                nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                nk_size_t a_stride_elements,
                                                                nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = (nk_f32_t)nk_dots_reduce_sumsq_u8_(a + row * a_stride_elements, depth,
                                                                    nk_cap_sapphireamx_k);
        nk_angulars_row_u32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_packed_u8_sapphireamx(                     //
    nk_u8_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                        //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_u8_sapphireamx_(a, b_packed, (nk_u32_t *)c, rows, columns, depth,
                                                              a_stride_in_bytes, c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_angulars_packed_u8_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_packed_u8_sapphireamx_finalize_(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_u32_t const *b_norms = (nk_u32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = (nk_f32_t)nk_dots_reduce_sumsq_u8_(a + row * a_stride_elements, depth,
                                                                    nk_cap_sapphireamx_k);
        nk_euclideans_row_u32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_packed_u8_sapphireamx(                   //
    nk_u8_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                        //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_u8_sapphireamx_(a, b_packed, (nk_u32_t *)c, rows, columns, depth,
                                                              a_stride_in_bytes, c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_u8_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion U8 Packed

#pragma region U8 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_u8_sapphireamx_finalize_(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                                   nk_size_t depth, nk_size_t stride_elements,
                                                                   nk_f32_t *result, nk_size_t result_stride_elements,
                                                                   nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal (stored as u32 reinterpreted in f32 slot)
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        ((nk_u32_t *)(result + row * result_stride_elements))[row] = nk_dots_reduce_sumsq_u8_(
            vectors + row * stride_elements, depth, nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_u32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_(vectors + col * stride_elements, depth,
                                                                             nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_f32_t query_norm_sq_f32 = (nk_f32_t)((nk_u32_t *)r_row)[row];
            nk_angulars_row_u32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 query_norm_sq_f32, chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_symmetric_u8_sapphireamx( //
    nk_u8_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_u8_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                       (nk_u32_t *)result, result_stride_in_bytes, row_start,
                                                       row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_u8_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                   result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_symmetric_u8_sapphireamx_finalize_(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal (stored as u32 reinterpreted in f32 slot)
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        ((nk_u32_t *)(result + row * result_stride_elements))[row] = nk_dots_reduce_sumsq_u8_(
            vectors + row * stride_elements, depth, nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_u32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_(vectors + col * stride_elements, depth,
                                                                             nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_f32_t query_norm_sq_f32 = (nk_f32_t)((nk_u32_t *)r_row)[row];
            nk_euclideans_row_u32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   query_norm_sq_f32, chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_sapphireamx( //
    nk_u8_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_u8_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                       (nk_u32_t *)result, result_stride_in_bytes, row_start,
                                                       row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_u8_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion U8 Symmetric

#pragma region E4M3 Packed

NUMKONG_INLINE void nk_angulars_packed_e4m3_sapphireamx_finalize_(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e4m3_(a + row * a_stride_elements, depth, nk_cap_sapphireamx_k);
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_packed_e4m3_sapphireamx(                     //
    nk_e4m3_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                          //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e4m3_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride_in_bytes,
                                                                c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_angulars_packed_e4m3_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_packed_e4m3_sapphireamx_finalize_(nk_e4m3_t const *a, void const *b_packed,
                                                                    nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                    nk_size_t depth, nk_size_t a_stride_elements,
                                                                    nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e4m3_(a + row * a_stride_elements, depth, nk_cap_sapphireamx_k);
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_sapphireamx(                   //
    nk_e4m3_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                          //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e4m3_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride_in_bytes,
                                                                c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_e4m3_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion E4M3 Packed

#pragma region E5M2 Packed

NUMKONG_INLINE void nk_angulars_packed_e5m2_sapphireamx_finalize_(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e5m2_(a + row * a_stride_elements, depth, nk_cap_sapphireamx_k);
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_packed_e5m2_sapphireamx(                     //
    nk_e5m2_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                          //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e5m2_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride_in_bytes,
                                                                c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_angulars_packed_e5m2_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_packed_e5m2_sapphireamx_finalize_(nk_e5m2_t const *a, void const *b_packed,
                                                                    nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                    nk_size_t depth, nk_size_t a_stride_elements,
                                                                    nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e5m2_(a + row * a_stride_elements, depth, nk_cap_sapphireamx_k);
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_sapphireamx(                   //
    nk_e5m2_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                          //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e5m2_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride_in_bytes,
                                                                c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_e5m2_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion E5M2 Packed

#pragma region E5M2 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_e5m2_sapphireamx_finalize_(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e5m2_(vectors + row * stride_elements, depth,
                                                                                nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_(vectors + col * stride_elements, depth,
                                                                               nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_sapphireamx( //
    nk_e5m2_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e5m2_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                         (nk_f32_t *)result, result_stride_in_bytes, row_start,
                                                         row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_e5m2_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_symmetric_e5m2_sapphireamx_finalize_(nk_e5m2_t const *vectors,
                                                                       nk_size_t vectors_count, nk_size_t depth,
                                                                       nk_size_t stride_elements, nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e5m2_(vectors + row * stride_elements, depth,
                                                                                nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_(vectors + col * stride_elements, depth,
                                                                               nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_sapphireamx( //
    nk_e5m2_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e5m2_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                         (nk_f32_t *)result, result_stride_in_bytes, row_start,
                                                         row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_e5m2_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion E5M2 Symmetric

#pragma region E4M3 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_e4m3_sapphireamx_finalize_(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e4m3_(vectors + row * stride_elements, depth,
                                                                                nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_(vectors + col * stride_elements, depth,
                                                                               nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_sapphireamx( //
    nk_e4m3_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e4m3_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                         (nk_f32_t *)result, result_stride_in_bytes, row_start,
                                                         row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_e4m3_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_symmetric_e4m3_sapphireamx_finalize_(nk_e4m3_t const *vectors,
                                                                       nk_size_t vectors_count, nk_size_t depth,
                                                                       nk_size_t stride_elements, nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e4m3_(vectors + row * stride_elements, depth,
                                                                                nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_(vectors + col * stride_elements, depth,
                                                                               nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_sapphireamx( //
    nk_e4m3_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e4m3_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                         (nk_f32_t *)result, result_stride_in_bytes, row_start,
                                                         row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_e4m3_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion E4M3 Symmetric

#pragma region E2M3 Packed

NUMKONG_INLINE void nk_angulars_packed_e2m3_sapphireamx_finalize_(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e2m3_(a + row * a_stride_elements, depth, nk_cap_sapphireamx_k);
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_packed_e2m3_sapphireamx(                     //
    nk_e2m3_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                          //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e2m3_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride_in_bytes,
                                                                c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_angulars_packed_e2m3_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_packed_e2m3_sapphireamx_finalize_(nk_e2m3_t const *a, void const *b_packed,
                                                                    nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                    nk_size_t depth, nk_size_t a_stride_elements,
                                                                    nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e2m3_(a + row * a_stride_elements, depth, nk_cap_sapphireamx_k);
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_sapphireamx(                   //
    nk_e2m3_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                          //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e2m3_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride_in_bytes,
                                                                c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_e2m3_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion E2M3 Packed

#pragma region E2M3 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_e2m3_sapphireamx_finalize_(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e2m3_(vectors + row * stride_elements, depth,
                                                                                nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_(vectors + col * stride_elements, depth,
                                                                               nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_sapphireamx( //
    nk_e2m3_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e2m3_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                         (nk_f32_t *)result, result_stride_in_bytes, row_start,
                                                         row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_e2m3_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_symmetric_e2m3_sapphireamx_finalize_(nk_e2m3_t const *vectors,
                                                                       nk_size_t vectors_count, nk_size_t depth,
                                                                       nk_size_t stride_elements, nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e2m3_(vectors + row * stride_elements, depth,
                                                                                nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_(vectors + col * stride_elements, depth,
                                                                               nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_sapphireamx( //
    nk_e2m3_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e2m3_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                         (nk_f32_t *)result, result_stride_in_bytes, row_start,
                                                         row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_e2m3_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion E2M3 Symmetric

#pragma region E2M1 Packed

NUMKONG_INLINE void nk_angulars_packed_e2m1_sapphireamx_finalize_(nk_e2m1x2_t const *a, void const *b_packed,
                                                                  nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                  nk_size_t depth, nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e2m1_(a + row * a_stride_elements, depth, nk_cap_sapphireamx_k);
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_packed_e2m1_sapphireamx(                       //
    nk_e2m1x2_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e2m1x2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e2m1_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride_in_bytes,
                                                                c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_angulars_packed_e2m1_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_packed_e2m1_sapphireamx_finalize_(nk_e2m1x2_t const *a, void const *b_packed,
                                                                    nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                    nk_size_t depth, nk_size_t a_stride_elements,
                                                                    nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e2m1_(a + row * a_stride_elements, depth, nk_cap_sapphireamx_k);
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_sapphireamx(                     //
    nk_e2m1x2_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                            //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e2m1x2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e2m1_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride_in_bytes,
                                                                c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_e2m1_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion E2M1 Packed

#pragma region E2M1 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_e2m1_sapphireamx_finalize_(nk_e2m1x2_t const *vectors,
                                                                     nk_size_t vectors_count, nk_size_t depth,
                                                                     nk_size_t stride_elements, nk_f32_t *result,
                                                                     nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e2m1_(vectors + row * stride_elements, depth,
                                                                                nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m1_(vectors + col * stride_elements, depth,
                                                                               nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_sapphireamx( //
    nk_e2m1x2_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 &&
               stride_in_bytes >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e2m1x2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e2m1_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                         (nk_f32_t *)result, result_stride_in_bytes, row_start,
                                                         row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_e2m1_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_symmetric_e2m1_sapphireamx_finalize_(nk_e2m1x2_t const *vectors,
                                                                       nk_size_t vectors_count, nk_size_t depth,
                                                                       nk_size_t stride_elements, nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e2m1_(vectors + row * stride_elements, depth,
                                                                                nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m1_(vectors + col * stride_elements, depth,
                                                                               nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_sapphireamx( //
    nk_e2m1x2_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 &&
               stride_in_bytes >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e2m1x2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e2m1_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                         (nk_f32_t *)result, result_stride_in_bytes, row_start,
                                                         row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_e2m1_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion E2M1 Symmetric

#pragma region E3M2 Packed

NUMKONG_INLINE void nk_angulars_packed_e3m2_sapphireamx_finalize_(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                                  nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                                  nk_size_t a_stride_elements,
                                                                  nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e3m2_(a + row * a_stride_elements, depth, nk_cap_sapphireamx_k);
        nk_angulars_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_packed_e3m2_sapphireamx(                     //
    nk_e3m2_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                          //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e3m2_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride_in_bytes,
                                                                c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_angulars_packed_e3m2_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                  c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_packed_e3m2_sapphireamx_finalize_(nk_e3m2_t const *a, void const *b_packed,
                                                                    nk_f32_t *c, nk_size_t rows, nk_size_t columns,
                                                                    nk_size_t depth, nk_size_t a_stride_elements,
                                                                    nk_size_t c_stride_elements) {
    nk_dots_amx_packed_header_t const *header = (nk_dots_amx_packed_header_t const *)b_packed;
    nk_f32_t const *b_norms = (nk_f32_t const *)((char const *)b_packed + header->norms_byte_offset);
    for (nk_size_t row = 0; row < rows; row++) {
        nk_f32_t query_norm_sq = nk_dots_reduce_sumsq_e3m2_(a + row * a_stride_elements, depth, nk_cap_sapphireamx_k);
        nk_euclideans_row_f32dots_sapphireamx_(c + row * c_stride_elements, b_norms, query_norm_sq, columns);
    }
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_sapphireamx(                   //
    nk_e3m2_t const *a, void const *a_scales, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,                          //
    nk_size_t a_stride_in_bytes, nk_size_t a_scales_stride, nk_size_t c_stride_in_bytes, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gemm_packed_e3m2_sapphireamx_(a, b_packed, c, rows, columns, depth, a_stride_in_bytes,
                                                                c_stride_in_bytes);
    if (status != nk_success_k) return status;
    nk_euclideans_packed_e3m2_sapphireamx_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements,
                                                    c_stride_elements);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion E3M2 Packed

#pragma region E3M2 Symmetric

NUMKONG_INLINE void nk_angulars_symmetric_e3m2_sapphireamx_finalize_(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                                     nk_size_t depth, nk_size_t stride_elements,
                                                                     nk_f32_t *result, nk_size_t result_stride_elements,
                                                                     nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e3m2_(vectors + row * stride_elements, depth,
                                                                                nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_(vectors + col * stride_elements, depth,
                                                                               nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_angulars_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                 r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_sapphireamx( //
    nk_e3m2_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e3m2_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                         (nk_f32_t *)result, result_stride_in_bytes, row_start,
                                                         row_count);
    if (status != nk_success_k) return status;
    nk_angulars_symmetric_e3m2_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                     result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_INLINE void nk_euclideans_symmetric_e3m2_sapphireamx_finalize_(nk_e3m2_t const *vectors,
                                                                       nk_size_t vectors_count, nk_size_t depth,
                                                                       nk_size_t stride_elements, nk_f32_t *result,
                                                                       nk_size_t result_stride_elements,
                                                                       nk_size_t row_start, nk_size_t row_count) {

    // Cache row norms on diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++)
        result[row * result_stride_elements + row] = nk_dots_reduce_sumsq_e3m2_(vectors + row * stride_elements, depth,
                                                                                nk_cap_sapphireamx_k);

    // 256-column chunks with cached norms
    nk_f32_t column_norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; col++)
            column_norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_(vectors + col * stride_elements, depth,
                                                                               nk_cap_sapphireamx_k);

        for (nk_size_t row = row_start; row < row_start + row_count; row++) {
            nk_f32_t *r_row = result + row * result_stride_elements;
            nk_size_t col_start = chunk_start > row + 1 ? chunk_start : row + 1;
            if (col_start >= chunk_end) continue;
            nk_euclideans_row_f32dots_sapphireamx_(r_row + col_start, column_norms_cache + col_start - chunk_start,
                                                   r_row[row], chunk_end - col_start);
        }
    }

    // Zero diagonal
    for (nk_size_t row = row_start; row < row_start + row_count; row++) result[row * result_stride_elements + row] = 0;
}

#if NUMKONG_TARGET_SAPPHIREAMX

NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_sapphireamx( //
    nk_e3m2_t const *vectors, void const *vector_scales, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride_in_bytes, nk_size_t scales_stride, //
    nk_f32_t *result, nk_size_t result_stride_in_bytes, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride_in_bytes % sizeof(*vectors) == 0 && stride_in_bytes >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride_in_bytes / sizeof(nk_f32_t);
    nk_status_t const status = nk_gram_e3m2_sapphireamx_(vectors, vectors_count, depth, stride_in_bytes,
                                                         (nk_f32_t *)result, result_stride_in_bytes, row_start,
                                                         row_count);
    if (status != nk_success_k) return status;
    nk_euclideans_symmetric_e3m2_sapphireamx_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                                       result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_SAPPHIREAMX

#pragma endregion E3M2 Symmetric

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_SAPPHIREAMX_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_SPATIALS_SAPPHIREAMX_H
