/**
 *  @brief Batched Spatial Distances for RISC-V Vector (RVV).
 *  @file include/numkong/spatials/rvv.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NK_SPATIALS_RVV_H
#define NK_SPATIALS_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/dots/serial.h"
#include "numkong/dots/rvv.h"
#include "numkong/spatial/rvv.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#pragma region Single Precision Floats

NK_INTERNAL void nk_angulars_packed_f32_rvv_finalize_(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f64_t const *target_norms = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f32_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *result_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f32_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f64_t *result_ptr = result_row;
        nk_f64_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e64m1(count_columns);
            vfloat64m1_t dots_f64m1 = __riscv_vle64_v_f64m1(result_ptr, vector_length);
            vfloat64m1_t target_norms_sq_f64m1 = __riscv_vle64_v_f64m1(norms_ptr, vector_length);
            vfloat64m1_t norms_product_f64m1 = __riscv_vfmul_vf_f64m1(target_norms_sq_f64m1, query_norm_sq_f64,
                                                                      vector_length);
            vfloat64m1_t rsqrt_f64m1 = nk_rsqrt_f64m1_rvv_(norms_product_f64m1, vector_length);
            vfloat64m1_t normalized_dots_f64m1 = __riscv_vfmul_vv_f64m1(dots_f64m1, rsqrt_f64m1, vector_length);
            vfloat64m1_t angular_f64m1 = __riscv_vfrsub_vf_f64m1(normalized_dots_f64m1, 1.0, vector_length);
            angular_f64m1 = __riscv_vfmax_vf_f64m1(angular_f64m1, 0.0, vector_length);
            __riscv_vse64_v_f64m1(result_ptr, angular_f64m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_angulars_packed_f32_rvv(                //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f64_t);
    nk_dots_packed_f32_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_f32_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_f32_rvv_finalize_(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f64_t const *target_norms = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f32_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *result_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f32_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f64_t *result_ptr = result_row;
        nk_f64_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e64m1(count_columns);
            vfloat64m1_t dots_f64m1 = __riscv_vle64_v_f64m1(result_ptr, vector_length);
            vfloat64m1_t target_norms_sq_f64m1 = __riscv_vle64_v_f64m1(norms_ptr, vector_length);
            vfloat64m1_t sum_sq_f64m1 = __riscv_vfadd_vf_f64m1(target_norms_sq_f64m1, query_norm_sq_f64, vector_length);
            vfloat64m1_t dist_sq_f64m1 = __riscv_vfsub_vv_f64m1(
                sum_sq_f64m1, __riscv_vfmul_vf_f64m1(dots_f64m1, 2.0, vector_length), vector_length);
            dist_sq_f64m1 = __riscv_vfmax_vf_f64m1(dist_sq_f64m1, 0.0, vector_length);
            __riscv_vse64_v_f64m1(result_ptr, __riscv_vfsqrt_v_f64m1(dist_sq_f64m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_f32_rvv(              //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f64_t);
    nk_dots_packed_f32_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_f32_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_angulars_symmetric_f32_rvv_finalize_(nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                                         nk_size_t stride_elements, nk_f64_t *result,
                                                         nk_size_t result_stride_elements, nk_size_t row_start,
                                                         nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f32_(vectors + row_index * stride_elements, depth);
    }
    nk_f64_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f32_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f64_t *result_row = result + row_index * result_stride_elements;
            nk_f64_t query_norm_sq_f64 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f64_t *result_ptr = result_row + col_start;
            nk_f64_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e64m1(count_remaining);
                vfloat64m1_t dots_f64m1 = __riscv_vle64_v_f64m1(result_ptr, vector_length);
                vfloat64m1_t target_norms_sq_f64m1 = __riscv_vle64_v_f64m1(norms_ptr, vector_length);
                vfloat64m1_t norms_product_f64m1 = __riscv_vfmul_vf_f64m1(target_norms_sq_f64m1, query_norm_sq_f64,
                                                                          vector_length);
                vfloat64m1_t rsqrt_f64m1 = nk_rsqrt_f64m1_rvv_(norms_product_f64m1, vector_length);
                vfloat64m1_t normalized_dots_f64m1 = __riscv_vfmul_vv_f64m1(dots_f64m1, rsqrt_f64m1, vector_length);
                vfloat64m1_t angular_f64m1 = __riscv_vfrsub_vf_f64m1(normalized_dots_f64m1, 1.0, vector_length);
                angular_f64m1 = __riscv_vfmax_vf_f64m1(angular_f64m1, 0.0, vector_length);
                __riscv_vse64_v_f64m1(result_ptr, angular_f64m1, vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_f32_rvv(                                        //
    nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_dots_symmetric_f32_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_angulars_symmetric_f32_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                            row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_f32_rvv_finalize_(nk_f32_t const *vectors, nk_size_t n_vectors,
                                                           nk_size_t depth, nk_size_t stride_elements, nk_f64_t *result,
                                                           nk_size_t result_stride_elements, nk_size_t row_start,
                                                           nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f32_(vectors + row_index * stride_elements, depth);
    }
    nk_f64_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f32_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f64_t *result_row = result + row_index * result_stride_elements;
            nk_f64_t query_norm_sq_f64 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f64_t *result_ptr = result_row + col_start;
            nk_f64_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e64m1(count_remaining);
                vfloat64m1_t dots_f64m1 = __riscv_vle64_v_f64m1(result_ptr, vector_length);
                vfloat64m1_t target_norms_sq_f64m1 = __riscv_vle64_v_f64m1(norms_ptr, vector_length);
                vfloat64m1_t sum_sq_f64m1 = __riscv_vfadd_vf_f64m1(target_norms_sq_f64m1, query_norm_sq_f64,
                                                                   vector_length);
                vfloat64m1_t dist_sq_f64m1 = __riscv_vfsub_vv_f64m1(
                    sum_sq_f64m1, __riscv_vfmul_vf_f64m1(dots_f64m1, 2.0, vector_length), vector_length);
                dist_sq_f64m1 = __riscv_vfmax_vf_f64m1(dist_sq_f64m1, 0.0, vector_length);
                __riscv_vse64_v_f64m1(result_ptr, __riscv_vfsqrt_v_f64m1(dist_sq_f64m1, vector_length), vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_f32_rvv(                                      //
    nk_f32_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_dots_symmetric_f32_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_euclideans_symmetric_f32_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                              result_stride_elements, row_start, row_count);
}

#pragma endregion // Single Precision Floats

#pragma region Double Precision Floats

NK_INTERNAL void nk_angulars_packed_f64_rvv_finalize_(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f64_t const *target_norms = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f64_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f64_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *result_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f64_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f64_t *result_ptr = result_row;
        nk_f64_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e64m1(count_columns);
            vfloat64m1_t dots_f64m1 = __riscv_vle64_v_f64m1(result_ptr, vector_length);
            vfloat64m1_t target_norms_sq_f64m1 = __riscv_vle64_v_f64m1(norms_ptr, vector_length);
            vfloat64m1_t norms_product_f64m1 = __riscv_vfmul_vf_f64m1(target_norms_sq_f64m1, query_norm_sq_f64,
                                                                      vector_length);
            vfloat64m1_t rsqrt_f64m1 = nk_rsqrt_f64m1_rvv_(norms_product_f64m1, vector_length);
            vfloat64m1_t normalized_dots_f64m1 = __riscv_vfmul_vv_f64m1(dots_f64m1, rsqrt_f64m1, vector_length);
            vfloat64m1_t angular_f64m1 = __riscv_vfrsub_vf_f64m1(normalized_dots_f64m1, 1.0, vector_length);
            angular_f64m1 = __riscv_vfmax_vf_f64m1(angular_f64m1, 0.0, vector_length);
            __riscv_vse64_v_f64m1(result_ptr, angular_f64m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_angulars_packed_f64_rvv(                //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f64_t);
    nk_dots_packed_f64_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_f64_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_f64_rvv_finalize_(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f64_t const *target_norms = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f64_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f64_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *result_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f64_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f64_t *result_ptr = result_row;
        nk_f64_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e64m1(count_columns);
            vfloat64m1_t dots_f64m1 = __riscv_vle64_v_f64m1(result_ptr, vector_length);
            vfloat64m1_t target_norms_sq_f64m1 = __riscv_vle64_v_f64m1(norms_ptr, vector_length);
            vfloat64m1_t sum_sq_f64m1 = __riscv_vfadd_vf_f64m1(target_norms_sq_f64m1, query_norm_sq_f64, vector_length);
            vfloat64m1_t dist_sq_f64m1 = __riscv_vfsub_vv_f64m1(
                sum_sq_f64m1, __riscv_vfmul_vf_f64m1(dots_f64m1, 2.0, vector_length), vector_length);
            dist_sq_f64m1 = __riscv_vfmax_vf_f64m1(dist_sq_f64m1, 0.0, vector_length);
            __riscv_vse64_v_f64m1(result_ptr, __riscv_vfsqrt_v_f64m1(dist_sq_f64m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_f64_rvv(              //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f64_t);
    nk_dots_packed_f64_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_f64_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_angulars_symmetric_f64_rvv_finalize_(nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                                         nk_size_t stride_elements, nk_f64_t *result,
                                                         nk_size_t result_stride_elements, nk_size_t row_start,
                                                         nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f64_(vectors + row_index * stride_elements, depth);
    }
    nk_f64_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f64_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f64_t *result_row = result + row_index * result_stride_elements;
            nk_f64_t query_norm_sq_f64 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f64_t *result_ptr = result_row + col_start;
            nk_f64_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e64m1(count_remaining);
                vfloat64m1_t dots_f64m1 = __riscv_vle64_v_f64m1(result_ptr, vector_length);
                vfloat64m1_t target_norms_sq_f64m1 = __riscv_vle64_v_f64m1(norms_ptr, vector_length);
                vfloat64m1_t norms_product_f64m1 = __riscv_vfmul_vf_f64m1(target_norms_sq_f64m1, query_norm_sq_f64,
                                                                          vector_length);
                vfloat64m1_t rsqrt_f64m1 = nk_rsqrt_f64m1_rvv_(norms_product_f64m1, vector_length);
                vfloat64m1_t normalized_dots_f64m1 = __riscv_vfmul_vv_f64m1(dots_f64m1, rsqrt_f64m1, vector_length);
                vfloat64m1_t angular_f64m1 = __riscv_vfrsub_vf_f64m1(normalized_dots_f64m1, 1.0, vector_length);
                angular_f64m1 = __riscv_vfmax_vf_f64m1(angular_f64m1, 0.0, vector_length);
                __riscv_vse64_v_f64m1(result_ptr, angular_f64m1, vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_f64_rvv(                                        //
    nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_dots_symmetric_f64_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_angulars_symmetric_f64_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                            row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_f64_rvv_finalize_(nk_f64_t const *vectors, nk_size_t n_vectors,
                                                           nk_size_t depth, nk_size_t stride_elements, nk_f64_t *result,
                                                           nk_size_t result_stride_elements, nk_size_t row_start,
                                                           nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f64_(vectors + row_index * stride_elements, depth);
    }
    nk_f64_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f64_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f64_t *result_row = result + row_index * result_stride_elements;
            nk_f64_t query_norm_sq_f64 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f64_t *result_ptr = result_row + col_start;
            nk_f64_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e64m1(count_remaining);
                vfloat64m1_t dots_f64m1 = __riscv_vle64_v_f64m1(result_ptr, vector_length);
                vfloat64m1_t target_norms_sq_f64m1 = __riscv_vle64_v_f64m1(norms_ptr, vector_length);
                vfloat64m1_t sum_sq_f64m1 = __riscv_vfadd_vf_f64m1(target_norms_sq_f64m1, query_norm_sq_f64,
                                                                   vector_length);
                vfloat64m1_t dist_sq_f64m1 = __riscv_vfsub_vv_f64m1(
                    sum_sq_f64m1, __riscv_vfmul_vf_f64m1(dots_f64m1, 2.0, vector_length), vector_length);
                dist_sq_f64m1 = __riscv_vfmax_vf_f64m1(dist_sq_f64m1, 0.0, vector_length);
                __riscv_vse64_v_f64m1(result_ptr, __riscv_vfsqrt_v_f64m1(dist_sq_f64m1, vector_length), vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_f64_rvv(                                      //
    nk_f64_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_dots_symmetric_f64_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_euclideans_symmetric_f64_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                              result_stride_elements, row_start, row_count);
}

#pragma endregion // Double Precision Floats

#pragma region Half Precision Floats

NK_INTERNAL void nk_angulars_packed_f16_rvv_finalize_(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                      nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                      nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_f16_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                      vector_length);
            vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
            vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
            vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
            angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_angulars_packed_f16_rvv(                //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_f16_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_f16_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_f16_rvv_finalize_(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_f16_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32, vector_length);
            vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
            dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_f16_rvv(              //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_f16_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_f16_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_angulars_symmetric_f16_rvv_finalize_(nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                                         nk_size_t stride_elements, nk_f32_t *result,
                                                         nk_size_t result_stride_elements, nk_size_t row_start,
                                                         nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_(vectors + row_index * stride_elements, depth);
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_f32_t query_norm_sq_f32 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_f32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
                vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                          vector_length);
                vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
                vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
                vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
                angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_f16_rvv(                                        //
    nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_angulars_symmetric_f16_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                            row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_f16_rvv_finalize_(nk_f16_t const *vectors, nk_size_t n_vectors,
                                                           nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
                                                           nk_size_t result_stride_elements, nk_size_t row_start,
                                                           nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_(vectors + row_index * stride_elements, depth);
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_f32_t query_norm_sq_f32 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_f32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
                vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                   vector_length);
                vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                    sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
                dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_f16_rvv(                                      //
    nk_f16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_euclideans_symmetric_f16_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                              result_stride_elements, row_start, row_count);
}

#pragma endregion // Half Precision Floats

#pragma region Brain Float 16

NK_INTERNAL void nk_angulars_packed_bf16_rvv_finalize_(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_bf16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_bf16_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                      vector_length);
            vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
            vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
            vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
            angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_angulars_packed_bf16_rvv(                //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_bf16_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_bf16_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_bf16_rvv_finalize_(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_bf16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_bf16_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32, vector_length);
            vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
            dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_bf16_rvv(              //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_bf16_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_bf16_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_angulars_symmetric_bf16_rvv_finalize_(nk_bf16_t const *vectors, nk_size_t n_vectors,
                                                          nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
                                                          nk_size_t result_stride_elements, nk_size_t row_start,
                                                          nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_(vectors + row_index * stride_elements, depth);
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_f32_t query_norm_sq_f32 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_f32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
                vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                          vector_length);
                vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
                vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
                vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
                angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_bf16_rvv(                                        //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_angulars_symmetric_bf16_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                             row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_bf16_rvv_finalize_(nk_bf16_t const *vectors, nk_size_t n_vectors,
                                                            nk_size_t depth, nk_size_t stride_elements,
                                                            nk_f32_t *result, nk_size_t result_stride_elements,
                                                            nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_(vectors + row_index * stride_elements, depth);
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_f32_t query_norm_sq_f32 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_f32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
                vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                   vector_length);
                vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                    sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
                dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_bf16_rvv(                                      //
    nk_bf16_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_euclideans_symmetric_bf16_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                               result_stride_elements, row_start, row_count);
}

#pragma endregion // Brain Float 16

#pragma region Micro Precision E2M3

NK_INTERNAL void nk_angulars_packed_e2m3_rvv_finalize_(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_e2m3_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m3_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                      vector_length);
            vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
            vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
            vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
            angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_angulars_packed_e2m3_rvv(                //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e2m3_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_e2m3_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_e2m3_rvv_finalize_(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_e2m3_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m3_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32, vector_length);
            vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
            dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_e2m3_rvv(              //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e2m3_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_e2m3_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_angulars_symmetric_e2m3_rvv_finalize_(nk_e2m3_t const *vectors, nk_size_t n_vectors,
                                                          nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
                                                          nk_size_t result_stride_elements, nk_size_t row_start,
                                                          nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_(vectors + row_index * stride_elements, depth);
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_f32_t query_norm_sq_f32 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_f32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
                vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                          vector_length);
                vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
                vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
                vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
                angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e2m3_rvv(                                        //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_angulars_symmetric_e2m3_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                             row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_e2m3_rvv_finalize_(nk_e2m3_t const *vectors, nk_size_t n_vectors,
                                                            nk_size_t depth, nk_size_t stride_elements,
                                                            nk_f32_t *result, nk_size_t result_stride_elements,
                                                            nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_(vectors + row_index * stride_elements, depth);
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_f32_t query_norm_sq_f32 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_f32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
                vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                   vector_length);
                vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                    sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
                dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e2m3_rvv(                                      //
    nk_e2m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_euclideans_symmetric_e2m3_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                               result_stride_elements, row_start, row_count);
}

#pragma endregion // Micro Precision E2M3

#pragma region Micro Precision E3M2

NK_INTERNAL void nk_angulars_packed_e3m2_rvv_finalize_(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_i16_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e3m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e3m2_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                      vector_length);
            vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
            vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
            vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
            angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_angulars_packed_e3m2_rvv(                //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e3m2_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_e3m2_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_e3m2_rvv_finalize_(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_i16_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e3m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e3m2_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32, vector_length);
            vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
            dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_e3m2_rvv(              //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e3m2_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_e3m2_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_angulars_symmetric_e3m2_rvv_finalize_(nk_e3m2_t const *vectors, nk_size_t n_vectors,
                                                          nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
                                                          nk_size_t result_stride_elements, nk_size_t row_start,
                                                          nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_(vectors + row_index * stride_elements, depth);
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_f32_t query_norm_sq_f32 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_f32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
                vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                          vector_length);
                vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
                vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
                vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
                angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e3m2_rvv(                                        //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_angulars_symmetric_e3m2_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                             row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_e3m2_rvv_finalize_(nk_e3m2_t const *vectors, nk_size_t n_vectors,
                                                            nk_size_t depth, nk_size_t stride_elements,
                                                            nk_f32_t *result, nk_size_t result_stride_elements,
                                                            nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_(vectors + row_index * stride_elements, depth);
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_f32_t query_norm_sq_f32 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_f32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
                vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                   vector_length);
                vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                    sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
                dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e3m2_rvv(                                      //
    nk_e3m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_euclideans_symmetric_e3m2_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                               result_stride_elements, row_start, row_count);
}

#pragma endregion // Micro Precision E3M2

#pragma region Quarter Precision E4M3

NK_INTERNAL void nk_angulars_packed_e4m3_rvv_finalize_(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e4m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e4m3_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                      vector_length);
            vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
            vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
            vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
            angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_angulars_packed_e4m3_rvv(                //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e4m3_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_e4m3_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_e4m3_rvv_finalize_(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e4m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e4m3_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32, vector_length);
            vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
            dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_e4m3_rvv(              //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e4m3_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_e4m3_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_angulars_symmetric_e4m3_rvv_finalize_(nk_e4m3_t const *vectors, nk_size_t n_vectors,
                                                          nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
                                                          nk_size_t result_stride_elements, nk_size_t row_start,
                                                          nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_(vectors + row_index * stride_elements, depth);
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_f32_t query_norm_sq_f32 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_f32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
                vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                          vector_length);
                vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
                vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
                vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
                angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e4m3_rvv(                                        //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_angulars_symmetric_e4m3_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                             row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_e4m3_rvv_finalize_(nk_e4m3_t const *vectors, nk_size_t n_vectors,
                                                            nk_size_t depth, nk_size_t stride_elements,
                                                            nk_f32_t *result, nk_size_t result_stride_elements,
                                                            nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_(vectors + row_index * stride_elements, depth);
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_f32_t query_norm_sq_f32 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_f32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
                vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                   vector_length);
                vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                    sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
                dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e4m3_rvv(                                      //
    nk_e4m3_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_euclideans_symmetric_e4m3_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                               result_stride_elements, row_start, row_count);
}

#pragma endregion // Quarter Precision E4M3

#pragma region Quarter Precision E5M2

NK_INTERNAL void nk_angulars_packed_e5m2_rvv_finalize_(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e5m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e5m2_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                      vector_length);
            vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
            vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
            vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
            angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_angulars_packed_e5m2_rvv(                //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e5m2_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_e5m2_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_e5m2_rvv_finalize_(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e5m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e5m2_(a_row, depth);
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32, vector_length);
            vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
            dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_e5m2_rvv(              //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_e5m2_rvv(a, b_packed, c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_e5m2_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_angulars_symmetric_e5m2_rvv_finalize_(nk_e5m2_t const *vectors, nk_size_t n_vectors,
                                                          nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
                                                          nk_size_t result_stride_elements, nk_size_t row_start,
                                                          nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_(vectors + row_index * stride_elements, depth);
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_f32_t query_norm_sq_f32 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_f32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
                vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                          vector_length);
                vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
                vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
                vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
                angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_e5m2_rvv(                                        //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_angulars_symmetric_e5m2_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                             row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_e5m2_rvv_finalize_(nk_e5m2_t const *vectors, nk_size_t n_vectors,
                                                            nk_size_t depth, nk_size_t stride_elements,
                                                            nk_f32_t *result, nk_size_t result_stride_elements,
                                                            nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_(vectors + row_index * stride_elements, depth);
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_f32_t query_norm_sq_f32 = result_row[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_f32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
                vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                   vector_length);
                vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                    sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
                dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_e5m2_rvv(                                      //
    nk_e5m2_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_rvv(vectors, n_vectors, depth, stride, result, result_stride, row_start, row_count);
    nk_euclideans_symmetric_e5m2_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result,
                                               result_stride_elements, row_start, row_count);
}

#pragma endregion // Quarter Precision E5M2

#pragma region Signed 8-bit Integers

NK_INTERNAL void nk_angulars_packed_i8_rvv_finalize_(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_u32_t const *target_norms = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_i8_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq = nk_dots_reduce_sumsq_i8_(a_row, depth);
        nk_f32_t query_norm_sq_f32 = (nk_f32_t)query_norm_sq;
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_u32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vfcvt_f_x_v_f32m1(
                __riscv_vle32_v_i32m1((nk_i32_t const *)result_ptr, vector_length), vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vfcvt_f_xu_v_f32m1(
                __riscv_vle32_v_u32m1((nk_u32_t const *)norms_ptr, vector_length), vector_length);
            vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                      vector_length);
            vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
            vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
            vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
            angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_angulars_packed_i8_rvv(                //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_i8_rvv(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_i8_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_i8_rvv_finalize_(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_u32_t const *target_norms = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_i8_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq = nk_dots_reduce_sumsq_i8_(a_row, depth);
        nk_f32_t query_norm_sq_f32 = (nk_f32_t)query_norm_sq;
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_u32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vfcvt_f_x_v_f32m1(
                __riscv_vle32_v_i32m1((nk_i32_t const *)result_ptr, vector_length), vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vfcvt_f_xu_v_f32m1(
                __riscv_vle32_v_u32m1((nk_u32_t const *)norms_ptr, vector_length), vector_length);
            vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32, vector_length);
            vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
            dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_i8_rvv(              //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_i8_rvv(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_i8_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_angulars_symmetric_i8_rvv_finalize_(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                                        nk_size_t stride_elements, nk_f32_t *result,
                                                        nk_size_t result_stride_elements, nk_size_t row_start,
                                                        nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t norm = nk_dots_reduce_sumsq_i8_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = norm;
    }
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_norm_sq = ((nk_u32_t *)result_row)[row_index];
            nk_f32_t query_norm_sq_f32 = (nk_f32_t)query_norm_sq;
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_u32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vfcvt_f_x_v_f32m1(
                    __riscv_vle32_v_i32m1((nk_i32_t const *)result_ptr, vector_length), vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vfcvt_f_xu_v_f32m1(
                    __riscv_vle32_v_u32m1((nk_u32_t const *)norms_ptr, vector_length), vector_length);
                vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                          vector_length);
                vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
                vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
                vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
                angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_i8_rvv(                                        //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i8_rvv(vectors, n_vectors, depth, stride, (nk_i32_t *)result, result_stride, row_start,
                             row_count);
    nk_angulars_symmetric_i8_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                           row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_i8_rvv_finalize_(nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                                          nk_size_t stride_elements, nk_f32_t *result,
                                                          nk_size_t result_stride_elements, nk_size_t row_start,
                                                          nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t norm = nk_dots_reduce_sumsq_i8_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = norm;
    }
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_norm_sq = ((nk_u32_t *)result_row)[row_index];
            nk_f32_t query_norm_sq_f32 = (nk_f32_t)query_norm_sq;
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_u32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vfcvt_f_x_v_f32m1(
                    __riscv_vle32_v_i32m1((nk_i32_t const *)result_ptr, vector_length), vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vfcvt_f_xu_v_f32m1(
                    __riscv_vle32_v_u32m1((nk_u32_t const *)norms_ptr, vector_length), vector_length);
                vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                   vector_length);
                vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                    sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
                dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_i8_rvv(                                      //
    nk_i8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i8_rvv(vectors, n_vectors, depth, stride, (nk_i32_t *)result, result_stride, row_start,
                             row_count);
    nk_euclideans_symmetric_i8_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                             row_start, row_count);
}

#pragma endregion // Signed 8-bit Integers

#pragma region Unsigned 8-bit Integers

NK_INTERNAL void nk_angulars_packed_u8_rvv_finalize_(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                     nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                     nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_u32_t const *target_norms = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_u8_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq = nk_dots_reduce_sumsq_u8_(a_row, depth);
        nk_f32_t query_norm_sq_f32 = (nk_f32_t)query_norm_sq;
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_u32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vfcvt_f_xu_v_f32m1(
                __riscv_vle32_v_u32m1((nk_u32_t const *)result_ptr, vector_length), vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vfcvt_f_xu_v_f32m1(
                __riscv_vle32_v_u32m1((nk_u32_t const *)norms_ptr, vector_length), vector_length);
            vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                      vector_length);
            vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
            vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
            vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
            angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_angulars_packed_u8_rvv(                //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_u8_rvv(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_angulars_packed_u8_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_euclideans_packed_u8_rvv_finalize_(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                       nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                       nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_u32_t const *target_norms = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_u8_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq = nk_dots_reduce_sumsq_u8_(a_row, depth);
        nk_f32_t query_norm_sq_f32 = (nk_f32_t)query_norm_sq;
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_u32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vfcvt_f_xu_v_f32m1(
                __riscv_vle32_v_u32m1((nk_u32_t const *)result_ptr, vector_length), vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vfcvt_f_xu_v_f32m1(
                __riscv_vle32_v_u32m1((nk_u32_t const *)norms_ptr, vector_length), vector_length);
            vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32, vector_length);
            vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
            dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NK_PUBLIC void nk_euclideans_packed_u8_rvv(              //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride_in_bytes, nk_size_t c_stride_in_bytes) {
    nk_size_t const a_stride_elements = a_stride_in_bytes / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride_in_bytes / sizeof(nk_f32_t);
    nk_dots_packed_u8_rvv(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride_in_bytes, c_stride_in_bytes);
    nk_euclideans_packed_u8_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
}

NK_INTERNAL void nk_angulars_symmetric_u8_rvv_finalize_(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                                        nk_size_t stride_elements, nk_f32_t *result,
                                                        nk_size_t result_stride_elements, nk_size_t row_start,
                                                        nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t norm = nk_dots_reduce_sumsq_u8_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = norm;
    }
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_norm_sq = ((nk_u32_t *)result_row)[row_index];
            nk_f32_t query_norm_sq_f32 = (nk_f32_t)query_norm_sq;
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_u32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vfcvt_f_xu_v_f32m1(
                    __riscv_vle32_v_u32m1((nk_u32_t const *)result_ptr, vector_length), vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vfcvt_f_xu_v_f32m1(
                    __riscv_vle32_v_u32m1((nk_u32_t const *)norms_ptr, vector_length), vector_length);
                vfloat32m1_t norms_product_f32m1 = __riscv_vfmul_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                          vector_length);
                vfloat32m1_t rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(norms_product_f32m1, vector_length);
                vfloat32m1_t normalized_dots_f32m1 = __riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length);
                vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(normalized_dots_f32m1, 1.0f, vector_length);
                angular_f32m1 = __riscv_vfmax_vf_f32m1(angular_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_angulars_symmetric_u8_rvv(                                        //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u8_rvv(vectors, n_vectors, depth, stride, (nk_u32_t *)result, result_stride, row_start,
                             row_count);
    nk_angulars_symmetric_u8_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                           row_start, row_count);
}

NK_INTERNAL void nk_euclideans_symmetric_u8_rvv_finalize_(nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth,
                                                          nk_size_t stride_elements, nk_f32_t *result,
                                                          nk_size_t result_stride_elements, nk_size_t row_start,
                                                          nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t norm = nk_dots_reduce_sumsq_u8_(vectors + row_index * stride_elements, depth);
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = norm;
    }
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < n_vectors; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < n_vectors ? chunk_start + 256 : n_vectors;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_(vectors + col * stride_elements, depth);
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_norm_sq = ((nk_u32_t *)result_row)[row_index];
            nk_f32_t query_norm_sq_f32 = (nk_f32_t)query_norm_sq;
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_u32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vfloat32m1_t dots_f32m1 = __riscv_vfcvt_f_xu_v_f32m1(
                    __riscv_vle32_v_u32m1((nk_u32_t const *)result_ptr, vector_length), vector_length);
                vfloat32m1_t target_norms_sq_f32m1 = __riscv_vfcvt_f_xu_v_f32m1(
                    __riscv_vle32_v_u32m1((nk_u32_t const *)norms_ptr, vector_length), vector_length);
                vfloat32m1_t sum_sq_f32m1 = __riscv_vfadd_vf_f32m1(target_norms_sq_f32m1, query_norm_sq_f32,
                                                                   vector_length);
                vfloat32m1_t dist_sq_f32m1 = __riscv_vfsub_vv_f32m1(
                    sum_sq_f32m1, __riscv_vfmul_vf_f32m1(dots_f32m1, 2.0f, vector_length), vector_length);
                dist_sq_f32m1 = __riscv_vfmax_vf_f32m1(dist_sq_f32m1, 0.0f, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

NK_PUBLIC void nk_euclideans_symmetric_u8_rvv(                                      //
    nk_u8_t const *vectors, nk_size_t n_vectors, nk_size_t depth, nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count) {
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u8_rvv(vectors, n_vectors, depth, stride, (nk_u32_t *)result, result_stride, row_start,
                             row_count);
    nk_euclideans_symmetric_u8_rvv_finalize_(vectors, n_vectors, depth, stride_elements, result, result_stride_elements,
                                             row_start, row_count);
}

#pragma endregion // Unsigned 8-bit Integers

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_SPATIALS_RVV_H
