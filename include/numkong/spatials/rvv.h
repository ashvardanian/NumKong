/**
 *  @file include/numkong/spatials/rvv.h
 *  @author Ash Vardanian
 *  @date February 23, 2026
 *  @brief Batched spatial distances for RISC-V Vector, RVV.
 *
 *  @sa include/numkong/spatials.h
 */
#ifndef NUMKONG_SPATIALS_RVV_H
#define NUMKONG_SPATIALS_RVV_H

#if NUMKONG_ARCH_RISCV64_
#if NUMKONG_ARCH_RISCV64_RVV_

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

/** Angular distances of one f32 register group from dots and squared norms, as the serial ones. */
NUMKONG_INLINE vfloat32m1_t nk_angular_f32m1_from_dot_rvv_(vfloat32m1_t dots_f32m1, nk_f32_t query_norm_sq_f32,
                                                           vfloat32m1_t target_norms_sq_f32m1, size_t vector_length) {
    // The Newton steps turn the estimate of 0 for an infinite input into NaN
    vfloat32m1_t target_rsqrt_f32m1 = nk_rsqrt_f32m1_rvv_(target_norms_sq_f32m1, vector_length);
    vbool32_t infinite_b32 = __riscv_vmfeq_vf_f32m1_b32(target_norms_sq_f32m1, NUMKONG_F32_INF, vector_length);
    target_rsqrt_f32m1 = __riscv_vfmerge_vfm_f32m1(target_rsqrt_f32m1, 0.0f, infinite_b32, vector_length);
    vfloat32m1_t rsqrt_f32m1 = __riscv_vfmul_vf_f32m1(target_rsqrt_f32m1, nk_f32_rsqrt_newton_rvv_(query_norm_sq_f32),
                                                      vector_length);
    vfloat32m1_t angular_f32m1 = __riscv_vfrsub_vf_f32m1(__riscv_vfmul_vv_f32m1(dots_f32m1, rsqrt_f32m1, vector_length),
                                                         1.0f, vector_length);
    vbool32_t target_zero_b32 = __riscv_vmfeq_vf_f32m1_b32(target_norms_sq_f32m1, 0.0f, vector_length);
    vbool32_t one_b32 = __riscv_vmor_mm_b32(target_zero_b32,
                                            __riscv_vmfeq_vf_f32m1_b32(dots_f32m1, 0.0f, vector_length), vector_length);
    if (query_norm_sq_f32 == 0.0f) one_b32 = __riscv_vmset_m_b32(vector_length);
    angular_f32m1 = __riscv_vfmerge_vfm_f32m1(angular_f32m1, 1.0f, one_b32, vector_length);
    if (query_norm_sq_f32 == 0.0f)
        angular_f32m1 = __riscv_vfmerge_vfm_f32m1(angular_f32m1, 0.0f, target_zero_b32, vector_length);
    angular_f32m1 = __riscv_vmerge_vvm_f32m1(
        angular_f32m1, dots_f32m1, __riscv_vmfne_vv_f32m1_b32(dots_f32m1, dots_f32m1, vector_length), vector_length);
    // `vfmax` would return the other operand for a NaN, so the clamp is a merge
    return __riscv_vfmerge_vfm_f32m1(angular_f32m1, 0.0f,
                                     __riscv_vmflt_vf_f32m1_b32(angular_f32m1, 0.0f, vector_length), vector_length);
}

/** Angular distances of one f64 register group from dots and squared norms, as the serial ones. */
NUMKONG_INLINE vfloat64m1_t nk_angular_f64m1_from_dot_rvv_(vfloat64m1_t dots_f64m1, nk_f64_t query_norm_sq_f64,
                                                           vfloat64m1_t target_norms_sq_f64m1, size_t vector_length) {
    vfloat64m1_t target_rsqrt_f64m1 = nk_rsqrt_f64m1_rvv_(target_norms_sq_f64m1, vector_length);
    vbool64_t infinite_b64 = __riscv_vmfeq_vf_f64m1_b64(target_norms_sq_f64m1, NUMKONG_F64_INF, vector_length);
    target_rsqrt_f64m1 = __riscv_vfmerge_vfm_f64m1(target_rsqrt_f64m1, 0.0, infinite_b64, vector_length);
    vfloat64m1_t rsqrt_f64m1 = __riscv_vfmul_vf_f64m1(target_rsqrt_f64m1, nk_f64_rsqrt_newton_rvv_(query_norm_sq_f64),
                                                      vector_length);
    vfloat64m1_t angular_f64m1 = __riscv_vfrsub_vf_f64m1(__riscv_vfmul_vv_f64m1(dots_f64m1, rsqrt_f64m1, vector_length),
                                                         1.0, vector_length);
    vbool64_t target_zero_b64 = __riscv_vmfeq_vf_f64m1_b64(target_norms_sq_f64m1, 0.0, vector_length);
    vbool64_t one_b64 = __riscv_vmor_mm_b64(target_zero_b64, __riscv_vmfeq_vf_f64m1_b64(dots_f64m1, 0.0, vector_length),
                                            vector_length);
    if (query_norm_sq_f64 == 0.0) one_b64 = __riscv_vmset_m_b64(vector_length);
    angular_f64m1 = __riscv_vfmerge_vfm_f64m1(angular_f64m1, 1.0, one_b64, vector_length);
    if (query_norm_sq_f64 == 0.0)
        angular_f64m1 = __riscv_vfmerge_vfm_f64m1(angular_f64m1, 0.0, target_zero_b64, vector_length);
    angular_f64m1 = __riscv_vmerge_vvm_f64m1(
        angular_f64m1, dots_f64m1, __riscv_vmfne_vv_f64m1_b64(dots_f64m1, dots_f64m1, vector_length), vector_length);
    return __riscv_vfmerge_vfm_f64m1(angular_f64m1, 0.0, __riscv_vmflt_vf_f64m1_b64(angular_f64m1, 0.0, vector_length),
                                     vector_length);
}

/** Angular distances of one i32 dot register group with u32 norms. Widening @c vwmul keeps ab
 *  and d² exact, so @c vfncvt rounds the gap ab − d² once, and FMA keeps the rest in f32. */
NUMKONG_INLINE vfloat32m1_t nk_angular_i32m1_from_dot_rvv_(vint32m1_t dots_i32m1, nk_u32_t query_norm_sq,
                                                           vuint32m1_t target_norms_sq_u32m1, size_t vector_length) {
    // A zero norm gives 1, and two zero norms give 0
    vbool32_t const target_zero_b32 = __riscv_vmseq_vx_u32m1_b32(target_norms_sq_u32m1, 0, vector_length);
    if (query_norm_sq == 0)
        return __riscv_vfmerge_vfm_f32m1(__riscv_vfmv_v_f_f32m1(1.0f, vector_length), 0.0f, target_zero_b32,
                                         vector_length);
    vuint64m2_t const product_u64m2 = __riscv_vwmulu_vx_u64m2(target_norms_sq_u32m1, query_norm_sq, vector_length);
    // Cauchy–Schwarz keeps the gap non-negative
    vuint64m2_t const gap_u64m2 = __riscv_vsub_vv_u64m2(
        product_u64m2,
        __riscv_vreinterpret_v_i64m2_u64m2(__riscv_vwmul_vv_i64m2(dots_i32m1, dots_i32m1, vector_length)),
        vector_length);
    vfloat32m1_t const product_f32m1 = __riscv_vfncvt_f_xu_w_f32m1(product_u64m2, vector_length);
    vfloat32m1_t const dots_f32m1 = __riscv_vfcvt_f_x_v_f32m1(dots_i32m1, vector_length);
    vfloat32m1_t const norm_f32m1 = __riscv_vfsqrt_v_f32m1(product_f32m1, vector_length);
    // A positive dot takes (ab − d²) / (ab + d × s), any other 1 + |d| / s
    vbool32_t const nonpositive_b32 = __riscv_vmsle_vx_i32m1_b32(dots_i32m1, 0, vector_length);
    vfloat32m1_t const numerator_f32m1 = __riscv_vmerge_vvm_f32m1(__riscv_vfncvt_f_xu_w_f32m1(gap_u64m2, vector_length),
                                                                  __riscv_vfneg_v_f32m1(dots_f32m1, vector_length),
                                                                  nonpositive_b32, vector_length);
    vfloat32m1_t const denominator_f32m1 = __riscv_vmerge_vvm_f32m1(
        __riscv_vfmacc_vv_f32m1(product_f32m1, dots_f32m1, norm_f32m1, vector_length), norm_f32m1, nonpositive_b32,
        vector_length);
    vfloat32m1_t angular_f32m1 = __riscv_vfdiv_vv_f32m1(numerator_f32m1, denominator_f32m1, vector_length);
    angular_f32m1 = __riscv_vfadd_vf_f32m1_mu(nonpositive_b32, angular_f32m1, angular_f32m1, 1.0f, vector_length);
    return __riscv_vfmerge_vfm_f32m1(angular_f32m1, 1.0f, target_zero_b32, vector_length);
}

/** Angular distances of one u32 dot register group with u32 norms. Widening @c vwmulu keeps ab
 *  and d² exact, so @c vfncvt rounds the gap ab − d² once, and FMA keeps the rest in f32. */
NUMKONG_INLINE vfloat32m1_t nk_angular_u32m1_from_dot_rvv_(vuint32m1_t dots_u32m1, nk_u32_t query_norm_sq,
                                                           vuint32m1_t target_norms_sq_u32m1, size_t vector_length) {
    // A zero norm or a zero dot gives 1, and two zero norms give 0
    vbool32_t const target_zero_b32 = __riscv_vmseq_vx_u32m1_b32(target_norms_sq_u32m1, 0, vector_length);
    if (query_norm_sq == 0)
        return __riscv_vfmerge_vfm_f32m1(__riscv_vfmv_v_f_f32m1(1.0f, vector_length), 0.0f, target_zero_b32,
                                         vector_length);
    vuint64m2_t const product_u64m2 = __riscv_vwmulu_vx_u64m2(target_norms_sq_u32m1, query_norm_sq, vector_length);
    // Cauchy–Schwarz keeps the gap non-negative
    vuint64m2_t const gap_u64m2 = __riscv_vsub_vv_u64m2(
        product_u64m2, __riscv_vwmulu_vv_u64m2(dots_u32m1, dots_u32m1, vector_length), vector_length);
    vfloat32m1_t const product_f32m1 = __riscv_vfncvt_f_xu_w_f32m1(product_u64m2, vector_length);
    vfloat32m1_t const norm_f32m1 = __riscv_vfsqrt_v_f32m1(product_f32m1, vector_length);
    // (ab − d²) / (ab + d × s) keeps the small angles that 1 − d / s cancels
    vfloat32m1_t const angular_f32m1 = __riscv_vfdiv_vv_f32m1(
        __riscv_vfncvt_f_xu_w_f32m1(gap_u64m2, vector_length),
        __riscv_vfmacc_vv_f32m1(product_f32m1, __riscv_vfcvt_f_xu_v_f32m1(dots_u32m1, vector_length), norm_f32m1,
                                vector_length),
        vector_length);
    vbool32_t const one_b32 = __riscv_vmor_mm_b32(
        target_zero_b32, __riscv_vmseq_vx_u32m1_b32(dots_u32m1, 0, vector_length), vector_length);
    return __riscv_vfmerge_vfm_f32m1(angular_f32m1, 1.0f, one_b32, vector_length);
}

/** Euclidean distances of one i32 dot register group with u32 norms. Widening adds make the 34-bit
 *  q + t − 2d exact in 64 bits, and @c vfncvt rounds it once, in fewer ops than 16-bit halves. */
NUMKONG_INLINE vfloat32m1_t nk_euclidean_i32m1_from_dot_rvv_(vint32m1_t dots_i32m1, nk_u32_t query_norm_sq,
                                                             vuint32m1_t target_norms_sq_u32m1, size_t vector_length) {
    vint64m2_t const distance_sq_i64m2 = __riscv_vsub_vv_i64m2(
        __riscv_vreinterpret_v_u64m2_i64m2(
            __riscv_vwaddu_vx_u64m2(target_norms_sq_u32m1, query_norm_sq, vector_length)),
        __riscv_vwadd_vv_i64m2(dots_i32m1, dots_i32m1, vector_length), vector_length);
    return __riscv_vfsqrt_v_f32m1(__riscv_vfncvt_f_x_w_f32m1(distance_sq_i64m2, vector_length), vector_length);
}

/** Euclidean distances of one u32 dot register group with u32 norms. Widening adds make the 33-bit
 *  q + t − 2d exact in 64 bits, and @c vfncvt rounds it once, in fewer ops than 16-bit halves. */
NUMKONG_INLINE vfloat32m1_t nk_euclidean_u32m1_from_dot_rvv_(vuint32m1_t dots_u32m1, nk_u32_t query_norm_sq,
                                                             vuint32m1_t target_norms_sq_u32m1, size_t vector_length) {
    vuint64m2_t const distance_sq_u64m2 = __riscv_vsub_vv_u64m2(
        __riscv_vwaddu_vx_u64m2(target_norms_sq_u32m1, query_norm_sq, vector_length),
        __riscv_vwaddu_vv_u64m2(dots_u32m1, dots_u32m1, vector_length), vector_length);
    return __riscv_vfsqrt_v_f32m1(__riscv_vfncvt_f_xu_w_f32m1(distance_sq_u64m2, vector_length), vector_length);
}

#pragma region F32 Floats

NUMKONG_INLINE void nk_angulars_packed_f32_rvv_finalize_(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f64_t const *target_norms = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f32_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *result_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f32_rvv_(a_row, depth, sizeof(nk_f32_t));
        nk_size_t count_columns = columns;
        nk_f64_t *result_ptr = result_row;
        nk_f64_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e64m1(count_columns);
            vfloat64m1_t dots_f64m1 = __riscv_vle64_v_f64m1(result_ptr, vector_length);
            vfloat64m1_t target_norms_sq_f64m1 = __riscv_vle64_v_f64m1(norms_ptr, vector_length);
            vfloat64m1_t angular_f64m1 = nk_angular_f64m1_from_dot_rvv_(dots_f64m1, query_norm_sq_f64,
                                                                        target_norms_sq_f64m1, vector_length);
            __riscv_vse64_v_f64m1(result_ptr, angular_f64m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_euclideans_packed_f32_rvv_finalize_(nk_f32_t const *a, void const *b_packed, nk_f64_t *c,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f64_t const *target_norms = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f32_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *result_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f32_rvv_(a_row, depth, sizeof(nk_f32_t));
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
            dist_sq_f64m1 = __riscv_vfmerge_vfm_f64m1(
                dist_sq_f64m1, 0.0, __riscv_vmflt_vf_f64m1_b64(dist_sq_f64m1, 0.0, vector_length), vector_length);
            __riscv_vse64_v_f64m1(result_ptr, __riscv_vfsqrt_v_f64m1(dist_sq_f64m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_angulars_symmetric_f32_rvv_finalize_(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride_elements,
                                                            nk_f64_t *result, nk_size_t result_stride_elements,
                                                            nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f32_rvv_(vectors + row_index * stride_elements, depth,
                                                              sizeof(nk_f32_t));
    }
    nk_f64_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f32_rvv_(vectors + col * stride_elements, depth,
                                                                           sizeof(nk_f32_t));
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
                vfloat64m1_t angular_f64m1 = nk_angular_f64m1_from_dot_rvv_(dots_f64m1, query_norm_sq_f64,
                                                                            target_norms_sq_f64m1, vector_length);
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

NUMKONG_INLINE void nk_euclideans_symmetric_f32_rvv_finalize_(nk_f32_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride_elements,
                                                              nk_f64_t *result, nk_size_t result_stride_elements,
                                                              nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f32_rvv_(vectors + row_index * stride_elements, depth,
                                                              sizeof(nk_f32_t));
    }
    nk_f64_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f32_rvv_(vectors + col * stride_elements, depth,
                                                                           sizeof(nk_f32_t));
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
                dist_sq_f64m1 = __riscv_vfmerge_vfm_f64m1(
                    dist_sq_f64m1, 0.0, __riscv_vmflt_vf_f64m1_b64(dist_sq_f64m1, 0.0, vector_length), vector_length);
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

#pragma endregion F32 Floats

#pragma region F64 Floats

NUMKONG_INLINE void nk_angulars_packed_f64_rvv_finalize_(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f64_t const *target_norms = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f64_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f64_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *result_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f64_rvv_(a_row, depth, sizeof(nk_f64_t));
        nk_size_t count_columns = columns;
        nk_f64_t *result_ptr = result_row;
        nk_f64_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e64m1(count_columns);
            vfloat64m1_t dots_f64m1 = __riscv_vle64_v_f64m1(result_ptr, vector_length);
            vfloat64m1_t target_norms_sq_f64m1 = __riscv_vle64_v_f64m1(norms_ptr, vector_length);
            vfloat64m1_t angular_f64m1 = nk_angular_f64m1_from_dot_rvv_(dots_f64m1, query_norm_sq_f64,
                                                                        target_norms_sq_f64m1, vector_length);
            __riscv_vse64_v_f64m1(result_ptr, angular_f64m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_euclideans_packed_f64_rvv_finalize_(nk_f64_t const *a, void const *b_packed, nk_f64_t *c,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f64_t const *target_norms = (nk_f64_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f64_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f64_t const *a_row = a + row_index * a_stride_elements;
        nk_f64_t *result_row = c + row_index * c_stride_elements;
        nk_f64_t query_norm_sq_f64 = nk_dots_reduce_sumsq_f64_rvv_(a_row, depth, sizeof(nk_f64_t));
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
            dist_sq_f64m1 = __riscv_vfmerge_vfm_f64m1(
                dist_sq_f64m1, 0.0, __riscv_vmflt_vf_f64m1_b64(dist_sq_f64m1, 0.0, vector_length), vector_length);
            __riscv_vse64_v_f64m1(result_ptr, __riscv_vfsqrt_v_f64m1(dist_sq_f64m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_angulars_symmetric_f64_rvv_finalize_(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride_elements,
                                                            nk_f64_t *result, nk_size_t result_stride_elements,
                                                            nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f64_rvv_(vectors + row_index * stride_elements, depth,
                                                              sizeof(nk_f64_t));
    }
    nk_f64_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f64_rvv_(vectors + col * stride_elements, depth,
                                                                           sizeof(nk_f64_t));
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
                vfloat64m1_t angular_f64m1 = nk_angular_f64m1_from_dot_rvv_(dots_f64m1, query_norm_sq_f64,
                                                                            target_norms_sq_f64m1, vector_length);
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

NUMKONG_INLINE void nk_euclideans_symmetric_f64_rvv_finalize_(nk_f64_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride_elements,
                                                              nk_f64_t *result, nk_size_t result_stride_elements,
                                                              nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f64_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f64_rvv_(vectors + row_index * stride_elements, depth,
                                                              sizeof(nk_f64_t));
    }
    nk_f64_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f64_rvv_(vectors + col * stride_elements, depth,
                                                                           sizeof(nk_f64_t));
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
                dist_sq_f64m1 = __riscv_vfmerge_vfm_f64m1(
                    dist_sq_f64m1, 0.0, __riscv_vmflt_vf_f64m1_b64(dist_sq_f64m1, 0.0, vector_length), vector_length);
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

#pragma endregion F64 Floats

#pragma region F16 Floats

NUMKONG_INLINE void nk_angulars_packed_f16_rvv_finalize_(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                         nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                         nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_f16_rvv_(a_row, depth, sizeof(nk_f16_t));
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                        target_norms_sq_f32m1, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_euclideans_packed_f16_rvv_finalize_(nk_f16_t const *a, void const *b_packed, nk_f32_t *c,
                                                           nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                           nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_f16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_f16_rvv_(a_row, depth, sizeof(nk_f16_t));
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
            dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_angulars_symmetric_f16_rvv_finalize_(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                            nk_size_t depth, nk_size_t stride_elements,
                                                            nk_f32_t *result, nk_size_t result_stride_elements,
                                                            nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_rvv_(vectors + row_index * stride_elements, depth,
                                                              sizeof(nk_f16_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_rvv_(vectors + col * stride_elements, depth,
                                                                           sizeof(nk_f16_t));
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
                vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                            target_norms_sq_f32m1, vector_length);
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

NUMKONG_INLINE void nk_euclideans_symmetric_f16_rvv_finalize_(nk_f16_t const *vectors, nk_size_t vectors_count,
                                                              nk_size_t depth, nk_size_t stride_elements,
                                                              nk_f32_t *result, nk_size_t result_stride_elements,
                                                              nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_f16_rvv_(vectors + row_index * stride_elements, depth,
                                                              sizeof(nk_f16_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_f16_rvv_(vectors + col * stride_elements, depth,
                                                                           sizeof(nk_f16_t));
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
                dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                    dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
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

#pragma endregion F16 Floats

#pragma region BF16 Floats

NUMKONG_INLINE void nk_angulars_packed_bf16_rvv_finalize_(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_bf16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_bf16_rvv_(a_row, depth, sizeof(nk_bf16_t));
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                        target_norms_sq_f32m1, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_euclideans_packed_bf16_rvv_finalize_(nk_bf16_t const *a, void const *b_packed, nk_f32_t *c,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_bf16_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_bf16_rvv_(a_row, depth, sizeof(nk_bf16_t));
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
            dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_angulars_symmetric_bf16_rvv_finalize_(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride_elements,
                                                             nk_f32_t *result, nk_size_t result_stride_elements,
                                                             nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_rvv_(vectors + row_index * stride_elements, depth,
                                                               sizeof(nk_bf16_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_rvv_(vectors + col * stride_elements, depth,
                                                                            sizeof(nk_bf16_t));
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
                vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                            target_norms_sq_f32m1, vector_length);
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

NUMKONG_INLINE void nk_euclideans_symmetric_bf16_rvv_finalize_(nk_bf16_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride_elements,
                                                               nk_f32_t *result, nk_size_t result_stride_elements,
                                                               nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_bf16_rvv_(vectors + row_index * stride_elements, depth,
                                                               sizeof(nk_bf16_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_bf16_rvv_(vectors + col * stride_elements, depth,
                                                                            sizeof(nk_bf16_t));
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
                dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                    dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
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

#pragma endregion BF16 Floats

#pragma region E2M3 Floats

NUMKONG_INLINE void nk_angulars_packed_e2m3_rvv_finalize_(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_e2m3_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m3_rvv_(a_row, depth, sizeof(nk_e2m3_t));
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                        target_norms_sq_f32m1, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_euclideans_packed_e2m3_rvv_finalize_(nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_e2m3_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m3_rvv_(a_row, depth, sizeof(nk_e2m3_t));
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
            dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_angulars_symmetric_e2m3_rvv_finalize_(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride_elements,
                                                             nk_f32_t *result, nk_size_t result_stride_elements,
                                                             nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_rvv_(vectors + row_index * stride_elements, depth,
                                                               sizeof(nk_e2m3_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_rvv_(vectors + col * stride_elements, depth,
                                                                            sizeof(nk_e2m3_t));
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
                vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                            target_norms_sq_f32m1, vector_length);
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

NUMKONG_INLINE void nk_euclideans_symmetric_e2m3_rvv_finalize_(nk_e2m3_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride_elements,
                                                               nk_f32_t *result, nk_size_t result_stride_elements,
                                                               nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m3_rvv_(vectors + row_index * stride_elements, depth,
                                                               sizeof(nk_e2m3_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m3_rvv_(vectors + col * stride_elements, depth,
                                                                            sizeof(nk_e2m3_t));
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
                dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                    dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
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

#pragma endregion E2M3 Floats

#pragma region E2M1 Floats

NUMKONG_INLINE void nk_angulars_packed_e2m1_rvv_finalize_(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_e2m1x2_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m1x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m1_rvv_(a_row, depth, sizeof(nk_e2m1x2_t));
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                        target_norms_sq_f32m1, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_euclideans_packed_e2m1_rvv_finalize_(nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_e2m1x2_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e2m1x2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e2m1_rvv_(a_row, depth, sizeof(nk_e2m1x2_t));
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
            dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_angulars_symmetric_e2m1_rvv_finalize_(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride_elements,
                                                             nk_f32_t *result, nk_size_t result_stride_elements,
                                                             nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m1_rvv_(vectors + row_index * stride_elements, depth,
                                                               sizeof(nk_e2m1x2_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m1_rvv_(vectors + col * stride_elements, depth,
                                                                            sizeof(nk_e2m1x2_t));
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
                vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                            target_norms_sq_f32m1, vector_length);
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

NUMKONG_INLINE void nk_euclideans_symmetric_e2m1_rvv_finalize_(nk_e2m1x2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride_elements,
                                                               nk_f32_t *result, nk_size_t result_stride_elements,
                                                               nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e2m1_rvv_(vectors + row_index * stride_elements, depth,
                                                               sizeof(nk_e2m1x2_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e2m1_rvv_(vectors + col * stride_elements, depth,
                                                                            sizeof(nk_e2m1x2_t));
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
                dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                    dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
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

#pragma endregion E2M1 Floats

#pragma region E3M2 Floats

NUMKONG_INLINE void nk_angulars_packed_e3m2_rvv_finalize_(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_i16_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e3m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e3m2_rvv_(a_row, depth, sizeof(nk_e3m2_t));
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                        target_norms_sq_f32m1, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_euclideans_packed_e3m2_rvv_finalize_(nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_i16_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e3m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e3m2_rvv_(a_row, depth, sizeof(nk_e3m2_t));
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
            dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_angulars_symmetric_e3m2_rvv_finalize_(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride_elements,
                                                             nk_f32_t *result, nk_size_t result_stride_elements,
                                                             nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_rvv_(vectors + row_index * stride_elements, depth,
                                                               sizeof(nk_e3m2_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_rvv_(vectors + col * stride_elements, depth,
                                                                            sizeof(nk_e3m2_t));
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
                vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                            target_norms_sq_f32m1, vector_length);
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

NUMKONG_INLINE void nk_euclideans_symmetric_e3m2_rvv_finalize_(nk_e3m2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride_elements,
                                                               nk_f32_t *result, nk_size_t result_stride_elements,
                                                               nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e3m2_rvv_(vectors + row_index * stride_elements, depth,
                                                               sizeof(nk_e3m2_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e3m2_rvv_(vectors + col * stride_elements, depth,
                                                                            sizeof(nk_e3m2_t));
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
                dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                    dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
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

#pragma endregion E3M2 Floats

#pragma region E4M3 Floats

NUMKONG_INLINE void nk_angulars_packed_e4m3_rvv_finalize_(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e4m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e4m3_rvv_(a_row, depth, sizeof(nk_e4m3_t));
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                        target_norms_sq_f32m1, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_euclideans_packed_e4m3_rvv_finalize_(nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e4m3_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e4m3_rvv_(a_row, depth, sizeof(nk_e4m3_t));
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
            dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_angulars_symmetric_e4m3_rvv_finalize_(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride_elements,
                                                             nk_f32_t *result, nk_size_t result_stride_elements,
                                                             nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_rvv_(vectors + row_index * stride_elements, depth,
                                                               sizeof(nk_e4m3_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_rvv_(vectors + col * stride_elements, depth,
                                                                            sizeof(nk_e4m3_t));
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
                vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                            target_norms_sq_f32m1, vector_length);
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

NUMKONG_INLINE void nk_euclideans_symmetric_e4m3_rvv_finalize_(nk_e4m3_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride_elements,
                                                               nk_f32_t *result, nk_size_t result_stride_elements,
                                                               nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e4m3_rvv_(vectors + row_index * stride_elements, depth,
                                                               sizeof(nk_e4m3_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e4m3_rvv_(vectors + col * stride_elements, depth,
                                                                            sizeof(nk_e4m3_t));
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
                dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                    dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
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

#pragma endregion E4M3 Floats

#pragma region E5M2 Floats

NUMKONG_INLINE void nk_angulars_packed_e5m2_rvv_finalize_(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e5m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e5m2_rvv_(a_row, depth, sizeof(nk_e5m2_t));
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_f32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vfloat32m1_t dots_f32m1 = __riscv_vle32_v_f32m1(result_ptr, vector_length);
            vfloat32m1_t target_norms_sq_f32m1 = __riscv_vle32_v_f32m1(norms_ptr, vector_length);
            vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                        target_norms_sq_f32m1, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_euclideans_packed_e5m2_rvv_finalize_(nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c,
                                                            nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                            nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_f32_t const *target_norms = (nk_f32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_f32_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_e5m2_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_f32_t query_norm_sq_f32 = nk_dots_reduce_sumsq_e5m2_rvv_(a_row, depth, sizeof(nk_e5m2_t));
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
            dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
            __riscv_vse32_v_f32m1(result_ptr, __riscv_vfsqrt_v_f32m1(dist_sq_f32m1, vector_length), vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_angulars_symmetric_e5m2_rvv_finalize_(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride_elements,
                                                             nk_f32_t *result, nk_size_t result_stride_elements,
                                                             nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_rvv_(vectors + row_index * stride_elements, depth,
                                                               sizeof(nk_e5m2_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_rvv_(vectors + col * stride_elements, depth,
                                                                            sizeof(nk_e5m2_t));
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
                vfloat32m1_t angular_f32m1 = nk_angular_f32m1_from_dot_rvv_(dots_f32m1, query_norm_sq_f32,
                                                                            target_norms_sq_f32m1, vector_length);
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

NUMKONG_INLINE void nk_euclideans_symmetric_e5m2_rvv_finalize_(nk_e5m2_t const *vectors, nk_size_t vectors_count,
                                                               nk_size_t depth, nk_size_t stride_elements,
                                                               nk_f32_t *result, nk_size_t result_stride_elements,
                                                               nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_f32_t *result_row = result + row_index * result_stride_elements;
        result_row[row_index] = nk_dots_reduce_sumsq_e5m2_rvv_(vectors + row_index * stride_elements, depth,
                                                               sizeof(nk_e5m2_t));
    }
    nk_f32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_e5m2_rvv_(vectors + col * stride_elements, depth,
                                                                            sizeof(nk_e5m2_t));
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
                dist_sq_f32m1 = __riscv_vfmerge_vfm_f32m1(
                    dist_sq_f32m1, 0.0f, __riscv_vmflt_vf_f32m1_b32(dist_sq_f32m1, 0.0f, vector_length), vector_length);
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

#pragma endregion E5M2 Floats

#pragma region I8 Integers

NUMKONG_INLINE void nk_angulars_packed_i8_rvv_finalize_(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_u32_t const *target_norms = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_i8_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq = nk_dots_reduce_sumsq_i8_rvv_(a_row, depth, sizeof(nk_i8_t));
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_u32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vint32m1_t dots_i32m1 = __riscv_vle32_v_i32m1((nk_i32_t const *)result_ptr, vector_length);
            vuint32m1_t target_norms_sq_u32m1 = __riscv_vle32_v_u32m1(norms_ptr, vector_length);
            vfloat32m1_t angular_f32m1 = nk_angular_i32m1_from_dot_rvv_(dots_i32m1, query_norm_sq,
                                                                        target_norms_sq_u32m1, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_euclideans_packed_i8_rvv_finalize_(nk_i8_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_u32_t const *target_norms = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_i8_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_i8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq = nk_dots_reduce_sumsq_i8_rvv_(a_row, depth, sizeof(nk_i8_t));
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_u32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vint32m1_t dots_i32m1 = __riscv_vle32_v_i32m1((nk_i32_t const *)result_ptr, vector_length);
            vuint32m1_t target_norms_sq_u32m1 = __riscv_vle32_v_u32m1(norms_ptr, vector_length);
            vfloat32m1_t euclidean_f32m1 = nk_euclidean_i32m1_from_dot_rvv_(dots_i32m1, query_norm_sq,
                                                                            target_norms_sq_u32m1, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, euclidean_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_angulars_symmetric_i8_rvv_finalize_(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
                                                           nk_size_t result_stride_elements, nk_size_t row_start,
                                                           nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t norm = nk_dots_reduce_sumsq_i8_rvv_(vectors + row_index * stride_elements, depth, sizeof(nk_i8_t));
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = norm;
    }
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_rvv_(vectors + col * stride_elements, depth,
                                                                          sizeof(nk_i8_t));
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_norm_sq = ((nk_u32_t *)result_row)[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_u32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vint32m1_t dots_i32m1 = __riscv_vle32_v_i32m1((nk_i32_t const *)result_ptr, vector_length);
                vuint32m1_t target_norms_sq_u32m1 = __riscv_vle32_v_u32m1(norms_ptr, vector_length);
                vfloat32m1_t angular_f32m1 = nk_angular_i32m1_from_dot_rvv_(dots_i32m1, query_norm_sq,
                                                                            target_norms_sq_u32m1, vector_length);
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

NUMKONG_INLINE void nk_euclideans_symmetric_i8_rvv_finalize_(nk_i8_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride_elements,
                                                             nk_f32_t *result, nk_size_t result_stride_elements,
                                                             nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t norm = nk_dots_reduce_sumsq_i8_rvv_(vectors + row_index * stride_elements, depth, sizeof(nk_i8_t));
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = norm;
    }
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_i8_rvv_(vectors + col * stride_elements, depth,
                                                                          sizeof(nk_i8_t));
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_norm_sq = ((nk_u32_t *)result_row)[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_u32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vint32m1_t dots_i32m1 = __riscv_vle32_v_i32m1((nk_i32_t const *)result_ptr, vector_length);
                vuint32m1_t target_norms_sq_u32m1 = __riscv_vle32_v_u32m1(norms_ptr, vector_length);
                vfloat32m1_t euclidean_f32m1 = nk_euclidean_i32m1_from_dot_rvv_(dots_i32m1, query_norm_sq,
                                                                                target_norms_sq_u32m1, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, euclidean_f32m1, vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

#pragma endregion I8 Integers

#pragma region U8 Integers

NUMKONG_INLINE void nk_angulars_packed_u8_rvv_finalize_(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                        nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                        nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_u32_t const *target_norms = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_u8_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq = nk_dots_reduce_sumsq_u8_rvv_(a_row, depth, sizeof(nk_u8_t));
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_u32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vuint32m1_t dots_u32m1 = __riscv_vle32_v_u32m1((nk_u32_t const *)result_ptr, vector_length);
            vuint32m1_t target_norms_sq_u32m1 = __riscv_vle32_v_u32m1(norms_ptr, vector_length);
            vfloat32m1_t angular_f32m1 = nk_angular_u32m1_from_dot_rvv_(dots_u32m1, query_norm_sq,
                                                                        target_norms_sq_u32m1, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, angular_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_euclideans_packed_u8_rvv_finalize_(nk_u8_t const *a, void const *b_packed, nk_f32_t *c,
                                                          nk_size_t rows, nk_size_t columns, nk_size_t depth,
                                                          nk_size_t a_stride_elements, nk_size_t c_stride_elements) {
    nk_cross_packed_buffer_header_t const *header = (nk_cross_packed_buffer_header_t const *)b_packed;
    nk_u32_t const *target_norms = (nk_u32_t const *)((char const *)b_packed + sizeof(nk_cross_packed_buffer_header_t) +
                                                      header->column_count * header->depth_padded_values *
                                                          sizeof(nk_u8_t));
    for (nk_size_t row_index = 0; row_index < rows; row_index++) {
        nk_u8_t const *a_row = a + row_index * a_stride_elements;
        nk_f32_t *result_row = c + row_index * c_stride_elements;
        nk_u32_t query_norm_sq = nk_dots_reduce_sumsq_u8_rvv_(a_row, depth, sizeof(nk_u8_t));
        nk_size_t count_columns = columns;
        nk_f32_t *result_ptr = result_row;
        nk_u32_t const *norms_ptr = target_norms;
        while (count_columns > 0) {
            size_t vector_length = __riscv_vsetvl_e32m1(count_columns);
            vuint32m1_t dots_u32m1 = __riscv_vle32_v_u32m1((nk_u32_t const *)result_ptr, vector_length);
            vuint32m1_t target_norms_sq_u32m1 = __riscv_vle32_v_u32m1(norms_ptr, vector_length);
            vfloat32m1_t euclidean_f32m1 = nk_euclidean_u32m1_from_dot_rvv_(dots_u32m1, query_norm_sq,
                                                                            target_norms_sq_u32m1, vector_length);
            __riscv_vse32_v_f32m1(result_ptr, euclidean_f32m1, vector_length);
            result_ptr += vector_length;
            norms_ptr += vector_length;
            count_columns -= vector_length;
        }
    }
}

NUMKONG_INLINE void nk_angulars_symmetric_u8_rvv_finalize_(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                           nk_size_t depth, nk_size_t stride_elements, nk_f32_t *result,
                                                           nk_size_t result_stride_elements, nk_size_t row_start,
                                                           nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t norm = nk_dots_reduce_sumsq_u8_rvv_(vectors + row_index * stride_elements, depth, sizeof(nk_u8_t));
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = norm;
    }
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_rvv_(vectors + col * stride_elements, depth,
                                                                          sizeof(nk_u8_t));
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_norm_sq = ((nk_u32_t *)result_row)[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_u32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vuint32m1_t dots_u32m1 = __riscv_vle32_v_u32m1((nk_u32_t const *)result_ptr, vector_length);
                vuint32m1_t target_norms_sq_u32m1 = __riscv_vle32_v_u32m1(norms_ptr, vector_length);
                vfloat32m1_t angular_f32m1 = nk_angular_u32m1_from_dot_rvv_(dots_u32m1, query_norm_sq,
                                                                            target_norms_sq_u32m1, vector_length);
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

NUMKONG_INLINE void nk_euclideans_symmetric_u8_rvv_finalize_(nk_u8_t const *vectors, nk_size_t vectors_count,
                                                             nk_size_t depth, nk_size_t stride_elements,
                                                             nk_f32_t *result, nk_size_t result_stride_elements,
                                                             nk_size_t row_start, nk_size_t row_count) {
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
        nk_u32_t norm = nk_dots_reduce_sumsq_u8_rvv_(vectors + row_index * stride_elements, depth, sizeof(nk_u8_t));
        ((nk_u32_t *)(result + row_index * result_stride_elements))[row_index] = norm;
    }
    nk_u32_t norms_cache[256];
    for (nk_size_t chunk_start = 0; chunk_start < vectors_count; chunk_start += 256) {
        nk_size_t chunk_end = chunk_start + 256 < vectors_count ? chunk_start + 256 : vectors_count;
        for (nk_size_t col = chunk_start; col < chunk_end; ++col)
            norms_cache[col - chunk_start] = nk_dots_reduce_sumsq_u8_rvv_(vectors + col * stride_elements, depth,
                                                                          sizeof(nk_u8_t));
        for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index) {
            nk_size_t col_start = row_index + 1 > chunk_start ? row_index + 1 : chunk_start;
            if (col_start >= chunk_end) continue;
            nk_f32_t *result_row = result + row_index * result_stride_elements;
            nk_u32_t query_norm_sq = ((nk_u32_t *)result_row)[row_index];
            nk_size_t count_remaining = chunk_end - col_start;
            nk_f32_t *result_ptr = result_row + col_start;
            nk_u32_t const *norms_ptr = norms_cache + (col_start - chunk_start);
            while (count_remaining > 0) {
                size_t vector_length = __riscv_vsetvl_e32m1(count_remaining);
                vuint32m1_t dots_u32m1 = __riscv_vle32_v_u32m1((nk_u32_t const *)result_ptr, vector_length);
                vuint32m1_t target_norms_sq_u32m1 = __riscv_vle32_v_u32m1(norms_ptr, vector_length);
                vfloat32m1_t euclidean_f32m1 = nk_euclidean_u32m1_from_dot_rvv_(dots_u32m1, query_norm_sq,
                                                                                target_norms_sq_u32m1, vector_length);
                __riscv_vse32_v_f32m1(result_ptr, euclidean_f32m1, vector_length);
                result_ptr += vector_length;
                norms_ptr += vector_length;
                count_remaining -= vector_length;
            }
        }
    }
    for (nk_size_t row_index = row_start; row_index < row_start + row_count; ++row_index)
        result[row_index * result_stride_elements + row_index] = 0;
}

#pragma endregion U8 Integers

#if NUMKONG_TARGET_RVV
#pragma region F32 Floats

NUMKONG_API nk_status_t nk_angulars_packed_f32_rvv(       //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_f32_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_angulars_packed_f32_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_f32_rvv(     //
    nk_f32_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f32_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_f32_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_euclideans_packed_f32_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f32_rvv( //
    nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_dots_symmetric_f32_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
    nk_angulars_symmetric_f32_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                            result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f32_rvv( //
    nk_f32_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_f32_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_dots_symmetric_f32_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
    nk_euclideans_symmetric_f32_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                              result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#pragma endregion F32 Floats

#pragma region F64 Floats

NUMKONG_API nk_status_t nk_angulars_packed_f64_rvv(       //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_f64_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_angulars_packed_f64_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_f64_rvv(     //
    nk_f64_t const *a, void const *b_packed, nk_f64_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f64_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f64_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_f64_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_euclideans_packed_f64_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f64_rvv( //
    nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_dots_symmetric_f64_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
    nk_angulars_symmetric_f64_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                            result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f64_rvv( //
    nk_f64_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f64_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_f64_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f64_t);
    nk_dots_symmetric_f64_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
    nk_euclideans_symmetric_f64_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                              result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#pragma endregion F64 Floats

#pragma region F16 Floats

NUMKONG_API nk_status_t nk_angulars_packed_f16_rvv(       //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_f16_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_angulars_packed_f16_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_f16_rvv(     //
    nk_f16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,   //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_f16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_f16_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_euclideans_packed_f16_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_f16_rvv( //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
    nk_angulars_symmetric_f16_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                            result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_f16_rvv( //
    nk_f16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_f16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_f16_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                     row_count);
    nk_euclideans_symmetric_f16_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                              result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#pragma endregion F16 Floats

#pragma region BF16 Floats

NUMKONG_API nk_status_t nk_angulars_packed_bf16_rvv(       //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_bf16_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_angulars_packed_bf16_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_bf16_rvv(     //
    nk_bf16_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_bf16_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_bf16_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_euclideans_packed_bf16_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_bf16_rvv( //
    nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
    nk_angulars_symmetric_bf16_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                             result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_bf16_rvv( //
    nk_bf16_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_bf16_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_bf16_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
    nk_euclideans_symmetric_bf16_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                               result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#pragma endregion BF16 Floats

#pragma region E2M3 Floats

NUMKONG_API nk_status_t nk_angulars_packed_e2m3_rvv(       //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_e2m3_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_angulars_packed_e2m3_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e2m3_rvv(     //
    nk_e2m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_e2m3_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_euclideans_packed_e2m3_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m3_rvv( //
    nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
    nk_angulars_symmetric_e2m3_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                             result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m3_rvv( //
    nk_e2m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e2m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m3_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
    nk_euclideans_symmetric_e2m3_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                               result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#pragma endregion E2M3 Floats

#pragma region E2M1 Floats

NUMKONG_API nk_status_t nk_angulars_packed_e2m1_rvv(         //
    nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,      //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m1x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_e2m1_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_angulars_packed_e2m1_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e2m1_rvv(       //
    nk_e2m1x2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,      //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e2m1x2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_e2m1_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_euclideans_packed_e2m1_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e2m1_rvv( //
    nk_e2m1x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e2m1x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m1_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
    nk_angulars_symmetric_e2m1_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                             result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e2m1_rvv( //
    nk_e2m1x2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= nk_size_divide_round_up_(depth, 2) * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e2m1x2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e2m1_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
    nk_euclideans_symmetric_e2m1_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                               result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#pragma endregion E2M1 Floats

#pragma region E3M2 Floats

NUMKONG_API nk_status_t nk_angulars_packed_e3m2_rvv(       //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_e3m2_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_angulars_packed_e3m2_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e3m2_rvv(     //
    nk_e3m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e3m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_e3m2_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_euclideans_packed_e3m2_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e3m2_rvv( //
    nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
    nk_angulars_symmetric_e3m2_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                             result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e3m2_rvv( //
    nk_e3m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e3m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e3m2_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
    nk_euclideans_symmetric_e3m2_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                               result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#pragma endregion E3M2 Floats

#pragma region E4M3 Floats

NUMKONG_API nk_status_t nk_angulars_packed_e4m3_rvv(       //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_e4m3_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_angulars_packed_e4m3_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e4m3_rvv(     //
    nk_e4m3_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e4m3_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_e4m3_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_euclideans_packed_e4m3_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e4m3_rvv( //
    nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
    nk_angulars_symmetric_e4m3_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                             result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e4m3_rvv( //
    nk_e4m3_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e4m3_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e4m3_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
    nk_euclideans_symmetric_e4m3_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                               result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#pragma endregion E4M3 Floats

#pragma region E5M2 Floats

NUMKONG_API nk_status_t nk_angulars_packed_e5m2_rvv(       //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_e5m2_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_angulars_packed_e5m2_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_e5m2_rvv(     //
    nk_e5m2_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,    //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_e5m2_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_e5m2_rvv_aligned_(a, b_packed, c, rows, columns, depth, a_stride, c_stride);
    nk_euclideans_packed_e5m2_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_e5m2_rvv( //
    nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
    nk_angulars_symmetric_e5m2_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                             result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_e5m2_rvv( //
    nk_e5m2_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_e5m2_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_e5m2_rvv_upper_(vectors, vectors_count, depth, stride, result, result_stride, row_start,
                                      row_count);
    nk_euclideans_symmetric_e5m2_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                               result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#pragma endregion E5M2 Floats

#pragma region I8 Integers

NUMKONG_API nk_status_t nk_angulars_packed_i8_rvv(       //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_i8_rvv_aligned_(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride, c_stride);
    nk_angulars_packed_i8_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_i8_rvv(     //
    nk_i8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_i8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_i8_rvv_aligned_(a, b_packed, (nk_i32_t *)c, rows, columns, depth, a_stride, c_stride);
    nk_euclideans_packed_i8_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_i8_rvv( //
    nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i8_rvv_upper_(vectors, vectors_count, depth, stride, (nk_i32_t *)result, result_stride, row_start,
                                    row_count);
    nk_angulars_symmetric_i8_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                           result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_i8_rvv( //
    nk_i8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_i8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_i8_rvv_upper_(vectors, vectors_count, depth, stride, (nk_i32_t *)result, result_stride, row_start,
                                    row_count);
    nk_euclideans_symmetric_i8_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                             result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#pragma endregion I8 Integers

#pragma region U8 Integers

NUMKONG_API nk_status_t nk_angulars_packed_u8_rvv(       //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_u8_rvv_aligned_(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride, c_stride);
    nk_angulars_packed_u8_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_packed_u8_rvv(     //
    nk_u8_t const *a, void const *b_packed, nk_f32_t *c, //
    nk_size_t rows, nk_size_t columns, nk_size_t depth,  //
    nk_size_t a_stride, nk_size_t c_stride, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t const a_stride_elements = a_stride / sizeof(nk_u8_t);
    nk_size_t const c_stride_elements = c_stride / sizeof(nk_f32_t);
    if (((nk_cross_packed_buffer_header_t const *)b_packed)->capability != nk_cap_rvv_k) return nk_pack_mismatch_k;
    nk_dots_packed_u8_rvv_aligned_(a, b_packed, (nk_u32_t *)c, rows, columns, depth, a_stride, c_stride);
    nk_euclideans_packed_u8_rvv_finalize_(a, b_packed, c, rows, columns, depth, a_stride_elements, c_stride_elements);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_angulars_symmetric_u8_rvv( //
    nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u8_rvv_upper_(vectors, vectors_count, depth, stride, (nk_u32_t *)result, result_stride, row_start,
                                    row_count);
    nk_angulars_symmetric_u8_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                           result_stride_elements, row_start, row_count);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_euclideans_symmetric_u8_rvv( //
    nk_u8_t const *vectors, nk_size_t vectors_count, nk_size_t depth,
    nk_size_t stride, //
    nk_f32_t *result, nk_size_t result_stride, nk_size_t row_start, nk_size_t row_count, void *stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_assert_(stride % sizeof(*vectors) == 0 && stride >= depth * sizeof(*vectors));
    row_count = row_start < vectors_count ? nk_min_of_two(row_count, vectors_count - row_start) : 0;
    nk_size_t const stride_elements = stride / sizeof(nk_u8_t);
    nk_size_t const result_stride_elements = result_stride / sizeof(nk_f32_t);
    nk_dots_symmetric_u8_rvv_upper_(vectors, vectors_count, depth, stride, (nk_u32_t *)result, result_stride, row_start,
                                    row_count);
    nk_euclideans_symmetric_u8_rvv_finalize_(vectors, vectors_count, depth, stride_elements, result,
                                             result_stride_elements, row_start, row_count);
    return nk_success_k;
}

#pragma endregion U8 Integers
#endif // NUMKONG_TARGET_RVV

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_RISCV64_RVV_
#endif // NUMKONG_ARCH_RISCV64_
#endif // NUMKONG_SPATIALS_RVV_H
