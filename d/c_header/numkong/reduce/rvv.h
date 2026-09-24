/**
 *  @brief SIMD-accelerated Reductions for RISC-V.
 *  @file include/numkong/reduce/rvv.h
 *  @author Ash Vardanian
 *  @date February 13, 2026
 *
 *  @sa include/numkong/reduce.h
 */
#ifndef NK_REDUCE_RVV_H
#define NK_REDUCE_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/cast/rvv.h"
#include "numkong/reduce/serial.h"

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/** @brief Saturating horizontal sum of u64m1 via tree fold: O(log vlmax) vector ops. */
NK_INTERNAL nk_u64_t nk_reduce_vsaddu_u64m1_rvv_(vuint64m1_t acc_u64m1, nk_size_t vlmax) {
    for (nk_size_t half = vlmax >> 1; half > 0; half >>= 1) {
        vuint64m1_t shifted_u64m1 = __riscv_vslidedown_vx_u64m1(acc_u64m1, half, vlmax);
        acc_u64m1 = __riscv_vsaddu_vv_u64m1(acc_u64m1, shifted_u64m1, vlmax);
    }
    return __riscv_vmv_x_s_u64m1_u64(acc_u64m1);
}

/** @brief Saturating horizontal sum of u64m2 via tree fold: O(log vlmax) vector ops. */
NK_INTERNAL nk_u64_t nk_reduce_vsaddu_u64m2_rvv_(vuint64m2_t acc_u64m2, nk_size_t vlmax) {
    for (nk_size_t half = vlmax >> 1; half > 0; half >>= 1) {
        vuint64m2_t shifted_u64m2 = __riscv_vslidedown_vx_u64m2(acc_u64m2, half, vlmax);
        acc_u64m2 = __riscv_vsaddu_vv_u64m2(acc_u64m2, shifted_u64m2, vlmax);
    }
    return __riscv_vmv_x_s_u64m2_u64(acc_u64m2);
}

/** @brief 128-bit horizontal sum of (upper:i64m1, lower:u64m1) via tree fold, then saturate to i64. */
NK_INTERNAL nk_i64_t nk_reduce_128bit_sum_i64m1_rvv_( //
    vuint64m1_t sum_lower_u64m1, vint64m1_t sum_upper_i64m1, nk_size_t vlmax) {
    for (nk_size_t half = vlmax >> 1; half > 0; half >>= 1) {
        vuint64m1_t shifted_lower_u64m1 = __riscv_vslidedown_vx_u64m1(sum_lower_u64m1, half, vlmax);
        vint64m1_t shifted_upper_i64m1 = __riscv_vslidedown_vx_i64m1(sum_upper_i64m1, half, vlmax);
        vuint64m1_t new_lower_u64m1 = __riscv_vadd_vv_u64m1(sum_lower_u64m1, shifted_lower_u64m1, vlmax);
        vbool64_t carry_b64 = __riscv_vmsltu_vv_u64m1_b64(new_lower_u64m1, sum_lower_u64m1, vlmax);
        vint64m1_t carry_i64m1 = __riscv_vmerge_vxm_i64m1(__riscv_vmv_v_x_i64m1(0, vlmax), 1, carry_b64, vlmax);
        sum_upper_i64m1 = __riscv_vadd_vv_i64m1(sum_upper_i64m1, shifted_upper_i64m1, vlmax);
        sum_upper_i64m1 = __riscv_vadd_vv_i64m1(sum_upper_i64m1, carry_i64m1, vlmax);
        sum_lower_u64m1 = new_lower_u64m1;
    }
    nk_u64_t total_lower = __riscv_vmv_x_s_u64m1_u64(sum_lower_u64m1);
    nk_i64_t total_upper = __riscv_vmv_x_s_i64m1_i64(sum_upper_i64m1);
    nk_i64_t total_lower_signed = (nk_i64_t)total_lower;
    if (total_upper == (total_lower_signed >> 63)) return total_lower_signed;
    else if (total_upper >= 0) return NK_I64_MAX;
    else return NK_I64_MIN;
}

/** @brief 128-bit horizontal sum of (upper:i64m2, lower:u64m2) via tree fold, then saturate to i64. */
NK_INTERNAL nk_i64_t nk_reduce_128bit_sum_i64m2_rvv_( //
    vuint64m2_t sum_lower_u64m2, vint64m2_t sum_upper_i64m2, nk_size_t vlmax) {
    for (nk_size_t half = vlmax >> 1; half > 0; half >>= 1) {
        vuint64m2_t shifted_lower_u64m2 = __riscv_vslidedown_vx_u64m2(sum_lower_u64m2, half, vlmax);
        vint64m2_t shifted_upper_i64m2 = __riscv_vslidedown_vx_i64m2(sum_upper_i64m2, half, vlmax);
        vuint64m2_t new_lower_u64m2 = __riscv_vadd_vv_u64m2(sum_lower_u64m2, shifted_lower_u64m2, vlmax);
        vbool32_t carry_b32 = __riscv_vmsltu_vv_u64m2_b32(new_lower_u64m2, sum_lower_u64m2, vlmax);
        vint64m2_t carry_i64m2 = __riscv_vmerge_vxm_i64m2(__riscv_vmv_v_x_i64m2(0, vlmax), 1, carry_b32, vlmax);
        sum_upper_i64m2 = __riscv_vadd_vv_i64m2(sum_upper_i64m2, shifted_upper_i64m2, vlmax);
        sum_upper_i64m2 = __riscv_vadd_vv_i64m2(sum_upper_i64m2, carry_i64m2, vlmax);
        sum_lower_u64m2 = new_lower_u64m2;
    }
    nk_u64_t total_lower = __riscv_vmv_x_s_u64m2_u64(sum_lower_u64m2);
    nk_i64_t total_upper = __riscv_vmv_x_s_i64m2_i64(sum_upper_i64m2);
    nk_i64_t total_lower_signed = (nk_i64_t)total_lower;
    if (total_upper == (total_lower_signed >> 63)) return total_lower_signed;
    else if (total_upper >= 0) return NK_I64_MAX;
    else return NK_I64_MIN;
}

NK_INTERNAL void nk_reduce_moments_f32_rvv_contiguous_( //
    nk_f32_t const *data, nk_size_t count,              //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sumsq_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vfloat32m1_t data_f32m1 = __riscv_vle32_v_f32m1(data, vector_length);
        sum_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_f64m2, sum_f64m2, data_f32m1, vector_length);
        sumsq_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(sumsq_f64m2, data_f32m1, data_f32m1, vector_length);
    }
    vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_f64m2, zero, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sumsq_f64m2, zero, vlmax));
}

NK_INTERNAL void nk_reduce_moments_f32_rvv_strided_(               //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sumsq_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vfloat32m1_t data_f32m1 = __riscv_vlse32_v_f32m1((nk_f32_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);
        sum_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_f64m2, sum_f64m2, data_f32m1, vector_length);
        sumsq_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(sumsq_f64m2, data_f32m1, data_f32m1, vector_length);
    }
    vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_f64m2, zero, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sumsq_f64m2, zero, vlmax));
}

NK_PUBLIC void nk_reduce_moments_f32_rvv(                          //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f32_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_f32_rvv_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_f32_rvv_strided_(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_f32_rvv_contiguous_( //
    nk_f32_t const *data, nk_size_t count,             //
    nk_f32_t *min_value, nk_size_t *min_index,         //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t min = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, vlmax);
    vfloat32m1_t max = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, vlmax);
    vuint64m2_t min_indices = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t max_indices = __riscv_vmv_v_x_u64m2(0, vlmax);
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t data_f32m1 = __riscv_vle32_v_f32m1(data + offset, vector_length);
        vuint64m2_t position_u64m2 = __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vector_length), (nk_u64_t)offset,
                                                           vector_length);
        vbool32_t less_b32 = __riscv_vmflt_vv_f32m1_b32(data_f32m1, min, vector_length);
        min = __riscv_vmerge_vvm_f32m1_tu(min, min, data_f32m1, less_b32, vector_length);
        min_indices = __riscv_vmerge_vvm_u64m2_tu(min_indices, min_indices, position_u64m2, less_b32, vector_length);
        vbool32_t greater_b32 = __riscv_vmflt_vv_f32m1_b32(max, data_f32m1, vector_length);
        max = __riscv_vmerge_vvm_f32m1_tu(max, max, data_f32m1, greater_b32, vector_length);
        max_indices = __riscv_vmerge_vvm_u64m2_tu(max_indices, max_indices, position_u64m2, greater_b32, vector_length);
    }
    vfloat32m1_t id_max = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    nk_f32_t mn = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(min, id_max, vlmax));
    vfloat32m1_t id_min = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    nk_f32_t mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(max, id_min, vlmax));
    if (mn == NK_F32_MAX && mx == NK_F32_MIN) {
        *min_value = NK_F32_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_F32_MIN, *max_index = NK_SIZE_MAX;
        return;
    }
    vbool32_t min_match_b32 = __riscv_vmfeq_vf_f32m1_b32(min, mn, vlmax);
    vuint64m2_t sentinel = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vlmax);
    vuint64m2_t min_cands = __riscv_vmerge_vvm_u64m2(sentinel, min_indices, min_match_b32, vlmax);
    vuint64m1_t id_umax = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value = mn,
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m2_u64m1(min_cands, id_umax, vlmax));
    vbool32_t max_match_b32 = __riscv_vmfeq_vf_f32m1_b32(max, mx, vlmax);
    vuint64m2_t max_cands = __riscv_vmerge_vvm_u64m2(sentinel, max_indices, max_match_b32, vlmax);
    *max_value = mx,
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m2_u64m1(max_cands, id_umax, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_f32_rvv_strided_(                //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index,                     //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t min = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, vlmax);
    vfloat32m1_t max = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, vlmax);
    vuint64m2_t min_indices = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t max_indices = __riscv_vmv_v_x_u64m2(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1_t data_f32m1 = __riscv_vlse32_v_f32m1((nk_f32_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);
        vuint64m2_t position_u64m2 = __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vector_length), (nk_u64_t)offset,
                                                           vector_length);
        vbool32_t less_b32 = __riscv_vmflt_vv_f32m1_b32(data_f32m1, min, vector_length);
        min = __riscv_vmerge_vvm_f32m1_tu(min, min, data_f32m1, less_b32, vector_length);
        min_indices = __riscv_vmerge_vvm_u64m2_tu(min_indices, min_indices, position_u64m2, less_b32, vector_length);
        vbool32_t greater_b32 = __riscv_vmflt_vv_f32m1_b32(max, data_f32m1, vector_length);
        max = __riscv_vmerge_vvm_f32m1_tu(max, max, data_f32m1, greater_b32, vector_length);
        max_indices = __riscv_vmerge_vvm_u64m2_tu(max_indices, max_indices, position_u64m2, greater_b32, vector_length);
    }
    vfloat32m1_t id_max = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    nk_f32_t mn = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(min, id_max, vlmax));
    vfloat32m1_t id_min = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    nk_f32_t mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(max, id_min, vlmax));
    if (mn == NK_F32_MAX && mx == NK_F32_MIN) {
        *min_value = NK_F32_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_F32_MIN, *max_index = NK_SIZE_MAX;
        return;
    }
    vbool32_t min_match_b32 = __riscv_vmfeq_vf_f32m1_b32(min, mn, vlmax);
    vuint64m2_t sentinel = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vlmax);
    vuint64m2_t min_cands = __riscv_vmerge_vvm_u64m2(sentinel, min_indices, min_match_b32, vlmax);
    vuint64m1_t id_umax = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value = mn,
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m2_u64m1(min_cands, id_umax, vlmax));
    vbool32_t max_match_b32 = __riscv_vmfeq_vf_f32m1_b32(max, mx, vlmax);
    vuint64m2_t max_cands = __riscv_vmerge_vvm_u64m2(sentinel, max_indices, max_match_b32, vlmax);
    *max_value = mx,
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m2_u64m1(max_cands, id_umax, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_f32_rvv(                           //
    nk_f32_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *min_value, nk_size_t *min_index,                     //
    nk_f32_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f32_t);
    int aligned = (stride_bytes % sizeof(nk_f32_t) == 0);
    if (count == 0)
        *min_value = NK_F32_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_F32_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f32_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1)
        nk_reduce_minmax_f32_rvv_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_f32_rvv_strided_(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL void nk_reduce_moments_f64_rvv_contiguous_( //
    nk_f64_t const *data, nk_size_t count,              //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t sumsq_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    for (nk_size_t vector_length; count > 0; count -= vector_length, data += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(count);
        vfloat64m4_t data_f64m4 = __riscv_vle64_v_f64m4(data, vector_length);
        sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(sum_f64m4, sum_f64m4, data_f64m4, vector_length);
        sumsq_f64m4 = __riscv_vfmacc_vv_f64m4_tu(sumsq_f64m4, data_f64m4, data_f64m4, vector_length);
    }
    vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sumsq_f64m4, zero, vlmax));
}

NK_INTERNAL void nk_reduce_moments_f64_rvv_strided_(               //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum_ptr, nk_f64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t sumsq_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m4(count);
        vfloat64m4_t data_f64m4 = __riscv_vlse64_v_f64m4((nk_f64_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);
        sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(sum_f64m4, sum_f64m4, data_f64m4, vector_length);
        sumsq_f64m4 = __riscv_vfmacc_vv_f64m4_tu(sumsq_f64m4, data_f64m4, data_f64m4, vector_length);
    }
    vfloat64m1_t zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sumsq_f64m4, zero, vlmax));
}

NK_PUBLIC void nk_reduce_moments_f64_rvv(                          //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *sum, nk_f64_t *sumsq) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0) *sum = 0, *sumsq = 0;
    else if (!aligned) nk_reduce_moments_f64_serial(data, count, stride_bytes, sum, sumsq);
    else if (stride_elements == 1) nk_reduce_moments_f64_rvv_contiguous_(data, count, sum, sumsq);
    else nk_reduce_moments_f64_rvv_strided_(data, count, stride_bytes, sum, sumsq);
}

NK_INTERNAL void nk_reduce_minmax_f64_rvv_contiguous_( //
    nk_f64_t const *data, nk_size_t count,             //
    nk_f64_t *min_value, nk_size_t *min_index,         //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t min = __riscv_vfmv_v_f_f64m1(NK_F64_MAX, vlmax);
    vfloat64m1_t max = __riscv_vfmv_v_f_f64m1(NK_F64_MIN, vlmax);
    vuint64m1_t min_indices = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t max_indices = __riscv_vmv_v_x_u64m1(0, vlmax);
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1_t data_f64m1 = __riscv_vle64_v_f64m1(data + offset, vector_length);
        vuint64m1_t position_u64m1 = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vector_length), (nk_u64_t)offset,
                                                           vector_length);
        vbool64_t less_b64 = __riscv_vmflt_vv_f64m1_b64(data_f64m1, min, vector_length);
        min = __riscv_vmerge_vvm_f64m1_tu(min, min, data_f64m1, less_b64, vector_length);
        min_indices = __riscv_vmerge_vvm_u64m1_tu(min_indices, min_indices, position_u64m1, less_b64, vector_length);
        vbool64_t greater_b64 = __riscv_vmflt_vv_f64m1_b64(max, data_f64m1, vector_length);
        max = __riscv_vmerge_vvm_f64m1_tu(max, max, data_f64m1, greater_b64, vector_length);
        max_indices = __riscv_vmerge_vvm_u64m1_tu(max_indices, max_indices, position_u64m1, greater_b64, vector_length);
    }
    vfloat64m1_t id_max = __riscv_vfmv_v_f_f64m1(NK_F64_MAX, 1);
    nk_f64_t mn = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmin_vs_f64m1_f64m1(min, id_max, vlmax));
    vfloat64m1_t id_min = __riscv_vfmv_v_f_f64m1(NK_F64_MIN, 1);
    nk_f64_t mx = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmax_vs_f64m1_f64m1(max, id_min, vlmax));
    if (mn == NK_F64_MAX && mx == NK_F64_MIN) {
        *min_value = NK_F64_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_F64_MIN, *max_index = NK_SIZE_MAX;
        return;
    }
    vbool64_t min_match_b64 = __riscv_vmfeq_vf_f64m1_b64(min, mn, vlmax);
    vuint64m1_t sentinel = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t min_cands = __riscv_vmerge_vvm_u64m1(sentinel, min_indices, min_match_b64, vlmax);
    vuint64m1_t id_umax = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value = mn,
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(min_cands, id_umax, vlmax));
    vbool64_t max_match_b64 = __riscv_vmfeq_vf_f64m1_b64(max, mx, vlmax);
    vuint64m1_t max_cands = __riscv_vmerge_vvm_u64m1(sentinel, max_indices, max_match_b64, vlmax);
    *max_value = mx,
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(max_cands, id_umax, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_f64_rvv_strided_(                //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index,                     //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t min = __riscv_vfmv_v_f_f64m1(NK_F64_MAX, vlmax);
    vfloat64m1_t max = __riscv_vfmv_v_f_f64m1(NK_F64_MIN, vlmax);
    vuint64m1_t min_indices = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t max_indices = __riscv_vmv_v_x_u64m1(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data;
    nk_size_t offset = 0;
    for (nk_size_t remaining = count, vector_length; remaining > 0;
         remaining -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1_t data_f64m1 = __riscv_vlse64_v_f64m1((nk_f64_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                         vector_length);
        vuint64m1_t position_u64m1 = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vector_length), (nk_u64_t)offset,
                                                           vector_length);
        vbool64_t less_b64 = __riscv_vmflt_vv_f64m1_b64(data_f64m1, min, vector_length);
        min = __riscv_vmerge_vvm_f64m1_tu(min, min, data_f64m1, less_b64, vector_length);
        min_indices = __riscv_vmerge_vvm_u64m1_tu(min_indices, min_indices, position_u64m1, less_b64, vector_length);
        vbool64_t greater_b64 = __riscv_vmflt_vv_f64m1_b64(max, data_f64m1, vector_length);
        max = __riscv_vmerge_vvm_f64m1_tu(max, max, data_f64m1, greater_b64, vector_length);
        max_indices = __riscv_vmerge_vvm_u64m1_tu(max_indices, max_indices, position_u64m1, greater_b64, vector_length);
    }
    vfloat64m1_t id_max = __riscv_vfmv_v_f_f64m1(NK_F64_MAX, 1);
    nk_f64_t mn = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmin_vs_f64m1_f64m1(min, id_max, vlmax));
    vfloat64m1_t id_min = __riscv_vfmv_v_f_f64m1(NK_F64_MIN, 1);
    nk_f64_t mx = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmax_vs_f64m1_f64m1(max, id_min, vlmax));
    if (mn == NK_F64_MAX && mx == NK_F64_MIN) {
        *min_value = NK_F64_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_F64_MIN, *max_index = NK_SIZE_MAX;
        return;
    }
    vbool64_t min_match_b64 = __riscv_vmfeq_vf_f64m1_b64(min, mn, vlmax);
    vuint64m1_t sentinel = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t min_cands = __riscv_vmerge_vvm_u64m1(sentinel, min_indices, min_match_b64, vlmax);
    vuint64m1_t id_umax = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value = mn,
    *min_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(min_cands, id_umax, vlmax));
    vbool64_t max_match_b64 = __riscv_vmfeq_vf_f64m1_b64(max, mx, vlmax);
    vuint64m1_t max_cands = __riscv_vmerge_vvm_u64m1(sentinel, max_indices, max_match_b64, vlmax);
    *max_value = mx,
    *max_index = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(max_cands, id_umax, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_f64_rvv(                           //
    nk_f64_t const *data, nk_size_t count, nk_size_t stride_bytes, //
    nk_f64_t *min_value, nk_size_t *min_index,                     //
    nk_f64_t *max_value, nk_size_t *max_index) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f64_t);
    int aligned = (stride_bytes % sizeof(nk_f64_t) == 0);
    if (count == 0)
        *min_value = NK_F64_MAX, *min_index = NK_SIZE_MAX, *max_value = NK_F64_MIN, *max_index = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f64_serial(data, count, stride_bytes, min_value, min_index, max_value, max_index);
    else if (stride_elements == 1)
        nk_reduce_minmax_f64_rvv_contiguous_(data, count, min_value, min_index, max_value, max_index);
    else nk_reduce_minmax_f64_rvv_strided_(data, count, stride_bytes, min_value, min_index, max_value, max_index);
}

NK_INTERNAL vuint8m1_t nk_fp8m1_to_comparable_u8m1_rvv_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    // Convert FP8 (e4m3/e5m2) to comparable unsigned form (sign bit 7)
    // Positive (sign=0): XOR 0x80 → [0x80, 0xFF]
    // Negative (sign=1): Bitwise NOT → [0x00, 0x7F]
    vbool8_t is_negative_b8 = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(raw_u8m1, 0x80, vector_length), 0,
                                                       vector_length);
    vuint8m1_t flip_positive_u8m1 = __riscv_vxor_vx_u8m1(raw_u8m1, 0x80, vector_length);
    vuint8m1_t flip_negative_u8m1 = __riscv_vnot_v_u8m1(raw_u8m1, vector_length);
    return __riscv_vmerge_vvm_u8m1(flip_positive_u8m1, flip_negative_u8m1, is_negative_b8, vector_length);
}

NK_INTERNAL vuint8m1_t nk_comparable_to_fp8m1_rvv_(vuint8m1_t comparable_u8m1, nk_size_t vector_length) {
    // Reverse: if >= 0x80 (was positive), XOR; else NOT
    vbool8_t was_positive_b8 = __riscv_vmsgeu_vx_u8m1_b8(comparable_u8m1, 0x80, vector_length);
    vuint8m1_t from_positive_u8m1 = __riscv_vxor_vx_u8m1(comparable_u8m1, 0x80, vector_length);
    vuint8m1_t from_negative_u8m1 = __riscv_vnot_v_u8m1(comparable_u8m1, vector_length);
    return __riscv_vmerge_vvm_u8m1(from_negative_u8m1, from_positive_u8m1, was_positive_b8, vector_length);
}

NK_INTERNAL vuint8m1_t nk_fp6m1_to_comparable_u8m1_rvv_(vuint8m1_t raw_u8m1, nk_size_t vector_length) {
    // Convert FP6 (e2m3/e3m2) to comparable unsigned form (sign bit 5)
    // Positive (sign=0): XOR 0x20 → [0x20, 0x3F]
    // Negative (sign=1): XOR 0x3F (NOT lower 6 bits) → [0x00, 0x1F]
    vbool8_t is_negative_b8 = __riscv_vmsne_vx_u8m1_b8(__riscv_vand_vx_u8m1(raw_u8m1, 0x20, vector_length), 0,
                                                       vector_length);
    vuint8m1_t flip_positive_u8m1 = __riscv_vxor_vx_u8m1(raw_u8m1, 0x20, vector_length);
    vuint8m1_t flip_negative_u8m1 = __riscv_vxor_vx_u8m1(raw_u8m1, 0x3F, vector_length);
    return __riscv_vmerge_vvm_u8m1(flip_positive_u8m1, flip_negative_u8m1, is_negative_b8, vector_length);
}

NK_INTERNAL vuint8m1_t nk_comparable_to_fp6m1_rvv_(vuint8m1_t comparable_u8m1, nk_size_t vector_length) {
    // Reverse: if >= 0x20 (was positive), XOR 0x20; else XOR 0x3F (NOT lower 6 bits)
    vbool8_t was_positive_b8 = __riscv_vmsgeu_vx_u8m1_b8(comparable_u8m1, 0x20, vector_length);
    vuint8m1_t from_positive_u8m1 = __riscv_vxor_vx_u8m1(comparable_u8m1, 0x20, vector_length);
    vuint8m1_t from_negative_u8m1 = __riscv_vxor_vx_u8m1(comparable_u8m1, 0x3F, vector_length);
    return __riscv_vmerge_vvm_u8m1(from_negative_u8m1, from_positive_u8m1, was_positive_b8, vector_length);
}

NK_INTERNAL void nk_reduce_moments_i8_rvv_contiguous_( //
    nk_i8_t const *data_ptr, nk_size_t count,          //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    nk_size_t vlmax_elements = __riscv_vsetvlmax_e8m1();
    vint64m4_t sum_i64m4 = __riscv_vmv_v_x_i64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vint8m1_t zero_i8m1 = __riscv_vmv_v_x_i8m1(0, vlmax_elements);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vint8m1_t data_i8m1 = __riscv_vle8_v_i8m1_tu(zero_i8m1, data_ptr, vector_length);

        // Widen i8 → i16 → i32 → i64 for sum
        vint16m2_t data_i16m2 = __riscv_vsext_vf2_i16m2(data_i8m1, vlmax_elements);
        vint32m4_t data_i32m4 = __riscv_vsext_vf2_i32m4(data_i16m2, vlmax_elements);
        vint64m8_t data_i64m8 = __riscv_vsext_vf2_i64m8(data_i32m4, vlmax_elements);

        // Accumulate sum (split m8 into two m4)
        sum_i64m4 = __riscv_vadd_vv_i64m4(sum_i64m4, __riscv_vget_v_i64m8_i64m4(data_i64m8, 0), vlmax);
        sum_i64m4 = __riscv_vadd_vv_i64m4(sum_i64m4, __riscv_vget_v_i64m8_i64m4(data_i64m8, 1), vlmax);

        // Sumsq: i8 × i8 → i16 (widening multiply)
        vint16m2_t squares_i16m2 = __riscv_vwmul_vv_i16m2(data_i8m1, data_i8m1, vlmax_elements);
        // Widen i16 → u32 → u64
        vuint32m4_t squares_u32m4 = __riscv_vwcvtu_x_x_v_u32m4(__riscv_vreinterpret_v_i16m2_u16m2(squares_i16m2),
                                                               vlmax_elements);
        vuint64m8_t squares_u64m8 = __riscv_vwcvtu_x_x_v_u64m8(squares_u32m4, vlmax_elements);

        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 0), vlmax);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 1), vlmax);
    }

    // Horizontal reduction
    vint64m1_t zero_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m4_i64m1(sum_i64m4, zero_i64m1, vlmax));

    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_i8_rvv_strided_(                   //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    nk_size_t vlmax_elements = __riscv_vsetvlmax_e8m1();
    vint64m4_t sum_i64m4 = __riscv_vmv_v_x_i64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vint8m1_t zero_i8m1 = __riscv_vmv_v_x_i8m1(0, vlmax_elements);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vint8m1_t data_i8m1 = __riscv_vlse8_v_i8m1_tu(zero_i8m1, (nk_i8_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                      vector_length);

        // Widen i8 → i16 → i32 → i64 for sum
        vint16m2_t data_i16m2 = __riscv_vsext_vf2_i16m2(data_i8m1, vlmax_elements);
        vint32m4_t data_i32m4 = __riscv_vsext_vf2_i32m4(data_i16m2, vlmax_elements);
        vint64m8_t data_i64m8 = __riscv_vsext_vf2_i64m8(data_i32m4, vlmax_elements);

        // Accumulate sum (split m8 into two m4)
        sum_i64m4 = __riscv_vadd_vv_i64m4(sum_i64m4, __riscv_vget_v_i64m8_i64m4(data_i64m8, 0), vlmax);
        sum_i64m4 = __riscv_vadd_vv_i64m4(sum_i64m4, __riscv_vget_v_i64m8_i64m4(data_i64m8, 1), vlmax);

        // Sumsq: i8 × i8 → i16 (widening multiply)
        vint16m2_t squares_i16m2 = __riscv_vwmul_vv_i16m2(data_i8m1, data_i8m1, vlmax_elements);
        // Widen i16 → u32 → u64
        vuint32m4_t squares_u32m4 = __riscv_vwcvtu_x_x_v_u32m4(__riscv_vreinterpret_v_i16m2_u16m2(squares_i16m2),
                                                               vlmax_elements);
        vuint64m8_t squares_u64m8 = __riscv_vwcvtu_x_x_v_u64m8(squares_u32m4, vlmax_elements);

        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 0), vlmax);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 1), vlmax);
    }

    // Horizontal reduction
    vint64m1_t zero_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m4_i64m1(sum_i64m4, zero_i64m1, vlmax));

    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_i8_rvv(                              //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_i8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_i8_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_i8_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_i8_rvv_contiguous_( //
    nk_i8_t const *data_ptr, nk_size_t count,         //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vint8m1_t min_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MAX, vlmax);
    vint8m1_t max_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MIN, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vint8m1_t data_i8m1 = __riscv_vle8_v_i8m1(data_ptr, vector_length);

        // VID-based absolute indices
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool8_t less_b8 = __riscv_vmslt_vv_i8m1_b8(data_i8m1, min_i8m1, vector_length);
        min_i8m1 = __riscv_vmerge_vvm_i8m1_tu(min_i8m1, min_i8m1, data_i8m1, less_b8, vector_length);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(min_indices_u64m8, min_indices_u64m8, pos_u64m8, less_b8,
                                                        vector_length);

        vbool8_t greater_b8 = __riscv_vmslt_vv_i8m1_b8(max_i8m1, data_i8m1, vector_length);
        max_i8m1 = __riscv_vmerge_vvm_i8m1_tu(max_i8m1, max_i8m1, data_i8m1, greater_b8, vector_length);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(max_indices_u64m8, max_indices_u64m8, pos_u64m8, greater_b8,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vint8m1_t init_max_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MAX, 1);
    nk_i8_t min_val = __riscv_vmv_x_s_i8m1_i8(__riscv_vredmin_vs_i8m1_i8m1(min_i8m1, init_max_i8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_i8m1_b8(min_i8m1, min_val, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint8m1_t init_min_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MIN, 1);
    nk_i8_t max_val = __riscv_vmv_x_s_i8m1_i8(__riscv_vredmax_vs_i8m1_i8m1(max_i8m1, init_min_i8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_i8m1_b8(max_i8m1, max_val, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_i8_rvv_strided_(                    //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vint8m1_t min_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MAX, vlmax);
    vint8m1_t max_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MIN, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vint8m1_t data_i8m1 = __riscv_vlse8_v_i8m1((nk_i8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // VID-based absolute indices
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool8_t less_b8 = __riscv_vmslt_vv_i8m1_b8(data_i8m1, min_i8m1, vector_length);
        min_i8m1 = __riscv_vmerge_vvm_i8m1_tu(min_i8m1, min_i8m1, data_i8m1, less_b8, vector_length);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(min_indices_u64m8, min_indices_u64m8, pos_u64m8, less_b8,
                                                        vector_length);

        vbool8_t greater_b8 = __riscv_vmslt_vv_i8m1_b8(max_i8m1, data_i8m1, vector_length);
        max_i8m1 = __riscv_vmerge_vvm_i8m1_tu(max_i8m1, max_i8m1, data_i8m1, greater_b8, vector_length);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(max_indices_u64m8, max_indices_u64m8, pos_u64m8, greater_b8,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vint8m1_t init_max_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MAX, 1);
    nk_i8_t min_val = __riscv_vmv_x_s_i8m1_i8(__riscv_vredmin_vs_i8m1_i8m1(min_i8m1, init_max_i8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_i8m1_b8(min_i8m1, min_val, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint8m1_t init_min_i8m1 = __riscv_vmv_v_x_i8m1(NK_I8_MIN, 1);
    nk_i8_t max_val = __riscv_vmv_x_s_i8m1_i8(__riscv_vredmax_vs_i8m1_i8m1(max_i8m1, init_min_i8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_i8m1_b8(max_i8m1, max_val, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_i8_rvv(                               //
    nk_i8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i8_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i8_t);
    int aligned = (stride_bytes % sizeof(nk_i8_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_I8_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I8_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i8_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_i8_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                            max_index_ptr);
    else
        nk_reduce_minmax_i8_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                         max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u8_rvv_contiguous_( //
    nk_u8_t const *data_ptr, nk_size_t count,          //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    nk_size_t vlmax_elements = __riscv_vsetvlmax_e8m1();
    vuint64m4_t sum_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint8m1_t zero_u8m1 = __riscv_vmv_v_x_u8m1(0, vlmax_elements);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1_tu(zero_u8m1, data_ptr, vector_length);

        // Widen u8 → u16 → u32 → u64 for sum
        vuint16m2_t data_u16m2 = __riscv_vzext_vf2_u16m2(data_u8m1, vlmax_elements);
        vuint32m4_t data_u32m4 = __riscv_vzext_vf2_u32m4(data_u16m2, vlmax_elements);
        vuint64m8_t data_u64m8 = __riscv_vzext_vf2_u64m8(data_u32m4, vlmax_elements);

        // Accumulate sum (split m8 into two m4)
        sum_u64m4 = __riscv_vadd_vv_u64m4(sum_u64m4, __riscv_vget_v_u64m8_u64m4(data_u64m8, 0), vlmax);
        sum_u64m4 = __riscv_vadd_vv_u64m4(sum_u64m4, __riscv_vget_v_u64m8_u64m4(data_u64m8, 1), vlmax);

        // Sumsq: u8 × u8 → u16 (widening multiply)
        vuint16m2_t squares_u16m2 = __riscv_vwmulu_vv_u16m2(data_u8m1, data_u8m1, vlmax_elements);
        // Widen u16 → u32 → u64
        vuint32m4_t squares_u32m4 = __riscv_vzext_vf2_u32m4(squares_u16m2, vlmax_elements);
        vuint64m8_t squares_u64m8 = __riscv_vzext_vf2_u64m8(squares_u32m4, vlmax_elements);

        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 0), vlmax);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 1), vlmax);
    }

    // Horizontal reduction
    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sum_u64m4, zero_u64m1, vlmax)),
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_u8_rvv_strided_(                   //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    nk_size_t vlmax_elements = __riscv_vsetvlmax_e8m1();
    vuint64m4_t sum_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint8m1_t zero_u8m1 = __riscv_vmv_v_x_u8m1(0, vlmax_elements);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1_tu(zero_u8m1, (nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes,
                                                       vector_length);

        // Widen u8 → u16 → u32 → u64 for sum
        vuint16m2_t data_u16m2 = __riscv_vzext_vf2_u16m2(data_u8m1, vlmax_elements);
        vuint32m4_t data_u32m4 = __riscv_vzext_vf2_u32m4(data_u16m2, vlmax_elements);
        vuint64m8_t data_u64m8 = __riscv_vzext_vf2_u64m8(data_u32m4, vlmax_elements);

        // Accumulate sum (split m8 into two m4)
        sum_u64m4 = __riscv_vadd_vv_u64m4(sum_u64m4, __riscv_vget_v_u64m8_u64m4(data_u64m8, 0), vlmax);
        sum_u64m4 = __riscv_vadd_vv_u64m4(sum_u64m4, __riscv_vget_v_u64m8_u64m4(data_u64m8, 1), vlmax);

        // Sumsq: u8 × u8 → u16 (widening multiply)
        vuint16m2_t squares_u16m2 = __riscv_vwmulu_vv_u16m2(data_u8m1, data_u8m1, vlmax_elements);
        // Widen u16 → u32 → u64
        vuint32m4_t squares_u32m4 = __riscv_vzext_vf2_u32m4(squares_u16m2, vlmax_elements);
        vuint64m8_t squares_u64m8 = __riscv_vzext_vf2_u64m8(squares_u32m4, vlmax_elements);

        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 0), vlmax);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4(sumsq_u64m4, __riscv_vget_v_u64m8_u64m4(squares_u64m8, 1), vlmax);
    }

    // Horizontal reduction
    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sum_u64m4, zero_u64m1, vlmax)),
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_u8_rvv(                              //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_u8_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_u8_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_u8_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_u8_rvv_contiguous_( //
    nk_u8_t const *data_ptr, nk_size_t count,         //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MAX, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MIN, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1(data_ptr, vector_length);

        // VID-based absolute indices
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(data_u8m1, min_u8m1, vector_length);
        min_u8m1 = __riscv_vmerge_vvm_u8m1_tu(min_u8m1, min_u8m1, data_u8m1, less_b8, vector_length);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(min_indices_u64m8, min_indices_u64m8, pos_u64m8, less_b8,
                                                        vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, data_u8m1, vector_length);
        max_u8m1 = __riscv_vmerge_vvm_u8m1_tu(max_u8m1, max_u8m1, data_u8m1, greater_b8, vector_length);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(max_indices_u64m8, max_indices_u64m8, pos_u64m8, greater_b8,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MAX, 1);
    nk_u8_t min_val = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_val, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MIN, 1);
    nk_u8_t max_val = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_val, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_u8_rvv_strided_(                    //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MAX, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MIN, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((nk_u8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // VID-based absolute indices
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(data_u8m1, min_u8m1, vector_length);
        min_u8m1 = __riscv_vmerge_vvm_u8m1_tu(min_u8m1, min_u8m1, data_u8m1, less_b8, vector_length);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(min_indices_u64m8, min_indices_u64m8, pos_u64m8, less_b8,
                                                        vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, data_u8m1, vector_length);
        max_u8m1 = __riscv_vmerge_vvm_u8m1_tu(max_u8m1, max_u8m1, data_u8m1, greater_b8, vector_length);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(max_indices_u64m8, max_indices_u64m8, pos_u64m8, greater_b8,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MAX, 1);
    nk_u8_t min_val = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_val, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(NK_U8_MIN, 1);
    nk_u8_t max_val = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_val, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_u8_rvv(                               //
    nk_u8_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u8_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u8_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u8_t);
    int aligned = (stride_bytes % sizeof(nk_u8_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_U8_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_U8_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u8_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                   max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_u8_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                            max_index_ptr);
    else
        nk_reduce_minmax_u8_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                         max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i16_rvv_contiguous_( //
    nk_i16_t const *data_ptr, nk_size_t count,          //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vint64m4_t sum_i64m4 = __riscv_vmv_v_x_i64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vint16m1_t data_i16m1 = __riscv_vle16_v_i16m1(data_ptr, vector_length);

        // Widen i16 → i32 → i64 for sum
        vint32m2_t data_i32m2 = __riscv_vsext_vf2_i32m2(data_i16m1, vector_length);
        vint64m4_t data_i64m4 = __riscv_vsext_vf2_i64m4(data_i32m2, vector_length);
        sum_i64m4 = __riscv_vadd_vv_i64m4_tu(sum_i64m4, sum_i64m4, data_i64m4, vector_length);

        // Sumsq: i16 × i16 → i32 (widening multiply)
        vint32m2_t squares_i32m2 = __riscv_vwmul_vv_i32m2(data_i16m1, data_i16m1, vector_length);
        // Widen i32 → u64
        vuint64m4_t squares_u64m4 = __riscv_vwcvtu_x_x_v_u64m4(__riscv_vreinterpret_v_i32m2_u32m2(squares_i32m2),
                                                               vector_length);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4_tu(sumsq_u64m4, sumsq_u64m4, squares_u64m4, vector_length);
    }

    // Horizontal reduction
    vint64m1_t zero_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m4_i64m1(sum_i64m4, zero_i64m1, vlmax));

    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_i16_rvv_strided_(                   //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vint64m4_t sum_i64m4 = __riscv_vmv_v_x_i64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vint16m1_t data_i16m1 = __riscv_vlse16_v_i16m1((nk_i16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // Widen i16 → i32 → i64 for sum
        vint32m2_t data_i32m2 = __riscv_vsext_vf2_i32m2(data_i16m1, vector_length);
        vint64m4_t data_i64m4 = __riscv_vsext_vf2_i64m4(data_i32m2, vector_length);
        sum_i64m4 = __riscv_vadd_vv_i64m4_tu(sum_i64m4, sum_i64m4, data_i64m4, vector_length);

        // Sumsq: i16 × i16 → i32 (widening multiply)
        vint32m2_t squares_i32m2 = __riscv_vwmul_vv_i32m2(data_i16m1, data_i16m1, vector_length);
        // Widen i32 → u64
        vuint64m4_t squares_u64m4 = __riscv_vwcvtu_x_x_v_u64m4(__riscv_vreinterpret_v_i32m2_u32m2(squares_i32m2),
                                                               vector_length);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4_tu(sumsq_u64m4, sumsq_u64m4, squares_u64m4, vector_length);
    }

    // Horizontal reduction
    vint64m1_t zero_i64m1 = __riscv_vmv_v_x_i64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m4_i64m1(sum_i64m4, zero_i64m1, vlmax));

    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_i16_rvv(                              //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_i16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_i16_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_i16_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_i16_rvv_contiguous_( //
    nk_i16_t const *data_ptr, nk_size_t count,         //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vint16m1_t min_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MAX, vlmax);
    vint16m1_t max_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MIN, vlmax);
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vint16m1_t data_i16m1 = __riscv_vle16_v_i16m1(data_ptr, vector_length);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool16_t less_b16 = __riscv_vmslt_vv_i16m1_b16(data_i16m1, min_i16m1, vector_length);
        min_i16m1 = __riscv_vmerge_vvm_i16m1_tu(min_i16m1, min_i16m1, data_i16m1, less_b16, vector_length);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(min_indices_u64m4, min_indices_u64m4, pos_u64m4, less_b16,
                                                        vector_length);

        vbool16_t greater_b16 = __riscv_vmslt_vv_i16m1_b16(max_i16m1, data_i16m1, vector_length);
        max_i16m1 = __riscv_vmerge_vvm_i16m1_tu(max_i16m1, max_i16m1, data_i16m1, greater_b16, vector_length);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(max_indices_u64m4, max_indices_u64m4, pos_u64m4, greater_b16,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vint16m1_t init_max_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MAX, 1);
    nk_i16_t min_val = __riscv_vmv_x_s_i16m1_i16(__riscv_vredmin_vs_i16m1_i16m1(min_i16m1, init_max_i16m1, vlmax));
    vbool16_t min_match_b16 = __riscv_vmseq_vx_i16m1_b16(min_i16m1, min_val, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint16m1_t init_min_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MIN, 1);
    nk_i16_t max_val = __riscv_vmv_x_s_i16m1_i16(__riscv_vredmax_vs_i16m1_i16m1(max_i16m1, init_min_i16m1, vlmax));
    vbool16_t max_match_b16 = __riscv_vmseq_vx_i16m1_b16(max_i16m1, max_val, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_i16_rvv_strided_(                    //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vint16m1_t min_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MAX, vlmax);
    vint16m1_t max_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MIN, vlmax);
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vint16m1_t data_i16m1 = __riscv_vlse16_v_i16m1((nk_i16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool16_t less_b16 = __riscv_vmslt_vv_i16m1_b16(data_i16m1, min_i16m1, vector_length);
        min_i16m1 = __riscv_vmerge_vvm_i16m1_tu(min_i16m1, min_i16m1, data_i16m1, less_b16, vector_length);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(min_indices_u64m4, min_indices_u64m4, pos_u64m4, less_b16,
                                                        vector_length);

        vbool16_t greater_b16 = __riscv_vmslt_vv_i16m1_b16(max_i16m1, data_i16m1, vector_length);
        max_i16m1 = __riscv_vmerge_vvm_i16m1_tu(max_i16m1, max_i16m1, data_i16m1, greater_b16, vector_length);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(max_indices_u64m4, max_indices_u64m4, pos_u64m4, greater_b16,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vint16m1_t init_max_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MAX, 1);
    nk_i16_t min_val = __riscv_vmv_x_s_i16m1_i16(__riscv_vredmin_vs_i16m1_i16m1(min_i16m1, init_max_i16m1, vlmax));
    vbool16_t min_match_b16 = __riscv_vmseq_vx_i16m1_b16(min_i16m1, min_val, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint16m1_t init_min_i16m1 = __riscv_vmv_v_x_i16m1(NK_I16_MIN, 1);
    nk_i16_t max_val = __riscv_vmv_x_s_i16m1_i16(__riscv_vredmax_vs_i16m1_i16m1(max_i16m1, init_min_i16m1, vlmax));
    vbool16_t max_match_b16 = __riscv_vmseq_vx_i16m1_b16(max_i16m1, max_val, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_i16_rvv(                               //
    nk_i16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i16_t);
    int aligned = (stride_bytes % sizeof(nk_i16_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_I16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_i16_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_i16_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u16_rvv_contiguous_( //
    nk_u16_t const *data_ptr, nk_size_t count,          //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vuint64m4_t sum_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1(data_ptr, vector_length);

        // Widen u16 → u32 → u64 for sum
        vuint32m2_t data_u32m2 = __riscv_vzext_vf2_u32m2(data_u16m1, vector_length);
        vuint64m4_t data_u64m4 = __riscv_vzext_vf2_u64m4(data_u32m2, vector_length);
        sum_u64m4 = __riscv_vadd_vv_u64m4_tu(sum_u64m4, sum_u64m4, data_u64m4, vector_length);

        // Sumsq: u16 × u16 → u32 (widening multiply)
        vuint32m2_t squares_u32m2 = __riscv_vwmulu_vv_u32m2(data_u16m1, data_u16m1, vector_length);
        // Widen u32 → u64
        vuint64m4_t squares_u64m4 = __riscv_vzext_vf2_u64m4(squares_u32m2, vector_length);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4_tu(sumsq_u64m4, sumsq_u64m4, squares_u64m4, vector_length);
    }

    // Horizontal reduction
    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sum_u64m4, zero_u64m1, vlmax)),
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_u16_rvv_strided_(                   //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vuint64m4_t sum_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t sumsq_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // Widen u16 → u32 → u64 for sum
        vuint32m2_t data_u32m2 = __riscv_vzext_vf2_u32m2(data_u16m1, vector_length);
        vuint64m4_t data_u64m4 = __riscv_vzext_vf2_u64m4(data_u32m2, vector_length);
        sum_u64m4 = __riscv_vadd_vv_u64m4_tu(sum_u64m4, sum_u64m4, data_u64m4, vector_length);

        // Sumsq: u16 × u16 → u32 (widening multiply)
        vuint32m2_t squares_u32m2 = __riscv_vwmulu_vv_u32m2(data_u16m1, data_u16m1, vector_length);
        // Widen u32 → u64
        vuint64m4_t squares_u64m4 = __riscv_vzext_vf2_u64m4(squares_u32m2, vector_length);
        sumsq_u64m4 = __riscv_vadd_vv_u64m4_tu(sumsq_u64m4, sumsq_u64m4, squares_u64m4, vector_length);
    }

    // Horizontal reduction
    vuint64m1_t zero_u64m1 = __riscv_vmv_v_x_u64m1(0, 1);
    *sum_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sum_u64m4, zero_u64m1, vlmax)),
    *sumsq_ptr = __riscv_vmv_x_s_u64m1_u64(__riscv_vredsum_vs_u64m4_u64m1(sumsq_u64m4, zero_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_u16_rvv(                              //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_u16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_u16_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_u16_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_u16_rvv_contiguous_( //
    nk_u16_t const *data_ptr, nk_size_t count,         //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vuint16m1_t min_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MAX, vlmax);
    vuint16m1_t max_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MIN, vlmax);
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1(data_ptr, vector_length);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool16_t less_b16 = __riscv_vmsltu_vv_u16m1_b16(data_u16m1, min_u16m1, vector_length);
        min_u16m1 = __riscv_vmerge_vvm_u16m1_tu(min_u16m1, min_u16m1, data_u16m1, less_b16, vector_length);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(min_indices_u64m4, min_indices_u64m4, pos_u64m4, less_b16,
                                                        vector_length);

        vbool16_t greater_b16 = __riscv_vmsltu_vv_u16m1_b16(max_u16m1, data_u16m1, vector_length);
        max_u16m1 = __riscv_vmerge_vvm_u16m1_tu(max_u16m1, max_u16m1, data_u16m1, greater_b16, vector_length);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(max_indices_u64m4, max_indices_u64m4, pos_u64m4, greater_b16,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vuint16m1_t init_max_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MAX, 1);
    nk_u16_t min_val = __riscv_vmv_x_s_u16m1_u16(__riscv_vredminu_vs_u16m1_u16m1(min_u16m1, init_max_u16m1, vlmax));
    vbool16_t min_match_b16 = __riscv_vmseq_vx_u16m1_b16(min_u16m1, min_val, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint16m1_t init_min_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MIN, 1);
    nk_u16_t max_val = __riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m1_u16m1(max_u16m1, init_min_u16m1, vlmax));
    vbool16_t max_match_b16 = __riscv_vmseq_vx_u16m1_b16(max_u16m1, max_val, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_u16_rvv_strided_(                    //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vuint16m1_t min_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MAX, vlmax);
    vuint16m1_t max_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MIN, vlmax);
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool16_t less_b16 = __riscv_vmsltu_vv_u16m1_b16(data_u16m1, min_u16m1, vector_length);
        min_u16m1 = __riscv_vmerge_vvm_u16m1_tu(min_u16m1, min_u16m1, data_u16m1, less_b16, vector_length);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(min_indices_u64m4, min_indices_u64m4, pos_u64m4, less_b16,
                                                        vector_length);

        vbool16_t greater_b16 = __riscv_vmsltu_vv_u16m1_b16(max_u16m1, data_u16m1, vector_length);
        max_u16m1 = __riscv_vmerge_vvm_u16m1_tu(max_u16m1, max_u16m1, data_u16m1, greater_b16, vector_length);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(max_indices_u64m4, max_indices_u64m4, pos_u64m4, greater_b16,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vuint16m1_t init_max_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MAX, 1);
    nk_u16_t min_val = __riscv_vmv_x_s_u16m1_u16(__riscv_vredminu_vs_u16m1_u16m1(min_u16m1, init_max_u16m1, vlmax));
    vbool16_t min_match_b16 = __riscv_vmseq_vx_u16m1_b16(min_u16m1, min_val, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint16m1_t init_min_u16m1 = __riscv_vmv_v_x_u16m1(NK_U16_MIN, 1);
    nk_u16_t max_val = __riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m1_u16m1(max_u16m1, init_min_u16m1, vlmax));
    vbool16_t max_match_b16 = __riscv_vmseq_vx_u16m1_b16(max_u16m1, max_val, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_u16_rvv(                               //
    nk_u16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u16_t);
    int aligned = (stride_bytes % sizeof(nk_u16_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_U16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_U16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_u16_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_u16_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i32_rvv_contiguous_( //
    nk_i32_t const *data_ptr, nk_size_t count,          //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    // 128-bit per-lane accumulator for sum: (sum_upper, sum_lower)
    vuint64m2_t sum_lower_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vint64m2_t sum_upper_i64m2 = __riscv_vmv_v_x_i64m2(0, vlmax);
    vuint64m2_t sumsq_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vint32m1_t data_i32m1 = __riscv_vle32_v_i32m1(data_ptr, vector_length);

        // Widen i32 → i64
        vint64m2_t data_i64m2 = __riscv_vsext_vf2_i64m2(data_i32m1, vector_length);
        vuint64m2_t data_u64m2 = __riscv_vreinterpret_v_i64m2_u64m2(data_i64m2);

        // 128-bit accumulation: wrapping add on lower half
        vuint64m2_t sum_before_u64m2 = sum_lower_u64m2;
        sum_lower_u64m2 = __riscv_vadd_vv_u64m2_tu(sum_lower_u64m2, sum_lower_u64m2, data_u64m2, vector_length);

        // Carry: new < old means unsigned overflow occurred
        vbool32_t carry_b32 = __riscv_vmsltu_vv_u64m2_b32(sum_lower_u64m2, sum_before_u64m2, vector_length);
        vint64m2_t carry_i64m2 = __riscv_vmerge_vxm_i64m2(__riscv_vmv_v_x_i64m2(0, vector_length), 1, carry_b32,
                                                          vector_length);
        sum_upper_i64m2 = __riscv_vadd_vv_i64m2_tu(sum_upper_i64m2, sum_upper_i64m2, carry_i64m2, vector_length);

        // Sign extension: -1 for negative, 0 for non-negative
        vint64m2_t sign_ext_i64m2 = __riscv_vsra_vx_i64m2(data_i64m2, 63, vector_length);
        sum_upper_i64m2 = __riscv_vadd_vv_i64m2_tu(sum_upper_i64m2, sum_upper_i64m2, sign_ext_i64m2, vector_length);

        // Sumsq: i32 × i32 → i64 (widening multiply, result ≤ 2^62), saturating accumulation
        vint64m2_t squares_i64m2 = __riscv_vwmul_vv_i64m2(data_i32m1, data_i32m1, vector_length);
        sumsq_u64m2 = __riscv_vsaddu_vv_u64m2_tu(sumsq_u64m2, sumsq_u64m2,
                                                 __riscv_vreinterpret_v_i64m2_u64m2(squares_i64m2), vector_length);
    }

    *sum_ptr = nk_reduce_128bit_sum_i64m2_rvv_(sum_lower_u64m2, sum_upper_i64m2, vlmax);
    *sumsq_ptr = nk_reduce_vsaddu_u64m2_rvv_(sumsq_u64m2, vlmax);
}

NK_INTERNAL void nk_reduce_moments_i32_rvv_strided_(                   //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    // 128-bit per-lane accumulator for sum: (sum_upper, sum_lower)
    vuint64m2_t sum_lower_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vint64m2_t sum_upper_i64m2 = __riscv_vmv_v_x_i64m2(0, vlmax);
    vuint64m2_t sumsq_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vint32m1_t data_i32m1 = __riscv_vlse32_v_i32m1((nk_i32_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // Widen i32 → i64
        vint64m2_t data_i64m2 = __riscv_vsext_vf2_i64m2(data_i32m1, vector_length);
        vuint64m2_t data_u64m2 = __riscv_vreinterpret_v_i64m2_u64m2(data_i64m2);

        // 128-bit accumulation: wrapping add on lower half
        vuint64m2_t sum_before_u64m2 = sum_lower_u64m2;
        sum_lower_u64m2 = __riscv_vadd_vv_u64m2_tu(sum_lower_u64m2, sum_lower_u64m2, data_u64m2, vector_length);

        // Carry: new < old means unsigned overflow occurred
        vbool32_t carry_b32 = __riscv_vmsltu_vv_u64m2_b32(sum_lower_u64m2, sum_before_u64m2, vector_length);
        vint64m2_t carry_i64m2 = __riscv_vmerge_vxm_i64m2(__riscv_vmv_v_x_i64m2(0, vector_length), 1, carry_b32,
                                                          vector_length);
        sum_upper_i64m2 = __riscv_vadd_vv_i64m2_tu(sum_upper_i64m2, sum_upper_i64m2, carry_i64m2, vector_length);

        // Sign extension: -1 for negative, 0 for non-negative
        vint64m2_t sign_ext_i64m2 = __riscv_vsra_vx_i64m2(data_i64m2, 63, vector_length);
        sum_upper_i64m2 = __riscv_vadd_vv_i64m2_tu(sum_upper_i64m2, sum_upper_i64m2, sign_ext_i64m2, vector_length);

        // Sumsq: i32 × i32 → i64 (widening multiply, result ≤ 2^62), saturating accumulation
        vint64m2_t squares_i64m2 = __riscv_vwmul_vv_i64m2(data_i32m1, data_i32m1, vector_length);
        sumsq_u64m2 = __riscv_vsaddu_vv_u64m2_tu(sumsq_u64m2, sumsq_u64m2,
                                                 __riscv_vreinterpret_v_i64m2_u64m2(squares_i64m2), vector_length);
    }

    *sum_ptr = nk_reduce_128bit_sum_i64m2_rvv_(sum_lower_u64m2, sum_upper_i64m2, vlmax);
    *sumsq_ptr = nk_reduce_vsaddu_u64m2_rvv_(sumsq_u64m2, vlmax);
}

NK_PUBLIC void nk_reduce_moments_i32_rvv(                              //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_i32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_i32_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_i32_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_i32_rvv_contiguous_( //
    nk_i32_t const *data_ptr, nk_size_t count,         //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m1();
    vint32m1_t min_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MAX, vlmax);
    vint32m1_t max_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MIN, vlmax);
    vuint64m2_t min_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t max_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vint32m1_t data_i32m1 = __riscv_vle32_v_i32m1(data_ptr, vector_length);
        vuint64m2_t pos_u64m2 = __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool32_t less_b32 = __riscv_vmslt_vv_i32m1_b32(data_i32m1, min_i32m1, vector_length);
        min_i32m1 = __riscv_vmerge_vvm_i32m1_tu(min_i32m1, min_i32m1, data_i32m1, less_b32, vector_length);
        min_indices_u64m2 = __riscv_vmerge_vvm_u64m2_tu(min_indices_u64m2, min_indices_u64m2, pos_u64m2, less_b32,
                                                        vector_length);

        vbool32_t greater_b32 = __riscv_vmslt_vv_i32m1_b32(max_i32m1, data_i32m1, vector_length);
        max_i32m1 = __riscv_vmerge_vvm_i32m1_tu(max_i32m1, max_i32m1, data_i32m1, greater_b32, vector_length);
        max_indices_u64m2 = __riscv_vmerge_vvm_u64m2_tu(max_indices_u64m2, max_indices_u64m2, pos_u64m2, greater_b32,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vint32m1_t init_max_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MAX, 1);
    nk_i32_t min_val = __riscv_vmv_x_s_i32m1_i32(__riscv_vredmin_vs_i32m1_i32m1(min_i32m1, init_max_i32m1, vlmax));
    vbool32_t min_match_b32 = __riscv_vmseq_vx_i32m1_b32(min_i32m1, min_val, vlmax);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vlmax);
    vuint64m2_t min_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, min_indices_u64m2, min_match_b32, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(min_cands_u64m2, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint32m1_t init_min_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MIN, 1);
    nk_i32_t max_val = __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m1_i32m1(max_i32m1, init_min_i32m1, vlmax));
    vbool32_t max_match_b32 = __riscv_vmseq_vx_i32m1_b32(max_i32m1, max_val, vlmax);
    vuint64m2_t max_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, max_indices_u64m2, max_match_b32, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(max_cands_u64m2, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_i32_rvv_strided_(                    //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m1();
    vint32m1_t min_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MAX, vlmax);
    vint32m1_t max_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MIN, vlmax);
    vuint64m2_t min_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t max_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vint32m1_t data_i32m1 = __riscv_vlse32_v_i32m1((nk_i32_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint64m2_t pos_u64m2 = __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool32_t less_b32 = __riscv_vmslt_vv_i32m1_b32(data_i32m1, min_i32m1, vector_length);
        min_i32m1 = __riscv_vmerge_vvm_i32m1_tu(min_i32m1, min_i32m1, data_i32m1, less_b32, vector_length);
        min_indices_u64m2 = __riscv_vmerge_vvm_u64m2_tu(min_indices_u64m2, min_indices_u64m2, pos_u64m2, less_b32,
                                                        vector_length);

        vbool32_t greater_b32 = __riscv_vmslt_vv_i32m1_b32(max_i32m1, data_i32m1, vector_length);
        max_i32m1 = __riscv_vmerge_vvm_i32m1_tu(max_i32m1, max_i32m1, data_i32m1, greater_b32, vector_length);
        max_indices_u64m2 = __riscv_vmerge_vvm_u64m2_tu(max_indices_u64m2, max_indices_u64m2, pos_u64m2, greater_b32,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vint32m1_t init_max_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MAX, 1);
    nk_i32_t min_val = __riscv_vmv_x_s_i32m1_i32(__riscv_vredmin_vs_i32m1_i32m1(min_i32m1, init_max_i32m1, vlmax));
    vbool32_t min_match_b32 = __riscv_vmseq_vx_i32m1_b32(min_i32m1, min_val, vlmax);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vlmax);
    vuint64m2_t min_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, min_indices_u64m2, min_match_b32, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(min_cands_u64m2, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint32m1_t init_min_i32m1 = __riscv_vmv_v_x_i32m1(NK_I32_MIN, 1);
    nk_i32_t max_val = __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m1_i32m1(max_i32m1, init_min_i32m1, vlmax));
    vbool32_t max_match_b32 = __riscv_vmseq_vx_i32m1_b32(max_i32m1, max_val, vlmax);
    vuint64m2_t max_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, max_indices_u64m2, max_match_b32, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(max_cands_u64m2, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_i32_rvv(                               //
    nk_i32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i32_t);
    int aligned = (stride_bytes % sizeof(nk_i32_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_I32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_i32_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_i32_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u32_rvv_contiguous_( //
    nk_u32_t const *data_ptr, nk_size_t count,          //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vuint64m2_t sum_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t sumsq_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vuint32m1_t data_u32m1 = __riscv_vle32_v_u32m1(data_ptr, vector_length);

        // Widen u32 → u64 for saturating sum
        vuint64m2_t data_u64m2 = __riscv_vzext_vf2_u64m2(data_u32m1, vector_length);
        sum_u64m2 = __riscv_vsaddu_vv_u64m2_tu(sum_u64m2, sum_u64m2, data_u64m2, vector_length);

        // Sumsq: u32 × u32 → u64 (widening multiply, no overflow), saturating accumulation
        vuint64m2_t squares_u64m2 = __riscv_vwmulu_vv_u64m2(data_u32m1, data_u32m1, vector_length);
        sumsq_u64m2 = __riscv_vsaddu_vv_u64m2_tu(sumsq_u64m2, sumsq_u64m2, squares_u64m2, vector_length);
    }

    *sum_ptr = nk_reduce_vsaddu_u64m2_rvv_(sum_u64m2, vlmax);
    *sumsq_ptr = nk_reduce_vsaddu_u64m2_rvv_(sumsq_u64m2, vlmax);
}

NK_INTERNAL void nk_reduce_moments_u32_rvv_strided_(                   //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vuint64m2_t sum_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t sumsq_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vuint32m1_t data_u32m1 = __riscv_vlse32_v_u32m1((nk_u32_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // Widen u32 → u64 for saturating sum
        vuint64m2_t data_u64m2 = __riscv_vzext_vf2_u64m2(data_u32m1, vector_length);
        sum_u64m2 = __riscv_vsaddu_vv_u64m2_tu(sum_u64m2, sum_u64m2, data_u64m2, vector_length);

        // Sumsq: u32 × u32 → u64 (widening multiply, no overflow), saturating accumulation
        vuint64m2_t squares_u64m2 = __riscv_vwmulu_vv_u64m2(data_u32m1, data_u32m1, vector_length);
        sumsq_u64m2 = __riscv_vsaddu_vv_u64m2_tu(sumsq_u64m2, sumsq_u64m2, squares_u64m2, vector_length);
    }

    *sum_ptr = nk_reduce_vsaddu_u64m2_rvv_(sum_u64m2, vlmax);
    *sumsq_ptr = nk_reduce_vsaddu_u64m2_rvv_(sumsq_u64m2, vlmax);
}

NK_PUBLIC void nk_reduce_moments_u32_rvv(                              //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_u32_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_u32_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_u32_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_u32_rvv_contiguous_( //
    nk_u32_t const *data_ptr, nk_size_t count,         //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m1();
    vuint32m1_t min_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MAX, vlmax);
    vuint32m1_t max_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MIN, vlmax);
    vuint64m2_t min_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t max_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vuint32m1_t data_u32m1 = __riscv_vle32_v_u32m1(data_ptr, vector_length);
        vuint64m2_t pos_u64m2 = __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool32_t less_b32 = __riscv_vmsltu_vv_u32m1_b32(data_u32m1, min_u32m1, vector_length);
        min_u32m1 = __riscv_vmerge_vvm_u32m1_tu(min_u32m1, min_u32m1, data_u32m1, less_b32, vector_length);
        min_indices_u64m2 = __riscv_vmerge_vvm_u64m2_tu(min_indices_u64m2, min_indices_u64m2, pos_u64m2, less_b32,
                                                        vector_length);

        vbool32_t greater_b32 = __riscv_vmsltu_vv_u32m1_b32(max_u32m1, data_u32m1, vector_length);
        max_u32m1 = __riscv_vmerge_vvm_u32m1_tu(max_u32m1, max_u32m1, data_u32m1, greater_b32, vector_length);
        max_indices_u64m2 = __riscv_vmerge_vvm_u64m2_tu(max_indices_u64m2, max_indices_u64m2, pos_u64m2, greater_b32,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vuint32m1_t init_max_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MAX, 1);
    nk_u32_t min_val = __riscv_vmv_x_s_u32m1_u32(__riscv_vredminu_vs_u32m1_u32m1(min_u32m1, init_max_u32m1, vlmax));
    vbool32_t min_match_b32 = __riscv_vmseq_vx_u32m1_b32(min_u32m1, min_val, vlmax);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vlmax);
    vuint64m2_t min_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, min_indices_u64m2, min_match_b32, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(min_cands_u64m2, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint32m1_t init_min_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MIN, 1);
    nk_u32_t max_val = __riscv_vmv_x_s_u32m1_u32(__riscv_vredmaxu_vs_u32m1_u32m1(max_u32m1, init_min_u32m1, vlmax));
    vbool32_t max_match_b32 = __riscv_vmseq_vx_u32m1_b32(max_u32m1, max_val, vlmax);
    vuint64m2_t max_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, max_indices_u64m2, max_match_b32, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(max_cands_u64m2, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_u32_rvv_strided_(                    //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m1();
    vuint32m1_t min_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MAX, vlmax);
    vuint32m1_t max_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MIN, vlmax);
    vuint64m2_t min_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    vuint64m2_t max_indices_u64m2 = __riscv_vmv_v_x_u64m2(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e32m1(count);
        vuint32m1_t data_u32m1 = __riscv_vlse32_v_u32m1((nk_u32_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint64m2_t pos_u64m2 = __riscv_vadd_vx_u64m2(__riscv_vid_v_u64m2(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool32_t less_b32 = __riscv_vmsltu_vv_u32m1_b32(data_u32m1, min_u32m1, vector_length);
        min_u32m1 = __riscv_vmerge_vvm_u32m1_tu(min_u32m1, min_u32m1, data_u32m1, less_b32, vector_length);
        min_indices_u64m2 = __riscv_vmerge_vvm_u64m2_tu(min_indices_u64m2, min_indices_u64m2, pos_u64m2, less_b32,
                                                        vector_length);

        vbool32_t greater_b32 = __riscv_vmsltu_vv_u32m1_b32(max_u32m1, data_u32m1, vector_length);
        max_u32m1 = __riscv_vmerge_vvm_u32m1_tu(max_u32m1, max_u32m1, data_u32m1, greater_b32, vector_length);
        max_indices_u64m2 = __riscv_vmerge_vvm_u64m2_tu(max_indices_u64m2, max_indices_u64m2, pos_u64m2, greater_b32,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vuint32m1_t init_max_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MAX, 1);
    nk_u32_t min_val = __riscv_vmv_x_s_u32m1_u32(__riscv_vredminu_vs_u32m1_u32m1(min_u32m1, init_max_u32m1, vlmax));
    vbool32_t min_match_b32 = __riscv_vmseq_vx_u32m1_b32(min_u32m1, min_val, vlmax);
    vuint64m2_t sentinel_u64m2 = __riscv_vmv_v_x_u64m2(NK_U64_MAX, vlmax);
    vuint64m2_t min_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, min_indices_u64m2, min_match_b32, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(min_cands_u64m2, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint32m1_t init_min_u32m1 = __riscv_vmv_v_x_u32m1(NK_U32_MIN, 1);
    nk_u32_t max_val = __riscv_vmv_x_s_u32m1_u32(__riscv_vredmaxu_vs_u32m1_u32m1(max_u32m1, init_min_u32m1, vlmax));
    vbool32_t max_match_b32 = __riscv_vmseq_vx_u32m1_b32(max_u32m1, max_val, vlmax);
    vuint64m2_t max_cands_u64m2 = __riscv_vmerge_vvm_u64m2(sentinel_u64m2, max_indices_u64m2, max_match_b32, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m2_u64m1(max_cands_u64m2, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_u32_rvv(                               //
    nk_u32_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u32_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u32_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u32_t);
    int aligned = (stride_bytes % sizeof(nk_u32_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_U32_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_U32_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u32_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_u32_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_u32_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_i64_rvv_contiguous_( //
    nk_i64_t const *data_ptr, nk_size_t count,          //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    // 128-bit per-lane accumulator for sum: (sum_upper, sum_lower)
    vuint64m1_t sum_lower_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vint64m1_t sum_upper_i64m1 = __riscv_vmv_v_x_i64m1(0, vlmax);
    vuint64m1_t sumsq_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count);
        vint64m1_t data_i64m1 = __riscv_vle64_v_i64m1(data_ptr, vector_length);

        // 128-bit sum accumulation: wrapping add on lower half
        vuint64m1_t data_u64m1 = __riscv_vreinterpret_v_i64m1_u64m1(data_i64m1);
        vuint64m1_t sum_before_u64m1 = sum_lower_u64m1;
        sum_lower_u64m1 = __riscv_vadd_vv_u64m1_tu(sum_lower_u64m1, sum_lower_u64m1, data_u64m1, vector_length);

        // Carry: new < old means unsigned overflow occurred
        vbool64_t carry_b64 = __riscv_vmsltu_vv_u64m1_b64(sum_lower_u64m1, sum_before_u64m1, vector_length);
        vint64m1_t carry_i64m1 = __riscv_vmerge_vxm_i64m1(__riscv_vmv_v_x_i64m1(0, vector_length), 1, carry_b64,
                                                          vector_length);
        sum_upper_i64m1 = __riscv_vadd_vv_i64m1_tu(sum_upper_i64m1, sum_upper_i64m1, carry_i64m1, vector_length);

        // Sign extension: -1 for negative, 0 for non-negative
        vint64m1_t sign_ext_i64m1 = __riscv_vsra_vx_i64m1(data_i64m1, 63, vector_length);
        sum_upper_i64m1 = __riscv_vadd_vv_i64m1_tu(sum_upper_i64m1, sum_upper_i64m1, sign_ext_i64m1, vector_length);

        // Sumsq: abs(val)² with overflow detection
        vint64m1_t negated_i64m1 = __riscv_vneg_v_i64m1(data_i64m1, vector_length);
        vint64m1_t absolute_i64m1 = __riscv_vmax_vv_i64m1(data_i64m1, negated_i64m1, vector_length);
        vuint64m1_t absolute_u64m1 = __riscv_vreinterpret_v_i64m1_u64m1(absolute_i64m1);
        vuint64m1_t product_low_u64m1 = __riscv_vmul_vv_u64m1(absolute_u64m1, absolute_u64m1, vector_length);
        vuint64m1_t product_high_u64m1 = __riscv_vmulhu_vv_u64m1(absolute_u64m1, absolute_u64m1, vector_length);
        vbool64_t overflow_b64 = __riscv_vmsne_vx_u64m1_b64(product_high_u64m1, 0, vector_length);
        vuint64m1_t squares_u64m1 = __riscv_vmerge_vxm_u64m1(product_low_u64m1, NK_U64_MAX, overflow_b64,
                                                             vector_length);
        sumsq_u64m1 = __riscv_vsaddu_vv_u64m1_tu(sumsq_u64m1, sumsq_u64m1, squares_u64m1, vector_length);
    }

    *sum_ptr = nk_reduce_128bit_sum_i64m1_rvv_(sum_lower_u64m1, sum_upper_i64m1, vlmax);
    *sumsq_ptr = nk_reduce_vsaddu_u64m1_rvv_(sumsq_u64m1, vlmax);
}

NK_INTERNAL void nk_reduce_moments_i64_rvv_strided_(                   //
    nk_i64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    // 128-bit per-lane accumulator for sum: (sum_upper, sum_lower)
    vuint64m1_t sum_lower_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vint64m1_t sum_upper_i64m1 = __riscv_vmv_v_x_i64m1(0, vlmax);
    vuint64m1_t sumsq_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(count);
        vint64m1_t data_i64m1 = __riscv_vlse64_v_i64m1((nk_i64_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // 128-bit sum accumulation: wrapping add on lower half
        vuint64m1_t data_u64m1 = __riscv_vreinterpret_v_i64m1_u64m1(data_i64m1);
        vuint64m1_t sum_before_u64m1 = sum_lower_u64m1;
        sum_lower_u64m1 = __riscv_vadd_vv_u64m1_tu(sum_lower_u64m1, sum_lower_u64m1, data_u64m1, vector_length);

        // Carry: new < old means unsigned overflow occurred
        vbool64_t carry_b64 = __riscv_vmsltu_vv_u64m1_b64(sum_lower_u64m1, sum_before_u64m1, vector_length);
        vint64m1_t carry_i64m1 = __riscv_vmerge_vxm_i64m1(__riscv_vmv_v_x_i64m1(0, vector_length), 1, carry_b64,
                                                          vector_length);
        sum_upper_i64m1 = __riscv_vadd_vv_i64m1_tu(sum_upper_i64m1, sum_upper_i64m1, carry_i64m1, vector_length);

        // Sign extension: -1 for negative, 0 for non-negative
        vint64m1_t sign_ext_i64m1 = __riscv_vsra_vx_i64m1(data_i64m1, 63, vector_length);
        sum_upper_i64m1 = __riscv_vadd_vv_i64m1_tu(sum_upper_i64m1, sum_upper_i64m1, sign_ext_i64m1, vector_length);

        // Sumsq: abs(val)² with overflow detection
        vint64m1_t negated_i64m1 = __riscv_vneg_v_i64m1(data_i64m1, vector_length);
        vint64m1_t absolute_i64m1 = __riscv_vmax_vv_i64m1(data_i64m1, negated_i64m1, vector_length);
        vuint64m1_t absolute_u64m1 = __riscv_vreinterpret_v_i64m1_u64m1(absolute_i64m1);
        vuint64m1_t product_low_u64m1 = __riscv_vmul_vv_u64m1(absolute_u64m1, absolute_u64m1, vector_length);
        vuint64m1_t product_high_u64m1 = __riscv_vmulhu_vv_u64m1(absolute_u64m1, absolute_u64m1, vector_length);
        vbool64_t overflow_b64 = __riscv_vmsne_vx_u64m1_b64(product_high_u64m1, 0, vector_length);
        vuint64m1_t squares_u64m1 = __riscv_vmerge_vxm_u64m1(product_low_u64m1, NK_U64_MAX, overflow_b64,
                                                             vector_length);
        sumsq_u64m1 = __riscv_vsaddu_vv_u64m1_tu(sumsq_u64m1, sumsq_u64m1, squares_u64m1, vector_length);
    }

    *sum_ptr = nk_reduce_128bit_sum_i64m1_rvv_(sum_lower_u64m1, sum_upper_i64m1, vlmax);
    *sumsq_ptr = nk_reduce_vsaddu_u64m1_rvv_(sumsq_u64m1, vlmax);
}

NK_PUBLIC void nk_reduce_moments_i64_rvv(                              //
    nk_i64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_i64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_i64_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_i64_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_i64_rvv_contiguous_( //
    nk_i64_t const *data_ptr, nk_size_t count,         //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vint64m1_t min_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MAX, vlmax);
    vint64m1_t max_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MIN, vlmax);
    vuint64m1_t min_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t max_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count);
        vint64m1_t data_i64m1 = __riscv_vle64_v_i64m1(data_ptr, vector_length);
        vuint64m1_t pos_u64m1 = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool64_t less_b64 = __riscv_vmslt_vv_i64m1_b64(data_i64m1, min_i64m1, vector_length);
        min_i64m1 = __riscv_vmerge_vvm_i64m1_tu(min_i64m1, min_i64m1, data_i64m1, less_b64, vector_length);
        min_indices_u64m1 = __riscv_vmerge_vvm_u64m1_tu(min_indices_u64m1, min_indices_u64m1, pos_u64m1, less_b64,
                                                        vector_length);

        vbool64_t greater_b64 = __riscv_vmslt_vv_i64m1_b64(max_i64m1, data_i64m1, vector_length);
        max_i64m1 = __riscv_vmerge_vvm_i64m1_tu(max_i64m1, max_i64m1, data_i64m1, greater_b64, vector_length);
        max_indices_u64m1 = __riscv_vmerge_vvm_u64m1_tu(max_indices_u64m1, max_indices_u64m1, pos_u64m1, greater_b64,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vint64m1_t init_max_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MAX, 1);
    nk_i64_t min_val = __riscv_vmv_x_s_i64m1_i64(__riscv_vredmin_vs_i64m1_i64m1(min_i64m1, init_max_i64m1, vlmax));
    vbool64_t min_match_b64 = __riscv_vmseq_vx_i64m1_b64(min_i64m1, min_val, vlmax);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t min_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, min_indices_u64m1, min_match_b64, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(min_cands_u64m1, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint64m1_t init_min_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MIN, 1);
    nk_i64_t max_val = __riscv_vmv_x_s_i64m1_i64(__riscv_vredmax_vs_i64m1_i64m1(max_i64m1, init_min_i64m1, vlmax));
    vbool64_t max_match_b64 = __riscv_vmseq_vx_i64m1_b64(max_i64m1, max_val, vlmax);
    vuint64m1_t max_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, max_indices_u64m1, max_match_b64, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(max_cands_u64m1, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_i64_rvv_strided_(                    //
    nk_i64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vint64m1_t min_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MAX, vlmax);
    vint64m1_t max_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MIN, vlmax);
    vuint64m1_t min_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t max_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(count);
        vint64m1_t data_i64m1 = __riscv_vlse64_v_i64m1((nk_i64_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint64m1_t pos_u64m1 = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool64_t less_b64 = __riscv_vmslt_vv_i64m1_b64(data_i64m1, min_i64m1, vector_length);
        min_i64m1 = __riscv_vmerge_vvm_i64m1_tu(min_i64m1, min_i64m1, data_i64m1, less_b64, vector_length);
        min_indices_u64m1 = __riscv_vmerge_vvm_u64m1_tu(min_indices_u64m1, min_indices_u64m1, pos_u64m1, less_b64,
                                                        vector_length);

        vbool64_t greater_b64 = __riscv_vmslt_vv_i64m1_b64(max_i64m1, data_i64m1, vector_length);
        max_i64m1 = __riscv_vmerge_vvm_i64m1_tu(max_i64m1, max_i64m1, data_i64m1, greater_b64, vector_length);
        max_indices_u64m1 = __riscv_vmerge_vvm_u64m1_tu(max_indices_u64m1, max_indices_u64m1, pos_u64m1, greater_b64,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vint64m1_t init_max_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MAX, 1);
    nk_i64_t min_val = __riscv_vmv_x_s_i64m1_i64(__riscv_vredmin_vs_i64m1_i64m1(min_i64m1, init_max_i64m1, vlmax));
    vbool64_t min_match_b64 = __riscv_vmseq_vx_i64m1_b64(min_i64m1, min_val, vlmax);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t min_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, min_indices_u64m1, min_match_b64, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(min_cands_u64m1, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vint64m1_t init_min_i64m1 = __riscv_vmv_v_x_i64m1(NK_I64_MIN, 1);
    nk_i64_t max_val = __riscv_vmv_x_s_i64m1_i64(__riscv_vredmax_vs_i64m1_i64m1(max_i64m1, init_min_i64m1, vlmax));
    vbool64_t max_match_b64 = __riscv_vmseq_vx_i64m1_b64(max_i64m1, max_val, vlmax);
    vuint64m1_t max_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, max_indices_u64m1, max_match_b64, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(max_cands_u64m1, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_i64_rvv(                               //
    nk_i64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_i64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_i64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_i64_t);
    int aligned = (stride_bytes % sizeof(nk_i64_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_I64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_I64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_i64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_i64_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_i64_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_u64_rvv_contiguous_( //
    nk_u64_t const *data_ptr, nk_size_t count,          //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vuint64m1_t sum_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t sumsq_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count);
        vuint64m1_t data_u64m1 = __riscv_vle64_v_u64m1(data_ptr, vector_length);

        // Saturating unsigned sum
        sum_u64m1 = __riscv_vsaddu_vv_u64m1_tu(sum_u64m1, sum_u64m1, data_u64m1, vector_length);

        // Sumsq: u64 × u64 with overflow detection via vmul + vmulhu
        vuint64m1_t product_low_u64m1 = __riscv_vmul_vv_u64m1(data_u64m1, data_u64m1, vector_length);
        vuint64m1_t product_high_u64m1 = __riscv_vmulhu_vv_u64m1(data_u64m1, data_u64m1, vector_length);
        vbool64_t overflow_b64 = __riscv_vmsne_vx_u64m1_b64(product_high_u64m1, 0, vector_length);
        vuint64m1_t squares_u64m1 = __riscv_vmerge_vxm_u64m1(product_low_u64m1, NK_U64_MAX, overflow_b64,
                                                             vector_length);
        sumsq_u64m1 = __riscv_vsaddu_vv_u64m1_tu(sumsq_u64m1, sumsq_u64m1, squares_u64m1, vector_length);
    }

    *sum_ptr = nk_reduce_vsaddu_u64m1_rvv_(sum_u64m1, vlmax);
    *sumsq_ptr = nk_reduce_vsaddu_u64m1_rvv_(sumsq_u64m1, vlmax);
}

NK_INTERNAL void nk_reduce_moments_u64_rvv_strided_(                   //
    nk_u64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vuint64m1_t sum_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t sumsq_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(count);
        vuint64m1_t data_u64m1 = __riscv_vlse64_v_u64m1((nk_u64_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // Saturating unsigned sum
        sum_u64m1 = __riscv_vsaddu_vv_u64m1_tu(sum_u64m1, sum_u64m1, data_u64m1, vector_length);

        // Sumsq: u64 × u64 with overflow detection via vmul + vmulhu
        vuint64m1_t product_low_u64m1 = __riscv_vmul_vv_u64m1(data_u64m1, data_u64m1, vector_length);
        vuint64m1_t product_high_u64m1 = __riscv_vmulhu_vv_u64m1(data_u64m1, data_u64m1, vector_length);
        vbool64_t overflow_b64 = __riscv_vmsne_vx_u64m1_b64(product_high_u64m1, 0, vector_length);
        vuint64m1_t squares_u64m1 = __riscv_vmerge_vxm_u64m1(product_low_u64m1, NK_U64_MAX, overflow_b64,
                                                             vector_length);
        sumsq_u64m1 = __riscv_vsaddu_vv_u64m1_tu(sumsq_u64m1, sumsq_u64m1, squares_u64m1, vector_length);
    }

    *sum_ptr = nk_reduce_vsaddu_u64m1_rvv_(sum_u64m1, vlmax);
    *sumsq_ptr = nk_reduce_vsaddu_u64m1_rvv_(sumsq_u64m1, vlmax);
}

NK_PUBLIC void nk_reduce_moments_u64_rvv(                              //
    nk_u64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *sum_ptr, nk_u64_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);

    if (count == 0) *sum_ptr = 0, *sumsq_ptr = 0;
    else if (!aligned) { nk_reduce_moments_u64_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
    else if (stride_elements == 1) { nk_reduce_moments_u64_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr); }
    else { nk_reduce_moments_u64_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr); }
}

NK_INTERNAL void nk_reduce_minmax_u64_rvv_contiguous_( //
    nk_u64_t const *data_ptr, nk_size_t count,         //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vuint64m1_t min_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t max_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MIN, vlmax);
    vuint64m1_t min_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t max_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count);
        vuint64m1_t data_u64m1 = __riscv_vle64_v_u64m1(data_ptr, vector_length);
        vuint64m1_t pos_u64m1 = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool64_t less_b64 = __riscv_vmsltu_vv_u64m1_b64(data_u64m1, min_u64m1, vector_length);
        min_u64m1 = __riscv_vmerge_vvm_u64m1_tu(min_u64m1, min_u64m1, data_u64m1, less_b64, vector_length);
        min_indices_u64m1 = __riscv_vmerge_vvm_u64m1_tu(min_indices_u64m1, min_indices_u64m1, pos_u64m1, less_b64,
                                                        vector_length);

        vbool64_t greater_b64 = __riscv_vmsltu_vv_u64m1_b64(max_u64m1, data_u64m1, vector_length);
        max_u64m1 = __riscv_vmerge_vvm_u64m1_tu(max_u64m1, max_u64m1, data_u64m1, greater_b64, vector_length);
        max_indices_u64m1 = __riscv_vmerge_vvm_u64m1_tu(max_indices_u64m1, max_indices_u64m1, pos_u64m1, greater_b64,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vuint64m1_t init_max_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    nk_u64_t min_val = __riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(min_u64m1, init_max_u64m1, vlmax));
    vbool64_t min_match_b64 = __riscv_vmseq_vx_u64m1_b64(min_u64m1, min_val, vlmax);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t min_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, min_indices_u64m1, min_match_b64, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(min_cands_u64m1, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint64m1_t init_min_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MIN, 1);
    nk_u64_t max_val = __riscv_vmv_x_s_u64m1_u64(__riscv_vredmaxu_vs_u64m1_u64m1(max_u64m1, init_min_u64m1, vlmax));
    vbool64_t max_match_b64 = __riscv_vmseq_vx_u64m1_b64(max_u64m1, max_val, vlmax);
    vuint64m1_t max_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, max_indices_u64m1, max_match_b64, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(max_cands_u64m1, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_u64_rvv_strided_(                    //
    nk_u64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vuint64m1_t min_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t max_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MIN, vlmax);
    vuint64m1_t min_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    vuint64m1_t max_indices_u64m1 = __riscv_vmv_v_x_u64m1(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e64m1(count);
        vuint64m1_t data_u64m1 = __riscv_vlse64_v_u64m1((nk_u64_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint64m1_t pos_u64m1 = __riscv_vadd_vx_u64m1(__riscv_vid_v_u64m1(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool64_t less_b64 = __riscv_vmsltu_vv_u64m1_b64(data_u64m1, min_u64m1, vector_length);
        min_u64m1 = __riscv_vmerge_vvm_u64m1_tu(min_u64m1, min_u64m1, data_u64m1, less_b64, vector_length);
        min_indices_u64m1 = __riscv_vmerge_vvm_u64m1_tu(min_indices_u64m1, min_indices_u64m1, pos_u64m1, less_b64,
                                                        vector_length);

        vbool64_t greater_b64 = __riscv_vmsltu_vv_u64m1_b64(max_u64m1, data_u64m1, vector_length);
        max_u64m1 = __riscv_vmerge_vvm_u64m1_tu(max_u64m1, max_u64m1, data_u64m1, greater_b64, vector_length);
        max_indices_u64m1 = __riscv_vmerge_vvm_u64m1_tu(max_indices_u64m1, max_indices_u64m1, pos_u64m1, greater_b64,
                                                        vector_length);
    }

    // Horizontal reduction for min
    vuint64m1_t init_max_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    nk_u64_t min_val = __riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(min_u64m1, init_max_u64m1, vlmax));
    vbool64_t min_match_b64 = __riscv_vmseq_vx_u64m1_b64(min_u64m1, min_val, vlmax);
    vuint64m1_t sentinel_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, vlmax);
    vuint64m1_t min_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, min_indices_u64m1, min_match_b64, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_value_ptr = min_val;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(min_cands_u64m1, init_umax_u64m1, vlmax));

    // Horizontal reduction for max
    vuint64m1_t init_min_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MIN, 1);
    nk_u64_t max_val = __riscv_vmv_x_s_u64m1_u64(__riscv_vredmaxu_vs_u64m1_u64m1(max_u64m1, init_min_u64m1, vlmax));
    vbool64_t max_match_b64 = __riscv_vmseq_vx_u64m1_b64(max_u64m1, max_val, vlmax);
    vuint64m1_t max_cands_u64m1 = __riscv_vmerge_vvm_u64m1(sentinel_u64m1, max_indices_u64m1, max_match_b64, vlmax);
    *max_value_ptr = max_val;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m1_u64m1(max_cands_u64m1, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_u64_rvv(                               //
    nk_u64_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_u64_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_u64_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_u64_t);
    int aligned = (stride_bytes % sizeof(nk_u64_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_U64_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_U64_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_u64_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_u64_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_u64_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_bf16_rvv_contiguous_( //
    nk_bf16_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t sumsq_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1((uint16_t const *)data_ptr, vector_length);

        // Convert bf16 → f32 (m1 → m2)
        vfloat32m2_t data_f32m2 = nk_bf16m1_to_f32m2_rvv_(data_u16m1, vector_length);

        // Widen f32 → f64 (m2 → m4)
        vfloat64m4_t data_f64m4 = __riscv_vfwcvt_f_f_v_f64m4(data_f32m2, vector_length);
        sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(sum_f64m4, sum_f64m4, data_f64m4, vector_length);

        // Sumsq via widening FMA: f32×f32 → f64
        sumsq_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(sumsq_f64m4, data_f32m2, data_f32m2, vector_length);
    }

    // Horizontal reduction
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sumsq_f64m4, zero_f64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_bf16_rvv_strided_(                   //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t sumsq_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((uint16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // Convert bf16 → f32 (m1 → m2)
        vfloat32m2_t data_f32m2 = nk_bf16m1_to_f32m2_rvv_(data_u16m1, vector_length);

        // Widen f32 → f64 (m2 → m4)
        vfloat64m4_t data_f64m4 = __riscv_vfwcvt_f_f_v_f64m4(data_f32m2, vector_length);
        sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(sum_f64m4, sum_f64m4, data_f64m4, vector_length);

        // Sumsq via widening FMA: f32×f32 → f64
        sumsq_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(sumsq_f64m4, data_f32m2, data_f32m2, vector_length);
    }

    // Horizontal reduction
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sumsq_f64m4, zero_f64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_bf16_rvv(                              //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);

    if (count == 0) *sum_ptr = 0.0f, *sumsq_ptr = 0.0f;
    else if (!aligned) nk_reduce_moments_bf16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_bf16_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_bf16_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_bf16_rvv_contiguous_( //
    nk_bf16_t const *data_ptr, nk_size_t count,         //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vuint16m1_t min_u16m1 = __riscv_vmv_v_x_u16m1(0x7F80, vlmax); // +inf in bf16
    vuint16m1_t max_u16m1 = __riscv_vmv_v_x_u16m1(0xFF80, vlmax); // -inf in bf16
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1((uint16_t const *)data_ptr, vector_length);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        // Convert to f32 for comparison
        vfloat32m2_t data_f32m2 = nk_bf16m1_to_f32m2_rvv_(data_u16m1, vector_length);
        vfloat32m2_t min_f32m2 = nk_bf16m1_to_f32m2_rvv_(min_u16m1, vector_length);
        vfloat32m2_t max_f32m2 = nk_bf16m1_to_f32m2_rvv_(max_u16m1, vector_length);

        vbool16_t less_b16 = __riscv_vmflt_vv_f32m2_b16(data_f32m2, min_f32m2, vector_length);
        min_u16m1 = __riscv_vmerge_vvm_u16m1_tu(min_u16m1, min_u16m1, data_u16m1, less_b16, vector_length);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(min_indices_u64m4, min_indices_u64m4, pos_u64m4, less_b16,
                                                        vector_length);

        vbool16_t greater_b16 = __riscv_vmflt_vv_f32m2_b16(max_f32m2, data_f32m2, vector_length);
        max_u16m1 = __riscv_vmerge_vvm_u16m1_tu(max_u16m1, max_u16m1, data_u16m1, greater_b16, vector_length);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(max_indices_u64m4, max_indices_u64m4, pos_u64m4, greater_b16,
                                                        vector_length);
    }

    // Horizontal reduction
    vfloat32m2_t final_min_f32m2 = nk_bf16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vfloat32m1_t init_max_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    nk_f32_t min_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m2_f32m1(final_min_f32m2, init_max_f32m1, vlmax));
    vfloat32m2_t final_max_f32m2 = nk_bf16m1_to_f32m2_rvv_(max_u16m1, vlmax);
    vfloat32m1_t init_min_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    nk_f32_t max_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m2_f32m1(final_max_f32m2, init_min_f32m1, vlmax));
    if (min_val_f32 == NK_F32_MAX && max_val_f32 == NK_F32_MIN) {
        *min_value_ptr = NK_BF16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_BF16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    vfloat32m2_t converted_f32m2 = nk_bf16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vbool16_t min_match_b16 = __riscv_vmfeq_vf_f32m2_b16(converted_f32m2, min_val_f32, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);

    nk_u16_t min_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(min_u16m1, (nk_size_t)__riscv_vfirst_m_b16(min_match_b16, vlmax), vlmax));
    *min_value_ptr = *(nk_bf16_t *)&min_raw;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    vbool16_t max_match_b16 = __riscv_vmfeq_vf_f32m2_b16(nk_bf16m1_to_f32m2_rvv_(max_u16m1, vlmax), max_val_f32, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);

    nk_u16_t max_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(max_u16m1, (nk_size_t)__riscv_vfirst_m_b16(max_match_b16, vlmax), vlmax));
    *max_value_ptr = *(nk_bf16_t *)&max_raw;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_bf16_rvv_strided_(                    //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vuint16m1_t min_u16m1 = __riscv_vmv_v_x_u16m1(0x7F80, vlmax); // +inf in bf16
    vuint16m1_t max_u16m1 = __riscv_vmv_v_x_u16m1(0xFF80, vlmax); // -inf in bf16
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((uint16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        // Convert to f32 for comparison
        vfloat32m2_t data_f32m2 = nk_bf16m1_to_f32m2_rvv_(data_u16m1, vector_length);
        vfloat32m2_t min_f32m2 = nk_bf16m1_to_f32m2_rvv_(min_u16m1, vector_length);
        vfloat32m2_t max_f32m2 = nk_bf16m1_to_f32m2_rvv_(max_u16m1, vector_length);

        vbool16_t less_b16 = __riscv_vmflt_vv_f32m2_b16(data_f32m2, min_f32m2, vector_length);
        min_u16m1 = __riscv_vmerge_vvm_u16m1_tu(min_u16m1, min_u16m1, data_u16m1, less_b16, vector_length);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(min_indices_u64m4, min_indices_u64m4, pos_u64m4, less_b16,
                                                        vector_length);

        vbool16_t greater_b16 = __riscv_vmflt_vv_f32m2_b16(max_f32m2, data_f32m2, vector_length);
        max_u16m1 = __riscv_vmerge_vvm_u16m1_tu(max_u16m1, max_u16m1, data_u16m1, greater_b16, vector_length);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(max_indices_u64m4, max_indices_u64m4, pos_u64m4, greater_b16,
                                                        vector_length);
    }

    // Horizontal reduction (same as contiguous)
    vfloat32m2_t final_min_f32m2 = nk_bf16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vfloat32m1_t init_max_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    nk_f32_t min_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m2_f32m1(final_min_f32m2, init_max_f32m1, vlmax));
    vfloat32m2_t final_max_f32m2 = nk_bf16m1_to_f32m2_rvv_(max_u16m1, vlmax);
    vfloat32m1_t init_min_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    nk_f32_t max_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m2_f32m1(final_max_f32m2, init_min_f32m1, vlmax));
    if (min_val_f32 == NK_F32_MAX && max_val_f32 == NK_F32_MIN) {
        *min_value_ptr = NK_BF16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_BF16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    vfloat32m2_t converted_f32m2 = nk_bf16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vbool16_t min_match_b16 = __riscv_vmfeq_vf_f32m2_b16(converted_f32m2, min_val_f32, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);

    nk_u16_t min_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(min_u16m1, (nk_size_t)__riscv_vfirst_m_b16(min_match_b16, vlmax), vlmax));
    *min_value_ptr = *(nk_bf16_t *)&min_raw;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    vbool16_t max_match_b16 = __riscv_vmfeq_vf_f32m2_b16(nk_bf16m1_to_f32m2_rvv_(max_u16m1, vlmax), max_val_f32, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);

    nk_u16_t max_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(max_u16m1, (nk_size_t)__riscv_vfirst_m_b16(max_match_b16, vlmax), vlmax));
    *max_value_ptr = *(nk_bf16_t *)&max_raw;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_bf16_rvv(                               //
    nk_bf16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_bf16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_bf16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_bf16_t);
    int aligned = (stride_bytes % sizeof(nk_bf16_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_BF16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_BF16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_bf16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_bf16_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_bf16_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                           max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_f16_rvv_contiguous_( //
    nk_f16_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t sumsq_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1((uint16_t const *)data_ptr, vector_length);

        // Convert f16 → f32 (m1 → m2)
        vfloat32m2_t data_f32m2 = nk_f16m1_to_f32m2_rvv_(data_u16m1, vector_length);

        // Widen f32 → f64 (m2 → m4)
        vfloat64m4_t data_f64m4 = __riscv_vfwcvt_f_f_v_f64m4(data_f32m2, vector_length);
        sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(sum_f64m4, sum_f64m4, data_f64m4, vector_length);

        // Sumsq via widening FMA: f32×f32 → f64
        sumsq_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(sumsq_f64m4, data_f32m2, data_f32m2, vector_length);
    }

    // Horizontal reduction
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sumsq_f64m4, zero_f64m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_f16_rvv_strided_(                   //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t sum_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    vfloat64m4_t sumsq_f64m4 = __riscv_vfmv_v_f_f64m4(0.0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((uint16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // Convert f16 → f32 (m1 → m2)
        vfloat32m2_t data_f32m2 = nk_f16m1_to_f32m2_rvv_(data_u16m1, vector_length);

        // Widen f32 → f64 (m2 → m4)
        vfloat64m4_t data_f64m4 = __riscv_vfwcvt_f_f_v_f64m4(data_f32m2, vector_length);
        sum_f64m4 = __riscv_vfadd_vv_f64m4_tu(sum_f64m4, sum_f64m4, data_f64m4, vector_length);

        // Sumsq via widening FMA: f32×f32 → f64
        sumsq_f64m4 = __riscv_vfwmacc_vv_f64m4_tu(sumsq_f64m4, data_f32m2, data_f32m2, vector_length);
    }

    // Horizontal reduction
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    *sum_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sum_f64m4, zero_f64m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(sumsq_f64m4, zero_f64m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_f16_rvv(                              //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);

    if (count == 0) *sum_ptr = 0.0f, *sumsq_ptr = 0.0f;
    else if (!aligned) nk_reduce_moments_f16_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_f16_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_f16_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_f16_rvv_contiguous_( //
    nk_f16_t const *data_ptr, nk_size_t count,         //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vuint16m1_t min_u16m1 = __riscv_vmv_v_x_u16m1(0x7C00, vlmax); // +inf in f16
    vuint16m1_t max_u16m1 = __riscv_vmv_v_x_u16m1(0xFC00, vlmax); // -inf in f16
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vle16_v_u16m1((uint16_t const *)data_ptr, vector_length);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        // Convert to f32 for comparison
        vfloat32m2_t data_f32m2 = nk_f16m1_to_f32m2_rvv_(data_u16m1, vector_length);
        vfloat32m2_t min_f32m2 = nk_f16m1_to_f32m2_rvv_(min_u16m1, vector_length);
        vfloat32m2_t max_f32m2 = nk_f16m1_to_f32m2_rvv_(max_u16m1, vector_length);

        vbool16_t less_b16 = __riscv_vmflt_vv_f32m2_b16(data_f32m2, min_f32m2, vector_length);
        min_u16m1 = __riscv_vmerge_vvm_u16m1_tu(min_u16m1, min_u16m1, data_u16m1, less_b16, vector_length);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(min_indices_u64m4, min_indices_u64m4, pos_u64m4, less_b16,
                                                        vector_length);

        vbool16_t greater_b16 = __riscv_vmflt_vv_f32m2_b16(max_f32m2, data_f32m2, vector_length);
        max_u16m1 = __riscv_vmerge_vvm_u16m1_tu(max_u16m1, max_u16m1, data_u16m1, greater_b16, vector_length);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(max_indices_u64m4, max_indices_u64m4, pos_u64m4, greater_b16,
                                                        vector_length);
    }

    // Horizontal reduction
    vfloat32m2_t final_min_f32m2 = nk_f16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vfloat32m1_t init_max_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    nk_f32_t min_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m2_f32m1(final_min_f32m2, init_max_f32m1, vlmax));
    vfloat32m2_t final_max_f32m2 = nk_f16m1_to_f32m2_rvv_(max_u16m1, vlmax);
    vfloat32m1_t init_min_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    nk_f32_t max_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m2_f32m1(final_max_f32m2, init_min_f32m1, vlmax));
    if (min_val_f32 == NK_F32_MAX && max_val_f32 == NK_F32_MIN) {
        *min_value_ptr = NK_F16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    vfloat32m2_t converted_f32m2 = nk_f16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vbool16_t min_match_b16 = __riscv_vmfeq_vf_f32m2_b16(converted_f32m2, min_val_f32, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);

    nk_u16_t min_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(min_u16m1, (nk_size_t)__riscv_vfirst_m_b16(min_match_b16, vlmax), vlmax));
    *min_value_ptr = *(nk_f16_t *)&min_raw;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    vbool16_t max_match_b16 = __riscv_vmfeq_vf_f32m2_b16(nk_f16m1_to_f32m2_rvv_(max_u16m1, vlmax), max_val_f32, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);

    nk_u16_t max_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(max_u16m1, (nk_size_t)__riscv_vfirst_m_b16(max_match_b16, vlmax), vlmax));
    *max_value_ptr = *(nk_f16_t *)&max_raw;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_INTERNAL void nk_reduce_minmax_f16_rvv_strided_(                    //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e16m1();
    vuint16m1_t min_u16m1 = __riscv_vmv_v_x_u16m1(0x7C00, vlmax); // +inf in f16
    vuint16m1_t max_u16m1 = __riscv_vmv_v_x_u16m1(0xFC00, vlmax); // -inf in f16
    vuint64m4_t min_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    vuint64m4_t max_indices_u64m4 = __riscv_vmv_v_x_u64m4(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e16m1(count);
        vuint16m1_t data_u16m1 = __riscv_vlse16_v_u16m1((uint16_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);
        vuint64m4_t pos_u64m4 = __riscv_vadd_vx_u64m4(__riscv_vid_v_u64m4(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        // Convert to f32 for comparison
        vfloat32m2_t data_f32m2 = nk_f16m1_to_f32m2_rvv_(data_u16m1, vector_length);
        vfloat32m2_t min_f32m2 = nk_f16m1_to_f32m2_rvv_(min_u16m1, vector_length);
        vfloat32m2_t max_f32m2 = nk_f16m1_to_f32m2_rvv_(max_u16m1, vector_length);

        vbool16_t less_b16 = __riscv_vmflt_vv_f32m2_b16(data_f32m2, min_f32m2, vector_length);
        min_u16m1 = __riscv_vmerge_vvm_u16m1_tu(min_u16m1, min_u16m1, data_u16m1, less_b16, vector_length);
        min_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(min_indices_u64m4, min_indices_u64m4, pos_u64m4, less_b16,
                                                        vector_length);

        vbool16_t greater_b16 = __riscv_vmflt_vv_f32m2_b16(max_f32m2, data_f32m2, vector_length);
        max_u16m1 = __riscv_vmerge_vvm_u16m1_tu(max_u16m1, max_u16m1, data_u16m1, greater_b16, vector_length);
        max_indices_u64m4 = __riscv_vmerge_vvm_u64m4_tu(max_indices_u64m4, max_indices_u64m4, pos_u64m4, greater_b16,
                                                        vector_length);
    }

    // Horizontal reduction (same as contiguous)
    vfloat32m2_t final_min_f32m2 = nk_f16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vfloat32m1_t init_max_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MAX, 1);
    nk_f32_t min_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m2_f32m1(final_min_f32m2, init_max_f32m1, vlmax));
    vfloat32m2_t final_max_f32m2 = nk_f16m1_to_f32m2_rvv_(max_u16m1, vlmax);
    vfloat32m1_t init_min_f32m1 = __riscv_vfmv_v_f_f32m1(NK_F32_MIN, 1);
    nk_f32_t max_val_f32 = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m2_f32m1(final_max_f32m2, init_min_f32m1, vlmax));
    if (min_val_f32 == NK_F32_MAX && max_val_f32 == NK_F32_MIN) {
        *min_value_ptr = NK_F16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    vfloat32m2_t converted_f32m2 = nk_f16m1_to_f32m2_rvv_(min_u16m1, vlmax);
    vbool16_t min_match_b16 = __riscv_vmfeq_vf_f32m2_b16(converted_f32m2, min_val_f32, vlmax);
    vuint64m4_t sentinel_u64m4 = __riscv_vmv_v_x_u64m4(NK_U64_MAX, vlmax);
    vuint64m4_t min_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, min_indices_u64m4, min_match_b16, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);

    nk_u16_t min_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(min_u16m1, (nk_size_t)__riscv_vfirst_m_b16(min_match_b16, vlmax), vlmax));
    *min_value_ptr = *(nk_f16_t *)&min_raw;
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(min_cands_u64m4, init_umax_u64m1, vlmax));

    vbool16_t max_match_b16 = __riscv_vmfeq_vf_f32m2_b16(nk_f16m1_to_f32m2_rvv_(max_u16m1, vlmax), max_val_f32, vlmax);
    vuint64m4_t max_cands_u64m4 = __riscv_vmerge_vvm_u64m4(sentinel_u64m4, max_indices_u64m4, max_match_b16, vlmax);

    nk_u16_t max_raw = __riscv_vmv_x_s_u16m1_u16(
        __riscv_vslidedown_vx_u16m1(max_u16m1, (nk_size_t)__riscv_vfirst_m_b16(max_match_b16, vlmax), vlmax));
    *max_value_ptr = *(nk_f16_t *)&max_raw;
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m4_u64m1(max_cands_u64m4, init_umax_u64m1, vlmax));
}

NK_PUBLIC void nk_reduce_minmax_f16_rvv(                               //
    nk_f16_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f16_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_f16_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_f16_t);
    int aligned = (stride_bytes % sizeof(nk_f16_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_F16_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_F16_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_f16_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                    max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_f16_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                             max_index_ptr);
    else
        nk_reduce_minmax_f16_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                          max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e4m3_rvv_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vector_length);

        // Convert e4m3 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e4m3m1_to_f32m4_rvv_(data_u8m1, vector_length);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4_tu(sum_f32m4, sum_f32m4, data_f32m4, vector_length);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4_tu(sumsq_f32m4, data_f32m4, data_f32m4, vector_length);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_e4m3_rvv_strided_(                   //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // Convert e4m3 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e4m3m1_to_f32m4_rvv_(data_u8m1, vector_length);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4_tu(sum_f32m4, sum_f32m4, data_f32m4, vector_length);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4_tu(sumsq_f32m4, data_f32m4, data_f32m4, vector_length);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_e4m3_rvv(                              //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);

    if (count == 0) *sum_ptr = 0.0f, *sumsq_ptr = 0.0f;
    else if (!aligned) nk_reduce_moments_e4m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_e4m3_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e4m3_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e4m3_rvv_contiguous_( //
    nk_e4m3_t const *data_ptr, nk_size_t count,         //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, vlmax); // Largest comparable
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax); // Smallest comparable
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vector_length);

        // Convert to comparable form
        vuint8m1_t comparable_u8m1 = nk_fp8m1_to_comparable_u8m1_rvv_(raw_u8m1, vector_length);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        // Detect E4M3 NaN: comparable == 0x00 (neg NaN) or comparable == 0xFF (pos NaN)
        vbool8_t nan_low_b8 = __riscv_vmseq_vx_u8m1_b8(comparable_u8m1, 0x00, vector_length);
        vbool8_t nan_high_b8 = __riscv_vmseq_vx_u8m1_b8(comparable_u8m1, 0xFF, vector_length);
        vbool8_t is_nan_b8 = __riscv_vmor_mm_b8(nan_low_b8, nan_high_b8, vector_length);
        vuint8m1_t data_min_u8m1 = __riscv_vmerge_vxm_u8m1(comparable_u8m1, 0xFF, is_nan_b8, vector_length);
        vuint8m1_t data_max_u8m1 = __riscv_vmerge_vxm_u8m1(comparable_u8m1, 0x00, is_nan_b8, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(data_min_u8m1, min_u8m1, vector_length);
        min_u8m1 = __riscv_vmerge_vvm_u8m1_tu(min_u8m1, min_u8m1, data_min_u8m1, less_b8, vector_length);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(min_indices_u64m8, min_indices_u64m8, pos_u64m8, less_b8,
                                                        vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, data_max_u8m1, vector_length);
        max_u8m1 = __riscv_vmerge_vvm_u8m1_tu(max_u8m1, max_u8m1, data_max_u8m1, greater_b8, vector_length);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(max_indices_u64m8, max_indices_u64m8, pos_u64m8, greater_b8,
                                                        vector_length);
    }

    // Horizontal reduction + convert back
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));

    // All-NaN case
    if (min_comparable == 0xFF) {
        *min_value_ptr = (nk_e4m3_t)NK_E4M3_MAX, *min_index_ptr = NK_SIZE_MAX;
        *max_value_ptr = (nk_e4m3_t)NK_E4M3_MIN, *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e4m3_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    // Similar for max
    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e4m3_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_INTERNAL void nk_reduce_minmax_e4m3_rvv_strided_(                    //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vuint8m1_t comparable_u8m1 = nk_fp8m1_to_comparable_u8m1_rvv_(raw_u8m1, vector_length);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        // Detect E4M3 NaN: comparable == 0x00 (neg NaN) or comparable == 0xFF (pos NaN)
        vbool8_t nan_low_b8 = __riscv_vmseq_vx_u8m1_b8(comparable_u8m1, 0x00, vector_length);
        vbool8_t nan_high_b8 = __riscv_vmseq_vx_u8m1_b8(comparable_u8m1, 0xFF, vector_length);
        vbool8_t is_nan_b8 = __riscv_vmor_mm_b8(nan_low_b8, nan_high_b8, vector_length);
        vuint8m1_t data_min_u8m1 = __riscv_vmerge_vxm_u8m1(comparable_u8m1, 0xFF, is_nan_b8, vector_length);
        vuint8m1_t data_max_u8m1 = __riscv_vmerge_vxm_u8m1(comparable_u8m1, 0x00, is_nan_b8, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(data_min_u8m1, min_u8m1, vector_length);
        min_u8m1 = __riscv_vmerge_vvm_u8m1_tu(min_u8m1, min_u8m1, data_min_u8m1, less_b8, vector_length);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(min_indices_u64m8, min_indices_u64m8, pos_u64m8, less_b8,
                                                        vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, data_max_u8m1, vector_length);
        max_u8m1 = __riscv_vmerge_vvm_u8m1_tu(max_u8m1, max_u8m1, data_max_u8m1, greater_b8, vector_length);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(max_indices_u64m8, max_indices_u64m8, pos_u64m8, greater_b8,
                                                        vector_length);
    }

    // Horizontal reduction (same as contiguous)
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));

    // All-NaN case
    if (min_comparable == 0xFF) {
        *min_value_ptr = (nk_e4m3_t)NK_E4M3_MAX, *min_index_ptr = NK_SIZE_MAX;
        *max_value_ptr = (nk_e4m3_t)NK_E4M3_MIN, *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e4m3_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e4m3_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_PUBLIC void nk_reduce_minmax_e4m3_rvv(                               //
    nk_e4m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e4m3_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e4m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e4m3_t);
    int aligned = (stride_bytes % sizeof(nk_e4m3_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_E4M3_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E4M3_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e4m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_e4m3_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_e4m3_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                           max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e5m2_rvv_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vector_length);

        // Convert e5m2 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e5m2m1_to_f32m4_rvv_(data_u8m1, vector_length);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4_tu(sum_f32m4, sum_f32m4, data_f32m4, vector_length);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4_tu(sumsq_f32m4, data_f32m4, data_f32m4, vector_length);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_e5m2_rvv_strided_(                   //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // Convert e5m2 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e5m2m1_to_f32m4_rvv_(data_u8m1, vector_length);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4_tu(sum_f32m4, sum_f32m4, data_f32m4, vector_length);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4_tu(sumsq_f32m4, data_f32m4, data_f32m4, vector_length);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_e5m2_rvv(                              //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);

    if (count == 0) *sum_ptr = 0.0f, *sumsq_ptr = 0.0f;
    else if (!aligned) nk_reduce_moments_e5m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_e5m2_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e5m2_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e5m2_rvv_contiguous_( //
    nk_e5m2_t const *data_ptr, nk_size_t count,         //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vector_length);

        vuint8m1_t comparable_u8m1 = nk_fp8m1_to_comparable_u8m1_rvv_(raw_u8m1, vector_length);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        // Detect E5M2 NaN: comparable <= 0x02 (neg NaN) or comparable >= 0xFD (pos NaN)
        vbool8_t nan_low_b8 = __riscv_vmsleu_vx_u8m1_b8(comparable_u8m1, 0x02, vector_length);
        vbool8_t nan_high_b8 = __riscv_vmsgeu_vx_u8m1_b8(comparable_u8m1, 0xFD, vector_length);
        vbool8_t is_nan_b8 = __riscv_vmor_mm_b8(nan_low_b8, nan_high_b8, vector_length);
        vuint8m1_t data_min_u8m1 = __riscv_vmerge_vxm_u8m1(comparable_u8m1, 0xFF, is_nan_b8, vector_length);
        vuint8m1_t data_max_u8m1 = __riscv_vmerge_vxm_u8m1(comparable_u8m1, 0x00, is_nan_b8, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(data_min_u8m1, min_u8m1, vector_length);
        min_u8m1 = __riscv_vmerge_vvm_u8m1_tu(min_u8m1, min_u8m1, data_min_u8m1, less_b8, vector_length);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(min_indices_u64m8, min_indices_u64m8, pos_u64m8, less_b8,
                                                        vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, data_max_u8m1, vector_length);
        max_u8m1 = __riscv_vmerge_vvm_u8m1_tu(max_u8m1, max_u8m1, data_max_u8m1, greater_b8, vector_length);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(max_indices_u64m8, max_indices_u64m8, pos_u64m8, greater_b8,
                                                        vector_length);
    }

    // Horizontal reduction + convert back
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));

    // All-NaN case
    if (min_comparable == 0xFF) {
        *min_value_ptr = (nk_e5m2_t)NK_E5M2_MAX, *min_index_ptr = NK_SIZE_MAX;
        *max_value_ptr = (nk_e5m2_t)NK_E5M2_MIN, *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e5m2_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e5m2_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_INTERNAL void nk_reduce_minmax_e5m2_rvv_strided_(                    //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vuint8m1_t comparable_u8m1 = nk_fp8m1_to_comparable_u8m1_rvv_(raw_u8m1, vector_length);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        // Detect E5M2 NaN: comparable <= 0x02 (neg NaN) or comparable >= 0xFD (pos NaN)
        vbool8_t nan_low_b8 = __riscv_vmsleu_vx_u8m1_b8(comparable_u8m1, 0x02, vector_length);
        vbool8_t nan_high_b8 = __riscv_vmsgeu_vx_u8m1_b8(comparable_u8m1, 0xFD, vector_length);
        vbool8_t is_nan_b8 = __riscv_vmor_mm_b8(nan_low_b8, nan_high_b8, vector_length);
        vuint8m1_t data_min_u8m1 = __riscv_vmerge_vxm_u8m1(comparable_u8m1, 0xFF, is_nan_b8, vector_length);
        vuint8m1_t data_max_u8m1 = __riscv_vmerge_vxm_u8m1(comparable_u8m1, 0x00, is_nan_b8, vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(data_min_u8m1, min_u8m1, vector_length);
        min_u8m1 = __riscv_vmerge_vvm_u8m1_tu(min_u8m1, min_u8m1, data_min_u8m1, less_b8, vector_length);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(min_indices_u64m8, min_indices_u64m8, pos_u64m8, less_b8,
                                                        vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, data_max_u8m1, vector_length);
        max_u8m1 = __riscv_vmerge_vvm_u8m1_tu(max_u8m1, max_u8m1, data_max_u8m1, greater_b8, vector_length);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(max_indices_u64m8, max_indices_u64m8, pos_u64m8, greater_b8,
                                                        vector_length);
    }

    // Horizontal reduction (same as contiguous)
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0xFF, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));

    // All-NaN case
    if (min_comparable == 0xFF) {
        *min_value_ptr = (nk_e5m2_t)NK_E5M2_MAX, *min_index_ptr = NK_SIZE_MAX;
        *max_value_ptr = (nk_e5m2_t)NK_E5M2_MIN, *max_index_ptr = NK_SIZE_MAX;
        return;
    }

    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e5m2_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp8m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e5m2_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_PUBLIC void nk_reduce_minmax_e5m2_rvv(                               //
    nk_e5m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e5m2_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e5m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e5m2_t);
    int aligned = (stride_bytes % sizeof(nk_e5m2_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_E5M2_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E5M2_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e5m2_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_e5m2_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_e5m2_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                           max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e2m3_rvv_contiguous_( //
    nk_e2m3_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vector_length);

        // Convert e2m3 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e2m3m1_to_f32m4_rvv_(data_u8m1, vector_length);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4_tu(sum_f32m4, sum_f32m4, data_f32m4, vector_length);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4_tu(sumsq_f32m4, data_f32m4, data_f32m4, vector_length);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_e2m3_rvv_strided_(                   //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // Convert e2m3 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e2m3m1_to_f32m4_rvv_(data_u8m1, vector_length);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4_tu(sum_f32m4, sum_f32m4, data_f32m4, vector_length);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4_tu(sumsq_f32m4, data_f32m4, data_f32m4, vector_length);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_e2m3_rvv(                              //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);

    if (count == 0) *sum_ptr = 0.0f, *sumsq_ptr = 0.0f;
    else if (!aligned) nk_reduce_moments_e2m3_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_e2m3_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e2m3_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e2m3_rvv_contiguous_( //
    nk_e2m3_t const *data_ptr, nk_size_t count,         //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, vlmax); // Largest FP6 comparable
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax); // Smallest FP6 comparable
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vector_length);

        // Convert to FP6 comparable form
        vuint8m1_t comparable_u8m1 = nk_fp6m1_to_comparable_u8m1_rvv_(raw_u8m1, vector_length);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(comparable_u8m1, min_u8m1, vector_length);
        min_u8m1 = __riscv_vmerge_vvm_u8m1_tu(min_u8m1, min_u8m1, comparable_u8m1, less_b8, vector_length);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(min_indices_u64m8, min_indices_u64m8, pos_u64m8, less_b8,
                                                        vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, comparable_u8m1, vector_length);
        max_u8m1 = __riscv_vmerge_vvm_u8m1_tu(max_u8m1, max_u8m1, comparable_u8m1, greater_b8, vector_length);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(max_indices_u64m8, max_indices_u64m8, pos_u64m8, greater_b8,
                                                        vector_length);
    }

    // Horizontal reduction + convert back
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e2m3_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e2m3_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_INTERNAL void nk_reduce_minmax_e2m3_rvv_strided_(                    //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vuint8m1_t comparable_u8m1 = nk_fp6m1_to_comparable_u8m1_rvv_(raw_u8m1, vector_length);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(comparable_u8m1, min_u8m1, vector_length);
        min_u8m1 = __riscv_vmerge_vvm_u8m1_tu(min_u8m1, min_u8m1, comparable_u8m1, less_b8, vector_length);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(min_indices_u64m8, min_indices_u64m8, pos_u64m8, less_b8,
                                                        vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, comparable_u8m1, vector_length);
        max_u8m1 = __riscv_vmerge_vvm_u8m1_tu(max_u8m1, max_u8m1, comparable_u8m1, greater_b8, vector_length);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(max_indices_u64m8, max_indices_u64m8, pos_u64m8, greater_b8,
                                                        vector_length);
    }

    // Horizontal reduction (same as contiguous)
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e2m3_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e2m3_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_PUBLIC void nk_reduce_minmax_e2m3_rvv(                               //
    nk_e2m3_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e2m3_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e2m3_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e2m3_t);
    int aligned = (stride_bytes % sizeof(nk_e2m3_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_E2M3_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E2M3_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e2m3_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_e2m3_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_e2m3_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                           max_index_ptr);
}

NK_INTERNAL void nk_reduce_moments_e3m2_rvv_contiguous_( //
    nk_e3m2_t const *data_ptr, nk_size_t count,          //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    for (nk_size_t vector_length; count > 0; count -= vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vector_length);

        // Convert e3m2 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e3m2m1_to_f32m4_rvv_(data_u8m1, vector_length);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4_tu(sum_f32m4, sum_f32m4, data_f32m4, vector_length);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4_tu(sumsq_f32m4, data_f32m4, data_f32m4, vector_length);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_INTERNAL void nk_reduce_moments_e3m2_rvv_strided_(                   //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t sum_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    vfloat32m4_t sumsq_f32m4 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    for (nk_size_t vector_length; count > 0; count -= vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t data_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        // Convert e3m2 → f32 (m1 → m4)
        vfloat32m4_t data_f32m4 = nk_e3m2m1_to_f32m4_rvv_(data_u8m1, vector_length);

        // Accumulate at f32 precision
        sum_f32m4 = __riscv_vfadd_vv_f32m4_tu(sum_f32m4, sum_f32m4, data_f32m4, vector_length);
        sumsq_f32m4 = __riscv_vfmacc_vv_f32m4_tu(sumsq_f32m4, data_f32m4, data_f32m4, vector_length);
    }

    // Horizontal reduction
    vfloat32m1_t zero_f32m1 = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    *sum_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_f32m4, zero_f32m1, vlmax)),
    *sumsq_ptr = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumsq_f32m4, zero_f32m1, vlmax));
}

NK_PUBLIC void nk_reduce_moments_e3m2_rvv(                              //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_f32_t *sum_ptr, nk_f32_t *sumsq_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);

    if (count == 0) *sum_ptr = 0.0f, *sumsq_ptr = 0.0f;
    else if (!aligned) nk_reduce_moments_e3m2_serial(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
    else if (stride_elements == 1) nk_reduce_moments_e3m2_rvv_contiguous_(data_ptr, count, sum_ptr, sumsq_ptr);
    else nk_reduce_moments_e3m2_rvv_strided_(data_ptr, count, stride_bytes, sum_ptr, sumsq_ptr);
}

NK_INTERNAL void nk_reduce_minmax_e3m2_rvv_contiguous_( //
    nk_e3m2_t const *data_ptr, nk_size_t count,         //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr, //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, data_ptr += vector_length) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vle8_v_u8m1((uint8_t const *)data_ptr, vector_length);

        vuint8m1_t comparable_u8m1 = nk_fp6m1_to_comparable_u8m1_rvv_(raw_u8m1, vector_length);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(comparable_u8m1, min_u8m1, vector_length);
        min_u8m1 = __riscv_vmerge_vvm_u8m1_tu(min_u8m1, min_u8m1, comparable_u8m1, less_b8, vector_length);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(min_indices_u64m8, min_indices_u64m8, pos_u64m8, less_b8,
                                                        vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, comparable_u8m1, vector_length);
        max_u8m1 = __riscv_vmerge_vvm_u8m1_tu(max_u8m1, max_u8m1, comparable_u8m1, greater_b8, vector_length);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(max_indices_u64m8, max_indices_u64m8, pos_u64m8, greater_b8,
                                                        vector_length);
    }

    // Horizontal reduction + convert back
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e3m2_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e3m2_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_INTERNAL void nk_reduce_minmax_e3m2_rvv_strided_(                    //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t vlmax = __riscv_vsetvlmax_e8m1();
    vuint8m1_t min_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, vlmax);
    vuint8m1_t max_u8m1 = __riscv_vmv_v_x_u8m1(0x00, vlmax);
    vuint64m8_t min_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    vuint64m8_t max_indices_u64m8 = __riscv_vmv_v_x_u64m8(0, vlmax);
    unsigned char const *ptr = (unsigned char const *)data_ptr;

    nk_size_t offset = 0;
    for (nk_size_t vector_length; count > 0;
         count -= vector_length, offset += vector_length, ptr += vector_length * stride_bytes) {
        vector_length = __riscv_vsetvl_e8m1(count);
        vuint8m1_t raw_u8m1 = __riscv_vlse8_v_u8m1((uint8_t const *)ptr, (nk_ssize_t)stride_bytes, vector_length);

        vuint8m1_t comparable_u8m1 = nk_fp6m1_to_comparable_u8m1_rvv_(raw_u8m1, vector_length);
        vuint64m8_t pos_u64m8 = __riscv_vadd_vx_u64m8(__riscv_vid_v_u64m8(vector_length), (nk_u64_t)offset,
                                                      vector_length);

        vbool8_t less_b8 = __riscv_vmsltu_vv_u8m1_b8(comparable_u8m1, min_u8m1, vector_length);
        min_u8m1 = __riscv_vmerge_vvm_u8m1_tu(min_u8m1, min_u8m1, comparable_u8m1, less_b8, vector_length);
        min_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(min_indices_u64m8, min_indices_u64m8, pos_u64m8, less_b8,
                                                        vector_length);

        vbool8_t greater_b8 = __riscv_vmsltu_vv_u8m1_b8(max_u8m1, comparable_u8m1, vector_length);
        max_u8m1 = __riscv_vmerge_vvm_u8m1_tu(max_u8m1, max_u8m1, comparable_u8m1, greater_b8, vector_length);
        max_indices_u64m8 = __riscv_vmerge_vvm_u64m8_tu(max_indices_u64m8, max_indices_u64m8, pos_u64m8, greater_b8,
                                                        vector_length);
    }

    // Horizontal reduction (same as contiguous)
    vuint8m1_t init_max_u8m1 = __riscv_vmv_v_x_u8m1(0x3F, 1);
    nk_u8_t min_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(min_u8m1, init_max_u8m1, vlmax));
    vbool8_t min_match_b8 = __riscv_vmseq_vx_u8m1_b8(min_u8m1, min_comparable, vlmax);
    vuint64m8_t sentinel_u64m8 = __riscv_vmv_v_x_u64m8(NK_U64_MAX, vlmax);
    vuint64m8_t min_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, min_indices_u64m8, min_match_b8, vlmax);
    vuint64m1_t init_umax_u64m1 = __riscv_vmv_v_x_u64m1(NK_U64_MAX, 1);
    *min_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(min_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t min_vec_u8m1 = __riscv_vmv_v_x_u8m1(min_comparable, 1);
    vuint8m1_t min_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(min_vec_u8m1, 1);
    *min_value_ptr = (nk_e3m2_t)__riscv_vmv_x_s_u8m1_u8(min_raw_u8m1);

    vuint8m1_t init_min_u8m1 = __riscv_vmv_v_x_u8m1(0x00, 1);
    nk_u8_t max_comparable = __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(max_u8m1, init_min_u8m1, vlmax));
    vbool8_t max_match_b8 = __riscv_vmseq_vx_u8m1_b8(max_u8m1, max_comparable, vlmax);
    vuint64m8_t max_cands_u64m8 = __riscv_vmerge_vvm_u64m8(sentinel_u64m8, max_indices_u64m8, max_match_b8, vlmax);
    *max_index_ptr = (nk_size_t)__riscv_vmv_x_s_u64m1_u64(
        __riscv_vredminu_vs_u64m8_u64m1(max_cands_u64m8, init_umax_u64m1, vlmax));

    vuint8m1_t max_vec_u8m1 = __riscv_vmv_v_x_u8m1(max_comparable, 1);
    vuint8m1_t max_raw_u8m1 = nk_comparable_to_fp6m1_rvv_(max_vec_u8m1, 1);
    *max_value_ptr = (nk_e3m2_t)__riscv_vmv_x_s_u8m1_u8(max_raw_u8m1);
}

NK_PUBLIC void nk_reduce_minmax_e3m2_rvv(                               //
    nk_e3m2_t const *data_ptr, nk_size_t count, nk_size_t stride_bytes, //
    nk_e3m2_t *min_value_ptr, nk_size_t *min_index_ptr,                 //
    nk_e3m2_t *max_value_ptr, nk_size_t *max_index_ptr) {
    nk_size_t stride_elements = stride_bytes / sizeof(nk_e3m2_t);
    int aligned = (stride_bytes % sizeof(nk_e3m2_t) == 0);

    if (count == 0)
        *min_value_ptr = NK_E3M2_MAX, *min_index_ptr = NK_SIZE_MAX, *max_value_ptr = NK_E3M2_MIN,
        *max_index_ptr = NK_SIZE_MAX;
    else if (!aligned)
        nk_reduce_minmax_e3m2_serial(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                     max_index_ptr);
    else if (stride_elements == 1)
        nk_reduce_minmax_e3m2_rvv_contiguous_(data_ptr, count, min_value_ptr, min_index_ptr, max_value_ptr,
                                              max_index_ptr);
    else
        nk_reduce_minmax_e3m2_rvv_strided_(data_ptr, count, stride_bytes, min_value_ptr, min_index_ptr, max_value_ptr,
                                           max_index_ptr);
}

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
#endif // NK_REDUCE_RVV_H
