/**
 *  @file include/numkong/mesh/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief SIMD-accelerated mesh operations for RISC-V.
 *
 *  @sa include/numkong/mesh.h
 *
 *  RVV mesh operations leverage:
 *
 *  - @c vlseg3e32 and @c vlseg3e64: deinterleave xyz triplets in hardware
 *  - @c vfwcvt: widens f32 to f64 before any arithmetic, so f32 inputs go through f64 logic
 *  - @c vfredusum: single-instruction horizontal reduction, deferred until after the loop
 *  - Serial SVD/determinant from mesh/serial.h for fixed 3×3 matrix operations
 *
 *  Kabsch and Umeyama take two passes:
 *
 *  - `nk_centroid_and_cross_covariance_*_rvv_`: centroids and H (Kabsch), plus the centered ‖a‖²
 *    in the @c _and_variance_ variant (Umeyama)
 *  - `nk_transformed_ssd_*_rvv_`: Σ‖s · R · (aᵢ − ā) − (bᵢ − b̄)‖² from the residuals
 *
 *  The first pass shifts every point by the first one, then fixes up after the loop:
 *
 *    H[i][j] = Σ (a'[i] - mean_a[i]) * (b'[j] - mean_b[j])
 *            = Σ a'[i] * b'[j] - n * mean_a[i] * mean_b[j]
 *
 *  Unshifted raw moments cancel far from the origin, and folding the SSD through trace(R · H)
 *  cancels to √ε once the clouds align, hence the second pass.
 */
#ifndef NUMKONG_MESH_RVV_H
#define NUMKONG_MESH_RVV_H

#if NUMKONG_ARCH_RISCV64_
#if NUMKONG_ARCH_RISCV64_RVV_

#include "numkong/types.h"
#include "numkong/dot/rvv.h"
#include "numkong/mesh/serial.h" // `nk_svd3x3_f32_`, `nk_rmsd_f16_`, `nk_kabsch_bf16_`, `nk_umeyama_f16_`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NUMKONG_INLINE void nk_accumulate_sum_f64m1_rvv_(vfloat64m1_t *sum_f64m1, vfloat64m1_t *compensation_f64m1,
                                                 vfloat64m1_t addend_f64m1, nk_size_t vector_length) {
    vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(*sum_f64m1, addend_f64m1, vector_length);
    vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, *sum_f64m1, vector_length);
    vfloat64m1_t sum_error_f64m1 = __riscv_vfadd_vv_f64m1(
        __riscv_vfsub_vv_f64m1(*sum_f64m1,
                               __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vector_length),
                               vector_length),
        __riscv_vfsub_vv_f64m1(addend_f64m1, virtual_addend_f64m1, vector_length), vector_length);
    *sum_f64m1 = __riscv_vslideup_vx_f64m1_tu(*sum_f64m1, tentative_sum_f64m1, 0, vector_length);
    *compensation_f64m1 = __riscv_vfadd_vv_f64m1_tu(*compensation_f64m1, *compensation_f64m1, sum_error_f64m1,
                                                    vector_length);
}

NUMKONG_INLINE void nk_accumulate_product_f64m1_rvv_(vfloat64m1_t *sum_f64m1, vfloat64m1_t *compensation_f64m1,
                                                     vfloat64m1_t left_f64m1, vfloat64m1_t right_f64m1,
                                                     nk_size_t vector_length) {
    vfloat64m1_t product_f64m1 = __riscv_vfmul_vv_f64m1(left_f64m1, right_f64m1, vector_length);
    vfloat64m1_t product_error_f64m1 = __riscv_vfmsac_vv_f64m1(product_f64m1, left_f64m1, right_f64m1, vector_length);
    vfloat64m1_t tentative_sum_f64m1 = __riscv_vfadd_vv_f64m1(*sum_f64m1, product_f64m1, vector_length);
    vfloat64m1_t virtual_addend_f64m1 = __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, *sum_f64m1, vector_length);
    vfloat64m1_t sum_error_f64m1 = __riscv_vfadd_vv_f64m1(
        __riscv_vfsub_vv_f64m1(*sum_f64m1,
                               __riscv_vfsub_vv_f64m1(tentative_sum_f64m1, virtual_addend_f64m1, vector_length),
                               vector_length),
        __riscv_vfsub_vv_f64m1(product_f64m1, virtual_addend_f64m1, vector_length), vector_length);
    *sum_f64m1 = __riscv_vslideup_vx_f64m1_tu(*sum_f64m1, tentative_sum_f64m1, 0, vector_length);
    vfloat64m1_t total_error_f64m1 = __riscv_vfadd_vv_f64m1(sum_error_f64m1, product_error_f64m1, vector_length);
    *compensation_f64m1 = __riscv_vfadd_vv_f64m1_tu(*compensation_f64m1, *compensation_f64m1, total_error_f64m1,
                                                    vector_length);
}

/**
 *  @brief Compute centroids and cross-covariance matrix in a single pass (f32).
 *
 *  Accumulates Σ a[i]*b[j] and Σ a[i], Σ b[j] over the points shifted by the first one, then:
 *    ca = pivot + Σa / n,  cb = pivot + Σb / n
 *    H[i][j] = raw[i][j] - Σa[i] * Σb[j] / n
 *
 *  Reduces Kabsch from 4 passes to 2 (fused centroid+covariance + SSD).
 *  Cross-products use per-lane @c vfmacc_vv accumulation of the widened values (vfloat64m2_t) with
 *  deferred @c vfredusum after the loop — eliminates 9 reductions per iteration.
 */
NUMKONG_INLINE void nk_centroid_and_cross_covariance_f32_rvv_(              //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t points_count,           //
    nk_f64_t *centroid_a_x, nk_f64_t *centroid_a_y, nk_f64_t *centroid_a_z, //
    nk_f64_t *centroid_b_x, nk_f64_t *centroid_b_y, nk_f64_t *centroid_b_z, //
    nk_f64_t cross_covariance[9]) {
    nk_size_t max_vector_length = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_a_x_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length),
                 sum_a_y_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t sum_a_z_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t sum_b_x_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length),
                 sum_b_y_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t sum_b_z_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t cross_00_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length),
                 cross_01_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t cross_02_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length),
                 cross_10_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t cross_11_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length),
                 cross_12_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t cross_20_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length),
                 cross_21_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t cross_22_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    nk_f32_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = points_count;
    for (nk_size_t vector_length; remaining > 0;
         remaining -= vector_length, a_ptr += vector_length * 3, b_ptr += vector_length * 3) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1x3_t a_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(a_ptr, vector_length);
        vfloat64m2_t a_x_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 0), vector_length), a[0], vector_length);
        vfloat64m2_t a_y_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 1), vector_length), a[1], vector_length);
        vfloat64m2_t a_z_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 2), vector_length), a[2], vector_length);
        vfloat32m1x3_t b_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(b_ptr, vector_length);
        vfloat64m2_t b_x_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 0), vector_length), b[0], vector_length);
        vfloat64m2_t b_y_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 1), vector_length), b[1], vector_length);
        vfloat64m2_t b_z_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 2), vector_length), b[2], vector_length);
        sum_a_x_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_a_x_f64m2, sum_a_x_f64m2, a_x_f64m2, vector_length);
        sum_a_y_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_a_y_f64m2, sum_a_y_f64m2, a_y_f64m2, vector_length);
        sum_a_z_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_a_z_f64m2, sum_a_z_f64m2, a_z_f64m2, vector_length);
        sum_b_x_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_b_x_f64m2, sum_b_x_f64m2, b_x_f64m2, vector_length);
        sum_b_y_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_b_y_f64m2, sum_b_y_f64m2, b_y_f64m2, vector_length);
        sum_b_z_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_b_z_f64m2, sum_b_z_f64m2, b_z_f64m2, vector_length);
        cross_00_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_00_f64m2, a_x_f64m2, b_x_f64m2, vector_length);
        cross_01_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_01_f64m2, a_x_f64m2, b_y_f64m2, vector_length);
        cross_02_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_02_f64m2, a_x_f64m2, b_z_f64m2, vector_length);
        cross_10_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_10_f64m2, a_y_f64m2, b_x_f64m2, vector_length);
        cross_11_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_11_f64m2, a_y_f64m2, b_y_f64m2, vector_length);
        cross_12_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_12_f64m2, a_y_f64m2, b_z_f64m2, vector_length);
        cross_20_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_20_f64m2, a_z_f64m2, b_x_f64m2, vector_length);
        cross_21_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_21_f64m2, a_z_f64m2, b_y_f64m2, vector_length);
        cross_22_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_22_f64m2, a_z_f64m2, b_z_f64m2, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    // Compute centroids
    nk_f64_t inv_points_count = 1.0 / (nk_f64_t)points_count;
    nk_f64_t mean_a_x = __riscv_vfmv_f_s_f64m1_f64(
                            __riscv_vfredusum_vs_f64m2_f64m1(sum_a_x_f64m2, zero_f64m1, max_vector_length)) *
                        inv_points_count;
    nk_f64_t mean_a_y = __riscv_vfmv_f_s_f64m1_f64(
                            __riscv_vfredusum_vs_f64m2_f64m1(sum_a_y_f64m2, zero_f64m1, max_vector_length)) *
                        inv_points_count;
    nk_f64_t mean_a_z = __riscv_vfmv_f_s_f64m1_f64(
                            __riscv_vfredusum_vs_f64m2_f64m1(sum_a_z_f64m2, zero_f64m1, max_vector_length)) *
                        inv_points_count;
    nk_f64_t mean_b_x = __riscv_vfmv_f_s_f64m1_f64(
                            __riscv_vfredusum_vs_f64m2_f64m1(sum_b_x_f64m2, zero_f64m1, max_vector_length)) *
                        inv_points_count;
    nk_f64_t mean_b_y = __riscv_vfmv_f_s_f64m1_f64(
                            __riscv_vfredusum_vs_f64m2_f64m1(sum_b_y_f64m2, zero_f64m1, max_vector_length)) *
                        inv_points_count;
    nk_f64_t mean_b_z = __riscv_vfmv_f_s_f64m1_f64(
                            __riscv_vfredusum_vs_f64m2_f64m1(sum_b_z_f64m2, zero_f64m1, max_vector_length)) *
                        inv_points_count;
    *centroid_a_x = a[0] + mean_a_x;
    *centroid_a_y = a[1] + mean_a_y;
    *centroid_a_z = a[2] + mean_a_z;
    *centroid_b_x = b[0] + mean_b_x;
    *centroid_b_y = b[1] + mean_b_y;
    *centroid_b_z = b[2] + mean_b_z;
    // Fix up the sums shifted by the first point:
    // H[i][j] = raw[i][j] - points_count * mean_a[i] * mean_b[j]
    nk_f64_t n_f64 = (nk_f64_t)points_count;
    cross_covariance[0] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_00_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_x * mean_b_x;
    cross_covariance[1] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_01_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_x * mean_b_y;
    cross_covariance[2] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_02_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_x * mean_b_z;
    cross_covariance[3] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_10_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_y * mean_b_x;
    cross_covariance[4] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_11_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_y * mean_b_y;
    cross_covariance[5] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_12_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_y * mean_b_z;
    cross_covariance[6] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_20_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_z * mean_b_x;
    cross_covariance[7] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_21_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_z * mean_b_y;
    cross_covariance[8] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_22_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_z * mean_b_z;
}

/**
 *  @brief Compute centroids and cross-covariance matrix in a single pass (f64).
 *
 *  Per-lane @c vfadd_vv and @c vfmacc_vv accumulation with deferred @c vfredusum after the loop
 *  — eliminates 15 horizontal reductions per iteration.
 *  Sums run over the points shifted by the first one, so the centering correction doesn't cancel.
 */
NUMKONG_INLINE void nk_centroid_and_cross_covariance_f64_rvv_(              //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t points_count,           //
    nk_f64_t *centroid_a_x, nk_f64_t *centroid_a_y, nk_f64_t *centroid_a_z, //
    nk_f64_t *centroid_b_x, nk_f64_t *centroid_b_y, nk_f64_t *centroid_b_z, //
    nk_f64_t cross_covariance[9]) {
    nk_size_t max_vector_length = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length),
                 sum_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t sum_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t sum_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length),
                 sum_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t sum_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t cross_00_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length),
                 cross_01_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t cross_02_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length),
                 cross_10_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t cross_11_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length),
                 cross_12_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t cross_20_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length),
                 cross_21_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t cross_22_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_00_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_01_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_02_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_10_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_11_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_12_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_20_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_21_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_22_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    nk_f64_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = points_count;
    for (nk_size_t vector_length; remaining > 0;
         remaining -= vector_length, a_ptr += vector_length * 3, b_ptr += vector_length * 3) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1x3_t a_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(a_ptr, vector_length);
        vfloat64m1_t a_x_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 0), a[0],
                                                        vector_length);
        vfloat64m1_t a_y_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 1), a[1],
                                                        vector_length);
        vfloat64m1_t a_z_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 2), a[2],
                                                        vector_length);
        vfloat64m1x3_t b_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(b_ptr, vector_length);
        vfloat64m1_t b_x_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 0), b[0],
                                                        vector_length);
        vfloat64m1_t b_y_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 1), b[1],
                                                        vector_length);
        vfloat64m1_t b_z_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 2), b[2],
                                                        vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_a_x_f64m1, &compensation_a_x_f64m1, a_x_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_a_y_f64m1, &compensation_a_y_f64m1, a_y_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_a_z_f64m1, &compensation_a_z_f64m1, a_z_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_b_x_f64m1, &compensation_b_x_f64m1, b_x_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_b_y_f64m1, &compensation_b_y_f64m1, b_y_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_b_z_f64m1, &compensation_b_z_f64m1, b_z_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_00_f64m1, &compensation_00_f64m1, a_x_f64m1, b_x_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_01_f64m1, &compensation_01_f64m1, a_x_f64m1, b_y_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_02_f64m1, &compensation_02_f64m1, a_x_f64m1, b_z_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_10_f64m1, &compensation_10_f64m1, a_y_f64m1, b_x_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_11_f64m1, &compensation_11_f64m1, a_y_f64m1, b_y_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_12_f64m1, &compensation_12_f64m1, a_y_f64m1, b_z_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_20_f64m1, &compensation_20_f64m1, a_z_f64m1, b_x_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_21_f64m1, &compensation_21_f64m1, a_z_f64m1, b_y_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_22_f64m1, &compensation_22_f64m1, a_z_f64m1, b_z_f64m1, vector_length);
    }
    // Compute centroids.
    nk_f64_t inv_points_count = 1.0 / (nk_f64_t)points_count;
    nk_f64_t mean_a_x = nk_dot_stable_sum_f64m1_rvv_(sum_a_x_f64m1, compensation_a_x_f64m1) * inv_points_count;
    nk_f64_t mean_a_y = nk_dot_stable_sum_f64m1_rvv_(sum_a_y_f64m1, compensation_a_y_f64m1) * inv_points_count;
    nk_f64_t mean_a_z = nk_dot_stable_sum_f64m1_rvv_(sum_a_z_f64m1, compensation_a_z_f64m1) * inv_points_count;
    nk_f64_t mean_b_x = nk_dot_stable_sum_f64m1_rvv_(sum_b_x_f64m1, compensation_b_x_f64m1) * inv_points_count;
    nk_f64_t mean_b_y = nk_dot_stable_sum_f64m1_rvv_(sum_b_y_f64m1, compensation_b_y_f64m1) * inv_points_count;
    nk_f64_t mean_b_z = nk_dot_stable_sum_f64m1_rvv_(sum_b_z_f64m1, compensation_b_z_f64m1) * inv_points_count;
    *centroid_a_x = a[0] + mean_a_x;
    *centroid_a_y = a[1] + mean_a_y;
    *centroid_a_z = a[2] + mean_a_z;
    *centroid_b_x = b[0] + mean_b_x;
    *centroid_b_y = b[1] + mean_b_y;
    *centroid_b_z = b[2] + mean_b_z;
    nk_f64_t n_f64 = (nk_f64_t)points_count;
    cross_covariance[0] = nk_dot_stable_sum_f64m1_rvv_(cross_00_f64m1, compensation_00_f64m1) -
                          n_f64 * mean_a_x * mean_b_x;
    cross_covariance[1] = nk_dot_stable_sum_f64m1_rvv_(cross_01_f64m1, compensation_01_f64m1) -
                          n_f64 * mean_a_x * mean_b_y;
    cross_covariance[2] = nk_dot_stable_sum_f64m1_rvv_(cross_02_f64m1, compensation_02_f64m1) -
                          n_f64 * mean_a_x * mean_b_z;
    cross_covariance[3] = nk_dot_stable_sum_f64m1_rvv_(cross_10_f64m1, compensation_10_f64m1) -
                          n_f64 * mean_a_y * mean_b_x;
    cross_covariance[4] = nk_dot_stable_sum_f64m1_rvv_(cross_11_f64m1, compensation_11_f64m1) -
                          n_f64 * mean_a_y * mean_b_y;
    cross_covariance[5] = nk_dot_stable_sum_f64m1_rvv_(cross_12_f64m1, compensation_12_f64m1) -
                          n_f64 * mean_a_y * mean_b_z;
    cross_covariance[6] = nk_dot_stable_sum_f64m1_rvv_(cross_20_f64m1, compensation_20_f64m1) -
                          n_f64 * mean_a_z * mean_b_x;
    cross_covariance[7] = nk_dot_stable_sum_f64m1_rvv_(cross_21_f64m1, compensation_21_f64m1) -
                          n_f64 * mean_a_z * mean_b_y;
    cross_covariance[8] = nk_dot_stable_sum_f64m1_rvv_(cross_22_f64m1, compensation_22_f64m1) -
                          n_f64 * mean_a_z * mean_b_z;
}

/**
 *  @brief Compute centroids, cross-covariance, and the centered norm-squared of a (f32).
 *
 *  Same as centroid_and_cross_covariance but also outputs the Umeyama scale's denominator:
 *    centered_norm_squared_a = Σ ||a[i] - ca||² = Σ ||a[i] - pivot||² - n * ||ca - pivot||²
 *
 *  Cross-products use per-lane @c vfmacc_vv accumulation of the widened values (vfloat64m2_t) with
 *  deferred @c vfredusum after the loop — eliminates 9 reductions per iteration.
 */
NUMKONG_INLINE void nk_centroid_and_cross_covariance_and_variance_f32_rvv_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t points_count,           //
    nk_f64_t *centroid_a_x, nk_f64_t *centroid_a_y, nk_f64_t *centroid_a_z, //
    nk_f64_t *centroid_b_x, nk_f64_t *centroid_b_y, nk_f64_t *centroid_b_z, //
    nk_f64_t cross_covariance[9], nk_f64_t *centered_norm_squared_a) {
    nk_size_t max_vector_length = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_a_x_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length),
                 sum_a_y_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t sum_a_z_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t sum_b_x_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length),
                 sum_b_y_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t sum_b_z_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t cross_00_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length),
                 cross_01_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t cross_02_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length),
                 cross_10_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t cross_11_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length),
                 cross_12_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t cross_20_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length),
                 cross_21_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t cross_22_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    vfloat64m2_t norm_sq_a_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    nk_f32_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = points_count;
    for (nk_size_t vector_length; remaining > 0;
         remaining -= vector_length, a_ptr += vector_length * 3, b_ptr += vector_length * 3) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1x3_t a_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(a_ptr, vector_length);
        vfloat64m2_t a_x_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 0), vector_length), a[0], vector_length);
        vfloat64m2_t a_y_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 1), vector_length), a[1], vector_length);
        vfloat64m2_t a_z_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 2), vector_length), a[2], vector_length);
        vfloat32m1x3_t b_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(b_ptr, vector_length);
        vfloat64m2_t b_x_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 0), vector_length), b[0], vector_length);
        vfloat64m2_t b_y_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 1), vector_length), b[1], vector_length);
        vfloat64m2_t b_z_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 2), vector_length), b[2], vector_length);
        sum_a_x_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_a_x_f64m2, sum_a_x_f64m2, a_x_f64m2, vector_length);
        sum_a_y_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_a_y_f64m2, sum_a_y_f64m2, a_y_f64m2, vector_length);
        sum_a_z_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_a_z_f64m2, sum_a_z_f64m2, a_z_f64m2, vector_length);
        sum_b_x_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_b_x_f64m2, sum_b_x_f64m2, b_x_f64m2, vector_length);
        sum_b_y_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_b_y_f64m2, sum_b_y_f64m2, b_y_f64m2, vector_length);
        sum_b_z_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_b_z_f64m2, sum_b_z_f64m2, b_z_f64m2, vector_length);
        cross_00_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_00_f64m2, a_x_f64m2, b_x_f64m2, vector_length);
        cross_01_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_01_f64m2, a_x_f64m2, b_y_f64m2, vector_length);
        cross_02_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_02_f64m2, a_x_f64m2, b_z_f64m2, vector_length);
        cross_10_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_10_f64m2, a_y_f64m2, b_x_f64m2, vector_length);
        cross_11_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_11_f64m2, a_y_f64m2, b_y_f64m2, vector_length);
        cross_12_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_12_f64m2, a_y_f64m2, b_z_f64m2, vector_length);
        cross_20_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_20_f64m2, a_z_f64m2, b_x_f64m2, vector_length);
        cross_21_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_21_f64m2, a_z_f64m2, b_y_f64m2, vector_length);
        cross_22_f64m2 = __riscv_vfmacc_vv_f64m2_tu(cross_22_f64m2, a_z_f64m2, b_z_f64m2, vector_length);
        // Accumulate norm-squared for a (centering fixup applied after reduction).
        norm_sq_a_f64m2 = __riscv_vfmacc_vv_f64m2_tu(norm_sq_a_f64m2, a_x_f64m2, a_x_f64m2, vector_length);
        norm_sq_a_f64m2 = __riscv_vfmacc_vv_f64m2_tu(norm_sq_a_f64m2, a_y_f64m2, a_y_f64m2, vector_length);
        norm_sq_a_f64m2 = __riscv_vfmacc_vv_f64m2_tu(norm_sq_a_f64m2, a_z_f64m2, a_z_f64m2, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t inv_points_count = 1.0 / (nk_f64_t)points_count;
    nk_f64_t mean_a_x = __riscv_vfmv_f_s_f64m1_f64(
                            __riscv_vfredusum_vs_f64m2_f64m1(sum_a_x_f64m2, zero_f64m1, max_vector_length)) *
                        inv_points_count;
    nk_f64_t mean_a_y = __riscv_vfmv_f_s_f64m1_f64(
                            __riscv_vfredusum_vs_f64m2_f64m1(sum_a_y_f64m2, zero_f64m1, max_vector_length)) *
                        inv_points_count;
    nk_f64_t mean_a_z = __riscv_vfmv_f_s_f64m1_f64(
                            __riscv_vfredusum_vs_f64m2_f64m1(sum_a_z_f64m2, zero_f64m1, max_vector_length)) *
                        inv_points_count;
    nk_f64_t mean_b_x = __riscv_vfmv_f_s_f64m1_f64(
                            __riscv_vfredusum_vs_f64m2_f64m1(sum_b_x_f64m2, zero_f64m1, max_vector_length)) *
                        inv_points_count;
    nk_f64_t mean_b_y = __riscv_vfmv_f_s_f64m1_f64(
                            __riscv_vfredusum_vs_f64m2_f64m1(sum_b_y_f64m2, zero_f64m1, max_vector_length)) *
                        inv_points_count;
    nk_f64_t mean_b_z = __riscv_vfmv_f_s_f64m1_f64(
                            __riscv_vfredusum_vs_f64m2_f64m1(sum_b_z_f64m2, zero_f64m1, max_vector_length)) *
                        inv_points_count;
    *centroid_a_x = a[0] + mean_a_x;
    *centroid_a_y = a[1] + mean_a_y;
    *centroid_a_z = a[2] + mean_a_z;
    *centroid_b_x = b[0] + mean_b_x;
    *centroid_b_y = b[1] + mean_b_y;
    *centroid_b_z = b[2] + mean_b_z;
    nk_f64_t n_f64 = (nk_f64_t)points_count;
    cross_covariance[0] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_00_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_x * mean_b_x;
    cross_covariance[1] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_01_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_x * mean_b_y;
    cross_covariance[2] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_02_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_x * mean_b_z;
    cross_covariance[3] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_10_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_y * mean_b_x;
    cross_covariance[4] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_11_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_y * mean_b_y;
    cross_covariance[5] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_12_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_y * mean_b_z;
    cross_covariance[6] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_20_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_z * mean_b_x;
    cross_covariance[7] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_21_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_z * mean_b_y;
    cross_covariance[8] = __riscv_vfmv_f_s_f64m1_f64(
                              __riscv_vfredusum_vs_f64m2_f64m1(cross_22_f64m2, zero_f64m1, max_vector_length)) -
                          n_f64 * mean_a_z * mean_b_z;
    // Centered norm-squared via parallel-axis identity; clamp at zero for numeric safety.
    nk_f64_t norm_squared_a_sum = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m2_f64m1(norm_sq_a_f64m2, zero_f64m1, max_vector_length));
    *centered_norm_squared_a = norm_squared_a_sum -
                               n_f64 * (mean_a_x * mean_a_x + mean_a_y * mean_a_y + mean_a_z * mean_a_z);
    if (*centered_norm_squared_a < 0.0) *centered_norm_squared_a = 0.0;
}

/**
 *  @brief Compute centroids, cross-covariance, and the centered norm-squared of a (f64).
 *
 *  Used by the Umeyama caller for the scale; sums run over the points shifted by the first one.
 *  Per-lane @c vfadd_vv and @c vfmacc_vv accumulation with deferred @c vfredusum after the loop
 *  — eliminates 16 horizontal reductions per iteration.
 */
NUMKONG_INLINE void nk_centroid_and_cross_covariance_and_variance_f64_rvv_( //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t points_count,           //
    nk_f64_t *centroid_a_x, nk_f64_t *centroid_a_y, nk_f64_t *centroid_a_z, //
    nk_f64_t *centroid_b_x, nk_f64_t *centroid_b_y, nk_f64_t *centroid_b_z, //
    nk_f64_t cross_covariance[9], nk_f64_t *centered_norm_squared_a) {
    nk_size_t max_vector_length = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length),
                 sum_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t sum_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t sum_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length),
                 sum_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t sum_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t cross_00_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length),
                 cross_01_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t cross_02_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length),
                 cross_10_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t cross_11_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length),
                 cross_12_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t cross_20_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length),
                 cross_21_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t cross_22_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_00_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_01_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_02_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_10_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_11_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_12_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_20_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_21_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_22_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t norm_sq_a_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_norm_sq_a_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    nk_f64_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = points_count;
    for (nk_size_t vector_length; remaining > 0;
         remaining -= vector_length, a_ptr += vector_length * 3, b_ptr += vector_length * 3) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1x3_t a_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(a_ptr, vector_length);
        vfloat64m1_t a_x_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 0), a[0],
                                                        vector_length);
        vfloat64m1_t a_y_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 1), a[1],
                                                        vector_length);
        vfloat64m1_t a_z_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 2), a[2],
                                                        vector_length);
        vfloat64m1x3_t b_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(b_ptr, vector_length);
        vfloat64m1_t b_x_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 0), b[0],
                                                        vector_length);
        vfloat64m1_t b_y_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 1), b[1],
                                                        vector_length);
        vfloat64m1_t b_z_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 2), b[2],
                                                        vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_a_x_f64m1, &compensation_a_x_f64m1, a_x_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_a_y_f64m1, &compensation_a_y_f64m1, a_y_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_a_z_f64m1, &compensation_a_z_f64m1, a_z_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_b_x_f64m1, &compensation_b_x_f64m1, b_x_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_b_y_f64m1, &compensation_b_y_f64m1, b_y_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_b_z_f64m1, &compensation_b_z_f64m1, b_z_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_00_f64m1, &compensation_00_f64m1, a_x_f64m1, b_x_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_01_f64m1, &compensation_01_f64m1, a_x_f64m1, b_y_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_02_f64m1, &compensation_02_f64m1, a_x_f64m1, b_z_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_10_f64m1, &compensation_10_f64m1, a_y_f64m1, b_x_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_11_f64m1, &compensation_11_f64m1, a_y_f64m1, b_y_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_12_f64m1, &compensation_12_f64m1, a_y_f64m1, b_z_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_20_f64m1, &compensation_20_f64m1, a_z_f64m1, b_x_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_21_f64m1, &compensation_21_f64m1, a_z_f64m1, b_y_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&cross_22_f64m1, &compensation_22_f64m1, a_z_f64m1, b_z_f64m1, vector_length);
        // Accumulate norm-squared for a via Kahan-compensated products (self*self).
        nk_accumulate_product_f64m1_rvv_(&norm_sq_a_f64m1, &compensation_norm_sq_a_f64m1, a_x_f64m1, a_x_f64m1,
                                         vector_length);
        nk_accumulate_product_f64m1_rvv_(&norm_sq_a_f64m1, &compensation_norm_sq_a_f64m1, a_y_f64m1, a_y_f64m1,
                                         vector_length);
        nk_accumulate_product_f64m1_rvv_(&norm_sq_a_f64m1, &compensation_norm_sq_a_f64m1, a_z_f64m1, a_z_f64m1,
                                         vector_length);
    }
    nk_f64_t inv_points_count = 1.0 / (nk_f64_t)points_count;
    nk_f64_t mean_a_x = nk_dot_stable_sum_f64m1_rvv_(sum_a_x_f64m1, compensation_a_x_f64m1) * inv_points_count;
    nk_f64_t mean_a_y = nk_dot_stable_sum_f64m1_rvv_(sum_a_y_f64m1, compensation_a_y_f64m1) * inv_points_count;
    nk_f64_t mean_a_z = nk_dot_stable_sum_f64m1_rvv_(sum_a_z_f64m1, compensation_a_z_f64m1) * inv_points_count;
    nk_f64_t mean_b_x = nk_dot_stable_sum_f64m1_rvv_(sum_b_x_f64m1, compensation_b_x_f64m1) * inv_points_count;
    nk_f64_t mean_b_y = nk_dot_stable_sum_f64m1_rvv_(sum_b_y_f64m1, compensation_b_y_f64m1) * inv_points_count;
    nk_f64_t mean_b_z = nk_dot_stable_sum_f64m1_rvv_(sum_b_z_f64m1, compensation_b_z_f64m1) * inv_points_count;
    *centroid_a_x = a[0] + mean_a_x;
    *centroid_a_y = a[1] + mean_a_y;
    *centroid_a_z = a[2] + mean_a_z;
    *centroid_b_x = b[0] + mean_b_x;
    *centroid_b_y = b[1] + mean_b_y;
    *centroid_b_z = b[2] + mean_b_z;
    nk_f64_t n_f64 = (nk_f64_t)points_count;
    cross_covariance[0] = nk_dot_stable_sum_f64m1_rvv_(cross_00_f64m1, compensation_00_f64m1) -
                          n_f64 * mean_a_x * mean_b_x;
    cross_covariance[1] = nk_dot_stable_sum_f64m1_rvv_(cross_01_f64m1, compensation_01_f64m1) -
                          n_f64 * mean_a_x * mean_b_y;
    cross_covariance[2] = nk_dot_stable_sum_f64m1_rvv_(cross_02_f64m1, compensation_02_f64m1) -
                          n_f64 * mean_a_x * mean_b_z;
    cross_covariance[3] = nk_dot_stable_sum_f64m1_rvv_(cross_10_f64m1, compensation_10_f64m1) -
                          n_f64 * mean_a_y * mean_b_x;
    cross_covariance[4] = nk_dot_stable_sum_f64m1_rvv_(cross_11_f64m1, compensation_11_f64m1) -
                          n_f64 * mean_a_y * mean_b_y;
    cross_covariance[5] = nk_dot_stable_sum_f64m1_rvv_(cross_12_f64m1, compensation_12_f64m1) -
                          n_f64 * mean_a_y * mean_b_z;
    cross_covariance[6] = nk_dot_stable_sum_f64m1_rvv_(cross_20_f64m1, compensation_20_f64m1) -
                          n_f64 * mean_a_z * mean_b_x;
    cross_covariance[7] = nk_dot_stable_sum_f64m1_rvv_(cross_21_f64m1, compensation_21_f64m1) -
                          n_f64 * mean_a_z * mean_b_y;
    cross_covariance[8] = nk_dot_stable_sum_f64m1_rvv_(cross_22_f64m1, compensation_22_f64m1) -
                          n_f64 * mean_a_z * mean_b_z;
    // Centered norm-squared via parallel-axis identity; clamp at zero for numeric safety.
    nk_f64_t norm_squared_a_sum = nk_dot_stable_sum_f64m1_rvv_(norm_sq_a_f64m1, compensation_norm_sq_a_f64m1);
    *centered_norm_squared_a = norm_squared_a_sum -
                               n_f64 * (mean_a_x * mean_a_x + mean_a_y * mean_a_y + mean_a_z * mean_a_z);
    if (*centered_norm_squared_a < 0.0) *centered_norm_squared_a = 0.0;
}

/** Σ‖s · R · (aᵢ − ā) − (bᵢ − b̄)‖² summed from the residuals, since folding it via trace(R · H)
 *  cancels to √ε once the clouds align. */
NUMKONG_INLINE nk_f64_t nk_transformed_ssd_f64_rvv_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t points_count,
                                                    nk_f64_t const *centroid_a, nk_f64_t const *centroid_b,
                                                    nk_f64_t const *rotation, nk_f64_t scale) {
    nk_f64_t r[9];
    for (int j = 0; j < 9; ++j) r[j] = scale * rotation[j];
    nk_size_t max_vector_length = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_squared_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_squared_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    nk_f64_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = points_count;
    for (nk_size_t vector_length; remaining > 0;
         remaining -= vector_length, a_ptr += vector_length * 3, b_ptr += vector_length * 3) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1x3_t a_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(a_ptr, vector_length);
        vfloat64m1x3_t b_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(b_ptr, vector_length);
        vfloat64m1_t a_x_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 0), centroid_a[0],
                                                        vector_length);
        vfloat64m1_t a_y_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 1), centroid_a[1],
                                                        vector_length);
        vfloat64m1_t a_z_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 2), centroid_a[2],
                                                        vector_length);
        vfloat64m1_t b_x_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 0), centroid_b[0],
                                                        vector_length);
        vfloat64m1_t b_y_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 1), centroid_b[1],
                                                        vector_length);
        vfloat64m1_t b_z_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 2), centroid_b[2],
                                                        vector_length);
        vfloat64m1_t delta_x_f64m1 = __riscv_vfmacc_vf_f64m1(
            __riscv_vfmacc_vf_f64m1(__riscv_vfmsac_vf_f64m1(b_x_f64m1, r[0], a_x_f64m1, vector_length), r[1], a_y_f64m1,
                                    vector_length),
            r[2], a_z_f64m1, vector_length);
        vfloat64m1_t delta_y_f64m1 = __riscv_vfmacc_vf_f64m1(
            __riscv_vfmacc_vf_f64m1(__riscv_vfmsac_vf_f64m1(b_y_f64m1, r[3], a_x_f64m1, vector_length), r[4], a_y_f64m1,
                                    vector_length),
            r[5], a_z_f64m1, vector_length);
        vfloat64m1_t delta_z_f64m1 = __riscv_vfmacc_vf_f64m1(
            __riscv_vfmacc_vf_f64m1(__riscv_vfmsac_vf_f64m1(b_z_f64m1, r[6], a_x_f64m1, vector_length), r[7], a_y_f64m1,
                                    vector_length),
            r[8], a_z_f64m1, vector_length);
        nk_accumulate_product_f64m1_rvv_(&sum_squared_f64m1, &compensation_squared_f64m1, delta_x_f64m1, delta_x_f64m1,
                                         vector_length);
        nk_accumulate_product_f64m1_rvv_(&sum_squared_f64m1, &compensation_squared_f64m1, delta_y_f64m1, delta_y_f64m1,
                                         vector_length);
        nk_accumulate_product_f64m1_rvv_(&sum_squared_f64m1, &compensation_squared_f64m1, delta_z_f64m1, delta_z_f64m1,
                                         vector_length);
    }
    return nk_dot_stable_sum_f64m1_rvv_(sum_squared_f64m1, compensation_squared_f64m1);
}

/** Σ‖s · R · (aᵢ − ā) − (bᵢ − b̄)‖² of f32 clouds in f64, like @ref nk_transformed_ssd_f64_rvv_. */
NUMKONG_INLINE nk_f64_t nk_transformed_ssd_f32_rvv_(nk_f32_t const *a, nk_f32_t const *b, nk_size_t points_count,
                                                    nk_f64_t const *centroid_a, nk_f64_t const *centroid_b,
                                                    nk_f64_t const *rotation, nk_f64_t scale) {
    nk_f64_t r[9];
    for (int j = 0; j < 9; ++j) r[j] = scale * rotation[j];
    nk_size_t max_vector_length = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_squared_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    nk_f32_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = points_count;
    for (nk_size_t vector_length; remaining > 0;
         remaining -= vector_length, a_ptr += vector_length * 3, b_ptr += vector_length * 3) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1x3_t a_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(a_ptr, vector_length);
        vfloat32m1x3_t b_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(b_ptr, vector_length);
        vfloat64m2_t a_x_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 0), vector_length), centroid_a[0],
            vector_length);
        vfloat64m2_t a_y_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 1), vector_length), centroid_a[1],
            vector_length);
        vfloat64m2_t a_z_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 2), vector_length), centroid_a[2],
            vector_length);
        vfloat64m2_t b_x_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 0), vector_length), centroid_b[0],
            vector_length);
        vfloat64m2_t b_y_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 1), vector_length), centroid_b[1],
            vector_length);
        vfloat64m2_t b_z_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 2), vector_length), centroid_b[2],
            vector_length);
        vfloat64m2_t delta_x_f64m2 = __riscv_vfmacc_vf_f64m2(
            __riscv_vfmacc_vf_f64m2(__riscv_vfmsac_vf_f64m2(b_x_f64m2, r[0], a_x_f64m2, vector_length), r[1], a_y_f64m2,
                                    vector_length),
            r[2], a_z_f64m2, vector_length);
        vfloat64m2_t delta_y_f64m2 = __riscv_vfmacc_vf_f64m2(
            __riscv_vfmacc_vf_f64m2(__riscv_vfmsac_vf_f64m2(b_y_f64m2, r[3], a_x_f64m2, vector_length), r[4], a_y_f64m2,
                                    vector_length),
            r[5], a_z_f64m2, vector_length);
        vfloat64m2_t delta_z_f64m2 = __riscv_vfmacc_vf_f64m2(
            __riscv_vfmacc_vf_f64m2(__riscv_vfmsac_vf_f64m2(b_z_f64m2, r[6], a_x_f64m2, vector_length), r[7], a_y_f64m2,
                                    vector_length),
            r[8], a_z_f64m2, vector_length);
        sum_squared_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_squared_f64m2, delta_x_f64m2, delta_x_f64m2, vector_length);
        sum_squared_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_squared_f64m2, delta_y_f64m2, delta_y_f64m2, vector_length);
        sum_squared_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_squared_f64m2, delta_z_f64m2, delta_z_f64m2, vector_length);
    }
    return __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m2_f64m1(sum_squared_f64m2, __riscv_vfmv_v_f_f64m1(0.0, 1), max_vector_length));
}

#if NUMKONG_TARGET_RVV

NUMKONG_API nk_status_t nk_rmsd_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t points_count,
                                        nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                        nk_f64_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;

    if (points_count == 0) {
        *result = 0;
        return nk_success_k;
    }
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    nk_size_t max_vector_length = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_sq_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, max_vector_length);
    nk_f32_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = points_count;
    for (nk_size_t vector_length; remaining > 0;
         remaining -= vector_length, a_ptr += vector_length * 3, b_ptr += vector_length * 3) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1x3_t a_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(a_ptr, vector_length);
        vfloat32m1_t a_x_f32m1 = __riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 0);
        vfloat32m1_t a_y_f32m1 = __riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 1);
        vfloat32m1_t a_z_f32m1 = __riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 2);
        vfloat32m1x3_t b_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(b_ptr, vector_length);
        vfloat32m1_t b_x_f32m1 = __riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 0);
        vfloat32m1_t b_y_f32m1 = __riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 1);
        vfloat32m1_t b_z_f32m1 = __riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 2);
        // Accumulate (a−b)² per component, widening to f64.
        vfloat64m2_t a_x_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(a_x_f32m1, vector_length);
        vfloat64m2_t b_x_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(b_x_f32m1, vector_length);
        vfloat64m2_t a_y_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(a_y_f32m1, vector_length);
        vfloat64m2_t b_y_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(b_y_f32m1, vector_length);
        vfloat64m2_t a_z_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(a_z_f32m1, vector_length);
        vfloat64m2_t b_z_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(b_z_f32m1, vector_length);
        vfloat64m2_t delta_x_f64m2 = __riscv_vfsub_vv_f64m2(a_x_f64m2, b_x_f64m2, vector_length);
        vfloat64m2_t delta_y_f64m2 = __riscv_vfsub_vv_f64m2(a_y_f64m2, b_y_f64m2, vector_length);
        vfloat64m2_t delta_z_f64m2 = __riscv_vfsub_vv_f64m2(a_z_f64m2, b_z_f64m2, vector_length);
        sum_sq_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_sq_f64m2, delta_x_f64m2, delta_x_f64m2, vector_length);
        sum_sq_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_sq_f64m2, delta_y_f64m2, delta_y_f64m2, vector_length);
        sum_sq_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_sq_f64m2, delta_z_f64m2, delta_z_f64m2, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t sum_squared = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m2_f64m1(sum_sq_f64m2, zero_f64m1, max_vector_length));
    *result = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfsqrt_v_f64m1(__riscv_vfmv_s_f_f64m1(sum_squared / (nk_f64_t)points_count, 1), 1));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsd_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t points_count,
                                        nk_f64_t *a_centroid, nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale,
                                        nk_f64_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0;

    if (points_count == 0) {
        *result = 0;
        return nk_success_k;
    }
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    nk_size_t max_vector_length = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_sq_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    vfloat64m1_t compensation_sq_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, max_vector_length);
    nk_f64_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = points_count;
    for (nk_size_t vector_length; remaining > 0;
         remaining -= vector_length, a_ptr += vector_length * 3, b_ptr += vector_length * 3) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1x3_t a_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(a_ptr, vector_length);
        vfloat64m1_t a_x_f64m1 = __riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 0);
        vfloat64m1_t a_y_f64m1 = __riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 1);
        vfloat64m1_t a_z_f64m1 = __riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 2);
        vfloat64m1x3_t b_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(b_ptr, vector_length);
        vfloat64m1_t b_x_f64m1 = __riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 0);
        vfloat64m1_t b_y_f64m1 = __riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 1);
        vfloat64m1_t b_z_f64m1 = __riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 2);
        // Accumulate (a-b)^2 per component.
        vfloat64m1_t delta_x_f64m1 = __riscv_vfsub_vv_f64m1(a_x_f64m1, b_x_f64m1, vector_length);
        vfloat64m1_t delta_y_f64m1 = __riscv_vfsub_vv_f64m1(a_y_f64m1, b_y_f64m1, vector_length);
        vfloat64m1_t delta_z_f64m1 = __riscv_vfsub_vv_f64m1(a_z_f64m1, b_z_f64m1, vector_length);
        vfloat64m1_t dist_sq_f64m1 = __riscv_vfmul_vv_f64m1(delta_x_f64m1, delta_x_f64m1, vector_length);
        dist_sq_f64m1 = __riscv_vfmacc_vv_f64m1(dist_sq_f64m1, delta_y_f64m1, delta_y_f64m1, vector_length);
        dist_sq_f64m1 = __riscv_vfmacc_vv_f64m1(dist_sq_f64m1, delta_z_f64m1, delta_z_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_sq_f64m1, &compensation_sq_f64m1, dist_sq_f64m1, vector_length);
    }
    nk_f64_t sum_squared = nk_dot_stable_sum_f64m1_rvv_(sum_sq_f64m1, compensation_sq_f64m1);
    *result = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfsqrt_v_f64m1(__riscv_vfmv_s_f_f64m1(sum_squared / (nk_f64_t)points_count, 1), 1));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kabsch_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t points_count,
                                          nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                          nk_f32_t *scale, nk_f64_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (points_count == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0f;
        *result = 0;
        return nk_success_k;
    }

    if (scale) *scale = 1.0f;
    nk_f64_t centroid_a_x, centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z;
    nk_f64_t cross_covariance[9];
    nk_centroid_and_cross_covariance_f32_rvv_(a, b, points_count, &centroid_a_x, &centroid_a_y, &centroid_a_z,
                                              &centroid_b_x, &centroid_b_y, &centroid_b_z, cross_covariance);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    // Identity-dominant short-circuit: if H ≈ diag(positive), R = I.
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        if (nk_det3x3_f64_(optimal_rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f32_t)optimal_rotation[j];
    nk_f64_t const centroid_a[3] = {centroid_a_x, centroid_a_y, centroid_a_z};
    nk_f64_t const centroid_b[3] = {centroid_b_x, centroid_b_y, centroid_b_z};
    nk_f64_t sum_squared = nk_transformed_ssd_f32_rvv_(a, b, points_count, centroid_a, centroid_b, optimal_rotation,
                                                       1.0);
    *result = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfsqrt_v_f64m1(__riscv_vfmv_s_f_f64m1(sum_squared / (nk_f64_t)points_count, 1), 1));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kabsch_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t points_count,
                                          nk_f64_t *a_centroid, nk_f64_t *b_centroid, nk_f64_t *rotation,
                                          nk_f64_t *scale, nk_f64_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (points_count == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0;
        *result = 0;
        return nk_success_k;
    }

    if (scale) *scale = 1.0;
    nk_f64_t centroid_a_x, centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z;
    nk_f64_t cross_covariance[9];
    nk_centroid_and_cross_covariance_f64_rvv_(a, b, points_count, &centroid_a_x, &centroid_a_y, &centroid_a_z,
                                              &centroid_b_x, &centroid_b_y, &centroid_b_z, cross_covariance);
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Identity-dominant short-circuit: if H ≈ diag(positive), R = I.
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        if (nk_det3x3_f64_(optimal_rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    nk_f64_t const centroid_a[3] = {centroid_a_x, centroid_a_y, centroid_a_z};
    nk_f64_t const centroid_b[3] = {centroid_b_x, centroid_b_y, centroid_b_z};
    nk_f64_t sum_squared = nk_transformed_ssd_f64_rvv_(a, b, points_count, centroid_a, centroid_b, optimal_rotation,
                                                       1.0);
    *result = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfsqrt_v_f64m1(__riscv_vfmv_s_f_f64m1(sum_squared / (nk_f64_t)points_count, 1), 1));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t points_count,
                                           nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                           nk_f32_t *scale, nk_f64_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (points_count == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0f;
        *result = 0;
        return nk_success_k;
    }

    nk_f64_t centroid_a_x, centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z;
    nk_f64_t centered_norm_squared_a;
    nk_f64_t cross_covariance[9];
    nk_centroid_and_cross_covariance_and_variance_f32_rvv_(a, b, points_count, &centroid_a_x, &centroid_a_y,
                                                           &centroid_a_z, &centroid_b_x, &centroid_b_y, &centroid_b_z,
                                                           cross_covariance, &centered_norm_squared_a);
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    // Identity-dominant short-circuit: if H ≈ diag(positive), R = I and trace(R · H) = trace(H).
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t scale_factor;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        scale_factor = (cross_covariance[0] + cross_covariance[4] + cross_covariance[8]) / centered_norm_squared_a;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        nk_f64_t det = nk_det3x3_f64_(optimal_rotation);
        nk_f64_t sign_det = det < 0 ? -1.0 : 1.0;
        nk_f64_t trace_ds = nk_sum_three_products_f64_(svd_diagonal[0], 1.0, svd_diagonal[4], 1.0, svd_diagonal[8],
                                                       sign_det);
        scale_factor = trace_ds / centered_norm_squared_a;
        if (det < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
    }
    if (scale) *scale = (nk_f32_t)scale_factor;
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f32_t)optimal_rotation[j];
    nk_f64_t const centroid_a[3] = {centroid_a_x, centroid_a_y, centroid_a_z};
    nk_f64_t const centroid_b[3] = {centroid_b_x, centroid_b_y, centroid_b_z};
    nk_f64_t sum_squared = nk_transformed_ssd_f32_rvv_(a, b, points_count, centroid_a, centroid_b, optimal_rotation,
                                                       scale_factor);
    *result = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfsqrt_v_f64m1(__riscv_vfmv_s_f_f64m1(sum_squared / (nk_f64_t)points_count, 1), 1));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t points_count,
                                           nk_f64_t *a_centroid, nk_f64_t *b_centroid, nk_f64_t *rotation,
                                           nk_f64_t *scale, nk_f64_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (points_count == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0;
        *result = 0;
        return nk_success_k;
    }

    nk_f64_t centroid_a_x, centroid_a_y, centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z;
    nk_f64_t centered_norm_squared_a;
    nk_f64_t cross_covariance[9];
    nk_centroid_and_cross_covariance_and_variance_f64_rvv_(a, b, points_count, &centroid_a_x, &centroid_a_y,
                                                           &centroid_a_z, &centroid_b_x, &centroid_b_y, &centroid_b_z,
                                                           cross_covariance, &centered_norm_squared_a);
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Identity-dominant short-circuit: if H ≈ diag(positive), R = I and trace(R · H) = trace(H).
    nk_f64_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f64_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f64_t optimal_rotation[9];
    nk_f64_t scale_factor;
    if (covariance_offdiagonal_norm_squared < 1e-20 * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0 &&
        cross_covariance[4] > 0.0 && cross_covariance[8] > 0.0) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        scale_factor = (cross_covariance[0] + cross_covariance[4] + cross_covariance[8]) / centered_norm_squared_a;
    }
    else {
        nk_f64_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f64_(cross_covariance, svd_left, svd_diagonal, svd_right);
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        nk_f64_t det = nk_det3x3_f64_(optimal_rotation);
        nk_f64_t sign_det = det < 0 ? -1.0 : 1.0;
        nk_f64_t trace_ds = nk_sum_three_products_f64_(svd_diagonal[0], 1.0, svd_diagonal[4], 1.0, svd_diagonal[8],
                                                       sign_det);
        scale_factor = trace_ds / centered_norm_squared_a;
        if (det < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, optimal_rotation);
        }
    }
    if (scale) *scale = scale_factor;
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    nk_f64_t const centroid_a[3] = {centroid_a_x, centroid_a_y, centroid_a_z};
    nk_f64_t const centroid_b[3] = {centroid_b_x, centroid_b_y, centroid_b_z};
    nk_f64_t sum_squared = nk_transformed_ssd_f64_rvv_(a, b, points_count, centroid_a, centroid_b, optimal_rotation,
                                                       scale_factor);
    *result = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfsqrt_v_f64m1(__riscv_vfmv_s_f_f64m1(sum_squared / (nk_f64_t)points_count, 1), 1));
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsd_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t points_count,
                                        nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale,
                                        nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_rmsd_f16_(a, b, points_count, a_centroid, b_centroid, rotation, scale, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kabsch_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t points_count,
                                          nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                          nk_f32_t *scale, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_kabsch_f16_(a, b, points_count, a_centroid, b_centroid, rotation, scale, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t points_count,
                                           nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                           nk_f32_t *scale, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_umeyama_f16_(a, b, points_count, a_centroid, b_centroid, rotation, scale, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsd_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t points_count,
                                         nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                         nk_f32_t *scale, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_rmsd_bf16_(a, b, points_count, a_centroid, b_centroid, rotation, scale, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kabsch_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t points_count,
                                           nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                           nk_f32_t *scale, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_kabsch_bf16_(a, b, points_count, a_centroid, b_centroid, rotation, scale, result);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t points_count,
                                            nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                            nk_f32_t *scale, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_umeyama_bf16_(a, b, points_count, a_centroid, b_centroid, rotation, scale, result);
    return nk_success_k;
}

#endif // NUMKONG_TARGET_RVV

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NUMKONG_ARCH_RISCV64_RVV_
#endif // NUMKONG_ARCH_RISCV64_
#endif // NUMKONG_MESH_RVV_H
