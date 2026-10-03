/**
 *  @brief SIMD-accelerated Mesh Operations for RISC-V.
 *  @file include/numkong/mesh/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/mesh.h
 *
 *  RVV mesh operations leverage:
 *
 *  - `vlseg3e32`/`vlseg3e64`: deinterleave xyz triplets in hardware
 *  - `vfwcvt`/`vfwmacc`: widening FMA for f32→f64 accumulation
 *  - `vfredusum`: single-instruction horizontal reduction
 *  - Serial SVD/determinant from mesh/serial.h for fixed 3×3 matrix operations
 *
 *  Fused helpers minimize data passes:
 *
 *  - RMSD: fully fused single-pass (centroids + squared diffs), no separate helper
 *  - `nk_centroid_and_cross_covariance_*_rvv_`: centroids + H in one pass (Kabsch)
 *  - `nk_centroid_and_cross_covariance_and_variance_*_rvv_`: + variance (Umeyama)
 *
 *  Math for fused centroid+covariance:
 *    H[i][j] = Σ (a[i] - ca[i]) * (b[j] - cb[j])
 *            = Σ a[i] * b[j] - n * ca[i] * cb[j]
 *  So we accumulate raw Σ a[i] * b[j] in the loop, then fix up after.
 *
 *  Key RVV-specific optimizations (vs. scalar or x86 backends):
 *
 *  - Deferred horizontal reduction in bicentroid: per-lane `vfwadd_wv` (f32)
 *    or `vfadd_vv` (f64) accumulation across loop iterations, with a single
 *    `vfredusum` after the loop — eliminates 6 `vfredusum` per iteration.
 *  - `vfwmacc_vv` in f32 SSD: accumulates widened squared distances per-lane
 *    (dx²+dy²+dz²) before a single reduction — saves 2 `vfredusum` per iteration.
 *  - Vectorized R = V×Uᵀ via `vfmul_vf`/`vfmacc_vf`: each output row computed
 *    as a 3-element vector dot product — 15 vector ops vs 45 scalar ops.
 *  - `vfncvt_f_f_w` for f64→f32 narrowing of H matrix before SVD.
 */
#ifndef NK_MESH_RVV_H
#define NK_MESH_RVV_H

#if NK_TARGET_RISCV_
#if NK_TARGET_RVV

#include "numkong/types.h"
#include "numkong/dot/rvv.h"
#include "numkong/spatial/rvv.h" // `nk_f32_sqrt_rvv`, `nk_f64_sqrt_rvv`
#include "numkong/mesh/serial.h" // `nk_svd3x3_f32_`, `nk_svd3x3_f64_`, `nk_det3x3_f32_`, `nk_det3x3_f64_`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

NK_INTERNAL void nk_accumulate_sum_f64m1_rvv_(vfloat64m1_t *sum_f64m1, vfloat64m1_t *compensation_f64m1,
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

NK_INTERNAL void nk_accumulate_product_f64m1_rvv_(vfloat64m1_t *sum_f64m1, vfloat64m1_t *compensation_f64m1,
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
 *  Accumulates raw Σ a[i]*b[j] and Σ a[i], Σ b[j] simultaneously, then:
 *    ca = Σa / n,  cb = Σb / n
 *    H[i][j] = raw[i][j] - n * ca[i] * cb[j]
 *
 *  Reduces Kabsch from 4 passes to 2 (fused centroid+covariance + SSD).
 *  Cross-products use per-lane `vfwmacc_vv` accumulation (vfloat64m2_t) with
 *  deferred `vfredusum` after the loop — eliminates 9 reductions per iteration.
 */
NK_INTERNAL void nk_centroid_and_cross_covariance_f32_rvv_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,      //
    nk_f64_t *ca_x, nk_f64_t *ca_y, nk_f64_t *ca_z,         //
    nk_f64_t *cb_x, nk_f64_t *cb_y, nk_f64_t *cb_z,         //
    nk_f64_t h[9]) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_a_x_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax), sum_a_y_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_a_z_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_b_x_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax), sum_b_y_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_b_z_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_00_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax),
                 cross_01_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_02_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax),
                 cross_10_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_11_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax),
                 cross_12_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_20_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax),
                 cross_21_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_22_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    nk_f32_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
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
        sum_a_x_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_a_x_f64m2, sum_a_x_f64m2, a_x_f32m1, vector_length);
        sum_a_y_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_a_y_f64m2, sum_a_y_f64m2, a_y_f32m1, vector_length);
        sum_a_z_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_a_z_f64m2, sum_a_z_f64m2, a_z_f32m1, vector_length);
        sum_b_x_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_b_x_f64m2, sum_b_x_f64m2, b_x_f32m1, vector_length);
        sum_b_y_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_b_y_f64m2, sum_b_y_f64m2, b_y_f32m1, vector_length);
        sum_b_z_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_b_z_f64m2, sum_b_z_f64m2, b_z_f32m1, vector_length);
        cross_00_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_00_f64m2, a_x_f32m1, b_x_f32m1, vector_length);
        cross_01_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_01_f64m2, a_x_f32m1, b_y_f32m1, vector_length);
        cross_02_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_02_f64m2, a_x_f32m1, b_z_f32m1, vector_length);
        cross_10_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_10_f64m2, a_y_f32m1, b_x_f32m1, vector_length);
        cross_11_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_11_f64m2, a_y_f32m1, b_y_f32m1, vector_length);
        cross_12_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_12_f64m2, a_y_f32m1, b_z_f32m1, vector_length);
        cross_20_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_20_f64m2, a_z_f32m1, b_x_f32m1, vector_length);
        cross_21_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_21_f64m2, a_z_f32m1, b_y_f32m1, vector_length);
        cross_22_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_22_f64m2, a_z_f32m1, b_z_f32m1, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t ca_x_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_a_x_f64m2, zero_f64m1, vlmax)) *
                     inv_n;
    nk_f64_t ca_y_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_a_y_f64m2, zero_f64m1, vlmax)) *
                     inv_n;
    nk_f64_t ca_z_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_a_z_f64m2, zero_f64m1, vlmax)) *
                     inv_n;
    nk_f64_t cb_x_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_b_x_f64m2, zero_f64m1, vlmax)) *
                     inv_n;
    nk_f64_t cb_y_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_b_y_f64m2, zero_f64m1, vlmax)) *
                     inv_n;
    nk_f64_t cb_z_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_b_z_f64m2, zero_f64m1, vlmax)) *
                     inv_n;
    *ca_x = ca_x_;
    *ca_y = ca_y_;
    *ca_z = ca_z_;
    *cb_x = cb_x_;
    *cb_y = cb_y_;
    *cb_z = cb_z_;
    // Fix up: H[i][j] = raw[i][j] - n * ca[i] * cb[j]
    nk_f64_t n_f64 = (nk_f64_t)n;
    h[0] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_00_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_x_ * cb_x_;
    h[1] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_01_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_x_ * cb_y_;
    h[2] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_02_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_x_ * cb_z_;
    h[3] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_10_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_y_ * cb_x_;
    h[4] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_11_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_y_ * cb_y_;
    h[5] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_12_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_y_ * cb_z_;
    h[6] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_20_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_z_ * cb_x_;
    h[7] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_21_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_z_ * cb_y_;
    h[8] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_22_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_z_ * cb_z_;
}

/**
 *  @brief Compute centroids and cross-covariance matrix in a single pass (f64).
 *
 *  Per-lane `vfadd_vv`/`vfmacc_vv` accumulation with deferred `vfredusum` after the loop
 *  — eliminates 15 horizontal reductions per iteration.
 */
NK_INTERNAL void nk_centroid_and_cross_covariance_f64_rvv_( //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,      //
    nk_f64_t *ca_x, nk_f64_t *ca_y, nk_f64_t *ca_z,         //
    nk_f64_t *cb_x, nk_f64_t *cb_y, nk_f64_t *cb_z,         //
    nk_f64_t h[9]) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), sum_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), sum_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_00_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax),
                 cross_01_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_02_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax),
                 cross_10_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_11_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax),
                 cross_12_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_20_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax),
                 cross_21_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_22_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_00_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_01_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_02_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_10_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_11_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_12_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_20_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_21_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_22_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    nk_f64_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
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
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t ca_x_ = nk_dot_stable_sum_f64m1_rvv_(sum_a_x_f64m1, compensation_a_x_f64m1) * inv_n;
    nk_f64_t ca_y_ = nk_dot_stable_sum_f64m1_rvv_(sum_a_y_f64m1, compensation_a_y_f64m1) * inv_n;
    nk_f64_t ca_z_ = nk_dot_stable_sum_f64m1_rvv_(sum_a_z_f64m1, compensation_a_z_f64m1) * inv_n;
    nk_f64_t cb_x_ = nk_dot_stable_sum_f64m1_rvv_(sum_b_x_f64m1, compensation_b_x_f64m1) * inv_n;
    nk_f64_t cb_y_ = nk_dot_stable_sum_f64m1_rvv_(sum_b_y_f64m1, compensation_b_y_f64m1) * inv_n;
    nk_f64_t cb_z_ = nk_dot_stable_sum_f64m1_rvv_(sum_b_z_f64m1, compensation_b_z_f64m1) * inv_n;
    *ca_x = ca_x_;
    *ca_y = ca_y_;
    *ca_z = ca_z_;
    *cb_x = cb_x_;
    *cb_y = cb_y_;
    *cb_z = cb_z_;
    nk_f64_t n_f64 = (nk_f64_t)n;
    h[0] = nk_dot_stable_sum_f64m1_rvv_(cross_00_f64m1, compensation_00_f64m1) - n_f64 * ca_x_ * cb_x_;
    h[1] = nk_dot_stable_sum_f64m1_rvv_(cross_01_f64m1, compensation_01_f64m1) - n_f64 * ca_x_ * cb_y_;
    h[2] = nk_dot_stable_sum_f64m1_rvv_(cross_02_f64m1, compensation_02_f64m1) - n_f64 * ca_x_ * cb_z_;
    h[3] = nk_dot_stable_sum_f64m1_rvv_(cross_10_f64m1, compensation_10_f64m1) - n_f64 * ca_y_ * cb_x_;
    h[4] = nk_dot_stable_sum_f64m1_rvv_(cross_11_f64m1, compensation_11_f64m1) - n_f64 * ca_y_ * cb_y_;
    h[5] = nk_dot_stable_sum_f64m1_rvv_(cross_12_f64m1, compensation_12_f64m1) - n_f64 * ca_y_ * cb_z_;
    h[6] = nk_dot_stable_sum_f64m1_rvv_(cross_20_f64m1, compensation_20_f64m1) - n_f64 * ca_z_ * cb_x_;
    h[7] = nk_dot_stable_sum_f64m1_rvv_(cross_21_f64m1, compensation_21_f64m1) - n_f64 * ca_z_ * cb_y_;
    h[8] = nk_dot_stable_sum_f64m1_rvv_(cross_22_f64m1, compensation_22_f64m1) - n_f64 * ca_z_ * cb_z_;
}

/**
 *  @brief Compute centroids, cross-covariance, and variance_a in a single pass (f32).
 *
 *  Same as centroid_and_cross_covariance but also computes:
 *    variance_a = (1/n) * Σ ||a[i] - ca||²
 *               = (1/n) * (Σ ||a[i]||² - n * ||ca||²)
 *
 *  Cross-products use per-lane `vfwmacc_vv` accumulation (vfloat64m2_t) with
 *  deferred `vfredusum` after the loop — eliminates 9 reductions per iteration.
 */
NK_INTERNAL void nk_centroid_and_cross_covariance_and_variance_f32_rvv_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n,                   //
    nk_f64_t *ca_x, nk_f64_t *ca_y, nk_f64_t *ca_z,                      //
    nk_f64_t *cb_x, nk_f64_t *cb_y, nk_f64_t *cb_z,                      //
    nk_f64_t h[9], nk_f64_t *variance_a) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_a_x_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax), sum_a_y_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_a_z_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_b_x_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax), sum_b_y_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_b_z_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_00_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax),
                 cross_01_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_02_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax),
                 cross_10_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_11_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax),
                 cross_12_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_20_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax),
                 cross_21_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t cross_22_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_norm_squared_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    nk_f32_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
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
        sum_a_x_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_a_x_f64m2, sum_a_x_f64m2, a_x_f32m1, vector_length);
        sum_a_y_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_a_y_f64m2, sum_a_y_f64m2, a_y_f32m1, vector_length);
        sum_a_z_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_a_z_f64m2, sum_a_z_f64m2, a_z_f32m1, vector_length);
        sum_b_x_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_b_x_f64m2, sum_b_x_f64m2, b_x_f32m1, vector_length);
        sum_b_y_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_b_y_f64m2, sum_b_y_f64m2, b_y_f32m1, vector_length);
        sum_b_z_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_b_z_f64m2, sum_b_z_f64m2, b_z_f32m1, vector_length);
        cross_00_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_00_f64m2, a_x_f32m1, b_x_f32m1, vector_length);
        cross_01_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_01_f64m2, a_x_f32m1, b_y_f32m1, vector_length);
        cross_02_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_02_f64m2, a_x_f32m1, b_z_f32m1, vector_length);
        cross_10_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_10_f64m2, a_y_f32m1, b_x_f32m1, vector_length);
        cross_11_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_11_f64m2, a_y_f32m1, b_y_f32m1, vector_length);
        cross_12_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_12_f64m2, a_y_f32m1, b_z_f32m1, vector_length);
        cross_20_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_20_f64m2, a_z_f32m1, b_x_f32m1, vector_length);
        cross_21_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_21_f64m2, a_z_f32m1, b_y_f32m1, vector_length);
        cross_22_f64m2 = __riscv_vfwmacc_vv_f64m2_tu(cross_22_f64m2, a_z_f32m1, b_z_f32m1, vector_length);
        // Variance: Σ (a_x² + a_y² + a_z²) — raw, not centered.
        vfloat64m2_t norm_squared_f64m2 = __riscv_vfwmul_vv_f64m2(a_x_f32m1, a_x_f32m1, vector_length);
        norm_squared_f64m2 = __riscv_vfwmacc_vv_f64m2(norm_squared_f64m2, a_y_f32m1, a_y_f32m1, vector_length);
        norm_squared_f64m2 = __riscv_vfwmacc_vv_f64m2(norm_squared_f64m2, a_z_f32m1, a_z_f32m1, vector_length);
        sum_norm_squared_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_norm_squared_f64m2, sum_norm_squared_f64m2,
                                                           norm_squared_f64m2, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t ca_x_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_a_x_f64m2, zero_f64m1, vlmax)) *
                     inv_n;
    nk_f64_t ca_y_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_a_y_f64m2, zero_f64m1, vlmax)) *
                     inv_n;
    nk_f64_t ca_z_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_a_z_f64m2, zero_f64m1, vlmax)) *
                     inv_n;
    nk_f64_t cb_x_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_b_x_f64m2, zero_f64m1, vlmax)) *
                     inv_n;
    nk_f64_t cb_y_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_b_y_f64m2, zero_f64m1, vlmax)) *
                     inv_n;
    nk_f64_t cb_z_ = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_b_z_f64m2, zero_f64m1, vlmax)) *
                     inv_n;
    *ca_x = ca_x_;
    *ca_y = ca_y_;
    *ca_z = ca_z_;
    *cb_x = cb_x_;
    *cb_y = cb_y_;
    *cb_z = cb_z_;
    nk_f64_t n_f64 = (nk_f64_t)n;
    h[0] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_00_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_x_ * cb_x_;
    h[1] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_01_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_x_ * cb_y_;
    h[2] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_02_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_x_ * cb_z_;
    h[3] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_10_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_y_ * cb_x_;
    h[4] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_11_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_y_ * cb_y_;
    h[5] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_12_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_y_ * cb_z_;
    h[6] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_20_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_z_ * cb_x_;
    h[7] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_21_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_z_ * cb_y_;
    h[8] = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(cross_22_f64m2, zero_f64m1, vlmax)) -
           n_f64 * ca_z_ * cb_z_;
    // variance_a = (1/n) * (Σ ||a[i]||² - n * ||ca||²)
    *variance_a = __riscv_vfmv_f_s_f64m1_f64(
                      __riscv_vfredusum_vs_f64m2_f64m1(sum_norm_squared_f64m2, zero_f64m1, vlmax)) *
                      inv_n -
                  (ca_x_ * ca_x_ + ca_y_ * ca_y_ + ca_z_ * ca_z_);
}

/**
 *  @brief Compute centroids, cross-covariance, and variance_a in a single pass (f64).
 *
 *  Per-lane `vfadd_vv`/`vfmacc_vv` accumulation with deferred `vfredusum` after the loop
 *  — eliminates 16 horizontal reductions per iteration.
 */
NK_INTERNAL void nk_centroid_and_cross_covariance_and_variance_f64_rvv_( //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n,                   //
    nk_f64_t *ca_x, nk_f64_t *ca_y, nk_f64_t *ca_z,                      //
    nk_f64_t *cb_x, nk_f64_t *cb_y, nk_f64_t *cb_z,                      //
    nk_f64_t h[9], nk_f64_t *variance_a) {
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), sum_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax), sum_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_00_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax),
                 cross_01_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_02_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax),
                 cross_10_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_11_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax),
                 cross_12_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_20_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax),
                 cross_21_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t cross_22_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_00_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_01_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_02_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_10_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_11_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_12_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_20_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_21_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_22_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_norm_squared_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_norm_squared_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    nk_f64_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
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
        vfloat64m1_t norm_squared_f64m1 = __riscv_vfmul_vv_f64m1(a_x_f64m1, a_x_f64m1, vector_length);
        norm_squared_f64m1 = __riscv_vfmacc_vv_f64m1(norm_squared_f64m1, a_y_f64m1, a_y_f64m1, vector_length);
        norm_squared_f64m1 = __riscv_vfmacc_vv_f64m1(norm_squared_f64m1, a_z_f64m1, a_z_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_norm_squared_f64m1, &compensation_norm_squared_f64m1, norm_squared_f64m1,
                                     vector_length);
    }
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t ca_x_ = nk_dot_stable_sum_f64m1_rvv_(sum_a_x_f64m1, compensation_a_x_f64m1) * inv_n;
    nk_f64_t ca_y_ = nk_dot_stable_sum_f64m1_rvv_(sum_a_y_f64m1, compensation_a_y_f64m1) * inv_n;
    nk_f64_t ca_z_ = nk_dot_stable_sum_f64m1_rvv_(sum_a_z_f64m1, compensation_a_z_f64m1) * inv_n;
    nk_f64_t cb_x_ = nk_dot_stable_sum_f64m1_rvv_(sum_b_x_f64m1, compensation_b_x_f64m1) * inv_n;
    nk_f64_t cb_y_ = nk_dot_stable_sum_f64m1_rvv_(sum_b_y_f64m1, compensation_b_y_f64m1) * inv_n;
    nk_f64_t cb_z_ = nk_dot_stable_sum_f64m1_rvv_(sum_b_z_f64m1, compensation_b_z_f64m1) * inv_n;
    *ca_x = ca_x_;
    *ca_y = ca_y_;
    *ca_z = ca_z_;
    *cb_x = cb_x_;
    *cb_y = cb_y_;
    *cb_z = cb_z_;
    nk_f64_t n_f64 = (nk_f64_t)n;
    h[0] = nk_dot_stable_sum_f64m1_rvv_(cross_00_f64m1, compensation_00_f64m1) - n_f64 * ca_x_ * cb_x_;
    h[1] = nk_dot_stable_sum_f64m1_rvv_(cross_01_f64m1, compensation_01_f64m1) - n_f64 * ca_x_ * cb_y_;
    h[2] = nk_dot_stable_sum_f64m1_rvv_(cross_02_f64m1, compensation_02_f64m1) - n_f64 * ca_x_ * cb_z_;
    h[3] = nk_dot_stable_sum_f64m1_rvv_(cross_10_f64m1, compensation_10_f64m1) - n_f64 * ca_y_ * cb_x_;
    h[4] = nk_dot_stable_sum_f64m1_rvv_(cross_11_f64m1, compensation_11_f64m1) - n_f64 * ca_y_ * cb_y_;
    h[5] = nk_dot_stable_sum_f64m1_rvv_(cross_12_f64m1, compensation_12_f64m1) - n_f64 * ca_y_ * cb_z_;
    h[6] = nk_dot_stable_sum_f64m1_rvv_(cross_20_f64m1, compensation_20_f64m1) - n_f64 * ca_z_ * cb_x_;
    h[7] = nk_dot_stable_sum_f64m1_rvv_(cross_21_f64m1, compensation_21_f64m1) - n_f64 * ca_z_ * cb_y_;
    h[8] = nk_dot_stable_sum_f64m1_rvv_(cross_22_f64m1, compensation_22_f64m1) - n_f64 * ca_z_ * cb_z_;
    *variance_a = nk_dot_stable_sum_f64m1_rvv_(sum_norm_squared_f64m1, compensation_norm_squared_f64m1) * inv_n -
                  (ca_x_ * ca_x_ + ca_y_ * ca_y_ + ca_z_ * ca_z_);
}

NK_INTERNAL nk_f64_t nk_transformed_ssd_f32_rvv_(      //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, //
    nk_f64_t const *r, nk_f64_t scale,                 //
    nk_f64_t ca_x, nk_f64_t ca_y, nk_f64_t ca_z,       //
    nk_f64_t cb_x, nk_f64_t cb_y, nk_f64_t cb_z) {
    nk_f64_t scaled_rotation_x_x = scale * r[0], scaled_rotation_x_y = scale * r[1], scaled_rotation_x_z = scale * r[2];
    nk_f64_t scaled_rotation_y_x = scale * r[3], scaled_rotation_y_y = scale * r[4], scaled_rotation_y_z = scale * r[5];
    nk_f64_t scaled_rotation_z_x = scale * r[6], scaled_rotation_z_y = scale * r[7], scaled_rotation_z_z = scale * r[8];
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_distance_squared_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f32_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
    for (nk_size_t vector_length; remaining > 0;
         remaining -= vector_length, a_ptr += vector_length * 3, b_ptr += vector_length * 3) {
        vector_length = __riscv_vsetvl_e32m1(remaining);
        vfloat32m1x3_t a_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(a_ptr, vector_length);
        vfloat64m2_t centered_a_x_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 0), vector_length), ca_x, vector_length);
        vfloat64m2_t centered_a_y_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 1), vector_length), ca_y, vector_length);
        vfloat64m2_t centered_a_z_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(a_f32m1x3, 2), vector_length), ca_z, vector_length);
        vfloat64m2_t rotated_a_x_f64m2 = __riscv_vfmul_vf_f64m2(centered_a_x_f64m2, scaled_rotation_x_x, vector_length);
        rotated_a_x_f64m2 = __riscv_vfmacc_vf_f64m2(rotated_a_x_f64m2, scaled_rotation_x_y, centered_a_y_f64m2,
                                                    vector_length);
        rotated_a_x_f64m2 = __riscv_vfmacc_vf_f64m2(rotated_a_x_f64m2, scaled_rotation_x_z, centered_a_z_f64m2,
                                                    vector_length);
        vfloat64m2_t rotated_a_y_f64m2 = __riscv_vfmul_vf_f64m2(centered_a_x_f64m2, scaled_rotation_y_x, vector_length);
        rotated_a_y_f64m2 = __riscv_vfmacc_vf_f64m2(rotated_a_y_f64m2, scaled_rotation_y_y, centered_a_y_f64m2,
                                                    vector_length);
        rotated_a_y_f64m2 = __riscv_vfmacc_vf_f64m2(rotated_a_y_f64m2, scaled_rotation_y_z, centered_a_z_f64m2,
                                                    vector_length);
        vfloat64m2_t rotated_a_z_f64m2 = __riscv_vfmul_vf_f64m2(centered_a_x_f64m2, scaled_rotation_z_x, vector_length);
        rotated_a_z_f64m2 = __riscv_vfmacc_vf_f64m2(rotated_a_z_f64m2, scaled_rotation_z_y, centered_a_y_f64m2,
                                                    vector_length);
        rotated_a_z_f64m2 = __riscv_vfmacc_vf_f64m2(rotated_a_z_f64m2, scaled_rotation_z_z, centered_a_z_f64m2,
                                                    vector_length);
        vfloat32m1x3_t b_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(b_ptr, vector_length);
        vfloat64m2_t centered_b_x_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 0), vector_length), cb_x, vector_length);
        vfloat64m2_t centered_b_y_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 1), vector_length), cb_y, vector_length);
        vfloat64m2_t centered_b_z_f64m2 = __riscv_vfsub_vf_f64m2(
            __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(b_f32m1x3, 2), vector_length), cb_z, vector_length);
        vfloat64m2_t delta_x_f64m2 = __riscv_vfsub_vv_f64m2(rotated_a_x_f64m2, centered_b_x_f64m2, vector_length);
        vfloat64m2_t delta_y_f64m2 = __riscv_vfsub_vv_f64m2(rotated_a_y_f64m2, centered_b_y_f64m2, vector_length);
        vfloat64m2_t delta_z_f64m2 = __riscv_vfsub_vv_f64m2(rotated_a_z_f64m2, centered_b_z_f64m2, vector_length);
        sum_distance_squared_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_distance_squared_f64m2, delta_x_f64m2,
                                                                delta_x_f64m2, vector_length);
        sum_distance_squared_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_distance_squared_f64m2, delta_y_f64m2,
                                                                delta_y_f64m2, vector_length);
        sum_distance_squared_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_distance_squared_f64m2, delta_z_f64m2,
                                                                delta_z_f64m2, vector_length);
    }
    return __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_distance_squared_f64m2, zero_f64m1, vlmax));
}

NK_INTERNAL nk_f64_t nk_transformed_ssd_f64_rvv_(      //
    nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, //
    nk_f64_t const *r, nk_f64_t scale,                 //
    nk_f64_t ca_x, nk_f64_t ca_y, nk_f64_t ca_z,       //
    nk_f64_t cb_x, nk_f64_t cb_y, nk_f64_t cb_z) {
    nk_f64_t scaled_rotation_x_x = scale * r[0], scaled_rotation_x_y = scale * r[1], scaled_rotation_x_z = scale * r[2];
    nk_f64_t scaled_rotation_y_x = scale * r[3], scaled_rotation_y_y = scale * r[4], scaled_rotation_y_z = scale * r[5];
    nk_f64_t scaled_rotation_z_x = scale * r[6], scaled_rotation_z_y = scale * r[7], scaled_rotation_z_z = scale * r[8];
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_distance_squared_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_distance_squared_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    nk_f64_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
    for (nk_size_t vector_length; remaining > 0;
         remaining -= vector_length, a_ptr += vector_length * 3, b_ptr += vector_length * 3) {
        vector_length = __riscv_vsetvl_e64m1(remaining);
        vfloat64m1x3_t a_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(a_ptr, vector_length);
        vfloat64m1_t centered_a_x_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 0), ca_x,
                                                                 vector_length);
        vfloat64m1_t centered_a_y_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 1), ca_y,
                                                                 vector_length);
        vfloat64m1_t centered_a_z_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(a_f64m1x3, 2), ca_z,
                                                                 vector_length);
        vfloat64m1_t rotated_a_x_f64m1 = __riscv_vfmul_vf_f64m1(centered_a_x_f64m1, scaled_rotation_x_x, vector_length);
        rotated_a_x_f64m1 = __riscv_vfmacc_vf_f64m1(rotated_a_x_f64m1, scaled_rotation_x_y, centered_a_y_f64m1,
                                                    vector_length);
        rotated_a_x_f64m1 = __riscv_vfmacc_vf_f64m1(rotated_a_x_f64m1, scaled_rotation_x_z, centered_a_z_f64m1,
                                                    vector_length);
        vfloat64m1_t rotated_a_y_f64m1 = __riscv_vfmul_vf_f64m1(centered_a_x_f64m1, scaled_rotation_y_x, vector_length);
        rotated_a_y_f64m1 = __riscv_vfmacc_vf_f64m1(rotated_a_y_f64m1, scaled_rotation_y_y, centered_a_y_f64m1,
                                                    vector_length);
        rotated_a_y_f64m1 = __riscv_vfmacc_vf_f64m1(rotated_a_y_f64m1, scaled_rotation_y_z, centered_a_z_f64m1,
                                                    vector_length);
        vfloat64m1_t rotated_a_z_f64m1 = __riscv_vfmul_vf_f64m1(centered_a_x_f64m1, scaled_rotation_z_x, vector_length);
        rotated_a_z_f64m1 = __riscv_vfmacc_vf_f64m1(rotated_a_z_f64m1, scaled_rotation_z_y, centered_a_y_f64m1,
                                                    vector_length);
        rotated_a_z_f64m1 = __riscv_vfmacc_vf_f64m1(rotated_a_z_f64m1, scaled_rotation_z_z, centered_a_z_f64m1,
                                                    vector_length);
        vfloat64m1x3_t b_f64m1x3 = __riscv_vlseg3e64_v_f64m1x3(b_ptr, vector_length);
        vfloat64m1_t centered_b_x_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 0), cb_x,
                                                                 vector_length);
        vfloat64m1_t centered_b_y_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 1), cb_y,
                                                                 vector_length);
        vfloat64m1_t centered_b_z_f64m1 = __riscv_vfsub_vf_f64m1(__riscv_vget_v_f64m1x3_f64m1(b_f64m1x3, 2), cb_z,
                                                                 vector_length);
        vfloat64m1_t delta_x_f64m1 = __riscv_vfsub_vv_f64m1(rotated_a_x_f64m1, centered_b_x_f64m1, vector_length);
        vfloat64m1_t delta_y_f64m1 = __riscv_vfsub_vv_f64m1(rotated_a_y_f64m1, centered_b_y_f64m1, vector_length);
        vfloat64m1_t delta_z_f64m1 = __riscv_vfsub_vv_f64m1(rotated_a_z_f64m1, centered_b_z_f64m1, vector_length);
        vfloat64m1_t distance_squared_f64m1 = __riscv_vfmul_vv_f64m1(delta_x_f64m1, delta_x_f64m1, vector_length);
        distance_squared_f64m1 = __riscv_vfmacc_vv_f64m1(distance_squared_f64m1, delta_y_f64m1, delta_y_f64m1,
                                                         vector_length);
        distance_squared_f64m1 = __riscv_vfmacc_vv_f64m1(distance_squared_f64m1, delta_z_f64m1, delta_z_f64m1,
                                                         vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_distance_squared_f64m1, &compensation_distance_squared_f64m1,
                                     distance_squared_f64m1, vector_length);
    }
    return nk_dot_stable_sum_f64m1_rvv_(sum_distance_squared_f64m1, compensation_distance_squared_f64m1);
}

/** @brief Compute R = V * Uᵀ from SVD factors (f32), vectorized with `vfmul_vf`/`vfmacc_vf`. */
NK_INTERNAL void nk_rotation_from_svd_f32_rvv_( //
    nk_f32_t *svd_u, nk_f32_t *svd_v, nk_f32_t r[9]) {
    nk_size_t vl3 = __riscv_vsetvl_e32m1(3);
    vfloat32m1_t u_row0_f32m1 = __riscv_vle32_v_f32m1(svd_u + 0, vl3);
    vfloat32m1_t u_row1_f32m1 = __riscv_vle32_v_f32m1(svd_u + 3, vl3);
    vfloat32m1_t u_row2_f32m1 = __riscv_vle32_v_f32m1(svd_u + 6, vl3);
    // Row 0: R[0..2] = V[0]*U_row0 + V[1]*U_row1 + V[2]*U_row2
    vfloat32m1_t rotation_row_f32m1 = __riscv_vfmul_vf_f32m1(u_row0_f32m1, svd_v[0], vl3);
    rotation_row_f32m1 = __riscv_vfmacc_vf_f32m1(rotation_row_f32m1, svd_v[1], u_row1_f32m1, vl3);
    rotation_row_f32m1 = __riscv_vfmacc_vf_f32m1(rotation_row_f32m1, svd_v[2], u_row2_f32m1, vl3);
    __riscv_vse32_v_f32m1(r + 0, rotation_row_f32m1, vl3);
    // Row 1: R[3..5]
    rotation_row_f32m1 = __riscv_vfmul_vf_f32m1(u_row0_f32m1, svd_v[3], vl3);
    rotation_row_f32m1 = __riscv_vfmacc_vf_f32m1(rotation_row_f32m1, svd_v[4], u_row1_f32m1, vl3);
    rotation_row_f32m1 = __riscv_vfmacc_vf_f32m1(rotation_row_f32m1, svd_v[5], u_row2_f32m1, vl3);
    __riscv_vse32_v_f32m1(r + 3, rotation_row_f32m1, vl3);
    // Row 2: R[6..8]
    rotation_row_f32m1 = __riscv_vfmul_vf_f32m1(u_row0_f32m1, svd_v[6], vl3);
    rotation_row_f32m1 = __riscv_vfmacc_vf_f32m1(rotation_row_f32m1, svd_v[7], u_row1_f32m1, vl3);
    rotation_row_f32m1 = __riscv_vfmacc_vf_f32m1(rotation_row_f32m1, svd_v[8], u_row2_f32m1, vl3);
    __riscv_vse32_v_f32m1(r + 6, rotation_row_f32m1, vl3);
}

/** @brief Compute R = V * Uᵀ from SVD factors (f64), vectorized with `vfmul_vf`/`vfmacc_vf`. */
NK_INTERNAL void nk_rotation_from_svd_f64_rvv_( //
    nk_f64_t *svd_u, nk_f64_t *svd_v, nk_f64_t r[9]) {
    nk_rotation_from_svd_f64_serial_(svd_u, svd_v, r);
}

NK_PUBLIC void nk_rmsd_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    // Fused single-pass: accumulate centroids and squared differences simultaneously.
    // RMSD = √(E[(a−b)²] − (ā − b̄)²)
    nk_size_t vlmax = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t sum_a_x_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_a_y_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_a_z_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_b_x_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_b_y_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_b_z_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    vfloat64m2_t sum_squared_f64m2 = __riscv_vfmv_v_f_f64m2(0.0, vlmax);
    nk_f32_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
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
        // Accumulate centroids in f64.
        sum_a_x_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_a_x_f64m2, sum_a_x_f64m2, a_x_f32m1, vector_length);
        sum_a_y_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_a_y_f64m2, sum_a_y_f64m2, a_y_f32m1, vector_length);
        sum_a_z_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_a_z_f64m2, sum_a_z_f64m2, a_z_f32m1, vector_length);
        sum_b_x_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_b_x_f64m2, sum_b_x_f64m2, b_x_f32m1, vector_length);
        sum_b_y_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_b_y_f64m2, sum_b_y_f64m2, b_y_f32m1, vector_length);
        sum_b_z_f64m2 = __riscv_vfwadd_wv_f64m2_tu(sum_b_z_f64m2, sum_b_z_f64m2, b_z_f32m1, vector_length);
        // Accumulate (a−b)² per component. Widen a,b to f64 before subtracting to avoid f32
        // cancellation in the single-pass formula RMSD = √(E[(a−b)²] − (ā − b̄)²).
        vfloat64m2_t a_x_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(a_x_f32m1, vector_length);
        vfloat64m2_t b_x_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(b_x_f32m1, vector_length);
        vfloat64m2_t a_y_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(a_y_f32m1, vector_length);
        vfloat64m2_t b_y_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(b_y_f32m1, vector_length);
        vfloat64m2_t a_z_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(a_z_f32m1, vector_length);
        vfloat64m2_t b_z_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(b_z_f32m1, vector_length);
        vfloat64m2_t delta_x_f64m2 = __riscv_vfsub_vv_f64m2(a_x_f64m2, b_x_f64m2, vector_length);
        vfloat64m2_t delta_y_f64m2 = __riscv_vfsub_vv_f64m2(a_y_f64m2, b_y_f64m2, vector_length);
        vfloat64m2_t delta_z_f64m2 = __riscv_vfsub_vv_f64m2(a_z_f64m2, b_z_f64m2, vector_length);
        sum_squared_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_squared_f64m2, delta_x_f64m2, delta_x_f64m2, vector_length);
        sum_squared_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_squared_f64m2, delta_y_f64m2, delta_y_f64m2, vector_length);
        sum_squared_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_squared_f64m2, delta_z_f64m2, delta_z_f64m2, vector_length);
    }
    vfloat64m1_t zero_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, 1);
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t ca_x = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_a_x_f64m2, zero_f64m1, vlmax)) *
                    inv_n;
    nk_f64_t ca_y = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_a_y_f64m2, zero_f64m1, vlmax)) *
                    inv_n;
    nk_f64_t ca_z = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_a_z_f64m2, zero_f64m1, vlmax)) *
                    inv_n;
    nk_f64_t cb_x = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_b_x_f64m2, zero_f64m1, vlmax)) *
                    inv_n;
    nk_f64_t cb_y = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_b_y_f64m2, zero_f64m1, vlmax)) *
                    inv_n;
    nk_f64_t cb_z = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m2_f64m1(sum_b_z_f64m2, zero_f64m1, vlmax)) *
                    inv_n;
    if (a_centroid) a_centroid[0] = (nk_f32_t)ca_x, a_centroid[1] = (nk_f32_t)ca_y, a_centroid[2] = (nk_f32_t)ca_z;
    if (b_centroid) b_centroid[0] = (nk_f32_t)cb_x, b_centroid[1] = (nk_f32_t)cb_y, b_centroid[2] = (nk_f32_t)cb_z;

    nk_f64_t sum_squared = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m2_f64m1(sum_squared_f64m2, zero_f64m1, vlmax));
    nk_f64_t mean_diff_x = ca_x - cb_x, mean_diff_y = ca_y - cb_y, mean_diff_z = ca_z - cb_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;
    *result = nk_f64_sqrt_rvv(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_rmsd_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                               nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0;

    // Fused single-pass: accumulate centroids and squared differences simultaneously.
    // RMSD = √(E[(a−b)²] − (ā − b̄)²)
    nk_size_t vlmax = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t sum_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_a_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_a_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_a_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_b_x_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_b_y_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_b_z_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t sum_squared_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    vfloat64m1_t compensation_squared_f64m1 = __riscv_vfmv_v_f_f64m1(0.0, vlmax);
    nk_f64_t const *a_ptr = a, *b_ptr = b;
    nk_size_t remaining = n;
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
        // Accumulate centroids with Kahan compensation.
        nk_accumulate_sum_f64m1_rvv_(&sum_a_x_f64m1, &compensation_a_x_f64m1, a_x_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_a_y_f64m1, &compensation_a_y_f64m1, a_y_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_a_z_f64m1, &compensation_a_z_f64m1, a_z_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_b_x_f64m1, &compensation_b_x_f64m1, b_x_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_b_y_f64m1, &compensation_b_y_f64m1, b_y_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_b_z_f64m1, &compensation_b_z_f64m1, b_z_f64m1, vector_length);
        // Accumulate (a-b)^2 per component.
        vfloat64m1_t delta_x_f64m1 = __riscv_vfsub_vv_f64m1(a_x_f64m1, b_x_f64m1, vector_length);
        vfloat64m1_t delta_y_f64m1 = __riscv_vfsub_vv_f64m1(a_y_f64m1, b_y_f64m1, vector_length);
        vfloat64m1_t delta_z_f64m1 = __riscv_vfsub_vv_f64m1(a_z_f64m1, b_z_f64m1, vector_length);
        vfloat64m1_t dist_sq_f64m1 = __riscv_vfmul_vv_f64m1(delta_x_f64m1, delta_x_f64m1, vector_length);
        dist_sq_f64m1 = __riscv_vfmacc_vv_f64m1(dist_sq_f64m1, delta_y_f64m1, delta_y_f64m1, vector_length);
        dist_sq_f64m1 = __riscv_vfmacc_vv_f64m1(dist_sq_f64m1, delta_z_f64m1, delta_z_f64m1, vector_length);
        nk_accumulate_sum_f64m1_rvv_(&sum_squared_f64m1, &compensation_squared_f64m1, dist_sq_f64m1, vector_length);
    }
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t ca_x = nk_dot_stable_sum_f64m1_rvv_(sum_a_x_f64m1, compensation_a_x_f64m1) * inv_n;
    nk_f64_t ca_y = nk_dot_stable_sum_f64m1_rvv_(sum_a_y_f64m1, compensation_a_y_f64m1) * inv_n;
    nk_f64_t ca_z = nk_dot_stable_sum_f64m1_rvv_(sum_a_z_f64m1, compensation_a_z_f64m1) * inv_n;
    nk_f64_t cb_x = nk_dot_stable_sum_f64m1_rvv_(sum_b_x_f64m1, compensation_b_x_f64m1) * inv_n;
    nk_f64_t cb_y = nk_dot_stable_sum_f64m1_rvv_(sum_b_y_f64m1, compensation_b_y_f64m1) * inv_n;
    nk_f64_t cb_z = nk_dot_stable_sum_f64m1_rvv_(sum_b_z_f64m1, compensation_b_z_f64m1) * inv_n;
    if (a_centroid) a_centroid[0] = ca_x, a_centroid[1] = ca_y, a_centroid[2] = ca_z;
    if (b_centroid) b_centroid[0] = cb_x, b_centroid[1] = cb_y, b_centroid[2] = cb_z;

    nk_f64_t sum_squared = nk_dot_stable_sum_f64m1_rvv_(sum_squared_f64m1, compensation_squared_f64m1);
    nk_f64_t mean_diff_x = ca_x - cb_x, mean_diff_y = ca_y - cb_y, mean_diff_z = ca_z - cb_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;
    *result = nk_f64_sqrt_rvv(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                 nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    if (scale) *scale = 1.0f;
    nk_f64_t ca_x, ca_y, ca_z, cb_x, cb_y, cb_z;
    nk_f64_t h[9];
    nk_centroid_and_cross_covariance_f32_rvv_(a, b, n, &ca_x, &ca_y, &ca_z, &cb_x, &cb_y, &cb_z, h);
    if (a_centroid) a_centroid[0] = (nk_f32_t)ca_x, a_centroid[1] = (nk_f32_t)ca_y, a_centroid[2] = (nk_f32_t)ca_z;
    if (b_centroid) b_centroid[0] = (nk_f32_t)cb_x, b_centroid[1] = (nk_f32_t)cb_y, b_centroid[2] = (nk_f32_t)cb_z;
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(h, svd_u, svd_s, svd_v);
    nk_f64_t r[9];
    nk_rotation_from_svd_f64_rvv_(svd_u, svd_v, r);
    if (nk_det3x3_f64_(r) < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_rvv_(svd_u, svd_v, r);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f32_t)r[j];
    nk_f64_t ssd = nk_transformed_ssd_f32_rvv_(a, b, n, r, 1.0, ca_x, ca_y, ca_z, cb_x, cb_y, cb_z);
    *result = nk_f64_sqrt_rvv(ssd / (nk_f64_t)n);
}

NK_PUBLIC void nk_kabsch_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                 nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    if (scale) *scale = 1.0;
    nk_f64_t ca_x, ca_y, ca_z, cb_x, cb_y, cb_z;
    nk_f64_t h[9];
    nk_centroid_and_cross_covariance_f64_rvv_(a, b, n, &ca_x, &ca_y, &ca_z, &cb_x, &cb_y, &cb_z, h);
    if (a_centroid) a_centroid[0] = ca_x, a_centroid[1] = ca_y, a_centroid[2] = ca_z;
    if (b_centroid) b_centroid[0] = cb_x, b_centroid[1] = cb_y, b_centroid[2] = cb_z;
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(h, svd_u, svd_s, svd_v);
    nk_f64_t r[9];
    nk_rotation_from_svd_f64_rvv_(svd_u, svd_v, r);
    if (nk_det3x3_f64_(r) < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_rvv_(svd_u, svd_v, r);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    nk_f64_t ssd = nk_transformed_ssd_f64_rvv_(a, b, n, r, 1.0, ca_x, ca_y, ca_z, cb_x, cb_y, cb_z);
    *result = nk_f64_sqrt_rvv(ssd / (nk_f64_t)n);
}

NK_PUBLIC void nk_umeyama_f32_rvv(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    nk_f64_t ca_x, ca_y, ca_z, cb_x, cb_y, cb_z;
    nk_f64_t h[9], variance_a;
    nk_centroid_and_cross_covariance_and_variance_f32_rvv_(a, b, n, &ca_x, &ca_y, &ca_z, &cb_x, &cb_y, &cb_z, h,
                                                           &variance_a);
    if (a_centroid) a_centroid[0] = (nk_f32_t)ca_x, a_centroid[1] = (nk_f32_t)ca_y, a_centroid[2] = (nk_f32_t)ca_z;
    if (b_centroid) b_centroid[0] = (nk_f32_t)cb_x, b_centroid[1] = (nk_f32_t)cb_y, b_centroid[2] = (nk_f32_t)cb_z;
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(h, svd_u, svd_s, svd_v);
    nk_f64_t r[9];
    nk_rotation_from_svd_f64_rvv_(svd_u, svd_v, r);
    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t sign_det = det < 0 ? -1.0 : 1.0;
    nk_f64_t trace_ds = nk_sum_three_products_f64_(svd_s[0], 1.0, svd_s[4], 1.0, svd_s[8], sign_det);
    nk_f64_t scale_factor = trace_ds / ((nk_f64_t)n * variance_a);
    if (scale) *scale = (nk_f32_t)scale_factor;
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_rvv_(svd_u, svd_v, r);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = (nk_f32_t)r[j];
    nk_f64_t ssd = nk_transformed_ssd_f32_rvv_(a, b, n, r, scale_factor, ca_x, ca_y, ca_z, cb_x, cb_y, cb_z);
    *result = nk_f64_sqrt_rvv(ssd / (nk_f64_t)n);
}

NK_PUBLIC void nk_umeyama_f64_rvv(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                  nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    nk_f64_t ca_x, ca_y, ca_z, cb_x, cb_y, cb_z;
    nk_f64_t h[9], variance_a;
    nk_centroid_and_cross_covariance_and_variance_f64_rvv_(a, b, n, &ca_x, &ca_y, &ca_z, &cb_x, &cb_y, &cb_z, h,
                                                           &variance_a);
    if (a_centroid) a_centroid[0] = ca_x, a_centroid[1] = ca_y, a_centroid[2] = ca_z;
    if (b_centroid) b_centroid[0] = cb_x, b_centroid[1] = cb_y, b_centroid[2] = cb_z;
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(h, svd_u, svd_s, svd_v);
    nk_f64_t r[9];
    nk_rotation_from_svd_f64_rvv_(svd_u, svd_v, r);
    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t sign_det = det < 0 ? -1.0 : 1.0;
    nk_f64_t trace_ds = nk_sum_three_products_f64_(svd_s[0], 1.0, svd_s[4], 1.0, svd_s[8], sign_det);
    nk_f64_t scale_factor = trace_ds / ((nk_f64_t)n * variance_a);
    if (scale) *scale = scale_factor;
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_rvv_(svd_u, svd_v, r);
    }
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    nk_f64_t ssd = nk_transformed_ssd_f64_rvv_(a, b, n, r, scale_factor, ca_x, ca_y, ca_z, cb_x, cb_y, cb_z);
    *result = nk_f64_sqrt_rvv(ssd / (nk_f64_t)n);
}

NK_PUBLIC void nk_rmsd_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                               nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_rmsd_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_kabsch_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                 nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_kabsch_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_umeyama_f16_rvv(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_umeyama_f16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_rmsd_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_rmsd_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_kabsch_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_kabsch_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

NK_PUBLIC void nk_umeyama_bf16_rvv(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    nk_umeyama_bf16_serial(a, b, n, a_centroid, b_centroid, rotation, scale, result);
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NK_TARGET_RVV
#endif // NK_TARGET_RISCV_
#endif // NK_MESH_RVV_H
