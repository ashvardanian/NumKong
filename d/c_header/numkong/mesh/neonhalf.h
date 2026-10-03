/**
 *  @brief SIMD-accelerated Point Cloud Alignment for NEON FP16.
 *  @file include/numkong/mesh/neonhalf.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section mesh_neonhalf_instructions ARM NEON FP16 Instructions (ARMv8.2-FP16)
 *
 *      Intrinsic                   Instruction                     Latency     Throughput
 *                                                                              A76         M4+/V1+/Oryon
 *      vld3_f16                    LD3 (V.4H x 3)                  6cy         1/cy        2/cy
 *      vcvt_f32_f16                FCVTL (V.4S, V.4H)              3cy         2/cy        4/cy
 *      vfmaq_f32                   FMLA (V.4S, V.4S, V.4S)         4cy         2/cy        4/cy
 *      vaddq_f32                   FADD (V.4S, V.4S, V.4S)         2cy         2/cy        4/cy
 *      vsubq_f32                   FSUB (V.4S, V.4S, V.4S)         2cy         2/cy        4/cy
 *      vmulq_f32                   FMUL (V.4S, V.4S, V.4S)         3cy         2/cy        4/cy
 *      vdupq_n_f32                 DUP (V.4S, scalar)              2cy         2/cy        4/cy
 *      vaddvq_f32                  FADDP+FADDP (V.4S)              4cy         1/cy        2/cy
 *
 *  Mesh alignment algorithms (RMSD, Kabsch, Umeyama) for 3D point cloud registration using F16 input
 *  with F32 intermediate precision. VLD3 provides efficient stride-3 deinterleaving for XYZ triplets,
 *  then FCVTL widens to F32 for rotation matrix and centroid computations.
 *
 *  These algorithms compute optimal rigid body (Kabsch) or similarity (Umeyama) transformations
 *  between point sets, commonly used in structural biology (protein alignment) and computer vision.
 *  F16 storage halves memory for large point clouds while F32 arithmetic ensures numerical stability.
 */
#ifndef NK_MESH_NEONHALF_H
#define NK_MESH_NEONHALF_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEONHALF

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16")
#endif

NK_INTERNAL void nk_deinterleave_f16x4_to_f32x4_neonhalf_(nk_f16_t const *ptr, float32x4_t *x_out, float32x4_t *y_out,
                                                          float32x4_t *z_out) {
    // Deinterleave 12 f16 values (4 xyz triplets) into separate x, y, z vectors.
    // Uses NEON vld3_f16 for efficient stride-3 deinterleaving, then converts to f32.
    //
    // Input: 12 contiguous f16 values [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
    // Output: x[4], y[4], z[4] vectors in f32
    float16x4x3_t xyz = vld3_f16((nk_f16_for_arm_simd_t const *)ptr);
    *x_out = vcvt_f32_f16(xyz.val[0]);
    *y_out = vcvt_f32_f16(xyz.val[1]);
    *z_out = vcvt_f32_f16(xyz.val[2]);
}

NK_INTERNAL void nk_partial_deinterleave_f16_to_f32x4_neonhalf_(nk_f16_t const *ptr, nk_size_t n_points,
                                                                float32x4_t *x_out, float32x4_t *y_out,
                                                                float32x4_t *z_out) {
    nk_u16_t buf[12] = {0};
    nk_u16_t const *src = (nk_u16_t const *)ptr;
    for (nk_size_t k = 0; k < n_points * 3; ++k) buf[k] = src[k];
    nk_deinterleave_f16x4_to_f32x4_neonhalf_((nk_f16_t const *)buf, x_out, y_out, z_out);
}

NK_INTERNAL nk_f32_t nk_transformed_ssd_f16_neonhalf_(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n,
                                                      nk_f32_t const *r, nk_f32_t scale, nk_f32_t centroid_a_x,
                                                      nk_f32_t centroid_a_y, nk_f32_t centroid_a_z,
                                                      nk_f32_t centroid_b_x, nk_f32_t centroid_b_y,
                                                      nk_f32_t centroid_b_z) {
    // Compute sum of squared differences after rigid transformation.
    // Used by Kabsch algorithm for RMSD computation after rotation is applied.
    float32x4_t const centroid_a_x_f32x4 = vdupq_n_f32(centroid_a_x);
    float32x4_t const centroid_a_y_f32x4 = vdupq_n_f32(centroid_a_y);
    float32x4_t const centroid_a_z_f32x4 = vdupq_n_f32(centroid_a_z);
    float32x4_t const centroid_b_x_f32x4 = vdupq_n_f32(centroid_b_x);
    float32x4_t const centroid_b_y_f32x4 = vdupq_n_f32(centroid_b_y);
    float32x4_t const centroid_b_z_f32x4 = vdupq_n_f32(centroid_b_z);
    float32x4_t const scale_f32x4 = vdupq_n_f32(scale);

    // Load rotation matrix elements
    float32x4_t const r00_f32x4 = vdupq_n_f32(r[0]), r01_f32x4 = vdupq_n_f32(r[1]), r02_f32x4 = vdupq_n_f32(r[2]);
    float32x4_t const r10_f32x4 = vdupq_n_f32(r[3]), r11_f32x4 = vdupq_n_f32(r[4]), r12_f32x4 = vdupq_n_f32(r[5]);
    float32x4_t const r20_f32x4 = vdupq_n_f32(r[6]), r21_f32x4 = vdupq_n_f32(r[7]), r22_f32x4 = vdupq_n_f32(r[8]);

    float32x4_t sum_squared_f32x4 = vdupq_n_f32(0);
    float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;

    nk_size_t j = 0;
    for (; j + 4 <= n; j += 4) {
        nk_deinterleave_f16x4_to_f32x4_neonhalf_(a + j * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f16x4_to_f32x4_neonhalf_(b + j * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        // Center points
        float32x4_t pa_x_f32x4 = vsubq_f32(a_x_f32x4, centroid_a_x_f32x4);
        float32x4_t pa_y_f32x4 = vsubq_f32(a_y_f32x4, centroid_a_y_f32x4);
        float32x4_t pa_z_f32x4 = vsubq_f32(a_z_f32x4, centroid_a_z_f32x4);
        float32x4_t pb_x_f32x4 = vsubq_f32(b_x_f32x4, centroid_b_x_f32x4);
        float32x4_t pb_y_f32x4 = vsubq_f32(b_y_f32x4, centroid_b_y_f32x4);
        float32x4_t pb_z_f32x4 = vsubq_f32(b_z_f32x4, centroid_b_z_f32x4);

        // Apply rotation: R * pa (with optional scaling)
        float32x4_t ra_x_f32x4 = vmulq_f32(
            scale_f32x4,
            vfmaq_f32(vfmaq_f32(vmulq_f32(r00_f32x4, pa_x_f32x4), r01_f32x4, pa_y_f32x4), r02_f32x4, pa_z_f32x4));
        float32x4_t ra_y_f32x4 = vmulq_f32(
            scale_f32x4,
            vfmaq_f32(vfmaq_f32(vmulq_f32(r10_f32x4, pa_x_f32x4), r11_f32x4, pa_y_f32x4), r12_f32x4, pa_z_f32x4));
        float32x4_t ra_z_f32x4 = vmulq_f32(
            scale_f32x4,
            vfmaq_f32(vfmaq_f32(vmulq_f32(r20_f32x4, pa_x_f32x4), r21_f32x4, pa_y_f32x4), r22_f32x4, pa_z_f32x4));

        // Compute squared differences
        float32x4_t delta_x_f32x4 = vsubq_f32(ra_x_f32x4, pb_x_f32x4);
        float32x4_t delta_y_f32x4 = vsubq_f32(ra_y_f32x4, pb_y_f32x4);
        float32x4_t delta_z_f32x4 = vsubq_f32(ra_z_f32x4, pb_z_f32x4);

        sum_squared_f32x4 = vfmaq_f32(sum_squared_f32x4, delta_x_f32x4, delta_x_f32x4);
        sum_squared_f32x4 = vfmaq_f32(sum_squared_f32x4, delta_y_f32x4, delta_y_f32x4);
        sum_squared_f32x4 = vfmaq_f32(sum_squared_f32x4, delta_z_f32x4, delta_z_f32x4);
    }

    // Reduce to scalar
    nk_f32_t sum_squared = vaddvq_f32(sum_squared_f32x4);

    if (j < n) {
        float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_partial_deinterleave_f16_to_f32x4_neonhalf_(a + j * 3, n - j, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_partial_deinterleave_f16_to_f32x4_neonhalf_(b + j * 3, n - j, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        uint32x4_t lane_u32x4 = {0, 1, 2, 3};
        uint32x4_t valid_u32x4 = vcltq_u32(lane_u32x4, vdupq_n_u32((uint32_t)(n - j)));
        float32x4_t zero_f32x4 = vdupq_n_f32(0);
        a_x_f32x4 = vbslq_f32(valid_u32x4, a_x_f32x4, zero_f32x4);
        a_y_f32x4 = vbslq_f32(valid_u32x4, a_y_f32x4, zero_f32x4);
        a_z_f32x4 = vbslq_f32(valid_u32x4, a_z_f32x4, zero_f32x4);
        b_x_f32x4 = vbslq_f32(valid_u32x4, b_x_f32x4, zero_f32x4);
        b_y_f32x4 = vbslq_f32(valid_u32x4, b_y_f32x4, zero_f32x4);
        b_z_f32x4 = vbslq_f32(valid_u32x4, b_z_f32x4, zero_f32x4);

        float32x4_t pa_x_f32x4 = vsubq_f32(a_x_f32x4, centroid_a_x_f32x4);
        float32x4_t pa_y_f32x4 = vsubq_f32(a_y_f32x4, centroid_a_y_f32x4);
        float32x4_t pa_z_f32x4 = vsubq_f32(a_z_f32x4, centroid_a_z_f32x4);
        float32x4_t pb_x_f32x4 = vsubq_f32(b_x_f32x4, centroid_b_x_f32x4);
        float32x4_t pb_y_f32x4 = vsubq_f32(b_y_f32x4, centroid_b_y_f32x4);
        float32x4_t pb_z_f32x4 = vsubq_f32(b_z_f32x4, centroid_b_z_f32x4);

        float32x4_t ra_x_f32x4 = vmulq_f32(
            scale_f32x4,
            vfmaq_f32(vfmaq_f32(vmulq_f32(r00_f32x4, pa_x_f32x4), r01_f32x4, pa_y_f32x4), r02_f32x4, pa_z_f32x4));
        float32x4_t ra_y_f32x4 = vmulq_f32(
            scale_f32x4,
            vfmaq_f32(vfmaq_f32(vmulq_f32(r10_f32x4, pa_x_f32x4), r11_f32x4, pa_y_f32x4), r12_f32x4, pa_z_f32x4));
        float32x4_t ra_z_f32x4 = vmulq_f32(
            scale_f32x4,
            vfmaq_f32(vfmaq_f32(vmulq_f32(r20_f32x4, pa_x_f32x4), r21_f32x4, pa_y_f32x4), r22_f32x4, pa_z_f32x4));

        float32x4_t delta_x_f32x4 = vsubq_f32(ra_x_f32x4, pb_x_f32x4);
        float32x4_t delta_y_f32x4 = vsubq_f32(ra_y_f32x4, pb_y_f32x4);
        float32x4_t delta_z_f32x4 = vsubq_f32(ra_z_f32x4, pb_z_f32x4);

        float32x4_t tail_sum_f32x4 = vmulq_f32(delta_x_f32x4, delta_x_f32x4);
        tail_sum_f32x4 = vfmaq_f32(tail_sum_f32x4, delta_y_f32x4, delta_y_f32x4);
        tail_sum_f32x4 = vfmaq_f32(tail_sum_f32x4, delta_z_f32x4, delta_z_f32x4);
        sum_squared += vaddvq_f32(tail_sum_f32x4);
    }

    return sum_squared;
}

/**
 *  @brief RMSD (Root Mean Square Deviation) computation using NEON FP16 with widening to FP32.
 *  Computes the RMS of distances between corresponding points after centroid alignment.
 */
NK_PUBLIC void nk_rmsd_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                    nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // RMSD uses identity rotation and scale=1.0
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    // Accumulators for centroids and squared differences (all in f32)
    float32x4_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;
    float32x4_t sum_squared_x_f32x4 = zeros_f32x4, sum_squared_y_f32x4 = zeros_f32x4, sum_squared_z_f32x4 = zeros_f32x4;

    float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
    nk_size_t i = 0;

    // Main loop processing 4 points at a time
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f16x4_to_f32x4_neonhalf_(a + i * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f16x4_to_f32x4_neonhalf_(b + i * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);

        float32x4_t delta_x_f32x4 = vsubq_f32(a_x_f32x4, b_x_f32x4);
        float32x4_t delta_y_f32x4 = vsubq_f32(a_y_f32x4, b_y_f32x4);
        float32x4_t delta_z_f32x4 = vsubq_f32(a_z_f32x4, b_z_f32x4);

        sum_squared_x_f32x4 = vfmaq_f32(sum_squared_x_f32x4, delta_x_f32x4, delta_x_f32x4);
        sum_squared_y_f32x4 = vfmaq_f32(sum_squared_y_f32x4, delta_y_f32x4, delta_y_f32x4);
        sum_squared_z_f32x4 = vfmaq_f32(sum_squared_z_f32x4, delta_z_f32x4, delta_z_f32x4);
    }

    if (i < n) {
        nk_partial_deinterleave_f16_to_f32x4_neonhalf_(a + i * 3, n - i, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_partial_deinterleave_f16_to_f32x4_neonhalf_(b + i * 3, n - i, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);

        float32x4_t delta_x_f32x4 = vsubq_f32(a_x_f32x4, b_x_f32x4);
        float32x4_t delta_y_f32x4 = vsubq_f32(a_y_f32x4, b_y_f32x4);
        float32x4_t delta_z_f32x4 = vsubq_f32(a_z_f32x4, b_z_f32x4);

        sum_squared_x_f32x4 = vfmaq_f32(sum_squared_x_f32x4, delta_x_f32x4, delta_x_f32x4);
        sum_squared_y_f32x4 = vfmaq_f32(sum_squared_y_f32x4, delta_y_f32x4, delta_y_f32x4);
        sum_squared_z_f32x4 = vfmaq_f32(sum_squared_z_f32x4, delta_z_f32x4, delta_z_f32x4);
    }

    // Reduce vectors to scalars
    nk_f32_t total_ax = vaddvq_f32(sum_a_x_f32x4);
    nk_f32_t total_ay = vaddvq_f32(sum_a_y_f32x4);
    nk_f32_t total_az = vaddvq_f32(sum_a_z_f32x4);
    nk_f32_t total_bx = vaddvq_f32(sum_b_x_f32x4);
    nk_f32_t total_by = vaddvq_f32(sum_b_y_f32x4);
    nk_f32_t total_bz = vaddvq_f32(sum_b_z_f32x4);
    nk_f32_t total_sq_x = vaddvq_f32(sum_squared_x_f32x4);
    nk_f32_t total_sq_y = vaddvq_f32(sum_squared_y_f32x4);
    nk_f32_t total_sq_z = vaddvq_f32(sum_squared_z_f32x4);

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = total_ax * inv_n;
    nk_f32_t centroid_a_y = total_ay * inv_n;
    nk_f32_t centroid_a_z = total_az * inv_n;
    nk_f32_t centroid_b_x = total_bx * inv_n;
    nk_f32_t centroid_b_y = total_by * inv_n;
    nk_f32_t centroid_b_z = total_bz * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute RMSD
    nk_f32_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f32_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f32_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f32_t sum_squared = total_sq_x + total_sq_y + total_sq_z;
    nk_f32_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_f32_sqrt_neon(sum_squared * inv_n - mean_diff_sq);
}

/**
 *  @brief Kabsch algorithm for optimal rigid body superposition using NEON FP16 with widening to FP32.
 *  Finds the rotation matrix R that minimizes RMSD between two point sets.
 */
NK_PUBLIC void nk_kabsch_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load f16, convert to f32, compute centroids and covariance
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    // Accumulators for centroids (f32)
    float32x4_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;

    // Accumulators for covariance matrix (sum of outer products)
    float32x4_t cov_xx_f32x4 = zeros_f32x4, cov_xy_f32x4 = zeros_f32x4, cov_xz_f32x4 = zeros_f32x4;
    float32x4_t cov_yx_f32x4 = zeros_f32x4, cov_yy_f32x4 = zeros_f32x4, cov_yz_f32x4 = zeros_f32x4;
    float32x4_t cov_zx_f32x4 = zeros_f32x4, cov_zy_f32x4 = zeros_f32x4, cov_zz_f32x4 = zeros_f32x4;

    nk_size_t i = 0;
    float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;

    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f16x4_to_f32x4_neonhalf_(a + i * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f16x4_to_f32x4_neonhalf_(b + i * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        // Accumulate centroids
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);

        // Accumulate outer products
        cov_xx_f32x4 = vfmaq_f32(cov_xx_f32x4, a_x_f32x4, b_x_f32x4);
        cov_xy_f32x4 = vfmaq_f32(cov_xy_f32x4, a_x_f32x4, b_y_f32x4);
        cov_xz_f32x4 = vfmaq_f32(cov_xz_f32x4, a_x_f32x4, b_z_f32x4);
        cov_yx_f32x4 = vfmaq_f32(cov_yx_f32x4, a_y_f32x4, b_x_f32x4);
        cov_yy_f32x4 = vfmaq_f32(cov_yy_f32x4, a_y_f32x4, b_y_f32x4);
        cov_yz_f32x4 = vfmaq_f32(cov_yz_f32x4, a_y_f32x4, b_z_f32x4);
        cov_zx_f32x4 = vfmaq_f32(cov_zx_f32x4, a_z_f32x4, b_x_f32x4);
        cov_zy_f32x4 = vfmaq_f32(cov_zy_f32x4, a_z_f32x4, b_y_f32x4);
        cov_zz_f32x4 = vfmaq_f32(cov_zz_f32x4, a_z_f32x4, b_z_f32x4);
    }

    if (i < n) {
        nk_partial_deinterleave_f16_to_f32x4_neonhalf_(a + i * 3, n - i, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_partial_deinterleave_f16_to_f32x4_neonhalf_(b + i * 3, n - i, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);

        cov_xx_f32x4 = vfmaq_f32(cov_xx_f32x4, a_x_f32x4, b_x_f32x4);
        cov_xy_f32x4 = vfmaq_f32(cov_xy_f32x4, a_x_f32x4, b_y_f32x4);
        cov_xz_f32x4 = vfmaq_f32(cov_xz_f32x4, a_x_f32x4, b_z_f32x4);
        cov_yx_f32x4 = vfmaq_f32(cov_yx_f32x4, a_y_f32x4, b_x_f32x4);
        cov_yy_f32x4 = vfmaq_f32(cov_yy_f32x4, a_y_f32x4, b_y_f32x4);
        cov_yz_f32x4 = vfmaq_f32(cov_yz_f32x4, a_y_f32x4, b_z_f32x4);
        cov_zx_f32x4 = vfmaq_f32(cov_zx_f32x4, a_z_f32x4, b_x_f32x4);
        cov_zy_f32x4 = vfmaq_f32(cov_zy_f32x4, a_z_f32x4, b_y_f32x4);
        cov_zz_f32x4 = vfmaq_f32(cov_zz_f32x4, a_z_f32x4, b_z_f32x4);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = vaddvq_f32(sum_a_x_f32x4);
    nk_f32_t sum_a_y = vaddvq_f32(sum_a_y_f32x4);
    nk_f32_t sum_a_z = vaddvq_f32(sum_a_z_f32x4);
    nk_f32_t sum_b_x = vaddvq_f32(sum_b_x_f32x4);
    nk_f32_t sum_b_y = vaddvq_f32(sum_b_y_f32x4);
    nk_f32_t sum_b_z = vaddvq_f32(sum_b_z_f32x4);

    nk_f32_t covariance_x_x = vaddvq_f32(cov_xx_f32x4);
    nk_f32_t covariance_x_y = vaddvq_f32(cov_xy_f32x4);
    nk_f32_t covariance_x_z = vaddvq_f32(cov_xz_f32x4);
    nk_f32_t covariance_y_x = vaddvq_f32(cov_yx_f32x4);
    nk_f32_t covariance_y_y = vaddvq_f32(cov_yy_f32x4);
    nk_f32_t covariance_y_z = vaddvq_f32(cov_yz_f32x4);
    nk_f32_t covariance_z_x = vaddvq_f32(cov_zx_f32x4);
    nk_f32_t covariance_z_y = vaddvq_f32(cov_zy_f32x4);
    nk_f32_t covariance_z_z = vaddvq_f32(cov_zz_f32x4);

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n;
    nk_f32_t centroid_a_y = sum_a_y * inv_n;
    nk_f32_t centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n;
    nk_f32_t centroid_b_y = sum_b_y * inv_n;
    nk_f32_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) {
        a_centroid[0] = centroid_a_x;
        a_centroid[1] = centroid_a_y;
        a_centroid[2] = centroid_a_z;
    }
    if (b_centroid) {
        b_centroid[0] = centroid_b_x;
        b_centroid[1] = centroid_b_y;
        b_centroid[2] = centroid_b_z;
    }

    // Compute centered covariance: H = (A - centroid_A)ᵀ * (B - centroid_B)
    // H = sum(a * bᵀ) - n * centroid_a * centroid_bᵀ
    nk_f32_t h[9];
    h[0] = covariance_x_x - n * centroid_a_x * centroid_b_x;
    h[1] = covariance_x_y - n * centroid_a_x * centroid_b_y;
    h[2] = covariance_x_z - n * centroid_a_x * centroid_b_z;
    h[3] = covariance_y_x - n * centroid_a_y * centroid_b_x;
    h[4] = covariance_y_y - n * centroid_a_y * centroid_b_y;
    h[5] = covariance_y_z - n * centroid_a_y * centroid_b_z;
    h[6] = covariance_z_x - n * centroid_a_z * centroid_b_x;
    h[7] = covariance_z_y - n * centroid_a_z * centroid_b_y;
    h[8] = covariance_z_z - n * centroid_a_z * centroid_b_z;

    // SVD of H = U * S * Vᵀ
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(h, svd_u, svd_s, svd_v);

    // R = V * Uᵀ
    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection: if det(R) < 0, negate third column of V and recompute
    nk_f32_t det_r = nk_det3x3_f32_(r);
    if (det_r < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }
    if (scale) *scale = 1.0f;

    // Compute RMSD after rotation
    nk_f32_t sum_squared = nk_transformed_ssd_f16_neonhalf_(a, b, n, r, 1.0f, centroid_a_x, centroid_a_y, centroid_a_z,
                                                            centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f32_sqrt_neon(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f16_neonhalf(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                       nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // Fused single-pass: load f16, convert to f32, compute centroids, covariance, and variance
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    float32x4_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;
    float32x4_t cov_xx_f32x4 = zeros_f32x4, cov_xy_f32x4 = zeros_f32x4, cov_xz_f32x4 = zeros_f32x4;
    float32x4_t cov_yx_f32x4 = zeros_f32x4, cov_yy_f32x4 = zeros_f32x4, cov_yz_f32x4 = zeros_f32x4;
    float32x4_t cov_zx_f32x4 = zeros_f32x4, cov_zy_f32x4 = zeros_f32x4, cov_zz_f32x4 = zeros_f32x4;
    float32x4_t variance_a_f32x4 = zeros_f32x4;

    nk_size_t i = 0;
    float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;

    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f16x4_to_f32x4_neonhalf_(a + i * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_f16x4_to_f32x4_neonhalf_(b + i * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        // Accumulate centroids
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);

        // Accumulate outer products
        cov_xx_f32x4 = vfmaq_f32(cov_xx_f32x4, a_x_f32x4, b_x_f32x4);
        cov_xy_f32x4 = vfmaq_f32(cov_xy_f32x4, a_x_f32x4, b_y_f32x4);
        cov_xz_f32x4 = vfmaq_f32(cov_xz_f32x4, a_x_f32x4, b_z_f32x4);
        cov_yx_f32x4 = vfmaq_f32(cov_yx_f32x4, a_y_f32x4, b_x_f32x4);
        cov_yy_f32x4 = vfmaq_f32(cov_yy_f32x4, a_y_f32x4, b_y_f32x4);
        cov_yz_f32x4 = vfmaq_f32(cov_yz_f32x4, a_y_f32x4, b_z_f32x4);
        cov_zx_f32x4 = vfmaq_f32(cov_zx_f32x4, a_z_f32x4, b_x_f32x4);
        cov_zy_f32x4 = vfmaq_f32(cov_zy_f32x4, a_z_f32x4, b_y_f32x4);
        cov_zz_f32x4 = vfmaq_f32(cov_zz_f32x4, a_z_f32x4, b_z_f32x4);

        // Accumulate variance of A
        variance_a_f32x4 = vfmaq_f32(variance_a_f32x4, a_x_f32x4, a_x_f32x4);
        variance_a_f32x4 = vfmaq_f32(variance_a_f32x4, a_y_f32x4, a_y_f32x4);
        variance_a_f32x4 = vfmaq_f32(variance_a_f32x4, a_z_f32x4, a_z_f32x4);
    }

    if (i < n) {
        nk_partial_deinterleave_f16_to_f32x4_neonhalf_(a + i * 3, n - i, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_partial_deinterleave_f16_to_f32x4_neonhalf_(b + i * 3, n - i, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);

        cov_xx_f32x4 = vfmaq_f32(cov_xx_f32x4, a_x_f32x4, b_x_f32x4);
        cov_xy_f32x4 = vfmaq_f32(cov_xy_f32x4, a_x_f32x4, b_y_f32x4);
        cov_xz_f32x4 = vfmaq_f32(cov_xz_f32x4, a_x_f32x4, b_z_f32x4);
        cov_yx_f32x4 = vfmaq_f32(cov_yx_f32x4, a_y_f32x4, b_x_f32x4);
        cov_yy_f32x4 = vfmaq_f32(cov_yy_f32x4, a_y_f32x4, b_y_f32x4);
        cov_yz_f32x4 = vfmaq_f32(cov_yz_f32x4, a_y_f32x4, b_z_f32x4);
        cov_zx_f32x4 = vfmaq_f32(cov_zx_f32x4, a_z_f32x4, b_x_f32x4);
        cov_zy_f32x4 = vfmaq_f32(cov_zy_f32x4, a_z_f32x4, b_y_f32x4);
        cov_zz_f32x4 = vfmaq_f32(cov_zz_f32x4, a_z_f32x4, b_z_f32x4);

        variance_a_f32x4 = vfmaq_f32(variance_a_f32x4, a_x_f32x4, a_x_f32x4);
        variance_a_f32x4 = vfmaq_f32(variance_a_f32x4, a_y_f32x4, a_y_f32x4);
        variance_a_f32x4 = vfmaq_f32(variance_a_f32x4, a_z_f32x4, a_z_f32x4);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = vaddvq_f32(sum_a_x_f32x4);
    nk_f32_t sum_a_y = vaddvq_f32(sum_a_y_f32x4);
    nk_f32_t sum_a_z = vaddvq_f32(sum_a_z_f32x4);
    nk_f32_t sum_b_x = vaddvq_f32(sum_b_x_f32x4);
    nk_f32_t sum_b_y = vaddvq_f32(sum_b_y_f32x4);
    nk_f32_t sum_b_z = vaddvq_f32(sum_b_z_f32x4);
    nk_f32_t covariance_x_x = vaddvq_f32(cov_xx_f32x4);
    nk_f32_t covariance_x_y = vaddvq_f32(cov_xy_f32x4);
    nk_f32_t covariance_x_z = vaddvq_f32(cov_xz_f32x4);
    nk_f32_t covariance_y_x = vaddvq_f32(cov_yx_f32x4);
    nk_f32_t covariance_y_y = vaddvq_f32(cov_yy_f32x4);
    nk_f32_t covariance_y_z = vaddvq_f32(cov_yz_f32x4);
    nk_f32_t covariance_z_x = vaddvq_f32(cov_zx_f32x4);
    nk_f32_t covariance_z_y = vaddvq_f32(cov_zy_f32x4);
    nk_f32_t covariance_z_z = vaddvq_f32(cov_zz_f32x4);
    nk_f32_t variance_a_sum = vaddvq_f32(variance_a_f32x4);

    // Compute centroids
    nk_f32_t inv_n = 1.0f / (nk_f32_t)n;
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered covariance and variance
    nk_f32_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    nk_f32_t h[9];
    h[0] = covariance_x_x - n * centroid_a_x * centroid_b_x;
    h[1] = covariance_x_y - n * centroid_a_x * centroid_b_y;
    h[2] = covariance_x_z - n * centroid_a_x * centroid_b_z;
    h[3] = covariance_y_x - n * centroid_a_y * centroid_b_x;
    h[4] = covariance_y_y - n * centroid_a_y * centroid_b_y;
    h[5] = covariance_y_z - n * centroid_a_y * centroid_b_z;
    h[6] = covariance_z_x - n * centroid_a_z * centroid_b_x;
    h[7] = covariance_z_y - n * centroid_a_z * centroid_b_y;
    h[8] = covariance_z_z - n * centroid_a_z * centroid_b_z;

    // SVD of H = U * S * Vᵀ
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(h, svd_u, svd_s, svd_v);

    // R = V * Uᵀ
    nk_f32_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    // Handle reflection and compute scale: c = trace(D × S) / variance(a)
    nk_f32_t det_r = nk_det3x3_f32_(r);
    nk_f32_t sign_det = det_r < 0 ? -1.0f : 1.0f;
    nk_f32_t trace_scaled_s = svd_s[0] + svd_s[4] + sign_det * svd_s[8];
    nk_f32_t scale_factor = trace_scaled_s / ((nk_f32_t)n * variance_a);
    if (scale) *scale = scale_factor;

    if (det_r < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
        r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
        r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
        r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
        r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
        r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
        r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
        r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
        r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];
    }

    if (rotation) {
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    }

    // Compute RMSD after similarity transform
    nk_f32_t sum_squared = nk_transformed_ssd_f16_neonhalf_(a, b, n, r, scale_factor, centroid_a_x, centroid_a_y,
                                                            centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f32_sqrt_neon(sum_squared * inv_n);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEONHALF
#endif // NK_TARGET_ARM_
#endif // NK_MESH_NEONHALF_H
