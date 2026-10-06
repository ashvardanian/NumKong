/**
 *  @brief SIMD-accelerated Point Cloud Alignment for NEON.
 *  @file include/numkong/mesh/neon.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section neon_mesh_instructions Key NEON Mesh Instructions
 *
 *  Point cloud operations use these ARM NEON instructions:
 *
 *      Intrinsic         Instruction                   Latency     Throughput
 *                                                                  A76     M4+/V1+/Oryon
 *      vfmaq_f32         FMLA (V.4S, V.4S, V.4S)       4cy         2/cy    4/cy
 *      vmulq_n_f32       FMUL (V.4S, V.4S, V.S[0])     3cy         2/cy    4/cy
 *      vsubq_f32         FSUB (V.4S, V.4S, V.4S)       2cy         2/cy    4/cy
 *      vaddvq_f32        FADDP+FADDP (reduce)          5cy         1/cy    1/cy
 *      vld3q_f32         LD3 ({Vt.4S, Vt2.4S, Vt3.4S}) 6cy         1/cy    1/cy
 *
 *  LD3 provides hardware stride-3 deinterleaving for XYZ point data. The 6cy latency and
 *  1/cy throughput make it the memory bottleneck regardless of core microarchitecture.
 *
 *  FMA throughput doubles on 4-pipe cores (Apple M4+, Graviton3+, Oryon). Using 2x loop
 *  unrolling with independent accumulators hides FMA latency and saturates 2 FP pipes on
 *  A76-class cores; 4x unrolling may further benefit 4-pipe cores.
 */
#ifndef NK_MESH_NEON_H
#define NK_MESH_NEON_H

#if NK_TARGET_ARM_
#if NK_TARGET_NEON

#include "numkong/types.h"
#include "numkong/dot/neon.h"
#include "numkong/mesh/serial.h"
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`, `nk_f64_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

NK_INTERNAL void nk_deinterleave_f32x4_neon_(nk_f32_t const *ptr, float32x4_t *x_out, float32x4_t *y_out,
                                             float32x4_t *z_out) {
    // Deinterleave 12 floats (4 xyz triplets) into separate x, y, z vectors.
    // Uses NEON vld3q for efficient stride-3 deinterleaving.
    //
    // Input: 12 contiguous floats [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
    // Output: x[4], y[4], z[4] vectors
    float32x4x3_t xyz = vld3q_f32(ptr);
    *x_out = xyz.val[0];
    *y_out = xyz.val[1];
    *z_out = xyz.val[2];
}

NK_INTERNAL void nk_deinterleave_f64x2_neon_(nk_f64_t const *ptr, float64x2_t *x_out, float64x2_t *y_out,
                                             float64x2_t *z_out) {
    // Deinterleave 6 f64 values (2 xyz triplets) into separate x, y, z vectors.
    //
    // Input: 6 contiguous f64 [x0,y0,z0, x1,y1,z1]
    // Output: x[2], y[2], z[2] vectors
    // NEON doesn't have vld3q_f64, so we use vcombine to avoid stack round-trips
    // Load 2 xyz triplets: [x0,y0,z0, x1,y1,z1]
    *x_out = vcombine_f64(vld1_f64(&ptr[0]), vld1_f64(&ptr[3]));
    *y_out = vcombine_f64(vld1_f64(&ptr[1]), vld1_f64(&ptr[4]));
    *z_out = vcombine_f64(vld1_f64(&ptr[2]), vld1_f64(&ptr[5]));
}

NK_INTERNAL float64x2_t nk_promote_upper_f32x4_to_f64x2_neon_(float32x4_t values_f32x4) {
    return vcvt_f64_f32(vget_high_f32(values_f32x4));
}

NK_INTERNAL nk_f64_t nk_reduce_stable_f64x2_neon_(float64x2_t values_f64x2) {
    nk_b128_vec_t values;
    values.f64x2 = values_f64x2;
    nk_f64_t sum = 0.0, compensation = 0.0;
    nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[0]);
    nk_accumulate_sum_f64_(&sum, &compensation, values.f64s[1]);
    return sum + compensation;
}

NK_INTERNAL void nk_rotation_from_svd_f64_neon_(nk_f64_t const *svd_u, nk_f64_t const *svd_v, nk_f64_t *rotation) {
    nk_rotation_from_svd_f64_serial_(svd_u, svd_v, rotation);
}

NK_INTERNAL void nk_accumulate_square_f64x2_neon_(float64x2_t *sum_f64x2, float64x2_t *compensation_f64x2,
                                                  float64x2_t values_f64x2) {
    float64x2_t product_f64x2 = vmulq_f64(values_f64x2, values_f64x2);
    float64x2_t product_error_f64x2 = vfmaq_f64(vnegq_f64(product_f64x2), values_f64x2, values_f64x2);
    float64x2_t tentative_sum_f64x2 = vaddq_f64(*sum_f64x2, product_f64x2);
    float64x2_t virtual_addend_f64x2 = vsubq_f64(tentative_sum_f64x2, *sum_f64x2);
    float64x2_t sum_error_f64x2 = vaddq_f64(vsubq_f64(*sum_f64x2, vsubq_f64(tentative_sum_f64x2, virtual_addend_f64x2)),
                                            vsubq_f64(product_f64x2, virtual_addend_f64x2));
    *sum_f64x2 = tentative_sum_f64x2;
    *compensation_f64x2 = vaddq_f64(*compensation_f64x2, vaddq_f64(sum_error_f64x2, product_error_f64x2));
}

NK_INTERNAL nk_f64_t nk_transformed_ssd_f32_neon_( //
    nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f64_t const *r, nk_f64_t scale, nk_f64_t centroid_a_x,
    nk_f64_t centroid_a_y, nk_f64_t centroid_a_z, nk_f64_t centroid_b_x, nk_f64_t centroid_b_y, nk_f64_t centroid_b_z) {
    float64x2_t scaled_rotation_x_x_f64x2 = vdupq_n_f64(scale * r[0]);
    float64x2_t scaled_rotation_x_y_f64x2 = vdupq_n_f64(scale * r[1]);
    float64x2_t scaled_rotation_x_z_f64x2 = vdupq_n_f64(scale * r[2]);
    float64x2_t scaled_rotation_y_x_f64x2 = vdupq_n_f64(scale * r[3]);
    float64x2_t scaled_rotation_y_y_f64x2 = vdupq_n_f64(scale * r[4]);
    float64x2_t scaled_rotation_y_z_f64x2 = vdupq_n_f64(scale * r[5]);
    float64x2_t scaled_rotation_z_x_f64x2 = vdupq_n_f64(scale * r[6]);
    float64x2_t scaled_rotation_z_y_f64x2 = vdupq_n_f64(scale * r[7]);
    float64x2_t scaled_rotation_z_z_f64x2 = vdupq_n_f64(scale * r[8]);
    float64x2_t centroid_a_x_f64x2 = vdupq_n_f64(centroid_a_x), centroid_a_y_f64x2 = vdupq_n_f64(centroid_a_y);
    float64x2_t centroid_a_z_f64x2 = vdupq_n_f64(centroid_a_z), centroid_b_x_f64x2 = vdupq_n_f64(centroid_b_x);
    float64x2_t centroid_b_y_f64x2 = vdupq_n_f64(centroid_b_y), centroid_b_z_f64x2 = vdupq_n_f64(centroid_b_z);
    float64x2_t sum_squared_lower_f64x2 = vdupq_n_f64(0.0), sum_squared_upper_f64x2 = vdupq_n_f64(0.0);
    nk_size_t index = 0;

    for (; index + 4 <= n; index += 4) {
        float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_f32x4_neon_(a + index * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4),
            nk_deinterleave_f32x4_neon_(b + index * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        float64x2_t centered_a_x_lower_f64x2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(a_x_f32x4)), centroid_a_x_f64x2);
        float64x2_t centered_a_x_upper_f64x2 = vsubq_f64(nk_promote_upper_f32x4_to_f64x2_neon_(a_x_f32x4),
                                                         centroid_a_x_f64x2);
        float64x2_t centered_a_y_lower_f64x2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(a_y_f32x4)), centroid_a_y_f64x2);
        float64x2_t centered_a_y_upper_f64x2 = vsubq_f64(nk_promote_upper_f32x4_to_f64x2_neon_(a_y_f32x4),
                                                         centroid_a_y_f64x2);
        float64x2_t centered_a_z_lower_f64x2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(a_z_f32x4)), centroid_a_z_f64x2);
        float64x2_t centered_a_z_upper_f64x2 = vsubq_f64(nk_promote_upper_f32x4_to_f64x2_neon_(a_z_f32x4),
                                                         centroid_a_z_f64x2);
        float64x2_t centered_b_x_lower_f64x2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(b_x_f32x4)), centroid_b_x_f64x2);
        float64x2_t centered_b_x_upper_f64x2 = vsubq_f64(nk_promote_upper_f32x4_to_f64x2_neon_(b_x_f32x4),
                                                         centroid_b_x_f64x2);
        float64x2_t centered_b_y_lower_f64x2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(b_y_f32x4)), centroid_b_y_f64x2);
        float64x2_t centered_b_y_upper_f64x2 = vsubq_f64(nk_promote_upper_f32x4_to_f64x2_neon_(b_y_f32x4),
                                                         centroid_b_y_f64x2);
        float64x2_t centered_b_z_lower_f64x2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(b_z_f32x4)), centroid_b_z_f64x2);
        float64x2_t centered_b_z_upper_f64x2 = vsubq_f64(nk_promote_upper_f32x4_to_f64x2_neon_(b_z_f32x4),
                                                         centroid_b_z_f64x2);

        float64x2_t rotated_a_x_lower_f64x2 = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_x_x_f64x2, centered_a_x_lower_f64x2), scaled_rotation_x_y_f64x2,
                      centered_a_y_lower_f64x2),
            scaled_rotation_x_z_f64x2, centered_a_z_lower_f64x2);
        float64x2_t rotated_a_x_upper_f64x2 = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_x_x_f64x2, centered_a_x_upper_f64x2), scaled_rotation_x_y_f64x2,
                      centered_a_y_upper_f64x2),
            scaled_rotation_x_z_f64x2, centered_a_z_upper_f64x2);
        float64x2_t rotated_a_y_lower_f64x2 = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_y_x_f64x2, centered_a_x_lower_f64x2), scaled_rotation_y_y_f64x2,
                      centered_a_y_lower_f64x2),
            scaled_rotation_y_z_f64x2, centered_a_z_lower_f64x2);
        float64x2_t rotated_a_y_upper_f64x2 = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_y_x_f64x2, centered_a_x_upper_f64x2), scaled_rotation_y_y_f64x2,
                      centered_a_y_upper_f64x2),
            scaled_rotation_y_z_f64x2, centered_a_z_upper_f64x2);
        float64x2_t rotated_a_z_lower_f64x2 = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_z_x_f64x2, centered_a_x_lower_f64x2), scaled_rotation_z_y_f64x2,
                      centered_a_y_lower_f64x2),
            scaled_rotation_z_z_f64x2, centered_a_z_lower_f64x2);
        float64x2_t rotated_a_z_upper_f64x2 = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_z_x_f64x2, centered_a_x_upper_f64x2), scaled_rotation_z_y_f64x2,
                      centered_a_y_upper_f64x2),
            scaled_rotation_z_z_f64x2, centered_a_z_upper_f64x2);

        float64x2_t delta_x_lower_f64x2 = vsubq_f64(rotated_a_x_lower_f64x2, centered_b_x_lower_f64x2);
        float64x2_t delta_x_upper_f64x2 = vsubq_f64(rotated_a_x_upper_f64x2, centered_b_x_upper_f64x2);
        float64x2_t delta_y_lower_f64x2 = vsubq_f64(rotated_a_y_lower_f64x2, centered_b_y_lower_f64x2);
        float64x2_t delta_y_upper_f64x2 = vsubq_f64(rotated_a_y_upper_f64x2, centered_b_y_upper_f64x2);
        float64x2_t delta_z_lower_f64x2 = vsubq_f64(rotated_a_z_lower_f64x2, centered_b_z_lower_f64x2);
        float64x2_t delta_z_upper_f64x2 = vsubq_f64(rotated_a_z_upper_f64x2, centered_b_z_upper_f64x2);

        sum_squared_lower_f64x2 = vfmaq_f64(sum_squared_lower_f64x2, delta_x_lower_f64x2, delta_x_lower_f64x2),
        sum_squared_upper_f64x2 = vfmaq_f64(sum_squared_upper_f64x2, delta_x_upper_f64x2, delta_x_upper_f64x2);
        sum_squared_lower_f64x2 = vfmaq_f64(sum_squared_lower_f64x2, delta_y_lower_f64x2, delta_y_lower_f64x2),
        sum_squared_upper_f64x2 = vfmaq_f64(sum_squared_upper_f64x2, delta_y_upper_f64x2, delta_y_upper_f64x2);
        sum_squared_lower_f64x2 = vfmaq_f64(sum_squared_lower_f64x2, delta_z_lower_f64x2, delta_z_lower_f64x2),
        sum_squared_upper_f64x2 = vfmaq_f64(sum_squared_upper_f64x2, delta_z_upper_f64x2, delta_z_upper_f64x2);
    }

    nk_f64_t sum_squared = vaddvq_f64(vaddq_f64(sum_squared_lower_f64x2, sum_squared_upper_f64x2));
    for (; index < n; ++index) {
        nk_f64_t centered_a_x = (nk_f64_t)a[index * 3 + 0] - centroid_a_x;
        nk_f64_t centered_a_y = (nk_f64_t)a[index * 3 + 1] - centroid_a_y;
        nk_f64_t centered_a_z = (nk_f64_t)a[index * 3 + 2] - centroid_a_z;
        nk_f64_t centered_b_x = (nk_f64_t)b[index * 3 + 0] - centroid_b_x;
        nk_f64_t centered_b_y = (nk_f64_t)b[index * 3 + 1] - centroid_b_y;
        nk_f64_t centered_b_z = (nk_f64_t)b[index * 3 + 2] - centroid_b_z;
        nk_f64_t rotated_a_x = scale * (r[0] * centered_a_x + r[1] * centered_a_y + r[2] * centered_a_z);
        nk_f64_t rotated_a_y = scale * (r[3] * centered_a_x + r[4] * centered_a_y + r[5] * centered_a_z);
        nk_f64_t rotated_a_z = scale * (r[6] * centered_a_x + r[7] * centered_a_y + r[8] * centered_a_z);
        nk_f64_t delta_x = rotated_a_x - centered_b_x, delta_y = rotated_a_y - centered_b_y,
                 delta_z = rotated_a_z - centered_b_z;
        sum_squared += delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
    }

    return sum_squared;
}

/*  Compute sum of squared distances for f64 after applying rotation (and optional scale).
 *
 *  Optimization: 2x loop unrolling with multiple accumulators hides FMA latency (3-7 cycles).
 */
NK_INTERNAL nk_f64_t nk_transformed_ssd_f64_neon_(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t const *r,
                                                  nk_f64_t scale, nk_f64_t centroid_a_x, nk_f64_t centroid_a_y,
                                                  nk_f64_t centroid_a_z, nk_f64_t centroid_b_x, nk_f64_t centroid_b_y,
                                                  nk_f64_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    float64x2_t scaled_rotation_x_x_f64x2 = vdupq_n_f64(scale * r[0]);
    float64x2_t scaled_rotation_x_y_f64x2 = vdupq_n_f64(scale * r[1]);
    float64x2_t scaled_rotation_x_z_f64x2 = vdupq_n_f64(scale * r[2]);
    float64x2_t scaled_rotation_y_x_f64x2 = vdupq_n_f64(scale * r[3]);
    float64x2_t scaled_rotation_y_y_f64x2 = vdupq_n_f64(scale * r[4]);
    float64x2_t scaled_rotation_y_z_f64x2 = vdupq_n_f64(scale * r[5]);
    float64x2_t scaled_rotation_z_x_f64x2 = vdupq_n_f64(scale * r[6]);
    float64x2_t scaled_rotation_z_y_f64x2 = vdupq_n_f64(scale * r[7]);
    float64x2_t scaled_rotation_z_z_f64x2 = vdupq_n_f64(scale * r[8]);

    // Broadcast centroids
    float64x2_t centroid_a_x_f64x2 = vdupq_n_f64(centroid_a_x);
    float64x2_t centroid_a_y_f64x2 = vdupq_n_f64(centroid_a_y);
    float64x2_t centroid_a_z_f64x2 = vdupq_n_f64(centroid_a_z);
    float64x2_t centroid_b_x_f64x2 = vdupq_n_f64(centroid_b_x);
    float64x2_t centroid_b_y_f64x2 = vdupq_n_f64(centroid_b_y);
    float64x2_t centroid_b_z_f64x2 = vdupq_n_f64(centroid_b_z);

    // Two independent accumulators to hide FMA latency
    float64x2_t sum_squared_a_f64x2 = vdupq_n_f64(0), sum_squared_a_compensation_f64x2 = vdupq_n_f64(0);
    float64x2_t sum_squared_b_f64x2 = vdupq_n_f64(0), sum_squared_b_compensation_f64x2 = vdupq_n_f64(0);
    nk_size_t j = 0;

    // Main loop: process 4 points per iteration (2x unrolled, 2 points per batch)
    for (; j + 4 <= n; j += 4) {
        // First batch of 2 points
        float64x2_t a1_x, a1_y, a1_z, b1_x, b1_y, b1_z;
        nk_deinterleave_f64x2_neon_(a + j * 3, &a1_x, &a1_y, &a1_z);
        nk_deinterleave_f64x2_neon_(b + j * 3, &b1_x, &b1_y, &b1_z);

        // Second batch of 2 points
        float64x2_t a2_x, a2_y, a2_z, b2_x, b2_y, b2_z;
        nk_deinterleave_f64x2_neon_(a + (j + 2) * 3, &a2_x, &a2_y, &a2_z);
        nk_deinterleave_f64x2_neon_(b + (j + 2) * 3, &b2_x, &b2_y, &b2_z);

        // Center first batch
        float64x2_t pa1_x = vsubq_f64(a1_x, centroid_a_x_f64x2);
        float64x2_t pa1_y = vsubq_f64(a1_y, centroid_a_y_f64x2);
        float64x2_t pa1_z = vsubq_f64(a1_z, centroid_a_z_f64x2);
        float64x2_t pb1_x = vsubq_f64(b1_x, centroid_b_x_f64x2);
        float64x2_t pb1_y = vsubq_f64(b1_y, centroid_b_y_f64x2);
        float64x2_t pb1_z = vsubq_f64(b1_z, centroid_b_z_f64x2);

        // Center second batch
        float64x2_t pa2_x = vsubq_f64(a2_x, centroid_a_x_f64x2);
        float64x2_t pa2_y = vsubq_f64(a2_y, centroid_a_y_f64x2);
        float64x2_t pa2_z = vsubq_f64(a2_z, centroid_a_z_f64x2);
        float64x2_t pb2_x = vsubq_f64(b2_x, centroid_b_x_f64x2);
        float64x2_t pb2_y = vsubq_f64(b2_y, centroid_b_y_f64x2);
        float64x2_t pb2_z = vsubq_f64(b2_z, centroid_b_z_f64x2);

        // Rotate and scale first batch
        float64x2_t ra1_x = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_x_x_f64x2, pa1_x), scaled_rotation_x_y_f64x2, pa1_y),
            scaled_rotation_x_z_f64x2, pa1_z);
        float64x2_t ra1_y = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_y_x_f64x2, pa1_x), scaled_rotation_y_y_f64x2, pa1_y),
            scaled_rotation_y_z_f64x2, pa1_z);
        float64x2_t ra1_z = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_z_x_f64x2, pa1_x), scaled_rotation_z_y_f64x2, pa1_y),
            scaled_rotation_z_z_f64x2, pa1_z);

        // Rotate and scale second batch
        float64x2_t ra2_x = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_x_x_f64x2, pa2_x), scaled_rotation_x_y_f64x2, pa2_y),
            scaled_rotation_x_z_f64x2, pa2_z);
        float64x2_t ra2_y = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_y_x_f64x2, pa2_x), scaled_rotation_y_y_f64x2, pa2_y),
            scaled_rotation_y_z_f64x2, pa2_z);
        float64x2_t ra2_z = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_z_x_f64x2, pa2_x), scaled_rotation_z_y_f64x2, pa2_y),
            scaled_rotation_z_z_f64x2, pa2_z);

        // Deltas
        float64x2_t delta1_x = vsubq_f64(ra1_x, pb1_x);
        float64x2_t delta1_y = vsubq_f64(ra1_y, pb1_y);
        float64x2_t delta1_z = vsubq_f64(ra1_z, pb1_z);
        float64x2_t delta2_x = vsubq_f64(ra2_x, pb2_x);
        float64x2_t delta2_y = vsubq_f64(ra2_y, pb2_y);
        float64x2_t delta2_z = vsubq_f64(ra2_z, pb2_z);

        // Accumulate to independent accumulators (interleaved for latency hiding)
        nk_accumulate_square_f64x2_neon_(&sum_squared_a_f64x2, &sum_squared_a_compensation_f64x2, delta1_x);
        nk_accumulate_square_f64x2_neon_(&sum_squared_b_f64x2, &sum_squared_b_compensation_f64x2, delta2_x);
        nk_accumulate_square_f64x2_neon_(&sum_squared_a_f64x2, &sum_squared_a_compensation_f64x2, delta1_y);
        nk_accumulate_square_f64x2_neon_(&sum_squared_b_f64x2, &sum_squared_b_compensation_f64x2, delta2_y);
        nk_accumulate_square_f64x2_neon_(&sum_squared_a_f64x2, &sum_squared_a_compensation_f64x2, delta1_z);
        nk_accumulate_square_f64x2_neon_(&sum_squared_b_f64x2, &sum_squared_b_compensation_f64x2, delta2_z);
    }

    // Handle remaining 2 points
    if (j + 2 <= n) {
        float64x2_t a_x, a_y, a_z, b_x, b_y, b_z;
        nk_deinterleave_f64x2_neon_(a + j * 3, &a_x, &a_y, &a_z);
        nk_deinterleave_f64x2_neon_(b + j * 3, &b_x, &b_y, &b_z);

        float64x2_t pa_x = vsubq_f64(a_x, centroid_a_x_f64x2);
        float64x2_t pa_y = vsubq_f64(a_y, centroid_a_y_f64x2);
        float64x2_t pa_z = vsubq_f64(a_z, centroid_a_z_f64x2);
        float64x2_t pb_x = vsubq_f64(b_x, centroid_b_x_f64x2);
        float64x2_t pb_y = vsubq_f64(b_y, centroid_b_y_f64x2);
        float64x2_t pb_z = vsubq_f64(b_z, centroid_b_z_f64x2);

        float64x2_t ra_x = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_x_x_f64x2, pa_x), scaled_rotation_x_y_f64x2, pa_y),
            scaled_rotation_x_z_f64x2, pa_z);
        float64x2_t ra_y = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_y_x_f64x2, pa_x), scaled_rotation_y_y_f64x2, pa_y),
            scaled_rotation_y_z_f64x2, pa_z);
        float64x2_t ra_z = vfmaq_f64(
            vfmaq_f64(vmulq_f64(scaled_rotation_z_x_f64x2, pa_x), scaled_rotation_z_y_f64x2, pa_y),
            scaled_rotation_z_z_f64x2, pa_z);

        float64x2_t delta_x = vsubq_f64(ra_x, pb_x);
        float64x2_t delta_y = vsubq_f64(ra_y, pb_y);
        float64x2_t delta_z = vsubq_f64(ra_z, pb_z);

        nk_accumulate_square_f64x2_neon_(&sum_squared_a_f64x2, &sum_squared_a_compensation_f64x2, delta_x);
        nk_accumulate_square_f64x2_neon_(&sum_squared_a_f64x2, &sum_squared_a_compensation_f64x2, delta_y);
        nk_accumulate_square_f64x2_neon_(&sum_squared_a_f64x2, &sum_squared_a_compensation_f64x2, delta_z);
        j += 2;
    }

    // Combine accumulators and reduce
    float64x2_t sum_squared_f64x2 = vaddq_f64(sum_squared_a_f64x2, sum_squared_b_f64x2);
    float64x2_t sum_squared_compensation_f64x2 = vaddq_f64(sum_squared_a_compensation_f64x2,
                                                           sum_squared_b_compensation_f64x2);
    nk_f64_t sum_squared = nk_dot_stable_sum_f64x2_neon_(sum_squared_f64x2, sum_squared_compensation_f64x2);
    nk_f64_t sum_squared_compensation = 0.0;

    // Scalar tail
    for (; j < n; ++j) {
        nk_f64_t pa_x = a[j * 3 + 0] - centroid_a_x;
        nk_f64_t pa_y = a[j * 3 + 1] - centroid_a_y;
        nk_f64_t pa_z = a[j * 3 + 2] - centroid_a_z;
        nk_f64_t pb_x = b[j * 3 + 0] - centroid_b_x;
        nk_f64_t pb_y = b[j * 3 + 1] - centroid_b_y;
        nk_f64_t pb_z = b[j * 3 + 2] - centroid_b_z;

        nk_f64_t ra_x = scale * (r[0] * pa_x + r[1] * pa_y + r[2] * pa_z);
        nk_f64_t ra_y = scale * (r[3] * pa_x + r[4] * pa_y + r[5] * pa_z);
        nk_f64_t ra_z = scale * (r[6] * pa_x + r[7] * pa_y + r[8] * pa_z);

        nk_f64_t delta_x = ra_x - pb_x;
        nk_f64_t delta_y = ra_y - pb_y;
        nk_f64_t delta_z = ra_z - pb_z;
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, delta_x);
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, delta_y);
        nk_accumulate_square_f64_(&sum_squared, &sum_squared_compensation, delta_z);
    }

    return sum_squared + sum_squared_compensation;
}

NK_PUBLIC void nk_rmsd_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0f;

    float64x2_t zero_f64x2 = vdupq_n_f64(0.0);
    float64x2_t sum_a_x_lower_f64x2 = zero_f64x2, sum_a_x_upper_f64x2 = zero_f64x2;
    float64x2_t sum_a_y_lower_f64x2 = zero_f64x2, sum_a_y_upper_f64x2 = zero_f64x2;
    float64x2_t sum_a_z_lower_f64x2 = zero_f64x2, sum_a_z_upper_f64x2 = zero_f64x2;
    float64x2_t sum_b_x_lower_f64x2 = zero_f64x2, sum_b_x_upper_f64x2 = zero_f64x2;
    float64x2_t sum_b_y_lower_f64x2 = zero_f64x2, sum_b_y_upper_f64x2 = zero_f64x2;
    float64x2_t sum_b_z_lower_f64x2 = zero_f64x2, sum_b_z_upper_f64x2 = zero_f64x2;
    float64x2_t sum_squared_x_lower_f64x2 = zero_f64x2, sum_squared_x_upper_f64x2 = zero_f64x2;
    float64x2_t sum_squared_y_lower_f64x2 = zero_f64x2, sum_squared_y_upper_f64x2 = zero_f64x2;
    float64x2_t sum_squared_z_lower_f64x2 = zero_f64x2, sum_squared_z_upper_f64x2 = zero_f64x2;
    nk_size_t index = 0;

    for (; index + 4 <= n; index += 4) {
        float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_f32x4_neon_(a + index * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4),
            nk_deinterleave_f32x4_neon_(b + index * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        float64x2_t a_x_lower_f64x2 = vcvt_f64_f32(vget_low_f32(a_x_f32x4));
        float64x2_t a_x_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(a_x_f32x4);
        float64x2_t a_y_lower_f64x2 = vcvt_f64_f32(vget_low_f32(a_y_f32x4));
        float64x2_t a_y_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(a_y_f32x4);
        float64x2_t a_z_lower_f64x2 = vcvt_f64_f32(vget_low_f32(a_z_f32x4));
        float64x2_t a_z_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(a_z_f32x4);
        float64x2_t b_x_lower_f64x2 = vcvt_f64_f32(vget_low_f32(b_x_f32x4));
        float64x2_t b_x_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(b_x_f32x4);
        float64x2_t b_y_lower_f64x2 = vcvt_f64_f32(vget_low_f32(b_y_f32x4));
        float64x2_t b_y_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(b_y_f32x4);
        float64x2_t b_z_lower_f64x2 = vcvt_f64_f32(vget_low_f32(b_z_f32x4));
        float64x2_t b_z_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(b_z_f32x4);

        sum_a_x_lower_f64x2 = vaddq_f64(sum_a_x_lower_f64x2, a_x_lower_f64x2),
        sum_a_x_upper_f64x2 = vaddq_f64(sum_a_x_upper_f64x2, a_x_upper_f64x2);
        sum_a_y_lower_f64x2 = vaddq_f64(sum_a_y_lower_f64x2, a_y_lower_f64x2),
        sum_a_y_upper_f64x2 = vaddq_f64(sum_a_y_upper_f64x2, a_y_upper_f64x2);
        sum_a_z_lower_f64x2 = vaddq_f64(sum_a_z_lower_f64x2, a_z_lower_f64x2),
        sum_a_z_upper_f64x2 = vaddq_f64(sum_a_z_upper_f64x2, a_z_upper_f64x2);
        sum_b_x_lower_f64x2 = vaddq_f64(sum_b_x_lower_f64x2, b_x_lower_f64x2),
        sum_b_x_upper_f64x2 = vaddq_f64(sum_b_x_upper_f64x2, b_x_upper_f64x2);
        sum_b_y_lower_f64x2 = vaddq_f64(sum_b_y_lower_f64x2, b_y_lower_f64x2),
        sum_b_y_upper_f64x2 = vaddq_f64(sum_b_y_upper_f64x2, b_y_upper_f64x2);
        sum_b_z_lower_f64x2 = vaddq_f64(sum_b_z_lower_f64x2, b_z_lower_f64x2),
        sum_b_z_upper_f64x2 = vaddq_f64(sum_b_z_upper_f64x2, b_z_upper_f64x2);

        float64x2_t delta_x_lower_f64x2 = vsubq_f64(a_x_lower_f64x2, b_x_lower_f64x2);
        float64x2_t delta_x_upper_f64x2 = vsubq_f64(a_x_upper_f64x2, b_x_upper_f64x2);
        float64x2_t delta_y_lower_f64x2 = vsubq_f64(a_y_lower_f64x2, b_y_lower_f64x2);
        float64x2_t delta_y_upper_f64x2 = vsubq_f64(a_y_upper_f64x2, b_y_upper_f64x2);
        float64x2_t delta_z_lower_f64x2 = vsubq_f64(a_z_lower_f64x2, b_z_lower_f64x2);
        float64x2_t delta_z_upper_f64x2 = vsubq_f64(a_z_upper_f64x2, b_z_upper_f64x2);

        sum_squared_x_lower_f64x2 = vfmaq_f64(sum_squared_x_lower_f64x2, delta_x_lower_f64x2, delta_x_lower_f64x2),
        sum_squared_x_upper_f64x2 = vfmaq_f64(sum_squared_x_upper_f64x2, delta_x_upper_f64x2, delta_x_upper_f64x2);
        sum_squared_y_lower_f64x2 = vfmaq_f64(sum_squared_y_lower_f64x2, delta_y_lower_f64x2, delta_y_lower_f64x2),
        sum_squared_y_upper_f64x2 = vfmaq_f64(sum_squared_y_upper_f64x2, delta_y_upper_f64x2, delta_y_upper_f64x2);
        sum_squared_z_lower_f64x2 = vfmaq_f64(sum_squared_z_lower_f64x2, delta_z_lower_f64x2, delta_z_lower_f64x2),
        sum_squared_z_upper_f64x2 = vfmaq_f64(sum_squared_z_upper_f64x2, delta_z_upper_f64x2, delta_z_upper_f64x2);
    }

    nk_f64_t sum_a_x = vaddvq_f64(vaddq_f64(sum_a_x_lower_f64x2, sum_a_x_upper_f64x2));
    nk_f64_t sum_a_y = vaddvq_f64(vaddq_f64(sum_a_y_lower_f64x2, sum_a_y_upper_f64x2));
    nk_f64_t sum_a_z = vaddvq_f64(vaddq_f64(sum_a_z_lower_f64x2, sum_a_z_upper_f64x2));
    nk_f64_t sum_b_x = vaddvq_f64(vaddq_f64(sum_b_x_lower_f64x2, sum_b_x_upper_f64x2));
    nk_f64_t sum_b_y = vaddvq_f64(vaddq_f64(sum_b_y_lower_f64x2, sum_b_y_upper_f64x2));
    nk_f64_t sum_b_z = vaddvq_f64(vaddq_f64(sum_b_z_lower_f64x2, sum_b_z_upper_f64x2));
    nk_f64_t sum_squared_x = vaddvq_f64(vaddq_f64(sum_squared_x_lower_f64x2, sum_squared_x_upper_f64x2));
    nk_f64_t sum_squared_y = vaddvq_f64(vaddq_f64(sum_squared_y_lower_f64x2, sum_squared_y_upper_f64x2));
    nk_f64_t sum_squared_z = vaddvq_f64(vaddq_f64(sum_squared_z_lower_f64x2, sum_squared_z_upper_f64x2));

    for (; index < n; ++index) {
        nk_f64_t a_x = a[index * 3 + 0], a_y = a[index * 3 + 1], a_z = a[index * 3 + 2];
        nk_f64_t b_x = b[index * 3 + 0], b_y = b[index * 3 + 1], b_z = b[index * 3 + 2];
        sum_a_x += a_x, sum_a_y += a_y, sum_a_z += a_z;
        sum_b_x += b_x, sum_b_y += b_y, sum_b_z += b_z;
        nk_f64_t delta_x = a_x - b_x, delta_y = a_y - b_y, delta_z = a_z - b_z;
        sum_squared_x += delta_x * delta_x, sum_squared_y += delta_y * delta_y, sum_squared_z += delta_z * delta_z;
    }

    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    nk_f64_t mean_diff_x = centroid_a_x - centroid_b_x, mean_diff_y = centroid_a_y - centroid_b_y,
             mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;
    *result = nk_f64_sqrt_neon((sum_squared_x + sum_squared_y + sum_squared_z) * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_rmsd_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    // RMSD uses identity rotation and scale=1.0.
    if (rotation) {
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0;
        rotation[3] = 0, rotation[4] = 1, rotation[5] = 0;
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    }
    if (scale) *scale = 1.0;

    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);

    // Accumulators for centroids and squared differences
    float64x2_t sum_a_x_f64x2 = zeros_f64x2, sum_a_y_f64x2 = zeros_f64x2, sum_a_z_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_f64x2 = zeros_f64x2, sum_b_y_f64x2 = zeros_f64x2, sum_b_z_f64x2 = zeros_f64x2;
    float64x2_t sum_squared_x_f64x2 = zeros_f64x2, sum_squared_y_f64x2 = zeros_f64x2, sum_squared_z_f64x2 = zeros_f64x2;

    float64x2_t a_x_f64x2, a_y_f64x2, a_z_f64x2, b_x_f64x2, b_y_f64x2, b_z_f64x2;
    nk_size_t i = 0;

    // Main loop processing 2 points at a time
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a_x_f64x2, &a_y_f64x2, &a_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b_x_f64x2, &b_y_f64x2, &b_z_f64x2);

        sum_a_x_f64x2 = vaddq_f64(sum_a_x_f64x2, a_x_f64x2);
        sum_a_y_f64x2 = vaddq_f64(sum_a_y_f64x2, a_y_f64x2);
        sum_a_z_f64x2 = vaddq_f64(sum_a_z_f64x2, a_z_f64x2);
        sum_b_x_f64x2 = vaddq_f64(sum_b_x_f64x2, b_x_f64x2);
        sum_b_y_f64x2 = vaddq_f64(sum_b_y_f64x2, b_y_f64x2);
        sum_b_z_f64x2 = vaddq_f64(sum_b_z_f64x2, b_z_f64x2);

        float64x2_t delta_x_f64x2 = vsubq_f64(a_x_f64x2, b_x_f64x2);
        float64x2_t delta_y_f64x2 = vsubq_f64(a_y_f64x2, b_y_f64x2);
        float64x2_t delta_z_f64x2 = vsubq_f64(a_z_f64x2, b_z_f64x2);

        sum_squared_x_f64x2 = vfmaq_f64(sum_squared_x_f64x2, delta_x_f64x2, delta_x_f64x2);
        sum_squared_y_f64x2 = vfmaq_f64(sum_squared_y_f64x2, delta_y_f64x2, delta_y_f64x2);
        sum_squared_z_f64x2 = vfmaq_f64(sum_squared_z_f64x2, delta_z_f64x2, delta_z_f64x2);
    }

    // Reduce vectors to scalars.
    nk_f64_t total_ax = nk_reduce_stable_f64x2_neon_(sum_a_x_f64x2), total_ax_compensation = 0.0;
    nk_f64_t total_ay = nk_reduce_stable_f64x2_neon_(sum_a_y_f64x2), total_ay_compensation = 0.0;
    nk_f64_t total_az = nk_reduce_stable_f64x2_neon_(sum_a_z_f64x2), total_az_compensation = 0.0;
    nk_f64_t total_bx = nk_reduce_stable_f64x2_neon_(sum_b_x_f64x2), total_bx_compensation = 0.0;
    nk_f64_t total_by = nk_reduce_stable_f64x2_neon_(sum_b_y_f64x2), total_by_compensation = 0.0;
    nk_f64_t total_bz = nk_reduce_stable_f64x2_neon_(sum_b_z_f64x2), total_bz_compensation = 0.0;
    nk_f64_t total_squared_x = nk_reduce_stable_f64x2_neon_(sum_squared_x_f64x2), total_squared_x_compensation = 0.0;
    nk_f64_t total_squared_y = nk_reduce_stable_f64x2_neon_(sum_squared_y_f64x2), total_squared_y_compensation = 0.0;
    nk_f64_t total_squared_z = nk_reduce_stable_f64x2_neon_(sum_squared_z_f64x2), total_squared_z_compensation = 0.0;

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&total_ax, &total_ax_compensation, ax);
        nk_accumulate_sum_f64_(&total_ay, &total_ay_compensation, ay);
        nk_accumulate_sum_f64_(&total_az, &total_az_compensation, az);
        nk_accumulate_sum_f64_(&total_bx, &total_bx_compensation, bx);
        nk_accumulate_sum_f64_(&total_by, &total_by_compensation, by);
        nk_accumulate_sum_f64_(&total_bz, &total_bz_compensation, bz);
        nk_f64_t delta_x = ax - bx, delta_y = ay - by, delta_z = az - bz;
        nk_accumulate_square_f64_(&total_squared_x, &total_squared_x_compensation, delta_x);
        nk_accumulate_square_f64_(&total_squared_y, &total_squared_y_compensation, delta_y);
        nk_accumulate_square_f64_(&total_squared_z, &total_squared_z_compensation, delta_z);
    }

    total_ax += total_ax_compensation, total_ay += total_ay_compensation, total_az += total_az_compensation;
    total_bx += total_bx_compensation, total_by += total_by_compensation, total_bz += total_bz_compensation;
    total_squared_x += total_squared_x_compensation, total_squared_y += total_squared_y_compensation,
        total_squared_z += total_squared_z_compensation;

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = total_ax * inv_n, centroid_a_y = total_ay * inv_n, centroid_a_z = total_az * inv_n;
    nk_f64_t centroid_b_x = total_bx * inv_n, centroid_b_y = total_by * inv_n, centroid_b_z = total_bz * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute RMSD
    nk_f64_t mean_diff_x = centroid_a_x - centroid_b_x;
    nk_f64_t mean_diff_y = centroid_a_y - centroid_b_y;
    nk_f64_t mean_diff_z = centroid_a_z - centroid_b_z;
    nk_f64_t sum_squared = total_squared_x + total_squared_y + total_squared_z;
    nk_f64_t mean_diff_sq = mean_diff_x * mean_diff_x + mean_diff_y * mean_diff_y + mean_diff_z * mean_diff_z;

    *result = nk_f64_sqrt_neon(sum_squared * inv_n - mean_diff_sq);
}

NK_PUBLIC void nk_kabsch_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                  nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    float64x2_t zero_f64x2 = vdupq_n_f64(0.0);

    // Centroid accumulators (f64, lower/upper halves of f32x4)
    float64x2_t sum_a_x_lower_f64x2 = zero_f64x2, sum_a_x_upper_f64x2 = zero_f64x2;
    float64x2_t sum_a_y_lower_f64x2 = zero_f64x2, sum_a_y_upper_f64x2 = zero_f64x2;
    float64x2_t sum_a_z_lower_f64x2 = zero_f64x2, sum_a_z_upper_f64x2 = zero_f64x2;
    float64x2_t sum_b_x_lower_f64x2 = zero_f64x2, sum_b_x_upper_f64x2 = zero_f64x2;
    float64x2_t sum_b_y_lower_f64x2 = zero_f64x2, sum_b_y_upper_f64x2 = zero_f64x2;
    float64x2_t sum_b_z_lower_f64x2 = zero_f64x2, sum_b_z_upper_f64x2 = zero_f64x2;

    // Covariance accumulators (f64, lower/upper halves)
    float64x2_t cov_xx_lower_f64x2 = zero_f64x2, cov_xx_upper_f64x2 = zero_f64x2;
    float64x2_t cov_xy_lower_f64x2 = zero_f64x2, cov_xy_upper_f64x2 = zero_f64x2;
    float64x2_t cov_xz_lower_f64x2 = zero_f64x2, cov_xz_upper_f64x2 = zero_f64x2;
    float64x2_t cov_yx_lower_f64x2 = zero_f64x2, cov_yx_upper_f64x2 = zero_f64x2;
    float64x2_t cov_yy_lower_f64x2 = zero_f64x2, cov_yy_upper_f64x2 = zero_f64x2;
    float64x2_t cov_yz_lower_f64x2 = zero_f64x2, cov_yz_upper_f64x2 = zero_f64x2;
    float64x2_t cov_zx_lower_f64x2 = zero_f64x2, cov_zx_upper_f64x2 = zero_f64x2;
    float64x2_t cov_zy_lower_f64x2 = zero_f64x2, cov_zy_upper_f64x2 = zero_f64x2;
    float64x2_t cov_zz_lower_f64x2 = zero_f64x2, cov_zz_upper_f64x2 = zero_f64x2;

    nk_size_t index = 0;
    for (; index + 4 <= n; index += 4) {
        float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_f32x4_neon_(a + index * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4),
            nk_deinterleave_f32x4_neon_(b + index * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        float64x2_t a_x_lower_f64x2 = vcvt_f64_f32(vget_low_f32(a_x_f32x4));
        float64x2_t a_x_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(a_x_f32x4);
        float64x2_t a_y_lower_f64x2 = vcvt_f64_f32(vget_low_f32(a_y_f32x4));
        float64x2_t a_y_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(a_y_f32x4);
        float64x2_t a_z_lower_f64x2 = vcvt_f64_f32(vget_low_f32(a_z_f32x4));
        float64x2_t a_z_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(a_z_f32x4);
        float64x2_t b_x_lower_f64x2 = vcvt_f64_f32(vget_low_f32(b_x_f32x4));
        float64x2_t b_x_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(b_x_f32x4);
        float64x2_t b_y_lower_f64x2 = vcvt_f64_f32(vget_low_f32(b_y_f32x4));
        float64x2_t b_y_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(b_y_f32x4);
        float64x2_t b_z_lower_f64x2 = vcvt_f64_f32(vget_low_f32(b_z_f32x4));
        float64x2_t b_z_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(b_z_f32x4);

        // Accumulate centroids
        sum_a_x_lower_f64x2 = vaddq_f64(sum_a_x_lower_f64x2, a_x_lower_f64x2),
        sum_a_x_upper_f64x2 = vaddq_f64(sum_a_x_upper_f64x2, a_x_upper_f64x2);
        sum_a_y_lower_f64x2 = vaddq_f64(sum_a_y_lower_f64x2, a_y_lower_f64x2),
        sum_a_y_upper_f64x2 = vaddq_f64(sum_a_y_upper_f64x2, a_y_upper_f64x2);
        sum_a_z_lower_f64x2 = vaddq_f64(sum_a_z_lower_f64x2, a_z_lower_f64x2),
        sum_a_z_upper_f64x2 = vaddq_f64(sum_a_z_upper_f64x2, a_z_upper_f64x2);
        sum_b_x_lower_f64x2 = vaddq_f64(sum_b_x_lower_f64x2, b_x_lower_f64x2),
        sum_b_x_upper_f64x2 = vaddq_f64(sum_b_x_upper_f64x2, b_x_upper_f64x2);
        sum_b_y_lower_f64x2 = vaddq_f64(sum_b_y_lower_f64x2, b_y_lower_f64x2),
        sum_b_y_upper_f64x2 = vaddq_f64(sum_b_y_upper_f64x2, b_y_upper_f64x2);
        sum_b_z_lower_f64x2 = vaddq_f64(sum_b_z_lower_f64x2, b_z_lower_f64x2),
        sum_b_z_upper_f64x2 = vaddq_f64(sum_b_z_upper_f64x2, b_z_upper_f64x2);

        // Accumulate raw outer products (uncentered)
        cov_xx_lower_f64x2 = vfmaq_f64(cov_xx_lower_f64x2, a_x_lower_f64x2, b_x_lower_f64x2),
        cov_xx_upper_f64x2 = vfmaq_f64(cov_xx_upper_f64x2, a_x_upper_f64x2, b_x_upper_f64x2);
        cov_xy_lower_f64x2 = vfmaq_f64(cov_xy_lower_f64x2, a_x_lower_f64x2, b_y_lower_f64x2),
        cov_xy_upper_f64x2 = vfmaq_f64(cov_xy_upper_f64x2, a_x_upper_f64x2, b_y_upper_f64x2);
        cov_xz_lower_f64x2 = vfmaq_f64(cov_xz_lower_f64x2, a_x_lower_f64x2, b_z_lower_f64x2),
        cov_xz_upper_f64x2 = vfmaq_f64(cov_xz_upper_f64x2, a_x_upper_f64x2, b_z_upper_f64x2);
        cov_yx_lower_f64x2 = vfmaq_f64(cov_yx_lower_f64x2, a_y_lower_f64x2, b_x_lower_f64x2),
        cov_yx_upper_f64x2 = vfmaq_f64(cov_yx_upper_f64x2, a_y_upper_f64x2, b_x_upper_f64x2);
        cov_yy_lower_f64x2 = vfmaq_f64(cov_yy_lower_f64x2, a_y_lower_f64x2, b_y_lower_f64x2),
        cov_yy_upper_f64x2 = vfmaq_f64(cov_yy_upper_f64x2, a_y_upper_f64x2, b_y_upper_f64x2);
        cov_yz_lower_f64x2 = vfmaq_f64(cov_yz_lower_f64x2, a_y_lower_f64x2, b_z_lower_f64x2),
        cov_yz_upper_f64x2 = vfmaq_f64(cov_yz_upper_f64x2, a_y_upper_f64x2, b_z_upper_f64x2);
        cov_zx_lower_f64x2 = vfmaq_f64(cov_zx_lower_f64x2, a_z_lower_f64x2, b_x_lower_f64x2),
        cov_zx_upper_f64x2 = vfmaq_f64(cov_zx_upper_f64x2, a_z_upper_f64x2, b_x_upper_f64x2);
        cov_zy_lower_f64x2 = vfmaq_f64(cov_zy_lower_f64x2, a_z_lower_f64x2, b_y_lower_f64x2),
        cov_zy_upper_f64x2 = vfmaq_f64(cov_zy_upper_f64x2, a_z_upper_f64x2, b_y_upper_f64x2);
        cov_zz_lower_f64x2 = vfmaq_f64(cov_zz_lower_f64x2, a_z_lower_f64x2, b_z_lower_f64x2),
        cov_zz_upper_f64x2 = vfmaq_f64(cov_zz_upper_f64x2, a_z_upper_f64x2, b_z_upper_f64x2);
    }

    // Reduce centroid accumulators
    nk_f64_t sum_a_x = vaddvq_f64(vaddq_f64(sum_a_x_lower_f64x2, sum_a_x_upper_f64x2));
    nk_f64_t sum_a_y = vaddvq_f64(vaddq_f64(sum_a_y_lower_f64x2, sum_a_y_upper_f64x2));
    nk_f64_t sum_a_z = vaddvq_f64(vaddq_f64(sum_a_z_lower_f64x2, sum_a_z_upper_f64x2));
    nk_f64_t sum_b_x = vaddvq_f64(vaddq_f64(sum_b_x_lower_f64x2, sum_b_x_upper_f64x2));
    nk_f64_t sum_b_y = vaddvq_f64(vaddq_f64(sum_b_y_lower_f64x2, sum_b_y_upper_f64x2));
    nk_f64_t sum_b_z = vaddvq_f64(vaddq_f64(sum_b_z_lower_f64x2, sum_b_z_upper_f64x2));

    // Reduce covariance accumulators
    nk_f64_t covariance_x_x = vaddvq_f64(vaddq_f64(cov_xx_lower_f64x2, cov_xx_upper_f64x2));
    nk_f64_t covariance_x_y = vaddvq_f64(vaddq_f64(cov_xy_lower_f64x2, cov_xy_upper_f64x2));
    nk_f64_t covariance_x_z = vaddvq_f64(vaddq_f64(cov_xz_lower_f64x2, cov_xz_upper_f64x2));
    nk_f64_t covariance_y_x = vaddvq_f64(vaddq_f64(cov_yx_lower_f64x2, cov_yx_upper_f64x2));
    nk_f64_t covariance_y_y = vaddvq_f64(vaddq_f64(cov_yy_lower_f64x2, cov_yy_upper_f64x2));
    nk_f64_t covariance_y_z = vaddvq_f64(vaddq_f64(cov_yz_lower_f64x2, cov_yz_upper_f64x2));
    nk_f64_t covariance_z_x = vaddvq_f64(vaddq_f64(cov_zx_lower_f64x2, cov_zx_upper_f64x2));
    nk_f64_t covariance_z_y = vaddvq_f64(vaddq_f64(cov_zy_lower_f64x2, cov_zy_upper_f64x2));
    nk_f64_t covariance_z_z = vaddvq_f64(vaddq_f64(cov_zz_lower_f64x2, cov_zz_upper_f64x2));

    // Scalar tail
    for (; index < n; ++index) {
        nk_f64_t ax = (nk_f64_t)a[index * 3 + 0], ay = (nk_f64_t)a[index * 3 + 1], az = (nk_f64_t)a[index * 3 + 2];
        nk_f64_t bx = (nk_f64_t)b[index * 3 + 0], by = (nk_f64_t)b[index * 3 + 1], bz = (nk_f64_t)b[index * 3 + 2];
        sum_a_x += ax, sum_a_y += ay, sum_a_z += az;
        sum_b_x += bx, sum_b_y += by, sum_b_z += bz;
        covariance_x_x += ax * bx, covariance_x_y += ax * by, covariance_x_z += ax * bz;
        covariance_y_x += ay * bx, covariance_y_y += ay * by, covariance_y_z += ay * bz;
        covariance_z_x += az * bx, covariance_z_y += az * by, covariance_z_z += az * bz;
    }

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    // Apply centering correction: H_centered = sum(a * bᵀ) - n * centroid_a * centroid_bᵀ
    nk_f64_t h[9];
    h[0] = covariance_x_x - n * centroid_a_x * centroid_b_x;
    h[1] = covariance_x_y - n * centroid_a_x * centroid_b_y;
    h[2] = covariance_x_z - n * centroid_a_x * centroid_b_z;
    h[3] = covariance_y_x - n * centroid_a_y * centroid_b_x;
    h[4] = covariance_y_y - n * centroid_a_y * centroid_b_y;
    h[5] = covariance_y_z - n * centroid_a_y * centroid_b_z;
    h[6] = covariance_z_x - n * centroid_a_z * centroid_b_x;
    h[7] = covariance_z_y - n * centroid_a_z * centroid_b_y;
    h[8] = covariance_z_z - n * centroid_a_z * centroid_b_z;

    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(h, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    if (nk_det3x3_f64_(r) < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
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

    if (rotation)
        for (int j = 0; j != 9; ++j) rotation[j] = (nk_f32_t)r[j];
    if (scale) *scale = 1.0f;
    *result = nk_f64_sqrt_neon(nk_transformed_ssd_f32_neon_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y, centroid_a_z,
                                                            centroid_b_x, centroid_b_y, centroid_b_z) /
                               (nk_f64_t)n);
}

NK_PUBLIC void nk_kabsch_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                  nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);

    // 2x unrolling with dual accumulators to hide FMA latency.
    float64x2_t sum_a_x_a_f64x2 = zeros_f64x2, sum_a_y_a_f64x2 = zeros_f64x2, sum_a_z_a_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_a_f64x2 = zeros_f64x2, sum_b_y_a_f64x2 = zeros_f64x2, sum_b_z_a_f64x2 = zeros_f64x2;
    float64x2_t sum_a_x_b_f64x2 = zeros_f64x2, sum_a_y_b_f64x2 = zeros_f64x2, sum_a_z_b_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_b_f64x2 = zeros_f64x2, sum_b_y_b_f64x2 = zeros_f64x2, sum_b_z_b_f64x2 = zeros_f64x2;

    float64x2_t cov_xx_a_f64x2 = zeros_f64x2, cov_xy_a_f64x2 = zeros_f64x2, cov_xz_a_f64x2 = zeros_f64x2;
    float64x2_t cov_yx_a_f64x2 = zeros_f64x2, cov_yy_a_f64x2 = zeros_f64x2, cov_yz_a_f64x2 = zeros_f64x2;
    float64x2_t cov_zx_a_f64x2 = zeros_f64x2, cov_zy_a_f64x2 = zeros_f64x2, cov_zz_a_f64x2 = zeros_f64x2;
    float64x2_t cov_xx_b_f64x2 = zeros_f64x2, cov_xy_b_f64x2 = zeros_f64x2, cov_xz_b_f64x2 = zeros_f64x2;
    float64x2_t cov_yx_b_f64x2 = zeros_f64x2, cov_yy_b_f64x2 = zeros_f64x2, cov_yz_b_f64x2 = zeros_f64x2;
    float64x2_t cov_zx_b_f64x2 = zeros_f64x2, cov_zy_b_f64x2 = zeros_f64x2, cov_zz_b_f64x2 = zeros_f64x2;

    nk_size_t i = 0;
    float64x2_t a1_x_f64x2, a1_y_f64x2, a1_z_f64x2, b1_x_f64x2, b1_y_f64x2, b1_z_f64x2;
    float64x2_t a2_x_f64x2, a2_y_f64x2, a2_z_f64x2, b2_x_f64x2, b2_y_f64x2, b2_z_f64x2;

    // Main loop: 4 points per iteration (2x unrolled)
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a1_x_f64x2, &a1_y_f64x2, &a1_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b1_x_f64x2, &b1_y_f64x2, &b1_z_f64x2);
        nk_deinterleave_f64x2_neon_(a + (i + 2) * 3, &a2_x_f64x2, &a2_y_f64x2, &a2_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + (i + 2) * 3, &b2_x_f64x2, &b2_y_f64x2, &b2_z_f64x2);

        // Interleaved accumulation
        sum_a_x_a_f64x2 = vaddq_f64(sum_a_x_a_f64x2, a1_x_f64x2);
        sum_a_x_b_f64x2 = vaddq_f64(sum_a_x_b_f64x2, a2_x_f64x2);
        sum_a_y_a_f64x2 = vaddq_f64(sum_a_y_a_f64x2, a1_y_f64x2);
        sum_a_y_b_f64x2 = vaddq_f64(sum_a_y_b_f64x2, a2_y_f64x2);
        sum_a_z_a_f64x2 = vaddq_f64(sum_a_z_a_f64x2, a1_z_f64x2);
        sum_a_z_b_f64x2 = vaddq_f64(sum_a_z_b_f64x2, a2_z_f64x2);
        sum_b_x_a_f64x2 = vaddq_f64(sum_b_x_a_f64x2, b1_x_f64x2);
        sum_b_x_b_f64x2 = vaddq_f64(sum_b_x_b_f64x2, b2_x_f64x2);
        sum_b_y_a_f64x2 = vaddq_f64(sum_b_y_a_f64x2, b1_y_f64x2);
        sum_b_y_b_f64x2 = vaddq_f64(sum_b_y_b_f64x2, b2_y_f64x2);
        sum_b_z_a_f64x2 = vaddq_f64(sum_b_z_a_f64x2, b1_z_f64x2);
        sum_b_z_b_f64x2 = vaddq_f64(sum_b_z_b_f64x2, b2_z_f64x2);

        cov_xx_a_f64x2 = vfmaq_f64(cov_xx_a_f64x2, a1_x_f64x2, b1_x_f64x2);
        cov_xx_b_f64x2 = vfmaq_f64(cov_xx_b_f64x2, a2_x_f64x2, b2_x_f64x2);
        cov_xy_a_f64x2 = vfmaq_f64(cov_xy_a_f64x2, a1_x_f64x2, b1_y_f64x2);
        cov_xy_b_f64x2 = vfmaq_f64(cov_xy_b_f64x2, a2_x_f64x2, b2_y_f64x2);
        cov_xz_a_f64x2 = vfmaq_f64(cov_xz_a_f64x2, a1_x_f64x2, b1_z_f64x2);
        cov_xz_b_f64x2 = vfmaq_f64(cov_xz_b_f64x2, a2_x_f64x2, b2_z_f64x2);
        cov_yx_a_f64x2 = vfmaq_f64(cov_yx_a_f64x2, a1_y_f64x2, b1_x_f64x2);
        cov_yx_b_f64x2 = vfmaq_f64(cov_yx_b_f64x2, a2_y_f64x2, b2_x_f64x2);
        cov_yy_a_f64x2 = vfmaq_f64(cov_yy_a_f64x2, a1_y_f64x2, b1_y_f64x2);
        cov_yy_b_f64x2 = vfmaq_f64(cov_yy_b_f64x2, a2_y_f64x2, b2_y_f64x2);
        cov_yz_a_f64x2 = vfmaq_f64(cov_yz_a_f64x2, a1_y_f64x2, b1_z_f64x2);
        cov_yz_b_f64x2 = vfmaq_f64(cov_yz_b_f64x2, a2_y_f64x2, b2_z_f64x2);
        cov_zx_a_f64x2 = vfmaq_f64(cov_zx_a_f64x2, a1_z_f64x2, b1_x_f64x2);
        cov_zx_b_f64x2 = vfmaq_f64(cov_zx_b_f64x2, a2_z_f64x2, b2_x_f64x2);
        cov_zy_a_f64x2 = vfmaq_f64(cov_zy_a_f64x2, a1_z_f64x2, b1_y_f64x2);
        cov_zy_b_f64x2 = vfmaq_f64(cov_zy_b_f64x2, a2_z_f64x2, b2_y_f64x2);
        cov_zz_a_f64x2 = vfmaq_f64(cov_zz_a_f64x2, a1_z_f64x2, b1_z_f64x2);
        cov_zz_b_f64x2 = vfmaq_f64(cov_zz_b_f64x2, a2_z_f64x2, b2_z_f64x2);
    }

    // 2-point tail
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a1_x_f64x2, &a1_y_f64x2, &a1_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b1_x_f64x2, &b1_y_f64x2, &b1_z_f64x2);
        sum_a_x_a_f64x2 = vaddq_f64(sum_a_x_a_f64x2, a1_x_f64x2);
        sum_a_y_a_f64x2 = vaddq_f64(sum_a_y_a_f64x2, a1_y_f64x2);
        sum_a_z_a_f64x2 = vaddq_f64(sum_a_z_a_f64x2, a1_z_f64x2);
        sum_b_x_a_f64x2 = vaddq_f64(sum_b_x_a_f64x2, b1_x_f64x2);
        sum_b_y_a_f64x2 = vaddq_f64(sum_b_y_a_f64x2, b1_y_f64x2);
        sum_b_z_a_f64x2 = vaddq_f64(sum_b_z_a_f64x2, b1_z_f64x2);
        cov_xx_a_f64x2 = vfmaq_f64(cov_xx_a_f64x2, a1_x_f64x2, b1_x_f64x2);
        cov_xy_a_f64x2 = vfmaq_f64(cov_xy_a_f64x2, a1_x_f64x2, b1_y_f64x2);
        cov_xz_a_f64x2 = vfmaq_f64(cov_xz_a_f64x2, a1_x_f64x2, b1_z_f64x2);
        cov_yx_a_f64x2 = vfmaq_f64(cov_yx_a_f64x2, a1_y_f64x2, b1_x_f64x2);
        cov_yy_a_f64x2 = vfmaq_f64(cov_yy_a_f64x2, a1_y_f64x2, b1_y_f64x2);
        cov_yz_a_f64x2 = vfmaq_f64(cov_yz_a_f64x2, a1_y_f64x2, b1_z_f64x2);
        cov_zx_a_f64x2 = vfmaq_f64(cov_zx_a_f64x2, a1_z_f64x2, b1_x_f64x2);
        cov_zy_a_f64x2 = vfmaq_f64(cov_zy_a_f64x2, a1_z_f64x2, b1_y_f64x2);
        cov_zz_a_f64x2 = vfmaq_f64(cov_zz_a_f64x2, a1_z_f64x2, b1_z_f64x2);
    }

    // Combine dual accumulators
    float64x2_t sum_a_x_f64x2 = vaddq_f64(sum_a_x_a_f64x2, sum_a_x_b_f64x2);
    float64x2_t sum_a_y_f64x2 = vaddq_f64(sum_a_y_a_f64x2, sum_a_y_b_f64x2);
    float64x2_t sum_a_z_f64x2 = vaddq_f64(sum_a_z_a_f64x2, sum_a_z_b_f64x2);
    float64x2_t sum_b_x_f64x2 = vaddq_f64(sum_b_x_a_f64x2, sum_b_x_b_f64x2);
    float64x2_t sum_b_y_f64x2 = vaddq_f64(sum_b_y_a_f64x2, sum_b_y_b_f64x2);
    float64x2_t sum_b_z_f64x2 = vaddq_f64(sum_b_z_a_f64x2, sum_b_z_b_f64x2);
    float64x2_t cov_xx_f64x2 = vaddq_f64(cov_xx_a_f64x2, cov_xx_b_f64x2);
    float64x2_t cov_xy_f64x2 = vaddq_f64(cov_xy_a_f64x2, cov_xy_b_f64x2);
    float64x2_t cov_xz_f64x2 = vaddq_f64(cov_xz_a_f64x2, cov_xz_b_f64x2);
    float64x2_t cov_yx_f64x2 = vaddq_f64(cov_yx_a_f64x2, cov_yx_b_f64x2);
    float64x2_t cov_yy_f64x2 = vaddq_f64(cov_yy_a_f64x2, cov_yy_b_f64x2);
    float64x2_t cov_yz_f64x2 = vaddq_f64(cov_yz_a_f64x2, cov_yz_b_f64x2);
    float64x2_t cov_zx_f64x2 = vaddq_f64(cov_zx_a_f64x2, cov_zx_b_f64x2);
    float64x2_t cov_zy_f64x2 = vaddq_f64(cov_zy_a_f64x2, cov_zy_b_f64x2);
    float64x2_t cov_zz_f64x2 = vaddq_f64(cov_zz_a_f64x2, cov_zz_b_f64x2);

    // Reduce vector accumulators.
    nk_f64_t sum_a_x = nk_reduce_stable_f64x2_neon_(sum_a_x_f64x2), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x2_neon_(sum_a_y_f64x2), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x2_neon_(sum_a_z_f64x2), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x2_neon_(sum_b_x_f64x2), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x2_neon_(sum_b_y_f64x2), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x2_neon_(sum_b_z_f64x2), sum_b_z_compensation = 0.0;

    nk_f64_t covariance_x_x = nk_reduce_stable_f64x2_neon_(cov_xx_f64x2), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x2_neon_(cov_xy_f64x2), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x2_neon_(cov_xz_f64x2), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x2_neon_(cov_yx_f64x2), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x2_neon_(cov_yy_f64x2), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x2_neon_(cov_yz_f64x2), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x2_neon_(cov_zx_f64x2), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x2_neon_(cov_zy_f64x2), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x2_neon_(cov_zz_f64x2), covariance_z_z_compensation = 0.0;

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&sum_a_x, &sum_a_x_compensation, ax),
            nk_accumulate_sum_f64_(&sum_a_y, &sum_a_y_compensation, ay),
            nk_accumulate_sum_f64_(&sum_a_z, &sum_a_z_compensation, az);
        nk_accumulate_sum_f64_(&sum_b_x, &sum_b_x_compensation, bx),
            nk_accumulate_sum_f64_(&sum_b_y, &sum_b_y_compensation, by),
            nk_accumulate_sum_f64_(&sum_b_z, &sum_b_z_compensation, bz);
        nk_accumulate_product_f64_(&covariance_x_x, &covariance_x_x_compensation, ax, bx),
            nk_accumulate_product_f64_(&covariance_x_y, &covariance_x_y_compensation, ax, by),
            nk_accumulate_product_f64_(&covariance_x_z, &covariance_x_z_compensation, ax, bz);
        nk_accumulate_product_f64_(&covariance_y_x, &covariance_y_x_compensation, ay, bx),
            nk_accumulate_product_f64_(&covariance_y_y, &covariance_y_y_compensation, ay, by),
            nk_accumulate_product_f64_(&covariance_y_z, &covariance_y_z_compensation, ay, bz);
        nk_accumulate_product_f64_(&covariance_z_x, &covariance_z_x_compensation, az, bx),
            nk_accumulate_product_f64_(&covariance_z_y, &covariance_z_y_compensation, az, by),
            nk_accumulate_product_f64_(&covariance_z_z, &covariance_z_z_compensation, az, bz);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    covariance_x_x -= n * centroid_a_x * centroid_b_x;
    covariance_x_y -= n * centroid_a_x * centroid_b_y;
    covariance_x_z -= n * centroid_a_x * centroid_b_z;
    covariance_y_x -= n * centroid_a_y * centroid_b_x;
    covariance_y_y -= n * centroid_a_y * centroid_b_y;
    covariance_y_z -= n * centroid_a_y * centroid_b_z;
    covariance_z_x -= n * centroid_a_z * centroid_b_x;
    covariance_z_y -= n * centroid_a_z * centroid_b_y;
    covariance_z_z -= n * centroid_a_z * centroid_b_z;

    // Compute SVD and optimal rotation
    nk_f64_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    nk_rotation_from_svd_f64_neon_(svd_u, svd_v, r);

    // Handle reflection: if det(R) < 0, negate third column of V and recompute R
    if (nk_det3x3_f64_(r) < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_neon_(svd_u, svd_v, r);
    }

    // Output rotation matrix and scale=1.0.
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];

    if (scale) *scale = 1.0;

    // Compute RMSD after optimal rotation
    nk_f64_t sum_squared = nk_transformed_ssd_f64_neon_(a, b, n, r, 1.0, centroid_a_x, centroid_a_y, centroid_a_z,
                                                        centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_neon(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_f32_neon(nk_f32_t const *a, nk_f32_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f64_t *result) {
    float64x2_t zero_f64x2 = vdupq_n_f64(0.0);

    // Centroid accumulators (f64, lower/upper halves of f32x4)
    float64x2_t sum_a_x_lower_f64x2 = zero_f64x2, sum_a_x_upper_f64x2 = zero_f64x2;
    float64x2_t sum_a_y_lower_f64x2 = zero_f64x2, sum_a_y_upper_f64x2 = zero_f64x2;
    float64x2_t sum_a_z_lower_f64x2 = zero_f64x2, sum_a_z_upper_f64x2 = zero_f64x2;
    float64x2_t sum_b_x_lower_f64x2 = zero_f64x2, sum_b_x_upper_f64x2 = zero_f64x2;
    float64x2_t sum_b_y_lower_f64x2 = zero_f64x2, sum_b_y_upper_f64x2 = zero_f64x2;
    float64x2_t sum_b_z_lower_f64x2 = zero_f64x2, sum_b_z_upper_f64x2 = zero_f64x2;

    // Covariance accumulators (f64, lower/upper halves)
    float64x2_t cov_xx_lower_f64x2 = zero_f64x2, cov_xx_upper_f64x2 = zero_f64x2;
    float64x2_t cov_xy_lower_f64x2 = zero_f64x2, cov_xy_upper_f64x2 = zero_f64x2;
    float64x2_t cov_xz_lower_f64x2 = zero_f64x2, cov_xz_upper_f64x2 = zero_f64x2;
    float64x2_t cov_yx_lower_f64x2 = zero_f64x2, cov_yx_upper_f64x2 = zero_f64x2;
    float64x2_t cov_yy_lower_f64x2 = zero_f64x2, cov_yy_upper_f64x2 = zero_f64x2;
    float64x2_t cov_yz_lower_f64x2 = zero_f64x2, cov_yz_upper_f64x2 = zero_f64x2;
    float64x2_t cov_zx_lower_f64x2 = zero_f64x2, cov_zx_upper_f64x2 = zero_f64x2;
    float64x2_t cov_zy_lower_f64x2 = zero_f64x2, cov_zy_upper_f64x2 = zero_f64x2;
    float64x2_t cov_zz_lower_f64x2 = zero_f64x2, cov_zz_upper_f64x2 = zero_f64x2;

    // Variance of A accumulator
    float64x2_t variance_lower_f64x2 = zero_f64x2, variance_upper_f64x2 = zero_f64x2;

    nk_size_t index = 0;
    for (; index + 4 <= n; index += 4) {
        float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_f32x4_neon_(a + index * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4),
            nk_deinterleave_f32x4_neon_(b + index * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        float64x2_t a_x_lower_f64x2 = vcvt_f64_f32(vget_low_f32(a_x_f32x4));
        float64x2_t a_x_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(a_x_f32x4);
        float64x2_t a_y_lower_f64x2 = vcvt_f64_f32(vget_low_f32(a_y_f32x4));
        float64x2_t a_y_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(a_y_f32x4);
        float64x2_t a_z_lower_f64x2 = vcvt_f64_f32(vget_low_f32(a_z_f32x4));
        float64x2_t a_z_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(a_z_f32x4);
        float64x2_t b_x_lower_f64x2 = vcvt_f64_f32(vget_low_f32(b_x_f32x4));
        float64x2_t b_x_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(b_x_f32x4);
        float64x2_t b_y_lower_f64x2 = vcvt_f64_f32(vget_low_f32(b_y_f32x4));
        float64x2_t b_y_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(b_y_f32x4);
        float64x2_t b_z_lower_f64x2 = vcvt_f64_f32(vget_low_f32(b_z_f32x4));
        float64x2_t b_z_upper_f64x2 = nk_promote_upper_f32x4_to_f64x2_neon_(b_z_f32x4);

        // Accumulate centroids
        sum_a_x_lower_f64x2 = vaddq_f64(sum_a_x_lower_f64x2, a_x_lower_f64x2),
        sum_a_x_upper_f64x2 = vaddq_f64(sum_a_x_upper_f64x2, a_x_upper_f64x2);
        sum_a_y_lower_f64x2 = vaddq_f64(sum_a_y_lower_f64x2, a_y_lower_f64x2),
        sum_a_y_upper_f64x2 = vaddq_f64(sum_a_y_upper_f64x2, a_y_upper_f64x2);
        sum_a_z_lower_f64x2 = vaddq_f64(sum_a_z_lower_f64x2, a_z_lower_f64x2),
        sum_a_z_upper_f64x2 = vaddq_f64(sum_a_z_upper_f64x2, a_z_upper_f64x2);
        sum_b_x_lower_f64x2 = vaddq_f64(sum_b_x_lower_f64x2, b_x_lower_f64x2),
        sum_b_x_upper_f64x2 = vaddq_f64(sum_b_x_upper_f64x2, b_x_upper_f64x2);
        sum_b_y_lower_f64x2 = vaddq_f64(sum_b_y_lower_f64x2, b_y_lower_f64x2),
        sum_b_y_upper_f64x2 = vaddq_f64(sum_b_y_upper_f64x2, b_y_upper_f64x2);
        sum_b_z_lower_f64x2 = vaddq_f64(sum_b_z_lower_f64x2, b_z_lower_f64x2),
        sum_b_z_upper_f64x2 = vaddq_f64(sum_b_z_upper_f64x2, b_z_upper_f64x2);

        // Accumulate raw outer products (uncentered)
        cov_xx_lower_f64x2 = vfmaq_f64(cov_xx_lower_f64x2, a_x_lower_f64x2, b_x_lower_f64x2),
        cov_xx_upper_f64x2 = vfmaq_f64(cov_xx_upper_f64x2, a_x_upper_f64x2, b_x_upper_f64x2);
        cov_xy_lower_f64x2 = vfmaq_f64(cov_xy_lower_f64x2, a_x_lower_f64x2, b_y_lower_f64x2),
        cov_xy_upper_f64x2 = vfmaq_f64(cov_xy_upper_f64x2, a_x_upper_f64x2, b_y_upper_f64x2);
        cov_xz_lower_f64x2 = vfmaq_f64(cov_xz_lower_f64x2, a_x_lower_f64x2, b_z_lower_f64x2),
        cov_xz_upper_f64x2 = vfmaq_f64(cov_xz_upper_f64x2, a_x_upper_f64x2, b_z_upper_f64x2);
        cov_yx_lower_f64x2 = vfmaq_f64(cov_yx_lower_f64x2, a_y_lower_f64x2, b_x_lower_f64x2),
        cov_yx_upper_f64x2 = vfmaq_f64(cov_yx_upper_f64x2, a_y_upper_f64x2, b_x_upper_f64x2);
        cov_yy_lower_f64x2 = vfmaq_f64(cov_yy_lower_f64x2, a_y_lower_f64x2, b_y_lower_f64x2),
        cov_yy_upper_f64x2 = vfmaq_f64(cov_yy_upper_f64x2, a_y_upper_f64x2, b_y_upper_f64x2);
        cov_yz_lower_f64x2 = vfmaq_f64(cov_yz_lower_f64x2, a_y_lower_f64x2, b_z_lower_f64x2),
        cov_yz_upper_f64x2 = vfmaq_f64(cov_yz_upper_f64x2, a_y_upper_f64x2, b_z_upper_f64x2);
        cov_zx_lower_f64x2 = vfmaq_f64(cov_zx_lower_f64x2, a_z_lower_f64x2, b_x_lower_f64x2),
        cov_zx_upper_f64x2 = vfmaq_f64(cov_zx_upper_f64x2, a_z_upper_f64x2, b_x_upper_f64x2);
        cov_zy_lower_f64x2 = vfmaq_f64(cov_zy_lower_f64x2, a_z_lower_f64x2, b_y_lower_f64x2),
        cov_zy_upper_f64x2 = vfmaq_f64(cov_zy_upper_f64x2, a_z_upper_f64x2, b_y_upper_f64x2);
        cov_zz_lower_f64x2 = vfmaq_f64(cov_zz_lower_f64x2, a_z_lower_f64x2, b_z_lower_f64x2),
        cov_zz_upper_f64x2 = vfmaq_f64(cov_zz_upper_f64x2, a_z_upper_f64x2, b_z_upper_f64x2);

        // Accumulate variance of A (sum of squared coordinates)
        variance_lower_f64x2 = vfmaq_f64(variance_lower_f64x2, a_x_lower_f64x2, a_x_lower_f64x2),
        variance_upper_f64x2 = vfmaq_f64(variance_upper_f64x2, a_x_upper_f64x2, a_x_upper_f64x2);
        variance_lower_f64x2 = vfmaq_f64(variance_lower_f64x2, a_y_lower_f64x2, a_y_lower_f64x2),
        variance_upper_f64x2 = vfmaq_f64(variance_upper_f64x2, a_y_upper_f64x2, a_y_upper_f64x2);
        variance_lower_f64x2 = vfmaq_f64(variance_lower_f64x2, a_z_lower_f64x2, a_z_lower_f64x2),
        variance_upper_f64x2 = vfmaq_f64(variance_upper_f64x2, a_z_upper_f64x2, a_z_upper_f64x2);
    }

    // Reduce centroid accumulators
    nk_f64_t sum_a_x = vaddvq_f64(vaddq_f64(sum_a_x_lower_f64x2, sum_a_x_upper_f64x2));
    nk_f64_t sum_a_y = vaddvq_f64(vaddq_f64(sum_a_y_lower_f64x2, sum_a_y_upper_f64x2));
    nk_f64_t sum_a_z = vaddvq_f64(vaddq_f64(sum_a_z_lower_f64x2, sum_a_z_upper_f64x2));
    nk_f64_t sum_b_x = vaddvq_f64(vaddq_f64(sum_b_x_lower_f64x2, sum_b_x_upper_f64x2));
    nk_f64_t sum_b_y = vaddvq_f64(vaddq_f64(sum_b_y_lower_f64x2, sum_b_y_upper_f64x2));
    nk_f64_t sum_b_z = vaddvq_f64(vaddq_f64(sum_b_z_lower_f64x2, sum_b_z_upper_f64x2));

    // Reduce covariance accumulators
    nk_f64_t covariance_x_x = vaddvq_f64(vaddq_f64(cov_xx_lower_f64x2, cov_xx_upper_f64x2));
    nk_f64_t covariance_x_y = vaddvq_f64(vaddq_f64(cov_xy_lower_f64x2, cov_xy_upper_f64x2));
    nk_f64_t covariance_x_z = vaddvq_f64(vaddq_f64(cov_xz_lower_f64x2, cov_xz_upper_f64x2));
    nk_f64_t covariance_y_x = vaddvq_f64(vaddq_f64(cov_yx_lower_f64x2, cov_yx_upper_f64x2));
    nk_f64_t covariance_y_y = vaddvq_f64(vaddq_f64(cov_yy_lower_f64x2, cov_yy_upper_f64x2));
    nk_f64_t covariance_y_z = vaddvq_f64(vaddq_f64(cov_yz_lower_f64x2, cov_yz_upper_f64x2));
    nk_f64_t covariance_z_x = vaddvq_f64(vaddq_f64(cov_zx_lower_f64x2, cov_zx_upper_f64x2));
    nk_f64_t covariance_z_y = vaddvq_f64(vaddq_f64(cov_zy_lower_f64x2, cov_zy_upper_f64x2));
    nk_f64_t covariance_z_z = vaddvq_f64(vaddq_f64(cov_zz_lower_f64x2, cov_zz_upper_f64x2));
    nk_f64_t sum_sq_a = vaddvq_f64(vaddq_f64(variance_lower_f64x2, variance_upper_f64x2));

    // Scalar tail
    for (; index < n; ++index) {
        nk_f64_t ax = (nk_f64_t)a[index * 3 + 0], ay = (nk_f64_t)a[index * 3 + 1], az = (nk_f64_t)a[index * 3 + 2];
        nk_f64_t bx = (nk_f64_t)b[index * 3 + 0], by = (nk_f64_t)b[index * 3 + 1], bz = (nk_f64_t)b[index * 3 + 2];
        sum_a_x += ax, sum_a_y += ay, sum_a_z += az;
        sum_b_x += bx, sum_b_y += by, sum_b_z += bz;
        covariance_x_x += ax * bx, covariance_x_y += ax * by, covariance_x_z += ax * bz;
        covariance_y_x += ay * bx, covariance_y_y += ay * by, covariance_y_z += ay * bz;
        covariance_z_x += az * bx, covariance_z_y += az * by, covariance_z_z += az * bz;
        sum_sq_a += ax * ax + ay * ay + az * az;
    }

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid)
        a_centroid[0] = (nk_f32_t)centroid_a_x, a_centroid[1] = (nk_f32_t)centroid_a_y,
        a_centroid[2] = (nk_f32_t)centroid_a_z;
    if (b_centroid)
        b_centroid[0] = (nk_f32_t)centroid_b_x, b_centroid[1] = (nk_f32_t)centroid_b_y,
        b_centroid[2] = (nk_f32_t)centroid_b_z;

    // Compute variance of A (centered): var = sum(a^2)/n - centroid^2
    nk_f64_t variance_a = sum_sq_a * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

    // Apply centering correction: H_centered = sum(a * bᵀ) - n * centroid_a * centroid_bᵀ
    nk_f64_t h[9];
    h[0] = covariance_x_x - n * centroid_a_x * centroid_b_x;
    h[1] = covariance_x_y - n * centroid_a_x * centroid_b_y;
    h[2] = covariance_x_z - n * centroid_a_x * centroid_b_z;
    h[3] = covariance_y_x - n * centroid_a_y * centroid_b_x;
    h[4] = covariance_y_y - n * centroid_a_y * centroid_b_y;
    h[5] = covariance_y_z - n * centroid_a_y * centroid_b_z;
    h[6] = covariance_z_x - n * centroid_a_z * centroid_b_x;
    h[7] = covariance_z_y - n * centroid_a_z * centroid_b_y;
    h[8] = covariance_z_z - n * centroid_a_z * centroid_b_z;

    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(h, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    r[0] = svd_v[0] * svd_u[0] + svd_v[1] * svd_u[1] + svd_v[2] * svd_u[2];
    r[1] = svd_v[0] * svd_u[3] + svd_v[1] * svd_u[4] + svd_v[2] * svd_u[5];
    r[2] = svd_v[0] * svd_u[6] + svd_v[1] * svd_u[7] + svd_v[2] * svd_u[8];
    r[3] = svd_v[3] * svd_u[0] + svd_v[4] * svd_u[1] + svd_v[5] * svd_u[2];
    r[4] = svd_v[3] * svd_u[3] + svd_v[4] * svd_u[4] + svd_v[5] * svd_u[5];
    r[5] = svd_v[3] * svd_u[6] + svd_v[4] * svd_u[7] + svd_v[5] * svd_u[8];
    r[6] = svd_v[6] * svd_u[0] + svd_v[7] * svd_u[1] + svd_v[8] * svd_u[2];
    r[7] = svd_v[6] * svd_u[3] + svd_v[7] * svd_u[4] + svd_v[8] * svd_u[5];
    r[8] = svd_v[6] * svd_u[6] + svd_v[7] * svd_u[7] + svd_v[8] * svd_u[8];

    nk_f64_t det = nk_det3x3_f64_(r), sign_correction = det < 0 ? -1.0 : 1.0;
    if (det < 0) {
        svd_v[2] = -svd_v[2], svd_v[5] = -svd_v[5], svd_v[8] = -svd_v[8];
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

    nk_f64_t applied_scale = (svd_s[0] + svd_s[4] + sign_correction * svd_s[8]) / ((nk_f64_t)n * variance_a);
    if (rotation)
        for (int j = 0; j != 9; ++j) rotation[j] = (nk_f32_t)r[j];
    if (scale) *scale = (nk_f32_t)applied_scale;
    *result = nk_f64_sqrt_neon(nk_transformed_ssd_f32_neon_(a, b, n, r, applied_scale, centroid_a_x, centroid_a_y,
                                                            centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z) /
                               (nk_f64_t)n);
}

NK_PUBLIC void nk_umeyama_f64_neon(nk_f64_t const *a, nk_f64_t const *b, nk_size_t n, nk_f64_t *a_centroid,
                                   nk_f64_t *b_centroid, nk_f64_t *rotation, nk_f64_t *scale, nk_f64_t *result) {
    float64x2_t const zeros_f64x2 = vdupq_n_f64(0);

    // 2x unrolling with dual accumulators to hide FMA latency.
    float64x2_t sum_a_x_a_f64x2 = zeros_f64x2, sum_a_y_a_f64x2 = zeros_f64x2, sum_a_z_a_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_a_f64x2 = zeros_f64x2, sum_b_y_a_f64x2 = zeros_f64x2, sum_b_z_a_f64x2 = zeros_f64x2;
    float64x2_t sum_a_x_b_f64x2 = zeros_f64x2, sum_a_y_b_f64x2 = zeros_f64x2, sum_a_z_b_f64x2 = zeros_f64x2;
    float64x2_t sum_b_x_b_f64x2 = zeros_f64x2, sum_b_y_b_f64x2 = zeros_f64x2, sum_b_z_b_f64x2 = zeros_f64x2;

    float64x2_t cov_xx_a_f64x2 = zeros_f64x2, cov_xy_a_f64x2 = zeros_f64x2, cov_xz_a_f64x2 = zeros_f64x2;
    float64x2_t cov_yx_a_f64x2 = zeros_f64x2, cov_yy_a_f64x2 = zeros_f64x2, cov_yz_a_f64x2 = zeros_f64x2;
    float64x2_t cov_zx_a_f64x2 = zeros_f64x2, cov_zy_a_f64x2 = zeros_f64x2, cov_zz_a_f64x2 = zeros_f64x2;
    float64x2_t cov_xx_b_f64x2 = zeros_f64x2, cov_xy_b_f64x2 = zeros_f64x2, cov_xz_b_f64x2 = zeros_f64x2;
    float64x2_t cov_yx_b_f64x2 = zeros_f64x2, cov_yy_b_f64x2 = zeros_f64x2, cov_yz_b_f64x2 = zeros_f64x2;
    float64x2_t cov_zx_b_f64x2 = zeros_f64x2, cov_zy_b_f64x2 = zeros_f64x2, cov_zz_b_f64x2 = zeros_f64x2;
    float64x2_t variance_a_a_f64x2 = zeros_f64x2, variance_a_b_f64x2 = zeros_f64x2;

    nk_size_t i = 0;
    float64x2_t a1_x_f64x2, a1_y_f64x2, a1_z_f64x2, b1_x_f64x2, b1_y_f64x2, b1_z_f64x2;
    float64x2_t a2_x_f64x2, a2_y_f64x2, a2_z_f64x2, b2_x_f64x2, b2_y_f64x2, b2_z_f64x2;

    // Main loop: 4 points per iteration (2x unrolled)
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a1_x_f64x2, &a1_y_f64x2, &a1_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b1_x_f64x2, &b1_y_f64x2, &b1_z_f64x2);
        nk_deinterleave_f64x2_neon_(a + (i + 2) * 3, &a2_x_f64x2, &a2_y_f64x2, &a2_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + (i + 2) * 3, &b2_x_f64x2, &b2_y_f64x2, &b2_z_f64x2);

        // Interleaved accumulation
        sum_a_x_a_f64x2 = vaddq_f64(sum_a_x_a_f64x2, a1_x_f64x2);
        sum_a_x_b_f64x2 = vaddq_f64(sum_a_x_b_f64x2, a2_x_f64x2);
        sum_a_y_a_f64x2 = vaddq_f64(sum_a_y_a_f64x2, a1_y_f64x2);
        sum_a_y_b_f64x2 = vaddq_f64(sum_a_y_b_f64x2, a2_y_f64x2);
        sum_a_z_a_f64x2 = vaddq_f64(sum_a_z_a_f64x2, a1_z_f64x2);
        sum_a_z_b_f64x2 = vaddq_f64(sum_a_z_b_f64x2, a2_z_f64x2);
        sum_b_x_a_f64x2 = vaddq_f64(sum_b_x_a_f64x2, b1_x_f64x2);
        sum_b_x_b_f64x2 = vaddq_f64(sum_b_x_b_f64x2, b2_x_f64x2);
        sum_b_y_a_f64x2 = vaddq_f64(sum_b_y_a_f64x2, b1_y_f64x2);
        sum_b_y_b_f64x2 = vaddq_f64(sum_b_y_b_f64x2, b2_y_f64x2);
        sum_b_z_a_f64x2 = vaddq_f64(sum_b_z_a_f64x2, b1_z_f64x2);
        sum_b_z_b_f64x2 = vaddq_f64(sum_b_z_b_f64x2, b2_z_f64x2);

        cov_xx_a_f64x2 = vfmaq_f64(cov_xx_a_f64x2, a1_x_f64x2, b1_x_f64x2);
        cov_xx_b_f64x2 = vfmaq_f64(cov_xx_b_f64x2, a2_x_f64x2, b2_x_f64x2);
        cov_xy_a_f64x2 = vfmaq_f64(cov_xy_a_f64x2, a1_x_f64x2, b1_y_f64x2);
        cov_xy_b_f64x2 = vfmaq_f64(cov_xy_b_f64x2, a2_x_f64x2, b2_y_f64x2);
        cov_xz_a_f64x2 = vfmaq_f64(cov_xz_a_f64x2, a1_x_f64x2, b1_z_f64x2);
        cov_xz_b_f64x2 = vfmaq_f64(cov_xz_b_f64x2, a2_x_f64x2, b2_z_f64x2);
        cov_yx_a_f64x2 = vfmaq_f64(cov_yx_a_f64x2, a1_y_f64x2, b1_x_f64x2);
        cov_yx_b_f64x2 = vfmaq_f64(cov_yx_b_f64x2, a2_y_f64x2, b2_x_f64x2);
        cov_yy_a_f64x2 = vfmaq_f64(cov_yy_a_f64x2, a1_y_f64x2, b1_y_f64x2);
        cov_yy_b_f64x2 = vfmaq_f64(cov_yy_b_f64x2, a2_y_f64x2, b2_y_f64x2);
        cov_yz_a_f64x2 = vfmaq_f64(cov_yz_a_f64x2, a1_y_f64x2, b1_z_f64x2);
        cov_yz_b_f64x2 = vfmaq_f64(cov_yz_b_f64x2, a2_y_f64x2, b2_z_f64x2);
        cov_zx_a_f64x2 = vfmaq_f64(cov_zx_a_f64x2, a1_z_f64x2, b1_x_f64x2);
        cov_zx_b_f64x2 = vfmaq_f64(cov_zx_b_f64x2, a2_z_f64x2, b2_x_f64x2);
        cov_zy_a_f64x2 = vfmaq_f64(cov_zy_a_f64x2, a1_z_f64x2, b1_y_f64x2);
        cov_zy_b_f64x2 = vfmaq_f64(cov_zy_b_f64x2, a2_z_f64x2, b2_y_f64x2);
        cov_zz_a_f64x2 = vfmaq_f64(cov_zz_a_f64x2, a1_z_f64x2, b1_z_f64x2);
        cov_zz_b_f64x2 = vfmaq_f64(cov_zz_b_f64x2, a2_z_f64x2, b2_z_f64x2);

        variance_a_a_f64x2 = vfmaq_f64(variance_a_a_f64x2, a1_x_f64x2, a1_x_f64x2);
        variance_a_b_f64x2 = vfmaq_f64(variance_a_b_f64x2, a2_x_f64x2, a2_x_f64x2);
        variance_a_a_f64x2 = vfmaq_f64(variance_a_a_f64x2, a1_y_f64x2, a1_y_f64x2);
        variance_a_b_f64x2 = vfmaq_f64(variance_a_b_f64x2, a2_y_f64x2, a2_y_f64x2);
        variance_a_a_f64x2 = vfmaq_f64(variance_a_a_f64x2, a1_z_f64x2, a1_z_f64x2);
        variance_a_b_f64x2 = vfmaq_f64(variance_a_b_f64x2, a2_z_f64x2, a2_z_f64x2);
    }

    // 2-point tail
    for (; i + 2 <= n; i += 2) {
        nk_deinterleave_f64x2_neon_(a + i * 3, &a1_x_f64x2, &a1_y_f64x2, &a1_z_f64x2);
        nk_deinterleave_f64x2_neon_(b + i * 3, &b1_x_f64x2, &b1_y_f64x2, &b1_z_f64x2);
        sum_a_x_a_f64x2 = vaddq_f64(sum_a_x_a_f64x2, a1_x_f64x2);
        sum_a_y_a_f64x2 = vaddq_f64(sum_a_y_a_f64x2, a1_y_f64x2);
        sum_a_z_a_f64x2 = vaddq_f64(sum_a_z_a_f64x2, a1_z_f64x2);
        sum_b_x_a_f64x2 = vaddq_f64(sum_b_x_a_f64x2, b1_x_f64x2);
        sum_b_y_a_f64x2 = vaddq_f64(sum_b_y_a_f64x2, b1_y_f64x2);
        sum_b_z_a_f64x2 = vaddq_f64(sum_b_z_a_f64x2, b1_z_f64x2);
        cov_xx_a_f64x2 = vfmaq_f64(cov_xx_a_f64x2, a1_x_f64x2, b1_x_f64x2);
        cov_xy_a_f64x2 = vfmaq_f64(cov_xy_a_f64x2, a1_x_f64x2, b1_y_f64x2);
        cov_xz_a_f64x2 = vfmaq_f64(cov_xz_a_f64x2, a1_x_f64x2, b1_z_f64x2);
        cov_yx_a_f64x2 = vfmaq_f64(cov_yx_a_f64x2, a1_y_f64x2, b1_x_f64x2);
        cov_yy_a_f64x2 = vfmaq_f64(cov_yy_a_f64x2, a1_y_f64x2, b1_y_f64x2);
        cov_yz_a_f64x2 = vfmaq_f64(cov_yz_a_f64x2, a1_y_f64x2, b1_z_f64x2);
        cov_zx_a_f64x2 = vfmaq_f64(cov_zx_a_f64x2, a1_z_f64x2, b1_x_f64x2);
        cov_zy_a_f64x2 = vfmaq_f64(cov_zy_a_f64x2, a1_z_f64x2, b1_y_f64x2);
        cov_zz_a_f64x2 = vfmaq_f64(cov_zz_a_f64x2, a1_z_f64x2, b1_z_f64x2);
        variance_a_a_f64x2 = vfmaq_f64(variance_a_a_f64x2, a1_x_f64x2, a1_x_f64x2);
        variance_a_a_f64x2 = vfmaq_f64(variance_a_a_f64x2, a1_y_f64x2, a1_y_f64x2);
        variance_a_a_f64x2 = vfmaq_f64(variance_a_a_f64x2, a1_z_f64x2, a1_z_f64x2);
    }

    // Combine dual accumulators
    float64x2_t sum_a_x_f64x2 = vaddq_f64(sum_a_x_a_f64x2, sum_a_x_b_f64x2);
    float64x2_t sum_a_y_f64x2 = vaddq_f64(sum_a_y_a_f64x2, sum_a_y_b_f64x2);
    float64x2_t sum_a_z_f64x2 = vaddq_f64(sum_a_z_a_f64x2, sum_a_z_b_f64x2);
    float64x2_t sum_b_x_f64x2 = vaddq_f64(sum_b_x_a_f64x2, sum_b_x_b_f64x2);
    float64x2_t sum_b_y_f64x2 = vaddq_f64(sum_b_y_a_f64x2, sum_b_y_b_f64x2);
    float64x2_t sum_b_z_f64x2 = vaddq_f64(sum_b_z_a_f64x2, sum_b_z_b_f64x2);
    float64x2_t cov_xx_f64x2 = vaddq_f64(cov_xx_a_f64x2, cov_xx_b_f64x2);
    float64x2_t cov_xy_f64x2 = vaddq_f64(cov_xy_a_f64x2, cov_xy_b_f64x2);
    float64x2_t cov_xz_f64x2 = vaddq_f64(cov_xz_a_f64x2, cov_xz_b_f64x2);
    float64x2_t cov_yx_f64x2 = vaddq_f64(cov_yx_a_f64x2, cov_yx_b_f64x2);
    float64x2_t cov_yy_f64x2 = vaddq_f64(cov_yy_a_f64x2, cov_yy_b_f64x2);
    float64x2_t cov_yz_f64x2 = vaddq_f64(cov_yz_a_f64x2, cov_yz_b_f64x2);
    float64x2_t cov_zx_f64x2 = vaddq_f64(cov_zx_a_f64x2, cov_zx_b_f64x2);
    float64x2_t cov_zy_f64x2 = vaddq_f64(cov_zy_a_f64x2, cov_zy_b_f64x2);
    float64x2_t cov_zz_f64x2 = vaddq_f64(cov_zz_a_f64x2, cov_zz_b_f64x2);
    float64x2_t variance_a_f64x2 = vaddq_f64(variance_a_a_f64x2, variance_a_b_f64x2);

    // Reduce vector accumulators.
    nk_f64_t sum_a_x = nk_reduce_stable_f64x2_neon_(sum_a_x_f64x2), sum_a_x_compensation = 0.0;
    nk_f64_t sum_a_y = nk_reduce_stable_f64x2_neon_(sum_a_y_f64x2), sum_a_y_compensation = 0.0;
    nk_f64_t sum_a_z = nk_reduce_stable_f64x2_neon_(sum_a_z_f64x2), sum_a_z_compensation = 0.0;
    nk_f64_t sum_b_x = nk_reduce_stable_f64x2_neon_(sum_b_x_f64x2), sum_b_x_compensation = 0.0;
    nk_f64_t sum_b_y = nk_reduce_stable_f64x2_neon_(sum_b_y_f64x2), sum_b_y_compensation = 0.0;
    nk_f64_t sum_b_z = nk_reduce_stable_f64x2_neon_(sum_b_z_f64x2), sum_b_z_compensation = 0.0;
    nk_f64_t covariance_x_x = nk_reduce_stable_f64x2_neon_(cov_xx_f64x2), covariance_x_x_compensation = 0.0;
    nk_f64_t covariance_x_y = nk_reduce_stable_f64x2_neon_(cov_xy_f64x2), covariance_x_y_compensation = 0.0;
    nk_f64_t covariance_x_z = nk_reduce_stable_f64x2_neon_(cov_xz_f64x2), covariance_x_z_compensation = 0.0;
    nk_f64_t covariance_y_x = nk_reduce_stable_f64x2_neon_(cov_yx_f64x2), covariance_y_x_compensation = 0.0;
    nk_f64_t covariance_y_y = nk_reduce_stable_f64x2_neon_(cov_yy_f64x2), covariance_y_y_compensation = 0.0;
    nk_f64_t covariance_y_z = nk_reduce_stable_f64x2_neon_(cov_yz_f64x2), covariance_y_z_compensation = 0.0;
    nk_f64_t covariance_z_x = nk_reduce_stable_f64x2_neon_(cov_zx_f64x2), covariance_z_x_compensation = 0.0;
    nk_f64_t covariance_z_y = nk_reduce_stable_f64x2_neon_(cov_zy_f64x2), covariance_z_y_compensation = 0.0;
    nk_f64_t covariance_z_z = nk_reduce_stable_f64x2_neon_(cov_zz_f64x2), covariance_z_z_compensation = 0.0;
    nk_f64_t sum_sq_a = nk_reduce_stable_f64x2_neon_(variance_a_f64x2), sum_sq_a_compensation = 0.0;

    // Scalar tail
    for (; i < n; ++i) {
        nk_f64_t ax = a[i * 3 + 0], ay = a[i * 3 + 1], az = a[i * 3 + 2];
        nk_f64_t bx = b[i * 3 + 0], by = b[i * 3 + 1], bz = b[i * 3 + 2];
        nk_accumulate_sum_f64_(&sum_a_x, &sum_a_x_compensation, ax),
            nk_accumulate_sum_f64_(&sum_a_y, &sum_a_y_compensation, ay),
            nk_accumulate_sum_f64_(&sum_a_z, &sum_a_z_compensation, az);
        nk_accumulate_sum_f64_(&sum_b_x, &sum_b_x_compensation, bx),
            nk_accumulate_sum_f64_(&sum_b_y, &sum_b_y_compensation, by),
            nk_accumulate_sum_f64_(&sum_b_z, &sum_b_z_compensation, bz);
        nk_accumulate_product_f64_(&covariance_x_x, &covariance_x_x_compensation, ax, bx),
            nk_accumulate_product_f64_(&covariance_x_y, &covariance_x_y_compensation, ax, by),
            nk_accumulate_product_f64_(&covariance_x_z, &covariance_x_z_compensation, ax, bz);
        nk_accumulate_product_f64_(&covariance_y_x, &covariance_y_x_compensation, ay, bx),
            nk_accumulate_product_f64_(&covariance_y_y, &covariance_y_y_compensation, ay, by),
            nk_accumulate_product_f64_(&covariance_y_z, &covariance_y_z_compensation, ay, bz);
        nk_accumulate_product_f64_(&covariance_z_x, &covariance_z_x_compensation, az, bx),
            nk_accumulate_product_f64_(&covariance_z_y, &covariance_z_y_compensation, az, by),
            nk_accumulate_product_f64_(&covariance_z_z, &covariance_z_z_compensation, az, bz);
        nk_accumulate_square_f64_(&sum_sq_a, &sum_sq_a_compensation, ax),
            nk_accumulate_square_f64_(&sum_sq_a, &sum_sq_a_compensation, ay),
            nk_accumulate_square_f64_(&sum_sq_a, &sum_sq_a_compensation, az);
    }

    sum_a_x += sum_a_x_compensation, sum_a_y += sum_a_y_compensation, sum_a_z += sum_a_z_compensation;
    sum_b_x += sum_b_x_compensation, sum_b_y += sum_b_y_compensation, sum_b_z += sum_b_z_compensation;
    covariance_x_x += covariance_x_x_compensation, covariance_x_y += covariance_x_y_compensation,
        covariance_x_z += covariance_x_z_compensation;
    covariance_y_x += covariance_y_x_compensation, covariance_y_y += covariance_y_y_compensation,
        covariance_y_z += covariance_y_z_compensation;
    covariance_z_x += covariance_z_x_compensation, covariance_z_y += covariance_z_y_compensation,
        covariance_z_z += covariance_z_z_compensation;
    sum_sq_a += sum_sq_a_compensation;

    // Compute centroids
    nk_f64_t inv_n = 1.0 / (nk_f64_t)n;
    nk_f64_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f64_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;
    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute variance of A (centered)
    nk_f64_t centroid_sq = centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z;
    nk_f64_t var_a = sum_sq_a * inv_n - centroid_sq;

    // Apply centering correction: H_centered = H - n * centroid_a * centroid_bᵀ
    covariance_x_x -= n * centroid_a_x * centroid_b_x;
    covariance_x_y -= n * centroid_a_x * centroid_b_y;
    covariance_x_z -= n * centroid_a_x * centroid_b_z;
    covariance_y_x -= n * centroid_a_y * centroid_b_x;
    covariance_y_y -= n * centroid_a_y * centroid_b_y;
    covariance_y_z -= n * centroid_a_y * centroid_b_z;
    covariance_z_x -= n * centroid_a_z * centroid_b_x;
    covariance_z_y -= n * centroid_a_z * centroid_b_y;
    covariance_z_z -= n * centroid_a_z * centroid_b_z;

    // Compute SVD
    nk_f64_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};
    nk_f64_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f64_(cross_covariance, svd_u, svd_s, svd_v);

    nk_f64_t r[9];
    nk_rotation_from_svd_f64_neon_(svd_u, svd_v, r);

    // Handle reflection and compute scale
    nk_f64_t det = nk_det3x3_f64_(r);
    nk_f64_t trace_d_s = svd_s[0] + svd_s[4] + (det < 0 ? -svd_s[8] : svd_s[8]);
    nk_f64_t computed_scale = trace_d_s / (n * var_a);

    if (det < 0) {
        svd_v[2] = -svd_v[2];
        svd_v[5] = -svd_v[5];
        svd_v[8] = -svd_v[8];
        nk_rotation_from_svd_f64_neon_(svd_u, svd_v, r);
    }

    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    if (scale) *scale = computed_scale;

    // Compute RMSD after transformation
    nk_f64_t sum_squared = nk_transformed_ssd_f64_neon_(a, b, n, r, computed_scale, centroid_a_x, centroid_a_y,
                                                        centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f64_sqrt_neon(sum_squared * inv_n);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_NEON
#endif // NK_TARGET_ARM_
#endif // NK_MESH_NEON_H
