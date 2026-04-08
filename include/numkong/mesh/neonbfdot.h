/**
 *  @brief SIMD-accelerated Point Cloud Alignment for NEON BF16.
 *  @file include/numkong/mesh/neonbfdot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section mesh_neonbfdot_instructions ARM NEON BF16 Instructions (ARMv8.6-BF16)
 *
 *      Intrinsic    Instruction              A76       M5
 *      vld3_u16     LD3 (V.4H x 3)           4cy @ 1p  4cy @ 1p
 *      vshll_n_u16  USHLL (V.4S, V.4H, #16)  2cy @ 2p  2cy @ 4p
 *      vfmaq_f32    FMLA (V.4S, V.4S, V.4S)  4cy @ 2p  3cy @ 4p
 *      vaddq_f32    FADD (V.4S, V.4S, V.4S)  2cy @ 2p  2cy @ 4p
 *      vsubq_f32    FSUB (V.4S, V.4S, V.4S)  2cy @ 2p  2cy @ 4p
 *      vmulq_f32    FMUL (V.4S, V.4S, V.4S)  3cy @ 2p  3cy @ 4p
 *      vdupq_n_f32  DUP (V.4S, scalar)       2cy @ 2p  2cy @ 4p
 *      vaddvq_f32   FADDP+FADDP (V.4S)       5cy @ 1p  8cy @ 1p
 *
 *  The ARMv8.6-BF16 extension enables BF16 storage with F32 computation for 3D mesh alignment
 *  operations. BF16's wider exponent range (matching F32) prevents overflow in geometric calculations
 *  while halving memory bandwidth compared to F32.
 *
 *  For point cloud registration (RMSD, Kabsch, Umeyama), BF16 data is loaded using VLD3 de-interleave
 *  operations, converted to F32 via bit-shift widening, then processed with F32 FMA chains. The 2x
 *  unrolling with dual accumulators hides the 4-cycle FMA latency, achieving near-peak throughput
 *  for covariance matrix and centroid computations.
 */
#ifndef NK_MESH_NEONBFDOT_H
#define NK_MESH_NEONBFDOT_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEONBFDOT

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16+fp16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16+fp16")
#endif

/*  Load 4 bf16 xyz points (12 bf16 values) → 3x float32x4_t.
 *  Uses vld3_u16 to de-interleave xyz triplets, then converts bf16 to f32.
 *
 *  Input: 12 contiguous bf16 [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
 *  Output: x[4], y[4], z[4] vectors in f32
 */
NK_INTERNAL void nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(nk_bf16_t const *ptr, float32x4_t *x_out,
                                                            float32x4_t *y_out, float32x4_t *z_out) {
    // Load 12 bf16 values and de-interleave into x, y, z components
    uint16x4x3_t xyz_u16x4x3 = vld3_u16((nk_u16_t const *)ptr);
    // Convert bf16 to f32 by zero-extending to lower 16 bits, then shifting left by 16
    uint32x4_t x_u32x4 = vshll_n_u16(xyz_u16x4x3.val[0], 16);
    uint32x4_t y_u32x4 = vshll_n_u16(xyz_u16x4x3.val[1], 16);
    uint32x4_t z_u32x4 = vshll_n_u16(xyz_u16x4x3.val[2], 16);
    *x_out = vreinterpretq_f32_u32(x_u32x4);
    *y_out = vreinterpretq_f32_u32(y_u32x4);
    *z_out = vreinterpretq_f32_u32(z_u32x4);
}

NK_INTERNAL void nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(nk_bf16_t const *ptr, nk_size_t n_points,
                                                                  float32x4_t *x_out, float32x4_t *y_out,
                                                                  float32x4_t *z_out) {
    nk_u16_t buf[12] = {0};
    nk_u16_t const *src = (nk_u16_t const *)ptr;
    for (nk_size_t k = 0; k < n_points * 3; ++k) buf[k] = src[k];
    nk_deinterleave_bf16x4_to_f32x4_neonbfdot_((nk_bf16_t const *)buf, x_out, y_out, z_out);
}

/*  Compute sum of squared distances for bf16 data after applying rotation (and optional scale).
 *  Loads bf16 data, converts to f32 during processing.
 *  Note: rotation matrix r is f32 (from SVD), scale and computation done in f32.
 */
NK_INTERNAL nk_f32_t nk_transformed_ssd_bf16_neonbfdot_(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                        nk_f32_t const *r, nk_f32_t scale, nk_f32_t centroid_a_x,
                                                        nk_f32_t centroid_a_y, nk_f32_t centroid_a_z,
                                                        nk_f32_t centroid_b_x, nk_f32_t centroid_b_y,
                                                        nk_f32_t centroid_b_z) {
    // Broadcast scaled rotation matrix elements
    float32x4_t scaled_rotation_x_x_f32x4 = vdupq_n_f32(scale * r[0]);
    float32x4_t scaled_rotation_x_y_f32x4 = vdupq_n_f32(scale * r[1]);
    float32x4_t scaled_rotation_x_z_f32x4 = vdupq_n_f32(scale * r[2]);
    float32x4_t scaled_rotation_y_x_f32x4 = vdupq_n_f32(scale * r[3]);
    float32x4_t scaled_rotation_y_y_f32x4 = vdupq_n_f32(scale * r[4]);
    float32x4_t scaled_rotation_y_z_f32x4 = vdupq_n_f32(scale * r[5]);
    float32x4_t scaled_rotation_z_x_f32x4 = vdupq_n_f32(scale * r[6]);
    float32x4_t scaled_rotation_z_y_f32x4 = vdupq_n_f32(scale * r[7]);
    float32x4_t scaled_rotation_z_z_f32x4 = vdupq_n_f32(scale * r[8]);

    // Broadcast centroids
    float32x4_t centroid_a_x_f32x4 = vdupq_n_f32(centroid_a_x);
    float32x4_t centroid_a_y_f32x4 = vdupq_n_f32(centroid_a_y);
    float32x4_t centroid_a_z_f32x4 = vdupq_n_f32(centroid_a_z);
    float32x4_t centroid_b_x_f32x4 = vdupq_n_f32(centroid_b_x);
    float32x4_t centroid_b_y_f32x4 = vdupq_n_f32(centroid_b_y);
    float32x4_t centroid_b_z_f32x4 = vdupq_n_f32(centroid_b_z);

    // Two independent accumulators to hide FMA latency
    float32x4_t sum_squared_a_f32x4 = vdupq_n_f32(0);
    float32x4_t sum_squared_b_f32x4 = vdupq_n_f32(0);
    nk_size_t j = 0;

    // Main loop: process 8 points per iteration (2x unrolled)
    for (; j + 8 <= n; j += 8) {
        // First batch of 4 points
        float32x4_t a1_x_f32x4, a1_y_f32x4, a1_z_f32x4, b1_x_f32x4, b1_y_f32x4, b1_z_f32x4;
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(a + j * 3, &a1_x_f32x4, &a1_y_f32x4, &a1_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(b + j * 3, &b1_x_f32x4, &b1_y_f32x4, &b1_z_f32x4);

        // Second batch of 4 points
        float32x4_t a2_x_f32x4, a2_y_f32x4, a2_z_f32x4, b2_x_f32x4, b2_y_f32x4, b2_z_f32x4;
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(a + (j + 4) * 3, &a2_x_f32x4, &a2_y_f32x4, &a2_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(b + (j + 4) * 3, &b2_x_f32x4, &b2_y_f32x4, &b2_z_f32x4);

        // Center first batch
        float32x4_t pa1_x_f32x4 = vsubq_f32(a1_x_f32x4, centroid_a_x_f32x4);
        float32x4_t pa1_y_f32x4 = vsubq_f32(a1_y_f32x4, centroid_a_y_f32x4);
        float32x4_t pa1_z_f32x4 = vsubq_f32(a1_z_f32x4, centroid_a_z_f32x4);
        float32x4_t pb1_x_f32x4 = vsubq_f32(b1_x_f32x4, centroid_b_x_f32x4);
        float32x4_t pb1_y_f32x4 = vsubq_f32(b1_y_f32x4, centroid_b_y_f32x4);
        float32x4_t pb1_z_f32x4 = vsubq_f32(b1_z_f32x4, centroid_b_z_f32x4);

        // Center second batch
        float32x4_t pa2_x_f32x4 = vsubq_f32(a2_x_f32x4, centroid_a_x_f32x4);
        float32x4_t pa2_y_f32x4 = vsubq_f32(a2_y_f32x4, centroid_a_y_f32x4);
        float32x4_t pa2_z_f32x4 = vsubq_f32(a2_z_f32x4, centroid_a_z_f32x4);
        float32x4_t pb2_x_f32x4 = vsubq_f32(b2_x_f32x4, centroid_b_x_f32x4);
        float32x4_t pb2_y_f32x4 = vsubq_f32(b2_y_f32x4, centroid_b_y_f32x4);
        float32x4_t pb2_z_f32x4 = vsubq_f32(b2_z_f32x4, centroid_b_z_f32x4);

        // Rotate and scale first batch: ra1 = scale * R * pa1
        float32x4_t ra1_x_f32x4 = vfmaq_f32(
            vfmaq_f32(vmulq_f32(scaled_rotation_x_x_f32x4, pa1_x_f32x4), scaled_rotation_x_y_f32x4, pa1_y_f32x4),
            scaled_rotation_x_z_f32x4, pa1_z_f32x4);
        float32x4_t ra1_y_f32x4 = vfmaq_f32(
            vfmaq_f32(vmulq_f32(scaled_rotation_y_x_f32x4, pa1_x_f32x4), scaled_rotation_y_y_f32x4, pa1_y_f32x4),
            scaled_rotation_y_z_f32x4, pa1_z_f32x4);
        float32x4_t ra1_z_f32x4 = vfmaq_f32(
            vfmaq_f32(vmulq_f32(scaled_rotation_z_x_f32x4, pa1_x_f32x4), scaled_rotation_z_y_f32x4, pa1_y_f32x4),
            scaled_rotation_z_z_f32x4, pa1_z_f32x4);

        // Rotate and scale second batch: ra2 = scale * R * pa2
        float32x4_t ra2_x_f32x4 = vfmaq_f32(
            vfmaq_f32(vmulq_f32(scaled_rotation_x_x_f32x4, pa2_x_f32x4), scaled_rotation_x_y_f32x4, pa2_y_f32x4),
            scaled_rotation_x_z_f32x4, pa2_z_f32x4);
        float32x4_t ra2_y_f32x4 = vfmaq_f32(
            vfmaq_f32(vmulq_f32(scaled_rotation_y_x_f32x4, pa2_x_f32x4), scaled_rotation_y_y_f32x4, pa2_y_f32x4),
            scaled_rotation_y_z_f32x4, pa2_z_f32x4);
        float32x4_t ra2_z_f32x4 = vfmaq_f32(
            vfmaq_f32(vmulq_f32(scaled_rotation_z_x_f32x4, pa2_x_f32x4), scaled_rotation_z_y_f32x4, pa2_y_f32x4),
            scaled_rotation_z_z_f32x4, pa2_z_f32x4);

        // Deltas
        float32x4_t delta1_x_f32x4 = vsubq_f32(ra1_x_f32x4, pb1_x_f32x4);
        float32x4_t delta1_y_f32x4 = vsubq_f32(ra1_y_f32x4, pb1_y_f32x4);
        float32x4_t delta1_z_f32x4 = vsubq_f32(ra1_z_f32x4, pb1_z_f32x4);
        float32x4_t delta2_x_f32x4 = vsubq_f32(ra2_x_f32x4, pb2_x_f32x4);
        float32x4_t delta2_y_f32x4 = vsubq_f32(ra2_y_f32x4, pb2_y_f32x4);
        float32x4_t delta2_z_f32x4 = vsubq_f32(ra2_z_f32x4, pb2_z_f32x4);

        // Accumulate to independent accumulators
        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta1_x_f32x4, delta1_x_f32x4);
        sum_squared_b_f32x4 = vfmaq_f32(sum_squared_b_f32x4, delta2_x_f32x4, delta2_x_f32x4);
        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta1_y_f32x4, delta1_y_f32x4);
        sum_squared_b_f32x4 = vfmaq_f32(sum_squared_b_f32x4, delta2_y_f32x4, delta2_y_f32x4);
        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta1_z_f32x4, delta1_z_f32x4);
        sum_squared_b_f32x4 = vfmaq_f32(sum_squared_b_f32x4, delta2_z_f32x4, delta2_z_f32x4);
    }

    // Handle remaining 4 points
    if (j + 4 <= n) {
        float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(a + j * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(b + j * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        float32x4_t pa_x_f32x4 = vsubq_f32(a_x_f32x4, centroid_a_x_f32x4);
        float32x4_t pa_y_f32x4 = vsubq_f32(a_y_f32x4, centroid_a_y_f32x4);
        float32x4_t pa_z_f32x4 = vsubq_f32(a_z_f32x4, centroid_a_z_f32x4);
        float32x4_t pb_x_f32x4 = vsubq_f32(b_x_f32x4, centroid_b_x_f32x4);
        float32x4_t pb_y_f32x4 = vsubq_f32(b_y_f32x4, centroid_b_y_f32x4);
        float32x4_t pb_z_f32x4 = vsubq_f32(b_z_f32x4, centroid_b_z_f32x4);

        float32x4_t ra_x_f32x4 = vfmaq_f32(
            vfmaq_f32(vmulq_f32(scaled_rotation_x_x_f32x4, pa_x_f32x4), scaled_rotation_x_y_f32x4, pa_y_f32x4),
            scaled_rotation_x_z_f32x4, pa_z_f32x4);
        float32x4_t ra_y_f32x4 = vfmaq_f32(
            vfmaq_f32(vmulq_f32(scaled_rotation_y_x_f32x4, pa_x_f32x4), scaled_rotation_y_y_f32x4, pa_y_f32x4),
            scaled_rotation_y_z_f32x4, pa_z_f32x4);
        float32x4_t ra_z_f32x4 = vfmaq_f32(
            vfmaq_f32(vmulq_f32(scaled_rotation_z_x_f32x4, pa_x_f32x4), scaled_rotation_z_y_f32x4, pa_y_f32x4),
            scaled_rotation_z_z_f32x4, pa_z_f32x4);

        float32x4_t delta_x_f32x4 = vsubq_f32(ra_x_f32x4, pb_x_f32x4);
        float32x4_t delta_y_f32x4 = vsubq_f32(ra_y_f32x4, pb_y_f32x4);
        float32x4_t delta_z_f32x4 = vsubq_f32(ra_z_f32x4, pb_z_f32x4);

        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta_x_f32x4, delta_x_f32x4);
        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta_y_f32x4, delta_y_f32x4);
        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta_z_f32x4, delta_z_f32x4);
        j += 4;
    }

    // Partial tail: handle remaining 1-3 points with vectorized partial deinterleave
    if (j < n) {
        float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(a + j * 3, n - j, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(b + j * 3, n - j, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        // Mask invalid lanes to zero BEFORE centering
        uint32x4_t lane_u32x4 = vcombine_u32(vreinterpret_u32_u64(vcreate_u64(0x0000000100000000ULL)),
                                             vreinterpret_u32_u64(vcreate_u64(0x0000000300000002ULL)));
        uint32x4_t valid_u32x4 = vcltq_u32(lane_u32x4, vdupq_n_u32((nk_u32_t)(n - j)));
        float32x4_t zero_f32x4 = vdupq_n_f32(0);
        a_x_f32x4 = vbslq_f32(valid_u32x4, a_x_f32x4, zero_f32x4);
        a_y_f32x4 = vbslq_f32(valid_u32x4, a_y_f32x4, zero_f32x4);
        a_z_f32x4 = vbslq_f32(valid_u32x4, a_z_f32x4, zero_f32x4);
        b_x_f32x4 = vbslq_f32(valid_u32x4, b_x_f32x4, zero_f32x4);
        b_y_f32x4 = vbslq_f32(valid_u32x4, b_y_f32x4, zero_f32x4);
        b_z_f32x4 = vbslq_f32(valid_u32x4, b_z_f32x4, zero_f32x4);

        // Same centering + rotation + delta + FMA as body
        float32x4_t pa_x_f32x4 = vsubq_f32(a_x_f32x4, centroid_a_x_f32x4);
        float32x4_t pa_y_f32x4 = vsubq_f32(a_y_f32x4, centroid_a_y_f32x4);
        float32x4_t pa_z_f32x4 = vsubq_f32(a_z_f32x4, centroid_a_z_f32x4);
        float32x4_t pb_x_f32x4 = vsubq_f32(b_x_f32x4, centroid_b_x_f32x4);
        float32x4_t pb_y_f32x4 = vsubq_f32(b_y_f32x4, centroid_b_y_f32x4);
        float32x4_t pb_z_f32x4 = vsubq_f32(b_z_f32x4, centroid_b_z_f32x4);

        float32x4_t ra_x_f32x4 = vfmaq_f32(
            vfmaq_f32(vmulq_f32(scaled_rotation_x_x_f32x4, pa_x_f32x4), scaled_rotation_x_y_f32x4, pa_y_f32x4),
            scaled_rotation_x_z_f32x4, pa_z_f32x4);
        float32x4_t ra_y_f32x4 = vfmaq_f32(
            vfmaq_f32(vmulq_f32(scaled_rotation_y_x_f32x4, pa_x_f32x4), scaled_rotation_y_y_f32x4, pa_y_f32x4),
            scaled_rotation_y_z_f32x4, pa_z_f32x4);
        float32x4_t ra_z_f32x4 = vfmaq_f32(
            vfmaq_f32(vmulq_f32(scaled_rotation_z_x_f32x4, pa_x_f32x4), scaled_rotation_z_y_f32x4, pa_y_f32x4),
            scaled_rotation_z_z_f32x4, pa_z_f32x4);

        float32x4_t delta_x_f32x4 = vsubq_f32(ra_x_f32x4, pb_x_f32x4);
        float32x4_t delta_y_f32x4 = vsubq_f32(ra_y_f32x4, pb_y_f32x4);
        float32x4_t delta_z_f32x4 = vsubq_f32(ra_z_f32x4, pb_z_f32x4);

        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta_x_f32x4, delta_x_f32x4);
        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta_y_f32x4, delta_y_f32x4);
        sum_squared_a_f32x4 = vfmaq_f32(sum_squared_a_f32x4, delta_z_f32x4, delta_z_f32x4);
    }

    // Combine accumulators and reduce
    float32x4_t sum_squared_f32x4 = vaddq_f32(sum_squared_a_f32x4, sum_squared_b_f32x4);
    nk_f32_t sum_squared = vaddvq_f32(sum_squared_f32x4);

    return sum_squared;
}

NK_PUBLIC void nk_rmsd_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    // RMSD uses identity rotation and scale=1.0
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    // Accumulators for squared differences
    float32x4_t sum_squared_x_f32x4 = zeros_f32x4, sum_squared_y_f32x4 = zeros_f32x4, sum_squared_z_f32x4 = zeros_f32x4;

    float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
    nk_size_t i = 0;

    // Main loop processing 4 points at a time
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(a + i * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(b + i * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        float32x4_t delta_x_f32x4 = vsubq_f32(a_x_f32x4, b_x_f32x4);
        float32x4_t delta_y_f32x4 = vsubq_f32(a_y_f32x4, b_y_f32x4);
        float32x4_t delta_z_f32x4 = vsubq_f32(a_z_f32x4, b_z_f32x4);

        sum_squared_x_f32x4 = vfmaq_f32(sum_squared_x_f32x4, delta_x_f32x4, delta_x_f32x4);
        sum_squared_y_f32x4 = vfmaq_f32(sum_squared_y_f32x4, delta_y_f32x4, delta_y_f32x4);
        sum_squared_z_f32x4 = vfmaq_f32(sum_squared_z_f32x4, delta_z_f32x4, delta_z_f32x4);
    }

    // Partial tail: handle remaining 1-3 points with vectorized partial deinterleave
    if (i < n) {
        float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
        nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(a + i * 3, n - i, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(b + i * 3, n - i, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);

        float32x4_t delta_x_f32x4 = vsubq_f32(a_x_f32x4, b_x_f32x4);
        float32x4_t delta_y_f32x4 = vsubq_f32(a_y_f32x4, b_y_f32x4);
        float32x4_t delta_z_f32x4 = vsubq_f32(a_z_f32x4, b_z_f32x4);

        sum_squared_x_f32x4 = vfmaq_f32(sum_squared_x_f32x4, delta_x_f32x4, delta_x_f32x4);
        sum_squared_y_f32x4 = vfmaq_f32(sum_squared_y_f32x4, delta_y_f32x4, delta_y_f32x4);
        sum_squared_z_f32x4 = vfmaq_f32(sum_squared_z_f32x4, delta_z_f32x4, delta_z_f32x4);
    }

    // Reduce vectors to scalars
    nk_f32_t total_squared_x = vaddvq_f32(sum_squared_x_f32x4);
    nk_f32_t total_squared_y = vaddvq_f32(sum_squared_y_f32x4);
    nk_f32_t total_squared_z = vaddvq_f32(sum_squared_z_f32x4);

    *result = nk_f32_sqrt_neon((total_squared_x + total_squared_y + total_squared_z) / (nk_f32_t)n);
}

NK_PUBLIC void nk_kabsch_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                        nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    // 2x unrolling with dual accumulators to hide FMA latency.
    float32x4_t sum_a_x_a_f32x4 = zeros_f32x4, sum_a_y_a_f32x4 = zeros_f32x4, sum_a_z_a_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_a_f32x4 = zeros_f32x4, sum_b_y_a_f32x4 = zeros_f32x4, sum_b_z_a_f32x4 = zeros_f32x4;
    float32x4_t sum_a_x_b_f32x4 = zeros_f32x4, sum_a_y_b_f32x4 = zeros_f32x4, sum_a_z_b_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_b_f32x4 = zeros_f32x4, sum_b_y_b_f32x4 = zeros_f32x4, sum_b_z_b_f32x4 = zeros_f32x4;

    float32x4_t cov_xx_a_f32x4 = zeros_f32x4, cov_xy_a_f32x4 = zeros_f32x4, cov_xz_a_f32x4 = zeros_f32x4;
    float32x4_t cov_yx_a_f32x4 = zeros_f32x4, cov_yy_a_f32x4 = zeros_f32x4, cov_yz_a_f32x4 = zeros_f32x4;
    float32x4_t cov_zx_a_f32x4 = zeros_f32x4, cov_zy_a_f32x4 = zeros_f32x4, cov_zz_a_f32x4 = zeros_f32x4;
    float32x4_t cov_xx_b_f32x4 = zeros_f32x4, cov_xy_b_f32x4 = zeros_f32x4, cov_xz_b_f32x4 = zeros_f32x4;
    float32x4_t cov_yx_b_f32x4 = zeros_f32x4, cov_yy_b_f32x4 = zeros_f32x4, cov_yz_b_f32x4 = zeros_f32x4;
    float32x4_t cov_zx_b_f32x4 = zeros_f32x4, cov_zy_b_f32x4 = zeros_f32x4, cov_zz_b_f32x4 = zeros_f32x4;

    nk_size_t i = 0;
    float32x4_t a1_x_f32x4, a1_y_f32x4, a1_z_f32x4, b1_x_f32x4, b1_y_f32x4, b1_z_f32x4;
    float32x4_t a2_x_f32x4, a2_y_f32x4, a2_z_f32x4, b2_x_f32x4, b2_y_f32x4, b2_z_f32x4;

    // Main loop: 8 points per iteration (2x unrolled)
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(a + i * 3, &a1_x_f32x4, &a1_y_f32x4, &a1_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(b + i * 3, &b1_x_f32x4, &b1_y_f32x4, &b1_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(a + (i + 4) * 3, &a2_x_f32x4, &a2_y_f32x4, &a2_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(b + (i + 4) * 3, &b2_x_f32x4, &b2_y_f32x4, &b2_z_f32x4);

        // Interleaved accumulation to hide FMA latency
        sum_a_x_a_f32x4 = vaddq_f32(sum_a_x_a_f32x4, a1_x_f32x4);
        sum_a_x_b_f32x4 = vaddq_f32(sum_a_x_b_f32x4, a2_x_f32x4);
        sum_a_y_a_f32x4 = vaddq_f32(sum_a_y_a_f32x4, a1_y_f32x4);
        sum_a_y_b_f32x4 = vaddq_f32(sum_a_y_b_f32x4, a2_y_f32x4);
        sum_a_z_a_f32x4 = vaddq_f32(sum_a_z_a_f32x4, a1_z_f32x4);
        sum_a_z_b_f32x4 = vaddq_f32(sum_a_z_b_f32x4, a2_z_f32x4);
        sum_b_x_a_f32x4 = vaddq_f32(sum_b_x_a_f32x4, b1_x_f32x4);
        sum_b_x_b_f32x4 = vaddq_f32(sum_b_x_b_f32x4, b2_x_f32x4);
        sum_b_y_a_f32x4 = vaddq_f32(sum_b_y_a_f32x4, b1_y_f32x4);
        sum_b_y_b_f32x4 = vaddq_f32(sum_b_y_b_f32x4, b2_y_f32x4);
        sum_b_z_a_f32x4 = vaddq_f32(sum_b_z_a_f32x4, b1_z_f32x4);
        sum_b_z_b_f32x4 = vaddq_f32(sum_b_z_b_f32x4, b2_z_f32x4);

        cov_xx_a_f32x4 = vfmaq_f32(cov_xx_a_f32x4, a1_x_f32x4, b1_x_f32x4);
        cov_xx_b_f32x4 = vfmaq_f32(cov_xx_b_f32x4, a2_x_f32x4, b2_x_f32x4);
        cov_xy_a_f32x4 = vfmaq_f32(cov_xy_a_f32x4, a1_x_f32x4, b1_y_f32x4);
        cov_xy_b_f32x4 = vfmaq_f32(cov_xy_b_f32x4, a2_x_f32x4, b2_y_f32x4);
        cov_xz_a_f32x4 = vfmaq_f32(cov_xz_a_f32x4, a1_x_f32x4, b1_z_f32x4);
        cov_xz_b_f32x4 = vfmaq_f32(cov_xz_b_f32x4, a2_x_f32x4, b2_z_f32x4);
        cov_yx_a_f32x4 = vfmaq_f32(cov_yx_a_f32x4, a1_y_f32x4, b1_x_f32x4);
        cov_yx_b_f32x4 = vfmaq_f32(cov_yx_b_f32x4, a2_y_f32x4, b2_x_f32x4);
        cov_yy_a_f32x4 = vfmaq_f32(cov_yy_a_f32x4, a1_y_f32x4, b1_y_f32x4);
        cov_yy_b_f32x4 = vfmaq_f32(cov_yy_b_f32x4, a2_y_f32x4, b2_y_f32x4);
        cov_yz_a_f32x4 = vfmaq_f32(cov_yz_a_f32x4, a1_y_f32x4, b1_z_f32x4);
        cov_yz_b_f32x4 = vfmaq_f32(cov_yz_b_f32x4, a2_y_f32x4, b2_z_f32x4);
        cov_zx_a_f32x4 = vfmaq_f32(cov_zx_a_f32x4, a1_z_f32x4, b1_x_f32x4);
        cov_zx_b_f32x4 = vfmaq_f32(cov_zx_b_f32x4, a2_z_f32x4, b2_x_f32x4);
        cov_zy_a_f32x4 = vfmaq_f32(cov_zy_a_f32x4, a1_z_f32x4, b1_y_f32x4);
        cov_zy_b_f32x4 = vfmaq_f32(cov_zy_b_f32x4, a2_z_f32x4, b2_y_f32x4);
        cov_zz_a_f32x4 = vfmaq_f32(cov_zz_a_f32x4, a1_z_f32x4, b1_z_f32x4);
        cov_zz_b_f32x4 = vfmaq_f32(cov_zz_b_f32x4, a2_z_f32x4, b2_z_f32x4);
    }

    // 4-point tail
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(a + i * 3, &a1_x_f32x4, &a1_y_f32x4, &a1_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(b + i * 3, &b1_x_f32x4, &b1_y_f32x4, &b1_z_f32x4);
        sum_a_x_a_f32x4 = vaddq_f32(sum_a_x_a_f32x4, a1_x_f32x4);
        sum_a_y_a_f32x4 = vaddq_f32(sum_a_y_a_f32x4, a1_y_f32x4);
        sum_a_z_a_f32x4 = vaddq_f32(sum_a_z_a_f32x4, a1_z_f32x4);
        sum_b_x_a_f32x4 = vaddq_f32(sum_b_x_a_f32x4, b1_x_f32x4);
        sum_b_y_a_f32x4 = vaddq_f32(sum_b_y_a_f32x4, b1_y_f32x4);
        sum_b_z_a_f32x4 = vaddq_f32(sum_b_z_a_f32x4, b1_z_f32x4);
        cov_xx_a_f32x4 = vfmaq_f32(cov_xx_a_f32x4, a1_x_f32x4, b1_x_f32x4);
        cov_xy_a_f32x4 = vfmaq_f32(cov_xy_a_f32x4, a1_x_f32x4, b1_y_f32x4);
        cov_xz_a_f32x4 = vfmaq_f32(cov_xz_a_f32x4, a1_x_f32x4, b1_z_f32x4);
        cov_yx_a_f32x4 = vfmaq_f32(cov_yx_a_f32x4, a1_y_f32x4, b1_x_f32x4);
        cov_yy_a_f32x4 = vfmaq_f32(cov_yy_a_f32x4, a1_y_f32x4, b1_y_f32x4);
        cov_yz_a_f32x4 = vfmaq_f32(cov_yz_a_f32x4, a1_y_f32x4, b1_z_f32x4);
        cov_zx_a_f32x4 = vfmaq_f32(cov_zx_a_f32x4, a1_z_f32x4, b1_x_f32x4);
        cov_zy_a_f32x4 = vfmaq_f32(cov_zy_a_f32x4, a1_z_f32x4, b1_y_f32x4);
        cov_zz_a_f32x4 = vfmaq_f32(cov_zz_a_f32x4, a1_z_f32x4, b1_z_f32x4);
    }

    // Partial tail: handle remaining 1-3 points with vectorized partial deinterleave
    if (i < n) {
        nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(a + i * 3, n - i, &a1_x_f32x4, &a1_y_f32x4, &a1_z_f32x4);
        nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(b + i * 3, n - i, &b1_x_f32x4, &b1_y_f32x4, &b1_z_f32x4);
        sum_a_x_a_f32x4 = vaddq_f32(sum_a_x_a_f32x4, a1_x_f32x4);
        sum_a_y_a_f32x4 = vaddq_f32(sum_a_y_a_f32x4, a1_y_f32x4);
        sum_a_z_a_f32x4 = vaddq_f32(sum_a_z_a_f32x4, a1_z_f32x4);
        sum_b_x_a_f32x4 = vaddq_f32(sum_b_x_a_f32x4, b1_x_f32x4);
        sum_b_y_a_f32x4 = vaddq_f32(sum_b_y_a_f32x4, b1_y_f32x4);
        sum_b_z_a_f32x4 = vaddq_f32(sum_b_z_a_f32x4, b1_z_f32x4);
        cov_xx_a_f32x4 = vfmaq_f32(cov_xx_a_f32x4, a1_x_f32x4, b1_x_f32x4);
        cov_xy_a_f32x4 = vfmaq_f32(cov_xy_a_f32x4, a1_x_f32x4, b1_y_f32x4);
        cov_xz_a_f32x4 = vfmaq_f32(cov_xz_a_f32x4, a1_x_f32x4, b1_z_f32x4);
        cov_yx_a_f32x4 = vfmaq_f32(cov_yx_a_f32x4, a1_y_f32x4, b1_x_f32x4);
        cov_yy_a_f32x4 = vfmaq_f32(cov_yy_a_f32x4, a1_y_f32x4, b1_y_f32x4);
        cov_yz_a_f32x4 = vfmaq_f32(cov_yz_a_f32x4, a1_y_f32x4, b1_z_f32x4);
        cov_zx_a_f32x4 = vfmaq_f32(cov_zx_a_f32x4, a1_z_f32x4, b1_x_f32x4);
        cov_zy_a_f32x4 = vfmaq_f32(cov_zy_a_f32x4, a1_z_f32x4, b1_y_f32x4);
        cov_zz_a_f32x4 = vfmaq_f32(cov_zz_a_f32x4, a1_z_f32x4, b1_z_f32x4);
    }

    // Combine dual accumulators
    float32x4_t sum_a_x_f32x4 = vaddq_f32(sum_a_x_a_f32x4, sum_a_x_b_f32x4);
    float32x4_t sum_a_y_f32x4 = vaddq_f32(sum_a_y_a_f32x4, sum_a_y_b_f32x4);
    float32x4_t sum_a_z_f32x4 = vaddq_f32(sum_a_z_a_f32x4, sum_a_z_b_f32x4);
    float32x4_t sum_b_x_f32x4 = vaddq_f32(sum_b_x_a_f32x4, sum_b_x_b_f32x4);
    float32x4_t sum_b_y_f32x4 = vaddq_f32(sum_b_y_a_f32x4, sum_b_y_b_f32x4);
    float32x4_t sum_b_z_f32x4 = vaddq_f32(sum_b_z_a_f32x4, sum_b_z_b_f32x4);
    float32x4_t cov_xx_f32x4 = vaddq_f32(cov_xx_a_f32x4, cov_xx_b_f32x4);
    float32x4_t cov_xy_f32x4 = vaddq_f32(cov_xy_a_f32x4, cov_xy_b_f32x4);
    float32x4_t cov_xz_f32x4 = vaddq_f32(cov_xz_a_f32x4, cov_xz_b_f32x4);
    float32x4_t cov_yx_f32x4 = vaddq_f32(cov_yx_a_f32x4, cov_yx_b_f32x4);
    float32x4_t cov_yy_f32x4 = vaddq_f32(cov_yy_a_f32x4, cov_yy_b_f32x4);
    float32x4_t cov_yz_f32x4 = vaddq_f32(cov_yz_a_f32x4, cov_yz_b_f32x4);
    float32x4_t cov_zx_f32x4 = vaddq_f32(cov_zx_a_f32x4, cov_zx_b_f32x4);
    float32x4_t cov_zy_f32x4 = vaddq_f32(cov_zy_a_f32x4, cov_zy_b_f32x4);
    float32x4_t cov_zz_f32x4 = vaddq_f32(cov_zz_a_f32x4, cov_zz_b_f32x4);

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
    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

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

    // Handle reflection: if det(R) < 0, negate third column of V and recompute R
    if (nk_det3x3_f32_(r) < 0) {
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

    // Output rotation matrix and scale=1.0
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];
    if (scale) *scale = 1.0f;

    // Compute RMSD after optimal rotation
    nk_f32_t sum_squared = nk_transformed_ssd_bf16_neonbfdot_(a, b, n, r, 1.0f, centroid_a_x, centroid_a_y,
                                                              centroid_a_z, centroid_b_x, centroid_b_y, centroid_b_z);
    *result = nk_f32_sqrt_neon(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    // 2x unrolling with dual accumulators to hide FMA latency.
    float32x4_t sum_a_x_a_f32x4 = zeros_f32x4, sum_a_y_a_f32x4 = zeros_f32x4, sum_a_z_a_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_a_f32x4 = zeros_f32x4, sum_b_y_a_f32x4 = zeros_f32x4, sum_b_z_a_f32x4 = zeros_f32x4;
    float32x4_t sum_a_x_b_f32x4 = zeros_f32x4, sum_a_y_b_f32x4 = zeros_f32x4, sum_a_z_b_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_b_f32x4 = zeros_f32x4, sum_b_y_b_f32x4 = zeros_f32x4, sum_b_z_b_f32x4 = zeros_f32x4;

    float32x4_t cov_xx_a_f32x4 = zeros_f32x4, cov_xy_a_f32x4 = zeros_f32x4, cov_xz_a_f32x4 = zeros_f32x4;
    float32x4_t cov_yx_a_f32x4 = zeros_f32x4, cov_yy_a_f32x4 = zeros_f32x4, cov_yz_a_f32x4 = zeros_f32x4;
    float32x4_t cov_zx_a_f32x4 = zeros_f32x4, cov_zy_a_f32x4 = zeros_f32x4, cov_zz_a_f32x4 = zeros_f32x4;
    float32x4_t cov_xx_b_f32x4 = zeros_f32x4, cov_xy_b_f32x4 = zeros_f32x4, cov_xz_b_f32x4 = zeros_f32x4;
    float32x4_t cov_yx_b_f32x4 = zeros_f32x4, cov_yy_b_f32x4 = zeros_f32x4, cov_yz_b_f32x4 = zeros_f32x4;
    float32x4_t cov_zx_b_f32x4 = zeros_f32x4, cov_zy_b_f32x4 = zeros_f32x4, cov_zz_b_f32x4 = zeros_f32x4;

    // Variance of A accumulators
    float32x4_t variance_a_a_f32x4 = zeros_f32x4;
    float32x4_t variance_a_b_f32x4 = zeros_f32x4;

    nk_size_t i = 0;
    float32x4_t a1_x_f32x4, a1_y_f32x4, a1_z_f32x4, b1_x_f32x4, b1_y_f32x4, b1_z_f32x4;
    float32x4_t a2_x_f32x4, a2_y_f32x4, a2_z_f32x4, b2_x_f32x4, b2_y_f32x4, b2_z_f32x4;

    // Main loop: 8 points per iteration (2x unrolled)
    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(a + i * 3, &a1_x_f32x4, &a1_y_f32x4, &a1_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(b + i * 3, &b1_x_f32x4, &b1_y_f32x4, &b1_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(a + (i + 4) * 3, &a2_x_f32x4, &a2_y_f32x4, &a2_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(b + (i + 4) * 3, &b2_x_f32x4, &b2_y_f32x4, &b2_z_f32x4);

        // Interleaved accumulation to hide FMA latency
        sum_a_x_a_f32x4 = vaddq_f32(sum_a_x_a_f32x4, a1_x_f32x4);
        sum_a_x_b_f32x4 = vaddq_f32(sum_a_x_b_f32x4, a2_x_f32x4);
        sum_a_y_a_f32x4 = vaddq_f32(sum_a_y_a_f32x4, a1_y_f32x4);
        sum_a_y_b_f32x4 = vaddq_f32(sum_a_y_b_f32x4, a2_y_f32x4);
        sum_a_z_a_f32x4 = vaddq_f32(sum_a_z_a_f32x4, a1_z_f32x4);
        sum_a_z_b_f32x4 = vaddq_f32(sum_a_z_b_f32x4, a2_z_f32x4);
        sum_b_x_a_f32x4 = vaddq_f32(sum_b_x_a_f32x4, b1_x_f32x4);
        sum_b_x_b_f32x4 = vaddq_f32(sum_b_x_b_f32x4, b2_x_f32x4);
        sum_b_y_a_f32x4 = vaddq_f32(sum_b_y_a_f32x4, b1_y_f32x4);
        sum_b_y_b_f32x4 = vaddq_f32(sum_b_y_b_f32x4, b2_y_f32x4);
        sum_b_z_a_f32x4 = vaddq_f32(sum_b_z_a_f32x4, b1_z_f32x4);
        sum_b_z_b_f32x4 = vaddq_f32(sum_b_z_b_f32x4, b2_z_f32x4);

        // Covariance matrix
        cov_xx_a_f32x4 = vfmaq_f32(cov_xx_a_f32x4, a1_x_f32x4, b1_x_f32x4);
        cov_xx_b_f32x4 = vfmaq_f32(cov_xx_b_f32x4, a2_x_f32x4, b2_x_f32x4);
        cov_xy_a_f32x4 = vfmaq_f32(cov_xy_a_f32x4, a1_x_f32x4, b1_y_f32x4);
        cov_xy_b_f32x4 = vfmaq_f32(cov_xy_b_f32x4, a2_x_f32x4, b2_y_f32x4);
        cov_xz_a_f32x4 = vfmaq_f32(cov_xz_a_f32x4, a1_x_f32x4, b1_z_f32x4);
        cov_xz_b_f32x4 = vfmaq_f32(cov_xz_b_f32x4, a2_x_f32x4, b2_z_f32x4);
        cov_yx_a_f32x4 = vfmaq_f32(cov_yx_a_f32x4, a1_y_f32x4, b1_x_f32x4);
        cov_yx_b_f32x4 = vfmaq_f32(cov_yx_b_f32x4, a2_y_f32x4, b2_x_f32x4);
        cov_yy_a_f32x4 = vfmaq_f32(cov_yy_a_f32x4, a1_y_f32x4, b1_y_f32x4);
        cov_yy_b_f32x4 = vfmaq_f32(cov_yy_b_f32x4, a2_y_f32x4, b2_y_f32x4);
        cov_yz_a_f32x4 = vfmaq_f32(cov_yz_a_f32x4, a1_y_f32x4, b1_z_f32x4);
        cov_yz_b_f32x4 = vfmaq_f32(cov_yz_b_f32x4, a2_y_f32x4, b2_z_f32x4);
        cov_zx_a_f32x4 = vfmaq_f32(cov_zx_a_f32x4, a1_z_f32x4, b1_x_f32x4);
        cov_zx_b_f32x4 = vfmaq_f32(cov_zx_b_f32x4, a2_z_f32x4, b2_x_f32x4);
        cov_zy_a_f32x4 = vfmaq_f32(cov_zy_a_f32x4, a1_z_f32x4, b1_y_f32x4);
        cov_zy_b_f32x4 = vfmaq_f32(cov_zy_b_f32x4, a2_z_f32x4, b2_y_f32x4);
        cov_zz_a_f32x4 = vfmaq_f32(cov_zz_a_f32x4, a1_z_f32x4, b1_z_f32x4);
        cov_zz_b_f32x4 = vfmaq_f32(cov_zz_b_f32x4, a2_z_f32x4, b2_z_f32x4);

        // Variance of A
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_x_f32x4, a1_x_f32x4);
        variance_a_b_f32x4 = vfmaq_f32(variance_a_b_f32x4, a2_x_f32x4, a2_x_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_y_f32x4, a1_y_f32x4);
        variance_a_b_f32x4 = vfmaq_f32(variance_a_b_f32x4, a2_y_f32x4, a2_y_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_z_f32x4, a1_z_f32x4);
        variance_a_b_f32x4 = vfmaq_f32(variance_a_b_f32x4, a2_z_f32x4, a2_z_f32x4);
    }

    // 4-point tail
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(a + i * 3, &a1_x_f32x4, &a1_y_f32x4, &a1_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(b + i * 3, &b1_x_f32x4, &b1_y_f32x4, &b1_z_f32x4);
        sum_a_x_a_f32x4 = vaddq_f32(sum_a_x_a_f32x4, a1_x_f32x4);
        sum_a_y_a_f32x4 = vaddq_f32(sum_a_y_a_f32x4, a1_y_f32x4);
        sum_a_z_a_f32x4 = vaddq_f32(sum_a_z_a_f32x4, a1_z_f32x4);
        sum_b_x_a_f32x4 = vaddq_f32(sum_b_x_a_f32x4, b1_x_f32x4);
        sum_b_y_a_f32x4 = vaddq_f32(sum_b_y_a_f32x4, b1_y_f32x4);
        sum_b_z_a_f32x4 = vaddq_f32(sum_b_z_a_f32x4, b1_z_f32x4);
        cov_xx_a_f32x4 = vfmaq_f32(cov_xx_a_f32x4, a1_x_f32x4, b1_x_f32x4);
        cov_xy_a_f32x4 = vfmaq_f32(cov_xy_a_f32x4, a1_x_f32x4, b1_y_f32x4);
        cov_xz_a_f32x4 = vfmaq_f32(cov_xz_a_f32x4, a1_x_f32x4, b1_z_f32x4);
        cov_yx_a_f32x4 = vfmaq_f32(cov_yx_a_f32x4, a1_y_f32x4, b1_x_f32x4);
        cov_yy_a_f32x4 = vfmaq_f32(cov_yy_a_f32x4, a1_y_f32x4, b1_y_f32x4);
        cov_yz_a_f32x4 = vfmaq_f32(cov_yz_a_f32x4, a1_y_f32x4, b1_z_f32x4);
        cov_zx_a_f32x4 = vfmaq_f32(cov_zx_a_f32x4, a1_z_f32x4, b1_x_f32x4);
        cov_zy_a_f32x4 = vfmaq_f32(cov_zy_a_f32x4, a1_z_f32x4, b1_y_f32x4);
        cov_zz_a_f32x4 = vfmaq_f32(cov_zz_a_f32x4, a1_z_f32x4, b1_z_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_x_f32x4, a1_x_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_y_f32x4, a1_y_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_z_f32x4, a1_z_f32x4);
    }

    // Partial tail: handle remaining 1-3 points with vectorized partial deinterleave
    if (i < n) {
        nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(a + i * 3, n - i, &a1_x_f32x4, &a1_y_f32x4, &a1_z_f32x4);
        nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(b + i * 3, n - i, &b1_x_f32x4, &b1_y_f32x4, &b1_z_f32x4);
        sum_a_x_a_f32x4 = vaddq_f32(sum_a_x_a_f32x4, a1_x_f32x4);
        sum_a_y_a_f32x4 = vaddq_f32(sum_a_y_a_f32x4, a1_y_f32x4);
        sum_a_z_a_f32x4 = vaddq_f32(sum_a_z_a_f32x4, a1_z_f32x4);
        sum_b_x_a_f32x4 = vaddq_f32(sum_b_x_a_f32x4, b1_x_f32x4);
        sum_b_y_a_f32x4 = vaddq_f32(sum_b_y_a_f32x4, b1_y_f32x4);
        sum_b_z_a_f32x4 = vaddq_f32(sum_b_z_a_f32x4, b1_z_f32x4);
        cov_xx_a_f32x4 = vfmaq_f32(cov_xx_a_f32x4, a1_x_f32x4, b1_x_f32x4);
        cov_xy_a_f32x4 = vfmaq_f32(cov_xy_a_f32x4, a1_x_f32x4, b1_y_f32x4);
        cov_xz_a_f32x4 = vfmaq_f32(cov_xz_a_f32x4, a1_x_f32x4, b1_z_f32x4);
        cov_yx_a_f32x4 = vfmaq_f32(cov_yx_a_f32x4, a1_y_f32x4, b1_x_f32x4);
        cov_yy_a_f32x4 = vfmaq_f32(cov_yy_a_f32x4, a1_y_f32x4, b1_y_f32x4);
        cov_yz_a_f32x4 = vfmaq_f32(cov_yz_a_f32x4, a1_y_f32x4, b1_z_f32x4);
        cov_zx_a_f32x4 = vfmaq_f32(cov_zx_a_f32x4, a1_z_f32x4, b1_x_f32x4);
        cov_zy_a_f32x4 = vfmaq_f32(cov_zy_a_f32x4, a1_z_f32x4, b1_y_f32x4);
        cov_zz_a_f32x4 = vfmaq_f32(cov_zz_a_f32x4, a1_z_f32x4, b1_z_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_x_f32x4, a1_x_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_y_f32x4, a1_y_f32x4);
        variance_a_a_f32x4 = vfmaq_f32(variance_a_a_f32x4, a1_z_f32x4, a1_z_f32x4);
    }

    // Combine dual accumulators
    float32x4_t sum_a_x_f32x4 = vaddq_f32(sum_a_x_a_f32x4, sum_a_x_b_f32x4);
    float32x4_t sum_a_y_f32x4 = vaddq_f32(sum_a_y_a_f32x4, sum_a_y_b_f32x4);
    float32x4_t sum_a_z_f32x4 = vaddq_f32(sum_a_z_a_f32x4, sum_a_z_b_f32x4);
    float32x4_t sum_b_x_f32x4 = vaddq_f32(sum_b_x_a_f32x4, sum_b_x_b_f32x4);
    float32x4_t sum_b_y_f32x4 = vaddq_f32(sum_b_y_a_f32x4, sum_b_y_b_f32x4);
    float32x4_t sum_b_z_f32x4 = vaddq_f32(sum_b_z_a_f32x4, sum_b_z_b_f32x4);
    float32x4_t cov_xx_f32x4 = vaddq_f32(cov_xx_a_f32x4, cov_xx_b_f32x4);
    float32x4_t cov_xy_f32x4 = vaddq_f32(cov_xy_a_f32x4, cov_xy_b_f32x4);
    float32x4_t cov_xz_f32x4 = vaddq_f32(cov_xz_a_f32x4, cov_xz_b_f32x4);
    float32x4_t cov_yx_f32x4 = vaddq_f32(cov_yx_a_f32x4, cov_yx_b_f32x4);
    float32x4_t cov_yy_f32x4 = vaddq_f32(cov_yy_a_f32x4, cov_yy_b_f32x4);
    float32x4_t cov_yz_f32x4 = vaddq_f32(cov_yz_a_f32x4, cov_yz_b_f32x4);
    float32x4_t cov_zx_f32x4 = vaddq_f32(cov_zx_a_f32x4, cov_zx_b_f32x4);
    float32x4_t cov_zy_f32x4 = vaddq_f32(cov_zy_a_f32x4, cov_zy_b_f32x4);
    float32x4_t cov_zz_f32x4 = vaddq_f32(cov_zz_a_f32x4, cov_zz_b_f32x4);
    float32x4_t variance_a_f32x4 = vaddq_f32(variance_a_a_f32x4, variance_a_b_f32x4);

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
    nk_f32_t centroid_a_x = sum_a_x * inv_n;
    nk_f32_t centroid_a_y = sum_a_y * inv_n;
    nk_f32_t centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n;
    nk_f32_t centroid_b_y = sum_b_y * inv_n;
    nk_f32_t centroid_b_z = sum_b_z * inv_n;

    if (a_centroid) a_centroid[0] = centroid_a_x, a_centroid[1] = centroid_a_y, a_centroid[2] = centroid_a_z;
    if (b_centroid) b_centroid[0] = centroid_b_x, b_centroid[1] = centroid_b_y, b_centroid[2] = centroid_b_z;

    // Compute centered variance of A
    nk_f32_t variance_a = variance_a_sum * inv_n -
                          (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y + centroid_a_z * centroid_a_z);

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
    nk_f32_t cross_covariance[9] = {covariance_x_x, covariance_x_y, covariance_x_z, covariance_y_x, covariance_y_y,
                                    covariance_y_z, covariance_z_x, covariance_z_y, covariance_z_z};
    nk_f32_t svd_u[9], svd_s[9], svd_v[9];
    nk_svd3x3_f32_(cross_covariance, svd_u, svd_s, svd_v);

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
    // D = diag(1, 1, det(R)), svd_s contains proper positive singular values on diagonal
    nk_f32_t rotation_det = nk_det3x3_f32_(r);
    nk_f32_t sign_det = rotation_det < 0 ? -1.0f : 1.0f;
    nk_f32_t trace_scaled_s = svd_s[0] + svd_s[4] + sign_det * svd_s[8];
    nk_f32_t c = trace_scaled_s / ((nk_f32_t)n * variance_a);
    if (scale) *scale = c;

    if (rotation_det < 0) {
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

    // Output rotation matrix
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = r[j];

    // Compute RMSD after similarity transform: ‖c × R × a - b‖
    nk_f32_t sum_squared = nk_transformed_ssd_bf16_neonbfdot_(a, b, n, r, c, centroid_a_x, centroid_a_y, centroid_a_z,
                                                              centroid_b_x, centroid_b_y, centroid_b_z);
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

#endif // NK_TARGET_NEONBFDOT
#endif // NK_TARGET_ARM64_
#endif // NK_MESH_NEONBFDOT_H
