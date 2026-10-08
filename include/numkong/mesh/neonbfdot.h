/**
 *  @file include/numkong/mesh/neonbfdot.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief SIMD-accelerated point cloud alignment for NEON BF16.
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section mesh_neonbfdot_instructions ARM NEON BF16 Instructions (ARMv8.6-BF16)
 *
 *  @verbatim
 *  Intrinsic    Instruction               A76       M5
 *  vld3q_u16    LD3 (V.8H x 3)            4cy @ 1p  4cy @ 1p
 *  vld3_u16     LD3 (V.4H x 3)            4cy @ 1p  4cy @ 1p
 *  vbfdotq_f32  BFDOT (V.4S, V.8H, V.8H)  3cy @ 2p  2cy @ 1p
 *  vshll_n_u16  USHLL (V.4S, V.4H, #16)   2cy @ 2p  2cy @ 4p
 *  vfmaq_f32    FMLA (V.4S, V.4S, V.4S)   4cy @ 2p  3cy @ 4p
 *  vaddq_f32    FADD (V.4S, V.4S, V.4S)   2cy @ 2p  2cy @ 4p
 *  vsubq_f32    FSUB (V.4S, V.4S, V.4S)   2cy @ 2p  2cy @ 4p
 *  vmulq_f32    FMUL (V.4S, V.4S, V.4S)   3cy @ 2p  3cy @ 4p
 *  vdupq_n_f32  DUP (V.4S, scalar)        2cy @ 2p  2cy @ 4p
 *  vaddvq_f32   FADDP+FADDP (V.4S)        5cy @ 1p  8cy @ 1p
 *  @endverbatim
 *
 *  The ARMv8.6-BF16 extension enables BF16 storage with F32 computation for 3D mesh alignment
 *  operations. BF16's wider exponent range, matching F32, prevents overflow in geometric
 *  calculations while halving memory bandwidth compared to F32.
 *
 *  For Kabsch and Umeyama point cloud registration, BF16 data is loaded using VLD3 de-interleave
 *  operations and every moment, centroid sums included, goes through BFDOT, @c vbfdotq_f32, which
 *  computes two BF16 products per 32-bit lane with FP32 accumulation. Each point is first shifted
 *  by the first one, so the centering correction doesn't cancel far from the origin. With no BF16
 *  subtraction, the shift widens with @c vshll_n_u16, subtracts in FP32 and rounds back with
 *  BFCVTN, by at most half a ULP of the shifted value, below the quantization of the inputs.
 *  RMSD keeps the widen+subtract+fmaq pipeline, as BFDOT can't express the a − b difference.
 */
#ifndef NUMKONG_MESH_NEONBFDOT_H
#define NUMKONG_MESH_NEONBFDOT_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_TARGET_NEONBFDOT

#include "numkong/types.h"
#include "numkong/cast/neon.h"   // `nk_u16x8_splat_`
#include "numkong/mesh/serial.h" // `nk_det3x3_f32_`, `nk_svd3x3_f32_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.6-a+simd+bf16"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.6-a+simd+bf16")
#endif

/*  Load 4 bf16 xyz points (12 bf16 values) → 3x float32x4_t.
 *  Uses vld3_u16 to de-interleave xyz triplets, then converts bf16 to f32.
 *
 *  Input: 12 contiguous bf16 [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
 *  Output: x[4], y[4], z[4] vectors in f32 */
NUMKONG_INLINE void nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(nk_bf16_t const *ptr, float32x4_t *x_out,
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

NUMKONG_INLINE void nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(nk_bf16_t const *ptr, nk_size_t n_points,
                                                                     float32x4_t *x_out, float32x4_t *y_out,
                                                                     float32x4_t *z_out) {
    nk_u16_t buf[12] = {0};
    nk_u16_t const *src = (nk_u16_t const *)ptr;
    for (nk_size_t k = 0; k < n_points * 3; ++k) buf[k] = src[k];
    nk_deinterleave_bf16x4_to_f32x4_neonbfdot_((nk_bf16_t const *)buf, x_out, y_out, z_out);
}

/** Shifts 8 bf16 values by a pivot in f32 and rounds them back to bf16 for BFDOT. */
NUMKONG_INLINE bfloat16x8_t nk_shift_bf16x8_neonbfdot_(uint16x8_t values_u16x8, float32x4_t pivot_f32x4) {
    float32x4_t low_f32x4 = vsubq_f32(vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(values_u16x8), 16)), pivot_f32x4);
    float32x4_t high_f32x4 = vsubq_f32(vreinterpretq_f32_u32(vshll_high_n_u16(values_u16x8, 16)), pivot_f32x4);
    return vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(low_f32x4), high_f32x4);
}

/** Centroids, centered cross-covariance, ‖a − ā‖² and ‖b − b̄‖² of bf16 clouds,
 *  in one BFDOT pass shifted by the pivots. */
NUMKONG_INLINE void nk_centered_moments_bf16_neonbfdot_(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                        nk_f32_t *centroid_a, nk_f32_t *centroid_b,
                                                        nk_f32_t *cross_covariance, nk_f32_t *centered_norm_squared_a,
                                                        nk_f32_t *centered_norm_squared_b) {
    nk_u16_t const *a_u16 = (nk_u16_t const *)a, *b_u16 = (nk_u16_t const *)b;
    // The wrapper keeps GCC from emitting an FP16 `fmov` of 1.0, which `+bf16` can't assemble.
    bfloat16x8_t const ones_bf16x8 = vreinterpretq_bf16_u16(nk_u16x8_splat_(0x3F80));
    nk_f32_t pivot_a[3], pivot_b[3];
    float32x4_t pivot_a_f32x4[3], pivot_b_f32x4[3], sum_a_f32x4[3], sum_b_f32x4[3], covariance_f32x4[9];
    float32x4_t norm_squared_a_f32x4 = vdupq_n_f32(0), norm_squared_b_f32x4 = vdupq_n_f32(0);
    for (int j = 0; j != 3; ++j)
        nk_bf16_to_f32_(a + j, pivot_a + j), nk_bf16_to_f32_(b + j, pivot_b + j),
            pivot_a_f32x4[j] = vdupq_n_f32(pivot_a[j]), pivot_b_f32x4[j] = vdupq_n_f32(pivot_b[j]),
            sum_a_f32x4[j] = vdupq_n_f32(0), sum_b_f32x4[j] = vdupq_n_f32(0);
    for (int j = 0; j != 9; ++j) covariance_f32x4[j] = vdupq_n_f32(0);
    // Pads the tail with pivots, which shift to zeros and add nothing to the moments.
    nk_u16_t tail_a_u16[24], tail_b_u16[24];
    for (int k = 0; k != 24; ++k) tail_a_u16[k] = a_u16[k % 3], tail_b_u16[k] = b_u16[k % 3];
    for (nk_size_t i = 0; i < n; i += 8) {
        nk_u16_t const *a_chunk_u16 = a_u16 + i * 3, *b_chunk_u16 = b_u16 + i * 3;
        if (i + 8 > n) {
            for (nk_size_t k = 0; k != (n - i) * 3; ++k) tail_a_u16[k] = a_chunk_u16[k], tail_b_u16[k] = b_chunk_u16[k];
            a_chunk_u16 = tail_a_u16, b_chunk_u16 = tail_b_u16;
        }
        uint16x8x3_t a_u16x8x3 = vld3q_u16(a_chunk_u16), b_u16x8x3 = vld3q_u16(b_chunk_u16);
        bfloat16x8_t a_bf16x8[3], b_bf16x8[3];
        for (int j = 0; j != 3; ++j) {
            a_bf16x8[j] = nk_shift_bf16x8_neonbfdot_(a_u16x8x3.val[j], pivot_a_f32x4[j]);
            b_bf16x8[j] = nk_shift_bf16x8_neonbfdot_(b_u16x8x3.val[j], pivot_b_f32x4[j]);
            sum_a_f32x4[j] = vbfdotq_f32(sum_a_f32x4[j], a_bf16x8[j], ones_bf16x8);
            sum_b_f32x4[j] = vbfdotq_f32(sum_b_f32x4[j], b_bf16x8[j], ones_bf16x8);
            norm_squared_a_f32x4 = vbfdotq_f32(norm_squared_a_f32x4, a_bf16x8[j], a_bf16x8[j]);
            norm_squared_b_f32x4 = vbfdotq_f32(norm_squared_b_f32x4, b_bf16x8[j], b_bf16x8[j]);
        }
        for (int j = 0; j != 9; ++j)
            covariance_f32x4[j] = vbfdotq_f32(covariance_f32x4[j], a_bf16x8[j / 3], b_bf16x8[j % 3]);
    }
    nk_f32_t sum_a[3], sum_b[3], covariance[9];
    for (int j = 0; j != 3; ++j) sum_a[j] = vaddvq_f32(sum_a_f32x4[j]), sum_b[j] = vaddvq_f32(sum_b_f32x4[j]);
    for (int j = 0; j != 9; ++j) covariance[j] = vaddvq_f32(covariance_f32x4[j]);
    nk_centered_moments_finalize_f32_(n, pivot_a, pivot_b, sum_a, sum_b, covariance, vaddvq_f32(norm_squared_a_f32x4),
                                      vaddvq_f32(norm_squared_b_f32x4), centroid_a, centroid_b, cross_covariance,
                                      centered_norm_squared_a, centered_norm_squared_b);
}

NUMKONG_API nk_status_t nk_rmsd_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                               nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                               nk_f32_t *scale, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    // RMSD uses identity rotation and scale=1.0
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;

    if (n == 0) {
        *result = 0;
        return nk_success_k;
    }
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

    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32((total_squared_x + total_squared_y + total_squared_z) / (nk_f32_t)n)),
                            0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_kabsch_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                 nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                                 nk_f32_t *scale, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (n == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0f;
        *result = 0;
        return nk_success_k;
    }

    nk_f32_t centroid_a[3], centroid_b[3], cross_covariance[9], centered_norm_squared_a, centered_norm_squared_b;
    nk_centered_moments_bf16_neonbfdot_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_squared_a,
                                        &centered_norm_squared_b);
    if (a_centroid) a_centroid[0] = centroid_a[0], a_centroid[1] = centroid_a[1], a_centroid[2] = centroid_a[2];
    if (b_centroid) b_centroid[0] = centroid_b[0], b_centroid[1] = centroid_b[1], b_centroid[2] = centroid_b[2];

    // Identity-dominant short-circuit: if H ≈ diag(positive), R = I.
    nk_f32_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    if (covariance_offdiagonal_norm_squared < 1e-12f * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
    }
    else {
        nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);

        // R = V * Uᵀ
        optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
        optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
        optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
        optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
        optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
        optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
        optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
        optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
        optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];

        // Handle reflection: if det(R) < 0, negate third column of V and recompute R
        if (nk_det3x3_f32_(optimal_rotation) < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
            optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
            optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
            optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
            optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
            optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
            optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
            optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
            optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];
        }
    }

    // Output rotation matrix and scale=1.0
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0f;

    nk_f32_t sum_squared = nk_folded_ssd_f32_(optimal_rotation, 1.0f, cross_covariance, centered_norm_squared_a,
                                              centered_norm_squared_b);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(sum_squared / (nk_f32_t)n)), 0);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_umeyama_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n,
                                                  nk_f32_t *a_centroid, nk_f32_t *b_centroid, nk_f32_t *rotation,
                                                  nk_f32_t *scale, nk_f32_t *result, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (n == 0) {
        if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
        if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;
        if (rotation)
            rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
            rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
        if (scale) *scale = 1.0f;
        *result = 0;
        return nk_success_k;
    }

    nk_f32_t centroid_a[3], centroid_b[3], cross_covariance[9], centered_norm_squared_a, centered_norm_squared_b;
    nk_centered_moments_bf16_neonbfdot_(a, b, n, centroid_a, centroid_b, cross_covariance, &centered_norm_squared_a,
                                        &centered_norm_squared_b);
    if (a_centroid) a_centroid[0] = centroid_a[0], a_centroid[1] = centroid_a[1], a_centroid[2] = centroid_a[2];
    if (b_centroid) b_centroid[0] = centroid_b[0], b_centroid[1] = centroid_b[1], b_centroid[2] = centroid_b[2];

    // Identity-dominant short-circuit: if H ≈ diag(positive), R = I and trace(R · H) = trace(H).
    nk_f32_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    nk_f32_t c;
    if (covariance_offdiagonal_norm_squared < 1e-12f * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        c = (cross_covariance[0] + cross_covariance[4] + cross_covariance[8]) / centered_norm_squared_a;
    }
    else {
        nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);

        // R = V * Uᵀ
        optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
        optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
        optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
        optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
        optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
        optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
        optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
        optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
        optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];

        // Handle reflection and compute scale: c = trace(D · S) / ‖a-ā‖²
        // D = diag(1, 1, det(R)), svd_diagonal contains proper positive singular values on diagonal
        nk_f32_t rotation_det = nk_det3x3_f32_(optimal_rotation);
        nk_f32_t sign_det = rotation_det < 0 ? -1.0f : 1.0f;
        nk_f32_t trace_scaled_s = svd_diagonal[0] + svd_diagonal[4] + sign_det * svd_diagonal[8];
        c = trace_scaled_s / centered_norm_squared_a;

        if (rotation_det < 0) {
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
            optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
            optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
            optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
            optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
            optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
            optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
            optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
            optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
            optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];
        }
    }
    if (scale) *scale = c;

    // Output rotation matrix
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];

    nk_f32_t sum_squared = nk_folded_ssd_f32_(optimal_rotation, c, cross_covariance, centered_norm_squared_a,
                                              centered_norm_squared_b);
    *result = vget_lane_f32(vsqrt_f32(vdup_n_f32(sum_squared / (nk_f32_t)n)), 0);
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_NEONBFDOT
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_MESH_NEONBFDOT_H
