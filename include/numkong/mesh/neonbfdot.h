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
 *      Intrinsic    Instruction               A76       M5
 *      vld3q_u16    LD3 (V.8H x 3)            4cy @ 1p  4cy @ 1p
 *      vld3_u16     LD3 (V.4H x 3)            4cy @ 1p  4cy @ 1p
 *      vbfdotq_f32  BFDOT (V.4S, V.8H, V.8H)  3cy @ 2p  2cy @ 1p
 *      vshll_n_u16  USHLL (V.4S, V.4H, #16)   2cy @ 2p  2cy @ 4p
 *      vfmaq_f32    FMLA (V.4S, V.4S, V.4S)   4cy @ 2p  3cy @ 4p
 *      vaddq_f32    FADD (V.4S, V.4S, V.4S)   2cy @ 2p  2cy @ 4p
 *      vsubq_f32    FSUB (V.4S, V.4S, V.4S)   2cy @ 2p  2cy @ 4p
 *      vmulq_f32    FMUL (V.4S, V.4S, V.4S)   3cy @ 2p  3cy @ 4p
 *      vdupq_n_f32  DUP (V.4S, scalar)        2cy @ 2p  2cy @ 4p
 *      vaddvq_f32   FADDP+FADDP (V.4S)        5cy @ 1p  8cy @ 1p
 *
 *  The ARMv8.6-BF16 extension enables BF16 storage with F32 computation for 3D mesh alignment
 *  operations. BF16's wider exponent range (matching F32) prevents overflow in geometric calculations
 *  while halving memory bandwidth compared to F32.
 *
 *  For point cloud registration (Kabsch, Umeyama), BF16 data is loaded using VLD3 de-interleave
 *  operations and processed directly with BFDOT (`vbfdotq_f32`), which computes two BF16 products
 *  per 32-bit lane with FP32 accumulation. This skips the explicit bf16→f32 widening that the
 *  prior vshll+fmaq approach required, halving the front-end pressure on the covariance/centroid
 *  stats pass. RMSD keeps the widen+subtract+fmaq pipeline because it needs the (a - b) difference
 *  before squaring, which BFDOT can't express directly.
 */
#ifndef NK_MESH_NEONBFDOT_H
#define NK_MESH_NEONBFDOT_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEONBFDOT

#include "numkong/types.h"
#include "numkong/cast/neon.h"    // `nk_u16x8_splat_`
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`

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
    // bf16 representation of 1.0 is 0x3F80; splatted across 8 lanes for BFDOT-based horizontal sums.
    // The `nk_u16x8_splat_` wrapper prevents GCC from lowering this to `fmov v.8h, #1.0` (a FEAT_FP16
    // encoding) that fails to assemble under a `+bf16`-only pragma.
    bfloat16x8_t const ones_bf16x8 = vreinterpretq_bf16_u16(nk_u16x8_splat_(0x3F80));

    // Centroid numerators, norm-squared, and 3x3 cross-covariance accumulators.
    float32x4_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;
    float32x4_t covariance_xx_f32x4 = zeros_f32x4, covariance_xy_f32x4 = zeros_f32x4, covariance_xz_f32x4 = zeros_f32x4;
    float32x4_t covariance_yx_f32x4 = zeros_f32x4, covariance_yy_f32x4 = zeros_f32x4, covariance_yz_f32x4 = zeros_f32x4;
    float32x4_t covariance_zx_f32x4 = zeros_f32x4, covariance_zy_f32x4 = zeros_f32x4, covariance_zz_f32x4 = zeros_f32x4;
    float32x4_t norm_squared_a_f32x4 = zeros_f32x4, norm_squared_b_f32x4 = zeros_f32x4;

    nk_size_t i = 0;

    // Main loop: 8 triplets per iteration via vld3q_u16 + vbfdotq_f32.
    // Each vld3q_u16 de-interleaves 24 bf16 values into three 8-lane channel vectors.
    // vbfdotq_f32(acc, p, q) computes per-32bit-lane: acc[l] += p[2l]*q[2l] + p[2l+1]*q[2l+1]
    // with bf16 inputs and f32 accumulation. Summing the 4 lanes at the end yields the scalar.
    for (; i + 8 <= n; i += 8) {
        uint16x8x3_t a_xyz_u16x8x3 = vld3q_u16((nk_u16_t const *)(a + i * 3));
        uint16x8x3_t b_xyz_u16x8x3 = vld3q_u16((nk_u16_t const *)(b + i * 3));
        bfloat16x8_t a_x_bf16x8 = vreinterpretq_bf16_u16(a_xyz_u16x8x3.val[0]);
        bfloat16x8_t a_y_bf16x8 = vreinterpretq_bf16_u16(a_xyz_u16x8x3.val[1]);
        bfloat16x8_t a_z_bf16x8 = vreinterpretq_bf16_u16(a_xyz_u16x8x3.val[2]);
        bfloat16x8_t b_x_bf16x8 = vreinterpretq_bf16_u16(b_xyz_u16x8x3.val[0]);
        bfloat16x8_t b_y_bf16x8 = vreinterpretq_bf16_u16(b_xyz_u16x8x3.val[1]);
        bfloat16x8_t b_z_bf16x8 = vreinterpretq_bf16_u16(b_xyz_u16x8x3.val[2]);

        // Centroid numerators: Σ channel values (pairwise via BFDOT against bf16 1.0).
        sum_a_x_f32x4 = vbfdotq_f32(sum_a_x_f32x4, a_x_bf16x8, ones_bf16x8);
        sum_a_y_f32x4 = vbfdotq_f32(sum_a_y_f32x4, a_y_bf16x8, ones_bf16x8);
        sum_a_z_f32x4 = vbfdotq_f32(sum_a_z_f32x4, a_z_bf16x8, ones_bf16x8);
        sum_b_x_f32x4 = vbfdotq_f32(sum_b_x_f32x4, b_x_bf16x8, ones_bf16x8);
        sum_b_y_f32x4 = vbfdotq_f32(sum_b_y_f32x4, b_y_bf16x8, ones_bf16x8);
        sum_b_z_f32x4 = vbfdotq_f32(sum_b_z_f32x4, b_z_bf16x8, ones_bf16x8);

        // 3x3 cross-covariance H cells: Σ a_j · b_k.
        covariance_xx_f32x4 = vbfdotq_f32(covariance_xx_f32x4, a_x_bf16x8, b_x_bf16x8);
        covariance_xy_f32x4 = vbfdotq_f32(covariance_xy_f32x4, a_x_bf16x8, b_y_bf16x8);
        covariance_xz_f32x4 = vbfdotq_f32(covariance_xz_f32x4, a_x_bf16x8, b_z_bf16x8);
        covariance_yx_f32x4 = vbfdotq_f32(covariance_yx_f32x4, a_y_bf16x8, b_x_bf16x8);
        covariance_yy_f32x4 = vbfdotq_f32(covariance_yy_f32x4, a_y_bf16x8, b_y_bf16x8);
        covariance_yz_f32x4 = vbfdotq_f32(covariance_yz_f32x4, a_y_bf16x8, b_z_bf16x8);
        covariance_zx_f32x4 = vbfdotq_f32(covariance_zx_f32x4, a_z_bf16x8, b_x_bf16x8);
        covariance_zy_f32x4 = vbfdotq_f32(covariance_zy_f32x4, a_z_bf16x8, b_y_bf16x8);
        covariance_zz_f32x4 = vbfdotq_f32(covariance_zz_f32x4, a_z_bf16x8, b_z_bf16x8);

        // Norm-squared per point set: Σ (x² + y² + z²).
        norm_squared_a_f32x4 = vbfdotq_f32(norm_squared_a_f32x4, a_x_bf16x8, a_x_bf16x8);
        norm_squared_a_f32x4 = vbfdotq_f32(norm_squared_a_f32x4, a_y_bf16x8, a_y_bf16x8);
        norm_squared_a_f32x4 = vbfdotq_f32(norm_squared_a_f32x4, a_z_bf16x8, a_z_bf16x8);
        norm_squared_b_f32x4 = vbfdotq_f32(norm_squared_b_f32x4, b_x_bf16x8, b_x_bf16x8);
        norm_squared_b_f32x4 = vbfdotq_f32(norm_squared_b_f32x4, b_y_bf16x8, b_y_bf16x8);
        norm_squared_b_f32x4 = vbfdotq_f32(norm_squared_b_f32x4, b_z_bf16x8, b_z_bf16x8);
    }

    // 4-point and partial (1-3) tails: keep the widen+fmaq path on f32x4 channel vectors.
    // These branches run at most once each, so we skip another vbfdotq variant for them.
    float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(a + i * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(b + i * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);
        covariance_xx_f32x4 = vfmaq_f32(covariance_xx_f32x4, a_x_f32x4, b_x_f32x4);
        covariance_xy_f32x4 = vfmaq_f32(covariance_xy_f32x4, a_x_f32x4, b_y_f32x4);
        covariance_xz_f32x4 = vfmaq_f32(covariance_xz_f32x4, a_x_f32x4, b_z_f32x4);
        covariance_yx_f32x4 = vfmaq_f32(covariance_yx_f32x4, a_y_f32x4, b_x_f32x4);
        covariance_yy_f32x4 = vfmaq_f32(covariance_yy_f32x4, a_y_f32x4, b_y_f32x4);
        covariance_yz_f32x4 = vfmaq_f32(covariance_yz_f32x4, a_y_f32x4, b_z_f32x4);
        covariance_zx_f32x4 = vfmaq_f32(covariance_zx_f32x4, a_z_f32x4, b_x_f32x4);
        covariance_zy_f32x4 = vfmaq_f32(covariance_zy_f32x4, a_z_f32x4, b_y_f32x4);
        covariance_zz_f32x4 = vfmaq_f32(covariance_zz_f32x4, a_z_f32x4, b_z_f32x4);
        norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_x_f32x4, a_x_f32x4);
        norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_y_f32x4, a_y_f32x4);
        norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_z_f32x4, a_z_f32x4);
        norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_x_f32x4, b_x_f32x4);
        norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_y_f32x4, b_y_f32x4);
        norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_z_f32x4, b_z_f32x4);
    }
    if (i < n) {
        nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(a + i * 3, n - i, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(b + i * 3, n - i, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);
        covariance_xx_f32x4 = vfmaq_f32(covariance_xx_f32x4, a_x_f32x4, b_x_f32x4);
        covariance_xy_f32x4 = vfmaq_f32(covariance_xy_f32x4, a_x_f32x4, b_y_f32x4);
        covariance_xz_f32x4 = vfmaq_f32(covariance_xz_f32x4, a_x_f32x4, b_z_f32x4);
        covariance_yx_f32x4 = vfmaq_f32(covariance_yx_f32x4, a_y_f32x4, b_x_f32x4);
        covariance_yy_f32x4 = vfmaq_f32(covariance_yy_f32x4, a_y_f32x4, b_y_f32x4);
        covariance_yz_f32x4 = vfmaq_f32(covariance_yz_f32x4, a_y_f32x4, b_z_f32x4);
        covariance_zx_f32x4 = vfmaq_f32(covariance_zx_f32x4, a_z_f32x4, b_x_f32x4);
        covariance_zy_f32x4 = vfmaq_f32(covariance_zy_f32x4, a_z_f32x4, b_y_f32x4);
        covariance_zz_f32x4 = vfmaq_f32(covariance_zz_f32x4, a_z_f32x4, b_z_f32x4);
        norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_x_f32x4, a_x_f32x4);
        norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_y_f32x4, a_y_f32x4);
        norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_z_f32x4, a_z_f32x4);
        norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_x_f32x4, b_x_f32x4);
        norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_y_f32x4, b_y_f32x4);
        norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_z_f32x4, b_z_f32x4);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = vaddvq_f32(sum_a_x_f32x4);
    nk_f32_t sum_a_y = vaddvq_f32(sum_a_y_f32x4);
    nk_f32_t sum_a_z = vaddvq_f32(sum_a_z_f32x4);
    nk_f32_t sum_b_x = vaddvq_f32(sum_b_x_f32x4);
    nk_f32_t sum_b_y = vaddvq_f32(sum_b_y_f32x4);
    nk_f32_t sum_b_z = vaddvq_f32(sum_b_z_f32x4);

    nk_f32_t covariance_x_x = vaddvq_f32(covariance_xx_f32x4);
    nk_f32_t covariance_x_y = vaddvq_f32(covariance_xy_f32x4);
    nk_f32_t covariance_x_z = vaddvq_f32(covariance_xz_f32x4);
    nk_f32_t covariance_y_x = vaddvq_f32(covariance_yx_f32x4);
    nk_f32_t covariance_y_y = vaddvq_f32(covariance_yy_f32x4);
    nk_f32_t covariance_y_z = vaddvq_f32(covariance_yz_f32x4);
    nk_f32_t covariance_z_x = vaddvq_f32(covariance_zx_f32x4);
    nk_f32_t covariance_z_y = vaddvq_f32(covariance_zy_f32x4);
    nk_f32_t covariance_z_z = vaddvq_f32(covariance_zz_f32x4);
    nk_f32_t norm_squared_a = vaddvq_f32(norm_squared_a_f32x4);
    nk_f32_t norm_squared_b = vaddvq_f32(norm_squared_b_f32x4);

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

    // Centered norm-squared via parallel-axis identity; clamp at zero for numeric safety.
    nk_f32_t centered_norm_squared_a = norm_squared_a -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;

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

    // Identity-dominant short-circuit: if H ≈ diag(positive entries), R = I and trace(R·H) = trace(H).
    nk_f32_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    nk_f32_t trace_rotation_covariance;
    if (covariance_offdiagonal_norm_squared < 1e-12f * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
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

        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }

    // Output rotation matrix and scale=1.0
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0f;

    // Folded SSD via trace identity: SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered).
    nk_f32_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0f * trace_rotation_covariance;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
    *result = nk_f32_sqrt_neon(sum_squared * inv_n);
}

NK_PUBLIC void nk_umeyama_bf16_neonbfdot(nk_bf16_t const *a, nk_bf16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                         nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);
    // bf16 representation of 1.0 is 0x3F80; splatted across 8 lanes for BFDOT-based horizontal sums.
    // The `nk_u16x8_splat_` wrapper prevents GCC from lowering this to `fmov v.8h, #1.0` (a FEAT_FP16
    // encoding) that fails to assemble under a `+bf16`-only pragma.
    bfloat16x8_t const ones_bf16x8 = vreinterpretq_bf16_u16(nk_u16x8_splat_(0x3F80));

    // Centroid numerators, norm-squared, and 3x3 cross-covariance accumulators.
    float32x4_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;
    float32x4_t covariance_xx_f32x4 = zeros_f32x4, covariance_xy_f32x4 = zeros_f32x4, covariance_xz_f32x4 = zeros_f32x4;
    float32x4_t covariance_yx_f32x4 = zeros_f32x4, covariance_yy_f32x4 = zeros_f32x4, covariance_yz_f32x4 = zeros_f32x4;
    float32x4_t covariance_zx_f32x4 = zeros_f32x4, covariance_zy_f32x4 = zeros_f32x4, covariance_zz_f32x4 = zeros_f32x4;
    float32x4_t norm_squared_a_f32x4 = zeros_f32x4, norm_squared_b_f32x4 = zeros_f32x4;

    nk_size_t i = 0;

    // Main loop: 8 triplets per iteration via vld3q_u16 + vbfdotq_f32.
    // Each vld3q_u16 de-interleaves 24 bf16 values into three 8-lane channel vectors.
    // vbfdotq_f32(acc, p, q) computes per-32bit-lane: acc[l] += p[2l]*q[2l] + p[2l+1]*q[2l+1]
    // with bf16 inputs and f32 accumulation. Summing the 4 lanes at the end yields the scalar.
    for (; i + 8 <= n; i += 8) {
        uint16x8x3_t a_xyz_u16x8x3 = vld3q_u16((nk_u16_t const *)(a + i * 3));
        uint16x8x3_t b_xyz_u16x8x3 = vld3q_u16((nk_u16_t const *)(b + i * 3));
        bfloat16x8_t a_x_bf16x8 = vreinterpretq_bf16_u16(a_xyz_u16x8x3.val[0]);
        bfloat16x8_t a_y_bf16x8 = vreinterpretq_bf16_u16(a_xyz_u16x8x3.val[1]);
        bfloat16x8_t a_z_bf16x8 = vreinterpretq_bf16_u16(a_xyz_u16x8x3.val[2]);
        bfloat16x8_t b_x_bf16x8 = vreinterpretq_bf16_u16(b_xyz_u16x8x3.val[0]);
        bfloat16x8_t b_y_bf16x8 = vreinterpretq_bf16_u16(b_xyz_u16x8x3.val[1]);
        bfloat16x8_t b_z_bf16x8 = vreinterpretq_bf16_u16(b_xyz_u16x8x3.val[2]);

        // Centroid numerators: Σ channel values (pairwise via BFDOT against bf16 1.0).
        sum_a_x_f32x4 = vbfdotq_f32(sum_a_x_f32x4, a_x_bf16x8, ones_bf16x8);
        sum_a_y_f32x4 = vbfdotq_f32(sum_a_y_f32x4, a_y_bf16x8, ones_bf16x8);
        sum_a_z_f32x4 = vbfdotq_f32(sum_a_z_f32x4, a_z_bf16x8, ones_bf16x8);
        sum_b_x_f32x4 = vbfdotq_f32(sum_b_x_f32x4, b_x_bf16x8, ones_bf16x8);
        sum_b_y_f32x4 = vbfdotq_f32(sum_b_y_f32x4, b_y_bf16x8, ones_bf16x8);
        sum_b_z_f32x4 = vbfdotq_f32(sum_b_z_f32x4, b_z_bf16x8, ones_bf16x8);

        // 3x3 cross-covariance H cells: Σ a_j · b_k.
        covariance_xx_f32x4 = vbfdotq_f32(covariance_xx_f32x4, a_x_bf16x8, b_x_bf16x8);
        covariance_xy_f32x4 = vbfdotq_f32(covariance_xy_f32x4, a_x_bf16x8, b_y_bf16x8);
        covariance_xz_f32x4 = vbfdotq_f32(covariance_xz_f32x4, a_x_bf16x8, b_z_bf16x8);
        covariance_yx_f32x4 = vbfdotq_f32(covariance_yx_f32x4, a_y_bf16x8, b_x_bf16x8);
        covariance_yy_f32x4 = vbfdotq_f32(covariance_yy_f32x4, a_y_bf16x8, b_y_bf16x8);
        covariance_yz_f32x4 = vbfdotq_f32(covariance_yz_f32x4, a_y_bf16x8, b_z_bf16x8);
        covariance_zx_f32x4 = vbfdotq_f32(covariance_zx_f32x4, a_z_bf16x8, b_x_bf16x8);
        covariance_zy_f32x4 = vbfdotq_f32(covariance_zy_f32x4, a_z_bf16x8, b_y_bf16x8);
        covariance_zz_f32x4 = vbfdotq_f32(covariance_zz_f32x4, a_z_bf16x8, b_z_bf16x8);

        // Norm-squared per point set: Σ (x² + y² + z²).
        norm_squared_a_f32x4 = vbfdotq_f32(norm_squared_a_f32x4, a_x_bf16x8, a_x_bf16x8);
        norm_squared_a_f32x4 = vbfdotq_f32(norm_squared_a_f32x4, a_y_bf16x8, a_y_bf16x8);
        norm_squared_a_f32x4 = vbfdotq_f32(norm_squared_a_f32x4, a_z_bf16x8, a_z_bf16x8);
        norm_squared_b_f32x4 = vbfdotq_f32(norm_squared_b_f32x4, b_x_bf16x8, b_x_bf16x8);
        norm_squared_b_f32x4 = vbfdotq_f32(norm_squared_b_f32x4, b_y_bf16x8, b_y_bf16x8);
        norm_squared_b_f32x4 = vbfdotq_f32(norm_squared_b_f32x4, b_z_bf16x8, b_z_bf16x8);
    }

    // 4-point and partial (1-3) tails: keep the widen+fmaq path on f32x4 channel vectors.
    // These branches run at most once each, so we skip another vbfdotq variant for them.
    float32x4_t a_x_f32x4, a_y_f32x4, a_z_f32x4, b_x_f32x4, b_y_f32x4, b_z_f32x4;
    for (; i + 4 <= n; i += 4) {
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(a + i * 3, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_deinterleave_bf16x4_to_f32x4_neonbfdot_(b + i * 3, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);
        covariance_xx_f32x4 = vfmaq_f32(covariance_xx_f32x4, a_x_f32x4, b_x_f32x4);
        covariance_xy_f32x4 = vfmaq_f32(covariance_xy_f32x4, a_x_f32x4, b_y_f32x4);
        covariance_xz_f32x4 = vfmaq_f32(covariance_xz_f32x4, a_x_f32x4, b_z_f32x4);
        covariance_yx_f32x4 = vfmaq_f32(covariance_yx_f32x4, a_y_f32x4, b_x_f32x4);
        covariance_yy_f32x4 = vfmaq_f32(covariance_yy_f32x4, a_y_f32x4, b_y_f32x4);
        covariance_yz_f32x4 = vfmaq_f32(covariance_yz_f32x4, a_y_f32x4, b_z_f32x4);
        covariance_zx_f32x4 = vfmaq_f32(covariance_zx_f32x4, a_z_f32x4, b_x_f32x4);
        covariance_zy_f32x4 = vfmaq_f32(covariance_zy_f32x4, a_z_f32x4, b_y_f32x4);
        covariance_zz_f32x4 = vfmaq_f32(covariance_zz_f32x4, a_z_f32x4, b_z_f32x4);
        norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_x_f32x4, a_x_f32x4);
        norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_y_f32x4, a_y_f32x4);
        norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_z_f32x4, a_z_f32x4);
        norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_x_f32x4, b_x_f32x4);
        norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_y_f32x4, b_y_f32x4);
        norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_z_f32x4, b_z_f32x4);
    }
    if (i < n) {
        nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(a + i * 3, n - i, &a_x_f32x4, &a_y_f32x4, &a_z_f32x4);
        nk_partial_deinterleave_bf16_to_f32x4_neonbfdot_(b + i * 3, n - i, &b_x_f32x4, &b_y_f32x4, &b_z_f32x4);
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, a_x_f32x4);
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, a_y_f32x4);
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, a_z_f32x4);
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, b_x_f32x4);
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, b_y_f32x4);
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, b_z_f32x4);
        covariance_xx_f32x4 = vfmaq_f32(covariance_xx_f32x4, a_x_f32x4, b_x_f32x4);
        covariance_xy_f32x4 = vfmaq_f32(covariance_xy_f32x4, a_x_f32x4, b_y_f32x4);
        covariance_xz_f32x4 = vfmaq_f32(covariance_xz_f32x4, a_x_f32x4, b_z_f32x4);
        covariance_yx_f32x4 = vfmaq_f32(covariance_yx_f32x4, a_y_f32x4, b_x_f32x4);
        covariance_yy_f32x4 = vfmaq_f32(covariance_yy_f32x4, a_y_f32x4, b_y_f32x4);
        covariance_yz_f32x4 = vfmaq_f32(covariance_yz_f32x4, a_y_f32x4, b_z_f32x4);
        covariance_zx_f32x4 = vfmaq_f32(covariance_zx_f32x4, a_z_f32x4, b_x_f32x4);
        covariance_zy_f32x4 = vfmaq_f32(covariance_zy_f32x4, a_z_f32x4, b_y_f32x4);
        covariance_zz_f32x4 = vfmaq_f32(covariance_zz_f32x4, a_z_f32x4, b_z_f32x4);
        norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_x_f32x4, a_x_f32x4);
        norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_y_f32x4, a_y_f32x4);
        norm_squared_a_f32x4 = vfmaq_f32(norm_squared_a_f32x4, a_z_f32x4, a_z_f32x4);
        norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_x_f32x4, b_x_f32x4);
        norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_y_f32x4, b_y_f32x4);
        norm_squared_b_f32x4 = vfmaq_f32(norm_squared_b_f32x4, b_z_f32x4, b_z_f32x4);
    }

    // Reduce vector accumulators
    nk_f32_t sum_a_x = vaddvq_f32(sum_a_x_f32x4);
    nk_f32_t sum_a_y = vaddvq_f32(sum_a_y_f32x4);
    nk_f32_t sum_a_z = vaddvq_f32(sum_a_z_f32x4);
    nk_f32_t sum_b_x = vaddvq_f32(sum_b_x_f32x4);
    nk_f32_t sum_b_y = vaddvq_f32(sum_b_y_f32x4);
    nk_f32_t sum_b_z = vaddvq_f32(sum_b_z_f32x4);

    nk_f32_t covariance_x_x = vaddvq_f32(covariance_xx_f32x4);
    nk_f32_t covariance_x_y = vaddvq_f32(covariance_xy_f32x4);
    nk_f32_t covariance_x_z = vaddvq_f32(covariance_xz_f32x4);
    nk_f32_t covariance_y_x = vaddvq_f32(covariance_yx_f32x4);
    nk_f32_t covariance_y_y = vaddvq_f32(covariance_yy_f32x4);
    nk_f32_t covariance_y_z = vaddvq_f32(covariance_yz_f32x4);
    nk_f32_t covariance_z_x = vaddvq_f32(covariance_zx_f32x4);
    nk_f32_t covariance_z_y = vaddvq_f32(covariance_zy_f32x4);
    nk_f32_t covariance_z_z = vaddvq_f32(covariance_zz_f32x4);
    nk_f32_t norm_squared_a = vaddvq_f32(norm_squared_a_f32x4);
    nk_f32_t norm_squared_b = vaddvq_f32(norm_squared_b_f32x4);

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

    // Centered norm-squared via parallel-axis identity; clamp at zero for numeric safety.
    nk_f32_t centered_norm_squared_a = norm_squared_a -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;

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

    // Identity-dominant short-circuit: if H ≈ diag(positive entries), R = I and trace(R·H) = trace(H).
    nk_f32_t covariance_diagonal_norm_squared = cross_covariance[0] * cross_covariance[0] +
                                                cross_covariance[4] * cross_covariance[4] +
                                                cross_covariance[8] * cross_covariance[8];
    nk_f32_t covariance_offdiagonal_norm_squared =
        cross_covariance[1] * cross_covariance[1] + cross_covariance[2] * cross_covariance[2] +
        cross_covariance[3] * cross_covariance[3] + cross_covariance[5] * cross_covariance[5] +
        cross_covariance[6] * cross_covariance[6] + cross_covariance[7] * cross_covariance[7];
    nk_f32_t optimal_rotation[9];
    nk_f32_t trace_rotation_covariance;
    nk_f32_t c;
    if (covariance_offdiagonal_norm_squared < 1e-12f * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
        c = centered_norm_squared_a > 0.0f ? trace_rotation_covariance / centered_norm_squared_a : 0.0f;
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
        c = centered_norm_squared_a > 0.0f ? trace_scaled_s / centered_norm_squared_a : 0.0f;

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

        trace_rotation_covariance =
            optimal_rotation[0] * cross_covariance[0] + optimal_rotation[1] * cross_covariance[3] +
            optimal_rotation[2] * cross_covariance[6] + optimal_rotation[3] * cross_covariance[1] +
            optimal_rotation[4] * cross_covariance[4] + optimal_rotation[5] * cross_covariance[7] +
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }
    if (scale) *scale = c;

    // Output rotation matrix
    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];

    // Folded SSD with scale: c²·‖a-ā‖² + ‖b-b̄‖² − 2c·trace(R · H_centered).
    nk_f32_t sum_squared = c * c * centered_norm_squared_a + centered_norm_squared_b -
                           2.0f * c * trace_rotation_covariance;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
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
