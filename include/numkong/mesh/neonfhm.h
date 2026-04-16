/**
 *  @brief SIMD-accelerated Point Cloud Alignment for NEON FP16 FHM (widening FMA).
 *  @file include/numkong/mesh/neonfhm.h
 *  @author Ash Vardanian
 *  @date April 15, 2026
 *
 *  @sa include/numkong/mesh.h
 *
 *  @section mesh_neonfhm_instructions ARM NEON FP16 Matrix Instructions (ARMv8.4-FHM)
 *
 *      Intrinsic         Instruction                A76       M5
 *      vld3q_u16         LD3 (V.8H x 3)             6cy @ 1p  6cy @ 1p
 *      vfmlalq_low_f16   FMLAL (V.4S, V.8H, V.8H)   4cy @ 2p  4cy @ 4p
 *      vfmlalq_high_f16  FMLAL2 (V.4S, V.8H, V.8H)  4cy @ 2p  4cy @ 4p
 *      vcvt_f32_f16      FCVTL (V.4S, V.4H)         4cy @ 2p  3cy @ 4p
 *      vcvt_high_f32_f16 FCVTL2 (V.4S, V.8H)        4cy @ 2p  3cy @ 4p
 *      vfmaq_f32         FMLA (V.4S, V.4S, V.4S)    4cy @ 2p  3cy @ 4p
 *      vaddq_f32         FADD (V.4S, V.4S, V.4S)    2cy @ 2p  2cy @ 4p
 *      vaddvq_f32        FADDP+FADDP (V.4S)         5cy @ 1p  8cy @ 1p
 *
 *  The ARMv8.4-FHM extension (FEAT_FHM) provides FMLAL/FMLSL instructions that fuse FP16 to FP32
 *  widening with multiply-accumulate into a single operation. `vfmlalq_low_f16` operates on elements
 *  0-3 of the FP16 inputs; `vfmlalq_high_f16` operates on elements 4-7 — together they process a
 *  full `float16x8_t` of data into two `float32x4_t` accumulators with full FP32 accumulator precision.
 *
 *  For 3D mesh alignment (RMSD, Kabsch, Umeyama), this replaces the two-step FP16→FP32 widen
 *  (`vcvt_f32_f16` + `vcvt_high_f32_f16`) followed by FP32 FMA (`vfmaq_f32`) in the covariance and
 *  norm-squared accumulation, fusing widen + multiply-accumulate. The low/high halves are kept as
 *  separate `float32x4_t` accumulators and combined only at reduction time. Sums of raw coordinates
 *  (for centroids) still use a conventional widen-then-add path since there is no widening-add
 *  intrinsic for FP16 inputs.
 */
#ifndef NK_MESH_NEONFHM_H
#define NK_MESH_NEONFHM_H

#if NK_TARGET_ARM64_
#if NK_TARGET_NEONFHM

#include "numkong/types.h"
#include "numkong/spatial/neon.h" // `nk_f32_sqrt_neon`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd+fp16+fp16fml"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd+fp16+fp16fml")
#endif

/*  Load 8 fp16 xyz triplets (24 fp16 values) → 3x float16x8_t.
 *  Uses vld3q_u16 to de-interleave, then reinterprets as f16 (avoids vld3q_f16 which is
 *  unavailable on MSVC for ARM64).
 *
 *  Input: 24 contiguous fp16 [x0,y0,z0, ..., x7,y7,z7]
 *  Output: x_f16x8, y_f16x8, z_f16x8 channel vectors (8 lanes each)
 */
NK_INTERNAL void nk_deinterleave_f16x8_to_f16x8x3_neonfhm_(nk_f16_t const *ptr, //
                                                           float16x8_t *x_out, float16x8_t *y_out, float16x8_t *z_out) {
    uint16x8x3_t xyz_u16x8x3 = vld3q_u16((nk_u16_t const *)ptr);
    *x_out = vreinterpretq_f16_u16(xyz_u16x8x3.val[0]);
    *y_out = vreinterpretq_f16_u16(xyz_u16x8x3.val[1]);
    *z_out = vreinterpretq_f16_u16(xyz_u16x8x3.val[2]);
}

NK_INTERNAL void nk_partial_deinterleave_f16_to_f16x8x3_neonfhm_(nk_f16_t const *ptr, nk_size_t n_points, //
                                                                 float16x8_t *x_out, float16x8_t *y_out,
                                                                 float16x8_t *z_out) {
    nk_u16_t buf[24] = {0};
    nk_u16_t const *src = (nk_u16_t const *)ptr;
    for (nk_size_t k = 0; k < n_points * 3; ++k) buf[k] = src[k];
    nk_deinterleave_f16x8_to_f16x8x3_neonfhm_((nk_f16_t const *)buf, x_out, y_out, z_out);
}

/**
 *  @brief RMSD (Root Mean Square Deviation) using NEON FHM widening FMA.
 *  Matches the serial-RMSD contract: zero centroids, identity rotation, raw √(Σ‖a-b‖² / n).
 */
NK_PUBLIC void nk_rmsd_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                   nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    if (rotation)
        rotation[0] = 1, rotation[1] = 0, rotation[2] = 0, rotation[3] = 0, rotation[4] = 1, rotation[5] = 0,
        rotation[6] = 0, rotation[7] = 0, rotation[8] = 1;
    if (scale) *scale = 1.0f;
    if (a_centroid) a_centroid[0] = 0, a_centroid[1] = 0, a_centroid[2] = 0;
    if (b_centroid) b_centroid[0] = 0, b_centroid[1] = 0, b_centroid[2] = 0;

    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);
    // Squared-delta accumulators split into low (elements 0-3) and high (4-7) halves for FHM
    float32x4_t sum_squared_x_low_f32x4 = zeros_f32x4, sum_squared_x_high_f32x4 = zeros_f32x4;
    float32x4_t sum_squared_y_low_f32x4 = zeros_f32x4, sum_squared_y_high_f32x4 = zeros_f32x4;
    float32x4_t sum_squared_z_low_f32x4 = zeros_f32x4, sum_squared_z_high_f32x4 = zeros_f32x4;

    float16x8_t a_x_f16x8, a_y_f16x8, a_z_f16x8;
    float16x8_t b_x_f16x8, b_y_f16x8, b_z_f16x8;
    nk_size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f16x8x3_neonfhm_(a + i * 3, &a_x_f16x8, &a_y_f16x8, &a_z_f16x8);
        nk_deinterleave_f16x8_to_f16x8x3_neonfhm_(b + i * 3, &b_x_f16x8, &b_y_f16x8, &b_z_f16x8);

        float16x8_t delta_x_f16x8 = vsubq_f16(a_x_f16x8, b_x_f16x8);
        float16x8_t delta_y_f16x8 = vsubq_f16(a_y_f16x8, b_y_f16x8);
        float16x8_t delta_z_f16x8 = vsubq_f16(a_z_f16x8, b_z_f16x8);

        sum_squared_x_low_f32x4 = vfmlalq_low_f16(sum_squared_x_low_f32x4, delta_x_f16x8, delta_x_f16x8);
        sum_squared_x_high_f32x4 = vfmlalq_high_f16(sum_squared_x_high_f32x4, delta_x_f16x8, delta_x_f16x8);
        sum_squared_y_low_f32x4 = vfmlalq_low_f16(sum_squared_y_low_f32x4, delta_y_f16x8, delta_y_f16x8);
        sum_squared_y_high_f32x4 = vfmlalq_high_f16(sum_squared_y_high_f32x4, delta_y_f16x8, delta_y_f16x8);
        sum_squared_z_low_f32x4 = vfmlalq_low_f16(sum_squared_z_low_f32x4, delta_z_f16x8, delta_z_f16x8);
        sum_squared_z_high_f32x4 = vfmlalq_high_f16(sum_squared_z_high_f32x4, delta_z_f16x8, delta_z_f16x8);
    }

    if (i < n) {
        nk_partial_deinterleave_f16_to_f16x8x3_neonfhm_(a + i * 3, n - i, &a_x_f16x8, &a_y_f16x8, &a_z_f16x8);
        nk_partial_deinterleave_f16_to_f16x8x3_neonfhm_(b + i * 3, n - i, &b_x_f16x8, &b_y_f16x8, &b_z_f16x8);

        float16x8_t delta_x_f16x8 = vsubq_f16(a_x_f16x8, b_x_f16x8);
        float16x8_t delta_y_f16x8 = vsubq_f16(a_y_f16x8, b_y_f16x8);
        float16x8_t delta_z_f16x8 = vsubq_f16(a_z_f16x8, b_z_f16x8);

        sum_squared_x_low_f32x4 = vfmlalq_low_f16(sum_squared_x_low_f32x4, delta_x_f16x8, delta_x_f16x8);
        sum_squared_x_high_f32x4 = vfmlalq_high_f16(sum_squared_x_high_f32x4, delta_x_f16x8, delta_x_f16x8);
        sum_squared_y_low_f32x4 = vfmlalq_low_f16(sum_squared_y_low_f32x4, delta_y_f16x8, delta_y_f16x8);
        sum_squared_y_high_f32x4 = vfmlalq_high_f16(sum_squared_y_high_f32x4, delta_y_f16x8, delta_y_f16x8);
        sum_squared_z_low_f32x4 = vfmlalq_low_f16(sum_squared_z_low_f32x4, delta_z_f16x8, delta_z_f16x8);
        sum_squared_z_high_f32x4 = vfmlalq_high_f16(sum_squared_z_high_f32x4, delta_z_f16x8, delta_z_f16x8);
    }

    nk_f32_t sum_squared = vaddvq_f32(vaddq_f32(sum_squared_x_low_f32x4, sum_squared_x_high_f32x4)) +
                           vaddvq_f32(vaddq_f32(sum_squared_y_low_f32x4, sum_squared_y_high_f32x4)) +
                           vaddvq_f32(vaddq_f32(sum_squared_z_low_f32x4, sum_squared_z_high_f32x4));
    *result = nk_f32_sqrt_neon(sum_squared / (nk_f32_t)n);
}

/**
 *  @brief Kabsch algorithm for optimal rigid body superposition using NEON FHM widening FMA.
 *  Finds the rotation matrix R that minimizes RMSD between two point sets.
 */
NK_PUBLIC void nk_kabsch_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                     nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    // Centroid sums (widen-add path)
    float32x4_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;

    // Covariance matrix H: 9 cells, each split into low/high f32x4 FHM accumulators.
    float32x4_t covariance_xx_low_f32x4 = zeros_f32x4, covariance_xx_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_xy_low_f32x4 = zeros_f32x4, covariance_xy_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_xz_low_f32x4 = zeros_f32x4, covariance_xz_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_yx_low_f32x4 = zeros_f32x4, covariance_yx_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_yy_low_f32x4 = zeros_f32x4, covariance_yy_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_yz_low_f32x4 = zeros_f32x4, covariance_yz_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_zx_low_f32x4 = zeros_f32x4, covariance_zx_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_zy_low_f32x4 = zeros_f32x4, covariance_zy_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_zz_low_f32x4 = zeros_f32x4, covariance_zz_high_f32x4 = zeros_f32x4;
    float32x4_t norm_squared_a_low_f32x4 = zeros_f32x4, norm_squared_a_high_f32x4 = zeros_f32x4;
    float32x4_t norm_squared_b_low_f32x4 = zeros_f32x4, norm_squared_b_high_f32x4 = zeros_f32x4;

    float16x8_t a_x_f16x8, a_y_f16x8, a_z_f16x8;
    float16x8_t b_x_f16x8, b_y_f16x8, b_z_f16x8;
    nk_size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f16x8x3_neonfhm_(a + i * 3, &a_x_f16x8, &a_y_f16x8, &a_z_f16x8);
        nk_deinterleave_f16x8_to_f16x8x3_neonfhm_(b + i * 3, &b_x_f16x8, &b_y_f16x8, &b_z_f16x8);

        // Centroid sums via widen-then-add
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, vcvt_f32_f16(vget_low_f16(a_x_f16x8)));
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, vcvt_high_f32_f16(a_x_f16x8));
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, vcvt_f32_f16(vget_low_f16(a_y_f16x8)));
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, vcvt_high_f32_f16(a_y_f16x8));
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, vcvt_f32_f16(vget_low_f16(a_z_f16x8)));
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, vcvt_high_f32_f16(a_z_f16x8));
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, vcvt_f32_f16(vget_low_f16(b_x_f16x8)));
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, vcvt_high_f32_f16(b_x_f16x8));
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, vcvt_f32_f16(vget_low_f16(b_y_f16x8)));
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, vcvt_high_f32_f16(b_y_f16x8));
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, vcvt_f32_f16(vget_low_f16(b_z_f16x8)));
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, vcvt_high_f32_f16(b_z_f16x8));

        // Covariance H = sum a * bᵀ via FHM widening FMA (9 cells × 2 halves)
        covariance_xx_low_f32x4 = vfmlalq_low_f16(covariance_xx_low_f32x4, a_x_f16x8, b_x_f16x8);
        covariance_xx_high_f32x4 = vfmlalq_high_f16(covariance_xx_high_f32x4, a_x_f16x8, b_x_f16x8);
        covariance_xy_low_f32x4 = vfmlalq_low_f16(covariance_xy_low_f32x4, a_x_f16x8, b_y_f16x8);
        covariance_xy_high_f32x4 = vfmlalq_high_f16(covariance_xy_high_f32x4, a_x_f16x8, b_y_f16x8);
        covariance_xz_low_f32x4 = vfmlalq_low_f16(covariance_xz_low_f32x4, a_x_f16x8, b_z_f16x8);
        covariance_xz_high_f32x4 = vfmlalq_high_f16(covariance_xz_high_f32x4, a_x_f16x8, b_z_f16x8);
        covariance_yx_low_f32x4 = vfmlalq_low_f16(covariance_yx_low_f32x4, a_y_f16x8, b_x_f16x8);
        covariance_yx_high_f32x4 = vfmlalq_high_f16(covariance_yx_high_f32x4, a_y_f16x8, b_x_f16x8);
        covariance_yy_low_f32x4 = vfmlalq_low_f16(covariance_yy_low_f32x4, a_y_f16x8, b_y_f16x8);
        covariance_yy_high_f32x4 = vfmlalq_high_f16(covariance_yy_high_f32x4, a_y_f16x8, b_y_f16x8);
        covariance_yz_low_f32x4 = vfmlalq_low_f16(covariance_yz_low_f32x4, a_y_f16x8, b_z_f16x8);
        covariance_yz_high_f32x4 = vfmlalq_high_f16(covariance_yz_high_f32x4, a_y_f16x8, b_z_f16x8);
        covariance_zx_low_f32x4 = vfmlalq_low_f16(covariance_zx_low_f32x4, a_z_f16x8, b_x_f16x8);
        covariance_zx_high_f32x4 = vfmlalq_high_f16(covariance_zx_high_f32x4, a_z_f16x8, b_x_f16x8);
        covariance_zy_low_f32x4 = vfmlalq_low_f16(covariance_zy_low_f32x4, a_z_f16x8, b_y_f16x8);
        covariance_zy_high_f32x4 = vfmlalq_high_f16(covariance_zy_high_f32x4, a_z_f16x8, b_y_f16x8);
        covariance_zz_low_f32x4 = vfmlalq_low_f16(covariance_zz_low_f32x4, a_z_f16x8, b_z_f16x8);
        covariance_zz_high_f32x4 = vfmlalq_high_f16(covariance_zz_high_f32x4, a_z_f16x8, b_z_f16x8);

        // Norm-squared of both point sets via FHM
        norm_squared_a_low_f32x4 = vfmlalq_low_f16(norm_squared_a_low_f32x4, a_x_f16x8, a_x_f16x8);
        norm_squared_a_high_f32x4 = vfmlalq_high_f16(norm_squared_a_high_f32x4, a_x_f16x8, a_x_f16x8);
        norm_squared_a_low_f32x4 = vfmlalq_low_f16(norm_squared_a_low_f32x4, a_y_f16x8, a_y_f16x8);
        norm_squared_a_high_f32x4 = vfmlalq_high_f16(norm_squared_a_high_f32x4, a_y_f16x8, a_y_f16x8);
        norm_squared_a_low_f32x4 = vfmlalq_low_f16(norm_squared_a_low_f32x4, a_z_f16x8, a_z_f16x8);
        norm_squared_a_high_f32x4 = vfmlalq_high_f16(norm_squared_a_high_f32x4, a_z_f16x8, a_z_f16x8);
        norm_squared_b_low_f32x4 = vfmlalq_low_f16(norm_squared_b_low_f32x4, b_x_f16x8, b_x_f16x8);
        norm_squared_b_high_f32x4 = vfmlalq_high_f16(norm_squared_b_high_f32x4, b_x_f16x8, b_x_f16x8);
        norm_squared_b_low_f32x4 = vfmlalq_low_f16(norm_squared_b_low_f32x4, b_y_f16x8, b_y_f16x8);
        norm_squared_b_high_f32x4 = vfmlalq_high_f16(norm_squared_b_high_f32x4, b_y_f16x8, b_y_f16x8);
        norm_squared_b_low_f32x4 = vfmlalq_low_f16(norm_squared_b_low_f32x4, b_z_f16x8, b_z_f16x8);
        norm_squared_b_high_f32x4 = vfmlalq_high_f16(norm_squared_b_high_f32x4, b_z_f16x8, b_z_f16x8);
    }

    if (i < n) {
        nk_partial_deinterleave_f16_to_f16x8x3_neonfhm_(a + i * 3, n - i, &a_x_f16x8, &a_y_f16x8, &a_z_f16x8);
        nk_partial_deinterleave_f16_to_f16x8x3_neonfhm_(b + i * 3, n - i, &b_x_f16x8, &b_y_f16x8, &b_z_f16x8);

        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, vcvt_f32_f16(vget_low_f16(a_x_f16x8)));
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, vcvt_high_f32_f16(a_x_f16x8));
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, vcvt_f32_f16(vget_low_f16(a_y_f16x8)));
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, vcvt_high_f32_f16(a_y_f16x8));
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, vcvt_f32_f16(vget_low_f16(a_z_f16x8)));
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, vcvt_high_f32_f16(a_z_f16x8));
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, vcvt_f32_f16(vget_low_f16(b_x_f16x8)));
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, vcvt_high_f32_f16(b_x_f16x8));
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, vcvt_f32_f16(vget_low_f16(b_y_f16x8)));
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, vcvt_high_f32_f16(b_y_f16x8));
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, vcvt_f32_f16(vget_low_f16(b_z_f16x8)));
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, vcvt_high_f32_f16(b_z_f16x8));

        covariance_xx_low_f32x4 = vfmlalq_low_f16(covariance_xx_low_f32x4, a_x_f16x8, b_x_f16x8);
        covariance_xx_high_f32x4 = vfmlalq_high_f16(covariance_xx_high_f32x4, a_x_f16x8, b_x_f16x8);
        covariance_xy_low_f32x4 = vfmlalq_low_f16(covariance_xy_low_f32x4, a_x_f16x8, b_y_f16x8);
        covariance_xy_high_f32x4 = vfmlalq_high_f16(covariance_xy_high_f32x4, a_x_f16x8, b_y_f16x8);
        covariance_xz_low_f32x4 = vfmlalq_low_f16(covariance_xz_low_f32x4, a_x_f16x8, b_z_f16x8);
        covariance_xz_high_f32x4 = vfmlalq_high_f16(covariance_xz_high_f32x4, a_x_f16x8, b_z_f16x8);
        covariance_yx_low_f32x4 = vfmlalq_low_f16(covariance_yx_low_f32x4, a_y_f16x8, b_x_f16x8);
        covariance_yx_high_f32x4 = vfmlalq_high_f16(covariance_yx_high_f32x4, a_y_f16x8, b_x_f16x8);
        covariance_yy_low_f32x4 = vfmlalq_low_f16(covariance_yy_low_f32x4, a_y_f16x8, b_y_f16x8);
        covariance_yy_high_f32x4 = vfmlalq_high_f16(covariance_yy_high_f32x4, a_y_f16x8, b_y_f16x8);
        covariance_yz_low_f32x4 = vfmlalq_low_f16(covariance_yz_low_f32x4, a_y_f16x8, b_z_f16x8);
        covariance_yz_high_f32x4 = vfmlalq_high_f16(covariance_yz_high_f32x4, a_y_f16x8, b_z_f16x8);
        covariance_zx_low_f32x4 = vfmlalq_low_f16(covariance_zx_low_f32x4, a_z_f16x8, b_x_f16x8);
        covariance_zx_high_f32x4 = vfmlalq_high_f16(covariance_zx_high_f32x4, a_z_f16x8, b_x_f16x8);
        covariance_zy_low_f32x4 = vfmlalq_low_f16(covariance_zy_low_f32x4, a_z_f16x8, b_y_f16x8);
        covariance_zy_high_f32x4 = vfmlalq_high_f16(covariance_zy_high_f32x4, a_z_f16x8, b_y_f16x8);
        covariance_zz_low_f32x4 = vfmlalq_low_f16(covariance_zz_low_f32x4, a_z_f16x8, b_z_f16x8);
        covariance_zz_high_f32x4 = vfmlalq_high_f16(covariance_zz_high_f32x4, a_z_f16x8, b_z_f16x8);

        norm_squared_a_low_f32x4 = vfmlalq_low_f16(norm_squared_a_low_f32x4, a_x_f16x8, a_x_f16x8);
        norm_squared_a_high_f32x4 = vfmlalq_high_f16(norm_squared_a_high_f32x4, a_x_f16x8, a_x_f16x8);
        norm_squared_a_low_f32x4 = vfmlalq_low_f16(norm_squared_a_low_f32x4, a_y_f16x8, a_y_f16x8);
        norm_squared_a_high_f32x4 = vfmlalq_high_f16(norm_squared_a_high_f32x4, a_y_f16x8, a_y_f16x8);
        norm_squared_a_low_f32x4 = vfmlalq_low_f16(norm_squared_a_low_f32x4, a_z_f16x8, a_z_f16x8);
        norm_squared_a_high_f32x4 = vfmlalq_high_f16(norm_squared_a_high_f32x4, a_z_f16x8, a_z_f16x8);
        norm_squared_b_low_f32x4 = vfmlalq_low_f16(norm_squared_b_low_f32x4, b_x_f16x8, b_x_f16x8);
        norm_squared_b_high_f32x4 = vfmlalq_high_f16(norm_squared_b_high_f32x4, b_x_f16x8, b_x_f16x8);
        norm_squared_b_low_f32x4 = vfmlalq_low_f16(norm_squared_b_low_f32x4, b_y_f16x8, b_y_f16x8);
        norm_squared_b_high_f32x4 = vfmlalq_high_f16(norm_squared_b_high_f32x4, b_y_f16x8, b_y_f16x8);
        norm_squared_b_low_f32x4 = vfmlalq_low_f16(norm_squared_b_low_f32x4, b_z_f16x8, b_z_f16x8);
        norm_squared_b_high_f32x4 = vfmlalq_high_f16(norm_squared_b_high_f32x4, b_z_f16x8, b_z_f16x8);
    }

    // Combine low+high halves
    float32x4_t covariance_xx_f32x4 = vaddq_f32(covariance_xx_low_f32x4, covariance_xx_high_f32x4);
    float32x4_t covariance_xy_f32x4 = vaddq_f32(covariance_xy_low_f32x4, covariance_xy_high_f32x4);
    float32x4_t covariance_xz_f32x4 = vaddq_f32(covariance_xz_low_f32x4, covariance_xz_high_f32x4);
    float32x4_t covariance_yx_f32x4 = vaddq_f32(covariance_yx_low_f32x4, covariance_yx_high_f32x4);
    float32x4_t covariance_yy_f32x4 = vaddq_f32(covariance_yy_low_f32x4, covariance_yy_high_f32x4);
    float32x4_t covariance_yz_f32x4 = vaddq_f32(covariance_yz_low_f32x4, covariance_yz_high_f32x4);
    float32x4_t covariance_zx_f32x4 = vaddq_f32(covariance_zx_low_f32x4, covariance_zx_high_f32x4);
    float32x4_t covariance_zy_f32x4 = vaddq_f32(covariance_zy_low_f32x4, covariance_zy_high_f32x4);
    float32x4_t covariance_zz_f32x4 = vaddq_f32(covariance_zz_low_f32x4, covariance_zz_high_f32x4);
    float32x4_t norm_squared_a_f32x4 = vaddq_f32(norm_squared_a_low_f32x4, norm_squared_a_high_f32x4);
    float32x4_t norm_squared_b_f32x4 = vaddq_f32(norm_squared_b_low_f32x4, norm_squared_b_high_f32x4);

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

    // Apply centering: H_centered = H − n · centroid_a · centroid_bᵀ
    nk_f32_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - (nk_f32_t)n * centroid_a_x * centroid_b_x;
    cross_covariance[1] = covariance_x_y - (nk_f32_t)n * centroid_a_x * centroid_b_y;
    cross_covariance[2] = covariance_x_z - (nk_f32_t)n * centroid_a_x * centroid_b_z;
    cross_covariance[3] = covariance_y_x - (nk_f32_t)n * centroid_a_y * centroid_b_x;
    cross_covariance[4] = covariance_y_y - (nk_f32_t)n * centroid_a_y * centroid_b_y;
    cross_covariance[5] = covariance_y_z - (nk_f32_t)n * centroid_a_y * centroid_b_z;
    cross_covariance[6] = covariance_z_x - (nk_f32_t)n * centroid_a_z * centroid_b_x;
    cross_covariance[7] = covariance_z_y - (nk_f32_t)n * centroid_a_z * centroid_b_y;
    cross_covariance[8] = covariance_z_z - (nk_f32_t)n * centroid_a_z * centroid_b_z;

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
        // SVD of H = U · S · Vᵀ
        nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);

        // R = V · Uᵀ
        optimal_rotation[0] = svd_right[0] * svd_left[0] + svd_right[1] * svd_left[1] + svd_right[2] * svd_left[2];
        optimal_rotation[1] = svd_right[0] * svd_left[3] + svd_right[1] * svd_left[4] + svd_right[2] * svd_left[5];
        optimal_rotation[2] = svd_right[0] * svd_left[6] + svd_right[1] * svd_left[7] + svd_right[2] * svd_left[8];
        optimal_rotation[3] = svd_right[3] * svd_left[0] + svd_right[4] * svd_left[1] + svd_right[5] * svd_left[2];
        optimal_rotation[4] = svd_right[3] * svd_left[3] + svd_right[4] * svd_left[4] + svd_right[5] * svd_left[5];
        optimal_rotation[5] = svd_right[3] * svd_left[6] + svd_right[4] * svd_left[7] + svd_right[5] * svd_left[8];
        optimal_rotation[6] = svd_right[6] * svd_left[0] + svd_right[7] * svd_left[1] + svd_right[8] * svd_left[2];
        optimal_rotation[7] = svd_right[6] * svd_left[3] + svd_right[7] * svd_left[4] + svd_right[8] * svd_left[5];
        optimal_rotation[8] = svd_right[6] * svd_left[6] + svd_right[7] * svd_left[7] + svd_right[8] * svd_left[8];

        // Handle reflection: if det(R) < 0, negate third column of V and recompute
        nk_f32_t rotation_determinant = nk_det3x3_f32_(optimal_rotation);
        if (rotation_determinant < 0) {
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
            optimal_rotation[2] * cross_covariance[6] + //
            optimal_rotation[3] * cross_covariance[1] + optimal_rotation[4] * cross_covariance[4] +
            optimal_rotation[5] * cross_covariance[7] + //
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }

    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];
    if (scale) *scale = 1.0f;

    // Folded SSD via trace identity: SSD = ‖a-ā‖² + ‖b-b̄‖² − 2·trace(R · H_centered).
    nk_f32_t centered_norm_squared_a = norm_squared_a -
                                       (nk_f32_t)n * (centroid_a_x * centroid_a_x + centroid_a_y * centroid_a_y +
                                                      centroid_a_z * centroid_a_z);
    nk_f32_t centered_norm_squared_b = norm_squared_b -
                                       (nk_f32_t)n * (centroid_b_x * centroid_b_x + centroid_b_y * centroid_b_y +
                                                      centroid_b_z * centroid_b_z);
    if (centered_norm_squared_a < 0.0f) centered_norm_squared_a = 0.0f;
    if (centered_norm_squared_b < 0.0f) centered_norm_squared_b = 0.0f;
    nk_f32_t sum_squared = centered_norm_squared_a + centered_norm_squared_b - 2.0f * trace_rotation_covariance;
    if (sum_squared < 0.0f) sum_squared = 0.0f;
    *result = nk_f32_sqrt_neon(sum_squared * inv_n);
}

/**
 *  @brief Umeyama algorithm (Kabsch with uniform scale) using NEON FHM widening FMA.
 *  Finds rotation R and scale c minimizing ‖c·R·a − b‖² after centroid alignment.
 */
NK_PUBLIC void nk_umeyama_f16_neonfhm(nk_f16_t const *a, nk_f16_t const *b, nk_size_t n, nk_f32_t *a_centroid,
                                      nk_f32_t *b_centroid, nk_f32_t *rotation, nk_f32_t *scale, nk_f32_t *result) {
    float32x4_t const zeros_f32x4 = vdupq_n_f32(0);

    float32x4_t sum_a_x_f32x4 = zeros_f32x4, sum_a_y_f32x4 = zeros_f32x4, sum_a_z_f32x4 = zeros_f32x4;
    float32x4_t sum_b_x_f32x4 = zeros_f32x4, sum_b_y_f32x4 = zeros_f32x4, sum_b_z_f32x4 = zeros_f32x4;

    float32x4_t covariance_xx_low_f32x4 = zeros_f32x4, covariance_xx_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_xy_low_f32x4 = zeros_f32x4, covariance_xy_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_xz_low_f32x4 = zeros_f32x4, covariance_xz_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_yx_low_f32x4 = zeros_f32x4, covariance_yx_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_yy_low_f32x4 = zeros_f32x4, covariance_yy_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_yz_low_f32x4 = zeros_f32x4, covariance_yz_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_zx_low_f32x4 = zeros_f32x4, covariance_zx_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_zy_low_f32x4 = zeros_f32x4, covariance_zy_high_f32x4 = zeros_f32x4;
    float32x4_t covariance_zz_low_f32x4 = zeros_f32x4, covariance_zz_high_f32x4 = zeros_f32x4;
    float32x4_t norm_squared_a_low_f32x4 = zeros_f32x4, norm_squared_a_high_f32x4 = zeros_f32x4;
    float32x4_t norm_squared_b_low_f32x4 = zeros_f32x4, norm_squared_b_high_f32x4 = zeros_f32x4;

    float16x8_t a_x_f16x8, a_y_f16x8, a_z_f16x8;
    float16x8_t b_x_f16x8, b_y_f16x8, b_z_f16x8;
    nk_size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        nk_deinterleave_f16x8_to_f16x8x3_neonfhm_(a + i * 3, &a_x_f16x8, &a_y_f16x8, &a_z_f16x8);
        nk_deinterleave_f16x8_to_f16x8x3_neonfhm_(b + i * 3, &b_x_f16x8, &b_y_f16x8, &b_z_f16x8);

        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, vcvt_f32_f16(vget_low_f16(a_x_f16x8)));
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, vcvt_high_f32_f16(a_x_f16x8));
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, vcvt_f32_f16(vget_low_f16(a_y_f16x8)));
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, vcvt_high_f32_f16(a_y_f16x8));
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, vcvt_f32_f16(vget_low_f16(a_z_f16x8)));
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, vcvt_high_f32_f16(a_z_f16x8));
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, vcvt_f32_f16(vget_low_f16(b_x_f16x8)));
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, vcvt_high_f32_f16(b_x_f16x8));
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, vcvt_f32_f16(vget_low_f16(b_y_f16x8)));
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, vcvt_high_f32_f16(b_y_f16x8));
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, vcvt_f32_f16(vget_low_f16(b_z_f16x8)));
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, vcvt_high_f32_f16(b_z_f16x8));

        covariance_xx_low_f32x4 = vfmlalq_low_f16(covariance_xx_low_f32x4, a_x_f16x8, b_x_f16x8);
        covariance_xx_high_f32x4 = vfmlalq_high_f16(covariance_xx_high_f32x4, a_x_f16x8, b_x_f16x8);
        covariance_xy_low_f32x4 = vfmlalq_low_f16(covariance_xy_low_f32x4, a_x_f16x8, b_y_f16x8);
        covariance_xy_high_f32x4 = vfmlalq_high_f16(covariance_xy_high_f32x4, a_x_f16x8, b_y_f16x8);
        covariance_xz_low_f32x4 = vfmlalq_low_f16(covariance_xz_low_f32x4, a_x_f16x8, b_z_f16x8);
        covariance_xz_high_f32x4 = vfmlalq_high_f16(covariance_xz_high_f32x4, a_x_f16x8, b_z_f16x8);
        covariance_yx_low_f32x4 = vfmlalq_low_f16(covariance_yx_low_f32x4, a_y_f16x8, b_x_f16x8);
        covariance_yx_high_f32x4 = vfmlalq_high_f16(covariance_yx_high_f32x4, a_y_f16x8, b_x_f16x8);
        covariance_yy_low_f32x4 = vfmlalq_low_f16(covariance_yy_low_f32x4, a_y_f16x8, b_y_f16x8);
        covariance_yy_high_f32x4 = vfmlalq_high_f16(covariance_yy_high_f32x4, a_y_f16x8, b_y_f16x8);
        covariance_yz_low_f32x4 = vfmlalq_low_f16(covariance_yz_low_f32x4, a_y_f16x8, b_z_f16x8);
        covariance_yz_high_f32x4 = vfmlalq_high_f16(covariance_yz_high_f32x4, a_y_f16x8, b_z_f16x8);
        covariance_zx_low_f32x4 = vfmlalq_low_f16(covariance_zx_low_f32x4, a_z_f16x8, b_x_f16x8);
        covariance_zx_high_f32x4 = vfmlalq_high_f16(covariance_zx_high_f32x4, a_z_f16x8, b_x_f16x8);
        covariance_zy_low_f32x4 = vfmlalq_low_f16(covariance_zy_low_f32x4, a_z_f16x8, b_y_f16x8);
        covariance_zy_high_f32x4 = vfmlalq_high_f16(covariance_zy_high_f32x4, a_z_f16x8, b_y_f16x8);
        covariance_zz_low_f32x4 = vfmlalq_low_f16(covariance_zz_low_f32x4, a_z_f16x8, b_z_f16x8);
        covariance_zz_high_f32x4 = vfmlalq_high_f16(covariance_zz_high_f32x4, a_z_f16x8, b_z_f16x8);

        norm_squared_a_low_f32x4 = vfmlalq_low_f16(norm_squared_a_low_f32x4, a_x_f16x8, a_x_f16x8);
        norm_squared_a_high_f32x4 = vfmlalq_high_f16(norm_squared_a_high_f32x4, a_x_f16x8, a_x_f16x8);
        norm_squared_a_low_f32x4 = vfmlalq_low_f16(norm_squared_a_low_f32x4, a_y_f16x8, a_y_f16x8);
        norm_squared_a_high_f32x4 = vfmlalq_high_f16(norm_squared_a_high_f32x4, a_y_f16x8, a_y_f16x8);
        norm_squared_a_low_f32x4 = vfmlalq_low_f16(norm_squared_a_low_f32x4, a_z_f16x8, a_z_f16x8);
        norm_squared_a_high_f32x4 = vfmlalq_high_f16(norm_squared_a_high_f32x4, a_z_f16x8, a_z_f16x8);
        norm_squared_b_low_f32x4 = vfmlalq_low_f16(norm_squared_b_low_f32x4, b_x_f16x8, b_x_f16x8);
        norm_squared_b_high_f32x4 = vfmlalq_high_f16(norm_squared_b_high_f32x4, b_x_f16x8, b_x_f16x8);
        norm_squared_b_low_f32x4 = vfmlalq_low_f16(norm_squared_b_low_f32x4, b_y_f16x8, b_y_f16x8);
        norm_squared_b_high_f32x4 = vfmlalq_high_f16(norm_squared_b_high_f32x4, b_y_f16x8, b_y_f16x8);
        norm_squared_b_low_f32x4 = vfmlalq_low_f16(norm_squared_b_low_f32x4, b_z_f16x8, b_z_f16x8);
        norm_squared_b_high_f32x4 = vfmlalq_high_f16(norm_squared_b_high_f32x4, b_z_f16x8, b_z_f16x8);
    }

    if (i < n) {
        nk_partial_deinterleave_f16_to_f16x8x3_neonfhm_(a + i * 3, n - i, &a_x_f16x8, &a_y_f16x8, &a_z_f16x8);
        nk_partial_deinterleave_f16_to_f16x8x3_neonfhm_(b + i * 3, n - i, &b_x_f16x8, &b_y_f16x8, &b_z_f16x8);

        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, vcvt_f32_f16(vget_low_f16(a_x_f16x8)));
        sum_a_x_f32x4 = vaddq_f32(sum_a_x_f32x4, vcvt_high_f32_f16(a_x_f16x8));
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, vcvt_f32_f16(vget_low_f16(a_y_f16x8)));
        sum_a_y_f32x4 = vaddq_f32(sum_a_y_f32x4, vcvt_high_f32_f16(a_y_f16x8));
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, vcvt_f32_f16(vget_low_f16(a_z_f16x8)));
        sum_a_z_f32x4 = vaddq_f32(sum_a_z_f32x4, vcvt_high_f32_f16(a_z_f16x8));
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, vcvt_f32_f16(vget_low_f16(b_x_f16x8)));
        sum_b_x_f32x4 = vaddq_f32(sum_b_x_f32x4, vcvt_high_f32_f16(b_x_f16x8));
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, vcvt_f32_f16(vget_low_f16(b_y_f16x8)));
        sum_b_y_f32x4 = vaddq_f32(sum_b_y_f32x4, vcvt_high_f32_f16(b_y_f16x8));
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, vcvt_f32_f16(vget_low_f16(b_z_f16x8)));
        sum_b_z_f32x4 = vaddq_f32(sum_b_z_f32x4, vcvt_high_f32_f16(b_z_f16x8));

        covariance_xx_low_f32x4 = vfmlalq_low_f16(covariance_xx_low_f32x4, a_x_f16x8, b_x_f16x8);
        covariance_xx_high_f32x4 = vfmlalq_high_f16(covariance_xx_high_f32x4, a_x_f16x8, b_x_f16x8);
        covariance_xy_low_f32x4 = vfmlalq_low_f16(covariance_xy_low_f32x4, a_x_f16x8, b_y_f16x8);
        covariance_xy_high_f32x4 = vfmlalq_high_f16(covariance_xy_high_f32x4, a_x_f16x8, b_y_f16x8);
        covariance_xz_low_f32x4 = vfmlalq_low_f16(covariance_xz_low_f32x4, a_x_f16x8, b_z_f16x8);
        covariance_xz_high_f32x4 = vfmlalq_high_f16(covariance_xz_high_f32x4, a_x_f16x8, b_z_f16x8);
        covariance_yx_low_f32x4 = vfmlalq_low_f16(covariance_yx_low_f32x4, a_y_f16x8, b_x_f16x8);
        covariance_yx_high_f32x4 = vfmlalq_high_f16(covariance_yx_high_f32x4, a_y_f16x8, b_x_f16x8);
        covariance_yy_low_f32x4 = vfmlalq_low_f16(covariance_yy_low_f32x4, a_y_f16x8, b_y_f16x8);
        covariance_yy_high_f32x4 = vfmlalq_high_f16(covariance_yy_high_f32x4, a_y_f16x8, b_y_f16x8);
        covariance_yz_low_f32x4 = vfmlalq_low_f16(covariance_yz_low_f32x4, a_y_f16x8, b_z_f16x8);
        covariance_yz_high_f32x4 = vfmlalq_high_f16(covariance_yz_high_f32x4, a_y_f16x8, b_z_f16x8);
        covariance_zx_low_f32x4 = vfmlalq_low_f16(covariance_zx_low_f32x4, a_z_f16x8, b_x_f16x8);
        covariance_zx_high_f32x4 = vfmlalq_high_f16(covariance_zx_high_f32x4, a_z_f16x8, b_x_f16x8);
        covariance_zy_low_f32x4 = vfmlalq_low_f16(covariance_zy_low_f32x4, a_z_f16x8, b_y_f16x8);
        covariance_zy_high_f32x4 = vfmlalq_high_f16(covariance_zy_high_f32x4, a_z_f16x8, b_y_f16x8);
        covariance_zz_low_f32x4 = vfmlalq_low_f16(covariance_zz_low_f32x4, a_z_f16x8, b_z_f16x8);
        covariance_zz_high_f32x4 = vfmlalq_high_f16(covariance_zz_high_f32x4, a_z_f16x8, b_z_f16x8);

        norm_squared_a_low_f32x4 = vfmlalq_low_f16(norm_squared_a_low_f32x4, a_x_f16x8, a_x_f16x8);
        norm_squared_a_high_f32x4 = vfmlalq_high_f16(norm_squared_a_high_f32x4, a_x_f16x8, a_x_f16x8);
        norm_squared_a_low_f32x4 = vfmlalq_low_f16(norm_squared_a_low_f32x4, a_y_f16x8, a_y_f16x8);
        norm_squared_a_high_f32x4 = vfmlalq_high_f16(norm_squared_a_high_f32x4, a_y_f16x8, a_y_f16x8);
        norm_squared_a_low_f32x4 = vfmlalq_low_f16(norm_squared_a_low_f32x4, a_z_f16x8, a_z_f16x8);
        norm_squared_a_high_f32x4 = vfmlalq_high_f16(norm_squared_a_high_f32x4, a_z_f16x8, a_z_f16x8);
        norm_squared_b_low_f32x4 = vfmlalq_low_f16(norm_squared_b_low_f32x4, b_x_f16x8, b_x_f16x8);
        norm_squared_b_high_f32x4 = vfmlalq_high_f16(norm_squared_b_high_f32x4, b_x_f16x8, b_x_f16x8);
        norm_squared_b_low_f32x4 = vfmlalq_low_f16(norm_squared_b_low_f32x4, b_y_f16x8, b_y_f16x8);
        norm_squared_b_high_f32x4 = vfmlalq_high_f16(norm_squared_b_high_f32x4, b_y_f16x8, b_y_f16x8);
        norm_squared_b_low_f32x4 = vfmlalq_low_f16(norm_squared_b_low_f32x4, b_z_f16x8, b_z_f16x8);
        norm_squared_b_high_f32x4 = vfmlalq_high_f16(norm_squared_b_high_f32x4, b_z_f16x8, b_z_f16x8);
    }

    // Combine low+high halves
    float32x4_t covariance_xx_f32x4 = vaddq_f32(covariance_xx_low_f32x4, covariance_xx_high_f32x4);
    float32x4_t covariance_xy_f32x4 = vaddq_f32(covariance_xy_low_f32x4, covariance_xy_high_f32x4);
    float32x4_t covariance_xz_f32x4 = vaddq_f32(covariance_xz_low_f32x4, covariance_xz_high_f32x4);
    float32x4_t covariance_yx_f32x4 = vaddq_f32(covariance_yx_low_f32x4, covariance_yx_high_f32x4);
    float32x4_t covariance_yy_f32x4 = vaddq_f32(covariance_yy_low_f32x4, covariance_yy_high_f32x4);
    float32x4_t covariance_yz_f32x4 = vaddq_f32(covariance_yz_low_f32x4, covariance_yz_high_f32x4);
    float32x4_t covariance_zx_f32x4 = vaddq_f32(covariance_zx_low_f32x4, covariance_zx_high_f32x4);
    float32x4_t covariance_zy_f32x4 = vaddq_f32(covariance_zy_low_f32x4, covariance_zy_high_f32x4);
    float32x4_t covariance_zz_f32x4 = vaddq_f32(covariance_zz_low_f32x4, covariance_zz_high_f32x4);
    float32x4_t norm_squared_a_f32x4 = vaddq_f32(norm_squared_a_low_f32x4, norm_squared_a_high_f32x4);
    float32x4_t norm_squared_b_f32x4 = vaddq_f32(norm_squared_b_low_f32x4, norm_squared_b_high_f32x4);

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
    nk_f32_t centroid_a_x = sum_a_x * inv_n, centroid_a_y = sum_a_y * inv_n, centroid_a_z = sum_a_z * inv_n;
    nk_f32_t centroid_b_x = sum_b_x * inv_n, centroid_b_y = sum_b_y * inv_n, centroid_b_z = sum_b_z * inv_n;

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

    nk_f32_t cross_covariance[9];
    cross_covariance[0] = covariance_x_x - (nk_f32_t)n * centroid_a_x * centroid_b_x;
    cross_covariance[1] = covariance_x_y - (nk_f32_t)n * centroid_a_x * centroid_b_y;
    cross_covariance[2] = covariance_x_z - (nk_f32_t)n * centroid_a_x * centroid_b_z;
    cross_covariance[3] = covariance_y_x - (nk_f32_t)n * centroid_a_y * centroid_b_x;
    cross_covariance[4] = covariance_y_y - (nk_f32_t)n * centroid_a_y * centroid_b_y;
    cross_covariance[5] = covariance_y_z - (nk_f32_t)n * centroid_a_y * centroid_b_z;
    cross_covariance[6] = covariance_z_x - (nk_f32_t)n * centroid_a_z * centroid_b_x;
    cross_covariance[7] = covariance_z_y - (nk_f32_t)n * centroid_a_z * centroid_b_y;
    cross_covariance[8] = covariance_z_z - (nk_f32_t)n * centroid_a_z * centroid_b_z;

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
    nk_f32_t scale_factor;
    if (covariance_offdiagonal_norm_squared < 1e-12f * covariance_diagonal_norm_squared && cross_covariance[0] > 0.0f &&
        cross_covariance[4] > 0.0f && cross_covariance[8] > 0.0f) {
        optimal_rotation[0] = 1, optimal_rotation[1] = 0, optimal_rotation[2] = 0, optimal_rotation[3] = 0,
        optimal_rotation[4] = 1, optimal_rotation[5] = 0, optimal_rotation[6] = 0, optimal_rotation[7] = 0,
        optimal_rotation[8] = 1;
        trace_rotation_covariance = cross_covariance[0] + cross_covariance[4] + cross_covariance[8];
        scale_factor = centered_norm_squared_a > 0.0f ? trace_rotation_covariance / centered_norm_squared_a : 0.0f;
    }
    else {
        // SVD of H = U · S · Vᵀ
        nk_f32_t svd_left[9], svd_diagonal[9], svd_right[9];
        nk_svd3x3_f32_(cross_covariance, svd_left, svd_diagonal, svd_right);

        // R = V · Uᵀ
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
        nk_f32_t rotation_determinant = nk_det3x3_f32_(optimal_rotation);
        nk_f32_t sign_det = rotation_determinant < 0 ? -1.0f : 1.0f;
        nk_f32_t trace_scaled_s = svd_diagonal[0] + svd_diagonal[4] + sign_det * svd_diagonal[8];
        scale_factor = centered_norm_squared_a > 0.0f ? trace_scaled_s / centered_norm_squared_a : 0.0f;

        if (rotation_determinant < 0) {
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
            optimal_rotation[2] * cross_covariance[6] + //
            optimal_rotation[3] * cross_covariance[1] + optimal_rotation[4] * cross_covariance[4] +
            optimal_rotation[5] * cross_covariance[7] + //
            optimal_rotation[6] * cross_covariance[2] + optimal_rotation[7] * cross_covariance[5] +
            optimal_rotation[8] * cross_covariance[8];
    }
    if (scale) *scale = scale_factor;

    if (rotation)
        for (int j = 0; j < 9; ++j) rotation[j] = optimal_rotation[j];

    // Folded SSD with scale: c²·‖a-ā‖² + ‖b-b̄‖² − 2c·trace(R · H_centered).
    nk_f32_t sum_squared = scale_factor * scale_factor * centered_norm_squared_a + centered_norm_squared_b -
                           2.0f * scale_factor * trace_rotation_covariance;
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

#endif // NK_TARGET_NEONFHM
#endif // NK_TARGET_ARM64_
#endif // NK_MESH_NEONFHM_H
