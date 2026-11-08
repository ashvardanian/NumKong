/**
 *  @file include/numkong/shapes/neon.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief Many-to-many shape distances between 3D point clouds for NEON.
 *
 *  @sa c/numkong/shapes.h
 *
 *  @section shapes_neon_instructions Key NEON Shape Instructions
 *
 *  @verbatim
 *  Intrinsic     Instruction                  A76        M5
 *  vfmaq_f64     FMLA (V.2D, V.2D, V.2D)      4cy @ 2p   3cy @ 4p
 *  vfmaq_f32     FMLA (V.4S, V.4S, V.4S)      4cy @ 2p   3cy @ 4p
 *  vsubq_f64     FSUB (V.2D, V.2D, V.2D)      2cy @ 2p   2cy @ 4p
 *  vminq_f64     FMIN (V.2D, V.2D, V.2D)      2cy @ 2p   2cy @ 4p
 *  vpminq_f32    FMINP (V.4S, V.4S, V.4S)     2cy @ 2p   2cy @ 4p
 *  vcvt_f64_f32  FCVTL (V.2D, V.2S)           3cy @ 2p   3cy @ 4p
 *  vcvt_f32_f16  FCVTL (V.4S, V.4H)           3cy @ 2p   3cy @ 4p
 *  vsqrtq_f64    FSQRT (V.2D, V.2D)           12cy @ 1p  9cy @ 1p
 *  vcltq_f64     FCMGT (V.2D, V.2D, V.2D)     2cy @ 2p   2cy @ 4p
 *  vbslq_f64     BSL (V.16B, V.16B, V.16B)    1cy @ 2p   2cy @ 4p
 *  @endverbatim
 *
 *  Chamfer and Hausdorff broadcast one outer point and sweep the inner cloud's zero-padded axes a
 *  vector at a time, keeping the running minima of all 8 flips in registers: f32 packs widen two
 *  points into f64 lanes, f16 packs four into f32 lanes. Padded lanes take an infinite x, so they
 *  never win a minimum. Chamfer scores both directions in that one sweep, keeping the shorter
 *  cloud's minima in the workspace. Hausdorff runs each direction and stops once every flip passes
 *  the limit, so it reads only the points that decide a pair.
 *
 *  The transport kinds run the serial simplex around tile pricing two arcs at a time. Arc costs
 *  round in fused multiply-adds, and the scalar cost behind the tree's potentials runs the same
 *  lanes, so tree arcs keep a zero reduced cost.
 *
 *  RMSD sums four @c f16 points per step in f32 lanes, or two @c f32 or @c f64 points in f64 lanes,
 *  through the zero padding, and leaves the SVD scalar.
 */
#ifndef NUMKONG_SHAPES_NEON_H
#define NUMKONG_SHAPES_NEON_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include "numkong/types.h"
#include "numkong/cast/neon.h"     // `nk_load_f16x4_as_f32_neon_`
#include "numkong/scalar/neon.h"   // `nk_sqrt_f32_neon_`, `nk_sqrt_f64_neon_`
#include "numkong/shapes/serial.h" // `nk_shapes_best_flip_f32_serial_`, `nk_shapes_simplex_next_tile_serial_`
#include "numkong/dot/neon.h"      // `nk_dot2_f64x2_neon_`, `nk_dot_stable_sum_f64x2_neon_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

/** Lowers the running minima of the 8 flips by squared distances to four inner points, given each
 *  axis' difference and sum of the outer and inner coordinates. */
NUMKONG_INLINE void nk_shapes_lower_nearest_differences_f32_neon_(float32x4_t minus_x_f32x4, float32x4_t plus_x_f32x4,
                                                                  float32x4_t minus_y_f32x4, float32x4_t plus_y_f32x4,
                                                                  float32x4_t minus_z_f32x4, float32x4_t plus_z_f32x4,
                                                                  float32x4_t *nearest_f32x4) {
    float32x4_t const z_f32x4[2] = {minus_z_f32x4, plus_z_f32x4};
    float32x4_t const minus_xx_f32x4 = vmulq_f32(minus_x_f32x4, minus_x_f32x4);
    float32x4_t const plus_xx_f32x4 = vmulq_f32(plus_x_f32x4, plus_x_f32x4);
    float32x4_t const planar_f32x4[4] = {
        vfmaq_f32(minus_xx_f32x4, minus_y_f32x4, minus_y_f32x4), vfmaq_f32(plus_xx_f32x4, minus_y_f32x4, minus_y_f32x4),
        vfmaq_f32(minus_xx_f32x4, plus_y_f32x4, plus_y_f32x4), vfmaq_f32(plus_xx_f32x4, plus_y_f32x4, plus_y_f32x4)};
    for (int flip = 0; flip != 8; ++flip)
        nearest_f32x4[flip] = vminq_f32(nearest_f32x4[flip],
                                        vfmaq_f32(planar_f32x4[flip & 3], z_f32x4[flip >> 2], z_f32x4[flip >> 2]));
}

/** Lowers the 8 flips' nearest squares as @ref nk_shapes_lower_nearest_differences_f32_neon_ does,
 *  and the same squares into the inner points' @p reverse_minima , 8 axes @p reverse_stride apart
 *  in the workspace. */
NUMKONG_INLINE void nk_shapes_lower_mutual_differences_f32_neon_(float32x4_t minus_x_f32x4, float32x4_t plus_x_f32x4,
                                                                 float32x4_t minus_y_f32x4, float32x4_t plus_y_f32x4,
                                                                 float32x4_t minus_z_f32x4, float32x4_t plus_z_f32x4,
                                                                 float32x4_t *nearest_f32x4, nk_f32_t *reverse_minima,
                                                                 nk_size_t reverse_stride) {
    float32x4_t const z_f32x4[2] = {minus_z_f32x4, plus_z_f32x4};
    float32x4_t const minus_xx_f32x4 = vmulq_f32(minus_x_f32x4, minus_x_f32x4);
    float32x4_t const plus_xx_f32x4 = vmulq_f32(plus_x_f32x4, plus_x_f32x4);
    float32x4_t const planar_f32x4[4] = {
        vfmaq_f32(minus_xx_f32x4, minus_y_f32x4, minus_y_f32x4), vfmaq_f32(plus_xx_f32x4, minus_y_f32x4, minus_y_f32x4),
        vfmaq_f32(minus_xx_f32x4, plus_y_f32x4, plus_y_f32x4), vfmaq_f32(plus_xx_f32x4, plus_y_f32x4, plus_y_f32x4)};
    for (int flip = 0; flip != 8; ++flip) {
        float32x4_t const squares_f32x4 = vfmaq_f32(planar_f32x4[flip & 3], z_f32x4[flip >> 2], z_f32x4[flip >> 2]);
        nk_f32_t *reverse = reverse_minima + flip * reverse_stride;
        nearest_f32x4[flip] = vminq_f32(nearest_f32x4[flip], squares_f32x4);
        vst1q_f32(reverse, vminq_f32(vld1q_f32(reverse), squares_f32x4));
    }
}

/** The least of every flip's nearest squares over its lanes, flips 0 to 3 in the first vector and
 *  4 to 7 in the second. */
NUMKONG_INLINE void nk_shapes_least_flips_f32_neon_(float32x4_t const *nearest_f32x4, float32x4_t *least_f32x4) {
    least_f32x4[0] = vpminq_f32(vpminq_f32(nearest_f32x4[0], nearest_f32x4[1]),
                                vpminq_f32(nearest_f32x4[2], nearest_f32x4[3]));
    least_f32x4[1] = vpminq_f32(vpminq_f32(nearest_f32x4[4], nearest_f32x4[5]),
                                vpminq_f32(nearest_f32x4[6], nearest_f32x4[7]));
}

/** The roots of @p count reverse minima summed, in lanes past which the axis' padding stays out. */
NUMKONG_INLINE nk_f32_t nk_shapes_reverse_sum_f32_neon_(nk_f32_t const *reverse, nk_size_t count) {
    float32x4_t sum_f32x4 = vdupq_n_f32(0);
    nk_size_t index = 0;
    for (; index + 4 <= count; index += 4) sum_f32x4 = vaddq_f32(sum_f32x4, vsqrtq_f32(vld1q_f32(reverse + index)));
    nk_f32_t sum = vaddvq_f32(sum_f32x4);
    for (; index != count; ++index) sum += nk_sqrt_f32_neon_(reverse[index]);
    return sum;
}

/** The mask over a tail of @p inner_count points, in @p lanes of f16 or f32 inputs. */
NUMKONG_INLINE nk_b128_vec_t nk_shapes_tail_neon_(nk_size_t inner_count, nk_size_t lanes) {
    nk_b128_vec_t tail_vec;
    nk_size_t const remaining = inner_count & (lanes - 1);
    if (lanes == 8)
        tail_vec.u16x8 = vcltq_u16(vcombine_u16(vcreate_u16(0x0003000200010000ULL), vcreate_u16(0x0007000600050004ULL)),
                                   vdupq_n_u16((nk_u16_t)remaining));
    else
        tail_vec.u32x4 = vcltq_u32(vcombine_u32(vcreate_u32(0x0000000100000000ULL), vcreate_u32(0x0000000300000002ULL)),
                                   vdupq_n_u32((nk_u32_t)remaining));
    return tail_vec;
}

/** Adds the roots of one outer point's @p nearest_f32x4 onto the 8 @p sums_f32x4 , and whether
 *  every flip's finished rows pass @p limit , leaving those lower bounds in @p chamfers . */
NUMKONG_INLINE int nk_shapes_chamfer_row_f32_neon_(float32x4_t const *nearest_f32x4, float32x4_t *sums_f32x4,
                                                   float32x4_t threshold_f32x4, nk_f32_t outer_points, nk_f32_t limit,
                                                   nk_f32_t *chamfers) {
    float32x4_t least_f32x4[2];
    nk_shapes_least_flips_f32_neon_(nearest_f32x4, least_f32x4);
    sums_f32x4[0] = vaddq_f32(sums_f32x4[0], vsqrtq_f32(least_f32x4[0]));
    sums_f32x4[1] = vaddq_f32(sums_f32x4[1], vsqrtq_f32(least_f32x4[1]));
    if (!vminvq_u32(vandq_u32(vcgtq_f32(sums_f32x4[0], threshold_f32x4), vcgtq_f32(sums_f32x4[1], threshold_f32x4))))
        return 0;
    nk_f32_t sums[8];
    vst1q_f32(sums, sums_f32x4[0]), vst1q_f32(sums + 4, sums_f32x4[1]);
    return nk_shapes_mutual_past_limit_f32_serial_(sums, outer_points, limit, chamfers);
}

/** The 8 @p chamfers of a whole fused pair, from its rows' @p sums_f32x4 and the reverse minima. */
NUMKONG_INLINE void nk_shapes_chamfer_finish_f16_neon_(float32x4_t const *sums_f32x4,
                                                       nk_shapes_mutual_f16_t const *views, nk_f32_t *chamfers) {
    nk_f32_t const outer_points = (nk_f32_t)views->outer_count, inner_points = (nk_f32_t)views->inner_count;
    nk_f32_t sums[8];
    vst1q_f32(sums, sums_f32x4[0]), vst1q_f32(sums + 4, sums_f32x4[1]);
    for (int flip = 0; flip != 8; ++flip) {
        nk_f32_t const reverse_sum = nk_shapes_reverse_sum_f32_neon_(
            views->reverse_minima + flip * views->reverse_stride, views->inner_count);
        chamfers[flip] = (sums[flip] / outer_points + reverse_sum / inner_points) / 2;
    }
}

/** Raises the 8 @p farthest_f32x4 by one outer point's @p nearest_f32x4 , and whether every flip's
 *  farthest square passes @p limit , whose square rounds, so the distance confirms it. */
NUMKONG_INLINE int nk_shapes_hausdorff_row_f32_neon_(float32x4_t const *nearest_f32x4, float32x4_t *farthest_f32x4,
                                                     nk_f32_t limit, nk_f32_t limit_squared) {
    float32x4_t least_f32x4[2];
    nk_shapes_least_flips_f32_neon_(nearest_f32x4, least_f32x4);
    farthest_f32x4[0] = vmaxq_f32(farthest_f32x4[0], least_f32x4[0]);
    farthest_f32x4[1] = vmaxq_f32(farthest_f32x4[1], least_f32x4[1]);
    nk_f32_t const least = vminvq_f32(vminq_f32(farthest_f32x4[0], farthest_f32x4[1]));
    return least > limit_squared && nk_sqrt_f32_neon_(least) > limit;
}

#if NUMKONG_TARGET_NEON

/** Lowers the running minima of the 8 flips by one outer point's squared distances to two inner
 *  points, the difference and the sum of each axis standing for its two signs. */
NUMKONG_INLINE void nk_shapes_lower_nearest_f64_neon_(float64x2_t outer_x_f64x2, float64x2_t outer_y_f64x2,
                                                      float64x2_t outer_z_f64x2, float64x2_t inner_x_f64x2,
                                                      float64x2_t inner_y_f64x2, float64x2_t inner_z_f64x2,
                                                      float64x2_t *nearest_f64x2) {
    float64x2_t const minus_x_f64x2 = vsubq_f64(outer_x_f64x2, inner_x_f64x2);
    float64x2_t const plus_x_f64x2 = vaddq_f64(outer_x_f64x2, inner_x_f64x2);
    float64x2_t const minus_y_f64x2 = vsubq_f64(outer_y_f64x2, inner_y_f64x2);
    float64x2_t const plus_y_f64x2 = vaddq_f64(outer_y_f64x2, inner_y_f64x2);
    float64x2_t const z_f64x2[2] = {vsubq_f64(outer_z_f64x2, inner_z_f64x2), vaddq_f64(outer_z_f64x2, inner_z_f64x2)};
    float64x2_t const minus_xx_f64x2 = vmulq_f64(minus_x_f64x2, minus_x_f64x2);
    float64x2_t const plus_xx_f64x2 = vmulq_f64(plus_x_f64x2, plus_x_f64x2);
    float64x2_t const planar_f64x2[4] = {
        vfmaq_f64(minus_xx_f64x2, minus_y_f64x2, minus_y_f64x2), vfmaq_f64(plus_xx_f64x2, minus_y_f64x2, minus_y_f64x2),
        vfmaq_f64(minus_xx_f64x2, plus_y_f64x2, plus_y_f64x2), vfmaq_f64(plus_xx_f64x2, plus_y_f64x2, plus_y_f64x2)};
    for (int flip = 0; flip != 8; ++flip)
        nearest_f64x2[flip] = vminq_f64(nearest_f64x2[flip],
                                        vfmaq_f64(planar_f64x2[flip & 3], z_f64x2[flip >> 2], z_f64x2[flip >> 2]));
}

/** Lowers the 8 flips' nearest squares as @ref nk_shapes_lower_nearest_f64_neon_ does, and the same
 *  squares into the inner points' @p reverse_minima , 8 axes @p reverse_stride apart. */
NUMKONG_INLINE void nk_shapes_lower_mutual_f64_neon_(float64x2_t outer_x_f64x2, float64x2_t outer_y_f64x2,
                                                     float64x2_t outer_z_f64x2, float64x2_t inner_x_f64x2,
                                                     float64x2_t inner_y_f64x2, float64x2_t inner_z_f64x2,
                                                     float64x2_t *nearest_f64x2, nk_f64_t *reverse_minima,
                                                     nk_size_t reverse_stride) {
    float64x2_t const minus_x_f64x2 = vsubq_f64(outer_x_f64x2, inner_x_f64x2);
    float64x2_t const plus_x_f64x2 = vaddq_f64(outer_x_f64x2, inner_x_f64x2);
    float64x2_t const minus_y_f64x2 = vsubq_f64(outer_y_f64x2, inner_y_f64x2);
    float64x2_t const plus_y_f64x2 = vaddq_f64(outer_y_f64x2, inner_y_f64x2);
    float64x2_t const z_f64x2[2] = {vsubq_f64(outer_z_f64x2, inner_z_f64x2), vaddq_f64(outer_z_f64x2, inner_z_f64x2)};
    float64x2_t const minus_xx_f64x2 = vmulq_f64(minus_x_f64x2, minus_x_f64x2);
    float64x2_t const plus_xx_f64x2 = vmulq_f64(plus_x_f64x2, plus_x_f64x2);
    float64x2_t const planar_f64x2[4] = {
        vfmaq_f64(minus_xx_f64x2, minus_y_f64x2, minus_y_f64x2), vfmaq_f64(plus_xx_f64x2, minus_y_f64x2, minus_y_f64x2),
        vfmaq_f64(minus_xx_f64x2, plus_y_f64x2, plus_y_f64x2), vfmaq_f64(plus_xx_f64x2, plus_y_f64x2, plus_y_f64x2)};
    for (int flip = 0; flip != 8; ++flip) {
        float64x2_t const squares_f64x2 = vfmaq_f64(planar_f64x2[flip & 3], z_f64x2[flip >> 2], z_f64x2[flip >> 2]);
        nk_f64_t *reverse = reverse_minima + flip * reverse_stride;
        nearest_f64x2[flip] = vminq_f64(nearest_f64x2[flip], squares_f64x2);
        vst1q_f64(reverse, vminq_f64(vld1q_f64(reverse), squares_f64x2));
    }
}

/** One outer point's sweep of @c f32 inner points in f64 lanes, two at a time, the last pair of
 *  an odd count with an infinite x in its padded lane. */
NUMKONG_INLINE void nk_shapes_sweep_nearest_f32_neon_(float64x2_t outer_x_f64x2, float64x2_t outer_y_f64x2,
                                                      float64x2_t outer_z_f64x2, nk_f32_t const *inner_axes,
                                                      nk_size_t inner_stride, nk_size_t inner_count,
                                                      float64x2_t *nearest_f64x2) {
    nk_size_t const pairs_end = inner_count & ~(nk_size_t)1;
    for (nk_size_t inner_index = 0; inner_index != pairs_end; inner_index += 2)
        nk_shapes_lower_nearest_f64_neon_(
            outer_x_f64x2, outer_y_f64x2, outer_z_f64x2, vcvt_f64_f32(vld1_f32(inner_axes + inner_index)),
            vcvt_f64_f32(vld1_f32(inner_axes + inner_stride + inner_index)),
            vcvt_f64_f32(vld1_f32(inner_axes + 2 * inner_stride + inner_index)), nearest_f64x2);
    if (pairs_end != inner_count) {
        float64x2_t const inner_x_f64x2 = vsetq_lane_f64(NUMKONG_F64_INF,
                                                         vcvt_f64_f32(vld1_f32(inner_axes + pairs_end)), 1);
        nk_shapes_lower_nearest_f64_neon_(outer_x_f64x2, outer_y_f64x2, outer_z_f64x2, inner_x_f64x2,
                                          vcvt_f64_f32(vld1_f32(inner_axes + inner_stride + pairs_end)),
                                          vcvt_f64_f32(vld1_f32(inner_axes + 2 * inner_stride + pairs_end)),
                                          nearest_f64x2);
    }
}

/** @ref nk_shapes_sweep_nearest_f32_neon_ that also lowers the views' reverse minima. */
NUMKONG_INLINE void nk_shapes_sweep_mutual_f32_neon_(float64x2_t outer_x_f64x2, float64x2_t outer_y_f64x2,
                                                     float64x2_t outer_z_f64x2, nk_shapes_mutual_f32_t const *views,
                                                     float64x2_t *nearest_f64x2) {
    nk_f32_t const *inner_axes = views->inner_axes;
    nk_size_t const inner_stride = views->inner_stride, inner_count = views->inner_count;
    nk_size_t const pairs_end = inner_count & ~(nk_size_t)1;
    for (nk_size_t inner_index = 0; inner_index != pairs_end; inner_index += 2)
        nk_shapes_lower_mutual_f64_neon_(outer_x_f64x2, outer_y_f64x2, outer_z_f64x2,
                                         vcvt_f64_f32(vld1_f32(inner_axes + inner_index)),
                                         vcvt_f64_f32(vld1_f32(inner_axes + inner_stride + inner_index)),
                                         vcvt_f64_f32(vld1_f32(inner_axes + 2 * inner_stride + inner_index)),
                                         nearest_f64x2, views->reverse_minima + inner_index, views->reverse_stride);
    if (pairs_end != inner_count) {
        float64x2_t const inner_x_f64x2 = vsetq_lane_f64(NUMKONG_F64_INF,
                                                         vcvt_f64_f32(vld1_f32(inner_axes + pairs_end)), 1);
        nk_shapes_lower_mutual_f64_neon_(outer_x_f64x2, outer_y_f64x2, outer_z_f64x2, inner_x_f64x2,
                                         vcvt_f64_f32(vld1_f32(inner_axes + inner_stride + pairs_end)),
                                         vcvt_f64_f32(vld1_f32(inner_axes + 2 * inner_stride + pairs_end)),
                                         nearest_f64x2, views->reverse_minima + pairs_end, views->reverse_stride);
    }
}

/** The least of every flip's nearest squares over its lanes, two flips per vector. */
NUMKONG_INLINE void nk_shapes_least_flips_f64_neon_(float64x2_t const *nearest_f64x2, float64x2_t *least_f64x2) {
    for (int pair = 0; pair != 4; ++pair)
        least_f64x2[pair] = vpminq_f64(nearest_f64x2[2 * pair], nearest_f64x2[2 * pair + 1]);
}

/** Fused Chamfer flips of @c f32 packs in f64 lanes, reductions included. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f32_neon_(void const *row_packed, void const *column_packed,
                                                             nk_f64_t limit, void *workspace, nk_size_t workspace_bytes,
                                                             nk_f64_t *chamfers) {
    nk_shapes_mutual_f32_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f32_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    float64x2_t const infinity_f64x2 = vdupq_n_f64(NUMKONG_F64_INF);
    for (nk_size_t index = 0; index < 8 * views.reverse_stride; index += 2)
        vst1q_f64(views.reverse_minima + index, infinity_f64x2);
    nk_f64_t const outer_points = (nk_f64_t)views.outer_count, inner_points = (nk_f64_t)views.inner_count;
    float64x2_t const threshold_f64x2 = vdupq_n_f64(2 * limit * outer_points);
    float64x2_t sums_f64x2[4], nearest_f64x2[8], least_f64x2[4];
    for (int pair = 0; pair != 4; ++pair) sums_f64x2[pair] = vdupq_n_f64(0);
    nk_f64_t sums[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        float64x2_t const outer_x_f64x2 = vdupq_n_f64(views.outer_axes[outer_index]);
        float64x2_t const outer_y_f64x2 = vdupq_n_f64(views.outer_axes[views.outer_stride + outer_index]);
        float64x2_t const outer_z_f64x2 = vdupq_n_f64(views.outer_axes[2 * views.outer_stride + outer_index]);
        for (int flip = 0; flip != 8; ++flip) nearest_f64x2[flip] = infinity_f64x2;
        nk_shapes_sweep_mutual_f32_neon_(outer_x_f64x2, outer_y_f64x2, outer_z_f64x2, &views, nearest_f64x2);
        nk_shapes_least_flips_f64_neon_(nearest_f64x2, least_f64x2);
        uint64x2_t exceeded_u64x2 = vdupq_n_u64(~(nk_u64_t)0);
        for (int pair = 0; pair != 4; ++pair) {
            sums_f64x2[pair] = vaddq_f64(sums_f64x2[pair], vsqrtq_f64(least_f64x2[pair]));
            exceeded_u64x2 = vandq_u64(exceeded_u64x2, vcgtq_f64(sums_f64x2[pair], threshold_f64x2));
        }
        if (vgetq_lane_u64(exceeded_u64x2, 0) && vgetq_lane_u64(exceeded_u64x2, 1)) {
            for (int pair = 0; pair != 4; ++pair) vst1q_f64(sums + 2 * pair, sums_f64x2[pair]);
            if (nk_shapes_mutual_past_limit_f64_serial_(sums, outer_points, limit, chamfers)) return nk_success_k;
        }
    }
    for (int pair = 0; pair != 4; ++pair) vst1q_f64(sums + 2 * pair, sums_f64x2[pair]);
    for (int flip = 0; flip != 8; ++flip) {
        nk_f64_t const *reverse = views.reverse_minima + flip * views.reverse_stride;
        float64x2_t sum_f64x2 = vdupq_n_f64(0);
        nk_size_t index = 0;
        for (; index + 2 <= views.inner_count; index += 2)
            sum_f64x2 = vaddq_f64(sum_f64x2, vsqrtq_f64(vld1q_f64(reverse + index)));
        nk_f64_t reverse_sum = vaddvq_f64(sum_f64x2);
        if (index != views.inner_count) reverse_sum += nk_sqrt_f64_neon_(reverse[index]);
        chamfers[flip] = (sums[flip] / outer_points + reverse_sum / inner_points) / 2;
    }
    return nk_success_k;
}

/** Hausdorff flips of @c f32 packs in f64 lanes, stopping as the @c f16 pair does. */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f32_neon_(void const *row_packed, void const *column_packed,
                                                        nk_f64_t limit, nk_f64_t *hausdorffs) {
    nk_size_t const rows = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    nk_size_t const columns = (nk_size_t)((nk_shapes_packed_header_f64_t const *)column_packed)->points;
    if (!rows || !columns) {
        for (int flip = 0; flip != 8; ++flip) hausdorffs[flip] = rows == columns ? 0 : NUMKONG_F64_INF;
        return;
    }
    nk_f64_t const gap = nk_shapes_radius_gap_f64_serial_(row_packed, column_packed);
    if (gap > limit) {
        for (int flip = 0; flip != 8; ++flip) hausdorffs[flip] = gap;
        return;
    }
    nk_size_t row_stride, column_stride;
    nk_f32_t const *row_axes = nk_shapes_packed_axes_f32_serial_(row_packed, &row_stride);
    nk_f32_t const *column_axes = nk_shapes_packed_axes_f32_serial_(column_packed, &column_stride);
    nk_f64_t const limit_squared = limit * limit;
    float64x2_t const infinity_f64x2 = vdupq_n_f64(NUMKONG_F64_INF);
    float64x2_t farthest_f64x2[4], nearest_f64x2[8], least_f64x2[4];
    for (int pair = 0; pair != 4; ++pair) farthest_f64x2[pair] = vdupq_n_f64(0);
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_f32_t const *outer_axes = direction ? column_axes : row_axes,
                       *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            for (int flip = 0; flip != 8; ++flip) nearest_f64x2[flip] = infinity_f64x2;
            nk_shapes_sweep_nearest_f32_neon_(vdupq_n_f64(outer_axes[outer_index]),
                                              vdupq_n_f64(outer_axes[outer_stride + outer_index]),
                                              vdupq_n_f64(outer_axes[2 * outer_stride + outer_index]), inner_axes,
                                              inner_stride, inner_count, nearest_f64x2);
            nk_shapes_least_flips_f64_neon_(nearest_f64x2, least_f64x2);
            float64x2_t overall_least_f64x2 = vdupq_n_f64(NUMKONG_F64_INF);
            for (int pair = 0; pair != 4; ++pair) {
                farthest_f64x2[pair] = vmaxq_f64(farthest_f64x2[pair], least_f64x2[pair]);
                overall_least_f64x2 = vminq_f64(overall_least_f64x2, farthest_f64x2[pair]);
            }
            nk_f64_t const least = vminvq_f64(overall_least_f64x2);
            rejected = least > limit_squared && nk_sqrt_f64_neon_(least) > limit;
        }
    }
    for (int pair = 0; pair != 4; ++pair) vst1q_f64(hausdorffs + 2 * pair, vsqrtq_f64(farthest_f64x2[pair]));
}

/** @copydoc nk_shapes_lower_nearest_f64_neon_ */
NUMKONG_INLINE void nk_shapes_lower_nearest_f32_neon_(float32x4_t outer_x_f32x4, float32x4_t outer_y_f32x4,
                                                      float32x4_t outer_z_f32x4, float32x4_t inner_x_f32x4,
                                                      float32x4_t inner_y_f32x4, float32x4_t inner_z_f32x4,
                                                      float32x4_t *nearest_f32x4) {
    nk_shapes_lower_nearest_differences_f32_neon_(
        vsubq_f32(outer_x_f32x4, inner_x_f32x4), vaddq_f32(outer_x_f32x4, inner_x_f32x4),
        vsubq_f32(outer_y_f32x4, inner_y_f32x4), vaddq_f32(outer_y_f32x4, inner_y_f32x4),
        vsubq_f32(outer_z_f32x4, inner_z_f32x4), vaddq_f32(outer_z_f32x4, inner_z_f32x4), nearest_f32x4);
}

/** @copydoc nk_shapes_lower_mutual_f64_neon_ */
NUMKONG_INLINE void nk_shapes_lower_mutual_f32_neon_(float32x4_t outer_x_f32x4, float32x4_t outer_y_f32x4,
                                                     float32x4_t outer_z_f32x4, float32x4_t inner_x_f32x4,
                                                     float32x4_t inner_y_f32x4, float32x4_t inner_z_f32x4,
                                                     float32x4_t *nearest_f32x4, nk_f32_t *reverse_minima,
                                                     nk_size_t reverse_stride) {
    nk_shapes_lower_mutual_differences_f32_neon_(
        vsubq_f32(outer_x_f32x4, inner_x_f32x4), vaddq_f32(outer_x_f32x4, inner_x_f32x4),
        vsubq_f32(outer_y_f32x4, inner_y_f32x4), vaddq_f32(outer_y_f32x4, inner_y_f32x4),
        vsubq_f32(outer_z_f32x4, inner_z_f32x4), vaddq_f32(outer_z_f32x4, inner_z_f32x4), nearest_f32x4, reverse_minima,
        reverse_stride);
}

/** One outer point's sweep of @c f16 inner points in f32 lanes, four at a time, the last
 *  quadruple of a count off the multiple with an infinite x in its padded lanes. */
NUMKONG_INLINE void nk_shapes_sweep_nearest_f16_neon_(float32x4_t outer_x_f32x4, float32x4_t outer_y_f32x4,
                                                      float32x4_t outer_z_f32x4, nk_f16_t const *inner_axes,
                                                      nk_size_t inner_stride, nk_size_t inner_count,
                                                      nk_b128_vec_t const *tail_vec, float32x4_t *nearest_f32x4) {
    nk_size_t const quadruples_end = inner_count & ~(nk_size_t)3;
    for (nk_size_t inner_index = 0; inner_index != quadruples_end; inner_index += 4)
        nk_shapes_lower_nearest_f32_neon_(
            outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, nk_load_f16x4_as_f32_neon_(inner_axes + inner_index),
            nk_load_f16x4_as_f32_neon_(inner_axes + inner_stride + inner_index),
            nk_load_f16x4_as_f32_neon_(inner_axes + 2 * inner_stride + inner_index), nearest_f32x4);
    if (quadruples_end != inner_count)
        nk_shapes_lower_nearest_f32_neon_(
            outer_x_f32x4, outer_y_f32x4, outer_z_f32x4,
            vbslq_f32(tail_vec->u32x4, nk_load_f16x4_as_f32_neon_(inner_axes + quadruples_end),
                      vdupq_n_f32(NUMKONG_F32_INF)),
            nk_load_f16x4_as_f32_neon_(inner_axes + inner_stride + quadruples_end),
            nk_load_f16x4_as_f32_neon_(inner_axes + 2 * inner_stride + quadruples_end), nearest_f32x4);
}

/** @ref nk_shapes_sweep_nearest_f16_neon_ that also lowers the views' reverse minima. */
NUMKONG_INLINE void nk_shapes_sweep_mutual_f16_neon_(float32x4_t outer_x_f32x4, float32x4_t outer_y_f32x4,
                                                     float32x4_t outer_z_f32x4, nk_shapes_mutual_f16_t const *views,
                                                     nk_b128_vec_t const *tail_vec, float32x4_t *nearest_f32x4) {
    nk_f16_t const *inner_axes = views->inner_axes;
    nk_size_t const inner_stride = views->inner_stride, inner_count = views->inner_count;
    nk_size_t const quadruples_end = inner_count & ~(nk_size_t)3;
    for (nk_size_t inner_index = 0; inner_index != quadruples_end; inner_index += 4)
        nk_shapes_lower_mutual_f32_neon_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4,
                                         nk_load_f16x4_as_f32_neon_(inner_axes + inner_index),
                                         nk_load_f16x4_as_f32_neon_(inner_axes + inner_stride + inner_index),
                                         nk_load_f16x4_as_f32_neon_(inner_axes + 2 * inner_stride + inner_index),
                                         nearest_f32x4, views->reverse_minima + inner_index, views->reverse_stride);
    if (quadruples_end != inner_count)
        nk_shapes_lower_mutual_f32_neon_(
            outer_x_f32x4, outer_y_f32x4, outer_z_f32x4,
            vbslq_f32(tail_vec->u32x4, nk_load_f16x4_as_f32_neon_(inner_axes + quadruples_end),
                      vdupq_n_f32(NUMKONG_F32_INF)),
            nk_load_f16x4_as_f32_neon_(inner_axes + inner_stride + quadruples_end),
            nk_load_f16x4_as_f32_neon_(inner_axes + 2 * inner_stride + quadruples_end), nearest_f32x4,
            views->reverse_minima + quadruples_end, views->reverse_stride);
}

/** Fused Chamfer flips of @c f16 packs, four inner points per load, reductions in lanes. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f16_neon_(void const *row_packed, void const *column_packed,
                                                             nk_f32_t limit, void *workspace, nk_size_t workspace_bytes,
                                                             nk_f32_t *chamfers) {
    nk_shapes_mutual_f16_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f16_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    float32x4_t const infinity_f32x4 = vdupq_n_f32(NUMKONG_F32_INF);
    for (nk_size_t index = 0; index < 8 * views.reverse_stride; index += 4)
        vst1q_f32(views.reverse_minima + index, infinity_f32x4);
    nk_b128_vec_t const tail_vec = nk_shapes_tail_neon_(views.inner_count, 4);
    nk_u16_t const *outer_bits = (nk_u16_t const *)views.outer_axes;
    nk_f32_t const outer_points = (nk_f32_t)views.outer_count;
    float32x4_t const threshold_f32x4 = vdupq_n_f32(2 * limit * outer_points);
    float32x4_t sums_f32x4[2] = {vdupq_n_f32(0), vdupq_n_f32(0)}, nearest_f32x4[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        float32x4_t const outer_x_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_dup_u16(outer_bits + outer_index)));
        float32x4_t const outer_y_f32x4 = vcvt_f32_f16(
            vreinterpret_f16_u16(vld1_dup_u16(outer_bits + views.outer_stride + outer_index)));
        float32x4_t const outer_z_f32x4 = vcvt_f32_f16(
            vreinterpret_f16_u16(vld1_dup_u16(outer_bits + 2 * views.outer_stride + outer_index)));
        for (int flip = 0; flip != 8; ++flip) nearest_f32x4[flip] = infinity_f32x4;
        nk_shapes_sweep_mutual_f16_neon_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, &views, &tail_vec, nearest_f32x4);
        if (nk_shapes_chamfer_row_f32_neon_(nearest_f32x4, sums_f32x4, threshold_f32x4, outer_points, limit, chamfers))
            return nk_success_k;
    }
    nk_shapes_chamfer_finish_f16_neon_(sums_f32x4, &views, chamfers);
    return nk_success_k;
}

/** Hausdorff flips of @c f16 packs, four inner points per load, stopping once every flip's
 *  farthest square passes @p limit . */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f16_neon_(void const *row_packed, void const *column_packed,
                                                        nk_f32_t limit, nk_f32_t *hausdorffs) {
    nk_size_t const rows = ((nk_shapes_packed_header_f32_t const *)row_packed)->points;
    nk_size_t const columns = ((nk_shapes_packed_header_f32_t const *)column_packed)->points;
    if (!rows || !columns) {
        for (int flip = 0; flip != 8; ++flip) hausdorffs[flip] = rows == columns ? 0 : NUMKONG_F32_INF;
        return;
    }
    nk_f32_t const gap = nk_shapes_radius_gap_f32_serial_(row_packed, column_packed);
    if (gap > limit) {
        for (int flip = 0; flip != 8; ++flip) hausdorffs[flip] = gap;
        return;
    }
    nk_size_t row_stride, column_stride;
    nk_f16_t const *row_axes = nk_shapes_packed_axes_f16_serial_(row_packed, &row_stride);
    nk_f16_t const *column_axes = nk_shapes_packed_axes_f16_serial_(column_packed, &column_stride);
    nk_b128_vec_t const row_tail_vec = nk_shapes_tail_neon_(rows, 4);
    nk_b128_vec_t const column_tail_vec = nk_shapes_tail_neon_(columns, 4);
    nk_f32_t const limit_squared = limit * limit;
    float32x4_t const infinity_f32x4 = vdupq_n_f32(NUMKONG_F32_INF);
    float32x4_t farthest_f32x4[2] = {vdupq_n_f32(0), vdupq_n_f32(0)}, nearest_f32x4[8];
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_u16_t const *outer_bits = (nk_u16_t const *)(direction ? column_axes : row_axes);
        nk_f16_t const *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        nk_b128_vec_t const *tail_vec = direction ? &row_tail_vec : &column_tail_vec;
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            float32x4_t const outer_x_f32x4 = vcvt_f32_f16(
                vreinterpret_f16_u16(vld1_dup_u16(outer_bits + outer_index)));
            float32x4_t const outer_y_f32x4 = vcvt_f32_f16(
                vreinterpret_f16_u16(vld1_dup_u16(outer_bits + outer_stride + outer_index)));
            float32x4_t const outer_z_f32x4 = vcvt_f32_f16(
                vreinterpret_f16_u16(vld1_dup_u16(outer_bits + 2 * outer_stride + outer_index)));
            for (int flip = 0; flip != 8; ++flip) nearest_f32x4[flip] = infinity_f32x4;
            nk_shapes_sweep_nearest_f16_neon_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_axes, inner_stride,
                                              inner_count, tail_vec, nearest_f32x4);
            rejected = nk_shapes_hausdorff_row_f32_neon_(nearest_f32x4, farthest_f32x4, limit, limit_squared);
        }
    }
    vst1q_f32(hausdorffs, vsqrtq_f32(farthest_f32x4[0])), vst1q_f32(hausdorffs + 4, vsqrtq_f32(farthest_f32x4[1]));
}

/** Squared distances from a broadcast row point in fused multiply-adds, which also serve single
 *  arcs, so that tree arcs keep a zero reduced cost. */
NUMKONG_INLINE float32x2_t nk_shapes_point_costs_f32x2_neon_(float32x2_t row_x_f32x2, float32x2_t row_y_f32x2,
                                                             float32x2_t row_z_f32x2, float32x2_t column_x_f32x2,
                                                             float32x2_t column_y_f32x2, float32x2_t column_z_f32x2) {
    float32x2_t const delta_x_f32x2 = vsub_f32(row_x_f32x2, column_x_f32x2);
    float32x2_t const delta_y_f32x2 = vsub_f32(row_y_f32x2, column_y_f32x2);
    float32x2_t const delta_z_f32x2 = vsub_f32(row_z_f32x2, column_z_f32x2);
    return vfma_f32(vfma_f32(vmul_f32(delta_x_f32x2, delta_x_f32x2), delta_y_f32x2, delta_y_f32x2), delta_z_f32x2,
                    delta_z_f32x2);
}

NUMKONG_INLINE nk_f32_t nk_shapes_point_cost_f32_neon_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t row,
                                                       nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    return vget_lane_f32(
        nk_shapes_point_costs_f32x2_neon_(
            vld1_dup_f32(row_points + row), vld1_dup_f32(row_points + stride + row),
            vld1_dup_f32(row_points + 2 * stride + row), vld1_dup_f32(rotated_points + column),
            vld1_dup_f32(rotated_points + stride + column), vld1_dup_f32(rotated_points + 2 * stride + column)),
        0);
}

/** The reduced costs (c − πᵢ) + πⱼ of a row's arcs to two columns, with matrix costs −G taken as
 *  πⱼ − (G + πᵢ), which rounds the same. */
NUMKONG_INLINE float64x2_t nk_shapes_reduced_costs_f32_neon_(nk_shapes_workspace_f32_serial_t const *views,
                                                             nk_f64_t const *gradient, nk_size_t rows,
                                                             nk_size_t columns, nk_size_t row, nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    float64x2_t const row_potential_f64x2 = vld1q_dup_f64(views->potential + row);
    float64x2_t const column_potentials_f64x2 = vld1q_f64(views->potential + rows + column);
    if (gradient)
        return vsubq_f64(column_potentials_f64x2,
                         vaddq_f64(vld1q_f64(gradient + row * columns + column), row_potential_f64x2));
    float64x2_t const costs_f64x2 = vcvt_f64_f32(nk_shapes_point_costs_f32x2_neon_(
        vld1_dup_f32(row_points + row), vld1_dup_f32(row_points + stride + row),
        vld1_dup_f32(row_points + 2 * stride + row), vld1_f32(rotated_points + column),
        vld1_f32(rotated_points + stride + column), vld1_f32(rotated_points + 2 * stride + column)));
    return vaddq_f64(vsubq_f64(costs_f64x2, row_potential_f64x2), column_potentials_f64x2);
}

/** Keeps the first of two reduced costs from @p column that beats the running least. */
NUMKONG_INLINE void nk_shapes_keep_least_f64x2_neon_(float64x2_t reduced_f64x2, nk_size_t row, nk_size_t column,
                                                     float64x2_t *least_f64x2, nk_f64_t *entering_cost,
                                                     nk_size_t *entering_row, nk_size_t *entering_column) {
    if (!vmaxvq_u32(vreinterpretq_u32_u64(vcltq_f64(reduced_f64x2, *least_f64x2)))) return;
    nk_f64_t const reduced_low = vgetq_lane_f64(reduced_f64x2, 0), reduced_high = vgetq_lane_f64(reduced_f64x2, 1);
    int const takes_high = reduced_high < reduced_low;
    *entering_cost = takes_high ? reduced_high : reduced_low;
    *entering_row = row, *entering_column = column + takes_high;
    *least_f64x2 = vdupq_n_f64(*entering_cost);
}

/** Keeps a tile's first least reduced cost below @p entering_cost, two columns at a time. The lane
 *  past an odd tile reads neighboring workspace and costs infinity. */
NUMKONG_INLINE void nk_shapes_price_tile_f32_neon_(nk_shapes_workspace_f32_serial_t const *views,
                                                   nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                   nk_size_t row, nk_size_t column, nk_size_t tile,
                                                   nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                   nk_size_t *entering_column) {
    float64x2_t least_f64x2 = vdupq_n_f64(*entering_cost);
    nk_size_t const pairs_end = tile & ~(nk_size_t)1;
    for (nk_size_t step = 0; step != pairs_end; step += 2)
        nk_shapes_keep_least_f64x2_neon_(
            nk_shapes_reduced_costs_f32_neon_(views, gradient, rows, columns, row, column + step), row, column + step,
            &least_f64x2, entering_cost, entering_row, entering_column);
    if (pairs_end == tile) return;
    float64x2_t const reduced_f64x2 = nk_shapes_reduced_costs_f32_neon_(views, gradient, rows, columns, row,
                                                                        column + pairs_end);
    nk_shapes_keep_least_f64x2_neon_(vsetq_lane_f64(NUMKONG_F64_INF, reduced_f64x2, 1), row, column + pairs_end,
                                     &least_f64x2, entering_cost, entering_row, entering_column);
}

NUMKONG_INLINE void nk_shapes_tree_costs_f32_neon_(nk_shapes_workspace_f32_serial_t const *views,
                                                   nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        views->potential[node] = gradient ? -gradient[row * columns + column]
                                          : (nk_f64_t)nk_shapes_point_cost_f32_neon_(views, row, column);
    }
}

NUMKONG_INLINE void nk_shapes_network_simplex_f32_neon_(nk_shapes_workspace_f32_serial_t const *views,
                                                        nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                        nk_f64_t cost_scale) {
    nk_shapes_simplex_serial_t simplex;
    nk_shapes_tree_costs_f32_neon_(views, gradient, rows, columns);
    nk_shapes_tree_potentials_serial_(views->tree, views->tree_stride, views->potential, rows);
    nk_shapes_simplex_start_serial_(&simplex, views->tree, views->tree_stride, views->potential, rows, columns,
                                    cost_scale);
    while (nk_shapes_simplex_next_tile_serial_(&simplex))
        nk_shapes_price_tile_f32_neon_(views, gradient, rows, columns, simplex.row, simplex.column, simplex.tile,
                                       &simplex.entering_cost, &simplex.entering_row, &simplex.entering_column);
}

NUMKONG_INLINE nk_f32_t nk_shapes_tree_cost_f32_neon_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t rows,
                                                      nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f32_t cost = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        cost += (nk_f32_t)flow[node] * nk_shapes_point_cost_f32_neon_(views, row, column);
    }
    return cost / (nk_f32_t)(rows * columns);
}

NUMKONG_INLINE float64x2_t nk_shapes_point_costs_f64x2_neon_(float64x2_t row_x_f64x2, float64x2_t row_y_f64x2,
                                                             float64x2_t row_z_f64x2, float64x2_t column_x_f64x2,
                                                             float64x2_t column_y_f64x2, float64x2_t column_z_f64x2) {
    float64x2_t const delta_x_f64x2 = vsubq_f64(row_x_f64x2, column_x_f64x2);
    float64x2_t const delta_y_f64x2 = vsubq_f64(row_y_f64x2, column_y_f64x2);
    float64x2_t const delta_z_f64x2 = vsubq_f64(row_z_f64x2, column_z_f64x2);
    return vfmaq_f64(vfmaq_f64(vmulq_f64(delta_x_f64x2, delta_x_f64x2), delta_y_f64x2, delta_y_f64x2), delta_z_f64x2,
                     delta_z_f64x2);
}

NUMKONG_INLINE nk_f64_t nk_shapes_point_cost_f64_neon_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t row,
                                                       nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    return vgetq_lane_f64(
        nk_shapes_point_costs_f64x2_neon_(
            vld1q_dup_f64(row_points + row), vld1q_dup_f64(row_points + stride + row),
            vld1q_dup_f64(row_points + 2 * stride + row), vld1q_dup_f64(rotated_points + column),
            vld1q_dup_f64(rotated_points + stride + column), vld1q_dup_f64(rotated_points + 2 * stride + column)),
        0);
}

NUMKONG_INLINE float64x2_t nk_shapes_reduced_costs_f64_neon_(nk_shapes_workspace_f64_serial_t const *views,
                                                             nk_f64_t const *gradient, nk_size_t rows,
                                                             nk_size_t columns, nk_size_t row, nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    float64x2_t const row_potential_f64x2 = vld1q_dup_f64(views->potential + row);
    float64x2_t const column_potentials_f64x2 = vld1q_f64(views->potential + rows + column);
    if (gradient)
        return vsubq_f64(column_potentials_f64x2,
                         vaddq_f64(vld1q_f64(gradient + row * columns + column), row_potential_f64x2));
    float64x2_t const costs_f64x2 = nk_shapes_point_costs_f64x2_neon_(
        vld1q_dup_f64(row_points + row), vld1q_dup_f64(row_points + stride + row),
        vld1q_dup_f64(row_points + 2 * stride + row), vld1q_f64(rotated_points + column),
        vld1q_f64(rotated_points + stride + column), vld1q_f64(rotated_points + 2 * stride + column));
    return vaddq_f64(vsubq_f64(costs_f64x2, row_potential_f64x2), column_potentials_f64x2);
}

NUMKONG_INLINE void nk_shapes_price_tile_f64_neon_(nk_shapes_workspace_f64_serial_t const *views,
                                                   nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                   nk_size_t row, nk_size_t column, nk_size_t tile,
                                                   nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                   nk_size_t *entering_column) {
    float64x2_t least_f64x2 = vdupq_n_f64(*entering_cost);
    nk_size_t const pairs_end = tile & ~(nk_size_t)1;
    for (nk_size_t step = 0; step != pairs_end; step += 2)
        nk_shapes_keep_least_f64x2_neon_(
            nk_shapes_reduced_costs_f64_neon_(views, gradient, rows, columns, row, column + step), row, column + step,
            &least_f64x2, entering_cost, entering_row, entering_column);
    if (pairs_end == tile) return;
    float64x2_t const reduced_f64x2 = nk_shapes_reduced_costs_f64_neon_(views, gradient, rows, columns, row,
                                                                        column + pairs_end);
    nk_shapes_keep_least_f64x2_neon_(vsetq_lane_f64(NUMKONG_F64_INF, reduced_f64x2, 1), row, column + pairs_end,
                                     &least_f64x2, entering_cost, entering_row, entering_column);
}

NUMKONG_INLINE void nk_shapes_tree_costs_f64_neon_(nk_shapes_workspace_f64_serial_t const *views,
                                                   nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        views->potential[node] = gradient ? -gradient[row * columns + column]
                                          : nk_shapes_point_cost_f64_neon_(views, row, column);
    }
}

NUMKONG_INLINE void nk_shapes_network_simplex_f64_neon_(nk_shapes_workspace_f64_serial_t const *views,
                                                        nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                        nk_f64_t cost_scale) {
    nk_shapes_simplex_serial_t simplex;
    nk_shapes_tree_costs_f64_neon_(views, gradient, rows, columns);
    nk_shapes_tree_potentials_serial_(views->tree, views->tree_stride, views->potential, rows);
    nk_shapes_simplex_start_serial_(&simplex, views->tree, views->tree_stride, views->potential, rows, columns,
                                    cost_scale);
    while (nk_shapes_simplex_next_tile_serial_(&simplex))
        nk_shapes_price_tile_f64_neon_(views, gradient, rows, columns, simplex.row, simplex.column, simplex.tile,
                                       &simplex.entering_cost, &simplex.entering_row, &simplex.entering_column);
}

NUMKONG_INLINE nk_f64_t nk_shapes_tree_cost_f64_neon_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t rows,
                                                      nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f64_t cost = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        cost += (nk_f64_t)flow[node] * nk_shapes_point_cost_f64_neon_(views, row, column);
    }
    return cost / (nk_f64_t)(rows * columns);
}

NUMKONG_INLINE nk_f32_t nk_shapes_wasserstein_procrustes_f16_neon_(nk_shapes_workspace_f32_serial_t const *views,
                                                                   nk_size_t rows, nk_size_t columns,
                                                                   nk_f64_t cost_scale, nk_f32_t *rotation) {
    nk_shapes_procrustes_f32_serial_t search;
    nk_shapes_procrustes_start_f32_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_f32_serial_(&search)) {
        nk_shapes_network_simplex_f32_neon_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_f32_neon_(views, rows, columns);
    }
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f16_neon_(nk_shapes_workspace_f32_serial_t const *views,
                                                               nk_size_t rows, nk_size_t columns, nk_f64_t cost_scale,
                                                               nk_f64_t const *row_distances,
                                                               nk_f64_t const *column_distances, nk_f32_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f16_neon_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f32_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_serial_(&search);
    while (nk_shapes_gromov_next_f32_serial_(&search, views, rotation))
        nk_shapes_network_simplex_f32_neon_(views, search.gradient, rows, columns, search.gradient_scale);
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_wasserstein_procrustes_f32_neon_(nk_shapes_workspace_f64_serial_t const *views,
                                                                   nk_size_t rows, nk_size_t columns,
                                                                   nk_f64_t cost_scale, nk_f64_t *rotation) {
    nk_shapes_procrustes_f64_serial_t search;
    nk_shapes_procrustes_start_f64_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_f64_serial_(&search)) {
        nk_shapes_network_simplex_f64_neon_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_f64_neon_(views, rows, columns);
    }
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f32_neon_(nk_shapes_workspace_f64_serial_t const *views,
                                                               nk_size_t rows, nk_size_t columns, nk_f64_t cost_scale,
                                                               nk_f64_t const *row_distances,
                                                               nk_f64_t const *column_distances, nk_f64_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f32_neon_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f64_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_serial_(&search);
    while (nk_shapes_gromov_next_f64_serial_(&search, views, rotation))
        nk_shapes_network_simplex_f64_neon_(views, search.gradient, rows, columns, search.gradient_scale);
    return search.best;
}

NUMKONG_INLINE void nk_jacobi_eigenanalysis_f32_neon_(nk_f32_t *matrix, nk_f32_t *quaternion) {
    nk_jacobi_start_f32_serial_(quaternion);
    for (int sweep = 0; sweep != 16; ++sweep) {
        for (int axis = 0; axis != 3; ++axis) {
            nk_f32_t const norm = nk_jacobi_half_angle_f32_serial_(matrix);
            nk_jacobi_conjugate_f32_serial_(axis, nk_rsqrt_f32_neon_(norm), matrix, quaternion);
        }
        if (nk_jacobi_converged_f32_serial_(matrix, 0x1p-24f)) break;
    }
    nk_f32_t const length_squared = nk_jacobi_length_squared_f32_serial_(quaternion);
    nk_jacobi_rescale_f32_serial_(nk_rsqrt_f32_neon_(length_squared), quaternion);
}

NUMKONG_INLINE void nk_jacobi_eigenanalysis_f64_neon_(nk_f64_t *matrix, nk_f64_t *quaternion) {
    nk_jacobi_start_f64_serial_(quaternion);
    for (int sweep = 0; sweep != 16; ++sweep) {
        for (int axis = 0; axis != 3; ++axis) {
            nk_f64_t const norm = nk_jacobi_half_angle_f64_serial_(matrix);
            nk_jacobi_conjugate_f64_serial_(axis, nk_rsqrt_f64_neon_(norm), matrix, quaternion);
        }
        if (nk_jacobi_converged_f64_serial_(matrix, 0x1p-53)) break;
    }
    nk_f64_t const length_squared = nk_jacobi_length_squared_f64_serial_(quaternion);
    nk_jacobi_rescale_f64_serial_(nk_rsqrt_f64_neon_(length_squared), quaternion);
}

/** The next four @c f16 points of the @p left remaining, each axis in f32 lanes, with
 *  @p valid_u32x4 over the lanes holding one. Padded lanes read zero. */
NUMKONG_INLINE void nk_shapes_load_points_f16_neon_(nk_f16_t const *points, nk_size_t left, float32x4_t *x_f32x4,
                                                    float32x4_t *y_f32x4, float32x4_t *z_f32x4,
                                                    uint32x4_t *valid_u32x4) {
    nk_u16_t const *source = (nk_u16_t const *)points;
    uint16x4x3_t xyz_u16x4x3 = {{vdup_n_u16(0), vdup_n_u16(0), vdup_n_u16(0)}};
    if (left >= 4) xyz_u16x4x3 = vld3_u16(source);
    else {
        if (left > 0) xyz_u16x4x3 = vld3_lane_u16(source, xyz_u16x4x3, 0);
        if (left > 1) xyz_u16x4x3 = vld3_lane_u16(source + 3, xyz_u16x4x3, 1);
        if (left > 2) xyz_u16x4x3 = vld3_lane_u16(source + 6, xyz_u16x4x3, 2);
    }
    *x_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(xyz_u16x4x3.val[0]));
    *y_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(xyz_u16x4x3.val[1]));
    *z_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(xyz_u16x4x3.val[2]));
    *valid_u32x4 = vcltq_u32(vcombine_u32(vcreate_u32(0x0000000100000000ULL), vcreate_u32(0x0000000300000002ULL)),
                             vdupq_n_u32(left < 4 ? (nk_u32_t)left : 4));
}

/** The moments of a cloud of @c f16 points as @ref nk_shapes_moments_f16_serial_ defines them, four
 *  points per step. */
NUMKONG_INLINE nk_f32_t nk_shapes_moments_f16_neon_(nk_f16_t const *points, nk_size_t count, nk_f32_t *centroid,
                                                    nk_f32_t *covariance) {
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    float32x4_t x_f32x4, y_f32x4, z_f32x4;
    uint32x4_t valid_u32x4;
    nk_shapes_load_points_f16_neon_(points, 1, &x_f32x4, &y_f32x4, &z_f32x4, &valid_u32x4);
    float32x4_t const first_x_f32x4 = vdupq_laneq_f32(x_f32x4, 0), first_y_f32x4 = vdupq_laneq_f32(y_f32x4, 0);
    float32x4_t const first_z_f32x4 = vdupq_laneq_f32(z_f32x4, 0);
    float32x4_t sum_x_f32x4 = vdupq_n_f32(0), sum_y_f32x4 = sum_x_f32x4, sum_z_f32x4 = sum_x_f32x4;
    uint32x4_t coincident_u32x4 = vdupq_n_u32(~0u);
    for (nk_size_t index = 0; index < count; index += 4) {
        nk_shapes_load_points_f16_neon_(points + index * 3, count - index, &x_f32x4, &y_f32x4, &z_f32x4, &valid_u32x4);
        sum_x_f32x4 = vaddq_f32(sum_x_f32x4, x_f32x4), sum_y_f32x4 = vaddq_f32(sum_y_f32x4, y_f32x4);
        sum_z_f32x4 = vaddq_f32(sum_z_f32x4, z_f32x4);
        uint32x4_t const same_u32x4 = vandq_u32(
            vandq_u32(vceqq_f32(x_f32x4, first_x_f32x4), vceqq_f32(y_f32x4, first_y_f32x4)),
            vceqq_f32(z_f32x4, first_z_f32x4));
        coincident_u32x4 = vandq_u32(coincident_u32x4, vornq_u32(same_u32x4, valid_u32x4));
    }
    nk_f32_t const points_count = (nk_f32_t)count;
    centroid[0] = vaddvq_f32(sum_x_f32x4) / points_count, centroid[1] = vaddvq_f32(sum_y_f32x4) / points_count;
    centroid[2] = vaddvq_f32(sum_z_f32x4) / points_count;
    float32x4_t const centroid_x_f32x4 = vdupq_n_f32(centroid[0]), centroid_y_f32x4 = vdupq_n_f32(centroid[1]);
    float32x4_t const centroid_z_f32x4 = vdupq_n_f32(centroid[2]);
    float32x4_t moments_f32x4[6];
    for (int entry = 0; entry != 6; ++entry) moments_f32x4[entry] = vdupq_n_f32(0);
    for (nk_size_t index = 0; index < count; index += 4) {
        nk_shapes_load_points_f16_neon_(points + index * 3, count - index, &x_f32x4, &y_f32x4, &z_f32x4, &valid_u32x4);
        x_f32x4 = vbslq_f32(valid_u32x4, vsubq_f32(x_f32x4, centroid_x_f32x4), x_f32x4);
        y_f32x4 = vbslq_f32(valid_u32x4, vsubq_f32(y_f32x4, centroid_y_f32x4), y_f32x4);
        z_f32x4 = vbslq_f32(valid_u32x4, vsubq_f32(z_f32x4, centroid_z_f32x4), z_f32x4);
        moments_f32x4[0] = vfmaq_f32(moments_f32x4[0], x_f32x4, x_f32x4);
        moments_f32x4[1] = vfmaq_f32(moments_f32x4[1], y_f32x4, x_f32x4);
        moments_f32x4[2] = vfmaq_f32(moments_f32x4[2], y_f32x4, y_f32x4);
        moments_f32x4[3] = vfmaq_f32(moments_f32x4[3], z_f32x4, x_f32x4);
        moments_f32x4[4] = vfmaq_f32(moments_f32x4[4], z_f32x4, y_f32x4);
        moments_f32x4[5] = vfmaq_f32(moments_f32x4[5], z_f32x4, z_f32x4);
    }
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = vaddvq_f32(moments_f32x4[entry]) / points_count;
    return vminvq_u32(coincident_u32x4) ? 0 : nk_sqrt_f32_neon_(covariance[0] + covariance[2] + covariance[5]);
}

/** The pack of one cloud from its moments, as @c nk_shapes_pack_f16_serial_ lays it out, storing
 *  each step's points and measuring what was stored in one sweep, with zeros in the padding. */
NUMKONG_INLINE void nk_shapes_pack_f16_neon_(nk_f16_t const *points, nk_size_t count, nk_f32_t const *centroid,
                                             nk_f32_t const *covariance, nk_f32_t scale, void *packed) {
    nk_shapes_packed_header_f32_t *header = (nk_shapes_packed_header_f32_t *)packed;
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f32_t),
                                                                 nk_shapes_alignment_k);
    nk_size_t const axis_bytes = nk_size_round_up_to_multiple_(count * sizeof(nk_f16_t), nk_shapes_alignment_k);
    nk_u8_t *packed_bytes = (nk_u8_t *)packed;
    for (nk_size_t byte_index = 0; byte_index != header_bytes; ++byte_index) packed_bytes[byte_index] = 0;
    header->frame[3] = 1, header->scale = scale;
    if (!count) return;
    nk_size_t const axis_stride = axis_bytes / sizeof(nk_f16_t);
    nk_f16_t *packed_x = (nk_f16_t *)(packed_bytes + header_bytes);
    nk_f16_t *packed_y = packed_x + axis_stride, *packed_z = packed_y + axis_stride;
    nk_f32_t spreads[6] = {covariance[0], covariance[1], covariance[2], covariance[3], covariance[4], covariance[5]};
    nk_f32_t quaternion[4], basis[9], axes[9];
    int upper_tied, lower_tied;
    nk_jacobi_eigenanalysis_f32_neon_(spreads, quaternion);
    nk_shapes_sorted_basis_f32_serial_(spreads, quaternion, 0x1p-7f, basis, &upper_tied, &lower_tied);
    if (upper_tied && lower_tied) nk_shapes_pick_axis_f16_serial_(points, count, centroid, basis);
    else if (upper_tied || lower_tied) nk_shapes_turn_plane_f16_serial_(points, count, centroid, basis);
    nk_shapes_frame_f32_serial_(basis, upper_tied, header->frame, axes);
    for (int cell = 0; cell != 9; ++cell) axes[cell] /= scale;
    float32x4_t const axes_x_f32x4 = vld1q_f32(axes), axes_y_f32x4 = vld1q_f32(axes + 3);
    float32x4_t const axes_z_f32x4 = vcombine_f32(vld1_f32(axes + 6), vdup_n_f32(axes[8]));
    float32x4_t const centroid_x_f32x4 = vdupq_n_f32(centroid[0]), centroid_y_f32x4 = vdupq_n_f32(centroid[1]);
    float32x4_t const centroid_z_f32x4 = vdupq_n_f32(centroid[2]);
    float32x4_t moments_f32x4[6], largest_f32x4 = vdupq_n_f32(0);
    for (int entry = 0; entry != 6; ++entry) moments_f32x4[entry] = vdupq_n_f32(0);
    float32x4_t x_f32x4, y_f32x4, z_f32x4;
    uint32x4_t valid_u32x4;
    for (nk_size_t index = 0; index < count; index += 4) {
        nk_shapes_load_points_f16_neon_(points + index * 3, count - index, &x_f32x4, &y_f32x4, &z_f32x4, &valid_u32x4);
        x_f32x4 = vsubq_f32(x_f32x4, centroid_x_f32x4), y_f32x4 = vsubq_f32(y_f32x4, centroid_y_f32x4);
        z_f32x4 = vsubq_f32(z_f32x4, centroid_z_f32x4);
        float32x4_t const rotated_x_f32x4 = vfmaq_laneq_f32(
            vfmaq_laneq_f32(vmulq_laneq_f32(x_f32x4, axes_x_f32x4, 0), y_f32x4, axes_x_f32x4, 1), z_f32x4, axes_x_f32x4,
            2);
        float32x4_t const rotated_y_f32x4 = vfmaq_laneq_f32(
            vfmaq_laneq_f32(vmulq_laneq_f32(x_f32x4, axes_y_f32x4, 0), y_f32x4, axes_y_f32x4, 1), z_f32x4, axes_y_f32x4,
            2);
        float32x4_t const rotated_z_f32x4 = vfmaq_laneq_f32(
            vfmaq_laneq_f32(vmulq_laneq_f32(x_f32x4, axes_z_f32x4, 0), y_f32x4, axes_z_f32x4, 1), z_f32x4, axes_z_f32x4,
            2);
        uint32x4_t const zero_u32x4 = vdupq_n_u32(0);
        float32x4_t const zero_f32x4 = vreinterpretq_f32_u32(zero_u32x4);
        float16x4_t const stored_x_f16x4 = vcvt_f16_f32(vbslq_f32(valid_u32x4, rotated_x_f32x4, zero_f32x4));
        float16x4_t const stored_y_f16x4 = vcvt_f16_f32(vbslq_f32(valid_u32x4, rotated_y_f32x4, zero_f32x4));
        float16x4_t const stored_z_f16x4 = vcvt_f16_f32(vbslq_f32(valid_u32x4, rotated_z_f32x4, zero_f32x4));
        vst1_u16((nk_u16_t *)(packed_x + index), vreinterpret_u16_f16(stored_x_f16x4));
        vst1_u16((nk_u16_t *)(packed_y + index), vreinterpret_u16_f16(stored_y_f16x4));
        vst1_u16((nk_u16_t *)(packed_z + index), vreinterpret_u16_f16(stored_z_f16x4));
        // The spreads describe the stored points, so the sweep measures them after rounding
        x_f32x4 = vcvt_f32_f16(stored_x_f16x4), y_f32x4 = vcvt_f32_f16(stored_y_f16x4);
        z_f32x4 = vcvt_f32_f16(stored_z_f16x4);
        float32x4_t const xx_f32x4 = vmulq_f32(x_f32x4, x_f32x4), yy_f32x4 = vmulq_f32(y_f32x4, y_f32x4);
        float32x4_t const zz_f32x4 = vmulq_f32(z_f32x4, z_f32x4);
        largest_f32x4 = vmaxq_f32(largest_f32x4, vaddq_f32(vaddq_f32(xx_f32x4, yy_f32x4), zz_f32x4));
        moments_f32x4[0] = vaddq_f32(moments_f32x4[0], xx_f32x4);
        moments_f32x4[1] = vfmaq_f32(moments_f32x4[1], y_f32x4, x_f32x4);
        moments_f32x4[2] = vaddq_f32(moments_f32x4[2], yy_f32x4);
        moments_f32x4[3] = vfmaq_f32(moments_f32x4[3], z_f32x4, x_f32x4);
        moments_f32x4[4] = vfmaq_f32(moments_f32x4[4], z_f32x4, y_f32x4);
        moments_f32x4[5] = vaddq_f32(moments_f32x4[5], zz_f32x4);
    }
    nk_u16_t *padding_bits = (nk_u16_t *)packed_x;
    for (nk_size_t index = (count + 3) & ~(nk_size_t)3; index < axis_stride; ++index)
        for (int axis = 0; axis != 3; ++axis) padding_bits[axis * axis_stride + index] = 0;
    nk_f32_t moments[6];
    for (int entry = 0; entry != 6; ++entry) moments[entry] = vaddvq_f32(moments_f32x4[entry]) / (nk_f32_t)count;
    nk_jacobi_eigenanalysis_f32_neon_(moments, quaternion);
    nk_shapes_close_pack_f32_serial_(header, count, centroid, moments, nk_sqrt_f32_neon_(vmaxvq_f32(largest_f32x4)));
}

/** The next two @c f32 points of the @p left remaining, each axis in f64 lanes, with @p valid_u64x2
 *  over the lanes holding one. Padded lanes read zero. */
NUMKONG_INLINE void nk_shapes_load_points_f32_neon_(nk_f32_t const *points, nk_size_t left, float64x2_t *x_f64x2,
                                                    float64x2_t *y_f64x2, float64x2_t *z_f64x2,
                                                    uint64x2_t *valid_u64x2) {
    nk_f32_t padded[6] = {0, 0, 0, 0, 0, 0};
    nk_f32_t const *source = points;
    nk_b128_vec_t valid_vec;
    valid_vec.u64x2 = vdupq_n_u64(~(nk_u64_t)0);
    if (left < 2) {
        padded[0] = points[0], padded[1] = points[1], padded[2] = points[2];
        valid_vec.u64s[1] = 0, source = padded;
    }
    float32x2x3_t const xyz_f32x2x3 = vld3_f32(source);
    *x_f64x2 = vcvt_f64_f32(xyz_f32x2x3.val[0]), *y_f64x2 = vcvt_f64_f32(xyz_f32x2x3.val[1]);
    *z_f64x2 = vcvt_f64_f32(xyz_f32x2x3.val[2]), *valid_u64x2 = valid_vec.u64x2;
}

/** The moments of a cloud of @c f32 points as @ref nk_shapes_moments_f32_serial_ defines them, two
 *  points per step. */
NUMKONG_INLINE nk_f64_t nk_shapes_moments_f32_neon_(nk_f32_t const *points, nk_size_t count, nk_f64_t *centroid,
                                                    nk_f64_t *covariance) {
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    float64x2_t x_f64x2, y_f64x2, z_f64x2;
    uint64x2_t valid_u64x2;
    float64x2_t const first_x_f64x2 = vdupq_n_f64(points[0]), first_y_f64x2 = vdupq_n_f64(points[1]);
    float64x2_t const first_z_f64x2 = vdupq_n_f64(points[2]);
    float64x2_t sum_x_f64x2 = vdupq_n_f64(0), sum_y_f64x2 = sum_x_f64x2, sum_z_f64x2 = sum_x_f64x2;
    uint64x2_t coincident_u64x2 = vdupq_n_u64(~(nk_u64_t)0);
    for (nk_size_t index = 0; index < count; index += 2) {
        nk_shapes_load_points_f32_neon_(points + index * 3, count - index, &x_f64x2, &y_f64x2, &z_f64x2, &valid_u64x2);
        sum_x_f64x2 = vaddq_f64(sum_x_f64x2, x_f64x2), sum_y_f64x2 = vaddq_f64(sum_y_f64x2, y_f64x2);
        sum_z_f64x2 = vaddq_f64(sum_z_f64x2, z_f64x2);
        uint64x2_t const same_u64x2 = vandq_u64(
            vandq_u64(vceqq_f64(x_f64x2, first_x_f64x2), vceqq_f64(y_f64x2, first_y_f64x2)),
            vceqq_f64(z_f64x2, first_z_f64x2));
        coincident_u64x2 = vandq_u64(
            coincident_u64x2,
            vorrq_u64(same_u64x2, vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(valid_u64x2)))));
    }
    nk_f64_t const points_count = (nk_f64_t)count;
    centroid[0] = vaddvq_f64(sum_x_f64x2) / points_count, centroid[1] = vaddvq_f64(sum_y_f64x2) / points_count;
    centroid[2] = vaddvq_f64(sum_z_f64x2) / points_count;
    float64x2_t const centroid_x_f64x2 = vdupq_n_f64(centroid[0]), centroid_y_f64x2 = vdupq_n_f64(centroid[1]);
    float64x2_t const centroid_z_f64x2 = vdupq_n_f64(centroid[2]);
    float64x2_t moments_f64x2[6];
    for (int entry = 0; entry != 6; ++entry) moments_f64x2[entry] = vdupq_n_f64(0);
    for (nk_size_t index = 0; index < count; index += 2) {
        nk_shapes_load_points_f32_neon_(points + index * 3, count - index, &x_f64x2, &y_f64x2, &z_f64x2, &valid_u64x2);
        x_f64x2 = vbslq_f64(valid_u64x2, vsubq_f64(x_f64x2, centroid_x_f64x2), x_f64x2);
        y_f64x2 = vbslq_f64(valid_u64x2, vsubq_f64(y_f64x2, centroid_y_f64x2), y_f64x2);
        z_f64x2 = vbslq_f64(valid_u64x2, vsubq_f64(z_f64x2, centroid_z_f64x2), z_f64x2);
        moments_f64x2[0] = vfmaq_f64(moments_f64x2[0], x_f64x2, x_f64x2);
        moments_f64x2[1] = vfmaq_f64(moments_f64x2[1], y_f64x2, x_f64x2);
        moments_f64x2[2] = vfmaq_f64(moments_f64x2[2], y_f64x2, y_f64x2);
        moments_f64x2[3] = vfmaq_f64(moments_f64x2[3], z_f64x2, x_f64x2);
        moments_f64x2[4] = vfmaq_f64(moments_f64x2[4], z_f64x2, y_f64x2);
        moments_f64x2[5] = vfmaq_f64(moments_f64x2[5], z_f64x2, z_f64x2);
    }
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = vaddvq_f64(moments_f64x2[entry]) / points_count;
    int const coincident = vgetq_lane_u64(coincident_u64x2, 0) && vgetq_lane_u64(coincident_u64x2, 1);
    return coincident ? 0 : nk_sqrt_f64_neon_(covariance[0] + covariance[2] + covariance[5]);
}

/** @copydoc nk_shapes_pack_f16_neon_ */
NUMKONG_INLINE void nk_shapes_pack_f32_neon_(nk_f32_t const *points, nk_size_t count, nk_f64_t const *centroid,
                                             nk_f64_t const *covariance, nk_f64_t scale, void *packed) {
    nk_shapes_packed_header_f64_t *header = (nk_shapes_packed_header_f64_t *)packed;
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f64_t),
                                                                 nk_shapes_alignment_k);
    nk_size_t const axis_bytes = nk_size_round_up_to_multiple_(count * sizeof(nk_f32_t), nk_shapes_alignment_k);
    nk_u8_t *packed_bytes = (nk_u8_t *)packed;
    for (nk_size_t byte_index = 0; byte_index != header_bytes; ++byte_index) packed_bytes[byte_index] = 0;
    header->frame[3] = 1, header->scale = scale;
    if (!count) return;
    nk_size_t const axis_stride = axis_bytes / sizeof(nk_f32_t);
    nk_f32_t *packed_x = (nk_f32_t *)(packed_bytes + header_bytes);
    nk_f32_t *packed_y = packed_x + axis_stride, *packed_z = packed_y + axis_stride;
    nk_f64_t spreads[6] = {covariance[0], covariance[1], covariance[2], covariance[3], covariance[4], covariance[5]};
    nk_f64_t quaternion[4], basis[9], axes[9];
    int upper_tied, lower_tied;
    nk_jacobi_eigenanalysis_f64_neon_(spreads, quaternion);
    nk_shapes_sorted_basis_f64_serial_(spreads, quaternion, 0x1p-16, basis, &upper_tied, &lower_tied);
    if (upper_tied && lower_tied) nk_shapes_pick_axis_f32_serial_(points, count, centroid, basis);
    else if (upper_tied || lower_tied) nk_shapes_turn_plane_f32_serial_(points, count, centroid, basis);
    nk_shapes_frame_f64_serial_(basis, upper_tied, header->frame, axes);
    for (int cell = 0; cell != 9; ++cell) axes[cell] /= scale;
    float64x2_t const centroid_x_f64x2 = vdupq_n_f64(centroid[0]), centroid_y_f64x2 = vdupq_n_f64(centroid[1]);
    float64x2_t const centroid_z_f64x2 = vdupq_n_f64(centroid[2]);
    float64x2_t moments_f64x2[6], largest_f64x2 = vdupq_n_f64(0);
    for (int entry = 0; entry != 6; ++entry) moments_f64x2[entry] = vdupq_n_f64(0);
    float64x2_t x_f64x2, y_f64x2, z_f64x2;
    uint64x2_t valid_u64x2;
    for (nk_size_t index = 0; index < count; index += 2) {
        nk_shapes_load_points_f32_neon_(points + index * 3, count - index, &x_f64x2, &y_f64x2, &z_f64x2, &valid_u64x2);
        x_f64x2 = vsubq_f64(x_f64x2, centroid_x_f64x2), y_f64x2 = vsubq_f64(y_f64x2, centroid_y_f64x2);
        z_f64x2 = vsubq_f64(z_f64x2, centroid_z_f64x2);
        float64x2_t rotated_f64x2[3];
        for (int axis = 0; axis != 3; ++axis) {
            float64x2_t const unmasked_f64x2 = vfmaq_n_f64(
                vfmaq_n_f64(vmulq_n_f64(x_f64x2, axes[axis * 3]), y_f64x2, axes[axis * 3 + 1]), z_f64x2,
                axes[axis * 3 + 2]);
            rotated_f64x2[axis] = vbslq_f64(valid_u64x2, unmasked_f64x2, vdupq_n_f64(0));
        }
        float32x2_t const stored_x_f32x2 = vcvt_f32_f64(rotated_f64x2[0]);
        float32x2_t const stored_y_f32x2 = vcvt_f32_f64(rotated_f64x2[1]);
        float32x2_t const stored_z_f32x2 = vcvt_f32_f64(rotated_f64x2[2]);
        vst1_f32(packed_x + index, stored_x_f32x2), vst1_f32(packed_y + index, stored_y_f32x2);
        vst1_f32(packed_z + index, stored_z_f32x2);
        // The spreads describe the stored points, so the sweep measures them after rounding
        x_f64x2 = vcvt_f64_f32(stored_x_f32x2), y_f64x2 = vcvt_f64_f32(stored_y_f32x2);
        z_f64x2 = vcvt_f64_f32(stored_z_f32x2);
        float64x2_t const xx_f64x2 = vmulq_f64(x_f64x2, x_f64x2), yy_f64x2 = vmulq_f64(y_f64x2, y_f64x2);
        float64x2_t const zz_f64x2 = vmulq_f64(z_f64x2, z_f64x2);
        largest_f64x2 = vmaxq_f64(largest_f64x2, vaddq_f64(vaddq_f64(xx_f64x2, yy_f64x2), zz_f64x2));
        moments_f64x2[0] = vaddq_f64(moments_f64x2[0], xx_f64x2);
        moments_f64x2[1] = vfmaq_f64(moments_f64x2[1], y_f64x2, x_f64x2);
        moments_f64x2[2] = vaddq_f64(moments_f64x2[2], yy_f64x2);
        moments_f64x2[3] = vfmaq_f64(moments_f64x2[3], z_f64x2, x_f64x2);
        moments_f64x2[4] = vfmaq_f64(moments_f64x2[4], z_f64x2, y_f64x2);
        moments_f64x2[5] = vaddq_f64(moments_f64x2[5], zz_f64x2);
    }
    for (nk_size_t index = (count + 1) & ~(nk_size_t)1; index < axis_stride; ++index)
        for (int axis = 0; axis != 3; ++axis) packed_x[axis * axis_stride + index] = 0;
    nk_f64_t moments[6];
    for (int entry = 0; entry != 6; ++entry) moments[entry] = vaddvq_f64(moments_f64x2[entry]) / (nk_f64_t)count;
    nk_jacobi_eigenanalysis_f64_neon_(moments, quaternion);
    nk_shapes_close_pack_f64_serial_(header, count, centroid, moments, nk_sqrt_f64_neon_(vmaxvq_f64(largest_f64x2)));
}

NUMKONG_API nk_status_t nk_shapes_pack_f16_neon(nk_f16_t const *points, nk_size_t count, void *packed,
                                                nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t centroid[3], covariance[6];
    nk_shapes_moments_f16_neon_(points, count, centroid, covariance);
    nk_shapes_pack_f16_neon_(points, count, centroid, covariance, 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f16_neon(nk_f16_t const *points, nk_size_t count, void *packed,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t centroid[3], covariance[6];
    nk_f32_t const radius = nk_shapes_moments_f16_neon_(points, count, centroid, covariance);
    nk_shapes_pack_f16_neon_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_f32_neon(nk_f32_t const *points, nk_size_t count, void *packed,
                                                nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t centroid[3], covariance[6];
    nk_shapes_moments_f32_neon_(points, count, centroid, covariance);
    nk_shapes_pack_f32_neon_(points, count, centroid, covariance, 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f32_neon(nk_f32_t const *points, nk_size_t count, void *packed,
                                                           nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t centroid[3], covariance[6];
    nk_f64_t const radius = nk_shapes_moments_f32_neon_(points, count, centroid, covariance);
    nk_shapes_pack_f32_neon_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f16_neon(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                    nk_size_t results_stride, nk_f32_t *rotations,
                                                    nk_f32_t const *limit, void *workspace, nk_size_t workspace_bytes,
                                                    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t const pair_limit = *limit;
    nk_f32_t chamfers[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_status_t const status = nk_shapes_chamfer_flips_f16_neon_(a_packed[row], b_packed[column], pair_limit,
                                                                         workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f32_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_neon(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_f32_t const *limit, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t const pair_limit = *limit;
    nk_f32_t hausdorffs[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_hausdorff_flips_f16_neon_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f32_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f32_neon(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                    nk_size_t results_stride, nk_f32_t *rotations,
                                                    nk_f64_t const *limit, void *workspace, nk_size_t workspace_bytes,
                                                    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t const pair_limit = *limit;
    nk_f64_t chamfers[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_status_t const status = nk_shapes_chamfer_flips_f32_neon_(a_packed[row], b_packed[column], pair_limit,
                                                                         workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f64_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_neon(void const *const *a_packed, nk_size_t rows,
                                                      void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                      nk_size_t results_stride, nk_f32_t *rotations,
                                                      nk_f64_t const *limit, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t const pair_limit = *limit;
    nk_f64_t hausdorffs[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_hausdorff_flips_f32_neon_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f64_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_neon(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f32_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_shapes_workspace_f32_serial_t views;
    nk_status_t const status = nk_shapes_transport_views_f32_serial_(a_packed, rows, b_packed, columns, workspace,
                                                                     workspace_bytes, 0, &views);
    if (status != nk_success_k) return status;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_pair_f32_serial_t pair;
            if (nk_shapes_pair_open_f16_serial_(&pair, a_packed[row], b_packed[column], &views))
                results_row[column] = nk_sqrt_f32_serial_(nk_shapes_wasserstein_procrustes_f16_neon_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F32_INF;
            if (rotations) nk_shapes_pair_rotation_f32_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_neon(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f32_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_shapes_workspace_f32_serial_t views;
    nk_status_t const status = nk_shapes_transport_views_f32_serial_(a_packed, rows, b_packed, columns, workspace,
                                                                     workspace_bytes, 1, &views);
    if (status != nk_success_k) return status;
    nk_f64_t *cached_distances = views.matrices + 3 * views.matrix_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
        nk_size_t const cached_points = (nk_size_t)((nk_shapes_packed_header_f32_t const *)a_packed[row])->points;
        nk_f64_t *pair_distances = cached_distances + cached_points * cached_points;
        nk_shapes_widen_f16_serial_(a_packed[row], views.row_points, views.points_stride);
        nk_shapes_distances_f32_serial_(views.row_points, cached_points, views.points_stride, cached_distances);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_pair_f32_serial_t pair;
            if (nk_shapes_pair_open_f16_serial_(&pair, a_packed[row], b_packed[column], &views)) {
                // The cached matrix is this row's cloud, on whichever side it is solved
                nk_shapes_distances_f32_serial_(pair.swapped ? views.row_points : views.column_points,
                                                pair.swapped ? pair.rows : pair.columns, views.points_stride,
                                                pair_distances);
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f16_neon_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.swapped ? pair_distances : cached_distances,
                    pair.swapped ? cached_distances : pair_distances, pair.inner);
                results_row[column] = (nk_f32_t)(nk_sqrt_f64_serial_(loss) / 2);
            }
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F32_INF;
            if (rotations) nk_shapes_pair_rotation_f32_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_neon(void const *const *a_packed, nk_size_t rows,
                                                                 void const *const *b_packed, nk_size_t columns,
                                                                 nk_f64_t *results, nk_size_t results_stride,
                                                                 nk_f32_t *rotations, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_shapes_workspace_f64_serial_t views;
    nk_status_t const status = nk_shapes_transport_views_f64_serial_(a_packed, rows, b_packed, columns, workspace,
                                                                     workspace_bytes, 0, &views);
    if (status != nk_success_k) return status;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_pair_f64_serial_t pair;
            if (nk_shapes_pair_open_f32_serial_(&pair, a_packed[row], b_packed[column], &views))
                results_row[column] = nk_sqrt_f64_serial_(nk_shapes_wasserstein_procrustes_f32_neon_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F64_INF;
            if (rotations) nk_shapes_pair_rotation_f64_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_neon(void const *const *a_packed, nk_size_t rows,
                                                              void const *const *b_packed, nk_size_t columns,
                                                              nk_f64_t *results, nk_size_t results_stride,
                                                              nk_f32_t *rotations, void *workspace,
                                                              nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_shapes_workspace_f64_serial_t views;
    nk_status_t const status = nk_shapes_transport_views_f64_serial_(a_packed, rows, b_packed, columns, workspace,
                                                                     workspace_bytes, 1, &views);
    if (status != nk_success_k) return status;
    nk_f64_t *cached_distances = views.matrices + 3 * views.matrix_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        nk_size_t const cached_points = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed[row])->points;
        nk_f64_t *pair_distances = cached_distances + cached_points * cached_points;
        nk_shapes_widen_f32_serial_(a_packed[row], views.row_points, views.points_stride);
        nk_shapes_distances_f64_serial_(views.row_points, cached_points, views.points_stride, cached_distances);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_pair_f64_serial_t pair;
            if (nk_shapes_pair_open_f32_serial_(&pair, a_packed[row], b_packed[column], &views)) {
                // The cached matrix is this row's cloud, on whichever side it is solved
                nk_shapes_distances_f64_serial_(pair.swapped ? views.row_points : views.column_points,
                                                pair.swapped ? pair.rows : pair.columns, views.points_stride,
                                                pair_distances);
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f32_neon_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.swapped ? pair_distances : cached_distances,
                    pair.swapped ? cached_distances : pair_distances, pair.inner);
                results_row[column] = (nk_f64_t)(nk_sqrt_f64_serial_(loss) / 2);
            }
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F64_INF;
            if (rotations) nk_shapes_pair_rotation_f64_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

/** Σ‖xᵢ′‖² of a pack of @c f16 points, 4 points per step through the zero padding. */
NUMKONG_INLINE nk_f32_t nk_rmsds_norm_f16_neon_(void const *packed) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)packed)->points;
    nk_size_t axis_stride;
    nk_f16_t const *axes = nk_shapes_packed_axes_f16_serial_(packed, &axis_stride);
    float32x4_t norm_f32x4 = vdupq_n_f32(0);
    for (int axis = 0; axis != 3; ++axis)
        for (nk_size_t index = 0; index < count; index += 4) {
            float32x4_t const values_f32x4 = nk_load_f16x4_as_f32_neon_(axes + axis * axis_stride + index);
            norm_f32x4 = vfmaq_f32(norm_f32x4, values_f32x4, values_f32x4);
        }
    return vaddvq_f32(norm_f32x4);
}

/** Superposed RMSD of one pair of @c f16 packs with Ga + Gb in @p norms_squared, M in @c f32. */
NUMKONG_INLINE nk_f32_t nk_rmsds_pair_f16_neon_(void const *row_packed, void const *column_packed,
                                                nk_f32_t norms_squared) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f16_t const *a_axes = nk_shapes_packed_axes_f16_serial_(row_packed, &axis_stride);
    nk_f16_t const *b_axes = nk_shapes_packed_axes_f16_serial_(column_packed, &axis_stride);
    float32x4_t sums_f32x4[9], a_f32x4[3], b_f32x4[3];
    for (int cell = 0; cell != 9; ++cell) sums_f32x4[cell] = vdupq_n_f32(0);
    for (nk_size_t index = 0; index < count; index += 4) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f32x4[axis] = nk_load_f16x4_as_f32_neon_(a_axes + axis * axis_stride + index);
            b_f32x4[axis] = nk_load_f16x4_as_f32_neon_(b_axes + axis * axis_stride + index);
        }
        for (int row = 0; row != 3; ++row)
            for (int column = 0; column != 3; ++column)
                sums_f32x4[row * 3 + column] = vfmaq_f32(sums_f32x4[row * 3 + column], a_f32x4[row], b_f32x4[column]);
    }
    nk_f32_t cross_covariance[9];
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = vaddvq_f32(sums_f32x4[cell]);
    nk_f32_t const residual = nk_rmsds_folded_f32_serial_(cross_covariance, norms_squared);
    return nk_sqrt_f32_neon_(residual / (nk_f32_t)count);
}

/** Superposed RMSD of one pair of @c f32 packs: H in @c f64, then the residuals in plain @c f64,
 *  2 points per step through the zero padding. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f32_neon_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f32_t const *a_axes = nk_shapes_packed_axes_f32_serial_(row_packed, &axis_stride);
    nk_f32_t const *b_axes = nk_shapes_packed_axes_f32_serial_(column_packed, &axis_stride);
    float64x2_t sums_f64x2[9], a_f64x2[3], b_f64x2[3];
    for (int cell = 0; cell != 9; ++cell) sums_f64x2[cell] = vdupq_n_f64(0);
    for (nk_size_t index = 0; index < count; index += 2) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f64x2[axis] = vcvt_f64_f32(vld1_f32(a_axes + axis * axis_stride + index));
            b_f64x2[axis] = vcvt_f64_f32(vld1_f32(b_axes + axis * axis_stride + index));
        }
        for (int column = 0; column != 3; ++column)
            for (int row = 0; row != 3; ++row)
                sums_f64x2[column * 3 + row] = vfmaq_f64(sums_f64x2[column * 3 + row], b_f64x2[column], a_f64x2[row]);
    }
    nk_f64_t cross_covariance[9], rotation[9];
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = vaddvq_f64(sums_f64x2[cell]);
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    float64x2_t rotation_f64x2[9];
    for (int cell = 0; cell != 9; ++cell) rotation_f64x2[cell] = vdupq_n_f64(rotation[cell]);
    float64x2_t sum_f64x2 = vdupq_n_f64(0);
    for (nk_size_t index = 0; index < count; index += 2) {
        for (int axis = 0; axis != 3; ++axis)
            b_f64x2[axis] = vcvt_f64_f32(vld1_f32(b_axes + axis * axis_stride + index));
        for (int axis = 0; axis != 3; ++axis) {
            float64x2_t residual_f64x2 = vcvt_f64_f32(vld1_f32(a_axes + axis * axis_stride + index));
            for (int column = 0; column != 3; ++column)
                residual_f64x2 = vfmsq_f64(residual_f64x2, rotation_f64x2[axis * 3 + column], b_f64x2[column]);
            sum_f64x2 = vfmaq_f64(sum_f64x2, residual_f64x2, residual_f64x2);
        }
    }
    return nk_sqrt_f64_neon_(vaddvq_f64(sum_f64x2) / (nk_f64_t)count);
}

/** Superposed RMSD of one pair of @c f64 packs, like the @c f32 pair with H and the residuals
 *  compensated, 2 points per step through the zero padding. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f64_neon_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f64_t const *a_axes = nk_shapes_packed_axes_f64_serial_(row_packed, &axis_stride);
    nk_f64_t const *b_axes = nk_shapes_packed_axes_f64_serial_(column_packed, &axis_stride);
    float64x2_t sums_f64x2[9], compensations_f64x2[9], a_f64x2[3], b_f64x2[3];
    for (int cell = 0; cell != 9; ++cell) sums_f64x2[cell] = vdupq_n_f64(0), compensations_f64x2[cell] = vdupq_n_f64(0);
    for (nk_size_t index = 0; index < count; index += 2) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f64x2[axis] = vld1q_f64(a_axes + axis * axis_stride + index);
            b_f64x2[axis] = vld1q_f64(b_axes + axis * axis_stride + index);
        }
        for (int column = 0; column != 3; ++column)
            for (int row = 0; row != 3; ++row)
                nk_dot2_f64x2_neon_(&sums_f64x2[column * 3 + row], &compensations_f64x2[column * 3 + row],
                                    b_f64x2[column], a_f64x2[row]);
    }
    nk_f64_t cross_covariance[9], rotation[9];
    for (int cell = 0; cell != 9; ++cell)
        cross_covariance[cell] = nk_dot_stable_sum_f64x2_neon_(sums_f64x2[cell], compensations_f64x2[cell]);
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    float64x2_t rotation_f64x2[9];
    for (int cell = 0; cell != 9; ++cell) rotation_f64x2[cell] = vdupq_n_f64(rotation[cell]);
    float64x2_t sum_f64x2 = vdupq_n_f64(0), compensation_f64x2 = vdupq_n_f64(0);
    for (nk_size_t index = 0; index < count; index += 2) {
        for (int axis = 0; axis != 3; ++axis) b_f64x2[axis] = vld1q_f64(b_axes + axis * axis_stride + index);
        for (int axis = 0; axis != 3; ++axis) {
            float64x2_t residual_f64x2 = vld1q_f64(a_axes + axis * axis_stride + index);
            for (int column = 0; column != 3; ++column)
                residual_f64x2 = vfmsq_f64(residual_f64x2, rotation_f64x2[axis * 3 + column], b_f64x2[column]);
            nk_dot2_f64x2_neon_(&sum_f64x2, &compensation_f64x2, residual_f64x2, residual_f64x2);
        }
    }
    return nk_sqrt_f64_neon_(nk_dot_stable_sum_f64x2_neon_(sum_f64x2, compensation_f64x2) / (nk_f64_t)count);
}

NUMKONG_API nk_status_t nk_rmsds_packed_f64_neon(void const *const *a_packed, nk_size_t rows,
                                                 void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                 nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_size_t const points = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed[0])->points;
    if (!nk_rmsds_same_points_f64_serial_(a_packed, rows, points) ||
        !nk_rmsds_same_points_f64_serial_(b_packed, columns, points))
        return nk_unexpected_dimensions_k;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column)
            results_row[column] = nk_rmsds_pair_f64_neon_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f32_neon(void const *const *a_packed, nk_size_t rows,
                                                 void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                 nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_size_t const points = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed[0])->points;
    if (!nk_rmsds_same_points_f32_serial_(a_packed, rows, points) ||
        !nk_rmsds_same_points_f32_serial_(b_packed, columns, points))
        return nk_unexpected_dimensions_k;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column)
            results_row[column] = nk_rmsds_pair_f32_neon_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f16_neon(void const *const *a_packed, nk_size_t rows,
                                                 void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                 nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_size_t const points = ((nk_shapes_packed_header_f32_t const *)a_packed[0])->points;
    if (!nk_rmsds_same_points_f16_serial_(a_packed, rows, points) ||
        !nk_rmsds_same_points_f16_serial_(b_packed, columns, points))
        return nk_unexpected_dimensions_k;
    nk_f32_t row_norms[nk_rmsds_tile_frames_k], column_norms[nk_rmsds_tile_frames_k];
    for (nk_size_t row_begin = 0; row_begin < rows; row_begin += nk_rmsds_tile_frames_k) {
        nk_size_t const row_end = row_begin + nk_rmsds_tile_frames_k < rows ? row_begin + nk_rmsds_tile_frames_k : rows;
        for (nk_size_t row = row_begin; row != row_end; ++row)
            row_norms[row - row_begin] = nk_rmsds_norm_f16_neon_(a_packed[row]);
        for (nk_size_t column_begin = 0; column_begin < columns; column_begin += nk_rmsds_tile_frames_k) {
            nk_size_t const column_end = column_begin + nk_rmsds_tile_frames_k < columns
                                             ? column_begin + nk_rmsds_tile_frames_k
                                             : columns;
            for (nk_size_t column = column_begin; column != column_end; ++column)
                column_norms[column - column_begin] = nk_rmsds_norm_f16_neon_(b_packed[column]);
            for (nk_size_t row = row_begin; row != row_end; ++row) {
                nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
                for (nk_size_t column = column_begin; column != column_end; ++column)
                    results_row[column] = nk_rmsds_pair_f16_neon_(
                        a_packed[row], b_packed[column],
                        row_norms[row - row_begin] + column_norms[column - column_begin]);
            }
        }
    }
    return nk_success_k;
}

#endif // NUMKONG_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_ARM64_NEON_
#endif // NUMKONG_ARCH_ARM64_
#endif // NUMKONG_SHAPES_NEON_H
