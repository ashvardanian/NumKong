/**
 *  @file include/numkong/shapes/powervsx.h
 *  @author Ash Vardanian
 *  @date November 13, 2026
 *  @brief Many-to-many shape distances between 3D point clouds for Power VSX.
 *
 *  @sa c/numkong/shapes.h
 *
 *  @section shapes_powervsx_instructions Key Power VSX Shape Instructions
 *
 *  @verbatim
 *  Intrinsic                      Instruction   Operation
 *  vec_madd                       XVMADDADP     Fused multiply-add of f64 lanes
 *  vec_min                        XVMINDP       Running minima of the flips
 *  vec_sqrt                       XVSQRTDP      Roots of the nearest squares
 *  vec_doubleh                    XVCVSPDP      Two f32 points widened into f64 lanes
 *  vec_extract_fp32_from_shorth   XVCVHPSP      Four f16 points widened into f32 lanes
 *  vec_pack_to_short_fp32         XVCVSPHP      Packed points narrowed into f16
 *  vec_perm                       VPERM         Interleaved points gathered per axis
 *  vec_xl_len                     LXVL          Loads limited to the points left
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
 *  Packing gathers the interleaved @b [x,y,z] points of each step from two length-limited loads
 *  with @c vec_perm. RMSD sums four @c f16 points per step in f32 lanes, or two @c f32 or @c f64
 *  points in f64 lanes, through the zero padding, and leaves the SVD scalar.
 */
#ifndef NUMKONG_SHAPES_POWERVSX_H
#define NUMKONG_SHAPES_POWERVSX_H

#if NUMKONG_ARCH_PPC64_
#if NUMKONG_ARCH_PPC64_POWERVSX_

#include "numkong/types.h"
#include "numkong/cast/powervsx.h"   // `nk_load_f16x4_as_f32_powervsx_`
#include "numkong/scalar/powervsx.h" // `nk_sqrt_f32_powervsx_`, `nk_sqrt_f64_powervsx_`
#include "numkong/shapes/serial.h"   // `nk_shapes_best_flip_f32_serial_`, `nk_shapes_simplex_next_tile_serial_`
#include "numkong/dot/powervsx.h"    // `nk_dot2_f64x2_powervsx_`, `nk_hsum_f32x4_powervsx_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/** Lowers the running minima of the 8 flips by squared distances to four inner points, given each
 *  axis' difference and sum of the outer and inner coordinates. */
NUMKONG_INLINE void nk_shapes_lower_nearest_differences_f32_powervsx_(
    nk_vf32x4_t minus_x_f32x4, nk_vf32x4_t plus_x_f32x4, nk_vf32x4_t minus_y_f32x4, nk_vf32x4_t plus_y_f32x4,
    nk_vf32x4_t minus_z_f32x4, nk_vf32x4_t plus_z_f32x4, nk_vf32x4_t *nearest_f32x4) {
    nk_vf32x4_t const z_f32x4[2] = {minus_z_f32x4, plus_z_f32x4};
    nk_vf32x4_t const minus_xx_f32x4 = vec_mul(minus_x_f32x4, minus_x_f32x4);
    nk_vf32x4_t const plus_xx_f32x4 = vec_mul(plus_x_f32x4, plus_x_f32x4);
    nk_vf32x4_t const planar_f32x4[4] = {
        vec_madd(minus_y_f32x4, minus_y_f32x4, minus_xx_f32x4), vec_madd(minus_y_f32x4, minus_y_f32x4, plus_xx_f32x4),
        vec_madd(plus_y_f32x4, plus_y_f32x4, minus_xx_f32x4), vec_madd(plus_y_f32x4, plus_y_f32x4, plus_xx_f32x4)};
    for (int flip = 0; flip != 8; ++flip)
        nearest_f32x4[flip] = vec_min(nearest_f32x4[flip],
                                      vec_madd(z_f32x4[flip >> 2], z_f32x4[flip >> 2], planar_f32x4[flip & 3]));
}

/** Lowers the 8 flips' nearest squares as @ref nk_shapes_lower_nearest_differences_f32_powervsx_
 *  does, and the same squares into the inner points' @p reverse_minima , 8 axes @p reverse_stride
 *  apart in the workspace. */
NUMKONG_INLINE void nk_shapes_lower_mutual_differences_f32_powervsx_(
    nk_vf32x4_t minus_x_f32x4, nk_vf32x4_t plus_x_f32x4, nk_vf32x4_t minus_y_f32x4, nk_vf32x4_t plus_y_f32x4,
    nk_vf32x4_t minus_z_f32x4, nk_vf32x4_t plus_z_f32x4, nk_vf32x4_t *nearest_f32x4, nk_f32_t *reverse_minima,
    nk_size_t reverse_stride) {
    nk_vf32x4_t const z_f32x4[2] = {minus_z_f32x4, plus_z_f32x4};
    nk_vf32x4_t const minus_xx_f32x4 = vec_mul(minus_x_f32x4, minus_x_f32x4);
    nk_vf32x4_t const plus_xx_f32x4 = vec_mul(plus_x_f32x4, plus_x_f32x4);
    nk_vf32x4_t const planar_f32x4[4] = {
        vec_madd(minus_y_f32x4, minus_y_f32x4, minus_xx_f32x4), vec_madd(minus_y_f32x4, minus_y_f32x4, plus_xx_f32x4),
        vec_madd(plus_y_f32x4, plus_y_f32x4, minus_xx_f32x4), vec_madd(plus_y_f32x4, plus_y_f32x4, plus_xx_f32x4)};
    for (int flip = 0; flip != 8; ++flip) {
        nk_vf32x4_t const squares_f32x4 = vec_madd(z_f32x4[flip >> 2], z_f32x4[flip >> 2], planar_f32x4[flip & 3]);
        nk_f32_t *reverse = reverse_minima + flip * reverse_stride;
        nearest_f32x4[flip] = vec_min(nearest_f32x4[flip], squares_f32x4);
        vec_xst(vec_min(vec_xl(0, reverse), squares_f32x4), 0, reverse);
    }
}

/** The least of every flip's nearest squares over its lanes, flips 0 to 3 in the first vector and
 *  4 to 7 in the second. */
NUMKONG_INLINE void nk_shapes_least_flips_f32_powervsx_(nk_vf32x4_t const *nearest_f32x4, nk_vf32x4_t *least_f32x4) {
    for (int half = 0; half != 2; ++half) {
        nk_vf32x4_t const *flips_f32x4 = nearest_f32x4 + 4 * half;
        // Each merge interleaves two flips' lanes, so each minimum halves the lanes left per flip
        nk_vf32x4_t const first_f32x4 = vec_min(vec_mergeh(flips_f32x4[0], flips_f32x4[1]),
                                                vec_mergel(flips_f32x4[0], flips_f32x4[1]));
        nk_vf32x4_t const second_f32x4 = vec_min(vec_mergeh(flips_f32x4[2], flips_f32x4[3]),
                                                 vec_mergel(flips_f32x4[2], flips_f32x4[3]));
        least_f32x4[half] = vec_min((nk_vf32x4_t)vec_mergeh((nk_vu64x2_t)first_f32x4, (nk_vu64x2_t)second_f32x4),
                                    (nk_vf32x4_t)vec_mergel((nk_vu64x2_t)first_f32x4, (nk_vu64x2_t)second_f32x4));
    }
}

/** The roots of @p count reverse minima summed, in lanes past which the axis' padding stays out. */
NUMKONG_INLINE nk_f32_t nk_shapes_reverse_sum_f32_powervsx_(nk_f32_t const *reverse, nk_size_t count) {
    nk_vf32x4_t sum_f32x4 = vec_splats(0.0f);
    nk_size_t index = 0;
    for (; index + 4 <= count; index += 4) sum_f32x4 = vec_add(sum_f32x4, vec_sqrt(vec_xl(0, reverse + index)));
    nk_f32_t sum = nk_hsum_f32x4_powervsx_(sum_f32x4);
    for (; index != count; ++index) sum += nk_sqrt_f32_powervsx_(reverse[index]);
    return sum;
}

/** The mask over the lanes of the last quadruple of @p inner_count points. */
NUMKONG_INLINE nk_b128_vec_t nk_shapes_tail_powervsx_(nk_size_t inner_count) {
    nk_vu32x4_t const lanes_u32x4 = {0, 1, 2, 3};
    nk_b128_vec_t tail_vec;
    tail_vec.vu32x4 = (nk_vu32x4_t)vec_cmplt(lanes_u32x4, vec_splats((nk_u32_t)(inner_count & 3)));
    return tail_vec;
}

/** Adds the roots of one outer point's @p nearest_f32x4 onto the 8 @p sums_f32x4 , and whether
 *  every flip's finished rows pass @p limit , leaving those lower bounds in @p chamfers . */
NUMKONG_INLINE int nk_shapes_chamfer_row_f32_powervsx_(nk_vf32x4_t const *nearest_f32x4, nk_vf32x4_t *sums_f32x4,
                                                       nk_vf32x4_t threshold_f32x4, nk_f32_t outer_points,
                                                       nk_f32_t limit, nk_f32_t *chamfers) {
    nk_vf32x4_t least_f32x4[2];
    nk_shapes_least_flips_f32_powervsx_(nearest_f32x4, least_f32x4);
    sums_f32x4[0] = vec_add(sums_f32x4[0], vec_sqrt(least_f32x4[0]));
    sums_f32x4[1] = vec_add(sums_f32x4[1], vec_sqrt(least_f32x4[1]));
    if (!vec_all_gt(sums_f32x4[0], threshold_f32x4) || !vec_all_gt(sums_f32x4[1], threshold_f32x4)) return 0;
    nk_f32_t sums[8];
    vec_xst(sums_f32x4[0], 0, sums), vec_xst(sums_f32x4[1], 0, sums + 4);
    return nk_shapes_mutual_past_limit_f32_serial_(sums, outer_points, limit, chamfers);
}

/** The 8 @p chamfers of a whole fused pair, from its rows' @p sums_f32x4 and the reverse minima. */
NUMKONG_INLINE void nk_shapes_chamfer_finish_f16_powervsx_(nk_vf32x4_t const *sums_f32x4,
                                                           nk_shapes_mutual_f16_t const *views, nk_f32_t *chamfers) {
    nk_f32_t const outer_points = (nk_f32_t)views->outer_count, inner_points = (nk_f32_t)views->inner_count;
    nk_f32_t sums[8];
    vec_xst(sums_f32x4[0], 0, sums), vec_xst(sums_f32x4[1], 0, sums + 4);
    for (int flip = 0; flip != 8; ++flip) {
        nk_f32_t const reverse_sum = nk_shapes_reverse_sum_f32_powervsx_(
            views->reverse_minima + flip * views->reverse_stride, views->inner_count);
        chamfers[flip] = (sums[flip] / outer_points + reverse_sum / inner_points) / 2;
    }
}

/** Raises the 8 @p farthest_f32x4 by one outer point's @p nearest_f32x4 , and whether every flip's
 *  farthest square passes @p limit , whose square rounds, so the distance confirms it. */
NUMKONG_INLINE int nk_shapes_hausdorff_row_f32_powervsx_(nk_vf32x4_t const *nearest_f32x4, nk_vf32x4_t *farthest_f32x4,
                                                         nk_f32_t limit, nk_f32_t limit_squared) {
    nk_vf32x4_t least_f32x4[2];
    nk_shapes_least_flips_f32_powervsx_(nearest_f32x4, least_f32x4);
    farthest_f32x4[0] = vec_max(farthest_f32x4[0], least_f32x4[0]);
    farthest_f32x4[1] = vec_max(farthest_f32x4[1], least_f32x4[1]);
    nk_vf32x4_t overall_least_f32x4 = vec_min(farthest_f32x4[0], farthest_f32x4[1]);
    overall_least_f32x4 = vec_min(overall_least_f32x4, vec_sld(overall_least_f32x4, overall_least_f32x4, 8));
    overall_least_f32x4 = vec_min(overall_least_f32x4, vec_sld(overall_least_f32x4, overall_least_f32x4, 4));
    nk_f32_t const least = vec_extract(overall_least_f32x4, 0);
    return least > limit_squared && nk_sqrt_f32_powervsx_(least) > limit;
}

#if NUMKONG_TARGET_POWERVSX

/** Lowers the running minima of the 8 flips by one outer point's squared distances to two inner
 *  points, the difference and the sum of each axis standing for its two signs. */
NUMKONG_INLINE void nk_shapes_lower_nearest_f64_powervsx_(nk_vf64x2_t outer_x_f64x2, nk_vf64x2_t outer_y_f64x2,
                                                          nk_vf64x2_t outer_z_f64x2, nk_vf64x2_t inner_x_f64x2,
                                                          nk_vf64x2_t inner_y_f64x2, nk_vf64x2_t inner_z_f64x2,
                                                          nk_vf64x2_t *nearest_f64x2) {
    nk_vf64x2_t const minus_x_f64x2 = vec_sub(outer_x_f64x2, inner_x_f64x2);
    nk_vf64x2_t const plus_x_f64x2 = vec_add(outer_x_f64x2, inner_x_f64x2);
    nk_vf64x2_t const minus_y_f64x2 = vec_sub(outer_y_f64x2, inner_y_f64x2);
    nk_vf64x2_t const plus_y_f64x2 = vec_add(outer_y_f64x2, inner_y_f64x2);
    nk_vf64x2_t const z_f64x2[2] = {vec_sub(outer_z_f64x2, inner_z_f64x2), vec_add(outer_z_f64x2, inner_z_f64x2)};
    nk_vf64x2_t const minus_xx_f64x2 = vec_mul(minus_x_f64x2, minus_x_f64x2);
    nk_vf64x2_t const plus_xx_f64x2 = vec_mul(plus_x_f64x2, plus_x_f64x2);
    nk_vf64x2_t const planar_f64x2[4] = {
        vec_madd(minus_y_f64x2, minus_y_f64x2, minus_xx_f64x2), vec_madd(minus_y_f64x2, minus_y_f64x2, plus_xx_f64x2),
        vec_madd(plus_y_f64x2, plus_y_f64x2, minus_xx_f64x2), vec_madd(plus_y_f64x2, plus_y_f64x2, plus_xx_f64x2)};
    for (int flip = 0; flip != 8; ++flip)
        nearest_f64x2[flip] = vec_min(nearest_f64x2[flip],
                                      vec_madd(z_f64x2[flip >> 2], z_f64x2[flip >> 2], planar_f64x2[flip & 3]));
}

/** Lowers the 8 flips' nearest squares as @ref nk_shapes_lower_nearest_f64_powervsx_ does, and the
 *  same squares into the inner points' @p reverse_minima , 8 axes @p reverse_stride apart. */
NUMKONG_INLINE void nk_shapes_lower_mutual_f64_powervsx_(nk_vf64x2_t outer_x_f64x2, nk_vf64x2_t outer_y_f64x2,
                                                         nk_vf64x2_t outer_z_f64x2, nk_vf64x2_t inner_x_f64x2,
                                                         nk_vf64x2_t inner_y_f64x2, nk_vf64x2_t inner_z_f64x2,
                                                         nk_vf64x2_t *nearest_f64x2, nk_f64_t *reverse_minima,
                                                         nk_size_t reverse_stride) {
    nk_vf64x2_t const minus_x_f64x2 = vec_sub(outer_x_f64x2, inner_x_f64x2);
    nk_vf64x2_t const plus_x_f64x2 = vec_add(outer_x_f64x2, inner_x_f64x2);
    nk_vf64x2_t const minus_y_f64x2 = vec_sub(outer_y_f64x2, inner_y_f64x2);
    nk_vf64x2_t const plus_y_f64x2 = vec_add(outer_y_f64x2, inner_y_f64x2);
    nk_vf64x2_t const z_f64x2[2] = {vec_sub(outer_z_f64x2, inner_z_f64x2), vec_add(outer_z_f64x2, inner_z_f64x2)};
    nk_vf64x2_t const minus_xx_f64x2 = vec_mul(minus_x_f64x2, minus_x_f64x2);
    nk_vf64x2_t const plus_xx_f64x2 = vec_mul(plus_x_f64x2, plus_x_f64x2);
    nk_vf64x2_t const planar_f64x2[4] = {
        vec_madd(minus_y_f64x2, minus_y_f64x2, minus_xx_f64x2), vec_madd(minus_y_f64x2, minus_y_f64x2, plus_xx_f64x2),
        vec_madd(plus_y_f64x2, plus_y_f64x2, minus_xx_f64x2), vec_madd(plus_y_f64x2, plus_y_f64x2, plus_xx_f64x2)};
    for (int flip = 0; flip != 8; ++flip) {
        nk_vf64x2_t const squares_f64x2 = vec_madd(z_f64x2[flip >> 2], z_f64x2[flip >> 2], planar_f64x2[flip & 3]);
        nk_f64_t *reverse = reverse_minima + flip * reverse_stride;
        nearest_f64x2[flip] = vec_min(nearest_f64x2[flip], squares_f64x2);
        vec_xst(vec_min(vec_xl(0, reverse), squares_f64x2), 0, reverse);
    }
}

/** One outer point's sweep of @c f32 inner points in f64 lanes, two at a time, the last pair of
 *  an odd count with an infinite x in its padded lane. */
NUMKONG_INLINE void nk_shapes_sweep_nearest_f32_powervsx_(nk_vf64x2_t outer_x_f64x2, nk_vf64x2_t outer_y_f64x2,
                                                          nk_vf64x2_t outer_z_f64x2, nk_f32_t const *inner_axes,
                                                          nk_size_t inner_stride, nk_size_t inner_count,
                                                          nk_vf64x2_t *nearest_f64x2) {
    nk_size_t const pairs_end = inner_count & ~(nk_size_t)1;
    for (nk_size_t inner_index = 0; inner_index != pairs_end; inner_index += 2)
        nk_shapes_lower_nearest_f64_powervsx_(
            outer_x_f64x2, outer_y_f64x2, outer_z_f64x2,
            vec_doubleh(vec_xl_len((nk_f32_t *)inner_axes + inner_index, 8)),
            vec_doubleh(vec_xl_len((nk_f32_t *)inner_axes + inner_stride + inner_index, 8)),
            vec_doubleh(vec_xl_len((nk_f32_t *)inner_axes + 2 * inner_stride + inner_index, 8)), nearest_f64x2);
    if (pairs_end != inner_count) {
        nk_vf64x2_t const inner_x_f64x2 = vec_insert(NUMKONG_F64_INF,
                                                     vec_doubleh(vec_xl_len((nk_f32_t *)inner_axes + pairs_end, 8)), 1);
        nk_shapes_lower_nearest_f64_powervsx_(
            outer_x_f64x2, outer_y_f64x2, outer_z_f64x2, inner_x_f64x2,
            vec_doubleh(vec_xl_len((nk_f32_t *)inner_axes + inner_stride + pairs_end, 8)),
            vec_doubleh(vec_xl_len((nk_f32_t *)inner_axes + 2 * inner_stride + pairs_end, 8)), nearest_f64x2);
    }
}

/** @ref nk_shapes_sweep_nearest_f32_powervsx_ that also lowers the views' reverse minima. */
NUMKONG_INLINE void nk_shapes_sweep_mutual_f32_powervsx_(nk_vf64x2_t outer_x_f64x2, nk_vf64x2_t outer_y_f64x2,
                                                         nk_vf64x2_t outer_z_f64x2, nk_shapes_mutual_f32_t const *views,
                                                         nk_vf64x2_t *nearest_f64x2) {
    nk_f32_t *inner_axes = (nk_f32_t *)views->inner_axes;
    nk_size_t const inner_stride = views->inner_stride, inner_count = views->inner_count;
    nk_size_t const pairs_end = inner_count & ~(nk_size_t)1;
    for (nk_size_t inner_index = 0; inner_index != pairs_end; inner_index += 2)
        nk_shapes_lower_mutual_f64_powervsx_(outer_x_f64x2, outer_y_f64x2, outer_z_f64x2,
                                             vec_doubleh(vec_xl_len(inner_axes + inner_index, 8)),
                                             vec_doubleh(vec_xl_len(inner_axes + inner_stride + inner_index, 8)),
                                             vec_doubleh(vec_xl_len(inner_axes + 2 * inner_stride + inner_index, 8)),
                                             nearest_f64x2, views->reverse_minima + inner_index, views->reverse_stride);
    if (pairs_end != inner_count) {
        nk_vf64x2_t const inner_x_f64x2 = vec_insert(NUMKONG_F64_INF,
                                                     vec_doubleh(vec_xl_len(inner_axes + pairs_end, 8)), 1);
        nk_shapes_lower_mutual_f64_powervsx_(outer_x_f64x2, outer_y_f64x2, outer_z_f64x2, inner_x_f64x2,
                                             vec_doubleh(vec_xl_len(inner_axes + inner_stride + pairs_end, 8)),
                                             vec_doubleh(vec_xl_len(inner_axes + 2 * inner_stride + pairs_end, 8)),
                                             nearest_f64x2, views->reverse_minima + pairs_end, views->reverse_stride);
    }
}

/** The least of every flip's nearest squares over its lanes, two flips per vector. */
NUMKONG_INLINE void nk_shapes_least_flips_f64_powervsx_(nk_vf64x2_t const *nearest_f64x2, nk_vf64x2_t *least_f64x2) {
    for (int pair = 0; pair != 4; ++pair)
        least_f64x2[pair] = vec_min(vec_mergeh(nearest_f64x2[2 * pair], nearest_f64x2[2 * pair + 1]),
                                    vec_mergel(nearest_f64x2[2 * pair], nearest_f64x2[2 * pair + 1]));
}

/** Fused Chamfer flips of @c f32 packs in f64 lanes, reductions included. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f32_powervsx_(void const *row_packed, void const *column_packed,
                                                                 nk_f64_t limit, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_f64_t *chamfers) {
    nk_shapes_mutual_f32_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f32_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    nk_vf64x2_t const infinity_f64x2 = vec_splats(NUMKONG_F64_INF);
    for (nk_size_t index = 0; index < 8 * views.reverse_stride; index += 2)
        vec_xst(infinity_f64x2, 0, views.reverse_minima + index);
    nk_f64_t const outer_points = (nk_f64_t)views.outer_count, inner_points = (nk_f64_t)views.inner_count;
    nk_vf64x2_t const threshold_f64x2 = vec_splats(2 * limit * outer_points);
    nk_vf64x2_t sums_f64x2[4], nearest_f64x2[8], least_f64x2[4];
    for (int pair = 0; pair != 4; ++pair) sums_f64x2[pair] = vec_splats(0.0);
    nk_f64_t sums[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        nk_vf64x2_t const outer_x_f64x2 = vec_splats((nk_f64_t)views.outer_axes[outer_index]);
        nk_vf64x2_t const outer_y_f64x2 = vec_splats((nk_f64_t)views.outer_axes[views.outer_stride + outer_index]);
        nk_vf64x2_t const outer_z_f64x2 = vec_splats((nk_f64_t)views.outer_axes[2 * views.outer_stride + outer_index]);
        for (int flip = 0; flip != 8; ++flip) nearest_f64x2[flip] = infinity_f64x2;
        nk_shapes_sweep_mutual_f32_powervsx_(outer_x_f64x2, outer_y_f64x2, outer_z_f64x2, &views, nearest_f64x2);
        nk_shapes_least_flips_f64_powervsx_(nearest_f64x2, least_f64x2);
        int exceeded = 1;
        for (int pair = 0; pair != 4; ++pair) {
            sums_f64x2[pair] = vec_add(sums_f64x2[pair], vec_sqrt(least_f64x2[pair]));
            exceeded &= vec_all_gt(sums_f64x2[pair], threshold_f64x2);
        }
        if (exceeded) {
            for (int pair = 0; pair != 4; ++pair) vec_xst(sums_f64x2[pair], 0, sums + 2 * pair);
            if (nk_shapes_mutual_past_limit_f64_serial_(sums, outer_points, limit, chamfers)) return nk_success_k;
        }
    }
    for (int pair = 0; pair != 4; ++pair) vec_xst(sums_f64x2[pair], 0, sums + 2 * pair);
    for (int flip = 0; flip != 8; ++flip) {
        nk_f64_t const *reverse = views.reverse_minima + flip * views.reverse_stride;
        nk_vf64x2_t sum_f64x2 = vec_splats(0.0);
        nk_size_t index = 0;
        for (; index + 2 <= views.inner_count; index += 2)
            sum_f64x2 = vec_add(sum_f64x2, vec_sqrt(vec_xl(0, reverse + index)));
        nk_f64_t reverse_sum = nk_hsum_f64x2_powervsx_(sum_f64x2);
        if (index != views.inner_count) reverse_sum += nk_sqrt_f64_powervsx_(reverse[index]);
        chamfers[flip] = (sums[flip] / outer_points + reverse_sum / inner_points) / 2;
    }
    return nk_success_k;
}

/** Hausdorff flips of @c f32 packs in f64 lanes, stopping as the @c f16 pair does. */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f32_powervsx_(void const *row_packed, void const *column_packed,
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
    nk_vf64x2_t const infinity_f64x2 = vec_splats(NUMKONG_F64_INF);
    nk_vf64x2_t farthest_f64x2[4], nearest_f64x2[8], least_f64x2[4];
    for (int pair = 0; pair != 4; ++pair) farthest_f64x2[pair] = vec_splats(0.0);
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_f32_t const *outer_axes = direction ? column_axes : row_axes,
                       *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            for (int flip = 0; flip != 8; ++flip) nearest_f64x2[flip] = infinity_f64x2;
            nk_shapes_sweep_nearest_f32_powervsx_(vec_splats((nk_f64_t)outer_axes[outer_index]),
                                                  vec_splats((nk_f64_t)outer_axes[outer_stride + outer_index]),
                                                  vec_splats((nk_f64_t)outer_axes[2 * outer_stride + outer_index]),
                                                  inner_axes, inner_stride, inner_count, nearest_f64x2);
            nk_shapes_least_flips_f64_powervsx_(nearest_f64x2, least_f64x2);
            nk_vf64x2_t overall_least_f64x2 = vec_splats(NUMKONG_F64_INF);
            for (int pair = 0; pair != 4; ++pair) {
                farthest_f64x2[pair] = vec_max(farthest_f64x2[pair], least_f64x2[pair]);
                overall_least_f64x2 = vec_min(overall_least_f64x2, farthest_f64x2[pair]);
            }
            nk_f64_t const least = vec_extract(
                vec_min(overall_least_f64x2, vec_mergel(overall_least_f64x2, overall_least_f64x2)), 0);
            rejected = least > limit_squared && nk_sqrt_f64_powervsx_(least) > limit;
        }
    }
    for (int pair = 0; pair != 4; ++pair) vec_xst(vec_sqrt(farthest_f64x2[pair]), 0, hausdorffs + 2 * pair);
}

/** @copydoc nk_shapes_lower_nearest_f64_powervsx_ */
NUMKONG_INLINE void nk_shapes_lower_nearest_f32_powervsx_(nk_vf32x4_t outer_x_f32x4, nk_vf32x4_t outer_y_f32x4,
                                                          nk_vf32x4_t outer_z_f32x4, nk_vf32x4_t inner_x_f32x4,
                                                          nk_vf32x4_t inner_y_f32x4, nk_vf32x4_t inner_z_f32x4,
                                                          nk_vf32x4_t *nearest_f32x4) {
    nk_shapes_lower_nearest_differences_f32_powervsx_(
        vec_sub(outer_x_f32x4, inner_x_f32x4), vec_add(outer_x_f32x4, inner_x_f32x4),
        vec_sub(outer_y_f32x4, inner_y_f32x4), vec_add(outer_y_f32x4, inner_y_f32x4),
        vec_sub(outer_z_f32x4, inner_z_f32x4), vec_add(outer_z_f32x4, inner_z_f32x4), nearest_f32x4);
}

/** @copydoc nk_shapes_lower_mutual_f64_powervsx_ */
NUMKONG_INLINE void nk_shapes_lower_mutual_f32_powervsx_(nk_vf32x4_t outer_x_f32x4, nk_vf32x4_t outer_y_f32x4,
                                                         nk_vf32x4_t outer_z_f32x4, nk_vf32x4_t inner_x_f32x4,
                                                         nk_vf32x4_t inner_y_f32x4, nk_vf32x4_t inner_z_f32x4,
                                                         nk_vf32x4_t *nearest_f32x4, nk_f32_t *reverse_minima,
                                                         nk_size_t reverse_stride) {
    nk_shapes_lower_mutual_differences_f32_powervsx_(
        vec_sub(outer_x_f32x4, inner_x_f32x4), vec_add(outer_x_f32x4, inner_x_f32x4),
        vec_sub(outer_y_f32x4, inner_y_f32x4), vec_add(outer_y_f32x4, inner_y_f32x4),
        vec_sub(outer_z_f32x4, inner_z_f32x4), vec_add(outer_z_f32x4, inner_z_f32x4), nearest_f32x4, reverse_minima,
        reverse_stride);
}

/** Widens the @c f16 value at @p source into all four f32 lanes, reading only its two bytes. */
NUMKONG_INLINE nk_vf32x4_t nk_shapes_broadcast_f16_powervsx_(nk_f16_t const *source) {
    return vec_splat(vec_extract_fp32_from_shorth((nk_vu16x8_t)vec_xl_len((nk_u16_t *)source, 2)), 0);
}

/** One outer point's sweep of @c f16 inner points in f32 lanes, four at a time, the last
 *  quadruple of a count off the multiple with an infinite x in its padded lanes. */
NUMKONG_INLINE void nk_shapes_sweep_nearest_f16_powervsx_(nk_vf32x4_t outer_x_f32x4, nk_vf32x4_t outer_y_f32x4,
                                                          nk_vf32x4_t outer_z_f32x4, nk_f16_t const *inner_axes,
                                                          nk_size_t inner_stride, nk_size_t inner_count,
                                                          nk_b128_vec_t const *tail_vec, nk_vf32x4_t *nearest_f32x4) {
    nk_size_t const quadruples_end = inner_count & ~(nk_size_t)3;
    for (nk_size_t inner_index = 0; inner_index != quadruples_end; inner_index += 4)
        nk_shapes_lower_nearest_f32_powervsx_(
            outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, nk_load_f16x4_as_f32_powervsx_(inner_axes + inner_index),
            nk_load_f16x4_as_f32_powervsx_(inner_axes + inner_stride + inner_index),
            nk_load_f16x4_as_f32_powervsx_(inner_axes + 2 * inner_stride + inner_index), nearest_f32x4);
    if (quadruples_end != inner_count)
        nk_shapes_lower_nearest_f32_powervsx_(
            outer_x_f32x4, outer_y_f32x4, outer_z_f32x4,
            vec_sel(vec_splats(NUMKONG_F32_INF), nk_load_f16x4_as_f32_powervsx_(inner_axes + quadruples_end),
                    tail_vec->vu32x4),
            nk_load_f16x4_as_f32_powervsx_(inner_axes + inner_stride + quadruples_end),
            nk_load_f16x4_as_f32_powervsx_(inner_axes + 2 * inner_stride + quadruples_end), nearest_f32x4);
}

/** @ref nk_shapes_sweep_nearest_f16_powervsx_ that also lowers the views' reverse minima. */
NUMKONG_INLINE void nk_shapes_sweep_mutual_f16_powervsx_(nk_vf32x4_t outer_x_f32x4, nk_vf32x4_t outer_y_f32x4,
                                                         nk_vf32x4_t outer_z_f32x4, nk_shapes_mutual_f16_t const *views,
                                                         nk_b128_vec_t const *tail_vec, nk_vf32x4_t *nearest_f32x4) {
    nk_f16_t const *inner_axes = views->inner_axes;
    nk_size_t const inner_stride = views->inner_stride, inner_count = views->inner_count;
    nk_size_t const quadruples_end = inner_count & ~(nk_size_t)3;
    for (nk_size_t inner_index = 0; inner_index != quadruples_end; inner_index += 4)
        nk_shapes_lower_mutual_f32_powervsx_(
            outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, nk_load_f16x4_as_f32_powervsx_(inner_axes + inner_index),
            nk_load_f16x4_as_f32_powervsx_(inner_axes + inner_stride + inner_index),
            nk_load_f16x4_as_f32_powervsx_(inner_axes + 2 * inner_stride + inner_index), nearest_f32x4,
            views->reverse_minima + inner_index, views->reverse_stride);
    if (quadruples_end != inner_count)
        nk_shapes_lower_mutual_f32_powervsx_(
            outer_x_f32x4, outer_y_f32x4, outer_z_f32x4,
            vec_sel(vec_splats(NUMKONG_F32_INF), nk_load_f16x4_as_f32_powervsx_(inner_axes + quadruples_end),
                    tail_vec->vu32x4),
            nk_load_f16x4_as_f32_powervsx_(inner_axes + inner_stride + quadruples_end),
            nk_load_f16x4_as_f32_powervsx_(inner_axes + 2 * inner_stride + quadruples_end), nearest_f32x4,
            views->reverse_minima + quadruples_end, views->reverse_stride);
}

/** Fused Chamfer flips of @c f16 packs, four inner points per load, reductions in lanes. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f16_powervsx_(void const *row_packed, void const *column_packed,
                                                                 nk_f32_t limit, void *workspace,
                                                                 nk_size_t workspace_bytes, nk_f32_t *chamfers) {
    nk_shapes_mutual_f16_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f16_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    nk_vf32x4_t const infinity_f32x4 = vec_splats(NUMKONG_F32_INF);
    for (nk_size_t index = 0; index < 8 * views.reverse_stride; index += 4)
        vec_xst(infinity_f32x4, 0, views.reverse_minima + index);
    nk_b128_vec_t const tail_vec = nk_shapes_tail_powervsx_(views.inner_count);
    nk_f32_t const outer_points = (nk_f32_t)views.outer_count;
    nk_vf32x4_t const threshold_f32x4 = vec_splats(2 * limit * outer_points);
    nk_vf32x4_t sums_f32x4[2] = {vec_splats(0.0f), vec_splats(0.0f)}, nearest_f32x4[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        nk_vf32x4_t const outer_x_f32x4 = nk_shapes_broadcast_f16_powervsx_(views.outer_axes + outer_index);
        nk_vf32x4_t const outer_y_f32x4 = nk_shapes_broadcast_f16_powervsx_(views.outer_axes + views.outer_stride +
                                                                            outer_index);
        nk_vf32x4_t const outer_z_f32x4 = nk_shapes_broadcast_f16_powervsx_(views.outer_axes + 2 * views.outer_stride +
                                                                            outer_index);
        for (int flip = 0; flip != 8; ++flip) nearest_f32x4[flip] = infinity_f32x4;
        nk_shapes_sweep_mutual_f16_powervsx_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, &views, &tail_vec,
                                             nearest_f32x4);
        if (nk_shapes_chamfer_row_f32_powervsx_(nearest_f32x4, sums_f32x4, threshold_f32x4, outer_points, limit,
                                                chamfers))
            return nk_success_k;
    }
    nk_shapes_chamfer_finish_f16_powervsx_(sums_f32x4, &views, chamfers);
    return nk_success_k;
}

/** Hausdorff flips of @c f16 packs, four inner points per load, stopping once every flip's
 *  farthest square passes @p limit . */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f16_powervsx_(void const *row_packed, void const *column_packed,
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
    nk_b128_vec_t const row_tail_vec = nk_shapes_tail_powervsx_(rows);
    nk_b128_vec_t const column_tail_vec = nk_shapes_tail_powervsx_(columns);
    nk_f32_t const limit_squared = limit * limit;
    nk_vf32x4_t const infinity_f32x4 = vec_splats(NUMKONG_F32_INF);
    nk_vf32x4_t farthest_f32x4[2] = {vec_splats(0.0f), vec_splats(0.0f)}, nearest_f32x4[8];
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_f16_t const *outer_axes = direction ? column_axes : row_axes;
        nk_f16_t const *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        nk_b128_vec_t const *tail_vec = direction ? &row_tail_vec : &column_tail_vec;
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            nk_vf32x4_t const outer_x_f32x4 = nk_shapes_broadcast_f16_powervsx_(outer_axes + outer_index);
            nk_vf32x4_t const outer_y_f32x4 = nk_shapes_broadcast_f16_powervsx_(outer_axes + outer_stride +
                                                                                outer_index);
            nk_vf32x4_t const outer_z_f32x4 = nk_shapes_broadcast_f16_powervsx_(outer_axes + 2 * outer_stride +
                                                                                outer_index);
            for (int flip = 0; flip != 8; ++flip) nearest_f32x4[flip] = infinity_f32x4;
            nk_shapes_sweep_nearest_f16_powervsx_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_axes, inner_stride,
                                                  inner_count, tail_vec, nearest_f32x4);
            rejected = nk_shapes_hausdorff_row_f32_powervsx_(nearest_f32x4, farthest_f32x4, limit, limit_squared);
        }
    }
    vec_xst(vec_sqrt(farthest_f32x4[0]), 0, hausdorffs), vec_xst(vec_sqrt(farthest_f32x4[1]), 0, hausdorffs + 4);
}

/** Squared distances from a broadcast row point in fused multiply-adds, which also serve single
 *  arcs, so that tree arcs keep a zero reduced cost. */
NUMKONG_INLINE nk_vf32x4_t nk_shapes_point_costs_f32x4_powervsx_(nk_vf32x4_t row_x_f32x4, nk_vf32x4_t row_y_f32x4,
                                                                 nk_vf32x4_t row_z_f32x4, nk_vf32x4_t column_x_f32x4,
                                                                 nk_vf32x4_t column_y_f32x4,
                                                                 nk_vf32x4_t column_z_f32x4) {
    nk_vf32x4_t const delta_x_f32x4 = vec_sub(row_x_f32x4, column_x_f32x4);
    nk_vf32x4_t const delta_y_f32x4 = vec_sub(row_y_f32x4, column_y_f32x4);
    nk_vf32x4_t const delta_z_f32x4 = vec_sub(row_z_f32x4, column_z_f32x4);
    return vec_madd(delta_z_f32x4, delta_z_f32x4,
                    vec_madd(delta_y_f32x4, delta_y_f32x4, vec_mul(delta_x_f32x4, delta_x_f32x4)));
}

NUMKONG_INLINE nk_f32_t nk_shapes_point_cost_f32_powervsx_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t row,
                                                           nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    return vec_extract(
        nk_shapes_point_costs_f32x4_powervsx_(
            vec_splats(row_points[row]), vec_splats(row_points[stride + row]), vec_splats(row_points[2 * stride + row]),
            vec_splats(rotated_points[column]), vec_splats(rotated_points[stride + column]),
            vec_splats(rotated_points[2 * stride + column])),
        0);
}

/** The reduced costs (c − πᵢ) + πⱼ of a row's arcs to two columns, with matrix costs −G taken as
 *  πⱼ − (G + πᵢ), which rounds the same. */
NUMKONG_INLINE nk_vf64x2_t nk_shapes_reduced_costs_f32_powervsx_(nk_shapes_workspace_f32_serial_t const *views,
                                                                 nk_f64_t const *gradient, nk_size_t rows,
                                                                 nk_size_t columns, nk_size_t row, nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t *row_points = views->row_points, *rotated_points = views->rotated_points;
    nk_vf64x2_t const row_potential_f64x2 = vec_splats(views->potential[row]);
    nk_vf64x2_t const column_potentials_f64x2 = vec_xl(0, views->potential + rows + column);
    if (gradient)
        return vec_sub(column_potentials_f64x2,
                       vec_add(vec_xl(0, gradient + row * columns + column), row_potential_f64x2));
    nk_vf64x2_t const costs_f64x2 = vec_doubleh(nk_shapes_point_costs_f32x4_powervsx_(
        vec_splats(row_points[row]), vec_splats(row_points[stride + row]), vec_splats(row_points[2 * stride + row]),
        vec_xl_len(rotated_points + column, 8), vec_xl_len(rotated_points + stride + column, 8),
        vec_xl_len(rotated_points + 2 * stride + column, 8)));
    return vec_add(vec_sub(costs_f64x2, row_potential_f64x2), column_potentials_f64x2);
}

/** Keeps the first of two reduced costs from @p column that beats the running least. */
NUMKONG_INLINE void nk_shapes_keep_least_f64x2_powervsx_(nk_vf64x2_t reduced_f64x2, nk_size_t row, nk_size_t column,
                                                         nk_vf64x2_t *least_f64x2, nk_f64_t *entering_cost,
                                                         nk_size_t *entering_row, nk_size_t *entering_column) {
    if (!vec_any_lt(reduced_f64x2, *least_f64x2)) return;
    nk_f64_t const reduced_low = vec_extract(reduced_f64x2, 0), reduced_high = vec_extract(reduced_f64x2, 1);
    int const takes_high = reduced_high < reduced_low;
    *entering_cost = takes_high ? reduced_high : reduced_low;
    *entering_row = row, *entering_column = column + takes_high;
    *least_f64x2 = vec_splats(*entering_cost);
}

/** Keeps a tile's first least reduced cost below @p entering_cost, two columns at a time. The lane
 *  past an odd tile reads neighboring workspace and costs infinity. */
NUMKONG_INLINE void nk_shapes_price_tile_f32_powervsx_(nk_shapes_workspace_f32_serial_t const *views,
                                                       nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                       nk_size_t row, nk_size_t column, nk_size_t tile,
                                                       nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                       nk_size_t *entering_column) {
    nk_vf64x2_t least_f64x2 = vec_splats(*entering_cost);
    nk_size_t const pairs_end = tile & ~(nk_size_t)1;
    for (nk_size_t step = 0; step != pairs_end; step += 2)
        nk_shapes_keep_least_f64x2_powervsx_(
            nk_shapes_reduced_costs_f32_powervsx_(views, gradient, rows, columns, row, column + step), row,
            column + step, &least_f64x2, entering_cost, entering_row, entering_column);
    if (pairs_end == tile) return;
    nk_vf64x2_t const reduced_f64x2 = nk_shapes_reduced_costs_f32_powervsx_(views, gradient, rows, columns, row,
                                                                            column + pairs_end);
    nk_shapes_keep_least_f64x2_powervsx_(vec_insert(NUMKONG_F64_INF, reduced_f64x2, 1), row, column + pairs_end,
                                         &least_f64x2, entering_cost, entering_row, entering_column);
}

NUMKONG_INLINE void nk_shapes_tree_costs_f32_powervsx_(nk_shapes_workspace_f32_serial_t const *views,
                                                       nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        views->potential[node] = gradient ? -gradient[row * columns + column]
                                          : (nk_f64_t)nk_shapes_point_cost_f32_powervsx_(views, row, column);
    }
}

NUMKONG_INLINE void nk_shapes_network_simplex_f32_powervsx_(nk_shapes_workspace_f32_serial_t const *views,
                                                            nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                            nk_f64_t cost_scale) {
    nk_shapes_simplex_serial_t simplex;
    nk_shapes_tree_costs_f32_powervsx_(views, gradient, rows, columns);
    nk_shapes_tree_potentials_serial_(views->tree, views->tree_stride, views->potential, rows);
    nk_shapes_simplex_start_serial_(&simplex, views->tree, views->tree_stride, views->potential, rows, columns,
                                    cost_scale);
    while (nk_shapes_simplex_next_tile_serial_(&simplex))
        nk_shapes_price_tile_f32_powervsx_(views, gradient, rows, columns, simplex.row, simplex.column, simplex.tile,
                                           &simplex.entering_cost, &simplex.entering_row, &simplex.entering_column);
}

NUMKONG_INLINE nk_f32_t nk_shapes_tree_cost_f32_powervsx_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t rows,
                                                          nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f32_t cost = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        cost += (nk_f32_t)flow[node] * nk_shapes_point_cost_f32_powervsx_(views, row, column);
    }
    return cost / (nk_f32_t)(rows * columns);
}

NUMKONG_INLINE nk_vf64x2_t nk_shapes_point_costs_f64x2_powervsx_(nk_vf64x2_t row_x_f64x2, nk_vf64x2_t row_y_f64x2,
                                                                 nk_vf64x2_t row_z_f64x2, nk_vf64x2_t column_x_f64x2,
                                                                 nk_vf64x2_t column_y_f64x2,
                                                                 nk_vf64x2_t column_z_f64x2) {
    nk_vf64x2_t const delta_x_f64x2 = vec_sub(row_x_f64x2, column_x_f64x2);
    nk_vf64x2_t const delta_y_f64x2 = vec_sub(row_y_f64x2, column_y_f64x2);
    nk_vf64x2_t const delta_z_f64x2 = vec_sub(row_z_f64x2, column_z_f64x2);
    return vec_madd(delta_z_f64x2, delta_z_f64x2,
                    vec_madd(delta_y_f64x2, delta_y_f64x2, vec_mul(delta_x_f64x2, delta_x_f64x2)));
}

NUMKONG_INLINE nk_f64_t nk_shapes_point_cost_f64_powervsx_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t row,
                                                           nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    return vec_extract(
        nk_shapes_point_costs_f64x2_powervsx_(
            vec_splats(row_points[row]), vec_splats(row_points[stride + row]), vec_splats(row_points[2 * stride + row]),
            vec_splats(rotated_points[column]), vec_splats(rotated_points[stride + column]),
            vec_splats(rotated_points[2 * stride + column])),
        0);
}

NUMKONG_INLINE nk_vf64x2_t nk_shapes_reduced_costs_f64_powervsx_(nk_shapes_workspace_f64_serial_t const *views,
                                                                 nk_f64_t const *gradient, nk_size_t rows,
                                                                 nk_size_t columns, nk_size_t row, nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    nk_vf64x2_t const row_potential_f64x2 = vec_splats(views->potential[row]);
    nk_vf64x2_t const column_potentials_f64x2 = vec_xl(0, views->potential + rows + column);
    if (gradient)
        return vec_sub(column_potentials_f64x2,
                       vec_add(vec_xl(0, gradient + row * columns + column), row_potential_f64x2));
    nk_vf64x2_t const costs_f64x2 = nk_shapes_point_costs_f64x2_powervsx_(
        vec_splats(row_points[row]), vec_splats(row_points[stride + row]), vec_splats(row_points[2 * stride + row]),
        vec_xl(0, rotated_points + column), vec_xl(0, rotated_points + stride + column),
        vec_xl(0, rotated_points + 2 * stride + column));
    return vec_add(vec_sub(costs_f64x2, row_potential_f64x2), column_potentials_f64x2);
}

NUMKONG_INLINE void nk_shapes_price_tile_f64_powervsx_(nk_shapes_workspace_f64_serial_t const *views,
                                                       nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                       nk_size_t row, nk_size_t column, nk_size_t tile,
                                                       nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                       nk_size_t *entering_column) {
    nk_vf64x2_t least_f64x2 = vec_splats(*entering_cost);
    nk_size_t const pairs_end = tile & ~(nk_size_t)1;
    for (nk_size_t step = 0; step != pairs_end; step += 2)
        nk_shapes_keep_least_f64x2_powervsx_(
            nk_shapes_reduced_costs_f64_powervsx_(views, gradient, rows, columns, row, column + step), row,
            column + step, &least_f64x2, entering_cost, entering_row, entering_column);
    if (pairs_end == tile) return;
    nk_vf64x2_t const reduced_f64x2 = nk_shapes_reduced_costs_f64_powervsx_(views, gradient, rows, columns, row,
                                                                            column + pairs_end);
    nk_shapes_keep_least_f64x2_powervsx_(vec_insert(NUMKONG_F64_INF, reduced_f64x2, 1), row, column + pairs_end,
                                         &least_f64x2, entering_cost, entering_row, entering_column);
}

NUMKONG_INLINE void nk_shapes_tree_costs_f64_powervsx_(nk_shapes_workspace_f64_serial_t const *views,
                                                       nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        views->potential[node] = gradient ? -gradient[row * columns + column]
                                          : nk_shapes_point_cost_f64_powervsx_(views, row, column);
    }
}

NUMKONG_INLINE void nk_shapes_network_simplex_f64_powervsx_(nk_shapes_workspace_f64_serial_t const *views,
                                                            nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                            nk_f64_t cost_scale) {
    nk_shapes_simplex_serial_t simplex;
    nk_shapes_tree_costs_f64_powervsx_(views, gradient, rows, columns);
    nk_shapes_tree_potentials_serial_(views->tree, views->tree_stride, views->potential, rows);
    nk_shapes_simplex_start_serial_(&simplex, views->tree, views->tree_stride, views->potential, rows, columns,
                                    cost_scale);
    while (nk_shapes_simplex_next_tile_serial_(&simplex))
        nk_shapes_price_tile_f64_powervsx_(views, gradient, rows, columns, simplex.row, simplex.column, simplex.tile,
                                           &simplex.entering_cost, &simplex.entering_row, &simplex.entering_column);
}

NUMKONG_INLINE nk_f64_t nk_shapes_tree_cost_f64_powervsx_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t rows,
                                                          nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f64_t cost = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        cost += (nk_f64_t)flow[node] * nk_shapes_point_cost_f64_powervsx_(views, row, column);
    }
    return cost / (nk_f64_t)(rows * columns);
}

NUMKONG_INLINE nk_f32_t nk_shapes_wasserstein_procrustes_f16_powervsx_(nk_shapes_workspace_f32_serial_t const *views,
                                                                       nk_size_t rows, nk_size_t columns,
                                                                       nk_f64_t cost_scale, nk_f32_t *rotation) {
    nk_shapes_procrustes_f32_serial_t search;
    nk_shapes_procrustes_start_f32_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_f32_serial_(&search)) {
        nk_shapes_network_simplex_f32_powervsx_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_f32_powervsx_(views, rows, columns);
    }
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f16_powervsx_(nk_shapes_workspace_f32_serial_t const *views,
                                                                   nk_size_t rows, nk_size_t columns,
                                                                   nk_f64_t cost_scale, nk_f64_t const *row_distances,
                                                                   nk_f64_t const *column_distances,
                                                                   nk_f32_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f16_powervsx_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f32_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_serial_(&search);
    while (nk_shapes_gromov_next_f32_serial_(&search, views, rotation))
        nk_shapes_network_simplex_f32_powervsx_(views, search.gradient, rows, columns, search.gradient_scale);
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_wasserstein_procrustes_f32_powervsx_(nk_shapes_workspace_f64_serial_t const *views,
                                                                       nk_size_t rows, nk_size_t columns,
                                                                       nk_f64_t cost_scale, nk_f64_t *rotation) {
    nk_shapes_procrustes_f64_serial_t search;
    nk_shapes_procrustes_start_f64_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_f64_serial_(&search)) {
        nk_shapes_network_simplex_f64_powervsx_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_f64_powervsx_(views, rows, columns);
    }
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f32_powervsx_(nk_shapes_workspace_f64_serial_t const *views,
                                                                   nk_size_t rows, nk_size_t columns,
                                                                   nk_f64_t cost_scale, nk_f64_t const *row_distances,
                                                                   nk_f64_t const *column_distances,
                                                                   nk_f64_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f32_powervsx_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f64_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_serial_(&search);
    while (nk_shapes_gromov_next_f64_serial_(&search, views, rotation))
        nk_shapes_network_simplex_f64_powervsx_(views, search.gradient, rows, columns, search.gradient_scale);
    return search.best;
}

NUMKONG_INLINE void nk_jacobi_eigenanalysis_f32_powervsx_(nk_f32_t *matrix, nk_f32_t *quaternion) {
    nk_jacobi_start_f32_serial_(quaternion);
    for (int sweep = 0; sweep != 16; ++sweep) {
        for (int axis = 0; axis != 3; ++axis) {
            nk_f32_t const norm = nk_jacobi_half_angle_f32_serial_(matrix);
            nk_jacobi_conjugate_f32_serial_(axis, nk_rsqrt_f32_powervsx_(norm), matrix, quaternion);
        }
        if (nk_jacobi_converged_f32_serial_(matrix, 0x1p-24f)) break;
    }
    nk_f32_t const length_squared = nk_jacobi_length_squared_f32_serial_(quaternion);
    nk_jacobi_rescale_f32_serial_(nk_rsqrt_f32_powervsx_(length_squared), quaternion);
}

NUMKONG_INLINE void nk_jacobi_eigenanalysis_f64_powervsx_(nk_f64_t *matrix, nk_f64_t *quaternion) {
    nk_jacobi_start_f64_serial_(quaternion);
    for (int sweep = 0; sweep != 16; ++sweep) {
        for (int axis = 0; axis != 3; ++axis) {
            nk_f64_t const norm = nk_jacobi_half_angle_f64_serial_(matrix);
            nk_jacobi_conjugate_f64_serial_(axis, nk_rsqrt_f64_powervsx_(norm), matrix, quaternion);
        }
        if (nk_jacobi_converged_f64_serial_(matrix, 0x1p-53)) break;
    }
    nk_f64_t const length_squared = nk_jacobi_length_squared_f64_serial_(quaternion);
    nk_jacobi_rescale_f64_serial_(nk_rsqrt_f64_powervsx_(length_squared), quaternion);
}

/** The next four @c f16 points of the @p left remaining, each axis in f32 lanes, with
 *  @p valid_u32x4 over the lanes holding one. Padded lanes read zero. */
NUMKONG_INLINE void nk_shapes_load_points_f16_powervsx_(nk_f16_t const *points, nk_size_t left, nk_vf32x4_t *x_f32x4,
                                                        nk_vf32x4_t *y_f32x4, nk_vf32x4_t *z_f32x4,
                                                        nk_vu32x4_t *valid_u32x4) {
    // Halves 0 to 7 hold [x0 y0 z0 x1 y1 z1 x2 y2] and 8 to 11 hold [z2 x3 y3 z3]
    nk_vu8x16_t const x_bytes_u8x16 = {0, 1, 6, 7, 12, 13, 18, 19, 0, 1, 6, 7, 12, 13, 18, 19};
    nk_vu8x16_t const y_bytes_u8x16 = {2, 3, 8, 9, 14, 15, 20, 21, 2, 3, 8, 9, 14, 15, 20, 21};
    nk_vu8x16_t const z_bytes_u8x16 = {4, 5, 10, 11, 16, 17, 22, 23, 4, 5, 10, 11, 16, 17, 22, 23};
    nk_vu32x4_t const lanes_u32x4 = {0, 1, 2, 3};
    nk_size_t const points_count = left < 4 ? left : 4, bytes = points_count * 3 * sizeof(nk_f16_t);
    // `lxvl` takes at most 16 bytes, however many the length asks for
    nk_vu16x8_t const front_u16x8 = vec_xl_len((nk_u16_t *)points, bytes);
    nk_vu16x8_t const back_u16x8 = vec_xl_len((nk_u16_t *)points + 8, bytes > 16 ? bytes - 16 : 0);
    *x_f32x4 = vec_extract_fp32_from_shorth(vec_perm(front_u16x8, back_u16x8, x_bytes_u8x16));
    *y_f32x4 = vec_extract_fp32_from_shorth(vec_perm(front_u16x8, back_u16x8, y_bytes_u8x16));
    *z_f32x4 = vec_extract_fp32_from_shorth(vec_perm(front_u16x8, back_u16x8, z_bytes_u8x16));
    *valid_u32x4 = (nk_vu32x4_t)vec_cmplt(lanes_u32x4, vec_splats((nk_u32_t)points_count));
}

/** The moments of a cloud of @c f16 points as @ref nk_shapes_moments_f16_serial_ defines them, four
 *  points per step. */
NUMKONG_INLINE nk_f32_t nk_shapes_moments_f16_powervsx_(nk_f16_t const *points, nk_size_t count, nk_f32_t *centroid,
                                                        nk_f32_t *covariance) {
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    nk_vf32x4_t x_f32x4, y_f32x4, z_f32x4;
    nk_vu32x4_t valid_u32x4;
    nk_shapes_load_points_f16_powervsx_(points, 1, &x_f32x4, &y_f32x4, &z_f32x4, &valid_u32x4);
    nk_vf32x4_t const first_x_f32x4 = vec_splat(x_f32x4, 0), first_y_f32x4 = vec_splat(y_f32x4, 0);
    nk_vf32x4_t const first_z_f32x4 = vec_splat(z_f32x4, 0);
    nk_vf32x4_t sum_x_f32x4 = vec_splats(0.0f), sum_y_f32x4 = sum_x_f32x4, sum_z_f32x4 = sum_x_f32x4;
    nk_vu32x4_t coincident_u32x4 = vec_splats(~0u);
    for (nk_size_t index = 0; index < count; index += 4) {
        nk_shapes_load_points_f16_powervsx_(points + index * 3, count - index, &x_f32x4, &y_f32x4, &z_f32x4,
                                            &valid_u32x4);
        sum_x_f32x4 = vec_add(sum_x_f32x4, x_f32x4), sum_y_f32x4 = vec_add(sum_y_f32x4, y_f32x4);
        sum_z_f32x4 = vec_add(sum_z_f32x4, z_f32x4);
        nk_vu32x4_t const same_u32x4 = (nk_vu32x4_t)vec_and(
            vec_and(vec_cmpeq(x_f32x4, first_x_f32x4), vec_cmpeq(y_f32x4, first_y_f32x4)),
            vec_cmpeq(z_f32x4, first_z_f32x4));
        coincident_u32x4 = vec_and(coincident_u32x4, vec_orc(same_u32x4, valid_u32x4));
    }
    nk_f32_t const points_count = (nk_f32_t)count;
    centroid[0] = nk_hsum_f32x4_powervsx_(sum_x_f32x4) / points_count;
    centroid[1] = nk_hsum_f32x4_powervsx_(sum_y_f32x4) / points_count;
    centroid[2] = nk_hsum_f32x4_powervsx_(sum_z_f32x4) / points_count;
    nk_vf32x4_t const centroid_x_f32x4 = vec_splats(centroid[0]), centroid_y_f32x4 = vec_splats(centroid[1]);
    nk_vf32x4_t const centroid_z_f32x4 = vec_splats(centroid[2]);
    nk_vf32x4_t moments_f32x4[6];
    for (int entry = 0; entry != 6; ++entry) moments_f32x4[entry] = vec_splats(0.0f);
    for (nk_size_t index = 0; index < count; index += 4) {
        nk_shapes_load_points_f16_powervsx_(points + index * 3, count - index, &x_f32x4, &y_f32x4, &z_f32x4,
                                            &valid_u32x4);
        x_f32x4 = vec_sel(x_f32x4, vec_sub(x_f32x4, centroid_x_f32x4), valid_u32x4);
        y_f32x4 = vec_sel(y_f32x4, vec_sub(y_f32x4, centroid_y_f32x4), valid_u32x4);
        z_f32x4 = vec_sel(z_f32x4, vec_sub(z_f32x4, centroid_z_f32x4), valid_u32x4);
        moments_f32x4[0] = vec_madd(x_f32x4, x_f32x4, moments_f32x4[0]);
        moments_f32x4[1] = vec_madd(y_f32x4, x_f32x4, moments_f32x4[1]);
        moments_f32x4[2] = vec_madd(y_f32x4, y_f32x4, moments_f32x4[2]);
        moments_f32x4[3] = vec_madd(z_f32x4, x_f32x4, moments_f32x4[3]);
        moments_f32x4[4] = vec_madd(z_f32x4, y_f32x4, moments_f32x4[4]);
        moments_f32x4[5] = vec_madd(z_f32x4, z_f32x4, moments_f32x4[5]);
    }
    for (int entry = 0; entry != 6; ++entry)
        covariance[entry] = nk_hsum_f32x4_powervsx_(moments_f32x4[entry]) / points_count;
    return vec_all_ne(coincident_u32x4, vec_splats(0u))
               ? 0
               : nk_sqrt_f32_powervsx_(covariance[0] + covariance[2] + covariance[5]);
}

/** The pack of one cloud from its moments, as @c nk_shapes_pack_f16_serial_ lays it out, storing
 *  each step's points and measuring what was stored in one sweep, with zeros in the padding. */
NUMKONG_INLINE void nk_shapes_pack_f16_powervsx_(nk_f16_t const *points, nk_size_t count, nk_f32_t const *centroid,
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
    nk_f32_t spreads[6] = {covariance[0], covariance[1], covariance[2], covariance[3], covariance[4], covariance[5]};
    nk_f32_t quaternion[4], basis[9], axes[9];
    int upper_tied, lower_tied;
    nk_jacobi_eigenanalysis_f32_powervsx_(spreads, quaternion);
    nk_shapes_sorted_basis_f32_serial_(spreads, quaternion, 0x1p-7f, basis, &upper_tied, &lower_tied);
    if (upper_tied && lower_tied) nk_shapes_pick_axis_f16_serial_(points, count, centroid, basis);
    else if (upper_tied || lower_tied) nk_shapes_turn_plane_f16_serial_(points, count, centroid, basis);
    nk_shapes_frame_f32_serial_(basis, upper_tied, header->frame, axes);
    for (int cell = 0; cell != 9; ++cell) axes[cell] /= scale;
    nk_vf32x4_t const centroid_x_f32x4 = vec_splats(centroid[0]), centroid_y_f32x4 = vec_splats(centroid[1]);
    nk_vf32x4_t const centroid_z_f32x4 = vec_splats(centroid[2]);
    nk_vf32x4_t moments_f32x4[6], largest_f32x4 = vec_splats(0.0f);
    for (int entry = 0; entry != 6; ++entry) moments_f32x4[entry] = vec_splats(0.0f);
    nk_vf32x4_t x_f32x4, y_f32x4, z_f32x4, stored_f32x4[3];
    nk_vu32x4_t valid_u32x4;
    for (nk_size_t index = 0; index < count; index += 4) {
        nk_shapes_load_points_f16_powervsx_(points + index * 3, count - index, &x_f32x4, &y_f32x4, &z_f32x4,
                                            &valid_u32x4);
        x_f32x4 = vec_sub(x_f32x4, centroid_x_f32x4), y_f32x4 = vec_sub(y_f32x4, centroid_y_f32x4);
        z_f32x4 = vec_sub(z_f32x4, centroid_z_f32x4);
        for (int axis = 0; axis != 3; ++axis) {
            nk_vf32x4_t const rotated_f32x4 = vec_madd(
                z_f32x4, vec_splats(axes[axis * 3 + 2]),
                vec_madd(y_f32x4, vec_splats(axes[axis * 3 + 1]), vec_mul(x_f32x4, vec_splats(axes[axis * 3]))));
            nk_vf32x4_t const masked_f32x4 = vec_sel(vec_splats(0.0f), rotated_f32x4, valid_u32x4);
            nk_vu16x8_t const stored_u16x8 = vec_pack_to_short_fp32(masked_f32x4, masked_f32x4);
            vec_xst_len((nk_vu8x16_t)stored_u16x8, (nk_u8_t *)(packed_x + axis * axis_stride + index), 8);
            // The spreads describe the stored points, so the sweep measures them after rounding
            stored_f32x4[axis] = vec_extract_fp32_from_shorth(stored_u16x8);
        }
        x_f32x4 = stored_f32x4[0], y_f32x4 = stored_f32x4[1], z_f32x4 = stored_f32x4[2];
        nk_vf32x4_t const xx_f32x4 = vec_mul(x_f32x4, x_f32x4), yy_f32x4 = vec_mul(y_f32x4, y_f32x4);
        nk_vf32x4_t const zz_f32x4 = vec_mul(z_f32x4, z_f32x4);
        largest_f32x4 = vec_max(largest_f32x4, vec_add(vec_add(xx_f32x4, yy_f32x4), zz_f32x4));
        moments_f32x4[0] = vec_add(moments_f32x4[0], xx_f32x4);
        moments_f32x4[1] = vec_madd(y_f32x4, x_f32x4, moments_f32x4[1]);
        moments_f32x4[2] = vec_add(moments_f32x4[2], yy_f32x4);
        moments_f32x4[3] = vec_madd(z_f32x4, x_f32x4, moments_f32x4[3]);
        moments_f32x4[4] = vec_madd(z_f32x4, y_f32x4, moments_f32x4[4]);
        moments_f32x4[5] = vec_add(moments_f32x4[5], zz_f32x4);
    }
    nk_u16_t *padding_bits = (nk_u16_t *)packed_x;
    for (nk_size_t index = (count + 3) & ~(nk_size_t)3; index < axis_stride; ++index)
        for (int axis = 0; axis != 3; ++axis) padding_bits[axis * axis_stride + index] = 0;
    nk_f32_t moments[6];
    for (int entry = 0; entry != 6; ++entry)
        moments[entry] = nk_hsum_f32x4_powervsx_(moments_f32x4[entry]) / (nk_f32_t)count;
    nk_jacobi_eigenanalysis_f32_powervsx_(moments, quaternion);
    largest_f32x4 = vec_max(largest_f32x4, vec_sld(largest_f32x4, largest_f32x4, 8));
    largest_f32x4 = vec_max(largest_f32x4, vec_sld(largest_f32x4, largest_f32x4, 4));
    nk_shapes_close_pack_f32_serial_(header, count, centroid, moments,
                                     nk_sqrt_f32_powervsx_(vec_extract(largest_f32x4, 0)));
}

/** The next two @c f32 points of the @p left remaining, each axis in f64 lanes, with @p valid_u64x2
 *  over the lanes holding one. Padded lanes read zero. */
NUMKONG_INLINE void nk_shapes_load_points_f32_powervsx_(nk_f32_t const *points, nk_size_t left, nk_vf64x2_t *x_f64x2,
                                                        nk_vf64x2_t *y_f64x2, nk_vf64x2_t *z_f64x2,
                                                        nk_vu64x2_t *valid_u64x2) {
    // Lanes 0 to 3 hold [x0 y0 z0 x1] and 4 to 5 hold [y1 z1]
    nk_vu8x16_t const x_bytes_u8x16 = {0, 1, 2, 3, 12, 13, 14, 15, 0, 1, 2, 3, 12, 13, 14, 15};
    nk_vu8x16_t const y_bytes_u8x16 = {4, 5, 6, 7, 16, 17, 18, 19, 4, 5, 6, 7, 16, 17, 18, 19};
    nk_vu8x16_t const z_bytes_u8x16 = {8, 9, 10, 11, 20, 21, 22, 23, 8, 9, 10, 11, 20, 21, 22, 23};
    nk_vu64x2_t const lanes_u64x2 = {0, 1};
    nk_size_t const points_count = left < 2 ? left : 2, bytes = points_count * 3 * sizeof(nk_f32_t);
    // `lxvl` takes at most 16 bytes, however many the length asks for
    nk_vf32x4_t const front_f32x4 = vec_xl_len((nk_f32_t *)points, bytes);
    nk_vf32x4_t const back_f32x4 = vec_xl_len((nk_f32_t *)points + 4, bytes > 16 ? bytes - 16 : 0);
    *x_f64x2 = vec_doubleh(vec_perm(front_f32x4, back_f32x4, x_bytes_u8x16));
    *y_f64x2 = vec_doubleh(vec_perm(front_f32x4, back_f32x4, y_bytes_u8x16));
    *z_f64x2 = vec_doubleh(vec_perm(front_f32x4, back_f32x4, z_bytes_u8x16));
    *valid_u64x2 = (nk_vu64x2_t)vec_cmplt(lanes_u64x2, vec_splats((nk_u64_t)points_count));
}

/** The moments of a cloud of @c f32 points as @ref nk_shapes_moments_f32_serial_ defines them, two
 *  points per step. */
NUMKONG_INLINE nk_f64_t nk_shapes_moments_f32_powervsx_(nk_f32_t const *points, nk_size_t count, nk_f64_t *centroid,
                                                        nk_f64_t *covariance) {
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    nk_vf64x2_t x_f64x2, y_f64x2, z_f64x2;
    nk_vu64x2_t valid_u64x2;
    nk_vf64x2_t const first_x_f64x2 = vec_splats((nk_f64_t)points[0]), first_y_f64x2 = vec_splats((nk_f64_t)points[1]);
    nk_vf64x2_t const first_z_f64x2 = vec_splats((nk_f64_t)points[2]);
    nk_vf64x2_t sum_x_f64x2 = vec_splats(0.0), sum_y_f64x2 = sum_x_f64x2, sum_z_f64x2 = sum_x_f64x2;
    nk_vu64x2_t coincident_u64x2 = vec_splats(~(nk_u64_t)0);
    for (nk_size_t index = 0; index < count; index += 2) {
        nk_shapes_load_points_f32_powervsx_(points + index * 3, count - index, &x_f64x2, &y_f64x2, &z_f64x2,
                                            &valid_u64x2);
        sum_x_f64x2 = vec_add(sum_x_f64x2, x_f64x2), sum_y_f64x2 = vec_add(sum_y_f64x2, y_f64x2);
        sum_z_f64x2 = vec_add(sum_z_f64x2, z_f64x2);
        nk_vu64x2_t const same_u64x2 = (nk_vu64x2_t)vec_and(
            vec_and(vec_cmpeq(x_f64x2, first_x_f64x2), vec_cmpeq(y_f64x2, first_y_f64x2)),
            vec_cmpeq(z_f64x2, first_z_f64x2));
        coincident_u64x2 = vec_and(coincident_u64x2, vec_orc(same_u64x2, valid_u64x2));
    }
    nk_f64_t const points_count = (nk_f64_t)count;
    centroid[0] = nk_hsum_f64x2_powervsx_(sum_x_f64x2) / points_count;
    centroid[1] = nk_hsum_f64x2_powervsx_(sum_y_f64x2) / points_count;
    centroid[2] = nk_hsum_f64x2_powervsx_(sum_z_f64x2) / points_count;
    nk_vf64x2_t const centroid_x_f64x2 = vec_splats(centroid[0]), centroid_y_f64x2 = vec_splats(centroid[1]);
    nk_vf64x2_t const centroid_z_f64x2 = vec_splats(centroid[2]);
    nk_vf64x2_t moments_f64x2[6];
    for (int entry = 0; entry != 6; ++entry) moments_f64x2[entry] = vec_splats(0.0);
    for (nk_size_t index = 0; index < count; index += 2) {
        nk_shapes_load_points_f32_powervsx_(points + index * 3, count - index, &x_f64x2, &y_f64x2, &z_f64x2,
                                            &valid_u64x2);
        x_f64x2 = vec_sel(x_f64x2, vec_sub(x_f64x2, centroid_x_f64x2), valid_u64x2);
        y_f64x2 = vec_sel(y_f64x2, vec_sub(y_f64x2, centroid_y_f64x2), valid_u64x2);
        z_f64x2 = vec_sel(z_f64x2, vec_sub(z_f64x2, centroid_z_f64x2), valid_u64x2);
        moments_f64x2[0] = vec_madd(x_f64x2, x_f64x2, moments_f64x2[0]);
        moments_f64x2[1] = vec_madd(y_f64x2, x_f64x2, moments_f64x2[1]);
        moments_f64x2[2] = vec_madd(y_f64x2, y_f64x2, moments_f64x2[2]);
        moments_f64x2[3] = vec_madd(z_f64x2, x_f64x2, moments_f64x2[3]);
        moments_f64x2[4] = vec_madd(z_f64x2, y_f64x2, moments_f64x2[4]);
        moments_f64x2[5] = vec_madd(z_f64x2, z_f64x2, moments_f64x2[5]);
    }
    for (int entry = 0; entry != 6; ++entry)
        covariance[entry] = nk_hsum_f64x2_powervsx_(moments_f64x2[entry]) / points_count;
    return vec_all_ne(coincident_u64x2, vec_splats((nk_u64_t)0))
               ? 0
               : nk_sqrt_f64_powervsx_(covariance[0] + covariance[2] + covariance[5]);
}

/** @copydoc nk_shapes_pack_f16_powervsx_ */
NUMKONG_INLINE void nk_shapes_pack_f32_powervsx_(nk_f32_t const *points, nk_size_t count, nk_f64_t const *centroid,
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
    nk_f64_t spreads[6] = {covariance[0], covariance[1], covariance[2], covariance[3], covariance[4], covariance[5]};
    nk_f64_t quaternion[4], basis[9], axes[9];
    int upper_tied, lower_tied;
    nk_jacobi_eigenanalysis_f64_powervsx_(spreads, quaternion);
    nk_shapes_sorted_basis_f64_serial_(spreads, quaternion, 0x1p-16, basis, &upper_tied, &lower_tied);
    if (upper_tied && lower_tied) nk_shapes_pick_axis_f32_serial_(points, count, centroid, basis);
    else if (upper_tied || lower_tied) nk_shapes_turn_plane_f32_serial_(points, count, centroid, basis);
    nk_shapes_frame_f64_serial_(basis, upper_tied, header->frame, axes);
    for (int cell = 0; cell != 9; ++cell) axes[cell] /= scale;
    nk_vf64x2_t const centroid_x_f64x2 = vec_splats(centroid[0]), centroid_y_f64x2 = vec_splats(centroid[1]);
    nk_vf64x2_t const centroid_z_f64x2 = vec_splats(centroid[2]);
    nk_vf64x2_t moments_f64x2[6], largest_f64x2 = vec_splats(0.0);
    for (int entry = 0; entry != 6; ++entry) moments_f64x2[entry] = vec_splats(0.0);
    nk_vf64x2_t x_f64x2, y_f64x2, z_f64x2, stored_f64x2[3];
    nk_vu64x2_t valid_u64x2;
    for (nk_size_t index = 0; index < count; index += 2) {
        nk_shapes_load_points_f32_powervsx_(points + index * 3, count - index, &x_f64x2, &y_f64x2, &z_f64x2,
                                            &valid_u64x2);
        x_f64x2 = vec_sub(x_f64x2, centroid_x_f64x2), y_f64x2 = vec_sub(y_f64x2, centroid_y_f64x2);
        z_f64x2 = vec_sub(z_f64x2, centroid_z_f64x2);
        for (int axis = 0; axis != 3; ++axis) {
            nk_vf64x2_t const unmasked_f64x2 = vec_madd(
                z_f64x2, vec_splats(axes[axis * 3 + 2]),
                vec_madd(y_f64x2, vec_splats(axes[axis * 3 + 1]), vec_mul(x_f64x2, vec_splats(axes[axis * 3]))));
            nk_vf64x2_t const rotated_f64x2 = vec_sel(vec_splats(0.0), unmasked_f64x2, valid_u64x2);
            nk_vf32x4_t const stored_f32x4 = vec_float2(rotated_f64x2, rotated_f64x2);
            vec_xst_len(stored_f32x4, packed_x + axis * axis_stride + index, 8);
            // The spreads describe the stored points, so the sweep measures them after rounding
            stored_f64x2[axis] = vec_doubleh(stored_f32x4);
        }
        x_f64x2 = stored_f64x2[0], y_f64x2 = stored_f64x2[1], z_f64x2 = stored_f64x2[2];
        nk_vf64x2_t const xx_f64x2 = vec_mul(x_f64x2, x_f64x2), yy_f64x2 = vec_mul(y_f64x2, y_f64x2);
        nk_vf64x2_t const zz_f64x2 = vec_mul(z_f64x2, z_f64x2);
        largest_f64x2 = vec_max(largest_f64x2, vec_add(vec_add(xx_f64x2, yy_f64x2), zz_f64x2));
        moments_f64x2[0] = vec_add(moments_f64x2[0], xx_f64x2);
        moments_f64x2[1] = vec_madd(y_f64x2, x_f64x2, moments_f64x2[1]);
        moments_f64x2[2] = vec_add(moments_f64x2[2], yy_f64x2);
        moments_f64x2[3] = vec_madd(z_f64x2, x_f64x2, moments_f64x2[3]);
        moments_f64x2[4] = vec_madd(z_f64x2, y_f64x2, moments_f64x2[4]);
        moments_f64x2[5] = vec_add(moments_f64x2[5], zz_f64x2);
    }
    for (nk_size_t index = (count + 1) & ~(nk_size_t)1; index < axis_stride; ++index)
        for (int axis = 0; axis != 3; ++axis) packed_x[axis * axis_stride + index] = 0;
    nk_f64_t moments[6];
    for (int entry = 0; entry != 6; ++entry)
        moments[entry] = nk_hsum_f64x2_powervsx_(moments_f64x2[entry]) / (nk_f64_t)count;
    nk_jacobi_eigenanalysis_f64_powervsx_(moments, quaternion);
    largest_f64x2 = vec_max(largest_f64x2, vec_mergel(largest_f64x2, largest_f64x2));
    nk_shapes_close_pack_f64_serial_(header, count, centroid, moments,
                                     nk_sqrt_f64_powervsx_(vec_extract(largest_f64x2, 0)));
}

NUMKONG_API nk_status_t nk_shapes_pack_f16_powervsx(nk_f16_t const *points, nk_size_t count, void *packed,
                                                    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t centroid[3], covariance[6];
    nk_shapes_moments_f16_powervsx_(points, count, centroid, covariance);
    nk_shapes_pack_f16_powervsx_(points, count, centroid, covariance, 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f16_powervsx(nk_f16_t const *points, nk_size_t count, void *packed,
                                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t centroid[3], covariance[6];
    nk_f32_t const radius = nk_shapes_moments_f16_powervsx_(points, count, centroid, covariance);
    nk_shapes_pack_f16_powervsx_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_f32_powervsx(nk_f32_t const *points, nk_size_t count, void *packed,
                                                    nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t centroid[3], covariance[6];
    nk_shapes_moments_f32_powervsx_(points, count, centroid, covariance);
    nk_shapes_pack_f32_powervsx_(points, count, centroid, covariance, 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f32_powervsx(nk_f32_t const *points, nk_size_t count, void *packed,
                                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t centroid[3], covariance[6];
    nk_f64_t const radius = nk_shapes_moments_f32_powervsx_(points, count, centroid, covariance);
    nk_shapes_pack_f32_powervsx_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f16_powervsx(void const *const *a_packed, nk_size_t rows,
                                                        void const *const *b_packed, nk_size_t columns,
                                                        nk_f32_t *results, nk_size_t results_stride,
                                                        nk_f32_t *rotations, nk_f32_t const *limit, void *workspace,
                                                        nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t const pair_limit = *limit;
    nk_f32_t chamfers[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_status_t const status = nk_shapes_chamfer_flips_f16_powervsx_(
                a_packed[row], b_packed[column], pair_limit, workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f32_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_powervsx(void const *const *a_packed, nk_size_t rows,
                                                          void const *const *b_packed, nk_size_t columns,
                                                          nk_f32_t *results, nk_size_t results_stride,
                                                          nk_f32_t *rotations, nk_f32_t const *limit,
                                                          nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t const pair_limit = *limit;
    nk_f32_t hausdorffs[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_hausdorff_flips_f16_powervsx_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f32_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f32_powervsx(void const *const *a_packed, nk_size_t rows,
                                                        void const *const *b_packed, nk_size_t columns,
                                                        nk_f64_t *results, nk_size_t results_stride,
                                                        nk_f32_t *rotations, nk_f64_t const *limit, void *workspace,
                                                        nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t const pair_limit = *limit;
    nk_f64_t chamfers[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_status_t const status = nk_shapes_chamfer_flips_f32_powervsx_(
                a_packed[row], b_packed[column], pair_limit, workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f64_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_powervsx(void const *const *a_packed, nk_size_t rows,
                                                          void const *const *b_packed, nk_size_t columns,
                                                          nk_f64_t *results, nk_size_t results_stride,
                                                          nk_f32_t *rotations, nk_f64_t const *limit,
                                                          nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t const pair_limit = *limit;
    nk_f64_t hausdorffs[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_hausdorff_flips_f32_powervsx_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f64_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_powervsx(void const *const *a_packed, nk_size_t rows,
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
                results_row[column] = nk_sqrt_f32_serial_(nk_shapes_wasserstein_procrustes_f16_powervsx_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F32_INF;
            if (rotations) nk_shapes_pair_rotation_f32_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_powervsx(void const *const *a_packed, nk_size_t rows,
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
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f16_powervsx_(
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

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_powervsx(void const *const *a_packed, nk_size_t rows,
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
                results_row[column] = nk_sqrt_f64_serial_(nk_shapes_wasserstein_procrustes_f32_powervsx_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F64_INF;
            if (rotations) nk_shapes_pair_rotation_f64_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_powervsx(void const *const *a_packed, nk_size_t rows,
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
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f32_powervsx_(
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
NUMKONG_INLINE nk_f32_t nk_rmsds_norm_f16_powervsx_(void const *packed) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)packed)->points;
    nk_size_t axis_stride;
    nk_f16_t const *axes = nk_shapes_packed_axes_f16_serial_(packed, &axis_stride);
    nk_vf32x4_t norm_f32x4 = vec_splats(0.0f);
    for (int axis = 0; axis != 3; ++axis)
        for (nk_size_t index = 0; index < count; index += 4) {
            nk_vf32x4_t const values_f32x4 = nk_load_f16x4_as_f32_powervsx_(axes + axis * axis_stride + index);
            norm_f32x4 = vec_madd(values_f32x4, values_f32x4, norm_f32x4);
        }
    return nk_hsum_f32x4_powervsx_(norm_f32x4);
}

/** Superposed RMSD of one pair of @c f16 packs with Ga + Gb in @p norms_squared, M in @c f32. */
NUMKONG_INLINE nk_f32_t nk_rmsds_pair_f16_powervsx_(void const *row_packed, void const *column_packed,
                                                    nk_f32_t norms_squared) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f16_t const *a_axes = nk_shapes_packed_axes_f16_serial_(row_packed, &axis_stride);
    nk_f16_t const *b_axes = nk_shapes_packed_axes_f16_serial_(column_packed, &axis_stride);
    nk_vf32x4_t sums_f32x4[9], a_f32x4[3], b_f32x4[3];
    for (int cell = 0; cell != 9; ++cell) sums_f32x4[cell] = vec_splats(0.0f);
    for (nk_size_t index = 0; index < count; index += 4) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f32x4[axis] = nk_load_f16x4_as_f32_powervsx_(a_axes + axis * axis_stride + index);
            b_f32x4[axis] = nk_load_f16x4_as_f32_powervsx_(b_axes + axis * axis_stride + index);
        }
        for (int row = 0; row != 3; ++row)
            for (int column = 0; column != 3; ++column)
                sums_f32x4[row * 3 + column] = vec_madd(a_f32x4[row], b_f32x4[column], sums_f32x4[row * 3 + column]);
    }
    nk_f32_t cross_covariance[9];
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = nk_hsum_f32x4_powervsx_(sums_f32x4[cell]);
    nk_f32_t const residual = nk_rmsds_folded_f32_serial_(cross_covariance, norms_squared);
    return nk_sqrt_f32_powervsx_(residual / (nk_f32_t)count);
}

/** Superposed RMSD of one pair of @c f32 packs: H in @c f64, then the residuals in plain @c f64,
 *  2 points per step through the zero padding. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f32_powervsx_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f32_t *a_axes = (nk_f32_t *)nk_shapes_packed_axes_f32_serial_(row_packed, &axis_stride);
    nk_f32_t *b_axes = (nk_f32_t *)nk_shapes_packed_axes_f32_serial_(column_packed, &axis_stride);
    nk_vf64x2_t sums_f64x2[9], a_f64x2[3], b_f64x2[3];
    for (int cell = 0; cell != 9; ++cell) sums_f64x2[cell] = vec_splats(0.0);
    for (nk_size_t index = 0; index < count; index += 2) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f64x2[axis] = vec_doubleh(vec_xl_len(a_axes + axis * axis_stride + index, 8));
            b_f64x2[axis] = vec_doubleh(vec_xl_len(b_axes + axis * axis_stride + index, 8));
        }
        for (int column = 0; column != 3; ++column)
            for (int row = 0; row != 3; ++row)
                sums_f64x2[column * 3 + row] = vec_madd(b_f64x2[column], a_f64x2[row], sums_f64x2[column * 3 + row]);
    }
    nk_f64_t cross_covariance[9], rotation[9];
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = nk_hsum_f64x2_powervsx_(sums_f64x2[cell]);
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    nk_vf64x2_t rotation_f64x2[9];
    for (int cell = 0; cell != 9; ++cell) rotation_f64x2[cell] = vec_splats(rotation[cell]);
    nk_vf64x2_t sum_f64x2 = vec_splats(0.0);
    for (nk_size_t index = 0; index < count; index += 2) {
        for (int axis = 0; axis != 3; ++axis)
            b_f64x2[axis] = vec_doubleh(vec_xl_len(b_axes + axis * axis_stride + index, 8));
        for (int axis = 0; axis != 3; ++axis) {
            nk_vf64x2_t residual_f64x2 = vec_doubleh(vec_xl_len(a_axes + axis * axis_stride + index, 8));
            for (int column = 0; column != 3; ++column)
                residual_f64x2 = vec_nmsub(rotation_f64x2[axis * 3 + column], b_f64x2[column], residual_f64x2);
            sum_f64x2 = vec_madd(residual_f64x2, residual_f64x2, sum_f64x2);
        }
    }
    return nk_sqrt_f64_powervsx_(nk_hsum_f64x2_powervsx_(sum_f64x2) / (nk_f64_t)count);
}

/** Superposed RMSD of one pair of @c f64 packs, like the @c f32 pair with H and the residuals
 *  compensated, 2 points per step through the zero padding. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f64_powervsx_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f64_t const *a_axes = nk_shapes_packed_axes_f64_serial_(row_packed, &axis_stride);
    nk_f64_t const *b_axes = nk_shapes_packed_axes_f64_serial_(column_packed, &axis_stride);
    nk_vf64x2_t sums_f64x2[9], compensations_f64x2[9], a_f64x2[3], b_f64x2[3];
    for (int cell = 0; cell != 9; ++cell)
        sums_f64x2[cell] = vec_splats(0.0), compensations_f64x2[cell] = vec_splats(0.0);
    for (nk_size_t index = 0; index < count; index += 2) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f64x2[axis] = vec_xl(0, a_axes + axis * axis_stride + index);
            b_f64x2[axis] = vec_xl(0, b_axes + axis * axis_stride + index);
        }
        for (int column = 0; column != 3; ++column)
            for (int row = 0; row != 3; ++row)
                nk_dot2_f64x2_powervsx_(&sums_f64x2[column * 3 + row], &compensations_f64x2[column * 3 + row],
                                        b_f64x2[column], a_f64x2[row]);
    }
    nk_f64_t cross_covariance[9], rotation[9];
    for (int cell = 0; cell != 9; ++cell)
        cross_covariance[cell] = nk_dot_stable_sum_f64x2_powervsx_(sums_f64x2[cell], compensations_f64x2[cell]);
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    nk_vf64x2_t rotation_f64x2[9];
    for (int cell = 0; cell != 9; ++cell) rotation_f64x2[cell] = vec_splats(rotation[cell]);
    nk_vf64x2_t sum_f64x2 = vec_splats(0.0), compensation_f64x2 = vec_splats(0.0);
    for (nk_size_t index = 0; index < count; index += 2) {
        for (int axis = 0; axis != 3; ++axis) b_f64x2[axis] = vec_xl(0, b_axes + axis * axis_stride + index);
        for (int axis = 0; axis != 3; ++axis) {
            nk_vf64x2_t residual_f64x2 = vec_xl(0, a_axes + axis * axis_stride + index);
            for (int column = 0; column != 3; ++column)
                residual_f64x2 = vec_nmsub(rotation_f64x2[axis * 3 + column], b_f64x2[column], residual_f64x2);
            nk_dot2_f64x2_powervsx_(&sum_f64x2, &compensation_f64x2, residual_f64x2, residual_f64x2);
        }
    }
    return nk_sqrt_f64_powervsx_(nk_dot_stable_sum_f64x2_powervsx_(sum_f64x2, compensation_f64x2) / (nk_f64_t)count);
}

NUMKONG_API nk_status_t nk_rmsds_packed_f64_powervsx(void const *const *a_packed, nk_size_t rows,
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
            results_row[column] = nk_rmsds_pair_f64_powervsx_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f32_powervsx(void const *const *a_packed, nk_size_t rows,
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
            results_row[column] = nk_rmsds_pair_f32_powervsx_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f16_powervsx(void const *const *a_packed, nk_size_t rows,
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
            row_norms[row - row_begin] = nk_rmsds_norm_f16_powervsx_(a_packed[row]);
        for (nk_size_t column_begin = 0; column_begin < columns; column_begin += nk_rmsds_tile_frames_k) {
            nk_size_t const column_end = column_begin + nk_rmsds_tile_frames_k < columns
                                             ? column_begin + nk_rmsds_tile_frames_k
                                             : columns;
            for (nk_size_t column = column_begin; column != column_end; ++column)
                column_norms[column - column_begin] = nk_rmsds_norm_f16_powervsx_(b_packed[column]);
            for (nk_size_t row = row_begin; row != row_end; ++row) {
                nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
                for (nk_size_t column = column_begin; column != column_end; ++column)
                    results_row[column] = nk_rmsds_pair_f16_powervsx_(
                        a_packed[row], b_packed[column],
                        row_norms[row - row_begin] + column_norms[column - column_begin]);
            }
        }
    }
    return nk_success_k;
}

#endif // NUMKONG_TARGET_POWERVSX

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_PPC64_POWERVSX_
#endif // NUMKONG_ARCH_PPC64_
#endif // NUMKONG_SHAPES_POWERVSX_H
