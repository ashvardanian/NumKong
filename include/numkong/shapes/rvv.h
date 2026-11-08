/**
 *  @file include/numkong/shapes/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Many-to-many shape distances between 3D point clouds for RISC-V Vector, RVV.
 *
 *  @sa c/numkong/shapes.h
 *
 *  Chamfer and Hausdorff broadcast one outer point and sweep the inner cloud a register at a time,
 *  keeping the running minima of all 8 flips in a tuple of 8 registers at LMUL 1. A step's minima,
 *  squares and reverse minima take 16 register groups, which spill at LMUL 2 and leave half of the
 *  register file free at LMUL 1. Lanes past the last point stay infinite, so they never win a
 *  minimum, and the reverse minima take only the part that holds points.
 *
 *  - Chamfer scores both directions in that one sweep. It widens the inner cloud once per pair into
 *    the 3 workspace axes past the reverse minima, so a step neither converts nor changes the
 *    vector type
 *  - Hausdorff has no workspace, so each load widens into a register pair, @c f16 by bit
 *    manipulation since the base V extension has no @c f16 arithmetic or conversions, and it runs
 *    each direction until every flip passes the limit, reading only the points that decide a pair
 *
 *  Packing splits points with segment loads and measures what it stores in the same sweep.
 *
 *  The transport kinds run the serial simplex around tile pricing a register group of arcs at a
 *  time. The scalar cost behind the tree's potentials runs the pricing lanes at a vector length of
 *  1, so tree arcs keep a zero reduced cost.
 *
 *  RMSD steps by exact vector lengths, so at any VLEN it reads no padding, and it names each sum,
 *  as register groups cannot form arrays. The SVD stays scalar.
 *
 *  @section shapes_rvv_instructions Key RVV Shape Instructions
 *
 *  @verbatim
 *  Intrinsic                               Purpose
 *  __riscv_vfwcvt_f_f_v_f64m2              Widens f32 coordinates to f64
 *  __riscv_vlseg3e32_v_f32m1x3             Splits interleaved points into axes for packing
 *  __riscv_vfrsub_vf_f32m1                 Outer coordinate minus inner ones
 *  __riscv_vfmin_vv_f32m1                  Running minimum of one flip
 *  __riscv_vfredmin_vs_f32m1_f32m1         Folds one flip's minima onto a scalar
 *  __riscv_vfsqrt_v_f64m4                  Distances of all 8 flips at once
 *  __riscv_vmfeq_vf_f64m2_b32              Finds the entering arc's lane in a priced block
 *  __riscv_vfirst_m_b32                    First lane reaching the block's least reduced cost
 *  __riscv_vfmacc_vv_f64m2                 Arc costs in fused multiply-adds
 *  @endverbatim
 */
#ifndef NUMKONG_SHAPES_RVV_H
#define NUMKONG_SHAPES_RVV_H

#if NUMKONG_ARCH_RISCV64_
#if NUMKONG_ARCH_RISCV64_RVV_
#if NUMKONG_TARGET_RVV

#include "numkong/types.h"
#include "numkong/cast/rvv.h"      // `nk_f16m1_to_f32m2_rvv_`
#include "numkong/scalar/rvv.h"    // `nk_sqrt_f32_rvv_`, `nk_rsqrt_f32_rvv_`
#include "numkong/shapes/serial.h" // `nk_shapes_simplex_next_tile_serial_`, `nk_shapes_workspace_f64_serial_t`
#include "numkong/dot/rvv.h"       // `nk_dot2_f64m1_rvv_`, `nk_dot_stable_sum_f64m1_rvv_`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/** The parts of the 8 flips' squared distances from one outer point to @p vector_length inner
 *  points: flip f adds part f & 3, its planar sum, to part 4 + (f >> 2), its z square. */
NUMKONG_INLINE vfloat32m1x6_t nk_shapes_flip_parts_f32m1_rvv_(nk_f32_t outer_x, nk_f32_t outer_y, nk_f32_t outer_z,
                                                              vfloat32m1_t inner_x_f32m1, vfloat32m1_t inner_y_f32m1,
                                                              vfloat32m1_t inner_z_f32m1, nk_size_t vector_length) {
    // A flip negates an inner axis, turning its difference into a sum
    vfloat32m1_t const minus_x_f32m1 = __riscv_vfrsub_vf_f32m1(inner_x_f32m1, outer_x, vector_length);
    vfloat32m1_t const plus_x_f32m1 = __riscv_vfadd_vf_f32m1(inner_x_f32m1, outer_x, vector_length);
    vfloat32m1_t const minus_y_f32m1 = __riscv_vfrsub_vf_f32m1(inner_y_f32m1, outer_y, vector_length);
    vfloat32m1_t const plus_y_f32m1 = __riscv_vfadd_vf_f32m1(inner_y_f32m1, outer_y, vector_length);
    vfloat32m1_t const minus_z_f32m1 = __riscv_vfrsub_vf_f32m1(inner_z_f32m1, outer_z, vector_length);
    vfloat32m1_t const plus_z_f32m1 = __riscv_vfadd_vf_f32m1(inner_z_f32m1, outer_z, vector_length);
    vfloat32m1_t const minus_xx_f32m1 = __riscv_vfmul_vv_f32m1(minus_x_f32m1, minus_x_f32m1, vector_length);
    vfloat32m1_t const plus_xx_f32m1 = __riscv_vfmul_vv_f32m1(plus_x_f32m1, plus_x_f32m1, vector_length);
    vfloat32m1_t const minus_yy_f32m1 = __riscv_vfmul_vv_f32m1(minus_y_f32m1, minus_y_f32m1, vector_length);
    vfloat32m1_t const plus_yy_f32m1 = __riscv_vfmul_vv_f32m1(plus_y_f32m1, plus_y_f32m1, vector_length);
    return __riscv_vcreate_v_f32m1x6(__riscv_vfadd_vv_f32m1(minus_xx_f32m1, minus_yy_f32m1, vector_length),
                                     __riscv_vfadd_vv_f32m1(plus_xx_f32m1, minus_yy_f32m1, vector_length),
                                     __riscv_vfadd_vv_f32m1(minus_xx_f32m1, plus_yy_f32m1, vector_length),
                                     __riscv_vfadd_vv_f32m1(plus_xx_f32m1, plus_yy_f32m1, vector_length),
                                     __riscv_vfmul_vv_f32m1(minus_z_f32m1, minus_z_f32m1, vector_length),
                                     __riscv_vfmul_vv_f32m1(plus_z_f32m1, plus_z_f32m1, vector_length));
}

/** Lowers the running minima of the 8 flips by one outer point's squared distances to a register
 *  of inner points. */
NUMKONG_INLINE vfloat32m1x8_t nk_shapes_lower_nearest_f32m1_rvv_(vfloat32m1x8_t nearest_f32m1x8, nk_f32_t outer_x,
                                                                 nk_f32_t outer_y, nk_f32_t outer_z,
                                                                 vfloat32m1_t inner_x_f32m1, vfloat32m1_t inner_y_f32m1,
                                                                 vfloat32m1_t inner_z_f32m1, nk_size_t vector_length) {
    vfloat32m1x6_t const parts_f32m1x6 = nk_shapes_flip_parts_f32m1_rvv_(outer_x, outer_y, outer_z, inner_x_f32m1,
                                                                         inner_y_f32m1, inner_z_f32m1, vector_length);
    vfloat32m1_t const planar_0_f32m1 = __riscv_vget_v_f32m1x6_f32m1(parts_f32m1x6, 0);
    vfloat32m1_t const planar_1_f32m1 = __riscv_vget_v_f32m1x6_f32m1(parts_f32m1x6, 1);
    vfloat32m1_t const planar_2_f32m1 = __riscv_vget_v_f32m1x6_f32m1(parts_f32m1x6, 2);
    vfloat32m1_t const planar_3_f32m1 = __riscv_vget_v_f32m1x6_f32m1(parts_f32m1x6, 3);
    vfloat32m1_t const minus_zz_f32m1 = __riscv_vget_v_f32m1x6_f32m1(parts_f32m1x6, 4);
    vfloat32m1_t const plus_zz_f32m1 = __riscv_vget_v_f32m1x6_f32m1(parts_f32m1x6, 5);
    return __riscv_vcreate_v_f32m1x8(
        __riscv_vfmin_vv_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 0),
                               __riscv_vfadd_vv_f32m1(planar_0_f32m1, minus_zz_f32m1, vector_length), vector_length),
        __riscv_vfmin_vv_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 1),
                               __riscv_vfadd_vv_f32m1(planar_1_f32m1, minus_zz_f32m1, vector_length), vector_length),
        __riscv_vfmin_vv_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 2),
                               __riscv_vfadd_vv_f32m1(planar_2_f32m1, minus_zz_f32m1, vector_length), vector_length),
        __riscv_vfmin_vv_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 3),
                               __riscv_vfadd_vv_f32m1(planar_3_f32m1, minus_zz_f32m1, vector_length), vector_length),
        __riscv_vfmin_vv_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 4),
                               __riscv_vfadd_vv_f32m1(planar_0_f32m1, plus_zz_f32m1, vector_length), vector_length),
        __riscv_vfmin_vv_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 5),
                               __riscv_vfadd_vv_f32m1(planar_1_f32m1, plus_zz_f32m1, vector_length), vector_length),
        __riscv_vfmin_vv_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 6),
                               __riscv_vfadd_vv_f32m1(planar_2_f32m1, plus_zz_f32m1, vector_length), vector_length),
        __riscv_vfmin_vv_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 7),
                               __riscv_vfadd_vv_f32m1(planar_3_f32m1, plus_zz_f32m1, vector_length), vector_length));
}

/** Lowers one flip's running minima by its @p squares_f32m1 , and the first @p reverse_length of
 *  them into the inner points' @p reverse minima, which a last group fills only part of. */
NUMKONG_INLINE vfloat32m1_t nk_shapes_lower_flip_f32m1_rvv_(vfloat32m1_t nearest_f32m1, vfloat32m1_t squares_f32m1,
                                                            nk_f32_t *reverse, nk_size_t vector_length,
                                                            nk_size_t reverse_length) {
    vfloat32m1_t const reverse_f32m1 = __riscv_vle32_v_f32m1(reverse, reverse_length);
    __riscv_vse32_v_f32m1(reverse, __riscv_vfmin_vv_f32m1(reverse_f32m1, squares_f32m1, reverse_length),
                          reverse_length);
    return __riscv_vfmin_vv_f32m1(nearest_f32m1, squares_f32m1, vector_length);
}

/** Lowers the 8 flips' nearest squares as @ref nk_shapes_lower_nearest_f32m1_rvv_ does, and the
 *  same squares into the inner points' @p reverse_minima , 8 axes @p reverse_stride apart. */
NUMKONG_INLINE vfloat32m1x8_t nk_shapes_lower_mutual_f32m1_rvv_(vfloat32m1x8_t nearest_f32m1x8, nk_f32_t outer_x,
                                                                nk_f32_t outer_y, nk_f32_t outer_z,
                                                                vfloat32m1_t inner_x_f32m1, vfloat32m1_t inner_y_f32m1,
                                                                vfloat32m1_t inner_z_f32m1, nk_f32_t *reverse_minima,
                                                                nk_size_t reverse_stride, nk_size_t vector_length,
                                                                nk_size_t reverse_length) {
    vfloat32m1x6_t const parts_f32m1x6 = nk_shapes_flip_parts_f32m1_rvv_(outer_x, outer_y, outer_z, inner_x_f32m1,
                                                                         inner_y_f32m1, inner_z_f32m1, vector_length);
    vfloat32m1_t const planar_0_f32m1 = __riscv_vget_v_f32m1x6_f32m1(parts_f32m1x6, 0);
    vfloat32m1_t const planar_1_f32m1 = __riscv_vget_v_f32m1x6_f32m1(parts_f32m1x6, 1);
    vfloat32m1_t const planar_2_f32m1 = __riscv_vget_v_f32m1x6_f32m1(parts_f32m1x6, 2);
    vfloat32m1_t const planar_3_f32m1 = __riscv_vget_v_f32m1x6_f32m1(parts_f32m1x6, 3);
    vfloat32m1_t const minus_zz_f32m1 = __riscv_vget_v_f32m1x6_f32m1(parts_f32m1x6, 4);
    vfloat32m1_t const plus_zz_f32m1 = __riscv_vget_v_f32m1x6_f32m1(parts_f32m1x6, 5);
    return __riscv_vcreate_v_f32m1x8(
        nk_shapes_lower_flip_f32m1_rvv_(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 0),
                                        __riscv_vfadd_vv_f32m1(planar_0_f32m1, minus_zz_f32m1, vector_length),
                                        reverse_minima, vector_length, reverse_length),
        nk_shapes_lower_flip_f32m1_rvv_(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 1),
                                        __riscv_vfadd_vv_f32m1(planar_1_f32m1, minus_zz_f32m1, vector_length),
                                        reverse_minima + reverse_stride, vector_length, reverse_length),
        nk_shapes_lower_flip_f32m1_rvv_(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 2),
                                        __riscv_vfadd_vv_f32m1(planar_2_f32m1, minus_zz_f32m1, vector_length),
                                        reverse_minima + 2 * reverse_stride, vector_length, reverse_length),
        nk_shapes_lower_flip_f32m1_rvv_(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 3),
                                        __riscv_vfadd_vv_f32m1(planar_3_f32m1, minus_zz_f32m1, vector_length),
                                        reverse_minima + 3 * reverse_stride, vector_length, reverse_length),
        nk_shapes_lower_flip_f32m1_rvv_(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 4),
                                        __riscv_vfadd_vv_f32m1(planar_0_f32m1, plus_zz_f32m1, vector_length),
                                        reverse_minima + 4 * reverse_stride, vector_length, reverse_length),
        nk_shapes_lower_flip_f32m1_rvv_(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 5),
                                        __riscv_vfadd_vv_f32m1(planar_1_f32m1, plus_zz_f32m1, vector_length),
                                        reverse_minima + 5 * reverse_stride, vector_length, reverse_length),
        nk_shapes_lower_flip_f32m1_rvv_(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 6),
                                        __riscv_vfadd_vv_f32m1(planar_2_f32m1, plus_zz_f32m1, vector_length),
                                        reverse_minima + 6 * reverse_stride, vector_length, reverse_length),
        nk_shapes_lower_flip_f32m1_rvv_(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 7),
                                        __riscv_vfadd_vv_f32m1(planar_3_f32m1, plus_zz_f32m1, vector_length),
                                        reverse_minima + 7 * reverse_stride, vector_length, reverse_length));
}

/** The least of every flip's nearest squares over its lanes, a flip per lane, reduced to scalars as
 *  RVV reductions write only one lane. */
NUMKONG_INLINE vfloat32m2_t nk_shapes_least_flips_f32m1_rvv_(vfloat32m1x8_t nearest_f32m1x8,
                                                             vfloat32m1_t infinity_f32m1, nk_size_t full_length) {
    nk_f32_t least[8];
    least[0] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m1_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 0), infinity_f32m1, full_length));
    least[1] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m1_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 1), infinity_f32m1, full_length));
    least[2] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m1_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 2), infinity_f32m1, full_length));
    least[3] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m1_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 3), infinity_f32m1, full_length));
    least[4] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m1_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 4), infinity_f32m1, full_length));
    least[5] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m1_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 5), infinity_f32m1, full_length));
    least[6] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m1_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 6), infinity_f32m1, full_length));
    least[7] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m1_f32m1(__riscv_vget_v_f32m1x8_f32m1(nearest_f32m1x8, 7), infinity_f32m1, full_length));
    // Every RVV register holds at least 128 bits, so a pair of them holds all 8 flips
    return __riscv_vle32_v_f32m2(least, 8);
}

/** One @c f16 outer point in f32, as the 3 lanes of a strided load across the axes. */
NUMKONG_INLINE void nk_shapes_outer_point_f16_rvv_(nk_f16_t const *outer_axes, nk_size_t outer_stride,
                                                   nk_size_t outer_index, nk_f32_t *outer_point) {
    vuint16m1_t const bits_u16m1 = __riscv_vlse16_v_u16m1((nk_u16_t const *)outer_axes + outer_index,
                                                          outer_stride * sizeof(nk_u16_t), 3);
    __riscv_vse32_v_f32m2(outer_point, nk_f16m1_to_f32m2_rvv_(bits_u16m1, 3), 3);
}

/** A register pair of @c f16 coordinates widened to f32. */
NUMKONG_INLINE vfloat32m2_t nk_shapes_load_axis_f16_rvv_(nk_u16_t const *bits, nk_size_t vector_length) {
    return nk_f16m1_to_f32m2_rvv_(__riscv_vle16_v_u16m1(bits, vector_length), vector_length);
}

/** One outer point's sweep of @c f16 inner points, each load widened to a register pair that runs
 *  one register after the other. Lanes past the last point stay infinite, so they never win. */
NUMKONG_INLINE vfloat32m1x8_t nk_shapes_sweep_nearest_f16_rvv_(nk_f32_t const *outer_point, nk_f16_t const *inner_axes,
                                                               nk_size_t inner_stride, nk_size_t inner_count,
                                                               vfloat32m1_t infinity_f32m1, nk_size_t full_length) {
    nk_u16_t const *inner_x = (nk_u16_t const *)inner_axes, *inner_y = inner_x + inner_stride;
    nk_u16_t const *inner_z = inner_y + inner_stride;
    nk_size_t const pair_length = 2 * full_length, pairs_end = inner_count & ~(pair_length - 1);
    nk_f32_t const outer_x = outer_point[0], outer_y = outer_point[1], outer_z = outer_point[2];
    vfloat32m1x8_t nearest_f32m1x8 = __riscv_vcreate_v_f32m1x8(infinity_f32m1, infinity_f32m1, infinity_f32m1,
                                                               infinity_f32m1, infinity_f32m1, infinity_f32m1,
                                                               infinity_f32m1, infinity_f32m1);
    for (nk_size_t inner_index = 0; inner_index != pairs_end; inner_index += pair_length) {
        vfloat32m2_t const pair_x_f32m2 = nk_shapes_load_axis_f16_rvv_(inner_x + inner_index, pair_length);
        vfloat32m2_t const pair_y_f32m2 = nk_shapes_load_axis_f16_rvv_(inner_y + inner_index, pair_length);
        vfloat32m2_t const pair_z_f32m2 = nk_shapes_load_axis_f16_rvv_(inner_z + inner_index, pair_length);
        nearest_f32m1x8 = nk_shapes_lower_nearest_f32m1_rvv_(
            nearest_f32m1x8, outer_x, outer_y, outer_z, __riscv_vget_v_f32m2_f32m1(pair_x_f32m2, 0),
            __riscv_vget_v_f32m2_f32m1(pair_y_f32m2, 0), __riscv_vget_v_f32m2_f32m1(pair_z_f32m2, 0), full_length);
        nearest_f32m1x8 = nk_shapes_lower_nearest_f32m1_rvv_(
            nearest_f32m1x8, outer_x, outer_y, outer_z, __riscv_vget_v_f32m2_f32m1(pair_x_f32m2, 1),
            __riscv_vget_v_f32m2_f32m1(pair_y_f32m2, 1), __riscv_vget_v_f32m2_f32m1(pair_z_f32m2, 1), full_length);
    }
    if (pairs_end != inner_count) {
        nk_size_t const left = inner_count - pairs_end;
        vfloat32m2_t const infinity_f32m2 = __riscv_vcreate_v_f32m1_f32m2(infinity_f32m1, infinity_f32m1);
        vfloat32m2_t const pair_x_f32m2 = __riscv_vmv_v_v_f32m2_tu(
            infinity_f32m2, nk_shapes_load_axis_f16_rvv_(inner_x + pairs_end, left), left);
        vfloat32m2_t const pair_y_f32m2 = __riscv_vmv_v_v_f32m2_tu(
            infinity_f32m2, nk_shapes_load_axis_f16_rvv_(inner_y + pairs_end, left), left);
        vfloat32m2_t const pair_z_f32m2 = __riscv_vmv_v_v_f32m2_tu(
            infinity_f32m2, nk_shapes_load_axis_f16_rvv_(inner_z + pairs_end, left), left);
        nearest_f32m1x8 = nk_shapes_lower_nearest_f32m1_rvv_(
            nearest_f32m1x8, outer_x, outer_y, outer_z, __riscv_vget_v_f32m2_f32m1(pair_x_f32m2, 0),
            __riscv_vget_v_f32m2_f32m1(pair_y_f32m2, 0), __riscv_vget_v_f32m2_f32m1(pair_z_f32m2, 0), full_length);
        if (left > full_length)
            nearest_f32m1x8 = nk_shapes_lower_nearest_f32m1_rvv_(
                nearest_f32m1x8, outer_x, outer_y, outer_z, __riscv_vget_v_f32m2_f32m1(pair_x_f32m2, 1),
                __riscv_vget_v_f32m2_f32m1(pair_y_f32m2, 1), __riscv_vget_v_f32m2_f32m1(pair_z_f32m2, 1), full_length);
    }
    return nearest_f32m1x8;
}

/** One outer point's sweep of the inner cloud widened into f32 @p inner_axes , lowering the reverse
 *  minima too. Lanes past the last point stay infinite, so they never win, and the reverse minima
 *  take only the part that holds points. */
NUMKONG_INLINE vfloat32m1x8_t nk_shapes_sweep_mutual_f32m1_rvv_(nk_f32_t const *outer_point, nk_f32_t const *inner_axes,
                                                                nk_size_t inner_count, nk_f32_t *reverse_minima,
                                                                nk_size_t reverse_stride, vfloat32m1_t infinity_f32m1,
                                                                nk_size_t full_length) {
    nk_size_t const groups_end = inner_count & ~(full_length - 1);
    nk_f32_t const outer_x = outer_point[0], outer_y = outer_point[1], outer_z = outer_point[2];
    nk_f32_t const *inner_x = inner_axes, *inner_y = inner_x + reverse_stride, *inner_z = inner_y + reverse_stride;
    vfloat32m1x8_t nearest_f32m1x8 = __riscv_vcreate_v_f32m1x8(infinity_f32m1, infinity_f32m1, infinity_f32m1,
                                                               infinity_f32m1, infinity_f32m1, infinity_f32m1,
                                                               infinity_f32m1, infinity_f32m1);
    for (nk_size_t inner_index = 0; inner_index != groups_end; inner_index += full_length)
        nearest_f32m1x8 = nk_shapes_lower_mutual_f32m1_rvv_(
            nearest_f32m1x8, outer_x, outer_y, outer_z, __riscv_vle32_v_f32m1(inner_x + inner_index, full_length),
            __riscv_vle32_v_f32m1(inner_y + inner_index, full_length),
            __riscv_vle32_v_f32m1(inner_z + inner_index, full_length), reverse_minima + inner_index, reverse_stride,
            full_length, full_length);
    if (groups_end != inner_count) {
        nk_size_t const left = inner_count - groups_end;
        nearest_f32m1x8 = nk_shapes_lower_mutual_f32m1_rvv_(
            nearest_f32m1x8, outer_x, outer_y, outer_z,
            __riscv_vle32_v_f32m1_tu(infinity_f32m1, inner_x + groups_end, left),
            __riscv_vle32_v_f32m1_tu(infinity_f32m1, inner_y + groups_end, left),
            __riscv_vle32_v_f32m1_tu(infinity_f32m1, inner_z + groups_end, left), reverse_minima + groups_end,
            reverse_stride, full_length, left);
    }
    return nearest_f32m1x8;
}

/** The roots of @p count reverse minima summed. */
NUMKONG_INLINE nk_f32_t nk_shapes_reverse_sum_f32_rvv_(nk_f32_t const *reverse, nk_size_t count) {
    vfloat32m1_t sum_f32m1 = __riscv_vfmv_s_f_f32m1(0, 1);
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e32m4(count - index);
        vfloat32m4_t const roots_f32m4 = __riscv_vfsqrt_v_f32m4(__riscv_vle32_v_f32m4(reverse + index, vector_length),
                                                                vector_length);
        sum_f32m1 = __riscv_vfredusum_vs_f32m4_f32m1(roots_f32m4, sum_f32m1, vector_length);
    }
    return __riscv_vfmv_f_s_f32m1_f32(sum_f32m1);
}

/** Fused Chamfer flips of @c f16 packs in f32 lanes. The 3 axes past the reverse minima hold the
 *  inner cloud widened once per pair, so the sweeps neither convert nor change the vector type. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f16_rvv_(void const *row_packed, void const *column_packed,
                                                            nk_f32_t limit, void *workspace, nk_size_t workspace_bytes,
                                                            nk_f32_t *chamfers) {
    nk_shapes_mutual_f16_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f16_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    nk_size_t const reverse_stride = views.reverse_stride, inner_count = views.inner_count;
    if (workspace_bytes < 11 * reverse_stride * sizeof(nk_f32_t)) return nk_unexpected_dimensions_k;
    nk_f32_t *reverse_minima = views.reverse_minima, *inner_axes = reverse_minima + 8 * reverse_stride;
    nk_u16_t const *inner_bits = (nk_u16_t const *)views.inner_axes;
    nk_size_t vector_length;
    for (nk_size_t index = 0; index < 8 * reverse_stride; index += vector_length) {
        vector_length = __riscv_vsetvl_e32m8(8 * reverse_stride - index);
        __riscv_vse32_v_f32m8(reverse_minima + index, __riscv_vfmv_v_f_f32m8(NUMKONG_F32_INF, vector_length),
                              vector_length);
    }
    for (nk_size_t axis = 0; axis != 3; ++axis)
        for (nk_size_t index = 0; index < inner_count; index += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(inner_count - index);
            __riscv_vse32_v_f32m2(
                inner_axes + axis * reverse_stride + index,
                nk_shapes_load_axis_f16_rvv_(inner_bits + axis * views.inner_stride + index, vector_length),
                vector_length);
        }
    nk_size_t const full_length = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t const infinity_f32m1 = __riscv_vfmv_v_f_f32m1(NUMKONG_F32_INF, full_length);
    nk_f32_t const outer_points = (nk_f32_t)views.outer_count, inner_points = (nk_f32_t)inner_count;
    nk_f32_t const threshold = 2 * limit * outer_points;
    vfloat32m2_t sums_f32m2 = __riscv_vfmv_v_f_f32m2(0, 8);
    nk_f32_t outer_point[3], sums[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        nk_shapes_outer_point_f16_rvv_(views.outer_axes, views.outer_stride, outer_index, outer_point);
        vfloat32m1x8_t const nearest_f32m1x8 = nk_shapes_sweep_mutual_f32m1_rvv_(
            outer_point, inner_axes, inner_count, reverse_minima, reverse_stride, infinity_f32m1, full_length);
        sums_f32m2 = __riscv_vfadd_vv_f32m2(
            sums_f32m2,
            __riscv_vfsqrt_v_f32m2(nk_shapes_least_flips_f32m1_rvv_(nearest_f32m1x8, infinity_f32m1, full_length), 8),
            8);
        if (__riscv_vcpop_m_b16(__riscv_vmfgt_vf_f32m2_b16(sums_f32m2, threshold, 8), 8) == 8) {
            __riscv_vse32_v_f32m2(sums, sums_f32m2, 8);
            if (nk_shapes_mutual_past_limit_f32_serial_(sums, outer_points, limit, chamfers)) return nk_success_k;
        }
    }
    __riscv_vse32_v_f32m2(sums, sums_f32m2, 8);
    for (int flip = 0; flip != 8; ++flip) {
        nk_f32_t const reverse_sum = nk_shapes_reverse_sum_f32_rvv_(reverse_minima + flip * reverse_stride,
                                                                    inner_count);
        chamfers[flip] = (sums[flip] / outer_points + reverse_sum / inner_points) / 2;
    }
    return nk_success_k;
}

/** Hausdorff flips of @c f16 packs in f32 lanes: a pair stops once every flip's farthest square
 *  passes @p limit , whose square rounds, so the distance confirms it. */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f16_rvv_(void const *row_packed, void const *column_packed,
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
    nk_size_t const full_length = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t const infinity_f32m1 = __riscv_vfmv_v_f_f32m1(NUMKONG_F32_INF, full_length);
    nk_f32_t const limit_squared = limit * limit;
    vfloat32m2_t farthest_f32m2 = __riscv_vfmv_v_f_f32m2(0, 8);
    nk_f32_t outer_point[3];
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_f16_t const *outer_axes = direction ? column_axes : row_axes;
        nk_f16_t const *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            nk_shapes_outer_point_f16_rvv_(outer_axes, outer_stride, outer_index, outer_point);
            vfloat32m1x8_t const nearest_f32m1x8 = nk_shapes_sweep_nearest_f16_rvv_(
                outer_point, inner_axes, inner_stride, inner_count, infinity_f32m1, full_length);
            farthest_f32m2 = __riscv_vfmax_vv_f32m2(
                farthest_f32m2, nk_shapes_least_flips_f32m1_rvv_(nearest_f32m1x8, infinity_f32m1, full_length), 8);
            nk_f32_t const least = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredmin_vs_f32m2_f32m1(farthest_f32m2, infinity_f32m1, 8));
            rejected = least > limit_squared && nk_sqrt_f32_rvv_(least) > limit;
        }
    }
    __riscv_vse32_v_f32m2(hausdorffs, __riscv_vfsqrt_v_f32m2(farthest_f32m2, 8), 8);
}

/** The parts of the 8 flips' squared distances from one outer point to @p vector_length inner
 *  points: flip f adds part f & 3, its planar sum, to part 4 + (f >> 2), its z square. */
NUMKONG_INLINE vfloat64m1x6_t nk_shapes_flip_parts_f64m1_rvv_(nk_f64_t outer_x, nk_f64_t outer_y, nk_f64_t outer_z,
                                                              vfloat64m1_t inner_x_f64m1, vfloat64m1_t inner_y_f64m1,
                                                              vfloat64m1_t inner_z_f64m1, nk_size_t vector_length) {
    // A flip negates an inner axis, turning its difference into a sum
    vfloat64m1_t const minus_x_f64m1 = __riscv_vfrsub_vf_f64m1(inner_x_f64m1, outer_x, vector_length);
    vfloat64m1_t const plus_x_f64m1 = __riscv_vfadd_vf_f64m1(inner_x_f64m1, outer_x, vector_length);
    vfloat64m1_t const minus_y_f64m1 = __riscv_vfrsub_vf_f64m1(inner_y_f64m1, outer_y, vector_length);
    vfloat64m1_t const plus_y_f64m1 = __riscv_vfadd_vf_f64m1(inner_y_f64m1, outer_y, vector_length);
    vfloat64m1_t const minus_z_f64m1 = __riscv_vfrsub_vf_f64m1(inner_z_f64m1, outer_z, vector_length);
    vfloat64m1_t const plus_z_f64m1 = __riscv_vfadd_vf_f64m1(inner_z_f64m1, outer_z, vector_length);
    vfloat64m1_t const minus_xx_f64m1 = __riscv_vfmul_vv_f64m1(minus_x_f64m1, minus_x_f64m1, vector_length);
    vfloat64m1_t const plus_xx_f64m1 = __riscv_vfmul_vv_f64m1(plus_x_f64m1, plus_x_f64m1, vector_length);
    vfloat64m1_t const minus_yy_f64m1 = __riscv_vfmul_vv_f64m1(minus_y_f64m1, minus_y_f64m1, vector_length);
    vfloat64m1_t const plus_yy_f64m1 = __riscv_vfmul_vv_f64m1(plus_y_f64m1, plus_y_f64m1, vector_length);
    return __riscv_vcreate_v_f64m1x6(__riscv_vfadd_vv_f64m1(minus_xx_f64m1, minus_yy_f64m1, vector_length),
                                     __riscv_vfadd_vv_f64m1(plus_xx_f64m1, minus_yy_f64m1, vector_length),
                                     __riscv_vfadd_vv_f64m1(minus_xx_f64m1, plus_yy_f64m1, vector_length),
                                     __riscv_vfadd_vv_f64m1(plus_xx_f64m1, plus_yy_f64m1, vector_length),
                                     __riscv_vfmul_vv_f64m1(minus_z_f64m1, minus_z_f64m1, vector_length),
                                     __riscv_vfmul_vv_f64m1(plus_z_f64m1, plus_z_f64m1, vector_length));
}

/** Lowers the running minima of the 8 flips by one outer point's squared distances to a register
 *  of inner points. */
NUMKONG_INLINE vfloat64m1x8_t nk_shapes_lower_nearest_f64m1_rvv_(vfloat64m1x8_t nearest_f64m1x8, nk_f64_t outer_x,
                                                                 nk_f64_t outer_y, nk_f64_t outer_z,
                                                                 vfloat64m1_t inner_x_f64m1, vfloat64m1_t inner_y_f64m1,
                                                                 vfloat64m1_t inner_z_f64m1, nk_size_t vector_length) {
    vfloat64m1x6_t const parts_f64m1x6 = nk_shapes_flip_parts_f64m1_rvv_(outer_x, outer_y, outer_z, inner_x_f64m1,
                                                                         inner_y_f64m1, inner_z_f64m1, vector_length);
    vfloat64m1_t const planar_0_f64m1 = __riscv_vget_v_f64m1x6_f64m1(parts_f64m1x6, 0);
    vfloat64m1_t const planar_1_f64m1 = __riscv_vget_v_f64m1x6_f64m1(parts_f64m1x6, 1);
    vfloat64m1_t const planar_2_f64m1 = __riscv_vget_v_f64m1x6_f64m1(parts_f64m1x6, 2);
    vfloat64m1_t const planar_3_f64m1 = __riscv_vget_v_f64m1x6_f64m1(parts_f64m1x6, 3);
    vfloat64m1_t const minus_zz_f64m1 = __riscv_vget_v_f64m1x6_f64m1(parts_f64m1x6, 4);
    vfloat64m1_t const plus_zz_f64m1 = __riscv_vget_v_f64m1x6_f64m1(parts_f64m1x6, 5);
    return __riscv_vcreate_v_f64m1x8(
        __riscv_vfmin_vv_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 0),
                               __riscv_vfadd_vv_f64m1(planar_0_f64m1, minus_zz_f64m1, vector_length), vector_length),
        __riscv_vfmin_vv_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 1),
                               __riscv_vfadd_vv_f64m1(planar_1_f64m1, minus_zz_f64m1, vector_length), vector_length),
        __riscv_vfmin_vv_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 2),
                               __riscv_vfadd_vv_f64m1(planar_2_f64m1, minus_zz_f64m1, vector_length), vector_length),
        __riscv_vfmin_vv_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 3),
                               __riscv_vfadd_vv_f64m1(planar_3_f64m1, minus_zz_f64m1, vector_length), vector_length),
        __riscv_vfmin_vv_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 4),
                               __riscv_vfadd_vv_f64m1(planar_0_f64m1, plus_zz_f64m1, vector_length), vector_length),
        __riscv_vfmin_vv_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 5),
                               __riscv_vfadd_vv_f64m1(planar_1_f64m1, plus_zz_f64m1, vector_length), vector_length),
        __riscv_vfmin_vv_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 6),
                               __riscv_vfadd_vv_f64m1(planar_2_f64m1, plus_zz_f64m1, vector_length), vector_length),
        __riscv_vfmin_vv_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 7),
                               __riscv_vfadd_vv_f64m1(planar_3_f64m1, plus_zz_f64m1, vector_length), vector_length));
}

/** Lowers one flip's running minima by its @p squares_f64m1 , and the first @p reverse_length of
 *  them into the inner points' @p reverse minima, which a last group fills only part of. */
NUMKONG_INLINE vfloat64m1_t nk_shapes_lower_flip_f64m1_rvv_(vfloat64m1_t nearest_f64m1, vfloat64m1_t squares_f64m1,
                                                            nk_f64_t *reverse, nk_size_t vector_length,
                                                            nk_size_t reverse_length) {
    vfloat64m1_t const reverse_f64m1 = __riscv_vle64_v_f64m1(reverse, reverse_length);
    __riscv_vse64_v_f64m1(reverse, __riscv_vfmin_vv_f64m1(reverse_f64m1, squares_f64m1, reverse_length),
                          reverse_length);
    return __riscv_vfmin_vv_f64m1(nearest_f64m1, squares_f64m1, vector_length);
}

/** Lowers the 8 flips' nearest squares as @ref nk_shapes_lower_nearest_f64m1_rvv_ does, and the
 *  same squares into the inner points' @p reverse_minima , 8 axes @p reverse_stride apart. */
NUMKONG_INLINE vfloat64m1x8_t nk_shapes_lower_mutual_f64m1_rvv_(vfloat64m1x8_t nearest_f64m1x8, nk_f64_t outer_x,
                                                                nk_f64_t outer_y, nk_f64_t outer_z,
                                                                vfloat64m1_t inner_x_f64m1, vfloat64m1_t inner_y_f64m1,
                                                                vfloat64m1_t inner_z_f64m1, nk_f64_t *reverse_minima,
                                                                nk_size_t reverse_stride, nk_size_t vector_length,
                                                                nk_size_t reverse_length) {
    vfloat64m1x6_t const parts_f64m1x6 = nk_shapes_flip_parts_f64m1_rvv_(outer_x, outer_y, outer_z, inner_x_f64m1,
                                                                         inner_y_f64m1, inner_z_f64m1, vector_length);
    vfloat64m1_t const planar_0_f64m1 = __riscv_vget_v_f64m1x6_f64m1(parts_f64m1x6, 0);
    vfloat64m1_t const planar_1_f64m1 = __riscv_vget_v_f64m1x6_f64m1(parts_f64m1x6, 1);
    vfloat64m1_t const planar_2_f64m1 = __riscv_vget_v_f64m1x6_f64m1(parts_f64m1x6, 2);
    vfloat64m1_t const planar_3_f64m1 = __riscv_vget_v_f64m1x6_f64m1(parts_f64m1x6, 3);
    vfloat64m1_t const minus_zz_f64m1 = __riscv_vget_v_f64m1x6_f64m1(parts_f64m1x6, 4);
    vfloat64m1_t const plus_zz_f64m1 = __riscv_vget_v_f64m1x6_f64m1(parts_f64m1x6, 5);
    return __riscv_vcreate_v_f64m1x8(
        nk_shapes_lower_flip_f64m1_rvv_(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 0),
                                        __riscv_vfadd_vv_f64m1(planar_0_f64m1, minus_zz_f64m1, vector_length),
                                        reverse_minima, vector_length, reverse_length),
        nk_shapes_lower_flip_f64m1_rvv_(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 1),
                                        __riscv_vfadd_vv_f64m1(planar_1_f64m1, minus_zz_f64m1, vector_length),
                                        reverse_minima + reverse_stride, vector_length, reverse_length),
        nk_shapes_lower_flip_f64m1_rvv_(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 2),
                                        __riscv_vfadd_vv_f64m1(planar_2_f64m1, minus_zz_f64m1, vector_length),
                                        reverse_minima + 2 * reverse_stride, vector_length, reverse_length),
        nk_shapes_lower_flip_f64m1_rvv_(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 3),
                                        __riscv_vfadd_vv_f64m1(planar_3_f64m1, minus_zz_f64m1, vector_length),
                                        reverse_minima + 3 * reverse_stride, vector_length, reverse_length),
        nk_shapes_lower_flip_f64m1_rvv_(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 4),
                                        __riscv_vfadd_vv_f64m1(planar_0_f64m1, plus_zz_f64m1, vector_length),
                                        reverse_minima + 4 * reverse_stride, vector_length, reverse_length),
        nk_shapes_lower_flip_f64m1_rvv_(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 5),
                                        __riscv_vfadd_vv_f64m1(planar_1_f64m1, plus_zz_f64m1, vector_length),
                                        reverse_minima + 5 * reverse_stride, vector_length, reverse_length),
        nk_shapes_lower_flip_f64m1_rvv_(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 6),
                                        __riscv_vfadd_vv_f64m1(planar_2_f64m1, plus_zz_f64m1, vector_length),
                                        reverse_minima + 6 * reverse_stride, vector_length, reverse_length),
        nk_shapes_lower_flip_f64m1_rvv_(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 7),
                                        __riscv_vfadd_vv_f64m1(planar_3_f64m1, plus_zz_f64m1, vector_length),
                                        reverse_minima + 7 * reverse_stride, vector_length, reverse_length));
}

/** The least of every flip's nearest squares over its lanes, a flip per lane, reduced to scalars as
 *  RVV reductions write only one lane. */
NUMKONG_INLINE vfloat64m4_t nk_shapes_least_flips_f64m1_rvv_(vfloat64m1x8_t nearest_f64m1x8,
                                                             vfloat64m1_t infinity_f64m1, nk_size_t full_length) {
    nk_f64_t least[8];
    least[0] = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m1_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 0), infinity_f64m1, full_length));
    least[1] = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m1_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 1), infinity_f64m1, full_length));
    least[2] = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m1_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 2), infinity_f64m1, full_length));
    least[3] = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m1_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 3), infinity_f64m1, full_length));
    least[4] = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m1_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 4), infinity_f64m1, full_length));
    least[5] = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m1_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 5), infinity_f64m1, full_length));
    least[6] = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m1_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 6), infinity_f64m1, full_length));
    least[7] = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m1_f64m1(__riscv_vget_v_f64m1x8_f64m1(nearest_f64m1x8, 7), infinity_f64m1, full_length));
    // Every RVV register holds at least 128 bits, so a group of 4 holds all 8 flips
    return __riscv_vle64_v_f64m4(least, 8);
}

/** A register pair of @c f32 coordinates widened to f64. */
NUMKONG_INLINE vfloat64m2_t nk_shapes_load_axis_f32_rvv_(nk_f32_t const *axis, nk_size_t vector_length) {
    return __riscv_vfwcvt_f_f_v_f64m2(__riscv_vle32_v_f32m1(axis, vector_length), vector_length);
}

/** One outer point's sweep of @c f32 inner points, each load widened to a register pair that runs
 *  one register after the other. Lanes past the last point stay infinite, so they never win. */
NUMKONG_INLINE vfloat64m1x8_t nk_shapes_sweep_nearest_f32_rvv_(nk_f64_t const *outer_point, nk_f32_t const *inner_axes,
                                                               nk_size_t inner_stride, nk_size_t inner_count,
                                                               vfloat64m1_t infinity_f64m1, nk_size_t full_length) {
    nk_f32_t const *inner_x = inner_axes, *inner_y = inner_x + inner_stride, *inner_z = inner_y + inner_stride;
    nk_size_t const pair_length = 2 * full_length, pairs_end = inner_count & ~(pair_length - 1);
    nk_f64_t const outer_x = outer_point[0], outer_y = outer_point[1], outer_z = outer_point[2];
    vfloat64m1x8_t nearest_f64m1x8 = __riscv_vcreate_v_f64m1x8(infinity_f64m1, infinity_f64m1, infinity_f64m1,
                                                               infinity_f64m1, infinity_f64m1, infinity_f64m1,
                                                               infinity_f64m1, infinity_f64m1);
    for (nk_size_t inner_index = 0; inner_index != pairs_end; inner_index += pair_length) {
        vfloat64m2_t const pair_x_f64m2 = nk_shapes_load_axis_f32_rvv_(inner_x + inner_index, pair_length);
        vfloat64m2_t const pair_y_f64m2 = nk_shapes_load_axis_f32_rvv_(inner_y + inner_index, pair_length);
        vfloat64m2_t const pair_z_f64m2 = nk_shapes_load_axis_f32_rvv_(inner_z + inner_index, pair_length);
        nearest_f64m1x8 = nk_shapes_lower_nearest_f64m1_rvv_(
            nearest_f64m1x8, outer_x, outer_y, outer_z, __riscv_vget_v_f64m2_f64m1(pair_x_f64m2, 0),
            __riscv_vget_v_f64m2_f64m1(pair_y_f64m2, 0), __riscv_vget_v_f64m2_f64m1(pair_z_f64m2, 0), full_length);
        nearest_f64m1x8 = nk_shapes_lower_nearest_f64m1_rvv_(
            nearest_f64m1x8, outer_x, outer_y, outer_z, __riscv_vget_v_f64m2_f64m1(pair_x_f64m2, 1),
            __riscv_vget_v_f64m2_f64m1(pair_y_f64m2, 1), __riscv_vget_v_f64m2_f64m1(pair_z_f64m2, 1), full_length);
    }
    if (pairs_end != inner_count) {
        nk_size_t const left = inner_count - pairs_end;
        vfloat64m2_t const infinity_f64m2 = __riscv_vcreate_v_f64m1_f64m2(infinity_f64m1, infinity_f64m1);
        vfloat64m2_t const pair_x_f64m2 = __riscv_vmv_v_v_f64m2_tu(
            infinity_f64m2, nk_shapes_load_axis_f32_rvv_(inner_x + pairs_end, left), left);
        vfloat64m2_t const pair_y_f64m2 = __riscv_vmv_v_v_f64m2_tu(
            infinity_f64m2, nk_shapes_load_axis_f32_rvv_(inner_y + pairs_end, left), left);
        vfloat64m2_t const pair_z_f64m2 = __riscv_vmv_v_v_f64m2_tu(
            infinity_f64m2, nk_shapes_load_axis_f32_rvv_(inner_z + pairs_end, left), left);
        nearest_f64m1x8 = nk_shapes_lower_nearest_f64m1_rvv_(
            nearest_f64m1x8, outer_x, outer_y, outer_z, __riscv_vget_v_f64m2_f64m1(pair_x_f64m2, 0),
            __riscv_vget_v_f64m2_f64m1(pair_y_f64m2, 0), __riscv_vget_v_f64m2_f64m1(pair_z_f64m2, 0), full_length);
        if (left > full_length)
            nearest_f64m1x8 = nk_shapes_lower_nearest_f64m1_rvv_(
                nearest_f64m1x8, outer_x, outer_y, outer_z, __riscv_vget_v_f64m2_f64m1(pair_x_f64m2, 1),
                __riscv_vget_v_f64m2_f64m1(pair_y_f64m2, 1), __riscv_vget_v_f64m2_f64m1(pair_z_f64m2, 1), full_length);
    }
    return nearest_f64m1x8;
}

/** One outer point's sweep of the inner cloud widened into f64 @p inner_axes , lowering the reverse
 *  minima too. Lanes past the last point stay infinite, so they never win, and the reverse minima
 *  take only the part that holds points. */
NUMKONG_INLINE vfloat64m1x8_t nk_shapes_sweep_mutual_f64m1_rvv_(nk_f64_t const *outer_point, nk_f64_t const *inner_axes,
                                                                nk_size_t inner_count, nk_f64_t *reverse_minima,
                                                                nk_size_t reverse_stride, vfloat64m1_t infinity_f64m1,
                                                                nk_size_t full_length) {
    nk_size_t const groups_end = inner_count & ~(full_length - 1);
    nk_f64_t const outer_x = outer_point[0], outer_y = outer_point[1], outer_z = outer_point[2];
    nk_f64_t const *inner_x = inner_axes, *inner_y = inner_x + reverse_stride, *inner_z = inner_y + reverse_stride;
    vfloat64m1x8_t nearest_f64m1x8 = __riscv_vcreate_v_f64m1x8(infinity_f64m1, infinity_f64m1, infinity_f64m1,
                                                               infinity_f64m1, infinity_f64m1, infinity_f64m1,
                                                               infinity_f64m1, infinity_f64m1);
    for (nk_size_t inner_index = 0; inner_index != groups_end; inner_index += full_length)
        nearest_f64m1x8 = nk_shapes_lower_mutual_f64m1_rvv_(
            nearest_f64m1x8, outer_x, outer_y, outer_z, __riscv_vle64_v_f64m1(inner_x + inner_index, full_length),
            __riscv_vle64_v_f64m1(inner_y + inner_index, full_length),
            __riscv_vle64_v_f64m1(inner_z + inner_index, full_length), reverse_minima + inner_index, reverse_stride,
            full_length, full_length);
    if (groups_end != inner_count) {
        nk_size_t const left = inner_count - groups_end;
        nearest_f64m1x8 = nk_shapes_lower_mutual_f64m1_rvv_(
            nearest_f64m1x8, outer_x, outer_y, outer_z,
            __riscv_vle64_v_f64m1_tu(infinity_f64m1, inner_x + groups_end, left),
            __riscv_vle64_v_f64m1_tu(infinity_f64m1, inner_y + groups_end, left),
            __riscv_vle64_v_f64m1_tu(infinity_f64m1, inner_z + groups_end, left), reverse_minima + groups_end,
            reverse_stride, full_length, left);
    }
    return nearest_f64m1x8;
}

/** The roots of @p count reverse minima summed. */
NUMKONG_INLINE nk_f64_t nk_shapes_reverse_sum_f64_rvv_(nk_f64_t const *reverse, nk_size_t count) {
    vfloat64m1_t sum_f64m1 = __riscv_vfmv_s_f_f64m1(0, 1);
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(count - index);
        vfloat64m4_t const roots_f64m4 = __riscv_vfsqrt_v_f64m4(__riscv_vle64_v_f64m4(reverse + index, vector_length),
                                                                vector_length);
        sum_f64m1 = __riscv_vfredusum_vs_f64m4_f64m1(roots_f64m4, sum_f64m1, vector_length);
    }
    return __riscv_vfmv_f_s_f64m1_f64(sum_f64m1);
}

/** Fused Chamfer flips of @c f32 packs in f64 lanes, widening the inner cloud once per pair as the
 *  @c f16 ones do. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f32_rvv_(void const *row_packed, void const *column_packed,
                                                            nk_f64_t limit, void *workspace, nk_size_t workspace_bytes,
                                                            nk_f64_t *chamfers) {
    nk_shapes_mutual_f32_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f32_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    nk_size_t const reverse_stride = views.reverse_stride, inner_count = views.inner_count;
    if (workspace_bytes < 11 * reverse_stride * sizeof(nk_f64_t)) return nk_unexpected_dimensions_k;
    nk_f64_t *reverse_minima = views.reverse_minima, *inner_axes = reverse_minima + 8 * reverse_stride;
    nk_size_t vector_length;
    for (nk_size_t index = 0; index < 8 * reverse_stride; index += vector_length) {
        vector_length = __riscv_vsetvl_e64m8(8 * reverse_stride - index);
        __riscv_vse64_v_f64m8(reverse_minima + index, __riscv_vfmv_v_f_f64m8(NUMKONG_F64_INF, vector_length),
                              vector_length);
    }
    for (nk_size_t axis = 0; axis != 3; ++axis)
        for (nk_size_t index = 0; index < inner_count; index += vector_length) {
            vector_length = __riscv_vsetvl_e32m1(inner_count - index);
            __riscv_vse64_v_f64m2(
                inner_axes + axis * reverse_stride + index,
                nk_shapes_load_axis_f32_rvv_(views.inner_axes + axis * views.inner_stride + index, vector_length),
                vector_length);
        }
    nk_size_t const full_length = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t const infinity_f64m1 = __riscv_vfmv_v_f_f64m1(NUMKONG_F64_INF, full_length);
    nk_f64_t const outer_points = (nk_f64_t)views.outer_count, inner_points = (nk_f64_t)inner_count;
    nk_f64_t const threshold = 2 * limit * outer_points;
    vfloat64m4_t sums_f64m4 = __riscv_vfmv_v_f_f64m4(0, 8);
    nk_f64_t outer_point[3], sums[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        outer_point[0] = views.outer_axes[outer_index],
        outer_point[1] = views.outer_axes[views.outer_stride + outer_index];
        outer_point[2] = views.outer_axes[2 * views.outer_stride + outer_index];
        vfloat64m1x8_t const nearest_f64m1x8 = nk_shapes_sweep_mutual_f64m1_rvv_(
            outer_point, inner_axes, inner_count, reverse_minima, reverse_stride, infinity_f64m1, full_length);
        sums_f64m4 = __riscv_vfadd_vv_f64m4(
            sums_f64m4,
            __riscv_vfsqrt_v_f64m4(nk_shapes_least_flips_f64m1_rvv_(nearest_f64m1x8, infinity_f64m1, full_length), 8),
            8);
        if (__riscv_vcpop_m_b16(__riscv_vmfgt_vf_f64m4_b16(sums_f64m4, threshold, 8), 8) == 8) {
            __riscv_vse64_v_f64m4(sums, sums_f64m4, 8);
            if (nk_shapes_mutual_past_limit_f64_serial_(sums, outer_points, limit, chamfers)) return nk_success_k;
        }
    }
    __riscv_vse64_v_f64m4(sums, sums_f64m4, 8);
    for (int flip = 0; flip != 8; ++flip) {
        nk_f64_t const reverse_sum = nk_shapes_reverse_sum_f64_rvv_(reverse_minima + flip * reverse_stride,
                                                                    inner_count);
        chamfers[flip] = (sums[flip] / outer_points + reverse_sum / inner_points) / 2;
    }
    return nk_success_k;
}

/** Hausdorff flips of @c f32 packs in f64 lanes, stopping as the @c f16 pair does. */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f32_rvv_(void const *row_packed, void const *column_packed,
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
    nk_size_t const full_length = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t const infinity_f64m1 = __riscv_vfmv_v_f_f64m1(NUMKONG_F64_INF, full_length);
    nk_f64_t const limit_squared = limit * limit;
    vfloat64m4_t farthest_f64m4 = __riscv_vfmv_v_f_f64m4(0, 8);
    nk_f64_t outer_point[3];
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_f32_t const *outer_axes = direction ? column_axes : row_axes;
        nk_f32_t const *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            outer_point[0] = outer_axes[outer_index], outer_point[1] = outer_axes[outer_stride + outer_index];
            outer_point[2] = outer_axes[2 * outer_stride + outer_index];
            vfloat64m1x8_t const nearest_f64m1x8 = nk_shapes_sweep_nearest_f32_rvv_(
                outer_point, inner_axes, inner_stride, inner_count, infinity_f64m1, full_length);
            farthest_f64m4 = __riscv_vfmax_vv_f64m4(
                farthest_f64m4, nk_shapes_least_flips_f64m1_rvv_(nearest_f64m1x8, infinity_f64m1, full_length), 8);
            nk_f64_t const least = __riscv_vfmv_f_s_f64m1_f64(
                __riscv_vfredmin_vs_f64m4_f64m1(farthest_f64m4, infinity_f64m1, 8));
            rejected = least > limit_squared && nk_sqrt_f64_rvv_(least) > limit;
        }
    }
    __riscv_vse64_v_f64m4(hausdorffs, __riscv_vfsqrt_v_f64m4(farthest_f64m4, 8), 8);
}

/** Squared distances from a row point to rotated column points in fused multiply-adds, which also
 *  serve single arcs at a vector length of 1, so that tree arcs keep a zero reduced cost. */
NUMKONG_INLINE vfloat32m1_t nk_shapes_point_costs_f32m1_rvv_(nk_shapes_workspace_f32_serial_t const *views,
                                                             nk_size_t row, nk_size_t column, nk_size_t vector_length) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points + column;
    vfloat32m1_t const delta_x_f32m1 = __riscv_vfrsub_vf_f32m1(__riscv_vle32_v_f32m1(rotated_points, vector_length),
                                                               row_points[row], vector_length);
    vfloat32m1_t const delta_y_f32m1 = __riscv_vfrsub_vf_f32m1(
        __riscv_vle32_v_f32m1(rotated_points + stride, vector_length), row_points[stride + row], vector_length);
    vfloat32m1_t const delta_z_f32m1 = __riscv_vfrsub_vf_f32m1(
        __riscv_vle32_v_f32m1(rotated_points + 2 * stride, vector_length), row_points[2 * stride + row], vector_length);
    vfloat32m1_t const planar_f32m1 = __riscv_vfmacc_vv_f32m1(
        __riscv_vfmul_vv_f32m1(delta_y_f32m1, delta_y_f32m1, vector_length), delta_x_f32m1, delta_x_f32m1,
        vector_length);
    return __riscv_vfmacc_vv_f32m1(planar_f32m1, delta_z_f32m1, delta_z_f32m1, vector_length);
}

NUMKONG_INLINE nk_f32_t nk_shapes_point_cost_f32_rvv_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t row,
                                                      nk_size_t column) {
    return __riscv_vfmv_f_s_f32m1_f32(nk_shapes_point_costs_f32m1_rvv_(views, row, column, 1));
}

/** The reduced costs c(i, j) − πᵢ + πⱼ of a row's arcs to a register group of columns, with matrix
 *  costs −G taken as πⱼ − (G + πᵢ), which rounds the same. */
NUMKONG_INLINE vfloat64m2_t nk_shapes_reduced_costs_f32_rvv_(nk_shapes_workspace_f32_serial_t const *views,
                                                             nk_f64_t const *gradient, nk_size_t rows,
                                                             nk_size_t columns, nk_size_t row, nk_size_t column,
                                                             nk_size_t vector_length) {
    nk_f64_t const row_potential = views->potential[row];
    vfloat64m2_t const column_potentials_f64m2 = __riscv_vle64_v_f64m2(views->potential + rows + column, vector_length);
    if (gradient)
        return __riscv_vfsub_vv_f64m2(
            column_potentials_f64m2,
            __riscv_vfadd_vf_f64m2(__riscv_vle64_v_f64m2(gradient + row * columns + column, vector_length),
                                   row_potential, vector_length),
            vector_length);
    vfloat64m2_t const costs_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(
        nk_shapes_point_costs_f32m1_rvv_(views, row, column, vector_length), vector_length);
    return __riscv_vfadd_vv_f64m2(__riscv_vfsub_vf_f64m2(costs_f64m2, row_potential, vector_length),
                                  column_potentials_f64m2, vector_length);
}

/** Keeps a tile's first least reduced cost below @p entering_cost, a register group at a time,
 *  branching only where a group beats it. */
NUMKONG_INLINE void nk_shapes_price_tile_f32_rvv_(nk_shapes_workspace_f32_serial_t const *views,
                                                  nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                  nk_size_t row, nk_size_t column, nk_size_t tile,
                                                  nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                  nk_size_t *entering_column) {
    for (nk_size_t step = 0, vector_length; step < tile; step += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(tile - step);
        vfloat64m2_t const reduced_f64m2 = nk_shapes_reduced_costs_f32_rvv_(views, gradient, rows, columns, row,
                                                                            column + step, vector_length);
        if (__riscv_vfirst_m_b32(__riscv_vmflt_vf_f64m2_b32(reduced_f64m2, *entering_cost, vector_length),
                                 vector_length) < 0)
            continue;
        *entering_cost = __riscv_vfmv_f_s_f64m1_f64(
            __riscv_vfredmin_vs_f64m2_f64m1(reduced_f64m2, __riscv_vfmv_s_f_f64m1(*entering_cost, 1), vector_length));
        long const first = __riscv_vfirst_m_b32(
            __riscv_vmfeq_vf_f64m2_b32(reduced_f64m2, *entering_cost, vector_length), vector_length);
        *entering_row = row, *entering_column = column + step + (nk_size_t)first;
    }
}

NUMKONG_INLINE void nk_shapes_tree_costs_f32_rvv_(nk_shapes_workspace_f32_serial_t const *views,
                                                  nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        views->potential[node] = gradient ? -gradient[row * columns + column]
                                          : (nk_f64_t)nk_shapes_point_cost_f32_rvv_(views, row, column);
    }
}

NUMKONG_INLINE void nk_shapes_network_simplex_f32_rvv_(nk_shapes_workspace_f32_serial_t const *views,
                                                       nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                       nk_f64_t cost_scale) {
    nk_shapes_simplex_serial_t simplex;
    nk_shapes_tree_costs_f32_rvv_(views, gradient, rows, columns);
    nk_shapes_tree_potentials_serial_(views->tree, views->tree_stride, views->potential, rows);
    nk_shapes_simplex_start_serial_(&simplex, views->tree, views->tree_stride, views->potential, rows, columns,
                                    cost_scale);
    while (nk_shapes_simplex_next_tile_serial_(&simplex))
        nk_shapes_price_tile_f32_rvv_(views, gradient, rows, columns, simplex.row, simplex.column, simplex.tile,
                                      &simplex.entering_cost, &simplex.entering_row, &simplex.entering_column);
}

NUMKONG_INLINE nk_f32_t nk_shapes_tree_cost_f32_rvv_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t rows,
                                                     nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f32_t cost = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        cost += (nk_f32_t)flow[node] * nk_shapes_point_cost_f32_rvv_(views, row, column);
    }
    return cost / (nk_f32_t)(rows * columns);
}

NUMKONG_INLINE vfloat64m2_t nk_shapes_point_costs_f64m2_rvv_(nk_shapes_workspace_f64_serial_t const *views,
                                                             nk_size_t row, nk_size_t column, nk_size_t vector_length) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points + column;
    vfloat64m2_t const delta_x_f64m2 = __riscv_vfrsub_vf_f64m2(__riscv_vle64_v_f64m2(rotated_points, vector_length),
                                                               row_points[row], vector_length);
    vfloat64m2_t const delta_y_f64m2 = __riscv_vfrsub_vf_f64m2(
        __riscv_vle64_v_f64m2(rotated_points + stride, vector_length), row_points[stride + row], vector_length);
    vfloat64m2_t const delta_z_f64m2 = __riscv_vfrsub_vf_f64m2(
        __riscv_vle64_v_f64m2(rotated_points + 2 * stride, vector_length), row_points[2 * stride + row], vector_length);
    vfloat64m2_t const planar_f64m2 = __riscv_vfmacc_vv_f64m2(
        __riscv_vfmul_vv_f64m2(delta_y_f64m2, delta_y_f64m2, vector_length), delta_x_f64m2, delta_x_f64m2,
        vector_length);
    return __riscv_vfmacc_vv_f64m2(planar_f64m2, delta_z_f64m2, delta_z_f64m2, vector_length);
}

NUMKONG_INLINE nk_f64_t nk_shapes_point_cost_f64_rvv_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t row,
                                                      nk_size_t column) {
    return __riscv_vfmv_f_s_f64m2_f64(nk_shapes_point_costs_f64m2_rvv_(views, row, column, 1));
}

NUMKONG_INLINE vfloat64m2_t nk_shapes_reduced_costs_f64_rvv_(nk_shapes_workspace_f64_serial_t const *views,
                                                             nk_f64_t const *gradient, nk_size_t rows,
                                                             nk_size_t columns, nk_size_t row, nk_size_t column,
                                                             nk_size_t vector_length) {
    nk_f64_t const row_potential = views->potential[row];
    vfloat64m2_t const column_potentials_f64m2 = __riscv_vle64_v_f64m2(views->potential + rows + column, vector_length);
    if (gradient)
        return __riscv_vfsub_vv_f64m2(
            column_potentials_f64m2,
            __riscv_vfadd_vf_f64m2(__riscv_vle64_v_f64m2(gradient + row * columns + column, vector_length),
                                   row_potential, vector_length),
            vector_length);
    vfloat64m2_t const costs_f64m2 = nk_shapes_point_costs_f64m2_rvv_(views, row, column, vector_length);
    return __riscv_vfadd_vv_f64m2(__riscv_vfsub_vf_f64m2(costs_f64m2, row_potential, vector_length),
                                  column_potentials_f64m2, vector_length);
}

NUMKONG_INLINE void nk_shapes_price_tile_f64_rvv_(nk_shapes_workspace_f64_serial_t const *views,
                                                  nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                  nk_size_t row, nk_size_t column, nk_size_t tile,
                                                  nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                  nk_size_t *entering_column) {
    for (nk_size_t step = 0, vector_length; step < tile; step += vector_length) {
        vector_length = __riscv_vsetvl_e64m2(tile - step);
        vfloat64m2_t const reduced_f64m2 = nk_shapes_reduced_costs_f64_rvv_(views, gradient, rows, columns, row,
                                                                            column + step, vector_length);
        if (__riscv_vfirst_m_b32(__riscv_vmflt_vf_f64m2_b32(reduced_f64m2, *entering_cost, vector_length),
                                 vector_length) < 0)
            continue;
        *entering_cost = __riscv_vfmv_f_s_f64m1_f64(
            __riscv_vfredmin_vs_f64m2_f64m1(reduced_f64m2, __riscv_vfmv_s_f_f64m1(*entering_cost, 1), vector_length));
        long const first = __riscv_vfirst_m_b32(
            __riscv_vmfeq_vf_f64m2_b32(reduced_f64m2, *entering_cost, vector_length), vector_length);
        *entering_row = row, *entering_column = column + step + (nk_size_t)first;
    }
}

NUMKONG_INLINE void nk_shapes_tree_costs_f64_rvv_(nk_shapes_workspace_f64_serial_t const *views,
                                                  nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        views->potential[node] = gradient ? -gradient[row * columns + column]
                                          : nk_shapes_point_cost_f64_rvv_(views, row, column);
    }
}

NUMKONG_INLINE void nk_shapes_network_simplex_f64_rvv_(nk_shapes_workspace_f64_serial_t const *views,
                                                       nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                       nk_f64_t cost_scale) {
    nk_shapes_simplex_serial_t simplex;
    nk_shapes_tree_costs_f64_rvv_(views, gradient, rows, columns);
    nk_shapes_tree_potentials_serial_(views->tree, views->tree_stride, views->potential, rows);
    nk_shapes_simplex_start_serial_(&simplex, views->tree, views->tree_stride, views->potential, rows, columns,
                                    cost_scale);
    while (nk_shapes_simplex_next_tile_serial_(&simplex))
        nk_shapes_price_tile_f64_rvv_(views, gradient, rows, columns, simplex.row, simplex.column, simplex.tile,
                                      &simplex.entering_cost, &simplex.entering_row, &simplex.entering_column);
}

NUMKONG_INLINE nk_f64_t nk_shapes_tree_cost_f64_rvv_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t rows,
                                                     nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f64_t cost = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        cost += (nk_f64_t)flow[node] * nk_shapes_point_cost_f64_rvv_(views, row, column);
    }
    return cost / (nk_f64_t)(rows * columns);
}

NUMKONG_INLINE nk_f32_t nk_shapes_wasserstein_procrustes_f16_rvv_(nk_shapes_workspace_f32_serial_t const *views,
                                                                  nk_size_t rows, nk_size_t columns,
                                                                  nk_f64_t cost_scale, nk_f32_t *rotation) {
    nk_shapes_procrustes_f32_serial_t search;
    nk_shapes_procrustes_start_f32_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_f32_serial_(&search)) {
        nk_shapes_network_simplex_f32_rvv_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_f32_rvv_(views, rows, columns);
    }
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f16_rvv_(nk_shapes_workspace_f32_serial_t const *views,
                                                              nk_size_t rows, nk_size_t columns, nk_f64_t cost_scale,
                                                              nk_f64_t const *row_distances,
                                                              nk_f64_t const *column_distances, nk_f32_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f16_rvv_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f32_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_serial_(&search);
    while (nk_shapes_gromov_next_f32_serial_(&search, views, rotation))
        nk_shapes_network_simplex_f32_rvv_(views, search.gradient, rows, columns, search.gradient_scale);
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_wasserstein_procrustes_f32_rvv_(nk_shapes_workspace_f64_serial_t const *views,
                                                                  nk_size_t rows, nk_size_t columns,
                                                                  nk_f64_t cost_scale, nk_f64_t *rotation) {
    nk_shapes_procrustes_f64_serial_t search;
    nk_shapes_procrustes_start_f64_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_f64_serial_(&search)) {
        nk_shapes_network_simplex_f64_rvv_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_f64_rvv_(views, rows, columns);
    }
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f32_rvv_(nk_shapes_workspace_f64_serial_t const *views,
                                                              nk_size_t rows, nk_size_t columns, nk_f64_t cost_scale,
                                                              nk_f64_t const *row_distances,
                                                              nk_f64_t const *column_distances, nk_f64_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f32_rvv_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f64_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_serial_(&search);
    while (nk_shapes_gromov_next_f64_serial_(&search, views, rotation))
        nk_shapes_network_simplex_f64_rvv_(views, search.gradient, rows, columns, search.gradient_scale);
    return search.best;
}

NUMKONG_INLINE void nk_jacobi_eigenanalysis_f32_rvv_(nk_f32_t *matrix, nk_f32_t *quaternion) {
    nk_jacobi_start_f32_serial_(quaternion);
    for (int sweep = 0; sweep != 16; ++sweep) {
        for (int axis = 0; axis != 3; ++axis) {
            nk_f32_t const norm = nk_jacobi_half_angle_f32_serial_(matrix);
            nk_jacobi_conjugate_f32_serial_(axis, nk_rsqrt_f32_rvv_(norm), matrix, quaternion);
        }
        if (nk_jacobi_converged_f32_serial_(matrix, 0x1p-24f)) break;
    }
    nk_f32_t const length_squared = nk_jacobi_length_squared_f32_serial_(quaternion);
    nk_jacobi_rescale_f32_serial_(nk_rsqrt_f32_rvv_(length_squared), quaternion);
}

NUMKONG_INLINE void nk_jacobi_eigenanalysis_f64_rvv_(nk_f64_t *matrix, nk_f64_t *quaternion) {
    nk_jacobi_start_f64_serial_(quaternion);
    for (int sweep = 0; sweep != 16; ++sweep) {
        for (int axis = 0; axis != 3; ++axis) {
            nk_f64_t const norm = nk_jacobi_half_angle_f64_serial_(matrix);
            nk_jacobi_conjugate_f64_serial_(axis, nk_rsqrt_f64_rvv_(norm), matrix, quaternion);
        }
        if (nk_jacobi_converged_f64_serial_(matrix, 0x1p-53)) break;
    }
    nk_f64_t const length_squared = nk_jacobi_length_squared_f64_serial_(quaternion);
    nk_jacobi_rescale_f64_serial_(nk_rsqrt_f64_rvv_(length_squared), quaternion);
}

/** The sum of a register group's lanes, each holding a partial sum. */
NUMKONG_INLINE nk_f32_t nk_shapes_sum_lanes_f32m2_rvv_(vfloat32m2_t lanes_f32m2, nk_size_t max_vector_length) {
    return __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(lanes_f32m2, __riscv_vfmv_s_f_f32m1(0, 1), max_vector_length));
}

/** @copydoc nk_shapes_sum_lanes_f32m2_rvv_ */
NUMKONG_INLINE nk_f64_t nk_shapes_sum_lanes_f64m2_rvv_(vfloat64m2_t lanes_f64m2, nk_size_t max_vector_length) {
    return __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredusum_vs_f64m2_f64m1(lanes_f64m2, __riscv_vfmv_s_f_f64m1(0, 1), max_vector_length));
}

/** The next @p vector_length interleaved @c f16 points, each axis widened to f32 lanes. */
NUMKONG_INLINE void nk_shapes_load_points_f16_rvv_(nk_f16_t const *points, nk_size_t vector_length,
                                                   vfloat32m2_t *x_f32m2, vfloat32m2_t *y_f32m2,
                                                   vfloat32m2_t *z_f32m2) {
    vuint16m1x3_t const xyz_u16m1x3 = __riscv_vlseg3e16_v_u16m1x3((nk_u16_t const *)points, vector_length);
    *x_f32m2 = nk_f16m1_to_f32m2_rvv_(__riscv_vget_v_u16m1x3_u16m1(xyz_u16m1x3, 0), vector_length);
    *y_f32m2 = nk_f16m1_to_f32m2_rvv_(__riscv_vget_v_u16m1x3_u16m1(xyz_u16m1x3, 1), vector_length);
    *z_f32m2 = nk_f16m1_to_f32m2_rvv_(__riscv_vget_v_u16m1x3_u16m1(xyz_u16m1x3, 2), vector_length);
}

/** The moments of a cloud of @c f16 points as @ref nk_shapes_moments_f16_serial_ defines them, a
 *  register group of points per step. */
NUMKONG_INLINE nk_f32_t nk_shapes_moments_f16_rvv_(nk_f16_t const *points, nk_size_t count, nk_f32_t *centroid,
                                                   nk_f32_t *covariance) {
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t const zero_f32m2 = __riscv_vfmv_v_f_f32m2(0, max_vector_length);
    vfloat32m2_t x_f32m2, y_f32m2, z_f32m2;
    nk_shapes_load_points_f16_rvv_(points, 1, &x_f32m2, &y_f32m2, &z_f32m2);
    nk_f32_t const first_x = __riscv_vfmv_f_s_f32m2_f32(x_f32m2), first_y = __riscv_vfmv_f_s_f32m2_f32(y_f32m2);
    nk_f32_t const first_z = __riscv_vfmv_f_s_f32m2_f32(z_f32m2);
    vfloat32m2_t sum_x_f32m2 = zero_f32m2, sum_y_f32m2 = zero_f32m2, sum_z_f32m2 = zero_f32m2;
    int coincident = 1;
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count - index);
        nk_shapes_load_points_f16_rvv_(points + index * 3, vector_length, &x_f32m2, &y_f32m2, &z_f32m2);
        sum_x_f32m2 = __riscv_vfadd_vv_f32m2_tu(sum_x_f32m2, sum_x_f32m2, x_f32m2, vector_length);
        sum_y_f32m2 = __riscv_vfadd_vv_f32m2_tu(sum_y_f32m2, sum_y_f32m2, y_f32m2, vector_length);
        sum_z_f32m2 = __riscv_vfadd_vv_f32m2_tu(sum_z_f32m2, sum_z_f32m2, z_f32m2, vector_length);
        vbool16_t const same_b16 = __riscv_vmand_mm_b16(
            __riscv_vmand_mm_b16(__riscv_vmfeq_vf_f32m2_b16(x_f32m2, first_x, vector_length),
                                 __riscv_vmfeq_vf_f32m2_b16(y_f32m2, first_y, vector_length), vector_length),
            __riscv_vmfeq_vf_f32m2_b16(z_f32m2, first_z, vector_length), vector_length);
        coincident &= __riscv_vcpop_m_b16(same_b16, vector_length) == vector_length;
    }
    nk_f32_t const points_count = (nk_f32_t)count;
    centroid[0] = nk_shapes_sum_lanes_f32m2_rvv_(sum_x_f32m2, max_vector_length) / points_count;
    centroid[1] = nk_shapes_sum_lanes_f32m2_rvv_(sum_y_f32m2, max_vector_length) / points_count;
    centroid[2] = nk_shapes_sum_lanes_f32m2_rvv_(sum_z_f32m2, max_vector_length) / points_count;
    vfloat32m2_t xx_f32m2 = zero_f32m2, yx_f32m2 = zero_f32m2, yy_f32m2 = zero_f32m2;
    vfloat32m2_t zx_f32m2 = zero_f32m2, zy_f32m2 = zero_f32m2, zz_f32m2 = zero_f32m2;
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count - index);
        nk_shapes_load_points_f16_rvv_(points + index * 3, vector_length, &x_f32m2, &y_f32m2, &z_f32m2);
        x_f32m2 = __riscv_vfsub_vf_f32m2(x_f32m2, centroid[0], vector_length);
        y_f32m2 = __riscv_vfsub_vf_f32m2(y_f32m2, centroid[1], vector_length);
        z_f32m2 = __riscv_vfsub_vf_f32m2(z_f32m2, centroid[2], vector_length);
        xx_f32m2 = __riscv_vfmacc_vv_f32m2_tu(xx_f32m2, x_f32m2, x_f32m2, vector_length);
        yx_f32m2 = __riscv_vfmacc_vv_f32m2_tu(yx_f32m2, y_f32m2, x_f32m2, vector_length);
        yy_f32m2 = __riscv_vfmacc_vv_f32m2_tu(yy_f32m2, y_f32m2, y_f32m2, vector_length);
        zx_f32m2 = __riscv_vfmacc_vv_f32m2_tu(zx_f32m2, z_f32m2, x_f32m2, vector_length);
        zy_f32m2 = __riscv_vfmacc_vv_f32m2_tu(zy_f32m2, z_f32m2, y_f32m2, vector_length);
        zz_f32m2 = __riscv_vfmacc_vv_f32m2_tu(zz_f32m2, z_f32m2, z_f32m2, vector_length);
    }
    covariance[0] = nk_shapes_sum_lanes_f32m2_rvv_(xx_f32m2, max_vector_length) / points_count;
    covariance[1] = nk_shapes_sum_lanes_f32m2_rvv_(yx_f32m2, max_vector_length) / points_count;
    covariance[2] = nk_shapes_sum_lanes_f32m2_rvv_(yy_f32m2, max_vector_length) / points_count;
    covariance[3] = nk_shapes_sum_lanes_f32m2_rvv_(zx_f32m2, max_vector_length) / points_count;
    covariance[4] = nk_shapes_sum_lanes_f32m2_rvv_(zy_f32m2, max_vector_length) / points_count;
    covariance[5] = nk_shapes_sum_lanes_f32m2_rvv_(zz_f32m2, max_vector_length) / points_count;
    return coincident ? 0 : nk_sqrt_f32_rvv_(covariance[0] + covariance[2] + covariance[5]);
}

/** The pack of one cloud from its moments, as @c nk_shapes_pack_f16_serial_ lays it out, storing
 *  each step's points and measuring what was stored in one sweep, with zeros in the padding. */
NUMKONG_INLINE void nk_shapes_pack_f16_rvv_(nk_f16_t const *points, nk_size_t count, nk_f32_t const *centroid,
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
    nk_u16_t *packed_x = (nk_u16_t *)(packed_bytes + header_bytes);
    nk_u16_t *packed_y = packed_x + axis_stride, *packed_z = packed_y + axis_stride;
    nk_f32_t spreads[6] = {covariance[0], covariance[1], covariance[2], covariance[3], covariance[4], covariance[5]};
    nk_f32_t quaternion[4], basis[9], axes[9];
    int upper_tied, lower_tied;
    nk_jacobi_eigenanalysis_f32_rvv_(spreads, quaternion);
    nk_shapes_sorted_basis_f32_serial_(spreads, quaternion, 0x1p-7f, basis, &upper_tied, &lower_tied);
    if (upper_tied && lower_tied) nk_shapes_pick_axis_f16_serial_(points, count, centroid, basis);
    else if (upper_tied || lower_tied) nk_shapes_turn_plane_f16_serial_(points, count, centroid, basis);
    nk_shapes_frame_f32_serial_(basis, upper_tied, header->frame, axes);
    for (int cell = 0; cell != 9; ++cell) axes[cell] /= scale;
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t const zero_f32m2 = __riscv_vfmv_v_f_f32m2(0, max_vector_length);
    vfloat32m2_t xx_sum_f32m2 = zero_f32m2, yx_sum_f32m2 = zero_f32m2, yy_sum_f32m2 = zero_f32m2;
    vfloat32m2_t zx_sum_f32m2 = zero_f32m2, zy_sum_f32m2 = zero_f32m2, zz_sum_f32m2 = zero_f32m2;
    vfloat32m2_t largest_f32m2 = zero_f32m2, x_f32m2, y_f32m2, z_f32m2;
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count - index);
        nk_shapes_load_points_f16_rvv_(points + index * 3, vector_length, &x_f32m2, &y_f32m2, &z_f32m2);
        x_f32m2 = __riscv_vfsub_vf_f32m2(x_f32m2, centroid[0], vector_length);
        y_f32m2 = __riscv_vfsub_vf_f32m2(y_f32m2, centroid[1], vector_length);
        z_f32m2 = __riscv_vfsub_vf_f32m2(z_f32m2, centroid[2], vector_length);
        vfloat32m2_t const rotated_x_f32m2 = __riscv_vfmacc_vf_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(x_f32m2, axes[0], vector_length), axes[1], y_f32m2,
                                    vector_length),
            axes[2], z_f32m2, vector_length);
        vfloat32m2_t const rotated_y_f32m2 = __riscv_vfmacc_vf_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(x_f32m2, axes[3], vector_length), axes[4], y_f32m2,
                                    vector_length),
            axes[5], z_f32m2, vector_length);
        vfloat32m2_t const rotated_z_f32m2 = __riscv_vfmacc_vf_f32m2(
            __riscv_vfmacc_vf_f32m2(__riscv_vfmul_vf_f32m2(x_f32m2, axes[6], vector_length), axes[7], y_f32m2,
                                    vector_length),
            axes[8], z_f32m2, vector_length);
        vuint16m1_t const stored_x_u16m1 = nk_f32m2_to_f16m1_rvv_(rotated_x_f32m2, vector_length);
        vuint16m1_t const stored_y_u16m1 = nk_f32m2_to_f16m1_rvv_(rotated_y_f32m2, vector_length);
        vuint16m1_t const stored_z_u16m1 = nk_f32m2_to_f16m1_rvv_(rotated_z_f32m2, vector_length);
        __riscv_vse16_v_u16m1(packed_x + index, stored_x_u16m1, vector_length);
        __riscv_vse16_v_u16m1(packed_y + index, stored_y_u16m1, vector_length);
        __riscv_vse16_v_u16m1(packed_z + index, stored_z_u16m1, vector_length);
        // The spreads describe the stored points, so the sweep measures them after rounding
        x_f32m2 = nk_f16m1_to_f32m2_rvv_(stored_x_u16m1, vector_length);
        y_f32m2 = nk_f16m1_to_f32m2_rvv_(stored_y_u16m1, vector_length);
        z_f32m2 = nk_f16m1_to_f32m2_rvv_(stored_z_u16m1, vector_length);
        vfloat32m2_t const xx_f32m2 = __riscv_vfmul_vv_f32m2(x_f32m2, x_f32m2, vector_length);
        vfloat32m2_t const yy_f32m2 = __riscv_vfmul_vv_f32m2(y_f32m2, y_f32m2, vector_length);
        vfloat32m2_t const zz_f32m2 = __riscv_vfmul_vv_f32m2(z_f32m2, z_f32m2, vector_length);
        vfloat32m2_t const norms_f32m2 = __riscv_vfadd_vv_f32m2(
            __riscv_vfadd_vv_f32m2(xx_f32m2, yy_f32m2, vector_length), zz_f32m2, vector_length);
        largest_f32m2 = __riscv_vfmax_vv_f32m2_tu(largest_f32m2, largest_f32m2, norms_f32m2, vector_length);
        xx_sum_f32m2 = __riscv_vfadd_vv_f32m2_tu(xx_sum_f32m2, xx_sum_f32m2, xx_f32m2, vector_length);
        yx_sum_f32m2 = __riscv_vfmacc_vv_f32m2_tu(yx_sum_f32m2, y_f32m2, x_f32m2, vector_length);
        yy_sum_f32m2 = __riscv_vfadd_vv_f32m2_tu(yy_sum_f32m2, yy_sum_f32m2, yy_f32m2, vector_length);
        zx_sum_f32m2 = __riscv_vfmacc_vv_f32m2_tu(zx_sum_f32m2, z_f32m2, x_f32m2, vector_length);
        zy_sum_f32m2 = __riscv_vfmacc_vv_f32m2_tu(zy_sum_f32m2, z_f32m2, y_f32m2, vector_length);
        zz_sum_f32m2 = __riscv_vfadd_vv_f32m2_tu(zz_sum_f32m2, zz_sum_f32m2, zz_f32m2, vector_length);
    }
    for (nk_size_t index = count; index != axis_stride; ++index)
        packed_x[index] = 0, packed_y[index] = 0, packed_z[index] = 0;
    nk_f32_t const points_count = (nk_f32_t)count;
    nk_f32_t moments[6] = {nk_shapes_sum_lanes_f32m2_rvv_(xx_sum_f32m2, max_vector_length) / points_count,
                           nk_shapes_sum_lanes_f32m2_rvv_(yx_sum_f32m2, max_vector_length) / points_count,
                           nk_shapes_sum_lanes_f32m2_rvv_(yy_sum_f32m2, max_vector_length) / points_count,
                           nk_shapes_sum_lanes_f32m2_rvv_(zx_sum_f32m2, max_vector_length) / points_count,
                           nk_shapes_sum_lanes_f32m2_rvv_(zy_sum_f32m2, max_vector_length) / points_count,
                           nk_shapes_sum_lanes_f32m2_rvv_(zz_sum_f32m2, max_vector_length) / points_count};
    nk_f32_t const largest = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmax_vs_f32m2_f32m1(largest_f32m2, __riscv_vfmv_s_f_f32m1(0, 1), max_vector_length));
    nk_jacobi_eigenanalysis_f32_rvv_(moments, quaternion);
    nk_shapes_close_pack_f32_serial_(header, count, centroid, moments, nk_sqrt_f32_rvv_(largest));
}

/** The next @p vector_length interleaved @c f32 points, each axis widened to f32 lanes. */
NUMKONG_INLINE void nk_shapes_load_points_f32_rvv_(nk_f32_t const *points, nk_size_t vector_length,
                                                   vfloat64m2_t *x_f64m2, vfloat64m2_t *y_f64m2,
                                                   vfloat64m2_t *z_f64m2) {
    vfloat32m1x3_t const xyz_f32m1x3 = __riscv_vlseg3e32_v_f32m1x3(points, vector_length);
    *x_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(xyz_f32m1x3, 0), vector_length);
    *y_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(xyz_f32m1x3, 1), vector_length);
    *z_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(__riscv_vget_v_f32m1x3_f32m1(xyz_f32m1x3, 2), vector_length);
}

/** The moments of a cloud of @c f32 points as @ref nk_shapes_moments_f32_serial_ defines them, a
 *  register group of points per step. */
NUMKONG_INLINE nk_f64_t nk_shapes_moments_f32_rvv_(nk_f32_t const *points, nk_size_t count, nk_f64_t *centroid,
                                                   nk_f64_t *covariance) {
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t const zero_f64m2 = __riscv_vfmv_v_f_f64m2(0, max_vector_length);
    vfloat64m2_t x_f64m2, y_f64m2, z_f64m2;
    nk_shapes_load_points_f32_rvv_(points, 1, &x_f64m2, &y_f64m2, &z_f64m2);
    nk_f64_t const first_x = __riscv_vfmv_f_s_f64m2_f64(x_f64m2), first_y = __riscv_vfmv_f_s_f64m2_f64(y_f64m2);
    nk_f64_t const first_z = __riscv_vfmv_f_s_f64m2_f64(z_f64m2);
    vfloat64m2_t sum_x_f64m2 = zero_f64m2, sum_y_f64m2 = zero_f64m2, sum_z_f64m2 = zero_f64m2;
    int coincident = 1;
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count - index);
        nk_shapes_load_points_f32_rvv_(points + index * 3, vector_length, &x_f64m2, &y_f64m2, &z_f64m2);
        sum_x_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_x_f64m2, sum_x_f64m2, x_f64m2, vector_length);
        sum_y_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_y_f64m2, sum_y_f64m2, y_f64m2, vector_length);
        sum_z_f64m2 = __riscv_vfadd_vv_f64m2_tu(sum_z_f64m2, sum_z_f64m2, z_f64m2, vector_length);
        vbool32_t const same_b32 = __riscv_vmand_mm_b32(
            __riscv_vmand_mm_b32(__riscv_vmfeq_vf_f64m2_b32(x_f64m2, first_x, vector_length),
                                 __riscv_vmfeq_vf_f64m2_b32(y_f64m2, first_y, vector_length), vector_length),
            __riscv_vmfeq_vf_f64m2_b32(z_f64m2, first_z, vector_length), vector_length);
        coincident &= __riscv_vcpop_m_b32(same_b32, vector_length) == vector_length;
    }
    nk_f64_t const points_count = (nk_f64_t)count;
    centroid[0] = nk_shapes_sum_lanes_f64m2_rvv_(sum_x_f64m2, max_vector_length) / points_count;
    centroid[1] = nk_shapes_sum_lanes_f64m2_rvv_(sum_y_f64m2, max_vector_length) / points_count;
    centroid[2] = nk_shapes_sum_lanes_f64m2_rvv_(sum_z_f64m2, max_vector_length) / points_count;
    vfloat64m2_t xx_f64m2 = zero_f64m2, yx_f64m2 = zero_f64m2, yy_f64m2 = zero_f64m2;
    vfloat64m2_t zx_f64m2 = zero_f64m2, zy_f64m2 = zero_f64m2, zz_f64m2 = zero_f64m2;
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count - index);
        nk_shapes_load_points_f32_rvv_(points + index * 3, vector_length, &x_f64m2, &y_f64m2, &z_f64m2);
        x_f64m2 = __riscv_vfsub_vf_f64m2(x_f64m2, centroid[0], vector_length);
        y_f64m2 = __riscv_vfsub_vf_f64m2(y_f64m2, centroid[1], vector_length);
        z_f64m2 = __riscv_vfsub_vf_f64m2(z_f64m2, centroid[2], vector_length);
        xx_f64m2 = __riscv_vfmacc_vv_f64m2_tu(xx_f64m2, x_f64m2, x_f64m2, vector_length);
        yx_f64m2 = __riscv_vfmacc_vv_f64m2_tu(yx_f64m2, y_f64m2, x_f64m2, vector_length);
        yy_f64m2 = __riscv_vfmacc_vv_f64m2_tu(yy_f64m2, y_f64m2, y_f64m2, vector_length);
        zx_f64m2 = __riscv_vfmacc_vv_f64m2_tu(zx_f64m2, z_f64m2, x_f64m2, vector_length);
        zy_f64m2 = __riscv_vfmacc_vv_f64m2_tu(zy_f64m2, z_f64m2, y_f64m2, vector_length);
        zz_f64m2 = __riscv_vfmacc_vv_f64m2_tu(zz_f64m2, z_f64m2, z_f64m2, vector_length);
    }
    covariance[0] = nk_shapes_sum_lanes_f64m2_rvv_(xx_f64m2, max_vector_length) / points_count;
    covariance[1] = nk_shapes_sum_lanes_f64m2_rvv_(yx_f64m2, max_vector_length) / points_count;
    covariance[2] = nk_shapes_sum_lanes_f64m2_rvv_(yy_f64m2, max_vector_length) / points_count;
    covariance[3] = nk_shapes_sum_lanes_f64m2_rvv_(zx_f64m2, max_vector_length) / points_count;
    covariance[4] = nk_shapes_sum_lanes_f64m2_rvv_(zy_f64m2, max_vector_length) / points_count;
    covariance[5] = nk_shapes_sum_lanes_f64m2_rvv_(zz_f64m2, max_vector_length) / points_count;
    return coincident ? 0 : nk_sqrt_f64_rvv_(covariance[0] + covariance[2] + covariance[5]);
}

/** The pack of one cloud from its moments, as @c nk_shapes_pack_f32_serial_ lays it out, storing
 *  each step's points and measuring what was stored in one sweep, with zeros in the padding. */
NUMKONG_INLINE void nk_shapes_pack_f32_rvv_(nk_f32_t const *points, nk_size_t count, nk_f64_t const *centroid,
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
    nk_jacobi_eigenanalysis_f64_rvv_(spreads, quaternion);
    nk_shapes_sorted_basis_f64_serial_(spreads, quaternion, 0x1p-16, basis, &upper_tied, &lower_tied);
    if (upper_tied && lower_tied) nk_shapes_pick_axis_f32_serial_(points, count, centroid, basis);
    else if (upper_tied || lower_tied) nk_shapes_turn_plane_f32_serial_(points, count, centroid, basis);
    nk_shapes_frame_f64_serial_(basis, upper_tied, header->frame, axes);
    for (int cell = 0; cell != 9; ++cell) axes[cell] /= scale;
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t const zero_f64m2 = __riscv_vfmv_v_f_f64m2(0, max_vector_length);
    vfloat64m2_t xx_sum_f64m2 = zero_f64m2, yx_sum_f64m2 = zero_f64m2, yy_sum_f64m2 = zero_f64m2;
    vfloat64m2_t zx_sum_f64m2 = zero_f64m2, zy_sum_f64m2 = zero_f64m2, zz_sum_f64m2 = zero_f64m2;
    vfloat64m2_t largest_f64m2 = zero_f64m2, x_f64m2, y_f64m2, z_f64m2;
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count - index);
        nk_shapes_load_points_f32_rvv_(points + index * 3, vector_length, &x_f64m2, &y_f64m2, &z_f64m2);
        x_f64m2 = __riscv_vfsub_vf_f64m2(x_f64m2, centroid[0], vector_length);
        y_f64m2 = __riscv_vfsub_vf_f64m2(y_f64m2, centroid[1], vector_length);
        z_f64m2 = __riscv_vfsub_vf_f64m2(z_f64m2, centroid[2], vector_length);
        vfloat64m2_t const rotated_x_f64m2 = __riscv_vfmacc_vf_f64m2(
            __riscv_vfmacc_vf_f64m2(__riscv_vfmul_vf_f64m2(x_f64m2, axes[0], vector_length), axes[1], y_f64m2,
                                    vector_length),
            axes[2], z_f64m2, vector_length);
        vfloat64m2_t const rotated_y_f64m2 = __riscv_vfmacc_vf_f64m2(
            __riscv_vfmacc_vf_f64m2(__riscv_vfmul_vf_f64m2(x_f64m2, axes[3], vector_length), axes[4], y_f64m2,
                                    vector_length),
            axes[5], z_f64m2, vector_length);
        vfloat64m2_t const rotated_z_f64m2 = __riscv_vfmacc_vf_f64m2(
            __riscv_vfmacc_vf_f64m2(__riscv_vfmul_vf_f64m2(x_f64m2, axes[6], vector_length), axes[7], y_f64m2,
                                    vector_length),
            axes[8], z_f64m2, vector_length);
        vfloat32m1_t const stored_x_f32m1 = __riscv_vfncvt_f_f_w_f32m1(rotated_x_f64m2, vector_length);
        vfloat32m1_t const stored_y_f32m1 = __riscv_vfncvt_f_f_w_f32m1(rotated_y_f64m2, vector_length);
        vfloat32m1_t const stored_z_f32m1 = __riscv_vfncvt_f_f_w_f32m1(rotated_z_f64m2, vector_length);
        __riscv_vse32_v_f32m1(packed_x + index, stored_x_f32m1, vector_length);
        __riscv_vse32_v_f32m1(packed_y + index, stored_y_f32m1, vector_length);
        __riscv_vse32_v_f32m1(packed_z + index, stored_z_f32m1, vector_length);
        // The spreads describe the stored points, so the sweep measures them after rounding
        x_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(stored_x_f32m1, vector_length);
        y_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(stored_y_f32m1, vector_length);
        z_f64m2 = __riscv_vfwcvt_f_f_v_f64m2(stored_z_f32m1, vector_length);
        vfloat64m2_t const xx_f64m2 = __riscv_vfmul_vv_f64m2(x_f64m2, x_f64m2, vector_length);
        vfloat64m2_t const yy_f64m2 = __riscv_vfmul_vv_f64m2(y_f64m2, y_f64m2, vector_length);
        vfloat64m2_t const zz_f64m2 = __riscv_vfmul_vv_f64m2(z_f64m2, z_f64m2, vector_length);
        vfloat64m2_t const norms_f64m2 = __riscv_vfadd_vv_f64m2(
            __riscv_vfadd_vv_f64m2(xx_f64m2, yy_f64m2, vector_length), zz_f64m2, vector_length);
        largest_f64m2 = __riscv_vfmax_vv_f64m2_tu(largest_f64m2, largest_f64m2, norms_f64m2, vector_length);
        xx_sum_f64m2 = __riscv_vfadd_vv_f64m2_tu(xx_sum_f64m2, xx_sum_f64m2, xx_f64m2, vector_length);
        yx_sum_f64m2 = __riscv_vfmacc_vv_f64m2_tu(yx_sum_f64m2, y_f64m2, x_f64m2, vector_length);
        yy_sum_f64m2 = __riscv_vfadd_vv_f64m2_tu(yy_sum_f64m2, yy_sum_f64m2, yy_f64m2, vector_length);
        zx_sum_f64m2 = __riscv_vfmacc_vv_f64m2_tu(zx_sum_f64m2, z_f64m2, x_f64m2, vector_length);
        zy_sum_f64m2 = __riscv_vfmacc_vv_f64m2_tu(zy_sum_f64m2, z_f64m2, y_f64m2, vector_length);
        zz_sum_f64m2 = __riscv_vfadd_vv_f64m2_tu(zz_sum_f64m2, zz_sum_f64m2, zz_f64m2, vector_length);
    }
    for (nk_size_t index = count; index != axis_stride; ++index)
        packed_x[index] = 0, packed_y[index] = 0, packed_z[index] = 0;
    nk_f64_t const points_count = (nk_f64_t)count;
    nk_f64_t moments[6] = {nk_shapes_sum_lanes_f64m2_rvv_(xx_sum_f64m2, max_vector_length) / points_count,
                           nk_shapes_sum_lanes_f64m2_rvv_(yx_sum_f64m2, max_vector_length) / points_count,
                           nk_shapes_sum_lanes_f64m2_rvv_(yy_sum_f64m2, max_vector_length) / points_count,
                           nk_shapes_sum_lanes_f64m2_rvv_(zx_sum_f64m2, max_vector_length) / points_count,
                           nk_shapes_sum_lanes_f64m2_rvv_(zy_sum_f64m2, max_vector_length) / points_count,
                           nk_shapes_sum_lanes_f64m2_rvv_(zz_sum_f64m2, max_vector_length) / points_count};
    nk_f64_t const largest = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmax_vs_f64m2_f64m1(largest_f64m2, __riscv_vfmv_s_f_f64m1(0, 1), max_vector_length));
    nk_jacobi_eigenanalysis_f64_rvv_(moments, quaternion);
    nk_shapes_close_pack_f64_serial_(header, count, centroid, moments, nk_sqrt_f64_rvv_(largest));
}

NUMKONG_API nk_status_t nk_shapes_pack_f16_rvv(nk_f16_t const *points, nk_size_t count, void *packed,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t centroid[3], covariance[6];
    nk_shapes_moments_f16_rvv_(points, count, centroid, covariance);
    nk_shapes_pack_f16_rvv_(points, count, centroid, covariance, 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f16_rvv(nk_f16_t const *points, nk_size_t count, void *packed,
                                                          nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t centroid[3], covariance[6];
    nk_f32_t const radius = nk_shapes_moments_f16_rvv_(points, count, centroid, covariance);
    nk_shapes_pack_f16_rvv_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_f32_rvv(nk_f32_t const *points, nk_size_t count, void *packed,
                                               nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t centroid[3], covariance[6];
    nk_shapes_moments_f32_rvv_(points, count, centroid, covariance);
    nk_shapes_pack_f32_rvv_(points, count, centroid, covariance, 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f32_rvv(nk_f32_t const *points, nk_size_t count, void *packed,
                                                          nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t centroid[3], covariance[6];
    nk_f64_t const radius = nk_shapes_moments_f32_rvv_(points, count, centroid, covariance);
    nk_shapes_pack_f32_rvv_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f16_rvv(void const *const *a_packed, nk_size_t rows,
                                                   void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                   nk_size_t results_stride, nk_f32_t *rotations, nk_f32_t const *limit,
                                                   void *workspace, nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t const pair_limit = *limit;
    nk_f32_t chamfers[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_status_t const status = nk_shapes_chamfer_flips_f16_rvv_(a_packed[row], b_packed[column], pair_limit,
                                                                        workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f32_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_rvv(void const *const *a_packed, nk_size_t rows,
                                                     void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                     nk_size_t results_stride, nk_f32_t *rotations,
                                                     nk_f32_t const *limit, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t const pair_limit = *limit;
    nk_f32_t hausdorffs[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_hausdorff_flips_f16_rvv_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f32_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f32_rvv(void const *const *a_packed, nk_size_t rows,
                                                   void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                   nk_size_t results_stride, nk_f32_t *rotations, nk_f64_t const *limit,
                                                   void *workspace, nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t const pair_limit = *limit;
    nk_f64_t chamfers[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_status_t const status = nk_shapes_chamfer_flips_f32_rvv_(a_packed[row], b_packed[column], pair_limit,
                                                                        workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f64_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_rvv(void const *const *a_packed, nk_size_t rows,
                                                     void const *const *b_packed, nk_size_t columns, nk_f64_t *results,
                                                     nk_size_t results_stride, nk_f32_t *rotations,
                                                     nk_f64_t const *limit, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t const pair_limit = *limit;
    nk_f64_t hausdorffs[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_hausdorff_flips_f32_rvv_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f64_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_rvv(void const *const *a_packed, nk_size_t rows,
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
                results_row[column] = nk_sqrt_f32_serial_(nk_shapes_wasserstein_procrustes_f16_rvv_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F32_INF;
            if (rotations) nk_shapes_pair_rotation_f32_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_rvv(void const *const *a_packed, nk_size_t rows,
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
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f16_rvv_(
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

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_rvv(void const *const *a_packed, nk_size_t rows,
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
                results_row[column] = nk_sqrt_f64_serial_(nk_shapes_wasserstein_procrustes_f32_rvv_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F64_INF;
            if (rotations) nk_shapes_pair_rotation_f64_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_rvv(void const *const *a_packed, nk_size_t rows,
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
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f32_rvv_(
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

/** Σ‖xᵢ′‖² of a pack of @c f16 points, a register group per step. */
NUMKONG_INLINE nk_f32_t nk_rmsds_norm_f16_rvv_(void const *packed) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)packed)->points;
    nk_size_t axis_stride;
    nk_u16_t const *axes = (nk_u16_t const *)nk_shapes_packed_axes_f16_serial_(packed, &axis_stride);
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t norm_f32m2 = __riscv_vfmv_v_f_f32m2(0, max_vector_length);
    for (int axis = 0; axis != 3; ++axis)
        for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(count - index);
            vfloat32m2_t const values_f32m2 = nk_shapes_load_axis_f16_rvv_(axes + axis * axis_stride + index,
                                                                           vector_length);
            norm_f32m2 = __riscv_vfmacc_vv_f32m2_tu(norm_f32m2, values_f32m2, values_f32m2, vector_length);
        }
    return nk_shapes_sum_lanes_f32m2_rvv_(norm_f32m2, max_vector_length);
}

/** Superposed RMSD of one pair of @c f16 packs with Ga + Gb in @p norms_squared, M in @c f32. */
NUMKONG_INLINE nk_f32_t nk_rmsds_pair_f16_rvv_(void const *row_packed, void const *column_packed,
                                               nk_f32_t norms_squared) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_u16_t const *a_x = (nk_u16_t const *)nk_shapes_packed_axes_f16_serial_(row_packed, &axis_stride);
    nk_u16_t const *b_x = (nk_u16_t const *)nk_shapes_packed_axes_f16_serial_(column_packed, &axis_stride);
    nk_u16_t const *a_y = a_x + axis_stride, *a_z = a_y + axis_stride;
    nk_u16_t const *b_y = b_x + axis_stride, *b_z = b_y + axis_stride;
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t const zero_f32m2 = __riscv_vfmv_v_f_f32m2(0, max_vector_length);
    vfloat32m2_t xx_f32m2 = zero_f32m2, xy_f32m2 = zero_f32m2, xz_f32m2 = zero_f32m2;
    vfloat32m2_t yx_f32m2 = zero_f32m2, yy_f32m2 = zero_f32m2, yz_f32m2 = zero_f32m2;
    vfloat32m2_t zx_f32m2 = zero_f32m2, zy_f32m2 = zero_f32m2, zz_f32m2 = zero_f32m2;
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e16m1(count - index);
        vfloat32m2_t const a_x_f32m2 = nk_shapes_load_axis_f16_rvv_(a_x + index, vector_length);
        vfloat32m2_t const a_y_f32m2 = nk_shapes_load_axis_f16_rvv_(a_y + index, vector_length);
        vfloat32m2_t const a_z_f32m2 = nk_shapes_load_axis_f16_rvv_(a_z + index, vector_length);
        vfloat32m2_t const b_x_f32m2 = nk_shapes_load_axis_f16_rvv_(b_x + index, vector_length);
        vfloat32m2_t const b_y_f32m2 = nk_shapes_load_axis_f16_rvv_(b_y + index, vector_length);
        vfloat32m2_t const b_z_f32m2 = nk_shapes_load_axis_f16_rvv_(b_z + index, vector_length);
        xx_f32m2 = __riscv_vfmacc_vv_f32m2_tu(xx_f32m2, a_x_f32m2, b_x_f32m2, vector_length);
        xy_f32m2 = __riscv_vfmacc_vv_f32m2_tu(xy_f32m2, a_x_f32m2, b_y_f32m2, vector_length);
        xz_f32m2 = __riscv_vfmacc_vv_f32m2_tu(xz_f32m2, a_x_f32m2, b_z_f32m2, vector_length);
        yx_f32m2 = __riscv_vfmacc_vv_f32m2_tu(yx_f32m2, a_y_f32m2, b_x_f32m2, vector_length);
        yy_f32m2 = __riscv_vfmacc_vv_f32m2_tu(yy_f32m2, a_y_f32m2, b_y_f32m2, vector_length);
        yz_f32m2 = __riscv_vfmacc_vv_f32m2_tu(yz_f32m2, a_y_f32m2, b_z_f32m2, vector_length);
        zx_f32m2 = __riscv_vfmacc_vv_f32m2_tu(zx_f32m2, a_z_f32m2, b_x_f32m2, vector_length);
        zy_f32m2 = __riscv_vfmacc_vv_f32m2_tu(zy_f32m2, a_z_f32m2, b_y_f32m2, vector_length);
        zz_f32m2 = __riscv_vfmacc_vv_f32m2_tu(zz_f32m2, a_z_f32m2, b_z_f32m2, vector_length);
    }
    nk_f32_t const cross_covariance[9] = {nk_shapes_sum_lanes_f32m2_rvv_(xx_f32m2, max_vector_length),
                                          nk_shapes_sum_lanes_f32m2_rvv_(xy_f32m2, max_vector_length),
                                          nk_shapes_sum_lanes_f32m2_rvv_(xz_f32m2, max_vector_length),
                                          nk_shapes_sum_lanes_f32m2_rvv_(yx_f32m2, max_vector_length),
                                          nk_shapes_sum_lanes_f32m2_rvv_(yy_f32m2, max_vector_length),
                                          nk_shapes_sum_lanes_f32m2_rvv_(yz_f32m2, max_vector_length),
                                          nk_shapes_sum_lanes_f32m2_rvv_(zx_f32m2, max_vector_length),
                                          nk_shapes_sum_lanes_f32m2_rvv_(zy_f32m2, max_vector_length),
                                          nk_shapes_sum_lanes_f32m2_rvv_(zz_f32m2, max_vector_length)};
    nk_f32_t const residual = nk_rmsds_folded_f32_serial_(cross_covariance, norms_squared);
    return nk_sqrt_f32_rvv_(residual / (nk_f32_t)count);
}

/** A register group of @c f32 coordinates widened to f64. */
NUMKONG_INLINE vfloat64m2_t nk_rmsds_load_axis_f32_rvv_(nk_f32_t const *axis, nk_size_t vector_length) {
    return __riscv_vfwcvt_f_f_v_f64m2(__riscv_vle32_v_f32m1(axis, vector_length), vector_length);
}

/** Superposed RMSD of one pair of @c f32 packs: H in @c f64, then the residuals in plain @c f64,
 *  a register group per step. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f32_rvv_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f32_t const *a_x = nk_shapes_packed_axes_f32_serial_(row_packed, &axis_stride);
    nk_f32_t const *b_x = nk_shapes_packed_axes_f32_serial_(column_packed, &axis_stride);
    nk_f32_t const *a_y = a_x + axis_stride, *a_z = a_y + axis_stride;
    nk_f32_t const *b_y = b_x + axis_stride, *b_z = b_y + axis_stride;
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t const zero_f64m2 = __riscv_vfmv_v_f_f64m2(0, max_vector_length);
    vfloat64m2_t xx_f64m2 = zero_f64m2, xy_f64m2 = zero_f64m2, xz_f64m2 = zero_f64m2;
    vfloat64m2_t yx_f64m2 = zero_f64m2, yy_f64m2 = zero_f64m2, yz_f64m2 = zero_f64m2;
    vfloat64m2_t zx_f64m2 = zero_f64m2, zy_f64m2 = zero_f64m2, zz_f64m2 = zero_f64m2;
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count - index);
        vfloat64m2_t const a_x_f64m2 = nk_rmsds_load_axis_f32_rvv_(a_x + index, vector_length);
        vfloat64m2_t const a_y_f64m2 = nk_rmsds_load_axis_f32_rvv_(a_y + index, vector_length);
        vfloat64m2_t const a_z_f64m2 = nk_rmsds_load_axis_f32_rvv_(a_z + index, vector_length);
        vfloat64m2_t const b_x_f64m2 = nk_rmsds_load_axis_f32_rvv_(b_x + index, vector_length);
        vfloat64m2_t const b_y_f64m2 = nk_rmsds_load_axis_f32_rvv_(b_y + index, vector_length);
        vfloat64m2_t const b_z_f64m2 = nk_rmsds_load_axis_f32_rvv_(b_z + index, vector_length);
        xx_f64m2 = __riscv_vfmacc_vv_f64m2_tu(xx_f64m2, b_x_f64m2, a_x_f64m2, vector_length);
        xy_f64m2 = __riscv_vfmacc_vv_f64m2_tu(xy_f64m2, b_x_f64m2, a_y_f64m2, vector_length);
        xz_f64m2 = __riscv_vfmacc_vv_f64m2_tu(xz_f64m2, b_x_f64m2, a_z_f64m2, vector_length);
        yx_f64m2 = __riscv_vfmacc_vv_f64m2_tu(yx_f64m2, b_y_f64m2, a_x_f64m2, vector_length);
        yy_f64m2 = __riscv_vfmacc_vv_f64m2_tu(yy_f64m2, b_y_f64m2, a_y_f64m2, vector_length);
        yz_f64m2 = __riscv_vfmacc_vv_f64m2_tu(yz_f64m2, b_y_f64m2, a_z_f64m2, vector_length);
        zx_f64m2 = __riscv_vfmacc_vv_f64m2_tu(zx_f64m2, b_z_f64m2, a_x_f64m2, vector_length);
        zy_f64m2 = __riscv_vfmacc_vv_f64m2_tu(zy_f64m2, b_z_f64m2, a_y_f64m2, vector_length);
        zz_f64m2 = __riscv_vfmacc_vv_f64m2_tu(zz_f64m2, b_z_f64m2, a_z_f64m2, vector_length);
    }
    nk_f64_t const cross_covariance[9] = {nk_shapes_sum_lanes_f64m2_rvv_(xx_f64m2, max_vector_length),
                                          nk_shapes_sum_lanes_f64m2_rvv_(xy_f64m2, max_vector_length),
                                          nk_shapes_sum_lanes_f64m2_rvv_(xz_f64m2, max_vector_length),
                                          nk_shapes_sum_lanes_f64m2_rvv_(yx_f64m2, max_vector_length),
                                          nk_shapes_sum_lanes_f64m2_rvv_(yy_f64m2, max_vector_length),
                                          nk_shapes_sum_lanes_f64m2_rvv_(yz_f64m2, max_vector_length),
                                          nk_shapes_sum_lanes_f64m2_rvv_(zx_f64m2, max_vector_length),
                                          nk_shapes_sum_lanes_f64m2_rvv_(zy_f64m2, max_vector_length),
                                          nk_shapes_sum_lanes_f64m2_rvv_(zz_f64m2, max_vector_length)};
    nk_f64_t rotation[9];
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    vfloat64m2_t sum_f64m2 = zero_f64m2;
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e32m1(count - index);
        vfloat64m2_t const b_x_f64m2 = nk_rmsds_load_axis_f32_rvv_(b_x + index, vector_length);
        vfloat64m2_t const b_y_f64m2 = nk_rmsds_load_axis_f32_rvv_(b_y + index, vector_length);
        vfloat64m2_t const b_z_f64m2 = nk_rmsds_load_axis_f32_rvv_(b_z + index, vector_length);
        for (int axis = 0; axis != 3; ++axis) {
            vfloat64m2_t residual_f64m2 = nk_rmsds_load_axis_f32_rvv_(a_x + axis * axis_stride + index, vector_length);
            residual_f64m2 = __riscv_vfnmsac_vf_f64m2(residual_f64m2, rotation[axis * 3 + 0], b_x_f64m2, vector_length);
            residual_f64m2 = __riscv_vfnmsac_vf_f64m2(residual_f64m2, rotation[axis * 3 + 1], b_y_f64m2, vector_length);
            residual_f64m2 = __riscv_vfnmsac_vf_f64m2(residual_f64m2, rotation[axis * 3 + 2], b_z_f64m2, vector_length);
            sum_f64m2 = __riscv_vfmacc_vv_f64m2_tu(sum_f64m2, residual_f64m2, residual_f64m2, vector_length);
        }
    }
    return nk_sqrt_f64_rvv_(nk_shapes_sum_lanes_f64m2_rvv_(sum_f64m2, max_vector_length) / (nk_f64_t)count);
}

/** Superposed RMSD of one pair of @c f64 packs, like the @c f32 pair with H and the residuals
 *  compensated, a register per step. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f64_rvv_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f64_t const *a_x = nk_shapes_packed_axes_f64_serial_(row_packed, &axis_stride);
    nk_f64_t const *b_x = nk_shapes_packed_axes_f64_serial_(column_packed, &axis_stride);
    nk_f64_t const *a_y = a_x + axis_stride, *a_z = a_y + axis_stride;
    nk_f64_t const *b_y = b_x + axis_stride, *b_z = b_y + axis_stride;
    vfloat64m1_t const zero_f64m1 = __riscv_vfmv_v_f_f64m1(0, __riscv_vsetvlmax_e64m1());
    vfloat64m1_t xx_f64m1 = zero_f64m1, xy_f64m1 = zero_f64m1, xz_f64m1 = zero_f64m1;
    vfloat64m1_t yx_f64m1 = zero_f64m1, yy_f64m1 = zero_f64m1, yz_f64m1 = zero_f64m1;
    vfloat64m1_t zx_f64m1 = zero_f64m1, zy_f64m1 = zero_f64m1, zz_f64m1 = zero_f64m1;
    vfloat64m1_t xx_error_f64m1 = zero_f64m1, xy_error_f64m1 = zero_f64m1, xz_error_f64m1 = zero_f64m1;
    vfloat64m1_t yx_error_f64m1 = zero_f64m1, yy_error_f64m1 = zero_f64m1, yz_error_f64m1 = zero_f64m1;
    vfloat64m1_t zx_error_f64m1 = zero_f64m1, zy_error_f64m1 = zero_f64m1, zz_error_f64m1 = zero_f64m1;
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count - index);
        vfloat64m1_t const a_x_f64m1 = __riscv_vle64_v_f64m1(a_x + index, vector_length);
        vfloat64m1_t const a_y_f64m1 = __riscv_vle64_v_f64m1(a_y + index, vector_length);
        vfloat64m1_t const a_z_f64m1 = __riscv_vle64_v_f64m1(a_z + index, vector_length);
        vfloat64m1_t const b_x_f64m1 = __riscv_vle64_v_f64m1(b_x + index, vector_length);
        vfloat64m1_t const b_y_f64m1 = __riscv_vle64_v_f64m1(b_y + index, vector_length);
        vfloat64m1_t const b_z_f64m1 = __riscv_vle64_v_f64m1(b_z + index, vector_length);
        nk_dot2_f64m1_rvv_(&xx_f64m1, &xx_error_f64m1, b_x_f64m1, a_x_f64m1, vector_length);
        nk_dot2_f64m1_rvv_(&xy_f64m1, &xy_error_f64m1, b_x_f64m1, a_y_f64m1, vector_length);
        nk_dot2_f64m1_rvv_(&xz_f64m1, &xz_error_f64m1, b_x_f64m1, a_z_f64m1, vector_length);
        nk_dot2_f64m1_rvv_(&yx_f64m1, &yx_error_f64m1, b_y_f64m1, a_x_f64m1, vector_length);
        nk_dot2_f64m1_rvv_(&yy_f64m1, &yy_error_f64m1, b_y_f64m1, a_y_f64m1, vector_length);
        nk_dot2_f64m1_rvv_(&yz_f64m1, &yz_error_f64m1, b_y_f64m1, a_z_f64m1, vector_length);
        nk_dot2_f64m1_rvv_(&zx_f64m1, &zx_error_f64m1, b_z_f64m1, a_x_f64m1, vector_length);
        nk_dot2_f64m1_rvv_(&zy_f64m1, &zy_error_f64m1, b_z_f64m1, a_y_f64m1, vector_length);
        nk_dot2_f64m1_rvv_(&zz_f64m1, &zz_error_f64m1, b_z_f64m1, a_z_f64m1, vector_length);
    }
    nk_f64_t const cross_covariance[9] = {
        nk_dot_stable_sum_f64m1_rvv_(xx_f64m1, xx_error_f64m1), nk_dot_stable_sum_f64m1_rvv_(xy_f64m1, xy_error_f64m1),
        nk_dot_stable_sum_f64m1_rvv_(xz_f64m1, xz_error_f64m1), nk_dot_stable_sum_f64m1_rvv_(yx_f64m1, yx_error_f64m1),
        nk_dot_stable_sum_f64m1_rvv_(yy_f64m1, yy_error_f64m1), nk_dot_stable_sum_f64m1_rvv_(yz_f64m1, yz_error_f64m1),
        nk_dot_stable_sum_f64m1_rvv_(zx_f64m1, zx_error_f64m1), nk_dot_stable_sum_f64m1_rvv_(zy_f64m1, zy_error_f64m1),
        nk_dot_stable_sum_f64m1_rvv_(zz_f64m1, zz_error_f64m1)};
    nk_f64_t rotation[9];
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    vfloat64m1_t sum_f64m1 = zero_f64m1, compensation_f64m1 = zero_f64m1;
    for (nk_size_t index = 0, vector_length; index < count; index += vector_length) {
        vector_length = __riscv_vsetvl_e64m1(count - index);
        vfloat64m1_t const b_x_f64m1 = __riscv_vle64_v_f64m1(b_x + index, vector_length);
        vfloat64m1_t const b_y_f64m1 = __riscv_vle64_v_f64m1(b_y + index, vector_length);
        vfloat64m1_t const b_z_f64m1 = __riscv_vle64_v_f64m1(b_z + index, vector_length);
        for (int axis = 0; axis != 3; ++axis) {
            vfloat64m1_t residual_f64m1 = __riscv_vle64_v_f64m1(a_x + axis * axis_stride + index, vector_length);
            residual_f64m1 = __riscv_vfnmsac_vf_f64m1(residual_f64m1, rotation[axis * 3 + 0], b_x_f64m1, vector_length);
            residual_f64m1 = __riscv_vfnmsac_vf_f64m1(residual_f64m1, rotation[axis * 3 + 1], b_y_f64m1, vector_length);
            residual_f64m1 = __riscv_vfnmsac_vf_f64m1(residual_f64m1, rotation[axis * 3 + 2], b_z_f64m1, vector_length);
            nk_dot2_f64m1_rvv_(&sum_f64m1, &compensation_f64m1, residual_f64m1, residual_f64m1, vector_length);
        }
    }
    return nk_sqrt_f64_rvv_(nk_dot_stable_sum_f64m1_rvv_(sum_f64m1, compensation_f64m1) / (nk_f64_t)count);
}

NUMKONG_API nk_status_t nk_rmsds_packed_f64_rvv(void const *const *a_packed, nk_size_t rows,
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
            results_row[column] = nk_rmsds_pair_f64_rvv_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f32_rvv(void const *const *a_packed, nk_size_t rows,
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
            results_row[column] = nk_rmsds_pair_f32_rvv_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f16_rvv(void const *const *a_packed, nk_size_t rows,
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
            row_norms[row - row_begin] = nk_rmsds_norm_f16_rvv_(a_packed[row]);
        for (nk_size_t column_begin = 0; column_begin < columns; column_begin += nk_rmsds_tile_frames_k) {
            nk_size_t const column_end = column_begin + nk_rmsds_tile_frames_k < columns
                                             ? column_begin + nk_rmsds_tile_frames_k
                                             : columns;
            for (nk_size_t column = column_begin; column != column_end; ++column)
                column_norms[column - column_begin] = nk_rmsds_norm_f16_rvv_(b_packed[column]);
            for (nk_size_t row = row_begin; row != row_end; ++row) {
                nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
                for (nk_size_t column = column_begin; column != column_end; ++column)
                    results_row[column] = nk_rmsds_pair_f16_rvv_(
                        a_packed[row], b_packed[column],
                        row_norms[row - row_begin] + column_norms[column - column_begin]);
            }
        }
    }
    return nk_success_k;
}

#if defined(__cplusplus)
} // extern "C"
#endif

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // NUMKONG_TARGET_RVV
#endif // NUMKONG_ARCH_RISCV64_RVV_
#endif // NUMKONG_ARCH_RISCV64_
#endif // NUMKONG_SHAPES_RVV_H
