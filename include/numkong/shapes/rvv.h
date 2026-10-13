/**
 *  @file include/numkong/shapes/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Many-to-many shape distances between 3D point clouds for RISC-V Vector, RVV.
 *
 *  @sa include/numkong/shapes.h
 *
 *  Chamfer and Hausdorff run the serial pair driver around a nearest-partner pass over a register
 *  group of inner points: each outer point keeps 8 running minima, one per axis flip, in the serial
 *  difference form and summation order. The vector length covers the last group, and the minima
 *  update leaves the lanes past it alone, so the zero padding of a pack never wins a minimum.
 *
 *  - @c f32 packs widen each load to @c f64 with @c vfwcvt
 *  - @c f16 packs widen each load to @c f32 by bit manipulation, since the base V extension has no
 *    @c f16 arithmetic or conversions
 *
 *  The transport kinds run the serial solvers around block pricing a register group of arcs at a
 *  time. The scalar cost behind the tree's potentials runs the pricing lanes at a vector length of
 *  1, so tree arcs keep a zero reduced cost.
 *
 *  @section shapes_rvv_instructions Key RVV Shape Instructions
 *
 *  @verbatim
 *  Intrinsic                               Purpose
 *  __riscv_vfwcvt_f_f_v_f64m1              Widens f32 coordinates to f64
 *  __riscv_vfrsub_vf_f64m1                 Outer coordinate minus inner ones
 *  __riscv_vfmin_vv_f64m1_tu               Running minimum, tail lanes undisturbed
 *  __riscv_vfredmin_vs_f64m1_f64m1         Folds one flip's minima onto a scalar
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
#include "numkong/shapes/serial.h" // `nk_define_shapes_network_simplex_`, `nk_shapes_workspace_f64_serial_t`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/** Lowers the scalar @p minimum to the smallest lane of @p nearest_f32m2. */
NUMKONG_INLINE void nk_shapes_lower_minimum_f32m2_rvv_(vfloat32m2_t nearest_f32m2, nk_f32_t *minimum,
                                                       nk_size_t max_vector_length) {
    vfloat32m1_t const minimum_f32m1 = __riscv_vfmv_s_f_f32m1(*minimum, 1);
    *minimum = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m2_f32m1(nearest_f32m2, minimum_f32m1, max_vector_length));
}

/** Lowers the scalar @p minimum to the smallest lane of @p nearest_f64m1. */
NUMKONG_INLINE void nk_shapes_lower_minimum_f64m1_rvv_(vfloat64m1_t nearest_f64m1, nk_f64_t *minimum,
                                                       nk_size_t max_vector_length) {
    vfloat64m1_t const minimum_f64m1 = __riscv_vfmv_s_f_f64m1(*minimum, 1);
    *minimum = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m1_f64m1(nearest_f64m1, minimum_f64m1, max_vector_length));
}

/** One directed pass of Chamfer and Hausdorff over @c f16 packs, like the serial one, widening each
 *  load of inner points to @c f32. */
NUMKONG_INLINE void nk_shapes_nearest_f16_rvv_(nk_f16_t const *outer_axes, nk_size_t outer_stride,
                                               nk_size_t outer_count, nk_f16_t const *inner_axes,
                                               nk_size_t inner_stride, nk_size_t inner_count, nk_f32_t *sums,
                                               nk_f32_t *maxima) {
    nk_u16_t const *inner_bits = (nk_u16_t const *)inner_axes;
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t const infinity_f32m2 = __riscv_vfmv_v_f_f32m2(NUMKONG_F32_INF, max_vector_length);
    for (nk_size_t outer = 0; outer != outer_count; ++outer) {
        nk_f32_t outer_x, outer_y, outer_z;
        nk_f16_to_f32_serial_(outer_axes + outer, &outer_x);
        nk_f16_to_f32_serial_(outer_axes + outer_stride + outer, &outer_y);
        nk_f16_to_f32_serial_(outer_axes + 2 * outer_stride + outer, &outer_z);
        vfloat32m2_t nearest_0_f32m2 = infinity_f32m2, nearest_1_f32m2 = infinity_f32m2;
        vfloat32m2_t nearest_2_f32m2 = infinity_f32m2, nearest_3_f32m2 = infinity_f32m2;
        vfloat32m2_t nearest_4_f32m2 = infinity_f32m2, nearest_5_f32m2 = infinity_f32m2;
        vfloat32m2_t nearest_6_f32m2 = infinity_f32m2, nearest_7_f32m2 = infinity_f32m2;
        for (nk_size_t inner = 0, vector_length; inner < inner_count; inner += vector_length) {
            vector_length = __riscv_vsetvl_e16m1(inner_count - inner);
            vfloat32m2_t const inner_x_f32m2 = nk_f16m1_to_f32m2_rvv_(
                __riscv_vle16_v_u16m1(inner_bits + inner, vector_length), vector_length);
            vfloat32m2_t const inner_y_f32m2 = nk_f16m1_to_f32m2_rvv_(
                __riscv_vle16_v_u16m1(inner_bits + inner_stride + inner, vector_length), vector_length);
            vfloat32m2_t const inner_z_f32m2 = nk_f16m1_to_f32m2_rvv_(
                __riscv_vle16_v_u16m1(inner_bits + 2 * inner_stride + inner, vector_length), vector_length);
            // A flip negates an inner axis, turning its difference into a sum
            vfloat32m2_t const minus_x_f32m2 = __riscv_vfrsub_vf_f32m2(inner_x_f32m2, outer_x, vector_length);
            vfloat32m2_t const plus_x_f32m2 = __riscv_vfadd_vf_f32m2(inner_x_f32m2, outer_x, vector_length);
            vfloat32m2_t const minus_y_f32m2 = __riscv_vfrsub_vf_f32m2(inner_y_f32m2, outer_y, vector_length);
            vfloat32m2_t const plus_y_f32m2 = __riscv_vfadd_vf_f32m2(inner_y_f32m2, outer_y, vector_length);
            vfloat32m2_t const minus_z_f32m2 = __riscv_vfrsub_vf_f32m2(inner_z_f32m2, outer_z, vector_length);
            vfloat32m2_t const plus_z_f32m2 = __riscv_vfadd_vf_f32m2(inner_z_f32m2, outer_z, vector_length);
            vfloat32m2_t const minus_xx_f32m2 = __riscv_vfmul_vv_f32m2(minus_x_f32m2, minus_x_f32m2, vector_length);
            vfloat32m2_t const plus_xx_f32m2 = __riscv_vfmul_vv_f32m2(plus_x_f32m2, plus_x_f32m2, vector_length);
            vfloat32m2_t const minus_yy_f32m2 = __riscv_vfmul_vv_f32m2(minus_y_f32m2, minus_y_f32m2, vector_length);
            vfloat32m2_t const plus_yy_f32m2 = __riscv_vfmul_vv_f32m2(plus_y_f32m2, plus_y_f32m2, vector_length);
            vfloat32m2_t const minus_zz_f32m2 = __riscv_vfmul_vv_f32m2(minus_z_f32m2, minus_z_f32m2, vector_length);
            vfloat32m2_t const plus_zz_f32m2 = __riscv_vfmul_vv_f32m2(plus_z_f32m2, plus_z_f32m2, vector_length);
            vfloat32m2_t const planar_0_f32m2 = __riscv_vfadd_vv_f32m2(minus_xx_f32m2, minus_yy_f32m2, vector_length);
            vfloat32m2_t const planar_1_f32m2 = __riscv_vfadd_vv_f32m2(plus_xx_f32m2, minus_yy_f32m2, vector_length);
            vfloat32m2_t const planar_2_f32m2 = __riscv_vfadd_vv_f32m2(minus_xx_f32m2, plus_yy_f32m2, vector_length);
            vfloat32m2_t const planar_3_f32m2 = __riscv_vfadd_vv_f32m2(plus_xx_f32m2, plus_yy_f32m2, vector_length);
            nearest_0_f32m2 = __riscv_vfmin_vv_f32m2_tu(
                nearest_0_f32m2, nearest_0_f32m2, __riscv_vfadd_vv_f32m2(planar_0_f32m2, minus_zz_f32m2, vector_length),
                vector_length);
            nearest_1_f32m2 = __riscv_vfmin_vv_f32m2_tu(
                nearest_1_f32m2, nearest_1_f32m2, __riscv_vfadd_vv_f32m2(planar_1_f32m2, minus_zz_f32m2, vector_length),
                vector_length);
            nearest_2_f32m2 = __riscv_vfmin_vv_f32m2_tu(
                nearest_2_f32m2, nearest_2_f32m2, __riscv_vfadd_vv_f32m2(planar_2_f32m2, minus_zz_f32m2, vector_length),
                vector_length);
            nearest_3_f32m2 = __riscv_vfmin_vv_f32m2_tu(
                nearest_3_f32m2, nearest_3_f32m2, __riscv_vfadd_vv_f32m2(planar_3_f32m2, minus_zz_f32m2, vector_length),
                vector_length);
            nearest_4_f32m2 = __riscv_vfmin_vv_f32m2_tu(
                nearest_4_f32m2, nearest_4_f32m2, __riscv_vfadd_vv_f32m2(planar_0_f32m2, plus_zz_f32m2, vector_length),
                vector_length);
            nearest_5_f32m2 = __riscv_vfmin_vv_f32m2_tu(
                nearest_5_f32m2, nearest_5_f32m2, __riscv_vfadd_vv_f32m2(planar_1_f32m2, plus_zz_f32m2, vector_length),
                vector_length);
            nearest_6_f32m2 = __riscv_vfmin_vv_f32m2_tu(
                nearest_6_f32m2, nearest_6_f32m2, __riscv_vfadd_vv_f32m2(planar_2_f32m2, plus_zz_f32m2, vector_length),
                vector_length);
            nearest_7_f32m2 = __riscv_vfmin_vv_f32m2_tu(
                nearest_7_f32m2, nearest_7_f32m2, __riscv_vfadd_vv_f32m2(planar_3_f32m2, plus_zz_f32m2, vector_length),
                vector_length);
        }
        nk_f32_t nearest[8] = {NUMKONG_F32_INF, NUMKONG_F32_INF, NUMKONG_F32_INF, NUMKONG_F32_INF,
                               NUMKONG_F32_INF, NUMKONG_F32_INF, NUMKONG_F32_INF, NUMKONG_F32_INF};
        nk_shapes_lower_minimum_f32m2_rvv_(nearest_0_f32m2, nearest + 0, max_vector_length);
        nk_shapes_lower_minimum_f32m2_rvv_(nearest_1_f32m2, nearest + 1, max_vector_length);
        nk_shapes_lower_minimum_f32m2_rvv_(nearest_2_f32m2, nearest + 2, max_vector_length);
        nk_shapes_lower_minimum_f32m2_rvv_(nearest_3_f32m2, nearest + 3, max_vector_length);
        nk_shapes_lower_minimum_f32m2_rvv_(nearest_4_f32m2, nearest + 4, max_vector_length);
        nk_shapes_lower_minimum_f32m2_rvv_(nearest_5_f32m2, nearest + 5, max_vector_length);
        nk_shapes_lower_minimum_f32m2_rvv_(nearest_6_f32m2, nearest + 6, max_vector_length);
        nk_shapes_lower_minimum_f32m2_rvv_(nearest_7_f32m2, nearest + 7, max_vector_length);
        // Every RVV register holds at least 128 bits, so a pair of them holds all 8 flips
        vfloat32m2_t const distance_f32m2 = __riscv_vfsqrt_v_f32m2(__riscv_vle32_v_f32m2(nearest, 8), 8);
        __riscv_vse32_v_f32m2(sums, __riscv_vfadd_vv_f32m2(__riscv_vle32_v_f32m2(sums, 8), distance_f32m2, 8), 8);
        __riscv_vse32_v_f32m2(maxima, __riscv_vfmax_vv_f32m2(__riscv_vle32_v_f32m2(maxima, 8), distance_f32m2, 8), 8);
    }
}

/** One directed pass of Chamfer and Hausdorff over @c f32 packs, like the serial one. */
NUMKONG_INLINE void nk_shapes_nearest_f32_rvv_(nk_f32_t const *outer_axes, nk_size_t outer_stride,
                                               nk_size_t outer_count, nk_f32_t const *inner_axes,
                                               nk_size_t inner_stride, nk_size_t inner_count, nk_f64_t *sums,
                                               nk_f64_t *maxima) {
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e64m1();
    vfloat64m1_t const infinity_f64m1 = __riscv_vfmv_v_f_f64m1(NUMKONG_F64_INF, max_vector_length);
    for (nk_size_t outer = 0; outer != outer_count; ++outer) {
        nk_f64_t const outer_x = outer_axes[outer], outer_y = outer_axes[outer_stride + outer];
        nk_f64_t const outer_z = outer_axes[2 * outer_stride + outer];
        vfloat64m1_t nearest_0_f64m1 = infinity_f64m1, nearest_1_f64m1 = infinity_f64m1;
        vfloat64m1_t nearest_2_f64m1 = infinity_f64m1, nearest_3_f64m1 = infinity_f64m1;
        vfloat64m1_t nearest_4_f64m1 = infinity_f64m1, nearest_5_f64m1 = infinity_f64m1;
        vfloat64m1_t nearest_6_f64m1 = infinity_f64m1, nearest_7_f64m1 = infinity_f64m1;
        for (nk_size_t inner = 0, vector_length; inner < inner_count; inner += vector_length) {
            vector_length = __riscv_vsetvl_e64m1(inner_count - inner);
            vfloat64m1_t const inner_x_f64m1 = __riscv_vfwcvt_f_f_v_f64m1(
                __riscv_vle32_v_f32mf2(inner_axes + inner, vector_length), vector_length);
            vfloat64m1_t const inner_y_f64m1 = __riscv_vfwcvt_f_f_v_f64m1(
                __riscv_vle32_v_f32mf2(inner_axes + inner_stride + inner, vector_length), vector_length);
            vfloat64m1_t const inner_z_f64m1 = __riscv_vfwcvt_f_f_v_f64m1(
                __riscv_vle32_v_f32mf2(inner_axes + 2 * inner_stride + inner, vector_length), vector_length);
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
            vfloat64m1_t const minus_zz_f64m1 = __riscv_vfmul_vv_f64m1(minus_z_f64m1, minus_z_f64m1, vector_length);
            vfloat64m1_t const plus_zz_f64m1 = __riscv_vfmul_vv_f64m1(plus_z_f64m1, plus_z_f64m1, vector_length);
            vfloat64m1_t const planar_0_f64m1 = __riscv_vfadd_vv_f64m1(minus_xx_f64m1, minus_yy_f64m1, vector_length);
            vfloat64m1_t const planar_1_f64m1 = __riscv_vfadd_vv_f64m1(plus_xx_f64m1, minus_yy_f64m1, vector_length);
            vfloat64m1_t const planar_2_f64m1 = __riscv_vfadd_vv_f64m1(minus_xx_f64m1, plus_yy_f64m1, vector_length);
            vfloat64m1_t const planar_3_f64m1 = __riscv_vfadd_vv_f64m1(plus_xx_f64m1, plus_yy_f64m1, vector_length);
            nearest_0_f64m1 = __riscv_vfmin_vv_f64m1_tu(
                nearest_0_f64m1, nearest_0_f64m1, __riscv_vfadd_vv_f64m1(planar_0_f64m1, minus_zz_f64m1, vector_length),
                vector_length);
            nearest_1_f64m1 = __riscv_vfmin_vv_f64m1_tu(
                nearest_1_f64m1, nearest_1_f64m1, __riscv_vfadd_vv_f64m1(planar_1_f64m1, minus_zz_f64m1, vector_length),
                vector_length);
            nearest_2_f64m1 = __riscv_vfmin_vv_f64m1_tu(
                nearest_2_f64m1, nearest_2_f64m1, __riscv_vfadd_vv_f64m1(planar_2_f64m1, minus_zz_f64m1, vector_length),
                vector_length);
            nearest_3_f64m1 = __riscv_vfmin_vv_f64m1_tu(
                nearest_3_f64m1, nearest_3_f64m1, __riscv_vfadd_vv_f64m1(planar_3_f64m1, minus_zz_f64m1, vector_length),
                vector_length);
            nearest_4_f64m1 = __riscv_vfmin_vv_f64m1_tu(
                nearest_4_f64m1, nearest_4_f64m1, __riscv_vfadd_vv_f64m1(planar_0_f64m1, plus_zz_f64m1, vector_length),
                vector_length);
            nearest_5_f64m1 = __riscv_vfmin_vv_f64m1_tu(
                nearest_5_f64m1, nearest_5_f64m1, __riscv_vfadd_vv_f64m1(planar_1_f64m1, plus_zz_f64m1, vector_length),
                vector_length);
            nearest_6_f64m1 = __riscv_vfmin_vv_f64m1_tu(
                nearest_6_f64m1, nearest_6_f64m1, __riscv_vfadd_vv_f64m1(planar_2_f64m1, plus_zz_f64m1, vector_length),
                vector_length);
            nearest_7_f64m1 = __riscv_vfmin_vv_f64m1_tu(
                nearest_7_f64m1, nearest_7_f64m1, __riscv_vfadd_vv_f64m1(planar_3_f64m1, plus_zz_f64m1, vector_length),
                vector_length);
        }
        nk_f64_t nearest[8] = {NUMKONG_F64_INF, NUMKONG_F64_INF, NUMKONG_F64_INF, NUMKONG_F64_INF,
                               NUMKONG_F64_INF, NUMKONG_F64_INF, NUMKONG_F64_INF, NUMKONG_F64_INF};
        nk_shapes_lower_minimum_f64m1_rvv_(nearest_0_f64m1, nearest + 0, max_vector_length);
        nk_shapes_lower_minimum_f64m1_rvv_(nearest_1_f64m1, nearest + 1, max_vector_length);
        nk_shapes_lower_minimum_f64m1_rvv_(nearest_2_f64m1, nearest + 2, max_vector_length);
        nk_shapes_lower_minimum_f64m1_rvv_(nearest_3_f64m1, nearest + 3, max_vector_length);
        nk_shapes_lower_minimum_f64m1_rvv_(nearest_4_f64m1, nearest + 4, max_vector_length);
        nk_shapes_lower_minimum_f64m1_rvv_(nearest_5_f64m1, nearest + 5, max_vector_length);
        nk_shapes_lower_minimum_f64m1_rvv_(nearest_6_f64m1, nearest + 6, max_vector_length);
        nk_shapes_lower_minimum_f64m1_rvv_(nearest_7_f64m1, nearest + 7, max_vector_length);
        vfloat64m4_t const distance_f64m4 = __riscv_vfsqrt_v_f64m4(__riscv_vle64_v_f64m4(nearest, 8), 8);
        __riscv_vse64_v_f64m4(sums, __riscv_vfadd_vv_f64m4(__riscv_vle64_v_f64m4(sums, 8), distance_f64m4, 8), 8);
        __riscv_vse64_v_f64m4(maxima, __riscv_vfmax_vv_f64m4(__riscv_vle64_v_f64m4(maxima, 8), distance_f64m4, 8), 8);
    }
}

/** The block pricing of one math type over @p vector_type register groups: squared distances from a
 *  row point to rotated column points in fused multiply-adds; the scalar cost behind the
 *  potentials, run on the same lanes at a vector length of 1 so tree arcs keep a zero reduced cost;
 *  the reduced costs c(i, j) − πᵢ + πⱼ of a row's arcs to a register group of columns, with matrix
 *  costs −G taken as πⱼ − (G + πᵢ), which rounds the same; and the scan of @p count of them from
 *  the cursor in row-major order, keeping the first most negative one below @p entering_cost as the
 *  serial scan does. A first pass takes their least a register group at a time, and only a block
 *  whose least beats @p entering_cost is scanned again for the first arc reaching it. @p widen
 *  turns the costs into f64 lanes, a plain move when they are, and @p set_vector_length takes as
 *  many lanes as an f64 register group holds. */
#define nk_define_shapes_pricing_rvv_(math_type, vector_type, vector_suffix, load, reverse_subtract, multiply,         \
                                      multiply_accumulate, first_lane, widen, set_vector_length)                       \
    NUMKONG_INLINE vector_type nk_shapes_point_costs_##vector_suffix##_rvv_(                                           \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_size_t row, nk_size_t column,                      \
        nk_size_t vector_length) {                                                                                     \
        nk_size_t const stride = views->points_stride;                                                                 \
        nk_##math_type##_t const *row_points = views->row_points, *rotated_points = views->rotated_points + column;    \
        vector_type const delta_x_##vector_suffix = reverse_subtract(load(rotated_points, vector_length),              \
                                                                     row_points[row], vector_length);                  \
        vector_type const delta_y_##vector_suffix = reverse_subtract(load(rotated_points + stride, vector_length),     \
                                                                     row_points[stride + row], vector_length);         \
        vector_type const delta_z_##vector_suffix = reverse_subtract(load(rotated_points + 2 * stride, vector_length), \
                                                                     row_points[2 * stride + row], vector_length);     \
        vector_type const planar_##vector_suffix = multiply_accumulate(                                                \
            multiply(delta_y_##vector_suffix, delta_y_##vector_suffix, vector_length), delta_x_##vector_suffix,        \
            delta_x_##vector_suffix, vector_length);                                                                   \
        return multiply_accumulate(planar_##vector_suffix, delta_z_##vector_suffix, delta_z_##vector_suffix,           \
                                   vector_length);                                                                     \
    }                                                                                                                  \
    NUMKONG_INLINE nk_##math_type##_t nk_shapes_point_cost_##math_type##_rvv_(                                         \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_size_t row, nk_size_t column) {                    \
        return first_lane(nk_shapes_point_costs_##vector_suffix##_rvv_(views, row, column, 1));                        \
    }                                                                                                                  \
    NUMKONG_INLINE vfloat64m2_t nk_shapes_reduced_costs_##math_type##_rvv_(                                            \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_f64_t const *gradient, nk_size_t rows,             \
        nk_size_t columns, nk_size_t row, nk_size_t column, nk_size_t vector_length) {                                 \
        nk_f64_t const row_potential = views->potential[row];                                                          \
        vfloat64m2_t const column_potentials_f64m2 = __riscv_vle64_v_f64m2(views->potential + rows + column,           \
                                                                           vector_length);                             \
        if (gradient)                                                                                                  \
            return __riscv_vfsub_vv_f64m2(                                                                             \
                column_potentials_f64m2,                                                                               \
                __riscv_vfadd_vf_f64m2(__riscv_vle64_v_f64m2(gradient + row * columns + column, vector_length),        \
                                       row_potential, vector_length),                                                  \
                vector_length);                                                                                        \
        vfloat64m2_t const costs_f64m2 = widen(                                                                        \
            nk_shapes_point_costs_##vector_suffix##_rvv_(views, row, column, vector_length), vector_length);           \
        return __riscv_vfadd_vv_f64m2(__riscv_vfsub_vf_f64m2(costs_f64m2, row_potential, vector_length),               \
                                      column_potentials_f64m2, vector_length);                                         \
    }                                                                                                                  \
    NUMKONG_INLINE void nk_shapes_price_block_##math_type##_rvv_(                                                      \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_f64_t const *gradient, nk_size_t rows,             \
        nk_size_t columns, nk_size_t count, nk_size_t *cursor_row, nk_size_t *cursor_column, nk_f64_t *entering_cost,  \
        nk_size_t *entering_row, nk_size_t *entering_column) {                                                         \
        nk_size_t const max_vector_length = __riscv_vsetvlmax_e64m2();                                                 \
        vfloat64m2_t least_f64m2 = __riscv_vfmv_v_f_f64m2(NUMKONG_F64_INF, max_vector_length);                         \
        nk_size_t const start_row = *cursor_row, start_column = *cursor_column;                                        \
        nk_size_t row = start_row, column = start_column;                                                              \
        for (nk_size_t offset = 0; offset != count;) {                                                                 \
            nk_size_t const segment = columns - column < count - offset ? columns - column : count - offset;           \
            for (nk_size_t step = 0, vector_length; step < segment; step += vector_length) {                           \
                vector_length = set_vector_length(segment - step);                                                     \
                vfloat64m2_t const reduced_f64m2 = nk_shapes_reduced_costs_##math_type##_rvv_(                         \
                    views, gradient, rows, columns, row, column + step, vector_length);                                \
                least_f64m2 = __riscv_vfmin_vv_f64m2_tu(least_f64m2, least_f64m2, reduced_f64m2, vector_length);       \
            }                                                                                                          \
            offset += segment, column += segment;                                                                      \
            if (column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;                                    \
        }                                                                                                              \
        *cursor_row = row, *cursor_column = column;                                                                    \
        nk_f64_t const least = __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmin_vs_f64m2_f64m1(                             \
            least_f64m2, __riscv_vfmv_s_f_f64m1(NUMKONG_F64_INF, 1), max_vector_length));                              \
        if (!(least < *entering_cost)) return;                                                                         \
        *entering_cost = least, row = start_row, column = start_column;                                                \
        for (;;) {                                                                                                     \
            for (nk_size_t step = 0, vector_length; step < columns - column; step += vector_length) {                  \
                vector_length = set_vector_length(columns - column - step);                                            \
                vfloat64m2_t const reduced_f64m2 = nk_shapes_reduced_costs_##math_type##_rvv_(                         \
                    views, gradient, rows, columns, row, column + step, vector_length);                                \
                long const first = __riscv_vfirst_m_b32(                                                               \
                    __riscv_vmfeq_vf_f64m2_b32(reduced_f64m2, least, vector_length), vector_length);                   \
                if (first < 0) continue;                                                                               \
                *entering_row = row, *entering_column = column + step + (nk_size_t)first;                              \
                return;                                                                                                \
            }                                                                                                          \
            column = 0, row = row + 1 == rows ? 0 : row + 1;                                                           \
        }                                                                                                              \
    }

nk_define_shapes_pricing_rvv_(f64, vfloat64m2_t, f64m2, __riscv_vle64_v_f64m2, __riscv_vfrsub_vf_f64m2,
                              __riscv_vfmul_vv_f64m2, __riscv_vfmacc_vv_f64m2, __riscv_vfmv_f_s_f64m2_f64,
                              __riscv_vmv_v_v_f64m2, __riscv_vsetvl_e64m2)
nk_define_shapes_pricing_rvv_(f32, vfloat32m1_t, f32m1, __riscv_vle32_v_f32m1, __riscv_vfrsub_vf_f32m1,
                              __riscv_vfmul_vv_f32m1, __riscv_vfmacc_vv_f32m1, __riscv_vfmv_f_s_f32m1_f32,
                              __riscv_vfwcvt_f_f_v_f64m2, __riscv_vsetvl_e32m1)
#undef nk_define_shapes_pricing_rvv_

nk_define_shapes_network_simplex_(f32, rvv, nk_shapes_point_cost_f32_rvv_, nk_shapes_price_block_f32_rvv_)
nk_define_shapes_network_simplex_(f64, rvv, nk_shapes_point_cost_f64_rvv_, nk_shapes_price_block_f64_rvv_)
nk_define_shapes_wasserstein_procrustes_(f32, rvv, NUMKONG_F32_INF)
nk_define_shapes_wasserstein_procrustes_(f64, rvv, NUMKONG_F64_INF)
nk_define_shapes_gromov_wasserstein_(f32, rvv)
nk_define_shapes_gromov_wasserstein_(f64, rvv)

nk_define_shapes_flip_kernels_(f16, f32, rvv, nk_shapes_nearest_f16_rvv_, NUMKONG_F32_INF)
nk_define_shapes_flip_kernels_(f32, f64, rvv, nk_shapes_nearest_f32_rvv_, NUMKONG_F64_INF)
nk_define_shapes_transport_kernels_(f16, f32, rvv, NUMKONG_F32_INF)
nk_define_shapes_transport_kernels_(f32, f64, rvv, NUMKONG_F64_INF)

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
