/**
 *  @file include/numkong/meshes/rvv.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *  @brief Many-to-many shape distances between 3D point clouds for RISC-V Vector, RVV.
 *
 *  @sa include/numkong/meshes.h
 *
 *  Chamfer and Hausdorff run the serial pair driver around a nearest-partner pass over a register
 *  group of inner points: each outer point keeps 8 running minima, one per axis flip, in the serial
 *  difference form and summation order. The vector length covers the last group, and the minima
 *  update leaves the lanes past it alone, so the zero padding of a pack never wins a minimum.
 *
 *  - @c f32 packs widen each load to @c f64 with @c vfwcvt
 *  - @c f16 packs widen once per block of outer points into an @c f32 chunk on the stack, by bit
 *    manipulation, since the base V extension has no @c f16 arithmetic or conversions
 *
 *  The transport kinds run the serial solvers around vector passes: block pricing a register group
 *  of arcs at a time, and the f64 Gromov-Wasserstein gradient products, plan steps and scores. The
 *  scalar cost behind the tree's potentials runs the pricing lanes at a vector length of 1, so tree
 *  arcs keep a zero reduced cost.
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
 *  __riscv_vfmacc_vf_f64m4                 Gradient rows, plan steps and scores
 *  @endverbatim
 */
#ifndef NUMKONG_MESHES_RVV_H
#define NUMKONG_MESHES_RVV_H

#if NUMKONG_ARCH_RISCV64_
#if NUMKONG_ARCH_RISCV64_RVV_
#if NUMKONG_TARGET_RVV

#include "numkong/types.h"
#include "numkong/cast/rvv.h"      // `nk_f16m1_to_f32m2_rvv_`
#include "numkong/meshes/serial.h" // `nk_define_shapes_network_simplex_`, `nk_shapes_workspace_f64_serial_t`

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/** Lowers the scalar @p minimum to the smallest lane of @p nearest_f32m1. */
NUMKONG_INLINE void nk_shapes_lower_minimum_f32m1_rvv_(vfloat32m1_t nearest_f32m1, nk_f32_t *minimum,
                                                       nk_size_t max_vector_length) {
    vfloat32m1_t const minimum_f32m1 = __riscv_vfmv_s_f_f32m1(*minimum, 1);
    *minimum = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredmin_vs_f32m1_f32m1(nearest_f32m1, minimum_f32m1, max_vector_length));
}

/** Lowers the scalar @p minimum to the smallest lane of @p nearest_f64m1. */
NUMKONG_INLINE void nk_shapes_lower_minimum_f64m1_rvv_(vfloat64m1_t nearest_f64m1, nk_f64_t *minimum,
                                                       nk_size_t max_vector_length) {
    vfloat64m1_t const minimum_f64m1 = __riscv_vfmv_s_f_f64m1(*minimum, 1);
    *minimum = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m1_f64m1(nearest_f64m1, minimum_f64m1, max_vector_length));
}

/** One directed pass of Chamfer and Hausdorff over @c f16 packs, like the serial one. The inner
 *  cloud widens into an @c f32 chunk once per block of outer points, whose minima persist across
 *  chunks, so the conversion cost spreads over the whole block. */
NUMKONG_INLINE void nk_shapes_nearest_f16_rvv_(nk_f16_t const *outer_axes, nk_size_t outer_stride,
                                               nk_size_t outer_count, nk_f16_t const *inner_axes,
                                               nk_size_t inner_stride, nk_size_t inner_count, nk_f32_t *sums,
                                               nk_f32_t *maxima) {
    enum { chunk_capacity = 256, block_capacity = 32 };
    nk_f32_t chunk[3][chunk_capacity], block_nearest[block_capacity][8];
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t const infinity_f32m1 = __riscv_vfmv_v_f_f32m1(NUMKONG_F32_INF, max_vector_length);
    for (nk_size_t block_start = 0; block_start < outer_count; block_start += block_capacity) {
        nk_size_t const block_count = outer_count - block_start < block_capacity ? outer_count - block_start
                                                                                 : block_capacity;
        for (nk_size_t outer = 0; outer != block_count; ++outer)
            for (int flip = 0; flip != 8; ++flip) block_nearest[outer][flip] = NUMKONG_F32_INF;
        for (nk_size_t chunk_start = 0; chunk_start < inner_count; chunk_start += chunk_capacity) {
            nk_size_t const chunk_count = inner_count - chunk_start < chunk_capacity ? inner_count - chunk_start
                                                                                     : chunk_capacity;
            for (nk_size_t inner = 0, vector_length; inner < chunk_count; inner += vector_length) {
                vector_length = __riscv_vsetvl_e16m1(chunk_count - inner);
                for (int axis = 0; axis != 3; ++axis) {
                    vuint16m1_t const inner_u16m1 = __riscv_vle16_v_u16m1(
                        (nk_u16_t const *)inner_axes + axis * inner_stride + chunk_start + inner, vector_length);
                    __riscv_vse32_v_f32m2(chunk[axis] + inner, nk_f16m1_to_f32m2_rvv_(inner_u16m1, vector_length),
                                          vector_length);
                }
            }
            for (nk_size_t outer = 0; outer != block_count; ++outer) {
                nk_f32_t outer_x, outer_y, outer_z;
                nk_f16_to_f32_serial_(outer_axes + block_start + outer, &outer_x);
                nk_f16_to_f32_serial_(outer_axes + outer_stride + block_start + outer, &outer_y);
                nk_f16_to_f32_serial_(outer_axes + 2 * outer_stride + block_start + outer, &outer_z);
                vfloat32m1_t nearest_0_f32m1 = infinity_f32m1, nearest_1_f32m1 = infinity_f32m1;
                vfloat32m1_t nearest_2_f32m1 = infinity_f32m1, nearest_3_f32m1 = infinity_f32m1;
                vfloat32m1_t nearest_4_f32m1 = infinity_f32m1, nearest_5_f32m1 = infinity_f32m1;
                vfloat32m1_t nearest_6_f32m1 = infinity_f32m1, nearest_7_f32m1 = infinity_f32m1;
                for (nk_size_t inner = 0, vector_length; inner < chunk_count; inner += vector_length) {
                    vector_length = __riscv_vsetvl_e32m1(chunk_count - inner);
                    vfloat32m1_t const inner_x_f32m1 = __riscv_vle32_v_f32m1(chunk[0] + inner, vector_length);
                    vfloat32m1_t const inner_y_f32m1 = __riscv_vle32_v_f32m1(chunk[1] + inner, vector_length);
                    vfloat32m1_t const inner_z_f32m1 = __riscv_vle32_v_f32m1(chunk[2] + inner, vector_length);
                    // A flip negates an inner axis, turning its difference into a sum
                    vfloat32m1_t const minus_x_f32m1 = __riscv_vfrsub_vf_f32m1(inner_x_f32m1, outer_x, vector_length);
                    vfloat32m1_t const plus_x_f32m1 = __riscv_vfadd_vf_f32m1(inner_x_f32m1, outer_x, vector_length);
                    vfloat32m1_t const minus_y_f32m1 = __riscv_vfrsub_vf_f32m1(inner_y_f32m1, outer_y, vector_length);
                    vfloat32m1_t const plus_y_f32m1 = __riscv_vfadd_vf_f32m1(inner_y_f32m1, outer_y, vector_length);
                    vfloat32m1_t const minus_z_f32m1 = __riscv_vfrsub_vf_f32m1(inner_z_f32m1, outer_z, vector_length);
                    vfloat32m1_t const plus_z_f32m1 = __riscv_vfadd_vf_f32m1(inner_z_f32m1, outer_z, vector_length);
                    vfloat32m1_t const minus_xx_f32m1 = __riscv_vfmul_vv_f32m1(minus_x_f32m1, minus_x_f32m1,
                                                                               vector_length);
                    vfloat32m1_t const plus_xx_f32m1 = __riscv_vfmul_vv_f32m1(plus_x_f32m1, plus_x_f32m1,
                                                                              vector_length);
                    vfloat32m1_t const minus_yy_f32m1 = __riscv_vfmul_vv_f32m1(minus_y_f32m1, minus_y_f32m1,
                                                                               vector_length);
                    vfloat32m1_t const plus_yy_f32m1 = __riscv_vfmul_vv_f32m1(plus_y_f32m1, plus_y_f32m1,
                                                                              vector_length);
                    vfloat32m1_t const minus_zz_f32m1 = __riscv_vfmul_vv_f32m1(minus_z_f32m1, minus_z_f32m1,
                                                                               vector_length);
                    vfloat32m1_t const plus_zz_f32m1 = __riscv_vfmul_vv_f32m1(plus_z_f32m1, plus_z_f32m1,
                                                                              vector_length);
                    vfloat32m1_t const planar_0_f32m1 = __riscv_vfadd_vv_f32m1(minus_xx_f32m1, minus_yy_f32m1,
                                                                               vector_length);
                    vfloat32m1_t const planar_1_f32m1 = __riscv_vfadd_vv_f32m1(plus_xx_f32m1, minus_yy_f32m1,
                                                                               vector_length);
                    vfloat32m1_t const planar_2_f32m1 = __riscv_vfadd_vv_f32m1(minus_xx_f32m1, plus_yy_f32m1,
                                                                               vector_length);
                    vfloat32m1_t const planar_3_f32m1 = __riscv_vfadd_vv_f32m1(plus_xx_f32m1, plus_yy_f32m1,
                                                                               vector_length);
                    nearest_0_f32m1 = __riscv_vfmin_vv_f32m1_tu(
                        nearest_0_f32m1, nearest_0_f32m1,
                        __riscv_vfadd_vv_f32m1(planar_0_f32m1, minus_zz_f32m1, vector_length), vector_length);
                    nearest_1_f32m1 = __riscv_vfmin_vv_f32m1_tu(
                        nearest_1_f32m1, nearest_1_f32m1,
                        __riscv_vfadd_vv_f32m1(planar_1_f32m1, minus_zz_f32m1, vector_length), vector_length);
                    nearest_2_f32m1 = __riscv_vfmin_vv_f32m1_tu(
                        nearest_2_f32m1, nearest_2_f32m1,
                        __riscv_vfadd_vv_f32m1(planar_2_f32m1, minus_zz_f32m1, vector_length), vector_length);
                    nearest_3_f32m1 = __riscv_vfmin_vv_f32m1_tu(
                        nearest_3_f32m1, nearest_3_f32m1,
                        __riscv_vfadd_vv_f32m1(planar_3_f32m1, minus_zz_f32m1, vector_length), vector_length);
                    nearest_4_f32m1 = __riscv_vfmin_vv_f32m1_tu(
                        nearest_4_f32m1, nearest_4_f32m1,
                        __riscv_vfadd_vv_f32m1(planar_0_f32m1, plus_zz_f32m1, vector_length), vector_length);
                    nearest_5_f32m1 = __riscv_vfmin_vv_f32m1_tu(
                        nearest_5_f32m1, nearest_5_f32m1,
                        __riscv_vfadd_vv_f32m1(planar_1_f32m1, plus_zz_f32m1, vector_length), vector_length);
                    nearest_6_f32m1 = __riscv_vfmin_vv_f32m1_tu(
                        nearest_6_f32m1, nearest_6_f32m1,
                        __riscv_vfadd_vv_f32m1(planar_2_f32m1, plus_zz_f32m1, vector_length), vector_length);
                    nearest_7_f32m1 = __riscv_vfmin_vv_f32m1_tu(
                        nearest_7_f32m1, nearest_7_f32m1,
                        __riscv_vfadd_vv_f32m1(planar_3_f32m1, plus_zz_f32m1, vector_length), vector_length);
                }
                nk_f32_t *outer_nearest = block_nearest[outer];
                nk_shapes_lower_minimum_f32m1_rvv_(nearest_0_f32m1, outer_nearest + 0, max_vector_length);
                nk_shapes_lower_minimum_f32m1_rvv_(nearest_1_f32m1, outer_nearest + 1, max_vector_length);
                nk_shapes_lower_minimum_f32m1_rvv_(nearest_2_f32m1, outer_nearest + 2, max_vector_length);
                nk_shapes_lower_minimum_f32m1_rvv_(nearest_3_f32m1, outer_nearest + 3, max_vector_length);
                nk_shapes_lower_minimum_f32m1_rvv_(nearest_4_f32m1, outer_nearest + 4, max_vector_length);
                nk_shapes_lower_minimum_f32m1_rvv_(nearest_5_f32m1, outer_nearest + 5, max_vector_length);
                nk_shapes_lower_minimum_f32m1_rvv_(nearest_6_f32m1, outer_nearest + 6, max_vector_length);
                nk_shapes_lower_minimum_f32m1_rvv_(nearest_7_f32m1, outer_nearest + 7, max_vector_length);
            }
        }
        // Every RVV register holds at least 128 bits, so a pair of them holds all 8 flips
        vfloat32m2_t sums_f32m2 = __riscv_vle32_v_f32m2(sums, 8), maxima_f32m2 = __riscv_vle32_v_f32m2(maxima, 8);
        for (nk_size_t outer = 0; outer != block_count; ++outer) {
            vfloat32m2_t const distance_f32m2 = __riscv_vfsqrt_v_f32m2(__riscv_vle32_v_f32m2(block_nearest[outer], 8),
                                                                       8);
            sums_f32m2 = __riscv_vfadd_vv_f32m2(sums_f32m2, distance_f32m2, 8);
            maxima_f32m2 = __riscv_vfmax_vv_f32m2(maxima_f32m2, distance_f32m2, 8);
        }
        __riscv_vse32_v_f32m2(sums, sums_f32m2, 8), __riscv_vse32_v_f32m2(maxima, maxima_f32m2, 8);
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

/** Squared distances from a row point to a register group of rotated column points, in the
 *  rounding that the block pricing and the scalar cost behind the potentials share. */
NUMKONG_INLINE vfloat64m2_t nk_shapes_point_costs_f64m2_rvv_(nk_shapes_workspace_f64_serial_t const *views,
                                                             nk_size_t row, nk_size_t column, nk_size_t vector_length) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    vfloat64m2_t const delta_x_f64m2 = __riscv_vfrsub_vf_f64m2(
        __riscv_vle64_v_f64m2(rotated_points + column, vector_length), row_points[row], vector_length);
    vfloat64m2_t const delta_y_f64m2 = __riscv_vfrsub_vf_f64m2(
        __riscv_vle64_v_f64m2(rotated_points + stride + column, vector_length), row_points[stride + row],
        vector_length);
    vfloat64m2_t const delta_z_f64m2 = __riscv_vfrsub_vf_f64m2(
        __riscv_vle64_v_f64m2(rotated_points + 2 * stride + column, vector_length), row_points[2 * stride + row],
        vector_length);
    vfloat64m2_t const planar_f64m2 = __riscv_vfmacc_vv_f64m2(
        __riscv_vfmul_vv_f64m2(delta_y_f64m2, delta_y_f64m2, vector_length), delta_x_f64m2, delta_x_f64m2,
        vector_length);
    return __riscv_vfmacc_vv_f64m2(planar_f64m2, delta_z_f64m2, delta_z_f64m2, vector_length);
}

/** @copydoc nk_shapes_point_costs_f64m2_rvv_ */
NUMKONG_INLINE vfloat32m1_t nk_shapes_point_costs_f32m1_rvv_(nk_shapes_workspace_f32_serial_t const *views,
                                                             nk_size_t row, nk_size_t column, nk_size_t vector_length) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    vfloat32m1_t const delta_x_f32m1 = __riscv_vfrsub_vf_f32m1(
        __riscv_vle32_v_f32m1(rotated_points + column, vector_length), row_points[row], vector_length);
    vfloat32m1_t const delta_y_f32m1 = __riscv_vfrsub_vf_f32m1(
        __riscv_vle32_v_f32m1(rotated_points + stride + column, vector_length), row_points[stride + row],
        vector_length);
    vfloat32m1_t const delta_z_f32m1 = __riscv_vfrsub_vf_f32m1(
        __riscv_vle32_v_f32m1(rotated_points + 2 * stride + column, vector_length), row_points[2 * stride + row],
        vector_length);
    vfloat32m1_t const planar_f32m1 = __riscv_vfmacc_vv_f32m1(
        __riscv_vfmul_vv_f32m1(delta_y_f32m1, delta_y_f32m1, vector_length), delta_x_f32m1, delta_x_f32m1,
        vector_length);
    return __riscv_vfmacc_vv_f32m1(planar_f32m1, delta_z_f32m1, delta_z_f32m1, vector_length);
}

/** The cost of one arc between a row point and a rotated column point, rounded as in pricing. */
NUMKONG_INLINE nk_f64_t nk_shapes_point_cost_f64_rvv_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t row,
                                                      nk_size_t column) {
    return __riscv_vfmv_f_s_f64m2_f64(nk_shapes_point_costs_f64m2_rvv_(views, row, column, 1));
}

/** @copydoc nk_shapes_point_cost_f64_rvv_ */
NUMKONG_INLINE nk_f32_t nk_shapes_point_cost_f32_rvv_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t row,
                                                      nk_size_t column) {
    return __riscv_vfmv_f_s_f32m1_f32(nk_shapes_point_costs_f32m1_rvv_(views, row, column, 1));
}

/** The reduced costs c(i, j) − πᵢ + πⱼ of row @p row's arcs to a register group of columns from
 *  @p column, with matrix costs −G taken as πⱼ − (G + πᵢ), which rounds the same. */
NUMKONG_INLINE vfloat64m2_t nk_shapes_reduced_costs_f64m2_rvv_(nk_shapes_workspace_f64_serial_t const *views,
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

/** @copydoc nk_shapes_reduced_costs_f64m2_rvv_ */
NUMKONG_INLINE vfloat64m2_t nk_shapes_reduced_costs_f32m1_rvv_(nk_shapes_workspace_f32_serial_t const *views,
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

/** Scans @p count reduced costs from the cursor in row-major order and keeps the first most
 *  negative one below @p entering_cost, as the serial scan does. A first pass takes their least a
 *  register group at a time, and only a block whose least beats @p entering_cost is scanned again
 *  for the first arc reaching it. */
NUMKONG_INLINE void nk_shapes_price_block_f64_rvv_(nk_shapes_workspace_f64_serial_t const *views,
                                                   nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                   nk_size_t count, nk_size_t *cursor_row, nk_size_t *cursor_column,
                                                   nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                   nk_size_t *entering_column) {
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t least_f64m2 = __riscv_vfmv_v_f_f64m2(NUMKONG_F64_INF, max_vector_length);
    nk_size_t const start_row = *cursor_row, start_column = *cursor_column;
    nk_size_t row = start_row, column = start_column;
    for (nk_size_t offset = 0; offset != count;) {
        nk_size_t const segment = columns - column < count - offset ? columns - column : count - offset;
        for (nk_size_t step = 0, vector_length; step < segment; step += vector_length) {
            vector_length = __riscv_vsetvl_e64m2(segment - step);
            vfloat64m2_t const reduced_f64m2 = nk_shapes_reduced_costs_f64m2_rvv_(views, gradient, rows, columns, row,
                                                                                  column + step, vector_length);
            least_f64m2 = __riscv_vfmin_vv_f64m2_tu(least_f64m2, least_f64m2, reduced_f64m2, vector_length);
        }
        offset += segment, column += segment;
        if (column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
    *cursor_row = row, *cursor_column = column;
    nk_f64_t const least = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m2_f64m1(least_f64m2, __riscv_vfmv_s_f_f64m1(NUMKONG_F64_INF, 1), max_vector_length));
    if (!(least < *entering_cost)) return;
    *entering_cost = least, row = start_row, column = start_column;
    for (;;) {
        for (nk_size_t step = 0, vector_length; step < columns - column; step += vector_length) {
            vector_length = __riscv_vsetvl_e64m2(columns - column - step);
            vfloat64m2_t const reduced_f64m2 = nk_shapes_reduced_costs_f64m2_rvv_(views, gradient, rows, columns, row,
                                                                                  column + step, vector_length);
            long const first = __riscv_vfirst_m_b32(__riscv_vmfeq_vf_f64m2_b32(reduced_f64m2, least, vector_length),
                                                    vector_length);
            if (first < 0) continue;
            *entering_row = row, *entering_column = column + step + (nk_size_t)first;
            return;
        }
        column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
}

/** @copydoc nk_shapes_price_block_f64_rvv_ */
NUMKONG_INLINE void nk_shapes_price_block_f32_rvv_(nk_shapes_workspace_f32_serial_t const *views,
                                                   nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                   nk_size_t count, nk_size_t *cursor_row, nk_size_t *cursor_column,
                                                   nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                   nk_size_t *entering_column) {
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t least_f64m2 = __riscv_vfmv_v_f_f64m2(NUMKONG_F64_INF, max_vector_length);
    nk_size_t const start_row = *cursor_row, start_column = *cursor_column;
    nk_size_t row = start_row, column = start_column;
    for (nk_size_t offset = 0; offset != count;) {
        nk_size_t const segment = columns - column < count - offset ? columns - column : count - offset;
        for (nk_size_t step = 0, vector_length; step < segment; step += vector_length) {
            vector_length = __riscv_vsetvl_e32m1(segment - step);
            vfloat64m2_t const reduced_f64m2 = nk_shapes_reduced_costs_f32m1_rvv_(views, gradient, rows, columns, row,
                                                                                  column + step, vector_length);
            least_f64m2 = __riscv_vfmin_vv_f64m2_tu(least_f64m2, least_f64m2, reduced_f64m2, vector_length);
        }
        offset += segment, column += segment;
        if (column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
    *cursor_row = row, *cursor_column = column;
    nk_f64_t const least = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmin_vs_f64m2_f64m1(least_f64m2, __riscv_vfmv_s_f_f64m1(NUMKONG_F64_INF, 1), max_vector_length));
    if (!(least < *entering_cost)) return;
    *entering_cost = least, row = start_row, column = start_column;
    for (;;) {
        for (nk_size_t step = 0, vector_length; step < columns - column; step += vector_length) {
            vector_length = __riscv_vsetvl_e32m1(columns - column - step);
            vfloat64m2_t const reduced_f64m2 = nk_shapes_reduced_costs_f32m1_rvv_(views, gradient, rows, columns, row,
                                                                                  column + step, vector_length);
            long const first = __riscv_vfirst_m_b32(__riscv_vmfeq_vf_f64m2_b32(reduced_f64m2, least, vector_length),
                                                    vector_length);
            if (first < 0) continue;
            *entering_row = row, *entering_column = column + step + (nk_size_t)first;
            return;
        }
        column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
}

/** Dx × V × Dy into @p gradient for a plan V given by its arcs, a register group of columns at a
 *  time down every row, each arc adding its mass times a row of Dx and a row of Dy. */
NUMKONG_INLINE void nk_shapes_distance_gradient_f64_rvv_(nk_u32_t const *plan, nk_size_t plan_stride, nk_size_t arcs,
                                                         nk_f64_t mass, nk_f64_t const *row_distances, nk_size_t rows,
                                                         nk_f64_t const *column_distances, nk_size_t columns,
                                                         nk_f64_t *gradient) {
    nk_size_t const cells = rows * columns;
    for (nk_size_t cell = 0, vector_length; cell < cells; cell += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(cells - cell);
        __riscv_vse64_v_f64m4(gradient + cell, __riscv_vfmv_v_f_f64m4(0, vector_length), vector_length);
    }
    for (nk_size_t arc = 0; arc != arcs; ++arc) {
        if (!plan[2 * plan_stride + arc]) continue;
        nk_f64_t const weight = (nk_f64_t)plan[2 * plan_stride + arc] / mass;
        nk_f64_t const *row_distances_from = row_distances + plan[arc] * rows;
        nk_f64_t const *column_distances_from = column_distances + plan[plan_stride + arc] * columns;
        for (nk_size_t row = 0; row != rows; ++row) {
            nk_f64_t const weighted = weight * row_distances_from[row];
            nk_f64_t *gradient_row = gradient + row * columns;
            for (nk_size_t column = 0, vector_length; column < columns; column += vector_length) {
                vector_length = __riscv_vsetvl_e64m4(columns - column);
                vfloat64m4_t const gradient_f64m4 = __riscv_vfmacc_vf_f64m4(
                    __riscv_vle64_v_f64m4(gradient_row + column, vector_length), weighted,
                    __riscv_vle64_v_f64m4(column_distances_from + column, vector_length), vector_length);
                __riscv_vse64_v_f64m4(gradient_row + column, gradient_f64m4, vector_length);
            }
        }
    }
}

/** @copydoc nk_shapes_plan_step_f64_serial_ */
NUMKONG_INLINE void nk_shapes_plan_step_f64_rvv_(nk_f64_t *plan, nk_f64_t *gradient, nk_f64_t const *vertex_gradient,
                                                 nk_size_t cells, nk_f64_t keep, nk_f64_t step) {
    for (nk_size_t cell = 0, vector_length; cell < cells; cell += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(cells - cell);
        __riscv_vse64_v_f64m4(
            plan + cell, __riscv_vfmul_vf_f64m4(__riscv_vle64_v_f64m4(plan + cell, vector_length), keep, vector_length),
            vector_length);
        vfloat64m4_t const stepped_f64m4 = __riscv_vfmul_vf_f64m4(
            __riscv_vle64_v_f64m4(vertex_gradient + cell, vector_length), step, vector_length);
        __riscv_vse64_v_f64m4(
            gradient + cell,
            __riscv_vfmacc_vf_f64m4(stepped_f64m4, keep, __riscv_vle64_v_f64m4(gradient + cell, vector_length),
                                    vector_length),
            vector_length);
    }
}

/** @copydoc nk_shapes_plan_score_f64_serial_ */
NUMKONG_INLINE nk_f64_t nk_shapes_plan_score_f64_rvv_(nk_f64_t const *gradient, nk_f64_t const *plan, nk_size_t cells,
                                                      nk_f64_t *gradient_scale) {
    nk_size_t const max_vector_length = __riscv_vsetvlmax_e64m4();
    vfloat64m4_t score_f64m4 = __riscv_vfmv_v_f_f64m4(0, max_vector_length);
    vfloat64m4_t scale_f64m4 = __riscv_vfmv_v_f_f64m4(0, max_vector_length);
    for (nk_size_t cell = 0, vector_length; cell < cells; cell += vector_length) {
        vector_length = __riscv_vsetvl_e64m4(cells - cell);
        vfloat64m4_t const gradient_f64m4 = __riscv_vle64_v_f64m4(gradient + cell, vector_length);
        score_f64m4 = __riscv_vfmacc_vv_f64m4_tu(score_f64m4, gradient_f64m4,
                                                 __riscv_vle64_v_f64m4(plan + cell, vector_length), vector_length);
        scale_f64m4 = __riscv_vfmax_vv_f64m4_tu(scale_f64m4, scale_f64m4,
                                                __riscv_vfabs_v_f64m4(gradient_f64m4, vector_length), vector_length);
    }
    vfloat64m1_t const zero_f64m1 = __riscv_vfmv_s_f_f64m1(0, 1);
    *gradient_scale = __riscv_vfmv_f_s_f64m1_f64(
        __riscv_vfredmax_vs_f64m4_f64m1(scale_f64m4, zero_f64m1, max_vector_length));
    return __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredusum_vs_f64m4_f64m1(score_f64m4, zero_f64m1, max_vector_length));
}

nk_define_shapes_network_simplex_(f32, rvv, nk_shapes_point_cost_f32_rvv_, nk_shapes_price_block_f32_rvv_)
nk_define_shapes_network_simplex_(f64, rvv, nk_shapes_point_cost_f64_rvv_, nk_shapes_price_block_f64_rvv_)
nk_define_shapes_wasserstein_procrustes_(f32, rvv, NUMKONG_F32_INF)
nk_define_shapes_wasserstein_procrustes_(f64, rvv, NUMKONG_F64_INF)
nk_define_shapes_gromov_wasserstein_(f32, rvv, nk_shapes_distance_gradient_f64_rvv_, nk_shapes_plan_step_f64_rvv_,
                                     nk_shapes_plan_score_f64_rvv_)
nk_define_shapes_gromov_wasserstein_(f64, rvv, nk_shapes_distance_gradient_f64_rvv_, nk_shapes_plan_step_f64_rvv_,
                                     nk_shapes_plan_score_f64_rvv_)

nk_define_shapes_flip_kernels_(f16, f32, rvv, nk_shapes_nearest_f16_rvv_, NUMKONG_F32_INF)
nk_define_shapes_flip_kernels_(f32, f64, rvv, nk_shapes_nearest_f32_rvv_, NUMKONG_F64_INF)
nk_define_shapes_transport_kernels_(f16, f32, rvv, nk_sqrt_f32_serial_, NUMKONG_F32_INF)
nk_define_shapes_transport_kernels_(f32, f64, rvv, nk_sqrt_f64_serial_, NUMKONG_F64_INF)

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
#endif // NUMKONG_MESHES_RVV_H
