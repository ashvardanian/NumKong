/**
 *  @file include/numkong/meshes/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 10, 2026
 *  @brief Many-to-many shape distances between 3D point clouds for WASM Relaxed SIMD.
 *
 *  @sa include/numkong/meshes.h
 *
 *  Chamfer and Hausdorff run the serial pair driver around a nearest-partner pass over a vector of
 *  inner points: each outer point keeps 8 running minima, one per axis flip, in the serial
 *  difference form and summation order. A shuffle tree folds the 8 minima into lanes of one vector
 *  per 4 flips, so the square roots, sums and maxima of a pass run across flips. Squared distances
 *  are never −0 or NaN for finite packs, so the relaxed minima and maxima match the serial ones.
 *
 *  - @c f32 packs widen 2 points at a time to @c f64, and an odd last point pairs with an x of +∞
 *  - @c f16 packs widen once per block of outer points into an @c f32 chunk on the stack, whose x
 *    rounds up to whole vectors with +∞, since WASM has no @c f16 lanes
 *
 *  The transport kinds run the serial solvers around vector passes: block pricing two arcs at a
 *  time, and the f64 Gromov-Wasserstein gradient products, plan steps and scores. Arc costs round
 *  in separate multiplies and adds, never @c relaxed_madd, whose fusing differs between engines,
 *  and the scalar cost behind the tree's potentials runs the same lanes, so tree arcs keep a zero
 *  reduced cost. Odd tails stay out of the inner loops, as a mask in every step slows
 *  Gromov-Wasserstein by a fifth.
 *
 *  @section shapes_v128relaxed_instructions Key WASM Shape Instructions
 *
 *  @verbatim
 *  Intrinsic                               Operation
 *  wasm_f64x2_promote_low_f32x4            Widens 2 f32 coordinates to f64
 *  wasm_f32x4_relaxed_min                  Running minima, 1 instruction where pmin takes 2 on Arm
 *  wasm_i32x4_shuffle, wasm_i64x2_shuffle  Fold tree from 8 minima vectors to flips in lanes
 *  wasm_f32x4_sqrt, wasm_f64x2_sqrt        Distances of 4 or 2 flips at once
 *  wasm_v128_bitselect                     Infinite reduced costs past a row's priced arcs
 *  wasm_i64x2_bitmask                      First lane reaching a block's least reduced cost
 *  @endverbatim
 */
#ifndef NUMKONG_MESHES_V128RELAXED_H
#define NUMKONG_MESHES_V128RELAXED_H

#if NUMKONG_ARCH_WASM_
#if NUMKONG_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/cast/serial.h"      // `nk_load_b64_serial_`
#include "numkong/cast/v128relaxed.h" // `nk_f16x4_to_f32x4_v128relaxed_`
#include "numkong/meshes/serial.h"    // `nk_define_shapes_network_simplex_`, `nk_shapes_workspace_f64_serial_t`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/** Lowers the 8 running minima of one outer point by a vector of inner points. */
NUMKONG_INLINE void nk_shapes_lower_nearest_f32x4_v128relaxed_(v128_t *nearest_f32x4, v128_t outer_x_f32x4,
                                                               v128_t outer_y_f32x4, v128_t outer_z_f32x4,
                                                               v128_t inner_x_f32x4, v128_t inner_y_f32x4,
                                                               v128_t inner_z_f32x4) {
    // A flip negates an inner axis, turning its difference into a sum
    v128_t const minus_x_f32x4 = wasm_f32x4_sub(outer_x_f32x4, inner_x_f32x4);
    v128_t const plus_x_f32x4 = wasm_f32x4_add(outer_x_f32x4, inner_x_f32x4);
    v128_t const minus_y_f32x4 = wasm_f32x4_sub(outer_y_f32x4, inner_y_f32x4);
    v128_t const plus_y_f32x4 = wasm_f32x4_add(outer_y_f32x4, inner_y_f32x4);
    v128_t const minus_z_f32x4 = wasm_f32x4_sub(outer_z_f32x4, inner_z_f32x4);
    v128_t const plus_z_f32x4 = wasm_f32x4_add(outer_z_f32x4, inner_z_f32x4);
    v128_t const minus_xx_f32x4 = wasm_f32x4_mul(minus_x_f32x4, minus_x_f32x4);
    v128_t const plus_xx_f32x4 = wasm_f32x4_mul(plus_x_f32x4, plus_x_f32x4);
    v128_t const minus_yy_f32x4 = wasm_f32x4_mul(minus_y_f32x4, minus_y_f32x4);
    v128_t const plus_yy_f32x4 = wasm_f32x4_mul(plus_y_f32x4, plus_y_f32x4);
    v128_t const minus_zz_f32x4 = wasm_f32x4_mul(minus_z_f32x4, minus_z_f32x4);
    v128_t const plus_zz_f32x4 = wasm_f32x4_mul(plus_z_f32x4, plus_z_f32x4);
    v128_t const planar_f32x4[4] = {
        wasm_f32x4_add(minus_xx_f32x4, minus_yy_f32x4), wasm_f32x4_add(plus_xx_f32x4, minus_yy_f32x4),
        wasm_f32x4_add(minus_xx_f32x4, plus_yy_f32x4), wasm_f32x4_add(plus_xx_f32x4, plus_yy_f32x4)};
    for (int flip = 0; flip != 8; ++flip)
        nearest_f32x4[flip] = wasm_f32x4_relaxed_min(
            nearest_f32x4[flip], wasm_f32x4_add(planar_f32x4[flip & 3], (flip & 4) ? plus_zz_f32x4 : minus_zz_f32x4));
}

/** Lowers the 8 running minima of one outer point by a vector of inner points. */
NUMKONG_INLINE void nk_shapes_lower_nearest_f64x2_v128relaxed_(v128_t *nearest_f64x2, v128_t outer_x_f64x2,
                                                               v128_t outer_y_f64x2, v128_t outer_z_f64x2,
                                                               v128_t inner_x_f64x2, v128_t inner_y_f64x2,
                                                               v128_t inner_z_f64x2) {
    v128_t const minus_x_f64x2 = wasm_f64x2_sub(outer_x_f64x2, inner_x_f64x2);
    v128_t const plus_x_f64x2 = wasm_f64x2_add(outer_x_f64x2, inner_x_f64x2);
    v128_t const minus_y_f64x2 = wasm_f64x2_sub(outer_y_f64x2, inner_y_f64x2);
    v128_t const plus_y_f64x2 = wasm_f64x2_add(outer_y_f64x2, inner_y_f64x2);
    v128_t const minus_z_f64x2 = wasm_f64x2_sub(outer_z_f64x2, inner_z_f64x2);
    v128_t const plus_z_f64x2 = wasm_f64x2_add(outer_z_f64x2, inner_z_f64x2);
    v128_t const minus_xx_f64x2 = wasm_f64x2_mul(minus_x_f64x2, minus_x_f64x2);
    v128_t const plus_xx_f64x2 = wasm_f64x2_mul(plus_x_f64x2, plus_x_f64x2);
    v128_t const minus_yy_f64x2 = wasm_f64x2_mul(minus_y_f64x2, minus_y_f64x2);
    v128_t const plus_yy_f64x2 = wasm_f64x2_mul(plus_y_f64x2, plus_y_f64x2);
    v128_t const minus_zz_f64x2 = wasm_f64x2_mul(minus_z_f64x2, minus_z_f64x2);
    v128_t const plus_zz_f64x2 = wasm_f64x2_mul(plus_z_f64x2, plus_z_f64x2);
    v128_t const planar_f64x2[4] = {
        wasm_f64x2_add(minus_xx_f64x2, minus_yy_f64x2), wasm_f64x2_add(plus_xx_f64x2, minus_yy_f64x2),
        wasm_f64x2_add(minus_xx_f64x2, plus_yy_f64x2), wasm_f64x2_add(plus_xx_f64x2, plus_yy_f64x2)};
    for (int flip = 0; flip != 8; ++flip)
        nearest_f64x2[flip] = wasm_f64x2_relaxed_min(
            nearest_f64x2[flip], wasm_f64x2_add(planar_f64x2[flip & 3], (flip & 4) ? plus_zz_f64x2 : minus_zz_f64x2));
}

/** One directed pass of Chamfer and Hausdorff over @c f16 packs, like the serial one. The inner
 *  cloud widens into an @c f32 chunk once per block of outer points, whose minima persist across
 *  chunks, so the conversion cost spreads over the whole block. */
NUMKONG_INLINE void nk_shapes_nearest_f16_v128relaxed_(nk_f16_t const *outer_axes, nk_size_t outer_stride,
                                                       nk_size_t outer_count, nk_f16_t const *inner_axes,
                                                       nk_size_t inner_stride, nk_size_t inner_count, nk_f32_t *sums,
                                                       nk_f32_t *maxima) {
    enum { chunk_capacity = 256, block_capacity = 32 };
    nk_f32_t chunk[3][chunk_capacity];
    v128_t block_nearest_f32x4[block_capacity][2];
    v128_t const infinity_f32x4 = wasm_f32x4_splat(NUMKONG_F32_INF);
    v128_t sums_f32x4[2] = {wasm_v128_load(sums), wasm_v128_load(sums + 4)};
    v128_t maxima_f32x4[2] = {wasm_v128_load(maxima), wasm_v128_load(maxima + 4)};
    for (nk_size_t block_start = 0; block_start < outer_count; block_start += block_capacity) {
        nk_size_t const block_count = outer_count - block_start < block_capacity ? outer_count - block_start
                                                                                 : block_capacity;
        for (nk_size_t outer = 0; outer != block_count; ++outer)
            block_nearest_f32x4[outer][0] = infinity_f32x4, block_nearest_f32x4[outer][1] = infinity_f32x4;
        for (nk_size_t chunk_start = 0; chunk_start < inner_count; chunk_start += chunk_capacity) {
            nk_size_t const chunk_count = inner_count - chunk_start < chunk_capacity ? inner_count - chunk_start
                                                                                     : chunk_capacity;
            for (nk_size_t inner = 0; inner < chunk_count; inner += 4)
                for (int axis = 0; axis != 3; ++axis) {
                    nk_b64_vec_t inner_f16_vec;
                    nk_load_b64_serial_(inner_axes + axis * inner_stride + chunk_start + inner, &inner_f16_vec);
                    wasm_v128_store(chunk[axis] + inner, nk_f16x4_to_f32x4_v128relaxed_(inner_f16_vec).v128);
                }
            // The last vector reads the pack's zero padding, and an x of +∞ never wins a minimum
            for (nk_size_t inner = chunk_count; inner % 4; ++inner) chunk[0][inner] = NUMKONG_F32_INF;
            for (nk_size_t outer = 0; outer != block_count; ++outer) {
                nk_f32_t outer_x, outer_y, outer_z;
                nk_f16_to_f32_serial_(outer_axes + block_start + outer, &outer_x);
                nk_f16_to_f32_serial_(outer_axes + outer_stride + block_start + outer, &outer_y);
                nk_f16_to_f32_serial_(outer_axes + 2 * outer_stride + block_start + outer, &outer_z);
                v128_t const outer_x_f32x4 = wasm_f32x4_splat(outer_x), outer_y_f32x4 = wasm_f32x4_splat(outer_y);
                v128_t const outer_z_f32x4 = wasm_f32x4_splat(outer_z);
                v128_t nearest_f32x4[8];
                for (int flip = 0; flip != 8; ++flip) nearest_f32x4[flip] = infinity_f32x4;
                for (nk_size_t inner = 0; inner < chunk_count; inner += 4)
                    nk_shapes_lower_nearest_f32x4_v128relaxed_(
                        nearest_f32x4, outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, wasm_v128_load(chunk[0] + inner),
                        wasm_v128_load(chunk[1] + inner), wasm_v128_load(chunk[2] + inner));
                v128_t pairs_f32x4[4];
                for (int pair = 0; pair != 4; ++pair)
                    pairs_f32x4[pair] = wasm_f32x4_relaxed_min(
                        wasm_i32x4_shuffle(nearest_f32x4[2 * pair], nearest_f32x4[2 * pair + 1], 0, 1, 4, 5),
                        wasm_i32x4_shuffle(nearest_f32x4[2 * pair], nearest_f32x4[2 * pair + 1], 2, 3, 6, 7));
                for (int half = 0; half != 2; ++half)
                    block_nearest_f32x4[outer][half] = wasm_f32x4_relaxed_min(
                        block_nearest_f32x4[outer][half],
                        wasm_f32x4_relaxed_min(
                            wasm_i32x4_shuffle(pairs_f32x4[2 * half], pairs_f32x4[2 * half + 1], 0, 2, 4, 6),
                            wasm_i32x4_shuffle(pairs_f32x4[2 * half], pairs_f32x4[2 * half + 1], 1, 3, 5, 7)));
            }
        }
        for (nk_size_t outer = 0; outer != block_count; ++outer)
            for (int half = 0; half != 2; ++half) {
                v128_t const distance_f32x4 = wasm_f32x4_sqrt(block_nearest_f32x4[outer][half]);
                sums_f32x4[half] = wasm_f32x4_add(sums_f32x4[half], distance_f32x4);
                maxima_f32x4[half] = wasm_f32x4_relaxed_max(maxima_f32x4[half], distance_f32x4);
            }
    }
    wasm_v128_store(sums, sums_f32x4[0]), wasm_v128_store(sums + 4, sums_f32x4[1]);
    wasm_v128_store(maxima, maxima_f32x4[0]), wasm_v128_store(maxima + 4, maxima_f32x4[1]);
}

/** One directed pass of Chamfer and Hausdorff over @c f32 packs, like the serial one. */
NUMKONG_INLINE void nk_shapes_nearest_f32_v128relaxed_(nk_f32_t const *outer_axes, nk_size_t outer_stride,
                                                       nk_size_t outer_count, nk_f32_t const *inner_axes,
                                                       nk_size_t inner_stride, nk_size_t inner_count, nk_f64_t *sums,
                                                       nk_f64_t *maxima) {
    nk_size_t const paired_count = inner_count & ~(nk_size_t)1;
    v128_t const infinity_f64x2 = wasm_f64x2_splat(NUMKONG_F64_INF);
    v128_t odd_x_f64x2 = infinity_f64x2, odd_y_f64x2 = infinity_f64x2, odd_z_f64x2 = infinity_f64x2;
    if (inner_count & 1) {
        odd_x_f64x2 = wasm_f64x2_replace_lane(infinity_f64x2, 0, inner_axes[paired_count]);
        odd_y_f64x2 = wasm_f64x2_splat(inner_axes[inner_stride + paired_count]);
        odd_z_f64x2 = wasm_f64x2_splat(inner_axes[2 * inner_stride + paired_count]);
    }
    v128_t sums_f64x2[4], maxima_f64x2[4];
    for (int pair = 0; pair != 4; ++pair)
        sums_f64x2[pair] = wasm_v128_load(sums + 2 * pair), maxima_f64x2[pair] = wasm_v128_load(maxima + 2 * pair);
    for (nk_size_t outer = 0; outer != outer_count; ++outer) {
        v128_t const outer_x_f64x2 = wasm_f64x2_splat(outer_axes[outer]);
        v128_t const outer_y_f64x2 = wasm_f64x2_splat(outer_axes[outer_stride + outer]);
        v128_t const outer_z_f64x2 = wasm_f64x2_splat(outer_axes[2 * outer_stride + outer]);
        v128_t nearest_f64x2[8];
        for (int flip = 0; flip != 8; ++flip) nearest_f64x2[flip] = infinity_f64x2;
        for (nk_size_t inner = 0; inner != paired_count; inner += 2)
            nk_shapes_lower_nearest_f64x2_v128relaxed_(
                nearest_f64x2, outer_x_f64x2, outer_y_f64x2, outer_z_f64x2,
                wasm_f64x2_promote_low_f32x4(wasm_v128_load64_zero(inner_axes + inner)),
                wasm_f64x2_promote_low_f32x4(wasm_v128_load64_zero(inner_axes + inner_stride + inner)),
                wasm_f64x2_promote_low_f32x4(wasm_v128_load64_zero(inner_axes + 2 * inner_stride + inner)));
        if (inner_count & 1)
            nk_shapes_lower_nearest_f64x2_v128relaxed_(nearest_f64x2, outer_x_f64x2, outer_y_f64x2, outer_z_f64x2,
                                                       odd_x_f64x2, odd_y_f64x2, odd_z_f64x2);
        for (int pair = 0; pair != 4; ++pair) {
            v128_t const distance_f64x2 = wasm_f64x2_sqrt(
                wasm_f64x2_relaxed_min(wasm_i64x2_shuffle(nearest_f64x2[2 * pair], nearest_f64x2[2 * pair + 1], 0, 2),
                                       wasm_i64x2_shuffle(nearest_f64x2[2 * pair], nearest_f64x2[2 * pair + 1], 1, 3)));
            sums_f64x2[pair] = wasm_f64x2_add(sums_f64x2[pair], distance_f64x2);
            maxima_f64x2[pair] = wasm_f64x2_relaxed_max(maxima_f64x2[pair], distance_f64x2);
        }
    }
    for (int pair = 0; pair != 4; ++pair)
        wasm_v128_store(sums + 2 * pair, sums_f64x2[pair]), wasm_v128_store(maxima + 2 * pair, maxima_f64x2[pair]);
}

/** Squared distances from a broadcast row point to the column points in the lanes, unfused, the
 *  rounding both the block pricing and the scalar cost behind the potentials share. */
NUMKONG_INLINE v128_t nk_shapes_point_costs_f64x2_v128relaxed_(v128_t row_x_f64x2, v128_t row_y_f64x2,
                                                               v128_t row_z_f64x2, v128_t column_x_f64x2,
                                                               v128_t column_y_f64x2, v128_t column_z_f64x2) {
    v128_t const delta_x_f64x2 = wasm_f64x2_sub(row_x_f64x2, column_x_f64x2);
    v128_t const delta_y_f64x2 = wasm_f64x2_sub(row_y_f64x2, column_y_f64x2);
    v128_t const delta_z_f64x2 = wasm_f64x2_sub(row_z_f64x2, column_z_f64x2);
    v128_t const planar_f64x2 = wasm_f64x2_add(wasm_f64x2_mul(delta_x_f64x2, delta_x_f64x2),
                                               wasm_f64x2_mul(delta_y_f64x2, delta_y_f64x2));
    return wasm_f64x2_add(planar_f64x2, wasm_f64x2_mul(delta_z_f64x2, delta_z_f64x2));
}

/** @copydoc nk_shapes_point_costs_f64x2_v128relaxed_ */
NUMKONG_INLINE v128_t nk_shapes_point_costs_f32x4_v128relaxed_(v128_t row_x_f32x4, v128_t row_y_f32x4,
                                                               v128_t row_z_f32x4, v128_t column_x_f32x4,
                                                               v128_t column_y_f32x4, v128_t column_z_f32x4) {
    v128_t const delta_x_f32x4 = wasm_f32x4_sub(row_x_f32x4, column_x_f32x4);
    v128_t const delta_y_f32x4 = wasm_f32x4_sub(row_y_f32x4, column_y_f32x4);
    v128_t const delta_z_f32x4 = wasm_f32x4_sub(row_z_f32x4, column_z_f32x4);
    v128_t const planar_f32x4 = wasm_f32x4_add(wasm_f32x4_mul(delta_x_f32x4, delta_x_f32x4),
                                               wasm_f32x4_mul(delta_y_f32x4, delta_y_f32x4));
    return wasm_f32x4_add(planar_f32x4, wasm_f32x4_mul(delta_z_f32x4, delta_z_f32x4));
}

/** The cost of one arc between a row point and a rotated column point, rounded as in pricing. */
NUMKONG_INLINE nk_f64_t nk_shapes_point_cost_f64_v128relaxed_(nk_shapes_workspace_f64_serial_t const *views,
                                                              nk_size_t row, nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    return wasm_f64x2_extract_lane(
        nk_shapes_point_costs_f64x2_v128relaxed_(
            wasm_v128_load64_splat(row_points + row), wasm_v128_load64_splat(row_points + stride + row),
            wasm_v128_load64_splat(row_points + 2 * stride + row), wasm_v128_load64_splat(rotated_points + column),
            wasm_v128_load64_splat(rotated_points + stride + column),
            wasm_v128_load64_splat(rotated_points + 2 * stride + column)),
        0);
}

/** @copydoc nk_shapes_point_cost_f64_v128relaxed_ */
NUMKONG_INLINE nk_f32_t nk_shapes_point_cost_f32_v128relaxed_(nk_shapes_workspace_f32_serial_t const *views,
                                                              nk_size_t row, nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    return wasm_f32x4_extract_lane(
        nk_shapes_point_costs_f32x4_v128relaxed_(
            wasm_v128_load32_splat(row_points + row), wasm_v128_load32_splat(row_points + stride + row),
            wasm_v128_load32_splat(row_points + 2 * stride + row), wasm_v128_load32_splat(rotated_points + column),
            wasm_v128_load32_splat(rotated_points + stride + column),
            wasm_v128_load32_splat(rotated_points + 2 * stride + column)),
        0);
}

/** The reduced costs c(i, j) − πᵢ + πⱼ of row @p row's arcs to column @p column and the next, with
 *  matrix costs −G taken as πⱼ − (G + πᵢ), which rounds the same. */
NUMKONG_INLINE v128_t nk_shapes_reduced_costs_f64_v128relaxed_(nk_shapes_workspace_f64_serial_t const *views,
                                                               nk_f64_t const *gradient, nk_size_t rows,
                                                               nk_size_t columns, nk_size_t row, nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    v128_t const row_potential_f64x2 = wasm_v128_load64_splat(views->potential + row);
    v128_t const column_potentials_f64x2 = wasm_v128_load(views->potential + rows + column);
    if (gradient)
        return wasm_f64x2_sub(column_potentials_f64x2,
                              wasm_f64x2_add(wasm_v128_load(gradient + row * columns + column), row_potential_f64x2));
    v128_t const costs_f64x2 = nk_shapes_point_costs_f64x2_v128relaxed_(
        wasm_v128_load64_splat(row_points + row), wasm_v128_load64_splat(row_points + stride + row),
        wasm_v128_load64_splat(row_points + 2 * stride + row), wasm_v128_load(rotated_points + column),
        wasm_v128_load(rotated_points + stride + column), wasm_v128_load(rotated_points + 2 * stride + column));
    return wasm_f64x2_add(wasm_f64x2_sub(costs_f64x2, row_potential_f64x2), column_potentials_f64x2);
}

/** @copydoc nk_shapes_reduced_costs_f64_v128relaxed_ */
NUMKONG_INLINE v128_t nk_shapes_reduced_costs_f32_v128relaxed_(nk_shapes_workspace_f32_serial_t const *views,
                                                               nk_f64_t const *gradient, nk_size_t rows,
                                                               nk_size_t columns, nk_size_t row, nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    v128_t const row_potential_f64x2 = wasm_v128_load64_splat(views->potential + row);
    v128_t const column_potentials_f64x2 = wasm_v128_load(views->potential + rows + column);
    if (gradient)
        return wasm_f64x2_sub(column_potentials_f64x2,
                              wasm_f64x2_add(wasm_v128_load(gradient + row * columns + column), row_potential_f64x2));
    v128_t const costs_f64x2 = wasm_f64x2_promote_low_f32x4(nk_shapes_point_costs_f32x4_v128relaxed_(
        wasm_v128_load32_splat(row_points + row), wasm_v128_load32_splat(row_points + stride + row),
        wasm_v128_load32_splat(row_points + 2 * stride + row), wasm_v128_load64_zero(rotated_points + column),
        wasm_v128_load64_zero(rotated_points + stride + column),
        wasm_v128_load64_zero(rotated_points + 2 * stride + column)));
    return wasm_f64x2_add(wasm_f64x2_sub(costs_f64x2, row_potential_f64x2), column_potentials_f64x2);
}

/** Scans @p count reduced costs from the cursor in row-major order and keeps the first most
 *  negative one below @p entering_cost, as the serial scan does. A first pass takes their least two
 *  columns at a time, and only a block whose least beats @p entering_cost is scanned again for the
 *  first arc reaching it. Lanes past the scanned part of a row read neighbouring workspace and take
 *  an infinite cost. */
NUMKONG_INLINE void nk_shapes_price_block_f64_v128relaxed_(nk_shapes_workspace_f64_serial_t const *views,
                                                           nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t count, nk_size_t *cursor_row,
                                                           nk_size_t *cursor_column, nk_f64_t *entering_cost,
                                                           nk_size_t *entering_row, nk_size_t *entering_column) {
    v128_t const infinity_f64x2 = wasm_f64x2_splat(NUMKONG_F64_INF), first_lane_u64x2 = wasm_i64x2_make(-1, 0);
    nk_size_t const start_row = *cursor_row, start_column = *cursor_column;
    nk_size_t row = start_row, column = start_column;
    v128_t least_f64x2 = infinity_f64x2;
    for (nk_size_t offset = 0; offset != count;) {
        nk_size_t const segment = columns - column < count - offset ? columns - column : count - offset;
        nk_size_t const paired = segment & ~(nk_size_t)1;
        for (nk_size_t step = 0; step != paired; step += 2)
            least_f64x2 = wasm_f64x2_relaxed_min(least_f64x2, nk_shapes_reduced_costs_f64_v128relaxed_(
                                                                  views, gradient, rows, columns, row, column + step));
        if (segment & 1)
            least_f64x2 = wasm_f64x2_relaxed_min(
                least_f64x2, wasm_v128_bitselect(nk_shapes_reduced_costs_f64_v128relaxed_(
                                                     views, gradient, rows, columns, row, column + paired),
                                                 infinity_f64x2, first_lane_u64x2));
        offset += segment, column += segment;
        if (column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
    *cursor_row = row, *cursor_column = column;
    nk_f64_t const least_low = wasm_f64x2_extract_lane(least_f64x2, 0);
    nk_f64_t const least_high = wasm_f64x2_extract_lane(least_f64x2, 1);
    nk_f64_t const least = least_low < least_high ? least_low : least_high;
    if (!(least < *entering_cost)) return;
    v128_t const least_splat_f64x2 = wasm_f64x2_splat(least);
    *entering_cost = least, row = start_row, column = start_column;
    for (;;) {
        nk_size_t const segment = columns - column;
        for (nk_size_t step = 0; step < segment; step += 2) {
            v128_t reduced_f64x2 = nk_shapes_reduced_costs_f64_v128relaxed_(views, gradient, rows, columns, row,
                                                                            column + step);
            if (step + 2 > segment)
                reduced_f64x2 = wasm_v128_bitselect(reduced_f64x2, infinity_f64x2, first_lane_u64x2);
            int const equal_lanes = wasm_i64x2_bitmask(wasm_f64x2_eq(reduced_f64x2, least_splat_f64x2));
            if (!equal_lanes) continue;
            *entering_row = row, *entering_column = column + step + !(equal_lanes & 1);
            return;
        }
        column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
}

/** @copydoc nk_shapes_price_block_f64_v128relaxed_ */
NUMKONG_INLINE void nk_shapes_price_block_f32_v128relaxed_(nk_shapes_workspace_f32_serial_t const *views,
                                                           nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                           nk_size_t count, nk_size_t *cursor_row,
                                                           nk_size_t *cursor_column, nk_f64_t *entering_cost,
                                                           nk_size_t *entering_row, nk_size_t *entering_column) {
    v128_t const infinity_f64x2 = wasm_f64x2_splat(NUMKONG_F64_INF), first_lane_u64x2 = wasm_i64x2_make(-1, 0);
    nk_size_t const start_row = *cursor_row, start_column = *cursor_column;
    nk_size_t row = start_row, column = start_column;
    v128_t least_f64x2 = infinity_f64x2;
    for (nk_size_t offset = 0; offset != count;) {
        nk_size_t const segment = columns - column < count - offset ? columns - column : count - offset;
        nk_size_t const paired = segment & ~(nk_size_t)1;
        for (nk_size_t step = 0; step != paired; step += 2)
            least_f64x2 = wasm_f64x2_relaxed_min(least_f64x2, nk_shapes_reduced_costs_f32_v128relaxed_(
                                                                  views, gradient, rows, columns, row, column + step));
        if (segment & 1)
            least_f64x2 = wasm_f64x2_relaxed_min(
                least_f64x2, wasm_v128_bitselect(nk_shapes_reduced_costs_f32_v128relaxed_(
                                                     views, gradient, rows, columns, row, column + paired),
                                                 infinity_f64x2, first_lane_u64x2));
        offset += segment, column += segment;
        if (column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
    *cursor_row = row, *cursor_column = column;
    nk_f64_t const least_low = wasm_f64x2_extract_lane(least_f64x2, 0);
    nk_f64_t const least_high = wasm_f64x2_extract_lane(least_f64x2, 1);
    nk_f64_t const least = least_low < least_high ? least_low : least_high;
    if (!(least < *entering_cost)) return;
    v128_t const least_splat_f64x2 = wasm_f64x2_splat(least);
    *entering_cost = least, row = start_row, column = start_column;
    for (;;) {
        nk_size_t const segment = columns - column;
        for (nk_size_t step = 0; step < segment; step += 2) {
            v128_t reduced_f64x2 = nk_shapes_reduced_costs_f32_v128relaxed_(views, gradient, rows, columns, row,
                                                                            column + step);
            if (step + 2 > segment)
                reduced_f64x2 = wasm_v128_bitselect(reduced_f64x2, infinity_f64x2, first_lane_u64x2);
            int const equal_lanes = wasm_i64x2_bitmask(wasm_f64x2_eq(reduced_f64x2, least_splat_f64x2));
            if (!equal_lanes) continue;
            *entering_row = row, *entering_column = column + step + !(equal_lanes & 1);
            return;
        }
        column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
}

/** Dx × V × Dy into @p gradient for a plan V given by its arcs, two columns at a time down every
 *  row, each arc adding its mass times a row of Dx and a row of Dy. An odd row length spills each
 *  row's last pair into the next row, so that lane adds a zero, kept out of the inner loop. */
NUMKONG_INLINE void nk_shapes_distance_gradient_f64_v128relaxed_(nk_u32_t const *plan, nk_size_t plan_stride,
                                                                 nk_size_t arcs, nk_f64_t mass,
                                                                 nk_f64_t const *row_distances, nk_size_t rows,
                                                                 nk_f64_t const *column_distances, nk_size_t columns,
                                                                 nk_f64_t *gradient) {
    nk_size_t const paired = columns & ~(nk_size_t)1;
    for (nk_size_t cell = 0; cell < rows * columns; cell += 2) wasm_v128_store(gradient + cell, wasm_f64x2_splat(0));
    for (nk_size_t arc = 0; arc != arcs; ++arc) {
        if (!plan[2 * plan_stride + arc]) continue;
        nk_f64_t const weight = (nk_f64_t)plan[2 * plan_stride + arc] / mass;
        nk_f64_t const *row_distances_from = row_distances + plan[arc] * rows;
        nk_f64_t const *column_distances_from = column_distances + plan[plan_stride + arc] * columns;
        v128_t const odd_distance_f64x2 = wasm_f64x2_replace_lane(wasm_f64x2_splat(0), 0,
                                                                  columns & 1 ? column_distances_from[paired] : 0);
        for (nk_size_t row = 0; row != rows; ++row) {
            v128_t const weighted_f64x2 = wasm_f64x2_splat(weight * row_distances_from[row]);
            nk_f64_t *gradient_row = gradient + row * columns;
            for (nk_size_t column = 0; column != paired; column += 2)
                wasm_v128_store(
                    gradient_row + column,
                    wasm_f64x2_add(wasm_v128_load(gradient_row + column),
                                   wasm_f64x2_mul(weighted_f64x2, wasm_v128_load(column_distances_from + column))));
            if (columns & 1)
                wasm_v128_store(gradient_row + paired,
                                wasm_f64x2_add(wasm_v128_load(gradient_row + paired),
                                               wasm_f64x2_mul(weighted_f64x2, odd_distance_f64x2)));
        }
    }
}

/** @copydoc nk_shapes_plan_step_f64_serial_ */
NUMKONG_INLINE void nk_shapes_plan_step_f64_v128relaxed_(nk_f64_t *plan, nk_f64_t *gradient,
                                                         nk_f64_t const *vertex_gradient, nk_size_t cells,
                                                         nk_f64_t keep, nk_f64_t step) {
    v128_t const keep_f64x2 = wasm_f64x2_splat(keep), step_f64x2 = wasm_f64x2_splat(step);
    for (nk_size_t cell = 0; cell < cells; cell += 2) {
        wasm_v128_store(plan + cell, wasm_f64x2_mul(wasm_v128_load(plan + cell), keep_f64x2));
        wasm_v128_store(gradient + cell,
                        wasm_f64x2_add(wasm_f64x2_mul(keep_f64x2, wasm_v128_load(gradient + cell)),
                                       wasm_f64x2_mul(step_f64x2, wasm_v128_load(vertex_gradient + cell))));
    }
}

/** @copydoc nk_shapes_plan_score_f64_serial_ */
NUMKONG_INLINE nk_f64_t nk_shapes_plan_score_f64_v128relaxed_(nk_f64_t const *gradient, nk_f64_t const *plan,
                                                              nk_size_t cells, nk_f64_t *gradient_scale) {
    nk_size_t const paired = cells & ~(nk_size_t)1;
    v128_t score_f64x2 = wasm_f64x2_splat(0), scale_f64x2 = wasm_f64x2_splat(0);
    for (nk_size_t cell = 0; cell != paired; cell += 2) {
        v128_t const gradient_f64x2 = wasm_v128_load(gradient + cell);
        score_f64x2 = wasm_f64x2_add(score_f64x2, wasm_f64x2_mul(gradient_f64x2, wasm_v128_load(plan + cell)));
        scale_f64x2 = wasm_f64x2_relaxed_max(scale_f64x2, wasm_f64x2_abs(gradient_f64x2));
    }
    if (cells & 1) {
        v128_t const gradient_f64x2 = wasm_v128_load64_zero(gradient + paired);
        score_f64x2 = wasm_f64x2_add(score_f64x2, wasm_f64x2_mul(gradient_f64x2, wasm_v128_load64_zero(plan + paired)));
        scale_f64x2 = wasm_f64x2_relaxed_max(scale_f64x2, wasm_f64x2_abs(gradient_f64x2));
    }
    nk_f64_t const scale_low = wasm_f64x2_extract_lane(scale_f64x2, 0);
    nk_f64_t const scale_high = wasm_f64x2_extract_lane(scale_f64x2, 1);
    *gradient_scale = scale_low > scale_high ? scale_low : scale_high;
    return wasm_f64x2_extract_lane(score_f64x2, 0) + wasm_f64x2_extract_lane(score_f64x2, 1);
}

nk_define_shapes_network_simplex_(f32, v128relaxed, nk_shapes_point_cost_f32_v128relaxed_,
                                  nk_shapes_price_block_f32_v128relaxed_)
nk_define_shapes_network_simplex_(f64, v128relaxed, nk_shapes_point_cost_f64_v128relaxed_,
                                  nk_shapes_price_block_f64_v128relaxed_)
nk_define_shapes_wasserstein_procrustes_(f32, v128relaxed, NUMKONG_F32_INF)
nk_define_shapes_wasserstein_procrustes_(f64, v128relaxed, NUMKONG_F64_INF)
nk_define_shapes_gromov_wasserstein_(f32, v128relaxed, nk_shapes_distance_gradient_f64_v128relaxed_,
                                     nk_shapes_plan_step_f64_v128relaxed_, nk_shapes_plan_score_f64_v128relaxed_)
nk_define_shapes_gromov_wasserstein_(f64, v128relaxed, nk_shapes_distance_gradient_f64_v128relaxed_,
                                     nk_shapes_plan_step_f64_v128relaxed_, nk_shapes_plan_score_f64_v128relaxed_)

nk_define_shapes_flip_kernels_(f16, f32, v128relaxed, nk_shapes_nearest_f16_v128relaxed_, NUMKONG_F32_INF)
nk_define_shapes_flip_kernels_(f32, f64, v128relaxed, nk_shapes_nearest_f32_v128relaxed_, NUMKONG_F64_INF)
nk_define_shapes_transport_kernels_(f16, f32, v128relaxed, nk_sqrt_f32_serial_, NUMKONG_F32_INF)
nk_define_shapes_transport_kernels_(f32, f64, v128relaxed, nk_sqrt_f64_serial_, NUMKONG_F64_INF)

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_V128RELAXED
#endif // NUMKONG_ARCH_WASM_
#endif // NUMKONG_MESHES_V128RELAXED_H
