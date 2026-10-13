/**
 *  @file include/numkong/shapes/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 10, 2026
 *  @brief Many-to-many shape distances between 3D point clouds for WASM Relaxed SIMD.
 *
 *  @sa include/numkong/shapes.h
 *
 *  Chamfer and Hausdorff run the serial pair driver around a nearest-partner pass over a vector of
 *  inner points: each outer point keeps 8 running minima, one per axis flip, in the serial
 *  difference form and summation order. A shuffle tree folds the 8 minima into lanes of one vector
 *  per 4 flips, so the square roots, sums and maxima of a pass run across flips. Squared distances
 *  are never −0 or NaN for finite packs, so the relaxed minima and maxima match the serial ones.
 *
 *  - @c f32 packs widen 2 points at a time to @c f64, and an odd last point pairs with an x of +∞
 *  - @c f16 packs widen 4 points per load to @c f32, since WASM has no @c f16 lanes, and the last
 *    quadruple of a count off the multiple takes an x of +∞ in its padded lanes
 *
 *  The transport kinds run the serial solvers around block pricing two arcs at a time. Arc costs
 *  round in separate multiplies and adds, never @c relaxed_madd, whose fusing differs between
 *  engines, and the scalar cost behind the tree's potentials runs the same lanes, so tree arcs keep
 *  a zero reduced cost.
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
#ifndef NUMKONG_SHAPES_V128RELAXED_H
#define NUMKONG_SHAPES_V128RELAXED_H

#if NUMKONG_ARCH_WASM_
#if NUMKONG_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/cast/serial.h"      // `nk_load_b64_serial_`
#include "numkong/cast/v128relaxed.h" // `nk_f16x4_to_f32x4_v128relaxed_`
#include "numkong/shapes/serial.h"    // `nk_define_shapes_network_simplex_`, `nk_shapes_workspace_f64_serial_t`

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

/** One directed pass of Chamfer and Hausdorff over @c f16 packs, like the serial one, widening four
 *  inner points per load. The last quadruple of a count off the multiple reads the pack's zero
 *  padding and takes an infinite x in its padded lanes. */
NUMKONG_INLINE void nk_shapes_nearest_f16_v128relaxed_(nk_f16_t const *outer_axes, nk_size_t outer_stride,
                                                       nk_size_t outer_count, nk_f16_t const *inner_axes,
                                                       nk_size_t inner_stride, nk_size_t inner_count, nk_f32_t *sums,
                                                       nk_f32_t *maxima) {
    nk_size_t const quadruples_end = inner_count & ~(nk_size_t)3;
    v128_t const infinity_f32x4 = wasm_f32x4_splat(NUMKONG_F32_INF);
    nk_b128_vec_t tail_vec;
    for (nk_size_t lane = 0; lane != 4; ++lane) tail_vec.u32s[lane] = quadruples_end + lane < inner_count ? ~0u : 0;
    v128_t sums_f32x4[2] = {wasm_v128_load(sums), wasm_v128_load(sums + 4)};
    v128_t maxima_f32x4[2] = {wasm_v128_load(maxima), wasm_v128_load(maxima + 4)};
    nk_b64_vec_t inner_x_vec, inner_y_vec, inner_z_vec;
    for (nk_size_t outer = 0; outer != outer_count; ++outer) {
        nk_f32_t outer_x, outer_y, outer_z;
        nk_f16_to_f32_serial_(outer_axes + outer, &outer_x);
        nk_f16_to_f32_serial_(outer_axes + outer_stride + outer, &outer_y);
        nk_f16_to_f32_serial_(outer_axes + 2 * outer_stride + outer, &outer_z);
        v128_t const outer_x_f32x4 = wasm_f32x4_splat(outer_x), outer_y_f32x4 = wasm_f32x4_splat(outer_y);
        v128_t const outer_z_f32x4 = wasm_f32x4_splat(outer_z);
        v128_t nearest_f32x4[8];
        for (int flip = 0; flip != 8; ++flip) nearest_f32x4[flip] = infinity_f32x4;
        for (nk_size_t inner = 0; inner != quadruples_end; inner += 4) {
            nk_load_b64_serial_(inner_axes + inner, &inner_x_vec);
            nk_load_b64_serial_(inner_axes + inner_stride + inner, &inner_y_vec);
            nk_load_b64_serial_(inner_axes + 2 * inner_stride + inner, &inner_z_vec);
            nk_shapes_lower_nearest_f32x4_v128relaxed_(nearest_f32x4, outer_x_f32x4, outer_y_f32x4, outer_z_f32x4,
                                                       nk_f16x4_to_f32x4_v128relaxed_(inner_x_vec).v128,
                                                       nk_f16x4_to_f32x4_v128relaxed_(inner_y_vec).v128,
                                                       nk_f16x4_to_f32x4_v128relaxed_(inner_z_vec).v128);
        }
        if (quadruples_end != inner_count) {
            nk_load_b64_serial_(inner_axes + quadruples_end, &inner_x_vec);
            nk_load_b64_serial_(inner_axes + inner_stride + quadruples_end, &inner_y_vec);
            nk_load_b64_serial_(inner_axes + 2 * inner_stride + quadruples_end, &inner_z_vec);
            nk_shapes_lower_nearest_f32x4_v128relaxed_(
                nearest_f32x4, outer_x_f32x4, outer_y_f32x4, outer_z_f32x4,
                wasm_v128_bitselect(nk_f16x4_to_f32x4_v128relaxed_(inner_x_vec).v128, infinity_f32x4, tail_vec.v128),
                nk_f16x4_to_f32x4_v128relaxed_(inner_y_vec).v128, nk_f16x4_to_f32x4_v128relaxed_(inner_z_vec).v128);
        }
        v128_t pairs_f32x4[4];
        for (int pair = 0; pair != 4; ++pair)
            pairs_f32x4[pair] = wasm_f32x4_relaxed_min(
                wasm_i32x4_shuffle(nearest_f32x4[2 * pair], nearest_f32x4[2 * pair + 1], 0, 1, 4, 5),
                wasm_i32x4_shuffle(nearest_f32x4[2 * pair], nearest_f32x4[2 * pair + 1], 2, 3, 6, 7));
        for (int half = 0; half != 2; ++half) {
            v128_t const distance_f32x4 = wasm_f32x4_sqrt(wasm_f32x4_relaxed_min(
                wasm_i32x4_shuffle(pairs_f32x4[2 * half], pairs_f32x4[2 * half + 1], 0, 2, 4, 6),
                wasm_i32x4_shuffle(pairs_f32x4[2 * half], pairs_f32x4[2 * half + 1], 1, 3, 5, 7)));
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

/** The block pricing of one math type over v128 lanes of two columns: squared distances from a
 *  broadcast row point, unfused; the scalar cost behind the potentials, run on the same lanes so
 *  tree arcs keep a zero reduced cost; the reduced costs c(i, j) − πᵢ + πⱼ of a row's arcs to two
 *  columns, with matrix costs −G taken as πⱼ − (G + πᵢ), which rounds the same; and the scan of
 *  @p count of them from the cursor in row-major order, keeping the first most negative one below
 *  @p entering_cost as the serial scan does. A first pass takes their least two columns at a time,
 *  and only a block whose least beats @p entering_cost is scanned again for the first arc reaching
 *  it. Lanes past the scanned part of a row read neighbouring workspace and take an infinite cost.
 *  @p widen turns the costs into f64 lanes, and stays empty when they are. */
#define nk_define_shapes_pricing_v128relaxed_(math_type, vector_suffix, load_broadcast, load, subtract, multiply, add, \
                                              first_lane, widen)                                                       \
    NUMKONG_INLINE v128_t nk_shapes_point_costs_##vector_suffix##_v128relaxed_(                                        \
        v128_t row_x_##vector_suffix, v128_t row_y_##vector_suffix, v128_t row_z_##vector_suffix,                      \
        v128_t column_x_##vector_suffix, v128_t column_y_##vector_suffix, v128_t column_z_##vector_suffix) {           \
        v128_t const delta_x_##vector_suffix = subtract(row_x_##vector_suffix, column_x_##vector_suffix);              \
        v128_t const delta_y_##vector_suffix = subtract(row_y_##vector_suffix, column_y_##vector_suffix);              \
        v128_t const delta_z_##vector_suffix = subtract(row_z_##vector_suffix, column_z_##vector_suffix);              \
        v128_t const planar_##vector_suffix = add(multiply(delta_x_##vector_suffix, delta_x_##vector_suffix),          \
                                                  multiply(delta_y_##vector_suffix, delta_y_##vector_suffix));         \
        return add(planar_##vector_suffix, multiply(delta_z_##vector_suffix, delta_z_##vector_suffix));                \
    }                                                                                                                  \
    NUMKONG_INLINE nk_##math_type##_t nk_shapes_point_cost_##math_type##_v128relaxed_(                                 \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_size_t row, nk_size_t column) {                    \
        nk_size_t const stride = views->points_stride;                                                                 \
        nk_##math_type##_t const *row_points = views->row_points, *rotated_points = views->rotated_points;             \
        return first_lane(nk_shapes_point_costs_##vector_suffix##_v128relaxed_(                                        \
                              load_broadcast(row_points + row), load_broadcast(row_points + stride + row),             \
                              load_broadcast(row_points + 2 * stride + row), load_broadcast(rotated_points + column),  \
                              load_broadcast(rotated_points + stride + column),                                        \
                              load_broadcast(rotated_points + 2 * stride + column)),                                   \
                          0);                                                                                          \
    }                                                                                                                  \
    NUMKONG_INLINE v128_t nk_shapes_reduced_costs_##math_type##_v128relaxed_(                                          \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_f64_t const *gradient, nk_size_t rows,             \
        nk_size_t columns, nk_size_t row, nk_size_t column) {                                                          \
        nk_size_t const stride = views->points_stride;                                                                 \
        nk_##math_type##_t const *row_points = views->row_points, *rotated_points = views->rotated_points;             \
        v128_t const row_potential_f64x2 = wasm_v128_load64_splat(views->potential + row);                             \
        v128_t const column_potentials_f64x2 = wasm_v128_load(views->potential + rows + column);                       \
        if (gradient)                                                                                                  \
            return wasm_f64x2_sub(                                                                                     \
                column_potentials_f64x2,                                                                               \
                wasm_f64x2_add(wasm_v128_load(gradient + row * columns + column), row_potential_f64x2));               \
        v128_t const costs_f64x2 = widen(nk_shapes_point_costs_##vector_suffix##_v128relaxed_(                         \
            load_broadcast(row_points + row), load_broadcast(row_points + stride + row),                               \
            load_broadcast(row_points + 2 * stride + row), load(rotated_points + column),                              \
            load(rotated_points + stride + column), load(rotated_points + 2 * stride + column)));                      \
        return wasm_f64x2_add(wasm_f64x2_sub(costs_f64x2, row_potential_f64x2), column_potentials_f64x2);              \
    }                                                                                                                  \
    NUMKONG_INLINE void nk_shapes_price_block_##math_type##_v128relaxed_(                                              \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_f64_t const *gradient, nk_size_t rows,             \
        nk_size_t columns, nk_size_t count, nk_size_t *cursor_row, nk_size_t *cursor_column, nk_f64_t *entering_cost,  \
        nk_size_t *entering_row, nk_size_t *entering_column) {                                                         \
        v128_t const infinity_f64x2 = wasm_f64x2_splat(NUMKONG_F64_INF), first_lane_u64x2 = wasm_i64x2_make(-1, 0);    \
        nk_size_t const start_row = *cursor_row, start_column = *cursor_column;                                        \
        nk_size_t row = start_row, column = start_column;                                                              \
        v128_t least_f64x2 = infinity_f64x2;                                                                           \
        for (nk_size_t offset = 0; offset != count;) {                                                                 \
            nk_size_t const segment = columns - column < count - offset ? columns - column : count - offset;           \
            nk_size_t const paired = segment & ~(nk_size_t)1;                                                          \
            for (nk_size_t step = 0; step != paired; step += 2)                                                        \
                least_f64x2 = wasm_f64x2_relaxed_min(                                                                  \
                    least_f64x2, nk_shapes_reduced_costs_##math_type##_v128relaxed_(views, gradient, rows, columns,    \
                                                                                    row, column + step));              \
            if (segment & 1)                                                                                           \
                least_f64x2 = wasm_f64x2_relaxed_min(                                                                  \
                    least_f64x2, wasm_v128_bitselect(nk_shapes_reduced_costs_##math_type##_v128relaxed_(               \
                                                         views, gradient, rows, columns, row, column + paired),        \
                                                     infinity_f64x2, first_lane_u64x2));                               \
            offset += segment, column += segment;                                                                      \
            if (column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;                                    \
        }                                                                                                              \
        *cursor_row = row, *cursor_column = column;                                                                    \
        nk_f64_t const least_low = wasm_f64x2_extract_lane(least_f64x2, 0);                                            \
        nk_f64_t const least_high = wasm_f64x2_extract_lane(least_f64x2, 1);                                           \
        nk_f64_t const least = least_low < least_high ? least_low : least_high;                                        \
        if (!(least < *entering_cost)) return;                                                                         \
        v128_t const least_broadcast_f64x2 = wasm_f64x2_splat(least);                                                  \
        *entering_cost = least, row = start_row, column = start_column;                                                \
        for (;;) {                                                                                                     \
            nk_size_t const segment = columns - column;                                                                \
            for (nk_size_t step = 0; step < segment; step += 2) {                                                      \
                v128_t reduced_f64x2 = nk_shapes_reduced_costs_##math_type##_v128relaxed_(                             \
                    views, gradient, rows, columns, row, column + step);                                               \
                if (step + 2 > segment)                                                                                \
                    reduced_f64x2 = wasm_v128_bitselect(reduced_f64x2, infinity_f64x2, first_lane_u64x2);              \
                int const equal_lanes = wasm_i64x2_bitmask(wasm_f64x2_eq(reduced_f64x2, least_broadcast_f64x2));       \
                if (!equal_lanes) continue;                                                                            \
                *entering_row = row, *entering_column = column + step + !(equal_lanes & 1);                            \
                return;                                                                                                \
            }                                                                                                          \
            column = 0, row = row + 1 == rows ? 0 : row + 1;                                                           \
        }                                                                                                              \
    }

nk_define_shapes_pricing_v128relaxed_(f64, f64x2, wasm_v128_load64_splat, wasm_v128_load, wasm_f64x2_sub,
                                      wasm_f64x2_mul, wasm_f64x2_add, wasm_f64x2_extract_lane, )
nk_define_shapes_pricing_v128relaxed_(f32, f32x4, wasm_v128_load32_splat, wasm_v128_load64_zero, wasm_f32x4_sub,
                                      wasm_f32x4_mul, wasm_f32x4_add, wasm_f32x4_extract_lane,
                                      wasm_f64x2_promote_low_f32x4)
#undef nk_define_shapes_pricing_v128relaxed_

nk_define_shapes_network_simplex_(f32, v128relaxed, nk_shapes_point_cost_f32_v128relaxed_,
                                  nk_shapes_price_block_f32_v128relaxed_)
nk_define_shapes_network_simplex_(f64, v128relaxed, nk_shapes_point_cost_f64_v128relaxed_,
                                  nk_shapes_price_block_f64_v128relaxed_)
nk_define_shapes_wasserstein_procrustes_(f32, v128relaxed, NUMKONG_F32_INF)
nk_define_shapes_wasserstein_procrustes_(f64, v128relaxed, NUMKONG_F64_INF)
nk_define_shapes_gromov_wasserstein_(f32, v128relaxed)
nk_define_shapes_gromov_wasserstein_(f64, v128relaxed)

nk_define_shapes_flip_kernels_(f16, f32, v128relaxed, nk_shapes_nearest_f16_v128relaxed_, NUMKONG_F32_INF)
nk_define_shapes_flip_kernels_(f32, f64, v128relaxed, nk_shapes_nearest_f32_v128relaxed_, NUMKONG_F64_INF)
nk_define_shapes_transport_kernels_(f16, f32, v128relaxed, NUMKONG_F32_INF)
nk_define_shapes_transport_kernels_(f32, f64, v128relaxed, NUMKONG_F64_INF)

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_V128RELAXED
#endif // NUMKONG_ARCH_WASM_
#endif // NUMKONG_SHAPES_V128RELAXED_H
