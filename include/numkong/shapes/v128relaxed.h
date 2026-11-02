/**
 *  @file include/numkong/shapes/v128relaxed.h
 *  @author Ash Vardanian
 *  @date March 10, 2026
 *  @brief Many-to-many shape distances between 3D point clouds for WASM Relaxed SIMD.
 *
 *  @sa c/numkong/shapes.h
 *
 *  Chamfer and Hausdorff broadcast one outer point and sweep the inner cloud a vector at a time,
 *  keeping the running minima of all 8 flips in vectors. A shuffle tree folds them into lanes of
 *  one vector per 4 flips, or per 2 in f64, so the square roots, sums and maxima run across flips.
 *  Squared distances are never −0 or NaN for finite packs, so the relaxed minima and maxima match
 *  the serial ones, and padded lanes take an infinite x, so they never win a minimum.
 *
 *  - Chamfer scores both directions in that one sweep. It widens the inner cloud once per pair into
 *    the 3 workspace axes past the reverse minima, as WASM has no @c f16 lanes and converting three
 *    axes per step costs more than the rest of the step
 *  - Hausdorff has no workspace, so @c f16 packs widen 4 points per load and @c f32 packs 2, and
 *    it runs each direction until every flip passes the limit, reading only the deciding points
 *
 *  Packing measures what it stores in the same sweep, in separate multiplies and adds.
 *
 *  The transport kinds run the serial simplex around tile pricing two arcs at a time. Arc costs
 *  round in separate multiplies and adds, never @c relaxed_madd, whose fusing differs between
 *  engines, and the scalar cost behind the tree's potentials runs the same lanes, so tree arcs keep
 *  a zero reduced cost.
 *
 *  RMSD sums four @c f16 points per step in f32 lanes, or two @c f32 or @c f64 points in f64 lanes,
 *  through the zero padding. It takes @c relaxed_madd, as no two of its sums must round alike, and
 *  leaves the SVD scalar.
 *
 *  @section shapes_v128relaxed_instructions Key WASM Shape Instructions
 *
 *  @verbatim
 *  Intrinsic                               Operation
 *  wasm_f64x2_promote_low_f32x4            Widens 2 f32 coordinates to f64
 *  wasm_f32x4_demote_f64x2_zero            Rounds 2 stored coordinates to f32
 *  wasm_i16x8_shuffle, wasm_i32x4_shuffle  Split interleaved points into axes for packing
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
#include "numkong/reduce/v128.h"      // `nk_reduce_add_f32x4_v128_`, `nk_reduce_add_f64x2_v128_`
#include "numkong/shapes/serial.h"    // `nk_shapes_simplex_next_tile_serial_`, `nk_shapes_workspace_f64_serial_t`
#include "numkong/dot/v128.h"         // `nk_dot2_f64x2_v128_`, `nk_dot_stable_sum_f64x2_v128_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/** The square root of @p number , which every SIMD128 engine computes exactly. */
NUMKONG_INLINE nk_f32_t nk_sqrt_f32_v128relaxed_(nk_f32_t number) {
    return wasm_f32x4_extract_lane(wasm_f32x4_sqrt(wasm_f32x4_splat(number)), 0);
}

/** @copydoc nk_sqrt_f32_v128relaxed_ */
NUMKONG_INLINE nk_f64_t nk_sqrt_f64_v128relaxed_(nk_f64_t number) {
    return wasm_f64x2_extract_lane(wasm_f64x2_sqrt(wasm_f64x2_splat(number)), 0);
}

/** The reciprocal square root of @p number . */
NUMKONG_INLINE nk_f32_t nk_rsqrt_f32_v128relaxed_(nk_f32_t number) { return 1 / nk_sqrt_f32_v128relaxed_(number); }

/** @copydoc nk_rsqrt_f32_v128relaxed_ */
NUMKONG_INLINE nk_f64_t nk_rsqrt_f64_v128relaxed_(nk_f64_t number) { return 1 / nk_sqrt_f64_v128relaxed_(number); }

/** The parts of the 8 flips' squared distances from one outer point to a vector of inner points:
 *  flip f adds @p planar_f32x4 [f & 3] to @p depths_f32x4 [f >> 2]. */
NUMKONG_INLINE void nk_shapes_flip_parts_f32x4_v128relaxed_(v128_t outer_x_f32x4, v128_t outer_y_f32x4,
                                                            v128_t outer_z_f32x4, v128_t inner_x_f32x4,
                                                            v128_t inner_y_f32x4, v128_t inner_z_f32x4,
                                                            v128_t *planar_f32x4, v128_t *depths_f32x4) {
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
    planar_f32x4[0] = wasm_f32x4_add(minus_xx_f32x4, minus_yy_f32x4);
    planar_f32x4[1] = wasm_f32x4_add(plus_xx_f32x4, minus_yy_f32x4);
    planar_f32x4[2] = wasm_f32x4_add(minus_xx_f32x4, plus_yy_f32x4);
    planar_f32x4[3] = wasm_f32x4_add(plus_xx_f32x4, plus_yy_f32x4);
    depths_f32x4[0] = wasm_f32x4_mul(minus_z_f32x4, minus_z_f32x4);
    depths_f32x4[1] = wasm_f32x4_mul(plus_z_f32x4, plus_z_f32x4);
}

/** Lowers the running minima of the 8 flips by one outer point's squared distances to a vector
 *  of inner points. */
NUMKONG_INLINE void nk_shapes_lower_nearest_f32x4_v128relaxed_(v128_t outer_x_f32x4, v128_t outer_y_f32x4,
                                                               v128_t outer_z_f32x4, v128_t inner_x_f32x4,
                                                               v128_t inner_y_f32x4, v128_t inner_z_f32x4,
                                                               v128_t *nearest_f32x4) {
    v128_t planar_f32x4[4], depths_f32x4[2];
    nk_shapes_flip_parts_f32x4_v128relaxed_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_x_f32x4, inner_y_f32x4,
                                            inner_z_f32x4, planar_f32x4, depths_f32x4);
    for (int flip = 0; flip != 8; ++flip)
        nearest_f32x4[flip] = wasm_f32x4_relaxed_min(nearest_f32x4[flip],
                                                     wasm_f32x4_add(planar_f32x4[flip & 3], depths_f32x4[flip >> 2]));
}

/** @ref nk_shapes_lower_nearest_f32x4_v128relaxed_ that also lowers the same squares
 *  into the inner points' @p reverse_minima , 8 axes @p reverse_stride apart. */
NUMKONG_INLINE void nk_shapes_lower_mutual_f32x4_v128relaxed_(v128_t outer_x_f32x4, v128_t outer_y_f32x4,
                                                              v128_t outer_z_f32x4, v128_t inner_x_f32x4,
                                                              v128_t inner_y_f32x4, v128_t inner_z_f32x4,
                                                              v128_t *nearest_f32x4, nk_f32_t *reverse_minima,
                                                              nk_size_t reverse_stride) {
    v128_t planar_f32x4[4], depths_f32x4[2];
    nk_shapes_flip_parts_f32x4_v128relaxed_(outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, inner_x_f32x4, inner_y_f32x4,
                                            inner_z_f32x4, planar_f32x4, depths_f32x4);
    for (int flip = 0; flip != 8; ++flip) {
        v128_t const squares_f32x4 = wasm_f32x4_add(planar_f32x4[flip & 3], depths_f32x4[flip >> 2]);
        nk_f32_t *reverse = reverse_minima + flip * reverse_stride;
        nearest_f32x4[flip] = wasm_f32x4_relaxed_min(nearest_f32x4[flip], squares_f32x4);
        wasm_v128_store(reverse, wasm_f32x4_relaxed_min(wasm_v128_load(reverse), squares_f32x4));
    }
}

/** The least of every flip's nearest squares over its lanes, flips 0 to 3 in the first vector and
 *  4 to 7 in the second, through a shuffle tree. */
NUMKONG_INLINE void nk_shapes_least_flips_f32x4_v128relaxed_(v128_t const *nearest_f32x4, v128_t *least_f32x4) {
    v128_t pairs_f32x4[4];
    for (int pair = 0; pair != 4; ++pair)
        pairs_f32x4[pair] = wasm_f32x4_relaxed_min(
            wasm_i32x4_shuffle(nearest_f32x4[2 * pair], nearest_f32x4[2 * pair + 1], 0, 1, 4, 5),
            wasm_i32x4_shuffle(nearest_f32x4[2 * pair], nearest_f32x4[2 * pair + 1], 2, 3, 6, 7));
    for (int half = 0; half != 2; ++half)
        least_f32x4[half] = wasm_f32x4_relaxed_min(
            wasm_i32x4_shuffle(pairs_f32x4[2 * half], pairs_f32x4[2 * half + 1], 0, 2, 4, 6),
            wasm_i32x4_shuffle(pairs_f32x4[2 * half], pairs_f32x4[2 * half + 1], 1, 3, 5, 7));
}

/** One @c f16 outer point in @p outer_f32x4 , an axis broadcast per vector, converted in one go. */
NUMKONG_INLINE void nk_shapes_outer_point_f16_v128relaxed_(nk_f16_t const *outer_axes, nk_size_t outer_stride,
                                                           nk_size_t outer_index, v128_t *outer_f32x4) {
    nk_u16_t const *outer_bits = (nk_u16_t const *)outer_axes;
    nk_b64_vec_t point_vec;
    point_vec.u16s[0] = outer_bits[outer_index], point_vec.u16s[1] = outer_bits[outer_stride + outer_index];
    point_vec.u16s[2] = outer_bits[2 * outer_stride + outer_index], point_vec.u16s[3] = 0;
    v128_t const point_f32x4 = nk_f16x4_to_f32x4_v128relaxed_(point_vec).v128;
    outer_f32x4[0] = wasm_i32x4_shuffle(point_f32x4, point_f32x4, 0, 0, 0, 0);
    outer_f32x4[1] = wasm_i32x4_shuffle(point_f32x4, point_f32x4, 1, 1, 1, 1);
    outer_f32x4[2] = wasm_i32x4_shuffle(point_f32x4, point_f32x4, 2, 2, 2, 2);
}

/** One outer point's sweep of @c f16 inner points in f32 lanes, four at a time, the last
 *  quadruple of a count off the multiple with an infinite x in its padded lanes. */
NUMKONG_INLINE void nk_shapes_sweep_nearest_f16_v128relaxed_(v128_t const *outer_f32x4, nk_f16_t const *inner_axes,
                                                             nk_size_t inner_stride, nk_size_t inner_count,
                                                             v128_t tail_u32x4, v128_t *nearest_f32x4) {
    nk_size_t const quadruples_end = inner_count & ~(nk_size_t)3;
    nk_b64_vec_t inner_x_vec, inner_y_vec, inner_z_vec;
    for (nk_size_t inner_index = 0; inner_index != quadruples_end; inner_index += 4) {
        nk_load_b64_serial_(inner_axes + inner_index, &inner_x_vec);
        nk_load_b64_serial_(inner_axes + inner_stride + inner_index, &inner_y_vec);
        nk_load_b64_serial_(inner_axes + 2 * inner_stride + inner_index, &inner_z_vec);
        nk_shapes_lower_nearest_f32x4_v128relaxed_(outer_f32x4[0], outer_f32x4[1], outer_f32x4[2],
                                                   nk_f16x4_to_f32x4_v128relaxed_(inner_x_vec).v128,
                                                   nk_f16x4_to_f32x4_v128relaxed_(inner_y_vec).v128,
                                                   nk_f16x4_to_f32x4_v128relaxed_(inner_z_vec).v128, nearest_f32x4);
    }
    if (quadruples_end != inner_count) {
        nk_load_b64_serial_(inner_axes + quadruples_end, &inner_x_vec);
        nk_load_b64_serial_(inner_axes + inner_stride + quadruples_end, &inner_y_vec);
        nk_load_b64_serial_(inner_axes + 2 * inner_stride + quadruples_end, &inner_z_vec);
        nk_shapes_lower_nearest_f32x4_v128relaxed_(outer_f32x4[0], outer_f32x4[1], outer_f32x4[2],
                                                   wasm_v128_bitselect(nk_f16x4_to_f32x4_v128relaxed_(inner_x_vec).v128,
                                                                       wasm_f32x4_splat(NUMKONG_F32_INF), tail_u32x4),
                                                   nk_f16x4_to_f32x4_v128relaxed_(inner_y_vec).v128,
                                                   nk_f16x4_to_f32x4_v128relaxed_(inner_z_vec).v128, nearest_f32x4);
    }
}

/** Fused Chamfer flips of @c f16 packs in f32 lanes. The 3 axes past the reverse minima hold
 *  the inner cloud widened once per pair, its padded lanes with an infinite x, and padded
 *  reverse lanes hold zeros that no square lowers, so neither the sweep nor the roots treat
 *  a tail apart. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f16_v128relaxed_(void const *row_packed, void const *column_packed,
                                                                    nk_f32_t limit, void *workspace,
                                                                    nk_size_t workspace_bytes, nk_f32_t *chamfers) {
    nk_shapes_mutual_f16_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f16_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    nk_size_t const reverse_stride = views.reverse_stride, inner_count = views.inner_count;
    if (workspace_bytes < 11 * reverse_stride * sizeof(nk_f32_t)) return nk_unexpected_dimensions_k;
    nk_f32_t *reverse_minima = views.reverse_minima, *inner_axes = reverse_minima + 8 * reverse_stride;
    v128_t const infinity_f32x4 = wasm_f32x4_splat(NUMKONG_F32_INF);
    for (nk_size_t index = 0; index < 8 * reverse_stride; index += 4)
        wasm_v128_store(reverse_minima + index, infinity_f32x4);
    nk_b64_vec_t inner_vec;
    for (nk_size_t axis = 0; axis != 3; ++axis)
        for (nk_size_t index = 0; index < inner_count; index += 4) {
            nk_load_b64_serial_(views.inner_axes + axis * views.inner_stride + index, &inner_vec);
            wasm_v128_store(inner_axes + axis * reverse_stride + index, nk_f16x4_to_f32x4_v128relaxed_(inner_vec).v128);
        }
    for (nk_size_t index = inner_count; index & 3; ++index) {
        inner_axes[index] = NUMKONG_F32_INF;
        for (int flip = 0; flip != 8; ++flip) reverse_minima[flip * reverse_stride + index] = 0;
    }
    nk_f32_t const *inner_y = inner_axes + reverse_stride, *inner_z = inner_axes + 2 * reverse_stride;
    nk_f32_t const outer_points = (nk_f32_t)views.outer_count, inner_points = (nk_f32_t)inner_count;
    v128_t const threshold_f32x4 = wasm_f32x4_splat(2 * limit * outer_points);
    v128_t sums_f32x4[2] = {wasm_f32x4_splat(0), wasm_f32x4_splat(0)};
    v128_t outer_f32x4[3], nearest_f32x4[8], least_f32x4[2];
    nk_f32_t sums[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        nk_shapes_outer_point_f16_v128relaxed_(views.outer_axes, views.outer_stride, outer_index, outer_f32x4);
        for (int flip = 0; flip != 8; ++flip) nearest_f32x4[flip] = infinity_f32x4;
        for (nk_size_t inner_index = 0; inner_index < inner_count; inner_index += 4)
            nk_shapes_lower_mutual_f32x4_v128relaxed_(
                outer_f32x4[0], outer_f32x4[1], outer_f32x4[2], wasm_v128_load(inner_axes + inner_index),
                wasm_v128_load(inner_y + inner_index), wasm_v128_load(inner_z + inner_index), nearest_f32x4,
                reverse_minima + inner_index, reverse_stride);
        nk_shapes_least_flips_f32x4_v128relaxed_(nearest_f32x4, least_f32x4);
        sums_f32x4[0] = wasm_f32x4_add(sums_f32x4[0], wasm_f32x4_sqrt(least_f32x4[0]));
        sums_f32x4[1] = wasm_f32x4_add(sums_f32x4[1], wasm_f32x4_sqrt(least_f32x4[1]));
        if (wasm_i32x4_all_true(wasm_v128_and(wasm_f32x4_gt(sums_f32x4[0], threshold_f32x4),
                                              wasm_f32x4_gt(sums_f32x4[1], threshold_f32x4)))) {
            wasm_v128_store(sums, sums_f32x4[0]), wasm_v128_store(sums + 4, sums_f32x4[1]);
            if (nk_shapes_mutual_past_limit_f32_serial_(sums, outer_points, limit, chamfers)) return nk_success_k;
        }
    }
    wasm_v128_store(sums, sums_f32x4[0]), wasm_v128_store(sums + 4, sums_f32x4[1]);
    for (int flip = 0; flip != 8; ++flip) {
        nk_f32_t const *reverse = reverse_minima + flip * reverse_stride;
        v128_t sum_f32x4 = wasm_f32x4_splat(0);
        for (nk_size_t index = 0; index < inner_count; index += 4)
            sum_f32x4 = wasm_f32x4_add(sum_f32x4, wasm_f32x4_sqrt(wasm_v128_load(reverse + index)));
        chamfers[flip] = (sums[flip] / outer_points + nk_reduce_add_f32x4_v128_(sum_f32x4) / inner_points) / 2;
    }
    return nk_success_k;
}

/** Hausdorff flips of @c f16 packs in f32 lanes: a pair stops once every flip's farthest square
 *  passes @p limit , whose square rounds, so the distance confirms it. */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f16_v128relaxed_(void const *row_packed, void const *column_packed,
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
    nk_b128_vec_t row_tail_vec, column_tail_vec;
    for (nk_size_t lane = 0; lane != 4; ++lane) {
        row_tail_vec.u32s[lane] = (rows & ~(nk_size_t)3) + lane < rows ? ~0u : 0;
        column_tail_vec.u32s[lane] = (columns & ~(nk_size_t)3) + lane < columns ? ~0u : 0;
    }
    nk_f32_t const limit_squared = limit * limit;
    v128_t const limit_f32x4 = wasm_f32x4_splat(limit), limit_squared_f32x4 = wasm_f32x4_splat(limit_squared);
    v128_t const infinity_f32x4 = wasm_f32x4_splat(NUMKONG_F32_INF);
    v128_t farthest_f32x4[2] = {wasm_f32x4_splat(0), wasm_f32x4_splat(0)};
    v128_t outer_f32x4[3], nearest_f32x4[8], least_f32x4[2];
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_f16_t const *outer_axes = direction ? column_axes : row_axes;
        nk_f16_t const *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        v128_t const tail_u32x4 = direction ? row_tail_vec.v128 : column_tail_vec.v128;
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            nk_shapes_outer_point_f16_v128relaxed_(outer_axes, outer_stride, outer_index, outer_f32x4);
            for (int flip = 0; flip != 8; ++flip) nearest_f32x4[flip] = infinity_f32x4;
            nk_shapes_sweep_nearest_f16_v128relaxed_(outer_f32x4, inner_axes, inner_stride, inner_count, tail_u32x4,
                                                     nearest_f32x4);
            nk_shapes_least_flips_f32x4_v128relaxed_(nearest_f32x4, least_f32x4);
            farthest_f32x4[0] = wasm_f32x4_relaxed_max(farthest_f32x4[0], least_f32x4[0]);
            farthest_f32x4[1] = wasm_f32x4_relaxed_max(farthest_f32x4[1], least_f32x4[1]);
            rejected = wasm_i32x4_all_true(wasm_v128_and(wasm_f32x4_gt(farthest_f32x4[0], limit_squared_f32x4),
                                                         wasm_f32x4_gt(farthest_f32x4[1], limit_squared_f32x4))) &&
                       wasm_i32x4_all_true(
                           wasm_v128_and(wasm_f32x4_gt(wasm_f32x4_sqrt(farthest_f32x4[0]), limit_f32x4),
                                         wasm_f32x4_gt(wasm_f32x4_sqrt(farthest_f32x4[1]), limit_f32x4)));
        }
    }
    wasm_v128_store(hausdorffs, wasm_f32x4_sqrt(farthest_f32x4[0]));
    wasm_v128_store(hausdorffs + 4, wasm_f32x4_sqrt(farthest_f32x4[1]));
}

/** @copydoc nk_shapes_flip_parts_f32x4_v128relaxed_ */
NUMKONG_INLINE void nk_shapes_flip_parts_f64x2_v128relaxed_(v128_t outer_x_f64x2, v128_t outer_y_f64x2,
                                                            v128_t outer_z_f64x2, v128_t inner_x_f64x2,
                                                            v128_t inner_y_f64x2, v128_t inner_z_f64x2,
                                                            v128_t *planar_f64x2, v128_t *depths_f64x2) {
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
    planar_f64x2[0] = wasm_f64x2_add(minus_xx_f64x2, minus_yy_f64x2);
    planar_f64x2[1] = wasm_f64x2_add(plus_xx_f64x2, minus_yy_f64x2);
    planar_f64x2[2] = wasm_f64x2_add(minus_xx_f64x2, plus_yy_f64x2);
    planar_f64x2[3] = wasm_f64x2_add(plus_xx_f64x2, plus_yy_f64x2);
    depths_f64x2[0] = wasm_f64x2_mul(minus_z_f64x2, minus_z_f64x2);
    depths_f64x2[1] = wasm_f64x2_mul(plus_z_f64x2, plus_z_f64x2);
}

/** @copydoc nk_shapes_lower_nearest_f32x4_v128relaxed_ */
NUMKONG_INLINE void nk_shapes_lower_nearest_f64x2_v128relaxed_(v128_t outer_x_f64x2, v128_t outer_y_f64x2,
                                                               v128_t outer_z_f64x2, v128_t inner_x_f64x2,
                                                               v128_t inner_y_f64x2, v128_t inner_z_f64x2,
                                                               v128_t *nearest_f64x2) {
    v128_t planar_f64x2[4], depths_f64x2[2];
    nk_shapes_flip_parts_f64x2_v128relaxed_(outer_x_f64x2, outer_y_f64x2, outer_z_f64x2, inner_x_f64x2, inner_y_f64x2,
                                            inner_z_f64x2, planar_f64x2, depths_f64x2);
    for (int flip = 0; flip != 8; ++flip)
        nearest_f64x2[flip] = wasm_f64x2_relaxed_min(nearest_f64x2[flip],
                                                     wasm_f64x2_add(planar_f64x2[flip & 3], depths_f64x2[flip >> 2]));
}

/** @copydoc nk_shapes_lower_mutual_f32x4_v128relaxed_ */
NUMKONG_INLINE void nk_shapes_lower_mutual_f64x2_v128relaxed_(v128_t outer_x_f64x2, v128_t outer_y_f64x2,
                                                              v128_t outer_z_f64x2, v128_t inner_x_f64x2,
                                                              v128_t inner_y_f64x2, v128_t inner_z_f64x2,
                                                              v128_t *nearest_f64x2, nk_f64_t *reverse_minima,
                                                              nk_size_t reverse_stride) {
    v128_t planar_f64x2[4], depths_f64x2[2];
    nk_shapes_flip_parts_f64x2_v128relaxed_(outer_x_f64x2, outer_y_f64x2, outer_z_f64x2, inner_x_f64x2, inner_y_f64x2,
                                            inner_z_f64x2, planar_f64x2, depths_f64x2);
    for (int flip = 0; flip != 8; ++flip) {
        v128_t const squares_f64x2 = wasm_f64x2_add(planar_f64x2[flip & 3], depths_f64x2[flip >> 2]);
        nk_f64_t *reverse = reverse_minima + flip * reverse_stride;
        nearest_f64x2[flip] = wasm_f64x2_relaxed_min(nearest_f64x2[flip], squares_f64x2);
        wasm_v128_store(reverse, wasm_f64x2_relaxed_min(wasm_v128_load(reverse), squares_f64x2));
    }
}

/** The least of every flip's nearest squares over its lanes, two flips per vector. */
NUMKONG_INLINE void nk_shapes_least_flips_f64x2_v128relaxed_(v128_t const *nearest_f64x2, v128_t *least_f64x2) {
    for (int pair = 0; pair != 4; ++pair)
        least_f64x2[pair] = wasm_f64x2_relaxed_min(
            wasm_i64x2_shuffle(nearest_f64x2[2 * pair], nearest_f64x2[2 * pair + 1], 0, 2),
            wasm_i64x2_shuffle(nearest_f64x2[2 * pair], nearest_f64x2[2 * pair + 1], 1, 3));
}

/** One outer point's sweep of @c f32 inner points in f64 lanes, two at a time, the odd last
 *  point in @p odd_f64x2 , built once per direction, with an infinite x in its padded lane. */
NUMKONG_INLINE void nk_shapes_sweep_nearest_f32_v128relaxed_(v128_t const *outer_f64x2, nk_f32_t const *inner_axes,
                                                             nk_size_t inner_stride, nk_size_t inner_count,
                                                             v128_t const *odd_f64x2, v128_t *nearest_f64x2) {
    nk_size_t const pairs_end = inner_count & ~(nk_size_t)1;
    for (nk_size_t inner_index = 0; inner_index != pairs_end; inner_index += 2)
        nk_shapes_lower_nearest_f64x2_v128relaxed_(
            outer_f64x2[0], outer_f64x2[1], outer_f64x2[2],
            wasm_f64x2_promote_low_f32x4(wasm_v128_load64_zero(inner_axes + inner_index)),
            wasm_f64x2_promote_low_f32x4(wasm_v128_load64_zero(inner_axes + inner_stride + inner_index)),
            wasm_f64x2_promote_low_f32x4(wasm_v128_load64_zero(inner_axes + 2 * inner_stride + inner_index)),
            nearest_f64x2);
    if (pairs_end != inner_count)
        nk_shapes_lower_nearest_f64x2_v128relaxed_(outer_f64x2[0], outer_f64x2[1], outer_f64x2[2], odd_f64x2[0],
                                                   odd_f64x2[1], odd_f64x2[2], nearest_f64x2);
}

/** Fused Chamfer flips of @c f32 packs in f64 lanes, widening the inner cloud once per pair as the
 *  @c f16 ones do. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f32_v128relaxed_(void const *row_packed, void const *column_packed,
                                                                    nk_f64_t limit, void *workspace,
                                                                    nk_size_t workspace_bytes, nk_f64_t *chamfers) {
    nk_shapes_mutual_f32_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f32_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    nk_size_t const reverse_stride = views.reverse_stride, inner_count = views.inner_count;
    if (workspace_bytes < 11 * reverse_stride * sizeof(nk_f64_t)) return nk_unexpected_dimensions_k;
    nk_f64_t *reverse_minima = views.reverse_minima, *inner_axes = reverse_minima + 8 * reverse_stride;
    v128_t const infinity_f64x2 = wasm_f64x2_splat(NUMKONG_F64_INF);
    for (nk_size_t index = 0; index < 8 * reverse_stride; index += 2)
        wasm_v128_store(reverse_minima + index, infinity_f64x2);
    for (nk_size_t axis = 0; axis != 3; ++axis)
        for (nk_size_t index = 0; index < inner_count; index += 2) {
            v128_t const axis_f32x4 = wasm_v128_load64_zero(views.inner_axes + axis * views.inner_stride + index);
            wasm_v128_store(inner_axes + axis * reverse_stride + index, wasm_f64x2_promote_low_f32x4(axis_f32x4));
        }
    if (inner_count & 1) {
        inner_axes[inner_count] = NUMKONG_F64_INF;
        for (int flip = 0; flip != 8; ++flip) reverse_minima[flip * reverse_stride + inner_count] = 0;
    }
    nk_f64_t const *inner_y = inner_axes + reverse_stride, *inner_z = inner_axes + 2 * reverse_stride;
    nk_f64_t const outer_points = (nk_f64_t)views.outer_count, inner_points = (nk_f64_t)inner_count;
    v128_t const threshold_f64x2 = wasm_f64x2_splat(2 * limit * outer_points);
    v128_t sums_f64x2[4], nearest_f64x2[8], least_f64x2[4];
    for (int pair = 0; pair != 4; ++pair) sums_f64x2[pair] = wasm_f64x2_splat(0);
    nk_f64_t sums[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        v128_t const outer_x_f64x2 = wasm_f64x2_splat(views.outer_axes[outer_index]);
        v128_t const outer_y_f64x2 = wasm_f64x2_splat(views.outer_axes[views.outer_stride + outer_index]);
        v128_t const outer_z_f64x2 = wasm_f64x2_splat(views.outer_axes[2 * views.outer_stride + outer_index]);
        for (int flip = 0; flip != 8; ++flip) nearest_f64x2[flip] = infinity_f64x2;
        for (nk_size_t inner_index = 0; inner_index < inner_count; inner_index += 2)
            nk_shapes_lower_mutual_f64x2_v128relaxed_(
                outer_x_f64x2, outer_y_f64x2, outer_z_f64x2, wasm_v128_load(inner_axes + inner_index),
                wasm_v128_load(inner_y + inner_index), wasm_v128_load(inner_z + inner_index), nearest_f64x2,
                reverse_minima + inner_index, reverse_stride);
        nk_shapes_least_flips_f64x2_v128relaxed_(nearest_f64x2, least_f64x2);
        v128_t exceeded_u64x2 = wasm_i64x2_splat(-1);
        for (int pair = 0; pair != 4; ++pair) {
            sums_f64x2[pair] = wasm_f64x2_add(sums_f64x2[pair], wasm_f64x2_sqrt(least_f64x2[pair]));
            exceeded_u64x2 = wasm_v128_and(exceeded_u64x2, wasm_f64x2_gt(sums_f64x2[pair], threshold_f64x2));
        }
        if (wasm_i64x2_all_true(exceeded_u64x2)) {
            for (int pair = 0; pair != 4; ++pair) wasm_v128_store(sums + 2 * pair, sums_f64x2[pair]);
            if (nk_shapes_mutual_past_limit_f64_serial_(sums, outer_points, limit, chamfers)) return nk_success_k;
        }
    }
    for (int pair = 0; pair != 4; ++pair) wasm_v128_store(sums + 2 * pair, sums_f64x2[pair]);
    for (int flip = 0; flip != 8; ++flip) {
        nk_f64_t const *reverse = reverse_minima + flip * reverse_stride;
        v128_t sum_f64x2 = wasm_f64x2_splat(0);
        for (nk_size_t index = 0; index < inner_count; index += 2)
            sum_f64x2 = wasm_f64x2_add(sum_f64x2, wasm_f64x2_sqrt(wasm_v128_load(reverse + index)));
        chamfers[flip] = (sums[flip] / outer_points + nk_reduce_add_f64x2_v128_(sum_f64x2) / inner_points) / 2;
    }
    return nk_success_k;
}

/** The odd last point of @p count , an axis per vector, with an infinite x in its padded lane. */
NUMKONG_INLINE void nk_shapes_odd_point_f32_v128relaxed_(nk_f32_t const *axes, nk_size_t stride, nk_size_t count,
                                                         v128_t *odd_f64x2) {
    nk_size_t const last = count & ~(nk_size_t)1;
    v128_t const infinity_f64x2 = wasm_f64x2_splat(NUMKONG_F64_INF);
    odd_f64x2[0] = infinity_f64x2, odd_f64x2[1] = infinity_f64x2, odd_f64x2[2] = infinity_f64x2;
    if (last == count) return;
    odd_f64x2[0] = wasm_f64x2_replace_lane(infinity_f64x2, 0, axes[last]);
    odd_f64x2[1] = wasm_f64x2_splat(axes[stride + last]);
    odd_f64x2[2] = wasm_f64x2_splat(axes[2 * stride + last]);
}

/** Hausdorff flips of @c f32 packs in f64 lanes, stopping as the @c f16 pair does. */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f32_v128relaxed_(void const *row_packed, void const *column_packed,
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
    v128_t const limit_f64x2 = wasm_f64x2_splat(limit), limit_squared_f64x2 = wasm_f64x2_splat(limit_squared);
    v128_t const infinity_f64x2 = wasm_f64x2_splat(NUMKONG_F64_INF);
    v128_t farthest_f64x2[4], nearest_f64x2[8], least_f64x2[4], outer_f64x2[3], odd_f64x2[3];
    for (int pair = 0; pair != 4; ++pair) farthest_f64x2[pair] = wasm_f64x2_splat(0);
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_f32_t const *outer_axes = direction ? column_axes : row_axes;
        nk_f32_t const *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        nk_shapes_odd_point_f32_v128relaxed_(inner_axes, inner_stride, inner_count, odd_f64x2);
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            outer_f64x2[0] = wasm_f64x2_splat(outer_axes[outer_index]);
            outer_f64x2[1] = wasm_f64x2_splat(outer_axes[outer_stride + outer_index]);
            outer_f64x2[2] = wasm_f64x2_splat(outer_axes[2 * outer_stride + outer_index]);
            for (int flip = 0; flip != 8; ++flip) nearest_f64x2[flip] = infinity_f64x2;
            nk_shapes_sweep_nearest_f32_v128relaxed_(outer_f64x2, inner_axes, inner_stride, inner_count, odd_f64x2,
                                                     nearest_f64x2);
            nk_shapes_least_flips_f64x2_v128relaxed_(nearest_f64x2, least_f64x2);
            v128_t past_square_u64x2 = wasm_i64x2_splat(-1), past_limit_u64x2 = past_square_u64x2;
            for (int pair = 0; pair != 4; ++pair) {
                farthest_f64x2[pair] = wasm_f64x2_relaxed_max(farthest_f64x2[pair], least_f64x2[pair]);
                past_square_u64x2 = wasm_v128_and(past_square_u64x2,
                                                  wasm_f64x2_gt(farthest_f64x2[pair], limit_squared_f64x2));
            }
            if (!wasm_i64x2_all_true(past_square_u64x2)) continue;
            for (int pair = 0; pair != 4; ++pair)
                past_limit_u64x2 = wasm_v128_and(past_limit_u64x2,
                                                 wasm_f64x2_gt(wasm_f64x2_sqrt(farthest_f64x2[pair]), limit_f64x2));
            rejected = wasm_i64x2_all_true(past_limit_u64x2);
        }
    }
    for (int pair = 0; pair != 4; ++pair) wasm_v128_store(hausdorffs + 2 * pair, wasm_f64x2_sqrt(farthest_f64x2[pair]));
}

/** Squared distances from a broadcast row point, unfused, which also serve single arcs, so that
 *  tree arcs keep a zero reduced cost. */
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

/** The reduced costs (c − πᵢ) + πⱼ of a row's arcs to two columns, with matrix costs −G taken as
 *  πⱼ − (G + πᵢ), which rounds the same. */
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

/** Keeps the first of two reduced costs from @p column that beats the running least. */
NUMKONG_INLINE void nk_shapes_keep_least_f64x2_v128relaxed_(v128_t reduced_f64x2, nk_size_t row, nk_size_t column,
                                                            v128_t *least_f64x2, nk_f64_t *entering_cost,
                                                            nk_size_t *entering_row, nk_size_t *entering_column) {
    if (!wasm_v128_any_true(wasm_f64x2_lt(reduced_f64x2, *least_f64x2))) return;
    nk_f64_t const reduced_low = wasm_f64x2_extract_lane(reduced_f64x2, 0);
    nk_f64_t const reduced_high = wasm_f64x2_extract_lane(reduced_f64x2, 1);
    int const takes_high = reduced_high < reduced_low;
    *entering_cost = takes_high ? reduced_high : reduced_low;
    *entering_row = row, *entering_column = column + takes_high;
    *least_f64x2 = wasm_f64x2_splat(*entering_cost);
}

/** Keeps a tile's first least reduced cost below @p entering_cost, two columns at a time. The lane
 *  past an odd tile reads neighboring workspace and costs infinity. */
NUMKONG_INLINE void nk_shapes_price_tile_f32_v128relaxed_(nk_shapes_workspace_f32_serial_t const *views,
                                                          nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t row, nk_size_t column, nk_size_t tile,
                                                          nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                          nk_size_t *entering_column) {
    v128_t least_f64x2 = wasm_f64x2_splat(*entering_cost);
    nk_size_t const pairs_end = tile & ~(nk_size_t)1;
    for (nk_size_t step = 0; step != pairs_end; step += 2)
        nk_shapes_keep_least_f64x2_v128relaxed_(
            nk_shapes_reduced_costs_f32_v128relaxed_(views, gradient, rows, columns, row, column + step), row,
            column + step, &least_f64x2, entering_cost, entering_row, entering_column);
    if (pairs_end == tile) return;
    v128_t const reduced_f64x2 = nk_shapes_reduced_costs_f32_v128relaxed_(views, gradient, rows, columns, row,
                                                                          column + pairs_end);
    nk_shapes_keep_least_f64x2_v128relaxed_(wasm_f64x2_replace_lane(reduced_f64x2, 1, NUMKONG_F64_INF), row,
                                            column + pairs_end, &least_f64x2, entering_cost, entering_row,
                                            entering_column);
}

NUMKONG_INLINE void nk_shapes_tree_costs_f32_v128relaxed_(nk_shapes_workspace_f32_serial_t const *views,
                                                          nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        views->potential[node] = gradient ? -gradient[row * columns + column]
                                          : (nk_f64_t)nk_shapes_point_cost_f32_v128relaxed_(views, row, column);
    }
}

NUMKONG_INLINE void nk_shapes_network_simplex_f32_v128relaxed_(nk_shapes_workspace_f32_serial_t const *views,
                                                               nk_f64_t const *gradient, nk_size_t rows,
                                                               nk_size_t columns, nk_f64_t cost_scale) {
    nk_shapes_simplex_serial_t simplex;
    nk_shapes_tree_costs_f32_v128relaxed_(views, gradient, rows, columns);
    nk_shapes_tree_potentials_serial_(views->tree, views->tree_stride, views->potential, rows);
    nk_shapes_simplex_start_serial_(&simplex, views->tree, views->tree_stride, views->potential, rows, columns,
                                    cost_scale);
    while (nk_shapes_simplex_next_tile_serial_(&simplex))
        nk_shapes_price_tile_f32_v128relaxed_(views, gradient, rows, columns, simplex.row, simplex.column, simplex.tile,
                                              &simplex.entering_cost, &simplex.entering_row, &simplex.entering_column);
}

NUMKONG_INLINE nk_f32_t nk_shapes_tree_cost_f32_v128relaxed_(nk_shapes_workspace_f32_serial_t const *views,
                                                             nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f32_t cost = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        cost += (nk_f32_t)flow[node] * nk_shapes_point_cost_f32_v128relaxed_(views, row, column);
    }
    return cost / (nk_f32_t)(rows * columns);
}

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

NUMKONG_INLINE void nk_shapes_price_tile_f64_v128relaxed_(nk_shapes_workspace_f64_serial_t const *views,
                                                          nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                          nk_size_t row, nk_size_t column, nk_size_t tile,
                                                          nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                          nk_size_t *entering_column) {
    v128_t least_f64x2 = wasm_f64x2_splat(*entering_cost);
    nk_size_t const pairs_end = tile & ~(nk_size_t)1;
    for (nk_size_t step = 0; step != pairs_end; step += 2)
        nk_shapes_keep_least_f64x2_v128relaxed_(
            nk_shapes_reduced_costs_f64_v128relaxed_(views, gradient, rows, columns, row, column + step), row,
            column + step, &least_f64x2, entering_cost, entering_row, entering_column);
    if (pairs_end == tile) return;
    v128_t const reduced_f64x2 = nk_shapes_reduced_costs_f64_v128relaxed_(views, gradient, rows, columns, row,
                                                                          column + pairs_end);
    nk_shapes_keep_least_f64x2_v128relaxed_(wasm_f64x2_replace_lane(reduced_f64x2, 1, NUMKONG_F64_INF), row,
                                            column + pairs_end, &least_f64x2, entering_cost, entering_row,
                                            entering_column);
}

NUMKONG_INLINE void nk_shapes_tree_costs_f64_v128relaxed_(nk_shapes_workspace_f64_serial_t const *views,
                                                          nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        views->potential[node] = gradient ? -gradient[row * columns + column]
                                          : nk_shapes_point_cost_f64_v128relaxed_(views, row, column);
    }
}

NUMKONG_INLINE void nk_shapes_network_simplex_f64_v128relaxed_(nk_shapes_workspace_f64_serial_t const *views,
                                                               nk_f64_t const *gradient, nk_size_t rows,
                                                               nk_size_t columns, nk_f64_t cost_scale) {
    nk_shapes_simplex_serial_t simplex;
    nk_shapes_tree_costs_f64_v128relaxed_(views, gradient, rows, columns);
    nk_shapes_tree_potentials_serial_(views->tree, views->tree_stride, views->potential, rows);
    nk_shapes_simplex_start_serial_(&simplex, views->tree, views->tree_stride, views->potential, rows, columns,
                                    cost_scale);
    while (nk_shapes_simplex_next_tile_serial_(&simplex))
        nk_shapes_price_tile_f64_v128relaxed_(views, gradient, rows, columns, simplex.row, simplex.column, simplex.tile,
                                              &simplex.entering_cost, &simplex.entering_row, &simplex.entering_column);
}

NUMKONG_INLINE nk_f64_t nk_shapes_tree_cost_f64_v128relaxed_(nk_shapes_workspace_f64_serial_t const *views,
                                                             nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f64_t cost = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        cost += (nk_f64_t)flow[node] * nk_shapes_point_cost_f64_v128relaxed_(views, row, column);
    }
    return cost / (nk_f64_t)(rows * columns);
}

NUMKONG_INLINE nk_f32_t nk_shapes_wasserstein_procrustes_f16_v128relaxed_(nk_shapes_workspace_f32_serial_t const *views,
                                                                          nk_size_t rows, nk_size_t columns,
                                                                          nk_f64_t cost_scale, nk_f32_t *rotation) {
    nk_shapes_procrustes_f32_serial_t search;
    nk_shapes_procrustes_start_f32_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_f32_serial_(&search)) {
        nk_shapes_network_simplex_f32_v128relaxed_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_f32_v128relaxed_(views, rows, columns);
    }
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f16_v128relaxed_(
    nk_shapes_workspace_f32_serial_t const *views, nk_size_t rows, nk_size_t columns, nk_f64_t cost_scale,
    nk_f64_t const *row_distances, nk_f64_t const *column_distances, nk_f32_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f16_v128relaxed_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f32_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_serial_(&search);
    while (nk_shapes_gromov_next_f32_serial_(&search, views, rotation))
        nk_shapes_network_simplex_f32_v128relaxed_(views, search.gradient, rows, columns, search.gradient_scale);
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_wasserstein_procrustes_f32_v128relaxed_(nk_shapes_workspace_f64_serial_t const *views,
                                                                          nk_size_t rows, nk_size_t columns,
                                                                          nk_f64_t cost_scale, nk_f64_t *rotation) {
    nk_shapes_procrustes_f64_serial_t search;
    nk_shapes_procrustes_start_f64_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_f64_serial_(&search)) {
        nk_shapes_network_simplex_f64_v128relaxed_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_f64_v128relaxed_(views, rows, columns);
    }
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f32_v128relaxed_(
    nk_shapes_workspace_f64_serial_t const *views, nk_size_t rows, nk_size_t columns, nk_f64_t cost_scale,
    nk_f64_t const *row_distances, nk_f64_t const *column_distances, nk_f64_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f32_v128relaxed_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f64_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_serial_(&search);
    while (nk_shapes_gromov_next_f64_serial_(&search, views, rotation))
        nk_shapes_network_simplex_f64_v128relaxed_(views, search.gradient, rows, columns, search.gradient_scale);
    return search.best;
}

NUMKONG_INLINE void nk_jacobi_eigenanalysis_f32_v128relaxed_(nk_f32_t *matrix, nk_f32_t *quaternion) {
    nk_jacobi_start_f32_serial_(quaternion);
    for (int sweep = 0; sweep != 16; ++sweep) {
        for (int axis = 0; axis != 3; ++axis) {
            nk_f32_t const norm = nk_jacobi_half_angle_f32_serial_(matrix);
            nk_jacobi_conjugate_f32_serial_(axis, nk_rsqrt_f32_v128relaxed_(norm), matrix, quaternion);
        }
        if (nk_jacobi_converged_f32_serial_(matrix, 0x1p-24f)) break;
    }
    nk_f32_t const length_squared = nk_jacobi_length_squared_f32_serial_(quaternion);
    nk_jacobi_rescale_f32_serial_(nk_rsqrt_f32_v128relaxed_(length_squared), quaternion);
}

NUMKONG_INLINE void nk_jacobi_eigenanalysis_f64_v128relaxed_(nk_f64_t *matrix, nk_f64_t *quaternion) {
    nk_jacobi_start_f64_serial_(quaternion);
    for (int sweep = 0; sweep != 16; ++sweep) {
        for (int axis = 0; axis != 3; ++axis) {
            nk_f64_t const norm = nk_jacobi_half_angle_f64_serial_(matrix);
            nk_jacobi_conjugate_f64_serial_(axis, nk_rsqrt_f64_v128relaxed_(norm), matrix, quaternion);
        }
        if (nk_jacobi_converged_f64_serial_(matrix, 0x1p-53)) break;
    }
    nk_f64_t const length_squared = nk_jacobi_length_squared_f64_serial_(quaternion);
    nk_jacobi_rescale_f64_serial_(nk_rsqrt_f64_v128relaxed_(length_squared), quaternion);
}

/** The next four @c f16 points of the @p left remaining, each axis in f32 lanes, with
 *  @p valid_u32x4 over the lanes holding one. Padded lanes read zero. */
NUMKONG_INLINE void nk_shapes_load_points_f16_v128relaxed_(nk_f16_t const *points, nk_size_t left, v128_t *x_f32x4,
                                                           v128_t *y_f32x4, v128_t *z_f32x4, v128_t *valid_u32x4) {
    nk_u16_t padded[12];
    nk_u16_t const *source = (nk_u16_t const *)points;
    nk_b128_vec_t valid_vec;
    valid_vec.v128 = wasm_i32x4_splat(-1);
    if (left < 4) {
        for (nk_size_t lane = 0; lane != 12; ++lane) padded[lane] = lane < left * 3 ? source[lane] : 0;
        for (nk_size_t lane = left; lane != 4; ++lane) valid_vec.u32s[lane] = 0;
        source = padded;
    }
    v128_t const low_u16x8 = wasm_v128_load(source), high_u16x8 = wasm_v128_load64_zero(source + 8);
    nk_b64_vec_t x_vec, y_vec, z_vec;
    x_vec.u64 = wasm_i64x2_extract_lane(wasm_i16x8_shuffle(low_u16x8, high_u16x8, 0, 3, 6, 9, 0, 0, 0, 0), 0);
    y_vec.u64 = wasm_i64x2_extract_lane(wasm_i16x8_shuffle(low_u16x8, high_u16x8, 1, 4, 7, 10, 0, 0, 0, 0), 0);
    z_vec.u64 = wasm_i64x2_extract_lane(wasm_i16x8_shuffle(low_u16x8, high_u16x8, 2, 5, 8, 11, 0, 0, 0, 0), 0);
    *x_f32x4 = nk_f16x4_to_f32x4_v128relaxed_(x_vec).v128;
    *y_f32x4 = nk_f16x4_to_f32x4_v128relaxed_(y_vec).v128;
    *z_f32x4 = nk_f16x4_to_f32x4_v128relaxed_(z_vec).v128;
    *valid_u32x4 = valid_vec.v128;
}

/** The moments of a cloud of @c f16 points as @ref nk_shapes_moments_f16_serial_ defines them, four
 *  points per step, unfused, as engines fuse relaxed multiply-adds differently. */
NUMKONG_INLINE nk_f32_t nk_shapes_moments_f16_v128relaxed_(nk_f16_t const *points, nk_size_t count, nk_f32_t *centroid,
                                                           nk_f32_t *covariance) {
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    v128_t x_f32x4, y_f32x4, z_f32x4, valid_u32x4;
    nk_shapes_load_points_f16_v128relaxed_(points, 1, &x_f32x4, &y_f32x4, &z_f32x4, &valid_u32x4);
    v128_t const first_x_f32x4 = wasm_i32x4_shuffle(x_f32x4, x_f32x4, 0, 0, 0, 0);
    v128_t const first_y_f32x4 = wasm_i32x4_shuffle(y_f32x4, y_f32x4, 0, 0, 0, 0);
    v128_t const first_z_f32x4 = wasm_i32x4_shuffle(z_f32x4, z_f32x4, 0, 0, 0, 0);
    v128_t sum_x_f32x4 = wasm_f32x4_splat(0), sum_y_f32x4 = sum_x_f32x4, sum_z_f32x4 = sum_x_f32x4;
    v128_t coincident_u32x4 = wasm_i32x4_splat(-1);
    for (nk_size_t index = 0; index < count; index += 4) {
        nk_shapes_load_points_f16_v128relaxed_(points + index * 3, count - index, &x_f32x4, &y_f32x4, &z_f32x4,
                                               &valid_u32x4);
        sum_x_f32x4 = wasm_f32x4_add(sum_x_f32x4, x_f32x4), sum_y_f32x4 = wasm_f32x4_add(sum_y_f32x4, y_f32x4);
        sum_z_f32x4 = wasm_f32x4_add(sum_z_f32x4, z_f32x4);
        v128_t const same_u32x4 = wasm_v128_and(
            wasm_v128_and(wasm_f32x4_eq(x_f32x4, first_x_f32x4), wasm_f32x4_eq(y_f32x4, first_y_f32x4)),
            wasm_f32x4_eq(z_f32x4, first_z_f32x4));
        coincident_u32x4 = wasm_v128_and(coincident_u32x4, wasm_v128_or(same_u32x4, wasm_v128_not(valid_u32x4)));
    }
    nk_f32_t const points_count = (nk_f32_t)count;
    centroid[0] = nk_reduce_add_f32x4_v128_(sum_x_f32x4) / points_count;
    centroid[1] = nk_reduce_add_f32x4_v128_(sum_y_f32x4) / points_count;
    centroid[2] = nk_reduce_add_f32x4_v128_(sum_z_f32x4) / points_count;
    v128_t const centroid_x_f32x4 = wasm_f32x4_splat(centroid[0]), centroid_y_f32x4 = wasm_f32x4_splat(centroid[1]);
    v128_t const centroid_z_f32x4 = wasm_f32x4_splat(centroid[2]);
    v128_t moments_f32x4[6];
    for (int entry = 0; entry != 6; ++entry) moments_f32x4[entry] = wasm_f32x4_splat(0);
    for (nk_size_t index = 0; index < count; index += 4) {
        nk_shapes_load_points_f16_v128relaxed_(points + index * 3, count - index, &x_f32x4, &y_f32x4, &z_f32x4,
                                               &valid_u32x4);
        x_f32x4 = wasm_v128_and(wasm_f32x4_sub(x_f32x4, centroid_x_f32x4), valid_u32x4);
        y_f32x4 = wasm_v128_and(wasm_f32x4_sub(y_f32x4, centroid_y_f32x4), valid_u32x4);
        z_f32x4 = wasm_v128_and(wasm_f32x4_sub(z_f32x4, centroid_z_f32x4), valid_u32x4);
        moments_f32x4[0] = wasm_f32x4_add(moments_f32x4[0], wasm_f32x4_mul(x_f32x4, x_f32x4));
        moments_f32x4[1] = wasm_f32x4_add(moments_f32x4[1], wasm_f32x4_mul(y_f32x4, x_f32x4));
        moments_f32x4[2] = wasm_f32x4_add(moments_f32x4[2], wasm_f32x4_mul(y_f32x4, y_f32x4));
        moments_f32x4[3] = wasm_f32x4_add(moments_f32x4[3], wasm_f32x4_mul(z_f32x4, x_f32x4));
        moments_f32x4[4] = wasm_f32x4_add(moments_f32x4[4], wasm_f32x4_mul(z_f32x4, y_f32x4));
        moments_f32x4[5] = wasm_f32x4_add(moments_f32x4[5], wasm_f32x4_mul(z_f32x4, z_f32x4));
    }
    for (int entry = 0; entry != 6; ++entry)
        covariance[entry] = nk_reduce_add_f32x4_v128_(moments_f32x4[entry]) / points_count;
    return wasm_i32x4_all_true(coincident_u32x4)
               ? 0
               : nk_sqrt_f32_v128relaxed_(covariance[0] + covariance[2] + covariance[5]);
}

/** The pack of one cloud from its moments, as @c nk_shapes_pack_f16_serial_ lays it out, storing
 *  each step's points and measuring what was stored in one sweep, with zeros in the padding. */
NUMKONG_INLINE void nk_shapes_pack_f16_v128relaxed_(nk_f16_t const *points, nk_size_t count, nk_f32_t const *centroid,
                                                    nk_f32_t const *covariance, nk_f32_t scale, void *packed) {
    nk_shapes_packed_header_f32_t *header = (nk_shapes_packed_header_f32_t *)packed;
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f32_t),
                                                                 nk_shapes_alignment_k_);
    nk_size_t const axis_bytes = nk_size_round_up_to_multiple_(count * sizeof(nk_f16_t), nk_shapes_alignment_k_);
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
    nk_jacobi_eigenanalysis_f32_v128relaxed_(spreads, quaternion);
    nk_shapes_sorted_basis_f32_serial_(spreads, quaternion, 0x1p-7f, basis, &upper_tied, &lower_tied);
    if (upper_tied && lower_tied) nk_shapes_pick_axis_f16_serial_(points, count, centroid, basis);
    else if (upper_tied || lower_tied) nk_shapes_turn_plane_f16_serial_(points, count, centroid, basis);
    nk_shapes_frame_f32_serial_(basis, upper_tied, header->frame, axes);
    for (int cell = 0; cell != 9; ++cell) axes[cell] /= scale;
    v128_t axes_f32x4[9];
    for (int cell = 0; cell != 9; ++cell) axes_f32x4[cell] = wasm_f32x4_splat(axes[cell]);
    v128_t const centroid_x_f32x4 = wasm_f32x4_splat(centroid[0]), centroid_y_f32x4 = wasm_f32x4_splat(centroid[1]);
    v128_t const centroid_z_f32x4 = wasm_f32x4_splat(centroid[2]);
    v128_t moments_f32x4[6], largest_f32x4 = wasm_f32x4_splat(0);
    for (int entry = 0; entry != 6; ++entry) moments_f32x4[entry] = wasm_f32x4_splat(0);
    v128_t x_f32x4, y_f32x4, z_f32x4, valid_u32x4;
    nk_b128_vec_t rotated_vec[3];
    nk_b64_vec_t stored_vec[3];
    for (nk_size_t index = 0; index < count; index += 4) {
        nk_shapes_load_points_f16_v128relaxed_(points + index * 3, count - index, &x_f32x4, &y_f32x4, &z_f32x4,
                                               &valid_u32x4);
        x_f32x4 = wasm_f32x4_sub(x_f32x4, centroid_x_f32x4), y_f32x4 = wasm_f32x4_sub(y_f32x4, centroid_y_f32x4);
        z_f32x4 = wasm_f32x4_sub(z_f32x4, centroid_z_f32x4);
        for (int axis = 0; axis != 3; ++axis) {
            v128_t const rotated_f32x4 = wasm_f32x4_add(
                wasm_f32x4_add(wasm_f32x4_mul(x_f32x4, axes_f32x4[axis * 3]),
                               wasm_f32x4_mul(y_f32x4, axes_f32x4[axis * 3 + 1])),
                wasm_f32x4_mul(z_f32x4, axes_f32x4[axis * 3 + 2]));
            rotated_vec[axis].v128 = wasm_v128_and(rotated_f32x4, valid_u32x4);
            stored_vec[axis] = nk_f32x4_to_f16x4_v128relaxed_(rotated_vec[axis]);
        }
        nk_store_b64_serial_(&stored_vec[0], packed_x + index), nk_store_b64_serial_(&stored_vec[1], packed_y + index);
        nk_store_b64_serial_(&stored_vec[2], packed_z + index);
        // The spreads describe the stored points, so the sweep measures them after rounding
        x_f32x4 = nk_f16x4_to_f32x4_v128relaxed_(stored_vec[0]).v128;
        y_f32x4 = nk_f16x4_to_f32x4_v128relaxed_(stored_vec[1]).v128;
        z_f32x4 = nk_f16x4_to_f32x4_v128relaxed_(stored_vec[2]).v128;
        v128_t const xx_f32x4 = wasm_f32x4_mul(x_f32x4, x_f32x4), yy_f32x4 = wasm_f32x4_mul(y_f32x4, y_f32x4);
        v128_t const zz_f32x4 = wasm_f32x4_mul(z_f32x4, z_f32x4);
        largest_f32x4 = wasm_f32x4_relaxed_max(largest_f32x4,
                                               wasm_f32x4_add(wasm_f32x4_add(xx_f32x4, yy_f32x4), zz_f32x4));
        moments_f32x4[0] = wasm_f32x4_add(moments_f32x4[0], xx_f32x4);
        moments_f32x4[1] = wasm_f32x4_add(moments_f32x4[1], wasm_f32x4_mul(y_f32x4, x_f32x4));
        moments_f32x4[2] = wasm_f32x4_add(moments_f32x4[2], yy_f32x4);
        moments_f32x4[3] = wasm_f32x4_add(moments_f32x4[3], wasm_f32x4_mul(z_f32x4, x_f32x4));
        moments_f32x4[4] = wasm_f32x4_add(moments_f32x4[4], wasm_f32x4_mul(z_f32x4, y_f32x4));
        moments_f32x4[5] = wasm_f32x4_add(moments_f32x4[5], zz_f32x4);
    }
    for (nk_size_t index = (count + 3) & ~(nk_size_t)3; index < axis_stride; ++index)
        packed_x[index] = 0, packed_y[index] = 0, packed_z[index] = 0;
    nk_f32_t moments[6];
    for (int entry = 0; entry != 6; ++entry)
        moments[entry] = nk_reduce_add_f32x4_v128_(moments_f32x4[entry]) / (nk_f32_t)count;
    v128_t const largest_high_f32x4 = wasm_f32x4_relaxed_max(
        largest_f32x4, wasm_i32x4_shuffle(largest_f32x4, largest_f32x4, 2, 3, 0, 1));
    nk_f32_t const largest = wasm_f32x4_extract_lane(
        wasm_f32x4_relaxed_max(largest_high_f32x4,
                               wasm_i32x4_shuffle(largest_high_f32x4, largest_high_f32x4, 1, 0, 3, 2)),
        0);
    nk_jacobi_eigenanalysis_f32_v128relaxed_(moments, quaternion);
    nk_shapes_close_pack_f32_serial_(header, count, centroid, moments, nk_sqrt_f32_v128relaxed_(largest));
}

/** The next two @c f32 points of the @p left remaining, each axis in f64 lanes, with @p valid_u64x2
 *  over the lanes holding one. Padded lanes read zero. */
NUMKONG_INLINE void nk_shapes_load_points_f32_v128relaxed_(nk_f32_t const *points, nk_size_t left, v128_t *x_f64x2,
                                                           v128_t *y_f64x2, v128_t *z_f64x2, v128_t *valid_u64x2) {
    nk_f32_t padded[6];
    nk_f32_t const *source = points;
    nk_b128_vec_t valid_vec;
    valid_vec.v128 = wasm_i32x4_splat(-1);
    if (left < 2) {
        padded[0] = points[0], padded[1] = points[1], padded[2] = points[2];
        padded[3] = 0, padded[4] = 0, padded[5] = 0;
        valid_vec.u64s[1] = 0, source = padded;
    }
    v128_t const low_f32x4 = wasm_v128_load(source), high_f32x4 = wasm_v128_load64_zero(source + 4);
    *x_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(low_f32x4, high_f32x4, 0, 3, 0, 0));
    *y_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(low_f32x4, high_f32x4, 1, 4, 0, 0));
    *z_f64x2 = wasm_f64x2_promote_low_f32x4(wasm_i32x4_shuffle(low_f32x4, high_f32x4, 2, 5, 0, 0));
    *valid_u64x2 = valid_vec.v128;
}

/** @copydoc nk_shapes_moments_f16_v128relaxed_ */
NUMKONG_INLINE nk_f64_t nk_shapes_moments_f32_v128relaxed_(nk_f32_t const *points, nk_size_t count, nk_f64_t *centroid,
                                                           nk_f64_t *covariance) {
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    v128_t x_f64x2, y_f64x2, z_f64x2, valid_u64x2;
    v128_t const first_x_f64x2 = wasm_f64x2_splat(points[0]), first_y_f64x2 = wasm_f64x2_splat(points[1]);
    v128_t const first_z_f64x2 = wasm_f64x2_splat(points[2]);
    v128_t sum_x_f64x2 = wasm_f64x2_splat(0), sum_y_f64x2 = sum_x_f64x2, sum_z_f64x2 = sum_x_f64x2;
    v128_t coincident_u64x2 = wasm_i64x2_splat(-1);
    for (nk_size_t index = 0; index < count; index += 2) {
        nk_shapes_load_points_f32_v128relaxed_(points + index * 3, count - index, &x_f64x2, &y_f64x2, &z_f64x2,
                                               &valid_u64x2);
        sum_x_f64x2 = wasm_f64x2_add(sum_x_f64x2, x_f64x2), sum_y_f64x2 = wasm_f64x2_add(sum_y_f64x2, y_f64x2);
        sum_z_f64x2 = wasm_f64x2_add(sum_z_f64x2, z_f64x2);
        v128_t const same_u64x2 = wasm_v128_and(
            wasm_v128_and(wasm_f64x2_eq(x_f64x2, first_x_f64x2), wasm_f64x2_eq(y_f64x2, first_y_f64x2)),
            wasm_f64x2_eq(z_f64x2, first_z_f64x2));
        coincident_u64x2 = wasm_v128_and(coincident_u64x2, wasm_v128_or(same_u64x2, wasm_v128_not(valid_u64x2)));
    }
    nk_f64_t const points_count = (nk_f64_t)count;
    centroid[0] = nk_reduce_add_f64x2_v128_(sum_x_f64x2) / points_count;
    centroid[1] = nk_reduce_add_f64x2_v128_(sum_y_f64x2) / points_count;
    centroid[2] = nk_reduce_add_f64x2_v128_(sum_z_f64x2) / points_count;
    v128_t const centroid_x_f64x2 = wasm_f64x2_splat(centroid[0]), centroid_y_f64x2 = wasm_f64x2_splat(centroid[1]);
    v128_t const centroid_z_f64x2 = wasm_f64x2_splat(centroid[2]);
    v128_t moments_f64x2[6];
    for (int entry = 0; entry != 6; ++entry) moments_f64x2[entry] = wasm_f64x2_splat(0);
    for (nk_size_t index = 0; index < count; index += 2) {
        nk_shapes_load_points_f32_v128relaxed_(points + index * 3, count - index, &x_f64x2, &y_f64x2, &z_f64x2,
                                               &valid_u64x2);
        x_f64x2 = wasm_v128_and(wasm_f64x2_sub(x_f64x2, centroid_x_f64x2), valid_u64x2);
        y_f64x2 = wasm_v128_and(wasm_f64x2_sub(y_f64x2, centroid_y_f64x2), valid_u64x2);
        z_f64x2 = wasm_v128_and(wasm_f64x2_sub(z_f64x2, centroid_z_f64x2), valid_u64x2);
        moments_f64x2[0] = wasm_f64x2_add(moments_f64x2[0], wasm_f64x2_mul(x_f64x2, x_f64x2));
        moments_f64x2[1] = wasm_f64x2_add(moments_f64x2[1], wasm_f64x2_mul(y_f64x2, x_f64x2));
        moments_f64x2[2] = wasm_f64x2_add(moments_f64x2[2], wasm_f64x2_mul(y_f64x2, y_f64x2));
        moments_f64x2[3] = wasm_f64x2_add(moments_f64x2[3], wasm_f64x2_mul(z_f64x2, x_f64x2));
        moments_f64x2[4] = wasm_f64x2_add(moments_f64x2[4], wasm_f64x2_mul(z_f64x2, y_f64x2));
        moments_f64x2[5] = wasm_f64x2_add(moments_f64x2[5], wasm_f64x2_mul(z_f64x2, z_f64x2));
    }
    for (int entry = 0; entry != 6; ++entry)
        covariance[entry] = nk_reduce_add_f64x2_v128_(moments_f64x2[entry]) / points_count;
    return wasm_i64x2_all_true(coincident_u64x2)
               ? 0
               : nk_sqrt_f64_v128relaxed_(covariance[0] + covariance[2] + covariance[5]);
}

/** @copydoc nk_shapes_pack_f16_v128relaxed_ */
NUMKONG_INLINE void nk_shapes_pack_f32_v128relaxed_(nk_f32_t const *points, nk_size_t count, nk_f64_t const *centroid,
                                                    nk_f64_t const *covariance, nk_f64_t scale, void *packed) {
    nk_shapes_packed_header_f64_t *header = (nk_shapes_packed_header_f64_t *)packed;
    nk_size_t const header_bytes = nk_size_round_up_to_multiple_(sizeof(nk_shapes_packed_header_f64_t),
                                                                 nk_shapes_alignment_k_);
    nk_size_t const axis_bytes = nk_size_round_up_to_multiple_(count * sizeof(nk_f32_t), nk_shapes_alignment_k_);
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
    nk_jacobi_eigenanalysis_f64_v128relaxed_(spreads, quaternion);
    nk_shapes_sorted_basis_f64_serial_(spreads, quaternion, 0x1p-16, basis, &upper_tied, &lower_tied);
    if (upper_tied && lower_tied) nk_shapes_pick_axis_f32_serial_(points, count, centroid, basis);
    else if (upper_tied || lower_tied) nk_shapes_turn_plane_f32_serial_(points, count, centroid, basis);
    nk_shapes_frame_f64_serial_(basis, upper_tied, header->frame, axes);
    for (int cell = 0; cell != 9; ++cell) axes[cell] /= scale;
    v128_t axes_f64x2[9];
    for (int cell = 0; cell != 9; ++cell) axes_f64x2[cell] = wasm_f64x2_splat(axes[cell]);
    v128_t const centroid_x_f64x2 = wasm_f64x2_splat(centroid[0]), centroid_y_f64x2 = wasm_f64x2_splat(centroid[1]);
    v128_t const centroid_z_f64x2 = wasm_f64x2_splat(centroid[2]);
    v128_t moments_f64x2[6], largest_f64x2 = wasm_f64x2_splat(0);
    for (int entry = 0; entry != 6; ++entry) moments_f64x2[entry] = wasm_f64x2_splat(0);
    v128_t x_f64x2, y_f64x2, z_f64x2, valid_u64x2, stored_f32x4[3];
    for (nk_size_t index = 0; index < count; index += 2) {
        nk_shapes_load_points_f32_v128relaxed_(points + index * 3, count - index, &x_f64x2, &y_f64x2, &z_f64x2,
                                               &valid_u64x2);
        x_f64x2 = wasm_f64x2_sub(x_f64x2, centroid_x_f64x2), y_f64x2 = wasm_f64x2_sub(y_f64x2, centroid_y_f64x2);
        z_f64x2 = wasm_f64x2_sub(z_f64x2, centroid_z_f64x2);
        for (int axis = 0; axis != 3; ++axis) {
            v128_t const rotated_f64x2 = wasm_f64x2_add(
                wasm_f64x2_add(wasm_f64x2_mul(x_f64x2, axes_f64x2[axis * 3]),
                               wasm_f64x2_mul(y_f64x2, axes_f64x2[axis * 3 + 1])),
                wasm_f64x2_mul(z_f64x2, axes_f64x2[axis * 3 + 2]));
            stored_f32x4[axis] = wasm_f32x4_demote_f64x2_zero(wasm_v128_and(rotated_f64x2, valid_u64x2));
        }
        wasm_v128_store64_lane(packed_x + index, stored_f32x4[0], 0);
        wasm_v128_store64_lane(packed_y + index, stored_f32x4[1], 0);
        wasm_v128_store64_lane(packed_z + index, stored_f32x4[2], 0);
        // The spreads describe the stored points, so the sweep measures them after rounding
        x_f64x2 = wasm_f64x2_promote_low_f32x4(stored_f32x4[0]);
        y_f64x2 = wasm_f64x2_promote_low_f32x4(stored_f32x4[1]);
        z_f64x2 = wasm_f64x2_promote_low_f32x4(stored_f32x4[2]);
        v128_t const xx_f64x2 = wasm_f64x2_mul(x_f64x2, x_f64x2), yy_f64x2 = wasm_f64x2_mul(y_f64x2, y_f64x2);
        v128_t const zz_f64x2 = wasm_f64x2_mul(z_f64x2, z_f64x2);
        largest_f64x2 = wasm_f64x2_relaxed_max(largest_f64x2,
                                               wasm_f64x2_add(wasm_f64x2_add(xx_f64x2, yy_f64x2), zz_f64x2));
        moments_f64x2[0] = wasm_f64x2_add(moments_f64x2[0], xx_f64x2);
        moments_f64x2[1] = wasm_f64x2_add(moments_f64x2[1], wasm_f64x2_mul(y_f64x2, x_f64x2));
        moments_f64x2[2] = wasm_f64x2_add(moments_f64x2[2], yy_f64x2);
        moments_f64x2[3] = wasm_f64x2_add(moments_f64x2[3], wasm_f64x2_mul(z_f64x2, x_f64x2));
        moments_f64x2[4] = wasm_f64x2_add(moments_f64x2[4], wasm_f64x2_mul(z_f64x2, y_f64x2));
        moments_f64x2[5] = wasm_f64x2_add(moments_f64x2[5], zz_f64x2);
    }
    for (nk_size_t index = (count + 1) & ~(nk_size_t)1; index < axis_stride; ++index)
        packed_x[index] = 0, packed_y[index] = 0, packed_z[index] = 0;
    nk_f64_t moments[6];
    for (int entry = 0; entry != 6; ++entry)
        moments[entry] = nk_reduce_add_f64x2_v128_(moments_f64x2[entry]) / (nk_f64_t)count;
    nk_f64_t const largest_low = wasm_f64x2_extract_lane(largest_f64x2, 0);
    nk_f64_t const largest_high = wasm_f64x2_extract_lane(largest_f64x2, 1);
    nk_jacobi_eigenanalysis_f64_v128relaxed_(moments, quaternion);
    nk_shapes_close_pack_f64_serial_(header, count, centroid, moments,
                                     nk_sqrt_f64_v128relaxed_(largest_low > largest_high ? largest_low : largest_high));
}

NUMKONG_API nk_status_t nk_shapes_pack_f16_v128relaxed(nk_f16_t const *points, nk_size_t count, void *packed,
                                                       nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t centroid[3], covariance[6];
    nk_shapes_moments_f16_v128relaxed_(points, count, centroid, covariance);
    nk_shapes_pack_f16_v128relaxed_(points, count, centroid, covariance, 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f16_v128relaxed(nk_f16_t const *points, nk_size_t count, void *packed,
                                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t centroid[3], covariance[6];
    nk_f32_t const radius = nk_shapes_moments_f16_v128relaxed_(points, count, centroid, covariance);
    nk_shapes_pack_f16_v128relaxed_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_f32_v128relaxed(nk_f32_t const *points, nk_size_t count, void *packed,
                                                       nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t centroid[3], covariance[6];
    nk_shapes_moments_f32_v128relaxed_(points, count, centroid, covariance);
    nk_shapes_pack_f32_v128relaxed_(points, count, centroid, covariance, 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f32_v128relaxed(nk_f32_t const *points, nk_size_t count, void *packed,
                                                                  nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t centroid[3], covariance[6];
    nk_f64_t const radius = nk_shapes_moments_f32_v128relaxed_(points, count, centroid, covariance);
    nk_shapes_pack_f32_v128relaxed_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f16_v128relaxed(void const *const *a_packed, nk_size_t rows,
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
            nk_status_t const status = nk_shapes_chamfer_flips_f16_v128relaxed_(
                a_packed[row], b_packed[column], pair_limit, workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f32_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_v128relaxed(void const *const *a_packed, nk_size_t rows,
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
            nk_shapes_hausdorff_flips_f16_v128relaxed_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f32_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f32_v128relaxed(void const *const *a_packed, nk_size_t rows,
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
            nk_status_t const status = nk_shapes_chamfer_flips_f32_v128relaxed_(
                a_packed[row], b_packed[column], pair_limit, workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f64_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_v128relaxed(void const *const *a_packed, nk_size_t rows,
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
            nk_shapes_hausdorff_flips_f32_v128relaxed_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f64_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_v128relaxed(void const *const *a_packed, nk_size_t rows,
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
                results_row[column] = nk_sqrt_f32_serial_(nk_shapes_wasserstein_procrustes_f16_v128relaxed_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F32_INF;
            if (rotations) nk_shapes_pair_rotation_f32_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_v128relaxed(void const *const *a_packed, nk_size_t rows,
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
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f16_v128relaxed_(
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

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_v128relaxed(void const *const *a_packed, nk_size_t rows,
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
                results_row[column] = nk_sqrt_f64_serial_(nk_shapes_wasserstein_procrustes_f32_v128relaxed_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F64_INF;
            if (rotations) nk_shapes_pair_rotation_f64_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_v128relaxed(void const *const *a_packed, nk_size_t rows,
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
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f32_v128relaxed_(
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

/** Four @c f16 coordinates of an axis in f32 lanes. */
NUMKONG_INLINE v128_t nk_rmsds_load_f16x4_v128relaxed_(nk_f16_t const *axis) {
    nk_b64_vec_t bits_vec;
    nk_load_b64_serial_(axis, &bits_vec);
    return nk_f16x4_to_f32x4_v128relaxed_(bits_vec).v128;
}

/** Σ‖xᵢ′‖² of a pack of @c f16 points, 4 points per step through the zero padding. */
NUMKONG_INLINE nk_f32_t nk_rmsds_norm_f16_v128relaxed_(void const *packed) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)packed)->points;
    nk_size_t axis_stride;
    nk_f16_t const *axes = nk_shapes_packed_axes_f16_serial_(packed, &axis_stride);
    v128_t norm_f32x4 = wasm_f32x4_splat(0);
    for (int axis = 0; axis != 3; ++axis)
        for (nk_size_t index = 0; index < count; index += 4) {
            v128_t const values_f32x4 = nk_rmsds_load_f16x4_v128relaxed_(axes + axis * axis_stride + index);
            norm_f32x4 = wasm_f32x4_relaxed_madd(values_f32x4, values_f32x4, norm_f32x4);
        }
    return nk_reduce_add_f32x4_v128_(norm_f32x4);
}

/** Superposed RMSD of one pair of @c f16 packs with Ga + Gb in @p norms_squared, M in @c f32. */
NUMKONG_INLINE nk_f32_t nk_rmsds_pair_f16_v128relaxed_(void const *row_packed, void const *column_packed,
                                                       nk_f32_t norms_squared) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f16_t const *a_axes = nk_shapes_packed_axes_f16_serial_(row_packed, &axis_stride);
    nk_f16_t const *b_axes = nk_shapes_packed_axes_f16_serial_(column_packed, &axis_stride);
    v128_t sums_f32x4[9], a_f32x4[3], b_f32x4[3];
    for (int cell = 0; cell != 9; ++cell) sums_f32x4[cell] = wasm_f32x4_splat(0);
    for (nk_size_t index = 0; index < count; index += 4) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f32x4[axis] = nk_rmsds_load_f16x4_v128relaxed_(a_axes + axis * axis_stride + index);
            b_f32x4[axis] = nk_rmsds_load_f16x4_v128relaxed_(b_axes + axis * axis_stride + index);
        }
        for (int row = 0; row != 3; ++row)
            for (int column = 0; column != 3; ++column)
                sums_f32x4[row * 3 + column] = wasm_f32x4_relaxed_madd(a_f32x4[row], b_f32x4[column],
                                                                       sums_f32x4[row * 3 + column]);
    }
    nk_f32_t cross_covariance[9];
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = nk_reduce_add_f32x4_v128_(sums_f32x4[cell]);
    nk_f32_t const residual = nk_rmsds_folded_f32_serial_(cross_covariance, norms_squared);
    return nk_sqrt_f32_v128relaxed_(residual / (nk_f32_t)count);
}

/** Superposed RMSD of one pair of @c f32 packs: H in @c f64, then the residuals in plain @c f64,
 *  2 points per step through the zero padding. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f32_v128relaxed_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f32_t const *a_axes = nk_shapes_packed_axes_f32_serial_(row_packed, &axis_stride);
    nk_f32_t const *b_axes = nk_shapes_packed_axes_f32_serial_(column_packed, &axis_stride);
    v128_t sums_f64x2[9], a_f64x2[3], b_f64x2[3];
    for (int cell = 0; cell != 9; ++cell) sums_f64x2[cell] = wasm_f64x2_splat(0);
    for (nk_size_t index = 0; index < count; index += 2) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f64x2[axis] = wasm_f64x2_promote_low_f32x4(wasm_v128_load64_zero(a_axes + axis * axis_stride + index));
            b_f64x2[axis] = wasm_f64x2_promote_low_f32x4(wasm_v128_load64_zero(b_axes + axis * axis_stride + index));
        }
        for (int column = 0; column != 3; ++column)
            for (int row = 0; row != 3; ++row)
                sums_f64x2[column * 3 + row] = wasm_f64x2_relaxed_madd(b_f64x2[column], a_f64x2[row],
                                                                       sums_f64x2[column * 3 + row]);
    }
    nk_f64_t cross_covariance[9], rotation[9];
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = nk_reduce_add_f64x2_v128_(sums_f64x2[cell]);
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    v128_t rotation_f64x2[9];
    for (int cell = 0; cell != 9; ++cell) rotation_f64x2[cell] = wasm_f64x2_splat(rotation[cell]);
    v128_t sum_f64x2 = wasm_f64x2_splat(0);
    for (nk_size_t index = 0; index < count; index += 2) {
        for (int axis = 0; axis != 3; ++axis)
            b_f64x2[axis] = wasm_f64x2_promote_low_f32x4(wasm_v128_load64_zero(b_axes + axis * axis_stride + index));
        for (int axis = 0; axis != 3; ++axis) {
            v128_t residual_f64x2 = wasm_f64x2_promote_low_f32x4(
                wasm_v128_load64_zero(a_axes + axis * axis_stride + index));
            for (int column = 0; column != 3; ++column)
                residual_f64x2 = wasm_f64x2_relaxed_nmadd(rotation_f64x2[axis * 3 + column], b_f64x2[column],
                                                          residual_f64x2);
            sum_f64x2 = wasm_f64x2_relaxed_madd(residual_f64x2, residual_f64x2, sum_f64x2);
        }
    }
    return nk_sqrt_f64_v128relaxed_(nk_reduce_add_f64x2_v128_(sum_f64x2) / (nk_f64_t)count);
}

/** Superposed RMSD of one pair of @c f64 packs, like the @c f32 pair with H and the residuals
 *  compensated, 2 points per step through the zero padding. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f64_v128relaxed_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f64_t const *a_axes = nk_shapes_packed_axes_f64_serial_(row_packed, &axis_stride);
    nk_f64_t const *b_axes = nk_shapes_packed_axes_f64_serial_(column_packed, &axis_stride);
    v128_t sums_f64x2[9], compensations_f64x2[9], a_f64x2[3], b_f64x2[3];
    for (int cell = 0; cell != 9; ++cell)
        sums_f64x2[cell] = wasm_f64x2_splat(0), compensations_f64x2[cell] = wasm_f64x2_splat(0);
    for (nk_size_t index = 0; index < count; index += 2) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f64x2[axis] = wasm_v128_load(a_axes + axis * axis_stride + index);
            b_f64x2[axis] = wasm_v128_load(b_axes + axis * axis_stride + index);
        }
        for (int column = 0; column != 3; ++column)
            for (int row = 0; row != 3; ++row)
                nk_dot2_f64x2_v128_(&sums_f64x2[column * 3 + row], &compensations_f64x2[column * 3 + row],
                                    b_f64x2[column], a_f64x2[row]);
    }
    nk_f64_t cross_covariance[9], rotation[9];
    for (int cell = 0; cell != 9; ++cell)
        cross_covariance[cell] = nk_dot_stable_sum_f64x2_v128_(sums_f64x2[cell], compensations_f64x2[cell]);
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    v128_t rotation_f64x2[9];
    for (int cell = 0; cell != 9; ++cell) rotation_f64x2[cell] = wasm_f64x2_splat(rotation[cell]);
    v128_t sum_f64x2 = wasm_f64x2_splat(0), compensation_f64x2 = wasm_f64x2_splat(0);
    for (nk_size_t index = 0; index < count; index += 2) {
        for (int axis = 0; axis != 3; ++axis) b_f64x2[axis] = wasm_v128_load(b_axes + axis * axis_stride + index);
        for (int axis = 0; axis != 3; ++axis) {
            v128_t residual_f64x2 = wasm_v128_load(a_axes + axis * axis_stride + index);
            for (int column = 0; column != 3; ++column)
                residual_f64x2 = wasm_f64x2_relaxed_nmadd(rotation_f64x2[axis * 3 + column], b_f64x2[column],
                                                          residual_f64x2);
            nk_dot2_f64x2_v128_(&sum_f64x2, &compensation_f64x2, residual_f64x2, residual_f64x2);
        }
    }
    return nk_sqrt_f64_v128relaxed_(nk_dot_stable_sum_f64x2_v128_(sum_f64x2, compensation_f64x2) / (nk_f64_t)count);
}

NUMKONG_API nk_status_t nk_rmsds_packed_f64_v128relaxed(void const *const *a_packed, nk_size_t rows,
                                                        void const *const *b_packed, nk_size_t columns,
                                                        nk_f64_t *results, nk_size_t results_stride,
                                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_size_t const points = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed[0])->points;
    if (!nk_rmsds_same_points_f64_serial_(a_packed, rows, points) ||
        !nk_rmsds_same_points_f64_serial_(b_packed, columns, points))
        return nk_unexpected_dimensions_k;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column)
            results_row[column] = nk_rmsds_pair_f64_v128relaxed_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f32_v128relaxed(void const *const *a_packed, nk_size_t rows,
                                                        void const *const *b_packed, nk_size_t columns,
                                                        nk_f64_t *results, nk_size_t results_stride,
                                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_size_t const points = (nk_size_t)((nk_shapes_packed_header_f64_t const *)a_packed[0])->points;
    if (!nk_rmsds_same_points_f32_serial_(a_packed, rows, points) ||
        !nk_rmsds_same_points_f32_serial_(b_packed, columns, points))
        return nk_unexpected_dimensions_k;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column)
            results_row[column] = nk_rmsds_pair_f32_v128relaxed_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f16_v128relaxed(void const *const *a_packed, nk_size_t rows,
                                                        void const *const *b_packed, nk_size_t columns,
                                                        nk_f32_t *results, nk_size_t results_stride,
                                                        nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    if (!rows || !columns) return nk_success_k;
    nk_size_t const points = ((nk_shapes_packed_header_f32_t const *)a_packed[0])->points;
    if (!nk_rmsds_same_points_f16_serial_(a_packed, rows, points) ||
        !nk_rmsds_same_points_f16_serial_(b_packed, columns, points))
        return nk_unexpected_dimensions_k;
    nk_f32_t row_norms[nk_rmsds_tile_frames_k_], column_norms[nk_rmsds_tile_frames_k_];
    for (nk_size_t row_begin = 0; row_begin < rows; row_begin += nk_rmsds_tile_frames_k_) {
        nk_size_t const row_end = row_begin + nk_rmsds_tile_frames_k_ < rows ? row_begin + nk_rmsds_tile_frames_k_
                                                                             : rows;
        for (nk_size_t row = row_begin; row != row_end; ++row)
            row_norms[row - row_begin] = nk_rmsds_norm_f16_v128relaxed_(a_packed[row]);
        for (nk_size_t column_begin = 0; column_begin < columns; column_begin += nk_rmsds_tile_frames_k_) {
            nk_size_t const column_end = column_begin + nk_rmsds_tile_frames_k_ < columns
                                             ? column_begin + nk_rmsds_tile_frames_k_
                                             : columns;
            for (nk_size_t column = column_begin; column != column_end; ++column)
                column_norms[column - column_begin] = nk_rmsds_norm_f16_v128relaxed_(b_packed[column]);
            for (nk_size_t row = row_begin; row != row_end; ++row) {
                nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
                for (nk_size_t column = column_begin; column != column_end; ++column)
                    results_row[column] = nk_rmsds_pair_f16_v128relaxed_(
                        a_packed[row], b_packed[column],
                        row_norms[row - row_begin] + column_norms[column - column_begin]);
            }
        }
    }
    return nk_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_TARGET_V128RELAXED
#endif // NUMKONG_ARCH_WASM_
#endif // NUMKONG_SHAPES_V128RELAXED_H
