/**
 *  @file include/numkong/shapes/neon.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief Many-to-many shape distances between 3D point clouds for NEON.
 *
 *  @sa include/numkong/shapes.h
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
 *  never win a minimum.
 *
 *  The transport kinds run the serial solvers around block pricing two arcs at a time. Arc costs
 *  round in fused multiply-adds, and the scalar cost behind the tree's potentials runs the same
 *  lanes, so tree arcs keep a zero reduced cost.
 */
#ifndef NUMKONG_SHAPES_NEON_H
#define NUMKONG_SHAPES_NEON_H

#if NUMKONG_ARCH_ARM64_
#if NUMKONG_ARCH_ARM64_NEON_

#include "numkong/types.h"
#include "numkong/cast/neon.h"     // `nk_load_f16x4_as_f32_neon_`
#include "numkong/shapes/serial.h" // `nk_define_shapes_network_simplex_`, `nk_shapes_workspace_f64_serial_t`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8-a+simd")
#endif

/** Folds one outer point's minima of the 8 flips, 4 lanes each, into the sums and maxima of their
 *  roots, flips 0 to 3 in the first vector and 4 to 7 in the second. */
NUMKONG_INLINE void nk_shapes_fold_nearest_f32_neon_(float32x4_t const *nearest_f32x4, float32x4_t *sums_f32x4,
                                                     float32x4_t *maxima_f32x4) {
    float32x4_t const distances_f32x4[2] = {vsqrtq_f32(vpminq_f32(vpminq_f32(nearest_f32x4[0], nearest_f32x4[1]),
                                                                  vpminq_f32(nearest_f32x4[2], nearest_f32x4[3]))),
                                            vsqrtq_f32(vpminq_f32(vpminq_f32(nearest_f32x4[4], nearest_f32x4[5]),
                                                                  vpminq_f32(nearest_f32x4[6], nearest_f32x4[7])))};
    for (int half = 0; half != 2; ++half)
        sums_f32x4[half] = vaddq_f32(sums_f32x4[half], distances_f32x4[half]),
        maxima_f32x4[half] = vmaxq_f32(maxima_f32x4[half], distances_f32x4[half]);
}

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

/** One directed Chamfer and Hausdorff pass over f32 packs in f64 lanes, two inner points at a time,
 *  the last pair of an odd count with an infinite x in its padded lane. */
NUMKONG_INLINE void nk_shapes_nearest_f32_neon_(nk_f32_t const *outer_axes, nk_size_t outer_stride,
                                                nk_size_t outer_count, nk_f32_t const *inner_axes,
                                                nk_size_t inner_stride, nk_size_t inner_count, nk_f64_t *sums,
                                                nk_f64_t *maxima) {
    float64x2_t const infinity_f64x2 = vdupq_n_f64(NUMKONG_F64_INF);
    nk_b128_vec_t first_lane_vec;
    first_lane_vec.u64s[0] = ~(nk_u64_t)0, first_lane_vec.u64s[1] = 0;
    nk_size_t const pairs_end = inner_count & ~(nk_size_t)1;
    float64x2_t sums_f64x2[4], maxima_f64x2[4];
    for (int pair = 0; pair != 4; ++pair)
        sums_f64x2[pair] = vld1q_f64(sums + 2 * pair), maxima_f64x2[pair] = vld1q_f64(maxima + 2 * pair);
    for (nk_size_t outer_index = 0; outer_index != outer_count; ++outer_index) {
        float64x2_t const outer_x_f64x2 = vdupq_n_f64(outer_axes[outer_index]);
        float64x2_t const outer_y_f64x2 = vdupq_n_f64(outer_axes[outer_stride + outer_index]);
        float64x2_t const outer_z_f64x2 = vdupq_n_f64(outer_axes[2 * outer_stride + outer_index]);
        float64x2_t nearest_f64x2[8];
        for (int flip = 0; flip != 8; ++flip) nearest_f64x2[flip] = infinity_f64x2;
        for (nk_size_t inner_index = 0; inner_index != pairs_end; inner_index += 2)
            nk_shapes_lower_nearest_f64_neon_(
                outer_x_f64x2, outer_y_f64x2, outer_z_f64x2, vcvt_f64_f32(vld1_f32(inner_axes + inner_index)),
                vcvt_f64_f32(vld1_f32(inner_axes + inner_stride + inner_index)),
                vcvt_f64_f32(vld1_f32(inner_axes + 2 * inner_stride + inner_index)), nearest_f64x2);
        if (pairs_end != inner_count)
            nk_shapes_lower_nearest_f64_neon_(
                outer_x_f64x2, outer_y_f64x2, outer_z_f64x2,
                vbslq_f64(first_lane_vec.u64x2, vcvt_f64_f32(vld1_f32(inner_axes + pairs_end)), infinity_f64x2),
                vcvt_f64_f32(vld1_f32(inner_axes + inner_stride + pairs_end)),
                vcvt_f64_f32(vld1_f32(inner_axes + 2 * inner_stride + pairs_end)), nearest_f64x2);
        for (int pair = 0; pair != 4; ++pair) {
            float64x2_t const distance_f64x2 = vsqrtq_f64(
                vpminq_f64(nearest_f64x2[2 * pair], nearest_f64x2[2 * pair + 1]));
            sums_f64x2[pair] = vaddq_f64(sums_f64x2[pair], distance_f64x2);
            maxima_f64x2[pair] = vmaxq_f64(maxima_f64x2[pair], distance_f64x2);
        }
    }
    for (int pair = 0; pair != 4; ++pair)
        vst1q_f64(sums + 2 * pair, sums_f64x2[pair]), vst1q_f64(maxima + 2 * pair, maxima_f64x2[pair]);
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

/** One directed Chamfer and Hausdorff pass over f16 packs in f32 lanes, four inner points at a
 *  time, the last quadruple of a count off the multiple with an infinite x in its padded lanes. */
NUMKONG_INLINE void nk_shapes_nearest_f16_neon_(nk_f16_t const *outer_axes, nk_size_t outer_stride,
                                                nk_size_t outer_count, nk_f16_t const *inner_axes,
                                                nk_size_t inner_stride, nk_size_t inner_count, nk_f32_t *sums,
                                                nk_f32_t *maxima) {
    nk_u16_t const *outer_bits = (nk_u16_t const *)outer_axes;
    float32x4_t const infinity_f32x4 = vdupq_n_f32(NUMKONG_F32_INF);
    nk_size_t const quadruples_end = inner_count & ~(nk_size_t)3;
    nk_b128_vec_t tail_vec;
    for (nk_size_t lane = 0; lane != 4; ++lane) tail_vec.u32s[lane] = quadruples_end + lane < inner_count ? ~0u : 0;
    float32x4_t sums_f32x4[2] = {vld1q_f32(sums), vld1q_f32(sums + 4)};
    float32x4_t maxima_f32x4[2] = {vld1q_f32(maxima), vld1q_f32(maxima + 4)};
    for (nk_size_t outer_index = 0; outer_index != outer_count; ++outer_index) {
        float32x4_t const outer_x_f32x4 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_dup_u16(outer_bits + outer_index)));
        float32x4_t const outer_y_f32x4 = vcvt_f32_f16(
            vreinterpret_f16_u16(vld1_dup_u16(outer_bits + outer_stride + outer_index)));
        float32x4_t const outer_z_f32x4 = vcvt_f32_f16(
            vreinterpret_f16_u16(vld1_dup_u16(outer_bits + 2 * outer_stride + outer_index)));
        float32x4_t nearest_f32x4[8];
        for (int flip = 0; flip != 8; ++flip) nearest_f32x4[flip] = infinity_f32x4;
        for (nk_size_t inner_index = 0; inner_index != quadruples_end; inner_index += 4)
            nk_shapes_lower_nearest_f32_neon_(
                outer_x_f32x4, outer_y_f32x4, outer_z_f32x4, nk_load_f16x4_as_f32_neon_(inner_axes + inner_index),
                nk_load_f16x4_as_f32_neon_(inner_axes + inner_stride + inner_index),
                nk_load_f16x4_as_f32_neon_(inner_axes + 2 * inner_stride + inner_index), nearest_f32x4);
        if (quadruples_end != inner_count)
            nk_shapes_lower_nearest_f32_neon_(
                outer_x_f32x4, outer_y_f32x4, outer_z_f32x4,
                vbslq_f32(tail_vec.u32x4, nk_load_f16x4_as_f32_neon_(inner_axes + quadruples_end), infinity_f32x4),
                nk_load_f16x4_as_f32_neon_(inner_axes + inner_stride + quadruples_end),
                nk_load_f16x4_as_f32_neon_(inner_axes + 2 * inner_stride + quadruples_end), nearest_f32x4);
        nk_shapes_fold_nearest_f32_neon_(nearest_f32x4, sums_f32x4, maxima_f32x4);
    }
    vst1q_f32(sums, sums_f32x4[0]), vst1q_f32(sums + 4, sums_f32x4[1]);
    vst1q_f32(maxima, maxima_f32x4[0]), vst1q_f32(maxima + 4, maxima_f32x4[1]);
}

/** The block pricing of one math type over @p vector_type lanes of two columns: squared distances
 *  from a broadcast row point in fused multiply-adds; the scalar cost behind the potentials, run on
 *  the same lanes so tree arcs keep a zero reduced cost; the reduced costs c(i, j) − πᵢ + πⱼ of a
 *  row's arcs to two columns, with matrix costs −G taken as πⱼ − (G + πᵢ), which rounds the same;
 *  and the scan of @p count of them from the cursor in row-major order, keeping the first most
 *  negative one below @p entering_cost as the serial scan does. A first pass takes their least two
 *  columns at a time, and only a block whose least beats @p entering_cost is scanned again for the
 *  first arc reaching it. Lanes past the scanned part of a row read neighbouring workspace and take
 *  an infinite cost. @p widen turns the costs into f64 lanes, and stays empty when they are. */
#define nk_define_shapes_pricing_neon_(math_type, vector_type, vector_suffix, load_broadcast, load, subtract,          \
                                       multiply, multiply_add, first_lane, widen)                                      \
    NUMKONG_INLINE vector_type nk_shapes_point_costs_##vector_suffix##_neon_(                                          \
        vector_type row_x_##vector_suffix, vector_type row_y_##vector_suffix, vector_type row_z_##vector_suffix,       \
        vector_type column_x_##vector_suffix, vector_type column_y_##vector_suffix,                                    \
        vector_type column_z_##vector_suffix) {                                                                        \
        vector_type const delta_x_##vector_suffix = subtract(row_x_##vector_suffix, column_x_##vector_suffix);         \
        vector_type const delta_y_##vector_suffix = subtract(row_y_##vector_suffix, column_y_##vector_suffix);         \
        vector_type const delta_z_##vector_suffix = subtract(row_z_##vector_suffix, column_z_##vector_suffix);         \
        return multiply_add(multiply_add(multiply(delta_x_##vector_suffix, delta_x_##vector_suffix),                   \
                                         delta_y_##vector_suffix, delta_y_##vector_suffix),                            \
                            delta_z_##vector_suffix, delta_z_##vector_suffix);                                         \
    }                                                                                                                  \
    NUMKONG_INLINE nk_##math_type##_t nk_shapes_point_cost_##math_type##_neon_(                                        \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_size_t row, nk_size_t column) {                    \
        nk_size_t const stride = views->points_stride;                                                                 \
        nk_##math_type##_t const *row_points = views->row_points, *rotated_points = views->rotated_points;             \
        return first_lane(nk_shapes_point_costs_##vector_suffix##_neon_(                                               \
                              load_broadcast(row_points + row), load_broadcast(row_points + stride + row),             \
                              load_broadcast(row_points + 2 * stride + row), load_broadcast(rotated_points + column),  \
                              load_broadcast(rotated_points + stride + column),                                        \
                              load_broadcast(rotated_points + 2 * stride + column)),                                   \
                          0);                                                                                          \
    }                                                                                                                  \
    NUMKONG_INLINE float64x2_t nk_shapes_reduced_costs_##math_type##_neon_(                                            \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_f64_t const *gradient, nk_size_t rows,             \
        nk_size_t columns, nk_size_t row, nk_size_t column) {                                                          \
        nk_size_t const stride = views->points_stride;                                                                 \
        nk_##math_type##_t const *row_points = views->row_points, *rotated_points = views->rotated_points;             \
        float64x2_t const row_potential_f64x2 = vld1q_dup_f64(views->potential + row);                                 \
        float64x2_t const column_potentials_f64x2 = vld1q_f64(views->potential + rows + column);                       \
        if (gradient)                                                                                                  \
            return vsubq_f64(column_potentials_f64x2,                                                                  \
                             vaddq_f64(vld1q_f64(gradient + row * columns + column), row_potential_f64x2));            \
        float64x2_t const costs_f64x2 = widen(nk_shapes_point_costs_##vector_suffix##_neon_(                           \
            load_broadcast(row_points + row), load_broadcast(row_points + stride + row),                               \
            load_broadcast(row_points + 2 * stride + row), load(rotated_points + column),                              \
            load(rotated_points + stride + column), load(rotated_points + 2 * stride + column)));                      \
        return vaddq_f64(vsubq_f64(costs_f64x2, row_potential_f64x2), column_potentials_f64x2);                        \
    }                                                                                                                  \
    NUMKONG_INLINE void nk_shapes_price_block_##math_type##_neon_(                                                     \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_f64_t const *gradient, nk_size_t rows,             \
        nk_size_t columns, nk_size_t count, nk_size_t *cursor_row, nk_size_t *cursor_column, nk_f64_t *entering_cost,  \
        nk_size_t *entering_row, nk_size_t *entering_column) {                                                         \
        float64x2_t const infinity_f64x2 = vdupq_n_f64(NUMKONG_F64_INF);                                               \
        uint64x2_t const first_lane_u64x2 = vcombine_u64(vcreate_u64(~(nk_u64_t)0), vcreate_u64(0));                   \
        nk_size_t const start_row = *cursor_row, start_column = *cursor_column;                                        \
        nk_size_t row = start_row, column = start_column;                                                              \
        float64x2_t least_f64x2 = infinity_f64x2;                                                                      \
        for (nk_size_t offset = 0; offset != count;) {                                                                 \
            nk_size_t const segment = columns - column < count - offset ? columns - column : count - offset;           \
            nk_size_t const pairs_end = segment & ~(nk_size_t)1;                                                       \
            for (nk_size_t step = 0; step != pairs_end; step += 2)                                                     \
                least_f64x2 = vminq_f64(least_f64x2, nk_shapes_reduced_costs_##math_type##_neon_(                      \
                                                         views, gradient, rows, columns, row, column + step));         \
            if (pairs_end != segment)                                                                                  \
                least_f64x2 = vminq_f64(                                                                               \
                    least_f64x2, vbslq_f64(first_lane_u64x2,                                                           \
                                           nk_shapes_reduced_costs_##math_type##_neon_(views, gradient, rows, columns, \
                                                                                       row, column + pairs_end),       \
                                           infinity_f64x2));                                                           \
            offset += segment, column += segment;                                                                      \
            if (column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;                                    \
        }                                                                                                              \
        *cursor_row = row, *cursor_column = column;                                                                    \
        nk_f64_t const least = vminvq_f64(least_f64x2);                                                                \
        if (!(least < *entering_cost)) return;                                                                         \
        float64x2_t const least_broadcast_f64x2 = vdupq_n_f64(least);                                                  \
        *entering_cost = least, row = start_row, column = start_column;                                                \
        for (;;) {                                                                                                     \
            nk_size_t const segment = columns - column;                                                                \
            for (nk_size_t step = 0; step < segment; step += 2) {                                                      \
                float64x2_t reduced_f64x2 = nk_shapes_reduced_costs_##math_type##_neon_(views, gradient, rows,         \
                                                                                        columns, row, column + step);  \
                if (step + 2 > segment) reduced_f64x2 = vbslq_f64(first_lane_u64x2, reduced_f64x2, infinity_f64x2);    \
                uint64x2_t const equal_u64x2 = vceqq_f64(reduced_f64x2, least_broadcast_f64x2);                        \
                if (!vmaxvq_u32(vreinterpretq_u32_u64(equal_u64x2))) continue;                                         \
                *entering_row = row, *entering_column = column + step + !vgetq_lane_u64(equal_u64x2, 0);               \
                return;                                                                                                \
            }                                                                                                          \
            column = 0, row = row + 1 == rows ? 0 : row + 1;                                                           \
        }                                                                                                              \
    }

nk_define_shapes_pricing_neon_(f64, float64x2_t, f64x2, vld1q_dup_f64, vld1q_f64, vsubq_f64, vmulq_f64, vfmaq_f64,
                               vgetq_lane_f64, )
nk_define_shapes_pricing_neon_(f32, float32x2_t, f32x2, vld1_dup_f32, vld1_f32, vsub_f32, vmul_f32, vfma_f32,
                               vget_lane_f32, vcvt_f64_f32)
#undef nk_define_shapes_pricing_neon_

nk_define_shapes_network_simplex_(f32, neon, nk_shapes_point_cost_f32_neon_, nk_shapes_price_block_f32_neon_)
nk_define_shapes_network_simplex_(f64, neon, nk_shapes_point_cost_f64_neon_, nk_shapes_price_block_f64_neon_)
nk_define_shapes_wasserstein_procrustes_(f32, neon, NUMKONG_F32_INF)
nk_define_shapes_wasserstein_procrustes_(f64, neon, NUMKONG_F64_INF)
nk_define_shapes_gromov_wasserstein_(f32, neon)
nk_define_shapes_gromov_wasserstein_(f64, neon)
nk_define_shapes_flip_kernels_(f16, f32, neon, nk_shapes_nearest_f16_neon_, NUMKONG_F32_INF)
nk_define_shapes_flip_kernels_(f32, f64, neon, nk_shapes_nearest_f32_neon_, NUMKONG_F64_INF)
nk_define_shapes_transport_kernels_(f16, f32, neon, NUMKONG_F32_INF)
nk_define_shapes_transport_kernels_(f32, f64, neon, NUMKONG_F64_INF)

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
