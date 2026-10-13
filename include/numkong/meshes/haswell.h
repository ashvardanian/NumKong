/**
 *  @file include/numkong/meshes/haswell.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief Many-to-many shape distances between 3D point clouds for Haswell, AVX2.
 *
 *  @sa include/numkong/meshes.h
 *
 *  @section shapes_haswell_instructions Key AVX2 Shapes Instructions
 *
 *  @verbatim
 *  Intrinsic               Instruction                     Haswell     Genoa
 *  _mm256_fmadd_ps/pd      VFMADD (YMM, YMM, YMM)          5cy @ p01   4cy @ p01
 *  _mm256_sub_ps/pd        VSUBPS/PD (YMM, YMM, YMM)       3cy @ p1    3cy @ p23
 *  _mm256_min_ps/pd        VMINPS/PD (YMM, YMM, YMM)       3cy @ p1    2cy @ p01
 *  _mm256_cvtph_ps         VCVTPH2PS (YMM, XMM)            6cy @ p1+p5 4cy @ p12+p23
 *  _mm256_cvtps_pd         VCVTPS2PD (YMM, XMM)            5cy @ p0+p5 4cy @ p12+p23
 *  _mm256_blendv_ps        VBLENDVPS (YMM, YMM, YMM, YMM)  2cy @ p015  1cy @ p01
 *  _mm256_shuffle_ps       VSHUFPS (YMM, YMM, YMM, I8)     1cy @ p5    1cy @ p12
 *  _mm256_permute2f128_ps  VPERM2F128 (YMM, YMM, YMM, I8)  3cy @ p5    2cy @ p12
 *  _mm256_sqrt_ps          VSQRTPS (YMM, YMM)              19cy @ p0   15cy @ p01
 *  _mm256_sqrt_pd          VSQRTPD (YMM, YMM)              35cy @ p0   21cy @ p01
 *  @endverbatim
 *
 *  Chamfer and Hausdorff broadcast one outer point against a register of inner points, which sit
 *  contiguous per axis in the pack, and keep one running minimum per axis flip. Haswell issues
 *  VADDPS and VMINPS on port 1 alone, so the squares fold into FMAs that also take port 0.
 *
 *  The transport kinds run the serial solvers around vector passes: block pricing four arcs at a
 *  time, and the f64 Gromov-Wasserstein gradient products, plan steps and scores. Arc costs round
 *  in fused multiply-adds, and the scalar cost behind the tree's potentials runs the same lanes, so
 *  tree arcs keep a zero reduced cost.
 */
#ifndef NUMKONG_MESHES_HASWELL_H
#define NUMKONG_MESHES_HASWELL_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_HASWELL_

#include "numkong/types.h"
#include "numkong/meshes/serial.h"  // `nk_define_shapes_network_simplex_`, `nk_shapes_workspace_f64_serial_t`
#include "numkong/reduce/haswell.h" // `nk_reduce_min_f64x4_haswell_`, `nk_reduce_add_f64x4_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

/** Minima of neighboring lanes of @p first and @p second, laid out as VHADDPS lays out sums. */
NUMKONG_INLINE __m256 nk_shapes_neighbor_minima_f32x8_haswell_(__m256 first_f32x8, __m256 second_f32x8) {
    return _mm256_min_ps(_mm256_shuffle_ps(first_f32x8, second_f32x8, 0x88),
                         _mm256_shuffle_ps(first_f32x8, second_f32x8, 0xDD));
}

/** Folds 8 per-flip running minima into one vector holding each flip's minimum in its own lane. */
NUMKONG_INLINE __m256 nk_shapes_flip_minima_f32x8_haswell_(__m256 const *minima_f32x8) {
    __m256 const quads_0123_f32x8 = nk_shapes_neighbor_minima_f32x8_haswell_(
        nk_shapes_neighbor_minima_f32x8_haswell_(minima_f32x8[0], minima_f32x8[1]),
        nk_shapes_neighbor_minima_f32x8_haswell_(minima_f32x8[2], minima_f32x8[3]));
    __m256 const quads_4567_f32x8 = nk_shapes_neighbor_minima_f32x8_haswell_(
        nk_shapes_neighbor_minima_f32x8_haswell_(minima_f32x8[4], minima_f32x8[5]),
        nk_shapes_neighbor_minima_f32x8_haswell_(minima_f32x8[6], minima_f32x8[7]));
    // Each 128-bit half holds 4 flips over its own half of the inner lanes
    return _mm256_min_ps(_mm256_permute2f128_ps(quads_0123_f32x8, quads_4567_f32x8, 0x20),
                         _mm256_permute2f128_ps(quads_0123_f32x8, quads_4567_f32x8, 0x31));
}

/** Folds 4 per-flip running minima into one vector holding each flip's minimum in its own lane. */
NUMKONG_INLINE __m256d nk_shapes_flip_minima_f64x4_haswell_(__m256d const *minima_f64x4) {
    __m256d const pairs_01_f64x4 = _mm256_min_pd(_mm256_unpacklo_pd(minima_f64x4[0], minima_f64x4[1]),
                                                 _mm256_unpackhi_pd(minima_f64x4[0], minima_f64x4[1]));
    __m256d const pairs_23_f64x4 = _mm256_min_pd(_mm256_unpacklo_pd(minima_f64x4[2], minima_f64x4[3]),
                                                 _mm256_unpackhi_pd(minima_f64x4[2], minima_f64x4[3]));
    return _mm256_min_pd(_mm256_permute2f128_pd(pairs_01_f64x4, pairs_23_f64x4, 0x20),
                         _mm256_permute2f128_pd(pairs_01_f64x4, pairs_23_f64x4, 0x31));
}

/** One directed Chamfer and Hausdorff pass between f16 packs, 8 inner points per step. */
NUMKONG_INLINE void nk_shapes_nearest_f16_haswell_(                            //
    nk_f16_t const *outer_axes, nk_size_t outer_stride, nk_size_t outer_count, //
    nk_f16_t const *inner_axes, nk_size_t inner_stride, nk_size_t inner_count, //
    nk_f32_t *sums, nk_f32_t *maxima) {
    __m256 const infinity_f32x8 = _mm256_set1_ps(NUMKONG_F32_INF);
    __m256i const lane_indices_i32x8 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    __m256 sums_f32x8 = _mm256_loadu_ps(sums), maxima_f32x8 = _mm256_loadu_ps(maxima);
    for (nk_size_t outer_index = 0; outer_index != outer_count; ++outer_index) {
        nk_f16_t const *outer_point = outer_axes + outer_index;
        __m256 const outer_x_f32x8 = _mm256_cvtph_ps(_mm_set1_epi16(*(nk_i16_t const *)outer_point));
        __m256 const outer_y_f32x8 = _mm256_cvtph_ps(_mm_set1_epi16(*(nk_i16_t const *)(outer_point + outer_stride)));
        __m256 const outer_z_f32x8 = _mm256_cvtph_ps(
            _mm_set1_epi16(*(nk_i16_t const *)(outer_point + 2 * outer_stride)));
        __m256 minima_f32x8[8];
        for (int flip = 0; flip != 8; ++flip) minima_f32x8[flip] = infinity_f32x8;
        for (nk_size_t inner_index = 0; inner_index < inner_count; inner_index += 8) {
            nk_f16_t const *inner_points = inner_axes + inner_index;
            nk_size_t const remaining = inner_count - inner_index;
            // An infinite x in the padding makes every flip's distance there infinite
            __m256i const remaining_i32x8 = _mm256_set1_epi32(remaining < 8 ? (int)remaining : 8);
            __m256 const valid_f32x8 = _mm256_castsi256_ps(_mm256_cmpgt_epi32(remaining_i32x8, lane_indices_i32x8));
            __m256 const inner_x_f32x8 = _mm256_blendv_ps(
                infinity_f32x8, _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)inner_points)), valid_f32x8);
            __m256 const inner_y_f32x8 = _mm256_cvtph_ps(
                _mm_loadu_si128((__m128i const *)(inner_points + inner_stride)));
            __m256 const inner_z_f32x8 = _mm256_cvtph_ps(
                _mm_loadu_si128((__m128i const *)(inner_points + 2 * inner_stride)));
            __m256 const minus_x_f32x8 = _mm256_sub_ps(outer_x_f32x8, inner_x_f32x8);
            __m256 const plus_x_f32x8 = _mm256_add_ps(outer_x_f32x8, inner_x_f32x8);
            __m256 const minus_y_f32x8 = _mm256_sub_ps(outer_y_f32x8, inner_y_f32x8);
            __m256 const plus_y_f32x8 = _mm256_add_ps(outer_y_f32x8, inner_y_f32x8);
            __m256 const minus_z_f32x8 = _mm256_sub_ps(outer_z_f32x8, inner_z_f32x8);
            __m256 const plus_z_f32x8 = _mm256_add_ps(outer_z_f32x8, inner_z_f32x8);
            __m256 const minus_xx_f32x8 = _mm256_mul_ps(minus_x_f32x8, minus_x_f32x8);
            __m256 const plus_xx_f32x8 = _mm256_mul_ps(plus_x_f32x8, plus_x_f32x8);
            __m256 const planar_f32x8[4] = {_mm256_fmadd_ps(minus_y_f32x8, minus_y_f32x8, minus_xx_f32x8),
                                            _mm256_fmadd_ps(minus_y_f32x8, minus_y_f32x8, plus_xx_f32x8),
                                            _mm256_fmadd_ps(plus_y_f32x8, plus_y_f32x8, minus_xx_f32x8),
                                            _mm256_fmadd_ps(plus_y_f32x8, plus_y_f32x8, plus_xx_f32x8)};
            for (int flip = 0; flip != 4; ++flip) {
                minima_f32x8[flip] = _mm256_min_ps(minima_f32x8[flip],
                                                   _mm256_fmadd_ps(minus_z_f32x8, minus_z_f32x8, planar_f32x8[flip]));
                minima_f32x8[flip + 4] = _mm256_min_ps(minima_f32x8[flip + 4],
                                                       _mm256_fmadd_ps(plus_z_f32x8, plus_z_f32x8, planar_f32x8[flip]));
            }
        }
        __m256 const distances_f32x8 = _mm256_sqrt_ps(nk_shapes_flip_minima_f32x8_haswell_(minima_f32x8));
        sums_f32x8 = _mm256_add_ps(sums_f32x8, distances_f32x8);
        maxima_f32x8 = _mm256_max_ps(maxima_f32x8, distances_f32x8);
    }
    _mm256_storeu_ps(sums, sums_f32x8);
    _mm256_storeu_ps(maxima, maxima_f32x8);
}

/** One directed Chamfer and Hausdorff pass between f32 packs, 4 inner points per step in f64. */
NUMKONG_INLINE void nk_shapes_nearest_f32_haswell_(                            //
    nk_f32_t const *outer_axes, nk_size_t outer_stride, nk_size_t outer_count, //
    nk_f32_t const *inner_axes, nk_size_t inner_stride, nk_size_t inner_count, //
    nk_f64_t *sums, nk_f64_t *maxima) {
    __m128 const infinity_f32x4 = _mm_set1_ps(NUMKONG_F32_INF);
    __m256d const infinity_f64x4 = _mm256_set1_pd(NUMKONG_F64_INF);
    __m128i const lane_indices_i32x4 = _mm_setr_epi32(0, 1, 2, 3);
    __m256d sums_f64x4[2] = {_mm256_loadu_pd(sums), _mm256_loadu_pd(sums + 4)};
    __m256d maxima_f64x4[2] = {_mm256_loadu_pd(maxima), _mm256_loadu_pd(maxima + 4)};
    for (nk_size_t outer_index = 0; outer_index != outer_count; ++outer_index) {
        nk_f32_t const *outer_point = outer_axes + outer_index;
        __m256d const outer_x_f64x4 = _mm256_set1_pd(outer_point[0]);
        __m256d const outer_y_f64x4 = _mm256_set1_pd(outer_point[outer_stride]);
        __m256d const outer_z_f64x4 = _mm256_set1_pd(outer_point[2 * outer_stride]);
        __m256d minima_f64x4[8];
        for (int flip = 0; flip != 8; ++flip) minima_f64x4[flip] = infinity_f64x4;
        for (nk_size_t inner_index = 0; inner_index < inner_count; inner_index += 4) {
            nk_f32_t const *inner_points = inner_axes + inner_index;
            nk_size_t const remaining = inner_count - inner_index;
            // An infinite x in the padding makes every flip's distance there infinite
            __m128i const remaining_i32x4 = _mm_set1_epi32(remaining < 4 ? (int)remaining : 4);
            __m128 const valid_f32x4 = _mm_castsi128_ps(_mm_cmpgt_epi32(remaining_i32x4, lane_indices_i32x4));
            __m256d const inner_x_f64x4 = _mm256_cvtps_pd(
                _mm_blendv_ps(infinity_f32x4, _mm_loadu_ps(inner_points), valid_f32x4));
            __m256d const inner_y_f64x4 = _mm256_cvtps_pd(_mm_loadu_ps(inner_points + inner_stride));
            __m256d const inner_z_f64x4 = _mm256_cvtps_pd(_mm_loadu_ps(inner_points + 2 * inner_stride));
            __m256d const minus_x_f64x4 = _mm256_sub_pd(outer_x_f64x4, inner_x_f64x4);
            __m256d const plus_x_f64x4 = _mm256_add_pd(outer_x_f64x4, inner_x_f64x4);
            __m256d const minus_y_f64x4 = _mm256_sub_pd(outer_y_f64x4, inner_y_f64x4);
            __m256d const plus_y_f64x4 = _mm256_add_pd(outer_y_f64x4, inner_y_f64x4);
            __m256d const minus_z_f64x4 = _mm256_sub_pd(outer_z_f64x4, inner_z_f64x4);
            __m256d const plus_z_f64x4 = _mm256_add_pd(outer_z_f64x4, inner_z_f64x4);
            __m256d const minus_xx_f64x4 = _mm256_mul_pd(minus_x_f64x4, minus_x_f64x4);
            __m256d const plus_xx_f64x4 = _mm256_mul_pd(plus_x_f64x4, plus_x_f64x4);
            __m256d const planar_f64x4[4] = {_mm256_fmadd_pd(minus_y_f64x4, minus_y_f64x4, minus_xx_f64x4),
                                             _mm256_fmadd_pd(minus_y_f64x4, minus_y_f64x4, plus_xx_f64x4),
                                             _mm256_fmadd_pd(plus_y_f64x4, plus_y_f64x4, minus_xx_f64x4),
                                             _mm256_fmadd_pd(plus_y_f64x4, plus_y_f64x4, plus_xx_f64x4)};
            for (int flip = 0; flip != 4; ++flip) {
                minima_f64x4[flip] = _mm256_min_pd(minima_f64x4[flip],
                                                   _mm256_fmadd_pd(minus_z_f64x4, minus_z_f64x4, planar_f64x4[flip]));
                minima_f64x4[flip + 4] = _mm256_min_pd(minima_f64x4[flip + 4],
                                                       _mm256_fmadd_pd(plus_z_f64x4, plus_z_f64x4, planar_f64x4[flip]));
            }
        }
        for (int half = 0; half != 2; ++half) {
            __m256d const distances_f64x4 = _mm256_sqrt_pd(
                nk_shapes_flip_minima_f64x4_haswell_(minima_f64x4 + 4 * half));
            sums_f64x4[half] = _mm256_add_pd(sums_f64x4[half], distances_f64x4);
            maxima_f64x4[half] = _mm256_max_pd(maxima_f64x4[half], distances_f64x4);
        }
    }
    _mm256_storeu_pd(sums, sums_f64x4[0]), _mm256_storeu_pd(sums + 4, sums_f64x4[1]);
    _mm256_storeu_pd(maxima, maxima_f64x4[0]), _mm256_storeu_pd(maxima + 4, maxima_f64x4[1]);
}

/** All bits set in the first @p remaining of 4 lanes, to blend or clear the lanes past a row. */
NUMKONG_INLINE __m256d nk_shapes_valid_lanes_f64x4_haswell_(nk_size_t remaining) {
    __m256i const remaining_i64x4 = _mm256_set1_epi64x(remaining < 4 ? (long long)remaining : 4);
    return _mm256_castsi256_pd(_mm256_cmpgt_epi64(remaining_i64x4, _mm256_setr_epi64x(0, 1, 2, 3)));
}

/** Squared distances from a broadcast row point to 4 column points in fused multiply-adds, the
 *  rounding both the block pricing and the scalar cost behind the potentials share. */
NUMKONG_INLINE __m256d nk_shapes_point_costs_f64x4_haswell_(__m256d row_x_f64x4, __m256d row_y_f64x4,
                                                            __m256d row_z_f64x4, __m256d column_x_f64x4,
                                                            __m256d column_y_f64x4, __m256d column_z_f64x4) {
    __m256d const delta_x_f64x4 = _mm256_sub_pd(row_x_f64x4, column_x_f64x4);
    __m256d const delta_y_f64x4 = _mm256_sub_pd(row_y_f64x4, column_y_f64x4);
    __m256d const delta_z_f64x4 = _mm256_sub_pd(row_z_f64x4, column_z_f64x4);
    return _mm256_fmadd_pd(delta_z_f64x4, delta_z_f64x4,
                           _mm256_fmadd_pd(delta_y_f64x4, delta_y_f64x4, _mm256_mul_pd(delta_x_f64x4, delta_x_f64x4)));
}

/** @copydoc nk_shapes_point_costs_f64x4_haswell_ */
NUMKONG_INLINE __m128 nk_shapes_point_costs_f32x4_haswell_(__m128 row_x_f32x4, __m128 row_y_f32x4, __m128 row_z_f32x4,
                                                           __m128 column_x_f32x4, __m128 column_y_f32x4,
                                                           __m128 column_z_f32x4) {
    __m128 const delta_x_f32x4 = _mm_sub_ps(row_x_f32x4, column_x_f32x4);
    __m128 const delta_y_f32x4 = _mm_sub_ps(row_y_f32x4, column_y_f32x4);
    __m128 const delta_z_f32x4 = _mm_sub_ps(row_z_f32x4, column_z_f32x4);
    return _mm_fmadd_ps(delta_z_f32x4, delta_z_f32x4,
                        _mm_fmadd_ps(delta_y_f32x4, delta_y_f32x4, _mm_mul_ps(delta_x_f32x4, delta_x_f32x4)));
}

/** The cost of one arc from a row point to a rotated column point, rounded as its pricing lane. */
NUMKONG_INLINE nk_f64_t nk_shapes_point_cost_f64_haswell_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t row,
                                                          nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    nk_b256_vec_t cost_vec;
    cost_vec.ymm_pd = nk_shapes_point_costs_f64x4_haswell_(
        _mm256_set1_pd(row_points[row]), _mm256_set1_pd(row_points[stride + row]),
        _mm256_set1_pd(row_points[2 * stride + row]), _mm256_set1_pd(rotated_points[column]),
        _mm256_set1_pd(rotated_points[stride + column]), _mm256_set1_pd(rotated_points[2 * stride + column]));
    return cost_vec.f64s[0];
}

/** @copydoc nk_shapes_point_cost_f64_haswell_ */
NUMKONG_INLINE nk_f32_t nk_shapes_point_cost_f32_haswell_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t row,
                                                          nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    nk_b128_vec_t cost_vec;
    cost_vec.xmm_ps = nk_shapes_point_costs_f32x4_haswell_(
        _mm_set1_ps(row_points[row]), _mm_set1_ps(row_points[stride + row]), _mm_set1_ps(row_points[2 * stride + row]),
        _mm_set1_ps(rotated_points[column]), _mm_set1_ps(rotated_points[stride + column]),
        _mm_set1_ps(rotated_points[2 * stride + column]));
    return cost_vec.f32s[0];
}

/** The reduced costs (c − πᵢ) + πⱼ of row @p row's arcs to column @p column and the next 3, with
 *  matrix costs −G taken as πⱼ − (G + πᵢ), which rounds the same. */
NUMKONG_INLINE __m256d nk_shapes_reduced_costs_f64_haswell_(nk_shapes_workspace_f64_serial_t const *views,
                                                            nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t row, nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points + column;
    __m256d const row_potential_f64x4 = _mm256_set1_pd(views->potential[row]);
    __m256d const column_potentials_f64x4 = _mm256_loadu_pd(views->potential + rows + column);
    if (gradient)
        return _mm256_sub_pd(column_potentials_f64x4,
                             _mm256_add_pd(_mm256_loadu_pd(gradient + row * columns + column), row_potential_f64x4));
    __m256d const costs_f64x4 = nk_shapes_point_costs_f64x4_haswell_(
        _mm256_set1_pd(row_points[row]), _mm256_set1_pd(row_points[stride + row]),
        _mm256_set1_pd(row_points[2 * stride + row]), _mm256_loadu_pd(rotated_points),
        _mm256_loadu_pd(rotated_points + stride), _mm256_loadu_pd(rotated_points + 2 * stride));
    return _mm256_add_pd(_mm256_sub_pd(costs_f64x4, row_potential_f64x4), column_potentials_f64x4);
}

/** @copydoc nk_shapes_reduced_costs_f64_haswell_ */
NUMKONG_INLINE __m256d nk_shapes_reduced_costs_f32_haswell_(nk_shapes_workspace_f32_serial_t const *views,
                                                            nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t row, nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points + column;
    __m256d const row_potential_f64x4 = _mm256_set1_pd(views->potential[row]);
    __m256d const column_potentials_f64x4 = _mm256_loadu_pd(views->potential + rows + column);
    if (gradient)
        return _mm256_sub_pd(column_potentials_f64x4,
                             _mm256_add_pd(_mm256_loadu_pd(gradient + row * columns + column), row_potential_f64x4));
    __m256d const costs_f64x4 = _mm256_cvtps_pd(nk_shapes_point_costs_f32x4_haswell_(
        _mm_set1_ps(row_points[row]), _mm_set1_ps(row_points[stride + row]), _mm_set1_ps(row_points[2 * stride + row]),
        _mm_loadu_ps(rotated_points), _mm_loadu_ps(rotated_points + stride),
        _mm_loadu_ps(rotated_points + 2 * stride)));
    return _mm256_add_pd(_mm256_sub_pd(costs_f64x4, row_potential_f64x4), column_potentials_f64x4);
}

/** Scans @p count reduced costs from the cursor in row-major order and keeps the first most
 *  negative one below @p entering_cost, as the serial scan does. A first pass takes their least
 *  4 columns at a time, and only a block whose least beats @p entering_cost is scanned again
 *  for the first arc reaching it. Lanes past the scanned part of a row read neighboring
 *  workspace and cost infinity. */
NUMKONG_INLINE void nk_shapes_price_block_f64_haswell_(nk_shapes_workspace_f64_serial_t const *views,
                                                       nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                       nk_size_t count, nk_size_t *cursor_row, nk_size_t *cursor_column,
                                                       nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                       nk_size_t *entering_column) {
    __m256d const infinity_f64x4 = _mm256_set1_pd(NUMKONG_F64_INF);
    nk_size_t const start_row = *cursor_row, start_column = *cursor_column;
    nk_size_t row = start_row, column = start_column;
    __m256d least_f64x4 = infinity_f64x4;
    for (nk_size_t offset = 0; offset != count;) {
        nk_size_t const segment = columns - column < count - offset ? columns - column : count - offset;
        for (nk_size_t step = 0; step < segment; step += 4) {
            __m256d reduced_f64x4 = nk_shapes_reduced_costs_f64_haswell_(views, gradient, rows, columns, row,
                                                                         column + step);
            if (step + 4 > segment)
                reduced_f64x4 = _mm256_blendv_pd(infinity_f64x4, reduced_f64x4,
                                                 nk_shapes_valid_lanes_f64x4_haswell_(segment - step));
            least_f64x4 = _mm256_min_pd(least_f64x4, reduced_f64x4);
        }
        offset += segment, column += segment;
        if (column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
    *cursor_row = row, *cursor_column = column;
    nk_f64_t const least = nk_reduce_min_f64x4_haswell_(least_f64x4);
    if (!(least < *entering_cost)) return;
    __m256d const least_broadcast_f64x4 = _mm256_set1_pd(least);
    *entering_cost = least, row = start_row, column = start_column;
    for (;;) {
        nk_size_t const segment = columns - column;
        for (nk_size_t step = 0; step < segment; step += 4) {
            __m256d reduced_f64x4 = nk_shapes_reduced_costs_f64_haswell_(views, gradient, rows, columns, row,
                                                                         column + step);
            if (step + 4 > segment)
                reduced_f64x4 = _mm256_blendv_pd(infinity_f64x4, reduced_f64x4,
                                                 nk_shapes_valid_lanes_f64x4_haswell_(segment - step));
            int const equal_lanes = _mm256_movemask_pd(_mm256_cmp_pd(reduced_f64x4, least_broadcast_f64x4, _CMP_EQ_OQ));
            if (!equal_lanes) continue;
            *entering_row = row, *entering_column = column + step + _tzcnt_u32((unsigned int)equal_lanes);
            return;
        }
        column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
}

/** @copydoc nk_shapes_price_block_f64_haswell_ */
NUMKONG_INLINE void nk_shapes_price_block_f32_haswell_(nk_shapes_workspace_f32_serial_t const *views,
                                                       nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                       nk_size_t count, nk_size_t *cursor_row, nk_size_t *cursor_column,
                                                       nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                       nk_size_t *entering_column) {
    __m256d const infinity_f64x4 = _mm256_set1_pd(NUMKONG_F64_INF);
    nk_size_t const start_row = *cursor_row, start_column = *cursor_column;
    nk_size_t row = start_row, column = start_column;
    __m256d least_f64x4 = infinity_f64x4;
    for (nk_size_t offset = 0; offset != count;) {
        nk_size_t const segment = columns - column < count - offset ? columns - column : count - offset;
        for (nk_size_t step = 0; step < segment; step += 4) {
            __m256d reduced_f64x4 = nk_shapes_reduced_costs_f32_haswell_(views, gradient, rows, columns, row,
                                                                         column + step);
            if (step + 4 > segment)
                reduced_f64x4 = _mm256_blendv_pd(infinity_f64x4, reduced_f64x4,
                                                 nk_shapes_valid_lanes_f64x4_haswell_(segment - step));
            least_f64x4 = _mm256_min_pd(least_f64x4, reduced_f64x4);
        }
        offset += segment, column += segment;
        if (column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
    *cursor_row = row, *cursor_column = column;
    nk_f64_t const least = nk_reduce_min_f64x4_haswell_(least_f64x4);
    if (!(least < *entering_cost)) return;
    __m256d const least_broadcast_f64x4 = _mm256_set1_pd(least);
    *entering_cost = least, row = start_row, column = start_column;
    for (;;) {
        nk_size_t const segment = columns - column;
        for (nk_size_t step = 0; step < segment; step += 4) {
            __m256d reduced_f64x4 = nk_shapes_reduced_costs_f32_haswell_(views, gradient, rows, columns, row,
                                                                         column + step);
            if (step + 4 > segment)
                reduced_f64x4 = _mm256_blendv_pd(infinity_f64x4, reduced_f64x4,
                                                 nk_shapes_valid_lanes_f64x4_haswell_(segment - step));
            int const equal_lanes = _mm256_movemask_pd(_mm256_cmp_pd(reduced_f64x4, least_broadcast_f64x4, _CMP_EQ_OQ));
            if (!equal_lanes) continue;
            *entering_row = row, *entering_column = column + step + _tzcnt_u32((unsigned int)equal_lanes);
            return;
        }
        column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
}

/** Dx × V × Dy into @p gradient for a plan V given by its arcs, 4 columns at a time down every row.
 *  A row length off the width spills each row's last vector into the next row, where its cleared
 *  lanes add zero. */
NUMKONG_INLINE void nk_shapes_distance_gradient_f64_haswell_(nk_u32_t const *plan, nk_size_t plan_stride,
                                                             nk_size_t arcs, nk_f64_t mass,
                                                             nk_f64_t const *row_distances, nk_size_t rows,
                                                             nk_f64_t const *column_distances, nk_size_t columns,
                                                             nk_f64_t *gradient) {
    for (nk_size_t cell = 0; cell < rows * columns; cell += 4) _mm256_storeu_pd(gradient + cell, _mm256_setzero_pd());
    for (nk_size_t arc = 0; arc != arcs; ++arc) {
        if (!plan[2 * plan_stride + arc]) continue;
        __m256d const weight_f64x4 = _mm256_set1_pd((nk_f64_t)plan[2 * plan_stride + arc] / mass);
        nk_f64_t const *row_distances_from = row_distances + plan[arc] * rows;
        nk_f64_t const *column_distances_from = column_distances + plan[plan_stride + arc] * columns;
        for (nk_size_t column = 0; column < columns; column += 4) {
            __m256d column_distances_f64x4 = _mm256_loadu_pd(column_distances_from + column);
            if (column + 4 > columns)
                column_distances_f64x4 = _mm256_and_pd(column_distances_f64x4,
                                                       nk_shapes_valid_lanes_f64x4_haswell_(columns - column));
            __m256d const weighted_f64x4 = _mm256_mul_pd(column_distances_f64x4, weight_f64x4);
            nk_f64_t *gradient_column = gradient + column;
            for (nk_size_t row = 0; row != rows; ++row, gradient_column += columns)
                _mm256_storeu_pd(gradient_column,
                                 _mm256_fmadd_pd(weighted_f64x4, _mm256_set1_pd(row_distances_from[row]),
                                                 _mm256_loadu_pd(gradient_column)));
        }
    }
}

/** @copydoc nk_shapes_plan_step_f64_serial_ */
NUMKONG_INLINE void nk_shapes_plan_step_f64_haswell_(nk_f64_t *plan, nk_f64_t *gradient,
                                                     nk_f64_t const *vertex_gradient, nk_size_t cells, nk_f64_t keep,
                                                     nk_f64_t step) {
    __m256d const keep_f64x4 = _mm256_set1_pd(keep), step_f64x4 = _mm256_set1_pd(step);
    for (nk_size_t cell = 0; cell < cells; cell += 4) {
        _mm256_storeu_pd(plan + cell, _mm256_mul_pd(_mm256_loadu_pd(plan + cell), keep_f64x4));
        _mm256_storeu_pd(gradient + cell, _mm256_fmadd_pd(_mm256_loadu_pd(vertex_gradient + cell), step_f64x4,
                                                          _mm256_mul_pd(_mm256_loadu_pd(gradient + cell), keep_f64x4)));
    }
}

/** @copydoc nk_shapes_plan_score_f64_serial_ */
NUMKONG_INLINE nk_f64_t nk_shapes_plan_score_f64_haswell_(nk_f64_t const *gradient, nk_f64_t const *plan,
                                                          nk_size_t cells, nk_f64_t *gradient_scale) {
    __m256d const magnitude_bits_f64x4 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFll));
    __m256d score_f64x4 = _mm256_setzero_pd(), scale_f64x4 = _mm256_setzero_pd();
    for (nk_size_t cell = 0; cell < cells; cell += 4) {
        __m256d gradient_f64x4 = _mm256_loadu_pd(gradient + cell), plan_f64x4 = _mm256_loadu_pd(plan + cell);
        // The matrices' padding is uninitialized, so its lanes clear to zero
        if (cell + 4 > cells) {
            __m256d const valid_f64x4 = nk_shapes_valid_lanes_f64x4_haswell_(cells - cell);
            gradient_f64x4 = _mm256_and_pd(gradient_f64x4, valid_f64x4);
            plan_f64x4 = _mm256_and_pd(plan_f64x4, valid_f64x4);
        }
        score_f64x4 = _mm256_fmadd_pd(gradient_f64x4, plan_f64x4, score_f64x4);
        scale_f64x4 = _mm256_max_pd(scale_f64x4, _mm256_and_pd(gradient_f64x4, magnitude_bits_f64x4));
    }
    *gradient_scale = nk_reduce_max_f64x4_haswell_(scale_f64x4);
    return nk_reduce_add_f64x4_haswell_(score_f64x4);
}

nk_define_shapes_network_simplex_(f32, haswell, nk_shapes_point_cost_f32_haswell_, nk_shapes_price_block_f32_haswell_)
nk_define_shapes_network_simplex_(f64, haswell, nk_shapes_point_cost_f64_haswell_, nk_shapes_price_block_f64_haswell_)
nk_define_shapes_wasserstein_procrustes_(f32, haswell, NUMKONG_F32_INF)
nk_define_shapes_wasserstein_procrustes_(f64, haswell, NUMKONG_F64_INF)
nk_define_shapes_gromov_wasserstein_(f32, haswell, nk_shapes_distance_gradient_f64_haswell_,
                                     nk_shapes_plan_step_f64_haswell_, nk_shapes_plan_score_f64_haswell_)
nk_define_shapes_gromov_wasserstein_(f64, haswell, nk_shapes_distance_gradient_f64_haswell_,
                                     nk_shapes_plan_step_f64_haswell_, nk_shapes_plan_score_f64_haswell_)

#if NUMKONG_TARGET_HASWELL
nk_define_shapes_flip_kernels_(f16, f32, haswell, nk_shapes_nearest_f16_haswell_, NUMKONG_F32_INF)
nk_define_shapes_flip_kernels_(f32, f64, haswell, nk_shapes_nearest_f32_haswell_, NUMKONG_F64_INF)
nk_define_shapes_transport_kernels_(f16, f32, haswell, nk_sqrt_f32_serial_, NUMKONG_F32_INF)
nk_define_shapes_transport_kernels_(f32, f64, haswell, nk_sqrt_f64_serial_, NUMKONG_F64_INF)
#endif // NUMKONG_TARGET_HASWELL

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_HASWELL_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_MESHES_HASWELL_H
