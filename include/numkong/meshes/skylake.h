/**
 *  @file include/numkong/meshes/skylake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief Many-to-many shape distances between 3D point clouds for Skylake, AVX-512.
 *
 *  @sa include/numkong/meshes.h
 *
 *  @section shapes_skylake_instructions Key AVX-512 Shapes Instructions
 *
 *  @verbatim
 *  Intrinsic              Instruction                          Skylake-X     Genoa
 *  _mm512_fmadd_ps/pd     VFMADD (ZMM, ZMM, ZMM)               4cy @ p05     4cy @ p01
 *  _mm512_sub_ps/pd       VSUBPS/PD (ZMM, ZMM, ZMM)            4cy @ p05     3cy @ p23
 *  _mm512_min_ps/pd       VMINPS/PD (ZMM, ZMM, ZMM)            4cy @ p05     2cy @ p01
 *  _mm512_mask_cvtph_ps   VCVTPH2PS (ZMM {K}, YMM)             7cy @ p0+p5   5cy @ p01
 *  _mm512_mask_cvtps_pd   VCVTPS2PD (ZMM {K}, YMM)             7cy @ p0+p5   4cy @ p12+p23
 *  _mm512_shuffle_ps      VSHUFPS (ZMM, ZMM, ZMM, I8)          1cy @ p5      1cy @ p12
 *  _mm512_shuffle_f32x4   VSHUFF32X4 (ZMM, ZMM, ZMM, I8)       3cy @ p5      4cy @ p12
 *  _mm512_permutexvar_ps  VPERMPS (ZMM, ZMM, ZMM)              3cy @ p5      4cy @ p12
 *  _mm256_sqrt_ps         VSQRTPS (YMM, YMM)                   12cy @ p0     15cy @ p01
 *  _mm512_sqrt_pd         VSQRTPD (ZMM, ZMM)                   23cy @ p0+p0+p05  21cy @ p01
 *  @endverbatim
 *
 *  Chamfer and Hausdorff broadcast one outer point against a register of inner points, which sit
 *  contiguous per axis in the pack, and keep one running minimum per axis flip. The x conversion
 *  of the last inner register writes infinity past the cloud, so its padding never wins a minimum.
 *
 *  The transport kinds run the serial solvers around vector passes: block pricing eight arcs at a
 *  time, and the f64 Gromov-Wasserstein gradient products, plan steps and scores, each masking
 *  the lanes past a row. Arc costs round in fused multiply-adds, and the scalar cost behind the
 *  tree's potentials runs the same lanes, so tree arcs keep a zero reduced cost.
 */
#ifndef NUMKONG_MESHES_SKYLAKE_H
#define NUMKONG_MESHES_SKYLAKE_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_SKYLAKE_

#include "numkong/types.h"
#include "numkong/meshes/serial.h"  // `nk_define_shapes_network_simplex_`, `nk_shapes_workspace_f64_serial_t`
#include "numkong/reduce/skylake.h" // `nk_reduce_min_f64x8_skylake_`, `nk_reduce_add_f64x8_skylake_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,avx512f,avx512vl,avx512bw,avx512dq,f16c,fma,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "avx512f", "avx512vl", "avx512bw", "avx512dq", "f16c", "fma", "bmi", "bmi2")
#endif

/** Minima of neighboring lanes of @p first and @p second, laid out as VHADDPS lays out sums. */
NUMKONG_INLINE __m512 nk_shapes_neighbor_minima_f32x16_skylake_(__m512 first_f32x16, __m512 second_f32x16) {
    return _mm512_min_ps(_mm512_shuffle_ps(first_f32x16, second_f32x16, 0x88),
                         _mm512_shuffle_ps(first_f32x16, second_f32x16, 0xDD));
}

/** Folds 8 per-flip running minima into one vector holding each flip's minimum in its own lane. */
NUMKONG_INLINE __m256 nk_shapes_flip_minima_f32x16_skylake_(__m512 const *minima_f32x16) {
    // Each pair of flips shares a register, one flip per 256-bit half
    __m512 halves_f32x16[4];
    for (int pair = 0; pair != 4; ++pair)
        halves_f32x16[pair] = _mm512_min_ps(
            _mm512_shuffle_f32x4(minima_f32x16[2 * pair], minima_f32x16[2 * pair + 1], 0x44),
            _mm512_shuffle_f32x4(minima_f32x16[2 * pair], minima_f32x16[2 * pair + 1], 0xEE));
    // The lower 256 bits now hold flips 0, 2, 4 and 6, the upper ones flips 1, 3, 5 and 7
    __m512 const interleaved_f32x16 = nk_shapes_neighbor_minima_f32x16_skylake_(
        nk_shapes_neighbor_minima_f32x16_skylake_(halves_f32x16[0], halves_f32x16[1]),
        nk_shapes_neighbor_minima_f32x16_skylake_(halves_f32x16[2], halves_f32x16[3]));
    __m512 const folded_f32x16 = _mm512_min_ps(interleaved_f32x16,
                                               _mm512_shuffle_f32x4(interleaved_f32x16, interleaved_f32x16, 0xB1));
    __m512i const flip_order_i32x16 = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 0, 8, 1, 9, 2, 10, 3, 11);
    return _mm512_castps512_ps256(_mm512_permutexvar_ps(flip_order_i32x16, folded_f32x16));
}

/** Folds 8 per-flip running minima into one vector holding each flip's minimum in its own lane. */
NUMKONG_INLINE __m512d nk_shapes_flip_minima_f64x8_skylake_(__m512d const *minima_f64x8) {
    // Each 128-bit lane holds a pair of flips over its own quarter of the inner lanes
    __m512d pairs_f64x8[4];
    for (int pair = 0; pair != 4; ++pair)
        pairs_f64x8[pair] = _mm512_min_pd(_mm512_unpacklo_pd(minima_f64x8[2 * pair], minima_f64x8[2 * pair + 1]),
                                          _mm512_unpackhi_pd(minima_f64x8[2 * pair], minima_f64x8[2 * pair + 1]));
    __m512d const quads_0123_f64x8 = _mm512_min_pd(_mm512_shuffle_f64x2(pairs_f64x8[0], pairs_f64x8[1], 0x88),
                                                   _mm512_shuffle_f64x2(pairs_f64x8[0], pairs_f64x8[1], 0xDD));
    __m512d const quads_4567_f64x8 = _mm512_min_pd(_mm512_shuffle_f64x2(pairs_f64x8[2], pairs_f64x8[3], 0x88),
                                                   _mm512_shuffle_f64x2(pairs_f64x8[2], pairs_f64x8[3], 0xDD));
    return _mm512_min_pd(_mm512_shuffle_f64x2(quads_0123_f64x8, quads_4567_f64x8, 0x88),
                         _mm512_shuffle_f64x2(quads_0123_f64x8, quads_4567_f64x8, 0xDD));
}

/** One directed Chamfer and Hausdorff pass between f16 packs, 16 inner points per step. */
NUMKONG_INLINE void nk_shapes_nearest_f16_skylake_(                            //
    nk_f16_t const *outer_axes, nk_size_t outer_stride, nk_size_t outer_count, //
    nk_f16_t const *inner_axes, nk_size_t inner_stride, nk_size_t inner_count, //
    nk_f32_t *sums, nk_f32_t *maxima) {
    __m512 const infinity_f32x16 = _mm512_set1_ps(NUMKONG_F32_INF);
    __m256 sums_f32x8 = _mm256_loadu_ps(sums), maxima_f32x8 = _mm256_loadu_ps(maxima);
    for (nk_size_t outer_index = 0; outer_index != outer_count; ++outer_index) {
        nk_f16_t const *outer_point = outer_axes + outer_index;
        __m512 const outer_x_f32x16 = _mm512_cvtph_ps(_mm256_set1_epi16(*(nk_i16_t const *)outer_point));
        __m512 const outer_y_f32x16 = _mm512_cvtph_ps(
            _mm256_set1_epi16(*(nk_i16_t const *)(outer_point + outer_stride)));
        __m512 const outer_z_f32x16 = _mm512_cvtph_ps(
            _mm256_set1_epi16(*(nk_i16_t const *)(outer_point + 2 * outer_stride)));
        __m512 minima_f32x16[8];
        for (int flip = 0; flip != 8; ++flip) minima_f32x16[flip] = infinity_f32x16;
        for (nk_size_t inner_index = 0; inner_index < inner_count; inner_index += 16) {
            nk_f16_t const *inner_points = inner_axes + inner_index;
            nk_size_t const remaining = inner_count - inner_index;
            __mmask16 const valid_m16 = (__mmask16)_bzhi_u32(0xFFFF, remaining < 16 ? (unsigned int)remaining : 16);
            __m512 const inner_x_f32x16 = _mm512_mask_cvtph_ps(infinity_f32x16, valid_m16,
                                                               _mm256_loadu_si256((__m256i const *)inner_points));
            __m512 const inner_y_f32x16 = _mm512_cvtph_ps(
                _mm256_loadu_si256((__m256i const *)(inner_points + inner_stride)));
            __m512 const inner_z_f32x16 = _mm512_cvtph_ps(
                _mm256_loadu_si256((__m256i const *)(inner_points + 2 * inner_stride)));
            __m512 const minus_x_f32x16 = _mm512_sub_ps(outer_x_f32x16, inner_x_f32x16);
            __m512 const plus_x_f32x16 = _mm512_add_ps(outer_x_f32x16, inner_x_f32x16);
            __m512 const minus_y_f32x16 = _mm512_sub_ps(outer_y_f32x16, inner_y_f32x16);
            __m512 const plus_y_f32x16 = _mm512_add_ps(outer_y_f32x16, inner_y_f32x16);
            __m512 const minus_z_f32x16 = _mm512_sub_ps(outer_z_f32x16, inner_z_f32x16);
            __m512 const plus_z_f32x16 = _mm512_add_ps(outer_z_f32x16, inner_z_f32x16);
            __m512 const minus_xx_f32x16 = _mm512_mul_ps(minus_x_f32x16, minus_x_f32x16);
            __m512 const plus_xx_f32x16 = _mm512_mul_ps(plus_x_f32x16, plus_x_f32x16);
            __m512 const planar_f32x16[4] = {_mm512_fmadd_ps(minus_y_f32x16, minus_y_f32x16, minus_xx_f32x16),
                                             _mm512_fmadd_ps(minus_y_f32x16, minus_y_f32x16, plus_xx_f32x16),
                                             _mm512_fmadd_ps(plus_y_f32x16, plus_y_f32x16, minus_xx_f32x16),
                                             _mm512_fmadd_ps(plus_y_f32x16, plus_y_f32x16, plus_xx_f32x16)};
            for (int flip = 0; flip != 4; ++flip) {
                minima_f32x16[flip] = _mm512_min_ps(
                    minima_f32x16[flip], _mm512_fmadd_ps(minus_z_f32x16, minus_z_f32x16, planar_f32x16[flip]));
                minima_f32x16[flip + 4] = _mm512_min_ps(
                    minima_f32x16[flip + 4], _mm512_fmadd_ps(plus_z_f32x16, plus_z_f32x16, planar_f32x16[flip]));
            }
        }
        __m256 const distances_f32x8 = _mm256_sqrt_ps(nk_shapes_flip_minima_f32x16_skylake_(minima_f32x16));
        sums_f32x8 = _mm256_add_ps(sums_f32x8, distances_f32x8);
        maxima_f32x8 = _mm256_max_ps(maxima_f32x8, distances_f32x8);
    }
    _mm256_storeu_ps(sums, sums_f32x8);
    _mm256_storeu_ps(maxima, maxima_f32x8);
}

/** One directed Chamfer and Hausdorff pass between f32 packs, 8 inner points per step in f64. */
NUMKONG_INLINE void nk_shapes_nearest_f32_skylake_(                            //
    nk_f32_t const *outer_axes, nk_size_t outer_stride, nk_size_t outer_count, //
    nk_f32_t const *inner_axes, nk_size_t inner_stride, nk_size_t inner_count, //
    nk_f64_t *sums, nk_f64_t *maxima) {
    __m512d const infinity_f64x8 = _mm512_set1_pd(NUMKONG_F64_INF);
    __m512d sums_f64x8 = _mm512_loadu_pd(sums), maxima_f64x8 = _mm512_loadu_pd(maxima);
    for (nk_size_t outer_index = 0; outer_index != outer_count; ++outer_index) {
        nk_f32_t const *outer_point = outer_axes + outer_index;
        __m512d const outer_x_f64x8 = _mm512_set1_pd(outer_point[0]);
        __m512d const outer_y_f64x8 = _mm512_set1_pd(outer_point[outer_stride]);
        __m512d const outer_z_f64x8 = _mm512_set1_pd(outer_point[2 * outer_stride]);
        __m512d minima_f64x8[8];
        for (int flip = 0; flip != 8; ++flip) minima_f64x8[flip] = infinity_f64x8;
        for (nk_size_t inner_index = 0; inner_index < inner_count; inner_index += 8) {
            nk_f32_t const *inner_points = inner_axes + inner_index;
            nk_size_t const remaining = inner_count - inner_index;
            __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, remaining < 8 ? (unsigned int)remaining : 8);
            __m512d const inner_x_f64x8 = _mm512_mask_cvtps_pd(infinity_f64x8, valid_m8, _mm256_loadu_ps(inner_points));
            __m512d const inner_y_f64x8 = _mm512_cvtps_pd(_mm256_loadu_ps(inner_points + inner_stride));
            __m512d const inner_z_f64x8 = _mm512_cvtps_pd(_mm256_loadu_ps(inner_points + 2 * inner_stride));
            __m512d const minus_x_f64x8 = _mm512_sub_pd(outer_x_f64x8, inner_x_f64x8);
            __m512d const plus_x_f64x8 = _mm512_add_pd(outer_x_f64x8, inner_x_f64x8);
            __m512d const minus_y_f64x8 = _mm512_sub_pd(outer_y_f64x8, inner_y_f64x8);
            __m512d const plus_y_f64x8 = _mm512_add_pd(outer_y_f64x8, inner_y_f64x8);
            __m512d const minus_z_f64x8 = _mm512_sub_pd(outer_z_f64x8, inner_z_f64x8);
            __m512d const plus_z_f64x8 = _mm512_add_pd(outer_z_f64x8, inner_z_f64x8);
            __m512d const minus_xx_f64x8 = _mm512_mul_pd(minus_x_f64x8, minus_x_f64x8);
            __m512d const plus_xx_f64x8 = _mm512_mul_pd(plus_x_f64x8, plus_x_f64x8);
            __m512d const planar_f64x8[4] = {_mm512_fmadd_pd(minus_y_f64x8, minus_y_f64x8, minus_xx_f64x8),
                                             _mm512_fmadd_pd(minus_y_f64x8, minus_y_f64x8, plus_xx_f64x8),
                                             _mm512_fmadd_pd(plus_y_f64x8, plus_y_f64x8, minus_xx_f64x8),
                                             _mm512_fmadd_pd(plus_y_f64x8, plus_y_f64x8, plus_xx_f64x8)};
            for (int flip = 0; flip != 4; ++flip) {
                minima_f64x8[flip] = _mm512_min_pd(minima_f64x8[flip],
                                                   _mm512_fmadd_pd(minus_z_f64x8, minus_z_f64x8, planar_f64x8[flip]));
                minima_f64x8[flip + 4] = _mm512_min_pd(minima_f64x8[flip + 4],
                                                       _mm512_fmadd_pd(plus_z_f64x8, plus_z_f64x8, planar_f64x8[flip]));
            }
        }
        __m512d const distances_f64x8 = _mm512_sqrt_pd(nk_shapes_flip_minima_f64x8_skylake_(minima_f64x8));
        sums_f64x8 = _mm512_add_pd(sums_f64x8, distances_f64x8);
        maxima_f64x8 = _mm512_max_pd(maxima_f64x8, distances_f64x8);
    }
    _mm512_storeu_pd(sums, sums_f64x8);
    _mm512_storeu_pd(maxima, maxima_f64x8);
}

/** Squared distances from a broadcast row point to 8 column points in fused multiply-adds, the
 *  rounding both the block pricing and the scalar cost behind the potentials share. */
NUMKONG_INLINE __m512d nk_shapes_point_costs_f64x8_skylake_(__m512d row_x_f64x8, __m512d row_y_f64x8,
                                                            __m512d row_z_f64x8, __m512d column_x_f64x8,
                                                            __m512d column_y_f64x8, __m512d column_z_f64x8) {
    __m512d const delta_x_f64x8 = _mm512_sub_pd(row_x_f64x8, column_x_f64x8);
    __m512d const delta_y_f64x8 = _mm512_sub_pd(row_y_f64x8, column_y_f64x8);
    __m512d const delta_z_f64x8 = _mm512_sub_pd(row_z_f64x8, column_z_f64x8);
    return _mm512_fmadd_pd(delta_z_f64x8, delta_z_f64x8,
                           _mm512_fmadd_pd(delta_y_f64x8, delta_y_f64x8, _mm512_mul_pd(delta_x_f64x8, delta_x_f64x8)));
}

/** @copydoc nk_shapes_point_costs_f64x8_skylake_ */
NUMKONG_INLINE __m256 nk_shapes_point_costs_f32x8_skylake_(__m256 row_x_f32x8, __m256 row_y_f32x8, __m256 row_z_f32x8,
                                                           __m256 column_x_f32x8, __m256 column_y_f32x8,
                                                           __m256 column_z_f32x8) {
    __m256 const delta_x_f32x8 = _mm256_sub_ps(row_x_f32x8, column_x_f32x8);
    __m256 const delta_y_f32x8 = _mm256_sub_ps(row_y_f32x8, column_y_f32x8);
    __m256 const delta_z_f32x8 = _mm256_sub_ps(row_z_f32x8, column_z_f32x8);
    return _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8,
                           _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, _mm256_mul_ps(delta_x_f32x8, delta_x_f32x8)));
}

/** The cost of one arc from a row point to a rotated column point, rounded as its pricing lane. */
NUMKONG_INLINE nk_f64_t nk_shapes_point_cost_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t row,
                                                          nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    nk_b512_vec_t cost_vec;
    cost_vec.zmm_pd = nk_shapes_point_costs_f64x8_skylake_(
        _mm512_set1_pd(row_points[row]), _mm512_set1_pd(row_points[stride + row]),
        _mm512_set1_pd(row_points[2 * stride + row]), _mm512_set1_pd(rotated_points[column]),
        _mm512_set1_pd(rotated_points[stride + column]), _mm512_set1_pd(rotated_points[2 * stride + column]));
    return cost_vec.f64s[0];
}

/** @copydoc nk_shapes_point_cost_f64_skylake_ */
NUMKONG_INLINE nk_f32_t nk_shapes_point_cost_f32_skylake_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t row,
                                                          nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    nk_b256_vec_t cost_vec;
    cost_vec.ymm_ps = nk_shapes_point_costs_f32x8_skylake_(
        _mm256_set1_ps(row_points[row]), _mm256_set1_ps(row_points[stride + row]),
        _mm256_set1_ps(row_points[2 * stride + row]), _mm256_set1_ps(rotated_points[column]),
        _mm256_set1_ps(rotated_points[stride + column]), _mm256_set1_ps(rotated_points[2 * stride + column]));
    return cost_vec.f32s[0];
}

/** The reduced costs (c − πᵢ) + πⱼ of row @p row's arcs to the @p valid_m8 columns from @p column,
 *  with matrix costs −G taken as πⱼ − (G + πᵢ), which rounds the same. */
NUMKONG_INLINE __m512d nk_shapes_reduced_costs_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views,
                                                            nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t row, nk_size_t column, __mmask8 valid_m8) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points + column;
    __m512d const row_potential_f64x8 = _mm512_set1_pd(views->potential[row]);
    __m512d const column_potentials_f64x8 = _mm512_maskz_loadu_pd(valid_m8, views->potential + rows + column);
    if (gradient)
        return _mm512_sub_pd(
            column_potentials_f64x8,
            _mm512_add_pd(_mm512_maskz_loadu_pd(valid_m8, gradient + row * columns + column), row_potential_f64x8));
    __m512d const costs_f64x8 = nk_shapes_point_costs_f64x8_skylake_(
        _mm512_set1_pd(row_points[row]), _mm512_set1_pd(row_points[stride + row]),
        _mm512_set1_pd(row_points[2 * stride + row]), _mm512_maskz_loadu_pd(valid_m8, rotated_points),
        _mm512_maskz_loadu_pd(valid_m8, rotated_points + stride),
        _mm512_maskz_loadu_pd(valid_m8, rotated_points + 2 * stride));
    return _mm512_add_pd(_mm512_sub_pd(costs_f64x8, row_potential_f64x8), column_potentials_f64x8);
}

/** @copydoc nk_shapes_reduced_costs_f64_skylake_ */
NUMKONG_INLINE __m512d nk_shapes_reduced_costs_f32_skylake_(nk_shapes_workspace_f32_serial_t const *views,
                                                            nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                            nk_size_t row, nk_size_t column, __mmask8 valid_m8) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points + column;
    __m512d const row_potential_f64x8 = _mm512_set1_pd(views->potential[row]);
    __m512d const column_potentials_f64x8 = _mm512_maskz_loadu_pd(valid_m8, views->potential + rows + column);
    if (gradient)
        return _mm512_sub_pd(
            column_potentials_f64x8,
            _mm512_add_pd(_mm512_maskz_loadu_pd(valid_m8, gradient + row * columns + column), row_potential_f64x8));
    __m512d const costs_f64x8 = _mm512_cvtps_pd(nk_shapes_point_costs_f32x8_skylake_(
        _mm256_set1_ps(row_points[row]), _mm256_set1_ps(row_points[stride + row]),
        _mm256_set1_ps(row_points[2 * stride + row]), _mm256_maskz_loadu_ps(valid_m8, rotated_points),
        _mm256_maskz_loadu_ps(valid_m8, rotated_points + stride),
        _mm256_maskz_loadu_ps(valid_m8, rotated_points + 2 * stride)));
    return _mm512_add_pd(_mm512_sub_pd(costs_f64x8, row_potential_f64x8), column_potentials_f64x8);
}

/** Scans @p count reduced costs from the cursor in row-major order and keeps the first most
 *  negative one below @p entering_cost, as the serial scan does. A first pass takes their least 8
 *  columns at a time, and only a block whose least beats @p entering_cost is scanned again for the
 *  first arc reaching it. In both passes, masks keep the lanes past the scanned part of a row out
 *  of the comparisons. */
NUMKONG_INLINE void nk_shapes_price_block_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views,
                                                       nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                       nk_size_t count, nk_size_t *cursor_row, nk_size_t *cursor_column,
                                                       nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                       nk_size_t *entering_column) {
    nk_size_t const start_row = *cursor_row, start_column = *cursor_column;
    nk_size_t row = start_row, column = start_column;
    __m512d least_f64x8 = _mm512_set1_pd(NUMKONG_F64_INF);
    for (nk_size_t offset = 0; offset != count;) {
        nk_size_t const segment = columns - column < count - offset ? columns - column : count - offset;
        for (nk_size_t step = 0; step < segment; step += 8) {
            nk_size_t const remaining = segment - step;
            __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, remaining < 8 ? (unsigned int)remaining : 8);
            __m512d const reduced_f64x8 = nk_shapes_reduced_costs_f64_skylake_(views, gradient, rows, columns, row,
                                                                               column + step, valid_m8);
            least_f64x8 = _mm512_mask_min_pd(least_f64x8, valid_m8, least_f64x8, reduced_f64x8);
        }
        offset += segment, column += segment;
        if (column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
    *cursor_row = row, *cursor_column = column;
    nk_f64_t const least = nk_reduce_min_f64x8_skylake_(least_f64x8);
    if (!(least < *entering_cost)) return;
    __m512d const least_broadcast_f64x8 = _mm512_set1_pd(least);
    *entering_cost = least, row = start_row, column = start_column;
    for (;;) {
        nk_size_t const segment = columns - column;
        for (nk_size_t step = 0; step < segment; step += 8) {
            nk_size_t const remaining = segment - step;
            __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, remaining < 8 ? (unsigned int)remaining : 8);
            __m512d const reduced_f64x8 = nk_shapes_reduced_costs_f64_skylake_(views, gradient, rows, columns, row,
                                                                               column + step, valid_m8);
            __mmask8 const equal_m8 = _mm512_mask_cmp_pd_mask(valid_m8, reduced_f64x8, least_broadcast_f64x8,
                                                              _CMP_EQ_OQ);
            if (!equal_m8) continue;
            *entering_row = row, *entering_column = column + step + _tzcnt_u32(equal_m8);
            return;
        }
        column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
}

/** @copydoc nk_shapes_price_block_f64_skylake_ */
NUMKONG_INLINE void nk_shapes_price_block_f32_skylake_(nk_shapes_workspace_f32_serial_t const *views,
                                                       nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                       nk_size_t count, nk_size_t *cursor_row, nk_size_t *cursor_column,
                                                       nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                       nk_size_t *entering_column) {
    nk_size_t const start_row = *cursor_row, start_column = *cursor_column;
    nk_size_t row = start_row, column = start_column;
    __m512d least_f64x8 = _mm512_set1_pd(NUMKONG_F64_INF);
    for (nk_size_t offset = 0; offset != count;) {
        nk_size_t const segment = columns - column < count - offset ? columns - column : count - offset;
        for (nk_size_t step = 0; step < segment; step += 8) {
            nk_size_t const remaining = segment - step;
            __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, remaining < 8 ? (unsigned int)remaining : 8);
            __m512d const reduced_f64x8 = nk_shapes_reduced_costs_f32_skylake_(views, gradient, rows, columns, row,
                                                                               column + step, valid_m8);
            least_f64x8 = _mm512_mask_min_pd(least_f64x8, valid_m8, least_f64x8, reduced_f64x8);
        }
        offset += segment, column += segment;
        if (column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
    *cursor_row = row, *cursor_column = column;
    nk_f64_t const least = nk_reduce_min_f64x8_skylake_(least_f64x8);
    if (!(least < *entering_cost)) return;
    __m512d const least_broadcast_f64x8 = _mm512_set1_pd(least);
    *entering_cost = least, row = start_row, column = start_column;
    for (;;) {
        nk_size_t const segment = columns - column;
        for (nk_size_t step = 0; step < segment; step += 8) {
            nk_size_t const remaining = segment - step;
            __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, remaining < 8 ? (unsigned int)remaining : 8);
            __m512d const reduced_f64x8 = nk_shapes_reduced_costs_f32_skylake_(views, gradient, rows, columns, row,
                                                                               column + step, valid_m8);
            __mmask8 const equal_m8 = _mm512_mask_cmp_pd_mask(valid_m8, reduced_f64x8, least_broadcast_f64x8,
                                                              _CMP_EQ_OQ);
            if (!equal_m8) continue;
            *entering_row = row, *entering_column = column + step + _tzcnt_u32(equal_m8);
            return;
        }
        column = 0, row = row + 1 == rows ? 0 : row + 1;
    }
}

/** Dx × V × Dy into @p gradient for a plan V given by its arcs, 8 columns at a time down every row,
 *  masking the columns past the row. */
NUMKONG_INLINE void nk_shapes_distance_gradient_f64_skylake_(nk_u32_t const *plan, nk_size_t plan_stride,
                                                             nk_size_t arcs, nk_f64_t mass,
                                                             nk_f64_t const *row_distances, nk_size_t rows,
                                                             nk_f64_t const *column_distances, nk_size_t columns,
                                                             nk_f64_t *gradient) {
    for (nk_size_t cell = 0; cell < rows * columns; cell += 8) _mm512_storeu_pd(gradient + cell, _mm512_setzero_pd());
    for (nk_size_t arc = 0; arc != arcs; ++arc) {
        if (!plan[2 * plan_stride + arc]) continue;
        __m512d const weight_f64x8 = _mm512_set1_pd((nk_f64_t)plan[2 * plan_stride + arc] / mass);
        nk_f64_t const *row_distances_from = row_distances + plan[arc] * rows;
        nk_f64_t const *column_distances_from = column_distances + plan[plan_stride + arc] * columns;
        for (nk_size_t column = 0; column < columns; column += 8) {
            nk_size_t const remaining = columns - column;
            __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, remaining < 8 ? (unsigned int)remaining : 8);
            __m512d const weighted_f64x8 = _mm512_mul_pd(
                _mm512_maskz_loadu_pd(valid_m8, column_distances_from + column), weight_f64x8);
            nk_f64_t *gradient_column = gradient + column;
            for (nk_size_t row = 0; row != rows; ++row, gradient_column += columns)
                _mm512_mask_storeu_pd(gradient_column, valid_m8,
                                      _mm512_fmadd_pd(weighted_f64x8, _mm512_set1_pd(row_distances_from[row]),
                                                      _mm512_maskz_loadu_pd(valid_m8, gradient_column)));
        }
    }
}

/** @copydoc nk_shapes_plan_step_f64_serial_ */
NUMKONG_INLINE void nk_shapes_plan_step_f64_skylake_(nk_f64_t *plan, nk_f64_t *gradient,
                                                     nk_f64_t const *vertex_gradient, nk_size_t cells, nk_f64_t keep,
                                                     nk_f64_t step) {
    __m512d const keep_f64x8 = _mm512_set1_pd(keep), step_f64x8 = _mm512_set1_pd(step);
    for (nk_size_t cell = 0; cell < cells; cell += 8) {
        _mm512_storeu_pd(plan + cell, _mm512_mul_pd(_mm512_loadu_pd(plan + cell), keep_f64x8));
        _mm512_storeu_pd(gradient + cell, _mm512_fmadd_pd(_mm512_loadu_pd(vertex_gradient + cell), step_f64x8,
                                                          _mm512_mul_pd(_mm512_loadu_pd(gradient + cell), keep_f64x8)));
    }
}

/** @copydoc nk_shapes_plan_score_f64_serial_ */
NUMKONG_INLINE nk_f64_t nk_shapes_plan_score_f64_skylake_(nk_f64_t const *gradient, nk_f64_t const *plan,
                                                          nk_size_t cells, nk_f64_t *gradient_scale) {
    __m512d score_f64x8 = _mm512_setzero_pd(), scale_f64x8 = _mm512_setzero_pd();
    for (nk_size_t cell = 0; cell < cells; cell += 8) {
        nk_size_t const remaining = cells - cell;
        // The matrices' padding is uninitialized, so its lanes load as zero
        __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, remaining < 8 ? (unsigned int)remaining : 8);
        __m512d const gradient_f64x8 = _mm512_maskz_loadu_pd(valid_m8, gradient + cell);
        score_f64x8 = _mm512_fmadd_pd(gradient_f64x8, _mm512_maskz_loadu_pd(valid_m8, plan + cell), score_f64x8);
        scale_f64x8 = _mm512_max_pd(scale_f64x8, _mm512_abs_pd(gradient_f64x8));
    }
    *gradient_scale = nk_reduce_max_f64x8_skylake_(scale_f64x8);
    return nk_reduce_add_f64x8_skylake_(score_f64x8);
}

nk_define_shapes_network_simplex_(f32, skylake, nk_shapes_point_cost_f32_skylake_, nk_shapes_price_block_f32_skylake_)
nk_define_shapes_network_simplex_(f64, skylake, nk_shapes_point_cost_f64_skylake_, nk_shapes_price_block_f64_skylake_)
nk_define_shapes_wasserstein_procrustes_(f32, skylake, NUMKONG_F32_INF)
nk_define_shapes_wasserstein_procrustes_(f64, skylake, NUMKONG_F64_INF)
nk_define_shapes_gromov_wasserstein_(f32, skylake, nk_shapes_distance_gradient_f64_skylake_,
                                     nk_shapes_plan_step_f64_skylake_, nk_shapes_plan_score_f64_skylake_)
nk_define_shapes_gromov_wasserstein_(f64, skylake, nk_shapes_distance_gradient_f64_skylake_,
                                     nk_shapes_plan_step_f64_skylake_, nk_shapes_plan_score_f64_skylake_)

#if NUMKONG_TARGET_SKYLAKE
nk_define_shapes_flip_kernels_(f16, f32, skylake, nk_shapes_nearest_f16_skylake_, NUMKONG_F32_INF)
nk_define_shapes_flip_kernels_(f32, f64, skylake, nk_shapes_nearest_f32_skylake_, NUMKONG_F64_INF)
nk_define_shapes_transport_kernels_(f16, f32, skylake, nk_sqrt_f32_serial_, NUMKONG_F32_INF)
nk_define_shapes_transport_kernels_(f32, f64, skylake, nk_sqrt_f64_serial_, NUMKONG_F64_INF)
#endif // NUMKONG_TARGET_SKYLAKE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NUMKONG_ARCH_X8664_SKYLAKE_
#endif // NUMKONG_ARCH_X8664_
#endif // NUMKONG_MESHES_SKYLAKE_H
