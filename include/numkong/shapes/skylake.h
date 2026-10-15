/**
 *  @file include/numkong/shapes/skylake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief Many-to-many shape distances between 3D point clouds for Skylake, AVX-512.
 *
 *  @sa include/numkong/shapes.h
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
 *  The transport kinds run the serial solvers around block pricing eight arcs at a time, masking
 *  the lanes past a row. Arc costs round in fused multiply-adds, and the scalar cost behind the
 *  tree's potentials runs the same lanes, so tree arcs keep a zero reduced cost.
 *
 *  f64 packs load their points directly and sum Chamfer's roots, each plan's cost and every
 *  Gromov-Wasserstein reduction with TwoProd and TwoSum.
 */
#ifndef NUMKONG_SHAPES_SKYLAKE_H
#define NUMKONG_SHAPES_SKYLAKE_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_SKYLAKE_

#include "numkong/types.h"
#include "numkong/cast/skylake.h"   // `nk_load_f16x16_to_f32x16_skylake_`
#include "numkong/shapes/serial.h"  // `nk_define_shapes_network_simplex_`, `nk_shapes_workspace_f64_serial_t`
#include "numkong/reduce/skylake.h" // `nk_reduce_min_f64x8_skylake_`
#include "numkong/dot/skylake.h"    // `nk_dot2_f64x8_skylake_`, `nk_dot_stable_sum_f64x8_skylake_`

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

#if NUMKONG_TARGET_SKYLAKE

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

/** Lowers the running minima of the 8 flips by one outer point's squared distances to 16 inner
 *  points, the difference and the sum of each axis standing for its two signs. */
NUMKONG_INLINE void nk_shapes_lower_nearest_f32x16_skylake_(__m512 outer_x_f32x16, __m512 outer_y_f32x16,
                                                            __m512 outer_z_f32x16, __m512 inner_x_f32x16,
                                                            __m512 inner_y_f32x16, __m512 inner_z_f32x16,
                                                            __m512 *minima_f32x16) {
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
        minima_f32x16[flip] = _mm512_min_ps(minima_f32x16[flip],
                                            _mm512_fmadd_ps(minus_z_f32x16, minus_z_f32x16, planar_f32x16[flip]));
        minima_f32x16[flip + 4] = _mm512_min_ps(minima_f32x16[flip + 4],
                                                _mm512_fmadd_ps(plus_z_f32x16, plus_z_f32x16, planar_f32x16[flip]));
    }
}

/** @copydoc nk_shapes_lower_nearest_f32x16_skylake_ */
NUMKONG_INLINE void nk_shapes_lower_nearest_f64x8_skylake_(__m512d outer_x_f64x8, __m512d outer_y_f64x8,
                                                           __m512d outer_z_f64x8, __m512d inner_x_f64x8,
                                                           __m512d inner_y_f64x8, __m512d inner_z_f64x8,
                                                           __m512d *minima_f64x8) {
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

/** One directed Chamfer and Hausdorff pass between f16 packs, 16 inner points per step, the last
 *  step of a count off the multiple with an infinite x in its padded lanes. */
NUMKONG_INLINE void nk_shapes_nearest_f16_skylake_(                            //
    nk_f16_t const *outer_axes, nk_size_t outer_stride, nk_size_t outer_count, //
    nk_f16_t const *inner_axes, nk_size_t inner_stride, nk_size_t inner_count, //
    nk_f32_t *sums, nk_f32_t *maxima) {
    __m512 const infinity_f32x16 = _mm512_set1_ps(NUMKONG_F32_INF);
    nk_size_t const sixteens_end = inner_count & ~(nk_size_t)15;
    __mmask16 const tail_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(inner_count - sixteens_end));
    __m256 sums_f32x8 = _mm256_loadu_ps(sums), maxima_f32x8 = _mm256_loadu_ps(maxima);
    nk_b512_vec_t inner_x_vec, inner_y_vec, inner_z_vec;
    for (nk_size_t outer_index = 0; outer_index != outer_count; ++outer_index) {
        nk_f16_t const *outer_point = outer_axes + outer_index;
        __m512 const outer_x_f32x16 = _mm512_cvtph_ps(_mm256_set1_epi16(*(nk_i16_t const *)outer_point));
        __m512 const outer_y_f32x16 = _mm512_cvtph_ps(
            _mm256_set1_epi16(*(nk_i16_t const *)(outer_point + outer_stride)));
        __m512 const outer_z_f32x16 = _mm512_cvtph_ps(
            _mm256_set1_epi16(*(nk_i16_t const *)(outer_point + 2 * outer_stride)));
        __m512 minima_f32x16[8];
        for (int flip = 0; flip != 8; ++flip) minima_f32x16[flip] = infinity_f32x16;
        for (nk_size_t inner_index = 0; inner_index != sixteens_end; inner_index += 16) {
            nk_load_f16x16_to_f32x16_skylake_(inner_axes + inner_index, &inner_x_vec);
            nk_load_f16x16_to_f32x16_skylake_(inner_axes + inner_stride + inner_index, &inner_y_vec);
            nk_load_f16x16_to_f32x16_skylake_(inner_axes + 2 * inner_stride + inner_index, &inner_z_vec);
            nk_shapes_lower_nearest_f32x16_skylake_(outer_x_f32x16, outer_y_f32x16, outer_z_f32x16, inner_x_vec.zmm_ps,
                                                    inner_y_vec.zmm_ps, inner_z_vec.zmm_ps, minima_f32x16);
        }
        if (sixteens_end != inner_count) {
            nk_load_f16x16_to_f32x16_skylake_(inner_axes + inner_stride + sixteens_end, &inner_y_vec);
            nk_load_f16x16_to_f32x16_skylake_(inner_axes + 2 * inner_stride + sixteens_end, &inner_z_vec);
            __m512 const inner_x_f32x16 = _mm512_mask_cvtph_ps(
                infinity_f32x16, tail_m16, _mm256_loadu_si256((__m256i const *)(inner_axes + sixteens_end)));
            nk_shapes_lower_nearest_f32x16_skylake_(outer_x_f32x16, outer_y_f32x16, outer_z_f32x16, inner_x_f32x16,
                                                    inner_y_vec.zmm_ps, inner_z_vec.zmm_ps, minima_f32x16);
        }
        __m256 const distances_f32x8 = _mm256_sqrt_ps(nk_shapes_flip_minima_f32x16_skylake_(minima_f32x16));
        sums_f32x8 = _mm256_add_ps(sums_f32x8, distances_f32x8);
        maxima_f32x8 = _mm256_max_ps(maxima_f32x8, distances_f32x8);
    }
    _mm256_storeu_ps(sums, sums_f32x8);
    _mm256_storeu_ps(maxima, maxima_f32x8);
}

/** One directed Chamfer and Hausdorff pass between f32 packs, 8 inner points per step in f64, the
 *  last step of a count off the multiple with an infinite x in its padded lanes. */
NUMKONG_INLINE void nk_shapes_nearest_f32_skylake_(                            //
    nk_f32_t const *outer_axes, nk_size_t outer_stride, nk_size_t outer_count, //
    nk_f32_t const *inner_axes, nk_size_t inner_stride, nk_size_t inner_count, //
    nk_f64_t *sums, nk_f64_t *maxima) {
    __m512d const infinity_f64x8 = _mm512_set1_pd(NUMKONG_F64_INF);
    nk_size_t const octuples_end = inner_count & ~(nk_size_t)7;
    __mmask8 const tail_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)(inner_count - octuples_end));
    __m512d sums_f64x8 = _mm512_loadu_pd(sums), maxima_f64x8 = _mm512_loadu_pd(maxima);
    for (nk_size_t outer_index = 0; outer_index != outer_count; ++outer_index) {
        nk_f32_t const *outer_point = outer_axes + outer_index;
        __m512d const outer_x_f64x8 = _mm512_set1_pd(outer_point[0]);
        __m512d const outer_y_f64x8 = _mm512_set1_pd(outer_point[outer_stride]);
        __m512d const outer_z_f64x8 = _mm512_set1_pd(outer_point[2 * outer_stride]);
        __m512d minima_f64x8[8];
        for (int flip = 0; flip != 8; ++flip) minima_f64x8[flip] = infinity_f64x8;
        for (nk_size_t inner_index = 0; inner_index != octuples_end; inner_index += 8)
            nk_shapes_lower_nearest_f64x8_skylake_(
                outer_x_f64x8, outer_y_f64x8, outer_z_f64x8, _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + inner_index)),
                _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + inner_stride + inner_index)),
                _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + 2 * inner_stride + inner_index)), minima_f64x8);
        if (octuples_end != inner_count)
            nk_shapes_lower_nearest_f64x8_skylake_(
                outer_x_f64x8, outer_y_f64x8, outer_z_f64x8,
                _mm512_mask_cvtps_pd(infinity_f64x8, tail_m8, _mm256_loadu_ps(inner_axes + octuples_end)),
                _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + inner_stride + octuples_end)),
                _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + 2 * inner_stride + octuples_end)), minima_f64x8);
        __m512d const distances_f64x8 = _mm512_sqrt_pd(nk_shapes_flip_minima_f64x8_skylake_(minima_f64x8));
        sums_f64x8 = _mm512_add_pd(sums_f64x8, distances_f64x8);
        maxima_f64x8 = _mm512_max_pd(maxima_f64x8, distances_f64x8);
    }
    _mm512_storeu_pd(sums, sums_f64x8);
    _mm512_storeu_pd(maxima, maxima_f64x8);
}

/** One directed Chamfer and Hausdorff pass between f64 packs, compensating each flip's sum of
 *  roots, 8 inner points per step, the last step of a count off the multiple with an infinite x in
 *  its padded lanes. */
NUMKONG_INLINE void nk_shapes_nearest_f64_skylake_(                            //
    nk_f64_t const *outer_axes, nk_size_t outer_stride, nk_size_t outer_count, //
    nk_f64_t const *inner_axes, nk_size_t inner_stride, nk_size_t inner_count, //
    nk_f64_t *sums, nk_f64_t *maxima) {
    __m512d const infinity_f64x8 = _mm512_set1_pd(NUMKONG_F64_INF), one_f64x8 = _mm512_set1_pd(1);
    nk_size_t const octuples_end = inner_count & ~(nk_size_t)7;
    __mmask8 const tail_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)(inner_count - octuples_end));
    __m512d sums_f64x8 = _mm512_loadu_pd(sums), compensations_f64x8 = _mm512_setzero_pd();
    __m512d maxima_f64x8 = _mm512_loadu_pd(maxima);
    for (nk_size_t outer_index = 0; outer_index != outer_count; ++outer_index) {
        nk_f64_t const *outer_point = outer_axes + outer_index;
        __m512d const outer_x_f64x8 = _mm512_set1_pd(outer_point[0]);
        __m512d const outer_y_f64x8 = _mm512_set1_pd(outer_point[outer_stride]);
        __m512d const outer_z_f64x8 = _mm512_set1_pd(outer_point[2 * outer_stride]);
        __m512d minima_f64x8[8];
        for (int flip = 0; flip != 8; ++flip) minima_f64x8[flip] = infinity_f64x8;
        for (nk_size_t inner_index = 0; inner_index != octuples_end; inner_index += 8)
            nk_shapes_lower_nearest_f64x8_skylake_(
                outer_x_f64x8, outer_y_f64x8, outer_z_f64x8, _mm512_loadu_pd(inner_axes + inner_index),
                _mm512_loadu_pd(inner_axes + inner_stride + inner_index),
                _mm512_loadu_pd(inner_axes + 2 * inner_stride + inner_index), minima_f64x8);
        if (octuples_end != inner_count)
            nk_shapes_lower_nearest_f64x8_skylake_(
                outer_x_f64x8, outer_y_f64x8, outer_z_f64x8,
                _mm512_mask_loadu_pd(infinity_f64x8, tail_m8, inner_axes + octuples_end),
                _mm512_loadu_pd(inner_axes + inner_stride + octuples_end),
                _mm512_loadu_pd(inner_axes + 2 * inner_stride + octuples_end), minima_f64x8);
        __m512d const distances_f64x8 = _mm512_sqrt_pd(nk_shapes_flip_minima_f64x8_skylake_(minima_f64x8));
        nk_dot2_f64x8_skylake_(&sums_f64x8, &compensations_f64x8, distances_f64x8, one_f64x8);
        maxima_f64x8 = _mm512_max_pd(maxima_f64x8, distances_f64x8);
    }
    _mm512_storeu_pd(sums, _mm512_add_pd(sums_f64x8, compensations_f64x8));
    _mm512_storeu_pd(maxima, maxima_f64x8);
}

/** The block pricing of one math type over @p vector_type lanes of 8 columns: squared distances
 *  from a broadcast row point in fused multiply-adds; the scalar cost behind the potentials, run
 *  on the same lanes so tree arcs keep a zero reduced cost; the reduced costs (c − πᵢ) + πⱼ of a
 *  row's @p valid_m8 arcs from @p column, with matrix costs −G taken as πⱼ − (G + πᵢ), which
 *  rounds the same; and the scan of @p count of them from the cursor in row-major order, keeping
 *  the first most negative one below @p entering_cost as the serial scan does. A first pass takes
 *  their least 8 columns at a time, and only a block whose least beats @p entering_cost is scanned
 *  again for the first arc reaching it. In both passes, masks keep the lanes past the scanned part
 *  of a row out of the comparisons. @p widen turns the costs into f64 lanes, empty for costs
 *  already in f64. */
#define nk_define_shapes_pricing_skylake_(math_type, vector_type, vector_suffix, broadcast, masked_load, subtract,    \
                                          multiply, multiply_add, first_lane, widen)                                  \
    NUMKONG_INLINE vector_type nk_shapes_point_costs_##vector_suffix##_skylake_(                                      \
        vector_type row_x_##vector_suffix, vector_type row_y_##vector_suffix, vector_type row_z_##vector_suffix,      \
        vector_type column_x_##vector_suffix, vector_type column_y_##vector_suffix,                                   \
        vector_type column_z_##vector_suffix) {                                                                       \
        vector_type const delta_x_##vector_suffix = subtract(row_x_##vector_suffix, column_x_##vector_suffix);        \
        vector_type const delta_y_##vector_suffix = subtract(row_y_##vector_suffix, column_y_##vector_suffix);        \
        vector_type const delta_z_##vector_suffix = subtract(row_z_##vector_suffix, column_z_##vector_suffix);        \
        return multiply_add(delta_z_##vector_suffix, delta_z_##vector_suffix,                                         \
                            multiply_add(delta_y_##vector_suffix, delta_y_##vector_suffix,                            \
                                         multiply(delta_x_##vector_suffix, delta_x_##vector_suffix)));                \
    }                                                                                                                 \
    NUMKONG_INLINE nk_##math_type##_t nk_shapes_point_cost_##math_type##_skylake_(                                    \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_size_t row, nk_size_t column) {                   \
        nk_size_t const stride = views->points_stride;                                                                \
        nk_##math_type##_t const *row_points = views->row_points, *rotated_points = views->rotated_points;            \
        return first_lane(nk_shapes_point_costs_##vector_suffix##_skylake_(                                           \
            broadcast(row_points[row]), broadcast(row_points[stride + row]), broadcast(row_points[2 * stride + row]), \
            broadcast(rotated_points[column]), broadcast(rotated_points[stride + column]),                            \
            broadcast(rotated_points[2 * stride + column])));                                                         \
    }                                                                                                                 \
    NUMKONG_INLINE __m512d nk_shapes_reduced_costs_##math_type##_skylake_(                                            \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_f64_t const *gradient, nk_size_t rows,            \
        nk_size_t columns, nk_size_t row, nk_size_t column, __mmask8 valid_m8) {                                      \
        nk_size_t const stride = views->points_stride;                                                                \
        nk_##math_type##_t const *row_points = views->row_points, *rotated_points = views->rotated_points + column;   \
        __m512d const row_potential_f64x8 = _mm512_set1_pd(views->potential[row]);                                    \
        __m512d const column_potentials_f64x8 = _mm512_maskz_loadu_pd(valid_m8, views->potential + rows + column);    \
        if (gradient)                                                                                                 \
            return _mm512_sub_pd(column_potentials_f64x8,                                                             \
                                 _mm512_add_pd(_mm512_maskz_loadu_pd(valid_m8, gradient + row * columns + column),    \
                                               row_potential_f64x8));                                                 \
        __m512d const costs_f64x8 = widen(nk_shapes_point_costs_##vector_suffix##_skylake_(                           \
            broadcast(row_points[row]), broadcast(row_points[stride + row]), broadcast(row_points[2 * stride + row]), \
            masked_load(valid_m8, rotated_points), masked_load(valid_m8, rotated_points + stride),                    \
            masked_load(valid_m8, rotated_points + 2 * stride)));                                                     \
        return _mm512_add_pd(_mm512_sub_pd(costs_f64x8, row_potential_f64x8), column_potentials_f64x8);               \
    }                                                                                                                 \
    NUMKONG_INLINE void nk_shapes_price_block_##math_type##_skylake_(                                                 \
        nk_shapes_workspace_##math_type##_serial_t const *views, nk_f64_t const *gradient, nk_size_t rows,            \
        nk_size_t columns, nk_size_t count, nk_size_t *cursor_row, nk_size_t *cursor_column, nk_f64_t *entering_cost, \
        nk_size_t *entering_row, nk_size_t *entering_column) {                                                        \
        nk_size_t const start_row = *cursor_row, start_column = *cursor_column;                                       \
        nk_size_t row = start_row, column = start_column;                                                             \
        __m512d least_f64x8 = _mm512_set1_pd(NUMKONG_F64_INF);                                                        \
        for (nk_size_t offset = 0; offset != count;) {                                                                \
            nk_size_t const segment = columns - column < count - offset ? columns - column : count - offset;          \
            for (nk_size_t step = 0; step < segment; step += 8) {                                                     \
                nk_size_t const remaining = segment - step;                                                           \
                __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, remaining < 8 ? (unsigned int)remaining : 8);     \
                __m512d const reduced_f64x8 = nk_shapes_reduced_costs_##math_type##_skylake_(                         \
                    views, gradient, rows, columns, row, column + step, valid_m8);                                    \
                least_f64x8 = _mm512_mask_min_pd(least_f64x8, valid_m8, least_f64x8, reduced_f64x8);                  \
            }                                                                                                         \
            offset += segment, column += segment;                                                                     \
            if (column == columns) column = 0, row = row + 1 == rows ? 0 : row + 1;                                   \
        }                                                                                                             \
        *cursor_row = row, *cursor_column = column;                                                                   \
        nk_f64_t const least = nk_reduce_min_f64x8_skylake_(least_f64x8);                                             \
        if (!(least < *entering_cost)) return;                                                                        \
        __m512d const least_broadcast_f64x8 = _mm512_set1_pd(least);                                                  \
        *entering_cost = least, row = start_row, column = start_column;                                               \
        for (;;) {                                                                                                    \
            nk_size_t const segment = columns - column;                                                               \
            for (nk_size_t step = 0; step < segment; step += 8) {                                                     \
                nk_size_t const remaining = segment - step;                                                           \
                __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, remaining < 8 ? (unsigned int)remaining : 8);     \
                __m512d const reduced_f64x8 = nk_shapes_reduced_costs_##math_type##_skylake_(                         \
                    views, gradient, rows, columns, row, column + step, valid_m8);                                    \
                __mmask8 const equal_m8 = _mm512_mask_cmp_pd_mask(valid_m8, reduced_f64x8, least_broadcast_f64x8,     \
                                                                  _CMP_EQ_OQ);                                        \
                if (!equal_m8) continue;                                                                              \
                *entering_row = row, *entering_column = column + step + _tzcnt_u32(equal_m8);                         \
                return;                                                                                               \
            }                                                                                                         \
            column = 0, row = row + 1 == rows ? 0 : row + 1;                                                          \
        }                                                                                                             \
    }

nk_define_shapes_pricing_skylake_(f64, __m512d, f64x8, _mm512_set1_pd, _mm512_maskz_loadu_pd, _mm512_sub_pd,
                                  _mm512_mul_pd, _mm512_fmadd_pd, _mm512_cvtsd_f64, )
nk_define_shapes_pricing_skylake_(f32, __m256, f32x8, _mm256_set1_ps, _mm256_maskz_loadu_ps, _mm256_sub_ps,
                                  _mm256_mul_ps, _mm256_fmadd_ps, _mm256_cvtss_f32, _mm512_cvtps_pd)
#undef nk_define_shapes_pricing_skylake_

nk_define_shapes_network_simplex_(f32, skylake, nk_shapes_point_cost_f32_skylake_, nk_shapes_price_block_f32_skylake_)
nk_define_shapes_network_simplex_(f64, skylake, nk_shapes_point_cost_f64_skylake_, nk_shapes_price_block_f64_skylake_)
nk_define_shapes_wasserstein_procrustes_(f16, f32, skylake, NUMKONG_F32_INF)
nk_define_shapes_wasserstein_procrustes_(f32, f64, skylake, NUMKONG_F64_INF)
nk_define_shapes_gromov_wasserstein_(f16, f32, skylake)
nk_define_shapes_gromov_wasserstein_(f32, f64, skylake)

/** Wasserstein-Procrustes of one canonically ordered pair of @c f64 clouds, like the serial one
 *  around the block pricing, the plan cost rounding as the priced lanes do. */
NUMKONG_INLINE nk_f64_t nk_shapes_wasserstein_procrustes_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views,
                                                                      nk_size_t rows, nk_size_t columns,
                                                                      nk_f64_t cost_scale, nk_f64_t *rotation) {
    nk_size_t const stride = views->points_stride;
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f64_t const mass = (nk_f64_t)(rows * columns);
    nk_f64_t best = NUMKONG_F64_INF, best_proper = NUMKONG_F64_INF, best_improper = NUMKONG_F64_INF;
    for (int flip = 0; flip != 8; ++flip) {
        nk_f64_t current[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, start_rotation[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        current[0] = (flip & 1) ? -1 : 1, current[4] = (flip & 2) ? -1 : 1, current[8] = (flip & 4) ? -1 : 1;
        nk_f64_t const flip_determinant = current[0] * current[4] * current[8];
        nk_u32_t *parity_plan = flip_determinant > 0 ? views->proper_plan : views->improper_plan;
        nk_f64_t *parity_best = flip_determinant > 0 ? &best_proper : &best_improper;
        nk_f64_t start_best = NUMKONG_F64_INF;
        nk_shapes_northwest_corner_serial_(rows, columns, views->tree, views->tree_stride);
        for (;;) {
            for (nk_size_t column = 0; column != columns; ++column) {
                nk_f64_t const x = views->column_points[column], y = views->column_points[stride + column];
                nk_f64_t const z = views->column_points[2 * stride + column];
                for (int axis = 0; axis != 3; ++axis)
                    views->rotated_points[axis * stride + column] = current[axis * 3] * x + current[axis * 3 + 1] * y +
                                                                    current[axis * 3 + 2] * z;
            }
            nk_shapes_network_simplex_f64_skylake_(views, NUMKONG_NULL, rows, columns, cost_scale);
            nk_f64_t cost = 0, cost_compensation = 0;
            for (nk_size_t node = 1; node != rows + columns; ++node) {
                nk_size_t const row = node < rows ? node : parent[node];
                nk_size_t const column = (node < rows ? parent[node] : node) - rows;
                nk_dot2_f64_serial_(&cost, &cost_compensation, (nk_f64_t)flow[node],
                                    nk_shapes_point_cost_f64_skylake_(views, row, column));
            }
            cost = (cost + cost_compensation) / mass;
            if (!(cost < start_best)) break;
            start_best = cost;
            for (int cell = 0; cell != 9; ++cell) start_rotation[cell] = current[cell];
            if (cost < *parity_best) {
                *parity_best = cost;
                nk_shapes_tree_arcs_serial_(rows, columns, views->tree, views->tree_stride, parity_plan,
                                            views->plan_stride);
            }
            nk_f64_t sums[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, compensations[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            for (nk_size_t node = 1; node != rows + columns; ++node) {
                if (!flow[node]) continue;
                nk_size_t const row = node < rows ? node : parent[node];
                nk_size_t const column = (node < rows ? parent[node] : node) - rows;
                nk_f64_t const weight = (nk_f64_t)flow[node] / mass;
                for (int column_axis = 0; column_axis != 3; ++column_axis) {
                    nk_f64_t const weighted = weight * views->column_points[column_axis * stride + column];
                    for (int row_axis = 0; row_axis != 3; ++row_axis)
                        nk_dot2_f64_serial_(&sums[column_axis * 3 + row_axis],
                                            &compensations[column_axis * 3 + row_axis], weighted,
                                            views->row_points[row_axis * stride + row]);
                }
            }
            nk_f64_t cross_covariance[9], svd_left[9], svd_diagonal[9], svd_right[9];
            for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = sums[cell] + compensations[cell];
            nk_svd3x3_f64_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);
            nk_rotation_from_svd_f64_serial_(svd_left, svd_right, current);
            // Each start stays among the rotations, or among the reflections, it began with
            if (nk_det3x3_f64_serial_(current) * flip_determinant < 0) {
                svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
                nk_rotation_from_svd_f64_serial_(svd_left, svd_right, current);
            }
        }
        if (start_best < best) {
            best = start_best;
            for (int cell = 0; cell != 9; ++cell) rotation[cell] = start_rotation[cell];
        }
    }
    return best;
}

/** Dx × V × Dy with each cell's sum over the arcs compensated, 8 columns per step through a masked
 *  tail, for @c f64 clouds. */
NUMKONG_INLINE void nk_shapes_distance_gradient_compensated_f64_skylake_(nk_u32_t const *plan, nk_size_t plan_stride,
                                                                         nk_size_t arcs, nk_f64_t mass,
                                                                         nk_f64_t const *row_distances, nk_size_t rows,
                                                                         nk_f64_t const *column_distances,
                                                                         nk_size_t columns, nk_f64_t *gradient) {
    for (nk_size_t row = 0; row != rows; ++row)
        for (nk_size_t column = 0; column < columns; column += 8) {
            nk_size_t const remaining = columns - column;
            __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, remaining < 8 ? (unsigned int)remaining : 8);
            __m512d sum_f64x8 = _mm512_setzero_pd(), compensation_f64x8 = _mm512_setzero_pd();
            for (nk_size_t arc = 0; arc != arcs; ++arc) {
                if (!plan[2 * plan_stride + arc]) continue;
                nk_f64_t const weighted = (nk_f64_t)plan[2 * plan_stride + arc] / mass *
                                          row_distances[plan[arc] * rows + row];
                nk_dot2_f64x8_skylake_(
                    &sum_f64x8, &compensation_f64x8, _mm512_set1_pd(weighted),
                    _mm512_maskz_loadu_pd(valid_m8, column_distances + plan[plan_stride + arc] * columns + column));
            }
            _mm512_mask_storeu_pd(gradient + row * columns + column, valid_m8,
                                  _mm512_add_pd(sum_f64x8, compensation_f64x8));
        }
}

/** Gromov-Wasserstein of one canonically ordered pair of @c f64 clouds, like the serial one
 *  around the block pricing and the gradients on fused multiply-add lanes. */
NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views,
                                                                  nk_size_t rows, nk_size_t columns,
                                                                  nk_f64_t cost_scale, nk_f64_t const *row_distances,
                                                                  nk_f64_t const *column_distances,
                                                                  nk_f64_t *rotation) {
    nk_size_t const cells = rows * columns, arcs = rows + columns - 1, stride = views->points_stride;
    nk_size_t const plan_stride = views->plan_stride;
    nk_f64_t *plan = views->matrices, *gradient = views->matrices + views->matrix_stride;
    nk_f64_t *vertex_gradient = views->matrices + 2 * views->matrix_stride;
    nk_f64_t const mass = (nk_f64_t)cells;
    nk_shapes_wasserstein_procrustes_f64_skylake_(views, rows, columns, cost_scale, rotation);
    nk_f64_t row_spread = 0, row_spread_compensation = 0, column_spread = 0, column_spread_compensation = 0;
    for (nk_size_t cell = 0; cell != rows * rows; ++cell)
        nk_dot2_f64_serial_(&row_spread, &row_spread_compensation, row_distances[cell], row_distances[cell]);
    for (nk_size_t cell = 0; cell != columns * columns; ++cell)
        nk_dot2_f64_serial_(&column_spread, &column_spread_compensation, column_distances[cell],
                            column_distances[cell]);
    row_spread = (row_spread + row_spread_compensation) / (nk_f64_t)(rows * rows);
    column_spread = (column_spread + column_spread_compensation) / (nk_f64_t)(columns * columns);
    // Once its start spreads into the dense plan, the proper plan holds each step's vertex
    nk_u32_t *vertex = views->proper_plan;
    nk_f64_t best = NUMKONG_F64_INF;
    for (int start = 0; start != 3; ++start) {
        if (start == 2) {
            // The product plan's gradient is the outer product of the two matrices' row sums
            for (nk_size_t column = 0; column != columns; ++column) {
                nk_f64_t column_sum = 0, column_sum_compensation = 0;
                for (nk_size_t other = 0; other != columns; ++other)
                    nk_dot2_f64_serial_(&column_sum, &column_sum_compensation,
                                        column_distances[other * columns + column], 1);
                vertex_gradient[column] = column_sum + column_sum_compensation;
            }
            for (nk_size_t row = 0; row != rows; ++row) {
                nk_f64_t row_sum = 0, row_sum_compensation = 0;
                for (nk_size_t other = 0; other != rows; ++other)
                    nk_dot2_f64_serial_(&row_sum, &row_sum_compensation, row_distances[row * rows + other], 1);
                row_sum += row_sum_compensation;
                for (nk_size_t column = 0; column != columns; ++column) {
                    plan[row * columns + column] = 1 / mass;
                    gradient[row * columns + column] = row_sum * vertex_gradient[column] / mass;
                }
            }
        }
        else {
            nk_u32_t const *start_plan = start ? views->improper_plan : views->proper_plan;
            for (nk_size_t cell = 0; cell != cells; ++cell) plan[cell] = 0;
            for (nk_size_t arc = 0; arc != arcs; ++arc)
                plan[start_plan[arc] * columns + start_plan[plan_stride + arc]] =
                    (nk_f64_t)start_plan[2 * plan_stride + arc] / mass;
            nk_shapes_distance_gradient_compensated_f64_skylake_(start_plan, plan_stride, arcs, mass, row_distances,
                                                                 rows, column_distances, columns, gradient);
        }
        nk_f64_t gradient_scale;
        nk_f64_t plan_score = nk_shapes_plan_score_compensated_f64_serial_(gradient, plan, cells, &gradient_scale);
        for (int step_index = 0; step_index != 32; ++step_index) {
            if (!step_index) nk_shapes_northwest_corner_serial_(rows, columns, views->tree, views->tree_stride);
            nk_shapes_network_simplex_f64_skylake_(views, gradient, rows, columns, gradient_scale);
            nk_shapes_tree_arcs_serial_(rows, columns, views->tree, views->tree_stride, vertex, plan_stride);
            nk_shapes_distance_gradient_compensated_f64_skylake_(vertex, plan_stride, arcs, mass, row_distances, rows,
                                                                 column_distances, columns, vertex_gradient);
            nk_f64_t vertex_score = 0, vertex_score_compensation = 0;
            nk_f64_t vertex_curvature = 0, vertex_curvature_compensation = 0;
            for (nk_size_t arc = 0; arc != arcs; ++arc) {
                nk_size_t const cell = vertex[arc] * columns + vertex[plan_stride + arc];
                nk_f64_t const weight = (nk_f64_t)vertex[2 * plan_stride + arc] / mass;
                nk_dot2_f64_serial_(&vertex_score, &vertex_score_compensation, gradient[cell], weight);
                nk_dot2_f64_serial_(&vertex_curvature, &vertex_curvature_compensation, vertex_gradient[cell], weight);
            }
            vertex_score += vertex_score_compensation, vertex_curvature += vertex_curvature_compensation;
            nk_f64_t const slope = vertex_score - plan_score;
            nk_f64_t const curvature = vertex_curvature - 2 * vertex_score + plan_score;
            nk_f64_t const quadratic = -2 * curvature, linear = -4 * slope;
            nk_f64_t step = quadratic + linear < 0 ? 1 : 0;
            if (quadratic > 0) {
                step = -linear / (2 * quadratic);
                step = step < 0 ? 0 : step > 1 ? 1 : step;
            }
            if (step == 0) break;
            nk_shapes_plan_step_f64_serial_(plan, gradient, vertex_gradient, cells, 1 - step, step);
            for (nk_size_t arc = 0; arc != arcs; ++arc)
                plan[vertex[arc] * columns + vertex[plan_stride + arc]] +=
                    step * ((nk_f64_t)vertex[2 * plan_stride + arc] / mass);
            plan_score = nk_shapes_plan_score_compensated_f64_serial_(gradient, plan, cells, &gradient_scale);
        }
        nk_f64_t const loss = row_spread + column_spread - 2 * plan_score;
        if (!(loss < best)) continue;
        best = loss;
        nk_f64_t sums[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, compensations[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (nk_size_t row = 0; row != rows; ++row)
            for (nk_size_t column = 0; column != columns; ++column)
                for (int column_axis = 0; column_axis != 3; ++column_axis) {
                    nk_f64_t const weighted = plan[row * columns + column] *
                                              views->column_points[column_axis * stride + column];
                    for (int row_axis = 0; row_axis != 3; ++row_axis)
                        nk_dot2_f64_serial_(&sums[column_axis * 3 + row_axis],
                                            &compensations[column_axis * 3 + row_axis], weighted,
                                            views->row_points[row_axis * stride + row]);
                }
        nk_f64_t cross_covariance[9], svd_left[9], svd_diagonal[9], svd_right[9];
        for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = sums[cell] + compensations[cell];
        nk_svd3x3_f64_serial_(cross_covariance, svd_left, svd_diagonal, svd_right);
        // The McAdams factors are proper, so det H's sign moves into V to fit over O(3)
        if (nk_det3x3_f64_serial_(cross_covariance) < 0)
            svd_right[2] = -svd_right[2], svd_right[5] = -svd_right[5], svd_right[8] = -svd_right[8];
        nk_rotation_from_svd_f64_serial_(svd_left, svd_right, rotation);
    }
    return best;
}

nk_define_shapes_flip_kernels_(f64, f64, skylake, nk_shapes_nearest_f64_skylake_, NUMKONG_F64_INF)
nk_define_shapes_flip_kernels_(f32, f64, skylake, nk_shapes_nearest_f32_skylake_, NUMKONG_F64_INF)
nk_define_shapes_flip_kernels_(f16, f32, skylake, nk_shapes_nearest_f16_skylake_, NUMKONG_F32_INF)
nk_define_shapes_transport_kernels_(f64, f64, skylake, NUMKONG_F64_INF)
nk_define_shapes_transport_kernels_(f32, f64, skylake, NUMKONG_F64_INF)
nk_define_shapes_transport_kernels_(f16, f32, skylake, NUMKONG_F32_INF)

/** Σ‖xᵢ′‖² of a pack of @c f16 points, 16 points per step through the zero padding. */
NUMKONG_INLINE nk_f32_t nk_rmsds_norm_f16_skylake_(void const *packed) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)packed)->points;
    nk_size_t axis_stride;
    nk_f16_t const *axes = nk_shapes_packed_axes_f16_serial_(packed, &axis_stride);
    __m512 norm_f32x16 = _mm512_setzero_ps();
    for (int axis = 0; axis != 3; ++axis)
        for (nk_size_t index = 0; index < count; index += 16) {
            __m512 const values_f32x16 = _mm512_cvtph_ps(
                _mm256_loadu_si256((__m256i const *)(axes + axis * axis_stride + index)));
            norm_f32x16 = _mm512_fmadd_ps(values_f32x16, values_f32x16, norm_f32x16);
        }
    return nk_reduce_add_f32x16_skylake_(norm_f32x16);
}

/** Superposed RMSD of one pair of @c f16 packs with Ga + Gb in @p norms_squared, M in @c f32. */
NUMKONG_INLINE nk_f32_t nk_rmsds_pair_f16_skylake_(void const *row_packed, void const *column_packed,
                                                   nk_f32_t norms_squared) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f16_t const *a_axes = nk_shapes_packed_axes_f16_serial_(row_packed, &axis_stride);
    nk_f16_t const *b_axes = nk_shapes_packed_axes_f16_serial_(column_packed, &axis_stride);
    __m512 sums_f32x16[9];
    for (int cell = 0; cell != 9; ++cell) sums_f32x16[cell] = _mm512_setzero_ps();
    for (nk_size_t index = 0; index < count; index += 16) {
        __m512 a_f32x16[3], b_f32x16[3];
        for (int axis = 0; axis != 3; ++axis) {
            a_f32x16[axis] = _mm512_cvtph_ps(
                _mm256_loadu_si256((__m256i const *)(a_axes + axis * axis_stride + index)));
            b_f32x16[axis] = _mm512_cvtph_ps(
                _mm256_loadu_si256((__m256i const *)(b_axes + axis * axis_stride + index)));
        }
        for (int row = 0; row != 3; ++row)
            for (int column = 0; column != 3; ++column)
                sums_f32x16[row * 3 + column] = _mm512_fmadd_ps(a_f32x16[row], b_f32x16[column],
                                                                sums_f32x16[row * 3 + column]);
    }
    nk_f32_t cross_covariance[9];
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = nk_reduce_add_f32x16_skylake_(sums_f32x16[cell]);
    nk_f32_t const residual = nk_rmsds_folded_f32_serial_(cross_covariance, norms_squared);
    return nk_sqrt_f32_serial_(residual / (nk_f32_t)count);
}

/** Superposed RMSD of one pair of @c f32 packs: H in @c f64, then compensated residuals, 8 points
 *  per step through the zero padding. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f32_skylake_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f32_t const *a_axes = nk_shapes_packed_axes_f32_serial_(row_packed, &axis_stride);
    nk_f32_t const *b_axes = nk_shapes_packed_axes_f32_serial_(column_packed, &axis_stride);
    __m512d sums_f64x8[9];
    for (int cell = 0; cell != 9; ++cell) sums_f64x8[cell] = _mm512_setzero_pd();
    for (nk_size_t index = 0; index < count; index += 8) {
        __m512d a_f64x8[3], b_f64x8[3];
        for (int axis = 0; axis != 3; ++axis) {
            a_f64x8[axis] = _mm512_cvtps_pd(_mm256_loadu_ps(a_axes + axis * axis_stride + index));
            b_f64x8[axis] = _mm512_cvtps_pd(_mm256_loadu_ps(b_axes + axis * axis_stride + index));
        }
        for (int column = 0; column != 3; ++column)
            for (int row = 0; row != 3; ++row)
                sums_f64x8[column * 3 + row] = _mm512_fmadd_pd(b_f64x8[column], a_f64x8[row],
                                                               sums_f64x8[column * 3 + row]);
    }
    nk_f64_t cross_covariance[9], rotation[9];
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = nk_reduce_add_f64x8_skylake_(sums_f64x8[cell]);
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    __m512d rotation_f64x8[9];
    for (int cell = 0; cell != 9; ++cell) rotation_f64x8[cell] = _mm512_set1_pd(rotation[cell]);
    __m512d sum_f64x8 = _mm512_setzero_pd(), compensation_f64x8 = _mm512_setzero_pd();
    for (nk_size_t index = 0; index < count; index += 8) {
        __m512d b_f64x8[3];
        for (int axis = 0; axis != 3; ++axis)
            b_f64x8[axis] = _mm512_cvtps_pd(_mm256_loadu_ps(b_axes + axis * axis_stride + index));
        for (int axis = 0; axis != 3; ++axis) {
            __m512d residual_f64x8 = _mm512_cvtps_pd(_mm256_loadu_ps(a_axes + axis * axis_stride + index));
            residual_f64x8 = _mm512_fnmadd_pd(rotation_f64x8[axis * 3 + 0], b_f64x8[0], residual_f64x8);
            residual_f64x8 = _mm512_fnmadd_pd(rotation_f64x8[axis * 3 + 1], b_f64x8[1], residual_f64x8);
            residual_f64x8 = _mm512_fnmadd_pd(rotation_f64x8[axis * 3 + 2], b_f64x8[2], residual_f64x8);
            nk_dot2_f64x8_skylake_(&sum_f64x8, &compensation_f64x8, residual_f64x8, residual_f64x8);
        }
    }
    nk_f64_t const residuals = nk_dot_stable_sum_f64x8_skylake_(sum_f64x8, compensation_f64x8);
    return nk_sqrt_f64_serial_(residuals / (nk_f64_t)count);
}

/** Superposed RMSD of one pair of @c f64 packs, like the @c f32 pair with H compensated too. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f64_skylake_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f64_t const *a_axes = nk_shapes_packed_axes_f64_serial_(row_packed, &axis_stride);
    nk_f64_t const *b_axes = nk_shapes_packed_axes_f64_serial_(column_packed, &axis_stride);
    __m512d sums_f64x8[9], compensations_f64x8[9];
    for (int cell = 0; cell != 9; ++cell)
        sums_f64x8[cell] = _mm512_setzero_pd(), compensations_f64x8[cell] = _mm512_setzero_pd();
    for (nk_size_t index = 0; index < count; index += 8) {
        __m512d a_f64x8[3], b_f64x8[3];
        for (int axis = 0; axis != 3; ++axis) {
            a_f64x8[axis] = _mm512_loadu_pd(a_axes + axis * axis_stride + index);
            b_f64x8[axis] = _mm512_loadu_pd(b_axes + axis * axis_stride + index);
        }
        for (int column = 0; column != 3; ++column)
            for (int row = 0; row != 3; ++row)
                nk_dot2_f64x8_skylake_(&sums_f64x8[column * 3 + row], &compensations_f64x8[column * 3 + row],
                                       b_f64x8[column], a_f64x8[row]);
    }
    nk_f64_t cross_covariance[9], rotation[9];
    for (int cell = 0; cell != 9; ++cell)
        cross_covariance[cell] = nk_dot_stable_sum_f64x8_skylake_(sums_f64x8[cell], compensations_f64x8[cell]);
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    __m512d rotation_f64x8[9];
    for (int cell = 0; cell != 9; ++cell) rotation_f64x8[cell] = _mm512_set1_pd(rotation[cell]);
    __m512d sum_f64x8 = _mm512_setzero_pd(), compensation_f64x8 = _mm512_setzero_pd();
    for (nk_size_t index = 0; index < count; index += 8) {
        __m512d b_f64x8[3];
        for (int axis = 0; axis != 3; ++axis) b_f64x8[axis] = _mm512_loadu_pd(b_axes + axis * axis_stride + index);
        for (int axis = 0; axis != 3; ++axis) {
            __m512d residual_f64x8 = _mm512_loadu_pd(a_axes + axis * axis_stride + index);
            residual_f64x8 = _mm512_fnmadd_pd(rotation_f64x8[axis * 3 + 0], b_f64x8[0], residual_f64x8);
            residual_f64x8 = _mm512_fnmadd_pd(rotation_f64x8[axis * 3 + 1], b_f64x8[1], residual_f64x8);
            residual_f64x8 = _mm512_fnmadd_pd(rotation_f64x8[axis * 3 + 2], b_f64x8[2], residual_f64x8);
            nk_dot2_f64x8_skylake_(&sum_f64x8, &compensation_f64x8, residual_f64x8, residual_f64x8);
        }
    }
    nk_f64_t const residuals = nk_dot_stable_sum_f64x8_skylake_(sum_f64x8, compensation_f64x8);
    return nk_sqrt_f64_serial_(residuals / (nk_f64_t)count);
}

NUMKONG_API nk_status_t nk_rmsds_packed_f64_skylake(void const *const *a_packed, nk_size_t rows,
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
            results_row[column] = nk_rmsds_pair_f64_skylake_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f32_skylake(void const *const *a_packed, nk_size_t rows,
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
            results_row[column] = nk_rmsds_pair_f32_skylake_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f16_skylake(void const *const *a_packed, nk_size_t rows,
                                                    void const *const *b_packed, nk_size_t columns, nk_f32_t *results,
                                                    nk_size_t results_stride, nk_stream_t stream) {
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
            row_norms[row - row_begin] = nk_rmsds_norm_f16_skylake_(a_packed[row]);
        for (nk_size_t column_begin = 0; column_begin < columns; column_begin += nk_rmsds_tile_frames_k_) {
            nk_size_t const column_end = column_begin + nk_rmsds_tile_frames_k_ < columns
                                             ? column_begin + nk_rmsds_tile_frames_k_
                                             : columns;
            for (nk_size_t column = column_begin; column != column_end; ++column)
                column_norms[column - column_begin] = nk_rmsds_norm_f16_skylake_(b_packed[column]);
            for (nk_size_t row = row_begin; row != row_end; ++row) {
                nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
                for (nk_size_t column = column_begin; column != column_end; ++column)
                    results_row[column] = nk_rmsds_pair_f16_skylake_(
                        a_packed[row], b_packed[column],
                        row_norms[row - row_begin] + column_norms[column - column_begin]);
            }
        }
    }
    return nk_success_k;
}

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
#endif // NUMKONG_SHAPES_SKYLAKE_H
