/**
 *  @file include/numkong/shapes/haswell.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief Many-to-many shape distances between 3D point clouds for Haswell, AVX2.
 *
 *  @sa c/numkong/shapes.h
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
 *  _mm256_cvtps_ph         VCVTPS2PH (XMM, YMM, I8)        5cy @ p01   4cy @ p12+p23
 *  _mm256_blendv_ps        VBLENDVPS (YMM, YMM, YMM, YMM)  2cy @ p015  1cy @ p01
 *  _mm256_shuffle_ps       VSHUFPS (YMM, YMM, YMM, I8)     1cy @ p5    1cy @ p12
 *  _mm256_permute2f128_ps  VPERM2F128 (YMM, YMM, YMM, I8)  3cy @ p5    2cy @ p12
 *  _mm256_sqrt_ps          VSQRTPS (YMM, YMM)              19cy @ p0   15cy @ p01
 *  _mm256_sqrt_pd          VSQRTPD (YMM, YMM)              35cy @ p0   21cy @ p01
 *  @endverbatim
 *
 *  Chamfer and Hausdorff broadcast one outer point against a register of inner points, which sit
 *  contiguous per axis in the pack, and keep one running minimum per axis flip. Haswell issues
 *  VADDPS and VMINPS on port 1 alone, so the squares fold into FMAs that also take port 0. The
 *  last register of a cloud takes an infinite x in its padded lanes, so they never win a minimum.
 *  Chamfer scores both directions in that one sweep, keeping the inner cloud's minima in the
 *  workspace, whose 64-byte strides take full-width stores. Hausdorff runs each direction and stops
 *  once every flip passes the limit, so it reads only the points that decide a pair.
 *
 *  Packing widens xyz triples through F16C or VCVTPS2PD, and narrows each rotated register back
 *  into its axis, measuring the stored points in the same sweep.
 *
 *  The transport kinds run the serial simplex around tile pricing four arcs at a time. Arc costs
 *  round in fused multiply-adds, and the scalar cost behind the tree's potentials runs the same
 *  lanes, so tree arcs keep a zero reduced cost.
 *
 *  RMSD sums eight @c f16 points per step in f32 lanes, or four @c f32 or @c f64 points in f64
 *  lanes, through the zero padding, and leaves the SVD scalar.
 */
#ifndef NUMKONG_SHAPES_HASWELL_H
#define NUMKONG_SHAPES_HASWELL_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_HASWELL_

#include "numkong/types.h"
#include "numkong/cast/haswell.h"   // `nk_load_f16x8_to_f32x8_haswell_`
#include "numkong/scalar/haswell.h" // `nk_rsqrt_f32_haswell_`
#include "numkong/shapes/serial.h"  // `nk_shapes_best_flip_f32_serial_`, `nk_shapes_simplex_next_tile_serial_`
#include "numkong/reduce/haswell.h" // `nk_reduce_min_f64x4_haswell_`
#include "numkong/dot/haswell.h"    // `nk_dot2_f64x4_haswell_`, `nk_dot_stable_sum_f64x4_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

#if NUMKONG_TARGET_HASWELL

NUMKONG_INLINE nk_f32_t nk_shapes_sqrt_f32_haswell_(nk_f32_t x) { return _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(x))); }
NUMKONG_INLINE nk_f64_t nk_shapes_sqrt_f64_haswell_(nk_f64_t x) { return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(x))); }

/** AVX2 estimates no double roots, so the exact one divides, as @c nk_f64_rsqrt_haswell does. */
NUMKONG_INLINE nk_f64_t nk_shapes_rsqrt_f64_haswell_(nk_f64_t x) { return 1 / nk_shapes_sqrt_f64_haswell_(x); }

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

/** Lowers the running minima of the 8 flips by one outer point's squared distances to 8 inner
 *  points, the difference and the sum of each axis standing for its two signs. */
NUMKONG_INLINE void nk_shapes_lower_nearest_f32x8_haswell_(__m256 outer_x_f32x8, __m256 outer_y_f32x8,
                                                           __m256 outer_z_f32x8, __m256 inner_x_f32x8,
                                                           __m256 inner_y_f32x8, __m256 inner_z_f32x8,
                                                           __m256 *nearest_f32x8) {
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
        nearest_f32x8[flip] = _mm256_min_ps(nearest_f32x8[flip],
                                            _mm256_fmadd_ps(minus_z_f32x8, minus_z_f32x8, planar_f32x8[flip]));
        nearest_f32x8[flip + 4] = _mm256_min_ps(nearest_f32x8[flip + 4],
                                                _mm256_fmadd_ps(plus_z_f32x8, plus_z_f32x8, planar_f32x8[flip]));
    }
}

/** @copydoc nk_shapes_lower_nearest_f32x8_haswell_ */
NUMKONG_INLINE void nk_shapes_lower_nearest_f64x4_haswell_(__m256d outer_x_f64x4, __m256d outer_y_f64x4,
                                                           __m256d outer_z_f64x4, __m256d inner_x_f64x4,
                                                           __m256d inner_y_f64x4, __m256d inner_z_f64x4,
                                                           __m256d *nearest_f64x4) {
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
        nearest_f64x4[flip] = _mm256_min_pd(nearest_f64x4[flip],
                                            _mm256_fmadd_pd(minus_z_f64x4, minus_z_f64x4, planar_f64x4[flip]));
        nearest_f64x4[flip + 4] = _mm256_min_pd(nearest_f64x4[flip + 4],
                                                _mm256_fmadd_pd(plus_z_f64x4, plus_z_f64x4, planar_f64x4[flip]));
    }
}

/** Lowers the 8 flips' nearest squares as @ref nk_shapes_lower_nearest_f32x8_haswell_ does, and the
 *  same squares into the inner points' @p reverse_minima , 8 axes @p reverse_stride apart. */
NUMKONG_INLINE void nk_shapes_lower_mutual_f32x8_haswell_(__m256 outer_x_f32x8, __m256 outer_y_f32x8,
                                                          __m256 outer_z_f32x8, __m256 inner_x_f32x8,
                                                          __m256 inner_y_f32x8, __m256 inner_z_f32x8,
                                                          __m256 *nearest_f32x8, nk_f32_t *reverse_minima,
                                                          nk_size_t reverse_stride) {
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
        __m256 const minus_squares_f32x8 = _mm256_fmadd_ps(minus_z_f32x8, minus_z_f32x8, planar_f32x8[flip]);
        __m256 const plus_squares_f32x8 = _mm256_fmadd_ps(plus_z_f32x8, plus_z_f32x8, planar_f32x8[flip]);
        nk_f32_t *minus_reverse = reverse_minima + flip * reverse_stride;
        nk_f32_t *plus_reverse = minus_reverse + 4 * reverse_stride;
        nearest_f32x8[flip] = _mm256_min_ps(nearest_f32x8[flip], minus_squares_f32x8);
        nearest_f32x8[flip + 4] = _mm256_min_ps(nearest_f32x8[flip + 4], plus_squares_f32x8);
        _mm256_storeu_ps(minus_reverse, _mm256_min_ps(minus_squares_f32x8, _mm256_loadu_ps(minus_reverse)));
        _mm256_storeu_ps(plus_reverse, _mm256_min_ps(plus_squares_f32x8, _mm256_loadu_ps(plus_reverse)));
    }
}

/** @copydoc nk_shapes_lower_mutual_f32x8_haswell_ */
NUMKONG_INLINE void nk_shapes_lower_mutual_f64x4_haswell_(__m256d outer_x_f64x4, __m256d outer_y_f64x4,
                                                          __m256d outer_z_f64x4, __m256d inner_x_f64x4,
                                                          __m256d inner_y_f64x4, __m256d inner_z_f64x4,
                                                          __m256d *nearest_f64x4, nk_f64_t *reverse_minima,
                                                          nk_size_t reverse_stride) {
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
        __m256d const minus_squares_f64x4 = _mm256_fmadd_pd(minus_z_f64x4, minus_z_f64x4, planar_f64x4[flip]);
        __m256d const plus_squares_f64x4 = _mm256_fmadd_pd(plus_z_f64x4, plus_z_f64x4, planar_f64x4[flip]);
        nk_f64_t *minus_reverse = reverse_minima + flip * reverse_stride;
        nk_f64_t *plus_reverse = minus_reverse + 4 * reverse_stride;
        nearest_f64x4[flip] = _mm256_min_pd(nearest_f64x4[flip], minus_squares_f64x4);
        nearest_f64x4[flip + 4] = _mm256_min_pd(nearest_f64x4[flip + 4], plus_squares_f64x4);
        _mm256_storeu_pd(minus_reverse, _mm256_min_pd(minus_squares_f64x4, _mm256_loadu_pd(minus_reverse)));
        _mm256_storeu_pd(plus_reverse, _mm256_min_pd(plus_squares_f64x4, _mm256_loadu_pd(plus_reverse)));
    }
}

/** All bits set in the first @p remaining of 8 lanes, to blend or clear the lanes past a cloud. */
NUMKONG_INLINE __m256 nk_shapes_valid_lanes_f32x8_haswell_(nk_size_t remaining) {
    __m256i const remaining_i32x8 = _mm256_set1_epi32(remaining < 8 ? (int)remaining : 8);
    return _mm256_castsi256_ps(_mm256_cmpgt_epi32(remaining_i32x8, _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
}

/** All bits set in the first @p remaining of 4 lanes, to blend or clear the lanes past a row. */
NUMKONG_INLINE __m256d nk_shapes_valid_lanes_f64x4_haswell_(nk_size_t remaining) {
    __m256i const remaining_i64x4 = _mm256_set1_epi64x(remaining < 4 ? (long long)remaining : 4);
    return _mm256_castsi256_pd(_mm256_cmpgt_epi64(remaining_i64x4, _mm256_setr_epi64x(0, 1, 2, 3)));
}

/** One outer point's sweep of @c f16 inner points in f32 lanes, 8 at a time, the last octuple of a
 *  count off the multiple with an infinite x in the padded lanes @p tail_f32x8 clears. */
NUMKONG_INLINE void nk_shapes_sweep_nearest_f16_haswell_(__m256 outer_x_f32x8, __m256 outer_y_f32x8,
                                                         __m256 outer_z_f32x8, nk_f16_t const *inner_axes,
                                                         nk_size_t inner_stride, nk_size_t inner_count,
                                                         __m256 tail_f32x8, __m256 *nearest_f32x8) {
    nk_size_t const octuples_end = inner_count & ~(nk_size_t)7;
    nk_b256_vec_t inner_x_vec, inner_y_vec, inner_z_vec;
    for (nk_size_t inner_index = 0; inner_index != octuples_end; inner_index += 8) {
        nk_load_f16x8_to_f32x8_haswell_(inner_axes + inner_index, &inner_x_vec);
        nk_load_f16x8_to_f32x8_haswell_(inner_axes + inner_stride + inner_index, &inner_y_vec);
        nk_load_f16x8_to_f32x8_haswell_(inner_axes + 2 * inner_stride + inner_index, &inner_z_vec);
        nk_shapes_lower_nearest_f32x8_haswell_(outer_x_f32x8, outer_y_f32x8, outer_z_f32x8, inner_x_vec.ymm_ps,
                                               inner_y_vec.ymm_ps, inner_z_vec.ymm_ps, nearest_f32x8);
    }
    if (octuples_end == inner_count) return;
    nk_load_f16x8_to_f32x8_haswell_(inner_axes + octuples_end, &inner_x_vec);
    nk_load_f16x8_to_f32x8_haswell_(inner_axes + inner_stride + octuples_end, &inner_y_vec);
    nk_load_f16x8_to_f32x8_haswell_(inner_axes + 2 * inner_stride + octuples_end, &inner_z_vec);
    nk_shapes_lower_nearest_f32x8_haswell_(
        outer_x_f32x8, outer_y_f32x8, outer_z_f32x8,
        _mm256_blendv_ps(_mm256_set1_ps(NUMKONG_F32_INF), inner_x_vec.ymm_ps, tail_f32x8), inner_y_vec.ymm_ps,
        inner_z_vec.ymm_ps, nearest_f32x8);
}

/** @ref nk_shapes_sweep_nearest_f16_haswell_ that also lowers the views' reverse minima. */
NUMKONG_INLINE void nk_shapes_sweep_mutual_f16_haswell_(__m256 outer_x_f32x8, __m256 outer_y_f32x8,
                                                        __m256 outer_z_f32x8, nk_shapes_mutual_f16_t const *views,
                                                        __m256 tail_f32x8, __m256 *nearest_f32x8) {
    nk_f16_t const *inner_axes = views->inner_axes;
    nk_size_t const inner_stride = views->inner_stride, inner_count = views->inner_count;
    nk_size_t const octuples_end = inner_count & ~(nk_size_t)7;
    nk_b256_vec_t inner_x_vec, inner_y_vec, inner_z_vec;
    for (nk_size_t inner_index = 0; inner_index != octuples_end; inner_index += 8) {
        nk_load_f16x8_to_f32x8_haswell_(inner_axes + inner_index, &inner_x_vec);
        nk_load_f16x8_to_f32x8_haswell_(inner_axes + inner_stride + inner_index, &inner_y_vec);
        nk_load_f16x8_to_f32x8_haswell_(inner_axes + 2 * inner_stride + inner_index, &inner_z_vec);
        nk_shapes_lower_mutual_f32x8_haswell_(outer_x_f32x8, outer_y_f32x8, outer_z_f32x8, inner_x_vec.ymm_ps,
                                              inner_y_vec.ymm_ps, inner_z_vec.ymm_ps, nearest_f32x8,
                                              views->reverse_minima + inner_index, views->reverse_stride);
    }
    if (octuples_end == inner_count) return;
    nk_load_f16x8_to_f32x8_haswell_(inner_axes + octuples_end, &inner_x_vec);
    nk_load_f16x8_to_f32x8_haswell_(inner_axes + inner_stride + octuples_end, &inner_y_vec);
    nk_load_f16x8_to_f32x8_haswell_(inner_axes + 2 * inner_stride + octuples_end, &inner_z_vec);
    nk_shapes_lower_mutual_f32x8_haswell_(
        outer_x_f32x8, outer_y_f32x8, outer_z_f32x8,
        _mm256_blendv_ps(_mm256_set1_ps(NUMKONG_F32_INF), inner_x_vec.ymm_ps, tail_f32x8), inner_y_vec.ymm_ps,
        inner_z_vec.ymm_ps, nearest_f32x8, views->reverse_minima + octuples_end, views->reverse_stride);
}

/** One outer point's sweep of @c f32 inner points in f64 lanes, 4 at a time, the last quadruple
 *  of a count off the multiple with an infinite x in the padded lanes @p tail_f64x4 clears. */
NUMKONG_INLINE void nk_shapes_sweep_nearest_f32_haswell_(__m256d outer_x_f64x4, __m256d outer_y_f64x4,
                                                         __m256d outer_z_f64x4, nk_f32_t const *inner_axes,
                                                         nk_size_t inner_stride, nk_size_t inner_count,
                                                         __m256d tail_f64x4, __m256d *nearest_f64x4) {
    nk_size_t const quadruples_end = inner_count & ~(nk_size_t)3;
    for (nk_size_t inner_index = 0; inner_index != quadruples_end; inner_index += 4)
        nk_shapes_lower_nearest_f64x4_haswell_(
            outer_x_f64x4, outer_y_f64x4, outer_z_f64x4, _mm256_cvtps_pd(_mm_loadu_ps(inner_axes + inner_index)),
            _mm256_cvtps_pd(_mm_loadu_ps(inner_axes + inner_stride + inner_index)),
            _mm256_cvtps_pd(_mm_loadu_ps(inner_axes + 2 * inner_stride + inner_index)), nearest_f64x4);
    if (quadruples_end == inner_count) return;
    nk_shapes_lower_nearest_f64x4_haswell_(
        outer_x_f64x4, outer_y_f64x4, outer_z_f64x4,
        _mm256_blendv_pd(_mm256_set1_pd(NUMKONG_F64_INF), _mm256_cvtps_pd(_mm_loadu_ps(inner_axes + quadruples_end)),
                         tail_f64x4),
        _mm256_cvtps_pd(_mm_loadu_ps(inner_axes + inner_stride + quadruples_end)),
        _mm256_cvtps_pd(_mm_loadu_ps(inner_axes + 2 * inner_stride + quadruples_end)), nearest_f64x4);
}

/** @ref nk_shapes_sweep_nearest_f32_haswell_ that also lowers the views' reverse minima. */
NUMKONG_INLINE void nk_shapes_sweep_mutual_f32_haswell_(__m256d outer_x_f64x4, __m256d outer_y_f64x4,
                                                        __m256d outer_z_f64x4, nk_shapes_mutual_f32_t const *views,
                                                        __m256d tail_f64x4, __m256d *nearest_f64x4) {
    nk_f32_t const *inner_axes = views->inner_axes;
    nk_size_t const inner_stride = views->inner_stride, inner_count = views->inner_count;
    nk_size_t const quadruples_end = inner_count & ~(nk_size_t)3;
    for (nk_size_t inner_index = 0; inner_index != quadruples_end; inner_index += 4)
        nk_shapes_lower_mutual_f64x4_haswell_(
            outer_x_f64x4, outer_y_f64x4, outer_z_f64x4, _mm256_cvtps_pd(_mm_loadu_ps(inner_axes + inner_index)),
            _mm256_cvtps_pd(_mm_loadu_ps(inner_axes + inner_stride + inner_index)),
            _mm256_cvtps_pd(_mm_loadu_ps(inner_axes + 2 * inner_stride + inner_index)), nearest_f64x4,
            views->reverse_minima + inner_index, views->reverse_stride);
    if (quadruples_end == inner_count) return;
    nk_shapes_lower_mutual_f64x4_haswell_(
        outer_x_f64x4, outer_y_f64x4, outer_z_f64x4,
        _mm256_blendv_pd(_mm256_set1_pd(NUMKONG_F64_INF), _mm256_cvtps_pd(_mm_loadu_ps(inner_axes + quadruples_end)),
                         tail_f64x4),
        _mm256_cvtps_pd(_mm_loadu_ps(inner_axes + inner_stride + quadruples_end)),
        _mm256_cvtps_pd(_mm_loadu_ps(inner_axes + 2 * inner_stride + quadruples_end)), nearest_f64x4,
        views->reverse_minima + quadruples_end, views->reverse_stride);
}

/** The roots of @p count reverse minima summed, @p tail_f32x8 clearing padded infinities. */
NUMKONG_INLINE nk_f32_t nk_shapes_reverse_sum_f32_haswell_(nk_f32_t const *reverse, nk_size_t count,
                                                           __m256 tail_f32x8) {
    nk_size_t const octuples_end = count & ~(nk_size_t)7;
    __m256 sum_f32x8 = _mm256_setzero_ps();
    for (nk_size_t index = 0; index != octuples_end; index += 8)
        sum_f32x8 = _mm256_add_ps(sum_f32x8, _mm256_sqrt_ps(_mm256_loadu_ps(reverse + index)));
    if (octuples_end != count)
        sum_f32x8 = _mm256_add_ps(sum_f32x8,
                                  _mm256_and_ps(tail_f32x8, _mm256_sqrt_ps(_mm256_loadu_ps(reverse + octuples_end))));
    return nk_reduce_add_f32x8_haswell_(sum_f32x8);
}

/** The roots of @p count reverse minima summed, @p tail_f64x4 clearing padded infinities. */
NUMKONG_INLINE nk_f64_t nk_shapes_reverse_sum_f64_haswell_(nk_f64_t const *reverse, nk_size_t count,
                                                           __m256d tail_f64x4) {
    nk_size_t const quadruples_end = count & ~(nk_size_t)3;
    __m256d sum_f64x4 = _mm256_setzero_pd();
    for (nk_size_t index = 0; index != quadruples_end; index += 4)
        sum_f64x4 = _mm256_add_pd(sum_f64x4, _mm256_sqrt_pd(_mm256_loadu_pd(reverse + index)));
    if (quadruples_end != count)
        sum_f64x4 = _mm256_add_pd(sum_f64x4,
                                  _mm256_and_pd(tail_f64x4, _mm256_sqrt_pd(_mm256_loadu_pd(reverse + quadruples_end))));
    return nk_reduce_add_f64x4_haswell_(sum_f64x4);
}

/** Fused Chamfer flips of @c f16 packs in f32 lanes: each outer point's sweep lowers its own minima
 *  and the inner points' reverse ones, and the 8 sums stay in one register. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f16_haswell_(void const *row_packed, void const *column_packed,
                                                                nk_f32_t limit, void *workspace,
                                                                nk_size_t workspace_bytes, nk_f32_t *chamfers) {
    nk_shapes_mutual_f16_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f16_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    __m256 const infinity_f32x8 = _mm256_set1_ps(NUMKONG_F32_INF);
    for (nk_size_t index = 0; index < 8 * views.reverse_stride; index += 8)
        _mm256_storeu_ps(views.reverse_minima + index, infinity_f32x8);
    __m256 const tail_f32x8 = nk_shapes_valid_lanes_f32x8_haswell_(views.inner_count & 7);
    nk_f32_t const outer_points = (nk_f32_t)views.outer_count, inner_points = (nk_f32_t)views.inner_count;
    __m256 const threshold_f32x8 = _mm256_set1_ps(2 * limit * outer_points);
    __m256 sums_f32x8 = _mm256_setzero_ps(), nearest_f32x8[8];
    nk_f32_t sums[8], reverse_sums[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        nk_f16_t const *outer_point = views.outer_axes + outer_index;
        __m256 const outer_x_f32x8 = _mm256_cvtph_ps(_mm_set1_epi16(*(nk_i16_t const *)outer_point));
        __m256 const outer_y_f32x8 = _mm256_cvtph_ps(
            _mm_set1_epi16(*(nk_i16_t const *)(outer_point + views.outer_stride)));
        __m256 const outer_z_f32x8 = _mm256_cvtph_ps(
            _mm_set1_epi16(*(nk_i16_t const *)(outer_point + 2 * views.outer_stride)));
        for (int flip = 0; flip != 8; ++flip) nearest_f32x8[flip] = infinity_f32x8;
        nk_shapes_sweep_mutual_f16_haswell_(outer_x_f32x8, outer_y_f32x8, outer_z_f32x8, &views, tail_f32x8,
                                            nearest_f32x8);
        sums_f32x8 = _mm256_add_ps(sums_f32x8, _mm256_sqrt_ps(nk_shapes_flip_minima_f32x8_haswell_(nearest_f32x8)));
        if (_mm256_movemask_ps(_mm256_cmp_ps(sums_f32x8, threshold_f32x8, _CMP_GT_OQ)) == 0xFF) {
            _mm256_storeu_ps(sums, sums_f32x8);
            if (nk_shapes_mutual_past_limit_f32_serial_(sums, outer_points, limit, chamfers)) return nk_success_k;
        }
    }
    for (int flip = 0; flip != 8; ++flip)
        reverse_sums[flip] = nk_shapes_reverse_sum_f32_haswell_(views.reverse_minima + flip * views.reverse_stride,
                                                                views.inner_count, tail_f32x8);
    __m256 const outer_mean_f32x8 = _mm256_div_ps(sums_f32x8, _mm256_set1_ps(outer_points));
    __m256 const inner_mean_f32x8 = _mm256_div_ps(_mm256_loadu_ps(reverse_sums), _mm256_set1_ps(inner_points));
    _mm256_storeu_ps(chamfers, _mm256_mul_ps(_mm256_add_ps(outer_mean_f32x8, inner_mean_f32x8), _mm256_set1_ps(0.5f)));
    return nk_success_k;
}

/** Hausdorff flips of @c f16 packs in f32 lanes: a pair stops once every flip's farthest square
 *  passes @p limit , whose square rounds, so the distance confirms it. */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f16_haswell_(void const *row_packed, void const *column_packed,
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
    __m256 const row_tail_f32x8 = nk_shapes_valid_lanes_f32x8_haswell_(rows & 7);
    __m256 const column_tail_f32x8 = nk_shapes_valid_lanes_f32x8_haswell_(columns & 7);
    __m256 const infinity_f32x8 = _mm256_set1_ps(NUMKONG_F32_INF);
    __m256 const limit_squared_f32x8 = _mm256_set1_ps(limit * limit);
    __m256 farthest_f32x8 = _mm256_setzero_ps(), nearest_f32x8[8];
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_f16_t const *outer_axes = direction ? column_axes : row_axes,
                       *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        __m256 const tail_f32x8 = direction ? row_tail_f32x8 : column_tail_f32x8;
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            nk_f16_t const *outer_point = outer_axes + outer_index;
            __m256 const outer_x_f32x8 = _mm256_cvtph_ps(_mm_set1_epi16(*(nk_i16_t const *)outer_point));
            __m256 const outer_y_f32x8 = _mm256_cvtph_ps(
                _mm_set1_epi16(*(nk_i16_t const *)(outer_point + outer_stride)));
            __m256 const outer_z_f32x8 = _mm256_cvtph_ps(
                _mm_set1_epi16(*(nk_i16_t const *)(outer_point + 2 * outer_stride)));
            for (int flip = 0; flip != 8; ++flip) nearest_f32x8[flip] = infinity_f32x8;
            nk_shapes_sweep_nearest_f16_haswell_(outer_x_f32x8, outer_y_f32x8, outer_z_f32x8, inner_axes, inner_stride,
                                                 inner_count, tail_f32x8, nearest_f32x8);
            farthest_f32x8 = _mm256_max_ps(farthest_f32x8, nk_shapes_flip_minima_f32x8_haswell_(nearest_f32x8));
            if (_mm256_movemask_ps(_mm256_cmp_ps(farthest_f32x8, limit_squared_f32x8, _CMP_GT_OQ)) == 0xFF)
                rejected = nk_shapes_sqrt_f32_haswell_(nk_reduce_min_f32x8_haswell_(farthest_f32x8)) > limit;
        }
    }
    _mm256_storeu_ps(hausdorffs, _mm256_sqrt_ps(farthest_f32x8));
}

/** Fused Chamfer flips of @c f32 packs in f64 lanes, reductions in two registers of 4 flips. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f32_haswell_(void const *row_packed, void const *column_packed,
                                                                nk_f64_t limit, void *workspace,
                                                                nk_size_t workspace_bytes, nk_f64_t *chamfers) {
    nk_shapes_mutual_f32_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f32_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    __m256d const infinity_f64x4 = _mm256_set1_pd(NUMKONG_F64_INF);
    for (nk_size_t index = 0; index < 8 * views.reverse_stride; index += 4)
        _mm256_storeu_pd(views.reverse_minima + index, infinity_f64x4);
    __m256d const tail_f64x4 = nk_shapes_valid_lanes_f64x4_haswell_(views.inner_count & 3);
    nk_f64_t const outer_points = (nk_f64_t)views.outer_count, inner_points = (nk_f64_t)views.inner_count;
    __m256d const threshold_f64x4 = _mm256_set1_pd(2 * limit * outer_points);
    __m256d sums_f64x4[2] = {_mm256_setzero_pd(), _mm256_setzero_pd()}, nearest_f64x4[8];
    nk_f64_t sums[8], reverse_sums[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        nk_f32_t const *outer_point = views.outer_axes + outer_index;
        __m256d const outer_x_f64x4 = _mm256_set1_pd(outer_point[0]);
        __m256d const outer_y_f64x4 = _mm256_set1_pd(outer_point[views.outer_stride]);
        __m256d const outer_z_f64x4 = _mm256_set1_pd(outer_point[2 * views.outer_stride]);
        for (int flip = 0; flip != 8; ++flip) nearest_f64x4[flip] = infinity_f64x4;
        nk_shapes_sweep_mutual_f32_haswell_(outer_x_f64x4, outer_y_f64x4, outer_z_f64x4, &views, tail_f64x4,
                                            nearest_f64x4);
        for (int half = 0; half != 2; ++half)
            sums_f64x4[half] = _mm256_add_pd(
                sums_f64x4[half], _mm256_sqrt_pd(nk_shapes_flip_minima_f64x4_haswell_(nearest_f64x4 + 4 * half)));
        if ((_mm256_movemask_pd(_mm256_cmp_pd(sums_f64x4[0], threshold_f64x4, _CMP_GT_OQ)) &
             _mm256_movemask_pd(_mm256_cmp_pd(sums_f64x4[1], threshold_f64x4, _CMP_GT_OQ))) == 0xF) {
            _mm256_storeu_pd(sums, sums_f64x4[0]), _mm256_storeu_pd(sums + 4, sums_f64x4[1]);
            if (nk_shapes_mutual_past_limit_f64_serial_(sums, outer_points, limit, chamfers)) return nk_success_k;
        }
    }
    for (int flip = 0; flip != 8; ++flip)
        reverse_sums[flip] = nk_shapes_reverse_sum_f64_haswell_(views.reverse_minima + flip * views.reverse_stride,
                                                                views.inner_count, tail_f64x4);
    for (int half = 0; half != 2; ++half) {
        __m256d const outer_mean_f64x4 = _mm256_div_pd(sums_f64x4[half], _mm256_set1_pd(outer_points));
        __m256d const inner_mean_f64x4 = _mm256_div_pd(_mm256_loadu_pd(reverse_sums + 4 * half),
                                                       _mm256_set1_pd(inner_points));
        _mm256_storeu_pd(chamfers + 4 * half,
                         _mm256_mul_pd(_mm256_add_pd(outer_mean_f64x4, inner_mean_f64x4), _mm256_set1_pd(0.5)));
    }
    return nk_success_k;
}

/** Hausdorff flips of @c f32 packs in f64 lanes, stopping as the @c f16 pair does. */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f32_haswell_(void const *row_packed, void const *column_packed,
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
    __m256d const row_tail_f64x4 = nk_shapes_valid_lanes_f64x4_haswell_(rows & 3);
    __m256d const column_tail_f64x4 = nk_shapes_valid_lanes_f64x4_haswell_(columns & 3);
    __m256d const infinity_f64x4 = _mm256_set1_pd(NUMKONG_F64_INF);
    __m256d const limit_squared_f64x4 = _mm256_set1_pd(limit * limit);
    __m256d farthest_f64x4[2] = {_mm256_setzero_pd(), _mm256_setzero_pd()}, nearest_f64x4[8];
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_f32_t const *outer_axes = direction ? column_axes : row_axes,
                       *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        __m256d const tail_f64x4 = direction ? row_tail_f64x4 : column_tail_f64x4;
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            nk_f32_t const *outer_point = outer_axes + outer_index;
            for (int flip = 0; flip != 8; ++flip) nearest_f64x4[flip] = infinity_f64x4;
            nk_shapes_sweep_nearest_f32_haswell_(_mm256_set1_pd(outer_point[0]),
                                                 _mm256_set1_pd(outer_point[outer_stride]),
                                                 _mm256_set1_pd(outer_point[2 * outer_stride]), inner_axes,
                                                 inner_stride, inner_count, tail_f64x4, nearest_f64x4);
            for (int half = 0; half != 2; ++half)
                farthest_f64x4[half] = _mm256_max_pd(farthest_f64x4[half],
                                                     nk_shapes_flip_minima_f64x4_haswell_(nearest_f64x4 + 4 * half));
            if ((_mm256_movemask_pd(_mm256_cmp_pd(farthest_f64x4[0], limit_squared_f64x4, _CMP_GT_OQ)) &
                 _mm256_movemask_pd(_mm256_cmp_pd(farthest_f64x4[1], limit_squared_f64x4, _CMP_GT_OQ))) == 0xF)
                rejected = nk_shapes_sqrt_f64_haswell_(nk_reduce_min_f64x4_haswell_(
                               _mm256_min_pd(farthest_f64x4[0], farthest_f64x4[1]))) > limit;
        }
    }
    _mm256_storeu_pd(hausdorffs, _mm256_sqrt_pd(farthest_f64x4[0]));
    _mm256_storeu_pd(hausdorffs + 4, _mm256_sqrt_pd(farthest_f64x4[1]));
}

/** Squared distances from a broadcast row point in fused multiply-adds, which also serve single
 *  arcs, so that tree arcs keep a zero reduced cost. */
NUMKONG_INLINE __m128 nk_shapes_point_costs_f32x4_haswell_(__m128 row_x_f32x4, __m128 row_y_f32x4, __m128 row_z_f32x4,
                                                           __m128 column_x_f32x4, __m128 column_y_f32x4,
                                                           __m128 column_z_f32x4) {
    __m128 const delta_x_f32x4 = _mm_sub_ps(row_x_f32x4, column_x_f32x4);
    __m128 const delta_y_f32x4 = _mm_sub_ps(row_y_f32x4, column_y_f32x4);
    __m128 const delta_z_f32x4 = _mm_sub_ps(row_z_f32x4, column_z_f32x4);
    return _mm_fmadd_ps(delta_z_f32x4, delta_z_f32x4,
                        _mm_fmadd_ps(delta_y_f32x4, delta_y_f32x4, _mm_mul_ps(delta_x_f32x4, delta_x_f32x4)));
}

NUMKONG_INLINE nk_f32_t nk_shapes_point_cost_f32_haswell_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t row,
                                                          nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    return _mm_cvtss_f32(nk_shapes_point_costs_f32x4_haswell_(
        _mm_set1_ps(row_points[row]), _mm_set1_ps(row_points[stride + row]), _mm_set1_ps(row_points[2 * stride + row]),
        _mm_set1_ps(rotated_points[column]), _mm_set1_ps(rotated_points[stride + column]),
        _mm_set1_ps(rotated_points[2 * stride + column])));
}

/** The reduced costs (c − πᵢ) + πⱼ of a row's arcs to 4 columns, with matrix costs −G taken as
 *  πⱼ − (G + πᵢ), which rounds the same. */
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

/** Keeps the first of four reduced costs from @p column that beats the running least. */
NUMKONG_INLINE void nk_shapes_keep_least_f64x4_haswell_(__m256d reduced_f64x4, nk_size_t row, nk_size_t column,
                                                        __m256d *least_f64x4, nk_f64_t *entering_cost,
                                                        nk_size_t *entering_row, nk_size_t *entering_column) {
    if (!_mm256_movemask_pd(_mm256_cmp_pd(reduced_f64x4, *least_f64x4, _CMP_LT_OQ))) return;
    *entering_cost = nk_reduce_min_f64x4_haswell_(reduced_f64x4);
    *least_f64x4 = _mm256_set1_pd(*entering_cost);
    int const equal_lanes = _mm256_movemask_pd(_mm256_cmp_pd(reduced_f64x4, *least_f64x4, _CMP_EQ_OQ));
    *entering_row = row, *entering_column = column + _tzcnt_u32((unsigned int)equal_lanes);
}

/** Keeps a tile's first least reduced cost below @p entering_cost, 4 columns at a time. Lanes past
 *  the tile read neighboring workspace and cost infinity. */
NUMKONG_INLINE void nk_shapes_price_tile_f32_haswell_(nk_shapes_workspace_f32_serial_t const *views,
                                                      nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                      nk_size_t row, nk_size_t column, nk_size_t tile,
                                                      nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                      nk_size_t *entering_column) {
    __m256d least_f64x4 = _mm256_set1_pd(*entering_cost);
    nk_size_t const quadruples_end = tile & ~(nk_size_t)3;
    for (nk_size_t step = 0; step != quadruples_end; step += 4)
        nk_shapes_keep_least_f64x4_haswell_(
            nk_shapes_reduced_costs_f32_haswell_(views, gradient, rows, columns, row, column + step), row,
            column + step, &least_f64x4, entering_cost, entering_row, entering_column);
    if (quadruples_end == tile) return;
    nk_shapes_keep_least_f64x4_haswell_(
        _mm256_blendv_pd(
            _mm256_set1_pd(NUMKONG_F64_INF),
            nk_shapes_reduced_costs_f32_haswell_(views, gradient, rows, columns, row, column + quadruples_end),
            nk_shapes_valid_lanes_f64x4_haswell_(tile - quadruples_end)),
        row, column + quadruples_end, &least_f64x4, entering_cost, entering_row, entering_column);
}

NUMKONG_INLINE void nk_shapes_tree_costs_f32_haswell_(nk_shapes_workspace_f32_serial_t const *views,
                                                      nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        views->potential[node] = gradient ? -gradient[row * columns + column]
                                          : (nk_f64_t)nk_shapes_point_cost_f32_haswell_(views, row, column);
    }
}

NUMKONG_INLINE void nk_shapes_network_simplex_f32_haswell_(nk_shapes_workspace_f32_serial_t const *views,
                                                           nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                           nk_f64_t cost_scale) {
    nk_shapes_simplex_serial_t simplex;
    nk_shapes_tree_costs_f32_haswell_(views, gradient, rows, columns);
    nk_shapes_tree_potentials_serial_(views->tree, views->tree_stride, views->potential, rows);
    nk_shapes_simplex_start_serial_(&simplex, views->tree, views->tree_stride, views->potential, rows, columns,
                                    cost_scale);
    while (nk_shapes_simplex_next_tile_serial_(&simplex))
        nk_shapes_price_tile_f32_haswell_(views, gradient, rows, columns, simplex.row, simplex.column, simplex.tile,
                                          &simplex.entering_cost, &simplex.entering_row, &simplex.entering_column);
}

NUMKONG_INLINE nk_f32_t nk_shapes_tree_cost_f32_haswell_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t rows,
                                                         nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f32_t cost = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        cost += (nk_f32_t)flow[node] * nk_shapes_point_cost_f32_haswell_(views, row, column);
    }
    return cost / (nk_f32_t)(rows * columns);
}

NUMKONG_INLINE __m256d nk_shapes_point_costs_f64x4_haswell_(__m256d row_x_f64x4, __m256d row_y_f64x4,
                                                            __m256d row_z_f64x4, __m256d column_x_f64x4,
                                                            __m256d column_y_f64x4, __m256d column_z_f64x4) {
    __m256d const delta_x_f64x4 = _mm256_sub_pd(row_x_f64x4, column_x_f64x4);
    __m256d const delta_y_f64x4 = _mm256_sub_pd(row_y_f64x4, column_y_f64x4);
    __m256d const delta_z_f64x4 = _mm256_sub_pd(row_z_f64x4, column_z_f64x4);
    return _mm256_fmadd_pd(delta_z_f64x4, delta_z_f64x4,
                           _mm256_fmadd_pd(delta_y_f64x4, delta_y_f64x4, _mm256_mul_pd(delta_x_f64x4, delta_x_f64x4)));
}

NUMKONG_INLINE nk_f64_t nk_shapes_point_cost_f64_haswell_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t row,
                                                          nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    return _mm256_cvtsd_f64(nk_shapes_point_costs_f64x4_haswell_(
        _mm256_set1_pd(row_points[row]), _mm256_set1_pd(row_points[stride + row]),
        _mm256_set1_pd(row_points[2 * stride + row]), _mm256_set1_pd(rotated_points[column]),
        _mm256_set1_pd(rotated_points[stride + column]), _mm256_set1_pd(rotated_points[2 * stride + column])));
}

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

NUMKONG_INLINE void nk_shapes_price_tile_f64_haswell_(nk_shapes_workspace_f64_serial_t const *views,
                                                      nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                      nk_size_t row, nk_size_t column, nk_size_t tile,
                                                      nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                      nk_size_t *entering_column) {
    __m256d least_f64x4 = _mm256_set1_pd(*entering_cost);
    nk_size_t const quadruples_end = tile & ~(nk_size_t)3;
    for (nk_size_t step = 0; step != quadruples_end; step += 4)
        nk_shapes_keep_least_f64x4_haswell_(
            nk_shapes_reduced_costs_f64_haswell_(views, gradient, rows, columns, row, column + step), row,
            column + step, &least_f64x4, entering_cost, entering_row, entering_column);
    if (quadruples_end == tile) return;
    nk_shapes_keep_least_f64x4_haswell_(
        _mm256_blendv_pd(
            _mm256_set1_pd(NUMKONG_F64_INF),
            nk_shapes_reduced_costs_f64_haswell_(views, gradient, rows, columns, row, column + quadruples_end),
            nk_shapes_valid_lanes_f64x4_haswell_(tile - quadruples_end)),
        row, column + quadruples_end, &least_f64x4, entering_cost, entering_row, entering_column);
}

NUMKONG_INLINE void nk_shapes_tree_costs_f64_haswell_(nk_shapes_workspace_f64_serial_t const *views,
                                                      nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        views->potential[node] = gradient ? -gradient[row * columns + column]
                                          : nk_shapes_point_cost_f64_haswell_(views, row, column);
    }
}

NUMKONG_INLINE void nk_shapes_network_simplex_f64_haswell_(nk_shapes_workspace_f64_serial_t const *views,
                                                           nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                           nk_f64_t cost_scale) {
    nk_shapes_simplex_serial_t simplex;
    nk_shapes_tree_costs_f64_haswell_(views, gradient, rows, columns);
    nk_shapes_tree_potentials_serial_(views->tree, views->tree_stride, views->potential, rows);
    nk_shapes_simplex_start_serial_(&simplex, views->tree, views->tree_stride, views->potential, rows, columns,
                                    cost_scale);
    while (nk_shapes_simplex_next_tile_serial_(&simplex))
        nk_shapes_price_tile_f64_haswell_(views, gradient, rows, columns, simplex.row, simplex.column, simplex.tile,
                                          &simplex.entering_cost, &simplex.entering_row, &simplex.entering_column);
}

NUMKONG_INLINE nk_f64_t nk_shapes_tree_cost_f64_haswell_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t rows,
                                                         nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f64_t cost = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        cost += (nk_f64_t)flow[node] * nk_shapes_point_cost_f64_haswell_(views, row, column);
    }
    return cost / (nk_f64_t)(rows * columns);
}

NUMKONG_INLINE nk_f32_t nk_shapes_wasserstein_procrustes_f16_haswell_(nk_shapes_workspace_f32_serial_t const *views,
                                                                      nk_size_t rows, nk_size_t columns,
                                                                      nk_f64_t cost_scale, nk_f32_t *rotation) {
    nk_shapes_procrustes_f32_serial_t search;
    nk_shapes_procrustes_start_f32_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_f32_serial_(&search)) {
        nk_shapes_network_simplex_f32_haswell_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_f32_haswell_(views, rows, columns);
    }
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f16_haswell_(nk_shapes_workspace_f32_serial_t const *views,
                                                                  nk_size_t rows, nk_size_t columns,
                                                                  nk_f64_t cost_scale, nk_f64_t const *row_distances,
                                                                  nk_f64_t const *column_distances,
                                                                  nk_f32_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f16_haswell_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f32_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_serial_(&search);
    while (nk_shapes_gromov_next_f32_serial_(&search, views, rotation))
        nk_shapes_network_simplex_f32_haswell_(views, search.gradient, rows, columns, search.gradient_scale);
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_wasserstein_procrustes_f32_haswell_(nk_shapes_workspace_f64_serial_t const *views,
                                                                      nk_size_t rows, nk_size_t columns,
                                                                      nk_f64_t cost_scale, nk_f64_t *rotation) {
    nk_shapes_procrustes_f64_serial_t search;
    nk_shapes_procrustes_start_f64_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_f64_serial_(&search)) {
        nk_shapes_network_simplex_f64_haswell_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_f64_haswell_(views, rows, columns);
    }
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f32_haswell_(nk_shapes_workspace_f64_serial_t const *views,
                                                                  nk_size_t rows, nk_size_t columns,
                                                                  nk_f64_t cost_scale, nk_f64_t const *row_distances,
                                                                  nk_f64_t const *column_distances,
                                                                  nk_f64_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f32_haswell_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f64_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_serial_(&search);
    while (nk_shapes_gromov_next_f64_serial_(&search, views, rotation))
        nk_shapes_network_simplex_f64_haswell_(views, search.gradient, rows, columns, search.gradient_scale);
    return search.best;
}

NUMKONG_INLINE void nk_jacobi_eigenanalysis_f32_haswell_(nk_f32_t *matrix, nk_f32_t *quaternion) {
    nk_jacobi_start_f32_serial_(quaternion);
    for (int sweep = 0; sweep != 16; ++sweep) {
        for (int axis = 0; axis != 3; ++axis) {
            nk_f32_t const norm = nk_jacobi_half_angle_f32_serial_(matrix);
            nk_jacobi_conjugate_f32_serial_(axis, nk_rsqrt_f32_haswell_(norm), matrix, quaternion);
        }
        if (nk_jacobi_converged_f32_serial_(matrix, 0x1p-24f)) break;
    }
    nk_f32_t const length_squared = nk_jacobi_length_squared_f32_serial_(quaternion);
    nk_jacobi_rescale_f32_serial_(nk_rsqrt_f32_haswell_(length_squared), quaternion);
}

NUMKONG_INLINE void nk_jacobi_eigenanalysis_f64_haswell_(nk_f64_t *matrix, nk_f64_t *quaternion) {
    nk_jacobi_start_f64_serial_(quaternion);
    for (int sweep = 0; sweep != 16; ++sweep) {
        for (int axis = 0; axis != 3; ++axis) {
            nk_f64_t const norm = nk_jacobi_half_angle_f64_serial_(matrix);
            nk_jacobi_conjugate_f64_serial_(axis, nk_shapes_rsqrt_f64_haswell_(norm), matrix, quaternion);
        }
        if (nk_jacobi_converged_f64_serial_(matrix, 0x1p-53)) break;
    }
    nk_f64_t const length_squared = nk_jacobi_length_squared_f64_serial_(quaternion);
    nk_jacobi_rescale_f64_serial_(nk_shapes_rsqrt_f64_haswell_(length_squared), quaternion);
}

/** The next 8 @c f16 points of the @p left remaining, each axis in f32 lanes. Padded lanes read
 *  zero, and three 16-byte chunks of xyz triples blend and shuffle into each axis. */
NUMKONG_INLINE void nk_shapes_load_points_f16_haswell_(nk_f16_t const *points, nk_size_t left, __m256 *x_f32x8,
                                                       __m256 *y_f32x8, __m256 *z_f32x8) {
    nk_b256_vec_t head_vec;
    nk_b128_vec_t last_vec;
    if (left < 8) {
        nk_size_t const words = left * 3;
        nk_partial_load_b16x16_haswell_(points, &head_vec, words < 16 ? words : 16);
        nk_partial_load_b16x8_haswell_(points + 16, &last_vec, words > 16 ? words - 16 : 0);
    }
    else {
        head_vec.ymm = _mm256_loadu_si256((__m256i const *)points);
        last_vec.xmm = _mm_loadu_si128((__m128i const *)(points + 16));
    }
    __m128i const first_u16x8 = head_vec.xmms[0], second_u16x8 = head_vec.xmms[1], third_u16x8 = last_vec.xmm;
    __m128i const x_u16x8 = _mm_blend_epi16(_mm_blend_epi16(first_u16x8, second_u16x8, 0x92), third_u16x8, 0x24);
    __m128i const y_u16x8 = _mm_blend_epi16(_mm_blend_epi16(first_u16x8, second_u16x8, 0x24), third_u16x8, 0x49);
    __m128i const z_u16x8 = _mm_blend_epi16(_mm_blend_epi16(first_u16x8, second_u16x8, 0x49), third_u16x8, 0x92);
    *x_f32x8 = _mm256_cvtph_ps(
        _mm_shuffle_epi8(x_u16x8, _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11)));
    *y_f32x8 = _mm256_cvtph_ps(
        _mm_shuffle_epi8(y_u16x8, _mm_setr_epi8(2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13)));
    *z_f32x8 = _mm256_cvtph_ps(
        _mm_shuffle_epi8(z_u16x8, _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15)));
}

/** The moments of a cloud of @c f16 points as @ref nk_shapes_moments_f16_serial_ defines them, 8
 *  points per step. */
NUMKONG_INLINE nk_f32_t nk_shapes_moments_f16_haswell_(nk_f16_t const *points, nk_size_t count, nk_f32_t *centroid,
                                                       nk_f32_t *covariance) {
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    __m256 x_f32x8, y_f32x8, z_f32x8;
    nk_shapes_load_points_f16_haswell_(points, 1, &x_f32x8, &y_f32x8, &z_f32x8);
    __m256 const first_x_f32x8 = _mm256_broadcastss_ps(_mm256_castps256_ps128(x_f32x8));
    __m256 const first_y_f32x8 = _mm256_broadcastss_ps(_mm256_castps256_ps128(y_f32x8));
    __m256 const first_z_f32x8 = _mm256_broadcastss_ps(_mm256_castps256_ps128(z_f32x8));
    __m256 sum_x_f32x8 = _mm256_setzero_ps(), sum_y_f32x8 = sum_x_f32x8, sum_z_f32x8 = sum_x_f32x8;
    __m256 differing_f32x8 = _mm256_setzero_ps();
    for (nk_size_t index = 0; index < count; index += 8) {
        nk_shapes_load_points_f16_haswell_(points + index * 3, count - index, &x_f32x8, &y_f32x8, &z_f32x8);
        sum_x_f32x8 = _mm256_add_ps(sum_x_f32x8, x_f32x8), sum_y_f32x8 = _mm256_add_ps(sum_y_f32x8, y_f32x8);
        sum_z_f32x8 = _mm256_add_ps(sum_z_f32x8, z_f32x8);
        __m256 const same_f32x8 = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(x_f32x8, first_x_f32x8, _CMP_EQ_OQ),
                                                              _mm256_cmp_ps(y_f32x8, first_y_f32x8, _CMP_EQ_OQ)),
                                                _mm256_cmp_ps(z_f32x8, first_z_f32x8, _CMP_EQ_OQ));
        differing_f32x8 = _mm256_or_ps(
            differing_f32x8, _mm256_andnot_ps(same_f32x8, nk_shapes_valid_lanes_f32x8_haswell_(count - index)));
    }
    nk_f32_t const points_count = (nk_f32_t)count;
    centroid[0] = nk_reduce_add_f32x8_haswell_(sum_x_f32x8) / points_count;
    centroid[1] = nk_reduce_add_f32x8_haswell_(sum_y_f32x8) / points_count;
    centroid[2] = nk_reduce_add_f32x8_haswell_(sum_z_f32x8) / points_count;
    __m256 const centroid_x_f32x8 = _mm256_set1_ps(centroid[0]), centroid_y_f32x8 = _mm256_set1_ps(centroid[1]);
    __m256 const centroid_z_f32x8 = _mm256_set1_ps(centroid[2]);
    __m256 moments_f32x8[6];
    for (int entry = 0; entry != 6; ++entry) moments_f32x8[entry] = _mm256_setzero_ps();
    for (nk_size_t index = 0; index < count; index += 8) {
        nk_shapes_load_points_f16_haswell_(points + index * 3, count - index, &x_f32x8, &y_f32x8, &z_f32x8);
        __m256 const valid_f32x8 = nk_shapes_valid_lanes_f32x8_haswell_(count - index);
        x_f32x8 = _mm256_and_ps(valid_f32x8, _mm256_sub_ps(x_f32x8, centroid_x_f32x8));
        y_f32x8 = _mm256_and_ps(valid_f32x8, _mm256_sub_ps(y_f32x8, centroid_y_f32x8));
        z_f32x8 = _mm256_and_ps(valid_f32x8, _mm256_sub_ps(z_f32x8, centroid_z_f32x8));
        moments_f32x8[0] = _mm256_fmadd_ps(x_f32x8, x_f32x8, moments_f32x8[0]);
        moments_f32x8[1] = _mm256_fmadd_ps(y_f32x8, x_f32x8, moments_f32x8[1]);
        moments_f32x8[2] = _mm256_fmadd_ps(y_f32x8, y_f32x8, moments_f32x8[2]);
        moments_f32x8[3] = _mm256_fmadd_ps(z_f32x8, x_f32x8, moments_f32x8[3]);
        moments_f32x8[4] = _mm256_fmadd_ps(z_f32x8, y_f32x8, moments_f32x8[4]);
        moments_f32x8[5] = _mm256_fmadd_ps(z_f32x8, z_f32x8, moments_f32x8[5]);
    }
    for (int entry = 0; entry != 6; ++entry)
        covariance[entry] = nk_reduce_add_f32x8_haswell_(moments_f32x8[entry]) / points_count;
    return _mm256_movemask_ps(differing_f32x8)
               ? nk_shapes_sqrt_f32_haswell_(covariance[0] + covariance[2] + covariance[5])
               : 0;
}

/** The pack of one cloud from its moments, as @c nk_shapes_pack_f16_serial_ lays it out, storing
 *  each step's points and measuring what was stored in one sweep, with zeros in the padding. */
NUMKONG_INLINE void nk_shapes_pack_f16_haswell_(nk_f16_t const *points, nk_size_t count, nk_f32_t const *centroid,
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
    nk_f16_t *packed_x = (nk_f16_t *)(packed_bytes + header_bytes);
    nk_f16_t *packed_y = packed_x + axis_stride, *packed_z = packed_y + axis_stride;
    nk_f32_t spreads[6] = {covariance[0], covariance[1], covariance[2], covariance[3], covariance[4], covariance[5]};
    nk_f32_t quaternion[4], basis[9], axes[9];
    int upper_tied, lower_tied;
    nk_jacobi_eigenanalysis_f32_haswell_(spreads, quaternion);
    nk_shapes_sorted_basis_f32_serial_(spreads, quaternion, 0x1p-7f, basis, &upper_tied, &lower_tied);
    if (upper_tied && lower_tied) nk_shapes_pick_axis_f16_serial_(points, count, centroid, basis);
    else if (upper_tied || lower_tied) nk_shapes_turn_plane_f16_serial_(points, count, centroid, basis);
    nk_shapes_frame_f32_serial_(basis, upper_tied, header->frame, axes);
    __m256 axes_f32x8[9];
    for (int cell = 0; cell != 9; ++cell) axes_f32x8[cell] = _mm256_set1_ps(axes[cell] / scale);
    __m256 const centroid_x_f32x8 = _mm256_set1_ps(centroid[0]), centroid_y_f32x8 = _mm256_set1_ps(centroid[1]);
    __m256 const centroid_z_f32x8 = _mm256_set1_ps(centroid[2]);
    __m256 moments_f32x8[6], largest_f32x8 = _mm256_setzero_ps();
    for (int entry = 0; entry != 6; ++entry) moments_f32x8[entry] = _mm256_setzero_ps();
    __m256 x_f32x8, y_f32x8, z_f32x8;
    for (nk_size_t index = 0; index < count; index += 8) {
        nk_shapes_load_points_f16_haswell_(points + index * 3, count - index, &x_f32x8, &y_f32x8, &z_f32x8);
        __m256 const valid_f32x8 = nk_shapes_valid_lanes_f32x8_haswell_(count - index);
        x_f32x8 = _mm256_sub_ps(x_f32x8, centroid_x_f32x8), y_f32x8 = _mm256_sub_ps(y_f32x8, centroid_y_f32x8);
        z_f32x8 = _mm256_sub_ps(z_f32x8, centroid_z_f32x8);
        __m128i stored_f16x8[3];
        for (int axis = 0; axis != 3; ++axis) {
            __m256 const rotated_f32x8 = _mm256_fmadd_ps(
                z_f32x8, axes_f32x8[axis * 3 + 2],
                _mm256_fmadd_ps(y_f32x8, axes_f32x8[axis * 3 + 1], _mm256_mul_ps(x_f32x8, axes_f32x8[axis * 3])));
            stored_f16x8[axis] = _mm256_cvtps_ph(_mm256_and_ps(valid_f32x8, rotated_f32x8), _MM_FROUND_TO_NEAREST_INT);
        }
        _mm_storeu_si128((__m128i *)(packed_x + index), stored_f16x8[0]);
        _mm_storeu_si128((__m128i *)(packed_y + index), stored_f16x8[1]);
        _mm_storeu_si128((__m128i *)(packed_z + index), stored_f16x8[2]);
        // The spreads describe the stored points, so the sweep measures them after rounding
        x_f32x8 = _mm256_cvtph_ps(stored_f16x8[0]), y_f32x8 = _mm256_cvtph_ps(stored_f16x8[1]);
        z_f32x8 = _mm256_cvtph_ps(stored_f16x8[2]);
        __m256 const xx_f32x8 = _mm256_mul_ps(x_f32x8, x_f32x8), yy_f32x8 = _mm256_mul_ps(y_f32x8, y_f32x8);
        __m256 const zz_f32x8 = _mm256_mul_ps(z_f32x8, z_f32x8);
        largest_f32x8 = _mm256_max_ps(largest_f32x8, _mm256_add_ps(_mm256_add_ps(xx_f32x8, yy_f32x8), zz_f32x8));
        moments_f32x8[0] = _mm256_add_ps(moments_f32x8[0], xx_f32x8);
        moments_f32x8[1] = _mm256_fmadd_ps(y_f32x8, x_f32x8, moments_f32x8[1]);
        moments_f32x8[2] = _mm256_add_ps(moments_f32x8[2], yy_f32x8);
        moments_f32x8[3] = _mm256_fmadd_ps(z_f32x8, x_f32x8, moments_f32x8[3]);
        moments_f32x8[4] = _mm256_fmadd_ps(z_f32x8, y_f32x8, moments_f32x8[4]);
        moments_f32x8[5] = _mm256_add_ps(moments_f32x8[5], zz_f32x8);
    }
    for (nk_size_t index = (count + 7) & ~(nk_size_t)7; index < axis_stride; index += 8)
        for (int axis = 0; axis != 3; ++axis)
            _mm_storeu_si128((__m128i *)(packed_x + axis * axis_stride + index), _mm_setzero_si128());
    nk_f32_t moments[6];
    for (int entry = 0; entry != 6; ++entry)
        moments[entry] = nk_reduce_add_f32x8_haswell_(moments_f32x8[entry]) / (nk_f32_t)count;
    nk_jacobi_eigenanalysis_f32_haswell_(moments, quaternion);
    nk_shapes_close_pack_f32_serial_(header, count, centroid, moments,
                                     nk_shapes_sqrt_f32_haswell_(nk_reduce_max_f32x8_haswell_(largest_f32x8)));
}

/** The next 4 @c f32 points of the @p left remaining, each axis in f64 lanes. Padded lanes read
 *  zero, and three 16-byte chunks of xyz triples blend and shuffle into each axis. */
NUMKONG_INLINE void nk_shapes_load_points_f32_haswell_(nk_f32_t const *points, nk_size_t left, __m256d *x_f64x4,
                                                       __m256d *y_f64x4, __m256d *z_f64x4) {
    nk_b256_vec_t head_vec;
    nk_b128_vec_t last_vec;
    if (left < 4) {
        nk_size_t const words = left * 3;
        nk_partial_load_b32x8_haswell_(points, &head_vec, words < 8 ? words : 8);
        nk_partial_load_b32x4_haswell_(points + 8, &last_vec, words > 8 ? words - 8 : 0);
    }
    else {
        head_vec.ymm_ps = _mm256_loadu_ps(points);
        last_vec.xmm_ps = _mm_loadu_ps(points + 8);
    }
    __m128 const first_f32x4 = _mm256_castps256_ps128(head_vec.ymm_ps);
    __m128 const second_f32x4 = _mm256_extractf128_ps(head_vec.ymm_ps, 1), third_f32x4 = last_vec.xmm_ps;
    __m128 const x_f32x4 = _mm_blend_ps(_mm_blend_ps(first_f32x4, second_f32x4, 0x4), third_f32x4, 0x2);
    __m128 const y_f32x4 = _mm_blend_ps(_mm_blend_ps(first_f32x4, second_f32x4, 0x9), third_f32x4, 0x4);
    __m128 const z_f32x4 = _mm_blend_ps(_mm_blend_ps(first_f32x4, second_f32x4, 0x2), third_f32x4, 0x9);
    *x_f64x4 = _mm256_cvtps_pd(_mm_permute_ps(x_f32x4, 0x6C));
    *y_f64x4 = _mm256_cvtps_pd(_mm_permute_ps(y_f32x4, 0xB1));
    *z_f64x4 = _mm256_cvtps_pd(_mm_permute_ps(z_f32x4, 0xC6));
}

/** The moments of a cloud of @c f32 points as @ref nk_shapes_moments_f32_serial_ defines them, 4
 *  points per step. */
NUMKONG_INLINE nk_f64_t nk_shapes_moments_f32_haswell_(nk_f32_t const *points, nk_size_t count, nk_f64_t *centroid,
                                                       nk_f64_t *covariance) {
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    __m256d x_f64x4, y_f64x4, z_f64x4;
    __m256d const first_x_f64x4 = _mm256_set1_pd(points[0]), first_y_f64x4 = _mm256_set1_pd(points[1]);
    __m256d const first_z_f64x4 = _mm256_set1_pd(points[2]);
    __m256d sum_x_f64x4 = _mm256_setzero_pd(), sum_y_f64x4 = sum_x_f64x4, sum_z_f64x4 = sum_x_f64x4;
    __m256d differing_f64x4 = _mm256_setzero_pd();
    for (nk_size_t index = 0; index < count; index += 4) {
        nk_shapes_load_points_f32_haswell_(points + index * 3, count - index, &x_f64x4, &y_f64x4, &z_f64x4);
        sum_x_f64x4 = _mm256_add_pd(sum_x_f64x4, x_f64x4), sum_y_f64x4 = _mm256_add_pd(sum_y_f64x4, y_f64x4);
        sum_z_f64x4 = _mm256_add_pd(sum_z_f64x4, z_f64x4);
        __m256d const same_f64x4 = _mm256_and_pd(_mm256_and_pd(_mm256_cmp_pd(x_f64x4, first_x_f64x4, _CMP_EQ_OQ),
                                                               _mm256_cmp_pd(y_f64x4, first_y_f64x4, _CMP_EQ_OQ)),
                                                 _mm256_cmp_pd(z_f64x4, first_z_f64x4, _CMP_EQ_OQ));
        differing_f64x4 = _mm256_or_pd(
            differing_f64x4, _mm256_andnot_pd(same_f64x4, nk_shapes_valid_lanes_f64x4_haswell_(count - index)));
    }
    nk_f64_t const points_count = (nk_f64_t)count;
    centroid[0] = nk_reduce_add_f64x4_haswell_(sum_x_f64x4) / points_count;
    centroid[1] = nk_reduce_add_f64x4_haswell_(sum_y_f64x4) / points_count;
    centroid[2] = nk_reduce_add_f64x4_haswell_(sum_z_f64x4) / points_count;
    __m256d const centroid_x_f64x4 = _mm256_set1_pd(centroid[0]), centroid_y_f64x4 = _mm256_set1_pd(centroid[1]);
    __m256d const centroid_z_f64x4 = _mm256_set1_pd(centroid[2]);
    __m256d moments_f64x4[6];
    for (int entry = 0; entry != 6; ++entry) moments_f64x4[entry] = _mm256_setzero_pd();
    for (nk_size_t index = 0; index < count; index += 4) {
        nk_shapes_load_points_f32_haswell_(points + index * 3, count - index, &x_f64x4, &y_f64x4, &z_f64x4);
        __m256d const valid_f64x4 = nk_shapes_valid_lanes_f64x4_haswell_(count - index);
        x_f64x4 = _mm256_and_pd(valid_f64x4, _mm256_sub_pd(x_f64x4, centroid_x_f64x4));
        y_f64x4 = _mm256_and_pd(valid_f64x4, _mm256_sub_pd(y_f64x4, centroid_y_f64x4));
        z_f64x4 = _mm256_and_pd(valid_f64x4, _mm256_sub_pd(z_f64x4, centroid_z_f64x4));
        moments_f64x4[0] = _mm256_fmadd_pd(x_f64x4, x_f64x4, moments_f64x4[0]);
        moments_f64x4[1] = _mm256_fmadd_pd(y_f64x4, x_f64x4, moments_f64x4[1]);
        moments_f64x4[2] = _mm256_fmadd_pd(y_f64x4, y_f64x4, moments_f64x4[2]);
        moments_f64x4[3] = _mm256_fmadd_pd(z_f64x4, x_f64x4, moments_f64x4[3]);
        moments_f64x4[4] = _mm256_fmadd_pd(z_f64x4, y_f64x4, moments_f64x4[4]);
        moments_f64x4[5] = _mm256_fmadd_pd(z_f64x4, z_f64x4, moments_f64x4[5]);
    }
    for (int entry = 0; entry != 6; ++entry)
        covariance[entry] = nk_reduce_add_f64x4_haswell_(moments_f64x4[entry]) / points_count;
    return _mm256_movemask_pd(differing_f64x4)
               ? nk_shapes_sqrt_f64_haswell_(covariance[0] + covariance[2] + covariance[5])
               : 0;
}

/** @copydoc nk_shapes_pack_f16_haswell_ */
NUMKONG_INLINE void nk_shapes_pack_f32_haswell_(nk_f32_t const *points, nk_size_t count, nk_f64_t const *centroid,
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
    nk_jacobi_eigenanalysis_f64_haswell_(spreads, quaternion);
    nk_shapes_sorted_basis_f64_serial_(spreads, quaternion, 0x1p-16, basis, &upper_tied, &lower_tied);
    if (upper_tied && lower_tied) nk_shapes_pick_axis_f32_serial_(points, count, centroid, basis);
    else if (upper_tied || lower_tied) nk_shapes_turn_plane_f32_serial_(points, count, centroid, basis);
    nk_shapes_frame_f64_serial_(basis, upper_tied, header->frame, axes);
    __m256d axes_f64x4[9];
    for (int cell = 0; cell != 9; ++cell) axes_f64x4[cell] = _mm256_set1_pd(axes[cell] / scale);
    __m256d const centroid_x_f64x4 = _mm256_set1_pd(centroid[0]), centroid_y_f64x4 = _mm256_set1_pd(centroid[1]);
    __m256d const centroid_z_f64x4 = _mm256_set1_pd(centroid[2]);
    __m256d moments_f64x4[6], largest_f64x4 = _mm256_setzero_pd();
    for (int entry = 0; entry != 6; ++entry) moments_f64x4[entry] = _mm256_setzero_pd();
    __m256d x_f64x4, y_f64x4, z_f64x4;
    for (nk_size_t index = 0; index < count; index += 4) {
        nk_shapes_load_points_f32_haswell_(points + index * 3, count - index, &x_f64x4, &y_f64x4, &z_f64x4);
        __m256d const valid_f64x4 = nk_shapes_valid_lanes_f64x4_haswell_(count - index);
        x_f64x4 = _mm256_sub_pd(x_f64x4, centroid_x_f64x4), y_f64x4 = _mm256_sub_pd(y_f64x4, centroid_y_f64x4);
        z_f64x4 = _mm256_sub_pd(z_f64x4, centroid_z_f64x4);
        __m128 stored_f32x4[3];
        for (int axis = 0; axis != 3; ++axis) {
            __m256d const rotated_f64x4 = _mm256_fmadd_pd(
                z_f64x4, axes_f64x4[axis * 3 + 2],
                _mm256_fmadd_pd(y_f64x4, axes_f64x4[axis * 3 + 1], _mm256_mul_pd(x_f64x4, axes_f64x4[axis * 3])));
            stored_f32x4[axis] = _mm256_cvtpd_ps(_mm256_and_pd(valid_f64x4, rotated_f64x4));
        }
        _mm_storeu_ps(packed_x + index, stored_f32x4[0]), _mm_storeu_ps(packed_y + index, stored_f32x4[1]);
        _mm_storeu_ps(packed_z + index, stored_f32x4[2]);
        // The spreads describe the stored points, so the sweep measures them after rounding
        x_f64x4 = _mm256_cvtps_pd(stored_f32x4[0]), y_f64x4 = _mm256_cvtps_pd(stored_f32x4[1]);
        z_f64x4 = _mm256_cvtps_pd(stored_f32x4[2]);
        __m256d const xx_f64x4 = _mm256_mul_pd(x_f64x4, x_f64x4), yy_f64x4 = _mm256_mul_pd(y_f64x4, y_f64x4);
        __m256d const zz_f64x4 = _mm256_mul_pd(z_f64x4, z_f64x4);
        largest_f64x4 = _mm256_max_pd(largest_f64x4, _mm256_add_pd(_mm256_add_pd(xx_f64x4, yy_f64x4), zz_f64x4));
        moments_f64x4[0] = _mm256_add_pd(moments_f64x4[0], xx_f64x4);
        moments_f64x4[1] = _mm256_fmadd_pd(y_f64x4, x_f64x4, moments_f64x4[1]);
        moments_f64x4[2] = _mm256_add_pd(moments_f64x4[2], yy_f64x4);
        moments_f64x4[3] = _mm256_fmadd_pd(z_f64x4, x_f64x4, moments_f64x4[3]);
        moments_f64x4[4] = _mm256_fmadd_pd(z_f64x4, y_f64x4, moments_f64x4[4]);
        moments_f64x4[5] = _mm256_add_pd(moments_f64x4[5], zz_f64x4);
    }
    for (nk_size_t index = (count + 3) & ~(nk_size_t)3; index < axis_stride; index += 4)
        for (int axis = 0; axis != 3; ++axis) _mm_storeu_ps(packed_x + axis * axis_stride + index, _mm_setzero_ps());
    nk_f64_t moments[6];
    for (int entry = 0; entry != 6; ++entry)
        moments[entry] = nk_reduce_add_f64x4_haswell_(moments_f64x4[entry]) / (nk_f64_t)count;
    nk_jacobi_eigenanalysis_f64_haswell_(moments, quaternion);
    nk_shapes_close_pack_f64_serial_(header, count, centroid, moments,
                                     nk_shapes_sqrt_f64_haswell_(nk_reduce_max_f64x4_haswell_(largest_f64x4)));
}

NUMKONG_API nk_status_t nk_shapes_pack_f16_haswell(nk_f16_t const *points, nk_size_t count, void *packed,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t centroid[3], covariance[6];
    nk_shapes_moments_f16_haswell_(points, count, centroid, covariance);
    nk_shapes_pack_f16_haswell_(points, count, centroid, covariance, 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f16_haswell(nk_f16_t const *points, nk_size_t count, void *packed,
                                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t centroid[3], covariance[6];
    nk_f32_t const radius = nk_shapes_moments_f16_haswell_(points, count, centroid, covariance);
    nk_shapes_pack_f16_haswell_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_f32_haswell(nk_f32_t const *points, nk_size_t count, void *packed,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t centroid[3], covariance[6];
    nk_shapes_moments_f32_haswell_(points, count, centroid, covariance);
    nk_shapes_pack_f32_haswell_(points, count, centroid, covariance, 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f32_haswell(nk_f32_t const *points, nk_size_t count, void *packed,
                                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t centroid[3], covariance[6];
    nk_f64_t const radius = nk_shapes_moments_f32_haswell_(points, count, centroid, covariance);
    nk_shapes_pack_f32_haswell_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f16_haswell(void const *const *a_packed, nk_size_t rows,
                                                       void const *const *b_packed, nk_size_t columns,
                                                       nk_f32_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                       nk_f32_t const *limit, void *workspace,
                                                       nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t const pair_limit = *limit;
    nk_f32_t chamfers[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_status_t const status = nk_shapes_chamfer_flips_f16_haswell_(a_packed[row], b_packed[column], pair_limit,
                                                                            workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f32_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_haswell(void const *const *a_packed, nk_size_t rows,
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
            nk_shapes_hausdorff_flips_f16_haswell_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f32_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f32_haswell(void const *const *a_packed, nk_size_t rows,
                                                       void const *const *b_packed, nk_size_t columns,
                                                       nk_f64_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                       nk_f64_t const *limit, void *workspace,
                                                       nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t const pair_limit = *limit;
    nk_f64_t chamfers[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_status_t const status = nk_shapes_chamfer_flips_f32_haswell_(a_packed[row], b_packed[column], pair_limit,
                                                                            workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f64_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_haswell(void const *const *a_packed, nk_size_t rows,
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
            nk_shapes_hausdorff_flips_f32_haswell_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f64_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_haswell(void const *const *a_packed, nk_size_t rows,
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
                results_row[column] = nk_sqrt_f32_serial_(nk_shapes_wasserstein_procrustes_f16_haswell_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F32_INF;
            if (rotations) nk_shapes_pair_rotation_f32_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_haswell(void const *const *a_packed, nk_size_t rows,
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
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f16_haswell_(
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

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_haswell(void const *const *a_packed, nk_size_t rows,
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
                results_row[column] = nk_sqrt_f64_serial_(nk_shapes_wasserstein_procrustes_f32_haswell_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F64_INF;
            if (rotations) nk_shapes_pair_rotation_f64_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_haswell(void const *const *a_packed, nk_size_t rows,
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
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f32_haswell_(
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

/** Σ‖xᵢ′‖² of a pack of @c f16 points, 8 points per step through the zero padding. */
NUMKONG_INLINE nk_f32_t nk_rmsds_norm_f16_haswell_(void const *packed) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)packed)->points;
    nk_size_t axis_stride;
    nk_f16_t const *axes = nk_shapes_packed_axes_f16_serial_(packed, &axis_stride);
    __m256 norm_f32x8 = _mm256_setzero_ps();
    for (int axis = 0; axis != 3; ++axis)
        for (nk_size_t index = 0; index < count; index += 8) {
            __m256 const values_f32x8 = _mm256_cvtph_ps(
                _mm_loadu_si128((__m128i const *)(axes + axis * axis_stride + index)));
            norm_f32x8 = _mm256_fmadd_ps(values_f32x8, values_f32x8, norm_f32x8);
        }
    return nk_reduce_add_f32x8_haswell_(norm_f32x8);
}

/** Superposed RMSD of one pair of @c f16 packs with Ga + Gb in @p norms_squared, M in @c f32. */
NUMKONG_INLINE nk_f32_t nk_rmsds_pair_f16_haswell_(void const *row_packed, void const *column_packed,
                                                   nk_f32_t norms_squared) {
    nk_size_t const count = ((nk_shapes_packed_header_f32_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f16_t const *a_axes = nk_shapes_packed_axes_f16_serial_(row_packed, &axis_stride);
    nk_f16_t const *b_axes = nk_shapes_packed_axes_f16_serial_(column_packed, &axis_stride);
    __m256 sums_f32x8[9], a_f32x8[3], b_f32x8[3];
    for (int cell = 0; cell != 9; ++cell) sums_f32x8[cell] = _mm256_setzero_ps();
    for (nk_size_t index = 0; index < count; index += 8) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f32x8[axis] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(a_axes + axis * axis_stride + index)));
            b_f32x8[axis] = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const *)(b_axes + axis * axis_stride + index)));
        }
        for (int row = 0; row != 3; ++row)
            for (int column = 0; column != 3; ++column)
                sums_f32x8[row * 3 + column] = _mm256_fmadd_ps(a_f32x8[row], b_f32x8[column],
                                                               sums_f32x8[row * 3 + column]);
    }
    nk_f32_t cross_covariance[9];
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = nk_reduce_add_f32x8_haswell_(sums_f32x8[cell]);
    nk_f32_t const residual = nk_rmsds_folded_f32_serial_(cross_covariance, norms_squared);
    return nk_shapes_sqrt_f32_haswell_(residual / (nk_f32_t)count);
}

/** Superposed RMSD of one pair of @c f32 packs: H in @c f64, then the residuals in plain @c f64,
 *  4 points per step through the zero padding. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f32_haswell_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f32_t const *a_axes = nk_shapes_packed_axes_f32_serial_(row_packed, &axis_stride);
    nk_f32_t const *b_axes = nk_shapes_packed_axes_f32_serial_(column_packed, &axis_stride);
    __m256d sums_f64x4[9], a_f64x4[3], b_f64x4[3];
    for (int cell = 0; cell != 9; ++cell) sums_f64x4[cell] = _mm256_setzero_pd();
    for (nk_size_t index = 0; index < count; index += 4) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f64x4[axis] = _mm256_cvtps_pd(_mm_loadu_ps(a_axes + axis * axis_stride + index));
            b_f64x4[axis] = _mm256_cvtps_pd(_mm_loadu_ps(b_axes + axis * axis_stride + index));
        }
        for (int column = 0; column != 3; ++column)
            for (int row = 0; row != 3; ++row)
                sums_f64x4[column * 3 + row] = _mm256_fmadd_pd(b_f64x4[column], a_f64x4[row],
                                                               sums_f64x4[column * 3 + row]);
    }
    nk_f64_t cross_covariance[9], rotation[9];
    for (int cell = 0; cell != 9; ++cell) cross_covariance[cell] = nk_reduce_add_f64x4_haswell_(sums_f64x4[cell]);
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    __m256d rotation_f64x4[9];
    for (int cell = 0; cell != 9; ++cell) rotation_f64x4[cell] = _mm256_set1_pd(rotation[cell]);
    __m256d sum_f64x4 = _mm256_setzero_pd();
    for (nk_size_t index = 0; index < count; index += 4) {
        for (int axis = 0; axis != 3; ++axis)
            b_f64x4[axis] = _mm256_cvtps_pd(_mm_loadu_ps(b_axes + axis * axis_stride + index));
        for (int axis = 0; axis != 3; ++axis) {
            __m256d residual_f64x4 = _mm256_cvtps_pd(_mm_loadu_ps(a_axes + axis * axis_stride + index));
            for (int column = 0; column != 3; ++column)
                residual_f64x4 = _mm256_fnmadd_pd(rotation_f64x4[axis * 3 + column], b_f64x4[column], residual_f64x4);
            sum_f64x4 = _mm256_fmadd_pd(residual_f64x4, residual_f64x4, sum_f64x4);
        }
    }
    return nk_shapes_sqrt_f64_haswell_(nk_reduce_add_f64x4_haswell_(sum_f64x4) / (nk_f64_t)count);
}

/** Superposed RMSD of one pair of @c f64 packs, like the @c f32 pair with H and the residuals
 *  compensated, 4 points per step through the zero padding. */
NUMKONG_INLINE nk_f64_t nk_rmsds_pair_f64_haswell_(void const *row_packed, void const *column_packed) {
    nk_size_t const count = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    if (!count) return 0;
    nk_size_t axis_stride;
    nk_f64_t const *a_axes = nk_shapes_packed_axes_f64_serial_(row_packed, &axis_stride);
    nk_f64_t const *b_axes = nk_shapes_packed_axes_f64_serial_(column_packed, &axis_stride);
    __m256d sums_f64x4[9], compensations_f64x4[9], a_f64x4[3], b_f64x4[3];
    for (int cell = 0; cell != 9; ++cell)
        sums_f64x4[cell] = _mm256_setzero_pd(), compensations_f64x4[cell] = _mm256_setzero_pd();
    for (nk_size_t index = 0; index < count; index += 4) {
        for (int axis = 0; axis != 3; ++axis) {
            a_f64x4[axis] = _mm256_loadu_pd(a_axes + axis * axis_stride + index);
            b_f64x4[axis] = _mm256_loadu_pd(b_axes + axis * axis_stride + index);
        }
        for (int column = 0; column != 3; ++column)
            for (int row = 0; row != 3; ++row)
                nk_dot2_f64x4_haswell_(&sums_f64x4[column * 3 + row], &compensations_f64x4[column * 3 + row],
                                       b_f64x4[column], a_f64x4[row]);
    }
    nk_f64_t cross_covariance[9], rotation[9];
    for (int cell = 0; cell != 9; ++cell)
        cross_covariance[cell] = nk_dot_stable_sum_f64x4_haswell_(sums_f64x4[cell], compensations_f64x4[cell]);
    nk_rmsds_rotation_f64_serial_(cross_covariance, rotation);
    __m256d rotation_f64x4[9];
    for (int cell = 0; cell != 9; ++cell) rotation_f64x4[cell] = _mm256_set1_pd(rotation[cell]);
    __m256d sum_f64x4 = _mm256_setzero_pd(), compensation_f64x4 = _mm256_setzero_pd();
    for (nk_size_t index = 0; index < count; index += 4) {
        for (int axis = 0; axis != 3; ++axis) b_f64x4[axis] = _mm256_loadu_pd(b_axes + axis * axis_stride + index);
        for (int axis = 0; axis != 3; ++axis) {
            __m256d residual_f64x4 = _mm256_loadu_pd(a_axes + axis * axis_stride + index);
            for (int column = 0; column != 3; ++column)
                residual_f64x4 = _mm256_fnmadd_pd(rotation_f64x4[axis * 3 + column], b_f64x4[column], residual_f64x4);
            nk_dot2_f64x4_haswell_(&sum_f64x4, &compensation_f64x4, residual_f64x4, residual_f64x4);
        }
    }
    return nk_shapes_sqrt_f64_haswell_(nk_dot_stable_sum_f64x4_haswell_(sum_f64x4, compensation_f64x4) /
                                       (nk_f64_t)count);
}

NUMKONG_API nk_status_t nk_rmsds_packed_f64_haswell(void const *const *a_packed, nk_size_t rows,
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
            results_row[column] = nk_rmsds_pair_f64_haswell_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f32_haswell(void const *const *a_packed, nk_size_t rows,
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
            results_row[column] = nk_rmsds_pair_f32_haswell_(a_packed[row], b_packed[column]);
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_rmsds_packed_f16_haswell(void const *const *a_packed, nk_size_t rows,
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
            row_norms[row - row_begin] = nk_rmsds_norm_f16_haswell_(a_packed[row]);
        for (nk_size_t column_begin = 0; column_begin < columns; column_begin += nk_rmsds_tile_frames_k) {
            nk_size_t const column_end = column_begin + nk_rmsds_tile_frames_k < columns
                                             ? column_begin + nk_rmsds_tile_frames_k
                                             : columns;
            for (nk_size_t column = column_begin; column != column_end; ++column)
                column_norms[column - column_begin] = nk_rmsds_norm_f16_haswell_(b_packed[column]);
            for (nk_size_t row = row_begin; row != row_end; ++row) {
                nk_f32_t *results_row = (nk_f32_t *)((char *)results + row * results_stride);
                for (nk_size_t column = column_begin; column != column_end; ++column)
                    results_row[column] = nk_rmsds_pair_f16_haswell_(
                        a_packed[row], b_packed[column],
                        row_norms[row - row_begin] + column_norms[column - column_begin]);
            }
        }
    }
    return nk_success_k;
}

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
#endif // NUMKONG_SHAPES_HASWELL_H
