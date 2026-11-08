/**
 *  @file include/numkong/shapes/skylake.h
 *  @author Ash Vardanian
 *  @date December 27, 2025
 *  @brief Many-to-many shape distances between 3D point clouds for Skylake, AVX-512.
 *
 *  @sa c/numkong/shapes.h
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
 *  _mm512_cvtps_ph        VCVTPS2PH (YMM, ZMM, I8)             5cy @ p05     4cy @ p01
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
 *  Chamfer scores both directions in that one sweep, keeping the inner cloud's minima in the
 *  workspace, whose 64-byte strides take full-width stores. Hausdorff runs each direction and stops
 *  once every flip passes the limit, so it reads only the points that decide a pair.
 *
 *  Packing permutes xyz triples out of two masked loads, widens them, and narrows each rotated
 *  register back into its axis, measuring the stored points in the same sweep.
 *
 *  The transport kinds run the serial simplex around tile pricing eight arcs at a time, masking
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
#include "numkong/scalar/haswell.h" // `nk_rsqrt_f32_haswell_`
#include "numkong/shapes/serial.h"  // `nk_shapes_best_flip_f32_serial_`, `nk_shapes_simplex_next_tile_serial_`
#include "numkong/reduce/haswell.h" // `nk_reduce_min_f32x8_haswell_`
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

NUMKONG_INLINE nk_f32_t nk_shapes_sqrt_f32_skylake_(nk_f32_t x) { return _mm_cvtss_f32(_mm_sqrt_ps(_mm_set_ss(x))); }
NUMKONG_INLINE nk_f64_t nk_shapes_sqrt_f64_skylake_(nk_f64_t x) { return _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(x))); }

/** Reciprocal square root: VRSQRT14SD refined by two Newton steps. Zero and infinity keep their
 *  exact estimates, which the steps would turn into NaNs. */
NUMKONG_INLINE nk_f64_t nk_shapes_rsqrt_f64_skylake_(nk_f64_t x) {
    __m128d const x_f64x2 = _mm_set_sd(x), half_f64x2 = _mm_set_sd(0.5), one_f64x2 = _mm_set_sd(1);
    __m128d const estimate_f64x2 = _mm_rsqrt14_sd(x_f64x2, x_f64x2);
    __m128d refined_f64x2 = estimate_f64x2;
    for (int step = 0; step != 2; ++step) {
        __m128d const error_f64x2 = _mm_fnmadd_sd(_mm_mul_sd(x_f64x2, refined_f64x2), refined_f64x2, one_f64x2);
        refined_f64x2 = _mm_fmadd_sd(_mm_mul_sd(half_f64x2, refined_f64x2), error_f64x2, refined_f64x2);
    }
    __mmask8 const ordered_m8 = _mm_cmp_sd_mask(refined_f64x2, refined_f64x2, _CMP_ORD_Q);
    return _mm_cvtsd_f64(_mm_mask_move_sd(estimate_f64x2, ordered_m8, estimate_f64x2, refined_f64x2));
}

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
                                                            __m512 *nearest_f32x16) {
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
        nearest_f32x16[flip] = _mm512_min_ps(nearest_f32x16[flip],
                                             _mm512_fmadd_ps(minus_z_f32x16, minus_z_f32x16, planar_f32x16[flip]));
        nearest_f32x16[flip + 4] = _mm512_min_ps(nearest_f32x16[flip + 4],
                                                 _mm512_fmadd_ps(plus_z_f32x16, plus_z_f32x16, planar_f32x16[flip]));
    }
}

/** @copydoc nk_shapes_lower_nearest_f32x16_skylake_ */
NUMKONG_INLINE void nk_shapes_lower_nearest_f64x8_skylake_(__m512d outer_x_f64x8, __m512d outer_y_f64x8,
                                                           __m512d outer_z_f64x8, __m512d inner_x_f64x8,
                                                           __m512d inner_y_f64x8, __m512d inner_z_f64x8,
                                                           __m512d *nearest_f64x8) {
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
        nearest_f64x8[flip] = _mm512_min_pd(nearest_f64x8[flip],
                                            _mm512_fmadd_pd(minus_z_f64x8, minus_z_f64x8, planar_f64x8[flip]));
        nearest_f64x8[flip + 4] = _mm512_min_pd(nearest_f64x8[flip + 4],
                                                _mm512_fmadd_pd(plus_z_f64x8, plus_z_f64x8, planar_f64x8[flip]));
    }
}

/** Lowers the 8 flips' nearest squares as @ref nk_shapes_lower_nearest_f32x16_skylake_ does, and
 *  the same squares into the inner points' @p reverse_minima , 8 axes @p reverse_stride apart. */
NUMKONG_INLINE void nk_shapes_lower_mutual_f32x16_skylake_(__m512 outer_x_f32x16, __m512 outer_y_f32x16,
                                                           __m512 outer_z_f32x16, __m512 inner_x_f32x16,
                                                           __m512 inner_y_f32x16, __m512 inner_z_f32x16,
                                                           __m512 *nearest_f32x16, nk_f32_t *reverse_minima,
                                                           nk_size_t reverse_stride) {
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
        __m512 const minus_squares_f32x16 = _mm512_fmadd_ps(minus_z_f32x16, minus_z_f32x16, planar_f32x16[flip]);
        __m512 const plus_squares_f32x16 = _mm512_fmadd_ps(plus_z_f32x16, plus_z_f32x16, planar_f32x16[flip]);
        nk_f32_t *minus_reverse = reverse_minima + flip * reverse_stride;
        nk_f32_t *plus_reverse = minus_reverse + 4 * reverse_stride;
        nearest_f32x16[flip] = _mm512_min_ps(nearest_f32x16[flip], minus_squares_f32x16);
        nearest_f32x16[flip + 4] = _mm512_min_ps(nearest_f32x16[flip + 4], plus_squares_f32x16);
        _mm512_storeu_ps(minus_reverse, _mm512_min_ps(minus_squares_f32x16, _mm512_loadu_ps(minus_reverse)));
        _mm512_storeu_ps(plus_reverse, _mm512_min_ps(plus_squares_f32x16, _mm512_loadu_ps(plus_reverse)));
    }
}

/** @copydoc nk_shapes_lower_mutual_f32x16_skylake_ */
NUMKONG_INLINE void nk_shapes_lower_mutual_f64x8_skylake_(__m512d outer_x_f64x8, __m512d outer_y_f64x8,
                                                          __m512d outer_z_f64x8, __m512d inner_x_f64x8,
                                                          __m512d inner_y_f64x8, __m512d inner_z_f64x8,
                                                          __m512d *nearest_f64x8, nk_f64_t *reverse_minima,
                                                          nk_size_t reverse_stride) {
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
        __m512d const minus_squares_f64x8 = _mm512_fmadd_pd(minus_z_f64x8, minus_z_f64x8, planar_f64x8[flip]);
        __m512d const plus_squares_f64x8 = _mm512_fmadd_pd(plus_z_f64x8, plus_z_f64x8, planar_f64x8[flip]);
        nk_f64_t *minus_reverse = reverse_minima + flip * reverse_stride;
        nk_f64_t *plus_reverse = minus_reverse + 4 * reverse_stride;
        nearest_f64x8[flip] = _mm512_min_pd(nearest_f64x8[flip], minus_squares_f64x8);
        nearest_f64x8[flip + 4] = _mm512_min_pd(nearest_f64x8[flip + 4], plus_squares_f64x8);
        _mm512_storeu_pd(minus_reverse, _mm512_min_pd(minus_squares_f64x8, _mm512_loadu_pd(minus_reverse)));
        _mm512_storeu_pd(plus_reverse, _mm512_min_pd(plus_squares_f64x8, _mm512_loadu_pd(plus_reverse)));
    }
}

/** One outer point's sweep of @c f16 inner points in f32 lanes, 16 at a time, the last register
 *  of a count off the multiple with an infinite x in the padded lanes past @p tail_m16 . */
NUMKONG_INLINE void nk_shapes_sweep_nearest_f16_skylake_(__m512 outer_x_f32x16, __m512 outer_y_f32x16,
                                                         __m512 outer_z_f32x16, nk_f16_t const *inner_axes,
                                                         nk_size_t inner_stride, nk_size_t inner_count,
                                                         __mmask16 tail_m16, __m512 *nearest_f32x16) {
    nk_size_t const sixteens_end = inner_count & ~(nk_size_t)15;
    nk_b512_vec_t inner_x_vec, inner_y_vec, inner_z_vec;
    for (nk_size_t inner_index = 0; inner_index != sixteens_end; inner_index += 16) {
        nk_load_f16x16_to_f32x16_skylake_(inner_axes + inner_index, &inner_x_vec);
        nk_load_f16x16_to_f32x16_skylake_(inner_axes + inner_stride + inner_index, &inner_y_vec);
        nk_load_f16x16_to_f32x16_skylake_(inner_axes + 2 * inner_stride + inner_index, &inner_z_vec);
        nk_shapes_lower_nearest_f32x16_skylake_(outer_x_f32x16, outer_y_f32x16, outer_z_f32x16, inner_x_vec.zmm_ps,
                                                inner_y_vec.zmm_ps, inner_z_vec.zmm_ps, nearest_f32x16);
    }
    if (sixteens_end == inner_count) return;
    nk_load_f16x16_to_f32x16_skylake_(inner_axes + inner_stride + sixteens_end, &inner_y_vec);
    nk_load_f16x16_to_f32x16_skylake_(inner_axes + 2 * inner_stride + sixteens_end, &inner_z_vec);
    __m512 const inner_x_f32x16 = _mm512_mask_cvtph_ps(
        _mm512_set1_ps(NUMKONG_F32_INF), tail_m16, _mm256_loadu_si256((__m256i const *)(inner_axes + sixteens_end)));
    nk_shapes_lower_nearest_f32x16_skylake_(outer_x_f32x16, outer_y_f32x16, outer_z_f32x16, inner_x_f32x16,
                                            inner_y_vec.zmm_ps, inner_z_vec.zmm_ps, nearest_f32x16);
}

/** @ref nk_shapes_sweep_nearest_f16_skylake_ that also lowers the views' reverse minima. */
NUMKONG_INLINE void nk_shapes_sweep_mutual_f16_skylake_(__m512 outer_x_f32x16, __m512 outer_y_f32x16,
                                                        __m512 outer_z_f32x16, nk_shapes_mutual_f16_t const *views,
                                                        __mmask16 tail_m16, __m512 *nearest_f32x16) {
    nk_f16_t const *inner_axes = views->inner_axes;
    nk_size_t const inner_stride = views->inner_stride, inner_count = views->inner_count;
    nk_size_t const sixteens_end = inner_count & ~(nk_size_t)15;
    nk_b512_vec_t inner_x_vec, inner_y_vec, inner_z_vec;
    for (nk_size_t inner_index = 0; inner_index != sixteens_end; inner_index += 16) {
        nk_load_f16x16_to_f32x16_skylake_(inner_axes + inner_index, &inner_x_vec);
        nk_load_f16x16_to_f32x16_skylake_(inner_axes + inner_stride + inner_index, &inner_y_vec);
        nk_load_f16x16_to_f32x16_skylake_(inner_axes + 2 * inner_stride + inner_index, &inner_z_vec);
        nk_shapes_lower_mutual_f32x16_skylake_(outer_x_f32x16, outer_y_f32x16, outer_z_f32x16, inner_x_vec.zmm_ps,
                                               inner_y_vec.zmm_ps, inner_z_vec.zmm_ps, nearest_f32x16,
                                               views->reverse_minima + inner_index, views->reverse_stride);
    }
    if (sixteens_end == inner_count) return;
    nk_load_f16x16_to_f32x16_skylake_(inner_axes + inner_stride + sixteens_end, &inner_y_vec);
    nk_load_f16x16_to_f32x16_skylake_(inner_axes + 2 * inner_stride + sixteens_end, &inner_z_vec);
    __m512 const inner_x_f32x16 = _mm512_mask_cvtph_ps(
        _mm512_set1_ps(NUMKONG_F32_INF), tail_m16, _mm256_loadu_si256((__m256i const *)(inner_axes + sixteens_end)));
    nk_shapes_lower_mutual_f32x16_skylake_(outer_x_f32x16, outer_y_f32x16, outer_z_f32x16, inner_x_f32x16,
                                           inner_y_vec.zmm_ps, inner_z_vec.zmm_ps, nearest_f32x16,
                                           views->reverse_minima + sixteens_end, views->reverse_stride);
}

/** One outer point's sweep of @c f32 inner points in f64 lanes, 8 at a time, the last register of a
 *  count off the multiple with an infinite x in the padded lanes past @p tail_m8 . */
NUMKONG_INLINE void nk_shapes_sweep_nearest_f32_skylake_(__m512d outer_x_f64x8, __m512d outer_y_f64x8,
                                                         __m512d outer_z_f64x8, nk_f32_t const *inner_axes,
                                                         nk_size_t inner_stride, nk_size_t inner_count,
                                                         __mmask8 tail_m8, __m512d *nearest_f64x8) {
    nk_size_t const octuples_end = inner_count & ~(nk_size_t)7;
    for (nk_size_t inner_index = 0; inner_index != octuples_end; inner_index += 8)
        nk_shapes_lower_nearest_f64x8_skylake_(
            outer_x_f64x8, outer_y_f64x8, outer_z_f64x8, _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + inner_index)),
            _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + inner_stride + inner_index)),
            _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + 2 * inner_stride + inner_index)), nearest_f64x8);
    if (octuples_end == inner_count) return;
    nk_shapes_lower_nearest_f64x8_skylake_(
        outer_x_f64x8, outer_y_f64x8, outer_z_f64x8,
        _mm512_mask_cvtps_pd(_mm512_set1_pd(NUMKONG_F64_INF), tail_m8, _mm256_loadu_ps(inner_axes + octuples_end)),
        _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + inner_stride + octuples_end)),
        _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + 2 * inner_stride + octuples_end)), nearest_f64x8);
}

/** @ref nk_shapes_sweep_nearest_f32_skylake_ that also lowers the views' reverse minima. */
NUMKONG_INLINE void nk_shapes_sweep_mutual_f32_skylake_(__m512d outer_x_f64x8, __m512d outer_y_f64x8,
                                                        __m512d outer_z_f64x8, nk_shapes_mutual_f32_t const *views,
                                                        __mmask8 tail_m8, __m512d *nearest_f64x8) {
    nk_f32_t const *inner_axes = views->inner_axes;
    nk_size_t const inner_stride = views->inner_stride, inner_count = views->inner_count;
    nk_size_t const octuples_end = inner_count & ~(nk_size_t)7;
    for (nk_size_t inner_index = 0; inner_index != octuples_end; inner_index += 8)
        nk_shapes_lower_mutual_f64x8_skylake_(
            outer_x_f64x8, outer_y_f64x8, outer_z_f64x8, _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + inner_index)),
            _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + inner_stride + inner_index)),
            _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + 2 * inner_stride + inner_index)), nearest_f64x8,
            views->reverse_minima + inner_index, views->reverse_stride);
    if (octuples_end == inner_count) return;
    nk_shapes_lower_mutual_f64x8_skylake_(
        outer_x_f64x8, outer_y_f64x8, outer_z_f64x8,
        _mm512_mask_cvtps_pd(_mm512_set1_pd(NUMKONG_F64_INF), tail_m8, _mm256_loadu_ps(inner_axes + octuples_end)),
        _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + inner_stride + octuples_end)),
        _mm512_cvtps_pd(_mm256_loadu_ps(inner_axes + 2 * inner_stride + octuples_end)), nearest_f64x8,
        views->reverse_minima + octuples_end, views->reverse_stride);
}

/** The roots of @p count reverse minima summed, leaving out the padded lanes past @p tail_m16 . */
NUMKONG_INLINE nk_f32_t nk_shapes_reverse_sum_f32_skylake_(nk_f32_t const *reverse, nk_size_t count,
                                                           __mmask16 tail_m16) {
    nk_size_t const sixteens_end = count & ~(nk_size_t)15;
    __m512 sum_f32x16 = _mm512_setzero_ps();
    for (nk_size_t index = 0; index != sixteens_end; index += 16)
        sum_f32x16 = _mm512_add_ps(sum_f32x16, _mm512_sqrt_ps(_mm512_loadu_ps(reverse + index)));
    if (tail_m16)
        sum_f32x16 = _mm512_add_ps(sum_f32x16, _mm512_maskz_sqrt_ps(tail_m16, _mm512_loadu_ps(reverse + sixteens_end)));
    return nk_reduce_add_f32x16_skylake_(sum_f32x16);
}

/** The roots of @p count reverse minima summed, leaving out the padded lanes past @p tail_m8 . */
NUMKONG_INLINE nk_f64_t nk_shapes_reverse_sum_f64_skylake_(nk_f64_t const *reverse, nk_size_t count, __mmask8 tail_m8) {
    nk_size_t const octuples_end = count & ~(nk_size_t)7;
    __m512d sum_f64x8 = _mm512_setzero_pd();
    for (nk_size_t index = 0; index != octuples_end; index += 8)
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_sqrt_pd(_mm512_loadu_pd(reverse + index)));
    if (tail_m8)
        sum_f64x8 = _mm512_add_pd(sum_f64x8, _mm512_maskz_sqrt_pd(tail_m8, _mm512_loadu_pd(reverse + octuples_end)));
    return nk_reduce_add_f64x8_skylake_(sum_f64x8);
}

/** Fused Chamfer flips of @c f16 packs in f32 lanes: each outer point's sweep lowers its own minima
 *  and the inner points' reverse ones, and the 8 sums stay in one register. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f16_skylake_(void const *row_packed, void const *column_packed,
                                                                nk_f32_t limit, void *workspace,
                                                                nk_size_t workspace_bytes, nk_f32_t *chamfers) {
    nk_shapes_mutual_f16_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f16_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    __m512 const infinity_f32x16 = _mm512_set1_ps(NUMKONG_F32_INF);
    for (nk_size_t index = 0; index < 8 * views.reverse_stride; index += 16)
        _mm512_storeu_ps(views.reverse_minima + index, infinity_f32x16);
    __mmask16 const tail_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(views.inner_count & 15));
    nk_f32_t const outer_points = (nk_f32_t)views.outer_count, inner_points = (nk_f32_t)views.inner_count;
    __m256 const threshold_f32x8 = _mm256_set1_ps(2 * limit * outer_points);
    __m256 sums_f32x8 = _mm256_setzero_ps();
    __m512 nearest_f32x16[8];
    nk_f32_t sums[8], reverse_sums[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        nk_f16_t const *outer_point = views.outer_axes + outer_index;
        __m512 const outer_x_f32x16 = _mm512_cvtph_ps(_mm256_set1_epi16(*(nk_i16_t const *)outer_point));
        __m512 const outer_y_f32x16 = _mm512_cvtph_ps(
            _mm256_set1_epi16(*(nk_i16_t const *)(outer_point + views.outer_stride)));
        __m512 const outer_z_f32x16 = _mm512_cvtph_ps(
            _mm256_set1_epi16(*(nk_i16_t const *)(outer_point + 2 * views.outer_stride)));
        for (int flip = 0; flip != 8; ++flip) nearest_f32x16[flip] = infinity_f32x16;
        nk_shapes_sweep_mutual_f16_skylake_(outer_x_f32x16, outer_y_f32x16, outer_z_f32x16, &views, tail_m16,
                                            nearest_f32x16);
        sums_f32x8 = _mm256_add_ps(sums_f32x8, _mm256_sqrt_ps(nk_shapes_flip_minima_f32x16_skylake_(nearest_f32x16)));
        if (_mm256_cmp_ps_mask(sums_f32x8, threshold_f32x8, _CMP_GT_OQ) == 0xFF) {
            _mm256_storeu_ps(sums, sums_f32x8);
            if (nk_shapes_mutual_past_limit_f32_serial_(sums, outer_points, limit, chamfers)) return nk_success_k;
        }
    }
    for (int flip = 0; flip != 8; ++flip)
        reverse_sums[flip] = nk_shapes_reverse_sum_f32_skylake_(views.reverse_minima + flip * views.reverse_stride,
                                                                views.inner_count, tail_m16);
    __m256 const outer_mean_f32x8 = _mm256_div_ps(sums_f32x8, _mm256_set1_ps(outer_points));
    __m256 const inner_mean_f32x8 = _mm256_div_ps(_mm256_loadu_ps(reverse_sums), _mm256_set1_ps(inner_points));
    _mm256_storeu_ps(chamfers, _mm256_mul_ps(_mm256_add_ps(outer_mean_f32x8, inner_mean_f32x8), _mm256_set1_ps(0.5f)));
    return nk_success_k;
}

/** Hausdorff flips of @c f16 packs in f32 lanes: a pair stops once every flip's farthest square
 *  passes @p limit , whose square rounds, so the distance confirms it. */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f16_skylake_(void const *row_packed, void const *column_packed,
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
    __mmask16 const row_tail_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(rows & 15));
    __mmask16 const column_tail_m16 = (__mmask16)_bzhi_u32(0xFFFF, (unsigned int)(columns & 15));
    __m512 const infinity_f32x16 = _mm512_set1_ps(NUMKONG_F32_INF);
    __m256 const limit_squared_f32x8 = _mm256_set1_ps(limit * limit);
    __m256 farthest_f32x8 = _mm256_setzero_ps();
    __m512 nearest_f32x16[8];
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_f16_t const *outer_axes = direction ? column_axes : row_axes,
                       *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        __mmask16 const tail_m16 = direction ? row_tail_m16 : column_tail_m16;
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            nk_f16_t const *outer_point = outer_axes + outer_index;
            __m512 const outer_x_f32x16 = _mm512_cvtph_ps(_mm256_set1_epi16(*(nk_i16_t const *)outer_point));
            __m512 const outer_y_f32x16 = _mm512_cvtph_ps(
                _mm256_set1_epi16(*(nk_i16_t const *)(outer_point + outer_stride)));
            __m512 const outer_z_f32x16 = _mm512_cvtph_ps(
                _mm256_set1_epi16(*(nk_i16_t const *)(outer_point + 2 * outer_stride)));
            for (int flip = 0; flip != 8; ++flip) nearest_f32x16[flip] = infinity_f32x16;
            nk_shapes_sweep_nearest_f16_skylake_(outer_x_f32x16, outer_y_f32x16, outer_z_f32x16, inner_axes,
                                                 inner_stride, inner_count, tail_m16, nearest_f32x16);
            farthest_f32x8 = _mm256_max_ps(farthest_f32x8, nk_shapes_flip_minima_f32x16_skylake_(nearest_f32x16));
            if (_mm256_cmp_ps_mask(farthest_f32x8, limit_squared_f32x8, _CMP_GT_OQ) == 0xFF)
                rejected = nk_shapes_sqrt_f32_skylake_(nk_reduce_min_f32x8_haswell_(farthest_f32x8)) > limit;
        }
    }
    _mm256_storeu_ps(hausdorffs, _mm256_sqrt_ps(farthest_f32x8));
}

/** Fused Chamfer flips of @c f32 packs in f64 lanes, the 8 sums in one register. */
NUMKONG_INLINE nk_status_t nk_shapes_chamfer_flips_f32_skylake_(void const *row_packed, void const *column_packed,
                                                                nk_f64_t limit, void *workspace,
                                                                nk_size_t workspace_bytes, nk_f64_t *chamfers) {
    nk_shapes_mutual_f32_t views;
    nk_status_t const status = nk_shapes_mutual_begin_f32_serial_(row_packed, column_packed, workspace, workspace_bytes,
                                                                  chamfers, &views);
    if (status != nk_success_k || !views.outer_count) return status;
    __m512d const infinity_f64x8 = _mm512_set1_pd(NUMKONG_F64_INF);
    for (nk_size_t index = 0; index < 8 * views.reverse_stride; index += 8)
        _mm512_storeu_pd(views.reverse_minima + index, infinity_f64x8);
    __mmask8 const tail_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)(views.inner_count & 7));
    nk_f64_t const outer_points = (nk_f64_t)views.outer_count, inner_points = (nk_f64_t)views.inner_count;
    __m512d const threshold_f64x8 = _mm512_set1_pd(2 * limit * outer_points);
    __m512d sums_f64x8 = _mm512_setzero_pd(), nearest_f64x8[8];
    nk_f64_t sums[8], reverse_sums[8];
    for (nk_size_t outer_index = 0; outer_index != views.outer_count; ++outer_index) {
        nk_f32_t const *outer_point = views.outer_axes + outer_index;
        __m512d const outer_x_f64x8 = _mm512_set1_pd(outer_point[0]);
        __m512d const outer_y_f64x8 = _mm512_set1_pd(outer_point[views.outer_stride]);
        __m512d const outer_z_f64x8 = _mm512_set1_pd(outer_point[2 * views.outer_stride]);
        for (int flip = 0; flip != 8; ++flip) nearest_f64x8[flip] = infinity_f64x8;
        nk_shapes_sweep_mutual_f32_skylake_(outer_x_f64x8, outer_y_f64x8, outer_z_f64x8, &views, tail_m8,
                                            nearest_f64x8);
        sums_f64x8 = _mm512_add_pd(sums_f64x8, _mm512_sqrt_pd(nk_shapes_flip_minima_f64x8_skylake_(nearest_f64x8)));
        if (_mm512_cmp_pd_mask(sums_f64x8, threshold_f64x8, _CMP_GT_OQ) == 0xFF) {
            _mm512_storeu_pd(sums, sums_f64x8);
            if (nk_shapes_mutual_past_limit_f64_serial_(sums, outer_points, limit, chamfers)) return nk_success_k;
        }
    }
    for (int flip = 0; flip != 8; ++flip)
        reverse_sums[flip] = nk_shapes_reverse_sum_f64_skylake_(views.reverse_minima + flip * views.reverse_stride,
                                                                views.inner_count, tail_m8);
    __m512d const outer_mean_f64x8 = _mm512_div_pd(sums_f64x8, _mm512_set1_pd(outer_points));
    __m512d const inner_mean_f64x8 = _mm512_div_pd(_mm512_loadu_pd(reverse_sums), _mm512_set1_pd(inner_points));
    _mm512_storeu_pd(chamfers, _mm512_mul_pd(_mm512_add_pd(outer_mean_f64x8, inner_mean_f64x8), _mm512_set1_pd(0.5)));
    return nk_success_k;
}

/** Hausdorff flips of @c f32 packs in f64 lanes, stopping as the @c f16 pair does. */
NUMKONG_INLINE void nk_shapes_hausdorff_flips_f32_skylake_(void const *row_packed, void const *column_packed,
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
    __mmask8 const row_tail_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)(rows & 7));
    __mmask8 const column_tail_m8 = (__mmask8)_bzhi_u32(0xFF, (unsigned int)(columns & 7));
    __m512d const infinity_f64x8 = _mm512_set1_pd(NUMKONG_F64_INF);
    __m512d const limit_squared_f64x8 = _mm512_set1_pd(limit * limit);
    __m512d farthest_f64x8 = _mm512_setzero_pd(), nearest_f64x8[8];
    int rejected = 0;
    for (int direction = 0; direction != 2 && !rejected; ++direction) {
        nk_f32_t const *outer_axes = direction ? column_axes : row_axes,
                       *inner_axes = direction ? row_axes : column_axes;
        nk_size_t const outer_stride = direction ? column_stride : row_stride;
        nk_size_t const inner_stride = direction ? row_stride : column_stride;
        nk_size_t const outer_count = direction ? columns : rows, inner_count = direction ? rows : columns;
        __mmask8 const tail_m8 = direction ? row_tail_m8 : column_tail_m8;
        for (nk_size_t outer_index = 0; outer_index != outer_count && !rejected; ++outer_index) {
            nk_f32_t const *outer_point = outer_axes + outer_index;
            for (int flip = 0; flip != 8; ++flip) nearest_f64x8[flip] = infinity_f64x8;
            nk_shapes_sweep_nearest_f32_skylake_(_mm512_set1_pd(outer_point[0]),
                                                 _mm512_set1_pd(outer_point[outer_stride]),
                                                 _mm512_set1_pd(outer_point[2 * outer_stride]), inner_axes,
                                                 inner_stride, inner_count, tail_m8, nearest_f64x8);
            farthest_f64x8 = _mm512_max_pd(farthest_f64x8, nk_shapes_flip_minima_f64x8_skylake_(nearest_f64x8));
            if (_mm512_cmp_pd_mask(farthest_f64x8, limit_squared_f64x8, _CMP_GT_OQ) == 0xFF)
                rejected = nk_shapes_sqrt_f64_skylake_(nk_reduce_min_f64x8_skylake_(farthest_f64x8)) > limit;
        }
    }
    _mm512_storeu_pd(hausdorffs, _mm512_sqrt_pd(farthest_f64x8));
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
        __m512d nearest_f64x8[8];
        for (int flip = 0; flip != 8; ++flip) nearest_f64x8[flip] = infinity_f64x8;
        for (nk_size_t inner_index = 0; inner_index != octuples_end; inner_index += 8)
            nk_shapes_lower_nearest_f64x8_skylake_(
                outer_x_f64x8, outer_y_f64x8, outer_z_f64x8, _mm512_loadu_pd(inner_axes + inner_index),
                _mm512_loadu_pd(inner_axes + inner_stride + inner_index),
                _mm512_loadu_pd(inner_axes + 2 * inner_stride + inner_index), nearest_f64x8);
        if (octuples_end != inner_count)
            nk_shapes_lower_nearest_f64x8_skylake_(
                outer_x_f64x8, outer_y_f64x8, outer_z_f64x8,
                _mm512_mask_loadu_pd(infinity_f64x8, tail_m8, inner_axes + octuples_end),
                _mm512_loadu_pd(inner_axes + inner_stride + octuples_end),
                _mm512_loadu_pd(inner_axes + 2 * inner_stride + octuples_end), nearest_f64x8);
        __m512d const distances_f64x8 = _mm512_sqrt_pd(nk_shapes_flip_minima_f64x8_skylake_(nearest_f64x8));
        nk_dot2_f64x8_skylake_(&sums_f64x8, &compensations_f64x8, distances_f64x8, one_f64x8);
        maxima_f64x8 = _mm512_max_pd(maxima_f64x8, distances_f64x8);
    }
    _mm512_storeu_pd(sums, _mm512_add_pd(sums_f64x8, compensations_f64x8));
    _mm512_storeu_pd(maxima, maxima_f64x8);
}

/** Squared distances from a broadcast row point in fused multiply-adds, which also serve single
 *  arcs, so that tree arcs keep a zero reduced cost. */
NUMKONG_INLINE __m256 nk_shapes_point_costs_f32x8_skylake_(__m256 row_x_f32x8, __m256 row_y_f32x8, __m256 row_z_f32x8,
                                                           __m256 column_x_f32x8, __m256 column_y_f32x8,
                                                           __m256 column_z_f32x8) {
    __m256 const delta_x_f32x8 = _mm256_sub_ps(row_x_f32x8, column_x_f32x8);
    __m256 const delta_y_f32x8 = _mm256_sub_ps(row_y_f32x8, column_y_f32x8);
    __m256 const delta_z_f32x8 = _mm256_sub_ps(row_z_f32x8, column_z_f32x8);
    return _mm256_fmadd_ps(delta_z_f32x8, delta_z_f32x8,
                           _mm256_fmadd_ps(delta_y_f32x8, delta_y_f32x8, _mm256_mul_ps(delta_x_f32x8, delta_x_f32x8)));
}

NUMKONG_INLINE nk_f32_t nk_shapes_point_cost_f32_skylake_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t row,
                                                          nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f32_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    return _mm256_cvtss_f32(nk_shapes_point_costs_f32x8_skylake_(
        _mm256_set1_ps(row_points[row]), _mm256_set1_ps(row_points[stride + row]),
        _mm256_set1_ps(row_points[2 * stride + row]), _mm256_set1_ps(rotated_points[column]),
        _mm256_set1_ps(rotated_points[stride + column]), _mm256_set1_ps(rotated_points[2 * stride + column])));
}

/** The reduced costs (c − πᵢ) + πⱼ of a row's arcs to 8 columns, with matrix costs −G taken as
 *  πⱼ − (G + πᵢ), which rounds the same. */
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

/** Keeps a tile's first least reduced cost below @p entering_cost, 8 columns at a time, branching
 *  only where a register beats it. Masks keep lanes past the tile out of the comparisons. */
NUMKONG_INLINE void nk_shapes_price_tile_f32_skylake_(nk_shapes_workspace_f32_serial_t const *views,
                                                      nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                      nk_size_t row, nk_size_t column, nk_size_t tile,
                                                      nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                      nk_size_t *entering_column) {
    __m512d const infinity_f64x8 = _mm512_set1_pd(NUMKONG_F64_INF);
    __m512d least_f64x8 = _mm512_set1_pd(*entering_cost);
    for (nk_size_t step = 0; step < tile; step += 8) {
        nk_size_t const remaining = tile - step;
        __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, remaining < 8 ? (unsigned int)remaining : 8);
        __m512d const reduced_f64x8 = nk_shapes_reduced_costs_f32_skylake_(views, gradient, rows, columns, row,
                                                                           column + step, valid_m8);
        if (!_mm512_mask_cmp_pd_mask(valid_m8, reduced_f64x8, least_f64x8, _CMP_LT_OQ)) continue;
        *entering_cost = nk_reduce_min_f64x8_skylake_(_mm512_mask_mov_pd(infinity_f64x8, valid_m8, reduced_f64x8));
        least_f64x8 = _mm512_set1_pd(*entering_cost);
        __mmask8 const equal_m8 = _mm512_mask_cmp_pd_mask(valid_m8, reduced_f64x8, least_f64x8, _CMP_EQ_OQ);
        *entering_row = row, *entering_column = column + step + _tzcnt_u32(equal_m8);
    }
}

NUMKONG_INLINE void nk_shapes_tree_costs_f32_skylake_(nk_shapes_workspace_f32_serial_t const *views,
                                                      nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        views->potential[node] = gradient ? -gradient[row * columns + column]
                                          : (nk_f64_t)nk_shapes_point_cost_f32_skylake_(views, row, column);
    }
}

NUMKONG_INLINE void nk_shapes_network_simplex_f32_skylake_(nk_shapes_workspace_f32_serial_t const *views,
                                                           nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                           nk_f64_t cost_scale) {
    nk_shapes_simplex_serial_t simplex;
    nk_shapes_tree_costs_f32_skylake_(views, gradient, rows, columns);
    nk_shapes_tree_potentials_serial_(views->tree, views->tree_stride, views->potential, rows);
    nk_shapes_simplex_start_serial_(&simplex, views->tree, views->tree_stride, views->potential, rows, columns,
                                    cost_scale);
    while (nk_shapes_simplex_next_tile_serial_(&simplex))
        nk_shapes_price_tile_f32_skylake_(views, gradient, rows, columns, simplex.row, simplex.column, simplex.tile,
                                          &simplex.entering_cost, &simplex.entering_row, &simplex.entering_column);
}

NUMKONG_INLINE nk_f32_t nk_shapes_tree_cost_f32_skylake_(nk_shapes_workspace_f32_serial_t const *views, nk_size_t rows,
                                                         nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f32_t cost = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        cost += (nk_f32_t)flow[node] * nk_shapes_point_cost_f32_skylake_(views, row, column);
    }
    return cost / (nk_f32_t)(rows * columns);
}

NUMKONG_INLINE __m512d nk_shapes_point_costs_f64x8_skylake_(__m512d row_x_f64x8, __m512d row_y_f64x8,
                                                            __m512d row_z_f64x8, __m512d column_x_f64x8,
                                                            __m512d column_y_f64x8, __m512d column_z_f64x8) {
    __m512d const delta_x_f64x8 = _mm512_sub_pd(row_x_f64x8, column_x_f64x8);
    __m512d const delta_y_f64x8 = _mm512_sub_pd(row_y_f64x8, column_y_f64x8);
    __m512d const delta_z_f64x8 = _mm512_sub_pd(row_z_f64x8, column_z_f64x8);
    return _mm512_fmadd_pd(delta_z_f64x8, delta_z_f64x8,
                           _mm512_fmadd_pd(delta_y_f64x8, delta_y_f64x8, _mm512_mul_pd(delta_x_f64x8, delta_x_f64x8)));
}

NUMKONG_INLINE nk_f64_t nk_shapes_point_cost_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t row,
                                                          nk_size_t column) {
    nk_size_t const stride = views->points_stride;
    nk_f64_t const *row_points = views->row_points, *rotated_points = views->rotated_points;
    return _mm512_cvtsd_f64(nk_shapes_point_costs_f64x8_skylake_(
        _mm512_set1_pd(row_points[row]), _mm512_set1_pd(row_points[stride + row]),
        _mm512_set1_pd(row_points[2 * stride + row]), _mm512_set1_pd(rotated_points[column]),
        _mm512_set1_pd(rotated_points[stride + column]), _mm512_set1_pd(rotated_points[2 * stride + column])));
}

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

NUMKONG_INLINE void nk_shapes_price_tile_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views,
                                                      nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                      nk_size_t row, nk_size_t column, nk_size_t tile,
                                                      nk_f64_t *entering_cost, nk_size_t *entering_row,
                                                      nk_size_t *entering_column) {
    __m512d const infinity_f64x8 = _mm512_set1_pd(NUMKONG_F64_INF);
    __m512d least_f64x8 = _mm512_set1_pd(*entering_cost);
    for (nk_size_t step = 0; step < tile; step += 8) {
        nk_size_t const remaining = tile - step;
        __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, remaining < 8 ? (unsigned int)remaining : 8);
        __m512d const reduced_f64x8 = nk_shapes_reduced_costs_f64_skylake_(views, gradient, rows, columns, row,
                                                                           column + step, valid_m8);
        if (!_mm512_mask_cmp_pd_mask(valid_m8, reduced_f64x8, least_f64x8, _CMP_LT_OQ)) continue;
        *entering_cost = nk_reduce_min_f64x8_skylake_(_mm512_mask_mov_pd(infinity_f64x8, valid_m8, reduced_f64x8));
        least_f64x8 = _mm512_set1_pd(*entering_cost);
        __mmask8 const equal_m8 = _mm512_mask_cmp_pd_mask(valid_m8, reduced_f64x8, least_f64x8, _CMP_EQ_OQ);
        *entering_row = row, *entering_column = column + step + _tzcnt_u32(equal_m8);
    }
}

NUMKONG_INLINE void nk_shapes_tree_costs_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views,
                                                      nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        views->potential[node] = gradient ? -gradient[row * columns + column]
                                          : nk_shapes_point_cost_f64_skylake_(views, row, column);
    }
}

NUMKONG_INLINE void nk_shapes_network_simplex_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views,
                                                           nk_f64_t const *gradient, nk_size_t rows, nk_size_t columns,
                                                           nk_f64_t cost_scale) {
    nk_shapes_simplex_serial_t simplex;
    nk_shapes_tree_costs_f64_skylake_(views, gradient, rows, columns);
    nk_shapes_tree_potentials_serial_(views->tree, views->tree_stride, views->potential, rows);
    nk_shapes_simplex_start_serial_(&simplex, views->tree, views->tree_stride, views->potential, rows, columns,
                                    cost_scale);
    while (nk_shapes_simplex_next_tile_serial_(&simplex))
        nk_shapes_price_tile_f64_skylake_(views, gradient, rows, columns, simplex.row, simplex.column, simplex.tile,
                                          &simplex.entering_cost, &simplex.entering_row, &simplex.entering_column);
}

NUMKONG_INLINE nk_f64_t nk_shapes_tree_cost_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views, nk_size_t rows,
                                                         nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f64_t cost = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        cost += (nk_f64_t)flow[node] * nk_shapes_point_cost_f64_skylake_(views, row, column);
    }
    return cost / (nk_f64_t)(rows * columns);
}

NUMKONG_INLINE nk_f32_t nk_shapes_wasserstein_procrustes_f16_skylake_(nk_shapes_workspace_f32_serial_t const *views,
                                                                      nk_size_t rows, nk_size_t columns,
                                                                      nk_f64_t cost_scale, nk_f32_t *rotation) {
    nk_shapes_procrustes_f32_serial_t search;
    nk_shapes_procrustes_start_f32_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_f32_serial_(&search)) {
        nk_shapes_network_simplex_f32_skylake_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_f32_skylake_(views, rows, columns);
    }
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f16_skylake_(nk_shapes_workspace_f32_serial_t const *views,
                                                                  nk_size_t rows, nk_size_t columns,
                                                                  nk_f64_t cost_scale, nk_f64_t const *row_distances,
                                                                  nk_f64_t const *column_distances,
                                                                  nk_f32_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f16_skylake_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f32_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_serial_(&search);
    while (nk_shapes_gromov_next_f32_serial_(&search, views, rotation))
        nk_shapes_network_simplex_f32_skylake_(views, search.gradient, rows, columns, search.gradient_scale);
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_wasserstein_procrustes_f32_skylake_(nk_shapes_workspace_f64_serial_t const *views,
                                                                      nk_size_t rows, nk_size_t columns,
                                                                      nk_f64_t cost_scale, nk_f64_t *rotation) {
    nk_shapes_procrustes_f64_serial_t search;
    nk_shapes_procrustes_start_f64_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_f64_serial_(&search)) {
        nk_shapes_network_simplex_f64_skylake_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_f64_skylake_(views, rows, columns);
    }
    return search.best;
}

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f32_skylake_(nk_shapes_workspace_f64_serial_t const *views,
                                                                  nk_size_t rows, nk_size_t columns,
                                                                  nk_f64_t cost_scale, nk_f64_t const *row_distances,
                                                                  nk_f64_t const *column_distances,
                                                                  nk_f64_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f32_skylake_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f64_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_serial_(&search);
    while (nk_shapes_gromov_next_f64_serial_(&search, views, rotation))
        nk_shapes_network_simplex_f64_skylake_(views, search.gradient, rows, columns, search.gradient_scale);
    return search.best;
}

/** The mean cost of the plan in a tree with the sum compensated, for @c f64 clouds, rounding as
 *  the priced lanes do. */
NUMKONG_INLINE nk_f64_t nk_shapes_tree_cost_compensated_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views,
                                                                     nk_size_t rows, nk_size_t columns) {
    nk_u32_t const *parent = views->tree, *flow = views->tree + views->tree_stride;
    nk_f64_t cost = 0, compensation = 0;
    for (nk_size_t node = 1; node != rows + columns; ++node) {
        nk_size_t const row = node < rows ? node : parent[node];
        nk_size_t const column = (node < rows ? parent[node] : node) - rows;
        nk_dot2_f64_serial_(&cost, &compensation, (nk_f64_t)flow[node],
                            nk_shapes_point_cost_f64_skylake_(views, row, column));
    }
    return (cost + compensation) / (nk_f64_t)(rows * columns);
}

NUMKONG_INLINE nk_f64_t nk_shapes_wasserstein_procrustes_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views,
                                                                      nk_size_t rows, nk_size_t columns,
                                                                      nk_f64_t cost_scale, nk_f64_t *rotation) {
    nk_shapes_procrustes_f64_serial_t search;
    nk_shapes_procrustes_start_f64_serial_(&search, views, rows, columns, rotation);
    while (nk_shapes_procrustes_next_compensated_f64_serial_(&search)) {
        nk_shapes_network_simplex_f64_skylake_(views, NUMKONG_NULL, rows, columns, cost_scale);
        search.cost = nk_shapes_tree_cost_compensated_f64_skylake_(views, rows, columns);
    }
    return search.best;
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

NUMKONG_INLINE nk_f64_t nk_shapes_gromov_wasserstein_f64_skylake_(nk_shapes_workspace_f64_serial_t const *views,
                                                                  nk_size_t rows, nk_size_t columns,
                                                                  nk_f64_t cost_scale, nk_f64_t const *row_distances,
                                                                  nk_f64_t const *column_distances,
                                                                  nk_f64_t *rotation) {
    nk_shapes_gromov_serial_t search;
    nk_shapes_wasserstein_procrustes_f64_skylake_(views, rows, columns, cost_scale, rotation);
    nk_shapes_gromov_bind_f64_serial_(&search, views, rows, columns, row_distances, column_distances);
    nk_shapes_gromov_start_compensated_f64_serial_(&search);
    for (int request; (request = nk_shapes_gromov_next_compensated_f64_serial_(&search, views, rotation)) != 0;) {
        if (request == 1)
            nk_shapes_network_simplex_f64_skylake_(views, search.gradient, rows, columns, search.gradient_scale);
        else
            nk_shapes_distance_gradient_compensated_f64_skylake_(search.gradient_plan, search.plan_stride, search.arcs,
                                                                 search.mass, row_distances, rows, column_distances,
                                                                 columns, search.gradient_output);
    }
    return search.best;
}

NUMKONG_INLINE void nk_jacobi_eigenanalysis_f32_skylake_(nk_f32_t *matrix, nk_f32_t *quaternion) {
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

NUMKONG_INLINE void nk_jacobi_eigenanalysis_f64_skylake_(nk_f64_t *matrix, nk_f64_t *quaternion) {
    nk_jacobi_start_f64_serial_(quaternion);
    for (int sweep = 0; sweep != 16; ++sweep) {
        for (int axis = 0; axis != 3; ++axis) {
            nk_f64_t const norm = nk_jacobi_half_angle_f64_serial_(matrix);
            nk_jacobi_conjugate_f64_serial_(axis, nk_shapes_rsqrt_f64_skylake_(norm), matrix, quaternion);
        }
        if (nk_jacobi_converged_f64_serial_(matrix, 0x1p-53)) break;
    }
    nk_f64_t const length_squared = nk_jacobi_length_squared_f64_serial_(quaternion);
    nk_jacobi_rescale_f64_serial_(nk_shapes_rsqrt_f64_skylake_(length_squared), quaternion);
}

/** The next 16 @c f16 points of the @p left remaining, each axis in f32 lanes. Padded lanes read
 *  zero, and two masked loads hold the 48 words the axes permute out of. */
NUMKONG_INLINE void nk_shapes_load_points_f16_skylake_(nk_f16_t const *points, nk_size_t left, __m512 *x_f32x16,
                                                       __m512 *y_f32x16, __m512 *z_f32x16) {
    unsigned int const words = left < 16 ? (unsigned int)left * 3 : 48;
    __m512i const low_u16x32 = _mm512_maskz_loadu_epi16((__mmask32)_bzhi_u32(0xFFFFFFFFu, words), points);
    __m512i const high_u16x32 = _mm512_maskz_loadu_epi16((__mmask32)_bzhi_u32(0xFFFFu, words > 32 ? words - 32 : 0),
                                                         points + 32);
    __m512i const x_indices_u16x32 = _mm512_set_epi16(  //
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
        45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0);
    __m512i const y_indices_u16x32 = _mm512_add_epi16(x_indices_u16x32, _mm512_set1_epi16(1));
    __m512i const z_indices_u16x32 = _mm512_add_epi16(x_indices_u16x32, _mm512_set1_epi16(2));
    *x_f32x16 = _mm512_cvtph_ps(
        _mm512_castsi512_si256(_mm512_permutex2var_epi16(low_u16x32, x_indices_u16x32, high_u16x32)));
    *y_f32x16 = _mm512_cvtph_ps(
        _mm512_castsi512_si256(_mm512_permutex2var_epi16(low_u16x32, y_indices_u16x32, high_u16x32)));
    *z_f32x16 = _mm512_cvtph_ps(
        _mm512_castsi512_si256(_mm512_permutex2var_epi16(low_u16x32, z_indices_u16x32, high_u16x32)));
}

/** The moments of a cloud of @c f16 points as @ref nk_shapes_moments_f16_serial_ defines them, 16
 *  points per step. */
NUMKONG_INLINE nk_f32_t nk_shapes_moments_f16_skylake_(nk_f16_t const *points, nk_size_t count, nk_f32_t *centroid,
                                                       nk_f32_t *covariance) {
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    __m512 x_f32x16, y_f32x16, z_f32x16;
    nk_shapes_load_points_f16_skylake_(points, 1, &x_f32x16, &y_f32x16, &z_f32x16);
    __m512 const first_x_f32x16 = _mm512_broadcastss_ps(_mm512_castps512_ps128(x_f32x16));
    __m512 const first_y_f32x16 = _mm512_broadcastss_ps(_mm512_castps512_ps128(y_f32x16));
    __m512 const first_z_f32x16 = _mm512_broadcastss_ps(_mm512_castps512_ps128(z_f32x16));
    __m512 sum_x_f32x16 = _mm512_setzero_ps(), sum_y_f32x16 = sum_x_f32x16, sum_z_f32x16 = sum_x_f32x16;
    int coincident = 1;
    for (nk_size_t index = 0; index < count; index += 16) {
        nk_size_t const left = count - index;
        __mmask16 const valid_m16 = (__mmask16)_bzhi_u32(0xFFFF, left < 16 ? (unsigned int)left : 16);
        nk_shapes_load_points_f16_skylake_(points + index * 3, left, &x_f32x16, &y_f32x16, &z_f32x16);
        sum_x_f32x16 = _mm512_add_ps(sum_x_f32x16, x_f32x16), sum_y_f32x16 = _mm512_add_ps(sum_y_f32x16, y_f32x16);
        sum_z_f32x16 = _mm512_add_ps(sum_z_f32x16, z_f32x16);
        __mmask16 same_m16 = _mm512_mask_cmp_ps_mask(valid_m16, x_f32x16, first_x_f32x16, _CMP_EQ_OQ);
        same_m16 = _mm512_mask_cmp_ps_mask(same_m16, y_f32x16, first_y_f32x16, _CMP_EQ_OQ);
        same_m16 = _mm512_mask_cmp_ps_mask(same_m16, z_f32x16, first_z_f32x16, _CMP_EQ_OQ);
        coincident &= same_m16 == valid_m16;
    }
    nk_f32_t const points_count = (nk_f32_t)count;
    centroid[0] = nk_reduce_add_f32x16_skylake_(sum_x_f32x16) / points_count;
    centroid[1] = nk_reduce_add_f32x16_skylake_(sum_y_f32x16) / points_count;
    centroid[2] = nk_reduce_add_f32x16_skylake_(sum_z_f32x16) / points_count;
    __m512 const centroid_x_f32x16 = _mm512_set1_ps(centroid[0]), centroid_y_f32x16 = _mm512_set1_ps(centroid[1]);
    __m512 const centroid_z_f32x16 = _mm512_set1_ps(centroid[2]);
    __m512 moments_f32x16[6];
    for (int entry = 0; entry != 6; ++entry) moments_f32x16[entry] = _mm512_setzero_ps();
    for (nk_size_t index = 0; index < count; index += 16) {
        nk_size_t const left = count - index;
        __mmask16 const valid_m16 = (__mmask16)_bzhi_u32(0xFFFF, left < 16 ? (unsigned int)left : 16);
        nk_shapes_load_points_f16_skylake_(points + index * 3, left, &x_f32x16, &y_f32x16, &z_f32x16);
        x_f32x16 = _mm512_maskz_sub_ps(valid_m16, x_f32x16, centroid_x_f32x16);
        y_f32x16 = _mm512_maskz_sub_ps(valid_m16, y_f32x16, centroid_y_f32x16);
        z_f32x16 = _mm512_maskz_sub_ps(valid_m16, z_f32x16, centroid_z_f32x16);
        moments_f32x16[0] = _mm512_fmadd_ps(x_f32x16, x_f32x16, moments_f32x16[0]);
        moments_f32x16[1] = _mm512_fmadd_ps(y_f32x16, x_f32x16, moments_f32x16[1]);
        moments_f32x16[2] = _mm512_fmadd_ps(y_f32x16, y_f32x16, moments_f32x16[2]);
        moments_f32x16[3] = _mm512_fmadd_ps(z_f32x16, x_f32x16, moments_f32x16[3]);
        moments_f32x16[4] = _mm512_fmadd_ps(z_f32x16, y_f32x16, moments_f32x16[4]);
        moments_f32x16[5] = _mm512_fmadd_ps(z_f32x16, z_f32x16, moments_f32x16[5]);
    }
    for (int entry = 0; entry != 6; ++entry)
        covariance[entry] = nk_reduce_add_f32x16_skylake_(moments_f32x16[entry]) / points_count;
    return coincident ? 0 : nk_shapes_sqrt_f32_skylake_(covariance[0] + covariance[2] + covariance[5]);
}

/** The pack of one cloud from its moments, as @c nk_shapes_pack_f16_serial_ lays it out, storing
 *  each step's points and measuring what was stored in one sweep, with zeros in the padding. */
NUMKONG_INLINE void nk_shapes_pack_f16_skylake_(nk_f16_t const *points, nk_size_t count, nk_f32_t const *centroid,
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
    nk_jacobi_eigenanalysis_f32_skylake_(spreads, quaternion);
    nk_shapes_sorted_basis_f32_serial_(spreads, quaternion, 0x1p-7f, basis, &upper_tied, &lower_tied);
    if (upper_tied && lower_tied) nk_shapes_pick_axis_f16_serial_(points, count, centroid, basis);
    else if (upper_tied || lower_tied) nk_shapes_turn_plane_f16_serial_(points, count, centroid, basis);
    nk_shapes_frame_f32_serial_(basis, upper_tied, header->frame, axes);
    __m512 axes_f32x16[9];
    for (int cell = 0; cell != 9; ++cell) axes_f32x16[cell] = _mm512_set1_ps(axes[cell] / scale);
    __m512 const centroid_x_f32x16 = _mm512_set1_ps(centroid[0]), centroid_y_f32x16 = _mm512_set1_ps(centroid[1]);
    __m512 const centroid_z_f32x16 = _mm512_set1_ps(centroid[2]);
    __m512 moments_f32x16[6], largest_f32x16 = _mm512_setzero_ps();
    for (int entry = 0; entry != 6; ++entry) moments_f32x16[entry] = _mm512_setzero_ps();
    __m512 x_f32x16, y_f32x16, z_f32x16;
    for (nk_size_t index = 0; index < count; index += 16) {
        nk_size_t const left = count - index;
        __mmask16 const valid_m16 = (__mmask16)_bzhi_u32(0xFFFF, left < 16 ? (unsigned int)left : 16);
        nk_shapes_load_points_f16_skylake_(points + index * 3, left, &x_f32x16, &y_f32x16, &z_f32x16);
        x_f32x16 = _mm512_sub_ps(x_f32x16, centroid_x_f32x16), y_f32x16 = _mm512_sub_ps(y_f32x16, centroid_y_f32x16);
        z_f32x16 = _mm512_sub_ps(z_f32x16, centroid_z_f32x16);
        __m256i stored_f16x16[3];
        for (int axis = 0; axis != 3; ++axis) {
            __m512 const rotated_f32x16 = _mm512_fmadd_ps(
                z_f32x16, axes_f32x16[axis * 3 + 2],
                _mm512_fmadd_ps(y_f32x16, axes_f32x16[axis * 3 + 1], _mm512_mul_ps(x_f32x16, axes_f32x16[axis * 3])));
            stored_f16x16[axis] = _mm512_cvtps_ph(_mm512_maskz_mov_ps(valid_m16, rotated_f32x16),
                                                  _MM_FROUND_TO_NEAREST_INT);
        }
        _mm256_storeu_si256((__m256i *)(packed_x + index), stored_f16x16[0]);
        _mm256_storeu_si256((__m256i *)(packed_y + index), stored_f16x16[1]);
        _mm256_storeu_si256((__m256i *)(packed_z + index), stored_f16x16[2]);
        // The spreads describe the stored points, so the sweep measures them after rounding
        x_f32x16 = _mm512_cvtph_ps(stored_f16x16[0]), y_f32x16 = _mm512_cvtph_ps(stored_f16x16[1]);
        z_f32x16 = _mm512_cvtph_ps(stored_f16x16[2]);
        __m512 const xx_f32x16 = _mm512_mul_ps(x_f32x16, x_f32x16), yy_f32x16 = _mm512_mul_ps(y_f32x16, y_f32x16);
        __m512 const zz_f32x16 = _mm512_mul_ps(z_f32x16, z_f32x16);
        largest_f32x16 = _mm512_max_ps(largest_f32x16, _mm512_add_ps(_mm512_add_ps(xx_f32x16, yy_f32x16), zz_f32x16));
        moments_f32x16[0] = _mm512_add_ps(moments_f32x16[0], xx_f32x16);
        moments_f32x16[1] = _mm512_fmadd_ps(y_f32x16, x_f32x16, moments_f32x16[1]);
        moments_f32x16[2] = _mm512_add_ps(moments_f32x16[2], yy_f32x16);
        moments_f32x16[3] = _mm512_fmadd_ps(z_f32x16, x_f32x16, moments_f32x16[3]);
        moments_f32x16[4] = _mm512_fmadd_ps(z_f32x16, y_f32x16, moments_f32x16[4]);
        moments_f32x16[5] = _mm512_add_ps(moments_f32x16[5], zz_f32x16);
    }
    for (nk_size_t index = (count + 15) & ~(nk_size_t)15; index < axis_stride; index += 16)
        for (int axis = 0; axis != 3; ++axis)
            _mm256_storeu_si256((__m256i *)(packed_x + axis * axis_stride + index), _mm256_setzero_si256());
    nk_f32_t moments[6];
    for (int entry = 0; entry != 6; ++entry)
        moments[entry] = nk_reduce_add_f32x16_skylake_(moments_f32x16[entry]) / (nk_f32_t)count;
    nk_jacobi_eigenanalysis_f32_skylake_(moments, quaternion);
    nk_shapes_close_pack_f32_serial_(header, count, centroid, moments,
                                     nk_shapes_sqrt_f32_skylake_(nk_reduce_max_f32x16_skylake_(largest_f32x16)));
}

/** The next 8 @c f32 points of the @p left remaining, each axis in f64 lanes. Padded lanes read
 *  zero, and two masked loads hold the 24 values the axes permute out of. */
NUMKONG_INLINE void nk_shapes_load_points_f32_skylake_(nk_f32_t const *points, nk_size_t left, __m512d *x_f64x8,
                                                       __m512d *y_f64x8, __m512d *z_f64x8) {
    unsigned int const words = left < 8 ? (unsigned int)left * 3 : 24;
    __m512 const low_f32x16 = _mm512_maskz_loadu_ps((__mmask16)_bzhi_u32(0xFFFF, words), points);
    __m512 const high_f32x16 = _mm512_maskz_loadu_ps((__mmask16)_bzhi_u32(0xFF, words > 16 ? words - 16 : 0),
                                                     points + 16);
    __m512i const x_indices_i32x16 = _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 0, 0, 0, 0, 0, 0, 0, 0);
    __m512i const y_indices_i32x16 = _mm512_add_epi32(x_indices_i32x16, _mm512_set1_epi32(1));
    __m512i const z_indices_i32x16 = _mm512_add_epi32(x_indices_i32x16, _mm512_set1_epi32(2));
    *x_f64x8 = _mm512_cvtps_pd(
        _mm512_castps512_ps256(_mm512_permutex2var_ps(low_f32x16, x_indices_i32x16, high_f32x16)));
    *y_f64x8 = _mm512_cvtps_pd(
        _mm512_castps512_ps256(_mm512_permutex2var_ps(low_f32x16, y_indices_i32x16, high_f32x16)));
    *z_f64x8 = _mm512_cvtps_pd(
        _mm512_castps512_ps256(_mm512_permutex2var_ps(low_f32x16, z_indices_i32x16, high_f32x16)));
}

/** The moments of a cloud of @c f32 points as @ref nk_shapes_moments_f32_serial_ defines them, 8
 *  points per step. */
NUMKONG_INLINE nk_f64_t nk_shapes_moments_f32_skylake_(nk_f32_t const *points, nk_size_t count, nk_f64_t *centroid,
                                                       nk_f64_t *covariance) {
    centroid[0] = 0, centroid[1] = 0, centroid[2] = 0;
    for (int entry = 0; entry != 6; ++entry) covariance[entry] = 0;
    if (!count) return 0;
    __m512d x_f64x8, y_f64x8, z_f64x8;
    __m512d const first_x_f64x8 = _mm512_set1_pd(points[0]), first_y_f64x8 = _mm512_set1_pd(points[1]);
    __m512d const first_z_f64x8 = _mm512_set1_pd(points[2]);
    __m512d sum_x_f64x8 = _mm512_setzero_pd(), sum_y_f64x8 = sum_x_f64x8, sum_z_f64x8 = sum_x_f64x8;
    int coincident = 1;
    for (nk_size_t index = 0; index < count; index += 8) {
        nk_size_t const left = count - index;
        __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, left < 8 ? (unsigned int)left : 8);
        nk_shapes_load_points_f32_skylake_(points + index * 3, left, &x_f64x8, &y_f64x8, &z_f64x8);
        sum_x_f64x8 = _mm512_add_pd(sum_x_f64x8, x_f64x8), sum_y_f64x8 = _mm512_add_pd(sum_y_f64x8, y_f64x8);
        sum_z_f64x8 = _mm512_add_pd(sum_z_f64x8, z_f64x8);
        __mmask8 same_m8 = _mm512_mask_cmp_pd_mask(valid_m8, x_f64x8, first_x_f64x8, _CMP_EQ_OQ);
        same_m8 = _mm512_mask_cmp_pd_mask(same_m8, y_f64x8, first_y_f64x8, _CMP_EQ_OQ);
        same_m8 = _mm512_mask_cmp_pd_mask(same_m8, z_f64x8, first_z_f64x8, _CMP_EQ_OQ);
        coincident &= same_m8 == valid_m8;
    }
    nk_f64_t const points_count = (nk_f64_t)count;
    centroid[0] = nk_reduce_add_f64x8_skylake_(sum_x_f64x8) / points_count;
    centroid[1] = nk_reduce_add_f64x8_skylake_(sum_y_f64x8) / points_count;
    centroid[2] = nk_reduce_add_f64x8_skylake_(sum_z_f64x8) / points_count;
    __m512d const centroid_x_f64x8 = _mm512_set1_pd(centroid[0]), centroid_y_f64x8 = _mm512_set1_pd(centroid[1]);
    __m512d const centroid_z_f64x8 = _mm512_set1_pd(centroid[2]);
    __m512d moments_f64x8[6];
    for (int entry = 0; entry != 6; ++entry) moments_f64x8[entry] = _mm512_setzero_pd();
    for (nk_size_t index = 0; index < count; index += 8) {
        nk_size_t const left = count - index;
        __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, left < 8 ? (unsigned int)left : 8);
        nk_shapes_load_points_f32_skylake_(points + index * 3, left, &x_f64x8, &y_f64x8, &z_f64x8);
        x_f64x8 = _mm512_maskz_sub_pd(valid_m8, x_f64x8, centroid_x_f64x8);
        y_f64x8 = _mm512_maskz_sub_pd(valid_m8, y_f64x8, centroid_y_f64x8);
        z_f64x8 = _mm512_maskz_sub_pd(valid_m8, z_f64x8, centroid_z_f64x8);
        moments_f64x8[0] = _mm512_fmadd_pd(x_f64x8, x_f64x8, moments_f64x8[0]);
        moments_f64x8[1] = _mm512_fmadd_pd(y_f64x8, x_f64x8, moments_f64x8[1]);
        moments_f64x8[2] = _mm512_fmadd_pd(y_f64x8, y_f64x8, moments_f64x8[2]);
        moments_f64x8[3] = _mm512_fmadd_pd(z_f64x8, x_f64x8, moments_f64x8[3]);
        moments_f64x8[4] = _mm512_fmadd_pd(z_f64x8, y_f64x8, moments_f64x8[4]);
        moments_f64x8[5] = _mm512_fmadd_pd(z_f64x8, z_f64x8, moments_f64x8[5]);
    }
    for (int entry = 0; entry != 6; ++entry)
        covariance[entry] = nk_reduce_add_f64x8_skylake_(moments_f64x8[entry]) / points_count;
    return coincident ? 0 : nk_shapes_sqrt_f64_skylake_(covariance[0] + covariance[2] + covariance[5]);
}

/** @copydoc nk_shapes_pack_f16_skylake_ */
NUMKONG_INLINE void nk_shapes_pack_f32_skylake_(nk_f32_t const *points, nk_size_t count, nk_f64_t const *centroid,
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
    nk_jacobi_eigenanalysis_f64_skylake_(spreads, quaternion);
    nk_shapes_sorted_basis_f64_serial_(spreads, quaternion, 0x1p-16, basis, &upper_tied, &lower_tied);
    if (upper_tied && lower_tied) nk_shapes_pick_axis_f32_serial_(points, count, centroid, basis);
    else if (upper_tied || lower_tied) nk_shapes_turn_plane_f32_serial_(points, count, centroid, basis);
    nk_shapes_frame_f64_serial_(basis, upper_tied, header->frame, axes);
    __m512d axes_f64x8[9];
    for (int cell = 0; cell != 9; ++cell) axes_f64x8[cell] = _mm512_set1_pd(axes[cell] / scale);
    __m512d const centroid_x_f64x8 = _mm512_set1_pd(centroid[0]), centroid_y_f64x8 = _mm512_set1_pd(centroid[1]);
    __m512d const centroid_z_f64x8 = _mm512_set1_pd(centroid[2]);
    __m512d moments_f64x8[6], largest_f64x8 = _mm512_setzero_pd();
    for (int entry = 0; entry != 6; ++entry) moments_f64x8[entry] = _mm512_setzero_pd();
    __m512d x_f64x8, y_f64x8, z_f64x8;
    for (nk_size_t index = 0; index < count; index += 8) {
        nk_size_t const left = count - index;
        __mmask8 const valid_m8 = (__mmask8)_bzhi_u32(0xFF, left < 8 ? (unsigned int)left : 8);
        nk_shapes_load_points_f32_skylake_(points + index * 3, left, &x_f64x8, &y_f64x8, &z_f64x8);
        x_f64x8 = _mm512_sub_pd(x_f64x8, centroid_x_f64x8), y_f64x8 = _mm512_sub_pd(y_f64x8, centroid_y_f64x8);
        z_f64x8 = _mm512_sub_pd(z_f64x8, centroid_z_f64x8);
        __m256 stored_f32x8[3];
        for (int axis = 0; axis != 3; ++axis) {
            __m512d const rotated_f64x8 = _mm512_fmadd_pd(
                z_f64x8, axes_f64x8[axis * 3 + 2],
                _mm512_fmadd_pd(y_f64x8, axes_f64x8[axis * 3 + 1], _mm512_mul_pd(x_f64x8, axes_f64x8[axis * 3])));
            stored_f32x8[axis] = _mm512_cvtpd_ps(_mm512_maskz_mov_pd(valid_m8, rotated_f64x8));
        }
        _mm256_storeu_ps(packed_x + index, stored_f32x8[0]), _mm256_storeu_ps(packed_y + index, stored_f32x8[1]);
        _mm256_storeu_ps(packed_z + index, stored_f32x8[2]);
        // The spreads describe the stored points, so the sweep measures them after rounding
        x_f64x8 = _mm512_cvtps_pd(stored_f32x8[0]), y_f64x8 = _mm512_cvtps_pd(stored_f32x8[1]);
        z_f64x8 = _mm512_cvtps_pd(stored_f32x8[2]);
        __m512d const xx_f64x8 = _mm512_mul_pd(x_f64x8, x_f64x8), yy_f64x8 = _mm512_mul_pd(y_f64x8, y_f64x8);
        __m512d const zz_f64x8 = _mm512_mul_pd(z_f64x8, z_f64x8);
        largest_f64x8 = _mm512_max_pd(largest_f64x8, _mm512_add_pd(_mm512_add_pd(xx_f64x8, yy_f64x8), zz_f64x8));
        moments_f64x8[0] = _mm512_add_pd(moments_f64x8[0], xx_f64x8);
        moments_f64x8[1] = _mm512_fmadd_pd(y_f64x8, x_f64x8, moments_f64x8[1]);
        moments_f64x8[2] = _mm512_add_pd(moments_f64x8[2], yy_f64x8);
        moments_f64x8[3] = _mm512_fmadd_pd(z_f64x8, x_f64x8, moments_f64x8[3]);
        moments_f64x8[4] = _mm512_fmadd_pd(z_f64x8, y_f64x8, moments_f64x8[4]);
        moments_f64x8[5] = _mm512_add_pd(moments_f64x8[5], zz_f64x8);
    }
    for (nk_size_t index = (count + 7) & ~(nk_size_t)7; index < axis_stride; index += 8)
        for (int axis = 0; axis != 3; ++axis)
            _mm256_storeu_ps(packed_x + axis * axis_stride + index, _mm256_setzero_ps());
    nk_f64_t moments[6];
    for (int entry = 0; entry != 6; ++entry)
        moments[entry] = nk_reduce_add_f64x8_skylake_(moments_f64x8[entry]) / (nk_f64_t)count;
    nk_jacobi_eigenanalysis_f64_skylake_(moments, quaternion);
    nk_shapes_close_pack_f64_serial_(header, count, centroid, moments,
                                     nk_shapes_sqrt_f64_skylake_(nk_reduce_max_f64x8_skylake_(largest_f64x8)));
}

NUMKONG_API nk_status_t nk_shapes_pack_f16_skylake(nk_f16_t const *points, nk_size_t count, void *packed,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t centroid[3], covariance[6];
    nk_shapes_moments_f16_skylake_(points, count, centroid, covariance);
    nk_shapes_pack_f16_skylake_(points, count, centroid, covariance, 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f16_skylake(nk_f16_t const *points, nk_size_t count, void *packed,
                                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f32_t centroid[3], covariance[6];
    nk_f32_t const radius = nk_shapes_moments_f16_skylake_(points, count, centroid, covariance);
    nk_shapes_pack_f16_skylake_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_f32_skylake(nk_f32_t const *points, nk_size_t count, void *packed,
                                                   nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t centroid[3], covariance[6];
    nk_shapes_moments_f32_skylake_(points, count, centroid, covariance);
    nk_shapes_pack_f32_skylake_(points, count, centroid, covariance, 1, packed);
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_shapes_pack_normalized_f32_skylake(nk_f32_t const *points, nk_size_t count, void *packed,
                                                              nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_f64_t centroid[3], covariance[6];
    nk_f64_t const radius = nk_shapes_moments_f32_skylake_(points, count, centroid, covariance);
    nk_shapes_pack_f32_skylake_(points, count, centroid, covariance, radius > 0 ? radius : 1, packed);
    return nk_success_k;
}

/** Chamfer flips of @c f64 packs through a compensated nearest pass each way; past @p limit after
 *  the first direction it skips the second, leaving per-flip lower bounds. */
NUMKONG_INLINE void nk_shapes_chamfer_directed_f64_skylake_(void const *row_packed, void const *column_packed,
                                                            nk_f64_t limit, nk_f64_t *chamfers) {
    nk_size_t const rows = (nk_size_t)((nk_shapes_packed_header_f64_t const *)row_packed)->points;
    nk_size_t const columns = (nk_size_t)((nk_shapes_packed_header_f64_t const *)column_packed)->points;
    if (!rows || !columns) {
        for (int flip = 0; flip != 8; ++flip) chamfers[flip] = rows == columns ? 0 : NUMKONG_F64_INF;
        return;
    }
    nk_size_t row_stride, column_stride;
    nk_f64_t const *row_axes = nk_shapes_packed_axes_f64_serial_(row_packed, &row_stride);
    nk_f64_t const *column_axes = nk_shapes_packed_axes_f64_serial_(column_packed, &column_stride);
    nk_f64_t forward_sums[8] = {0, 0, 0, 0, 0, 0, 0, 0}, backward_sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    nk_f64_t forward_maxima[8] = {0, 0, 0, 0, 0, 0, 0, 0}, backward_maxima[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    nk_shapes_nearest_f64_skylake_(row_axes, row_stride, rows, column_axes, column_stride, columns, forward_sums,
                                   forward_maxima);
    nk_f64_t const row_count = (nk_f64_t)rows, column_count = (nk_f64_t)columns;
    nk_f64_t least = NUMKONG_F64_INF;
    for (int flip = 0; flip != 8; ++flip) {
        chamfers[flip] = forward_sums[flip] / row_count / 2;
        least = chamfers[flip] < least ? chamfers[flip] : least;
    }
    if (least > limit) return;
    nk_shapes_nearest_f64_skylake_(column_axes, column_stride, columns, row_axes, row_stride, rows, backward_sums,
                                   backward_maxima);
    for (int flip = 0; flip != 8; ++flip)
        chamfers[flip] = (forward_sums[flip] / row_count + backward_sums[flip] / column_count) / 2;
}

/** Hausdorff flips of @c f64 packs through a nearest pass each way, stopping as the Chamfer one
 *  does, and before reading a point when the packs' largest norms alone settle it. */
NUMKONG_INLINE void nk_shapes_hausdorff_directed_f64_skylake_(void const *row_packed, void const *column_packed,
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
    nk_f64_t const *row_axes = nk_shapes_packed_axes_f64_serial_(row_packed, &row_stride);
    nk_f64_t const *column_axes = nk_shapes_packed_axes_f64_serial_(column_packed, &column_stride);
    nk_f64_t forward_sums[8] = {0, 0, 0, 0, 0, 0, 0, 0}, backward_sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    nk_f64_t backward_maxima[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int flip = 0; flip != 8; ++flip) hausdorffs[flip] = 0;
    nk_shapes_nearest_f64_skylake_(row_axes, row_stride, rows, column_axes, column_stride, columns, forward_sums,
                                   hausdorffs);
    nk_f64_t least = NUMKONG_F64_INF;
    for (int flip = 0; flip != 8; ++flip) least = hausdorffs[flip] < least ? hausdorffs[flip] : least;
    if (least > limit) return;
    nk_shapes_nearest_f64_skylake_(column_axes, column_stride, columns, row_axes, row_stride, rows, backward_sums,
                                   backward_maxima);
    for (int flip = 0; flip != 8; ++flip)
        hausdorffs[flip] = backward_maxima[flip] > hausdorffs[flip] ? backward_maxima[flip] : hausdorffs[flip];
}

NUMKONG_API nk_status_t nk_chamfers_packed_f64_skylake(void const *const *a_packed, nk_size_t rows,
                                                       void const *const *b_packed, nk_size_t columns,
                                                       nk_f64_t *results, nk_size_t results_stride, nk_f32_t *rotations,
                                                       nk_f64_t const *limit, void *workspace,
                                                       nk_size_t workspace_bytes, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_unused_(workspace), nk_unused_(workspace_bytes);
    nk_f64_t const pair_limit = *limit;
    nk_f64_t chamfers[8];
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *results_row = (nk_f64_t *)((char *)results + row * results_stride);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_chamfer_directed_f64_skylake_(a_packed[row], b_packed[column], pair_limit, chamfers);
            nk_shapes_best_flip_f64_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f64_skylake(void const *const *a_packed, nk_size_t rows,
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
            nk_shapes_hausdorff_directed_f64_skylake_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f64_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f32_skylake(void const *const *a_packed, nk_size_t rows,
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
            nk_status_t const status = nk_shapes_chamfer_flips_f32_skylake_(a_packed[row], b_packed[column], pair_limit,
                                                                            workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f64_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f32_skylake(void const *const *a_packed, nk_size_t rows,
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
            nk_shapes_hausdorff_flips_f32_skylake_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f64_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_chamfers_packed_f16_skylake(void const *const *a_packed, nk_size_t rows,
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
            nk_status_t const status = nk_shapes_chamfer_flips_f16_skylake_(a_packed[row], b_packed[column], pair_limit,
                                                                            workspace, workspace_bytes, chamfers);
            if (status != nk_success_k) return status;
            nk_shapes_best_flip_f32_serial_(chamfers, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_hausdorffs_packed_f16_skylake(void const *const *a_packed, nk_size_t rows,
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
            nk_shapes_hausdorff_flips_f16_skylake_(a_packed[row], b_packed[column], pair_limit, hausdorffs);
            nk_shapes_best_flip_f32_serial_(hausdorffs, a_packed[row], b_packed[column], results_row + column,
                                            rotations ? rotations + (row * columns + column) * 9 : NUMKONG_NULL);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f64_skylake(void const *const *a_packed, nk_size_t rows,
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
            if (nk_shapes_pair_open_f64_serial_(&pair, a_packed[row], b_packed[column], &views))
                results_row[column] = nk_sqrt_f64_serial_(nk_shapes_wasserstein_procrustes_f64_skylake_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F64_INF;
            if (rotations) nk_shapes_pair_rotation_f64_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f64_skylake(void const *const *a_packed, nk_size_t rows,
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
        nk_shapes_widen_f64_serial_(a_packed[row], views.row_points, views.points_stride);
        nk_shapes_distances_f64_serial_(views.row_points, cached_points, views.points_stride, cached_distances);
        for (nk_size_t column = 0; column != columns; ++column) {
            nk_shapes_pair_f64_serial_t pair;
            if (nk_shapes_pair_open_f64_serial_(&pair, a_packed[row], b_packed[column], &views)) {
                // The cached matrix is this row's cloud, on whichever side it is solved
                nk_shapes_distances_f64_serial_(pair.swapped ? views.row_points : views.column_points,
                                                pair.swapped ? pair.rows : pair.columns, views.points_stride,
                                                pair_distances);
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f64_skylake_(
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

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f32_skylake(void const *const *a_packed, nk_size_t rows,
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
                results_row[column] = nk_sqrt_f64_serial_(nk_shapes_wasserstein_procrustes_f32_skylake_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F64_INF;
            if (rotations) nk_shapes_pair_rotation_f64_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f32_skylake(void const *const *a_packed, nk_size_t rows,
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
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f32_skylake_(
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

NUMKONG_API nk_status_t nk_wassersteinprocrustes_packed_f16_skylake(void const *const *a_packed, nk_size_t rows,
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
                results_row[column] = nk_sqrt_f32_serial_(nk_shapes_wasserstein_procrustes_f16_skylake_(
                    &views, pair.rows, pair.columns, pair.cost_scale, pair.inner));
            else results_row[column] = pair.rows == pair.columns ? 0 : NUMKONG_F32_INF;
            if (rotations) nk_shapes_pair_rotation_f32_serial_(&pair, rotations + (row * columns + column) * 9);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_gromovwassersteins_packed_f16_skylake(void const *const *a_packed, nk_size_t rows,
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
                nk_f64_t const loss = nk_shapes_gromov_wasserstein_f16_skylake_(
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

/** Superposed RMSD of one pair of @c f32 packs: H in @c f64, then the residuals in plain @c f64, 8
 *  points per step through the zero padding. */
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
    __m512d sum_f64x8 = _mm512_setzero_pd();
    for (nk_size_t index = 0; index < count; index += 8) {
        __m512d b_f64x8[3];
        for (int axis = 0; axis != 3; ++axis)
            b_f64x8[axis] = _mm512_cvtps_pd(_mm256_loadu_ps(b_axes + axis * axis_stride + index));
        for (int axis = 0; axis != 3; ++axis) {
            __m512d residual_f64x8 = _mm512_cvtps_pd(_mm256_loadu_ps(a_axes + axis * axis_stride + index));
            residual_f64x8 = _mm512_fnmadd_pd(rotation_f64x8[axis * 3 + 0], b_f64x8[0], residual_f64x8);
            residual_f64x8 = _mm512_fnmadd_pd(rotation_f64x8[axis * 3 + 1], b_f64x8[1], residual_f64x8);
            residual_f64x8 = _mm512_fnmadd_pd(rotation_f64x8[axis * 3 + 2], b_f64x8[2], residual_f64x8);
            sum_f64x8 = _mm512_fmadd_pd(residual_f64x8, residual_f64x8, sum_f64x8);
        }
    }
    return nk_sqrt_f64_serial_(nk_reduce_add_f64x8_skylake_(sum_f64x8) / (nk_f64_t)count);
}

/** Superposed RMSD of one pair of @c f64 packs, like the @c f32 pair with H and the residuals
 *  compensated. */
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
    nk_f32_t row_norms[nk_rmsds_tile_frames_k], column_norms[nk_rmsds_tile_frames_k];
    for (nk_size_t row_begin = 0; row_begin < rows; row_begin += nk_rmsds_tile_frames_k) {
        nk_size_t const row_end = row_begin + nk_rmsds_tile_frames_k < rows ? row_begin + nk_rmsds_tile_frames_k : rows;
        for (nk_size_t row = row_begin; row != row_end; ++row)
            row_norms[row - row_begin] = nk_rmsds_norm_f16_skylake_(a_packed[row]);
        for (nk_size_t column_begin = 0; column_begin < columns; column_begin += nk_rmsds_tile_frames_k) {
            nk_size_t const column_end = column_begin + nk_rmsds_tile_frames_k < columns
                                             ? column_begin + nk_rmsds_tile_frames_k
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
