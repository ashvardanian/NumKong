/**
 *  @file include/numkong/geospatial/haswell.h
 *  @author Ash Vardanian
 *  @date December 21, 2025
 *  @brief SIMD-accelerated geospatial distances for Haswell.
 *
 *  @sa include/numkong/geospatial.h
 *
 *  Every kernel broadcasts one row and sweeps the columns four or eight at a time. The 3D squares
 *  load three registers of @b [x,y,z] rows and de-interleave them in place, the @c f64 ones with
 *  two blends, one @c VPERM2F128 and three in-lane @c VSHUFPD, and the @c i32 ones with two blends
 *  and one @c VPERMD per axis. That @c i32 order puts point k in the even half of 64-bit lane k and
 *  point k + 4 in its odd half, so @c VPMULDQ squares the first four points directly and the last
 *  four after a 32-bit shift, and both halves store contiguously. Haversine and Vincenty read the
 *  planar packs with plain loads.
 *
 *  @section geospatial_haswell_instructions Key AVX2 Geospatial Instructions
 *
 *  @verbatim
 *  Intrinsic         Instruction                  Icelake    Genoa
 *  _mm256_mul_epi32  VPMULDQ (YMM, YMM, YMM)      5cy @ p01  3cy @ p01
 *  _mm256_sqrt_pd    VSQRTPD (YMM, YMM)           13cy @ p0  21cy @ p01
 *  _mm256_div_pd     VDIVPD (YMM, YMM, YMM)       13cy @ p0  13cy @ p01
 *  _mm256_fmadd_pd   VFMADD231PD (YMM, YMM, YMM)  4cy @ p01  4cy @ p01
 *  @endverbatim
 */
#ifndef NUMKONG_GEOSPATIAL_HASWELL_H
#define NUMKONG_GEOSPATIAL_HASWELL_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_HASWELL_

#include "numkong/types.h"
#include "numkong/geospatial/serial.h"    // `nk_geospatial_packed_axes_f64_serial_`
#include "numkong/cast/haswell.h"         // `nk_partial_load_b64x4_haswell_`, `nk_partial_load_b32x8_haswell_`
#include "numkong/dot/haswell.h"          // `nk_dot2_f64x4_haswell_`
#include "numkong/trigonometry/haswell.h" // `nk_sin_f64x4_haswell_`, `nk_cos_f64x4_haswell_`, `nk_atan2_f64x4_haswell_`

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

/** Exact squared distances from a broadcast 3D point to 8 interleaved @b [x,y,z] points in three
 *  registers, the first four sums in @p sums_low_u64x4 and the last four in @p sums_high_u64x4. */
NUMKONG_INLINE void nk_sqeuclidean3d_i32x8_haswell_(                      //
    __m256i a_x_i32x8, __m256i a_y_i32x8, __m256i a_z_i32x8,              //
    __m256i b_first_i32x8, __m256i b_second_i32x8, __m256i b_third_i32x8, //
    __m256i *sums_low_u64x4, __m256i *sums_high_u64x4) {
    // Each lane finds an axis in one register: two blends gather it and a permute orders it
    __m256i const b_x_i32x8 = _mm256_permutevar8x32_epi32(
        _mm256_blend_epi32(_mm256_blend_epi32(b_first_i32x8, b_second_i32x8, 0x92), b_third_i32x8, 0x24),
        _mm256_setr_epi32(0, 4, 3, 7, 6, 2, 1, 5));
    __m256i const b_y_i32x8 = _mm256_permutevar8x32_epi32(
        _mm256_blend_epi32(_mm256_blend_epi32(b_first_i32x8, b_second_i32x8, 0x24), b_third_i32x8, 0x49),
        _mm256_setr_epi32(1, 5, 4, 0, 7, 3, 2, 6));
    __m256i const b_z_i32x8 = _mm256_permutevar8x32_epi32(
        _mm256_blend_epi32(_mm256_blend_epi32(b_first_i32x8, b_second_i32x8, 0x49), b_third_i32x8, 0x92),
        _mm256_setr_epi32(2, 6, 5, 1, 0, 4, 3, 7));
    __m256i const difference_x_i32x8 = _mm256_sub_epi32(a_x_i32x8, b_x_i32x8);
    __m256i const difference_y_i32x8 = _mm256_sub_epi32(a_y_i32x8, b_y_i32x8);
    __m256i const difference_z_i32x8 = _mm256_sub_epi32(a_z_i32x8, b_z_i32x8);
    __m256i const difference_x_odd_i32x8 = _mm256_srli_epi64(difference_x_i32x8, 32);
    __m256i const difference_y_odd_i32x8 = _mm256_srli_epi64(difference_y_i32x8, 32);
    __m256i const difference_z_odd_i32x8 = _mm256_srli_epi64(difference_z_i32x8, 32);
    *sums_low_u64x4 = _mm256_add_epi64(_mm256_add_epi64(_mm256_mul_epi32(difference_x_i32x8, difference_x_i32x8),
                                                        _mm256_mul_epi32(difference_y_i32x8, difference_y_i32x8)),
                                       _mm256_mul_epi32(difference_z_i32x8, difference_z_i32x8));
    *sums_high_u64x4 = _mm256_add_epi64(
        _mm256_add_epi64(_mm256_mul_epi32(difference_x_odd_i32x8, difference_x_odd_i32x8),
                         _mm256_mul_epi32(difference_y_odd_i32x8, difference_y_odd_i32x8)),
        _mm256_mul_epi32(difference_z_odd_i32x8, difference_z_odd_i32x8));
}

/** Squared distances from a broadcast 3D point to 4 interleaved @b [x,y,z] points in three
 *  registers, the three squares summed in Dot2. */
NUMKONG_INLINE __m256d nk_sqeuclidean3d_f64x4_haswell_(      //
    __m256d a_x_f64x4, __m256d a_y_f64x4, __m256d a_z_f64x4, //
    __m256d b_first_f64x4, __m256d b_second_f64x4, __m256d b_third_f64x4) {
    // [x0 y0 x2 y2], [z0 x1 z2 x3] and [y1 z1 y3 z3] pair every axis up within the 128-bit halves
    __m256d const xy_pairs_f64x4 = _mm256_blend_pd(b_first_f64x4, b_second_f64x4, 0xC);
    __m256d const zx_pairs_f64x4 = _mm256_permute2f128_pd(b_first_f64x4, b_third_f64x4, 0x21);
    __m256d const yz_pairs_f64x4 = _mm256_blend_pd(b_second_f64x4, b_third_f64x4, 0xC);
    __m256d const difference_x_f64x4 = _mm256_sub_pd(a_x_f64x4, _mm256_shuffle_pd(xy_pairs_f64x4, zx_pairs_f64x4, 0xA));
    __m256d const difference_y_f64x4 = _mm256_sub_pd(a_y_f64x4, _mm256_shuffle_pd(xy_pairs_f64x4, yz_pairs_f64x4, 0x5));
    __m256d const difference_z_f64x4 = _mm256_sub_pd(a_z_f64x4, _mm256_shuffle_pd(zx_pairs_f64x4, yz_pairs_f64x4, 0xA));
    __m256d sum_f64x4 = _mm256_setzero_pd(), compensation_f64x4 = _mm256_setzero_pd();
    nk_dot2_f64x4_haswell_(&sum_f64x4, &compensation_f64x4, difference_x_f64x4, difference_x_f64x4);
    nk_dot2_f64x4_haswell_(&sum_f64x4, &compensation_f64x4, difference_y_f64x4, difference_y_f64x4);
    nk_dot2_f64x4_haswell_(&sum_f64x4, &compensation_f64x4, difference_z_f64x4, difference_z_f64x4);
    return _mm256_add_pd(sum_f64x4, compensation_f64x4);
}

/** Great-circle metres between lanes of unit vectors, 2R · atan2(‖a − b‖, ‖a + b‖),
 *  with both norms summed in Dot2. */
NUMKONG_INLINE __m256d nk_haversine_f64x4_haswell_(                      //
    __m256d first_x_f64x4, __m256d first_y_f64x4, __m256d first_z_f64x4, //
    __m256d second_x_f64x4, __m256d second_y_f64x4, __m256d second_z_f64x4) {
    __m256d const difference_x_f64x4 = _mm256_sub_pd(first_x_f64x4, second_x_f64x4);
    __m256d const difference_y_f64x4 = _mm256_sub_pd(first_y_f64x4, second_y_f64x4);
    __m256d const difference_z_f64x4 = _mm256_sub_pd(first_z_f64x4, second_z_f64x4);
    __m256d const sum_x_f64x4 = _mm256_add_pd(first_x_f64x4, second_x_f64x4);
    __m256d const sum_y_f64x4 = _mm256_add_pd(first_y_f64x4, second_y_f64x4);
    __m256d const sum_z_f64x4 = _mm256_add_pd(first_z_f64x4, second_z_f64x4);
    __m256d chord_f64x4 = _mm256_setzero_pd(), chord_compensation_f64x4 = _mm256_setzero_pd();
    __m256d span_f64x4 = _mm256_setzero_pd(), span_compensation_f64x4 = _mm256_setzero_pd();
    nk_dot2_f64x4_haswell_(&chord_f64x4, &chord_compensation_f64x4, difference_x_f64x4, difference_x_f64x4);
    nk_dot2_f64x4_haswell_(&chord_f64x4, &chord_compensation_f64x4, difference_y_f64x4, difference_y_f64x4);
    nk_dot2_f64x4_haswell_(&chord_f64x4, &chord_compensation_f64x4, difference_z_f64x4, difference_z_f64x4);
    nk_dot2_f64x4_haswell_(&span_f64x4, &span_compensation_f64x4, sum_x_f64x4, sum_x_f64x4);
    nk_dot2_f64x4_haswell_(&span_f64x4, &span_compensation_f64x4, sum_y_f64x4, sum_y_f64x4);
    nk_dot2_f64x4_haswell_(&span_f64x4, &span_compensation_f64x4, sum_z_f64x4, sum_z_f64x4);
    __m256d const angle_f64x4 = nk_atan2_f64x4_haswell_(
        _mm256_sqrt_pd(_mm256_add_pd(chord_f64x4, chord_compensation_f64x4)),
        _mm256_sqrt_pd(_mm256_add_pd(span_f64x4, span_compensation_f64x4)));
    return _mm256_mul_pd(_mm256_set1_pd(2.0 * NUMKONG_EARTH_MEAN_RADIUS), angle_f64x4);
}

/**
 *  @brief AVX2 helper for Vincenty's geodesic distance on 4 f64 point pairs, from their longitudes
 *      and the sines and cosines of their reduced latitudes.
 *  @note This is a true SIMD implementation using masked convergence tracking via blending.
 */
NUMKONG_INLINE __m256d nk_vincenty_f64x4_haswell_(                                                    //
    __m256d first_longitudes_f64x4, __m256d sin_reduced_first_f64x4, __m256d cos_reduced_first_f64x4, //
    __m256d second_longitudes_f64x4, __m256d sin_reduced_second_f64x4, __m256d cos_reduced_second_f64x4) {

    nk_f64_t const equatorial_radius = NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    nk_f64_t const flattening = 1.0 / NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    nk_f64_t const polar_radius = equatorial_radius * (1.0 - flattening);
    __m256d const polar_radius_f64x4 = _mm256_set1_pd(polar_radius);
    __m256d const flattening_f64x4 = _mm256_set1_pd(flattening);
    __m256d const second_eccentricity_squared_f64x4 = _mm256_set1_pd(
        (equatorial_radius * equatorial_radius - polar_radius * polar_radius) / (polar_radius * polar_radius));
    __m256d const convergence_threshold_f64x4 = _mm256_set1_pd(NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    __m256d const one_f64x4 = _mm256_set1_pd(1.0);
    __m256d const two_f64x4 = _mm256_set1_pd(2.0);
    __m256d const three_f64x4 = _mm256_set1_pd(3.0);
    __m256d const four_f64x4 = _mm256_set1_pd(4.0);
    __m256d const six_f64x4 = _mm256_set1_pd(6.0);
    __m256d const sixteen_f64x4 = _mm256_set1_pd(16.0);

    __m256d const longitude_difference_f64x4 = _mm256_sub_pd(second_longitudes_f64x4, first_longitudes_f64x4);
    __m256d lambda_f64x4 = longitude_difference_f64x4;
    __m256d sin_angular_distance_f64x4, cos_angular_distance_f64x4, angular_distance_f64x4;
    __m256d sin_azimuth_f64x4, cos_squared_azimuth_f64x4, cos_double_angular_midpoint_f64x4;
    __m256d converged_mask_f64x4 = _mm256_setzero_pd();
    __m256d coincident_mask_f64x4 = _mm256_setzero_pd();

    for (nk_u32_t iteration = 0; iteration < NUMKONG_VINCENTY_MAX_ITERATIONS; ++iteration) {
        if (_mm256_movemask_pd(converged_mask_f64x4) == 0xF) break;

        __m256d sin_lambda_f64x4 = nk_sin_f64x4_haswell_(lambda_f64x4);
        __m256d cos_lambda_f64x4 = nk_cos_f64x4_haswell_(lambda_f64x4);

        // sin²(σ) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂)
        //   - sin(U₁) × cos(U₂) × cos(λ))²
        __m256d cross_term_f64x4 = _mm256_mul_pd(cos_reduced_second_f64x4, sin_lambda_f64x4);
        __m256d mixed_term_f64x4 = _mm256_sub_pd(
            _mm256_mul_pd(cos_reduced_first_f64x4, sin_reduced_second_f64x4),
            _mm256_mul_pd(_mm256_mul_pd(sin_reduced_first_f64x4, cos_reduced_second_f64x4), cos_lambda_f64x4));
        __m256d sin_squared_angular_distance_f64x4 = _mm256_fmadd_pd(cross_term_f64x4, cross_term_f64x4,
                                                                     _mm256_mul_pd(mixed_term_f64x4, mixed_term_f64x4));
        sin_angular_distance_f64x4 = _mm256_sqrt_pd(sin_squared_angular_distance_f64x4);
        coincident_mask_f64x4 = _mm256_cmp_pd(sin_angular_distance_f64x4, _mm256_setzero_pd(), _CMP_EQ_OQ);

        // cos(σ) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance_f64x4 = _mm256_fmadd_pd(_mm256_mul_pd(cos_reduced_first_f64x4, cos_reduced_second_f64x4),
                                                     cos_lambda_f64x4,
                                                     _mm256_mul_pd(sin_reduced_first_f64x4, sin_reduced_second_f64x4));
        angular_distance_f64x4 = nk_atan2_f64x4_haswell_(sin_angular_distance_f64x4, cos_angular_distance_f64x4);

        // sin(α) = cos(U₁) × cos(U₂) × sin(λ) / sin(σ), dividing coincident lanes by one
        __m256d safe_sin_angular_distance_f64x4 = _mm256_blendv_pd(sin_angular_distance_f64x4, one_f64x4,
                                                                   coincident_mask_f64x4);
        sin_azimuth_f64x4 = _mm256_div_pd(
            _mm256_mul_pd(_mm256_mul_pd(cos_reduced_first_f64x4, cos_reduced_second_f64x4), sin_lambda_f64x4),
            safe_sin_angular_distance_f64x4);
        cos_squared_azimuth_f64x4 = _mm256_sub_pd(one_f64x4, _mm256_mul_pd(sin_azimuth_f64x4, sin_azimuth_f64x4));

        // An equatorial geodesic, cos²α = 0, has no midpoint term
        __m256d equatorial_mask_f64x4 = _mm256_cmp_pd(cos_squared_azimuth_f64x4, _mm256_setzero_pd(), _CMP_EQ_OQ);
        __m256d safe_cos_squared_azimuth_f64x4 = _mm256_blendv_pd(cos_squared_azimuth_f64x4, one_f64x4,
                                                                  equatorial_mask_f64x4);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        __m256d sin_product_f64x4 = _mm256_mul_pd(sin_reduced_first_f64x4, sin_reduced_second_f64x4);
        cos_double_angular_midpoint_f64x4 = _mm256_sub_pd(
            cos_angular_distance_f64x4,
            _mm256_div_pd(_mm256_mul_pd(two_f64x4, sin_product_f64x4), safe_cos_squared_azimuth_f64x4));
        cos_double_angular_midpoint_f64x4 = _mm256_blendv_pd(cos_double_angular_midpoint_f64x4, _mm256_setzero_pd(),
                                                             equatorial_mask_f64x4);

        // C = f/16 × cos²α × (4 + f × (4 - 3 × cos²α))
        __m256d correction_factor_f64x4 = _mm256_mul_pd(
            _mm256_div_pd(flattening_f64x4, sixteen_f64x4),
            _mm256_mul_pd(
                cos_squared_azimuth_f64x4,
                _mm256_fmadd_pd(flattening_f64x4, _mm256_fnmadd_pd(three_f64x4, cos_squared_azimuth_f64x4, four_f64x4),
                                four_f64x4)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × middle), where
        // middle = cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))
        __m256d cos_double_angular_midpoint_squared_f64x4 = _mm256_mul_pd(cos_double_angular_midpoint_f64x4,
                                                                          cos_double_angular_midpoint_f64x4);
        __m256d innermost_f64x4 = _mm256_fmadd_pd(two_f64x4, cos_double_angular_midpoint_squared_f64x4,
                                                  _mm256_set1_pd(-1.0));
        __m256d middle_f64x4 = _mm256_fmadd_pd(_mm256_mul_pd(correction_factor_f64x4, cos_angular_distance_f64x4),
                                               innermost_f64x4, cos_double_angular_midpoint_f64x4);
        __m256d inner_f64x4 = _mm256_mul_pd(_mm256_mul_pd(correction_factor_f64x4, sin_angular_distance_f64x4),
                                            middle_f64x4);
        __m256d lambda_next_f64x4 = _mm256_fmadd_pd(
            _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(one_f64x4, correction_factor_f64x4), flattening_f64x4),
                          sin_azimuth_f64x4),
            _mm256_add_pd(angular_distance_f64x4, inner_f64x4), longitude_difference_f64x4);

        __m256d lambda_change_f64x4 = _mm256_andnot_pd(_mm256_set1_pd(-0.0),
                                                       _mm256_sub_pd(lambda_next_f64x4, lambda_f64x4));
        // A lane stops where serial does: once its change is not above the threshold, NaN included
        __m256d newly_converged_f64x4 = _mm256_cmp_pd(lambda_change_f64x4, convergence_threshold_f64x4, _CMP_NGT_UQ);
        converged_mask_f64x4 = _mm256_or_pd(converged_mask_f64x4, newly_converged_f64x4);

        // Converged lanes keep the λ their outputs came from, as the serial loop stops on it
        lambda_f64x4 = _mm256_blendv_pd(lambda_next_f64x4, lambda_f64x4, converged_mask_f64x4);
    }

    // u² = cos²α × (a² - b²) / b²
    __m256d u_squared_f64x4 = _mm256_mul_pd(cos_squared_azimuth_f64x4, second_eccentricity_squared_f64x4);

    // A = 1 + u²/16384 × (4096 + u² × (-768 + u² × (320 - 175 × u²)))
    __m256d series_a_f64x4 = _mm256_fmadd_pd(u_squared_f64x4, _mm256_set1_pd(-175.0), _mm256_set1_pd(320.0));
    series_a_f64x4 = _mm256_fmadd_pd(u_squared_f64x4, series_a_f64x4, _mm256_set1_pd(-768.0));
    series_a_f64x4 = _mm256_fmadd_pd(u_squared_f64x4, series_a_f64x4, _mm256_set1_pd(4096.0));
    series_a_f64x4 = _mm256_fmadd_pd(_mm256_div_pd(u_squared_f64x4, _mm256_set1_pd(16384.0)), series_a_f64x4,
                                     one_f64x4);

    // B = u²/1024 × (256 + u² × (-128 + u² × (74 - 47 × u²)))
    __m256d series_b_f64x4 = _mm256_fmadd_pd(u_squared_f64x4, _mm256_set1_pd(-47.0), _mm256_set1_pd(74.0));
    series_b_f64x4 = _mm256_fmadd_pd(u_squared_f64x4, series_b_f64x4, _mm256_set1_pd(-128.0));
    series_b_f64x4 = _mm256_fmadd_pd(u_squared_f64x4, series_b_f64x4, _mm256_set1_pd(256.0));
    series_b_f64x4 = _mm256_mul_pd(_mm256_div_pd(u_squared_f64x4, _mm256_set1_pd(1024.0)), series_b_f64x4);

    // Δσ = B × sin(σ) × (cos(2σₘ) + B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) -
    //      B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m256d cos_double_angular_midpoint_squared_f64x4 = _mm256_mul_pd(cos_double_angular_midpoint_f64x4,
                                                                      cos_double_angular_midpoint_f64x4);
    __m256d sin_squared_angular_distance_f64x4 = _mm256_mul_pd(sin_angular_distance_f64x4, sin_angular_distance_f64x4);
    __m256d cosine_term_f64x4 = _mm256_mul_pd(
        cos_angular_distance_f64x4,
        _mm256_fmadd_pd(two_f64x4, cos_double_angular_midpoint_squared_f64x4, _mm256_set1_pd(-1.0)));
    __m256d sine_factor_f64x4 = _mm256_fmadd_pd(four_f64x4, sin_squared_angular_distance_f64x4, _mm256_set1_pd(-3.0));
    __m256d midpoint_factor_f64x4 = _mm256_fmadd_pd(four_f64x4, cos_double_angular_midpoint_squared_f64x4,
                                                    _mm256_set1_pd(-3.0));
    __m256d series_term_f64x4 = _mm256_mul_pd(
        _mm256_mul_pd(_mm256_div_pd(series_b_f64x4, six_f64x4), cos_double_angular_midpoint_f64x4),
        _mm256_mul_pd(sine_factor_f64x4, midpoint_factor_f64x4));
    __m256d angular_correction_f64x4 = _mm256_mul_pd(
        series_b_f64x4,
        _mm256_mul_pd(sin_angular_distance_f64x4,
                      _mm256_add_pd(cos_double_angular_midpoint_f64x4,
                                    _mm256_mul_pd(_mm256_div_pd(series_b_f64x4, four_f64x4),
                                                  _mm256_sub_pd(cosine_term_f64x4, series_term_f64x4)))));

    // s = b × A × (σ - Δσ), and zero for coincident points
    __m256d distances_f64x4 = _mm256_mul_pd(_mm256_mul_pd(polar_radius_f64x4, series_a_f64x4),
                                            _mm256_sub_pd(angular_distance_f64x4, angular_correction_f64x4));
    return _mm256_blendv_pd(distances_f64x4, _mm256_setzero_pd(), coincident_mask_f64x4);
}

#if NUMKONG_TARGET_HASWELL

NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_haswell(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                      nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                      nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_u64_t *output = (nk_u64_t *)((char *)results + row * results_stride);
        __m256i const a_x_i32x8 = _mm256_set1_epi32(a[row * 3]);
        __m256i const a_y_i32x8 = _mm256_set1_epi32(a[row * 3 + 1]);
        __m256i const a_z_i32x8 = _mm256_set1_epi32(a[row * 3 + 2]);
        nk_size_t column = 0;
        for (; column + 8 <= columns; column += 8) {
            nk_i32_t const *b_point = b + column * 3;
            __m256i sums_low_u64x4, sums_high_u64x4;
            nk_sqeuclidean3d_i32x8_haswell_(
                a_x_i32x8, a_y_i32x8, a_z_i32x8, _mm256_loadu_si256((__m256i const *)b_point),
                _mm256_loadu_si256((__m256i const *)(b_point + 8)), _mm256_loadu_si256((__m256i const *)(b_point + 16)),
                &sums_low_u64x4, &sums_high_u64x4);
            _mm256_storeu_si256((__m256i *)(output + column), sums_low_u64x4);
            _mm256_storeu_si256((__m256i *)(output + column + 4), sums_high_u64x4);
        }
        if (column < columns) {
            nk_size_t const remaining = columns - column, values = remaining * 3;
            nk_b256_vec_t b_first_vec, b_second_vec, b_third_vec, sums_low_vec, sums_high_vec;
            nk_partial_load_b32x8_haswell_(b + column * 3, &b_first_vec, values);
            nk_partial_load_b32x8_haswell_(b + column * 3 + 8, &b_second_vec, values > 8 ? values - 8 : 0);
            nk_partial_load_b32x8_haswell_(b + column * 3 + 16, &b_third_vec, values > 16 ? values - 16 : 0);
            nk_sqeuclidean3d_i32x8_haswell_(a_x_i32x8, a_y_i32x8, a_z_i32x8, b_first_vec.ymm, b_second_vec.ymm,
                                            b_third_vec.ymm, &sums_low_vec.ymm, &sums_high_vec.ymm);
            nk_partial_store_b64x4_haswell_(&sums_low_vec, output + column, remaining);
            nk_partial_store_b64x4_haswell_(&sums_high_vec, output + column + 4, remaining > 4 ? remaining - 4 : 0);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_haswell(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                      nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                      nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        __m256d const a_x_f64x4 = _mm256_set1_pd(a[row * 3]);
        __m256d const a_y_f64x4 = _mm256_set1_pd(a[row * 3 + 1]);
        __m256d const a_z_f64x4 = _mm256_set1_pd(a[row * 3 + 2]);
        nk_size_t column = 0;
        for (; column + 4 <= columns; column += 4) {
            nk_f64_t const *b_point = b + column * 3;
            _mm256_storeu_pd(output + column, nk_sqeuclidean3d_f64x4_haswell_(
                                                  a_x_f64x4, a_y_f64x4, a_z_f64x4, _mm256_loadu_pd(b_point),
                                                  _mm256_loadu_pd(b_point + 4), _mm256_loadu_pd(b_point + 8)));
        }
        if (column < columns) {
            nk_size_t const remaining = columns - column, values = remaining * 3;
            nk_b256_vec_t b_first_vec, b_second_vec, b_third_vec, distances_vec;
            nk_partial_load_b64x4_haswell_(b + column * 3, &b_first_vec, values);
            nk_partial_load_b64x4_haswell_(b + column * 3 + 4, &b_second_vec, values > 4 ? values - 4 : 0);
            nk_partial_load_b64x4_haswell_(b + column * 3 + 8, &b_third_vec, values > 8 ? values - 8 : 0);
            distances_vec.ymm_pd = nk_sqeuclidean3d_f64x4_haswell_(a_x_f64x4, a_y_f64x4, a_z_f64x4, b_first_vec.ymm_pd,
                                                                   b_second_vec.ymm_pd, b_third_vec.ymm_pd);
            nk_partial_store_b64x4_haswell_(&distances_vec, output + column, remaining);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_haversines_packed_f64_haswell(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                         nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t rows, columns, a_stride, b_stride;
    nk_f64_t const *a_packed_x = nk_geospatial_packed_axes_f64_serial_(a_packed, &rows, &a_stride);
    nk_f64_t const *b_packed_x = nk_geospatial_packed_axes_f64_serial_(b_packed, &columns, &b_stride);
    nk_f64_t const *a_packed_y = a_packed_x + a_stride, *a_packed_z = a_packed_y + a_stride;
    nk_f64_t const *b_packed_y = b_packed_x + b_stride, *b_packed_z = b_packed_y + b_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        __m256d const a_x_f64x4 = _mm256_set1_pd(a_packed_x[row]);
        __m256d const a_y_f64x4 = _mm256_set1_pd(a_packed_y[row]);
        __m256d const a_z_f64x4 = _mm256_set1_pd(a_packed_z[row]);
        nk_size_t column = 0;
        for (; column + 4 <= columns; column += 4)
            _mm256_storeu_pd(output + column, nk_haversine_f64x4_haswell_(a_x_f64x4, a_y_f64x4, a_z_f64x4,
                                                                          _mm256_loadu_pd(b_packed_x + column),
                                                                          _mm256_loadu_pd(b_packed_y + column),
                                                                          _mm256_loadu_pd(b_packed_z + column)));
        if (column < columns) {
            nk_b256_vec_t b_x_vec, b_y_vec, b_z_vec, distances_vec;
            nk_partial_load_b64x4_haswell_(b_packed_x + column, &b_x_vec, columns - column);
            nk_partial_load_b64x4_haswell_(b_packed_y + column, &b_y_vec, columns - column);
            nk_partial_load_b64x4_haswell_(b_packed_z + column, &b_z_vec, columns - column);
            distances_vec.ymm_pd = nk_haversine_f64x4_haswell_(a_x_f64x4, a_y_f64x4, a_z_f64x4, b_x_vec.ymm_pd,
                                                               b_y_vec.ymm_pd, b_z_vec.ymm_pd);
            nk_partial_store_b64x4_haswell_(&distances_vec, output + column, columns - column);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_vincentys_packed_f64_haswell(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                        nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t rows, columns, a_stride, b_stride;
    nk_f64_t const *a_packed_longitudes = nk_geospatial_packed_axes_f64_serial_(a_packed, &rows, &a_stride);
    nk_f64_t const *b_packed_longitudes = nk_geospatial_packed_axes_f64_serial_(b_packed, &columns, &b_stride);
    nk_f64_t const *a_packed_sines = a_packed_longitudes + a_stride, *a_packed_cosines = a_packed_sines + a_stride;
    nk_f64_t const *b_packed_sines = b_packed_longitudes + b_stride, *b_packed_cosines = b_packed_sines + b_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        __m256d const a_longitude_f64x4 = _mm256_set1_pd(a_packed_longitudes[row]);
        __m256d const a_sine_f64x4 = _mm256_set1_pd(a_packed_sines[row]);
        __m256d const a_cosine_f64x4 = _mm256_set1_pd(a_packed_cosines[row]);
        nk_size_t column = 0;
        for (; column + 4 <= columns; column += 4)
            _mm256_storeu_pd(output + column,
                             nk_vincenty_f64x4_haswell_(a_longitude_f64x4, a_sine_f64x4, a_cosine_f64x4,
                                                        _mm256_loadu_pd(b_packed_longitudes + column),
                                                        _mm256_loadu_pd(b_packed_sines + column),
                                                        _mm256_loadu_pd(b_packed_cosines + column)));
        if (column < columns) {
            nk_b256_vec_t b_longitude_vec, b_sine_vec, b_cosine_vec, distances_vec;
            nk_partial_load_b64x4_haswell_(b_packed_longitudes + column, &b_longitude_vec, columns - column);
            nk_partial_load_b64x4_haswell_(b_packed_sines + column, &b_sine_vec, columns - column);
            nk_partial_load_b64x4_haswell_(b_packed_cosines + column, &b_cosine_vec, columns - column);
            distances_vec.ymm_pd = nk_vincenty_f64x4_haswell_(a_longitude_f64x4, a_sine_f64x4, a_cosine_f64x4,
                                                              b_longitude_vec.ymm_pd, b_sine_vec.ymm_pd,
                                                              b_cosine_vec.ymm_pd);
            nk_partial_store_b64x4_haswell_(&distances_vec, output + column, columns - column);
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
#endif // NUMKONG_GEOSPATIAL_HASWELL_H
