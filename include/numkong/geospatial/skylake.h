/**
 *  @file include/numkong/geospatial/skylake.h
 *  @author Ash Vardanian
 *  @date December 21, 2025
 *  @brief SIMD-accelerated geospatial distances for Skylake.
 *
 *  @sa include/numkong/geospatial.h
 *
 *  Every kernel broadcasts one row and sweeps the columns eight or sixteen at a time, with masked
 *  loads and stores for the last partial step. The 3D squares load three registers of @b [x,y,z]
 *  rows: each lane finds a given axis in exactly one of them, so two masked blends gather the axis
 *  and one @c VPERMD or @c VPERMPD orders it. The @c i32 order puts point k in the even half of
 *  64-bit lane k and point k + 8 in its odd half, so @c VPMULDQ squares the first eight points
 *  directly and the last eight after a 32-bit shift, and both halves store contiguously. Haversine
 *  and Vincenty read the planar packs with plain loads.
 *
 *  @section geospatial_skylake_instructions Key AVX-512 Geospatial Instructions
 *
 *  @verbatim
 *  Intrinsic           Instruction                  Icelake           Genoa
 *  _mm512_mul_epi32    VPMULDQ (ZMM, ZMM, ZMM)      5cy @ p0          3cy @ p01
 *  _mm512_sqrt_pd      VSQRTPD (ZMM, ZMM)           23cy @ p0+p0+p05  21cy @ p01
 *  _mm256_div_pd       VDIVPD (YMM, YMM, YMM)       13cy @ p0         13cy @ p01
 *  _mm256_fmadd_pd     VFMADD231PD (YMM, YMM, YMM)  4cy @ p01         4cy @ p01
 *  _mm512_cmp_pd_mask  VCMPPD (K, ZMM, ZMM, I8)     4cy @ p5          5cy @ p01
 *  @endverbatim
 */
#ifndef NUMKONG_GEOSPATIAL_SKYLAKE_H
#define NUMKONG_GEOSPATIAL_SKYLAKE_H

#if NUMKONG_ARCH_X8664_
#if NUMKONG_ARCH_X8664_SKYLAKE_

#include "numkong/types.h"
#include "numkong/geospatial/serial.h"    // `nk_geospatial_packed_axes_f64_serial_`
#include "numkong/dot/skylake.h"          // `nk_dot2_f64x8_skylake_`
#include "numkong/trigonometry/skylake.h" // `nk_sin_f64x8_skylake_`, `nk_cos_f64x8_skylake_`, `nk_atan2_f64x8_skylake_`

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

/** Exact squared distances from a broadcast 3D point to 16 interleaved @b [x,y,z] points in three
 *  registers, the first eight in @p sums_low_u64x8 and the last eight in @p sums_high_u64x8. */
NUMKONG_INLINE void nk_sqeuclidean3d_i32x16_skylake_(                        //
    __m512i a_x_i32x16, __m512i a_y_i32x16, __m512i a_z_i32x16,              //
    __m512i b_first_i32x16, __m512i b_second_i32x16, __m512i b_third_i32x16, //
    __m512i *sums_low_u64x8, __m512i *sums_high_u64x8) {
    __m512i const b_x_i32x16 = _mm512_permutexvar_epi32(
        _mm512_setr_epi32(0, 8, 3, 11, 6, 14, 9, 1, 12, 4, 15, 7, 2, 10, 5, 13),
        _mm512_mask_blend_epi32(0x2492, _mm512_mask_blend_epi32(0x4924, b_first_i32x16, b_second_i32x16),
                                b_third_i32x16));
    __m512i const b_y_i32x16 = _mm512_permutexvar_epi32(
        _mm512_setr_epi32(1, 9, 4, 12, 7, 15, 10, 2, 13, 5, 0, 8, 3, 11, 6, 14),
        _mm512_mask_blend_epi32(0x4924, _mm512_mask_blend_epi32(0x9249, b_first_i32x16, b_second_i32x16),
                                b_third_i32x16));
    __m512i const b_z_i32x16 = _mm512_permutexvar_epi32(
        _mm512_setr_epi32(2, 10, 5, 13, 8, 0, 11, 3, 14, 6, 1, 9, 4, 12, 7, 15),
        _mm512_mask_blend_epi32(0x9249, _mm512_mask_blend_epi32(0x2492, b_first_i32x16, b_second_i32x16),
                                b_third_i32x16));
    __m512i const difference_x_i32x16 = _mm512_sub_epi32(a_x_i32x16, b_x_i32x16);
    __m512i const difference_y_i32x16 = _mm512_sub_epi32(a_y_i32x16, b_y_i32x16);
    __m512i const difference_z_i32x16 = _mm512_sub_epi32(a_z_i32x16, b_z_i32x16);
    __m512i const difference_x_odd_i32x16 = _mm512_srli_epi64(difference_x_i32x16, 32);
    __m512i const difference_y_odd_i32x16 = _mm512_srli_epi64(difference_y_i32x16, 32);
    __m512i const difference_z_odd_i32x16 = _mm512_srli_epi64(difference_z_i32x16, 32);
    *sums_low_u64x8 = _mm512_add_epi64(_mm512_add_epi64(_mm512_mul_epi32(difference_x_i32x16, difference_x_i32x16),
                                                        _mm512_mul_epi32(difference_y_i32x16, difference_y_i32x16)),
                                       _mm512_mul_epi32(difference_z_i32x16, difference_z_i32x16));
    *sums_high_u64x8 = _mm512_add_epi64(
        _mm512_add_epi64(_mm512_mul_epi32(difference_x_odd_i32x16, difference_x_odd_i32x16),
                         _mm512_mul_epi32(difference_y_odd_i32x16, difference_y_odd_i32x16)),
        _mm512_mul_epi32(difference_z_odd_i32x16, difference_z_odd_i32x16));
}

/** Squared distances from a broadcast 3D point to 8 interleaved @b [x,y,z] points in three
 *  registers, the three squares summed in Dot2. */
NUMKONG_INLINE __m512d nk_sqeuclidean3d_f64x8_skylake_(      //
    __m512d a_x_f64x8, __m512d a_y_f64x8, __m512d a_z_f64x8, //
    __m512d b_first_f64x8, __m512d b_second_f64x8, __m512d b_third_f64x8) {
    __m512d const b_x_f64x8 = _mm512_permutexvar_pd(
        _mm512_setr_epi64(0, 3, 6, 1, 4, 7, 2, 5),
        _mm512_mask_blend_pd(0x24, _mm512_mask_blend_pd(0x92, b_first_f64x8, b_second_f64x8), b_third_f64x8));
    __m512d const b_y_f64x8 = _mm512_permutexvar_pd(
        _mm512_setr_epi64(1, 4, 7, 2, 5, 0, 3, 6),
        _mm512_mask_blend_pd(0x49, _mm512_mask_blend_pd(0x24, b_first_f64x8, b_second_f64x8), b_third_f64x8));
    __m512d const b_z_f64x8 = _mm512_permutexvar_pd(
        _mm512_setr_epi64(2, 5, 0, 3, 6, 1, 4, 7),
        _mm512_mask_blend_pd(0x92, _mm512_mask_blend_pd(0x49, b_first_f64x8, b_second_f64x8), b_third_f64x8));
    __m512d const difference_x_f64x8 = _mm512_sub_pd(a_x_f64x8, b_x_f64x8);
    __m512d const difference_y_f64x8 = _mm512_sub_pd(a_y_f64x8, b_y_f64x8);
    __m512d const difference_z_f64x8 = _mm512_sub_pd(a_z_f64x8, b_z_f64x8);
    __m512d sum_f64x8 = _mm512_setzero_pd(), compensation_f64x8 = _mm512_setzero_pd();
    nk_dot2_f64x8_skylake_(&sum_f64x8, &compensation_f64x8, difference_x_f64x8, difference_x_f64x8);
    nk_dot2_f64x8_skylake_(&sum_f64x8, &compensation_f64x8, difference_y_f64x8, difference_y_f64x8);
    nk_dot2_f64x8_skylake_(&sum_f64x8, &compensation_f64x8, difference_z_f64x8, difference_z_f64x8);
    return _mm512_add_pd(sum_f64x8, compensation_f64x8);
}

/** Great-circle metres between lanes of unit vectors, 2R · atan2(‖a − b‖, ‖a + b‖),
 *  with both norms summed in Dot2. */
NUMKONG_INLINE __m512d nk_haversine_f64x8_skylake_(                      //
    __m512d first_x_f64x8, __m512d first_y_f64x8, __m512d first_z_f64x8, //
    __m512d second_x_f64x8, __m512d second_y_f64x8, __m512d second_z_f64x8) {
    __m512d const difference_x_f64x8 = _mm512_sub_pd(first_x_f64x8, second_x_f64x8);
    __m512d const difference_y_f64x8 = _mm512_sub_pd(first_y_f64x8, second_y_f64x8);
    __m512d const difference_z_f64x8 = _mm512_sub_pd(first_z_f64x8, second_z_f64x8);
    __m512d const sum_x_f64x8 = _mm512_add_pd(first_x_f64x8, second_x_f64x8);
    __m512d const sum_y_f64x8 = _mm512_add_pd(first_y_f64x8, second_y_f64x8);
    __m512d const sum_z_f64x8 = _mm512_add_pd(first_z_f64x8, second_z_f64x8);
    __m512d chord_f64x8 = _mm512_setzero_pd(), chord_compensation_f64x8 = _mm512_setzero_pd();
    __m512d span_f64x8 = _mm512_setzero_pd(), span_compensation_f64x8 = _mm512_setzero_pd();
    nk_dot2_f64x8_skylake_(&chord_f64x8, &chord_compensation_f64x8, difference_x_f64x8, difference_x_f64x8);
    nk_dot2_f64x8_skylake_(&chord_f64x8, &chord_compensation_f64x8, difference_y_f64x8, difference_y_f64x8);
    nk_dot2_f64x8_skylake_(&chord_f64x8, &chord_compensation_f64x8, difference_z_f64x8, difference_z_f64x8);
    nk_dot2_f64x8_skylake_(&span_f64x8, &span_compensation_f64x8, sum_x_f64x8, sum_x_f64x8);
    nk_dot2_f64x8_skylake_(&span_f64x8, &span_compensation_f64x8, sum_y_f64x8, sum_y_f64x8);
    nk_dot2_f64x8_skylake_(&span_f64x8, &span_compensation_f64x8, sum_z_f64x8, sum_z_f64x8);
    __m512d const angle_f64x8 = nk_atan2_f64x8_skylake_(
        _mm512_sqrt_pd(_mm512_add_pd(chord_f64x8, chord_compensation_f64x8)),
        _mm512_sqrt_pd(_mm512_add_pd(span_f64x8, span_compensation_f64x8)));
    return _mm512_mul_pd(_mm512_set1_pd(2.0 * NUMKONG_EARTH_MEAN_RADIUS), angle_f64x8);
}

/**
 *  @brief AVX-512 helper for Vincenty's geodesic distance on 8 f64 point pairs, from their
 *      longitudes and the sines and cosines of their reduced latitudes.
 *  @note This is a true SIMD implementation using masked convergence tracking.
 */
NUMKONG_INLINE __m512d nk_vincenty_f64x8_skylake_(                                                    //
    __m512d first_longitudes_f64x8, __m512d sin_reduced_first_f64x8, __m512d cos_reduced_first_f64x8, //
    __m512d second_longitudes_f64x8, __m512d sin_reduced_second_f64x8, __m512d cos_reduced_second_f64x8) {

    nk_f64_t const equatorial_radius = NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS;
    nk_f64_t const flattening = 1.0 / NUMKONG_EARTH_ELLIPSOID_INVERSE_FLATTENING;
    nk_f64_t const polar_radius = equatorial_radius * (1.0 - flattening);
    __m512d const polar_radius_f64x8 = _mm512_set1_pd(polar_radius);
    __m512d const flattening_f64x8 = _mm512_set1_pd(flattening);
    __m512d const second_eccentricity_squared_f64x8 = _mm512_set1_pd(
        (equatorial_radius * equatorial_radius - polar_radius * polar_radius) / (polar_radius * polar_radius));
    __m512d const convergence_threshold_f64x8 = _mm512_set1_pd(NUMKONG_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    __m512d const one_f64x8 = _mm512_set1_pd(1.0);
    __m512d const two_f64x8 = _mm512_set1_pd(2.0);
    __m512d const three_f64x8 = _mm512_set1_pd(3.0);
    __m512d const four_f64x8 = _mm512_set1_pd(4.0);
    __m512d const six_f64x8 = _mm512_set1_pd(6.0);
    __m512d const sixteen_f64x8 = _mm512_set1_pd(16.0);

    __m512d const longitude_difference_f64x8 = _mm512_sub_pd(second_longitudes_f64x8, first_longitudes_f64x8);
    __m512d lambda_f64x8 = longitude_difference_f64x8;
    __m512d sin_angular_distance_f64x8, cos_angular_distance_f64x8, angular_distance_f64x8;
    __m512d sin_azimuth_f64x8, cos_squared_azimuth_f64x8, cos_double_angular_midpoint_f64x8;
    __mmask8 converged_m8 = 0;
    __mmask8 coincident_m8 = 0;

    for (nk_u32_t iteration = 0; iteration < NUMKONG_VINCENTY_MAX_ITERATIONS && converged_m8 != 0xFF; ++iteration) {
        __m512d sin_lambda_f64x8 = nk_sin_f64x8_skylake_(lambda_f64x8);
        __m512d cos_lambda_f64x8 = nk_cos_f64x8_skylake_(lambda_f64x8);

        // sin²(σ) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂)
        //   - sin(U₁) × cos(U₂) × cos(λ))²
        __m512d cross_term_f64x8 = _mm512_mul_pd(cos_reduced_second_f64x8, sin_lambda_f64x8);
        __m512d mixed_term_f64x8 = _mm512_sub_pd(
            _mm512_mul_pd(cos_reduced_first_f64x8, sin_reduced_second_f64x8),
            _mm512_mul_pd(_mm512_mul_pd(sin_reduced_first_f64x8, cos_reduced_second_f64x8), cos_lambda_f64x8));
        __m512d sin_squared_angular_distance_f64x8 = _mm512_fmadd_pd(cross_term_f64x8, cross_term_f64x8,
                                                                     _mm512_mul_pd(mixed_term_f64x8, mixed_term_f64x8));
        sin_angular_distance_f64x8 = _mm512_sqrt_pd(sin_squared_angular_distance_f64x8);
        coincident_m8 = _mm512_cmp_pd_mask(sin_angular_distance_f64x8, _mm512_setzero_pd(), _CMP_EQ_OQ);

        // cos(σ) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance_f64x8 = _mm512_fmadd_pd(_mm512_mul_pd(cos_reduced_first_f64x8, cos_reduced_second_f64x8),
                                                     cos_lambda_f64x8,
                                                     _mm512_mul_pd(sin_reduced_first_f64x8, sin_reduced_second_f64x8));
        angular_distance_f64x8 = nk_atan2_f64x8_skylake_(sin_angular_distance_f64x8, cos_angular_distance_f64x8);

        // sin(α) = cos(U₁) × cos(U₂) × sin(λ) / sin(σ), left zero in coincident lanes
        sin_azimuth_f64x8 = _mm512_maskz_div_pd(
            _knot_mask8(coincident_m8),
            _mm512_mul_pd(_mm512_mul_pd(cos_reduced_first_f64x8, cos_reduced_second_f64x8), sin_lambda_f64x8),
            sin_angular_distance_f64x8);
        cos_squared_azimuth_f64x8 = _mm512_sub_pd(one_f64x8, _mm512_mul_pd(sin_azimuth_f64x8, sin_azimuth_f64x8));

        // An equatorial geodesic, cos²α = 0, passes cos(σ) through: no midpoint term
        __mmask8 equatorial_m8 = _mm512_cmp_pd_mask(cos_squared_azimuth_f64x8, _mm512_setzero_pd(), _CMP_EQ_OQ);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        __m512d sin_product_f64x8 = _mm512_mul_pd(sin_reduced_first_f64x8, sin_reduced_second_f64x8);
        __m512d quotient_f64x8 = _mm512_mask_div_pd(cos_angular_distance_f64x8, _knot_mask8(equatorial_m8),
                                                    _mm512_mul_pd(two_f64x8, sin_product_f64x8),
                                                    cos_squared_azimuth_f64x8);
        cos_double_angular_midpoint_f64x8 = _mm512_sub_pd(cos_angular_distance_f64x8, quotient_f64x8);

        // C = f/16 × cos²α × (4 + f × (4 - 3 × cos²α))
        __m512d correction_factor_f64x8 = _mm512_mul_pd(
            _mm512_div_pd(flattening_f64x8, sixteen_f64x8),
            _mm512_mul_pd(
                cos_squared_azimuth_f64x8,
                _mm512_fmadd_pd(flattening_f64x8, _mm512_fnmadd_pd(three_f64x8, cos_squared_azimuth_f64x8, four_f64x8),
                                four_f64x8)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × middle), where
        // middle = cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))
        __m512d cos_double_angular_midpoint_squared_f64x8 = _mm512_mul_pd(cos_double_angular_midpoint_f64x8,
                                                                          cos_double_angular_midpoint_f64x8);
        __m512d innermost_f64x8 = _mm512_fmadd_pd(two_f64x8, cos_double_angular_midpoint_squared_f64x8,
                                                  _mm512_set1_pd(-1.0));
        __m512d middle_f64x8 = _mm512_fmadd_pd(_mm512_mul_pd(correction_factor_f64x8, cos_angular_distance_f64x8),
                                               innermost_f64x8, cos_double_angular_midpoint_f64x8);
        __m512d inner_f64x8 = _mm512_mul_pd(_mm512_mul_pd(correction_factor_f64x8, sin_angular_distance_f64x8),
                                            middle_f64x8);
        __m512d lambda_next_f64x8 = _mm512_fmadd_pd(
            _mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(one_f64x8, correction_factor_f64x8), flattening_f64x8),
                          sin_azimuth_f64x8),
            _mm512_add_pd(angular_distance_f64x8, inner_f64x8), longitude_difference_f64x8);

        __m512d lambda_change_f64x8 = _mm512_abs_pd(_mm512_sub_pd(lambda_next_f64x8, lambda_f64x8));
        // A lane stops where serial does: once its change is not above the threshold, NaN included
        converged_m8 = (__mmask8)(converged_m8 |
                                  _mm512_cmp_pd_mask(lambda_change_f64x8, convergence_threshold_f64x8, _CMP_NGT_UQ));

        // Converged lanes keep the λ their outputs came from, as the serial loop stops on it
        lambda_f64x8 = _mm512_mask_blend_pd(converged_m8, lambda_next_f64x8, lambda_f64x8);
    }

    // u² = cos²α × (a² - b²) / b²
    __m512d u_squared_f64x8 = _mm512_mul_pd(cos_squared_azimuth_f64x8, second_eccentricity_squared_f64x8);

    // A = 1 + u²/16384 × (4096 + u² × (-768 + u² × (320 - 175 × u²)))
    __m512d series_a_f64x8 = _mm512_fmadd_pd(u_squared_f64x8, _mm512_set1_pd(-175.0), _mm512_set1_pd(320.0));
    series_a_f64x8 = _mm512_fmadd_pd(u_squared_f64x8, series_a_f64x8, _mm512_set1_pd(-768.0));
    series_a_f64x8 = _mm512_fmadd_pd(u_squared_f64x8, series_a_f64x8, _mm512_set1_pd(4096.0));
    series_a_f64x8 = _mm512_fmadd_pd(_mm512_div_pd(u_squared_f64x8, _mm512_set1_pd(16384.0)), series_a_f64x8,
                                     one_f64x8);

    // B = u²/1024 × (256 + u² × (-128 + u² × (74 - 47 × u²)))
    __m512d series_b_f64x8 = _mm512_fmadd_pd(u_squared_f64x8, _mm512_set1_pd(-47.0), _mm512_set1_pd(74.0));
    series_b_f64x8 = _mm512_fmadd_pd(u_squared_f64x8, series_b_f64x8, _mm512_set1_pd(-128.0));
    series_b_f64x8 = _mm512_fmadd_pd(u_squared_f64x8, series_b_f64x8, _mm512_set1_pd(256.0));
    series_b_f64x8 = _mm512_mul_pd(_mm512_div_pd(u_squared_f64x8, _mm512_set1_pd(1024.0)), series_b_f64x8);

    // Δσ = B × sin(σ) × (cos(2σₘ) + B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) -
    //      B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m512d cos_double_angular_midpoint_squared_f64x8 = _mm512_mul_pd(cos_double_angular_midpoint_f64x8,
                                                                      cos_double_angular_midpoint_f64x8);
    __m512d sin_squared_angular_distance_f64x8 = _mm512_mul_pd(sin_angular_distance_f64x8, sin_angular_distance_f64x8);
    __m512d cosine_term_f64x8 = _mm512_mul_pd(
        cos_angular_distance_f64x8,
        _mm512_fmadd_pd(two_f64x8, cos_double_angular_midpoint_squared_f64x8, _mm512_set1_pd(-1.0)));
    __m512d sine_factor_f64x8 = _mm512_fmadd_pd(four_f64x8, sin_squared_angular_distance_f64x8, _mm512_set1_pd(-3.0));
    __m512d midpoint_factor_f64x8 = _mm512_fmadd_pd(four_f64x8, cos_double_angular_midpoint_squared_f64x8,
                                                    _mm512_set1_pd(-3.0));
    __m512d series_term_f64x8 = _mm512_mul_pd(
        _mm512_mul_pd(_mm512_div_pd(series_b_f64x8, six_f64x8), cos_double_angular_midpoint_f64x8),
        _mm512_mul_pd(sine_factor_f64x8, midpoint_factor_f64x8));
    __m512d angular_correction_f64x8 = _mm512_mul_pd(
        series_b_f64x8,
        _mm512_mul_pd(sin_angular_distance_f64x8,
                      _mm512_add_pd(cos_double_angular_midpoint_f64x8,
                                    _mm512_mul_pd(_mm512_div_pd(series_b_f64x8, four_f64x8),
                                                  _mm512_sub_pd(cosine_term_f64x8, series_term_f64x8)))));

    // s = b × A × (σ - Δσ), and zero for coincident points
    __m512d distances_f64x8 = _mm512_mul_pd(_mm512_mul_pd(polar_radius_f64x8, series_a_f64x8),
                                            _mm512_sub_pd(angular_distance_f64x8, angular_correction_f64x8));
    return _mm512_mask_blend_pd(coincident_m8, distances_f64x8, _mm512_setzero_pd());
}

#if NUMKONG_TARGET_SKYLAKE

NUMKONG_API nk_status_t nk_sqeuclideans3d_i32_skylake(nk_i32_t const *a, nk_size_t rows, nk_i32_t const *b,
                                                      nk_size_t columns, nk_u64_t *results, nk_size_t results_stride,
                                                      nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_u64_t *output = (nk_u64_t *)((char *)results + row * results_stride);
        __m512i const a_x_i32x16 = _mm512_set1_epi32(a[row * 3]);
        __m512i const a_y_i32x16 = _mm512_set1_epi32(a[row * 3 + 1]);
        __m512i const a_z_i32x16 = _mm512_set1_epi32(a[row * 3 + 2]);
        nk_size_t column = 0;
        for (; column + 16 <= columns; column += 16) {
            nk_i32_t const *b_point = b + column * 3;
            __m512i sums_low_u64x8, sums_high_u64x8;
            nk_sqeuclidean3d_i32x16_skylake_(a_x_i32x16, a_y_i32x16, a_z_i32x16, _mm512_loadu_si512(b_point),
                                             _mm512_loadu_si512(b_point + 16), _mm512_loadu_si512(b_point + 32),
                                             &sums_low_u64x8, &sums_high_u64x8);
            _mm512_storeu_si512(output + column, sums_low_u64x8);
            _mm512_storeu_si512(output + column + 8, sums_high_u64x8);
        }
        if (column < columns) {
            nk_i32_t const *b_point = b + column * 3;
            nk_u64_t const value_bits = _bzhi_u64(0xFFFFFFFFFFFFull, (columns - column) * 3);
            __mmask16 const point_m16 = (__mmask16)_bzhi_u32(0xFFFF, columns - column);
            __m512i sums_low_u64x8, sums_high_u64x8;
            nk_sqeuclidean3d_i32x16_skylake_(a_x_i32x16, a_y_i32x16, a_z_i32x16,
                                             _mm512_maskz_loadu_epi32((__mmask16)value_bits, b_point),
                                             _mm512_maskz_loadu_epi32((__mmask16)(value_bits >> 16), b_point + 16),
                                             _mm512_maskz_loadu_epi32((__mmask16)(value_bits >> 32), b_point + 32),
                                             &sums_low_u64x8, &sums_high_u64x8);
            _mm512_mask_storeu_epi64(output + column, (__mmask8)point_m16, sums_low_u64x8);
            _mm512_mask_storeu_epi64(output + column + 8, (__mmask8)(point_m16 >> 8), sums_high_u64x8);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_sqeuclideans3d_f64_skylake(nk_f64_t const *a, nk_size_t rows, nk_f64_t const *b,
                                                      nk_size_t columns, nk_f64_t *results, nk_size_t results_stride,
                                                      nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        __m512d const a_x_f64x8 = _mm512_set1_pd(a[row * 3]);
        __m512d const a_y_f64x8 = _mm512_set1_pd(a[row * 3 + 1]);
        __m512d const a_z_f64x8 = _mm512_set1_pd(a[row * 3 + 2]);
        nk_size_t column = 0;
        for (; column + 8 <= columns; column += 8) {
            nk_f64_t const *b_point = b + column * 3;
            _mm512_storeu_pd(output + column, nk_sqeuclidean3d_f64x8_skylake_(
                                                  a_x_f64x8, a_y_f64x8, a_z_f64x8, _mm512_loadu_pd(b_point),
                                                  _mm512_loadu_pd(b_point + 8), _mm512_loadu_pd(b_point + 16)));
        }
        if (column < columns) {
            nk_f64_t const *b_point = b + column * 3;
            nk_u32_t const value_bits = _bzhi_u32(0xFFFFFF, (columns - column) * 3);
            __m512d const distances_f64x8 = nk_sqeuclidean3d_f64x8_skylake_(
                a_x_f64x8, a_y_f64x8, a_z_f64x8, _mm512_maskz_loadu_pd((__mmask8)value_bits, b_point),
                _mm512_maskz_loadu_pd((__mmask8)(value_bits >> 8), b_point + 8),
                _mm512_maskz_loadu_pd((__mmask8)(value_bits >> 16), b_point + 16));
            _mm512_mask_storeu_pd(output + column, (__mmask8)_bzhi_u32(0xFF, columns - column), distances_f64x8);
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_haversines_packed_f64_skylake(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                         nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t rows, columns, a_stride, b_stride;
    nk_f64_t const *a_packed_x = nk_geospatial_packed_axes_f64_serial_(a_packed, &rows, &a_stride);
    nk_f64_t const *b_packed_x = nk_geospatial_packed_axes_f64_serial_(b_packed, &columns, &b_stride);
    nk_f64_t const *a_packed_y = a_packed_x + a_stride, *a_packed_z = a_packed_y + a_stride;
    nk_f64_t const *b_packed_y = b_packed_x + b_stride, *b_packed_z = b_packed_y + b_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        __m512d const a_x_f64x8 = _mm512_set1_pd(a_packed_x[row]);
        __m512d const a_y_f64x8 = _mm512_set1_pd(a_packed_y[row]);
        __m512d const a_z_f64x8 = _mm512_set1_pd(a_packed_z[row]);
        nk_size_t column = 0;
        for (; column + 8 <= columns; column += 8)
            _mm512_storeu_pd(output + column, nk_haversine_f64x8_skylake_(a_x_f64x8, a_y_f64x8, a_z_f64x8,
                                                                          _mm512_loadu_pd(b_packed_x + column),
                                                                          _mm512_loadu_pd(b_packed_y + column),
                                                                          _mm512_loadu_pd(b_packed_z + column)));
        if (column < columns) {
            __mmask8 const mask_m8 = (__mmask8)_bzhi_u32(0xFF, columns - column);
            _mm512_mask_storeu_pd(output + column, mask_m8,
                                  nk_haversine_f64x8_skylake_(a_x_f64x8, a_y_f64x8, a_z_f64x8,
                                                              _mm512_maskz_loadu_pd(mask_m8, b_packed_x + column),
                                                              _mm512_maskz_loadu_pd(mask_m8, b_packed_y + column),
                                                              _mm512_maskz_loadu_pd(mask_m8, b_packed_z + column)));
        }
    }
    return nk_success_k;
}

NUMKONG_API nk_status_t nk_vincentys_packed_f64_skylake(void const *a_packed, void const *b_packed, nk_f64_t *results,
                                                        nk_size_t results_stride, nk_stream_t stream) {
    nk_assert_(stream == NUMKONG_NULL);
    nk_size_t rows, columns, a_stride, b_stride;
    nk_f64_t const *a_packed_longitudes = nk_geospatial_packed_axes_f64_serial_(a_packed, &rows, &a_stride);
    nk_f64_t const *b_packed_longitudes = nk_geospatial_packed_axes_f64_serial_(b_packed, &columns, &b_stride);
    nk_f64_t const *a_packed_sines = a_packed_longitudes + a_stride, *a_packed_cosines = a_packed_sines + a_stride;
    nk_f64_t const *b_packed_sines = b_packed_longitudes + b_stride, *b_packed_cosines = b_packed_sines + b_stride;
    for (nk_size_t row = 0; row != rows; ++row) {
        nk_f64_t *output = (nk_f64_t *)((char *)results + row * results_stride);
        __m512d const a_longitude_f64x8 = _mm512_set1_pd(a_packed_longitudes[row]);
        __m512d const a_sine_f64x8 = _mm512_set1_pd(a_packed_sines[row]);
        __m512d const a_cosine_f64x8 = _mm512_set1_pd(a_packed_cosines[row]);
        nk_size_t column = 0;
        for (; column + 8 <= columns; column += 8)
            _mm512_storeu_pd(output + column,
                             nk_vincenty_f64x8_skylake_(a_longitude_f64x8, a_sine_f64x8, a_cosine_f64x8,
                                                        _mm512_loadu_pd(b_packed_longitudes + column),
                                                        _mm512_loadu_pd(b_packed_sines + column),
                                                        _mm512_loadu_pd(b_packed_cosines + column)));
        if (column < columns) {
            __mmask8 const mask_m8 = (__mmask8)_bzhi_u32(0xFF, columns - column);
            _mm512_mask_storeu_pd(
                output + column, mask_m8,
                nk_vincenty_f64x8_skylake_(a_longitude_f64x8, a_sine_f64x8, a_cosine_f64x8,
                                           _mm512_maskz_loadu_pd(mask_m8, b_packed_longitudes + column),
                                           _mm512_maskz_loadu_pd(mask_m8, b_packed_sines + column),
                                           _mm512_maskz_loadu_pd(mask_m8, b_packed_cosines + column)));
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
#endif // NUMKONG_GEOSPATIAL_SKYLAKE_H
