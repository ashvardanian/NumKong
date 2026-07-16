/**
 *  @brief SIMD-accelerated Geospatial Distances for Skylake.
 *  @file include/numkong/geospatial/skylake.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/geospatial.h
 *
 *  @section geospatial_skylake_instructions Key AVX-512 Geospatial Instructions
 *
 *      Intrinsic           Instruction                  Icelake           Genoa
 *      _mm512_sqrt_ps      VSQRTPS (ZMM, ZMM)           19cy @ p0+p0+p05  15cy @ p01
 *      _mm512_sqrt_pd      VSQRTPD (ZMM, ZMM)           23cy @ p0+p0+p05  21cy @ p01
 *      _mm256_div_ps       VDIVPS (YMM, YMM, YMM)       11cy @ p0         11cy @ p01
 *      _mm256_div_pd       VDIVPD (YMM, YMM, YMM)       13cy @ p0         13cy @ p01
 *      _mm256_fmadd_ps     VFMADD231PS (YMM, YMM, YMM)  4cy @ p01         4cy @ p01
 *      _mm256_fmadd_pd     VFMADD231PD (YMM, YMM, YMM)  4cy @ p01         4cy @ p01
 *      _mm512_cmp_ps_mask  VCMPPS (K, ZMM, ZMM, I8)     4cy @ p5          5cy @ p01
 */
#ifndef NK_GEOSPATIAL_SKYLAKE_H
#define NK_GEOSPATIAL_SKYLAKE_H

#if NK_TARGET_X8664_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
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

NK_HELPER_INLINE __m512d nk_haversine_f64x8_skylake_(              //
    __m512d first_latitudes_f64x8, __m512d first_longitudes_f64x8, //
    __m512d second_latitudes_f64x8, __m512d second_longitudes_f64x8) {

    __m512d const earth_radius_f64x8 = _mm512_set1_pd(NK_EARTH_MEDIATORIAL_RADIUS);
    __m512d const half_f64x8 = _mm512_set1_pd(0.5);
    __m512d const one_f64x8 = _mm512_set1_pd(1.0);
    __m512d const two_f64x8 = _mm512_set1_pd(2.0);

    __m512d latitude_delta_f64x8 = _mm512_sub_pd(second_latitudes_f64x8, first_latitudes_f64x8);
    __m512d longitude_delta_f64x8 = _mm512_sub_pd(second_longitudes_f64x8, first_longitudes_f64x8);

    // Haversine terms: sin²(Δ/2)
    __m512d latitude_delta_half_f64x8 = _mm512_mul_pd(latitude_delta_f64x8, half_f64x8);
    __m512d longitude_delta_half_f64x8 = _mm512_mul_pd(longitude_delta_f64x8, half_f64x8);
    __m512d sin_latitude_delta_half_f64x8 = nk_sin_f64x8_skylake_(latitude_delta_half_f64x8);
    __m512d sin_longitude_delta_half_f64x8 = nk_sin_f64x8_skylake_(longitude_delta_half_f64x8);
    __m512d sin_sq_latitude_delta_half_f64x8 = _mm512_mul_pd(sin_latitude_delta_half_f64x8,
                                                             sin_latitude_delta_half_f64x8);
    __m512d sin_sq_longitude_delta_half_f64x8 = _mm512_mul_pd(sin_longitude_delta_half_f64x8,
                                                              sin_longitude_delta_half_f64x8);

    // Latitude cosine product
    __m512d cos_first_latitude_f64x8 = nk_cos_f64x8_skylake_(first_latitudes_f64x8);
    __m512d cos_second_latitude_f64x8 = nk_cos_f64x8_skylake_(second_latitudes_f64x8);
    __m512d cos_latitude_product_f64x8 = _mm512_mul_pd(cos_first_latitude_f64x8, cos_second_latitude_f64x8);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    __m512d haversine_term_f64x8 = _mm512_add_pd(
        sin_sq_latitude_delta_half_f64x8, _mm512_mul_pd(cos_latitude_product_f64x8, sin_sq_longitude_delta_half_f64x8));
    // Clamp haversine_term_f64x8 to [0, 1] to prevent NaN from sqrt of negative values
    __m512d zero_f64x8 = _mm512_setzero_pd();
    haversine_term_f64x8 = _mm512_max_pd(zero_f64x8, _mm512_min_pd(one_f64x8, haversine_term_f64x8));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    __m512d sqrt_haversine_f64x8 = _mm512_sqrt_pd(haversine_term_f64x8);
    __m512d sqrt_complement_f64x8 = _mm512_sqrt_pd(_mm512_sub_pd(one_f64x8, haversine_term_f64x8));
    __m512d central_angle_f64x8 = _mm512_mul_pd(two_f64x8,
                                                nk_atan2_f64x8_skylake_(sqrt_haversine_f64x8, sqrt_complement_f64x8));

    return _mm512_mul_pd(earth_radius_f64x8, central_angle_f64x8);
}

NK_API_COMPTIME void nk_haversine_f64_skylake(      //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 8) {
        __m512d first_latitudes_f64x8 = _mm512_loadu_pd(a_lats);
        __m512d first_longitudes_f64x8 = _mm512_loadu_pd(a_lons);
        __m512d second_latitudes_f64x8 = _mm512_loadu_pd(b_lats);
        __m512d second_longitudes_f64x8 = _mm512_loadu_pd(b_lons);

        __m512d distances_f64x8 = nk_haversine_f64x8_skylake_(first_latitudes_f64x8, first_longitudes_f64x8,
                                                              second_latitudes_f64x8, second_longitudes_f64x8);
        _mm512_storeu_pd(results, distances_f64x8);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with masked operations
    if (n > 0) {
        __mmask8 mask_m8 = (__mmask8)_bzhi_u32(0xFF, n);
        __m512d first_latitudes_f64x8 = _mm512_maskz_loadu_pd(mask_m8, a_lats);
        __m512d first_longitudes_f64x8 = _mm512_maskz_loadu_pd(mask_m8, a_lons);
        __m512d second_latitudes_f64x8 = _mm512_maskz_loadu_pd(mask_m8, b_lats);
        __m512d second_longitudes_f64x8 = _mm512_maskz_loadu_pd(mask_m8, b_lons);

        __m512d distances_f64x8 = nk_haversine_f64x8_skylake_(first_latitudes_f64x8, first_longitudes_f64x8,
                                                              second_latitudes_f64x8, second_longitudes_f64x8);
        _mm512_mask_storeu_pd(results, mask_m8, distances_f64x8);
    }
}

/**
 *  @brief  AVX-512 helper for Vincenty's geodesic distance on 8 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking.
 */
NK_HELPER_INLINE __m512d nk_vincenty_f64x8_skylake_(               //
    __m512d first_latitudes_f64x8, __m512d first_longitudes_f64x8, //
    __m512d second_latitudes_f64x8, __m512d second_longitudes_f64x8) {

    __m512d const equatorial_radius_f64x8 = _mm512_set1_pd(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m512d const polar_radius_f64x8 = _mm512_set1_pd(NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m512d const flattening_f64x8 = _mm512_set1_pd(1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m512d const convergence_threshold_f64x8 = _mm512_set1_pd(NK_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    __m512d const one_f64x8 = _mm512_set1_pd(1.0);
    __m512d const two_f64x8 = _mm512_set1_pd(2.0);
    __m512d const three_f64x8 = _mm512_set1_pd(3.0);
    __m512d const four_f64x8 = _mm512_set1_pd(4.0);
    __m512d const six_f64x8 = _mm512_set1_pd(6.0);
    __m512d const sixteen_f64x8 = _mm512_set1_pd(16.0);

    // Longitude difference
    __m512d longitude_diff_f64x8 = _mm512_sub_pd(second_longitudes_f64x8, first_longitudes_f64x8);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    __m512d one_minus_f64x8 = _mm512_sub_pd(one_f64x8, flattening_f64x8);
    __m512d tan_first_f64x8 = _mm512_div_pd(nk_sin_f64x8_skylake_(first_latitudes_f64x8),
                                            nk_cos_f64x8_skylake_(first_latitudes_f64x8));
    __m512d tan_second_f64x8 = _mm512_div_pd(nk_sin_f64x8_skylake_(second_latitudes_f64x8),
                                             nk_cos_f64x8_skylake_(second_latitudes_f64x8));
    __m512d tan_reduced_first_f64x8 = _mm512_mul_pd(one_minus_f64x8, tan_first_f64x8);
    __m512d tan_reduced_second_f64x8 = _mm512_mul_pd(one_minus_f64x8, tan_second_f64x8);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    __m512d cos_reduced_first_f64x8 = _mm512_div_pd(
        one_f64x8, _mm512_sqrt_pd(_mm512_fmadd_pd(tan_reduced_first_f64x8, tan_reduced_first_f64x8, one_f64x8)));
    __m512d sin_reduced_first_f64x8 = _mm512_mul_pd(tan_reduced_first_f64x8, cos_reduced_first_f64x8);
    __m512d cos_reduced_second_f64x8 = _mm512_div_pd(
        one_f64x8, _mm512_sqrt_pd(_mm512_fmadd_pd(tan_reduced_second_f64x8, tan_reduced_second_f64x8, one_f64x8)));
    __m512d sin_reduced_second_f64x8 = _mm512_mul_pd(tan_reduced_second_f64x8, cos_reduced_second_f64x8);

    // Initialize lambda_f64x8 and tracking variables
    __m512d lambda_f64x8 = longitude_diff_f64x8;
    __m512d sin_angular_distance_f64x8, cos_angular_distance_f64x8, angular_distance_f64x8;
    __m512d sin_azimuth_f64x8, cos_sq_azimuth_f64x8, cos_double_angular_midpoint_f64x8;

    // Track convergence and coincident points
    __mmask8 converged_m8 = 0;
    __mmask8 coincident_m8 = 0;

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS && converged_m8 != 0xFF; ++iteration) {
        __m512d sin_lambda_f64x8 = nk_sin_f64x8_skylake_(lambda_f64x8);
        __m512d cos_lambda_f64x8 = nk_cos_f64x8_skylake_(lambda_f64x8);

        // sin²(angular_distance_f64x8) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        __m512d cross_term_f64x8 = _mm512_mul_pd(cos_reduced_second_f64x8, sin_lambda_f64x8);
        __m512d mixed_term_f64x8 = _mm512_sub_pd(
            _mm512_mul_pd(cos_reduced_first_f64x8, sin_reduced_second_f64x8),
            _mm512_mul_pd(_mm512_mul_pd(sin_reduced_first_f64x8, cos_reduced_second_f64x8), cos_lambda_f64x8));
        __m512d sin_angular_dist_sq_f64x8 = _mm512_fmadd_pd(cross_term_f64x8, cross_term_f64x8,
                                                            _mm512_mul_pd(mixed_term_f64x8, mixed_term_f64x8));
        sin_angular_distance_f64x8 = _mm512_sqrt_pd(sin_angular_dist_sq_f64x8);

        // Check for coincident points (sin_angular_distance_f64x8 ≈ 0)
        coincident_m8 = _mm512_cmp_pd_mask(sin_angular_distance_f64x8, _mm512_set1_pd(1e-15), _CMP_LT_OS);

        // cos(angular_distance_f64x8) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance_f64x8 = _mm512_fmadd_pd(_mm512_mul_pd(cos_reduced_first_f64x8, cos_reduced_second_f64x8),
                                                     cos_lambda_f64x8,
                                                     _mm512_mul_pd(sin_reduced_first_f64x8, sin_reduced_second_f64x8));

        // angular_distance_f64x8 = atan2(sin, cos)
        angular_distance_f64x8 = nk_atan2_f64x8_skylake_(sin_angular_distance_f64x8, cos_angular_distance_f64x8);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance_f64x8)
        // Use masked divide: zero result for coincident lanes, avoids division by zero
        sin_azimuth_f64x8 = _mm512_maskz_div_pd(
            _knot_mask8(coincident_m8),
            _mm512_mul_pd(_mm512_mul_pd(cos_reduced_first_f64x8, cos_reduced_second_f64x8), sin_lambda_f64x8),
            sin_angular_distance_f64x8);
        cos_sq_azimuth_f64x8 = _mm512_sub_pd(one_f64x8, _mm512_mul_pd(sin_azimuth_f64x8, sin_azimuth_f64x8));

        // Handle equatorial case: cos²α = 0
        __mmask8 equatorial_m8 = _mm512_cmp_pd_mask(cos_sq_azimuth_f64x8, _mm512_set1_pd(1e-15), _CMP_LT_OS);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        // Use masked divide: for equatorial lanes, quotient_f64x8 = cos_angular_distance_f64x8 (passthrough),
        // so subtraction yields zero. Avoids division by zero.
        __m512d sin_product_f64x8 = _mm512_mul_pd(sin_reduced_first_f64x8, sin_reduced_second_f64x8);
        __m512d quotient_f64x8 = _mm512_mask_div_pd(cos_angular_distance_f64x8, _knot_mask8(equatorial_m8),
                                                    _mm512_mul_pd(two_f64x8, sin_product_f64x8), cos_sq_azimuth_f64x8);
        cos_double_angular_midpoint_f64x8 = _mm512_sub_pd(cos_angular_distance_f64x8, quotient_f64x8);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        __m512d correction_factor_f64x8 = _mm512_mul_pd(
            _mm512_div_pd(flattening_f64x8, sixteen_f64x8),
            _mm512_mul_pd(
                cos_sq_azimuth_f64x8,
                _mm512_fmadd_pd(flattening_f64x8, _mm512_fnmadd_pd(three_f64x8, cos_sq_azimuth_f64x8, four_f64x8),
                                four_f64x8)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        __m512d cos_2sm_sq_f64x8 = _mm512_mul_pd(cos_double_angular_midpoint_f64x8, cos_double_angular_midpoint_f64x8);
        // innermost_f64x8 = -1 + 2 × cos²(2σₘ)
        __m512d innermost_f64x8 = _mm512_fmadd_pd(two_f64x8, cos_2sm_sq_f64x8, _mm512_set1_pd(-1.0));
        // middle_f64x8 = cos(2σₘ) + C × cos(σ) × innermost_f64x8
        __m512d middle_f64x8 = _mm512_fmadd_pd(_mm512_mul_pd(correction_factor_f64x8, cos_angular_distance_f64x8),
                                               innermost_f64x8, cos_double_angular_midpoint_f64x8);
        // inner_f64x8 = C × sin(σ) × middle_f64x8
        __m512d inner_f64x8 = _mm512_mul_pd(_mm512_mul_pd(correction_factor_f64x8, sin_angular_distance_f64x8),
                                            middle_f64x8);

        // λ' = L + (1-C) * f * sin_α * (σ + inner_f64x8)
        __m512d lambda_new_f64x8 = _mm512_fmadd_pd(
            _mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(one_f64x8, correction_factor_f64x8), flattening_f64x8),
                          sin_azimuth_f64x8),
            _mm512_add_pd(angular_distance_f64x8, inner_f64x8), longitude_diff_f64x8);

        // Check convergence: |λ - λ'| < threshold
        __m512d lambda_diff_f64x8 = _mm512_abs_pd(_mm512_sub_pd(lambda_new_f64x8, lambda_f64x8));
        converged_m8 = _mm512_cmp_pd_mask(lambda_diff_f64x8, convergence_threshold_f64x8, _CMP_LT_OS);

        lambda_f64x8 = lambda_new_f64x8;
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    __m512d a_sq_f64x8 = _mm512_mul_pd(equatorial_radius_f64x8, equatorial_radius_f64x8);
    __m512d b_sq_f64x8 = _mm512_mul_pd(polar_radius_f64x8, polar_radius_f64x8);
    __m512d u_sq_f64x8 = _mm512_div_pd(_mm512_mul_pd(cos_sq_azimuth_f64x8, _mm512_sub_pd(a_sq_f64x8, b_sq_f64x8)),
                                       b_sq_f64x8);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    __m512d series_a_f64x8 = _mm512_fmadd_pd(u_sq_f64x8, _mm512_set1_pd(-175.0), _mm512_set1_pd(320.0));
    series_a_f64x8 = _mm512_fmadd_pd(u_sq_f64x8, series_a_f64x8, _mm512_set1_pd(-768.0));
    series_a_f64x8 = _mm512_fmadd_pd(u_sq_f64x8, series_a_f64x8, _mm512_set1_pd(4096.0));
    series_a_f64x8 = _mm512_fmadd_pd(_mm512_div_pd(u_sq_f64x8, _mm512_set1_pd(16384.0)), series_a_f64x8, one_f64x8);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    __m512d series_b_f64x8 = _mm512_fmadd_pd(u_sq_f64x8, _mm512_set1_pd(-47.0), _mm512_set1_pd(74.0));
    series_b_f64x8 = _mm512_fmadd_pd(u_sq_f64x8, series_b_f64x8, _mm512_set1_pd(-128.0));
    series_b_f64x8 = _mm512_fmadd_pd(u_sq_f64x8, series_b_f64x8, _mm512_set1_pd(256.0));
    series_b_f64x8 = _mm512_mul_pd(_mm512_div_pd(u_sq_f64x8, _mm512_set1_pd(1024.0)), series_b_f64x8);

    // Δσ = B × sin(σ) × (cos(2σₘ) +
    //      B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m512d cos_2sm_sq_f64x8 = _mm512_mul_pd(cos_double_angular_midpoint_f64x8, cos_double_angular_midpoint_f64x8);
    __m512d sin_sq_f64x8 = _mm512_mul_pd(sin_angular_distance_f64x8, sin_angular_distance_f64x8);
    __m512d term1_f64x8 = _mm512_fmadd_pd(two_f64x8, cos_2sm_sq_f64x8, _mm512_set1_pd(-1.0));
    term1_f64x8 = _mm512_mul_pd(cos_angular_distance_f64x8, term1_f64x8);
    __m512d term2_f64x8 = _mm512_fmadd_pd(four_f64x8, sin_sq_f64x8, _mm512_set1_pd(-3.0));
    __m512d term3_f64x8 = _mm512_fmadd_pd(four_f64x8, cos_2sm_sq_f64x8, _mm512_set1_pd(-3.0));
    term2_f64x8 = _mm512_mul_pd(
        _mm512_mul_pd(_mm512_div_pd(series_b_f64x8, six_f64x8), cos_double_angular_midpoint_f64x8),
        _mm512_mul_pd(term2_f64x8, term3_f64x8));
    __m512d delta_sigma_f64x8 = _mm512_mul_pd(
        series_b_f64x8, _mm512_mul_pd(sin_angular_distance_f64x8,
                                      _mm512_add_pd(cos_double_angular_midpoint_f64x8,
                                                    _mm512_mul_pd(_mm512_div_pd(series_b_f64x8, four_f64x8),
                                                                  _mm512_sub_pd(term1_f64x8, term2_f64x8)))));

    // s = b * A * (σ - Δσ)
    __m512d distances_f64x8 = _mm512_mul_pd(_mm512_mul_pd(polar_radius_f64x8, series_a_f64x8),
                                            _mm512_sub_pd(angular_distance_f64x8, delta_sigma_f64x8));

    // Set coincident points to zero
    distances_f64x8 = _mm512_mask_blend_pd(coincident_m8, distances_f64x8, _mm512_setzero_pd());

    return distances_f64x8;
}

NK_API_COMPTIME void nk_vincenty_f64_skylake(       //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 8) {
        __m512d first_latitudes_f64x8 = _mm512_loadu_pd(a_lats);
        __m512d first_longitudes_f64x8 = _mm512_loadu_pd(a_lons);
        __m512d second_latitudes_f64x8 = _mm512_loadu_pd(b_lats);
        __m512d second_longitudes_f64x8 = _mm512_loadu_pd(b_lons);

        __m512d distances_f64x8 = nk_vincenty_f64x8_skylake_(first_latitudes_f64x8, first_longitudes_f64x8,
                                                             second_latitudes_f64x8, second_longitudes_f64x8);
        _mm512_storeu_pd(results, distances_f64x8);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with masked operations
    if (n > 0) {
        __mmask8 mask_m8 = (__mmask8)_bzhi_u32(0xFF, n);
        __m512d first_latitudes_f64x8 = _mm512_maskz_loadu_pd(mask_m8, a_lats);
        __m512d first_longitudes_f64x8 = _mm512_maskz_loadu_pd(mask_m8, a_lons);
        __m512d second_latitudes_f64x8 = _mm512_maskz_loadu_pd(mask_m8, b_lats);
        __m512d second_longitudes_f64x8 = _mm512_maskz_loadu_pd(mask_m8, b_lons);

        __m512d distances_f64x8 = nk_vincenty_f64x8_skylake_(first_latitudes_f64x8, first_longitudes_f64x8,
                                                             second_latitudes_f64x8, second_longitudes_f64x8);
        _mm512_mask_storeu_pd(results, mask_m8, distances_f64x8);
    }
}

NK_HELPER_INLINE __m512 nk_haversine_f32x16_skylake_(              //
    __m512 first_latitudes_f32x16, __m512 first_longitudes_f32x16, //
    __m512 second_latitudes_f32x16, __m512 second_longitudes_f32x16) {

    __m512 const earth_radius_f32x16 = _mm512_set1_ps((float)NK_EARTH_MEDIATORIAL_RADIUS);
    __m512 const half_f32x16 = _mm512_set1_ps(0.5f);
    __m512 const one_f32x16 = _mm512_set1_ps(1.0f);
    __m512 const two_f32x16 = _mm512_set1_ps(2.0f);

    __m512 latitude_delta_f32x16 = _mm512_sub_ps(second_latitudes_f32x16, first_latitudes_f32x16);
    __m512 longitude_delta_f32x16 = _mm512_sub_ps(second_longitudes_f32x16, first_longitudes_f32x16);

    // Haversine terms: sin²(Δ/2)
    __m512 latitude_delta_half_f32x16 = _mm512_mul_ps(latitude_delta_f32x16, half_f32x16);
    __m512 longitude_delta_half_f32x16 = _mm512_mul_ps(longitude_delta_f32x16, half_f32x16);
    __m512 sin_latitude_delta_half_f32x16 = nk_sin_f32x16_skylake_(latitude_delta_half_f32x16);
    __m512 sin_longitude_delta_half_f32x16 = nk_sin_f32x16_skylake_(longitude_delta_half_f32x16);
    __m512 sin_sq_latitude_delta_half_f32x16 = _mm512_mul_ps(sin_latitude_delta_half_f32x16,
                                                             sin_latitude_delta_half_f32x16);
    __m512 sin_sq_longitude_delta_half_f32x16 = _mm512_mul_ps(sin_longitude_delta_half_f32x16,
                                                              sin_longitude_delta_half_f32x16);

    // Latitude cosine product
    __m512 cos_first_latitude_f32x16 = nk_cos_f32x16_skylake_(first_latitudes_f32x16);
    __m512 cos_second_latitude_f32x16 = nk_cos_f32x16_skylake_(second_latitudes_f32x16);
    __m512 cos_latitude_product_f32x16 = _mm512_mul_ps(cos_first_latitude_f32x16, cos_second_latitude_f32x16);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    __m512 haversine_term_f32x16 = _mm512_add_ps(
        sin_sq_latitude_delta_half_f32x16,
        _mm512_mul_ps(cos_latitude_product_f32x16, sin_sq_longitude_delta_half_f32x16));

    // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
    __m512 zero_f32x16 = _mm512_setzero_ps();
    haversine_term_f32x16 = _mm512_max_ps(zero_f32x16, _mm512_min_ps(one_f32x16, haversine_term_f32x16));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    __m512 sqrt_haversine_f32x16 = _mm512_sqrt_ps(haversine_term_f32x16);
    __m512 sqrt_complement_f32x16 = _mm512_sqrt_ps(_mm512_sub_ps(one_f32x16, haversine_term_f32x16));
    __m512 central_angle_f32x16 = _mm512_mul_ps(
        two_f32x16, nk_atan2_f32x16_skylake_(sqrt_haversine_f32x16, sqrt_complement_f32x16));

    return _mm512_mul_ps(earth_radius_f32x16, central_angle_f32x16);
}

NK_API_COMPTIME void nk_haversine_f32_skylake(      //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 16) {
        __m512 first_latitudes_f32x16 = _mm512_loadu_ps(a_lats);
        __m512 first_longitudes_f32x16 = _mm512_loadu_ps(a_lons);
        __m512 second_latitudes_f32x16 = _mm512_loadu_ps(b_lats);
        __m512 second_longitudes_f32x16 = _mm512_loadu_ps(b_lons);

        __m512 distances_f32x16 = nk_haversine_f32x16_skylake_(first_latitudes_f32x16, first_longitudes_f32x16,
                                                               second_latitudes_f32x16, second_longitudes_f32x16);
        _mm512_storeu_ps(results, distances_f32x16);

        a_lats += 16, a_lons += 16, b_lats += 16, b_lons += 16, results += 16, n -= 16;
    }

    // Handle remaining elements with masked operations
    if (n > 0) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n);
        __m512 first_latitudes_f32x16 = _mm512_maskz_loadu_ps(mask_m16, a_lats);
        __m512 first_longitudes_f32x16 = _mm512_maskz_loadu_ps(mask_m16, a_lons);
        __m512 second_latitudes_f32x16 = _mm512_maskz_loadu_ps(mask_m16, b_lats);
        __m512 second_longitudes_f32x16 = _mm512_maskz_loadu_ps(mask_m16, b_lons);

        __m512 distances_f32x16 = nk_haversine_f32x16_skylake_(first_latitudes_f32x16, first_longitudes_f32x16,
                                                               second_latitudes_f32x16, second_longitudes_f32x16);
        _mm512_mask_storeu_ps(results, mask_m16, distances_f32x16);
    }
}

/**
 *  @brief  AVX-512 helper for Vincenty's geodesic distance on 16 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking.
 */
NK_HELPER_INLINE __m512 nk_vincenty_f32x16_skylake_(               //
    __m512 first_latitudes_f32x16, __m512 first_longitudes_f32x16, //
    __m512 second_latitudes_f32x16, __m512 second_longitudes_f32x16) {

    __m512 const equatorial_radius_f32x16 = _mm512_set1_ps((float)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m512 const polar_radius_f32x16 = _mm512_set1_ps((float)NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m512 const flattening_f32x16 = _mm512_set1_ps(1.0f / (float)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m512 const convergence_threshold_f32x16 = _mm512_set1_ps(NK_VINCENTY_CONVERGENCE_THRESHOLD_F32);
    __m512 const one_f32x16 = _mm512_set1_ps(1.0f);
    __m512 const two_f32x16 = _mm512_set1_ps(2.0f);
    __m512 const three_f32x16 = _mm512_set1_ps(3.0f);
    __m512 const four_f32x16 = _mm512_set1_ps(4.0f);
    __m512 const six_f32x16 = _mm512_set1_ps(6.0f);
    __m512 const sixteen_f32x16 = _mm512_set1_ps(16.0f);

    // Longitude difference
    __m512 longitude_diff_f32x16 = _mm512_sub_ps(second_longitudes_f32x16, first_longitudes_f32x16);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    __m512 one_minus_f32x16 = _mm512_sub_ps(one_f32x16, flattening_f32x16);
    __m512 tan_first_f32x16 = _mm512_div_ps(nk_sin_f32x16_skylake_(first_latitudes_f32x16),
                                            nk_cos_f32x16_skylake_(first_latitudes_f32x16));
    __m512 tan_second_f32x16 = _mm512_div_ps(nk_sin_f32x16_skylake_(second_latitudes_f32x16),
                                             nk_cos_f32x16_skylake_(second_latitudes_f32x16));
    __m512 tan_reduced_first_f32x16 = _mm512_mul_ps(one_minus_f32x16, tan_first_f32x16);
    __m512 tan_reduced_second_f32x16 = _mm512_mul_ps(one_minus_f32x16, tan_second_f32x16);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    __m512 cos_reduced_first_f32x16 = _mm512_div_ps(
        one_f32x16, _mm512_sqrt_ps(_mm512_fmadd_ps(tan_reduced_first_f32x16, tan_reduced_first_f32x16, one_f32x16)));
    __m512 sin_reduced_first_f32x16 = _mm512_mul_ps(tan_reduced_first_f32x16, cos_reduced_first_f32x16);
    __m512 cos_reduced_second_f32x16 = _mm512_div_ps(
        one_f32x16, _mm512_sqrt_ps(_mm512_fmadd_ps(tan_reduced_second_f32x16, tan_reduced_second_f32x16, one_f32x16)));
    __m512 sin_reduced_second_f32x16 = _mm512_mul_ps(tan_reduced_second_f32x16, cos_reduced_second_f32x16);

    // Initialize lambda_f32x16 and tracking variables
    __m512 lambda_f32x16 = longitude_diff_f32x16;
    __m512 sin_angular_distance_f32x16, cos_angular_distance_f32x16, angular_distance_f32x16;
    __m512 sin_azimuth_f32x16, cos_sq_azimuth_f32x16, cos_double_angular_midpoint_f32x16;

    // Track convergence and coincident points
    __mmask16 converged_m16 = 0;
    __mmask16 coincident_m16 = 0;

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS && converged_m16 != 0xFFFF; ++iteration) {
        __m512 sin_lambda_f32x16 = nk_sin_f32x16_skylake_(lambda_f32x16);
        __m512 cos_lambda_f32x16 = nk_cos_f32x16_skylake_(lambda_f32x16);

        // sin²(angular_distance_f32x16) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        __m512 cross_term_f32x16 = _mm512_mul_ps(cos_reduced_second_f32x16, sin_lambda_f32x16);
        __m512 mixed_term_f32x16 = _mm512_sub_ps(
            _mm512_mul_ps(cos_reduced_first_f32x16, sin_reduced_second_f32x16),
            _mm512_mul_ps(_mm512_mul_ps(sin_reduced_first_f32x16, cos_reduced_second_f32x16), cos_lambda_f32x16));
        __m512 sin_angular_dist_sq_f32x16 = _mm512_fmadd_ps(cross_term_f32x16, cross_term_f32x16,
                                                            _mm512_mul_ps(mixed_term_f32x16, mixed_term_f32x16));
        sin_angular_distance_f32x16 = _mm512_sqrt_ps(sin_angular_dist_sq_f32x16);

        // Check for coincident points (sin_angular_distance_f32x16 ≈ 0)
        coincident_m16 = _mm512_cmp_ps_mask(sin_angular_distance_f32x16, _mm512_set1_ps(1e-7f), _CMP_LT_OS);

        // cos(angular_distance_f32x16) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance_f32x16 = _mm512_fmadd_ps(
            _mm512_mul_ps(cos_reduced_first_f32x16, cos_reduced_second_f32x16), cos_lambda_f32x16,
            _mm512_mul_ps(sin_reduced_first_f32x16, sin_reduced_second_f32x16));

        // angular_distance_f32x16 = atan2(sin, cos)
        angular_distance_f32x16 = nk_atan2_f32x16_skylake_(sin_angular_distance_f32x16, cos_angular_distance_f32x16);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance_f32x16)
        // Use masked divide: zero result for coincident lanes, avoids division by zero
        sin_azimuth_f32x16 = _mm512_maskz_div_ps(
            _knot_mask16(coincident_m16),
            _mm512_mul_ps(_mm512_mul_ps(cos_reduced_first_f32x16, cos_reduced_second_f32x16), sin_lambda_f32x16),
            sin_angular_distance_f32x16);
        cos_sq_azimuth_f32x16 = _mm512_sub_ps(one_f32x16, _mm512_mul_ps(sin_azimuth_f32x16, sin_azimuth_f32x16));

        // Handle equatorial case: cos²α = 0
        __mmask16 equatorial_m16 = _mm512_cmp_ps_mask(cos_sq_azimuth_f32x16, _mm512_set1_ps(1e-7f), _CMP_LT_OS);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        // Use masked divide: for equatorial lanes, quotient_f32x16 = cos_angular_distance_f32x16 (passthrough),
        // so subtraction yields zero. Avoids division by zero.
        __m512 sin_product_f32x16 = _mm512_mul_ps(sin_reduced_first_f32x16, sin_reduced_second_f32x16);
        __m512 quotient_f32x16 = _mm512_mask_div_ps(cos_angular_distance_f32x16, _knot_mask16(equatorial_m16),
                                                    _mm512_mul_ps(two_f32x16, sin_product_f32x16),
                                                    cos_sq_azimuth_f32x16);
        cos_double_angular_midpoint_f32x16 = _mm512_sub_ps(cos_angular_distance_f32x16, quotient_f32x16);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        __m512 correction_factor_f32x16 = _mm512_mul_ps(
            _mm512_div_ps(flattening_f32x16, sixteen_f32x16),
            _mm512_mul_ps(
                cos_sq_azimuth_f32x16,
                _mm512_fmadd_ps(flattening_f32x16, _mm512_fnmadd_ps(three_f32x16, cos_sq_azimuth_f32x16, four_f32x16),
                                four_f32x16)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        __m512 cos_2sm_sq_f32x16 = _mm512_mul_ps(cos_double_angular_midpoint_f32x16,
                                                 cos_double_angular_midpoint_f32x16);
        // innermost_f32x16 = -1 + 2 × cos²(2σₘ)
        __m512 innermost_f32x16 = _mm512_fmadd_ps(two_f32x16, cos_2sm_sq_f32x16, _mm512_set1_ps(-1.0f));
        // middle_f32x16 = cos(2σₘ) + C × cos(σ) × innermost_f32x16
        __m512 middle_f32x16 = _mm512_fmadd_ps(_mm512_mul_ps(correction_factor_f32x16, cos_angular_distance_f32x16),
                                               innermost_f32x16, cos_double_angular_midpoint_f32x16);
        // inner_f32x16 = C × sin(σ) × middle_f32x16
        __m512 inner_f32x16 = _mm512_mul_ps(_mm512_mul_ps(correction_factor_f32x16, sin_angular_distance_f32x16),
                                            middle_f32x16);

        // λ' = L + (1-C) * f * sin_α * (σ + inner_f32x16)
        __m512 lambda_new_f32x16 = _mm512_fmadd_ps(
            _mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(one_f32x16, correction_factor_f32x16), flattening_f32x16),
                          sin_azimuth_f32x16),
            _mm512_add_ps(angular_distance_f32x16, inner_f32x16), longitude_diff_f32x16);

        // Check convergence: |λ - λ'| < threshold
        __m512 lambda_diff_f32x16 = _mm512_abs_ps(_mm512_sub_ps(lambda_new_f32x16, lambda_f32x16));
        converged_m16 = _mm512_cmp_ps_mask(lambda_diff_f32x16, convergence_threshold_f32x16, _CMP_LT_OS);

        lambda_f32x16 = lambda_new_f32x16;
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    __m512 a_sq_f32x16 = _mm512_mul_ps(equatorial_radius_f32x16, equatorial_radius_f32x16);
    __m512 b_sq_f32x16 = _mm512_mul_ps(polar_radius_f32x16, polar_radius_f32x16);
    __m512 u_sq_f32x16 = _mm512_div_ps(_mm512_mul_ps(cos_sq_azimuth_f32x16, _mm512_sub_ps(a_sq_f32x16, b_sq_f32x16)),
                                       b_sq_f32x16);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    __m512 series_a_f32x16 = _mm512_fmadd_ps(u_sq_f32x16, _mm512_set1_ps(-175.0f), _mm512_set1_ps(320.0f));
    series_a_f32x16 = _mm512_fmadd_ps(u_sq_f32x16, series_a_f32x16, _mm512_set1_ps(-768.0f));
    series_a_f32x16 = _mm512_fmadd_ps(u_sq_f32x16, series_a_f32x16, _mm512_set1_ps(4096.0f));
    series_a_f32x16 = _mm512_fmadd_ps(_mm512_div_ps(u_sq_f32x16, _mm512_set1_ps(16384.0f)), series_a_f32x16,
                                      one_f32x16);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    __m512 series_b_f32x16 = _mm512_fmadd_ps(u_sq_f32x16, _mm512_set1_ps(-47.0f), _mm512_set1_ps(74.0f));
    series_b_f32x16 = _mm512_fmadd_ps(u_sq_f32x16, series_b_f32x16, _mm512_set1_ps(-128.0f));
    series_b_f32x16 = _mm512_fmadd_ps(u_sq_f32x16, series_b_f32x16, _mm512_set1_ps(256.0f));
    series_b_f32x16 = _mm512_mul_ps(_mm512_div_ps(u_sq_f32x16, _mm512_set1_ps(1024.0f)), series_b_f32x16);

    // Δσ = B × sin(σ) × (cos(2σₘ) +
    //      B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m512 cos_2sm_sq_f32x16 = _mm512_mul_ps(cos_double_angular_midpoint_f32x16, cos_double_angular_midpoint_f32x16);
    __m512 sin_sq_f32x16 = _mm512_mul_ps(sin_angular_distance_f32x16, sin_angular_distance_f32x16);
    __m512 term1_f32x16 = _mm512_fmadd_ps(two_f32x16, cos_2sm_sq_f32x16, _mm512_set1_ps(-1.0f));
    term1_f32x16 = _mm512_mul_ps(cos_angular_distance_f32x16, term1_f32x16);
    __m512 term2_f32x16 = _mm512_fmadd_ps(four_f32x16, sin_sq_f32x16, _mm512_set1_ps(-3.0f));
    __m512 term3_f32x16 = _mm512_fmadd_ps(four_f32x16, cos_2sm_sq_f32x16, _mm512_set1_ps(-3.0f));
    term2_f32x16 = _mm512_mul_ps(
        _mm512_mul_ps(_mm512_div_ps(series_b_f32x16, six_f32x16), cos_double_angular_midpoint_f32x16),
        _mm512_mul_ps(term2_f32x16, term3_f32x16));
    __m512 delta_sigma_f32x16 = _mm512_mul_ps(
        series_b_f32x16, _mm512_mul_ps(sin_angular_distance_f32x16,
                                       _mm512_add_ps(cos_double_angular_midpoint_f32x16,
                                                     _mm512_mul_ps(_mm512_div_ps(series_b_f32x16, four_f32x16),
                                                                   _mm512_sub_ps(term1_f32x16, term2_f32x16)))));

    // s = b * A * (σ - Δσ)
    __m512 distances_f32x16 = _mm512_mul_ps(_mm512_mul_ps(polar_radius_f32x16, series_a_f32x16),
                                            _mm512_sub_ps(angular_distance_f32x16, delta_sigma_f32x16));

    // Set coincident points to zero
    distances_f32x16 = _mm512_mask_blend_ps(coincident_m16, distances_f32x16, _mm512_setzero_ps());

    return distances_f32x16;
}

NK_API_COMPTIME void nk_vincenty_f32_skylake(       //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 16) {
        __m512 first_latitudes_f32x16 = _mm512_loadu_ps(a_lats);
        __m512 first_longitudes_f32x16 = _mm512_loadu_ps(a_lons);
        __m512 second_latitudes_f32x16 = _mm512_loadu_ps(b_lats);
        __m512 second_longitudes_f32x16 = _mm512_loadu_ps(b_lons);

        __m512 distances_f32x16 = nk_vincenty_f32x16_skylake_(first_latitudes_f32x16, first_longitudes_f32x16,
                                                              second_latitudes_f32x16, second_longitudes_f32x16);
        _mm512_storeu_ps(results, distances_f32x16);

        a_lats += 16, a_lons += 16, b_lats += 16, b_lons += 16, results += 16, n -= 16;
    }

    // Handle remaining elements with masked operations
    if (n > 0) {
        __mmask16 mask_m16 = (__mmask16)_bzhi_u32(0xFFFF, n);
        __m512 first_latitudes_f32x16 = _mm512_maskz_loadu_ps(mask_m16, a_lats);
        __m512 first_longitudes_f32x16 = _mm512_maskz_loadu_ps(mask_m16, a_lons);
        __m512 second_latitudes_f32x16 = _mm512_maskz_loadu_ps(mask_m16, b_lats);
        __m512 second_longitudes_f32x16 = _mm512_maskz_loadu_ps(mask_m16, b_lons);

        __m512 distances_f32x16 = nk_vincenty_f32x16_skylake_(first_latitudes_f32x16, first_longitudes_f32x16,
                                                              second_latitudes_f32x16, second_longitudes_f32x16);
        _mm512_mask_storeu_ps(results, mask_m16, distances_f32x16);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_SKYLAKE
#endif // NK_TARGET_X8664_
#endif // NK_GEOSPATIAL_SKYLAKE_H
