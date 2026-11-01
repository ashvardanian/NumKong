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
 *      Intrinsic               Instruction                     Ice         Genoa
 *      _mm512_sqrt_ps          VSQRTPS (ZMM, ZMM)              19c @ p05   15c @ p01
 *      _mm512_sqrt_pd          VSQRTPD (ZMM, ZMM)              23c @ p05   21c @ p01
 *      _mm256_div_ps           VDIVPS (YMM, YMM, YMM)          11c @ p0    11c @ p01
 *      _mm256_div_pd           VDIVPD (YMM, YMM, YMM)          13c @ p0    13c @ p01
 *      _mm256_fmadd_ps         VFMADD231PS (YMM, YMM, YMM)     4c @ p01    4c @ p01
 *      _mm256_fmadd_pd         VFMADD231PD (YMM, YMM, YMM)     4c @ p01    4c @ p01
 */
#ifndef NK_GEOSPATIAL_SKYLAKE_H
#define NK_GEOSPATIAL_SKYLAKE_H

#if NK_TARGET_X86_
#if NK_TARGET_SKYLAKE

#include "numkong/types.h"
#include "numkong/trigonometry/skylake.h" // `nk_sin_f64x8_skylake_`, `nk_cos_f64x8_skylake_`, `nk_atan2_f64x8_skylake_`, etc.

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

NK_INTERNAL __m512d nk_haversine_f64x8_skylake_(       //
    __m512d first_latitudes, __m512d first_longitudes, //
    __m512d second_latitudes, __m512d second_longitudes) {

    __m512d const earth_radius = _mm512_set1_pd(NK_EARTH_MEDIATORIAL_RADIUS);
    __m512d const half = _mm512_set1_pd(0.5);
    __m512d const one = _mm512_set1_pd(1.0);
    __m512d const two = _mm512_set1_pd(2.0);

    __m512d latitude_delta = _mm512_sub_pd(second_latitudes, first_latitudes);
    __m512d longitude_delta = _mm512_sub_pd(second_longitudes, first_longitudes);

    // Haversine terms: sin²(Δ/2)
    __m512d latitude_delta_half = _mm512_mul_pd(latitude_delta, half);
    __m512d longitude_delta_half = _mm512_mul_pd(longitude_delta, half);
    __m512d sin_latitude_delta_half = nk_sin_f64x8_skylake_(latitude_delta_half);
    __m512d sin_longitude_delta_half = nk_sin_f64x8_skylake_(longitude_delta_half);
    __m512d sin_squared_latitude_delta_half = _mm512_mul_pd(sin_latitude_delta_half, sin_latitude_delta_half);
    __m512d sin_squared_longitude_delta_half = _mm512_mul_pd(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    __m512d cos_first_latitude = nk_cos_f64x8_skylake_(first_latitudes);
    __m512d cos_second_latitude = nk_cos_f64x8_skylake_(second_latitudes);
    __m512d cos_latitude_product = _mm512_mul_pd(cos_first_latitude, cos_second_latitude);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    __m512d haversine_term = _mm512_add_pd(sin_squared_latitude_delta_half,
                                           _mm512_mul_pd(cos_latitude_product, sin_squared_longitude_delta_half));
    // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
    __m512d zero = _mm512_setzero_pd();
    haversine_term = _mm512_max_pd(zero, _mm512_min_pd(one, haversine_term));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    __m512d sqrt_haversine = _mm512_sqrt_pd(haversine_term);
    __m512d sqrt_complement = _mm512_sqrt_pd(_mm512_sub_pd(one, haversine_term));
    __m512d central_angle = _mm512_mul_pd(two, nk_atan2_f64x8_skylake_(sqrt_haversine, sqrt_complement));

    return _mm512_mul_pd(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f64_skylake(            //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 8) {
        __m512d first_latitudes = _mm512_loadu_pd(a_lats);
        __m512d first_longitudes = _mm512_loadu_pd(a_lons);
        __m512d second_latitudes = _mm512_loadu_pd(b_lats);
        __m512d second_longitudes = _mm512_loadu_pd(b_lons);

        __m512d distances = nk_haversine_f64x8_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        _mm512_storeu_pd(results, distances);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with masked operations
    if (n > 0) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, n);
        __m512d first_latitudes = _mm512_maskz_loadu_pd(mask, a_lats);
        __m512d first_longitudes = _mm512_maskz_loadu_pd(mask, a_lons);
        __m512d second_latitudes = _mm512_maskz_loadu_pd(mask, b_lats);
        __m512d second_longitudes = _mm512_maskz_loadu_pd(mask, b_lons);

        __m512d distances = nk_haversine_f64x8_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        _mm512_mask_storeu_pd(results, mask, distances);
    }
}

/**
 *  @brief  AVX-512 helper for Vincenty's geodesic distance on 8 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking.
 */
NK_INTERNAL __m512d nk_vincenty_f64x8_skylake_(        //
    __m512d first_latitudes, __m512d first_longitudes, //
    __m512d second_latitudes, __m512d second_longitudes) {

    __m512d const equatorial_radius = _mm512_set1_pd(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m512d const polar_radius = _mm512_set1_pd(NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m512d const flattening = _mm512_set1_pd(1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m512d const convergence_threshold = _mm512_set1_pd(NK_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    __m512d const one = _mm512_set1_pd(1.0);
    __m512d const two = _mm512_set1_pd(2.0);
    __m512d const three = _mm512_set1_pd(3.0);
    __m512d const four = _mm512_set1_pd(4.0);
    __m512d const six = _mm512_set1_pd(6.0);
    __m512d const sixteen = _mm512_set1_pd(16.0);

    // Longitude difference
    __m512d longitude_difference = _mm512_sub_pd(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    __m512d one_minus_f = _mm512_sub_pd(one, flattening);
    __m512d tan_first = _mm512_div_pd(nk_sin_f64x8_skylake_(first_latitudes), nk_cos_f64x8_skylake_(first_latitudes));
    __m512d tan_second = _mm512_div_pd(nk_sin_f64x8_skylake_(second_latitudes),
                                       nk_cos_f64x8_skylake_(second_latitudes));
    __m512d tan_reduced_first = _mm512_mul_pd(one_minus_f, tan_first);
    __m512d tan_reduced_second = _mm512_mul_pd(one_minus_f, tan_second);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    __m512d cos_reduced_first = _mm512_div_pd(
        one, _mm512_sqrt_pd(_mm512_fmadd_pd(tan_reduced_first, tan_reduced_first, one)));
    __m512d sin_reduced_first = _mm512_mul_pd(tan_reduced_first, cos_reduced_first);
    __m512d cos_reduced_second = _mm512_div_pd(
        one, _mm512_sqrt_pd(_mm512_fmadd_pd(tan_reduced_second, tan_reduced_second, one)));
    __m512d sin_reduced_second = _mm512_mul_pd(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    __m512d lambda = longitude_difference;
    __m512d sin_angular_distance, cos_angular_distance, angular_distance;
    __m512d sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points
    __mmask8 converged_mask = 0;
    __mmask8 coincident_mask = 0;

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS && converged_mask != 0xFF; ++iteration) {
        __m512d sin_lambda = nk_sin_f64x8_skylake_(lambda);
        __m512d cos_lambda = nk_cos_f64x8_skylake_(lambda);

        // sin²(angular_distance) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        __m512d cross_term = _mm512_mul_pd(cos_reduced_second, sin_lambda);
        __m512d mixed_term = _mm512_sub_pd(
            _mm512_mul_pd(cos_reduced_first, sin_reduced_second),
            _mm512_mul_pd(_mm512_mul_pd(sin_reduced_first, cos_reduced_second), cos_lambda));
        __m512d sin_angular_dist_sq = _mm512_fmadd_pd(cross_term, cross_term, _mm512_mul_pd(mixed_term, mixed_term));
        sin_angular_distance = _mm512_sqrt_pd(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = _mm512_cmp_pd_mask(sin_angular_distance, _mm512_set1_pd(1e-15), _CMP_LT_OS);

        // cos(angular_distance) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance = _mm512_fmadd_pd(_mm512_mul_pd(cos_reduced_first, cos_reduced_second), cos_lambda,
                                               _mm512_mul_pd(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_atan2_f64x8_skylake_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance)
        // Use masked divide: zero result for coincident lanes, avoids division by zero
        sin_azimuth = _mm512_maskz_div_pd(
            _knot_mask8(coincident_mask),
            _mm512_mul_pd(_mm512_mul_pd(cos_reduced_first, cos_reduced_second), sin_lambda), sin_angular_distance);
        cos_squared_azimuth = _mm512_sub_pd(one, _mm512_mul_pd(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α = 0
        __mmask8 equatorial_mask = _mm512_cmp_pd_mask(cos_squared_azimuth, _mm512_set1_pd(1e-15), _CMP_LT_OS);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        // Use masked divide: for equatorial lanes, quotient = cos_angular_distance (passthrough),
        // so subtraction yields zero. Avoids division by zero.
        __m512d sin_product = _mm512_mul_pd(sin_reduced_first, sin_reduced_second);
        __m512d quotient = _mm512_mask_div_pd(cos_angular_distance, _knot_mask8(equatorial_mask),
                                              _mm512_mul_pd(two, sin_product), cos_squared_azimuth);
        cos_double_angular_midpoint = _mm512_sub_pd(cos_angular_distance, quotient);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        __m512d correction_factor = _mm512_mul_pd(
            _mm512_div_pd(flattening, sixteen),
            _mm512_mul_pd(cos_squared_azimuth,
                          _mm512_fmadd_pd(flattening, _mm512_fnmadd_pd(three, cos_squared_azimuth, four), four)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        __m512d cos_2sm_sq = _mm512_mul_pd(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2 × cos²(2σₘ)
        __m512d innermost = _mm512_fmadd_pd(two, cos_2sm_sq, _mm512_set1_pd(-1.0));
        // middle = cos(2σₘ) + C × cos(σ) × innermost
        __m512d middle = _mm512_fmadd_pd(_mm512_mul_pd(correction_factor, cos_angular_distance), innermost,
                                         cos_double_angular_midpoint);
        // inner = C × sin(σ) × middle
        __m512d inner = _mm512_mul_pd(_mm512_mul_pd(correction_factor, sin_angular_distance), middle);

        // λ' = L + (1-C) * f * sin_α * (σ + inner)
        __m512d lambda_new = _mm512_fmadd_pd(
            _mm512_mul_pd(_mm512_mul_pd(_mm512_sub_pd(one, correction_factor), flattening), sin_azimuth),
            _mm512_add_pd(angular_distance, inner), longitude_difference);

        // Check convergence: |λ - λ'| < threshold
        __m512d lambda_diff = _mm512_abs_pd(_mm512_sub_pd(lambda_new, lambda));
        converged_mask = _mm512_cmp_pd_mask(lambda_diff, convergence_threshold, _CMP_LT_OS);

        lambda = lambda_new;
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    __m512d a_sq = _mm512_mul_pd(equatorial_radius, equatorial_radius);
    __m512d b_sq = _mm512_mul_pd(polar_radius, polar_radius);
    __m512d u_squared = _mm512_div_pd(_mm512_mul_pd(cos_squared_azimuth, _mm512_sub_pd(a_sq, b_sq)), b_sq);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    __m512d series_a = _mm512_fmadd_pd(u_squared, _mm512_set1_pd(-175.0), _mm512_set1_pd(320.0));
    series_a = _mm512_fmadd_pd(u_squared, series_a, _mm512_set1_pd(-768.0));
    series_a = _mm512_fmadd_pd(u_squared, series_a, _mm512_set1_pd(4096.0));
    series_a = _mm512_fmadd_pd(_mm512_div_pd(u_squared, _mm512_set1_pd(16384.0)), series_a, one);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    __m512d series_b = _mm512_fmadd_pd(u_squared, _mm512_set1_pd(-47.0), _mm512_set1_pd(74.0));
    series_b = _mm512_fmadd_pd(u_squared, series_b, _mm512_set1_pd(-128.0));
    series_b = _mm512_fmadd_pd(u_squared, series_b, _mm512_set1_pd(256.0));
    series_b = _mm512_mul_pd(_mm512_div_pd(u_squared, _mm512_set1_pd(1024.0)), series_b);

    // Δσ = B × sin(σ) × (cos(2σₘ) +
    //      B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m512d cos_2sm_sq = _mm512_mul_pd(cos_double_angular_midpoint, cos_double_angular_midpoint);
    __m512d sin_sq = _mm512_mul_pd(sin_angular_distance, sin_angular_distance);
    __m512d term1 = _mm512_fmadd_pd(two, cos_2sm_sq, _mm512_set1_pd(-1.0));
    term1 = _mm512_mul_pd(cos_angular_distance, term1);
    __m512d term2 = _mm512_fmadd_pd(four, sin_sq, _mm512_set1_pd(-3.0));
    __m512d term3 = _mm512_fmadd_pd(four, cos_2sm_sq, _mm512_set1_pd(-3.0));
    term2 = _mm512_mul_pd(_mm512_mul_pd(_mm512_div_pd(series_b, six), cos_double_angular_midpoint),
                          _mm512_mul_pd(term2, term3));
    __m512d delta_sigma = _mm512_mul_pd(
        series_b, _mm512_mul_pd(sin_angular_distance, _mm512_add_pd(cos_double_angular_midpoint,
                                                                    _mm512_mul_pd(_mm512_div_pd(series_b, four),
                                                                                  _mm512_sub_pd(term1, term2)))));

    // s = b * A * (σ - Δσ)
    __m512d distances = _mm512_mul_pd(_mm512_mul_pd(polar_radius, series_a),
                                      _mm512_sub_pd(angular_distance, delta_sigma));

    // Set coincident points to zero
    distances = _mm512_mask_blend_pd(coincident_mask, distances, _mm512_setzero_pd());

    return distances;
}

NK_PUBLIC void nk_vincenty_f64_skylake(             //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 8) {
        __m512d first_latitudes = _mm512_loadu_pd(a_lats);
        __m512d first_longitudes = _mm512_loadu_pd(a_lons);
        __m512d second_latitudes = _mm512_loadu_pd(b_lats);
        __m512d second_longitudes = _mm512_loadu_pd(b_lons);

        __m512d distances = nk_vincenty_f64x8_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                       second_longitudes);
        _mm512_storeu_pd(results, distances);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with masked operations
    if (n > 0) {
        __mmask8 mask = (__mmask8)_bzhi_u32(0xFF, n);
        __m512d first_latitudes = _mm512_maskz_loadu_pd(mask, a_lats);
        __m512d first_longitudes = _mm512_maskz_loadu_pd(mask, a_lons);
        __m512d second_latitudes = _mm512_maskz_loadu_pd(mask, b_lats);
        __m512d second_longitudes = _mm512_maskz_loadu_pd(mask, b_lons);

        __m512d distances = nk_vincenty_f64x8_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                       second_longitudes);
        _mm512_mask_storeu_pd(results, mask, distances);
    }
}

NK_INTERNAL __m512 nk_haversine_f32x16_skylake_(     //
    __m512 first_latitudes, __m512 first_longitudes, //
    __m512 second_latitudes, __m512 second_longitudes) {

    __m512 const earth_radius = _mm512_set1_ps((float)NK_EARTH_MEDIATORIAL_RADIUS);
    __m512 const half = _mm512_set1_ps(0.5f);
    __m512 const one = _mm512_set1_ps(1.0f);
    __m512 const two = _mm512_set1_ps(2.0f);

    __m512 latitude_delta = _mm512_sub_ps(second_latitudes, first_latitudes);
    __m512 longitude_delta = _mm512_sub_ps(second_longitudes, first_longitudes);

    // Haversine terms: sin²(Δ/2)
    __m512 latitude_delta_half = _mm512_mul_ps(latitude_delta, half);
    __m512 longitude_delta_half = _mm512_mul_ps(longitude_delta, half);
    __m512 sin_latitude_delta_half = nk_sin_f32x16_skylake_(latitude_delta_half);
    __m512 sin_longitude_delta_half = nk_sin_f32x16_skylake_(longitude_delta_half);
    __m512 sin_squared_latitude_delta_half = _mm512_mul_ps(sin_latitude_delta_half, sin_latitude_delta_half);
    __m512 sin_squared_longitude_delta_half = _mm512_mul_ps(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    __m512 cos_first_latitude = nk_cos_f32x16_skylake_(first_latitudes);
    __m512 cos_second_latitude = nk_cos_f32x16_skylake_(second_latitudes);
    __m512 cos_latitude_product = _mm512_mul_ps(cos_first_latitude, cos_second_latitude);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    __m512 haversine_term = _mm512_add_ps(sin_squared_latitude_delta_half,
                                          _mm512_mul_ps(cos_latitude_product, sin_squared_longitude_delta_half));

    // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
    __m512 zero = _mm512_setzero_ps();
    haversine_term = _mm512_max_ps(zero, _mm512_min_ps(one, haversine_term));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    __m512 sqrt_haversine = _mm512_sqrt_ps(haversine_term);
    __m512 sqrt_complement = _mm512_sqrt_ps(_mm512_sub_ps(one, haversine_term));
    __m512 central_angle = _mm512_mul_ps(two, nk_atan2_f32x16_skylake_(sqrt_haversine, sqrt_complement));

    return _mm512_mul_ps(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f32_skylake(            //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 16) {
        __m512 first_latitudes = _mm512_loadu_ps(a_lats);
        __m512 first_longitudes = _mm512_loadu_ps(a_lons);
        __m512 second_latitudes = _mm512_loadu_ps(b_lats);
        __m512 second_longitudes = _mm512_loadu_ps(b_lons);

        __m512 distances = nk_haversine_f32x16_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        _mm512_storeu_ps(results, distances);

        a_lats += 16, a_lons += 16, b_lats += 16, b_lons += 16, results += 16, n -= 16;
    }

    // Handle remaining elements with masked operations
    if (n > 0) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        __m512 first_latitudes = _mm512_maskz_loadu_ps(mask, a_lats);
        __m512 first_longitudes = _mm512_maskz_loadu_ps(mask, a_lons);
        __m512 second_latitudes = _mm512_maskz_loadu_ps(mask, b_lats);
        __m512 second_longitudes = _mm512_maskz_loadu_ps(mask, b_lons);

        __m512 distances = nk_haversine_f32x16_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        _mm512_mask_storeu_ps(results, mask, distances);
    }
}

/**
 *  @brief  AVX-512 helper for Vincenty's geodesic distance on 16 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking.
 */
NK_INTERNAL __m512 nk_vincenty_f32x16_skylake_(      //
    __m512 first_latitudes, __m512 first_longitudes, //
    __m512 second_latitudes, __m512 second_longitudes) {

    __m512 const equatorial_radius = _mm512_set1_ps((float)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m512 const polar_radius = _mm512_set1_ps((float)NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m512 const flattening = _mm512_set1_ps(1.0f / (float)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m512 const convergence_threshold = _mm512_set1_ps(NK_VINCENTY_CONVERGENCE_THRESHOLD_F32);
    __m512 const one = _mm512_set1_ps(1.0f);
    __m512 const two = _mm512_set1_ps(2.0f);
    __m512 const three = _mm512_set1_ps(3.0f);
    __m512 const four = _mm512_set1_ps(4.0f);
    __m512 const six = _mm512_set1_ps(6.0f);
    __m512 const sixteen = _mm512_set1_ps(16.0f);

    // Longitude difference
    __m512 longitude_difference = _mm512_sub_ps(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    __m512 one_minus_f = _mm512_sub_ps(one, flattening);
    __m512 tan_first = _mm512_div_ps(nk_sin_f32x16_skylake_(first_latitudes), nk_cos_f32x16_skylake_(first_latitudes));
    __m512 tan_second = _mm512_div_ps(nk_sin_f32x16_skylake_(second_latitudes),
                                      nk_cos_f32x16_skylake_(second_latitudes));
    __m512 tan_reduced_first = _mm512_mul_ps(one_minus_f, tan_first);
    __m512 tan_reduced_second = _mm512_mul_ps(one_minus_f, tan_second);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    __m512 cos_reduced_first = _mm512_div_ps(
        one, _mm512_sqrt_ps(_mm512_fmadd_ps(tan_reduced_first, tan_reduced_first, one)));
    __m512 sin_reduced_first = _mm512_mul_ps(tan_reduced_first, cos_reduced_first);
    __m512 cos_reduced_second = _mm512_div_ps(
        one, _mm512_sqrt_ps(_mm512_fmadd_ps(tan_reduced_second, tan_reduced_second, one)));
    __m512 sin_reduced_second = _mm512_mul_ps(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    __m512 lambda = longitude_difference;
    __m512 sin_angular_distance, cos_angular_distance, angular_distance;
    __m512 sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points
    __mmask16 converged_mask = 0;
    __mmask16 coincident_mask = 0;

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS && converged_mask != 0xFFFF; ++iteration) {
        __m512 sin_lambda = nk_sin_f32x16_skylake_(lambda);
        __m512 cos_lambda = nk_cos_f32x16_skylake_(lambda);

        // sin²(angular_distance) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        __m512 cross_term = _mm512_mul_ps(cos_reduced_second, sin_lambda);
        __m512 mixed_term = _mm512_sub_ps(
            _mm512_mul_ps(cos_reduced_first, sin_reduced_second),
            _mm512_mul_ps(_mm512_mul_ps(sin_reduced_first, cos_reduced_second), cos_lambda));
        __m512 sin_angular_dist_sq = _mm512_fmadd_ps(cross_term, cross_term, _mm512_mul_ps(mixed_term, mixed_term));
        sin_angular_distance = _mm512_sqrt_ps(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = _mm512_cmp_ps_mask(sin_angular_distance, _mm512_set1_ps(1e-7f), _CMP_LT_OS);

        // cos(angular_distance) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance = _mm512_fmadd_ps(_mm512_mul_ps(cos_reduced_first, cos_reduced_second), cos_lambda,
                                               _mm512_mul_ps(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_atan2_f32x16_skylake_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance)
        // Use masked divide: zero result for coincident lanes, avoids division by zero
        sin_azimuth = _mm512_maskz_div_ps(
            _knot_mask16(coincident_mask),
            _mm512_mul_ps(_mm512_mul_ps(cos_reduced_first, cos_reduced_second), sin_lambda), sin_angular_distance);
        cos_squared_azimuth = _mm512_sub_ps(one, _mm512_mul_ps(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α = 0
        __mmask16 equatorial_mask = _mm512_cmp_ps_mask(cos_squared_azimuth, _mm512_set1_ps(1e-7f), _CMP_LT_OS);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        // Use masked divide: for equatorial lanes, quotient = cos_angular_distance (passthrough),
        // so subtraction yields zero. Avoids division by zero.
        __m512 sin_product = _mm512_mul_ps(sin_reduced_first, sin_reduced_second);
        __m512 quotient = _mm512_mask_div_ps(cos_angular_distance, _knot_mask16(equatorial_mask),
                                             _mm512_mul_ps(two, sin_product), cos_squared_azimuth);
        cos_double_angular_midpoint = _mm512_sub_ps(cos_angular_distance, quotient);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        __m512 correction_factor = _mm512_mul_ps(
            _mm512_div_ps(flattening, sixteen),
            _mm512_mul_ps(cos_squared_azimuth,
                          _mm512_fmadd_ps(flattening, _mm512_fnmadd_ps(three, cos_squared_azimuth, four), four)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        __m512 cos_2sm_sq = _mm512_mul_ps(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2 × cos²(2σₘ)
        __m512 innermost = _mm512_fmadd_ps(two, cos_2sm_sq, _mm512_set1_ps(-1.0f));
        // middle = cos(2σₘ) + C × cos(σ) × innermost
        __m512 middle = _mm512_fmadd_ps(_mm512_mul_ps(correction_factor, cos_angular_distance), innermost,
                                        cos_double_angular_midpoint);
        // inner = C × sin(σ) × middle
        __m512 inner = _mm512_mul_ps(_mm512_mul_ps(correction_factor, sin_angular_distance), middle);

        // λ' = L + (1-C) * f * sin_α * (σ + inner)
        __m512 lambda_new = _mm512_fmadd_ps(
            _mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(one, correction_factor), flattening), sin_azimuth),
            _mm512_add_ps(angular_distance, inner), longitude_difference);

        // Check convergence: |λ - λ'| < threshold
        __m512 lambda_diff = _mm512_abs_ps(_mm512_sub_ps(lambda_new, lambda));
        converged_mask = _mm512_cmp_ps_mask(lambda_diff, convergence_threshold, _CMP_LT_OS);

        lambda = lambda_new;
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    __m512 a_sq = _mm512_mul_ps(equatorial_radius, equatorial_radius);
    __m512 b_sq = _mm512_mul_ps(polar_radius, polar_radius);
    __m512 u_squared = _mm512_div_ps(_mm512_mul_ps(cos_squared_azimuth, _mm512_sub_ps(a_sq, b_sq)), b_sq);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    __m512 series_a = _mm512_fmadd_ps(u_squared, _mm512_set1_ps(-175.0f), _mm512_set1_ps(320.0f));
    series_a = _mm512_fmadd_ps(u_squared, series_a, _mm512_set1_ps(-768.0f));
    series_a = _mm512_fmadd_ps(u_squared, series_a, _mm512_set1_ps(4096.0f));
    series_a = _mm512_fmadd_ps(_mm512_div_ps(u_squared, _mm512_set1_ps(16384.0f)), series_a, one);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    __m512 series_b = _mm512_fmadd_ps(u_squared, _mm512_set1_ps(-47.0f), _mm512_set1_ps(74.0f));
    series_b = _mm512_fmadd_ps(u_squared, series_b, _mm512_set1_ps(-128.0f));
    series_b = _mm512_fmadd_ps(u_squared, series_b, _mm512_set1_ps(256.0f));
    series_b = _mm512_mul_ps(_mm512_div_ps(u_squared, _mm512_set1_ps(1024.0f)), series_b);

    // Δσ = B × sin(σ) × (cos(2σₘ) +
    //      B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m512 cos_2sm_sq = _mm512_mul_ps(cos_double_angular_midpoint, cos_double_angular_midpoint);
    __m512 sin_sq = _mm512_mul_ps(sin_angular_distance, sin_angular_distance);
    __m512 term1 = _mm512_fmadd_ps(two, cos_2sm_sq, _mm512_set1_ps(-1.0f));
    term1 = _mm512_mul_ps(cos_angular_distance, term1);
    __m512 term2 = _mm512_fmadd_ps(four, sin_sq, _mm512_set1_ps(-3.0f));
    __m512 term3 = _mm512_fmadd_ps(four, cos_2sm_sq, _mm512_set1_ps(-3.0f));
    term2 = _mm512_mul_ps(_mm512_mul_ps(_mm512_div_ps(series_b, six), cos_double_angular_midpoint),
                          _mm512_mul_ps(term2, term3));
    __m512 delta_sigma = _mm512_mul_ps(
        series_b, _mm512_mul_ps(sin_angular_distance, _mm512_add_ps(cos_double_angular_midpoint,
                                                                    _mm512_mul_ps(_mm512_div_ps(series_b, four),
                                                                                  _mm512_sub_ps(term1, term2)))));

    // s = b * A * (σ - Δσ)
    __m512 distances = _mm512_mul_ps(_mm512_mul_ps(polar_radius, series_a),
                                     _mm512_sub_ps(angular_distance, delta_sigma));

    // Set coincident points to zero
    distances = _mm512_mask_blend_ps(coincident_mask, distances, _mm512_setzero_ps());

    return distances;
}

NK_PUBLIC void nk_vincenty_f32_skylake(             //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 16) {
        __m512 first_latitudes = _mm512_loadu_ps(a_lats);
        __m512 first_longitudes = _mm512_loadu_ps(a_lons);
        __m512 second_latitudes = _mm512_loadu_ps(b_lats);
        __m512 second_longitudes = _mm512_loadu_ps(b_lons);

        __m512 distances = nk_vincenty_f32x16_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                       second_longitudes);
        _mm512_storeu_ps(results, distances);

        a_lats += 16, a_lons += 16, b_lats += 16, b_lons += 16, results += 16, n -= 16;
    }

    // Handle remaining elements with masked operations
    if (n > 0) {
        __mmask16 mask = (__mmask16)_bzhi_u32(0xFFFF, n);
        __m512 first_latitudes = _mm512_maskz_loadu_ps(mask, a_lats);
        __m512 first_longitudes = _mm512_maskz_loadu_ps(mask, a_lons);
        __m512 second_latitudes = _mm512_maskz_loadu_ps(mask, b_lats);
        __m512 second_longitudes = _mm512_maskz_loadu_ps(mask, b_lons);

        __m512 distances = nk_vincenty_f32x16_skylake_(first_latitudes, first_longitudes, second_latitudes,
                                                       second_longitudes);
        _mm512_mask_storeu_ps(results, mask, distances);
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
#endif // NK_TARGET_X86_
#endif // NK_GEOSPATIAL_SKYLAKE_H
