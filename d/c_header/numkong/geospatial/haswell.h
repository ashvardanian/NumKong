/**
 *  @brief SIMD-accelerated Geospatial Distances for Haswell.
 *  @file include/numkong/geospatial/haswell.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/geospatial.h
 *
 *  @section geospatial_haswell_instructions Key AVX2 Geospatial Instructions
 *
 *      Intrinsic               Instruction                     Ice         Genoa
 *      _mm256_sqrt_ps          VSQRTPS (YMM, YMM)              12c @ p0    15c @ p01
 *      _mm256_sqrt_pd          VSQRTPD (YMM, YMM)              13c @ p0    21c @ p01
 *      _mm256_div_ps           VDIVPS (YMM, YMM, YMM)          11c @ p0    11c @ p01
 *      _mm256_div_pd           VDIVPD (YMM, YMM, YMM)          13c @ p0    13c @ p01
 *      _mm256_fmadd_ps         VFMADD231PS (YMM, YMM, YMM)     4c @ p01    4c @ p01
 *      _mm256_fmadd_pd         VFMADD231PD (YMM, YMM, YMM)     4c @ p01    4c @ p01
 */
#ifndef NK_GEOSPATIAL_HASWELL_H
#define NK_GEOSPATIAL_HASWELL_H

#if NK_TARGET_X86_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
#include "numkong/trigonometry/haswell.h" // `nk_sin_f64x4_haswell_`, `nk_cos_f64x4_haswell_`, `nk_atan2_f64x4_haswell_`, etc.

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif

/*  Haswell AVX2 implementations using 4-wide f64 and 8-wide f32 SIMD.
 *  These require AVX2 trigonometric kernels from trigonometry.h.
 */

NK_INTERNAL __m256d nk_haversine_f64x4_haswell_(       //
    __m256d first_latitudes, __m256d first_longitudes, //
    __m256d second_latitudes, __m256d second_longitudes) {

    __m256d const earth_radius = _mm256_set1_pd(NK_EARTH_MEDIATORIAL_RADIUS);
    __m256d const half = _mm256_set1_pd(0.5);
    __m256d const one = _mm256_set1_pd(1.0);
    __m256d const two = _mm256_set1_pd(2.0);

    __m256d latitude_delta = _mm256_sub_pd(second_latitudes, first_latitudes);
    __m256d longitude_delta = _mm256_sub_pd(second_longitudes, first_longitudes);

    // Haversine terms: sin²(Δ/2)
    __m256d latitude_delta_half = _mm256_mul_pd(latitude_delta, half);
    __m256d longitude_delta_half = _mm256_mul_pd(longitude_delta, half);
    __m256d sin_latitude_delta_half = nk_sin_f64x4_haswell_(latitude_delta_half);
    __m256d sin_longitude_delta_half = nk_sin_f64x4_haswell_(longitude_delta_half);
    __m256d sin_squared_latitude_delta_half = _mm256_mul_pd(sin_latitude_delta_half, sin_latitude_delta_half);
    __m256d sin_squared_longitude_delta_half = _mm256_mul_pd(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    __m256d cos_first_latitude = nk_cos_f64x4_haswell_(first_latitudes);
    __m256d cos_second_latitude = nk_cos_f64x4_haswell_(second_latitudes);
    __m256d cos_latitude_product = _mm256_mul_pd(cos_first_latitude, cos_second_latitude);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    __m256d haversine_term = _mm256_add_pd(sin_squared_latitude_delta_half,
                                           _mm256_mul_pd(cos_latitude_product, sin_squared_longitude_delta_half));
    // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
    __m256d zero = _mm256_setzero_pd();
    haversine_term = _mm256_max_pd(zero, _mm256_min_pd(one, haversine_term));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    __m256d sqrt_haversine = _mm256_sqrt_pd(haversine_term);
    __m256d sqrt_complement = _mm256_sqrt_pd(_mm256_sub_pd(one, haversine_term));
    __m256d central_angle = _mm256_mul_pd(two, nk_atan2_f64x4_haswell_(sqrt_haversine, sqrt_complement));

    return _mm256_mul_pd(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f64_haswell(            //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 4) {
        __m256d first_latitudes = _mm256_loadu_pd(a_lats);
        __m256d first_longitudes = _mm256_loadu_pd(a_lons);
        __m256d second_latitudes = _mm256_loadu_pd(b_lats);
        __m256d second_longitudes = _mm256_loadu_pd(b_lons);

        __m256d distances = nk_haversine_f64x4_haswell_(first_latitudes, first_longitudes, second_latitudes,
                                                        second_longitudes);
        _mm256_storeu_pd(results, distances);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with partial loads (n can be 1-3 here)
    if (n > 0) {
        nk_b256_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x4_haswell_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x4_haswell_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x4_haswell_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x4_haswell_(b_lons, &b_lon_vec, n);
        __m256d distances = nk_haversine_f64x4_haswell_(a_lat_vec.ymm_pd, a_lon_vec.ymm_pd, b_lat_vec.ymm_pd,
                                                        b_lon_vec.ymm_pd);
        result_vec.ymm_pd = distances;
        nk_partial_store_b64x4_haswell_(&result_vec, results, n);
    }
}

NK_INTERNAL __m256 nk_haversine_f32x8_haswell_(      //
    __m256 first_latitudes, __m256 first_longitudes, //
    __m256 second_latitudes, __m256 second_longitudes) {

    __m256 const earth_radius = _mm256_set1_ps((float)NK_EARTH_MEDIATORIAL_RADIUS);
    __m256 const half = _mm256_set1_ps(0.5f);
    __m256 const one = _mm256_set1_ps(1.0f);
    __m256 const two = _mm256_set1_ps(2.0f);

    __m256 latitude_delta = _mm256_sub_ps(second_latitudes, first_latitudes);
    __m256 longitude_delta = _mm256_sub_ps(second_longitudes, first_longitudes);

    // Haversine terms: sin²(Δ/2)
    __m256 latitude_delta_half = _mm256_mul_ps(latitude_delta, half);
    __m256 longitude_delta_half = _mm256_mul_ps(longitude_delta, half);
    __m256 sin_latitude_delta_half = nk_sin_f32x8_haswell_(latitude_delta_half);
    __m256 sin_longitude_delta_half = nk_sin_f32x8_haswell_(longitude_delta_half);
    __m256 sin_squared_latitude_delta_half = _mm256_mul_ps(sin_latitude_delta_half, sin_latitude_delta_half);
    __m256 sin_squared_longitude_delta_half = _mm256_mul_ps(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    __m256 cos_first_latitude = nk_cos_f32x8_haswell_(first_latitudes);
    __m256 cos_second_latitude = nk_cos_f32x8_haswell_(second_latitudes);
    __m256 cos_latitude_product = _mm256_mul_ps(cos_first_latitude, cos_second_latitude);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    __m256 haversine_term = _mm256_add_ps(sin_squared_latitude_delta_half,
                                          _mm256_mul_ps(cos_latitude_product, sin_squared_longitude_delta_half));

    // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
    __m256 zero = _mm256_setzero_ps();
    haversine_term = _mm256_max_ps(zero, _mm256_min_ps(one, haversine_term));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    __m256 sqrt_haversine = _mm256_sqrt_ps(haversine_term);
    __m256 sqrt_complement = _mm256_sqrt_ps(_mm256_sub_ps(one, haversine_term));
    __m256 central_angle = _mm256_mul_ps(two, nk_atan2_f32x8_haswell_(sqrt_haversine, sqrt_complement));

    return _mm256_mul_ps(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f32_haswell(            //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 8) {
        __m256 first_latitudes = _mm256_loadu_ps(a_lats);
        __m256 first_longitudes = _mm256_loadu_ps(a_lons);
        __m256 second_latitudes = _mm256_loadu_ps(b_lats);
        __m256 second_longitudes = _mm256_loadu_ps(b_lons);

        __m256 distances = nk_haversine_f32x8_haswell_(first_latitudes, first_longitudes, second_latitudes,
                                                       second_longitudes);
        _mm256_storeu_ps(results, distances);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with partial loads (n can be 1-7 here)
    if (n > 0) {
        nk_b256_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x8_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x8_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x8_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x8_serial_(b_lons, &b_lon_vec, n);
        __m256 distances = nk_haversine_f32x8_haswell_(a_lat_vec.ymm_ps, a_lon_vec.ymm_ps, b_lat_vec.ymm_ps,
                                                       b_lon_vec.ymm_ps);
        result_vec.ymm_ps = distances;
        nk_partial_store_b32x8_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  AVX2 helper for Vincenty's geodesic distance on 4 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL __m256d nk_vincenty_f64x4_haswell_(        //
    __m256d first_latitudes, __m256d first_longitudes, //
    __m256d second_latitudes, __m256d second_longitudes) {

    __m256d const equatorial_radius = _mm256_set1_pd(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m256d const polar_radius = _mm256_set1_pd(NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m256d const flattening = _mm256_set1_pd(1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m256d const convergence_threshold = _mm256_set1_pd(NK_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    __m256d const one = _mm256_set1_pd(1.0);
    __m256d const two = _mm256_set1_pd(2.0);
    __m256d const three = _mm256_set1_pd(3.0);
    __m256d const four = _mm256_set1_pd(4.0);
    __m256d const six = _mm256_set1_pd(6.0);
    __m256d const sixteen = _mm256_set1_pd(16.0);
    __m256d const epsilon = _mm256_set1_pd(1e-15);

    // Longitude difference
    __m256d longitude_difference = _mm256_sub_pd(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    __m256d one_minus_f = _mm256_sub_pd(one, flattening);
    __m256d tan_first = _mm256_div_pd(nk_sin_f64x4_haswell_(first_latitudes), nk_cos_f64x4_haswell_(first_latitudes));
    __m256d tan_second = _mm256_div_pd(nk_sin_f64x4_haswell_(second_latitudes),
                                       nk_cos_f64x4_haswell_(second_latitudes));
    __m256d tan_reduced_first = _mm256_mul_pd(one_minus_f, tan_first);
    __m256d tan_reduced_second = _mm256_mul_pd(one_minus_f, tan_second);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    __m256d cos_reduced_first = _mm256_div_pd(
        one, _mm256_sqrt_pd(_mm256_fmadd_pd(tan_reduced_first, tan_reduced_first, one)));
    __m256d sin_reduced_first = _mm256_mul_pd(tan_reduced_first, cos_reduced_first);
    __m256d cos_reduced_second = _mm256_div_pd(
        one, _mm256_sqrt_pd(_mm256_fmadd_pd(tan_reduced_second, tan_reduced_second, one)));
    __m256d sin_reduced_second = _mm256_mul_pd(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    __m256d lambda = longitude_difference;
    __m256d sin_angular_distance, cos_angular_distance, angular_distance;
    __m256d sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points using masks
    __m256d converged_mask = _mm256_setzero_pd();
    __m256d coincident_mask = _mm256_setzero_pd();

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        int converged_bits = _mm256_movemask_pd(converged_mask);
        if (converged_bits == 0xF) break;

        __m256d sin_lambda = nk_sin_f64x4_haswell_(lambda);
        __m256d cos_lambda = nk_cos_f64x4_haswell_(lambda);

        // sin²(angular_distance) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        __m256d cross_term = _mm256_mul_pd(cos_reduced_second, sin_lambda);
        __m256d mixed_term = _mm256_sub_pd(
            _mm256_mul_pd(cos_reduced_first, sin_reduced_second),
            _mm256_mul_pd(_mm256_mul_pd(sin_reduced_first, cos_reduced_second), cos_lambda));
        __m256d sin_angular_dist_sq = _mm256_fmadd_pd(cross_term, cross_term, _mm256_mul_pd(mixed_term, mixed_term));
        sin_angular_distance = _mm256_sqrt_pd(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = _mm256_cmp_pd(sin_angular_distance, epsilon, _CMP_LT_OS);

        // cos(angular_distance) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance = _mm256_fmadd_pd(_mm256_mul_pd(cos_reduced_first, cos_reduced_second), cos_lambda,
                                               _mm256_mul_pd(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_atan2_f64x4_haswell_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance)
        // Avoid division by zero by using blending
        __m256d safe_sin_angular = _mm256_blendv_pd(sin_angular_distance, one, coincident_mask);
        sin_azimuth = _mm256_div_pd(_mm256_mul_pd(_mm256_mul_pd(cos_reduced_first, cos_reduced_second), sin_lambda),
                                    safe_sin_angular);
        cos_squared_azimuth = _mm256_sub_pd(one, _mm256_mul_pd(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α ≈ 0
        __m256d equatorial_mask = _mm256_cmp_pd(cos_squared_azimuth, epsilon, _CMP_LT_OS);
        __m256d safe_cos_sq_azimuth = _mm256_blendv_pd(cos_squared_azimuth, one, equatorial_mask);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        __m256d sin_product = _mm256_mul_pd(sin_reduced_first, sin_reduced_second);
        cos_double_angular_midpoint = _mm256_sub_pd(
            cos_angular_distance, _mm256_div_pd(_mm256_mul_pd(two, sin_product), safe_cos_sq_azimuth));
        cos_double_angular_midpoint = _mm256_blendv_pd(cos_double_angular_midpoint, _mm256_setzero_pd(),
                                                       equatorial_mask);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        __m256d correction_factor = _mm256_mul_pd(
            _mm256_div_pd(flattening, sixteen),
            _mm256_mul_pd(cos_squared_azimuth,
                          _mm256_fmadd_pd(flattening, _mm256_fnmadd_pd(three, cos_squared_azimuth, four), four)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        __m256d cos_2sm_sq = _mm256_mul_pd(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2 × cos²(2σₘ)
        __m256d innermost = _mm256_fmadd_pd(two, cos_2sm_sq, _mm256_set1_pd(-1.0));
        // middle = cos(2σₘ) + C × cos(σ) × innermost
        __m256d middle = _mm256_fmadd_pd(_mm256_mul_pd(correction_factor, cos_angular_distance), innermost,
                                         cos_double_angular_midpoint);
        // inner = C × sin(σ) × middle
        __m256d inner = _mm256_mul_pd(_mm256_mul_pd(correction_factor, sin_angular_distance), middle);

        // λ' = L + (1-C) * f * sin_α * (σ + inner)
        __m256d lambda_new = _mm256_fmadd_pd(
            _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(one, correction_factor), flattening), sin_azimuth),
            _mm256_add_pd(angular_distance, inner), longitude_difference);

        // Check convergence: |λ - λ'| < threshold
        __m256d lambda_diff_abs = _mm256_andnot_pd(_mm256_set1_pd(-0.0), _mm256_sub_pd(lambda_new, lambda));
        __m256d newly_converged = _mm256_cmp_pd(lambda_diff_abs, convergence_threshold, _CMP_LT_OS);
        converged_mask = _mm256_or_pd(converged_mask, newly_converged);

        // Only update lambda for non-converged lanes
        lambda = _mm256_blendv_pd(lambda_new, lambda, converged_mask);
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    __m256d a_sq = _mm256_mul_pd(equatorial_radius, equatorial_radius);
    __m256d b_sq = _mm256_mul_pd(polar_radius, polar_radius);
    __m256d u_squared = _mm256_div_pd(_mm256_mul_pd(cos_squared_azimuth, _mm256_sub_pd(a_sq, b_sq)), b_sq);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    __m256d series_a = _mm256_fmadd_pd(u_squared, _mm256_set1_pd(-175.0), _mm256_set1_pd(320.0));
    series_a = _mm256_fmadd_pd(u_squared, series_a, _mm256_set1_pd(-768.0));
    series_a = _mm256_fmadd_pd(u_squared, series_a, _mm256_set1_pd(4096.0));
    series_a = _mm256_fmadd_pd(_mm256_div_pd(u_squared, _mm256_set1_pd(16384.0)), series_a, one);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    __m256d series_b = _mm256_fmadd_pd(u_squared, _mm256_set1_pd(-47.0), _mm256_set1_pd(74.0));
    series_b = _mm256_fmadd_pd(u_squared, series_b, _mm256_set1_pd(-128.0));
    series_b = _mm256_fmadd_pd(u_squared, series_b, _mm256_set1_pd(256.0));
    series_b = _mm256_mul_pd(_mm256_div_pd(u_squared, _mm256_set1_pd(1024.0)), series_b);

    // Δσ = B × sin(σ) × (cos(2σₘ) +
    //      B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m256d cos_2sm_sq = _mm256_mul_pd(cos_double_angular_midpoint, cos_double_angular_midpoint);
    __m256d sin_sq = _mm256_mul_pd(sin_angular_distance, sin_angular_distance);
    __m256d term1 = _mm256_fmadd_pd(two, cos_2sm_sq, _mm256_set1_pd(-1.0));
    term1 = _mm256_mul_pd(cos_angular_distance, term1);
    __m256d term2 = _mm256_fmadd_pd(four, sin_sq, _mm256_set1_pd(-3.0));
    __m256d term3 = _mm256_fmadd_pd(four, cos_2sm_sq, _mm256_set1_pd(-3.0));
    term2 = _mm256_mul_pd(_mm256_mul_pd(_mm256_div_pd(series_b, six), cos_double_angular_midpoint),
                          _mm256_mul_pd(term2, term3));
    __m256d delta_sigma = _mm256_mul_pd(
        series_b, _mm256_mul_pd(sin_angular_distance, _mm256_add_pd(cos_double_angular_midpoint,
                                                                    _mm256_mul_pd(_mm256_div_pd(series_b, four),
                                                                                  _mm256_sub_pd(term1, term2)))));

    // s = b * A * (σ - Δσ)
    __m256d distances = _mm256_mul_pd(_mm256_mul_pd(polar_radius, series_a),
                                      _mm256_sub_pd(angular_distance, delta_sigma));

    // Set coincident points to zero
    distances = _mm256_blendv_pd(distances, _mm256_setzero_pd(), coincident_mask);

    return distances;
}

NK_PUBLIC void nk_vincenty_f64_haswell(             //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 4) {
        __m256d first_latitudes = _mm256_loadu_pd(a_lats);
        __m256d first_longitudes = _mm256_loadu_pd(a_lons);
        __m256d second_latitudes = _mm256_loadu_pd(b_lats);
        __m256d second_longitudes = _mm256_loadu_pd(b_lons);

        __m256d distances = nk_vincenty_f64x4_haswell_(first_latitudes, first_longitudes, second_latitudes,
                                                       second_longitudes);
        _mm256_storeu_pd(results, distances);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with partial loads (n can be 1-3 here)
    if (n > 0) {
        nk_b256_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x4_haswell_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x4_haswell_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x4_haswell_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x4_haswell_(b_lons, &b_lon_vec, n);
        __m256d distances = nk_vincenty_f64x4_haswell_(a_lat_vec.ymm_pd, a_lon_vec.ymm_pd, b_lat_vec.ymm_pd,
                                                       b_lon_vec.ymm_pd);
        result_vec.ymm_pd = distances;
        nk_partial_store_b64x4_haswell_(&result_vec, results, n);
    }
}

/**
 *  @brief  AVX2 helper for Vincenty's geodesic distance on 8 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL __m256 nk_vincenty_f32x8_haswell_(       //
    __m256 first_latitudes, __m256 first_longitudes, //
    __m256 second_latitudes, __m256 second_longitudes) {

    __m256 const equatorial_radius = _mm256_set1_ps((float)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m256 const polar_radius = _mm256_set1_ps((float)NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m256 const flattening = _mm256_set1_ps(1.0f / (float)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m256 const convergence_threshold = _mm256_set1_ps(NK_VINCENTY_CONVERGENCE_THRESHOLD_F32);
    __m256 const one = _mm256_set1_ps(1.0f);
    __m256 const two = _mm256_set1_ps(2.0f);
    __m256 const three = _mm256_set1_ps(3.0f);
    __m256 const four = _mm256_set1_ps(4.0f);
    __m256 const six = _mm256_set1_ps(6.0f);
    __m256 const sixteen = _mm256_set1_ps(16.0f);
    __m256 const epsilon = _mm256_set1_ps(1e-7f);

    // Longitude difference
    __m256 longitude_difference = _mm256_sub_ps(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    __m256 one_minus_f = _mm256_sub_ps(one, flattening);
    __m256 tan_first = _mm256_div_ps(nk_sin_f32x8_haswell_(first_latitudes), nk_cos_f32x8_haswell_(first_latitudes));
    __m256 tan_second = _mm256_div_ps(nk_sin_f32x8_haswell_(second_latitudes), nk_cos_f32x8_haswell_(second_latitudes));
    __m256 tan_reduced_first = _mm256_mul_ps(one_minus_f, tan_first);
    __m256 tan_reduced_second = _mm256_mul_ps(one_minus_f, tan_second);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    __m256 cos_reduced_first = _mm256_div_ps(
        one, _mm256_sqrt_ps(_mm256_fmadd_ps(tan_reduced_first, tan_reduced_first, one)));
    __m256 sin_reduced_first = _mm256_mul_ps(tan_reduced_first, cos_reduced_first);
    __m256 cos_reduced_second = _mm256_div_ps(
        one, _mm256_sqrt_ps(_mm256_fmadd_ps(tan_reduced_second, tan_reduced_second, one)));
    __m256 sin_reduced_second = _mm256_mul_ps(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    __m256 lambda = longitude_difference;
    __m256 sin_angular_distance, cos_angular_distance, angular_distance;
    __m256 sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points using masks
    __m256 converged_mask = _mm256_setzero_ps();
    __m256 coincident_mask = _mm256_setzero_ps();

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        int converged_bits = _mm256_movemask_ps(converged_mask);
        if (converged_bits == 0xFF) break;

        __m256 sin_lambda = nk_sin_f32x8_haswell_(lambda);
        __m256 cos_lambda = nk_cos_f32x8_haswell_(lambda);

        // sin²(angular_distance) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        __m256 cross_term = _mm256_mul_ps(cos_reduced_second, sin_lambda);
        __m256 mixed_term = _mm256_sub_ps(
            _mm256_mul_ps(cos_reduced_first, sin_reduced_second),
            _mm256_mul_ps(_mm256_mul_ps(sin_reduced_first, cos_reduced_second), cos_lambda));
        __m256 sin_angular_dist_sq = _mm256_fmadd_ps(cross_term, cross_term, _mm256_mul_ps(mixed_term, mixed_term));
        sin_angular_distance = _mm256_sqrt_ps(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ≈ 0)
        coincident_mask = _mm256_cmp_ps(sin_angular_distance, epsilon, _CMP_LT_OS);

        // cos(angular_distance) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance = _mm256_fmadd_ps(_mm256_mul_ps(cos_reduced_first, cos_reduced_second), cos_lambda,
                                               _mm256_mul_ps(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_atan2_f32x8_haswell_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance)
        // Avoid division by zero by using blending
        __m256 safe_sin_angular = _mm256_blendv_ps(sin_angular_distance, one, coincident_mask);
        sin_azimuth = _mm256_div_ps(_mm256_mul_ps(_mm256_mul_ps(cos_reduced_first, cos_reduced_second), sin_lambda),
                                    safe_sin_angular);
        cos_squared_azimuth = _mm256_sub_ps(one, _mm256_mul_ps(sin_azimuth, sin_azimuth));

        // Handle equatorial case: cos²α ≈ 0
        __m256 equatorial_mask = _mm256_cmp_ps(cos_squared_azimuth, epsilon, _CMP_LT_OS);
        __m256 safe_cos_sq_azimuth = _mm256_blendv_ps(cos_squared_azimuth, one, equatorial_mask);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        __m256 sin_product = _mm256_mul_ps(sin_reduced_first, sin_reduced_second);
        cos_double_angular_midpoint = _mm256_sub_ps(
            cos_angular_distance, _mm256_div_ps(_mm256_mul_ps(two, sin_product), safe_cos_sq_azimuth));
        cos_double_angular_midpoint = _mm256_blendv_ps(cos_double_angular_midpoint, _mm256_setzero_ps(),
                                                       equatorial_mask);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        __m256 correction_factor = _mm256_mul_ps(
            _mm256_div_ps(flattening, sixteen),
            _mm256_mul_ps(cos_squared_azimuth,
                          _mm256_fmadd_ps(flattening, _mm256_fnmadd_ps(three, cos_squared_azimuth, four), four)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        __m256 cos_2sm_sq = _mm256_mul_ps(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2 × cos²(2σₘ)
        __m256 innermost = _mm256_fmadd_ps(two, cos_2sm_sq, _mm256_set1_ps(-1.0f));
        // middle = cos(2σₘ) + C × cos(σ) × innermost
        __m256 middle = _mm256_fmadd_ps(_mm256_mul_ps(correction_factor, cos_angular_distance), innermost,
                                        cos_double_angular_midpoint);
        // inner = C × sin(σ) × middle
        __m256 inner = _mm256_mul_ps(_mm256_mul_ps(correction_factor, sin_angular_distance), middle);

        // λ' = L + (1-C) * f * sin_α * (σ + inner)
        __m256 lambda_new = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(one, correction_factor), flattening), sin_azimuth),
            _mm256_add_ps(angular_distance, inner), longitude_difference);

        // Check convergence: |λ - λ'| < threshold
        __m256 lambda_diff_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), _mm256_sub_ps(lambda_new, lambda));
        __m256 newly_converged = _mm256_cmp_ps(lambda_diff_abs, convergence_threshold, _CMP_LT_OS);
        converged_mask = _mm256_or_ps(converged_mask, newly_converged);

        // Only update lambda for non-converged lanes
        lambda = _mm256_blendv_ps(lambda_new, lambda, converged_mask);
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    __m256 a_sq = _mm256_mul_ps(equatorial_radius, equatorial_radius);
    __m256 b_sq = _mm256_mul_ps(polar_radius, polar_radius);
    __m256 u_squared = _mm256_div_ps(_mm256_mul_ps(cos_squared_azimuth, _mm256_sub_ps(a_sq, b_sq)), b_sq);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    __m256 series_a = _mm256_fmadd_ps(u_squared, _mm256_set1_ps(-175.0f), _mm256_set1_ps(320.0f));
    series_a = _mm256_fmadd_ps(u_squared, series_a, _mm256_set1_ps(-768.0f));
    series_a = _mm256_fmadd_ps(u_squared, series_a, _mm256_set1_ps(4096.0f));
    series_a = _mm256_fmadd_ps(_mm256_div_ps(u_squared, _mm256_set1_ps(16384.0f)), series_a, one);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    __m256 series_b = _mm256_fmadd_ps(u_squared, _mm256_set1_ps(-47.0f), _mm256_set1_ps(74.0f));
    series_b = _mm256_fmadd_ps(u_squared, series_b, _mm256_set1_ps(-128.0f));
    series_b = _mm256_fmadd_ps(u_squared, series_b, _mm256_set1_ps(256.0f));
    series_b = _mm256_mul_ps(_mm256_div_ps(u_squared, _mm256_set1_ps(1024.0f)), series_b);

    // Δσ = B × sin(σ) × (cos(2σₘ) +
    //      B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m256 cos_2sm_sq = _mm256_mul_ps(cos_double_angular_midpoint, cos_double_angular_midpoint);
    __m256 sin_sq = _mm256_mul_ps(sin_angular_distance, sin_angular_distance);
    __m256 term1 = _mm256_fmadd_ps(two, cos_2sm_sq, _mm256_set1_ps(-1.0f));
    term1 = _mm256_mul_ps(cos_angular_distance, term1);
    __m256 term2 = _mm256_fmadd_ps(four, sin_sq, _mm256_set1_ps(-3.0f));
    __m256 term3 = _mm256_fmadd_ps(four, cos_2sm_sq, _mm256_set1_ps(-3.0f));
    term2 = _mm256_mul_ps(_mm256_mul_ps(_mm256_div_ps(series_b, six), cos_double_angular_midpoint),
                          _mm256_mul_ps(term2, term3));
    __m256 delta_sigma = _mm256_mul_ps(
        series_b, _mm256_mul_ps(sin_angular_distance, _mm256_add_ps(cos_double_angular_midpoint,
                                                                    _mm256_mul_ps(_mm256_div_ps(series_b, four),
                                                                                  _mm256_sub_ps(term1, term2)))));

    // s = b * A * (σ - Δσ)
    __m256 distances = _mm256_mul_ps(_mm256_mul_ps(polar_radius, series_a),
                                     _mm256_sub_ps(angular_distance, delta_sigma));

    // Set coincident points to zero
    distances = _mm256_blendv_ps(distances, _mm256_setzero_ps(), coincident_mask);

    return distances;
}

NK_PUBLIC void nk_vincenty_f32_haswell(             //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 8) {
        __m256 first_latitudes = _mm256_loadu_ps(a_lats);
        __m256 first_longitudes = _mm256_loadu_ps(a_lons);
        __m256 second_latitudes = _mm256_loadu_ps(b_lats);
        __m256 second_longitudes = _mm256_loadu_ps(b_lons);

        __m256 distances = nk_vincenty_f32x8_haswell_(first_latitudes, first_longitudes, second_latitudes,
                                                      second_longitudes);
        _mm256_storeu_ps(results, distances);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with partial loads (n can be 1-7 here)
    if (n > 0) {
        nk_b256_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x8_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x8_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x8_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x8_serial_(b_lons, &b_lon_vec, n);
        __m256 distances = nk_vincenty_f32x8_haswell_(a_lat_vec.ymm_ps, a_lon_vec.ymm_ps, b_lat_vec.ymm_ps,
                                                      b_lon_vec.ymm_ps);
        result_vec.ymm_ps = distances;
        nk_partial_store_b32x8_serial_(&result_vec, results, n);
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

#endif // NK_TARGET_HASWELL
#endif // NK_TARGET_X86_
#endif // NK_GEOSPATIAL_HASWELL_H
