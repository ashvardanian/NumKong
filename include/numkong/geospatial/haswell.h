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
 *      Intrinsic        Instruction                  Icelake    Genoa
 *      _mm256_sqrt_ps   VSQRTPS (YMM, YMM)           12cy @ p0  15cy @ p01
 *      _mm256_sqrt_pd   VSQRTPD (YMM, YMM)           13cy @ p0  21cy @ p01
 *      _mm256_div_ps    VDIVPS (YMM, YMM, YMM)       11cy @ p0  11cy @ p01
 *      _mm256_div_pd    VDIVPD (YMM, YMM, YMM)       13cy @ p0  13cy @ p01
 *      _mm256_fmadd_ps  VFMADD231PS (YMM, YMM, YMM)  4cy @ p01  4cy @ p01
 *      _mm256_fmadd_pd  VFMADD231PD (YMM, YMM, YMM)  4cy @ p01  4cy @ p01
 *      _mm256_cmp_ps    VCMPPS (YMM, YMM, YMM, I8)   3cy @ p01  3cy @ p01
 */
#ifndef NK_GEOSPATIAL_HASWELL_H
#define NK_GEOSPATIAL_HASWELL_H

#if NK_TARGET_X8664_
#if NK_TARGET_HASWELL

#include "numkong/types.h"
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

/*  Haswell AVX2 implementations using 4-wide f64 and 8-wide f32 SIMD.
 *  These require AVX2 trigonometric kernels from trigonometry.h.
 */

NK_HELPER_INLINE __m256d nk_haversine_f64x4_haswell_(              //
    __m256d first_latitudes_f64x4, __m256d first_longitudes_f64x4, //
    __m256d second_latitudes_f64x4, __m256d second_longitudes_f64x4) {

    __m256d const earth_radius_f64x4 = _mm256_set1_pd(NK_EARTH_MEDIATORIAL_RADIUS);
    __m256d const half_f64x4 = _mm256_set1_pd(0.5);
    __m256d const one_f64x4 = _mm256_set1_pd(1.0);
    __m256d const two_f64x4 = _mm256_set1_pd(2.0);

    __m256d latitude_delta_f64x4 = _mm256_sub_pd(second_latitudes_f64x4, first_latitudes_f64x4);
    __m256d longitude_delta_f64x4 = _mm256_sub_pd(second_longitudes_f64x4, first_longitudes_f64x4);

    // Haversine terms: sin²(Δ/2)
    __m256d latitude_delta_half_f64x4 = _mm256_mul_pd(latitude_delta_f64x4, half_f64x4);
    __m256d longitude_delta_half_f64x4 = _mm256_mul_pd(longitude_delta_f64x4, half_f64x4);
    __m256d sin_latitude_delta_half_f64x4 = nk_sin_f64x4_haswell_(latitude_delta_half_f64x4);
    __m256d sin_longitude_delta_half_f64x4 = nk_sin_f64x4_haswell_(longitude_delta_half_f64x4);
    __m256d sin_sq_latitude_delta_half_f64x4 = _mm256_mul_pd(sin_latitude_delta_half_f64x4,
                                                             sin_latitude_delta_half_f64x4);
    __m256d sin_sq_longitude_delta_half_f64x4 = _mm256_mul_pd(sin_longitude_delta_half_f64x4,
                                                              sin_longitude_delta_half_f64x4);

    // Latitude cosine product
    __m256d cos_first_latitude_f64x4 = nk_cos_f64x4_haswell_(first_latitudes_f64x4);
    __m256d cos_second_latitude_f64x4 = nk_cos_f64x4_haswell_(second_latitudes_f64x4);
    __m256d cos_latitude_product_f64x4 = _mm256_mul_pd(cos_first_latitude_f64x4, cos_second_latitude_f64x4);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    __m256d haversine_term_f64x4 = _mm256_add_pd(
        sin_sq_latitude_delta_half_f64x4, _mm256_mul_pd(cos_latitude_product_f64x4, sin_sq_longitude_delta_half_f64x4));
    // Clamp haversine_term_f64x4 to [0, 1] to prevent NaN from sqrt of negative values
    __m256d zero_f64x4 = _mm256_setzero_pd();
    haversine_term_f64x4 = _mm256_max_pd(zero_f64x4, _mm256_min_pd(one_f64x4, haversine_term_f64x4));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    __m256d sqrt_haversine_f64x4 = _mm256_sqrt_pd(haversine_term_f64x4);
    __m256d sqrt_complement_f64x4 = _mm256_sqrt_pd(_mm256_sub_pd(one_f64x4, haversine_term_f64x4));
    __m256d central_angle_f64x4 = _mm256_mul_pd(two_f64x4,
                                                nk_atan2_f64x4_haswell_(sqrt_haversine_f64x4, sqrt_complement_f64x4));

    return _mm256_mul_pd(earth_radius_f64x4, central_angle_f64x4);
}

NK_API_COMPTIME void nk_haversine_f64_haswell(      //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 4) {
        __m256d first_latitudes_f64x4 = _mm256_loadu_pd(a_lats);
        __m256d first_longitudes_f64x4 = _mm256_loadu_pd(a_lons);
        __m256d second_latitudes_f64x4 = _mm256_loadu_pd(b_lats);
        __m256d second_longitudes_f64x4 = _mm256_loadu_pd(b_lons);

        __m256d distances_f64x4 = nk_haversine_f64x4_haswell_(first_latitudes_f64x4, first_longitudes_f64x4,
                                                              second_latitudes_f64x4, second_longitudes_f64x4);
        _mm256_storeu_pd(results, distances_f64x4);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with partial loads (n can be 1-3 here)
    if (n > 0) {
        nk_b256_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x4_haswell_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x4_haswell_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x4_haswell_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x4_haswell_(b_lons, &b_lon_vec, n);
        __m256d distances_f64x4 = nk_haversine_f64x4_haswell_(a_lat_vec.ymm_pd, a_lon_vec.ymm_pd, b_lat_vec.ymm_pd,
                                                              b_lon_vec.ymm_pd);
        result_vec.ymm_pd = distances_f64x4;
        nk_partial_store_b64x4_haswell_(&result_vec, results, n);
    }
}

NK_HELPER_INLINE __m256 nk_haversine_f32x8_haswell_(             //
    __m256 first_latitudes_f32x8, __m256 first_longitudes_f32x8, //
    __m256 second_latitudes_f32x8, __m256 second_longitudes_f32x8) {

    __m256 const earth_radius_f32x8 = _mm256_set1_ps((float)NK_EARTH_MEDIATORIAL_RADIUS);
    __m256 const half_f32x8 = _mm256_set1_ps(0.5f);
    __m256 const one_f32x8 = _mm256_set1_ps(1.0f);
    __m256 const two_f32x8 = _mm256_set1_ps(2.0f);

    __m256 latitude_delta_f32x8 = _mm256_sub_ps(second_latitudes_f32x8, first_latitudes_f32x8);
    __m256 longitude_delta_f32x8 = _mm256_sub_ps(second_longitudes_f32x8, first_longitudes_f32x8);

    // Haversine terms: sin²(Δ/2)
    __m256 latitude_delta_half_f32x8 = _mm256_mul_ps(latitude_delta_f32x8, half_f32x8);
    __m256 longitude_delta_half_f32x8 = _mm256_mul_ps(longitude_delta_f32x8, half_f32x8);
    __m256 sin_latitude_delta_half_f32x8 = nk_sin_f32x8_haswell_(latitude_delta_half_f32x8);
    __m256 sin_longitude_delta_half_f32x8 = nk_sin_f32x8_haswell_(longitude_delta_half_f32x8);
    __m256 sin_sq_latitude_delta_half_f32x8 = _mm256_mul_ps(sin_latitude_delta_half_f32x8,
                                                            sin_latitude_delta_half_f32x8);
    __m256 sin_sq_longitude_delta_half_f32x8 = _mm256_mul_ps(sin_longitude_delta_half_f32x8,
                                                             sin_longitude_delta_half_f32x8);

    // Latitude cosine product
    __m256 cos_first_latitude_f32x8 = nk_cos_f32x8_haswell_(first_latitudes_f32x8);
    __m256 cos_second_latitude_f32x8 = nk_cos_f32x8_haswell_(second_latitudes_f32x8);
    __m256 cos_latitude_product_f32x8 = _mm256_mul_ps(cos_first_latitude_f32x8, cos_second_latitude_f32x8);

    // a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
    __m256 haversine_term_f32x8 = _mm256_add_ps(
        sin_sq_latitude_delta_half_f32x8, _mm256_mul_ps(cos_latitude_product_f32x8, sin_sq_longitude_delta_half_f32x8));

    // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
    __m256 zero_f32x8 = _mm256_setzero_ps();
    haversine_term_f32x8 = _mm256_max_ps(zero_f32x8, _mm256_min_ps(one_f32x8, haversine_term_f32x8));

    // Central angle: c = 2 × atan2(√a, √(1-a))
    __m256 sqrt_haversine_f32x8 = _mm256_sqrt_ps(haversine_term_f32x8);
    __m256 sqrt_complement_f32x8 = _mm256_sqrt_ps(_mm256_sub_ps(one_f32x8, haversine_term_f32x8));
    __m256 central_angle_f32x8 = _mm256_mul_ps(two_f32x8,
                                               nk_atan2_f32x8_haswell_(sqrt_haversine_f32x8, sqrt_complement_f32x8));

    return _mm256_mul_ps(earth_radius_f32x8, central_angle_f32x8);
}

NK_API_COMPTIME void nk_haversine_f32_haswell(      //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 8) {
        __m256 first_latitudes_f32x8 = _mm256_loadu_ps(a_lats);
        __m256 first_longitudes_f32x8 = _mm256_loadu_ps(a_lons);
        __m256 second_latitudes_f32x8 = _mm256_loadu_ps(b_lats);
        __m256 second_longitudes_f32x8 = _mm256_loadu_ps(b_lons);

        __m256 distances_f32x8 = nk_haversine_f32x8_haswell_(first_latitudes_f32x8, first_longitudes_f32x8,
                                                             second_latitudes_f32x8, second_longitudes_f32x8);
        _mm256_storeu_ps(results, distances_f32x8);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with partial loads (n can be 1-7 here)
    if (n > 0) {
        nk_b256_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x8_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x8_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x8_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x8_serial_(b_lons, &b_lon_vec, n);
        __m256 distances_f32x8 = nk_haversine_f32x8_haswell_(a_lat_vec.ymm_ps, a_lon_vec.ymm_ps, b_lat_vec.ymm_ps,
                                                             b_lon_vec.ymm_ps);
        result_vec.ymm_ps = distances_f32x8;
        nk_partial_store_b32x8_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  AVX2 helper for Vincenty's geodesic distance on 4 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_HELPER_INLINE __m256d nk_vincenty_f64x4_haswell_(               //
    __m256d first_latitudes_f64x4, __m256d first_longitudes_f64x4, //
    __m256d second_latitudes_f64x4, __m256d second_longitudes_f64x4) {

    __m256d const equatorial_radius_f64x4 = _mm256_set1_pd(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m256d const polar_radius_f64x4 = _mm256_set1_pd(NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m256d const flattening_f64x4 = _mm256_set1_pd(1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m256d const convergence_threshold_f64x4 = _mm256_set1_pd(NK_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    __m256d const one_f64x4 = _mm256_set1_pd(1.0);
    __m256d const two_f64x4 = _mm256_set1_pd(2.0);
    __m256d const three_f64x4 = _mm256_set1_pd(3.0);
    __m256d const four_f64x4 = _mm256_set1_pd(4.0);
    __m256d const six_f64x4 = _mm256_set1_pd(6.0);
    __m256d const sixteen_f64x4 = _mm256_set1_pd(16.0);
    __m256d const epsilon_f64x4 = _mm256_set1_pd(1e-15);

    // Longitude difference
    __m256d longitude_diff_f64x4 = _mm256_sub_pd(second_longitudes_f64x4, first_longitudes_f64x4);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    __m256d one_minus_f64x4 = _mm256_sub_pd(one_f64x4, flattening_f64x4);
    __m256d tan_first_f64x4 = _mm256_div_pd(nk_sin_f64x4_haswell_(first_latitudes_f64x4),
                                            nk_cos_f64x4_haswell_(first_latitudes_f64x4));
    __m256d tan_second_f64x4 = _mm256_div_pd(nk_sin_f64x4_haswell_(second_latitudes_f64x4),
                                             nk_cos_f64x4_haswell_(second_latitudes_f64x4));
    __m256d tan_reduced_first_f64x4 = _mm256_mul_pd(one_minus_f64x4, tan_first_f64x4);
    __m256d tan_reduced_second_f64x4 = _mm256_mul_pd(one_minus_f64x4, tan_second_f64x4);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    __m256d cos_reduced_first_f64x4 = _mm256_div_pd(
        one_f64x4, _mm256_sqrt_pd(_mm256_fmadd_pd(tan_reduced_first_f64x4, tan_reduced_first_f64x4, one_f64x4)));
    __m256d sin_reduced_first_f64x4 = _mm256_mul_pd(tan_reduced_first_f64x4, cos_reduced_first_f64x4);
    __m256d cos_reduced_second_f64x4 = _mm256_div_pd(
        one_f64x4, _mm256_sqrt_pd(_mm256_fmadd_pd(tan_reduced_second_f64x4, tan_reduced_second_f64x4, one_f64x4)));
    __m256d sin_reduced_second_f64x4 = _mm256_mul_pd(tan_reduced_second_f64x4, cos_reduced_second_f64x4);

    // Initialize lambda_f64x4 and tracking variables
    __m256d lambda_f64x4 = longitude_diff_f64x4;
    __m256d sin_angular_distance_f64x4, cos_angular_distance_f64x4, angular_distance_f64x4;
    __m256d sin_azimuth_f64x4, cos_sq_azimuth_f64x4, cos_double_angular_midpoint_f64x4;

    // Track convergence and coincident points using masks
    __m256d converged_mask_f64x4 = _mm256_setzero_pd();
    __m256d coincident_mask_f64x4 = _mm256_setzero_pd();

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        int converged_bits = _mm256_movemask_pd(converged_mask_f64x4);
        if (converged_bits == 0xF) break;

        __m256d sin_lambda_f64x4 = nk_sin_f64x4_haswell_(lambda_f64x4);
        __m256d cos_lambda_f64x4 = nk_cos_f64x4_haswell_(lambda_f64x4);

        // sin²(angular_distance_f64x4) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        __m256d cross_term_f64x4 = _mm256_mul_pd(cos_reduced_second_f64x4, sin_lambda_f64x4);
        __m256d mixed_term_f64x4 = _mm256_sub_pd(
            _mm256_mul_pd(cos_reduced_first_f64x4, sin_reduced_second_f64x4),
            _mm256_mul_pd(_mm256_mul_pd(sin_reduced_first_f64x4, cos_reduced_second_f64x4), cos_lambda_f64x4));
        __m256d sin_angular_dist_sq_f64x4 = _mm256_fmadd_pd(cross_term_f64x4, cross_term_f64x4,
                                                            _mm256_mul_pd(mixed_term_f64x4, mixed_term_f64x4));
        sin_angular_distance_f64x4 = _mm256_sqrt_pd(sin_angular_dist_sq_f64x4);

        // Check for coincident points (sin_angular_distance_f64x4 ≈ 0)
        coincident_mask_f64x4 = _mm256_cmp_pd(sin_angular_distance_f64x4, epsilon_f64x4, _CMP_LT_OS);

        // cos(angular_distance_f64x4) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance_f64x4 = _mm256_fmadd_pd(_mm256_mul_pd(cos_reduced_first_f64x4, cos_reduced_second_f64x4),
                                                     cos_lambda_f64x4,
                                                     _mm256_mul_pd(sin_reduced_first_f64x4, sin_reduced_second_f64x4));

        // angular_distance_f64x4 = atan2(sin, cos)
        angular_distance_f64x4 = nk_atan2_f64x4_haswell_(sin_angular_distance_f64x4, cos_angular_distance_f64x4);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance_f64x4)
        // Avoid division by zero by using blending
        __m256d safe_sin_angular_f64x4 = _mm256_blendv_pd(sin_angular_distance_f64x4, one_f64x4, coincident_mask_f64x4);
        sin_azimuth_f64x4 = _mm256_div_pd(
            _mm256_mul_pd(_mm256_mul_pd(cos_reduced_first_f64x4, cos_reduced_second_f64x4), sin_lambda_f64x4),
            safe_sin_angular_f64x4);
        cos_sq_azimuth_f64x4 = _mm256_sub_pd(one_f64x4, _mm256_mul_pd(sin_azimuth_f64x4, sin_azimuth_f64x4));

        // Handle equatorial case: cos²α ≈ 0
        __m256d equatorial_mask_f64x4 = _mm256_cmp_pd(cos_sq_azimuth_f64x4, epsilon_f64x4, _CMP_LT_OS);
        __m256d safe_cos_sq_azimuth_f64x4 = _mm256_blendv_pd(cos_sq_azimuth_f64x4, one_f64x4, equatorial_mask_f64x4);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        __m256d sin_product_f64x4 = _mm256_mul_pd(sin_reduced_first_f64x4, sin_reduced_second_f64x4);
        cos_double_angular_midpoint_f64x4 = _mm256_sub_pd(
            cos_angular_distance_f64x4,
            _mm256_div_pd(_mm256_mul_pd(two_f64x4, sin_product_f64x4), safe_cos_sq_azimuth_f64x4));
        cos_double_angular_midpoint_f64x4 = _mm256_blendv_pd(cos_double_angular_midpoint_f64x4, _mm256_setzero_pd(),
                                                             equatorial_mask_f64x4);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        __m256d correction_factor_f64x4 = _mm256_mul_pd(
            _mm256_div_pd(flattening_f64x4, sixteen_f64x4),
            _mm256_mul_pd(
                cos_sq_azimuth_f64x4,
                _mm256_fmadd_pd(flattening_f64x4, _mm256_fnmadd_pd(three_f64x4, cos_sq_azimuth_f64x4, four_f64x4),
                                four_f64x4)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        __m256d cos_2sm_sq_f64x4 = _mm256_mul_pd(cos_double_angular_midpoint_f64x4, cos_double_angular_midpoint_f64x4);
        // innermost_f64x4 = -1 + 2 × cos²(2σₘ)
        __m256d innermost_f64x4 = _mm256_fmadd_pd(two_f64x4, cos_2sm_sq_f64x4, _mm256_set1_pd(-1.0));
        // middle_f64x4 = cos(2σₘ) + C × cos(σ) × innermost_f64x4
        __m256d middle_f64x4 = _mm256_fmadd_pd(_mm256_mul_pd(correction_factor_f64x4, cos_angular_distance_f64x4),
                                               innermost_f64x4, cos_double_angular_midpoint_f64x4);
        // inner_f64x4 = C × sin(σ) × middle_f64x4
        __m256d inner_f64x4 = _mm256_mul_pd(_mm256_mul_pd(correction_factor_f64x4, sin_angular_distance_f64x4),
                                            middle_f64x4);

        // λ' = L + (1-C) * f * sin_α * (σ + inner_f64x4)
        __m256d lambda_new_f64x4 = _mm256_fmadd_pd(
            _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(one_f64x4, correction_factor_f64x4), flattening_f64x4),
                          sin_azimuth_f64x4),
            _mm256_add_pd(angular_distance_f64x4, inner_f64x4), longitude_diff_f64x4);

        // Check convergence: |λ - λ'| < threshold
        __m256d lambda_diff_abs_f64x4 = _mm256_andnot_pd(_mm256_set1_pd(-0.0),
                                                         _mm256_sub_pd(lambda_new_f64x4, lambda_f64x4));
        __m256d newly_converged_f64x4 = _mm256_cmp_pd(lambda_diff_abs_f64x4, convergence_threshold_f64x4, _CMP_LT_OS);
        converged_mask_f64x4 = _mm256_or_pd(converged_mask_f64x4, newly_converged_f64x4);

        // Only update lambda_f64x4 for non-converged lanes
        lambda_f64x4 = _mm256_blendv_pd(lambda_new_f64x4, lambda_f64x4, converged_mask_f64x4);
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    __m256d a_sq_f64x4 = _mm256_mul_pd(equatorial_radius_f64x4, equatorial_radius_f64x4);
    __m256d b_sq_f64x4 = _mm256_mul_pd(polar_radius_f64x4, polar_radius_f64x4);
    __m256d u_sq_f64x4 = _mm256_div_pd(_mm256_mul_pd(cos_sq_azimuth_f64x4, _mm256_sub_pd(a_sq_f64x4, b_sq_f64x4)),
                                       b_sq_f64x4);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    __m256d series_a_f64x4 = _mm256_fmadd_pd(u_sq_f64x4, _mm256_set1_pd(-175.0), _mm256_set1_pd(320.0));
    series_a_f64x4 = _mm256_fmadd_pd(u_sq_f64x4, series_a_f64x4, _mm256_set1_pd(-768.0));
    series_a_f64x4 = _mm256_fmadd_pd(u_sq_f64x4, series_a_f64x4, _mm256_set1_pd(4096.0));
    series_a_f64x4 = _mm256_fmadd_pd(_mm256_div_pd(u_sq_f64x4, _mm256_set1_pd(16384.0)), series_a_f64x4, one_f64x4);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    __m256d series_b_f64x4 = _mm256_fmadd_pd(u_sq_f64x4, _mm256_set1_pd(-47.0), _mm256_set1_pd(74.0));
    series_b_f64x4 = _mm256_fmadd_pd(u_sq_f64x4, series_b_f64x4, _mm256_set1_pd(-128.0));
    series_b_f64x4 = _mm256_fmadd_pd(u_sq_f64x4, series_b_f64x4, _mm256_set1_pd(256.0));
    series_b_f64x4 = _mm256_mul_pd(_mm256_div_pd(u_sq_f64x4, _mm256_set1_pd(1024.0)), series_b_f64x4);

    // Δσ = B × sin(σ) × (cos(2σₘ) +
    //      B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m256d cos_2sm_sq_f64x4 = _mm256_mul_pd(cos_double_angular_midpoint_f64x4, cos_double_angular_midpoint_f64x4);
    __m256d sin_sq_f64x4 = _mm256_mul_pd(sin_angular_distance_f64x4, sin_angular_distance_f64x4);
    __m256d term1_f64x4 = _mm256_fmadd_pd(two_f64x4, cos_2sm_sq_f64x4, _mm256_set1_pd(-1.0));
    term1_f64x4 = _mm256_mul_pd(cos_angular_distance_f64x4, term1_f64x4);
    __m256d term2_f64x4 = _mm256_fmadd_pd(four_f64x4, sin_sq_f64x4, _mm256_set1_pd(-3.0));
    __m256d term3_f64x4 = _mm256_fmadd_pd(four_f64x4, cos_2sm_sq_f64x4, _mm256_set1_pd(-3.0));
    term2_f64x4 = _mm256_mul_pd(
        _mm256_mul_pd(_mm256_div_pd(series_b_f64x4, six_f64x4), cos_double_angular_midpoint_f64x4),
        _mm256_mul_pd(term2_f64x4, term3_f64x4));
    __m256d delta_sigma_f64x4 = _mm256_mul_pd(
        series_b_f64x4, _mm256_mul_pd(sin_angular_distance_f64x4,
                                      _mm256_add_pd(cos_double_angular_midpoint_f64x4,
                                                    _mm256_mul_pd(_mm256_div_pd(series_b_f64x4, four_f64x4),
                                                                  _mm256_sub_pd(term1_f64x4, term2_f64x4)))));

    // s = b * A * (σ - Δσ)
    __m256d distances_f64x4 = _mm256_mul_pd(_mm256_mul_pd(polar_radius_f64x4, series_a_f64x4),
                                            _mm256_sub_pd(angular_distance_f64x4, delta_sigma_f64x4));

    // Set coincident points to zero
    distances_f64x4 = _mm256_blendv_pd(distances_f64x4, _mm256_setzero_pd(), coincident_mask_f64x4);

    return distances_f64x4;
}

NK_API_COMPTIME void nk_vincenty_f64_haswell(       //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 4) {
        __m256d first_latitudes_f64x4 = _mm256_loadu_pd(a_lats);
        __m256d first_longitudes_f64x4 = _mm256_loadu_pd(a_lons);
        __m256d second_latitudes_f64x4 = _mm256_loadu_pd(b_lats);
        __m256d second_longitudes_f64x4 = _mm256_loadu_pd(b_lons);

        __m256d distances_f64x4 = nk_vincenty_f64x4_haswell_(first_latitudes_f64x4, first_longitudes_f64x4,
                                                             second_latitudes_f64x4, second_longitudes_f64x4);
        _mm256_storeu_pd(results, distances_f64x4);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with partial loads (n can be 1-3 here)
    if (n > 0) {
        nk_b256_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x4_haswell_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x4_haswell_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x4_haswell_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x4_haswell_(b_lons, &b_lon_vec, n);
        __m256d distances_f64x4 = nk_vincenty_f64x4_haswell_(a_lat_vec.ymm_pd, a_lon_vec.ymm_pd, b_lat_vec.ymm_pd,
                                                             b_lon_vec.ymm_pd);
        result_vec.ymm_pd = distances_f64x4;
        nk_partial_store_b64x4_haswell_(&result_vec, results, n);
    }
}

/**
 *  @brief  AVX2 helper for Vincenty's geodesic distance on 8 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_HELPER_INLINE __m256 nk_vincenty_f32x8_haswell_(              //
    __m256 first_latitudes_f32x8, __m256 first_longitudes_f32x8, //
    __m256 second_latitudes_f32x8, __m256 second_longitudes_f32x8) {

    __m256 const equatorial_radius_f32x8 = _mm256_set1_ps((float)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    __m256 const polar_radius_f32x8 = _mm256_set1_ps((float)NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    __m256 const flattening_f32x8 = _mm256_set1_ps(1.0f / (float)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    __m256 const convergence_threshold_f32x8 = _mm256_set1_ps(NK_VINCENTY_CONVERGENCE_THRESHOLD_F32);
    __m256 const one_f32x8 = _mm256_set1_ps(1.0f);
    __m256 const two_f32x8 = _mm256_set1_ps(2.0f);
    __m256 const three_f32x8 = _mm256_set1_ps(3.0f);
    __m256 const four_f32x8 = _mm256_set1_ps(4.0f);
    __m256 const six_f32x8 = _mm256_set1_ps(6.0f);
    __m256 const sixteen_f32x8 = _mm256_set1_ps(16.0f);
    __m256 const epsilon_f32x8 = _mm256_set1_ps(1e-7f);

    // Longitude difference
    __m256 longitude_diff_f32x8 = _mm256_sub_ps(second_longitudes_f32x8, first_longitudes_f32x8);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    __m256 one_minus_f32x8 = _mm256_sub_ps(one_f32x8, flattening_f32x8);
    __m256 tan_first_f32x8 = _mm256_div_ps(nk_sin_f32x8_haswell_(first_latitudes_f32x8),
                                           nk_cos_f32x8_haswell_(first_latitudes_f32x8));
    __m256 tan_second_f32x8 = _mm256_div_ps(nk_sin_f32x8_haswell_(second_latitudes_f32x8),
                                            nk_cos_f32x8_haswell_(second_latitudes_f32x8));
    __m256 tan_reduced_first_f32x8 = _mm256_mul_ps(one_minus_f32x8, tan_first_f32x8);
    __m256 tan_reduced_second_f32x8 = _mm256_mul_ps(one_minus_f32x8, tan_second_f32x8);

    // cos(U) = 1/√(1 + tan²(U)), sin(U) = tan(U) × cos(U)
    __m256 cos_reduced_first_f32x8 = _mm256_div_ps(
        one_f32x8, _mm256_sqrt_ps(_mm256_fmadd_ps(tan_reduced_first_f32x8, tan_reduced_first_f32x8, one_f32x8)));
    __m256 sin_reduced_first_f32x8 = _mm256_mul_ps(tan_reduced_first_f32x8, cos_reduced_first_f32x8);
    __m256 cos_reduced_second_f32x8 = _mm256_div_ps(
        one_f32x8, _mm256_sqrt_ps(_mm256_fmadd_ps(tan_reduced_second_f32x8, tan_reduced_second_f32x8, one_f32x8)));
    __m256 sin_reduced_second_f32x8 = _mm256_mul_ps(tan_reduced_second_f32x8, cos_reduced_second_f32x8);

    // Initialize lambda_f32x8 and tracking variables
    __m256 lambda_f32x8 = longitude_diff_f32x8;
    __m256 sin_angular_distance_f32x8, cos_angular_distance_f32x8, angular_distance_f32x8;
    __m256 sin_azimuth_f32x8, cos_sq_azimuth_f32x8, cos_double_angular_midpoint_f32x8;

    // Track convergence and coincident points using masks
    __m256 converged_mask_f32x8 = _mm256_setzero_ps();
    __m256 coincident_mask_f32x8 = _mm256_setzero_ps();

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        int converged_bits = _mm256_movemask_ps(converged_mask_f32x8);
        if (converged_bits == 0xFF) break;

        __m256 sin_lambda_f32x8 = nk_sin_f32x8_haswell_(lambda_f32x8);
        __m256 cos_lambda_f32x8 = nk_cos_f32x8_haswell_(lambda_f32x8);

        // sin²(angular_distance_f32x8) = (cos(U₂) × sin(λ))² + (cos(U₁) × sin(U₂) - sin(U₁) × cos(U₂) × cos(λ))²
        __m256 cross_term_f32x8 = _mm256_mul_ps(cos_reduced_second_f32x8, sin_lambda_f32x8);
        __m256 mixed_term_f32x8 = _mm256_sub_ps(
            _mm256_mul_ps(cos_reduced_first_f32x8, sin_reduced_second_f32x8),
            _mm256_mul_ps(_mm256_mul_ps(sin_reduced_first_f32x8, cos_reduced_second_f32x8), cos_lambda_f32x8));
        __m256 sin_angular_dist_sq_f32x8 = _mm256_fmadd_ps(cross_term_f32x8, cross_term_f32x8,
                                                           _mm256_mul_ps(mixed_term_f32x8, mixed_term_f32x8));
        sin_angular_distance_f32x8 = _mm256_sqrt_ps(sin_angular_dist_sq_f32x8);

        // Check for coincident points (sin_angular_distance_f32x8 ≈ 0)
        coincident_mask_f32x8 = _mm256_cmp_ps(sin_angular_distance_f32x8, epsilon_f32x8, _CMP_LT_OS);

        // cos(angular_distance_f32x8) = sin(U₁) × sin(U₂) + cos(U₁) × cos(U₂) × cos(λ)
        cos_angular_distance_f32x8 = _mm256_fmadd_ps(_mm256_mul_ps(cos_reduced_first_f32x8, cos_reduced_second_f32x8),
                                                     cos_lambda_f32x8,
                                                     _mm256_mul_ps(sin_reduced_first_f32x8, sin_reduced_second_f32x8));

        // angular_distance_f32x8 = atan2(sin, cos)
        angular_distance_f32x8 = nk_atan2_f32x8_haswell_(sin_angular_distance_f32x8, cos_angular_distance_f32x8);

        // sin(azimuth) = cos(U₁) × cos(U₂) × sin(λ) / sin(angular_distance_f32x8)
        // Avoid division by zero by using blending
        __m256 safe_sin_angular_f32x8 = _mm256_blendv_ps(sin_angular_distance_f32x8, one_f32x8, coincident_mask_f32x8);
        sin_azimuth_f32x8 = _mm256_div_ps(
            _mm256_mul_ps(_mm256_mul_ps(cos_reduced_first_f32x8, cos_reduced_second_f32x8), sin_lambda_f32x8),
            safe_sin_angular_f32x8);
        cos_sq_azimuth_f32x8 = _mm256_sub_ps(one_f32x8, _mm256_mul_ps(sin_azimuth_f32x8, sin_azimuth_f32x8));

        // Handle equatorial case: cos²α ≈ 0
        __m256 equatorial_mask_f32x8 = _mm256_cmp_ps(cos_sq_azimuth_f32x8, epsilon_f32x8, _CMP_LT_OS);
        __m256 safe_cos_sq_azimuth_f32x8 = _mm256_blendv_ps(cos_sq_azimuth_f32x8, one_f32x8, equatorial_mask_f32x8);

        // cos(2σₘ) = cos(σ) - 2 × sin(U₁) × sin(U₂) / cos²(α)
        __m256 sin_product_f32x8 = _mm256_mul_ps(sin_reduced_first_f32x8, sin_reduced_second_f32x8);
        cos_double_angular_midpoint_f32x8 = _mm256_sub_ps(
            cos_angular_distance_f32x8,
            _mm256_div_ps(_mm256_mul_ps(two_f32x8, sin_product_f32x8), safe_cos_sq_azimuth_f32x8));
        cos_double_angular_midpoint_f32x8 = _mm256_blendv_ps(cos_double_angular_midpoint_f32x8, _mm256_setzero_ps(),
                                                             equatorial_mask_f32x8);

        // C = f/16 * cos²α * (4 + f*(4 - 3*cos²α))
        __m256 correction_factor_f32x8 = _mm256_mul_ps(
            _mm256_div_ps(flattening_f32x8, sixteen_f32x8),
            _mm256_mul_ps(
                cos_sq_azimuth_f32x8,
                _mm256_fmadd_ps(flattening_f32x8, _mm256_fnmadd_ps(three_f32x8, cos_sq_azimuth_f32x8, four_f32x8),
                                four_f32x8)));

        // λ' = L + (1-C) × f × sin(α) × (σ + C × sin(σ) × (cos(2σₘ) + C × cos(σ) × (-1 + 2 × cos²(2σₘ))))
        __m256 cos_2sm_sq_f32x8 = _mm256_mul_ps(cos_double_angular_midpoint_f32x8, cos_double_angular_midpoint_f32x8);
        // innermost_f32x8 = -1 + 2 × cos²(2σₘ)
        __m256 innermost_f32x8 = _mm256_fmadd_ps(two_f32x8, cos_2sm_sq_f32x8, _mm256_set1_ps(-1.0f));
        // middle_f32x8 = cos(2σₘ) + C × cos(σ) × innermost_f32x8
        __m256 middle_f32x8 = _mm256_fmadd_ps(_mm256_mul_ps(correction_factor_f32x8, cos_angular_distance_f32x8),
                                              innermost_f32x8, cos_double_angular_midpoint_f32x8);
        // inner_f32x8 = C × sin(σ) × middle_f32x8
        __m256 inner_f32x8 = _mm256_mul_ps(_mm256_mul_ps(correction_factor_f32x8, sin_angular_distance_f32x8),
                                           middle_f32x8);

        // λ' = L + (1-C) * f * sin_α * (σ + inner_f32x8)
        __m256 lambda_new_f32x8 = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(one_f32x8, correction_factor_f32x8), flattening_f32x8),
                          sin_azimuth_f32x8),
            _mm256_add_ps(angular_distance_f32x8, inner_f32x8), longitude_diff_f32x8);

        // Check convergence: |λ - λ'| < threshold
        __m256 lambda_diff_abs_f32x8 = _mm256_andnot_ps(_mm256_set1_ps(-0.0f),
                                                        _mm256_sub_ps(lambda_new_f32x8, lambda_f32x8));
        __m256 newly_converged_f32x8 = _mm256_cmp_ps(lambda_diff_abs_f32x8, convergence_threshold_f32x8, _CMP_LT_OS);
        converged_mask_f32x8 = _mm256_or_ps(converged_mask_f32x8, newly_converged_f32x8);

        // Only update lambda_f32x8 for non-converged lanes
        lambda_f32x8 = _mm256_blendv_ps(lambda_new_f32x8, lambda_f32x8, converged_mask_f32x8);
    }

    // Final distance calculation
    // u² = cos²α * (a² - b²) / b²
    __m256 a_sq_f32x8 = _mm256_mul_ps(equatorial_radius_f32x8, equatorial_radius_f32x8);
    __m256 b_sq_f32x8 = _mm256_mul_ps(polar_radius_f32x8, polar_radius_f32x8);
    __m256 u_sq_f32x8 = _mm256_div_ps(_mm256_mul_ps(cos_sq_azimuth_f32x8, _mm256_sub_ps(a_sq_f32x8, b_sq_f32x8)),
                                      b_sq_f32x8);

    // A = 1 + u²/16384 * (4096 + u²*(-768 + u²*(320 - 175*u²)))
    __m256 series_a_f32x8 = _mm256_fmadd_ps(u_sq_f32x8, _mm256_set1_ps(-175.0f), _mm256_set1_ps(320.0f));
    series_a_f32x8 = _mm256_fmadd_ps(u_sq_f32x8, series_a_f32x8, _mm256_set1_ps(-768.0f));
    series_a_f32x8 = _mm256_fmadd_ps(u_sq_f32x8, series_a_f32x8, _mm256_set1_ps(4096.0f));
    series_a_f32x8 = _mm256_fmadd_ps(_mm256_div_ps(u_sq_f32x8, _mm256_set1_ps(16384.0f)), series_a_f32x8, one_f32x8);

    // B = u²/1024 * (256 + u²*(-128 + u²*(74 - 47*u²)))
    __m256 series_b_f32x8 = _mm256_fmadd_ps(u_sq_f32x8, _mm256_set1_ps(-47.0f), _mm256_set1_ps(74.0f));
    series_b_f32x8 = _mm256_fmadd_ps(u_sq_f32x8, series_b_f32x8, _mm256_set1_ps(-128.0f));
    series_b_f32x8 = _mm256_fmadd_ps(u_sq_f32x8, series_b_f32x8, _mm256_set1_ps(256.0f));
    series_b_f32x8 = _mm256_mul_ps(_mm256_div_ps(u_sq_f32x8, _mm256_set1_ps(1024.0f)), series_b_f32x8);

    // Δσ = B × sin(σ) × (cos(2σₘ) +
    //      B/4 × (cos(σ) × (-1 + 2 × cos²(2σₘ)) - B/6 × cos(2σₘ) × (-3 + 4 × sin²(σ)) × (-3 + 4 × cos²(2σₘ))))
    __m256 cos_2sm_sq_f32x8 = _mm256_mul_ps(cos_double_angular_midpoint_f32x8, cos_double_angular_midpoint_f32x8);
    __m256 sin_sq_f32x8 = _mm256_mul_ps(sin_angular_distance_f32x8, sin_angular_distance_f32x8);
    __m256 term1_f32x8 = _mm256_fmadd_ps(two_f32x8, cos_2sm_sq_f32x8, _mm256_set1_ps(-1.0f));
    term1_f32x8 = _mm256_mul_ps(cos_angular_distance_f32x8, term1_f32x8);
    __m256 term2_f32x8 = _mm256_fmadd_ps(four_f32x8, sin_sq_f32x8, _mm256_set1_ps(-3.0f));
    __m256 term3_f32x8 = _mm256_fmadd_ps(four_f32x8, cos_2sm_sq_f32x8, _mm256_set1_ps(-3.0f));
    term2_f32x8 = _mm256_mul_ps(
        _mm256_mul_ps(_mm256_div_ps(series_b_f32x8, six_f32x8), cos_double_angular_midpoint_f32x8),
        _mm256_mul_ps(term2_f32x8, term3_f32x8));
    __m256 delta_sigma_f32x8 = _mm256_mul_ps(
        series_b_f32x8, _mm256_mul_ps(sin_angular_distance_f32x8,
                                      _mm256_add_ps(cos_double_angular_midpoint_f32x8,
                                                    _mm256_mul_ps(_mm256_div_ps(series_b_f32x8, four_f32x8),
                                                                  _mm256_sub_ps(term1_f32x8, term2_f32x8)))));

    // s = b * A * (σ - Δσ)
    __m256 distances_f32x8 = _mm256_mul_ps(_mm256_mul_ps(polar_radius_f32x8, series_a_f32x8),
                                           _mm256_sub_ps(angular_distance_f32x8, delta_sigma_f32x8));

    // Set coincident points to zero
    distances_f32x8 = _mm256_blendv_ps(distances_f32x8, _mm256_setzero_ps(), coincident_mask_f32x8);

    return distances_f32x8;
}

NK_API_COMPTIME void nk_vincenty_f32_haswell(       //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 8) {
        __m256 first_latitudes_f32x8 = _mm256_loadu_ps(a_lats);
        __m256 first_longitudes_f32x8 = _mm256_loadu_ps(a_lons);
        __m256 second_latitudes_f32x8 = _mm256_loadu_ps(b_lats);
        __m256 second_longitudes_f32x8 = _mm256_loadu_ps(b_lons);

        __m256 distances_f32x8 = nk_vincenty_f32x8_haswell_(first_latitudes_f32x8, first_longitudes_f32x8,
                                                            second_latitudes_f32x8, second_longitudes_f32x8);
        _mm256_storeu_ps(results, distances_f32x8);

        a_lats += 8, a_lons += 8, b_lats += 8, b_lons += 8, results += 8, n -= 8;
    }

    // Handle remaining elements with partial loads (n can be 1-7 here)
    if (n > 0) {
        nk_b256_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x8_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x8_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x8_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x8_serial_(b_lons, &b_lon_vec, n);
        __m256 distances_f32x8 = nk_vincenty_f32x8_haswell_(a_lat_vec.ymm_ps, a_lon_vec.ymm_ps, b_lat_vec.ymm_ps,
                                                            b_lon_vec.ymm_ps);
        result_vec.ymm_ps = distances_f32x8;
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
#endif // NK_TARGET_X8664_
#endif // NK_GEOSPATIAL_HASWELL_H
