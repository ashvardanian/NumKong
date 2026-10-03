/**
 *  @brief SIMD-accelerated Geospatial Distances for WASM.
 *  @file include/numkong/geospatial/v128relaxed.h
 *  @author Ash Vardanian
 *  @date February 6, 2026
 *
 *  @sa include/numkong/geospatial.h
 *
 *  Implements Haversine and Vincenty great-circle distances for f32x4 and f64x2.
 *  Haversine uses sin/cos/atan2 with min/max clamping to keep the atan2 argument in [0,1].
 *  Vincenty iterates sin/cos/atan2 until convergence, using `i8x16_all_true` to test whether
 *  all SIMD lanes have converged without per-lane extraction.
 *
 *  @section geospatial_wasm_instructions Key WASM SIMD Instructions (beyond trig)
 *
 *      Intrinsic                               Operation
 *      wasm_f32x4_sqrt(a)                      Square root (4-way f32)
 *      wasm_f64x2_sqrt(a)                      Square root (2-way f64)
 *      wasm_f32x4_div(a, b)                    Division (4-way f32)
 *      wasm_f64x2_div(a, b)                    Division (2-way f64)
 *      wasm_f32x4_min/max(a, b)                Clamping for Haversine
 *      wasm_f64x2_min/max(a, b)                Clamping for Haversine
 *      wasm_f32x4_relaxed_min/max(a, b)        Min/max without NaN fixup (1 vs 6-9 on x86)
 *      wasm_f64x2_relaxed_min/max(a, b)        Min/max without NaN fixup (1 vs 6-9 on x86)
 *      wasm_i32x4_relaxed_laneselect(a, b, m)  Lane select (1 instr vs 3 on x86)
 *      wasm_i64x2_relaxed_laneselect(a, b, m)  Lane select for f64 masks
 *      wasm_i8x16_all_true(a)                  Vincenty convergence check (all lanes at once)
 */
#ifndef NK_GEOSPATIAL_V128RELAXED_H
#define NK_GEOSPATIAL_V128RELAXED_H

#if NK_TARGET_V128RELAXED

#include "numkong/types.h"
#include "numkong/trigonometry/v128relaxed.h"

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/*  WASM Relaxed SIMD implementations using 2-wide f64 and 4-wide f32 SIMD.
 *  These require WASM trigonometric kernels from trigonometry/v128relaxed.h.
 */

NK_INTERNAL v128_t nk_haversine_f64x2_v128relaxed_(  //
    v128_t first_latitudes, v128_t first_longitudes, //
    v128_t second_latitudes, v128_t second_longitudes) {

    v128_t const earth_radius = wasm_f64x2_splat(NK_EARTH_MEDIATORIAL_RADIUS);
    v128_t const half = wasm_f64x2_splat(0.5);
    v128_t const one = wasm_f64x2_splat(1.0);
    v128_t const two = wasm_f64x2_splat(2.0);

    v128_t latitude_delta = wasm_f64x2_sub(second_latitudes, first_latitudes);
    v128_t longitude_delta = wasm_f64x2_sub(second_longitudes, first_longitudes);

    // Haversine terms: sin^2(delta/2)
    v128_t latitude_delta_half = wasm_f64x2_mul(latitude_delta, half);
    v128_t longitude_delta_half = wasm_f64x2_mul(longitude_delta, half);
    v128_t sin_latitude_delta_half = nk_f64x2_sin_v128relaxed_(latitude_delta_half);
    v128_t sin_longitude_delta_half = nk_f64x2_sin_v128relaxed_(longitude_delta_half);
    v128_t sin_squared_latitude_delta_half = wasm_f64x2_mul(sin_latitude_delta_half, sin_latitude_delta_half);
    v128_t sin_squared_longitude_delta_half = wasm_f64x2_mul(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    v128_t cos_first_latitude = nk_f64x2_cos_v128relaxed_(first_latitudes);
    v128_t cos_second_latitude = nk_f64x2_cos_v128relaxed_(second_latitudes);
    v128_t cos_latitude_product = wasm_f64x2_mul(cos_first_latitude, cos_second_latitude);

    // a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    v128_t haversine_term = wasm_f64x2_add(sin_squared_latitude_delta_half,
                                           wasm_f64x2_mul(cos_latitude_product, sin_squared_longitude_delta_half));
    // Clamp haversine_term to [0, 1] to prevent NaN from sqrt of negative values
    // relaxed_min/max: 1 instruction (minpd/maxpd) vs 6-9 (with NaN/signed-zero fixup) on x86.
    // Safe because haversine_term is a product of finite sin/cos values — NaN is impossible.
    v128_t zero = wasm_f64x2_splat(0.0);
    haversine_term = wasm_f64x2_relaxed_max(zero, wasm_f64x2_relaxed_min(one, haversine_term));

    // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
    v128_t sqrt_haversine = wasm_f64x2_sqrt(haversine_term);
    v128_t sqrt_complement = wasm_f64x2_sqrt(wasm_f64x2_sub(one, haversine_term));
    v128_t central_angle = wasm_f64x2_mul(two, nk_f64x2_atan2_v128relaxed_(sqrt_haversine, sqrt_complement));

    return wasm_f64x2_mul(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f64_v128relaxed(        //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 2) {
        v128_t first_latitudes = wasm_v128_load(a_lats);
        v128_t first_longitudes = wasm_v128_load(a_lons);
        v128_t second_latitudes = wasm_v128_load(b_lats);
        v128_t second_longitudes = wasm_v128_load(b_lons);

        v128_t distances = nk_haversine_f64x2_v128relaxed_(first_latitudes, first_longitudes, second_latitudes,
                                                           second_longitudes);
        wasm_v128_store(results, distances);

        a_lats += 2, a_lons += 2, b_lats += 2, b_lons += 2, results += 2, n -= 2;
    }

    // Handle tail with partial loads (n can only be 0 or 1 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x2_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x2_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x2_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x2_serial_(b_lons, &b_lon_vec, n);
        v128_t distances = nk_haversine_f64x2_v128relaxed_(a_lat_vec.v128, a_lon_vec.v128, b_lat_vec.v128,
                                                           b_lon_vec.v128);
        result_vec.v128 = distances;
        nk_partial_store_b64x2_serial_(&result_vec, results, n);
    }
}

NK_INTERNAL v128_t nk_haversine_f32x4_v128relaxed_(  //
    v128_t first_latitudes, v128_t first_longitudes, //
    v128_t second_latitudes, v128_t second_longitudes) {

    v128_t const earth_radius = wasm_f32x4_splat((float)NK_EARTH_MEDIATORIAL_RADIUS);
    v128_t const half = wasm_f32x4_splat(0.5f);
    v128_t const one = wasm_f32x4_splat(1.0f);
    v128_t const two = wasm_f32x4_splat(2.0f);

    v128_t latitude_delta = wasm_f32x4_sub(second_latitudes, first_latitudes);
    v128_t longitude_delta = wasm_f32x4_sub(second_longitudes, first_longitudes);

    // Haversine terms: sin^2(delta/2)
    v128_t latitude_delta_half = wasm_f32x4_mul(latitude_delta, half);
    v128_t longitude_delta_half = wasm_f32x4_mul(longitude_delta, half);
    v128_t sin_latitude_delta_half = nk_f32x4_sin_v128relaxed_(latitude_delta_half);
    v128_t sin_longitude_delta_half = nk_f32x4_sin_v128relaxed_(longitude_delta_half);
    v128_t sin_squared_latitude_delta_half = wasm_f32x4_mul(sin_latitude_delta_half, sin_latitude_delta_half);
    v128_t sin_squared_longitude_delta_half = wasm_f32x4_mul(sin_longitude_delta_half, sin_longitude_delta_half);

    // Latitude cosine product
    v128_t cos_first_latitude = nk_f32x4_cos_v128relaxed_(first_latitudes);
    v128_t cos_second_latitude = nk_f32x4_cos_v128relaxed_(second_latitudes);
    v128_t cos_latitude_product = wasm_f32x4_mul(cos_first_latitude, cos_second_latitude);

    // a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    v128_t haversine_term = wasm_f32x4_add(sin_squared_latitude_delta_half,
                                           wasm_f32x4_mul(cos_latitude_product, sin_squared_longitude_delta_half));

    // Clamp to [0, 1] to avoid NaN from sqrt of negative numbers (due to floating point errors)
    // relaxed_min/max: 1 instruction (minps/maxps) vs 6-9 (with NaN/signed-zero fixup) on x86.
    // Safe because haversine_term is a product of finite sin/cos values — NaN is impossible.
    v128_t zero = wasm_f32x4_splat(0.0f);
    haversine_term = wasm_f32x4_relaxed_max(zero, wasm_f32x4_relaxed_min(one, haversine_term));

    // Central angle: c = 2 * atan2(sqrt(a), sqrt(1-a))
    v128_t sqrt_haversine = wasm_f32x4_sqrt(haversine_term);
    v128_t sqrt_complement = wasm_f32x4_sqrt(wasm_f32x4_sub(one, haversine_term));
    v128_t central_angle = wasm_f32x4_mul(two, nk_f32x4_atan2_v128relaxed_(sqrt_haversine, sqrt_complement));

    return wasm_f32x4_mul(earth_radius, central_angle);
}

NK_PUBLIC void nk_haversine_f32_v128relaxed(        //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 4) {
        v128_t first_latitudes = wasm_v128_load(a_lats);
        v128_t first_longitudes = wasm_v128_load(a_lons);
        v128_t second_latitudes = wasm_v128_load(b_lats);
        v128_t second_longitudes = wasm_v128_load(b_lons);

        v128_t distances = nk_haversine_f32x4_v128relaxed_(first_latitudes, first_longitudes, second_latitudes,
                                                           second_longitudes);
        wasm_v128_store(results, distances);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle tail with partial loads (n can be 0-3 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x4_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x4_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x4_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x4_serial_(b_lons, &b_lon_vec, n);
        v128_t distances = nk_haversine_f32x4_v128relaxed_(a_lat_vec.v128, a_lon_vec.v128, b_lat_vec.v128,
                                                           b_lon_vec.v128);
        result_vec.v128 = distances;
        nk_partial_store_b32x4_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  WASM Relaxed SIMD helper for Vincenty's geodesic distance on 2 f64 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL v128_t nk_vincenty_f64x2_v128relaxed_(   //
    v128_t first_latitudes, v128_t first_longitudes, //
    v128_t second_latitudes, v128_t second_longitudes) {

    v128_t const equatorial_radius = wasm_f64x2_splat(NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    v128_t const polar_radius = wasm_f64x2_splat(NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    v128_t const flattening = wasm_f64x2_splat(1.0 / NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    v128_t const convergence_threshold = wasm_f64x2_splat(NK_VINCENTY_CONVERGENCE_THRESHOLD_F64);
    v128_t const one = wasm_f64x2_splat(1.0);
    v128_t const two = wasm_f64x2_splat(2.0);
    v128_t const three = wasm_f64x2_splat(3.0);
    v128_t const four = wasm_f64x2_splat(4.0);
    v128_t const six = wasm_f64x2_splat(6.0);
    v128_t const sixteen = wasm_f64x2_splat(16.0);
    v128_t const epsilon = wasm_f64x2_splat(1e-15);

    // Longitude difference
    v128_t longitude_difference = wasm_f64x2_sub(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    v128_t one_minus_f = wasm_f64x2_sub(one, flattening);
    v128_t tan_first = wasm_f64x2_div(nk_f64x2_sin_v128relaxed_(first_latitudes),
                                      nk_f64x2_cos_v128relaxed_(first_latitudes));
    v128_t tan_second = wasm_f64x2_div(nk_f64x2_sin_v128relaxed_(second_latitudes),
                                       nk_f64x2_cos_v128relaxed_(second_latitudes));
    v128_t tan_reduced_first = wasm_f64x2_mul(one_minus_f, tan_first);
    v128_t tan_reduced_second = wasm_f64x2_mul(one_minus_f, tan_second);

    // cos(U) = 1/sqrt(1 + tan^2(U)), sin(U) = tan(U) * cos(U)
    v128_t cos_reduced_first = wasm_f64x2_div(
        one, wasm_f64x2_sqrt(wasm_f64x2_relaxed_madd(tan_reduced_first, tan_reduced_first, one)));
    v128_t sin_reduced_first = wasm_f64x2_mul(tan_reduced_first, cos_reduced_first);
    v128_t cos_reduced_second = wasm_f64x2_div(
        one, wasm_f64x2_sqrt(wasm_f64x2_relaxed_madd(tan_reduced_second, tan_reduced_second, one)));
    v128_t sin_reduced_second = wasm_f64x2_mul(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    v128_t lambda = longitude_difference;
    v128_t sin_angular_distance, cos_angular_distance, angular_distance;
    v128_t sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points using masks
    v128_t converged_mask = wasm_i64x2_splat(0);
    v128_t coincident_mask = wasm_i64x2_splat(0);

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        if (wasm_i8x16_all_true(converged_mask)) break;

        v128_t sin_lambda = nk_f64x2_sin_v128relaxed_(lambda);
        v128_t cos_lambda = nk_f64x2_cos_v128relaxed_(lambda);

        // sin^2(angular_distance) = (cos(U2) * sin(l))^2 + (cos(U1) * sin(U2) - sin(U1) * cos(U2) * cos(l))^2
        v128_t cross_term = wasm_f64x2_mul(cos_reduced_second, sin_lambda);
        v128_t mixed_term = wasm_f64x2_sub(
            wasm_f64x2_mul(cos_reduced_first, sin_reduced_second),
            wasm_f64x2_mul(wasm_f64x2_mul(sin_reduced_first, cos_reduced_second), cos_lambda));
        v128_t sin_angular_dist_sq = wasm_f64x2_relaxed_madd(cross_term, cross_term,
                                                             wasm_f64x2_mul(mixed_term, mixed_term));
        sin_angular_distance = wasm_f64x2_sqrt(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ~ 0)
        coincident_mask = wasm_f64x2_lt(sin_angular_distance, epsilon);

        // cos(angular_distance) = sin(U1) * sin(U2) + cos(U1) * cos(U2) * cos(l)
        cos_angular_distance = wasm_f64x2_relaxed_madd(wasm_f64x2_mul(cos_reduced_first, cos_reduced_second),
                                                       cos_lambda,
                                                       wasm_f64x2_mul(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_f64x2_atan2_v128relaxed_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U1) * cos(U2) * sin(l) / sin(angular_distance)
        // Avoid division by zero by using blending
        // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
        // Safe because mask is from comparison (all-ones or all-zeros per lane).
        v128_t safe_sin_angular = wasm_i64x2_relaxed_laneselect(one, sin_angular_distance, coincident_mask);
        sin_azimuth = wasm_f64x2_div(wasm_f64x2_mul(wasm_f64x2_mul(cos_reduced_first, cos_reduced_second), sin_lambda),
                                     safe_sin_angular);
        cos_squared_azimuth = wasm_f64x2_relaxed_nmadd(sin_azimuth, sin_azimuth, one);

        // Handle equatorial case: cos^2(a) ~ 0
        v128_t equatorial_mask = wasm_f64x2_lt(cos_squared_azimuth, epsilon);
        v128_t safe_cos_sq_azimuth = wasm_i64x2_relaxed_laneselect(one, cos_squared_azimuth, equatorial_mask);

        // cos(2sm) = cos(s) - 2 * sin(U1) * sin(U2) / cos^2(a)
        v128_t sin_product = wasm_f64x2_mul(sin_reduced_first, sin_reduced_second);
        cos_double_angular_midpoint = wasm_f64x2_sub(
            cos_angular_distance, wasm_f64x2_div(wasm_f64x2_mul(two, sin_product), safe_cos_sq_azimuth));
        cos_double_angular_midpoint = wasm_i64x2_relaxed_laneselect(wasm_f64x2_splat(0.0), cos_double_angular_midpoint,
                                                                    equatorial_mask);

        // C = f/16 * cos^2(a) * (4 + f*(4 - 3*cos^2(a)))
        v128_t correction_factor = wasm_f64x2_mul(
            wasm_f64x2_div(flattening, sixteen),
            wasm_f64x2_mul(
                cos_squared_azimuth,
                wasm_f64x2_relaxed_madd(flattening, wasm_f64x2_relaxed_nmadd(three, cos_squared_azimuth, four), four)));

        // l' = L + (1-C) * f * sin(a) * (s + C * sin(s) * (cos(2sm) + C * cos(s) * (-1 + 2 * cos^2(2sm))))
        v128_t cos_2sm_sq = wasm_f64x2_mul(cos_double_angular_midpoint, cos_double_angular_midpoint);
        // innermost = -1 + 2 * cos^2(2sm)
        v128_t innermost = wasm_f64x2_relaxed_madd(two, cos_2sm_sq, wasm_f64x2_splat(-1.0));
        // middle = cos(2sm) + C * cos(s) * innermost
        v128_t middle = wasm_f64x2_relaxed_madd(wasm_f64x2_mul(correction_factor, cos_angular_distance), innermost,
                                                cos_double_angular_midpoint);
        // inner = C * sin(s) * middle
        v128_t inner = wasm_f64x2_mul(wasm_f64x2_mul(correction_factor, sin_angular_distance), middle);

        // l' = L + (1-C) * f * sin_a * (s + inner)
        v128_t lambda_new = wasm_f64x2_relaxed_madd(
            wasm_f64x2_mul(wasm_f64x2_mul(wasm_f64x2_sub(one, correction_factor), flattening), sin_azimuth),
            wasm_f64x2_add(angular_distance, inner), longitude_difference);

        // Check convergence: |l - l'| < threshold
        v128_t lambda_diff = wasm_f64x2_sub(lambda_new, lambda);
        v128_t lambda_diff_abs = wasm_f64x2_abs(lambda_diff);
        v128_t newly_converged = wasm_f64x2_lt(lambda_diff_abs, convergence_threshold);
        converged_mask = wasm_v128_or(converged_mask, newly_converged);

        // Only update lambda for non-converged lanes
        // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
        // Safe because mask is from comparison (all-ones or all-zeros per lane).
        lambda = wasm_i64x2_relaxed_laneselect(lambda, lambda_new, converged_mask);
    }

    // Final distance calculation
    // u^2 = cos^2(a) * (a^2 - b^2) / b^2
    v128_t a_sq = wasm_f64x2_mul(equatorial_radius, equatorial_radius);
    v128_t b_sq = wasm_f64x2_mul(polar_radius, polar_radius);
    v128_t u_squared = wasm_f64x2_div(wasm_f64x2_mul(cos_squared_azimuth, wasm_f64x2_sub(a_sq, b_sq)), b_sq);

    // A = 1 + u^2/16384 * (4096 + u^2*(-768 + u^2*(320 - 175*u^2)))
    v128_t series_a = wasm_f64x2_relaxed_madd(u_squared, wasm_f64x2_splat(-175.0), wasm_f64x2_splat(320.0));
    series_a = wasm_f64x2_relaxed_madd(u_squared, series_a, wasm_f64x2_splat(-768.0));
    series_a = wasm_f64x2_relaxed_madd(u_squared, series_a, wasm_f64x2_splat(4096.0));
    series_a = wasm_f64x2_relaxed_madd(wasm_f64x2_div(u_squared, wasm_f64x2_splat(16384.0)), series_a, one);

    // B = u^2/1024 * (256 + u^2*(-128 + u^2*(74 - 47*u^2)))
    v128_t series_b = wasm_f64x2_relaxed_madd(u_squared, wasm_f64x2_splat(-47.0), wasm_f64x2_splat(74.0));
    series_b = wasm_f64x2_relaxed_madd(u_squared, series_b, wasm_f64x2_splat(-128.0));
    series_b = wasm_f64x2_relaxed_madd(u_squared, series_b, wasm_f64x2_splat(256.0));
    series_b = wasm_f64x2_mul(wasm_f64x2_div(u_squared, wasm_f64x2_splat(1024.0)), series_b);

    // Delta-sigma calculation
    v128_t cos_2sm_sq = wasm_f64x2_mul(cos_double_angular_midpoint, cos_double_angular_midpoint);
    v128_t sin_sq = wasm_f64x2_mul(sin_angular_distance, sin_angular_distance);
    v128_t term1 = wasm_f64x2_relaxed_madd(two, cos_2sm_sq, wasm_f64x2_splat(-1.0));
    term1 = wasm_f64x2_mul(cos_angular_distance, term1);
    v128_t term2 = wasm_f64x2_relaxed_madd(four, sin_sq, wasm_f64x2_splat(-3.0));
    v128_t term3 = wasm_f64x2_relaxed_madd(four, cos_2sm_sq, wasm_f64x2_splat(-3.0));
    term2 = wasm_f64x2_mul(wasm_f64x2_mul(wasm_f64x2_div(series_b, six), cos_double_angular_midpoint),
                           wasm_f64x2_mul(term2, term3));
    v128_t delta_sigma = wasm_f64x2_mul(
        series_b, wasm_f64x2_mul(sin_angular_distance, wasm_f64x2_add(cos_double_angular_midpoint,
                                                                      wasm_f64x2_mul(wasm_f64x2_div(series_b, four),
                                                                                     wasm_f64x2_sub(term1, term2)))));

    // s = b * A * (s - ds)
    v128_t distances = wasm_f64x2_mul(wasm_f64x2_mul(polar_radius, series_a),
                                      wasm_f64x2_sub(angular_distance, delta_sigma));

    // Set coincident points to zero
    // relaxed_laneselect: 1 instruction (vblendvpd) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    distances = wasm_i64x2_relaxed_laneselect(wasm_f64x2_splat(0.0), distances, coincident_mask);

    return distances;
}

NK_PUBLIC void nk_vincenty_f64_v128relaxed(         //
    nk_f64_t const *a_lats, nk_f64_t const *a_lons, //
    nk_f64_t const *b_lats, nk_f64_t const *b_lons, //
    nk_size_t n, nk_f64_t *results) {

    while (n >= 2) {
        v128_t first_latitudes = wasm_v128_load(a_lats);
        v128_t first_longitudes = wasm_v128_load(a_lons);
        v128_t second_latitudes = wasm_v128_load(b_lats);
        v128_t second_longitudes = wasm_v128_load(b_lons);

        v128_t distances = nk_vincenty_f64x2_v128relaxed_(first_latitudes, first_longitudes, second_latitudes,
                                                          second_longitudes);
        wasm_v128_store(results, distances);

        a_lats += 2, a_lons += 2, b_lats += 2, b_lons += 2, results += 2, n -= 2;
    }

    // Handle remaining elements with partial loads (n can only be 0 or 1 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b64x2_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b64x2_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b64x2_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b64x2_serial_(b_lons, &b_lon_vec, n);
        v128_t distances = nk_vincenty_f64x2_v128relaxed_(a_lat_vec.v128, a_lon_vec.v128, b_lat_vec.v128,
                                                          b_lon_vec.v128);
        result_vec.v128 = distances;
        nk_partial_store_b64x2_serial_(&result_vec, results, n);
    }
}

/**
 *  @brief  WASM Relaxed SIMD helper for Vincenty's geodesic distance on 4 f32 point pairs.
 *  @note   This is a true SIMD implementation using masked convergence tracking via blending.
 */
NK_INTERNAL v128_t nk_vincenty_f32x4_v128relaxed_(   //
    v128_t first_latitudes, v128_t first_longitudes, //
    v128_t second_latitudes, v128_t second_longitudes) {

    v128_t const equatorial_radius = wasm_f32x4_splat((float)NK_EARTH_ELLIPSOID_EQUATORIAL_RADIUS);
    v128_t const polar_radius = wasm_f32x4_splat((float)NK_EARTH_ELLIPSOID_POLAR_RADIUS);
    v128_t const flattening = wasm_f32x4_splat(1.0f / (float)NK_EARTH_ELLIPSOID_INVERSE_FLATTENING);
    v128_t const convergence_threshold = wasm_f32x4_splat(NK_VINCENTY_CONVERGENCE_THRESHOLD_F32);
    v128_t const one = wasm_f32x4_splat(1.0f);
    v128_t const two = wasm_f32x4_splat(2.0f);
    v128_t const three = wasm_f32x4_splat(3.0f);
    v128_t const four = wasm_f32x4_splat(4.0f);
    v128_t const six = wasm_f32x4_splat(6.0f);
    v128_t const sixteen = wasm_f32x4_splat(16.0f);
    v128_t const epsilon = wasm_f32x4_splat(1e-7f);

    // Longitude difference
    v128_t longitude_difference = wasm_f32x4_sub(second_longitudes, first_longitudes);

    // Reduced latitudes: tan(U) = (1-f) * tan(lat)
    v128_t one_minus_f = wasm_f32x4_sub(one, flattening);
    v128_t tan_first = wasm_f32x4_div(nk_f32x4_sin_v128relaxed_(first_latitudes),
                                      nk_f32x4_cos_v128relaxed_(first_latitudes));
    v128_t tan_second = wasm_f32x4_div(nk_f32x4_sin_v128relaxed_(second_latitudes),
                                       nk_f32x4_cos_v128relaxed_(second_latitudes));
    v128_t tan_reduced_first = wasm_f32x4_mul(one_minus_f, tan_first);
    v128_t tan_reduced_second = wasm_f32x4_mul(one_minus_f, tan_second);

    // cos(U) = 1/sqrt(1 + tan^2(U)), sin(U) = tan(U) * cos(U)
    v128_t cos_reduced_first = wasm_f32x4_div(
        one, wasm_f32x4_sqrt(wasm_f32x4_relaxed_madd(tan_reduced_first, tan_reduced_first, one)));
    v128_t sin_reduced_first = wasm_f32x4_mul(tan_reduced_first, cos_reduced_first);
    v128_t cos_reduced_second = wasm_f32x4_div(
        one, wasm_f32x4_sqrt(wasm_f32x4_relaxed_madd(tan_reduced_second, tan_reduced_second, one)));
    v128_t sin_reduced_second = wasm_f32x4_mul(tan_reduced_second, cos_reduced_second);

    // Initialize lambda and tracking variables
    v128_t lambda = longitude_difference;
    v128_t sin_angular_distance, cos_angular_distance, angular_distance;
    v128_t sin_azimuth, cos_squared_azimuth, cos_double_angular_midpoint;

    // Track convergence and coincident points using masks
    v128_t converged_mask = wasm_i32x4_splat(0);
    v128_t coincident_mask = wasm_i32x4_splat(0);

    for (nk_u32_t iteration = 0; iteration < NK_VINCENTY_MAX_ITERATIONS; ++iteration) {
        // Check if all lanes converged
        if (wasm_i8x16_all_true(converged_mask)) break;

        v128_t sin_lambda = nk_f32x4_sin_v128relaxed_(lambda);
        v128_t cos_lambda = nk_f32x4_cos_v128relaxed_(lambda);

        // sin^2(angular_distance) = (cos(U2) * sin(l))^2 + (cos(U1) * sin(U2) - sin(U1) * cos(U2) * cos(l))^2
        v128_t cross_term = wasm_f32x4_mul(cos_reduced_second, sin_lambda);
        v128_t mixed_term = wasm_f32x4_sub(
            wasm_f32x4_mul(cos_reduced_first, sin_reduced_second),
            wasm_f32x4_mul(wasm_f32x4_mul(sin_reduced_first, cos_reduced_second), cos_lambda));
        v128_t sin_angular_dist_sq = wasm_f32x4_relaxed_madd(cross_term, cross_term,
                                                             wasm_f32x4_mul(mixed_term, mixed_term));
        sin_angular_distance = wasm_f32x4_sqrt(sin_angular_dist_sq);

        // Check for coincident points (sin_angular_distance ~ 0)
        coincident_mask = wasm_f32x4_lt(sin_angular_distance, epsilon);

        // cos(angular_distance) = sin(U1) * sin(U2) + cos(U1) * cos(U2) * cos(l)
        cos_angular_distance = wasm_f32x4_relaxed_madd(wasm_f32x4_mul(cos_reduced_first, cos_reduced_second),
                                                       cos_lambda,
                                                       wasm_f32x4_mul(sin_reduced_first, sin_reduced_second));

        // angular_distance = atan2(sin, cos)
        angular_distance = nk_f32x4_atan2_v128relaxed_(sin_angular_distance, cos_angular_distance);

        // sin(azimuth) = cos(U1) * cos(U2) * sin(l) / sin(angular_distance)
        // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
        // Safe because mask is from comparison (all-ones or all-zeros per lane).
        v128_t safe_sin_angular = wasm_i32x4_relaxed_laneselect(one, sin_angular_distance, coincident_mask);
        sin_azimuth = wasm_f32x4_div(wasm_f32x4_mul(wasm_f32x4_mul(cos_reduced_first, cos_reduced_second), sin_lambda),
                                     safe_sin_angular);
        cos_squared_azimuth = wasm_f32x4_relaxed_nmadd(sin_azimuth, sin_azimuth, one);

        // Handle equatorial case: cos^2(a) ~ 0
        v128_t equatorial_mask = wasm_f32x4_lt(cos_squared_azimuth, epsilon);
        v128_t safe_cos_sq_azimuth = wasm_i32x4_relaxed_laneselect(one, cos_squared_azimuth, equatorial_mask);

        // cos(2sm) = cos(s) - 2 * sin(U1) * sin(U2) / cos^2(a)
        v128_t sin_product = wasm_f32x4_mul(sin_reduced_first, sin_reduced_second);
        cos_double_angular_midpoint = wasm_f32x4_sub(
            cos_angular_distance, wasm_f32x4_div(wasm_f32x4_mul(two, sin_product), safe_cos_sq_azimuth));
        cos_double_angular_midpoint = wasm_i32x4_relaxed_laneselect(wasm_f32x4_splat(0.0f), cos_double_angular_midpoint,
                                                                    equatorial_mask);

        // C = f/16 * cos^2(a) * (4 + f*(4 - 3*cos^2(a)))
        v128_t correction_factor = wasm_f32x4_mul(
            wasm_f32x4_div(flattening, sixteen),
            wasm_f32x4_mul(
                cos_squared_azimuth,
                wasm_f32x4_relaxed_madd(flattening, wasm_f32x4_relaxed_nmadd(three, cos_squared_azimuth, four), four)));

        // l' = L + (1-C) * f * sin(a) * (s + C * sin(s) * (cos(2sm) + C * cos(s) * (-1 + 2 * cos^2(2sm))))
        v128_t cos_2sm_sq = wasm_f32x4_mul(cos_double_angular_midpoint, cos_double_angular_midpoint);
        v128_t innermost = wasm_f32x4_relaxed_madd(two, cos_2sm_sq, wasm_f32x4_splat(-1.0f));
        v128_t middle = wasm_f32x4_relaxed_madd(wasm_f32x4_mul(correction_factor, cos_angular_distance), innermost,
                                                cos_double_angular_midpoint);
        v128_t inner = wasm_f32x4_mul(wasm_f32x4_mul(correction_factor, sin_angular_distance), middle);

        v128_t lambda_new = wasm_f32x4_relaxed_madd(
            wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_sub(one, correction_factor), flattening), sin_azimuth),
            wasm_f32x4_add(angular_distance, inner), longitude_difference);

        // Check convergence: |l - l'| < threshold
        v128_t lambda_diff = wasm_f32x4_sub(lambda_new, lambda);
        v128_t lambda_diff_abs = wasm_f32x4_abs(lambda_diff);
        v128_t newly_converged = wasm_f32x4_lt(lambda_diff_abs, convergence_threshold);
        converged_mask = wasm_v128_or(converged_mask, newly_converged);

        // Only update lambda for non-converged lanes
        // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
        // Safe because mask is from comparison (all-ones or all-zeros per lane).
        lambda = wasm_i32x4_relaxed_laneselect(lambda, lambda_new, converged_mask);
    }

    // Final distance calculation
    v128_t a_sq = wasm_f32x4_mul(equatorial_radius, equatorial_radius);
    v128_t b_sq = wasm_f32x4_mul(polar_radius, polar_radius);
    v128_t u_squared = wasm_f32x4_div(wasm_f32x4_mul(cos_squared_azimuth, wasm_f32x4_sub(a_sq, b_sq)), b_sq);

    // A = 1 + u^2/16384 * (4096 + u^2*(-768 + u^2*(320 - 175*u^2)))
    v128_t series_a = wasm_f32x4_relaxed_madd(u_squared, wasm_f32x4_splat(-175.0f), wasm_f32x4_splat(320.0f));
    series_a = wasm_f32x4_relaxed_madd(u_squared, series_a, wasm_f32x4_splat(-768.0f));
    series_a = wasm_f32x4_relaxed_madd(u_squared, series_a, wasm_f32x4_splat(4096.0f));
    series_a = wasm_f32x4_relaxed_madd(wasm_f32x4_div(u_squared, wasm_f32x4_splat(16384.0f)), series_a, one);

    // B = u^2/1024 * (256 + u^2*(-128 + u^2*(74 - 47*u^2)))
    v128_t series_b = wasm_f32x4_relaxed_madd(u_squared, wasm_f32x4_splat(-47.0f), wasm_f32x4_splat(74.0f));
    series_b = wasm_f32x4_relaxed_madd(u_squared, series_b, wasm_f32x4_splat(-128.0f));
    series_b = wasm_f32x4_relaxed_madd(u_squared, series_b, wasm_f32x4_splat(256.0f));
    series_b = wasm_f32x4_mul(wasm_f32x4_div(u_squared, wasm_f32x4_splat(1024.0f)), series_b);

    // Delta-sigma calculation
    v128_t cos_2sm_sq = wasm_f32x4_mul(cos_double_angular_midpoint, cos_double_angular_midpoint);
    v128_t sin_sq = wasm_f32x4_mul(sin_angular_distance, sin_angular_distance);
    v128_t term1 = wasm_f32x4_relaxed_madd(two, cos_2sm_sq, wasm_f32x4_splat(-1.0f));
    term1 = wasm_f32x4_mul(cos_angular_distance, term1);
    v128_t term2 = wasm_f32x4_relaxed_madd(four, sin_sq, wasm_f32x4_splat(-3.0f));
    v128_t term3 = wasm_f32x4_relaxed_madd(four, cos_2sm_sq, wasm_f32x4_splat(-3.0f));
    term2 = wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_div(series_b, six), cos_double_angular_midpoint),
                           wasm_f32x4_mul(term2, term3));
    v128_t delta_sigma = wasm_f32x4_mul(
        series_b, wasm_f32x4_mul(sin_angular_distance, wasm_f32x4_add(cos_double_angular_midpoint,
                                                                      wasm_f32x4_mul(wasm_f32x4_div(series_b, four),
                                                                                     wasm_f32x4_sub(term1, term2)))));

    // s = b * A * (s - ds)
    v128_t distances = wasm_f32x4_mul(wasm_f32x4_mul(polar_radius, series_a),
                                      wasm_f32x4_sub(angular_distance, delta_sigma));

    // Set coincident points to zero
    // relaxed_laneselect: 1 instruction (vblendvps) vs 3 (vpand+vpandn+vpor) on x86.
    // Safe because mask is from comparison (all-ones or all-zeros per lane).
    distances = wasm_i32x4_relaxed_laneselect(wasm_f32x4_splat(0.0f), distances, coincident_mask);

    return distances;
}

NK_PUBLIC void nk_vincenty_f32_v128relaxed(         //
    nk_f32_t const *a_lats, nk_f32_t const *a_lons, //
    nk_f32_t const *b_lats, nk_f32_t const *b_lons, //
    nk_size_t n, nk_f32_t *results) {

    while (n >= 4) {
        v128_t first_latitudes = wasm_v128_load(a_lats);
        v128_t first_longitudes = wasm_v128_load(a_lons);
        v128_t second_latitudes = wasm_v128_load(b_lats);
        v128_t second_longitudes = wasm_v128_load(b_lons);

        v128_t distances = nk_vincenty_f32x4_v128relaxed_(first_latitudes, first_longitudes, second_latitudes,
                                                          second_longitudes);
        wasm_v128_store(results, distances);

        a_lats += 4, a_lons += 4, b_lats += 4, b_lons += 4, results += 4, n -= 4;
    }

    // Handle remaining elements with partial loads (n can be 1-3 here)
    if (n > 0) {
        nk_b128_vec_t a_lat_vec, a_lon_vec, b_lat_vec, b_lon_vec, result_vec;
        nk_partial_load_b32x4_serial_(a_lats, &a_lat_vec, n);
        nk_partial_load_b32x4_serial_(a_lons, &a_lon_vec, n);
        nk_partial_load_b32x4_serial_(b_lats, &b_lat_vec, n);
        nk_partial_load_b32x4_serial_(b_lons, &b_lon_vec, n);
        v128_t distances = nk_vincenty_f32x4_v128relaxed_(a_lat_vec.v128, a_lon_vec.v128, b_lat_vec.v128,
                                                          b_lon_vec.v128);
        result_vec.v128 = distances;
        nk_partial_store_b32x4_serial_(&result_vec, results, n);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // NK_TARGET_V128RELAXED
#endif // NK_GEOSPATIAL_V128RELAXED_H
